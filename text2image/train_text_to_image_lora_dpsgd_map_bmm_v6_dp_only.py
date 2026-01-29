# coding=utf-8
# SD LoRA + DP-SGD (Opacus) -- DP ONLY, NO caption corruption
# Map-style PT shards dataset + DP micro-batching (BatchMemoryManager)
#
# Guarantees:
# 1) Use unified logical sample_rate = logical_batch_size / N
# 2) Manual accountant stepping each logical optimizer step (robust across wrappers/BMM)
# 3) High-precision epsilon logging + monotonicity check
#
# Notes:
# - Image quality not the goal, but we DO NOT alter captions.
# - Keep LoRA params FP32 for mixed precision stability.
# - Avoid accelerate.save_state hook crash: custom checkpoint.

import argparse
import logging
import os
import re
import shutil
import inspect
from pathlib import Path
from glob import glob
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__, log_level="INFO")


# -----------------------------
# Utilities
# -----------------------------
def _default_rdp_alphas():
    # Opacus FAQ recommended set:
    # [1.1..10.9 step 0.1] + [12..63]
    a = [1.0 + x / 10.0 for x in range(1, 100)]
    a += list(range(12, 64))
    return a


def _parse_alphas(s: str):
    if s is None:
        return None
    s = s.strip().lower()
    if s == "default":
        return _default_rdp_alphas()
    if s == "":
        return None
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in parts]


def unwrap_to_dpoptimizer(maybe_wrapped_opt):
    # accelerate.prepare may wrap optimizer. Unwrap until raw optimizer-like object.
    opt = maybe_wrapped_opt
    seen = set()
    for _ in range(30):
        if id(opt) in seen:
            break
        seen.add(id(opt))
        if hasattr(opt, "optimizer"):
            opt = getattr(opt, "optimizer")
            continue
        break
    return opt


def patch_opacus_add_noise_dtype_safe(dp_opt):
    """
    Ensure Opacus add_noise uses parameter dtype (mixed precision safety).
    """
    import types

    def _add_noise_dtype_safe(self):
        std = float(self.noise_multiplier) * float(self.max_grad_norm)
        for p in self.params:
            sg = getattr(p, "summed_grad", None)
            if sg is None:
                continue
            sg = sg.to(dtype=p.dtype)
            noise = torch.normal(
                mean=0.0,
                std=std,
                size=sg.shape,
                device=sg.device,
                dtype=p.dtype,
            )
            p.grad = (sg + noise).view_as(p)

    if hasattr(dp_opt, "add_noise"):
        dp_opt.add_noise = types.MethodType(_add_noise_dtype_safe, dp_opt)


def force_lora_fp32(unet, lora_layers, device):
    """
    Keep LoRA trainable params FP32 to avoid GradScaler unscale FP16 grads errors.
    """
    unet.to(device)
    lora_layers.to(device=device, dtype=torch.float32)
    for p in lora_layers.parameters():
        if p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)


def make_private_compat(
    privacy_engine, module, optimizer, data_loader,
    noise_multiplier, max_grad_norm, poisson_sampling: bool
):
    sig = inspect.signature(privacy_engine.make_private)
    kwargs = dict(
        module=module,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=float(noise_multiplier),
        max_grad_norm=float(max_grad_norm),
    )
    if "poisson_sampling" in sig.parameters:
        kwargs["poisson_sampling"] = bool(poisson_sampling)
    return privacy_engine.make_private(**kwargs)


def _get_epsilon_robust(privacy_engine, delta: float) -> float:
    out = privacy_engine.get_epsilon(delta)
    if isinstance(out, (tuple, list)):
        return float(out[0])
    return float(out)


def _manual_accountant_step(privacy_engine, noise_multiplier: float, sample_rate: float):
    """
    Force accountant to advance by one logical step (best effort across Opacus versions).
    """
    acc = getattr(privacy_engine, "accountant", None)
    if acc is None:
        return False
    try:
        acc.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
        return True
    except TypeError:
        pass
    try:
        acc.step(float(noise_multiplier), float(sample_rate))
        return True
    except Exception:
        return False


def _accountant_debug_snapshot(privacy_engine) -> str:
    acc = getattr(privacy_engine, "accountant", None)
    if acc is None:
        return "accountant=None"
    # best-effort summary without assuming internal fields
    keys = ["history", "_history", "steps", "_steps", "ledger", "_ledger"]
    for k in keys:
        if hasattr(acc, k):
            v = getattr(acc, k)
            try:
                ln = len(v)
            except Exception:
                ln = None
            s = str(v).replace("\n", " ")
            return f"accountant.{k}: type={type(v)} len={ln} head={s[:120]}"
    return f"accountant=type({type(acc)})"


# -----------------------------
# Custom checkpoint helpers
# -----------------------------
def _list_checkpoints(output_dir: str):
    if not os.path.isdir(output_dir):
        return []
    dirs = []
    for d in os.listdir(output_dir):
        if re.match(r"^checkpoint-\d+$", d):
            dirs.append(d)
    return sorted(dirs, key=lambda x: int(x.split("-")[1]))


def _get_latest_checkpoint_dir(output_dir: str):
    dirs = _list_checkpoints(output_dir)
    if not dirs:
        return None
    return os.path.join(output_dir, dirs[-1])


def save_checkpoint_custom(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    optimizer,
    lr_scheduler,
    privacy_engine,
    output_dir: str,
    global_step: int,
    checkpoints_total_limit: Optional[int] = None,
):
    if not accelerator.is_main_process:
        return

    if checkpoints_total_limit is not None:
        dirs = _list_checkpoints(output_dir)
        if len(dirs) >= checkpoints_total_limit:
            num_to_remove = len(dirs) - checkpoints_total_limit + 1
            for rm in dirs[:num_to_remove]:
                shutil.rmtree(os.path.join(output_dir, rm), ignore_errors=True)

    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)

    # Save LoRA
    accelerator.unwrap_model(unet).save_attn_procs(save_path)

    # Save trainer state
    torch.save({"global_step": int(global_step)}, os.path.join(save_path, "trainer_state.pt"))

    # Save optimizer/scheduler
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "lr_scheduler.pt"))

    # Save privacy engine state if supported
    if privacy_engine is not None and hasattr(privacy_engine, "state_dict"):
        try:
            torch.save(privacy_engine.state_dict(), os.path.join(save_path, "privacy_engine.pt"))
        except Exception as e:
            logger.warning(f"[CKPT] privacy_engine.state_dict save failed: {repr(e)}")

    logger.info(f"[CKPT] Saved custom checkpoint to {save_path}")


def load_checkpoint_custom(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    optimizer,
    lr_scheduler,
    privacy_engine,
    ckpt_dir: str,
):
    if ckpt_dir is None or (not os.path.isdir(ckpt_dir)):
        return 0

    # LoRA
    try:
        accelerator.unwrap_model(unet).load_attn_procs(ckpt_dir)
    except Exception as e:
        logger.warning(f"[CKPT] load_attn_procs failed: {repr(e)}")

    # trainer_state
    step = 0
    st_path = os.path.join(ckpt_dir, "trainer_state.pt")
    if os.path.isfile(st_path):
        st = torch.load(st_path, map_location="cpu")
        step = int(st.get("global_step", 0))

    # optimizer/scheduler
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    sch_path = os.path.join(ckpt_dir, "lr_scheduler.pt")
    if os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        except Exception as e:
            logger.warning(f"[CKPT] optimizer.load_state_dict failed: {repr(e)}")
    if os.path.isfile(sch_path):
        try:
            lr_scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
        except Exception as e:
            logger.warning(f"[CKPT] lr_scheduler.load_state_dict failed: {repr(e)}")

    # privacy_engine
    pe_path = os.path.join(ckpt_dir, "privacy_engine.pt")
    if privacy_engine is not None and os.path.isfile(pe_path) and hasattr(privacy_engine, "load_state_dict"):
        try:
            privacy_engine.load_state_dict(torch.load(pe_path, map_location="cpu"))
        except Exception as e:
            logger.warning(f"[CKPT] privacy_engine.load_state_dict failed: {repr(e)}")

    logger.info(f"[CKPT] Resumed from {ckpt_dir}, global_step={step}")
    return step


# -----------------------------
# Map-style PT shards dataset (NO caption changes)
# -----------------------------
class ShardedPTMapDataset(TorchDataset):
    """
    dir_path/
      batch_compressed_000.pt
      ...
    each .pt: {"image": list[PIL|bytes|ndarray|tensor], "text": list[str], ...}
    """
    def __init__(self, dir_path: str, transform, tokenizer, max_length: int, seed: int = 1337):
        super().__init__()
        self.dir_path = dir_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = np.random.default_rng(seed)

        patterns = [
            os.path.join(dir_path, "batch_compressed_*.pt"),
            os.path.join(dir_path, "batch_*.pt"),
            os.path.join(dir_path, "*.pt"),
        ]
        shard_files = []
        for p in patterns:
            shard_files.extend(glob(p))
        shard_files = sorted(list(set(shard_files)))
        if len(shard_files) == 0:
            raise AssertionError(f"no pt shards under {dir_path}")
        self.shard_files = shard_files

        self.index = []
        for sp in self.shard_files:
            data = torch.load(sp, map_location="cpu")
            imgs = data.get("image") or data.get("images")
            txts = data.get("text") or data.get("texts")
            if imgs is None or txts is None:
                continue
            n = min(len(imgs), len(txts))
            if n <= 0:
                continue
            for i in range(n):
                self.index.append((sp, i))
            del data

        if len(self.index) == 0:
            raise RuntimeError(f"Found shards but no valid samples under {dir_path}")

        logger.info(f"[DATA] Map-style shards dataset size N={len(self.index)}")

        self._cache_path = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def _to_pil(self, img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, (bytes, bytearray)):
            from io import BytesIO
            return Image.open(BytesIO(img)).convert("RGB")
        if torch.is_tensor(img):
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        if isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        return Image.fromarray(np.array(img)).convert("RGB")

    def _load_shard(self, sp: str):
        if self._cache_path == sp and self._cache_data is not None:
            return self._cache_data
        data = torch.load(sp, map_location="cpu")
        self._cache_path = sp
        self._cache_data = data
        return data

    def __getitem__(self, idx: int):
        sp, j = self.index[idx]
        data = self._load_shard(sp)
        imgs = data.get("image") or data.get("images")
        txts = data.get("text") or data.get("texts")

        img = self._to_pil(imgs[j])
        txt = txts[j]
        txt = txt if isinstance(txt, str) else str(txt)

        pixel = self.transform(img)  # float32
        token_ids = self.tokenizer(
            [txt],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]  # int64

        # tuple-batch: (pixel_values, input_ids)
        return pixel, token_ids


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)

    p.add_argument("--train_data_dir", type=str, required=True)
    p.add_argument("--dataloader_num_workers", type=int, default=2)

    p.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    p.add_argument("--allow_tf32", action="store_true")

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--random_flip", action="store_true")

    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=3000)

    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=0)

    p.add_argument("--rank", type=int, default=1)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--project_name", type=str, default=None)

    p.add_argument("--checkpointing_steps", type=int, default=1000)
    p.add_argument("--checkpoints_total_limit", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    # validation (optional)
    p.add_argument("--validation_prompt", type=str, default=None)
    p.add_argument("--validation_steps", type=int, default=7500)
    p.add_argument("--num_validation_images", type=int, default=4)
    p.add_argument("--disable_validation", action="store_true")

    p.add_argument("--seed", type=int, default=1337)

    # DP
    p.add_argument("--enable_dp", action="store_true")
    p.add_argument("--dp_noise_multiplier", type=float, default=12.0)
    p.add_argument("--dp_max_grad_norm", type=float, default=0.05)
    p.add_argument("--dp_delta", type=float, default=0.0)
    p.add_argument("--dp_max_physical_batch_size", type=int, default=1)
    p.add_argument("--dp_accountant", type=str, default="rdp", choices=["rdp", "prv"])
    p.add_argument("--dp_poisson_sampling", action="store_true")
    p.add_argument("--dp_secure_mode", action="store_true")

    # Logging
    p.add_argument("--dp_log_every", type=int, default=100)
    p.add_argument("--dp_rdp_alphas", type=str, default="default")
    p.add_argument("--dp_force_manual_accountant_step", action="store_true",
                   help="Force manual accountant.step each logical step (recommended).")

    return p.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb to log to W&B.")
        import wandb
        wandb.init(project=args.project_name, settings=wandb.Settings(start_method="thread"))

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------
    # Load SD components
    # -----------------------------
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)

    # -----------------------------
    # Install LoRA
    # -----------------------------
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )
    unet.set_attn_processor(lora_attn_procs)
    unet.to(accelerator.device, dtype=weight_dtype)

    lora_layers = AttnProcsLayers(unet.attn_processors)
    for p in lora_layers.parameters():
        p.requires_grad_(True)
    force_lora_fp32(unet, lora_layers, accelerator.device)

    trainable = sum(p.numel() for p in lora_layers.parameters() if p.requires_grad)
    logger.info(f"[LoRA] trainable params = {trainable} (rank={args.rank})")

    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=args.learning_rate)

    # -----------------------------
    # Dataset / Dataloader
    # -----------------------------
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_dataset = ShardedPTMapDataset(
        dir_path=args.train_data_dir,
        transform=train_transforms,
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
        seed=args.seed,
    )
    N = len(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=(args.dataloader_num_workers > 0),
        drop_last=True,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # -----------------------------
    # DP make_private BEFORE prepare
    # -----------------------------
    if args.enable_dp:
        delta = args.dp_delta if args.dp_delta > 0 else (1.0 / float(N))
        privacy_engine = PrivacyEngine(accountant=args.dp_accountant, secure_mode=bool(args.dp_secure_mode))

        alphas = _parse_alphas(args.dp_rdp_alphas)
        try:
            if alphas is not None and hasattr(privacy_engine, "accountant") and hasattr(privacy_engine.accountant, "alphas"):
                privacy_engine.accountant.alphas = alphas
        except Exception as e:
            logger.warning(f"[DP] set accountant.alphas failed: {repr(e)}")

        lora_layers, optimizer, train_dataloader = make_private_compat(
            privacy_engine=privacy_engine,
            module=lora_layers,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=args.dp_noise_multiplier,
            max_grad_norm=args.dp_max_grad_norm,
            poisson_sampling=bool(args.dp_poisson_sampling),
        )

        # unified logical sample rate definition
        logical_sample_rate = float(args.train_batch_size) / float(N)

        logger.info(
            f"[DP] enabled: N={N}, noise_multiplier={args.dp_noise_multiplier}, C={args.dp_max_grad_norm}, "
            f"delta={delta}, accountant={args.dp_accountant}, poisson_sampling={bool(args.dp_poisson_sampling)}, "
            f"secure_rng={bool(args.dp_secure_mode)}"
        )
        logger.info(f"[DP] logical_sample_rate = train_batch_size/N = {logical_sample_rate:.10e}")

        last_eps = None

        def maybe_log_epsilon(step: int, force: bool = False):
            nonlocal last_eps
            if (not force) and (args.dp_log_every <= 0 or step % args.dp_log_every != 0):
                return
            try:
                eps = _get_epsilon_robust(privacy_engine, delta=delta)

                # monotonicity check (warn only)
                if last_eps is not None and eps + 1e-12 < last_eps:
                    logger.warning(
                        f"[DP][WARN] epsilon decreased: prev={last_eps:.10f} now={eps:.10f}. "
                        f"This indicates accountant/version mismatch. {_accountant_debug_snapshot(privacy_engine)}"
                    )
                last_eps = eps

                accelerator.log(
                    {
                        "dp_epsilon": float(eps),
                        "dp_delta": float(delta),
                        "dp_noise_multiplier": float(args.dp_noise_multiplier),
                        "dp_C": float(args.dp_max_grad_norm),
                        "dp_logical_sample_rate": float(logical_sample_rate),
                    },
                    step=step,
                )
                if accelerator.is_main_process:
                    logger.info(
                        f"[DP] step={step} eps={eps:.10f} (delta={delta}) "
                        f"logical_sample_rate={logical_sample_rate:.10e}"
                    )
            except Exception as e:
                if accelerator.is_main_process:
                    logger.warning(f"[DP] epsilon compute/log failed: {repr(e)}")

    else:
        privacy_engine = None
        logical_sample_rate = None
        maybe_log_epsilon = lambda step, force=False: None

    # -----------------------------
    # Accelerator prepare
    # -----------------------------
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    force_lora_fp32(unet, lora_layers, accelerator.device)

    if args.enable_dp:
        dp_opt = unwrap_to_dpoptimizer(optimizer)
        patch_opacus_add_noise_dtype_safe(dp_opt)
        if accelerator.is_main_process:
            logger.info(f"[DP] patched add_noise dtype-safe on optimizer={type(dp_opt)}")
            logger.info(f"[DP] accountant snapshot: {_accountant_debug_snapshot(privacy_engine)}")

    # -----------------------------
    # Resume (custom)
    # -----------------------------
    global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            ckpt_dir = _get_latest_checkpoint_dir(args.output_dir)
        else:
            ckpt_dir = args.resume_from_checkpoint
            if not os.path.isabs(ckpt_dir):
                ckpt_dir = os.path.join(args.output_dir, ckpt_dir)
        if ckpt_dir is not None:
            global_step = load_checkpoint_custom(
                accelerator=accelerator,
                unet=unet,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                privacy_engine=privacy_engine,
                ckpt_dir=ckpt_dir,
            )
            # resume后强制打一条 epsilon
            if args.enable_dp:
                maybe_log_epsilon(global_step, force=True)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    unet.train()
    train_loss_accum = 0.0

    # -----------------------------
    # Train step
    # -----------------------------
    def train_one_batch(batch):
        nonlocal train_loss_accum, global_step

        pixel_values, input_ids = batch
        if pixel_values is None or pixel_values.ndim == 0 or pixel_values.shape[0] == 0:
            return
        if input_ids is None or input_ids.ndim == 0 or input_ids.shape[0] == 0:
            return

        with accelerator.accumulate(unet):
            pixel_values = pixel_values.to(accelerator.device, dtype=weight_dtype)
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

            bsz = latents.shape[0]
            if bsz == 0:
                return

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            input_ids = input_ids.to(accelerator.device)
            encoder_hidden_states = text_encoder(input_ids)[0]

            target = noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)

            train_loss_accum += float(loss.detach().float().item())
            accelerator.log({"train_loss": float(train_loss_accum)}, step=global_step)
            train_loss_accum = 0.0

            # ---- DP: manual accountant stepping + epsilon logging ----
            if args.enable_dp and args.dp_force_manual_accountant_step:
                ok = _manual_accountant_step(
                    privacy_engine=privacy_engine,
                    noise_multiplier=float(args.dp_noise_multiplier),
                    sample_rate=float(logical_sample_rate),
                )
                if (not ok) and accelerator.is_main_process and (global_step % args.dp_log_every == 0):
                    logger.warning(f"[DP] manual accountant.step failed. {_accountant_debug_snapshot(privacy_engine)}")

            maybe_log_epsilon(global_step, force=False)

            if args.checkpointing_steps and (global_step % args.checkpointing_steps == 0):
                accelerator.wait_for_everyone()
                save_checkpoint_custom(
                    accelerator=accelerator,
                    unet=unet,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    privacy_engine=privacy_engine,
                    output_dir=args.output_dir,
                    global_step=global_step,
                    checkpoints_total_limit=args.checkpoints_total_limit,
                )

            if (not args.disable_validation) and args.validation_prompt and args.validation_steps > 0:
                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    logger.info(f"[VAL] prompt={args.validation_prompt}")
                    pipe = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    ).to(accelerator.device)
                    pipe.set_progress_bar_config(disable=True)
                    gen = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                    images = []
                    for _ in range(args.num_validation_images):
                        images.append(pipe(args.validation_prompt, num_inference_steps=20, generator=gen).images[0])

                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            import wandb
                            tracker.log({"validation": [wandb.Image(im) for im in images]})

                    del pipe
                    torch.cuda.empty_cache()

    # -----------------------------
    # Training loop (DP micro-batching via BatchMemoryManager)
    # -----------------------------
    logger.info("***** Running training *****")
    logger.info(f"  Max train steps = {args.max_train_steps}")
    logger.info(f"  Train batch size per device = {args.train_batch_size}")
    logger.info(f"  Grad Accumulation steps = {args.gradient_accumulation_steps}")
    if args.enable_dp:
        logger.info(f"  DP max physical batch size = {args.dp_max_physical_batch_size}")
        # initial epsilon log
        maybe_log_epsilon(global_step, force=True)

    while global_step < args.max_train_steps:
        if args.enable_dp and args.dp_max_physical_batch_size < args.train_batch_size:
            dp_opt_for_bmm = unwrap_to_dpoptimizer(optimizer)
            with BatchMemoryManager(
                data_loader=train_dataloader,
                max_physical_batch_size=int(args.dp_max_physical_batch_size),
                optimizer=dp_opt_for_bmm,
            ) as memory_safe_data_loader:
                for batch in memory_safe_data_loader:
                    if global_step >= args.max_train_steps:
                        break
                    train_one_batch(batch)
        else:
            for batch in train_dataloader:
                if global_step >= args.max_train_steps:
                    break
                train_one_batch(batch)

    # -----------------------------
    # Save final LoRA
    # -----------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(unet).to(torch.float32).save_attn_procs(args.output_dir)
        logger.info(f"LoRA weights saved to {args.output_dir}")
        if args.enable_dp:
            # final epsilon
            maybe_log_epsilon(global_step, force=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
