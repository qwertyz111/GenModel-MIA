General workflow:

Step 1: Sample query images + build official captions + prepare ImageNet100 non-member set

Step 2: Synthesize shadow-member images from captions

Step 3: Build member/non-member centroids (supports multiple distances and encoders)

Step 4: Membership inference attack (MIA) evaluation


Datasets and member sets (template)

# COCO2017
export COCO_IMG=/path/to/COCO2017/train2017
export COCO_CAP_JSON=/path/to/COCO2017/annotations/captions_train2017.json
export COCO_MEMBER_SHARDS=/path/to/coco_member_shards

# CelebA-Dialog
export CELEBA_IMG=/path/to/CelebA-Dialog/image
export CELEBA_TEXT=/path/to/CelebA-Dialog/text
export MEMBER_SHARDS=/root/autodl-tmp/celeba_dialog_shards/train
export CELEBA_MEMBER_SHARDS=$MEMBER_SHARDS

# ImageNet100
export IN100_ROOT=/path/to/ImageNet100/imagenet100


Baseline reference

This work reuses the baseline fine-tuning method from the following open-source repository, and adopts it to run controlled comparisons against our experiments:
https://github.com/py85252876/Reconstruction-based-Attack


Non-DP experimental pipeline (two tracks: without noise and with noise)

The commands below use N=2000 as the template. For N=1000 or N=3000, only replace --num, --n_each, --q_each, and the numeric identifiers in output directory names accordingly.


Step 1: Sample query images + build official captions + prepare ImageNet100 non-member set

COCO2017

export OUT=/path/to/exp_coco
rm -rf "$OUT" && mkdir -p "$OUT"

export Q_CLEAN="$OUT/queries_clean_2000"
rm -rf "$Q_CLEAN" && mkdir -p "$Q_CLEAN"

python tools/sample_images_to_dir.py \
  --src_root "$COCO_IMG" \
  --dst_dir "$Q_CLEAN" \
  --num 2000 \
  --seed $SEED

export CAPS_JSON="$OUT/official_caps_2000.json"
python make_official_captions_array.py \
  --mode coco \
  --images_dir "$Q_CLEAN" \
  --coco_ann "$COCO_CAP_JSON" \
  --out_json "$CAPS_JSON" \
  --n_each 2000 \
  --pick first \
  --seed $SEED


CelebA-Dialog

export OUT=/path/to/exp_celeba
rm -rf "$OUT" && mkdir -p "$OUT"

export Q_CLEAN="$OUT/queries_clean_2000"
rm -rf "$Q_CLEAN" && mkdir -p "$Q_CLEAN"

python tools/sample_images_to_dir.py \
  --src_root "$CELEBA_IMG" \
  --dst_dir "$Q_CLEAN" \
  --num 2000 \
  --seed $SEED

export CAPS_JSON="$OUT/official_caps_2000.json"
python make_official_captions_array.py \
  --mode celeba \
  --images_dir "$Q_CLEAN" \
  --celeba_caps "$CELEBA_TEXT" \
  --out_json "$CAPS_JSON" \
  --n_each 2000 \
  --pick first \
  --seed $SEED


ImageNet100 non-member set sampling

export IN100_2K="$OUT/imagenet100_nonmember_2000"
rm -rf "$IN100_2K" && mkdir -p "$IN100_2K"

python tools/sample_eval_mix_to_dir.py \
  --src_dirs "$IN100_ROOT" \
  --out_dir "$IN100_2K" \
  --num 2000 \
  --seed $SEED \
  --mode copy


Step 2: Synthesize shadow-member images from captions (SD1.5 and SD2.1)

SD1.5 (same synthesis step for both no-noise and noisy tracks)

export SYNTH_DIR="$OUT/synth_sd15_offcaps_2000"
rm -rf "$SYNTH_DIR" && mkdir -p "$SYNTH_DIR"

python step2_sd_synthesize_from_captions.py \
  --captions_json "$CAPS_JSON" \
  --sd_dir "$SD15_DIR" \
  --lora_dir "$LORA_SD15_COCO" \
  --out_dir "$SYNTH_DIR" \
  --steps 30 \
  --guidance 7.5 \
  --seed $SEED \
  --gpu_id $GPU


To switch --lora_dir to the CelebA LoRA:

# COCO:  $LORA_SD15_COCO
# CelebA: $LORA_SD15_CELEBA


SD2.1 (same synthesis step for both no-noise and noisy tracks)

export SYNTH_DIR="$OUT/synth_sd21_offcaps_2000"
rm -rf "$SYNTH_DIR" && mkdir -p "$SYNTH_DIR"

python step2_sd_synthesize_from_captions.py \
  --captions_json "$CAPS_JSON" \
  --sd_dir "$SD21_DIR" \
  --lora_dir "$LORA_SD21_COCO" \
  --out_dir "$SYNTH_DIR" \
  --steps 30 \
  --guidance 7.5 \
  --seed $SEED \
  --gpu_id $GPU


To switch --lora_dir to the CelebA LoRA:

# COCO:  $LORA_SD21_COCO
# CelebA: $LORA_SD21_CELEBA


No-noise track (shadow-nonmember = ImageNet100)

Step 3: Build centroids (BLIP / switchable distance)

python step3_build_centroids_distance.py \
  --synth_dir "$SYNTH_DIR" \
  --noisy_dir "$IN100_2K" \
  --n_each 2000 \
  --blip_dir "$BLIP_DIR" \
  --centroid_out "$OUT/centroids_blip_cosine.pt" \
  --gpu_id $GPU \
  --batch_size 64 \
  --seed $SEED \
  --distance cosine \
  --scan_min -1.0 --scan_max 1.0 --scan_num 41


Optional distances:

# --distance cosine
# --distance l2
# --distance swd --swd_proj 128


Step 4: Final evaluation (member shards vs ImageNet100, distance switchable, with threshold scan table)

COCO members:

python step4_eval_mia_encoder_centroid_distance.py \
  --centroids_pt "$OUT/centroids_blip_cosine.pt" \
  --blip_dir "$BLIP_DIR" \
  --gpu_id $GPU \
  --batch_size 64 \
  --distance auto \
  --query_member_pt_shards_dir "$COCO_MEMBER_SHARDS" \
  --query_nonmember_dir "$IN100_2K" \
  --q_each 2000 \
  --scores_csv "$OUT/mia_scores.csv" \
  --roc_csv "$OUT/mia_roc.csv" \
  --roc_png "$OUT/mia_roc.png" \
  --scan_min -1.0 --scan_max 1.0 --scan_num 41


CelebA members:

python step4_eval_mia_encoder_centroid_distance.py \
  --centroids_pt "$OUT/centroids_blip_cosine.pt" \
  --blip_dir "$BLIP_DIR" \
  --gpu_id $GPU \
  --batch_size 64 \
  --distance auto \
  --query_member_pt_shards_dir "$CELEBA_MEMBER_SHARDS" \
  --query_nonmember_dir "$IN100_2K" \
  --q_each 2000 \
  --scores_csv "$OUT/mia_scores.csv" \
  --roc_csv "$OUT/mia_roc.csv" \
  --roc_png "$OUT/mia_roc.png" \
  --scan_min -1.0 --scan_max 1.0 --scan_num 41


Noisy track (eval-aware: shadow-nonmember = mixed evaluation set + noise; final evaluation remains member shards vs ImageNet100)

Step A: Export eval-member images (from member shards)

COCO:

export EVAL_MEM_DIR="$OUT/eval_member_imgs_2000"
rm -rf "$EVAL_MEM_DIR" && mkdir -p "$EVAL_MEM_DIR"

python tools/export_pt_shards_to_dir.py \
  --shards_dir "$COCO_MEMBER_SHARDS" \
  --out_dir "$EVAL_MEM_DIR" \
  --num 2000 \
  --prefix member


CelebA:

export EVAL_MEM_DIR="$OUT/eval_member_imgs_2000"
rm -rf "$EVAL_MEM_DIR" && mkdir -p "$EVAL_MEM_DIR"

python tools/export_pt_shards_to_dir.py \
  --shards_dir "$CELEBA_MEMBER_SHARDS" \
  --out_dir "$EVAL_MEM_DIR" \
  --num 2000 \
  --prefix member


Step B: Mixed sampling to build the shadow-nonmember source (eval member + ImageNet100)

export SHADOW_NM_SRC="$OUT/shadow_nm_source_2000"
rm -rf "$SHADOW_NM_SRC" && mkdir -p "$SHADOW_NM_SRC"

python tools/sample_eval_mix_to_dir.py \
  --src_dirs "$EVAL_MEM_DIR" "$IN100_2K" \
  --out_dir "$SHADOW_NM_SRC" \
  --num 2000 \
  --seed $SEED \
  --mode copy


Step C: Add noise to the shadow-nonmember source to obtain shadow-nonmember (sigma/jpeg are reproducible)

export SHADOW_NM_NOISY="$OUT/shadow_nm_noisy_2000"
rm -rf "$SHADOW_NM_NOISY" && mkdir -p "$SHADOW_NM_NOISY"

python tools/add_noise_dir.py \
  --src_dir "$SHADOW_NM_SRC" \
  --dst_dir "$SHADOW_NM_NOISY" \
  --sigma 0.10 \
  --jpeg_quality 85 \
  --seed $SEED


Step 3: Build centroids (synth vs shadow_nm_noisy, distance switchable)

python step3_build_centroids_distance.py \
  --synth_dir "$SYNTH_DIR" \
  --noisy_dir "$SHADOW_NM_NOISY" \
  --n_each 2000 \
  --blip_dir "$BLIP_DIR" \
  --centroid_out "$OUT/centroids_evalaware_blip_cosine.pt" \
  --gpu_id $GPU \
  --batch_size 64 \
  --seed $SEED \
  --distance cosine


Step 4: Final evaluation (member shards vs ImageNet100, consistent with historical comparisons)

COCO members:

python step4_eval_mia_encoder_centroid_distance.py \
  --centroids_pt "$OUT/centroids_evalaware_blip_cosine.pt" \
  --blip_dir "$BLIP_DIR" \
  --gpu_id $GPU \
  --batch_size 64 \
  --distance auto \
  --query_member_pt_shards_dir "$COCO_MEMBER_SHARDS" \
  --query_nonmember_dir "$IN100_2K" \
  --q_each 2000 \
  --scores_csv "$OUT/mia_scores_evalaware.csv" \
  --roc_csv "$OUT/mia_roc_evalaware.csv" \
  --roc_png "$OUT/mia_roc_evalaware.png" \
  --scan_min -1.0 --scan_max 1.0 --scan_num 41


CelebA members:

python step4_eval_mia_encoder_centroid_distance.py \
  --centroids_pt "$OUT/centroids_evalaware_blip_cosine.pt" \
  --blip_dir "$BLIP_DIR" \
  --gpu_id $GPU \
  --batch_size 64 \
  --distance auto \
  --query_member_pt_shards_dir "$CELEBA_MEMBER_SHARDS" \
  --query_nonmember_dir "$IN100_2K" \
  --q_each 2000 \
  --scores_csv "$OUT/mia_scores_evalaware.csv" \
  --roc_csv "$OUT/mia_roc_evalaware.csv" \
  --roc_png "$OUT/mia_roc_evalaware.png" \
  --scan_min -1.0 --scan_max 1.0 --scan_num 41


Encoder comparison variants for Step3/Step4 (ViT / CLIP)

ViT (cosine)

Step3:

python step3_build_centroids_vit_cosine.py \
  --synth_dir "$SYNTH_DIR" \
  --noisy_dir "$IN100_2K" \
  --n_each 2000 \
  --vit_dir "$VIT_DIR" \
  --centroid_out "$OUT/centroids_vit_cosine.pt" \
  --gpu_id $GPU \
  --batch_size 64 \
  --seed $SEED


Step4:

python step4_eval_mia_encoder_centroid_vit_cosine.py \
  --centroids_pt "$OUT/centroids_vit_cosine.pt" \
  --vit_dir "$VIT_DIR" \
  --gpu_id $GPU \
  --batch_size 64 \
  --query_member_pt_shards_dir "$MEMBER_SHARDS" \
  --query_nonmember_dir "$IN100_2K" \
  --q_each 2000 \
  --scores_csv "$OUT/mia_scores_vit.csv" \
  --roc_csv "$OUT/mia_roc_vit.csv" \
  --roc_png "$OUT/mia_roc_vit.png"


CLIP (cosine)

Step3:

python step3_build_centroids_clip_cosine.py \
  --synth_dir "$SYNTH_DIR" \
  --noisy_dir "$IN100_2K" \
  --n_each 2000 \
  --clip_dir "$CLIP_DIR" \
  --centroid_out "$OUT/centroids_clip_cosine.pt" \
  --gpu_id $GPU \
  --batch_size 64 \
  --seed $SEED


Step4:

python step4_eval_mia_encoder_centroid_clip_cosine.py \
  --centroids_pt "$OUT/centroids_clip_cosine.pt" \
  --clip_dir "$CLIP_DIR" \
  --gpu_id $GPU \
  --batch_size 64 \
  --query_member_pt_shards_dir "$MEMBER_SHARDS" \
  --query_nonmember_dir "$IN100_2K" \
  --q_each 2000 \
  --scores_csv "$OUT/mia_scores_clip.csv" \
  --roc_csv "$OUT/mia_roc_clip.csv" \
  --roc_png "$OUT/mia_roc_clip.png"


Style-classifier MIA variant (ImageNet original vs SD synth)

This variant trains a binary "style classifier" to separate ImageNet original images from SD synthesized images,
then uses its predicted synth-probability as an MIA score for member(pt-shards) vs non-member(ImageNet dir).
Outputs include ACC(at threshold), ROC/AUC, and TPR@FPR=1%.


Environment (offline)

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/path/to/hf_cache
source /path/to/miniconda3/bin/activate <ENV_NAME>


Step S1: Train the style binary classifier (freeze BLIP encoder + MLP head)

Inputs:
- ImageNet originals: --in_pt (expects {'image': list})
- SD synth dir:       --synth_dir (expects img_{i:04d}_01.jpg aligned to in_pt order)
- Encoder:            --blip_dir (local fine-tuned BLIP; frozen for feature extraction)

Outputs:
- <SAVE_DIR>/style_mlp_best.pt
- <SAVE_DIR>/style_mlp_last.pt

Key hyperparameters:
- batch_size=32, epochs=8, lr=3e-4, weight_decay=1e-4
- val_ratio=0.1
- MLP: cls_hidden=256, dropout=0.1

python train_style_binary_classifier.py \
  --in_pt <IN100_PT> \
  --synth_dir <IN100_SYNTH_DIR> \
  --blip_dir <BLIP_DIR> \
  --save_dir <STYLE_CLF_OUT_DIR> \
  --gpu_id $GPU \
  --batch_size 32 \
  --epochs 8 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --val_ratio 0.1 \
  --cls_hidden 256 \
  --dropout 0.1 \
  --num_workers 0


Step S2: MIA attack with ROC/AUC (member from pt-shards, non-member from ImageNet dir)

This script samples n_each images from each side:
- member:    --member_pt_shards_dir (label=1)
- nonmember: --nonmember_dir        (label=0)
Score is prob_synth = softmax(logits)[:,1] from the style classifier.

Outputs:
- per-sample detail CSV: --save_csv
- ROC points CSV:        --roc_csv
- ROC curve PNG:         --roc_png
- printed summary: ACC(threshold), AUC, TPR@FPR=1%

Key hyperparameters:
- n_each=1000 per side (template)
- threshold=0.5 for ACC
- batch_size=32

python mia_attack_ptshards_vs_imagenet_roc.py \
  --member_pt_shards_dir <MEMBER_PT_SHARDS_DIR> \
  --nonmember_dir <IN100_ROOT> \
  --blip_dir <BLIP_DIR> \
  --clf_ckpt <STYLE_CLF_OUT_DIR>/style_mlp_best.pt \
  --save_csv <OUT_DIR>/mia_results_detail.csv \
  --roc_csv <OUT_DIR>/mia_roc_points.csv \
  --roc_png <OUT_DIR>/mia_roc_curve.png \
  --gpu_id $GPU \
  --batch_size 32 \
  --num_workers 0 \
  --n_each 1000 \
  --threshold 0.5


DP-SGD LoRA training (training commands only)

SD1.5:

accelerate launch train_text_to_image_lora_dpsgd_map_bmm_v6_dp_only.py \
  --pretrained_model_name_or_path=$SD15_DIR \
  --train_data_dir=$MEMBER_SHARDS \
  --dataloader_num_workers=2 \
  --mixed_precision=fp16 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --learning_rate=1e-04 \
  --lr_scheduler=cosine --lr_warmup_steps=0 \
  --rank=2 \
  --output_dir=/path/to/lora_dp_sd15 \
  --report_to=wandb \
  --resume_from_checkpoint=latest \
  --checkpointing_steps=5000 \
  --validation_prompt="a portrait" \
  --validation_steps=7500 \
  --seed=$SEED \
  --enable_dp \
  --dp_noise_multiplier=6.0 \
  --dp_max_grad_norm=0.1 \
  --dp_poisson_sampling \
  --dp_max_physical_batch_size=1 \
  --dp_force_manual_accountant_step


SD2.1:

accelerate launch train_text_to_image_lora_dpsgd_map_bmm_v6_dp_only.py \
  --pretrained_model_name_or_path=$SD21_DIR \
  --train_data_dir=$MEMBER_SHARDS \
  --dataloader_num_workers=2 \
  --mixed_precision=fp16 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --learning_rate=1e-04 \
  --lr_scheduler=cosine --lr_warmup_steps=0 \
  --rank=2 \
  --output_dir=/path/to/lora_dp_sd21 \
  --report_to=wandb \
  --resume_from_checkpoint=latest \
  --checkpointing_steps=5000 \
  --validation_prompt="a portrait" \
  --validation_steps=7500 \
  --seed=$SEED \
  --enable_dp \
  --dp_noise_multiplier=6.0 \
  --dp_max_grad_norm=0.1 \
  --dp_poisson_sampling \
  --dp_max_physical_batch_size=1 \
  --dp_force_manual_accountant_step
