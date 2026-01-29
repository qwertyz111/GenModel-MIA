import os
import random
import datasets
from datasets import load_from_disk

import trl
from attack.utils import create_folder

block_size = None
tokenizer_ = None
max_buff_size = None
text_column = None


def packing_texts(examples):
    more_examples = True
    packed_texts = []
    packed_ids = []
    assert list(examples.keys()) == ["text"]
    iterator = iter(examples["text"])
    total_num = 0
    drop_num = 0
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size:
                break
            try:
                buffer.append(next(iterator))
                buffer_len += len(buffer[-1])
            except StopIteration:
                more_examples = False
                break
        tokenized_inputs = tokenizer_(buffer, truncation=False)["input_ids"]
        inputs = tokenizer_.batch_decode(tokenized_inputs)
        tokenized_inputs = tokenizer_(inputs, truncation=False)["input_ids"]
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input)
        for i in range(0, len(all_token_ids), block_size):
            input_ids = all_token_ids[i: i + block_size]
            if len(input_ids) == block_size:
                packed_ids.append(input_ids)
                input_text = tokenizer_.decode(input_ids)
                total_num += 1
                if len(tokenizer_.encode(input_text)) == block_size:
                    packed_texts.append(input_text)
                    drop_num += 1
    return {"text": packed_texts}


def _detect_text_column(dataset):
    """
    尝试在数据集中找到文本列名称，优先级：text > document > content。
    若都不存在且只有一列，则使用该唯一列；否则报错。
    """
    cols = dataset.column_names  # list[str]
    candidates = ["text", "document", "content"]
    for c in candidates:
        if c in cols:
            return c
    if len(cols) == 1:
        return cols[0]
    raise ValueError(f"Could not find a text column in columns: {cols}")


def dataset_prepare(args, tokenizer=None, num_of_sequences=1024, chars_per_token=3.6):
    """
    支持两种加载方式：
    1) 若传入 args.dataset_local_dir：从 datasets.save_to_disk 目录离线读取；
       并按照 args.validation_split_percentage 从 train 划分一部分作为 valid。
    2) 否则按原逻辑用 datasets.load_dataset(args.dataset_name, ...) 加载 train 的前/后百分比。
    然后统一标准化列名为 'text'，并在 args.packing=True 时进行打包。
    """
    # -------------------------
    # 1) 加载原始数据
    # -------------------------
    if getattr(args, "dataset_local_dir", None):
        raw = load_from_disk(args.dataset_local_dir)
        # raw 可能是 DatasetDict 或 Dataset
        if isinstance(raw, datasets.DatasetDict):
            if "train" in raw:
                train_full = raw["train"]
            else:
                # 若没有 'train'（极少见），取第一个 split 当 train_full
                first_split = next(iter(raw.keys()))
                train_full = raw[first_split]
        else:
            # 单个 Dataset 当作 train_full
            train_full = raw

        # 按比例从 train_full 划分 train/valid（保持与原脚本同等语义）
        val_pct = float(getattr(args, "validation_split_percentage", 0.1))
        cut = int((1.0 - val_pct) * len(train_full))
        cut = max(0, min(cut, len(train_full)))  # 边界保护
        train_dataset = train_full.select(range(cut)) if cut > 0 else train_full.select([])  # 可能为 0
        valid_dataset = train_full.select(range(cut, len(train_full)))
    else:
        # 原脚本逻辑：使用同一个 train split 的前/后百分比
        head_pct = int((1 - args.validation_split_percentage) * 100)
        tail_pct = int(args.validation_split_percentage * 100)
        train_dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{head_pct}%]"
        )
        valid_dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[-{tail_pct}%:]"
        )

    # -------------------------
    # 2) 统一文本列为 'text'
    # -------------------------
    global text_column
    text_column = _detect_text_column(train_dataset)

    # 只保留文本列
    train_dataset = train_dataset.select_columns([text_column])
    valid_dataset = valid_dataset.select_columns([text_column])

    # 重命名为 'text'
    if text_column != "text":
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")

    # -------------------------
    # 3) 可选：packing
    # -------------------------
    if args.packing:
        global block_size, tokenizer_, max_buff_size
        block_size = args.block_size
        # 注意取整，避免小数
        max_buff_size = int(block_size * chars_per_token * num_of_sequences)
        tokenizer_ = tokenizer

        # 统一 cache 路径：dataset_config_name 为 None 时用 "default"
        ds_conf = args.dataset_config_name if args.dataset_config_name is not None else "default"
        cache_dir = os.path.join(args.cache_path, args.dataset_name, str(ds_conf))
        create_folder(cache_dir)

        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=os.path.join(cache_dir, "train_dataset.arrow"),
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens",
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=os.path.join(cache_dir, "valid_dataset.arrow"),
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens",
        )

    # 确保无论是否 packing 都返回
    return train_dataset, valid_dataset
