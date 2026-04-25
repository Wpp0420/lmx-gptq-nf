"""
准备本地评估数据
================
从已缓存的 HuggingFace 数据集中导出文本文件，用于离线评估。
也可以用任何 UTF-8 文本文件作为评估数据。

使用:
    # 导出 WikiText-2 test 到 txt
    python prepare_local_eval_data.py \
        --dataset wikitext2 \
        --local_files_only \
        --hf_cache_dir /home/.cache/huggingface

    # 导出 WikiText-2 train 的前 2000 条到 txt
    python prepare_local_eval_data.py \
        --dataset wikitext2_train \
        --max_samples 2000 \
        --output ./eval_data/wikitext2_train.txt

    # 导出后使用:
    python run_pipeline.py --model /path/to/model --eval_datasets local:./eval_data/wikitext2.txt
    python run_pipeline.py --model /path/to/model --use_smoothquant --smoothquant_dataset local:./eval_data/wikitext2_train.txt
"""

import os
import logging
import argparse

from config import EVAL_DATASET_PRESETS
from utils import load_hf_dataset

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


EXPORT_PRESETS = {
    "wikitext2": {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "field": "text",
        "output": "/home/wuping/infra/test7_rtn/Qwen3_GPTQ_nf4/eval_data/wikitext2.txt",
    },
    "wikitext2_train": {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "train",
        "field": "text",
        "output": "/home/wuping/infra/test7_rtn/Qwen3_GPTQ_nf4/eval_data/wikitext2_train.txt",
    },
    "wikitext103": {
        "name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "test",
        "field": "text",
        "output": "/home/wuping/infra/test7_rtn/Qwen3_GPTQ_nf4/eval_data/wikitext103.txt",
    },
    "lambada": {
        "name": "EleutherAI/lambada_openai",
        "config": "en",
        "split": "test",
        "field": "text",
        "output": "/home/wuping/infra/test7_rtn/Qwen3_GPTQ_nf4/eval_data/lambada.txt",
    },
}


def export_cached_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_field: str,
    output_path: str,
    max_samples: int = None,
    local_files_only: bool = False,
    hf_cache_dir: str = None,
):
    """从 HF 缓存中导出数据集为纯文本文件。"""
    logger.info(f"加载: {dataset_name}/{dataset_config} ({split})...")

    dataset = load_hf_dataset(
        dataset_name,
        dataset_config,
        split=split,
        local_files_only=local_files_only,
        cache_dir=hf_cache_dir,
    )

    if text_field not in dataset.column_names:
        raise ValueError(
            f"数据集字段 '{text_field}' 不存在，可用字段: {dataset.column_names}"
        )

    texts = [s for s in dataset[text_field] if isinstance(s, str) and s.strip()]

    if max_samples:
        texts = texts[:max_samples]

    text = "\n\n".join(texts)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  -> {output_path} ({len(texts)} 条, {size_mb:.2f} MB)")
    return output_path


def resolve_export_spec(args):
    if args.dataset in EXPORT_PRESETS:
        spec = dict(EXPORT_PRESETS[args.dataset])
    elif args.dataset in EVAL_DATASET_PRESETS:
        preset = EVAL_DATASET_PRESETS[args.dataset]
        spec = {
            "name": preset["path"],
            "config": preset["config"],
            "split": preset["split"],
            "field": preset["text_field"],
            "output": f"./eval_data/{args.dataset}.txt",
        }
    else:
        spec = {
            "name": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "field": args.text_field,
            "output": "./eval_data/exported_dataset.txt",
        }

    if args.output:
        spec["output"] = args.output
    if args.max_samples is not None:
        spec["max_samples"] = args.max_samples
    return spec


def main():
    parser = argparse.ArgumentParser(description="从已缓存的 Hugging Face 数据集导出 txt 文件")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        help="导出预设名或 HF 数据集名，例如 wikitext2 / wikitext2_train / lambada / wikitext",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="当 --dataset 传 HF 数据集名时使用",
    )
    parser.add_argument("--split", type=str, default="test", help="数据集 split")
    parser.add_argument("--text_field", type=str, default="text", help="文本字段名")
    parser.add_argument("--output", type=str, default=None, help="输出 txt 路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最多导出多少条样本")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="仅从本地缓存读取，不访问网络",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Hugging Face 缓存目录，例如 /home/.cache/huggingface",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("准备本地评估数据")
    logger.info("=" * 60)

    spec = resolve_export_spec(args)
    logger.info(
        "导出配置: %s/%s (split=%s, field=%s)",
        spec["name"],
        spec["config"],
        spec["split"],
        spec["field"],
    )

    output_path = export_cached_dataset(
        dataset_name=spec["name"],
        dataset_config=spec["config"],
        split=spec["split"],
        text_field=spec["field"],
        output_path=spec["output"],
        max_samples=spec.get("max_samples"),
        local_files_only=args.local_files_only,
        hf_cache_dir=args.hf_cache_dir,
    )

    logger.info("\n使用示例:")
    logger.info(
        "  评估数据: python run_pipeline.py --model MODEL --skip_quantize --eval_datasets local:%s",
        output_path,
    )
    logger.info(
        "  SmoothQuant: python run_pipeline.py --model MODEL --skip_quantize --use_smoothquant --smoothquant_dataset local:%s",
        output_path,
    )


if __name__ == "__main__":
    main()
