"""
GPTQ W4A8 量化 - 一键运行脚本
===============================
一站式完成: 量化 -> A8包装 -> 困惑度评估

配置说明:
    量化与激活相关参数统一在 config.py 中修改。
    run_pipeline.py 只保留流程控制和运行环境相关参数，避免 CLI 默认值覆盖 config.py。

使用方法:
    # Qwen3-1.7B 完整流程
    python run_pipeline.py --model /path/to/Qwen3-1.7B

    # 只评估已有量化模型
    python run_pipeline.py --model /path/to/Qwen3-1.7B --skip_quantize --quantized_model ./Qwen3-1.7B_w4a8_gptq
"""

import os
import sys
import json
import argparse
import logging
from copy import deepcopy
from datetime import datetime

import torch

from config import EVAL_DATASET_PRESETS, W4A8Config, prepare_runtime_config, resolve_eval_config, resolve_dataset_source_args
from quantization_analysis import run_vector_demo
from utils import setup_logging, Timer, print_device_info, print_gpu_memory
from quantize_gptq import quantize_with_autogptq, quantize_with_transformers
from custom_gptq_backend import quantize_with_custom_backend
from evaluate_ppl import (
    evaluate_fp16_model,
    evaluate_gptq_w4_model,
    evaluate_w4a8_model,
)

logger = logging.getLogger(__name__)


def run_full_pipeline(
    pipeline_config: W4A8Config,
    skip_quantize: bool = False,
    skip_eval: bool = False,
    skip_fp16_eval: bool = False,
    quantized_model_path: str = None,
    backend: str = "autogptq",
    eval_datasets: list = None,
):
    """执行完整 W4A8 量化流程。"""
    print_device_info()

    gptq_config = deepcopy(pipeline_config.gptq)
    act_config = deepcopy(pipeline_config.activation)
    base_eval_config = deepcopy(pipeline_config.evaluation)

    model_name = gptq_config.model_name_or_path
    output_dir = gptq_config.output_dir

    if quantized_model_path is None:
        quantized_model_path = output_dir

    dataset_keys = eval_datasets or base_eval_config.eval_datasets or ["wikitext2"]

    smoothquant_source = resolve_dataset_source_args(
        act_config.smoothquant_dataset,
        act_config.smoothquant_dataset_config,
        act_config.smoothquant_split,
        act_config.smoothquant_text_field,
        fallback_dataset_key=dataset_keys[0] if dataset_keys else "wikitext2",
    )
    act_config.smoothquant_dataset = smoothquant_source["dataset"]
    act_config.smoothquant_dataset_config = smoothquant_source["dataset_config"]
    act_config.smoothquant_split = smoothquant_source["split"]
    act_config.smoothquant_text_field = smoothquant_source["text_field"]

    all_results = {
        "config": {
            "model": model_name,
            "output_dir": output_dir,
            "quantized_model_path": quantized_model_path,
            "weight_bits": gptq_config.bits,
            "weight_quant_scheme": gptq_config.weight_quant_scheme,
            "group_size": gptq_config.group_size,
            "num_calibration_samples": gptq_config.num_calibration_samples,
            "quant_max_length": gptq_config.max_length,
            "eval_max_length": base_eval_config.max_length,
            "eval_stride": base_eval_config.stride,
            "act_bits": act_config.act_bits,
            "activation_quant_scheme": act_config.activation_quant_scheme,
            "act_symmetric": act_config.symmetric,
            "act_granularity": act_config.granularity,
            "use_smoothquant": act_config.use_smoothquant,
            "smoothquant_alpha": act_config.smoothquant_alpha,
            "smoothquant_dataset": act_config.smoothquant_dataset,
            "smoothquant_dataset_config": act_config.smoothquant_dataset_config,
            "smoothquant_split": act_config.smoothquant_split,
            "smoothquant_text_field": act_config.smoothquant_text_field,
            "local_files_only": gptq_config.local_files_only,
            "hf_cache_dir": gptq_config.hf_cache_dir,
            "enable_vector_demo": pipeline_config.analysis.enable_vector_demo,
            "enable_layer_distribution": pipeline_config.analysis.enable_layer_distribution,
        }
    }

    logger.info("当前统一配置:")
    logger.info(
        "  W%sA%s, group_size=%s, weight_scheme=%s, act_scheme=%s, act_granularity=%s, smoothquant=%s",
        gptq_config.bits,
        act_config.act_bits,
        gptq_config.group_size,
        gptq_config.weight_quant_scheme,
        act_config.activation_quant_scheme,
        act_config.granularity,
        act_config.use_smoothquant,
    )
    if act_config.use_smoothquant:
        logger.info(
            "  SmoothQuant 校准数据源: %s/%s (split=%s, field=%s)",
            act_config.smoothquant_dataset,
            act_config.smoothquant_dataset_config,
            act_config.smoothquant_split,
            act_config.smoothquant_text_field,
        )

    if pipeline_config.analysis.enable_vector_demo:
        vector_output_dir = os.path.join(output_dir, pipeline_config.analysis.output_subdir, "vector_demo")
        run_vector_demo(
            output_dir=vector_output_dir,
            alpha=act_config.smoothquant_alpha,
        )
        logger.info("向量量化对比示例已生成: %s", vector_output_dir)

    if gptq_config.weight_quant_scheme == "nf4" and backend != "custom":
        logger.info("NF4 权重量化需要 custom 后端，已自动切换 backend=custom")
        backend = "custom"
    if pipeline_config.analysis.enable_layer_distribution and backend != "custom":
        logger.info("逐层分布分析依赖 custom 后端，已自动切换 backend=custom")
        backend = "custom"

    if not skip_quantize:
        logger.info("\n" + "=" * 70)
        logger.info("  阶段 1: GPTQ W4 权重量化")
        logger.info("=" * 70)

        with Timer("GPTQ 量化总耗时"):
            if backend == "autogptq":
                quantize_with_autogptq(gptq_config)
            elif backend == "transformers":
                quantize_with_transformers(gptq_config)
            else:
                quantize_with_custom_backend(pipeline_config)

        quantized_model_path = output_dir
        print_gpu_memory()
        torch.cuda.empty_cache()
    else:
        logger.info("跳过量化步骤，使用已有量化模型")
        if not os.path.exists(quantized_model_path):
            logger.error(f"量化模型路径不存在: {quantized_model_path}")
            sys.exit(1)

    if not skip_eval:
        logger.info("\n" + "=" * 70)
        logger.info("  阶段 2: 困惑度评估")
        logger.info("=" * 70)

        for ds_key in dataset_keys:
            logger.info(f"\n{'#' * 70}")
            logger.info(f"  数据集: {ds_key}")
            if ds_key in EVAL_DATASET_PRESETS:
                logger.info(f"  {EVAL_DATASET_PRESETS[ds_key]['description']}")
            logger.info(f"{'#' * 70}")

            try:
                eval_config = resolve_eval_config(ds_key, base_eval_config)
            except ValueError as exc:
                logger.warning(f"跳过未知数据集 '{ds_key}': {exc}")
                continue

            ds_results = {}

            if not skip_fp16_eval:
                logger.info("\n--- 评估 FP16 原始模型 ---")
                with Timer("FP16 评估"):
                    ds_results["fp16"] = evaluate_fp16_model(model_name, eval_config)
                torch.cuda.empty_cache()

            logger.info("\n--- 评估 W4A16 量化模型 ---")
            with Timer("W4A16 评估"):
                ds_results["gptq_w4"] = evaluate_gptq_w4_model(
                    quantized_model_path,
                    eval_config,
                )
            torch.cuda.empty_cache()

            logger.info("\n--- 评估 W4A8 量化模型 ---")
            with Timer("W4A8 评估"):
                ds_results["w4a8"] = evaluate_w4a8_model(
                    quantized_model_path,
                    eval_config,
                    act_config,
                )
            torch.cuda.empty_cache()

            all_results[ds_key] = ds_results

        logger.info("\n" + "=" * 70)
        logger.info("  最终评估对比")
        logger.info("=" * 70)

        for ds_key in dataset_keys:
            if ds_key not in all_results:
                continue

            ds_results = all_results[ds_key]
            first_result = next(
                (ds_results[k] for k in ["fp16", "gptq_w4", "w4a8"] if k in ds_results),
                None,
            )
            is_accuracy = first_result and "accuracy" in first_result

            if is_accuracy:
                logger.info(f"{'数据集':<15} {'模型':<25} {'Accuracy':>10} {'正确/总数':>15} {'变化':>10}")
                logger.info("-" * 75)
                base_acc = ds_results.get("fp16", {}).get("accuracy", None)

                for key in ["fp16", "gptq_w4", "w4a8"]:
                    if key not in ds_results:
                        continue
                    result = ds_results[key]
                    acc_pct = result["accuracy_pct"]
                    count_str = f"{result['num_correct']}/{result['num_total']}"
                    change = ""
                    if base_acc and key != "fp16" and base_acc > 0:
                        delta = ((result["accuracy"] - base_acc) / base_acc) * 100
                        sign = "+" if delta >= 0 else ""
                        change = f"{sign}{delta:.2f}%"
                    name = result.get("model_name", key)
                    logger.info(f"  {ds_key:<13} {name:<23} {acc_pct:>9.2f}% {count_str:>15} {change:>10}")
            else:
                logger.info(f"{'数据集':<15} {'模型':<25} {'PPL':>10} {'Loss':>12} {'PPL变化':>10}")
                logger.info("-" * 75)
                base_ppl = ds_results.get("fp16", {}).get("perplexity", None)

                for key in ["fp16", "gptq_w4", "w4a8"]:
                    if key not in ds_results:
                        continue
                    result = ds_results[key]
                    ppl = result["perplexity"]
                    loss = result["loss"]
                    change = ""
                    if base_ppl and key != "fp16":
                        delta = ((ppl - base_ppl) / base_ppl) * 100
                        change = f"+{delta:.2f}%"
                    name = result.get("model_name", key)
                    logger.info(f"  {ds_key:<13} {name:<23} {ppl:>10.4f} {loss:>12.6f} {change:>10}")

            if ds_key != dataset_keys[-1]:
                logger.info("-" * 75)

        logger.info("=" * 70)

    all_results["timestamp"] = datetime.now().isoformat()
    results_path = os.path.join(output_dir if os.path.isdir(output_dir) else ".", "pipeline_results.json")
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n完整结果已保存到: {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ W4A8 量化 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
统一配置方式:
  量化与激活相关参数统一在 config.py 修改。
  命令行只保留流程控制与运行环境参数。

示例:
  python run_pipeline.py --model /path/to/Qwen3-1.7B
  python run_pipeline.py --model /path/to/Qwen3-1.7B --skip_quantize --quantized_model ./Qwen3-1.7B_w4a8_gptq
        """,
    )

    parser.add_argument("--model", type=str, required=True, help="模型名称或本地路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录；默认由 config.py 自动推导")
    parser.add_argument("--quantized_model", type=str, default=None, help="已有的量化模型路径")
    parser.add_argument(
        "--eval_datasets",
        type=str,
        nargs="+",
        default=None,
        help=f"评估数据集列表，可用预设: {list(EVAL_DATASET_PRESETS.keys())}；不传则使用 config.py 默认值",
    )
    parser.add_argument("--skip_quantize", action="store_true", help="跳过量化步骤")
    parser.add_argument("--skip_eval", action="store_true", help="跳过评估步骤")
    parser.add_argument("--skip_fp16_eval", action="store_true", help="跳过 FP16 模型评估")

    # 兼容旧接口: 这些参数已迁移到 config.py，保留仅用于给出明确提示。
    parser.add_argument("--act_bits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--act_granularity", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--act_asymmetric", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use_smoothquant", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoothquant_alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoothquant_dataset", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoothquant_dataset_config", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoothquant_split", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoothquant_text_field", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoothquant_num_samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoothquant_scales_path", type=str, default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--backend",
        type=str,
        default="autogptq",
        choices=["autogptq", "transformers", "custom"],
        help="量化后端",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--local_files_only", action="store_true", help="仅从本地缓存加载 Hugging Face 模型/数据集")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="Hugging Face 缓存目录")
    parser.add_argument("--run_vector_demo", action="store_true", help="输出简单向量的 INT4/NF4 与 INT8/NF8 对比图")
    parser.add_argument("--run_layer_distribution", action="store_true", help="输出每层每个 block 的权重/激活分布对比")

    args = parser.parse_args()

    deprecated_args = []
    if args.act_bits is not None:
        deprecated_args.append("--act_bits -> config.py / ActivationQuantConfig.act_bits")
    if args.act_granularity is not None:
        deprecated_args.append("--act_granularity -> config.py / ActivationQuantConfig.granularity")
    if args.act_asymmetric:
        deprecated_args.append("--act_asymmetric -> config.py / ActivationQuantConfig.symmetric")
    if args.use_smoothquant:
        deprecated_args.append("--use_smoothquant -> config.py / ActivationQuantConfig.use_smoothquant")
    if args.smoothquant_alpha is not None:
        deprecated_args.append("--smoothquant_alpha -> config.py / ActivationQuantConfig.smoothquant_alpha")
    if args.smoothquant_dataset is not None:
        deprecated_args.append("--smoothquant_dataset -> config.py / ActivationQuantConfig.smoothquant_dataset")
    if args.smoothquant_dataset_config is not None:
        deprecated_args.append("--smoothquant_dataset_config -> config.py / ActivationQuantConfig.smoothquant_dataset_config")
    if args.smoothquant_split is not None:
        deprecated_args.append("--smoothquant_split -> config.py / ActivationQuantConfig.smoothquant_split")
    if args.smoothquant_text_field is not None:
        deprecated_args.append("--smoothquant_text_field -> config.py / ActivationQuantConfig.smoothquant_text_field")
    if args.smoothquant_num_samples is not None:
        deprecated_args.append("--smoothquant_num_samples -> config.py / ActivationQuantConfig.smoothquant_num_samples")
    if args.smoothquant_scales_path is not None:
        deprecated_args.append("--smoothquant_scales_path -> config.py / ActivationQuantConfig.smoothquant_scales_path")

    if deprecated_args:
        parser.error(
            "以下旧参数已迁移到 config.py，请在配置文件中修改后重新运行:\n  - "
            + "\n  - ".join(deprecated_args)
        )

    setup_logging()

    pipeline_config = prepare_runtime_config(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        local_files_only=args.local_files_only,
        hf_cache_dir=args.hf_cache_dir,
    )

    logger.info("GPTQ W4A8 量化 Pipeline")
    logger.info(f"模型: {pipeline_config.gptq.model_name_or_path}")

    pipeline_config.analysis.enable_vector_demo = args.run_vector_demo
    pipeline_config.analysis.enable_layer_distribution = args.run_layer_distribution
    logger.info(
        "统一配置: W%sA%s, group_size=%s, weight_scheme=%s, act_scheme=%s, act_granularity=%s",
        pipeline_config.gptq.bits,
        pipeline_config.activation.act_bits,
        pipeline_config.gptq.group_size,
        pipeline_config.gptq.weight_quant_scheme,
        pipeline_config.activation.activation_quant_scheme,
        pipeline_config.activation.granularity,
    )

    run_full_pipeline(
        pipeline_config=pipeline_config,
        skip_quantize=args.skip_quantize,
        skip_eval=args.skip_eval,
        skip_fp16_eval=args.skip_fp16_eval,
        quantized_model_path=args.quantized_model,
        backend=args.backend,
        eval_datasets=args.eval_datasets,
    )


if __name__ == "__main__":
    main()
