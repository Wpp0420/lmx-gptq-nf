"""
GPTQ W4 权重量化脚本
=====================
使用 GPTQ 算法将模型权重量化为 4-bit

GPTQ 核心原理:
1. 逐层量化：对每一层的权重矩阵单独进行量化
2. 基于 Hessian 的最优量化：利用校准数据计算 Hessian 矩阵 H = 2X^TX
   其中 X 是该层的输入激活值
3. 最优量化顺序：按 Hessian 对角线元素的大小排序，
   优先量化对输出影响较小的权重列
4. 误差补偿：量化一列权重后，将量化误差传播到未量化的列，
   通过 OBS (Optimal Brain Surgeon) 公式: δw = -(w_q - w) * H^{-1}_{:,q} / H^{-1}_{q,q}
5. 分组量化(Group-wise)：将权重行分成大小为 group_size 的组，
   每组共享 scale 和 zero_point，平衡精度和压缩率
"""

import os
import sys
import logging
import argparse
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import GPTQQuantConfig, prepare_runtime_config
from custom_gptq_backend import quantize_with_custom_backend
from utils import (
    setup_logging, set_seed, print_device_info,
    load_calibration_data_for_autogptq, Timer, print_gpu_memory,
    run_with_hf_fallback, hf_model_kwargs,
)

logger = logging.getLogger(__name__)


def _patch_autogptq_for_qwen3():
    """
    为 auto_gptq 添加 qwen3 支持。
    Qwen3 架构与 Qwen2 基本一致，复用 Qwen2 的量化实现即可。
    """
    from auto_gptq.modeling._utils import SUPPORTED_MODELS
    from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP
    from auto_gptq.modeling.qwen2 import Qwen2GPTQForCausalLM

    if "qwen3" not in SUPPORTED_MODELS:
        SUPPORTED_MODELS.append("qwen3")
        logger.info("已将 qwen3 添加到 auto_gptq SUPPORTED_MODELS")

    if "qwen3" not in GPTQ_CAUSAL_LM_MODEL_MAP:
        GPTQ_CAUSAL_LM_MODEL_MAP["qwen3"] = Qwen2GPTQForCausalLM
        logger.info("已将 qwen3 映射到 Qwen2GPTQForCausalLM")


def quantize_with_autogptq(config: GPTQQuantConfig):
    """
    使用 auto-gptq 库进行 GPTQ 量化
    
    流程:
    1. 加载预训练模型和分词器
    2. 准备校准数据集
    3. 配置 GPTQ 量化参数
    4. 执行量化
    5. 保存量化模型
    """
    if config.weight_quant_scheme != "int4":
        raise ValueError("auto_gptq 后端仅支持原始 INT4。NF4 方案请使用 --backend custom")

    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    _patch_autogptq_for_qwen3()
    
    set_seed(config.seed)
    print_device_info()
    
    # ======== Step 1: 配置量化参数 ========
    logger.info("Step 1: 配置 GPTQ 量化参数")
    quantize_config = BaseQuantizeConfig(
        bits=config.bits,                    # 量化位数: 4-bit
        group_size=config.group_size,        # 分组大小: 128
        damp_percent=config.damp_percent,    # 阻尼因子: 0.01
        desc_act=config.desc_act,            # 基于激活排序
        sym=config.sym,                      # 对称量化
        true_sequential=config.true_sequential,  # 真正顺序量化
    )
    
    logger.info(f"量化配置:")
    logger.info(f"  位数 (bits): {config.bits}")
    logger.info(f"  分组大小 (group_size): {config.group_size}")
    logger.info(f"  阻尼因子 (damp_percent): {config.damp_percent}")
    logger.info(f"  对称量化 (sym): {config.sym}")
    logger.info(f"  顺序量化 (true_sequential): {config.true_sequential}")
    logger.info(f"  激活排序 (desc_act): {config.desc_act}")
    
    # ======== Step 2: 加载模型 ========
    logger.info(f"\nStep 2: 加载模型 {config.model_name_or_path}")
    
    with Timer("模型加载"):
        tokenizer = run_with_hf_fallback(
            lambda local_files_only: AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=True,
                **hf_model_kwargs(local_files_only, cache_dir=config.hf_cache_dir),
            ),
            f"tokenizer {config.model_name_or_path}",
            local_files_only=config.local_files_only,
        )
        
        model = run_with_hf_fallback(
            lambda local_files_only: AutoGPTQForCausalLM.from_pretrained(
                config.model_name_or_path,
                quantize_config=quantize_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **hf_model_kwargs(local_files_only, cache_dir=config.hf_cache_dir),
            ),
            f"gptq model {config.model_name_or_path}",
            local_files_only=config.local_files_only,
        )
    
    print_gpu_memory()
    
    # ======== Step 3: 准备校准数据 ========
    logger.info("\nStep 3: 准备校准数据")
    
    with Timer("校准数据准备"):
        calibration_texts = load_calibration_data_for_autogptq(
            tokenizer=tokenizer,
            dataset_name=config.dataset,
            dataset_config=config.dataset_config,
            num_samples=config.num_calibration_samples,
            max_length=config.max_length,
            seed=config.seed,
            local_files_only=config.local_files_only,
            cache_dir=config.hf_cache_dir,
        )
        
        # 将文本转换为 auto-gptq 需要的格式
        calibration_dataset = []
        for text in calibration_texts:
            tokenized = tokenizer(
                text,
                return_tensors="pt",
                max_length=config.max_length,
                truncation=True,
                padding=False,
            )
            calibration_dataset.append(tokenized)
    
    logger.info(f"校准数据集大小: {len(calibration_dataset)} 条")
    
    # ======== Step 4: 执行 GPTQ 量化 ========
    logger.info("\nStep 4: 执行 GPTQ 量化（这可能需要较长时间）")
    logger.info("GPTQ 量化过程:")
    logger.info("  → 收集每层的激活值（前向传播校准数据）")
    logger.info("  → 逐层计算 Hessian 矩阵 H = 2X^TX")
    logger.info("  → 分组量化权重，使用 OBS 补偿量化误差")
    logger.info("  → 保存量化的权重和量化参数 (scale, zero_point)")
    
    with Timer("GPTQ 量化"):
        model.quantize(
            calibration_dataset,
            batch_size=1,
        )
    
    print_gpu_memory()
    
    # ======== Step 5: 保存量化模型 ========
    logger.info(f"\nStep 5: 保存量化模型到 {config.output_dir}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    with Timer("模型保存"):
        model.save_quantized(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
    
    # 计算压缩率
    original_size_gb = os.path.getsize(
        os.path.join(config.model_name_or_path, "model.safetensors")
    ) / (1024**3) if os.path.exists(os.path.join(config.model_name_or_path, "model.safetensors")) else None
    
    quantized_files = [f for f in os.listdir(config.output_dir) 
                       if f.endswith(('.safetensors', '.bin'))]
    quantized_size_gb = sum(
        os.path.getsize(os.path.join(config.output_dir, f)) 
        for f in quantized_files
    ) / (1024**3)
    
    logger.info(f"\n量化完成!")
    logger.info(f"  量化模型大小: {quantized_size_gb:.2f} GB")
    if original_size_gb:
        logger.info(f"  原始模型大小: {original_size_gb:.2f} GB")
        logger.info(f"  压缩比: {original_size_gb / quantized_size_gb:.2f}x")
    logger.info(f"  保存路径: {config.output_dir}")
    
    return model, tokenizer


def quantize_with_transformers(config: GPTQQuantConfig):
    """
    使用 transformers 集成的 GPTQ 进行量化
    (备选方案，适用于 transformers >= 4.36.0)
    
    这种方式通过 transformers 的 GPTQConfig 接口调用 auto-gptq，
    使用更加简洁
    """
    from transformers import GPTQConfig
    
    set_seed(config.seed)
    print_device_info()
    
    logger.info(f"使用 transformers 集成方式进行 GPTQ 量化")
    logger.info(f"模型: {config.model_name_or_path}")
    
    # 加载分词器
    tokenizer = run_with_hf_fallback(
        lambda local_files_only: AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            **hf_model_kwargs(local_files_only, cache_dir=config.hf_cache_dir),
        ),
        f"tokenizer {config.model_name_or_path}",
        local_files_only=config.local_files_only,
    )
    
    # 准备校准数据
    calibration_texts = load_calibration_data_for_autogptq(
        tokenizer=tokenizer,
        dataset_name=config.dataset,
        dataset_config=config.dataset_config,
        num_samples=config.num_calibration_samples,
        max_length=config.max_length,
        seed=config.seed,
        local_files_only=config.local_files_only,
        cache_dir=config.hf_cache_dir,
    )
    
    # 配置 GPTQ
    gptq_config = GPTQConfig(
        bits=config.bits,
        group_size=config.group_size,
        damp_percent=config.damp_percent,
        desc_act=config.desc_act,
        sym=config.sym,
        dataset=calibration_texts,  # 直接传入文本列表
        tokenizer=tokenizer,
    )
    
    # 加载并量化模型
    logger.info("加载模型并执行量化...")
    with Timer("模型加载与量化"):
        model = run_with_hf_fallback(
            lambda local_files_only: AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                quantization_config=gptq_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                **hf_model_kwargs(local_files_only, cache_dir=config.hf_cache_dir),
            ),
            f"model {config.model_name_or_path}",
            local_files_only=config.local_files_only,
        )
    
    # 保存
    os.makedirs(config.output_dir, exist_ok=True)
    with Timer("模型保存"):
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
    
    logger.info(f"量化模型已保存到: {config.output_dir}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ W4 权重量化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
统一配置方式:
  GPTQ 量化参数统一在 config.py 修改。
  命令行只保留模型路径、输出目录和运行环境参数。

示例:
  python quantize_gptq.py --model /path/to/Qwen3-1.7B
        """,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="模型名称或路径")
    parser.add_argument("--output_dir", type=str, default=None, help="量化模型输出路径；默认由 config.py 自动推导")
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

    # 兼容旧接口: 这些参数已迁移到 config.py，保留仅用于给出明确提示。
    parser.add_argument("--bits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--group_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--calibration_dataset", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--calibration_dataset_config", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sym", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--desc_act", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--damp_percent", type=float, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    deprecated_args = []
    if args.bits is not None:
        deprecated_args.append("--bits -> config.py / GPTQQuantConfig.bits")
    if args.group_size is not None:
        deprecated_args.append("--group_size -> config.py / GPTQQuantConfig.group_size")
    if args.num_samples is not None:
        deprecated_args.append("--num_samples -> config.py / GPTQQuantConfig.num_calibration_samples")
    if args.calibration_dataset is not None:
        deprecated_args.append("--calibration_dataset -> config.py / GPTQQuantConfig.dataset")
    if args.calibration_dataset_config is not None:
        deprecated_args.append("--calibration_dataset_config -> config.py / GPTQQuantConfig.dataset_config")
    if args.max_length is not None:
        deprecated_args.append("--max_length -> config.py / GPTQQuantConfig.max_length")
    if args.sym:
        deprecated_args.append("--sym -> config.py / GPTQQuantConfig.sym")
    if args.desc_act:
        deprecated_args.append("--desc_act -> config.py / GPTQQuantConfig.desc_act")
    if args.damp_percent is not None:
        deprecated_args.append("--damp_percent -> config.py / GPTQQuantConfig.damp_percent")

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
    config = pipeline_config.gptq

    logger.info("开始 GPTQ W%s 量化", config.bits)
    logger.info("模型: %s", config.model_name_or_path)
    logger.info(
        "统一配置: group_size=%s, calibration_dataset=%s/%s, max_length=%s, weight_quant_scheme=%s",
        config.group_size,
        config.dataset,
        config.dataset_config,
        config.max_length,
        config.weight_quant_scheme,
    )
    backend = args.backend
    if config.weight_quant_scheme == "nf4" and backend != "custom":
        logger.info("检测到 weight_quant_scheme=nf4，自动切换 backend=custom")
        backend = "custom"
    logger.info("后端: %s", backend)

    if backend == "autogptq":
        quantize_with_autogptq(config)
    elif backend == "transformers":
        quantize_with_transformers(config)
    else:
        quantize_with_custom_backend(pipeline_config)

    logger.info("GPTQ 量化完成!")


if __name__ == "__main__":
    main()
