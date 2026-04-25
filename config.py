"""
GPTQ W4A8 量化配置文件
=======================
W4: 4-bit 权重量化 (GPTQ)
A8: 8-bit 激活量化 (Dynamic Per-Token Quantization)
"""

import os

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional, List


# ============ 评估数据集预设 ============
# 格式: {名称: (HF数据集路径, 配置名, 分割, 文本字段名)}
# ★ 带 "local_path" 的预设支持从本地 txt 文件加载
EVAL_DATASET_PRESETS = {
    # --- 经典长文本数据集 ---
    "wikitext2": {
        "path": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "text_field": "text",
        "description": "WikiText-2 (经典小型LM评估集, ~2MB, 长文本)",
    },
    "wikitext103": {
        "path": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "test",
        "text_field": "text",
        "description": "WikiText-103 (大型LM评估集, ~250MB, 长文本)",
    },
    # --- 长文本: 书籍/文章 ---
    "pg19": {
        "path": "emozilla/pg19",
        "config": "default",
        "split": "test",
        "text_field": "text",
        "description": "PG-19 (Project Gutenberg长篇书籍, ~10GB, 需要流式)",
        "streaming": True,
    },
    "booksum": {
        "path": "kmfoda/booksum",
        "config": "default",
        "split": "test",
        "text_field": "chapter",
        "description": "BookSum (书籍章节摘要, 长文本段落)",
    },
    # --- 长文本: Web/新闻 ---
    "c4": {
        "path": "allenai/c4",
        "config": "en",
        "split": "validation",
        "text_field": "text",
        "description": "C4 (Common Crawl, 大规模通用长文本)",
        "streaming": True,
    },
    "pile_val": {
        "path": "monology/pile-uncopyrighted",
        "config": "default",
        "split": "validation",
        "text_field": "text",
        "description": "The Pile (多领域长文本混合, 需要流式)",
        "streaming": True,
    },
    # --- 短文本 (对比用) ---
    "lambada": {
        "path": "EleutherAI/lambada_openai",
        "config": "en",
        "split": "test",
        "text_field": "text",
        "description": "LAMBADA (上下文理解/预测最后一个词, 短文本)",
        "eval_metric": "accuracy",  # 使用 top-1 准确率而非 PPL
    },
    # --- 本地文本文件 ---
    "local": {
        "path": "local",
        "config": "",
        "split": "",
        "text_field": "",
        "local_path": "./eval_data.txt",
        "description": "本地文本文件 (将 eval_data.txt 放在项目目录下)",
    },
}


@dataclass
class GPTQQuantConfig:
    """GPTQ 权重量化配置"""
    
    # 模型路径
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    
    # 量化输出路径
    output_dir: str = "./qwen3_1.7b_w4a8_gptq"
    
    # 权重量化位数
    bits: int = 4

    # 权重量化码本:
    # - "int4": 原始线性 INT4 GPTQ
    # - "nf4":  将 QLoRA 的 NF4 码本引入 GPTQ 的量化器
    #wpp1
    weight_quant_scheme: str = "nf4"
    
    # 分组大小，影响量化精度和模型大小的平衡
    # 128 是常用默认值，越小精度越高但模型越大
    #wpp
    group_size: int = 64
    
    # 是否使用对称量化
    # True: 量化范围 [-max, max]，False: 量化范围 [min, max]
    #wpp
    sym: bool = False
    
    # 阻尼因子，用于 Hessian 矩阵求逆的数值稳定性
    # 建议范围: 0.01 - 0.1
    damp_percent: float = 0.01

    # 自定义 GPTQ block 大小
    #wpp
    blocksize: int = 64
    
    # 是否使用真正的顺序量化（按 Hessian 对角线排序）
    # True 通常能获得更好的精度
    true_sequential: bool = True
    
    # 描述字符串
    desc_act: bool = False
    
    # 校准数据集
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    
    # 校准样本数量（GPTQ 论文建议 128）
    num_calibration_samples: int = 128
    
    # 校准序列最大长度
    max_length: int = 2048
    
    # 设备
    device: str = "cuda:0"
    
    # 随机种子
    seed: int = 42

    # Hugging Face 离线/缓存配置
    local_files_only: bool = False
    hf_cache_dir: Optional[str] = None


@dataclass
class ActivationQuantConfig:
    """激活量化配置 (A8)"""
    
    # 激活量化位数
    act_bits: int = 8

    # 激活量化码本:
    # - "int8": 原始线性 INT8
    # - "nf8":  基于 NF4 思想扩展的 8-bit NormalFloat 码本
    #wpp1
    activation_quant_scheme: str = "nf8"
    
    # 量化方案
    # "dynamic": 动态 per-token 量化（推理时计算 scale/zero_point）
    # "static": 静态量化（需要校准数据确定 scale/zero_point）
    quant_scheme: str = "dynamic"
    
    # 静态量化的校准样本数
    calibration_samples: int = 256
    
    # 是否对称量化
    #wpp
    symmetric: bool = False
    
    # 量化粒度
    # "per_token": 每个 token 单独计算量化参数
    # "per_tensor": 整个张量共享量化参数
    # "per_channel": 每个通道单独计算量化参数
    #wpp
    granularity: str = "per-channel"
    
    # 是否使用 SmoothQuant 平滑技术
    #wpp
    use_smoothquant: bool = True
    smoothquant_alpha: float = 0.85
    
    # SmoothQuant 校准集配置
    smoothquant_dataset: str = "/home/wuping/infra/test7_rtn/Qwen3_GPTQ_nf4/eval_data/wikitext2_train.txt"  # 支持 "local:路径" 或 HF 数据集标识 auto
    smoothquant_dataset_config: str = ""
    smoothquant_split: str = "auto"
    smoothquant_text_field: str = "auto"
    smoothquant_num_samples: int = 128
    smoothquant_max_length: int = 2048
    smoothquant_seed: int = 42
    smoothquant_scales_path: Optional[str] = None

    # 是否单独导出部署所需的激活量化 scale。
    # 当 use_smoothquant=False 且该选项=True 时，仅导出 input/output scale。
    export_activation_scales: bool = True
    activation_scales_path: Optional[str] = None

    # Hugging Face 离线/缓存配置
    local_files_only: bool = False
    hf_cache_dir: Optional[str] = None


@dataclass
class EvalConfig:
    """困惑度评估配置"""
    
    # 评估数据集 (可用预设: wikitext2, wikitext103, ptb, c4, lambada)
    eval_dataset: str = "wikitext"
    eval_dataset_config: str = "wikitext-2-raw-v1"
    eval_split: str = "test"
    eval_text_field: str = "text"
    
    # 评估指标: "perplexity" 或 "accuracy" (LAMBADA 使用 top-1 准确率)
    eval_metric: str = "perplexity"
    
    # 多数据集评估 (为 None 时仅用上面的单数据集配置)
    eval_datasets: Optional[List[str]] = None
    
    # 评估序列长度
    max_length: int = 2048
    
    # 滑动窗口步长
    stride: int = 512
    
    # batch size
    batch_size: int = 1
    
    # 最大评估样本数 (None 表示使用全部)
    max_eval_samples: Optional[int] = None
    
    # 设备
    device: str = "cuda:0"

    # Hugging Face 离线/缓存配置
    local_files_only: bool = False
    hf_cache_dir: Optional[str] = None

#wpp1
@dataclass
class AnalysisConfig:
    """分析与可视化输出配置"""

    enable_vector_demo: bool = False
    enable_layer_distribution: bool = False
    output_subdir: str = "analysis"
    histogram_bins: int = 41


@dataclass
class W4A8Config:
    """W4A8 完整配置"""
    gptq: GPTQQuantConfig = field(default_factory=GPTQQuantConfig)
    activation: ActivationQuantConfig = field(default_factory=ActivationQuantConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def resolve_dataset_spec(dataset_key: str) -> Dict[str, object]:
    """将预设名或 local:/path 解析为完整数据集描述。"""
    local_path = None
    if dataset_key.startswith("local:"):
        local_path = dataset_key[len("local:"):]
        dataset_key = "local"

    if dataset_key not in EVAL_DATASET_PRESETS:
        raise ValueError(
            f"未知数据集预设 '{dataset_key}'，可用: {list(EVAL_DATASET_PRESETS.keys())}\n"
            f"也可使用 'local:/path/to/file.txt' 加载本地文本文件"
        )

    preset = EVAL_DATASET_PRESETS[dataset_key]
    spec = {
        "dataset_key": dataset_key,
        "eval_dataset": preset["path"],
        "eval_dataset_config": preset["config"],
        "eval_split": preset["split"],
        "eval_text_field": preset["text_field"],
        "eval_metric": preset.get("eval_metric", "perplexity"),
        "streaming": preset.get("streaming", False),
    }
    if dataset_key == "local":
        spec["local_path"] = local_path or preset.get("local_path", "./eval_data.txt")
    return spec


def resolve_dataset_source_args(
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    split: Optional[str],
    text_field: Optional[str],
    fallback_dataset_key: Optional[str] = None,
) -> Dict[str, str]:
    """解析 SmoothQuant/GPTQ 等附加数据源参数。"""
    if dataset_name in (None, "", "auto"):
        dataset_name = fallback_dataset_key or "wikitext2"

    if os.path.isfile(dataset_name):
        return {
            "dataset": f"local:{dataset_name}",
            "dataset_config": "",
            "split": "train",
            "text_field": "text",
        }

    if dataset_name in EVAL_DATASET_PRESETS or dataset_name.startswith("local:"):
        spec = resolve_dataset_spec(dataset_name)
        if spec["dataset_key"] == "local":
            return {
                "dataset": f"local:{spec['local_path']}",
                "dataset_config": "",
                "split": "train",
                "text_field": "text",
            }
        return {
            "dataset": spec["eval_dataset"],
            "dataset_config": spec["eval_dataset_config"],
            "split": spec["eval_split"],
            "text_field": spec["eval_text_field"],
        }

    return {
        "dataset": dataset_name,
        "dataset_config": dataset_config or "",
        "split": "train" if split in (None, "", "auto") else split,
        "text_field": "text" if text_field in (None, "", "auto") else text_field,
    }


def resolve_eval_config(dataset_key: str, base_config: EvalConfig) -> EvalConfig:
    """根据预设名称生成对应的 EvalConfig。"""
    spec = resolve_dataset_spec(dataset_key)

    ec = EvalConfig(
        eval_dataset=spec["eval_dataset"],
        eval_dataset_config=spec["eval_dataset_config"],
        eval_split=spec["eval_split"],
        eval_text_field=spec["eval_text_field"],
        eval_metric=spec["eval_metric"],
        max_length=base_config.max_length,
        stride=base_config.stride,
        batch_size=base_config.batch_size,
        max_eval_samples=base_config.max_eval_samples,
        device=base_config.device,
        local_files_only=base_config.local_files_only,
        hf_cache_dir=base_config.hf_cache_dir,
    )

    if spec["dataset_key"] == "local":
        ec._local_path = spec["local_path"]

    return ec


# ============ 预设配置 ============

def get_qwen3_1_7b_config() -> W4A8Config:
    """Qwen3-1.7B W4A8 量化配置"""
    config = W4A8Config()
    config.gptq.model_name_or_path = "Qwen/Qwen3-1.7B"
    config.gptq.output_dir = "./qwen3_1.7b_w4a8_gptq"
    config.gptq.bits = 4
    #wpp
    config.gptq.group_size = 64
    config.gptq.num_calibration_samples = 128
    config.gptq.max_length = 2048
    return config


def get_qwen3_8b_config() -> W4A8Config:
    """Qwen3-8B W4A8 量化配置"""
    config = W4A8Config()
    config.gptq.model_name_or_path = "Qwen/Qwen3-8B"
    config.gptq.output_dir = "./qwen3_8b_w4a8_gptq"
    config.gptq.bits = 4
    config.gptq.group_size = 64
    config.gptq.num_calibration_samples = 128
    config.gptq.max_length = 2048
    return config


def resolve_model_config(model_name_or_path: str) -> W4A8Config:
    """根据模型名返回一份独立的可修改配置。"""
    model_key = model_name_or_path.rstrip("/").split("/")[-1].lower()

    if model_key == "qwen3-1.7b":
        config = deepcopy(get_qwen3_1_7b_config())
    elif model_key == "qwen3-8b":
        config = deepcopy(get_qwen3_8b_config())
    else:
        config = W4A8Config()

    config.gptq.model_name_or_path = model_name_or_path
    return config


def prepare_runtime_config(
    model_name_or_path: str,
    output_dir: Optional[str] = None,
    device: str = "cuda:0",
    seed: int = 42,
    local_files_only: bool = False,
    hf_cache_dir: Optional[str] = None,
) -> W4A8Config:
    """生成本次运行使用的统一配置。"""
    config = resolve_model_config(model_name_or_path)

    config.gptq.model_name_or_path = model_name_or_path
    config.gptq.device = device
    config.gptq.seed = seed
    config.gptq.local_files_only = local_files_only
    config.gptq.hf_cache_dir = hf_cache_dir

    config.activation.smoothquant_seed = seed
    config.activation.local_files_only = local_files_only
    config.activation.hf_cache_dir = hf_cache_dir

    config.evaluation.device = device
    config.evaluation.local_files_only = local_files_only
    config.evaluation.hf_cache_dir = hf_cache_dir

    if output_dir is None:
        model_short = model_name_or_path.rstrip("/").split("/")[-1]
        weight_tag = "" if config.gptq.weight_quant_scheme == "int4" else f"_{config.gptq.weight_quant_scheme}"
        act_tag = "" if config.activation.activation_quant_scheme == "int8" else f"_{config.activation.activation_quant_scheme}"
        config.gptq.output_dir = f"./{model_short}_w{config.gptq.bits}a{config.activation.act_bits}{weight_tag}{act_tag}_gptq"
    else:
        config.gptq.output_dir = output_dir

    return config
