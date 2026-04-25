"""
工具函数集合
============
包含数据加载、量化辅助、日志等功能
"""

import os
import time
import logging
import random
from typing import Callable, List, Optional, Tuple

import torch
import numpy as np
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def should_use_local_files_only() -> bool:
    """根据环境变量判断是否强制 Hugging Face 离线模式。"""
    return any(
        _env_flag(name)
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "LOCAL_FILES_ONLY",
        )
    )


def is_hf_offline_error(exc: Exception) -> bool:
    text = str(exc)
    offline_markers = [
        "We couldn't connect to 'https://huggingface.co'",
        "couldn't find them in the cached files",
        "Offline mode is enabled",
        "ConnectionError",
        "Network is unreachable",
        "Temporary failure in name resolution",
    ]
    return any(marker in text for marker in offline_markers)


def normalize_hf_cache_dir(cache_dir: Optional[str], target: str) -> Optional[str]:
    """将用户传入的 HF 缓存路径标准化为 model/dataset 对应目录。"""
    if not cache_dir:
        return None

    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    base = os.path.basename(cache_dir.rstrip(os.sep))
    parent = os.path.dirname(cache_dir)

    if target == "model":
        if base == "hub":
            return cache_dir
        if base == "datasets":
            return os.path.join(parent, "hub")
        if os.path.basename(cache_dir) == "huggingface":
            return os.path.join(cache_dir, "hub")
        hub_candidate = os.path.join(cache_dir, "hub")
        if os.path.isdir(hub_candidate):
            return hub_candidate
        return cache_dir

    if target == "dataset":
        if base == "datasets":
            return cache_dir
        if base == "hub":
            return os.path.join(parent, "datasets")
        if os.path.basename(cache_dir) == "huggingface":
            return os.path.join(cache_dir, "datasets")
        datasets_candidate = os.path.join(cache_dir, "datasets")
        if os.path.isdir(datasets_candidate):
            return datasets_candidate
        return cache_dir

    raise ValueError(f"未知缓存目标类型: {target}")


def hf_model_kwargs(
    local_files_only: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> dict:
    """构造 transformers / huggingface_hub 加载参数。"""
    if local_files_only is None:
        local_files_only = should_use_local_files_only()
    kwargs = {"local_files_only": local_files_only}
    normalized_cache_dir = normalize_hf_cache_dir(cache_dir, target="model")
    if normalized_cache_dir:
        kwargs["cache_dir"] = normalized_cache_dir
    return kwargs


def hf_dataset_kwargs(
    local_files_only: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> dict:
    """构造 datasets.load_dataset 的离线参数。"""
    if local_files_only is None:
        local_files_only = should_use_local_files_only()
    kwargs = {}
    normalized_cache_dir = normalize_hf_cache_dir(cache_dir, target="dataset")
    if normalized_cache_dir:
        kwargs["cache_dir"] = normalized_cache_dir
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    return kwargs


def run_with_hf_fallback(
    loader: Callable[[bool], object],
    description: str,
    local_files_only: Optional[bool] = None,
):
    """在线加载失败时自动重试本地缓存模式。"""
    if local_files_only is None:
        local_files_only = should_use_local_files_only()
    try:
        return loader(local_files_only)
    except Exception as exc:
        if local_files_only or not is_hf_offline_error(exc):
            raise
        logger.warning("%s 在线加载失败，重试本地缓存模式: %s", description, exc)
        return loader(True)


def load_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    """统一的数据集加载入口，支持离线缓存回退。"""
    return run_with_hf_fallback(
        lambda local_only: load_dataset(
            dataset_name,
            dataset_config,
            **kwargs,
            **hf_dataset_kwargs(local_only, cache_dir=cache_dir),
        ),
        f"dataset {dataset_name}/{dataset_config}",
        local_files_only=local_files_only,
    )


def resolve_local_text_path(dataset_name: str, dataset_config: Optional[str] = None) -> Optional[str]:
    """解析 local:/path 或直接文件路径。"""
    if dataset_name.startswith("local:"):
        return dataset_name[len("local:"):]
    if dataset_name == "local" and dataset_config:
        return dataset_config
    if os.path.isfile(dataset_name):
        return dataset_name
    return None


def load_local_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def setup_logging(log_level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level.upper()),
    )


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_info() -> dict:
    """获取设备信息"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_version"] = torch.version.cuda
    return info


def print_device_info():
    """打印设备信息"""
    info = get_device_info()
    logger.info("=" * 50)
    logger.info("设备信息:")
    logger.info(f"  CUDA 可用: {info['cuda_available']}")
    if info['cuda_available']:
        logger.info(f"  GPU 数量: {info['device_count']}")
        logger.info(f"  GPU 型号: {info['device_name']}")
        logger.info(f"  显存大小: {info['total_memory_gb']:.1f} GB")
        logger.info(f"  CUDA 版本: {info['cuda_version']}")
    logger.info("=" * 50)


def load_calibration_data(
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    num_samples: int = 128,
    max_length: int = 2048,
    seed: int = 42,
    split: str = "train",
    local_files_only: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    """
    加载 GPTQ 校准数据
    
    Args:
        tokenizer: 分词器
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        num_samples: 样本数量
        max_length: 最大序列长度
        seed: 随机种子
        split: 数据集分割
    
    Returns:
        校准数据列表
    """
    logger.info(f"加载校准数据集: {dataset_name}/{dataset_config}")

    local_path = resolve_local_text_path(dataset_name, dataset_config)
    if local_path is not None:
        logger.info(f"从本地文本加载校准数据: {local_path}")
        all_text = load_local_text_file(local_path)
        encodings = tokenizer(all_text, return_tensors="pt", truncation=False)
        input_ids = encodings["input_ids"][0]
        total_length = len(input_ids)
        logger.info(f"校准文本总 token 数: {total_length}")

        calibration_data = []
        set_seed(seed)
        max_start = max(1, total_length - max_length)
        starts = sorted(random.sample(range(max_start), min(num_samples, max_start)))
        for start in starts:
            segment = input_ids[start : start + max_length]
            if len(segment) < 2:
                continue
            calibration_data.append({"input_ids": segment.unsqueeze(0)})
        logger.info(f"生成校准样本数: {len(calibration_data)}")
        return calibration_data

    dataset = load_hf_dataset(
        dataset_name,
        dataset_config,
        split=split,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )

    # 过滤空文本
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    # 随机采样
    set_seed(seed)
    if len(dataset) > num_samples * 10:
        indices = random.sample(range(len(dataset)), min(num_samples * 10, len(dataset)))
        dataset = dataset.select(indices)
    
    # 拼接文本并分词
    all_text = "\n\n".join(dataset["text"])
    
    encodings = tokenizer(
        all_text,
        return_tensors="pt",
        truncation=False,
    )
    
    input_ids = encodings["input_ids"][0]
    total_length = len(input_ids)
    
    logger.info(f"校准文本总 token 数: {total_length}")
    
    # 切分为固定长度的片段
    calibration_data = []
    for i in range(0, total_length - max_length, max_length):
        if len(calibration_data) >= num_samples:
            break
        segment = input_ids[i : i + max_length]
        calibration_data.append({"input_ids": segment.unsqueeze(0)})
    
    logger.info(f"生成校准样本数: {len(calibration_data)}")
    return calibration_data


def load_calibration_data_for_autogptq(
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    num_samples: int = 128,
    max_length: int = 2048,
    seed: int = 42,
    split: str = "train",
    text_field: str = "text",
    local_files_only: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    加载 auto-gptq 格式的校准数据（返回文本列表）
    
    Args:
        tokenizer: 分词器
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        num_samples: 样本数量
        max_length: 最大序列长度
        seed: 随机种子
        split: 数据集分割
    
    Returns:
        文本列表
    """
    logger.info(f"加载校准数据集 (auto-gptq 格式): {dataset_name}/{dataset_config}")

    local_path = resolve_local_text_path(dataset_name, dataset_config)
    if local_path is not None:
        logger.info(f"从本地文本加载 auto-gptq 校准数据: {local_path}")
        all_text = load_local_text_file(local_path)
        encodings = tokenizer(all_text, return_tensors="pt", truncation=False)
        input_ids = encodings["input_ids"][0]
        set_seed(seed)

        calibration_texts = []
        total_length = len(input_ids)
        max_start = max(1, total_length - max_length)
        starts = sorted(random.sample(range(max_start), min(num_samples, max_start)))
        for start in starts:
            segment_ids = input_ids[start : start + max_length]
            if len(segment_ids) < 2:
                continue
            text = tokenizer.decode(segment_ids, skip_special_tokens=True)
            calibration_texts.append(text)
        logger.info(f"生成校准文本样本数: {len(calibration_texts)}")
        return calibration_texts

    dataset = load_hf_dataset(
        dataset_name,
        dataset_config,
        split=split,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    if text_field not in dataset.column_names:
        raise ValueError(
            f"数据集 {dataset_name}/{dataset_config} 不包含字段 '{text_field}'，"
            f"可用字段: {dataset.column_names}"
        )
    
    # 过滤空文本
    dataset = dataset.filter(
        lambda x: isinstance(x[text_field], str) and len(x[text_field].strip()) > 0
    )
    
    # 拼接所有文本
    all_text = "\n\n".join(dataset[text_field])
    
    # 分词后切分
    encodings = tokenizer(all_text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"][0]
    
    set_seed(seed)
    
    calibration_texts = []
    total_length = len(input_ids)
    
    # 随机选择起始位置
    starts = sorted(random.sample(range(0, max(1, total_length - max_length)), 
                                   min(num_samples, max(1, total_length - max_length))))
    
    for start in starts[:num_samples]:
        segment_ids = input_ids[start : start + max_length]
        text = tokenizer.decode(segment_ids, skip_special_tokens=True)
        calibration_texts.append(text)
    
    logger.info(f"生成校准文本样本数: {len(calibration_texts)}")
    return calibration_texts


def get_model_size_mb(model) -> float:
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def get_model_param_count(model) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def print_model_info(model, model_name: str = "Model"):
    """打印模型信息"""
    param_count = get_model_param_count(model)
    size_mb = get_model_size_mb(model)
    logger.info(f"\n{'=' * 50}")
    logger.info(f"{model_name} 信息:")
    logger.info(f"  参数量: {param_count / 1e6:.1f}M ({param_count / 1e9:.2f}B)")
    logger.info(f"  模型大小: {size_mb:.1f} MB ({size_mb / 1024:.2f} GB)")
    logger.info(f"  数据类型: {next(model.parameters()).dtype}")
    logger.info(f"{'=' * 50}\n")


class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.name:
            logger.info(f"[{self.name}] 开始...")
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            logger.info(f"[{self.name}] 完成，耗时: {self.elapsed:.2f}s")


def print_gpu_memory():
    """打印 GPU 显存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(
                f"GPU {i}: 已分配 {allocated:.2f}GB / "
                f"已保留 {reserved:.2f}GB / 总计 {total:.1f}GB"
            )
