"""
困惑度 (Perplexity) 评估脚本
=============================
评估原始模型、GPTQ W4 模型、W4A8 模型的困惑度

困惑度 (PPL) 是语言模型质量的核心度量:
  PPL = exp(-1/N × Σ log P(x_i | x_{<i}))
其中 N 是总 token 数，P(x_i | x_{<i}) 是模型对第 i 个 token 的预测概率

PPL 越低说明模型越好。量化后 PPL 的上升幅度反映了量化带来的精度损失。

评估方法: 滑动窗口法
- 将长文本按固定长度 (max_length) 的窗口切分
- 使用 stride 参数控制窗口滑动步长
- 只计算每个窗口中"新增"部分的 loss (避免重复计算)
"""

import os
import sys
import math
import logging
import argparse
import json
from typing import Optional, Dict, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

from config import (
    EvalConfig,
    ActivationQuantConfig,
    EVAL_DATASET_PRESETS,
    prepare_runtime_config,
    resolve_eval_config,
    resolve_dataset_source_args,
)
from custom_gptq_backend import is_custom_quantized_model, load_custom_quantized_model
from utils import (
    setup_logging, Timer, print_device_info,
    load_hf_dataset, run_with_hf_fallback, hf_model_kwargs,
)

logger = logging.getLogger(__name__)


def _load_eval_text(config: EvalConfig) -> str:
    """
    根据配置加载评估文本，支持预设数据集、自定义数据集和本地文件。
    
    Returns:
        str: 拼接后的评估文本
    """
    dataset_path = config.eval_dataset
    dataset_config = config.eval_dataset_config
    split = config.eval_split
    text_field = config.eval_text_field
    streaming = False
    
    # ---- 本地文件模式 ----
    if dataset_path == "local":
        local_path = getattr(config, '_local_path', None) or "./eval_data.txt"
        logger.info(f"从本地文件加载评估数据: {local_path}")
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"本地评估文件不存在: {local_path}\n"
                f"请创建该文件或指定其他数据集。"
            )
        with open(local_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"评估文本长度: {len(text)} 字符")
        return text
    
    # ---- HuggingFace 数据集模式 ----
    # 检查是否匹配预设
    for preset_name, preset in EVAL_DATASET_PRESETS.items():
        if (dataset_path == preset["path"] and dataset_config == preset["config"]):
            split = preset["split"]
            text_field = preset["text_field"]
            streaming = preset.get("streaming", False)
            break
    
    logger.info(f"加载评估数据集: {dataset_path}/{dataset_config} (split={split}, field={text_field})")
    
    if streaming:
        # 流式加载 (如 C4, PG-19, Pile)，取前 N 条
        max_samples = config.max_eval_samples or 1000
        dataset = load_hf_dataset(
            dataset_path, dataset_config,
            split=split, streaming=True,
            local_files_only=config.local_files_only,
            cache_dir=config.hf_cache_dir,
        )
        texts = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            t = sample.get(text_field, "")
            if t and t.strip():
                texts.append(t)
        text = "\n\n".join(texts)
    else:
        dataset = load_hf_dataset(
            dataset_path, dataset_config,
            split=split,
            local_files_only=config.local_files_only,
            cache_dir=config.hf_cache_dir,
        )
        text = "\n\n".join(
            s for s in dataset[text_field] if s and s.strip()
        )
    
    logger.info(f"评估文本长度: {len(text)} 字符")
    return text


def evaluate_perplexity(
    model,
    tokenizer,
    config: EvalConfig,
    model_name: str = "Model",
) -> Dict:
    """
    计算语言模型在评估数据集上的困惑度
    
    使用滑动窗口方法:
    - 将数据集所有文本拼接成一个长序列
    - 以 max_length 为窗口大小, stride 为步长滑动
    - 每个窗口只计算新增 token 的 loss
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        config: 评估配置
        model_name: 模型名称 (用于日志)
    
    Returns:
        Dict: 包含 perplexity, loss 等评估结果
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"评估 {model_name} 的困惑度")
    logger.info(f"数据集: {config.eval_dataset}/{config.eval_dataset_config}")
    logger.info(f"序列长度: {config.max_length}, 步长: {config.stride}")
    logger.info(f"{'='*60}")
    
    # 当使用 device_map="auto" 时，model.device 可能错误返回 cpu，
    # 需要从 hf_device_map 或模型参数中获取真实设备
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_dev = next(iter(model.hf_device_map.values()))
        device = torch.device(first_dev)
    else:
        # 从模型参数中获取实际设备
        device = next(model.parameters()).device
    logger.info(f"推理设备: {device}")
    model.eval()
    
    # ======== Step 1: 加载并编码评估数据 ========
    text = _load_eval_text(config)
    
    # 分词
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"]
    
    total_tokens = input_ids.size(1)
    logger.info(f"总 token 数: {total_tokens}")
    
    if config.max_eval_samples:
        total_tokens = min(total_tokens, config.max_eval_samples * config.max_length)
        input_ids = input_ids[:, :total_tokens]
        logger.info(f"截断至 {total_tokens} tokens")
    
    # ======== Step 2: 滑动窗口计算 NLL ========
    max_length = config.max_length
    stride = config.stride
    
    # 计算窗口总数
    num_windows = max(1, (total_tokens - max_length) // stride + 1)
    logger.info(f"滑动窗口数: {num_windows}")
    
    total_nll = 0.0
    total_count = 0
    
    prev_end_pos = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, total_tokens - 1, stride), desc="评估中", unit="window"):
            begin = max(i + stride - max_length, 0)
            end = min(i + stride, total_tokens)
            
            if end - begin <= 1:
                continue
                
            target_begin = max(i, begin)
            
            input_chunk = input_ids[:, begin:end].to(device)
            target_chunk = input_chunk.clone()
            
            # 只计算新增部分的 loss
            # 将已计算过的位置设为 -100 (忽略)
            target_chunk[:, : target_begin - begin] = -100
            
            try:
                outputs = model(
                    input_ids=input_chunk,
                    labels=target_chunk,
                )
                
                # outputs.loss 是平均 NLL (只算非 -100 的 token)
                loss = outputs.loss
                
                # 计算这个窗口中有效 token 数
                num_valid = (target_chunk != -100).sum().item()
                
                if num_valid > 0 and not math.isnan(loss.item()):
                    total_nll += loss.item() * num_valid
                    total_count += num_valid
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"窗口 {i} OOM，跳过...")
                    torch.cuda.empty_cache()
                    continue
                raise
            
            # 避免超出
            if end >= total_tokens:
                break
    
    # ======== Step 3: 计算困惑度 ========
    if total_count == 0:
        logger.error("没有有效的评估 token!")
        return {"perplexity": float("inf"), "loss": float("inf"), "num_tokens": 0}
    
    avg_nll = total_nll / total_count
    perplexity = math.exp(avg_nll)
    
    results = {
        "model_name": model_name,
        "perplexity": round(perplexity, 4),
        "loss": round(avg_nll, 6),
        "num_tokens": total_count,
        "dataset": f"{config.eval_dataset}/{config.eval_dataset_config}",
        "max_length": config.max_length,
        "stride": config.stride,
        "timestamp": datetime.now().isoformat(),
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} 评估结果:")
    logger.info(f"  困惑度 (PPL): {perplexity:.4f}")
    logger.info(f"  平均 NLL Loss: {avg_nll:.6f}")
    logger.info(f"  评估 Token 数: {total_count}")
    logger.info(f"{'='*60}\n")
    
    return results


# ============================================================
# LAMBADA Top-1 准确率评估
# ============================================================

def evaluate_lambada_accuracy(
    model,
    tokenizer,
    config: EvalConfig,
    model_name: str = "Model",
) -> Dict:
    """
    在 LAMBADA 数据集上评估 top-1 准确率
    
    LAMBADA 任务: 给定上下文，预测段落的最后一个词。
    评估方式: 对每个样本，用除最后一个词外的文本作为上下文输入模型，
    检查模型 top-1 预测的 token 是否与真实最后一个词的首 token 一致。
    
    对于子词分词器，最后一个"词"可能被分为多个 token。这里采用
    贪心逐 token 匹配策略：依次预测每个 token，全部匹配才算正确。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        config: 评估配置
        model_name: 模型名称 (用于日志)
    
    Returns:
        Dict: 包含 accuracy, num_correct, num_total 等评估结果
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"评估 {model_name} 的 LAMBADA Top-1 准确率")
    logger.info(f"数据集: {config.eval_dataset}/{config.eval_dataset_config}")
    logger.info(f"{'='*60}")
    
    # 获取设备
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_dev = next(iter(model.hf_device_map.values()))
        device = torch.device(first_dev)
    else:
        device = next(model.parameters()).device
    logger.info(f"推理设备: {device}")
    model.eval()
    
    # ======== Step 1: 加载 LAMBADA 数据集 (逐条) ========
    logger.info("加载 LAMBADA 数据集...")
    dataset = load_hf_dataset(
        config.eval_dataset,
        config.eval_dataset_config,
        split=config.eval_split,
        local_files_only=config.local_files_only,
        cache_dir=config.hf_cache_dir,
    )
    
    total_samples = len(dataset)
    if config.max_eval_samples and config.max_eval_samples < total_samples:
        total_samples = config.max_eval_samples
    logger.info(f"评估样本数: {total_samples}")
    
    # ======== Step 2: 逐样本评估 ========
    num_correct = 0
    num_total = 0
    
    with torch.no_grad():
        for idx in tqdm(range(total_samples), desc="LAMBADA 评估", unit="sample"):
            text = dataset[idx][config.eval_text_field]
            if not text or not text.strip():
                continue
            
            # 将整个文本分词
            full_tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(full_tokens) < 2:
                continue
            
            # 提取最后一个"词": 以空格分割取最后一个词
            text_stripped = text.rstrip()
            last_space_idx = text_stripped.rfind(" ")
            if last_space_idx == -1:
                continue
            
            context_text = text_stripped[:last_space_idx]
            last_word = text_stripped[last_space_idx:]  # 包含前导空格
            
            # 分别编码上下文和最后一个词
            context_ids = tokenizer.encode(context_text, add_special_tokens=False)
            # 编码 "空格+最后一个词"，得到目标 token 序列
            target_ids = tokenizer.encode(last_word, add_special_tokens=False)
            
            if len(context_ids) == 0 or len(target_ids) == 0:
                continue
            
            # 截断上下文使其加上目标不超过 max_length
            max_ctx_len = config.max_length - len(target_ids)
            if max_ctx_len <= 0:
                continue
            context_ids = context_ids[-max_ctx_len:]
            
            # 贪心逐 token 匹配
            input_ids = torch.tensor([context_ids], device=device)
            is_correct = True
            
            for t_idx, target_token in enumerate(target_ids):
                try:
                    outputs = model(input_ids=input_ids)
                    # 取最后一个位置的 logits
                    logits = outputs.logits[:, -1, :]  # (1, vocab_size)
                    predicted_token = logits.argmax(dim=-1).item()
                    
                    if predicted_token != target_token:
                        is_correct = False
                        break
                    
                    # 将真实 token 追加到输入，继续预测下一个
                    next_token = torch.tensor([[target_token]], device=device)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # 防止超长
                    if input_ids.size(1) > config.max_length:
                        input_ids = input_ids[:, -config.max_length:]
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"样本 {idx} OOM，跳过")
                        torch.cuda.empty_cache()
                        is_correct = False
                        break
                    raise
            
            if is_correct:
                num_correct += 1
            num_total += 1
    
    # ======== Step 3: 计算准确率 ========
    if num_total == 0:
        logger.error("没有有效的评估样本!")
        return {"accuracy": 0.0, "num_correct": 0, "num_total": 0}
    
    accuracy = num_correct / num_total
    
    results = {
        "model_name": model_name,
        "accuracy": round(accuracy, 6),
        "accuracy_pct": round(accuracy * 100, 2),
        "num_correct": num_correct,
        "num_total": num_total,
        "dataset": f"{config.eval_dataset}/{config.eval_dataset_config}",
        "eval_metric": "accuracy",
        "max_length": config.max_length,
        "timestamp": datetime.now().isoformat(),
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} LAMBADA 评估结果:")
    logger.info(f"  Top-1 准确率: {accuracy*100:.2f}% ({num_correct}/{num_total})")
    logger.info(f"{'='*60}\n")
    
    return results


# ============================================================
# 评估不同精度的模型
# ============================================================

def evaluate_fp16_model(
    model_path: str,
    eval_config: EvalConfig,
) -> Dict:
    """评估 FP16 原始模型"""
    logger.info(f"加载 FP16 模型: {model_path}")
    
    tokenizer = run_with_hf_fallback(
        lambda local_files_only: AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            **hf_model_kwargs(local_files_only, cache_dir=eval_config.hf_cache_dir),
        ),
        f"tokenizer {model_path}",
        local_files_only=eval_config.local_files_only,
    )
    model = run_with_hf_fallback(
        lambda local_files_only: AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **hf_model_kwargs(local_files_only, cache_dir=eval_config.hf_cache_dir),
        ),
        f"model {model_path}",
        local_files_only=eval_config.local_files_only,
    ).to(eval_config.device)
    
    model_label = f"FP16 ({model_path.split('/')[-1]})"
    if eval_config.eval_metric == "accuracy":
        results = evaluate_lambada_accuracy(
            model, tokenizer, eval_config, model_name=model_label
        )
    else:
        results = evaluate_perplexity(
            model, tokenizer, eval_config, model_name=model_label
        )
    
    # 释放显存
    del model
    torch.cuda.empty_cache()
    
    return results


def evaluate_gptq_w4_model(
    quantized_model_path: str,
    eval_config: EvalConfig,
    use_autogptq: bool = True,
) -> Dict:
    """评估 W4A16 量化模型 (GPTQ W4 + FP16 激活)"""
    logger.info(f"加载 W4A16 模型 (GPTQ W4 + FP16 激活): {quantized_model_path}")
    
    if is_custom_quantized_model(quantized_model_path):
        model, tokenizer, manifest = load_custom_quantized_model(
            quantized_model_path=quantized_model_path,
            device=eval_config.device,
            local_files_only=eval_config.local_files_only,
            hf_cache_dir=eval_config.hf_cache_dir,
        )
        model_label = f"W4A16/{manifest['weight_quant_scheme'].upper()} ({quantized_model_path.split('/')[-1]})"
    else:
        tokenizer = run_with_hf_fallback(
            lambda local_files_only: AutoTokenizer.from_pretrained(
                quantized_model_path,
                trust_remote_code=True,
                **hf_model_kwargs(local_files_only, cache_dir=eval_config.hf_cache_dir),
            ),
            f"tokenizer {quantized_model_path}",
            local_files_only=eval_config.local_files_only,
        )

        if use_autogptq:
            from auto_gptq import AutoGPTQForCausalLM
            from quantize_gptq import _patch_autogptq_for_qwen3
            _patch_autogptq_for_qwen3()
            model = run_with_hf_fallback(
                lambda local_files_only: AutoGPTQForCausalLM.from_quantized(
                    quantized_model_path,
                    device=eval_config.device,
                    trust_remote_code=True,
                    use_safetensors=True,
                    **hf_model_kwargs(local_files_only, cache_dir=eval_config.hf_cache_dir),
                ),
                f"gptq model {quantized_model_path}",
                local_files_only=eval_config.local_files_only,
            )
        else:
            model = run_with_hf_fallback(
                lambda local_files_only: AutoModelForCausalLM.from_pretrained(
                    quantized_model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    **hf_model_kwargs(local_files_only, cache_dir=eval_config.hf_cache_dir),
                ),
                f"model {quantized_model_path}",
                local_files_only=eval_config.local_files_only,
            )
        model_label = f"W4A16/GPTQ-W4 ({quantized_model_path.split('/')[-1]})"

    if eval_config.eval_metric == "accuracy":
        results = evaluate_lambada_accuracy(
            model, tokenizer, eval_config, model_name=model_label
        )
    else:
        results = evaluate_perplexity(
            model, tokenizer, eval_config, model_name=model_label
        )
    
    del model
    torch.cuda.empty_cache()
    
    return results


def evaluate_w4a8_model(
    quantized_model_path: str,
    eval_config: EvalConfig,
    act_config: Optional[ActivationQuantConfig] = None,
) -> Dict:
    """评估 W4A8 量化模型"""
    from w4a8_inference import load_w4a8_model
    
    logger.info(f"加载 W4A8 模型: {quantized_model_path}")
    
    if act_config is None:
        act_config = ActivationQuantConfig()
    
    w4a8_model, tokenizer = load_w4a8_model(
        quantized_model_path,
        act_config=act_config,
        device=eval_config.device,
    )
    
    model_label = (
        f"W4A8/{act_config.activation_quant_scheme.upper()} "
        f"({quantized_model_path.split('/')[-1]})"
    )
    if eval_config.eval_metric == "accuracy":
        results = evaluate_lambada_accuracy(
            w4a8_model, tokenizer, eval_config, model_name=model_label
        )
    else:
        results = evaluate_perplexity(
            w4a8_model, tokenizer, eval_config, model_name=model_label
        )
    
    del w4a8_model
    torch.cuda.empty_cache()
    
    return results


# ============================================================
# 对比评估
# ============================================================

def compare_all_models(
    original_model_path: str,
    quantized_model_path: str,
    eval_config: Optional[EvalConfig] = None,
    act_config: Optional[ActivationQuantConfig] = None,
    skip_fp16: bool = False,
) -> Dict:
    """
    对比评估: FP16 vs W4 vs W4A8
    
    Args:
        original_model_path: 原始模型路径
        quantized_model_path: GPTQ W4 量化模型路径
        eval_config: 评估配置
        act_config: 激活量化配置
        skip_fp16: 是否跳过 FP16 评估 (节省时间/显存)
    
    Returns:
        包含所有模型评估结果的字典
    """
    if eval_config is None:
        eval_config = EvalConfig()
    if act_config is None:
        act_config = ActivationQuantConfig()
    
    all_results = {}
    
    print_device_info()
    
    # 1. FP16 原始模型
    if not skip_fp16:
        logger.info("\n" + "=" * 60)
        logger.info("阶段 1/3: 评估 FP16 原始模型")
        logger.info("=" * 60)
        with Timer("FP16 模型评估"):
            all_results["fp16"] = evaluate_fp16_model(
                original_model_path, eval_config
            )
    
    # 2. GPTQ W4 量化模型
    logger.info("\n" + "=" * 60)
    logger.info("阶段 2/3: 评估 GPTQ W4 量化模型")
    logger.info("=" * 60)
    with Timer("W4A16 模型评估"):
        all_results["gptq_w4"] = evaluate_gptq_w4_model(
            quantized_model_path, eval_config
        )
    
    # 3. W4A8 模型
    logger.info("\n" + "=" * 60)
    logger.info("阶段 3/3: 评估 W4A8 量化模型")
    logger.info("=" * 60)
    with Timer("W4A8 模型评估"):
        all_results["w4a8"] = evaluate_w4a8_model(
            quantized_model_path, eval_config, act_config
        )
    
    # ======== 结果对比 ========
    is_accuracy = eval_config.eval_metric == "accuracy"
    
    logger.info("\n" + "=" * 70)
    if is_accuracy:
        logger.info("  LAMBADA Top-1 准确率对比结果")
        logger.info("=" * 70)
        logger.info(f"{'模型':<30} {'Accuracy':>10} {'正确/总数':>15} {'变化':>10}")
        logger.info("-" * 70)
        
        base_acc = all_results.get("fp16", {}).get("accuracy", None)
        
        for key, result in all_results.items():
            acc = result["accuracy"]
            acc_pct = result["accuracy_pct"]
            count_str = f"{result['num_correct']}/{result['num_total']}"
            acc_change = ""
            if base_acc is not None and key != "fp16" and base_acc > 0:
                change = ((acc - base_acc) / base_acc) * 100
                sign = "+" if change >= 0 else ""
                acc_change = f"{sign}{change:.2f}%"
            
            logger.info(f"{result.get('model_name', key):<30} {acc_pct:>9.2f}% {count_str:>15} {acc_change:>10}")
    else:
        logger.info("  困惑度对比结果")
        logger.info("=" * 70)
        logger.info(f"{'模型':<30} {'PPL':>10} {'Loss':>10} {'PPL 变化':>10}")
        logger.info("-" * 70)
        
        base_ppl = all_results.get("fp16", {}).get("perplexity", None)
        
        for key, result in all_results.items():
            ppl = result["perplexity"]
            loss = result["loss"]
            ppl_change = ""
            if base_ppl is not None and key != "fp16":
                change = ((ppl - base_ppl) / base_ppl) * 100
                ppl_change = f"+{change:.2f}%"
            
            logger.info(f"{result.get('model_name', key):<30} {ppl:>10.4f} {loss:>10.6f} {ppl_change:>10}")
    
    logger.info("=" * 70)
    
    # 保存结果
    results_path = os.path.join(
        os.path.dirname(quantized_model_path) if os.path.isdir(quantized_model_path) 
        else ".",
        "eval_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n评估结果已保存到: {results_path}")
    
    return all_results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="困惑度 (Perplexity) 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
统一配置方式:
  评估与激活量化参数统一在 config.py 修改。
  命令行只保留评估模式、输入路径和运行环境参数。

示例:
  python evaluate_ppl.py --model /path/to/model --quantized_model ./Qwen3-1.7B_w4a8_gptq --eval_mode all
        """,
    )

    parser.add_argument("--model", type=str, required=True, help="原始模型路径 (用于 FP16 评估)")
    parser.add_argument("--quantized_model", type=str, default=None, help="GPTQ W4 量化模型路径；不传则使用 config.py 推导默认输出目录")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="all",
        choices=["fp16", "w4", "w4a8", "all"],
        help="评估模式: fp16/w4/w4a8/all",
    )
    parser.add_argument(
        "--eval_datasets",
        type=str,
        nargs="+",
        default=None,
        help=f"评估数据集列表，可用预设: {list(EVAL_DATASET_PRESETS.keys())}；不传则使用 config.py 默认值",
    )
    parser.add_argument("--output", type=str, default="eval_results.json", help="结果保存路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--local_files_only", action="store_true", help="仅从本地缓存加载 Hugging Face 模型/数据集")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="Hugging Face 缓存目录")

    # 兼容旧接口: 这些参数已迁移到 config.py，保留仅用于给出明确提示。
    parser.add_argument("--max_length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stride", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_eval_samples", type=int, default=None, help=argparse.SUPPRESS)
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

    args = parser.parse_args()

    deprecated_args = []
    if args.max_length is not None:
        deprecated_args.append("--max_length -> config.py / EvalConfig.max_length")
    if args.stride is not None:
        deprecated_args.append("--stride -> config.py / EvalConfig.stride")
    if args.max_eval_samples is not None:
        deprecated_args.append("--max_eval_samples -> config.py / EvalConfig.max_eval_samples")
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
        device=args.device,
        seed=args.seed,
        local_files_only=args.local_files_only,
        hf_cache_dir=args.hf_cache_dir,
    )
    eval_config = deepcopy(pipeline_config.evaluation)
    act_config = deepcopy(pipeline_config.activation)

    dataset_keys = args.eval_datasets or eval_config.eval_datasets or ["wikitext2"]
    quantized_path = args.quantized_model or pipeline_config.gptq.output_dir

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

    logger.info(
        "统一配置: eval_max_length=%s, stride=%s, act_granularity=%s, smoothquant=%s",
        eval_config.max_length,
        eval_config.stride,
        act_config.granularity,
        act_config.use_smoothquant,
    )
    if act_config.use_smoothquant:
        logger.info(
            "SmoothQuant 校准数据源: %s/%s (split=%s, field=%s)",
            act_config.smoothquant_dataset,
            act_config.smoothquant_dataset_config,
            act_config.smoothquant_split,
            act_config.smoothquant_text_field,
        )

    all_dataset_results = {}

    for ds_key in dataset_keys:
        logger.info(f"\n{'#' * 70}")
        logger.info(f"  数据集: {ds_key}")
        if ds_key in EVAL_DATASET_PRESETS:
            logger.info(f"  {EVAL_DATASET_PRESETS[ds_key]['description']}")
        logger.info(f"{'#' * 70}")

        try:
            ds_eval_config = resolve_eval_config(ds_key, eval_config)
        except ValueError:
            logger.warning(f"未知数据集预设 '{ds_key}'，跳过")
            continue

        results = {}

        if args.eval_mode == "fp16":
            results["fp16"] = evaluate_fp16_model(args.model, ds_eval_config)
        elif args.eval_mode == "w4":
            results["gptq_w4"] = evaluate_gptq_w4_model(quantized_path, ds_eval_config)
        elif args.eval_mode == "w4a8":
            results["w4a8"] = evaluate_w4a8_model(quantized_path, ds_eval_config, act_config)
        elif args.eval_mode == "all":
            results = compare_all_models(
                original_model_path=args.model,
                quantized_model_path=quantized_path,
                eval_config=ds_eval_config,
                act_config=act_config,
            )

        all_dataset_results[ds_key] = results

    if len(dataset_keys) > 1:
        logger.info(f"\n{'=' * 70}")
        logger.info("  多数据集评估汇总")
        logger.info(f"{'=' * 70}")
        logger.info(f"{'数据集':<15} {'模型':<25} {'指标':>12} {'值':>10}")
        logger.info("-" * 70)
        for ds_key, ds_results in all_dataset_results.items():
            for model_key, result in ds_results.items():
                if not isinstance(result, dict):
                    continue
                name = result.get("model_name", model_key)
                if "accuracy" in result:
                    logger.info(f"  {ds_key:<13} {name:<23} {'Accuracy':>12} {result['accuracy_pct']:>9.2f}%")
                elif "perplexity" in result:
                    logger.info(f"  {ds_key:<13} {name:<23} {'PPL':>12} {result['perplexity']:>10.4f}")
        logger.info(f"{'=' * 70}")

    with open(args.output, "w") as f:
        json.dump(all_dataset_results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
