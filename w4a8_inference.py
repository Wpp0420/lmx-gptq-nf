"""
W4A8 推理引擎
=============
在 GPTQ W4 量化模型基础上，添加 A8 (8-bit) 激活量化

W4A8 混合精度方案:
- W4: 权重使用 GPTQ 4-bit 量化 (离线，一次性)
- A8: 激活值使用 INT8 动态量化 (在线，推理时)

激活量化原理:
1. Per-Token 动态量化:
   - 对每个 token 的激活向量，实时计算 scale = max(|x|) / 127
   - x_int8 = round(x / scale), 量化到 [-128, 127]
   - 反量化: x_dequant = x_int8 * scale
   
2. 优势:
   - 减少激活值的内存占用 (FP16 → INT8, 2x)
   - 可利用 INT8 Tensor Core 加速矩阵乘法
   - 动态量化无需校准数据
"""

import os
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import ActivationQuantConfig
from custom_gptq_backend import is_custom_quantized_model, load_custom_quantized_model
from normal_float_quantization import compute_smooth_scale, quantize_activation_tensor
from utils import load_calibration_data_for_autogptq, run_with_hf_fallback, hf_model_kwargs

logger = logging.getLogger(__name__)


# ============================================================
# INT8 激活量化核心函数
# ============================================================

def quantize_activation_per_token(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Per-token 动态 INT8 量化
    
    对输入张量 x 的每个 token 独立计算量化参数
    
    Args:
        x: 输入张量, shape (..., hidden_size)
        bits: 量化位数 (默认: 8)
        symmetric: 是否对称量化
    
    Returns:
        x_quant: 量化后的整数张量
        scale: 缩放因子, shape (..., 1)
        zero_point: 零点 (对称量化时为 None)
    
    数学公式 (对称量化):
        qmax = 2^(bits-1) - 1 = 127
        scale = max(|x|, dim=-1) / qmax
        x_quant = round(clamp(x / scale, -qmax, qmax))
    """
    orig_dtype = x.dtype
    x = x.float()
    
    if symmetric:
        qmax = (1 << (bits - 1)) - 1  # 127 for 8-bit
        
        # 计算每个 token 的缩放因子
        # shape: (..., 1)
        abs_max = x.abs().amax(dim=-1, keepdim=True)
        # 避免除零
        abs_max = abs_max.clamp(min=1e-8)
        scale = abs_max / qmax
        
        # 量化
        x_quant = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
        
        return x_quant, scale.to(orig_dtype), None
    else:
        qmin = 0
        qmax = (1 << bits) - 1  # 255 for 8-bit
        
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        
        scale = (x_max - x_min).clamp(min=1e-8) / qmax
        zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)
        
        x_quant = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)
        
        return x_quant, scale.to(orig_dtype), zero_point.to(orig_dtype)


def quantize_activation_per_tensor(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Per-tensor 动态 INT8 量化
    
    整个张量共享一个 scale/zero_point
    
    Args:
        x: 输入张量, shape (..., hidden_size)
        bits: 量化位数 (默认: 8)
        symmetric: 是否对称量化
    
    Returns:
        x_quant: 量化后的整数张量
        scale: 缩放因子, scalar (broadcast-able)
        zero_point: 零点 (对称量化时为 None)
    """
    orig_dtype = x.dtype
    x = x.float()
    
    if symmetric:
        qmax = (1 << (bits - 1)) - 1  # 127 for 8-bit
        
        # 整个张量共享一个 scale
        abs_max = x.abs().amax().clamp(min=1e-8)
        scale = abs_max / qmax
        
        x_quant = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
        
        # scale 保持可广播的形状
        scale = scale.reshape(1).to(orig_dtype)
        return x_quant, scale, None
    else:
        qmin = 0
        qmax = (1 << bits) - 1  # 255 for 8-bit
        
        x_min = x.amin()
        x_max = x.amax()
        
        scale = (x_max - x_min).clamp(min=1e-8) / qmax
        zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)
        
        x_quant = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)
        
        scale = scale.reshape(1).to(orig_dtype)
        zero_point = zero_point.reshape(1).to(orig_dtype)
        return x_quant, scale, zero_point


def quantize_activation_per_channel(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Per-channel 动态 INT8 量化
    
    对最后一个维度 (hidden_size / channel) 的每个通道独立计算量化参数。
    即沿除最后一维外的所有维度取 abs_max。
    
    Args:
        x: 输入张量, shape (..., hidden_size)
        bits: 量化位数 (默认: 8)
        symmetric: 是否对称量化
    
    Returns:
        x_quant: 量化后的整数张量
        scale: 缩放因子, shape (1, ..., 1, hidden_size)
        zero_point: 零点 (对称量化时为 None)
    """
    orig_dtype = x.dtype
    x = x.float()
    
    # 将 x 展平为 (N, hidden_size)，沿 N 维度取统计量
    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_flat = x.reshape(-1, hidden_size)  # (N, hidden_size)
    
    if symmetric:
        qmax = (1 << (bits - 1)) - 1  # 127 for 8-bit
        
        # 每个 channel 独立计算 scale: shape (hidden_size,)
        abs_max = x_flat.abs().amax(dim=0).clamp(min=1e-8)
        scale = abs_max / qmax
        
        x_quant = torch.round(x_flat / scale.unsqueeze(0)).clamp(-qmax, qmax).to(torch.int8)
        x_quant = x_quant.reshape(orig_shape)
        
        # scale 调整为可广播的形状: (1, ..., 1, hidden_size)
        scale_shape = [1] * (len(orig_shape) - 1) + [hidden_size]
        scale = scale.reshape(scale_shape).to(orig_dtype)
        return x_quant, scale, None
    else:
        qmin = 0
        qmax = (1 << bits) - 1  # 255 for 8-bit
        
        x_min = x_flat.amin(dim=0)
        x_max = x_flat.amax(dim=0)
        
        scale = (x_max - x_min).clamp(min=1e-8) / qmax
        zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)
        
        x_quant = torch.round(x_flat / scale.unsqueeze(0) + zero_point.unsqueeze(0)).clamp(qmin, qmax).to(torch.uint8)
        x_quant = x_quant.reshape(orig_shape)
        
        scale_shape = [1] * (len(orig_shape) - 1) + [hidden_size]
        scale = scale.reshape(scale_shape).to(orig_dtype)
        zero_point = zero_point.reshape(scale_shape).to(orig_dtype)
        return x_quant, scale, zero_point


def dequantize_activation(
    x_quant: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    反量化: 将 INT8 值恢复为浮点数
    
    Args:
        x_quant: 量化的整数张量
        scale: 缩放因子
        zero_point: 零点 (对称量化时为 None)
    
    Returns:
        反量化的浮点张量
    """
    if zero_point is None:
        # 对称量化: x = x_quant * scale
        return x_quant.float() * scale.float()
    else:
        # 非对称量化: x = (x_quant - zero_point) * scale
        return (x_quant.float() - zero_point.float()) * scale.float()


# ============================================================
# 激活量化包装层
# ============================================================

class ActivationQuantWrapper(nn.Module):
    """
    激活量化包装器
    
    在线性层前后添加激活量化/反量化操作
    模拟 W4A8 的推理过程
    
    推理流程:
    1. 输入激活 x (FP16) → INT8 量化 → INT8_x
    2. 权重 W 已经是 GPTQ INT4 → 反量化为 FP16
    3. 计算: output = dequant(INT8_x) × W_dequant  
       (实际硬件可用 INT8×INT4 混合精度计算)
    4. 输出激活 → INT8 量化（传给下一层）
    """
    
    def __init__(
        self,
        module: nn.Module,
        act_config: ActivationQuantConfig,
    ):
        super().__init__()
        self.module = module
        self.act_bits = act_config.act_bits
        self.symmetric = act_config.symmetric
        self.granularity = act_config.granularity
        self.activation_quant_scheme = act_config.activation_quant_scheme
        self.use_smoothquant = act_config.use_smoothquant
        self.smoothquant_alpha = act_config.smoothquant_alpha
        
        # 用于静态量化的统计信息
        self.register_buffer("running_scale", None)
        self.register_buffer("running_zero_point", None)
        self.register_buffer("smooth_scale", None)
        self.register_buffer("input_quant_scale", None)
        self.register_buffer("output_scale", None)
        self.register_buffer("input_absmax", None)
        self.register_buffer("weight_absmax", None)
        self.calibrating = False
        self.smoothquant_calibrating = False
        self.scale_export_calibrating = False
        self._calibration_scales = []
        self._input_scale_sum = None
        self._input_scale_count = 0
        self._output_scale_sum = None
        self._output_scale_count = 0
        
        self._init_weight_statistics()
    
    def _init_weight_statistics(self):
        """尽可能从底层线性层提取输入通道的权重幅值统计。"""
        weight = getattr(self.module, "weight", None)
        if weight is None:
            return
        
        if isinstance(weight, nn.Parameter):
            weight = weight.detach()
        if not torch.is_tensor(weight) or weight.ndim != 2:
            return
        
        # 线性层权重通常是 [out_features, in_features]，SmoothQuant 需要按输入通道统计。
        self.weight_absmax = weight.float().abs().amax(dim=0)
    
    def _update_input_absmax(self, x: torch.Tensor):
        """记录该层输入激活在 hidden 维上的最大绝对值。"""
        x_flat = x.detach().float().abs().reshape(-1, x.shape[-1])
        current_absmax = x_flat.amax(dim=0)
        if self.input_absmax is None:
            self.input_absmax = current_absmax
        else:
            self.input_absmax = torch.maximum(self.input_absmax, current_absmax)
    
    def _reshape_smooth_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.smooth_scale
        if scale is None:
            raise RuntimeError("SmoothQuant scale 尚未初始化")
        view_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
        return scale.to(device=x.device, dtype=torch.float32).view(view_shape)

    def _quantize_activation(self, x: torch.Tensor):
        """根据 scheme + granularity 执行激活量化。"""
        return quantize_activation_tensor(
            x=x,
            bits=self.act_bits,
            scheme=self.activation_quant_scheme,
            granularity=self.granularity,
            symmetric=self.symmetric,
        )

    def _reduce_scale_for_export(self, scale: torch.Tensor) -> torch.Tensor:
        """
        将运行时 scale 聚合成可部署导出的形式。
        - per-tensor / per-token: 导出单个标量
        - per-channel: 导出最后一维的 channel scale
        """
        scale = scale.detach().float().cpu()
        if scale.numel() == 1:
            return scale.reshape(1)
        if scale.shape[-1] == 1:
            return scale.mean().reshape(1)
        return scale.reshape(-1, scale.shape[-1]).mean(dim=0)

    def _accumulate_export_scale(self, kind: str, scale: torch.Tensor):
        reduced = self._reduce_scale_for_export(scale)
        if kind == "input":
            if self._input_scale_sum is None:
                self._input_scale_sum = reduced
            else:
                self._input_scale_sum += reduced
            self._input_scale_count += 1
            return
        if kind == "output":
            if self._output_scale_sum is None:
                self._output_scale_sum = reduced
            else:
                self._output_scale_sum += reduced
            self._output_scale_count += 1
            return
        raise ValueError(f"未知 scale 类型: {kind}")
    
    def _compute_smooth_scale(self):
        """根据校准统计生成每个输入通道的 SmoothQuant 缩放因子。"""
        if self.input_absmax is None:
            return

        self.smooth_scale = compute_smooth_scale(
            activation_absmax=self.input_absmax,
            weight_absmax=self.weight_absmax,
            alpha=self.smoothquant_alpha,
        )
    
    def set_smoothquant_calibration(self, enabled: bool):
        self.smoothquant_calibrating = enabled
        if enabled:
            self.input_absmax = None

    def set_scale_export_calibration(self, enabled: bool):
        self.scale_export_calibrating = enabled
        if enabled:
            self._input_scale_sum = None
            self._input_scale_count = 0
            self._output_scale_sum = None
            self._output_scale_count = 0
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播:
        1. 量化输入激活 (FP16 → INT8)
        2. 反量化回 FP16 (模拟量化噪声)
        3. 通过原始模块
        """
        if self.smoothquant_calibrating:
            self._update_input_absmax(x)
            return self.module(x, **kwargs)
        
        smooth_scale = None
        x_to_quant = x
        if self.use_smoothquant and self.smooth_scale is not None:
            smooth_scale = self._reshape_smooth_scale(x)
            x_to_quant = x.float() / smooth_scale
        
        quant_artifact = self._quantize_activation(x_to_quant)
        x_quant = quant_artifact.quantized
        scale = quant_artifact.scale
        zero_point = quant_artifact.zero_point
        x_dequant = quant_artifact.dequantized
        if smooth_scale is not None:
            x_dequant = x_dequant * smooth_scale
        x_dequant = x_dequant.to(x.dtype)
        
        # 收集校准统计（静态量化模式）
        if self.calibrating:
            self._calibration_scales.append(scale.detach().cpu())
        if self.scale_export_calibrating:
            self._accumulate_export_scale("input", scale)
        
        output = self.module(x_dequant, **kwargs)

        if self.scale_export_calibrating:
            out_artifact = self._quantize_activation(output)
            self._accumulate_export_scale("output", out_artifact.scale)

        return output
    
    def finalize_calibration(self):
        """完成校准，计算静态量化参数"""
        if self._calibration_scales:
            all_scales = torch.cat(self._calibration_scales, dim=0)
            self.running_scale = all_scales.mean(dim=0)
            self._calibration_scales = []
        self.calibrating = False
        if self.use_smoothquant:
            self._compute_smooth_scale()
            self.smoothquant_calibrating = False

    def finalize_scale_export(self):
        """完成 input/output scale 统计，保存可导出的量化 scale。"""
        if self._input_scale_sum is not None and self._input_scale_count > 0:
            self.input_quant_scale = self._input_scale_sum / self._input_scale_count
            # 保留原 running_scale 字段，兼容旧逻辑。
            self.running_scale = self.input_quant_scale.clone()
        if self._output_scale_sum is not None and self._output_scale_count > 0:
            self.output_scale = self._output_scale_sum / self._output_scale_count
        self.scale_export_calibrating = False


class W4A8ModelWrapper(nn.Module):
    """
    W4A8 模型包装器
    
    在 GPTQ W4 量化模型的 Linear 层前添加 A8 激活量化
    """
    
    def __init__(self, model: nn.Module, act_config: ActivationQuantConfig):
        super().__init__()
        self.model = model
        self.act_config = act_config
        self.quant_wrappers: Dict[str, ActivationQuantWrapper] = {}
        
    def apply_activation_quantization(self):
        """
        遍历模型，对关键线性层添加激活量化
        
        量化目标层:
        - q_proj, k_proj, v_proj: 注意力机制的 QKV 投影
        - o_proj: 注意力输出投影
        - gate_proj, up_proj, down_proj: FFN 层
        """
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        
        wrapped_count = 0
        
        for name, module in self.model.named_modules():
            # 检查是否为目标模块
            module_name = name.split(".")[-1]
            if module_name in target_modules and isinstance(module, nn.Module):
                # 获取父模块
                parent = self.model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                # 创建量化包装
                wrapper = ActivationQuantWrapper(module, self.act_config)
                setattr(parent, parts[-1], wrapper)
                self.quant_wrappers[name] = wrapper
                wrapped_count += 1
        
        logger.info(f"已对 {wrapped_count} 个线性层添加 A8 激活量化")
        return wrapped_count
    
    def load_smoothquant_scales(self, scales: Dict[str, torch.Tensor]) -> int:
        loaded = 0
        for name, wrapper in self.quant_wrappers.items():
            smooth_scale = scales.get(f"{name}.smooth_scale")
            # 兼容旧格式：直接以层名存 smooth_scale
            if smooth_scale is None:
                smooth_scale = scales.get(name)
            input_scale = scales.get(f"{name}.input_scale")
            output_scale = scales.get(f"{name}.output_scale")

            layer_loaded = False
            if smooth_scale is not None:
                wrapper.smooth_scale = smooth_scale.detach().float()
                layer_loaded = True
            if input_scale is not None:
                wrapper.input_quant_scale = input_scale.detach().float()
                wrapper.running_scale = wrapper.input_quant_scale.clone()
                layer_loaded = True
            if output_scale is not None:
                wrapper.output_scale = output_scale.detach().float()
                layer_loaded = True

            if layer_loaded:
                loaded += 1
        return loaded
    
    def export_smoothquant_scales(self) -> Dict[str, torch.Tensor]:
        exported = {}
        for name, wrapper in self.quant_wrappers.items():
            if wrapper.smooth_scale is not None:
                exported[f"{name}.smooth_scale"] = wrapper.smooth_scale.detach().cpu()
            if wrapper.input_quant_scale is not None:
                exported[f"{name}.input_scale"] = wrapper.input_quant_scale.detach().cpu()
            if wrapper.output_scale is not None:
                exported[f"{name}.output_scale"] = wrapper.output_scale.detach().cpu()
        return exported

    def should_export_activation_scales(self) -> bool:
        return self.act_config.use_smoothquant or self.act_config.export_activation_scales

    def _activation_quant_tag(self) -> str:
        granularity = self.act_config.granularity.replace("_", "-")
        symmetry = "sym" if self.act_config.symmetric else "asym"
        return f"{self.act_config.activation_quant_scheme}_a{self.act_config.act_bits}_{granularity}_{symmetry}"

    def resolve_activation_scales_path(self, quantized_model_path: str) -> str:
        if self.act_config.activation_scales_path:
            return self.act_config.activation_scales_path
        if self.act_config.smoothquant_scales_path:
            return self.act_config.smoothquant_scales_path
        if self.act_config.use_smoothquant:
            return os.path.join(
                quantized_model_path,
                f"smoothquant_scales_{self._activation_quant_tag()}_alpha{self.act_config.smoothquant_alpha:.2f}.pt",
            )
        return os.path.join(
            quantized_model_path,
            f"activation_scales_{self._activation_quant_tag()}.pt",
        )
    
    def calibrate_activation_scales(self, tokenizer, device: str):
        if not self.should_export_activation_scales():
            return
        
        if self.act_config.use_smoothquant:
            logger.info(
                "开始 SmoothQuant 校准: dataset=%s/%s, samples=%d, alpha=%.2f",
                self.act_config.smoothquant_dataset,
                self.act_config.smoothquant_dataset_config,
                self.act_config.smoothquant_num_samples,
                self.act_config.smoothquant_alpha,
            )
        else:
            logger.info(
                "开始激活 scale 导出校准: dataset=%s/%s, samples=%d",
                self.act_config.smoothquant_dataset,
                self.act_config.smoothquant_dataset_config,
                self.act_config.smoothquant_num_samples,
            )
        
        calibration_texts = load_calibration_data_for_autogptq(
            tokenizer=tokenizer,
            dataset_name=self.act_config.smoothquant_dataset,
            dataset_config=self.act_config.smoothquant_dataset_config,
            num_samples=self.act_config.smoothquant_num_samples,
            max_length=self.act_config.smoothquant_max_length,
            seed=self.act_config.smoothquant_seed,
            split=self.act_config.smoothquant_split,
            text_field=self.act_config.smoothquant_text_field,
            local_files_only=self.act_config.local_files_only,
            cache_dir=self.act_config.hf_cache_dir,
        )
        
        if self.act_config.use_smoothquant:
            for wrapper in self.quant_wrappers.values():
                wrapper.set_smoothquant_calibration(True)
            
            self.model.eval()
            with torch.no_grad():
                for text in calibration_texts:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        max_length=self.act_config.smoothquant_max_length,
                        truncation=True,
                        padding=False,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    self.model(**inputs)
            
            for wrapper in self.quant_wrappers.values():
                wrapper.finalize_calibration()

        # 第二遍校准收集部署时需要的静态量化 scale。
        for wrapper in self.quant_wrappers.values():
            wrapper.set_scale_export_calibration(True)

        with torch.no_grad():
            for text in calibration_texts:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.act_config.smoothquant_max_length,
                    truncation=True,
                    padding=False,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                self.model(**inputs)

        for wrapper in self.quant_wrappers.values():
            wrapper.finalize_scale_export()
        
        available_weight_stats = sum(
            1 for wrapper in self.quant_wrappers.values() if wrapper.weight_absmax is not None
        )
        if self.act_config.use_smoothquant:
            logger.info(
                "SmoothQuant 校准完成: %d/%d 层使用了权重统计，其余层退化为激活平滑",
                available_weight_stats,
                len(self.quant_wrappers),
            )
            logger.info("已同时导出 smooth_scale / input_scale / output_scale")
        else:
            logger.info(
                "激活 scale 校准完成: 已为 %d 个量化层导出 input_scale / output_scale",
                len(self.quant_wrappers),
            )

    def calibrate_smoothquant(self, tokenizer, device: str):
        """兼容旧调用名，统一走新的激活 scale 校准导出逻辑。"""
        self.calibrate_activation_scales(tokenizer=tokenizer, device=device)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


# ============================================================
# 加载 W4A8 模型
# ============================================================

def load_w4a8_model(
    quantized_model_path: str,
    act_config: Optional[ActivationQuantConfig] = None,
    device: str = "cuda:0",
):
    """
    加载 GPTQ W4 量化模型并添加 A8 激活量化
    
    Args:
        quantized_model_path: GPTQ 量化模型路径
        act_config: 激活量化配置
        device: 设备
    
    Returns:
        w4a8_model: W4A8 模型
        tokenizer: 分词器
    """

    if act_config is None:
        act_config = ActivationQuantConfig()
    
    logger.info(f"加载 GPTQ W4 量化模型: {quantized_model_path}")

    if is_custom_quantized_model(quantized_model_path):
        model, tokenizer, manifest = load_custom_quantized_model(
            quantized_model_path=quantized_model_path,
            device=device,
            local_files_only=act_config.local_files_only,
            hf_cache_dir=act_config.hf_cache_dir,
        )
        logger.info("检测到自定义量化模型格式: %s", manifest["weight_quant_scheme"])
    else:
        from auto_gptq.modeling._utils import SUPPORTED_MODELS
        from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP
        from auto_gptq.modeling.qwen2 import Qwen2GPTQForCausalLM
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer

        if "qwen3" not in SUPPORTED_MODELS:
            SUPPORTED_MODELS.append("qwen3")
            logger.info("已将 qwen3 添加到 auto_gptq SUPPORTED_MODELS")

        if "qwen3" not in GPTQ_CAUSAL_LM_MODEL_MAP:
            GPTQ_CAUSAL_LM_MODEL_MAP["qwen3"] = Qwen2GPTQForCausalLM
            logger.info("已将 qwen3 映射到 Qwen2GPTQForCausalLM")

        tokenizer = run_with_hf_fallback(
            lambda local_files_only: AutoTokenizer.from_pretrained(
                quantized_model_path,
                trust_remote_code=True,
                **hf_model_kwargs(local_files_only, cache_dir=act_config.hf_cache_dir),
            ),
            f"tokenizer {quantized_model_path}",
            local_files_only=act_config.local_files_only,
        )

        model = run_with_hf_fallback(
            lambda local_files_only: AutoGPTQForCausalLM.from_quantized(
                quantized_model_path,
                device=device,
                trust_remote_code=True,
                use_safetensors=True,
                **hf_model_kwargs(local_files_only, cache_dir=act_config.hf_cache_dir),
            ),
            f"gptq model {quantized_model_path}",
            local_files_only=act_config.local_files_only,
        )
    
    logger.info("添加 A8 激活量化层...")
    w4a8_wrapper = W4A8ModelWrapper(model, act_config)
    num_wrapped = w4a8_wrapper.apply_activation_quantization()
    
    if w4a8_wrapper.should_export_activation_scales():
        scales_path = w4a8_wrapper.resolve_activation_scales_path(quantized_model_path)
        if os.path.exists(scales_path):
            logger.info(f"加载激活量化 scales: {scales_path}")
            scales = torch.load(scales_path, map_location="cpu")
            loaded = w4a8_wrapper.load_smoothquant_scales(scales)
            logger.info(f"已加载 {loaded} 个层的激活量化 scales")
        else:
            w4a8_wrapper.calibrate_activation_scales(tokenizer=tokenizer, device=device)
            torch.save(w4a8_wrapper.export_smoothquant_scales(), scales_path)
            logger.info(f"激活量化 scales 已保存到: {scales_path}")
    
    logger.info(f"W4A8 模型加载完成")
    logger.info(f"  权重量化: W4 ({'NF4-GPTQ' if is_custom_quantized_model(quantized_model_path) else 'GPTQ INT4'})")
    logger.info(
        f"  激活量化: {act_config.activation_quant_scheme.upper()} "
        f"(A{act_config.act_bits}, {act_config.quant_scheme}, {act_config.granularity})"
    )
    if act_config.use_smoothquant:
        logger.info(f"  SmoothQuant: enabled (alpha={act_config.smoothquant_alpha:.2f})")
    if act_config.export_activation_scales:
        logger.info("  Activation scale export: enabled")
    logger.info(f"  量化层数: {num_wrapped}")
    
    return w4a8_wrapper, tokenizer


def load_w4a8_model_transformers(
    quantized_model_path: str,
    act_config: Optional[ActivationQuantConfig] = None,
    device: str = "cuda:0",
):
    """
    使用 transformers 加载 GPTQ W4 模型并添加 A8 激活量化
    (备选方案)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if act_config is None:
        act_config = ActivationQuantConfig()
    
    logger.info(f"加载 GPTQ W4 模型 (transformers): {quantized_model_path}")
    
    tokenizer = run_with_hf_fallback(
        lambda local_files_only: AutoTokenizer.from_pretrained(
            quantized_model_path,
            trust_remote_code=True,
            **hf_model_kwargs(local_files_only, cache_dir=act_config.hf_cache_dir),
        ),
        f"tokenizer {quantized_model_path}",
        local_files_only=act_config.local_files_only,
    )
    
    model = run_with_hf_fallback(
        lambda local_files_only: AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            device_map="auto",
            trust_remote_code=True,
            **hf_model_kwargs(local_files_only, cache_dir=act_config.hf_cache_dir),
        ),
        f"model {quantized_model_path}",
        local_files_only=act_config.local_files_only,
    )
    
    logger.info("添加 A8 激活量化层...")
    w4a8_wrapper = W4A8ModelWrapper(model, act_config)
    num_wrapped = w4a8_wrapper.apply_activation_quantization()
    
    if w4a8_wrapper.should_export_activation_scales():
        scales_path = w4a8_wrapper.resolve_activation_scales_path(quantized_model_path)
        if os.path.exists(scales_path):
            scales = torch.load(scales_path, map_location="cpu")
            w4a8_wrapper.load_smoothquant_scales(scales)
        else:
            w4a8_wrapper.calibrate_activation_scales(tokenizer=tokenizer, device=device)
            torch.save(w4a8_wrapper.export_smoothquant_scales(), scales_path)
    
    logger.info(f"W4A8 模型加载完成 (量化层数: {num_wrapped})")
    return w4a8_wrapper, tokenizer


# ============================================================
# 测试
# ============================================================

def test_quantization_functions():
    """测试量化函数的正确性"""
    logger.info("测试量化函数...")
    
    # 创建测试数据
    x = torch.randn(2, 4, 128)  # batch=2, seq_len=4, hidden=128
    
    # 对称量化
    x_q, scale, zp = quantize_activation_per_token(x, bits=8, symmetric=True)
    x_deq = dequantize_activation(x_q, scale, zp)
    
    error = (x - x_deq).abs().mean()
    max_error = (x - x_deq).abs().max()
    
    logger.info(f"对称 INT8 量化测试:")
    logger.info(f"  输入范围: [{x.min():.4f}, {x.max():.4f}]")
    logger.info(f"  量化值范围: [{x_q.min()}, {x_q.max()}]")
    logger.info(f"  平均量化误差: {error:.6f}")
    logger.info(f"  最大量化误差: {max_error:.6f}")
    logger.info(f"  量化数据类型: {x_q.dtype}")
    
    # 非对称量化
    x_q2, scale2, zp2 = quantize_activation_per_token(x, bits=8, symmetric=False)
    x_deq2 = dequantize_activation(x_q2, scale2, zp2)
    
    error2 = (x - x_deq2).abs().mean()
    logger.info(f"\n非对称 INT8 量化测试:")
    logger.info(f"  平均量化误差: {error2:.6f}")
    logger.info(f"  量化数据类型: {x_q2.dtype}")
    
    logger.info("\n量化函数测试通过 ✓")


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    test_quantization_functions()
