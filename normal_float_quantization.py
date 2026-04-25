"""
NormalFloat / Uniform 量化公共组件
==================================

提供:
1. NF4 / NF8 码本
2. Uniform / NormalFloat 的权重与激活量化
3. SmoothQuant scale 计算
4. INT4 打包 / 解包
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from scipy.stats import norm


NF4_CODEBOOK = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


@dataclass
class WeightQuantArtifacts:
    scheme: str
    bits: int
    group_size: int
    codes: torch.Tensor
    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]
    dequantized: torch.Tensor


@dataclass
class ActivationQuantArtifacts:
    scheme: str
    bits: int
    quantized: torch.Tensor
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor]
    dequantized: torch.Tensor


@lru_cache(maxsize=4)
def _build_nf8_codebook() -> torch.Tensor:
    levels = 256
    probabilities = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
    values = torch.tensor(norm.ppf(probabilities.tolist()), dtype=torch.float32)
    values = values / values.abs().max()
    return values


def get_normal_float_codebook(bits: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if bits == 4:
        codebook = NF4_CODEBOOK
    elif bits == 8:
        codebook = _build_nf8_codebook()
    else:
        raise ValueError(f"仅支持 NF4/NF8，当前 bits={bits}")
    return codebook.to(device=device, dtype=dtype)


def compute_smooth_scale(
    activation_absmax: torch.Tensor,
    weight_absmax: Optional[torch.Tensor],
    alpha: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    act_absmax = activation_absmax.float().clamp(min=eps)
    if weight_absmax is None:
        ref = act_absmax.mean().clamp(min=eps)
        return (act_absmax / ref).pow(alpha).clamp(min=1e-4, max=1e4)
    weight_absmax = weight_absmax.float().to(act_absmax.device).clamp(min=eps)
    scale = act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)
    return scale.clamp(min=1e-4, max=1e4)


def _reshape_weight_groups(weight: torch.Tensor, group_size: int):
    rows, columns = weight.shape
    if group_size <= 0:
        group_size = columns
    num_groups = math.ceil(columns / group_size)
    group_view = weight.new_zeros(rows, num_groups, group_size)
    for idx in range(num_groups):
        start = idx * group_size
        end = min(start + group_size, columns)
        group_view[:, idx, : end - start] = weight[:, start:end]
    return group_view, num_groups


def _midpoints(codebook: torch.Tensor) -> torch.Tensor:
    return (codebook[1:] + codebook[:-1]) / 2


def codebook_lookup(
    normalized: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    boundaries = _midpoints(codebook).to(device=normalized.device, dtype=normalized.dtype)
    clipped = normalized.clamp(min=codebook[0].item(), max=codebook[-1].item())
    codes = torch.bucketize(clipped, boundaries)
    values = codebook.to(device=normalized.device, dtype=normalized.dtype)[codes]
    return codes, values


def _search_nf_scale(
    x: torch.Tensor,
    base_scale: torch.Tensor,
    codebook: torch.Tensor,
    axis: int,
    grid: int = 80,
    min_shrink: float = 0.5,
    max_expand: float = 1.25,
) -> torch.Tensor:
    """Search a codebook scale that minimizes local reconstruction MSE."""
    if grid <= 1:
        return base_scale

    candidates = torch.linspace(
        min_shrink,
        max_expand,
        steps=grid,
        device=x.device,
        dtype=torch.float32,
    )
    best_scale = base_scale.float().clone()
    best_error = torch.full_like(best_scale, float("inf"), dtype=torch.float32)

    for factor in candidates:
        scale = (base_scale.float() * factor).clamp(min=1e-8)
        view_shape = [1] * x.ndim
        view_shape[axis] = scale.numel()
        scale_view = scale.reshape(view_shape)
        _, values = codebook_lookup(x.float() / scale_view, codebook)
        error = (values * scale_view - x.float()).pow(2)
        reduce_dims = tuple(dim for dim in range(error.ndim) if dim != axis)
        error = error.mean(dim=reduce_dims) if reduce_dims else error
        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_scale = torch.where(improve, scale, best_scale)

    return best_scale.clamp(min=1e-8)


def uniform_affine_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.round(x / scale + zero).clamp(0, maxq).to(torch.int16)
    dequant = (codes.float() - zero.float()) * scale.float()
    return codes, dequant


def _reshape_activation_scale(scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.reshape([1] * x.ndim)
    if scale.shape[-1] == x.shape[-1]:
        view_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
        return scale.reshape(view_shape)
    return scale


class UniformAffineQuantizer(nn.Module):
    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits: int,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x: torch.Tensor, weight: bool = False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape

        if self.perchannel:
            x = x.flatten(1) if weight else x.reshape((-1, shape[-1])).t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            negative_mask = xmin < 0
            if torch.any(negative_mask):
                xmin[negative_mask] = -xmax[negative_mask]

        zero_mask = (xmin == 0) & (xmax == 0)
        xmin[zero_mask] = -1
        xmax[zero_mask] = 1

        self.scale = (xmax - xmin).clamp(min=1e-8) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            repeat = shape[0] if weight else shape[-1]
            self.scale = self.scale.repeat(repeat)
            self.zero = self.zero.repeat(repeat)

        if weight:
            self.scale = self.scale.reshape([-1, 1])
            self.zero = self.zero.reshape([-1, 1])

    def quantize(self, x: torch.Tensor, return_codes: bool = False):
        codes, dequant = uniform_affine_quantize(
            x.float(),
            self.scale.float(),
            self.zero.float(),
            int(self.maxq.item()),
        )
        if return_codes:
            return dequant.to(x.dtype), codes
        return dequant.to(x.dtype)

    def ready(self):
        return torch.all(self.scale != 0)


class NormalFloatQuantizer(nn.Module):
    def __init__(self, bits: int = 4, shape=1):
        super().__init__()
        self.bits = bits
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self.register_buffer("maxq", torch.tensor(2**bits - 1))
        self.perchannel = True
        self.mse = True
        self.grid = 80

    def configure(
        self,
        bits: int,
        perchannel: bool = True,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.maxq = torch.tensor(2**bits - 1)
        self.mse = mse
        self.grid = grid

    def find_params(self, x: torch.Tensor, weight: bool = False):
        shape = x.shape
        if self.perchannel:
            x = x.flatten(1) if weight else x.reshape((-1, shape[-1])).t()
        else:
            x = x.flatten().unsqueeze(0)

        self.scale = x.abs().amax(dim=1).clamp(min=1e-8)
        if self.mse:
            codebook = get_normal_float_codebook(self.bits, device=x.device, dtype=torch.float32)
            self.scale = _search_nf_scale(x, self.scale, codebook, axis=0, grid=self.grid)
        self.zero = torch.zeros_like(self.scale)

        if not self.perchannel:
            repeat = shape[0] if weight else shape[-1]
            self.scale = self.scale.repeat(repeat)
            self.zero = self.zero.repeat(repeat)

        if weight:
            self.scale = self.scale.reshape([-1, 1])
            self.zero = self.zero.reshape([-1, 1])

    def quantize(self, x: torch.Tensor, return_codes: bool = False):
        scale = self.scale.float().clamp(min=1e-8)
        normalized = x.float() / scale
        codebook = get_normal_float_codebook(self.bits, device=x.device, dtype=torch.float32)
        codes, values = codebook_lookup(normalized, codebook)
        dequant = values * scale
        if return_codes:
            return dequant.to(x.dtype), codes.to(torch.int16)
        return dequant.to(x.dtype)

    def ready(self):
        return torch.all(self.scale != 0)


def quantize_weight_uniform(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> WeightQuantArtifacts:
    rows, columns = weight.shape
    if group_size <= 0:
        group_size = columns
    num_groups = math.ceil(columns / group_size)
    codes = torch.zeros_like(weight, dtype=torch.int16)
    dequant = torch.zeros_like(weight, dtype=torch.float32)
    scales = torch.zeros(rows, num_groups, dtype=torch.float32, device=weight.device)
    zeros = torch.zeros(rows, num_groups, dtype=torch.float32, device=weight.device)

    for idx in range(num_groups):
        start = idx * group_size
        end = min(start + group_size, columns)
        group = weight[:, start:end].float()

        if symmetric:
            absmax = group.abs().amax(dim=1).clamp(min=1e-8)
            maxq = (1 << bits) - 1
            scale = (2 * absmax).clamp(min=1e-8) / maxq
            zero = torch.full_like(scale, (maxq + 1) / 2)
            group_codes = torch.round(group / scale.unsqueeze(1) + zero.unsqueeze(1)).clamp(0, maxq).to(torch.int16)
            group_dequant = (group_codes.float() - zero.unsqueeze(1)) * scale.unsqueeze(1)
        else:
            xmin = group.amin(dim=1)
            xmax = group.amax(dim=1)
            qmax = (1 << bits) - 1
            scale = (xmax - xmin).clamp(min=1e-8) / qmax
            zero = torch.round(-xmin / scale).clamp(0, qmax)
            group_codes = torch.round(group / scale.unsqueeze(1) + zero.unsqueeze(1)).clamp(0, qmax).to(torch.int16)
            group_dequant = (group_codes.float() - zero.unsqueeze(1)) * scale.unsqueeze(1)

        codes[:, start:end] = group_codes
        dequant[:, start:end] = group_dequant
        scales[:, idx] = scale
        zeros[:, idx] = zero

    return WeightQuantArtifacts(
        scheme="int4",
        bits=bits,
        group_size=group_size,
        codes=codes,
        scales=scales,
        zero_points=zeros,
        dequantized=dequant.to(weight.dtype),
    )


def quantize_weight_nf(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    mse: bool = True,
    grid: int = 80,
) -> WeightQuantArtifacts:
    rows, columns = weight.shape
    if group_size <= 0:
        group_size = columns
    num_groups = math.ceil(columns / group_size)
    codes = torch.zeros_like(weight, dtype=torch.int16)
    dequant = torch.zeros_like(weight, dtype=torch.float32)
    scales = torch.zeros(rows, num_groups, dtype=torch.float32, device=weight.device)
    codebook = get_normal_float_codebook(bits, device=weight.device, dtype=torch.float32)

    for idx in range(num_groups):
        start = idx * group_size
        end = min(start + group_size, columns)
        group = weight[:, start:end].float()
        scale = group.abs().amax(dim=1).clamp(min=1e-8)
        if mse:
            scale = _search_nf_scale(group, scale, codebook, axis=0, grid=grid)
        normalized = group / scale.unsqueeze(1)
        group_codes, values = codebook_lookup(normalized, codebook)
        codes[:, start:end] = group_codes.to(torch.int16)
        dequant[:, start:end] = values * scale.unsqueeze(1)
        scales[:, idx] = scale

    return WeightQuantArtifacts(
        scheme=f"nf{bits}",
        bits=bits,
        group_size=group_size,
        codes=codes,
        scales=scales,
        zero_points=None,
        dequantized=dequant.to(weight.dtype),
    )


def dequantize_weight(
    codes: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scheme: str,
    zero_points: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    rows, columns = codes.shape
    if group_size <= 0:
        group_size = columns
    out = torch.zeros(rows, columns, dtype=torch.float32, device=codes.device)

    for idx in range(math.ceil(columns / group_size)):
        start = idx * group_size
        end = min(start + group_size, columns)
        group_codes = codes[:, start:end].float()
        scale = scales[:, idx].unsqueeze(1).float()

        if scheme == "int4":
            if zero_points is None:
                raise ValueError("INT4 反量化需要 zero_points")
            zero = zero_points[:, idx].unsqueeze(1).float()
            out[:, start:end] = (group_codes - zero) * scale
            continue

        bits = 4 if scheme == "nf4" else 8
        codebook = get_normal_float_codebook(bits, device=codes.device, dtype=torch.float32)
        out[:, start:end] = codebook[group_codes.long()] * scale

    return out.to(dtype)


def pack_int4_codes(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.uint8)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)], dim=0)
    return flat[0::2] | (flat[1::2] << 4)


def unpack_int4_codes(
    packed: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    flat = packed.reshape(-1).to(torch.uint8)
    unpacked = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=flat.device)
    unpacked[0::2] = flat & 0x0F
    unpacked[1::2] = (flat >> 4) & 0x0F
    total = shape[0] * shape[1]
    return unpacked[:total].reshape(shape).to(torch.int16)


def _activation_scale_from_granularity(x: torch.Tensor, granularity: str, symmetric: bool, bits: int):
    x = x.float()
    if granularity in {"per_tensor", "per-tensor"}:
        if symmetric:
            qmax = (1 << (bits - 1)) - 1
            scale = x.abs().amax().clamp(min=1e-8) / qmax
            return scale.reshape(1), None
        qmax = (1 << bits) - 1
        xmin = x.amin()
        xmax = x.amax()
        scale = (xmax - xmin).clamp(min=1e-8) / qmax
        zero = torch.round(-xmin / scale).clamp(0, qmax).reshape(1)
        return scale.reshape(1), zero

    if granularity in {"per_channel", "per-channel"}:
        x_flat = x.reshape(-1, x.shape[-1])
        if symmetric:
            qmax = (1 << (bits - 1)) - 1
            scale = x_flat.abs().amax(dim=0).clamp(min=1e-8) / qmax
            return scale, None
        qmax = (1 << bits) - 1
        xmin = x_flat.amin(dim=0)
        xmax = x_flat.amax(dim=0)
        scale = (xmax - xmin).clamp(min=1e-8) / qmax
        zero = torch.round(-xmin / scale).clamp(0, qmax)
        return scale, zero

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        return scale, None
    qmax = (1 << bits) - 1
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin).clamp(min=1e-8) / qmax
    zero = torch.round(-xmin / scale).clamp(0, qmax)
    return scale, zero


def quantize_activation_uniform(
    x: torch.Tensor,
    bits: int,
    granularity: str,
    symmetric: bool,
) -> ActivationQuantArtifacts:
    scale, zero = _activation_scale_from_granularity(x, granularity, symmetric, bits)
    scale_view = _reshape_activation_scale(scale, x).float()
    if zero is None:
        qmax = (1 << (bits - 1)) - 1
        quantized = torch.round(x.float() / scale_view).clamp(-qmax, qmax).to(torch.int8)
        dequantized = quantized.float() * scale_view
    else:
        qmax = (1 << bits) - 1
        zero_view = _reshape_activation_scale(zero, x).float()
        quantized = torch.round(x.float() / scale_view + zero_view).clamp(0, qmax).to(torch.uint8)
        dequantized = (quantized.float() - zero_view) * scale_view
    return ActivationQuantArtifacts(
        scheme="int8",
        bits=bits,
        quantized=quantized,
        scale=scale.to(x.dtype),
        zero_point=None if zero is None else zero.to(x.dtype),
        dequantized=dequantized.to(x.dtype),
    )


def quantize_activation_nf(
    x: torch.Tensor,
    bits: int,
    granularity: str,
    mse: bool = False,
) -> ActivationQuantArtifacts:
    if bits != 8:
        raise ValueError(f"当前仅支持 NF8 激活量化，收到 bits={bits}")
    codebook = get_normal_float_codebook(bits, device=x.device, dtype=torch.float32)
    scale_axis = None
    if granularity in {"per_tensor", "per-tensor"}:
        scale = x.float().abs().amax().clamp(min=1e-8).reshape(1)
    elif granularity in {"per_channel", "per-channel"}:
        scale = x.float().reshape(-1, x.shape[-1]).abs().amax(dim=0).clamp(min=1e-8)
        scale_axis = x.ndim - 1
    else:
        scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    if mse:
        if scale.numel() == 1:
            scale = _search_nf_scale(x.float().reshape(1, -1), scale.reshape(1), codebook, axis=0)
        elif scale_axis is not None:
            scale = _search_nf_scale(x.float(), scale, codebook, axis=scale_axis)

    scale_view = _reshape_activation_scale(scale, x).float()
    normalized = x.float() / scale_view
    quantized, values = codebook_lookup(normalized, codebook)
    dequantized = values * scale_view

    return ActivationQuantArtifacts(
        scheme="nf8",
        bits=bits,
        quantized=quantized.to(torch.uint8),
        scale=scale.to(x.dtype),
        zero_point=None,
        dequantized=dequantized.to(x.dtype),
    )


def quantize_activation_tensor(
    x: torch.Tensor,
    bits: int,
    scheme: str,
    granularity: str,
    symmetric: bool,
) -> ActivationQuantArtifacts:
    if scheme == "int8":
        return quantize_activation_uniform(
            x=x,
            bits=bits,
            granularity=granularity,
            symmetric=symmetric,
        )
    if scheme == "nf8":
        return quantize_activation_nf(
            x=x,
            bits=bits,
            granularity=granularity,
        )
    raise ValueError(f"未知激活量化方案: {scheme}")
