"""
自定义 GPTQ 后端
================

目标:
1. 复用 GPTQ 的 Hessian/OBS 量化流程
2. 在量化器层面支持 Uniform INT4 与 NF4
3. 保存为自定义压缩格式，加载时恢复为反量化权重
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from config import W4A8Config
from normal_float_quantization import (
    NormalFloatQuantizer,
    UniformAffineQuantizer,
    compute_smooth_scale,
    dequantize_weight,
    pack_int4_codes,
    quantize_activation_tensor,
    quantize_weight_nf,
    quantize_weight_uniform,
    unpack_int4_codes,
)
from quantization_analysis import (
    build_distribution_record,
    export_distribution_reports,
)
from utils import (
    Timer,
    hf_model_kwargs,
    load_calibration_data_for_autogptq,
    print_gpu_memory,
    run_with_hf_fallback,
    set_seed,
)

logger = logging.getLogger(__name__)


MANIFEST_NAME = "custom_quant_manifest.json"
NON_QUANT_STATE_NAME = "non_quantized_state.safetensors"
PACKED_WEIGHTS_NAME = "quantized_weights.pt"
TARGET_MODULE_GROUPS = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]


@dataclass
class CalibrationAccumulator:
    in_features: int
    device: torch.device
    max_sample_rows: int = 64

    def __post_init__(self):
        self.H = torch.zeros((self.in_features, self.in_features), device=self.device)
        self.nsamples = 0
        self.input_absmax = None
        self._sample_rows: List[torch.Tensor] = []

    def add_batch(self, inp: torch.Tensor):
        original = inp.detach().float()
        if original.ndim == 2:
            original = original.unsqueeze(0)
        flat_rows = original.reshape(-1, original.shape[-1])
        if self.input_absmax is None:
            self.input_absmax = flat_rows.abs().amax(dim=0)
        else:
            self.input_absmax = torch.maximum(self.input_absmax, flat_rows.abs().amax(dim=0))

        if sum(row.shape[0] for row in self._sample_rows) < self.max_sample_rows:
            remain = self.max_sample_rows - sum(row.shape[0] for row in self._sample_rows)
            self._sample_rows.append(flat_rows[:remain].cpu())

        gptq_inp = inp
        if len(gptq_inp.shape) == 2:
            gptq_inp = gptq_inp.unsqueeze(0)
        batch_size = gptq_inp.shape[0]
        if len(gptq_inp.shape) == 3:
            gptq_inp = gptq_inp.reshape((-1, gptq_inp.shape[-1]))
        gptq_inp = gptq_inp.t()

        self.H *= self.nsamples / (self.nsamples + batch_size) if self.nsamples > 0 else 0
        self.nsamples += batch_size
        gptq_inp = math.sqrt(2 / self.nsamples) * gptq_inp.float()
        self.H += gptq_inp.matmul(gptq_inp.t())

    def sample_matrix(self) -> Optional[torch.Tensor]:
        if not self._sample_rows:
            return None
        return torch.cat(self._sample_rows, dim=0)

    def free(self):
        self.H = None
        self._sample_rows = []
        self.input_absmax = None


def _get_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("当前自定义 GPTQ 后端仅支持具有 model.layers 结构的 Qwen3 类模型")


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    current = model
    for part in name.split("."):
        current = getattr(current, part)
    return current


def _prepare_examples(tokenizer, texts: List[str], max_length: int) -> List[Dict[str, torch.Tensor]]:
    examples = []
    for text in texts:
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        if "attention_mask" not in tokenized:
            tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"])
        examples.append(tokenized)
    return examples


def _capture_first_layer_inputs(
    model: nn.Module,
    examples: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], List[Dict[str, object]]]:
    layers = _get_layers(model)
    layer_inputs: List[torch.Tensor] = []
    attention_masks: List[Optional[torch.Tensor]] = []
    position_ids: List[Optional[torch.Tensor]] = []
    extra_kwargs: List[Dict[str, object]] = []
    original_layer = layers[0]

    class LayerHijacker(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, inp=None, **kwargs):
            if inp is None:
                inp = kwargs.get("hidden_states")
            layer_inputs.append(inp.detach().cpu())
            attention_masks.append(
                kwargs["attention_mask"].detach().cpu() if kwargs.get("attention_mask") is not None else None
            )
            pos = kwargs.get("position_ids")
            position_ids.append(pos.detach().cpu() if pos is not None else None)
            one_kwargs = {}
            for key, value in kwargs.items():
                if key in {"hidden_states", "attention_mask", "position_ids"}:
                    continue
                if torch.is_tensor(value):
                    one_kwargs[key] = value.detach().cpu()
                else:
                    one_kwargs[key] = value
            extra_kwargs.append(one_kwargs)
            raise ValueError("layer_hijack")

    layers[0] = LayerHijacker(original_layer)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    try:
        with torch.no_grad():
            for example in examples:
                moved = {k: v.to(device) for k, v in example.items()}
                try:
                    model(**moved)
                except ValueError as exc:
                    if str(exc) != "layer_hijack":
                        raise
    finally:
        layers[0] = original_layer
        model.config.use_cache = use_cache

    return layer_inputs, attention_masks, position_ids, extra_kwargs


def _solve_gptq(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    quantizer: nn.Module,
    group_size: int,
    percdamp: float,
    actorder: bool,
    blocksize: int = 128,
) -> Tuple[torch.Tensor, float]:
    W = weight.detach().clone().float()
    H = hessian.detach().clone().float()
    rows, columns = W.shape

    if not quantizer.ready():
        quantizer.find_params(W, weight=True)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        invperm = torch.argsort(perm)
        W = W[:, perm]
        H = H[perm][:, perm]
    else:
        invperm = None

    losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=W.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    if group_size <= 0:
        group_size = columns

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            col_idx = i1 + i
            if col_idx % group_size == 0:
                quantizer.find_params(W[:, col_idx : min(col_idx + group_size, columns)], weight=True)

            w = W1[:, i]
            d = Hinv1[i, i]
            q = quantizer.quantize(w.unsqueeze(1)).flatten()

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        losses[:, i1:i2] = Losses1 / 2
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if invperm is not None:
        Q = Q[:, invperm]

    avg_loss = torch.sum(losses).item() / max(1, rows * columns)
    return Q.to(weight.dtype), avg_loss


def _build_activation_record(sample_inputs, uniform_weight, act_config):
    if act_config.use_smoothquant:
        weight_absmax = uniform_weight.abs().amax(dim=0)
        smooth_scale = compute_smooth_scale(
            activation_absmax=sample_inputs.abs().amax(dim=0),
            weight_absmax=weight_absmax,
            alpha=act_config.smoothquant_alpha,
        )
        smoothed = sample_inputs / smooth_scale.unsqueeze(0)
    else:
        smooth_scale = torch.ones(sample_inputs.shape[-1], dtype=sample_inputs.dtype)
        smoothed = sample_inputs
    int8_artifact = quantize_activation_tensor(
        x=smoothed,
        bits=act_config.act_bits,
        scheme="int8",
        granularity=act_config.granularity,
        symmetric=act_config.symmetric,
    )
    nf8_artifact = quantize_activation_tensor(
        x=smoothed,
        bits=act_config.act_bits,
        scheme="nf8",
        granularity=act_config.granularity,
        symmetric=True,
    )
    restored_int8 = int8_artifact.dequantized * smooth_scale.unsqueeze(0)
    restored_nf8 = nf8_artifact.dequantized * smooth_scale.unsqueeze(0)
    return restored_int8, restored_nf8


def _save_custom_quantized_model(
    model: nn.Module,
    tokenizer,
    pipeline_config: W4A8Config,
    quantized_module_names: List[str],
):
    output_dir = pipeline_config.gptq.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model.config.save_pretrained(output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    full_state = model.state_dict()
    quantized_weight_keys = {f"{name}.weight" for name in quantized_module_names}
    non_quant_state = {
        key: value.detach().cpu().clone().contiguous()
        for key, value in full_state.items()
        if key not in quantized_weight_keys
    }
    save_file(non_quant_state, os.path.join(output_dir, NON_QUANT_STATE_NAME))

    packed_payload = {}
    for module_name in quantized_module_names:
        module = _get_module_by_name(model, module_name)
        weight = module.weight.detach().cpu().float()
        if pipeline_config.gptq.weight_quant_scheme == "nf4":
            artifacts = quantize_weight_nf(
                weight=weight,
                bits=pipeline_config.gptq.bits,
                group_size=pipeline_config.gptq.group_size,
                mse=True,
            )
        else:
            artifacts = quantize_weight_uniform(
                weight=weight,
                bits=pipeline_config.gptq.bits,
                group_size=pipeline_config.gptq.group_size,
                symmetric=pipeline_config.gptq.sym,
            )
        packed_payload[module_name] = {
            "scheme": artifacts.scheme,
            "shape": list(weight.shape),
            "group_size": artifacts.group_size,
            "packed_codes": pack_int4_codes(artifacts.codes.cpu()),
            "scales": artifacts.scales.cpu().to(torch.float16),
            "zero_points": None if artifacts.zero_points is None else artifacts.zero_points.cpu().to(torch.float16),
        }

    torch.save(packed_payload, os.path.join(output_dir, PACKED_WEIGHTS_NAME))

    manifest = {
        "format": "custom_gptq_codebook_v1",
        "base_model_name_or_path": pipeline_config.gptq.model_name_or_path,
        "weight_quant_scheme": pipeline_config.gptq.weight_quant_scheme,
        "group_size": pipeline_config.gptq.group_size,
        "bits": pipeline_config.gptq.bits,
        "act_quant_scheme": pipeline_config.activation.activation_quant_scheme,
        "act_bits": pipeline_config.activation.act_bits,
        "quantized_modules": quantized_module_names,
        "non_quant_state": NON_QUANT_STATE_NAME,
        "packed_weights": PACKED_WEIGHTS_NAME,
    }
    with open(os.path.join(output_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def is_custom_quantized_model(path: str) -> bool:
    return os.path.exists(os.path.join(path, MANIFEST_NAME))


def load_custom_quantized_model(
    quantized_model_path: str,
    device: str = "cuda:0",
    local_files_only: bool = False,
    hf_cache_dir: Optional[str] = None,
):
    manifest_path = os.path.join(quantized_model_path, MANIFEST_NAME)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tokenizer = run_with_hf_fallback(
        lambda local_only: AutoTokenizer.from_pretrained(
            quantized_model_path,
            trust_remote_code=True,
            **hf_model_kwargs(local_only, cache_dir=hf_cache_dir),
        ),
        f"tokenizer {quantized_model_path}",
        local_files_only=local_files_only,
    )

    config = AutoConfig.from_pretrained(
        quantized_model_path,
        trust_remote_code=True,
        **hf_model_kwargs(local_files_only, cache_dir=hf_cache_dir),
    )
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    non_quant_state = load_file(os.path.join(quantized_model_path, manifest["non_quant_state"]))
    missing, unexpected = model.load_state_dict(non_quant_state, strict=False)
    if unexpected:
        logger.warning("加载自定义量化模型时发现未使用参数: %s", unexpected)
    quantized_expected = {f"{name}.weight" for name in manifest["quantized_modules"]}
    unresolved = [name for name in missing if name not in quantized_expected]
    if unresolved:
        logger.warning("加载自定义量化模型时仍有缺失参数: %s", unresolved)

    packed_payload = torch.load(os.path.join(quantized_model_path, manifest["packed_weights"]), map_location="cpu")
    for module_name, item in packed_payload.items():
        module = _get_module_by_name(model, module_name)
        shape = tuple(item["shape"])
        codes = unpack_int4_codes(item["packed_codes"], shape)
        weight = dequantize_weight(
            codes=codes,
            scales=item["scales"].float(),
            zero_points=None if item["zero_points"] is None else item["zero_points"].float(),
            group_size=int(item["group_size"]),
            scheme=item["scheme"],
            dtype=module.weight.dtype,
        )
        module.weight.data.copy_(weight)

    model.to(device)
    model.eval()
    return model, tokenizer, manifest


def quantize_with_custom_backend(pipeline_config: W4A8Config):
    set_seed(pipeline_config.gptq.seed)
    device = torch.device(pipeline_config.gptq.device)

    tokenizer = run_with_hf_fallback(
        lambda local_only: AutoTokenizer.from_pretrained(
            pipeline_config.gptq.model_name_or_path,
            trust_remote_code=True,
            **hf_model_kwargs(local_only, cache_dir=pipeline_config.gptq.hf_cache_dir),
        ),
        f"tokenizer {pipeline_config.gptq.model_name_or_path}",
        local_files_only=pipeline_config.gptq.local_files_only,
    )
    model = run_with_hf_fallback(
        lambda local_only: AutoModelForCausalLM.from_pretrained(
            pipeline_config.gptq.model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **hf_model_kwargs(local_only, cache_dir=pipeline_config.gptq.hf_cache_dir),
        ),
        f"model {pipeline_config.gptq.model_name_or_path}",
        local_files_only=pipeline_config.gptq.local_files_only,
    ).to(device)
    model.eval()

    calibration_texts = load_calibration_data_for_autogptq(
        tokenizer=tokenizer,
        dataset_name=pipeline_config.gptq.dataset,
        dataset_config=pipeline_config.gptq.dataset_config,
        num_samples=pipeline_config.gptq.num_calibration_samples,
        max_length=pipeline_config.gptq.max_length,
        seed=pipeline_config.gptq.seed,
        local_files_only=pipeline_config.gptq.local_files_only,
        cache_dir=pipeline_config.gptq.hf_cache_dir,
    )
    examples = _prepare_examples(tokenizer, calibration_texts, pipeline_config.gptq.max_length)
    logger.info("自定义 GPTQ 校准样本数: %d", len(examples))

    with Timer("捕获第一层输入"):
        layer_inputs, attention_masks, position_ids, extra_kwargs = _capture_first_layer_inputs(
            model=model,
            examples=examples,
            device=device,
        )

    layers = _get_layers(model)
    analysis_records: Dict[str, object] = {}
    quantized_module_names: List[str] = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            logger.info("Start custom quantizing layer %d/%d", layer_idx + 1, len(layers))
            named_modules = dict(layer.named_modules())

            for group in TARGET_MODULE_GROUPS:
                subset = {name: named_modules[name] for name in group if name in named_modules}
                if not subset:
                    continue

                accumulators = {
                    name: CalibrationAccumulator(module.weight.shape[1], device=device)
                    for name, module in subset.items()
                }

                handles = []
                for name, module in subset.items():
                    handles.append(
                        module.register_forward_hook(
                            lambda _, inp, out, name=name: accumulators[name].add_batch(inp[0].data)
                        )
                    )

                for idx in range(len(layer_inputs)):
                    kwargs = {"attention_mask": None if attention_masks[idx] is None else attention_masks[idx].to(device)}
                    if position_ids[idx] is not None:
                        kwargs["position_ids"] = position_ids[idx].to(device)
                    for key, value in extra_kwargs[idx].items():
                        kwargs[key] = value.to(device) if torch.is_tensor(value) else value
                    layer(layer_inputs[idx].to(device), **kwargs)

                for handle in handles:
                    handle.remove()

                for name, module in subset.items():
                    full_name = f"model.layers.{layer_idx}.{name}"
                    original_weight = module.weight.data.detach().float()
                    need_uniform = (
                        pipeline_config.gptq.weight_quant_scheme == "int4"
                        or pipeline_config.analysis.enable_layer_distribution
                    )
                    need_nf = (
                        pipeline_config.gptq.weight_quant_scheme == "nf4"
                        or pipeline_config.analysis.enable_layer_distribution
                    )

                    uniform_weight = None
                    nf_weight = None
                    uniform_loss = None
                    nf_loss = None

                    if need_uniform:
                        uniform_quantizer = UniformAffineQuantizer()
                        uniform_quantizer.configure(
                            bits=pipeline_config.gptq.bits,
                            perchannel=True,
                            sym=pipeline_config.gptq.sym,
                            mse=False,
                        )
                        uniform_weight, uniform_loss = _solve_gptq(
                            weight=original_weight,
                            hessian=accumulators[name].H,
                            quantizer=uniform_quantizer,
                            group_size=pipeline_config.gptq.group_size,
                            percdamp=pipeline_config.gptq.damp_percent,
                            actorder=pipeline_config.gptq.desc_act,
                        )

                    if need_nf:
                        nf_quantizer = NormalFloatQuantizer(bits=pipeline_config.gptq.bits)
                        nf_quantizer.configure(bits=pipeline_config.gptq.bits, perchannel=True, mse=True)
                        nf_weight, nf_loss = _solve_gptq(
                            weight=original_weight,
                            hessian=accumulators[name].H,
                            quantizer=nf_quantizer,
                            group_size=pipeline_config.gptq.group_size,
                            percdamp=pipeline_config.gptq.damp_percent,
                            actorder=pipeline_config.gptq.desc_act,
                        )

                    chosen = nf_weight if pipeline_config.gptq.weight_quant_scheme == "nf4" else uniform_weight
                    module.weight.data.copy_(chosen.to(module.weight.dtype))
                    quantized_module_names.append(full_name)

                    if pipeline_config.analysis.enable_layer_distribution:
                        sample_inputs = accumulators[name].sample_matrix()
                        if sample_inputs is not None:
                            restored_int8, restored_nf8 = _build_activation_record(
                                sample_inputs=sample_inputs,
                                uniform_weight=original_weight,
                                act_config=pipeline_config.activation,
                            )
                            analysis_records[full_name] = {
                                "weight": build_distribution_record(
                                    original=original_weight,
                                    baseline=uniform_weight,
                                    nf_variant=nf_weight,
                                    baseline_name="gptq_int4",
                                    nf_name="gptq_nf4",
                                    bins=pipeline_config.analysis.histogram_bins,
                                ),
                                "activation": build_distribution_record(
                                    original=sample_inputs,
                                    baseline=restored_int8,
                                    nf_variant=restored_nf8,
                                    baseline_name="smooth_int8",
                                    nf_name="smooth_nf8",
                                    bins=pipeline_config.analysis.histogram_bins,
                                ),
                                "quant_loss": {
                                    "gptq_int4_avg_loss": None if uniform_loss is None else round(uniform_loss, 8),
                                    "gptq_nf4_avg_loss": None if nf_loss is None else round(nf_loss, 8),
                                },
                            }

                    accumulators[name].free()

            next_inputs = []
            for idx in range(len(layer_inputs)):
                kwargs = {"attention_mask": None if attention_masks[idx] is None else attention_masks[idx].to(device)}
                if position_ids[idx] is not None:
                    kwargs["position_ids"] = position_ids[idx].to(device)
                for key, value in extra_kwargs[idx].items():
                    kwargs[key] = value.to(device) if torch.is_tensor(value) else value
                output = layer(layer_inputs[idx].to(device), **kwargs)[0]
                next_inputs.append(output.detach().cpu())
            layer_inputs = next_inputs
            print_gpu_memory()

    with Timer("保存自定义量化模型"):
        _save_custom_quantized_model(
            model=model.cpu(),
            tokenizer=tokenizer,
            pipeline_config=pipeline_config,
            quantized_module_names=quantized_module_names,
        )

    if pipeline_config.analysis.enable_layer_distribution and analysis_records:
        export_distribution_reports(
            records=analysis_records,
            output_dir=os.path.join(pipeline_config.gptq.output_dir, pipeline_config.analysis.output_subdir, "layers"),
        )

    return model, tokenizer
