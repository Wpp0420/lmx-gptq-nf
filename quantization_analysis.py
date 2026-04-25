"""
量化分析与可视化
================

输出:
1. 简单向量的 INT4/NF4、INT8/NF8 对比
2. 每层/每个 block 的权重与激活分布统计
3. SVG 报告
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional

import torch

from normal_float_quantization import (
    compute_smooth_scale,
    quantize_activation_tensor,
    quantize_weight_nf,
    quantize_weight_uniform,
)


def _default_weight_vector() -> torch.Tensor:
    return torch.tensor(
        [
            -2.30,
            -1.75,
            -1.21,
            -0.83,
            -0.51,
            -0.29,
            -0.12,
            -0.03,
            0.07,
            0.18,
            0.33,
            0.52,
            0.78,
            1.08,
            1.47,
            2.05,
        ],
        dtype=torch.float32,
    )


def _default_activation_vectors() -> torch.Tensor:
    return torch.tensor(
        [
            [-1.80, -1.20, -0.92, -0.61, -0.38, -0.22, -0.10, -0.01, 0.05, 0.14, 0.29, 0.44, 0.66, 0.98, 1.35, 1.82],
            [-1.35, -0.95, -0.72, -0.49, -0.31, -0.18, -0.07, 0.02, 0.10, 0.19, 0.34, 0.51, 0.70, 0.91, 1.18, 1.54],
            [-2.05, -1.42, -1.06, -0.74, -0.49, -0.31, -0.15, -0.04, 0.02, 0.16, 0.31, 0.49, 0.73, 1.06, 1.44, 1.95],
        ],
        dtype=torch.float32,
    )


def _safe_float_list(values: torch.Tensor) -> List[float]:
    return [round(v, 6) for v in values.detach().float().cpu().reshape(-1).tolist()]


def _safe_int_list(values: torch.Tensor) -> List[int]:
    return [int(v) for v in values.detach().cpu().reshape(-1).tolist()]


def sample_rows(x: torch.Tensor, max_rows: int = 64) -> torch.Tensor:
    flat = x.detach().float().reshape(-1, x.shape[-1]).cpu()
    return flat[:max_rows]


def summarize_histogram(values: torch.Tensor, bins: int = 41) -> Dict[str, object]:
    flat = values.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "hist": [],
            "edges": [],
        }

    max_abs = max(flat.abs().max().item(), 1e-6)
    hist = torch.histc(flat, bins=bins, min=-max_abs, max=max_abs)
    edges = torch.linspace(-max_abs, max_abs, bins + 1)
    return {
        "count": int(flat.numel()),
        "mean": round(flat.mean().item(), 6),
        "std": round(flat.std(unbiased=False).item(), 6),
        "min": round(flat.min().item(), 6),
        "max": round(flat.max().item(), 6),
        "hist": [round(v, 6) for v in hist.tolist()],
        "edges": [round(v, 6) for v in edges.tolist()],
    }


def summarize_error(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    ref = reference.detach().float().reshape(-1).cpu()
    cand = candidate.detach().float().reshape(-1).cpu()
    delta = ref - cand
    denom = ref.norm().item() * cand.norm().item()
    cosine = float(torch.dot(ref, cand).item() / denom) if denom > 0 else 1.0
    return {
        "mae": round(delta.abs().mean().item(), 8),
        "mse": round(delta.pow(2).mean().item(), 8),
        "max_abs": round(delta.abs().max().item(), 8),
        "cosine": round(cosine, 8),
    }


def build_distribution_record(
    original: torch.Tensor,
    baseline: torch.Tensor,
    nf_variant: torch.Tensor,
    baseline_name: str,
    nf_name: str,
    bins: int,
) -> Dict[str, object]:
    return {
        "original": summarize_histogram(original, bins=bins),
        baseline_name: {
            "distribution": summarize_histogram(baseline, bins=bins),
            "error": summarize_error(original, baseline),
        },
        nf_name: {
            "distribution": summarize_histogram(nf_variant, bins=bins),
            "error": summarize_error(original, nf_variant),
        },
    }


def _polyline_points(hist: Iterable[float], x0: float, y0: float, width: float, height: float) -> str:
    values = list(hist)
    if not values or max(values) <= 0:
        return ""
    max_value = max(values)
    step = width / max(1, len(values) - 1)
    points = []
    for idx, value in enumerate(values):
        x = x0 + idx * step
        y = y0 + height - (value / max_value) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _render_panel(title: str, series: List[Dict[str, object]], x0: float, y0: float, width: float, height: float) -> str:
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="white" stroke="#d0d7de"/>',
        f'<text x="{x0 + 8}" y="{y0 + 18}" font-size="14" font-family="monospace">{title}</text>',
    ]
    plot_y = y0 + 30
    plot_h = height - 42
    for item in series:
        points = _polyline_points(item["hist"], x0 + 8, plot_y, width - 16, plot_h)
        if not points:
            continue
        parts.append(
            f'<polyline fill="none" stroke="{item["color"]}" stroke-width="2" points="{points}"/>'
        )
    legend_y = y0 + height - 10
    legend_x = x0 + 8
    for item in series:
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 8}" width="10" height="10" fill="{item["color"]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y}" font-size="11" font-family="monospace">{item["name"]}</text>'
        )
        legend_x += 110
    return "".join(parts)


def write_distribution_svg(
    output_path: str,
    title: str,
    weight_record: Dict[str, object],
    activation_record: Dict[str, object],
    baseline_weight_name: str,
    nf_weight_name: str,
    baseline_act_name: str,
    nf_act_name: str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    weight_panel = _render_panel(
        "Weight Distribution",
        [
            {"name": "original", "color": "#0969da", "hist": weight_record["original"]["hist"]},
            {"name": baseline_weight_name, "color": "#cf222e", "hist": weight_record[baseline_weight_name]["distribution"]["hist"]},
            {"name": nf_weight_name, "color": "#1a7f37", "hist": weight_record[nf_weight_name]["distribution"]["hist"]},
        ],
        20,
        50,
        820,
        220,
    )
    activation_panel = _render_panel(
        "Activation Distribution",
        [
            {"name": "original", "color": "#0969da", "hist": activation_record["original"]["hist"]},
            {"name": baseline_act_name, "color": "#cf222e", "hist": activation_record[baseline_act_name]["distribution"]["hist"]},
            {"name": nf_act_name, "color": "#1a7f37", "hist": activation_record[nf_act_name]["distribution"]["hist"]},
        ],
        20,
        290,
        820,
        220,
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="860" height="530">
<rect width="100%" height="100%" fill="#f6f8fa"/>
<text x="20" y="28" font-size="18" font-family="monospace">{title}</text>
{weight_panel}
{activation_panel}
</svg>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def export_distribution_reports(
    records: Dict[str, object],
    output_dir: str,
    baseline_weight_name: str = "gptq_int4",
    nf_weight_name: str = "gptq_nf4",
    baseline_act_name: str = "smooth_int8",
    nf_act_name: str = "smooth_nf8",
):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "layer_distribution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "# Layer Distribution Summary",
        "",
        f"- Records: `{json_path}`",
        "",
        "| Module | Weight INT4 MSE | Weight NF4 MSE | Act INT8 MSE | Act NF8 MSE | SVG |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for module_name, record in records.items():
        svg_name = module_name.replace(".", "_") + ".svg"
        svg_path = os.path.join(output_dir, svg_name)
        write_distribution_svg(
            output_path=svg_path,
            title=module_name,
            weight_record=record["weight"],
            activation_record=record["activation"],
            baseline_weight_name=baseline_weight_name,
            nf_weight_name=nf_weight_name,
            baseline_act_name=baseline_act_name,
            nf_act_name=nf_act_name,
        )
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{module_name}`",
                    str(record["weight"][baseline_weight_name]["error"]["mse"]),
                    str(record["weight"][nf_weight_name]["error"]["mse"]),
                    str(record["activation"][baseline_act_name]["error"]["mse"]),
                    str(record["activation"][nf_act_name]["error"]["mse"]),
                    f"`{svg_name}`",
                ]
            )
            + " |"
        )

    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


def run_vector_demo(
    output_dir: str,
    alpha: float,
    weight_vector: Optional[torch.Tensor] = None,
    activation_vectors: Optional[torch.Tensor] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    weight_vector = (_default_weight_vector() if weight_vector is None else weight_vector).float()
    activation_vectors = (_default_activation_vectors() if activation_vectors is None else activation_vectors).float()

    weight_uniform = quantize_weight_uniform(
        weight=weight_vector.unsqueeze(0),
        bits=4,
        group_size=weight_vector.numel(),
        symmetric=False,
    )
    weight_nf4 = quantize_weight_nf(
        weight=weight_vector.unsqueeze(0),
        bits=4,
        group_size=weight_vector.numel(),
    )

    smooth_scale = compute_smooth_scale(
        activation_absmax=activation_vectors.abs().amax(dim=0),
        weight_absmax=weight_vector.abs(),
        alpha=alpha,
    )
    smoothed = activation_vectors / smooth_scale.unsqueeze(0)

    act_int8 = quantize_activation_tensor(
        x=smoothed,
        bits=8,
        scheme="int8",
        granularity="per_token",
        symmetric=False,
    )
    act_nf8 = quantize_activation_tensor(
        x=smoothed,
        bits=8,
        scheme="nf8",
        granularity="per_token",
        symmetric=True,
    )

    restored_int8 = act_int8.dequantized * smooth_scale.unsqueeze(0)
    restored_nf8 = act_nf8.dequantized * smooth_scale.unsqueeze(0)

    report = {
        "weight_vector": _safe_float_list(weight_vector),
        "activation_vectors": [_safe_float_list(row) for row in activation_vectors],
        "smooth_scale": _safe_float_list(smooth_scale),
        "weight": {
            "gptq_int4_codes": _safe_int_list(weight_uniform.codes[0]),
            "gptq_int4_dequant": _safe_float_list(weight_uniform.dequantized[0]),
            "gptq_nf4_codes": _safe_int_list(weight_nf4.codes[0]),
            "gptq_nf4_dequant": _safe_float_list(weight_nf4.dequantized[0]),
        },
        "activation": {
            "smooth_int8_codes": [_safe_int_list(row) for row in act_int8.quantized],
            "smooth_int8_dequant": [_safe_float_list(row) for row in restored_int8],
            "smooth_nf8_codes": [_safe_int_list(row) for row in act_nf8.quantized],
            "smooth_nf8_dequant": [_safe_float_list(row) for row in restored_nf8],
        },
    }

    with open(os.path.join(output_dir, "vector_demo.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    bars = []
    width = 900
    height = 520
    origin_y = 170
    bar_width = 18
    gap = 10
    x = 30
    max_abs = max(
        weight_vector.abs().max().item(),
        weight_uniform.dequantized.abs().max().item(),
        weight_nf4.dequantized.abs().max().item(),
    )
    scale = 100 / max_abs if max_abs > 0 else 1.0
    colors = ["#0969da", "#cf222e", "#1a7f37"]
    labels = ["orig", "int4", "nf4"]
    for idx in range(weight_vector.numel()):
        values = [
            weight_vector[idx].item(),
            weight_uniform.dequantized[0, idx].item(),
            weight_nf4.dequantized[0, idx].item(),
        ]
        for value_idx, value in enumerate(values):
            bar_h = value * scale
            y = origin_y - max(bar_h, 0)
            bars.append(
                f'<rect x="{x + value_idx * bar_width}" y="{y:.2f}" width="{bar_width - 2}" '
                f'height="{abs(bar_h):.2f}" fill="{colors[value_idx]}"/>'
            )
        bars.append(
            f'<text x="{x}" y="{origin_y + 18}" font-size="10" font-family="monospace">{idx}</text>'
        )
        x += 3 * bar_width + gap

    table_lines = [
        "weight idx  orig     int4(code/deq)        nf4(code/deq)",
    ]
    for idx in range(weight_vector.numel()):
        table_lines.append(
            f"{idx:>3} {weight_vector[idx].item():>8.4f} "
            f"{int(weight_uniform.codes[0, idx]):>3}/{weight_uniform.dequantized[0, idx].item():>8.4f} "
            f"{int(weight_nf4.codes[0, idx]):>3}/{weight_nf4.dequantized[0, idx].item():>8.4f}"
        )

    act_lines = [
        "activation row 0  orig     int8(code/deq)         nf8(code/deq)",
    ]
    first_row = activation_vectors[0]
    for idx in range(first_row.numel()):
        act_lines.append(
            f"{idx:>3} {first_row[idx].item():>8.4f} "
            f"{int(act_int8.quantized[0, idx]):>4}/{restored_int8[0, idx].item():>8.4f} "
            f"{int(act_nf8.quantized[0, idx]):>4}/{restored_nf8[0, idx].item():>8.4f}"
        )

    text_y = 240
    text_blocks = []
    for line in table_lines + [""] + act_lines:
        text_blocks.append(
            f'<text x="30" y="{text_y}" font-size="12" font-family="monospace">{line}</text>'
        )
        text_y += 16

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#f6f8fa"/>
<text x="30" y="28" font-size="20" font-family="monospace">Vector Demo: GPTQ INT4 vs GPTQ+NF4, Smooth INT8 vs Smooth+NF8</text>
<text x="30" y="52" font-size="13" font-family="monospace">weight bar chart: blue=orig red=int4 green=nf4</text>
<line x1="20" y1="{origin_y}" x2="870" y2="{origin_y}" stroke="#8c959f"/>
{''.join(bars)}
{''.join(text_blocks)}
</svg>"""
    with open(os.path.join(output_dir, "vector_demo.svg"), "w", encoding="utf-8") as f:
        f.write(svg)
