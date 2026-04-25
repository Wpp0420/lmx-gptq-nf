# GPTQ W4A8 量化 - Qwen3 模型

## 项目概述

对 **Qwen3-1.7B / Qwen3-8B** 模型进行 **W4A8 混合精度量化**，并评估量化后的困惑度变化。

| 组件 | 说明 |
|------|------|
| **W4A16** | 权重 4-bit GPTQ 量化，激活保持 FP16 |
| **W4A8** | 权重 4-bit GPTQ 量化，激活做 8-bit 量化仿真 |
| **NF4-GPTQ** | 用 GPTQ 的 Hessian/OBS 流程，量化器从线性 INT4 改为 NF4 码本 |
| **NF8-Smooth** | 在 SmoothQuant 平滑后，把激活量化器从线性 INT8 改为 NF8 码本 |
| **评估** | WikiText-2 / 本地文本困惑度 (Perplexity) |

## 技术方案

### GPTQ 算法原理

GPTQ (Generalized Post-Training Quantization) 是一种基于二阶信息的后训练量化方法：

1. **逐层量化**：对 Transformer 每一层的权重矩阵独立处理
2. **Hessian 矩阵**：利用校准数据计算 $H = 2X^TX$，其中 $X$ 是该层的输入激活
3. **OBS 最优量化**：按 Hessian 对角线排序权重列，优先量化对输出影响小的列
4. **误差补偿**：量化一列后，将误差传播到剩余列：
   $$\delta \mathbf{w} = -\frac{w_q - w}{[H^{-1}]_{qq}} \cdot H^{-1}_{:,q}$$
5. **分组量化**：将权重行分为若干组 (group_size=128)，每组共享量化参数

### W4A8 方案

```
推理流程:
Input (FP16) → [A8量化] → INT8 激活
                            ↓
W4权重 (GPTQ INT4) → [反量化] → FP16
                            ↓
                    矩阵乘法 (FP16)
                            ↓
                    Output (FP16) → 下一层
```

- **W4 (权重)**：GPTQ 离线量化，一次完成
- **A8 (激活)**：推理时 per-token 动态 INT8 量化

### SmoothQuant 增强

当前仓库已支持在 W4A8 推理前增加一轮 **SmoothQuant 风格的激活平滑校准**。需要注意：当前实现是激活量化仿真路径，不是替换为真实 INT8xINT4 推理内核。

- 对目标线性层收集输入通道的激活最大值
- 根据 `alpha` 生成每层的 channel-wise smooth scale
- 推理时先对激活做平滑，再执行 INT8 量化/反量化
- 对能直接访问浮点权重的层，使用更标准的 `act^alpha / weight^(1-alpha)` 公式
- 对 GPTQ 打包层拿不到浮点权重时，退化为仅基于激活统计的平滑模式

这不是完整的离线权重重参数化版 SmoothQuant，但对当前项目这种“GPTQ 权重 + 在线激活 fake quant”路径更容易接入，也通常能明显优于直接 per-tensor A8。

### NF4 / NF8 扩展

本仓库新增了两条扩展路径：

- **NF4 + GPTQ**：保留 GPTQ 的逐层 Hessian 近似、按列量化和误差补偿，但把每一步的权重量化器从线性 INT4 改为 NF4 非均匀码本。
- **NF8 + SmoothQuant**：先按 SmoothQuant 对激活做 channel-wise 平滑，再将激活从线性 INT8 fake quant 改为非均匀的 NF8 fake quant。

实现上：

- `autogptq` / `transformers` 后端仍保留原始 INT4 路线
- `custom` 后端支持 `int4` 与 `nf4`
- `w4a8_inference.py` 现在支持 `int8` 与 `nf8` 两种激活量化方案
- 会额外输出
  - `analysis/vector_demo/*`：简单向量对比图
  - `analysis/layers/*`：每层每个 block 的权重/激活分布 SVG 和 JSON

### NF4 / NF8 量化流程与优化方向

`test1` 与 `test2` 的最终 PPL 对比显示：`test1` 的 `INT4 + INT8` 为 **16.0392**，`test2` 开启 `NF4 + NF8 + SmoothQuant` 后 PPL 变大，说明非均匀码本并不会在当前流程中天然优于线性量化。当前实现的端到端函数流程如下：

1. `run_pipeline.run_full_pipeline()` 读取 `config.py`，当 `weight_quant_scheme="nf4"` 时自动切到 `custom` 后端。
2. `custom_gptq_backend.quantize_with_custom_backend()` 加载 FP16 模型、准备校准样本，并用 `_capture_first_layer_inputs()` 截获第 0 层输入。
3. 每层内对目标 `nn.Linear` 注册 hook，`CalibrationAccumulator.add_batch()` 收集 GPTQ Hessian、输入样本和激活 absmax。
4. `_solve_gptq()` 执行 GPTQ/OBS 主循环：按 block/列取权重，调用量化器 `find_params()` 与 `quantize()`，再用 Hessian 逆矩阵做误差补偿。
5. INT4 路径使用 `UniformAffineQuantizer`；NF4 路径使用 `NormalFloatQuantizer`、`get_normal_float_codebook()`、`codebook_lookup()`。
6. `_save_custom_quantized_model()` 将量化后的权重再次编码为 `packed_weights.pt`，加载时 `load_custom_quantized_model()` 通过 `dequantize_weight()` 恢复 fake-quant 权重。
7. PPL 评估调用 `w4a8_inference.load_w4a8_model()`，`W4A8ModelWrapper.apply_activation_quantization()` 包装 Linear，运行时 `ActivationQuantWrapper.forward()` 对输入激活做 `quantize_activation_tensor()`。
8. NF8 激活路径由 `quantize_activation_nf()` 完成：按粒度计算 scale，查 NF8 codebook，反量化后再送入线性层。

本次针对 PPL 退化做了两点代码级优化：

- **NF4 权重 scale 搜索**：`NormalFloatQuantizer.find_params()` 和 `quantize_weight_nf()` 现在使用 `_search_nf_scale()` 在 absmax 附近搜索重构 MSE 更低的 scale，避免 NF4 固定 absmax scale 被少量 outlier 拉大、压缩主体权重分辨率。
- **SmoothQuant 等价折叠**：原推理路径只做 `x / smooth_scale` 后直接使用原权重，会改变线性层数学语义；现在 `ActivationQuantWrapper.forward()` 对 `nn.Linear` 使用 `F.linear(x_quant_dequant, W * smooth_scale, bias)`，恢复 SmoothQuant 的 `x/s × (W*s)` 等价关系，只保留激活量化误差。

后续优化方向建议：

- 对 NF4：继续调 `group_size`、`desc_act`、`blocksize` 和 `damp_percent`，并优先观察每层 `gptq_nf4_avg_loss`，对异常层回退 INT4 或使用更小 group。
- 对 NF8：保持 SmoothQuant 权重折叠，不建议默认做每次前向的 NF8 MSE scale 搜索；如做部署前静态校准，可在导出的激活 scale 上离线搜索 clipping/percentile scale。
- 对组合策略：NF4 与 NF8 的误差会叠加，建议分别评估 `NF4+INT8` 与 `INT4+NF8`，先确认主要 PPL 退化来自权重还是激活。

## 项目结构

```
GPTQ_Opus/
├── config.py            # 量化和评估的配置参数
├── utils.py             # 工具函数 (数据加载、设备信息等)
├── quantize_gptq.py     # GPTQ W4 权重量化实现
├── w4a8_inference.py     # W4A8 推理引擎 (A8 激活量化)
├── evaluate_ppl.py       # 困惑度评估
├── run_pipeline.py       # 一键运行脚本
├── requirements.txt      # 依赖
└── README.md            # 本文档
```

## 环境安装

```bash
pip install -r requirements.txt
```

> **注意**: `auto-gptq` 需要匹配 CUDA 版本，若安装失败请参考
> [auto-gptq 官方指南](https://github.com/AutoGPTQ/AutoGPTQ)

## 使用方法

### 1. 统一配置方式

项目现在统一采用 **config.py 作为唯一配置源**：

- 在 `config.py` 中修改 GPTQ、激活量化、SmoothQuant、评估超参数
- 在命令行中只传模型路径、缓存路径、流程控制参数

常见需要修改的位置：

```python
# config.py
class ActivationQuantConfig:
    act_bits: int = 8
    activation_quant_scheme: str = "nf8"
    granularity: str = "per-channel"
    use_smoothquant: bool = True
    smoothquant_dataset: str = "local:./eval_data/wikitext2_train.txt"

class GPTQQuantConfig:
    bits: int = 4
    weight_quant_scheme: str = "nf4"

class EvalConfig:
    max_length: int = 2048
    stride: int = 512
    eval_datasets = ["local:./eval_data/wikitext2.txt"]
```

### 2. 一键完整流程 (推荐)

```bash
# Qwen3-1.7B 完整流程：量化 + 评估
python run_pipeline.py --model /path/to/Qwen3-1.7B

# 生成简单向量的 INT4/NF4 与 INT8/NF8 对比图
python run_pipeline.py --model /path/to/Qwen3-1.7B --run_vector_demo

# 使用自定义后端做 NF4-GPTQ，并导出逐层分布分析
python run_pipeline.py \
    --model /path/to/Qwen3-1.7B \
    --backend custom \
    --run_layer_distribution

# Qwen3-8B 完整流程
python run_pipeline.py --model /path/to/Qwen3-8B
```

### 3. 分步执行

#### Step 1: GPTQ W4 量化

```bash
python quantize_gptq.py --model /path/to/Qwen3-1.7B
```

#### Step 2: 困惑度评估

```bash
# 评估所有版本 (FP16 / W4 / W4A8)
python evaluate_ppl.py \
    --model /path/to/Qwen3-1.7B \
    --quantized_model ./Qwen3-1.7B_w4a8_gptq \
    --eval_mode all

# 只评估 W4A8
python evaluate_ppl.py \
    --model /path/to/Qwen3-1.7B \
    --quantized_model ./Qwen3-1.7B_w4a8_gptq \
    --eval_mode w4a8
```

### 4. 运行时选项

```bash
# 跳过 FP16 评估 (节省显存)
python run_pipeline.py --model /path/to/Qwen3-1.7B --skip_fp16_eval

# 只量化 (跳过评估)
python run_pipeline.py --model /path/to/Qwen3-1.7B --skip_eval

# 只评估已有量化模型
python run_pipeline.py \
    --model /path/to/Qwen3-1.7B \
    --skip_quantize \
    --quantized_model ./Qwen3-1.7B_w4a8_gptq

# 使用 transformers 内置后端量化
python run_pipeline.py --model /path/to/Qwen3-1.7B --backend transformers

# 使用 custom 后端（支持 NF4-GPTQ 和逐层分布分析）
python run_pipeline.py --model /path/to/Qwen3-1.7B --backend custom

# 使用本地 txt 作为评估集和 SmoothQuant 校准集
python run_pipeline.py \
    --model /path/to/Qwen3-1.7B \
    --skip_quantize \
    --quantized_model ./Qwen3-1.7B_w4a8_gptq \
    --local_files_only \
    --hf_cache_dir /home/lmx/.cache/huggingface/hub \
    --eval_datasets local:./eval_data/wikitext2.txt
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必填) | 模型名称或本地路径 |
| `config.py` | - | 统一修改量化、激活、SmoothQuant、评估超参数 |
| `--model` | (必填) | 模型名称或本地路径 |
| `--quantized_model` | 自动推导 | 已有量化模型路径 |
| `--eval_datasets` | config.py 或 wikitext2 | 评估数据集列表 |
| `--backend` | autogptq | 量化后端 |
| `--run_vector_demo` | False | 导出简单向量的量化前后对比图 |
| `--run_layer_distribution` | False | 导出每层每个 block 的权重/激活分布 |
| `--device` | cuda:0 | 运行设备 |
| `--local_files_only` | False | 仅从本地缓存加载模型/数据集 |
| `--hf_cache_dir` | None | Hugging Face 缓存目录 |
| `--skip_quantize` | False | 跳过量化 |
| `--skip_eval` | False | 跳过评估 |
| `--skip_fp16_eval` | False | 跳过 FP16 评估 |

## 预期结果 (参考)

以 Qwen3-1.7B 为例，在 WikiText-2 上的困惑度预期结果：

| 模型 | PPL | PPL 变化 | 模型大小 |
|------|-----|----------|----------|
| FP16 (基线) | ~12-14 | - | ~3.4 GB |
| GPTQ W4 | ~12.5-15 | +3~8% | ~1.0 GB |
| W4A8 | ~13-16 | +5~12% | ~1.0 GB |

> 实际数值取决于模型版本、校准数据量和量化参数

## 显存需求

| 模型 | FP16 | 量化过程 | W4 推理 |
|------|------|----------|---------|
| Qwen3-1.7B | ~4 GB | ~8 GB | ~2 GB |
| Qwen3-8B | ~16 GB | ~24 GB | ~6 GB |

## 关键代码说明

### 激活量化核心 (Per-Token INT8)

```python
# 对称量化: scale = max(|x|) / 127
abs_max = x.abs().amax(dim=-1, keepdim=True)
scale = abs_max / 127

# 量化: round + clip
x_int8 = torch.round(x / scale).clamp(-127, 127).to(torch.int8)

# 反量化
x_dequant = x_int8.float() * scale
```

### NF4/NF8 码本量化核心

```python
# 先按 group / granularity 求 absmax 做归一化
x_norm = x / absmax

# 在排序后的 NormalFloat 码本上做最近邻查找
codes = bucketize(x_norm, codebook_midpoints)
x_dequant = codebook[codes] * absmax
```

### GPTQ 误差补偿

```python
# 对每一列权重:
# 1. 量化: w_q = round(w / scale) * scale  
# 2. 计算误差: err = w - w_q
# 3. 补偿: w_remaining -= err * H_inv[:, col] / H_inv[col, col]
```

## License

本项目仅用于研究目的。模型权重受 Qwen3 原始许可证约束。
