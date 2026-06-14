# Stage4 Float32 Compression

说明代码的整体功能、核心函数拆分、如何单独调用、如何训练、如何推理，以及如何对 float32 的符号位、指数位和尾数位进行拆解与统计。

## 0. 路径占位符约定

本文档中的命令使用占位符表示路径。读者需要替换为自己机器上的真实路径：

| 占位符 | 含义 |
| --- | --- |
| `<PYTHON>` | Python 可执行文件路径，例如 `python`、`python.exe` 或某个 conda 环境下的 Python |
| `<PROJECT_ROOT>` | 项目根目录，即包含 `20260420/` 和 `experiments/` 的目录 |
| `<SGY_PATH>` | 输入 SEG-Y 文件路径 |
| `<DAT_PATH>` | 转换后的 float32 `.dat` 文件路径 |
| `<META_JSON>` | `.dat` 对应的 metadata JSON 路径 |
| `<MANIFEST>` | block manifest JSON 路径 |
| `<BLOCK_DIR>` | TUI block dat 输出目录 |
| `<OUTPUT_DIR>` | 当前实验输出目录 |
| `<CHECKPOINT>` | 训练得到的 `checkpoint.pt` 路径 |
| `<BITSTREAM>` | 手动编码输出的 `.s4rc` bitstream 路径 |

Windows PowerShell 路径可以写成 `C:\path\to\file`，Linux/macOS 可以写成 `/path/to/file`。下面命令中的占位符都需要替换后再运行。

## 1. 整体功能

核心思想是：

1. 将 float32 数据拆成 `sign`、`exponent`、`mantissa` 三部分。
2. 对 `exponent` 使用 Stage4 CNN 预测概率分布。
3. 将真实 exponent 或 residual symbol 用 range coder 编码成真实 bitstream。
4. 对 `sign` 和 `mantissa` 使用规则方法统计或压缩，例如 bitpack、RLE、zstd、lzma、PathL-lite。
5. 汇总得到：

```text
总字节数 = 指数位压缩字节数 + 符号位压缩字节数 + 尾数字节数
```

推荐主路径：

```text
diagonal_causal_edge 特征
+ residual 目标
+ global_diag 推理/编码布局
```

也就是训练时使用：

```text
--feature-mode diagonal_causal_edge
--target-mode residual
```

推理/真实编码时使用：

```text
--codec-layout global_diag
```

## 2. 目录和文件职责

### 2.1 主算法文件

| 文件 | 作用 |
| --- | --- |
| `common.py` | 通用数据结构、float32 位拆解、VolumeData、随机采样、配置类 |
| `stage4.py` | Stage4 CNN 指数预测模型、特征构造、residual 目标、训练与 benchmark |
| `codec.py` | Stage4 + range coding 的真实编码/解码，包括 raster、tile64、global_diag |
| `range_coder.py` | arithmetic/range coder 的底层实现 |
| `compute_tui_aux_bits.py` | TUI 数据的 sign/mantissa 精确统计与压缩候选比较 |
| `hybrid_codec.py` | ROI 内用 Stage4，ROI 外用 zstd 的混合实验路径 |

### 2.2 TUI 数据入口

| 文件 | 作用 |
| --- | --- |
| `tui_blocks.py` | 从 TUI metadata 构建不规则 block manifest，并可抽取 block dat |
| `run_tui_multiblock.py` | TUI 多 block 训练、benchmark、roundtrip 主入口 |
| `tui_blocks_manifest.json` | TUI block manifest |
| `tui_blocks/` | 抽取出来的 TUI block float32 dat |

### 2.3 PostData / 规则三维数据入口

| 文件 | 作用 |
| --- | --- |
| `postdata/postdata_blocks.py` | 为规则三维 float32 数据构建 manifest |
| `postdata/run_postdata_multiblock.py` | PostData 或其他规则三维数据的训练、benchmark、roundtrip 主入口 |
| `postdata/compute_postdata_aux_bits.py` | PostData 或规则三维数据的 sign/mantissa 统计 |

如果换新数据集，只要新数据能整理成规则三维 float32 dat，优先复用 `postdata/` 这条路径。

## 3. float32 位拆解

核心函数在 `common.py`：

```python
def extract_float_components(data_float32):
    data_u32 = np.asarray(data_float32, dtype=np.float32).view(np.uint32)
    signs = ((data_u32 >> 31) & 0x1).astype(np.uint8)
    exps = ((data_u32 >> 23) & 0xFF).astype(np.uint8)
    mants = (data_u32 & 0x7FFFFF).astype(np.uint32)
    return signs, exps, mants
```

三部分含义：

| 分量 | 位数 | 代码类型 | 含义 |
| --- | --- | --- | --- |
| `signs` | 1 bit | `uint8` | float32 符号位 |
| `exps` | 8 bits | `uint8` | IEEE754 biased exponent |
| `mants` | 23 bits | `uint32` | IEEE754 mantissa |

### 3.1 单独拆解一个 dat 文件

在任意 Python 脚本中可以这样调用：

```python
import sys
import numpy as np

sys.path.insert(0, r"<PROJECT_ROOT>\20260420")

from common import extract_float_components

path = r"<DAT_PATH>"
shape = (350, 600, 2001)

data = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
signs, exps, mants = extract_float_components(data)

print(signs.shape, signs.dtype)
print(exps.shape, exps.dtype)
print(mants.shape, mants.dtype)
```

### 3.2 从三部分还原 float32

```python
import numpy as np

restored_u32 = (
    (signs.astype(np.uint32) << 31)
    | (exps.astype(np.uint32) << 23)
    | mants.astype(np.uint32)
)
restored_float32 = restored_u32.view(np.float32)
```

如果 `signs/exps/mants` 没有被有损修改，`restored_float32` 应与原始 float32 按 bit 完全一致。

## 4. Stage4 指数预测算法

Stage4 只训练和预测 exponent，不直接预测完整 float32。

### 4.1 模型

模型定义：

```python
stage4.Small2DCNN
```

输入是当前点周围的二维 causal patch，输出是 256 类 logits，对应 8-bit exponent symbol 或 residual symbol。

### 4.2 特征模式

主要函数：

```python
stage4.build_stage4_features(...)
stage4.build_single_stage4_feature(...)
```

推荐使用：

```text
feature_mode = diagonal_causal_edge
```

它只使用因果可见区域，适合 lossless 编码，因为编码和解码时都不能看到未来点。

### 4.3 目标模式

主要函数：

```python
stage4.predictor_loco_i_2d(...)
stage4.residual_symbol(...)
stage4.reconstruct_exp_from_symbol(...)
stage4.target_symbol_for_coord(...)
```

推荐使用：

```text
target_mode = residual
```

含义是：

```text
symbol = (真实 exponent - LOCO-I 预测 exponent) mod 256
```

解码时：

```text
真实 exponent = (LOCO-I 预测 exponent + symbol) mod 256
```

这样通常比直接预测 raw exponent 更容易压缩。

### 4.4 单独构造一个点的特征

```python
import sys
import numpy as np

sys.path.insert(0, r"<PROJECT_ROOT>\20260420")

from common import extract_float_exponents
from stage4 import build_single_stage4_feature, target_symbol_for_coord

path = r"<DAT_PATH>"
shape = (350, 600, 2001)

floats = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
exps = extract_float_exponents(floats).reshape(shape)

coord = (0, 100, 1000)  # profile, trace, sample
feature = build_single_stage4_feature(
    exps,
    coord,
    patch_shape=(17, 17),
    feature_mode="diagonal_causal_edge",
    target_mode="residual",
)
label = target_symbol_for_coord(exps, coord, target_mode="residual")

print(feature.shape)  # [1, C, 17, 17]
print(label)
```

## 5. Range Coding 指数编码

核心文件：

```text
codec.py
range_coder.py
```

流程是：

1. Stage4 模型输出 256 类概率。
2. `codec.probs_to_cdf` 或 `codec.probs_to_cdfs` 将概率量化为 CDF。
3. `range_coder.RangeEncoder` 按 CDF 编码真实 symbol。
4. 解码时 `RangeDecoder` 使用同样的因果上下文和模型预测 CDF，恢复 symbol。
5. 如果 `target_mode=residual`，再用 LOCO-I 预测值还原 exponent。

主要 codec 类：

| 类 | 对应布局 | 说明 |
| --- | --- | --- |
| `Stage4RangeCodec` | `raster` | 最保守的逐点 raster 编码 |
| `Stage4TileRangeCodec` | `tile64` | 分 tile 编码 |
| `Stage4GlobalDiagonalRangeCodec` | `global_diag` | 推荐路径，按全局对角线批处理 |

推荐使用：

```python
from codec import Stage4GlobalDiagonalRangeCodec
```

## 6. TUI 数据训练和推理

### 6.1 准备 TUI block manifest

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\run_tui_multiblock.py `
  --action prepare `
  --tui-meta <META_JSON> `
  --manifest <MANIFEST> `
  --block-dir <BLOCK_DIR> `
  --extract-blocks
```

输出：

```text
<MANIFEST>
<BLOCK_DIR>\*.dat
```

### 6.2 训练 held-out 模型

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\run_tui_multiblock.py `
  --action train `
  --manifest <MANIFEST> `
  --split-mode heldout `
  --device cuda `
  --feature-mode diagonal_causal_edge `
  --target-mode residual `
  --materialize-features `
  --train-samples 500000 `
  --val-samples 50000 `
  --min-samples-per-block 5000 `
  --epochs-stage4 120 `
  --base-channels 16 `
  --output-dir <OUTPUT_DIR>
```

默认 held-out 范围写在 `README.md` 中。也可以手动指定：

```powershell
--val-range 3898-3912 `
--val-range 4075-4089 `
--test-range 3913-3927 `
--test-range 4090-4104
```

训练输出 checkpoint：

```text
<OUTPUT_DIR>\stage4\causal\checkpoint.pt
```

### 6.3 benchmark 推理

benchmark 只计算 sampled NLL、perplexity、top-k accuracy，不产生真实完整 bitstream。

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\run_tui_multiblock.py `
  --action benchmark `
  --manifest <MANIFEST> `
  --split-mode heldout `
  --benchmark-split test `
  --benchmark-samples 50000 `
  --checkpoint-path <CHECKPOINT> `
  --device cuda `
  --batch-size 512 `
  --output-dir <OUTPUT_DIR>
```

### 6.4 真实 roundtrip 推理/编码

roundtrip 会真实生成 exponent bitstream，并解码验证是否完全一致。

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\run_tui_multiblock.py `
  --action roundtrip `
  --manifest <MANIFEST> `
  --split-mode heldout `
  --roundtrip-split test `
  --checkpoint-path <CHECKPOINT> `
  --codec-layout global_diag `
  --codec-device cuda `
  --inference-batch 1024 `
  --output-dir <OUTPUT_DIR>
```

输出 JSON 中重点看：

```text
ok
total_encoded_bytes
total_raw_bytes
compression_ratio
```

### 6.5 TUI sign/mantissa 统计

如果只有 benchmark 的 `average_nll_bits`，可以用估计指数字节：

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\compute_tui_aux_bits.py `
  --manifest <MANIFEST> `
  --average-nll-bits 1.231013614639919 `
  --chunk-values 8000000 `
  --output-json <OUTPUT_DIR>\tui_aux_bits.json
```

如果已经有真实 roundtrip 的 `total_encoded_bytes`，更推荐：

```powershell
--exp-bytes FULL_EXPONENT_ACTUAL_BYTES
```

## 7. PostData 或其他规则三维数据训练和推理

### 7.1 新数据要求

新数据最好整理成一个连续 float32 dat：

```text
shape = (profile_count, traces_per_profile, samples_per_trace)
storage_order = trace_major_row_contiguous
```

写出 dat 示例：

```python
import numpy as np

volume = volume.astype(np.float32)
volume.tofile(r"<DAT_PATH>")
```

metadata JSON 至少包含：

```json
{
  "dat_path": "<DAT_PATH>",
  "trace_count": 210000,
  "samples_per_trace": 2001,
  "profile_count": 350,
  "traces_per_profile": 600,
  "total_samples": 420210000,
  "dtype": "float32",
  "storage_order": "trace_major_row_contiguous",
  "expected_bytes": 1680840000,
  "actual_bytes": 1680840000
}
```

其中：

```text
trace_count = profile_count * traces_per_profile
total_samples = trace_count * samples_per_trace
expected_bytes = total_samples * 4
```

### 7.2 自动生成 metadata

如果新数据是 `.sgy` 或 `.segy`，可以先用 `experiments/sgy_to_dat.py` 自动转换成 float32 `.dat`，并同时生成 metadata JSON：

```powershell
<PYTHON> <PROJECT_ROOT>\experiments\sgy_to_dat.py `
  --input <SGY_PATH> `
  --output <DAT_PATH> `
  --traces-per-profile 600 `
  --overwrite
```

输出文件：

```text
<DAT_PATH>
<DAT_PATH>.json
```

生成的 metadata 会包含：

```json
{
  "source_file": "<SGY_PATH>",
  "dat_path": "<DAT_PATH>",
  "trace_count": 210000,
  "samples_per_trace": 2001,
  "profile_source": "manual",
  "traces_per_profile": 600,
  "profile_count": 350,
  "profile_remainder_traces": 0,
  "total_samples": 420210000,
  "expected_total_samples": 420210000,
  "dtype": "float32",
  "storage_order": "trace_major_row_contiguous",
  "expected_bytes": 1680840000,
  "actual_bytes": 1680840000
}
```

`--traces-per-profile` 表示每个 profile 中有多少条 trace。如果 SEG-Y header 里的 inline/crossline 信息可靠，可以不传该参数，脚本会尝试从 header 推断：

```powershell
<PYTHON> <PROJECT_ROOT>\experiments\sgy_to_dat.py `
  --input <SGY_PATH> `
  --output <DAT_PATH> `
  --overwrite
```

脚本的 `profile_source` 字段说明 shape 来源：

| `profile_source` | 含义 |
| --- | --- |
| `manual` | 使用命令行传入的 `--traces-per-profile` |
| `header_inline` | 从 SEG-Y inline header 推断 |
| `header_crossline` | 从 SEG-Y crossline header 推断 |
| `heuristic` | 从常见 traces-per-profile 候选值推断 |
| `unknown` | 未能可靠推断，需要手动指定 |

如果不确定 `traces_per_profile`，可以先用坐标推断脚本辅助判断：

```powershell
<PYTHON> <PROJECT_ROOT>\experiments\infer_sgy_grid_from_coords.py `
  --input <SGY_PATH>
```

它会输出候选的：

```text
profile_count
traces_per_profile
```

如果数据像 TUI 一样是不规则网格、每个 subline 的 xline 范围不同，应使用 TUI 专用转换脚本：

```powershell
<PYTHON> <PROJECT_ROOT>\experiments\tui_sgy_to_dat.py `
  --input <SGY_PATH_1> <SGY_PATH_2> `
  --output <DAT_PATH> `
  --overwrite
```

它会生成 TUI 风格的 ragged-grid metadata，后续应走 `run_tui_multiblock.py --action prepare`，而不是 PostData 的规则三维 manifest 流程。

如果数据已经是 `.dat`、`.bin`、`.npy` 或其他数组格式，目前没有单独的通用 metadata 生成入口。此时需要手动确认：

```text
profile_count
traces_per_profile
samples_per_trace
dtype=float32
storage_order=trace_major_row_contiguous
```

然后写出前面 `7.1` 中的 metadata JSON。只要这些字段正确，后面的 manifest、训练和推理流程都可以复用。

### 7.3 生成 manifest

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\postdata\run_postdata_multiblock.py `
  --action prepare `
  --postdata-meta <META_JSON> `
  --manifest <MANIFEST>
```

如果数据太大，可以按 profile 分块：

```powershell
--profiles-per-block 50
```

### 7.4 训练新数据集

`--val-range` 和 `--test-range` 是 profile 范围。比如新数据有 100 个 profile，可以设：

```text
train: 0-79
val:   80-89
test:  90-99
```

命令：

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\postdata\run_postdata_multiblock.py `
  --action train `
  --manifest <MANIFEST> `
  --split-mode heldout `
  --val-range 80-89 `
  --test-range 90-99 `
  --device cuda `
  --feature-mode diagonal_causal_edge `
  --target-mode residual `
  --materialize-features `
  --train-samples 500000 `
  --val-samples 50000 `
  --benchmark-samples 50000 `
  --min-samples-per-block 5000 `
  --epochs-stage4 120 `
  --base-channels 16 `
  --output-dir <OUTPUT_DIR>
```

### 7.5 benchmark 新数据集

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\postdata\run_postdata_multiblock.py `
  --action benchmark `
  --manifest <MANIFEST> `
  --split-mode heldout `
  --val-range 80-89 `
  --test-range 90-99 `
  --benchmark-split test `
  --benchmark-samples 50000 `
  --checkpoint-path <CHECKPOINT> `
  --device cuda `
  --batch-size 512 `
  --output-dir <OUTPUT_DIR>
```

### 7.6 roundtrip 新数据集

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\postdata\run_postdata_multiblock.py `
  --action roundtrip `
  --manifest <MANIFEST> `
  --split-mode heldout `
  --val-range 80-89 `
  --test-range 90-99 `
  --roundtrip-split test `
  --checkpoint-path <CHECKPOINT> `
  --codec-layout global_diag `
  --codec-device cuda `
  --inference-batch 1024 `
  --output-dir <OUTPUT_DIR>
```

### 7.7 新数据集 sign/mantissa 统计

```powershell
<PYTHON> <PROJECT_ROOT>\20260420\postdata\compute_postdata_aux_bits.py `
  --manifest <MANIFEST> `
  --exp-bytes FULL_EXPONENT_ACTUAL_BYTES `
  --chunk-values 8000000 `
  --output-json <OUTPUT_DIR>\aux_bits.json
```

如果只想统计某个 profile 范围：

```powershell
--profile-range 90-99
```

## 8. 单独调用训练好的 codec

如果已经有 checkpoint，并且已经有 exponent volume，可以直接实例化 codec：

```python
import sys
import numpy as np

sys.path.insert(0, r"<PROJECT_ROOT>\20260420")

from common import ExperimentConfig, extract_float_exponents
from codec import Stage4GlobalDiagonalRangeCodec

dat_path = r"<DAT_PATH>"
shape = (350, 600, 2001)
checkpoint = r"<CHECKPOINT>"
bitstream = r"<BITSTREAM>"

floats = np.memmap(dat_path, dtype=np.float32, mode="r", shape=shape)
exps = extract_float_exponents(floats).reshape(shape)

cfg = ExperimentConfig(
    feature_mode="diagonal_causal_edge",
    target_mode="residual",
    codec_device="cuda",
)

codec = Stage4GlobalDiagonalRangeCodec(
    checkpoint_path=checkpoint,
    config=cfg,
    device="cuda",
    inference_batch=1024,
)

encode_metrics = codec.encode_exponents(exps, bitstream)
decoded, header = codec.decode_exponents(bitstream)

print(encode_metrics)
print(np.array_equal(decoded, exps))
```

注意：直接调用 `Stage4GlobalDiagonalRangeCodec` 时，输入必须是 `uint8 exponent volume`，不是原始 float32 volume。

## 9. 典型实验顺序

### 9.1 方法验证

```text
prepare manifest
-> train heldout
-> benchmark test
-> roundtrip test
-> compute aux bits for test or full data
```

### 9.2 最终全数据压缩

```text
prepare manifest
-> train full_train
-> roundtrip all
-> compute aux bits with --exp-bytes
```

`full_train` 的结果是数据集特定压缩结果。和传统压缩器比较时，建议同时报告：

```text
不含 checkpoint size
含 checkpoint size
```

## 10. 常见注意事项

1. `benchmark` 的 `average_nll_bits` 是采样估计，不是真实 bitstream 字节数。
2. `roundtrip` 的 `total_encoded_bytes` 才是真实 exponent bitstream 字节数。
3. `sign` 和 `mantissa` 不经过 Stage4 训练，使用 `compute_*_aux_bits.py` 单独统计。
4. `global_diag` 要求 causal-compatible 的 feature mode，推荐 `diagonal_causal_edge`。
5. `target_mode=residual` 依赖 LOCO-I 预测器，编码和解码必须使用相同配置。
6. 新数据集必须保证 metadata 中的 shape、trace_count、total_samples、byte_count 与 dat 文件真实大小一致。
7. Windows PowerShell 多行命令使用反引号 `` ` ``，CMD 使用 `^`。
