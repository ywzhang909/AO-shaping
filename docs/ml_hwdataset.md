# ml/hwdataset 模块详解

> 本文件从根 [AGENTS.md](../AGENTS.md) 的 `ml/hwdataset/ Module` 节外迁（2026-10-27）。
> 详尽实测结论集中于此，AGENTS.md 只保留索引指针。

## ml/hwdataset/ Module — 硬件调试转 Dataset

`src/ml/hwdataset/` 把 `data/debug/` 下的**全部**硬件调试产物转成 PyTorch Dataset:
**输入 = 下发给 SLM 的相位 + 相机曝光参数, 输出 = 相机画面**。独立包 (`src/ml/`),
不 import 任何硬件驱动即可 import (`ml` 是可选依赖组)。

### 语料实测 (2026-10-02, 已核验)

| 项 | 值 |
|---|---|
| 语料 | `data/debug/` 996 文件 / **265 个 `.pkl` / 65.7 GB** / 单文件最大 **3.42 GB** / 中位数 55 MB |
| 可用记录 | **11,393** / 261 文件 / 7 个 family (总记录 11,441) |
| 相位来源 | `panel_gray` 7866 (uint16 灰度) · `panel_rad` 2199 (float32 弧度) · `zernike` 1202 (由 `_c` 反演) · `freeform` 126 (由 `_c` 上采样) |
| 排除 | `missing_phase` 48 (`bench_stability` 42 条平场漂移无指令相位 + `sim_calib_abba` 6/18) · `no_records` 3 (`None` 占位) |
| 曝光 | 11,393/11,393 全部可解析; 0.1–80.0 ms, 9 个不同值 |
| 索引构建 | 全量 35 s; 带 JSON cache **0.09 s** |
| 整轮 epoch | 11,393 条 / 179 batch (bs=64) / 单进程 **450 s** |

### 模块

| 文件 | 职责 |
|---|---|
| `index.py` | 递归发现全部 `*.pkl` + 合并 sidecar JSON + **逐记录**判定相位来源。`PhaseSource` / `ExclusionReason` / `HwRecordRef` / `HwCorpusIndex` / `build_hw_index` / `family_of` |
| `transforms.py` | 纯 array→array: 灰度→弧度、ROI 裁剪、**相干**块平均、Zernike/freeform 面板还原、远场裁剪 |
| `records.py` | 单条 `HwRecordRef` → `HwSample` 的唯一实现 (dataset 与 cache 共用, 保证两条路径**逐位一致**); `MaterialiserConfig` / `PayloadStore`(有界 LRU) / `Materialiser`(也是缓存读取点, `use_cache` 默认 True) |
| `train_amp.py` | `python -m ml.zernike.train_amp` 训练入口: DataLoader 驱动, 记录 loss / grad norm / R² / PSNR / SSIM / NRMSE / perplexity / corr / efficiency / 质心偏移 / 光斑直径比, 出 true-vs-pred 对比图, 推 wandb (无 key 时 offline) |
| `metrics.py` | img2img 标准指标 (MSE/RMSE/MAE/NRMSE/PSNR/SSIM) + 光斑域指标 (质心偏移 / 90% 环围直径 / 峰比), 复用 `beam_metrics.compute_beam_metrics` / `measure_spot_diameter_cam` |
| `dataset.py` | `HwPhaseImageDataset` / `FileGroupedSampler` / `build_hw_dataloader` / `create_hw_dataloaders` |
| `cache.py` | 派生网格的 mmap 缓存 (`.hw_cache/<stem>/`), 消除每轮 212 s Zernike 反演 + 35 s 读盘 |
| `inspect.py` | `python -m ml.hwdataset.inspect` 语料统计 / 抽样自检 / 建缓存 |

#### 缓存是**被读取**的, 不是只能写 (2026-10-02 实测并接线)

`Materialiser(use_cache=True)` (默认) 在 `materialise()` 里先查缓存, 命中就走 mmap,
不命中 (无缓存 / 配置不符 / 该 ref 未缓存 / 缓存损坏) 逐条回落到直接路径。`prepare_hw_cache`
与 Dataset 共用同一个 `Materialiser`, 所以两条路径**逐位一致**, `use_cache` 只改速度与内存。

真语料 `slm_zernike_shaping` (1010 记录 / 10 pickle) 实测:

| 路径 | 耗时 | 打开的 pickle 数 |
|---|---|---|
| `use_cache=False` | 127.0 s | 10 |
| `use_cache=True` | **0.4 s** | **0** |

**317× 加速, 且一个 pickle 都不再打开** —— 峰值 RAM 从「一个整 pickle (最大 3.42 GB)」
降到「一页」。实测 1010/1010 样本 (含 `exposure_ms` / `fov_px`) 与直接路径逐位一致。

接线时踩到并修掉的四个真问题 (勿回退):

1. **`prepare_hw_cache` 必须 `use_cache=False`。** 否则 builder 会拿正要写的缓存当输入,
   Windows 下已打开的 mmap 让目标文件不可覆写, 每个 `np.save` 报 `Invalid argument` (errno 22),
   整族被静默跳过。
2. **`exposure.npy` 必须是 float64, 不是 float32。** float32 把 `0.1 ms` 变成
   `0.10000000149011612`, 真语料 1010/1010 样本只在这一个字段上不一致 (合成语料的
   1.0/2.0/0.5 恰好可被 float32 精确表示, 所以测试盲区掩盖了它)。数组是 `(N,)`, 全库 91 KB。
3. **`fov_px` 必须走缓存。** 原先 `Dataset._fov_px` 从 payload LRU 反查 `_img` 形状, 靠
   「`materialise` 刚好填过一次 LRU」成立 —— 缓存命中时 LRU 从未被填, 于是每条记录都要**重新
   打开整个 pickle** (最大 3.42 GB), 恰好抵消缓存的意义。现在几何尺寸由 `HwSample.fov_px` 携带。
4. **`Materialiser.__getstate__` 必须丢掉 `_cached`。** `np.memmap` 不可 pickle;
   Windows `spawn` 每 worker 重建 Dataset, 丢了 flag 会静默退回慢路径整个 epoch。

> **缓存槽位 = 该 pickle 内的成功物化序号 (rank); 唯一定址方式是
> `HwCachedSource.position_of(ref.key if ref.key is not None else ref.position)`。**
> 缓存只为**成功物化**的 ref 存一个槽, 顺序跟随 index 顺序; 每个槽写入的是**该 ref 自己的
> stored key** (`keys.npy`), 所以上面这个查表对任何语料都精确 —— 即使
> `key != position` (真语料 `sim_calib_abba_20261001_163743.pkl` 存 key 10000-10011 于
> position 6-17), 即使有记录被排除 (两者都会让 rank ≠ position, 但都不动 stored key)。
> 实测真语料 1010/1010 记录按此取槽逐位一致。
> ⚠️ 键在单个 pickle 内唯一, 故 `position_of()` 单射; 但它只在「该 ref 确实被缓存」时可用,
> `KeyError` 表示回落到直接路径, 不是错误。

### 样本契约

```python
{
  "phase_cos":     (1, g, g) f32,   # 相位网格的 Re —— 连续表示, 无 arctan2 分支切口
  "phase_sin":     (1, g, g) f32,   # 相位网格的 Im
  "image":         (1, g, g) f32,   # 远场画面, [0,1] 绝对强度 (未做 peak 归一)
  "exposure_log10": (1,)      f32,   # (log10(ms) - mean) / std, 按 index 级统计标准化
  "exposure_ms":   float,           # 原始 ms (default_collate 会转成 (B,) f64 张量)
  "contrast":      (g, g)    f32,   # hypot(cos, sin) = 每格相位相干度 [0,1]
  "source": str, "family": str, "sample_idx": int, "path": str, "fov_px": int | None,
}
```

> `fov_px` 由 `HwSample.fov_px` 携带 (`records.py`), 不再由 Dataset 回查 payload ——
> 缓存命中时 payload 根本不在内存里, 回查会重新打开整个 pickle。

### 🔴 四条不可回退的实测结论 (改动前必读)

> **单指标会骗人 —— 用 `ZernikeAmpModel` 时三个归一化/可观测量选项各自埋了一个坑,
> 都是实测出来的 (真语料 `slm_zernike_shaping`, 固定 split, 逐项单变量):**
>
> | 选项 | val MSE | R² | PSNR | SSIM | 结论 |
> |---|---|---|---|---|---|
> | `normalization="peak"` | 0.00429 | **+0.706** | 24.8 dB | 0.594 | ✅ 唯一正确选择 |
> | `normalization="sum"` | **0.00000** | +0.632 | **72.1 dB** | **0.9996** | ⚠️ **陷阱** |
> | `normalization="none"` | 0.47534 | **−158.98** | 3.2 dB | 0.037 | ❌ 不可用 |
>
> `sum` 把预测与目标各自除以总能量, 两者于是"几乎一致" —— MSE/PSNR/SSIM 全部好到离谱,
> 而 **R² 反而更差**。任何含 sum 归一化的指标都在度量归一化本身, 不是拟合质量。
> **判据: 只看 R² / corr / 光斑域指标, 永远不要用 MSE/PSNR/SSIM 选这个开关。**
>
> | `observable` | n_max=4 | n_max=11 | n_max=15 |
> |---|---|---|---|
> | `"amplitude"` (原始需求写的 `amp`) | +0.634 | +0.659 | +0.675 |
> | `"intensity"` (**默认**) | **+0.750** | **+0.792** | **+0.797** |
>
> **CCD 测的是强度, 不是场振幅**, 所以 `intensity` 既是物理上正确的, 也是实测最优的,
> 三个 `n_max` 上都稳定领先 ~0.12 R²。默认值已改为 `intensity`, `amplitude` 保留可选。
>
> **噪声地板: 只换 split seed, R² 就从 0.780 跑到 0.923 (σ≈0.073)。** 同一 config
> 同 seed 重复 3 次结果**逐位相同** (确定性没问题), 所以这个方差全部来自
> "val 只有 2 个文件 / 128 条"。**单次 run 的 R² 差 ±0.13 是噪声, 不要据此排序**。
> 逐项单变量扫描里只有 `n_max` (0.48→0.80) 明显高于噪声; `l2_penalty` /
> `grad_clip` / `momentum` / `optimizer` 的差异全在噪声内 —— 想要可信排序必须多种子取均值。

> **噪声地板的真因 (2026-10-04): 主要不是"验证集只有 128 条", 而是折间难度差。**
> `slm_zernike_shaping` 的 1010 条记录均匀分布在 10 个 pickle (每份恰好 101 条),
> 但这 10 份只来自 **4 个优化目标** (`rms_pib` 4 / `rmse_out` 3 / `shape` 2 / `roi_pib` 1)。
> **四个目标的图像分布确实不同** —— 用 A 目标均值图预测 B 目标, 峰值归一后 R² 落在
> **+0.54 ~ +0.97** (最低 `shape` → `roi_pib` = +0.538), 对角线为 1。
> ⚠️ 但**没有一对是负的**。早期文档称 `rms_pib → roi_pib = −0.670`「比预测常数还差」
> —— **那个数是算错的** (分母误用了另一个 group 的方差, 量纲不一致而放大; 正确分母是
> 被预测组自身的 `SS_tot`)。结论方向 (目标异质 → 折的组成会变) 成立, 但"比常数还差"作废。
> **真正的大头是折难度本身**: 留一 pickle 的逐折 R² 从 0.737 到 0.941 (σ≈0.08),
> 同一模型换一折就差 0.2 ⇒ 任何**非配对**比较都淹没在折间方差里。
> **所以必须分组 + 配对**: 三个模型共用同一批折, 折难度在差值里抵消, 配对 σ 降到 ~0.01。
> ⚠️ 另: `train_amp._select_records` 的划分是**按 objective 分层的按文件划分**
> (2026-10-07 改为 objective-stratified, 此前是"按文件随机 75/25 + `max_val=128` 截断")。
> **但分层并没有消解 train/val 缺口 —— 别再宣称它消解了。** 6 seed × 2 arm × 40 epoch
> 配对实测 (同 epoch 对比, `n_max=4`, 唯一变量是划分):
>
> | | gap@best_ep 均值±sd | val R² 均值±sd |
> |---|---|---|
> | 旧划分 | 1.16 ± 0.71 (0.63–2.32) | +0.8418 ± 0.0774 |
> | 分层划分 | 1.13 ± 0.33 (0.74–1.62) | +0.8499 ± 0.0201 |
>
> 配对差 (旧−新) `p=0.9375` / `d_z=+0.03` ⇒ **均值无效应**, 两边缺口都 ≈1.15。
> ⚠️ **早期"缺口 1.97 → 0.78 已消解"的说法是单 seed 噪声, 已作废**: 旧划分的 val 组成
> **随 seed 剧变** (seed 0 抽到 2 个 `rmse_out` → gap 1.99; seed 2 抽到单个 `rms_pib`
> → gap 0.63), 当时只跑了 seed 0, 正好抽到最差情形。
> **分层真正修掉的是方差与可复现性, 不是均值**: val 组成恒为 `{rms_pib, rmse_out, shape}`
> (旧划分 6 个 seed 出现 5 种不同组合), gap sd 0.71→0.33, val R² sd 0.077→0.020 (3.9×),
> 且逐目标 val R² 每 epoch 可见 (旧划分常只有 1 个目标, 看不见别的目标崩)。
> 单一目标 (`roi_pib`, 仅 1 文件) 无法拆分, 只进 train。"train on one objective"
> 用 `--file-contains <substring>`。
> ⚠️ **`gap` 指标本身有缺陷**: `best_val_mse / final_train_mse` 取的是**不同 epoch**
> (early-stop 的 val vs 最后一轮 train), 而 history 里的 `train_mse` 是**轮内平均**
> (权重还在动)。要看真实泛化缺口必须用 `train_eval_mse` / `train_eval_r2` / `gap_mse`
> (同 epoch、同一权重、同一 peak 归一)。真实缺口很小 (val R² ≈ +0.85), 无过拟合
> (14 参数 vs 512 样本) ⇒ **"差异很大"主要是单 seed + val 组成漂移的错觉**。
> ⚠️ `HWRecordRef.source` 是**相位表示** (`panel_gray`/`panel_rad`/`zernike`/`freeform`),
> **不是来源文件**; 文件标识是 `HWRecordRef.path`。

> **与 U-Net 的对比 — 权威结论是 grouped CV (`scripts/ml/compare_models_cv.py`)**
> 仓库**完全没有 sklearn 依赖** (全树零引用), 故折按 `str(record.path)` 手工构造。
> 统计用**精确 sign-flip 置换检验** (2¹⁰=1024 次枚举, 最小 p=0.00195, 不假设正态)
> + Cohen's `d_z` + Holm-Bonferroni。同 split / 同输入 / 同 loss / 同 50 epoch 预算:
>
> | 模型 | 参数 | val R² (10 折) | val SSIM | val PSNR |
> |---|---|---|---|---|
> | `hybrid` (physics+CNN 残差) | 10,408 | +0.8721 ± 0.0799 | 0.7404 ± 0.0817 | 29.07 ± 2.51 |
> | `physics(n_max=15)` | **135** | +0.8727 ± 0.0867 | 0.7320 ± 0.0704 | 29.21 ± 2.64 |
> | `unet`[16…256] | 7,778,465 | **+0.9004 ± 0.0610** | **+0.8419 ± 0.0618** | **+31.05 ± 2.44** |
>
> 配对差 (physics − unet, 负 = unet 更好): **SSIM −0.110 (`d_z`=−2.49, p=0.0020)**,
> PSNR −1.84 dB (p=0.0039), R² −0.0277 (p=0.0117) ⇒ **U-Net 显著更好, 不是打平**。
> 4 折 leave-one-objective-out 的**最小可达 p = 2/2⁴ = 0.125**, 结构上无法判显著 ——
> 该协议问题最对但功效不足, 要靠**增加目标数**而非增加折数解决。
>
> ⚠️ **physics 的最优是 (n_max, lr) 联合点, 单变量扫描会扫错** (10 折 CV 实测):
> `lr=0.02` 单独用 (n_max=15) **毫无改善** (+0.8719 vs +0.8719 基线),
> 但同一个 `lr` 配 `n_max=20` 就是全表最好 (+0.8803 / SSIM 0.7646) ⇒ 非可加性
> 在**有功效的协议下同样成立**, 贪心坐标下降必然失败。配对检验 R² +0.0076
> (d_z=0.79, p=0.023)、SSIM +0.033 (d_z=1.98, p=0.004)。
> ⚠️ 该配置是**在这 10 折上从 5 个候选里挑出来的** ⇒ 带 winner's curse,
> 只能当"提示性"证据; 要确证需嵌套 CV。保守可选 `n_max=20, lr=0.01`。
> ⚠️ **U-Net 的配置没有过拟合到噪声 split**: 在 10 折下复验 5 个候选
> (ep 50/100 × lr 0.005/0.01 × 宽度 16/24), 原选择 [16…256]/50ep/0.01 仍最优,
> 全部候选 R² 只差 0.010、SSIM 差 0.031, 远在 fold-σ 0.06 内。
>
> ⚠️ **三条被推翻的结论 (勿再犯, 两次都因为噪声来源没查清)**:
> (1) "U-Net R² 高 +0.022" —— 单 seed + **不公平预算** (U-Net 只给 25 epoch, 而物理预测模型第 13 epoch 就收敛)。
> (2) "SSIM 是真实且区间不重叠的差距" —— 同一单 seed; 3 seed 下区间重叠。
> (3) **"两者打平, 无任何指标能区分"** —— **过度纠正**, 是我把 ±0.07 的配比噪声当成了
>     模型差异。真实效应一度被掩盖、又被否认, 两次都源于**未先定位方差来源**。
>     最终结论: U-Net 显著更好 (R² p=0.012, SSIM p=0.002), 且调优 physics 后仍领先。
>
> ⚠️ **hybrid(物理包络 + 零初始化 CNN 残差) 没有补上差距**: 两个协议、5 个指标
> 全部 p ≥ 0.61, 与 physics **统计不可区分**, 却多 45× 参数 ⇒ **不提升为训练入口的
> 模型选项**, 只作为已记录的反面结果留在对比脚本里。
>
> 但两者**用途不可互换**: U-Net 出的是**图像**, 反解成可下发的 SLM 相位是另一个问题;
> 就"预测要下发的修正"而言, 230 参数物理预测模型是唯一能**闭式给出可实现相位**
> (`Σ Z_k B_k`, 230 个可解释弧度) 的, 且参数少 34000×、墙钟少 40%。
> 完整报告见 [`report/zernike_phase2amp/zernike_amp_report/report.md`](../report/zernike_phase2amp/zernike_amp_report/report.md),
> 含 11 节"被推翻结论"存档。旧对比表见
> [`report/zernike_phase2amp/models_cv/unet_comparison.md`](../report/zernike_phase2amp/models_cv/unet_comparison.md)。

0. **拿 `hwdataset` 的 `image` 当 Zernike 物理预测模型的 target 时, 必须校准远场尺度。**
   `_anchored_window` (`transforms.py:440`) 取的是**以 0 阶为中心的 `grid×grid` 窗口**,
   它**不缩放** —— 真语料 `slm_zernike_shaping` 的相机窗口是 248 px, 所以 target 只覆盖
   `64/248 ≈ 26%` 的视场。而 FFT 输出若取 `far_field_padding=1` (整孔衍射全展), 预测与
   target 的**角尺度差约 4 倍**, 逐像素不可比, 实测 `Z=0` 时 **R² = −0.20 (比直接预测均值还差)**。
   修法: `far_field_padding = 10` + **中心裁剪**回 `grid×grid`。128 张真样本、`n_max=4`、
   `Z=0` 时的 R² 实测: `pad=1 → −0.20`, `4 → −0.12`, `8 → +0.33`, **`10 → +0.46`**,
   `12 → +0.53`, `16 → +0.49`, `20 → +0.18` —— 明显的**内点最优**, 不是"越锐越好",
   所以它是真正的物理标定。训练后 (25 epoch) **R² = +0.63, PSNR = 22.7 dB**, 且
   **`max|c|` 从失控的 1.77 rad 降到 0.165 rad** —— 标定对了之后小系数就能解释数据,
   不再需要靠巨大相位硬凑 loss。换 family (即换 `fov_px`) 必须**重跑这个 sweep**。

   ⚠️ `ZernikeAmpConfig.far_field_padding` 的 CLI 默认值必须**从 dataclass 读**
   (`train_amp.py::_build_parser`), 不要写字面量: 曾因 CLI 写死 `default=1` 静默覆盖
   标定值 10, 每轮都从 R² = −0.38 起步。

   ⚠️ **逐项单变量扫描的结果不可叠加。** 实测 `far_field_padding=12` 与 `lr=0.1`
   各自单跑都比默认好, 但与 `n_max=11` 组合后反而都变差 (R² 0.794 → 0.753)。
   贪心逐坐标下降在这个问题上不成立 —— **改完必须联合复测**。
   另: `optimizer` 杠杆曾因 `train()` 里硬写 `torch.optim.Adam` 而三个选项
   逐位相同 (白测), 现已改为转发 `cfg.optimizer`; 注意 `weight_decay=0` 时
   `adamw` 与 `adam` **本就等价**, 不是 bug。

1. **相位必须用相干 (复数相量) 块平均, 不能用算术平均。**
   语料里的相位**已被 writer 卷到 [0, 2π)** (整块面板实测 max 6.2605 < 2π = 6.2832)。
   对被卷绕的相位取算术均值, 每一格都偏向 ~π, 结构被完全抹掉; 实测算术均值给
   `[2.82 3.32 2.78 3.29 …]` 而相干恢复 `[0.90 -2.97 1.08 2.98 …]`。
   **输出 `(cos, sin)` 而非 `(angle, contrast)`**: `arctan2` 在 ±π 有分支切口,
   会让 CNN 面对一个输入不连续面。

2. **填充必须填在 cos/sin 上, 不能填在相位上。**
   先给相位补 0 再取 cos, 填充区变成 `cos(0)=1` —— 一个指向 +Re 的单位相量,
   会在**所有边界块**引入约 **+0.42** 的伪 DC 项 (实测 0.414 vs 正确 -0.007)。
   `pupil_phase_to_grid` 正是对 cos/sin 各自补 0。

3. **ROI 默认半径是 500 px, 不是照明半径 450。**
   语料里指令相位的最宽结构达 **半径 479 px** (`slm_pib_*`, sidecar `zernike_radius=480`),
   占可用记录的 69%。450 会整族裁掉。1000×1000 ROI 落在 1200×1920 面板内
   (行 100..1099, 列 460..1459)。**且必须先裁后降采样**: 整面板降到 64 格时
   每格 ~19×30 px, 500 px 光斑只占 ~53×34 格, 其余全是平场本底。
   裁剪框按样本的非零 bbox 自适应是**错的** —— 那会让同一个格子在不同样本里
   对应不同的物理尺度, 卷积网格的对齐就毁了。

4. **`_img` 的 dtype 不是量纲。** 语料同时有 `uint8` / `float32` / `float64` 帧,
   但逐帧实测最大值是 255 (uint8) / 239.6 (float32) / 100.9 (float64) —— **同一个
   0–255 探测器量纲**, 全库无一处超过 255。所以 `/255` 对所有 family 都正确,
   `[0,1]` 截断实际从不触发。把浮点帧当成已归一化会把它压缩最多 255 倍。

### 其他已定决策

- **FOV 不统一, 这是已知口径而非 bug。** 各 family 的相机窗口真的不同
  (64 px `region=32` / 248 / 320 / 1944 px 整幅传感器), 所以同一个输出格对应不同的
  物理角尺度。**不做全局 resize 去掩盖它** (会破坏 0 级对齐), 而是把 `fov_px`
  随样本返回, 供使用方按 family 过滤或分头训练。
- **默认 `image_mode="abs255"`, 不做 peak 归一。** 曝光是模型**输入**, 目标里必须
  保留编码它的绝对亮度。
- **`exposure_ms` 走 log10 再标准化。** 原始线性值跨 2.9 个数量级 (0.1–80 ms),
  直接喂小 CNN 不合适; 统计量在 `__init__` 里由 index 一次算出, 不做逐样本估计。
- **`len(_c)==36` 二义性按 family 裁决。** 36 既是三角 Zernike 数 (n_max=7) 又是
  完全平方 (6×6)。`_FREEFORM_FAMILIES` (`slm_gsnet_square` / `sim_calib_abba`) 里
  平方优先, 其余 family 三角优先 —— 否则 `slm_pib_online` 的 192 条 (占 1.7%)
  会被按完全不同的基重建。
- **排除 `bench_stability` (42 条)。** 平场漂移序列没有指令相位; 用全零相位混进去
  会把"指令平场"和"无指令"混为一谈, 污染 `相位 → 画面` 的监督映射。
- **`FileGroupedSampler` 不是优化而是必需。** 每个 pickle 整体读取, 随机采样会让
  几乎每次 `__getitem__` 都重新反序列化最多 3.42 GB; 按文件分组后每轮每文件只读一次。
- **Windows `spawn` 契约。** `__getstate__` 必须丢掉 LRU 与全部派生数组, 且 `self`
  上不得持有任何打开的文件句柄; 否则每个 worker 都会通过管道复制一份几百 MB。
- **分层说明 (有意为之)。** 本包 import `ao_shaping.runners.gsnet_offline` 的纯函数
  (`infer_n_max` / `reconstruct_pupil_phase_rad` / `pupil_phase_to_grid` /
  `farfield_to_grid`) 而非复刻 —— 本仓库已多次被"重复实现漂移"咬伤。
  `gsnet_offline` 只依赖 `ao_shaping.utils.*`, 且 `runners/__init__.py` 用 PEP-562
  惰性 `__getattr__`, 因此无循环。

### 用法

```python
from ml.hwdataset import (
    build_hw_index, build_hw_dataloader, prepare_hw_cache,
    HwPhaseImageDataset, MaterialiserConfig,
)

index = build_hw_index(index_cache="data/hw_index_cache.json")   # 35 s 首次, 0.09 s 之后

# 一次性预建派生网格缓存, 之后每轮省 212 s Zernike 反演 + 35 s 读盘 (实测 317×)
prepare_hw_cache(index, config=MaterialiserConfig(grid=64))

loader = build_hw_dataloader(
    index,
    config=MaterialiserConfig(grid=64),   # 输入/输出都是 (1, 64, 64)
    batch_size=64,
    num_workers=4,
    use_cache=True,                       # 默认已为 True; 无缓存时逐条回落直接路径
)
for batch in loader:
    phase = torch.cat([batch["phase_cos"], batch["phase_sin"]], dim=1)  # (B, 2, 64, 64)
    exposure = batch["exposure_log10"]                                  # (B, 1)
    target = batch["image"]                                             # (B, 1, 64, 64)
```

```bash
# 语料统计 / 抽样自检 (JSON cache 已建时是瞬时的)
python -m ml.hwdataset.inspect --index-cache data/hw_index_cache.json --sample 5

# 预建派生网格缓存 (消除每轮 212 s Zernike 反演 + 35 s 读盘; 之后 loader 自动读取)
python -m ml.hwdataset.inspect --index-cache data/hw_index_cache.json --build-cache
```
