# diff-beam 固定边长方形整形 — 硬件经验记录 (Santec SLM200 + 大恒 CCD)

> `diff-beam` (backprop/gs, `src/ao_shaping/runners/diff_beam_runner.py`) 在
> **SLM200 + 大恒相机** 上做方形整形的实测经验。记录了 2026-09-10 真机验证
> 发现的 **0 级光斑定位 bug 链** (最终以 argmax 收尾) 与 **target 尺寸换算坑**。

## 1. 语义: `--target-px` = CCD 图片空间固定边长

`--target-shape square` + `--target-px <N>` 时, target square 的定义:

- **空间**: CCD 相机图片空间 (不是 SLM 网格 FFT bin!)
- **边长**: 固定 `N` 个 CCD 像素
- **中心**: 实测帧 (flat phase 初始帧) 的 **0 级光斑位置 = 帧全局最大 (argmax)** —
  项目规则 (AGENTS.md): "0-order = frame global max, 永远用 argmax, 不用几何中心"
- **亮度**: 均匀 `1 / n_pixels` (实际填充像素数), 总和精确 = 1
- **loss 对比**: 实测帧 `/ frame.sum()` 归一化后再与 target_ccd 对比

因此目标 square **与曝光/总亮度无关** (曝光只缩放原始帧, 两边同除
`frame.sum()` 抵消), 且中心**跟随当前光束位置** (0 级光斑), 不用帧中心。

推荐命令 (实测基线 1200µs 曝光, 需要自动曝光可加 `--auto-exposure`):

```powershell
$env:PYTHONPATH = "src;libs"
uv run --group ml python -m ao_shaping.runners.diff_beam_runner `
  --algorithm backprop --target-shape square --target-px 20 `
  --use-hardware -e 200 --cam-exposure-us 1200 --auto-exposure
```

等价 CLI: `python src/ao_shaping/main.py diff-beam --algorithm backprop --target-shape square --target-px 20 --use-hardware -e 200 --cam-exposure-us 1200 --auto-exposure`

## 2. 🔴 0 级光斑定位 bug 链 (2026-09-10 真机修复) — 目标方框与实测光斑错位

### 最终结论 (真机验证后)

**0 级光斑位置必须用帧全局最大 (argmax)**。三种候选方法在全帧
(2592x1944, 大恒) 实测对比:

| 方法 | RUN1 (900µs) | RUN2 (1200µs) | 结论 |
|------|-------------|---------------|------|
| **argmax (0 级)** | **(705, 1781)** | **(701, 1780)** | ✅ 真正光斑, 两次运行稳定 |
| corner-max 二值质心 | (712, 1779.7) | (713, 1806.4) | ❌ 偶然近似 (RUN1 差 7px), 高曝光漂移 26px |
| 去背景强度质心 | (863.9, 1329.6) | (870.8, 1437.7) | ❌ 被全帧光晕拖偏 150~450px |

argmax 是唯一可靠锚点: 真机 20×20 方形成形验证 Correlation **0.0000 → 0.8776**,
footprint **20×22px**, target 框中心 (698.5, 1780.5) 与最终光斑 argmax (694,1777)
偏移仅 4.5/3.5px。诊断过程详见下方时间线。

### 时间线 (为什么会错两次)

1. **初版**: 复刻 axis_beam_runner 的 `intellij_center` (corner-max 二值质心 +
   0.4*peak 实心回退)。在 axis 有效 (配合 `--cam-size 160` 开窗), 在 diff-beam
   全帧失效: 杂散光亮点污染二值 mask; 小光斑 (fd~16px) 使实心检查恒 False。
   原始 bug 运行 target 落在 (712, 1780) —— 恰巧接近真实光斑 (705,1781),
   Correlation 0.69, footprint 15x16px (尺寸换算问题, 见 §3)。

2. **第一次"修复" (错误方向)**: 改为**去背景信号强度质心**
   `centroid(clip(frame-bg,0))`, 并把它与 `_record_frame` 的强度质心记录
   (863.9, 1329.6) 对齐当作"真值"。**这是错误的**: 强度质心在这台相机上
   被全帧光晕 (6.9-7.3% 像素非零, 光晕总能量 >> 光斑) 拖偏 150~450px。
   用所谓"记录一致"自我验证, 反而把 target 移到远离光斑的位置 →
   Correlation 0.0000, footprint 20x22px 但 target 框内能量仅 0.0001
   (框在 (870,1437), 真光斑在 (694,1779), 完全没盖住)。

3. **最终修复 (正确)**: `cy, cx = np.unravel_index(np.argmax(signal), signal.shape)`
   —— 0 级光斑 = 帧全局最大。这一次真机验证 Correlation 0.8776, target 框
   覆盖光斑, 信号能量 68~70% 汇入框内。

### 关键教训

> **大恒 CCD 全帧上不要用任何"质心"定位 0 级光斑**:
> 1) **强度质心**被全帧杂散光晕拖偏 (实测 150~450px) —— 光晕像素数
>    量级大 (占帧 7%), 总能量远超光斑; 减去常量背景 (10th percentile)
>    只能去均匀底噪, 去不掉空间结构的光晕。
> 2) corner-max **二值质心**在全帧下时准时偏 (低曝光偶然接近, 高曝光
>    漂移 26px), 阈值依赖光晕强度, 不可靠。
> 3) **argmax (0 级 = 帧全局最大) 是唯一稳定锚点** (两次运行差 ≤4px)。
>    这也是 AGENTS.md 的既有规则。轴上光斑实心时 argmax ≈ 光斑中心。

- Runner 逐帧记录器 `_record_frame` 同步新增 `"spot": [cy, cx]` (argmax) 字段,
  不再用强度质心当光斑位置; `centroid` 保留仅供统计参考 (会被光晕污染,
  勿当光斑位置使用)。
- 测试: `tests/ao_shaping/runners/test_diff_beam_runner.py::TestFixedSideSquare`
  27/27 全绿 (测试帧高斯中心 == argmax, 断言兼容)。

### 补充: 1200µs 平场原始帧质心调试 (2026-09-11, 全帧 2592×1944)

按"重新采集 1200µs 原始光斑 TIFF 图调试质心算法"的要求, 采集了平场
(零相位) 帧 `data/diff_beam/flat_tiff_20260911/flat_1200us.tiff/.npy`
(28.3 万总能量, 帧峰值 255, 72px 饱和)。同帧五种算法对比:

| 方法 | 结果 (row, col) | 距 argmax | 结论 |
|------|-----------------|-----------|------|
| **argmax (0 级)** | **(689, 1776)** | — | ✅ 真光斑 (与 212934 final (693,1779) 差 <5px) |
| 原始强度质心 | (834.3, 1577.8) | 145/200px | ❌ 被 1 灰度层拖偏 |
| 去背景强度质心 | (834.3, 1577.8) | 145/200px | ❌ **与原始完全相同** — 中位数=0, 减背景无效 |
| corner-max 二值质心 (thr=1) | (715.3, 1834.4) | 26/58px | ❌ 1 灰度层到达角落, corner 阈值无法滤除 |
| spots_calc.centroid | (834.0, 1578.0) | 同原始 | ❌ runner 日志参考值 (勿当光斑位置) |

**帧结构量化** (为何中位数去背景失效):

- 96.37% 像素灰度 = 0 (零底噪); **3.51% 像素灰度 = 1, 遍布每一行每一列
  (帧投影非零行/列 = 全帧), 占总能量 62.46%** — 这就是"光晕"主体。
- 中位数 = 0 → `clip(frame - median)` 不改变任何像素 → 去背景强度质心
  与原始质心逐位相同 (834.3, 1577.8)。
- 真光斑信号 (≥10 灰度): 32.4% 总能量; 光斑 120px box 占 37.8% 能量。

**阈值扫描质心 (调试核心结论)**:

```
img > t  质心 (row, col)          能量占比
t=0   (834.25, 1577.84) 100.00%   ← 全帧, 被 1 灰度层拖偏
t=1   (696.99, 1783.23)  37.54%   ← 仅剔除 1 灰度层, 立即收斂
t=2   (695.28, 1778.05)  35.50%
t=5   (694.37, 1775.82)  33.65%
t=10  (694.17, 1775.88)  32.38%
t=50  (693.79, 1776.30)  27.09%
t=200 (693.46, 1776.97)  11.60%
```

**结论**: 质心的唯一失效原因是帧内 ~62% 能量的 **1 灰度噪声层**;
只要 `img > 1` 掩码, 任何强度质心立刻收敛到 (694-695, 1776-1778),
与 argmax (689,1776) 一致 (差异 = 饱和平顶下 argmax 取单像素 vs 全光斑
质心)。**argmax 仍是默认 (零参数、无需选阈值、已验证)**; 若未来要
亚像素质心抗饱和平顶, 用 `img > 1` 掩码后的强度质心即可, 不要用
中位数/百分位背景 (中位数=0 无效)。

## 3. 🟡 target 尺寸换算坑: `side_grid_bins` ≠ 期望的 CCD 边长

### 现象

`--target-px 20` 时配置里:

```
cam_px_per_bin: h=1.259, v=2.015
side_grid_bins: [14.81, 14.81]   (≈ 15 SLM grid bins)
target_size:    15
```

即 20 CCD px 的方形 → grid 空间只有 ~15 bins, 而 20 CCD px 按 (v=2.015)
应该 ≈ 10 bins、按 (h=1.259) ≈ 16 bins —— **三个换算途径互相矛盾**。

### 根因

`side_grid_bins = side * grid_h / crop_h` (crop_h = 1620, 由帧 2592x1944
按 grid 宽高比 1920:1200 裁剪得到), 是**纯比例缩放**, 与光路实际放大率
(cam_px_per_bin) **无关**。而 `--target-px 20` 的语义是 "CCD 上 20px 宽",
grid 上多少 bins 应由 **CCD↔SLM 像素比例** (pixel-scale) 决定, 不是画面比例。

### 影响与建议

- backprop 的 loss 在 **CCD 图片空间**算 (`frame/frame.sum()` vs target_ccd),
  所以优化目标本身是 20x20 CCD px, 尺寸语义正确;
  `side_grid_bins` 只用于日志/绘图/grid 目标生成, 对硬件 loss 影响较小。
- 若要对齐 grid 目标与 CCD 目标的物理尺寸, 应改用
  `side_grid_bins[0] = side_px * grid_h / (frame_h * cam_px_per_bin_v)` 之类
  的物理换算 (对齐 slm_square_spgd / gs-square 的 pixel-scale 思路)。

## 4. 🟡 指标解读: 归一化 MSE/Correlation vs Efficiency

真机 200 steps backprop 收敛后:

```
MSE: 0.000000   Correlation: 0.8776   Efficiency: 0.0761   (归一化对比)
```

- **MSE ≈ 0 / Correlation 0.88**: 归一化后实测帧与 20×20 方形 target 高度相似,
  成形成功。footprint 实测 20×22px 佐证。
- **Efficiency 0.076 很低 ≠ 成形失败**: runner 的 efficiency =
  帧总能量在 target 框内占比, 但该相机帧总能量 **92% 在 1-5 灰度噪声层**
  (数十万低值像素), 真实光斑信号 (>10 灰度) 只占 ~11%! 因此效率被噪声
  分母拉死, 物理上限仅 ~10%。**信号级评估** (框内信号/全帧信号) 才是
  有效指标: 本次实测 final 帧信号能量 68.5~69.6% 汇入 20×20 target 框。
- **建议**: 评估 efficiency 时先做信号掩膜 (>10 灰度或去背景), 不要让
  噪声层主导分母; MSE/Correlation 因归一化天然免疫噪声层, 可放心用。

## 5. 🟡 曝光与饱和

- 基线 `--cam-exposure-us 1200`: flat 帧峰值实测 **216~221** (不饱和),
  final 收敛后能量集中 20×20 → **饱和 147~152px (255)**。
- `--auto-exposure` 生效 (target 峰值 180 ±20%): final 曝光自动降至
  **0.847ms** 但仍残余 ~152px 饱和 —— 自动曝光在优化循环内/最终帧各调
  一次, 但每个曝光台阶后 SLM 相位重写 + 重采样的时序下可能来不及收敛。
- 若要彻底避免饱和: 初始曝光降到 ~800µs, 或增大 `--auto-exposure-tol`。
- 自动曝光参数: `auto_exposure_target_ms` 纯函数 + `_apply_auto_exposure`
  (目标峰值 180 ±20%, boost ≤4x, cut 下限 0.25x/极端 0.1x, 范围 0.02~1000ms)。

## 6. 硬件运行注意事项 (SLM200 + 大恒)

- SLM 内存槽: 相位写入用 2~125 槽随机轮换, 排除当前显示槽 (同 diff-shaping)
- 0 级定位: 帧 `argmax`, 不要用几何中心/任何质心 (见 §2; 2f 光路轴心不居中)
- **看门狗**: runner 对 SLM open / 相机首帧采集有 `--capture-timeout` 超时;
  挂死处置见 AGENTS.md (diff-shaping 硬件闭环挂起条目, 同 SLM/相机)
- 分析: `uv run --group ml python scripts/diff_beam_frame_analysis.py
  --run-dir <dir> --plot` → `frames_overview.png` + `spot_vs_target.png`

## 7. 产物目录 (示例: data/diff_beam/backprop_square_20260910_212934)

```
config.json            # 运行参数 + target_info (标量; 含 argmax 定位的 centroid)
target_ccd.npy / .png  # CCD 空间目标方形 (sum=1)
target_square.png      # grid 空间目标
measured_frame.npy     # 最终实测帧(归一化)
phase_pattern.npy      # 最优 SLM 相位 (grid 空间)
loss_history.npy       # 逐 epoch loss
frame_analysis.json    # 逐帧峰值/总和/质心/足迹/CV
frames_overview.png    # 逐帧快照
spot_vs_target.png     # 原始光斑 vs 目标对照
frames/frame_*.npy+.png + frame_meta.jsonl   # 逐帧原始记录 (含 spot 字段)
```

## 8. 真机复验记录 (2026-09-10, SLM200 + 大恒)

| 运行 | 定位 | Correlation | footprint | target 框覆盖 | 备注 |
|------|------|-------------|-----------|--------------|------|
| 204948 | intellij_center (BUG 初版) | 0.69 | 15×16px | 偶然接近 (712,1780)~光斑 | 尺寸换算偏小 |
| 211600 | 强度质心 (错误修复) | **0.0000** | 20×22px | ❌ 框 (870,1437) vs 光斑 (694,1779) | 框完全没盖住 |
| 212335 | **argmax** | **0.8713** | 20×22px | ✅ 偏移 4.5/3.5px | 147px 饱和 |
| 212934 | **argmax + auto-exposure** | **0.8776** | 20×22px | ✅ 偏移 ~5px | 曝光 1.2→0.847ms, 152px 饱和 |
| 130907 | **argmax + auto-exposure** | 0.7391 | 15×15px | ✅ 偏移 ~5px | **参数扫描: epochs 400**; loss 50 步即收敛 (0.000096) 无增益, 1200µs 下 initial peak=255 饱和削顶 → Corr 低于 200 步基线 |

> **参数扫描结论 (2026-09-11)**: backprop 模拟 loss 在 ~50 步内收敛
> (step1 0.0063 → step50 0.000096 后恒定), `--epochs 400` 对模拟无增益;
> 真机 Corr 0.7391 < 200 步基线 0.8776 (饱和削顶 + 硬件逐次波动)。
> **`--epochs 200` 为最优配置**; 若饱和, 用 `--auto-exposure` (目标峰值 180)。

## 9. 关联

- Runner: `src/ao_shaping/runners/diff_beam_runner.py`
- 目标: `src/ao_shaping/utils/targets.py` (`square_target_from_measurement`)
- 指标: `src/ao_shaping/utils/beam_metrics.py` (`compute_metrics`)
- 测试: `tests/ao_shaping/runners/test_diff_beam_runner.py`
- 质心工具: `src/ao_shaping/utils/spots_calc.py::centroid`
- 同主题 (MiiCam 线): `docs/slm_shaping_diff/readme.md`