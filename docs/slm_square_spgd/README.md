# SLM 方形光斑 SPGD 整形 (`slm_square_runner`) — 经验总结与故障分析

> 记录 `spgd-square` / `slm_square_runner` 的完整分析、硬件实测、缺陷根因与修复，
> 供后续开发与评审参考。
>
> 分析日期: 2026-09-10 · 硬件: Santec SLM200 #1 (SN 22030108) + Daheng MER2-507-23GM NIR (cam_id=0)
> 相关文件:
> - `src/ao_shaping/runners/slm_square_runner.py` (CLI)
> - `src/ao_shaping/optimizer/wfless/slm_square_shaping.py` (核心算法)
> - `src/ao_shaping/utils/pattern_helper.py` (`_zernike_to_uint16`)
> - `src/ao_shaping/drivers/slm/santec_slm200.py` (`create_phase_from_array`, `display_data`)

---

## 1. 任务目标

把 CCD 上的光斑整形成一个**均匀的 20×20 像素、中心与现质心相同的正方形**。

入口（等价）:

```powershell
python src/ao_shaping/main.py spgd-square --target-side 20 --center shape
python -m ao_shaping.runners.slm_square_runner --target-side 20 --center shape
```

---

## 2. 结论

**原版 `slm_square_runner` 无法完成该任务。** 经代码分析 + Oracle 复核 + 硬件实测确认，
存在 **5 个阻断性问题**。修复其中 4 个 bug 后，目标框能正确落在光斑上并显著变亮，
但**模型无关 SPGD 收敛到散斑**，仍非干净方形（见 §5、§6）。

几何参数层面其实"看似支持"（`--target-side 20` → 框恰好 20×20；`--center shape`
用 argmax+质心定位现质心），失败在算法与物理层面。

---

## 3. 发现的阻断问题（含证据）

| # | 问题 | 位置 | 严重度 |
|---|------|------|--------|
| B1 | Zernike n≤4（15 个低阶模）是圆对称光滑基底，**物理上无法合成方形远场**（方形需要 2D-sinc 型近场 / 高频边角） | 架构 | 阻断 |
| B2 | `_zernike_to_uint16` 用 **min-max 归一化** `(p-pmin)/(pmax-pmin)*1023`，而非 `mod 2π` 的弧度→灰度转换 | `pattern_helper.py:459-461` | 阻断 |
| B3 | 相位路径**绕过 SLM 驱动** `create_phase_from_array`（缺 2π=993 灰度标定、波前矫正、LUT） | `slm_square_shaping.py` | 阻断 |
| B4 | SPGD 梯度目标只用 `-CV`，**不含能量项** → 优化器通过**清空目标框**来降低 CV | `slm_square_shaping.py` (原 744 行) | 阻断 |
| B5 | **硬件实测新增**: ROI 窗口重置后目标框偏离光斑（epoch-0 框内 `mean_b=0.01`，几乎无光） | `slm_square_shaping.py` | 阻断 |

### B2 证据：相位尺度不变（参数化退化）

```python
import numpy as np
from ao_shaping.utils.pattern_helper import PatternHelper

ph = PatternHelper(resolution=(1920, 1200), bits=10)
g1 = ph.generate_zernike_polynomial(n_max=4, coefficients={(2, 0): 1.0})
g4 = ph.generate_zernike_polynomial(n_max=4, coefficients={(2, 0): 4.0})
assert np.array_equal(g1, g4)          # True —— 系数 ×1 与 ×4 生成逐字节相同图案
assert (g1.min(), g1.max()) == (0, 1022)  # 图案恒占满 10-bit 全域
```

后果：整个系数向量**尺度不变**，只有比例有意义 → SPGD 的 `clip(-5,5)`、`delta`
失去物理意义，代价景观存在梯度消失的"脊"。

### B3 证据：正确转换在驱动里

`SantecSLM200.create_phase_from_array` (`santec_slm200.py:1315`):

```python
grayscale = phase_rad / (2 * np.pi) * max_grayscale   # max_grayscale=993 @1064nm
# 之后叠加波前误差矫正 + 相位→灰度 LUT
```

而 `gs_square_runner` / `diff_shaping_runner` 走的正是 `slm.create_phase_from_array(result.phase)`
（`diff_shaping_runner.py:729`），SPGD 路径完全绕开了它。

### B4 证据：优化器"清空"目标框

设备实测 epoch-0 起 `cost=-CV` 单一目标下，优化器把能量移出 20×20 框，
框内接近均匀（暗）→ CV 降低，但 `EE→0`。修复为质量分（CV+EE+AR）后框内恢复有光。

### B5 证据：窗口重置后框位置错误

`reset_window(center, (300,300))` 在光斑靠近帧边时 offset 被 clamp，但返回的
`(width//2, height//2)` 并非真实光斑位置；原代码直接用该返回值当框中心 →
框与光斑错位（实测 epoch-0 `mean_b=0.01`，而窗口内 `max_brt=202`）。

---

## 4. 离线可行性验证（FFT 物理模型）

用 2f 傅里叶光路模型 `CCD = |FFT(入射振幅 · exp(i·SLM相位))|²`，复用 runner 的
精确相位/代价/优化器，扫描基底与阶数（目标框 20 px，中心=初始 argmax）：

| 入射光束 | 基底 | 最优 CV | 最优 EE | 形状 |
|---|---|---|---|---|
| Gaussian | Zernike n=4 | 0.264 | 0.543 | 圆斑 |
| Gaussian | Zernike n=8 | 0.292 | 0.579 | 圆斑 |
| Flat-top | Zernike n=4 | 0.379 | 0.526 | 圆斑 |
| Gaussian | 自由相位 G=16/32 | ~0.30 | ~0.5 | 圆斑 |

**结论**：无论 Zernike 阶数还是自由相位，模型无关 SPGD 都收敛到圆斑/散斑；
方形需要模型化相位恢复（GS / 可微传播）。

---

## 5. 硬件实测结果

| 指标 | 原版 (100 ep) | 修复版 freeform (120 ep) |
|------|--------------|--------------------------|
| quality | 0.078 | **0.270** |
| CV (20×20 框) | 0.825 | **0.251** |
| EE（框内能量占比） | 0.002（空框） | 0.047 |
| 框内 mean_b | 0.34 | **104（已点亮）** |
| 速度 | 1.51 s/epoch | 1.6 s/epoch |
| 形状 | 能量被偏折出框 | 散斑团 |

两次运行均**未触发** AGENTS.md 记录的 SLM+相机闭环挂起。
产物: `data/slm_square/20260910_203509/`（原版）、`data/slm_square/20260910_204311/`（修复版）。

**修复有效**：B5 修复后框正确落点（mean_b 0.34→104）；B2/B3/B4 修复后 quality 提升 3.5×。
**仍不足**：模型无关 SPGD 高维（576 维）收敛慢、落到散斑局部最优，EE 仅 4.7%。

---

## 6. 已实施修复

`src/ao_shaping/optimizer/wfless/slm_square_shaping.py`
- 新增 `_zernike_phase_radians` / `_freeform_phase_radians`，相位统一为 **mod-2π 弧度**，
  经 `slm.create_phase_from_array` 转灰度（修 B2/B3）。
- 新增 `basis`（`"freeform"` 默认，SLM 原生逐像素自由度，能表达方形；`"zernike"` 保留）
  与 `phase_grid` 参数（修 B1 的可用性）。
- 梯度目标改为**质量分**（CV+EE+AR）而非 `-CV`（修 B4）；freeform 的 lr/delta 缩放 0.1。
- 窗口重置后**在窗内重新检测质心**作为目标框中心（修 B5）。

`src/ao_shaping/runners/slm_square_runner.py`
- 新增 `--basis [freeform|zernike]`（默认 freeform）、`--phase-grid`（默认 24）。

---

## 7. 经验教训（可复用结论）

1. **相位生成必须走驱动**：任何"弧度相位→SLM 灰度"都必须经
   `SantecSLM200.create_phase_from_array`（2π=993 + 矫正 + LUT），
   **禁止**用 `PatternHelper._zernike_to_uint16` 的 min-max 归一化。
2. **低阶 Zernike 不能做方形**：方形整形需要 SLM 全像素自由度（GS / 可微 / 自由相位），
   Zernike 仅适合低阶像差补偿与圆对称整形。
3. **代价函数必须含能量约束**：只优化 `-CV` 会让优化器"清空目标框"来降低 CV；
   必须联合 EE（或对框内平均亮度设下限）。
4. **模型无关 SPGD 不适合高维相位恢复**：硬件每步 ~1.6 s，576 维 2 点估计收敛慢、
   易落散斑；方形整形优先用模型化方法（离线算相位再下发，无逐帧硬件写入）。
5. **ROI 窗口重置后必须重新定位光斑**：`reset_window` 返回值在光斑靠边时不可信，
   应在窗内图像上重新 argmax/质心作为目标框中心。
6. **0 级定位用 argmax**（AGENTS.md 既有约定）：光轴落点不在帧几何中心。

---

## 8. 推荐方案

要得到干净的 20×20 方形，使用项目已验证的**模型化引擎**：

- `python src/ao_shaping/main.py diff-shaping --target-shape square --target-px 20 ...`
  （PyTorch 可微传播，文档记载 CV<0.1）
- 或 `python src/ao_shaping/main.py gs-square --target-px 20 ...`（Gerchberg-Saxton）
- 或 `python src/ao_shaping/main.py diff-beam --algorithm backprop --target-shape square
  --target-px 20 --use-hardware -e 200 --cam-exposure-us 1200 --auto-exposure`
  （可微 backprop, **已在 SLM200+大恒真机完成 20×20 方形成形, Corr 0.8776,
  footprint 20×22px, 曝光无关归一化 target**; 详见
  `docs/diff_beam/README.md`, 参数扫描: 200 步最优, 400 步无增益）

若要保留 `slm_square_runner` 的 CLI，可将其相位合成接到 `train_beam_shaping` /
GS 引擎（保留 `--target-side` / `--center`），而非依赖模型无关 SPGD。

---

## 9. 复现命令

```powershell
$env:PYTHONPATH = "src;libs"

# 原版（有缺陷）—— 目标框为空，EE≈0
python -m ao_shaping.runners.slm_square_runner `
    --target-side 20 --center shape -e 100 --exposure-ms 0 --save-best-image

# 修复版 freeform —— 框已点亮，但仍为散斑
python -m ao_shaping.runners.slm_square_runner `
    --target-side 20 --center shape -e 120 --exposure-ms 0 `
    --basis freeform --phase-grid 24 --save-best-image
```
