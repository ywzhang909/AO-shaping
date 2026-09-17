# slm/ - 空间光调制器驱动

空间光调制器 (SLM) 驱动模块, 提供 Santec SLM-200 SDK 封装、相位标定、波前校正等功能。

## 结构

```
slm/
├── santec/                      # Santec 驱动子包 (SLM-200/300 共用)
│   ├── __init__.py              # re-export (Santec, SantecError, 常量, WavefrontCorrection)
│   ├── driver.py                # Santec SDK 权威驱动
│   ├── constants.py             # 驱动常量 (SDK 协议/错误码/模式/范围)
│   ├── slm200_constants.py      # SLM-200 设备硬件常量 (像素/面板/翻转时间)
│   ├── _slm_win.py              # SDK bindings (internal, Windows)
│   └── wavefront_correction.py  # 波前误差校正 (CSV → correction map)
├── slm_calibration.py           # SLM 相位标定 (闪耀光栅法/零级比值法/干涉法/衍射效率法)
├── zernike_slm.py               # Zernike 系数驱动 SLM
├── README.md                    # SLM 响应标定详细说明
└── __init__.py
```

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `Santec` | `santec/driver.py` | Santec 主驱动 (SLM-200/300 共用) |
| `ZernikeSLM` | `zernike_slm.py` | Zernike 系数驱动 SLM |
| `SantecCalibrator` | `slm_calibration.py` | 标定器 |
| `AutoExposureController` | `slm_calibration.py` | 自动曝光控制器 |
| `WavefrontCorrection` | `santec/wavefront_correction.py` | 波前误差校正 |

## CSV I/O 统一格式 (2026-09-16 重构)

SLM 相位/灰度 CSV 的导入导出**唯一实现在 `WavefrontCorrection`**（底层工具类，`driver.py` 委托调用，无循环导入）。`Santec.load_gray_from_csv` / `csv_to_phase` / `save_phase_to_csv` 均为薄 wrapper，保证：

- **格式一致**: 灰度 CSV 与矫正 CSV（`libs/SLM_DLL_ver.2.51/Wavefront_correction_Data/*.csv`）共用同一 `Y/X` 标题 + 行列索引格式，可互相 round-trip。
- **单一真源**: 所有尺寸校验、`Y/X` 标题校验、值范围校验、行列索引剥离/恢复逻辑只在 `WavefrontCorrection` 实现一次。

```python
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.drivers.slm.santec.wavefront_correction import WavefrontCorrection

# 三种等价调用（输出字节级一致）
gray = Santec.load_gray_from_csv(path)          # → WavefrontCorrection.load_gray_from_csv
phase = Santec.csv_to_phase(path)                 # → WavefrontCorrection.csv_to_phase
Santec.save_phase_to_csv(phase, dest)            # → WavefrontCorrection.save_phase_to_csv
```

**矫正灰度偏移导出契约 (2026-09-16)**: 离线矫正导出 (`tools/slm/slm_zernike_response.py` 的
`--export-correction`) 走 `WavefrontCorrection` 的静态方法
`correction_gray_offsets(phase_rad, max_grayscale=None)` → `save_gray_correction_csv(gray_offsets, filepath)`：

- **满量程 2π = 1023** (`slm200_constants.get_max_grayscale()`)；**严禁**用波长相关
  `two_pi_gray`（如 532nm→998）——官方软件按 1023 换算矫正偏移，用 998 会把幅度
  缩放 `1023/998 ≈ 1.025×`。
- **不烘焙 shift**：只写面板坐标逐像素偏移图（Zernike 图案居中、未经平移）；平移由
  官方软件 shift 设置 / `Santec.apply_shift` 在显示时单独应用。侧车 JSON 带
  `shift_included: false`。
- 数值 = `rint(mod(φ/(2π)·1023, 1024))` 再 `mod 1024`（环绕边界 1024≡0，循环域最近邻
  量化），输出严格 `0..1023` uint16。
- `tools/slm/slm_zernike_common.save_driver_correction_csv` 是薄 wrapper，唯一实现如上述。

> 详见 `tests/ao_shaping/drivers/slm/test_gray_csv.py` 与 `test_correction_csv.py`。

## Santec 接口

| 方法 | 说明 |
|------|------|
| `open()` | 打开 SLM 连接 |
| `close()` | 关闭 SLM 连接 |
| `set_wavelength(wavelength)` | 设置工作波长 |
| `write_phase(phase, memory_number)` | 写入相位数据 |
| `display_memory(memory_number)` | 显示内存中的相位图 |
| `display_data(phase)` | 直接显示相位数据 |
| `set_grayscale(gs)` | 设置灰度值 |
| `save_phase_to_csv(phase_rad, destination)` | **导出弧度制相位 CSV**（保留 `Y/X` 行列索引；目标可为路径、`BytesIO` 或文本流） |
| `shift_phase(phase, sx, sy)` | **纯函数**平移 (static; 平移数学唯一实现, 空白填0) |
| `apply_shift(sx, sy, *, wait_time_s, save_config)` | **平移 + 自动重绘当前显示相位** (绝对定位不累积, 槽轮换, 配置保存) → 驱动层统一入口, 与多SLM控制器"应用平移"按钮一致 |

## SLM 故障排查 (2026-09 固化)

`tools/slm/slm_diagnose.py` 三步自检定位"面板不调制光"类故障:

1. **freeze (面板冻结)**: flat/全屏光栅/上下半屏光栅写**轮换内存槽** → 帧必须随图案变化 (≥2/3 帧不同才 PASS)。
2. **modulate (调制能力)**: `set_grayscale` 0→1023 扫描, 0 级桶能量须有 ~993 灰度周期 (相对变化 >15% 才 PASS); 无周期 ⇒ 无振幅耦合 ⇒ 面板不调制。
3. **linearity (到达光强)**: 曝光 ×4、×20, 峰值亮度须增长 (>2× 才 PASS); 恒定峰值 ⇒ 到达相机光强比已知 ~0.02ms 近饱和基线弱 >100×。

## 已知坑 (etched in drivers)

- **DVI 模式 `open()` 会挂起**: `video_mode=1` 可挂 120s/300s, 且挂起后 memory 模式也挂直到**物理断电**。诊断脚本永不自动切 DVI。
- **diff-shaping 硬件闭环挂起 (2026-09-08)**: 日志止于 `成功打开SLM #1`, 之后首次 `get_numpy_image()` 前无限阻塞 (900s 强杀; 分步探针脚本同挂)。`WaitImageV3` 原生等待由 SDK 内部驱动, 不受 Python 超时保护。强杀不会走 finally close → SLM 控制器可能残留异常态; 重跑前确认无残留进程, memory 模式 open() >数秒无日志则物理断电重置 (与 DVI 挂起同一处置)。见 `diff_shaping_runner.py` SLM 连接段注释。
- **`get_displayed_memory_number` 报错码 1 正常**: set_grayscale 模式无内存槽显示, SLM_Ctrl_ReadDS 返回 1 是合理行为, 非故障。
- **同内存槽连续 `display_memory` 是 no-op**: 固件对"正在显示的同一槽位"的 `display_memory(slot)` 不刷新 LCOS 面板 (第二次写入的新相位不会上屏)。驱动已在 `display_memory()` 内置 warning: 检测到连续同槽位调用时 `logger.warning` 提示轮换槽位 (`display_data()` 内部自动轮换 127 槽, 不受影响; 工具类如 `gray_response._display_rotate_slot` 已自带轮换)。
- **已缓存相位重写必须走 raw 路径 (`_write_to_memory`/`apply_shift`), 严禁再传 `write_phase`**: 缓存的显示相位 (`get_displayed_phase()`) 已包含底相位/矫正叠加, `write_phase` 会再次叠加 → 矫正被应用两次产生错误图案。平移后重写由 `apply_shift` 内部经 `_write_to_memory` 完成; 其他需要重写缓存相位的场景请直接调 raw 写入，勿用 `write_phase`。
- **SLM flat-phase 严禁走 `create_phase_from_array()`**: 该函数对输入按 `mod 2π` → 灰度转换 (rad/2π × 1023), 导致 uint16 灰度值被静默破坏。扁平相位必须用 `np.full((h,w), gray, dtype=np.uint16)` 直接发送。
- **相位生成 raw-only 契约 (2026-09)**: 所有相位生成函数只需产生 **raw 未包裹弧度**, 严禁自行 `mod 2π`; 唯一的 wrap 点在 `create_phase_from_array()` 的弧度→灰度转换 (L1382)。弧度→灰度统一经 `utils/slm_utils.phase_to_slm_grayscale(phase, slm=slm)` (传 SLM 委托驱动管线; `slm=None` 时纯数学 fallback)。
- **方形光斑 SPGD 不能用低阶 Zernike (n≤4)**: Zernike 模态是圆对称平滑基, 物理上无法合成方形远场 (需要 2D-sinc 类近场/高频)。
- **SPGD 目标函数不能只用 `-CV`**: 无能量项时优化器会清空目标盒 (硬件观测 EE→0.002)。目标函数必须包含环围能量。
- **`reset_window()` 返回的中心不可信**: 当光斑靠近帧边缘时 ROI 偏移被 clamp 但返回的 `(w//2, h//2)` 不是真实光斑位置 → 目标框偏移。用 `argmax`/centroid 重新定位**窗口化**图像中的光斑。
- **不要信任 `PatternHelper._zernike_to_uint16`**: 它做 min-max 归一化而非 `mod 2π` rad→灰度, 使图案对系数缩放不变 (×1 和 ×4 产生完全相同的字节)。Zernike 系数相位转换始终走 SLM 驱动的 `create_phase_from_array()`。
- **2f Fourier 光路**: SLM 前焦面 125mm → f=125mm 透镜 → CCD 后焦面; CCD 坐标=空间频率; 0 级=帧全局最大 (argmax), 非相机几何中心。
- **不要假设 0 级在相机帧中心**: 在 2f Fourier  benches 中光学轴 (0 级 = 帧全局最大值) 仅在巧合时落在相机中心。观测: 帧中心 (1344,760) vs 0 级 (1441-1443, 705-706)。始终用 `argmax` 定位 0 级。

## 平移标定 (shift_x / shift_y) — defocus 零点法 (2026-09-15 实测固化)

`shift_x`/`shift_y` 把相位图案在面板上平移, 用于让**图案中心对准光束光轴**。标定原理:

- 图案平移 `(sx,sy)` 后光束感受到 `P(b−s+ξ)` (`b` = 光束光轴在 SLM 坐标中的偏移, 未知)。
  对 defocus `P = D·u²` 有梯度 `∝ 2D(b−s)` → **WFS 读出的 tip/tilt 关于 shift 线性,
  零点即 `s = b`**(图案中心与光束对齐)。
- 判据必须用**相对纯平的"附加"倾斜**: `‖z_tilt(defocus@shift) − z_tilt(flat)‖` —— 系统
  本身有静态倾斜 (实测 flat 下 tilt ≈ −0.14λ), 绝对归零是错的判据。

**实测结果** (SLM#22030102 + WFS M01219666, 532nm): **`shift_x=106, shift_y=40`**
(附加倾斜 0.854λ → 0.0245λ, **降低 97.1%**; 原值 (60,0) 附加倾斜 0.46λ)。
标定脚本: `src/ao_shaping/tools/slm/slm_shift_calib.py`。

### 三个必须遵守的参数约束 (踩过的坑)

| 约束 | 原因 |
|------|------|
| **defocus 幅度 A 必须足够大** (实测 A=20 rad @ R=600) | A=2 时响应被 `(r_beam/R)²` 压制到噪声级, 扫描完全看不到趋势 (v1 失败根因) |
| **Zernike 半径 R 必须 > 光束半径** (实测光束在 SLM 上半径 ≈200px ≈1.6mm) | R=200 响应最强但一平移就裁切光束; R=600 可平移 ±400px 不裁切 (光束尺寸由 R 扫描诊断得出: R=200 时 Δdefocus 最大 0.163λ) |
| **shift 必须限制在 ±500** | defocus 盘中心 `(960+sx, 600+sy)` 超出 1920×1200 面板后光束几乎看不到图案 (v2 用 sx=1173 → 数据全废) |

### ⚠️ SLM 轴 ↔ WFS 轴存在 90° 交换

实测 (本机中继光路): **SLM-x 平移驱动 WFS Noll3 (y-tilt)**, **SLM-y 平移驱动 WFS Noll2 (x-tip)**。
X 粗扫 tip 几乎不变 (+0.39→+0.22) 而 tilt 强线性 (+2.72→−1.50); Y 粗扫反之 (+2.41→−1.83)。
→ 标定/对位时**不要假设 SLM 轴与 WFS 轴对应**, 用轴无关判据 `‖Δz_tilt‖` 或同时测两分量。

## Zernike 响应矩阵标定 (Zernike 模式法波前矫正) — 2026-09-15 实测

`tools/slm/slm_zernike_response.py`: 在标定后的 shift 下对每个 SLM Zernike 模式做 ±A 推拉扰动,
测 WFS 的 Zernike 读数增量, 构建 `matrix[wfs_coeff, slm_mode] = Δ(WFS)/Δ(SLM幅度)` 并保存为
项目标准 h5 格式 (`optimizer/wf/zernike_response_matrix.py` 的 `save/load_zernike_response_matrix`,
含 `pinv_matrix` 逆矩阵 → 矫正控制律 `c = pinv @ w`)。

**实测结果** (SLM#22030102 + WFS M01219666, 532nm, shift=(106,40), R=250px, A=5 rad):

- `data/zernike_response_matrix/zm_slm22030102_wfsM01219666_532nm_*.h5` — shape (66, 14),
  **强对角** (14/14 模式主导项为同索引 WFS 系数), 条件数 **5.12**, 平均方差 3.8e-6
- 闭环反解验证: 合成像差 defocus[5]=0.30 + coma[9]=−0.20 → 反解 SLM[5]=−0.62, SLM[9]=−0.52,
  残差降低 87%
- 副产物 `*.json` 存全量原始读数 (baseline + 每模式响应)

> ⚠️ **索引约定**: 矩阵行 = DLL 系数索引 (顺序 m 枚举, **非标准 Noll**), 详见
> `docs/thorlab-wfs/agent.md` 与工具的模块 docstring。跨光束半径 (R=250px) 标定, 光路调整后需重标。

### ⚠️ 三个已实测踩过的坑 (2026-09-15, 闭环仅 5.8% 的根因)

| 坑 | 后果 | 处置 |
|---|---|---|
| **矩阵列来自不同 Zernike 半径** | 矫正相位按单一 R 生成 → R=300 标定的模式被按 R=200 加载, 相位放大 `(300/200)²=2.25×` (R=400→4.0×) → 反解失真 | 矩阵**强制统一半径**; 多半径仅作诊断; 在覆盖度容差内取**最小** R |
| **反解时保留 piston** | 矩阵行 0 = DLL[1] = piston 是 WFS 参考偏置 (不可矫正), 但该行数值很大 (平移引入) → 通过 `pinv` 污染整个解 | `c = pinv(M) @ w` 前 **`w[0] = 0`** |
| **只用光斑有效比做门控** | `wfs_validity` 抓不到"光斑正常但 LSF 拟合崩溃" (实测 `\|resp\|` 暴涨到 4288) | 加**逐点幅度合理性剔除**: 同 (模式,R) 组内 `\|resp\|` 偏离中位数 >3× 则剔除 |
| **µm 与 λ 混用** | WFS `get_zernike()` 返回 **µm**; 矩阵若用原始 µm 构建而矫正的 `w` 用 λ, 反解系数被放大 **1/0.532 = 1.88×** | 一律经 `slm_zernike_common.um_to_waves()` 换算; `units` 字段写入 h5 |
| **λ 与 弧度混用** | 反解系数是 **λ(波长)** 但 `make_phase`/`generate_zernike_polynomial` 收 **弧度** → 加载相位缩小 **2π = 6.28×** | 加载前 `× 2π`; 闭环实测 13.8% → **42.1%** |
| **产物缺设备参数** | 无法复现/审计 (波长、2π灰度、温度、曝光、pupil 等) | 统一经 `slm_zernike_common.collect_device_info()` 记录进 h5 的 `device_config.device` |
| **矩阵含零列时直接 pinv** | 线性度门控置零的列使 `np.linalg.cond` 爆到 1e18 (数值秩亏) | 用 `slm_zernike_common.safe_pinv()` (仅对有效列求逆) 与 `effective_cond()` |

**离线验证** (用 raw_scan 重建矩阵): 统一 R=300 → 13/14 列, cond=10.89, 反解残差降 **91.7%**;
混合半径 (旧行为) → cond=65.33, 反解残差降 **70.9%**; 统一 R=400 → 14/14 列但 cond=52.33
(覆盖更全但条件数差 5×)。

## SLM 标定

详细标定方法见 [`README.md`](./README.md) 和 [`slm_calibration.py`](./slm_calibration.py)。

详细硬件实测与故障记录见 [`docs/slm_square_spgd/README.md`](../../../../docs/slm_square_spgd/README.md)。

## GUI CSV 相位加载与导出

`multi_slm_controller.py` 的"从CSV加载相位"按钮走标准三步管线（详见 `docs/slm/slm_gui_manual.md` §5.7）：

1. `slm.load_gray_from_csv(path)` — 驱动层格式校验（Y/X 标题、尺寸=PANEL_RES、值 0..1023）
2. `Santec.csv_to_phase(path)` — 灰度 → 弧度制相位（静态，无硬件转换）
3. `slm.display_phase(phase_rad)` — `create_phase_from_array()` 完成 弧度→灰度 + 矫正 + LUT + 平移，写入内存槽

"导出相位"使用 `Santec.save_phase_to_csv(phase_rad, destination)`：

- 输入必须是**弧度制**相位数组，shape 为 `(PANEL_RES[1], PANEL_RES[0])`，即 `(1200, 1920)`；函数拒绝非面板尺寸及 NaN/无穷值。
- 输出保留 Santec 索引格式：首行首列为 `Y/X`，随后是列索引 `0..1919`；每行首列为行索引 `0..1199`，数据区为 float 弧度值。
- `destination` 可为文件路径、`io.BytesIO` 或文本流；写入路径不存在时自动创建父目录。
- GUI 导出前按当前设备的 `_max_gray` 将灰度相位转换为弧度，再调用该函数；导出的弧度 CSV **不能**交给 `load_gray_from_csv()` 或 `csv_to_phase()`，后两者只接受 0..1023 灰度 CSV。需要重新加载时，应读取数据区为弧度并传入 `create_phase_from_array()`。

**注意**: CSV 灰度值必须走 `csv_to_phase`（视为原始灰度），**不能**直接当弧度传入 `create_phase_from_array()`。扁平相位仍须用 `np.full((h,w), gray, dtype=np.uint16)` 直接发送。
