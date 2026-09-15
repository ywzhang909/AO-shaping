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
- **方形光斑 SPGD 不能用低阶 Zernike (n≤4)**: Zernike 模态是圆对称平滑基, 物理上无法合成方形远场 (需要 2D-sinc 类近场/高频)。
- **SPGD 目标函数不能只用 `-CV`**: 无能量项时优化器会清空目标盒 (硬件观测 EE→0.002)。目标函数必须包含环围能量。
- **`reset_window()` 返回的中心不可信**: 当光斑靠近帧边缘时 ROI 偏移被 clamp 但返回的 `(w//2, h//2)` 不是真实光斑位置 → 目标框偏移。用 `argmax`/centroid 重新定位**窗口化**图像中的光斑。
- **不要信任 `PatternHelper._zernike_to_uint16`**: 它做 min-max 归一化而非 `mod 2π` rad→灰度, 使图案对系数缩放不变 (×1 和 ×4 产生完全相同的字节)。Zernike 系数相位转换始终走 SLM 驱动的 `create_phase_from_array()`。
- **2f Fourier 光路**: SLM 前焦面 125mm → f=125mm 透镜 → CCD 后焦面; CCD 坐标=空间频率; 0 级=帧全局最大 (argmax), 非相机几何中心。
- **不要假设 0 级在相机帧中心**: 在 2f Fourier  benches 中光学轴 (0 级 = 帧全局最大值) 仅在巧合时落在相机中心。观测: 帧中心 (1344,760) vs 0 级 (1441-1443, 705-706)。始终用 `argmax` 定位 0 级。

## SLM 标定

详细标定方法见 [`README.md`](./README.md) 和 [`slm_calibration.py`](./slm_calibration.py)。

详细硬件实测与故障记录见 [`docs/slm_square_spgd/README.md`](../../../../docs/slm_square_spgd/README.md)。
