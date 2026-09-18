# SLM+CCD 装配对准 (align) 窗口化实现

2026-09-18. 软件化装配辅助: 使用大恒相机 ROI 开窗自动完成 0级/±1级 对准, 取代人工
手动平移/目视对齐。实现在 `src/ao_shaping/tools/slm/calibration.py`
`SLMCCDCalibrator.align()` (CLI: `slm-calib` 的装配步骤, `--skip-align` 可跳过)。

## 背景问题

原 `align()` 只是"提示型"辅助 (打印 0级/±1级 位置与建议平移量), 用户仍需手工对准,
且大视场下 0级 光斑小、视场内还有杂散级次, 人工对准耗时且不稳定。

## 算法 (四阶段)

- **阶段A 全幅检测**: flat 相位下测 0级 质心 `c0` 与 FWHM `f0`; 边缘余量
  `margin = window_margin_factor * f0`。
- **阶段B 周期选择**: 按 `period_candidates` (默认 16/24/32/48/64/96) 升序测 ±1级
  位移, 首个同时满足「0级/+1级x/+1级y 均在视场(留 margin)」且「窗口
  `2*(half+margin)` (下限 `min_window_side`) 不超传感器」的周期胜出; 全部不满足
  → 恢复全幅返回 `no_period`。
- **阶段C 窗口定心**: 每轮把窗口中心移到 0级 (全幅坐标, 边缘 clamp), 窗口大小固定;
  收敛判据 = 下一轮窗口内 0级 与窗口返回中心偏差 `<= center_tol_px`。窗口内 0级
  用 `_moments` 阈值质心重新定位 (**不信任 `reset_window()` 返回的中心**,
  AGENTS.md L592), 也天然抗视场边缘/杂散干扰。
- **阶段D 收尾**: 成功 → **保留窗口**并写 calib keys; 失败 (no_spot/no_period/
  not_converged) → 恢复全幅、清空 align keys、`align_ok=False`。

## calib keys (保存在 self.calib, calibrate()/save() 保留不丢)

| key | 含义 |
|---|---|
| `align_ok` | 是否成功 |
| `align_period` | 选中的闪耀周期 (px) |
| `align_center_full` | 0级全幅坐标 `(cy, cx)` |
| `align_window_center_xy` | 最终窗口中心 `(cx, cy)` (驱动序) |
| `align_window_size_xy` | 最终窗口尺寸 `(sx, sy)` (驱动序) |

`apply_stored_window()` 用 center/size 恢复窗口 (offset 由驱动从 (center,size)
确定性推导, 无需持久化); 无窗口信息/非法窗口时返回 False 且不主动复位相机。

## CLI

```bash
python src/ao_shaping/tools/slm/calibration.py            # 全流程 (含窗口化 align)
python src/ao_shaping/tools/slm/calibration.py --align-margin 6 --align-min-window 256
python src/ao_shaping/tools/slm/calibration.py --skip-align  # 沿用已有窗口
```

新选项: `--align-margin` (默认 4.0, 边缘余量 ×FWHM)、`--align-min-window`
(默认 128, 最小窗口边长 px)。`--verify-only` 前可用 `apply_stored_window()` 恢复
上次窗口再验证。

## 离线测试

`tests/ao_shaping/tools/slm/test_slm_calibration_windowed.py` (12 例, 无硬件):
`FakeWindowedCCD` 镜像 Daheng `reset_window`/`get_numpy_image` 契约 (center=(x,y)、
size=(0,0) 全幅、inc 向下取整、offset≥0 断言、返回 `((w,h),(cx-x0,cy-y0))`),
`FakeSLM` + 光学链路把 SLM 相位映射为 CCD 衍射图案 (flat/blaze x/y + 0级/±1级),
覆盖: reset_window 契约 (全幅/量化/越界)、窗口裁剪、成功保留窗口+keys、漂移收敛
(3px/帧 → sy=664)、not_converged/no_spot/no_period 恢复全幅、apply_stored_window、
calibrate 保留 align keys。

运行: `python -m pytest tests/ao_shaping/tools/slm/ -q` (90 passed, 3 skipped)。