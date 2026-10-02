"""SLM 相关独立工具包。

包含:
- phase_capture.py — SLM 相位捕获 (湍流/Zernike 相位数据集采集)
- gray_response.py — SLM 灰度响应测量
- slm_diagnose.py — SLM 硬件自检 (freeze/modulate/linearity 三步证据链)
- slm_phase_response.py — SLM 相位→CCD 响应探针 (离焦/透镜用例 + 共享采集/渲染通道)
- slm_lut_runner.py — SLM 灰度→相位 LUT 校准工具 (CLI 入口, canonical LUT)
- calibration.py — SLM 标定统一模块 (合并原 drivers/slm/slm_calibration.py 与
  tools/slm/slm_shift_calib.py): 装配/光束位置/几何标定 (geo CLI: 直接运行
  `python src/ao_shaping/tools/slm/calibration.py`, `--skip-align` 可跳过) +
  shift 平移标定 (CLI: `python -m ao_shaping.tools.slm.calibration shift <args>`),
  含已废弃的 SLMLUTCalibrator
- slm_scan_analysis.py — SLM 扫描数据分析共享助手 (纯 numpy/stdlib, 无硬件依赖)
- slm_bench_probe.py — SLM 台架探针共享测量内核 (纯测量, 设备由参数传入):
  平滑/光斑 FWHM+质心+中心凹陷度/0 阶能量占比/面板 Zernike 放置/倾斜斜坡/线性拟合。
  固化了三条踩过坑的台架事实: 暗帧不可用裸 argmax 定位光斑 (参考质心曾漂 60px)、
  SLM 保留上次图案 (所以"平场"必须先下发)、固定 memory_number 是固件 no-op
- slm_tilt_probe.py — **判定面板是否真的在调制** (倾斜斜坡)。比光栅可靠: 光斑
  位移只取决于斜坡周期, 与衍射效率无关。用于推翻过一次"面板冻结"的假故障
- slm_panel_locate.py — 面板坐标上定位光斑中心 (扫描随机相位圆盘)。相机 0 阶
  不是面板坐标: 两轴互换 90° 且尺度差 >10 倍, 换算会把相位 roll 到不含光斑处
- slm_beam_extent.py — 半平面随机相位边界扫描测光斑中心/半径。文档里曾长期写着
  "~192 SLM px" 而实测 ~450px, 用陈旧数字会只调制一半光瞳
- slm_phase_resolution.py — 比较逐像素随机相位与光滑 Zernike 相位, 判定面板
  **等效相位分辨率**。实测逐像素随机相位对远场毫无影响, 而 Zernike 逐阶生效 ⇒
  面板无法分辨像素级相位, 因此散斑相关度标定路线在此台架不可用, 必须走 sweep
- slm_exposure_check.py — 相机自动曝光状态 + 固定设置下漂移。区分"相机漂移"与
  "SLM 保留上次图案"两种同样表现为"变暗 4 倍"的原因
- slm_snr_probe.py — SLM 扰动灵敏度 (SNR) 测量, **设备实例由参数传入**:
  噪底 sigma / `+ - - +` 回文漂移对消 dJ / 单模式与多模式 (SPGD 实际) SNR。
  用于按实测 SNR 选 `--delta`; 离线可用 fake 设备单测
- slm_zernike_sweep_probe.py — **光滑 Zernike 台架扫描探针** (CLI 入口,
  `python -m ao_shaping.tools.slm.slm_zernike_sweep_probe`)。一次扫描采集
  ramp + tilt + defocus + astig_x/y + coma_x/y + spherical 共 42 点 (前置一个 flat
  参考), 逐点用**稳定判据** (而非固定等待) 读帧, 落盘 `sweep_records.npz` +
  标准 Recorder 产物 (含下发相位与 CCD 帧)。设备由参数注入 ⇒ 可脱机单测;
  `--no-hw` 只打印采集计划。`scripts/model_in_loop_hw_runbook.py --stage sweep`
  的采集内核已收敛到本模块 (`zernike_panel` / `capture_settled` 单一来源)
- slm_zernike_common.py — SLM Zernike 工具集**共享常量与测量原语** (被 calibration /
  slm_zernike_response / slm_zernike_correction / slm_wfs_reference 复用)。固化
  WFS `get_zernike()` 的索引约定 (67 长数组, `coeff[1..66]` 有效, **顺序 m 枚举**,
  非标准 Noll 1976)
- slm_zernike_response.py — SLM Zernike 模式 → WFS 读数**响应矩阵标定**: 逐模式施加
  ±A 推拉扰动测 WFS 系数增量, 求逆即得 Zernike 模式法矫正控制律。全矩阵须用**同一
  Zernike 半径**, 否则系数被错误缩放
- slm_zernike_correction.py — SLM Zernike 模式法矫正**三阶段全流程** (自动定标+参考
  波前 / 稳健响应矩阵 / 闭环反向矫正, 带增益与泄漏)。反解前必须把 `w[0]` (piston) 置零
- slm_wfs_reference.py — SLM + WFS **参考波前标定与倾斜线性度** (三步: 纯平相位存参考
  → 还原/加载参考互验 → 不同强度 Zernike 倾斜读出线性度)。`optimize_pupil()` 只算不
  设, pupil 必须显式写回 `wfs.pupil`
- slm_wfs_probe.py — SLM + WFS **光强与 pupil 探针**, 标定前的硬件状态门控: 平相位下
  WFS 点阵强度/有效子孔径比例检查 + pupil 显式写回, 输出 JSON 报告
- delta_explorer.py — SPGD 扰动幅度 `delta` 的**纯分析**收敛性判据 (不构造设备,
  轨迹由调用方提供): 用 `frac_decreasing` (改善步占比) 与 `late_gain` (前 1/3 与
  后 1/3 均值之差) 排名候选值, 取代"末值对比首值"这种会被随机游走骗到的判据
- cartographer/   — SLM 标定综合工具 (余弦图样/Hartmann 波前重建/灰度-LUT/动态补偿)

LUT 路径约定:
- canonical (驱动可消费):  slm_lut_runner.py + utils/slm_lut.py → lut_forward.csv/
  lut_inverse.csv → Santec.load_lut(). 唯一进入 SLM 灰度↔相位补偿的 LUT。
- deprecated:  calibration.py::SLMLUTCalibrator (legacy 8-bit 自参考干涉法),
  保留供参考/单测, 已移出 CLI, 直接使用触发 DeprecationWarning。
- research:  cartographer/phase_grayscale_lut.py (WFS 实测研究 LUT, 仅供
  cartographer 闭环, 不可被 Santec.load_lut 消费)。
"""

from ao_shaping.tools.slm.slm_bench_probe import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    TILT_SHIFT_SCALE,
    SpotMeasurement,
    core_fraction,
    despike_frame,
    display_and_average,
    fit_linear_slope,
    measure_flat_reference,
    measure_spot,
    random_phase,
    ramp_panel,
    smooth_frame,
    tilt_shift_px,
    zernike_panel,
)
from ao_shaping.tools.slm.slm_phase_response import build_sequence, lens_cases, defocus_cases
from ao_shaping.tools.slm.slm_scan_analysis import (
    LINEARITY_AMPS,
    analyze_linearity,
    clamp_shift,
    group_raw_scan,
    latest_match,
    outlier_mask,
    parabolic_min,
)
from ao_shaping.tools.slm.slm_snr_probe import (
    SIGMA_FLOOR,
    SNR_STRONG,
    SNR_USABLE,
    SnrSweepResult,
    abba_signal,
    measure_noise_floor,
    snr_sweep,
    snr_verdict,
)

from ao_shaping.tools.slm.slm_zernike_sweep_probe import (
    AUX_MODES,
    MODE_CODES,
    SINGLE_LOBE_MIN_HOLLOWNESS,
    SWEEP_MODE_NM,
    TILT_AXES_NM,
    SweepPoint,
    SweepPointResult,
    SweepResult,
    acquire_sweep,
    capture_settled,
    default_sweep_points,
    point_panel_phase,
    save_sweep_npz,
    sweep_to_recorder_rows,
)

__all__ = [
    "build_sequence",
    "lens_cases",
    "defocus_cases",
    "AUX_MODES",
    "MODE_CODES",
    "SINGLE_LOBE_MIN_HOLLOWNESS",
    "SWEEP_MODE_NM",
    "TILT_AXES_NM",
    "SweepPoint",
    "SweepPointResult",
    "SweepResult",
    "acquire_sweep",
    "capture_settled",
    "default_sweep_points",
    "point_panel_phase",
    "save_sweep_npz",
    "sweep_to_recorder_rows",
    "BEAM_CENTER_PANEL",
    "BEAM_RADIUS_PANEL",
    "SLM_PANEL_H",
    "SLM_PANEL_W",
    "TILT_SHIFT_SCALE",
    "SpotMeasurement",
    "smooth_frame",
    "despike_frame",
    "measure_spot",
    "measure_flat_reference",
    "display_and_average",
    "core_fraction",
    "zernike_panel",
    "ramp_panel",
    "tilt_shift_px",
    "fit_linear_slope",
    "random_phase",
    "LINEARITY_AMPS",
    "outlier_mask",
    "clamp_shift",
    "parabolic_min",
    "latest_match",
    "group_raw_scan",
    "analyze_linearity",
    "SIGMA_FLOOR",
    "SNR_STRONG",
    "SNR_USABLE",
    "SnrSweepResult",
    "abba_signal",
    "measure_noise_floor",
    "snr_sweep",
    "snr_verdict",
]