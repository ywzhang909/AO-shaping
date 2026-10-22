"""光路有效性探针: 曝光阶梯 + 离焦响应, 判定光路能不能继续做几何标定。

为什么需要它
------------
2026-10-22 的 MIL 硬件 campaign 两次死在几何标定, 报的是
``no swept mode produced a usable width response; the spot width does not
respond to any of them on this bench``
(``optimizer/wfless/model_in_loop_shaping.py::calibrate_bench_geometry_from_sweep``)。
那条报错只说「宽度不响应」, 完全没提**信噪比不足**这一层: 当时的曝光窗口是
1-10 ms (峰值 24-213 counts), 而几何 sweep 靠「光斑宽度随扫描模式变化」来拟合
焦面标度 —— 那个宽度响应整个淹在噪声里。

所以几何标定之前应当先有一个**前置判定**: 光路到底通不通、够不够亮、SLM 到底
有没有在调制。这个探针一次跑完给出三态判定:

* ``ok`` —— 光斑明显高于噪声底, **且**离焦带来了显著响应 (宽度或峰值的变化超过
  噪声水平) ⇒ 光路有效, 可以继续标定。
* ``low_snr`` —— 光斑勉强高于噪声底, 或离焦响应淹没在噪声里 ⇒ 光路未必坏, 是
  **曝光/信噪比不足**; 给出建议曝光 (峰值落在 ~100-200 counts 的那一档)。
* ``no_response`` —— 光斑清晰可见但**离焦毫无响应** ⇒ 光路或 SLM 可疑 (面板没调制 /
  光束不在模型以为的位置 / 离焦没真正加上去); 给出排查方向
  (``tools/slm/slm_diagnose.py`` 的三步链: 冻结 / 调制 / 线性)。

**判定必须给出实测数字** (噪声底、可用窗口、离焦前后的峰值与宽度、判定阈值),
不许只给结论。

证据落盘: ``data/debug/optical_path_probe/<UTC戳>/`` —— 每档曝光与每种离焦状态的
npz (原始帧)、``optical_path_probe.json`` (全部读数 + 判定 + 阈值 + 建议)、
``optical_path_probe.png`` (曝光-峰值曲线 + 离焦响应曲线)。

退出码: ``0`` = 判定为 ``ok``; ``2`` = 判定为 ``low_snr`` 或 ``no_response``
(可以当上层门禁用)。

用法::

    python scripts/slm/probe_optical_path_validity.py --wait-until-idle 600
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT, ROOT / "src", ROOT / "libs", ROOT / "libs" / "OOPAO"):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ao_shaping.drivers.ccd.common import (
    create_camera,
    get_camera_exposure_ms,
    get_camera_exposure_range,
    set_camera_exposure_ms,
)
from ao_shaping.utils.image.beam_metrics import (
    beam_radius_second_moment,
    prepare_frame,
)
from ao_shaping.utils.image.frame_analysis import (
    DEFAULT_MIN_PEAK,
    measure_spot,
    spot_fwhm,
)
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
)
from ao_shaping.utils.slm.phase_display import (
    display_phase as display_phase_on_slm,
)
from ao_shaping.utils.wavefront.zernike_utils import (
    generate_zernike_phase,
)

#: 采集驱动 (``scripts/slm/run_hw_sweep_groups.py``) —— 设备互斥的探针与分类
#: 都在那里, 这里按路径加载复用, 不另写一份。
SWEEP_SCRIPT = ROOT / "scripts" / "slm" / "run_hw_sweep_groups.py"

# ---------------------------------------------------------------------------
# 判定阈值 (全部写死在这里, 报告会原样抄进 JSON, 便于事后核对)
# ---------------------------------------------------------------------------
#: 噪声底的显著性倍数: 峰值必须超过 ``中位数 + k * sigma`` 才算「有光」。
#: 与 ``utils/image/exposure_control.py::NOISE_FLOOR_SIGMAS`` 同值 (4.0)。
NOISE_FLOOR_SIGMAS = 4.0
#: 帧的 MAD 恰好为 0 时用的噪声尺度 (1 count = 数字化的最小可表达差异)。
NOISE_FLOOR_FALLBACK_SIGMA = 1.0
#: 峰值达到满量程的这个比例就算饱和 (该档不再可用)。
SATURATION_FRACTION = 0.95
#: 离焦响应要算「显著」, 需要超过**合并标准误**的多少倍 sigma。
#:
#: 用 SEM (标准误) 而不是逐帧标准差: 比较的是两个**均值*, N 帧的均值比单帧准
#: sqrt(N) 倍。
RESPONSE_SIGMAS = 3.0
#: 相对变化的**物理下限**: 离焦 ±0.5 rad (Noll 4, 幅度 1 时孔径边缘 ±1.73 rad)
#: 必然把远场光斑的宽度改变 ≥20%、把峰值改变 ≥25%。低于这个量级的变化在物理上
#: 不该发生 —— 这条是「显著」的兜底判据 (只在测得准时才允许, 见
#: ``analyse_defocus_response``)。
WIDTH_RELATIVE_FLOOR = 0.20
PEAK_RELATIVE_FLOOR = 0.25
#: 「测得准」的分辨率上限: 宽度要能分辨 25% 的变化, 峰值要能分辨 50% 的变化。
#: 分辨率比这还差 ⇒ 就算离焦真的在起作用, 我们也**看不出来** ⇒ 信噪比不足,
#: 而不是「面板没调制」。
#:
#: 取值依据: 几何 sweep 拟合的是 ``w**2(c)``, 宽度读数的相对误差会**平方**地
#: 进到拟合里 —— 宽度 25% 的分辨率对应 w² 约 50% 的不确定度, 再叠上逐点噪声,
#: 曲率就拟不出来了。这正是 2026-10-22 那次 campaign 报「宽度不响应」时
#: 实际处的位置 (曝光 1-10 ms, 峰值 24-213 counts)。
WIDTH_RESOLUTION_MAX = 0.25
PEAK_RESOLUTION_MAX = 0.50
#: 宽度的 sigma 下限 (px) —— 亚像素级的散布按 0 计会让判据过于苛刻。
FWHM_SIGMA_FLOOR_PX = 0.3
#: 峰值的 sigma 下限 (counts)。
PEAK_SIGMA_FLOOR_COUNTS = 1.0
#: 建议曝光的目标峰值区间 (counts)。
RECOMMENDED_PEAK_RANGE = (100.0, 200.0)
#: Noll 4 = 离焦 (仓库约定)。
DEFOCUS_NOLL_INDEX = 4
#: 建议曝光的落点 (counts) —— 区间的中点。
RECOMMENDED_PEAK_TARGET = 0.5 * (
    RECOMMENDED_PEAK_RANGE[0] + RECOMMENDED_PEAK_RANGE[1]
)

#: 三态判定的取值。
STATE_OK = "ok"
STATE_LOW_SNR = "low_snr"
STATE_NO_RESPONSE = "no_response"


@dataclass
class FrameStats:
    """一帧的读数。

    **原始帧与处理后帧分开记**: ``prepare_frame`` 做的是「减中值 + clip 到 0」,
    它把探测器本底拿掉, 于是 ``peak_prepared`` 与 ``peak_raw`` 差一个本底。
    噪声底、饱和判据用原始帧 (绝对 counts), 形状类指标用处理后帧。
    """

    peak_raw: float
    peak_prepared: float
    mean_raw: float
    median_raw: float
    p99_raw: float
    std_raw: float
    fwhm_px: float
    fwhm_raw_px: float
    second_moment_radius_px: float
    centroid_x: float
    centroid_y: float
    hollowness: float

    def as_dict(self) -> dict[str, float]:
        """转成可直接 JSON 化的字典。"""
        return {
            "peak_raw": self.peak_raw,
            "peak_prepared": self.peak_prepared,
            "mean_raw": self.mean_raw,
            "median_raw": self.median_raw,
            "p99_raw": self.p99_raw,
            "std_raw": self.std_raw,
            "fwhm_px": self.fwhm_px,
            "fwhm_raw_px": self.fwhm_raw_px,
            "second_moment_radius_px": self.second_moment_radius_px,
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "hollowness": self.hollowness,
        }


def frame_stats(frame: np.ndarray) -> FrameStats:
    """量一帧: 峰值/均值/p99/质心/FWHM/二阶矩半径。

    宽度用 :func:`spot_fwhm` (与几何 sweep 同一个口径), 但喂给它的是
    :func:`prepare_frame` 处理过的帧 —— 原始帧带着探测器本底, 半高位置会被本底
    推偏, 宽度读数会偏大。二阶矩半径用
    :func:`beam_radius_second_moment` (ISO-11146), 同样在处理后帧上算。
    """
    raw = np.asarray(frame, dtype=np.float64)
    prepared = prepare_frame(raw)
    spot = measure_spot(prepared)
    return FrameStats(
        peak_raw=float(raw.max()),
        peak_prepared=float(prepared.max()),
        mean_raw=float(raw.mean()),
        median_raw=float(np.median(raw)),
        p99_raw=float(np.percentile(raw, 99)),
        std_raw=float(raw.std()),
        fwhm_px=float(spot_fwhm(prepared)),
        fwhm_raw_px=float(spot_fwhm(raw)),
        second_moment_radius_px=float(beam_radius_second_moment(prepared)),
        centroid_x=float(spot.centroid_x),
        centroid_y=float(spot.centroid_y),
        hollowness=float(spot.hollowness),
    )


def robust_noise_floor(frame: np.ndarray) -> dict[str, float]:
    """从一帧估计噪声底 (中位数 / sigma / 阈值 / 峰值)。

    sigma 用稳健尺度 ``1.4826 * MAD``: 光斑只占画幅的一小块, 不会把 MAD 抬起来。
    阈值是 ``中位数 + NOISE_FLOOR_SIGMAS * sigma`` —— 峰值超过它才算「有光」。
    """
    data = np.asarray(frame, dtype=np.float64)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    sigma = 1.4826 * mad
    if sigma <= 0.0:
        sigma = NOISE_FLOOR_FALLBACK_SIGMA
    return {
        "median_counts": median,
        "sigma_counts": sigma,
        "threshold_counts": median + NOISE_FLOOR_SIGMAS * sigma,
        "peak_counts": float(data.max()),
    }


def build_defocus_phase(
    amplitude_rad: float,
    radius_px: int,
    pupil_center: tuple[int, int] = BEAM_CENTER_PANEL,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
) -> np.ndarray:
    """离焦相位 (raw 未包裹弧度), 走 canonical 的 Zernike 入口。

    Noll 4 = 离焦 (仓库约定)。幅度是 **Noll 单位 RMS** 归一化下的系数, 所以
    幅度 1.0 = 瞳面内 RMS 相位 1 rad (瞳面边缘约 ±1.73 rad)。

    离焦盘**以瞳面中心为准** (``pupil_center``, 面板坐标), 而不是面板几何中心:
    离焦盘一旦偏心, 它带上一级倾斜与像散, 就不再是纯离焦了。孔径外的相位按 0
    处理 (平场), 与 ``tools/kernels/sweep_analysis.py`` 的同一口径。

    ``generate_zernike_phase`` 在孔径外返回 NaN, 这里填 0。
    """
    height, width = int(panel_shape[0]), int(panel_shape[1])
    if abs(float(amplitude_rad)) < 1e-12:
        return np.zeros((height, width), dtype=np.float64)
    radius = int(max(int(radius_px), 1))
    cx = min(max(int(pupil_center[0]), 0), width - 1)
    cy = min(max(int(pupil_center[1]), 0), height - 1)
    if (cx, cy) == (width // 2, height // 2):
        # 瞳面中心就是面板中心 ⇒ 直接在整个面板上生成 (canonical 调用, 与其它
        # 脚本逐位一致)。
        raw = generate_zernike_phase(
            {DEFOCUS_NOLL_INDEX: float(amplitude_rad)},
            resolution=(width, height),
            n_max=DEFOCUS_NOLL_INDEX,
            radius=radius,
        )
        return np.nan_to_num(
            np.asarray(raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
    # 瞳面中心偏离面板中心 ⇒ 在以瞳面为准的方块上生成再贴回面板 (与
    # ``tools/kernels/bench_kernels.py::zernike_panel`` 同一几何)。
    size = 2 * radius + 1
    patch = generate_zernike_phase(
        {DEFOCUS_NOLL_INDEX: float(amplitude_rad)},
        resolution=(size, size),
        n_max=DEFOCUS_NOLL_INDEX,
        radius=radius,
    )
    patch = np.nan_to_num(np.asarray(patch, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros((height, width), dtype=np.float64)
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius + 1, cy + radius + 1
    dx0, dy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, width), min(y1, height)
    if x1c > x0c and y1c > y0c:
        out[y0c:y1c, x0c:x1c] = patch[dy0 : dy0 + (y1c - y0c), dx0 : dx0 + (x1c - x0c)]
    return out


def capture_frames(cam: Any, n_frames: int) -> list[np.ndarray]:
    """连抓 ``n_frames`` 帧原始画面。"""
    frames: list[np.ndarray] = []
    for _ in range(max(1, int(n_frames))):
        frames.append(np.asarray(cam.get_numpy_image(n_sample=1)))
    return frames


def _save_npz(
    path: Path,
    frames: list[np.ndarray],
    meta: dict[str, Any],
) -> None:
    """把原始帧与元信息落成一个 npz。"""
    np.savez_compressed(path, frames=np.stack(frames), **{k: np.asarray(v) for k, v in meta.items()})


def measure_exposure_ladder(
    cam: Any,
    exposures_ms: list[float],
    n_frames: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """曝光阶梯: 每档抓 ``n_frames`` 帧, 记峰值/均值/p99/质心/FWHM。

    阶梯的每一档都对应一个 npz (``exposure_<ms>ms.npz``), 存的是**原始帧**。
    """
    readings: list[dict[str, Any]] = []
    for requested in exposures_ms:
        set_camera_exposure_ms(cam, float(requested))
        effective = float(get_camera_exposure_ms(cam))
        frames = capture_frames(cam, n_frames)
        stats = [frame_stats(frame) for frame in frames]
        peaks = [s.peak_raw for s in stats]
        noise = robust_noise_floor(frames[0])
        dtype = np.asarray(frames[0]).dtype
        full_scale = (
            float(np.iinfo(dtype).max)
            if np.issubdtype(dtype, np.integer)
            else 255.0
        )
        record: dict[str, Any] = {
            "requested_ms": float(requested),
            "effective_ms": effective,
            "n_frames": len(frames),
            "full_scale": full_scale,
            "peak_raw": float(max(peaks)),
            "peak_raw_per_frame": [float(p) for p in peaks],
            "peak_prepared": float(max(s.peak_prepared for s in stats)),
            "mean_raw": float(np.mean([s.mean_raw for s in stats])),
            "median_raw": float(np.mean([s.median_raw for s in stats])),
            "p99_raw": float(np.max([s.p99_raw for s in stats])),
            "fwhm_px": float(np.median([s.fwhm_px for s in stats])),
            "fwhm_per_frame": [float(s.fwhm_px) for s in stats],
            "second_moment_radius_px": float(
                np.median([s.second_moment_radius_px for s in stats])
            ),
            "centroid_x": float(np.median([s.centroid_x for s in stats])),
            "centroid_y": float(np.median([s.centroid_y for s in stats])),
            "hollowness": float(np.median([s.hollowness for s in stats])),
            "noise_median_counts": noise["median_counts"],
            "noise_sigma_counts": noise["sigma_counts"],
            "noise_threshold_counts": noise["threshold_counts"],
            "npz": f"exposure_{float(requested):g}ms.npz",
        }
        _save_npz(
            out_dir / record["npz"],
            frames,
            {
                "requested_ms": float(requested),
                "effective_ms": effective,
                "peak_raw": record["peak_raw"],
            },
        )
        readings.append(record)
    return readings


def measure_defocus_sweep(
    cam: Any,
    slm: Any,
    amplitudes_rad: list[float],
    exposure_ms: float,
    n_frames: int,
    radius_px: int,
    pupil_center: tuple[int, int],
    settle_s: float,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """离焦扫描: 平场与若干离焦幅度之间切换, 每个状态取同样曝光下的帧。

    末尾再做一次**平场复测**, 用来量整轮的漂移 (面板重显示 + 曝光不变), 也就是
    「同一个状态能重复测到多少」的真实验证上限。
    """
    set_camera_exposure_ms(cam, float(exposure_ms))
    effective = float(get_camera_exposure_ms(cam))
    states: list[dict[str, Any]] = []

    def _capture_state(amplitude: float, tag: str) -> dict[str, Any]:
        phase = build_defocus_phase(amplitude, radius_px, pupil_center)
        display_phase_on_slm(slm, phase, settle_s)
        frames = capture_frames(cam, n_frames)
        stats = [frame_stats(frame) for frame in frames]
        record: dict[str, Any] = {
            "tag": tag,
            "defocus_rad": float(amplitude),
            "exposure_ms": float(exposure_ms),
            "effective_ms": effective,
            "n_frames": len(frames),
            "peak_raw": float(np.mean([s.peak_raw for s in stats])),
            "peak_raw_max": float(max(s.peak_raw for s in stats)),
            "peak_raw_per_frame": [float(s.peak_raw) for s in stats],
            "peak_prepared": float(max(s.peak_prepared for s in stats)),
            "fwhm_px": float(np.median([s.fwhm_px for s in stats])),
            "fwhm_per_frame": [float(s.fwhm_px) for s in stats],
            "second_moment_radius_px": float(
                np.median([s.second_moment_radius_px for s in stats])
            ),
            "centroid_x": float(np.median([s.centroid_x for s in stats])),
            "centroid_y": float(np.median([s.centroid_y for s in stats])),
            "hollowness": float(np.median([s.hollowness for s in stats])),
            "npz": f"defocus_{tag}.npz",
        }
        _save_npz(
            out_dir / record["npz"],
            frames,
            {
                "defocus_rad": float(amplitude),
                "exposure_ms": float(exposure_ms),
                "peak_raw": record["peak_raw"],
            },
        )
        return record

    for amplitude in amplitudes_rad:
        tag = f"{float(amplitude):+g}rad".replace("+", "p").replace("-", "m")
        states.append(_capture_state(float(amplitude), tag))
    states.append(_capture_state(0.0, "flat_repeat"))
    return states


def analyse_ladder(
    ladder: list[dict[str, Any]],
    full_scale: float,
) -> dict[str, Any]:
    """从曝光阶梯里取出噪声底、可用窗口与建议曝光。

    * **噪声底**: 最短曝光档那一帧的 ``中位数 + 4 * sigma`` 与它的峰值。
    * **可用窗口**: 峰值高于噪声底、且没饱和的档位。
    * **建议曝光**: 可用窗口里峰值落在 ~100-200 counts 的那一档; 没有就取可用
      窗口里最亮且不饱和的那一档。
    """
    shortest = min(ladder, key=lambda r: r["effective_ms"])
    scales = [float(r["full_scale"]) for r in ladder if "full_scale" in r]
    if scales:
        full_scale = max(scales)
    noise_floor = {
        "exposure_ms": float(shortest["effective_ms"]),
        "median_counts": float(shortest["noise_median_counts"]),
        "sigma_counts": float(shortest["noise_sigma_counts"]),
        "threshold_counts": float(shortest["noise_threshold_counts"]),
        "peak_counts": float(shortest["peak_raw"]),
    }
    saturation_counts = SATURATION_FRACTION * full_scale
    for record in ladder:
        record["carries_signal"] = bool(record["peak_raw"] > noise_floor["threshold_counts"])
        record["saturated"] = bool(record["peak_raw"] >= saturation_counts)
        record["usable"] = bool(record["carries_signal"] and not record["saturated"])

    usable = [r for r in ladder if r["usable"]]
    recommended: dict[str, Any] | None = None
    if usable:
        in_band = [
            r
            for r in usable
            if RECOMMENDED_PEAK_RANGE[0] <= r["peak_raw"] <= RECOMMENDED_PEAK_RANGE[1]
        ]
        pool = in_band or usable
        pick = min(
            pool,
            key=lambda r: abs(float(r["peak_raw"]) - RECOMMENDED_PEAK_TARGET),
        )
        recommended = {
            "exposure_ms": float(pick["effective_ms"]),
            "peak_counts": float(pick["peak_raw"]),
            "in_target_band": bool(in_band),
            "target_band_counts": list(RECOMMENDED_PEAK_RANGE),
        }

    # 阶梯的单调性: 改曝光后第一帧就是新曝光是大恒驱动层的不变量, 阶梯不单调
    # 说明有别的因素 (错档 / 增益漂移 / 曝光没真正生效)。
    peaks = [float(r["peak_raw"]) for r in sorted(ladder, key=lambda r: r["effective_ms"])]
    monotonic = all(b >= a for a, b in zip(peaks, peaks[1:], strict=False))
    linearity_r2 = float("nan")
    if len(peaks) >= 3:
        xs = np.array([r["effective_ms"] for r in sorted(ladder, key=lambda r: r["effective_ms"])])
        ys = np.array(peaks)
        mask = ys > noise_floor["threshold_counts"]
        if int(mask.sum()) >= 3:
            design = np.stack([np.ones(int(mask.sum())), xs[mask]], axis=1)
            sol, *_ = np.linalg.lstsq(design, ys[mask], rcond=None)
            resid = design @ sol - ys[mask]
            ss_tot = float(np.sum((ys[mask] - ys[mask].mean()) ** 2))
            ss_res = float(np.sum(resid**2))
            linearity_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "noise_floor": noise_floor,
        "saturation_counts": float(saturation_counts),
        "usable_window_ms": [float(r["effective_ms"]) for r in usable],
        "max_ladder_peak_counts": float(max((r["peak_raw"] for r in ladder), default=0.0)),
        "best_usable_peak_counts": float(max((r["peak_raw"] for r in usable), default=0.0)),
        "recommended_exposure": recommended,
        "ladder_monotonic": bool(monotonic),
        "ladder_linearity_r2": linearity_r2,
    }


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    """一个离焦状态的读数汇总 (逐帧样本的均值/散布)。"""
    peaks = [float(v) for v in state.get("peak_raw_per_frame", [])] or [
        float(state["peak_raw"])
    ]
    widths = [float(v) for v in state.get("fwhm_per_frame", [])] or [
        float(state["fwhm_px"])
    ]
    return {
        "defocus_rad": float(state["defocus_rad"]),
        "n": len(peaks),
        "peak_mean": float(np.mean(peaks)),
        "peak_std": float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0,
        "fwhm_mean": float(np.mean(widths)),
        "fwhm_std": float(np.std(widths, ddof=1)) if len(widths) > 1 else 0.0,
    }


def analyse_defocus_response(
    states: list[dict[str, Any]],
    noise_threshold_counts: float,
) -> dict[str, Any]:
    """从离焦扫描里量「离焦有没有真的起作用」。

    两条判据, 互相独立:

    1. **统计判据** —— 某个幅度上的读数与平场的差, 超过**合并标准误**的
       ``RESPONSE_SIGMAS`` 倍 (``SEM = sigma * sqrt(1/N_flat + 1/N_state)``,
       比较的是均值, 所以要用标准误而不是单帧标准差)。**每个非零幅度都要看**
       —— 响应可以是不对称的 (实验系统自身的离焦偏置会让 +A 与 −A 的效果差很多),
       只看其中一个会漏掉真正的响应。
    2. **物理下限判据** —— 相对变化达到 :data:`WIDTH_RELATIVE_FLOOR` /
       :data:`PEAK_RELATIVE_FLOOR`。离焦 ±0.5 rad 在物理上必然让光斑宽度变化
       ≥20%、峰值变化 ≥25%; 达不到这个量级就是「没真正加上去」。这条**只在测得
       准的时候**才允许: 噪声也能造出很大的相对变化。

    同时算出**分辨率**: ``3 * sigma / 基线值`` —— 「能分辨多小的相对变化」。
    分辨率比物理下限还差 ⇒ 即使面板真的在调制我们也看不出来 ⇒ 这是
    **信噪比不足** (``low_snr``), 不是「面板没调制」 (``no_response``)。

    噪声尺度取自**平场**的那几组: 组内逐帧散布与**组间差** (平场复测) 取大者。
    组间差必须算进来 —— 它量的是「换一次显示再测一遍」能差多少, 而几何 sweep
    恰恰是在换显示状态下比较的, 只用组内散布会低估噪声。
    """
    summaries = [_state_summary(s) for s in states]
    flat = [s for s in summaries if abs(s["defocus_rad"]) < 1e-12]
    non_flat = [s for s in summaries if abs(s["defocus_rad"]) >= 1e-12]
    if not flat:
        flat = [summaries[0]]

    def _sigma(groups: list[dict[str, Any]], mean_key: str, std_key: str, floor: float) -> float:
        within = [
            float(g[std_key]) for g in groups if int(g["n"]) > 1 and g[std_key] > 0.0
        ]
        pooled = float(np.sqrt(np.mean(np.square(within)))) if within else 0.0
        between = 0.0
        if len(groups) >= 2:
            means = [float(g[mean_key]) for g in groups]
            between = (max(means) - min(means)) / math.sqrt(2.0)
        return max(pooled, between, floor)

    sigma_peak = _sigma(flat, "peak_mean", "peak_std", PEAK_SIGMA_FLOOR_COUNTS)
    sigma_fwhm = _sigma(flat, "fwhm_mean", "fwhm_std", FWHM_SIGMA_FLOOR_PX)

    base_peak = float(np.mean([g["peak_mean"] for g in flat]))
    base_fwhm = float(np.mean([g["fwhm_mean"] for g in flat]))
    n_flat = max(sum(int(g["n"]) for g in flat), 1)

    peak_signal = 0.0
    width_signal = 0.0
    peak_signal_at = 0.0
    width_signal_at = 0.0
    peak_relative = 0.0
    width_relative = 0.0
    peak_stat = False
    width_stat = False
    per_amp: list[dict[str, Any]] = []
    for state, summary in zip(states, summaries, strict=True):
        peak = float(summary["peak_mean"])
        fwhm = float(summary["fwhm_mean"])
        n_state = max(int(summary["n"]), 1)
        # 该状态自己的噪声 (暗状态的光子噪声更大), 与平场的噪声取大者。
        sigma_peak_state = max(sigma_peak, float(summary["peak_std"]))
        sigma_fwhm_state = max(sigma_fwhm, float(summary["fwhm_std"]))
        sem_peak = sigma_peak_state * math.sqrt(1.0 / n_flat + 1.0 / n_state)
        sem_fwhm = sigma_fwhm_state * math.sqrt(1.0 / n_flat + 1.0 / n_state)
        d_peak = peak - base_peak
        d_fwhm = fwhm - base_fwhm
        is_flat = abs(summary["defocus_rad"]) < 1e-12
        if not is_flat:
            if abs(d_peak) >= peak_signal:
                peak_signal = abs(d_peak)
                peak_signal_at = summary["defocus_rad"]
            if abs(d_fwhm) >= width_signal:
                width_signal = abs(d_fwhm)
                width_signal_at = summary["defocus_rad"]
            if base_peak > 0:
                peak_relative = max(peak_relative, abs(d_peak) / base_peak)
            if base_fwhm > 0:
                width_relative = max(width_relative, abs(d_fwhm) / base_fwhm)
            peak_stat = peak_stat or abs(d_peak) >= RESPONSE_SIGMAS * sem_peak
            width_stat = width_stat or abs(d_fwhm) >= RESPONSE_SIGMAS * sem_fwhm
        per_amp.append(
            {
                "defocus_rad": summary["defocus_rad"],
                "peak_raw": peak,
                "peak_std": summary["peak_std"],
                "fwhm_px": fwhm,
                "fwhm_std": summary["fwhm_std"],
                "second_moment_radius_px": float(state["second_moment_radius_px"]),
                "centroid_x": float(state["centroid_x"]),
                "centroid_y": float(state["centroid_y"]),
                "hollowness": float(state["hollowness"]),
                "delta_peak_counts": d_peak,
                "delta_fwhm_px": d_fwhm,
                "sem_peak_counts": float(sem_peak),
                "sem_fwhm_px": float(sem_fwhm),
                "carries_signal": bool(peak > noise_threshold_counts),
                "visible": bool(peak >= DEFAULT_MIN_PEAK),
                "peak_statistically_significant": (
                    None if is_flat else bool(abs(d_peak) >= RESPONSE_SIGMAS * sem_peak)
                ),
                "fwhm_statistically_significant": (
                    None if is_flat else bool(abs(d_fwhm) >= RESPONSE_SIGMAS * sem_fwhm)
                ),
            }
        )

    width_resolution = (
        float(RESPONSE_SIGMAS * sigma_fwhm / base_fwhm) if base_fwhm > 0 else float("inf")
    )
    peak_resolution = (
        float(RESPONSE_SIGMAS * sigma_peak / base_peak) if base_peak > 0 else float("inf")
    )
    precision_ok = bool(
        width_resolution <= WIDTH_RESOLUTION_MAX and peak_resolution <= PEAK_RESOLUTION_MAX
    )
    # 相对变化这条兜底判据**只在测得准的时候**才允许: 噪声也能造出很大的相对
    # 变化 (暗帧里宽度读数乱跳), 不加这道门会把噪声当成响应。
    responds_relative = bool(
        precision_ok
        and (width_relative >= WIDTH_RELATIVE_FLOOR or peak_relative >= PEAK_RELATIVE_FLOOR)
    )
    return {
        "baseline_peak_counts": base_peak,
        "baseline_fwhm_px": base_fwhm,
        "flat_group_peak_means": [float(g["peak_mean"]) for g in flat],
        "flat_group_fwhm_means": [float(g["fwhm_mean"]) for g in flat],
        "sigma_peak_counts": sigma_peak,
        "sigma_fwhm_px": sigma_fwhm,
        "peak_snr": float(base_peak / sigma_peak) if sigma_peak > 0 else float("inf"),
        "width_snr": float(base_fwhm / sigma_fwhm) if sigma_fwhm > 0 else float("inf"),
        "peak_signal_counts": float(peak_signal),
        "width_signal_px": float(width_signal),
        "peak_signal_at_rad": float(peak_signal_at),
        "width_signal_at_rad": float(width_signal_at),
        "peak_relative_change": float(peak_relative),
        "width_relative_change": float(width_relative),
        "peak_resolution": peak_resolution,
        "width_resolution": width_resolution,
        "precision_ok": precision_ok,
        "peak_statistically_significant": peak_stat,
        "width_statistically_significant": width_stat,
        "peak_significant": bool(peak_stat or (precision_ok and peak_relative >= PEAK_RELATIVE_FLOOR)),
        "width_significant": bool(width_stat or (precision_ok and width_relative >= WIDTH_RELATIVE_FLOOR)),
        "responds": bool(peak_stat or width_stat or responds_relative),
        "per_amp": per_amp,
    }


def decide(
    ladder_summary: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    """三态判定 + 依据 + 建议。

    判定顺序是有意的: 先看「有没有信号」, 再看「响应测不准还是根本没响应」。
    """
    reasons: list[str] = []
    advice: list[str] = []
    usable = ladder_summary["usable_window_ms"]
    best_peak = float(ladder_summary["best_usable_peak_counts"])
    noise = ladder_summary["noise_floor"]
    recommended = ladder_summary.get("recommended_exposure")

    if not usable:
        state = STATE_LOW_SNR
        reasons.append(
            "曝光阶梯里没有任何一档把光斑抬到噪声底之上 "
            f"(噪声底 {noise['threshold_counts']:.1f} counts @ "
            f"{noise['exposure_ms']:g} ms, 阶梯最高峰值 "
            f"{ladder_summary['max_ladder_peak_counts']:.1f} counts)"
        )
        advice.append(
            "把曝光阶梯整体抬高 (例如 --exposures 0.2,1,5,20,50,100)，或检查光源与光路"
        )
    else:
        reasons.append(
            f"可用曝光窗口 {', '.join(f'{v:g}' for v in usable)} ms "
            f"(峰值高于噪声底 {noise['threshold_counts']:.1f} counts 且未饱和)"
        )
        if recommended is not None:
            reasons.append(
                f"建议曝光 {recommended['exposure_ms']:g} ms (峰值 "
                f"{recommended['peak_counts']:.0f} counts"
                + ("" if recommended["in_target_band"] else ", 落在目标区间外")
                + ")"
            )

    responds = bool(response["responds"])
    width_resolution = float(response["width_resolution"])
    peak_resolution = float(response["peak_resolution"])
    precision_ok = bool(response["precision_ok"])
    if usable:
        if responds:
            state = STATE_OK if best_peak >= DEFAULT_MIN_PEAK else STATE_LOW_SNR
            reasons.append(
                "离焦有显著响应: 宽度最大变化 "
                f"{response['width_signal_px']:.2f} px "
                f"({response['width_relative_change']:.1%}, 出现在离焦 "
                f"{response['width_signal_at_rad']:+g} rad), 峰值最大变化 "
                f"{response['peak_signal_counts']:.1f} counts "
                f"({response['peak_relative_change']:.1%}, 出现在离焦 "
                f"{response['peak_signal_at_rad']:+g} rad); 判据 = "
                f"{RESPONSE_SIGMAS:g}×合并标准误 "
                f"(σ_FWHM={response['sigma_fwhm_px']:.2f} px, "
                f"σ_peak={response['sigma_peak_counts']:.1f} counts) "
                f"或相对变化下限 ({WIDTH_RELATIVE_FLOOR:.0%} / "
                f"{PEAK_RELATIVE_FLOOR:.0%})"
            )
        elif not precision_ok:
            state = STATE_LOW_SNR
            reasons.append(
                "离焦响应淹没在噪声里: 宽度只能分辨 "
                f"{width_resolution:.1%} 的变化 (上限 {WIDTH_RESOLUTION_MAX:.0%}), "
                f"峰值只能分辨 {peak_resolution:.1%} (上限 {PEAK_RESOLUTION_MAX:.0%}); "
                f"实测宽度只变了 {response['width_signal_px']:.2f} px "
                f"({response['width_relative_change']:.1%}, "
                f"σ={response['sigma_fwhm_px']:.2f} px), 峰值只变了 "
                f"{response['peak_signal_counts']:.1f} counts "
                f"({response['peak_relative_change']:.1%}, "
                f"σ={response['sigma_peak_counts']:.1f})"
            )
            advice.append(
                "这不是「光路坏」而是「测不准」: 把曝光提到峰值 100-200 counts "
                "再复测, 或在采集侧多帧平均 (曝光阶梯里挑更亮的那一档); "
                "几何 sweep 的宽度响应在当前信噪比下必然淹没"
            )
        else:
            state = STATE_NO_RESPONSE
            reasons.append(
                "光斑清晰可见 (峰值 "
                f"{response['baseline_peak_counts']:.0f} counts) 且测得准 "
                f"(宽度分辨率 {width_resolution:.1%} ≤ {WIDTH_RESOLUTION_MAX:.0%}, "
                f"峰值分辨率 {peak_resolution:.1%} ≤ {PEAK_RESOLUTION_MAX:.0%}) "
                "但离焦毫无响应: 宽度只变了 "
                f"{response['width_signal_px']:.2f} px "
                f"({response['width_relative_change']:.1%}), 峰值只变了 "
                f"{response['peak_signal_counts']:.1f} counts "
                f"({response['peak_relative_change']:.1%})"
            )
            advice.append(
                "排查顺序: 1) python -m ao_shaping.tools.slm.slm_diagnose 的三步链 "
                "(冻结 / 调制 / 线性) —— 面板没调制时离焦不会改变任何读数; "
                "2) 确认离焦真的加到了光束照到的那块面板上 (瞳面位置); "
                "3) 确认相机仍在焦面附近 (离焦幅度是否已把焦面推出视场)"
            )
    else:
        state = STATE_LOW_SNR

    if state != STATE_OK and recommended is not None and usable:
        advice.append(
            f"下一次标定建议用曝光 {recommended['exposure_ms']:g} ms "
            f"(峰值 {recommended['peak_counts']:.0f} counts)"
        )

    return {
        "state": state,
        "reasons": reasons,
        "advice": advice,
        "exit_code": 0 if state == STATE_OK else 2,
    }


def plot_results(
    ladder: list[dict[str, Any]],
    ladder_summary: dict[str, Any],
    response: dict[str, Any],
    out_path: Path,
) -> None:
    """曝光-峰值曲线 + 离焦响应曲线 (峰值/宽度 vs 离焦幅度)。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    xs = [float(r["effective_ms"]) for r in ladder]
    ys = [float(r["peak_raw"]) for r in ladder]
    ax.plot(xs, ys, "o-", color="#c2410c", label="峰值 (原始帧)")
    threshold = float(ladder_summary["noise_floor"]["threshold_counts"])
    ax.axhline(threshold, ls="--", color="#64748b", label=f"噪声底 {threshold:.1f} counts")
    ax.axhline(
        SATURATION_FRACTION * 255.0,
        ls="--",
        color="#dc2626",
        label=f"饱和线 {SATURATION_FRACTION * 255.0:.0f} counts",
    )
    usable = [
        (float(r["effective_ms"]), float(r["peak_raw"]))
        for r in ladder
        if r.get("usable")
    ]
    if usable:
        ax.fill_between(
            [x for x, _ in usable],
            0,
            [y for _, y in usable],
            color="#16a34a",
            alpha=0.12,
            label="可用窗口",
        )
    recommended = ladder_summary.get("recommended_exposure")
    if recommended is not None:
        ax.plot(
            [recommended["exposure_ms"]],
            [recommended["peak_counts"]],
            marker="*",
            ms=16,
            color="#16a34a",
            label=f"建议曝光 {recommended['exposure_ms']:g} ms",
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("曝光 (ms)")
    ax.set_ylabel("峰值 (counts)")
    ax.set_title("曝光阶梯: 峰值 vs 曝光")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    per_amp = sorted(
        (p for p in response["per_amp"] if p["defocus_rad"] != 0.0),
        key=lambda p: float(p["defocus_rad"]),
    )
    amps = [float(p["defocus_rad"]) for p in per_amp]
    widths = [float(p["fwhm_px"]) for p in per_amp]
    peaks = [float(p["peak_raw"]) for p in per_amp]
    ax.plot(amps, widths, "s-", color="#1d4ed8", label="光斑 FWHM (px)")
    ax.axhline(
        float(response["baseline_fwhm_px"]),
        ls="--",
        color="#64748b",
        label=f"平场 FWHM {response['baseline_fwhm_px']:.1f} px",
    )
    ax.set_xlabel("离焦幅度 (rad, Noll 4)")
    ax.set_ylabel("FWHM (px)", color="#1d4ed8")
    ax.tick_params(axis="y", labelcolor="#1d4ed8")
    ax.set_title("离焦响应: 宽度/峰值 vs 离焦幅度")

    twin = ax.twinx()
    twin.plot(amps, peaks, "o-", color="#b45309", label="峰值 (counts)")
    twin.set_ylabel("峰值 (counts)", color="#b45309")
    twin.tick_params(axis="y", labelcolor="#b45309")

    lines = ax.get_lines() + twin.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def load_sweep_module() -> Any:
    """按路径加载采集驱动 (``scripts/slm/run_hw_sweep_groups.py``)。

    **必须先注册进 ``sys.modules`` 再执行**: ``dataclasses`` 通过
    ``sys.modules[cls.__module__].__dict__`` 解析 ``KW_ONLY`` /
    ``_HAS_DEFAULT_FACTORY``, 按路径加载而不注册的话, 里面每个 dataclass 都会炸
    ``'NoneType' object has no attribute '__dict__'``。
    """
    spec = importlib.util.spec_from_file_location("run_hw_sweep_groups", SWEEP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the device probe from {SWEEP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def acquire_devices(args: argparse.Namespace) -> None:
    """开跑前确认设备没被别人占着。

    SLM 的 USB 句柄是独占的: 谁先打开谁拥有, 另一个会话只会拿到 -10003。所以
    先探一次; 报「被占」且属于**可以等**的那两类 (别人在用) 时, 按
    ``--wait-until-idle`` 的预算等。设备没插上 / 探针自己没装好这类等不来,
    立刻报错退出。
    """
    module = load_sweep_module()
    budget = float(args.wait_until_idle)
    if budget > 0:
        code, message = module.wait_until_idle(
            budget,
            slm_number=int(args.slm_number),
            cam_id=int(args.cam_id),
            cam_type=str(args.cam_type),
            exposure_ms=float(args.exposures[0]),
            own_paths=[Path(__file__).resolve()],
        )
        if code == 0:
            return
        verdict = module.classify_device_unavailable(message)
        waitable = verdict is not None and verdict.kind in module.WAITABLE_KINDS
        kind = verdict.kind if verdict is not None else "unknown"
        raise SystemExit(
            f"等 {budget:.0f}s 后仍拿不到设备 ({kind}, 可等={waitable}): {message}"
        )

    result = module.probe_devices(
        slm_number=int(args.slm_number),
        cam_id=int(args.cam_id),
        cam_type=str(args.cam_type),
        exposure_ms=float(args.exposures[0]),
    )
    if result.status == module.PROBE_IDLE:
        return
    verdict = module.classify_device_unavailable(result.message)
    if verdict is None:
        raise SystemExit(f"设备探针没有给出可用结论: {result.message}")
    raise SystemExit(
        f"设备不可用 ({verdict.kind}): {verdict.headline()} —— {verdict.advice()}"
        + ("" if verdict.kind in module.WAITABLE_KINDS else " (加 --wait-until-idle 也等不来)")
    )


@dataclass
class ProbeConfig:
    """一次探测的配置 (落进 JSON, 便于复现)。"""

    exposures_ms: list[float] = field(default_factory=list)
    defocus_amps_rad: list[float] = field(default_factory=list)
    frames_per_exposure: int = 2
    frames_per_defocus: int = 5
    defocus_radius_px: int = BEAM_RADIUS_PANEL
    pupil_center: tuple[int, int] = BEAM_CENTER_PANEL
    defocus_exposure_ms: float | None = None
    settle_s: float = 0.5
    slm_number: int = 1
    slm_wavelength_nm: int = 1064
    cam_type: str = "daheng"
    cam_id: int = 0
    wait_until_idle_s: float = 0.0


def _parse_float_list(text: str, label: str) -> list[float]:
    values: list[float] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} 里 {chunk!r} 不是数字") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} 不能是空列表")
    return values


def _parse_center(text: str) -> tuple[int, int]:
    """把 ``"x,y"`` 解析成面板坐标。"""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--pupil-center 必须是 'x,y', 收到 {text!r}")
    try:
        return (int(float(parts[0])), int(float(parts[1])))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--pupil-center 不是两个数字: {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--exposures",
        default="0.05,0.2,1,2,5,10,20,50",
        help="曝光阶梯 (ms, 逗号分隔)",
    )
    parser.add_argument(
        "--defocus-amps",
        default="0,0.5,1.0,-1.0",
        help="离焦幅度 (rad, Noll 4, 逗号分隔; 0 = 平场)",
    )
    parser.add_argument("--slm-number", type=int, default=1, help="SLM 编号")
    parser.add_argument("--slm-wavelength", type=int, default=1064, help="SLM 工作波长 (nm)")
    parser.add_argument("--cam-type", default="daheng", help="相机类型")
    parser.add_argument("--cam-id", type=int, default=0, help="相机编号")
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        help="设备被占时的等待预算 (s); 0 = 不等, 立刻报错",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "debug" / "optical_path_probe",
        help="证据输出根目录",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=2,
        help="曝光阶梯每档抓几帧",
    )
    parser.add_argument(
        "--defocus-frames",
        type=int,
        default=5,
        help="离焦扫描每个状态抓几帧 (平场 + 平场复测 ⇒ 噪声尺度按组估计)",
    )
    parser.add_argument(
        "--defocus-radius",
        type=int,
        default=BEAM_RADIUS_PANEL,
        help="离焦 Zernike 的孔径半径 (面板 px)",
    )
    parser.add_argument(
        "--pupil-center",
        default=f"{BEAM_CENTER_PANEL[0]},{BEAM_CENTER_PANEL[1]}",
        help="瞳面中心 (面板 px, x,y) —— 离焦盘以它为准, 偏心会带上倾斜与像散",
    )
    parser.add_argument(
        "--defocus-exposure-ms",
        type=float,
        default=None,
        help="离焦扫描用的曝光 (ms); 缺省 = 自动取建议曝光",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.5,
        help="每次改面板后的稳定等待 (s)",
    )
    parser.add_argument("--no-plot", action="store_true", help="不画图")
    return parser


def main(argv: list[str] | None = None) -> int:
    """跑一次探测, 返回退出码 (0 = ok, 2 = low_snr / no_response)。"""
    args = build_parser().parse_args(argv)
    exposures = _parse_float_list(args.exposures, "--exposures")
    amplitudes = _parse_float_list(args.defocus_amps, "--defocus-amps")
    pupil_center = _parse_center(args.pupil_center)
    for exposure in exposures:
        if exposure <= 0:
            raise SystemExit(f"曝光必须为正: {exposure}")

    acquire_devices(args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = Path(args.out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ProbeConfig(
        exposures_ms=[float(v) for v in exposures],
        defocus_amps_rad=[float(v) for v in amplitudes],
        frames_per_exposure=int(args.frames),
        frames_per_defocus=int(args.defocus_frames),
        defocus_radius_px=int(args.defocus_radius),
        pupil_center=pupil_center,
        defocus_exposure_ms=(
            float(args.defocus_exposure_ms)
            if args.defocus_exposure_ms is not None
            else None
        ),
        settle_s=float(args.settle_s),
        slm_number=int(args.slm_number),
        slm_wavelength_nm=int(args.slm_wavelength),
        cam_type=str(args.cam_type),
        cam_id=int(args.cam_id),
        wait_until_idle_s=float(args.wait_until_idle),
    )

    from ao_shaping.drivers.slm.santec import Santec

    slm = Santec(
        slm_number=config.slm_number,
        wavelength=config.slm_wavelength_nm,
        video_mode=0,  # 恒用内存模式: DVI 模式的 open() 会挂死
    )
    camera = create_camera(
        config.cam_type,
        cam_id=config.cam_id,
        exposure_time_ms=float(exposures[0]),
    )

    device_info: dict[str, Any] = {}
    with slm:
        with camera:
            device_info = _collect_device_info(slm, camera)
            print(
                f"设备就绪: SLM #{config.slm_number} "
                f"(序列号 {device_info.get('slm_serial')}), 相机 {config.cam_type} "
                f"#{config.cam_id} (序列号 {device_info.get('cam_serial')})",
                flush=True,
            )
            lo, hi = get_camera_exposure_range(camera)
            print(f"相机曝光范围: {lo:g}-{hi:g} ms", flush=True)

            # -- 阶段一: 曝光阶梯 (面板保持平场) --------------------------
            flat_phase = build_defocus_phase(
                0.0, config.defocus_radius_px, config.pupil_center
            )
            display_phase_on_slm(slm, flat_phase, config.settle_s)
            print("\n[阶段一] 曝光阶梯 (平场):", flush=True)
            ladder = measure_exposure_ladder(
                camera, exposures, config.frames_per_exposure, out_dir
            )
            for record in ladder:
                print(
                    f"  {record['effective_ms']:>8.3f} ms  峰值={record['peak_raw']:>7.1f} "
                    f"(处理后 {record['peak_prepared']:>7.1f})  "
                    f"均值={record['mean_raw']:>7.3f}  p99={record['p99_raw']:>7.1f}  "
                    f"FWHM={record['fwhm_px']:>6.2f} px  "
                    f"质心=({record['centroid_x']:.0f}, {record['centroid_y']:.0f})",
                    flush=True,
                )

            ladder_summary = analyse_ladder(ladder, 255.0)
            noise = ladder_summary["noise_floor"]
            print(
                f"  噪声底: 中位数 {noise['median_counts']:.1f} + "
                f"{NOISE_FLOOR_SIGMAS:g}σ ({noise['sigma_counts']:.2f}) = "
                f"{noise['threshold_counts']:.1f} counts @ {noise['exposure_ms']:g} ms",
                flush=True,
            )
            print(
                "  可用窗口: "
                + (
                    ", ".join(f"{v:g}" for v in ladder_summary["usable_window_ms"])
                    + " ms"
                    if ladder_summary["usable_window_ms"]
                    else "(空)"
                ),
                flush=True,
            )
            recommended = ladder_summary.get("recommended_exposure")
            if recommended is not None:
                print(
                    f"  建议曝光: {recommended['exposure_ms']:g} ms "
                    f"(峰值 {recommended['peak_counts']:.0f} counts)",
                    flush=True,
                )

            # -- 阶段二: 离焦响应 ---------------------------------------
            if config.defocus_exposure_ms is not None:
                defocus_exposure = float(config.defocus_exposure_ms)
            elif recommended is not None:
                defocus_exposure = float(recommended["exposure_ms"])
            else:
                usable = [r for r in ladder if r.get("usable")]
                pool = usable or ladder
                defocus_exposure = float(max(pool, key=lambda r: r["peak_raw"])["effective_ms"])
            print(
                f"\n[阶段二] 离焦响应 (曝光 {defocus_exposure:g} ms, "
                f"幅度 {config.defocus_amps_rad}):",
                flush=True,
            )
            states = measure_defocus_sweep(
                camera,
                slm,
                config.defocus_amps_rad,
                defocus_exposure,
                config.frames_per_defocus,
                config.defocus_radius_px,
                config.pupil_center,
                config.settle_s,
                out_dir,
            )
            for state in states:
                print(
                    f"  离焦 {state['defocus_rad']:>+6.2f} rad  "
                    f"峰值={state['peak_raw']:>7.1f}  "
                    f"FWHM={state['fwhm_px']:>6.2f} px  "
                    f"二阶矩半径={state['second_moment_radius_px']:>6.2f} px  "
                    f"质心=({state['centroid_x']:.0f}, {state['centroid_y']:.0f})  "
                    f"[{state['tag']}]",
                    flush=True,
                )

            response = analyse_defocus_response(
                states, float(noise["threshold_counts"])
            )
            print(
                f"  平场: 峰值 {response['baseline_peak_counts']:.1f} counts, "
                f"FWHM {response['baseline_fwhm_px']:.2f} px; "
                f"逐帧散布 σ(峰值)={response['sigma_peak_counts']:.2f} counts, "
                f"σ(FWHM)={response['sigma_fwhm_px']:.2f} px",
                flush=True,
            )
            print(
                f"  离焦带来的变化 (各幅度中最大): 峰值 "
                f"{response['peak_signal_counts']:.1f} counts "
                f"({response['peak_relative_change']:.1%}, 离焦 "
                f"{response['peak_signal_at_rad']:+g} rad), "
                f"宽度 {response['width_signal_px']:.2f} px "
                f"({response['width_relative_change']:.1%}, 离焦 "
                f"{response['width_signal_at_rad']:+g} rad); "
                f"判据 = {RESPONSE_SIGMAS:g}×合并标准误 或 相对下限 "
                f"({WIDTH_RELATIVE_FLOOR:.0%} / {PEAK_RELATIVE_FLOOR:.0%}) ⇒ "
                f"宽度显著={response['width_significant']}, "
                f"峰值显著={response['peak_significant']}",
                flush=True,
            )
            print(
                f"  分辨率: 宽度 {response['width_resolution']:.1%} "
                f"(上限 {WIDTH_RESOLUTION_MAX:.0%}), 峰值 "
                f"{response['peak_resolution']:.1%} "
                f"(上限 {PEAK_RESOLUTION_MAX:.0%})",
                flush=True,
            )

            verdict = decide(ladder_summary, response)

    thresholds = {
        "noise_floor_sigmas": NOISE_FLOOR_SIGMAS,
        "saturation_fraction": SATURATION_FRACTION,
        "response_sigmas": RESPONSE_SIGMAS,
        "fwhm_sigma_floor_px": FWHM_SIGMA_FLOOR_PX,
        "peak_sigma_floor_counts": PEAK_SIGMA_FLOOR_COUNTS,
        "min_visible_peak_counts": DEFAULT_MIN_PEAK,
        "recommended_peak_range_counts": list(RECOMMENDED_PEAK_RANGE),
        "width_relative_floor": WIDTH_RELATIVE_FLOOR,
        "peak_relative_floor": PEAK_RELATIVE_FLOOR,
        "width_resolution_max": WIDTH_RESOLUTION_MAX,
        "peak_resolution_max": PEAK_RESOLUTION_MAX,
        "defocus_noll_index": DEFOCUS_NOLL_INDEX,
    }
    payload = {
        "schema": "optical_path_probe/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "config": {
            "exposures_ms": config.exposures_ms,
            "defocus_amps_rad": config.defocus_amps_rad,
            "frames_per_exposure": config.frames_per_exposure,
            "frames_per_defocus": config.frames_per_defocus,
            "defocus_radius_px": config.defocus_radius_px,
            "pupil_center_px": list(config.pupil_center),
            "defocus_exposure_ms": defocus_exposure,
            "settle_s": config.settle_s,
            "slm_number": config.slm_number,
            "slm_wavelength_nm": config.slm_wavelength_nm,
            "cam_type": config.cam_type,
            "cam_id": config.cam_id,
            "wait_until_idle_s": config.wait_until_idle_s,
        },
        "device": device_info,
        "exposure_ladder": ladder,
        "ladder_summary": ladder_summary,
        "defocus_sweep": states,
        "defocus_response": response,
        "thresholds": thresholds,
        "verdict": verdict,
    }
    json_path = out_dir / "optical_path_probe.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_plot:
        plot_results(
            ladder,
            ladder_summary,
            response,
            out_dir / "optical_path_probe.png",
        )

    print(f"\n判定: {verdict['state']}")
    for reason in verdict["reasons"]:
        print(f"  - {reason}")
    for item in verdict["advice"]:
        print(f"  * {item}")
    print(f"证据: {out_dir}")
    return int(verdict["exit_code"])


def _collect_device_info(slm: Any, camera: Any) -> dict[str, Any]:
    """收集设备标识 (序列号等), 失败不致命。"""
    info: dict[str, Any] = {}
    try:
        info["slm_serial"] = slm.serial_number
    except Exception:  # 设备信息拿不到不影响判定
        info["slm_serial"] = None
    try:
        info["slm_video_mode"] = int(getattr(slm, "video_mode", 0))
    except Exception:
        info["slm_video_mode"] = None
    try:
        info["slm_wavelength_nm"] = getattr(slm, "wavelength", None)
    except Exception:
        info["slm_wavelength_nm"] = None
    for attr in ("serial_number", "serial", "sn"):
        value = getattr(camera, attr, None)
        if isinstance(value, str) and value:
            info["cam_serial"] = value
            break
    else:
        info["cam_serial"] = None
    try:
        hardware = camera.get_hardware_info()
        if isinstance(hardware, dict):
            info["cam_hardware"] = {
                str(k): str(v) for k, v in hardware.items() if isinstance(v, (str, int, float))
            }
    except Exception:  # 大恒驱动没有实现 get_hardware_info
        pass
    try:
        lo, hi = get_camera_exposure_range(camera)
        info["cam_exposure_range_ms"] = [float(lo), float(hi)]
    except Exception:
        pass
    return info


if __name__ == "__main__":
    raise SystemExit(main())
