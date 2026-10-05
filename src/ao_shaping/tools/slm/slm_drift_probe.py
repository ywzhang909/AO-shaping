"""Flat-field drift and exposure-ladder probe for the SLM 2f-Fourier bench.

Two questions, one run, at a *fixed* exposure:

**(a) Drift.** How much does the flat field move between two reads? The
observable is the L2 norm of the difference between consecutive reads plus the
relative spread of a box sum -- *not* peak intensity. Peak is not reproducible
run to run on this bench (measured), so a peak-based drift number reports bench
non-reproducibility as if it were drift.

**(b) Exposure ladder.** Step the exposure through a ladder, repeat each level,
and judge whether brightness rises monotonically and stays off the detector's
full scale. This is the cheap check that the camera is being driven inside its
linear range; the verdict comes from
:func:`~ao_shaping.tools.slm.slm_bench_metrics.exposure_monotonicity`.

Layering
--------
The analysis is two pure functions (:func:`drift_series`,
:func:`exposure_ladder`) that take a ``capture() -> frame`` callable and never
touch a device, so the whole probe logic is unit-testable offline. Only
:func:`main` constructs hardware. All the frame maths is reused from
:mod:`ao_shaping.tools.slm.slm_bench_metrics` -- nothing here re-implements
``finite_clip``, ``crop_roi``, ``roi_l2``, ``flat_to_flat_floor`` or
``exposure_monotonicity``, because a second copy of a metric silently drifts
from the first (see the sweep probe's docstring on the same hazard).

Bench rules this probe obeys, and why
------------------------------------
* **Flat phase is displayed, never assumed.** The panel retains the last pattern,
  so a "flat" read taken before any write is really the previous experiment's
  speckle.
* **No pinned ``memory_number``.** ``display_memory(slot)`` is a firmware no-op
  when that slot is already displayed, so the panel silently keeps the previous
  pattern. :func:`capture_settled` -> :func:`display_and_average` calls
  ``display_data()`` and lets the driver rotate slots.
* **No fixed settle wait.** The driver's LCOS flip-time estimate is driven by how
  much the gray map changed and reports **0.0 ms** for two phases with similar
  gray statistics -- measured on this bench as the same ramp reading fwhm 43.2 px
  immediately and 12.8 px three seconds later. Passing ``wait_time_s=None``
  leaves the driver on its estimate and lets :func:`capture_settled`'s
  two-consecutive-agreement loop be the thing that actually decides the panel
  has settled. A fixed sleep is never a substitute for that test.
* **The ROI is located once.** ``argmax`` on a speckle field hops between
  near-equal grains, so a re-located ROI makes the metrics discontinuous and the
  numbers meaningless. The centre is found on the *first* settled flat frame and
  frozen for the whole run.
* **The box sum uses :func:`finite_clip`, not ``finite_median_subtract`.**
  ``finite_median_subtract`` is for whole-frame-denominator ratios: it nulls the
  background pedestal, which is exactly what a *plain box sum* needs to keep.
  Here the pedestal is part of the observable (it is most of the frame), so it
  must survive; only non-finite values are masked and negatives clipped.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import click
import numpy as np

from loguru import logger

from ao_shaping.tools.slm.slm_bench_metrics import (
    crop_roi,
    exposure_monotonicity,
    finite_clip,
    flat_to_flat_floor,
)
from ao_shaping.tools.slm.slm_bench_probe import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    despike_frame,
    measure_spot,
    smooth_frame,
)
from ao_shaping.tools.slm.slm_zernike_sweep_probe import capture_settled
from ao_shaping.utils.cli.params import option, with_params

__all__ = [
    "DRIFT_REPEAT_TOL",
    "LADDER_DEFAULT",
    "ROW_KEYS",
    "SETTLE_DISCARD",
    "SETTLE_FRAMES",
    "SETTLE_MAX_WAIT_S",
    "SETTLE_STABLE_TOL",
    "DriftProbeParams",
    "drift_series",
    "exposure_ladder",
    "main",
]

#: Every persisted row carries exactly these keys.
ROW_KEYS: tuple[str, ...] = (
    "peak", "sum", "norm", "fwhm_px", "exposure_ms", "repeat", "_epoch",
)

#: Default exposure ladder in ms: short enough to resolve the dark end, long
#: enough to approach full scale.
LADDER_DEFAULT: tuple[float, ...] = (
    0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0,
)

#: Settle configuration for :func:`capture_settled`. These are module constants
#: rather than CLI flags on purpose: the flag surface stays the documented one,
#: and every value lands in the JSON sidecar for provenance.
SETTLE_FRAMES: int = 4
SETTLE_DISCARD: int = 3
SETTLE_STABLE_TOL: float = 0.02
SETTLE_MAX_WAIT_S: float = 6.0

#: Relative band for the drift repeatability verdict. Deliberately loose (the
#: ladder's linearity test uses 2%): at a fixed exposure the expected ratio is
#: 1.0 and ordinary read noise moves the peak by a few percent, so a 2% band
#: would report read noise as a fault.
DRIFT_REPEAT_TOL: float = 0.10


# ---------------------------------------------------------------------------
# Pure analysis -- no devices, no I/O
# ---------------------------------------------------------------------------


def _region(prepped: np.ndarray, center: tuple[int, int] | None,
            half: int | None) -> np.ndarray:
    """The area the box-sum observable is computed over.

    Falls back to the whole frame when no ROI is supplied, so the pure functions
    are usable without one.

    Args:
        prepped: A frame already passed through :func:`finite_clip`.
        center: ROI centre as ``(x, y)``, or ``None`` for the whole frame.
        half: ROI half-width in pixels.

    Returns:
        The ROI crop, or ``prepped`` itself.
    """
    if center is None or half is None:
        return prepped
    return crop_roi(prepped, center, int(half))


def _measure(
    frame: np.ndarray,
    index: int,
    exposure_ms: float | None,
    repeat: int | None,
    center: tuple[int, int] | None,
    half: int | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """One row of :data:`ROW_KEYS`, plus the region it was measured on.

    The box sum and the L2 norm are taken over ``finite_clip(frame)`` restricted
    to the frozen ROI. Peak and FWHM come from :func:`measure_spot`, which
    despikes and blurs first -- a bare ``argmax`` on a dim speckle frame locks
    onto a single hot pixel (bench fact 1).

    Args:
        frame: The raw camera frame.
        index: Value for ``_epoch``.
        exposure_ms: Exposure in ms, or ``None`` if unknown.
        repeat: Repeat index within an exposure level, or ``None`` during drift.
        center: Frozen ROI centre ``(x, y)``, or ``None`` for the whole frame.
        half: ROI half-width in pixels.

    Returns:
        ``(row, region)``; the region is kept so consecutive differences can be
        taken between exactly the pixels the box sum was measured on.
    """
    img = np.asarray(frame, dtype=np.float64)
    region = _region(finite_clip(img), center, half)
    spot = measure_spot(img)
    row: dict[str, Any] = {
        "peak": float(spot.peak),
        "sum": float(region.sum()),
        "norm": float(np.linalg.norm(region.ravel())),
        "fwhm_px": float(spot.fwhm_px),
        "exposure_ms": None if exposure_ms is None else float(exposure_ms),
        "repeat": None if repeat is None else int(repeat),
        "_epoch": int(index),
    }
    return row, region


def _cv_pct(values: Sequence[float]) -> float:
    """Relative spread of ``values`` as a percentage (``nan`` if undefined).

    Args:
        values: The samples to spread.

    Returns:
        ``100 * std / mean``, or ``nan`` when there is nothing to spread or the
        mean is not positive (a relative spread of a non-positive signal is
        undefined, and reporting 0.0 there would read as "perfectly stable").
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    mean = float(arr.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        return float("nan")
    return float(arr.std() / mean * 100.0)


def drift_series(
    capture: Callable[[], np.ndarray],
    n: int,
    *,
    first_frame: np.ndarray | None = None,
    center: tuple[int, int] | None = None,
    half: int | None = None,
    exposure_ms: float | None = None,
    rel_tol: float = DRIFT_REPEAT_TOL,
) -> tuple[list[dict], dict]:
    """Capture the flat field ``n`` times and quantify how much it moves.

    Drift is read as two numbers: the median L2 norm between consecutive reads
    (the *drift floor*), and the relative spread of the box sum across the run.
    Both are region observables, not peak observables -- see the module
    docstring.

    Args:
        capture: Takes no arguments, returns one frame.
        n: Number of records to produce.
        first_frame: An already-captured frame to use as record 0. Lets the
            caller locate the ROI from the first settled flat frame without
            paying for a second read of it.
        center: Frozen ROI centre ``(x, y)``, or ``None`` for the whole frame.
        half: ROI half-width in pixels.
        exposure_ms: Exposure held fixed across the series.
        rel_tol: Relative band for the repeatability verdict.

    Returns:
        ``(records, summary)``. ``records`` has exactly ``n`` dicts with the
        keys of :data:`ROW_KEYS` and ``repeat`` set to ``None``. ``summary``
        carries ``drift_floor_l2``, ``box_sum_cv_pct`` and ``verdict``.

    Raises:
        ValueError: If fewer than two frames are requested.
    """
    count = int(n)
    if count < 2:
        raise ValueError(f"need n >= 2 to measure drift, got {n!r}")
    frames = [np.asarray(first_frame, dtype=np.float64)] if first_frame is not None else []
    frames.extend(np.asarray(capture(), dtype=np.float64) for _ in range(count - len(frames)))

    measured = [
        _measure(frames[i], i, exposure_ms, None, center, half) for i in range(count)
    ]
    records = [row for row, _ in measured]
    regions = [region for _, region in measured]

    floor, series = flat_to_flat_floor(regions)
    # At a fixed exposure every expected ratio is 1.0, so feeding the fixed
    # exposure back through `exposure_monotonicity` turns the kernel into a
    # repeatability test: does the peak repeat to within `rel_tol`?
    fixed_ms = 1.0 if exposure_ms is None else float(exposure_ms)
    repeatability = exposure_monotonicity(
        [fixed_ms] * count, [r["peak"] for r in records], rel_tol=rel_tol
    )
    summary: dict[str, Any] = {
        "n": count,
        "exposure_ms": exposure_ms,
        "drift_floor_l2": float(floor),
        "drift_l2_series": [float(v) for v in series],
        "box_sum_cv_pct": _cv_pct([r["sum"] for r in records]),
        "box_sum_mean": float(np.mean([r["sum"] for r in records])),
        "peak_cv_pct": _cv_pct([r["peak"] for r in records]),
        "peak_spread_pct": float(
            100.0 * (max(r["peak"] for r in records) - min(r["peak"] for r in records))
            / max(max(r["peak"] for r in records), 1e-12)
        ),
        "verdict": str(repeatability["verdict"]),
        "verdict_kind": "repeatability_at_fixed_exposure",
        "roi_center": None if center is None else [int(center[0]), int(center[1])],
        "roi_half": None if half is None else int(half),
    }
    logger.info(
        "漂移: {} 帧, 地板 {:.4g}, box-sum cv {:.3f}%, 峰值离散 {:.2f}%",
        count, floor, summary["box_sum_cv_pct"], summary["peak_spread_pct"],
    )
    return records, summary


def exposure_ladder(
    capture: Callable[[], np.ndarray],
    set_exposure: Callable[[float], None],
    ladder: Sequence[float],
    repeats: int,
    *,
    center: tuple[int, int] | None = None,
    half: int | None = None,
    saturation_level: float | None = None,
    rel_tol: float = 0.02,
) -> tuple[list[dict], dict]:
    """Step through an exposure ladder and judge monotonicity and saturation.

    Each level is read ``repeats`` times. The per-level peak is the mean of its
    repeats; the repeat-to-repeat spread within a level is itself a drift
    measurement, so it is folded into the summary rather than discarded.

    Args:
        capture: Takes no arguments, returns one frame.
        set_exposure: Sets the camera exposure in ms.
        ladder: Exposures in ms. Need not be sorted; used in the given order.
        repeats: Reads per level.
        center: Frozen ROI centre ``(x, y)``, or ``None`` for the whole frame.
        half: ROI half-width in pixels.
        saturation_level: Detector full-scale value; flags saturation when a
            peak reaches it. ``None`` disables the check.
        rel_tol: Allowed relative shortfall on each step.

    Returns:
        ``(records, summary)``. ``records`` has ``len(ladder) * repeats`` dicts
        with the keys of :data:`ROW_KEYS` and a populated ``repeat``.
        ``summary`` carries ``verdict`` (the ladder verdict string), plus
        ``drift_floor_l2`` and ``box_sum_cv_pct`` measured within each level.

    Raises:
        ValueError: If the ladder is shorter than two levels or ``repeats`` < 1.
    """
    levels = [float(v) for v in ladder]
    if len(levels) < 2:
        raise ValueError(f"ladder needs at least two exposures, got {levels!r}")
    n_repeat = int(repeats)
    if n_repeat < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats!r}")

    records: list[dict[str, Any]] = []
    level_peaks: list[float] = []
    level_sums: list[float] = []
    within_l2: list[float] = []
    within_cv: list[float] = []

    for ms in levels:
        set_exposure(ms)
        measured: list[tuple[dict[str, Any], np.ndarray]] = []
        for repeat in range(n_repeat):
            frame = np.asarray(capture(), dtype=np.float64)
            measured.append(
                _measure(frame, len(records), ms, repeat, center, half)
            )
            records.append(measured[-1][0])
        if len(measured) >= 2:
            _, series = flat_to_flat_floor([region for _, region in measured])
            within_l2.extend(float(v) for v in series)
        level_peaks.append(float(np.mean([row["peak"] for row, _ in measured])))
        level_sums.append(float(np.mean([row["sum"] for row, _ in measured])))
        within_cv.append(_cv_pct([row["sum"] for row, _ in measured]))

    mono = exposure_monotonicity(
        levels, level_peaks, rel_tol=rel_tol, saturation_level=saturation_level
    )
    summary: dict[str, Any] = {
        "ladder": levels,
        "repeats": n_repeat,
        "level_peaks": level_peaks,
        "level_sums": level_sums,
        "verdict": str(mono["verdict"]),
        "verdict_kind": "exposure_ladder",
        "ratios": [float(v) for v in mono["ratios"]],
        "expected": [float(v) for v in mono["expected"]],
        "saturated": bool(mono["saturated"]),
        "saturation_level": saturation_level,
        "rel_tol": float(rel_tol),
        "drift_floor_l2": float(np.median(within_l2)) if within_l2 else 0.0,
        "box_sum_cv_pct": float(np.median(within_cv)) if within_cv else float("nan"),
        "roi_center": None if center is None else [int(center[0]), int(center[1])],
        "roi_half": None if half is None else int(half),
    }
    logger.info(
        "曝光阶梯: {} ms x{} -> {} (比值 {} / 期望 {})",
        levels, n_repeat, summary["verdict"],
        [round(v, 3) for v in summary["ratios"]],
        [round(v, 3) for v in summary["expected"]],
    )
    return records, summary


# ---------------------------------------------------------------------------
# Hardware orchestration -- only this reaches a device
# ---------------------------------------------------------------------------


def _locate_zero_order(frame: np.ndarray) -> tuple[int, int]:
    """Find the 0-order on a settled flat frame, as ``(x, y)``.

    ``argmax`` on the raw frame is unsafe (bench fact 1: one hot pixel beats a
    dim spot and drags the centre with it), so the frame is despiked and blurred
    first -- the same preparation :func:`measure_spot` uses.

    Args:
        frame: A settled flat-field frame.

    Returns:
        The 0-order centre in camera pixels.
    """
    img = smooth_frame(despike_frame(np.asarray(frame, dtype=np.float64), 3), 5)
    cy, cx = np.unravel_index(int(np.argmax(img)), img.shape)
    return int(cx), int(cy)


def _paced(capture: Callable[[], np.ndarray], period_s: float) -> Callable[[], np.ndarray]:
    """Wrap ``capture`` so successive calls are ``period_s`` apart.

    The wait here is the *sampling cadence* of the drift series, not a settle
    wait. Settling is already handled by :func:`capture_settled`'s
    two-consecutive-agreement loop, and this function does not sleep before the
    first read.

    Args:
        capture: The underlying capture callable.
        period_s: Target seconds between the start of successive reads.

    Returns:
        The wrapped callable.
    """
    state: dict[str, float] = {"next": 0.0}
    period = float(period_s)

    def take() -> np.ndarray:
        delay = state["next"] - time.time()
        if delay > 0.0:
            time.sleep(delay)
        state["next"] = time.time() + period
        return capture()

    return take


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in str(text).split(",") if v.strip()]


@dataclass
class DriftProbeParams:
    """平场漂移 + 曝光阶梯探针的 CLI 参数。

    默认值来自本台架的实测标定, 不与其他探针共享: ``--exposure-ms`` 默认 0.4 ms
    (1.1 ms 时 0 阶峰值约 60, 但 0.4 ms 才能看出漂移幅度), ``--exposure-ladder``
    默认取 :data:`LADDER_DEFAULT`。

    ``--saturation-level`` 默认 ``None``, 也就是**不判饱和** —— argparse 的
    ``type=float, default=None`` 在 dataclass 里对应 ``float | None``;
    :func:`main` 再把它转成 npz 的 ``nan``。

    ``--no-hw`` 是**裸 flag** (``is_flag=True``), 不是 click 的
    ``--hw/--no-hw`` 配对布尔: 配对写法会凭空多出一个 ``--hw``, 改变本探针
    声明的 flag 集合。
    """

    out: Annotated[
        str, option("--out", help="输出目录 (默认 data/slm_drift)")
    ] = "data/slm_drift"
    slm_number: Annotated[
        int, option("--slm-number", help="SLM 设备编号 (默认 1)")
    ] = 1
    slm_wavelength: Annotated[
        int, option("--slm-wavelength", help="SLM 波长 nm (默认 1064)")
    ] = 1064
    cam_type: Annotated[
        str,
        option(
            "--cam-type",
            type=click.Choice(["daheng", "miicam"]),
            help="相机类型 (daheng/miicam, 默认 daheng)",
        ),
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id", help="相机 ID (默认 0)")] = 0
    exposure_ms: Annotated[
        float, option("--exposure-ms", help="漂移段固定曝光 (ms, 默认 0.4)")
    ] = 0.4
    n_drift: Annotated[
        int, option("--n-drift", help="漂移采集帧数 (默认 24)")
    ] = 24
    drift_period_s: Annotated[
        float, option("--drift-period-s", help="漂移采样间隔 s (默认 5.0)")
    ] = 5.0
    exposure_ladder: Annotated[
        str, option("--exposure-ladder", help="曝光阶梯 (ms, 逗号分隔)")
    ] = ",".join(str(v) for v in LADDER_DEFAULT)
    ladder_repeats: Annotated[
        int, option("--ladder-repeats", help="每级曝光重复次数 (默认 2)")
    ] = 2
    roi: Annotated[
        int, option("--roi", help="漂移/阶梯 ROI 边长 px (默认 192)")
    ] = 192
    saturation_level: Annotated[
        float | None,
        option("--saturation-level", help="探测器满量程值; 峰值达到即判饱和 (默认不判)"),
    ] = None
    no_hw: Annotated[
        bool,
        option("--no-hw", is_flag=True, help="不打开硬件, 只打印采集计划 (自检用)"),
    ] = False


def _summary_npz(
    drift_summary: dict,
    ladder_summary: dict,
    rows: Sequence[dict[str, Any]],
    params: DriftProbeParams,
) -> dict[str, np.ndarray]:
    """Columns for ``summary.npz``: every series plus the scalar verdicts.

    Args:
        drift_summary: The :func:`drift_series` summary.
        ladder_summary: The :func:`exposure_ladder` summary.
        rows: All persisted rows, drift first then ladder, epochs contiguous.
        params: Parsed CLI parameters.

    Returns:
        A dict of arrays for :func:`numpy.savez_compressed`.
    """

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=np.float64)

    return {
        # Per-record series.
        "peak": col("peak"),
        "sum": col("sum"),
        "norm": col("norm"),
        "fwhm_px": col("fwhm_px"),
        # Was `"exposure_ms"`, colliding with the run-configuration scalar of the
        # same name further down: Python kept the LAST one, so the per-record
        # series never reached summary.npz at all (ruff F601). Named after the
        # existing `drift_l2_series` convention.
        "exposure_ms_series": col("exposure_ms"),
        # Drift rows carry `repeat=None`; NaN is the numpy spelling of that.
        "repeat": col("repeat"),
        "epoch": col("_epoch"),
        # Drift.
        "drift_l2_series": np.array(drift_summary["drift_l2_series"], dtype=np.float64),
        "drift_floor_l2": np.array(drift_summary["drift_floor_l2"]),
        "drift_box_sum_cv_pct": np.array(drift_summary["box_sum_cv_pct"]),
        "drift_peak_spread_pct": np.array(drift_summary["peak_spread_pct"]),
        "drift_verdict": np.array(drift_summary["verdict"]),
        # Ladder.
        "ladder": np.array(ladder_summary["ladder"], dtype=np.float64),
        "ladder_level_peaks": np.array(ladder_summary["level_peaks"], dtype=np.float64),
        "ladder_level_sums": np.array(ladder_summary["level_sums"], dtype=np.float64),
        "ladder_ratios": np.array(ladder_summary["ratios"], dtype=np.float64),
        "ladder_expected": np.array(ladder_summary["expected"], dtype=np.float64),
        "ladder_verdict": np.array(ladder_summary["verdict"]),
        "ladder_saturated": np.array(ladder_summary["saturated"]),
        "ladder_repeats": np.array(ladder_summary["repeats"]),
        # Run configuration.
        "roi_center": np.array(
            [-1, -1] if drift_summary["roi_center"] is None
            else drift_summary["roi_center"]
        ),
        "roi_half": np.array(-1 if drift_summary["roi_half"] is None
                             else drift_summary["roi_half"]),
        "exposure_ms": np.array(params.exposure_ms),
        "n_drift": np.array(params.n_drift),
        "drift_period_s": np.array(params.drift_period_s),
        "settle_frames": np.array(SETTLE_FRAMES),
        "settle_discard": np.array(SETTLE_DISCARD),
        "settle_stable_tol": np.array(SETTLE_STABLE_TOL),
        "settle_max_wait_s": np.array(SETTLE_MAX_WAIT_S),
        "saturation_level": np.array(np.nan if params.saturation_level is None
                                     else params.saturation_level),
    }


@click.command()
@with_params(DriftProbeParams, kw_name="params")
def main(params: DriftProbeParams) -> None:
    """Measure drift then the exposure ladder on hardware, and persist both."""
    from ao_shaping.utils.io.cli_helpers import setup_coredumpy

    setup_coredumpy()
    ladder = _parse_floats(params.exposure_ladder)

    if params.no_hw:
        logger.info("--no-hw: 不打开硬件, 只打印采集计划")
        logger.info("  漂移: {} 帧 @ {} ms, 间隔 {} s", params.n_drift,
                    params.exposure_ms, params.drift_period_s)
        logger.info("  曝光阶梯: {} ms, 每级 {} 次", ladder, params.ladder_repeats)
        logger.info("  ROI 边长 {} px, 饱和阈值 {}", params.roi, params.saturation_level)
        logger.info("  稳定判据: {} 帧平均 / 丢 {} 帧 / tol {} / 上限 {} s",
                    SETTLE_FRAMES, SETTLE_DISCARD, SETTLE_STABLE_TOL,
                    SETTLE_MAX_WAIT_S)
        return

    # Imported here, not at module scope: importing a driver package must never be
    # a side effect of a `--no-hw` self-check.
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.io.file import Recorder, save_recorder_debug_artifacts

    out_dir = Path(params.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    half = int(params.roi) // 2
    flat_phase = np.zeros((SLM_PANEL_H, SLM_PANEL_W), dtype=np.float64)

    with Santec(
        slm_number=params.slm_number, wavelength=params.slm_wavelength, video_mode=0
    ) as slm, create_camera(
        params.cam_type, params.cam_id, exposure_time_ms=params.exposure_ms
    ) as cam:

        def capture_flat() -> np.ndarray:
            """One settled flat-field read.

            ``wait_time_s=None`` leaves the driver's flip-time estimate in place
            and lets ``capture_settled``'s agreement loop decide the panel has
            settled; no fixed sleep is ever used as a settle criterion.
            """
            return capture_settled(
                cam, slm, flat_phase, n_frames=SETTLE_FRAMES,
                n_discard=SETTLE_DISCARD, wait_time_s=None,
                stable_tol=SETTLE_STABLE_TOL, max_wait_s=SETTLE_MAX_WAIT_S,
            )

        def set_exposure(exposure_ms: float) -> None:
            cam.reset_exposure_time(float(exposure_ms))

        # The ROI comes from the FIRST settled flat frame and is then frozen:
        # re-locating it per iteration makes the metrics discontinuous, because
        # argmax hops between near-equal speckle grains.
        reference = capture_flat()
        centre = _locate_zero_order(reference)
        logger.info("0 阶定位于 ({}), ROI {} px (整轮冻结)", centre, params.roi)

        drift_records, drift_summary = drift_series(
            _paced(capture_flat, params.drift_period_s), params.n_drift,
            first_frame=reference, center=centre, half=half,
            exposure_ms=params.exposure_ms,
        )
        ladder_records, ladder_summary = exposure_ladder(
            capture_flat, set_exposure, ladder, params.ladder_repeats,
            center=centre, half=half, saturation_level=params.saturation_level,
        )
        set_exposure(float(params.exposure_ms))

    rows: list[dict[str, Any]] = [{**r, "_epoch": i} for i, r in enumerate(
        [*drift_records, *ladder_records]
    )]
    columns = _summary_npz(drift_summary, ladder_summary, rows, params)
    npz_path = out_dir / "summary.npz"
    np.savez_compressed(npz_path, **columns)
    logger.info("summary -> {}", npz_path)

    recorder = Recorder(mark="peak", mode="min")
    for row in rows:
        recorder.append(row)
    # `repeat` is deliberately absent from `scalar_keys`: it is None on every
    # drift row and the serialiser casts scalars with float(), which would raise
    # on it. It rides along in the npz instead, as NaN.
    save_recorder_debug_artifacts(
        recorder,
        str(out_dir),
        "slm_drift",
        scalar_keys=("peak", "sum", "norm", "fwhm_px", "exposure_ms"),
        json_payload={
            "summary_npz": str(npz_path),
            "drift_summary": drift_summary,
            "ladder_summary": ladder_summary,
            "roi_center": list(centre),
            "roi_side_px": int(params.roi),
            "cam_type": params.cam_type,
            "cam_id": params.cam_id,
            "slm_number": params.slm_number,
            "slm_wavelength": params.slm_wavelength,
            "exposure_ms": params.exposure_ms,
            "n_drift": params.n_drift,
            "drift_period_s": params.drift_period_s,
            "exposure_ladder": ladder,
            "ladder_repeats": params.ladder_repeats,
            "saturation_level": params.saturation_level,
            "settle_frames": SETTLE_FRAMES,
            "settle_discard": SETTLE_DISCARD,
            "settle_stable_tol": SETTLE_STABLE_TOL,
            "settle_max_wait_s": SETTLE_MAX_WAIT_S,
        },
        title="SLM drift + exposure ladder",
    )
    logger.info("漂移地板 {:.4g}, 阶梯判定 {}, 产物 -> {}",
                drift_summary["drift_floor_l2"], ladder_summary["verdict"], out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()