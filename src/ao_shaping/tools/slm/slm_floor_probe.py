"""SLM floor-characterisation probe for the 2f-Fourier bench.

Three questions, in the order they have to be answered before any closed-loop
work is worth attempting:

1. **What is the bench's own noise floor?** :func:`repeatability_floor` --
   consecutive flat reads, median ``roi_l2`` difference
   (:func:`~ao_shaping.tools.slm.slm_bench_metrics.flat_to_flat_floor`).
2. **How long does the panel take to settle after a phase write?**
   :func:`settle_curve` -- sample one observable until it plateaus
   (:func:`~ao_shaping.tools.slm.slm_bench_metrics.settle_time_s`). A fixed
   sleep is invalid here: the driver's flip-time estimate under-reports (it
   reports 0.0 ms for two phases with similar grey statistics), so the same ramp
   read 43.2 px FWHM immediately and 12.8 px three seconds later.
3. **Does averaging K frames actually buy SNR?**
   :func:`snr_ladder` -- :func:`~ao_shaping.tools.slm.slm_bench_metrics.snr_vs_averages`.
   This is the diagnostic that decides whether the residual is independent read
   noise (SNR grows like ``sqrt(K)``, longer averaging helps) or drift (SNR
   flat or falling, longer averaging is wasted exposure).

Nothing here re-derives frame preparation or drift statistics: the pure kernels
live in :mod:`ao_shaping.tools.slm.slm_bench_metrics` and this module is only
the protocol plus the CLI, exactly as
:mod:`~ao_shaping.tools.slm.slm_abba_probe` relates to the same kernels.

Units discipline (see ``AGENTS.md``): every phase is **raw unwrapped radians**
and is converted to grayscale exactly once, by
:func:`~ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale`. No generator
in this file wraps mod 2*pi.

Run it with ``python -m ao_shaping.tools.slm.slm_floor_probe``;
``--no-hw`` prints the acquisition plan and returns without importing a driver.
"""

from __future__ import annotations

import math
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.slm_bench_metrics import (
    build_block_pattern,
    crop_roi,
    finite_clip,
    flat_to_flat_floor,
    roi_l2,
    settle_time_s,
    snr_vs_averages,
)
from ao_shaping.tools.slm.slm_bench_probe import measure_spot
from ao_shaping.tools.slm.slm_snr_probe import SIGMA_FLOOR
from ao_shaping.utils.cli.params import option, with_params

#: Full width of the analysis ROI in pixels; half of this is the half-width
#: handed to :func:`~ao_shaping.tools.slm.slm_bench_metrics.crop_roi`.
DEFAULT_ROI = 192
DEFAULT_ROI_HALF = DEFAULT_ROI // 2

#: Perturbation amplitude in radians for the settle and SNR stages. Large enough
#: to rearrange the focal-plane speckle well above the read-noise floor.
DEFAULT_PERTURBATION_RAD = 0.6

#: Edge length of the block-coefficient grid driving the perturbation
#: (24x24 = 576 DOF, the freeform basis the shaping optimizers use).
DEFAULT_GRID = 24

DEFAULT_EXPOSURE_MS = 1.5
DEFAULT_N_REPEAT = 20
DEFAULT_SETTLE_CURVE_S = 4.0
DEFAULT_SETTLE_SAMPLE_MS = 120.0
DEFAULT_KS: tuple[int, ...] = (1, 4, 9)

#: Settle tolerance as a fraction of the curve's final value, and how many
#: consecutive samples must sit inside it. These are
#: :func:`~ao_shaping.tools.slm.slm_bench_metrics.settle_time_s` defaults.
DEFAULT_SETTLE_FRAC = 0.10
DEFAULT_SETTLE_RUN = 3

# ``capture_settled`` discipline -- discard frames until two consecutive reads
# agree, because a single-shot read silently records an unsettled panel.
DEFAULT_FRAMES = 4
DEFAULT_DISCARD = 3
DEFAULT_CAPTURE_SETTLE_S = 0.5
DEFAULT_STABLE_TOL = 0.02
DEFAULT_MAX_WAIT_S = 6.0

STAGE_REPEAT = "repeat"
STAGE_SETTLE = "settle"
STAGE_SNR = "snr"

Frame = np.ndarray
Capture = Callable[[], Frame]


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------


def prepare_roi_frame(
    frame: np.ndarray, center: tuple[int, int], half: int
) -> np.ndarray:
    """Clip negatives, then crop the frozen ROI.

    Two steps, both from :mod:`~ao_shaping.tools.slm.slm_bench_metrics`:

    1. :func:`~ao_shaping.tools.slm.slm_bench_metrics.finite_clip` masks
       non-finite pixels and clips negatives **while leaving the pedestal
       alone**. This is deliberate and is not
       :func:`~ao_shaping.tools.slm.slm_bench_metrics.finite_median_subtract`:
       every observable here is a *region norm* (a plain L2 over the ROI, or a
       box sum against a flat reference), not a whole-frame ratio, so the flat
       pedestal is part of the signal we are measuring rather than the offset we
       want to remove. Subtracting the median would delete the very energy the
       drift floor and the SNR ladder are built on, and on this bench it is
       also what turns a peak-to-background ratio into ~1e5 when the
       denominator collapses.
    2. :func:`~ao_shaping.tools.slm.slm_bench_metrics.crop_roi` cuts a
       ``2*half`` square about the **measured** centre. The 0-order is the frame
       global maximum and is routinely nowhere near the geometric centre (the
       ROI maths must never be driven by ``shape // 2``).

    Args:
        frame: Raw 2-D camera frame.
        center: ROI centre as ``(x, y)`` pixels, frozen for the whole run.
        half: ROI half-width in pixels.

    Returns:
        A non-negative ``(2*half, 2*half)`` float64 array.
    """
    return crop_roi(finite_clip(frame), center, half)


def _fwhm_px(frame: np.ndarray) -> float:
    """FWHM via :func:`measure_spot`, or ``nan`` on a degenerate ROI.

    A synthetic or saturated crop can leave the spot profile without a lobe,
    and a ``nan`` width is a far better outcome for a 34-sample settle run than
    an exception from the width estimator.
    """
    try:
        return float(measure_spot(frame).fwhm_px)
    except (ValueError, IndexError, ZeroDivisionError) as exc:
        logger.debug("FWHM 估计失败, 记为 nan: {}", exc)
        return float("nan")


def _frame_metrics(frame: np.ndarray) -> dict[str, float]:
    """Per-row observables shared by all three stages.

    ``peak``/``sum`` describe the frame; ``norm`` is deliberately *not* filled
    here -- each stage defines its own ``norm`` observable and documents what it
    means (``stage`` says which).
    """
    arr = np.asarray(frame, dtype=np.float64)
    return {
        "peak": float(arr.max()) if arr.size else 0.0,
        "sum": float(arr.sum()),
        "fwhm_px": _fwhm_px(arr),
    }


def _magnitude(frame: np.ndarray) -> float:
    """L2 magnitude of a whole ROI crop.

    Used as the settle observable because it *plateaus at a nonzero value* once
    the panel stops moving. ``settle_time_s`` derives its tolerance from
    ``frac * |d[-1]|``, so a "distance to the final frame" observable would end
    at exactly 0.0, collapse the tolerance to zero and take its degenerate
    ``tol <= 0`` branch -- returning ``None`` for a perfectly ordinary decay.
    """
    return float(np.linalg.norm(np.asarray(frame, dtype=np.float64).ravel()))


# ---------------------------------------------------------------------------
# Stage 1 -- repeatability floor
# ---------------------------------------------------------------------------


def repeatability_floor(
    capture: Capture, n: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drift floor from ``n`` consecutive flat reads.

    The caller must leave the bench untouched between calls (a flat phase
    already written), because this is what every later stage competes against:
    shot noise *and* slow intensity drift, not shot noise alone.

    Args:
        capture: Zero-arg callable returning one prepared ROI crop (see
            :func:`prepare_roi_frame`).
        n: Number of consecutive reads; must be at least 2.

    Returns:
        ``(rows, summary)``. Each row carries ``stage="repeat"``, ``peak``,
        ``sum``, ``fwhm_px`` and a ``norm`` equal to the drift from the
        previous read (0.0 on the first). The summary carries ``floor`` (the
        median consecutive ``roi_l2``) and ``series``.

    Raises:
        ValueError: If ``n`` is less than 2.
    """
    count = int(n)
    if count < 2:
        raise ValueError(f"n must be >= 2, got {n!r}")
    frames = [np.asarray(capture(), dtype=np.float64) for _ in range(count)]
    floor, series = flat_to_flat_floor(frames)
    rows = [
        {
            "stage": STAGE_REPEAT,
            "peak": _frame_metrics(frames[i])["peak"],
            "sum": _frame_metrics(frames[i])["sum"],
            "fwhm_px": _frame_metrics(frames[i])["fwhm_px"],
            "norm": float(series[i - 1]) if i > 0 else 0.0,
            "index": int(i),
            "_epoch": int(i),
        }
        for i in range(count)
    ]
    summary: dict[str, Any] = {
        "stage": STAGE_REPEAT,
        "floor": float(floor),
        "series": [float(v) for v in series],
        "n": count,
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Stage 2 -- settle curve
# ---------------------------------------------------------------------------


def settle_curve(
    capture: Capture,
    total_s: float,
    sample_ms: float,
    *,
    sleep: Callable[[float], None] | None = None,
    frac: float = DEFAULT_SETTLE_FRAC,
    run: int = DEFAULT_SETTLE_RUN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample one observable every ``sample_ms`` until it plateaus.

    The bench must already be showing the phase being settled (the caller
    displays it first -- the panel retains the last displayed pattern, so a read
    taken before the write is really the previous experiment).

    This is the **only** place in the module allowed to sleep: every other
    measurement path is either instantaneous or delegates settling to
    ``capture_settled``. ``sleep`` is injectable so tests never wait.

    Sample count is deterministic: ``floor(total_s / step) + 1`` samples,
    including ``t = 0``, so the curve is reproducible frame-for-frame.

    Args:
        capture: Zero-arg callable returning one prepared ROI crop.
        total_s: Curve duration in seconds.
        sample_ms: Interval between samples in milliseconds.
        sleep: ``seconds -> None``; defaults to :func:`time.sleep`.
        frac: Settle tolerance as a fraction of the final value.
        run: Consecutive in-tolerance samples required to declare settled.

    Returns:
        ``(rows, summary)``. Rows carry ``stage="settle"``, ``t_s``, the four
        metrics and the plateau observable in ``norm``. The summary carries
        ``times_s``, ``norms``, ``settle_s`` (``None`` when it never settles),
        ``settled`` and ``n_samples``.

    Raises:
        ValueError: On a non-positive interval or a negative duration.
    """
    step = float(sample_ms) / 1000.0
    if step <= 0.0:
        raise ValueError(f"sample_ms must be positive, got {sample_ms!r}")
    total = float(total_s)
    if total < 0.0:
        raise ValueError(f"total_s must be >= 0, got {total_s!r}")

    n_samples = int(math.floor(total / step)) + 1
    sleeper = time.sleep if sleep is None else sleep

    times: list[float] = []
    frames: list[np.ndarray] = []
    for i in range(n_samples):
        if i:
            sleeper(step)
        times.append(i * step)
        frames.append(np.asarray(capture(), dtype=np.float64))

    norms = [_magnitude(frame) for frame in frames]
    settle_s = settle_time_s(norms, times, frac=float(frac), run=int(run))
    rows = [
        {
            "stage": STAGE_SETTLE,
            "peak": _frame_metrics(frames[i])["peak"],
            "sum": _frame_metrics(frames[i])["sum"],
            "fwhm_px": _frame_metrics(frames[i])["fwhm_px"],
            "norm": float(norms[i]),
            "t_s": float(times[i]),
            "index": int(i),
            "_epoch": int(i),
        }
        for i in range(n_samples)
    ]
    summary: dict[str, Any] = {
        "stage": STAGE_SETTLE,
        "times_s": [float(v) for v in times],
        "norms": [float(v) for v in norms],
        "settle_s": None if settle_s is None else float(settle_s),
        "settled": settle_s is not None,
        "n_samples": n_samples,
        "frac": float(frac),
        "run": int(run),
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Stage 3 -- SNR ladder
# ---------------------------------------------------------------------------


def snr_ladder(
    capture: Capture,
    perturb: Callable[[], None],
    floor: float,
    ks: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """SNR of a fixed perturbation against ``floor``, as a function of K.

    The bench must be showing the **flat** state when this is called: the first
    ``capture()`` is therefore read as the flat reference, and every subsequent
    observable is an ``roi_l2`` distance *from that reference*. Keeping the
    observable and the floor in the same units (both L2 distances over the same
    ROI) is what makes ``snr`` dimensionless and actually comparable -- a bounded
    metric like PIB would make it depend on that metric's arbitrary scale too.

    The verdict is the point of the ladder. Averaging K frames shrinks
    independent read noise by ``sqrt(K)``, so ``snr`` grows and the residual is
    noise -- a longer exposure or a brighter beam would help. If ``snr`` is flat
    or falling, the residual is drift, and averaging longer only burns exposure.
    The bench needs a better settle protocol instead.

    Args:
        capture: Zero-arg callable returning one prepared ROI crop.
        perturb: ``() -> None``; applies the perturbation to the panel. Called
            once, after the flat reference has been read.
        floor: Single-frame drift floor from :func:`repeatability_floor`, in the
            same units as the observable. Clamped to ``SIGMA_FLOOR`` so a
            perfectly noiseless synthetic bench stays defined.
        ks: Averaging factors to report.

    Returns:
        ``(rows, summary)``. Rows carry ``stage="snr"``, ``k``, ``snr``, the
        four metrics and the K-frame mean observable in ``norm``. The summary
        carries ``snr`` (``{K: snr}``), ``norms``, ``floor``, ``improves`` and
        a ``verdict`` of ``"noise_limited"``/``"drift_limited"``.

    Raises:
        ValueError: On an empty ladder or a K below 1.
    """
    factors = sorted({int(k) for k in ks})
    if not factors:
        raise ValueError("ks must not be empty")
    if factors[0] < 1:
        raise ValueError(f"K must be >= 1, got {ks!r}")

    guard = max(float(floor), SIGMA_FLOOR)
    flat_ref = np.asarray(capture(), dtype=np.float64)

    perturb()

    k_max = factors[-1]
    frames = [np.asarray(capture(), dtype=np.float64) for _ in range(k_max)]
    norms = [roi_l2(frame, flat_ref) for frame in frames]
    snr = snr_vs_averages(norms, guard, factors)
    improves = bool(snr[factors[-1]] > snr[factors[0]])

    rows = []
    for position, k in enumerate(factors):
        rows.append(
            {
                "stage": STAGE_SNR,
                "peak": _frame_metrics(frames[k - 1])["peak"],
                "sum": _frame_metrics(frames[k - 1])["sum"],
                "fwhm_px": _frame_metrics(frames[k - 1])["fwhm_px"],
                "norm": float(np.mean(norms[:k])),
                "snr": float(snr[k]),
                "k": int(k),
                "index": int(position),
                "_epoch": int(position),
            }
        )
    summary: dict[str, Any] = {
        "stage": STAGE_SNR,
        "snr": {int(k): float(v) for k, v in snr.items()},
        "norms": [float(v) for v in norms],
        "floor": float(guard),
        "ks": [int(k) for k in factors],
        "improves": improves,
        "verdict": "noise_limited" if improves else "drift_limited",
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Perturbation pattern
# ---------------------------------------------------------------------------


def checkerboard_coeffs(grid: int, amplitude_rad: float = 1.0) -> np.ndarray:
    """Deterministic ``+-amplitude`` checkerboard, ``grid*grid`` values.

    A checkerboard rather than a uniform sign because a spatially dense
    pattern scatters light out of the focal plane more strongly, which is what
    makes the settle and SNR responses clear the drift floor. No RNG: the
    pattern is a function of ``grid`` alone, so a run is reproducible without
    pinning a seed (see ``AGENTS.md`` -- a seed fixes the *pattern*, never the
    measurements).

    Args:
        grid: Grid edge length.
        amplitude_rad: Coefficient amplitude in radians.

    Returns:
        Row-major coefficient vector of length ``grid*grid``.

    Raises:
        ValueError: If ``grid`` is not positive.
    """
    g = int(grid)
    if g <= 0:
        raise ValueError(f"grid must be positive, got {grid!r}")
    rows, cols = np.divmod(np.arange(g * g), g)
    signs = np.where((rows + cols) % 2 == 0, 1.0, -1.0)
    return float(amplitude_rad) * signs


def perturbation_pattern(
    grid: int, amplitude_rad: float, panel_shape: tuple[int, int]
) -> np.ndarray:
    """Checkerboard block phase on the panel, as raw unwrapped radians."""
    unit = checkerboard_coeffs(grid, 1.0)
    return float(amplitude_rad) * build_block_pattern(unit, int(grid), panel_shape)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_floor_records_pkl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write ``{epoch: row}`` as a checkpoint pickle; parents are created.

    Keyed by ``_epoch`` rather than a bare list so a later append can be merged
    into an existing checkpoint without renumbering, which is what makes writing
    after *every* sample safe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {int(row["_epoch"]): row for row in rows}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def save_floor_summary_npz(path: Path, summary: dict[str, Any]) -> Path:
    """Write the summary dict to a compressed ``.npz``; parents are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in summary.items()})
    logger.info("summary -> {}", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class SlmFloorProbeParams:
    """SLM 台架本底探针的 CLI 参数。

    三个阶段的默认值全部来自 ``docs/slm/pre_run_characterization.md``
    (``--roi`` 192 px、``--exposure-ms`` 1.5 ms、``--n-repeat`` 20、
    ``--settle-curve-s`` 4.0 s、``--settle-sample-ms`` 120 ms、``--ks``
    ``1,4,9``) —— 改任何一个都要同步改那份文档, 否则文档就在撒谎。

    ``--ks`` 保持**原始逗号分隔字符串**, 不在 CLI 层解析成 ``list[int]``:
    click 的 ``multiple`` 语义是重复传同一个 flag (``--ks 1 --ks 4``), 与
    argparse 时代一次传 ``"1,4"`` 的写法不兼容; 而 :func:`_parse_ints` 是被
    测试直接覆盖的纯函数, 所以解析留在 body 里由 :func:`_plan_lines` 和
    :func:`main` 各调一次, 与迁移前完全一致。
    """

    out: Annotated[str, option("--out", help="输出目录")] = "data/slm_floor"
    slm_number: Annotated[int, option("--slm-number", help="SLM 设备编号 (默认 1)")] = 1
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
        float, option("--exposure-ms", help="相机曝光 ms (默认 1.5)")
    ] = DEFAULT_EXPOSURE_MS
    n_repeat: Annotated[
        int, option("--n-repeat", help="重复性本底用的连续平场读帧数")
    ] = DEFAULT_N_REPEAT
    settle_curve_s: Annotated[
        float, option("--settle-curve-s", help="稳定曲线总时长 s")
    ] = DEFAULT_SETTLE_CURVE_S
    settle_sample_ms: Annotated[
        float, option("--settle-sample-ms", help="稳定曲线采样间隔 ms")
    ] = DEFAULT_SETTLE_SAMPLE_MS
    settle_frac: Annotated[
        float, option("--settle-frac", help="判稳容差 = frac * 末值")
    ] = DEFAULT_SETTLE_FRAC
    settle_run: Annotated[
        int, option("--settle-run", help="判稳所需的连续在容差内采样数")
    ] = DEFAULT_SETTLE_RUN
    ks: Annotated[
        str, option("--ks", help="SNR 梯度的平均帧数 (逗号分隔)")
    ] = ",".join(str(k) for k in DEFAULT_KS)
    perturbation_rad: Annotated[
        float, option("--perturbation-rad", help="扰动幅度 rad (默认 0.6)")
    ] = DEFAULT_PERTURBATION_RAD
    roi: Annotated[
        int, option("--roi", help="分析 ROI 全宽 px (取半宽送入 crop_roi)")
    ] = DEFAULT_ROI
    cell_grid: Annotated[
        int, option("--cell-grid", help="扰动块系数网格边长 (cell-grid^2 DOF)")
    ] = DEFAULT_GRID
    frames: Annotated[
        int, option("--frames", help="每帧平均张数 (默认 4)")
    ] = DEFAULT_FRAMES
    discard: Annotated[
        int, option("--discard", help="丢弃前 N 帧等稳定 (默认 3)")
    ] = DEFAULT_DISCARD
    settle_s: Annotated[
        float,
        option("--settle-s", help="单次读帧的稳定等待 s (交给 capture_settled)"),
    ] = DEFAULT_CAPTURE_SETTLE_S
    stable_tol: Annotated[
        float, option("--stable-tol", help="连续两帧读数的相对容差 (默认 0.02)")
    ] = DEFAULT_STABLE_TOL
    max_wait_s: Annotated[
        float, option("--max-wait-s", help="单次读帧稳定等待上限 s (默认 6.0)")
    ] = DEFAULT_MAX_WAIT_S
    no_hw: Annotated[
        bool,
        option("--no-hw", is_flag=True, help="不打开硬件, 只打印采集计划 (自检用)"),
    ] = False


def _parse_ints(text: str) -> list[int]:
    return [int(v) for v in str(text).split(",") if v.strip()]


def _plan_lines(
    params: SlmFloorProbeParams, shape: tuple[int, int]
) -> list[str]:
    ks = _parse_ints(params.ks)
    step = float(params.settle_sample_ms) / 1000.0
    n_settle = int(math.floor(float(params.settle_curve_s) / step)) + 1
    return [
        f"panel {shape}  roi {params.roi}px (half {params.roi // 2})",
        f"exposure {params.exposure_ms} ms  cam {params.cam_type}#{params.cam_id}  "
        f"slm #{params.slm_number} @ {params.slm_wavelength} nm",
        f"stage repeat: {params.n_repeat} consecutive flat reads",
        f"stage settle: {n_settle} samples over {params.settle_curve_s}s "
        f"every {params.settle_sample_ms}ms (frac {params.settle_frac}, "
        f"run {params.settle_run})",
        f"stage snr: perturbation {params.perturbation_rad} rad  "
        f"grid {params.cell_grid} ({params.cell_grid * params.cell_grid} DOF)  "
        f"ks {ks}",
        f"capture: frames {params.frames} discard {params.discard} "
        f"settle {params.settle_s}s tol {params.stable_tol} "
        f"max-wait {params.max_wait_s}s",
        f"outputs: {Path(params.out) / 'floor_records.pkl'}, "
        f"{Path(params.out) / 'floor_summary.npz'}",
    ]


@click.command()
@with_params(SlmFloorProbeParams, kw_name="params")
def main(params: SlmFloorProbeParams) -> None:
    """Run the floor probe on hardware and persist it.

    ``--no-hw`` prints the acquisition plan and returns without importing a driver.
    """
    from ao_shaping.utils.io.cli_helpers import setup_coredumpy

    setup_coredumpy()

    # Deferred so the self-check is safe on a machine with no SLM and no camera
    # SDK installed at all.
    from ao_shaping.tools.slm.slm_zernike_sweep_probe import (
        DEFAULT_PANEL_SHAPE,
        capture_settled,
    )
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    shape = DEFAULT_PANEL_SHAPE
    if params.no_hw:
        logger.info("--no-hw: SLM 本底探针采集计划, 不打开硬件")
        for line in _plan_lines(params, shape):
            logger.info("  {}", line)
        return

    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec

    ks = _parse_ints(params.ks)
    out_dir = Path(params.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_phase = np.zeros(shape, dtype=np.float64)
    current: dict[str, np.ndarray] = {"phase": flat_phase}
    pattern = perturbation_pattern(
        int(params.cell_grid), float(params.perturbation_rad), shape
    )
    rows: list[dict[str, Any]] = []
    records_path = out_dir / "floor_records.pkl"

    def checkpoint(row: dict[str, Any]) -> None:
        rows.append(row)
        save_floor_records_pkl(records_path, rows)

    with Santec(
        slm_number=params.slm_number, wavelength=params.slm_wavelength, video_mode=0
    ) as slm, create_camera(
        params.cam_type, params.cam_id, exposure_time_ms=params.exposure_ms
    ) as cam:

        def display_phase(phase: np.ndarray) -> None:
            # No ``memory_number``: re-displaying the slot already on the panel
            # is a firmware no-op and the LCOS never refreshes. ``display_data``
            # rotates all 127 slots itself.
            current["phase"] = np.asarray(phase, dtype=np.float64)
            slm.display_data(phase_to_slm_grayscale(current["phase"], slm=slm))

        def capture() -> np.ndarray:
            # ``capture_settled`` re-shows the phase currently held by
            # ``current`` and returns an averaged frame once two consecutive
            # readings agree -- the discipline a single-shot read would skip.
            return capture_settled(
                cam, slm, current["phase"],
                n_frames=params.frames, n_discard=params.discard,
                wait_time_s=params.settle_s, stable_tol=params.stable_tol,
                max_wait_s=params.max_wait_s,
            )

        cam.reset_exposure_time(float(params.exposure_ms))

        # Anchor the ROI once on the flat state, then freeze it. Re-locating by
        # argmax per sample would roll the box between speckle grains under a
        # ~1e-3 perturbation and make every observable discontinuous.
        display_phase(flat_phase)
        probe = np.asarray(capture(), dtype=np.float64)
        spot = measure_spot(probe)
        center = (int(round(spot.centroid_x)), int(round(spot.centroid_y)))
        half = int(params.roi) // 2
        logger.info(
            "ROI 中心 {} half={} ({})", center, half, spot.as_row(),
        )

        def roi_capture() -> np.ndarray:
            return prepare_roi_frame(capture(), center, half)

        # Stage 1 -- drift floor, on an untouched flat bench.
        display_phase(flat_phase)
        repeat_rows, repeat_summary = repeatability_floor(
            roi_capture, params.n_repeat
        )
        for row in repeat_rows:
            checkpoint(row)
        floor = float(repeat_summary["floor"])
        logger.info(
            "本底 {:.4g} ({} 次连续差)", floor, len(repeat_summary["series"])
        )

        # Stage 2 -- settle after a real phase write.
        display_phase(pattern)
        settle_rows, settle_summary = settle_curve(
            roi_capture,
            float(params.settle_curve_s),
            float(params.settle_sample_ms),
            frac=float(params.settle_frac),
            run=int(params.settle_run),
        )
        for row in settle_rows:
            checkpoint(row)
        logger.info(
            "稳定时间 {} ({} 次采样)", settle_summary["settle_s"],
            settle_summary["n_samples"],
        )

        # Stage 3 -- SNR ladder; the bench must be flat so the first read is the
        # flat reference the observable is measured against.
        display_phase(flat_phase)
        snr_rows, snr_summary = snr_ladder(
            roi_capture, lambda: display_phase(pattern), floor, ks
        )
        for row in snr_rows:
            checkpoint(row)

        display_phase(flat_phase)  # leave the bench flat, not on a perturbation

    summary: dict[str, Any] = {
        "stage_repeat_floor": floor,
        "repeat_series": np.asarray(repeat_summary["series"], dtype=np.float64),
        "n_repeat": int(params.n_repeat),
        "settle_times_s": np.asarray(settle_summary["times_s"], dtype=np.float64),
        "settle_norms": np.asarray(settle_summary["norms"], dtype=np.float64),
        "settle_s": (
            float("nan")
            if settle_summary["settle_s"] is None
            else float(settle_summary["settle_s"])
        ),
        "settled": bool(settle_summary["settled"]),
        "snr_ks": np.asarray(snr_summary["ks"], dtype=np.int64),
        "snr_values": np.asarray(
            [snr_summary["snr"][k] for k in snr_summary["ks"]], dtype=np.float64
        ),
        "snr_norms": np.asarray(snr_summary["norms"], dtype=np.float64),
        "snr_improves": bool(snr_summary["improves"]),
        "verdict": str(snr_summary["verdict"]),
        "exposure_ms": float(params.exposure_ms),
        "perturbation_rad": float(params.perturbation_rad),
        "cell_grid": int(params.cell_grid),
        "n_dof": int(params.cell_grid) * int(params.cell_grid),
        "panel_shape": np.asarray(shape, dtype=np.int64),
        "roi_center": np.asarray(center, dtype=np.int64),
        "roi_half": int(half),
    }

    save_floor_records_pkl(records_path, rows)
    summary_path = save_floor_summary_npz(out_dir / "floor_summary.npz", summary)

    logger.info(
        "判定 {} (SNR K={} -> K={}, 本底 {:.4g}; 稳定 {} s)",
        summary["verdict"], ks[0], ks[-1], floor, summary["settle_s"],
    )
    if not summary["snr_improves"]:
        logger.warning(
            "平均帧数没有提升 SNR: 残余不是独立读出噪声而是漂移 —— "
            "继续加 K 只浪费曝光, 应先改稳定协议"
        )
    logger.info("本底探针完成 -> {} / {}", records_path, summary_path)


if __name__ == "__main__":  # pragma: no cover
    main()