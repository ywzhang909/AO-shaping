"""ABBA dense-random-phase detectability probe for the SLM bench.

Question this probe answers: **can a seeded dense random-phase perturbation be
measured above the bench's own drift floor?** That is the precondition for any
model-in-the-loop speckle route -- if ``|mean(+) - mean(-)|`` for a random
block phase cannot clear the flat-to-flat drift, then no amount of averaging
will make the sign of a gradient trustworthy.

Method, reusing the existing kernels rather than re-deriving them:

* exposure is bracketed up a ladder and the **brightest unsaturated** rung kept
  (:func:`rebracket_exposure`);
* the drift floor is the median consecutive flat-to-flat ``roi_l2``
  (:func:`drift_floor_at_exposure`, delegating to ``flat_to_flat_floor``);
* each dense pattern is measured with the drift-cancelling ``+ - - +``
  palindrome (:func:`~ao_shaping.tools.slm.slm_snr_probe.abba_signal`);
* ``snr = response / floor`` is graded against the thresholds that
  ``slm_snr_probe`` already uses (:data:`SNR_USABLE`, :data:`SNR_STRONG`).

The objective scalar is deliberately the **L2 distance from the flat
reference**, not a normalised image-quality metric, because the drift floor is
an L2 distance too -- the quotient is then dimensionless and the two numbers
are actually comparable. Mixing in a bounded metric (PIB, CV) would make
``snr`` depend on the metric's arbitrary scale as well as on the bench.

Units discipline (see ``AGENTS.md``): every phase here is **raw unwrapped
radians** and is converted to grayscale exactly once, by
``phase_to_slm_grayscale``. No generator in this file wraps mod 2*pi.

Run it with ``python -m ao_shaping.tools.slm.slm_abba_probe``;
``--no-hw`` prints the acquisition plan and returns without touching a device.
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.tools.slm.slm_bench_metrics import (
    build_block_pattern,
    crop_roi,
    finite_clip,
    flat_to_flat_floor,
    roi_l2,
)
from ao_shaping.tools.slm.slm_snr_probe import (
    SNR_STRONG,
    SNR_USABLE,
    SIGMA_FLOOR,
    abba_signal,
    snr_verdict,
)
from ao_shaping.tools.slm.slm_zernike_sweep_probe import (
    DEFAULT_PANEL_SHAPE,
    capture_settled,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

#: RNG seed for the dense coefficient maps. Pinned so a run is reproducible up
#: to device noise (see ``AGENTS.md``: seed fixes the *pattern*, never the
#: measurements, so assert approximate reproducibility, not equality).
DEFAULT_SEED = 20261001

#: Perturbation amplitude in radians applied to the +/-1 dense pattern.
DEFAULT_DELTA_RAD = 0.5

#: Edge length of the block-coefficient grid (24x24 = 576 DOF, the same freeform
#: basis the shaping optimizers use).
DEFAULT_GRID = 24

DEFAULT_N_PATTERNS = 12
DEFAULT_PAIRS = 3

#: Full width of the analysis ROI in pixels; the half-width handed to
#: :func:`crop_roi` is half of this.
DEFAULT_ROI = 192

DEFAULT_EXPOSURE_LADDER: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.25, 1.5)

#: Detector level above which a frame counts as clipped. Reported, never used
#: to pick an exposure -- that is :data:`DEFAULT_SATURATION_MAX_PEAK`'s job.
DEFAULT_SATURATION_LEVEL = 250.0

#: Ceiling for "bright enough but not clipped" when choosing an exposure rung.
DEFAULT_SATURATION_MAX_PEAK = 245.0

#: Consecutive flat reads used for the drift floor.
DEFAULT_DRIFT_FRAMES = 8


def prepare_roi_frame(
    frame: np.ndarray, center: tuple[int, int], half: int
) -> np.ndarray:
    """Detach the readout pedestal, then crop the frozen ROI.

    Two steps in this order, both from ``slm_bench_metrics``:

    1. :func:`finite_clip` removes the symmetric read noise by subtracting the
       frame median and clipping at zero. A raw CCD frame is roughly half
       negative; leaving that in makes every downstream norm meaningless.
    2. :func:`crop_roi` cuts a ``2*half`` square about the **measured** centre.
       The 0-order is the frame global maximum on this bench and is routinely
       nowhere near the geometric centre, so the centre must be passed in.

    Args:
        frame: Raw 2-D camera frame.
        center: ROI centre as ``(x, y)`` pixels, frozen for the whole run.
        half: ROI half-width in pixels.

    Returns:
        A non-negative ``(2*half, 2*half)`` float64 array.
    """
    return crop_roi(finite_clip(frame), center, half)


def rebracket_exposure(
    capture: Callable[[], np.ndarray],
    set_exposure: Callable[[float], None],
    ladder: Sequence[float],
    *,
    saturation_level: float = 245.0,
) -> tuple[float, dict[str, Any]]:
    """Walk ``ladder`` and keep the brightest unsaturated rung.

    Peak brightness is what actually limits the measurement, so the rule is
    "as bright as possible without clipping": among the rungs whose peak stays
    at or below ``saturation_level``, take the one with the largest peak.

    Args:
        capture: Zero-arg callable returning one frame at the current exposure.
        set_exposure: ``exposure_ms -> None``; applies the exposure.
        ladder: Candidate exposures in ms, tried in order.
        saturation_level: Peak ceiling; a rung above this is treated as clipped.

    Returns:
        ``(chosen_exposure_ms, info)`` where ``info`` holds every rung's peak
        and its clipped flag, the chosen index, and ``all_saturated``.

    Raises:
        ValueError: If ``ladder`` is empty.
    """
    rungs = [float(v) for v in ladder]
    if not rungs:
        raise ValueError("exposure ladder must not be empty")

    peaks: list[float] = []
    for exposure_ms in rungs:
        set_exposure(exposure_ms)
        frame = np.asarray(capture(), dtype=np.float64)
        peaks.append(float(frame.max()) if frame.size else 0.0)

    usable = [i for i, peak in enumerate(peaks) if peak <= saturation_level]
    all_saturated = not usable
    if all_saturated:
        # Every rung clips: fall back to the dimmest peak, which is the closest
        # thing to a linear read. The caller is told, because an SNR measured on
        # a clipped frame is not a bench characterisation.
        chosen_index = int(np.argmin(peaks))
        logger.warning(
            "曝光梯度全部超过饱和上限 {} (峰值 {}), 退回最暗档 {} ms",
            saturation_level,
            [round(p, 1) for p in peaks],
            rungs[chosen_index],
        )
    else:
        chosen_index = int(max(usable, key=lambda i: peaks[i]))

    info: dict[str, Any] = {
        "ladder": rungs,
        "peaks": peaks,
        "clipped": [bool(p > saturation_level) for p in peaks],
        "chosen_index": chosen_index,
        "chosen_peak": peaks[chosen_index],
        "all_saturated": all_saturated,
        "saturation_level": float(saturation_level),
    }
    return rungs[chosen_index], info


def drift_floor_at_exposure(
    capture: Callable[[], np.ndarray], n: int
) -> tuple[float, list[float]]:
    """Drift floor from ``n`` consecutive reads at one fixed exposure.

    The caller must leave the bench untouched between calls (a flat phase
    already written). This is what a single ABBA step actually competes
    against: shot noise *and* slow intensity drift, not shot noise alone.

    Delegates to ``flat_to_flat_floor``, which takes the median of the
    consecutive ``roi_l2`` differences.

    Args:
        capture: Zero-arg callable returning one prepared frame.
        n: Number of consecutive reads; must be at least 2.

    Returns:
        ``(median, series)`` -- the median drift and every consecutive
        difference.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    frames = [np.asarray(capture(), dtype=np.float64) for _ in range(n)]
    return flat_to_flat_floor(frames)


def dense_abba_sweep(
    capture: Callable[[], np.ndarray],
    display_pattern: Callable[[np.ndarray], None],
    *,
    n_patterns: int,
    pairs: int,
    delta_rad: float,
    grid: int,
    panel_shape: tuple[int, int],
    floor: float,
    seed: int = DEFAULT_SEED,
    exposure_ms: float = float("nan"),
    on_pattern: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure the ABBA response of ``n_patterns`` seeded dense block patterns.

    For every pattern the +/-1 coefficient grid is drawn from a seeded RNG,
    tiled onto the panel by :func:`build_block_pattern` (raw radians), and then
    driven at +/- ``delta_rad`` through the ``+ - - +`` palindrome. The score is
    the L2 distance from the flat reference captured before the loop, so
    ``response`` and ``floor`` share units and ``snr`` is dimensionless.

    The three optional keyword arguments are provenance and progress hooks.
    They default to values that keep a bare call correct and side-effect free:
    ``seed`` pins the coefficient maps, ``exposure_ms`` is recorded on every row
    (the pure sweep cannot know it), and ``on_pattern`` fires once per finished
    pattern so a caller can checkpoint incrementally.

    Args:
        capture: Zero-arg callable returning one **prepared** ROI frame (see
            :func:`prepare_roi_frame`). It must not display anything itself
            beyond re-showing the phase it is asked to measure.
        display_pattern: ``phase_rad -> None``; writes the phase to the panel.
        n_patterns: Number of independent dense patterns.
        pairs: ABBA repetitions per pattern.
        delta_rad: Perturbation amplitude in radians.
        grid: Edge length of the block-coefficient grid.
        panel_shape: ``(height, width)`` of the panel raster.
        floor: Drift floor from :func:`drift_floor_at_exposure`.
        seed: RNG seed for the coefficient maps.
        exposure_ms: Exposure stamped on each row.
        on_pattern: Optional ``row -> None`` observer, called per pattern.

    Returns:
        ``(rows, summary)``. Each row carries ``peak``, ``sum``, ``norm``,
        ``response``, ``floor``, ``snr``, ``pattern``, ``pair``,
        ``exposure_ms``, ``_img`` and ``_epoch``. The summary carries the drift
        floor, the median/min response, how many patterns cleared
        :data:`SNR_USABLE`, the fraction, and a ``usable``/``unusable``
        verdict.

    Raises:
        ValueError: On a non-positive amplitude, an empty pattern list, or too
            few pairs.
    """
    if n_patterns < 1:
        raise ValueError(f"n_patterns must be >= 1, got {n_patterns}")
    if pairs < 1:
        raise ValueError(f"pairs must be >= 1, got {pairs}")
    if delta_rad <= 0.0:
        raise ValueError(f"delta_rad must be positive, got {delta_rad}")

    shape = (int(panel_shape[0]), int(panel_shape[1]))
    flat_phase = np.zeros(shape, dtype=np.float64)

    # Anchor the objective on the flat state, on the panel state the writer
    # actually leaves behind. Writing first matters: the panel retains the
    # previous pattern, so a "flat" read taken before any write is really the
    # last frame of the last experiment.
    display_pattern(flat_phase)
    flat_ref = np.asarray(capture(), dtype=np.float64)

    denominator = max(float(floor), SIGMA_FLOOR)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for index in range(n_patterns):
        coefficients = rng.uniform(-1.0, 1.0, size=int(grid) * int(grid))
        base = build_block_pattern(coefficients, grid, shape)

        # Last score call of a `+ - - +` palindrome is the trailing `+`, so this
        # ends up holding a positively-perturbed frame -- deterministic, and the
        # representative one for the pattern's statistics.
        latest: dict[str, Any] = {"img": None, "peak": 0.0, "sum": 0.0, "norm": 0.0}

        def score() -> float:
            frame = np.asarray(capture(), dtype=np.float64)
            latest["img"] = frame
            latest["peak"] = float(frame.max())
            latest["sum"] = float(frame.sum())
            latest["norm"] = roi_l2(frame, flat_ref)
            return float(latest["norm"])

        response = abba_signal(
            score,
            lambda amplitude, _base=base: display_pattern(amplitude * _base),
            lambda: display_pattern(flat_phase),
            float(delta_rad),
            pairs=int(pairs),
        )

        rows.append(
            {
                "peak": float(latest["peak"]),
                "sum": float(latest["sum"]),
                "norm": float(latest["norm"]),
                "response": float(response),
                "floor": float(floor),
                "snr": float(response) / denominator,
                "pattern": int(index),
                "pair": int(pairs),
                "exposure_ms": float(exposure_ms),
                "_img": latest["img"],
                "_epoch": int(index),
            }
        )
        if on_pattern is not None:
            on_pattern(rows[-1])

    display_pattern(flat_phase)  # leave the bench flat, not on a perturbation

    snrs = [float(row["snr"]) for row in rows]
    responses = [float(row["response"]) for row in rows]
    median_snr = float(np.median(snrs))
    snr_class = snr_verdict(median_snr)
    n_exceeding = int(sum(1 for value in snrs if value >= SNR_USABLE))

    summary: dict[str, Any] = {
        "drift_floor": float(floor),
        "median_response": float(np.median(responses)),
        "min_response": float(np.min(responses)),
        "median_snr": median_snr,
        "min_snr": float(np.min(snrs)),
        "n_patterns": int(n_patterns),
        "pairs": int(pairs),
        "n_exceeding": n_exceeding,
        "exceeding_fraction": float(n_exceeding) / float(len(rows)),
        "verdict": "usable" if snr_class in ("strong", "usable") else "unusable",
        "snr_class": snr_class,
        "snr_strong": float(SNR_STRONG),
        "snr_usable": float(SNR_USABLE),
        "delta_rad": float(delta_rad),
        "grid": int(grid),
        "n_dof": int(grid) * int(grid),
        "panel_shape": np.asarray(shape, dtype=np.int64),
        "exposure_ms": float(exposure_ms),
        "seed": int(seed),
    }
    return rows, summary


def save_abba_summary_npz(path: Path, summary: dict[str, Any]) -> Path:
    """Write the summary dict to a compressed ``.npz``; parents are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in summary.items()})
    logger.info("summary -> {}", path)
    return path


def save_abba_records_pkl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write ``{epoch: row}`` as a checkpoint pickle; parents are created.

    Keyed by ``_epoch`` rather than a bare list so a later append can be merged
    into an existing checkpoint without renumbering, which is what makes
    writing after *every* pattern safe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {int(row["_epoch"]): row for row in rows}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in str(text).split(",") if v.strip()]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from ao_shaping.utils.io.cli_helpers import setup_coredumpy

    setup_coredumpy()
    ap = argparse.ArgumentParser(
        description=(
            "ABBA 密集随机相位可探测性探针 "
            "(需硬件: Santec SLM-200 + 远场相机)"
        ),
    )
    ap.add_argument("--out", default="data/slm_abba", help="输出目录")
    ap.add_argument("--slm-number", type=int, default=1)
    ap.add_argument("--slm-wavelength", type=int, default=1064)
    ap.add_argument("--cam-type", default="daheng", choices=["daheng", "miicam"])
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--exposure-ms", type=float, default=1.0,
                    help="初始曝光; --rebracket-exposure 时仅作起点参考")
    ap.add_argument("--rebracket-exposure", action="store_true",
                    help="沿曝光梯度选最亮未饱和档")
    ap.add_argument("--exposure-ladder", default="0.4,0.6,0.8,1.0,1.25,1.5")
    ap.add_argument("--saturation-level", type=float, default=250.0,
                    help="判定为截断的峰值灰度 (仅报告)")
    ap.add_argument("--saturation-max-peak", type=float, default=245.0,
                    help="选曝光档时的峰值上限")
    ap.add_argument("--delta-rad", type=float, default=DEFAULT_DELTA_RAD)
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID)
    ap.add_argument("--n-patterns", type=int, default=DEFAULT_N_PATTERNS)
    ap.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    ap.add_argument("--roi", type=int, default=DEFAULT_ROI,
                    help="分析 ROI 全宽 px (取半宽送入 crop_roi)")
    ap.add_argument("--drift-frames", type=int, default=DEFAULT_DRIFT_FRAMES,
                    help="漂移地板用的连续平场读帧数")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--discard", type=int, default=3)
    ap.add_argument("--settle-s", type=float, default=0.5)
    ap.add_argument("--stable-tol", type=float, default=0.02)
    ap.add_argument("--max-wait-s", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-hw", action="store_true",
                    help="不打开硬件, 只打印采集计划 (自检用)")
    return ap.parse_args(argv)


def _plan_lines(args: argparse.Namespace) -> list[str]:
    ladder = _parse_floats(args.exposure_ladder)
    return [
        f"panel {DEFAULT_PANEL_SHAPE}  roi {args.roi}px  "
        f"(half {args.roi // 2})",
        f"exposure: {'rebracket' if args.rebracket_exposure else 'fixed'} "
        f"start {args.exposure_ms} ms  ladder {ladder}  "
        f"max-peak {args.saturation_max_peak}  clip {args.saturation_level}",
        f"drift floor: {args.drift_frames} consecutive flat reads",
        f"sweep: {args.n_patterns} patterns x {args.pairs} ABBA pairs  "
        f"delta {args.delta_rad} rad  grid {args.grid} "
        f"({args.grid * args.grid} DOF)  seed {args.seed}",
        f"capture: frames {args.frames} discard {args.discard} "
        f"settle {args.settle_s}s tol {args.stable_tol} "
        f"max-wait {args.max_wait_s}s",
        f"outputs: {Path(args.out) / 'abba_records.pkl'}, "
        f"{Path(args.out) / 'abba_summary.npz'}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ABBA probe on hardware and persist it.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, including on the ``--no-hw`` self-check path.
    """
    args = _parse_args(argv)

    if args.no_hw:
        # Before any driver import: the self-check must be safe to run on a
        # machine with no SLM and no camera SDK installed at all.
        logger.info("--no-hw: ABBA 采集计划, 不打开硬件")
        for line in _plan_lines(args):
            logger.info("  {}", line)
        logger.info(
            "计划帧数 ≈ {} 次读帧 (曝光梯度 {} + 漂移地板 {} + 平场 {} + 扫描 {})",
            len(_parse_floats(args.exposure_ladder))
            + args.drift_frames
            + 1
            + args.n_patterns * args.pairs * 4,
            len(_parse_floats(args.exposure_ladder)),
            args.drift_frames,
            1,
            args.n_patterns * args.pairs * 4,
        )
        return 0

    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = DEFAULT_PANEL_SHAPE
    flat_phase = np.zeros(shape, dtype=np.float64)
    current: dict[str, np.ndarray] = {"phase": flat_phase}

    with Santec(
        slm_number=args.slm_number, wavelength=args.slm_wavelength, video_mode=0
    ) as slm, create_camera(
        args.cam_type, args.cam_id, exposure_time_ms=args.exposure_ms
    ) as cam:

        def display_pattern(phase: np.ndarray) -> None:
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
                n_frames=args.frames, n_discard=args.discard,
                wait_time_s=args.settle_s, stable_tol=args.stable_tol,
                max_wait_s=args.max_wait_s,
            )

        cam.reset_exposure_time(float(args.exposure_ms))
        if args.rebracket_exposure:
            exposure_ms, bracket_info = rebracket_exposure(
                capture,
                lambda ms: cam.reset_exposure_time(ms),
                _parse_floats(args.exposure_ladder),
                saturation_level=args.saturation_max_peak,
            )
        else:
            exposure_ms, bracket_info = float(args.exposure_ms), {}
        cam.reset_exposure_time(exposure_ms)
        logger.info(
            "曝光 {} ms (峰值 {})", exposure_ms,
            bracket_info.get("chosen_peak", float("nan")),
        )

        # Anchor the ROI once on the flat state, then freeze it. Re-locating by
        # argmax per iteration would roll the box between speckle grains under a
        # ~1e-3 perturbation and make the objective discontinuous.
        display_pattern(flat_phase)
        probe = np.asarray(capture(), dtype=np.float64)
        cx, cy = zero_order_center(probe)
        center = (int(cx), int(cy))
        half = int(args.roi) // 2
        logger.info("ROI 中心 {} half={}", center, half)

        def roi_capture() -> np.ndarray:
            return prepare_roi_frame(capture(), center, half)

        floor, drift_series = drift_floor_at_exposure(roi_capture, args.drift_frames)
        logger.info(
            "漂移地板 {:.4g} ({} 次连续差: [{}])",
            floor, len(drift_series),
            ", ".join(f"{v:.3g}" for v in drift_series),
        )

        records_path = out_dir / "abba_records.pkl"
        rows: list[dict[str, Any]] = []

        def checkpoint(row: dict[str, Any]) -> None:
            rows.append(row)
            save_abba_records_pkl(records_path, rows)
            logger.info(
                "  pattern {}/{}  response {:.4g}  snr {:.3g}  (已存 {})",
                row["pattern"] + 1, args.n_patterns, row["response"], row["snr"],
                records_path.name,
            )

        rows, summary = dense_abba_sweep(
            roi_capture,
            display_pattern,
            n_patterns=args.n_patterns,
            pairs=args.pairs,
            delta_rad=args.delta_rad,
            grid=args.grid,
            panel_shape=shape,
            floor=floor,
            seed=args.seed,
            exposure_ms=exposure_ms,
            on_pattern=checkpoint,
        )

    summary["drift_series"] = np.asarray(drift_series, dtype=np.float64)
    summary["bracket_peaks"] = np.asarray(
        bracket_info.get("peaks", []), dtype=np.float64
    )
    summary["bracket_ladder"] = np.asarray(
        bracket_info.get("ladder", []), dtype=np.float64
    )
    summary["all_saturated"] = bool(bracket_info.get("all_saturated", False))
    summary["roi_center"] = np.asarray(center, dtype=np.int64)
    summary["roi_half"] = int(half)
    summary["saturation_level"] = float(args.saturation_level)
    summary["saturation_max_peak"] = float(args.saturation_max_peak)

    save_abba_records_pkl(records_path, rows)
    summary_path = save_abba_summary_npz(out_dir / "abba_summary.npz", summary)

    logger.info(
        "判定 {} (median snr {:.3g}, 阈值 usable>={} strong>={}; "
        "{}/{} 图案达标)",
        summary["verdict"], summary["median_snr"], SNR_USABLE, SNR_STRONG,
        summary["n_exceeding"], summary["n_patterns"],
    )
    if summary["verdict"] == "unusable":
        logger.warning(
            "密集随机相位响应淹没在漂移里: 该台架当前 delta={} rad 不足以支撑"
            "model-in-the-loop 散斑标定 (可先加大 --delta-rad 或 --pairs)",
            args.delta_rad,
        )
    logger.info("ABBA 探针完成 -> {} / {}", records_path, summary_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())