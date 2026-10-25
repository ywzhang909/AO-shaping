"""Locate the beam on the SLM panel, in panel coordinates.

Why this is needed
------------------
Everything downstream -- the pupil phase, the injected defocus, the modelled
aperture -- has to be placed where the beam actually is. The camera's 0-order
position is **not** that place: on this bench the two axes are swapped 90
degrees and the scales differ by more than an order of magnitude. The runbook
used to derive a panel offset from the camera 0-order and computed ``(+426, -290)``
panel px, which rolled the pupil phase clean off the beam.

So measure it instead of deriving it: write a random-phase disc at candidate
panel positions and keep the one whose frame differs most from flat. Only a patch
that actually covers the beam can scatter the 0-order.

Metric
------
The change in energy inside a small box at the 0-order. Two alternatives were
tried and both failed:

* *total* frame energy is conserved -- measured 0.999-1.005 across aperture radii
  from 100 to 850 px -- so it carries no position or size information;
* the 0-order box energy is **not** monotone in patch size, because a disc both
  scatters light out of the box and redistributes light into it.

Usage
-----
::

    python -m ao_shaping.tools.slm.panel.panel_locate --exposure-ms 0.1
    python -m ao_shaping.tools.slm.panel.panel_locate --grid-xs 400,700,1000,1300 --grid-ys 300,600,900
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Annotated, Any

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.panel.display_state import (
    preserve_slm_display as _restore_slm_display,
)
from ao_shaping.utils.cli.exposure import (
    DEFAULT_EXPOSURE,
    exposure_ms_legacy,
    resolve_exposure,
)

from ao_shaping.tools.kernels.bench_kernels import (
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    display_and_average,
    measure_flat_reference,
)
from ao_shaping.utils.cli.params import option, with_params


def _box_mask(shape: tuple[int, int], cx: int, cy: int, r: int) -> np.ndarray:
    iy, ix = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (iy - cy) ** 2 + (ix - cx) ** 2 <= r * r


def _paired_delta(before: np.ndarray, candidate: np.ndarray, after: np.ndarray,
                  box: np.ndarray, max_drift: float) -> tuple[float, float]:
    energies = [float(frame[box].sum()) for frame in (before, candidate, after)]
    reference = (energies[0] + energies[2]) / 2
    if reference <= 0:
        raise ValueError("local flat reference has zero energy")
    drift = abs(energies[2] - energies[0]) / reference
    if drift > max_drift:
        raise ValueError(f"flat reference drift {drift:.1%} exceeds {max_drift:.1%}")
    return (energies[1] - reference) / reference, drift


@dataclass
class PanelLocateParams:
    """CLI surface of the panel-locate probe.

    Field order *is* the ``--help`` order, and the whole ten-option surface is
    frozen by ``tests/ao_shaping/runners/cli_help_golden.json``.

    Declared locally rather than spliced from :mod:`ao_shaping.tools.slm.params`:
    the shared ``SlmBenchParams`` types ``--cam-type`` as a ``click.Choice``
    (renders ``[daheng|miicam]``, not ``TEXT``) and leaves ``--exposure-ms`` /
    ``--frames`` without help text, so reuse would change the frozen help.

    ``exposure_ms`` is a token (``"auto"`` / ``"default"`` / a decimal string)
    rather than a float because the two non-numeric states are genuinely
    different here -- see :mod:`ao_shaping.utils.cli.exposure`, which resolves
    them.  It used to be a plain float with ``0`` meaning what ``auto`` now
    means.
    """

    slm_number: Annotated[
        int, option("--slm-number", help="SLM 设备编号 (默认 1)")
    ] = 1
    slm_wavelength: Annotated[
        int, option("--slm-wavelength", help="SLM 波长 nm (默认 1064)")
    ] = 1064
    cam_type: Annotated[
        str, option("--cam-type", help="相机类型 (daheng/miicam, 默认 daheng)")
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id", help="相机 ID (默认 0)")] = 0
    exposure_ms: Annotated[
        str,
        option(
            "--exposure-ms",
            metavar="[default|auto|FLOAT]",
            help=(
                "相机曝光 ms (默认 0.1)。'auto' = 探测曝光上限并在其下工作"
                "(env AO_DAHENG_MAX_EXPOSURE_MS 优先, 否则按画面倍增探测); "
                "'default' = 完全不动曝光 (不探测/不重置/不钳位), 延用相机当前值; "
                "0 不再等价于 auto, 会被拒绝"
            ),
        ),
    ] = "0.1"
    patch_radius: Annotated[
        int,
        option(
            "--patch-radius",
            help="随机相位圆盘半径 (面板 px, 默认 450=实测光斑半径)",
        ),
    ] = BEAM_RADIUS_PANEL
    grid_xs: Annotated[
        str,
        option("--grid-xs", help="候选 x (面板 px, 逗号分隔)"),
    ] = "240,600,960,1320,1680"
    grid_ys: Annotated[
        str,
        option("--grid-ys", help="候选 y (面板 px, 逗号分隔)"),
    ] = "220,480,720,980"
    frames: Annotated[int, option("--frames", help="每帧平均张数 (默认 4)")] = 4
    seed: Annotated[int, option("--seed", help="随机相位种子 (默认 7)")] = 7
    debug_dir: Annotated[
        str, option("--debug-dir", help="Save raw CCD frames and scan parameters.")
    ] = ""
    max_flat_drift: Annotated[
        float, option("--max-flat-drift", help="Maximum relative change between paired flats.")
    ] = 0.2


@dataclass(frozen=True)
class PanelLocateResult:
    center: tuple[int, int]
    response: float
    max_pair_drift: float
    session_drift: float
    rejected_count: int
    #: Session flat **centroid** shift in camera px -- the quantity that actually
    #: invalidates a locate. ``session_drift`` is an intensity ratio that the
    #: per-point pairing already cancels, so it is reported, not enforced.
    session_shift_px: float = 0.0


def _centroid(flat: Any) -> tuple[float, float]:
    return (float(flat.centroid_x), float(flat.centroid_y))


def scan_panel(
    params: PanelLocateParams, *, strict: bool = False,
    max_session_drift: float = 0.05,
    max_session_shift_px: float | None = None,
    max_session_shift_fwhm: float = 0.5,
) -> PanelLocateResult:
    """Scan candidate panel positions and verify paired and session flat stability."""
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.image.exposure_control import (
        DAHENG_EXPOSURE_LIMIT_ENV,
        configured_daheng_exposure_limit,
        probe_exposure_ceiling_ms,
        resolve_daheng_exposure_limit,
        working_exposure_ms,
    )

    # Local aliases keep the measurement body below verbatim.
    slm_number = params.slm_number
    slm_wavelength = params.slm_wavelength
    cam_type = params.cam_type
    cam_id = params.cam_id
    # The ONE explicit/implicit translation point for this probe. `auto` is the
    # legacy `0`: probe the ceiling and work under it. `default` is new and is
    # the only call site in the repo that can honour it -- touch nothing, reset
    # nothing, clamp nothing. Everything below `exposure_ms == 0` is
    # pre-existing behaviour, unchanged.
    resolved_exposure = resolve_exposure(params.exposure_ms, param_hint="--exposure-ms")
    keep_device_exposure = resolved_exposure is DEFAULT_EXPOSURE
    exposure_ms = exposure_ms_legacy(resolved_exposure)
    patch_radius = params.patch_radius
    grid_xs = params.grid_xs
    grid_ys = params.grid_ys
    frames = params.frames
    seed = params.seed
    max_flat_drift = params.max_flat_drift
    debug_dir = Path(params.debug_dir) if params.debug_dir else None

    if cam_type.lower() == "daheng":
        env_ceiling = configured_daheng_exposure_limit()
        if env_ceiling > 0 and exposure_ms > env_ceiling:
            raise click.BadParameter(
                f"Daheng exposure must be in (0, {env_ceiling}] ms "
                f"({DAHENG_EXPOSURE_LIMIT_ENV})",
                param_hint="--exposure-ms",
            )
    if not 0 <= max_flat_drift < 1:
        raise click.BadParameter("must be in [0, 1)", param_hint="--max-flat-drift")
    if not 0 <= max_session_drift < 1:
        raise ValueError("max_session_drift must be in [0, 1)")

    xs = [int(float(v)) for v in str(grid_xs).split(",") if v.strip()]
    ys = [int(float(v)) for v in str(grid_ys).split(",") if v.strip()]
    panel = (SLM_PANEL_H, SLM_PANEL_W)
    rng = np.random.default_rng(int(seed))

    with Santec(
        slm_number=slm_number, wavelength=slm_wavelength, video_mode=0
    ) as slm, create_camera(cam_type, cam_id, exposure_time_ms=0.02) as cam, _restore_slm_display(slm):
        if keep_device_exposure:
            # 'default': probe nothing, reset nothing, clamp nothing. The 0.02
            # the driver applied during open() is already in force; take it as
            # the working exposure and report the readback for the record only.
            # (On Daheng open() itself always writes a clamped value, so the
            # device's "own" setting here is its ~0.02 ms minimum -- which is
            # exactly why this is NOT the default for this probe.)
            actual_ms = float(get_camera_exposure_ms(cam))
            logger.info(
                "exposure 'default': keeping the camera's own {} ms (no probe, no reset)",
                actual_ms,
            )
        else:
            actual_ms = cam.reset_exposure_time(0.02)
            if exposure_ms == 0:
                measure_flat_reference(cam, slm, n_frames=1, panel_shape=panel)
            if cam_type.lower() == "daheng":
                ceiling = resolve_daheng_exposure_limit(camera=cam)
                # The probe (if any) left the camera at the last ladder step; re-arm.
                if exposure_ms == 0:
                    exposure_ms = working_exposure_ms(ceiling)
                actual_ms = cam.reset_exposure_time(float(exposure_ms))
                if actual_ms is None or not 0 < float(actual_ms) <= ceiling + 1e-9:
                    raise RuntimeError(
                        f"unsafe Daheng exposure readback: {actual_ms} ms "
                        f"(ceiling {ceiling} ms)"
                    )
            elif exposure_ms == 0:
                estimate = probe_exposure_ceiling_ms(
                    cam.reset_exposure_time,
                    lambda: np.asarray(cam.get_numpy_image(n_sample=2)),
                )
                exposure_ms = working_exposure_ms(estimate.ceiling_ms)
                actual_ms = cam.reset_exposure_time(float(exposure_ms))
            else:
                actual_ms = cam.reset_exposure_time(float(exposure_ms))
        ref, flat = measure_flat_reference(cam, slm, n_frames=frames, panel_shape=panel)
        box = _box_mask(ref.shape, int(round(flat.centroid_x)),
                        int(round(flat.centroid_y)), 60)
        base = float(ref[box].sum())
        logger.info(
            "flat reference: {}  0-order box energy={:.0f}", flat.as_row(), base
        )
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            np.save(debug_dir / "flat.npy", ref)
            (debug_dir / "config.json").write_text(
                json.dumps(
                    {
                        "slm_number": slm_number,
                        "slm_wavelength": slm_wavelength,
                        "cam_type": cam_type,
                        "cam_id": cam_id,
                        "exposure_ms": exposure_ms,
                        "exposure_ms_readback": float(actual_ms),
                        "patch_radius": patch_radius,
                        "grid_xs": xs,
                        "grid_ys": ys,
                        "frames": frames,
                        "seed": seed,
                        "max_flat_drift": max_flat_drift,
                        "flat_center_xy": [float(flat.centroid_x), float(flat.centroid_y)],
                        "flat_box_energy": base,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if np.max(ref) >= 250:
            raise ValueError(
                f"flat CCD frame peaks at {np.max(ref):.1f}/255; reduce exposure "
                "before using patch energy to locate the beam"
            )
        if base <= 0:
            raise SystemExit("0-order box energy is zero; cannot score patches")

        results: list[tuple[float, int, int, float]] = []
        measurements: list[dict] = []
        for py in ys:
            row: list[float] = []
            for px in xs:
                phase = np.zeros(panel, dtype=np.float64)
                yy, xx = np.mgrid[0 : panel[0], 0 : panel[1]]
                m = (yy - py) ** 2 + (xx - px) ** 2 <= int(patch_radius) ** 2
                phase[m] = rng.uniform(0.0, 2.0 * np.pi, int(m.sum()))
                before, _flat = measure_flat_reference(cam, slm, n_frames=frames, panel_shape=panel)
                img = display_and_average(cam, slm, phase, n_frames=frames)
                after, _ = measure_flat_reference(cam, slm, n_frames=frames, panel_shape=panel)
                if debug_dir is not None:
                    np.savez_compressed(debug_dir / f"frame_x{px}_y{py}.npz",
                                        flat_before=before, image=img, flat_after=after)
                peak = max(float(np.max(frame)) for frame in (before, img, after))
                if peak >= 250:
                    raise ValueError(
                        f"CCD frame at panel ({px}, {py}) peaks at "
                        f"{peak:.1f}/255; reduce exposure and rerun"
                    )
                try:
                    delta, drift = _paired_delta(before, img, after, box, max_flat_drift)
                except ValueError as exc:
                    delta = float("nan")
                    pair_reference = (float(before[box].sum()) + float(after[box].sum())) / 2
                    drift = (abs(float(after[box].sum()) - float(before[box].sum()))
                             / pair_reference if pair_reference > 0 else float("nan"))
                    logger.warning("panel ({}, {}): {}", px, py, exc)
                row.append(delta)
                if np.isfinite(delta):
                    results.append((abs(delta), py, px, delta))
                measurements.append({"x": px, "y": py, "delta": delta if np.isfinite(delta) else None,
                                     "flat_drift": drift if np.isfinite(drift) else None,
                                     "flat_before_energy": float(before[box].sum()),
                                     "candidate_energy": float(img[box].sum()),
                                     "flat_after_energy": float(after[box].sum()),
                                     "flat_centroid_x": _centroid(_flat)[0],
                                     "flat_centroid_y": _centroid(_flat)[1],
                                     "flat_fwhm_px": getattr(_flat, "fwhm_px", None),
                                     "peak": peak, "time_utc": datetime.now(timezone.utc).isoformat()})
            logger.info(
                "panel y={:>4}: {}", py,
                "  ".join(f"x={px}:{v:+7.3f}" for px, v in zip(xs, row, strict=True)),
            )
            if debug_dir is not None:
                (debug_dir / "measurements.json").write_text(
                    json.dumps(measurements, indent=2),
                    encoding="utf-8",
                )

    if not measurements:
        raise ValueError("panel scan has no candidates")
    first_flat = measurements[0]["flat_before_energy"]
    last_flat = measurements[-1]["flat_after_energy"]
    session_reference = (first_flat + last_flat) / 2
    session_drift = abs(last_flat - first_flat) / session_reference if session_reference > 0 else float("inf")
    rejected_count = sum(item["delta"] is None for item in measurements)
    max_pair_drift = max(
        (item["flat_drift"] for item in measurements if item["flat_drift"] is not None),
        default=float("inf"),
    )
    # The beam **moving** invalidates a locate; the beam **brightening or
    # dimming** does not, because every grid point normalises its own reading
    # against the flat measured immediately before and after it
    # (:func:`_paired_delta`). So the session gate is on centroid shift, and the
    # intensity ratio is only reported.
    #
    # Measured 2026-10-17: gating on the intensity ratio refused every run on
    # this bench at 7.9-8.0% while the flat peak itself moved 63 -> 90 over the
    # session -- power drift the pairing had already cancelled. The centroid
    # moved <1 px over the same 6 minutes.
    session_shift_px = float(
        np.hypot(
            measurements[-1]["flat_centroid_x"] - measurements[0]["flat_centroid_x"],
            measurements[-1]["flat_centroid_y"] - measurements[0]["flat_centroid_y"],
        )
    )
    # An absolute pixel limit cannot serve both a 13 px and a 63 px spot: on the
    # broad, dim spot the centroid's own wander is several px, so a fixed 3 px
    # threshold refuses a scan whose answer provably did not move (measured
    # 2026-10-17: 3.47 px shift on FWHM 62.9 px at peak 18/255, while the ranking
    # between the top two patches differed by 9%). Scale the limit with the spot
    # and keep the absolute value as a floor for a tight, bright spot.
    flat_fwhm_px = float(
        np.median([item.get("flat_fwhm_px") or 0.0 for item in measurements]) or 0.0
    )
    shift_limit_px = max_session_shift_px
    if shift_limit_px is not None and max_session_shift_fwhm > 0 and flat_fwhm_px > 0:
        shift_limit_px = max(
            float(max_session_shift_px),
            float(max_session_shift_fwhm) * flat_fwhm_px,
        )
    else:
        flat_fwhm_px = flat_fwhm_px if flat_fwhm_px > 0 else 0.0
    if debug_dir is not None:
        (debug_dir / "summary.json").write_text(
            json.dumps({"session_drift": session_drift, "max_pair_drift": max_pair_drift,
                        "session_shift_px": session_shift_px,
                        "session_shift_limit_px": shift_limit_px,
                        "flat_fwhm_px": flat_fwhm_px,
                        "rejected_count": rejected_count,
                        "max_flat_drift": max_flat_drift,
                        "max_session_drift": max_session_drift,
                        "max_session_shift_px": max_session_shift_px,
                        "max_session_shift_fwhm": max_session_shift_fwhm,
                        "strict": strict,
                        "best_center": [max(results)[2], max(results)[1]] if results else None}, indent=2),
            encoding="utf-8",
        )
    if not results:
        raise ValueError("all panel candidates rejected; check laser stability")
    if strict and rejected_count:
        raise ValueError(f"{rejected_count} panel candidates rejected by flat drift")

    if shift_limit_px is not None and session_shift_px > shift_limit_px:
        raise ValueError(
            f"session beam shift {session_shift_px:.2f} cam px exceeds "
            f"{shift_limit_px:.2f} px ({max_session_shift_fwhm:g} x FWHM "
            f"{flat_fwhm_px:.1f} px, floor {max_session_shift_px} px) -- the beam "
            "moved during the scan, so no single centre describes it; re-run or "
            "stabilise the optical path"
        )
    if session_drift > max_session_drift:
        logger.warning(
            "session flat *intensity* drift {:.1%} exceeds {:.1%}, but every "
            "grid point is normalised against its own paired flat, so this does "
            "not bias the located centre (beam moved {:.2f} cam px, limit "
            "{})",
            session_drift, max_session_drift, session_shift_px,
            "n/a" if max_session_shift_px is None else f"{max_session_shift_px:.2f} px",
        )
    results.sort(reverse=True)
    logger.info("")
    logger.info("=== strongest patch positions (|0-order energy change|) ===")
    for score, py, px, delta in results[:5]:
        logger.info("  panel (x={:>4}, y={:>4})  dE/E={:+.4f}", px, py, delta)
    _, by, bx, _ = results[0]
    logger.info("")
    logger.info(
        "beam is near panel (x={}, y={}); panel centre is ({}, {})",
        bx, by, SLM_PANEL_W // 2, SLM_PANEL_H // 2,
    )
    logger.info(
        "offset from panel centre: dx={:+d}, dy={:+d} panel px -- pass this to the "
        "runbook as --pupil-center, never a camera coordinate",
        bx - SLM_PANEL_W // 2, by - SLM_PANEL_H // 2,
    )
    logger.info(
        "flat drift: max paired {:.2%}, whole scan {:.2%} (intensity; "
        "paired-normalised), beam shift {:.2f} cam px",
        max_pair_drift, session_drift, session_shift_px,
    )
    return PanelLocateResult(
        (bx, by), results[0][3], max_pair_drift, session_drift, rejected_count,
        session_shift_px,
    )


@click.command()
@with_params(PanelLocateParams, kw_name="params")
def main(params: PanelLocateParams) -> None:
    """在面板坐标上定位光斑 (扫描随机相位圆盘, 取 0 阶能量变化最大的位置)。"""
    scan_panel(params)


if __name__ == "__main__":
    main()
