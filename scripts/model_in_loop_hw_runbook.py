"""Hardware runbook: model-in-the-loop square shaping on the real bench.

Needs devices (Santec SLM-200 in memory mode + far-field camera). The sequence
below is packaged so a session does not re-derive it:

    stage collect   phase-diverse calibration records (varied pupil phase)
    stage calibrate solve panel disc / beam waist / far-field scale
    stage shape     compute a square phase with the calibrated model, display
                    it, and measure the real far field

Stage 1 needs *varied* phase on purpose. The geometry is solved by correlating
speckle; Zernike runs built over a large panel radius leave the illuminated core
nearly flat, so there is no speckle to correlate and the solve saturates at the
edge of its search grid (the calibrator warns when that happens).

Usage:
    python scripts/model_in_loop_hw_runbook.py --stage all --target-cam-px 40
    python scripts/model_in_loop_hw_runbook.py --stage collect --probes 12
    python scripts/model_in_loop_hw_runbook.py --stage calibrate
    python scripts/model_in_loop_hw_runbook.py --stage shape --dry-run

See docs/slm/model_in_loop_bench_calibration.md for the measured constants and
the failure modes this script works around.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (  # noqa: E402
    create_target_mask,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (  # noqa: E402
    ZernikeCoefficientOptimizer,
)
from ao_shaping.drivers.ccd.common import create_camera  # noqa: E402
from ao_shaping.drivers.slm.santec import Santec  # noqa: E402
from ao_shaping.optimizer.wfless.model_in_loop_shaping import (  # noqa: E402
    BenchGeometry,
    CalibrationRecord,
    StepBConfig,
    calibrate_bench_geometry,
    calibrate_shared_aberration,
    shape_phase_with_frozen_aberration,
    square_metrics_at_zero_order,
)
from ao_shaping.tools.slm.bench_kernels import (  # noqa: E402
    SLM_PITCH_M,
    core_fraction,
    display_and_average,
    estimate_shift,
    ramp_panel,
)
from ao_shaping.tools.slm.bench_kernels import (  # noqa: E402
    measure_spot as _measure_spot,
)
from ao_shaping.tools.slm.bench_kernels import (  # noqa: E402
    zernike_panel,
)
from ao_shaping.tools.slm.slm_zernike_sweep_probe import (  # noqa: E402
    capture_settled,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center  # noqa: E402


class _LiveStream:
    """Loguru sink that resolves ``sys.<name>`` on every write.

    Loguru binds a stream once, when ``logger.add`` runs, so ``logger.add(sys.stdout)``
    keeps writing to whichever object was bound at import time rather than the one a
    test capture installed afterwards. Resolving the attribute per write keeps records
    on the live stream, which is also what ``capsys`` inspects.

    ``write`` takes ``str`` and returns ``None`` to match loguru's ``Writable``
    sink protocol, which drops the return value.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def write(self, message: str) -> None:
        getattr(sys, self._name).write(message)

    def flush(self) -> None:
        getattr(sys, self._name).flush()


# This runbook narrates to the operator, so its own report stays on stdout -- the
# channel `print` wrote to here before -- while records raised by the libraries it
# calls keep loguru's default stderr sink. The two filters route each record to one
# stream only, so no line is emitted twice. `configure` replaces the default handler
# and passes each dict straight to `add`.
logger.configure(
    handlers=[
        {"sink": _LiveStream("stdout"), "filter": lambda record: record["name"] == __name__},
        {"sink": _LiveStream("stderr"), "filter": lambda record: record["name"] != __name__},
    ]
)

PANEL_H, PANEL_W = 1200, 1920
FULL_SCALE = 255


def probe_phase(region: int, seed: int) -> np.ndarray:
    """Random pupil phase, uniform [0, 2*pi): scatters the beam for speckle."""
    return np.random.default_rng(int(seed)).uniform(
        0.0, 2.0 * np.pi, (int(region), int(region))
    )


def phase_to_panel(
    phase_model: np.ndarray, disc_radius: int, pupil_center: tuple[int, int]
) -> np.ndarray:
    """Resize a model-grid phase onto the panel disc, centred on the beam.

    `pupil_center` is in **panel pixel** coordinates (x, y). It must be measured
    on the panel -- the camera's 0-order position is a different frame entirely
    (the two axes are swapped on this bench and the scales differ), so deriving
    one from the other silently writes the phase where the beam is not.
    """
    r = int(disc_radius)
    region = phase_model.shape[0]
    sub = np.asarray(
        zoom(phase_model, (2 * r / region, 2 * r / region), order=1), dtype=np.float64
    )
    panel = np.zeros((PANEL_H, PANEL_W), dtype=np.float64)
    cx, cy = int(pupil_center[0]), int(pupil_center[1])
    # Clip the window so an off-centre or oversized disc stays in bounds.
    x0, x1 = max(cx - r, 0), min(cx + r, PANEL_W)
    y0, y1 = max(cy - r, 0), min(cy + r, PANEL_H)
    sub = sub[y0 - (cy - r) : y1 - (cy - r), x0 - (cx - r) : x1 - (cx - r)]
    panel[y0:y1, x0:x1] = sub
    return panel


def gaussian_grid(region: int, waist_grid: float) -> np.ndarray:
    """Gaussian illumination on the model grid, masked to the inscribed circle."""
    yy, xx = np.mgrid[0:region, 0:region]
    r2 = (xx - region / 2.0) ** 2 + (yy - region / 2.0) ** 2
    amp = np.exp(-r2 / (2.0 * max(float(waist_grid), 1e-6) ** 2))
    amp[r2 > (region / 2.0) ** 2] = 0.0
    return amp


def save_records(records: list[CalibrationRecord], out: Path) -> None:
    """Persist calibration records so the solve can be re-run offline."""
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=np.stack([np.asarray(r.image, np.float32) for r in records]),
        phases=np.stack([np.asarray(r.phase, np.float32) for r in records]),
        labels=np.array([r.label for r in records]),
    )


def load_records(path: Path) -> list[CalibrationRecord]:
    """Load records written by save_records."""
    data = np.load(path, allow_pickle=False)
    return [
        CalibrationRecord(
            image=np.asarray(data["images"][i], np.float64),
            phase=np.asarray(data["phases"][i], np.float64),
            label=str(data["labels"][i]),
        )
        for i in range(len(data["labels"]))
    ]


def geometry_to_json(geo: BenchGeometry) -> str:
    """Serialise a BenchGeometry."""
    return json.dumps(asdict(geo), indent=2)


def geometry_from_json(text: str) -> BenchGeometry:
    """Inverse of geometry_to_json."""
    return BenchGeometry(**json.loads(text))


def save_phase_png(path: Path, phase: np.ndarray) -> None:
    """Save a displayed phase as a viewable PNG (cyclic map, mod 2*pi).

    The panel phase is what the hardware actually saw, so it is the primary
    evidence that the pupil phase landed on the beam.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrapped = np.mod(np.asarray(phase, dtype=np.float64), 2.0 * np.pi)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, wrapped, cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
    plt.close("all")


def save_frame_png(path: Path, frame: np.ndarray) -> None:
    """Save a CCD frame with a robust percentile stretch.

    A speckle frame has a bright core over a wide dark halo, so a plain min/max
    stretch hides everything; the 99.5th percentile puts the core at full scale
    while keeping the halo visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.asarray(frame, dtype=np.float64)
    hi = float(np.percentile(img, 99.5)) or 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(img / hi, 0.0, 1.0), cmap="inferno", vmin=0.0, vmax=1.0)
    plt.close("all")


def defocus_panel(
    defocus_rad: float, radius: int, pupil_center: tuple[int, int]
) -> np.ndarray:
    """A Noll-4 defocus over the illuminated disc, in panel coordinates.

    Injected into the *displayed* phase only, never into the model's pupil phase,
    so Step A has to recover it from the measurement. That is the whole point: on
    this bench an aberration-free pupil left Step A with no signal (the fit gained
    +0.3%, pure noise), so a known ground truth is what makes the Step A hardware
    result meaningful.
    """
    return zernike_panel({(2, 0): float(defocus_rad)}, radius, pupil_center)


def _pupil_center(args: argparse.Namespace) -> tuple[int, int]:
    """Beam centre in panel pixels (x, y) from the ``--pupil-center`` option.

    Measured on the panel by writing a random-phase disc at candidate positions
    and keeping the one that changes the far field most; the camera 0-order is a
    different coordinate frame and must not be used here.
    """
    parts = str(args.pupil_center).split(",")
    if len(parts) != 2:
        raise SystemExit(
            f"--pupil-center must be 'x,y' in panel px, got {args.pupil_center!r}"
        )
    return int(parts[0]), int(parts[1])


def measure_spot(frame: np.ndarray) -> tuple[float, float, float, float, float]:
    """``(peak, fwhm_px, cx, cy, hollowness)`` for one CCD frame.

    Thin tuple-shaped adapter over the shared kernel in
    :mod:`ao_shaping.tools.slm.bench_kernels`, which owns the logic and the
    offline tests. The kernel despikes with a median before blurring -- a box blur
    alone does not stop a single hot pixel, and on this bench the reference
    centroid used to wander 60 px, which was enough to call a healthy panel
    "unmoved".
    """
    m = _measure_spot(frame)
    return m.peak, m.fwhm_px, m.centroid_x, m.centroid_y, m.hollowness


def crop_around_zero_order(frame: np.ndarray, size: int = 512) -> np.ndarray:
    """Crop a fixed-size window centred on the frame's global argmax.

    The optics axis is the frame's brightest point, never the geometric centre:
    on this bench the 0-order sits ~620 px off-centre in x. The geometry solve
    compares the model's *central* far-field window with the stored frame, so an
    uncropped frame would be compared against a misaligned window and the
    correlation would collapse. This mirrors the pkl records, which are already
    cropped around the spot.

    Args:
        frame: 2D far-field frame.
        size: Window side, pixels. Clamped to the frame.

    Returns:
        The cropped window, zero-padded if the frame is smaller than ``size``.
    """
    data = np.asarray(frame, dtype=np.float64)
    side = int(min(size, data.shape[0], data.shape[1]))
    cy, cx = np.unravel_index(int(np.argmax(data)), data.shape)
    y0, x0 = int(cy) - side // 2, int(cx) - side // 2
    out = np.zeros((side, side), dtype=np.float64)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + side, data.shape[0]), min(x0 + side, data.shape[1])
    out[: sy1 - sy0, : sx1 - sx0] = data[sy0:sy1, sx0:sx1]
    return out


def shoot(cam, slm, phase_panel: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Display a panel phase and return the averaged far-field frame.

    Delegates to :func:`capture_settled` in the sweep probe, which owns the
    settle discipline. It settles on a *stability* criterion rather than a fixed
    wait: the panel was measured still relaxing more than a second after a large
    gray-map change, and an unsettled frame silently corrupts a single-shot sweep
    (a ramp sweep came out non-monotone purely for this reason).
    """
    return capture_settled(
        cam, slm, phase_panel, n_frames=args.frames, n_discard=args.discard,
        wait_time_s=args.settle_s, stable_tol=args.stable_tol,
        max_wait_s=args.max_wait_s,
    )


def stage_collect(args: argparse.Namespace, out_dir: Path) -> None:
    """Capture phase-diverse records for the geometry solve."""
    probes: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    dframes: list[np.ndarray] = []
    vis = out_dir / "frames" / "collect"
    from ao_shaping.utils.io.file import Recorder

    recorder = Recorder(mark="peak", mode="min")
    offset = (0, 0)
    with Santec(
        slm_number=args.slm_number, wavelength=args.slm_wavelength, video_mode=0
    ) as slm, create_camera(
        args.cam_type, args.cam_id, exposure_time_ms=args.exposure_ms
    ) as cam:
        # reset_exposure_time takes milliseconds; passing us clamps to 1000 ms
        # and blows every frame out to full scale.
        cam.reset_exposure_time(float(args.exposure_ms))
        flat = shoot(cam, slm, np.zeros((PANEL_H, PANEL_W)), args)
        if args.save_frames:
            save_frame_png(vis / "flat_ccd.png", flat)
        cx, cy = zero_order_center(flat, refine=False)
        pupil = (int(args.pupil_center.split(",")[0]), int(args.pupil_center.split(",")[1]))
        logger.info(
            "[*] flat peak={:.0f}/{} camera 0-order=({},{})",
            flat.max(), FULL_SCALE, cx, cy,
        )
        logger.info(
            "[*] pupil on panel (x,y)={}  [panel centre ({},{})]  disc r={}px",
            pupil, PANEL_W // 2, PANEL_H // 2, args.collect_disc,
        )
        if flat.max() >= FULL_SCALE * 0.98:
            logger.info("    WARNING: clipping; lower --exposure-ms")
        defocus = defocus_panel(args.defocus_rad, args.zernike_radius, pupil)
        logger.info(
            "[*] capturing TWO sets per probe: bare (geometry solve) and "
            "+{:+.3f} rad Noll-4 defocus over a radius-{} px disc "
            "(Step A ground truth, withheld from the model)",
            args.defocus_rad, args.zernike_radius,
        )
        for index in range(args.probes):
            phase = probe_phase(args.region, args.seed + index)
            base_panel = phase_to_panel(phase, args.collect_disc, pupil)
            # Bare set: the model prediction must match this one exactly, so no
            # aberration may be displayed here or the geometry solve is fitting
            # a model of a beam that does not exist.
            frame = shoot(cam, slm, base_panel, args)
            probes.append(phase)
            frames.append(crop_around_zero_order(frame, args.crop))
            # Defocus set: identical probe seed, plus a known aberration the
            # model never sees, so Step A has something to recover.
            dpanel = base_panel + defocus
            dframe = shoot(cam, slm, dpanel, args)
            dframes.append(crop_around_zero_order(dframe, args.crop))
            tag = f"probe{index:02d}"
            for variant, panel_v, frame_v in (
                ("bare", base_panel, frame),
                ("defocus", dpanel, dframe),
            ):
                pk, fw, rcx, rcy, rh = measure_spot(frame_v)
                recorder.append(
                    {
                        "peak": pk,
                        # save_recorder_debug_artifacts keys the output dict by
                        # `_epoch`, so it must be explicit; two records per probe.
                        "_epoch": 2 * index + (0 if variant == "bare" else 1),
                        "probe": index,
                        # 0 = bare, 2 = +defocus; names in the sidecar.
                        "mode_id": 0 if variant == "bare" else 2,
                        "axis_id": 0,
                        "label": f"{tag}_{variant}",
                        "coefficient": 0.0 if variant == "bare" else args.defocus_rad,
                        "fwhm_px": fw,
                        "centroid_x": rcx,
                        "centroid_y": rcy,
                        "hollowness": rh,
                        "flat_peak": float(flat.max()),
                        "flat_core40": core_fraction(flat, int(cx), int(cy), 40.0),
                        "_phase": np.asarray(panel_v, dtype=np.float32),
                        "_img": np.asarray(frame_v, dtype=np.float32),
                    }
                )
            if args.save_frames:
                save_phase_png(vis / f"{tag}_phase_bare.png", base_panel)
                save_phase_png(vis / f"{tag}_phase_defocus.png", dpanel)
                save_frame_png(vis / f"{tag}_ccd_bare.png", frame)
                save_frame_png(vis / f"{tag}_ccd_defocus.png", dframe)
            logger.info(
                "    probe {}/{}: bare peak={:.0f} defocus peak={:.0f} 0-order={}",
                index + 1, args.probes, frames[-1].max(), dframes[-1].max(),
                zero_order_center(frame, refine=False),
            )
    save_records(
        [
            CalibrationRecord(image=f, phase=p, label=f"probe{i}")
            for i, (p, f) in enumerate(zip(probes, frames, strict=True))
        ],
        out_dir / "calibration_records.npz",
    )
    save_records(
        [
            CalibrationRecord(image=f, phase=p, label=f"probe{i}")
            for i, (p, f) in enumerate(zip(probes, dframes, strict=True))
        ],
        out_dir / "defocus_records.npz",
    )
    (out_dir / "beam_offset.json").write_text(
        json.dumps(
            {
                "pupil_center_panel": list(pupil),
                "collect_disc": args.collect_disc,
                "camera_zero_order": [int(cx), int(cy)],
                "exposure_ms": args.exposure_ms,
            }
        ),
        encoding="utf-8",
    )
    logger.info("[*] saved {} bare records -> {}",
                len(frames), out_dir / "calibration_records.npz")
    logger.info("[*] saved {} defocus records -> {}",
                len(dframes), out_dir / "defocus_records.npz")
    if args.save_frames:
        logger.info("[*] saved phase + CCD images -> {}", vis)
    # Step A needs the model-side pupil phase (the probe alone, without the
    # injected defocus), which the geometry solve does not use.
    np.save(
        out_dir / "probe_phases.npy",
        np.stack([np.asarray(p, dtype=np.float32) for p in probes]),
    )
    logger.info("[*] saved model-side probe phases -> {}",
                out_dir / "probe_phases.npy")
    png = _save_recorder(
        recorder, args, "collect", "SLM phase-diverse probe records (bare + defocus)",
        extra={
            "n_probes": len(probes),
            "camera_zero_order": [int(cx), int(cy)],
            "mode_codes": {"bare": 0, "defocus": 2},
            "array_keys": ["_phase (displayed panel, radians)",
                           "_img (full CCD frame)"],
        },
    )
    logger.info("[*] saved recorder artefacts -> {}", png.parent)


#: Stable integer codes for the sweep modes, so the per-record pkl keeps the
#: provenance as numeric columns. ``save_recorder_debug_artifacts`` casts every
#: ``scalar_keys`` entry with ``float()`` and silently drops anything not
#: declared, so string columns cannot survive it; the name mapping goes into the
#: json sidecar instead (see ``_recorder_payload``).
SWEEP_MODE_CODES: dict[str, int] = {
    "flat": 0,
    "tilt": 1,
    "defocus": 2,
    "astig_x": 3,
    "astig_y": 4,
    "ramp": 5,
    "coma_x": 6,
    "coma_y": 7,
    "spherical": 8,
}
AXIS_CODES: dict[str, int] = {"": 0, "x": 1, "y": 2}


def _recorder_payload(
    args: argparse.Namespace, stage: str, extra: dict | None = None
) -> dict:
    """The ``.json`` sidecar describing a run, in the repo's standard shape."""
    payload = {
        "stage": stage,
        "bench": "Santec SLM-200 #1 22030108, 1920x1200, 2pi=993 @1064nm; "
        "Daheng MER2-507 2592x1944 @2.2um",
        "pupil_center_panel": list(_pupil_center(args)),
        "collect_disc": args.collect_disc,
        "zernike_radius": args.zernike_radius,
        "defocus_rad": args.defocus_rad,
        "exposure_ms": args.exposure_ms,
        "frames_per_point": args.frames,
        "slm_number": args.slm_number,
        "slm_wavelength": args.slm_wavelength,
        "cam_type": args.cam_type,
        "cam_id": args.cam_id,
        "region": args.region,
        "far_field_size": args.far_field_size,
        "method": args.method,
        "note": "panel<->camera axes are swapped 90 deg on this bench; the camera "
        "0-order is NOT a panel coordinate",
        "mode_codes": SWEEP_MODE_CODES,
        "axis_codes": AXIS_CODES,
    }
    if extra:
        payload.update(extra)
    return payload


def _save_recorder(
    recorder,
    args: argparse.Namespace,
    stage: str,
    title: str,
    extra: dict | None = None,
):
    """Persist a :class:`Recorder` in the repo's standard debug-artefact shape.

    Writes ``data/debug/<prefix>_<ts>/{recorder_<ts>.pkl, .json, .png}``. The pkl is
    a plain ``{epoch: record}`` dict, which is the form every report generator
    under ``scripts/generate_*_report.py`` already knows how to read, so a hardware
    sweep can be re-rendered offline without re-running the bench.
    """
    from ao_shaping.utils.io.file import save_recorder_debug_artifacts

    return save_recorder_debug_artifacts(
        recorder,
        root_dir=str(ROOT / "data"),
        subdir_prefix=f"model_in_loop_hw_{stage}",
        scalar_keys=(
            "coefficient", "fwhm_px", "centroid_x", "centroid_y", "peak",
            "hollowness", "flat_peak", "flat_core40", "mode_id", "axis_id",
            "probe", "shift_x", "shift_y", "shift_corr",
        ),
        img_keys=("_img", "_phase"),
        json_payload=_recorder_payload(args, stage, extra),
        title=title,
    )


def stage_sweep(args: argparse.Namespace, out_dir: Path) -> None:
    """Capture a smooth Zernike tilt + defocus sweep for the sweep calibration.

    Replaces the speckle route, which cannot work on this bench: the panel's
    effective phase resolution is far coarser than one pixel, so a random pupil
    phase leaves the far field a single tight focus and there is no speckle to
    correlate. Smooth modes *do* produce a measurable response, and a tilt sweep
    fixes the far-field scale with no assumption about f, pitch or pixel size.
    """
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import SweepRecord
    from ao_shaping.utils.io.file import Recorder

    pupil = _pupil_center(args)
    vis = out_dir / "frames" / "sweep"
    tilt_axis = [(1, 1), (1, -1)]  # (n, m) for the two tilts
    points: list[SweepRecord] = []
    recorder = Recorder(mark="peak", mode="min")
    flat_peak = float("nan")
    flat_core = float("nan")
    flat_frame: np.ndarray | None = None

    with Santec(
        slm_number=args.slm_number, wavelength=args.slm_wavelength, video_mode=0
    ) as slm, create_camera(
        args.cam_type, args.cam_id, exposure_time_ms=args.exposure_ms
    ) as cam:
        cam.reset_exposure_time(float(args.exposure_ms))

        def acquire(mode: str, coeff: float, axis: str, tag: str) -> None:
            nonlocal flat_peak, flat_core, flat_frame
            if mode == "flat":
                panel = np.zeros((PANEL_H, PANEL_W))
            else:
                if mode == "tilt":
                    nm = tilt_axis[0] if axis == "x" else tilt_axis[1]
                elif mode == "defocus":
                    nm = (2, 0)
                else:
                    nm = {
                        "astig_x": (2, -2), "astig_y": (2, 2),
                        "coma_x": (3, -1), "coma_y": (3, 1),
                        "spherical": (4, 0),
                    }[mode]
                panel = zernike_panel({nm: float(coeff)}, args.zernike_radius, pupil)
            frame = shoot(cam, slm, panel, args)
            peak, fwhm, cx, cy, hollow = measure_spot(frame)
            if mode == "flat":
                flat_peak = peak
                flat_core = core_fraction(frame, cx, cy, 40.0)
                flat_frame = frame
            points.append(
                SweepRecord(
                    label=tag, mode=mode, coefficient=float(coeff), axis=axis,
                    fwhm_px=fwhm, centroid_x=cx, centroid_y=cy, peak=peak,
                    hollowness=hollow,
                )
            )
            recorder.append(
                {
                    # `peak` is the Recorder mark (mode="min"), so a run can be
                    # ranked by "sharpest focus" without re-reading the arrays.
                    "peak": peak,
                    # `peak` is the Recorder mark (mode="min"), so a run can be
                    # ranked by "sharpest focus" without re-reading the arrays.
                    # save_recorder_debug_artifacts keys the output dict by
                    # `_epoch` (Recorder.append only sets `_id`), so it must be
                    # explicit -- this is the index into the sweep order.
                    "_epoch": len(points) - 1,
                    # Provenance as numeric columns; names live in the sidecar.
                    "mode_id": SWEEP_MODE_CODES[mode],
                    "axis_id": AXIS_CODES[axis],
                    "label": tag,
                    "coefficient": float(coeff),
                    "fwhm_px": fwhm,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "hollowness": hollow,
                    "flat_peak": flat_peak,
                    "flat_core40": flat_core,
                    # Raw process data: the displayed panel phase (radians) and
                    # the CCD frame it produced.
                    "_phase": np.asarray(panel, dtype=np.float32),
                    "_img": np.asarray(frame, dtype=np.float32),
                }
            )
            if args.save_frames:
                save_phase_png(vis / f"{tag}_phase.png", panel)
                save_frame_png(vis / f"{tag}_ccd.png", frame)
            logger.info(
                "    {:<16} peak={:>6.0f} fwhm={:>6.1f}px "
                "hollow={:>4.2f} c=({:7.1f},{:7.1f})",
                tag, peak, fwhm, hollow, cx, cy,
            )

        # Flat FIRST: the panel retains the previous run's pattern, so a "flat"
        # read before any write is really the last speckle frame.
        logger.info("[*] flat reference")
        acquire("flat", 0.0, "", "flat")

        # Ramp sweep -- the authoritative focal-scale measurement. A 2*pi phase
        # ramp over P panel px, applied across the WHOLE panel, moves the spot by
        # S/P camera px and the spot stays a single lobe at any amplitude. A
        # Zernike tilt of the equivalent coefficient does not behave the same
        # (measured 3x too small above ~1.5 rad), so this is a real ramp, not an
        # equivalent tilt.
        for axis in ("x", "y"):
            for period in args.sweep_ramps:
                panel = ramp_panel(
                    int(period), 1 if axis == "x" else 0, (PANEL_H, PANEL_W)
                )
                frame = shoot(cam, slm, panel, args)
                pk, fw, rcx, rcy, rh = measure_spot(frame)
                points.append(
                    SweepRecord(
                        label=f"ramp{axis}{period}", mode="ramp",
                        coefficient=float(period), axis=axis, fwhm_px=fw,
                        centroid_x=rcx, centroid_y=rcy, peak=pk, hollowness=rh,
                    )
                )
                recorder.append(
                    {
                        "peak": pk,
                        "_epoch": len(points) - 1,
                        "mode_id": SWEEP_MODE_CODES["ramp"],
                        "axis_id": AXIS_CODES[axis],
                        "coefficient": float(period),
                        "fwhm_px": fw,
                        "centroid_x": rcx,
                        "centroid_y": rcy,
                        "hollowness": rh,
                        "flat_peak": flat_peak,
                        "flat_core40": flat_core,
                        "_phase": np.asarray(panel, dtype=np.float32),
                        "_img": np.asarray(frame, dtype=np.float32),
                    }
                )
                logger.info(
                    "    ramp {} P={:>5}px  peak={:>6.0f} "
                    "fwhm={:>6.1f}px c=({:7.1f},{:7.1f})",
                    axis, period, pk, fw, rcx, rcy,
                )
        for axis in ("x", "y"):
            # ABBA interleave: A B B A around each point, so slow intensity drift
            # (the reason the two tilt axes disagreed by 11% at +/-1 rad) cancels
            # instead of biasing one side of the sweep.
            for coeff in args.sweep_tilt:
                seq = [coeff, -coeff, -coeff, coeff] if args.sweep_repeats >= 2 else [coeff]
                logger.info("[*] tilt {} = {:+.3f} rad (x{})", axis, coeff, len(seq))
                spots = []
                for k, c in enumerate(seq):
                    nm = tilt_axis[0] if axis == "x" else tilt_axis[1]
                    panel = zernike_panel({nm: float(c)}, args.zernike_radius, pupil)
                    frame = shoot(cam, slm, panel, args)
                    _pk, _fw, rcx, rcy, _rh = measure_spot(frame)
                    # Phase correlation against the flat reference: immune to the
                    # spot deforming, unlike the centroid.
                    if flat_frame is not None:
                        sdy, sdx, scorr = estimate_shift(flat_frame, frame)
                    else:
                        sdy = sdx = scorr = float("nan")
                    spots.append((rcx, rcy))
                    points.append(
                        SweepRecord(
                            label=f"tilt{axis}{c:+.2f}#{k}", mode="tilt",
                            coefficient=float(c), axis=axis, fwhm_px=_fw,
                            centroid_x=rcx, centroid_y=rcy, peak=_pk, hollowness=_rh,
                            shift_x=sdx, shift_y=sdy, shift_corr=scorr,
                        )
                    )
                    recorder.append(
                        {
                            "peak": _pk,
                            "_epoch": len(points) - 1,
                            "mode_id": SWEEP_MODE_CODES["tilt"],
                            "axis_id": AXIS_CODES[axis],
                            "coefficient": float(c),
                            "fwhm_px": _fw,
                            "centroid_x": rcx,
                            "centroid_y": rcy,
                            "hollowness": _rh,
                            "shift_x": sdx,
                            "shift_y": sdy,
                            "shift_corr": scorr,
                            "flat_peak": flat_peak,
                            "flat_core40": flat_core,
                            "_phase": np.asarray(panel, dtype=np.float32),
                            "_img": np.asarray(frame, dtype=np.float32),
                        }
                    )
                    if args.save_frames:
                        tag = f"tilt{axis}{c:+.2f}_{k}"
                        save_phase_png(vis / f"{tag}_phase.png", panel)
                        save_frame_png(vis / f"{tag}_ccd.png", frame)
                mcx = float(np.mean([s[0] for s in spots]))
                mcy = float(np.mean([s[1] for s in spots]))
                logger.info(
                    "    tilt {} {:+.3f} mean c=({:7.1f},{:7.1f}) "
                    "spread={:.2f}px "
                    "shift=({:+.2f},{:+.2f}) "
                    "corr={:.3f}",
                    axis, coeff, mcx, mcy,
                    max(abs(s[0] - mcx) for s in spots),
                    np.nanmean([p.shift_x for p in points if p.axis == axis]),
                    np.nanmean([p.shift_y for p in points if p.axis == axis]),
                    np.nanmean([p.shift_corr for p in points if p.axis == axis]),
                )
        for coeff in args.sweep_defocus:
            logger.info("[*] defocus = {:+.3f} rad", coeff)
            acquire("defocus", coeff, "", f"defocus{coeff:+.2f}")
        for mode in ("astig_x", "astig_y"):
            for coeff in args.sweep_astig:
                logger.info("[*] {} = {:+.3f} rad", mode, coeff)
                acquire(mode, coeff, "", f"{mode}{coeff:+.2f}")
        for mode, coeffs in (
            ("coma_x", args.sweep_coma), ("coma_y", args.sweep_coma),
            ("spherical", args.sweep_spherical),
        ):
            for coeff in coeffs:
                logger.info("[*] {} = {:+.3f} rad", mode, coeff)
                acquire(mode, coeff, "", f"{mode}{coeff:+.2f}")

    np.savez_compressed(
        out_dir / "sweep_records.npz",
        label=np.array([p.label for p in points]),
        mode=np.array([p.mode for p in points]),
        axis=np.array([p.axis for p in points]),
        coefficient=np.array([p.coefficient for p in points]),
        fwhm_px=np.array([p.fwhm_px for p in points]),
        centroid_x=np.array([p.centroid_x for p in points]),
        centroid_y=np.array([p.centroid_y for p in points]),
        peak=np.array([p.peak for p in points]),
        hollowness=np.array([p.hollowness for p in points]),
        shift_x=np.array([p.shift_x for p in points]),
        shift_y=np.array([p.shift_y for p in points]),
        shift_corr=np.array([p.shift_corr for p in points]),
        # Provenance the fit needs, so a later `--stage calibrate` cannot
        # silently substitute its own defaults (it used to: a 200 px default
        # against a sweep run at 450 px made the physical check read "1.6 mm").
        zernike_radius=np.array(args.zernike_radius),
        collect_disc=np.array(args.collect_disc),
        pupil_center=np.array(_pupil_center(args)),
    )
    logger.info("[*] saved {} sweep points -> {}",
                len(points), out_dir / "sweep_records.npz")
    if args.save_frames:
        logger.info("[*] saved phase + CCD images -> {}", vis)
    # Standard debug artefacts, so the run can be re-rendered offline by the
    # report generators without re-running the bench.
    png = _save_recorder(
        recorder, args, "sweep", "SLM smooth-Zernike sweep (tilt/defocus/astig)",
        extra={
            "n_points": len(points),
            "sweep_tilt_rad": list(args.sweep_tilt),
            "sweep_defocus_rad": list(args.sweep_defocus),
            "sweep_astig_rad": list(args.sweep_astig),
            "sweep_ramps_panel_px": list(args.sweep_ramps),
            "sweep_coma_rad": list(args.sweep_coma),
            "sweep_spherical_rad": list(args.sweep_spherical),
            "array_keys": ["_phase (displayed panel, radians)",
                           "_img (CCD frame)"],
        },
    )
    logger.info("[*] saved recorder artefacts -> {}", png.parent)


def stage_calibrate_sweep(args: argparse.Namespace, out_dir: Path) -> BenchGeometry:
    """Solve the geometry from the tilt + defocus sweep."""
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
        SweepRecord,
        calibrate_bench_geometry_from_sweep,
    )

    path = out_dir / "sweep_records.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run --stage sweep first")
    d = np.load(path, allow_pickle=False)
    # Provenance from the sweep that was actually run, not this command's
    # defaults: the physical aperture check is only meaningful against the radius
    # the Zernike modes were built on.
    sweep_radius = int(d["zernike_radius"]) if "zernike_radius" in d else args.zernike_radius
    sweep_disc = int(d["collect_disc"]) if "collect_disc" in d else args.collect_disc
    if sweep_radius != args.zernike_radius or sweep_disc != args.collect_disc:
        logger.info(
            "    NOTE: sweep was run with zernike_radius={}, "
            "collect_disc={}; this command has "
            "{}/{}. Using the sweep's values.",
            sweep_radius, sweep_disc, args.zernike_radius, args.collect_disc,
        )
    records = [
        SweepRecord(
            label=str(d["label"][i]), mode=str(d["mode"][i]),
            coefficient=float(d["coefficient"][i]), axis=str(d["axis"][i]),
            fwhm_px=float(d["fwhm_px"][i]), centroid_x=float(d["centroid_x"][i]),
            centroid_y=float(d["centroid_y"][i]), peak=float(d["peak"][i]),
            hollowness=float(d["hollowness"][i]) if "hollowness" in d else 1.0,
            shift_x=float(d["shift_x"][i]) if "shift_x" in d else float("nan"),
            shift_y=float(d["shift_y"][i]) if "shift_y" in d else float("nan"),
            shift_corr=float(d["shift_corr"][i]) if "shift_corr" in d else float("nan"),
        )
        for i in range(len(d["label"]))
    ]
    geo = calibrate_bench_geometry_from_sweep(
        records,
        region=args.region,
        far_field_size=args.far_field_size,
        model_to_panel=2.0 * sweep_disc / args.region,
        # Physical cross-check on the tilt slope: the measured shift per radian
        # of Zernike tilt must be consistent with the illuminated aperture, or the
        # spot has deformed and the measurement is worthless.
        aperture_m=sweep_radius * SLM_PITCH_M,
    )
    (out_dir / "bench_geometry.json").write_text(geometry_to_json(geo), encoding="utf-8")
    logger.info("{}", json.dumps(asdict(geo), indent=2))
    logger.info("  notes: {}", geo.calibration_notes)
    return geo


def stage_calibrate(args: argparse.Namespace, out_dir: Path) -> BenchGeometry:
    """Solve the bench geometry from the captured records."""
    path = out_dir / "calibration_records.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run --stage collect first")
    records = load_records(path)
    logger.info("[*] calibrating on {} records", len(records))
    geo = calibrate_bench_geometry(
        records,
        region=args.region,
        far_field_size=args.far_field_size,
        # The probes were displayed over a 2 x collect_disc box, so the model
        # grid samples the pupil at (2*collect_disc/region) x the SLM pitch.
        # Omitting this makes the modelled aperture 3.5x too small physically.
        panel_span_px=2.0 * args.collect_disc,
    )
    (out_dir / "bench_geometry.json").write_text(geometry_to_json(geo), encoding="utf-8")
    logger.info("{}", json.dumps(asdict(geo), indent=2))
    if geo.correlation < 0.75:
        logger.info(
            "    NOTE: correlation < 0.75 -- the pupil phase probably did not vary"
        )
        logger.info(
            "    across the illuminated area, so this geometry is NOT usable."
        )
    return geo


def stage_fit(args: argparse.Namespace, out_dir: Path) -> None:
    """Run Step A on the real captures and score it against the injected defocus.

    This is the validation that was impossible on an aberration-free bench: the
    displayed phase carries a known Noll-4 defocus that the model never sees, so
    the recovered coefficient is a true ground-truth comparison.
    """
    # The images must come from the DEFOCUS set: the bare set is what the
    # geometry solve consumed, and it contains no aberration to recover.
    rec_path = out_dir / "defocus_records.npz"
    probe_path = out_dir / "probe_phases.npy"
    geo_path = out_dir / "bench_geometry.json"
    for path in (rec_path, probe_path, geo_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; run --stage collect/calibrate first")
    records = load_records(rec_path)
    probes = np.load(probe_path)
    geo = geometry_from_json(geo_path.read_text(encoding="utf-8"))
    logger.info("[*] Step A on {} real captures, injected defocus {:+.3f} rad",
                len(records), args.defocus_rad)

    fit_records = [
        CalibrationRecord(
            image=r.image, phase=np.asarray(probes[i], dtype=np.float64), label=r.label
        )
        for i, r in enumerate(records)
    ]
    result = calibrate_shared_aberration(
        fit_records, region=geo.region, n_orders=args.n_orders,
        epochs=args.fit_epochs, lr=args.fit_lr, seed=args.seed,
        far_field_size=geo.far_field_size,
    )
    noll = 4  # (2,0) defocus
    recovered = float(result.coefficients[noll - 1])
    logger.info("  source={}  records={}  steps={}",
                result.source, result.n_records, result.iterations)
    logger.info("  injected Noll 4 : {:+.4f} rad", args.defocus_rad)
    logger.info("  recovered Noll 4: {:+.4f} rad   error {:.4f} rad",
                recovered, abs(recovered - args.defocus_rad))
    others = np.delete(result.coefficients, noll - 1)
    logger.info("  ||other modes|| = {:.4f} rad (max |c| {:.4f})",
                float(np.linalg.norm(others)), float(np.abs(others).max()))
    if result.loss_history:
        logger.info("  loss {:.4e} -> {:.4e}",
                    result.loss_history[0], result.loss_history[-1])
    else:
        logger.info("  no loss history")
    (out_dir / "step_a_fit.json").write_text(
        json.dumps(
            {
                "injected_defocus_rad": args.defocus_rad,
                "recovered_defocus_rad": recovered,
                "error_rad": abs(recovered - args.defocus_rad),
                "n_records": result.n_records,
                "iterations": result.iterations,
                "other_modes_rms": float(np.sqrt(np.mean(others**2))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "RECOVERED" if abs(recovered - args.defocus_rad) < 0.3 else "NOT RECOVERED"
    logger.info("  VERDICT: {} (tolerance 0.3 rad)", verdict)


def stage_shape(args: argparse.Namespace, out_dir: Path) -> None:
    """Compute a square phase with the calibrated model and measure the result."""
    geo_path = out_dir / "bench_geometry.json"
    if not geo_path.exists():
        raise SystemExit(f"missing {geo_path}; run --stage calibrate first")
    geo = geometry_from_json(geo_path.read_text(encoding="utf-8"))
    # compute_square_metrics indexes with the side, so it needs an int; the
    # float form is only meaningful for converting to model pixels.
    side_cam = max(1, int(round(args.target_cam_px)))
    side = geo.model_side_for_camera(side_cam)
    logger.info("[*] target {} camera px -> {} model px", side_cam, side)
    logger.info(
        "    region={} far_field_size={} disc={}px w0={:.0f}px corr={:.3f}",
        geo.region, geo.far_field_size, geo.panel_disc_radius,
        geo.beam_waist_panel_px, geo.correlation,
    )

    region = geo.region
    opt = ZernikeCoefficientOptimizer(
        n_orders=6, region=region, dtype="float64", device="cpu", seed=0,
        far_field_size=geo.far_field_size,
    )
    zeros = np.zeros(opt.n_coefficients)
    amp = gaussian_grid(
        region, geo.beam_waist_panel_px * region / (2.0 * geo.panel_disc_radius)
    )
    target = create_target_mask("square", (region, region), side)
    step_b = StepBConfig(iterations=args.shape_iterations, lr=0.05, target_side=side)

    if args.dry_run:
        phase, history = shape_phase_with_frozen_aberration(
            opt, zeros, target, np.zeros((region, region)), step_b,
            dtype="float64", device="cpu", seed=0, source_amplitude=amp,
        )
        predicted = opt.forward_intensity(zeros, phase, amp)
        logger.info("    [dry-run] Step B loss {:.4f} -> {:.4f}",
                    history[0], history[-1])
        logger.info("    [dry-run] target {} model px = {:.1f} camera px",
                    side, side * geo.camera_px_per_model_px)
        logger.info("    [dry-run] predicted far field {} finite={}",
                    predicted.shape, bool(np.isfinite(predicted).all()))
        logger.info(
            "    [dry-run] panel placement {}",
            phase_to_panel(phase, geo.panel_disc_radius, _pupil_center(args)).shape,
        )
        return

    recorded = json.loads((out_dir / "beam_offset.json").read_text(encoding="utf-8"))
    pupil = _pupil_center(args)
    if tuple(recorded.get("pupil_center_panel", pupil)) != tuple(pupil):
        logger.info("    NOTE: pupil {} differs from the collect run {}; "
                    "geometry solve used the latter",
                    pupil, recorded.get("pupil_center_panel"))
    with Santec(
        slm_number=args.slm_number, wavelength=args.slm_wavelength, video_mode=0
    ) as slm, create_camera(
        args.cam_type, args.cam_id, exposure_time_ms=args.exposure_ms
    ) as cam:
        cam.reset_exposure_time(float(args.exposure_ms))
        vis = out_dir / "frames" / "shape"
        before = shoot(cam, slm, np.zeros((PANEL_H, PANEL_W)), args)
        if args.save_frames:
            save_frame_png(vis / "before_ccd.png", before)
        base, base_c = square_metrics_at_zero_order(before, side_cam)
        logger.info(
            "    baseline EE={:.4f} cv={:.4f} flat={:.4f} centre={}",
            base["encircled_energy"], base["uniformity_cv"],
            base["flatness_factor"], base_c,
        )
        phase, history = shape_phase_with_frozen_aberration(
            opt, zeros, target, np.zeros((region, region)), step_b,
            dtype="float64", device="cpu", seed=0, source_amplitude=amp,
        )
        logger.info("    Step B loss {:.4f} -> {:.4f}", history[0], history[-1])
        shaped_panel = phase_to_panel(phase, geo.panel_disc_radius, pupil)
        if args.save_frames:
            save_phase_png(vis / "shaped_phase.png", shaped_panel)
        after = shoot(cam, slm, shaped_panel, args)
        if args.save_frames:
            save_frame_png(vis / "after_ccd.png", after)
            logger.info("    saved phase + CCD images -> {}", vis)
        shaped, shaped_c = square_metrics_at_zero_order(after, side_cam)
        logger.info(
            "    shaped   EE={:.4f} cv={:.4f} flat={:.4f} centre={}",
            shaped["encircled_energy"], shaped["uniformity_cv"],
            shaped["flatness_factor"], shaped_c,
        )
        logger.info("\n    VERDICT")
        logger.info(
            "      cv   {:.4f} -> {:.4f}  ({})",
            base["uniformity_cv"], shaped["uniformity_cv"],
            "FLATTER" if shaped["uniformity_cv"] < base["uniformity_cv"]
            else "NOT flatter",
        )
        logger.info("      flat {:.4f} -> {:.4f}",
                    base["flatness_factor"], shaped["flatness_factor"])
        logger.info("      EE   {:.4f} -> {:.4f}",
                    base["encircled_energy"], shaped["encircled_energy"])
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        np.save(out_dir / f"hw_before_{stamp}.npy", before)
        np.save(out_dir / f"hw_shaped_{stamp}.npy", after)
        np.save(out_dir / f"hw_phase_{stamp}.npy", phase)


def dry_run(args: argparse.Namespace, out_dir: Path) -> None:
    """Exercise every non-device step, including a synthetic geometry solve."""
    logger.info("[dry-run] geometry search on synthetic records")
    region, ffs, disc, waist = 128, 2048, 200, 24.0
    cam_px = 2.2e-6
    cam_per_model = (1064e-9 * 0.125 / (ffs * 8e-6)) / cam_px
    rng = np.random.default_rng(3)
    panel = 4 * disc
    opt = ZernikeCoefficientOptimizer(
        n_orders=1, region=region, dtype="float64", device="cpu", far_field_size=ffs
    )
    zeros = np.zeros(opt.n_coefficients)
    amp = gaussian_grid(region, waist * region / (2.0 * disc))
    records = []
    for _ in range(3):
        panel_phase = rng.uniform(0.0, 2 * np.pi, (panel, panel))
        cy = cx = panel // 2
        sub = panel_phase[cy - disc : cy + disc, cx - disc : cx + disc]
        grid = np.asarray(zoom(sub, (region / sub.shape[0],) * 2, order=1))
        model = opt.forward_intensity(zeros, grid, amp)
        rows = int(round(160 / cam_per_model))
        top = (model.shape[0] - rows) // 2
        frame = np.asarray(
            zoom(model[top : top + rows, top : top + rows],
                 (160 / rows, 160 / rows), order=1)
        )
        records.append(CalibrationRecord(image=frame, phase=panel_phase, label="dry"))
    geo = calibrate_bench_geometry(records, region=region, far_field_size=ffs)
    logger.info("[dry-run] disc={} (truth {})  w0={:.1f} (truth {})  corr={:.4f}",
                geo.panel_disc_radius, disc, geo.beam_waist_panel_px, waist,
                geo.correlation)
    logger.info("[dry-run] target 40 camera px -> {} model px",
                geo.model_side_for_camera(40))
    logger.info(
        "[dry-run] panel placement {}",
        phase_to_panel(np.zeros((region, region)), geo.panel_disc_radius, (0, 0)).shape,
    )
    stage_shape(args, out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("collect", "sweep", "calibrate", "fit", "shape", "all"),
                    default="all")
    ap.add_argument(
        "--sweep-tilt", default="-1.0,1.0",
        help="Tilt coefficients in rad. Keep this SMALL: theory says 5.34 cam px "
        "per rad for the 450 px illuminated radius, and the small-angle runs "
        "measured 3.3-3.6 (ratio 0.63, credible). At +/-1.5-3 rad the spot "
        "deforms, the centroid stops tracking the peak, and the measured slope "
        "halves to 1.7 -- which would imply an 1120 mm aperture, larger than the "
        "panel. The calibrator now checks this ratio and warns.",
    )
    ap.add_argument(
        "--sweep-coma", default="-1.2,-0.6,0.6,1.2",
        help="coma_x/coma_y sweep coefficients in rad. Together with spherical "
        "these are what make the beam waist identifiable; the quadratic modes "
        "alone stay degenerate with residual defocus.",
    )
    ap.add_argument("--sweep-spherical", default="-1.2,-0.6,0.6,1.2",
                    help="spherical (4,0) sweep coefficients in rad")
    ap.add_argument(
        "--sweep-ramps", default="120,240,480,960,1920",
        help="Phase-ramp periods in panel px for the authoritative focal-scale "
        "measurement. Displacement is S/P, measured cleanly at any amplitude; "
        "the equivalent Zernike tilt does NOT behave the same above ~1.5 rad.",
    )
    ap.add_argument(
        "--sweep-repeats", type=int, default=2,
        help="Extra interleaved (ABBA) repeats per tilt point, to cancel slow "
        "intensity drift. Averages into one measurement.",
    )
    ap.add_argument(
        "--sweep-defocus", default="-4.0,-2.5,-1.5,-0.75,0.75,1.5,2.5,4.0",
        help="Defocus coefficients in rad. The range must reach well past the "
        "point where the spot width saturates, otherwise the illumination "
        "profile (waist) is not constrained -- a ±2 rad sweep on this bench "
        "fitted to only 25% residual",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "model_in_loop_hw")
    ap.add_argument("--probes", type=int, default=12)
    ap.add_argument("--region", type=int, default=256)
    ap.add_argument("--far-field-size", type=int, default=4096)
    ap.add_argument("--collect-disc", type=int, default=200)
    # argparse has no `--x/--no-x` syntax (that is Click); it needs two options
    # sharing one dest, or `--x/--no-x` silently becomes a value-taking option.
    ap.add_argument(
        "--save-frames", dest="save_frames", action="store_true", default=True,
        help="Save every displayed panel phase and CCD frame under <out>/frames "
        "(the primary evidence that the pupil phase landed on the beam)",
    )
    ap.add_argument(
        "--no-save-frames", dest="save_frames", action="store_false",
        help="Do not write the phase/CCD image record",
    )
    ap.add_argument(
        "--pupil-center",
        default="1000,480",
        help=(
            "Beam centre in PANEL pixels 'x,y'. Measured on the panel (a random-"
            "phase disc written at candidate positions, keep the one that changes "
            "the far field most). Do NOT use the camera 0-order: on this bench the "
            "two axes are swapped and the scales differ."
        ),
    )
    ap.add_argument("--crop", type=int, default=512,
                    help="window size stored per capture, centred on the 0-order")
    ap.add_argument("--target-cam-px", type=float, default=40.0)
    ap.add_argument("--shape-iterations", type=int, default=600)
    ap.add_argument("--slm-number", type=int, default=1)
    ap.add_argument("--slm-wavelength", type=int, default=1064)
    ap.add_argument("--cam-type", default="daheng")
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--exposure-ms", type=float, default=1.1)
    ap.add_argument("--defocus-rad", type=float, default=1.0,
                    help="Noll-4 defocus injected into the DISPLAYED phase only")
    ap.add_argument("--zernike-radius", type=int, default=200,
                    help="Zernike radius of the injected defocus, panel px")
    ap.add_argument(
        "--method", choices=("speckle", "sweep"), default="sweep",
        help="Geometry calibration: 'sweep' (smooth Zernike tilt+defocus, the "
        "only route that works on this bench) or 'speckle' (random-phase "
        "correlation, needs pixel-scale phase the panel cannot deliver)",
    )
    ap.add_argument(
        "--sweep-astig", default="-3.0,-1.5,1.5,3.0",
        help="Astigmatism coefficients in rad for both astig axes. Defocus alone "
        "leaves the bench's astigmatism/coma unmodelled and the waist "
        "unidentifiable; sweeping astig too lets the two be separated",
    )
    ap.add_argument("--n-orders", type=int, default=6)
    ap.add_argument("--fit-epochs", type=int, default=150)
    ap.add_argument("--fit-lr", type=float, default=0.1)
    ap.add_argument("--settle-s", type=float, default=0.5,
                    help="显式液晶翻转等待 s (默认 0.5)")
    ap.add_argument("--stable-tol", type=float, default=0.02,
                    help="稳定判据: 连续两次的 peak/质心 相对偏差 (默认 0.02)")
    ap.add_argument("--max-wait-s", type=float, default=6.0,
                    help="等待稳定的上限 s (默认 6.0)")
    ap.add_argument("--discard", type=int, default=3)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.sweep_tilt = [float(v) for v in str(args.sweep_tilt).split(",") if v.strip()]
    args.sweep_defocus = [
        float(v) for v in str(args.sweep_defocus).split(",") if v.strip()
    ]
    args.sweep_astig = [float(v) for v in str(args.sweep_astig).split(",") if v.strip()]
    args.sweep_ramps = [float(v) for v in str(args.sweep_ramps).split(",") if v.strip()]
    args.sweep_coma = [float(v) for v in str(args.sweep_coma).split(",") if v.strip()]
    args.sweep_spherical = [float(v) for v in str(args.sweep_spherical).split(",") if v.strip()]
    args.sweep_repeats = int(args.sweep_repeats)

    if args.dry_run:
        # Give the dry run a self-consistent synthetic geometry to consume.
        geo = BenchGeometry(
            panel_disc_radius=200, region=128, beam_waist_panel_px=24.0,
            far_field_size=2048, camera_px_per_model_px=(1064e-9 * 0.125 / (2048 * 8e-6)) / 2.2e-6,
            spot_fwhm_camera_px=12.0, spot_fwhm_model_px=44.0, correlation=0.97,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "bench_geometry.json").write_text(
            geometry_to_json(geo), encoding="utf-8"
        )
        dry_run(args, args.out)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("collect", "all"):
        stage_collect(args, args.out)
    if args.stage in ("sweep", "all"):
        stage_sweep(args, args.out)
    if args.stage in ("calibrate", "all"):
        if args.method == "sweep" or not (args.out / "calibration_records.npz").exists():
            stage_calibrate_sweep(args, args.out)
        else:
            stage_calibrate(args, args.out)
    if args.stage in ("fit", "all"):
        stage_fit(args, args.out)
    if args.stage in ("shape", "all"):
        stage_shape(args, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
