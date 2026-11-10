"""Issue #237 P0: offline 2D registration replay on the 2026-11-03 refine data.

Question answered: once the model far field is cropped on the *same* sampling
window as the measured CCD crop, can a learned registration (2 translations, or
a 2x2 affine plus 2 translations) recover the alignment that was previously
obtained by an external, out-of-model window search?

Reference numbers (issue #237 section 1(1), reproduced offline in this repo):

* integer shift (-3, +32) px, zero-shift NCC 0.0316, best-shift NCC 0.6802.

What this script does NOT claim:

* It does not claim the 0.68 baseline is "recovered" by the model. The 0.68
  figure is a best-shift value found by an external integer search. The
  registration result is reported as numbers; the comparison is labelled, not
  scored as success.
* It does not auto-pair files across directories. Each model/measured pair is
  taken from the refine run's own ``refine_summary.json`` round records
  (``phase``, ``native_frame`` and their recorded SHA-256). A pair whose file
  hash does not match the recorded hash is refused and recorded as refused.
* Missing sampling-window metadata is recorded as missing, never filled in.

Offline only: no SLM, camera or laser is opened and no ``ao_shaping.drivers``
module is imported. The default mode is plan-only and writes nothing;
``--execute`` runs the replay on CPU and writes one JSON summary.

Usage::

    python scripts/slm/replay_2d_registration_1103.py            # plan only
    python scripts/slm/replay_2d_registration_1103.py --execute  # translation fit
    python scripts/slm/replay_2d_registration_1103.py --execute --affine
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEVICE_STATEMENT = (
    "离线脚本：未打开 SLM / 相机 / 激光，未 import 任何 ao_shaping.drivers 模块；"
    "模型与测量均来自已归档的 npy 文件与 refine_summary.json。"
)

#: Half-size of the compared window (``analysis_window.size`` is 256 px).
HALF = 128
#: Search margin around the window in the model far field (px).
MARGIN = 60
#: Absolute clamp on the translation parameters (px).
CLAMP_SHIFT_PX = 64.0
#: Absolute clamp on the 2x2 linear part (dimensionless).
CLAMP_LINEAR = 0.2

#: Reference values from issue #237 section 1(1) (external window alignment).
BASELINE_SHIFT_XY = (-3, 32)
BASELINE_ZERO_NCC = 0.0315531992032807
BASELINE_BEST_NCC = 0.6802495575769043
BASELINE_TOLERANCE = 1e-6

DEFAULT_DATA_ROOT = ROOT / "data" / "debug"
CAPTURE_GLOB = "sim2f_fov_hw_20261103*"
REFINE_GLOB = "sim2f_fov_hw_refine_ring_20261103_*"
DEFAULT_CALIBRATION = DEFAULT_DATA_ROOT / "sim2f_fov_hw_20261102" / "calibration"
DEFAULT_OUT = (
    DEFAULT_DATA_ROOT / "registration_replay_20261103" / "replay_2d_registration_1103.json"
)

#: Window keys the crop convention depends on (must be recorded, else refuse).
REQUIRED_WINDOW_KEYS: tuple[str, ...] = (
    "analysis_window",
    "native_square_offset_xy",
    "square_zero_order_xy",
    "full_model_px",
)

#: Keys that a sampling-window export would carry. Absence is recorded.
OPTIONAL_CAMERA_KEYS: tuple[str, ...] = ("camera_xy",)


@dataclass(frozen=True)
class PairSpec:
    """One model/measured pair, as declared by the refine run itself.

    Attributes:
        name: Round identifier, e.g. ``round_0001`` or ``reference``.
        phase_path: Commanded phase file (128x128 radians).
        frame_path: Native full-sensor frame file.
        phase_sha256: Recorded phase hash, or ``None`` when not recorded.
        frame_sha256: Recorded frame hash, or ``None`` when not recorded.
    """

    name: str
    phase_path: Path
    frame_path: Path
    phase_sha256: str | None
    frame_sha256: str | None


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(root: Path, pattern: str) -> list[Path]:
    """List directories under ``root`` matching ``pattern``, sorted by name."""
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob(pattern) if path.is_dir())


def describe_dir(directory: Path) -> dict[str, Any]:
    """Count the arrays and manifest items of one capture or refine directory."""
    counts: dict[str, int] = {}
    for sub in ("frames", "phases", "panel", "panel_gray"):
        folder = directory / sub
        if folder.is_dir():
            counts[sub] = len(list(folder.glob("*.npy")))
    manifest: dict[str, Any] = {}
    manifest_path = directory / "capture_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["items"] = len(payload.get("items", {}))
    return {
        "dir": str(directory),
        "frame_counts": counts,
        "has_refine_summary": (directory / "refine_summary.json").is_file(),
        "has_capture_manifest": manifest_path.is_file(),
        **manifest,
    }


def load_refine_summary(refine_dir: Path) -> dict[str, Any]:
    """Read ``refine_summary.json`` of a refine run."""
    path = refine_dir / "refine_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def missing_window_metadata(summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(missing_required, missing_optional)`` window keys.

    A missing required key refuses the replay of the run. A missing optional key
    (the model's own ``camera_xy`` sampling grid) is recorded and does not refuse,
    because the crop convention is fully determined by the required keys.
    """
    round_keys: set[str] = set()
    for record in summary.get("rounds", []):
        round_keys.update(record.keys())
    available = set(summary.keys()) | round_keys
    missing_required = [key for key in REQUIRED_WINDOW_KEYS if key not in summary]
    missing_optional = [key for key in OPTIONAL_CAMERA_KEYS if key not in available]
    return missing_required, missing_optional


def build_pairs(refine_dir: Path, summary: dict[str, Any]) -> list[PairSpec]:
    """Declare the pairs of one refine run from its own round records only."""
    pairs: list[PairSpec] = []
    for record in summary.get("rounds", []):
        pairs.append(
            PairSpec(
                name=str(record["id"]),
                phase_path=refine_dir / str(record["phase"]),
                frame_path=refine_dir / str(record["native_frame"]),
                phase_sha256=record.get("phase_sha256"),
                frame_sha256=record.get("native_frame_sha256"),
            )
        )
    reference_frame = refine_dir / "frames" / "reference.npy"
    reference_phase = refine_dir / "phases" / "reference.npy"
    if reference_frame.is_file() and reference_phase.is_file():
        pairs.append(
            PairSpec(
                name="reference",
                phase_path=reference_phase,
                frame_path=reference_frame,
                phase_sha256=None,
                frame_sha256=None,
            )
        )
    return pairs


def check_calibration(summary: dict[str, Any], calibration_dir: Path) -> dict[str, Any]:
    """Compare the local calibration.json hash with the hash the run recorded."""
    recorded = (summary.get("request", {}).get("calibration_files") or {}).get(
        "calibration.json"
    )
    local_path = calibration_dir / "calibration.json"
    local = sha256_file(local_path) if local_path.is_file() else None
    return {
        "dir": str(calibration_dir),
        "recorded_sha256": recorded,
        "local_sha256": local,
        "match": recorded is not None and recorded == local,
    }


def _zncc(a: Any, b: Any) -> float:
    """Zero-mean normalised cross-correlation of two equal-shape arrays."""
    import numpy as np

    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 0 else 0.0


def _register(region: Any, meas: Any, affine: bool) -> dict[str, Any]:
    """Fit the registration of one model region against one measured crop.

    The integer best shift from ``match_template`` initialises a bounded
    continuous fit (Powell). In translation mode the parameters are ``(tx, ty)``.
    In affine mode they are ``(a, b, c, d, tx, ty)`` with the linear part
    ``[[1+a, b], [c, 1+d]]``. The sampled model is ``region`` evaluated at
    ``A @ (x, y) + t`` on the measured grid (bilinear, zero outside).
    """
    import numpy as np
    from scipy.ndimage import map_coordinates
    from scipy.optimize import minimize
    from skimage.feature import match_template

    size = int(meas.shape[0])
    half = size // 2
    centre = float(MARGIN + half)
    zero_before = _zncc(region[MARGIN : MARGIN + size, MARGIN : MARGIN + size], meas)
    ncc_map = match_template(region, meas, pad_input=False)
    iy, ix = np.unravel_index(int(np.nanargmax(ncc_map)), ncc_map.shape)
    integer_shift = (int(ix) - MARGIN, int(iy) - MARGIN)

    yy, xx = np.mgrid[-half:half, -half:half].astype(np.float64)

    def sample(params: Any) -> Any:
        if affine:
            a, b, c, d, tx, ty = (float(v) for v in params)
        else:
            tx, ty = (float(v) for v in params)
            a = b = c = d = 0.0
        xs = (1.0 + a) * xx + b * yy + tx + centre
        ys = c * xx + (1.0 + d) * yy + ty + centre
        return map_coordinates(region, [ys, xs], order=1, mode="constant", cval=0.0)

    def negative_ncc(params: Any) -> float:
        return -_zncc(sample(params), meas)

    if affine:
        x0 = np.array([0.0, 0.0, 0.0, 0.0, *map(float, integer_shift)])
        bounds = [(-CLAMP_LINEAR, CLAMP_LINEAR)] * 4 + [(-CLAMP_SHIFT_PX, CLAMP_SHIFT_PX)] * 2
    else:
        x0 = np.array([float(v) for v in integer_shift], dtype=np.float64)
        bounds = [(-CLAMP_SHIFT_PX, CLAMP_SHIFT_PX)] * 2

    result = minimize(
        negative_ncc,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 4000, "xtol": 1e-3, "ftol": 1e-7},
    )
    params = [float(v) for v in result.x]
    names = ["a", "b", "c", "d", "tx", "ty"] if affine else ["tx", "ty"]
    ncc_after = _zncc(sample(result.x), meas)
    if affine:
        a, b, c, d, tx, ty = params
        matrix = [[1.0 + a, b], [c, 1.0 + d]]
        translation = [tx, ty]
    else:
        tx, ty = params
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        translation = [tx, ty]
    # A parameter resting on its clamp means the fit wanted to leave the allowed
    # neighbourhood: the estimate is then not identifiable on this data and the
    # reported NCC gain must not be read as a physical scale or shear.
    at_bound = [
        name
        for name, value, (low, high) in zip(names, params, bounds)
        if min(abs(value - low), abs(value - high)) <= 1e-6
    ]
    return {
        "mode": "affine_2x2_plus_translation" if affine else "translation",
        "ncc_zero_before": zero_before,
        "integer_shift_xy": list(integer_shift),
        "ncc_integer_best": float(ncc_map[iy, ix]),
        "matrix": matrix,
        "translation_xy": translation,
        "ncc_after_registration": ncc_after,
        "improved_vs_zero_before": bool(ncc_after > zero_before),
        "parameters_at_bound": at_bound,
        "clamp_saturated": bool(at_bound),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_evaluations": int(result.nfev),
    }


def _crop_pair(
    pred: Any,
    frame: Any,
    *,
    square_offset_xy: tuple[int, int],
    centre_xy: tuple[int, int],
    full_px: int,
) -> tuple[Any, Any]:
    """Return ``(model_region, measured_crop)`` on the recorded sampling window."""
    ox, oy = square_offset_xy
    cx, cy = centre_xy
    square = frame[oy : oy + full_px, ox : ox + full_px]
    meas = square[cy - HALF : cy + HALF, cx - HALF : cx + HALF]
    region = pred[cy - HALF - MARGIN : cy + HALF + MARGIN, cx - HALF - MARGIN : cx + HALF + MARGIN]
    return region, meas


def load_model(calibration_dir: Path, full_px: int) -> Any:
    """Build the calibrated bench on CPU (same construction as the 11-03 diagnostics)."""
    from dataclasses import replace

    from ml.sim2f.calibrate import CalibratedBench, load_calibration
    from ml.sim2f.optics import Sim2f, Sim2fConfig

    cfg = replace(Sim2fConfig.calibrated(), grid=128, ccd_px=full_px)
    return CalibratedBench(Sim2f(cfg, device="cpu"), load_calibration(calibration_dir))


def predict(model: Any, phase_path: Path, full_px: int) -> Any:
    """Predict the full-sensor far field of one commanded phase on CPU."""
    import numpy as np
    import torch

    phase = np.load(phase_path).astype(np.float32)
    with torch.no_grad():
        out = model.forward(torch.as_tensor(phase)[None], detector=False)
    return out[0, 0].cpu().numpy().astype(np.float64)


def replay_pair(
    model: Any,
    spec: PairSpec,
    summary: dict[str, Any],
    *,
    affine: bool,
) -> dict[str, Any]:
    """Replay one declared pair; refuse it when a recorded hash does not match."""
    import numpy as np

    record: dict[str, Any] = {"round": spec.name, "phase": str(spec.phase_path),
                              "frame": str(spec.frame_path)}
    if not spec.phase_path.is_file() or not spec.frame_path.is_file():
        record.update(status="refused", reason="declared file missing")
        return record

    phase_hash = sha256_file(spec.phase_path)
    frame_hash = sha256_file(spec.frame_path)
    record["phase_sha256_local"] = phase_hash
    record["frame_sha256_local"] = frame_hash
    if spec.phase_sha256 is not None and spec.phase_sha256 != phase_hash:
        record.update(status="refused", reason="phase sha256 differs from recorded value")
        return record
    if spec.frame_sha256 is not None and spec.frame_sha256 != frame_hash:
        record.update(status="refused", reason="frame sha256 differs from recorded value")
        return record
    record["provenance"] = (
        "recorded_sha256" if spec.frame_sha256 is not None else "file_name_only_unhashed"
    )

    full_px = int(summary["full_model_px"])
    window = summary["analysis_window"]
    centre_xy = (int(window["center_xy"][0]), int(window["center_xy"][1]))
    square_offset = (
        int(summary["native_square_offset_xy"][0]),
        int(summary["native_square_offset_xy"][1]),
    )
    pred = predict(model, spec.phase_path, full_px)
    frame = np.load(spec.frame_path).astype(np.float64)
    region, meas = _crop_pair(
        pred, frame, square_offset_xy=square_offset, centre_xy=centre_xy, full_px=full_px
    )
    fit = _register(region, meas, affine)
    record.update(status="ok", fit=fit)
    if spec.name == "reference":
        record["reproduces_baseline_shift"] = bool(
            tuple(fit["integer_shift_xy"]) == BASELINE_SHIFT_XY
            and abs(fit["ncc_zero_before"] - BASELINE_ZERO_NCC) < BASELINE_TOLERANCE
            and abs(fit["ncc_integer_best"] - BASELINE_BEST_NCC) < BASELINE_TOLERANCE
        )
    record["comparison_to_external_window_baseline"] = {
        "baseline_shift_xy": list(BASELINE_SHIFT_XY),
        "baseline_zero_ncc": BASELINE_ZERO_NCC,
        "baseline_best_ncc": BASELINE_BEST_NCC,
        "ncc_after_registration": fit["ncc_after_registration"],
        "difference": fit["ncc_after_registration"] - BASELINE_BEST_NCC,
        "improved_vs_baseline_best_ncc": bool(
            fit["ncc_after_registration"] >= BASELINE_BEST_NCC - BASELINE_TOLERANCE
        ),
        "note": (
            "numeric comparison only; the 0.68 baseline is an external integer "
            "window search, so matching it is not a claim that it was recovered"
        ),
    }
    return record


def plan_refine_dir(refine_dir: Path, calibration_dir: Path) -> None:
    """Log the plan for one refine run (no model load, no writes)."""
    summary = load_refine_summary(refine_dir)
    missing_required, missing_optional = missing_window_metadata(summary)
    calib = check_calibration(summary, calibration_dir)
    logger.info(f"[plan] refine dir        : {refine_dir}")
    logger.info(f"[plan]   frame counts    : {describe_dir(refine_dir)['frame_counts']}")
    logger.info(f"[plan]   window          : center_xy={summary.get('analysis_window', {}).get('center_xy')} "
                f"native_square_offset_xy={summary.get('native_square_offset_xy')} "
                f"full_model_px={summary.get('full_model_px')}")
    logger.info(f"[plan]   missing required window keys : {missing_required or 'none'}")
    logger.info(f"[plan]   missing optional keys        : {missing_optional or 'none'}"
                " (recorded, not filled in)")
    logger.info(f"[plan]   calibration hash match       : {calib['match']} "
                f"(recorded={calib['recorded_sha256']}, local={calib['local_sha256']})")
    for spec in build_pairs(refine_dir, summary):
        logger.info(f"[plan]   pair {spec.name:<12} phase={spec.phase_path.name} "
                    f"frame={spec.frame_path.name} recorded_frame_sha256="
                    f"{(spec.frame_sha256 or 'none')[:16]}")


def run_refine_dir(
    refine_dir: Path, calibration_dir: Path, *, affine: bool, model_cache: dict[int, Any]
) -> dict[str, Any]:
    """Run the replay for one refine run and return its JSON-ready record."""
    summary = load_refine_summary(refine_dir)
    missing_required, missing_optional = missing_window_metadata(summary)
    calib = check_calibration(summary, calibration_dir)
    base: dict[str, Any] = {
        "refine_dir": str(refine_dir),
        "missing_required_window_keys": missing_required,
        "missing_optional_keys": missing_optional,
        "calibration": calib,
        "mode": "affine" if affine else "translation",
    }
    if missing_required:
        base.update(status="refused", reason="missing required sampling-window metadata")
        logger.error(f"[replay] refused {refine_dir.name}: missing {missing_required}")
        return base
    full_px = int(summary["full_model_px"])
    if full_px not in model_cache:
        model_cache[full_px] = load_model(calibration_dir, full_px)
    model = model_cache[full_px]
    pairs = [
        replay_pair(model, spec, summary, affine=affine)
        for spec in build_pairs(refine_dir, summary)
    ]
    for item in pairs:
        if item["status"] == "ok":
            fit = item["fit"]
            logger.info(
                f"[replay] {refine_dir.name} {item['round']}: "
                f"zero {fit['ncc_zero_before']:.4f} -> {fit['ncc_after_registration']:.4f}; "
                f"integer {fit['integer_shift_xy']} ncc_best {fit['ncc_integer_best']:.4f}; "
                f"translation {[round(v, 3) for v in fit['translation_xy']]}"
            )
            if fit["clamp_saturated"]:
                logger.warning(
                    f"[replay] {item['round']} hit the parameter clamp on "
                    f"{fit['parameters_at_bound']}: the affine estimate is not "
                    "identifiable here and the NCC gain is not a physical scale"
                )
        else:
            logger.error(f"[replay] {refine_dir.name} {item['round']} refused: {item['reason']}")
    base.update(status="ok", pairs=pairs)
    return base


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success (including refused pairs, which are recorded), ``1``
        when no refine run was found or the calibration directory is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--refine-dir",
        type=Path,
        action="append",
        default=None,
        help="Refine run directory (repeatable). Default: auto-discover 11-03 refine runs.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--affine", action="store_true",
        help="Fit a 2x2 affine plus translation (6 params) instead of 2 translations.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Run the replay on CPU and write the JSON summary. Default: plan only.",
    )
    args = parser.parse_args(argv)

    logger.info(DEVICE_STATEMENT)
    captures = discover(args.data_root.resolve(), CAPTURE_GLOB)
    logger.info(f"[plan] capture dirs discovered: {len(captures)}")
    for directory in captures:
        info = describe_dir(directory)
        logger.info(f"[plan]   {directory.name}: frames={info['frame_counts']} "
                    f"manifest_items={info.get('items')}")

    refine_dirs = (
        [path.resolve() for path in args.refine_dir]
        if args.refine_dir
        else discover(args.data_root.resolve(), REFINE_GLOB)
    )
    if not refine_dirs:
        logger.error("[plan] no refine run directory found")
        return 1
    calibration_dir = args.calibration.resolve()
    if not (calibration_dir / "calibration.json").is_file():
        logger.error(f"[plan] calibration directory missing: {calibration_dir}")
        return 1

    if not args.execute:
        for refine_dir in refine_dirs:
            plan_refine_dir(refine_dir, calibration_dir)
        logger.info("[plan] mode: " + ("affine" if args.affine else "translation")
                    + "; nothing was written (add --execute to replay and write JSON)")
        return 0

    model_cache: dict[int, Any] = {}
    runs = [
        run_refine_dir(refine_dir, calibration_dir, affine=args.affine, model_cache=model_cache)
        for refine_dir in refine_dirs
    ]
    payload = {
        "schema": "replay_2d_registration_1103/1",
        "issue": 237,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_statement": DEVICE_STATEMENT,
        "crop": {"half_px": HALF, "margin_px": MARGIN},
        "clamp": {"shift_px": CLAMP_SHIFT_PX, "linear": CLAMP_LINEAR},
        "baseline_issue_237": {
            "shift_xy": list(BASELINE_SHIFT_XY),
            "zero_ncc": BASELINE_ZERO_NCC,
            "best_ncc": BASELINE_BEST_NCC,
        },
        "runs": runs,
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"[execute] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
