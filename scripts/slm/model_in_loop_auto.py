"""Run a fresh model-in-loop optical session from a bench-specific JSON profile.

The existing hardware CLI owns device access, exposure discovery, panel location,
geometry calibration, holdout probes and phase acceptance. This wrapper records a
repeatable invocation and reviews the result without duplicating those steps.
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.model_in_loop_summary import summarize_session


def _exposure_ms(raw: object) -> float | None:
    """Return the profile's exposure in ms, or ``None`` for a session-chosen one.

    ``auto`` and ``default`` let the session pick (or keep) the exposure, so
    there is no number to compare against ``max_exposure_ms``. A bare ``0`` is
    **not** accepted: the runner rejects it outright (``0`` is not a synonym
    for ``auto``), so a profile that still carries the legacy ``0`` would fail
    only after the device was already open. Rejecting it here keeps the failure
    in the dry run.

    Args:
        raw: The profile value -- a positive number of milliseconds, or one of
            the words ``auto`` / ``default``.

    Returns:
        The exposure in ms, or ``None`` when the session chooses it.

    Raises:
        ValueError: For anything else, including a bare ``0``.
    """
    text = str(raw).strip().lower()
    if text in ("auto", "default"):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "exposure_time_ms must be a positive number of milliseconds or the "
            "word 'auto'/'default'"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "exposure_time_ms must be a positive number of milliseconds or the "
            "word 'auto'/'default'; a bare 0 is not a synonym for 'auto'"
        )
    return value


def build_command(profile: dict) -> list[str]:
    """Validate the portable profile before any device can be opened.

    Two profile fields change what the run does rather than only how it is
    parameterised, and both are documented here because they are the ones a
    caller can get wrong:

    * ``geometry_file`` -- a measured geometry JSON. Passing it **skips** this
      run's automatic session calibration (the full-panel locate, the exposure
      probe and the geometry sweep), so the caller must guarantee that the
      geometry belongs to the *current* experimental setup. The hardware A/B
      campaign relies on that: it measures the geometry once and hands the same
      file to every arm, so the arms differ only in the gate under test.
    * ``error_params`` -- a calibration JSON with the forward model's error
      terms (the SLM's LUT nonlinearity above all). Without it the model
      assumes "command phase = actual phase", which the measured gamma on this
      bench contradicts.
    """
    required = ("cam_type", "cam_id", "slm_number", "slm_wavelength", "target_shape",
                "target_side", "region", "far_field_padding", "n_rounds",
                "max_cv", "min_ee", "min_iou")
    missing = [name for name in required if name not in profile]
    if missing:
        raise ValueError(f"missing profile fields: {', '.join(missing)}")
    if profile["cam_type"] == "sim":
        raise ValueError("this session entry point requires a real camera")
    if profile["target_shape"] not in ("square", "rectangle", "triangle", "star", "circle"):
        raise ValueError("unsupported target_shape")
    for name in ("target_side", "region", "far_field_padding", "n_rounds"):
        if type(profile[name]) is not int or profile[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("max_cv", "min_ee", "min_iou"):
        value = profile[name]
        upper = float("inf") if name == "max_cv" else 1.0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"{name} is outside its valid range")
    exposure = _exposure_ms(profile.get("exposure_time_ms", "auto"))
    ceiling = float(profile.get("max_exposure_ms", 0))
    if not math.isfinite(ceiling) or ceiling < 0 or (exposure is not None and ceiling > 0 and exposure > ceiling):
        raise ValueError("invalid exposure and session ceiling")
    if profile.get("geometry_file"):
        # Allowed on purpose: a geometry file SKIPS this run's automatic session
        # calibration (the full-panel locate and the geometry sweep), so the
        # caller has to guarantee that the geometry was measured on the current
        # experimental setup. The campaign uses exactly that to give its A and B
        # arms the same geometry, so that the only difference between them is
        # the gate under test.
        pass
    mapping = {
        "cam_type": "--cam-type", "cam_id": "--cam-id", "slm_number": "--slm-number",
        "slm_wavelength": "--slm-wavelength", "target_shape": "--target-shape",
        "target_side": "--target-side", "region": "--region",
        "far_field_padding": "--far-field-padding", "n_rounds": "--n-rounds",
        "exposure_time_ms": "--exposure-time-ms", "max_exposure_ms": "--max-exposure-ms",
        "cam_size": "--cam-size", "panel_span_px": "--panel-span-px",
        "camera_pixel_um": "--camera-pixel-um", "slm_pixel_um": "--slm-pixel-um",
        "probe_count": "--probe-count", "holdout_probe_count": "--holdout-probe-count",
        "step_b_iterations": "--step-b-iterations", "seed": "--seed",
        "w_uniformity": "--w-uniformity", "w_efficiency": "--w-efficiency",
        "max_cv_regression": "--max-cv-regression", "min_geometry_correlation": "--min-geometry-correlation",
        "loss_kind": "--loss-kind",
        "min_holdout_correlation": "--min-holdout-correlation",
        "panel_scan_max_flat_drift": "--panel-scan-max-flat-drift",
        "panel_scan_step_px": "--panel-scan-step-px",
        "panel_scan_radius_px": "--panel-scan-radius-px",
        "panel_scan_max_session_shift_px": "--panel-scan-max-session-shift-px",
        "geometry_file": "--geometry-file",
        "error_params": "--error-params",
    }
    unknown = set(profile) - set(mapping) - {"max_cv", "min_ee", "min_iou"}
    if unknown:
        raise ValueError(f"unknown profile fields: {', '.join(sorted(unknown))}")
    command = [sys.executable, str(ROOT / "src/ao_shaping/main.py"), "--debug", "slm-model-in-loop"]
    for name, option in mapping.items():
        if name in profile:
            command.extend((option, str(profile[name])))
    return command


def session_dir_from_output(output: str) -> Path | None:
    matches = re.findall(r"(?:Recorder pkl/json\s*:|Preflight evidence\s*:)\s*(.+)", output)
    return Path(matches[-1].strip()) if matches else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the command without opening devices")
    args = parser.parse_args(argv)
    try:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        command = build_command(profile)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    logger.info("Command:", subprocess.list2cmdline(command))
    if args.dry_run:
        return 0
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", bufsize=1)
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    session_dir = session_dir_from_output("".join(lines))
    if session_dir is not None and code == 0:
        summary = summarize_session(
            session_dir, max_cv=profile["max_cv"], min_ee=profile["min_ee"],
            min_iou=profile["min_iou"],
        )
        (session_dir / "handoff_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Recorded gate verdict: {summary['verdict']} ({session_dir})")
    elif session_dir is not None:
        logger.info(f"Run stopped; inspect evidence in {session_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
