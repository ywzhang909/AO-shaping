"""Campaign orchestration for the model-in-loop hardware A/B study.

One command answers three questions, each with an expectation that is written
down before the hardware is touched:

* **baseline_a** -- the holdout correlation gate is left where it was
  (``0.5``). The A arm is the **gate arm**: the threshold is not moved, so the
  expectation is that the run is stopped -- by the holdout gate, or by the
  bake-off against the flat baseline. It is *not* expected to produce a
  shaping score.
* **baseline_b** -- the gate is forced open (``--min-holdout-correlation 0``),
  so Step B runs to completion. The expectation is the real one: the shaped
  phase's composite score must be **strictly greater** than the flat
  baseline's. A completed run that lost the bake-off (``bake_off_rejected``)
  has *not* met the expectation.
* **large_1000`` / ``large_1600`` -- the same forced gate with a square target
  of 1000-1600 camera pixels. The camera window has to be at least as large as
  the target, otherwise the target box falls outside the readout.

Shared geometry
---------------
The two arms must differ **only** in the gate under test, so they have to see
the same bench geometry. Two runs of the automatic session calibration on this
bench produced camera/model scales of 1.3043 and 2.6531 -- a factor of two --
so letting each arm calibrate on its own would compare two different
experiments. The campaign therefore measures the geometry **once**:

* with ``--geometry-file`` the given (sweep) geometry is used by every arm;
* otherwise the **first** model-in-loop arm runs the automatic session
  calibration, and the geometry file it produced is copied to
  ``<session>/bench_geometry.json`` and handed to every later arm.

If the first arm produced no geometry file, the later arms fall back to their
own session calibration and the manifest says so (a silent fallback would make
the arms incomparable without anyone noticing).

Every hardware stage is preceded by the light-level gate
(``scripts/slm/diagnose_light_level.py``): without light there is nothing to
shape, so the campaign stops instead of blaming the forward model.

The script owns no device code. The hardware command line is built by
``scripts/slm/model_in_loop_auto.py`` and the recorded verdict by
``ao_shaping.tools.slm.model_in_loop_summary``; this file only sequences them,
collects the evidence and writes the manifest. ``--plan`` prints the whole
campaign (commands + profiles) without opening a device, so the plan can be
reviewed while the bench is offline.

Usage:
    python scripts/slm/model_in_loop_hw_ab.py --plan
    python scripts/slm/model_in_loop_hw_ab.py
    python scripts/slm/model_in_loop_hw_ab.py --stage baseline_a --stage baseline_b

Exit codes:
    0  the campaign ran to the end (an unmet expectation is a *result*, not an error)
    1  a stage failed unexpectedly (crash, timeout, unreadable output)
    2  the campaign was refused: the light gate blocked it, or a stage was
       rejected by the geometry bake-off (``bench_outside_model``)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.model_in_loop_summary import summarize_session  # noqa: E402

AUTO_SCRIPT = ROOT / "scripts" / "slm" / "model_in_loop_auto.py"
LIGHT_SCRIPT = ROOT / "scripts" / "slm" / "diagnose_light_level.py"
#: 设备可用性探针 (SLM USB 句柄是独占的: 谁先 open 谁占住, 别人只拿到 -10003)。
#: 复用采集驱动 (scripts/slm/run_hw_sweep_groups.py) 已有的探针与分类, 不另写一份。
DEVICE_PROBE_SCRIPT = ROOT / "scripts" / "slm" / "run_hw_sweep_groups.py"

MIL_SUBCOMMAND = "slm-model-in-loop"

#: The fields ``model_in_loop_auto.build_command`` accepts. A profile that
#: carries anything else is rejected with ``ValueError``.
PROFILE_FIELDS = frozenset(
    {
        "cam_type", "cam_id", "slm_number", "slm_wavelength", "target_shape",
        "target_side", "region", "far_field_padding", "n_rounds",
        "exposure_time_ms", "max_exposure_ms", "cam_size", "panel_span_px",
        "camera_pixel_um", "slm_pixel_um", "probe_count", "holdout_probe_count",
        "step_b_iterations", "seed", "w_uniformity", "w_efficiency",
        "max_cv_regression", "min_geometry_correlation",
        "min_holdout_correlation", "max_cv", "min_ee", "min_iou", "loss_kind",
        "panel_scan_max_flat_drift", "panel_scan_step_px",
        "geometry_file", "error_params",
    }
)

#: The stages the campaign knows, in the order they are executed.
STAGE_ORDER = ("light", "baseline_a", "baseline_b", "large")

#: The exposure words the runner accepts instead of a number of milliseconds.
EXPOSURE_TOKENS = ("auto", "default")

#: Name of the shared geometry file inside the campaign session directory.
SHARED_GEOMETRY_NAME = "bench_geometry.json"

#: What each stage is expected to show. Written into the manifest before the
#: run, so a result can be compared against it afterwards.
EXPECTED_TEXT = {
    "light": "verdict ok (light reaches the camera)",
    "baseline_a": (
        "A arm = the gate is left where it was; expected to be stopped -- by the "
        "holdout gate or by the bake-off against the flat baseline -- not to "
        "produce a shaping score"
    ),
    "baseline_b": (
        "completed + the shaped composite score strictly greater than the flat "
        "baseline score"
    ),
    "large": (
        "completed + the shaped composite score strictly greater than the flat "
        "baseline score"
    ),
}


def _load_script_module(name: str, path: Path):
    """Load a sibling script by path so its helpers are reused verbatim."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_AUTO = _load_script_module("model_in_loop_auto", AUTO_SCRIPT)
build_command = _AUTO.build_command
session_dir_from_output = _AUTO.session_dir_from_output


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def exposure_value(raw: object) -> str | float:
    """Normalise an ``--exposure-time-ms`` value into a profile value.

    Args:
        raw: The token (``auto`` / ``default``) or a number of milliseconds.

    Returns:
        The token as a plain string, a positive float otherwise.

    Raises:
        ValueError: For anything that is neither a token nor a positive number.
    """
    text = str(raw).strip()
    if text.lower() in EXPOSURE_TOKENS:
        return text.lower()
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{raw!r} is not a usable exposure: use 'auto', 'default' or a "
            "positive number of milliseconds"
        ) from exc
    if not value > 0.0:
        raise ValueError(
            f"{raw!r} is not a usable exposure: 0 is not a synonym for 'auto'"
        )
    return value


def _exposure_arg(raw: str) -> str | float:
    """argparse adapter around :func:`exposure_value`."""
    try:
        return exposure_value(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the campaign arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        action="append",
        choices=list(STAGE_ORDER),
        help="Stage to run; repeatable. Default: every stage.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "debug" / "model_in_loop_hw_ab",
        help="Campaign artefact root.",
    )
    parser.add_argument("--cam-type", choices=["daheng", "miicam"], default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument(
        "--exposure-time-ms",
        type=_exposure_arg,
        default="auto",
        help="Exposure token ('auto' / 'default') or a positive number of ms.",
    )
    parser.add_argument(
        "--max-exposure-ms",
        type=float,
        default=0.0,
        help="Exposure ceiling in ms; 0 lets the frame decide.",
    )
    parser.add_argument("--cam-size", type=int, default=300)
    parser.add_argument("--n-rounds", type=int, default=1)
    parser.add_argument("--probe-count", type=int, default=6)
    parser.add_argument("--holdout-probe-count", type=int, default=2)
    parser.add_argument("--step-b-iterations", type=int, default=600)
    parser.add_argument("--region", type=int, default=128)
    parser.add_argument("--far-field-padding", type=int, default=8)
    parser.add_argument(
        "--target-shape", default="rectangle", help="Target shape of the A/B arms."
    )
    parser.add_argument(
        "--target-side", type=int, default=60, help="Target side of the A/B arms (camera px)."
    )
    parser.add_argument(
        "--large-sides",
        default="1000,1600",
        help="Comma separated target sides of the large-area arm (camera px).",
    )
    parser.add_argument("--large-target-shape", default="square")
    parser.add_argument(
        "--gate-holdout",
        type=float,
        default=0.5,
        help="Holdout correlation gate of the A arm (left where it was).",
    )
    parser.add_argument(
        "--forced-holdout",
        type=float,
        default=0.0,
        help="Holdout correlation gate of the B arm and the large-area arm (forced open).",
    )
    parser.add_argument("--max-cv", type=float, default=0.5)
    parser.add_argument("--min-ee", type=float, default=0.4)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--camera-pixel-um", type=float, default=2.2)
    parser.add_argument("--slm-pixel-um", type=float, default=8.0)
    parser.add_argument("--panel-span-px", type=float, default=900.0)
    parser.add_argument("--max-cv-regression", type=float, default=0.02)
    parser.add_argument("--min-geometry-correlation", type=float, default=0.5)
    parser.add_argument(
        "--geometry-file",
        type=Path,
        default=None,
        help=(
            "Measured sweep geometry JSON shared by EVERY arm. Passing it skips "
            "each arm's automatic session calibration, so the arms differ only "
            "in the gate under test. Without it the first arm measures the "
            "geometry and the later arms reuse that file."
        ),
    )
    parser.add_argument(
        "--error-params",
        type=Path,
        default=None,
        help=(
            "Calibration JSON with the forward model's error terms (the SLM's "
            "LUT nonlinearity gamma above all). Handed to every arm."
        ),
    )
    parser.add_argument(
        "--panel-scan-max-flat-drift",
        type=float,
        default=0.02,
        help="Paired flat/candidate/flat drift limit of the panel-locate preflight. "
        "The panel scan aborts the session when ANY candidate exceeds it, so the "
        "limit has to sit above the bench's own shot-noise floor: at 20 ms with a "
        "45-count peak the 0-order box energy carries ~1.8% shot noise (2026-10-21 "
        "measured), so 0.02 rejects candidates on noise alone.",
    )
    parser.add_argument(
        "--panel-scan-step-px",
        type=int,
        default=120,
        help="Panel-locate grid spacing in panel pixels.",
    )
    parser.add_argument(
        "--light-exposures",
        default="0.02,0.1,0.5,1,2,5,20",
        help="Exposure sweep of the light-level gate (ms).",
    )
    parser.add_argument(
        "--skip-light-gate", action="store_true", help="Do not run the light-level gate."
    )
    parser.add_argument(
        "--allow-weak",
        action="store_true",
        help="Continue when the light-level gate reports 'weak'.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the campaign (commands + profiles) without opening a device.",
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=1800.0,
        help="Seconds to wait for the SLM/camera to become free before the first "
        "stage (0 disables). The SLM's USB handle is exclusive, and a collision "
        "can also surface mid-run as -10032 during the panel scan.",
    )
    parser.add_argument(
        "--loss-kind",
        choices=("mse", "l1"),
        default="mse",
        help="Step A fit loss: peak-normalised MSE (default) or the image-domain L1 "
        "that GRAPHYCS uses (the only single switch that improved every metric on "
        "the recorded probes).",
    )
    parser.add_argument(
        "--timeout-s", type=float, default=5400.0, help="Timeout of one stage in seconds."
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Plan construction (pure)
# ---------------------------------------------------------------------------


def large_sides(text: str) -> list[int]:
    """Parse the ``--large-sides`` list into positive integers."""
    sides: list[int] = []
    for chunk in str(text).split(","):
        item = chunk.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"target side must be a positive integer, got {value}")
        sides.append(value)
    return sides


def build_profiles(args: argparse.Namespace) -> dict[str, dict]:
    """Return the profile of every model-in-loop stage, keyed by stage name.

    The profiles carry **only** fields ``model_in_loop_auto.build_command``
    accepts; an unknown field is a ``ValueError`` there, so the schema is
    spelled out here rather than spread over the call sites.
    """
    base: dict = {
        "cam_type": args.cam_type,
        "cam_id": args.cam_id,
        "slm_number": args.slm_number,
        "slm_wavelength": args.slm_wavelength,
        "target_shape": args.target_shape,
        "target_side": args.target_side,
        "region": args.region,
        "far_field_padding": args.far_field_padding,
        "n_rounds": args.n_rounds,
        "exposure_time_ms": args.exposure_time_ms,
        "max_exposure_ms": args.max_exposure_ms,
        "cam_size": args.cam_size,
        "panel_span_px": args.panel_span_px,
        "camera_pixel_um": args.camera_pixel_um,
        "slm_pixel_um": args.slm_pixel_um,
        "probe_count": args.probe_count,
        "holdout_probe_count": args.holdout_probe_count,
        "step_b_iterations": args.step_b_iterations,
        "seed": args.seed,
        "max_cv_regression": args.max_cv_regression,
        "min_geometry_correlation": args.min_geometry_correlation,
        "loss_kind": args.loss_kind,
        "panel_scan_max_flat_drift": args.panel_scan_max_flat_drift,
        "panel_scan_step_px": args.panel_scan_step_px,
        "max_cv": args.max_cv,
        "min_ee": args.min_ee,
        "min_iou": args.min_iou,
    }
    if args.error_params is not None:
        base["error_params"] = str(args.error_params)
    profiles: dict[str, dict] = {
        "baseline_a": {**base, "min_holdout_correlation": args.gate_holdout},
        "baseline_b": {**base, "min_holdout_correlation": args.forced_holdout},
    }
    for side in large_sides(args.large_sides):
        profiles[f"large_{side}"] = {
            **base,
            "target_shape": args.large_target_shape,
            "target_side": side,
            # The target box has to sit inside the readout window, with room
            # around it: the 0-order is not exactly at the window centre, so a
            # box that fills the window would get clipped and bias EE / CV.
            "cam_size": window_for_target(side, args.cam_size),
            "min_holdout_correlation": args.forced_holdout,
        }
    for name, profile in profiles.items():
        unknown = set(profile) - PROFILE_FIELDS
        if unknown:
            raise ValueError(
                f"stage {name}: unknown profile fields {', '.join(sorted(unknown))}"
            )
        _assert_window_fits_target(name, profile)
    return profiles


def window_for_target(target_side: int, base_cam_size: int) -> int:
    """Return the camera window that comfortably contains a target box.

    The window must be **larger** than the target's side (the runner measures
    the target in camera pixels, so a window that is not larger truncates the
    box and biases EE / CV), and the margin has to absorb the 0-order's
    off-centre position -- the box is centred on the spot, not on the frame.
    The margin is therefore the larger of 100 px and 10% of the side.
    """
    margin = max(100, int(round(0.10 * target_side)))
    return int(max(base_cam_size, target_side + margin))


def _assert_window_fits_target(name: str, profile: dict) -> None:
    """Fail before any device is opened when the window cannot hold the target."""
    cam_size = int(profile["cam_size"])
    target_side = int(profile["target_side"])
    if cam_size <= target_side:
        raise ValueError(
            f"stage {name}: camera window {cam_size} px must be larger than the "
            f"target side {target_side} px -- a truncated target box biases "
            f"EE and CV"
        )


def stage_kind(name: str) -> str:
    """The expectation class of a stage name."""
    if name == "light":
        return "light"
    if name.startswith("large_"):
        return "large"
    return name


def select_stages(requested: list[str] | None, profiles: dict[str, dict]) -> list[str]:
    """Expand the requested stage names into the ordered list of stage names."""
    names = list(requested) if requested else list(STAGE_ORDER)
    stages: list[str] = []
    for name in names:
        if name == "light":
            stages.append("light")
        elif name == "large":
            stages.extend(key for key in profiles if key.startswith("large_"))
        else:
            stages.append(name)
    return stages


def build_light_command(args: argparse.Namespace, out_root: Path) -> list[str]:
    """The light-level gate command line."""
    return [
        sys.executable,
        str(LIGHT_SCRIPT),
        "--cam-type",
        args.cam_type,
        "--cam-id",
        str(args.cam_id),
        "--slm-number",
        str(args.slm_number),
        "--slm-wavelength",
        str(args.slm_wavelength),
        "--exposures",
        str(args.light_exposures),
        "--out-root",
        str(out_root),
    ]


def with_output_root(command: list[str], root: Path) -> list[str]:
    """Point the runner's ``--dir`` at this stage's own directory.

    ``--dir`` is a group option of ``main.py``, so it has to sit in front of the
    sub-command name; inserting it there keeps every stage's artefacts apart.
    """
    index = command.index(MIL_SUBCOMMAND)
    return command[:index] + ["--dir", str(root)] + command[index:]


def build_plan(args: argparse.Namespace, session_dir: Path) -> dict:
    """Describe the whole campaign without touching a device.

    The shared geometry is part of the plan: the first model-in-loop stage is
    the **donor** (it runs the automatic session calibration and produces
    ``bench_geometry.json``), and every later stage is handed that file. The
    path is fixed up front (``<session>/bench_geometry.json``) so the plan can
    be reviewed before anything runs, and so the manifest can point at one
    stable artefact.
    """
    profiles = build_profiles(args)
    selected = select_stages(args.stage, profiles)
    mil_stages = [name for name in selected if name != "light"]
    donor = mil_stages[0] if mil_stages else None
    shared_geometry = session_dir / SHARED_GEOMETRY_NAME
    stages: list[dict] = []
    for name in selected:
        kind = stage_kind(name)
        if kind == "light":
            stages.append(
                {
                    "name": name,
                    "kind": kind,
                    "expected": EXPECTED_TEXT["light"],
                    "profile": None,
                    "command": build_light_command(args, session_dir / "light"),
                    "gate_command": None,
                    "geometry_file": None,
                    "geometry_donor": False,
                }
            )
            continue
        profile = dict(profiles[name])
        geometry_file: str | None = None
        if args.geometry_file is not None:
            # An externally measured geometry is used by every arm, the donor
            # included: there is nothing left for the donor to measure.
            geometry_file = str(args.geometry_file)
        elif name != donor:
            geometry_file = str(shared_geometry)
        if geometry_file is not None:
            profile["geometry_file"] = geometry_file
        stage_dir = session_dir / name
        stages.append(
            {
                "name": name,
                "kind": kind,
                "expected": EXPECTED_TEXT[kind],
                "profile": profile,
                "command": with_output_root(build_command(profile), stage_dir),
                "gate_command": (
                    None
                    if args.skip_light_gate
                    else build_light_command(args, stage_dir / "light")
                ),
                "geometry_file": geometry_file,
                "geometry_donor": name == donor and args.geometry_file is None,
            }
        )
    return {
        "session_dir": str(session_dir),
        "light_command": build_light_command(args, session_dir / "light"),
        "shared_geometry": str(shared_geometry),
        "geometry_donor": donor,
        "stages": stages,
    }


def render_plan(plan: dict) -> str:
    """Render the campaign plan for a human review."""
    lines = [
        "=== model-in-loop hardware A/B campaign (plan; no device is opened) ===",
        f"session root: {plan['session_dir']}",
        "",
        "light gate command (re-run in front of every model-in-loop stage):",
        f"  {subprocess.list2cmdline(plan['light_command'])}",
        "",
        f"shared geometry: {plan['shared_geometry']}",
        f"geometry donor (runs the automatic session calibration): "
        f"{plan.get('geometry_donor') or '(none: an external --geometry-file was given)'}",
        "",
    ]
    for stage in plan["stages"]:
        lines.append(f"[{stage['name']}] expectation: {stage['expected']}")
        if stage.get("geometry_file"):
            lines.append(f"  geometry file: {stage['geometry_file']}")
        if stage.get("profile") is not None:
            lines.append(
                "  profile: "
                + json.dumps(stage["profile"], indent=2, ensure_ascii=False).replace(
                    "\n", "\n  "
                )
            )
        lines.append(f"  command: {subprocess.list2cmdline(stage['command'])}")
        if stage.get("gate_command"):
            lines.append(
                f"  gate: {subprocess.list2cmdline(stage['gate_command'])}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Reading the evidence (pure)
# ---------------------------------------------------------------------------


def _float(value: object) -> float | None:
    """Parse a number, tolerating empty and non-numeric text."""
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # NaN never counts.


def _text(value: str | None) -> str | None:
    """Normalise an optional captured line."""
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _path(value: str | None) -> str | None:
    """Extract a path from a captured line, dropping a trailing remark."""
    text = _text(value)
    if text is None:
        return None
    if " (" in text:
        text = text.split(" (", 1)[0].strip()
    return text or None


def parse_runner_output(text: str) -> dict:
    """Pull the runner's own summary lines out of a captured stdout.

    The runner prints one aligned block at the end of a run; the labels are
    fixed, so they are matched by name rather than by column.
    """

    def search(pattern: str) -> str | None:
        match = re.search(pattern, text, re.M)
        return _text(match.group("v")) if match else None

    parsed: dict = {}
    parsed["status"] = search(r"^Status\s*:\s*(?P<v>.*)$")
    rounds = search(r"^Rounds accepted\s*:\s*(?P<v>.*)$")
    if rounds:
        match = re.match(r"(\d+)\s*/\s*(\d+)", rounds)
        if match:
            parsed["rounds_accepted"] = int(match.group(1))
            parsed["rounds_total"] = int(match.group(2))
    geometry = search(r"^Geometry\s*:\s*(?P<v>.*)$")
    if geometry:
        correlation = re.search(r"correlation\s+([-+0-9.eE]+)", geometry)
        parsed["geometry_correlation"] = _float(correlation.group(1)) if correlation else None
    parsed["flat_score"] = _float(search(r"^Flat baseline score\s*:\s*(?P<v>.*)$"))
    parsed["best_score"] = _float(search(r"^Best shaped score\s*:\s*(?P<v>.*)$"))
    parsed["history_file"] = _path(search(r"^History saved\s*:\s*(?P<v>.*)$"))
    parsed["best_phase_file"] = _path(search(r"^Best phase saved\s*:\s*(?P<v>.*)$"))
    parsed["best_image_file"] = _path(search(r"^Best image saved\s*:\s*(?P<v>.*)$"))
    parsed["preflight_evidence"] = _path(search(r"^Preflight evidence\s*:\s*(?P<v>.*)$"))
    parsed["abort_reason"] = search(r"^运行中止,\s*设备已释放:\s*(?P<v>.*)$")
    return parsed


def read_history_rows(path: Path) -> list[dict]:
    """Read the history CSV of one run, tolerating missing columns."""
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except (OSError, csv.Error):
        return []


def summarize_history_rows(rows: list[dict]) -> dict:
    """Reduce the history rows to the numbers the campaign judges.

    Columns come and go between runner versions, so every field is read
    defensively and a missing one simply stays ``None``.
    """
    rounds = [row for row in rows if str(row.get("stage", "")).strip() == "round"]
    accepted = [
        row
        for row in rounds
        if str(row.get("accepted", "")).strip().lower() in ("true", "1", "yes")
    ]
    mismatches = [
        row
        for row in rounds
        if str(row.get("reason", "")).strip() == "holdout_forward_mismatch"
    ]
    correlations = [
        value
        for value in (_float(row.get("holdout_correlation")) for row in rounds)
        if value is not None
    ]
    eligible = [
        row
        for row in accepted
        if _float(row.get("score_after")) is not None
    ]
    best = max(eligible, key=lambda row: float(row["score_after"])) if eligible else None
    baseline = next(
        (row for row in rows if str(row.get("stage", "")).strip() == "calibration"), None
    )
    summary = {
        "rounds_total": len(rounds),
        "rounds_accepted": len(accepted),
        "holdout_mismatch_rounds": len(mismatches),
        "holdout_correlation": correlations,
        "score_after": _float(best.get("score_after")) if best else None,
        "score_before": _float(best.get("score_before")) if best else None,
        "cv": _float(best.get("cv_after")) if best else None,
        "ee": _float(best.get("ee_after")) if best else None,
        "iou": _float(best.get("shape_iou_after")) if best else None,
        "flat_score": _float(baseline.get("score_after")) if baseline else None,
    }
    if summary["cv"] is None:
        summary["cv"] = _float(best.get("uniformity_cv")) if best else None
    return summary


def _first(*values: object) -> object:
    """The first value that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def collect_observed(stdout: str, artifacts: dict) -> dict:
    """Merge the runner's summary lines with the recorded history."""
    parsed = parse_runner_output(stdout)
    history = summarize_history_rows(artifacts.get("rows") or [])
    return {
        "status": parsed.get("status"),
        "rounds_total": _first(parsed.get("rounds_total"), history.get("rounds_total")),
        "rounds_accepted": _first(
            parsed.get("rounds_accepted"), history.get("rounds_accepted")
        ),
        "holdout_correlation": history.get("holdout_correlation", []),
        "holdout_mismatch_rounds": history.get("holdout_mismatch_rounds", 0),
        "score_before": _first(parsed.get("score_before"), history.get("score_before")),
        "score_after": _first(parsed.get("score_after"), history.get("score_after")),
        "cv": _first(parsed.get("cv"), history.get("cv")),
        "ee": _first(parsed.get("ee"), history.get("ee")),
        "iou": _first(parsed.get("iou"), history.get("iou")),
        "flat_score": _first(parsed.get("flat_score"), history.get("flat_score")),
        "best_score": parsed.get("best_score"),
        "geometry_correlation": parsed.get("geometry_correlation"),
        "best_phase_file": _first(
            artifacts.get("best_phase_file"), parsed.get("best_phase_file")
        ),
        "best_image_file": _first(
            artifacts.get("best_image_file"), parsed.get("best_image_file")
        ),
        "history_file": _first(artifacts.get("history_file"), parsed.get("history_file")),
        "preflight_evidence": parsed.get("preflight_evidence"),
        "run_dir": artifacts.get("run_dir"),
        "failure_reason": artifacts.get("failure_reason"),
        "abort_reason": parsed.get("abort_reason"),
        "verdict": artifacts.get("verdict"),
        "config": artifacts.get("config"),
        "light_state": artifacts.get("light_state"),
        "light_reasons": artifacts.get("light_reasons", []),
    }


def classify_stage(kind: str, exit_code: int | None, observed: dict) -> dict:
    """Judge one stage against the expectation the campaign recorded for it.

    Pure: the same inputs always give the same verdict, so the rule can be
    checked offline, without a bench.

    Args:
        kind: The expectation class (``light`` / ``baseline_a`` / ``baseline_b``
            / ``large``).
        exit_code: The runner's exit code, ``None`` on a timeout.
        observed: The merged observations of the stage.

    Returns:
        A dict with ``outcome``, ``expected``, ``expectation_met`` and ``notes``.
    """
    expected = EXPECTED_TEXT.get(kind, "")
    notes: list[str] = []

    if kind == "light":
        state = observed.get("light_state")
        if state is None:
            return {
                "outcome": "no_verdict",
                "expected": expected,
                "expectation_met": False,
                "notes": ["the light probe recorded no verdict"],
            }
        if state == "ok":
            return {
                "outcome": "light_ok",
                "expected": expected,
                "expectation_met": True,
                "notes": ["light reaches the camera"],
            }
        return {
            "outcome": f"blocked_{state}",
            "expected": expected,
            "expectation_met": False,
            "notes": [f"light verdict: {state}"],
        }

    failure_reason = observed.get("failure_reason")
    if failure_reason:
        return {
            "outcome": "preflight_rejected",
            "expected": expected,
            "expectation_met": False,
            "notes": [
                f"the geometry bake-off rejected the experiment system: {failure_reason}"
            ],
        }

    status = observed.get("status")
    accepted = observed.get("rounds_accepted") or 0
    mismatches = observed.get("holdout_mismatch_rounds") or 0

    if status is None:
        notes.append(f"the runner exited with code {exit_code} and recorded no summary")
        return {
            "outcome": "failed",
            "expected": expected,
            "expectation_met": False,
            "notes": notes,
        }

    if kind == "baseline_a":
        if status == "completed" and accepted:
            return {
                "outcome": "unexpected_pass",
                "expected": expected,
                "expectation_met": False,
                "notes": [
                    "the holdout gate did not stop the run -- the expectation did not hold"
                ],
            }
        if status == "aborted_unidentifiable" and mismatches:
            return {
                "outcome": "held_out_as_expected",
                "expected": expected,
                "expectation_met": True,
                "notes": [
                    f"{mismatches} round(s) were rejected by the holdout gate; no phase was kept"
                ],
            }
        if status == "bake_off_rejected":
            # The A arm is the *gate* arm: the threshold is not moved, so being
            # stopped by the bake-off is an equally valid "held out" outcome.
            return {
                "outcome": "held_out_as_expected",
                "expected": expected,
                "expectation_met": True,
                "notes": [
                    "the shaped phase lost the bake-off against the flat baseline; "
                    "no phase was kept"
                ],
            }
        notes.append(f"status={status}, holdout rejections={mismatches}")
        return {
            "outcome": "unexpected_outcome",
            "expected": expected,
            "expectation_met": False,
            "notes": notes,
        }

    # The shaping arms (B and the large-area arms) are judged by the REAL
    # criterion: the shaped composite score must be strictly greater than the
    # flat baseline's. A run that completed but lost the bake-off
    # (``bake_off_rejected``) has NOT met the expectation -- the old rule
    # ("completed + an accepted round + a best phase file") counted it as a
    # pass, which is exactly the misjudgement this criterion removes.
    flat_score = observed.get("flat_score")
    best_score = observed.get("best_score")
    beat_flat = (
        flat_score is not None
        and best_score is not None
        and float(best_score) > float(flat_score)
    )
    if status == "completed" and beat_flat and observed.get("best_phase_file"):
        return {
            "outcome": "completed_as_expected",
            "expected": expected,
            "expectation_met": True,
            "notes": [
                f"{accepted} accepted round(s); the shaped score {best_score:.4f} "
                f"beats the flat baseline {flat_score:.4f}"
            ],
        }
    if status == "bake_off_rejected" or (status == "completed" and not beat_flat):
        notes.append(
            f"status={status}, shaped score {best_score} against the flat "
            f"baseline {flat_score} -- shaping did not beat flat"
        )
        return {
            "outcome": "shaping_did_not_beat_flat",
            "expected": expected,
            "expectation_met": False,
            "notes": notes,
        }
    notes.append(
        f"status={status}, accepted rounds={accepted}, "
        f"best phase={'yes' if observed.get('best_phase_file') else 'no'}"
    )
    return {
        "outcome": "unexpected_outcome",
        "expected": expected,
        "expectation_met": False,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Running the campaign
# ---------------------------------------------------------------------------


def utc_now() -> str:
    """The current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    """The short commit of the working tree, or ``None`` outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    return commit or None


def run_command(
    command: list[str], cwd: Path, timeout_s: float | None
) -> tuple[int | None, str, float]:
    """Run one command, merging stdout and stderr into a single text blob."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return None, str(output), time.monotonic() - started
    return completed.returncode, completed.stdout, time.monotonic() - started


def read_light_verdict(out_root: Path) -> dict | None:
    """Read the verdict the light probe wrote under its output root."""
    candidates = sorted(out_root.glob("*/light_level.json"))
    if not candidates:
        return None
    try:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    verdict = payload.get("verdict")
    return verdict if isinstance(verdict, dict) else None


def run_light_gate(
    args: argparse.Namespace, out_root: Path, log_path: Path
) -> dict:
    """Run the light-level gate and classify what it found."""
    command = build_light_command(args, out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    code, output, duration = run_command(command, ROOT, args.timeout_s)
    ended_utc = utc_now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    verdict = read_light_verdict(out_root) or {}
    return {
        "command": command,
        "exit_code": code,
        "duration_s": round(duration, 3),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "output": output,
        "log_file": str(log_path),
        "state": verdict.get("state"),
        "reasons": list(verdict.get("reasons") or []),
    }


def normalize_stage_layout(stage_dir: Path) -> None:
    """Lift the runner's session directory next to the stage log.

    The runner always writes ``<--dir>/debug/<prefix>/<stamp>``; the campaign
    keeps one directory per stage, so the session directory is moved up by one
    level. The ``debug_dir`` recorded inside ``run_config.json`` still names
    the launch path -- it is evidence, not a pointer.
    """
    source_root = stage_dir / "debug" / "slm_model_in_loop"
    if not source_root.is_dir():
        return
    target_root = stage_dir / "slm_model_in_loop"
    target_root.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_root.iterdir()):
        if not child.is_dir():
            continue
        destination = target_root / child.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(child), str(destination))
    for path in (source_root, stage_dir / "debug"):
        try:
            path.rmdir()
        except OSError:
            pass


def collect_artifacts(run_dir: Path | None) -> dict:
    """Read what one run recorded: config, history, failure, best artefacts."""
    artifacts: dict = {
        "run_dir": None,
        "config": None,
        "rows": [],
        "failure_reason": None,
        "best_phase_file": None,
        "best_image_file": None,
        "history_file": None,
        "verdict": None,
    }
    if run_dir is None:
        return artifacts
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return artifacts
    artifacts["run_dir"] = str(run_dir)

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            artifacts["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            artifacts["config"] = None

    failure_path = run_dir / "failure.json"
    if failure_path.exists():
        reason = "preflight rejected"
        try:
            payload = json.loads(failure_path.read_text(encoding="utf-8"))
            reason = str(payload.get("reason") or reason)
        except (OSError, ValueError):
            pass
        artifacts["failure_reason"] = reason

    csv_files = sorted(run_dir.glob("slm_model_in_loop_*.csv"))
    if csv_files:
        artifacts["history_file"] = str(csv_files[0])
        artifacts["rows"] = read_history_rows(csv_files[0])

    for name, key in (
        ("best_phase.npy", "best_phase_file"),
        ("best_far_field.png", "best_image_file"),
    ):
        candidate = run_dir / name
        if candidate.exists():
            artifacts[key] = str(candidate)
    return artifacts


def write_handoff_summary(run_dir: Path | None, args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Summarise the recorded run against the campaign thresholds.

    Returns:
        The verdict (``passes_recorded_gates`` / ``review_required``) and an
        error message when the run cannot be summarised.
    """
    if run_dir is None or not Path(run_dir).is_dir():
        return None, None
    try:
        summary = summarize_session(
            Path(run_dir),
            max_cv=args.max_cv,
            min_ee=args.min_ee,
            min_iou=args.min_iou,
        )
    except (OSError, ValueError) as exc:
        return None, str(exc)
    (Path(run_dir) / "handoff_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary.get("verdict"), None


def find_stage_geometry(run_dir: Path | None) -> Path | None:
    """The geometry file a model-in-loop stage produced, if any.

    The automatic session calibration writes the runbook's artefact under
    ``session_geometry/``; the runner's own saver copies it next to the run.
    Both are looked for, the session one first (it exists even when the run
    later aborts, which is exactly when the geometry is most valuable).

    Args:
        run_dir: The stage's run directory, or ``None``.

    Returns:
        The geometry file's path, or ``None`` when the stage produced none.
    """
    if run_dir is None:
        return None
    root = Path(run_dir)
    for relative in ("session_geometry/bench_geometry.json", "bench_geometry.json"):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def run_mil_stage(
    args: argparse.Namespace, stage: dict, stage_dir: Path
) -> dict:
    """Run one model-in-loop stage and collect everything it recorded."""
    kind = stage["kind"]
    profile = stage["profile"]
    command = stage["command"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    started_utc = utc_now()
    code, output, duration = run_command(command, ROOT, args.timeout_s)
    ended_utc = utc_now()

    log_path = stage_dir / "stage.log"
    log_path.write_text(output, encoding="utf-8")

    parsed_dir = session_dir_from_output(output)
    normalize_stage_layout(stage_dir)
    run_dir: Path | None = None
    if parsed_dir is not None:
        candidate = stage_dir / "slm_model_in_loop" / Path(parsed_dir).name
        run_dir = candidate if candidate.is_dir() else Path(parsed_dir)

    artifacts = collect_artifacts(run_dir)
    verdict, summary_error = write_handoff_summary(run_dir, args)
    artifacts["verdict"] = verdict
    observed = collect_observed(output, artifacts)
    judgment = classify_stage(kind, code, observed)

    notes = list(judgment["notes"])
    if summary_error:
        notes.append(f"the recorded run could not be summarised: {summary_error}")
    if code is None:
        notes.append(f"the stage timed out after {args.timeout_s:g} s")
    if observed.get("abort_reason"):
        notes.append(f"runner: {observed['abort_reason']}")

    return {
        "name": stage["name"],
        "kind": kind,
        "command": command,
        "exit_code": code,
        "duration_s": round(duration, 3),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "expected": judgment["expected"],
        "expectation_met": judgment["expectation_met"],
        "outcome": judgment["outcome"],
        "observed": observed,
        "log_file": str(log_path),
        "notes": notes,
        "geometry_file": stage.get("geometry_file"),
        "geometry_donor": bool(stage.get("geometry_donor")),
        "geometry_fallback": bool(stage.get("geometry_fallback")),
    }


def run_light_stage(args: argparse.Namespace, session_dir: Path) -> dict:
    """Run the light-level gate as a stage of its own."""
    gate = run_light_gate(args, session_dir / "light", session_dir / "light.log")
    observed = collect_observed(
        "",
        {
            "light_state": gate["state"],
            "light_reasons": gate["reasons"],
        },
    )
    judgment = classify_stage("light", gate["exit_code"], observed)
    notes = list(judgment["notes"])
    notes.extend(str(reason) for reason in gate["reasons"])
    return {
        "name": "light",
        "kind": "light",
        "command": gate["command"],
        "exit_code": gate["exit_code"],
        "duration_s": gate["duration_s"],
        "expected": judgment["expected"],
        "expectation_met": judgment["expectation_met"],
        "outcome": judgment["outcome"],
        "observed": observed,
        "log_file": gate["log_file"],
        "started_utc": gate["started_utc"],
        "ended_utc": gate["ended_utc"],
        "notes": notes,
    }


def skipped_light_record(args: argparse.Namespace, session_dir: Path) -> dict:
    """The record of a light gate that was switched off."""
    return {
        "name": "light",
        "kind": "light",
        "command": build_light_command(args, session_dir / "light"),
        "exit_code": None,
        "duration_s": 0.0,
        "started_utc": utc_now(),
        "ended_utc": utc_now(),
        "expected": EXPECTED_TEXT["light"],
        "expectation_met": None,
        "outcome": "skipped",
        "observed": {"light_state": None, "light_reasons": []},
        "log_file": None,
        "notes": ["the light gate was switched off with --skip-light-gate"],
    }


def blocked_stage_record(stage: dict, gate: dict) -> dict:
    """The record of a stage that the light gate kept from running."""
    return {
        "name": stage["name"],
        "kind": stage["kind"],
        "command": stage["command"],
        "exit_code": None,
        "duration_s": 0.0,
        "started_utc": utc_now(),
        "ended_utc": utc_now(),
        "expected": stage["expected"],
        "expectation_met": None,
        "outcome": f"blocked_{gate['state']}",
        "observed": {
            "status": None,
            "light_state": gate["state"],
            "light_reasons": gate["reasons"],
            "run_dir": None,
            "failure_reason": None,
            "verdict": None,
        },
        "log_file": gate["log_file"],
        "notes": [
            f"the light gate reported {gate['state']} -- the stage was not executed",
        ],
        "geometry_file": stage.get("geometry_file"),
        "geometry_donor": bool(stage.get("geometry_donor")),
        "geometry_fallback": bool(stage.get("geometry_fallback")),
    }


def build_manifest(args: argparse.Namespace, session_dir: Path) -> dict:
    """Assemble the campaign manifest."""
    return {
        "session_id": session_dir.name,
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "host": platform.node(),
        "python": platform.python_version(),
        "argv": list(sys.argv),
        "campaign_state": "completed",
        "stages": [],
    }


def observed_numbers(observed: dict) -> str:
    """A compact ``key=value`` summary of what a stage measured."""
    parts: list[str] = []
    keys = [
        "status",
        "rounds_accepted",
        "rounds_total",
        "holdout_mismatch_rounds",
        "cv",
        "ee",
        "iou",
        "flat_score",
        "best_score",
        "geometry_correlation",
        "light_state",
    ]
    if observed.get("status") is None:
        # No run summary: the round counters would be empty noise.
        keys = [key for key in keys if not key.startswith("rounds") and key != "holdout_mismatch_rounds"]
    for key in keys:
        value = observed.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    failure = observed.get("failure_reason")
    if failure:
        parts.append(f"failure={str(failure)[:60]}")
    correlations = observed.get("holdout_correlation") or []
    if correlations:
        parts.append("holdout=" + "/".join(f"{value:.3g}" for value in correlations))
    return ", ".join(parts) if parts else "-"


def render_summary(manifest: dict) -> str:
    """Render the human-readable campaign summary (markdown)."""
    lines = [
        "# model-in-loop hardware A/B campaign",
        "",
        f"- session: `{manifest.get('session_id')}`",
        f"- created (UTC): {manifest.get('created_utc')}",
        f"- campaign state: **{manifest.get('campaign_state')}**",
        f"- git commit: `{manifest.get('git_commit')}`",
        f"- host / python: {manifest.get('host')} / {manifest.get('python')}",
        "",
        "| Stage | Expectation | Observed | Met | Key numbers | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for stage in manifest.get("stages", []):
        observed = stage.get("observed") or {}
        met = stage.get("expectation_met")
        met_text = "yes" if met is True else ("no" if met is False else "n/a")
        evidence = observed.get("run_dir") or stage.get("log_file") or "-"
        lines.append(
            f"| `{stage.get('name')}` | {stage.get('expected')} | "
            f"{stage.get('outcome')} | {met_text} | {observed_numbers(observed)} | "
            f"`{evidence}` |"
        )
    lines.append("")
    notes = [
        (stage.get("name"), stage.get("notes"))
        for stage in manifest.get("stages", [])
        if stage.get("notes")
    ]
    if notes:
        lines.append("## Notes")
        lines.append("")
        for name, stage_notes in notes:
            for note in stage_notes or []:
                lines.append(f"- `{name}`: {note}")
        lines.append("")
    return "\n".join(lines)


def create_session_dir(out_root: Path) -> Path:
    """The timestamped directory that holds one campaign.

    The path is absolute: the runner is launched with ``--dir`` pointing at
    it and reports its session directory back on stdout, so both sides have to
    agree on where the artefacts are.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(out_root).expanduser().resolve() / stamp)


def _load_device_probe_module():
    """Load the shared device-probe helper (the sweep collector's).

    The module **must** be registered in ``sys.modules`` before it executes:
    ``dataclasses`` resolves ``KW_ONLY`` / ``_HAS_DEFAULT_FACTORY`` through
    ``sys.modules[cls.__module__].__dict__``, so a module loaded by path
    without registration makes every dataclass in it blow up with
    ``'NoneType' object has no attribute '__dict__'``.
    """
    spec = importlib.util.spec_from_file_location(
        "run_hw_sweep_groups", DEVICE_PROBE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the device probe from {DEVICE_PROBE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wait_for_device(args: argparse.Namespace) -> str:
    """Block until the SLM and the camera are free.

    The SLM's USB handle is exclusive: whoever opens it first owns it, and the
    other session only gets -10003.  Worse, the collision does not always show
    up at open time -- on 2026-10-22 the campaign instead died with -10032
    ("USB其他错误") in the middle of the panel scan, after the other session had
    grabbed the device.  So probe before touching anything and wait it out.

    Args:
        args: Parsed arguments; ``wait_until_idle`` is the budget in seconds
            (``0`` disables waiting).

    Returns:
        A short record of what happened, for the manifest.

    Raises:
        RuntimeError: If the device is unavailable for a reason that will not
            clear by waiting, or the budget runs out.
    """
    budget = float(getattr(args, "wait_until_idle", 0.0) or 0.0)
    if budget <= 0:
        return "skipped"
    probe = _load_device_probe_module()
    deadline = time.monotonic() + budget
    announced = False
    while True:
        result = probe.probe_devices(
            slm_number=int(args.slm_number),
            cam_id=int(args.cam_id),
            cam_type=str(args.cam_type),
        )
        if result.status == "idle":
            return "idle after waiting" if announced else "idle"
        waitable = result.status == "unavailable" and result.kind in probe.WAITABLE_KINDS
        detail = result.message or result.kind or result.status
        if not waitable:
            raise RuntimeError(f"device unavailable and will not clear by waiting: {detail}")
        if not announced:
            print(f"[device] 等待设备空闲: {detail}", flush=True)
            announced = True
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"device still busy after {budget:g} s: {detail} "
                "(another session holds the SLM's exclusive USB handle)"
            )
        time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))


def main(argv: list[str] | None = None) -> int:
    """Run the campaign (or print its plan)."""
    args = parse_args(argv)
    session_dir = create_session_dir(args.out_root)
    try:
        plan = build_plan(args, session_dir)
    except (ValueError, TypeError) as exc:
        print(f"campaign refused: {exc}", flush=True)
        return 2

    if args.plan:
        print(render_plan(plan), flush=True)
        return 0

    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, session_dir)
    campaign_state = "completed"
    shared_geometry = Path(plan["shared_geometry"])

    # --- 设备互斥: SLM 的 USB 句柄是独占的, 别的会话正拿着时会报 -10003 / -10032
    # (2026-10-22 实测: 抢设备的失败表现为面板扫描写内存时 -10032, 而不是开就报
    # -10003 —— 所以开跑前先探一次, 探到被占就等, 别直接撞上去)。
    try:
        manifest["device_wait"] = wait_for_device(args)
    except RuntimeError as exc:
        print(f"campaign refused: {exc}", flush=True)
        manifest["campaign_state"] = "device_unavailable"
        manifest["notes"] = [str(exc)]
        (session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 2

    for stage in plan["stages"]:
        kind = stage["kind"]
        if kind == "light":
            if args.skip_light_gate:
                manifest["stages"].append(skipped_light_record(args, session_dir))
                continue
            record = run_light_stage(args, session_dir)
            manifest["stages"].append(record)
            state = (record.get("observed") or {}).get("light_state")
            if state == "no_light":
                campaign_state = "blocked_no_light"
                break
            if state == "weak" and not args.allow_weak:
                campaign_state = "blocked_weak"
                break
            continue

        stage_dir = session_dir / stage["name"]
        if not stage.get("geometry_donor") and stage.get("geometry_file"):
            # The shared geometry is only usable once it exists. If the donor
            # produced none, this arm falls back to its own session
            # calibration -- and says so, because two arms that each calibrated
            # on their own are not comparable.
            if not Path(stage["geometry_file"]).is_file():
                profile = dict(stage["profile"])
                profile.pop("geometry_file", None)
                stage["profile"] = profile
                stage["command"] = with_output_root(
                    build_command(profile), stage_dir
                )
                stage["geometry_file"] = None
                stage["geometry_fallback"] = True
        if not args.skip_light_gate:
            gate = run_light_gate(args, stage_dir / "light", stage_dir / "light.log")
            allowed = gate["state"] == "ok" or (
                gate["state"] == "weak" and args.allow_weak
            )
            if not allowed:
                manifest["stages"].append(blocked_stage_record(stage, gate))
                campaign_state = (
                    "blocked_no_light" if gate["state"] == "no_light" else "blocked_weak"
                )
                break

        record = run_mil_stage(args, stage, stage_dir)
        manifest["stages"].append(record)
        print(
            f"[{record['name']}] outcome={record['outcome']} "
            f"met={record['expectation_met']} exit={record['exit_code']}",
            flush=True,
        )
        if stage.get("geometry_donor"):
            # Hand the donor's geometry to the later arms, so they differ from
            # the donor only in the gate under test.
            produced = find_stage_geometry(
                (record.get("observed") or {}).get("run_dir")
            )
            if produced is None:
                record["notes"].append(
                    "the geometry donor produced no bench_geometry.json; the "
                    "later arms fall back to their own session calibration"
                )
            else:
                shutil.copy2(produced, shared_geometry)
                record["notes"].append(
                    f"shared geometry written to {shared_geometry}"
                )
        if record["outcome"] == "preflight_rejected":
            campaign_state = "bench_outside_model"
            break
        if record["outcome"] == "failed":
            campaign_state = "aborted"
            break

    manifest["campaign_state"] = campaign_state
    manifest["shared_geometry"] = str(shared_geometry)
    manifest["shared_geometry_exists"] = shared_geometry.is_file()
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (session_dir / "summary.md").write_text(render_summary(manifest), encoding="utf-8")
    print(f"campaign state: {campaign_state}", flush=True)
    print(f"manifest: {session_dir / 'manifest.json'}", flush=True)
    print(f"summary:  {session_dir / 'summary.md'}", flush=True)

    if campaign_state in ("blocked_no_light", "blocked_weak", "bench_outside_model"):
        return 2
    if campaign_state == "aborted":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
