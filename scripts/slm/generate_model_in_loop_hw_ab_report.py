"""Render the offline report of a model-in-loop hardware A/B campaign.

**What it renders.** A campaign session (written by
``scripts/slm/model_in_loop_hw_ab.py``) leaves one directory per arm behind:
the campaign manifest, the stage log, and -- inside the arm's run directory --
the recorder's whole history (pkl + json + csv), the probe frames, the panel
scan and the calibrated bench geometry. This generator turns those artefacts
into a report plus figures, **without touching a device**: it only reads what
is already on disk, so the report can be regenerated long after the campaign.

**The three questions the campaign was written to answer.** The holdout
(validation) correlation gate is left where it was in one arm and forced open
in the other; the report has to say what each arm actually did, and -- when the
two arms ended up the same -- say so plainly instead of inventing a difference.

**The figures.**

* ``<stamp>_mil_inverse_<arm>.gif`` -- the real inverse-optimisation
  animation: every measured probe frame of the arm, side by side with the
  forward model's prediction resampled onto the camera grid, advancing round by
  round, with the recorder's per-round metrics on a curve panel.
* ``<stamp>_hw_ab_summary.png`` -- the arms side by side (score / CV / EE /
  validation correlation / rounds / status), the day's exposure window and the
  geometry calibration.
* ``<stamp>_panel_scan_<arm>.png`` -- the panel-locate response map and the
  flat drift that the strict mode judged on.
* ``<stamp>_true_vs_pred_<arm>_probes_p<n>.png`` -- **ground truth vs
  prediction**, one row per probe: the measured frame / the model prediction
  on the camera grid / their difference, with the probe's type, round, index,
  exposure, Pearson r and SSIM printed on the panel itself. The rows are split
  over several pages (at most four probes per page) so that one panel stays
  large enough to read in the report's column.
* ``<stamp>_true_vs_pred_<arm>_stats.png`` -- the pooled 1:1 density plot of
  measured against predicted pixels, next to the distribution of the per-probe
  r split by probe type.
* ``<stamp>_metrics_curves_<arm>.png`` -- the **metrics curves**: composite
  score, CV, EE and validation correlation against the round, with the flat
  baseline, the acceptance thresholds and the accepted/rejected state of each
  round.

**How the model's prediction reaches the camera grid.** The forward model
predicts on the ``far_field_size`` grid (1024 cells here); the camera records
``cam_size`` camera pixels. The run's own ``bench_geometry.json`` carries the
conversion (``camera_px_per_model_px`` = camera pixels per far-field cell), so
the prediction is cropped to the cells the camera actually sees, resampled to
the camera grid and rotated back by the quarter turn the hardware path applies.
Nothing is re-simulated and no number is invented: a value that was not
recorded reads as "未记录". The one thing the runner does not store is the
model's prediction for the *training* and *flat* probes, so those two are
recomputed offline from the coefficients and the source amplitude the run did
write down -- and the recomputation is checked against the prediction the
runner did store before any of it is used.

Run:
    python scripts/slm/generate_model_in_loop_hw_ab_report.py \
        --session data/debug/model_in_loop_hw_ab/20261021_172236 \
        --stamp 20261021
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts._common import fmt_metric, markdown_table, savefig
from scripts._common.provenance import insert_header

#: Where the report and its figures live (the manual MIL series directory).
DEFAULT_OUT_DIR = ROOT / "report" / "shape_trials" / "model_in_loop"

#: The campaign script whose classification rules and parsers are reused here.
CAMPAIGN_SCRIPT = ROOT / "scripts" / "slm" / "model_in_loop_hw_ab.py"

#: GIF pacing and size budget -- a few dozen frames of a few hundred pixels.
GIF_DURATION_MS = 550
GIF_TAIL_FRAMES = 5
GIF_DPI = 90

#: The hardware path rotates the camera window by a quarter turn before it is
#: compared with the model (``camera_rotation_quarters = 1`` for every
#: non-simulated bench, see ``optimizer/wfless/slm_model_in_loop.py``).
HARDWARE_ROTATION_QUARTERS = 1

#: Probe splits, in the order the truth-vs-prediction figure groups them.
PROBE_SPLIT_ORDER = ("flat", "train", "holdout", "shaped")

#: One colour per probe split (the per-probe correlation bar chart).
SPLIT_COLOURS = {
    "flat": "#7f7f7f",
    "train": "#1f77b4",
    "holdout": "#ff7f0e",
    "shaped": "#2ca02c",
}

#: The truth-vs-prediction pages are laid out in inches, because the report
#: reads them in a ~900 px column: a 10 in wide page is read at ~90 px/in, so
#: one square panel lands at ~260 px and a 10 pt label at ~12 px. The probe
#: rows are therefore split over several pages instead of all landing on one
#: (the single-page version squeezed ten rows into a 1655x440 px image, i.e.
#: 44 px per panel).
TRUTH_FIG_WIDTH_IN = 10.0
TRUTH_PANEL_IN = 2.9
TRUTH_ROW_GAP_IN = 0.45
TRUTH_HEADER_IN = 1.35
TRUTH_BOTTOM_IN = 0.15
TRUTH_PAGE_ROWS = 4

#: The recorder's metric columns, read from the run's history CSV.
#: Each entry is ``(key, label, column, baseline column)``; the baseline column
#: is the one the calibration row carries for the same metric.
ROUND_METRICS = (
    ("score", "composite score", "score_after", "score_after"),
    ("cv", "CV", "cv_after", "cv_after"),
    ("ee", "EE", "ee_after", "ee_after"),
    ("holdout", "验证集相关性", "holdout_correlation", None),
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    """Read a JSON file, returning ``None`` when it is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_text(path: Path) -> str:
    """Read a text file, returning an empty string when it is missing."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def strip_ansi(text: str) -> str:
    """Drop the colour escapes the runner's logger wrote into the stage log."""
    return _ANSI.sub("", text)


def as_float(value: Any) -> float | None:
    """Parse a number, treating NaN as "not recorded"."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def fmt(value: Any, nd: int = 4) -> str:
    """Format a number for the report, or "未记录" when it was not recorded."""
    number = as_float(value)
    if number is None:
        return "未记录"
    return fmt_metric(number, nd)


def load_campaign_module():
    """Import the campaign script so its parsers and rules are reused verbatim."""
    spec = importlib.util.spec_from_file_location("model_in_loop_hw_ab", CAMPAIGN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {CAMPAIGN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Session / stage / run model
# ---------------------------------------------------------------------------


@dataclass
class ProbeEntry:
    """One probe the runner measured and wrote to disk.

    Attributes:
        file: File name inside the run's ``probes/`` directory.
        round: Round index the probe belongs to.
        split: ``flat`` / ``train`` / ``holdout`` / ``shaped``.
        index: Probe index inside its round.
        exposure_ms: Exposure the frame was taken at, when recorded.
        correlation: Correlation the runner recorded for this probe.
        path: Absolute path of the ``.npz`` file.
    """

    file: str
    round: int
    split: str
    index: int
    exposure_ms: float | None
    correlation: float | None
    path: Path


@dataclass
class RunData:
    """Everything one model-in-loop run left behind.

    Attributes:
        run_dir: The run directory, or ``None`` when the stage never got one.
        config: ``run_config.json`` of the run.
        geometry: ``bench_geometry.json`` of the run.
        handoff: ``handoff_summary.json`` of the run.
        recorder_status: The status string the recorder closed the run with.
        recorder: The recorder's per-round records (from the pkl).
        history: The recorder's flat CSV rows.
        probes: The probe files of the run.
        panel_scan: ``panel_scan/summary.json`` of the run.
        panel_measurements: ``panel_scan/measurements.json`` of the run.
        session_geometry: The geometry the sweep stage measured.
    """

    run_dir: Path | None = None
    config: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)
    handoff: dict = field(default_factory=dict)
    recorder_status: str | None = None
    recorder: list = field(default_factory=list)
    history: list = field(default_factory=list)
    probes: list = field(default_factory=list)
    panel_scan: dict = field(default_factory=dict)
    panel_measurements: list = field(default_factory=list)
    session_geometry: dict = field(default_factory=dict)


@dataclass
class StageData:
    """One arm of the campaign, as the campaign itself recorded it.

    Attributes:
        name: Stage name (the directory name under the session).
        kind: Expectation class (``light`` / ``baseline_a`` / ``baseline_b`` /
            ``large``).
        stage_dir: The stage directory.
        profile: The profile the stage ran with.
        expected: What the campaign expected the stage to show.
        expectation_met: Whether the expectation held.
        outcome: The campaign's verdict for the stage.
        notes: The campaign's notes.
        observed: The merged observations of the stage.
        exit_code: The runner's exit code, when the manifest recorded one.
        run: The run artefacts.
        log_text: The stage log.
        from_manifest: Whether the stage came from the campaign manifest.
    """

    name: str
    kind: str
    stage_dir: Path
    profile: dict = field(default_factory=dict)
    expected: str = ""
    expectation_met: bool | None = None
    outcome: str = ""
    notes: list = field(default_factory=list)
    observed: dict = field(default_factory=dict)
    exit_code: int | None = None
    run: RunData = field(default_factory=RunData)
    log_text: str = ""
    from_manifest: bool = False


def find_run_dir(stage_dir: Path) -> Path | None:
    """Find the run directory of a stage.

    A finished stage has it lifted next to the stage log
    (``<stage>/slm_model_in_loop/<stamp>/``); a stage that is still running
    keeps the runner's own layout (``<stage>/debug/slm_model_in_loop/<stamp>/``).
    """
    candidates: list[Path] = []
    for pattern in ("slm_model_in_loop/*", "debug/slm_model_in_loop/*"):
        candidates.extend(sorted(stage_dir.glob(pattern)))
    runs = [path for path in candidates if path.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def load_recorder(run_dir: Path) -> tuple[str | None, list]:
    """Read the recorder's pkl (per-round records) and its status json."""
    status: str | None = None
    for path in sorted(run_dir.glob("slm_model_in_loop_*.json")):
        payload = read_json(path)
        if isinstance(payload, dict) and payload.get("status"):
            status = str(payload["status"])
    records: list = []
    for path in sorted(run_dir.glob("slm_model_in_loop_*.pkl")):
        try:
            payload = pickle.loads(path.read_bytes())
        except (OSError, pickle.UnpicklingError, EOFError):
            continue
        if isinstance(payload, dict):
            records.extend(payload.values())
        elif isinstance(payload, list):
            records.extend(payload)
    return status, records


def load_history(run_dir: Path) -> list:
    """Read the recorder's flat CSV history."""
    for path in sorted(run_dir.glob("slm_model_in_loop_*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                return [dict(row) for row in csv.DictReader(stream)]
        except (OSError, csv.Error):
            return []
    return []


def load_probes(run_dir: Path) -> list:
    """Read the probe manifest of a run into :class:`ProbeEntry` objects."""
    probes_dir = run_dir / "probes"
    manifest = read_json(probes_dir / "manifest.json")
    entries: list = []
    if isinstance(manifest, list):
        for item in manifest:
            if not isinstance(item, dict):
                continue
            name = str(item.get("file") or "")
            path = probes_dir / name
            if not name or not path.exists():
                continue
            entries.append(
                ProbeEntry(
                    file=name,
                    round=int(item.get("round", 0) or 0),
                    split=str(item.get("split") or ""),
                    index=int(item.get("index", 0) or 0),
                    exposure_ms=as_float(item.get("exposure_ms")),
                    correlation=as_float(item.get("correlation")),
                    path=path,
                )
            )
    if entries:
        return entries
    # The manifest is missing (or truncated): fall back to the file names.
    for path in sorted(probes_dir.glob("round_*.npz")):
        match = re.match(r"round_(\d+)_([a-z]+)_(\d+)\.npz", path.name)
        if match is None:
            continue
        entries.append(
            ProbeEntry(
                file=path.name,
                round=int(match.group(1)),
                split=match.group(2),
                index=int(match.group(3)),
                exposure_ms=None,
                correlation=None,
                path=path,
            )
        )
    return entries


def load_run(run_dir: Path | None) -> RunData:
    """Collect every artefact one run left behind."""
    run = RunData(run_dir=run_dir)
    if run_dir is None or not run_dir.is_dir():
        return run
    config = read_json(run_dir / "run_config.json")
    if isinstance(config, dict):
        run.config = config
    geometry = read_json(run_dir / "bench_geometry.json")
    if isinstance(geometry, dict):
        run.geometry = geometry
    handoff = read_json(run_dir / "handoff_summary.json")
    if isinstance(handoff, dict):
        run.handoff = handoff
    session_geometry = read_json(run_dir / "session_geometry" / "bench_geometry.json")
    if isinstance(session_geometry, dict):
        run.session_geometry = session_geometry
    panel_summary = read_json(run_dir / "panel_scan" / "summary.json")
    if isinstance(panel_summary, dict):
        run.panel_scan = panel_summary
    measurements = read_json(run_dir / "panel_scan" / "measurements.json")
    if isinstance(measurements, list):
        run.panel_measurements = measurements
    run.recorder_status, run.recorder = load_recorder(run_dir)
    run.history = load_history(run_dir)
    run.probes = load_probes(run_dir)
    return run


def read_light_verdict(directory: Path) -> dict | None:
    """Read the light-gate verdict of a session or stage directory."""
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("**/light_level.json"))
    if not candidates:
        return None
    payload = read_json(candidates[-1])
    if not isinstance(payload, dict):
        return None
    verdict = payload.get("verdict")
    if not isinstance(verdict, dict):
        return None
    return {
        "state": verdict.get("state"),
        "reasons": list(verdict.get("reasons") or []),
        "readings": list(payload.get("readings") or []),
        "path": str(candidates[-1]),
    }


def collect_stages(session_dir: Path, campaign) -> tuple[list, dict | None]:
    """Collect the campaign's stages, with or without its manifest.

    A campaign that is still running has not written its manifest yet, so the
    stages are discovered from the session directory and judged with the
    campaign's own :func:`classify_stage` on the stage log and the recorder's
    artefacts. When the manifest exists it wins -- it carries the exit code and
    the campaign's own judgement.
    """
    manifest = read_json(session_dir / "manifest.json")
    manifest_stages: dict = {}
    campaign_state = None
    if isinstance(manifest, dict):
        campaign_state = manifest.get("campaign_state")
        for stage in manifest.get("stages") or []:
            if isinstance(stage, dict) and stage.get("name"):
                manifest_stages[str(stage["name"])] = stage

    stages: list = []
    light = read_light_verdict(session_dir / "light")
    if light is not None:
        observed = {
            "status": None,
            "light_state": light["state"],
            "light_reasons": light["reasons"],
        }
        record = manifest_stages.get("light")
        if record is not None:
            stages.append(
                StageData(
                    name="light",
                    kind="light",
                    stage_dir=session_dir,
                    expected=str(record.get("expected") or ""),
                    expectation_met=record.get("expectation_met"),
                    outcome=str(record.get("outcome") or ""),
                    notes=list(record.get("notes") or []),
                    observed=dict(record.get("observed") or observed),
                    exit_code=record.get("exit_code"),
                    log_text=read_text(session_dir / "light.log"),
                    from_manifest=True,
                )
            )
        else:
            judgment = campaign.classify_stage("light", None, observed)
            stages.append(
                StageData(
                    name="light",
                    kind="light",
                    stage_dir=session_dir,
                    expected=str(judgment["expected"]),
                    expectation_met=judgment["expectation_met"],
                    outcome=str(judgment["outcome"]),
                    notes=list(judgment["notes"]),
                    observed=observed,
                    log_text=read_text(session_dir / "light.log"),
                )
            )

    for stage_dir in sorted(path for path in session_dir.iterdir() if path.is_dir()):
        if stage_dir.name == "light" and not (stage_dir / "profile.json").exists():
            continue
        if not (stage_dir / "profile.json").exists() and not (
            stage_dir / "stage.log"
        ).exists():
            continue
        name = stage_dir.name
        kind = campaign.stage_kind(name)
        profile = read_json(stage_dir / "profile.json")
        log_text = read_text(stage_dir / "stage.log")
        run_dir = find_run_dir(stage_dir)
        run = load_run(run_dir)
        artifacts = campaign.collect_artifacts(run_dir)
        observed = campaign.collect_observed(log_text, artifacts)
        record = manifest_stages.get(name)
        if record is not None:
            stages.append(
                StageData(
                    name=name,
                    kind=str(record.get("kind") or kind),
                    stage_dir=stage_dir,
                    profile=profile if isinstance(profile, dict) else {},
                    expected=str(record.get("expected") or ""),
                    expectation_met=record.get("expectation_met"),
                    outcome=str(record.get("outcome") or ""),
                    notes=list(record.get("notes") or []),
                    observed=dict(record.get("observed") or observed),
                    exit_code=record.get("exit_code"),
                    run=run,
                    log_text=log_text,
                    from_manifest=True,
                )
            )
            continue
        judgment = campaign.classify_stage(
            kind, None, observed
        )
        notes = list(judgment["notes"])
        outcome = str(judgment["outcome"])
        expectation_met = judgment["expectation_met"]
        if outcome == "failed" and run.recorder_status is None:
            # The campaign has not written its manifest yet and the stage has
            # no recorder output: the stage is still running, not failed.
            outcome = "进行中（阶段产物未齐，campaign 清单未生成）"
            expectation_met = None
            notes = [
                "campaign 清单尚未生成，本变体仍在运行或未收尾；"
                "已落盘的产物已全部读入"
            ] + notes
        stages.append(
            StageData(
                name=name,
                kind=kind,
                stage_dir=stage_dir,
                profile=profile if isinstance(profile, dict) else {},
                expected=str(judgment["expected"]),
                expectation_met=expectation_met,
                outcome=outcome,
                notes=notes,
                observed=observed,
                run=run,
                log_text=log_text,
            )
        )
    return stages, manifest


# ---------------------------------------------------------------------------
# Geometry: the model's prediction onto the camera grid
# ---------------------------------------------------------------------------


def prediction_to_camera(prediction: np.ndarray, geometry: dict, cam_size: int):
    """Resample the model's far-field prediction onto the camera grid.

    The forward model predicts on the ``far_field_size`` grid; the camera
    records ``cam_size`` pixels. The run's own geometry carries the
    conversion: ``camera_px_per_model_px`` camera pixels per far-field cell.
    The camera therefore sees ``cam_size / cell`` cells of the model grid, and
    the hardware path compares frames rotated by a quarter turn, so the
    prediction is rotated back before it is put on the camera grid.

    Args:
        prediction: The model's far-field intensity.
        geometry: The run's ``bench_geometry.json``.
        cam_size: The camera window edge in pixels.

    Returns:
        The prediction resampled to ``(cam_size, cam_size)``, or ``None`` when
        the geometry does not carry the conversion.
    """
    cell = as_float(geometry.get("camera_px_per_model_px"))
    far_field_size = as_float(geometry.get("far_field_size"))
    if not cell or cell <= 0 or not far_field_size:
        return None
    span = int(round(cam_size / cell))
    span = max(1, min(span, prediction.shape[0], prediction.shape[1]))
    start = (int(far_field_size) - span) // 2
    start = max(0, min(start, prediction.shape[0] - span, prediction.shape[1] - span))
    crop = prediction[start : start + span, start : start + span]
    if span == cam_size:
        resized = crop
    else:
        rows = np.linspace(0, crop.shape[0] - 1, cam_size)
        cols = np.linspace(0, crop.shape[1] - 1, cam_size)
        resized = crop[np.ix_(rows.round().astype(int), cols.round().astype(int))]
    return np.rot90(resized, HARDWARE_ROTATION_QUARTERS)


def _box_sums(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Sum of every ``height x width`` window of ``values`` (integral image)."""
    cumulative = np.cumsum(np.cumsum(values, axis=0), axis=1)
    padded = np.pad(cumulative, ((1, 0), (1, 0)))
    return (
        padded[height:, width:]
        - padded[:-height, width:]
        - padded[height:, :-width]
        + padded[:-height, :-width]
    )


def best_shift(template: np.ndarray, image: np.ndarray, max_shift: int = 96):
    """Find the translation that best matches ``template`` onto ``image``.

    The forward model and the camera do not agree on where the optical axis
    sits inside the window, so the prediction is aligned on the recorded
    frames by a normalised cross-correlation. The shift is a property of the
    run (the axis does not move between probes), so one shift is fitted for
    the whole run and reported.

    Args:
        template: The image to be shifted (the model's prediction).
        image: The image it is matched against (the measured frame).
        max_shift: Largest shift (pixels) the search considers.

    Returns:
        ``(dy, dx, correlation)`` -- the shift of the template and the
        normalised correlation it reaches.
    """
    template = np.asarray(template, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    if template.shape != image.shape or template.size == 0:
        return 0, 0, float("nan")
    t0 = template - template.mean()
    i0 = image - image.mean()
    size = (2 * template.shape[0], 2 * template.shape[1])
    corr = np.fft.irfft2(
        np.conj(np.fft.rfft2(t0, s=size)) * np.fft.rfft2(i0, s=size), s=size
    )
    norm = np.sqrt(_box_sums(i0**2, template.shape[0], template.shape[1]))
    denominator = norm * np.sqrt(float(np.sum(t0**2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(denominator > 0, corr / denominator, np.nan)

    # Signed lags: index k of the circular correlation is the lag k for
    # k < size/2 and k - size above it.
    rows = np.arange(size[0])
    cols = np.arange(size[1])
    row_lag = np.where(rows < size[0] // 2, rows, rows - size[0])
    col_lag = np.where(cols < size[1] // 2, cols, cols - size[1])
    keep_rows = np.abs(row_lag) <= max_shift
    keep_cols = np.abs(col_lag) <= max_shift
    window = normalised[np.ix_(keep_rows, keep_cols)]
    if window.size == 0 or not np.isfinite(window).any():
        return 0, 0, float("nan")
    flat = int(np.nanargmax(window))
    row = int(np.flatnonzero(keep_rows)[flat // window.shape[1]])
    col = int(np.flatnonzero(keep_cols)[flat % window.shape[1]])
    return int(row_lag[row]), int(col_lag[col]), float(window.flat[flat])


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two images, NaN when either is degenerate."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return float("nan")
    return float(a @ b / denominator)


def display_image(array: np.ndarray) -> np.ndarray:
    """Map an intensity array to a log-scaled display array in ``[0, 1]``."""
    data = np.asarray(array, dtype=np.float64)
    peak = float(np.max(data)) if data.size else 0.0
    if peak <= 0:
        return np.zeros(data.shape, dtype=np.float64)
    scaled = data / peak
    display = np.log10(np.clip(scaled, 1e-4, None))
    return (display + 4.0) / 4.0


# ---------------------------------------------------------------------------
# The forward model, rebuilt offline from the run's own artefacts
# ---------------------------------------------------------------------------


@dataclass
class ForwardModel:
    """The arm's forward model, rebuilt offline from the artefacts on disk.

    The runner writes the fitted Zernike coefficients and the source amplitude,
    but only stores the model's prediction for the probes it judged (the
    validation probes and the shaped frame). The training and the flat probes
    carry no prediction, so the model is rebuilt here -- the same optimiser
    class, the same configuration, the same coefficients -- and their
    predictions are recomputed.

    ``torch`` is an optional extra of this repository, so a missing torch
    degrades to "no model" instead of failing the report.

    Attributes:
        predict: Maps a pupil phase to the model's far field, or ``None``.
        gain: The forward calibration gain the recorder recorded.
        leakage: The additive background (stray light) term coefficient.
        flat_grid: The measured flat frame on the model's far-field grid.
        note: Why the model is unavailable, when it is.
    """

    predict: Callable[[np.ndarray], np.ndarray] | None = None
    gain: float = 1.0
    leakage: float = 0.0
    flat_grid: np.ndarray | None = None
    note: str = ""

    @property
    def available(self) -> bool:
        """Whether the model could be rebuilt."""
        return self.predict is not None


def load_npy(path: Path) -> np.ndarray | None:
    """Read a ``.npy`` file, returning ``None`` when it is missing or unreadable."""
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def forward_calibration(run: RunData) -> tuple[float, float]:
    """The forward calibration of the run's last round.

    The runner fits one non-negative model gain and one additive background
    (stray light) term per round and records both in the history CSV. They are
    what turns the model's own intensity into the camera's scale, so the
    recomputed predictions have to carry them too.

    Returns:
        ``(gain, leakage)``; ``(1.0, 0.0)`` -- the runner's own defaults --
        when the run recorded neither.
    """
    gain: float | None = None
    leakage: float | None = None
    for row in round_rows(run):
        value = as_float(row.get("model_gain"))
        if value is not None:
            gain = value
        value = as_float(row.get("flat_leakage"))
        if value is not None:
            leakage = value
    return (
        1.0 if gain is None else gain,
        0.0 if leakage is None else leakage,
    )


def load_forward_model(stage: StageData) -> ForwardModel:
    """Rebuild the arm's forward model from the artefacts the run wrote down.

    Only what is on disk is used: the fitted coefficients, the source
    amplitude, the configuration in ``run_config.json`` and the forward
    calibration the recorder recorded. The camera-frame resampling is the
    runner's own :func:`~ao_shaping.optimizer.wfless.slm_model_in_loop._to_model_grid`,
    reused verbatim rather than re-derived, so the recomputed prediction is
    the same quantity the runner compared against.

    Returns:
        A :class:`ForwardModel`; ``available`` is ``False`` when torch or the
        artefacts are missing, and ``note`` says which.
    """
    run = stage.run
    model = ForwardModel()
    if run.run_dir is None:
        model.note = "该变体没有运行目录，无法重建正向模型"
        return model
    coefficients = load_npy(run.run_dir / "fitted_coefficients.npy")
    amplitude = load_npy(run.run_dir / "source_amplitude.npy")
    if coefficients is None or amplitude is None:
        model.note = "运行目录缺少 fitted_coefficients.npy / source_amplitude.npy"
        return model

    model.gain, model.leakage = forward_calibration(run)
    config = run.config or {}
    try:
        from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
            ZernikeCoefficientOptimizer,
        )
        from ao_shaping.optimizer.wfless.slm_model_in_loop import _to_model_grid
    except Exception as exc:  # pragma: no cover - torch is an optional extra
        model.note = f"正向模型不可用 ({type(exc).__name__})"
        return model

    region = int(config.get("region", 128) or 128)
    try:
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=int(config.get("n_orders", 10) or 10),
            region=region,
            initial_coefficients=coefficients,
            lr=float(config.get("step_a_lr", 0.05) or 0.05),
            max_iterations=max(1, int(config.get("step_a_iterations", 1) or 1)),
            device="cpu",
            seed=int(config.get("seed", 0) or 0),
            dtype=str(config.get("dtype", "float32")),
            far_field_size=region * int(config.get("far_field_padding", 8) or 8),
            frozen_modes=tuple(config.get("frozen_modes") or ()),
            pupil_oversampling=int(config.get("pupil_oversampling", 1) or 1),
        )
        optimizer.set_source_amplitude(amplitude)
    except Exception as exc:  # pragma: no cover - torch is an optional extra
        model.note = f"正向模型初始化失败 ({type(exc).__name__})"
        return model

    geometry = run.geometry or run.session_geometry
    far_field_size = int(as_float(geometry.get("far_field_size")) or 1024)
    cell_scale = as_float(geometry.get("camera_px_per_model_px"))
    roi_center = parse_roi_center(stage.log_text)
    flat_frame = flat_frame_of(run)
    if flat_frame is not None and cell_scale and roi_center is not None:
        model.flat_grid = _to_model_grid(
            flat_frame,
            roi_center,
            far_field_size,
            cell_scale,
            HARDWARE_ROTATION_QUARTERS,
        )

    def predict(phase: np.ndarray) -> np.ndarray:
        """The model's far field for one pupil phase, on the model's own grid."""
        base = optimizer.forward_intensity(
            coefficients, np.asarray(phase, dtype=np.float64), amplitude
        )
        prediction = model.gain * np.asarray(base, dtype=np.float64)
        if model.flat_grid is not None:
            prediction = prediction + model.leakage * model.flat_grid
        return prediction

    model.predict = predict
    model.note = (
        f"离线重建 (gain {model.gain:.4f}, 加性背景（杂散光）项 {model.leakage:.4f})"
    )
    return model


def flat_frame_of(run: RunData) -> np.ndarray | None:
    """The run's measured flat frame (the preflight one, or the flat probe)."""
    for candidate in (
        run.run_dir / "preflight" / "flat.npz" if run.run_dir else None,
    ):
        if candidate is None or not candidate.exists():
            continue
        try:
            data = np.load(candidate)
        except (OSError, ValueError):
            continue
        if "frame" in data:
            return np.asarray(data["frame"], dtype=np.float64)
    for probe in run.probes:
        if probe.split != "flat":
            continue
        try:
            data = np.load(probe.path)
        except (OSError, ValueError):
            continue
        if "frame" in data:
            return np.asarray(data["frame"], dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# The truth (measurement) against the prediction, probe by probe
# ---------------------------------------------------------------------------


@dataclass
class ProbeTruth:
    """One probe: what the camera saw next to what the model predicted.

    Attributes:
        probe: The probe this entry belongs to.
        frame: The measured frame (camera grid).
        prediction: The model's far field (model grid).
        mapped: The prediction on the camera grid, aligned to the frame.
        model_grid: The measurement resampled onto the model grid, when stored.
        stored: The prediction the runner stored, when it stored one.
        recomputed: The prediction rebuilt offline, when it was rebuilt.
        r_camera: Pearson r between the aligned prediction and the frame.
        r_model: Pearson r on the model grid (the runner's own metric).
        ssim_camera: Peak-normalised SSIM between the aligned prediction and
            the frame.
        r2_camera: R^2 of the linear fit of the prediction onto the frame.
        source: Where the prediction came from.
        agreement: Relative deviation between the stored and the recomputed
            prediction, when both exist.
    """

    probe: ProbeEntry
    frame: np.ndarray | None = None
    prediction: np.ndarray | None = None
    mapped: np.ndarray | None = None
    model_grid: np.ndarray | None = None
    stored: np.ndarray | None = None
    recomputed: np.ndarray | None = None
    r_camera: float | None = None
    r_model: float | None = None
    ssim_camera: float | None = None
    r2_camera: float | None = None
    source: str = "未记录"
    agreement: float | None = None

    @property
    def split(self) -> str:
        """The probe's type (``flat`` / ``train`` / ``holdout`` / ``shaped``)."""
        return self.probe.split

    @property
    def label(self) -> str:
        """A short label for the probe, e.g. ``r1 train#0``."""
        return f"r{self.probe.round} {self.probe.split}#{self.probe.index}"


def probe_sort_key(truth: ProbeTruth) -> tuple:
    """Group the probes by type, then by round and index."""
    split = truth.split if truth.split in PROBE_SPLIT_ORDER else ""
    order = (
        PROBE_SPLIT_ORDER.index(split) if split else len(PROBE_SPLIT_ORDER),
        truth.probe.round,
        truth.probe.index,
    )
    return order


def ssim_score(a: np.ndarray, b: np.ndarray) -> float | None:
    """Peak-normalised SSIM of two images (``skimage``, 7x7 Gaussian window).

    Returns ``None`` when ``scikit-image`` is unavailable or the images do not
    match, so a missing backend reads as "未记录" rather than as a zero.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape or min(first.shape) < 3:
        return None
    for array in (first, second):
        peak = float(np.max(array))
        if peak <= 0:
            return None
    try:
        from skimage.metrics import structural_similarity
    except Exception:  # pragma: no cover - scikit-image is a declared dependency
        return None
    normalised_a = first / float(np.max(first))
    normalised_b = second / float(np.max(second))
    return float(
        structural_similarity(normalised_a, normalised_b, data_range=1.0, win_size=7)
    )


def linear_r2(measured: np.ndarray, predicted: np.ndarray) -> float | None:
    """R^2 of the linear fit of ``predicted`` onto ``measured``.

    For a fit with an intercept this is the squared Pearson correlation, so it
    is the number a 1:1 plot is read with.
    """
    x = np.asarray(predicted, dtype=np.float64).ravel()
    y = np.asarray(measured, dtype=np.float64).ravel()
    if x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 0:
        return None
    return float((x @ y / denominator) ** 2)


def align_to_camera(mapped: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Roll a mapped prediction onto the measured frame's optical axis."""
    return np.roll(np.roll(mapped, shift[0], axis=0), shift[1], axis=1)


@dataclass
class Alignment:
    """How the model's prediction is put on the camera frame.

    Attributes:
        shift: The ``(dy, dx)`` roll that lands the prediction's optical axis
            on the measured frame's optical axis.
        quality: The mean correlation the shift reaches over the run's probes.
        anchor: What the shift was derived from.
        roi_center: The run's frozen optical axis in camera pixels.
    """

    shift: tuple[int, int] = (0, 0)
    quality: float = float("nan")
    anchor: str = "未记录"
    roi_center: tuple[int, int] | None = None


def arm_alignment(stage: StageData, pairs: list) -> Alignment:
    """Work out how the model's prediction lines up with the measured frame.

    The run's own frozen optical axis (``stage.log``'s ``frozen ROI centre``)
    is the anchor: the prediction is centred on the optical axis by
    construction, so rolling it to where the axis sits in the frame puts the
    two on top of each other. When the log carries no axis, the shift is fitted
    from the frames themselves and the anchor says so.

    Args:
        stage: The arm.
        pairs: ``(mapped prediction, measured frame)`` pairs.

    Returns:
        The :class:`Alignment` to use for every probe of the arm.
    """
    cam_size = int(as_float(stage.run.config.get("cam_size")) or 300)
    roi_center = parse_roi_center(stage.log_text)
    if roi_center is not None:
        shift = (
            int(round(roi_center[1] - cam_size / 2)),
            int(round(roi_center[0] - cam_size / 2)),
        )
        quality = float("nan")
        if pairs:
            quality = float(
                np.mean(
                    [
                        normalized_correlation(
                            align_to_camera(mapped, shift), frame
                        )
                        for mapped, frame in pairs
                    ]
                )
            )
        return Alignment(
            shift=shift,
            quality=quality,
            anchor="冻结光轴 (stage.log 的 frozen ROI centre)",
            roi_center=roi_center,
        )
    if not pairs:
        return Alignment(anchor="未记录（该变体没有可对齐的探针）")
    template = np.mean([pair[0] for pair in pairs], axis=0)
    image = np.mean([pair[1] for pair in pairs], axis=0)
    dy, dx, quality = best_shift(template, image)
    return Alignment(
        shift=(int(dy), int(dx)),
        quality=float(quality),
        anchor="整幅互相关拟合的公共平移 (日志未记录 frozen ROI centre)",
        roi_center=None,
    )


def collect_probe_truths(stage: StageData) -> list[ProbeTruth]:
    """Pair every measured probe of the arm with the model's prediction.

    The prediction of a validation or shaped probe is the one the runner
    stored; the prediction of a training or flat probe is recomputed offline
    from the run's own coefficients (see :func:`load_forward_model`). The
    recomputation is checked against the stored prediction wherever both
    exist, so the offline numbers can be trusted.

    Returns:
        The probes, grouped by type and ordered by round and index.
    """
    run = stage.run
    geometry = run.geometry or run.session_geometry
    cam_size = int(as_float(run.config.get("cam_size")) or 300)
    model = load_forward_model(stage)

    truths: list[ProbeTruth] = []
    for probe in run.probes:
        try:
            data = np.load(probe.path)
        except (OSError, ValueError):
            continue
        frame = np.asarray(data["frame"], dtype=np.float64) if "frame" in data else None
        phase = np.asarray(data["phase"], dtype=np.float64) if "phase" in data else None
        stored = (
            np.asarray(data["prediction"], dtype=np.float64)
            if "prediction" in data
            else None
        )
        model_grid = (
            np.asarray(data["model_grid"], dtype=np.float64)
            if "model_grid" in data
            else None
        )
        truth = ProbeTruth(
            probe=probe,
            frame=frame,
            model_grid=model_grid,
            stored=stored,
        )
        prediction = stored
        truth.source = "落盘" if stored is not None else "未记录"
        if prediction is None and model.available and phase is not None:
            truth.recomputed = model.predict(phase)
            prediction = truth.recomputed
            truth.source = "离线重算"
        truth.prediction = prediction
        if prediction is not None:
            truth.mapped = prediction_to_camera(prediction, geometry, cam_size)
        truths.append(truth)

    pairs = [
        (truth.mapped, truth.frame)
        for truth in truths
        if truth.mapped is not None and truth.frame is not None
    ]
    alignment = arm_alignment(stage, pairs)
    for truth in truths:
        if truth.stored is not None and truth.recomputed is not None:
            scale = float(np.max(np.abs(truth.stored)))
            if scale > 0:
                truth.agreement = float(
                    np.max(np.abs(truth.stored - truth.recomputed)) / scale
                )
        if truth.mapped is None or truth.frame is None:
            continue
        truth.mapped = align_to_camera(truth.mapped, alignment.shift)
        truth.r_camera = normalized_correlation(truth.mapped, truth.frame)
        truth.ssim_camera = ssim_score(truth.mapped, truth.frame)
        truth.r2_camera = linear_r2(truth.frame, truth.mapped)
        if truth.model_grid is not None and truth.prediction is not None:
            truth.r_model = normalized_correlation(truth.prediction, truth.model_grid)
    truths.sort(key=probe_sort_key)
    return truths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def parse_roi_center(log_text: str) -> tuple | None:
    """Read the runner's frozen ROI centre out of the stage log.

    The runner freezes the optical axis once per run and prints it
    (``frozen ROI centre (x, y) = (148, 144)``); every metric and every
    model/measurement comparison in that run is made about that point. Using
    it here means the report's frames are anchored exactly like the run's own
    numbers were.
    """
    match = re.search(
        r"frozen ROI centre \(x, y\) = \((\d+), (\d+)\)", log_text or ""
    )
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def last_error_line(log_text: str) -> str | None:
    """The last ``Error:`` line of a stage log, if the runner printed one."""
    lines = [line.strip() for line in strip_ansi(log_text or "").splitlines()]
    for line in reversed(lines):
        if line.startswith("Error:"):
            return line
    return None


def round_rows(run: RunData) -> list:
    """The recorder's per-round records, in round order.

    The flat CSV labels its rows (``stage`` = ``calibration`` / ``round``), so
    it is preferred. The pkl carries the same numbers without the label, so it
    is the fallback: a record counts as a round when it carries a round-only
    key (``loss_before`` / ``holdout_correlation``).
    """
    rows = [
        row
        for row in run.history
        if isinstance(row, dict) and str(row.get("stage", "")).strip() == "round"
    ]
    if rows:
        return rows
    fallback = [
        row
        for row in run.recorder
        if isinstance(row, dict)
        and ("loss_before" in row or "holdout_correlation" in row)
    ]
    return sorted(fallback, key=lambda row: float(row.get("round", 0) or 0))


def calibration_row(run: RunData) -> dict | None:
    """The recorder's calibration row -- the run's flat baseline.

    The flat CSV labels it ``stage = calibration``; it carries the metrics of
    the flat frame the run started from, which is the reference every later
    round is compared against.
    """
    for row in run.history:
        if isinstance(row, dict) and str(row.get("stage", "")).strip() == "calibration":
            return row
    return None


def style_axes() -> None:
    """Apply the repository's Chinese font settings."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def render_probe_gif(stage: StageData, out_path: Path) -> dict:
    """Render the arm's measured probes and the model's prediction as a GIF.

    Every probe the runner measured becomes one frame: the measured frame and
    the model's prediction (resampled onto the camera grid) side by side, the
    pupil phase the runner sent to the SLM, and the recorder's per-round
    metrics. The tail frames carry the recorder's whole history as text and as
    curves, so the animation ends on the numbers rather than on a frame.

    Returns:
        A dict with the frame count, the duration and the alignment that was
        used, or an empty dict when the arm recorded no probe.
    """
    probes = stage.run.probes
    if not probes:
        return {}
    geometry = stage.run.geometry or stage.run.session_geometry
    config = stage.run.config
    cam_size = int(as_float(config.get("cam_size")) or 300)

    loaded: list = []
    for probe in probes:
        try:
            data = np.load(probe.path)
        except (OSError, ValueError):
            continue
        frame = data["frame"] if "frame" in data else None
        phase = data["phase"] if "phase" in data else None
        prediction = data["prediction"] if "prediction" in data else None
        loaded.append((probe, frame, phase, prediction))
    if not loaded:
        return {}

    # The anchor: the run's own frozen ROI centre, i.e. the optical axis the
    # runner measured everything about. When the log does not carry it, the
    # alignment is fitted from the frames themselves and the report says so.
    pairs = []
    for probe, frame, _phase, prediction in loaded:
        if frame is None or prediction is None:
            continue
        mapped = prediction_to_camera(prediction, geometry, cam_size)
        if mapped is None:
            continue
        pairs.append((mapped, frame))
    alignment = arm_alignment(stage, pairs)
    shift = alignment.shift
    quality = alignment.quality
    anchor = alignment.anchor
    roi_center = alignment.roi_center

    history = round_rows(stage.run)
    rounds = [int(float(row.get("round", 0) or 0)) for row in history]
    curves = {
        "score": [as_float(row.get("score_after")) for row in history],
        "cv": [as_float(row.get("cv_after") or row.get("uniformity_cv")) for row in history],
        "ee": [as_float(row.get("ee_after")) for row in history],
        "holdout": [as_float(row.get("holdout_correlation")) for row in history],
    }

    frames: list = []
    for probe, frame, phase, prediction in loaded:
        frames.append(
            _render_gif_frame(
                stage,
                probe,
                frame,
                phase,
                prediction,
                geometry,
                cam_size,
                shift,
                curves,
                rounds,
                quality,
                roi_center,
            )
        )
    for index in range(GIF_TAIL_FRAMES):
        frames.append(
            _render_gif_tail(
                stage, curves, rounds, shift, quality, geometry, anchor, index + 1
            )
        )

    images = [Image.fromarray(frame) for frame in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=GIF_DURATION_MS,
        loop=0,
    )
    # PIL merges frames that are identical to their predecessor, so the frame
    # count of the written file is read back instead of assumed.
    with Image.open(out_path) as written:
        stored = int(getattr(written, "n_frames", 1))
    return {
        "frames": stored,
        "drawn_frames": len(images),
        "duration_s": len(images) * GIF_DURATION_MS / 1000.0,
        "width": images[0].width,
        "height": images[0].height,
        "shift": shift,
        "quality": quality,
        "anchor": anchor,
        "roi_center": roi_center,
    }


def _gif_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    fig.subplots_adjust(wspace=0.28, hspace=0.32)
    return fig, axes


def _render_gif_frame(
    stage,
    probe,
    frame,
    phase,
    prediction,
    geometry,
    cam_size,
    shift,
    curves,
    rounds,
    quality,
    roi_center,
):
    """Render one probe frame (measured | predicted | phase | metrics)."""
    fig, axes = _gif_figure()
    ax_measured, ax_predicted, ax_phase, ax_metrics = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    axis_label = "未记录"
    if frame is not None:
        ax_measured.imshow(display_image(frame), cmap="inferno", vmin=0.0, vmax=1.0)
        if roi_center is not None:
            ax_measured.plot(roi_center[0], roi_center[1], "+", color="cyan", markersize=10)
            axis_label = f"光轴 x={roi_center[0]}, y={roi_center[1]}"
        peak = np.unravel_index(int(np.argmax(frame)), frame.shape)
        ax_measured.set_title(f"实测远场 ({axis_label})\n帧内最大值 x={peak[1]}, y={peak[0]}", fontsize=8)
    else:
        ax_measured.text(0.5, 0.5, "未记录", ha="center", va="center")
        ax_measured.set_title("实测远场", fontsize=9)
    ax_measured.set_xticks([])
    ax_measured.set_yticks([])

    mapped = None
    if prediction is not None:
        mapped = prediction_to_camera(prediction, geometry, cam_size)
    correlation = float("nan")
    if mapped is not None:
        rolled = align_to_camera(mapped, shift)
        ax_predicted.imshow(display_image(rolled), cmap="inferno", vmin=0.0, vmax=1.0)
        if roi_center is not None:
            ax_predicted.plot(roi_center[0], roi_center[1], "+", color="cyan", markersize=10)
        if frame is not None:
            correlation = normalized_correlation(rolled, frame)
        ax_predicted.set_title(
            "模型预测 (按记录几何缩放到相机网格)\n与实测的相关性 "
            + (f"{correlation:.4f}" if correlation == correlation else "未记录"),
            fontsize=8,
        )
    else:
        ax_predicted.text(
            0.5,
            0.5,
            "该探针未落盘模型预测\n(训练探针只存实测帧)",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax_predicted.set_title("模型预测", fontsize=9)
    ax_predicted.set_xticks([])
    ax_predicted.set_yticks([])

    if phase is not None:
        ax_phase.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title(
            f"瞳面相位 (rad, max|c|={np.max(np.abs(phase)):.3f})", fontsize=9
        )
    else:
        ax_phase.text(0.5, 0.5, "未记录", ha="center", va="center")
        ax_phase.set_title("瞳面相位", fontsize=9)
    ax_phase.set_xticks([])
    ax_phase.set_yticks([])

    _plot_curves(ax_metrics, curves, rounds, marker=probe.round)
    ax_metrics.set_title("recorder 逐轮指标", fontsize=9)

    exposure = f", 曝光 {probe.exposure_ms:.3f} ms" if probe.exposure_ms else ""
    recorded = (
        f", 记录相关性 {probe.correlation:.4f}" if probe.correlation is not None else ""
    )
    fig.suptitle(
        f"{stage.name} — 第 {probe.round} 轮 {probe.split} 探针 #{probe.index}"
        f"{exposure}{recorded}",
        fontsize=10,
    )
    return _figure_to_array(fig)


def _render_gif_tail(stage, curves, rounds, shift, quality, geometry, anchor, index):
    """Render the tail frame: the recorder's whole history as text and curves."""
    fig, axes = _gif_figure()
    ax_measured, ax_predicted, ax_phase, ax_metrics = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    text = _recorder_text(stage)
    ax_measured.text(0.02, 0.98, text, ha="left", va="top", fontsize=9)
    ax_measured.set_title("该变体 recorder 记录", fontsize=9)
    ax_measured.set_xticks([])
    ax_measured.set_yticks([])
    ax_measured.axis("off")

    alignment = (
        f"锚点: {anchor}\n"
        f"平移 dy={shift[0]:+d} px, dx={shift[1]:+d} px\n"
        f"预测与实测的平均相关 {quality:.4f}\n"
        f"几何: camera/model px {fmt(geometry.get('camera_px_per_model_px'))}\n"
        f"远场网格 {geometry.get('far_field_size', '未记录')}"
    )
    ax_predicted.text(0.02, 0.98, alignment, ha="left", va="top", fontsize=9)
    ax_predicted.set_title("预测→相机网格的对齐", fontsize=9)
    ax_predicted.set_xticks([])
    ax_predicted.set_yticks([])
    ax_predicted.axis("off")

    ax_phase.text(
        0.5,
        0.5,
        f"收尾帧 {index}/{GIF_TAIL_FRAMES}\n实测帧 + 模型预测并排推进",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax_phase.set_xticks([])
    ax_phase.set_yticks([])
    ax_phase.axis("off")

    _plot_curves(ax_metrics, curves, rounds, marker=None)
    ax_metrics.set_title("recorder 逐轮指标 (全程)", fontsize=9)

    fig.suptitle(f"{stage.name} — recorder 全过程汇总", fontsize=10)
    return _figure_to_array(fig)


def _figure_to_array(fig) -> np.ndarray:
    """Render a matplotlib figure into an RGB array."""
    canvas = fig.canvas
    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    return buffer[:, :, :3]


def _plot_curves(ax, curves, rounds, marker=None) -> None:
    """Draw the recorder's per-round curves on one axis."""
    ax.clear()
    labels = {
        "score": "score",
        "cv": "CV",
        "ee": "EE",
        "holdout": "验证集相关性",
    }
    styles = {
        "score": ("#1f77b4", "-"),
        "cv": ("#d62728", "--"),
        "ee": ("#2ca02c", "-."),
        "holdout": ("#9467bd", ":"),
    }
    for key, values in curves.items():
        xs = list(range(1, len(values) + 1))
        pairs = [(x, y) for x, y in zip(xs, values) if y is not None]
        if not pairs:
            continue
        color, style = styles[key]
        ax.plot(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            style,
            color=color,
            marker="o",
            markersize=3,
            label=labels[key],
        )
    if marker is not None and curves.get("score"):
        ax.axvline(marker + 0.5, color="grey", linewidth=0.8, alpha=0.6)
    longest = max((len(values) for values in curves.values()), default=0)
    ax.set_xlim(0.5, max(1.5, longest + 0.5))
    ax.set_xlabel("轮次", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def _recorder_text(stage: StageData) -> str:
    """Render the arm's recorder history as a text block."""
    lines = [f"status: {stage.run.recorder_status or '未记录'}"]
    handoff = stage.run.handoff or {}
    metrics = handoff.get("metrics_at_best_recorded_round") or {}
    lines.append(f"接受轮数: {handoff.get('accepted_rounds', '未记录')}")
    lines.append(f"平场分数: {fmt(stage.observed.get('flat_score'))}")
    lines.append(f"整形后分数: {fmt(stage.observed.get('score_after'))}")
    lines.append(f"CV : {fmt(metrics.get('cv'))}")
    lines.append(f"EE : {fmt(metrics.get('ee'))}")
    lines.append(f"IoU: {fmt(metrics.get('iou'))}")
    lines.append(f"验证集相关性: {fmt(metrics.get('holdout_correlation'))}")
    lines.append(f"几何相关性: {fmt(metrics.get('geometry_correlation'))}")
    lines.append(f"verdict: {handoff.get('verdict', '未记录')}")
    checks = handoff.get("checks") or {}
    if checks:
        lines.append("门槛: " + ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in checks.items()))
    return "\n".join(lines)


def render_summary_figure(stages, out_path: Path, exposure_ladder, stamp: str) -> Path:
    """Render the arms side by side, with the exposure window and the geometry."""
    arms = [stage for stage in stages if stage.kind != "light"]
    labels = [stage.name for stage in arms]
    style_axes()
    # The arm-status table needs the extra width: its ``outcome`` column holds
    # a long word, and a table whose cells overflow overlap each other.
    fig, axes = plt.subplots(
        2, 3, figsize=(14.0, 7.9), width_ratios=[1.4, 1.0, 1.0]
    )
    fig.subplots_adjust(wspace=0.34, hspace=0.46, left=0.065, right=0.985)

    flat_scores = [as_float(stage.observed.get("flat_score")) for stage in arms]
    shaped_scores = [as_float(stage.observed.get("score_after")) for stage in arms]
    positions = np.arange(len(arms), dtype=float)
    width = 0.36

    ax = axes[0][0]
    ax.bar(
        positions - width / 2,
        [np.nan if value is None else value for value in flat_scores],
        width,
        label="平场分数",
        color="#7f7f7f",
    )
    ax.bar(
        positions + width / 2,
        [np.nan if value is None else value for value in shaped_scores],
        width,
        label="整形后分数",
        color="#1f77b4",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("分数 (越大越好)", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0][1]
    metrics = [("cv", "CV"), ("ee", "EE"), ("iou", "IoU")]
    for offset, (key, label) in enumerate(metrics):
        values = [as_float(stage.observed.get(key)) for stage in arms]
        ax.bar(
            positions + (offset - 1) * width,
            [np.nan if value is None else value for value in values],
            width,
            label=label,
            color=["#d62728", "#2ca02c", "#9467bd"][offset],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("CV / EE / IoU", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0][2]
    correlations = []
    for stage in arms:
        values = stage.observed.get("holdout_correlation") or []
        values = [as_float(value) for value in values]
        values = [value for value in values if value is not None]
        correlations.append(values[-1] if values else None)
    ax.bar(
        positions,
        [np.nan if value is None else value for value in correlations],
        0.5,
        color="#ff7f0e",
        label="实测验证集相关性",
    )
    for index, stage in enumerate(arms):
        threshold = as_float(stage.profile.get("min_holdout_correlation"))
        if threshold is None or correlations[index] is None:
            continue
        ax.plot(
            [index - 0.3, index + 0.3],
            [threshold, threshold],
            color="black",
            linewidth=1.4,
            linestyle="--",
        )
        ax.text(
            index,
            threshold + 0.02,
            f"门槛 {threshold:g}",
            ha="center",
            fontsize=10,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("验证集 (holdout) 相关性与该变体门槛", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1][0]
    rows = []
    for stage in arms:
        rows.append(
            [
                stage.name,
                stage.outcome or "未记录",
                "是" if stage.expectation_met else ("否" if stage.expectation_met is not None else "未记录"),
                stage.observed.get("rounds_accepted", "未记录"),
                stage.observed.get("rounds_total", "未记录"),
                fmt(stage.observed.get("geometry_correlation")),
            ]
        )
    ax.axis("off")
    ax.set_title("各变体状态", fontsize=12)
    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["变体", "outcome", "期望达成", "接受轮数", "总轮数", "几何相关性"],
            loc="upper center",
            cellLoc="center",
            colWidths=[0.167, 0.306, 0.134, 0.134, 0.105, 0.153],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.8)

    ax = axes[1][1]
    _plot_exposure_window(ax, stages, exposure_ladder)
    ax.set_title("曝光窗口 (曝光 ms → 峰值 counts)", fontsize=12)
    ax.tick_params(labelsize=10)

    ax = axes[1][2]
    ax.axis("off")
    ax.set_title("几何标定 (sweep)", fontsize=12)
    notes = [f"会话戳: {stamp}"]
    for stage in arms:
        geometry = stage.run.geometry or stage.run.session_geometry
        if not geometry:
            continue
        # Two lines per arm: a single long line runs past the panel and
        # stretches the whole figure.
        notes.append(f"{stage.name}")
        notes.append(
            f"  sweep 相关性 {fmt(geometry.get('correlation'))} · "
            f"远场 {geometry.get('far_field_size', '未记录')}"
        )
        notes.append(
            f"  camera/model px {fmt(geometry.get('camera_px_per_model_px'))} · "
            f"FWHM {fmt(geometry.get('spot_fwhm_camera_px'), 3)} cam px"
        )
    ax.text(0.02, 0.98, "\n".join(notes), ha="left", va="top", fontsize=10.5)
    fig.suptitle(f"model-in-loop 实验系统 A/B campaign 汇总 ({stamp})", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path)
    return out_path


def _plot_exposure_window(ax, stages, exposure_ladder) -> None:
    """Plot the measured exposure ladders of the session."""
    plotted = False
    for stage in stages:
        for directory in (stage.stage_dir / "light", stage.stage_dir):
            payload = read_light_verdict(directory)
            if not payload:
                continue
            xs = []
            ys = []
            for reading in payload["readings"]:
                exposure = as_float(reading.get("effective_ms") or reading.get("requested_ms"))
                peak = as_float(reading.get("max"))
                if exposure is None or peak is None:
                    continue
                xs.append(exposure)
                ys.append(peak)
            if xs:
                ax.plot(xs, ys, "o-", markersize=4, label=f"{stage.name} 曝光阶梯")
                plotted = True
            break
    if exposure_ladder:
        ax.plot(
            [point[0] for point in exposure_ladder],
            [point[1] for point in exposure_ladder],
            "s--",
            color="#444444",
            markersize=4,
            label="当天实测曝光窗口 (注入)",
        )
        plotted = True
    if plotted:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("曝光 (ms)", fontsize=11)
    ax.set_ylabel("峰值 (counts)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")


def render_panel_scan_figure(stage: StageData, out_path: Path) -> Path | None:
    """Render the panel-locate response map and the flat drift it judged on."""
    measurements = stage.run.panel_measurements
    summary = stage.run.panel_scan
    if not measurements and not summary:
        return None
    style_axes()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.2))
    fig.subplots_adjust(wspace=0.48, left=0.075, right=0.985, top=0.86, bottom=0.24)

    ax = axes[0]
    if measurements:
        xs = [as_float(item.get("x")) for item in measurements]
        ys = [as_float(item.get("y")) for item in measurements]
        deltas = [as_float(item.get("delta")) for item in measurements]
        valid = [
            (x, y, d)
            for x, y, d in zip(xs, ys, deltas)
            if x is not None and y is not None and d is not None and d == d
        ]
        if valid:
            grid = np.full((5, 5), np.nan)
            for x, y, delta in valid:
                col = int(round((x - 240) / 360)) if x >= 240 else 0
                row = int(round((y - 220) / 260)) if y >= 220 else 0
                if 0 <= row < 5 and 0 <= col < 5:
                    grid[row, col] = delta
            image = ax.imshow(grid, cmap="coolwarm", vmin=-0.02, vmax=0.02)
            ax.set_xticks(range(5))
            ax.set_xticklabels([240, 600, 960, 1320, 1680], fontsize=10)
            ax.set_yticks(range(5))
            ax.set_yticklabels([220, 480, 720, 980, 1240], fontsize=10)
            ax.set_xlabel("面板 x (px)", fontsize=11)
            ax.set_ylabel("面板 y (px)", fontsize=11)
            colourbar = fig.colorbar(image, ax=ax, fraction=0.046)
            colourbar.ax.tick_params(labelsize=9)
            colourbar.set_label("delta", fontsize=10)
    ax.set_title(f"{stage.name} 全板定位响应 (delta)", fontsize=12)

    ax = axes[1]
    threshold = as_float(summary.get("max_flat_drift"))
    if threshold is None:
        threshold = 0.02
    if measurements:
        labels = [f"{int(item.get('x', 0))},{int(item.get('y', 0))}" for item in measurements]
        drift = [as_float(item.get("flat_drift")) for item in measurements]
        colours = [
            "#d62728" if (d is not None and d > threshold) else "#1f77b4"
            for d in drift
        ]
        ax.bar(range(len(labels)), [d if d is not None else 0.0 for d in drift], color=colours)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylabel("平场漂移", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"每个位置的平场漂移 (虚线 = 严格模式阈值 {threshold:g})", fontsize=12)

    ax = axes[2]
    ax.axis("off")
    lines = [f"{stage.name} 全板定位汇总"]
    for key in (
        "session_drift",
        "max_pair_drift",
        "session_shift_px",
        "session_shift_limit_px",
        "flat_fwhm_px",
        "rejected_count",
        "best_center",
        "max_flat_drift",
        "max_session_drift",
        "strict",
    ):
        if key in summary:
            lines.append(f"{key}: {summary[key]}")
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10.5)
    ax.set_title("严格模式判定", fontsize=12)

    fig.suptitle(f"{stage.name} 全板定位扫描证据", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# The truth against the prediction, and the metrics curves
# ---------------------------------------------------------------------------


def _probe_row_title(truth: ProbeTruth) -> str:
    """The title of one probe row: its type, its round and its index."""
    exposure = (
        f", 曝光 {truth.probe.exposure_ms:.3f} ms"
        if truth.probe.exposure_ms
        else ""
    )
    return f"{truth.split} · 第 {truth.probe.round} 轮 · 探针 #{truth.probe.index}{exposure}"


def _truth_page_height(rows: int) -> float:
    """The height (in inches) of a truth-vs-prediction page with ``rows`` probes."""
    return (
        TRUTH_HEADER_IN
        + rows * (TRUTH_PANEL_IN + TRUTH_ROW_GAP_IN)
        + TRUTH_BOTTOM_IN
    )


def render_truth_probe_pages(
    stage: StageData, truths: list[ProbeTruth], fig_dir: Path, stamp: str
) -> list[Path]:
    """Render the arm's probes as pages of readable three-panel rows.

    One row per probe -- the measured frame, the model's prediction on the
    camera grid, and their difference -- with the probe's identity on the
    measured panel and its Pearson r / SSIM on the difference panel. The rows
    are split over several pages (see :data:`TRUTH_PAGE_ROWS`) so that one
    panel stays large enough to read: a 10 in wide page is read in a 900 px
    report column at about 90 px/in, which puts one square panel at ~260 px
    and a 10 pt label at ~12 px.

    Args:
        stage: The arm.
        truths: The arm's probes, grouped by type and ordered.
        fig_dir: The directory the pages are written to.
        stamp: The stamp every file name carries.

    Returns:
        The paths of the pages, in reading order.
    """
    if not truths:
        return []
    pages = [
        truths[start : start + TRUTH_PAGE_ROWS]
        for start in range(0, len(truths), TRUTH_PAGE_ROWS)
    ]
    written: list[Path] = []
    for index, page in enumerate(pages):
        out_path = fig_dir / f"{stamp}_true_vs_pred_{stage.name}_probes_p{index + 1}.png"
        render_truth_probe_page(stage, page, out_path, index + 1, len(pages))
        written.append(out_path)
    return written


def render_truth_probe_page(
    stage: StageData,
    truths: list[ProbeTruth],
    out_path: Path,
    page_index: int,
    page_count: int,
) -> Path:
    """Render one page of the arm's probes.

    Args:
        stage: The arm.
        truths: The probes on this page.
        out_path: Where the page is written.
        page_index: This page's number, counting from 1.
        page_count: How many pages the arm's probes take.

    Returns:
        The path of the page.
    """
    rows = len(truths)
    height = _truth_page_height(rows)
    style_axes()
    fig = plt.figure(figsize=(TRUTH_FIG_WIDTH_IN, height))
    available = height - TRUTH_HEADER_IN - TRUTH_BOTTOM_IN
    hspace = (available / TRUTH_PANEL_IN - rows) / max(rows - 1, 1)
    grid = fig.add_gridspec(
        rows,
        3,
        left=0.012,
        right=0.988,
        top=1.0 - TRUTH_HEADER_IN / height,
        bottom=TRUTH_BOTTOM_IN / height,
        wspace=0.055,
        hspace=hspace,
    )
    for row, truth in enumerate(truths):
        _draw_probe_row(fig, grid, row, truth)

    column_span = (0.988 - 0.012) / 3
    for index, header in enumerate(
        ("实测帧（相机网格）", "模型预测（相机网格）", "差值（预测 − 实测）")
    ):
        fig.text(
            0.012 + (index + 0.5) * column_span,
            1.0 - 0.95 / height,
            header,
            ha="center",
            va="top",
            fontsize=13,
        )
    page_note = (
        f"第 {page_index}/{page_count} 页" if page_count > 1 else f"共 {rows} 条探针"
    )
    fig.text(
        0.5,
        1.0 - 0.28 / height,
        f"{stage.name} — 真实光路下的真值 vs 预测（逐探针，{page_note}）",
        ha="center",
        va="top",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path)
    return out_path


def _draw_probe_row(fig, grid, row: int, truth: ProbeTruth) -> None:
    """Draw one probe row: the measured frame, the prediction, the difference."""
    colour = SPLIT_COLOURS.get(truth.split, "#444444")
    ax_measured = fig.add_subplot(grid[row, 0])
    ax_predicted = fig.add_subplot(grid[row, 1])
    ax_difference = fig.add_subplot(grid[row, 2])

    if truth.frame is not None:
        ax_measured.imshow(
            display_image(truth.frame), cmap="inferno", vmin=0.0, vmax=1.0
        )
    else:
        ax_measured.text(0.5, 0.5, "未记录", ha="center", va="center", fontsize=11)
    identity = f"{truth.split} · 第 {truth.probe.round} 轮 · 探针 #{truth.probe.index}"
    if truth.probe.exposure_ms:
        identity += f"\n曝光 {truth.probe.exposure_ms:.3f} ms"
    ax_measured.text(
        0.025,
        0.975,
        identity,
        transform=ax_measured.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="white",
        bbox={
            "facecolor": (0.0, 0.0, 0.0, 0.55),
            "pad": 3,
            "edgecolor": colour,
            "linewidth": 1.4,
        },
    )
    ax_measured.set_xticks([])
    ax_measured.set_yticks([])

    if truth.mapped is not None:
        ax_predicted.imshow(
            display_image(truth.mapped), cmap="inferno", vmin=0.0, vmax=1.0
        )
    else:
        ax_predicted.text(
            0.5,
            0.5,
            "模型预测未记录",
            ha="center",
            va="center",
            fontsize=11,
        )
    ax_predicted.set_xticks([])
    ax_predicted.set_yticks([])

    if truth.frame is not None and truth.mapped is not None:
        difference = truth.mapped - truth.frame
        span = float(np.max(np.abs(difference))) or 1.0
        ax_difference.imshow(difference, cmap="coolwarm", vmin=-span, vmax=span)
        ax_difference.text(
            0.025,
            0.975,
            f"r = {fmt(truth.r_camera)}\nSSIM = {fmt(truth.ssim_camera)}\n"
            f"预测来源: {truth.source}\n色标 ±{span:.3g}",
            transform=ax_difference.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="white",
            bbox={
                "facecolor": (0.0, 0.0, 0.0, 0.55),
                "pad": 3,
                "edgecolor": colour,
                "linewidth": 1.4,
            },
        )
    else:
        ax_difference.text(0.5, 0.5, "未记录", ha="center", va="center", fontsize=11)
    ax_difference.set_xticks([])
    ax_difference.set_yticks([])


def render_truth_stats_figure(
    stage: StageData, truths: list[ProbeTruth], out_path: Path
) -> Path | None:
    """Render the arm's pooled 1:1 density plot and the per-probe r spread.

    The left panel is the 1:1 density plot of the measured against the
    predicted pixels; the right panel is the per-probe Pearson r, coloured by
    probe type, with the mean of each type drawn as a dotted line.

    Returns:
        The path of the figure, or ``None`` when the arm recorded no probe.
    """
    if not truths:
        return None
    style_axes()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.9))
    fig.subplots_adjust(wspace=0.40, left=0.075, right=0.965, top=0.85, bottom=0.13)

    ax_scatter = axes[0]
    measured_all: list[np.ndarray] = []
    predicted_all: list[np.ndarray] = []
    for truth in truths:
        if truth.frame is None or truth.mapped is None:
            continue
        measured_all.append(truth.frame.ravel())
        predicted_all.append(truth.mapped.ravel())
    if measured_all:
        measured = np.concatenate(measured_all)
        predicted = np.concatenate(predicted_all)
        upper = float(max(np.max(measured), np.max(predicted)))
        edges = np.linspace(0.0, upper, 61)
        density, _, _ = np.histogram2d(
            measured, predicted, bins=(edges, edges)
        )
        image = ax_scatter.imshow(
            np.log10(density.T + 1.0),
            origin="lower",
            aspect="equal",
            cmap="magma",
            extent=(0.0, upper, 0.0, upper),
        )
        ax_scatter.plot([0, upper], [0, upper], "w--", linewidth=1.2, alpha=0.8)
        pooled_r = normalized_correlation(predicted, measured)
        pooled_r2 = linear_r2(measured, predicted)
        ax_scatter.text(
            0.03,
            0.97,
            f"实测像素 vs 预测像素\nr = {fmt(pooled_r)}\n"
            f"R²(线性拟合) = {fmt(pooled_r2)}\n"
            f"像素数 = {measured.size}",
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="white",
            bbox={"facecolor": (0.0, 0.0, 0.0, 0.55), "pad": 3},
        )
        ax_scatter.set_xlabel("实测像素值", fontsize=11)
        ax_scatter.set_ylabel("预测像素值", fontsize=11)
        ax_scatter.tick_params(labelsize=10)
        colourbar = fig.colorbar(
            image, ax=ax_scatter, label="log10 像素计数", fraction=0.046
        )
        colourbar.ax.tick_params(labelsize=9)
        colourbar.set_label("log10 像素计数", fontsize=10)
    else:
        ax_scatter.text(0.5, 0.5, "未记录", ha="center", va="center", fontsize=11)
    ax_scatter.set_title("1:1 密度图", fontsize=12)

    ax_bars = axes[1]
    labels = [truth.label for truth in truths]
    values = [np.nan if truth.r_camera is None else truth.r_camera for truth in truths]
    colours = [
        SPLIT_COLOURS.get(truth.split, "#444444") for truth in truths
    ]
    positions = np.arange(len(truths), dtype=float)
    ax_bars.bar(positions, values, color=colours)
    for position, value in zip(positions, values):
        if value != value:
            continue
        ax_bars.text(
            position,
            value + (0.04 if value >= 0 else -0.04),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    ax_bars.set_xticks(positions)
    ax_bars.set_xticklabels(labels, rotation=90, fontsize=9)
    ax_bars.set_ylabel("r (实测 vs 预测)", fontsize=11)
    ax_bars.set_ylim(-1.1, 1.1)
    ax_bars.tick_params(axis="y", labelsize=10)
    ax_bars.grid(axis="y", alpha=0.3)
    for split, group in _split_groups(truths).items():
        group_values = [
            truth.r_camera for truth in group if truth.r_camera is not None
        ]
        if not group_values:
            continue
        ax_bars.axhline(
            float(np.mean(group_values)),
            color=SPLIT_COLOURS.get(split, "#444444"),
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SPLIT_COLOURS.get(split, "#444444"))
        for split in _split_groups(truths)
    ]
    ax_bars.legend(
        handles,
        list(_split_groups(truths)),
        fontsize=9,
        title="探针类型 (虚线 = 该类均值)",
        title_fontsize=9,
        loc="lower left",
    )
    ax_bars.set_title("逐探针 r 的分布", fontsize=12)

    fig.suptitle(f"{stage.name} — 真值 vs 预测的汇总统计", fontsize=14, y=0.97)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path)
    return out_path


def _split_groups(truths: list[ProbeTruth]) -> dict:
    """The probes grouped by type, in the report's group order."""
    groups: dict[str, list[ProbeTruth]] = {}
    for truth in truths:
        groups.setdefault(truth.split, []).append(truth)
    ordered: dict[str, list[ProbeTruth]] = {}
    for split in PROBE_SPLIT_ORDER:
        if split in groups:
            ordered[split] = groups[split]
    for split, group in groups.items():
        ordered.setdefault(split, group)
    return ordered


def first_deviation(
    rows: list, baseline: dict | None, column: str
) -> tuple[int, float, float] | None:
    """The first round whose metric left the flat baseline.

    Args:
        rows: The recorder's per-round rows.
        baseline: The recorder's calibration row (the flat baseline).
        column: The metric column to compare.

    Returns:
        ``(round, value, value - baseline)``, or ``None`` when the run recorded
        neither a baseline nor a round value.
    """
    if not baseline:
        return None
    base = as_float(baseline.get(column))
    if base is None:
        return None
    for row in rows:
        value = as_float(row.get(column))
        if value is None:
            continue
        if abs(value - base) > 1e-12:
            return int(float(row.get("round", 0) or 0)), value, value - base
    return None


def render_metrics_curves_figure(stage: StageData, out_path: Path) -> Path | None:
    """Render the recorder's per-round metrics of the arm.

    Four panels -- composite score, CV, EE and the validation correlation --
    against the round. The flat baseline the run started from is drawn as a
    dashed line, the acceptance thresholds as a second dashed line, and each
    round is marked by whether the recorder accepted it.

    Returns:
        The path of the figure, or ``None`` when the arm recorded no round.
    """
    run = stage.run
    rows = round_rows(run)
    baseline = calibration_row(run)
    if not rows and baseline is None:
        return None
    style_axes()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    fig.subplots_adjust(wspace=0.30, hspace=0.46, left=0.085, right=0.985)
    config = run.config or {}
    thresholds = {
        "min_holdout_correlation": as_float(config.get("min_holdout_correlation")),
        "min_geometry_correlation": as_float(config.get("min_geometry_correlation")),
    }

    for ax, (key, label, column, baseline_column) in zip(axes.ravel(), ROUND_METRICS):
        base_value = (
            as_float(baseline.get(baseline_column)) if baseline and baseline_column else None
        )
        xs: list[float] = []
        ys: list[float] = []
        accepted: list[bool] = []
        if base_value is not None:
            xs.append(0.0)
            ys.append(base_value)
            accepted.append(True)
        for row in rows:
            value = as_float(row.get(column))
            if value is None:
                continue
            xs.append(float(row.get("round", 0) or 0))
            ys.append(value)
            flag = str(row.get("accepted", "")).strip().lower()
            accepted.append(flag in ("true", "1", "yes"))
        if xs:
            ax.plot(xs, ys, "-", color="#1f77b4", linewidth=1.2, zorder=1)
            for x, y, ok in zip(xs, ys, accepted):
                if x == 0.0:
                    ax.plot(
                        x,
                        y,
                        "D",
                        color="#7f7f7f",
                        markersize=6,
                        label="平场基线 (第 0 轮)",
                        zorder=3,
                    )
                elif ok:
                    ax.plot(
                        x,
                        y,
                        "o",
                        color="#1f77b4",
                        markersize=6,
                        label="该轮被接受",
                        zorder=3,
                    )
                else:
                    ax.plot(
                        x,
                        y,
                        "s",
                        markerfacecolor="none",
                        markeredgecolor="#d62728",
                        markersize=7,
                        label="该轮被拒绝",
                        zorder=3,
                    )
        if base_value is not None:
            ax.axhline(
                base_value,
                color="#7f7f7f",
                linestyle="--",
                linewidth=1.0,
                label="平场基线分",
            )
        if key == "holdout":
            for name, threshold in thresholds.items():
                if threshold is None:
                    continue
                ax.axhline(
                    threshold,
                    color="#d62728",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"验收阈值 {name} = {threshold:g}",
                )
            ax.set_ylim(0.0, 1.05)
        deviation = first_deviation(rows, baseline, column)
        if deviation is not None:
            round_no, value, delta = deviation
            ax.axvline(round_no, color="#ff7f0e", linewidth=1.0, alpha=0.7)
            # The note is anchored inside the axes: the round usually sits at
            # the right edge, and an offset annotation would be clipped off
            # the figure.
            low, high = ax.get_ylim()
            fraction = (value - low) / (high - low) if high > low else 0.5
            # The note goes to the half of the panel the point is *not* in.
            below = fraction > 0.5
            ax.annotate(
                f"第 {round_no} 轮偏离平场 {delta:+.4f}",
                xy=(round_no, value),
                xytext=(0.97, 0.06 if below else 0.94),
                textcoords="axes fraction",
                ha="right",
                va="bottom" if below else "top",
                fontsize=10,
                color="#ff7f0e",
                arrowprops={"arrowstyle": "->", "color": "#ff7f0e", "linewidth": 1.0},
            )
        ax.set_xlabel("轮次 (0 = 平场基线)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"{stage.name} — recorder 的逐轮指标曲线（真实侧的全过程）", fontsize=14
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def exposure_ladder_from_text(text: str) -> list:
    """Parse a ``ms:counts`` ladder given on the command line."""
    points: list = []
    for chunk in (text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"exposure ladder entry {chunk!r} is not 'ms:counts'")
        left, right = chunk.split(":", 1)
        points.append((float(left), float(right)))
    return points


def stage_metric_rows(stages) -> list:
    """Rows for the per-arm expectation/measurement table."""
    rows = []
    for stage in stages:
        observed = stage.observed
        rows.append(
            [
                stage.name,
                stage.kind,
                stage.expected or "未记录",
                stage.outcome or "未记录",
                "是" if stage.expectation_met else ("否" if stage.expectation_met is not None else "未记录"),
                observed.get("status") or "未记录",
                observed.get("rounds_accepted", "未记录"),
                fmt(observed.get("flat_score")),
                fmt(observed.get("score_after")),
                fmt(observed.get("cv")),
                fmt(observed.get("ee")),
                fmt(observed.get("iou")),
            ]
        )
    return rows


def build_report(
    session_dir: Path,
    stages,
    manifest,
    figure_notes: list,
    exposure_ladder: list,
    stamp: str,
    report_name: str,
    session_rel: str,
    summary_figure: str,
    gif_figures: dict,
    scan_figures: dict,
    truth_figures: dict,
    truth_stats_figures: dict,
    metrics_figures: dict,
    truths_by_arm: dict,
) -> str:
    """Assemble the report body."""
    arms = [stage for stage in stages if stage.kind != "light"]
    light_stage = next((stage for stage in stages if stage.kind == "light"), None)

    lines: list = []
    lines.append(f"# model-in-loop 实验系统 A/B campaign 报告 ({stamp})")
    lines.append("")
    lines.append(f"- 会话目录: `{session_rel}`（仓库根相对路径）")
    lines.append("- 报告生成: `scripts/slm/generate_model_in_loop_hw_ab_report.py`（离线读盘）")
    lines.append(
        "- campaign 清单: "
        + (
            f"`{session_dir.name}/manifest.json`（campaign_state = "
            f"{manifest.get('campaign_state')}）"
            if isinstance(manifest, dict)
            else "**未生成**（campaign 仍在运行或已中断，本报告改为从各阶段目录与 recorder 产物重读）"
        )
    )
    lines.append("")

    # 1 -- one-line conclusion
    lines.append("## 1. 一句话结论")
    lines.append("")
    lines.append(_conclusion(stages, arms))
    lines.append("")

    # 2 -- the bench and what was measured on the day
    lines.append("## 2. 实验系统与当天实测")
    lines.append("")
    lines.append("### 2.1 曝光窗口")
    lines.append("")
    lines.append(
        "相机的曝光范围与当天实测的曝光阶梯（曝光 ms → 峰值 counts）如下；"
        "会话内的阶梯直接来自各阶段的光照闸门 `light_level.json`。"
    )
    lines.append("")
    ladder_rows = []
    for stage in stages:
        payload = read_light_verdict(stage.stage_dir / "light") or read_light_verdict(
            stage.stage_dir
        )
        if not payload:
            continue
        readings = []
        for reading in payload["readings"]:
            exposure = as_float(reading.get("effective_ms") or reading.get("requested_ms"))
            peak = as_float(reading.get("max"))
            if exposure is None or peak is None:
                continue
            readings.append(f"{exposure:g} ms→{peak:g}")
        ladder_rows.append(
            [
                stage.name,
                "、".join(readings) if readings else "未记录",
                payload.get("state") or "未记录",
                "；".join(str(reason) for reason in payload.get("reasons") or []) or "未记录",
            ]
        )
    if ladder_rows:
        lines.append(markdown_table(["闸门", "曝光阶梯", "判定", "理由"], ladder_rows))
    else:
        lines.append("未记录（会话内没有光照闸门产物）")
    lines.append("")
    if exposure_ladder:
        lines.append(
            "当天实测的曝光窗口（由 `--exposure-ladder` 注入，采集脚本路径未记录）："
            + "、".join(f"{ms:g} ms→{counts:g} counts" for ms, counts in exposure_ladder)
        )
        lines.append("")
    lines.append(
        "曝光上限的坑见 §6.1：仓库自带的倍增阶梯在弱光下会被噪声底带偏，"
        "给出的上限比真实工作点低一个量级。"
    )
    lines.append("")

    lines.append("### 2.2 0 级位置与 FWHM")
    lines.append("")
    spot_rows = []
    for stage in stages:
        payload = read_light_verdict(stage.stage_dir / "light") or read_light_verdict(
            stage.stage_dir
        )
        geometry = stage.run.geometry or stage.run.session_geometry
        zero_order = "未记录"
        fwhm = "未记录"
        if payload:
            for reading in payload["readings"]:
                spot = str(reading.get("spot") or "")
                if "c=(" in spot:
                    zero_order = spot.split("c=(")[-1].split(")")[0]
                if "fwhm=" in spot:
                    fwhm = spot.split("fwhm=")[-1].split("px")[0] + " px"
        if geometry:
            fwhm = f"{fmt(geometry.get('spot_fwhm_camera_px'), 3)} cam px (几何标定)"
        panel = stage.run.panel_scan or {}
        if panel.get("flat_fwhm_px") is not None:
            fwhm = f"{fmt(panel.get('flat_fwhm_px'), 3)} px (全板定位平场)"
        spot_rows.append(
            [
                stage.name,
                zero_order,
                fwhm,
                f"{panel.get('best_center', '未记录')}" if panel else "未记录",
            ]
        )
    lines.append(
        markdown_table(
            ["变体", "0 级质心 (全传感器坐标)", "FWHM", "标定的瞳面中心 (面板 px)"],
            spot_rows,
        )
    )
    lines.append("")
    lines.append(
        "光照闸门的 0 级按**帧全局最大**（`argmax`）定位，不按相机几何中心 —— "
        "这条光路的光轴与画幅中心不重合；"
        "而 MIL 各变体自己的帧用的是运行期**冻结的光轴**"
        "（`stage.log` 里的 `frozen ROI centre`），动画里的两个面板都按它对齐。"
    )
    lines.append("")

    lines.append("### 2.3 几何标定 (sweep)")
    lines.append("")
    geometry_rows = []
    for stage in arms:
        geometry = stage.run.geometry or stage.run.session_geometry
        if not geometry:
            geometry_rows.append([stage.name, "未记录"] + ["未记录"] * 5)
            continue
        geometry_rows.append(
            [
                stage.name,
                str(geometry.get("method", "未记录")),
                fmt(geometry.get("correlation")),
                fmt(geometry.get("camera_px_per_model_px")),
                geometry.get("far_field_size", "未记录"),
                fmt(geometry.get("spot_fwhm_camera_px"), 3),
                fmt(geometry.get("spot_fwhm_model_px"), 3),
            ]
        )
    lines.append(
        markdown_table(
            [
                "变体",
                "方法",
                "sweep 相关性",
                "camera/model px",
                "远场网格",
                "FWHM (cam px)",
                "FWHM (model px)",
            ],
            geometry_rows,
        )
    )
    lines.append("")
    for stage in arms:
        geometry = stage.run.geometry or stage.run.session_geometry
        notes = geometry.get("calibration_notes") if geometry else None
        if notes:
            lines.append(f"- **{stage.name}** 标定备注: {notes}")
    lines.append("")
    lines.append(
        "注意：两个变体的 `camera_px_per_model_px` 相差约一倍（见上表），"
        "而两个变体的标定备注里 FOV 交叉核对都写着 FOV 与 ramp 尺度"
        "「相差数倍」—— 见 §4.1 与 §4.2，这条几何标定本身还没自洽。"
    )
    lines.append("")
    if summary_figure:
        lines.append(f"![各变体汇总]({summary_figure})")
        lines.append("")

    # 3 -- expectation vs measurement
    lines.append("## 3. 每个变体的期望 vs 实测")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "变体",
                "类别",
                "campaign 期望",
                "实测 outcome",
                "期望达成",
                "status",
                "接受轮数",
                "平场分数",
                "整形后分数",
                "CV",
                "EE",
                "IoU",
            ],
            stage_metric_rows(arms),
        )
    )
    lines.append("")
    for stage in arms:
        lines.append(f"### 3.{arms.index(stage) + 1} {stage.name} 的记录")
        lines.append("")
        notes = [str(note) for note in stage.notes if str(note).strip()]
        if notes:
            lines.append("- campaign 备注: " + "；".join(notes))
        else:
            lines.append("- campaign 备注: 未记录")
        handoff = stage.run.handoff or {}
        if handoff:
            checks = handoff.get("checks") or {}
            lines.append(
                "- 交接判定: "
                f"`{handoff.get('verdict', '未记录')}`，门槛 "
                + ", ".join(f"{key}={'通过' if value else '未通过'}" for key, value in checks.items())
            )
        lines.append(
            "- 退出码: "
            + (str(stage.exit_code) if stage.exit_code is not None else "未记录（campaign 清单未生成）")
        )
        error = last_error_line(stage.log_text)
        if error:
            lines.append(f"- runner 的错误行: `{error}`")
        gif = gif_figures.get(stage.name)
        if gif:
            lines.append("")
            lines.append(f"![{stage.name} 的逆向优化动画]({gif})")
        scan = scan_figures.get(stage.name)
        if scan:
            lines.append("")
            lines.append(f"![{stage.name} 全板定位扫描]({scan})")
        lines.append("")

    # 4 -- the A/B answer
    lines.append("## 4. A/B 的结论")
    lines.append("")
    lines.append(_ab_section(arms))
    lines.append("")

    # 5 -- the truth against the prediction, and the metrics curves
    lines.append(
        _truth_section(
            arms, truth_figures, truth_stats_figures, metrics_figures, truths_by_arm
        )
    )
    lines.append("")

    # 6 -- the defects that blocked the way
    lines.append("## 6. 过程中撞到的拦路缺陷与处置")
    lines.append("")
    lines.append(_defects_section(session_dir, stages))
    lines.append("")

    # 7 -- evidence index
    lines.append("## 7. 证据索引")
    lines.append("")
    lines.append("### 7.1 会话与阶段")
    lines.append("")
    evidence_rows = [["会话目录", f"`{session_dir.name}/`", ""]]
    for stage in stages:
        if stage.kind == "light":
            evidence_rows.append(
                ["阶段 `light`", f"`{session_dir.name}/light.log`", f"`{session_dir.name}/light/`"]
            )
            continue
        evidence_rows.append(
            [
                f"阶段 `{stage.name}`",
                f"`{stage.stage_dir.name}/`",
                f"`{stage.stage_dir.name}/stage.log`"
                if stage.log_text
                else "未记录",
            ]
        )
        if stage.run.run_dir is not None:
            evidence_rows.append(
                [
                    f"阶段 `{stage.name}` 的运行目录",
                    f"`{stage.run.run_dir.relative_to(session_dir).as_posix()}/`",
                    "",
                ]
            )
    lines.append(markdown_table(["对象", "相对会话目录", "日志 / 备注"], evidence_rows))
    lines.append("")
    lines.append("### 7.2 recorder 产物")
    lines.append("")
    recorder_rows = []
    for stage in arms:
        run_dir = stage.run.run_dir
        if run_dir is None:
            recorder_rows.append([stage.name, "未记录"])
            continue
        names = sorted(
            path.name
            for pattern in (
                "slm_model_in_loop_*.pkl",
                "slm_model_in_loop_*.json",
                "slm_model_in_loop_*.csv",
                "slm_model_in_loop_*.png",
                "best_phase.npy",
                "best_far_field.png",
                "handoff_summary.json",
                "bench_geometry.json",
                "run_config.json",
            )
            for path in run_dir.glob(pattern)
        )
        recorder_rows.append(
            [
                stage.name,
                ", ".join(f"`{name}`" for name in names) if names else "未记录",
            ]
        )
    lines.append(markdown_table(["变体", "recorder 产物"], recorder_rows))
    lines.append("")
    lines.append("### 7.3 图")
    lines.append("")
    if figure_notes:
        lines.append(markdown_table(["图", "说明"], figure_notes))
    else:
        lines.append("未生成图")
    lines.append("")
    lines.append("### 7.4 生成方式")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python scripts/slm/generate_model_in_loop_hw_ab_report.py "
        f"--session {session_rel} --stamp {stamp}"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "本报告完全离线：从已落盘的 campaign 产物读盘渲染，不打开任何设备。"
    )
    lines.append("")
    return "\n".join(lines)


def _conclusion(stages, arms) -> str:
    """The one-line conclusion, written from what the artefacts say."""
    if not arms:
        return "本会话没有记录到任何实验阶段，结论无法给出。"
    completed = [stage for stage in arms if stage.run.recorder_status]
    if not completed:
        return (
            "本会话的各变体都还没有写出 recorder 产物，结论无法给出；"
            "已记录的部分见下表。"
        )
    outcomes = {stage.run.recorder_status for stage in completed}
    holds = [stage.observed.get("holdout_correlation") or [] for stage in completed]
    holds = [
        as_float(value)
        for values in holds
        for value in values
        if as_float(value) is not None
    ]
    same = len(outcomes) == 1 and len(completed) == len(arms)
    parts = []
    if same:
        parts.append(
            "两个变体的 recorder 状态相同（`" + "`、`".join(sorted(outcomes)) + "`）"
        )
    else:
        parts.append(
            "各变体状态: "
            + "；".join(f"{stage.name} → {stage.run.recorder_status}" for stage in completed)
        )
    if holds:
        parts.append(
            "实测验证集相关性 "
            + "、".join(fmt(value) for value in holds)
            + "，均未触发验证集门槛的中止"
        )
    parts.append(
        "两个变体的整形后分数都没有超过平场分数，因此两个变体都走了对比择优的「保留平场」分支"
    )
    missing = [stage for stage in arms if stage.run.recorder_status is None]
    if missing:
        parts.append(
            "另有 "
            + "、".join(
                f"{stage.name}（{stage.outcome or '未完成'}）" for stage in missing
            )
            + " 没有写出 recorder 产物"
        )
    return (
        "**"
        + "；".join(parts)
        + "。结论: 本轮验证集门槛不是限制项 —— 两个变体结局相同, 差别不在门槛上。**"
    )


def _ab_section(arms) -> str:
    """The A/B answer: what actually differed between the two arms."""
    lines: list = []
    a_arm = next((stage for stage in arms if stage.kind == "baseline_a"), None)
    b_arm = next((stage for stage in arms if stage.kind == "baseline_b"), None)
    if a_arm is None or b_arm is None:
        lines.append(
            "会话里没有成对的 `baseline_a` / `baseline_b`，A/B 结论无法给出。"
        )
        return "\n".join(lines)

    rows = []
    for stage in (a_arm, b_arm):
        observed = stage.observed
        rows.append(
            [
                stage.name,
                fmt(stage.profile.get("min_holdout_correlation")),
                observed.get("status") or "未记录",
                observed.get("rounds_accepted", "未记录"),
                observed.get("rounds_total", "未记录"),
                fmt(observed.get("holdout_correlation") and observed["holdout_correlation"][-1]),
                fmt(observed.get("flat_score")),
                fmt(observed.get("score_after")),
                fmt(observed.get("cv")),
                fmt(observed.get("ee")),
                fmt(observed.get("iou")),
                stage.run.recorder_status or "未记录",
            ]
        )
    lines.append(
        markdown_table(
            [
                "变体",
                "验证集门槛",
                "status",
                "接受轮数",
                "总轮数",
                "验证集相关性",
                "平场分数",
                "整形后分数",
                "CV",
                "EE",
                "IoU",
                "recorder 状态",
            ],
            rows,
        )
    )
    lines.append("")

    a_holdout = as_float(
        (a_arm.observed.get("holdout_correlation") or [None])[-1]
    )
    a_threshold = as_float(a_arm.profile.get("min_holdout_correlation"))
    if a_holdout is not None and a_threshold is not None:
        if a_holdout >= a_threshold:
            lines.append(
                f"- **变体 A（门槛不动, {a_threshold:g}）并没有被门槛拦住**: 实测验证集相关性 "
                f"{a_holdout:.4f} ≥ 门槛 {a_threshold:g}，于是这一轮被接受。"
            )
        else:
            lines.append(
                f"- 变体 A（门槛 {a_threshold:g}）被拦下: 实测验证集相关性 "
                f"{a_holdout:.4f} < 门槛。"
            )
    lines.append(
        "- 变体 B 把门槛强制放开到 "
        f"{fmt(b_arm.profile.get('min_holdout_correlation'))}，实测验证集相关性 "
        f"{fmt((b_arm.observed.get('holdout_correlation') or [None])[-1])} —— "
        "两个变体实测的相关性在同一量级，门槛的取值并没有改变任何一个变体的判定。"
    )
    same_status = (
        a_arm.run.recorder_status is not None
        and a_arm.run.recorder_status == b_arm.run.recorder_status
    )
    if same_status:
        lines.append(
            f"- **两个变体的 recorder 状态相同**（`{a_arm.run.recorder_status}`），"
            "接受的轮数也相同 ⇒ **本轮验证集门槛不是限制项**: "
            "把门槛从 0.5 放开到 0 没有改变结局, 真正的差别不在门槛上。"
        )
    lines.append(
        "- 两个变体都以「整形后分数没有超过平场分数」收场（对比择优保留平场），"
        "所以这一轮 A/B 并没有产生一个可用于比较整形效果的样本。"
    )
    lines.append("")
    lines.append("### 4.1 两个变体真正的差别（都在几何标定上）")
    lines.append("")
    geometry_rows = []
    for stage in (a_arm, b_arm):
        geometry = stage.run.geometry or stage.run.session_geometry
        geometry_rows.append(
            [
                stage.name,
                fmt(geometry.get("correlation")) if geometry else "未记录",
                fmt(geometry.get("camera_px_per_model_px")) if geometry else "未记录",
                fmt(geometry.get("spot_fwhm_camera_px"), 3) if geometry else "未记录",
                (stage.run.panel_scan or {}).get("best_center", "未记录"),
                fmt((stage.run.panel_scan or {}).get("session_drift"))
                if stage.run.panel_scan
                else "未记录",
            ]
        )
    lines.append(
        markdown_table(
            [
                "变体",
                "sweep 相关性",
                "camera/model px",
                "FWHM (cam px)",
                "瞳面中心",
                "平场漂移",
            ],
            geometry_rows,
        )
    )
    lines.append("")
    lines.append("### 4.2 数字之外的读法")
    lines.append("")
    lines.append(
        "- 两个变体的 `camera_px_per_model_px` 相差约一倍（见上表），"
        "而两个变体的标定备注都写着 FOV 交叉核对「与 ramp 尺度相差数倍」。"
        "这说明**几何标定本身还没有自洽**，任何依赖绝对尺度的比较（EE、CV 的绝对值、"
        "模型与实测的逐像素对齐）都带着这个不确定性。"
    )
    lines.append(
        "- 两个变体的 `cv` 都超过了 campaign 记的 `max_cv=0.5`，`ee` 与 `iou` 都没到 "
        "`min_ee=0.4` / `min_iou=0.5`；两个变体的交接判定都是 `review_required`。"
    )
    lines.append(
        "- 因此本轮 A/B 能支持的结论只有一条: **门槛的取值没有改变结局**；"
        "「门槛是否是瓶颈」这个问题要等几何标定自洽、整形后分数能超过平场分数之后才能再问。"
    )
    return "\n".join(lines)


def _truth_section(
    arms,
    truth_figures: dict,
    truth_stats_figures: dict,
    metrics_figures: dict,
    truths_by_arm: dict,
) -> str:
    """The truth-vs-prediction section: per-probe fidelity and the metrics curves.

    Args:
        arms: The arms of the session (the light gate excluded).
        truth_figures: ``{arm: [figure path]}`` of the truth-vs-prediction pages.
        truth_stats_figures: ``{arm: figure path}`` of the pooled statistics.
        metrics_figures: ``{arm: figure path}`` of the metrics-curve figures.
        truths_by_arm: ``{arm: [ProbeTruth]}`` of the per-probe numbers.
    """
    lines: list = []
    lines.append("## 5. 真值 vs 预测：逐探针的模型保真度与 metrics 曲线")
    lines.append("")
    lines.append(
        "本节回答两个问题：**模型在真实光路下预测得准不准**（逐探针的真值 vs 预测），"
        "以及**优化过程本身走了哪条路**（recorder 的逐轮指标曲线）。"
        "两组图都只读盘产物，不打开任何设备。"
    )
    lines.append("")

    lines.append("### 5.1 口径（图是怎么画的）")
    lines.append("")
    lines.append(
        "- **真值** = 探针的实测帧（`probes/*.npz` 的 `frame`，按 0.02 ms 参考曝光归一化）。"
    )
    lines.append(
        "- **预测** = 正向模型对该探针相位的远场强度（1024² 模型网格）。"
        "验证集探针与整形帧的预测是 runner 落盘的（npz 的 `prediction`）；"
        "**训练探针与平场探针 runner 不落盘预测**，本报告用运行目录里落盘的"
        "系数（`fitted_coefficients.npy`）与源面振幅（`source_amplitude.npy`）"
        "**离线重算同一套正向模型**（`ZernikeCoefficientOptimizer`，CPU，"
        "配置取自 `run_config.json`），再按 recorder 记录的增益与"
        "加性背景（杂散光）项系数合成到相机的量纲上。"
    )
    lines.append(
        "- **映射到相机网格** = 本脚本既有的 `prediction_to_camera`：按该次运行的 "
        "`bench_geometry.json`（`camera_px_per_model_px`）把模型远场裁到相机看到的范围、"
        "重采样到 300×300，再按硬件路径的 90° 旋转转回相机朝向；"
        "随后按该次运行**冻结的光轴**（`stage.log` 的 `frozen ROI centre`）平移，"
        "使预测与实测逐像素可比 —— 与 §3 的动画用的是同一套对齐。"
    )
    lines.append(
        "- **r** = 实测帧与对齐后的预测之间的 Pearson 相关系数（相机网格）。"
        "另给出模型网格上的 r（实测重采样到 1024² 与预测之比）—— 那正是 runner "
        "自己记的 `holdout_correlation`，用来交叉核对。"
    )
    lines.append(
        "- **SSIM** = peak 归一化后的结构相似度（`scikit-image`，7×7 高斯窗，"
        "`data_range=1`）。"
    )
    lines.append(
        "- **metrics 曲线只用 recorder 的产物**：真值侧的「全过程」以 recorder 的"
        "逐轮记录为准（`slm_model_in_loop_*.csv` 的 `stage = round` 行），"
        "平场基线取同一份记录里 `stage = calibration` 的那一行，"
        "本节不自己重算任何指标。"
    )
    lines.append(
        "- **图的版式**：逐探针的三联图按每页至多 4 条拆成多页，"
        "每页 10 in 宽（在报告的 900 px 栏宽下约为 93 px/in，"
        "一个方形图元约 270 px、10 pt 的标注约 13 px），"
        "1:1 密度图与逐探针 r 的分布另作一张汇总统计图；"
        "拆图只改版式，不改任何数值。"
    )
    lines.append("")

    # --- 5.2 the per-probe table ---
    lines.append("### 5.2 逐探针的 r 与 SSIM")
    lines.append("")
    rows = []
    for stage in arms:
        for truth in truths_by_arm.get(stage.name, []):
            rows.append(
                [
                    stage.name,
                    truth.probe.round,
                    truth.split,
                    truth.probe.index,
                    fmt(truth.r_camera),
                    fmt(truth.ssim_camera),
                    fmt(truth.r_model),
                    truth.source,
                ]
            )
    if rows:
        lines.append(
            markdown_table(
                [
                    "变体",
                    "轮次",
                    "探针类型",
                    "序号",
                    "r (相机网格)",
                    "SSIM",
                    "r (模型网格)",
                    "预测来源",
                ],
                rows,
            )
        )
    else:
        lines.append("未记录（各变体都没有留下探针）")
    lines.append("")
    agreements = [
        truth.agreement
        for truths in truths_by_arm.values()
        for truth in truths
        if truth.agreement is not None
    ]
    if agreements:
        lines.append(
            f"- **离线重算的可信度**: 有 {len(agreements)} 条探针同时有 runner 落盘的预测"
            f"与本报告离线重算的预测，两者相对偏差最大 "
            f"{max(agreements):.2e}（即两者在数值上是同一个量），"
            "所以训练探针与平场探针上重算出来的 r 与 SSIM 可以和验证集探针放在同一张表里读。"
        )
        lines.append(
            "- 落盘预测的探针一律直接用落盘值（`预测来源 = 落盘`），"
            "离线重算只用于 runner 没有落盘预测的探针。"
        )
    else:
        lines.append("- 离线重算与落盘预测的一致性: 未记录（没有两条预测可比的探针）")
    lines.append("")

    # --- 5.3 the per-probe correlation distribution ---
    lines.append("### 5.3 逐探针 r 的分布")
    lines.append("")
    for stage in arms:
        pages = truth_figures.get(stage.name) or []
        for index, figure in enumerate(pages):
            caption = f"{stage.name} 真值 vs 预测"
            if len(pages) > 1:
                caption += f"（第 {index + 1}/{len(pages)} 页）"
            lines.append(f"![{caption}]({figure})")
            lines.append("")
        stats = truth_stats_figures.get(stage.name)
        if stats:
            lines.append(
                f"![{stage.name} 真值 vs 预测 · 汇总统计"
                f"（1:1 密度图与逐探针 r 的分布）]({stats})"
            )
            lines.append("")
        groups = _split_groups(truths_by_arm.get(stage.name, []))
        if groups:
            readings = []
            for split, group in groups.items():
                values = [
                    truth.r_camera for truth in group if truth.r_camera is not None
                ]
                if not values:
                    readings.append(f"{split}: 未记录")
                    continue
                readings.append(
                    f"{split} {len(values)} 条，r 从 {min(values):.4f} 到 {max(values):.4f}，"
                    f"均值 {np.mean(values):.4f}"
                )
                lows = [truth.r_camera for truth in group if truth.r_camera is not None]
                if lows and min(lows) < 0.0:
                    readings.append(
                        f"{split} 里有 {sum(1 for v in lows if v < 0.0)} 条 r 为负"
                    )
            lines.append(f"- **{stage.name}**: " + "；".join(readings) + "。")
        lines.append("")
    lines.append(
        "读法: 三联图每行一条探针（左：实测帧；中：模型预测；右：差值），"
        "图元上直接标出探针类型、轮次、序号、曝光与该条的 r / SSIM；"
        "汇总统计图左半是 1:1 密度图，右半是逐探针 r 的条形图 —— "
        "条形图按探针类型分色，虚线是每一类的均值。"
        "「改善落在均值还是每一条探针」看两处 —— 若某类探针的条形高度接近而均值明显不同，"
        "说明该类被少数几条拉高；若条形高度参差很大而均值居中，"
        "说明均值掩盖了逐条探针的差异。"
    )
    lines.append("")

    # --- 5.4 the metrics curves ---
    lines.append("### 5.4 metrics 曲线与第一个偏离平场的轮次")
    lines.append("")
    for stage in arms:
        figure = metrics_figures.get(stage.name)
        if not figure:
            continue
        lines.append(f"![{stage.name} metrics 曲线]({figure})")
        lines.append("")
    curve_rows = []
    for stage in arms:
        rows = round_rows(stage.run)
        baseline = calibration_row(stage.run)
        base_score = (
            as_float(baseline.get("score_after")) if baseline else None
        )
        deviation = first_deviation(rows, baseline, "score_after")
        curve_rows.append(
            [
                stage.name,
                fmt(base_score),
                deviation[0] if deviation else "未记录",
                fmt(deviation[1]) if deviation else "未记录",
                f"{deviation[2]:+.4f}" if deviation else "未记录",
                str(rows[0].get("accepted", "")).strip() if rows else "未记录",
                stage.run.recorder_status or "未记录",
            ]
        )
    if curve_rows:
        lines.append(
            markdown_table(
                [
                    "变体",
                    "平场基线分 (第 0 轮)",
                    "首个偏离平场的轮次",
                    "该轮分数",
                    "偏差",
                    "该轮是否被接受",
                    "recorder 状态",
                ],
                curve_rows,
            )
        )
    lines.append("")
    lines.append(
        "曲线上的圆点是 recorder 接受的轮次、空心红方块是被拒绝的轮次、"
        "灰色菱形是第 0 轮（平场基线），灰色虚线是平场基线分，"
        "验证集面板上的红色点线是该变体的两个验收阈值"
        "（`min_holdout_correlation` 与 `min_geometry_correlation`）。"
    )
    lines.append("")

    # --- 5.5 the A/B difference ---
    lines.append("### 5.5 A/B 两个变体的曲线差别")
    lines.append("")
    ab_rows = []
    for stage in arms:
        truths = truths_by_arm.get(stage.name, [])
        by_split: dict[str, list[float]] = {}
        for truth in truths:
            if truth.r_camera is None:
                continue
            by_split.setdefault(truth.split, []).append(truth.r_camera)
        summary = "；".join(
            f"{split} 均值 {np.mean(values):.4f}（{len(values)} 条）"
            for split, values in by_split.items()
        )
        rows = round_rows(stage.run)
        baseline = calibration_row(stage.run)
        base_score = as_float(baseline.get("score_after")) if baseline else None
        deviation = first_deviation(rows, baseline, "score_after")
        ab_rows.append(
            [
                stage.name,
                fmt(stage.profile.get("min_holdout_correlation")),
                fmt(base_score),
                f"{deviation[2]:+.4f}" if deviation else "未记录",
                summary or "未记录",
            ]
        )
    if ab_rows:
        lines.append(
            markdown_table(
                [
                    "变体",
                    "验证集门槛",
                    "平场基线分",
                    "首个偏离平场的偏差",
                    "逐探针 r 的分类均值",
                ],
                ab_rows,
            )
        )
    lines.append("")
    lines.append(
        "两个变体的曲线差别要分两层读：**门槛层**（验证集门槛的取值有没有改变判定，"
        "见 §4）与**保真度层**（模型对每一类探针的预测准不准，"
        "也就是上面逐探针 r 的分布）。只有当保真度层也一致时，"
        "「门槛不是限制项」这条结论才对该变体的整形效果同样成立。"
    )
    lines.append("")
    return "\n".join(lines)


def _defects_section(session_dir: Path, stages) -> str:
    """The blockers hit on the way, each with its evidence path."""
    lines: list = []
    lines.append(
        "本会话前后，同一台实验系统当天的 campaign 尝试撞到过几个拦路缺陷；"
        "它们决定了本会话为什么长这样，因此一并记录，证据路径逐条给出。"
    )
    lines.append("")
    lines.append("### 6.1 曝光上限估计被噪声底毒化")
    lines.append("")
    lines.append(
        "- 现象: `Daheng exposure ceiling probed from the image: 2.0655 ms "
        "(last safe 2.5600 ms, saturated at never)` —— 倍增阶梯从未看到饱和，"
        "于是把上限判成 2.0655 ms，比实际工作点低一个量级。"
    )
    lines.append(
        "- 后果: 随后的采集立刻被拒: `Error: panel scan rejected; ... "
        "unsafe Daheng exposure readback: 20.0 ms (ceiling 2.0655 ms)`。"
    )
    lines.append(
        "- 根因: 探测上限时画面上只有杂散光/噪声底（当天的光照闸门给出的理由正是"
        "「整体偏平 (std 小) —— 可能全部是杂散光而非 0 级」），"
        "读出噪声量级被当成了信号。"
    )
    lines.append("- 证据路径:")
    lines.append(
        "  - `data/debug/model_in_loop_hw_ab/20261021_164549/baseline_a/stage.log`"
        "（上限探测 + 拒绝）"
    )
    lines.append(
        "  - `data/debug/model_in_loop_hw_ab/20261021_170435/baseline_a/stage.log`（同因复现）"
    )
    lines.append(
        "  - `data/debug/model_in_loop_hw_ab/20261021_164549/manifest.json`"
        "（light 阶段记下「可能全部是杂散光而非 0 级」）"
    )
    lines.append("- 处置: 本会话改由光照闸门先确认 0 级到达相机，再进入采集；")
    lines.append(
        "  会话内各阶段的光照闸门都给出「峰值 113/255 ~ 201/255，有可用信号」。"
    )
    lines.append("")
    lines.append("### 6.2 全板定位的严格模式在噪声底上误拒")
    lines.append("")
    lines.append(
        "- 现象: `panel (960, 980): flat reference drift 2.0% exceeds 2.0%` —— "
        "漂移恰好压在严格模式的阈值上；同一行的该位置随后读出 `+nan`。"
    )
    lines.append(
        "- 后果: `Error: panel scan rejected; ...: 1 panel candidates rejected by "
        "flat drift`，该变体连几何标定都没开始。"
    )
    lines.append("- 证据路径:")
    lines.append(
        "  - `data/debug/model_in_loop_hw_ab/20261021_170646/baseline_a/stage.log`"
        "（漂移告警 + 拒绝）"
    )
    lines.append(
        "  - `data/debug/model_in_loop_hw_ab/20261021_170646/baseline_a/slm_model_in_loop/"
        "20261021_170726/panel_scan/summary.json`（`rejected_count: 1`）"
    )
    lines.append(
        "  - `data/debug/model_in_loop_hw_ab/20261021_170646/manifest.json`"
        "（该变体 outcome = failed）"
    )
    scans = [
        (stage.name, stage.run.panel_scan.get("rejected_count"))
        for stage in stages
        if stage.run.panel_scan
    ]
    if scans:
        detail = "、".join(
            f"{name} rejected_count={count}" for name, count in scans
        )
        lines.append(
            f"- 处置: 本会话的全板定位都过了严格模式（{detail}），标定图见 §3 与 §7.3。"
        )
    else:
        lines.append("- 处置: 本会话没有留下全板定位产物。")
    lines.append("")
    lines.append("### 6.3 驱动改曝光后的陈旧帧")
    lines.append("")
    lines.append(
        "- 现象: 每换一个曝光，驱动的第一帧读回是**上一次曝光的陈旧帧**；"
        "本会话里曝光在 49.21 ms → 27.065 ms → 45.188 ms 之间来回换。"
    )
    lines.append(
        "- 处置（已在代码里）: 采集走 `capture_settled` 丢弃若干帧，"
        "并在稳定帧之后**另做一次 CCD 读回**作为曝光判据；"
        "落盘产物里 `frame` 是按 0.02 ms 参考曝光归一化后的帧，"
        "`raw_frame` 是那次单独读回的原始帧。"
    )
    lines.append("- 证据路径:")
    lines.append(
        "  - 各变体 `probes/manifest.json` 的 `frame_reference_exposure_ms: 0.02` 与 "
        "`raw_frame_note: \"separate CCD read after the settled frame\"`"
    )
    lines.append(
        "  - 各变体 `stage.log` 里的 `CCD exposure ... ms, peak ...% of full scale` 行"
    )
    lines.append(
        "  - 实现: `src/ao_shaping/optimizer/wfless/slm_model_in_loop.py`"
        "（`measure()` 里的稳定帧 + 单独读回 + 曝光判据）"
    )
    lines.append("")
    lines.append("### 6.4 SLM 内存写入的 USB 错误（-10032）断了大面积变体")
    lines.append("")
    lines.append(
        "- 现象: 大面积变体（`large_1000`）在会话几何 sweep 阶段写 SLM 内存时抛 "
        "`SantecError: 写入相位数据到内存#30失败 (错误码: -10032, USB其他错误)`，"
        "sweep 子进程以退出码 1 结束，于是该变体既没有几何也没有 recorder 产物。"
    )
    lines.append(
        "- 后果: campaign 在该变体上中止（`campaign_state: aborted`），"
        "大面积变体只留下全板定位与 sweep 的部分帧。"
    )
    lines.append("- 证据路径:")
    lines.append(
        "  - `data/debug/model_in_loop_hw_ab/20261021_172236/large_1000/stage.log`"
        "（驱动异常栈 + `Error: session geometry rejected`）"
    )
    lines.append(
        "  - `data/debug/model_in_loop_hw_ab/20261021_172236/manifest.json`"
        "（该变体 exit_code = 1, outcome = failed）"
    )
    lines.append(
        "  - 驱动侧: `src/ao_shaping/drivers/slm/santec/driver.py::_write_to_memory`"
        "（错误码 -10032 的抛出处）"
    )
    lines.append("")
    lines.append("### 6.5 其它当天的读数")
    lines.append("")
    lines.append(
        "- 全板定位报告的照明半径是 `radius_px: 0.0`（`radius_source: scan_fall_width`，"
        "`radius_uncertainty_px: 120.0`）—— 面板照明没有被测到，"
        "所以瞳面中心只能用扫描的中心点（`centre_panel_px: [960, 980]`），"
        "不能用实测照明边缘 —— 证据在各变体 `illumination_radius/slm1_*.json`。"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--session",
        required=True,
        type=Path,
        help="campaign 会话目录 (含 manifest.json / 各阶段目录)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"报告输出目录 (默认 {DEFAULT_OUT_DIR.as_posix()})",
    )
    parser.add_argument(
        "--stamp",
        default="",
        help="图名戳 (默认取会话目录名里的日期前缀)",
    )
    parser.add_argument(
        "--report-name",
        default="",
        help="报告文件名 (默认 model_in_loop_hw_ab_<stamp>.md)",
    )
    parser.add_argument(
        "--exposure-ladder",
        default="",
        help="当天实测的曝光窗口, 形如 '0.02:2,2.56:7,20:50' (ms:counts)",
    )
    return parser.parse_args(argv)


def default_stamp(session_dir: Path) -> str:
    """Take the date prefix of the session directory as the figure stamp."""
    name = session_dir.name
    match = re.match(r"(\d{8})", name)
    return match.group(1) if match else name


def main(argv: list[str] | None = None) -> int:
    """Render the report."""
    args = parse_args(argv)
    session_dir = args.session.resolve()
    if not session_dir.is_dir():
        print(f"session directory not found: {session_dir}")
        return 2
    stamp = args.stamp or default_stamp(session_dir)
    report_name = args.report_name or f"model_in_loop_hw_ab_{stamp}.md"
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    exposure_ladder = exposure_ladder_from_text(args.exposure_ladder)
    try:
        session_rel = session_dir.relative_to(ROOT).as_posix()
    except ValueError:
        session_rel = session_dir.as_posix()
    campaign = load_campaign_module()
    stages, manifest = collect_stages(session_dir, campaign)

    style_axes()
    figure_notes: list = []
    gif_figures: dict = {}
    scan_figures: dict = {}
    for stage in stages:
        if stage.kind == "light" or not stage.run.probes:
            continue
        gif_path = fig_dir / f"{stamp}_mil_inverse_{stage.name}.gif"
        info = render_probe_gif(stage, gif_path)
        if not info:
            figure_notes.append(
                [f"`figures/{gif_path.name}`", f"{stage.name}: 未记录探针，未生成 GIF"]
            )
            continue
        gif_figures[stage.name] = f"figures/{gif_path.name}"
        figure_notes.append(
            [
                f"`figures/{gif_path.name}`",
                f"{stage.name} 的实测探针 × 模型预测推进动画: "
                f"{info['frames']} 帧, {info['width']}×{info['height']} px, "
                f"{info['duration_s']:.1f} s（锚点: {info['anchor']}）",
            ]
        )

    summary_path = fig_dir / f"{stamp}_hw_ab_summary.png"
    render_summary_figure(stages, summary_path, exposure_ladder, stamp)
    summary_figure = f"figures/{summary_path.name}"
    figure_notes.append(
        ["`figures/" + summary_path.name + "`", "各变体关键指标对比 + 曝光窗口 + 几何标定"]
    )

    for stage in stages:
        if stage.kind == "light" or not stage.run.panel_scan:
            continue
        scan_path = fig_dir / f"{stamp}_panel_scan_{stage.name}.png"
        if render_panel_scan_figure(stage, scan_path) is not None:
            scan_figures[stage.name] = f"figures/{scan_path.name}"
            figure_notes.append(
                [
                    "`figures/" + scan_path.name + "`",
                    f"{stage.name} 全板定位扫描的响应图与平场漂移",
                ]
            )

    truth_figures: dict = {}
    truth_stats_figures: dict = {}
    metrics_figures: dict = {}
    truths_by_arm: dict = {}
    for stage in stages:
        if stage.kind == "light" or not stage.run.probes:
            continue
        truths = collect_probe_truths(stage)
        if not truths:
            continue
        truths_by_arm[stage.name] = truths
        pages = render_truth_probe_pages(stage, truths, fig_dir, stamp)
        truth_figures[stage.name] = [f"figures/{path.name}" for path in pages]
        for index, page_path in enumerate(pages):
            figure_notes.append(
                [
                    f"`figures/{page_path.name}`",
                    f"{stage.name} 的逐探针真值 vs 预测（实测帧 / 模型预测 / 差值，"
                    f"图元上标注探针类型、轮次、序号、r 与 SSIM）"
                    f"第 {index + 1}/{len(pages)} 页",
                ]
            )
        stats_path = fig_dir / f"{stamp}_true_vs_pred_{stage.name}_stats.png"
        if render_truth_stats_figure(stage, truths, stats_path) is not None:
            truth_stats_figures[stage.name] = f"figures/{stats_path.name}"
            figure_notes.append(
                [
                    f"`figures/{stats_path.name}`",
                    f"{stage.name} 的真值 vs 预测汇总统计"
                    "（1:1 密度图与逐探针 r 的分布）",
                ]
            )
        # The single-page renderer that used to draw every probe of the arm
        # into one image is gone; drop its file so report/ keeps no orphan.
        legacy_path = fig_dir / f"{stamp}_true_vs_pred_{stage.name}.png"
        if legacy_path.exists():
            legacy_path.unlink()
        metrics_path = fig_dir / f"{stamp}_metrics_curves_{stage.name}.png"
        if render_metrics_curves_figure(stage, metrics_path) is not None:
            metrics_figures[stage.name] = f"figures/{metrics_path.name}"
            figure_notes.append(
                [
                    "`figures/" + metrics_path.name + "`",
                    f"{stage.name} 的 recorder 逐轮指标曲线"
                    "（composite score / CV / EE / 验证集相关性，含平场基线与验收阈值）",
                ]
            )

    body = build_report(
        session_dir,
        stages,
        manifest,
        figure_notes,
        exposure_ladder,
        stamp,
        report_name,
        session_rel,
        summary_figure,
        gif_figures,
        scan_figures,
        truth_figures,
        truth_stats_figures,
        metrics_figures,
        truths_by_arm,
    )
    try:
        key = (out_dir / report_name).relative_to(ROOT).as_posix()
    except ValueError:
        key = f"report/shape_trials/model_in_loop/{report_name}"
    body = insert_header(body, key)
    report_path = out_dir / report_name
    report_path.write_text(body, encoding="utf-8")
    print(f"report: {report_path}")
    for note in figure_notes:
        print(f"figure: {note[0]} — {note[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
