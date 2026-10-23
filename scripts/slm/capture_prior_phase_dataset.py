"""Capture a prior-phase training set: phase -> measured CCD far field.

Why this exists
---------------
The audit of the historical MIL runs (``logs/mil_loss_audit/``) concluded that
the closed loop was limited by **forward-model mismatch**, not by a shortage of
iterations: the model error is strongly correlated (51/51 same sign) and is
mostly a scale/width error.  A model calibrated on random probes alone cannot
see that, because random probes never visit the part of the phase space where
the model is wrong.

This script therefore measures **two kinds of phase in one session** on the
same bench:

* the random probes of the baseline capture
  (``scripts/slm/capture_defocus_forward_dataset.py``) -- loaded by path so the
  phases are **bit-identical** for the same ``--seed``/``--region``, which makes
  the two datasets pair sample by sample;
* the **prior phases** -- phases that a historical run actually drove the loss
  down, taken from the audit's ``candidates.json``.

Both are measured on the current plane (``U``) and on the defocus-compensated
plane (``C``), so each pair isolates the defocus term.

Output format
-------------
Byte-compatible with the baseline capture, so the fitting side
(``fit_defocus_forward_dataset.py``) consumes it unchanged: ``run_config.json``,
``focus_phase.npy``, ``sample_%04d.npz`` (``frame`` / ``raw_frames`` /
``panel_phase`` / ``model_phase``), ``manifest.json`` and
``capture_records.pkl``.

Resampling loss
---------------
A prior phase recorded on a 64-cell grid cannot be shown on a 128-cell grid
without interpolation.  The script resamples it with
``scipy.ndimage.zoom(..., order=1)`` (linear interpolation) and records
``resampled_from_region`` in the manifest row.  Linear interpolation is
**lossy** -- it smooths the phase, so the resampled phase is *not* the phase
that was validated in the historical run.  Those samples are therefore
"neighbourhood of a validated phase" data, not exact reproductions.

中文：先验相位从 64 格重采样到 128 格用的是线性插值，属于**有损**操作；重采样后
的相位并不是历史运行里实测验证过的那一张相位，只能当作该相位附近的近似样本使用。

Defaults to a device-free plan: the plan is printed as JSON plus a capture
list, and nothing is opened.  ``--execute`` opens the bench.

Cadence and exposure
--------------------
Two options trade monitoring density against wall-clock time.  Both default to
the behaviour that produced the measured B2 run, so a default run stays
comparable with it.

``--reference-every N`` (default 6)
    A pair of reference captures (one per plane) is inserted after every ``N``
    content captures.  One counter covers both families: a random probe adds
    one unit, a prior phase adds :data:`PRIOR_PROBE_EQUIVALENTS` units because
    it is a shaping phase and perturbs the panel state far more than a random
    probe.  With the default ``N=6`` that reproduces the measured cadence
    exactly -- a reference pair after every sixth random probe **and** after
    every prior phase -- so a default run stays comparable with the B2
    dataset.

    **The interval has to be chosen per drift rate**: the B2 run measured the
    compensated plane (``C``) at 0.014-0.034 px/min of centroid drift and a
    0.6-2.6 % FWHM span over 28.5 min, so its interval can be stretched to a
    dozen captures; the uncompensated plane (``U``) drifted at 1.0-1.35 px/min
    with a 68-71 % FWHM span, so it needs the denser monitoring.  A reference
    pair covers both planes at once, so the cadence is a compromise -- pick it
    from the drift of the plane you care about.

``--fixed-exposure`` (default off)
    Pins the exposure of each plane to the value the exposure ladder picked the
    first time that plane was measured, and stops running the ladder.  That
    removes the ladder's settle from every capture (about 5 s each in the B2
    run) and it is what makes the peak a usable drift monitor: with an
    adaptive exposure the peak moves because the exposure moved, not because
    the light did.

    The price: a fixed exposure no longer tracks the light level per capture,
    so it may only be enabled on the compensated plane (or once the light
    level is known to be stable).  Data taken with a fixed exposure cannot be
    compared with adaptively exposed data on an absolute intensity scale --
    normalise each set on its own.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.utils.slm.panel_geometry import phase_to_panel

#: The baseline capture whose ``probe_grid`` is reused verbatim.  Loading it by
#: path (instead of copying the function) is what makes the random probes
#: bit-identical to the baseline group, i.e. a paired A/B measurement.
BASELINE_SCRIPT = Path(__file__).resolve().parent / "capture_defocus_forward_dataset.py"

#: Keys every manifest row must carry (the format contract shared with the
#: baseline group).
MANIFEST_KEYS = (
    "index",
    "file",
    "label",
    "branch",
    "family",
    "split",
    "exposure_ms",
    "exposure_status",
    "settled",
    "utc_start",
    "timestamp_utc",
    "monotonic_start_ns",
    "monotonic_end_ns",
    "peak",
    "spot",
)

#: Keys every ``sample_%04d.npz`` must carry.
NPZ_KEYS = ("frame", "raw_frames", "panel_phase", "model_phase")

#: Split rule of the baseline group: probes 0,1,2 -> train, 3,4,5 -> test, ...
PROBE_TEST_MODULUS = 6
PROBE_TEST_OFFSET = 3

#: Default cadence of the drift references: one reference pair after every
#: ``REFERENCE_EVERY`` content captures.
REFERENCE_EVERY = 6

#: How many probe-equivalents one prior phase counts as in the reference
#: counter.  A prior phase is a shaping phase: it perturbs the panel state far
#: more than a random probe, and the measured B2 run gave every one of them
#: its own reference pair.  Expressing that as a weight keeps **one** counter
#: for both families, so ``--reference-every`` means the same interval
#: everywhere while the default stays bit-identical to the B2 cadence.
PRIOR_PROBE_EQUIVALENTS = 6


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------


def load_module_by_path(name: str, path: Path) -> Any:
    """Import a sibling script by file path.

    ``sys.modules`` must be populated **before** ``exec_module``: a module-level
    ``@dataclass`` resolves the classes it creates through
    ``sys.modules[cls.__module__].__dict__``, and that lookup fails with
    ``NoneType object has no attribute '__dict__'`` when the module was never
    registered.  Registering first is what keeps such a module loadable.

    Args:
        name: Name to register the module under.
        path: File to load.

    Returns:
        The loaded module object.

    Raises:
        ImportError: The path cannot be turned into an import spec.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_probe_grid(path: Path = BASELINE_SCRIPT) -> Callable[..., tuple[str, np.ndarray]]:
    """Return the baseline ``probe_grid`` so the random probes stay identical.

    Raises:
        ImportError: The baseline script is missing or does not define
            ``probe_grid``.  There is deliberately **no** fallback: silently
            switching to another phase family would break the pairing with the
            baseline group while looking like a successful run.
    """
    if not path.is_file():
        raise ImportError(f"baseline capture script not found: {path}")
    module = load_module_by_path("capture_defocus_forward_dataset", path)
    grid = getattr(module, "probe_grid", None)
    if grid is None:
        raise ImportError(f"{path} does not define probe_grid")
    return grid


# ---------------------------------------------------------------------------
# Prior candidates
# ---------------------------------------------------------------------------


@dataclass
class PriorPhase:
    """One prior phase: a grid, its provenance and its dataset split."""

    label: str
    grid: np.ndarray
    source_region: int
    resampled_from_region: int | None
    split: str
    source: dict


def _candidate_entries(payload: dict) -> list[tuple[int, dict]]:
    """Flatten ``candidates.json`` into ``(region, entry)`` pairs.

    The main group sits under ``candidates``; additional groups sit under
    ``secondary_groups`` and carry their own region.  Every entry is kept, so
    the audit's main group and its region-128 secondary group both contribute.
    """
    entries: list[tuple[int, dict]] = []
    for entry in payload.get("candidates", []) or []:
        entries.append((int(entry.get("region", 0)), entry))
    for group in payload.get("secondary_groups", []) or []:
        group_region = int(group.get("region", 0))
        for entry in group.get("candidates", []) or []:
            entries.append((int(entry.get("region", group_region)), entry))
    return entries


def resample_grid(grid: np.ndarray, region: int) -> tuple[np.ndarray, int | None]:
    """Bring a model-grid phase onto the target region.

    Args:
        grid: Model-grid phase, raw unwrapped radians.
        region: Target grid size.

    Returns:
        ``(grid, resampled_from_region)``.  The second element is ``None`` when
        no interpolation was needed, and the source region otherwise.
    """
    grid = np.asarray(grid, dtype=np.float64)
    source = int(grid.shape[0])
    if source == region:
        return grid, None
    scaled = zoom(grid, region / source, order=1)
    return np.asarray(scaled, dtype=np.float64), source


def load_prior_candidates(directory: Path, region: int, holdout: int) -> list[PriorPhase]:
    """Read the audit's prior phases and prepare them for capture.

    Args:
        directory: Directory holding ``candidates.json`` and ``candidates/``.
        region: Target grid size for the capture.
        holdout: Number of prior phases to move to the validation split
            (``test``); the last ones are taken.

    Returns:
        The prior phases in acquisition order.

    Raises:
        FileNotFoundError: ``candidates.json`` or a phase file is missing.
        ValueError: The holdout count exceeds the number of candidates.
    """
    manifest = directory / "candidates.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = _candidate_entries(payload)
    if not entries:
        raise ValueError(f"{manifest} lists no candidates")
    if holdout < 0 or holdout > len(entries):
        raise ValueError(
            f"--prior-holdout must be between 0 and {len(entries)}; got {holdout}"
        )
    split_from = len(entries) - holdout
    phases: list[PriorPhase] = []
    for index, (entry_region, entry) in enumerate(entries):
        relative = entry.get("phase_file")
        if not relative:
            raise ValueError(f"candidate {index} has no phase_file")
        path = directory / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing candidate phase {path}")
        grid = np.load(path)
        source_region = int(entry_region or grid.shape[0])
        if grid.shape[0] != source_region:
            source_region = int(grid.shape[0])
        resized, resampled_from = resample_grid(grid, region)
        phases.append(
            PriorPhase(
                label=f"prior_{index:03d}",
                grid=resized,
                source_region=source_region,
                resampled_from_region=resampled_from,
                split="test" if index >= split_from else "train",
                source={
                    "run_path": entry.get("run_path"),
                    "stamp": entry.get("stamp"),
                    "round": entry.get("round"),
                    "relative_improvement": entry.get("relative_improvement"),
                    "cv_after": entry.get("cv_after"),
                    "ee_after": entry.get("ee_after"),
                },
            )
        )
    return phases


# ---------------------------------------------------------------------------
# Capture plan
# ---------------------------------------------------------------------------


@dataclass
class CaptureContext:
    """Everything needed to turn a planned step into a panel phase."""

    shape: tuple[int, int]
    corrected: np.ndarray
    radius: int
    center: tuple[int, int]
    region: int
    seed: int
    probe_grid: Callable[..., tuple[str, np.ndarray]]
    priors: list[PriorPhase]


def _panel_from_grid(model_grid: np.ndarray, ctx: CaptureContext) -> np.ndarray:
    """Map a model-grid phase onto the panel disc."""
    return phase_to_panel(model_grid, ctx.radius, ctx.center)


def resolve_step(step: dict, ctx: CaptureContext) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the panel phase (radians) and the model-grid phase of one step.

    The model-grid phase is ``None`` for the steps that have none (flat
    references and the calibration ramps); those samples store an empty
    ``(0, 0)`` array in the npz.
    """
    spec = step["phase_spec"]
    kind = spec["kind"]
    if kind == "flat":
        return np.zeros(ctx.shape, dtype=np.float64), None
    if kind == "focus":
        return np.asarray(ctx.corrected, dtype=np.float64), None
    if kind == "ramp":
        # ``np.indices`` yields (row index, column index); the x ramp rides on
        # the column index and the y ramp on the row index.
        rows_index, columns_index = np.indices(ctx.shape)
        line = columns_index if spec["axis"] == "x" else rows_index
        phase = ctx.corrected + spec["sign"] * 2 * np.pi * line / spec["period"]
        return np.asarray(phase, dtype=np.float64), None
    if kind == "probe":
        _, model = ctx.probe_grid(spec["index"], ctx.region, ctx.seed)
        panel = _panel_from_grid(model, ctx)
    elif kind == "prior":
        model = ctx.priors[spec["prior_index"]].grid
        panel = _panel_from_grid(model, ctx)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown phase kind {kind!r}")
    if step["branch"] == "C":
        panel = panel + ctx.corrected
    return np.asarray(panel, dtype=np.float64), np.asarray(model, dtype=np.float64)


def build_steps(
    count: int,
    priors: list[PriorPhase],
    reference_every: int = REFERENCE_EVERY,
) -> list[dict]:
    """Describe the whole capture sequence, in acquisition order.

    The baseline part (references, ramps, random probes, drift references) comes
    first and is **index-identical** to the baseline group, so the two datasets
    pair sample by sample.  The prior phases follow.

    The drift references are placed by **one** counter that covers both
    families: a random probe adds one unit, a prior phase adds
    :data:`PRIOR_PROBE_EQUIVALENTS` units (it is a shaping phase and drifts the
    panel state more readily), and a reference pair is inserted as soon as the
    counter reaches ``reference_every``.  With the default ``reference_every=6``
    this reproduces the measured cadence exactly -- a reference pair after every
    sixth random probe and after every prior phase -- so a default run stays
    comparable with the B2 dataset.  The calibration ramps are not counted:
    they measure the camera geometry, they are not content.

    Args:
        count: Number of random probes.
        priors: Prior phases to capture after the probes.
        reference_every: Content captures between two reference pairs.

    Returns:
        The planned captures, in acquisition order.

    Raises:
        ValueError: ``reference_every`` is below one.
    """
    if reference_every < 1:
        raise ValueError(f"reference_every must be >= 1; got {reference_every}")
    steps: list[dict] = []

    def add(label: str, branch: str, family: str, split: str, source: dict,
            phase_spec: dict) -> None:
        steps.append({
            "index": len(steps),
            "label": label,
            "branch": branch,
            "family": family,
            "split": split,
            "source": source,
            "phase_spec": phase_spec,
        })

    def add_reference(label: str, note: str) -> None:
        add(label, "U", "reference", "calibration",
            {"kind": "flat", "note": note}, {"kind": "flat"})
        add(label, "C", "reference", "calibration",
            {"kind": "focus", "note": note}, {"kind": "focus"})

    add_reference("reference_start", "flat panel / defocus compensation")
    for axis in ("x", "y"):
        for period in (480, 960):
            for sign in (-1, 1):
                add(f"ramp_{axis}_{period}_{sign}", "C", "ramp", "calibration",
                    {"kind": "ramp", "axis": axis, "period": period, "sign": sign},
                    {"kind": "ramp", "axis": axis, "period": period, "sign": sign})
    units = 0
    for index in range(count):
        family = ("zernike", "smooth", "bandlimited")[index % 3]
        split = "test" if index % PROBE_TEST_MODULUS >= PROBE_TEST_OFFSET else "train"
        order = ("U", "C") if index % 2 == 0 else ("C", "U")
        for branch in order:
            add(f"probe_{index:03d}", branch, family, split,
                {"kind": "probe", "index": index, "family": family},
                {"kind": "probe", "index": index})
        units += 1
        if units >= reference_every:
            add_reference(f"reference_{index}", "drift reference")
            units = 0
    for order_index, prior in enumerate(priors):
        order = ("U", "C") if order_index % 2 == 0 else ("C", "U")
        for branch in order:
            add(prior.label, branch, "prior", prior.split,
                {"kind": "prior", "label": prior.label,
                 "run_path": prior.source.get("run_path"),
                 "relative_improvement": prior.source.get("relative_improvement")},
                {"kind": "prior", "prior_index": order_index})
        units += PRIOR_PROBE_EQUIVALENTS
        if units >= reference_every:
            add_reference(f"reference_prior_{order_index:03d}",
                          "drift reference after a prior phase")
            units = 0
    return steps


def summarize_plan(steps: list[dict]) -> dict:
    """Counts the plan reports: samples per split, family and branch."""
    splits: dict[str, int] = {}
    families: dict[str, int] = {}
    branches: dict[str, int] = {}
    for step in steps:
        splits[step["split"]] = splits.get(step["split"], 0) + 1
        families[step["family"]] = families.get(step["family"], 0) + 1
        branches[step["branch"]] = branches.get(step["branch"], 0) + 1
    return {"split_counts": splits, "family_counts": families, "branch_counts": branches}


def reference_coverage(steps: list[dict]) -> dict:
    """How densely the drift references cover the content captures.

    A content capture is one phase measured on both planes (the ``U`` row
    counts it once); the calibration ramps are geometry probes, not content.

    Args:
        steps: The planned captures.

    Returns:
        ``{"pairs": int, "max_gap": int, "max_units": int, "gaps": list[int]}``
        -- the number of reference pairs, the largest number of content
        captures between two pairs, and the same interval measured in the
        counter units of :func:`build_steps`.
    """
    gaps: list[int] = []
    units: list[int] = []
    since = 0
    since_units = 0
    last_label: str | None = None
    for step in steps:
        if step["family"] == "reference":
            if step["label"] != last_label:
                gaps.append(since)
                units.append(since_units)
                since = 0
                since_units = 0
                last_label = step["label"]
            continue
        if step["family"] == "ramp" or step["branch"] != "U":
            continue
        since += 1
        since_units += (PRIOR_PROBE_EQUIVALENTS
                        if step["phase_spec"]["kind"] == "prior" else 1)
    return {"pairs": len(gaps), "max_gap": max(gaps) if gaps else 0,
            "max_units": max(units) if units else 0, "gaps": gaps}


def branch_drift(rows: list[dict], branch: str) -> dict | None:
    """Drift statistics of one branch, measured on its reference captures.

    Args:
        rows: The manifest rows.
        branch: ``"U"`` or ``"C"``.

    Returns:
        ``None`` when the branch has fewer than two references (not enough to
        measure a drift), otherwise the centroid drift rate in px/min, the
        largest neighbouring centroid step, the FWHM span and the peak span.
    """
    references = sorted((row for row in rows
                         if row["family"] == "reference" and row["branch"] == branch),
                        key=lambda row: row["monotonic_start_ns"])
    if len(references) < 2:
        return None
    stamps = np.array([row["monotonic_start_ns"] for row in references], dtype=float)
    minutes = float(stamps[-1] - stamps[0]) / 1e9 / 60.0
    if minutes <= 0:
        return None
    centroids = np.array([[row["spot"]["centroid_x"], row["spot"]["centroid_y"]]
                          for row in references])
    steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    net = float(np.linalg.norm(centroids[-1] - centroids[0]))
    fwhm = np.array([row["spot"]["fwhm_px"] for row in references])
    peaks = np.array([row["peak"] for row in references])

    def span(values: np.ndarray) -> float:
        mean = float(values.mean())
        return float((values.max() - values.min()) / mean * 100.0) if mean > 0 else float("nan")

    return {
        "references": len(references),
        "minutes": minutes,
        "centroid_drift_px_per_min": net / minutes,
        "centroid_step_max_px": float(steps.max()) if steps.size else 0.0,
        "fwhm_span_pct": span(fwhm),
        "peak_span_pct": span(peaks),
    }


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def write_run_config(out: Path, plan: dict) -> None:
    """Write ``run_config.json`` -- the record of how the run was set up.

    The file is rewritten whenever the run learns something new about itself
    (in fixed-exposure mode, which exposure each plane ended up with and which
    capture produced it), so an interrupted run still says what it was doing.
    """
    (out / "run_config.json").write_text(json.dumps(plan, default=str, indent=2),
                                          encoding="utf-8")


def acquire_device_lock(
    args: argparse.Namespace, settings: dict, plan: dict
) -> DeviceLock | None:
    """Wait for the bench, then hold the device lock for the whole capture.

    This script used to have **no** waiting at all: it opened the SLM and simply
    died with -10003 when another session held the USB handle.  The verdict now
    comes from :mod:`ao_shaping.tools.kernels.device_mutex`, which separates the
    three cases that used to be conflated:

    * **another session holds it** -- wait out ``--wait-until-idle``, reporting
      the holder's pid and command line;
    * **our own process tree holds it** -- not a conflict, pass straight through;
    * **the lock is free but the SDK still says -10003** -- a stale handle that
      no amount of waiting clears; it stops with the physical power-cycle advice.

    The lock stays held until the capture finishes: releasing it as soon as the
    device is merely *free* would let another session grab the handle mid-run.

    ``--wait-until-idle 0`` (the default) keeps the pre-2026-10-22 behaviour:
    no probe, no lock, just open the devices.

    Args:
        args: Parsed command line (``wait_until_idle`` is the budget in seconds).
        settings: The focus settings (device identity).
        plan: The run config, updated with a ``device_wait`` record.

    Returns:
        The held device lock, or ``None`` when waiting was disabled.

    Raises:
        SystemExit: The device will not become usable within the budget.
    """
    budget = float(args.wait_until_idle)
    if budget <= 0:
        plan["device_wait"] = {
            "budget_s": 0.0,
            "status": "skipped",
            "waited_s": 0.0,
            "own_process_tree": False,
            "holder_pid": None,
            "holder_cmdline": None,
            "detail": "waiting disabled (--wait-until-idle 0)",
            "remedy": "",
        }
        return None
    verdict = device_mutex.wait_until_idle(
        budget,
        slm_number=int(settings["slm_number"]),
        cam_id=int(settings["cam_id"]),
        cam_type=str(settings["cam_type"]),
        own_paths=[Path(__file__).resolve()],
    )
    plan["device_wait"] = {
        "budget_s": budget,
        "status": verdict.status,
        "waited_s": round(verdict.waited_s, 3),
        "own_process_tree": verdict.holder_is_me,
        "holder_pid": verdict.holder.pid if verdict.holder is not None else None,
        "holder_cmdline": verdict.holder.cmdline if verdict.holder is not None else None,
        "detail": verdict.detail,
        "remedy": verdict.remedy,
    }
    if verdict.ok:
        lock = verdict.lock if verdict.lock is not None else DeviceLock.default()
        if not lock.is_locked():
            lock.acquire(timeout=0.0)
        return lock
    if verdict.status == "stale":
        raise SystemExit(
            f"device unavailable ({verdict.kind}): {verdict.headline()} -- "
            f"{verdict.remedy}"
        )
    if verdict.status == "timeout":
        holder = verdict.holder
        who = f" (held by {holder.describe()})" if holder is not None else ""
        raise SystemExit(
            f"device busy after {budget:g}s: {verdict.reason}{who}; "
            "re-run with a larger --wait-until-idle to queue instead"
        )
    raise SystemExit(
        f"device unavailable ({verdict.kind or verdict.status}): {verdict.headline()}"
        + (f" -- {verdict.remedy}" if verdict.remedy else "")
    )


def capture_dataset(args: argparse.Namespace, settings: dict, focus: dict,
                    priors: list[PriorPhase], steps: list[dict],
                    probe_grid: Callable[..., tuple[str, np.ndarray]],
                    plan: dict) -> list[dict]:
    """Run the acquisition and return the manifest rows.

    Every row is flushed to ``manifest.json`` and ``capture_records.pkl`` right
    after it is written, so an interrupted run keeps everything it measured.
    With ``--fixed-exposure`` the exposure ladder runs once per branch and the
    value it picked is recorded in ``run_config.json`` together with the capture
    it came from.
    """
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels.bench_kernels import (
        SLM_PANEL_H, SLM_PANEL_W, capture_settled_report, fresnel_panel, measure_spot,
    )
    from ao_shaping.tools.slm.panel.focus_probe import ExposureSetting, adapt_exposure
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    out = args.output.resolve()
    shape = (SLM_PANEL_H, SLM_PANEL_W)
    center = (int(settings["center_x"]), int(settings["center_y"]))
    radius = int(settings["radius"])
    corrected = fresnel_panel(focus["best_focal_mm"], radius=radius,
                              pupil_center=center, wavelength_nm=settings["wavelength"])
    np.save(out / "focus_phase.npy", corrected)

    ctx = CaptureContext(
        shape=shape,
        corrected=corrected,
        radius=radius,
        center=center,
        region=int(args.region),
        seed=int(args.seed),
        probe_grid=probe_grid,
        priors=priors,
    )

    rows: list[dict] = []
    with Santec(slm_number=settings["slm_number"], wavelength=settings["wavelength"],
                video_mode=0) as slm:
        try:
            with create_camera(settings["cam_type"], settings["cam_id"],
                               exposure_time_ms=1) as cam:
                floor, ceiling = get_camera_exposure_range(cam)
                limit = (configured_daheng_exposure_limit()
                         if settings["cam_type"] == "daheng" else 0)
                if limit > 0:
                    ceiling = min(ceiling, limit)
                current = 1.0
                fixed: dict[str, float] = {}
                fixed_source: dict[str, dict] = {}

                def capture(phase, label, branch, family, split, model_phase=None,
                            extra=None):
                    nonlocal current
                    started = time.monotonic_ns()
                    utc_start = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                    slm.display_data(slm.create_phase_from_array(phase), .8)
                    if args.fixed_exposure and branch in fixed:
                        # The ladder already vetted this exposure when it was
                        # picked; running it again would only cost its settle.
                        value = fixed[branch]
                        cam.reset_exposure_time(value)
                        setting = ExposureSetting(value, 0.0, 0.0, "ok", 0)
                        exposure_fixed = True
                    else:
                        setting = adapt_exposure(cam.reset_exposure_time,
                                                 lambda: cam.get_numpy_image(n_sample=1),
                                                 start_ms=current, floor_ms=floor,
                                                 ceiling_ms=ceiling, full_scale=255)
                        current = float(cam.exposure_time)
                        exposure_fixed = False
                        if args.fixed_exposure:
                            fixed[branch] = current
                            fixed_source[branch] = {
                                "exposure_ms": current,
                                "source_index": len(rows),
                                "source_label": label,
                                "source_utc_start": utc_start,
                            }
                            plan["fixed_exposure"] = {
                                "enabled": True,
                                "branches": dict(fixed_source),
                            }
                            write_run_config(out, plan)
                    acquisition = capture_settled_report(cam, slm, phase, n_frames=3,
                                                         stable_tol=.05, max_wait_s=12)
                    raw = np.stack([np.asarray(cam.get_numpy_image(n_sample=1))
                                    for _ in range(3)])
                    number = len(rows)
                    file = f"sample_{number:04d}.npz"
                    np.savez_compressed(
                        out / file,
                        frame=acquisition.frame,
                        raw_frames=raw,
                        panel_phase=phase.astype(np.float32),
                        model_phase=(np.asarray(model_phase)
                                     if model_phase is not None else np.zeros((0, 0))),
                    )
                    row = dict(index=number, file=file, label=label, branch=branch,
                               family=family, split=split, exposure_ms=float(cam.exposure_time),
                               exposure_status=setting.status,
                               settled=asdict(acquisition.settle),
                               utc_start=utc_start,
                               timestamp_utc=datetime.now(timezone.utc).isoformat(
                                   timespec="microseconds"),
                               monotonic_start_ns=started,
                               monotonic_end_ns=time.monotonic_ns(),
                               peak=float(raw.max()),
                               spot=asdict(measure_spot(acquisition.frame)))
                    if extra:
                        row.update(extra)
                    if args.fixed_exposure:
                        # Only present in fixed mode: with an adaptive exposure
                        # there is no fixed value to report.
                        row["exposure_fixed"] = bool(exposure_fixed)
                        row["fixed_exposure_ms"] = fixed.get(branch)
                    rows.append(row)
                    (out / "manifest.json").write_text(json.dumps(rows, indent=2),
                                                       encoding="utf-8")
                    with (out / "capture_records.pkl").open("wb") as stream:
                        pickle.dump(rows, stream, protocol=5)
                    print(f"{number:03d} {label} {branch}: "
                          f"exposure={row['exposure_ms']:.4f} "
                          f"peak={raw.max()} settled={acquisition.settle.settled}",
                          flush=True)
                    if setting.status != "ok" or raw.max() >= 250:
                        raise RuntimeError(f"unusable exposure: {label} {setting.status}")

                for step in steps:
                    phase, model_phase = resolve_step(step, ctx)
                    extra = None
                    if step["phase_spec"]["kind"] == "prior":
                        prior = priors[step["phase_spec"]["prior_index"]]
                        extra = {"prior_source": prior.source,
                                 "resampled_from_region": prior.resampled_from_region}
                    capture(phase, step["label"], step["branch"], step["family"],
                            step["split"], model_phase, extra)
        finally:
            slm.display_data(slm.create_phase_from_array(np.zeros(shape)), .5)
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(rows: list[dict], priors: list[PriorPhase], steps: list[dict]) -> None:
    """Print what the run measured: counts, ranges and the prior provenance.

    Besides the per-branch counts it reports the **reference coverage density**
    (how many content captures may sit between two drift references) and the
    drift rate that density has to cover, measured on the reference captures
    themselves.
    """
    print("\n=== capture summary ===", flush=True)
    branches = sorted({row["branch"] for row in rows})
    for branch in branches:
        selected = [row for row in rows if row["branch"] == branch]
        splits: dict[str, int] = {}
        for row in selected:
            splits[row["split"]] = splits.get(row["split"], 0) + 1
        exposures = [row["exposure_ms"] for row in selected]
        peaks = [row["peak"] for row in selected]
        settled = sum(1 for row in selected if row["settled"]["settled"])
        print(f"branch {branch}: {len(selected)} samples, splits={splits}, "
              f"exposure={min(exposures):.4f}..{max(exposures):.4f} ms, "
              f"peak={min(peaks):.0f}..{max(peaks):.0f}, "
              f"settled={settled}/{len(selected)}", flush=True)
    coverage = reference_coverage(steps)
    print(f"reference 覆盖密度：最多间隔 {coverage['max_gap']} 条内容采集"
          f"（reference 对 {coverage['pairs']} 对）", flush=True)
    for branch in branches:
        drift = branch_drift(rows, branch)
        if drift is None:
            print(f"  {branch} 分支漂移率：未记录（该分支的 reference 少于 2 条）",
                  flush=True)
            continue
        print(f"  {branch} 分支漂移率：质心 {drift['centroid_drift_px_per_min']:.3f} px/min"
              f"（相邻最大 {drift['centroid_step_max_px']:.2f} px，"
              f"跨 {drift['minutes']:.1f} min / {drift['references']} 条 reference），"
              f"FWHM 跨度 {drift['fwhm_span_pct']:.1f}%，"
              f"峰值跨度 {drift['peak_span_pct']:.1f}%"
              f"（自适应曝光引起，非光漂）", flush=True)
    fixed = {row["branch"]: row.get("fixed_exposure_ms") for row in rows
             if "fixed_exposure_ms" in row}
    if fixed:
        print("fixed exposure: "
              + ", ".join(f"{branch} {value:.4f} ms" for branch, value in sorted(fixed.items())),
              flush=True)
    print("prior phases:", flush=True)
    measured = {row["label"]: row for row in rows if row["family"] == "prior"}
    for prior in priors:
        source = prior.source
        row = measured.get(prior.label)
        resampled = ("" if prior.resampled_from_region is None
                     else f" -> resampled from {prior.resampled_from_region}")
        peak = "" if row is None else f" measured_peak={row['peak']:.0f}"
        print(f"  {prior.label} [{prior.split}] region={prior.source_region}{resampled}"
              f" run={source.get('run_path')} stamp={source.get('stamp')}"
              f" round={source.get('round')}"
              f" improvement={source.get('relative_improvement')}"
              f" cv_after={source.get('cv_after')} ee_after={source.get('ee_after')}{peak}",
              flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_plan(plan: dict, steps: list[dict]) -> None:
    """Print the plan as JSON, then the capture list it implies."""
    print("=== plan (JSON) ===", flush=True)
    print(json.dumps(plan, default=str, indent=2), flush=True)
    print("=== capture list ===", flush=True)
    for step in steps:
        source = step["source"]
        detail = source.get("note") or source.get("run_path") or source.get("family") or ""
        print(f"  {step['index']:03d} {step['label']:<22s} {step['branch']} "
              f"{step['family']:<10s} {step['split']:<11s} {detail}", flush=True)
    print(f"  total: {len(steps)} captures", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """命令行参数 (``main`` 与测试共用同一个 parser)。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-directory", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Audit directory holding candidates.json")
    parser.add_argument("--count", type=int, default=12,
                        help="Number of random probes (U/C pairs)")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--region", type=int, default=128,
                        help="Model-grid size of the probe and prior phases")
    parser.add_argument("--prior-holdout", type=int, default=0,
                        help="Number of prior phases moved to the validation split")
    parser.add_argument("--reference-every", type=int, default=REFERENCE_EVERY,
                        help="每 N 条内容采集插入一对 reference（漂移监测）。"
                             "默认 6 = 现有节奏（每 6 条随机探针后一对、每条先验相位后一对）。"
                             "间隔要按分支的漂移率选：补偿面（C）实测质心漂移 0.014–0.034 px/min、"
                             "FWHM 跨度 0.6–2.6%%，可放宽到十几条；未补偿面（U）实测 1.0–1.35 px/min、"
                             "FWHM 跨度 68–71%%，应保持较密的 reference。")
    parser.add_argument("--fixed-exposure", action="store_true",
                        help="按分支固定曝光：沿用该分支第一次自适应出来的值，之后不再走曝光阶梯。"
                             "每条可省下阶梯里的沉降（约 5 s），且峰值才可用于漂移监测。"
                             "代价：不再逐条跟踪光强漂移，只宜在补偿面（或已确认光强稳定）开启；"
                             "固定曝光采的数据不能与自适应曝光的数据在强度口径上直接混比（要各自归一化）。")
    parser.add_argument("--execute", action="store_true",
                        help="Open the bench; without it only the plan is printed")
    parser.add_argument("--wait-until-idle", type=float, default=0.0,
                        metavar="SECONDS",
                        help="开跑前等设备空闲的预算（s）。SLM 的 USB 句柄是独占的，"
                             "别的会话持有时只会拿到 -10003 —— 以前本脚本直接崩，现在按"
                             "这个预算**排队**。三种情况分开：别人持有 ⇒ 等并报出对方的"
                             "pid+命令行；自己这棵进程树持有 ⇒ 不是冲突，直接放行；"
                             "锁空但 SDK 报 -10003 ⇒ SDK 陈旧状态，排队等不来，"
                             "需给 SLM 控制器物理断电重置。0 = 不等（只探一次）。")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    focus_file = args.focus_directory / "focus.json"
    if not focus_file.is_file():
        raise SystemExit(f"missing {focus_file}")
    saved = json.loads(focus_file.read_text(encoding="utf-8"))
    settings = saved.get("settings", {})
    focus = saved.get("focus", {})
    missing = [key for key in ("center_x", "center_y", "radius", "wavelength",
                               "slm_number", "cam_type", "cam_id") if key not in settings]
    if missing:
        raise SystemExit(f"{focus_file} is missing settings keys: {missing}")
    if focus.get("best_focal_mm") is None:
        raise SystemExit(f"{focus_file} is missing focus.best_focal_mm")
    if args.count < 6 or args.region < 24:
        parser.error("count >= 6 and region >= 24 required")
    if args.prior_holdout < 0:
        parser.error("--prior-holdout must not be negative")
    if args.reference_every < 1:
        parser.error("--reference-every must be >= 1")

    try:
        probe_grid = load_probe_grid()
    except (ImportError, OSError) as exc:
        raise SystemExit(
            f"cannot reuse the baseline probe grid ({exc}); the random probes must "
            "stay bit-identical to the baseline group, so this script refuses to "
            "fall back to another phase family"
        ) from exc
    try:
        priors = load_prior_candidates(args.candidates, args.region, args.prior_holdout)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"cannot read the prior candidates: {exc}") from exc

    steps = build_steps(args.count, priors, int(args.reference_every))
    counts = summarize_plan(steps)
    coverage = reference_coverage(steps)
    plan = {
        **vars(args),
        "settings": settings,
        "best_diopter_m1": focus.get("best_diopter_m1"),
        "best_focal_mm": focus.get("best_focal_mm"),
        "random_probe_count": int(args.count),
        "prior_count": len(priors),
        "capture_count": len(steps),
        "pairing": "U/C pairs (current plane / defocus-compensated plane)",
        **counts,
        "reference_every": int(args.reference_every),
        "reference_pairs": coverage["pairs"],
        "reference_max_gap": coverage["max_gap"],
        "reference_max_gap_units": coverage["max_units"],
        "fixed_exposure": {"enabled": bool(args.fixed_exposure), "branches": {}},
        "prior_holdout_entries": int(args.prior_holdout),
        "prior_sources": [
            {
                "label": prior.label,
                "split": prior.split,
                "source_region": prior.source_region,
                "resampled_from_region": prior.resampled_from_region,
                **prior.source,
            }
            for prior in priors
        ],
        "samples": [
            {key: step[key] for key in ("index", "label", "branch", "family", "split", "source")}
            for step in steps
        ],
    }
    print_plan(plan, steps)
    if not args.execute:
        return 0

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_run_config(out, plan)
    # 设备互斥: 撞上 -10003 时是**排队**而不是崩 (以前没有等待能力)。
    # 0 = 不等 (保持旧行为: 直接开设备)。
    lock: DeviceLock | None = None
    try:
        lock = acquire_device_lock(args, settings, plan)
    finally:
        # 等待的结论也要落盘 —— **失败时同样** (报告要能回答「当时为什么没跑」)。
        write_run_config(out, plan)
    try:
        rows = capture_dataset(args, settings, focus, priors, steps, probe_grid, plan)
    finally:
        if lock is not None:
            lock.release()
    if args.fixed_exposure:
        # The capture loop already recorded which capture produced each fixed
        # exposure; keep that detail and only make sure the section is there.
        plan["fixed_exposure"] = {
            "enabled": True,
            "branches": plan.get("fixed_exposure", {}).get("branches", {}),
        }
        write_run_config(out, plan)
    print_summary(rows, priors, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
