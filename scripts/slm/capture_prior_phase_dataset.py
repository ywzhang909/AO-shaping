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

from ao_shaping.utils.slm.panel_geometry import phase_to_panel  # noqa: E402

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

#: How many random probes share one drift reference pair.
REFERENCE_EVERY = 6


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
) -> list[dict]:
    """Describe the whole capture sequence, in acquisition order.

    The baseline part (references, ramps, random probes, drift references) comes
    first and is **index-identical** to the baseline group, so the two datasets
    pair sample by sample.  The prior phases follow, each followed by its own
    reference pair: a prior phase is a shaping phase and drifts the panel state
    more readily than a random probe.
    """
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

    add("reference_start", "U", "reference", "calibration",
        {"kind": "flat", "note": "flat panel"},
        {"kind": "flat"})
    add("reference_start", "C", "reference", "calibration",
        {"kind": "focus", "note": "defocus compensation"},
        {"kind": "focus"})
    for axis in ("x", "y"):
        for period in (480, 960):
            for sign in (-1, 1):
                add(f"ramp_{axis}_{period}_{sign}", "C", "ramp", "calibration",
                    {"kind": "ramp", "axis": axis, "period": period, "sign": sign},
                    {"kind": "ramp", "axis": axis, "period": period, "sign": sign})
    for index in range(count):
        family = ("zernike", "smooth", "bandlimited")[index % 3]
        split = "test" if index % PROBE_TEST_MODULUS >= PROBE_TEST_OFFSET else "train"
        order = ("U", "C") if index % 2 == 0 else ("C", "U")
        for branch in order:
            add(f"probe_{index:03d}", branch, family, split,
                {"kind": "probe", "index": index, "family": family},
                {"kind": "probe", "index": index})
        if index % REFERENCE_EVERY == REFERENCE_EVERY - 1:
            add(f"reference_{index}", "U", "reference", "calibration",
                {"kind": "flat", "note": "drift reference"},
                {"kind": "flat"})
            add(f"reference_{index}", "C", "reference", "calibration",
                {"kind": "focus", "note": "drift reference"},
                {"kind": "focus"})
    for order_index, prior in enumerate(priors):
        order = ("U", "C") if order_index % 2 == 0 else ("C", "U")
        for branch in order:
            add(prior.label, branch, "prior", prior.split,
                {"kind": "prior", "label": prior.label,
                 "run_path": prior.source.get("run_path"),
                 "relative_improvement": prior.source.get("relative_improvement")},
                {"kind": "prior", "prior_index": order_index})
        add(f"reference_prior_{order_index:03d}", "U", "reference", "calibration",
            {"kind": "flat", "note": "drift reference after a prior phase"},
            {"kind": "flat"})
        add(f"reference_prior_{order_index:03d}", "C", "reference", "calibration",
            {"kind": "focus", "note": "drift reference after a prior phase"},
            {"kind": "focus"})
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


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture_dataset(args: argparse.Namespace, settings: dict, focus: dict,
                    priors: list[PriorPhase], steps: list[dict],
                    probe_grid: Callable[..., tuple[str, np.ndarray]]) -> list[dict]:
    """Run the acquisition and return the manifest rows.

    Every row is flushed to ``manifest.json`` and ``capture_records.pkl`` right
    after it is written, so an interrupted run keeps everything it measured.
    """
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_range
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels.bench_kernels import (
        SLM_PANEL_H, SLM_PANEL_W, capture_settled_report, fresnel_panel, measure_spot,
    )
    from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure
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

                def capture(phase, label, branch, family, split, model_phase=None,
                            extra=None):
                    nonlocal current
                    started = time.monotonic_ns()
                    utc_start = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                    slm.display_data(slm.create_phase_from_array(phase), .8)
                    setting = adapt_exposure(cam.reset_exposure_time,
                                             lambda: cam.get_numpy_image(n_sample=1),
                                             start_ms=current, floor_ms=floor,
                                             ceiling_ms=ceiling, full_scale=255)
                    current = float(cam.exposure_time)
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
                    rows.append(row)
                    (out / "manifest.json").write_text(json.dumps(rows, indent=2),
                                                       encoding="utf-8")
                    with (out / "capture_records.pkl").open("wb") as stream:
                        pickle.dump(rows, stream, protocol=5)
                    print(f"{number:03d} {label} {branch}: exposure={current:.4f} "
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


def print_summary(rows: list[dict], priors: list[PriorPhase]) -> None:
    """Print what the run measured: counts, ranges and the prior provenance."""
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


def main() -> int:
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
    parser.add_argument("--execute", action="store_true",
                        help="Open the bench; without it only the plan is printed")
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

    steps = build_steps(args.count, priors)
    counts = summarize_plan(steps)
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
    (out / "run_config.json").write_text(
        json.dumps(plan, default=str, indent=2), encoding="utf-8")
    rows = capture_dataset(args, settings, focus, priors, steps, probe_grid)
    print_summary(rows, priors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
