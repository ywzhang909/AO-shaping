"""Score a recorded model-in-loop run against the *achievable* correlation ceiling.

The run's holdout gate scores the forward model over the whole model grid, but the
camera only ever observes a crop of it: ``_to_model_grid`` zero-pads everything the
camera did not see. This script rebuilds the same prediction the run made, measures
how much of the modelled energy falls inside that crop, and reports three numbers per
probe:

``corr_full``
    What the run recorded -- Pearson over the whole model grid, structural zeros
    included. Reproducing it proves this reconstruction is faithful.
``ceiling``
    ``corr(prediction, prediction * support)``: the score a *perfect* model would get,
    because its prediction would also be zero-padded by the crop. Nothing on this
    bench can exceed it.
``corr_support``
    Pearson restricted to the pixels the camera actually saw -- the model's real
    accuracy, free of the crop penalty.

No device is opened: every number comes from a previously recorded run directory.

Evidence is written to ``data/debug/analyze_holdout_ceiling/<stamp>/``.
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
import shutil
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom as _zoom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    _make_optimizer,
    _prediction_correlation,
    _to_model_grid,
)
from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
    load_bench_geometry,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center


def _load_config(run_dir: Path):
    """Rebuild the run's config from its own ``run_config.json``.

    The saved mapping is filtered to the dataclass's real field names so a run that
    predates a field rename still loads.
    """
    from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig

    raw = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    known = {f.name for f in fields(SlmModelInLoopConfig)}
    payload = {k: v for k, v in raw.items() if k in known}
    return SlmModelInLoopConfig(**payload), sorted(set(raw) - known)


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(p for p in root.iterdir() if (p / "run_config.json").is_file())
    if not candidates:
        raise FileNotFoundError(f"no recorded run under {root}")
    return candidates[-1]


def _support_mask(
    frame: np.ndarray, roi_center, far_field_size: int, cell_scale: float, rotation: int, shape: tuple[int, int]
) -> np.ndarray:
    """Model-grid cells the camera actually observed.

    ``model_grid != 0`` looks equivalent and is not, for two measured reasons. The
    mapping is a non-negative linear operator (crop, zero-pad, block average or
    order-1 spline, rotation), so the nonzero mask is *always* a subset of the true
    footprint -- it can only under-report, never over-report: on the recorded
    geometry a frame bright across the window gives 7.91%, the same as the
    footprint, while a dim one gives 3.96%. And it depends entirely on the scene:
    a black frame collapses it to the empty set while the camera plainly did see
    those pixels.

    That second failure is not hypothetical -- a bench whose light has dropped
    leaves a near-black frame, and then the naive mask reports that the camera saw
    nothing at all. Pushing an all-ones frame of the camera's shape through the very
    same mapping makes the support independent of the scene while inheriting the
    mapping's exact geometry, crop and interpolation and rotation included.
    ``shape`` is the shape a real mapping returns, which is the padded far field
    rather than the pupil grid.
    """
    footprint = _to_model_grid(
        np.ones(frame.shape, dtype=np.float64), roi_center, far_field_size, cell_scale, rotation
    )
    if footprint.shape != shape:
        raise ValueError(f"footprint {footprint.shape} does not match the mapped grid {shape}")
    return footprint > 0


def _bandwidth(phase: np.ndarray) -> tuple[float, float]:
    """(high-frequency power fraction, gradient rms) of a pupil phase.

    The recorded manifest does not name a probe's family, so it is recovered from the
    phase itself. Low-order probes concentrate their power near the origin; a white
    noise probe spreads half of it past a quarter of the model's Nyquist frequency.
    """
    grid = np.asarray(phase, dtype=np.float64)
    grid = grid - grid.mean()
    power = np.fft.fftshift(np.abs(np.fft.fft2(grid)) ** 2)
    n = grid.shape[0]
    y, x = np.mgrid[0:n, 0:n] - n // 2
    radius = np.hypot(x, y) / n
    total = power.sum()
    high = float(power[radius > 0.25].sum() / total) if total > 0 else 0.0
    return high, float(np.hypot(*np.gradient(grid)).std())


def _family_of(phase: np.ndarray) -> str:
    """Name a probe's phase family from its spectrum.

    Thresholds sit in the empty gaps of the measured separation, not on the values:
    the recorded run gives high-frequency fractions 0.004-0.030 for the smooth and
    Zernike probes against 0.805 for the freeform ones, and gradient rms 0.10-0.20
    against 0.72-1.09 for the Zernike pair.
    """
    high, gradient = _bandwidth(phase)
    if high > 0.3:
        return "freeform"
    return "zernike" if gradient > 0.4 else "smooth"


def _remove_flat(measured: np.ndarray, flat_grid: np.ndarray) -> np.ndarray:
    """Strip the least-squares multiple of the flat reference from a measurement.

    A probe whose light scatters out of the zeroth order leaves the flat field itself
    behind as an additive pedestal. The run models that as ``gain * prediction +
    leakage * flat_grid`` with one ``leakage`` shared across probes, so a probe whose
    flux collapsed cannot be served by a gain learned on bright probes. Subtracting the
    best multiple isolates what is left of the probe after the pedestal is gone.

    Note this subtracts unconditionally, including when no pedestal is present: it
    always projects out the flat direction, which perturbs a pedestal-free field by a
    measured 0.75 counts on a 0-100 frame (``corr(stripped, speckle) = 0.99997``). So a
    correlation computed after this is against a slightly projected field even for a
    clean probe -- which is why it is reported next to the untouched number rather than
    instead of it.
    """
    a = measured - measured.mean()
    b = flat_grid - flat_grid.mean()
    denominator = float((b * b).sum())
    if denominator <= 0:
        return measured
    return measured - (float((a * b).sum()) / denominator) * flat_grid


def analyze_run(run_dir: Path) -> dict:
    """Rebuild and score every recorded probe of one run."""
    config, unknown = _load_config(run_dir)
    geometry = load_bench_geometry(run_dir / "bench_geometry.json")
    coefficients = np.load(run_dir / "fitted_coefficients.npy")
    source_amplitude = np.load(run_dir / "source_amplitude.npy")

    optimizer = _make_optimizer(config, coefficients, source_amplitude)

    scale = float(geometry.camera_px_per_model_px)
    cell_scale = (
        scale * float(config.region) / float(geometry.far_field_size)
        if config.cam_type == "sim"
        else scale
    )
    rotation = 0 if config.cam_type == "sim" else 1

    manifest = json.loads((run_dir / "probes" / "manifest.json").read_text(encoding="utf-8"))
    flat = next(e for e in manifest if e["split"] == "flat")
    flat_frame = np.load(run_dir / "probes" / flat["file"])["frame"]
    roi_center = zero_order_center(flat_frame, refine=False)

    def grid_of(frame: np.ndarray) -> np.ndarray:
        return _to_model_grid(frame, roi_center, int(geometry.far_field_size), cell_scale, rotation)

    flat_frame_grid = grid_of(flat_frame)
    support = _support_mask(
        flat_frame, roi_center, int(geometry.far_field_size), cell_scale, rotation, flat_frame_grid.shape
    )

    records = []
    grids: dict[str, np.ndarray] = {}
    for entry in manifest:
        if entry["split"] == "flat":
            continue
        blob = np.load(run_dir / "probes" / entry["file"])
        prediction = np.asarray(
            optimizer.forward_intensity(coefficients, blob["phase"], source_amplitude)
        )
        model_grid = grid_of(np.asarray(blob["frame"]))
        high, gradient = _bandwidth(blob["phase"])
        masked_prediction = prediction * support
        observed = model_grid[support]
        records.append(
            {
                "file": entry["file"],
                "split": entry["split"],
                "family": _family_of(blob["phase"]),
                "high_frequency_fraction": high,
                "gradient_rms": gradient,
                "exposure_ms": entry.get("exposure_ms"),
                "recorded_correlation": entry.get("recorded_correlation", entry.get("correlation")),
                "corr_full": _prediction_correlation(prediction, model_grid),
                "corr_support": _prediction_correlation(prediction[support], observed),
                "ceiling": _prediction_correlation(prediction, masked_prediction),
                "energy_fraction_in_support": float(
                    prediction[support].sum() / prediction.sum()
                ),
                "flux_ratio_to_flat": float(model_grid.sum() / flat_frame_grid.sum()),
                "flat_correlation": _prediction_correlation(model_grid, flat_frame_grid),
                "corr_support_flat_removed": _prediction_correlation(
                    prediction[support], _remove_flat(model_grid, flat_frame_grid)[support]
                ),
            }
        )
        grids[entry["file"]] = model_grid

    nulls = _speckle_nulls(records, grids)
    return {
        "run_dir": str(run_dir),
        "unknown_config_fields": unknown,
        "roi_center": [float(roi_center[0]), float(roi_center[1])],
        "cell_scale": cell_scale,
        "camera_rotation_quarters": rotation,
        "grid": {
            "pupil_region": int(config.region),
            "far_field_size": int(geometry.far_field_size),
            "mapped_shape": list(int(v) for v in flat_frame_grid.shape),
            "cam_size": int(config.cam_size),
            "camera_px_per_model_px": scale,
            "support_model_px": round(float(np.sqrt(support.sum())), 2),
            "support_area_pct": round(100.0 * float(support.mean()), 3),
        },
        "gate": float(config.min_holdout_correlation),
        "probes": records,
        "holdout_null": _speckle_nulls(records, grids),
    }


def _speckle_nulls(records: list[dict], grids: dict[str, np.ndarray]) -> dict:
    """Cross-correlate measured probes of the same family against *each other*.

    Two speckle realisations share no structure, so this is the score a model would
    earn by predicting the right distribution with no knowledge of that particular
    sample. Any claimed accuracy near this level is noise, not skill.
    """
    scored = [r for r in records if r["recorded_correlation"] is not None]
    values: dict[str, list[float]] = {}
    for i, a in enumerate(scored):
        for b in scored[i + 1 :]:
            if a["family"] != b["family"]:
                continue
            values.setdefault(a["family"], []).append(
                _prediction_correlation(grids[a["file"]], grids[b["file"]])
            )
    if not values:
        return {}
    return {
        family: {"pairs": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))}
        for family, v in sorted(values.items())
    }


def _family_rows(result: dict) -> list[dict]:
    rows: dict[str, list[dict]] = {}
    for probe in result["probes"]:
        rows.setdefault(probe["family"], []).append(probe)
    summary = []
    for family, probes in rows.items():
        def agg(key: str, _probes=probes) -> float:
            return float(np.mean([p[key] for p in _probes]))

        summary.append(
            {
                "family": family,
                "n": len(probes),
                "corr_full": agg("corr_full"),
                "corr_support": agg("corr_support"),
                "ceiling": agg("ceiling"),
                "energy_fraction_in_support": agg("energy_fraction_in_support"),
            }
        )
    return sorted(summary, key=lambda r: r["family"])


def render(result: dict, families: list[dict], out_dir: Path) -> Path:
    """Write the three-score comparison as a figure."""
    labels = [f"{r['family']}\n(n={r['n']})" for r in families]
    xs = np.arange(len(families))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    for offset, key, name in (
        (-width, "corr_full", "as recorded (whole grid)"),
        (0.0, "corr_support", "camera support only"),
        (width, "ceiling", "perfect-model ceiling"),
    ):
        ax.bar(xs + offset, [r[key] for r in families], width, label=name)
    gate = result["gate"]
    ax.axhline(gate, color="crimson", ls="--", lw=1.2, label=f"gate = {gate:g}")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("correlation")
    ax.set_title("Holdout score vs what the camera can see")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(xs, [100 * r["energy_fraction_in_support"] for r in families], 0.5)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("% of modelled energy inside the camera crop")
    ax.set_title("Observability of each phase family")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "holdout_ceiling.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def render_frames(
    run_dir: Path, records: list[dict], out_path: Path, max_edge: int = 420, mode: str = "flat"
) -> Path:
    """Write the measured CCD far fields as a labelled contact sheet.

    These are the raw camera frames the gate scored -- not predictions -- so a report
    can show what the bench actually returned. Frames are decimated for display only;
    the scoring above never touches them.

    Two scalings, because either one alone misleads. ``mode="flat"`` puts every frame
    on the flat frame's peak, which makes the collapse in absolute throughput visible
    but crushes every probe dimmer than the flat to black. ``mode="peak"`` normalises
    each frame to its own peak, which reveals each probe's speckle structure but hides
    exactly that throughput loss.
    """
    manifest = json.loads((run_dir / "probes" / "manifest.json").read_text(encoding="utf-8"))
    flat_entry = next(e for e in manifest if e["split"] == "flat")
    entries = [flat_entry] + [e for e in manifest if e["split"] != "flat" and e.get("correlation") is not None]
    by_file = {r["file"]: r for r in records}

    def load(entry: dict) -> np.ndarray:
        frame = np.asarray(np.load(run_dir / "probes" / entry["file"])["frame"], dtype=np.float64)
        edge = max(frame.shape)
        if edge > max_edge:
            factor = edge / float(max_edge)
            small = _zoom(frame, (1 / factor, 1 / factor), order=1)
            frame = np.clip(small, 0, None)
        return frame

    cols = 4
    rows = int(np.ceil(len(entries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.35 * rows), squeeze=False)
    flat_peak = float(load(flat_entry).max())
    for index, entry in enumerate(entries):
        ax = axes[index // cols][index % cols]
        frame = load(entry)
        own_peak = float(frame.max())
        vmax = flat_peak if mode == "flat" else (own_peak or 1.0)
        ax.imshow(frame, cmap="inferno", vmin=0, vmax=vmax)
        if entry["split"] == "flat":
            title = f"flat  {entry['exposure_ms']:.3g} ms\npeak {own_peak:.0f}"
        else:
            record = by_file.get(entry["file"], {})
            title = (
                f"{record.get('family', '?')}  {entry['exposure_ms']:.3g} ms "
                f"({100 * record.get('flux_ratio_to_flat', float('nan')):.1f}% of flat)\n"
                f"corr {record.get('corr_full', float('nan')):.3f} / cap {record.get('ceiling', float('nan')):.3f}"
            )
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for index in range(len(entries), rows * cols):
        axes[index // cols][index % cols].axis("off")
    caption = (
        "every frame scaled to the flat peak (shows absolute throughput)"
        if mode == "flat"
        else "every frame scaled to its own peak (shows each probe's speckle structure)"
    )
    fig.suptitle(f"{run_dir.name}: measured CCD far fields -- {caption}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--debug-root", type=Path, default=ROOT / "data" / "debug" / "slm_model_in_loop"
    )
    parser.add_argument(
        "--report-assets",
        type=Path,
        default=None,
        help="Also write the contact sheet of measured CCD far fields and a copy of the "
        "ceiling chart here, for embedding in a report.",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir or _latest_run_dir(args.debug_root)
    run_dir = run_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "debug" / "analyze_holdout_ceiling" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    result = analyze_run(run_dir)
    families = _family_rows(result)
    figure = render(result, families, out_dir)

    payload = {"families": families, **result}
    (out_dir / "holdout_ceiling.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.report_assets is not None:
        assets = args.report_assets
        written = [
            render_frames(run_dir, result["probes"], assets / "holdout_frames.png", mode="flat"),
            render_frames(
                run_dir, result["probes"], assets / "holdout_frames_normalized.png", mode="peak"
            ),
        ]
        shutil.copy2(figure, assets / "holdout_ceiling.png")
        for path in written:
            logger.info(f"report asset: {path}")

    header = (
        f"run: {run_dir.name}   support: {result['grid']['support_model_px']} px of a "
        f"{result['grid']['mapped_shape'][0]}-px far-field grid "
        f"({result['grid']['support_area_pct']}%)   gate: {result['gate']:g}"
    )
    logger.info(header)
    logger.info(f"{'family':10} {'n':>3} {'full':>8} {'support':>9} {'ceiling':>9} {'energy%':>8}")
    for row in families:
        logger.info(
            f"{row['family']:10} {row['n']:>3} {row['corr_full']:>8.4f} "
            f"{row['corr_support']:>9.4f} {row['ceiling']:>9.4f} "
            f"{100 * row['energy_fraction_in_support']:>8.2f}"
        )
    logger.info()
    logger.info(f"{'holdout':26} {'family':10} {'recorded':>9} {'rebuilt':>8} {'ceiling':>8} {'flux/flat':>10} {'corr~flat':>10}")
    for probe in result["probes"]:
        if probe["recorded_correlation"] is None:
            continue
        logger.info(
            f"{probe['file']:26} {probe['family']:10} {probe['recorded_correlation']:>9.4f} "
            f"{probe['corr_full']:>8.4f} {probe['ceiling']:>8.4f} "
            f"{probe['flux_ratio_to_flat']:>10.4f} {probe['flat_correlation']:>10.4f}"
        )
    logger.info()
    logger.info(f"{'holdout':26} {'family':10} {'support':>9} {'support-flat-removed':>21}")
    for probe in result["probes"]:
        if probe["recorded_correlation"] is None:
            continue
        logger.info(
            f"{probe['file']:26} {probe['family']:10} {probe['corr_support']:>9.4f} "
            f"{probe['corr_support_flat_removed']:>21.4f}"
        )
    for family, null in result["holdout_null"].items():
        logger.info(f"same-family measured-vs-measured null [{family}]: {null['mean']:.4f} (n={null['pairs']})")
    logger.info(f"evidence: {out_dir}")
    logger.info(f"figure:   {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())