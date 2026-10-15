"""Golden harness for generate_bench_probe_report.render_report (issue #128 B-group).

Reproduces the two report arms from test_generate_bench_probe_report.py
(consistent vs. stale geometry) and prints both, so before/after refactors can
be diffed byte-for-byte.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "slm" / "generate_bench_probe_report.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_generator():
    spec = importlib.util.spec_from_file_location("_bench_report", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_report"] = module
    spec.loader.exec_module(module)
    return module


def _report(gen, conflicts, tmp_path: Path) -> str:
    npz = tmp_path / "sweep_records.npz"
    if not npz.exists():
        np.savez_compressed(
            npz,
            label=np.array(["flat", "rampx120", "rampx240", "tiltx-1", "tiltx+1"]),
            mode=np.array(["flat", "ramp", "ramp", "tilt", "tilt"]),
            axis=np.array(["", "x", "x", "x", "x"]),
            coefficient=np.array([0.0, 120.0, 240.0, -1.0, 1.0]),
            fwhm_px=np.array([13.0, 12.0, 12.1, 12.5, 12.4]),
            centroid_x=np.array([668.5, 667.9, 668.0, 668.5, 668.9]),
            centroid_y=np.array([1026.7, 963.8, 995.3, 1032.1, 1021.3]),
            peak=np.array([135.0, 120.0, 122.0, 127.0, 124.0]),
            hollowness=np.array([0.92, 0.87, 0.88, 0.89, 0.90]),
            shift_x=np.array([np.nan] * 5),
            shift_y=np.array([np.nan] * 5),
            shift_corr=np.array([np.nan] * 5),
            zernike_radius=np.array(450),
            collect_disc=np.array(450),
            pupil_center=np.array([960, 600]),
        )
    sweep = gen.load_sweep(npz)
    geometry = (
        {"panel_disc_radius": 200, "region": 128, "far_field_size": 2048,
         "method": "speckle", "calibration_notes": "old"}
        if conflicts else
        {"panel_disc_radius": 450, "region": 256, "far_field_size": 4096,
         "method": "sweep", "calibration_notes": "fresh"}
    )
    return gen.render_report(
        sweep=sweep,
        geometry=geometry,
        geometry_path=tmp_path / "bench_geometry.json",
        sidecar=None,
        sidecar_path=None,
        npz_path=npz,
        ramp_routes=gen.compute_ramp_routes(sweep),
        tilt_routes=gen.compute_tilt_routes(sweep, 0.30),
        curve_map={},
        constants=gen._import_bench_constants(),
        figures=[],
        frames_status="skipped",
        frames_detail="",
        geometry_conflicts=conflicts,
    )


def main() -> None:
    gen = _load_generator()
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        good = _report(gen, [], tp)
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        stale = _report(gen, ["panel_disc_radius: geometry=200 vs npz=450"], tp)
    print("====ARM_GOOD====")
    print(good)
    print("====ARM_STALE====")
    print(stale)


if __name__ == "__main__":
    main()
