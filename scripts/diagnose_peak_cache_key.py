"""Is the peak/abs255 target discrepancy a cache-key bug or a real transform difference?

`scripts/diagnose_peak_pipeline_gap.py` found the two 'peak-normalised' pipelines
disagree on 53.8% of pixels, which algebraically they cannot:

    (raw / 255) / max(raw / 255)  ==  raw / max(raw)

so the difference has to come from somewhere else. The prime suspect is the
derived-grid mmap cache: `Materialiser.use_cache=True` writes
``.hw_cache/<stem>/`` next to the pickle, and if that directory is keyed without
``image_mode`` then whichever pipeline ran first owns the arrays and the second
reads the wrong normalisation while still reporting its own ``image_mode``.

This script isolates it by materialising the *same* records twice per mode:

* with ``use_cache=True``  -- the production path, cache shared or not;
* with ``use_cache=False`` -- the direct path, no cache involved.

If the direct paths agree with each other but the cached ones disagree with them,
the cache is not keyed on ``image_mode`` and every cached run in this project has
been reading a neighbour's normalisation.

Usage
-----
    python scripts/diagnose_peak_cache_key.py
"""
from __future__ import annotations

import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from ml.hwdataset import MaterialiserConfig, build_hw_index
from ml.hwdataset.zernike_dataset import ZERNIKE_COEFF_SOURCES, ZernikeCoeffDataset

FAMILY = "slm_zernike_shaping"


def grab(ds, positions):
    return np.stack([ds[p]["image"].reshape(-1).numpy() for p in positions])


def main() -> int:
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY])
    records = list(index.records)
    positions = list(range(min(24, len(records))))

    report: dict = {"n_probe_records": len(positions)}

    def build(mode: str, use_cache: bool):
        return ZernikeCoeffDataset(
            index,
            config=MaterialiserConfig(grid=64, image_mode=mode),
            use_cache=use_cache,
        )

    # --- direct path: no cache anywhere, so this is the ground truth ---------
    direct_peak = grab(build("peak", use_cache=False), positions)
    direct_abs = grab(build("abs255", use_cache=False), positions)

    # What the dataset says it was asked for, to catch a silent override.
    report["reported_mode_peak"] = build("peak", use_cache=False).config.image_mode
    report["reported_mode_abs255"] = build("abs255", use_cache=False).config.image_mode

    def stats(a: np.ndarray, b: np.ndarray) -> dict:
        diff = np.abs(a - b)
        return {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "frac_pixels_differing": float((diff > 0).mean()),
        }

    report["direct_peak_vs_direct_abs255"] = stats(direct_peak, direct_abs)
    print("direct(peak) vs direct(abs255)          : "
          "max|d|=%.6g  pixels differing=%.2f%%"
          % (report["direct_peak_vs_direct_abs255"]["max_abs_diff"],
             100 * report["direct_peak_vs_direct_abs255"]["frac_pixels_differing"]))

    # A direct abs255 frame divided by its own amax must equal the direct peak
    # frame if the two code paths implement the same normalisation.
    amp = np.abs(direct_abs).max(axis=1, keepdims=True)
    manual = direct_abs / np.maximum(amp, 1e-8)
    report["direct_peak_vs_manual_abs255_over_amax"] = stats(direct_peak, manual)
    print("direct(peak) vs abs255/max(abs255)      : "
          "max|d|=%.6g  pixels differing=%.2f%%"
          % (report["direct_peak_vs_manual_abs255_over_amax"]["max_abs_diff"],
             100 * report["direct_peak_vs_manual_abs255_over_amax"]["frac_pixels_differing"]))

    # --- cached path: peak first, then abs255, both sharing .hw_cache --------
    cached_peak = grab(build("peak", use_cache=True), positions)
    cached_abs_after_peak = grab(build("abs255", use_cache=True), positions)
    report["cached_peak_vs_cached_abs255_after_peak"] = stats(
        cached_peak, cached_abs_after_peak)
    print("cached(peak) vs cached(abs255) [peak 1st]: "
          "max|d|=%.6g  pixels differing=%.2f%%"
          % (report["cached_peak_vs_cached_abs255_after_peak"]["max_abs_diff"],
             100 * report["cached_peak_vs_cached_abs255_after_peak"]["frac_pixels_differing"]))

    report["cached_peak_vs_direct_peak"] = stats(cached_peak, direct_peak)
    print("cached(peak) vs direct(peak)             : "
          "max|d|=%.6g  pixels differing=%.2f%%"
          % (report["cached_peak_vs_direct_peak"]["max_abs_diff"],
             100 * report["cached_peak_vs_direct_peak"]["frac_pixels_differing"]))

    report["cached_abs_after_peak_vs_direct_abs255"] = stats(
        cached_abs_after_peak, direct_abs)
    print("cached(abs255 after peak) vs direct(abs255): "
          "max|d|=%.6g  pixels differing=%.2f%%"
          % (report["cached_abs_after_peak_vs_direct_abs255"]["max_abs_diff"],
             100 * report["cached_abs_after_peak_vs_direct_abs255"]["frac_pixels_differing"]))

    # The decisive question: does a cached abs255 read return abs255 values, or
    # the peak values a previous cached peak run left behind?
    cache_leak = stats(cached_abs_after_peak, direct_peak)
    report["cached_abs_vs_direct_peak_is_cache_leak"] = cache_leak
    leak = cache_leak["frac_pixels_differing"] < 0.01
    report["verdict_cache_leak"] = bool(leak)
    print("\ncached(abs255) matches direct(peak) to within %.2f%% of pixels: %s"
          % (100 * cache_leak["frac_pixels_differing"], "YES -> CACHE LEAK" if leak else "no"))

    Path("logs/peak_cache_key.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote logs/peak_cache_key.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
