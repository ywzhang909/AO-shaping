"""Array-level distribution analysis of the ``ml/hwdataset`` hardware-debug corpus.

Where :mod:`scripts.generate_hwdataset_corpus_report` is metadata-only (index JSON
plus sidecar ``.json``, never opens a pickle), this script materialises a
**stratified, seeded sample** of records through the canonical
``ml.hwdataset.records.Materialiser`` and measures the *arrays* themselves --
per-family quantiles of the far-field image (brightness, contrast, 90 %-encircled
diameter, central-window energy, entropy) and of the coherent phase phasor
(contrast / spread). The result is ``report/hwdataset_corpus/distribution_stats.json``
-- the only artefact the report generator reads, so the generator stays a pure
offline renderer.

Method:
  * index from the existing ``data/hw_index_cache.json`` (no full re-scan);
  * per family, deterministically choose at most ``--per-family`` refs
    (``numpy.random.default_rng`` seeded on (SEED, family hash));
  * refs are grouped by pickle so each file is unpickled once (``cache_size=1``);
  * any record that raises :class:`ml.hwdataset.records.HwRecordError` is skipped
    and counted (coverage stays honest, no silent drop).

Determinism: fixed seed, fixed ``MaterialiserConfig`` (grid=64, ``abs255``).
Runtime is dominated by unpickling the ~40 sampled files; a full run takes on the
order of one to two minutes.

Usage (repo root)::

    python scripts/analyze_hwdataset_distributions.py
    python scripts/analyze_hwdataset_distributions.py --per-family 80 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from ml.hwdataset.index import build_hw_index  # noqa: E402
from ml.hwdataset.records import HwRecordError, Materialiser  # noqa: E402

INDEX_CACHE = ROOT / "data" / "hw_index_cache.json"
OUTPUT_PATH = ROOT / "report" / "hwdataset_corpus" / "distribution_stats.json"

SEED = 42
PER_FAMILY = 40
GRID = 64

#: Quantiles stored per metric column.
_QUANTILES = (10, 50, 90)


def _q(values: list[float], pct: float) -> float | None:
    """Percentile of a (possibly empty) list, ``None`` for the empty case."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _mean_median(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    arr = np.asarray(values, dtype=np.float64)
    return (float(arr.mean()), float(np.median(arr)))


def _col_stats(values: list[float]) -> dict[str, float | None]:
    """Quantile + mean + median block for one metric column."""
    m, med = _mean_median(values)
    return {f"p{p}": _q(values, p) for p in _QUANTILES} | {"mean": m, "median": med}


def _encircled_d90(image: np.ndarray) -> float | None:
    """90 %-energy encircled radius (grid px) around the intensity centroid.

    Returns ``None`` for frames whose total energy is below 1e-6 (dark / flat
    frames have no meaningful spot), so the caller counts them instead of
    emitting a garbage radius.
    """
    img = image.astype(np.float64)
    total = float(img.sum())
    if total <= 1e-6:
        return None
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    cx = float((img * xx).sum()) / total
    cy = float((img * yy).sum()) / total
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = float(r.max())
    radii = np.arange(0.5, rmax + 0.5, 1.0)
    cum = np.array([img[r <= rr].sum() for rr in radii]) / total
    hit = np.where(cum >= 0.90)[0]
    return float(radii[int(hit[0])]) if hit.size else float(rmax)


def _entropy256(image: np.ndarray) -> float | None:
    """Shannon entropy (bits) of the peak-normalised image over 256 bins."""
    img = image.astype(np.float64)
    peak = float(img.max())
    if peak <= 1e-12:
        return None
    p = (img / peak).ravel()
    p = p[p > 0]
    if p.size == 0:
        return None
    levels = np.clip((p * 256).astype(np.int64), 0, 255)
    _, counts = np.unique(levels, return_counts=True)
    q = counts / counts.sum()
    return float(-(q * np.log2(q)).sum())


def _analyze_sample(sample: Any) -> dict[str, float | None]:
    """All per-record metrics from one :class:`ml.hwdataset.records.HwSample`."""
    img = sample.image.astype(np.float64)
    total = float(img.sum())
    mean = float(img.mean())
    std = float(img.std())
    cos, sin = sample.phase_cos, sample.phase_sin
    ph = np.arctan2(sin, cos)
    p99 = float(np.percentile(img, 99))
    out: dict[str, float | None] = {
        "frame_max": float(img.max()),
        "frame_mean": mean,
        "frame_log10mean": math.log10(mean) if mean > 0 else None,
        "frame_cv": (std / mean) if mean > 0 else None,
        "frame_p99": p99,
        "frame_log10p99": math.log10(p99) if p99 > 0 else None,
        "d90": _encircled_d90(img),
        "central_energy": (float(img[16:48, 16:48].sum()) / total) if total > 1e-9 else None,
        "entropy256": _entropy256(img),
        "phase_coh_mean": float(sample.contrast.mean()),
        "phase_coh_max": float(sample.contrast.max()),
        "phase_spread": float(
            np.sqrt(1.0 - (np.cos(ph).mean()) ** 2 + (np.sin(ph).mean()) ** 2)
        ),
    }
    # FOV-relative 90%-energy radius: the one diameter number that is comparable
    # across families with different camera windows (report §5 / new §10).
    if out["d90"] is not None and sample.fov_px:
        out["d90_ratio"] = float(out["d90"]) / float(sample.fov_px)
    return out


def _pick_refs(index: Any, per_family: int, seed: int) -> list[Any]:
    """Deterministic per-family stratified sample of record refs."""
    by_family: dict[str, list[Any]] = defaultdict(list)
    for ref in index.records:
        by_family[ref.family].append(ref)
    refs: list[Any] = []
    for family in sorted(by_family):
        pool = by_family[family]
        rng = np.random.default_rng([seed, sum(ord(c) for c in family) % 10_000])
        k = min(per_family, len(pool))
        chosen = rng.choice(len(pool), size=k, replace=False)
        refs.extend(pool[i] for i in sorted(int(c) for c in chosen))
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Array-level distribution analysis of the hwdataset corpus (sampled)."
    )
    parser.add_argument("--index-cache", type=Path, default=INDEX_CACHE)
    parser.add_argument("--per-family", type=int, default=PER_FAMILY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    t0 = time.monotonic()
    logger.info("building index from {}", args.index_cache)
    index = build_hw_index(index_cache=args.index_cache)
    logger.info("index: {} records", len(index.records))

    refs = _pick_refs(index, args.per_family, args.seed)
    logger.info("sampled {} refs", len(refs))

    # Group by file so each pickle is unpickled once (PayloadStore LRU = 1).
    by_file: dict[Path, list[Any]] = defaultdict(list)
    for ref in refs:
        by_file[Path(ref.path)].append(ref)

    materialiser = Materialiser(cache_size=1, use_cache=False)
    per_family_cols: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_family_exposure: dict[str, list[float]] = defaultdict(list)
    per_family_n: Counter[str] = Counter()
    skipped: list[dict[str, Any]] = []
    n_ok = 0

    for path in sorted(by_file):
        for ref in by_file[path]:
            try:
                sample = materialiser.materialise(ref)
            except HwRecordError as exc:
                skipped.append(
                    {"path": str(ref.path), "position": ref.position, "error": str(exc)}
                )
                continue
            stats = _analyze_sample(sample)
            n_ok += 1
            per_family_n[ref.family] += 1
            if ref.exposure_ms is not None:
                per_family_exposure[ref.family].append(float(ref.exposure_ms))
            for key, value in stats.items():
                if value is not None:
                    per_family_cols[ref.family][key].append(float(value))

    families_out: dict[str, Any] = {}
    for family in sorted(per_family_cols):
        stats_out = {key: _col_stats(values) for key, values in sorted(per_family_cols[family].items())}
        exp = per_family_exposure.get(family, [])
        families_out[family] = {
            "n": int(per_family_n[family]),
            "exposure": {
                "min": min(exp) if exp else None,
                "max": max(exp) if exp else None,
                "median": float(np.median(exp)) if exp else None,
            },
            "stats": stats_out,
        }

    payload: dict[str, Any] = {
        "version": 1,
        "seed": args.seed,
        "per_family": args.per_family,
        "grid": GRID,
        "image_mode": "abs255",
        "index_records": len(index.records),
        "sampled": n_ok,
        "files_touched": len(by_file),
        "families": families_out,
        "skipped": skipped,
        "wall_s": round(time.monotonic() - t0, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "wrote {} ({} samples, {} files, {} skipped, {:.1f} s)",
        args.output.relative_to(ROOT),
        n_ok,
        len(by_file),
        len(skipped),
        payload["wall_s"],
    )


if __name__ == "__main__":
    main()
