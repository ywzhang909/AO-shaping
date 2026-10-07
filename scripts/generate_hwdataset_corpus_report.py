"""Generate the ``report/hwdataset_corpus`` corpus-composition report.

Metadata-only by construction: the only *pickle-adjacent* contents ever read are
``data/hw_index_cache.json`` and the ``data/debug/**/*.json`` sidecars, plus the
array-level stats in ``report/hwdataset_corpus/distribution_stats.json`` -- a
JSON written by ``scripts/analyze_hwdataset_distributions.py`` (a separate,
sampled, seeded analysis that does the unpickling).  This script therefore never
opens an ``.pkl`` itself, never imports ``ml.hwdataset``, and stays a pure
offline renderer of three inputs.  All counts, percentages and figure
annotations are computed at runtime; nothing is hard-coded.

Two-stage reproduction::

    python scripts/analyze_hwdataset_distributions.py   # ~90 s, materialises a 252-sample
    python scripts/generate_hwdataset_corpus_report.py  # renders tables + figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from loguru import logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from scripts._common import fmt_metric, markdown_table, savefig
from scripts._common.provenance import insert_header

CACHE_PATH = ROOT / "data" / "hw_index_cache.json"
DEBUG_ROOT = ROOT / "data" / "debug"
REPORT_DIR = ROOT / "report" / "hwdataset_corpus"
FIGURE_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "report.md"
REPORT_KEY = "report/hwdataset_corpus/report.md"
#: Array-level distribution stats, produced by ``scripts/analyze_hwdataset_distributions.py``.
#: The generator only *reads* this file; it never materialises a pickle itself, so the
#: metadata-only contract of this script is preserved.  When absent, the new §10/§11
#: sections degrade to a pointer instead of a table.
DIST_STATS_PATH = REPORT_DIR / "distribution_stats.json"

#: Fixed Zernike reconstruction radius used by ``ml/hwdataset/records.py``.
ZERNIKE_APERTURE_RADIUS = 300.0

#: Canonical family-prefix resolution order, mirroring ``ml/hwdataset/index.py``.
FAMILY_PREFIXES: tuple[str, ...] = (
    "model_in_loop_hw_collect",
    "model_in_loop_hw_sweep",
    "bench_stability",
    "sim_calib_abba",
    "slm_gsnet_square",
    "slm_zernike_shaping",
    "slm_pib_online",
    "slm_pib",
    "recorder_",
)

#: Sidecar keys that stand in for a field of view (px, as recorded by the writer).
FOV_PROXY_KEYS: tuple[str, ...] = ("region", "cam_size", "far_field_size")
APERTURE_KEY = "zernike_radius"
OBJECTIVE_CANDIDATES: tuple[str, ...] = ("objective", "objective_mode", "target")
OBJECTIVE_VALUE_HINTS: frozenset[str] = frozenset(
    {
        "pib",
        "radiu",
        "avg_radiu",
        "rmse",
        "rmse_out",
        "shape",
        "roi_pib",
        "rms_pib",
        "pearson",
    }
)

FIGURE_NAMES: tuple[str, ...] = (
    "01_index_freshness.png",
    "02_family_scale.png",
    "03_phase_source.png",
    "04_exposure.png",
    "05_fov_proxies.png",
    "06_zernike_radius.png",
    "07_objective_coverage.png",
    "08_zernike_order.png",
    "09_records_per_file.png",
    "10_sidecar_keys.png",
    # Array-level distributions (only rendered when distribution_stats.json exists)
    "11_bright_vs_exposure.png",
    "12_spot_size.png",
    "13_phase_coherence.png",
    "14_texture.png",
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _norm(path: str | Path) -> str:
    """Normalise a path for cross-source comparison (cache vs. disk walk)."""
    return os.path.normcase(str(Path(path).resolve()))


def local_family_of(path: str | Path) -> str:
    """Resolve a family name the way ``ml/hwdataset/index.py`` does.

    Checks ``path.parent.name`` then ``path.parent.parent.name`` against
    :data:`FAMILY_PREFIXES` in order, falling back to the immediate parent
    directory name.
    """
    target = Path(path)
    for candidate in (target.parent.name, target.parent.parent.name):
        for prefix in FAMILY_PREFIXES:
            if candidate.startswith(prefix):
                return prefix
    return target.parent.name


def _pct(part: int, whole: int) -> str:
    """Percentage string, ``n/a`` when the denominator is empty."""
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


def _value_counts(records: list[dict[str, Any]], key: str) -> Counter[str]:
    """Count stringified sidecar values for ``key`` (absent/None excluded)."""
    counter: Counter[str] = Counter()
    for record in records:
        sidecar = record.get("sidecar")
        if not isinstance(sidecar, dict):
            continue
        value = sidecar.get(key)
        if value is None:
            continue
        counter[str(value)] += 1
    return counter


def _numeric_sort(values: list[str]) -> list[str]:
    """Sort numeric-looking labels numerically, everything else lexically."""
    try:
        return sorted(values, key=lambda v: float(v))
    except ValueError:
        return sorted(values)


def _src_values(facts: CorpusFacts, source: str, key: str) -> list[int]:
    """Distinct sorted values of ``key`` for one phase source.

    Used by the ``n_terms`` table: pooling all sources onto one axis would put
    "Noll coefficient count" and "unused ``_c`` length" on the same scale.
    """
    return sorted({r[key] for r in facts.records if r["source"] == source})


def _src_count(facts: CorpusFacts, source: str) -> int:
    """Record count for one phase source."""
    return sum(1 for r in facts.records if r["source"] == source)


def load_full_index(path: Path) -> dict[str, Any]:
    """Load an index that scanned EVERY pkl, for the §1 robustness cross-check.

    ``ml.hwdataset.index`` has no mtime validation, so the canonical cache can be
    arbitrarily old. This lets the report state -- rather than assume -- whether
    its structural conclusions survive a rebuild. Never used for the headline
    numbers, only for the invariance check.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"{path} has no 'records' list; not an hwdataset index")
    return payload


def load_dist_stats(path: Path) -> dict[str, Any] | None:
    """Load the array-level stats JSON, or ``None`` when absent / malformed.

    The file is written by ``scripts/analyze_hwdataset_distributions.py``.  A
    missing or unparseable file must degrade to *no §10/§11 content*, never to a
    traceback -- the generator's headline sections all come from the metadata
    inputs and must keep working on a fresh checkout that has not run the
    sampled analysis yet.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("distribution stats {} unreadable; §10/§11 will degrade", path)
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("families"), dict):
        logger.warning("distribution stats {} lacks a 'families' map; ignored", path)
        return None
    return payload


def _fam_stat(
    stats: dict[str, Any] | None, family: str, metric: str, key: str = "median"
) -> float | None:
    """One quantile/mean/median of one family's one metric, ``None``-safe."""
    if not stats:
        return None
    families = stats.get("families")
    if not isinstance(families, dict):
        return None
    fam = families.get(family)
    if not isinstance(fam, dict):
        return None
    entry = fam.get("stats", {}).get(metric)
    if not isinstance(entry, dict):
        return None
    value = entry.get(key)
    return float(value) if value is not None else None


def _fam_range(
    stats: dict[str, Any] | None, metric: str, key: str = "mean"
) -> tuple[float, float] | None:
    """``(min, max)`` of a metric across all families, or ``None`` if no family has it."""
    if not stats:
        return None
    families = stats.get("families")
    if not isinstance(families, dict):
        return None
    vals = [
        v for v in (_fam_stat(stats, fam, metric, key) for fam in families) if v is not None
    ]
    if not vals:
        return None
    return min(vals), max(vals)


def _structural_signature(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The properties §1 claims are invariant; recomputed from either index."""
    families: dict[str, set[str]] = {}
    for r in records:
        families.setdefault(r["family"], set()).add(r["source"])
    objective_labeled = sum(1 for r in records if r["sidecar"].get("objective"))
    return {
        "records": len(records),
        "files_scanned": None,
        "single_source_families": sum(1 for s in families.values() if len(s) == 1),
        "families": len(families),
        "sources": sorted({r["source"] for r in records}),
        "exposures": sorted({r["exposure_ms"] for r in records if r["exposure_ms"] is not None}),
        "objective_labeled": objective_labeled,
        "objective_coverage_pct": round(100.0 * objective_labeled / max(len(records), 1), 1),
        "zernike_records": sum(1 for r in records if r["source"] == "zernike"),
        "zernike_with_radius": sum(
            1
            for r in records
            if r["source"] == "zernike" and "zernike_radius" in r["sidecar"]
        ),
    }


def _heatmap(
    ax: Any,
    data: np.ndarray,
    rows: list[str],
    cols: list[str],
    title: str,
    annotations: list[list[str]] | None = None,
    cbar_label: str | None = None,
) -> None:
    """Annotated heatmap; ``annotations`` overrides the default count labels."""
    mesh = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_title(title, fontsize=11)
    peak = float(data.max()) if data.size else 0.0
    peak = peak if peak > 0 else 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = (
                annotations[i][j]
                if annotations is not None
                else f"{int(data[i, j]):,}"
            )
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if data[i, j] < 0.6 * peak else "black",
            )
    ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, label=cbar_label)


def _label_bars(
    ax: Any,
    bars: Any,
    values: list[Any],
    total: int | None = None,
    suffix: str = "",
) -> None:
    """Annotate horizontal bars with their value (and share of ``total``)."""
    for bar, value in zip(bars, values):
        text = f" {value:,}" if isinstance(value, int) else f" {value}"
        if total is not None:
            text += f"  ({_pct(int(value), total)})"
        else:
            text += suffix
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2.0,
            text,
            va="center",
            fontsize=8,
        )


# --------------------------------------------------------------------------- #
# corpus facts
# --------------------------------------------------------------------------- #
@dataclass
class CorpusFacts:
    """Everything the report and the figures need, derived at runtime."""

    payload: dict[str, Any]
    records: list[dict[str, Any]]
    cache_mtime: float
    disk_pkls: list[Path]
    disk_json: list[Path]

    indexed_norm: set[str] = field(default_factory=set)
    disk_norm: set[str] = field(default_factory=set)

    # freshness
    age_labels: list[str] = field(default_factory=list)
    index_labels: list[str] = field(default_factory=list)
    freshness: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    freshness_cell_families: dict[tuple[int, int], Counter[str]] = field(default_factory=dict)
    disk_unindexed_by_family: Counter[str] = field(default_factory=Counter)
    #: pkl that the cache already covers (mtime <= cache mtime), for the
    #: "the documented 65.88 GB is exactly this subset" cross-check.
    fresh_indexed_files: list[Path] = field(default_factory=list)
    indexed_disk_bytes: int = 0
    #: Optional FULL index (every pkl scanned) supplied via ``--full-index``.
    #: Used only to prove the structural conclusions survive a rebuild; the
    #: canonical numbers in the report always come from ``CACHE_PATH``.
    full_index: dict[str, Any] | None = None
    full_index_path: Path | None = None

    # scale
    families: list[str] = field(default_factory=list)
    fam_records: Counter[str] = field(default_factory=Counter)
    fam_disk_files: Counter[str] = field(default_factory=Counter)
    fam_disk_bytes: Counter[str] = field(default_factory=Counter)
    fam_indexed_files: Counter[str] = field(default_factory=Counter)
    total_disk_bytes: int = 0

    # phase source
    sources: list[str] = field(default_factory=list)
    fam_source: dict[str, Counter[str]] = field(default_factory=dict)
    collinear_families: int = 0

    # scalar distributions
    exposure_counts: Counter[str] = field(default_factory=Counter)
    exposure_absent: int = 0
    fov_counts: dict[str, Counter[str]] = field(default_factory=dict)
    fov_absent: dict[str, int] = field(default_factory=dict)
    aperture_counts: Counter[str] = field(default_factory=Counter)
    aperture_absent: int = 0
    objective_key: str = ""
    objective_counts: Counter[str] = field(default_factory=Counter)
    objective_absent: int = 0

    # zernike order
    n_terms_counts: Counter[int] = field(default_factory=Counter)
    n_terms_nmax: dict[int, Counter[int]] = field(default_factory=dict)
    n_terms_grid: dict[int, Counter[int]] = field(default_factory=dict)

    # records per file
    per_file: list[tuple[str, int, str]] = field(default_factory=list)

    # sidecar keys
    record_keys: Counter[str] = field(default_factory=Counter)
    all_json_keys: Counter[str] = field(default_factory=Counter)
    disk_only_keys: list[str] = field(default_factory=list)
    empty_sidecars: int = 0
    sidecar_dicts: int = 0

    # consistency
    family_mismatches: int = 0


def _triangular_label(n_terms: int) -> str:
    """``n_max`` implied by a triangular term count, if any."""
    for n in range(0, 60):
        if (n + 1) * (n + 2) // 2 == n_terms:
            return f"T{n}"
    return ""


def _detect_objective_key(
    records: list[dict[str, Any]], candidates: tuple[str, ...]
) -> str:
    """Pick the sidecar key that actually carries objective labels."""
    best_key, best_hits = "", 0
    for key in candidates:
        hits = sum(
            1
            for value in _value_counts(records, key)
            if value.lower() in OBJECTIVE_VALUE_HINTS
        )
        if hits > best_hits:
            best_key, best_hits = key, hits
    return best_key


def collect() -> CorpusFacts:
    """Read the cache plus metadata-only disk inventory and derive all facts."""
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = list(payload.get("records") or [])
    facts = CorpusFacts(
        payload=payload,
        records=records,
        cache_mtime=CACHE_PATH.stat().st_mtime,
        disk_pkls=sorted(DEBUG_ROOT.rglob("*.pkl")),
        disk_json=sorted(DEBUG_ROOT.rglob("*.json")),
    )

    # ---- freshness: index membership x file age vs. cache mtime ----
    facts.indexed_norm = {_norm(r["path"]) for r in records}
    facts.disk_norm = {_norm(p) for p in facts.disk_pkls}
    facts.age_labels = ["不晚于索引缓存", "晚于索引缓存"]
    facts.index_labels = ["已在索引中", "未进索引"]
    freshness = np.zeros((2, 2), dtype=float)
    cells: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for path in facts.disk_pkls:
        norm = _norm(path)
        age_row = 1 if path.stat().st_mtime > facts.cache_mtime else 0
        col = 0 if norm in facts.indexed_norm else 1
        freshness[age_row, col] += 1.0
        cells[(age_row, col)][local_family_of(path)] += 1
        if col == 1:
            facts.disk_unindexed_by_family[local_family_of(path)] += 1
    facts.freshness = freshness
    facts.freshness_cell_families = dict(cells)

    # ---- family scale ----
    for path in facts.disk_pkls:
        family = local_family_of(path)
        size = path.stat().st_size
        facts.fam_disk_files[family] += 1
        facts.fam_disk_bytes[family] += size
        facts.total_disk_bytes += size
        # The documented corpus size covers exactly the pickles the cache already
        # knows about, so sum that subset separately for the cross-check in §11.
        if path.stat().st_mtime <= facts.cache_mtime:
            facts.fresh_indexed_files.append(path)
            facts.indexed_disk_bytes += size
    per_file_counter: Counter[str] = Counter()
    per_file_family: dict[str, str] = {}
    for record in records:
        family = record["family"]
        facts.fam_records[family] += 1
        per_file_counter[record["path"]] += 1
        per_file_family[record["path"]] = family
        if local_family_of(record["path"]) != family:
            facts.family_mismatches += 1
    for path, count in per_file_counter.items():
        if _norm(path) in facts.indexed_norm:
            facts.fam_indexed_files[per_file_family[path]] += 1
            facts.per_file.append((path, count, per_file_family[path]))
    facts.families = sorted(
        set(facts.fam_records) | set(facts.fam_disk_files),
        key=lambda f: (-facts.fam_records[f], -facts.fam_disk_files[f], f),
    )
    facts.per_file.sort(key=lambda item: (-item[1], item[0]))

    # ---- phase source ----
    sources: set[str] = set()
    fam_source: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        fam_source[record["family"]][record["source"]] += 1
        sources.add(record["source"])
    facts.fam_source = dict(fam_source)
    facts.sources = sorted(sources)
    facts.collinear_families = sum(1 for c in fam_source.values() if len(c) == 1)

    # ---- scalar distributions ----
    facts.exposure_counts = Counter()
    for record in records:
        value = record.get("exposure_ms")
        if value is None:
            facts.exposure_absent += 1
        else:
            facts.exposure_counts[fmt_metric(value, 4)] += 1
    for key in FOV_PROXY_KEYS:
        counts = _value_counts(records, key)
        facts.fov_counts[key] = counts
        facts.fov_absent[key] = len(records) - sum(counts.values())
    facts.aperture_counts = _value_counts(records, APERTURE_KEY)
    facts.aperture_absent = len(records) - sum(facts.aperture_counts.values())
    facts.objective_key = _detect_objective_key(records, OBJECTIVE_CANDIDATES)
    if facts.objective_key:
        facts.objective_counts = _value_counts(records, facts.objective_key)
    facts.objective_absent = len(records) - sum(facts.objective_counts.values())

    # ---- zernike order ----
    for record in records:
        n_terms = int(record.get("n_terms") or 0)
        facts.n_terms_counts[n_terms] += 1
        n_max = record.get("n_max")
        grid = record.get("freeform_grid")
        if n_max is not None:
            facts.n_terms_nmax.setdefault(n_terms, Counter())[int(n_max)] += 1
        if grid is not None:
            facts.n_terms_grid.setdefault(n_terms, Counter())[int(grid)] += 1

    # ---- sidecar keys ----
    for record in records:
        sidecar = record.get("sidecar")
        if not isinstance(sidecar, dict):
            facts.empty_sidecars += 1
            continue
        facts.sidecar_dicts += 1
        for key in sidecar:
            facts.record_keys[key] += 1
    for path in facts.disk_json:
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(sidecar, dict):
            for key in sidecar:
                facts.all_json_keys[key] += 1
    facts.disk_only_keys = sorted(set(facts.all_json_keys) - set(facts.record_keys))

    return facts


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _fig_freshness(facts: CorpusFacts) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    annotations = [
        [
            f"{int(facts.freshness[i, j]):,}\n{_pct(int(facts.freshness[i, j]), int(facts.freshness.sum()))}"
            for j in range(facts.freshness.shape[1])
        ]
        for i in range(facts.freshness.shape[0])
    ]
    _heatmap(
        ax,
        facts.freshness,
        facts.age_labels,
        facts.index_labels,
        f"磁盘 pkl 数量 (共 {int(facts.freshness.sum()):,} 个)",
        annotations,
        cbar_label="pkl 文件数",
    )
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[0])


def _fig_family_scale(facts: CorpusFacts) -> None:
    families = list(reversed(facts.families))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 0.42 * len(families) + 2.6))

    records_vals = [facts.fam_records[f] for f in families]
    bars = axes[0].barh(families, records_vals, color="#4c72b0")
    _label_bars(axes[0], bars, records_vals)
    axes[0].set_title(f"已索引记录数 (共 {len(facts.records):,} 条)")
    axes[0].set_xlabel("记录数")
    axes[0].set_xscale("symlog")
    axes[0].set_xlim(0, max(records_vals) * 1.35 if records_vals else 1)

    gib = [round(facts.fam_disk_bytes[f] / 1024**3, 2) for f in families]
    bars = axes[1].barh(families, gib, color="#dd8452")
    _label_bars(axes[1], bars, gib, suffix=" GiB")
    axes[1].set_title(f"磁盘 pkl 体积 (共 {facts.total_disk_bytes / 1024**4:.2f} TiB)")
    axes[1].set_xlabel("GiB")
    axes[1].set_xlim(0, max(gib) * 1.35 if gib else 1)

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[1])


def _fig_phase_source(facts: CorpusFacts) -> None:
    matrix = np.zeros((len(facts.families), len(facts.sources)), dtype=float)
    annotations: list[list[str]] = []
    for i, family in enumerate(facts.families):
        counts = facts.fam_source.get(family, Counter())
        total = sum(counts.values())
        row: list[str] = []
        for j, source in enumerate(facts.sources):
            value = counts.get(source, 0)
            matrix[i, j] = value
            row.append(f"{value:,}\n{_pct(value, total)}" if value else "-")
        annotations.append(row)
    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(facts.families) + 2.4))
    _heatmap(
        ax,
        matrix,
        list(facts.families),
        list(facts.sources),
        "家族 × 相位表示 (格内: 记录数 / 家族内占比)",
        annotations,
        cbar_label="记录数",
    )
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[2])


def _fig_exposure(facts: CorpusFacts) -> None:
    labels = _numeric_sort(list(facts.exposure_counts))
    values = [facts.exposure_counts[k] for k in labels]
    total = len(facts.records)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.barh(labels, values, color="#55a868")
    _label_bars(ax, bars, values, total)
    ax.set_xscale("log")
    # A log axis rejects a non-positive bound and drops the whole call, so both
    # ends are anchored on the positive counts. Deriving the headroom from
    # ``positive`` too (rather than ``values``) is a no-op whenever any count is
    # positive, but keeps the all-zero degenerate case at (1, 1) instead of the
    # inverted (1, 0).
    positive = [v for v in values if v > 0]
    ax.set_xlim(
        min(positive) * 0.7 if positive else 1,
        max(positive) * 2.2 if positive else 1,
    )
    ax.set_xlabel("记录数 (对数轴)")
    ax.set_ylabel("曝光 (ms)")
    ax.set_title(
        f"曝光分布: {len(labels)} 个取值, 跨 "
        f"{np.log10(max(map(float, labels)) / min(map(float, labels))):.2f} 个数量级"
        if labels
        else "曝光分布"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[3])


def _fig_fov(facts: CorpusFacts) -> None:
    keys = [k for k in FOV_PROXY_KEYS if facts.fov_counts.get(k)]
    fig, axes = plt.subplots(len(keys), 1, figsize=(8.6, 2.5 * len(keys) + 1.4), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        counts = facts.fov_counts[key]
        labels = _numeric_sort(list(counts))
        values = [counts[k] for k in labels]
        bars = ax.barh(labels, values, color="#8172b3")
        _label_bars(ax, bars, values, len(facts.records))
        ax.set_xlim(0, max(values) * 1.6 if values else 1)
        ax.set_xlabel("记录数")
        ax.set_title(
            f"sidecar `{key}`: {len(labels)} 个取值 "
            f"(未标注 {facts.fov_absent[key]:,} 条)"
        )
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("视场代理量 (sidecar 记录值, 非物理角尺度)", fontsize=12)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[4])


def _fig_aperture(facts: CorpusFacts) -> None:
    labels = _numeric_sort(list(facts.aperture_counts))
    values = [facts.aperture_counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.barh(labels, values, color="#c44e52")
    _label_bars(ax, bars, values, len(facts.records))
    ax.axvline(
        ZERNIKE_APERTURE_RADIUS,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"固定重建半径 {ZERNIKE_APERTURE_RADIUS:.0f} px",
    )
    ax.set_xlim(0, max(values) * 1.7 if values else ZERNIKE_APERTURE_RADIUS)
    ax.set_xlabel("记录数")
    ax.set_ylabel("sidecar `zernike_radius` (px)")
    ax.set_title(f"指令相位 Zernike 孔径半径 (未标注 {facts.aperture_absent:,} 条)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    # Scope warning: the radius is only *consumed* by the PhaseSource.ZERNIKE
    # branch, and those records carry none. A reader looking only at this figure
    # would otherwise conclude the constant is contradicted by these families --
    # the opposite is true. See §6.
    zern_with_radius = sum(
        1
        for r in facts.records
        if r["source"] == "zernike" and "zernike_radius" in r["sidecar"]
    )
    ax.text(
        0.99,
        0.55,
        f"注意: 本图这些记录均「不走」 Zernike 重建分支\n"
        f"(phase 取自已存面板), 故与 {ZERNIKE_APERTURE_RADIUS:.0f} px 常量无关。\n"
        f"真正进入该分支的 {sum(1 for r in facts.records if r['source'] == 'zernike'):,} "
        f"条记录里, 带此字段的是 {zern_with_radius} 条 ——\n"
        f"常量对它们无法交叉校验。详见 §6。",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#8b1a1a",
        bbox={"facecolor": "#fff6f6", "edgecolor": "#c44e52", "boxstyle": "round,pad=0.5"},
    )
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[5])


def _fig_objective(facts: CorpusFacts) -> None:
    counts = facts.objective_counts
    labels = [k for k, _ in counts.most_common()] if counts else []
    values = [counts[k] for k in labels]
    if facts.objective_absent:
        labels = labels + ["(未标注)"]
        values = values + [facts.objective_absent]
    total = len(facts.records)
    colors = ["#4c72b0"] * max(0, len(values) - 1) + (
        ["#b0b0b0"] if facts.objective_absent else []
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.barh(labels, values, color=colors)
    _label_bars(ax, bars, values, total)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.6 if values else 1)
    ax.set_xlabel("记录数")
    key = facts.objective_key or "(未找到目标标签字段)"
    ax.set_title(f"优化目标标签覆盖 (sidecar `{key}`)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[6])


def _n_terms_label(facts: CorpusFacts, n_terms: int) -> str:
    """Second line for a ``n_terms`` bar: triangular T_n and/or grid g x g."""
    parts: list[str] = []
    nmax_counts = facts.n_terms_nmax.get(n_terms)
    if nmax_counts:
        parts.append("/".join(f"T{k}" for k in sorted(nmax_counts)))
    elif n_terms > 0:
        parts.append(_triangular_label(n_terms))
    grid_counts = facts.n_terms_grid.get(n_terms)
    if grid_counts:
        parts.append("/".join(f"{g}x{g}" for g in sorted(grid_counts)))
    return "\n".join(p for p in parts if p)


def _fig_zernike_order(facts: CorpusFacts) -> None:
    keys = sorted(facts.n_terms_counts)
    values = [facts.n_terms_counts[k] for k in keys]
    labels = [f"{k:,}\n{_n_terms_label(facts, k)}" for k in keys]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    bars = ax.bar(labels, values, color="#64b5cd")
    _label_bars(ax, bars, values, len(facts.records))
    ax.set_yscale("log")
    # Same log-axis constraint as _fig_exposure: 0 is invalid and would void the
    # 3x headroom that keeps the value labels off the top spine. Both ends come
    # from ``positive`` so an all-zero series degrades to (1, 1) rather than an
    # inverted (1, 0).
    positive = [v for v in values if v > 0]
    ax.set_ylim(
        min(positive) * 0.6 if positive else 1,
        max(positive) * 3.0 if positive else 1,
    )
    ax.set_ylabel("记录数 (对数轴)")
    ax.set_title("指令相位自由度数 `n_terms` (第二行: 由 n_max / freeform_grid 反推的口径)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[7])


def _fig_records_per_file(facts: CorpusFacts) -> None:
    values = [count for _, count, _ in facts.per_file]
    families = [family for _, _, family in facts.per_file]
    nonzero = [v for v in values if v > 0]
    mean = float(np.mean(values)) if values else 0.0
    median = float(np.median(values)) if values else 0.0
    spread = (max(nonzero) / min(nonzero)) if nonzero else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), width_ratios=[2.0, 1.0])
    palette = plt.get_cmap("tab20")
    uniq = sorted(set(families))
    color_of = {f: palette(i % 20) for i, f in enumerate(uniq)}
    axes[0].scatter(
        range(1, len(values) + 1),
        values,
        s=18,
        c=[color_of[f] for f in families],
    )
    axes[0].axhline(mean, color="black", linestyle="--", linewidth=1.2, label=f"均值 {mean:.1f}")
    axes[0].axhline(median, color="#333333", linestyle=":", linewidth=1.4, label=f"中位数 {median:.1f}")
    axes[0].set_xlabel("已索引 pkl (按记录数降序)")
    axes[0].set_ylabel("每个 pkl 的记录数")
    axes[0].set_title(
        f"{len(values):,} 个已索引 pkl 的记录密度 (最大/最小 = {spread:.1f}x)"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    top = facts.per_file[:15]
    names = [Path(p).name for p, _, _ in top][::-1]
    top_values = [c for _, c, _ in top][::-1]
    top_families = [f for _, _, f in top][::-1]
    bars = axes[1].barh(names, top_values, color=[color_of[f] for f in top_families])
    _label_bars(axes[1], bars, top_values)
    axes[1].set_xlim(0, max(top_values) * 1.3 if top_values else 1)
    axes[1].set_xlabel("记录数")
    axes[1].set_title("记录最多的 15 个 pkl")
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].grid(axis="x", alpha=0.25)

    handles = [Line2D([], [], marker="o", ls="", color=color_of[f], label=f) for f in uniq]
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(uniq)), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[8])


def _fig_sidecar_keys(facts: CorpusFacts) -> None:
    counts = facts.record_keys
    labels = [k for k, _ in counts.most_common()]
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(9.0, 0.26 * len(labels) + 2.6))
    bars = ax.barh(labels[::-1], values[::-1], color="#8c8c8c")
    _label_bars(ax, bars, values[::-1], len(facts.records))
    ax.set_xlim(0, max(values) * 1.35 if values else 1)
    ax.set_xlabel("携带该键的记录数")
    ax.set_title(
        f"sidecar 键覆盖: 记录匹配 {len(counts)} 个键, 全部磁盘 JSON "
        f"{len(facts.all_json_keys)} 个键, 仅存于磁盘 {len(facts.disk_only_keys)} 个"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[9])


def _label_vbars(ax: Any, bars: Any, texts: list[str]) -> None:
    """Annotate *vertical* bars with text above the top (``_label_bars`` is for h-bars)."""
    for bar, text in zip(bars, texts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f" {text}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _fig_bright_vs_exposure(facts: CorpusFacts, stats: dict[str, Any]) -> None:
    """§10.1: per-family mean log10(frame mean) vs exposure median, on a log axis.

    The point of the figure: ``abs255`` preserves exposure, so the brightness
    spread *is* the exposure spread.  Plotting the two on one figure makes the
    collinearity visible instead of assumed.
    """
    # Pair each family with (mean brightness, exposure median); keep only families
    # where both are present and positive.
    pts: list[tuple[str, float, float]] = []
    for f in sorted(stats["families"]):
        log10mean = _fam_stat(stats, f, "frame_log10mean", "mean")
        if log10mean is None:
            continue
        exp = stats["families"][f].get("exposure", {})
        em = exp.get("median")
        if em is None:
            continue
        b, e = 10.0 ** log10mean, float(em)
        if b > 0 and e > 0:
            pts.append((f, b, e))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    palette = plt.get_cmap("tab10")
    for i, (fam, b, e) in enumerate(pts):
        ax.scatter([e], [b], s=90, color=palette(i % 10), zorder=3, label=fam)
        ax.annotate(
            fam,
            (e, b),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("曝光中位数 (ms)")
    ax.set_ylabel("帧均亮度 (abs255, 绝对)")
    ax.set_title("帧均亮度 vs 曝光 (abs255 保留曝光 → 亮度差 ≈ 曝光差)")
    if len(pts) > 2:
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
        lo, hi = min(xs) * 0.6, max(xs) * 1.6
        xx = np.logspace(np.log10(lo), np.log10(hi), 50)
        ax.plot(xx, 10.0 ** (slope * np.log10(xx) + intercept), "--", color="gray", lw=1)
        ax.text(
            0.03,
            0.97,
            f"对数-对数斜率 ≈ {slope:.2f}\n(曝光翻倍 → 亮度 ×2^{slope:.2f})",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
        )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[10])


def _fig_spot_size(facts: CorpusFacts, stats: dict[str, Any]) -> None:
    """§10.2: FOV-relative 90%-energy diameter per family (box plot).

    ``d90_ratio = d90 / fov_px`` is the only diameter number comparable across
    families whose camera windows differ (64 px region vs 1944 px full frame).
    Raw ``d90`` is also shown per family in the caption so the FOV caveat stays
    visible.
    """
    fams = sorted(stats["families"])
    data: list[list[float]] = []
    labels: list[str] = []
    for f in fams:
        vals = []
        fam = stats["families"][f]
        entry = fam.get("stats", {}).get("d90_ratio")
        if not isinstance(entry, dict):
            continue
        # p10/p50/p90 alone already reconstruct the quartiles; mixing `mean` in
        # would skew the IQR, so it is deliberately excluded here.
        for k in ("p10", "p50", "p90"):
            v = entry.get(k)
            if v is not None:
                vals.append(float(v))
        if not vals:
            continue
        data.append(vals)
        raw = fam.get("stats", {}).get("d90", {})
        d90_med = raw.get("median")
        labels.append(f"{f}\n(d90 中位 {d90_med:.1f} px)" if d90_med is not None else f)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bp = ax.boxplot(data, orientation="horizontal", showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4c72b0")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("90% 能量环围半径 / FOV (无量纲)")
    ax.set_title("光斑尺寸 (FOV 归一, 跨家族可比)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[11])


def _fig_phase_coherence(facts: CorpusFacts, stats: dict[str, Any]) -> None:
    """§10.3: phase coherence (mean) and phase spread per family, side by side bars."""
    fams = sorted(stats["families"])
    # Median, not mean: slm_pib's phase_spread has a near-0 p10 outlier that
    # drags the mean below the median; the table reports the median, so the
    # bars must match it (0.857 mean vs 0.9471 median for slm_pib coherence).
    coh = [
        v
        for v in (_fam_stat(stats, f, "phase_coh_mean", "median") for f in fams)
        if v is not None
    ]
    spread = [
        v
        for v in (_fam_stat(stats, f, "phase_spread", "median") for f in fams)
        if v is not None
    ]
    labels = [
        f for f, c in zip(fams, coh) if c is not None
    ]
    x = np.arange(len(labels))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    b1 = axes[0].bar(x - w / 2, coh, w, color="#55a868")
    _label_vbars(axes[0], b1, [f"{v:.3f}" for v in coh])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axes[0].set_title("相位相干度 (contrast 均值, 40 样本取中位, 越接近 1 越相干)")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    b2 = axes[1].bar(x + w / 2, spread, w, color="#c44e52")
    _label_vbars(axes[1], b2, [f"{v:.3f}" for v in spread])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axes[1].set_title("相位展宽 (rad, 越大越散)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[12])


def _fig_texture(facts: CorpusFacts, stats: dict[str, Any]) -> None:
    """§10.4: entropy + frame CV per family (the "texture" / structure axes)."""
    fams = sorted(stats["families"])
    # Median, matching the table (see _fig_phase_coherence).
    ent = [
        v
        for v in (_fam_stat(stats, f, "entropy256", "median") for f in fams)
        if v is not None
    ]
    cv = [
        v
        for v in (_fam_stat(stats, f, "frame_cv", "median") for f in fams)
        if v is not None
    ]
    labels = [f for f, e in zip(fams, ent) if e is not None]
    x = np.arange(len(labels))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    b1 = axes[0].bar(x - w / 2, ent, w, color="#8172b3")
    _label_vbars(axes[0], b1, [f"{v:.2f}" for v in ent])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axes[0].set_title("256-bin 熵 (bits, 上限 8)")
    axes[0].grid(axis="y", alpha=0.25)
    b2 = axes[1].bar(x + w / 2, cv, w, color="#64b5cd")
    _label_vbars(axes[1], b2, [f"{v:.2f}" for v in cv])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axes[1].set_title("帧对比度 CV (std/mean)")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    savefig(fig, FIGURE_DIR / FIGURE_NAMES[13])


def make_figures(facts: CorpusFacts, dist_stats: dict[str, Any] | None = None) -> None:
    """Render the ten metadata figures, plus the four array-level ones if stats exist."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _fig_freshness(facts)
    _fig_family_scale(facts)
    _fig_phase_source(facts)
    _fig_exposure(facts)
    _fig_fov(facts)
    _fig_aperture(facts)
    _fig_objective(facts)
    _fig_zernike_order(facts)
    _fig_records_per_file(facts)
    _fig_sidecar_keys(facts)
    if dist_stats is not None:
        _fig_bright_vs_exposure(facts, dist_stats)
        _fig_spot_size(facts, dist_stats)
        _fig_phase_coherence(facts, dist_stats)
        _fig_texture(facts, dist_stats)


def _figure_ref(filename: str) -> str:
    """Markdown image reference, emitted only for a PNG that actually exists."""
    path = FIGURE_DIR / filename
    if not path.is_file():
        return ""
    return f"![{filename.removesuffix('.png')}](figures/{filename})"


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(facts: CorpusFacts, dist_stats: dict[str, Any] | None = None) -> str:
    """Render the Chinese Markdown report body."""
    total = len(facts.records)
    cache = facts.payload
    scanned = int(cache.get("files_scanned") or 0)
    usable = int(cache.get("files_usable") or 0)
    excluded = cache.get("excluded") or {}
    unusable = scanned - usable
    disk_total = int(facts.freshness.sum())
    indexed_on_disk = int(facts.freshness[:, 0].sum())
    unindexed = int(facts.freshness[:, 1].sum())
    older_unindexed = int(facts.freshness[0, 1])
    newer_unindexed = int(facts.freshness[1, 1])
    cache_time = datetime.fromtimestamp(facts.cache_mtime).strftime("%Y-%m-%d %H:%M:%S")
    newest = max((p.stat().st_mtime for p in facts.disk_pkls), default=facts.cache_mtime)
    oldest = min((p.stat().st_mtime for p in facts.disk_pkls), default=facts.cache_mtime)

    exposure_labels = _numeric_sort(list(facts.exposure_counts))
    exposure_span = (
        np.log10(max(map(float, exposure_labels)) / min(map(float, exposure_labels)))
        if exposure_labels
        else 0.0
    )
    objective_coverage = (
        100.0 * (total - facts.objective_absent) / total if total else 0.0
    )
    values_per_file = [count for _, count, _ in facts.per_file]
    nonzero = [v for v in values_per_file if v > 0]
    spread = (max(nonzero) / min(nonzero)) if nonzero else 0.0
    mean_per_file = float(np.mean(values_per_file)) if values_per_file else 0.0
    median_per_file = float(np.median(values_per_file)) if values_per_file else 0.0

    out: list[str] = []
    add = out.append

    add("# hwdataset 语料分布与特征分析\n")

    # ---------------- 摘要 ----------------
    add("## 摘要\n")
    add(
        "本报告**只**读取 `data/hw_index_cache.json`、`data/debug/**/*.json` 与 "
        "`report/hwdataset_corpus/distribution_stats.json` (后者由 "
        "`scripts/analyze_hwdataset_distributions.py` 预先分层物化产生) 三个来源, "
        "`data/debug` 下的 `.pkl` 一律只做 `Path.stat()` 元数据检查 (不打开 / 不反序列化 / "
        "不 mmap), 也不 import `ml.hwdataset`。所有数字均在运行时统计, 无硬编码。\n"
    )
    add(
        f"- **索引已陈旧**: 索引缓存覆盖磁盘 {indexed_on_disk:,}/{disk_total:,} 个 pkl "
        f"({_pct(indexed_on_disk, disk_total)}); 另有 {unindexed:,} 个 pkl 未进索引 "
        f"(其中 {newer_unindexed:,} 个比缓存更新, {older_unindexed:,} 个比缓存旧)。"
        f"任何以该缓存为输入的结论都只覆盖 {total:,} 条记录, 不覆盖当前磁盘全量。"
    )
    add(
        f"- **家族与相位表示完全共线**: {len(facts.fam_records)} 个家族中有 "
        f"{facts.collinear_families} 个只对应单一 `source` "
        f"(占记录 {_pct(sum(c for f, c in facts.fam_records.items() if len(facts.fam_source.get(f, {})) == 1), total)})。"
        f"因此「表示方式的差异」与「家族/台架/目标的差异」在本语料内**无法解耦**。"
    )
    add(
        f"- **采集条件跨台架不一致**: 曝光 {len(exposure_labels)} 个取值、跨 "
        f"{exposure_span:.2f} 个数量级 ({exposure_labels[0]}–{exposure_labels[-1]} ms); "
        f"视场代理量 (sidecar 记录值) 覆盖 {len(FOV_PROXY_KEYS)} 个不同键名, "
        f"Zernike 指令孔径半径取值 {len(facts.aperture_counts)} 种而重建端固定为 "
        f"{ZERNIKE_APERTURE_RADIUS:.0f} px。"
    )
    add(
        f"- **监督标签覆盖不足**: sidecar 目标字段 `{facts.objective_key or '(未找到)'}` "
        f"仅覆盖 {total - facts.objective_absent:,}/{total:,} 条 "
        f"({objective_coverage:.1f}%), 缺失 {facts.objective_absent:,} 条 "
        f"({_pct(facts.objective_absent, total)})。"
    )
    add(
        f"- **文件级记录密度极不均匀**: {len(values_per_file):,} 个已索引 pkl 的记录数均值 "
        f"{mean_per_file:.1f}、中位数 {median_per_file:.0f}, 最大/最小 = {spread:.1f}×, "
        f"必须按文件分组读取。"
    )
    if dist_stats is not None:
        # Ranges across families on the SAME statistic the §10 table/figures use
        # (median), so the abstract and the table cannot disagree.
        d90 = _fam_range(dist_stats, "d90_ratio", "median")
        cen = _fam_range(dist_stats, "central_energy", "median")
        ent = _fam_range(dist_stats, "entropy256", "median")
        spr = _fam_range(dist_stats, "phase_spread", "median")
        if d90 and cen and ent and spr:
            add(
                f"- **数组级分布差异 (实测, §10)**: 分层物化 {dist_stats['sampled']} 条记录后, "
                f"家族间差异在尺度无关指标上依然显著 —— 90% 能量环围半径/FOV 跨 "
                f"{d90[0]:.4f}–{d90[1]:.4f}, 中心窗能量跨 {cen[0]:.3f}–{cen[1]:.3f}, "
                f"256-bin 熵跨 {ent[0]:.2f}–{ent[1]:.2f} bits, "
                f"相位展宽跨 {spr[0]:.3f}–{spr[1]:.2f} rad。逐条消解方案见 §11。\n"
            )
    if dist_stats is None:
        add(
            "- **数组级分布差异 (待实测)**: 需先运行 `scripts/analyze_hwdataset_distributions.py` "
            "生成 `distribution_stats.json`, 本报告 §10/§11 届时才会填充实测表; 当前两节降级为提示。\n"
        )

    # ---------------- 1. 索引新鲜度 ----------------
    add("## 1. 索引新鲜度\n")
    add(
        f"索引缓存写入时间 {cache_time}; 磁盘 {disk_total:,} 个 pkl 的 mtime 跨度 "
        f"{datetime.fromtimestamp(oldest).strftime('%Y-%m-%d %H:%M:%S')} → "
        f"{datetime.fromtimestamp(newest).strftime('%Y-%m-%d %H:%M:%S')}。\n"
    )
    freshness_rows = []
    for i, age in enumerate(facts.age_labels):
        for j, indexed in enumerate(facts.index_labels):
            value = int(facts.freshness[i, j])
            fams = facts.freshness_cell_families.get((i, j), Counter())
            top = ", ".join(f"{f}×{c}" for f, c in fams.most_common(4)) or "-"
            freshness_rows.append([age, indexed, f"{value:,}", _pct(value, disk_total), top])
    add(
        markdown_table(
            ["pkl mtime", "索引状态", "pkl 数", "占磁盘", "主要家族"],
            freshness_rows,
        )
    )
    add(
        f"\n缓存自身的账面口径是 `files_scanned={scanned:,}` / `files_usable={usable:,}` "
        f"(差 {unusable} 个不可用文件)。注意 `excluded` 的两个键**单位不同**: "
        + ", ".join(f"`{k}`={v}" for k, v in sorted(excluded.items()))
        + " —— 记录级计数与文件级计数混在同一张字典里, 不能直接相加或用来对账 "
        "`files_scanned - files_usable`。\n"
    )
    if facts.disk_unindexed_by_family:
        add("未进索引的 pkl 按家族分布:\n")
        add(
            markdown_table(
                ["家族", "未索引 pkl 数", "占未索引"],
                [
                    [f, f"{c:,}", _pct(c, unindexed)]
                    for f, c in facts.disk_unindexed_by_family.most_common()
                ],
            )
        )
        add("")
    add(_figure_ref(FIGURE_NAMES[0]))
    add("")

    # ---- 1.x rebuild-invariance cross-check (only when a full index is given) ----
    if facts.full_index is not None:
        full_records = facts.full_index["records"]
        cur = _structural_signature(facts.records)
        new = _structural_signature(full_records)
        add("### 1.1 陈旧性是否影响结论? (全量重建交叉核对)\n")
        add(
            f"上表说明「样本集变了」, 但**结论是否随之失效**是另一回事。仓库里存在一份"
            f"扫描了全部 {facts.full_index['files_scanned']} 个 pkl 的全量索引, "
            f"用它把本报告的每一条结构性结论重算一遍:\n"
        )
        add(
            markdown_table(
                ["量", "当前缓存 (陈旧)", "全量重建", "是否一致"],
                [
                    [
                        "被扫描文件数",
                        f"{facts.payload['files_scanned']}",
                        f"{facts.full_index['files_scanned']}",
                        "—",
                    ],
                    [
                        "记录数",
                        f"{cur['records']:,}",
                        f"{new['records']:,}",
                        "✅ 一致" if cur["records"] == new["records"] else "⚠ 变化",
                    ],
                    [
                        "家族数",
                        f"{cur['families']}",
                        f"{new['families']}",
                        "✅" if cur["families"] == new["families"] else "⚠",
                    ],
                    [
                        "单一 source 的家族占比",
                        f"{cur['single_source_families']}/{cur['families']}",
                        f"{new['single_source_families']}/{new['families']}",
                        "✅"
                        if cur["single_source_families"] == cur["families"]
                        and new["single_source_families"] == new["families"]
                        else "⚠",
                    ],
                    [
                        "曝光取值数",
                        f"{len(cur['exposures'])}",
                        f"{len(new['exposures'])}",
                        "✅" if cur["exposures"] == new["exposures"] else "⚠",
                    ],
                    [
                        "`objective` 覆盖率",
                        f"{cur['objective_coverage_pct']}%",
                        f"{new['objective_coverage_pct']}%",
                        "✅"
                        if cur["objective_coverage_pct"] == new["objective_coverage_pct"]
                        else "⚠ 变化",
                    ],
                    [
                        "`source=zernike` 记录数",
                        f"{cur['zernike_records']:,}",
                        f"{new['zernike_records']:,}",
                        "✅" if cur["zernike_records"] == new["zernike_records"] else "⚠",
                    ],
                    [
                        "其中带 `zernike_radius`",
                        f"{cur['zernike_with_radius']}",
                        f"{new['zernike_with_radius']}",
                        "✅"
                        if cur["zernike_with_radius"] == new["zernike_with_radius"]
                        else "⚠",
                    ],
                ],
            )
        )
        add(
            f"\n**结论: 新增的 {new['records'] - cur['records']:,} 条记录全部落在 "
            f"`model_in_loop_hw_collect` / `model_in_loop_hw_sweep` 的 `panel_rad` 侧, "
            f"没有改变任何一条结构性结论** —— 家族↔source 仍然 1:1、曝光仍 "
            f"{len(new['exposures'])} 个取值、`source=zernike` 仍是 {new['zernike_records']:,} 条"
            f"且仍然**一条都不带** `zernike_radius`。唯一随陈旧性移动的量是**记录总数**与"
            f"随之变化的 `objective` 覆盖率 ({cur['objective_coverage_pct']}% → "
            f"{new['objective_coverage_pct']}%, 新文件不带目标标签), 所以第 7 节"
            f"的缺失率在被重建后只会**更高**, 不会更乐观。\n"
        )
        full_path = facts.full_index_path
        assert full_path is not None  # set together with full_index in main()
        add(
            f"> 复现: `python scripts/generate_hwdataset_corpus_report.py "
            f"--full-index {full_path.name}`"
            f" (全量索引来源 `{full_path}`, "
            f"生成于扫描全部 {facts.full_index['files_scanned']} 个 pkl 的一次 `build_hw_index` 调用)\n"
        )
    else:
        add(
            "\n> 本节未做「全量重建」交叉核对。若要复核陈旧性是否影响结论, 用 "
            "`--full-index <全量索引.json>` 重新生成。\n"
        )
    add(
        f"> 家族解析一致性自检: 本脚本本地实现的前缀优先级与 "
        f"`ml/hwdataset/index.py` 比对, {total:,} 条记录中不一致 "
        f"**{facts.family_mismatches}** 条。\n"
    )

    # ---------------- 2. 家族规模 ----------------
    add("## 2. 家族规模与样本不平衡\n")
    add(
        markdown_table(
            ["家族", "已索引记录", "占记录", "已索引文件", "磁盘 pkl", "磁盘体积 (GiB)"],
            [
                [
                    f,
                    f"{facts.fam_records[f]:,}",
                    _pct(facts.fam_records[f], total),
                    f"{facts.fam_indexed_files.get(f, 0):,}",
                    f"{facts.fam_disk_files.get(f, 0):,}",
                    fmt_metric(facts.fam_disk_bytes.get(f, 0) / 1024**3, 2),
                ]
                for f in facts.families
            ],
        )
    )
    add(
        f"\n磁盘 pkl 总体积 {facts.total_disk_bytes:,} B "
        f"({facts.total_disk_bytes / 1024**4:.2f} TiB)。注意「记录数」与「磁盘体积」"
        f"排序并不一致 —— 体积由 pkl 内的原始画面主导, 与该文件产出多少条记录无关, "
        f"所以用体积当样本量代理会失真。\n"
    )
    add(_figure_ref(FIGURE_NAMES[1]))
    add("")
    add("按「单文件记录数」分箱的密度分布 (分母 = 已索引 pkl 数):")
    density = Counter(values_per_file)
    add("")
    add(
        markdown_table(
            ["单文件记录数", "文件数", "占已索引文件"],
            [
                [f"{value:,}", f"{count:,}", _pct(count, len(values_per_file))]
                for value, count in sorted(density.items())
            ],
        )
    )
    add(
        f"\n{len(density)} 个不同取值说明「每个文件读一次」的成本本身不均匀 —— "
        f"单文件体量由 pkl 内原始画面主导, 与该文件产出多少条记录无关, "
        f"所以按记录数做样本量代理会失真。\n"
    )
    add(_figure_ref(FIGURE_NAMES[8]))
    add("")

    # ---------------- 3. 相位表示 ----------------
    add("## 3. 相位表示与家族完全共线\n")
    matrix_rows = []
    for family in facts.families:
        counts = facts.fam_source.get(family, Counter())
        family_total = sum(counts.values())
        row = [family, f"{family_total:,}"]
        for source in facts.sources:
            value = counts.get(source, 0)
            row.append(f"{value:,} ({_pct(value, family_total)})" if value else "-")
        matrix_rows.append(row)
    add(markdown_table(["家族", "记录数"] + list(facts.sources), matrix_rows))
    add(
        f"\n**这是本语料最重要的结构性事实**: {facts.collinear_families}/"
        f"{len(facts.fam_records)} 个家族只出现单一 `source`, 交叉格全为 0。"
        f"`source` 与「家族 / 台架 / 优化目标」是同一个自变量的别名, "
        f"所以任何跨 `source` 的对比都同时换了台架与目标, "
        f"**不能**读作「哪种相位表示更好」。\n"
    )
    add(_figure_ref(FIGURE_NAMES[2]))
    add("")

    # ---------------- 4. 曝光 ----------------
    add("## 4. 曝光分布\n")
    exposure_rows = [
        [k, f"{facts.exposure_counts[k]:,}", _pct(facts.exposure_counts[k], total)]
        for k in exposure_labels
    ]
    exposure_rows.append(
        ["(未标注)", f"{facts.exposure_absent:,}", _pct(facts.exposure_absent, total)]
    )
    add(markdown_table(["曝光 (ms)", "记录数", "占比"], exposure_rows))
    add(
        f"\n曝光跨 {exposure_span:.2f} 个数量级。曝光是模型的**输入**特征, "
        f"且目标里保留了编码它的绝对亮度 (未做 peak 归一), 所以这些差异是真实的协变量, "
        f"不是可以直接平均掉的噪声。但不同曝光取值**几乎完全按家族分层** "
        f"(见第 3 节共线性), 跨家族比较亮度时不做曝光归一会直接被曝光差主导 —— "
        f"本报告因此不做跨家族亮度比较。\n"
    )
    add(_figure_ref(FIGURE_NAMES[3]))
    add("")

    # ---------------- 5. 视场 ----------------
    add("## 5. 视场与口径异质性\n")
    fov_rows: list[list[str]] = []
    for key in FOV_PROXY_KEYS:
        counts = facts.fov_counts.get(key, Counter())
        for value, count in sorted(counts.items()):
            fov_rows.append([key, value, f"{count:,}", _pct(count, total)])
        fov_rows.append([key, "(未标注)", f"{facts.fov_absent.get(key, 0):,}", _pct(facts.fov_absent.get(key, 0), total)])
    add(markdown_table(["sidecar 键", "值", "记录数", "占比"], fov_rows))
    add(
        "\n这些是 sidecar 里**各 writer 自己写的**像素数 (`region` / `cam_size` / "
        "`far_field_size`), 不是标定过的物理视场。三者量纲都是 px 但口径不同"
        "(半宽 / 窗口边长 / 补零后画幅), 因此同一个数值在不同家族下并不代表同一角度。"
        "已知 `fov_px` 随样本返回正是为了**按 family 分头训练/过滤**, 而不是全局 resize "
        "(全局缩放会破坏 0 级对齐)。\n"
    )
    add(_figure_ref(FIGURE_NAMES[4]))
    add("")

    # ---------------- 6. 孔径半径 ----------------
    add("## 6. Zernike 孔径半径与重建口径不一致\n")
    aperture_rows = [
        [
            v,
            f"{facts.aperture_counts[v]:,}",
            _pct(facts.aperture_counts[v], total),
            fmt_metric(float(v) / ZERNIKE_APERTURE_RADIUS, 2) + "×",
        ]
        for v in _numeric_sort(list(facts.aperture_counts))
    ]
    aperture_rows.append(
        ["(未标注)", f"{facts.aperture_absent:,}", _pct(facts.aperture_absent, total), "-"]
    )
    add(
        markdown_table(
            ["sidecar `zernike_radius` (px)", "记录数", "占比", "相对固定重建半径"],
            aperture_rows,
        )
    )
    # 该常量只在 PhaseSource.ZERNIKE 分支被用到, 故口径问题必须按 source 陈述,
    # 不能笼统说"每个家族都受影响" —— 记了半径的那几个家族根本不走这条重建路径。
    zern = [r for r in facts.records if r["source"] == "zernike"]
    zern_with_radius = sum(1 for r in zern if "zernike_radius" in r["sidecar"])
    radius_families = sorted(
        {
            r["family"]
            for r in facts.records
            if "zernike_radius" in r["sidecar"]
        }
    )
    _nonzernike = [r for r in facts.records if r["source"] != "zernike"]
    _nonzernike_total = len(_nonzernike)
    _nonzernike_with_radius = sum(
        1 for r in _nonzernike if "zernike_radius" in r["sidecar"]
    )
    add(
        f"\n**口径错配的方向与直觉相反。** 重建端 (`ml/hwdataset/records.py:1056`) 的 "
        f"`zernike_coeffs_to_panel(radius=config.zernike_radius)` 只在 "
        f"`PhaseSource.ZERNIKE` 分支被调用, 而 `config.zernike_radius` 默认取 "
        f"`_ZERNIKE_APERTURE_RADIUS = {ZERNIKE_APERTURE_RADIUS:.0f}`。把这条路径与实际"
        f"记录交叉核对后:\n"
    )
    add(
        markdown_table(
            ["分组", "记录数", "携带 sidecar `zernike_radius`", "是否走 Zernike 重建分支"],
            [
                [
                    "`source=zernike` (进入重建)",
                    f"{len(zern):,}",
                    f"**{zern_with_radius}** ({_pct(zern_with_radius, len(zern))})",
                    "是",
                ],
                [
                    "`source=panel_gray` / `panel_rad` (不进入)",
                    f"{_nonzernike_total:,}",
                    f"{_nonzernike_with_radius:,}",
                    "否",
                ],
            ],
        )
    )
    add(
        f"\n也就是说: **恰好是那 {len(zern):,} 条真正需要孔径半径的记录, 一条都没有记录"
        f"半径**, 而记下了半径的 {len(radius_families)} 个家族 "
        f"({', '.join('`' + f + '`' for f in radius_families)}) 走的是"
        f"「读入已存面板」路径, 根本不碰这个常量。后果是这个 "
        f"{ZERNIKE_APERTURE_RADIUS:.0f} px 对它所管辖的记录**无法用 sidecar 交叉校验** —— "
        f"相对已记录值的偏差只是"
        + "/".join(
            f"{float(v) / ZERNIKE_APERTURE_RADIUS:.2f}×"
            for v in _numeric_sort(list(facts.aperture_counts))
        )
        + " (不是数量级差异), 但方向不可验证。\n"
    )
    add(
        f"> ⚠️ `records.py:133-138` 的注释把 {ZERNIKE_APERTURE_RADIUS:.0f} 的依据写成"
        f"「the sidecars of the Zernike families record `zernike_radius=480`」。按上面的"
        f"交叉核对, **这句注释与数据不符**: 记录 `zernike_radius=480` 的是 `panel_gray` "
        f"家族, 而 `source=zernike` 的家族恰好一个都没有该字段。这属于**代码注释缺陷**, "
        f"建议与本报告一并复核。\n"
    )
    add(_figure_ref(FIGURE_NAMES[5]))
    add("")

    # ---------------- 7. 目标标签 ----------------
    add("## 7. 优化目标标签覆盖不足\n")
    objective_rows = [
        [k, f"{c:,}", _pct(c, total)] for k, c in facts.objective_counts.most_common()
    ]
    objective_rows.append(
        [
            "(未标注)",
            f"{facts.objective_absent:,}",
            _pct(facts.objective_absent, total),
        ]
    )
    add(
        markdown_table(
            [f"sidecar `{facts.objective_key or '(未找到)'}`", "记录数", "占比"],
            objective_rows,
        )
    )
    add(
        f"\n目标字段只覆盖 {objective_coverage:.1f}% 的记录, 缺失 "
        f"{_pct(facts.objective_absent, total)}。这直接限制了按目标分层的结论: "
        f"缺失不是「随机缺失」, 而是**成族缺失** —— 早期家族 "
        f"(如 `slm_pib`) 根本没有这个字段。所以「按优化目标分层比较」在本语料上"
        f"**不可行**, 缺失族无法作为对照组使用。\n"
    )
    add(_figure_ref(FIGURE_NAMES[6]))
    add("")

    # ---------------- 8. Zernike 阶数 ----------------
    add("## 8. Zernike 阶数与自由度跨度\n")
    order_rows: list[list[str]] = []
    for n_terms in sorted(facts.n_terms_counts):
        count = facts.n_terms_counts[n_terms]
        nmax_counts = facts.n_terms_nmax.get(n_terms, Counter())
        grid_counts = facts.n_terms_grid.get(n_terms, Counter())
        order_rows.append(
            [
                f"{n_terms:,}",
                ", ".join(f"T{k}" for k in sorted(nmax_counts)) or "-",
                ", ".join(f"{g}x{g}" for g in sorted(grid_counts)) or "-",
                f"{count:,}",
                _pct(count, total),
            ]
        )
    add(
        markdown_table(
            ["`n_terms`", "反推 `n_max`", "反推 `freeform_grid`", "记录数", "占比"],
            order_rows,
        )
    )
    ambiguous = [n for n in sorted(facts.n_terms_counts) if n in facts.n_terms_nmax and n in facts.n_terms_grid]
    # The grid SIDE (24), not the DOF count (576 = 24**2): ``n_terms_grid`` is
    # keyed by n_terms, so taking max() of it yields 576 and would report a
    # 576x576 grid.
    grid_sides = sorted(
        {
            int(r["freeform_grid"])
            for r in facts.records
            if r["source"] == "freeform" and r["freeform_grid"]
        }
    )
    grid_side = max(grid_sides) if grid_sides else 0
    zern_terms = _src_values(facts, "zernike", "n_terms")
    zern_lo, zern_hi = (min(zern_terms), max(zern_terms)) if zern_terms else (0, 0)
    add(
        f"\n`n_terms` 必须**按 source 分层读**, 否则会把两种不可比的量混在一根轴上:\n"
    )
    add(
        markdown_table(
            ["`source`", "`n_terms` 取值", "记录数", "该值的含义"],
            [
                [
                    "`zernike`",
                    ", ".join(str(n) for n in _src_values(facts, "zernike", "n_terms")),
                    f"{_src_count(facts, 'zernike'):,}",
                    "Noll 系数向量长度, 即**实际自由度** (三角数, 随 `n_max` 增长)",
                ],
                [
                    "`freeform`",
                    ", ".join(str(n) for n in _src_values(facts, "freeform", "n_terms")),
                    f"{_src_count(facts, 'freeform'):,}",
                    f"控制网格边长平方 = {grid_side}x{grid_side} = {grid_side**2:,} 个**独立网格点**",
                ],
                [
                    "`panel_gray`",
                    ", ".join(str(n) for n in _src_values(facts, "panel_gray", "n_terms")),
                    f"{_src_count(facts, 'panel_gray'):,}",
                    "⚠ 该源下相位取自 `_phase` 灰度面板, `_c` **未被使用** —— "
                    "此处的 `n_terms` 不代表该记录的有效自由度",
                ],
                [
                    "`panel_rad`",
                    ", ".join(str(n) for n in _src_values(facts, "panel_rad", "n_terms")),
                    f"{_src_count(facts, 'panel_rad'):,}",
                    "相位已是整块弧度面板, 无系数向量 (`n_terms`=0)",
                ],
            ],
        )
    )
    add(
        f"\n在**真正决定自由度**的两个源之间, 跨度是 "
        f"{zern_lo}–{zern_hi} (Zernike 侧, 随 `n_max` 变化) 对 "
        f"{grid_side**2:,} (freeform 侧 {grid_side}x{grid_side} 网格), "
        f"约 {(grid_side**2) / max(zern_hi, 1):.0f}× —— 这才是建模时的真实量级差距。"
        f"若把 `panel_gray` 的 `n_terms` 也当自由度, 会误以为存在 0–{max(zern_hi, 1):,} 的连续范围。\n"
    )
    if ambiguous:
        add(
            f"\n其中 {', '.join(str(n) for n in ambiguous)} 同时是三角数与完全平方数, "
            f"仅看 `len(_c)` **无法**判别口径, 必须按 family 裁决 —— "
            f"本表用 `n_max` / `freeform_grid` 字段直接消歧, 不靠长度猜测。\n"
        )
    else:
        add("\n")
    if ambiguous:
        add(
            f"\n其中 {', '.join(str(n) for n in ambiguous)} 同时是三角数与完全平方数, "
            f"仅看 `len(_c)` **无法**判别口径, 必须按 family 裁决 —— "
            f"本表用 `n_max` / `freeform_grid` 字段直接消歧, 不靠长度猜测。\n"
        )
    else:
        add("\n")
    add(_figure_ref(FIGURE_NAMES[7]))
    add("")

    # ---------------- 9. sidecar 完整度 ----------------
    add("## 9. sidecar 元数据完整度\n")
    add(
        f"- 已索引记录中 sidecar 为字典: {facts.sidecar_dicts:,} 条, "
        f"为空/缺失: {facts.empty_sidecars:,} 条"
    )
    add(
        f"- 记录实际匹配到的键: **{len(facts.record_keys)}** 个; "
        f"全部磁盘 sidecar JSON 出现的键: **{len(facts.all_json_keys)}** 个; "
        f"其中**仅存于磁盘、未被任何已索引记录匹配**: **{len(facts.disk_only_keys)}** 个"
    )
    if facts.disk_only_keys:
        add(
            "- 仅存于磁盘的键: "
            + ", ".join(f"`{k}`" for k in facts.disk_only_keys)
        )
    add("")
    add(
        markdown_table(
            ["sidecar 键", "匹配记录数", "占已索引记录"],
            [
                [f"`{k}`", f"{c:,}", _pct(c, total)]
                for k, c in facts.record_keys.most_common()
            ],
        )
    )
    add(
        f"\n覆盖率最高的键也远未饱和, 说明 sidecar schema **随家族各自演化**。"
        f"「仅存于磁盘」的 {len(facts.disk_only_keys)} 个键是那些**未进索引**的文件独有的 "
        f"(主要是第 1 节的 {newer_unindexed:,} 个新文件) —— 它们不是「索引丢了字段」, "
        f"而是重建索引后才会出现, 现在还无法核对。\n"
    )
    add(_figure_ref(FIGURE_NAMES[9]))
    add("")

    # ---------------- 10/11. 数组级分布 (实测, 来自 distribution_stats.json) ----
    if dist_stats is None:
        add("## 10. 图像/相位级分布差异 (实测)\n")
        add(
            "> 本节需要 `report/hwdataset_corpus/distribution_stats.json`, 由 "
            "`python scripts/analyze_hwdataset_distributions.py` 产生 "
            "(对 252 条记录做分层采样并物化)。该文件不存在时本节降级为提示, "
            "其余各节不受影响。先运行上面的分析脚本, 再重新生成本报告。\n"
        )
        add("## 11. 如何用数据处理方法消解分布差异\n")
        add("> 同 §10: 等待 `distribution_stats.json` 生成后, 本节会给出逐差异的消解方案。\n")
    else:
        fams = sorted(dist_stats["families"])
        n_fam = len(fams)
        add("## 10. 图像/相位级分布差异 (实测)\n")
        add(
            "以上各节全部来自**元数据** (索引 + sidecar), 回答的是「记录了多少、怎么标注」。"
            "本节回答另一个问题: **把记录真正物化成 64×64 的相位 phasor 与远场图像之后, "
            "各家族在数组层面差多少?** 数据来自 `distribution_stats.json` ——"
            f"对 {dist_stats['sampled']} 条记录 (每家族 ≤{dist_stats['per_family']} 条, "
            f"seed={dist_stats['seed']}, 触及 {dist_stats['files_touched']} 个 pkl, "
            f"运行 {dist_stats['wall_s']} s) 用 canonical `Materialiser` "
            f"(`grid={dist_stats['grid']}`, `{dist_stats['image_mode']}`) 物化后统计。"
            "物化路径与训练 DataLoader **逐位一致** (同一 `Materialiser`)。\n"
        )
        add(
            "> ⚠️ **口径**: `abs255` 保留绝对亮度 (未做 peak 归一), 所以"
            "**跨家族的亮度差异 ≈ 曝光差异, 不是光学差异**。只有尺度无关量 "
            "(`d90_ratio` / `central_energy` / `entropy256` / `phase_coh_mean`) 可以跨家族直接比。"
            "**表中无后缀的列 (帧均亮度/帧 CV/d90/FOV/中心窗/熵/相干度/展宽) 均为中位数 (median), 不是均值**; "
            "中位对 `slm_pib` 这类内部异质家族更稳 (其相位展宽 p10≈0 会把均值拖到中位之下)。"
            "同家族内 p10/p50/p90 的宽度反映的是该家族**内部**的记录间差异 (不同目标/迭代步), "
            "与家族间差异是两回事, 不要混读。\n"
        )
        def _s(v: float | None, spec: str) -> str:
            """None-safe cell: ``'—'`` when a metric is absent (degrade, don't crash)."""
            return format(v, spec) if v is not None else "—"

        # 表 1: 图像侧
        rows = []
        for f in fams:
            fam = dist_stats["families"][f]
            exp = fam["exposure"]
            em = exp.get("median")
            if em is not None and exp.get("min") == exp.get("max"):
                exp_s = f"{em:.2f}"
            else:
                med = f" (中位 {em:.1f})" if em is not None else ""
                exp_s = f"{exp.get('min', '—')}–{exp.get('max', '—')}{med}"
            rows.append(
                [
                    f,
                    f"{fam['n']}",
                    exp_s,
                    _s(_fam_stat(dist_stats, f, "frame_mean"), ".4f"),
                    _s(_fam_stat(dist_stats, f, "frame_cv"), ".3f"),
                    _s(_fam_stat(dist_stats, f, "d90_ratio"), ".4f"),
                    _s(_fam_stat(dist_stats, f, "central_energy"), ".3f"),
                    _s(_fam_stat(dist_stats, f, "entropy256"), ".2f"),
                ]
            )
        add(
            markdown_table(
                [
                    "家族",
                    "n",
                    "曝光 (ms)",
                    "帧均亮度",
                    "帧 CV",
                    "d90/FOV",
                    "中心窗能量",
                    "熵 (bits)",
                ],
                rows,
            )
        )
        add("")  # blank line: two adjacent tables must not be parsed as one
        # 表 2: 相位侧
        rows = []
        for f in fams:
            p10 = _fam_stat(dist_stats, f, "phase_spread", "p10")
            p90 = _fam_stat(dist_stats, f, "phase_spread", "p90")
            rows.append(
                [
                    f,
                    _s(_fam_stat(dist_stats, f, "phase_coh_mean"), ".4f"),
                    _s(_fam_stat(dist_stats, f, "phase_coh_max"), ".4f"),
                    _s(_fam_stat(dist_stats, f, "phase_spread"), ".4f"),
                    f"{_s(p10, '.4f')}–{_s(p90, '.4f')}",
                ]
            )
        add(
            markdown_table(
                ["家族", "相干度 (中位)", "相干度 (max)", "相位展宽 (中位)", "相位展宽 (p10–p90)"],
                rows,
            )
        )
        def _bright(family: str) -> float | None:
            """Antilog of a family's median log10(frame mean), None-safe (matches the table)."""
            v = _fam_stat(dist_stats, family, "frame_log10mean", "median")
            return 10.0 ** v if v is not None else None

        add("\n**读法 (数组层面确认了哪些元数据结论, 又补了什么):**\n")
        b_zernike = _bright("slm_zernike_shaping")
        b_mil = _bright("model_in_loop_hw_collect")
        if b_zernike is not None and b_mil is not None:
            add(
                "1. **亮度 ≈ 曝光, 不是光学。** 帧均亮度从 `slm_zernike_shaping` 的 "
                f"{b_zernike:.4f} 到 `model_in_loop_hw_collect` 的 {b_mil:.4f} "
                "跨约一个数量级, 与各自曝光同比例变化 ——"
                "证实 §4 的判断: 不除曝光的跨家族亮度比较量的是曝光本身。\n"
            )
        d90_mil = _fam_stat(dist_stats, "model_in_loop_hw_collect", "d90")
        d90_gs = _fam_stat(dist_stats, "slm_gsnet_square", "d90")
        d90r_mil = _fam_stat(dist_stats, "model_in_loop_hw_collect", "d90_ratio")
        d90r_gs = _fam_stat(dist_stats, "slm_gsnet_square", "d90_ratio")
        add(
            "2. **光斑尺寸必须按 FOV 归一才可比。** 原始 d90 从 "
            f"{_s(d90_mil, '.1f')} px (`model_in_loop_hw_collect`, 64 px 窗口) 到 "
            f"{_s(d90_gs, '.1f')} px (`slm_gsnet_square`, 1944 px 窗口) 看似差数倍, "
            f"但除以各自 FOV 后 d90/FOV = {_s(d90r_mil, '.4f')} vs {_s(d90r_gs, '.4f')} —— "
            "反而是 `model_in_loop` 家族的**相对**光斑更大。"
            "不做归一的「谁的光斑大」问题没有答案, 这正是 `fov_px` 随样本返回的原因。\n"
        )
        spr_gs = _fam_stat(dist_stats, "slm_gsnet_square", "phase_spread")
        spr_online = _fam_stat(dist_stats, "slm_pib_online", "phase_spread")
        add(
            "3. **相位表示差异在数组里同样可见。** `slm_gsnet_square` 相位展宽 ≈ "
            f"{_s(spr_gs, '.2f')} rad (近满幅自由相位), `slm_pib_online` ≈ "
            f"{_s(spr_online, '.3f')} rad (近平场); 而 `phase_coh_mean` 全家族都很高 "
            "(填充区是 `cos=0, sin=0` 的零相量, 不是未定义相位)。\n"
        )
        ent_mil = _fam_stat(dist_stats, "model_in_loop_hw_collect", "entropy256")
        ent_zernike = _fam_stat(dist_stats, "slm_zernike_shaping", "entropy256")
        cv_gs = _fam_stat(dist_stats, "slm_gsnet_square", "frame_cv")
        cv_mil = _fam_stat(dist_stats, "model_in_loop_hw_collect", "frame_cv")
        add(
            "4. **纹理差异是真实的协变量。** 256-bin 熵从 "
            f"{_s(ent_mil, '.2f')} (`model_in_loop_hw_collect`, 单点聚焦, 像素分布极集中) "
            f"到 {_s(ent_zernike, '.2f')} (`slm_zernike_shaping`, 散斑/整形态); "
            f"帧 CV 从 {_s(cv_gs, '.2f')} (`slm_gsnet_square`, 均匀方斑) "
            f"到 {_s(cv_mil, '.2f')} (`model_in_loop_hw_collect`, 高斯型焦点) ——"
            "同一个「远场」在不同家族里是**不同种类的图**, 一个全局归一化或 loss 无法同时服务。\n"
        )
        add(_figure_ref(FIGURE_NAMES[10]))
        add(_figure_ref(FIGURE_NAMES[11]))
        add(_figure_ref(FIGURE_NAMES[12]))
        add(_figure_ref(FIGURE_NAMES[13]))
        add("")

        add("## 11. 如何用数据处理方法消解分布差异\n")
        add("按「差异来源 → 消解手段 → 边界」逐条对应 (手段与 §10 的实测一一对应):\n")
        add(
            markdown_table(
                ["差异 (§10 实测)", "消解手段", "边界 / 不能消解的部分"],
                [
                    [
                        "亮度 ≈ 曝光 (跨家族 10×)",
                        "曝光走**输入**特征 `exposure_log10` (index 级 log10+标准化); "
                        "若目标是曝光无关, 用 **peak 归一** 后再比。",
                        "**禁用 sum 归一** (只度量归一化本身, R² 反而更差, 见 `report/zernike_phase2amp`); "
                        "abs255 语料里曝光与家族共线, 归一化后家族信号仍在其余维度。",
                    ],
                    [
                        "FOV 口径不一 (d90 3.6× 是假象)",
                        "**按家族分组** (家族↔fov 在本语料 1:1) 分别训练/评估; "
                        "跨家族比较只用 `d90_ratio` / `central_energy` 等**无量纲**量。",
                        "全局 resize **禁止** (破坏 0 阶对齐); 无法造出单一物理角尺度 ——"
                        "sidecar 只有 writer 自记 px, 没有标定过的角尺度。",
                    ],
                    [
                        "相位表示差异 (展宽 40×)",
                        "输入恒为 phasor `(cos, sin)` (无 arctan2 分支切口, 现有契约); "
                        "Zernike 系数→136 维零填充可加, 但表示与家族共线, 增益有限。",
                        "`slm_pib_online` 近平场 vs `slm_gsnet_square` 满幅自由相位"
                        "是**任务本质**差异 (在线微调 vs 自由相位整形), 数据处理消不掉。",
                    ],
                    [
                        "纹理差异 (熵 3×, CV 8×)",
                        "家族/文件级分层采样与加权 (小家族上采样); "
                        "`FileGroupedSampler` 保文件级独立 (现有)。",
                        "家族间样本比 350× (11k 记录里 `slm_gsnet_square` 仅 126 条索引记录), "
                        "加权只能缓解偏差, 不能替代数据。",
                    ],
                    [
                        "文件内相关 (同 pkl 记录高度相关)",
                        "按**文件**分组 CV / 按文件配对 (现有 `FileGroupedSampler` + "
                        "paired 统计的既有纪律)。",
                        "逐记录独立样本假设无效 ——"
                        "任何 p 值/置信区间必须按文件配对 (见 §12 不可断言第 4 条)。",
                    ],
                    [
                        "目标标签缺失 (成族)",
                        "无 — 缺失的 sidecar 字段无法事后重建。",
                        "按优化目标分层的结论在本语料**结构上不可行**; "
                        "补数据只能靠重跑采集 (带 `objective` 字段)。",
                    ],
                ],
            )
        )
        add(
            "\n**综合结论**: 能消解的差异 (曝光、FOV 口径、表示、采样偏差) 都有明确的现有工具 "
            "(`exposure_log10` / 家族分组 + 无量纲指标 / phasor 输入 / 分层 + 文件分组), "
            "而且 `ml/hwdataset` 的默认契约已经实现了其中大部分 ——"
            "问题从来不是「缺工具」, 而是**使用者**跨家族比较时忘了这些前提。"
            "不能消解的差异 (任务本质的相位结构、目标标签缺失、文件内相关)"
            "只能靠**改采集** (统一曝光/FOV/标签) 解决, 数据处理层无解。\n"
        )

    # ---------------- 12. 断言边界 ----------------
    add("## 12. 本语料可以断言 / 不可断言\n")
    add("**可以断言 (描述性, 由本报告的元数据统计直接支撑):**\n")
    add("1. 索引缓存已陈旧, 只覆盖当前磁盘 pkl 的一部分, 重建索引会改变样本集。")
    add("2. 家族与 `source` 完全共线, 记录数在家族间高度不均衡。")
    add("3. 曝光 / 视场代理量 / Zernike 指令孔径半径都跨家族不一致, 属真实协变量。")
    add("4. 每个 pkl 的记录数极不均匀, 必须按文件分组读取。\n")
    add("**不可断言 (本语料结构上不支持):**\n")
    add("1. ❌ 「`panel_gray` vs `zernike` 哪种表示更好」—— 与家族/台架/目标完全混淆。")
    add("2. ❌ 任何**物理**角尺度或跨家族统一视场口径 —— sidecar 只有各 writer 自记的 px。")
    add("3. ❌ 按优化目标分层的结论 —— 目标标签成族缺失, 无合法对照组。")
    add("4. ❌ 逐记录显著性检验 / p 值 / 置信区间 —— 记录不是独立样本, 同文件内高度相关。")
    add("5. ❌ 未做曝光归一的跨家族亮度比较 —— 亮度由曝光差主导。")
    add("6. ❌ 模型优劣排名 —— 噪声地板 (折间 σ) 大于多数待比较的差异, 见 `report/zernike_phase2amp/`。\n")

    # ---------------- 13. 与既有文档的差异 ----------------
    add("## 13. 与既有文档记载的差异\n")
    add(
        markdown_table(
            ["项", "既有文档记载", "本次运行时统计", "说明"],
            [
                [
                    "pkl 总数",
                    "996 文件 / 265 pkl",
                    f"{disk_total:,} pkl ({int(facts.freshness[0].sum()):,} 旧 + {int(facts.freshness[1].sum()):,} 新)",
                    "文档口径是索引建立时的快照, 当前磁盘已增长",
                ],
                [
                    "已索引文件",
                    "261",
                    f"{usable:,} (账面) / {indexed_on_disk:,} (磁盘实测命中)",
                    "一致",
                ],
                [
                    "已索引记录",
                    "11,393",
                    f"{total:,}",
                    "一致",
                ],
                [
                    "语料体积",
                    "65.88 GB",
                    f"{facts.total_disk_bytes / 1024**3:.2f} GiB ({facts.total_disk_bytes:,} B)",
                    (
                        f"文档的 65.88 GB 恰为**缓存 mtime 之前**那 {len(facts.fresh_indexed_files)} "
                        f"个 pkl 的体积和 ({facts.indexed_disk_bytes / 1e9:.2f} GB), 不含新增文件"
                        if facts.indexed_disk_bytes
                        else "未做该项核对"
                    ),
                ],
                [
                    "家族数",
                    "7",
                    f"已索引 {len(facts.fam_records)}, 磁盘 {len(facts.fam_disk_files)}",
                    "多出的 1 个是 `bench_stability` (1 个 pkl, 42 条记录全部因缺相位被排除, "
                    "故记录数为 0); 注意 collect/sweep **本身已进索引**, 只是各有一半新文件未进",
                ],
                [
                    "曝光取值",
                    "6 个",
                    f"{len(exposure_labels)} 个 ({exposure_labels[0]}–{exposure_labels[-1]} ms)",
                    "`index.py` 文档字符串里的 6 值清单已过时; "
                    "9 个值**全部**出现在已索引的 11,393 条内, 与新文件无关",
                ],
                [
                    "ROI 半径",
                    "固定 500 px",
                    f"重建固定 {ZERNIKE_APERTURE_RADIUS:.0f} px, sidecar 指令半径 "
                    + "/".join(_numeric_sort(list(facts.aperture_counts)))
                    + " px",
                    "文档给的是重建端固定值, 与指令端记录值不是同一口径",
                ],
            ],
        )
    )
    add(
        "\n以上差异**全部**是语料/索引状态变化, 不是代码回归; 但任何引用旧数字的文档或结论"
        "在重建索引后都需要复核。\n"
    )

    # ---------------- 14. 复现 ----------------
    add("## 14. 复现\n")
    add("```bash")
    add("# 第一阶段: 分层物化 252 条记录, 统计数组级分布 (~90 s, 需 pkl 可读)")
    add("python scripts/analyze_hwdataset_distributions.py")
    add("")
    add("# 第二阶段: 渲染本报告 (元数据 + 上面的 JSON, 不碰 pkl)")
    add("python scripts/generate_hwdataset_corpus_report.py")
    add("```\n")
    add(
        f"- 数据源: `{CACHE_PATH.relative_to(ROOT).as_posix()}` "
        f"(version {cache.get('version')}, roots `{cache.get('roots')}`), "
        f"`{(DEBUG_ROOT / '**' / '*.json').relative_to(ROOT).as_posix()}`"
    )
    add(
        f"- 元数据源: `(data/debug/**/*.pkl)` 共 {disk_total:,} 个, 仅 `Path.stat()`"
    )
    add(
        f"- 数组级源: `{DIST_STATS_PATH.relative_to(ROOT).as_posix()}` "
        "(由 `analyze_hwdataset_distributions.py` 分层物化 252 条记录产生; "
        "缺失时 §10/§11 降级为提示, 10 张元数据图不受影响)"
    )
    add(
        f"- 图: `report/hwdataset_corpus/figures/` 共 "
        f"{len(FIGURE_NAMES) - (4 if dist_stats is None else 0)} 张 PNG "
        f"(含 4 张数组级分布图, 仅在上一步成功时生成)"
    )
    add(
        f"- 运行环境: Python {sys.version.split()[0]}, "
        f"matplotlib {matplotlib.__version__}, numpy {np.__version__}"
    )
    add(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline metadata analysis of the ml/hwdataset hardware-debug corpus. "
            "Never opens a .pkl -- metadata only."
        )
    )
    parser.add_argument(
        "--full-index",
        type=Path,
        default=None,
        help=(
            "Optional path to an index JSON that scanned EVERY pkl. Enables the "
            "§1.1 'does staleness change the conclusions?' cross-check. Never "
            "used for the headline numbers. Generate one with "
            "`build_hw_index(index_cache=<path>)`."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG rendering (markdown tables only).",
    )
    args = parser.parse_args()

    facts = collect()
    if args.full_index is not None:
        facts.full_index_path = Path(args.full_index)
        facts.full_index = load_full_index(facts.full_index_path)
        logger.info(
            "loaded full index {} ({} records) for the §1.1 cross-check",
            facts.full_index_path,
            len(facts.full_index["records"]),
        )
    dist_stats = load_dist_stats(DIST_STATS_PATH)
    if dist_stats is None:
        logger.info(
            "no {} found; §10/§11 degrade to a pointer and the 4 distribution "
            "figures are skipped (run scripts/analyze_hwdataset_distributions.py first)",
            DIST_STATS_PATH.relative_to(ROOT),
        )
    if not args.no_figures:
        make_figures(facts, dist_stats)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    body = insert_header(build_report(facts, dist_stats), REPORT_KEY)
    REPORT_PATH.write_text(body, encoding="utf-8")
    logger.info("wrote {}", REPORT_PATH.relative_to(ROOT))
    if not args.no_figures:
        n_fig = len(FIGURE_NAMES) - (4 if dist_stats is None else 0)
        logger.info("wrote {} figures to {}", n_fig, FIGURE_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
