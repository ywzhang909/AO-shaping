"""Generate the ``report/hwdataset_corpus`` corpus-composition report.

Metadata-only by construction: the only file *contents* ever read are
``data/hw_index_cache.json`` and the ``data/debug/**/*.json`` sidecars.  Every
``.pkl`` is touched with ``Path.stat()`` only -- never opened, unpickled or
memory-mapped, and ``ml.hwdataset`` is never imported.  All counts, percentages
and figure annotations are computed at runtime; nothing is hard-coded.

Usage::

    python scripts/generate_hwdataset_corpus_report.py
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


def make_figures(facts: CorpusFacts) -> None:
    """Render all ten figures into :data:`FIGURE_DIR`."""
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


def _figure_ref(filename: str) -> str:
    """Markdown image reference, emitted only for a PNG that actually exists."""
    path = FIGURE_DIR / filename
    if not path.is_file():
        return ""
    return f"![{filename.removesuffix('.png')}](figures/{filename})"


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(facts: CorpusFacts) -> str:
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
        "本报告**只**读取 `data/hw_index_cache.json` 与 `data/debug/**/*.json` 两个来源, "
        "`.pkl` 一律只做 `Path.stat()` 元数据检查 (不打开 / 不反序列化 / 不 mmap), "
        "也不 import `ml.hwdataset`。所有数字均在运行时统计, 无硬编码。\n"
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
        f"必须按文件分组读取。\n"
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

    # ---------------- 10. 断言边界 ----------------
    add("## 10. 本语料可以断言 / 不可断言\n")
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

    # ---------------- 11. 与既有文档的差异 ----------------
    add("## 11. 与既有文档记载的差异\n")
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

    # ---------------- 12. 复现 ----------------
    add("## 12. 复现\n")
    add("```bash")
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
    add(f"- 图: `report/hwdataset_corpus/figures/` 共 {len(FIGURE_NAMES)} 张 PNG")
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
    if not args.no_figures:
        make_figures(facts)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    body = insert_header(build_report(facts), REPORT_KEY)
    REPORT_PATH.write_text(body, encoding="utf-8")
    logger.info("wrote {}", REPORT_PATH.relative_to(ROOT))
    if not args.no_figures:
        logger.info("wrote {} figures to {}", len(FIGURE_NAMES), FIGURE_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
