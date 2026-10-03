"""Corpus statistics and smoke run for the hardware dataset.

Run as ``python -m ml.hwdataset.inspect``. It builds (or loads) the corpus index,
reports what is and is not in the dataset and why, optionally materialises a few
samples to prove the numeric contract holds, and can build the derived-grid
cache.

The output is a compact diagnostic table (or JSON with ``--json``). It
deliberately does **not** write markdown or figures: this repository requires
report generation to live in ``scripts/``.

What the report must make obvious, because both are easy to get wrong:

* which files and records are **excluded** and under which
  :class:`~ml.hwdataset.index.ExclusionReason`;
* that the families have **different fields of view**, so one output grid means a
  different physical angular scale per family -- surfaced per item as ``fov_px``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from loguru import logger

from ml.hwdataset.index import (
    DEFAULT_ROOTS,
    ExclusionReason,
    HwCorpusIndex,
    HwRecordRef,
    PhaseSource,
    build_hw_index,
)
from ml.hwdataset.records import (
    HwRecordError,
    Materialiser,
    MaterialiserConfig,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_EMPTY = 2


def _fov_px(ref: HwRecordRef) -> int | None:
    """Sidecar-derived field-of-view hint for one record, in camera pixels.

    Mirrors the Dataset's fallback chain (``region`` -> ``cam_size``) so the
    report agrees with the sample metadata without paying to materialise every
    record: the true value always comes from the ``_img`` shape, which needs the
    payload.

    Args:
        ref: One indexed record.

    Returns:
        The hint in pixels, or ``None`` when the sidecar carries none.
    """
    sidecar: Mapping[str, Any] = ref.sidecar
    region = sidecar.get("region")
    if isinstance(region, (int, float)) and not isinstance(region, bool) and region > 0:
        return int(2 * float(region))
    cam_size = sidecar.get("cam_size")
    if (
        isinstance(cam_size, (int, float))
        and not isinstance(cam_size, bool)
        and cam_size > 0
    ):
        return int(cam_size)
    return None


def _exposure_summary(records: Sequence[HwRecordRef]) -> dict[str, Any]:
    """Summarise the resolved exposure times, in milliseconds.

    Args:
        records: Records to summarise.

    Returns:
        Mapping with ``count``, ``min``, ``median``, ``max``, ``distinct`` and
        the sorted ``values``.
    """
    values = sorted(
        float(r.exposure_ms) for r in records if r.exposure_ms is not None
    )
    return {
        "count": len(values),
        "min": values[0] if values else None,
        "median": float(median(values)) if values else None,
        "max": values[-1] if values else None,
        "distinct": len(set(values)),
        "values": values,
    }


def _family_shapes(records: Sequence[HwRecordRef]) -> dict[str, list[list[int | None]]]:
    """Per family, the distinct ``(n_terms, n_max, freeform_grid)`` triples.

    Args:
        records: Records to summarise.

    Returns:
        Mapping from family name to a sorted list of triples.
    """
    grouped: dict[str, set[tuple[int | None, int | None, int | None]]] = {}
    for ref in records:
        grouped.setdefault(ref.family, set()).add(
            (ref.n_terms, ref.n_max, ref.freeform_grid)
        )
    return {
        family: [list(triple) for triple in sorted(triples)]
        for family, triples in sorted(grouped.items())
    }


def _sample_reports(
    index: HwCorpusIndex, *, grid: int, count: int
) -> list[dict[str, Any]]:
    """Materialise ``count`` evenly strided samples and describe each one.

    Args:
        index: The (already filtered) index to sample.
        grid: Output grid side length.
        count: How many samples to materialise.

    Returns:
        One report mapping per sample that materialised successfully.
    """
    records = index.records
    if not records or count < 1:
        return []
    config = MaterialiserConfig(grid=grid)
    materialiser = Materialiser(config=config, cache_size=1)
    stride = max(1, len(records) // count)
    reports: list[dict[str, Any]] = []
    for sample_number in range(count):
        ref = records[min(sample_number * stride, len(records) - 1)]
        started = time.perf_counter()
        try:
            sample = materialiser.materialise(ref)
        except HwRecordError as exc:
            logger.warning("Sample {} failed for {}: {}", sample_number, ref.path, exc)
            continue
        elapsed_ms = (time.perf_counter() - started) * 1e3
        image = np.asarray(sample.image)
        reports.append(
            {
                "sample": sample_number,
                "record_position": ref.position,
                "family": ref.family,
                "source": ref.source.value,
                "exposure_ms": sample.exposure_ms,
                "fov_px": _fov_px(ref),
                "phase_cos_shape": list(sample.phase_cos.shape),
                "phase_cos_dtype": str(sample.phase_cos.dtype),
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "image_min": float(image.min()),
                "image_max": float(image.max()),
                "contrast_max": float(sample.contrast.max()),
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )
    return reports


def _build_report(
    index: HwCorpusIndex, *, samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Assemble the machine-readable corpus report.

    Args:
        index: The index actually in use (post-filter).
        samples: Per-sample reports, possibly empty.

    Returns:
        A JSON-serialisable mapping.
    """
    exposures = _exposure_summary(index.records)
    fovs = sorted({v for v in (_fov_px(r) for r in index.records) if v is not None})
    by_source = index.counts_by_source()
    return {
        "files_scanned": index.files_scanned,
        "files_usable": index.files_usable,
        "total_records": len(index),
        "by_source": {
            source.value: int(by_source.get(source, 0)) for source in PhaseSource
        },
        "by_family": {k: int(v) for k, v in index.counts_by_family().items()},
        "family_shapes": _family_shapes(index.records),
        "excluded": {reason.value: int(n) for reason, n in index.excluded.items()},
        "exposure_ms": exposures,
        "fov_px": {
            "distinct_count": len(fovs),
            "values": fovs,
            # More than one means the families are not physically comparable.
            "differs_by_family": len(fovs) > 1,
        },
        "samples": list(samples),
    }


def _print_report(report: Mapping[str, Any]) -> None:
    """Print the human-readable corpus report.

    Args:
        report: Mapping produced by :func:`_build_report`.
    """
    print(
        f"files scanned / usable : {report['files_scanned']} / {report['files_usable']}"
    )
    print(f"records                : {report['total_records']}")

    print("\nby phase source")
    total = max(int(report["total_records"]), 1)
    for name, count in report["by_source"].items():
        share = 100.0 * int(count) / total
        print(f"  {name:<11s} {int(count):>7d}  {share:5.1f}%")

    print("\nby family  (n_terms, n_max, freeform_grid)")
    shapes = report["family_shapes"]
    for family, count in report["by_family"].items():
        rendered = " ".join(
            f"({a},{b},{c})" for a, b, c in shapes.get(family, [])
        )
        print(f"  {family:<26s} {int(count):>6d}  {rendered}")

    print("\nexcluded (not in the dataset)")
    excluded = report["excluded"]
    if not excluded:
        print("  (nothing excluded)")
    else:
        for reason, count in excluded.items():
            print(f"  {reason:<26s} {int(count):>6d}")
        print("  reason codes: " + ", ".join(r.value for r in ExclusionReason))

    exposure = report["exposure_ms"]
    print("\nexposure (ms)")
    if exposure["count"] == 0:
        print("  no record carries a resolvable exposure")
    else:
        print(
            f"  count={exposure['count']} distinct={exposure['distinct']} "
            f"min={exposure['min']} median={exposure['median']} max={exposure['max']}"
        )

    fov = report["fov_px"]
    print(f"\nfield of view (sidecar hint, px): {fov['values']}")
    if fov["differs_by_family"]:
        print(
            f"  WARNING: {fov['distinct_count']} distinct fields of view in one corpus. "
            "A single output grid therefore means a different physical angular "
            "scale per family. This is surfaced per item as `fov_px`; filter or "
            "train per family rather than resampling."
        )

    if report["samples"]:
        print("\nmaterialised samples")
        for sample in report["samples"]:
            print(
                f"  #{sample['sample']} {sample['family']}/{sample['source']} "
                f"exposure={sample['exposure_ms']}ms fov={sample['fov_px']}px "
                f"{sample['elapsed_ms']:.1f}ms"
            )
            print(
                f"      phase_cos {tuple(sample['phase_cos_shape'])} "
                f"{sample['phase_cos_dtype']}"
            )
            print(
                f"      image     {tuple(sample['image_shape'])} "
                f"{sample['image_dtype']} "
                f"range [{sample['image_min']:.4f}, {sample['image_max']:.4f}] "
                f"contrast_max {sample['contrast_max']:.4f}"
            )


def _maybe_build_cache(args: argparse.Namespace) -> int | None:
    """Build the derived-grid cache when ``--build-cache`` was requested.

    The cache module is imported lazily and optionally: it is an optimisation, so
    a missing module is a clear notice and a non-zero exit, never an
    ``ImportError`` escaping ``main``.

    Args:
        args: Parsed arguments.

    Returns:
        ``None`` when the flag was absent, otherwise the process exit code.
    """
    if not args.build_cache:
        return None
    try:
        from ml.hwdataset.cache import prepare_hw_cache
    except ImportError as exc:
        print(f"ml.hwdataset.cache is unavailable: {exc}", file=sys.stderr)
        return EXIT_ERROR

    index = _build_index(args)
    if index is None:
        return EXIT_ERROR
    directories = prepare_hw_cache(index, config=MaterialiserConfig(grid=args.grid))
    print(f"built {len(directories)} cache directories")
    return EXIT_OK


def _build_index(args: argparse.Namespace) -> HwCorpusIndex | None:
    """Build and filter the index according to ``args``.

    Args:
        args: Parsed arguments.

    Returns:
        The filtered index, or ``None`` when the corpus is empty.
    """
    families: Iterable[str] | None = None
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
    try:
        index = build_hw_index(
            roots=args.roots,
            limit_files=args.limit_files,
            index_cache=args.index_cache,
            progress_every=args.progress_every,
        )
    except (OSError, ValueError) as exc:
        print(f"could not index the corpus: {exc}", file=sys.stderr)
        return None
    index = index.filter(families=families, require_exposure=args.require_exposure)
    if len(index) == 0:
        print(
            "no usable records in the corpus: every record was excluded "
            "(see the exclusion reasons in ml.hwdataset.index.build_hw_index)",
            file=sys.stderr,
        )
        return None
    return index


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ml.hwdataset.inspect",
        description="Report the hardware debug corpus and smoke-test sampling.",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Roots to scan recursively for *.pkl (default: data/debug).",
    )
    parser.add_argument(
        "--limit-files", type=int, default=None, help="Cap the number of pickles scanned."
    )
    parser.add_argument(
        "--index-cache", default=None, help="JSON index cache; skips the scan when present."
    )
    parser.add_argument(
        "--progress-every", type=int, default=10, help="Log progress every N files (0 = off)."
    )
    parser.add_argument(
        "--grid", type=int, default=64, help="Output grid side length for sampling."
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Materialise N strided samples (0 = skip)."
    )
    parser.add_argument("--families", default=None, help="Comma-separated family allow-list.")
    parser.add_argument(
        "--require-exposure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop records whose exposure could not be resolved.",
    )
    parser.add_argument(
        "--build-cache", action="store_true", help="Build the derived-grid cache and exit."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m ml.hwdataset.inspect``.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a runtime failure, ``2`` when the corpus holds
        no usable records.
    """
    args = _parse_args(argv)

    cache_status = _maybe_build_cache(args)
    if cache_status is not None:
        return cache_status

    index = _build_index(args)
    if index is None:
        return EXIT_EMPTY

    report = _build_report(
        index, samples=_sample_reports(index, grid=args.grid, count=args.sample)
    )
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_report(report)
    return EXIT_OK


if __name__ == "__main__":
    # The guard matters: DataLoader workers on Windows use `spawn`, which
    # re-imports this module in the child process.
    raise SystemExit(main())
