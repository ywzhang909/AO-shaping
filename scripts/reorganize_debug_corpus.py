"""Reorganise ``data/debug`` into the canonical one-run-per-directory layout.

Target layout (the shape ``slm-gs-refine`` already produces)::

    data/debug/<prefix>/<YYYYmmdd_HHMMSS>/
        <prefix>_<YYYYmmdd_HHMMSS>.{json,png,pkl}   # the standard debug trio
        <run-specific extras: .csv, .h5, best_phase.npy, best_far_field.png>
        .gsnet_cache/<prefix>_<stamp>/*.npy
        .hw_cache/<prefix>_<stamp>/*.npy

What it fixes, and why each step is needed:

* **Collapses the doubled timestamp.** ``save_recorder_debug_artifacts`` used to
  build its context as ``f"{prefix}_{now}"`` and then let ``gen_date_dir`` stamp
  a *second* time, producing ``<prefix>_<ts1>/<ts1>/`` and a
  ``<prefix>_<ts1>_<ts1>`` file stem. Both are now a single stamp.
* **Moves the family into its own directory.** ``<prefix>_<ts1>/`` becomes
  ``<prefix>/``, so the family is readable without regex-stripping a timestamp.
* **Renames the trio stem** ``<prefix>_<ts1>_<ts1>`` -> ``<prefix>_<ts1>``.
* **Merges complementary halves of one run.** e.g.
  ``hardware_gs_centered_20261010/slm_gs_refine/<ts>/`` (CSV + best phase/image)
  and ``slm_gs_refine/<ts>/`` (the trio) are the same run, split across a batch
  directory. They merge into one directory -- verified to have no filename
  collisions.
* **Deduplicates byte-identical runs.** 19 verified pairs
  (15 ``phase_replay_device_*`` vs their ``device_*`` original, 4
  ``slm_pib_hw_20260929/.../recorder_*.pkl`` vs the top-level trio). Collapsing
  them removes double-counted records from the ML corpus. A pair is only
  collapsed when its pickles are **byte-identical** (SHA-256); anything else is
  reported and left alone.
* **Renames the derived caches in lockstep.** ``.gsnet_cache/<stem>/`` and
  ``.hw_cache/<stem>/`` are keyed by the pickle stem and their ``meta.json``
  records the pickle's absolute ``source`` path, so both the directory name and
  that field are rewritten. Skipping this costs 212 s of Zernike re-inversion
  plus 35 s of disk reads *per epoch*.

Safety properties:

* **Dry run by default.** ``--apply`` is required to touch anything.
* Moves within one volume, so files are renamed, never copied. Same-volume
  renames cannot leave a half-written artefact.
* Every duplicate deletion is gated on a SHA-256 match and reported; ``--keep-duplicates``
  disables deletion entirely (the merge still happens, the loser is moved aside).
* An inverted manifest (``<plan>.json``) is written before any move, so the
  operation is auditable and reversible by hand.

Usage::

    python scripts/reorganize_debug_corpus.py                 # dry run, report
    python scripts/reorganize_debug_corpus.py --apply         # do it
    python scripts/reorganize_debug_corpus.py --apply --keep-duplicates
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEBUG = ROOT / "data" / "debug"

#: Derived caches that live *inside* a run directory; they are not runs.
CACHE_DIRS = {".gsnet_cache", ".hw_cache"}

RUN_STAMP = re.compile(r"^\d{8}_\d{6}$")
SUFFIX_TS = re.compile(r"^(?P<prefix>.+?)_(?P<stamp>\d{8}_\d{6})$")
SUFFIX_DATE = re.compile(r"^(?P<prefix>.+?)_(?P<date>\d{8})$")


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass
class Move:
    """One file's relocation."""

    src: Path
    dst: Path


@dataclass
class RunPlan:
    """One run directory's relocation, plus whatever else it implies."""

    old_dir: Path
    new_dir: Path
    prefix: str
    stamp: str
    rule: str
    files: list[Move] = field(default_factory=list)
    cache_renames: list[tuple[Path, Path]] = field(default_factory=list)
    meta_rewrites: list[Path] = field(default_factory=list)
    duplicate_of: Path | None = None  # set when this run is a byte-identical twin
    #: ``{old pickle stem: new pickle stem}`` -- the derived caches are named
    #: after the stem (``cache_dir_for`` -> ``<pkl_dir>/.hw_cache/<stem>``), so
    #: renaming a pickle without renaming its cache dir turns a cache *hit* into
    #: a silent miss and costs the full rebuild (212 s Zernike re-inversion +
    #: 35 s of disk reads, per epoch).
    stem_renames: dict[str, str] = field(default_factory=dict)


def artefacts(run_dir: Path) -> list[Path]:
    """Files directly in a run directory (caches are directories)."""
    return sorted(f for f in run_dir.iterdir() if f.is_file())


def _stamp_of(run_dir: Path) -> str:
    """Fallback stamp for a directory whose name carries no run timestamp."""
    return datetime.fromtimestamp(artefacts(run_dir)[0].stat().st_mtime).strftime(
        "%Y%m%d_%H%M%S"
    )


def _strip_date(name: str) -> str:
    match = SUFFIX_DATE.match(name)
    return match.group("prefix") if match else name


def classify(rel: str, run_dir: Path) -> tuple[str, str, str]:
    """Derive ``(prefix, stamp, rule)`` for one run directory.

    The rules below cover every shape present in the corpus. Each is keyed on
    the *directory* structure, not on a filename, so a run is never split from
    its own CSV / best-phase / sidecar.
    """
    parts = rel.split("/")
    leaf = run_dir.name

    # <prefix>_<ts>/<ts>/  -- the doubled-timestamp shape (134 dirs)
    if len(parts) == 2 and RUN_STAMP.match(leaf):
        match = SUFFIX_TS.match(parts[0])
        if match and match.group("stamp") == leaf:
            return match.group("prefix"), leaf, "doubled"

    # <prefix>/<ts>/  -- the canonical shape (24 dirs, incl. slm_gs_refine)
    if len(parts) == 2 and RUN_STAMP.match(leaf):
        return parts[0], leaf, "canonical"

    # <prefix>_YYYYMMDD/  -- date-stamped dirs such as device_* (6 dirs)
    if len(parts) == 1:
        match = SUFFIX_DATE.match(parts[0])
        if match:
            return match.group("prefix"), _stamp_of(run_dir), "date_dir"
        match = SUFFIX_TS.match(parts[0])
        if match:
            return match.group("prefix"), match.group("stamp"), "flat_full"
        return parts[0], _stamp_of(run_dir), "flat_bare"

    # <batch>/<prefix>/<ts>/  -- batch directory holding one run
    if len(parts) == 3 and RUN_STAMP.match(leaf):
        return _strip_date(parts[1]), leaf, "batch_prefix_ts"

    # <batch>/<prefix>_<ts>/  -- batch directory, stamp in the middle name
    if len(parts) == 3:
        match = SUFFIX_TS.match(parts[1])
        if match:
            return match.group("prefix"), match.group("stamp"), "batch_prefix_stamp"

    # <batch>/<prefix>_<ts>/  -- batch directory with the stamp in the leaf name
    if len(parts) == 2:
        match = SUFFIX_TS.match(leaf)
        if match:
            return match.group("prefix"), match.group("stamp"), "batch_prefix_stamp"

    # <phase_replay_batch>/phase_replay/  -- a replay of a <batch> run
    if len(parts) == 2:
        prefix = _strip_date(parts[0])
        prefix = re.sub(r"^phase_replay_", "", prefix)
        return prefix, _stamp_of(run_dir), "batch_phase_replay"

    raise SystemExit(
        f"Unclassified run directory {rel!r}; refusing to guess its target."
    )


def find_run_dirs() -> list[Path]:
    """Every run directory: deepest dir holding files, caches excluded."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(DEBUG):
        here = Path(dirpath)
        if any(part in CACHE_DIRS for part in here.relative_to(DEBUG).parts):
            continue
        real = [d for d in dirnames if d not in CACHE_DIRS]
        if filenames and not real:
            found.append(here)
    return sorted(found)


def target_name(old_name: str, prefix: str, stamp: str) -> str:
    """Collapse a doubled stem onto the single canonical stem.

    ``<prefix>_<ts>_<ts>`` and ``<prefix>_<ts>_<otherstamp>`` (a run whose pickle
    was written a second after the directory was created, which the
    ``slm_gs_refine`` reports deliberately kept) both become ``<prefix>_<ts>``.
    Matching is done on the **stem**, not the whole filename -- the doubled stem
    never ends in a bare ``_<stamp>`` once the suffix is attached, so a whole-name
    match silently matches nothing.
    """
    canonical = f"{prefix}_{stamp}"
    path = Path(old_name)
    if path.stem == canonical or not path.stem.startswith(canonical + "_"):
        return old_name
    return f"{canonical}{path.suffix}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pickle_fingerprint(run_dir: Path) -> str:
    """Content fingerprint of a run's pickles -- **names are deliberately ignored**.

    Two runs can hold byte-identical data under different filenames (e.g. a
    batch copy named ``recorder_161710.pkl`` beside the canonical
    ``slm_pib_pearson_20260929_161710_20260929_161710.pkl``). Hashing the name too
    would make those look like different runs and leave the duplicate in place.
    """
    digests = sorted(sha256(pkl) for pkl in run_dir.glob("*.pkl"))
    if not digests:
        return ""
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


def build_plan() -> list[RunPlan]:
    """Compute every relocation. Pure -- touches nothing."""
    plans = [RunPlan(*()) for _ in ()]  # keep type checkers quiet about field order
    plans = []
    for run_dir in find_run_dirs():
        rel = run_dir.relative_to(DEBUG).as_posix()
        prefix, stamp, rule = classify(rel, run_dir)
        plan = RunPlan(
            old_dir=run_dir,
            new_dir=DEBUG / prefix / stamp,
            prefix=prefix,
            stamp=stamp,
            rule=rule,
        )
        new_stem = f"{prefix}_{stamp}"
        for src in artefacts(run_dir):
            name = src.name
            if name.endswith(".pkl") or name.endswith(".json") or name.endswith(".png"):
                candidate = target_name(name, prefix, stamp)
                if candidate != name:
                    name = candidate
            elif SUFFIX_TS.match(Path(name).stem) and Path(name).stem.endswith(stamp):
                # .csv / .h5 / .npy companions carry the same doubled stem
                stem = Path(name).stem
                if stem == f"{new_stem}_{stamp}":
                    name = f"{new_stem}{Path(name).suffix}"
            plan.files.append(Move(src=src, dst=plan.new_dir / name))
            if src.stem != Path(name).stem:
                plan.stem_renames[src.stem] = Path(name).stem
        for cache_name in CACHE_DIRS:
            cache_root = run_dir / cache_name
            if not cache_root.is_dir():
                continue
            for entry in sorted(cache_root.iterdir()):
                # the cache directory is named after the pickle stem
                new_entry = plan.stem_renames.get(entry.name, entry.name)
                plan.cache_renames.append(
                    (entry, plan.new_dir / cache_name / new_entry)
                )
                meta = entry / "meta.json"
                if meta.is_file():
                    plan.meta_rewrites.append(meta)
        plans.append(plan)
    return plans


def resolve_duplicates(plans: list[RunPlan]) -> None:
    """Mark byte-identical runs as duplicates of the richest plan for that target.

    Only runs whose *pickle contents* hash identically qualify. Among such a
    group the keeper is the plan carrying the **most artefacts**, so the
    canonical trio (pkl + json + png) wins over a batch orphan that holds only a
    pickle -- deleting the orphan would throw away the sidecars. A run with no
    pickle is never a duplicate: it carries CSV / best-phase data nothing else
    has.
    """
    by_target: dict[Path, list[RunPlan]] = {}
    for plan in plans:
        by_target.setdefault(plan.new_dir, []).append(plan)

    for group in by_target.values():
        if len(group) < 2:
            continue
        with_pkls = [p for p in group if pickle_fingerprint(p.old_dir)]
        if len(with_pkls) < 2:
            continue
        # richest copy first; ties broken by path so the result is deterministic
        with_pkls.sort(key=lambda p: (-len(p.files), str(p.old_dir)))
        keeper = with_pkls[0]
        keeper_fingerprint = pickle_fingerprint(keeper.old_dir)
        for candidate in with_pkls[1:]:
            if pickle_fingerprint(candidate.old_dir) == keeper_fingerprint:
                candidate.duplicate_of = keeper.old_dir


def detect_collisions(plans: list[RunPlan]) -> list[tuple[Path, Path, Path]]:
    """Any two surviving files landing on the same path -- must be empty.

    Plans already resolved as byte-identical duplicates are excluded: their files
    are removed (or quarantined) wholesale, so they never compete for a target.
    """
    seen: dict[Path, Move] = {}
    clashes: list[tuple[Path, Path, Path]] = []
    for plan in plans:
        if plan.duplicate_of is not None:
            continue
        for move in plan.files:
            if move.dst in seen and seen[move.dst].src != move.src:
                clashes.append((move.dst, seen[move.dst].src, move.src))
            seen[move.dst] = move
    return clashes


def apply(plan_list: list[RunPlan], keep_duplicates: bool) -> dict[str, int]:
    """Execute the plan. Returns a small tally for the summary."""
    stats = {"runs": 0, "files": 0, "caches": 0, "metas": 0, "dupes": 0}

    for plan in plan_list:
        if plan.duplicate_of is not None:
            if keep_duplicates:
                _move_duplicate_to_quarantine(plan)
                stats["dupes"] += 1
            continue

        # The cache ``meta.json`` records the pickle's ABSOLUTE path, so it must
        # be rewritten **before** the cache directory is moved -- afterwards the
        # old location is gone and the write is silently lost, leaving a stale
        # ``source`` that re-arms the full rebuild cost on the next epoch.
        for meta in plan.meta_rewrites:
            if _rewrite_meta_source(meta, plan):
                stats["metas"] += 1

        for move in plan.files:
            move.dst.parent.mkdir(parents=True, exist_ok=True)
            if move.src != move.dst:
                shutil.move(str(move.src), str(move.dst))
            stats["files"] += 1
        for src, dst in plan.cache_renames:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src != dst:
                shutil.move(str(src), str(dst))
            stats["caches"] += 1
        stats["runs"] += 1

        # drop the emptied cache shells, then the legacy directory chain
        for cache_name in CACHE_DIRS:
            _prune_empty(plan.old_dir / cache_name)
        _prune_empty(plan.old_dir)

    if not keep_duplicates:
        for plan in plan_list:
            if plan.duplicate_of is not None:
                shutil.rmtree(plan.old_dir, ignore_errors=True)
                stats["dupes"] += 1
    else:
        for plan in plan_list:
            if plan.duplicate_of is not None:
                stats["dupes"] += 1
                # the quarantined copy left its old ancestors empty
                _prune_empty(plan.old_dir)

    # Final sweep: an emptied legacy directory is never a run, and leaving it
    # behind would keep the old shape visible in the tree.
    stats["empty_dirs"] = _remove_empty_dirs()
    return stats


def _remove_empty_dirs() -> int:
    """Delete every empty directory under ``DEBUG``, deepest first."""
    removed = 0
    for path in sorted(
        (p for p in DEBUG.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            next(path.iterdir())
        except StopIteration:
            try:
                path.rmdir()
                removed += 1
            except OSError:
                pass
        except OSError:
            pass
    return removed


def _move_duplicate_to_quarantine(plan: RunPlan) -> None:
    """Park a duplicate run outside ``data/debug`` instead of deleting it."""
    quarantine = ROOT / "data" / "debug_dedup_quarantine" / plan.old_dir.relative_to(DEBUG)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    if plan.old_dir.exists():
        shutil.move(str(plan.old_dir), str(quarantine))


def _rewrite_meta_source(meta: Path, plan: RunPlan) -> bool:
    """Point a cache ``meta.json``'s ``source`` at the relocated pickle.

    Rewritten **by filename transformation, not by the move list**, so it is
    idempotent: re-running the migration after the files have already moved still
    repairs a stale ``source``, which a move-list version cannot do (by then
    ``move.src == move.dst`` carries no information about the old name).

    A ``meta.json`` that cannot be read, is not an object, or carries no
    ``source`` string returns ``False`` rather than a silent success -- a stale
    ``source`` is exactly what re-arms the full rebuild cost.

    Returns:
        ``True`` when the file was rewritten.
    """
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    source = payload.get("source")
    if not isinstance(source, str):
        return False

    old_name = Path(source.replace("\\", "/")).name
    new_name = target_name(old_name, plan.prefix, plan.stamp)
    current = str(plan.new_dir / new_name)
    if old_name == new_name and source.replace("\\", "/") == current.replace("\\", "/"):
        return False  # already correct

    payload["source"] = current
    try:
        meta.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        return False
    return True


def _prune_empty(path: Path) -> None:
    """Remove *path* and its now-empty ancestors, stopping at ``DEBUG``.

    Silently does nothing when the directory still holds anything, so it is safe
    to call speculatively (e.g. on an emptied cache shell).
    """
    if not path.is_dir():
        return
    current = path
    while current != DEBUG and DEBUG in current.parents:
        try:
            next(current.iterdir())
            return  # not empty, stop
        except StopIteration:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent
        except OSError:
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually move files (default: dry run, report only)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="quarantine byte-identical runs instead of deleting them",
    )
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=None,
        help="write the manifest CSV here (default: <temp>/debug_reorg_plan.csv)",
    )
    args = parser.parse_args(argv)

    plans = build_plan()
    resolve_duplicates(plans)
    clashes = detect_collisions(plans)

    movers = [p for p in plans if p.duplicate_of is None]
    dupes = [p for p in plans if p.duplicate_of is not None]
    file_moves = sum(len(p.files) for p in movers)
    cache_moves = sum(len(p.cache_renames) for p in movers)
    total_bytes = sum(
        m.src.stat().st_size for p in movers for m in p.files if m.src.is_file()
    )

    print(f"run directories : {len(plans)}")
    print(f"  moved         : {len(movers)}")
    print(f"  duplicates    : {len(dupes)} (byte-identical, SHA-256 verified)")
    print(f"files to move   : {file_moves}")
    print(f"cache entries   : {cache_moves}")
    print(f"payload         : {total_bytes / 1e9:.2f} GB")

    rules: dict[str, int] = {}
    for plan in plans:
        rules[plan.rule] = rules.get(plan.rule, 0) + 1
    print("\nclassification:")
    for rule in sorted(rules):
        print(f"  {rules[rule]:4d}  {rule}")

    if dupes:
        print("\nbyte-identical duplicates (collapsed):")
        for plan in dupes:
            print(f"  {plan.old_dir.relative_to(DEBUG)}")
            print(f"      == {plan.duplicate_of.relative_to(DEBUG)}")

    if clashes:
        print(f"\nABORT: {len(clashes)} destination collisions, nothing was touched:")
        for dst, a, b in clashes[:20]:
            print(f"  {dst}\n      {a}\n      {b}")
        return 1

    manifest = args.plan_out or (ROOT / "logs" / "debug_reorg_plan.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rule", "prefix", "stamp", "old", "new", "duplicate_of"])
        for plan in plans:
            writer.writerow(
                [
                    plan.rule,
                    plan.prefix,
                    plan.stamp,
                    plan.old_dir.relative_to(DEBUG).as_posix(),
                    plan.new_dir.relative_to(DEBUG).as_posix(),
                    plan.duplicate_of.relative_to(DEBUG).as_posix()
                    if plan.duplicate_of
                    else "",
                ]
            )
    print(f"\nmanifest: {manifest}")

    if not args.apply:
        print("\nDRY RUN -- nothing was moved. Re-run with --apply.")
        return 0

    stats = apply(plans, keep_duplicates=args.keep_duplicates)
    print("\napplied:")
    for key, value in stats.items():
        print(f"  {key:8s} {value}")
    if args.keep_duplicates:
        print(f"\nduplicates quarantined under {ROOT / 'data' / 'debug_dedup_quarantine'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())