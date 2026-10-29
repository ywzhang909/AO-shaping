"""Shape agreement for the archived replay frames: is the target mask actually *filled*?

Why this exists
---------------
The 2026-10-29 live log's biggest caveat is that every verdict only covers
"more uniform (CV) + energy not lost (``ee_rel``)", **not** "grew into the target
shape" (§5.5 there).  The primitive for the missing half already exists --
:func:`ao_shaping.tools.slm.transfer.protocol.shape_agreement` -- and had no
caller anywhere in the tree, while every replay ``*.npz`` already archives the
exact ``target`` mask (and the ``camera_xy`` index grid) the optimiser used.

So this is an offline read of data that already exists: the same mask, the same
coordinates, the same implementation.  Nothing is re-rasterised here -- a second
target mask would be a second truth.

The threshold reference matters
-------------------------------
On this bench the replay window holds **both** the shaped (+1 order) pattern and
the zero-order residual, and the residual can out-peak the shaped spot.  A
threshold taken from the window peak therefore measures the residual, not the
beam: at ``0.5 * window peak`` the read-out is a few dozen pixels (measured
IoU 0.0004 on a session whose ``dCV`` is −0.62).  Every term below is therefore
referenced to the peak **inside the target mask**, and the zero-order disk is
excluded by the archived ``core`` mask.  Both choices are printed with the
result so a reader can re-derive them.

Read-out per (session, shape, stage, phase)
-------------------------------------------
* ``filled_fraction`` -- share of target-mask pixels at or above half of the
  in-target peak.  This is the direct answer to "is it a plateau or a spike".
* ``iou`` / ``boundary_mean_px`` / ``boundary_p95_px`` / ``coverage_fraction``
  -- from :func:`shape_agreement` at half the in-target peak.
* ``energy_in_target`` -- share of the window's energy inside the target mask.
* ``inside_cv`` -- CV of the target-mask pixels (the same quantity the loop
  optimised, but on the un-eroded mask, so it can differ from the audit's
  ``core``-based CV).

Clipped frames (``>= 98%`` of full scale) are listed but dropped from the group
means, exactly as ``audit_replay_blocks`` does for CV.

Usage
-----
    python scripts/slm/audit_shape_agreement.py --campaign data/debug/slm_transfer/multishape_nogs_20261029
    python scripts/slm/audit_shape_agreement.py --campaign ... --session p20261029_analytic_init_s600/square
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.transfer.protocol import shape_agreement

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_replay_blocks import REPLAY_PEAK_GUARD_FRACTION, REPLAY_DIRS, _label_parts

# `scripts._common` lives in this package, so the REPO ROOT (not just `src`)
# must be importable. A direct `python scripts/...` does not put it there;
# pytest does, via `pythonpath = ["src", ".", "scripts"]` in pyproject.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts._common import fmt_general

DEFAULT_CAMPAIGN = ROOT / "data" / "debug" / "slm_transfer" / "multishape_nogs_20261029"
TERMS = ("filled_fraction", "iou", "boundary_mean_px", "boundary_p95_px",
         "coverage_fraction", "energy_in_target", "inside_cv")


def _replay_dirs(root: Path) -> list[Path]:
    """Every replay block under ``root`` (a session dir, or the whole campaign).

    Directories whose name starts with ``_aborted`` are skipped: they hold runs
    that were interrupted mid-way and archived only for the record, so they must
    not leak into a measurement table.
    """
    found: list[Path] = []
    for name in REPLAY_DIRS:
        found.extend(sorted(root.glob(f"**/{name}")))
    return [d for d in found
            if (d / "manifest.json").is_file()
            and not any(part.startswith("_aborted") for part in d.parts)]


def measure(field: np.ndarray, target: np.ndarray, core: np.ndarray, *,
            level: float) -> dict:
    """All shape terms of one frame, referenced to the in-target peak."""
    inside = target > 0.5
    values = field[inside]
    peak = float(values.max())
    if peak <= 0:
        return dict(peak=0.0, filled_fraction=0.0, iou=0.0, boundary_mean_px=None,
                    boundary_p95_px=None, coverage_fraction=0.0,
                    energy_in_target=0.0, inside_cv=None)
    # Zero-order residual (everything outside core and outside the target) is
    # removed before the agreement; the level is then rescaled so that
    # shape_agreement's internal ``level * measured.max()`` is ``level * peak``
    # even when a core pixel outside the target is the brightest one.
    masked = np.where(core | inside, field, 0.0)
    masked_peak = float(masked.max())
    level_eff = level * peak / masked_peak if masked_peak > 0 else level
    agreement = shape_agreement(masked, target, level=level_eff)
    total = float(field.sum())
    return dict(
        peak=peak,
        filled_fraction=float((values >= level * peak).mean()),
        iou=float(agreement["iou"]),
        boundary_mean_px=agreement["boundary_mean_px"],
        boundary_p95_px=agreement["boundary_p95_px"],
        coverage_fraction=float(agreement["coverage_fraction"]),
        energy_in_target=float(values.sum() / total) if total > 0 else 0.0,
        inside_cv=float(values.std() / values.mean()) if values.mean() > 0 else None,
    )


def read_block(directory: Path, *, level: float, session: str = "",
               campaign: Path | None = None) -> list[dict]:
    """One replay block -> one row per archive, with the shape terms."""
    if campaign is not None and directory.is_relative_to(campaign):
        session = str(directory.relative_to(campaign).parent)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for entry in manifest:
        stage, shape, phase = _label_parts(str(entry["label"]))
        row = dict(session=session, block=directory.name, label=entry["label"],
                   stage=stage, shape=shape, phase=phase,
                   exposure_ms=entry.get("exposure_ms"))
        path = directory / str(entry["file"])
        if not path.is_file():
            row["error"] = "frame file missing"
            rows.append(row)
            continue
        with np.load(path, allow_pickle=False) as payload:
            raw = payload["raw_frames"].astype(np.float64)
            xy = payload["camera_xy"]
            target = payload["target"]
            core = payload["core"]
        frame = raw.mean(axis=0)
        field = frame[xy[..., 1].astype(int), xy[..., 0].astype(int)]
        full_scale = float((entry.get("metrics") or {}).get("full_scale") or 4095.0)
        row.update(
            full_scale=full_scale,
            clipped=bool(frame.max() >= REPLAY_PEAK_GUARD_FRACTION * full_scale),
            target_px=int((np.asarray(target) > 0.5).sum()),
        )
        row.update(measure(field, np.asarray(target), np.asarray(core), level=level))
        rows.append(row)
    return rows


def summarise(rows: list[dict], *, key: tuple[str, ...]) -> list[dict]:
    """Group rows by ``key`` and average the shape terms over usable frames."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("error") or row.get("clipped") or row.get("iou") is None:
            continue
        groups.setdefault(tuple(row[k] for k in key), []).append(row)
    out = []
    for group_key, members in sorted(groups.items()):
        record = dict(zip(key, group_key))
        record["n"] = len(members)
        for name in TERMS:
            values = [float(m[name]) for m in members if m.get(name) is not None]
            if values:
                record[name] = float(np.mean(values))
                record[f"{name}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            else:
                record[name] = None
        out.append(record)
    return out


def print_table(records: list[dict], *, key: tuple[str, ...]) -> None:
    head = (f"{'/'.join(key):<40} {'n':>2} {'filled':>7} {'iou':>6} {'b_mean':>7} "
            f"{'b_p95':>7} {'cover':>6} {'E_in':>6} {'cv_in':>6}")
    print(head)
    print("-" * len(head))
    for record in records:
        label = "/".join(str(record[k]) for k in key)
        print(f"{label:<40} {record['n']:>2} "
              f"{fmt_general(record['filled_fraction'], '.3f', missing_none='--'):>7} "
              f"{fmt_general(record['iou'], '.3f', missing_none='--'):>6} "
              f"{fmt_general(record['boundary_mean_px'], '.2f', missing_none='--'):>7} "
              f"{fmt_general(record['boundary_p95_px'], '.2f', missing_none='--'):>7} "
              f"{fmt_general(record['coverage_fraction'], '.3f', missing_none='--'):>6} "
              f"{fmt_general(record['energy_in_target'], '.3f', missing_none='--'):>6} "
              f"{fmt_general(record['inside_cv'], '.3f', missing_none='--'):>6}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--session", default=None,
                        help="restrict to one session, e.g. p20261029_analytic_init_s600/square")
    parser.add_argument("--level", type=float, default=0.5,
                        help="threshold as a fraction of each frame's in-target peak")
    parser.add_argument("--json", type=Path, default=None, help="also write the rows here")
    parser.add_argument("--jsonl", type=Path, default=None,
                        help="stream one JSON line per replay block as soon as it is "
                             "measured (interruption-safe; pair with --resume)")
    parser.add_argument("--resume", action="store_true",
                        help="skip the blocks already present in --jsonl")
    parser.add_argument("--by", choices=("stage_phase", "stage", "phase", "shape"), default="stage_phase")
    return parser.parse_args(argv)


KEYS = {
    "stage_phase": ("stage", "phase"),
    "stage": ("stage",),
    "phase": ("phase",),
    "shape": ("shape",),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.campaign / args.session if args.session else args.campaign
    if not root.is_dir():
        print(f"no such session: {root}", file=sys.stderr)
        return 2
    key = KEYS[args.by]

    # Streaming output: one line per replay block, written and flushed as soon as
    # the block is measured.  The npz reads are the expensive part of this
    # script, so an interrupted sweep (device session ends, machine restarts)
    # must leave every finished block on disk; `--resume` then skips them.
    done: set[tuple[str, str]] = set()
    stream = None
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and args.jsonl.is_file():
            for line in args.jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    saved = json.loads(line)
                    done.add((str(saved.get("session", "")), str(saved.get("block", ""))))
            print(f"[resume] {len(done)} blocks already in {args.jsonl}")
        stream = args.jsonl.open("a", encoding="utf-8")

    rows: list[dict] = []
    try:
        for directory in _replay_dirs(root):
            session = (str(directory.parent.relative_to(args.campaign))
                       if directory.is_relative_to(args.campaign) else "")
            if (session, directory.name) in done:
                continue
            block_rows = read_block(directory, level=args.level, campaign=args.campaign)
            rows.extend(block_rows)
            if stream is not None and block_rows:
                stream.write(json.dumps(dict(session=session, block=directory.name,
                                             level=args.level, rows=block_rows),
                                        ensure_ascii=False) + "\n")
                stream.flush()
    finally:
        if stream is not None:
            stream.close()
    if not rows:
        print(f"nothing new to measure under {root} "
              f"(all {len(done)} blocks already in {args.jsonl})", file=sys.stderr)
        return 0

    clipped = [row["label"] for row in rows if row.get("clipped")]
    missing = [row["label"] for row in rows if row.get("error")]
    shown = root.relative_to(ROOT) if root.is_relative_to(ROOT) else root
    print(f"# {shown}")
    print(f"{len(rows)} archives, level={args.level} (of the in-target peak), "
          f"clipped={len(clipped)}, missing={len(missing)}")
    print_table(summarise(rows, key=key), key=key)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[json] {args.json}")
    if args.jsonl:
        print(f"[jsonl] {args.jsonl} (+{len(rows)} rows this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
