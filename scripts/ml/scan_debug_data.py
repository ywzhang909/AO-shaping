"""List low-value debug runs (sim-only or <2 epochs) for review. Never deletes.

Offline triage helper: opens every ``*.pkl`` under ``data/debug`` and prints the
run directories that look disposable, so a human decides what to remove. It
**only prints** -- there is no delete path in this file.

Note on layout: runs follow the canonical
``data/debug/<prefix>/<YYYYmmdd_HHMMSS>/`` shape, so the family is resolved by
walking ancestors and skipping bare run stamps (``YYYYMMDD`` /
``YYYYmmdd_HHMMSS``), not by taking ``parent.parent`` blindly.
"""

import json
import pickle
import re
from pathlib import Path

debug_root = Path(__file__).resolve().parents[2] / "data" / "debug"

#: Derived caches live inside a run directory and are not runs themselves.
CACHE_DIRS = (".cache", ".gsnet_cache", ".hw_cache")

#: A run directory named with a bare stamp carries no family meaning.
RUN_STAMP = re.compile(r"^\d{8}(_\d{6})?$")

to_delete = []


def family_of(pkl_path: Path) -> str:
    """Nearest ancestor directory name that is not a bare run stamp."""
    for parent in pkl_path.parents:
        if parent == debug_root:
            break
        name = parent.name
        if name and not RUN_STAMP.match(name):
            return name
    return pkl_path.parent.name


# Scan all pickle files
for pkl_path in debug_root.rglob('*.pkl'):
    # Skip derived-cache directories
    if any(part in CACHE_DIRS for part in pkl_path.relative_to(debug_root).parts):
        continue

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"ERROR loading {pkl_path}: {e}")
        continue

    # Count iterations (epochs/rounds)
    if isinstance(data, dict):
        num_iters = len(data)
    elif data is None:
        num_iters = 0
    else:
        num_iters = 1

    # Check for simulation marker in sidecar JSON
    is_sim = False
    json_path = pkl_path.with_suffix('.json')
    if json_path.exists():
        try:
            with open(json_path, encoding='utf-8') as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                config_str = meta.get('config', '')
                if 'cam_type' in config_str and 'sim' in config_str:
                    is_sim = True
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    # Also check family name for sim_calib_abba
    family = family_of(pkl_path)
    if 'sim_calib_abba' in family:
        is_sim = True

    run_dir = pkl_path.parent
    if is_sim or num_iters < 2:
        to_delete.append((run_dir, is_sim, num_iters))
        print(f"{'SIM' if is_sim else 'LOW_ITER'} | iters={num_iters:3d} | {run_dir}")

print(f"\nTotal directories to review: {len(to_delete)}")
for d, is_sim, iters in to_delete:
    print(f"  {'SIM' if is_sim else 'LOW_ITER'} | iters={iters:3d} | {d}")