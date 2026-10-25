"""Guards for ``scripts/slm/reproduce_transfer_block_without_gs.py``.

The script exists for one claim only: the anchored block can be reproduced from an
already accepted phase, **without** a Gerchberg-Saxton seed. Two things must therefore
stay true no matter how the script is refactored:

1. it never resolves the name ``gs_seed`` (not imported, not called, not aliased);
2. its frame sampling really indexes ``raw_frames`` by ``camera_xy``, because a silent
   transpose there would still produce plausible-looking numbers.

Both are asserted here instead of in the report prose: prose rots, a test does not.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "reproduce_transfer_block_without_gs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reproduce_transfer_block_without_gs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_never_references_gs_seed():
    """The no-GS claim is a property of the source, not of a run's logs."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported |= {alias.name for alias in node.names} | {alias.asname for alias in node.names if alias.asname}
        elif isinstance(node, ast.Import):
            imported |= {alias.asname or alias.name.split(".")[0] for alias in node.names}

    offenders = [name for name in (*names, *attributes, *imported) if "gs_seed" in name]
    assert not offenders, f"reproduction script must not touch GS seeding, found {offenders}"


def test_sampled_indexes_by_xy_columns():
    """``camera_xy[..., 0]`` is x (columns) and ``[..., 1]`` is y (rows)."""
    module = _load_module()
    frame = np.zeros((4, 5), dtype=np.float64)
    frame[1, 2] = 7.0
    frame[3, 0] = 11.0
    xy = np.array([[[2, 1], [0, 3]]], dtype=np.int64)

    sampled = module._sampled({"raw_frames": np.stack([frame, frame])}, xy)

    assert sampled.shape == (1, 2)
    assert np.allclose(sampled, [[7.0, 11.0]])
