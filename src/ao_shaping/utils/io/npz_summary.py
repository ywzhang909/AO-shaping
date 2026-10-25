"""Compressed ``.npz`` summary writer shared by the bench probe scripts.

``floor_probe`` and ``abba_probe`` each declared their own
``save_*_summary_npz`` with a byte-identical body (create parents,
``np.savez_compressed`` over ``np.asarray``-ed values, log, return the
path). One writer here; the probe scripts keep their domain-specific
names as thin wrappers where external consumers exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

__all__ = ["save_summary_npz"]


def save_summary_npz(path: Path, summary: dict[str, Any]) -> Path:
    """Write the summary dict to a compressed ``.npz``; parents are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in summary.items()})
    logger.info("summary -> {}", path)
    return path
