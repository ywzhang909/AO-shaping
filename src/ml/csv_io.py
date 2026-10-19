"""CSV readers shared by the ``ml`` subpackages.

Kept free of ``torch`` so importing it never pulls in the ML stack.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_zernike_coefficients(csv_path: Path, n_terms: int | None = None) -> np.ndarray:
    """Load Zernike coefficients from a one-row CSV.

    Args:
        csv_path: Path to CSV file with coefficients.
        n_terms: Number of terms to load. If None, load all.

    Returns:
        ``(n_terms,)`` ``float32`` array of Zernike coefficients.

    Raises:
        ValueError: if the file has fewer than two rows.
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) < 2:
        raise ValueError(f"Empty or invalid CSV: {csv_path}")

    # First row is header, second row is values
    values = [float(v) for v in rows[1]]

    if n_terms is not None:
        values = values[:n_terms]

    return np.array(values, dtype=np.float32)
