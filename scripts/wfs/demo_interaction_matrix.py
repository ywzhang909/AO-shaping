#!/usr/bin/env python3
"""Interaction-matrix calibration demo (extracted from
``ao_shaping.optimizer.wf.interaction_matrix`` — see issue #90).

Calculates and re-loads the DM interaction matrix via the slope method
(requires hardware: ThorlabWFS + NLight DM).
"""

from __future__ import annotations

import sys
from loguru import logger
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.optimizer.wf.interaction_matrix import (
    calculate_interaction_matrix,
    load_interaction_matrix,
)
from ao_shaping.utils import logger


def main() -> None:
    # Example usage
    try:
        # Calculate and save interaction matrix
        matrix = calculate_interaction_matrix(
            wfs_res="768",
            disturb_voltage=50.0,
            wait_time_s=0.1,
            pupil_diameter=2.27,
            save_path="data/interaction_matrix.txt",
        )
        logger.info(f"Interaction matrix shape: {matrix.shape}")

        # Load matrix
        loaded_matrix = load_interaction_matrix("data/interaction_matrix.txt")
        logger.info(f"Loaded matrix shape: {loaded_matrix.shape}")
        logger.info("Interaction matrix calculation completed successfully!")

    except Exception as e:
        logger.error(f"Error during interaction matrix calculation: {e}")
        raise


if __name__ == "__main__":
    main()
