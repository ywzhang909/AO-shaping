"""Offline regression: ``optimize_pib`` must support every ``objective``.

The recorder is built with ``mark=objective``, so the per-epoch log must carry a
column named after the objective. A hardcoded ``"pib"`` key made
``--objective radiu/avg_radiu`` fail the ``Recorder.append`` assert.

Runs entirely in simulation via ``pib_sim_eval.patched_pib_simulation`` (SimDM
registry + SimCamera), so no hardware is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import pib as pib_module
from ao_shaping.optimizer.wfless.pib_sim_eval import (
    ACTIVE_MASK,
    LANDSCAPE,
    patched_pib_simulation,
)


def _run(objective: str):
    with patched_pib_simulation():
        return pib_module.optimize_pib(
            center=LANDSCAPE.center,
            epochs=4,
            r_bucket=10,
            delta=1.2,
            lr=0.5,
            exposure_time_ms=20,
            shrink_iter=0,
            cam_id=0,
            show=False,
            init_v=np.zeros(8, dtype=np.float64),
            cam_size=LANDSCAPE.shape[0],
            target_max_brightness=0,
            dm_unit_mask=ACTIVE_MASK.copy(),
            dm_neibor_diff=100,
            dm_max_voltage=12,
            dm_min_voltage=-12,
            random_seed=1,
            objective=objective,
        )


@pytest.mark.parametrize("objective", ["pib", "radiu", "avg_radiu"])
def test_recorder_column_is_named_after_objective(objective):
    rec = _run(objective)

    assert rec.mark == objective
    assert objective in rec.dataframe.columns
    # Every per-epoch row must carry the mark column (this is what the assert checks).
    assert all(objective in row for row in rec.history)
