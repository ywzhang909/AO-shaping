"""R-69 P2 precondition: a per-epoch characterization of both ``optimize_pib`` paths.

The P2 plan is to extract the shared SPGD-PIB skeleton into one ``SpidPibLoop``
so the two ``optimize_pib`` implementations stop duplicating it. Before such an
extraction can be *proven* behavior-preserving, each path needs a pinned
per-epoch characterization: if the extraction moves either path's numerics,
this test must fail.

Measured facts that shaped this file:

* Both paths are deterministic under simulation once their RNG is pinned.
  ``wfless.pib`` draws from a local ``default_rng(random_seed)``;
  ``combined_optimizer`` draws from the **global** numpy RNG
  (``np.random.binomial``), so it needs ``np.random.seed(...)``. The digests
  below were byte-identical across ``AO_CPU_THREADS`` 1 / 8 / 16, so they are
  not sensitive to float noise from thread counts.
* The two paths **diverge at epoch 1** -- different perturbation signs, because
  they draw their perturbation patterns from different RNG streams. So "compare
  the two paths epoch by epoch and expect equal numbers" cannot hold literally.
  What can be pinned is each path separately, plus the facts a shared skeleton
  would have to preserve.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ao_shaping.optimizer import combined_optimizer as combined_module
from ao_shaping.optimizer.wfless import pib as pib_module
from ao_shaping.optimizer.wfless.pib_sim_eval import (
    ACTIVE_MASK,
    LANDSCAPE,
    patched_sensorless_simulation,
)

_ARGS = dict(
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
)

#: Per-epoch trajectory digests: sha256 over the rounded voltage / difference
#: vectors. A change here means that path's numerics moved.
PIB_VOLTAGE_DIGEST = "8cd1dc9b82c5692c"
PIB_DIFF_DIGEST = "c11f42c0b7c6e8fd"
COMBINED_VOLTAGE_DIGEST = "ae36d20b20dec693"
COMBINED_DIFF_DIGEST = "54202c04dec3d012"

#: Row counts for the pinned setting (initial frame + one row per epoch).
PIB_ROWS = 5
COMBINED_ROWS = 5


def _digest(rows: list[dict], key: str) -> str:
    payload = [
        np.asarray(row.get(key, []), dtype=np.float64).round(9).tolist() for row in rows
    ]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def _run_pib() -> list[dict]:
    with patched_sensorless_simulation():
        recorder = pib_module.optimize_pib(
            **_ARGS, random_seed=1, objective="pib"
        )
    return recorder.history


def _run_combined() -> list[dict]:
    state = np.random.get_state()
    try:
        with patched_sensorless_simulation():
            np.random.seed(1)
            recorder = combined_module.optimize_pib(**_ARGS)
    finally:
        np.random.set_state(state)
    return recorder.history


def test_pib_path_trajectory_is_pinned() -> None:
    rows = _run_pib()

    assert len(rows) == PIB_ROWS
    assert _digest(rows, "_v") == PIB_VOLTAGE_DIGEST
    assert _digest(rows, "_diff") == PIB_DIFF_DIGEST


def test_combined_path_trajectory_is_pinned() -> None:
    rows = _run_combined()

    assert len(rows) == COMBINED_ROWS
    assert _digest(rows, "_v") == COMBINED_VOLTAGE_DIGEST
    assert _digest(rows, "_diff") == COMBINED_DIFF_DIGEST


def test_each_path_is_deterministic() -> None:
    """Pinning only works if a re-run reproduces the run."""
    assert _digest(_run_pib(), "_v") == _digest(_run_pib(), "_v")
    assert _digest(_run_combined(), "_v") == _digest(_run_combined(), "_v")


def test_both_paths_start_from_the_same_frame() -> None:
    """The shared prologue must keep producing the same starting point.

    Both optimizers build the camera through the same seam and start from the
    same flat field, so their first recorded objective value has to agree.
    """
    pib_rows = _run_pib()
    combined_rows = _run_combined()

    assert np.allclose(
        np.asarray(pib_rows[0]["_v"], dtype=np.float64),
        np.asarray(combined_rows[0]["_v"], dtype=np.float64),
    )
    assert np.isclose(
        float(pib_rows[0]["J"]), float(combined_rows[0]["J"]), rtol=1e-9
    )


def test_the_two_paths_diverge_after_the_first_epoch() -> None:
    """They are not interchangeable -- an extractor must not merge the tails.

    The perturbation signs come from different RNG streams, so the paths part
    company on the first frame and then walk apart. This is the measured reason
    the P2 plan says "extract the skeleton, keep the tails", and the reason a
    cross-path equality check is not a valid acceptance criterion.
    """
    pib_rows = _run_pib()
    combined_rows = _run_combined()

    assert _digest(pib_rows, "_diff") != _digest(combined_rows, "_diff")
    assert not np.allclose(
        np.asarray(pib_rows[1]["_v"], dtype=np.float64),
        np.asarray(combined_rows[1]["_v"], dtype=np.float64),
    )


def test_the_recorder_columns_stay_path_specific() -> None:
    """Each path keeps its own columns; the shared skeleton must not merge them."""
    pib_keys = set(_run_pib()[0])
    combined_keys = set(_run_combined()[0])

    assert {"_grad", "tabu_size", "best_pib"} <= pib_keys
    assert "_raw_pib" in combined_keys
    assert not {"_grad", "tabu_size", "best_pib"} & combined_keys
