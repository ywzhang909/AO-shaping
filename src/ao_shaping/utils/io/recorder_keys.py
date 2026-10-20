"""Single source for the ``Recorder`` record-dict key names.

Why a leaf module
-----------------
The keys below are an **on-disk contract**: they become column names in the
pickled ``Recorder`` history and the sidecar HDF5/CSV that ``report/`` generators
and ``ml/hwdataset`` read back. Before this module the same names were spelled as
bare string literals in ~40 ``src/`` files *and* re-declared privately in
``ml/hwdataset/index.py`` (``_PHASE_KEY`` / ``_COEFF_KEY`` / ``_IMAGE_KEY``), so
writer and readers had two disconnected vocabularies for one format -- the exact
"two implementations drift" failure this repo has been bitten by repeatedly.

This module holds **only** string constants and imports nothing, so any layer may
use it without an import cycle or a driver side effect (same rationale as the
zero-import leaf ``ao_shaping.utils.cli.params``).

Red line
--------
Changing a value here changes the on-disk format: previously saved runs stop
matching. Values are pinned byte-for-byte by
``tests/ao_shaping/utils/test_recorder_keys.py``; a rename must be a deliberate,
migration-backed decision, never a drive-by edit. Callers that want the same
guarantee should reference these names instead of re-typing the literal.
"""

from __future__ import annotations

#: Row index, assigned by :meth:`Recorder.append` when a record omits it.
RECORD_ID = "_id"

#: Epoch counter of the optimization loop.
RECORD_EPOCH = "_epoch"

#: Measured camera frame for the row (2-D ndarray).
RECORD_IMAGE = "_img"

#: Command vector written to the actuator -- Zernike coefficients, whose meaning
#: is carried by ``len`` (triangular Zernike count vs a perfect square grid).
RECORD_COEFFS = "_c"

#: Actuator voltages for the row.
RECORD_VOLTAGES = "_v"

#: Panel phase as displayed, when recorded alongside the command vector.
RECORD_PHASE = "_phase"

#: Gradient estimate for the row.
RECORD_GRAD = "_grad"

#: Signed difference between the +/- perturbation evaluations.
RECORD_DIFF = "_diff"

#: Gate verdict for the epoch: ``applied`` / ``fold`` / ``noise``.
RECORD_GATE = "_gate"

#: Bucket radius used for the row's measurement.
RECORD_RADIUS = "_r"

#: Encircled energy inside the target ROI, as a percentage string key.
RECORD_ROI_ENERGY = "_p%"

#: Every key this module names, for contract tests and iteration.
RECORD_KEYS: tuple[str, ...] = (
    RECORD_ID,
    RECORD_EPOCH,
    RECORD_IMAGE,
    RECORD_COEFFS,
    RECORD_VOLTAGES,
    RECORD_PHASE,
    RECORD_GRAD,
    RECORD_DIFF,
    RECORD_GATE,
    RECORD_RADIUS,
    RECORD_ROI_ENERGY,
)
