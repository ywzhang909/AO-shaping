"""Offline tests for the ``rms_pib`` objective (adaptively-weighted PIB + in-ROI RMS).

``rms_pib`` combines the bucket-ratio ("PIB") over the **target-shaped** ROI with
the in-ROI intensity uniformity (RMS) into a single objective
``J = w_pib(t)*pib_term + w_rms(t)*rms_term``. The weights adapt automatically so
the term that improves ``J`` more gets the higher weight ("哪个对J提升大则哪个
权重大"). These tests pin:

* ``rms_pib_terms``: the two terms are the in-ROI energy fraction and
  ``1 - u/(1+u)`` with ``u = std/mean`` over the ROI (``1`` = perfectly flat);
* ``_update_dynamic_weights``: the softmax/EMA weight adaptation (first call
  keeps 50/50, improving terms win, both-degrading steps keep the weights,
  weights always sum to 1 and respect the floor);
* the runner wiring: the new CLI options exist and ``rms_pib`` is a valid
  ``--objective`` choice.
"""

from __future__ import annotations

import numpy as np
from click.testing import CliRunner

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    _resolve_init_weights,
    _update_dynamic_weights,
    rms_pib_terms,
    target_shape_roi,
)
from ao_shaping.runners.slm_pib_runner import ObjectiveParams, run


# --------------------------------------------------------------------------- #
# rms_pib_terms: the two objective terms
# --------------------------------------------------------------------------- #


def test_rms_pib_terms_returns_pib_and_rms_terms() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0

    pib_term, rms_term = rms_pib_terms(image, (40.0, 32.0), "rectangle", 10, 4 / 3)

    assert 0.0 < pib_term <= 1.0
    assert 0.0 < rms_term <= 1.0


def test_rms_pib_terms_pib_term_is_the_in_roi_energy_fraction() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0
    center = (40.0, 32.0)
    roi = target_shape_roi((64, 80), center, "rectangle", 10, 4 / 3)

    pib_term, _ = rms_pib_terms(image, center, "rectangle", 10, 4 / 3)

    assert np.isclose(pib_term, float(image[roi].sum()) / float(image.sum()))


def test_rms_pib_terms_rms_term_is_one_for_a_flat_roi() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0

    _, rms_term = rms_pib_terms(image, (40.0, 32.0), "square", 4)

    assert np.isclose(rms_term, 1.0)


def test_rms_pib_terms_rms_term_penalises_non_uniform_roi() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0

    _, flat_rms = rms_pib_terms(image, (40.0, 32.0), "square", 4)
    # A larger ROI includes zero pixels -> non-uniform -> lower RMS term.
    _, nonuniform_rms = rms_pib_terms(image, (40.0, 32.0), "square", 8)

    assert nonuniform_rms < flat_rms
    assert nonuniform_rms < 1.0


def test_rms_pib_terms_handles_zero_total_intensity() -> None:
    pib_term, rms_term = rms_pib_terms(np.zeros((32, 32)), (16.0, 16.0))

    assert pib_term == 0.0
    assert rms_term == 0.0


def test_rms_pib_terms_handles_empty_roi() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[5:9, 5:9] = 1.0

    pib_term, rms_term = rms_pib_terms(image, (60.0, 50.0), "square", 4)

    assert pib_term == 0.0
    assert rms_term == 0.0


def test_rms_pib_terms_rejects_non_2d_input() -> None:
    try:
        rms_pib_terms(np.zeros((4, 4, 3)), (2.0, 2.0))
    except ValueError:
        return
    raise AssertionError("3D input must raise ValueError")


def test_rms_pib_terms_default_target_size_derives_from_frame() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0

    # target_size=None -> min(height, width) = 64 -> the whole frame is the ROI.
    pib_term, _ = rms_pib_terms(image, (40.0, 32.0), "square", None)

    assert np.isclose(pib_term, 1.0)


def test_rms_pib_terms_terms_are_bounded() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        image = rng.random((48, 64))
        pib_term, rms_term = rms_pib_terms(image, (32.0, 24.0), "rectangle", 12, 4 / 3)
        assert 0.0 <= pib_term <= 1.0
        assert 0.0 <= rms_term <= 1.0


# --------------------------------------------------------------------------- #
# _update_dynamic_weights: adaptive PIB/RMS weighting
# --------------------------------------------------------------------------- #


def test_update_dynamic_weights_first_call_keeps_50_50() -> None:
    state: dict = {}

    w_pib, w_rms = _update_dynamic_weights(state, pib=0.3, rms=0.7, j=0.5)

    assert w_pib == 0.5
    assert w_rms == 0.5
    assert state["prev_pib"] == 0.3
    assert state["prev_rms"] == 0.7
    assert state["prev_j"] == 0.5
    assert state["prev_set"] is True


def test_update_dynamic_weights_improving_pib_raises_w_pib() -> None:
    state: dict = {}
    _update_dynamic_weights(state, pib=0.5, rms=0.5, j=0.5)

    w_pib, w_rms = _update_dynamic_weights(
        state, pib=0.6, rms=0.5, j=0.55, w_ema_decay=0.0
    )

    assert w_pib > 0.5
    assert w_rms < 0.5


def test_update_dynamic_weights_improving_rms_raises_w_rms() -> None:
    state: dict = {}
    _update_dynamic_weights(state, pib=0.5, rms=0.5, j=0.5)

    w_pib, w_rms = _update_dynamic_weights(
        state, pib=0.5, rms=0.6, j=0.55, w_ema_decay=0.0
    )

    assert w_rms > 0.5
    assert w_pib < 0.5


def test_update_dynamic_weights_both_improve_keeps_balance() -> None:
    state: dict = {}
    _update_dynamic_weights(state, pib=0.5, rms=0.5, j=0.5)

    w_pib, w_rms = _update_dynamic_weights(
        state, pib=0.6, rms=0.6, j=0.6, w_ema_decay=0.0
    )

    assert np.isclose(w_pib, 0.5, atol=1e-6)
    assert np.isclose(w_rms, 0.5, atol=1e-6)


def test_update_dynamic_weights_both_degrade_keeps_weights() -> None:
    state: dict = {}
    w0 = _update_dynamic_weights(state, pib=0.5, rms=0.5, j=0.5)

    w1 = _update_dynamic_weights(state, pib=0.4, rms=0.4, j=0.4)

    assert w1 == w0


def test_update_dynamic_weights_weights_sum_to_one() -> None:
    state: dict = {}
    _update_dynamic_weights(state, pib=0.5, rms=0.5, j=0.5)

    for _ in range(5):
        w_pib, w_rms = _update_dynamic_weights(
            state, pib=0.51, rms=0.5, j=0.505, w_ema_decay=0.0
        )
        assert np.isclose(w_pib + w_rms, 1.0)


def test_update_dynamic_weights_weights_drift_back_toward_balance() -> None:
    """After a strong PIB improvement, a subsequent RMS improvement shifts the
    weight back toward balance (the term that improves J more wins)."""
    state: dict = {}
    _update_dynamic_weights(state, pib=0.5, rms=0.5, j=0.5)

    w_after_pib = _update_dynamic_weights(
        state, pib=0.7, rms=0.5, j=0.6, w_ema_decay=0.0
    )
    assert w_after_pib[0] > 0.5

    w_after_rms = _update_dynamic_weights(
        state, pib=0.7, rms=0.7, j=0.7, w_ema_decay=0.0
    )
    assert w_after_rms[0] < w_after_pib[0]
    assert w_after_rms[1] > w_after_pib[1]


def test_update_dynamic_weights_floor_bounds_weights() -> None:
    state: dict = {}
    _update_dynamic_weights(state, pib=0.5, rms=0.5, j=0.5)

    w_pib, w_rms = _update_dynamic_weights(
        state, pib=1.0, rms=0.5, j=0.75, w_ema_decay=0.0, w_floor=0.4
    )

    assert w_pib >= 0.4
    assert w_rms >= 0.4
    assert np.isclose(w_pib + w_rms, 1.0)


# --------------------------------------------------------------------------- #
# _update_dynamic_weights: three-term mode (energy term participates)
# --------------------------------------------------------------------------- #


def test_update_dynamic_weights_three_term_first_call_keeps_third_thirds() -> None:
    state: dict = {}

    w_pib, w_rms, w_ee = _update_dynamic_weights(state, pib=0.3, rms=0.7, ee=1.0, j=0.5)

    assert np.isclose(w_pib, 1.0 / 3)
    assert np.isclose(w_rms, 1.0 / 3)
    assert np.isclose(w_ee, 1.0 / 3)
    assert state["prev_ee"] == 1.0
    assert state["prev_set"] is True


def test_update_dynamic_weights_three_term_improving_ee_raises_w_ee() -> None:
    state: dict = {}
    _update_dynamic_weights(state, pib=0.5, rms=0.5, ee=0.9, j=0.5)

    w_pib, w_rms, w_ee = _update_dynamic_weights(
        state, pib=0.5, rms=0.5, ee=1.0, j=0.55, w_ema_decay=0.0
    )

    assert w_ee > 1.0 / 3
    assert np.isclose(w_pib + w_rms + w_ee, 1.0)


def test_update_dynamic_weights_three_term_improving_pib_raises_w_pib() -> None:
    state: dict = {}
    _update_dynamic_weights(state, pib=0.4, rms=0.5, ee=0.9, j=0.5)

    w_pib, w_rms, w_ee = _update_dynamic_weights(
        state, pib=0.6, rms=0.5, ee=0.9, j=0.55, w_ema_decay=0.0
    )

    assert w_pib > w_rms
    assert w_pib > w_ee
    assert np.isclose(w_pib + w_rms + w_ee, 1.0)


def test_update_dynamic_weights_three_term_all_degrade_keeps_weights() -> None:
    state: dict = {}
    w0 = _update_dynamic_weights(state, pib=0.5, rms=0.5, ee=0.9, j=0.5)

    w1 = _update_dynamic_weights(state, pib=0.4, rms=0.4, ee=0.8, j=0.4)

    assert w1 == w0


def test_update_dynamic_weights_three_term_floor_bounds_weights() -> None:
    state: dict = {}
    _update_dynamic_weights(state, pib=0.5, rms=0.5, ee=0.9, j=0.5)

    w_pib, w_rms, w_ee = _update_dynamic_weights(
        state, pib=1.0, rms=0.5, ee=0.9, j=0.75, w_ema_decay=0.0, w_floor=0.1
    )

    # 3-term floor: each weight >= 0.1 and they sum to 1 (1 - 3*0.1 = 0.7 shared).
    assert w_pib >= 0.1
    assert w_rms >= 0.1
    assert w_ee >= 0.1
    assert np.isclose(w_pib + w_rms + w_ee, 1.0)


def test_objective_params_default_target_size_is_44() -> None:
    """Shared ``ObjectiveParamsPib`` default (runner_common, not local)."""
    obj = ObjectiveParams()

    assert obj.target_size == 44.0


# --------------------------------------------------------------------------- #
# _resolve_init_weights: initial PIB/RMS/EE weights of the rms_pib objective
# --------------------------------------------------------------------------- #


def test_resolve_init_weights_defaults_to_thirds() -> None:
    assert _resolve_init_weights(None, None, None) == (1.0 / 3, 1.0 / 3, 1.0 / 3)


def test_resolve_init_weights_single_provided_shares_remainder() -> None:
    assert _resolve_init_weights(0.6, None, None) == (0.6, 0.2, 0.2)
    assert _resolve_init_weights(None, 0.6, None) == (0.2, 0.6, 0.2)
    assert _resolve_init_weights(None, None, 0.6) == (0.2, 0.2, 0.6)


def test_resolve_init_weights_two_provided_share_remainder() -> None:
    assert np.allclose(_resolve_init_weights(0.6, 0.3, None), (0.6, 0.3, 0.1))
    assert np.allclose(_resolve_init_weights(0.4, None, 0.4), (0.4, 0.2, 0.4))


def test_resolve_init_weights_all_provided_are_normalised() -> None:
    w_pib, w_rms, w_ee = _resolve_init_weights(0.5, 0.3, 0.1)

    assert np.isclose(w_pib, 5.0 / 9)
    assert np.isclose(w_rms, 3.0 / 9)
    assert np.isclose(w_ee, 1.0 / 9)
    assert np.isclose(w_pib + w_rms + w_ee, 1.0)


def test_resolve_init_weights_always_normalises_to_one() -> None:
    for args in (
        (0.8, None, None),
        (0.8, 0.1, None),
        (0.5, 0.3, 0.2),
        (0.0, 0.0, 1.0),
    ):
        out = _resolve_init_weights(*args)
        assert np.isclose(sum(out), 1.0)
        assert all(w >= 0.0 for w in out)


def test_resolve_init_weights_sum_exceeds_one_raises() -> None:
    try:
        _resolve_init_weights(0.6, 0.5, None)
    except ValueError:
        return
    raise AssertionError("weights summing to > 1 must raise ValueError")


def test_resolve_init_weights_all_zero_raises() -> None:
    try:
        _resolve_init_weights(0.0, 0.0, 0.0)
    except ValueError:
        return
    raise AssertionError("all-zero weights must raise ValueError")


# --------------------------------------------------------------------------- #
# Runner wiring: CLI options + dataclass defaults (no hardware)
# --------------------------------------------------------------------------- #


def test_objective_params_defaults_include_rms_pib_weights() -> None:
    obj = ObjectiveParams()

    assert obj.w_ema_decay == 0.9
    assert obj.w_floor == 0.1
    assert obj.w_temperature == 8.0
    assert obj.w_pib_init is None
    assert obj.w_rms_init is None
    assert obj.w_ee_init is None


def test_cli_help_lists_rms_pib_objective() -> None:
    result = CliRunner().invoke(run, ["spgd", "--help"])

    assert result.exit_code == 0, result.output
    assert "rms_pib" in result.output
    for opt in (
        "--w_ema_decay",
        "--w_floor",
        "--w_temperature",
        "--w_pib_init",
        "--w_rms_init",
        "--w_ee_init",
    ):
        assert opt in result.output