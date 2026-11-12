"""The optimizer's cached anchor forward: one fewer full ``predict`` per step, exactly.

``DifferentiableUniformityOptimizer._anchor_prediction()`` keeps
``predict(seed + accepted)`` across steps, and only ``accept()`` drops it. The
invariant holds because ``_accepted`` is written by ``accept()`` alone and the
frozen forward is a deterministic pure function of its inputs, so a step without
an accept asks for exactly the tensor the previous step already had. This file
pins both halves:

* **no duplicate work** -- the first step pays for the anchor, later steps reuse
  it (2 ``predict`` calls instead of 3), and ``accept()`` makes the next step pay
  for it again;
* **exactness** -- what the cache serves is bit-for-bit (``torch.equal``) a fresh
  computation on the current ``seed + accepted``, so the recorded diagnostics are
  the same numbers as before the cache existed.

A second duplicate forward sits in the *caller*: the anchored closed loop
(``tools/slm/transfer/validate.py``) asks for ``forward.predict(optimizer.current_phase)[0]``
once per step, and that phase is exactly the post-step ``seed + delta`` whose
prediction ``update()`` already made for its diagnostics. :attr:`
DifferentiableUniformityOptimizer.last_candidate_prediction` hands that value back
instead, which the tests below pin as bit-identical to the recomputation and as one
fewer full forward per step (3 -> 2 with the anchor cache also hot).

Deliberately **no** ``skipif`` on the fitted checkpoint
``data/debug/physical_forward_defocus/`` (the module-level gate used by
``tests/ao_shaping/algorithm/test_differentiable_defocus_uniformity.py``): that
directory is gitignored and absent on most machines, so a checkpoint-gated test
of this invariant would guard nothing locally. The forward here is synthetic
(:func:`_small_forward`, the same geometry as that file's helper), so the guard
runs on any checkout, CI included.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    DifferentiableUniformityOptimizer,
    FrozenDefocusForward,
    FrozenDefocusParameters,
    UniformityObjective,
)


class _CountingAnchorForward:
    """Forward wrapper counting ``predict`` calls and spotting the anchor one.

    The anchor forward is the no-grad call made on ``seed + accepted``: the main
    forward keeps grad on its input, and the candidate forward runs on
    ``seed + delta`` (projected, so a different value). Everything else delegates.
    """

    def __init__(self, inner: FrozenDefocusForward,
                 optimizer: DifferentiableUniformityOptimizer) -> None:
        self._inner = inner
        self._optimizer = optimizer
        self.calls = 0
        self.anchor_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def predict(self, phase, **kwargs):
        anchor_input = self._optimizer._seed + self._optimizer._accepted
        self.calls += 1
        # ``phase`` may be a numpy grid when the *caller* predicts (the loop's
        # ``predict(optimizer.current_phase)``); only a tensor can be the anchor call.
        if (getattr(phase, "requires_grad", False) is False
                and getattr(phase, "shape", None) == anchor_input.shape
                and bool((phase == anchor_input).all())):
            self.anchor_calls += 1
        return self._inner.predict(phase, **kwargs)


def _small_forward() -> tuple[FrozenDefocusForward, FrozenDefocusParameters]:
    """Small CPU-friendly frozen forward (32 grid / FFT 64 / 13x13 camera window)."""
    params = FrozenDefocusParameters(
        defocus_diopter=-0.1, origin_xy=(1360.0, 1020.0),
        amplitude_center_xy=(960.0, 600.0), amplitude_width_xy=(400.0, 300.0),
        residual_noll_radians={5: 0.3}, gain=25.0, background=1.0, sigma_ccd_px=2.0,
        ramp_matrix=((8000.0, 0.0), (0.0, -8000.0)),
        fft_size=64, pixel_samples=1, grid_size=32,
    )
    offsets = np.arange(-12, 13, 2.0)
    xx, yy = np.meshgrid(1360.0 + offsets, 1020.0 + offsets, indexing="xy")
    camera_xy = np.stack([xx, yy], axis=-1)
    return FrozenDefocusForward(params, camera_xy, device="cpu"), params


def _small_masks() -> tuple[np.ndarray, np.ndarray]:
    box = np.zeros((13, 13), dtype=bool)
    box[2:11, 2:11] = True
    core = np.zeros((13, 13), dtype=bool)
    core[4:9, 4:9] = True
    return box, core


def test_anchor_prediction_is_cached_between_accepts() -> None:
    """The anchor forward is recomputed only when ``accept()`` moves its input.

    ``_accepted`` is written by :meth:`accept` alone and the frozen forward is a
    deterministic pure function, so a step without an accept asks for exactly the
    same tensor it asked for last step: the cache must serve it, and the numbers
    must stay bit-identical.
    """
    import torch

    fwd, _ = _small_forward()
    box, core = _small_masks()
    objective = UniformityObjective(box, core, w_phase=0.0)
    rng = np.random.default_rng(7)
    seed = (rng.standard_normal((32, 32)) * 0.05).astype(np.float32)

    opt = DifferentiableUniformityOptimizer(fwd, objective, seed, device="cpu",
                                            lr=0.02, step_rms_cap=0.1)
    counter = _CountingAnchorForward(fwd, opt)
    opt.forward = counter
    # The first step has no cached anchor, so it is the reference for "one call".
    opt.update()
    assert counter.calls == 3 and counter.anchor_calls == 1
    first_anchor = opt._anchor_pred

    counter.calls = counter.anchor_calls = 0
    opt.update()
    opt.update()
    # Two more steps without an accept: the anchor input never moved, so neither
    # did the tensor -- 2 predicts per step (main + candidate), 0 anchor recompute.
    assert counter.calls == 4 and counter.anchor_calls == 0
    assert opt._anchor_pred is first_anchor

    # An accept moves ``_accepted``: the cache must be dropped and rebuilt.
    counter.calls = counter.anchor_calls = 0
    opt.accept()
    opt.update()
    assert counter.calls == 3 and counter.anchor_calls == 1  # anchor recomputed once
    assert not torch.equal(opt._anchor_pred, first_anchor)

    # Diagnostics still compare candidate against the cached anchor (bit-identical
    # to recomputing it), so test_measured_phase_regularization's identity holds.
    records = opt.loss_history
    assert records[0]["anchor_model_loss"] == records[1]["anchor_model_loss"]
    for record in records:
        assert record["candidate_model_loss_change"] == pytest.approx(
            record["candidate_model_loss"] - record["anchor_model_loss"], rel=1e-6)
    with torch.no_grad():
        reference = fwd.predict(opt._seed + opt._accepted)[0]
    assert torch.equal(opt._anchor_pred, reference)


def _anchored_loop(fwd, objective, seed, anchor, anchor_prediction, steps, *, reuse):
    """Replay ``validate.py``'s anchored block loop with the reuse switched on/off.

    Returns the optimizer, its counting forward, the per-step phase grids and the
    recorded history, so the two runs can be compared bit-for-bit. ``reuse=False``
    is the pre-change loop: it recomputes ``predict(current_phase)`` every step.
    """
    import torch

    from ao_shaping.algorithm.signal_processing.multishape_objective import anchored_intensity

    opt = DifferentiableUniformityOptimizer(fwd, objective, seed, device="cpu",
                                            lr=0.02, step_rms_cap=0.1)
    counter = _CountingAnchorForward(fwd, opt)
    opt.forward = counter
    reuse_prediction = anchor_prediction if reuse else None
    phases: list[np.ndarray] = []
    history: list[dict[str, Any]] = []
    for _ in range(steps):
        if reuse_prediction is None:
            # ``validate.py`` predicts on the same forward object the optimizer holds,
            # so this call is counted too -- it is the duplicate the reuse removes.
            with torch.no_grad():
                reuse_prediction = counter.predict(opt.current_phase)[0].detach().cpu().numpy()
        opt.update(anchored_intensity(anchor, reuse_prediction, anchor_prediction))
        reuse_prediction = opt.last_candidate_prediction if reuse else None
        phases.append(opt.current_phase.copy())
        history.append(dict(opt.loss_history[-1]))
    return opt, counter, phases, history


def test_candidate_prediction_matches_the_loop_recomputation_bit_for_bit() -> None:
    """``last_candidate_prediction`` is the very array ``validate.py`` recomputed."""
    import torch

    fwd, _ = _small_forward()
    box, core = _small_masks()
    objective = UniformityObjective(box, core, w_phase=0.0)
    rng = np.random.default_rng(11)
    seed = (rng.standard_normal((32, 32)) * 0.05).astype(np.float32)

    opt = DifferentiableUniformityOptimizer(fwd, objective, seed, device="cpu",
                                            lr=0.02, step_rms_cap=0.1)
    assert opt.last_candidate_prediction is None, "nothing was predicted before the first update"
    opt.update()
    cached = opt.last_candidate_prediction
    assert cached is not None
    with torch.no_grad():
        fresh = fwd.predict(opt.current_phase)[0].detach().cpu().numpy()
    assert cached.shape == fresh.shape
    assert cached.dtype == fresh.dtype == np.float32
    assert cached.tobytes() == fresh.tobytes(), "reuse must be bit-identical to the recomputation"


def test_anchored_loop_reuse_is_bitwise_identical_and_saves_one_forward_per_step() -> None:
    """The deployment loop's phases and diagnostics do not move; forwards go 3 -> 2 per step."""
    import torch

    fwd, _ = _small_forward()
    box, core = _small_masks()
    objective = UniformityObjective(box, core, w_phase=0.0)
    rng = np.random.default_rng(7)
    seed = (rng.standard_normal((32, 32)) * 0.05).astype(np.float32)
    with torch.no_grad():
        anchor_prediction = fwd.predict(seed)[0].cpu().numpy()
    #: Stand-in for the measured anchor frame: same grid, a slightly different level.
    anchor = (anchor_prediction * 1.07).astype(np.float32)
    steps = 6

    _, reusing, phases_reuse, history_reuse = _anchored_loop(
        fwd, objective, seed, anchor, anchor_prediction, steps, reuse=True)
    _, recomputing, phases_fresh, history_fresh = _anchored_loop(
        fwd, objective, seed, anchor, anchor_prediction, steps, reuse=False)

    for index, (reused, fresh) in enumerate(zip(phases_reuse, phases_fresh)):
        assert reused.tobytes() == fresh.tobytes(), f"phase grid at step {index} differs"
    assert history_reuse == history_fresh, "recorded diagnostics differ"
    # Anchored mode pays, per step: the grad forward + the post-step candidate forward.
    # The anchor is paid once either way (a block never accepts, so that cache stays
    # hot); without reuse the loop recomputes the candidate once more per step.
    assert reusing.anchor_calls == 1
    assert reusing.calls == 2 * steps + 1, f"expected 2 forwards/step with reuse, got {reusing.calls}"
    assert recomputing.calls == 3 * steps + 1, f"expected 3/step without reuse, got {recomputing.calls}"


def test_accept_does_not_invalidate_the_candidate_prediction() -> None:
    """``accept()`` moves the anchor reference, not the post-step candidate forward."""
    fwd, _ = _small_forward()
    box, core = _small_masks()
    objective = UniformityObjective(box, core, w_phase=0.0)
    seed = np.zeros((32, 32), dtype=np.float32)
    opt = DifferentiableUniformityOptimizer(fwd, objective, seed, device="cpu",
                                            lr=0.02, step_rms_cap=0.1)
    opt.update()
    before = opt.last_candidate_prediction
    assert before is not None
    opt.accept()
    after = opt.last_candidate_prediction
    assert after is not None and after.tobytes() == before.tobytes()
    assert opt._anchor_pred is None, "accept() must still drop the anchor cache"


def test_candidate_prediction_is_absent_when_focus_or_carrier_is_set() -> None:
    """With ``phi_focus``/``phi_carrier`` the caller's input is *not* ``current_phase``.

    ``current_phase`` folds both components into the grid, so a caller that predicts
    ``current_phase`` without passing them computes a different forward than the one
    ``update()`` made on ``seed + delta``. The accessor must hand back ``None`` there so
    the caller falls back to its own forward instead of reusing the wrong array.
    """
    fwd, params = _small_forward()
    box, core = _small_masks()
    objective = UniformityObjective(box, core, w_phase=0.0)
    seed = np.zeros((32, 32), dtype=np.float32)
    focus = np.full((params.grid_size, params.grid_size), 0.25, dtype=np.float32)
    assert focus.shape == seed.shape
    opt = DifferentiableUniformityOptimizer(fwd, objective, seed, device="cpu", phi_focus=focus,
                                            lr=0.02, step_rms_cap=0.1)
    opt.update()
    assert opt.last_candidate_prediction is None, "focus/carrier makes current_phase a different input"
    # The anchored loop's own reuse stays correct: it feeds predict(current_phase) instead.
    assert opt.current_phase.shape == seed.shape
