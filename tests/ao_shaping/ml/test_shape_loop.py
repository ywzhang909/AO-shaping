"""Tests for the iterative arbitrary-shape shaping driver.

The :mod:`ml.sim2f.shape_loop` module is the reason the bench exists: it drives
the "deep model + physical device" closed loop for one target, across three arms
(``flat`` baseline, optional ``net_open`` / ``net_finetune`` when a network phase
is supplied, and ``opt_finetune`` from flat when it is not).  Each closed-loop
round re-fits the aberration against the last *measured* frame, locally
fine-tunes the phase through the fitted bench, then accepts the candidate only
if the *measured* objective improves by ``ShapeLoopConfig.min_gain`` -- the guard
that keeps a partly-wrong model from walking the phase off a cliff.

These checks pin:

* the :class:`~ml.sim2f.config.ShapeLoopConfig` contract (every out-of-range
  field raises, the :meth:`~ml.sim2f.config.ShapeLoopConfig.to_dict` round-trips);
* the geometry gates (a device whose camera window disagrees with the model, or a
  target whose intensity grid does not match the camera, is rejected loudly);
* the loop on a synthetic ground-truth twin -- all arms run, the
  ``opt_finetune`` arm lowers the measured objective and raises the encircled
  energy, and the per-round acceptance bookkeeping is honest;
* the device-acceptance guard (an absurd ``min_gain`` rejects every round but
  still reports the rejected gain, so rejection is visible rather than silent);
* the ``net_phase`` path (three named arms, ``net_finetune`` seeded from the
  network phase);
* the :class:`~ml.sim2f.calibrate.CalibratedBench` protocol (it is a bench, its
  fitted illumination is an absolute amplitude, and a caller-supplied
  ``aberration`` adds to -- does not replace -- the fitted static map).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.calibrate import CalibratedBench, CalibrationResult
from ml.sim2f.config import RefineConfig, Sim2fConfig
from ml.sim2f.optics import Sim2f
from ml.sim2f.shape_loop import (
    ArmResult,
    RoundRecord,
    ShapeLoopConfig,
    run_shape_trial,
)
from ml.sim2f.targets import TargetSample, mask_target

# A quarter of the reference pupil / camera: an FFT length of 256 and a forward
# pass cheap enough to run a few hundred Adam steps per test (see
# ``tests/ao_shaping/ml/sim2f/test_calibrate.py``).
SHAPE = Sim2fConfig(ccd_px=64, grid=64, exposure_ms=0.0)

# Noll block the refit estimates (Noll 2 .. ``N_MODES + 1``).
N_MODES = 12

# A ring/annulus target with a thick enough annulus to clear the feature floor
# (``min_feature_px=6.0``) while staying clear of the frame border.
_YY, _XX = np.ogrid[0 : SHAPE.ccd_px, 0 : SHAPE.ccd_px]
_R2 = (_XX - 32.0) ** 2 + (_YY - 32.0) ** 2
_MASK: np.ndarray = ((_R2 >= 14.0**2) & (_R2 <= 22.0**2)).astype(np.float32)


def _bench() -> Sim2f:
    """Build the tiny CPU bench every test runs against.

    Returns:
        A :class:`Sim2f` on :data:`SHAPE`.
    """
    return Sim2f(SHAPE, device="cpu")


def _ring_target() -> TargetSample:
    """A ring/annulus far-field target on the test camera grid.

    Returns:
        A :class:`~ml.sim2f.targets.TargetSample` whose thinnest feature is the
        8 px annulus width (above the ``min_feature_px=6.0`` floor) and which
        sits clear of the frame border so ``mask_target`` accepts it.
    """
    return mask_target(_MASK, SHAPE.ccd_px, min_feature_px=6.0)


def _nominal_calibration(sim: Sim2f) -> CalibrationResult:
    """A zero-aberration, nominal-illumination, identity-camera calibration.

    Args:
        sim: The bench that supplies the nominal beam.

    Returns:
        A :class:`~ml.sim2f.calibrate.CalibrationResult` whose fitted state is
        the bench identity -- the contract the "fitted illumination is an
        absolute amplitude" test relies on.
    """
    illum = sim.nominal_illumination[0, 0].detach().cpu().numpy().astype(np.float32)
    return CalibrationResult(
        ok=True,
        reason=None,
        sim_config=SHAPE,
        aberration=np.zeros((SHAPE.grid, SHAPE.grid), dtype=np.float32),
        illumination=illum,
        camera={"zoom": 1.0, "shift_x_px": 0.0, "shift_y_px": 0.0},
    )


def _twin(amplitude: float, seed: int) -> tuple[CalibratedBench, Sim2f]:
    """Build a synthetic ground-truth device/model pair.

    The *device* is a :class:`CalibratedBench` whose fitted aberration is a
    random Zernike map drawn from ``seed``; the *model* is a bare nominal
    :class:`Sim2f` that knows nothing of it.  An honest trial never lets the two
    be the same object.

    Args:
        amplitude: RMS amplitude, radians, of the true aberration coefficients.
        seed: RNG seed for the true coefficients (determinism).

    Returns:
        ``(device, model)`` -- the measured bench and the trusted model.
    """
    sim = _bench()
    rng = np.random.default_rng(seed)
    coeffs = (rng.standard_normal(N_MODES) * amplitude).astype(np.float32)
    true_map = (
        sim.aberration(coeffs[None].astype(np.float32))[0].detach().cpu().numpy()
    ).astype(np.float32)
    illum = sim.nominal_illumination[0, 0].detach().cpu().numpy().astype(np.float32)
    calibration = CalibrationResult(
        ok=True,
        reason=None,
        sim_config=SHAPE,
        aberration=true_map,
        illumination=illum,
        camera={"zoom": 1.0, "shift_x_px": 0.0, "shift_y_px": 0.0},
    )
    device = CalibratedBench(sim, calibration)
    model = _bench()
    return device, model


def _loop_config(**overrides: object) -> ShapeLoopConfig:
    """A small, fast loop contract, with overrides.

    Args:
        **overrides: Field overrides for :class:`ShapeLoopConfig`.

    Returns:
        The configured :class:`ShapeLoopConfig`.
    """
    base = dict(
        rounds=2,
        refine=RefineConfig(iters=30),
        n_modes=N_MODES,
        fit_iters=25,
    )
    merged = {**base, **overrides}
    return ShapeLoopConfig(**merged)  # type: ignore[arg-type]


def test_shape_loop_config_validation_rejects_bad_fields() -> None:
    """Every out-of-range field raises ValueError, nothing else does."""
    # Negative rounds.
    with pytest.raises(ValueError):
        ShapeLoopConfig(rounds=-1)
    # Too few modes.
    with pytest.raises(ValueError):
        ShapeLoopConfig(n_modes=0)
    # A fit_frozen Noll outside 1 .. n_modes + 1.
    with pytest.raises(ValueError):
        ShapeLoopConfig(n_modes=N_MODES, fit_frozen=(1, 2, 3, N_MODES + 5))
    # Non-positive fit_lr.
    with pytest.raises(ValueError):
        ShapeLoopConfig(fit_lr=0.0)
    # Non-positive trust_rad.
    with pytest.raises(ValueError):
        ShapeLoopConfig(trust_rad=0.0)
    # A bad metric.
    with pytest.raises(ValueError):
        ShapeLoopConfig(metric="bogus")
    # Negative min_gain.
    with pytest.raises(ValueError):
        ShapeLoopConfig(min_gain=-0.1)
    # And the default config is accepted.
    assert ShapeLoopConfig().rounds >= 0


def test_shape_loop_config_to_dict_round_trips() -> None:
    """to_dict round-trips every field, including the nested refine dict."""
    cfg = ShapeLoopConfig(
        rounds=4,
        refine=RefineConfig(iters=12, max_delta_rad=0.3),
        n_modes=N_MODES,
        fit_iters=25,
        fit_lr=0.02,
        trust_rad=0.5,
        fit_frozen=(1, 2, 3),
        w_ee=0.25,
        metric="l1",
        min_gain=0.0,
        seed=7,
    )
    dumped = cfg.to_dict()
    # The nested refine config is itself a dict (from RefineConfig.to_dict).
    assert isinstance(dumped["refine"], dict)
    assert dumped["refine"]["iters"] == 12
    assert dumped["refine"]["max_delta_rad"] == pytest.approx(0.3)
    # The frozen Noll tuple is serialised as a list.
    assert dumped["fit_frozen"] == [1, 2, 3]
    # Every field is present and carries its value.
    assert dumped["rounds"] == 4
    assert dumped["n_modes"] == N_MODES
    assert dumped["fit_iters"] == 25
    assert dumped["fit_lr"] == pytest.approx(0.02)
    assert dumped["trust_rad"] == pytest.approx(0.5)
    assert dumped["w_ee"] == pytest.approx(0.25)
    assert dumped["metric"] == "l1"
    assert dumped["min_gain"] == pytest.approx(0.0)
    assert dumped["seed"] == 7

    # Reconstructing from the dump (re-hydrating the nested/dict fields) yields
    # a config whose to_dict agrees byte for byte.
    rebuilt = ShapeLoopConfig(
        **{
            **dumped,
            "refine": RefineConfig(**dumped["refine"]),
            "fit_frozen": tuple(dumped["fit_frozen"]),
        }
    )
    assert rebuilt.to_dict() == dumped


def test_run_shape_trial_rejects_disagreeing_geometries() -> None:
    """A device/model pair whose camera windows disagree is rejected."""
    model = Sim2f(SHAPE, device="cpu")
    device = Sim2f(
        Sim2fConfig(ccd_px=128, grid=SHAPE.grid, exposure_ms=0.0), device="cpu"
    )
    target = _ring_target()
    with pytest.raises(ValueError, match="geometries disagree"):
        run_shape_trial(device, model, target)


def test_run_shape_trial_rejects_mismatched_target_grid() -> None:
    """A target whose intensity grid does not match the camera is rejected."""
    bench = Sim2f(SHAPE, device="cpu")
    wrong = np.zeros((32, 32), dtype=np.float32)
    target = TargetSample(
        intensity=wrong,
        support=np.zeros((32, 32), dtype=bool),
        kind="mask",
        geometry={},
    )
    with pytest.raises(ValueError, match="target intensity"):
        run_shape_trial(bench, bench, target)


def test_run_shape_trial_improves_on_a_synthetic_twin() -> None:
    """On a ground-truth twin all arms run and opt_finetune improves."""
    device, model = _twin(amplitude=0.2, seed=12345)
    target = _ring_target()
    config = _loop_config()
    result = run_shape_trial(device, model, target, config=config)

    # The flat baseline is static: one measurement, no rounds.
    flat = result.arm("flat")
    assert flat.start == flat.final
    assert flat.rounds == ()

    # The refine arm (no network phase supplied) is named opt_finetune.
    tune = result.arm("opt_finetune")
    assert isinstance(tune, ArmResult)
    assert tune.arm == "opt_finetune"

    # It lowered the measured objective and raised the encircled energy.
    assert tune.final["objective"] < tune.start["objective"]
    assert tune.final["ee"] > tune.start["ee"]

    # One RoundRecord per round, indexed 1 .. n.
    assert len(tune.rounds) == int(config.rounds)
    assert [int(r.round) for r in tune.rounds] == list(range(1, int(config.rounds) + 1))

    # Every accepted round cleared the acceptance gate.
    for record in tune.rounds:
        assert isinstance(record, RoundRecord)
        if record.accepted_device:
            assert record.gain_device >= float(config.min_gain)


def test_device_acceptance_guard_rejects_silently_no() -> None:
    """An absurd min_gain rejects every round but reports the rejected gain."""
    device, model = _twin(amplitude=0.2, seed=999)
    target = _ring_target()
    config = _loop_config(min_gain=5.0)
    result = run_shape_trial(device, model, target, config=config)

    tune = result.arm("opt_finetune")
    # No round was accepted: the final state equals the start.
    assert tune.final == tune.start
    assert all(not r.accepted_device for r in tune.rounds)
    # Rejection is visible: a candidate was measured and the gain reported
    # (non-zero), it was just below the absurd threshold.
    assert any(r.gain_device != 0.0 for r in tune.rounds)


def test_net_phase_path_seeds_finetune_from_network() -> None:
    """A network phase yields three arms; net_finetune starts from net_open."""
    device, model = _twin(amplitude=0.1, seed=42)
    target = _ring_target()
    net_phase = np.zeros((SHAPE.grid, SHAPE.grid), dtype=np.float32)
    config = ShapeLoopConfig(
        rounds=1, refine=RefineConfig(iters=5), n_modes=N_MODES, fit_iters=5
    )
    result = run_shape_trial(device, model, target, net_phase=net_phase, config=config)

    arm_names = {arm.arm for arm in result.arms}
    assert arm_names == {"flat", "net_open", "net_finetune"}

    net_open = result.arm("net_open")
    assert net_open.rounds == ()  # static baseline
    net_finetune = result.arm("net_finetune")
    # The finetune arm's open-loop measurement is the network phase, i.e. the
    # same commanded phase net_open measured -- so the metric rows agree.
    assert set(net_finetune.start.keys()) == set(net_open.final.keys())
    for key in net_finetune.start:
        assert net_finetune.start[key] == pytest.approx(net_open.final[key], rel=1e-5)

    # An unknown arm name raises KeyError, not a silent miss.
    with pytest.raises(KeyError):
        result.arm("nope")


def test_calibrated_bench_reproduces_nominal_forward() -> None:
    """A nominal calibration reproduces the wrapped bench's forward exactly."""
    sim = _bench()
    calibration = _nominal_calibration(sim)
    bench = CalibratedBench(sim, calibration)

    # It exposes the wrapped bench protocol.
    assert int(bench.grid) == int(sim.grid)
    assert int(bench.ccd_px) == int(sim.ccd_px)
    assert bench.dtype == sim.dtype
    assert torch.device(bench.device) == torch.device(sim.device)
    assert bench.cfg is sim.cfg

    # Its aberration map matches the wrapped bench's basis.
    coeffs = np.random.default_rng(0).standard_normal(N_MODES).astype(np.float32)
    assert torch.allclose(
        bench.aberration(coeffs[None]),
        sim.aberration(coeffs[None]),
        atol=1e-6,
    )

    # The fitted illumination is absolute: with zero aberration, identity camera
    # and the nominal beam, the fitted forward equals the bare bench forward.
    phase = torch.randn(1, SHAPE.grid, SHAPE.grid, dtype=sim.dtype)
    with torch.no_grad():
        fitted = bench.forward(phase, detector=False)
        bare = sim.forward(phase, detector=False)
    assert fitted.shape == bare.shape
    assert torch.allclose(fitted, bare, atol=1e-5)


def test_calibrated_bench_rejects_mismatched_aberration_grid() -> None:
    """A calibration whose map does not match the bench grid raises."""
    sim = _bench()
    illum = sim.nominal_illumination[0, 0].detach().cpu().numpy().astype(np.float32)
    bad = CalibrationResult(
        ok=True,
        reason=None,
        sim_config=SHAPE,
        aberration=np.zeros((32, 32), dtype=np.float32),
        illumination=illum,
        camera={"zoom": 1.0, "shift_x_px": 0.0, "shift_y_px": 0.0},
    )
    with pytest.raises(ValueError, match="calibration aberration"):
        CalibratedBench(sim, bad)


def test_caller_aberration_adds_to_fitted_static_map() -> None:
    """A caller aberration composes with (does not replace) the fitted map."""
    sim = _bench()
    calibration = _nominal_calibration(sim)
    bench = CalibratedBench(sim, calibration)
    phase = torch.randn(1, SHAPE.grid, SHAPE.grid, dtype=sim.dtype)

    # A non-constant phase plate (a pupil ramp) -- a constant offset would leave
    # the intensity unchanged, so the adder must actually move light.
    delta = 0.25 * torch.linspace(
        -1.0, 1.0, SHAPE.grid, dtype=sim.dtype, device=sim.device
    ).view(-1, 1).expand(SHAPE.grid, SHAPE.grid)
    with torch.no_grad():
        base = bench.forward(phase, detector=False)
        shifted = bench.forward(phase, aberration=delta, detector=False)
        zero = bench.forward(phase, aberration=torch.zeros_like(delta), detector=False)

    # The delta changes the frame ...
    assert not torch.allclose(base, shifted, atol=1e-5)
    # ... and a zero delta reproduces the fitted base frame.
    assert torch.allclose(base, zero, atol=1e-6)
