"""Tests for the sim-to-real bench-state calibrator.

A bench-state fit is only useful if it recovers the state it was measured from,
and only safe if it refuses to be applied when it has not generalised.  These
checks pin both halves, plus the two contracts the fit depends on:

* a synthetic session whose ground truth is known must come back out of the fit
  -- beam widths, camera zoom and shift -- and the holdout it never saw must be
  predicted better after the fit than before it;
* a session recorded at the nominal state must fit back to the nominal state,
  and the same session *with* a holdout must be rejected, because a noiseless
  nominal fit cannot improve on exact and the gate is a strict comparison;
* :func:`ml.sim2f.calibrate.load_bench_session` must read the recorder's own
  on-disk contract (CSV + sidecars, ``slm_zernike`` lists, group filtering) and
  fail loudly on a missing session or on a group nothing matches;
* the bench geometry and the session geometry must be checked against each other
  rather than broadcast into a nonsense fit;
* :func:`ml.sim2f.calibrate.save_calibration` /
  :func:`ml.sim2f.calibrate.load_calibration` must round-trip the fitted state.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f import calibrate
from ml.sim2f.calibrate import (
    BenchSample,
    BenchSession,
    CalibratedBench,
    _beam_amplitude,
    _camera_affine,
    calibrate_bench,
    load_bench_session,
    load_calibration,
    save_calibration,
)
from ml.sim2f.config import CalibrateConfig, OpticsErrors, Sim2fConfig
from ml.sim2f.measurement import MeasurementTransfer, apply_transfer
from ml.sim2f.optics import Sim2f

#: One quarter of the reference pupil grid and camera: an FFT length of 256 and
#: a forward pass cheap enough to run a few hundred Adam steps per test.
SHAPE = Sim2fConfig(grid=64, ccd_px=64, exposure_ms=0.0)

#: Noll 2..8 -- the coefficient block ``CalibrateConfig(zernike_n_max=8)`` fits.
N_COEFFS = 7

#: The bench state the recovery test has to find (Noll 4..8 at ``COEFF_RAD``).
TRUE_BEAM = {"width_scale_x": 1.25, "width_scale_y": 1.0, "angle_rad": 0.2}
TRUE_CAMERA = {"zoom": 1.3, "shift_x_px": 4.0, "shift_y_px": -6.0}
TRUE_COEFF_RAD = 0.15

#: The nominal bench: every fitted knob at its identity.
NOMINAL_BEAM = {"width_scale_x": 1.0, "width_scale_y": 1.0, "angle_rad": 0.0}
NOMINAL_CAMERA = {"zoom": 1.0, "shift_x_px": 0.0, "shift_y_px": 0.0}

#: The error model the fake on-disk session records, as ``session.json`` holds
#: it (Zernike pairs are JSON lists) and as it must come back out (tuples).
RECORDED_ERRORS = dataclasses.replace(SHAPE.errors, slm_zernike=((4, 0.05),))


def _bench() -> Sim2f:
    """Build the tiny CPU bench every test fits against.

    Returns:
        A :class:`Sim2f` on :data:`SHAPE`.
    """
    return Sim2f(SHAPE, device="cpu")


def _scalar(sim: Sim2f, value: float) -> torch.Tensor:
    """Lift one Python float onto the bench's device and dtype.

    Args:
        sim: The bench (supplies device and dtype).
        value: The scalar.

    Returns:
        A 0-d tensor.
    """
    return torch.as_tensor(float(value), dtype=sim.dtype, device=sim.device)


def _render(
    sim: Sim2f,
    n: int = 8,
    *,
    seed: int = 3,
    beam: dict[str, float] = TRUE_BEAM,
    camera: dict[str, float] = TRUE_CAMERA,
    coeff_rad: float = TRUE_COEFF_RAD,
) -> tuple[BenchSample, ...]:
    """Render ``n`` (command, frame) pairs from a known bench state.

    The synthesiser reuses the calibrator's own beam and camera helpers: what
    the tests pin is the *recovery* of a state, not how the state is drawn.

    Args:
        sim: The bench the frames are rendered through.
        n: Number of samples.
        seed: Seed of the commanded Zernike coefficients.
        beam: Ground-truth beam widths and angle.
        camera: Ground-truth camera zoom and shift, camera pixels.
        coeff_rad: Ground-truth aberration, radians on each of Noll 4..8.

    Returns:
        One :class:`BenchSample` per command, in order.
    """
    rng = np.random.default_rng(seed)
    commands = rng.normal(0.0, 0.5, size=(n, N_COEFFS)).astype(np.float32)
    phases = sim.aberration(torch.as_tensor(commands))
    illumination = _beam_amplitude(
        sim,
        _scalar(sim, math.log(beam["width_scale_x"])),
        _scalar(sim, math.log(beam["width_scale_y"])),
        _scalar(sim, beam["angle_rad"]),
        _scalar(sim, 0.0),
        _scalar(sim, 0.0),
    )
    coefficients = torch.zeros((1, N_COEFFS), dtype=sim.dtype, device=sim.device)
    coefficients[0, 2:] = coeff_rad
    intensity = sim.forward(
        phases,
        aberration=sim.aberration(coefficients),
        illumination=illumination,
        detector=False,
    )
    frames = _camera_affine(
        intensity,
        _scalar(sim, camera["shift_x_px"]),
        _scalar(sim, camera["shift_y_px"]),
        _scalar(sim, math.log(camera["zoom"])),
    )
    return tuple(
        BenchSample(
            phase=phases[index].detach().cpu().numpy().astype(np.float32),
            frame=frames[index, 0].detach().cpu().numpy().astype(np.float32),
            label=f"s{index}/a1",
        )
        for index in range(n)
    )


def _session(sim: Sim2f, n: int = 8, **kwargs: object) -> BenchSession:
    """Wrap :func:`_render` in a :class:`BenchSession` rooted at the CWD.

    Args:
        sim: The bench the frames are rendered through.
        n: Number of samples.
        kwargs: Ground-truth overrides for :func:`_render`.

    Returns:
        The session, as a twin session would hand it to the fit.
    """
    return BenchSession(
        root=Path("."),
        samples=_render(sim, n, **kwargs),  # type: ignore[arg-type]
        sim_config=SHAPE,
        backend="sim",
        note="twin run",
    )


def _write_session(
    root: Path,
    *,
    conditions: tuple[str, ...] = ("a1", "b1", "b2", "a2"),
    n_samples: int = 2,
) -> Path:
    """Write a session directory shaped like :mod:`ml.sim2f.bench`'s contract.

    Args:
        root: Destination directory (created).
        conditions: Conditions written per sample, in recording order.
        n_samples: Number of samples (rows = ``n_samples`` x ``conditions``).

    Returns:
        ``root``.
    """
    root.mkdir(parents=True, exist_ok=True)
    bench = dict(SHAPE.to_dict())
    # ``session.json`` is JSON, so the Zernike pairs are lists there; the loader
    # has to coerce them back to tuples for the frozen error model.
    bench["errors"] = {**bench["errors"], "slm_zernike": [[4, 0.05]]}
    (root / "session.json").write_text(
        json.dumps({"backend": "sim", "notes": "twin run", "bench": bench}),
        encoding="utf-8",
    )
    table_dir = root / "process_data"
    sidecars = table_dir / "bench_session_sidecars"
    sidecars.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for sample in range(n_samples):
        for condition in conditions:
            index = len(rows)
            np.save(
                sidecars / f"bench_session__c_{index:04d}.npy",
                rng.random((int(SHAPE.grid), int(SHAPE.grid)), dtype=np.float32),
            )
            np.save(
                sidecars / f"bench_session__img_{index:04d}.npy",
                rng.random((int(SHAPE.ccd_px), int(SHAPE.ccd_px)), dtype=np.float32),
            )
            rows.append(
                {"_id": index, "index": sample, "condition": condition, "kind": "frame"}
            )
    with (table_dir / "bench_session.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["_id", "index", "condition", "kind"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_calibration_recovers_a_perturbed_bench_state() -> None:
    """A twin session recovers its own beam, camera and aberration, and generalises."""
    sim = _bench()
    session = _session(sim)

    result = calibrate_bench(
        sim, session, CalibrateConfig(iters=250, holdout_frac=0.25, seed=0)
    )

    assert result.ok is True
    assert result.reason is None
    assert (result.n_train, result.n_holdout) == (6, 2)
    assert result.score["holdout"].l1 < result.score_start["holdout"].l1
    assert result.camera["zoom"] == pytest.approx(TRUE_CAMERA["zoom"], rel=0.1)
    assert result.camera["shift_x_px"] == pytest.approx(
        TRUE_CAMERA["shift_x_px"], abs=2.0
    )
    assert result.camera["shift_y_px"] == pytest.approx(
        TRUE_CAMERA["shift_y_px"], abs=2.0
    )
    assert result.beam["width_scale_x"] == pytest.approx(
        TRUE_BEAM["width_scale_x"], rel=0.15
    )
    assert result.beam["width_scale_y"] == pytest.approx(
        TRUE_BEAM["width_scale_y"], rel=0.15
    )
    assert float(np.sqrt(np.mean(np.square(result.aberration)))) > 0.1


def test_nominal_session_fits_back_to_the_nominal_state() -> None:
    """A session recorded at the nominal state leaves every fitted knob at nominal."""
    sim = _bench()
    session = _session(sim, beam=NOMINAL_BEAM, camera=NOMINAL_CAMERA, coeff_rad=0.0)

    # No holdout: a noiseless nominal session starts at a zero residual, so any
    # holdout split can only get marginally worse and the strict gate would
    # reject a fit that moved nothing (pinned by the next test).
    result = calibrate_bench(
        sim, session, CalibrateConfig(iters=250, holdout_frac=0.0, seed=0)
    )

    assert result.ok is True
    assert result.camera["zoom"] == pytest.approx(1.0, abs=0.05)
    assert result.camera["shift_x_px"] == pytest.approx(0.0, abs=1.0)
    assert result.camera["shift_y_px"] == pytest.approx(0.0, abs=1.0)
    assert result.beam["width_scale_x"] == pytest.approx(1.0, abs=0.1)
    assert result.beam["width_scale_y"] == pytest.approx(1.0, abs=0.1)


def test_gate_rejects_a_fit_that_does_not_improve_the_holdout() -> None:
    """A fit that leaves the held-out frames worse returns the nominal state, not a guess."""
    sim = _bench()
    session = _session(sim, beam=NOMINAL_BEAM, camera=NOMINAL_CAMERA, coeff_rad=0.0)

    result = calibrate_bench(
        sim, session, CalibrateConfig(iters=250, holdout_frac=0.25, seed=0)
    )

    assert result.ok is False
    assert isinstance(result.reason, str) and result.reason != ""
    assert result.score["holdout"].l1 > result.score_start["holdout"].l1
    assert np.all(result.aberration == 0.0)
    assert result.camera["zoom"] == pytest.approx(1.0)
    assert result.camera["shift_x_px"] == pytest.approx(0.0)
    assert result.camera["shift_y_px"] == pytest.approx(0.0)


def test_mismatched_session_grid_is_rejected_before_fitting() -> None:
    """A session whose commands are not on the bench's grid fails instead of broadcasting."""
    sim = _bench()
    samples = tuple(
        BenchSample(
            phase=np.zeros((32, 32), dtype=np.float32),
            frame=np.zeros((int(SHAPE.ccd_px), int(SHAPE.ccd_px)), dtype=np.float32),
            label=f"s{index}/a1",
        )
        for index in range(8)
    )
    session = BenchSession(
        root=Path("."), samples=samples, sim_config=SHAPE, backend="sim"
    )

    with pytest.raises(ValueError, match="bench grid"):
        calibrate_bench(
            sim, session, CalibrateConfig(iters=10, holdout_frac=0.25, seed=0)
        )


def test_load_bench_session_reads_the_recorder_contract(tmp_path: Path) -> None:
    """The loader reads session.json + CSV + sidecars and rebuilds the bench config."""
    root = _write_session(tmp_path / "session")

    session = load_bench_session(root)

    assert len(session.samples) == 8
    assert session.backend == "sim"
    assert session.note == "twin run"
    assert session.root == root
    assert session.sim_config == dataclasses.replace(SHAPE, errors=RECORDED_ERRORS)
    assert session.sim_config.errors.slm_zernike == ((4, 0.05),)
    assert [sample.label for sample in session.samples] == [
        "s0/a1",
        "s0/b1",
        "s0/b2",
        "s0/a2",
        "s1/a1",
        "s1/b1",
        "s1/b2",
        "s1/a2",
    ]
    for sample in session.samples:
        assert sample.phase.shape == (int(SHAPE.grid), int(SHAPE.grid))
        assert sample.frame.shape == (int(SHAPE.ccd_px), int(SHAPE.ccd_px))
        assert sample.phase.dtype == np.float32
        assert sample.frame.dtype == np.float32
    assert session.phases().shape == (8, int(SHAPE.grid), int(SHAPE.grid))
    assert session.frames().shape == (8, int(SHAPE.ccd_px), int(SHAPE.ccd_px))


def test_load_bench_session_keeps_only_the_requested_group(tmp_path: Path) -> None:
    """``groups=("flat",)`` keeps the A pair and drops the network's B pair."""
    root = _write_session(tmp_path / "session")

    session = load_bench_session(root, groups=("flat",))

    assert len(session.samples) == 4
    assert {sample.label.split("/")[1] for sample in session.samples} == {"a1", "a2"}


def test_load_bench_session_rejects_a_missing_session(tmp_path: Path) -> None:
    """A directory with no session.json is a missing session, not an empty one."""
    with pytest.raises(FileNotFoundError, match="session.json"):
        load_bench_session(tmp_path / "nothing")


def test_load_bench_session_rejects_a_group_nothing_matches(tmp_path: Path) -> None:
    """Asking for the A pair of a session that only recorded the B pair is an error."""
    root = _write_session(tmp_path / "session", conditions=("b1", "b2"))

    with pytest.raises(ValueError, match="no rows matched"):
        load_bench_session(root, groups=("flat",))


def test_save_and_load_calibration_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved calibration comes back with its state, its scores and its arrays."""
    sim = _bench()
    session = _session(sim)
    result = calibrate_bench(
        sim, session, CalibrateConfig(iters=30, holdout_frac=0.25, seed=0)
    )

    json_path, npz_path = save_calibration(tmp_path, result)

    assert json_path == tmp_path / "calibration.json"
    assert npz_path == tmp_path / "calibration.npz"
    loaded = load_calibration(tmp_path)

    assert loaded.ok == result.ok
    assert loaded.reason == result.reason
    assert loaded.sim_config == result.sim_config
    assert loaded.beam == pytest.approx(result.beam)
    assert loaded.camera == pytest.approx(result.camera)
    assert loaded.measurement == result.measurement
    assert loaded.zero_order == result.zero_order
    assert loaded.free == result.free
    assert (loaded.n_train, loaded.n_holdout) == (result.n_train, result.n_holdout)
    assert loaded.iterations == result.iterations
    assert loaded.score.keys() == result.score.keys()
    assert loaded.score_start.keys() == result.score_start.keys()
    for name in result.score:
        assert dataclasses.asdict(loaded.score[name]) == pytest.approx(
            dataclasses.asdict(result.score[name])
        )
    np.testing.assert_allclose(loaded.aberration, result.aberration, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.illumination, result.illumination, rtol=0, atol=0)


def test_a_calibration_written_before_the_measurement_stage_loads_as_the_identity(
    tmp_path: Path,
) -> None:
    """Every artifact on disk predates the ``measurement`` block; none may fail to load."""
    sim = _bench()
    result = calibrate_bench(
        sim, _session(sim), CalibrateConfig(iters=20, holdout_frac=0.25, seed=0)
    )
    json_path, _ = save_calibration(tmp_path, result)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload.pop("measurement", None)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    loaded = load_calibration(tmp_path)

    assert loaded.measurement == MeasurementTransfer().to_dict()


def test_fit_recovers_a_synthetic_detector_mtf(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second stage recovers a post-affine detector MTF with geometry frozen.

    Camera initialisation is suppressed here on purpose: this is a test of the
    sequential MTF stage, not of how the preceding geometry stage trades beam
    width, zoom and sigma against one another.
    """
    monkeypatch.setattr(calibrate, "_init_camera", lambda *args, **kwargs: None)
    sim = _bench()
    truth = MeasurementTransfer(psf_sigma_px=1.5)
    samples = tuple(
        dataclasses.replace(
            sample,
            frame=truth.apply(
                torch.as_tensor(sample.frame, dtype=sim.dtype).view(
                    1, 1, *sample.frame.shape
                )
            )[0, 0]
            .numpy()
            .astype(np.float32),
        )
        for sample in _render(
            sim, beam=NOMINAL_BEAM, camera=NOMINAL_CAMERA, coeff_rad=0.0
        )
    )
    session = BenchSession(
        root=Path("."), samples=samples, sim_config=SHAPE, backend="sim"
    )

    result = calibrate_bench(
        sim,
        session,
        CalibrateConfig(
            iters=300,
            holdout_frac=0.25,
            seed=0,
            fit_zernike=False,
            fit_beam=False,
            fit_center=False,
            fit_scale=False,
            fit_measurement=True,
        ),
    )

    assert result.ok is True
    assert result.beam["width_scale_x"] == pytest.approx(1.0)
    assert result.camera["zoom"] == pytest.approx(1.0)
    fitted = MeasurementTransfer.from_dict(result.measurement)
    assert fitted.psf_sigma_px == pytest.approx(truth.psf_sigma_px, abs=0.25)
    # The pedestal is structural: it is not fitted unless its separate,
    # experimental flag is enabled, because it is degenerate with fitted gain.
    assert fitted.background_fraction == 0.0
    assert "background_fraction" not in result.free
    assert result.score["holdout"].l1 < result.score_start["holdout"].l1


def test_the_measurement_stage_is_off_unless_it_is_asked_for() -> None:
    """The default fit must leave the transfer at the identity, as before it existed."""
    sim = _bench()
    result = calibrate_bench(
        sim, _session(sim), CalibrateConfig(iters=20, holdout_frac=0.25, seed=0)
    )

    assert result.measurement == MeasurementTransfer().to_dict()
    assert "psf_sigma_px" not in result.free


def _frozen_geometry(**kwargs: object) -> CalibrateConfig:
    """Only the post-affine stages are free; the geometry init is out of scope."""
    return CalibrateConfig(
        iters=int(kwargs.pop("iters", 300)),
        lr=float(kwargs.pop("lr", 0.05)),
        holdout_frac=0.25,
        seed=0,
        fit_zernike=False,
        fit_beam=False,
        fit_center=False,
        fit_scale=False,
        **kwargs,  # type: ignore[arg-type]
    )


def test_fit_recovers_a_synthetic_zero_order_eta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device with a coherent zero order gets its eta back, and generalises."""
    truth_eta = 0.3
    device = Sim2f(
        dataclasses.replace(SHAPE, errors=OpticsErrors(zero_order_eta=truth_eta)),
        device="cpu",
    )
    session = _session(
        device, beam=NOMINAL_BEAM, camera=NOMINAL_CAMERA, coeff_rad=0.0
    )
    nominal = _bench()
    # The camera init is geometry, and this test freezes geometry: without this
    # the estimator would read the zero-order spot as a zoom/shift error.
    monkeypatch.setattr(calibrate, "_init_camera", lambda *args, **kwargs: None)

    result = calibrate_bench(
        nominal, session, _frozen_geometry(fit_zero_order=True)
    )

    assert result.ok, result.reason
    assert result.zero_order["eta"] == pytest.approx(truth_eta, abs=0.03)
    assert result.score["holdout"].l1 < result.score_start["holdout"].l1
    assert result.score["holdout"].ncc > result.score_start["holdout"].ncc

    save_calibration(tmp_path, result)
    loaded = load_calibration(tmp_path)
    assert loaded.zero_order == pytest.approx(result.zero_order)
    assert loaded.config.fit_zero_order


def test_fit_recovers_measurement_and_zero_order_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MTF then eta, each alone: both come back and the holdout improves."""
    truth_eta = 0.25
    device = Sim2f(
        dataclasses.replace(SHAPE, errors=OpticsErrors(zero_order_eta=truth_eta)),
        device="cpu",
    )
    session = _session(
        device, beam=NOMINAL_BEAM, camera=NOMINAL_CAMERA, coeff_rad=0.0
    )
    samples = [
        BenchSample(
            phase=sample.phase,
            frame=apply_transfer(
                torch.as_tensor(sample.frame)[None, None], 0.8, 0.0
            )[0, 0].numpy(),
            label=sample.label,
        )
        for sample in session.samples
    ]
    blurred = dataclasses.replace(session, samples=samples)
    monkeypatch.setattr(calibrate, "_init_camera", lambda *args, **kwargs: None)

    result = calibrate_bench(
        _bench(),
        blurred,
        _frozen_geometry(fit_measurement=True, fit_zero_order=True),
    )

    assert result.ok, result.reason
    assert result.measurement["psf_sigma_px"] == pytest.approx(0.8, abs=0.4)
    assert result.zero_order["eta"] == pytest.approx(truth_eta, abs=0.05)
    assert result.score["holdout"].l1 < result.score_start["holdout"].l1
    assert result.iterations == 3 * 300


def test_the_zero_order_stage_is_off_unless_it_is_asked_for() -> None:
    """Default: no fitted eta, so the wrapped bench's nominal one applies."""
    sim = _bench()
    result = calibrate_bench(
        sim, _session(sim), CalibrateConfig(iters=20, holdout_frac=0.25, seed=0)
    )

    assert result.zero_order == {}
    assert "zero_order_eta" not in result.free


def test_a_calibration_without_zero_order_uses_the_benchs_nominal_eta(
    tmp_path: Path,
) -> None:
    """Old files carry no eta: the wrapper must not silently set it to zero."""
    sim = _bench()
    nominal_session = _session(
        sim, beam=NOMINAL_BEAM, camera=NOMINAL_CAMERA, coeff_rad=0.0
    )
    result = calibrate_bench(
        sim, nominal_session, CalibrateConfig(iters=20, holdout_frac=0.25, seed=0)
    )
    assert result.zero_order == {}
    save_calibration(tmp_path, result)
    payload = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    payload.pop("zero_order")
    (tmp_path / "calibration.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_calibration(tmp_path)
    assert loaded.zero_order == {}

    phase = np.zeros((1, SHAPE.grid, SHAPE.grid), dtype=np.float32)
    bench = Sim2f(
        dataclasses.replace(SHAPE, errors=OpticsErrors(zero_order_eta=0.2)),
        device="cpu",
    )
    wrapped = CalibratedBench(bench, loaded)
    explicit = dataclasses.replace(loaded, zero_order={"eta": 0.2})
    # Empty state == the wrapped bench's own nominal eta, not a zeroed field:
    # an explicit eta equal to it is the same frame, a different one is not.
    assert torch.allclose(
        wrapped.forward(phase), CalibratedBench(bench, explicit).forward(phase)
    )
    other = dataclasses.replace(loaded, zero_order={"eta": 0.4})
    assert not torch.allclose(
        wrapped.forward(phase), CalibratedBench(bench, other).forward(phase)
    )


def test_a_rejected_fit_returns_the_nominal_eta(monkeypatch: pytest.MonkeyPatch) -> None:
    """A noiseless nominal session cannot improve on exact, so the gate rejects
    the fit -- and the returned eta is the config's, not the raw logit's."""
    sim = Sim2f(
        dataclasses.replace(SHAPE, errors=OpticsErrors(zero_order_eta=0.2)),
        device="cpu",
    )
    monkeypatch.setattr(calibrate, "_init_camera", lambda *args, **kwargs: None)

    result = calibrate_bench(
        sim, _session(sim), _frozen_geometry(iters=5, fit_zero_order=True)
    )

    assert not result.ok
    assert result.zero_order["eta"] == pytest.approx(0.2)
    assert result.measurement == MeasurementTransfer().to_dict()
