"""Tests for the class-based API of ``DifferentiableBeamOptimizer``.

Covers input validation, seeded determinism, early stopping, edge cases
(``epochs=0``, ``update()`` contract), result-field types, the
``optimize_beam_shaping`` loop's recorder history and best-phase tracking,
and exact parity between ``optimize_beam_shaping`` and a manual ``update()``
loop. Simulation content (reference FFTs, shaped far fields) lives in a
separate test file.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from ao_shaping.algorithm import far_field_intensity
from ao_shaping.optimizer import (
    DifferentiableBeamOptimizer,
    optimize_beam_shaping,
)
from ao_shaping.utils import Recorder
from ao_shaping.utils.image.targets import crop_resize_to_grid


def _gaussian_target(size: int = 32) -> np.ndarray:
    """A smooth 2D gaussian target intensity (peak-normalized)."""
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r2 = x**2 + y**2
    g = np.exp(-r2 / (size / 4.0))
    return g.astype(np.float32) / g.max()


class _MockSLM:
    """Records applied radian phases; exposes only ``write_phase``.

    This is the mock path of ``optimize_beam_shaping``'s ``_display_phase``
    (no ``create_phase_from_array`` attribute), so phases are recorded in
    radians exactly as the optimizer sends them.
    """

    def __init__(self, resolution: tuple[int, int]) -> None:
        self.resolution = resolution
        self.applied_phases: list[np.ndarray] = []
        self.current_phase: np.ndarray | None = None

    def write_phase(self, phase_rad: np.ndarray) -> None:
        phase = np.asarray(phase_rad, dtype=np.float32)
        assert phase.shape == self.resolution
        self.applied_phases.append(phase.copy())
        self.current_phase = phase.copy()


class _MockCCD:
    """Returns the model far-field intensity of the last applied SLM phase.

    The noiseless mock makes the measured image exactly the model's own
    forward pass, so the hardware loop's measured loss tracks the simulated
    loss (up to float32 round-trip noise) and the loop's per-iteration
    behavior is exercised without any real hardware.
    """

    def __init__(self, source_amplitude: np.ndarray, slm: _MockSLM) -> None:
        self.source_amplitude = source_amplitude
        self.slm = slm
        self.capture_count = 0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        self.capture_count += 1
        assert self.slm.current_phase is not None, "no phase applied before capture"
        phase = torch.from_numpy(self.slm.current_phase)
        i_far = far_field_intensity(self.source_amplitude, phase)
        return i_far.detach().cpu().numpy().astype(np.float32)


class TestDifferentiableBeamOptimizerValidation:
    """Input validation: bad targets, shapes, and initial phases."""

    def test_rejects_non_2d_target(self):
        with pytest.raises(ValueError):
            DifferentiableBeamOptimizer(
                target_intensity=np.ones((10,)), device="cpu", seed=0
            )

    def test_rejects_negative_target(self):
        bad = _gaussian_target(16)
        bad[0, 0] = -1.0
        with pytest.raises(ValueError):
            DifferentiableBeamOptimizer(
                target_intensity=bad, device="cpu", seed=0
            )

    def test_rejects_source_amplitude_shape_mismatch(self):
        target = _gaussian_target(16)
        with pytest.raises(ValueError):
            DifferentiableBeamOptimizer(
                target_intensity=target,
                source_amplitude=np.ones((8, 8), dtype=np.float32),
                device="cpu",
                seed=0,
            )

    def test_rejects_init_phase_shape_mismatch(self):
        target = _gaussian_target(16)
        with pytest.raises(ValueError):
            DifferentiableBeamOptimizer(
                target_intensity=target,
                init_phase=np.zeros((8, 8), dtype=np.float32),
                device="cpu",
                seed=0,
            )


class TestDifferentiableBeamOptimizerInit:
    """Construction, default device, and seeded initialization."""

    def test_default_device_is_non_empty_string(self):
        opt = DifferentiableBeamOptimizer(
            target_intensity=_gaussian_target(16), seed=0
        )
        assert isinstance(opt.device, str)
        assert len(opt.device) > 0
        assert isinstance(opt.step, int)
        assert isinstance(opt.phase_tensor, torch.Tensor)
        res_list = optimize_beam_shaping(
            target_intensity=_gaussian_target(16), seed=0, epochs=1, log_every=0
        )
        assert isinstance(res_list.last["device"], str)
        assert len(res_list.last["device"]) > 0

    def test_same_seed_gives_identical_trajectory(self):
        target = _gaussian_target(32)
        r1 = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=42, epochs=10, log_every=0
        )
        r2 = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=42, epochs=10, log_every=0
        )
        assert [x["loss"] for x in r1.history] == [x["loss"] for x in r2.history]
        best1, _ = r1.get_best_iter()
        best2, _ = r2.get_best_iter()
        assert np.array_equal(best1["best_phase"], best2["best_phase"])

    def test_different_seeds_give_different_phases(self):
        target = _gaussian_target(32)
        r1 = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=1, epochs=10, log_every=0
        )
        r2 = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=2, epochs=10, log_every=0
        )
        best1, _ = r1.get_best_iter()
        best2, _ = r2.get_best_iter()
        assert not np.array_equal(best1["best_phase"], best2["best_phase"])

    def test_init_phase_override(self):
        target = _gaussian_target(16)
        init = np.full((16, 16), 0.5, dtype=np.float32)
        opt = DifferentiableBeamOptimizer(
            target, init_phase=init, device="cpu", seed=0
        )
        assert np.array_equal(opt.current_phase, init)


class TestDifferentiableBeamOptimizerRun:
    """The optimize_beam_shaping loop: zero-epoch edge cases, early stopping,
    result fields."""

    def test_run_zero_epochs_with_init_phase(self):
        target = _gaussian_target(16)
        init = np.full((16, 16), 0.5, dtype=np.float32)
        res_list = optimize_beam_shaping(
            target_intensity=target,
            init_phase=init,
            device="cpu",
            seed=0,
            epochs=0,
            log_every=0,
        )
        assert len(res_list.history) == 0
        final_loss = res_list.last["loss"] if res_list.history else float("nan")
        converged = res_list.last["converged"] if res_list.history else False
        assert math.isnan(final_loss)
        assert converged is False

    def test_run_zero_epochs_without_init_phase(self):
        res_list = optimize_beam_shaping(
            target_intensity=_gaussian_target(16),
            device="cpu",
            seed=0,
            epochs=0,
            log_every=0,
        )
        assert len(res_list.history) == 0
        final_loss = res_list.last["loss"] if res_list.history else float("nan")
        converged = res_list.last["converged"] if res_list.history else False
        assert math.isnan(final_loss)
        assert converged is False

    def test_early_stopping(self):
        target = _gaussian_target(32)
        res_list = optimize_beam_shaping(
            target_intensity=target,
            lr=0.01,
            device="cpu",
            seed=0,
            epochs=300,
            early_stop_patience=50,
            early_stop_delta=1e-3,
            log_every=0,
        )
        assert res_list.last["converged"] is True
        assert len(res_list.history) < 300

    def test_result_fields(self):
        target = _gaussian_target(32)
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=10, log_every=0
        )
        assert isinstance(res_list, Recorder)
        best_iter, _ = res_list.get_best_iter()
        assert isinstance(best_iter["best_phase"], np.ndarray)
        assert best_iter["best_phase"].shape == target.shape
        assert best_iter["best_phase"].dtype == np.float32
        loss_history = [r["loss"] for r in res_list.history]
        assert isinstance(loss_history, list)
        assert all(isinstance(v, float) for v in loss_history)
        assert len(loss_history) == len(res_list.history)
        assert res_list.last["loss"] == loss_history[-1]
        assert isinstance(res_list.last["converged"], bool)
        assert isinstance(res_list.last["device"], str)


class TestDifferentiableBeamOptimizerUpdate:
    """The manual update() API contract."""

    def test_update_increments_step_and_appends_loss(self):
        target = _gaussian_target(16)
        opt = DifferentiableBeamOptimizer(target, device="cpu", seed=0)
        assert opt.step == 0
        assert opt.loss_history == []
        assert opt.last_loss is None
        phase = opt.update()
        assert opt.step == 1
        assert len(opt.loss_history) == 1
        assert isinstance(opt.last_loss, float)
        assert opt.last_loss == opt.loss_history[-1]
        assert isinstance(phase, np.ndarray)
        assert phase.shape == target.shape
        assert np.array_equal(phase, opt.current_phase)


class TestDifferentiableBeamOptimizerParity:
    """``optimize_beam_shaping`` must reproduce a manual ``update()`` loop
    exactly."""

    @staticmethod
    def _manual_run(opt, epochs: int, delta: float = 1e-5):
        """Drive ``update()`` manually, mirroring the loop's best tracking."""
        best_loss = float("inf")
        best_phase = None
        for _ in range(epochs):
            opt.update()
            loss_val = opt.loss_history[-1]
            if loss_val < best_loss - delta:
                best_loss = loss_val
                best_phase = opt.current_phase
        return best_loss, best_phase

    def test_parity_with_manual_update_loop(self):
        target = _gaussian_target(32)
        source = np.ones((32, 32), dtype=np.float32)
        res_list = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=source,
            lr=0.01,
            device="cpu",
            seed=7,
            epochs=10,
            log_every=0,
        )

        manual = DifferentiableBeamOptimizer(
            target, source_amplitude=source, lr=0.01, device="cpu", seed=7
        )
        best_loss, best_phase = self._manual_run(manual, 10)

        loss_history = [r["loss"] for r in res_list.history]
        assert loss_history == manual.loss_history
        best_iter, _ = res_list.get_best_iter()
        assert best_iter["best_loss"] == best_loss
        assert best_iter["best_phase"] is not None
        assert best_phase is not None
        assert np.array_equal(best_iter["best_phase"], best_phase)

    def test_parity_with_init_phase(self):
        target = _gaussian_target(32)
        init = np.full((32, 32), 0.3, dtype=np.float32)
        res_list = optimize_beam_shaping(
            target_intensity=target,
            lr=0.01,
            init_phase=init,
            device="cpu",
            seed=7,
            epochs=10,
            log_every=0,
        )

        manual = DifferentiableBeamOptimizer(
            target, lr=0.01, init_phase=init, device="cpu", seed=7
        )
        best_loss, best_phase = self._manual_run(manual, 10)

        loss_history = [r["loss"] for r in res_list.history]
        assert loss_history == manual.loss_history
        best_iter, _ = res_list.get_best_iter()
        assert best_iter["best_loss"] == best_loss
        assert best_iter["best_phase"] is not None
        assert best_phase is not None
        assert np.array_equal(best_iter["best_phase"], best_phase)


class TestDifferentiableBeamOptimizerRunRecording:
    """The optimize_beam_shaping loop's recorder history and best-phase
    tracking."""

    def test_recorder_history_fields(self):
        target = _gaussian_target(32)
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=5, log_every=0
        )
        assert len(res_list.history) == 5
        record = res_list.history[0]
        assert set(record) >= {
            "loss",
            "lr",
            "_epoch",
            "best_loss",
            "best_phase",
            "_id",
        }
        assert record["_epoch"] == 0
        assert record["lr"] == 0.01
        assert record["loss"] == res_list.history[0]["loss"]
        assert record["best_phase"].shape == target.shape

    def test_best_loss_and_best_phase(self):
        target = _gaussian_target(32)
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=20, log_every=0
        )
        best_iter, _ = res_list.get_best_iter()
        loss_history = [r["loss"] for r in res_list.history]
        assert best_iter["best_loss"] is not None
        assert best_iter["best_loss"] <= loss_history[0]
        assert best_iter["best_loss"] <= res_list.last["loss"]
        assert best_iter["best_phase"] is not None
        assert best_iter["best_phase"].shape == target.shape

    def test_tqdm_runs_with_progress_true_and_false(self):
        target = _gaussian_target(16)
        r1 = optimize_beam_shaping(
            target_intensity=target,
            device="cpu",
            seed=0,
            epochs=3,
            log_every=0,
            progress=True,
        )
        r2 = optimize_beam_shaping(
            target_intensity=target,
            device="cpu",
            seed=0,
            epochs=3,
            log_every=0,
            progress=False,
        )
        assert [x["loss"] for x in r1.history] == [x["loss"] for x in r2.history]

    def test_epochs_int_coercion(self):
        target = _gaussian_target(16)
        epochs_any: Any = "5"
        res_list = optimize_beam_shaping(
            target_intensity=target,
            device="cpu",
            seed=0,
            epochs=epochs_any,
            log_every=0,
        )
        assert len(res_list.history) == 5

    def test_invalid_progress_raises_value_error(self):
        target = _gaussian_target(16)
        progress_any: Any = "yes"
        with pytest.raises(ValueError):
            optimize_beam_shaping(
                target_intensity=target,
                device="cpu",
                seed=0,
                epochs=1,
                progress=progress_any,
            )

    def test_seed_reproducibility_includes_best_phase(self):
        target = _gaussian_target(32)
        r1 = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=42, epochs=10, log_every=0
        )
        r2 = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=42, epochs=10, log_every=0
        )
        assert [x["loss"] for x in r1.history] == [x["loss"] for x in r2.history]
        best1, _ = r1.get_best_iter()
        best2, _ = r2.get_best_iter()
        assert best1["best_loss"] == best2["best_loss"]
        assert best1["best_phase"] is not None
        assert best2["best_phase"] is not None
        assert np.array_equal(best1["best_phase"], best2["best_phase"])


class TestHardwareClosedLoop:
    """Hardware-closed-loop mode: the forward pass goes through mock SLM/CCD.

    The mock CCD returns the model far-field of the last applied SLM phase,
    so the loop's per-iteration behavior (apply -> measure -> update ->
    apply) is exercised without any real hardware.
    """

    @staticmethod
    def _make_devices(size: int = 32) -> tuple[np.ndarray, _MockSLM, _MockCCD]:
        amp = np.ones((size, size), dtype=np.float32)
        slm = _MockSLM((size, size))
        ccd = _MockCCD(amp, slm)
        return amp, slm, ccd

    def test_apply_measure_update_apply_ordering(self):
        target = _gaussian_target(32)
        amp, slm, ccd = self._make_devices(32)
        init = (0.1 * np.random.default_rng(0).random((32, 32))).astype(np.float32)
        # ``torch.from_numpy`` shares memory with ``init``, so the optimizer's
        # in-place Adam steps mutate the caller's array. Snapshot it before the
        # call to assert on the true initial phase.
        init_snapshot = init.copy()
        res_list = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=amp,
            lr=0.01,
            init_phase=init,
            device="cpu",
            seed=0,
            epochs=5,
            log_every=0,
            progress=False,
            slm=slm,
            ccd=ccd,
            wait_time_s=0.0,
            discard_count=2,
        )
        # Initial display + one display per epoch.
        assert len(slm.applied_phases) == 6
        # The first applied phase is the optimizer's initial phase.
        assert np.array_equal(slm.applied_phases[0], init_snapshot)
        # Each epoch: discard_count discards + 1 real capture.
        assert ccd.capture_count == (2 + 1) * 5
        # The update actually changes the phase.
        assert not np.array_equal(slm.applied_phases[0], slm.applied_phases[-1])
        # Each record's measured image is the resized far field of the phase
        # that was applied BEFORE that epoch's capture (apply -> measure
        # ordering). Replicate the optimizer's crop_resize_to_grid so the
        # assertion holds regardless of where the far-field peak lands.
        for k, record in enumerate(res_list.history):
            applied = slm.applied_phases[k]
            expected = far_field_intensity(amp, torch.from_numpy(applied))
            expected_np = expected.detach().cpu().numpy().astype(np.float32)
            expected_resized = crop_resize_to_grid(
                expected_np, grid_h=target.shape[0], grid_w=target.shape[1]
            )
            assert np.allclose(record["measured_image"], expected_resized, atol=1e-5)

    def test_recorder_fields_and_best_iter(self):
        target = _gaussian_target(32)
        amp, slm, ccd = self._make_devices(32)
        res_list = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=amp,
            lr=0.01,
            device="cpu",
            seed=0,
            epochs=10,
            log_every=0,
            progress=False,
            slm=slm,
            ccd=ccd,
            wait_time_s=0.0,
            discard_count=1,
        )
        assert len(res_list.history) == 10
        record = res_list.history[0]
        assert set(record) >= {
            "loss",
            "lr",
            "_epoch",
            "best_loss",
            "best_phase",
            "converged",
            "device",
            "measured_image",
        }
        assert record["measured_image"].shape == target.shape
        assert record["measured_image"].dtype == np.float32
        best_iter, _ = res_list.get_best_iter()
        assert best_iter["best_phase"].shape == target.shape
        # Best phase is one of the APPLIED phases (tracked on the phase that
        # was on the SLM during the capture, not the post-update phase).
        assert any(
            np.array_equal(best_iter["best_phase"], p)
            for p in slm.applied_phases
        )

    def test_zero_epochs_applies_init_phase_once(self):
        target = _gaussian_target(16)
        amp, slm, ccd = self._make_devices(16)
        init = np.full((16, 16), 0.5, dtype=np.float32)
        res_list = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=amp,
            init_phase=init,
            device="cpu",
            seed=0,
            epochs=0,
            log_every=0,
            progress=False,
            slm=slm,
            ccd=ccd,
            wait_time_s=0.0,
            discard_count=2,
        )
        assert len(res_list.history) == 0
        # The initial phase is still applied once (SLM state matches the
        # optimizer state), but no capture happens.
        assert len(slm.applied_phases) == 1
        assert np.array_equal(slm.applied_phases[0], init)
        assert ccd.capture_count == 0

    def test_seed_determinism_hardware_loop(self):
        target = _gaussian_target(32)
        amp, slm1, ccd1 = self._make_devices(32)
        _, slm2, ccd2 = self._make_devices(32)
        r1 = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=amp,
            lr=0.01,
            device="cpu",
            seed=42,
            epochs=10,
            log_every=0,
            progress=False,
            slm=slm1,
            ccd=ccd1,
            wait_time_s=0.0,
            discard_count=1,
        )
        r2 = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=amp,
            lr=0.01,
            device="cpu",
            seed=42,
            epochs=10,
            log_every=0,
            progress=False,
            slm=slm2,
            ccd=ccd2,
            wait_time_s=0.0,
            discard_count=1,
        )
        assert [x["loss"] for x in r1.history] == [x["loss"] for x in r2.history]
        best1, _ = r1.get_best_iter()
        best2, _ = r2.get_best_iter()
        assert np.array_equal(best1["best_phase"], best2["best_phase"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])