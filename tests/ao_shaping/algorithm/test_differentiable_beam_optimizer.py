"""Tests for the class-based API of ``DifferentiableBeamOptimizer``.

Covers input validation, seeded determinism, early stopping, edge cases
(``epochs=0``, ``update()`` contract), result-field types, the
``optimize_beam_shaping`` loop's recorder history and best-phase tracking,
and exact parity between ``optimize_beam_shaping`` and a manual ``update()``
loop. Simulation content (reference FFTs, shaped far fields) lives in a
separate test file.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
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


class TestSetLr:
    """``set_lr``: drives an external per-iteration LR schedule.

    Used by ``scripts/slm/sweep_differentiable_lr.py`` (offline LR-schedule
    study). The method must update every Adam param group, keep ``update()``
    working afterwards, and reject non-positive / non-finite values.
    """

    def _make_opt(self, lr: float = 0.05) -> DifferentiableBeamOptimizer:
        return DifferentiableBeamOptimizer(
            _gaussian_target(16), lr=lr, device="cpu", seed=0
        )

    def test_set_lr_updates_all_param_groups(self):
        opt = self._make_opt()
        opt.set_lr(0.3)
        assert opt._lr == pytest.approx(0.3)
        groups = opt._opt.param_groups
        assert len(groups) >= 1
        for group in groups:
            assert group["lr"] == pytest.approx(0.3)

    @pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
    def test_set_lr_rejects_invalid(self, bad: float):
        opt = self._make_opt()
        with pytest.raises(ValueError):
            opt.set_lr(bad)
        # A rejected call must leave the optimizer untouched.
        assert opt._lr == pytest.approx(0.05)
        for group in opt._opt.param_groups:
            assert group["lr"] == pytest.approx(0.05)

    def test_update_after_set_lr_still_decreases_loss(self):
        opt = self._make_opt(lr=0.05)
        opt.update()
        first = opt.last_loss
        assert first is not None
        opt.set_lr(0.02)
        for _ in range(20):
            opt.update()
        assert opt.last_loss is not None
        assert math.isfinite(opt.last_loss)
        assert opt.last_loss < first

    def test_decreasing_schedule_is_applied(self):
        """A cosine decay fed through ``set_lr`` lands in the param groups."""
        opt = self._make_opt(lr=0.1)
        iters = 12
        floor = 0.1
        seen_lrs: list[float] = []
        losses: list[float] = []
        for i in range(iters):
            lr_i = 0.1 * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * i / iters)))
            opt.set_lr(lr_i)
            seen_lrs.append(float(opt._opt.param_groups[0]["lr"]))
            opt.update()
            assert opt.last_loss is not None
            losses.append(opt.last_loss)
        # The schedule is strictly decreasing and actually applied per step.
        assert all(b < a for a, b in zip(seen_lrs, seen_lrs[1:]))
        assert seen_lrs[0] == pytest.approx(0.1, rel=1e-6)
        assert all(math.isfinite(loss) for loss in losses)
        assert losses[-1] < losses[0]


class TestDifferentiableProblemRebuild:
    """Smoke test for the shared rebuild helper (``scripts/_common``).

    ``scripts/_common/differentiable_problem.py`` is the single bootstrap that
    both the offline LR sweep and the hardware script's re-analysis consume.
    The run directory is gitignored local data, so the test skips when absent.
    """

    def test_rebuild_and_first_loss_match_run_dir(self):
        root = Path(__file__).resolve().parents[3]
        run_dir = root / "data" / "debug" / "differentiable_opt" / "phys_20261116_01"
        if not (run_dir / "run_config.json").is_file():
            pytest.skip(f"gitignored run directory absent: {run_dir}")
        history_path = run_dir / "history.csv"
        if not history_path.is_file():
            pytest.skip(f"history.csv absent: {history_path}")

        from scripts._common.differentiable_problem import (
            build_optimizer,
            load_problem_from_run_dir,
        )

        problem = load_problem_from_run_dir(run_dir)
        m = int(problem.spec.m_grid)
        assert problem.amplitude.shape == (m, m)
        assert problem.target.shape == (m, m)
        assert problem.init_phase is not None
        assert problem.k_mean > 0.0

        with history_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "empty history.csv"
        first_loss_ref = float(rows[0]["loss"])

        opt = build_optimizer(
            problem,
            lr=float(problem.params.get("lr", 0.05)),
            device="cpu",
            seed=int(problem.params.get("seed", 0)),
        )
        opt.update()
        assert opt.last_loss is not None
        # CPU FFT round-off vs the recorded CUDA run: a few percent is ample.
        assert opt.last_loss == pytest.approx(first_loss_ref, rel=0.05)

    def test_init_phase_not_shared_between_builds(self):
        """Regression: rebuilding from one ``problem`` must not alias init_phase.

        ``DifferentiableBeamOptimizer.__init__`` does
        ``np.asarray(init_phase, dtype=np.float32)``; when the saved
        ``phase_initial.npy`` is already float32 this does **not** copy, and on
        CPU ``torch.from_numpy(...).to("cpu")`` shares that memory, so Adam's
        in-place ``step()`` would corrupt ``problem.init_phase``. Without a copy
        in ``build_optimizer``, every sweep arm after the first would start from
        the previous arm's final phase. This asserts the helper hands each
        optimizer a private array.
        """
        root = Path(__file__).resolve().parents[3]
        run_dir = root / "data" / "debug" / "differentiable_opt" / "phys_20261116_01"
        if not (run_dir / "run_config.json").is_file():
            pytest.skip(f"gitignored run directory absent: {run_dir}")

        from scripts._common.differentiable_problem import (
            build_optimizer,
            load_problem_from_run_dir,
        )

        problem = load_problem_from_run_dir(run_dir)
        assert problem.init_phase is not None
        init_before = problem.init_phase.copy()
        lr = float(problem.params.get("lr", 0.05))
        seed = int(problem.params.get("seed", 0))

        opt_a = build_optimizer(problem, lr=lr, device="cpu", seed=seed)
        opt_a.update()
        first_a = opt_a.last_loss
        assert first_a is not None
        # The shared problem array must be untouched by an optimizer step.
        assert np.array_equal(problem.init_phase, init_before)

        # A second build from the same problem must reproduce the same first
        # loss (it would be history iter2's loss if init_phase were aliased).
        opt_b = build_optimizer(problem, lr=lr, device="cpu", seed=seed)
        opt_b.update()
        assert opt_b.last_loss == pytest.approx(first_a, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])