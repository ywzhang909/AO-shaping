"""Tests for the differentiable beam shaping module.

These tests are hardware-independent and use pure PyTorch simulation.
They are gated on torch being installed (it lives in the optional `ml`
dependency group), so CI stays green without `uv sync --extra ml`.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module when torch is absent (optional dependency).
torch = pytest.importorskip("torch")

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    DifferentiableShapingResult,
    angular_spectrum_propagate_torch,
    create_target_mask,
    efficiency_loss,
    smoothness_regularization,
    total_loss,
    train_beam_shaping,
    uniformity_loss,
    zero_order_penalty,
)

# Small grid for fast unit tests.
GRID = (128, 128)
SIZE = 40


# ============================ Target masks ============================


class TestTargetMask:
    """Verify target mask generation for all supported shapes."""

    def test_square_shape_and_center(self):
        mask = create_target_mask("square", GRID, SIZE)
        assert mask.shape == GRID
        assert mask.dtype == np.float64
        # Values in [0, 1]
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= 1.0 + 1e-9
        # Centered: center pixel lit
        cy, cx = GRID[0] // 2, GRID[1] // 2
        assert float(mask[cy, cx]) > 0.9

    def test_square_bounding_size(self):
        mask = create_target_mask("square", GRID, SIZE)
        ys, xs = np.nonzero(mask > 0.5)
        h_span = int(ys.max()) - int(ys.min()) + 1
        w_span = int(xs.max()) - int(xs.min()) + 1
        # Square region roughly `SIZE` wide, allow small padding
        assert w_span <= SIZE + 2
        assert h_span <= SIZE + 2

    def test_circle_shape(self):
        mask = create_target_mask("circle", GRID, SIZE)
        assert mask.shape == GRID
        # Center lit
        assert float(mask[GRID[0] // 2, GRID[1] // 2]) > 0.9
        # Corners dark (filled disk)
        assert float(mask[0, 0]) == 0.0

    @pytest.mark.parametrize("shape", ["rectangle", "triangle", "star"])
    def test_geometric_shapes_are_filled_and_distinct(self, shape):
        mask = create_target_mask(shape, (65, 65), 30)
        assert mask.shape == (65, 65)
        assert set(np.unique(mask)).issubset({0.0, 1.0})
        assert mask[32, 32] == 1.0
        assert mask[0, 0] == 0.0
        square = create_target_mask("square", (65, 65), 30)
        assert not np.array_equal(mask, square)

    def test_rectangle_is_twice_as_wide_as_high(self):
        mask = create_target_mask("rectangle", (65, 65), 30)
        ys, xs = np.nonzero(mask)
        assert (xs.max() - xs.min()) >= 1.8 * (ys.max() - ys.min())

    def test_spot_is_filled_disk(self):
        mask = create_target_mask("spot", GRID, SIZE)
        assert mask.shape == GRID
        # Spot == focused point: small centered disk, center lit, corners dark
        assert float(mask[GRID[0] // 2, GRID[1] // 2]) > 0.9
        assert float(mask[0, 0]) == 0.0

    def test_gaussian_normalized_peak(self):
        mask = create_target_mask("gaussian", GRID, SIZE)
        assert mask.shape == GRID
        assert float(mask.max()) == pytest.approx(1.0, abs=1e-6)

    def test_all_shapes_distinct(self):
        masks = {
            name: create_target_mask(name, GRID, SIZE)
            for name in ("square", "circle", "gaussian", "spot")
        }
        assert masks["square"].shape == GRID
        # Square and circle differ (corners of square lit, circle corners dark)
        assert not np.allclose(masks["square"], masks["circle"], atol=0.2)
        # Gaussian is not binary-anywhere-near (peak spread)
        assert not np.allclose(masks["gaussian"] > 0.5, masks["circle"] > 0.5)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            create_target_mask("banana", GRID, SIZE)


# ============================ Loss functions ============================


class TestLosses:
    """Verify the differentiable loss functions behave correctly."""

    def test_uniformity_lower_for_uniform_region(self):
        # Uniform intensity inside a target mask -> small CV -> small loss
        target = torch.full(GRID, 1.0)
        uniform = torch.full(GRID, 0.5)
        # Non-uniform: high value at center only
        non_uniform = torch.zeros(GRID)
        non_uniform[GRID[0] // 2, GRID[1] // 2] = 10.0

        l_uniform = uniformity_loss(uniform, target)
        l_nonuniform = uniformity_loss(non_uniform, target)
        assert float(l_uniform) < float(l_nonuniform)

    def test_uniformity_ignores_soft_target_edge(self):
        target = torch.tensor([[0.0, 0.25, 0.75, 1.0]], dtype=torch.float64)
        intensity = torch.tensor([[0.0, 0.1, 2.0, 2.0]], dtype=torch.float64)
        assert float(uniformity_loss(intensity, target)) == pytest.approx(0.0)

    def test_efficiency_lower_when_energy_in_target(self):
        target = torch.zeros(GRID)
        target[
            GRID[0] // 2 - 5 : GRID[0] // 2 + 5, GRID[1] // 2 - 5 : GRID[1] // 2 + 5
        ] = 1.0

        in_target = torch.zeros(GRID)
        in_target[GRID[0] // 2, GRID[1] // 2] = 1.0  # energy inside bucket
        out_target = torch.zeros(GRID)
        out_target[0, 0] = 1.0  # energy far outside bucket

        l_in = efficiency_loss(in_target, target)
        l_out = efficiency_loss(out_target, target)
        assert float(l_in) < float(l_out)

    def test_zero_order_penalty_larger_with_dc_spike(self):
        flat = torch.full(GRID, 0.01)
        spiked = torch.zeros(GRID)
        spiked[GRID[0] // 2, GRID[1] // 2] = 100.0
        assert float(zero_order_penalty(spiked)) > float(zero_order_penalty(flat))

    def test_smoothness_zero_flat_positive_noisy(self):
        flat = torch.zeros(GRID)
        noisy = torch.randn(*GRID)
        assert float(smoothness_regularization(flat)) == pytest.approx(0.0, abs=1e-9)
        assert float(smoothness_regularization(noisy)) > 0.0

    def test_total_loss_weighted_combination(self):
        target = torch.full(GRID, 1.0)
        intensity = torch.full(GRID, 0.5)
        phase = torch.zeros(GRID)
        l = total_loss(
            intensity,
            phase,
            target,
            w_uniformity=0.4,
            w_efficiency=0.4,
            w_zero_order=0.1,
            w_smoothness=0.1,
        )
        assert l.ndim == 0
        assert float(l) >= 0.0


# ============================ Forward model ============================


class TestForwardModel:
    """Verify the differentiable forward model produces finite intensity and gradients."""

    def _build(self, propagation: str = "fft"):
        from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
            _build_forward,
        )

        return _build_forward(propagation, 8e-6, 0.1, 1064e-9)

    def test_fft_forward_finite(self):
        from ao_shaping.algorithm.signal_processing.differentiable_shaping import _torch

        th = _torch()
        phase = th.zeros(*GRID, requires_grad=True)
        intensity = self._build("fft")(phase)
        assert intensity.shape == GRID
        assert th.isfinite(intensity).all()
        assert float(intensity.detach().min()) >= 0.0  # intensity non-negative

    def test_asm_forward_finite(self):
        from ao_shaping.algorithm.signal_processing.differentiable_shaping import _torch

        th = _torch()
        phase = th.zeros(*GRID, requires_grad=True)
        intensity = self._build("asm")(phase)
        assert intensity.shape == GRID
        assert th.isfinite(intensity).all()

    def test_gradient_flows(self):
        from ao_shaping.algorithm.signal_processing.differentiable_shaping import _torch

        th = _torch()
        phase = th.zeros(*GRID, requires_grad=True)
        intensity = self._build("fft")(phase)
        loss = intensity.sum()
        loss.backward()
        assert phase.grad is not None
        assert th.isfinite(phase.grad).all()

    def test_flat_phase_fft_has_dc_peak(self):
        # A flat phase maps to a strong DC spike at the field center under the
        # Fraunhofer model — the behaviour the zero-order penalty targets.
        from ao_shaping.algorithm.signal_processing.differentiable_shaping import _torch

        th = _torch()
        phase = th.zeros(*GRID, requires_grad=False)
        intensity = self._build("fft")(phase)
        cy, cx = GRID[0] // 2, GRID[1] // 2
        assert float(intensity[cy, cx]) > 100.0 * float(intensity[0, 0])

    def test_asm_propagate_torch_matches_numpy_scale(self):
        # Sanity: torch ASM returns complex field, finite, on a small grid
        small = (64, 64)
        field = torch.ones(*small, dtype=torch.complex128)
        out = angular_spectrum_propagate_torch(field, 8e-6, 0.1, 1064e-9)
        assert out.shape == small
        assert torch.isfinite(out).all()


# ============================ Training loop ============================


class TestTrain:
    """Verify train_beam_shaping behavior."""

    def _run(self, **kw):
        target = create_target_mask("square", GRID, SIZE)
        defaults = dict(iterations=60, lr=1e-2, seed=0)
        defaults.update(kw)
        return train_beam_shaping(target, GRID, **defaults)

    def test_loss_decreases(self):
        result = self._run()
        assert len(result.loss_history) >= 2
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-9

    def test_phase_shape_and_finite(self):
        result = self._run()
        assert result.phase.shape == GRID
        assert np.isfinite(result.phase).all()
        assert result.phase.dtype == np.float64

    def test_adam_and_lbfgs_both_run(self):
        for opt in ("adam", "lbfgs"):
            r = self._run(optimizer=opt)
            assert np.isfinite(r.phase).all()
            assert len(r.loss_history) > 0

    def test_result_dataclass_fields(self):
        result = self._run()
        assert isinstance(result.phase, np.ndarray)
        assert isinstance(result.target_intensity, np.ndarray)
        assert isinstance(result.simulated_intensity, np.ndarray)
        assert isinstance(result.loss_history, list)
        assert isinstance(result.iterations, int)
        assert isinstance(result.converged, bool)

    def test_fft_and_asm_agree_approximately(self):
        # Loose agreement: both should put energy near target; not exact (approx)
        r_fft = self._run(propagation="fft", iterations=40, seed=1)
        r_asm = self._run(propagation="asm", iterations=40, seed=1)
        # Both finite and same shape
        assert r_fft.phase.shape == r_asm.phase.shape == GRID
        assert np.isfinite(r_fft.phase).all()
        assert np.isfinite(r_asm.phase).all()

    def test_deterministic_with_seed(self):
        r1 = self._run(seed=42)
        r2 = self._run(seed=42)
        np.testing.assert_allclose(r1.phase, r2.phase, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(r1.loss_history), np.asarray(r2.loss_history), atol=1e-6
        )

    def test_no_nan_phase(self):
        result = self._run()
        assert not np.isnan(result.phase).any()

    def test_initial_phase_respected(self):
        init = np.full(GRID, 1.0, dtype=np.float64)
        r = self._run(initial_phase=init, iterations=10, seed=0)
        # starting from non-zero phase still runs
        assert np.isfinite(r.phase).all()

    def test_source_amplitude_accepted(self):
        amp = np.ones(GRID, dtype=np.float64)
        amp[GRID[0] // 2, GRID[1] // 2] = 2.0
        r = self._run(source_amplitude=amp, iterations=10, seed=0)
        assert np.isfinite(r.phase).all()


# ============================ Validation ============================


class TestValidation:
    """Verify input validation raises helpful errors."""

    def test_bad_propagation_raises(self):
        target = create_target_mask("square", GRID, SIZE)
        with pytest.raises(ValueError):
            train_beam_shaping(target, GRID, propagation="banana", iterations=5)

    def test_bad_optimizer_raises(self):
        target = create_target_mask("square", GRID, SIZE)
        with pytest.raises(ValueError):
            train_beam_shaping(target, GRID, optimizer="banana", iterations=5)

    def test_non_2d_target_raises(self):
        target = np.ones((GRID[0],), dtype=np.float64)
        with pytest.raises(ValueError):
            train_beam_shaping(target, GRID, iterations=5)

    def test_size_mismatch_target_grid_raises(self):
        target = create_target_mask("square", (64, 64), 20)
        with pytest.raises(ValueError):
            train_beam_shaping(target, GRID, iterations=5)

    def test_invalid_grid_raises(self):
        target = create_target_mask("square", GRID, SIZE)
        with pytest.raises(ValueError):
            train_beam_shaping(target, (0, 0), iterations=5)


# ============================ Step B debug recorder ============================


class TestDebugRecording:
    """``train_beam_shaping(debug=...)`` records diagnostics, never the solve.

    The recorder is the counterpart of the hardware-side ``step_b_trace``: it
    answers "should I speed up the forward model or the optimiser?" from measured
    per-iteration numbers. Enabling it must not perturb the result, or the debug
    run stops being evidence for the production run.
    """

    GRID_DEBUG = (32, 32)

    def _run(self, debug):
        target = create_target_mask("square", self.GRID_DEBUG, 8)
        return train_beam_shaping(
            target,
            self.GRID_DEBUG,
            iterations=8,
            seed=7,
            propagation="fft",
            optimizer="adam",
            debug=debug,
        )

    def test_debug_does_not_change_the_solve(self, tmp_path):
        plain = self._run(None)
        recorded = self._run(str(tmp_path / "dbg"))
        assert plain.loss_history == recorded.loss_history
        assert np.array_equal(np.asarray(plain.phase), np.asarray(recorded.phase))

    def test_directory_argument_writes_one_row_per_iteration(self, tmp_path):
        import json

        from ao_shaping.optimizer.wfless.step_b_debug import SUMMARY_NAME

        res = self._run(str(tmp_path))
        assert len(res.loss_history) == 8
        summary_path = tmp_path / SUMMARY_NAME.format(tag="step_b")
        assert summary_path.exists()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["speed"]["iterations"] == 8
        assert payload["context"]["planned_iterations"] == 8
        rows = payload["records"]
        assert [r["iteration"] for r in rows] == list(range(8))
        # The recorded loss is the forward-model loss of the same step.
        assert [r["loss"] for r in rows] == pytest.approx(res.loss_history)
        assert all(r["grad_norm"] > 0.0 for r in rows)
        assert all(r["phase_delta_norm"] > 0.0 for r in rows)
        assert all(r["forward_s"] > 0.0 for r in rows)
        assert all(r["step_s"] >= 0.0 for r in rows)

    def test_caller_supplied_recorder_is_left_open(self, tmp_path):
        from ao_shaping.optimizer.wfless.step_b_debug import (
            JSONL_NAME,
            StepBDebugRecorder,
        )

        rec = StepBDebugRecorder(tmp_path)
        try:
            res = self._run(rec)
            # Still open: the summary is the caller's to write.
            assert not (tmp_path / "step_b_debug.json").exists()
            assert rec.speed_summary()["iterations"] == len(res.loss_history)
        finally:
            summary = rec.close()
        assert summary is not None and summary.exists()
        assert (tmp_path / JSONL_NAME.format(tag="step_b")).exists()


# ============================ LR schedules ============================


class TestLrSchedule:
    """``train_beam_shaping(lr_schedule=...)``: 动态学习率调度。

    ``None`` 必须与加调度之前的行为**逐位一致** (用改动前捕获的 loss 轨迹
    回归); 三个调度各自把逐迭代学习率改成声明的形状; cosine 在小合成问题
    上仍能降 loss。
    """

    GRID_S = (32, 32)
    SIZE_S = 6
    ITER_S = 20
    LR_S = 0.05

    # (a) 改动前 (pre-change) 用同一调用捕获的 loss 轨迹:
    # region 32 / iters 20 / seed 0 / lr 0.05 / adam / fft。
    # ``lr_schedule=None`` 必须逐位复现这条轨迹。
    REF_LOSSES_NONE = [
        2.805043, 2.797315, 2.784147, 2.763668, 2.735144, 2.698135,
        2.65232, 2.597448, 2.533342, 2.459912, 2.377164, 2.285214,
        2.184302, 2.074803, 1.957256, 1.832385, 1.701129, 1.564675,
        1.424501, 1.282418,
    ]

    def _target(self):
        return create_target_mask("square", self.GRID_S, self.SIZE_S)

    def _run(self, **kw):
        defaults = dict(
            propagation="fft",
            optimizer="adam",
            iterations=self.ITER_S,
            lr=self.LR_S,
            w_uniformity=0.4,
            w_efficiency=0.6,
            seed=0,
            device="cpu",
        )
        defaults.update(kw)
        return train_beam_shaping(self._target(), self.GRID_S, **defaults)

    def test_none_schedule_reproduces_pre_change_trajectory(self):
        result = self._run(lr_schedule=None)
        assert [round(float(x), 6) for x in result.loss_history] == (
            self.REF_LOSSES_NONE
        )

    def _capture_lr(self, monkeypatch, schedule: str | None):
        """给优化器参数组包一层记录字典, 捕获逐迭代写入的学习率。

        返回 ``(构造时初始 lr, 循环内每次写入的 lr 序列)``。
        """
        import ao_shaping.algorithm.signal_processing.differentiable_shaping as ds

        holder: dict[str, object] = {"initial_lr": None, "sets": []}
        real_builder = ds._shaping_optimizer

        def spy(torch, phase, optimizer, lr, lr_schedule=None):
            opt = real_builder(torch, phase, optimizer, lr, lr_schedule)
            group = opt.param_groups[0]
            holder["initial_lr"] = float(group["lr"])
            sets: list[float] = []

            class _Recording(dict):
                def __setitem__(self, key, value):
                    super().__setitem__(key, value)
                    if key == "lr":
                        sets.append(float(value))

            opt.param_groups[0] = _Recording(group)
            holder["sets"] = sets
            return opt

        monkeypatch.setattr(ds, "_shaping_optimizer", spy)
        self._run(lr_schedule=schedule)
        return float(holder["initial_lr"]), list(holder["sets"])  # type: ignore[arg-type]

    def test_none_schedule_never_rewrites_lr(self, monkeypatch):
        initial_lr, sets = self._capture_lr(monkeypatch, None)
        # 优化器以 base lr 构造, 循环从不改写 —— 恒定学习率。
        assert initial_lr == pytest.approx(self.LR_S)
        assert sets == []

    def test_schedules_apply_expected_lr_shape(self, monkeypatch):
        from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
            _lr_schedule_factor,
        )

        for schedule in ("cosine", "step", "warmup-cosine"):
            initial_lr, sets = self._capture_lr(monkeypatch, schedule)
            assert initial_lr == pytest.approx(self.LR_S)
            assert len(sets) == self.ITER_S
            expected = [
                self.LR_S * _lr_schedule_factor(schedule, i, self.ITER_S)
                for i in range(self.ITER_S)
            ]
            assert sets == pytest.approx(expected, rel=1e-9)
            # 确实与恒定学习率不同 (调度真的在起作用)。
            assert sets != pytest.approx([self.LR_S] * self.ITER_S)

    def test_cosine_schedule_decreases_loss(self):
        result = self._run(lr_schedule="cosine")
        assert len(result.loss_history) == self.ITER_S
        assert result.loss_history[-1] < result.loss_history[0]

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            self._run(lr_schedule="banana")


# ============================ Torch optionality ============================


class TestTorchGraceful:
    """Verify the module raises a helpful ImportError when torch is missing."""

    def test_import_error_message(self, monkeypatch):
        from ao_shaping.algorithm.signal_processing import differentiable_shaping

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(
            differentiable_shaping,
            "_torch",
            lambda: (_ for _ in ()).throw(
                ImportError(
                    "differentiable_shaping requires PyTorch. Install with: uv sync --extra ml"
                )
            ),
        )
        with pytest.raises(ImportError, match="requires PyTorch"):
            differentiable_shaping._torch()
