"""K 个候选并行前向：``DifferentiableUniformityOptimizer(n_candidates=K)``。

K=1 的两条路必须逐位一致（缺省 vs 显式 ``n_candidates=1``）；K>1 对外仍是单相位
（``current_phase`` / ``delta_phase`` / ``last_candidate_prediction`` 都是 2D，取
argmin 的 best 候选），而增量 cap 逐候选各自生效（批量投影不能混成一个全局 RMS）。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (  # noqa: E402
    DifferentiableUniformityOptimizer,
    FrozenDefocusForward,
    FrozenDefocusParameters,
    UniformityObjective,
)

GRID = 8


class _AnalyticForward:
    """非均匀照明权重 + 批安全的解析前向：不需要数据文件或光路检查点。"""

    class params:
        grid_size = GRID

    _device = torch.device("cpu")

    def illumination_weights(self):
        weights = np.ones((GRID, GRID), dtype=np.float32)
        weights[:2, :] = 0.5  # 非均匀权重：批量投影若按全局 RMS 缩放会露馅
        return weights / weights.sum()

    def predict(self, phase, **kwargs):
        tensor = phase if torch.is_tensor(phase) else torch.as_tensor(phase, dtype=torch.float32)
        return 1.0 + 0.3 * tensor + 0.1 * torch.cos(tensor)

    def total_phase_grid(self, phase, **kwargs):
        return phase


def _objective():
    box = np.zeros((GRID, GRID), dtype=bool)
    box[1:7, 1:7] = True
    core = np.zeros((GRID, GRID), dtype=bool)
    core[3:5, 3:5] = True
    return UniformityObjective(box, core, w_phase=0.5)


def _optimizer(*, n_candidates=None, cap=0.05, seed=None):
    seed_phase = np.zeros((GRID, GRID), dtype=np.float32) if seed is None else seed
    kwargs = {} if n_candidates is None else {"n_candidates": n_candidates}
    return DifferentiableUniformityOptimizer(
        _AnalyticForward(), _objective(), seed_phase, lr=0.05, step_rms_cap=cap, **kwargs
    )


def test_missing_n_candidates_is_bit_identical_to_explicit_one():
    """缺省 K=1 与显式 ``n_candidates=1``：历史与相位逐位相等。"""
    default = _optimizer()
    explicit = _optimizer(n_candidates=1)
    assert default.n_candidates == 1 and explicit.n_candidates == 1
    for _ in range(4):
        default.update()
        explicit.update()
    assert len(default.loss_history) == len(explicit.loss_history) == 4
    assert default.loss_history == explicit.loss_history
    assert np.array_equal(default.current_phase, explicit.current_phase)
    assert np.array_equal(default.delta_phase, explicit.delta_phase)
    assert "loss_per_candidate" not in default.loss_history[-1]


def test_single_candidate_keeps_the_original_history_keys():
    optimizer = _optimizer(n_candidates=1)
    optimizer.update()
    record = optimizer.loss_history[-1]
    assert {
        "step",
        "mode",
        "loss",
        "gradient_norm",
        "increment_rms_before_projection",
        "increment_rms",
        "phase_rms",
        "l_cv",
        "l_phase",
        "l_shape",
        "anchor_model_loss",
        "candidate_model_loss",
        "candidate_model_loss_change",
        "raw_to_final_step_cosine",
    } <= set(record)
    # K>1 才追加的三个账目必须**不**出现，K=1 的历史与改动前逐键一致。
    assert not {"loss_best", "loss_per_candidate", "candidate"} & set(record)


def test_four_candidates_are_distinct_and_best_is_argmin():
    optimizer = _optimizer(n_candidates=4, cap=0.2)
    for _ in range(3):
        optimizer.update()
    deltas = optimizer._deltas.detach().numpy()
    assert deltas.shape == (4, GRID, GRID)
    # 候选之间必须真的不同，否则批量前向只是 K 份同样的计算。
    for first in range(4):
        for second in range(first + 1, 4):
            assert not np.array_equal(deltas[first], deltas[second])
    record = optimizer.loss_history[-1]
    assert len(record["loss_per_candidate"]) == 4
    best = int(np.argmin(record["loss_per_candidate"]))
    assert record["candidate"] == best == optimizer.candidate_index
    assert record["loss_best"] == record["loss_per_candidate"][best]
    # 对外仍是单相位：best 候选的 2D 增量。
    assert optimizer.delta_phase.ndim == 2
    assert optimizer.current_phase.ndim == 2
    assert np.allclose(optimizer.delta_phase, deltas[best])
    assert np.allclose(optimizer.current_phase, deltas[best])  # seed = 0，无 focus/carrier


def test_each_candidate_increment_respects_its_own_cap():
    """批量投影逐候选归约：每条增量 ≤ cap，且不同候选的 RMS 不被混成一个值。"""
    cap = 0.01
    optimizer = _optimizer(n_candidates=4, cap=cap)
    for _ in range(3):
        optimizer.update()
    rms = optimizer.candidate_increment_rms
    assert len(rms) == 4
    assert all(value <= cap + 1e-6 for value in rms)
    assert len({round(value, 9) for value in rms}) > 1


def test_anchored_four_candidates_stay_2d():
    optimizer = _optimizer(n_candidates=4, cap=0.2)
    rng = np.random.default_rng(7)
    measured = (1.0 + 0.05 * rng.standard_normal((GRID, GRID))).astype(np.float32)
    for _ in range(2):
        optimizer.update(measured)
    record = optimizer.loss_history[-1]
    assert record["mode"] == "anchored"
    assert optimizer.last_candidate_prediction is not None
    assert optimizer.last_candidate_prediction.shape == (GRID, GRID)
    assert optimizer.current_phase.shape == (GRID, GRID)


def test_rejects_non_positive_candidate_count():
    with pytest.raises(ValueError):
        _optimizer(n_candidates=0)


def _physical_forward(size=32, fft_size=64, camera_window=13):
    """CPU 友好的小型冻结前向（与 test_differentiable_defocus_uniformity 同口径）。"""
    params = FrozenDefocusParameters(
        defocus_diopter=-0.1, origin_xy=(1360.0, 1020.0),
        amplitude_center_xy=(960.0, 600.0), amplitude_width_xy=(400.0, 300.0),
        residual_noll_radians={5: 0.3}, gain=25.0, background=1.0, sigma_ccd_px=2.0,
        ramp_matrix=((8000.0, 0.0), (0.0, -8000.0)),
        fft_size=fft_size, pixel_samples=1, grid_size=size,
    )
    offsets = np.arange(-(camera_window - 1), camera_window, 2.0)
    xx, yy = np.meshgrid(1360.0 + offsets, 1020.0 + offsets, indexing="xy")
    return FrozenDefocusForward(params, np.stack([xx, yy], axis=-1), device="cpu"), camera_window


def test_real_frozen_forward_accepts_a_batch_of_candidates():
    """冻结物理预测对 (K, H, H) 输入本就批安全：一次批前向给出 K 张强度图。"""
    forward, window = _physical_forward()
    size = forward.params.grid_size
    phase = np.zeros((2, size, size), dtype=np.float32)
    phase[1] = np.linspace(0.0, 0.3, size, dtype=np.float32)[None, :]
    with torch.no_grad():
        images = forward.predict(torch.as_tensor(phase))
    assert tuple(images.shape) == (2, window, window)
    assert torch.isfinite(images).all()
    assert not torch.equal(images[0], images[1])
