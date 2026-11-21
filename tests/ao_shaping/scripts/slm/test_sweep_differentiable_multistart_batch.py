"""合成小网格测试: 多起点 batch FFT 可微光束优化 + 逐位串行对照 (离线, 无设备)。

按路径 importlib 加载脚本 (与 ``test_replay_diff_phase_monotonicity.py`` 同写法),
覆盖纯函数: 每条独立峰值归一化、多起点确定性、batch vs 单条前向一致、
batch vs 串行多轮 loss/相位一致、Adam 逐元素独立、初相 copy 正确性、
目标归一化与损失公式、CLI 参数解析。所有测试用 8×8 小网格, 强制 ``--device cpu``。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "slm" / "sweep_differentiable_multistart_batch.py"


def _load_script():
    """从文件路径加载脚本 (scripts/slm 不是稳定的包导入路径)。"""
    spec = importlib.util.spec_from_file_location("sweep_differentiable_multistart_batch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script()

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# 小合成问题
# ---------------------------------------------------------------------------


def _make_problem(seed: int = 7, n: int = 8) -> SimpleNamespace:
    """合成一个带 amplitude/target/init_phase 的问题对象。"""
    rng = np.random.default_rng(seed)
    amp = np.ones((n, n), dtype=np.float32)
    target = rng.random((n, n)).astype(np.float32) * 3.0
    target[n // 2 - 1 : n // 2 + 1, n // 2 - 1 : n // 2 + 1] = 7.0  # 峰值 (测试归一化)
    init = rng.standard_normal((n, n)).astype(np.float32) * 0.2
    return SimpleNamespace(amplitude=amp, target=target, init_phase=init)


def _device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. 每条独立峰值归一化 (不同峰值 → 归一化后各自峰值为 1, 不串味)
# ---------------------------------------------------------------------------


def test_per_trajectory_peak_normalization() -> None:
    amp = torch.ones(2, 8, 8)
    phase = torch.zeros(2, 8, 8)
    # 手动构造强度峰值相差 100 倍的 batch: 用振幅缩放。
    amp[1] *= 10.0
    target = torch.zeros(8, 8)
    target[4, 4] = 1.0
    _, normalized, intensity = mod.batch_forward(amp, phase, target, torch=torch)
    peaks = intensity.amax(dim=(-2, -1))
    assert float(peaks[1] / peaks[0]) == pytest.approx(100.0, rel=1e-3)
    # 归一化后每条峰值都应为 1 (float32 + eps 精度)
    for b in range(2):
        assert float(normalized[b].amax()) == pytest.approx(1.0, abs=1e-5)
    # 关键: 不能用全 batch 一个 max, 否则弱的那条峰值会是 0.01。
    assert float(normalized[0].amax()) > 0.99


# ---------------------------------------------------------------------------
# 2. 多起点确定性 + 两条初相不同
# ---------------------------------------------------------------------------


def test_multistart_deterministic_and_distinct() -> None:
    problem = _make_problem()
    a = mod.make_initial_phases(problem.init_phase, batch=2, seed=123, sigma=0.05)
    b = mod.make_initial_phases(problem.init_phase, batch=2, seed=123, sigma=0.05)
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa, pb)
    assert a[0].dtype == np.float32
    info = mod.initial_phase_info(a)
    assert info["rms_diff_0_vs_1"] > 0.0
    assert info["distinct_0_vs_1"] is True
    assert info["count"] == 2


# ---------------------------------------------------------------------------
# 3. batch 前向 vs 单条前向数值一致 (小网格)
# ---------------------------------------------------------------------------


def test_batch_forward_matches_single_forward() -> None:
    problem = _make_problem(seed=11)
    amp_np = problem.amplitude
    tgt_np = problem.target.astype(np.float32) / float(problem.target.max())
    amp_t, tgt_t = mod.prepare_tensors(amp_np, tgt_np, device=_device(), torch=torch)
    phases = mod.make_initial_phases(problem.init_phase, batch=2, seed=5, sigma=0.1)
    stacked = torch.from_numpy(np.stack(phases)).to(_device())

    losses_b, norm_b, int_b = mod.batch_forward(amp_t, stacked, tgt_t, torch=torch)
    for b in range(2):
        single = torch.from_numpy(phases[b]).to(_device())
        l_s, n_s, i_s = mod.batch_forward(amp_t, single, tgt_t, torch=torch)
        assert torch.equal(int_b[b], i_s)
        assert torch.equal(norm_b[b], n_s)
        assert float(losses_b[b]) == pytest.approx(float(l_s), abs=1e-7)


# ---------------------------------------------------------------------------
# 4. batch vs 串行多轮 loss/相位一致 (小网格, 容差内)
# ---------------------------------------------------------------------------


def test_batch_matches_serial_over_iterations() -> None:
    problem = _make_problem(seed=21)
    phases = mod.make_initial_phases(problem.init_phase, batch=2, seed=9, sigma=0.05)
    iters = 6
    batch_res = mod.run_batch(
        problem, phases, lr=0.05, iters=iters, device=_device(), torch=torch
    )
    serial_res = mod.run_serial(
        problem, phases, lr=0.05, iters=iters, device=_device(), torch=torch, indices=(0,)
    )
    verify = mod.verify_repro(batch_res, serial_res, tol=1e-4)
    assert verify["passed"] is True, verify["verdict"]
    assert len(verify["batch_loss_trace"]) == iters
    assert verify["loss_max_rel_diff"] <= 1e-4
    assert verify["final_phase_max_abs_diff"] <= 1e-4


# ---------------------------------------------------------------------------
# 5. Adam 逐元素独立: 一条 trajectory 的梯度不污染另一条
# ---------------------------------------------------------------------------


def test_adam_elements_are_independent() -> None:
    problem = _make_problem(seed=31)
    phases = mod.make_initial_phases(problem.init_phase, batch=2, seed=3, sigma=0.5)
    iters = 4
    batch_res = mod.run_batch(
        problem, phases, lr=0.05, iters=iters, device=_device(), torch=torch
    )
    # 单条 trajectory 1 单独跑, 结果应与 batch 中的 trajectory 1 逐位一致。
    serial_res = mod.run_serial(
        problem, phases, lr=0.05, iters=iters, device=_device(), torch=torch, indices=(1,)
    )
    np.testing.assert_allclose(
        batch_res["final_phases"][1],
        serial_res["final_phases"][1],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        batch_res["losses"][1], serial_res["losses"][1], rtol=1e-4, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 6. init_phase 不被原地改写 (copy 正确性)
# ---------------------------------------------------------------------------


def test_init_phase_not_mutated() -> None:
    problem = _make_problem(seed=41)
    original = np.array(problem.init_phase, dtype=np.float32, copy=True)
    phases = mod.make_initial_phases(problem.init_phase, batch=2, seed=1, sigma=0.2)
    snapshots = [np.array(p, copy=True) for p in phases]
    mod.run_batch(problem, phases, lr=0.05, iters=3, device=_device(), torch=torch)
    np.testing.assert_array_equal(problem.init_phase, original)
    for p, snap in zip(phases, snapshots):
        np.testing.assert_array_equal(p, snap)


# ---------------------------------------------------------------------------
# 7. 目标归一化 + 损失公式数值
# ---------------------------------------------------------------------------


def test_target_normalization_and_loss_formula() -> None:
    problem = _make_problem(seed=51)
    amp_t, tgt_t = mod.prepare_tensors(
        problem.amplitude, problem.target, device=_device(), torch=torch
    )
    # 目标峰值归一化到 1。
    assert float(tgt_t.max()) == pytest.approx(1.0, abs=1e-6)
    # 平相位前向: 手工复刻损失公式并对照。
    phase = torch.zeros(8, 8)
    losses, normalized, _ = mod.batch_forward(amp_t, phase, tgt_t, torch=torch)
    expected = float(((normalized - tgt_t) ** 2).mean())
    assert float(losses) == pytest.approx(expected, rel=1e-6, abs=1e-7)


# ---------------------------------------------------------------------------
# 8. CLI 参数解析 / 缺 --out 报错
# ---------------------------------------------------------------------------


def test_arg_parser_defaults_and_required_out() -> None:
    parser = mod.build_arg_parser()
    args = parser.parse_args(["--out", "x.json"])
    assert args.batch == 2
    assert args.iters == mod.DEFAULT_ITERS
    assert args.device == "cuda"
    assert args.repro_tol == mod.DEFAULT_REPRO_TOL
    with pytest.raises(SystemExit):
        parser.parse_args([])  # 缺 --out 应报错
