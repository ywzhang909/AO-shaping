"""离线吞吐研究: 可微分光束整形的 B>=2 多起点批量并行 + 逐位串行对照。

要回答什么
----------
把 ``DifferentiableBeamOptimizer`` 的「一次前向 / 一次反传 / 一次 Adam step」
从单条 trajectory 扩成一次 batch FFT (B 条 trajectory 共享一次 ``fft2`` 调用),
并**逐位对照**单条串行路径, 证明数值一致。这是纯吞吐优化研究:

* 不改优化器源码, 不改任何默认参数;
* 唯一问题重建入口是 :func:`scripts._common.differentiable_problem.load_problem_from_run_dir`,
  口径与 ``hw_differentiable_phase_opt.py::run_analyze_only`` 一致;
* 全程离线, **不操作 SLM / CCD / 激光**。

正确性要点 (照 ``DifferentiableBeamOptimizer`` 逐项复刻)
-------------------------------------------------------
1. 目标峰值归一化: ``t = target.astype(float32); t_max = float(t.max());
   if t_max > 0: t = t / t_max``; batch 内所有 trajectory 共享同一张目标。
2. 前向: ``field = amp * exp(i*phase)``,
   ``far = fftshift(fft2(field, dim=(-2,-1)), dim=(-2,-1))``,
   ``intensity = far.real**2 + far.imag**2``。
   ⚠️ ``fftshift`` 默认会平移**所有**维度; batch 张量上必须用 ``dim=(-2,-1)``,
   否则 B=2 时 batch 维被平移, 两条 trajectory 被互换, 对照必然失败。
3. **每条 trajectory 独立**峰值归一化 (分母取 ``amax(dim=(-2,-1), keepdim=True)``),
   绝不能全 batch 共用一个 max —— 那会让两条 trajectory 的损失串味。
4. ``losses = ((normalized - t)**2).mean(dim=(-2,-1))``,
   ``losses.sum().backward()`` 一次反传 (逐元素梯度与单条串行一致)。
5. Adam 完全照抄 ``DifferentiableBeamOptimizer.__init__``:
   ``torch.optim.Adam([phase], lr=lr)`` (默认 betas=(0.9,0.999), eps=1e-8,
   weight_decay=0); 状态逐元素, 单个 optimizer 覆盖 (B,M,M) 即等价。
6. 读过完整 ``update()``: 仿真步只有 ``zero_grad(set_to_none=True)`` +
   ``loss.backward()`` + ``optimizer.step()``, 没有梯度裁剪 / step_rms_cap /
   lr schedule, 故批量路径无需额外复刻。

复现门禁
--------
trajectory 0 与 ``build_optimizer(problem, lr, device, init_phase=copy)`` 单条
串行跑相同 iters 对照 (a) 每轮 loss 的 ``max_rel_diff`` 与 (b) 最终相位的
``max_abs_diff``。任一超过 ``--repro-tol`` (默认 1e-4) 即 JSON 标 FAIL、
exit 2, 且**不报告吞吐**。实测若为 0 才写 bit-exact, 有微小差异如实写数值容差。

用法
----
::

    .\\.venv\\Scripts\\python.exe -m scripts.slm.sweep_differentiable_multistart_batch \
        --run-dir data/debug/differentiable_opt/20261116_114424 \
        --iters 120 --device cuda --out <out.json> --figure <out.png>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

#: 仓库根 (scripts/slm/x.py -> parents[2])。
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUN_DIR = ROOT / "data" / "debug" / "differentiable_opt" / "20261116_114424"
DEFAULT_ITERS = 20
DEFAULT_LR = 0.05
DEFAULT_BATCH = 2
DEFAULT_START_SEED = 20261121
DEFAULT_SIGMA = 0.05
DEFAULT_REPRO_TOL = 1e-4


def _ensure_sys_path() -> None:
    """把仓库根 / src / scripts 放进 ``sys.path`` (幂等, 不 import torch)。"""
    for entry in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


_ensure_sys_path()


def _torch_module() -> Any:
    """延迟导入 torch (顶层只有标准库 + numpy)。"""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 环境缺依赖
        raise ImportError(
            "sweep_differentiable_multistart_batch requires PyTorch. "
            "Install with: uv sync --extra ml"
        ) from exc
    return torch


def _resolve_device(device: str, torch: Any) -> Any:
    """解析 torch 设备; cuda 不可用时给出明确提示。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} 但 torch.cuda.is_available()=False; 请用 --device cpu")
    return torch.device(device)


def _sync(torch: Any, device: Any) -> None:
    """计时口径: 每次计时前后做 device synchronize。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# 问题装配 / 初相
# ---------------------------------------------------------------------------


def make_initial_phases(
    init_phase: Any,
    *,
    batch: int,
    seed: int,
    sigma: float,
) -> list[np.ndarray]:
    """生成 B 条**互不共享内存**的 float32 初相。

    trajectory 0 是存档初相本身 (逐位验证用); trajectory b>=1 是
    ``init_phase + sigma * rng.standard_normal(shape)``, 用单个
    ``np.random.default_rng(seed)`` 按序生成, 完全确定性。

    每条都 ``np.array(..., dtype=np.float32, copy=True)``:
    ``DifferentiableBeamOptimizer`` 用 ``np.asarray`` 不复制, Adam 会原地
    改写传入数组, 多起点共用内存必然污染。

    Args:
        init_phase: 存档初相 (N×N 或 M×M)。
        batch: trajectory 条数, 至少 1。
        seed: 扰动 RNG 种子。
        sigma: 扰动标准差 (弧度)。

    Returns:
        长度为 ``batch`` 的独立 float32 数组列表。
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    base = np.asarray(init_phase, dtype=np.float32)
    rng = np.random.default_rng(seed)
    phases: list[np.ndarray] = [np.array(base, dtype=np.float32, copy=True)]
    for _ in range(1, batch):
        perturbed = base + sigma * rng.standard_normal(base.shape).astype(np.float32)
        phases.append(np.array(perturbed, dtype=np.float32, copy=True))
    return phases


def initial_phase_info(phases: list[np.ndarray]) -> dict[str, Any]:
    """记录两条初相的统计量与彼此 RMS 差 (证明不是重复计算)。"""
    out: dict[str, Any] = {
        "count": len(phases),
        "shape": list(phases[0].shape),
        "dtype": str(phases[0].dtype),
        "trajectories": [
            {
                "index": i,
                "rms": float(np.sqrt(np.mean(p.astype(np.float64) ** 2))),
                "peak_abs": float(np.max(np.abs(p.astype(np.float64)))),
            }
            for i, p in enumerate(phases)
        ],
    }
    if len(phases) >= 2:
        delta = phases[0].astype(np.float64) - phases[1].astype(np.float64)
        out["rms_diff_0_vs_1"] = float(np.sqrt(np.mean(delta**2)))
        out["distinct_0_vs_1"] = bool(np.any(delta != 0.0))
    return out


# ---------------------------------------------------------------------------
# 批量前向 / 损失 (关键正确性路径)
# ---------------------------------------------------------------------------


def _amax_values(values: Any, torch: Any) -> Any:
    """取 ``amax(dim=..., keepdim=True)`` 的值张量。

    torch 2.14 返回普通 Tensor (其 ``.values`` 是稀疏张量方法, 调用会抛错);
    老版本/其他 reduction 可能返回带 ``.values`` 的 namedtuple。两种都兼容。
    """
    result = values
    if not isinstance(result, torch.Tensor) and hasattr(result, "values"):
        result = result.values
    return result


def prepare_tensors(
    amplitude: Any,
    target: Any,
    *,
    device: Any,
    torch: Any,
) -> tuple[Any, Any]:
    """把振幅与目标转成 device 上的 float32 张量 (目标做峰值归一化)。"""
    amp_np = np.ascontiguousarray(np.asarray(amplitude, dtype=np.float32))
    amp_t = torch.from_numpy(amp_np).to(device)
    t = np.asarray(target).astype(np.float32)
    t_max = float(t.max())
    if t_max > 0:
        t = t / t_max
    tgt_t = torch.from_numpy(np.ascontiguousarray(t)).to(device)
    return amp_t, tgt_t


def batch_forward(
    amplitude_t: Any,
    phase_t: Any,
    target_t: Any,
    *,
    torch: Any,
) -> tuple[Any, Any, Any]:
    """批量 (或单条) 可微前向 + 逐 trajectory 峰值归一化 MSE。

    Args:
        amplitude_t: (M,M) 源面振幅张量。
        phase_t: (B,M,M) 相位张量 (或单条 (M,M))。
        target_t: (M,M) 峰值归一化目标张量, batch 内共享。
        torch: 已导入的 torch 模块。

    Returns:
        ``(losses, normalized, intensity)``; ``losses`` 形状为 (B,) (单条输入
        时为 0 维标量), ``normalized`` / ``intensity`` 与 ``phase_t`` 同形状。
    """
    field = amplitude_t * torch.exp(1j * phase_t)
    # fft2 默认就是最后两维, 但显式写明; fftshift 必须显式 dim, 否则 batch 维
    # 也会被平移 (B=2 时两条 trajectory 互换)。
    far = torch.fft.fftshift(torch.fft.fft2(field, dim=(-2, -1)), dim=(-2, -1))
    intensity = far.real**2 + far.imag**2
    # 每条 trajectory 独立峰值归一化 —— 绝不对全 batch 取一个 max。
    peak = _amax_values(intensity.amax(dim=(-2, -1), keepdim=True), torch)
    normalized = intensity / (peak + 1e-8)
    losses = ((normalized - target_t) ** 2).mean(dim=(-2, -1))
    return losses, normalized, intensity


def make_batch_optimizer(phase_t: Any, lr: float, *, torch: Any) -> Any:
    """逐项复刻 ``DifferentiableBeamOptimizer`` 的 Adam 构造 (无裁剪/调度)。"""
    return torch.optim.Adam([phase_t], lr=lr)


def batch_step(
    optimizer: Any,
    phase_t: Any,
    amplitude_t: Any,
    target_t: Any,
    *,
    torch: Any,
    keep_losses: bool = False,
) -> Any | None:
    """一次 zero_grad + 批量前向 + 一次反传 + 一次 Adam step。

    Returns:
        ``keep_losses`` 为 True 时返回本步 (B,) 损失的 detach 副本; 否则 None。
    """
    optimizer.zero_grad(set_to_none=True)
    losses, _, _ = batch_forward(amplitude_t, phase_t, target_t, torch=torch)
    if keep_losses:
        recorded = losses.detach().clone()
    else:
        recorded = None
    losses.sum().backward()
    optimizer.step()
    return recorded


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------


def run_batch(
    problem: Any,
    initial_phases: list[np.ndarray],
    *,
    lr: float,
    iters: int,
    device: Any,
    torch: Any,
    warmup: int = 0,
) -> dict[str, Any]:
    """B 条 trajectory 共用一次 FFT / 一次反传 / 一个 Adam 的完整循环。"""
    amp_t, tgt_t = prepare_tensors(problem.amplitude, problem.target, device=device, torch=torch)
    stacked = np.array(np.stack(initial_phases), dtype=np.float32, copy=True)
    phase_t = torch.from_numpy(stacked).to(device)
    phase_t.requires_grad_(True)
    optimizer = make_batch_optimizer(phase_t, lr, torch=torch)
    # 按 trajectory 收集: trace[b] 是第 b 条的逐轮 loss。
    traces: list[list[float]] = [[] for _ in initial_phases]
    for i in range(iters + warmup):
        rec = batch_step(optimizer, phase_t, amp_t, tgt_t, torch=torch, keep_losses=i >= warmup)
        if rec is not None:
            values = rec.reshape(-1).tolist()
            for b, v in enumerate(values):
                traces[b].append(float(v))
    final = phase_t.detach().cpu().numpy()
    return {
        "losses": traces,
        "final_phases": [np.array(final[b], dtype=np.float32, copy=True) for b in range(len(initial_phases))],
    }


def run_serial(
    problem: Any,
    initial_phases: list[np.ndarray],
    *,
    lr: float,
    iters: int,
    device: Any,
    torch: Any,
    indices: tuple[int, ...] | None = None,
    warmup: int = 0,
) -> dict[str, Any]:
    """单条串行路径: 每条都用 ``build_optimizer`` + ``update()`` 逐次跑。"""
    del torch  # build_optimizer 内部延迟导入
    from scripts._common.differentiable_problem import build_optimizer

    wanted = tuple(range(len(initial_phases))) if indices is None else indices
    losses: dict[int, list[float]] = {}
    finals: dict[int, np.ndarray] = {}
    for b in wanted:
        # 与 make_initial_phases 同理: 显式副本, 防止 Adam 原地改写调用方数组。
        phase_copy = np.array(initial_phases[b], dtype=np.float32, copy=True)
        opt = build_optimizer(problem, lr=lr, device=str(device), init_phase=phase_copy)
        for i in range(iters + warmup):
            opt.update()
            if i >= warmup:
                losses.setdefault(b, []).append(float(opt.loss_history[-1]))
        finals[b] = np.array(opt.current_phase, dtype=np.float32, copy=True)
    return {"losses": losses, "final_phases": finals}


def _max_rel_diff(a: list[float], b: list[float]) -> float:
    """逐轮 loss 的最大相对差 (分母加 1e-12 下限防止除零)。"""
    if len(a) != len(b):
        raise ValueError(f"loss trace length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    return max(abs(float(x) - float(y)) / max(abs(float(y)), 1e-12) for x, y in zip(a, b))


def verify_repro(
    batch_result: dict[str, Any],
    serial_result: dict[str, Any],
    *,
    tol: float,
) -> dict[str, Any]:
    """trajectory 0 的 batch vs 单条串行逐位对照。"""
    batch_losses = batch_result["losses"][0]
    serial_losses = serial_result["losses"][0]
    rel = _max_rel_diff(batch_losses, serial_losses)
    abs_phase = float(
        np.max(
            np.abs(
                batch_result["final_phases"][0].astype(np.float64)
                - serial_result["final_phases"][0].astype(np.float64)
            )
        )
    )
    passed = bool(rel <= tol and abs_phase <= tol)
    exact = bool(rel == 0.0 and abs_phase == 0.0)
    return {
        "passed": passed,
        "tolerance": float(tol),
        "loss_max_rel_diff": float(rel),
        "final_phase_max_abs_diff": abs_phase,
        "bit_exact": exact,
        "verdict": (
            "bit-exact: batch trajectory 0 与单条串行逐位相同"
            if exact
            else ("PASS: 差异在数值容差内" if passed else "FAIL: 差异超出容差")
        ),
        "serial_loss_trace": serial_losses,
        "batch_loss_trace": batch_losses,
    }


# ---------------------------------------------------------------------------
# 吞吐
# ---------------------------------------------------------------------------


def measure_throughput(
    problem: Any,
    initial_phases: list[np.ndarray],
    *,
    lr: float,
    iters: int,
    device: Any,
    torch: Any,
    warmup_steps: int = 3,
) -> dict[str, Any]:
    """B 个独立 optimizer 串行 vs 1 个 batch optimizer 的 ms/step 与 fields/s。"""
    b = len(initial_phases)

    from scripts._common.differentiable_problem import build_optimizer

    # ---- 计时对象 (batch 与 B 个串行 optimizer) ----
    amp_t, tgt_t = prepare_tensors(problem.amplitude, problem.target, device=device, torch=torch)
    stacked = np.array(np.stack(initial_phases), dtype=np.float32, copy=True)
    phase_t = torch.from_numpy(stacked).to(device)
    phase_t.requires_grad_(True)
    batch_opt = make_batch_optimizer(phase_t, lr, torch=torch)
    serial_opts = [
        build_optimizer(
            problem, lr=lr, device=str(device), init_phase=np.array(p, dtype=np.float32, copy=True)
        )
        for p in initial_phases
    ]

    # ---- warmup: 直接在计时对象上跑, 预热 cuFFT plan / CUDA allocator / Adam 状态 ----
    for _ in range(warmup_steps):
        batch_step(batch_opt, phase_t, amp_t, tgt_t, torch=torch)
        for opt in serial_opts:
            opt.update()

    # ---- batch 计时 (仅后续 iters 步, 稳态) ----
    _sync(torch, device)
    t0 = time.perf_counter()
    for _ in range(iters):
        batch_step(batch_opt, phase_t, amp_t, tgt_t, torch=torch)
    _sync(torch, device)
    batch_s = time.perf_counter() - t0

    # 串行计时: 复用上面已 warmup 的 B 个 optimizer, 跑 iters 步。
    _sync(torch, device)
    t0 = time.perf_counter()
    for _ in range(iters):
        for opt in serial_opts:
            opt.update()
    _sync(torch, device)
    serial_s = time.perf_counter() - t0

    total_fields = b * iters
    serial_ms_step = serial_s / total_fields * 1000.0
    batch_ms_step = batch_s / iters * 1000.0
    serial_fps = total_fields / serial_s if serial_s > 0 else float("inf")
    batch_fps = total_fields / batch_s if batch_s > 0 else float("inf")
    return {
        "batch_size": b,
        "iters": iters,
        "total_fields": total_fields,
        "warmup_steps": warmup_steps,
        "serial": {
            "wall_s": float(serial_s),
            "ms_per_step_per_field": float(serial_ms_step),
            "fields_per_s": float(serial_fps),
        },
        "batch": {
            "wall_s": float(batch_s),
            "ms_per_step_batch": float(batch_ms_step),
            "ms_per_step_per_field": float(batch_ms_step / b),
            "fields_per_s": float(batch_fps),
        },
        "speedup_fields_per_s": float(batch_fps / serial_fps) if serial_fps > 0 else None,
        "step_time_ratio_serial_over_batch": float(serial_ms_step / (batch_ms_step / b))
        if batch_ms_step > 0
        else None,
        "note": "warmup + device synchronize 前后计时; 串行跑 B 次以对齐总 field 数",
    }


# ---------------------------------------------------------------------------
# 图 / JSON
# ---------------------------------------------------------------------------


def render_figure(path: Path, batch_losses: list[list[float]], out: dict[str, Any]) -> None:
    """画逐轮 loss 图; CJK 字体探测, 找不到就退回英文标签 + DejaVu Sans。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from scripts._common.plot_style import setup_cjk_font

        chosen = setup_cjk_font(probe_installed=True)
    except Exception:  # pragma: no cover - 字体模块缺失不应中断产物
        chosen = None
    if chosen is None:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        labels = [f"trajectory {i}" for i in range(len(batch_losses))]
        title = "Differentiable beam shaping: multi-start batch vs serial"
        xlabel, ylabel = "iteration", "loss (batch, per trajectory)"
    else:
        labels = [f"trajectory {i}" for i in range(len(batch_losses))]
        title = "可微分光束整形: 多起点批量轨迹"
        xlabel, ylabel = "迭代", "损失 (batch, 逐 trajectory)"

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for b, losses in enumerate(batch_losses):
        ax.plot(range(1, len(losses) + 1), losses, label=labels[b], linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 参数。"""
    p = argparse.ArgumentParser(
        description="纯离线: 可微分光束整形多起点 batch FFT 并行 + 逐位串行对照 (不碰任何设备)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="问题重建用的运行目录")
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="每条 trajectory 的迭代次数")
    p.add_argument("--lr", type=float, default=DEFAULT_LR, help="Adam 学习率")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="trajectory 条数")
    p.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED, help="扰动初相 RNG 种子")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="扰动标准差 (rad)")
    p.add_argument("--device", default="cuda", help="torch 设备字符串 (cuda / cpu)")
    p.add_argument("--out", type=Path, required=True, help="JSON 产物路径 (必填)")
    p.add_argument("--figure", type=Path, default=None, help="可选 loss 曲线 PNG 路径")
    p.add_argument(
        "--repro-tol",
        type=float,
        default=DEFAULT_REPRO_TOL,
        help="batch 与串行对照的容差 (超过即 FAIL, exit 2)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """入口: 重建 -> 初相 -> batch -> 串行对照 -> (通过才) 吞吐 -> 产物。"""
    from loguru import logger

    args = build_arg_parser().parse_args(argv)
    if args.iters < 1:
        raise ValueError(f"--iters must be >= 1, got {args.iters}")
    if args.sigma < 0:
        raise ValueError(f"--sigma must be >= 0, got {args.sigma}")

    torch = _torch_module()
    device = _resolve_device(args.device, torch)

    from scripts._common.differentiable_problem import load_problem_from_run_dir

    logger.info("重建问题: {}", args.run_dir)
    problem = load_problem_from_run_dir(args.run_dir)
    phases = make_initial_phases(
        problem.init_phase, batch=args.batch, seed=args.start_seed, sigma=args.sigma
    )
    phase_info = initial_phase_info(phases)
    logger.info(
        "trajectory={} 初相 RMS 差(0 vs 1)={}",
        phase_info["count"],
        phase_info.get("rms_diff_0_vs_1"),
    )

    t0 = time.perf_counter()
    batch_result = run_batch(
        problem, phases, lr=args.lr, iters=args.iters, device=device, torch=torch
    )
    logger.info("batch 完成: {} iters x {} trajectory", args.iters, args.batch)

    serial_result = run_serial(
        problem, phases, lr=args.lr, iters=args.iters, device=device, torch=torch, indices=(0,)
    )
    verify = verify_repro(batch_result, serial_result, tol=args.repro_tol)
    logger.info(
        "复现对照: rel_diff={} phase_abs_diff={} passed={}",
        verify["loss_max_rel_diff"],
        verify["final_phase_max_abs_diff"],
        verify["passed"],
    )

    final_stats = [
        {
            "index": b,
            "rms": float(np.sqrt(np.mean(p.astype(np.float64) ** 2))),
            "peak_abs": float(np.max(np.abs(p.astype(np.float64)))),
        }
        for b, p in enumerate(batch_result["final_phases"])
    ]

    out: dict[str, Any] = {
        "config": {
            "run_dir": str(args.run_dir.resolve()),
            "iters": args.iters,
            "lr": args.lr,
            "batch": args.batch,
            "start_seed": args.start_seed,
            "sigma": args.sigma,
            "device": str(device),
            "repro_tol": args.repro_tol,
            "problem_shape": list(np.asarray(problem.amplitude).shape),
        },
        "verify": verify,
        "initial_phases": phase_info,
        "losses_per_iter": batch_result["losses"],
        "final_phase_stats": final_stats,
        "throughput": None,
        "wall_s": 0.0,
        "conclusion": "",
    }

    if not verify["passed"]:
        out["conclusion"] = (
            f"FAIL: batch trajectory 0 与单条串行差异超出 tol={args.repro_tol} "
            f"(loss rel={verify['loss_max_rel_diff']:.3e}, "
            f"phase abs={verify['final_phase_max_abs_diff']:.3e}); 未测量吞吐。"
        )
        out["wall_s"] = float(time.perf_counter() - t0)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        logger.error(out["conclusion"])
        return 2

    throughput = measure_throughput(
        problem, phases, lr=args.lr, iters=args.iters, device=device, torch=torch
    )
    out["throughput"] = throughput
    out["wall_s"] = float(time.perf_counter() - t0)
    speedup = throughput["speedup_fields_per_s"]
    out["conclusion"] = (
        f"PASS: batch trajectory 0 对照单条串行 {verify['verdict']} "
        f"(loss rel={verify['loss_max_rel_diff']:.3e}, "
        f"phase abs={verify['final_phase_max_abs_diff']:.3e}); "
        f"B={args.batch} batch {throughput['batch']['fields_per_s']:.1f} fields/s vs "
        f"串行 {throughput['serial']['fields_per_s']:.1f} fields/s, speedup={speedup:.3f}x."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("JSON 写入: {}", args.out)
    if args.figure is not None:
        render_figure(args.figure, batch_result["losses"], out)
        logger.info("图写入: {}", args.figure)
    logger.info(out["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
