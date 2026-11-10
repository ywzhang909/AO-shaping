"""可微优化的 CPU 调度探针（离线；不占用任何设备）。

问题来源：``aperture_closed_shaping_20261109.md`` §5.2 用 torch.profiler 拆出
「每步 ~19 ms 是 Python 逐算子调度、GPU 实算仅 ~4.6 ms」，并把「合并 roll+FFT /
粗网格 fft256」列为下一步方向。2026-11-10 的复测证明那个方向不对：FFT/roll 在
一次 ``update()`` 里合计只有 ~0.1 ms，真正的开销是**每次迭代的同步次数与算子
发射次数**。本探针把这两件事测成可复算的数字。

测项（同一脚本、同一冻结 checkpoint，A/B 两棵源码树）：

* **墙钟**：预热后每次 ``update()`` 的中位/p10/p90（GPU 侧每次迭代显式
  ``torch.cuda.synchronize()``，报的是真实墙钟而不是队列时间）；
* **算子混合**：一次 ``update()`` 里的 kernel launch / stream sync / memcpy /
  ``nonzero`` 次数与总算子数——「调度主导」这一结论的量化依据；
* **逐位性**：``ShapeObjective.evaluate`` 与其布尔掩膜原型的逐位对照在
  ``tests/ao_shaping/algorithm/test_multishape_objective_bit_exact.py``（CI 门禁），
  本脚本只负责耗时侧。

A/B 两棵树的取法（``--src``）::

    git archive --format=tar --output=head.tar HEAD src
    tar -xf head.tar -C <dir>
    python scripts/slm/diffopt_dispatch_probe_20261110.py --src <dir>/src --label head
    python scripts/slm/diffopt_dispatch_probe_20261110.py --label worktree

设备纪律：全程只读磁盘上的冻结 checkpoint 与代码；未打开 SLM / 相机 / 激光，
未 import 任何 ``ao_shaping.drivers`` 模块。产物写 ``data/debug/``（不入库）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]


def _src_root(argv: list[str]) -> Path:
    """``--src`` 指定的源码树（默认仓库 ``src``）。

    必须在 import ``ao_shaping`` 之前决定，所以这里直接读 ``sys.argv``。
    """
    for i, arg in enumerate(argv):
        if arg == "--src" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if arg.startswith("--src="):
            return Path(arg.split("=", 1)[1]).resolve()
    return (ROOT / "src").resolve()


SRC = _src_root(sys.argv[1:])
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_shaping.algorithm.signal_processing import differentiable_defocus_uniformity as dd
from ao_shaping.algorithm.signal_processing import multishape_objective as mo

DEVICE_STATEMENT = (
    "离线分析，未占用任何设备：未打开 SLM / 相机 / 激光，未 import 任何 "
    "ao_shaping.drivers 模块；全部数字来自磁盘上的冻结 checkpoint，"
    "同一脚本、同一输入，GPU 侧每次迭代显式 synchronize 后计时。"
)

DEFAULT_OUT = ROOT / "data" / "debug" / "slm_transfer" / "diffopt_dispatch_20261110"
DEFAULT_CHECKPOINT = (ROOT / "data" / "debug" / "slm_transfer"
                      / "repro1200_aperture_closed_20261109_1910" / "fit")
SHAPE = "square"
SIDE = 1200.0
ZERO_R = 60.0

#: 调度侧的关键算子：次数本身就是结论（每次同步都要等 GPU 排空队列）。
KEY_OPS = ("cudaLaunchKernel", "cudaStreamSynchronize", "cudaMemcpyAsync",
           "aten::nonzero", "aten::masked_select", "aten::index_select",
           "aten::item", "aten::_local_scalar_dense")


def _setup(device: str, checkpoint: Path):
    z = np.load(checkpoint / "inputs.npz", allow_pickle=True)
    xy = z["camera_xy"]
    params = dd.load_frozen_parameters(checkpoint, fft_size=1024)
    forward = dd.FrozenDefocusForward(params, xy, device=device)
    centre = tuple(params.origin_xy)
    target = mo.target_at(xy, SHAPE, centre, SIDE)
    objective = mo.ShapeObjective(target, cv_only=True)
    radius = np.hypot(xy[..., 0] - centre[0], xy[..., 1] - centre[1])
    objective.core &= radius > ZERO_R
    seed = mo.gs_seed(forward, centre, SIDE, SHAPE)
    return forward, objective, seed


def _wall_times(opt: Any, device: str, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        opt.update(None)
    if device == "cuda":
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        opt.update(None)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _op_mix(opt: Any, device: str, iters: int = 10) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(iters):
            opt.update(None)
        if device == "cuda":
            torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    for event in prof.key_averages():
        cuda_us = getattr(event, "self_device_time_total", 0) or 0
        cpu_us = getattr(event, "self_cpu_time_total", 0) or 0
        if cuda_us <= 0 and cpu_us <= 0:
            continue
        rows.append({"op": event.key, "count_per_iter": event.count / iters,
                     "cuda_self_ms_per_iter": cuda_us / iters / 1e3,
                     "cpu_self_ms_per_iter": cpu_us / iters / 1e3})
    rows.sort(key=lambda r: -(r["cuda_self_ms_per_iter"] + r["cpu_self_ms_per_iter"]))
    counts = {r["op"]: r["count_per_iter"] for r in rows}
    key_counts = {name: counts.get(name, 0.0) for name in KEY_OPS}
    key_counts["total_ops"] = sum(r["count_per_iter"] for r in rows)
    return key_counts, rows[:30]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=None,
                        help="源码树（默认仓库 src）；A/B 时指向另一次检出")
    parser.add_argument("--label", default="worktree", help="本次测量的标签（写进结果）")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--iters", type=int, default=30, help="计时迭代数")
    parser.add_argument("--warmup", type=int, default=10, help="预热迭代数")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="冻结 checkpoint 目录（默认 11-09 复现件的 fit/）")
    parser.add_argument("--out", default=str(DEFAULT_OUT / "results.json"),
                        help="结果 JSON 路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / "inputs.npz").exists():
        logger.error("checkpoint 不存在：{}", checkpoint)
        return 2
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    forward, objective, seed = _setup(device, checkpoint)
    opt = dd.DifferentiableUniformityOptimizer(
        forward, objective, seed, lr=0.01, step_rms_cap=0.1, device=device)

    times = _wall_times(opt, device, args.warmup, args.iters)
    key_counts, ops = _op_mix(opt, device)

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "src": str(SRC),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "checkpoint": str(checkpoint),
        "iters": args.iters,
        "warmup": args.warmup,
        "wall_ms": {"median": statistics.median(times),
                    "mean": statistics.fmean(times),
                    "p10": float(np.percentile(times, 10)),
                    "p90": float(np.percentile(times, 90))},
        "op_counts_per_iter": key_counts,
        "ops": ops,
        "device_statement": DEVICE_STATEMENT,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[{}] {} median {:.2f} ms/步 | launch {:.0f} sync {:.0f} memcpy {:.0f} nonzero {:.0f} | 算子总数 {:.1f}",
                args.label, device, result["wall_ms"]["median"],
                key_counts["cudaLaunchKernel"], key_counts["cudaStreamSynchronize"],
                key_counts["cudaMemcpyAsync"], key_counts["aten::nonzero"],
                key_counts["total_ops"])
    logger.info("结果写入 {}", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
