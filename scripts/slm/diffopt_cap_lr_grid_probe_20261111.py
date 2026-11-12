"""``step_rms_cap`` × ``lr`` 网格探针（离线、CPU；不占用任何设备）。

要回答的问题（本探针存在的唯一理由）
--------------------------------------------------
在**模块原生** ``DifferentiableUniformityOptimizer.update()`` 这条部署路径上，
``step_rms_cap`` 与 ``lr`` 各自要付多少**步数**、多少次 **``accept()`` 确认**，
才能把核心 CV 打到 ≤0.20 / ≤0.15 / ≤0.10。

20261108 的探针只扫了 lr（cap 恒 0.1，accept 只有「从不」与「每步」两个极端），
回答不了「cap 提一档能省多少确认周期」。本脚本把两个参数做满 4×4 网格，并把
accept 语义拆成三个变体。

三个变体语义（本报告的核心区分，读表前必须知道）
--------------------------------------------------
``update()`` 内部把 ``delta - accepted`` 投影到照明加权 RMS ≤ ``cap``
（``_project_increment``），投影参照的是**锚点** ``accepted``，而锚点只由外部
``accept()`` 推进 —— 于是「谁在什么时候 accept」决定 cap 到底在限制什么：

* ``cumulative`` —— 从不 ``accept()``。锚点恒为起点，投影把**总**偏移永久钉在
  cap 球内：走完任意多步都有 ``weighted_rms(delta) ≤ cap``，相位行程被 cap 封顶。
  这是「完全没有实测确认」时的可行域上界（模型侧一步不浪费，但硬件上没有任何一次
  CCD 测量为这些步背书）。该变体的 accept 次数按定义恒为 0。
* ``per_step`` —— 每步都 ``accept()``。锚点随行走，cap 只限**单步增量**，相位行程
  可达 ``cap × 步数``。这是「实测每一步都变好」的乐观上界；真实闭环不可能每步白拿
  一次确认，所以该变体的 accept 次数 = 步数，是确认成本的**上**界。
* ``gated_oracle`` —— 仅当本步 ``candidate_model_loss < anchor_model_loss``（冻结
  模型自评真的变好）才 ``accept()``。**这一个变体的 accept 次数才是硬件上「需要多少次
  实测确认」的直接代理**：它假设判定信号无噪声（oracle），给出确认成本的下界；真实
  闭环用实测 CV + 噪声容差判（``shape_transition_timing_20261106.py`` 的
  ``measured_cv < best_cv`` 门），只会更贵，不会更便宜。

三个变体共用同一条原生 ``update()``（同前向、同损失、同「先去 piston/倾斜再投影」次序），
唯一区别是 ``accept()`` 的调用条件 —— 所以「步数」列三个变体可直接互比，「确认数」列
读的是 ``gated_oracle``。

口径（逐字沿用 ``diffopt_efficiency_probe_20261108.py``，不自创）
--------------------------------------------------
* checkpoint 与光路配置：``offline_benchmark_20261107._TRANSFER.load_fit()`` +
  ``make_forward(fit, -1.0)``（``fft_size=512``、``grid_size=_TRANSFER.GRID``）；
  不留第二份历史光路配置。
* 掩膜：零相位基场 ``core = base >= 0.1 * base.max()``，
  ``UniformityObjective(mask_box=ones, mask_core=core, w_shape=0.0, w_phase=0.0)``。
* CV：取 ``loss_history[-1]`` 的 ``l_cv``（无则 ``loss``）再开方 —— 与该脚本 :420
  同一取法（``l_cv`` 是 CV²）。注意 ``l_cv`` 在该步 **Adam 移动之前** 的 ``delta``
  上评估，于是「第 i 步记录的 ``l_cv``」= 「走完 i−1 步之后的 CV」；本脚本按
  **已走步数** 对齐成 ``cv_after[k]``（k = 0 即起点 CV），末位再用最后一步记录里的
  ``candidate_model_loss``（投影**之后**的 delta）补上，使「走完全部步数的 CV」不缺席。
  步数结论因此不会被这个错位污染。
* 参数化：``bench.C_MODES``（Noll 5–15，11 模，piston/整体倾斜不进指令）；起点 =
  零网格 + 1 个每系数 ~ ``U(−bench.C_RANDOM_RAD, +bench.C_RANDOM_RAD)`` rad 的随机
  像差（``build_basis``/``command_of`` 与 20261108 同一构造，种子固定并写进
  ``settings``，与遍历顺序无关）。

计时纪律（本脚本的结论边界）
--------------------------------------------------
**结论只依赖「步数」与「accept 次数」，不依赖墙钟。** ms/步只是如实记录（连同
``machine`` 字段），不得当结论，原因有三：

1. 跑本探针时这台机器正被并行任务占用，单核抢占能让 ms/步漂移数倍；
2. 模型模式下 ``update()`` 每步的前向次数**随该步是否换锚点而变**：主预测 1 次 +
   候选评分 1 次，锚点评分的前向只在缓存失效（起点、或 ``accept()`` 之后的那一步）
   时才重算 —— 即不 accept 的步 2 次、accept 后的步 3 次；与 20261108 「单次前向」
   的计时口径不可直接比，也不同变体之间不可直接比；
3. 上述锚点缓存（``differentiable_defocus_uniformity.py`` 的 ``_anchor_pred``）在取样
   当时已在工作树落地，正是「会改 ms/步而不改步数与 accept 次数」的那类改动 —— 本
   脚本两次运行观测到的 ms/步差异即由此与并行占用共同造成。要拿计时结论，必须在
   安静机器上重跑并核对代码版本（``settings.generated_utc`` 与该改动的落地时间）。

设备纪律：全程只读磁盘上的冻结 checkpoint 与代码，CPU，固定种子；未打开 SLM /
相机 / 激光，未 import ``ao_shaping.drivers``。产物写 ``data/debug/``（不入库）。

契约自检：每格每个变体跑完就地核一遍记账口径 —— 三个变体 accept 计数的定义值、
``anchor_model_loss`` 序列的分段常数结构（只在 accept 的那一步换值，换的值 = 该步
``candidate_model_loss``）、硬界 ``未锁定相位 RMS ≤ cap`` 与 ``距起点 RMS ≤
cap×(accepts+1)``、绑定步投影后恰为 cap / 未绑定步不被改动、达标步数与确认前缀和
自洽。任何不符打 WARNING 并在该格留 ``contract_ok=false``。核对逻辑本身住在
``differentiable_defocus_uniformity.audit_cap_policy``（与 ``run_accept_policy``
同处），本脚本只负责把轨迹装配成 ``results.json`` 的行。
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import platform
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    ACCEPT_POLICIES,
    FrozenDefocusForward,
    UniformityObjective,
    audit_cap_policy,
    run_accept_policy,
)
from scripts._common import fmt_general, pad, wlen

DEVICE_STATEMENT = (
    "离线分析，未占用任何设备：未打开 SLM / 相机 / 激光，未 import 任何 "
    "ao_shaping.drivers 模块；全部数字来自磁盘上的冻结 checkpoint 与纯 CPU 前向/反向，"
    "固定种子、确定性运行。"
)

DEFAULT_OUT = ROOT / "data" / "debug" / "slm_transfer" / "diffopt_cap_lr_grid_20261111"
CURVE_AT = (0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 300)
TARGETS = (0.20, 0.15, 0.10)
CAPS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
LRS: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30)
ARMS = ACCEPT_POLICIES  # 策略定义在模块里，探针不再自持一份名单
STARTS = ("zero", "random")
RANDOM_SEED = 0  # 随机起点：np.random.default_rng(RANDOM_SEED)
ITERS = 300
QUICK_ITERS = 120
QUICK_CAPS: tuple[float, ...] = (0.10,)
QUICK_LRS: tuple[float, ...] = (0.01, 0.30)
RMS_TOL = 1e-6  # float32 累加下的硬界核对容差


def load_frozen_bench():
    """按路径导入基准脚本，复用它的 checkpoint 加载与 C 轴参数化（同一口径）。

    基准 ``offline_benchmark_20261107.py`` 是**脚本**而非包模块，且它的产物已被报告
    锁定，不能为了复用而重构成 ``src/ao_shaping`` 里的模块；所以这里用 importlib 按
    文件路径加载，取它的 ``_TRANSFER``（光路配置 + ``load_fit``）与 C 轴常量
    （``C_MODES`` / ``C_RANDOM_RAD`` / ``GRID``），避免本探针留一份历史光路配置。

    ``sys.modules`` 的登记**不是可选的**：基准模块里用 ``@dataclass`` 定义的类型在
    执行期要靠模块名解析自身，模块未登记时 ``spec.loader.exec_module`` 会抛
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``。

    Returns:
        已执行的基准模块对象。

    Raises:
        ImportError: 该路径无法构造模块 spec（文件缺失或不是 Python 模块）。
    """
    path = ROOT / "scripts" / "slm" / "offline_benchmark_20261107.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:  # pragma: no cover - 环境问题
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 构造：冻结前向 + 掩膜 + 模式基（与 20261108 脚本逐字同口径）
# --------------------------------------------------------------------------- #
def predict_one(forward: FrozenDefocusForward, phase: torch.Tensor) -> torch.Tensor:
    pred = forward.predict(phase)
    return pred[0] if pred.ndim == 3 else pred


def build_forward(bench, *, fft_size: int | None = None):
    fit = bench._TRANSFER.load_fit()
    forward, _offset = bench._TRANSFER.make_forward(fit, -1.0)
    if fft_size is None:
        return forward
    new = dataclasses.replace(forward.params, fft_size=int(fft_size))
    return FrozenDefocusForward(new, forward.camera_xy, device="cpu")


def build_objective(forward: FrozenDefocusForward, bench) -> tuple[UniformityObjective, np.ndarray]:
    with torch.no_grad():
        base = predict_one(forward, torch.zeros(int(bench._TRANSFER.GRID), int(bench._TRANSFER.GRID)))
    base_np = base.detach().cpu().numpy().astype(np.float32)
    core = base_np >= 0.1 * float(base_np.max())
    objective = UniformityObjective(
        mask_box=np.ones_like(core), mask_core=core, w_shape=0.0, w_phase=0.0
    )
    return objective, core


def build_basis(forward: FrozenDefocusForward, modes) -> torch.Tensor:
    return torch.stack([forward._mode_basis(int(j)) for j in modes])


def command_of(forward: FrozenDefocusForward, basis: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    size = int(forward.params.grid_size)
    return torch.einsum("k,khw->hw", coeffs, basis).reshape(size, size)


def make_seeds(bench, forward: FrozenDefocusForward) -> dict[str, np.ndarray]:
    """两个起点：零网格；每系数 ~ ``U(±bench.C_RANDOM_RAD)`` rad 的小随机像差。

    随机起点用固定种子的**独立**生成器（``RANDOM_SEED``），因此每格每个变体拿到的起点
    逐位相同，与遍历顺序无关。
    """
    coeffs = np.random.default_rng(RANDOM_SEED).uniform(
        -float(bench.C_RANDOM_RAD), float(bench.C_RANDOM_RAD), len(bench.C_MODES)
    ).astype(np.float32)
    basis = build_basis(forward, bench.C_MODES)
    with torch.no_grad():
        random_seed = command_of(forward, basis, torch.from_numpy(coeffs)).cpu().numpy()
    size = int(forward.params.grid_size)
    return {
        "zero": np.zeros((size, size), dtype=np.float32),
        "random": np.asarray(random_seed, dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
# 一格 × 一个变体 × 一起点
# --------------------------------------------------------------------------- #
def run_arm(
    forward: FrozenDefocusForward,
    objective: UniformityObjective,
    seed: np.ndarray,
    *,
    cap: float,
    lr: float,
    arm: str,
    iters: int,
    label: str,
) -> dict[str, Any]:
    """跑一格一个变体一起点，返回可直接进 ``results.json`` 的行。

    accept 策略循环与契约核对都在模块里（:func:`run_accept_policy` /
    :func:`audit_cap_policy`）；这里只留探针侧的落盘口径——目标 CV 表 ``TARGETS``、
    曲线采样点 ``CURVE_AT``、以及各统计量怎么进表，不改任何数值取法。
    """
    trace = run_accept_policy(
        forward, objective, seed, cap=cap, lr=lr, arm=arm, iters=iters
    )
    reach_steps, reach_accepts = trace.reach(TARGETS)
    problems = audit_cap_policy(
        trace, label=label, targets=TARGETS, rms_tol=RMS_TOL
    )
    cv_after = trace.cv_after
    rms_before = trace.rms_before
    rms_after = trace.rms_after
    per_step_ms = trace.per_step_ms
    accepts = trace.accepts
    row: dict[str, Any] = {
        "cap": float(cap),
        "lr": float(lr),
        "arm": arm,
        "start": label,
        "iters": int(iters),
        "initial_cv": float(cv_after[0]),
        "final_cv": float(cv_after[-1]),
        "best_cv": float(min(cv_after)),
        "cv_curve_after_steps": {str(k): float(cv_after[min(k, iters)]) for k in CURVE_AT},
        "reach_steps": reach_steps,
        "reach_accepts": reach_accepts,
        "accepts": int(accepts),
        "accept_rate": float(accepts / max(iters, 1)),
        "cap_binding_fraction": float(sum(b > cap for b in rms_before) / len(rms_before)),
        "increment_rms_before_projection_median": float(statistics.median(rms_before)),
        "increment_rms_median": float(statistics.median(rms_after)),
        "unlocked_phase_rms_end": trace.unlocked_rms,
        "total_phase_rms_end": trace.total_rms,
        "phase_travel_rms_sum": float(sum(rms_after)),
        "ms_per_step_median": float(statistics.median(per_step_ms)),
        "ms_per_step_p90": float(np.percentile(per_step_ms, 90)),
        "wall_s": float(sum(per_step_ms) / 1e3),
        "contract_problems": problems,
        "contract_ok": not problems,
    }
    return row


# --------------------------------------------------------------------------- #
# 可读摘要
# --------------------------------------------------------------------------- #
def _cell(rows: dict[str, Any], target_key: str) -> str:
    """单元格：``步数/确认数/最终 CV``，零起点在左、随机起点在右（``‖`` 分隔）。"""
    parts = []
    for start in STARTS:
        row = rows.get(start)
        if row is None:  # pragma: no cover - 网格完整性
            parts.append("–|–|–")
            continue
        steps = row["reach_steps"][target_key]
        acc = row["reach_accepts"][target_key]
        parts.append(
            f"{fmt_general(None if steps is None else float(steps), '.0f', missing_none='未达')}"
            f"/{fmt_general(None if acc is None else float(acc), '.0f', missing_none='未达')}"
            f"/{row['final_cv']:.3f}"
        )
    return " ‖ ".join(parts)


def summarize(grid: dict[float, dict[float, dict[str, dict[str, Any]]]],
              caps: tuple[float, ...], lrs: tuple[float, ...], *,
              target: float, iters: int) -> None:
    """每个变体一张表：行 = ``cap``、列 = ``lr``，单元格 = 「步数/确认数/最终 CV」。

    读法：**同列纵向**比 —— ``cap`` 提一档后「确认数」少多少，那就是把 cap 换成
    确认周期的价格；**同行横向**比 —— 那是 ``lr`` 的价格。``cumulative`` 列确认数
    恒 0（该变体从不 accept），所以它的「步数」要对照 ``gated_oracle`` 才知道值多少。
    """
    key = f"{target:g}"
    for arm in ARMS:
        head = ["cap\\lr"] + [f"{lr:g}" for lr in lrs]
        widths = [max(wlen(h), 9) for h in head]
        body: list[list[str]] = []
        for cap in caps:
            row = [f"{cap:g}"]
            for j, lr in enumerate(lrs):
                cell = _cell(grid[cap][lr][arm], key)
                row.append(cell)
                widths[j + 1] = max(widths[j + 1], wlen(cell))
            body.append(row)
        logger.info(
            f"▍变体 {arm} —— CV ≤ {target:g} 需「步数/实测确认次数/最终 CV」"
            f"（左=零起点 ‖ 右=随机起点；“未达”= {iters} 步内未达）"
        )
        line = "  ".join(pad(h, widths[i]) for i, h in enumerate(head))
        logger.info("  " + line)
        logger.info("  " + "-" * wlen(line))
        for row in body:
            logger.info("  " + "  ".join(pad(c, widths[i]) for i, c in enumerate(row)))


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _machine(threads: int) -> str:
    return (
        f"{platform.node()}｜{platform.system()}-{platform.machine()}"
        f"｜cpu {os.cpu_count()}｜torch {torch.__version__}｜threads {threads}"
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="可微整形：step_rms_cap × lr 网格与三个变体 accept 语义（离线、CPU，不占用设备）"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="产物目录（写 results.json）")
    parser.add_argument("--iters", type=int, default=None,
                        help=f"每格每个变体的 update() 步数（默认 {ITERS}，--quick 时 {QUICK_ITERS}）")
    parser.add_argument("--quick", action="store_true",
                        help=f"冒烟：只跑 cap {list(QUICK_CAPS)} × lr {list(QUICK_LRS)}，"
                             f"步数 {QUICK_ITERS}")
    parser.add_argument("--threads", type=int, default=min(6, os.cpu_count() or 6),
                        help="torch CPU 线程数（确定性用；不改任何数值定义）")
    parser.add_argument("--caps", default=None, help="逗号分隔覆盖 cap（单格手核用）")
    parser.add_argument("--lrs", default=None, help="逗号分隔覆盖 lr（单格手核用）")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, int(args.threads)))
    caps = QUICK_CAPS if args.quick else CAPS
    lrs = QUICK_LRS if args.quick else LRS
    if args.caps:
        caps = tuple(float(v) for v in args.caps.split(","))
    if args.lrs:
        lrs = tuple(float(v) for v in args.lrs.split(","))
    iters = QUICK_ITERS if args.quick else ITERS
    if args.iters is not None:
        iters = int(args.iters)
    if iters < 1:
        logger.warning(f"--iters {iters} 非法，按 1 步跑")
        iters = 1

    logger.info(DEVICE_STATEMENT)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = load_frozen_bench()
    forward = build_forward(bench)
    objective, core = build_objective(forward, bench)
    seeds = make_seeds(bench, forward)
    size = int(forward.params.grid_size)
    logger.info(
        f"口径：grid {size}², fft {int(forward.params.fft_size)}, 掩膜核心 "
        f"{int(core.sum())}/{core.size} 像素, 参数化 Noll {list(bench.C_MODES)}"
        f"（{len(bench.C_MODES)} 模）, 起点 {list(seeds)}"
        f"（随机起点系数 ~ U(±{float(bench.C_RANDOM_RAD)}, ±{float(bench.C_RANDOM_RAD)}) rad, "
        f"seed {RANDOM_SEED}）, iters {iters}, 网格 cap {list(caps)} × lr {list(lrs)} "
        f"× 变体 {list(ARMS)}"
    )

    t_start = time.perf_counter()
    grid: dict[float, dict[float, dict[str, dict[str, Any]]]] = {}
    rows: list[dict[str, Any]] = []
    for cap in caps:
        lr_map: dict[float, dict[str, dict[str, Any]]] = {}
        for lr in lrs:
            arm_map: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                start_map: dict[str, Any] = {}
                for start, seed in seeds.items():
                    row = run_arm(forward, objective, seed,
                                  cap=cap, lr=lr, arm=arm, iters=iters, label=start)
                    start_map[start] = row
                    rows.append(row)
                    logger.info(
                        f"cap {cap:g} lr {lr:g} {arm:<13}({start:<6}): CV "
                        f"{row['initial_cv']:.3f}→{row['final_cv']:.3f}"
                        f"（最佳 {row['best_cv']:.3f}）, ≤0.20 需 "
                        f"{row['reach_steps']['0.2']} 步 / "
                        f"{row['reach_accepts']['0.2']} 次确认, accept "
                        f"{row['accepts']}/{iters}（{row['accept_rate']:.0%}）, 撞 cap "
                        f"{row['cap_binding_fraction']:.0%}, 投影前增量 RMS 中位 "
                        f"{row['increment_rms_before_projection_median']:.3f}→投影后 "
                        f"{row['increment_rms_median']:.3f}, 末态未锁定 RMS "
                        f"{row['unlocked_phase_rms_end']:.3f}, "
                        f"{row['ms_per_step_median']:.1f} ms/步"
                        + ("" if row["contract_ok"] else " ⚠ 契约不符（见上）")
                    )
                arm_map[arm] = start_map
            lr_map[lr] = arm_map
        grid[cap] = lr_map

    elapsed_s = float(time.perf_counter() - t_start)
    settings: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": bool(args.quick),
        "iters": int(iters),
        "caps": list(caps),
        "lrs": list(lrs),
        "arms": list(ARMS),
        "starts": list(seeds),
        "random_start_seed": RANDOM_SEED,
        "random_start_rad": float(bench.C_RANDOM_RAD),
        "modes_noll": [int(j) for j in bench.C_MODES],
        "grid_size": size,
        "fft_size": int(forward.params.fft_size),
        "phase_sign": -1.0,
        "core_pixels": int(core.sum()),
        "core_rule": "零相位基场 base >= 0.1 * base.max()",
        "objective": "mask_box=ones, mask_core=core, w_shape=0.0, w_phase=0.0",
        "cv_rule": "sqrt(loss_history[-1]['l_cv'] or ['loss']) —— 同 20261108 脚本 :420；"
                   "末位由 candidate_model_loss 补足「走完最后一步之后」的 CV",
        "curve_at": list(CURVE_AT),
        "targets": list(TARGETS),
        "threads": int(args.threads),
        "torch_threads": int(torch.get_num_threads()),
        "machine": _machine(int(args.threads)),
        "timing_discipline": (
            "结论只依赖步数与 accept 次数，不依赖墙钟。ms/步仅如实记录：本机正被并行任务"
            "占用；模型模式下每步前向次数随是否换锚点而变（不 accept 的步 2 次、accept 后的"
            "步 3 次，锚点前向走 _anchor_pred 缓存），与 20261108 的单前向计时口径不可直比；"
            "该缓存已在工作树落地，正是会改 ms/步而不改步数与 accept 次数的那类改动。"
        ),
        "checkpoint_fit_path": str(bench._TRANSFER.FIT_PATH),
        "rows": len(rows),
        "contract_failures": int(sum(0 if r["contract_ok"] else 1 for r in rows)),
    }
    results: dict[str, Any] = {
        "device_statement": DEVICE_STATEMENT,
        "settings": settings,
        "grid": grid,
        "elapsed_s": elapsed_s,
    }
    path = out_dir / "results.json"
    path.write_text(json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果写入：{path}")

    for target in TARGETS:
        summarize(grid, caps, lrs, target=target, iters=iters)
    logger.info(
        f"总计 {len(rows)} 行（格×变体×起点），用时 {elapsed_s:.1f} s，"
        f"契约自检不合格 {settings['contract_failures']} 行；"
        f"计时不作结论（见 settings.timing_discipline）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
