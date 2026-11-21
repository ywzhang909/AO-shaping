"""离线回放可微光束优化的逐轮 SLM 相位，诊断逐像素变化单调性并给出 lr 建议。

全程离线：不连接、不操作 SLM / CCD / 激光。本脚本所有相位都来自**离线逐位
精确重放**（复用 :mod:`scripts.slm.render_differentiable_opt_animation` 的重放
设施与 ``hw_differentiable_phase_opt.py`` 的前向模型），已验证可逐位复现存档
loss / 最终相位。因此"逐轮相位"虽未落盘，却能被精确重建。

分析口径（与硬件实际下发一致）：先按 ``phase_model_to_panel`` 把模型网格相位
中心对齐 + 最近邻缩放到 SLM 面板，再在瞳面窗口上统计逐像素相邻增量的符号
一致性；全零区域不参与统计，避免被"没在动"的像素稀释。

产物：
* ``--out``：JSON 汇总（verify、monotonicity、lr_recommendation、逐轮一致性、
  lr 反事实扫描结果）；
* ``--figure``：可选 PNG（初始相位 / 相位快照网格 / 符号一致性热力图）；
* ``--phases-out``：可选 .npz（抽样相位栈 + loss + iterations），便于后续动画。

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# 与 render 脚本一致：ROOT 本身、ROOT/scripts、ROOT/src 都放进 sys.path，且 src
# 在最前（src 与 scripts 都提供名为 ml 的包，src 必须先被解析）。
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # must precede pyplot

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import font_manager

from scripts._common.plot_style import setup_cjk_font
from scripts.slm.render_differentiable_opt_animation import (
    build_model_inputs,
    find_run_dir,
    load_main_module,
    load_run,
    read_csv_rows,
)

__all__ = [
    "DEFAULT_LR_GRID",
    "lr_recommendation",
    "map_phases_to_panel",
    "monotonicity_report",
    "per_iteration_monotonicity",
    "phase_delta_stack",
    "pixel_monotonicity",
    "replay_phases",
    "run",
    "sweep_lr",
]

#: lr 反事实扫描默认网格（真实重放，非代理）。
DEFAULT_LR_GRID: tuple[float, ...] = (0.02, 0.05, 0.08, 0.12)
#: 符号一致性 ≥ 该阈值视为"该像素单调"。
DEFAULT_MONOTONIC_THRESH = 0.9
#: 面板相位图默认下采样窗口边长（面板像素），只画瞳面块附近。
PANEL_WINDOW_PX = 560

# ---------------------------------------------------------------------------
# CJK 字体探测（复用 setup_cjk_font，缺字则退回英文标签）
# ---------------------------------------------------------------------------

_CJK_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
)


def _pick_cjk_font() -> str | None:
    """返回第一个已安装的中文字体族，找不到返回 ``None``（图内改用英文）。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


def _labels(cjk: str | None) -> dict[str, str]:
    """图中文本：有中文字体用中文，否则英文（保证无方块）。"""
    if cjk is not None:
        return {
            "initial": "初始相位（瞳面窗口）",
            "snapshot": "逐轮相位快照",
            "iteration": "迭代",
            "sign_map": "相邻步符号一致性",
            "active_only": "仅 active 像素",
            "inactive": "非 active（灰）",
            "title": "可微优化逐轮 SLM 相位单调性诊断",
        }
    return {
        "initial": "Initial phase (pupil window)",
        "snapshot": "Phase snapshots over iterations",
        "iteration": "iteration",
        "sign_map": "Sign consistency of consecutive steps",
        "active_only": "active pixels",
        "inactive": "inactive (grey)",
        "title": "Per-iteration SLM phase monotonicity diagnosis",
    }


# ---------------------------------------------------------------------------
# 纯函数（独立可测）
# ---------------------------------------------------------------------------


def phase_delta_stack(phases: np.ndarray) -> np.ndarray:
    """相邻增量栈 ``phases[1:] - phases[:-1]``，输入 (N,H,W) → (N-1,H,W)。"""
    arr = np.asarray(phases)
    if arr.ndim != 3:
        raise ValueError(f"phases 必须是 (N,H,W) 三维栈, got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"phases 至少需要 2 帧才能算增量, got {arr.shape[0]} 帧")
    return arr[1:].astype(np.float64, copy=False) - arr[:-1].astype(np.float64, copy=False)


def pixel_monotonicity(
    deltas: np.ndarray,
    *,
    min_total_change: float = 1e-3,
    monotonic_thresh: float = 0.9,
) -> dict[str, Any]:
    """逐像素单调一致性统计（仅统计"真的在动"的 active 像素）。

    Args:
        deltas: 相邻增量栈 (N-1,H,W)。
        min_total_change: active 判定阈值，``|phase[-1]-phase[0]|`` 超过它才算
            "真的在动"，全零区域被排除，避免稀释统计。
        monotonic_thresh: 判定单像素"单调"的一致性阈值。

    Returns:
        dict：``active_mask``(H,W bool)、``sign_consistency_map``(H,W float)、
        ``frac_pixels_monotonic``、``mean_sign_consistency``、
        ``frac_active_pixels``、``mean_total_change_active``。
    """
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 3:
        raise ValueError(f"deltas 必须是 (N-1,H,W) 三维栈, got shape {d.shape}")
    if d.shape[0] < 1:
        raise ValueError("deltas 至少需要 1 步")
    n_steps = d.shape[0]
    positive = np.count_nonzero(d > 0, axis=0)
    negative = np.count_nonzero(d < 0, axis=0)
    # 零增量用 0.5 中性计（不偏向任何方向）；总步数含零步。
    consistency = np.maximum(positive, negative) / float(n_steps)
    zero = positive + negative == 0
    if np.any(zero):
        consistency = consistency.copy()
        consistency[zero] = 0.5

    total_change = np.abs(np.sum(d, axis=0))
    active = total_change > float(min_total_change)
    active_count = int(np.count_nonzero(active))
    if active_count:
        active_consistency = consistency[active]
        frac_monotonic = float(np.mean(active_consistency >= float(monotonic_thresh)))
        mean_consistency = float(np.mean(active_consistency))
        mean_total_change = float(np.mean(total_change[active]))
    else:
        frac_monotonic = 0.0
        mean_consistency = 0.0
        mean_total_change = 0.0

    return {
        "active_mask": active,
        "sign_consistency_map": consistency,
        "frac_pixels_monotonic": frac_monotonic,
        "mean_sign_consistency": mean_consistency,
        "frac_active_pixels": float(active_count / active.size),
        "mean_total_change_active": mean_total_change,
    }


def per_iteration_monotonicity(deltas: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """每步的全局方向一致性：返回长度 N-2 的数组。

    第 ``i`` 项 = 第 ``i`` 步与第 ``i+1`` 步增量方向一致的 active 像素占比；
    任一步增量为零时用 0 保护（不算"一致"）。
    """
    d = np.asarray(deltas, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if d.ndim != 3:
        raise ValueError(f"deltas 必须是 (N-1,H,W) 三维栈, got shape {d.shape}")
    if d.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    if active.shape != d.shape[1:]:
        raise ValueError(f"active_mask 形状 {active.shape} 与增量 {d.shape[1:]} 不匹配")
    prev, nxt = d[:-1], d[1:]
    same = (prev > 0) & (nxt > 0)
    same |= (prev < 0) & (nxt < 0)
    n_active = int(np.count_nonzero(active))
    if n_active == 0:
        return np.zeros(d.shape[0] - 1, dtype=np.float64)
    return np.count_nonzero(same & active, axis=(1, 2)).astype(np.float64) / float(n_active)


def lr_recommendation(
    metrics: dict[str, Any],
    *,
    cos_plateau: float = 0.7,
    monotone_thresh: float = 0.9,
) -> str:
    """按逐像素单调性给出学习率建议（三分支）。

    Args:
        metrics: :func:`pixel_monotonicity` 的结果（可读其分数与平均总变化）。
        cos_plateau: 进入"方向不一致"区的一致性占比阈值。
        monotone_thresh: 这里只作语义提示（单像素单调阈值，实际分支用
            ``frac_pixels_monotonic`` 与 ``cos_plateau`` 比较）。
    """
    frac = float(metrics.get("frac_pixels_monotonic", 0.0))
    mean_change = float(metrics.get("mean_total_change_active", 0.0))
    # "未饱和"的自定合理条件：active 像素的平均总变化仍明显小于 π。
    unsaturated = mean_change < float(np.pi)
    if frac >= float(cos_plateau) and unsaturated:
        return (
            "逐像素变化高度单调，优化处于稳定弹道：可在早期/中期放大 lr 加速收敛；"
            "进入符号翻转（一致性<0.7）后收回。"
        )
    if 0.4 <= frac < float(cos_plateau):
        return "单调性中等，放大 lr 收益不确定，建议小倍数 A/B 灰度验证。"
    return "逐像素方向不一致/已到噪声地板，不建议放大 lr，宜减小或保持。"


def monotonicity_report(
    phases: np.ndarray,
    *,
    min_total_change: float = 1e-3,
    monotonic_thresh: float = 0.9,
    cos_plateau: float = 0.7,
) -> dict[str, Any]:
    """从原始相位栈出发，一次性算齐逐像素与逐轮单调性 + lr 建议。

    单调性只依赖相邻增量的**符号**与 ``|总变化|`` 阈值。面板映射是最近邻
    正比例拷贝（瞳面块内 pv 一一对应、块外恒零），不改变瞳面块内任一像素
    增量的符号或幅度，块外恒零像素本来就被 active mask 排除；因此直接在
    模型网格 1024² 上统计与在面板 1920×1208 上统计等价，且省内存。
    """
    deltas = phase_delta_stack(phases)
    metrics = pixel_monotonicity(
        deltas,
        min_total_change=min_total_change,
        monotonic_thresh=monotonic_thresh,
    )
    per_step = per_iteration_monotonicity(deltas, metrics["active_mask"])
    metrics["per_iteration_direction_consistency"] = per_step
    metrics["lr_recommendation"] = lr_recommendation(
        metrics, cos_plateau=cos_plateau, monotone_thresh=monotonic_thresh
    )
    return metrics


# ---------------------------------------------------------------------------
# 重放与映射
# ---------------------------------------------------------------------------


def _make_optimizer(
    target: np.ndarray,
    amplitude: np.ndarray,
    *,
    lr: float,
    device: str,
    seed: int,
    init_phase: np.ndarray | None,
) -> Any:
    """构造优化器（在独立的内部函数里 import torch，方便离线无 GPU 时延迟加载）。"""
    from ao_shaping.algorithm.signal_processing.differentiable_beam import (
        DifferentiableBeamOptimizer,
    )

    return DifferentiableBeamOptimizer(
        target,
        source_amplitude=amplitude,
        lr=float(lr),
        device=device,
        seed=int(seed),
        init_phase=init_phase,
    )


def replay_phases(
    ctx: Any,
    main_mod: ModuleType,
    *,
    init_phase: np.ndarray | None,
    device: str,
    target: np.ndarray | None = None,
    amplitude: np.ndarray | None = None,
) -> dict[str, Any]:
    """逐轮重放，收集每轮**开始时**的相位，并校验 loss / 最终相位对存档。

    Returns:
        dict：``phases`` (N,iters+1,H,W)、``losses`` (iters,)、``verify``（对照
        ``history.csv`` 与 ``phase_final.npy`` 的误差证据）。验证不一致时仅
        warning，不中断，保留证据。
    """
    if target is None or amplitude is None:
        _spec, amplitude, target, _info = build_model_inputs(main_mod, ctx)
    seed = int(ctx.params.get("seed", 0))
    lr = float(ctx.params.get("lr", 0.05))
    n_iters = int(ctx.iters)
    opt = _make_optimizer(target, amplitude, lr=lr, device=device, seed=seed, init_phase=init_phase)

    phases: list[np.ndarray] = []
    losses: list[float] = []
    for _ in range(n_iters):
        phases.append(np.asarray(opt.current_phase, dtype=np.float64).copy())
        opt.update()
        losses.append(float(opt.last_loss if opt.last_loss is not None else float("nan")))
    phases.append(np.asarray(opt.current_phase, dtype=np.float64).copy())

    verify: dict[str, Any] = {
        "history_csv": False,
        "compared": 0,
        "max_rel_loss_diff": float("nan"),
        "phase_final_max_abs_diff_rad": float("nan"),
    }
    reference = read_csv_rows(ctx.run_dir / "history.csv")
    if reference:
        diffs: list[float] = []
        for row, ref in zip(losses, reference):
            try:
                ref_loss = float(ref.get("loss", "nan"))
            except (TypeError, ValueError):
                break
            if not np.isfinite(ref_loss):
                continue
            diffs.append(abs(row - ref_loss) / max(abs(ref_loss), 1e-30))
        if diffs:
            verify.update(
                history_csv=True,
                compared=len(diffs),
                max_rel_loss_diff=float(max(diffs)),
            )
    phase_path = ctx.run_dir / "phase_final.npy"
    if phase_path.is_file():
        saved = np.load(phase_path).astype(np.float64)
        final_phase = phases[-1]
        if saved.shape == final_phase.shape:
            verify["phase_final_max_abs_diff_rad"] = float(
                np.max(np.abs(saved - final_phase))
            )

    bad: list[str] = []
    if verify["history_csv"] and verify["max_rel_loss_diff"] > 1e-6:
        bad.append(f"loss max_rel_diff={verify['max_rel_loss_diff']:.3e}")
    if np.isfinite(verify["phase_final_max_abs_diff_rad"]):
        if verify["phase_final_max_abs_diff_rad"] > 1e-6:
            bad.append(f"phase max_abs_diff={verify['phase_final_max_abs_diff_rad']:.3e}")
    if bad:
        logger.warning("离线重放与存档不一致（保留证据，不中断）: {}", "; ".join(bad))
    else:
        logger.info(
            "离线重放校验通过: loss max_rel_diff={}, phase max_abs_diff={}",
            verify.get("max_rel_loss_diff"),
            verify.get("phase_final_max_abs_diff_rad"),
        )
    return {
        "phases": np.stack(phases),
        "losses": np.asarray(losses, dtype=np.float64),
        "verify": verify,
    }


def map_phases_to_panel(
    phases: np.ndarray,
    spec: Any,
    *,
    indices: list[int] | None = None,
    main_mod: ModuleType | None = None,
) -> np.ndarray:
    """把相位栈逐帧经 ``main_mod.phase_model_to_panel`` 映射到 SLM 面板。

    Args:
        phases: (N,H,W) 模型相位栈。
        spec: main 模块的 ``ModelSpec``（含 pupil_center/n_pupil/dp/m_grid 等）。
        indices: 只映射这些帧下标；None 表示全部。为省内存可只映射抽样帧。
        main_mod: 提供 ``phase_model_to_panel`` 的硬件脚本模块；None 时按路径加载。

    Returns:
        (K,Hp,Wp) 面板相位栈。
    """
    arr = np.asarray(phases)
    if arr.ndim != 3:
        raise ValueError(f"phases 必须是 (N,H,W) 三维栈, got shape {arr.shape}")
    if indices is None:
        idx = list(range(arr.shape[0]))
    else:
        idx = [int(i) for i in indices]
    if main_mod is None:
        main_mod = load_main_module()
    fn = getattr(main_mod, "phase_model_to_panel", None)
    if fn is None:
        raise AttributeError("main_mod 缺少 phase_model_to_panel（应使用硬件脚本模块）")
    return np.stack([np.asarray(fn(arr[i], spec), dtype=np.float64) for i in idx])


def sweep_lr(
    target: np.ndarray,
    amplitude: np.ndarray,
    *,
    init_phase: np.ndarray | None,
    device: str,
    seed: int,
    n_iters: int,
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID,
    target_loss: float,
) -> dict[str, Any]:
    """lr 反事实扫描：同一 init_phase/seed/target 冷启动，逐位真实重放。

    对每个 lr 记录 ``final_loss``、``iters_to_target``（首次达到存档最终 loss
    水平的轮数；未达到或超过 ``n_iters`` 记 None）、``converged``。
    """
    out: dict[str, Any] = {}
    for lr in lr_grid:
        opt = _make_optimizer(
            target, amplitude, lr=lr, device=device, seed=seed, init_phase=init_phase
        )
        losses: list[float] = []
        iters_to_target: int | None = None
        for it in range(n_iters):
            opt.update()
            loss = float(opt.last_loss if opt.last_loss is not None else float("nan"))
            losses.append(loss)
            if iters_to_target is None and np.isfinite(loss) and loss <= target_loss:
                iters_to_target = it + 1
        reached = iters_to_target is not None
        out[f"{lr:g}"] = {
            "lr": float(lr),
            "final_loss": float(losses[-1]) if losses else float("nan"),
            "iters_to_target": iters_to_target,
            "converged": bool(reached),
            "note": (
                "达到存档最终 loss 水平"
                if reached
                else f"{n_iters} 轮内未达到存档最终 loss 水平"
            ),
        }
        logger.info(
            "lr={} 重放完成: final_loss={} iters_to_target={}",
            lr,
            out[f"{lr:g}"]["final_loss"],
            iters_to_target,
        )
    return out


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def _panel_window(panel: np.ndarray, spec: Any, size: int = PANEL_WINDOW_PX) -> np.ndarray:
    """裁出以瞳面中心为中心、边长 ``size`` 的面板窗口（含周围零填充）。

    直接画整块面板会是一片几乎全零的黑图（瞳面块只占中央一小片），所以只画
    瞳面中心附近窗口，既能看到中心对齐 + 最近邻放大的块状相位，也不会浪费画布。
    """
    cx, cy = spec.pupil_center
    half = int(size) // 2
    x0 = max(0, int(round(cx)) - half)
    x1 = min(panel.shape[1], int(round(cx)) + half)
    y0 = max(0, int(round(cy)) - half)
    y1 = min(panel.shape[0], int(round(cy)) + half)
    if x1 <= x0 or y1 <= y0:
        return panel
    return panel[y0:y1, x0:x1]


def _pupil_window_from_panel(panel: np.ndarray, spec: Any) -> np.ndarray:
    """从面板相位里裁瞳面方块窗口（按 spec 的 pupil_center 与半径）。"""
    cx, cy = spec.pupil_center
    half = int(round(float(spec.pupil_radius)))
    x0 = max(0, int(round(cx)) - half)
    x1 = min(panel.shape[1], int(round(cx)) + half + 1)
    y0 = max(0, int(round(cy)) - half)
    y1 = min(panel.shape[0], int(round(cy)) + half + 1)
    if x1 <= x0 or y1 <= y0:
        return panel
    return panel[y0:y1, x0:x1]


def _draw_sign_map(ax: Any, map2d: np.ndarray, active: np.ndarray, labels: dict[str, str]) -> None:
    """符号一致性热力图：非 active 像素灰掉。"""
    shown = np.full(map2d.shape, np.nan, dtype=np.float64)
    shown[active] = map2d[active]
    im = ax.imshow(shown, cmap="viridis", vmin=0.5, vmax=1.0, interpolation="nearest")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(labels["sign_map"], fontsize=8)
    ax.set_title(labels["sign_map"] + f" ({labels['active_only']})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def write_figure(
    path: Path,
    *,
    phases: np.ndarray,
    frame_indices: list[int],
    panel_phases: np.ndarray,
    metrics: dict[str, Any],
    spec: Any,
) -> None:
    """三子图：初始相位窗口 / 相位快照网格 / 符号一致性热力图（active 外灰）。"""
    cjk = _pick_cjk_font()
    if cjk is not None:
        setup_cjk_font((cjk, "DejaVu Sans"))
    labels = _labels(cjk)
    n_snapshots = panel_phases.shape[0]
    if n_snapshots == 0:
        logger.warning("没有可画相位快照的帧，跳过 PNG")
        return
    fig = plt.figure(figsize=(4.2, 4.2 + 1.0 * n_snapshots / 3.0), dpi=110)
    fig.suptitle(labels["title"], fontsize=11)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    # 3 个子图：初始相位（瞳面窗口）/ 相位快照网格 / 符号一致性热力图。
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(_panel_window(panel_phases[0], spec), cmap="gray", interpolation="nearest")
    ax0.set_title(labels["initial"], fontsize=9)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax2 = fig.add_subplot(gs[:, 1])
    cols = min(3, max(1, n_snapshots))
    rows = int(np.ceil(n_snapshots / cols))
    for k, (frame, it_idx) in enumerate(zip(panel_phases, frame_indices)):
        r, c = divmod(k, cols)
        sub = ax2.inset_axes(
            [c / cols + 0.01, 1.0 - (r + 1) / rows + 0.02, 1.0 / cols - 0.02, 1.0 / rows - 0.03]
        )
        sub.imshow(_pupil_window_from_panel(frame, spec), cmap="gray", interpolation="nearest")
        sub.set_title(f"{labels['iteration']} {it_idx}", fontsize=6)
        sub.set_xticks([])
        sub.set_yticks([])
    ax2.set_title(labels["snapshot"], fontsize=9)
    ax2.set_axis_off()

    ax3 = fig.add_subplot(gs[1, 0])
    _draw_sign_map(ax3, metrics["sign_consistency_map"], metrics["active_mask"], labels)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("相位单调性可视化已写入: {}", path)


# ---------------------------------------------------------------------------
# CLI / 编排
# ---------------------------------------------------------------------------


def _select_frame_indices(n_frames_total: int, max_frames: int) -> list[int]:
    """均匀抽样要落盘/画图的帧下标（含首末），分析仍用全部帧。"""
    n = max(1, min(int(max_frames), n_frames_total))
    if n == 1:
        return [0]
    picks = np.unique(np.round(np.linspace(0, n_frames_total - 1, n)).astype(int)).tolist()
    return picks


def _resolve_device(requested: str) -> str:
    """cuda 不可用时退回 CPU 并告警（CPU 可能有轻微数值差异，验收优先 cuda）。"""
    if requested != "cuda":
        return requested
    try:
        import torch
    except ImportError:
        logger.warning("未安装 torch，无法使用 cuda；退回 cpu")
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning("cuda 不可用，退回 cpu（数值可能与存档有轻微差异）")
        return "cpu"
    return "cuda"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """编排：加载 → 重放 → 映射 → 单调性 → lr 建议 → 反事实扫描 → 写产物。"""
    setup_cjk_font()
    run_dir = find_run_dir(args.run_dir)
    logger.info("离线诊断运行目录: {}", run_dir)
    ctx = load_run(run_dir)
    main_mod = load_main_module()
    spec, amplitude, target, _target_info = build_model_inputs(main_mod, ctx)
    init_path = ctx.run_dir / "phase_initial.npy"
    init_phase = np.load(init_path) if init_path.is_file() else None
    device = _resolve_device(args.device)

    replay = replay_phases(
        ctx, main_mod, init_phase=init_phase, device=device, target=target, amplitude=amplitude
    )
    phases = replay["phases"]
    logger.info("重放相位栈: shape={}, device={}", phases.shape, device)

    metrics = monotonicity_report(phases)
    logger.info(
        "单调性: frac_monotonic={:.3f} mean_consistency={:.3f} frac_active={:.3f}",
        metrics["frac_pixels_monotonic"],
        metrics["mean_sign_consistency"],
        metrics["frac_active_pixels"],
    )

    # lr 反事实扫描：复用同一 target/amplitude/spec/seed/init_phase 冷启动。
    if np.isfinite(replay["losses"]).all() and len(replay["losses"]):
        target_loss = float(replay["losses"][-1])
    else:
        reference = read_csv_rows(ctx.run_dir / "history.csv")
        target_loss = float(reference[-1].get("loss", "nan")) if reference else float("nan")
    lr_grid = tuple(float(x) for x in args.lr_grid)
    sweep = sweep_lr(
        target,
        amplitude,
        init_phase=init_phase,
        device=device,
        seed=int(ctx.params.get("seed", 0)),
        n_iters=int(ctx.iters),
        lr_grid=lr_grid,
        target_loss=target_loss,
    )

    frame_indices = _select_frame_indices(phases.shape[0], args.max_frames)
    panel_phases = map_phases_to_panel(
        phases, spec, indices=frame_indices, main_mod=main_mod
    )

    # 清理 numpy 数组为可 JSON 序列化形式；数组证据单独存 .npz / PNG。
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "script": str(Path(__file__).relative_to(ROOT).as_posix()),
        "device": device,
        "lr_recorded": float(ctx.params.get("lr", 0.05)),
        "iters": int(ctx.iters),
        "verify": replay["verify"],
        "monotonicity": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in metrics.items()
            if key not in ("active_mask", "sign_consistency_map")
        },
        "lr_recommendation": metrics["lr_recommendation"],
        "lr_sweep": sweep,
        "target_loss": target_loss,
        "visualized_iterations": frame_indices,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    logger.info("JSON 汇总已写入: {}", out_path)

    if args.figure:
        write_figure(
            Path(args.figure),
            phases=phases,
            frame_indices=frame_indices,
            panel_phases=panel_phases,
            metrics=metrics,
            spec=spec,
        )
    if args.phases_out:
        npz_path = Path(args.phases_out)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            phases=phases[frame_indices],
            losses=replay["losses"],
            iterations=np.asarray(frame_indices, dtype=np.int64),
        )
        logger.info("抽样相位栈已写入: {}", npz_path)
    return summary


def main(argv: list[str] | None = None) -> None:
    """argparse 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="离线运行目录（默认取最新）")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--max-frames", type=int, default=20, help="相位帧抽样上限（分析用全部帧）"
    )
    parser.add_argument("--out", required=True, help="JSON 汇总输出路径")
    parser.add_argument("--figure", default=None, help="可选相位变化可视化 PNG")
    parser.add_argument("--phases-out", default=None, help="可选抽样相位栈 .npz")
    parser.add_argument(
        "--lr-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_LR_GRID),
        help="lr 反事实扫描网格（默认 0.02 0.05 0.08 0.12）",
    )
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
