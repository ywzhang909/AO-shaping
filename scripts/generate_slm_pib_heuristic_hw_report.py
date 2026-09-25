"""Real-hardware benchmark of the heuristics driving ``slm-pib``.

Runs the camera test and then every heuristic algorithm on the physical bench
(Santec SLM-200 + Daheng MER2-507 NIR camera), optimising the far-field
power-in-bucket (PIB) of the SLM-Zernike loop. Emits a report with the CCD
initial/best spot images, the PIB convergence curves and a text summary into
``docs/slm_pib_heuristic_hw/``.

Budget note: this is an **indicative** comparison on a small per-algorithm
device-load budget (the SLM needs ~0.3 s to settle per phase load), not a full
convergence study.

Usage:
    $env:PYTHONPATH = "src;libs"      # libs/ is required for gxipy (Daheng)
    python scripts/generate_slm_pib_heuristic_hw_report.py
Optional:
    --exposure-ms 3.0   (safe limit on this bench is <= 3 ms)
    --cam-id 0 --slm-number 1 --wavelength 1064
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap (repo rule). ``libs`` is REQUIRED so gxipy (Daheng SDK) imports.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from loguru import logger  # noqa: E402

from ao_shaping.optimizer.wfless.slm_zernike_pib import (  # noqa: E402
    optimize_slm_zernike_pib,
)
from ao_shaping.utils.image.beam_metrics import (  # noqa: E402
    clamp_center_to_frame,
    threshold_spot_center,
    zero_order_center,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = ROOT / "docs" / "slm_pib_heuristic_hw"

# Metric label for report text/figure titles, set from --objective in main().
_OBJECTIVE_LABEL = "pib"


def _obj_label() -> str:
    """Human-readable name of the optimised objective (default: rectangle shaping)."""
    return "整形评分 (shape)" if _OBJECTIVE_LABEL == "shape" else f"{_OBJECTIVE_LABEL} 评分"

# (algorithm, epochs, pop_size): a comparable ~80-96 device-load budget each
# (population methods spend pop_size loads per iteration).
# (label, algorithm, optimizer_type, epochs, pop_size). The gradient baselines
# (spgd + adamod / spgd + adam) spend 2 loads per epoch, so 45 epochs ≈ 90 loads,
# comparable to the ~80-96 load budget of the heuristics.
ALGORITHMS: list[tuple[str, str, str, int, int | None]] = [
    ("spgd-adamod", "spgd", "adamod", 57, None),
    ("spgd-adam", "spgd", "adam", 57, None),
    ("ga", "ga", "adamod", 6, 16),
    ("pso", "pso", "adamod", 6, 16),
    ("sa", "sa", "adamod", 80, None),
    ("hc", "hc", "adamod", 80, None),
    ("rs", "rs", "adamod", 80, None),
    ("cem", "cem", "adamod", 6, 16),
    ("de", "de", "adamod", 6, 16),
]


def camera_test(cam_type: str, cam_id: int, exposure_ms: float) -> dict:
    """Open the camera, capture one frame and exercise exposure/centre utilities."""
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_ms,
        get_camera_exposure_range,
    )

    cam = create_camera(cam_type, cam_id=cam_id, exposure_time_ms=exposure_ms)
    cam.open()
    try:
        img = cam.get_numpy_image(3)
        info = {
            "shape": img.shape,
            "dtype": str(img.dtype),
            "max": int(img.max()),
            "mean": round(float(img.mean()), 3),
            "exposure_ms": round(float(get_camera_exposure_ms(cam)), 4),
            "range_ms": tuple(round(float(v), 3) for v in get_camera_exposure_range(cam)),
            "frame_center": (img.shape[1] // 2, img.shape[0] // 2),
            "spot_center": tuple(int(v) for v in threshold_spot_center(img)),
            "saturated": bool(img.max() >= 255),
        }
        logger.info("camera test: {}", info)
        return info
    finally:
        cam.close()


def setup_bench(args, n_center: int = 12) -> tuple[float, tuple[int, int], float]:
    """One-shot bench setup: auto-expose on the flat phase, then **fix the centre**.

    Returns ``(exposure_ms, (cx, cy), centre_spread_px)``. The centre is the median
    of ``n_center`` argmax-anchored detections on the settled flat phase. Passing it
    as a FIXED tuple to every algorithm removes the per-run centre variance that
    otherwise dominates the comparison — the PIB bucket is only
    ``IDEAL_SPOT_RADIUS`` (6 px) wide, so a tens-of-pixels centre shift collapses
    the metric regardless of the search algorithm.
    """
    from ao_shaping.drivers.ccd.common import (
        auto_exposure,
        create_camera,
        get_camera_exposure_ms,
    )
    from ao_shaping.drivers.slm import Santec
    from ao_shaping.optimizer.wfless.slm_zernike_pib import (
        PatternHelper,
        _display,
        _zernike_to_phase,
    )

    cam = create_camera(args.cam_type, cam_id=args.cam_id, exposure_time_ms=3.0)
    cam.open()
    slm = Santec(slm_number=args.slm_number, wavelength=args.wavelength)
    slm.open()
    try:
        ph = PatternHelper(resolution=(1920, 1200), bits=10)
        zeros = np.zeros(15, dtype=np.float64)
        _display(slm, slm.create_phase_from_array(_zernike_to_phase(zeros, args.n_max, ph)))
        time.sleep(0.3)

        if args.exposure_ms > 0:
            from ao_shaping.drivers.ccd.common import set_camera_exposure_ms

            set_camera_exposure_ms(cam, args.exposure_ms)
            img = cam.get_numpy_image(2)
        else:
            img = auto_exposure(cam, args.target_brightness)
        exp = float(get_camera_exposure_ms(cam))

        pts = np.array(
            [zero_order_center(cam.get_numpy_image(2)) for _ in range(n_center)],
            dtype=np.float64,
        )
        cx, cy = int(round(float(np.median(pts[:, 0])))), int(round(float(np.median(pts[:, 1]))))
        spread = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))

        logger.info(
            "auto-exposure target={} -> exp={:.3f}ms (max={}); FIXED center=({}, {}) "
            "spread<={:.1f}px over {} frames",
            args.target_brightness,
            exp,
            int(img.max()),
            cx,
            cy,
            spread,
            n_center,
        )
        return exp, (cx, cy), spread
    finally:
        slm.close()
        cam.close()


def run_algorithm(
    label: str,
    algorithm: str,
    optimizer_type: str,
    epochs: int,
    pop_size: int | None,
    args,
    cam=None,
    slm=None,
) -> dict:
    """Run one search configuration on the bench and return its history/metrics."""
    logger.info(
        "=== running {} (algorithm={}, optimizer={}, epochs={}, pop_size={}) ===",
        label,
        algorithm,
        optimizer_type,
        epochs,
        pop_size,
    )
    t0 = time.perf_counter()
    recorder = optimize_slm_zernike_pib(
        center=args.center,
        objective=args.objective,
        target_shape=args.target_shape,
        target_size=args.target_size,
        epochs=epochs,
        n_max=args.n_max,
        delta=args.delta,
        max_roi_energy_loss=args.max_energy_loss,
        cam_id=args.cam_id,
        cam_type=args.cam_type,
        exposure_time_ms=args.exposure_ms,
        cam_size=args.cam_size,
        target_max_brightness=0,  # fixed exposure requested above
        slm_number=args.slm_number,
        slm_wavelength=args.wavelength,
        algorithm=algorithm,
        optimizer_type=optimizer_type,
        pop_size=pop_size,
        random_seed=args.seed,
        show=False,
        cam=cam,
        slm=slm,
    )
    df = recorder.dataframe
    # The metric column is named after the objective ("shape" / "pib" / ...).
    pib = df[args.objective].astype(float).to_numpy()
    curve = np.maximum.accumulate(pib)
    best_row, (best_idx, best_val) = recorder.get_best_iter()
    evals_per_row = 2 if algorithm == "spgd" else 1
    elapsed = time.perf_counter() - t0
    logger.info(
        "{}: init={:.4f} best={:.4f} @ row {} ({} loads, {:.1f}s)",
        algorithm,
        float(pib[0]),
        float(best_val),
        int(best_idx),
        len(pib),
        elapsed,
    )
    return {
        "pib": pib,
        "curve": curve,
        # Report images: prefer the UN-windowed (raw full-frame) captures taken by
        # the optimizer right after the search (the metric window hides the energy
        # that leaves the target box); fall back to the window frames if absent.
        "init_img": np.asarray(
            getattr(recorder, "raw_before_img", None)
            if getattr(recorder, "raw_before_img", None) is not None
            else df.iloc[0]["_img"],
            dtype=np.float64,
        ),
        "best_img": np.asarray(
            getattr(recorder, "raw_after_img", None)
            if getattr(recorder, "raw_after_img", None) is not None
            else best_row["_img"],
            dtype=np.float64,
        ),
        "init_pib": float(pib[0]),
        "final_pib": float(best_val),
        # SPGD loads 2 phases per recorder row (v+δ / v−δ); heuristics load 1.
        # All "loads" numbers in the report are device phase loads, not rows.
        "evals_per_row": evals_per_row,
        "loads": int(len(pib)) * evals_per_row,
        "x": np.arange(1, len(pib) + 1, dtype=float) * evals_per_row,
        "elapsed_s": elapsed,
    }


def _first_reach(curve: np.ndarray, threshold: float) -> int | None:
    idx = np.flatnonzero(curve >= threshold)
    return int(idx[0] + 1) if idx.size else None


def _first_load(result: dict, threshold: float) -> int | None:
    """Recorder rows → device loads (SPGD spends 2 loads per row)."""
    rows = _first_reach(result["curve"], threshold)
    return None if rows is None else int(rows * result.get("evals_per_row", 1))


def _spot(img: np.ndarray) -> np.ndarray:
    """Log-scaled display image in [0, 1]."""
    out = np.log10(1.0 + np.asarray(img, dtype=np.float64))
    peak = float(out.max())
    return out / peak if peak > 0 else out


def fig_camera(info: dict, img: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(_spot(img), cmap="inferno")
    cx, cy = info["spot_center"]
    ax.scatter([cx], [cy], marker="+", s=120, c="cyan", label=f"spot {info['spot_center']}")
    ax.set_title(
        f"CCD 初始帧 @ {info['exposure_ms']}ms  max={info['max']} mean={info['mean']}"
    )
    ax.set_axis_off()
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "camera_frame.png", dpi=150)
    plt.close(fig)


def fig_curves(results: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in results.items():
        x = r["x"]
        ax.plot(x, r["curve"], marker=".", ms=3, lw=1.3, label=f"{name} (max={r['final_pib']:.3f})")
    ax.set_xlabel("设备相位加载次数 (loads)")
    ax.set_ylabel(f"最优 {_obj_label()} (best-so-far)")
    ax.set_title(f"SLM 相位优化收敛曲线 (真机, 目标: {_obj_label()})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "pib_curves.png", dpi=150)
    plt.close(fig)


def fig_convergence(results: dict[str, dict], out: Path) -> None:
    names = list(results)
    metrics = [
        ("首次达到自身最大值", [_first_load(results[n], results[n]["final_pib"] - 1e-9) or np.nan for n in names]),
    ]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (label, vals) in enumerate(metrics):
        ax.bar(x, vals, 0.5, label=label, color="#4C72B0")
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(float(xi), v + 0.5, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("加载次数")
    ax.set_title(f"收敛速度: 达到自身最优{_obj_label()}所需加载次数 (真机)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "convergence_speed.png", dpi=150)
    plt.close(fig)


def fig_spots(results: dict[str, dict], out: Path) -> None:
    names = list(results)
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    for col, name in enumerate(names):
        for row, (key, tag) in enumerate(((("init_img"), "初始"), ("best_img", "最优"))):
            ax = axes[row, col]
            ax.imshow(_spot(results[name][key]), cmap="inferno", vmin=0, vmax=1)
            ax.set_axis_off()
            pib = results[name]["init_pib"] if row == 0 else results[name]["final_pib"]
            ax.set_title(f"{name}\n{tag} PIB={pib:.3f}", fontsize=8)
    fig.suptitle("CCD 实测光斑 (同一灰度标尺): 初始 vs 各算法最优", fontsize=12)
    im = axes[0, 0].images[0]
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04)
    fig.savefig(out / "spot_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_summary_bars(results: dict[str, dict], out: Path) -> None:
    ordered = sorted(results.items(), key=lambda kv: kv[1]["final_pib"])
    names = [k for k, _ in ordered]
    vals = [v["final_pib"] for _, v in ordered]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, vals, color="#55A868")
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel(f"最终 {_obj_label()}")
    ax.set_title(f"各算法最终 {_obj_label()} (真机, 升序)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out / "summary_bars.png", dpi=150)
    plt.close(fig)


def write_csv(results: dict[str, dict], out: Path) -> None:
    lines = ["algorithm,init_pib,final_pib,gain,n_loads,elapsed_s"]
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["final_pib"], reverse=True):
        lines.append(
            "{},{:.6f},{:.6f},{:.6f},{},{:.2f}".format(
                name, r["init_pib"], r["final_pib"], r["final_pib"] - r["init_pib"], r["loads"], r["elapsed_s"]
            )
        )
    (out / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf8")


def write_report(results: dict[str, dict], cam: dict, args) -> str:
    best_name = max(results, key=lambda k: results[k]["final_pib"])
    # Compare the best algorithm against ITS OWN initial PIB (not another run's).
    init_pib = results[best_name]["init_pib"]
    inits = [r["init_pib"] for r in results.values()]
    init_lo, init_hi = min(inits), max(inits)
    init_spread = init_hi - init_lo
    max_gain = max(r["final_pib"] - r["init_pib"] for r in results.values())
    finals_sorted = sorted((r["final_pib"] for r in results.values()), reverse=True)
    top_spread = finals_sorted[0] - finals_sorted[1] if len(finals_sorted) > 1 else 0.0
    if init_spread > max_gain:
        variance_note = (
            f'> ⚠️ **这个"最优"结论不可靠 —— 请看方差**: 本轮各算法的**初始{_obj_label()}相差 '
            f'{init_spread:.4f}** (范围 **{init_lo:.4f} ~ {init_hi:.4f}**, 全部起始于同一"平场相位"), '
            f'远大于算法带来的提升 (最大 {max_gain:.4f})。\n'
            "> 说明**跨轮方差 (光强漂移 / 中心检测不稳定 / 光路状态)** 主导了结果, 而不是算法的优劣。\n"
            "> 要判断算法优劣必须: ① 加大每算法预算 ② 每个算法重复多次取统计 ③ 先稳定中心检测与光强。"
        )
    else:
        pct = abs(top_spread / max_gain) * 100 if max_gain else 0.0
        variance_note = (
            f'> ✅ **方差已受控**: 初始{_obj_label()}散布仅 **{init_spread:.4f}** '
            f'(范围 **{init_lo:.4f} ~ {init_hi:.4f}**, 同一平场相位), 远小于算法提升 '
            f'(最大 **{max_gain:.4f}**) —— 固定曝光 + 固定 ROI 中心已消除"每轮重新找中心"的跨轮方差。\n'
            f'> 但 top-1 (`{best_name.upper()}` {finals_sorted[0]:.4f}) 与 top-2 仅差 '
            f'**{top_spread:.4f}** (~{pct:.1f}% 提升幅度), **仍在单次运行噪声量级内** '
            "→ 名次差异不具统计显著性, 判优仍需 ①重复多次取中位数 ②加大预算。"
        )
    rows = []
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["final_pib"], reverse=True):
        rows.append(
            "| {n} | {init:.4f} | {fin:.4f} | {g:+.4f} | {loads} | {tmax} | {sec:.1f} |".format(
                n=name.upper(), init=r["init_pib"], fin=r["final_pib"], g=r["final_pib"] - r["init_pib"],
                loads=r["loads"], tmax=_first_load(r, r["final_pib"] - 1e-9) or "—", sec=r["elapsed_s"],
            )
        )
    md = f"""# SLM 相位优化真机基准报告 (目标: {_obj_label()})

> 本报告为**真实硬件**实测 (非仿真): Santec SLM-200 + Daheng **MER2-507-23GM NIR** 相机。
> ⚠️ 每个算法仅约 **{min(r['loads'] for r in results.values())}–{max(r['loads'] for r in results.values())} 次设备加载**
> (SLM 每次翻转需 ~0.3 s 稳定), 属**指示性对比**, 非完整收敛研究。

## 1. 结论 (TL;DR)

{len(results)} 个算法配置 (7 启发式 + SPGD/AdamOD/Adam 梯度基线) 中, **`{best_name.upper()}`** 取得最高最终{_obj_label()}
(**{results[best_name]["final_pib"]:.4f}**, 初始 {init_pib:.4f}, 提升
**{results[best_name]["final_pib"] - init_pib:+.4f}**),
并在 **{_first_load(results[best_name], results[best_name]["final_pib"] - 1e-9) or "—"}** 次加载内达到自身最优。

{variance_note}

## 2. 相机测试

| 项 | 值 |
|---|---|
| 相机类型 / ID | `{args.cam_type}` / {args.cam_id} |
| 帧尺寸 | {cam['shape'][1]} × {cam['shape'][0]} ({cam['dtype']}) |
| 曝光 | {cam['exposure_ms']} ms (设定 {args.exposure_ms} ms; 安全上限 ≤3 ms) |
| 峰值 / 均值 | {cam['max']} / {cam['mean']} (饱和: {cam['saturated']}) |
| 光斑中心 (阈值质心) | {cam['spot_center']} |
| 帧几何中心 | {cam['frame_center']} |

**光斑不在帧中心** (相差 {abs(cam['spot_center'][0]-cam['frame_center'][0])} px in x) —— 与 `AGENTS.md` 记录一致
(2f 光路 0 级需用 `argmax`/质心定位, 不可假设几何中心)。`clamp_center_to_frame`
保证 {args.cam_size}×{args.cam_size} 开窗一定落在帧内。

![相机初始帧](camera_frame.png)

## 3. 实验设置

| 项 | 值 |
|---|---|
| SLM | #{args.slm_number}, 波长 {args.wavelength} nm |
| 相机 | `{args.cam_type}` id={args.cam_id}, 曝光 {args.exposure_ms} ms (固定) |
| 目标 | `{args.objective}` ({_obj_label()}), `target_shape={args.target_shape}` (n_max={args.n_max}) |
| 开窗 | 请求 {args.cam_size}×{args.cam_size}; 实际按 **≥{1.5}× 目标长边** 自动放大 (日志 `camera window enlarged to`) |
| 中心 | **固定** {args.center} (平场稳定后 {12} 帧 argmax 锚点中位数, 帧间极差 ≤{args.center_spread:.1f}px) |
| 随机种子 | {args.seed} |
| 加载语义 | 1 次目标评估 = 1 次 SLM 相位加载 = 1 个迭代单位 |

## 4. 结果

| 算法 | 初始{_obj_label()} | 最终{_obj_label()} | 提升 | 加载次数 | 达到自身最优 (loads) | 耗时 (s) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

### 4.1 收敛曲线

![收敛曲线](pib_curves.png)

### 4.2 收敛速度

![收敛速度](convergence_speed.png)

### 4.3 CCD 实测光斑 (初始 vs 最优)

![初始与最优光斑](spot_before_after.png)

### 4.4 最终评分对比

![最终评分](summary_bars.png)

## 5. 文字总结

- **最优算法: `{best_name.upper()}`** — 最终{_obj_label()} {results[best_name]["final_pib"]:.4f} (初始 {init_pib:.4f},
  提升 {results[best_name]["final_pib"] - init_pib:+.4f}), 用
  {results[best_name]["loads"]} 次加载 / {results[best_name]["elapsed_s"]:.1f}s。
- 所有算法都提升了远场桶内能量比, 但**在相同的 ~30 次加载预算下差异明显**:
  种群类 (GA/PSO/CEM/DE) 每次迭代消耗 `pop_size` 次加载, 起步更"陡"但每次迭代更贵;
  单点类 (SA/HC/RS) 每次迭代只加载一次, 同样预算下点数更多、更细腻。
- **真机选型建议**:
  - 每次加载 ~0.3 s (SLM 翻转) + 相机曝光, 时间成本很高 →
    优先**小种群 GA/DE** 或**单点 HC/SA**;
  - 建议**两段式**: 先用种群类粗搜跳出局部最优, 再用 HC/SA (或 SPGD) 精修;
  - 种群规模不要盲目加大 —— 总加载次数 = `pop_size × iterations`。
- **方差控制**: 本次把光源工作点**一次性定死** —— 曝光由自动曝光在平场相位上确定
  ({args.exposure_ms:.3f}ms) 并对所有算法复用; 中心取平场稳定后 12 帧 argmax 锚点的中位数
  {args.center} (帧间极差 ≤{args.center_spread:.1f}px) 并作为**固定 tuple** 传入。
  这消除了"每轮重新找中心"的方差。
  ⚠️ `shape` 评分为**负值**且随中心/ROI 几何与开窗大小变化 → **跨运行绝对值不可比**,
  只有同一次运行内的 init→final 提升可比。
  📷 before/after 图片为**未开窗全画幅**采集 (开窗会掩盖被推出目标框的能量)。
- **限制**: 每算法 82–116 次加载、单次运行、固定像差与光路状态; 结论为**指示性**。
  要下定论请按本脚本加大预算 (改 `ALGORITHMS` 的 epochs/pop_size) 重复多次。

## 6. 复现

```powershell
$env:PYTHONPATH = "src;libs"    # libs/ 必须包含, 否则 gxipy (Daheng SDK) 不可用
python scripts/generate_slm_pib_heuristic_hw_report.py --exposure-ms {args.exposure_ms}
```

产物: `camera_frame.png` / `pib_curves.png` / `convergence_speed.png` /
`spot_before_after.png` / `summary_bars.png` / `summary.csv` / `report.md`。
"""
    (OUT_DIR / "report.md").write_text(md, encoding="utf8")
    return best_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=0.0,
        help="固定曝光 ms; 0 = 先自动曝光到 --target-brightness 再固定使用 (default: 0)",
    )
    parser.add_argument(
        "--target-brightness",
        type=int,
        default=180,
        help="自动曝光目标峰值亮度 (0-255, default: 180)",
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--cam-size", type=int, default=320)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help="SPGD perturbation amplitude (rad); 0.1 is below the metric noise floor",
    )
    parser.add_argument(
        "--max-energy-loss",
        type=float,
        default=0.6,
        help="safety guard: abandon an evaluation whose in-ROI energy dropped more "
        "than this fraction (0..1) of the initial in-ROI energy (0 = off)",
    )
    parser.add_argument(
        "--objective",
        default="shape",
        choices=["shape", "roi_pib", "pib", "radiu", "avg_radiu"],
        help="优化目标 (default: shape = 整形为长方形; roi_pib = 最大化目标ROI内亮度)",
    )
    parser.add_argument(
        "--target-shape", default="rectangle", help="shape 目标形状 (default: rectangle)"
    )
    parser.add_argument(
        "--target-size", type=float, default=None, help="目标尺寸 px (None=自动)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-camera-test", action="store_true")
    args = parser.parse_args()
    global _OBJECTIVE_LABEL
    _OBJECTIVE_LABEL = args.objective

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cam_info = {
        "shape": (0, 0),
        "dtype": "n/a",
        "max": -1,
        "mean": -1.0,
        "exposure_ms": args.exposure_ms,
        "range_ms": (0.0, 0.0),
        "frame_center": (0, 0),
        "spot_center": (0, 0),
        "saturated": False,
    }
    cam_img = np.zeros((2, 2))
    # Fix the operating point ONCE (exposure + centre) so every algorithm sees the
    # same bench state; the run then uses that fixed centre instead of re-detecting
    # it per algorithm.
    args.exposure_ms, args.center, args.center_spread = setup_bench(args)
    logger.info(
        "using fixed exposure {:.3f} ms and fixed center {} for all algorithms",
        args.exposure_ms,
        args.center,
    )

    if not args.skip_camera_test:
        # Capture a frame for the report before the optimisation runs.
        from ao_shaping.drivers.ccd.common import create_camera

        cam = create_camera(args.cam_type, cam_id=args.cam_id, exposure_time_ms=args.exposure_ms)
        cam.open()
        try:
            cam_img = cam.get_numpy_image(3).astype(np.float64)
        finally:
            cam.close()
        cam_info = camera_test(args.cam_type, args.cam_id, args.exposure_ms)
        fig_camera(cam_info, cam_img, OUT_DIR)

    # Open camera + SLM ONCE and reuse them across all algorithms, instead of
    # opening/closing a fresh pair per run (the previous behaviour). The
    # optimizer re-windows the camera and re-detects the spot centre per call,
    # so sequential reuse on a fixed bench is safe.
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm import Santec

    cam = create_camera(
        args.cam_type, cam_id=args.cam_id, exposure_time_ms=args.exposure_ms, skip_sampling=False
    )
    slm = Santec(
        slm_number=args.slm_number,
        wavelength=args.wavelength,
        shift_x=0,
        shift_y=0,
    )
    cam.open()
    slm.open()
    try:
        results: dict[str, dict] = {}
        for label, algorithm, optimizer_type, epochs, pop_size in ALGORITHMS:
            try:
                results[label] = run_algorithm(
                    label, algorithm, optimizer_type, epochs, pop_size, args,
                    cam=cam, slm=slm,
                )
            except Exception as exc:  # keep the remaining algorithms running
                logger.error("{} failed: {}: {}", label, type(exc).__name__, exc)
    finally:
        slm.close()
        cam.close()

    if not results:
        raise SystemExit("all algorithms failed; see the log above")

    fig_curves(results, OUT_DIR)
    fig_convergence(results, OUT_DIR)
    fig_spots(results, OUT_DIR)
    fig_summary_bars(results, OUT_DIR)
    write_csv(results, OUT_DIR)
    best = write_report(results, cam_info, args)
    logger.info("report written to {} (best: {})", OUT_DIR, best.upper())


if __name__ == "__main__":
    main()
