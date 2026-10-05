"""Sweep SPGD's perturbation amplitude on the real bench and pick a winner.

Runs the **genuine** ``slm_pib_runner`` CLI once per candidate ``delta`` and
judges each run by **convergence**, not by how much the objective moved from the
first epoch to the last.

=============================================================================
原理 (Principle)
=============================================================================

**1. SPGD 在做什么**

SPGD (Stochastic Parallel Gradient Descent) 每步随机生成一个扰动方向
``p ∈ {±1}^N`` (N = Zernike 系数个数), 然后测量两个值::

    J+ = score(c + delta·p)        J- = score(c - delta·p)
    g  ∝ (J+ - J-) · p              c ← c + lr · g

``delta`` 是**扰动幅度** (rad)。它决定"信号": ``delta`` 越大, 两个相位屏的
差异越大, 目标函数的差异也越大。

**2. 为什么不能只看"目标值降了多少"**

直觉上"降得多 = delta 选得好"。但这是**错的**, 而且在这个台架上被实测证伪:

    真实轨迹 (200 epoch, n_max=9, delta=0.02):
    0.529 → 0.400 → 0.522 → 0.527 → 0.533 → 0.534 → ... → 0.456 → 0.402

首尾一比是 **+24%**, 但轨迹先掉到 0.400、又**完全涨回** 0.534、最后落在
0.402 —— 全程没有下降趋势。随机游走只要"碰巧停在低点"就会被记成一次胜利。
在 51 个已记录 run 中, 有 **32 个**的 `dec` 在 0.5±0.05 内, 即抛硬币。

**3. 本工具用的两个稳健判据**

``dec`` (frac_decreasing) = 下降步占比。
    真正收敛的 run 会显著 > 0.5; ``= 0.5`` 就是随机游走。这是主判据。

``late`` (late_gain) = 前 1/3 均值 → 后 1/3 均值 的改善百分比。
    对"停在哪一步"不敏感, 不会被端点噪声欺骗。

**4. 噪声预算: 为什么 delta 调不动它**

每步梯度估计的误差来自 ``J+`` 与 ``J-`` 两次采样之间的**共模变化**::

    误差 ∝ |J(+δ) − 噪声 + 漂移|  vs  |J(−δ) − 噪声 − 漂移|

  - **信号** ∝ delta        → 加大 delta 可以提高信噪比
  - **慢漂移** 与 delta 无关 → 加大 delta **不能**提高信噪比
  - **散粒噪声** 可用多帧平均压低 (∝1/√N), 漂移**不可**

实测确认: 同相位 40 帧平均并未降低 σ (2.8e-3 vs 10 帧的 1.7e-3), 且
单日三次扫描 σ 相差 **21×** (3.7e-4 → 1.7e-5)。所以噪声是**慢漂移**。

由于 SPGD 的 ``J+``/``J-`` 是**相邻两次**采样, 漂移在其间近似恒定并直接进入
梯度估计 —— 这是 95%+ 迭代被噪声门拒绝的根因。

**5. 为什么本工具"拒绝"给推荐值**

``explore_delta().recommended`` 在没有候选真正收敛时返回 ``None``, 而不是
退而求其次给"最好的那个"。**拒绝封王才是本工具的意义**: 一个被标成
"+24% 改善"的随机游走若被选中, 会被当成结论继续用下去。

**6. 实机扫描结论 (2026-09-30, n_max=9, 150-200 epoch, Pearson)**

===============  =======  ========  ========  ==========================
delta             dec     late %   guard %  结论
===============  =======  ========  ========  ==========================
0.0005            0.52     -1.4      0.0   停滞 (first==final)
0.001             0.52     +0.1      0.0   停滞
0.02              0.56     +8.5      0.0   随机游走
0.05              0.53    +21.5      0.0   随机游走
0.10              0.43    -16.7     31.4   折叠门大量拒绝
0.20              0.52     +8.9      0.0   随机游走
0.30              0.51     +1.7      0.7   随机游走
0.50              0.43     -1.7     84.1   折叠门主导
===============  =======  ========  ========  ==========================

**delta 跨越 1000 倍, ``dec`` 始终 ≈ 0.5。** 所以:

  - **delta 不是限制因素**, 再调它也解决不了问题;
  - 小 delta (5e-4) 确实彻底停滞 (符合用户约束 `delta<0.001` 的历史观察);
  - 大 delta (0.5) 被**亮度折叠门**拒绝 —— 扰动过猛导致亮度崩塌, 这是上界;
  - 把 n_max 从 9 降到 1 (54 → 3 自由度) 反而更差 (3/200 采纳), 说明
    "自由度稀释"能解释 SNR 数值, **不能**解释实机为什么不收敛。

**7. 真正的下一步: 把回文采样接入 SPGD 主循环**

漂移是共模的, 所以只要让 ``J+`` 与 ``J-`` 的采样在时间上**对称**, 就能抵消::

    现在:  J+ = score(+δ)                 →  漂移 d
           J- = score(−δ)                 →  漂移 d'      差分残留 (d−d')

    回文:  J+ = score(+δ)                 →  漂移 d1
           J- = score(−δ)                 →  漂移 d2
           J- = score(−δ)                 →  漂移 d3      d1+d3 ≈ d2+d4
           J+ = score(+δ)                 →  漂移 d4         (线性漂移下精确抵消)

``ao_shaping.tools.slm.sweep_analysis.abba_signal`` 已实现并单测 (``+ - - +``
可精确对消线性漂移), 但**主循环仍是单次正负采样** —— 这是投入产出比最高的改动。

=============================================================================

**Usage:**
    python scripts/explore_delta.py --deltas 0.02,0.05,0.1
    python scripts/explore_delta.py --deltas 0.05,0.1 --objective shape --epochs 100
    python scripts/explore_delta.py --deltas 0.05,0.1 --n-max 3
    python scripts/explore_delta.py --analyze-only --deltas 0.05,0.1   # 无硬件重算

**Requires hardware** (Santec SLM + camera) unless ``--analyze-only``. Devices
are opened and closed per candidate by the runner itself, and every run keeps
``--debug`` so its recorder pickle / JSON sidecar / summary PNG stay under
``data/debug/`` for offline re-analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

from loguru import logger  # noqa: E402

from ao_shaping.tools.slm.sweep_analysis import explore_delta  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sweep SPGD delta on the bench and recommend the one that CONVERGES "
            "(not the one with the biggest first-vs-last jump)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--deltas", default="0.02,0.05,0.1",
        help="comma-separated perturbation amplitudes (rad)",
    )
    p.add_argument("--epochs", type=int, default=200, help="epochs per candidate")
    p.add_argument(
        "--objective", default="pearson",
        help="shaping objective (pearson / shape / roi_pib / ...)",
    )
    p.add_argument("--n-max", type=int, default=9, help="max Zernike radial order")
    p.add_argument("--lr", type=float, default=0.5, help="SPGD learning rate")
    p.add_argument(
        "--cam-type", default="daheng", help="camera backend (daheng / miicam)"
    )
    p.add_argument("--cam-id", type=int, default=0, help="camera id")
    p.add_argument("--cam-size", type=int, default=320, help="ROI window (px)")
    p.add_argument(
        "--exposure-ms", type=float, default=1.2, help="camera exposure (ms)"
    )
    p.add_argument("--zernike-radius", type=float, default=480.0, help="aperture (px)")
    p.add_argument("--target-size", type=float, default=50.0, help="target size (px)")
    p.add_argument("--target-shape", default="square", help="target shape")
    p.add_argument(
        "--out", default=str(ROOT / "docs" / "slm_pib_bench" / "delta_scan.md"),
        help="where to write the markdown summary",
    )
    p.add_argument(
        "--analyze-only", action="store_true",
        help=(
            "do not touch hardware; re-judge the newest existing debug run per "
            "delta (use after editing the judging logic, costs nothing)"
        ),
    )
    return p


def _newest_recorder(objective: str, delta: float | None = None):
    """Load the most recent debug pickle for ``objective`` (optionally a delta).

    Run directories are ``slm_pib_<objective>_<timestamp>/``, so the glob must be
    ``slm_pib_<objective>_*`` — a bare ``slm_pib_<objective>/`` matches nothing.
    """
    import pickle

    pattern = f"data/debug/slm_pib_{objective}_*/**/*.pkl"
    files = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    if delta is not None:
        exact = [p for p in files if _delta_of(p) == delta]
        if exact:
            files = exact
    if not files:
        return None
    return pickle.loads(files[-1].read_bytes())


def _delta_of(pkl_path: Path) -> float | None:
    """Read ``delta`` out of a run's JSON sidecar, if present."""
    import json

    for js in sorted(pkl_path.parent.glob("**/*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "delta" in data:
            try:
                return float(data["delta"])
            except (TypeError, ValueError):
                return None
    return None


def main() -> int:
    args = build_parser().parse_args()
    deltas = [float(d) for d in str(args.deltas).split(",") if d.strip()]
    if not deltas:
        logger.error("--deltas is empty")
        return 1

    from ao_shaping.runners.slm_pib_runner import run as slm_pib_run

    def run_one(delta: float):
        if args.analyze_only:
            return _newest_recorder(args.objective, delta)

        click_args = [
            "spgd",
            "-d", "data",
            "--debug",
            "--cam_type", args.cam_type,
            "--cam-id", str(args.cam_id),
            "--exposure_time_ms", str(args.exposure_ms),
            "--cam_size", str(args.cam_size),
            "-c", "max",
            "--slm_number", "1",
            "--slm_wavelength", "1064",
            "-n", str(args.n_max),
            "--zernike_radius", str(args.zernike_radius),
            "--target_shape", args.target_shape,
            "--target_size", str(args.target_size),
            "--objective", args.objective,
            "-e", str(args.epochs),
            "--delta", str(delta),
            "--lr", str(args.lr),
        ]
        # The Click entry writes its own debug artefacts and closes the devices.
        slm_pib_run.main(args=click_args, standalone_mode=False)
        # Re-read the freshest recorder so the analysis uses this run's trace.
        return _newest_recorder(args.objective, delta)

    logger.info(
        "sweeping {} deltas x {} epochs (objective={}, n_max={}, analyze_only={})",
        len(deltas), args.epochs, args.objective, args.n_max, args.analyze_only,
    )
    scan = explore_delta(
        run_one,
        deltas,
        progress=lambda d, i, n: logger.info("[{}/{}] delta={}", i, n, d),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SPGD delta 扫描 (实机)",
        "",
        f"- 目标 `{args.objective}`, n_max={args.n_max}, epochs={args.epochs}, "
        f"lr={args.lr}, 相机 `{args.cam_type}`/{args.cam_id}, "
        f"开窗 {args.cam_size}px, 曝光 {args.exposure_ms}ms",
        f"- 每个候选都带 `--debug`, recorder/sidecar/PNG 落在 `data/debug/slm_pib_{args.objective}/`",
        "",
        "判据: `dec` = 下降步占比 (0.5 为随机游走), `late` = 前 1/3 与后 1/3 均值之差。",
        "**不使用 final-vs-first** —— 实机上它会被端点噪声骗到 (见 docs/slm_pib_bench/EXPERIMENT_REPORT.md §3.2)。",
        "",
        scan.table(),
        "",
    ]
    if scan.recommended is None:
        lines += [
            "> **没有候选真正收敛。** 所有 delta 的 `dec` 都在 0.5 附近 = 随机游走。",
            "> 这不是 delta 选错, 而是每步 SPGD 信噪比不足; 优先排查慢漂移对梯度估计的污染",
            "> (把 `+ - - +` 回文采样接入主循环), 再回来扫 delta。",
        ]
    else:
        best = scan.converged[0]
        lines += [
            f"> **推荐 `--delta {scan.recommended:g}`** "
            f"(dec={best.frac_decreasing:.2f}, late={best.late_gain_pct:+.1f}%)",
        ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote {}", out)
    print()
    print(scan.table())
    print()
    print(f"recommended delta: {scan.recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())