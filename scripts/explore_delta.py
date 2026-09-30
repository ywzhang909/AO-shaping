"""Sweep SPGD's perturbation amplitude on the real bench and pick a winner.

Runs the **genuine** ``slm_pib_runner`` CLI once per candidate ``delta`` and
judges each run by **convergence**, not by how much the objective moved from the
first epoch to the last. Every run is written with ``--debug`` so its recorder
pickle, JSON sidecar and summary PNG land under ``data/debug/`` for offline
re-analysis.

Why not just take the biggest improvement? Because that metric is an artefact
on a noisy bench. A real 200-epoch trace from this bench read

    0.529 -> 0.400 -> 0.522 -> 0.527 -> 0.533 -> 0.534 -> ... -> 0.456 -> 0.402

whose first-vs-last improvement is **+24 %** while the run never descended — it
dipped, fully recovered, and stopped low. Across 51 recorded runs, 32 had a
decreasing-step fraction within 0.05 of 0.5, i.e. were coin flips. This tool
reports ``dec`` (decreasing-step fraction) and ``late`` (first-third vs
last-third mean) and refuses to recommend a delta when nothing truly converged.

The judging logic is :mod:`ao_shaping.tools.slm.delta_explorer` (device-agnostic,
unit-tested against synthetic traces of known behaviour); this file only wires it
to the hardware CLI.

**Usage:**
    python scripts/explore_delta.py --deltas 0.02,0.05,0.1
    python scripts/explore_delta.py --deltas 0.05,0.1 --objective shape --epochs 100
    python scripts/explore_delta.py --deltas 0.05,0.1 --n-max 3

**Requires hardware** (Santec SLM + camera). Devices are opened and closed per
candidate by the runner itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

from loguru import logger  # noqa: E402

from ao_shaping.tools.slm.delta_explorer import explore_delta  # noqa: E402


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