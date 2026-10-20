"""issue #165 P1 实验系统 A/B：GSNet 初值 vs 物理初值（交错重放）。

回答的问题只有一个：**把网络预测的相位当作 model-in-loop 的 Step B 热启动，
在实验系统上到底有没有增益？**

协议（issue #165 验收条款「硬件运行」）：

* **交错重放**（interleaved replay）：同一会话内交替跑 `gsnet` 臂与 `physical` 臂，
  第 i 对的顺序按 i 的奇偶翻转（偶数对 gsnet 先、奇数对 physical 先），
  以抵消「先跑的那一臂占了更好的光功率时段」这一系统误差。
* **同一 seed、同一目标、同一探针数、同一曝光档位**：两臂只差一个开关
  （`--init-from-gsnet`）。
* **报均值差与 p 值**，不报单轮 best 分数——本台架的分数分布是双峰的
  （见 `report/shape_trials/gs_refine_live_20261012.md`：25.7k / 33.6k 两簇），
  单轮 best 不得作为可复现增益声明。
* **每一次 MIL 运行前都过光功率门**（`scripts/slm/diagnose_light_level.py`）：
  没有光就没有可整形的东西，campaign 应当停下，而不是去怪正向模型。

本脚本**不含任何设备代码**：硬件命令行由 `scripts/slm/model_in_loop_auto.py` 构造，
判定由 `ao_shaping.tools.slm.model_in_loop_summary` 给出；本文件只负责排序、
收集证据、写清单。`--plan` 在设备离线时打印整份 campaign 供复核。

用法::

    python scripts/slm/gsnet_init_hw_ab.py --plan
    python scripts/slm/gsnet_init_hw_ab.py --gsnet-phase /path/to/phase.npy
    python scripts/slm/gsnet_init_hw_ab.py --pairs 3 --skip-light-gate

退出码::

    0  campaign 跑完（未达预期是**结果**，不是错误）
    1  某个 stage 意外失败（崩溃 / 超时 / 产物不可读）
    2  campaign 被拒绝：光功率门挡住，或 GSNet 相位文件不合法
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.model_in_loop_summary import summarize_session  # noqa: E402
from ml.zernike.eval_stats import paired_comparison  # noqa: E402

AUTO_SCRIPT = ROOT / "scripts" / "slm" / "model_in_loop_auto.py"
LIGHT_SCRIPT = ROOT / "scripts" / "slm" / "diagnose_light_level.py"
MIL_SUBCOMMAND = "slm-model-in-loop"

ARMS = ("gsnet", "physical")


# ---------------------------------------------------------------------------
# 命令构造
# ---------------------------------------------------------------------------
def _load_auto_module():
    """按文件路径加载 ``model_in_loop_auto.py``（它不在包内）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("model_in_loop_auto", AUTO_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AUTO_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    """两臂共用的 MIL profile。"""
    return {
        "cam_type": args.cam_type,
        "cam_id": args.cam_id,
        "slm_number": args.slm_number,
        "slm_wavelength": args.slm_wavelength,
        "target_shape": args.target_shape,
        "target_side": args.target_side,
        "region": args.region,
        "far_field_padding": args.far_field_padding,
        "n_rounds": args.n_rounds,
        "exposure_time_ms": args.exposure_time_ms,
        "max_exposure_ms": args.max_exposure_ms,
        "cam_size": args.cam_size,
        "panel_span_px": args.panel_span_px,
        "camera_pixel_um": args.camera_pixel_um,
        "slm_pixel_um": args.slm_pixel_um,
        "probe_count": args.probe_count,
        "holdout_probe_count": args.holdout_probe_count,
        "step_b_iterations": args.step_b_iterations,
        "seed": args.seed,
        "max_cv_regression": args.max_cv_regression,
        "min_geometry_correlation": args.min_geometry_correlation,
        "min_holdout_correlation": args.min_holdout_correlation,
        "max_cv": args.max_cv,
        "min_ee": args.min_ee,
        "min_iou": args.min_iou,
    }


def build_light_command(args: argparse.Namespace, out_root: Path) -> list[str]:
    return [
        sys.executable,
        str(LIGHT_SCRIPT),
        "--cam-type",
        args.cam_type,
        "--cam-id",
        str(args.cam_id),
        "--slm-number",
        str(args.slm_number),
        "--slm-wavelength",
        str(args.slm_wavelength),
        "--exposures",
        str(args.light_exposures),
        "--out-root",
        str(out_root),
    ]


def with_output_root(command: list[str], root: Path) -> list[str]:
    """把 runner 的 ``--dir`` 指向本 stage 自己的目录。"""
    index = command.index(MIL_SUBCOMMAND)
    return command[:index] + ["--dir", str(root)] + command[index:]


def build_arm_command(
    args: argparse.Namespace, auto, arm: str, stage_dir: Path
) -> list[str]:
    """构造一臂的 MIL 命令行。

    ``gsnet`` 臂多一个 ``--init-from-gsnet <phase>``；其余参数两臂完全一致。
    """
    profile = build_profile(args)
    if arm == "gsnet":
        profile["init_from_gsnet"] = str(args.gsnet_phase)
    command = with_output_root(auto.build_command(profile), stage_dir)
    return command


def stage_order(pairs: int) -> list[tuple[int, str]]:
    """交错重放的执行顺序：第 i 对里两臂的先后按 i 的奇偶翻转。"""
    order: list[tuple[int, str]] = []
    for index in range(pairs):
        arms = list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
        for arm in arms:
            order.append((index, arm))
    return order


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------
def build_plan(args: argparse.Namespace, session_dir: Path) -> dict[str, Any]:
    auto = _load_auto_module()
    stages: list[dict[str, Any]] = []
    for index, arm in stage_order(args.pairs):
        name = f"pair{index:02d}_{arm}"
        stage_dir = session_dir / name
        stages.append(
            {
                "name": name,
                "pair": index,
                "arm": arm,
                "command": build_arm_command(args, auto, arm, stage_dir),
                "gate_command": (
                    None
                    if args.skip_light_gate
                    else build_light_command(args, stage_dir / "light")
                ),
            }
        )
    return {
        "session_dir": str(session_dir),
        "gsnet_phase": str(args.gsnet_phase),
        "pairs": args.pairs,
        "stages": stages,
    }


def render_plan(plan: dict[str, Any]) -> str:
    lines = [
        "=== issue #165 GSNet-init hardware A/B campaign (plan; no device is opened) ===",
        f"session root: {plan['session_dir']}",
        f"gsnet phase:  {plan['gsnet_phase']}",
        f"pairs:        {plan['pairs']} (interleaved, order flipped every other pair)",
        "",
    ]
    for stage in plan["stages"]:
        lines.append(f"[{stage['name']}] arm={stage['arm']} pair={stage['pair']}")
        if stage["gate_command"]:
            lines.append(f"  gate:  {subprocess.list2cmdline(stage['gate_command'])}")
        lines.append(f"  run:   {subprocess.list2cmdline(stage['command'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _run(command: list[str], timeout_s: float, cwd: Path) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"timeout after {timeout_s} s",
            "seconds": time.time() - started,
        }
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "seconds": time.time() - started,
    }


def _session_dir_from_output(stdout: str) -> Path | None:
    import re

    matches = re.findall(r"(?:Recorder pkl/json\s*:|Preflight evidence\s*:)\s*(.+)", stdout)
    return Path(matches[-1].strip()) if matches else None


def collect_stage(plan: dict[str, Any], stage: dict[str, Any], args) -> dict[str, Any]:
    """跑一个 stage 并收集它的证据。"""
    record: dict[str, Any] = {
        "name": stage["name"],
        "pair": stage["pair"],
        "arm": stage["arm"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    stage_dir = Path(plan["session_dir"]) / stage["name"]
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage["gate_command"]:
        gate = _run(stage["gate_command"], args.timeout_s, ROOT)
        record["gate"] = {
            "returncode": gate["returncode"],
            "stdout_tail": gate["stdout"][-2000:],
            "stderr_tail": gate["stderr"][-2000:],
        }
        if gate["returncode"] != 0:
            record["status"] = "blocked_by_light_gate"
            return record

    run = _run(stage["command"], args.timeout_s, ROOT)
    record["run"] = {
        "returncode": run["returncode"],
        "stdout_tail": run["stdout"][-4000:],
        "stderr_tail": run["stderr"][-2000:],
        "seconds": run["seconds"],
    }
    if run["returncode"] != 0:
        record["status"] = "stage_failed"
        return record

    session_dir = _session_dir_from_output(run["stdout"])
    if session_dir is None or not session_dir.exists():
        record["status"] = "no_session_dir"
        return record
    record["session_dir"] = str(session_dir)

    try:
        summary = summarize_session(
            session_dir,
            max_cv=args.max_cv,
            min_ee=args.min_ee,
            min_iou=args.min_iou,
        )
    except Exception as exc:  # noqa: BLE001 - 采集失败是结果，不是崩溃
        record["status"] = "summary_failed"
        record["summary_error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["summary"] = summary
    record["status"] = "ok"
    return record


def _score_of(record: dict[str, Any]) -> float | None:
    summary = record.get("summary") or {}
    metrics = summary.get("metrics_at_best_recorded_round") or {}
    value = metrics.get("score_after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(record: dict[str, Any], key: str) -> float | None:
    summary = record.get("summary") or {}
    metrics = summary.get("metrics_at_best_recorded_round") or {}
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyse(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按对配对，报均值差与精确 sign-flip p 值。"""
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(record["pair"], {})[record["arm"]] = record

    paired_scores: list[float] = []
    paired_cv: list[float] = []
    paired_ee: list[float] = []
    incomplete: list[int] = []
    for index, arms in sorted(by_pair.items()):
        gsnet = arms.get("gsnet")
        physical = arms.get("physical")
        if not gsnet or not physical:
            incomplete.append(index)
            continue
        gsnet_score = _score_of(gsnet)
        physical_score = _score_of(physical)
        if gsnet_score is None or physical_score is None:
            incomplete.append(index)
            continue
        paired_scores.append(gsnet_score - physical_score)
        gsnet_cv = _metric(gsnet, "cv")
        physical_cv = _metric(physical, "cv")
        if gsnet_cv is not None and physical_cv is not None:
            paired_cv.append(gsnet_cv - physical_cv)
        gsnet_ee = _metric(gsnet, "ee")
        physical_ee = _metric(physical, "ee")
        if gsnet_ee is not None and physical_ee is not None:
            paired_ee.append(gsnet_ee - physical_ee)

    return {
        "n_complete_pairs": len(paired_scores),
        "incomplete_pairs": incomplete,
        "score": paired_comparison(paired_scores, "gsnet - physical (composite score)"),
        "cv": paired_comparison(paired_cv, "gsnet - physical (cv_target_box)"),
        "ee": paired_comparison(paired_ee, "gsnet - physical (EE)"),
        "note": (
            "单轮 best 分数不得作为可复现增益声明；本台架分数分布呈双峰，"
            "结论只认交错重放的配对均值差与 p 值。"
        ),
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "pair",
                "arm",
                "status",
                "score",
                "cv",
                "ee",
                "iou",
                "holdout_correlation",
                "geometry_correlation",
                "verdict",
                "session_dir",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.get("pair"),
                    record.get("arm"),
                    record.get("status"),
                    _score_of(record),
                    _metric(record, "cv"),
                    _metric(record, "ee"),
                    _metric(record, "iou"),
                    _metric(record, "holdout_correlation"),
                    _metric(record, "geometry_correlation"),
                    (record.get("summary") or {}).get("verdict"),
                    record.get("session_dir"),
                ]
            )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gsnet-phase", type=Path, required=True, help="网络预测的瞳面相位 .npy")
    parser.add_argument("--pairs", type=int, default=3, help="每臂的重放次数（交错）")
    parser.add_argument("--out-root", type=Path, default=None, help="campaign 根目录")
    parser.add_argument("--plan", action="store_true", help="只打印计划，不打开设备")
    parser.add_argument("--skip-light-gate", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=5400.0)
    parser.add_argument("--cam-type", choices=["daheng", "miicam"], default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument("--target-shape", default="square")
    parser.add_argument("--target-side", type=int, default=100)
    parser.add_argument("--region", type=int, default=128)
    parser.add_argument("--far-field-padding", type=int, default=8)
    parser.add_argument("--n-rounds", type=int, default=1)
    parser.add_argument("--probe-count", type=int, default=6)
    parser.add_argument("--holdout-probe-count", type=int, default=2)
    parser.add_argument("--step-b-iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--exposure-time-ms", default="auto")
    parser.add_argument("--max-exposure-ms", type=float, default=0.4)
    parser.add_argument("--cam-size", type=int, default=300)
    parser.add_argument("--panel-span-px", type=float, default=900.0)
    parser.add_argument("--camera-pixel-um", type=float, default=2.2)
    parser.add_argument("--slm-pixel-um", type=float, default=8.0)
    parser.add_argument("--max-cv-regression", type=float, default=0.02)
    parser.add_argument("--min-geometry-correlation", type=float, default=0.5)
    parser.add_argument("--min-holdout-correlation", type=float, default=0.5)
    parser.add_argument("--max-cv", type=float, default=0.5)
    parser.add_argument("--min-ee", type=float, default=0.4)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--light-exposures", default="0.02,0.05,0.1,0.2,0.4")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.gsnet_phase.exists():
        print(f"GSNet 相位文件不存在: {args.gsnet_phase}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or (ROOT / "data" / "debug" / "gsnet_init_hw_ab" / stamp)
    out_root.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args, out_root)
    (out_root / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(render_plan(plan))

    if args.plan:
        return 0

    records: list[dict[str, Any]] = []
    for stage in plan["stages"]:
        record = collect_stage(plan, stage, args)
        records.append(record)
        print(
            f"[{record['name']}] status={record['status']} "
            f"score={_score_of(record)} cv={_metric(record, 'cv')} ee={_metric(record, 'ee')}",
            flush=True,
        )
        if record["status"] == "blocked_by_light_gate":
            break

    analysis = analyse(records)
    manifest = {
        "session_dir": str(out_root),
        "gsnet_phase": str(args.gsnet_phase),
        "pairs_requested": args.pairs,
        "records": records,
        "analysis": analysis,
    }
    (out_root / "campaign.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(records, out_root / "runs.csv")

    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    print(f"campaign 产物: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
