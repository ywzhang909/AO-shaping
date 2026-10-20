"""issue #165 P1 仿真验收：model-in-loop 闭环的「网络初值 vs 物理初值」A/B。

**纯离线仿真**（``--cam-type sim``），不打开任何设备。

协议（issue #165 验收条款「sim 先行」）：

* 同一台架、同一目标、同一探针数、同一 seed，只差一个开关
  （``--init-from-gsnet``）。
* 三臂：
  1. ``physical`` —— 不给初值（零相位起步），这是对照臂；
  2. ``gsnet_full`` —— 把**闭环自己上一轮的最优相位**（rms ≈ 1.7 rad）当初值。
     这代表「网络给出了一个真正的整形相位」时会发生什么；
  3. ``gsnet_scaled`` —— 同一个相位缩放到 rms ≈ 0.43 rad，用来单独检验
     信任域这一道门（第一臂会被它挡掉）。
* 每一臂都记录 ``init_from_gsnet_status``：``disabled`` /
  ``trust_region_rejected`` / ``flat_bake_off_rejected`` /
  ``holdout_correlation_rejected`` / ``adopted``。

**判据**：带初值的臂**不得劣于**对照臂（issue 的「劣化即关闭开关回退」）。

用法::

    python scripts/slm/gsnet_init_sim_ab.py
    python scripts/slm/gsnet_init_sim_ab.py --out data/debug/gsnet_init_sim_ab/<ts>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MAIN = ROOT / "src" / "ao_shaping" / "main.py"

#: 三臂共用的一组闭环参数。target-side 用**相机像素**；仿真台架的
#: camera_px_per_model_px = 8，所以 288 相机像素 = 36 模型像素的目标方框——
#: 与 issue 的 36×36 口径一致。平场的 0 阶斑在这个方框里 CV 高达 12，
#: 整形才有可测的改善（目标方框只有 4 模型像素时平场就「赢」了）。
COMMON_ARGS = [
    "--cam-type", "sim",
    "--n-rounds", "3",
    "--probe-count", "4",
    "--holdout-probe-count", "2",
    "--step-a-iterations", "40",
    "--step-b-iterations", "200",
    "--region", "64",
    "--far-field-padding", "8",
    "--target-side", "288",
    "--cam-size", "400",
]

_SCORE = re.compile(r"Best shaped score\s*:\s*([0-9.]+)")
_FLAT = re.compile(r"Flat baseline score\s*:\s*([0-9.]+)")
_STATUS = re.compile(r"init_from_gsnet ([a-z_]+)")
_ROUND = re.compile(r"^\s*(\d+)\s+([YN])\s+(\S.*?)\s{2,}([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")


def run_arm(
    out_dir: Path, extra: list[str], label: str, timeout_s: float
) -> dict[str, Any]:
    """跑一条 model-in-loop 仿真命令并抽取它的判决。"""
    arm_dir = out_dir / label
    arm_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(MAIN),
        "--debug",
        "slm-model-in-loop",
        *COMMON_ARGS,
        *extra,
    ]
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    stdout = completed.stdout
    (arm_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (arm_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    # loguru writes to stderr, so the gate's verdict line lives there.
    combined = stdout + "\n" + completed.stderr

    best = _SCORE.search(combined)
    flat = _FLAT.search(combined)
    status = _STATUS.search(combined)
    rounds = [
        {
            "round": int(m.group(1)),
            "accepted": m.group(2) == "Y",
            "reason": m.group(3).strip(),
            "score": float(m.group(4)),
            "cv": float(m.group(5)),
            "ee": float(m.group(6)),
        }
        for m in _ROUND.finditer(combined)
        if m.group(4) not in {"nan", ""}
    ]
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.time() - started,
        "flat_score": float(flat.group(1)) if flat else None,
        "best_score": float(best.group(1)) if best else None,
        "init_from_gsnet_status": status.group(1) if status else "disabled",
        "rounds": rounds,
        "stdout_tail": stdout[-1500:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 data/debug/gsnet_init_sim_ab/<时间戳>）",
    )
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="第三臂的相位缩放系数（用来单独检验信任域那一道门）",
    )
    args = parser.parse_args(argv)

    out_dir = args.out or (
        ROOT / "data" / "debug" / "gsnet_init_sim_ab" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    physical = run_arm(out_dir, [], "physical", args.timeout_s)

    # 从对照臂自己的产物里取出最优相位，代表「一个真正的整形相位」
    phase_path: Path | None = None
    run_dirs = sorted(
        (ROOT / "data" / "debug" / "slm_model_in_loop").glob("*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in run_dirs:
        best = candidate / "best_phase.npy"
        if best.exists():
            phase_path = best
            break
    if phase_path is None:
        print("对照臂没有产出 best_phase.npy，无法构造初值臂", file=sys.stderr)
        return 1

    import numpy as np

    phase = np.load(phase_path)
    scaled_path = out_dir / "gsnet_phase_scaled.npy"
    np.save(scaled_path, phase * args.scale)

    gsnet_full = run_arm(
        out_dir, ["--init-from-gsnet", str(phase_path)], "gsnet_full", args.timeout_s
    )
    gsnet_scaled = run_arm(
        out_dir,
        ["--init-from-gsnet", str(scaled_path)],
        "gsnet_scaled",
        args.timeout_s,
    )

    baseline = physical["best_score"]
    rows = []
    for arm in (physical, gsnet_full, gsnet_scaled):
        score = arm["best_score"]
        rows.append(
            {
                "arm": arm["label"],
                "init_from_gsnet_status": arm["init_from_gsnet_status"],
                "flat_score": arm["flat_score"],
                "best_score": score,
                "delta_vs_physical": (
                    None
                    if score is None or baseline is None
                    else float(score - baseline)
                ),
                "not_worse_than_physical": (
                    None
                    if score is None or baseline is None
                    else bool(score >= baseline - 1e-9)
                ),
                "returncode": arm["returncode"],
                "seconds": arm["seconds"],
            }
        )

    result = {
        "common_args": COMMON_ARGS,
        "phase_source": str(phase_path),
        "phase_rms_rad": float(np.sqrt((phase**2).mean())),
        "scaled_phase_rms_rad": float(np.sqrt(((phase * args.scale) ** 2).mean())),
        "arms": rows,
    }
    (out_dir / "sim_ab.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "sim_ab.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['arm']:<13} status={row['init_from_gsnet_status']:<28} "
            f"flat={row['flat_score']} best={row['best_score']} "
            f"delta={row['delta_vs_physical']}"
        )
    print(f"产物目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
