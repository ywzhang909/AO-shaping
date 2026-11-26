"""实验系统光路复用门试跑：跑一次复用门并按计划 §4 校验结果可信度。

计划: docs/dev/next_hw_plan_optical_path_reuse_gate_trial_20261124.md

本脚本**调用** :func:`optical_path_reuse_gate.run_reuse_gate` 在实验系统上跑一次复用门
（持锁、开 SLM+CCD、拍一帧当前平场、单曝光饱和校验、``assess_reuse`` 判一致、fail-closed），
再按计划 §4 通过判据校验其返回结果与退出码是否**可信**：

- 退出码 ∈ {0, 3} 且与 ``reuse_gate`` 一致；
- JSON schema = ``optical_path_reuse_gate/1``、``closed_normally=true``、结果形状完整；
- ``passed`` → ``exposure_check.passed=true``、``drive_board_id`` 非空、``criteria`` 非空、``reference`` 存在；
- ``failed`` → ``reasons`` 非空、``remedy`` 可行动。

试跑**不**替代复用门：门的 passed/failed 仍由 ``optical_path_reuse_gate.py`` 决定。
本脚本只判「这个门可信」。无论门结果 passed 或 failed，只要结果自洽，试跑都算**有效**；
结果不自洽 / 门崩溃 / 退出码异常 → 试跑**无效**（退出码 1），须先修门再试。

本脚本不产生归因/冻结数据，不改 issue #247 状态；实验运行前须报告设备占用并 ``--wait-until-idle``。

用法::

    python scripts/slm/optical_path_reuse_gate_trial.py \
        --out-root data/debug/optical_path_probe \
        --run-dir data/debug/optical_path_reuse_gate_trial_20261124 \
        --wait-until-idle 600
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT, ROOT / "src", ROOT / "libs", ROOT / "scripts" / "slm"):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import optical_path_reuse_gate as gate

#: 试跑判读结果文件名（写在 ``--run-dir`` 下）。
TRIAL_FILE = "optical_path_reuse_gate_trial.json"
#: 试跑判读结果 schema。
TRIAL_SCHEMA = "optical_path_reuse_gate_trial/1"
#: 试跑有效（门结果可信）。
EXIT_TRIAL_VALID = 0
#: 试跑无效（门结果不可信，须先修门）。
EXIT_TRIAL_INVALID = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_trial(result: Any, gate_exit: int | None) -> dict[str, Any]:
    """纯函数：按计划 §4 校验复用门结果是否可信（离线可测，不碰设备）。

    Args:
        result: :func:`optical_path_reuse_gate.run_reuse_gate` 返回的结果 dict；
            门崩溃 / 未返回时为 ``None``。
        gate_exit: 门的退出码（0 / 3）；门未正常返回时为 ``None``。

    Returns:
        判读 dict：``trial`` ∈ {"valid","invalid"}、``checks`` 逐项明细、
        ``reasons`` 为失败项明细，另附 ``reuse_gate`` / ``gate_exit``。
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if not isinstance(result, dict):
        check(
            "result_shape",
            False,
            f"复用门结果是 {type(result).__name__}，不是 dict（门可能崩溃）",
        )
    else:
        reuse = result.get("reuse_gate")
        check("reuse_gate_value", reuse in ("passed", "failed"), f"reuse_gate={reuse!r}")

        if gate_exit is None:
            check("gate_exit_consistent", False, "复用门未返回退出码（异常 / 中断）")
        else:
            expected = gate.EXIT_PASSED if reuse == "passed" else gate.EXIT_FAILED
            check(
                "gate_exit_consistent",
                gate_exit in (gate.EXIT_PASSED, gate.EXIT_FAILED) and gate_exit == expected,
                f"gate_exit={gate_exit}，期望 {expected}",
            )

        check(
            "schema_ok",
            result.get("schema") == gate.GATE_SCHEMA,
            f"schema={result.get('schema')!r}",
        )
        check(
            "closed_normally",
            result.get("closed_normally") is True,
            f"closed_normally={result.get('closed_normally')!r}",
        )
        check(
            "reasons_is_list",
            isinstance(result.get("reasons"), list),
            f"reasons 类型={type(result.get('reasons')).__name__}",
        )

        if reuse == "passed":
            exposure = result.get("exposure_check") or {}
            check(
                "passed_exposure",
                exposure.get("passed") is True,
                f"exposure_check={exposure!r}",
            )
            devices = result.get("devices") or {}
            check(
                "passed_driveboard",
                bool(devices.get("drive_board_id")),
                f"drive_board_id={devices.get('drive_board_id')!r}",
            )
            check(
                "passed_criteria",
                bool(result.get("criteria")),
                f"criteria={result.get('criteria')!r}",
            )
            check(
                "passed_reference",
                isinstance(result.get("reference"), dict) and bool(result.get("reference")),
                f"reference={result.get('reference')!r}",
            )
        elif reuse == "failed":
            check(
                "failed_reasons",
                bool(result.get("reasons")),
                f"reasons={result.get('reasons')!r}",
            )
            check(
                "failed_remedy",
                bool(result.get("remedy")),
                f"remedy={result.get('remedy')!r}",
            )

    failed = [c for c in checks if not c["passed"]]
    reuse = result.get("reuse_gate") if isinstance(result, dict) else None
    return {
        "trial": "invalid" if failed else "valid",
        "checks": checks,
        "reasons": [c["detail"] for c in failed],
        "reuse_gate": reuse,
        "gate_exit": gate_exit,
    }


def _build_gate_args(args: argparse.Namespace) -> argparse.Namespace:
    """把试跑 CLI 参数透传成 ``run_reuse_gate`` 所需的 Namespace。"""
    return argparse.Namespace(
        out_root=args.out_root,
        run_dir=args.run_dir,
        slm_number=args.slm_number,
        wavelength=args.wavelength,
        cam_type=args.cam_type,
        cam_id=args.cam_id,
        wait_until_idle=args.wait_until_idle,
        peak_fraction=args.peak_fraction,
        saturation_fraction=args.saturation_fraction,
    )


def run_trial(args: argparse.Namespace) -> int:
    """跑一次实验系统复用门并按计划 §4 判读，写试判读 JSON，返回试跑退出码。"""
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    result: Any = None
    gate_exit: int | None = None
    error: str | None = None
    try:
        result = gate.run_reuse_gate(_build_gate_args(args))
        gate_exit = (
            gate.EXIT_PASSED if result.get("reuse_gate") == "passed" else gate.EXIT_FAILED
        )
    except Exception as exc:  # 门内部已 fail-closed；这里只捕获未预料异常
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("复用门执行异常")

    verdict = validate_trial(result, gate_exit)
    if error:
        verdict["reasons"].insert(0, f"复用门执行异常: {error}")

    doc: dict[str, Any] = {
        "schema": TRIAL_SCHEMA,
        "created_utc": _utc_now(),
        "trial": verdict["trial"],
        "gate": {
            "reuse_gate": verdict["reuse_gate"],
            "exit_code": verdict["gate_exit"],
            "result_file": str(run_dir / gate.GATE_FILE),
        },
        "checks": verdict["checks"],
        "reasons": verdict["reasons"],
        "remedy": (
            "复用门结果不可信：先修复 optical_path_reuse_gate.py / 本试跑，再重新试跑；"
            "未验证的复用门不得用于正式实验"
            if verdict["trial"] == "invalid"
            else "试跑有效：后续实验可正常启用复用门跳过完整阶梯"
        ),
        "run_dir": str(run_dir),
    }
    target = run_dir / TRIAL_FILE
    target.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"实验系统复用门试跑: trial={verdict['trial']} → {target}")
    return EXIT_TRIAL_VALID if verdict["trial"] == "valid" else EXIT_TRIAL_INVALID


def build_parser() -> argparse.ArgumentParser:
    """CLI 参数，默认值与 ``optical_path_reuse_gate.build_parser`` 一致。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", default="data/debug/optical_path_probe", help="探针记录根目录")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="本次试跑结果目录；缺省在 out-root 下新建时间戳目录",
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    parser.add_argument("--peak-fraction", type=float, default=gate.DEFAULT_PEAK_FRACTION)
    parser.add_argument(
        "--saturation-fraction", type=float, default=gate.DEFAULT_SATURATION_FRACTION
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_dir = str(Path(args.out_root) / f"reuse_gate_trial_{stamp}")
    return run_trial(args)


if __name__ == "__main__":
    sys.exit(main())
