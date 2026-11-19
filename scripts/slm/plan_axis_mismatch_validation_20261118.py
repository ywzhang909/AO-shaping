"""独立验证采集的预注册脚本：只读、不连设备、不启动采集。

把 production ``capture.probes`` 的精确相位计划枚举出来，按 ``index % 6`` 冻结
train / validation 划分，记录每个相位族的字节级 sha256 与打分统计，并在采集前
校验新会话输出目录为空、参考检查点 ``grid_size`` 与 ``--grid`` 一致。产出
``<session>/probe_plan.json``（schema ``axis_mismatch_validation/1``）以及可直接
复制的采集 / 拟合 / 三个变体回验命令序列。

设计约束（与采集计划文档一致）：

* 本脚本只读本地文件，**不 import 任何 ``ao_shaping.drivers``**、不打开 SLM/CCD/
  激光；``capture.probes`` 的 import 已验证不会拉入驱动模块。
* 划分在采集前写死；采集后若 ``manifest`` 的实际 ``split`` 与预注册不符，整批作废，
  不得事后改划分。
* 修正参数（``panel_to_ccd_deg`` / ``scale_ratio``）取自 2026-11-17 标定并冻结，
  本次评估不重新估计——这才是“把先前假设放到新数据上检验”的独立测试。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ao_shaping.tools.slm.transfer.capture import probes as production_probes

SCHEMA = "axis_mismatch_validation/1"
FAMILIES = ("zernike", "smooth", "bandlimited")
DEFAULT_CALIBRATION = Path(
    "data/debug/issue247_analysis_20261117/issue247_analysis.json"
)
DEFAULT_COMPARE_OUT_ROOT = Path("data/debug/deswivel_forward_inverse_20261118")
DEFAULT_REFERENCE_CHECKPOINT = Path(
    "data/debug/slm_transfer/20261117_202916/initial/frozen_parameters.json"
)
DEVICE_STATEMENT = (
    "本脚本只读：未导入 ao_shaping.drivers，未打开 SLM / CCD / 激光，未启动采集。"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_calibration(path: Path) -> dict[str, Any]:
    """读冻结的几何修正参数（与 ``compare_deswivel_forward_inverse`` 同源）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    alignment = data.get("image_alignment")
    if not isinstance(alignment, dict):
        raise ValueError(f"{path} 没有 image_alignment 块")
    if "panel_to_ccd_deg" not in alignment:
        raise ValueError(f"{path} 的 image_alignment 缺少 panel_to_ccd_deg")
    ratio = alignment.get("scale_ratio")
    if ratio is None:
        magnitude_x = float(alignment["panel_x_magnitude_px"])
        magnitude_y = float(alignment["panel_y_magnitude_px"])
        ratio = magnitude_y / magnitude_x
    return {
        "panel_to_ccd_deg": float(alignment["panel_to_ccd_deg"]),
        "scale_ratio": float(ratio),
        "orthogonality_error_deg": float(
            alignment.get("orthogonality_error_deg", float("nan"))
        ),
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def split_of(index: int) -> str:
    return "validation" if index % 6 >= 3 else "train"


def family_of(index: int) -> str:
    return FAMILIES[index % 3]


def enumerate_probes(
    grid: int, count: int, seed: int, periods: tuple[int, int]
) -> list[dict[str, Any]]:
    """把 production ``probes`` 的产出逐条落成可审阅的 JSON 记录。"""
    records: list[dict[str, Any]] = []
    probe_index = -1
    for order, (label, split, phase) in enumerate(
        production_probes(grid, count, seed, periods)
    ):
        phase = np.asarray(phase, dtype=np.float32)
        if phase.shape != (grid, grid):
            raise ValueError(f"probe {label} 形状 {phase.shape} != ({grid},{grid})")
        entry: dict[str, Any] = {
            "order": order,
            "label": label,
            "split": split,
            "shape": list(phase.shape),
            "dtype": str(phase.dtype),
            "sha256": _sha256_array(phase),
            "min": float(phase.min()),
            "max": float(phase.max()),
            "mean": float(phase.mean()),
            "std": float(phase.std()),
        }
        if label.startswith("probe_"):
            probe_index += 1
            entry["probe_index"] = probe_index
            entry["family"] = family_of(probe_index)
            # 双保险：production 的 split 必须与预注册规则一致。
            if split != split_of(probe_index):
                raise ValueError(
                    f"{label} production split={split} 与规则 split={split_of(probe_index)} 不一致"
                )
        records.append(entry)
    return records


def family_balance(probe_indices: list[int]) -> dict[str, int]:
    balance = {family: 0 for family in FAMILIES}
    for index in probe_indices:
        balance[family_of(index)] += 1
    return balance


def check_build_dirs(session_root: Path) -> dict[str, Any]:
    """capture / initial / fit 都用 ``exist_ok=False``，三者必须不存在。"""
    dirs = {name: session_root / name for name in ("capture", "initial", "fit")}
    existing = [name for name, path in dirs.items() if path.exists()]
    return {
        "build_dirs": {name: str(path) for name, path in dirs.items()},
        "existing_build_dirs": existing,
        "absent": not existing,
    }


def check_reference_grid(checkpoint: Path | None, grid: int) -> dict[str, Any]:
    """参考检查点存在时校验 grid_size，用于跨会话可比性软检查。"""
    if checkpoint is None or not checkpoint.is_file():
        return {
            "checkpoint_path": None if checkpoint is None else str(checkpoint),
            "checked": False,
            "reason": "参考检查点未提供或不存在；跳过 grid_size 软检查",
            "grid_size": None,
            "matches": None,
        }
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    grid_size = data.get("grid_size")
    return {
        "checkpoint_path": str(checkpoint),
        "checked": True,
        "grid_size": grid_size,
        "fft_size": data.get("fft_size"),
        "pixel_samples": data.get("pixel_samples"),
        "matches": (int(grid_size) == int(grid)) if grid_size is not None else None,
    }


def build_command_sequence(
    *,
    session_root: Path,
    grid: int,
    count: int,
    frames: int,
    seed: int,
    exposure_max_steps: int,
    compare_out_root: Path,
    calibration: Path,
    run_name: str,
) -> list[str]:
    s = session_root.as_posix()
    return [
        "# 1) 采集（占用 SLM #1 + Daheng CCD #0；结束自动 blank + 关设备）",
        f"python -m ao_shaping.tools.slm.transfer.capture --out {s}/capture --execute "
        f"--grid {grid} --count {count} --frames {frames} --seed {seed} "
        f"--exposure-max-steps {exposure_max_steps} --save-dataset",
        "# 2) 本会话冷启动检查点",
        f"python -m ao_shaping.tools.slm.transfer.bootstrap --dataset {s}/capture --out {s}/initial",
        "# 3) 前向拟合（产出 fit/inputs.npz + provenance.json；cuda 可换 CPU）",
        f"python -m ao_shaping.tools.slm.transfer.fit --dataset {s}/capture "
        f"--checkpoint {s}/initial/frozen_parameters.json --out {s}/fit "
        f"--iterations 200 --span 120 --adaptation full --device cuda",
        "# 4) 三个变体回验（离线；可在设备释放后进行）",
        f"python scripts/slm/compare_deswivel_forward_inverse_20261118.py "
        f"--fit-dir {s}/fit --calibration {calibration.as_posix()} "
        f"--out-root {compare_out_root.as_posix()} --run-name {run_name} "
        f"--device cuda --iterations 200 --patience 8 --min-delta 1e-4 --inverse-steps 60",
    ]


def plan_fingerprint(payload: dict[str, Any]) -> str:
    """对划分与相位计划做规范化哈希，用于防止采集后偷改预注册。"""
    canonical = {
        "grid": payload["grid"],
        "count": payload["count"],
        "frames": payload["frames"],
        "seed": payload["seed"],
        "ramp_periods": payload["ramp_periods"],
        "probes": [(p["label"], p["split"], p["sha256"]) for p in payload["probes"]],
        "train_indices": payload["probe_indices"]["train"],
        "validation_indices": payload["probe_indices"]["validation"],
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root", type=Path, default=Path("data/debug/slm_transfer")
    )
    parser.add_argument("--session", default="axis_mismatch_validation_20261118")
    parser.add_argument("--grid", type=int, default=320)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--exposure-max-steps", type=int, default=12)
    parser.add_argument("--ramp-periods", type=int, nargs=2, default=(64, 128))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="参考检查点，用于 grid_size 软检查；默认用归档会话 initial 检查点（若存在）",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="冻结的几何修正参数来源（issue247 JSON）",
    )
    parser.add_argument(
        "--compare-out-root",
        type=Path,
        default=DEFAULT_COMPARE_OUT_ROOT,
        help="三个变体回验(--run-name)的输出根，与采集 --out-root 相互独立",
    )
    parser.add_argument("--run-name", default="independent_20261118")
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖指纹不同的既有 probe_plan.json（采集前修正计划时才用）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.grid <= 0 or args.count <= 0 or args.frames <= 0:
        raise SystemExit("--grid / --count / --frames 必须为正")
    periods = (int(args.ramp_periods[0]), int(args.ramp_periods[1]))

    session_root = (args.out_root / args.session).resolve()
    checks = check_build_dirs(session_root)

    probes = enumerate_probes(args.grid, args.count, args.seed, periods)
    probe_indices = {
        "train": [p["probe_index"] for p in probes if p.get("split") == "train"],
        "validation": [
            p["probe_index"] for p in probes if p.get("split") == "validation"
        ],
    }
    balance = {
        split: family_balance(indices) for split, indices in probe_indices.items()
    }
    for split, counts in balance.items():
        missing = [family for family, n in counts.items() if n == 0]
        if missing:
            raise SystemExit(f"{split} split 缺少相位族 {missing}；实际 {counts}")

    reference_checkpoint = args.checkpoint
    if reference_checkpoint is None and DEFAULT_REFERENCE_CHECKPOINT.is_file():
        reference_checkpoint = DEFAULT_REFERENCE_CHECKPOINT
    grid_check = check_reference_grid(reference_checkpoint, args.grid)

    calibration: dict[str, Any] | None = None
    if args.calibration.is_file():
        calibration = load_calibration(args.calibration)
    else:
        logger.warning("冻结标定 {} 不存在；frozen_hypothesis 记为空", args.calibration)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": utc_now(),
        "device_statement": DEVICE_STATEMENT,
        "out_root": str(args.out_root.resolve()),
        "compare_out_root": str(args.compare_out_root.resolve()),
        "session": args.session,
        "session_root": str(session_root),
        "grid": int(args.grid),
        "count": int(args.count),
        "frames": int(args.frames),
        "seed": int(args.seed),
        "exposure_max_steps": int(args.exposure_max_steps),
        "ramp_periods": list(periods),
        "splits": {
            "calibration": [p["label"] for p in probes if p["split"] == "calibration"],
            "train": [p["label"] for p in probes if p["split"] == "train"],
            "validation": [p["label"] for p in probes if p["split"] == "validation"],
        },
        "probe_indices": probe_indices,
        "family_by_probe_index": [family_of(i) for i in range(args.count)],
        "family_balance": balance,
        "frozen_hypothesis": calibration,
        "checks": {**checks, "reference_grid": grid_check},
        "probes": probes,
        "command_sequence": build_command_sequence(
            session_root=session_root,
            grid=args.grid,
            count=args.count,
            frames=args.frames,
            seed=args.seed,
            exposure_max_steps=args.exposure_max_steps,
            compare_out_root=args.compare_out_root.resolve(),
            calibration=args.calibration.resolve(),
            run_name=args.run_name,
        ),
    }
    payload["plan_fingerprint"] = plan_fingerprint(payload)

    plan_path = session_root / "probe_plan.json"
    if plan_path.exists() and not args.force:
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_fingerprint") != payload["plan_fingerprint"]:
            raise SystemExit(
                f"{plan_path} 已存在且指纹不同；预注册不能静默改写。"
                "确认是采集前的计划修正后加 --force。"
            )

    if not checks["absent"]:
        raise SystemExit(
            f"会话目录已有构建产物 {checks['existing_build_dirs']}；"
            "capture/bootstrap/fit 都要求全新目录，请换一个空会话名。"
        )

    session_root.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("{}", DEVICE_STATEMENT)
    logger.info(
        "probes total={} train={} validation={} calibration={}",
        len(probes),
        len(probe_indices["train"]),
        len(probe_indices["validation"]),
        len(payload["splits"]["calibration"]),
    )
    logger.info("train indices  = {}", probe_indices["train"])
    logger.info("validation idx = {}", probe_indices["validation"])
    logger.info("family balance = {}", json.dumps(balance, ensure_ascii=False))
    if grid_check["checked"]:
        logger.info(
            "reference checkpoint grid_size={} matches --grid {}: {}",
            grid_check["grid_size"],
            args.grid,
            grid_check["matches"],
        )
    else:
        logger.info("reference checkpoint: {}", grid_check["reason"])
    if calibration is not None:
        logger.info(
            "frozen hypothesis: panel_to_ccd_deg={} scale_ratio={}",
            calibration["panel_to_ccd_deg"],
            calibration["scale_ratio"],
        )
    logger.info("wrote {}", plan_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
