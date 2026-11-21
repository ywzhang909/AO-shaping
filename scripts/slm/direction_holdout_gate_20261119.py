"""P3 上游资格方向门：任意个 pass 的 ±Z5..Z11 方向一致性回放。

用途（2026-11-19）：把 11/18 的一次性脚本
``data/debug/offline_refit_20261118_freshep/eval_direction_gate_2pass_20261118.py``
改成可复用的参数化工具，口径逐字保留：

- 每对探针先按 ``raw_frames.mean / exposure_ms`` 归一为速率，再裁 ``CROP=128``；
- 每个 pass 各自用 ``zero_order_center_by_method(method="ellipse")` 估计椭圆中心
  （epoch-correct），模型 ``origin`` 设到该 pass 的中心；
- ``FrozenDefocusForward`` / ``FrozenDefocusParameters`` 从 checkpoint 构造，
  ``phase_sign=-1.0``、``phase_sign_scope="command"``、``sigma_ccd_px=0.0``；
- Pearson 取自 ``jacobian_guard.direction_agreement``；
- grid 取该 run ``model_phase`` 的边长。

判据：一对通过当且仅当 Pearson >= ``--min-pearson``；``pooled`` 通过数
>= ``--min-pairs`` 且所有需要的行都可读 ⇒ ``pass``；否则 ``fail``；
有 run 缺某个模式的 ± 对、npz 缺失或 ``exposure_ms`` 缺失 ⇒ ``not_assessable``。

离线：只读归档 npz，不打开 SLM/CCD/激光，不碰 DeviceLock。
退出码：0 = pass，1 = fail，2 = not_assessable。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
)
from ao_shaping.tools.slm.transfer.jacobian_guard import direction_agreement
from ao_shaping.utils.image.zero_order import zero_order_center_by_method

CROP = 128
MODES: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)
# 标签形如 holdout_000_Z5_+0.05 / ztrain_003_Z7_-0.05：只认末尾的符号+幅度。
_AMP_RE = re.compile(r"_Z\d+_([+-])(\d+\.\d+)$")


def sha256_file(path: Path) -> str:
    """返回文件内容的 SHA256（不存在时返回空串）。"""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def pair_rows(manifest: list[dict], mode: int) -> tuple[dict, dict, str] | None:
    """在清单里找某个模式的 ± 行，返回 (负向行, 正向行, 幅度字符串)；缺一为 None。"""
    plus = [r for r in manifest if re.search(rf"_Z{mode}_\+", str(r.get("label", "")))]
    minus = [r for r in manifest if re.search(rf"_Z{mode}_-", str(r.get("label", "")))]
    if not plus or not minus:
        return None
    match = _AMP_RE.search(str(plus[0]["label"]))
    return minus[0], plus[0], match.group(2) if match else "?"


def load_checkpoint(path: Path) -> tuple[dict, list[list[float]], str]:
    """读取 checkpoint，返回 (model 字典, ramp_matrix, 变体名)。

    接受两种布局：

    - 混合包：``{"model": {...}, "ramp_matrix": ...}``（ramp 也可嵌在 model 里）；
    - 裸 ``frozen_parameters.json``：顶层直接是模型字段 + ``ramp_matrix``。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("model"), dict):
        model_src = data["model"]
        ramp = data.get("ramp_matrix") or model_src.get("ramp_matrix")
        variant = "hybrid_bundle"
    else:
        model_src = data
        ramp = data.get("ramp_matrix")
        variant = "frozen_parameters"
    if ramp is None:
        raise SystemExit(f"checkpoint 缺少 ramp_matrix: {path}")
    model = {
        "defocus_diopter": model_src["defocus_diopter"],
        "amplitude_center": model_src["amplitude_center"],
        "amplitude_width": model_src["amplitude_width"],
        "gain": model_src["gain"],
        "background": model_src["background"],
        "residual_noll_radians": model_src.get("residual_noll_radians", {}),
    }
    return model, ramp, variant


def build_forward(
    model: dict, ramp: list[list[float]], centre_xy: tuple[float, float], grid: int
) -> FrozenDefocusForward:
    """按归档脚本的同一组参数构造前向模型（origin 用本 pass 的实测中心）。"""
    params = FrozenDefocusParameters(
        defocus_diopter=float(model["defocus_diopter"]),
        origin_xy=(float(centre_xy[0]), float(centre_xy[1])),
        amplitude_center_xy=tuple(model["amplitude_center"]),
        amplitude_width_xy=tuple(model["amplitude_width"]),
        residual_noll_radians={
            int(mode): float(value)
            for mode, value in model.get("residual_noll_radians", {}).items()
        },
        gain=float(model["gain"]),
        background=float(model["background"]),
        sigma_ccd_px=0.0,
        ramp_matrix=((ramp[0][0], ramp[0][1]), (ramp[1][0], ramp[1][1])),
        phase_sign=-1.0,
        phase_sign_scope="command",
        grid_size=grid,
    )
    yy, xx = np.mgrid[:CROP, :CROP].astype(np.float64)
    camera_xy = np.stack(
        [xx + centre_xy[0] - CROP / 2, yy + centre_xy[1] - CROP / 2], axis=-1
    )
    return FrozenDefocusForward(params, camera_xy, device="cpu")


def crop(img: np.ndarray, centre_xy: tuple[float, float]) -> np.ndarray:
    """以椭圆中心为基准裁出 ``CROP×CROP``。"""
    x0 = int(round(centre_xy[0] - CROP / 2))
    y0 = int(round(centre_xy[1] - CROP / 2))
    return img[y0 : y0 + CROP, x0 : x0 + CROP]


def measured_centre(dataset: Path, manifest: list[dict]) -> tuple[float, float]:
    """该 run 的椭圆零级中心（对清单全部行取均值后估椭圆，epoch-correct）。"""
    means = [
        np.load(dataset / str(row["file"]))["raw_frames"].astype(np.float64).mean(axis=0)
        for row in manifest
    ]
    cx, cy = zero_order_center_by_method(np.mean(means, axis=0), method="ellipse")
    return float(cx), float(cy)


def run_grid(dataset: Path, manifest: list[dict]) -> int | None:
    """取该 run 的 grid：第一条带 ``model_phase`` 的 npz 边长。"""
    for row in manifest:
        path = dataset / str(row["file"])
        if not path.exists():
            continue
        try:
            with np.load(path) as data:
                if "model_phase" in data:
                    return int(data["model_phase"].shape[0])
        except OSError:
            continue
    return None


def decide(pairs: list[dict], n_expected: int, min_pairs: int, unreadable: list[str]) -> str:
    """判决：有任何不可读行 ⇒ not_assessable；否则按 pooled 通过数比阈值。"""
    if unreadable:
        return "not_assessable"
    if len(pairs) != n_expected:
        return "not_assessable"
    passed = sum(1 for p in pairs if p.get("passed") is True)
    return "pass" if passed >= min_pairs else "fail"


def _pair_stub(mode: int, pass_name: str, reason: str) -> dict:
    """构造一个不可评估的 pair 记录（pearson=None，passed=None）。"""
    return {
        "mode": int(mode),
        "amp": None,
        "pass": pass_name,
        "pearson": None,
        "passed": None,
        "reason": reason,
    }


def evaluate_run(
    dataset: Path,
    pass_name: str,
    model: dict,
    ramp: list[list[float]],
    min_pearson: float,
) -> tuple[dict, list[dict]]:
    """评估一个 pass，返回 (run 记录, 该 pass 的全部 pair 记录)。"""
    pairs: list[dict] = []
    unreadable: list[str] = []
    record: dict = {
        "path": str(dataset),
        "pass": pass_name,
        "manifest_sha256": "",
        "grid": None,
        "centre": None,
        "per_pass_passed": None,
        "n_modes": len(MODES),
        "unreadable": unreadable,
    }

    manifest_path = dataset / "manifest.json"
    record["manifest_sha256"] = sha256_file(manifest_path)
    if not manifest_path.exists():
        unreadable.append(f"manifest 缺失: {manifest_path}")
        pairs = [_pair_stub(m, pass_name, "manifest 缺失") for m in MODES]
        return record, pairs
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        unreadable.append(f"manifest 不可读: {type(exc).__name__}: {exc}")
        pairs = [_pair_stub(m, pass_name, "manifest 不可读") for m in MODES]
        return record, pairs

    grid = run_grid(dataset, manifest)
    record["grid"] = grid
    if grid is None:
        unreadable.append("没有任何 npz 带 model_phase")

    # 先校验每个模式的行是否齐全、npz 是否在、exposure_ms 是否存在。
    checked: dict[int, tuple[dict, dict, str]] = {}
    for mode in MODES:
        pair = pair_rows(manifest, mode)
        if pair is None:
            unreadable.append(f"Z{mode} 缺 ± 对")
            pairs.append(_pair_stub(mode, pass_name, "缺 ± 对"))
            continue
        rminus, rplus, amp = pair
        problems: list[str] = []
        for row in (rminus, rplus):
            npz_path = dataset / str(row["file"])
            if not npz_path.exists():
                problems.append(f"npz 缺失: {npz_path.name}")
            exposure = row.get("exposure_ms")
            if exposure is None or not math.isfinite(float(exposure)) or float(exposure) <= 0:
                problems.append(f"exposure_ms 缺失: {row['file']}")
        if problems:
            unreadable.extend(f"Z{mode}: {p}" for p in problems)
            pairs.append(_pair_stub(mode, pass_name, "; ".join(problems)))
            continue
        checked[mode] = (rminus, rplus, amp)

    if unreadable or grid is None:
        # 有不可读行时不估中心/不跑前向：判决必然是 not_assessable。
        pairs.extend(
            _pair_stub(mode, pass_name, "同 run 存在不可读行")
            for mode in MODES
            if mode not in checked and not any(p["mode"] == mode for p in pairs)
        )
        return record, _sorted_pairs(pairs)

    try:
        centre = measured_centre(dataset, manifest)
    except (OSError, ValueError) as exc:
        unreadable.append(f"椭圆中心估计失败: {type(exc).__name__}: {exc}")
        pairs = [_pair_stub(m, pass_name, "椭圆中心估计失败") for m in MODES]
        return record, pairs
    record["centre"] = [float(centre[0]), float(centre[1])]

    fwd = build_forward(model, ramp, centre, int(grid))
    for mode in MODES:
        rminus, rplus, amp = checked[mode]
        try:
            dm = np.load(dataset / str(rminus["file"]))
            dp = np.load(dataset / str(rplus["file"]))
            phase_m = dm["model_phase"].astype(np.float32)
            phase_p = dp["model_phase"].astype(np.float32)
            meas_m = crop(
                dm["raw_frames"].astype(np.float64).mean(axis=0)
                / float(rminus["exposure_ms"]),
                centre,
            )
            meas_p = crop(
                dp["raw_frames"].astype(np.float64).mean(axis=0)
                / float(rplus["exposure_ms"]),
                centre,
            )
            with torch.no_grad():
                pred_m = fwd.predict(torch.as_tensor(phase_m)).cpu().numpy()[0]
                pred_p = fwd.predict(torch.as_tensor(phase_p)).cpu().numpy()[0]
        except (OSError, KeyError, ValueError) as exc:
            unreadable.append(f"Z{mode}: 读取/前向失败 {type(exc).__name__}: {exc}")
            pairs.append(_pair_stub(mode, pass_name, "读取/前向失败"))
            continue
        agreement = direction_agreement(pred_m, pred_p, meas_m, meas_p)
        pearson = float(agreement.correlation)
        passed = bool(math.isfinite(pearson) and pearson >= min_pearson)
        reason = (
            f"pearson {pearson:.4f} {'>=' if passed else '<'} {min_pearson:.2f}"
            if math.isfinite(pearson)
            else "pearson 非有限（预测或实测无对比度）"
        )
        pairs.append(
            {
                "mode": int(mode),
                "amp": float(amp),
                "pass": pass_name,
                "pearson": pearson if math.isfinite(pearson) else None,
                "passed": passed,
                "reason": reason,
            }
        )

    record["per_pass_passed"] = sum(1 for p in pairs if p.get("passed") is True)
    return record, _sorted_pairs(pairs)


def _sorted_pairs(pairs: list[dict]) -> list[dict]:
    """按模式号排序，保证证据文件稳定可比。"""
    return sorted(pairs, key=lambda p: int(p["mode"]))


def summarise(runs: list[dict], all_pairs: list[dict], min_pairs: int) -> dict:
    """汇总 per_pass / pooled 计数与判决。"""
    per_pass = {
        run["pass"]: {
            "passed": run["per_pass_passed"],
            "n_modes": run["n_modes"],
            "unreadable": run["unreadable"],
        }
        for run in runs
    }
    pooled = {
        "n_runs": len(runs),
        "n_pairs": len(all_pairs),
        "expected": len(MODES) * len(runs),
        "passed": sum(1 for p in all_pairs if p.get("passed") is True),
        "min_pairs": int(min_pairs),
    }
    unreadable = [reason for run in runs for reason in run["unreadable"]]
    verdict = decide(all_pairs, pooled["expected"], min_pairs, unreadable)
    return {"per_pass": per_pass, "pooled": pooled, "verdict": verdict}


def main() -> None:
    """CLI 入口：逐个 run 回放并写证据 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        metavar="PATH",
        help="一个或多个数据集目录（各含 manifest.json + sample_*.npz），每个 = 一个独立 pass。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="混合包 JSON（含 model/ramp_matrix）或裸 frozen_parameters.json。",
    )
    parser.add_argument("--out", type=Path, required=True, help="证据 JSON 输出路径。")
    parser.add_argument(
        "--center-method",
        default="ellipse",
        choices=["ellipse"],
        help="零级中心估计方法（冻结为 ellipse）。",
    )
    parser.add_argument("--min-pearson", type=float, default=0.50)
    parser.add_argument("--min-pairs", type=int, default=10)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint 不存在: {args.checkpoint}")
    if args.min_pairs < 0:
        raise SystemExit("--min-pairs 必须非负")

    model, ramp, variant = load_checkpoint(args.checkpoint)

    runs: list[dict] = []
    all_pairs: list[dict] = []
    for index, dataset in enumerate(args.runs, start=1):
        pass_name = f"pass{index}"
        run, pairs = evaluate_run(
            dataset, pass_name, model, ramp, float(args.min_pearson)
        )
        runs.append(run)
        all_pairs.extend(pairs)

    summary = summarise(runs, all_pairs, int(args.min_pairs))
    exit_code = {"pass": 0, "fail": 1, "not_assessable": 2}[summary["verdict"]]

    evidence = {
        "schema": "direction_holdout_gate_20261119/1",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_variant": variant,
        "crop": CROP,
        "modes": [int(m) for m in MODES],
        "min_pearson": float(args.min_pearson),
        "min_pairs": int(args.min_pairs),
        "center_method": args.center_method,
        "center_source": "measured per pass (all rows)",
        "convention": {
            "metric": "jacobian_guard.direction_agreement Pearson",
            "phase_sign": -1.0,
            "phase_sign_scope": "command",
            "rate": "raw_frames mean / exposure_ms",
        },
        "runs": runs,
        "per_pass": summary["per_pass"],
        "pooled": summary["pooled"],
        "pairs": all_pairs,
        "verdict": summary["verdict"],
        "exit_code": exit_code,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    for run in runs:
        passed = run["per_pass_passed"]
        passed_txt = "?" if passed is None else str(passed)
        print(f"{run['pass']}: {passed_txt}/{run['n_modes']}  {run['path']}")
        for reason in run["unreadable"]:
            print(f"  UNREADABLE: {reason}")
    pooled = summary["pooled"]
    print(
        f"pooled: {pooled['passed']}/{pooled['n_pairs']} "
        f"(need {pooled['min_pairs']}, expected {pooled['expected']})"
    )
    print(f"verdict: {summary['verdict']}  -> {args.out}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
