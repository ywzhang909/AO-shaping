# -*- coding: utf-8 -*-
"""S1f 随机抖动归因探针的离线判读（预注册 §3.4）。

本模块只**执行** `docs/dev/s1f_jitter_attribution_prereg_20261127.md` §3.4 的
度量与归因口径，不重新定义阈值、不改 S1e 冻结判据、不改变 S1 状态：

- 三层分辨：变体 A（同一次写屏后连采 24 帧，地板）/ 变体 B（同字节重写 24 次）/
  变体 C（flat 头-中-尾三组各 16 帧，漂移）。
- 亚像素质心与主瓣选取直接复用冻结求解器
  `scripts/slm/issue247_axis_map_solve_20261121.py` 的 ``find_spots`` /
  ``principal``；注意 ``principal`` 返回的是相对 ``ref`` 的 ``dr/dc``，
  绝对坐标须显式加回参考。
- 全部离线：只读 ``frames/*.npy`` 与 ``capture_journal.jsonl``，不打开设备、
  不取锁。

CLI::

    python scripts/slm/s1f_jitter_attribution_judge_20261127.py \
        --capture-dir data/experiments/.../run_01

判读失败（缺帧、帧形状异常、找不到 0 级/主瓣）一律抛 ``JudgmentError``，
``main`` 返回退出码 2，不静默降级。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src", ROOT / "libs", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.slm.issue247_axis_map_solve_20261121 import find_spots, principal
from scripts.slm.s1d_orientation_probe_capture_20261124 import utc_now, write_json

JUDGMENT_SCHEMA = "issue247-s1f-jitter-attribution-judgment/v1"
DEFAULT_OUT_DIR = ROOT / "report/shape_trials/s1f_jitter_attribution_20261127"
DEFAULT_JUDGMENT_NAME = "s1f_jitter_attribution_judgment"

#: §3.2 采集顺序：变体 A/B 各 24 个 pattern_id，flat 头/中/尾各一组。
ARM_PATTERNS: dict[str, tuple[str, ...]] = {
    "A": tuple(f"s1f_a{index:02d}" for index in range(1, 25)),
    "B": tuple(f"s1f_b{index:02d}" for index in range(1, 25)),
}
FLAT_PATTERNS: tuple[str, ...] = ("flat_head", "flat_mid", "flat_tail")
#: 变体 -> 所用 0 级参考（§3.4：A 用 zr_head，B 用 zr_mid）。
ARM_REF_KEY: dict[str, str] = {"A": "zr_head", "B": "zr_mid"}
FLAT_REF_KEY: dict[str, str] = {
    "flat_head": "zr_head",
    "flat_mid": "zr_mid",
    "flat_tail": "zr_tail",
}

#: ``principal`` 冻结语义（core 排除 + 距 0 级最小距离），与 S1e 判读一致。
EXCLUDE_CORE_PX = 6.0
MIN_DIST_PX = 15.0
#: §3.4 归因阈值。
RATIO_V1_MAX = 1.3
RATIO_V2_MIN = 2.0
DRIFT_MATERIAL_PX = 1.0
#: §3.4 交叉核对基准：S1e `run_05_postfix` 同写屏两帧质心差（仅佐证）。
RUN_05_INTRA_UPLOAD_REF_PX: tuple[float, float] = (1.68, 1.85)

VERDICT_V1 = "V1"
VERDICT_V2 = "V2"
VERDICT_V3 = "V3"
#: A.rms_px <= 0 时比值无定义（§3.4 记 None，不写 inf；JSON 不允许 Infinity）。
VERDICT_UNDETERMINED = "undetermined"
VERDICTS: tuple[str, ...] = (VERDICT_V1, VERDICT_V2, VERDICT_V3, VERDICT_UNDETERMINED)
VERDICT_LABELS: dict[str, str] = {
    VERDICT_V1: "读取/提取主导 (readout/extraction dominant)",
    VERDICT_V2: "写屏随机性主导 (write-screen dominant)",
    VERDICT_V3: "混合 (mixed)",
    VERDICT_UNDETERMINED: "无法裁定 (A.rms_px<=0，地板为 0)",
}


class JudgmentError(RuntimeError):
    """判读无法完成：缺帧、形状异常、找不到 0 级或主瓣。"""


# ---------------------------------------------------------------------------
# 采集目录读取
# ---------------------------------------------------------------------------
def read_journal(capture_dir: Path) -> list[dict[str, Any]]:
    """解析 ``capture_journal.jsonl``；每个非空行一个 JSON 对象。"""
    path = capture_dir / "capture_journal.jsonl"
    if not path.is_file():
        raise JudgmentError(f"missing journal: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discover_frames(
    capture_dir: Path, journal: list[dict[str, Any]]
) -> dict[str, list[Path]]:
    """``pattern_id`` -> 帧文件（优先 journal 的 ``frame_saved``，缺则按文件名回退）。"""
    frames_dir = capture_dir / "frames"
    if not frames_dir.is_dir():
        raise JudgmentError(f"missing frames directory: {frames_dir}")
    mapping: dict[str, list[Path]] = {}
    for row in journal:
        if row.get("event") != "frame_saved":
            continue
        path = frames_dir / str(row["filename"])
        if not path.is_file():
            raise JudgmentError(f"journal lists a missing frame: {path}")
        mapping.setdefault(str(row["pattern_id"]), []).append(path)
    expected = (*FLAT_PATTERNS, *ARM_PATTERNS["A"], *ARM_PATTERNS["B"])
    for pattern_id in expected:
        if pattern_id not in mapping:
            mapping[pattern_id] = sorted(frames_dir.glob(f"*_{pattern_id}.npy"))
            if not mapping[pattern_id]:
                raise JudgmentError(
                    f"no frames for pattern_id {pattern_id!r} in {frames_dir}"
                )
        mapping[pattern_id].sort(key=lambda frame: frame.name)
    return mapping


def load_frame(path: Path) -> np.ndarray:
    """读取单帧并校验为二维；形状异常直接报错。"""
    frame = np.load(path)
    if frame.ndim != 2:
        raise JudgmentError(f"frame {path.name} is not 2-D: shape={frame.shape}")
    return frame.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# 0 级参考与主瓣
# ---------------------------------------------------------------------------
def zero_order_reference(frame: np.ndarray) -> tuple[float, float]:
    """flat 帧的 0 级亚像素质心 ``(row, col)``。

    ``find_spots`` 每项已是对局部 17x17 窗口做强度加权得到的亚像素质心，
    按 ``peak`` 降序返回；取最亮的一个即为 0 级。
    """
    spots = find_spots(frame, min_dist=20)
    if not spots:
        raise JudgmentError("no 0th-order spot on the flat reference frame")
    best = max(spots, key=lambda spot: spot["peak"])
    return float(best["row"]), float(best["col"])


def group_zero_order_reference(
    frames: list[np.ndarray], label: str
) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """一组 flat 帧的 0 级参考 = 逐帧位置的逐轴中位（抗离群）。"""
    if not frames:
        raise JudgmentError(f"empty flat group: {label}")
    positions = [zero_order_reference(frame) for frame in frames]
    ref = (
        float(np.median([row for row, _ in positions])),
        float(np.median([col for _, col in positions])),
    )
    return ref, positions


def principal_position(
    frame: np.ndarray, ref: tuple[float, float]
) -> dict[str, float] | None:
    """主瓣绝对位置；``principal`` 只给相对 ``ref`` 的 ``dr/dc``，此处加回参考。"""
    spot = principal(frame, ref, exclude_core=EXCLUDE_CORE_PX, min_dist=MIN_DIST_PX)
    if spot is None:
        return None
    return {
        "row": float(ref[0] + spot["dr"]),
        "col": float(ref[1] + spot["dc"]),
        "dr": float(spot["dr"]),
        "dc": float(spot["dc"]),
        "peak": float(spot["peak"]),
        "dist": float(spot["dist"]),
    }


# ---------------------------------------------------------------------------
# 散布与漂移
# ---------------------------------------------------------------------------
def pairwise_max_px(points: list[tuple[float, float]]) -> float:
    """所有帧间位置两两 ``hypot`` 的最大值（<2 点时记 0）。"""
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    rows = arr[:, 0]
    cols = arr[:, 1]
    dr = rows[:, None] - rows[None, :]
    dc = cols[:, None] - cols[None, :]
    return float(np.hypot(dr, dc).max())


def rms_px(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """``(rms_px, std_row, std_col)``；总体标准差（``ddof=0``）。"""
    if not points:
        raise JudgmentError("cannot compute spread of an empty position list")
    arr = np.asarray(points, dtype=np.float64)
    std_row = float(arr[:, 0].std())
    std_col = float(arr[:, 1].std())
    return float(np.hypot(std_row, std_col)), std_row, std_col


def spread_metrics(points: list[tuple[float, float]]) -> dict[str, float]:
    rms, std_row, std_col = rms_px(points)
    return {
        "pairwise_max_px": pairwise_max_px(points),
        "rms_px": rms,
        "std_row": std_row,
        "std_col": std_col,
    }


def drift_metrics(
    zr_head: tuple[float, float], zr_tail: tuple[float, float]
) -> dict[str, Any]:
    """头-尾 0 级漂移（px），逐轴差 + 是否达到 1.0 px 物质阈值。"""
    dr = float(zr_tail[0] - zr_head[0])
    dc = float(zr_tail[1] - zr_head[1])
    head_to_tail = float(np.hypot(dr, dc))
    return {
        "head_to_tail_px": head_to_tail,
        "dr": dr,
        "dc": dc,
        "drift_material": head_to_tail >= DRIFT_MATERIAL_PX,
    }


# ---------------------------------------------------------------------------
# 归因
# ---------------------------------------------------------------------------
def classify(b_rms: float, a_rms: float) -> dict[str, Any]:
    """§3.4 裁定：ratio<=1.3 -> V1；>=2.0 -> V2；否则 V3。"""
    if a_rms <= 0.0:
        return {
            "ratio": None,
            "verdict": VERDICT_UNDETERMINED,
            "verdict_label": VERDICT_LABELS[VERDICT_UNDETERMINED],
            "ratio_note": "A.rms_px<=0: readout floor is zero, ratio undefined",
        }
    ratio = float(b_rms) / float(a_rms)
    if ratio <= RATIO_V1_MAX:
        verdict = VERDICT_V1
    elif ratio >= RATIO_V2_MIN:
        verdict = VERDICT_V2
    else:
        verdict = VERDICT_V3
    return {
        "ratio": ratio,
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[verdict],
        "ratio_note": None,
    }


def attribute(
    arms: dict[str, dict[str, Any]], drift: dict[str, Any]
) -> dict[str, Any]:
    """两分量（读取地板 / 写屏随机性）加性占比 + 漂移。"""
    a_rms = float(arms["A"]["rms_px"])
    b_rms = float(arms["B"]["rms_px"])
    readout_floor = a_rms
    write_screen = max(0.0, b_rms - a_rms)
    total = readout_floor + write_screen
    verdict = classify(b_rms, a_rms)
    return {
        "readout_floor_rms": readout_floor,
        "write_screen_rms": write_screen,
        "drift_px": float(drift["head_to_tail_px"]),
        "ratio": verdict["ratio"],
        "ratio_note": verdict["ratio_note"],
        "verdict": verdict["verdict"],
        "verdict_label": verdict["verdict_label"],
        "readout_share": (readout_floor / total) if total > 0 else None,
        "write_screen_share": (write_screen / total) if total > 0 else None,
    }


# ---------------------------------------------------------------------------
# 判读装配
# ---------------------------------------------------------------------------
def judge(capture_dir: Path) -> dict[str, Any]:
    """执行 §3.4 全部度量并返回判读 payload。"""
    capture_dir = Path(capture_dir)
    journal = read_journal(capture_dir)
    frames = discover_frames(capture_dir, journal)

    shapes: set[tuple[int, ...]] = set()

    def check_shape(shape: tuple[int, ...]) -> None:
        shapes.add(shape)
        if len(shapes) > 1:
            raise JudgmentError(
                f"inconsistent frame shapes across capture: {sorted(shapes)}"
            )

    zero_order_refs: dict[str, dict[str, Any]] = {}
    flat_track: dict[str, dict[str, Any]] = {}
    flat_refs: dict[str, tuple[float, float]] = {}
    flat_counts: dict[str, int] = {}

    for pattern_id in FLAT_PATTERNS:
        loaded = [load_frame(path) for path in frames[pattern_id]]
        for frame in loaded:
            check_shape(frame.shape)
        ref, positions = group_zero_order_reference(loaded, pattern_id)
        key = FLAT_REF_KEY[pattern_id]
        flat_refs[key] = ref
        flat_counts[pattern_id] = len(loaded)
        zero_order_refs[key] = {
            "pattern_id": pattern_id,
            "n_frames": len(loaded),
            "row": round(ref[0], 3),
            "col": round(ref[1], 3),
            "method": "per-frame find_spots(min_dist=20) max-peak centroid, per-axis median",
        }
        flat_track[pattern_id] = {
            "ref_key": key,
            "positions": [
                {"row": round(row, 3), "col": round(col, 3)} for row, col in positions
            ],
        }

    arms: dict[str, dict[str, Any]] = {}
    for arm, pattern_ids in ARM_PATTERNS.items():
        ref_key = ARM_REF_KEY[arm]
        ref = flat_refs[ref_key]
        points: list[tuple[float, float]] = []
        positions: dict[str, dict[str, Any]] = {}
        for pattern_id in pattern_ids:
            for path in frames[pattern_id]:
                frame = load_frame(path)
                check_shape(frame.shape)
                spot = principal_position(frame, ref)
                if spot is None:
                    raise JudgmentError(
                        f"{path.name}: no principal lobe clear of the 0th-order "
                        f"(ref {ref_key}={ref})"
                    )
                points.append((spot["row"], spot["col"]))
                positions[path.stem] = {
                    "pattern_id": pattern_id,
                    "row": round(spot["row"], 3),
                    "col": round(spot["col"], 3),
                    "dr": round(spot["dr"], 3),
                    "dc": round(spot["dc"], 3),
                    "peak": round(spot["peak"], 1),
                }
        metrics = spread_metrics(points)
        arms[arm] = {
            "ref_key": ref_key,
            "ref": {"row": round(ref[0], 3), "col": round(ref[1], 3)},
            "n_frames": len(points),
            "positions": positions,
            "pairwise_max_px": metrics["pairwise_max_px"],
            "rms_px": metrics["rms_px"],
            "std_row": metrics["std_row"],
            "std_col": metrics["std_col"],
        }

    if not shapes:
        raise JudgmentError("no frames loaded from capture")

    drift = drift_metrics(flat_refs["zr_head"], flat_refs["zr_tail"])
    attribution = attribute(arms, drift)
    arm_counts = {
        "A": arms["A"]["n_frames"],
        "B": arms["B"]["n_frames"],
        **flat_counts,
    }

    return {
        "schema": JUDGMENT_SCHEMA,
        "capture_dir": (
            capture_dir.relative_to(ROOT).as_posix()
            if capture_dir.is_relative_to(ROOT)
            else str(capture_dir)
        ),
        "generated_at_utc": utc_now(),
        "n_frames": sum(arm_counts.values()),
        "arm_counts": arm_counts,
        "frame_shape": list(next(iter(shapes))),
        "zero_order_refs": zero_order_refs,
        "arms": arms,
        "drift": drift,
        "attribution": attribution,
        "crosscheck": {
            "run_05_intra_upload_ref_px": list(RUN_05_INTRA_UPLOAD_REF_PX),
            "A_rms_px": arms["A"]["rms_px"],
            "B_rms_px": arms["B"]["rms_px"],
            "note": "evidence only: compare with S1e run_05_postfix intra_upload_dist 1.68-1.85 px",
        },
        "flat_track": flat_track,
        "state_effect": "none: this probe does not change S1 status or the S1e D1 rule",
    }


# ---------------------------------------------------------------------------
# 人类可读表格
# ---------------------------------------------------------------------------
def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# S1f 随机抖动归因判读 ({JUDGMENT_SCHEMA})")
    lines.append("")
    lines.append("## 采集")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| capture_dir | `{payload['capture_dir']}` |")
    lines.append(f"| generated_at_utc | {payload['generated_at_utc']} |")
    lines.append(f"| 帧数合计 | {payload['n_frames']} |")
    lines.append(f"| 帧形状 | {tuple(payload['frame_shape'])} |")
    counts = payload["arm_counts"]
    lines.append(
        "| 各段帧数 | "
        f"A={counts['A']}, B={counts['B']}, "
        f"flat_head={counts['flat_head']}, flat_mid={counts['flat_mid']}, "
        f"flat_tail={counts['flat_tail']} |"
    )
    lines.append("")
    lines.append("## 0 级参考（flat 组逐帧 max-peak 质心，逐轴中位）")
    lines.append("")
    lines.append("| 参考 | 来源 | 帧数 | row (px) | col (px) |")
    lines.append("|---|---|---|---|---|")
    for key, entry in payload["zero_order_refs"].items():
        lines.append(
            f"| {key} | {entry['pattern_id']} | {entry['n_frames']} | "
            f"{entry['row']:.3f} | {entry['col']:.3f} |"
        )
    lines.append("")
    lines.append("## 抖动变体")
    lines.append("")
    lines.append(
        "| 变体 | 含义 | 参考 | 帧数 | rms_px | pairwise_max_px | std_row | std_col |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    meaning = {"A": "同写屏多帧（读取/提取地板）", "B": "同字节重写（写屏随机性）"}
    for arm in ("A", "B"):
        rec = payload["arms"][arm]
        lines.append(
            f"| {arm} | {meaning[arm]} | {rec['ref_key']} | {rec['n_frames']} | "
            f"{rec['rms_px']:.3f} | {rec['pairwise_max_px']:.3f} | "
            f"{rec['std_row']:.3f} | {rec['std_col']:.3f} |"
        )
    lines.append("")
    lines.append("## 漂移（flat 头 -> 尾）")
    lines.append("")
    drift = payload["drift"]
    lines.append("| 量 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| head_to_tail_px | {drift['head_to_tail_px']:.3f} |")
    lines.append(f"| dr (row) | {drift['dr']:+.3f} |")
    lines.append(f"| dc (col) | {drift['dc']:+.3f} |")
    lines.append(
        f"| drift_material (>= {DRIFT_MATERIAL_PX:.1f} px) | {drift['drift_material']} |"
    )
    lines.append("")
    lines.append("## 归因")
    lines.append("")
    attribution = payload["attribution"]
    lines.append("| 量 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| readout_floor_rms | {attribution['readout_floor_rms']:.3f} px |")
    lines.append(f"| write_screen_rms | {attribution['write_screen_rms']:.3f} px |")
    lines.append(f"| drift_px | {attribution['drift_px']:.3f} px |")
    ratio = attribution["ratio"]
    ratio_text = "None (A.rms_px<=0)" if ratio is None else f"{ratio:.3f}"
    lines.append(f"| ratio = B.rms_px / A.rms_px | {ratio_text} |")
    share_readout = attribution["readout_share"]
    share_write = attribution["write_screen_share"]
    lines.append(
        "| 分量占比 | "
        f"读取/提取 {('n/a' if share_readout is None else f'{share_readout:.1%}')} / "
        f"写屏 {('n/a' if share_write is None else f'{share_write:.1%}')} |"
    )
    lines.append(
        f"| 裁定 | **{attribution['verdict']}** {attribution['verdict_label']} |"
    )
    lines.append("")
    lines.append("## 交叉核对（仅佐证，非门禁）")
    lines.append("")
    cross = payload["crosscheck"]
    refs = cross["run_05_intra_upload_ref_px"]
    lines.append("| 量 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| S1e run_05 intra_upload_dist | {refs[0]:.2f}-{refs[1]:.2f} px |")
    lines.append(f"| A.rms_px | {cross['A_rms_px']:.3f} px |")
    lines.append(f"| B.rms_px | {cross['B_rms_px']:.3f} px |")
    lines.append("")
    lines.append(f"> {cross['note']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--judgment-name", default=DEFAULT_JUDGMENT_NAME)
    args = parser.parse_args(argv)

    logger.info(f"S1f 判读: capture_dir={args.capture_dir}")
    try:
        payload = judge(args.capture_dir)
    except JudgmentError as exc:
        logger.error(f"判读失败: {exc}")
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.judgment_name}.json"
    tables_path = out_dir / f"{args.judgment_name}_tables.md"
    write_json(json_path, payload)
    tables = render_markdown(payload)
    tables_path.write_text(tables + "\n", encoding="utf-8")

    print(tables)
    print()
    print(f"verdict: {payload['attribution']['verdict']} "
          f"({payload['attribution']['verdict_label']})")
    print(f"json   -> {json_path}")
    print(f"tables -> {tables_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
