"""双口径 CV 的离线后处理 —— issue #165 的 P1 并行条款。

**为什么需要它.** 一个 CV 数字只有在**口径明确**时才有意义. 本仓库现行闭环优化
用的是口径 B（目标方框 std/mean）, 而 Zhou 2010 一类文献对标用的是口径 A
（小平坦区 rms）. 两者数值不同, **不可互相比较**, 也不可合成一个数. 本脚本对
**已经落盘**的远场帧同时算出两个口径, 并给每个数字标出它属于哪个口径.

**两个口径的定义（代码 / CSV / MD 逐字一致, 见 :data:`APERTURE_DEFINITIONS`）**

* **口径 B / ``target_full``**（现行闭环口径）: 以 ``center`` 为中心、边长
  ``target_side``（相机像素）的目标方框, ``cv = std(region)/mean(region)``.
* **口径 A / ``plateau_inner``**（文献对标口径, Zhou 2010 “小平坦区 rms”）:
  同一 ``center``、边长 ``max(3, round(plateau_fraction * target_side))`` 的
  **内层**窗口（居中, 裁剪到画幅内）, ``cv = rms(region)/mean(region)``,
  ``rms = sqrt(mean((I-mean)^2))``.

**完全离线.** 只读磁盘上已有的 ``probes/*.npz`` / ``preflight/*.npz`` / 历史
CSV; **不打开任何设备**（无相机 / 无 SLM）, 不 import 任何驱动包.

**中心的确定顺序（不许回退到画幅几何中心）.** ① ``--center X,Y``; ②
``preflight/flat.npz`` 的平场帧; ③ ``probes/`` 里 ``split == "flat"`` 的帧;
两者都用仓库既有的
:func:`ao_shaping.utils.image.beam_metrics.zero_order_center`（``refine=False``,
与运行时 ``slm_model_in_loop.py`` 一致）; ④ 都拿不到 ⇒ **明确报错退出（退出码
2）**. 2f 光路上 0 阶不在画幅中心, 用几何中心兜底是仓库记录的反模式.

用法::

    python scripts/slm/dual_aperture_cv.py --from-dir data/debug/slm_model_in_loop/<ts>
    python scripts/slm/dual_aperture_cv.py --from-dir <run> --split shaped --strict
    python scripts/slm/dual_aperture_cv.py --from-dir <run> --csv out.csv --json out.json --md out.md

退出码: ``0`` 正常; ``1`` ``--strict`` 下自检不过; ``2`` 输入缺失（找不到中心 /
缺 ``run_config.json`` 等）.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ao_shaping.utils.image.beam_metrics import (  # noqa: E402
    compute_square_metrics,
    zero_order_center,
)
from scripts._common import markdown_table  # noqa: E402

# ---------------------------------------------------------------------------
# 口径定义 —— 代码 / CSV / MD 共用同一份字符串, 保证「逐字一致」
# ---------------------------------------------------------------------------

APERTURE_TARGET_FULL = "target_full"
"""口径 B: 现行闭环优化的目标方框（std/mean）."""

APERTURE_PLATEAU_INNER = "plateau_inner"
"""口径 A: 文献对标的小平坦区内层窗口（rms/mean）."""

APERTURE_DEFINITIONS: dict[str, str] = {
    APERTURE_TARGET_FULL: (
        "口径 B / target_full（现行闭环口径）：以 center 为中心、边长 target_side"
        "（相机像素）的目标方框，cv = std(region)/mean(region)。"
    ),
    APERTURE_PLATEAU_INNER: (
        "口径 A / plateau_inner（文献对标口径，Zhou 2010 “小平坦区 rms”）："
        "同一 center、边长 max(3, round(plateau_fraction * target_side)) 的内层"
        "窗口（居中，裁剪到画幅内），cv = rms(region)/mean(region)，"
        "rms = sqrt(mean((I-mean)^2))。"
    ),
}

APERTURE_LABELS: dict[str, str] = {
    APERTURE_TARGET_FULL: "口径 B：目标方框 std/mean",
    APERTURE_PLATEAU_INNER: "口径 A：小平坦区 rms/mean",
}

#: 紧凑的口径字母标注（stdout 表用）.
APERTURE_SHORT: dict[str, str] = {
    APERTURE_TARGET_FULL: "B",
    APERTURE_PLATEAU_INNER: "A",
}

CROSS_APERTURE_RULE = (
    "两个口径的数字不得互相比较，也不得合成一个数；引用时必须带上口径标注。"
)

#: 自检没能做时的原因说明（``self_check["reason"]`` ⇒ 人话）.
SELF_CHECK_REASONS: dict[str, str] = {
    "ok": "自检已做",
    "no_history_csv": "运行目录里没有 slm_model_in_loop_*.csv",
    "no_comparable_round": "历史 CSV 里没有与所选帧同轮的 uniformity_cv 记录",
}

DEFAULT_PLATEAU_FRACTION = 0.4
DEFAULT_TOL = 0.05

CSV_COLUMNS: tuple[str, ...] = (
    "source_file",
    "round",
    "split",
    "aperture",
    "aperture_definition",
    "center_x",
    "center_y",
    "target_side_px",
    "plateau_side_px",
    "cv",
    "ee_target",
    "peak",
    "mean_region",
    "exposure_ms",
    "timestamp_utc",
    "cv_recorded",
    "cv_check_rel_delta",
)


class DualApertureError(RuntimeError):
    """输入缺失或不自洽（找不到中心 / 缺配置 / 帧不可用）."""


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FrameRecord:
    """一条已落盘的远场帧.

    Attributes:
        file: ``probes/`` 下的文件名（与 manifest 的 ``file`` 一致）.
        path: 磁盘上的绝对路径.
        round: 所属轮次（0 = 平场轮）.
        split: ``flat`` / ``train`` / ``holdout`` / ``shaped``.
        family: 探针族（manifest 可能给 ``None``）.
        index: 该轮内的探针序号.
        exposure_ms: 曝光时间（ms），manifest 可能给 ``None``.
        timestamp_utc: 采集时刻（ISO 8601）, manifest 可能给 ``None``.
        frame: **已 ``prepare_frame``** 的相机帧（2D float 数组）.
    """

    file: str
    path: Path
    round: int
    split: str
    family: str | None
    index: int
    exposure_ms: float | None
    timestamp_utc: str | None
    frame: np.ndarray


@dataclass
class Session:
    """一次 ``slm-model-in-loop --debug`` 运行的落盘产物.

    Attributes:
        run_dir: 运行目录.
        config: ``run_config.json`` 的内容（不存在时为空字典）.
        frames: 已加载的远场帧.
        history: recorder 历史 CSV 的行（``dict[str, str]``）.
        history_path: 历史 CSV 的路径（没有则 ``None``）.
    """

    run_dir: Path
    config: dict[str, Any]
    frames: list[FrameRecord]
    history: list[dict[str, str]]
    history_path: Path | None


# ---------------------------------------------------------------------------
# 纯函数: 窗口与两个口径的指标
# ---------------------------------------------------------------------------


def target_window(
    center: tuple[float, float],
    target_side: int,
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """口径 B 的目标方框像素范围 ``(x0, x1, y0, y1)``（半开区间）.

    窗口公式与
    :func:`ao_shaping.utils.image.beam_metrics.compute_square_metrics`
    **逐字一致**（``half = max(target_side // 2, 1)``, 先按中心取窗口再裁剪到
    画幅）, 这样重算出的 ``cv_target_full`` 才能与运行时记录的 ``uniformity_cv``
    对得上 —— 口径 B 的自检就建立在这个一致性上.

    Args:
        center: 冻结的光轴中心 ``(x, y)``（相机像素）.
        target_side: 目标方框边长（相机像素）.
        frame_shape: ``(height, width)``.

    Returns:
        ``(x0, x1, y0, y1)``, 覆盖 ``[x0, x1) x [y0, y1)``.
    """
    height, width = frame_shape
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    half = max(int(target_side) // 2, 1)
    x0 = max(cx - half, 0)
    x1 = min(cx + half, width)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, height)
    return x0, x1, y0, y1


def plateau_window(
    center: tuple[float, float],
    target_side: int,
    plateau_fraction: float,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """口径 A 的内层窗口像素范围 ``(x0, x1, y0, y1)``（半开区间）.

    边长 = ``max(3, round(plateau_fraction * target_side))``, 与 ``center``
    **同心**（内层窗口, 不是偏心的一小块）.

    Args:
        center: 冻结的光轴中心 ``(x, y)``（相机像素）.
        target_side: 口径 B 的目标方框边长（相机像素）.
        plateau_fraction: 内层边长占 ``target_side`` 的比例.
        frame_shape: ``(height, width)``；给出时把窗口裁剪到画幅内.

    Returns:
        ``(x0, x1, y0, y1)``, 覆盖 ``[x0, x1) x [y0, y1)``.
    """
    side = max(3, int(round(float(plateau_fraction) * int(target_side))))
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    half = side // 2
    # 奇数边长时多带一行/一列, 使实际宽度恒等于 side。
    extra = side - 2 * half
    x0 = cx - half
    x1 = cx + half + extra
    y0 = cy - half
    y1 = cy + half + extra
    if frame_shape is not None:
        height, width = frame_shape
        x0 = max(x0, 0)
        x1 = min(x1, width)
        y0 = max(y0, 0)
        y1 = min(y1, height)
    return x0, x1, y0, y1


def _region(frame: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    """按 ``(x0, x1, y0, y1)`` 取子区域, 空区域直接报错."""
    x0, x1, y0, y1 = window
    if x1 <= x0 or y1 <= y0:
        raise DualApertureError(
            f"窗口落在画幅之外: {window}（中心落在画幅外, 或边长为 0）"
        )
    return frame[y0:y1, x0:x1]


def aperture_metrics(
    frame: np.ndarray,
    center: tuple[float, float],
    target_side: int,
    plateau_fraction: float,
) -> dict[str, float]:
    """一帧远场图的两个口径指标.

    Args:
        frame: 已 ``prepare_frame`` 的相机帧（2D）.
        center: 冻结的光轴中心 ``(x, y)``（相机像素）.
        target_side: 口径 B 的目标方框边长（相机像素）.
        plateau_fraction: 口径 A 的内层边长比例.

    Returns:
        至少含 ``cv_target_full``（口径 B）、``cv_plateau_inner``（口径 A）、
        ``ee_target``、``peak``、``mean_target``、``plateau_side_px``;
        另外给出 ``mean_plateau`` / ``rms_plateau`` / ``std_target`` /
        ``plateau_side_actual_px`` / ``target_side_actual_px`` 供核对.
    """
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim != 2 or image.size == 0:
        raise DualApertureError(f"帧必须是非空 2D 数组, 实际 shape={image.shape}")

    height, width = image.shape
    box_b = target_window(center, target_side, (height, width))
    box_a = plateau_window(center, target_side, plateau_fraction, (height, width))
    region_b = _region(image, box_b)
    region_a = _region(image, box_a)

    mean_b = float(np.mean(region_b))
    std_b = float(np.std(region_b))
    # 与 compute_square_metrics 逐字一致（含 1e-10 的分母下限）。
    cv_b = std_b / max(mean_b, 1e-10) if mean_b > 0 else 0.0

    mean_a = float(np.mean(region_a))
    rms_a = float(np.sqrt(np.mean((region_a - mean_a) ** 2)))
    cv_a = rms_a / mean_a if mean_a > 0 else 0.0

    total = float(np.sum(image))
    # 与 compute_square_metrics 的 encircled_energy 同口径（total<=0 时取 0）。
    ee_target = float(np.sum(region_b)) / max(total, 1e-10) if total > 0 else 0.0

    # 口径 B 必须与运行时的 compute_square_metrics 逐位一致（自检的基础）。
    canonical = compute_square_metrics(image, int(target_side), center)
    if abs(float(canonical["uniformity_cv"]) - cv_b) > 1e-12:
        raise DualApertureError(
            "口径 B 的窗口公式与 ao_shaping.utils.image.beam_metrics."
            f"compute_square_metrics 不一致 ({cv_b!r} vs "
            f"{float(canonical['uniformity_cv'])!r}) —— 仓库的窗口约定变了, "
            "本脚本必须同步, 否则自检永远报警"
        )

    return {
        "cv_target_full": float(cv_b),
        "cv_plateau_inner": float(cv_a),
        "ee_target": float(ee_target),
        "peak": float(np.max(image)),
        "mean_target": float(mean_b),
        "plateau_side_px": float(
            max(3, int(round(float(plateau_fraction) * int(target_side))))
        ),
        "mean_plateau": float(mean_a),
        "rms_plateau": float(rms_a),
        "std_target": float(std_b),
        "plateau_side_actual_px": float(box_a[1] - box_a[0]),
        "target_side_actual_px": float(box_b[1] - box_b[0]),
    }


# ---------------------------------------------------------------------------
# 落盘产物的读取
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_frame(path: Path) -> np.ndarray | None:
    """从一个 ``probes/*.npz`` 里取 ``frame`` 数组.

    只取 ``frame``（**已 ``prepare_frame``** 的相机帧）; ``raw_frame`` 是
    prepare 之前的读出, 量纲与噪声底都不同, 拿它算 CV 会得到另一个口径的数.
    """
    with np.load(path, allow_pickle=False) as payload:
        if "frame" not in payload:
            return None
        return np.asarray(payload["frame"], dtype=np.float64)


def _history_csv(run_dir: Path) -> tuple[Path | None, list[dict[str, str]]]:
    """读 ``slm_model_in_loop_*.csv``（可选）."""
    candidates = sorted(run_dir.glob("slm_model_in_loop_*.csv"))
    if not candidates:
        return None, []
    path = candidates[-1]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return path, rows


def load_session(
    run_dir: str | Path,
    *,
    splits: set[str] | None = None,
    rounds: set[int] | None = None,
) -> Session:
    """读入一次 ``slm-model-in-loop --debug`` 运行的落盘产物.

    只读磁盘: ``run_config.json`` + ``probes/manifest.json`` +
    ``probes/*.npz`` + (可选) ``slm_model_in_loop_*.csv``. **不碰任何设备.**

    Args:
        run_dir: 运行目录（``data/debug/slm_model_in_loop/<ts>/``）.
        splits: 只加载这些 ``split``; ``None`` = 全部.
        rounds: 只加载这些轮次; ``None`` = 全部.

    Returns:
        :class:`Session`.

    Raises:
        DualApertureError: 目录不存在 / manifest 不可读.
    """
    root = Path(run_dir).expanduser()
    if not root.is_dir():
        raise DualApertureError(f"运行目录不存在: {root}")

    config: dict[str, Any] = {}
    config_path = root / "run_config.json"
    if config_path.is_file():
        loaded = _read_json(config_path)
        if isinstance(loaded, dict):
            config = loaded

    manifest_path = root / "probes" / "manifest.json"
    manifest: list[dict[str, Any]] = []
    if manifest_path.is_file():
        loaded_manifest = _read_json(manifest_path)
        if isinstance(loaded_manifest, list):
            manifest = loaded_manifest

    frames: list[FrameRecord] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        split = str(entry.get("split", ""))
        if splits is not None and split not in splits:
            continue
        try:
            round_index = int(entry.get("round", 0))
        except (TypeError, ValueError):
            continue
        if rounds is not None and round_index not in rounds:
            continue
        name = str(entry.get("file", ""))
        if not name:
            continue
        path = root / "probes" / name
        if not path.is_file():
            continue
        frame = _load_npz_frame(path)
        if frame is None:
            continue
        exposure = entry.get("exposure_ms")
        try:
            probe_index = int(entry.get("index", 0) or 0)
        except (TypeError, ValueError):
            probe_index = 0
        frames.append(
            FrameRecord(
                file=name,
                path=path,
                round=round_index,
                split=split,
                family=(
                    str(entry["family"]) if entry.get("family") is not None else None
                ),
                index=probe_index,
                exposure_ms=float(exposure) if exposure is not None else None,
                timestamp_utc=(
                    str(entry["timestamp_utc"])
                    if entry.get("timestamp_utc") is not None
                    else None
                ),
                frame=frame,
            )
        )

    history_path, history = _history_csv(root)
    return Session(
        run_dir=root,
        config=config,
        frames=frames,
        history=history,
        history_path=history_path,
    )


# ---------------------------------------------------------------------------
# 中心定位
# ---------------------------------------------------------------------------


def locate_center(
    session: Session,
    override: tuple[float, float] | None = None,
) -> tuple[tuple[int, int], str]:
    """确定冻结的光轴中心 ``(x, y)``.

    顺序: ① ``--center`` 覆盖; ② ``preflight/flat.npz`` 的平场帧; ③
    ``probes/`` 里 ``split == "flat"`` 的帧. ②③ 都用仓库既有的
    :func:`ao_shaping.utils.image.beam_metrics.zero_order_center`
    （``refine=False``, 与运行时一致）. 都没有 ⇒ 报错.

    **绝不用画幅几何中心兜底** —— 2f 光路上 0 阶不在画幅中心, 那是仓库记录的
    反模式.

    Args:
        session: 已加载的运行目录; 它的 ``frames`` 里最好含 ``split == "flat"``
            的平场帧（``--rounds`` 过滤会把第 0 轮滤掉, 所以调用方要另外准备
            一份只含平场帧的 session）.
        override: ``--center`` 给出的覆盖值.

    Returns:
        ``((x, y), 来源说明)``.

    Raises:
        DualApertureError: 找不到任何可用的平场帧.
    """
    if override is not None:
        center = (int(round(override[0])), int(round(override[1])))
        return center, "--center 显式覆盖"

    preflight = session.run_dir / "preflight" / "flat.npz"
    if preflight.is_file():
        frame = _load_npz_frame(preflight)
        if frame is not None and float(np.max(frame)) > 0:
            center = zero_order_center(frame, refine=False)
            return (int(center[0]), int(center[1])), "preflight/flat.npz（平场帧）"

    for record in session.frames:
        if record.split != "flat":
            continue
        if float(np.max(record.frame)) <= 0:
            continue
        center = zero_order_center(record.frame, refine=False)
        return (
            (int(center[0]), int(center[1])),
            f"probes/{record.file}（split=flat 的平场帧）",
        )

    raise DualApertureError(
        "找不到平场帧, 无法确定光轴中心: 既没有 preflight/flat.npz, "
        "probes/ 里也没有 split == \"flat\" 的帧. "
        "请用 --center X,Y 显式给出中心 —— 不要用画幅几何中心兜底"
        "（2f 光路上 0 阶不在画幅中心）"
    )


# ---------------------------------------------------------------------------
# 自检: 口径 B 与运行时 uniformity_cv 的一致性
# ---------------------------------------------------------------------------


def recorded_cv_by_round(history: list[dict[str, str]]) -> dict[int, float]:
    """从历史 CSV 里取出「轮次 → 运行时记录的 ``uniformity_cv``」.

    ``round == 0`` 是 ``stage == "calibration"`` 的平场基线行, ``round >= 1``
    是 ``stage == "round"`` 的整形行; 后者覆盖前者.
    """
    out: dict[int, float] = {}
    for row in history:
        raw = row.get("uniformity_cv")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        try:
            round_index = int(float(row.get("round", "")))
        except (TypeError, ValueError):
            continue
        stage = str(row.get("stage", "")).strip()
        if round_index in out and stage != "round":
            continue
        out[round_index] = value
    return out


def relative_delta(recomputed: float, recorded: float) -> float:
    """相对偏差 ``|recomputed - recorded| / max(|recorded|, 1e-9)``."""
    return abs(float(recomputed) - float(recorded)) / max(abs(float(recorded)), 1e-9)


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _num(value: float | int | None) -> str:
    """CSV 用的数值格式化（``None`` / NaN ⇒ 空串）."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.10g}"


def build_rows(
    frames: list[FrameRecord],
    center: tuple[int, int],
    target_side: int,
    plateau_fraction: float,
    recorded: dict[int, float],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """逐帧算出两行（两个口径）, 并顺带做口径 B 的自检.

    Returns:
        ``(CSV 行, 自检结论)``.
    """
    rows: list[dict[str, str]] = []
    checks: list[dict[str, Any]] = []
    for record in frames:
        metrics = aperture_metrics(
            record.frame, center, target_side, plateau_fraction
        )
        recorded_value = recorded.get(record.round)
        if recorded_value is None:
            delta: float | None = None
        else:
            delta = relative_delta(metrics["cv_target_full"], recorded_value)
            checks.append(
                {
                    "source_file": record.file,
                    "round": record.round,
                    "split": record.split,
                    "cv_target_full": metrics["cv_target_full"],
                    "cv_recorded": recorded_value,
                    "cv_check_rel_delta": delta,
                }
            )
        for aperture, cv_key, mean_key in (
            (APERTURE_TARGET_FULL, "cv_target_full", "mean_target"),
            (APERTURE_PLATEAU_INNER, "cv_plateau_inner", "mean_plateau"),
        ):
            rows.append(
                {
                    "source_file": record.file,
                    "round": str(record.round),
                    "split": record.split,
                    "aperture": aperture,
                    "aperture_definition": APERTURE_DEFINITIONS[aperture],
                    "center_x": str(center[0]),
                    "center_y": str(center[1]),
                    "target_side_px": str(int(target_side)),
                    "plateau_side_px": _num(metrics["plateau_side_px"]),
                    "cv": _num(metrics[cv_key]),
                    "ee_target": _num(metrics["ee_target"]),
                    "peak": _num(metrics["peak"]),
                    "mean_region": _num(metrics[mean_key]),
                    "exposure_ms": _num(record.exposure_ms),
                    "timestamp_utc": record.timestamp_utc or "",
                    "cv_recorded": _num(recorded_value),
                    "cv_check_rel_delta": _num(delta),
                }
            )
    summary = {
        "performed": bool(checks),
        "reason": "ok" if checks else "no_comparable_round",
        "rows": checks,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """写 CSV（列名固定为 :data:`CSV_COLUMNS`）."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON（UTF-8, ``ensure_ascii=False`` 便于人读）."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _md_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    """把 CSV 行渲染成 MD 表的行（每个 CV 数字后面标出口径）."""
    out: list[list[str]] = []
    for row in rows:
        cv_text = row["cv"]
        label = APERTURE_LABELS[row["aperture"]]
        out.append(
            [
                row["source_file"],
                row["round"],
                row["split"],
                row["aperture"],
                f"{cv_text}（{label}）" if cv_text else "-",
                row["target_side_px"],
                row["plateau_side_px"],
                row["ee_target"],
                row["peak"],
                row["mean_region"],
                row["exposure_ms"],
                row["timestamp_utc"],
                row["cv_recorded"],
                row["cv_check_rel_delta"],
            ]
        )
    return out


def _md_header() -> list[str]:
    return [
        "源文件",
        "round",
        "split",
        "口径",
        "CV（**必须带口径标注**）",
        "target_side_px",
        "plateau_side_px",
        "ee_target",
        "peak",
        "mean_region",
        "exposure_ms",
        "timestamp_utc",
        "cv_recorded（口径 B）",
        "cv_check_rel_delta（口径 B）",
    ]


def render_markdown(
    rows: list[dict[str, str]],
    *,
    run_dir: Path,
    center: tuple[int, int],
    center_source: str,
    target_side: int,
    plateau_fraction: float,
    plateau_side: int,
    self_check: dict[str, Any],
    tol: float,
    strict: bool,
) -> str:
    """渲染 MD 报告（口径定义 + 自检结论 + 一张逐帧表）."""
    lines: list[str] = []
    lines.append("# 双口径 CV 离线后处理")
    lines.append("")
    lines.append(f"- 运行目录：`{run_dir}`")
    lines.append(
        f"- 冻结中心：`({center[0]}, {center[1]})`（相机像素），来源：{center_source}"
    )
    lines.append(f"- 目标边长 `target_side`：**{target_side} px**（相机像素）")
    lines.append(
        f"- 内层比例 `plateau_fraction` = {plateau_fraction} ⇒ "
        f"`plateau_side` = **{plateau_side} px**"
    )
    lines.append("")
    lines.append("## 1. 口径定义")
    lines.append("")
    for aperture in (APERTURE_TARGET_FULL, APERTURE_PLATEAU_INNER):
        lines.append(f"- {APERTURE_DEFINITIONS[aperture]}")
    lines.append("")
    lines.append(f"> {CROSS_APERTURE_RULE}")
    lines.append("")
    lines.append("## 2. 自检（口径 B 与运行时 `uniformity_cv` 的一致性）")
    lines.append("")
    if not self_check["performed"]:
        reason = SELF_CHECK_REASONS.get(
            str(self_check.get("reason", "")), str(self_check.get("reason", ""))
        )
        lines.append(f"- **未做自检**：{reason}")
        lines.append(
            "- 口径 B 与运行时记录的一致性**没有被验证** —— 引用本表的数字前"
            "请自行核对中心坐标与目标边长"
        )
    else:
        deltas = [
            float(row["cv_check_rel_delta"])
            for row in self_check["rows"]
            if row.get("cv_check_rel_delta") is not None
        ]
        worst = max(deltas) if deltas else 0.0
        verdict = "通过" if worst <= tol else "**不通过**"
        lines.append(
            f"- 比对 {len(self_check['rows'])} 轮，容差 `--tol {tol}`，"
            f"最大相对偏差 **{worst:.4g}** ⇒ {verdict}"
        )
        lines.append(
            "- `cv_recorded` / `cv_check_rel_delta` 两列都是**口径 B** 的量"
            "（运行时记录的 `uniformity_cv` 与它的相对偏差），两行相同"
        )
        lines.append("")
        lines.append(
            markdown_table(
                ["源文件", "round", "口径 B 重算", "运行时记录", "相对偏差"],
                [
                    [
                        row["source_file"],
                        str(row["round"]),
                        f"{float(row['cv_target_full']):.6g}",
                        f"{float(row['cv_recorded']):.6g}",
                        f"{float(row['cv_check_rel_delta']):.4g}",
                    ]
                    for row in self_check["rows"]
                ],
            )
        )
        if worst > tol:
            lines.append("")
            lines.append(
                "> ⚠️ 偏差超出容差。**最常见的原因是坐标系搞错**（把窗口中心与"
                "画幅中心搞混）；另一个可能是运行时的 `target_camera` 由标定出的"
                "几何比例换算而来、与 `run_config.json` 的 `target_side` 不完全"
                "相等（硬件路径）。先核对中心坐标，再看 `target_side`。"
            )
    lines.append("")
    lines.append("## 3. 逐帧结果")
    lines.append("")
    lines.append(markdown_table(_md_header(), _md_rows(rows)))
    lines.append("")
    lines.append(
        "> 表中每个 CV 数字后面都写明了它属于哪个口径；"
        "`ee_target` / `peak` / `mean_region` 是**该行口径所在区域**的量"
        "（`peak` 是整帧最大值）。"
    )
    lines.append("")
    lines.append(f"> {CROSS_APERTURE_RULE}")
    lines.append("")
    if strict:
        lines.append(f"（本次以 `--strict` 运行, 容差 {tol}）")
        lines.append("")
    return "\n".join(lines)


def render_stdout(
    rows: list[dict[str, str]],
    *,
    center: tuple[int, int],
    target_side: int,
    plateau_side: int,
    self_check: dict[str, Any],
    tol: float,
) -> str:
    """渲染 stdout 上的文本表（不给 ``--csv/--json/--md`` 时用）."""
    header = (
        f"{'source_file':<34} {'rnd':>3} {'split':<8} {'口径(aperture)':<22} "
        f"{'cv':>10} {'ee':>8} {'peak':>10} {'mean':>10} {'check':>9}"
    )
    lines = [
        f"中心 = ({center[0]}, {center[1]}) 目标边长 = {target_side} px  "
        f"内层边长 = {plateau_side} px",
        header,
        "-" * len(header),
    ]
    for row in rows:
        check = row["cv_check_rel_delta"] or "-"
        aperture = (
            f"口径{APERTURE_SHORT[row['aperture']]}:{row['aperture']}"
        )
        lines.append(
            f"{row['source_file']:<34} {row['round']:>3} {row['split']:<8} "
            f"{aperture:<22} {row['cv']:>10} {row['ee_target']:>8} "
            f"{row['peak']:>10} {row['mean_region']:>10} {check:>9}"
        )
    if self_check["performed"]:
        deltas = [
            float(row["cv_check_rel_delta"])
            for row in self_check["rows"]
            if row.get("cv_check_rel_delta") is not None
        ]
        worst = max(deltas) if deltas else 0.0
        lines.append("")
        lines.append(
            f"自检（口径 B vs 运行时 uniformity_cv）: {len(self_check['rows'])} 轮, "
            f"最大相对偏差 {worst:.4g}, 容差 {tol} ⇒ "
            f"{'通过' if worst <= tol else '不通过'}"
        )
    else:
        lines.append("")
        reason = SELF_CHECK_REASONS.get(
            str(self_check.get("reason", "")), str(self_check.get("reason", ""))
        )
        lines.append(f"自检: 未做（{reason}）")
    lines.append("")
    lines.append(CROSS_APERTURE_RULE)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_center(text: str) -> tuple[float, float]:
    parts = [piece.strip() for piece in text.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            f"--center 需要 \"X,Y\" 形式, 实际收到 {text!r}"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--center 的坐标必须是数字, 实际收到 {text!r}"
        ) from exc


def _parse_rounds(text: str) -> set[int]:
    parts = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            f"--rounds 需要 \"1,2\" 形式, 实际收到 {text!r}"
        )
    out: set[int] = set()
    for piece in parts:
        try:
            out.add(int(piece))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--rounds 的轮次必须是整数, 实际收到 {piece!r}"
            ) from exc
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "双口径 CV 的离线后处理: 对已落盘的远场帧同时算出"
            "口径 B (目标方框 std/mean) 与口径 A (小平坦区 rms), "
            "并给每个数字标出口径. 完全离线, 不打开任何设备."
        )
    )
    parser.add_argument(
        "--from-dir",
        required=True,
        help="slm-model-in-loop --debug 的运行目录",
    )
    parser.add_argument(
        "--split",
        default="shaped",
        help="要统计的 split（flat / train / holdout / shaped）, 默认 shaped",
    )
    parser.add_argument(
        "--rounds",
        type=_parse_rounds,
        default=None,
        help="只统计这些轮次, 如 \"1,2\"; 默认全部",
    )
    parser.add_argument(
        "--plateau-fraction",
        type=float,
        default=DEFAULT_PLATEAU_FRACTION,
        help=f"口径 A 的内层边长比例, 默认 {DEFAULT_PLATEAU_FRACTION}",
    )
    parser.add_argument(
        "--center",
        type=_parse_center,
        default=None,
        help="显式给出光轴中心 \"X,Y\"（相机像素）; 默认从平场帧重算",
    )
    parser.add_argument(
        "--target-side",
        type=int,
        default=None,
        help="目标方框边长（相机像素）; 默认取 run_config.json 的 target_side",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"自检的相对容差, 默认 {DEFAULT_TOL}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="自检不过（相对偏差 > --tol）时以退出码 1 结束",
    )
    parser.add_argument("--csv", default=None, help="CSV 输出路径")
    parser.add_argument("--json", default=None, help="JSON 输出路径")
    parser.add_argument("--md", default=None, help="Markdown 输出路径")
    args = parser.parse_args(argv)
    if not 0.0 < args.plateau_fraction <= 1.0:
        parser.error("--plateau-fraction 必须在 (0, 1] 区间内")
    if args.target_side is not None and args.target_side < 1:
        parser.error("--target-side 必须是正整数")
    if args.tol < 0:
        parser.error("--tol 不能为负")
    return args


def _run(args: argparse.Namespace) -> int:
    """真正干活的地方（异常统一在这里被翻译成退出码）."""
    run_dir = Path(args.from_dir).expanduser()
    session = load_session(run_dir, splits={args.split}, rounds=args.rounds)
    if not session.frames:
        raise DualApertureError(
            f"运行目录里没有可用的探针帧: {run_dir}（split={args.split!r}）"
        )

    # 平场帧始终要单独读一份: 它是中心定位的依据, 而 ``--rounds`` 会把第 0 轮
    # 一起滤掉（中心不能因为「只统计第 2 轮」就找不到）。
    flat_session = session
    if not any(record.split == "flat" for record in session.frames):
        flat_session = load_session(run_dir, splits={"flat"})

    target_side = args.target_side
    if target_side is None:
        raw = session.config.get("target_side")
        if raw is None:
            raise DualApertureError(
                f"{run_dir / 'run_config.json'} 里没有 target_side, "
                "请用 --target-side 显式给出目标方框边长（相机像素）"
            )
        target_side = int(raw)
    if target_side < 1:
        raise DualApertureError(f"target_side 必须是正整数, 实际 {target_side}")

    center, center_source = locate_center(flat_session, args.center)

    frames = [record for record in session.frames if record.split == args.split]
    if not frames:
        raise DualApertureError(
            f"运行目录里没有 split == {args.split!r} 的帧: {run_dir}"
        )
    frames.sort(key=lambda record: (record.round, record.index, record.file))

    recorded = recorded_cv_by_round(session.history)
    rows, self_check = build_rows(
        frames,
        center,
        target_side,
        args.plateau_fraction,
        recorded,
    )
    self_check["tolerance"] = float(args.tol)
    if session.history_path is None:
        self_check["reason"] = "no_history_csv"
    elif self_check["performed"]:
        self_check["reason"] = "ok"
    else:
        self_check["reason"] = "no_comparable_round"
    deltas = [
        float(row["cv_check_rel_delta"])
        for row in self_check["rows"]
        if row.get("cv_check_rel_delta") is not None
    ]
    worst = max(deltas) if deltas else 0.0
    self_check["max_rel_delta"] = worst
    self_check["passed"] = bool(self_check["performed"]) and worst <= float(args.tol)
    self_check["strict"] = bool(args.strict)

    plateau_side = max(
        3, int(round(float(args.plateau_fraction) * int(target_side)))
    )

    if args.csv:
        write_csv(Path(args.csv), rows)
    if args.json:
        payload = {
            "run_dir": str(run_dir),
            "split": args.split,
            "rounds": sorted(args.rounds) if args.rounds else None,
            "center": {"x": center[0], "y": center[1], "source": center_source},
            "target_side_px": int(target_side),
            "plateau_fraction": float(args.plateau_fraction),
            "plateau_side_px": int(plateau_side),
            "aperture_definitions": dict(APERTURE_DEFINITIONS),
            "aperture_labels": dict(APERTURE_LABELS),
            "cross_aperture_rule": CROSS_APERTURE_RULE,
            "self_check": self_check,
            "rows": rows,
        }
        write_json(Path(args.json), payload)
    if args.md:
        text = render_markdown(
            rows,
            run_dir=run_dir,
            center=center,
            center_source=center_source,
            target_side=int(target_side),
            plateau_fraction=float(args.plateau_fraction),
            plateau_side=plateau_side,
            self_check=self_check,
            tol=float(args.tol),
            strict=bool(args.strict),
        )
        out_md = Path(args.md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(text, encoding="utf-8")

    if not (args.csv or args.json or args.md):
        print(
            render_stdout(
                rows,
                center=center,
                target_side=int(target_side),
                plateau_side=plateau_side,
                self_check=self_check,
                tol=float(args.tol),
            )
        )

    if args.strict and self_check["performed"] and not self_check["passed"]:
        print(
            f"自检不通过: 口径 B 的重算值与运行时记录的 uniformity_cv 的最大相对偏差 "
            f"{worst:.4g} > 容差 {args.tol}",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """入口: 返回进程退出码（0 正常 / 1 自检不过 / 2 输入缺失）."""
    args = _parse_args(argv)
    try:
        return _run(args)
    except DualApertureError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
