"""整形效率与衍射效率匹配性：离线分析器（2026-11-23）。

为什么需要它
------------
``docs/dev/efficiency_match_experiment_plan_20261121.md`` 预注册了比较式

    eta_s = (S - S0) / F0      # 整形帧：目标掩膜内来自可调制分量的能量比例
    eta_d = G1 / F0            # 光栅帧：+1 级相对全平相位总入射基准的比例

本脚本把这个口径逐字落到代码里，只读采集目录（不碰设备、不写标定文件）：

- F0 取全平相位帧在固定半径 R0 圆盘内的暗减积分，逐帧算后取均值与组内标准差；
- 目标掩膜是冻结点中心的正方形（SIDE x SIDE），S 是整形帧内的暗减积分，
  S0 是全平帧在同一方窗内的暗减积分均值；
- +1 级在 ``SEP ± TOL`` 环带内做局部极大搜索（排除 0 级邻域），G1 是该峰附近
  半径 R1 圆盘内的暗减积分；反方向同法取 G-1 只作 blaze 泄漏诊断（只报告，
  不门控）；
- 判据见计划 §6：``|mean(d)| <= 0.05``、``CI95 上限 <= 0.08``、两效率组内
  sd ``<= 0.03``；不可判定门（缺暗帧、饱和、+1 级不可定位、G1 两轮组内离散
  超限、F0 不可定位、哈希/设备不符、漂移轮 > 1、效率越界）优先级最高；
  ±1 泄漏诊断不触发不可判定，也不进入匹配判定。

所有口径值（R0/R1/SIDE/SEP/TOL/排除半径/判据阈值）都来自采集目录 ``config.json``
的 ``roi_frozen`` 段（采集端冻结值真源；兼容旧 ``frozen_mask`` 键名）并写进
``analysis.json`` 的阈值快照；脚本里没有第二套口径可以偷偷漂移。产物只写采集
目录内的 ``analysis.json`` 与 ``EM_REPORT.md``，默认拒绝覆盖。

用法::

    python scripts/slm/efficiency_match_analyze_20261123.py --dir <采集目录>
    python scripts/slm/efficiency_match_analyze_20261123.py --dir <采集目录> --overwrite
    python scripts/slm/efficiency_match_analyze_20261123.py --self-test

``--self-test`` 只构造合成 uint16 帧（匹配 / 不匹配 / blaze 泄漏超预期 /
缺暗帧四个场景），不读真设备数据。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.ndimage import maximum_filter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._common import fmt_fixed_or_text

SCHEMA = "efficiency_match_analysis_v1"
ANALYSIS_NAME = "analysis.json"
REPORT_NAME = "EM_REPORT.md"

# 计划 §4 的冻结顺序：每一遍 8 个槽位，第二遍在帧 id 后加 ``_r2``。
PASS_ROLES: tuple[str, ...] = (
    "flat_pre",
    "blaze_grating",
    "shaping_target",
    "dark",
    "flat_mid",
    "blaze_grating",
    "shaping_target",
    "flat_post",
)
# 帧 id -> 逻辑角色；``blaze_grating`` / ``shaping_target`` 在同一遍里出现两次，
# 因此角色只是类别，重复序号由时间戳顺序另行分配。
ROLE_BY_ID: dict[str, str] = {
    "flat_pre": "flat_pre",
    "flat_mid": "flat_mid",
    "flat_post": "flat_post",
    "dark": "dark",
    "blaze_grating": "blaze",
    "shaping_target": "shaping",
}
FLAT_ROLES = frozenset({"flat_pre", "flat_mid", "flat_post"})

# 不可判定门的优先顺序（计划 §6：不可判定优先级最高）。
UNDETERMINABLE_GATES: tuple[tuple[str, str], ...] = (
    ("missing_same_exposure_dark", "missing_same_exposure_dark"),
    ("hash_or_device_mismatch", "phase_hash_or_device_identity_mismatch"),
    ("no_flat_frame", "no_flat_frame"),
    ("f0_not_locatable", "f0_not_locatable"),
    ("saturated_frame", "integration_frame_saturated"),
    ("insufficient_repeats", "insufficient_repeats"),
    (
        "energy_out_of_range",
        "out_of_range_energy_check_background_or_saturation",
    ),
    ("plus1_not_locatable", "plus1_order_not_locatable"),
    ("group_sd", "g1_group_sd_exceeds_limit"),
    ("drift_rounds", "drift_rounds_gt_1"),
)


# ---------------------------------------------------------------------------
# 配置、阈值与判据
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    """冻结积分口径与判据阈值（全部来自采集配置，写进报告）。"""

    r0: float = 90.0
    r1: float = 60.0
    side: float = 60.0
    sep: float = 471.0
    tol: float = 150.0
    zero_exclusion: float = 150.0
    match_abs_d: float = 0.05
    match_ci_high: float = 0.08
    match_sd: float = 0.03
    blaze_leakage_expected: float = 0.05
    drift_px: float = 2.0
    flat_drift_rel: float = 0.05
    saturation_fraction: float = 0.9
    peak_snr_min: float = 5.0
    roi_frozen_source: str = "builtin_defaults"
    roi_frozen_audit: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        """给 ``analysis.json`` 用的阈值快照。"""
        return {
            "r0_px": self.r0,
            "r1_px": self.r1,
            "side_px": self.side,
            "separation_px": self.sep,
            "separation_tolerance_px": self.tol,
            "zero_exclusion_radius_px": self.zero_exclusion,
            "match_abs_d": self.match_abs_d,
            "match_ci95_high": self.match_ci_high,
            "match_group_sd": self.match_sd,
            "blaze_leakage_expected": self.blaze_leakage_expected,
            "drift_centroid_px": self.drift_px,
            "flat_drift_relative": self.flat_drift_rel,
            "saturation_fraction": self.saturation_fraction,
            "peak_snr_min": self.peak_snr_min,
            "roi_frozen_source": self.roi_frozen_source,
            "roi_frozen_audit": self.roi_frozen_audit,
        }


def _as_float(value: Any, default: float) -> float:
    """把配置里的标量转 float；缺失/非法则回退默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def thresholds_from_config(config: dict[str, Any]) -> Thresholds:
    """从 ``config.json`` 读出冻结口径。

    优先读采集端写在顶层的 ``roi_frozen`` 块（冻结值真源，键名与采集脚本
    ``ROI_FROZEN`` 一致），做显式键映射（``r0_px``/``separation_seed_px``/
    ``separation_search_px``/``zero_order_exclusion_px`` 等）；没有
    ``roi_frozen`` 时退回旧键 ``frozen_mask`` / 顶层，再退回内置默认。来源写入
    ``roi_frozen_source``；``roi_frozen`` 里的 blaze 审计键（周期/轴/几何）原样
    带出到 ``roi_frozen_audit``。
    """
    base = Thresholds()
    frozen = config.get("roi_frozen")
    if not isinstance(frozen, dict):
        frozen = {}
    mask = config.get("frozen_mask")
    if not isinstance(mask, dict):
        mask = {}
    source = (
        "roi_frozen"
        if frozen
        else ("frozen_mask" if mask else "builtin_defaults")
    )
    audit = (
        {
            "blaze_period_panel_px": frozen.get("blaze_period_panel_px"),
            "blaze_axis": frozen.get("blaze_axis"),
            "geometry": frozen.get("geometry"),
        }
        if frozen
        else {}
    )

    def pick(*keys: str) -> Any:
        """先查 ``roi_frozen``，再查 ``frozen_mask``，再查顶层。"""
        for key in keys:
            if key in frozen:
                return frozen[key]
            if key in mask:
                return mask[key]
            if key in config:
                return config[key]
        return None

    return Thresholds(
        r0=_as_float(pick("r0_px", "r0", "zero_radius"), base.r0),
        r1=_as_float(pick("r1_px", "r1", "order_radius"), base.r1),
        side=_as_float(pick("side_px", "side", "target_side"), base.side),
        sep=_as_float(pick("separation_seed_px", "sep", "separation_px"), base.sep),
        tol=_as_float(
            pick("separation_search_px", "tol", "separation_tolerance_px"),
            base.tol,
        ),
        zero_exclusion=_as_float(
            pick("zero_order_exclusion_px", "zero_exclusion", "zero_exclusion_radius_px"),
            base.zero_exclusion,
        ),
        match_abs_d=_as_float(pick("match_abs_d"), base.match_abs_d),
        match_ci_high=_as_float(pick("match_ci95_high"), base.match_ci_high),
        match_sd=_as_float(pick("match_group_sd"), base.match_sd),
        blaze_leakage_expected=_as_float(
            pick("blaze_leakage_expected"), base.blaze_leakage_expected
        ),
        drift_px=_as_float(pick("drift_centroid_px"), base.drift_px),
        flat_drift_rel=_as_float(pick("flat_drift_relative"), base.flat_drift_rel),
        saturation_fraction=_as_float(
            pick("saturation_fraction"), base.saturation_fraction
        ),
        peak_snr_min=_as_float(pick("peak_snr_min"), base.peak_snr_min),
        roi_frozen_source=source,
        roi_frozen_audit=audit,
    )


# ---------------------------------------------------------------------------
# 输入读取
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    """一帧原始数据及其采集元信息。"""

    frame_id: str
    role: str
    pass_index: int
    path: Path
    timestamp: float
    exposure_ms: float
    saturated_count: int = 0
    phase_hash: str | None = None
    ordinal: int = 0
    image: np.ndarray | None = field(default=None, repr=False)
    meta: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """报告里的可读帧名。"""
        return f"{self.pass_index}:{self.role}#{self.ordinal}"


def read_json(path: Path) -> dict[str, Any]:
    """读 JSON；缺失文件返回空 dict 而不是抛错（采集端可能还没写全）。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def parse_timestamp(value: Any) -> float:
    """时间戳统一转 UTC 秒；数字原样接受，ISO 字符串解析。"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            return 0.0
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    return 0.0


def read_journal(path: Path) -> list[dict[str, Any]]:
    """逐行读 ``journal.jsonl``；坏行跳过但不静默。"""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"journal.jsonl 第 {line_no} 行不是合法 JSON，已跳过")
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _entry_frame_id(entry: dict[str, Any]) -> str:
    """从 journal 条目取出帧 id。"""
    for key in ("id", "frame_id", "phase_id", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _entry_saturated(entry: dict[str, Any]) -> int:
    """饱和计数：接受 int 计数或 bool 标记。"""
    for key in ("saturated_count", "saturated_pixels", "saturated"):
        value = entry.get(key)
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return 0


def _entry_exposure(entry: dict[str, Any], config: dict[str, Any]) -> float:
    """单帧曝光；缺失时回退到配置里的冻结曝光。"""
    for key in ("exposure_ms", "exposure", "exposure_ms_readback"):
        value = entry.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    for key in ("exposure_ms", "exposure"):
        value = config.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return 0.0


def _resolve_raw_path(
    raw_dir: Path, entry: dict[str, Any], frame_id: str
) -> Path | None:
    """把 journal 条目映射到 ``raw/*.npy``。

    采集端命名是确定性的 ``{frame_index:04d}_{frame_id}.npy``（同 id 的
    blaze/shaping 一遍出现两次，靠序号区分），所以按可靠度依次尝试：显式字段
    -> ``frame_index`` 拼出的名字 -> ``<id>.npy`` -> 含 id 的唯一文件 ->
    序号前缀兜底。
    """
    for key in ("raw_file", "file", "path", "npy"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            candidate = raw_dir / Path(value).name
            if candidate.is_file():
                return candidate
    index = entry.get("frame_index")
    if isinstance(index, str) and index.isdigit():
        index = int(index)
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        indexed = raw_dir / f"{index:04d}_{frame_id}.npy"
        if indexed.is_file():
            return indexed
    direct = raw_dir / f"{frame_id}.npy"
    if direct.is_file():
        return direct
    if frame_id:
        hits = sorted(raw_dir.glob(f"*{frame_id}*.npy"))
        if len(hits) == 1:
            return hits[0]
        prefixed = sorted(raw_dir.glob(f"[0-9][0-9][0-9][0-9]_{frame_id}.npy"))
        if prefixed:
            return prefixed[0]
    return None


def build_frames(
    root: Path, journal: list[dict[str, Any]], config: dict[str, Any]
) -> list[Frame]:
    """把 journal（或裸 ``raw/``）整理成按时间排序的帧列表。

    优先信 journal；若 journal 缺失，则按文件名排序后的顺序套用计划 §4 的槽位。
    """
    raw_dir = root / "raw"
    entries: list[dict[str, Any]] = list(journal)
    if not entries and raw_dir.is_dir():
        # 无 journal：只能按文件名顺序假设采集顺序与计划一致。
        entries = [{"id": p.stem} for p in sorted(raw_dir.glob("*.npy"))]

    # 判断第二遍是否用 ``_r2`` 后缀；否则用前后半段切分。
    has_suffix = any(_entry_frame_id(e).endswith("_r2") for e in entries)
    split = len(entries) // 2 if not has_suffix and len(entries) >= 16 else 0

    frames: list[Frame] = []
    for index, entry in enumerate(entries):
        raw_id = _entry_frame_id(entry)
        base_id = raw_id.removesuffix("_r2")
        role = ROLE_BY_ID.get(base_id)
        if role is None:
            logger.warning(f"未知帧 id {raw_id!r}（第 {index} 条），已跳过")
            continue
        path = _resolve_raw_path(raw_dir, entry, raw_id)
        if path is None and base_id != raw_id:
            # 共享暗帧库：``dark_r2`` 没有独立文件，回退到基础 id 再找一次。
            path = _resolve_raw_path(raw_dir, entry, base_id)
        if path is None:
            logger.warning(f"帧 {raw_id!r} 找不到对应的 raw/*.npy，已跳过")
            continue
        pass_index = 2 if (has_suffix and raw_id.endswith("_r2")) else 1
        if split and index >= split:
            pass_index = 2
        frames.append(
            Frame(
                frame_id=raw_id,
                role=role,
                pass_index=pass_index,
                path=path,
                timestamp=parse_timestamp(
                    entry.get("timestamp", entry.get("time", entry.get("utc")))
                ),
                exposure_ms=_entry_exposure(entry, config),
                saturated_count=_entry_saturated(entry),
                phase_hash=entry.get("phase_hash") or entry.get("hash"),
                ordinal=sum(
                    1
                    for f in frames
                    if f.pass_index == pass_index and f.role == role
                ),
            )
        )

    # 时间戳可能有缺省 0；保留相对顺序（Python 排序稳定）。
    frames.sort(key=lambda f: f.timestamp)
    return frames


def load_image(frame: Frame) -> np.ndarray:
    """按需读原始帧并转 float64（不缩放到 0-1，积分保持原生 counts）。"""
    if frame.image is None:
        frame.image = np.asarray(np.load(frame.path), dtype=np.float64)
    return frame.image


# ---------------------------------------------------------------------------
# 光度学
# ---------------------------------------------------------------------------
def robust_sigma(image: np.ndarray) -> float:
    """用 1.4826*MAD 估计背景噪声（对光斑这类重尾分布更稳健）。"""
    median = float(np.median(image))
    mad = float(np.median(np.abs(image - median)))
    return 1.4826 * mad


def disk_mask(
    shape: tuple[int, int], center: tuple[float, float], radius: float
) -> np.ndarray:
    """以 ``center`` 为心、半径 ``radius`` 的圆盘掩膜。"""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    dist2 = (rows - center[0]) ** 2 + (cols - center[1]) ** 2
    return dist2 <= radius * radius


def square_mask(
    shape: tuple[int, int], center: tuple[float, float], side: float
) -> np.ndarray:
    """以 ``center`` 为心、边长 ``side`` 的轴对齐正方形掩膜（冻结目标掩膜）。"""
    half = side / 2.0
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    in_rows = np.abs(rows - center[0]) <= half
    in_cols = np.abs(cols - center[1]) <= half
    return np.logical_and(in_rows, in_cols)


def integrate(image: np.ndarray, mask: np.ndarray) -> float:
    """掩膜内积分（输入已是暗减图）。"""
    return float(np.sum(image[mask]))


def global_peak(image: np.ndarray) -> tuple[int, int]:
    """全图最亮点索引。"""
    flat_index = int(np.argmax(image))
    return divmod(flat_index, image.shape[1])


def refine_centroid(
    image: np.ndarray, peak: tuple[int, int], half: int = 12
) -> tuple[float, float]:
    """在峰值附近做强度加权质心（扣除全图中位数背景），返回 ``(row, col)``。"""
    row, col = int(peak[0]), int(peak[1])
    r0, r1 = max(0, row - half), min(image.shape[0], row + half + 1)
    c0, c1 = max(0, col - half), min(image.shape[1], col + half + 1)
    window = image[r0:r1, c0:c1]
    background = float(np.median(image))
    weight = np.clip(window - background, 0.0, None)
    total = float(weight.sum())
    if total <= 0.0:
        return float(row), float(col)
    rows = np.arange(r0, r1, dtype=np.float64)[:, None]
    cols = np.arange(c0, c1, dtype=np.float64)[None, :]
    return (
        float((weight * rows).sum() / total),
        float((weight * cols).sum() / total),
    )


def peak_snr(image: np.ndarray, peak: tuple[int, int]) -> float:
    """峰值相对背景噪声的显著度（sigma 倍数）。"""
    sigma = robust_sigma(image)
    value = float(image[peak])
    if sigma <= 0.0:
        return float("inf") if value > 0.0 else 0.0
    return value / sigma


def _ring_candidates(
    image: np.ndarray,
    center: tuple[float, float],
    sep: float,
    tol: float,
    exclude: float,
    min_distance: float = 20.0,
    max_candidates: int = 400,
) -> list[tuple[float, int, int]]:
    """环带内的局部极大候选，返回 ``[(value, row, col), ...]`` 按强度降序。

    噪声/常数帧会在环带内产生大量并列局部极大（每个都过 ``maximum_filter``），
    纯 Python 两两距离比较是平方复杂度，能挂住整个分析。因此：

    - 只保留强度排前 ``max_candidates`` 的候选（找 \xb11 级峰只需要前几名）；
    - 距离比较用 ``math.hypot`` 标量运算并提前退出。
    """
    rows, cols = np.ogrid[: image.shape[0], : image.shape[1]]
    dist = np.hypot(rows - center[0], cols - center[1])
    ring = (dist >= sep - tol) & (dist <= sep + tol) & (dist > exclude)
    if not bool(ring.any()):
        return []
    masked = np.where(ring, image, -np.inf)
    local_max = maximum_filter(masked, size=9, mode="constant", cval=-np.inf)
    candidates = np.argwhere((masked > -np.inf) & (masked == local_max))
    ordered = sorted(
        ((float(image[r, c]), int(r), int(c)) for r, c in candidates),
        reverse=True,
    )[:max_candidates]
    picked: list[tuple[float, int, int]] = []
    for value, row, col in ordered:
        if value <= 0.0:
            continue
        ok = True
        for _, pr, pc in picked:
            if math.hypot(row - pr, col - pc) < min_distance:
                ok = False
                break
        if ok:
            picked.append((value, row, col))
    return picked


def find_order_peaks(
    image: np.ndarray,
    center: tuple[float, float],
    thresholds: Thresholds,
) -> tuple[tuple[float, int, int] | None, tuple[float, int, int] | None]:
    """找 +1 与 -1 级峰：最强峰为 +1，其反方向最强峰为 -1。

    反方向判据是方向余弦 ``<= -0.5``（夹角 >= 120 度）；若反方向没有候选，
    就退回除 +1 外最强的候选（此时 ±1 级差只作为 blaze 泄漏诊断报告，
    不触发不可判定）。
    """
    candidates = _ring_candidates(
        image, center, thresholds.sep, thresholds.tol, thresholds.zero_exclusion
    )
    if not candidates:
        return None, None
    plus = candidates[0]
    pr, pc = float(plus[1]) - center[0], float(plus[2]) - center[1]
    norm = math.hypot(pr, pc)
    minus: tuple[float, int, int] | None = None
    if norm > 0.0:
        ux, uy = pr / norm, pc / norm
        for candidate in candidates[1:]:
            ox = float(candidate[1]) - center[0]
            oy = float(candidate[2]) - center[1]
            other_norm = math.hypot(ox, oy)
            if other_norm <= 0.0:
                continue
            if (ux * ox + uy * oy) / other_norm <= -0.5:
                minus = candidate
                break
    if minus is None and len(candidates) > 1:
        minus = candidates[1]
    return plus, minus


def resolve_dark(
    root: Path, frames: list[Frame], config: dict[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """解析同曝光暗帧：采集目录内的暗帧优先，其次配置指向的 provided 暗帧。

    都没有时返回 ``(None, info)``，调用方按预注册直接判 undeterminable。
    """
    info: dict[str, Any] = {
        "source": "missing",
        "frames": [],
        "exposure_ms": None,
        "note": "",
    }
    dark_source = str(config.get("dark_source", "capture")).lower()

    dark_frames = [f for f in frames if f.role == "dark"]
    # 共享暗帧库：两个 dark 槽位可能指向同一份文件（如 dark / dark_r2 共享
    # dark_library_0.04ms.npy）。按路径去重后才是真实独立采样数；同一份文件
    # 取均值与取一次等价，所以口径不变，只是报告的「张数」如实。
    unique_darks: dict[str, Frame] = {}
    for frame in dark_frames:
        unique_darks.setdefault(str(frame.path), frame)
    dark_frames = list(unique_darks.values())
    if dark_frames:
        exposures = [f.exposure_ms for f in dark_frames]
        median_exposure = float(np.median(exposures)) if exposures else 0.0
        same = [
            f
            for f in dark_frames
            if median_exposure <= 0.0
            or abs(f.exposure_ms - median_exposure) <= 0.1 * median_exposure
        ]
        if same:
            stack = np.stack([load_image(f) for f in same], axis=0)
            info.update(
                source="capture",
                frames=[f.path.name for f in same],
                exposure_ms=median_exposure,
                note=f"{len(same)}/{len(dark_frames)} 张暗帧曝光一致",
            )
            return np.mean(stack, axis=0), info
        info["note"] = "采集目录里的暗帧曝光与主体不一致"

    provided = config.get("provided_dark") or config.get("dark_path")
    if isinstance(provided, str) and provided:
        candidate = Path(provided)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            info.update(
                source="provided",
                frames=[candidate.name],
                exposure_ms=_as_float(config.get("exposure_ms"), 0.0),
                note="使用配置提供的暗帧文件",
            )
            return np.asarray(np.load(candidate), dtype=np.float64), info

    if dark_source not in ("missing", ""):
        info["note"] = f"dark_source={dark_source!r} 但未找到可用暗帧"
    return None, info


# ---------------------------------------------------------------------------
# 漂移
# ---------------------------------------------------------------------------
def drift_report(flats: list[Frame], thresholds: Thresholds) -> dict[str, Any]:
    """相邻 flat 的两项检查：质心位移（>2 px 只记告警）与 F0 相对变化。

    - 质心位移超阈值 -> ``drift_warnings``：不作废，只进 verdict 说明；
    - F0 相对变化超 5% -> 漂移轮：>1 轮才判不可判定（计划 §4/§6）。
    """
    shifts: list[float] = []
    rel_changes: list[float] = []
    warnings: list[str] = []
    drift_labels: list[str] = []
    details: list[dict[str, Any]] = []
    for left, right in zip(flats, flats[1:]):
        center_left = left.meta["center"]
        center_right = right.meta["center"]
        shift = float(
            np.hypot(
                center_right[0] - center_left[0], center_right[1] - center_left[1]
            )
        )
        f0_left = float(left.meta["f0"])
        f0_right = float(right.meta["f0"])
        denom = max(abs(f0_left), abs(f0_right), 1e-12)
        rel = abs(f0_right - f0_left) / denom
        shifts.append(shift)
        rel_changes.append(rel)
        if shift > thresholds.drift_px:
            warnings.append(
                f"{left.label()} -> {right.label()}: 质心位移 {shift:.2f} px"
            )
        if rel > thresholds.flat_drift_rel:
            drift_labels.extend([left.label(), right.label()])
            details.append(
                {
                    "between": [left.label(), right.label()],
                    "f0_relative_change": rel,
                }
            )
    return {
        "centroid_shifts_px": shifts,
        "f0_relative_changes": rel_changes,
        "drift_warnings": warnings,
        "drift_rounds": len(details),
        "drift_flat_labels": sorted(set(drift_labels)),
        "details": details,
    }


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """安全除法；分母非正或缺失返回 None。"""
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)


def build_rounds(
    blazes: list[Frame], shapings: list[Frame], f0_mean: float, s0_mean: float
) -> list[dict[str, Any]]:
    """按 pass 分组，把同一遍里时间相邻的 blaze/shaping 配成重复轮。"""
    rounds: list[dict[str, Any]] = []
    pass_ids = sorted(
        {f.pass_index for f in blazes} | {f.pass_index for f in shapings}
    )
    for pass_index in pass_ids:
        pass_blazes = sorted(
            (f for f in blazes if f.pass_index == pass_index),
            key=lambda f: f.timestamp,
        )
        pass_shapings = sorted(
            (f for f in shapings if f.pass_index == pass_index),
            key=lambda f: f.timestamp,
        )
        for blaze, shaping in zip(pass_blazes, pass_shapings):
            eta_d = _ratio(blaze.meta.get("g1"), f0_mean)
            eta_s = _ratio(
                float(shaping.meta.get("s", 0.0)) - s0_mean, f0_mean
            )
            rounds.append(
                {
                    "pass": pass_index,
                    "blaze_frame": blaze.label(),
                    "shaping_frame": shaping.label(),
                    "eta_d": eta_d,
                    "eta_s": eta_s,
                    "d": None
                    if eta_d is None or eta_s is None
                    else eta_s - eta_d,
                }
            )
    return rounds


def bootstrap_difference(
    rounds: list[dict[str, Any]], iterations: int, seed: int
) -> dict[str, Any]:
    """对重复轮的 ``d`` 做固定种子 bootstrap，报均值与 95% 区间。"""
    diffs = np.asarray(
        [r["d"] for r in rounds if r.get("d") is not None and np.isfinite(r["d"])],
        dtype=np.float64,
    )
    out: dict[str, Any] = {
        "iterations": int(iterations),
        "seed": int(seed),
        "n_rounds": int(diffs.size),
        "mean_d": None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if diffs.size == 0:
        return out
    out["mean_d"] = float(diffs.mean())
    if diffs.size == 1:
        # 单点无法给区间；明确写成点值而不是伪造 CI。
        out["ci95_low"] = float(diffs[0])
        out["ci95_high"] = float(diffs[0])
        out["degenerate"] = "single_round_ci_is_point"
        return out
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diffs.size, size=(int(iterations), diffs.size))
    samples = diffs[indices].mean(axis=1)
    out["ci95_low"] = float(np.percentile(samples, 2.5))
    out["ci95_high"] = float(np.percentile(samples, 97.5))
    return out


def within_pass_sd(
    rounds: list[dict[str, Any]], key: str
) -> tuple[float | None, str]:
    """两效率各自组内标准差。

    每个 pass 内有 >=2 个重复轮时取 pass 内样本标准差的最大值（``max_within_pass``）；
    否则退回全部重复轮的全局标准差（``global_rounds``）；不足 2 个则返回 None。
    """
    by_pass: dict[int, list[float]] = {}
    for row in rounds:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            by_pass.setdefault(row["pass"], []).append(float(value))
    sds = [
        float(np.std(values, ddof=1))
        for values in by_pass.values()
        if len(values) >= 2
    ]
    if sds:
        return max(sds), "max_within_pass"
    merged = [v for values in by_pass.values() for v in values]
    if len(merged) >= 2:
        return float(np.std(merged, ddof=1)), "global_rounds"
    return None, "insufficient"


def _check_hashes(
    root: Path, frames: list[Frame], config: dict[str, Any]
) -> dict[str, Any]:
    """相位哈希/设备身份核对；只在双方都存在且不一致时算失败。"""
    checks: dict[str, Any] = {
        "checked": False,
        "mismatches": [],
        "available": [],
    }
    hashes_path = root / "phases" / "hashes.json"
    expected: dict[str, str] = {}
    for key, value in read_json(hashes_path).items():
        if isinstance(value, str):
            expected[key] = value
        elif isinstance(value, dict):
            digest = value.get("sha256") or value.get("hash")
            if isinstance(digest, str):
                expected[key] = digest
    from_config = config.get("phase_hashes")
    if isinstance(from_config, dict):
        expected.update(
            {k: v for k, v in from_config.items() if isinstance(v, str)}
        )
    if expected:
        checks["available"].append("phases/hashes.json")
    for frame in frames:
        digest = frame.phase_hash
        if not digest:
            continue
        checks["checked"] = True
        keys = (frame.frame_id, frame.frame_id.removesuffix("_r2"), frame.role)
        wanted = next((expected[k] for k in keys if k in expected), None)
        if wanted and wanted != digest:
            checks["mismatches"].append(
                {
                    "frame": frame.frame_id,
                    "expected": wanted,
                    "journal": digest,
                }
            )
    config_device = {
        key: config.get(key)
        for key in ("slm_serial", "ccd_serial", "device_id", "slm_id")
        if config.get(key) is not None
    }
    journal_device = config.get("journal_device")
    if isinstance(journal_device, dict) and config_device:
        checks["checked"] = True
        for key, value in config_device.items():
            other = journal_device.get(key)
            if other is not None and other != value:
                checks["mismatches"].append(
                    {"device_field": key, "expected": value, "journal": other}
                )
    return checks


def compare_verdict(
    result: dict[str, Any], thresholds: Thresholds
) -> tuple[str, str, list[str]]:
    """按计划 §6 给出 ``(verdict, reason, reasons)``；不可判定门优先。"""
    gates = result["gates"]
    reasons = [
        reason for key, reason in UNDETERMINABLE_GATES if gates.get(key)
    ]
    if reasons:
        return "undeterminable", reasons[0], reasons

    stats = result["stats"]
    mean_d = stats["mean_d"]
    ci_high = stats["ci95_high"]
    if mean_d is None or ci_high is None:
        return "undeterminable", "insufficient_repeats", ["insufficient_repeats"]
    if abs(mean_d) > thresholds.match_abs_d or ci_high > thresholds.match_ci_high:
        why = []
        if abs(mean_d) > thresholds.match_abs_d:
            why.append(
                f"abs(mean_d)={abs(mean_d):.3f} > {thresholds.match_abs_d}"
            )
        if ci_high > thresholds.match_ci_high:
            why.append(
                f"ci95_high={ci_high:.3f} > {thresholds.match_ci_high}"
            )
        return "mismatched", "; ".join(why), why
    return "matched", "preregistered_bounds_all_met", []


# ---------------------------------------------------------------------------
# 主分析
# ---------------------------------------------------------------------------
def _interp_center(
    times: list[float], centers: list[tuple[float, float]], stamp: float
) -> tuple[float, float]:
    """按时间戳线性插值参考 0 级质心；外推时夹到端点。"""
    if not centers:
        raise ValueError("没有 flat 帧可作参考")
    if len(centers) == 1:
        return centers[0]
    order = np.argsort(times)
    ts = np.asarray(times)[order]
    rs = np.asarray([c[0] for c in centers])[order]
    cs = np.asarray([c[1] for c in centers])[order]
    return float(np.interp(stamp, ts, rs)), float(np.interp(stamp, ts, cs))


def analyze_dir(
    root: Path,
    thresholds: Thresholds | None = None,
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """分析一个采集目录，返回可直接写盘的 ``analysis.json`` 结构。"""
    root = Path(root)
    config = read_json(root / "config.json")
    th = thresholds or thresholds_from_config(config)
    full_scale = _as_float(
        config.get("camera_full_scale", config.get("full_scale", 4095.0)),
        4095.0,
    )
    journal = read_journal(root / "journal.jsonl")
    frames = build_frames(root, journal, config)

    gates: dict[str, bool] = {key: False for key, _ in UNDETERMINABLE_GATES}
    gates["group_sd"] = False

    dark, dark_info = resolve_dark(root, frames, config)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_dir": str(root.resolve()),
        "config": {
            "exposure_ms": _as_float(config.get("exposure_ms"), 0.0),
            "panel_shape": config.get("panel_shape"),
            "camera_full_scale": full_scale,
            "dark_source": config.get("dark_source", "capture"),
            "matrix_interpretation": config.get("matrix_interpretation"),
        },
        "thresholds": th.snapshot(),
        "bootstrap": {"iterations": int(bootstrap), "seed": int(seed)},
        "dark": dark_info,
        "frames": [],
        "passes": [],
        "stats": {
            "mean_d": None,
            "ci95_low": None,
            "ci95_high": None,
            "sd_eta_s": None,
            "sd_eta_s_basis": None,
            "sd_eta_d": None,
            "sd_eta_d_basis": None,
            "rounds": [],
        },
        "photometry": {
            "f0_mean": None,
            "f0_sd": None,
            "s0_mean": None,
            "s0_sd": None,
            "g0_mean": None,
        },
        "drift": {
            "drift_rounds": 0,
            "drift_warnings": [],
            "drift_flat_labels": [],
        },
        "hashes": {"checked": False, "mismatches": []},
        "gates": gates,
        "verdict": "undeterminable",
        "reason": "",
        "reasons": [],
    }

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        """收尾：判 verdict 并返回。"""
        verdict, reason, reasons = compare_verdict(result, th)
        result["verdict"] = verdict
        result["reason"] = reason
        result["reasons"] = reasons
        return result

    if dark is None:
        gates["missing_same_exposure_dark"] = True
        result["reason"] = "missing_same_exposure_dark"
        result["reasons"] = ["missing_same_exposure_dark"]
        logger.warning("没有同曝光暗帧，按预注册直接判 undeterminable")
        return result

    hashes = _check_hashes(root, frames, config)
    result["hashes"] = hashes
    if hashes["mismatches"]:
        gates["hash_or_device_mismatch"] = True

    flats = sorted(
        (f for f in frames if f.role in FLAT_ROLES), key=lambda f: f.timestamp
    )
    blazes = sorted(
        (f for f in frames if f.role == "blaze"), key=lambda f: f.timestamp
    )
    shapings = sorted(
        (f for f in frames if f.role == "shaping"), key=lambda f: f.timestamp
    )
    if not flats:
        gates["no_flat_frame"] = True
        result["reason"] = "no_flat_frame"
        result["reasons"] = ["no_flat_frame"]
        return result

    # 暗减后的图统一成负值截零的 counts（暗帧是偏置，不是物理能量）。
    for frame in frames:
        frame.image = np.clip(load_image(frame) - dark, 0.0, None)

    # 逐 flat 帧：质心（peak + 质心细化）、F0、S0、峰显著度。
    times: list[float] = []
    centers: list[tuple[float, float]] = []
    for flat in flats:
        image = load_image(flat)
        peak = global_peak(image)
        center = refine_centroid(image, peak)
        flat.meta = {
            "center": center,
            "f0": integrate(image, disk_mask(image.shape, center, th.r0)),
            "s0": integrate(image, square_mask(image.shape, center, th.side)),
            "snr": peak_snr(image, peak),
        }
        times.append(flat.timestamp)
        centers.append(center)

    drift = drift_report(flats, th)
    result["drift"] = drift
    if drift["drift_rounds"] > 1:
        gates["drift_rounds"] = True

    # 计划 §4：漂移轮的全平参考不参与平均；全部被标记时退回全部帧。
    stable = [
        f for f in flats if f.label() not in set(drift["drift_flat_labels"])
    ] or flats
    f0_values = [float(f.meta["f0"]) for f in stable]
    s0_values = [float(f.meta["s0"]) for f in stable]
    f0_mean = float(np.mean(f0_values))
    f0_sd = float(np.std(f0_values, ddof=1)) if len(f0_values) > 1 else 0.0
    s0_mean = float(np.mean(s0_values))
    s0_sd = float(np.std(s0_values, ddof=1)) if len(s0_values) > 1 else 0.0
    result["photometry"].update(
        f0_mean=f0_mean,
        f0_sd=f0_sd,
        s0_mean=s0_mean,
        s0_sd=s0_sd,
    )

    # 峰不显著才不可定位；snr=inf（背景噪声为零的完美扣除）视为显著。
    if f0_mean <= 0.0 or any(
        float(f.meta["snr"]) < th.peak_snr_min for f in flats
    ):
        gates["f0_not_locatable"] = True

    # 逐帧光度学与饱和检查。
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        image = load_image(frame)
        max_pixel = float(image.max()) if image.size else 0.0
        row: dict[str, Any] = {
            "id": frame.frame_id,
            "role": frame.role,
            "pass": frame.pass_index,
            "ordinal": frame.ordinal,
            "file": frame.path.name,
            "timestamp": frame.timestamp,
            "exposure_ms": frame.exposure_ms,
            "saturated_count": frame.saturated_count,
            "max_pixel": max_pixel,
            "saturated": frame.saturated_count > 0
            or max_pixel >= th.saturation_fraction * full_scale,
        }
        if frame.role in FLAT_ROLES:
            row.update(
                center=frame.meta["center"],
                f0=frame.meta["f0"],
                s0=frame.meta["s0"],
                snr=frame.meta["snr"],
            )
        else:
            center = _interp_center(times, centers, frame.timestamp)
            row["reference_center"] = center
            if frame.role == "blaze":
                plus, minus = find_order_peaks(image, center, th)
                if plus is None:
                    gates["plus1_not_locatable"] = True
                g1 = (
                    integrate(
                        image, disk_mask(image.shape, (plus[1], plus[2]), th.r1)
                    )
                    if plus
                    else None
                )
                gm1 = (
                    integrate(
                        image, disk_mask(image.shape, (minus[1], minus[2]), th.r1)
                    )
                    if minus
                    else None
                )
                g0 = integrate(image, disk_mask(image.shape, center, th.r0))
                frame.meta = {
                    "center": center,
                    "g0": g0,
                    "g1": g1,
                    "gm1": gm1,
                }
                row.update(
                    plus1_peak=None if plus is None else [plus[1], plus[2]],
                    minus1_peak=None if minus is None else [minus[1], minus[2]],
                    g0=g0,
                    g1=g1,
                    gm1=gm1,
                )
            elif frame.role == "shaping":
                s_value = integrate(
                    image, square_mask(image.shape, center, th.side)
                )
                frame.meta = {"center": center, "s": s_value}
                row["s"] = s_value
        if row["saturated"] and frame.role != "dark":
            gates["saturated_frame"] = True
        frame_rows.append(row)

    # 每帧效率（原始 S/F0 仅对照，不参与判定）。
    for frame, row in zip(frames, frame_rows):
        if frame.role == "blaze":
            row["eta_d"] = _ratio(frame.meta.get("g1"), f0_mean)
        elif frame.role == "shaping":
            s_value = float(frame.meta.get("s", 0.0))
            row["eta_s"] = _ratio(s_value - s0_mean, f0_mean)
            row["raw_s_over_f0"] = _ratio(s_value, f0_mean)
        elif frame.role in FLAT_ROLES:
            row["raw_s_over_f0"] = _ratio(float(frame.meta["f0"]), f0_mean)

    result["frames"] = frame_rows

    g0_values = [float(f.meta["g0"]) for f in blazes if "g0" in f.meta]
    result["photometry"]["g0_mean"] = (
        float(np.mean(g0_values)) if g0_values else None
    )

    # 效率越界（<0 或 >1）：先判 undeterminable，不做物理解释。
    etas = [
        row[key]
        for row in frame_rows
        for key in ("eta_s", "eta_d")
        if row.get(key) is not None
    ]
    if any(value < 0.0 or value > 1.0 for value in etas):
        gates["energy_out_of_range"] = True

    # ±1 对称性：仅作 blaze 保真度诊断，不再触发不可判定。
    # 修订依据（2026-11-23，采集前登记）：单连续 blaze ramp 的 −1 级只是泄漏底，
    # 不是共轭级次——模型核算 G1/G−1 = 3484:1（相对差 0.9997），2026-11-22 笔记实测
    # 同结论（闪耀两侧天然不对称，不得要求平分能量）。原 20% 门与 blaze 物理冲突，
    # 良好 blaze 数据必然触发；经用户批准改为 diagnostic，不作废窗口。
    # 期望 G−1/G1 <= 0.05（泄漏底量级）；超预期仅标注，不判 undeterminable。
    # 作废改由：+1 级不可定位、饱和、漂移、缺暗帧、哈希/设备不符等门承担。
    for frame, row in zip(frames, frame_rows):
        if frame.role != "blaze":
            continue
        g1 = frame.meta.get("g1")
        gm1 = frame.meta.get("gm1")
        if g1 is None or gm1 is None:
            row["blaze_leakage_diagnostic"] = None
            continue
        g1v, gm1v = abs(float(g1)), abs(float(gm1))
        row["blaze_leakage_diagnostic"] = {
            "gm1_over_g1": gm1v / max(g1v, 1e-12),
            "asymmetry_abs_diff_over_max": abs(g1v - gm1v) / max(g1v, gm1v, 1e-12),
            "expected_gm1_over_g1_max": th.blaze_leakage_expected,
            "within_expectation": (gm1v / max(g1v, 1e-12)) <= th.blaze_leakage_expected,
        }

    rounds = build_rounds(blazes, shapings, f0_mean, s0_mean)
    result["stats"]["rounds"] = rounds
    if len([r for r in rounds if r.get("d") is not None]) < 2:
        gates["insufficient_repeats"] = True

    boot = bootstrap_difference(rounds, bootstrap, seed)
    result["stats"].update(
        mean_d=boot["mean_d"],
        ci95_low=boot["ci95_low"],
        ci95_high=boot["ci95_high"],
    )

    sd_s, basis_s = within_pass_sd(rounds, "eta_s")
    sd_d, basis_d = within_pass_sd(rounds, "eta_d")
    result["stats"]["sd_eta_s"] = sd_s
    result["stats"]["sd_eta_s_basis"] = basis_s
    result["stats"]["sd_eta_d"] = sd_d
    result["stats"]["sd_eta_d_basis"] = basis_d
    if (sd_s is not None and sd_s > th.match_sd) or (
        sd_d is not None and sd_d > th.match_sd
    ):
        gates["group_sd"] = True

    passes: list[dict[str, Any]] = []
    for pass_index in sorted({r["pass"] for r in rounds}):
        subset = [r for r in rounds if r["pass"] == pass_index]
        passes.append(
            {
                "pass": pass_index,
                "eta_s": [r["eta_s"] for r in subset],
                "eta_d": [r["eta_d"] for r in subset],
                "d": [r["d"] for r in subset],
            }
        )
    result["passes"] = passes

    result = finish(result)
    # 质心漂移只作告警，不自动作废，但必须进 verdict 说明。
    if drift["drift_warnings"] and result["verdict"] != "undeterminable":
        result["reason"] += (
            f"; drift_warning={len(drift['drift_warnings'])}（见 drift.drift_warnings）"
        )
    return result


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------
_fmt = partial(fmt_fixed_or_text, nonfinite_text="inf")


def render_report(result: dict[str, Any]) -> str:
    """生成中文 ``EM_REPORT.md``。"""
    stats = result["stats"]
    photo = result["photometry"]
    th = result["thresholds"]
    gates = result["gates"]
    drift = result["drift"]
    lines: list[str] = []
    lines.append("# 整形效率与衍射效率匹配性分析报告")
    lines.append("")
    lines.append(f"- 输入目录：`{result['input_dir']}`")
    lines.append(f"- 生成时间（UTC）：{result['created_utc']}")
    lines.append(f"- 口径版本：`{result['schema']}`")
    lines.append("")
    lines.append("## 1. 摘要")
    lines.append("")
    lines.append(f"- 判定：**{result['verdict']}**")
    lines.append(f"- 理由：`{result['reason']}`")
    lines.append(
        f"- mean(d) = {_fmt(stats['mean_d'])}，"
        f"CI95 = [{_fmt(stats['ci95_low'])}, {_fmt(stats['ci95_high'])}]"
    )
    lines.append(
        f"- eta_s 组内 sd = {_fmt(stats['sd_eta_s'])}"
        f"（{_fmt(stats.get('sd_eta_s_basis'))}），"
        f"eta_d 组内 sd = {_fmt(stats['sd_eta_d'])}"
        f"（{_fmt(stats.get('sd_eta_d_basis'))}）"
    )
    lines.append(
        f"- F0 = {_fmt(photo['f0_mean'], 1)} ± {_fmt(photo['f0_sd'], 1)}，"
        f"S0 = {_fmt(photo['s0_mean'], 1)} ± {_fmt(photo['s0_sd'], 1)}"
    )
    lines.append("")
    lines.append("## 2. 口径")
    lines.append("")
    lines.append("- `eta_s = (S - S0) / F0`，`eta_d = G1 / F0`；分母保留完整 F0。")
    lines.append(
        "- F0：全平帧在 R0 圆盘内的暗减积分；S0：全平帧在冻结方窗内的暗减积分均值。"
    )
    lines.append(
        "- G1：环带内最强峰附近 R1 圆盘内的暗减积分；G-1 只作 blaze 泄漏诊断（只报告，不门控）。"
    )
    lines.append(
        "- 所有帧先扣同曝光暗帧；原始 `S/F0` 仅作对照，不参与判定。"
    )
    lines.append(f"- 暗帧来源：`{result['dark']['source']}`（{result['dark']['note']}）")
    lines.append("")
    lines.append("### 2.1 阈值快照")
    lines.append("")
    lines.append("| 口径 | 值 |")
    lines.append("|---|---:|")
    for key, value in th.items():
        value_text = (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, dict)
            else _fmt(value)
        )
        lines.append(f"| `{key}` | {value_text} |")
    lines.append("")
    lines.append("## 3. 逐帧表")
    lines.append("")
    lines.append(
        "| 遍 | 帧 | 角色 | 曝光 ms | 饱和 | F0 / S | G0 | G1 | G-1 | "
        "eta_s | eta_d | S/F0 |"
    )
    lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["frames"]:
        lines.append(
            f"| {row['pass']} | `{row['id']}` | {row['role']} | "
            f"{_fmt(row.get('exposure_ms'), 3)} | "
            f"{'是' if row.get('saturated') else '否'} | "
            f"{_fmt(row.get('f0', row.get('s')), 1)} | "
            f"{_fmt(row.get('g0'), 1)} | {_fmt(row.get('g1'), 1)} | "
            f"{_fmt(row.get('gm1'), 1)} | {_fmt(row.get('eta_s'))} | "
            f"{_fmt(row.get('eta_d'))} | {_fmt(row.get('raw_s_over_f0'))} |"
        )
    lines.append("")
    lines.append("## 4. 逐 pass 重复表")
    lines.append("")
    lines.append("| pass | eta_s | eta_d | d |")
    lines.append("|---:|---|---|---|")
    for row in result["passes"]:
        lines.append(
            f"| {row['pass']} | "
            + ", ".join(_fmt(v) for v in row["eta_s"])
            + " | "
            + ", ".join(_fmt(v) for v in row["eta_d"])
            + " | "
            + ", ".join(_fmt(v) for v in row["d"])
            + " |"
        )
    lines.append("")
    lines.append(
        f"bootstrap：{result['bootstrap']['iterations']} 次，随机种子 "
        f"{result['bootstrap']['seed']}（结果可复现）。"
    )
    lines.append("")
    lines.append("## 5. 判据表")
    lines.append("")
    lines.append("| 判据 | 阈值 | 实测 | 结果 |")
    lines.append("|---|---|---|---|")
    mean_d = stats["mean_d"]
    ci_high = stats["ci95_high"]
    lines.append(
        f"| abs(mean(d)) | <= {th['match_abs_d']} | "
        f"{_fmt(None if mean_d is None else abs(mean_d))} | "
        f"{'通过' if mean_d is not None and abs(mean_d) <= th['match_abs_d'] else '未通过'} |"
    )
    lines.append(
        f"| CI95 上限 | <= {th['match_ci95_high']} | {_fmt(ci_high)} | "
        f"{'通过' if ci_high is not None and ci_high <= th['match_ci95_high'] else '未通过'} |"
    )
    sd_s = stats["sd_eta_s"]
    sd_d = stats["sd_eta_d"]
    lines.append(
        f"| eta_s 组内 sd | <= {th['match_group_sd']} | {_fmt(sd_s)} | "
        f"{'通过' if sd_s is None or sd_s <= th['match_group_sd'] else '未通过'} |"
    )
    lines.append(
        f"| eta_d 组内 sd | <= {th['match_group_sd']} | {_fmt(sd_d)} | "
        f"{'通过' if sd_d is None or sd_d <= th['match_group_sd'] else '未通过'} |"
    )
    blaze_diag = [
        row["blaze_leakage_diagnostic"]
        for row in result["frames"]
        if row.get("blaze_leakage_diagnostic")
    ]
    leak_max = _fmt(max((d["gm1_over_g1"] for d in blaze_diag), default=None))
    lines.append(
        f"| blaze 泄漏诊断（只报告，不门控） | gm1/g1 <= {th['blaze_leakage_expected']} | "
        f"{leak_max} | 诊断 |"
    )
    lines.append(
        f"| +1 级可定位 | 必须定位 | "
        f"{'缺失' if gates['plus1_not_locatable'] else '已定位'} | "
        f"{'不可判定' if gates['plus1_not_locatable'] else '通过'} |"
    )
    lines.append(
        f"| G1 两轮组内离散 | <= {th['match_group_sd']} | "
        f"{'超限' if gates['group_sd'] else '未超限'} | "
        f"{'不可判定' if gates['group_sd'] else '通过'} |"
    )
    lines.append(
        f"| 漂移轮 | <= 1 | {drift['drift_rounds']} | "
        f"{'不可判定' if gates['drift_rounds'] else '通过'} |"
    )
    lines.append(
        f"| 同曝光暗帧 | 必须存在 | "
        f"{'有' if not gates['missing_same_exposure_dark'] else '缺'} | "
        f"{'不可判定' if gates['missing_same_exposure_dark'] else '通过'} |"
    )
    lines.append(
        f"| 积分帧饱和 | 不得饱和 | "
        f"{'有饱和' if gates['saturated_frame'] else '未饱和'} | "
        f"{'不可判定' if gates['saturated_frame'] else '通过'} |"
    )
    lines.append(
        f"| 效率越界 | [0, 1] | "
        f"{'越界' if gates['energy_out_of_range'] else '在界内'} | "
        f"{'不可判定' if gates['energy_out_of_range'] else '通过'} |"
    )
    lines.append("")
    if drift["drift_warnings"]:
        lines.append("### 5.1 漂移告警（不作废，仅说明）")
        lines.append("")
        for warning in drift["drift_warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    lines.append("## 6. 结论")
    lines.append("")
    if result["verdict"] == "matched":
        lines.append(
            "在本比较式与冻结掩膜下，整形利用效率与衍射效率**匹配**："
            "差值均值、95% 区间与组内离散度都在预注册界内。"
        )
    elif result["verdict"] == "mismatched":
        lines.append(
            "在本比较式与冻结掩膜下，整形利用效率与衍射效率**不匹配**。"
            "该结论只针对本比较式，不能直接归因为 SLM 物理效率不足。"
        )
    else:
        lines.append(
            "本次**不可判定**（`"
            + result["reason"]
            + "`）。触发原因："
            + "、".join(f"`{r}`" for r in (result["reasons"] or ["—"]))
            + "。"
        )
    if drift["drift_warnings"]:
        lines.append("")
        lines.append(
            f"另有质心漂移告警 {len(drift['drift_warnings'])} 条"
            "（相邻 flat 质心位移 > 2 px），不影响判定但已在 verdict 说明。"
        )
    lines.append("")
    lines.append("本次分析**不改标定、不解禁任何门控**。")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    root: Path, result: dict[str, Any], overwrite: bool
) -> tuple[Path, Path]:
    """写 ``analysis.json`` 与 ``EM_REPORT.md``；默认拒绝覆盖。"""
    root = Path(root)
    analysis_path = root / ANALYSIS_NAME
    report_path = root / REPORT_NAME
    if not overwrite:
        existing = [p.name for p in (analysis_path, report_path) if p.exists()]
        if existing:
            raise FileExistsError(
                f"{', '.join(existing)} 已存在；确认要覆盖请加 --overwrite"
            )
    with analysis_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    report_path.write_text(
        render_report(result), encoding="utf-8", newline="\n"
    )
    return analysis_path, report_path


# ---------------------------------------------------------------------------
# 合成自检
# ---------------------------------------------------------------------------
SCENARIOS = ("matched", "mismatched", "asymmetric", "missing_dark")


def _gaussian(
    shape: tuple[int, int],
    center: tuple[float, float],
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    """画一个二维高斯光斑。"""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    dist2 = (rows - center[0]) ** 2 + (cols - center[1]) ** 2
    return amplitude * np.exp(-dist2 / (2.0 * sigma * sigma))


def _canvas(
    shape: tuple[int, int], background: float, rng: np.random.Generator
) -> np.ndarray:
    """带高斯读出噪声的暗底画布。"""
    noise = rng.normal(0.0, 3.0, size=shape)
    return np.clip(background + noise, 0.0, None)


def build_synthetic_dataset(
    root: Path,
    scenario: str,
    shape: tuple[int, int] = (1024, 1024),
    seed: int = 20261123,
) -> dict[str, Any]:
    """构造一个满足输入契约的合成采集目录，返回真值供断言。

    帧序列严格按计划 §4 的 8 槽位顺序，两遍共 16 帧（missing_dark 少 2 帧），
    第二遍帧 id 带 ``_r2``。journal 写 ``raw_file`` 以区分同 id 的重复槽位。
    """
    root = Path(root)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "phases").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # 半像素中心：方窗/圆盘掩膜的边界不压在整数像元上，真值与分析口径一致。
    center = (float(shape[0] / 2 - 0.5), float(shape[1] / 2 - 0.5))
    background = 40.0
    full_scale = 4095.0
    sep = 471.0
    side = 60.0
    r0 = 90.0
    r1 = 60.0
    sigma_zero = 18.0
    sigma_order = 14.0

    dark = _canvas(shape, background, rng)
    zero_spot = _gaussian(shape, center, sigma_zero, 2500.0)
    flat = np.clip(dark + zero_spot, 0.0, full_scale)

    # 平面口径下的参考量（用与 analyzer 相同的掩膜积分，而不是解析式）。
    f0_target = float(integrate(flat - dark, disk_mask(shape, center, r0)))
    s0_target = float(integrate(flat - dark, square_mask(shape, center, side)))

    if scenario == "mismatched":
        eta_target_s, eta_target_d = 0.4, 0.2
    else:
        eta_target_s, eta_target_d = 0.4, 0.4

    plus_amp = eta_target_d * f0_target / (2.0 * np.pi * sigma_order**2)
    # 单一连续 blaze ramp 的 −1 级只是泄漏底（约 2%），不是共轭级次；
    # asymmetric 场景故意把泄漏抬到超预期（gm1/g1≈0.375 > 0.05），只作
    # 诊断，不触发不可判定。
    minus_amp = 0.02 * plus_amp
    if scenario == "asymmetric":
        minus_amp = 0.15 * f0_target / (2.0 * np.pi * sigma_order**2)
    plus_center = (center[0] - sep, center[1])
    minus_center = (center[0] + sep, center[1])

    # 目标方窗内：0 级贡献 + 目标分量（按 eta_s 比例缩放的平顶 patch）。
    patch = np.zeros(shape, dtype=np.float64)
    patch[square_mask(shape, center, side)] = 1.0

    blaze_base = np.clip(
        dark
        + zero_spot
        + _gaussian(shape, plus_center, sigma_order, plus_amp)
        + _gaussian(shape, minus_center, sigma_order, minus_amp),
        0.0,
        full_scale,
    )

    # 目标方窗内：0 级贡献 + 目标分量（按 eta_s 比例缩放的平顶 patch）。
    patch = np.zeros(shape, dtype=np.float64)
    patch[square_mask(shape, center, side)] = 1.0
    patch_amp = eta_target_s * f0_target / float(patch.sum())

    # 每个重复轮带轻微幅度起伏（真实采集中功率/对准总有抖动）。若所有重复帧
    # 完全相同，bootstrap 的 d 会退化成单点、CI 与随机种子无关，自检与测试都
    # 测不出可复现性。键为 (pass_index, 该角色在本遍内的 ordinal)。
    shaping_factors = {(1, 0): 1.00, (1, 1): 1.03, (2, 0): 0.97, (2, 1): 1.01}
    blaze_factors = {(1, 0): 1.00, (1, 1): 0.97, (2, 0): 1.02, (2, 1): 1.00}

    flat_images = {"flat_pre": flat, "flat_mid": flat, "flat_post": flat}
    t0 = 1_700_000_000.0
    entries: list[dict[str, Any]] = []
    for pass_index, suffix in ((1, ""), (2, "_r2")):
        ordinals = {"blaze_grating": 0, "shaping_target": 0}
        for slot, role in enumerate(PASS_ROLES):
            if role == "dark" and scenario == "missing_dark":
                continue
            frame_id = role + suffix
            path = root / "raw" / f"{pass_index}_{slot}_{frame_id}.npy"
            if role in ordinals:
                # 按轮给 blaze / 整形帧套上轻微幅度起伏。
                table = (
                    blaze_factors
                    if role == "blaze_grating"
                    else shaping_factors
                )
                factor = table[(pass_index, ordinals[role])]
                ordinals[role] += 1
                if role == "blaze_grating":
                    image = blaze_base * factor
                else:
                    # 只放大 patch 分量，0 级斑保持不变。
                    image = dark + zero_spot + patch * (patch_amp * factor)
            elif role == "dark":
                image = dark
            else:
                image = flat_images[role]
            np.save(path, np.round(np.clip(image, 0.0, full_scale)).astype(np.uint16))
            entries.append(
                {
                    "id": frame_id,
                    "timestamp": t0 + pass_index * 100.0 + slot * 5.0,
                    "exposure_ms": 0.1,
                    "saturated_count": 0,
                    "raw_file": path.name,
                    "phase_hash": f"sha256:selftest:{role}",
                }
            )

    config = {
        "exposure_ms": 0.1,
        "panel_shape": list(shape),
        "camera_full_scale": full_scale,
        "dark_source": "missing" if scenario == "missing_dark" else "capture",
        "matrix_interpretation": "synthetic_self_test",
        "frozen_mask": {
            "r0": r0,
            "r1": r1,
            "side": side,
            "sep": sep,
            "tol": 150.0,
            "zero_exclusion": 150.0,
        },
    }
    with (root / "config.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (root / "journal.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    hashes = {role: f"sha256:selftest:{role}" for role in ROLE_BY_ID}
    (root / "phases" / "hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "scenario": scenario,
        "f0": f0_target,
        "s0": s0_target,
        "eta_s_target": eta_target_s,
        "eta_d_target": eta_target_d,
    }


def run_self_test() -> dict[str, Any]:
    """四个合成场景跑完整分析，返回 ``{scenario: analysis}``。"""
    outcomes: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="em_selftest_") as tmp:
        for scenario in SCENARIOS:
            root = Path(tmp) / scenario
            build_synthetic_dataset(root, scenario)
            outcomes[scenario] = analyze_dir(root, bootstrap=2000, seed=0)
    return outcomes


def _self_test_main() -> int:
    """打印四个场景的 verdict/reason 并返回退出码。"""
    outcomes = run_self_test()
    expected = {
        "matched": ("matched", None),
        "mismatched": ("mismatched", None),
        "asymmetric": ("matched", None),
        "missing_dark": ("undeterminable", "missing_same_exposure_dark"),
    }
    ok = True
    for scenario in SCENARIOS:
        result = outcomes[scenario]
        want_verdict, want_reason = expected[scenario]
        verdict_ok = result["verdict"] == want_verdict
        reason_ok = want_reason is None or result["reason"] == want_reason
        ok = ok and verdict_ok and reason_ok
        stats = result["stats"]
        rounds = stats["rounds"] or [{}]
        print(
            f"[{'PASS' if verdict_ok and reason_ok else 'FAIL'}] "
            f"{scenario:12s} verdict={result['verdict']:14s} "
            f"reason={result['reason']} "
            f"mean_d={_fmt(stats['mean_d'])} "
            f"eta_s={_fmt(rounds[0].get('eta_s'))} "
            f"eta_d={_fmt(rounds[0].get('eta_d'))}"
        )
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        description="整形效率与衍射效率匹配性离线分析（不改标定、不碰设备）"
    )
    parser.add_argument(
        "--dir", type=Path, help="采集目录（含 config.json/raw/journal.jsonl）"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="跑四个合成场景自检"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有 analysis.json/EM_REPORT.md",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="bootstrap 次数（默认 2000）"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="bootstrap 随机种子（默认 0）"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test_main()
    if args.dir is None:
        parser.error("--dir 必填（或用 --self-test）")
    root = Path(args.dir)
    if not root.is_dir():
        logger.error(f"目录不存在：{root}")
        return 2
    result = analyze_dir(root, bootstrap=args.bootstrap, seed=args.seed)
    try:
        analysis_path, report_path = write_outputs(root, result, args.overwrite)
    except FileExistsError as exc:
        logger.error(str(exc))
        return 3
    stats = result["stats"]
    print(f"verdict={result['verdict']} reason={result['reason']}")
    print(
        f"eta_s/eta_d 差均值={_fmt(stats['mean_d'])} "
        f"CI95=[{_fmt(stats['ci95_low'])}, {_fmt(stats['ci95_high'])}]"
    )
    print(f"已写 {analysis_path.name} 与 {report_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
