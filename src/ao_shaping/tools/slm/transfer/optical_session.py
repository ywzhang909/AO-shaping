"""光路指纹：把一次会话的光路条件压成可比较的字段，并给出加载推荐。

一份检查点能不能续用，取决于它是否在**同一条光路**上测得（窗口、零级位置、曝光、
相机格式、控制分辨率任一换了，模型映射就不再适用）。本模块只做纯函数：从会话目录
已有产物（``journal.json`` / ``config.json`` / ``manifest.json`` / 拟合产物 /
``formats.json`` / ``optical_path_probe.json``）收集指纹，缺什么写 ``None``、从不
抛异常；指纹的唯一落盘/读取入口；以及逐维度打分与推荐（权重与硬拒绝条件只在这里
有一份实现）。字段在任一侧缺席时该维度记 ``0.5`` 并注明 ``字段缺失`` ——「不知道」
既不能当兼容也不能当不兼容。硬拒绝四种：光路探针 ``low_snr`` / ``no_response``、
零级位移超过窗口跨度一半、条目已标 ``stale``、注册哈希对不上。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "AUTO_APPLY_THRESHOLD",
    "COMPAT_THRESHOLD",
    "DIMENSION_WEIGHTS",
    "FINGERPRINT_NAME",
    "compatibility_score",
    "compatibility_verdict",
    "compute_optical_fingerprint",
    "load_fingerprint",
    "recommend_load",
    "save_fingerprint",
]

#: 指纹文件名（会话目录里一份；``history_browser`` 按这个名字回退查找）。
FINGERPRINT_NAME = "fingerprint.json"
#: 「可用但需确认」的下限：分数低于它的候选仍可手选，只是不会被自动应用。
COMPAT_THRESHOLD = 0.60
#: 「推荐 / 可自动应用」的下限：分数达到它才允许不经确认自动应用。
AUTO_APPLY_THRESHOLD = 0.80
#: 逐维度权重。改这里等于改推荐口径，必须同时改 GUI 的文案阈值。
DIMENSION_WEIGHTS: dict[str, float] = {
    "照明重叠": 0.40,
    "曝光": 0.20,
    "相机格式与像元": 0.15,
    "控制分辨率": 0.15,
    "零级位移": 0.10,
}
#: 光路探针里表示"这条光路不该拿来加载模型"的状态。
_INVALID_OPTICS = ("low_snr", "no_response")
#: 零级位移上限：超过窗口跨度的一半即硬拒绝；无窗口信息时退回 400 px。
_ORIGIN_LIMIT_FRACTION = 0.5
_ORIGIN_LIMIT_FALLBACK_PX = 400.0
#: 指纹的全部键（顺序即落盘顺序的提示）。
_FINGERPRINT_KEYS = (
    "created_at", "ramp_matrix_sha256", "window_span", "window_step", "zero_xy", "zero_radius",
    "period", "peak_counts", "full_scale", "exposure_ms", "camera_format", "pixel_size_um",
    "control_grid", "fft_size", "optics_validity", "origin_px",
)
#: 计划类产物（journal 的 ``plan`` / config.json）里要搬进指纹的键。
_PARAM_KEYS = ("window_span", "window_step", "zero_radius", "period", "control_grid",
               "fft_size", "zero_xy", "pixel_size_um", "camera_format")
#: 各来源产物所在的模式（相对会话目录；先读到的优先，见 :func:`_put`）。
_CONFIG_PATTERNS = ("config.json", "shape/*/*/config.json", "*/*/config.json")
_MANIFEST_PATTERNS = ("shape/*/*/manifest.json", "capture/manifest.json",
                      "common_replay/*/manifest.json", "*/*/manifest.json")
_CAMERA_PATTERNS = ("formats.json", "camera_probe/formats.json",
                    "*/formats.json", "*/*/formats.json")
_PROBE_PATTERNS = ("optical_path_probe.json", "*/optical_path_probe.json",
                   "*/*/optical_path_probe.json")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _iter_json(root: Path, patterns: Sequence[str]) -> Iterator[Mapping[str, Any]]:
    """按模式顺序产出能解析成映射的 JSON 产物。"""
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            payload = _read_json(path)
            if isinstance(payload, Mapping):
                yield payload


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    first, second = _number(value[0]), _number(value[1])
    if first is None or second is None:
        return None
    return first, second


def _put(fingerprint: dict[str, Any], key: str, value: Any) -> None:
    """只在尚未取到值时写入（先读到的产物优先），``None`` 永不上写。"""
    if fingerprint.get(key) is None and value is not None:
        fingerprint[key] = value


def _copy(fingerprint: dict[str, Any], payload: Mapping[str, Any],
          keys: Sequence[str]) -> None:
    for key in keys:
        _put(fingerprint, key, payload.get(key))


def _entries(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _matrix_sha256(matrix: Any) -> str | None:
    if matrix is None:
        return None
    payload = json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fill_from_params(fingerprint: dict[str, Any], root: Path) -> None:
    """campaign 的 journal ``plan`` 与逐阶段 ``config.json``（非 campaign 会话）。"""
    for entry in _entries(_read_json(root / "journal.json")):
        _put(fingerprint, "created_at", entry.get("timestamp_utc"))
        if isinstance(entry.get("plan"), Mapping):
            _copy(fingerprint, entry["plan"], _PARAM_KEYS)
        if entry.get("stage") == "geometry":
            _put(fingerprint, "zero_xy", entry.get("zero_xy"))
    for config in _iter_json(root, _CONFIG_PATTERNS):
        _copy(fingerprint, config, _PARAM_KEYS)
        _put(fingerprint, "created_at", config.get("timestamp_utc"))


def _fill_from_manifests(fingerprint: dict[str, Any], root: Path) -> None:
    """阶段 manifest：段内峰值 / 满量程 / 曝光。最亮一段的读数先写。"""
    rows = [row for pattern in _MANIFEST_PATTERNS for path in sorted(root.glob(pattern))
            for row in _entries(_read_json(path))]
    if not rows:
        return

    def peak(row: Mapping[str, Any]) -> float:
        return _number(row.get("peak")) or _number(row.get("peak_counts")) or -1.0

    brightest = max(rows, key=peak)
    _put(fingerprint, "peak_counts", peak(brightest) if peak(brightest) >= 0 else None)
    for row in (brightest, *rows):
        _put(fingerprint, "exposure_ms", _number(row.get("exposure_ms")))
        _put(fingerprint, "full_scale", _number(row.get("full_scale")))


def _fill_from_model(fingerprint: dict[str, Any], root: Path) -> None:
    """拟合/冻结参数里的载波矩阵、控制网格、FFT 尺寸与瞳面原点。"""
    sources = [root / "fit" / "provenance.json"]
    sources += sorted(root.glob("**/frozen_parameters.json"))
    for path in sources:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        parameters = payload.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = payload
        _put(fingerprint, "ramp_matrix_sha256", _matrix_sha256(parameters.get("ramp_matrix")))
        _put(fingerprint, "origin_px", parameters.get("origin"))
        _put(fingerprint, "control_grid",
             payload.get("grid_size") or payload.get("control_grid"))
        _put(fingerprint, "fft_size", payload.get("fft_size"))


def _fill_from_camera(fingerprint: dict[str, Any], root: Path) -> None:
    """相机格式：优先取 12 位那一条观测，否则取最后一条。"""
    for payload in _iter_json(root, _CAMERA_PATTERNS):
        rows = _entries(payload.get("observations"))
        if not rows:
            continue
        chosen = next((row for row in reversed(rows)
                       if "12" in str(row.get("format", "")).upper()), rows[-1])
        _put(fingerprint, "camera_format", chosen.get("format"))
        _put(fingerprint, "pixel_size_um", chosen.get("pixel_size_um"))
        _put(fingerprint, "exposure_ms", _number(chosen.get("exposure_ms")))


def _fill_from_probe(fingerprint: dict[str, Any], root: Path) -> None:
    """光路探针结论：``ok`` 之外的值（``low_snr`` / ``no_response``）会触发硬拒绝。"""
    for payload in _iter_json(root, _PROBE_PATTERNS):
        verdict = payload.get("verdict")
        state = verdict.get("state") if isinstance(verdict, Mapping) else verdict
        if isinstance(state, str) and state:
            _put(fingerprint, "optics_validity", state)
            return


def compute_optical_fingerprint(root: str | Path | None = None,
                                **overrides: Any) -> dict[str, Any]:
    """从会话目录的已有产物收集指纹；读不到的字段为 ``None``，从不抛异常。"""
    fingerprint: dict[str, Any] = dict.fromkeys(_FINGERPRINT_KEYS)
    if root is not None:
        base = Path(root)
        base = base.parent if base.is_file() else base
        if base.is_dir():
            for fill in (_fill_from_params, _fill_from_manifests, _fill_from_model,
                         _fill_from_camera, _fill_from_probe):
                fill(fingerprint, base)
    fingerprint.update({key: value for key, value in overrides.items()
                        if key in fingerprint and value is not None})
    if fingerprint.get("created_at") is None:
        fingerprint["created_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    for key in ("zero_xy", "origin_px"):
        pair = _pair(fingerprint[key]) if fingerprint.get(key) is not None else None
        if pair is not None:
            fingerprint[key] = list(pair)
    return fingerprint


def save_fingerprint(fingerprint: Mapping[str, Any], path: str | Path) -> Path:
    """把指纹写到 ``path``（目录则写其中的 :data:`FINGERPRINT_NAME`），返回文件路径。"""
    target = Path(path)
    if target.is_dir() or target.suffix.lower() != ".json":
        target = target / FINGERPRINT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(fingerprint), indent=2, ensure_ascii=False,
                                 sort_keys=True, default=str), encoding="utf-8")
    return target


def load_fingerprint(path: str | Path) -> dict[str, Any] | None:
    """读指纹；目录则读其中的 :data:`FINGERPRINT_NAME`。缺失或损坏返回 ``None``。"""
    target = Path(path)
    if target.is_dir():
        target = target / FINGERPRINT_NAME
    payload = _read_json(target)
    return dict(payload) if isinstance(payload, Mapping) else None


def _relative(value_a: float, value_b: float) -> float:
    """1 表示完全相同, 0 表示量级差一倍以上; 两侧同为 0 视为相同。"""
    denominator = max(abs(value_a), abs(value_b), 1e-12)
    return max(0.0, min(1.0, 1.0 - abs(value_a - value_b) / denominator))


def _dimension(name: str, value: float, reason: str) -> dict[str, Any]:
    return {"name": name, "weight": DIMENSION_WEIGHTS[name],
            "value": round(max(0.0, min(1.0, value)), 6), "reason": reason}


def _illumination_dimension(current: Mapping[str, Any],
                            candidate: Mapping[str, Any]) -> dict[str, Any]:
    a_peak, b_peak = _number(current.get("peak_counts")), _number(candidate.get("peak_counts"))
    if a_peak is None or b_peak is None:
        return _dimension("照明重叠", 0.5, "字段缺失（peak_counts），按 0.5 计")
    a_full, b_full = _number(current.get("full_scale")), _number(candidate.get("full_scale"))
    if a_full and b_full:
        a_ratio, b_ratio = a_peak / a_full, b_peak / b_full
        return _dimension("照明重叠", _relative(a_ratio, b_ratio),
                          f"峰值占满量程 {a_ratio:.3f} 对 {b_ratio:.3f}")
    score = _relative(a_peak, b_peak) if a_peak or b_peak else 1.0
    return _dimension("照明重叠", score, f"全幅峰值 {a_peak:.0f} 对 {b_peak:.0f}")


def _exposure_dimension(current: Mapping[str, Any],
                        candidate: Mapping[str, Any]) -> dict[str, Any]:
    a_value, b_value = _number(current.get("exposure_ms")), _number(candidate.get("exposure_ms"))
    if a_value is None or b_value is None:
        return _dimension("曝光", 0.5, "字段缺失（exposure_ms），按 0.5 计")
    return _dimension("曝光", _relative(a_value, b_value),
                      f"曝光 {a_value:g} ms 对 {b_value:g} ms")


def _camera_dimension(current: Mapping[str, Any],
                      candidate: Mapping[str, Any]) -> dict[str, Any]:
    a_format, b_format = current.get("camera_format"), candidate.get("camera_format")
    a_pixel, b_pixel = _number(current.get("pixel_size_um")), _number(candidate.get("pixel_size_um"))
    if a_format is None or b_format is None:
        format_value, format_reason = 0.5, "相机格式字段缺失（按 0.5 计）"
    elif str(a_format).strip().upper() == str(b_format).strip().upper():
        format_value, format_reason = 1.0, f"相机格式同为 {a_format}"
    else:
        format_value, format_reason = 0.0, f"相机格式 {a_format} 对 {b_format}"
    if a_pixel is None or b_pixel is None:
        pixel_value, pixel_reason = 0.5, "像元尺寸字段缺失（按 0.5 计）"
    else:
        pixel_value = _relative(a_pixel, b_pixel)
        pixel_reason = f"像元 {a_pixel:g} µm 对 {b_pixel:g} µm"
    return _dimension("相机格式与像元", (format_value + pixel_value) / 2.0,
                      f"{format_reason}；{pixel_reason}")


def _control_dimension(current: Mapping[str, Any],
                       candidate: Mapping[str, Any]) -> dict[str, Any]:
    a_grid, b_grid = _pair(current.get("control_grid")), _pair(candidate.get("control_grid"))
    if a_grid is None or b_grid is None:
        return _dimension("控制分辨率", 0.5, "字段缺失（control_grid），按 0.5 计")
    score = (_relative(a_grid[0], b_grid[0]) + _relative(a_grid[1], b_grid[1])) / 2.0
    return _dimension("控制分辨率", score,
                      f"控制网格 {a_grid[0]:g}×{a_grid[1]:g} 对 {b_grid[0]:g}×{b_grid[1]:g}")


def _origin_terms(current: Mapping[str, Any],
                  candidate: Mapping[str, Any]) -> tuple[float | None, float]:
    """返回 ``(零级位移, 允许上限)``；任一侧没有零级坐标时位移为 ``None``。"""
    spans = [value for value in (_number(current.get("window_span")),
                                 _number(candidate.get("window_span"))) if value]
    limit = _ORIGIN_LIMIT_FRACTION * max(spans) if spans else _ORIGIN_LIMIT_FALLBACK_PX
    a_point, b_point = _pair(current.get("zero_xy")), _pair(candidate.get("zero_xy"))
    if a_point is None or b_point is None:
        return None, limit
    return ((a_point[0] - b_point[0]) ** 2 + (a_point[1] - b_point[1]) ** 2) ** 0.5, limit


def _origin_dimension(current: Mapping[str, Any],
                      candidate: Mapping[str, Any]) -> dict[str, Any]:
    shift, limit = _origin_terms(current, candidate)
    if shift is None:
        return _dimension("零级位移", 0.5, "字段缺失（zero_xy），按 0.5 计")
    return _dimension("零级位移", 1.0 - shift / limit,
                      f"零级位移 {shift:.1f} px（上限 {limit:.1f} px）")


def compatibility_score(current: Mapping[str, Any],
                        candidate: Mapping[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """逐维度打分，返回 ``(加权总分, 维度明细)``；权重见 :data:`DIMENSION_WEIGHTS`。"""
    if not isinstance(current, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("compatibility_score 需要两个指纹映射")
    scorers = (_illumination_dimension, _exposure_dimension, _camera_dimension,
               _control_dimension, _origin_dimension)
    dimensions = [scorer(current, candidate) for scorer in scorers]
    total_weight = sum(item["weight"] for item in dimensions)
    score = sum(item["weight"] * item["value"] for item in dimensions) / total_weight
    return round(score, 6), dimensions


def compatibility_verdict(score: float) -> str:
    """由兼容分数给出中文结论：``推荐`` / ``谨慎`` / ``不推荐``。

    三档由两个阈值切开：``>= AUTO_APPLY_THRESHOLD`` 为 ``推荐``（可不经确认
    自动应用），``>= COMPAT_THRESHOLD`` 为 ``谨慎``（可用但需人工确认），
    其余 ``不推荐``。页面与检查卡都读这一个函数，不要在 GUI 里另写阈值判断。
    """
    if score >= AUTO_APPLY_THRESHOLD:
        return "推荐"
    if score >= COMPAT_THRESHOLD:
        return "谨慎"
    return "不推荐"


def _entry_fingerprint(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """内联映射 / 内联路径 / 条目 ``path``（其本身或其下的指纹文件）。"""
    inline = entry.get("fingerprint")
    if isinstance(inline, Mapping) and inline:
        return dict(inline)
    sources: list[str] = [inline] if isinstance(inline, str) and inline.strip() else []
    raw = entry.get("path")
    if raw:
        sources.append(str(raw))
    for source in sources:
        loaded = load_fingerprint(source)
        if loaded:
            return loaded
    return None


def _reject_reasons(current: Mapping[str, Any], candidate: Mapping[str, Any],
                    entry: Mapping[str, Any]) -> tuple[list[str], bool]:
    """返回 ``(硬拒绝原因, 是否被拒)``；被拒候选仍留在 ``ranked`` 里供手选。"""
    reasons: list[str] = []
    if entry.get("stale"):
        reasons.append("检查点已标 stale（文件缺失或注册哈希不符）")
    if entry.get("sha256_ok") is False or entry.get("verified") is False:
        reasons.append("检查点哈希校验失败")
    state = candidate.get("optics_validity")
    state = state.get("state") if isinstance(state, Mapping) else state
    if state is not None and str(state) in _INVALID_OPTICS:
        reasons.append(f"光路探针结论为 {state}")
    shift, limit = _origin_terms(current, candidate)
    if shift is not None and shift > limit:
        reasons.append(f"零级位移 {shift:.1f} px 超过上限 {limit:.1f} px")
    return reasons, bool(reasons)


def recommend_load(current: Mapping[str, Any] | None,
                   candidates: Sequence[Mapping[str, Any]],
                   *, min_score: float = COMPAT_THRESHOLD) -> dict[str, Any]:
    """对候选检查点排序并给出推荐；被拒候选仍留在 ``ranked`` 里（带原因）。"""
    if current is not None and not isinstance(current, Mapping):
        raise TypeError("recommend_load 的 current 必须是映射或 None")
    ranked: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        entry = dict(item)
        fingerprint = _entry_fingerprint(entry)
        if current is None or fingerprint is None:
            note = ("未提供当前光路指纹，无法评分" if current is None
                    else "缺少光路指纹，无法判定兼容")
            ranked.append({"candidate": entry, "score": 0.0, "dimensions": [],
                           "rejected": current is not None, "reasons": [note]})
            continue
        score, dimensions = compatibility_score(current, fingerprint)
        reasons, rejected = _reject_reasons(current, fingerprint, entry)
        if score < min_score:
            reasons.append(f"兼容分数 {score:.2f} 低于阈值 {min_score:.2f}")
        ranked.append({"candidate": entry, "score": score, "dimensions": dimensions,
                       "rejected": rejected, "reasons": reasons})
    ranked.sort(key=lambda row: (row["rejected"], -row["score"]))
    recommended = next((row for row in ranked
                        if not row["rejected"] and row["score"] >= min_score), None)
    return {"recommended": recommended, "ranked": ranked, "threshold": min_score}
