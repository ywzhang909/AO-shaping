"""整形探测判据采集与离线报告构建。

本模块为「先小规模探测、再选整形路线」的流程提供判据采集：把一次 transfer
会话 / 采集目录里的产物（``frozen_parameters.json``、``manifest.json`` /
``config.json``、``journal.json`` / ``session.json``）读取出来，映射成路由决策
需要的 :class:`ProbeReport`，配合 :func:`ao_shaping.tools.slm.shaping.router.choose_route`
给出推荐的整形路线。

设计要点：

* **离线为主**：主路径 ``build_report`` 只读磁盘产物，**不触碰任何设备**。设备
  主动探测放在 :func:`collect_device_probes` 占位，其内部的设备导入都是函数级
  惰性导入，保证模块本身可无设备导入、可单测。
* **best-effort 读取**：产物缺失 / JSON 解析失败 → 对应判据留 ``None`` 并在
  ``notes`` 追加说明，绝不向调用方抛异常。路由把这些 ``None`` 视为「未知而非
  坏」，不会触发任何风险分支（见 router 契约）。
* **建议器而非执行器**：本模块及其 CLI 只产出建议，绝不执行任何整形 / 优化。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ao_shaping.tools.slm.shaping.router import ProbeReport

#: 允许覆盖的字段：即 ``ProbeReport`` 的全部字段名。
_OVERRIDEABLE_FIELDS = frozenset(ProbeReport.__dataclass_fields__)

#: session_dir 下可能存在的分支目录名，用于定位 frozen/manifest/config。
_SUBDIRS = ("fit", "initial", "model", "model_wide640", "model640", "capture")


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """best-effort 读取一个 JSON 文件；失败返回 ``None``，绝不抛。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _append_note(notes: list[str], note: str) -> None:
    """把 note 追加到列表，避免重复。"""
    if note not in notes:
        notes.append(note)


def _find_in_subdirs(session_dir: Path, *names: str) -> Path | None:
    """在 session 根及其常见分支目录里找第一个存在的命名文件。"""
    candidates = [session_dir / name for name in names]
    for sub in _SUBDIRS:
        candidates += [session_dir / sub / name for name in names]
    return next((p for p in candidates if p.is_file()), None)


def _locate_frozen(session_dir: Path) -> Path | None:
    """定位 ``frozen_parameters.json``：fit/、initial/ 或 session 根。"""
    return _find_in_subdirs(session_dir, "frozen_parameters.json")


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """把任意值规整为 ``tuple[str, ...]``（放 geometry_problems）。"""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _read_frozen(frozen_path: Path | None, notes: list[str]) -> tuple[bool, str | None]:
    """读取 frozen_parameters.json 的判据。

    返回值：(是否有可用模型, 是否含 ramp_matrix 的说明)。存在性用于
    ``has_checkpoint`` 判断；``ramp_matrix`` 存在与否记入 notes。
    """
    if frozen_path is None:
        return False, None
    data = _read_json(frozen_path)
    if not isinstance(data, dict):
        _append_note(notes, f"{frozen_path.name} 无法解析，判定不存在可用模型")
        return False, None
    ramp_note = None
    if isinstance(data.get("ramp_matrix"), (list, tuple)):
        ramp_note = f"{frozen_path.name} 含 ramp_matrix，存在可用标定模型"
    elif "ramp_matrix" in data:
        ramp_note = f"{frozen_path.name} 的 ramp_matrix 非数组形状，按不可用处理"
    else:
        ramp_note = f"{frozen_path.name} 不含 ramp_matrix，可能为非标定模型"
    return True, ramp_note


def _read_manifest_config(
    session_dir: Path, notes: list[str]
) -> tuple[bool | None, float | None, list[str]]:
    """读取 manifest.json/config.json，映射 light_ok / snr / noise_floor_ok。

    优先 ``light_check.json``（若有），其次从 manifest/config 里的曝光与零级
    信息提取说明性 notes。返回 (light_ok, snr, extra_notes)。
    """
    light_ok: bool | None = None
    snr: float | None = None
    noise_note: str | None = None
    extra: list[str] = []

    # 优先 light_check.json（记录的正是光路有效性三态 + 噪声底/SNR）。
    light_check = _find_in_subdirs(session_dir, "light_check.json")
    if light_check is not None:
        lc = _read_json(light_check)
        if isinstance(lc, dict):
            ok = lc.get("ok")
            if isinstance(ok, bool):
                light_ok = ok
            peak = lc.get("peak_counts")
            nf = lc.get("noise_floor_counts")
            if (
                isinstance(peak, (int, float))
                and isinstance(nf, (int, float))
                and nf > 0
            ):
                snr = float(peak) / float(nf)
            noise_note = (
                f"light_check.json: ok={lc.get('ok')} 峰值={peak} 噪声底={nf} "
                f"曝光={lc.get('exposure_ms')}ms"
            )
        else:
            _append_note(notes, f"{light_check.name} 无法解析，跳过光路判据")

    # manifest.json / config.json 的存在性只作补充检查（零级等参考信息不提取）。
    manifest = _find_in_subdirs(session_dir, "manifest.json")
    if manifest is not None and not isinstance(_read_json(manifest), (dict, list)):
        _append_note(notes, f"{manifest.name} 无法解析，跳过参考信息")

    config = _find_in_subdirs(session_dir, "config.json")
    if config is not None and not isinstance(_read_json(config), dict):
        _append_note(notes, f"{config.name} 无法解析，跳过配置信息")

    if noise_note is not None:
        extra.append(noise_note)
    return light_ok, snr, extra


def _read_session_geometry(session_dir: Path, notes: list[str]) -> dict[str, Any]:
    """读取 journal.json / session.json，映射几何/方向/相关判据。

    返回一个 dict，键对齐 ``ProbeReport`` 字段名（geometry_correlation /
    direction_agreement / zero_shift_px / focus_changed / model_anchor_error /
    geometry_problems / noise_floor_ok）；缺省键的字段保持 ``None``。
    """
    out: dict[str, Any] = {}
    session_file = _find_in_subdirs(session_dir, "session.json", "journal.json")
    if session_file is None:
        _append_note(notes, "未见 session.json/journal.json，几何/方向判据留 None")
        return out

    data = _read_json(session_file)
    if data is None:
        _append_note(notes, f"{session_file.name} 无法解析，几何/方向判据留 None")
        return out

    # 支持 {data:[...]} 包裹或顶层列表（journal 常是记录数组）。
    records = data if isinstance(data, list) else [data]
    acc: dict[str, Any] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if isinstance(rec.get("data"), dict):
            rec = rec["data"]
        for key, target in (
            ("geometry_correlation", "geometry_correlation"),
            ("correlation", "geometry_correlation"),
            ("direction_agreement", "direction_agreement"),
            ("direction_check", "direction_agreement"),
            ("zero_shift_px", "zero_shift_px"),
            ("focus_changed", "focus_changed"),
            ("model_anchor_error", "model_anchor_error"),
        ):
            if key in rec and acc.get(target) is None:
                acc[target] = rec[key]
        if "geometry_problems" in rec and isinstance(
            rec["geometry_problems"], (list, tuple)
        ):
            acc["geometry_problems"] = _as_str_tuple(rec["geometry_problems"])
        if "noise_floor_ok" in rec and isinstance(rec["noise_floor_ok"], bool):
            acc["noise_floor_ok"] = rec["noise_floor_ok"]

    if "geometry_problems" in acc:
        out["geometry_problems"] = acc["geometry_problems"]
        _append_note(
            notes,
            f"{session_file.name} 记录 geometry_problems="
            f"{list(acc['geometry_problems'])}",
        )
    else:
        out["geometry_problems"] = ()
    for key in (
        "geometry_correlation",
        "direction_agreement",
        "zero_shift_px",
        "focus_changed",
        "model_anchor_error",
        "noise_floor_ok",
    ):
        if acc.get(key) is not None:
            out[key] = acc[key]
            _append_note(notes, f"{session_file.name} 记录 {key}={acc[key]}")
        else:
            out.setdefault(key, None)
    return out


def _resolve_checkpoint(session_dir: Path | None, checkpoint: Path | None) -> bool:
    """has_checkpoint：显式 checkpoint 存在，或 session 内能找到 frozen。"""
    if checkpoint is not None and checkpoint.is_file():
        return True
    if session_dir is None:
        return False
    if _locate_frozen(session_dir) is not None:
        return True
    return False


def _resolve_seed_run(seed_run: Path | None) -> bool:
    """has_seed_run：seed_run 显式给且存在。"""
    return seed_run is not None and seed_run.is_file()


def _apply_overrides(
    report: ProbeReport, overrides: dict[str, Any] | None
) -> ProbeReport:
    """把 overrides 应用到 report，返回新 report；非法键抛 ValueError。"""
    if not overrides:
        return report
    unknown = sorted(set(overrides) - _OVERRIDEABLE_FIELDS)
    if unknown:
        raise ValueError(
            f"overrides 含非法 ProbeReport 字段：{unknown}"
            f"；合法字段：{sorted(_OVERRIDEABLE_FIELDS)}"
        )
    return type(report)(**{**_report_as_dict(report), **overrides})


def _report_as_dict(report: ProbeReport) -> dict[str, Any]:
    """把 ProbeReport 转成普通 dict（保留元组类型）。"""
    return {f: getattr(report, f) for f in _OVERRIDEABLE_FIELDS}


def build_report(
    session_dir: Path | None = None,
    *,
    intent: str = "auto",
    shape: str = "square",
    checkpoint: Path | None = None,
    seed_run: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ProbeReport:
    """离线构建探测报告：判据采集 + collection 提示 + 覆盖。

    参数
    ----
    session_dir:
        一次 transfer 会话 / 采集目录的可选路径。给定时读其产物填判据；否则判据
        全靠 ``overrides``（留给人工实测补充）。
    intent / shape:
        透传给 :class:`ProbeReport` 的任务意图与目标形状。
    checkpoint / seed_run:
        冻结模型 / 种子运行的显式路径；存在性决定 ``has_checkpoint`` /
        ``has_seed_run``。
    overrides:
        调用方直接覆盖任意 :class:`ProbeReport` 字段（如人工实测的 snr /
        geometry_correlation）。键必须是合法字段，否则 ``ValueError``。

    返回一个字段全部取到、未取到的判据为 ``None`` 的 :class:`ProbeReport`。
    本函数**不抛异常**（``overrides`` 传非法键除外），产物缺失一律进 notes。
    """
    notes: list[str] = []
    if session_dir is not None and not session_dir.is_dir():
        _append_note(notes, f"session_dir {session_dir} 不存在，判据全部留 None")

    base: dict[str, Any] = dict(
        intent=intent,
        shape=shape,
        calibrated_shapes=("square", "rectangle", "triangle"),
        has_checkpoint=False,
        has_seed_run=False,
        light_ok=None,
        geometry_correlation=None,
        zero_shift_px=None,
        direction_agreement=None,
        focus_changed=None,
        model_anchor_error=None,
        snr=None,
        noise_floor_ok=None,
        geometry_problems=(),
        notes=(),
    )

    # frozen_parameters.json：确定 has_checkpoint + ramp_matrix 提示。
    frozen_path = None
    if session_dir is not None and session_dir.is_dir():
        frozen_path = _locate_frozen(session_dir)
        if frozen_path is not None:
            _frozen_exists, ramp_note = _read_frozen(frozen_path, notes)
            if ramp_note:
                _append_note(notes, ramp_note)
        else:
            _append_note(notes, "session 内未找到 frozen_parameters.json")

    base["has_checkpoint"] = _resolve_checkpoint(session_dir, checkpoint)
    base["has_seed_run"] = _resolve_seed_run(seed_run)

    if session_dir is not None and session_dir.is_dir():
        light_ok, snr, light_notes = _read_manifest_config(session_dir, notes)
        base["light_ok"] = light_ok
        base["snr"] = snr
        notes.extend(light_notes)
        base.update(_read_session_geometry(session_dir, notes))

    base["notes"] = tuple(notes)
    report = ProbeReport(**base)
    return _apply_overrides(report, overrides)


def collect_device_probes(
    session_dir: Path,
    *,
    intent: str = "auto",
    shape: str = "square",
    exposure_ms: float | None = None,
) -> ProbeReport:
    """设备主动探测——尚未接入。

    预留的硬件主动探测入口。本任务不实现真实设备采集：真实判据（噪声底、零级
    漂移、方向检验、相关性）请先用 :func:`build_report` 读离线产物，或直接经
    ``overrides`` / CLI ``--set`` 提供实测值。

    设备相关导入（相机 / SLM 驱动）只在函数内部懒加载，已在模块级保持无设备
    依赖、可在无硬件环境导入与测试。
    """
    # 惰性占位：未来在此懒加载 camera / SLM 驱动做曝光阶梯与几何探测。
    raise NotImplementedError(
        "设备主动探测尚未接入，请用 --set 提供实测判据，或改用 build_report 读离线产物"
    )


def coerce_value(raw: str) -> Any:
    """把 ``--set`` 的字符串值解析为 Python 值。

    优先用 ``json.loads``（覆盖数字 / bool / None / 数组），解析失败时按字符串：
    ``"square"`` → ``"square"``，``"true"`` → ``"true"``（不隐式 bool）。
    """
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def parse_override_pair(pair: str) -> tuple[str, Any]:
    """解析 ``key=value`` 覆盖对（经 :func:`coerce_value` 规整值）。"""
    key, sep, raw = pair.partition("=")
    if not sep or not key:
        raise ValueError(f"--set 需形如 key=value，收到：{pair!r}")
    return key, coerce_value(raw)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """批量解析 ``key=value`` 覆盖对，非法键在 build_report 时抛 ValueError。"""
    out: dict[str, Any] = {}
    for pair in pairs or ():
        key, value = parse_override_pair(pair)
        out[key] = value
    return out
