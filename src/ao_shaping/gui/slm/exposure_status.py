"""会话曝光探测：一次实测 → 证据落盘 → 时效判定，不建任何 widget。

为什么要单独成模块
------------------
这四件事在 ``bench_settings`` 里本来是同一条链：:func:`probe_session_exposure_ms`
在相机上走倍增阶梯实测饱和上限，:func:`record_exposure_probe` 把结果连同取样时刻
写进探测值表（**唯一**写入点，所以「探测值一定有取样时刻」是结构性保证），
:func:`exposure_state` 只回答「这个值多旧了」。它既不建 widget 也不画状态条
（渲染在 ``bench_status_view``），与侧栏编排无关，却长期占着 ``bench_settings``
的行数 —— 主文件已到模块规模棘轮线 (``scripts/check_module_size.py``)，因此整族
搬到这里。判定、阈值、文案、键名逐字未改（2026-10-29 纯搬迁）。

依赖方向
--------
本模块的公开名由 ``bench_settings`` **re-export**（调用方仍写
``bench_settings.exposure_state``），而本模块自己要用 ``bench_settings`` 的三态表
(:func:`detected_state` / :func:`record_detection`)，所以那两个函数在**调用点内**
延迟导入 —— 模块级互导会让「先 import 谁」决定导入成功与否。

术语见 ``docs/terminology_glossary.md``：本模块一律用规范术语（实验系统、
瞳面），不引入表外口语说法。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.utils.image.exposure_control import (
    configured_daheng_exposure_limit,
    working_exposure_ms,
)

__all__ = [
    "EXPOSURE_PROBE_DEBUG_DIR",
    "EXPOSURE_STALE_S",
    "ExposureProbeFailure",
    "bench_session_id",
    "current_exposure_ms",
    "exposure_state",
    "probe_session_exposure_ms",
    "record_exposure_probe",
]

# ---------------------------------------------------------------------------
# 曝光探测
# ---------------------------------------------------------------------------

#: 探测证据落在 ``data/debug/<脚本名>/`` 下，与本模块的其它探测分开。
EXPOSURE_PROBE_DEBUG_DIR = Path("data") / "debug" / "bench_settings" / "exposure_probe"

#: 确认帧的合格区间。返回的曝光必须让实测峰值落在这里，否则就是截断值。
_CONFIRM_SATURATION = 0.9
_CONFIRM_TARGET = 0.55

#: 会话曝光的**有效期**（秒）：实测同一光路 1.5 h 后光强漂 ×1.35（2026-10-22
#: 连续两次曝光阶梯），故 2 h 是「还认得出来」与「该重测了」的分界。过期只上屏
#: 一个黄点，**不自动改**值 —— 自动重测会换掉操作员刚写进实验记录的参数。
EXPOSURE_STALE_S = 7200.0


def _clock_text(moment: float) -> str:
    """epoch 秒 → 操作员读的 ``HH:MM:SS``（本机时区）。"""
    return datetime.fromtimestamp(float(moment)).strftime("%H:%M:%S")


def bench_session_id(prefix: str = "") -> str:
    """会话标识 = 已有的标定时刻，不起新编号；从未标定过则为空串。

    标定（:func:`publish_session_calibration`）已经写下 ``calibrated_at``，那就是
    「这一次实验」的时间锚点。没标定时冒一个编号，反而会让人以为它能对上某次
    已归档的会话。
    """
    from ao_shaping.gui.slm.bench_settings import detected_state

    for candidate in (prefix, "slm1", ""):
        try:
            stamp = detected_state(candidate).get("calibrated_at")
        except Exception:  # policy: 会话标识只是证据里的附带信息：不在脚本运行里（没有 session_state）时留空，绝不让探测本身失败
            return ""
        if stamp:
            return str(stamp)
    return ""


def record_exposure_probe(
    prefix: str, exposure_ms: float, evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """把一次**实测**曝光连同取样时刻写进探测值表（侧栏「探测」与状态栏「重测」
    的唯一写入点，所以「探测值一定有取样时刻」是结构性保证）。只写实验级表：
    ``per_shape`` / ``per_block`` 的曝光语义完全不动。
    """
    from ao_shaping.gui.slm.bench_settings import detected_state, record_detection

    payload = dict(evidence or {})
    moment = payload.get("probed_at")
    moment = (
        float(moment) if isinstance(moment, (int, float)) else datetime.now().timestamp()
    )
    record_detection(
        prefix,
        {
            "exposure_ms": float(exposure_ms),
            "probed_at": moment,
            "probed_at_text": str(payload.get("probed_at_text") or _clock_text(moment)),
            "session_id": str(payload.get("session_id") or ""),
        },
    )
    return detected_state(prefix)


def exposure_state(prefix: str = "") -> dict[str, Any]:
    """会话曝光的**时效**视图：``exposure_ms`` / ``probed_at`` / ``probed_at_text``
    / ``age_s`` / ``stale``。

    曝光是实验级字段，所以先查给定前缀、查不到退回实验级 —— 每台 SLM 的状态栏
    拿到同一个值。``probed_at`` 缺失（旧记录 / 手填）时 ``age_s`` 为 ``None``、
    ``stale`` 为 ``False``：「不知道多旧」不等于「陈旧」。
    """
    from ao_shaping.gui.slm.bench_settings import detected_state

    detected = detected_state(prefix)
    if detected.get("exposure_ms") is None and prefix:
        detected = detected_state("")
    raw_exposure = detected.get("exposure_ms")
    probed_at = detected.get("probed_at")
    probed_at = float(probed_at) if isinstance(probed_at, (int, float)) else None
    age: float | None = None
    if probed_at is not None:
        age = max(0.0, datetime.now().timestamp() - probed_at)
    return {
        "exposure_ms": float(raw_exposure) if raw_exposure else None,
        "probed_at": probed_at,
        "probed_at_text": str(
            detected.get("probed_at_text")
            or ("" if probed_at is None else _clock_text(probed_at))
        ),
        "age_s": age,
        "stale": bool(age is not None and age > EXPOSURE_STALE_S),
    }


def current_exposure_ms() -> float:
    """曝光控件**此刻**的值（L3：读 widget，不读探测表或运行前快照）。"""
    return float(st.session_state.get("bench_exposure_ms", 0.0) or 0.0)


class ExposureProbeFailure(RuntimeError):
    """曝光探测失败，携带一条可直接上屏的中文说明。"""


def _confirm_working_exposure(
    read_frame: Callable[[], np.ndarray],
    reset_exposure: Callable[[float], float | None],
    ceiling_ms: float,
    *,
    work_fraction: float,
    floor_ms: float,
    max_attempts: int = 3,
) -> tuple[float, float, float, list[dict[str, float]], np.dtype]:
    """回读验证工作曝光，返回 ``(工作曝光, 读回曝光, 峰值, 逐次回读记录, 帧 dtype)``。

    阶梯给出的上限只是响应模型；必须用**那一档曝光**回读一帧再判定，否则会把
    截断的曝光当成可用的交出去（实测 2026-10-17：阶梯定上限 0.243 ms，取其一半
    回读却是 255/255）。

    饱和时按比例回退再试；``max_attempts`` 次仍饱和则抛
    :class:`RuntimeError`。本函数不碰设备状态，只用注入的两个回调，因此可以
    用合成帧直接做数值回归。最后一个返回值是回读帧的 dtype，调用方据此算满量程。
    """
    working = working_exposure_ms(ceiling_ms, fraction=work_fraction, floor_ms=floor_ms)
    confirmations: list[dict[str, float]] = []
    readback = working
    peak = 0.0
    dtype: np.dtype = np.dtype(np.uint8)
    for _ in range(max_attempts):
        readback = float(reset_exposure(working) or working)
        frame = np.asarray(read_frame())
        peak = float(np.max(frame)) if frame.size else 0.0
        dtype = frame.dtype
        full = (
            float(np.iinfo(frame.dtype).max)
            if np.issubdtype(frame.dtype, np.integer)
            else 255.0
        )
        frac = peak / full if full else 0.0
        confirmations.append({"ms": working, "peak": peak, "peak_fraction": frac})
        if frac >= _CONFIRM_SATURATION:
            # Clipped: back off proportionally rather than returning it.
            working = max(floor_ms, working * _CONFIRM_TARGET / max(frac, 1e-9))
            continue
        break
    else:
        raise RuntimeError(
            f"工作曝光在 {len(confirmations)} 次回退后仍然饱和"
            f"（最后一次 {confirmations[-1]['peak_fraction']:.0%}）；"
            "光路比阶梯看到的高出很多，请降低激光功率或加衰减后重测"
        )
    return working, readback, peak, confirmations, dtype


def probe_session_exposure_ms(
    cam_type: str = "daheng",
    cam_id: int = 0,
    *,
    work_fraction: float = 0.5,
    floor_ms: float = 0.02,
    debug_dir: Path | None = None,
) -> tuple[float, dict[str, Any]]:
    """在当前光路下实测一个可用的会话曝光（ms），返回 ``(曝光, 证据)``。

    探测用**倍增阶梯**找饱和上限（``probe_exposure_ceiling_ms``：曝光翻倍直到
    峰值达到 ``saturation_fraction``×满量程），再用 ``working_exposure_ms`` 取
    ``work_fraction``×上限作为工作曝光。上限取「峰值将达满量程 90%」的那一档
    再乘 0.9 余量（≈81%），所以工作曝光下峰值约在 40%——既能拿到足够原始
    计数，又留有余量防截断。

    失败时抛 :class:`ExposureProbeFailure`：曝光是**光路属性**，不随传感器变，
    所以宁可中止也不能给一个猜的值。若 ``AO_DAHENG_MAX_EXPOSURE_MS`` 有配置则
    优先用它（不探测），这是既有契约，此处不改变优先级。

    上限一律再受**相机自己的曝光上限**约束：相机最大值是「传感器还能积分多久」，
    不是「何时饱和」，所以它只是上限而不是一次测量——但超过它的曝光根本设不下去
    （驱动会静默钳位，于是实测峰值与请求曝光对不上）。相机报不出范围的后端按
    **无上限**处理，与旧行为一致。

    相机在函数内打开并关闭；``debug_dir`` 给定时把阶梯每一档的
    ``(曝光, 峰值)`` 写成 JSON，便于复盘「当时画面到底多亮」。
    """
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_range,
    )
    from ao_shaping.utils.image.exposure_control import probe_exposure_ceiling_ms

    if cam_type.lower() != "daheng":
        # 其他相机类型走各自的曝光语义，本探测只服务大恒的定量 CCD 路径。
        raise ExposureProbeFailure(
            f"曝光自动探测目前只支持大恒（daheng），当前相机类型为 {cam_type!r}。"
            "请在「会话曝光」直接填入实测值。"
        )
    camera_max_ms: float | None = None
    try:
        with create_camera(cam_type, int(cam_id), exposure_time_ms=floor_ms) as cam:
            # ``n_sample=1`` 返回整数原始帧；取 n>1 会得到 float32 平均帧，
            # 而阶梯判定要求整数帧（见 docs/slm/slm_model_in_loop.md 实测记录）。
            #
            # 相机上限先问设备要：拿不到（后端不支持 / 调用失败）就按无上限
            # 继续，绝不让「读不到范围」把一次可用的探测变成失败。
            try:
                _, camera_max_ms = get_camera_exposure_range(cam)
            except Exception as exc:  # policy: 后端不支持曝光范围查询不是探测失败：按无上限继续本次探测
                logger.debug("相机曝光范围不可用，本次探测按无上限处理: {}", exc)
                camera_max_ms = None
            if camera_max_ms is not None and (
                not np.isfinite(camera_max_ms) or camera_max_ms <= 0
            ):
                camera_max_ms = None

            # Walk the ladder **once**: letting ``resolve_daheng_exposure_limit``
            # probe and then probing again here would cost a second walk and
            # could contradict itself on a bench whose light drifts (measured
            # 2026-10-17: peak at 0.1 ms moved 63 -> 79 within 20 min). Taking
            # the operator's env ceiling when it is set keeps that precedence
            # without touching the camera, exactly as the shared resolver does.
            configured = configured_daheng_exposure_limit()
            if configured > 0:
                # 操作员的取值压过探测，但不压过传感器：取二者中更紧的那个。
                ceiling = float(configured)
                if camera_max_ms is not None:
                    ceiling = min(ceiling, camera_max_ms)
                estimate = None
            else:
                estimate = probe_exposure_ceiling_ms(
                    cam.reset_exposure_time,
                    lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                    start_ms=floor_ms,
                    hard_max_ms=camera_max_ms,
                )
                ceiling = float(estimate.ceiling_ms)

            working, readback, peak, confirmations, dtype = _confirm_working_exposure(
                lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                cam.reset_exposure_time,
                ceiling,
                work_fraction=work_fraction,
                floor_ms=floor_ms,
            )
    except ExposureProbeFailure:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        raise ExposureProbeFailure(
            f"曝光探测失败：{exc}。"
            "请确认激光已出光、光路未遮挡，并在「会话曝光」手动填入实测值。"
        ) from exc

    full_scale = (
        float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 255.0
    )
    # 取样时刻取在这里（相机已关、测量已完），而不是函数入口：曝光探测本身要跑
    # 阶梯，入口时刻与「这个值对应哪一刻的光强」能差几十秒。
    probed_at = datetime.now().timestamp()
    evidence: dict[str, Any] = {
        # 附加键（不改函数签名/返回）：这次探测属于哪个会话、什么时候做的。
        # 状态栏据此判过期，复盘时也能把一次探测对回某次实验。
        "session_id": bench_session_id(),
        "probed_at": probed_at,
        "probed_at_text": _clock_text(probed_at),
        "cam_type": cam_type,
        "cam_id": int(cam_id),
        "ceiling_ms": ceiling,
        "work_fraction": float(work_fraction),
        "working_ms": working,
        "readback_ms": readback,
        "peak": peak,
        "full_scale": full_scale,
        "peak_fraction": peak / full_scale if full_scale else 0.0,
        "ceiling_source": "env" if estimate is None else "image_probe",
        "confirmations": confirmations,
        "last_safe_ms": float(estimate.last_safe_ms) if estimate else 0.0,
        "saturated_ms": None
        if estimate is None or estimate.saturated_ms is None
        else float(estimate.saturated_ms),
        "ladder_peak_fractions": (
            [] if estimate is None else [float(v) for v in estimate.peak_fractions]
        ),
        # How the ceiling was reached, so the next probe can be replayed: the
        # per-rung extrapolations (the old ``min`` of these) with their
        # exposures, the fitted saturation exposure, and the measured bound
        # that clamped it.  ``ladder_peak_fractions`` has one entry per rung;
        # ``ladder_estimates_ms`` only one per rung that carried signal, so zip
        # it with ``ladder_estimate_exposures_ms``.
        "ladder_estimates_ms": (
            [] if estimate is None else [float(v) for v in estimate.estimates_ms]
        ),
        "ladder_estimate_exposures_ms": (
            []
            if estimate is None
            else [float(v) for v in estimate.estimate_exposures_ms]
        ),
        "fit_ms": None if estimate is None else estimate.fit_ms,
        "fit_points": 0 if estimate is None else int(estimate.fit_points),
        "bound_ms": None if estimate is None else estimate.bound_ms,
        "first_clipped_ms": None if estimate is None else estimate.first_clipped_ms,
        # 相机自己报的曝光上限（None = 后端报不出，按无上限处理）。探测路径下它
        # 已被 ``probe_exposure_ceiling_ms`` 用作 ``hard_max_ms``，所以命中时
        # ``bound_ms`` 与它相等；env 路径没有 ``estimate``，这个键是唯一的记录。
        "camera_max_ms": camera_max_ms,
    }
    target = Path(debug_dir) if debug_dir is not None else EXPOSURE_PROBE_DEBUG_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (target / f"{stamp}.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # 证据落盘失败不否定已完成的测量
        logger.warning("曝光探测证据写入 {} 失败: {}", target, exc)
    logger.info(
        "曝光探测：上限 {:.4f} ms（最后未饱和 {:.4f} ms），工作曝光 {:.4f} ms，"
        "峰值 {:.0f}/{:.0f} ({:.0%})",
        ceiling,
        evidence["last_safe_ms"],
        working,
        peak,
        full_scale,
        evidence["peak_fraction"],
    )
    return working, evidence
