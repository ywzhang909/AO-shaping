"""Closed-loop far-field shaping section for the multi-SLM controller GUI.

Why this module exists
----------------------
The three open-loop shaping controls that used to live in ``pattern_controls``
("GS方形整形" / "GS方形整形(高斯版)" / "稳像法整形") are gone: on this
2f-Fourier bench open-loop square shaping with uncalibrated parameters is a
measured failure (``report/shape_trials/README.md``,
``report/shape_trials/online_20261010.md``). What replaced them is the
closed-loop pipeline the CLI runners drive:

- **GS+SPGD** (``slm-gs-refine``): interleaved-replay validated 2026-10-12
  (``report/shape_trials/gs_refine_live_20261012.md``) — flat 0.534 →
  GS 0.767 → best 0.774 (re-verified 2026-10-14 through this very section).
- **Model-in-loop** (``slm-model-in-loop``): the frontier method
  (``report/shape_trials/model_in_loop_live_20261013.md``), best uniformity so
  far but not yet an accepted uniform square; needs torch.
- **迁移学习** (``slm_transfer`` / ``ao_shaping.tools.slm.transfer``): 少参数物理
  物理预测模型 + 当前光路重测几何 + 检查点迁移，再用 CCD 锚定短块与 A–B–B–A 配对
  验收（``report/shape_trials/multishape_capability/``）；同样需要 torch。

Architecture — why adapters + decorators
----------------------------------------
All three optimizers are **actively being developed**, and they do not share a
result type, a parameter object or even a lifecycle. Rather than letting that
leak into the page, every algorithm is a small adapter implementing
:class:`ShapingMethod`, and the cross-cutting concerns are decorators:

- :func:`register_shaping_method` — class decorator that registers an adapter
  in :data:`SHAPING_METHODS`. Adding a fourth algorithm is one decorated class;
  the selector, the run button and the lifecycle need no edit.
- :func:`shaping_session` — method decorator wrapping :meth:`ShapingMethod.run`.
  It owns the parts that must never be forgotten: the optimizers open and close
  the SLM + camera themselves (Santec SDK handles are not re-entrant), so every
  run stops the background toggle, releases the GUI handle, and re-acquires it
  in ``finally`` — an optimizer-side crash cannot strand the page without a
  device. Errors are logged and surfaced once, here.
- :class:`ShapingOutcome` — the single normalized result the page renders. The
  adapters translate ``Recorder`` / ``ModelInLoopResult`` into it, so the UI
  never touches an optimizer-specific type and an API change stays confined to
  one ``execute()`` / ``save()`` pair.

The concrete per-algorithm work is deliberately thin: build the canonical
runner parameter object (:mod:`ao_shaping.runners.runner_common` — the same one
the CLI uses, so GUI and CLI runs cannot drift), call the optimizer, normalise,
save through the canonical saver, display through the normal ``display_phase``
driver path (raw radians → correction + LUT + shift).

The run is synchronous on the main Streamlit thread (same pattern as the
removed open-loop GS generation): the page blocks while the loop drives
hardware, and the optimizer's ``callback(epoch, score)`` only updates widgets
owned by this handler — ``st.session_state`` is never touched from another
thread.
"""

from __future__ import annotations

import functools
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.slm.santec import (
    MEMORY_MODE_INTERNAL,
    Santec,
    describe_santec_error,
)
from ao_shaping.drivers.slm.santec.slm200_constants import (
    PITCH_UM as SLM_PITCH_UM,
)
from ao_shaping.drivers.slm.santec.slm200_constants import (
    PIXEL_SIZE_UM as SLM_PIXEL_SIZE_UM,
)
from ao_shaping.gui.slm import bench_settings
from ao_shaping.gui.slm.bench_settings import (
    CAM_TYPE_OPTIONS,
    MIL_TARGET_SIDE_MIN_PX,
)
from ao_shaping.gui.slm.link_design_assist import (
    DEFAULT_FAR_SIDE,
    FAR_UNITS,
    LINK_FIELDS,
    build_params,
    compute_link_summary,
    compute_reverse,
    far_side_to_m,
    format_ccd_length,
    format_far_length,
    summary_rows,
)
from ao_shaping.gui.slm import sampling_assist
from ao_shaping.gui.slm.pattern_controls import refresh_phase_preview
from ao_shaping.gui.slm.transfer_progress import collect_progress, render_progress
from ao_shaping.gui.utils.phase_overlay import apply_overlays_to_handle
from ao_shaping.optimizer.wfless.slm_gs_refine import VALIDATED_GS_REFINE_RECIPE
from ao_shaping.utils.io.timestamp import format_ts
from ao_shaping.runners.runner_common import (
    SlmGsRefineParams,
    SlmModelInLoopParams,
    build_gs_refine_config,
    build_model_in_loop_config,
    save_gs_refine_results,
    save_model_in_loop_results,
)
from ao_shaping.utils.cli.exposure import (
    AUTO_EXPOSURE_TOKEN,
    DEFAULT_EXPOSURE_TOKEN,
)
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    configured_daheng_exposure_limit,
)
from ao_shaping.utils.image.target.shapes import (
    TARGET_SHAPE_LABELS as MIL_TARGET_SHAPE_LABELS,
)
from ao_shaping.utils.wavefront.sampling_limit import evaluate_sampling

__all__ = [
    "CAM_TYPE_OPTIONS",
    "GS_REFINE_RECIPE",
    "MIL_BUDGET_PRESETS",
    "SHAPING_METHODS",
    "TRANSFER_RECIPE",
    "TRANSFER_SHAPE_LABELS",
    "TRANSFER_STAGE_LABELS",
    "GsRefineMethod",
    "ModelInLoopMethod",
    "SlmOwnership",
    "ShapingMethod",
    "ShapingOutcome",
    "TransferMethod",
    "build_gs_refine_params",
    "build_model_in_loop_params",
    "discover_transfer_checkpoint",
    "get_shaping_method",
    "get_shaping_method_by_label",
    "outcome_from_gs_recorder",
    "outcome_from_model_in_loop",
    "outcome_from_transfer",
    "plan_transfer_session_dir",
    "register_shaping_method",
    "render_closed_loop_shaping",
    "shaping_method_labels",
    "shaping_session",
    "validate_daheng_exposure",
    "validate_daheng_fixed_exposure",
    "validate_exposure",
    "validate_torch",
    "with_slm_ownership",
]

#: The interleaved-replay validated gs-refine recipe
#: (report/shape_trials/gs_refine_live_20261012.md).  Changing a default here
#: silently abandons that evidence — change it only with new replay evidence.
#:
#: The values live in the optimiser
#: (:data:`ao_shaping.optimizer.wfless.slm_gs_refine.VALIDATED_GS_REFINE_RECIPE`)
#: so the CLI/library side and this GUI preset cannot drift apart; this name is
#: kept because the GUI and its tests import it, and it is a **copy** so a widget
#: default can never mutate the validated table.
GS_REFINE_RECIPE: dict[str, Any] = dict(VALIDATED_GS_REFINE_RECIPE)

#: Model-in-loop budget presets: ``(轮数, Step A 步数, Step B 步数)``.
#: The "标准" row is verbatim the ``docs/slm/slm_model_in_loop.md`` §5 defaults
#: (``n_rounds=6`` / ``step_a_iterations=80`` / ``step_b_iterations=600``); the
#: other two are coarse/fine brackets around it for the operator, not new
#: evidence.  Selecting a preset only overwrites the three budget fields when the
#: selection **changed**, so hand-tuned values are never clobbered.
MIL_BUDGET_PRESETS: dict[str, tuple[int, int, int]] = {
    "快速 2/40/300": (2, 40, 300),
    "标准 6/80/600": (6, 80, 600),
    "长跑 12/160/1200": (12, 160, 1200),
}

#: The GS+SPGD validation recipe budget (SPGD 轮数 / GS 迭代).  Shown read-only —
#: it is validated evidence (see :data:`GS_REFINE_RECIPE`), never a preset.
GS_VALIDATED_BUDGET: tuple[int, int] = (12, 12)

#: The Daheng saturation guard is per-session (the optical path sets it), so it
#: is never a constant here: ``configured_daheng_exposure_limit()`` reads
#: ``AO_DAHENG_MAX_EXPOSURE_MS``, and 0 means the optimizers probe the image.

#: Widget defaults for model-in-loop, keyed by widget field name.  Declared as
#: data (not read from the config at import time) so ``import`` stays free of
#: the optimizer package; :func:`_mil_default` still resolves the *authoritative*
#: value from ``SlmModelInLoopConfig`` and this table only documents intent.
#:
#: ``pupil_center`` here is **not** the widget default any more — the widget
#: starts empty（「留空 = 自动定位」）.  This entry is the last-resort fallback,
#: used only when a session has *never* been calibrated and is what the sidebar
#: back-fills when a calibration skipped the locate step.  It is this bench's
#: measured panel centre, so a different optical path has to calibrate.
MIL_WIDGET_DEFAULTS: dict[str, Any] = {
    "n_rounds": 6,
    "target_shape": "square",
    "target_side": 60,
    "seed": 0,
    "probe_count": 8,
    "probe_spread": 4.0,
    "step_a_iterations": 80,
    "step_b_iterations": 600,
    "holdout_probe_count": 2,
    "freeform_holdout_count": 0,
    "panel_span_px": 900.0,
    "region": 64,
    "far_field_padding": 8,
    "geometry_file": "",
    "pupil_center": "960,690",
}

#: Bare fallback when nothing at all is known about the session.  Owned by
#: :mod:`bench_settings` (the sidebar is what back-fills it) and re-exported here
#: because ``MIL_WIDGET_DEFAULTS`` is the table tests and callers reach for.
MIL_PUPIL_CENTER_FALLBACK: str = bench_settings.PUPIL_CENTER_FALLBACK

#: Shared 目标与评分 caption — one formula for the methods that score a far field,
#: so switching the algorithm never means re-learning what the score means.
_OBJECTIVE_CAPTION = (
    "目标与评分（相机像素；复合分数 = w_pib·PIB + w_unif·1/(1+CV)）。"
    "GS 的权重是 w_pib / w_unif，物理预测模型的是 w_eff / w_uni，迁移学习的是形状 + "
    "目标尺寸。"
)

#: GS 相位网格上限放开后的耗时提示（上界 64 → 128 时的代价必须说清）。
_PHASE_GRID_HELP = (
    "全频随机相位网格边长。验证配方用 12；网格边长翻倍 = 可控自由度 ×4，"
    "而 SPGD 每轮要遍历整个网格，耗时随之上升（本实验系统 12 网格约 1 s/轮）。"
)

#: MIL 探针 σ 的可辨识性上限提示。
_PROBE_SPREAD_HELP = (
    "必须 ≳3 rad，否则拟合对像差不敏感（探针相位 std 要越过 16-bit "
    "量化本底两个数量级才良态）"
)

#: 物理预测模型瞳面网格 / 远场补零的耗时与内存提示。
_MIL_REGION_HELP = (
    "物理预测模型瞳面网格边长，必须是 8 的倍数。网格边长 ×8 = 远场采样点 ×64，"
    "内存与耗时随之上升（本实验系统 64 网格可跑）。"
)
_MIL_PADDING_HELP = (
    "远场补零倍数。物理预测模型默认 8（docs/slm/slm_model_in_loop.md §5），"
    "与 GS 验证值 3 **不合并** —— 两者是各自标定产物的一部分。"
)

#: MIL 探针数放开上限后的耗时提示。
_PROBE_COUNT_HELP = (
    "每轮使用的不同探针相位数。8 是实测默认；每加 1 个探针，Step A 的设备"
    "往返就多一轮（本实验系统一轮探针约 2 s）。"
)
_HOLDOUT_HELP = (
    "验证集探针数（不参与拟合，只用来验相关度）。0 = 不做验证集验证；"
    "注意 Step A 拟合本身要求 ≥ 1，真要关闭验证集请用「全频随机相位验证集探针」= 0。"
)

#: MIL 权重默认值（``SlmModelInLoopParams`` 的 0.6 / 0.4）。
_MIL_W_EFFICIENCY = 0.6
_MIL_W_UNIFORMITY = 0.4


def flush_pending_widgets(method: ShapingMethod) -> dict[str, Any]:
    """Flush this method's queued widget writes into the widget keys.

    Called as the **first** statement of :meth:`ShapingMethod.render_inputs`,
    before any widget is instantiated — that is the only window in which a
    widget key is still writable (Streamlit locks a key for the rest of the
    script run once a widget claims it).
    """
    key = f"{method.prefix}_shp_pending"
    queued = dict(st.session_state.get(key) or {})
    if not queued:
        return {}
    for widget_key, value in queued.items():
        st.session_state[widget_key] = value
    st.session_state[key] = {}
    return queued


def queue_widget_writes(method: ShapingMethod, values: Mapping[str, Any]) -> None:
    """Queue ``values`` to be written into this method's widget keys on flush.

    Needed by anything that reacts to a widget the operator just changed *within
    the same render* — the budget preset is the only such case today.
    """
    key = f"{method.prefix}_shp_pending"
    queued = dict(st.session_state.get(key) or {})
    queued.update({method.wkey(name): value for name, value in values.items()})
    st.session_state[key] = queued


# ---------------------------------------------------------------------------
# Normalized result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapingOutcome:
    """Optimizer-agnostic view of one completed shaping run.

    Attributes:
        panel_phase: Raw-unwrapped-radian phase already embedded on the SLM
            panel grid, ready for ``Santec.display_phase``. ``None`` when the
            optimizer produced no phase.
        best_score: Composite objective of ``panel_phase``.
        flat_score: Composite objective of the flat baseline, when the
            optimizer measured one (used for the improvement caption).
        rows: Per-stage rows for the results table (stage / epoch / score /
            PIB / CV).
        note: One-line verdict (GS bake-off, terminal status, ...).
        saved_dir: Directory the canonical saver wrote to, if saving ran.
        raw_result: The optimizer's own result object, kept for the canonical
            savers that need more than the normalised view (recorder, geometry,
            probe records).  Never inspected by the UI.
    """

    panel_phase: np.ndarray | None
    best_score: float
    flat_score: float | None = None
    rows: tuple[dict[str, Any], ...] = ()
    note: str = ""
    saved_dir: Path | None = None
    raw_result: Any = None


# ---------------------------------------------------------------------------
# Guards (mirrored from the CLI runners)
# ---------------------------------------------------------------------------


def validate_daheng_exposure(cam_type: str, exposure_ms: float) -> str | None:
    """Return an error message when a Daheng exposure is unusable, else None.

    Mirrors the CLI guards: the ceiling is per-session (the optical path sets
    it), taken from ``AO_DAHENG_MAX_EXPOSURE_MS``; when that is unset the
    optimizers probe the image at run time, so the UI cannot bound it.
    model-in-loop may start at 0 (auto from 0.02 ms) because it owns the
    auto-exposure ladder.
    """
    if cam_type == "daheng":
        ceiling = configured_daheng_exposure_limit()
        if ceiling > 0 and exposure_ms > ceiling:
            return (
                f"Daheng 曝光必须 ≤ {ceiling} ms"
                f"（饱和保护，{DAHENG_EXPOSURE_LIMIT_ENV}，见 report/shape_trials/）"
            )
    return None


def validate_daheng_fixed_exposure(cam_type: str, exposure_ms: float) -> str | None:
    """Like :func:`validate_daheng_exposure` but also require > 0 (gs-refine).

    ``gs_refine_runner`` demands an explicit positive exposure for Daheng (the
    ceiling is ``AO_DAHENG_MAX_EXPOSURE_MS``, else probed from the image):
    auto-exposure can observe the *previous* iteration's phase.
    """
    error = validate_daheng_exposure(cam_type, exposure_ms)
    if error is not None:
        return error
    if cam_type == "daheng" and exposure_ms <= 0:
        return (
            "Daheng 需显式设置正曝光（上限取 "
            f"{DAHENG_EXPOSURE_LIMIT_ENV}，缺省时由画面探测；"
            "自动曝光会观察到上一相位残留）"
        )
    return None


_AUTO_EXPOSURE_TOKENS = frozenset({AUTO_EXPOSURE_TOKEN, DEFAULT_EXPOSURE_TOKEN})


def exposure_param(raw: Any) -> float | str:
    """Translate the GUI's numeric ``0`` into the explicit ``auto`` token.

    The runner contract rejects a bare ``0`` so it can never be confused with
    "leave the device alone"; the 会话曝光 widget keeps using ``0`` for "let the
    session decide", which is exactly ``auto``. The translation lives here, at the
    boundary between the widget and the params object, so the CLI stays strict
    while the widget keeps the meaning it has always had.

    Args:
        raw: Widget value -- a number, ``None``, or an already-explicit token.

    Returns:
        A positive float, or one of the two tokens. A malformed string is passed
        through unchanged so the shared resolver reports it, rather than this
        helper inventing a second error message.
    """
    if isinstance(raw, str):
        text = raw.strip()
        return text.lower() if text.lower() in _AUTO_EXPOSURE_TOKENS else text
    if raw is None:
        return AUTO_EXPOSURE_TOKEN
    value = float(raw)
    return AUTO_EXPOSURE_TOKEN if value <= 0 else value


def numeric_exposure(raw: Any) -> float | None:
    """The number to range-check, or ``None`` when the value defers to the device.

    Validation compares against the saturation ceiling, and there is nothing to
    compare when no exposure has been chosen yet.
    """
    value = exposure_param(raw)
    return None if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# Device lifecycle — the optimizer owns the handles while a run is in flight
# ---------------------------------------------------------------------------


def _stop_background_toggle(slm_num: int) -> None:
    """Stop any phase-A/B toggle thread bound to this SLM.

    Lazy import: ``multi_slm_controller`` imports this module at module level,
    so the reverse edge has to be deferred.
    """
    from ao_shaping.gui.slm.multi_slm_controller import _stop_toggle

    _stop_toggle(slm_num)


def _release_slm_handle(slm_num: int) -> None:
    """Close the GUI's SLM handle so the optimizer can open its own.

    Only the target SLM is released; the other SLM (dual layout) stays
    connected.  Session state is cleared the same way ``disconnect_slm`` does
    (minus the rerun), so the sidebar reflects reality during the run.
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is not None and getattr(slm, "is_open", False):
        try:
            slm.close()
        except Exception as exc:  # policy: must not block the reacquire: the handle slot is cleared either way, so a refusing close cannot be allowed to abort the handover
            logger.exception("SLM {} 整形前释放句柄失败: {}", slm_num, exc)
    st.session_state[prefix] = None
    st.session_state[f"{prefix}_connected"] = False
    st.session_state[f"{prefix}_phase_preview"] = None
    st.session_state[f"{prefix}_phase_source"] = "闭环整形运行中"


def _reacquire_slm_handle(slm_num: int) -> Any:
    """Re-open the SLM after the optimizer closed its own handle.

    ``Santec.open()`` reloads the per-serial config file itself (wavelength,
    shift, correction), so the reacquired device carries the same settings the
    sidebar had before the run.  The fresh :class:`Santec` is wrapped in the
    controller-layer overlay proxy (same as ``connect_slm``) so the send
    pipeline overlay (defocus + base phase) is carried on the session handle,
    and the proxy is both stored and returned (the session holds exactly what
    the caller gets back).
    """
    from ao_shaping.gui.slm.multi_slm_controller import _sync_slm_session_state
    from ao_shaping.gui.utils.phase_overlay import OverlaidSantec

    prefix = f"slm{slm_num}"
    try:
        raw = Santec(slm_number=slm_num)
        raw.open()
    except Exception as exc:  # 保留宽泛捕获: 内部调用链 Santec 的失败类型不可枚举
        logger.exception("SLM {} 整形后重新连接失败: {}", slm_num, exc)
        # policy: 与侧边栏「连接」按钮同一套提示: 说明是设备被占用还是 USB 断链,
        # 而不是把原始错误码直接甩给操作员。
        st.error(
            f"SLM {slm_num} 整形后重新连接失败。\n\n"
            f"{describe_santec_error(exc)}\n\n"
            "请在侧边栏手动重连。"
        )
        return None
    slm = OverlaidSantec(raw)
    st.session_state[prefix] = slm
    st.session_state[f"{prefix}_connected"] = True
    _sync_slm_session_state(slm_num, slm)
    return slm


@dataclass
class SlmOwnership:
    """The SLM-handle handover slot yielded by :func:`with_slm_ownership`.

    Inside the ``with`` body the GUI owns **no** handle (``slm`` is ``None`` by
    construction — the optimizer or the calibrator has it).  On exit the
    re-acquired handle lands in :attr:`slm`, so the caller can adopt a phase
    afterwards — the only point at which the GUI owns the device again.

    ``st.session_state`` lives on the ``Streamlit`` object, not on this module,
    so the tests can swap in a plain dict and drive the whole handover offline.
    """

    slm_num: int
    slm: Any = None


@contextmanager
def with_slm_ownership(
    slm_num: int, slot: SlmOwnership | None = None
) -> Iterator[SlmOwnership]:
    """Hand the SLM to whoever needs it for the whole body, then take it back.

    Three steps, in this order:

    1. stop the background phase toggle (it would fight for the panel);
    2. release the GUI's handle — Santec SDK handles are not re-entrant, and
       the optimizers *and* the session calibrator open their own;
    3. ``finally``: re-acquire, so a crash or an abort cannot strand the page
       without a device.

    The whole body also runs under the session ``bench_busy`` flag, which is what
    disables the sidebar optical-path controls while a run or a calibration is
    in flight.

    Both entry points share it: :func:`shaping_session` (a shaping run) and the
    sidebar calibration button.  ``slot`` is optional, so a caller can hold the
    reference *before* entering — that is what lets the ``finally`` of
    :func:`shaping_session` read :attr:`SlmOwnership.slm` even when step 1 or 2
    raised.  Exceptions from the body propagate untouched — surfacing them once
    is the caller's job.
    """
    owned = slot if slot is not None else SlmOwnership(slm_num)
    _stop_background_toggle(slm_num)
    bench_settings.set_bench_busy(True)
    _release_slm_handle(slm_num)
    try:
        yield owned
    finally:
        # Nested, not sequential: a re-acquire that raised would otherwise leave
        # ``bench_busy`` stuck True, which disables the sidebar optical-path
        # controls for the rest of the session with no way to clear them.
        try:
            owned.slm = _reacquire_slm_handle(slm_num)
        finally:
            bench_settings.set_bench_busy(False)


# ---------------------------------------------------------------------------
# Registry + lifecycle decorator
# ---------------------------------------------------------------------------

SHAPING_METHODS: dict[str, type["ShapingMethod"]] = {}


def register_shaping_method(cls: type["ShapingMethod"]) -> type["ShapingMethod"]:
    """Class decorator: register a shaping adapter under its :attr:`key`.

    The class is returned unchanged, so registration is purely additive: a new
    algorithm is one decorated :class:`ShapingMethod` subclass and the page
    picks it up through :data:`SHAPING_METHODS` with no UI edit.
    """
    key = getattr(cls, "key", "")
    if not key:
        raise ValueError(f"{cls.__name__} 必须声明非空 key")
    if key in SHAPING_METHODS:
        raise ValueError(f"闭环整形方法 key 重复: {key!r}")
    SHAPING_METHODS[key] = cls
    return cls


def get_shaping_method(key: str) -> type["ShapingMethod"]:
    """Look up a registered adapter class by its stable :attr:`ShapingMethod.key`."""
    try:
        return SHAPING_METHODS[key]
    except KeyError:
        raise KeyError(
            f"未知闭环整形方法 {key!r}（已注册: {sorted(SHAPING_METHODS)}）"
        ) from None


def shaping_method_labels() -> list[str]:
    """Selector options, in registration order.

    The labels *are* the options: a ``format_func`` that rewrites keys would
    make Streamlit's own widget-state round-trip fail (``format_func(value)``
    is fed back to ``options.index()``), which breaks AppTest and anything else
    that inspects the widget.
    """
    return [cls.label for cls in SHAPING_METHODS.values()]


def get_shaping_method_by_label(label: str) -> type["ShapingMethod"]:
    """Resolve the adapter behind a selector label."""
    for cls in SHAPING_METHODS.values():
        if cls.label == label:
            return cls
    raise KeyError(
        f"未知闭环整形方法标签 {label!r}（已注册: {shaping_method_labels()}）"
    )


def shaping_session(
    run: Callable[..., ShapingOutcome | None],
) -> Callable[..., ShapingOutcome | None]:
    """Method decorator: own the SLM handle for the whole of a run.

    Wraps :meth:`ShapingMethod.run` on top of :func:`with_slm_ownership`, which
    owns the three steps that must not be duplicated per algorithm (stop the
    background toggle, release the GUI handle because the optimizer opens its
    own and the SDK handles are not re-entrant, re-acquire in ``finally``).

    What the decorator adds on top is the **reporting** order:

    1. the phase is displayed only *after* the handle is back (the body by
       contract owns no handle, so the adopted phase could not be sent before);
    2. any escaping exception is logged and surfaced exactly once, here.

    Because the decorator owns both, a new adapter cannot forget either and an
    optimizer-side API change stays inside ``execute()``.
    """

    @functools.wraps(run)
    def wrapper(
        self: ShapingMethod, *args: Any, **kwargs: Any
    ) -> ShapingOutcome | None:
        outcome: ShapingOutcome | None = None
        # Hold the slot before entering: step 1/2 of the handover can raise, and
        # the ``finally`` still has to be able to read ``owned.slm``.
        owned = SlmOwnership(self.slm_num)
        try:
            with with_slm_ownership(self.slm_num, owned):
                outcome = run(self, *args, **kwargs)
            return outcome
        except Exception as exc:  # policy: surfaced to the operator as panel feedback: the finally still releases ownership, so the error text is the operator's record of why a closed-loop run stopped
            logger.exception("闭环整形 [{}] 失败: {}", self.key, exc)
            st.error(f"{self.label} 失败: {exc}")
            return None
        finally:
            if outcome is not None:
                self.show_phase(owned.slm, outcome)

    return wrapper


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


def validate_target_fits_window(values: Mapping[str, Any]) -> str | None:
    """目标方边长必须整个落在读出窗内。

    The reverse clamp (window ≥ target) used to live in the widget bounds, which
    cannot work any more: the sidebar renders the window before the main area
    knows the target side, and Streamlit *raises* on an out-of-range widget value
    rather than clamping it.  So the constraint is checked here, before the run.
    """
    side = int(values.get("target_side", 0) or 0)
    if side <= 0:
        return None  # 0 = 自动推导，由算法按瞳径算
    window_side = int(values.get("cam_size") or 0)
    if window_side <= 0:
        return None  # 关窗 = 整幅读出，目标必然放得下
    if side > window_side:
        return (
            f"目标方边长 {side} px 超过读出窗口边长 {window_side} px —— "
            "在左侧「CCD参数设置」把窗口边长调大，或把目标边长调小"
        )
    return None


def validate_torch(requirement: str) -> str | None:
    """算法需要 torch 时统一的一条拒绝路径（未安装 → 错误文本）。"""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"{requirement}需要 torch（未安装）: {exc}"
    return None


def validate_exposure(
    values: Mapping[str, Any], *, fixed: bool, auto_rejected: str | None = None
) -> str | None:
    """共享曝光校验。

    ``fixed=True``：该算法每轮都观察同一相位，「让设备自己决定」会观察到上一轮
    残留，所以自动曝光是错误（``auto_rejected`` 给出该算法的措辞）。
    ``fixed=False``：允许 ``default``/``auto`` 令牌，只对显式数值做上限检查。
    """
    exposure = numeric_exposure(
        values.get("exposure_time_ms", DEFAULT_EXPOSURE_TOKEN)
    )
    if exposure is None:
        return auto_rejected if fixed else None
    validator = validate_daheng_fixed_exposure if fixed else validate_daheng_exposure
    return validator(str(values.get("cam_type", "")), exposure)


# ---------------------------------------------------------------------------
# 响应式行宽（issue #131 变更 7「双 SLM 布局」）
# ---------------------------------------------------------------------------


def _slms_on_screen() -> int:
    """本页当前渲染几台 SLM 的区块 —— 与 ``multi_slm_controller.main`` 同一判定。

    只读**已存在**的 session_state 信号（``slm{N}`` 句柄与 ``slm{N}_connected``），
    不新增键、不读硬件、不探测窗口尺寸。双 SLM 并排时每台主区列只有 ~500 px，
    一行 4/5 个控件会把 number_input 挤到 ~100 px（标签换行、控件变形）。
    """
    return sum(
        1
        for slm_num in (1, 2)
        if st.session_state.get(f"slm{slm_num}_connected")
        and st.session_state.get(f"slm{slm_num}") is not None
    )


def _param_cells(count: int) -> list[Any]:
    """为 ``count`` 个参数控件铺格子：单 SLM 一行铺满，双 SLM 每行最多 2 个。

    格子顺序 = ``st.columns`` 逐行创建顺序；调用方按返回顺序放置控件，
    ``values[key] = ...`` 的写入顺序就与重排前逐一相同（预设联动依赖这个顺序）。
    """
    per_row = count if _slms_on_screen() < 2 else min(2, count)
    per_row = max(1, per_row)
    cells: list[Any] = []
    while len(cells) < count:
        cells.extend(st.columns(per_row))
    return cells[:count]


class ShapingMethod(ABC):
    """One closed-loop shaping algorithm as seen by the page.

    Subclasses supply the algorithm-specific pieces only — widget set, the
    canonical parameter object, the config builder, and the ``execute()`` call
    that normalises the optimizer's result. Everything shared (device
    lifecycle, error handling, saving, results table, phase display) lives in
    :meth:`run`, which is wrapped by :func:`shaping_session`.
    """

    #: Stable registry key (also the widget-key discriminator).
    key: ClassVar[str] = ""
    #: Selector label shown to the operator.
    label: ClassVar[str] = ""
    #: Short caption under the selector; should state what is verified.
    blurb: ClassVar[str] = ""
    #: Verb used in the run button, e.g. "开始 GS+SPGD 闭环整形".
    action: ClassVar[str] = "开始闭环整形"
    #: True when the optimizer takes the GUI's send-pipeline overlay (载波 / 离焦 /
    #: 底相位) and applies it to the SLM handle it opens itself.
    applies_send_overlay: ClassVar[bool] = True
    #: True when the adopted phase already carries the zero-order carrier
    #: (``ramp_matrix`` baked into the model phase).  Such a phase must be sent
    #: with the sidebar carrier layer **skipped**, or the tilt is applied twice.
    phase_carries_carrier: ClassVar[bool] = False

    def __init__(self, prefix: str, slm_num: int) -> None:
        self.prefix = prefix
        self.slm_num = slm_num

    def wkey(self, suffix: str) -> str:
        """Widget key namespaced by SLM *and* method.

        Switching methods then never mixes one algorithm's remembered inputs
        into another's defaults.
        """
        return f"{self.prefix}_shp_{self.key}_{suffix}"

    # --- algorithm-specific hooks ----------------------------------------

    def render_objective(self, values: dict[str, Any]) -> None:
        """目标与评分：一行 4 列（目标形状/边长 + 两个权重）。

        Both methods render the *same* block title and formula caption — the
        operator should not have to re-learn the page when switching algorithms —
        but the four fields inside are method-specific, so each adapter
        implements this hook.
        """

    def render_budget(self, values: dict[str, Any]) -> None:
        """预算区（轮数 / 迭代）。

        Default is empty: GS+SPGD's budget is validated evidence, so it is shown
        read-only rather than offered as a preset that could be nudged.
        """

    def render_advanced(self, values: dict[str, Any]) -> None:
        """方法专属高级参数（渲染在默认收起的 expander 内）。"""

    def render_inputs(self) -> dict[str, Any]:
        """Render this algorithm's widgets; return the values dict.

        The order is fixed by the page contract: 目标与评分 → 预算 → 高级参数.
        Optical-path settings (camera, window, exposure, carrier, panel centre,
        geometry) are **not** here — they live in the sidebar 「CCD参数设置」
        block and come back through :func:`bench_settings.merge_bench_values`,
        applied by :func:`render_closed_loop_shaping`.
        """
        values: dict[str, Any] = {"slm_number": self.slm_num, "method": self.key}
        flush_pending_widgets(self)
        st.caption(_OBJECTIVE_CAPTION)
        self.render_objective(values)
        self.render_budget(values)
        with st.expander("高级参数", expanded=False):
            self.render_advanced(values)
        st.caption(
            "面板与瞳面几何、相机、读出窗口、曝光与去零级载波在左侧「CCD参数设置」"
            "（一次实验固定；运行直接复用那里的标定结果，换光路后在那里点一次"
            "「标定」即可，不必重填主区）。"
        )
        return values

    @abstractmethod
    def build_params(self, values: Mapping[str, Any]) -> Any:
        """Map the widget values to the canonical runner parameter object."""

    @abstractmethod
    def make_config(self, params: Any, *, debug: bool) -> Any:
        """Map the parameters to the optimizer config (shared with the CLI)."""

    @abstractmethod
    def execute(self, config: Any) -> ShapingOutcome:
        """Run the optimizer and normalise its result.

        The **only** place that calls an optimizer. An optimizer API change is
        confined to this method plus :meth:`save`.
        """

    def save(self, params: Any, outcome: ShapingOutcome, *, debug: bool) -> Path | None:
        """Persist through the canonical saver; return the directory written."""
        return None

    def validate(self, values: Mapping[str, Any]) -> str | None:
        """Return an error message when the inputs must not be run as-is."""
        return None

    def hint_bench_calibration(self, values: Mapping[str, Any]) -> None:
        """「本次直接复用侧栏标定」这一事实必须出声 —— 但只是提示，不是错误。

        三种方法都不在运行时重新标定：入射中心 / sweep 几何 / 曝光只来自侧栏。
        从未标定过时光路会落到 ``PUPIL_CENTER_FALLBACK``（本实验系统的实测值），
        换一条光路则静默失效 —— 正是这一行存在的理由。
        """
        prefix = str(values.get("bench_prefix", self.prefix))
        if bench_settings.detected_state(prefix).get("pupil_center"):
            return
        st.caption(
            "本次直接复用左侧「CCD参数设置」的标定结果：还没有标定记录，"
            f"入射中心用的是兜底实测值 {bench_settings.PUPIL_CENTER_FALLBACK}。"
            f"换过光路请先点「标定 SLM {self.slm_num} 光路」或手填入射中心。"
        )

    def prepare_config(self, config: Any, values: Mapping[str, Any]) -> None:
        """共有：把 GUI 同一条发送管线用的叠加层交给优化器。

        优化器在自己打开 SLM 句柄时装上这一层，所以整轮闭环与最终下发的口径
        一致；CLI 运行 ``send_overlay`` 仍为 ``None``。整条相位自带载波的方法
        （``applies_send_overlay = False``）不接这一层，否则倾斜会加两遍。
        """
        if not self.applies_send_overlay:
            return
        prefix = str(values.get("bench_prefix", self.prefix))
        config.send_overlay = bench_settings.build_send_overlay(prefix)

    # --- shared template --------------------------------------------------

    @shaping_session
    def run(
        self, values: Mapping[str, Any], *, debug: bool = False
    ) -> ShapingOutcome | None:
        """Validate → build → execute → save → summarise.

        Wrapped by :func:`shaping_session`, which owns the device lifecycle,
        the error surfacing and the final phase display; this method owns only
        the reporting order.  Note it never touches the SLM: by contract the
        optimizer holds the device for the whole body.
        """
        error = self.validate(values)
        if error is not None:
            st.error(error)
            return None

        params = self.build_params(values)
        config = self.make_config(params, debug=debug)
        self.prepare_config(config, values)
        # Snapshot the optical path **as it is right now**, before any device
        # moves: afterwards a re-acquire + re-render would read different values.
        prefix = str(values.get("bench_prefix", self.prefix))
        values.setdefault("method", self.key)
        bench_settings.set_effective_state(prefix, effective_bench_snapshot(values))

        logger.info("GUI 闭环整形启动 [{}]: {}", self.key, type(params).__name__)
        outcome = self.execute(config)
        outcome = replace(outcome, saved_dir=self.save(params, outcome, debug=debug))
        write_bench_snapshot(outcome.saved_dir, prefix)
        self.render_results(outcome)
        return outcome

    def render_results(self, outcome: ShapingOutcome) -> None:
        """结果区：生效光路参数快照 → 表格 → 保存目录 → 结论。

        The optical-path snapshot comes **first** because it is the part that
        answers "which optical path produced the numbers below" — the results
        table carries no geometry information at all.
        """
        st.caption(snapshot_summary_line(str(self.prefix)))
        self.summarise(outcome)

    @contextmanager
    def _phase_send_overlay(self, slm: Any) -> Iterator[None]:
        """下发最佳相位期间的发送叠加层。

        只有 ``phase_carries_carrier`` 的方法（迁移学习）需要临时摘掉侧栏载波层；
        换层成功与否都不许丢相位，且退出时立刻恢复完整叠加层，否则操作员之后的
        每一次手动下发都会丢掉载波。
        """
        if not self.phase_carries_carrier or slm is None:
            yield
            return
        try:
            apply_overlays_to_handle(
                slm,
                bench_settings.build_send_overlay(
                    self.prefix, slm, include_carrier=False
                ),
            )
        except Exception as exc:  # policy: 换层失败不该丢掉相位：切换叠加层失败只记 warning，相位仍按当前叠加下发
            logger.warning("下发接受相位前切换叠加层失败（按当前叠加继续）: {}", exc)
        try:
            yield
        finally:
            try:
                apply_overlays_to_handle(
                    slm, bench_settings.build_send_overlay(self.prefix, slm)
                )
            # policy: restoring the send overlay after the phase send is best effort: failing here leaves the panel overlay state uncertain but must not mask the send that already happened
            except Exception as exc:
                logger.warning("恢复发送叠加层失败: {}", exc)

    def show_phase(self, slm: Any, outcome: ShapingOutcome | None) -> None:
        """Display the adopted phase once the GUI owns the SLM again.

        Called by :func:`shaping_session` after re-acquiring the handle, so the
        raw-radian panel phase goes through the normal driver pipeline
        (correction + LUT + shift) exactly like every other phase in this GUI.
        Uses ``display_data`` directly (same as optimizer modules) for explicit
        control over memory slot rotation and internal memory mode.
        """
        if outcome is None or outcome.panel_phase is None:
            return
        with self._phase_send_overlay(slm):
            try:
                phase_gray = slm.create_phase_from_array(outcome.panel_phase)
                slm.display_data(phase_gray, memory_mode=MEMORY_MODE_INTERNAL)
                refresh_phase_preview(self.slm_num)
                max_gray = getattr(slm, "_max_gray", None) or Santec.MAX_GRAYSCALE_VALUE
                st.success(f"最佳相位已下发（raw 弧度走驱动管线，2π={max_gray}）")
            except Exception as exc:  # policy: surfaced to the operator: a failed best-phase send must not look like a completed one, and the run returns instead of falling through to the success message
                logger.exception("下发最佳相位失败: {}", exc)
                st.error(f"最佳相位下发失败: {exc}")
                return

    def summarise(self, outcome: ShapingOutcome) -> None:
        """Render the results table plus the verdict line (shared)."""
        if outcome.rows:
            st.dataframe(list(outcome.rows), hide_index=True)
        if outcome.saved_dir is not None:
            st.caption(f"结果已保存: {outcome.saved_dir}")
        if outcome.note:
            st.caption(outcome.note)

    # --- shared widget helpers -------------------------------------------

    def num_input(
        self,
        key: str,
        label: str,
        default: Any,
        lo: float,
        hi: float,
        step: float,
        help: str | None = None,
    ) -> float:
        """One numeric widget whose types always agree.

        Streamlit rejects a mixed int/float widget (``value`` int against
        ``min_value`` float), which is easy to hit when the widget set is
        table-driven. The value type follows the *bounds*, so a table row only
        has to spell out its bounds.
        """
        is_float = (
            isinstance(lo, float) or isinstance(hi, float) or isinstance(step, float)
        )
        value = float(default) if is_float else int(default)
        return st.number_input(
            label,
            min_value=lo,
            max_value=hi,
            value=value,
            step=step,
            key=self.wkey(key),
            help=help,
        )

    def render_extra_row(
        self, values: dict[str, Any], extra: Mapping[str, dict[str, Any]]
    ) -> None:
        """One row of method-specific numeric inputs (e.g. frames per measurement).

        相机类型 / ID / 像元间距 / 曝光 / 读出窗口 / 载波 **不在这里** ——
        它们已经搬到侧栏「CCD参数设置」区块
        (:mod:`ao_shaping.gui.slm.bench_settings`)，因为一次实验内固定，本来
        就不该每跑一次就重填一遍。真正喂给优化器的值经
        :func:`merge_bench_values` 回来，由
        :func:`render_closed_loop_shaping` 在 ``render_inputs`` 之后并入。
        """
        for cell, (name, spec) in zip(
            _param_cells(len(extra)), extra.items(), strict=True
        ):
            with cell:
                values[name] = st.number_input(**spec, key=self.wkey(name))

    def render_target_side(
        self,
        values: dict[str, Any],
        default: int,
        *,
        lo: int,
        step: int,
        help: str | None = None,
    ) -> int:
        """目标方边长：上界跟着侧栏窗口走，下界固定 + 小值告警。

        上界取当前窗口边长（关窗 = 传感器短边），因为目标方斑必须整个落在读出
        窗内。这里**不再**用 ``cam_size_min = max(16, target_side)`` 去反向夹
        窗口：侧栏先渲染，拿不到主区的目标边长，而 Streamlit 对「min 变化后越界」
        是抛异常而不是夹取 —— 所以那个交叉校验挪到 :meth:`validate`（运行前）。

        控件**之前**先渲染 :meth:`render_link_assist` 的「远场尺寸换算」popover：
        它紧挨着目标方边长，换算结果一键就能填进来（填写走 pending-flush，见该
        方法的 docstring）。
        """
        # 上界取当前窗口边长（关窗 = 传感器短边），因为目标方斑必须整个落在读出
        # 窗内。但窗口可以比目标默认边长**小**（操作员刚把窗口调小，或窗口默认值
        # 还没建立），而 Streamlit 对 ``value > max_value`` 是**抛异常**而不是夹取
        # —— 所以 default 必须与上界一起夹到合法区间。这与 docstring 里把交叉校验
        # 挪到 :meth:`validate` 是同一个理由，只是那条只管 min/hi，管不到 value。
        window_side = max(int(lo), bench_settings.bench_window_side())
        default = max(int(lo), min(int(default), window_side))
        self.render_link_assist(lo=lo, window_side=window_side, current_side=default)
        value = self.num_input(
            "target_side", "目标方边长", default, lo, window_side, step, help=help
        )
        side = int(value)
        if 0 < side < MIL_TARGET_SIDE_MIN_PX:
            st.warning(
                f"目标方边长 {side} px < {MIL_TARGET_SIDE_MIN_PX} px，"
                "目标框在读出噪声里没有意义"
            )
        return side

    def link_wkey(self, name: str) -> str:
        """换算 popover 自己的 widget key（统一 ``link_`` 前缀）。

        与算法参数键分开：这个 popover 是换算器，不是优化器输入。将来某个参数
        若与本助手的字段同名（``wavelength_nm`` 之类），也不会互相覆盖。
        """
        return self.wkey(f"link_{name}")

    def render_link_assist(
        self, *, lo: int, window_side: int, current_side: int
    ) -> None:
        """「远场尺寸换算」popover：望远镜远场光斑 ↔ 诊断相机像素。

        操作员手上通常只有望远镜一侧的指标（1 km 处 10 cm 光斑），而这里要填的
        是诊断相机上的像素数。这个助手把两者连起来，并能一键把换算结果填进
        :meth:`render_target_side` 的目标方边长。

        换算**全部**走 :mod:`ao_shaping.utils.wavefront.link_design`（issue #204
        的单一事实源，见 :mod:`ao_shaping.gui.slm.link_design_assist`）；本方法
        只摆控件、显示结果、处理「填入」与「恢复锁定参数」。

        为什么「填入」不直接写 widget key：Streamlit 只允许在控件实例化**之前**
        写它的 key，而按钮被点击时 ``target_side`` 早已在本次运行里创建。所以两个
        按钮都走本模块既有的 pending-flush —— :func:`queue_widget_writes` 排队 +
        ``st.rerun()``，下一次 :meth:`render_inputs` 的第一行
        :func:`flush_pending_widgets` 在任何控件之前把值写进 key（与「预算预设」
        同一条路径）。

        Args:
            lo: 目标方边长的下界（GS 允许 0 = 由照明瞳自动推导）。
            window_side: 当前读出窗边长，也是目标方边长的上界。
            current_side: 目标方边长控件的当前值（反向换算的默认值）。
        """
        state = st.session_state
        fields = {
            field.name: float(state.get(self.link_wkey(field.name), field.default))
            for field in LINK_FIELDS
        }
        params = build_params(**fields)
        current_px = int(state.get(self.wkey("target_side"), current_side))

        with st.popover(
            "远场尺寸换算",
            help="把目标处的远场光斑边长换算成诊断相机上的像素，可直接填入目标方边长。",
        ):
            st.caption(
                "远场边长 ↔ 相机上的方斑边长。换算用 f_eff = f_c·M_t，口径取出瞳直径。"
            )
            unit_col, value_col = st.columns([1, 2])
            unit = unit_col.selectbox(
                "单位",
                options=list(FAR_UNITS),
                index=list(FAR_UNITS).index(str(DEFAULT_FAR_SIDE["unit"])),
                key=self.link_wkey("far_unit"),
                label_visibility="collapsed",
            )
            far_value = value_col.number_input(
                "远场边长",
                min_value=0.001,
                value=bench_settings.widget_default(
                    self.link_wkey("far_value"), float(DEFAULT_FAR_SIDE["value"])
                ),
                step=1.0,
                key=self.link_wkey("far_value"),
            )
            far_side_m = far_side_to_m(float(far_value), str(unit))

            summary = compute_link_summary(
                far_side_m, window_side_px=window_side, params=params
            )
            for label, text in summary_rows(summary):
                st.markdown(f"- **{label}**：{text}")
            st.caption(f"换算系数 L/(f_c·M_t) = {summary['scale']:.0f}")

            px = int(round(summary["ccd_side_px"]))
            filled = min(max(px, int(lo)), int(window_side))
            if filled != px:
                st.warning(
                    f"远场 {format_far_length(far_side_m)} 换算得 {px} px，"
                    f"超出当前读出窗能容纳的 {int(lo)}–{int(window_side)} px；"
                    f"已夹到 {filled} px。这个物理边长超出相机视场："
                    "要完整覆盖它，得先开大读出窗或提高望远镜放大率。"
                )
            if st.button("填入目标方边长", key=self.link_wkey("apply"), type="primary"):
                # 走本模块既有的 pending-flush（``queue_widget_writes`` →
                # ``flush_pending_widgets``），而不是在这一帧直接写 widget key：
                # 排队 + rerun 后，下一次 ``render_inputs`` 的第一行在任何控件
                # 实例化之前把值写进 key，与「预算预设」是同一条路径。
                queue_widget_writes(self, {"target_side": int(filled)})
                st.rerun()

            st.divider()
            st.caption("反向：相机像素 → 远场")
            rev_px = st.number_input(
                "CCD 边长 (px)",
                min_value=1,
                value=bench_settings.widget_default(
                    self.link_wkey("rev_px"), max(1, current_px)
                ),
                step=1,
                key=self.link_wkey("rev_px"),
            )
            rev = compute_reverse(float(rev_px), params=params)
            st.caption(
                f"{int(rev_px)} px = {format_ccd_length(rev['ccd_side_m'])}"
                f" → 远场 {format_far_length(rev['far_side_m'])}"
            )

            with st.expander("光路参数"):
                st.caption("默认值 = issue #204 的锁定设计点；改动只影响本助手的换算。")
                left, right = st.columns(2)
                for index, field in enumerate(LINK_FIELDS):
                    cell = left if index % 2 == 0 else right
                    cell.number_input(
                        field.label,
                        min_value=field.lo,
                        max_value=field.hi,
                        value=bench_settings.widget_default(
                            self.link_wkey(field.name), field.default
                        ),
                        step=field.step,
                        key=self.link_wkey(field.name),
                    )
                if st.button("恢复锁定参数", key=self.link_wkey("reset")):
                    # queue_widget_writes 自己加 ``wkey`` 前缀，所以这里传的键名
                    # 要与 :meth:`link_wkey` 的前缀一致（``link_<字段>``）。
                    queue_widget_writes(
                        self,
                        {f"link_{field.name}": field.default for field in LINK_FIELDS},
                    )
                    st.rerun()

            st.divider()
            self._render_sampling_assist(lo=lo, window_side=window_side)

    def _render_sampling_assist(self, *, lo: int, window_side: int) -> None:
        """popover 的第二段：主路方斑 ``S`` ↔ 采样相机像素 + 物理可达性判据。

        与上一段（望远镜远场）**口径不同、不可互换**：上一段量的是一公里外目标的
        角尺度，这一段量的是主路焦面上已有的方斑被采样透镜以 ``f_sample/f_main``
        再次成像后的大小。换算与全部判据在
        :mod:`ao_shaping.utils.wavefront.sampling_limit`（单一实现），字段表与文本
        在 :mod:`ao_shaping.gui.slm.sampling_assist`；这里只摆控件、显示、填入。

        「填入」把 ``S_CCD`` 折成采样相机像元数，与本页「目标方边长」同一个单位 ——
        前提是侧栏那个读出窗就是采样相机。超出窗口时照上一段的做法**夹取 + 警告**，
        不静默改数值。
        """
        sampling_state = st.session_state
        fields = {
            field.name: float(
                sampling_state.get(self.link_wkey(field.name), field.default)
            )
            for field in sampling_assist.SAMPLING_FIELDS
        }
        # 相机像元是相机侧的唯一事实源（侧栏设置），本段不另设控件。
        camera_pixel_um = float(
            sampling_state.get(
                "camera_pixel_um", bench_settings.DEFAULT_CAMERA_PIXEL_UM
            )
        )
        st.caption(
            "主路方斑 S ↔ 采样相机像素：S_CCD = S·f_sample/f_main"
            "（N 与 w 不进这条式子，只决定下面的可达性判据）。"
        )
        left, right = st.columns(2)
        for index, field in enumerate(sampling_assist.SAMPLING_FIELDS):
            cell = left if index % 2 == 0 else right
            cell.number_input(
                field.label,
                min_value=field.lo,
                max_value=field.hi,
                value=bench_settings.widget_default(
                    self.link_wkey(field.name), field.default
                ),
                step=field.step,
                key=self.link_wkey(field.name),
                help=field.help or None,
            )
        setup = sampling_assist.build_setup(
            **fields, camera_pixel_m=camera_pixel_um * 1.0e-6
        )
        summary = sampling_assist.compute_sampling_summary(
            evaluate_sampling(setup), window_side_px=int(window_side)
        )
        for label, text in sampling_assist.summary_rows(summary):
            st.markdown(f"- **{label}**：{text}")
        for label, text in sampling_assist.violation_rows(summary):
            st.warning(f"{label}：{text}")

        px = int(round(float(summary["side_on_camera_px"])))
        filled = min(max(px, int(lo)), int(window_side))
        if st.button(
            "填入目标方边长（采样相机）",
            key=self.link_wkey("sampling_apply"),
            type="primary",
        ):
            queue_widget_writes(self, {"target_side": int(filled)})
            st.rerun()

        if st.button(
            "恢复采样光路参数", key=self.link_wkey("sampling_reset")
        ):
            # 只恢复**光路参数**：目标边长 S 每次实验都不同，复位它会把操作员
            # 刚测出来的目标抹掉。
            queue_widget_writes(
                self,
                {
                    f"link_{field.name}": field.default
                    for field in sampling_assist.SAMPLING_FIELDS
                    if field.is_optical
                },
            )
            st.rerun()

    def common_param_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """所有优化器参数对象共有的光路字段（一个映射，三份参数对象共用）。

        ``blaze_period_px`` 恒为 -1：载波由发送叠加层承载并在优化器打开句柄时
        装上，优化器自己再加一次会让相位斜率翻倍。
        """
        return {
            "camera_pixel_um": float(
                st.session_state.get(
                    "camera_pixel_um",
                    values.get("camera_pixel_um", bench_settings.DEFAULT_CAMERA_PIXEL_UM),
                )
            ),
            "cam_type": str(values.get("cam_type", "daheng")),
            "cam_id": int(values.get("cam_id", 0)),
            "cam_size": int(values.get("cam_size", bench_settings.DEFAULT_WINDOW_SIDE_PX)),
            "pupil_center": str(
                values.get("pupil_center", bench_settings.PUPIL_CENTER_FALLBACK)
            ),
            "blaze_period_px": -1,
            "blaze_direction": str(values.get("blaze_direction", "vertical")),
        }


# ---------------------------------------------------------------------------
# 生效光路参数快照
# ---------------------------------------------------------------------------

#: Snapshot rows as ``(三态字段名, values 表里的键)``.  Fixed order so two runs
#: can be diffed line by line.  ``None`` marks a value that has no three-state
#: counterpart (the carrier switches, the camera type) — it is still recorded,
#: just always labelled ``unset``.
_SNAPSHOT_FIELDS: tuple[tuple[str, str], ...] = (
    ("cam_type", "cam_type"),
    ("cam_id", "cam_id"),
    ("cam_crop_enabled", "cam_crop_enabled"),
    ("cam_size", "cam_size"),
    ("camera_pixel_um", "camera_pixel_um"),
    ("exposure_ms", "exposure_time_ms"),
    ("pupil_center", "pupil_center"),
    ("panel_span_px", "panel_span_px"),
    ("geometry_file", "geometry_file"),
    ("blaze_period_px", "blaze_period_px"),
    ("blaze_direction", "blaze_direction"),
)


def _snapshot_field_state(prefix: str, name: str) -> str:
    """三态 for a snapshot field, across both namespaces.

    A snapshot mixes experiment-level values (exposure, window side, pixel pitch)
    and per-SLM values (pupil centre, model span, geometry file), so both field
    tables have to be consulted — otherwise every experiment-level field would
    look ``unset`` and the manual/detected split would be lost exactly where the
    operator most needs it.
    """
    for namespace in (prefix, ""):
        if name in bench_settings.bench_fields(namespace):
            return bench_settings.field_state(namespace, name)
    return "unset"


def effective_bench_snapshot(values: Mapping[str, Any]) -> dict[str, Any]:
    """本次运行**生效**的光路参数 + 每个字段的三态标记。

    快照是判据：某个参数的结论永远能追溯到「跑的时候到底用的什么光路」，
    而不是事后在侧栏看到的那一屏。
    """
    prefix = str(values.get("bench_prefix", "slm1"))
    detected = bench_settings.detected_state(prefix)
    snap: dict[str, Any] = {
        "slm_number": int(values.get("slm_number", values.get("bench_slm_number", 1))),
        "method": str(values.get("method", "")),
        "geometry_source": str(values.get("bench_geometry_source", "auto")),
    }
    for name, key in _SNAPSHOT_FIELDS:
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, tuple):
            value = list(value)
        snap[name] = value
        snap[f"{name}__state"] = _snapshot_field_state(prefix, name)
    for read_only in ("geometry_correlation", "calibrated_at", "evidence_dir"):
        if read_only in detected:
            snap[read_only] = detected[read_only]
    return snap


def snapshot_summary_line(prefix: str) -> str:
    """结果区那行「本次运行生效的光路参数」说明。"""
    snap = bench_settings.effective_state(prefix)
    if not snap:
        return "本次运行生效的光路参数：无快照（未跑过）"
    manual = [
        k[: -len("__state")]
        for k in snap
        if k.endswith("__state") and snap[k] == "manual"
    ]
    detected = [
        k[: -len("__state")]
        for k in snap
        if k.endswith("__state") and snap[k] == "detected"
    ]
    parts = [f"{len(detected)} 项探测"]
    if manual:
        parts.append(
            f"{len(manual)} 项手动（" + " ".join(f"{k}={snap[k]}" for k in manual) + ")"
        )
    saved = bench_settings.effective_key(prefix)
    return f"本次运行生效的光路参数：{' / '.join(parts)}。快照见 {saved}"


def write_bench_snapshot(saved_dir: Path | None, prefix: str) -> Path | None:
    """把生效快照落到 ``{saved_dir}/params.json``。

    只有目录已存在才写 —— saver 刚建好的目录必然存在，而一个不存在的路径说明
    ``save()`` 返回了别处（测试替身），这时不该凭空建目录。
    """
    if saved_dir is None:
        return None
    snap = bench_settings.effective_state(prefix)
    if not snap:
        return None
    directory = Path(saved_dir)
    if not directory.is_dir():
        logger.debug("运行目录不存在，跳过光路快照: {}", directory)
        return None
    target = directory / "params.json"
    if target.exists():
        logger.debug("{} 已存在，保留原有内容", target)
        return target
    try:
        target.write_text(
            json.dumps(snap, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("写光路快照失败 {}: {}", target, exc)
        return None
    return target


# ---------------------------------------------------------------------------
# Result normalisation (pure, unit-testable)
# ---------------------------------------------------------------------------


def _result_row(stage: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise one recorder row into the results-table schema."""
    return {
        "stage": stage,
        "epoch": str(row.get("epoch", "-")),
        "score": round(float(row["score"]), 4),
        "PIB": round(float(row.get("pib", float("nan"))), 4),
        "CV": round(float(row.get("cv", float("nan"))), 4),
    }


def outcome_from_gs_recorder(recorder: Any) -> ShapingOutcome:
    """Translate a gs-refine :class:`Recorder` into a :class:`ShapingOutcome`.

    Kept as a module-level function (not a method) so it can be unit-tested
    against a hand-built Recorder without instantiating an adapter.
    """
    history = list(recorder.history)
    best_row = max(history, key=lambda r: r["score"])
    rows = [
        _result_row(row["stage"], row)
        for row in history
        if row["stage"] in ("flat", "gs")
    ]
    rows.append(_result_row(f"best ({best_row['stage']})", best_row))
    scores = {row["stage"]: float(row["score"]) for row in history}
    flat, gs = scores.get("flat"), scores.get("gs")
    if flat is not None and gs is not None:
        note = (
            f"GS 预矫正优于平场（{gs:.4f} > {flat:.4f}），已采用"
            if gs > flat
            else f"GS 预矫正未胜平场（{gs:.4f} ≤ {flat:.4f}），从平场继续"
            "（检查相机像素间距 / 光斑半径）"
        )
    else:
        note = ""
    return ShapingOutcome(
        panel_phase=getattr(recorder, "best_phase", None),
        best_score=float(best_row["score"]),
        flat_score=flat,
        rows=tuple(rows),
        note=note,
        raw_result=recorder,
    )


def outcome_from_model_in_loop(result: Any, config: Any) -> ShapingOutcome:
    """Translate a :class:`ModelInLoopResult` into a :class:`ShapingOutcome`.

    The adopted phase lives on the coarse *model* grid, so it is embedded onto
    the panel disc here — the one place that knows about ``panel_span_px`` and
    the panel-frame ``pupil_center``.
    """
    from ao_shaping.tools.kernels.bench_kernels import phase_to_panel

    best_phase = getattr(result, "best_phase", None)
    panel_phase = None
    if best_phase is not None:
        panel_phase = phase_to_panel(
            np.asarray(best_phase, dtype=np.float64),
            disc_radius=int(round(config.panel_span_px / 2.0)),
            pupil_center=(int(config.pupil_center[0]), int(config.pupil_center[1])),
        )
        blaze_params = getattr(result, "blaze_params", None)
        if blaze_params is not None:
            from ao_shaping.utils.wavefront.zero_order_overlay import (
                apply_zero_order_overlay,
            )

            panel_phase = apply_zero_order_overlay(panel_phase, blaze_params)
    flat_score = getattr(result, "flat_score", None)
    best_score = float(getattr(result, "best_score", float("nan")))
    rows: tuple[dict[str, Any], ...] = ()
    if flat_score is not None:
        rows = (
            {"stage": "flat", "epoch": "-", "score": round(float(flat_score), 4)},
            {
                "stage": f"best ({getattr(result, 'status', '?')})",
                "epoch": "-",
                "score": round(best_score, 4),
            },
        )
    note = (
        f"状态 {getattr(result, 'status', '?')}；目标框 "
        f"{getattr(result, 'target_side_camera', '?')} 相机px "
        f"({getattr(result, 'target_side_model', '?')} 模型px)"
        "（尚未达到验收的均匀方斑口径，见 model_in_loop_live_20261013.md）"
    )
    return ShapingOutcome(
        panel_phase=panel_phase,
        best_score=best_score,
        flat_score=float(flat_score) if flat_score is not None else None,
        rows=rows,
        note=note,
        raw_result=result,
    )


# ---------------------------------------------------------------------------
# Adapter: GS + SPGD  (the validated recipe)
# ---------------------------------------------------------------------------


@register_shaping_method
class GsRefineMethod(ShapingMethod):
    """GS warm start (bake-off) + sensorless SPGD refinement.

    Same code path as the CLI ``slm-gs-refine``; the widget defaults are that
    command's interleaved-replay validated recipe. The optimizer is still being
    developed — only :meth:`execute` / :meth:`save` know about
    ``SlmGsRefineConfig`` and :class:`Recorder`.
    """

    key: ClassVar[str] = "gs-refine"
    label: ClassVar[str] = "GS+SPGD 闭环整形（已验证配方）"
    action: ClassVar[str] = "开始 GS+SPGD 闭环整形"
    blurb: ClassVar[str] = (
        "交错重放验证配方（flat 0.534 → GS 0.767 → best 0.774，2026-10-12 / "
        "2026-10-14 两次复核）。GS 预矫正走对比择优：实测不优于平场就丢弃，"
        "从平场继续。"
    )

    def render_objective(self, values: dict[str, Any]) -> None:
        """GS 目标与评分，一行 4 列（与 MIL 共用区块标题与公式 caption）。"""
        r = GS_REFINE_RECIPE
        cells = _param_cells(4)
        with cells[0]:
            values["target_side"] = self.render_target_side(
                values,
                int(r["target_side"]),
                lo=0,
                step=1,
                help="0 = 由照明瞳自动推导",
            )
        for cell, (key, label, lo, hi, step, help_) in zip(
            cells[1:],
            (
                ("side_factor", "边长因子", 0.1, 5.0, 0.1, None),
                ("w_pib", "权重 w_pib", 0.0, 2.0, 0.05, None),
                ("w_unif", "权重 w_unif", 0.0, 2.0, 0.05, None),
            ),
            strict=True,
        ):
            with cell:
                values[key] = self.num_input(
                    key, label, r[key], lo, hi, step, help=help_
                )

    def render_budget(self, values: dict[str, Any]) -> None:
        """GS **不给**预算预设：12/12 是已验证配方，不是可随手改的旋钮。

        Editing it silently abandons the interleaved-replay evidence in
        ``report/shape_trials/gs_refine_live_20261012.md``, so the two budget
        fields show up under 高级参数 with the validated values in their caption.
        """
        st.caption(
            f"已验证配方预算：SPGD {GS_VALIDATED_BUDGET[0]} 轮 / "
            f"GS {GS_VALIDATED_BUDGET[1]} 迭代（report/shape_trials/）"
        )

    def render_advanced(self, values: dict[str, Any]) -> None:
        """GS 专属高级参数（默认收起）。"""
        r = GS_REFINE_RECIPE
        st.caption("SPGD 细化（粗网格全频随机相位）")
        spgd_fields = (
            ("epochs", "SPGD 轮数", 1, 400, 1, None),
            ("phase_grid", "相位网格", 2, 128, 1, _PHASE_GRID_HELP),
            ("delta", "扰动 δ (rad)", 0.0, 2.0, 0.05, None),
            ("seed", "随机种子", 0, 99999, 1, None),
        )
        cells = _param_cells(5)
        # 第 5 格留给下面的 confirm_best 复选框, 这里只配前 len(spgd_fields) 格。
        for cell, (key, label, lo, hi, step, help_) in zip(
            cells[: len(spgd_fields)], spgd_fields, strict=True
        ):
            with cell:
                values[key] = self.num_input(
                    key, label, r[key], lo, hi, step, help=help_
                )
        with cells[4]:
            values["confirm_best"] = st.checkbox(
                "confirm-best 复测",
                value=bool(r["confirm_best"]),
                key=self.wkey("confirm_best"),
                help="破纪录候选重新测量确认，抑制噪声假提升",
            )

        st.caption("GS 预矫正（对比择优：仅实测优于平场才采用）")
        c1, c2, c3 = _param_cells(3)
        with c1:
            values["gs_warm_start"] = st.checkbox(
                "启用 GS 预矫正",
                value=bool(r["gs_warm_start"]),
                key=self.wkey("gs_warm_start"),
            )
        with c2:
            values["gs_iters"] = self.num_input(
                "gs_iters", "GS 迭代", r["gs_iters"], 1, 2000, 1
            )
        with c3:
            gs_methods = (
                "plain",
                "weighted",
                "hio",
                "hio-weighted",
                "adaptive-feedback",
            )
            values["gs_method"] = st.selectbox(
                "GS 变体",
                options=gs_methods,
                index=gs_methods.index(str(r["gs_method"])),
                key=self.wkey("gs_method"),
                help="实验对照结论: plain 最优 (gs_refine_live_20261012.md)",
            )

        st.caption("数值细节")
        cells = _param_cells(3)
        with cells[0]:
            values["far_field_padding"] = self.num_input(
                "far_field_padding",
                "远场补零",
                r["far_field_padding"],
                1,
                64,
                1,
                help="GS 验证值 3；与 MIL 的默认值 8 不同，故两者不合并",
            )
        with cells[1]:
            values["panel_pixel_um"] = self.num_input(
                "panel_pixel_um",
                "面板像素间距 (μm)",
                r["panel_pixel_um"],
                1.0,
                50.0,
                0.1,
                help=(
                    f"面板模型用**像素间距** {SLM_PITCH_UM} μm（= 器件常量 PITCH_UM）；"
                    f"与像素有效口径 {SLM_PIXEL_SIZE_UM} μm 相差 0.2 μm，那是真实器件"
                    "几何、不是待修误差，改动前先看 report/shape_trials/"
                ),
            )
        with cells[2]:
            self.render_extra_row(
                values,
                {
                    "n_eval_frames": {
                        "label": "每测量均帧",
                        "min_value": 1,
                        "max_value": 16,
                        "value": int(r["n_eval_frames"]),
                        "step": 1,
                    }
                },
            )

    def build_params(self, values: Mapping[str, Any]) -> SlmGsRefineParams:
        r = GS_REFINE_RECIPE
        w_pib = values.get("w_pib", values.get("w_efficiency", r.get("w_pib", 0.5)))
        w_unif = values.get(
            "w_unif",
            values.get("w_uniformity", values.get("w_uni", r.get("w_unif", 0.5))),
        )
        return SlmGsRefineParams(
            target_side=int(values.get("target_side", r.get("target_side", 100))),
            side_factor=float(values.get("side_factor", r.get("side_factor", 1.0))),
            w_pib=float(w_pib),
            w_unif=float(w_unif),
            phase_grid=int(values.get("phase_grid", r.get("phase_grid", 12))),
            delta=float(values.get("delta", r.get("delta", 0.35))),
            confirm_best=bool(values.get("confirm_best", r.get("confirm_best", True))),
            epochs=int(values.get("epochs", r.get("epochs", 12))),
            seed=int(values.get("seed", r.get("seed", 0))),
            gs_warm_start=bool(
                values.get("gs_warm_start", r.get("gs_warm_start", True))
            ),
            gs_iters=int(values.get("gs_iters", r.get("gs_iters", 12))),
            gs_method=str(values.get("gs_method", r.get("gs_method", "plain"))),
            panel_pixel_um=float(
                values.get("panel_pixel_um", r.get("panel_pixel_um", 8.0))
            ),
            beam_radius_px=float(
                values.get("beam_radius_px", r.get("beam_radius_px", 450.0))
            ),
            far_field_padding=int(
                values.get("far_field_padding", r.get("far_field_padding", 3))
            ),
            exposure_time_ms=exposure_param(values.get("exposure_time_ms", 0.1)),
            slm_number=self.slm_num,
            n_eval_frames=int(values.get("n_eval_frames", r.get("n_eval_frames", 4))),
            **self.common_param_values(values),
        )

    def make_config(self, params: SlmGsRefineParams, *, debug: bool) -> Any:
        return build_gs_refine_config(params, debug=debug)

    def prepare_config(self, config: Any, values: Mapping[str, Any]) -> None:
        """Wire the optimizer's ``callback(epoch, score)`` to a progress bar.

        The run is synchronous on the main Streamlit thread, so updating
        widgets from the callback is safe.
        """
        epochs = max(1, int(values["epochs"]))
        bar = st.progress(0.0, text="准备：平场基线 → GS 预矫正 → SPGD 细化…")

        def _callback(epoch: int, score: float) -> None:
            bar.progress(
                min(1.0, (epoch + 1) / epochs),
                text=f"epoch {epoch}/{epochs - 1} — score {score:.4f}",
            )

        config.callback = _callback
        super().prepare_config(config, values)

    def execute(self, config: Any) -> ShapingOutcome:
        from ao_shaping.optimizer.wfless.slm_gs_refine import optimize_slm_gs_refine

        return outcome_from_gs_recorder(optimize_slm_gs_refine(config))

    def save(
        self, params: SlmGsRefineParams, outcome: ShapingOutcome, *, debug: bool
    ) -> Path | None:
        recorder = outcome.raw_result
        if recorder is None:
            return None
        result = save_gs_refine_results(
            recorder, params, root_dir=Path("data"), debug=debug
        )
        return result.save_dir

    def validate(self, values: Mapping[str, Any]) -> str | None:
        error = validate_exposure(
            values,
            fixed=True,
            auto_rejected="gs-refine 需要显式正曝光；自动曝光会观察到上一相位残留",
        ) or validate_target_fits_window(values)
        if error is not None:
            return error
        self.hint_bench_calibration(values)
        return None


# ---------------------------------------------------------------------------
# Adapter: model-in-loop (frontier)
# ---------------------------------------------------------------------------


@register_shaping_method
class ModelInLoopMethod(ShapingMethod):
    """Forward-model-in-the-loop shaping: probe fit (Step A) + synthesis (Step B).

    Same code path as the CLI ``slm-model-in-loop``. Needs torch, is the most
    recently developed of the two, and has the strictest failure modes
    (geometry bake-off, holdout correlation, rejection streak) — so it stays
    behind the same adapter boundary and its ``RuntimeError`` messages are
    acceptance criteria, not noise to swallow.
    """

    key: ClassVar[str] = "model-in-loop"
    label: ClassVar[str] = "物理预测模型闭环整形（前沿，需 torch）"
    action: ClassVar[str] = "开始物理预测模型闭环整形"
    blurb: ClassVar[str] = (
        "每轮重拟合物理预测模型 Zernike 像差（Step A）后冻结并合成目标相位（Step B），"
        "带 trust region 与逐轮验收。需 torch；目前最佳 CV 0.643 / EE 0.786，"
        "尚未达到验收的均匀方斑口径。"
    )

    def render_objective(self, values: dict[str, Any]) -> None:
        """MIL 目标与评分，一行 4 列：目标形状 / 目标边长 / w_eff / w_uni。"""
        d = MIL_WIDGET_DEFAULTS
        cells = _param_cells(4)
        with cells[0]:
            values["target_shape"] = st.selectbox(
                "目标形状",
                options=list(MIL_TARGET_SHAPE_LABELS),
                index=list(MIL_TARGET_SHAPE_LABELS).index(str(d["target_shape"])),
                format_func=MIL_TARGET_SHAPE_LABELS.__getitem__,
                key=self.wkey("target_shape"),
            )
        with cells[1]:
            values["target_side"] = self.render_target_side(
                values,
                int(d["target_side"]),
                lo=MIL_TARGET_SIDE_MIN_PX,
                step=5,
                help="目标方斑边长（相机像素），上界 = 当前读出窗口边长",
            )
        with cells[2]:
            values["w_efficiency"] = self.num_input(
                "w_eff",
                "权重 w_eff",
                _MIL_W_EFFICIENCY,
                0.01,
                2.0,
                0.05,
                help="必须 >0：只优化 -CV 会掏空目标框（实测环围能量 → 0.002）",
            )
        with cells[3]:
            values["w_uniformity"] = self.num_input(
                "w_uni", "权重 w_uni", _MIL_W_UNIFORMITY, 0.0, 2.0, 0.05
            )

    def render_budget(self, values: dict[str, Any]) -> None:
        """MIL 预算：预设 + 轮数 / Step A 步数 / Step B 步数。

        A preset only writes when the selection **changed**.  Writing on every
        render would silently undo any hand-tuned budget the moment the operator
        touched a neighbouring field (Streamlit keeps the selectbox value across
        reruns, so "changed" is the only honest trigger).
        """
        d = MIL_WIDGET_DEFAULTS
        labels = list(MIL_BUDGET_PRESETS)
        chosen = st.selectbox(
            "预设",
            options=labels,
            index=labels.index("标准 6/80/600"),
            key=self.wkey("budget_preset"),
            help="「标准」= docs/slm/slm_model_in_loop.md §5 默认值（6/80/600）；"
            "只在切换预设时覆写预算，手改的值不会被冲掉",
        )
        rounds, step_a, step_b = MIL_BUDGET_PRESETS[chosen]
        if st.session_state.get(self.wkey("budget_preset_applied")) != chosen:
            st.session_state[self.wkey("budget_preset_applied")] = chosen
            queue_widget_writes(
                self,
                {
                    "n_rounds": rounds,
                    "step_a_iterations": step_a,
                    "step_b_iterations": step_b,
                },
            )
        for cell, (key, label, lo, hi, step, default) in zip(
            _param_cells(4),
            (
                ("n_rounds", "轮数", 1, 20, 1, d["n_rounds"]),
                (
                    "step_a_iterations",
                    "Step A 步数",
                    0,
                    2000,
                    10,
                    d["step_a_iterations"],
                ),
                (
                    "step_b_iterations",
                    "Step B 步数",
                    1,
                    4000,
                    50,
                    d["step_b_iterations"],
                ),
                ("seed", "随机种子", 0, 99999, 1, d["seed"]),
            ),
            strict=True,
        ):
            with cell:
                values[key] = self.num_input(key, label, default, lo, hi, step)

    def render_advanced(self, values: dict[str, Any]) -> None:
        """MIL 专属高级参数（默认收起）。"""
        d = MIL_WIDGET_DEFAULTS
        st.caption("Step A 探针拟合")
        probe_fields = (
            ("probe_count", "探针数", 1, 64, 1, _PROBE_COUNT_HELP),
            ("probe_spread", "探针 σ (rad)", 3.0, 12.0, 0.5, _PROBE_SPREAD_HELP),
            ("holdout_probe_count", "验证集探针", 0, 8, 1, _HOLDOUT_HELP),
            ("freeform_holdout_count", "全频随机相位验证集探针", 0, 8, 1, None),
        )
        for cell, (key, label, lo, hi, step, help_) in zip(
            _param_cells(4), probe_fields, strict=True
        ):
            with cell:
                values[key] = self.num_input(
                    key, label, d[key], lo, hi, step, help=help_
                )
        values["min_holdout_correlation"] = self.num_input(
            "min_holdout_correlation",
            "相关度门限",
            0.5,
            0.0,
            1.0,
            0.01,
            help="验证集探针相关度低于此值即判定拟合失败（实测 speckle 路线 corr≈0.07）",
        )
        st.caption("物理预测模型几何")
        for cell, (key, label, lo, hi, step, help_) in zip(
            st.columns(2),
            (
                ("region", "模型网格", 8, 512, 8, _MIL_REGION_HELP),
                ("far_field_padding", "远场补零", 1, 64, 1, _MIL_PADDING_HELP),
            ),
            strict=True,
        ):
            with cell:
                values[key] = self.num_input(
                    key, label, d[key], lo, hi, step, help=help_
                )
        st.caption("求解器与开关")
        c1, c2 = st.columns(2)
        with c1:
            values["warm_start"] = st.checkbox(
                "热启动 (Step B 沿用上一轮相位)",
                value=True,
                key=self.wkey("warm_start"),
            )
        with c2:
            # ``SlmModelInLoopParams`` has no ``panel_pixel_um``: the forward model
            # gets the panel pitch from the panel geometry sweep, so this is shown
            # as a read-only device fact rather than an editable knob that would
            # go nowhere.
            st.caption(
                f"面板像素间距 {SLM_PITCH_UM} μm（器件常量 PITCH_UM）· "
                f"有效口径 {SLM_PIXEL_SIZE_UM} μm —— 相差 0.2 μm 是真实器件几何"
            )
            # Both, because the two numbers are different physical quantities and
            # showing only the smaller one invites "fixing" it.
            values["panel_pixel_um"] = float(SLM_PITCH_UM)
        self.render_extra_row(
            values,
            {
                "step_a_lr": {
                    "label": "Step A 学习率",
                    "min_value": 1e-4,
                    "max_value": 1.0,
                    "value": 0.05,
                    "step": 0.005,
                    "format": "%.4f",
                },
                "step_b_lr": {
                    "label": "Step B 学习率",
                    "min_value": 1e-4,
                    "max_value": 1.0,
                    "value": 0.05,
                    "step": 0.005,
                    "format": "%.4f",
                },
                "n_orders": {
                    "label": "Zernike 阶数",
                    "min_value": 1,
                    "max_value": 10,
                    "value": 4,
                    "step": 1,
                },
            },
        )

    def build_params(self, values: Mapping[str, Any]) -> SlmModelInLoopParams:
        return SlmModelInLoopParams(
            n_rounds=int(values["n_rounds"]),
            seed=int(values["seed"]),
            target_shape=str(values["target_shape"]),
            target_side=int(values["target_side"]),
            w_uniformity=float(values["w_uniformity"]),
            w_efficiency=float(values["w_efficiency"]),
            probe_count=int(values["probe_count"]),
            probe_spread=float(values["probe_spread"]),
            step_a_iterations=int(values["step_a_iterations"]),
            step_b_iterations=int(values["step_b_iterations"]),
            holdout_probe_count=int(values["holdout_probe_count"]),
            freeform_holdout_count=int(values.get("freeform_holdout_count", 0)),
            min_holdout_correlation=float(values["min_holdout_correlation"]),
            step_a_lr=float(values.get("step_a_lr", 0.05)),
            step_b_lr=float(values.get("step_b_lr", 0.05)),
            n_orders=int(values.get("n_orders", 4)),
            warm_start=bool(values.get("warm_start", True)),
            region=int(values["region"]),
            panel_span_px=float(values["panel_span_px"]),
            far_field_padding=int(values["far_field_padding"]),
            geometry_file=str(values.get("geometry_file", "")).strip() or None,
            # ``bench_settings`` may hand back the ``default`` token when the session
            # exposure was never probed; anything numeric passes through as-is.
            exposure_time_ms=exposure_param(values["exposure_time_ms"]),
            max_exposure_ms=float(values["max_exposure_ms"]),
            slm_number=self.slm_num,
            **self.common_param_values(values),
        )

    def make_config(self, params: SlmModelInLoopParams, *, debug: bool) -> Any:
        return build_model_in_loop_config(params, debug=debug)

    def prepare_config(self, config: Any, values: Mapping[str, Any]) -> None:
        """Give the optimizer a debug directory for probe frames.

        ``SlmModelInLoopConfig`` has no ``record_debug`` field
        (``build_model_in_loop_config`` accepts ``debug`` only for signature
        symmetry), so the run directory is the config's own ``debug_dir`` knob.
        """
        from ao_shaping.utils.io.file import debug_run_dir

        if not getattr(config, "debug_dir", None):
            config.debug_dir = str(debug_run_dir(Path("data"), "slm_model_in_loop"))
        super().prepare_config(config, values)
        st.info(
            f"物理预测模型闭环运行中：{config.n_rounds} 轮 × (Step A "
            f"{config.step_a_iterations} 步拟合 + Step B {config.step_b_iterations} "
            "步优化)，耗时较长，请勿断开设备。"
        )

    def execute(self, config: Any) -> ShapingOutcome:
        from ao_shaping.optimizer.wfless.slm_model_in_loop import (
            optimize_slm_model_in_loop,
        )

        return outcome_from_model_in_loop(optimize_slm_model_in_loop(config), config)

    def save(
        self, params: SlmModelInLoopParams, outcome: ShapingOutcome, *, debug: bool
    ) -> Path | None:
        # The canonical saver needs the live ModelInLoopResult (recorder,
        # geometry, probe records) — that is what ``raw_result`` carries.
        result = outcome.raw_result
        if result is None:
            return None
        saved = save_model_in_loop_results(
            result, params, root_dir=Path("data"), debug=debug
        )
        return saved.save_dir

    def validate(self, values: Mapping[str, Any]) -> str | None:
        error = (
            validate_torch("物理预测模型闭环")
            or validate_exposure(values, fixed=False)
            or validate_target_fits_window(values)
            or self._validate_holdout(values)
            or self._validate_geometry_region(values)
            or self._require_session_geometry(values)
        )
        if error is None:
            self.hint_bench_calibration(values)
        return error

    def _validate_holdout(self, values: Mapping[str, Any]) -> str | None:
        """验证集探针 0 被控件允许，但 Step A 拟合本身要求 ≥ 1。

        The widget bound is 0 so the operator can *see* that holdout validation is
        optional (the caption says how to actually turn it off), but
        ``SlmModelInLoopConfig.__post_init__`` rejects 0 — so catch it here with
        a message that points at the knob that actually works.
        """
        if int(values.get("holdout_probe_count", 1) or 0) >= 1:
            return None
        return (
            "验证集探针必须 ≥ 1（Step A 拟合依赖它）。真要关闭验证集验证请把"
            "「全频随机相位验证集探针」设为 0。"
        )

    def _require_session_geometry(self, values: Mapping[str, Any]) -> str | None:
        """硬件上物理预测模型没有 sweep 几何就跑不起来，且**不再**在运行时替操作员测。

        无 ``geometry_file`` 时优化器回落到 speckle 求解 —— 在本实验系统是实测
        失败（相关度 ≈0.07，低于 0.50 门槛），要等几分钟才中止。所以运行前就拒绝，
        并把唯一可行的动作写清楚。
        """
        if str(values.get("cam_type", "")) == "sim":
            return None
        if str(values.get("geometry_file", "") or "").strip():
            return None
        return (
            "没有可复用的 sweep 几何：物理预测模型闭环不再在运行时自动标定。"
            f"请在左侧「CCD参数设置」的「SLM {self.slm_num} · 光路几何与标定」点一次"
            f"「标定 SLM {self.slm_num} 光路」（入射定位 → 曝光探测 → sweep 几何，"
            "约 1–3 分钟），之后每次运行直接复用那次结果；"
            "或把几何来源切到「复用已标定几何」并填一条几何文件。"
        )

    def _validate_geometry_region(self, values: Mapping[str, Any]) -> str | None:
        """复用几何文件时，它的 ``region`` 必须与当前模型网格一致。

        A mismatch is a genuine incompatibility, not a rescaling: the pupil grid
        itself differs.  ``discover_model_in_loop_geometry(region=...)`` filters
        by region, but a hand-typed path bypasses that filter.
        """
        path = str(values.get("geometry_file", "") or "").strip()
        if not path:
            return None
        found = bench_settings.geometry_region_mismatch(path, int(values["region"]))
        if found is None:
            return None
        return (
            f"几何文件的 region={found} 与当前模型网格 region={values['region']} "
            "不一致（模型网格本身不同）：在左侧「CCD参数设置」改 region，"
            f"或点一次「标定 SLM {self.slm_num} 光路」重标（运行路径不测量）"
        )


# ---------------------------------------------------------------------------
# Transfer: physical-forward-model closed loop (docs/slm/slm_transfer_workflow.md)
# ---------------------------------------------------------------------------

#: Shapes the transfer entry actually implements.  Not "any text pattern":
#: ``TransferSession.validate`` rejects anything outside this set, and 五角星 is
#: implemented but has **no device result yet** (workflow doc §7).
TRANSFER_SHAPE_LABELS: dict[str, str] = {
    "square": "正方形",
    "rectangle": "长方形",
    "triangle": "三角形",
    "star": "五角星（已实现，尚无实验结果）",
}

#: The recipe the reproduction report ran (report/shape_trials/multishape_capability/
#: physical_multishape_reproduction_20261023.md); the GUI defaults to it so the
#: operator reproduces measured behaviour rather than a guessed budget.
TRANSFER_RECIPE: dict[str, Any] = {
    "shapes": ["square"],
    "count": 24,
    "iterations": 60,
    "blocks": 4,
    "updates": 10,
    "frames": 3,
    "side": 80.0,
    "period": 64.0,
    "seed": 102319,
    "adaptation": "auto",
    "objective": "shape",
    "accept_metric": "loss",
    "seed_policy": "transfer",
    "abba": False,
    "adaptive_exposure": False,
    "save_dataset": False,
    "device": "auto",
}

#: Stage names emitted by ``run_session`` → operator-readable labels.  Keeping the
#: mapping here (not in the transfer package) means the measurement code has no
#: presentation knowledge.
TRANSFER_STAGE_LABELS: dict[str, str] = {
    "capture": "当前光路采集（探针 + 逐帧自动曝光）",
    "reuse_capture": "复用已有采集",
    "nominal_initialization": "名义模型初始化（冷启动，未标定）",
    "fit": "物理预测模型迁移 / 拟合",
    "CCD_model_blocks": "CCD 模型配对块",
    "exposure_response_capture": "曝光阶梯诊断",
    "exposure_response_analysis": "曝光响应分析",
    "blur_calibration": "有效展宽 / 辐射尺度校准",
    "CCD_anchored_blocks": "CCD 锚定优化（验收：A–B–B–A 或仅模型损失）",
    "GS_and_differentiable_synthesis": "GS + 可微相位合成（离线）",
    "transferred_phase_differentiable_synthesis": "历史接受相位迁移 + 可微合成（离线）",
    "raw_data_audit": "原始数据审计",
    "summary": "指标摘要",
    "direction_check": "方向核对",
}


def _device_choices() -> tuple[str, ...]:
    """Accepted ``device`` values, imported lazily (the GUI stays torch-free)."""
    from ao_shaping.tools.slm.transfer.device import DEVICE_CHOICES

    return DEVICE_CHOICES


def discover_transfer_checkpoint() -> Path | None:
    """最近一次 transfer 会话留下的 ``fit/frozen_parameters.json``（按 mtime）。

    The workflow's recommended loop is "keep the last accepted checkpoint and
    re-run against it" (§7).  Finding it automatically removes the one step where
    an operator most easily types a stale path; the field stays editable, and an
    empty box still means a cold start.
    """
    root = Path("data") / "debug" / "slm_transfer"
    if not root.is_dir():
        return None
    candidates = sorted(
        root.glob("*/fit/frozen_parameters.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def plan_transfer_session_dir(root_dir: Path = Path("data")) -> Path:
    """Planned ``<root>/debug/slm_transfer/<stamp>`` session dir, **not created**.

    ``run_session`` creates the session directory itself with ``exist_ok=False``
    (refusing to merge two sessions' evidence is the documented behaviour), so the
    GUI must hand it a path that does not exist yet.  ``debug_run_dir`` creates on
    call, which would collide -- hence this planned variant, on the same
    two-level ``data/debug/<script>/<stamp>`` convention.
    """
    base = root_dir / "debug" / "slm_transfer"
    stamp = format_ts()
    candidate = base / stamp
    # Second resolution is enough for a human, but two runs started in the same
    # second are possible when an operator double-clicks.
    suffix = 2
    while candidate.exists():
        candidate = base / f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def outcome_from_transfer(session_dir: Path, shapes: Sequence[str]) -> ShapingOutcome:
    """Normalise one finished transfer session into the page's result view.

    Reads only what the session wrote (``anchored/manifest.json`` +
    ``anchored/<shape>_best.npy`` + ``journal.json``) -- the transfer pipeline is
    the only writer, so the GUI never re-derives a metric and cannot disagree
    with the audit.

    The accepted phase is on the **model control grid** (the pipeline tiling it
    onto the panel is ``capture.to_panel``), so it is tiled here, exactly the way
    the pipeline tiled it when it was measured.  No resampling, no re-wrapping:
    the phase stays raw radians and goes through the normal driver pipeline.
    """
    anchored = session_dir / "anchored"
    rows: list[dict[str, Any]] = []
    cv_by_shape: dict[str, dict[str, float]] = {}
    manifest = anchored / "manifest.json"
    if manifest.is_file():
        records = json.loads(manifest.read_text(encoding="utf-8"))
        by_label = {str(r.get("label")): r for r in records}
        for shape in shapes:
            for suffix, stage in (("_seed", "接受初值 (seed)"), ("_final", "验收后 (final)")):
                row = by_label.get(f"{shape}{suffix}")
                if row is None:
                    continue
                metrics = row.get("metrics") or {}
                cv = _round_metric(metrics.get("cv"))
                ee = _round_metric(metrics.get("ee_roi"))
                rows.append(
                    {
                        "形状": TRANSFER_SHAPE_LABELS.get(shape, shape),
                        "阶段": stage,
                        "CV": cv,
                        "EE_ROI": ee,
                        "曝光 ms": _round_metric(row.get("exposure_ms"), 3),
                    }
                )
                if cv is not None:
                    cv_by_shape.setdefault(shape, {})[suffix] = cv

    # Accepted phases are saved per shape as <shape>_best.npy; the page can put
    # exactly one on the panel, so it takes the first requested shape that has one.
    displayed = next(
        (s for s in shapes if (anchored / f"{s}_best.npy").is_file()), None
    )
    panel_phase: np.ndarray | None = None
    best_score = float("nan")
    flat_score: float | None = None
    if displayed is not None:
        from ao_shaping.tools.slm.transfer.capture import to_panel

        model_phase = np.load(anchored / f"{displayed}_best.npy")
        panel_phase = to_panel(np.asarray(model_phase, dtype=np.float64))
        scores = cv_by_shape.get(displayed, {})
        best_score = float(scores.get("_final", float("nan")))
        if "_seed" in scores:
            flat_score = float(scores["_seed"])

    stages = _transfer_stage_states(session_dir)
    failed = [name for name, status in stages.items() if status == "failed"]
    note_parts = [
        "已下发形状："
        + TRANSFER_SHAPE_LABELS.get(displayed, displayed or "")
        + f"（其余形状的接受相位在 {anchored}/<shape>_best.npy）"
        if displayed
        else "本轮没有可下发的接受相位"
    ]
    if len(shapes) > 1:
        note_parts.append("多个形状时下发第一个已保存的形状；其余请用「一次下发」的 CSV/npz 载入")
    if failed:
        note_parts.append("失败阶段：" + "、".join(TRANSFER_STAGE_LABELS.get(s, s) for s in failed))
    return ShapingOutcome(
        panel_phase=panel_phase,
        best_score=best_score,
        flat_score=flat_score,
        rows=tuple(rows),
        note="；".join(note_parts),
        saved_dir=session_dir,
        raw_result=session_dir,
    )


def _round_metric(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _transfer_stage_states(session_dir: Path) -> dict[str, str]:
    """``journal.json`` → ``{stage: status}`` (last entry per stage wins)."""
    journal = session_dir / "journal.json"
    if not journal.is_file():
        return {}
    states: dict[str, str] = {}
    for entry in json.loads(journal.read_text(encoding="utf-8")):
        states[str(entry.get("stage"))] = str(entry.get("status"))
    return states


@register_shaping_method
class TransferMethod(ShapingMethod):
    """Physical-forward-model transfer closed loop (``ao_shaping.tools.slm.transfer``).

    Same entry as the CLI: a :class:`TransferSession` handed to ``run_session``.
    There is deliberately no GUI-side orchestrator -- the stage sequence, the
    journal, the audit and the acceptance rules must be identical to the runs the
    reproduction report describes.

    What this pipeline is *not* like the other two adapters
    -------------------------------------------------------
    * It carries its **own** optical-path model: it re-measures the geometry with
      tilt probes, fits/migrates a low-parameter physical forward model, and the
      carrier is baked into the phase through ``ramp_matrix``.  Consequently
      ``validate.run`` clears the driver's radian overlay, and the sidebar 离焦
      叠加 is *not* applied during a transfer run -- the model fits defocus
      instead.  Showing the accepted phase must therefore also skip the carrier
      layer, or the tilt would be applied twice.
    * It does its **own** per-frame exposure search and reads its own fixed
      measurement window (``half_span_px=220``), so the sidebar exposure and
      readout window do not steer the measurement.  Only
      ``AO_DAHENG_MAX_EXPOSURE_MS`` still bounds it.
    """

    key: ClassVar[str] = "transfer"
    label: ClassVar[str] = "迁移学习闭环整形（物理预测模型，需 torch）"
    action: ClassVar[str] = "开始迁移学习闭环整形"
    #: 本方法的相位自带载波（``ramp_matrix``），且 ``validate.run`` 测量期间会
    #: 清掉驱动叠加层 —— 所以运行时不接侧栏叠加层。
    applies_send_overlay: ClassVar[bool] = False
    phase_carries_carrier: ClassVar[bool] = True
    blurb: ClassVar[str] = (
        "少参数物理预测模型 + 当前光路重测几何 + 检查点迁移，再用 CCD 锚定短块与 "
        "A–B–B–A 配对验收生成目标相位（配对验收可关闭，关闭后只按模型损失接受）。"
        "与 CLI ``-m ao_shaping.tools.slm.transfer`` "
        "同一入口；配方与验收见 "
        "report/shape_trials/multishape_capability/physical_multishape_reproduction_20261023.md。"
    )

    def __init__(self, prefix: str, slm_num: int) -> None:
        super().__init__(prefix, slm_num)
        #: Widget handle for stage progress; created in :meth:`prepare_config`.
        self._progress: Any = None
        #: Placeholder for the detail panel under the progress bar; same lifecycle.
        self._panel: Any = None
        self._session_dir: Path | None = None
        # Initialised here (not just in ``prepare_config``) so a direct
        # ``execute()`` / ``_on_stage()`` cannot raise ``AttributeError``.
        self._stage_total: int = 0
        self._stage_index: int = 0

    # --- widgets ----------------------------------------------------------

    def render_objective(self, values: dict[str, Any]) -> None:
        """目标：形状（可多选）/ 目标尺寸 / 载波周期。"""
        d = TRANSFER_RECIPE
        cells = _param_cells(3)
        with cells[0]:
            values["shapes"] = st.multiselect(
                "目标形状",
                options=list(TRANSFER_SHAPE_LABELS),
                default=list(d["shapes"]),
                format_func=TRANSFER_SHAPE_LABELS.__getitem__,
                key=self.wkey("shapes"),
                help="长方形用于验证、正方形+三角形用于跨形状展宽校准；"
                "只选一个形状时自动跳过校准阶段（等价 CLI --skip-calibration）",
            )
        with cells[1]:
            values["side"] = self.num_input(
                "side",
                "目标尺寸 (CCD px)",
                d["side"],
                8.0,
                400.0,
                5.0,
                help="正方形边长 / 长方形 2:1 的短边 / 三角形外接圆半径的一半 —— "
                "沿用 CLI --side 的口径，三角形不按边长解释",
            )
        with cells[2]:
            values["period"] = self.num_input(
                "period",
                "载波周期 (面板 px)",
                d["period"],
                4.0,
                1024.0,
                1.0,
                help="去零级载波的斜率光栅周期，写进物理预测模型的 ramp_matrix；"
                "支持任意周期（无 8 倍网格限制），目标落在窗内的组合约束由运行前检查校验",
            )
        st.caption(
            "本方法的载波在**模型与相位内部**（``ramp_matrix``），不是侧栏「相位叠加」里那一层 —— "
            "两者不能同时施加，下发接受相位时会自动跳过叠加层里的载波。"
        )

    def render_budget(self, values: dict[str, Any]) -> None:
        """预算：探针数 / 拟合迭代 / 锚定块数 / 每块更新数 / 每点帧数 / 随机种子。"""
        d = TRANSFER_RECIPE
        for cell, (name, label, lo, hi, step, help_text) in zip(
            _param_cells(4),
            (
                ("count", "探针数", 6, 96, 6, "分组随机探针数（CLI --count）"),
                ("iterations", "拟合迭代", 0, 99999, 10, "模型参数拟合迭代（CLI --iterations）；上限由时间预算约束"),
                ("blocks", "锚定块数", 1, 99999, 1, "CCD 锚定短块数，每块一次 A–B–B–A 验收；上限由时间预算约束"),
                ("updates", "每块更新数", 1, 99999, 1, "每块内的模型更新次数；上限由时间预算约束"),
            ),
            strict=True,
        ):
            with cell:
                values[name] = self.num_input(
                    name, label, d[name], lo, hi, step, help=help_text
                )
        for cell, (name, label, lo, hi, step, help_text) in zip(
            _param_cells(2),
            (
                ("frames", "每点帧数", 1, 9999, 1, "每个测量点的原始帧数（中位数前的样本数）；1 帧即物理合法"),
                ("seed", "随机种子", 0, 999999999, 1, "探针/初始化随机种子（CLI --seed）；上限容纳 CLI 默认（如 20261027）"),
            ),
            strict=True,
        ):
            with cell:
                values[name] = self.num_input(
                    name, label, d[name], lo, hi, step, help=help_text
                )

    def render_advanced(self, values: dict[str, Any]) -> None:
        """迁移来源与验收口径（与 CLI 的 --checkpoint/--seed-run/--adaptation/... 同名）。"""
        d = TRANSFER_RECIPE
        found = discover_transfer_checkpoint()
        values["checkpoint"] = st.text_input(
            "历史检查点 (frozen_parameters.json)",
            value=str(found) if found is not None else "",
            key=self.wkey("checkpoint"),
            help="留空 = 冷启动（先名义模型初始化再完整拟合，慢且初值未标定）。"
            "默认自动填入最近一次 transfer 会话的冻结检查点",
        )
        if found is not None:
            st.caption(f"自动发现最近检查点：{found}")
        values["seed_run"] = st.text_input(
            "历史接受相位目录 (跳过 GS)",
            value="",
            key=self.wkey("seed_run"),
            help="指向历史会话的 anchored/ 目录；填写后初值来自历史接受相位，"
            "本轮不调用 GS（CLI --seed-run）",
        )
        cells = _param_cells(4)
        with cells[0]:
            values["adaptation"] = st.selectbox(
                "模型适配",
                options=["auto", "alignment", "full"],
                index=["auto", "alignment", "full"].index(str(d["adaptation"])),
                key=self.wkey("adaptation"),
                help="auto = 先只调位置（≤30 次），验证不足再解冻离焦/宽度/像差完整拟合",
            )
        with cells[1]:
            values["seed_policy"] = st.selectbox(
                "历史相位迁移策略",
                options=["transfer", "strict"],
                index=["transfer", "strict"].index(str(d["seed_policy"])),
                key=self.wkey("seed_policy"),
                help="transfer = 按新旧模型静态像差差值与几何修正历史相位；"
                "strict = 要求网格与目标逐项相同",
            )
        with cells[2]:
            values["objective"] = st.selectbox(
                "图像优化目标",
                options=["shape", "cv"],
                index=["shape", "cv"].index(str(d["objective"])),
                key=self.wkey("objective"),
                help="cv = 只以目标核心内 CV² 为图像项（匀化优先，EE 仍记录但不参与目标）",
            )
        with cells[3]:
            values["accept_metric"] = st.selectbox(
                "A–B–B–A 验收指标",
                options=["loss", "cv"],
                index=["loss", "cv"].index(str(d["accept_metric"])),
                key=self.wkey("accept_metric"),
                help="cv = 候选平均 CV 必须优于参考平均 CV 减去前后参考漂移；"
                "这样 EE 上升而 CV 退化的候选不会被接受",
            )
        values["device"] = st.selectbox(
            "计算设备",
            options=list(_device_choices()),
            index=list(_device_choices()).index(str(d["device"])),
            key=self.wkey("device"),
            help="auto = 有可用 CUDA 就用 GPU，否则 CPU；cpu = 与历史产物逐位对照"
            "（拟合 / 合成 / CCD 块都受影响）。实际使用的设备写进 session.json 与各阶段 config.json",
        )
        values["calibration"] = st.checkbox(
            "跨形状校准（曝光阶梯 + 有效展宽）",
            value=True,
            key=self.wkey("calibration"),
            help="需要同时选正方形/长方形/三角形；只选一个形状时自动跳过（CLI --skip-calibration）",
        )
        values["abba"] = st.checkbox(
            "A–B–B–A 配对验收（CCD 实测）",
            value=bool(d["abba"]),
            key=self.wkey("abba"),
            help="开 = 每块测 A1/B1/B2/A2，由实测指标判定候选是否接受；"
            "关（默认）= 保留 CCD 锚定反馈，跳过 B1/B2/A2 与方向核对，候选只按模型损失接受，"
            "摘要写 acceptance=model_only（等价 CLI --no-abba）",
        )
        values["adaptive_exposure"] = st.checkbox(
            "运行中自动调整 CCD 曝光（块间）",
            value=bool(d["adaptive_exposure"]),
            key=self.wkey("adaptive_exposure"),
            help="每个优化块前重新测光；启用配对时块内曝光保持一致。保存搜索帧和曝光读回值，结束时同曝光重拍初始与最终相位。",
        )
        values["save_dataset"] = st.checkbox(
            "保存采集 npz（离线 --dataset 用）",
            value=bool(d["save_dataset"]),
            key=self.wkey("save_dataset"),
            help="关（默认）= 探针只留在内存里直接交给拟合，不写 sample_*.npz；"
            "开 = 按原样写出（离线 --dataset 与 scripts/slm 的分析脚本需要），"
            "每条探针多 60-80 MB 与大部分耗时（等价 CLI --save-dataset）",
        )
        st.caption(
            "曝光与读出窗由本流程自己确定：逐帧自动曝光、测量窗固定 half_span=220 px。"
            f"附加曝光上限仍受 {DAHENG_EXPOSURE_LIMIT_ENV} 约束"
            f"（当前 {configured_daheng_exposure_limit() or '未设置 = 由画面探测'}）。"
        )

    # --- mapping ----------------------------------------------------------

    def build_params(self, values: Mapping[str, Any]) -> Any:
        """Widget values → :class:`TransferSession`（与 CLI 同一个参数对象）。"""
        from ao_shaping.tools.slm.transfer.workflow import TransferSession

        session_dir = plan_transfer_session_dir()
        self._session_dir = session_dir
        checkpoint = str(values.get("checkpoint", "") or "").strip()
        seed_run = str(values.get("seed_run", "") or "").strip()
        shapes = tuple(values.get("shapes") or ())
        calibration = values.get("calibration", True)
        return TransferSession(
            out=session_dir,
            checkpoint=Path(checkpoint) if checkpoint else None,
            seed_run=Path(seed_run) if seed_run else None,
            shapes=shapes,
            count=int(values["count"]),
            iterations=int(values["iterations"]),
            blocks=int(values["blocks"]),
            updates=int(values["updates"]),
            frames=int(values["frames"]),
            side=float(values["side"]),
            period=float(values["period"]),
            seed=int(values["seed"]),
            camera=str(values.get("cam_type", "daheng")),
            cam_id=int(values.get("cam_id", 0)),
            slm_number=int(values.get("slm_number", self.slm_num)),
            calibration=None if calibration is True else False,
            adaptation=str(values["adaptation"]),
            accept_metric=str(values["accept_metric"]),
            objective=str(values["objective"]),
            seed_policy=str(values["seed_policy"]),
            abba=bool(values["abba"]),
            adaptive_exposure=bool(values.get("adaptive_exposure", False)),
            save_dataset=bool(values["save_dataset"]),
            device=str(values["device"]),
        )

    def make_config(self, params: Any, *, debug: bool) -> Any:
        """``TransferSession`` *is* the config; ``debug`` never switches evidence off.

        The pipeline's evidence model is a session directory with journal/audit
        from the first stage on -- there is no "run without saving" mode to mirror,
        so the debug checkbox cannot remove it.
        """
        return params

    def prepare_config(self, config: Any, values: Mapping[str, Any]) -> None:
        """Stage progress: the run blocks the page, so the only live signal is this bar."""
        stages = config.plan()["stages"]
        self._progress = st.progress(0.0, text=f"共 {len(stages)} 个阶段：{stages[0]}")
        # The detail panel lives under the bar; it is redrawn from the stage
        # hook, so it keeps updating while the run blocks this script.
        self._panel = st.empty()
        self._stage_total = len(stages)
        self._stage_index = 0

    def _on_stage(self, name: str, status: str) -> None:
        """``run_session`` stage hook → progress bar + detail panel.

        Runs on the Streamlit thread; the panel is redrawn on every event
        (including failures) so the operator sees *where* the run is.
        """
        if self._panel is not None and self._session_dir is not None:
            with self._panel.container():
                render_progress(
                    collect_progress(self._session_dir), running=status != "failed"
                )
        if status != "complete" or self._progress is None:
            return
        self._stage_index += 1
        label = TRANSFER_STAGE_LABELS.get(name, name)
        self._progress.progress(
            min(1.0, self._stage_index / max(1, self._stage_total)),
            text=f"{self._stage_index}/{self._stage_total} {label}",
        )

    def execute(self, config: Any) -> ShapingOutcome:
        """The only place the transfer pipeline is invoked."""
        from ao_shaping.tools.slm.transfer.workflow import run_session

        run_session(config, on_stage=self._on_stage)
        return outcome_from_transfer(Path(config.out), tuple(config.shapes))

    def validate(self, values: Mapping[str, Any]) -> str | None:
        """运行前拒绝：让 transfer 自己的规则先跑一遍，而不是等硬件阶段失败。

        ``TransferSession.validate`` is the authoritative rule set; calling it here
        means the run button is disabled instead of the operator losing several
        minutes to a device stage that was going to fail anyway.
        """
        error = validate_torch("迁移学习闭环")
        if error is not None:
            return error
        cam_type = str(values.get("cam_type", ""))
        if cam_type != "daheng":
            return (
                f"迁移学习闭环目前只对大恒相机做过实验验证（相机类型 {cam_type!r}）。"
                "其他后端仅有接口支持；请先在左侧「CCD参数设置」把相机类型设为 daheng"
            )
        if not values.get("shapes"):
            return "至少选择一个目标形状"
        # 参数级前置检查：载波周期若不是标定档、或目标根本放不进采样窗，
        # 以前要等到第一个 CCD 分块才报 "empty target core"（2026-10-26 实测），
        # 那时采集与拟合已经跑了几分钟。这里在按「开始」之前就给出可执行的改法。
        from ao_shaping.tools.slm.transfer.preflight import check_parameters

        problems = check_parameters(
            side=float(values.get("side", 80.0)),
            period=float(values.get("period", 64.0)),
            count=int(values.get("count", 24)),
            iterations=int(values.get("iterations", 60)),
            blocks=int(values.get("blocks", 4)),
            updates=int(values.get("updates", 10)),
            frames=int(values.get("frames", 3)),
            shapes=tuple(values.get("shapes", ())),
        )
        errors = [problem for problem in problems if problem.severity == "error"]
        if errors:
            first = errors[0]
            return f"{first.message} → {first.hint}"
        try:
            self.build_params(values).validate()
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)
        return None


# ---------------------------------------------------------------------------
# Back-compat wrappers (the pre-refactor module-level API)
# ---------------------------------------------------------------------------


def build_gs_refine_params(values: Mapping[str, Any]) -> SlmGsRefineParams:
    """Back-compat: map widget values to :class:`SlmGsRefineParams`.

    Kept because scripts and tests import it directly; delegates to
    :meth:`GsRefineMethod.build_params` so there is exactly one mapping.
    """
    return GsRefineMethod("slm1", int(values.get("slm_number", 1))).build_params(values)


def build_model_in_loop_params(values: Mapping[str, Any]) -> SlmModelInLoopParams:
    """Back-compat: map widget values to :class:`SlmModelInLoopParams`."""
    return ModelInLoopMethod("slm1", int(values.get("slm_number", 1))).build_params(
        values
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_bench_status(prefix: str, slm_num: int) -> str | None:
    """状态条：一句话说清「这条光路现在是什么状态」。

    Three cases, and each one answers a question the operator actually has:
    not calibrated at all (the sidebar 标定 button has never run), calibrated with
    some fields hand-overridden, or calibrated and trusted.
    """
    detected = bench_settings.detected_state(prefix)
    manual = bench_settings.manual_fields(prefix)
    # One predicate, one implementation: ``is_calibrated`` also drives the
    # sidebar expander title, so the status bar cannot disagree with it.
    calibrated = bench_settings.is_calibrated(prefix)
    if not calibrated:
        return (
            f"光路 SLM {slm_num}：○ 未标定 —— {bench_settings.uncalibrated_note(prefix)}；"
            f"标定按钮在左侧「CCD参数设置」→「SLM {slm_num} · 光路几何与标定」"
        )
    parts = [f"光路 SLM {slm_num}：● 已标定 {detected.get('calibrated_at') or '-'}"]
    parts.append(
        f"入射中心 {bench_settings.bench_pupil_center(prefix, '未定位')} 探测值"
    )
    if detected.get("geometry_correlation") is not None:
        parts.append(f"相关度 {float(detected['geometry_correlation']):.4f}")
    parts.append(f"曝光 {detected.get('exposure_ms', '?')} ms")
    if detected.get("geometry_file"):
        parts.append(f"几何 {detected['geometry_file']}")
    line = " · ".join(parts)
    if manual:
        line += f" · {len(manual)} 项手动覆盖（{'、'.join(manual)}）"
    return line


def render_closed_loop_shaping(slm_num: int) -> None:
    """Render the closed-loop shaping section for one SLM.

    Called from ``multi_slm_controller.main`` after the phase preview in both
    the single- and dual-SLM layouts. The method selector, the run button and
    the debug switch are generated from :data:`SHAPING_METHODS`, so a newly
    registered adapter appears here automatically.

    The page order is a contract, not a preference::

        方法 → 状态条 → 目标与评分 → 预算 → 高级参数 → 校验区 → 运行按钮 → 结果

    校验区 has to come *before* the button so the button can be ``disabled``;
    结果 last because a run blocks the page and only then has anything to show.
    Optical-path settings are **not** here — they are in the sidebar
    「CCD参数设置」 block (:mod:`ao_shaping.gui.slm.bench_settings`), because one
    experiment fixes one optical path.
    """
    prefix = f"slm{slm_num}"
    st.divider()
    st.subheader("闭环整形 (SLM + CCD)")
    st.caption(
        "开环整形（旧 GS方形整形/稳像法）在本实验系统已被实测证伪；此处复用 CLI "
        "`slm-gs-refine` / `slm-model-in-loop` / 迁移学习工具入口的闭环管线，默认值即 "
        "[report/shape_trials](../../../../report/shape_trials/README.md) 的"
        "交错重放验证配方。运行期间本页阻塞，且 SLM 句柄临时交给优化器"
        "（另一台 SLM 不受影响）。"
    )

    connected = bool(st.session_state.get(f"{prefix}_connected")) and (
        st.session_state.get(prefix) is not None
    )

    # --- 1. 方法 -----------------------------------------------------------
    label = st.selectbox(
        "整形方法",
        options=shaping_method_labels(),
        key=f"{prefix}_shp_method",
    )
    method = get_shaping_method_by_label(label)(prefix, slm_num)
    st.caption(method.blurb)

    # --- 2. 状态条 ---------------------------------------------------------
    status = render_bench_status(prefix, slm_num)
    if status is not None:
        st.caption(status)
    if bench_settings.manual_fields(prefix):
        st.warning(
            "光路参数有手动覆盖，标定不会改动它们："
            + "；".join(
                bench_settings.field_hint(prefix, name)
                for name in bench_settings.manual_fields(prefix)
            )
        )

    if not connected:
        st.info("请先在左侧连接 SLM 后再运行闭环整形。")
        return

    # --- 3-5. 目标与评分 / 预算 / 高级参数 ----------------------------------
    values = method.render_inputs()
    bench_settings.merge_bench_values(values, prefix, slm_num, method.key)

    # --- 6. 校验区（必须在按钮之前，才能 disabled）--------------------------
    errors = []
    bench_error = bench_settings.validate_bench_inputs()
    if bench_error:
        errors.append(bench_error)
    method_error = method.validate(values)
    if method_error:
        errors.append(method_error)
    for error in errors:
        st.error(error)

    debug = st.checkbox(
        "保存调试产物 (data/debug/…)",
        value=False,
        key=f"{prefix}_shp_debug",
        disabled=bool(errors),
        help="CSV / best_phase.npy / 远场 PNG / 原始三件套，与 CLI --debug 同布局",
    )

    # --- 7. 运行按钮 -------------------------------------------------------
    if st.button(
        method.action,
        key=f"{prefix}_shp_run_{method.key}",
        type="primary",
        disabled=bool(errors) or bench_settings.bench_busy(),
        use_container_width=True,
    ):
        method.run(values, debug=bool(debug))
        return

    # --- 8. 结果（未跑过时只给一句占位）-----------------------------------
    if not bench_settings.effective_state(prefix):
        st.caption("尚无运行结果。")

    # --- 9. 离焦叠加（只读状态行）-------------------------------------------
    # 离焦已升级为发送管线 T4 叠加层（驱动 set_phase_overlay），控件在侧栏
    # 「离焦叠加 (发送叠加)」区。主区只保留只读状态行，不再挂 fragment/滑块 ——
    # 拖滑块触发整页 rerun、面板消失的症状随 fragment 删除一并消除。
    defocus_on = st.session_state.get(f"{prefix}_defocus_enabled", False)
    if defocus_on:
        st.caption("离焦叠加：已启用（侧栏「离焦叠加 (发送叠加)」）")
    else:
        st.caption("离焦叠加：已关闭（侧栏「离焦叠加 (发送叠加)」可开启）")
