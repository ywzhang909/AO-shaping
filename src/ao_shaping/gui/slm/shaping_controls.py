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

Architecture — why adapters + decorators
----------------------------------------
Both optimizers are **actively being developed**, and they do not share a
result type, a parameter object or even a lifecycle. Rather than letting that
leak into the page, every algorithm is a small adapter implementing
:class:`ShapingMethod`, and the cross-cutting concerns are decorators:

- :func:`register_shaping_method` — class decorator that registers an adapter
  in :data:`SHAPING_METHODS`. Adding a third algorithm is one decorated class;
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
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL, Santec
from ao_shaping.gui.slm.pattern_controls import refresh_phase_preview
from ao_shaping.runners.runner_common import (
    SlmGsRefineParams,
    SlmModelInLoopParams,
    build_gs_refine_config,
    build_model_in_loop_config,
    describe_bench_geometry,
    discover_model_in_loop_geometry,
    save_gs_refine_results,
    save_model_in_loop_results,
)
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    configured_daheng_exposure_limit,
)
from ao_shaping.utils.image.target.shapes import (
    TARGET_SHAPE_LABELS as MIL_TARGET_SHAPE_LABELS,
)

__all__ = [
    "CAM_TYPE_OPTIONS",
    "GS_REFINE_RECIPE",
    "SHAPING_METHODS",
    "GsRefineMethod",
    "ModelInLoopMethod",
    "ShapingMethod",
    "ShapingOutcome",
    "build_gs_refine_params",
    "build_model_in_loop_params",
    "get_shaping_method",
    "get_shaping_method_by_label",
    "outcome_from_gs_recorder",
    "outcome_from_model_in_loop",
    "register_shaping_method",
    "render_closed_loop_shaping",
    "shaping_method_labels",
    "shaping_session",
    "validate_daheng_exposure",
    "validate_daheng_fixed_exposure",
]

#: The interleaved-replay validated gs-refine recipe
#: (report/shape_trials/gs_refine_live_20261012.md).  Changing a default here
#: silently abandons that evidence — change it only with new replay evidence.
GS_REFINE_RECIPE: dict[str, Any] = {
    "target_side": 100,
    "side_factor": 1.0,
    "w_pib": 0.5,
    "w_unif": 0.5,
    "phase_grid": 12,
    "delta": 0.35,
    "confirm_best": True,
    "epochs": 12,
    "seed": 0,
    "gs_warm_start": True,
    "gs_iters": 12,
    "gs_method": "plain",
    "panel_pixel_um": 8.0,
    "camera_pixel_um": 2.2,
    "beam_radius_px": 450.0,
    "pupil_center": "960,690",
    "far_field_padding": 3,
    "cam_type": "daheng",
    "cam_id": 0,
    "exposure_time_ms": 0.05,
    "cam_size": 250,
    "n_eval_frames": 4,
}

#: CCD backends, in the order the selector shows them.
CAM_TYPE_OPTIONS: tuple[str, ...] = ("daheng", "miicam", "sim")

#: The Daheng saturation guard is per-session (the optical path sets it), so it
#: is never a constant here: ``configured_daheng_exposure_limit()`` reads
#: ``AO_DAHENG_MAX_EXPOSURE_MS``, and 0 means the optimizers probe the image.

#: Widget defaults for model-in-loop, keyed by widget field name.  Declared as
#: data (not read from the config at import time) so ``import`` stays free of
#: the optimizer package; :func:`_mil_default` still resolves the *authoritative*
#: value from ``SlmModelInLoopConfig`` and this table only documents intent.
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
        except Exception as exc:  # noqa: BLE001 - must not block the reacquire
            logger.exception("SLM {} 整形前释放句柄失败: {}", slm_num, exc)
    st.session_state[prefix] = None
    st.session_state[f"{prefix}_connected"] = False
    st.session_state[f"{prefix}_phase_preview"] = None
    st.session_state[f"{prefix}_phase_source"] = "闭环整形运行中"


def _reacquire_slm_handle(slm_num: int) -> Santec | None:
    """Re-open the SLM after the optimizer closed its own handle.

    ``Santec.open()`` reloads the per-serial config file itself (wavelength,
    shift, correction), so the reacquired device carries the same settings the
    sidebar had before the run.
    """
    from ao_shaping.gui.slm.multi_slm_controller import _sync_slm_session_state

    prefix = f"slm{slm_num}"
    try:
        slm = Santec(slm_number=slm_num)
        slm.open()
    except Exception as exc:
        logger.exception("SLM {} 整形后重新连接失败: {}", slm_num, exc)
        st.error(f"SLM {slm_num} 整形后重新连接失败: {exc}（请在侧边栏手动重连）")
        return None
    st.session_state[prefix] = slm
    st.session_state[f"{prefix}_connected"] = True
    _sync_slm_session_state(slm_num, slm)
    return slm


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

    Wraps :meth:`ShapingMethod.run` with the cross-cutting concerns that must
    not be duplicated per algorithm:

    1. stop the background phase toggle (it would fight the run for the panel);
    2. release the GUI's SLM handle — the optimizers open their own and the
       SDK handles are not re-entrant;
    3. re-acquire the handle in ``finally``, so a failed or aborted optimizer
       cannot leave the page without a device;
    4. log + surface any escaping exception exactly once.

    Because the decorator owns (2)–(4), a new adapter cannot forget them and
    an optimizer-side API change stays inside ``execute()``.
    """

    @functools.wraps(run)
    def wrapper(
        self: ShapingMethod, *args: Any, **kwargs: Any
    ) -> ShapingOutcome | None:
        slm_num = self.slm_num
        outcome: ShapingOutcome | None = None
        _stop_background_toggle(slm_num)
        _release_slm_handle(slm_num)
        try:
            outcome = run(self, *args, **kwargs)
            return outcome
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            logger.exception("闭环整形 [{}] 失败: {}", self.key, exc)
            st.error(f"{self.label} 失败: {exc}")
            return None
        finally:
            # Re-acquire first, then display: the body has no handle by design
            # (the optimizer owned it), so the adopted phase can only be sent
            # after the GUI owns the device again.
            slm = _reacquire_slm_handle(slm_num)
            if slm is not None:
                self.show_phase(slm, outcome)

    return wrapper


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


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

    @abstractmethod
    def render_inputs(self) -> dict[str, Any]:
        """Render this algorithm's widgets; return the values dict."""

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

    def prepare_config(self, config: Any, values: Mapping[str, Any]) -> None:
        """Hook for per-run config tweaks (progress callback, debug dir, ...)."""

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

        logger.info("GUI 闭环整形启动 [{}]: {}", self.key, type(params).__name__)
        outcome = self.execute(config)
        outcome = replace(outcome, saved_dir=self.save(params, outcome, debug=debug))
        self.summarise(outcome)
        return outcome

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
        try:
            phase_gray = slm.create_phase_from_array(outcome.panel_phase)
            slm.display_data(phase_gray, memory_mode=MEMORY_MODE_INTERNAL)
            refresh_phase_preview(self.slm_num)
            max_gray = getattr(slm, "_max_gray", None) or Santec.MAX_GRAYSCALE_VALUE
            st.success(f"最佳相位已下发（raw 弧度走驱动管线，2π={max_gray}）")
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            logger.exception("下发最佳相位失败: {}", exc)
            st.error(f"最佳相位下发失败: {exc}")

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

    def render_camera_inputs(
        self,
        values: dict[str, Any],
        *,
        exposure_default: float,
        cam_size_default: int,
        target_side: int = 0,
        extra: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        """Render the CCD block shared by every method, in one row.

        ``extra`` adds method-specific numeric inputs (e.g. frames averaged per
        measurement) to the same row, so the camera block exists exactly once.

        ``target_side`` is the objective target side length (camera px). When
        window-cropping is enabled, ``cam_size`` is constrained to be at least
        ``max(16, target_side)`` so the target square fits inside the camera
        ROI. When cropping is disabled, ``cam_size`` is set to 0 (full frame).
        """
        cells = st.columns(4 + (1 if extra else 0))
        with cells[0]:
            values["cam_type"] = st.selectbox(
                "相机",
                options=CAM_TYPE_OPTIONS,
                index=list(CAM_TYPE_OPTIONS).index(
                    str(values.get("cam_type", "daheng"))
                )
                if values.get("cam_type") in CAM_TYPE_OPTIONS
                else 0,
                key=self.wkey("cam_type"),
                help="sim = 2f-Fourier 数字孪生（无硬件自检）",
            )
        with cells[1]:
            values["cam_id"] = st.number_input(
                "相机 ID",
                min_value=0,
                max_value=8,
                value=int(values.get("cam_id", 0)),
                step=1,
                key=self.wkey("cam_id"),
            )
        with cells[2]:
            values["exposure_time_ms"] = st.number_input(
                "曝光 (ms)",
                min_value=0.0,
                value=float(values.get("exposure_time_ms", exposure_default)),
                step=0.01,
                format="%.3f",
                key=self.wkey("exposure"),
                help="Daheng 饱和保护上限取 AO_DAHENG_MAX_EXPOSURE_MS，"
                     "缺省时由画面探测（启动时校验）；0 = 由算法自动起曝",
            )
        idx = 3
        with cells[idx]:
            values["cam_crop_enabled"] = st.checkbox(
                "是否开窗",
                value=bool(values.get("cam_crop_enabled", True)),
                key=self.wkey("cam_crop"),
                help="开启后设置相机 ROI 窗口 (目标方边长须 ≤ 窗口)；关闭则整幅读出",
            )
        if values.get("cam_crop_enabled"):
            # The camera window must be at least as large as the target square
            # so the entire target fits inside the ROI.
            cam_size_min = max(16, target_side)
            with cells[idx]:
                values["cam_size"] = st.number_input(
                    "相机窗口",
                    min_value=cam_size_min,
                    max_value=1944,
                    value=max(
                        cam_size_min,
                        int(values.get("cam_size", cam_size_default)),
                    ),
                    step=10,
                    key=self.wkey("cam_size"),
                    help="ROI 窗口边长 (px)，须 ≥ 目标方边长",
                )
        else:
            values["cam_size"] = 0  # full frame
        if extra:
            for cell, (name, spec) in zip(cells[4:], extra.items()):
                with cell:
                    values[name] = st.number_input(**spec, key=self.wkey(name))


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
            from ao_shaping.utils.wavefront.zero_order_overlay import apply_zero_order_overlay

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
        "2026-10-14 两次复核）。GS 预矫正走 bake-off：实测不优于平场就丢弃，"
        "从平场继续。"
    )

    def render_inputs(self) -> dict[str, Any]:
        r = GS_REFINE_RECIPE
        values: dict[str, Any] = {}

        st.caption("目标与评分（相机像素；综合分 = w_pib·PIB + w_unif·1/(1+CV)）")
        objective_fields = (
            ("target_side", "目标方边长", 0, 3000, 1, "0 = 由照明瞳自动推导"),
            ("side_factor", "边长因子", 0.1, 5.0, 0.1, None),
            ("w_pib", "权重 w_pib", 0.0, 2.0, 0.05, None),
            ("w_unif", "权重 w_unif", 0.0, 2.0, 0.05, None),
        )
        for cell, (key, label, lo, hi, step, help_) in zip(
            st.columns(4), objective_fields
        ):
            with cell:
                values[key] = self.num_input(
                    key, label, r[key], lo, hi, step, help=help_
                )

        st.caption("SPGD 细化（粗网格自由相位）")
        spgd_fields = (
            ("epochs", "SPGD 轮数", 1, 400, 1, None),
            ("phase_grid", "相位网格", 2, 64, 1, None),
            ("delta", "扰动 δ (rad)", 0.0, 2.0, 0.05, None),
            ("seed", "随机种子", 0, 99999, 1, None),
        )
        for cell, (key, label, lo, hi, step, help_) in zip(st.columns(5), spgd_fields):
            with cell:
                values[key] = self.num_input(
                    key, label, r[key], lo, hi, step, help=help_
                )
        with st.columns(5)[4]:
            values["confirm_best"] = st.checkbox(
                "confirm-best 复测",
                value=bool(r["confirm_best"]),
                key=self.wkey("confirm_best"),
                help="破纪录候选重新测量确认，抑制噪声假提升",
            )

        st.caption("GS 预矫正（bake-off：仅实测优于平场才采用）")
        c1, c2, c3 = st.columns(3)
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
                help="实机对照结论: plain 最优 (gs_refine_live_20261012.md)",
            )

        st.caption("台架模型（实测标定值，改动前先看 report/shape_trials/）")
        with st.columns(4)[0]:
            values["pupil_center"] = st.text_input(
                "瞳孔中心 x,y (面板px)",
                value=str(values.get("pupil_center", r["pupil_center"])),
                key=self.wkey("pupil_center"),
            )
        bench_fields = (
            ("beam_radius_px", "照明半径 (面板px)", 8.0, 1000.0, 10.0, None),
            ("far_field_padding", "远场补零", 1, 8, 1, None),
            (
                "camera_pixel_um",
                "相机像素 (μm)",
                0.5,
                50.0,
                0.1,
                "测量锚点: 大恒 MER2-507 为 2.2 μm",
            ),
        )
        for cell, (key, label, lo, hi, step, help_) in zip(st.columns(3), bench_fields):
            with cell:
                default_val = r[key]
                if key == "camera_pixel_um" and "camera_pixel_um" in st.session_state:
                    default_val = st.session_state["camera_pixel_um"]
                values[key] = self.num_input(
                    key, label, default_val, lo, hi, step, help=help_
                )
        values["panel_pixel_um"] = float(r["panel_pixel_um"])

        st.caption("硬件")
        self.render_camera_inputs(
            values,
            exposure_default=float(r["exposure_time_ms"]),
            cam_size_default=int(r["cam_size"]),
            target_side=int(values.get("target_side", 0)),
            extra={
                "n_eval_frames": {
                    "label": "每测量均帧",
                    "min_value": 1,
                    "max_value": 16,
                    "value": int(r["n_eval_frames"]),
                    "step": 1,
                }
            },
        )
        values["blaze_period_px"] = int(st.selectbox(
            "去零级载波",
            options=(-1, 0, 128, 96, 64),
            format_func=lambda p: "关闭" if p == -1 else "实测自动选择" if p == 0 else f"实测 {p} px 周期",
            key=self.wkey("blaze_period_px"),
        ))
        values["blaze_direction"] = st.selectbox(
            "光栅方向", ("vertical", "horizontal"), key=self.wkey("blaze_direction")
        )
        return values

    def build_params(self, values: Mapping[str, Any]) -> SlmGsRefineParams:
        return SlmGsRefineParams(
            target_side=int(values["target_side"]),
            side_factor=float(values["side_factor"]),
            w_pib=float(values["w_pib"]),
            w_unif=float(values["w_unif"]),
            phase_grid=int(values["phase_grid"]),
            delta=float(values["delta"]),
            confirm_best=bool(values["confirm_best"]),
            epochs=int(values["epochs"]),
            seed=int(values["seed"]),
            gs_warm_start=bool(values["gs_warm_start"]),
            gs_iters=int(values["gs_iters"]),
            gs_method=str(values["gs_method"]),
            panel_pixel_um=float(values["panel_pixel_um"]),
            camera_pixel_um=float(
                st.session_state.get(
                    "camera_pixel_um", values.get("camera_pixel_um", 2.2)
                )
            ),
            beam_radius_px=float(values["beam_radius_px"]),
            pupil_center=str(values["pupil_center"]),
            far_field_padding=int(values["far_field_padding"]),
            cam_type=str(values["cam_type"]),
            cam_id=int(values["cam_id"]),
            exposure_time_ms=float(values["exposure_time_ms"]),
            cam_size=int(values["cam_size"]),
            slm_number=self.slm_num,
            n_eval_frames=int(values["n_eval_frames"]),
            blaze_period_px=int(values.get("blaze_period_px", -1)),
            blaze_direction=str(values.get("blaze_direction", "vertical")),
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
        return validate_daheng_fixed_exposure(
            str(values["cam_type"]), float(values["exposure_time_ms"])
        )


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
    label: ClassVar[str] = "正向模型闭环整形（前沿，需 torch）"
    action: ClassVar[str] = "开始正向模型闭环整形"
    blurb: ClassVar[str] = (
        "每轮重拟合正向模型 Zernike 像差（Step A）后冻结并合成目标相位（Step B），"
        "带 trust region 与逐轮验收。需 torch；目前最佳 CV 0.643 / EE 0.786，"
        "尚未达到验收的均匀方斑口径。"
    )

    def render_inputs(self) -> dict[str, Any]:
        d = MIL_WIDGET_DEFAULTS
        values: dict[str, Any] = {}

        st.caption("轮次与目标")
        round_fields = (
            ("n_rounds", "轮数", 1, 20, 1, None),
            ("target_side", "目标边长 (相机px)", 3, 1000, 5, None),
            ("seed", "随机种子", 0, 99999, 1, None),
        )
        for cell, (key, label, lo, hi, step, help_) in zip(st.columns(4), round_fields):
            with cell:
                values[key] = self.num_input(
                    key, label, d[key], lo, hi, step, help=help_
                )
        with st.columns(4)[3]:
            values["target_shape"] = st.selectbox(
                "目标形状",
                options=list(MIL_TARGET_SHAPE_LABELS),
                index=list(MIL_TARGET_SHAPE_LABELS).index(str(d["target_shape"])),
                format_func=MIL_TARGET_SHAPE_LABELS.__getitem__,
                key=self.wkey("target_shape"),
            )
        c1, c2 = st.columns(2)
        with c1:
            values["w_efficiency"] = self.num_input(
                "w_eff",
                "权重 w_efficiency",
                0.6,
                0.0,
                2.0,
                0.05,
                help="必须 >0: 只优化 -CV 会掏空目标框 (实测 EE→0.002)",
            )
        with c2:
            values["w_uniformity"] = self.num_input(
                "w_uni", "权重 w_uniformity", 0.4, 0.0, 2.0, 0.05
            )

        st.caption("Step A 探针拟合 / Step B 相位合成")
        step_fields = (
            ("probe_count", "探针数", 1, 32, 1, None),
            (
                "probe_spread",
                "探针 σ (rad)",
                3.0,
                8.0,
                0.5,
                "必须 ≳3 rad，否则拟合对像差不敏感",
            ),
            ("step_a_iterations", "Step A 步数", 0, 2000, 10, None),
            ("step_b_iterations", "Step B 步数", 1, 4000, 50, None),
            ("holdout_probe_count", "留出探针", 1, 8, 1, None),
        )
        for cell, (key, label, lo, hi, step, help_) in zip(st.columns(5), step_fields):
            with cell:
                values[key] = self.num_input(
                    key, label, d[key], lo, hi, step, help=help_
                )
        with st.columns(5)[0]:
            values["freeform_holdout_count"] = self.num_input(
                "freeform_holdout_count",
                "自由相位留出探针",
                d["freeform_holdout_count"],
                0,
                8,
                1,
            )
        values["min_holdout_correlation"] = 0.5

        st.caption("前向模型几何（实测标定值）")
        with st.columns(4)[0]:
            values["pupil_center"] = st.text_input(
                "瞳孔中心 x,y (面板px)",
                value=str(values.get("pupil_center", d["pupil_center"])),
                key=self.wkey("pupil_center"),
            )
        geometry_fields = (
            (
                "panel_span_px",
                "模型覆盖 (面板px)",
                16.0,
                1920.0,
                50.0,
                "= 2×照明半径; 用 SLM 像素直接采样会把孔径算小 ~3.5 倍",
            ),
            ("region", "模型网格", 8, 256, 8, None),
            ("far_field_padding", "远场补零", 1, 16, 1, None),
        )
        for cell, (key, label, lo, hi, step, help_) in zip(
            st.columns(3), geometry_fields
        ):
            with cell:
                values[key] = self.num_input(
                    key, label, d[key], lo, hi, step, help=help_
                )

        # The geometry bake-off needs an independently measured geometry.  The
        # GUI cannot run the calibration runbook, so it auto-fills the newest
        # sweep artefact it can find (first render only -- afterwards the widget
        # owns the value and an empty box stays empty).
        discovered = discover_model_in_loop_geometry(
            int(values.get("region", d["region"]))
        )
        values["geometry_file"] = st.text_input(
            "实测几何文件 (bench_geometry.json)",
            value=str(discovered) if discovered is not None else str(d["geometry_file"]),
            key=self.wkey("geometry_file"),
            help=(
                "scripts/slm/model_in_loop_hw_runbook.py 的 sweep 标定产物。留空时"
                "真机会退回已废弃的 speckle 路线 (本台实测 corr≈0.07 < 0.5 "
                "bake-off 门限) 并中止; 仿真不受影响。"
            ),
        )
        geometry_path = str(values["geometry_file"]).strip()
        if geometry_path and Path(geometry_path).is_file():
            st.caption(describe_bench_geometry(geometry_path))
        elif geometry_path:
            st.caption(f"⚠ 几何文件不存在: {geometry_path}")
        else:
            st.caption(
                "⚠ 未指定几何文件: 真机会退回已废弃的 speckle 路线，可能中止 "
                "bake-off。"
            )

        st.caption("硬件")
        self.render_camera_inputs(
            values,
            exposure_default=0.0,
            cam_size_default=250,
            target_side=int(values.get("target_side", 0)),
        )
        values["blaze_period_px"] = int(st.selectbox(
            "去零级载波",
            options=(-1, 0, 128, 96, 64),
            format_func=lambda p: "关闭" if p == -1 else "实测自动选择" if p == 0 else f"实测 {p} px 周期",
            key=self.wkey("blaze_period_px"),
        ))
        values["blaze_direction"] = st.selectbox(
            "光栅方向", ("vertical", "horizontal"), key=self.wkey("blaze_direction")
        )
        values["max_exposure_ms"] = configured_daheng_exposure_limit()
        return values

    def build_params(self, values: Mapping[str, Any]) -> SlmModelInLoopParams:
        camera_pixel_um = float(
            st.session_state.get("camera_pixel_um", values.get("camera_pixel_um", 2.2))
        )
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
            region=int(values["region"]),
            panel_span_px=float(values["panel_span_px"]),
            pupil_center=str(values["pupil_center"]),
            far_field_padding=int(values["far_field_padding"]),
            geometry_file=str(values.get("geometry_file", "")).strip() or None,
            cam_type=str(values["cam_type"]),
            cam_id=int(values["cam_id"]),
            exposure_time_ms=float(values["exposure_time_ms"]),
            max_exposure_ms=float(values["max_exposure_ms"]),
            cam_size=int(values["cam_size"]),
            camera_pixel_um=camera_pixel_um,
            slm_number=self.slm_num,
            blaze_period_px=int(values.get("blaze_period_px", -1)),
            blaze_direction=str(values.get("blaze_direction", "vertical")),
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
        st.info(
            f"正向模型闭环运行中：{config.n_rounds} 轮 × (Step A "
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
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            return f"正向模型闭环需要 torch（未安装）: {exc}"
        return validate_daheng_exposure(
            str(values["cam_type"]), float(values["exposure_time_ms"])
        )


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


def render_closed_loop_shaping(slm_num: int) -> None:
    """Render the closed-loop shaping section for one SLM.

    Called from ``multi_slm_controller.main`` after the phase preview in both
    the single- and dual-SLM layouts. The method selector, the run button and
    the debug switch are generated from :data:`SHAPING_METHODS`, so a newly
    registered adapter appears here automatically.
    """
    prefix = f"slm{slm_num}"
    st.divider()
    st.subheader("闭环整形 (SLM + CCD)")
    st.caption(
        "开环整形（旧 GS方形整形/稳像法）在本台架已被实测证伪；此处复用 CLI "
        "`slm-gs-refine` / `slm-model-in-loop` 的闭环管线，默认值即 "
        "[report/shape_trials](../../../../report/shape_trials/README.md) 的"
        "交错重放验证配方。运行期间本页阻塞，且 SLM 句柄临时交给优化器"
        "（另一台 SLM 不受影响）。"
    )

    if (
        not st.session_state.get(f"{prefix}_connected")
        or st.session_state.get(prefix) is None
    ):
        st.info("请先在左侧连接 SLM 后再运行闭环整形。")
        return

    label = st.selectbox(
        "整形方法",
        options=shaping_method_labels(),
        key=f"{prefix}_shp_method",
    )
    method = get_shaping_method_by_label(label)(prefix, slm_num)
    st.caption(method.blurb)

    debug = st.checkbox(
        "保存调试产物 (data/debug/…)",
        value=False,
        key=f"{prefix}_shp_debug",
        help="CSV / best_phase.npy / 远场 PNG / 原始三件套，与 CLI --debug 同布局",
    )

    values = method.render_inputs()
    if st.button(method.action, key=f"{prefix}_shp_run_{method.key}", type="primary"):
        method.run(values, debug=bool(debug))
