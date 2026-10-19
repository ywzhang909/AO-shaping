"""Hardware session preflight for model-in-loop shaping.

One session, two measurements, one entry point
----------------------------------------------
``slm-model-in-loop`` needs two facts about the *live* 实验系统 before Step A
opens its devices:

1. **入射定位** -- where the beam actually lands on the SLM panel. It cannot be
   derived from the camera 0-order position (the two axes are swapped 90 degrees
   and the scales differ by an order of magnitude), so it is *measured*: a
   random-phase disc is written at candidate panel positions and the position
   whose frame moves the 0-order box most is kept
   (:mod:`ao_shaping.tools.slm.panel.panel_locate`).
2. **sweep 几何标定** -- the camera/model pixel scale of this session, measured
   by the runbook's ``--stage sweep`` then ``--stage calibrate``, and gated on
   ``config.min_geometry_correlation``. A model calibrated on another session
   makes the real spot *worse*, which is why the correlation is a floor and not a
   report.

Both steps used to live inline in the click runner. That made the GUI unable to
reach them without copying the block -- and a copy would drift, which is the
failure this repository has already paid for elsewhere. This module is the
single implementation: the CLI calls :func:`calibrate_session` from its
``run()``, and a GUI calls the same function with ``locate`` / ``geometry``
toggled off for the steps it already satisfied.

Contract
--------
* :func:`calibrate_session` raises **plain** exceptions only: ``ValueError``
  (unusable scan parameters, unreadable geometry JSON), ``RuntimeError`` (the
  scan rejected the session, or the geometry failed validation), ``OSError`` /
  :class:`subprocess.CalledProcessError` (the runbook could not be launched).
  The CLI translates those into ``click.BadParameter`` /
  ``click.ClickException``; a GUI shows them with ``st.error``. Nothing here
  raises a click exception on purpose; the one exception is
  :func:`ao_shaping.runners.runner_common.parse_point` keeping its own
  ``click.BadParameter`` for a malformed ``"x,y"``, carried over verbatim from
  the CLI path it replaced.
* :class:`SessionPreflightParamError` is a ``ValueError`` subclass that marks
  *the caller's parameters* as unusable rather than a measurement failure, so a
  frontend can attach its own argument hint without matching on message text.
* A skipped step leaves its fields ``None``, and :func:`apply_session_calibration`
  writes only the fields that are not ``None``. "Skipped" can therefore never
  silently overwrite a value the caller supplied.
* Evidence lands under ``save_dir``: ``panel_scan/`` for the locate probe,
  ``session_geometry/`` for the sweep, ``evidence_dir`` being their parent.

术语见 ``docs/terminology_glossary.md``。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

from loguru import logger

from ao_shaping.runners.runner_common import parse_point

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import time cheap
    from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig
    from ao_shaping.runners.runner_common import SlmModelInLoopParams


class SessionPreflightParamError(ValueError):
    """The caller's scan parameters cannot produce a valid probe at all.

    Distinct from a measurement failure -- which is also a ``ValueError`` --
    because the remedy differs: this one is fixed by editing an argument (the
    CLI renders it as ``click.BadParameter``), the other by changing the
    experimental setup. ``param_hint`` lets a frontend name the offending
    argument without matching on the message text.
    """

    def __init__(self, message: str, *, param_hint: str | None = None) -> None:
        super().__init__(message)
        self.param_hint = param_hint


#: Step names, also attached to a failure as ``exc.preflight_step``.
STEP_PANEL_SCAN = "panel scan"
STEP_SESSION_GEOMETRY = "session geometry"


def _mark_step(exc: Exception, step: str) -> Exception:
    """Record which preflight step raised ``exc``, then return it for ``raise``.

    Both steps now travel through one call, so this is how a frontend keeps its
    own per-step wording. The exception keeps its type and message, so
    ``except ValueError`` / ``except RuntimeError`` still behave as documented.
    """
    exc.preflight_step = step  # type: ignore[attr-defined]
    return exc


@dataclass(frozen=True)
class SessionCalibration:
    """一次会话标定的结果；字段全部可为 None（该步骤被跳过时）。

    ``pupil_center`` 是面板坐标 (x, y)；``evidence_dir`` 是 ``panel_scan/`` 与
    ``session_geometry/`` 的父目录，两步的证据都在它下面。
    """

    pupil_center: tuple[int, int] | None = None
    panel_scan_response: float | None = None
    panel_scan_max_pair_drift: float | None = None
    panel_scan_session_drift: float | None = None
    panel_scan_session_shift_px: float | None = None
    geometry_file: Path | None = None
    geometry_correlation: float | None = None
    illumination_radius_px: float | None = None
    evidence_dir: Path | None = None
    notes: tuple[str, ...] = ()

    def summary_lines(self) -> tuple[str, ...]:
        """每项一行中文摘要，供前端直接展示（GUI 用 st.caption）。"""
        lines: list[str] = []
        if self.pupil_center is not None:
            cx, cy = self.pupil_center
            lines.append(f"入射定位: 瞳面中心面板坐标 ({cx}, {cy})")
        if self.panel_scan_response is not None:
            lines.append(f"入射定位响应: {self.panel_scan_response:+.3f}")
        if self.panel_scan_max_pair_drift is not None:
            lines.append(f"配对平场漂移上限: {self.panel_scan_max_pair_drift:.2%}")
        if self.panel_scan_session_drift is not None:
            lines.append(
                f"整轮平场强度漂移: {self.panel_scan_session_drift:.2%}"
                "（逐点配对已归一化，不影响定位）"
            )
        if self.panel_scan_session_shift_px is not None:
            lines.append(
                f"整轮光斑位移: {self.panel_scan_session_shift_px:.2f} 相机px"
                "（定位判据）"
            )
        if self.geometry_file is not None:
            lines.append(f"sweep 几何文件: {self.geometry_file}")
        if self.geometry_correlation is not None:
            lines.append(f"几何相关系数: {self.geometry_correlation:.4f}")
        if self.evidence_dir is not None:
            lines.append(f"证据目录: {self.evidence_dir}")
        # One line per measurement: an aperture that was not measured gets no
        # line, exactly like every other skipped measurement here. Absence is
        # not ambiguity — the geometry file records what the fit actually used.
        if self.illumination_radius_px is not None:
            lines.append(f"实测照明半径: {self.illumination_radius_px:.0f} 面板px")
        lines.extend(self.notes)
        return tuple(lines)


def _auto_session(params: SlmModelInLoopParams) -> bool:
    """True for a hardware run that brings no geometry file of its own.

    Such a run must (re)measure everything this session, which is also why its
    panel scan sweeps the whole panel instead of a 3x3 neighbourhood.
    """
    return params.cam_type != "sim" and params.geometry_file is None


def _locate_pupil_center(
    params: SlmModelInLoopParams, save_dir: Path
) -> tuple[tuple[int, int], float, float, float]:
    """Locate the beam on the panel; return the scan result's reportable fields."""
    # Deferred: the probe reaches the camera and the SLM, and importing the
    # module must stay free of device-package side effects.
    from ao_shaping.tools.kernels.bench_kernels import SLM_PANEL_H, SLM_PANEL_W
    from ao_shaping.tools.slm.panel.panel_locate import PanelLocateParams, scan_panel

    if params.cam_type == "sim":
        raise SessionPreflightParamError(
            "panel scan requires a hardware camera", param_hint="--panel-scan"
        )
    if params.panel_scan_step_px <= 0 or params.panel_scan_radius_px <= 0:
        raise SessionPreflightParamError(
            "scan step and radius must be positive", param_hint="--panel-scan"
        )
    if _auto_session(params):
        xs, ys = (240, 600, 960, 1320, 1680), (220, 480, 720, 980)
        scan_radius = max(220, params.panel_scan_radius_px)
    else:
        cx, cy = parse_point(params.pupil_center)
        xs = (cx - params.panel_scan_step_px, cx, cx + params.panel_scan_step_px)
        ys = (cy - params.panel_scan_step_px, cy, cy + params.panel_scan_step_px)
        if min(xs) < 0 or max(xs) >= SLM_PANEL_W or min(ys) < 0 or max(ys) >= SLM_PANEL_H:
            raise SessionPreflightParamError(
                "3x3 grid must lie on the SLM panel", param_hint="--panel-scan-step-px"
            )
        scan_radius = params.panel_scan_radius_px
    scan_params = PanelLocateParams(
        slm_number=params.slm_number,
        slm_wavelength=params.slm_wavelength,
        cam_type=params.cam_type,
        cam_id=params.cam_id,
        # Forwarded as the raw token, not a float: both dataclasses speak the
        # `auto`-or-decimal vocabulary and `scan_panel` is the single place that
        # resolves it.  Note the two `auto`s do NOT mean the same thing and are
        # deliberately not unified -- here it is "let the driver choose", while
        # `panel_locate` reads it as "probe the ceiling" (its pre-existing
        # meaning for a non-positive value, kept verbatim).
        exposure_ms=params.exposure_time_ms,
        patch_radius=scan_radius,
        grid_xs=",".join(map(str, xs)),
        grid_ys=",".join(map(str, ys)),
        frames=2,
        seed=params.seed,
        debug_dir=str(save_dir / "panel_scan"),
        max_flat_drift=params.panel_scan_max_flat_drift,
    )
    scan = scan_panel(
        scan_params,
        strict=True,
        max_session_drift=params.panel_scan_max_session_drift,
        max_session_shift_px=(
            float(params.panel_scan_max_session_shift_px) or None
        ),
    )
    if abs(scan.response) < 0.02:
        # A patch that misses the beam scatters nothing: the box energy barely
        # moves and the argmax is noise. Refuse instead of shaping blind.
        raise RuntimeError(f"panel scan response {scan.response:+.3f} is too weak")
    return (
        scan.center,
        scan.response,
        scan.max_pair_drift,
        scan.session_drift,
        scan.session_shift_px,
    )


def measure_session_geometry(
    config: SlmModelInLoopConfig,
    save_dir: Path,
    *,
    illumination_radius_px: float | None = None,
) -> Path:
    """Run the measured sweep for this optical session before Step A opens devices.

    ``--exposure-ms`` carries ``config.exposure_time_ms`` instead of a hard-coded
    ``0``.  ``0`` is the runbook's *auto* rung, and auto starts from the Daheng
    driver's own clamp (~0.02 ms) — 20–75× darker than this 光路's usable
    0.4–1.5 ms band (``docs/slm/slm_model_in_loop.md`` §7), so the sweep would
    measure a geometry from a nearly black frame.  A caller that never set an
    exposure still gets ``0`` and therefore exactly the CLI behaviour it had
    before; a caller that measured one now actually gets it.

    ``:g`` rather than ``str(float(...))``: the runbook parses it back with
    ``argparse`` ``type=float``, so the two are the same number, but ``:g`` keeps
    the unset case spelled ``0`` — the exact argv this function used to hard-code.
    """
    root = Path(__file__).resolve().parents[4]
    runbook = root / "scripts" / "slm" / "model_in_loop_hw_runbook.py"
    calibration_dir = save_dir / "session_geometry"
    exposure_ms = float(getattr(config, "exposure_time_ms", 0.0) or 0.0)
    common = [
        sys.executable, str(runbook), "--out", str(calibration_dir.resolve()),
        "--pupil-center", f"{config.pupil_center[0]},{config.pupil_center[1]}",
        "--region", str(config.region),
        "--far-field-size", str(config.region * config.far_field_padding),
        "--collect-disc", str(round(config.panel_span_px / 2)),
        "--zernike-radius", str(round(config.panel_span_px / 2)),
        "--slm-number", str(config.slm_number),
        "--slm-wavelength", str(config.slm_wavelength),
        "--cam-type", config.cam_type, "--cam-id", str(config.cam_id),
        "--exposure-ms", f"{exposure_ms:g}",
    ]
    if illumination_radius_px is not None and float(illumination_radius_px) > 0:
        # A measured aperture makes the two focal-scale routes comparable;
        # without it the runbook falls back to the sweep's *constructed*
        # Zernike radius, which is what made them disagree (152% on this bench).
        common.extend(
            ["--illumination-radius-px", f"{float(illumination_radius_px):g}"]
        )
    for stage in ("sweep", "calibrate"):
        subprocess.run(common + ["--stage", stage], cwd=root, check=True)
    path = calibration_dir / "bench_geometry.json"
    geometry = json.loads(path.read_text(encoding="utf-8"))
    if geometry.get("method") != "sweep" or geometry.get("correlation", 0) < config.min_geometry_correlation:
        raise RuntimeError(f"session geometry failed validation; evidence in {calibration_dir}")
    return path


def _should_measure_radius(
    params: SlmModelInLoopParams, config: SlmModelInLoopConfig
) -> bool:
    """Measure the aperture only where a measurement is meaningful.

    A simulated bench has no panel to sweep, and a caller that brought its own
    geometry file has already decided the aperture. Both would just burn minutes
    to reproduce a number that is not used.
    """
    return (
        params.cam_type != "sim"
        and params.geometry_file is None
        and float(getattr(config, "exposure_time_ms", 0.0) or 0.0) > 0
    )


def _measure_illumination_radius(
    params: SlmModelInLoopParams,
    config: SlmModelInLoopConfig,
    center: tuple[int, int],
    save_dir: Path,
) -> float | None:
    """Boundary-sweep the illuminated radius; ``None`` on failure.

    Never raises: the sweep geometry still runs without it (falling back to the
    constructed Zernike radius), and losing the aperture must not cost the run
    its pupil centre and exposure. The omission is recorded in the notes so it
    cannot pass unnoticed as a measurement.
    """
    from ao_shaping.gui.slm import bench_settings

    evidence = bench_settings.measure_illuminated_radius(
        params.slm_number,
        params.cam_type,
        params.cam_id,
        exposure_ms=float(getattr(config, "exposure_time_ms", 0.0) or 0.0),
        centre_hint=(float(center[0]), float(center[1])),
        debug_dir=Path(save_dir) / "illumination_radius",
    )
    radius = evidence.get("radius_px")
    if evidence.get("error") or radius is None:
        logger.warning(
            "照明半径实测失败（{}），几何标定退回 sweep 的 Zernike 半径（构造值）",
            evidence.get("error", "无结果"),
        )
        return None
    logger.info(
        "实测照明半径 {:.0f} 面板px（中心 {},{}；x {} / y {}）",
        radius, center[0], center[1],
        evidence.get("radius_x_px"), evidence.get("radius_y_px"),
    )
    return float(radius)


def _read_correlation(path: Path) -> float | None:
    """Report the validated correlation; never fail a good run over the summary."""
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("correlation")
    except (OSError, ValueError):
        return None
    return None if value is None else float(value)


def calibrate_session(
    params: SlmModelInLoopParams,
    config: SlmModelInLoopConfig,
    save_dir: Path,
    *,
    locate: bool = True,
    geometry: bool = True,
) -> SessionCalibration:
    """Measure this session's panel position and geometry, then report both.

    ``locate`` runs the panel-locate probe (5x4 whole-panel grid for a session
    that brings no geometry, a 3x3 grid around ``params.pupil_center``
    otherwise). ``geometry`` runs the runbook sweep through
    :func:`measure_session_geometry`, using the centre just measured when
    ``locate`` ran.

    Raises plain exceptions -- see the module docstring -- and returns the
    measurements; the caller applies them with :func:`apply_session_calibration`.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    center: tuple[int, int] | None = None
    response: float | None = None
    pair_drift: float | None = None
    session_drift: float | None = None
    session_shift: float | None = None
    if locate:
        try:
            center, response, pair_drift, session_drift, session_shift = _locate_pupil_center(
            params, save_dir
        )
        except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
            raise _mark_step(exc, STEP_PANEL_SCAN)
    else:
        notes.append("已跳过全板入射定位, 沿用给定的瞳面中心")
    geometry_file: Path | None = None
    correlation: float | None = None
    radius_px: float | None = None
    if geometry:
        # The sweep must aim at the centre measured a moment ago, not at the
        # caller's original guess.
        measured = config if center is None else replace(config, pupil_center=center)
        # Measure the illuminated radius while the centre from this same session
        # is in hand: radius = far_edge - centre, and both terms must come from
        # one session. Reusing a previous session's centre changes the radius by
        # tens of pixels (measured 2026-10-17: 180 px vs 300 px on the same scan).
        if center is not None and _should_measure_radius(params, measured):
            radius_px = _measure_illumination_radius(params, measured, center, save_dir)
        try:
            geometry_file = measure_session_geometry(
                measured, save_dir, illumination_radius_px=radius_px
            )
        except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
            raise _mark_step(exc, STEP_SESSION_GEOMETRY)
        correlation = _read_correlation(geometry_file)
    else:
        notes.append(
            "已跳过 sweep 几何标定, 沿用给定的几何文件 (本次未做对比择优验收)"
        )
    return SessionCalibration(
        pupil_center=center,
        panel_scan_response=response,
        panel_scan_max_pair_drift=pair_drift,
        panel_scan_session_drift=session_drift,
        panel_scan_session_shift_px=session_shift,
        geometry_file=geometry_file,
        geometry_correlation=correlation,
        illumination_radius_px=radius_px,
        evidence_dir=save_dir,
        notes=tuple(notes),
    )


def apply_session_calibration(
    params: SlmModelInLoopParams,
    config: SlmModelInLoopConfig,
    calibration: SessionCalibration,
) -> tuple[SlmModelInLoopParams, SlmModelInLoopConfig]:
    """Return ``(params, config)`` with the non-``None`` calibration fields written."""
    if calibration.pupil_center is not None:
        params = replace(
            params, pupil_center=f"{calibration.pupil_center[0]},{calibration.pupil_center[1]}"
        )
        config = replace(config, pupil_center=calibration.pupil_center)
    if calibration.geometry_file is not None:
        config = replace(config, geometry_file=str(calibration.geometry_file))
    return params, config
