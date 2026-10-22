"""Did the optical path change since it was calibrated? Measure, don't assume.

The problem this solves
-----------------------
A calibration result (pupil centre, aperture, focal scale) is only valid for the
bench that produced it, and on this bench the bench is physically re-configurable
between sessions: the camera gets moved off the 2f plane, a mount gets nudged, an
SLM is swapped. Nothing in the artefact records *what it was measured on*, so a
stale calibration is reused silently and every downstream number is confidently
wrong. A TTL cannot fix this -- the path can change in a second or hold for a
week -- so the check has to be a measurement.

The fingerprint
---------------
Display two known phases and read the spot:

1. **flat** -> ``FWHM``, ``hollowness``, ``centroid``
2. **flat + a known Zernike tilt** -> ``centroid``

Those three numbers encode the three things a reconfiguration actually moves:

* ``FWHM`` is the **axial position**. Measured on this bench: with the camera
  ~11 mm behind the 2f plane the flat FWHM reads ~39 px; in focus it reads
  ~16 px. That is a 2.4x change, far outside any noise floor.
* ``tilt_shift`` is the **focal scale** -- the panel-to-camera angular
  magnification. A different lens or a changed SLM-camera spacing moves it.
* ``centroid`` is the **beam axis**. Shift calibration and staging moves show up
  here.

What is deliberately *not* part of the verdict: brightness. Measured on this
bench, the same flat phase read 20.6 counts/ms and then 2.9 counts/ms between
runs -- a 7.3x step -- while the stability within a single run was 1.05x over two
minutes. Any brightness threshold would fire on the light level rather than on
the optical path. Brightness is therefore recorded as context and never compared.

Every reading is exposure-adapted and clipped frames are rejected, because the
FWHM of an underexposed or saturated frame is not a measurement: at 0.05 ms on
this bench the same spot measured 1064 px wide.

Usage
-----
::

    python -m ao_shaping.tools.slm.panel.bench_fingerprint --capture
    python -m ao_shaping.tools.slm.panel.bench_fingerprint --compare
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.kernels.bench_kernels import zernike_panel
from ao_shaping.tools.slm.panel.focus_probe import (
    ACCEPT_HIGH,
    ACCEPT_LOW,
    adapt_exposure,
)
from ao_shaping.utils.cli.params import option, with_params
from ao_shaping.utils.image.frame_analysis import estimate_shift, measure_spot

#: Fingerprint evidence, beside the other per-bench probes.
FINGERPRINT_DEBUG_DIR = Path("data") / "debug" / "bench_settings" / "fingerprint"

#: Tilt amplitude (radians of unit-RMS Zernike (1,1)) used for the shift probe.
#: ~5 camera px per radian on this bench, so this lands around 20 px -- well
#: above the ~1 px centroid noise and far below anything that would spoil the
#: spot shape.
TILT_RAD = 4.0

#: Comparison tolerances. Deliberately asymmetric with respect to what they
#: guard: ``fwhm`` carries the axial signal (measured change 140%), so 25% is a
#: wide but decisive band; ``shift`` is the steadiest of the three, so 15%; the
#: centroid moves with staging, so an absolute 30 px.
FWHM_RTOL = 0.25
SHIFT_RTOL = 0.15
CENTROID_TOL_PX = 30.0

#: Above this flat-spot width the fingerprint is **not trusted**, and it refuses
#: to block a run. Measured on this bench today: an in-focus spot reads ~16 px,
#: while a camera ~11 mm off the 2f plane reads 58-79 px -- and in that regime
#: every shape statistic is unstable (two captures 20 s apart gave 79.2 then
#: 59.9 px; phase correlation of the tilt pair returns 0.0 because the speckle
#: inside the blob decorrelates). A check that vetoes runs on that noise is worse
#: than no check, so it says "cannot decide" instead.
FWHM_FOCUSED_PX = 30.0


@dataclass(frozen=True)
class BenchFingerprint:
    """What the bench looked like at one moment, in geometry terms only."""

    measured_at: str
    slm_number: int
    slm_wavelength: int
    pupil_center_px: tuple[int, int]
    tilt_rad: float
    flat_fwhm_px: float
    flat_hollowness: float
    flat_centroid: tuple[float, float]
    tilt_centroid: tuple[float, float]
    #: Displacement the known tilt produced, by **phase correlation** between the
    #: two frames -- not by differencing the centroids. Measured on this bench: a
    #: tilt big enough to be measurable also deforms the spot, and centroid
    #: readings then jump +-8 px between repeats of the *same* phase, which is a
    #: 2x error in the number the check depends on (see ``frame_analysis``).
    tilt_shift_px: float
    #: Context only -- see the module docstring. Never part of the verdict.
    flat_peak_per_ms: float
    flat_peak_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchFingerprint:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in payload.items() if k in known}
        # A JSON round-trip turns the centre tuple into a list; comparing those
        # directly reports a difference every single time and vetoes every run.
        centre = kwargs.get("pupil_center_px")
        if isinstance(centre, (list, tuple)) and len(centre) == 2:
            kwargs["pupil_center_px"] = (int(centre[0]), int(centre[1]))
        return cls(**kwargs)


@dataclass(frozen=True)
class FingerprintVerdict:
    """Whether the current bench still matches the reference."""

    ok: bool
    skipped: bool
    reasons: tuple[str, ...] = ()
    deltas: dict[str, float] = field(default_factory=dict)
    #: "Could not decide" -- the measurement ran but the spot is too far out of
    #: focus for any shape statistic to mean anything. Never a reason to block.
    notes: tuple[str, ...] = ()

    def raise_if_changed(self) -> None:
        if not self.ok:
            raise BenchPathChanged(self)


class BenchPathChanged(RuntimeError):
    """The bench no longer matches the calibration it would be run against."""

    def __init__(self, verdict: FingerprintVerdict) -> None:
        self.verdict = verdict
        super().__init__("; ".join(verdict.reasons) or "bench fingerprint mismatch")


def _average(read: Callable[[], np.ndarray], n: int) -> np.ndarray:
    """Mean of ``n`` raw frames -- averaging first keeps one moment per feature."""
    acc: np.ndarray | None = None
    for _ in range(max(int(n), 1)):
        f = np.asarray(read(), dtype=np.float64)
        acc = f if acc is None else acc + f
    assert acc is not None
    return acc / max(int(n), 1)


def measure_fingerprint(
    *,
    apply_phase: Callable[[np.ndarray], None],
    read_frame: Callable[[], np.ndarray],
    set_exposure_ms: Callable[[float], float],
    pupil_center_px: tuple[int, int],
    panel_shape: tuple[int, int] = (1200, 1920),
    radius_px: float = 450.0,
    extra_phase: np.ndarray | None = None,
    slm_number: int = 1,
    slm_wavelength: int = 1064,
    tilt_rad: float = TILT_RAD,
    start_exposure_ms: float = 0.2,
    floor_ms: float = 0.02,
    ceiling_ms: float = 20.0,
    frames: int = 3,
    discard: int = 2,
    wavelength_nm: float = 1064.0,
    pixel_um: float = 8.0,
    measured_at: str | None = None,
) -> BenchFingerprint:
    """Capture the fingerprint. Two displays, a handful of frames, seconds.

    Raises:
        RuntimeError: If neither phase could be brought into the measurable
            exposure band. A fingerprint taken from a clipped or dark frame is
            worse than none -- it would veto the run for the wrong reason.
    """
    from ao_shaping.tools.slm.panel.focus_probe import (
        frame_full_scale,
        robust_peak,
    )

    radius = int(max(round(radius_px), 1))
    centre = (int(pupil_center_px[0]), int(pupil_center_px[1]))
    full_scale = frame_full_scale(read_frame())
    exposure = min(max(float(start_exposure_ms), floor_ms), ceiling_ms)
    peaks: dict[str, float] = {}
    fractions: dict[str, float] = {}
    features: dict[str, Any] = {}
    images: dict[str, np.ndarray] = {}

    def capture(tag: str, phase: np.ndarray) -> None:
        nonlocal exposure
        apply_phase(phase)
        setting = adapt_exposure(
            set_exposure_ms,
            read_frame,
            start_ms=exposure,
            floor_ms=floor_ms,
            ceiling_ms=ceiling_ms,
            full_scale=full_scale,
            discard=discard,
        )
        exposure = setting.exposure_ms
        if setting.status != "ok":
            raise RuntimeError(
                f"{tag}: 曝光无法进入可测带（{setting.status}，"
                f"{setting.exposure_ms:.3f} ms 处峰值占满量程 "
                f"{setting.fraction:.0%}）；光路状态不明确，拒绝取指纹"
            )
        # One averaged frame feeds both the shape and the brightness context, so
        # the two can never come from different moments of a drifting bench.
        frame = _average(read_frame, frames)
        features[tag] = measure_spot(frame)
        images[tag] = frame
        peak = robust_peak(frame)
        peaks[tag] = peak / max(exposure, 1e-9)
        fractions[tag] = peak / (full_scale or 1.0)

    flat = np.zeros(panel_shape, dtype=np.float64)
    if extra_phase is not None:
        # A standing correction (e.g. the focus-plane lens) has to be part of
        # *both* captures. Leaving it out of the tilt shot would compare a
        # tilt measured in focus against one measured out of it, and the
        # difference would look like a changed focal scale.
        flat = flat + np.asarray(extra_phase, dtype=np.float64)
    capture("flat", flat)
    capture(
        "tilt",
        flat
        + zernike_panel(
            {(1, 1): float(tilt_rad)},
            radius=radius,
            pupil_center=centre,
            panel_shape=panel_shape,
        ),
    )

    dy, dx, corr_peak = estimate_shift(images["flat"], images["tilt"])
    shift_px = (
        float("inf")
        if not (np.isfinite(dy) and np.isfinite(dx))
        else float(math.hypot(dx, dy))
    )

    return BenchFingerprint(
        measured_at=measured_at or datetime.now().isoformat(timespec="seconds"),
        slm_number=int(slm_number),
        slm_wavelength=int(slm_wavelength),
        pupil_center_px=centre,
        tilt_rad=float(tilt_rad),
        flat_fwhm_px=float(features["flat"].fwhm_px),
        flat_hollowness=float(features["flat"].hollowness),
        flat_centroid=(
            float(features["flat"].centroid_x),
            float(features["flat"].centroid_y),
        ),
        tilt_centroid=(
            float(features["tilt"].centroid_x),
            float(features["tilt"].centroid_y),
        ),
        tilt_shift_px=shift_px,
        flat_peak_per_ms=float(peaks["flat"]),
        flat_peak_fraction=float(fractions["flat"]),
    )


def compare_fingerprints(
    current: BenchFingerprint,
    reference: BenchFingerprint | None,
    *,
    fwhm_rtol: float = FWHM_RTOL,
    shift_rtol: float = SHIFT_RTOL,
    centroid_tol_px: float = CENTROID_TOL_PX,
    enabled: bool = True,
) -> FingerprintVerdict:
    """Compare against a reference. ``enabled=False`` is the operator's override.

    A missing reference is *not* a mismatch: the first run on a bench has nothing
    to compare against, and refusing it would be a worse default than recording
    one.
    """
    if not enabled:
        return FingerprintVerdict(
            ok=True,
            skipped=True,
            reasons=("已按参数跳过光路指纹比对（操作员判断本次光路未变）",),
        )
    if reference is None:
        return FingerprintVerdict(
            ok=True,
            skipped=True,
            reasons=("没有可比对的历史指纹，本次只记录",),
        )

    # Trustworthiness gate, before anything is compared: on a defocused spot
    # every number below is noise, and a noise-driven veto is a false alarm.
    widest = max(current.flat_fwhm_px, reference.flat_fwhm_px)
    if widest > FWHM_FOCUSED_PX:
        return FingerprintVerdict(
            ok=True,
            skipped=False,
            notes=(
                f"平场光斑宽 {widest:.0f} px 超过 {FWHM_FOCUSED_PX:.0f} px："
                "光斑没合焦，形状统计量不可信，本次不判定光路是否变更"
                "（先把光路调回焦面，焦面探测：python -m "
                "ao_shaping.tools.slm.panel.focus_probe）",
            ),
            deltas={"flat_fwhm_px": widest},
        )

    reasons: list[str] = []
    deltas: dict[str, float] = {}

    fwhm_now, fwhm_ref = current.flat_fwhm_px, reference.flat_fwhm_px
    fwhm_rel = abs(fwhm_now - fwhm_ref) / max(fwhm_ref, 1e-9)
    deltas["fwhm_rel"] = fwhm_rel
    if fwhm_rel > fwhm_rtol:
        reasons.append(
            f"平场光斑宽度 {fwhm_now:.1f} px 与参考 {fwhm_ref:.1f} px 相差 "
            f"{fwhm_rel:.0%}（容差 {fwhm_rtol:.0%}）；这通常意味着相机离开了焦面"
        )

    shift_now, shift_ref = current.tilt_shift_px, reference.tilt_shift_px
    shift_rel = abs(shift_now - shift_ref) / max(shift_ref, 1e-9)
    deltas["tilt_shift_rel"] = shift_rel
    if shift_ref > 1.0 and shift_rel > shift_rtol:
        reasons.append(
            f"已知倾斜产生的位移 {shift_now:.1f} px 与参考 {shift_ref:.1f} px 相差 "
            f"{shift_rel:.0%}（容差 {shift_rtol:.0%}）；焦面尺度变了"
        )

    shift_px = math.hypot(
        current.flat_centroid[0] - reference.flat_centroid[0],
        current.flat_centroid[1] - reference.flat_centroid[1],
    )
    deltas["centroid_px"] = shift_px
    if shift_px > centroid_tol_px:
        reasons.append(
            f"平场光斑质心移动了 {shift_px:.0f} px（容差 {centroid_tol_px:.0f} px）；"
            "光轴/平移标定不再是这个"
        )

    if reference.pupil_center_px != current.pupil_center_px:
        reasons.append(
            f"指纹使用的入射中心 {current.pupil_center_px} 与参考 "
            f"{reference.pupil_center_px} 不同，两次不是同一个口径"
        )

    return FingerprintVerdict(
        ok=not reasons, skipped=False, reasons=tuple(reasons), deltas=deltas
    )


# ---------------------------------------------------------------------------
# 硬件入口
# ---------------------------------------------------------------------------


def _save(payload: dict[str, Any], stem: str, debug_dir: Path | None) -> Path | None:
    target = Path(debug_dir) if debug_dir is not None else FINGERPRINT_DEBUG_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = target / f"{stem}{stamp}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path
    except OSError as exc:  # 证据落盘失败不否定已完成的测量
        logger.warning("指纹证据写入 {} 失败: {}", target, exc)
        return None


def probe_session_fingerprint(
    slm_number: int = 1,
    cam_type: str = "daheng",
    cam_id: int = 0,
    *,
    pupil_center_px: tuple[int, int] = (960, 690),
    radius_px: float = 450.0,
    tilt_rad: float = TILT_RAD,
    exposure_ms: float | None = None,
    max_exposure_ms: float | None = None,
    wavelength_nm: int = 1064,
    frames: int = 3,
    debug_dir: Path | None = None,
    reference: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Capture the fingerprint on hardware, compare it, and return evidence.

    Never raises for a *measurement* problem -- the GUI calls this from a button.
    A mismatch is returned as ``{"ok": False, "reasons": [...]}`` so the caller
    decides whether to abort or to warn.
    """
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    out: dict[str, Any] = {"slm_number": int(slm_number), "cam_id": int(cam_id)}
    start = float(exposure_ms or 0.0) or 0.2
    ceiling = float(max_exposure_ms or 0.0) or 20.0
    configured = configured_daheng_exposure_limit()
    if configured > 0:
        ceiling = min(ceiling, configured)

    def apply_phase(slm: Any, phase: np.ndarray) -> None:
        slm.display_data(slm.create_phase_from_array(phase), 0.4)

    try:
        with Santec(
            slm_number=int(slm_number), wavelength=int(wavelength_nm), video_mode=0
        ) as slm:
            with create_camera(cam_type, int(cam_id), exposure_time_ms=start) as cam:
                fingerprint = measure_fingerprint(
                    apply_phase=lambda p: apply_phase(slm, p),
                    read_frame=lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                    set_exposure_ms=lambda ms: float(
                        cam.reset_exposure_time(float(ms)) or ms
                    ),
                    pupil_center_px=pupil_center_px,
                    radius_px=radius_px,
                    slm_number=int(slm_number),
                    slm_wavelength=int(wavelength_nm),
                    tilt_rad=float(tilt_rad),
                    start_exposure_ms=start,
                    ceiling_ms=ceiling,
                    frames=int(frames),
                    wavelength_nm=float(wavelength_nm),
                )
                # Leave a plain phase behind: the next run writes its own pattern,
                # and a stray tilt would silently skew it.
                apply_phase(slm, np.zeros((1200, 1920), dtype=np.float64))
    except Exception as exc:  # 探测失败不该掀翻界面
        logger.exception("光路指纹采集失败")
        out["error"] = str(exc)
        return out

    ref = BenchFingerprint.from_dict(reference) if reference else None
    verdict = compare_fingerprints(fingerprint, ref, enabled=enabled)
    payload = fingerprint.to_dict()
    payload.update(
        {
            "verdict": {
                "ok": verdict.ok,
                "skipped": verdict.skipped,
                "reasons": list(verdict.reasons),
                "notes": list(verdict.notes),
                "deltas": verdict.deltas,
            },
            "reference_measured_at": None if ref is None else ref.measured_at,
            "accept_band": [ACCEPT_LOW, ACCEPT_HIGH],
        }
    )
    out.update(payload)
    out["path"] = str(_save(payload, f"fingerprint_slm{slm_number}_", debug_dir) or "")
    logger.info(
        "光路指纹：平场 FWHM {:.1f} px，倾斜位移 {:.1f} px，质心 ({:.0f},{:.0f})；{}",
        fingerprint.flat_fwhm_px,
        fingerprint.tilt_shift_px,
        fingerprint.flat_centroid[0],
        fingerprint.flat_centroid[1],
        "与参考一致" if verdict.ok else "与参考不一致：" + "；".join(verdict.reasons),
    )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class FingerprintParams:
    """CLI surface. Field order is the ``--help`` order."""

    slm_number: Annotated[int, option("--slm-number", help="SLM 编号 (默认 1)")] = 1
    slm_wavelength: Annotated[
        int, option("--slm-wavelength", help="SLM 波长 nm (默认 1064)")
    ] = 1064
    cam_type: Annotated[
        str, option("--cam-type", help="相机类型 (默认 daheng)")
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id", help="相机 ID (默认 0)")] = 0
    centre_x: Annotated[
        float, option("--centre-x", help="入射中心 x 面板px (默认 960)")
    ] = 960.0
    centre_y: Annotated[
        float, option("--centre-y", help="入射中心 y 面板px (默认 690)")
    ] = 690.0
    radius_px: Annotated[
        float, option("--radius-px", help="照明半径 面板px (默认 450)")
    ] = 450.0
    tilt_rad: Annotated[
        float, option("--tilt-rad", help=f"已知倾斜幅度 rad (默认 {TILT_RAD})")
    ] = TILT_RAD
    exposure_ms: Annotated[
        float, option("--exposure-ms", help="起始曝光 ms (默认 0.2，自动调整)")
    ] = 0.2
    reference: Annotated[
        str, option("--reference", help="参考指纹 JSON 路径（默认取最新一份）")
    ] = ""
    skip_compare: Annotated[
        bool,
        option(
            "--skip-compare",
            is_flag=True,
            help="只采集不比对（操作员判断光路未变）",
        ),
    ] = False


def _latest_reference(debug_dir: Path | None) -> dict[str, Any] | None:
    target = Path(debug_dir) if debug_dir is not None else FINGERPRINT_DEBUG_DIR
    files = sorted(target.glob("fingerprint_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _load_reference(
    params: FingerprintParams,
) -> dict[str, Any] | None:
    """Reference fingerprint: explicit path, else the newest on disk.

    A path that cannot be read degrades to "no reference" with a warning rather
    than raising: a typo in a convenience flag must not take the probe down, and
    the verdict already distinguishes "no reference" from "matches".
    """
    if params.skip_compare:
        return None
    raw = params.reference.strip()
    if not raw:
        return _latest_reference(None)
    try:
        return json.loads(Path(raw).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("参考指纹读取失败，按「没有参考」处理: {} ({})", raw, exc)
        return None


@click.command()
@with_params(FingerprintParams, kw_name="params")
def main(params: FingerprintParams) -> None:
    """采集并比对光路指纹（平场宽度 / 已知倾斜位移 / 质心）。"""
    result = probe_session_fingerprint(
        slm_number=params.slm_number,
        cam_type=params.cam_type,
        cam_id=params.cam_id,
        pupil_center_px=(int(params.centre_x), int(params.centre_y)),
        radius_px=params.radius_px,
        tilt_rad=params.tilt_rad,
        exposure_ms=params.exposure_ms,
        wavelength_nm=params.slm_wavelength,
        reference=_load_reference(params),
        enabled=not params.skip_compare,
    )
    if result.get("error"):
        logger.error("光路指纹采集失败: {}", result["error"])
        raise SystemExit(1)
    verdict = result["verdict"]
    logger.info(
        "平场 FWHM {:.1f} px，倾斜位移 {:.1f} px，质心 ({:.0f},{:.0f})",
        result["flat_fwhm_px"],
        result["tilt_shift_px"],
        result["flat_centroid"][0],
        result["flat_centroid"][1],
    )
    if verdict.skipped:
        logger.info("跳过比对：{}", "；".join(verdict.reasons))
        return
    for note in verdict.notes:
        logger.warning("光路校验未判定：{}", note)
    if verdict.ok:
        logger.info("与参考一致（参考测量于 {}）", result["reference_measured_at"])
        return
    logger.warning("光路已变更：")
    for reason in verdict["reasons"]:
        logger.warning("  · {}", reason)
    raise SystemExit(3)


if __name__ == "__main__":
    main()