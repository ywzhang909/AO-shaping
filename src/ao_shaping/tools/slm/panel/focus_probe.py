"""Find the Fresnel focal length that brings the detector back to focus.

Why this exists
---------------
A plain (flat) phase puts the 0-order at the 2f plane. Move the camera off that
plane — even by a few millimetres — and the spot stops being a spot: behind
focus it is a diverging cone whose blur grows linearly with the offset, so the
peak collapses and any downstream metric (PIB, square shaping, Zernike response)
is measured on a defocused blob. The fix is a quadratic phase on the panel,
which moves the focus back onto the detector. Its magnitude is *not* derivable
from the nominal 125 mm lens alone: the effective offset is a bench fact, the
panel's phase sign convention is not written down anywhere authoritative, and
the diffraction efficiency of a wrapped lens phase depends on how many pixels
per fringe the panel can actually resolve. So it is measured.

The measurement, and its one trap
---------------------------------
Sweep the Fresnel focal length on a *plain* phase, read the frame, and pick the
brightest point. The trap is that brightness changes by orders of magnitude
across the sweep — near focus the peak clips at 255 while a few dioptres away it
is in the read noise. A fixed exposure therefore cannot rank the points: it
saturates the good ones and buries the rest.

So **the exposure is re-armed at every single point** and the comparison uses
``peak / exposure_ms`` (counts per millisecond), never the raw peak. Since the
sensor is linear in exposure, that ratio is proportional to the true intensity at
the panel and is comparable across points measured at different exposures. This
module also *verifies* the linearity assumption at the winner (a second reading at
half the exposure) rather than trusting it, because a camera with auto-gain would
silently break the whole ranking.

What is scanned
---------------
The natural parameter is dioptric power ``D = 1/f`` (m^-1), because defocus is
quadratic in radius, so ``D`` is what the correction is linear in. ``D = 0`` is
the plain phase, the baseline. The range is scanned symmetrically on purpose: the
sign of the phase that pulls focus back depends on the panel convention and on
which side of the 2f plane the camera moved to, and a wrong guess is expensive to
discover one-directionally. A coarse sweep finds the basin; a parabolic vertex
plus a re-measurement refines it. If the winner sits on the edge of the range the
range is extended outward once or twice, because "the best point is the last one"
means the answer is outside.

Usage
-----
::

    python -m ao_shaping.tools.slm.panel.focus_probe --exposure-ms 0.1
    python -m ao_shaping.tools.slm.panel.focus_probe --steps 25 --frames 3

退出码
------

=====  ==========================================================================
 码    含义
=====  ==========================================================================
  0    找到了焦面 (最佳 1/f 与焦距都写进了证据文件)
  1    探测本身失败 (设备报错, 证据文件里有 ``error``)
  2    没找到焦面 —— 扫描范围内没有任何一点落进可用曝光带
  3    **光路受阻** —— 画面里根本没有光, 不是「没对上焦」, 是光没到相机
=====  ==========================================================================

🔴 **2 与 3 必须分开**: 光路被挡住时画面里同样「找不到光斑」, 而被挡住这件事
**只能靠结构看出来** (峰值停在读出噪声本底 + 没有任何局域结构)。两者的处置完全
不同: 2 要重新扫焦距 / 挪相机, 3 要去查 光束→SLM 面板 → 2f 透镜 → CCD 三段光路。
判据见 :func:`ao_shaping.utils.image.frame_analysis.assess_spot`。

Which SLM is used
-----------------

The panel comes from the bench backend factory
(:func:`ao_shaping.drivers.slm.factory.open_slm`) whenever the backend is the
simulation, so ``--cam-type sim`` runs the very same sweep on the digital twin
(``SimSLMPib``); a hardware backend builds the Santec panel in memory mode
(``video_mode=0``) exactly as before. This module still opens exactly **one**
SLM session — only the class behind it depends on the backend.

In simulation the sweep returns the **simulation's own** best 1/f, decided by
the twin's geometry; it is *not* a bench measurement and it is generally not the
measured focal length (1549.3 mm on this bench). The evidence file therefore
carries ``backend: "sim"`` and an explicit note, so the number can never be
mistaken for a measured value.

The GUI entry point is :func:`probe_session_focus`, called from the SLM bench
sidebar "探测焦面" (focus-plane probe) button; the recommended focal it writes
feeds the sidebar defocus-overlay block (see
``ao_shaping.gui.slm.bench_settings.build_defocus_overlay``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable, Sequence

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.kernels.bench_kernels import (
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    despike_frame,
    fresnel_panel,
)
from ao_shaping.utils.cli.params import option, with_params
from ao_shaping.utils.image.frame_analysis import (
    BLOCKED_PATH_REASON,
    SpotUsability,
    assess_spot,
)

#: Probe evidence, next to the exposure and radius probes.
FOCUS_PROBE_DEBUG_DIR = Path("data") / "debug" / "bench_settings" / "focus_probe"

#: The exposure band a point must land in to be *comparable*. Below the lower
#: edge the frame is read noise; above the upper edge it may already be clipped
#: (a clipped peak stops being linear in exposure, which is the one thing the
#: ranking depends on).
ACCEPT_LOW = 0.25
ACCEPT_HIGH = 0.80

#: Where the adaptive exposure aims inside that band. The geometric middle of
#: the band, so one multiplicative correction usually lands in it.
TARGET_FRACTION = 0.45

#: Daheng clamps below this; asking for less silently returns ~0.02 ms anyway.
FLOOR_MS = 0.02

#: Coarse sweep of dioptric power (m^-1), symmetric about the plain phase.
#: +-6 m^-1 covers focal lengths of +-167 mm, which corrects a detector offset
#: up to ~200 mm behind the 2f plane on this bench. Beyond that the wrapped
#: phase's fringe period drops under ~6 panel px and diffraction efficiency
#: collapses before the geometry does.
DEFAULT_DIOPTER_RANGE = 6.0
DEFAULT_COARSE_POINTS = 25

#: 仿真后端 (``--cam-type sim``) 的默认种子与远场 FFT 补零因子。
#:
#: 补零 4 = 仿真孪生的库默认值 (:data:`slm_pib_sim.FAR_FIELD_PADDING`): 远场
#: 像元够细, 代价可接受。``scripts/slm/run_gs_refine_zernike_sweep.py`` 用 16,
#: 因为它要的是**可分辨的方形目标框**; 本工具只要一个「最佳 1/f」, 4 足够。
SIM_SEED = 0
SIM_FAR_FIELD_PADDING = 4

#: 仿真结果的说明 —— 写进证据文件与日志, 免得它被当成实测焦距引用。
SIM_RESULT_NOTE = (
    "本结果来自仿真后端 (SimSLMPib + SimPibCCD): 它是**仿真几何**自己给出的最佳 "
    "1/f, 不是实验系统实测值, 也不等于实测标定出的焦距"
)


@dataclass(frozen=True)
class ExposureSetting:
    """One point's settled exposure and what it read there."""

    exposure_ms: float
    peak: float
    fraction: float
    status: str
    reads: int


@dataclass(frozen=True)
class FocusPoint:
    """One Fresnel focal length's measurement.

    ``peak_per_ms`` is the only comparable number: ``peak`` alone is not, because
    every point is measured at its own exposure.

    ``spot`` carries the *structural* verdict for this frame (is there a localisable
    spot at all?). It is what lets the sweep tell "the path is blocked" apart from
    "the focus is not where we scanned" -- both look like "no usable point".
    """

    diopter_m1: float
    focal_mm: float
    exposure_ms: float
    peak: float
    peak_fraction: float
    peak_per_ms: float
    fwhm_px: float
    hollowness: float
    status: str
    spot: SpotUsability | None = None


@dataclass(frozen=True)
class FocusProbeResult:
    """The sweep, its winner, and the evidence needed to distrust it."""

    points: tuple[FocusPoint, ...]
    best: FocusPoint | None
    best_focal_mm: float | None
    best_diopter_m1: float | None
    gain_vs_flat: float | None
    linearity_ratio: float | None
    range_m1: tuple[float, float]
    exposure_bounds_ms: tuple[float, float]
    notes: tuple[str, ...]
    #: True when *every* measured point showed a structureless frame: the path is
    #: not delivering light at all. That is a different fault from "no focus" and
    #: gets its own exit code (3) so a caller can tell them apart.
    path_blocked: bool = False
    #: One Chinese sentence naming the fault (operator-facing).
    blockage_reason: str = ""

    def as_evidence(self) -> dict[str, Any]:
        """JSON-able form, for the probe evidence file."""
        return {
            "best_focal_mm": self.best_focal_mm,
            "best_diopter_m1": self.best_diopter_m1,
            "gain_vs_flat": self.gain_vs_flat,
            "linearity_ratio": self.linearity_ratio,
            "range_m1": list(self.range_m1),
            "exposure_bounds_ms": list(self.exposure_bounds_ms),
            "notes": list(self.notes),
            "path_blocked": self.path_blocked,
            "blockage_reason": self.blockage_reason,
            "points": [
                {
                    "diopter_m1": p.diopter_m1,
                    "focal_mm": p.focal_mm,
                    "exposure_ms": p.exposure_ms,
                    "peak": p.peak,
                    "peak_fraction": p.peak_fraction,
                    "peak_per_ms": p.peak_per_ms,
                    "fwhm_px": p.fwhm_px,
                    "hollowness": p.hollowness,
                    "status": p.status,
                    "spot": p.spot.as_dict() if p.spot is not None else None,
                }
                for p in self.points
            ],
        }


def robust_peak(frame: np.ndarray) -> float:
    """Frame peak with isolated hot pixels removed.

    A 3x3 median (``despike_frame``) deletes single-pixel defects while leaving a
    real spot, which is spatially correlated, essentially untouched. Without it a
    hot pixel outranks a dim but genuinely focused spot and the sweep locks onto
    a sensor defect that does not move with the phase at all.
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(despike_frame(arr, 3).max())


def frame_full_scale(frame: np.ndarray, override: float | None = None) -> float:
    """Full-scale value for a frame: the dtype maximum, or ``override``.

    Integer frames carry their own maximum. Float frames do not, and this bench's
    consumers all report on the 0-255 detector scale, so 255 is the convention.
    """
    if override is not None and float(override) > 0:
        return float(override)
    dtype = np.asarray(frame).dtype
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 255.0


def adapt_exposure(
    set_exposure_ms: Callable[[float], float],
    read_frame: Callable[[], np.ndarray],
    *,
    start_ms: float,
    floor_ms: float,
    ceiling_ms: float,
    full_scale: float,
    target: float = TARGET_FRACTION,
    accept: tuple[float, float] = (ACCEPT_LOW, ACCEPT_HIGH),
    max_steps: int = 6,
    discard: int = 2,
) -> ExposureSetting:
    """Find an exposure that puts the peak inside ``accept``, then report it.

    Multiplicative correction, because the response is linear in exposure: if the
    peak reads 0.9 of full scale at 0.4 ms, 0.45 needs 0.2 ms, and one step gets
    there. ``discard`` frames are dropped after each change so the reading is not
    taken on the previous exposure's integration.

    Returns the best reachable setting. Its ``status`` says whether it landed in
    the band (``ok``), was still clipped at the floor (``clipped`` — unusable for
    ranking, the peak is saturated), or was still dark at the ceiling (``dim`` —
    nothing measurable there).
    """
    lo, hi = float(accept[0]), float(accept[1])
    exposure = min(max(float(start_ms), floor_ms), ceiling_ms)
    peak = 0.0
    fraction = 0.0
    reads = 0

    for _ in range(int(max_steps)):
        set_exposure_ms(exposure)
        for _ in range(int(discard)):
            read_frame()
            reads += 1
        frame = read_frame()
        reads += 1
        peak = robust_peak(frame)
        fraction = peak / full_scale if full_scale else 0.0
        if lo <= fraction <= hi:
            return ExposureSetting(exposure, peak, fraction, "ok", reads)
        ratio = target / fraction if fraction > 1e-9 else 4.0
        ratio = float(min(max(ratio, 0.15), 6.0))
        nxt = float(min(max(exposure * ratio, floor_ms), ceiling_ms))
        # Clamped at a rail: more steps cannot improve, so stop and report why.
        if abs(nxt - exposure) <= 1e-6:
            break
        exposure = nxt

    status = "clipped" if fraction > hi else "dim"
    return ExposureSetting(exposure, peak, fraction, status, reads)


def _measure_frame(
    read_frame: Callable[[], np.ndarray], frames: int
) -> np.ndarray:
    """Average ``frames`` raw frames (mean suppresses temporal noise)."""
    acc: np.ndarray | None = None
    for _ in range(max(int(frames), 1)):
        f = np.asarray(read_frame(), dtype=np.float64)
        acc = f if acc is None else acc + f
    assert acc is not None
    return acc / max(int(frames), 1)


def _measure_point(
    diopter: float,
    *,
    apply_phase: Callable[[np.ndarray], None],
    read_frame: Callable[[], np.ndarray],
    set_exposure_ms: Callable[[float], float],
    start_exposure_ms: float,
    floor_ms: float,
    ceiling_ms: float,
    full_scale: float,
    frames: int,
    discard: int,
    radius_px: float,
    centre_px: tuple[float, float],
    panel_shape: tuple[int, int],
    wavelength_nm: float,
    pixel_um: float,
) -> FocusPoint:
    """Display one Fresnel focal length and measure it at its own exposure."""
    focal_mm = float("inf") if abs(diopter) < 1e-12 else 1000.0 / float(diopter)
    phase = fresnel_panel(
        focal_mm,
        radius=int(round(radius_px)),
        pupil_center=(int(round(centre_px[0])), int(round(centre_px[1]))),
        panel_shape=panel_shape,
        wavelength_nm=wavelength_nm,
        pixel_um=pixel_um,
    )
    apply_phase(phase)

    # A warm start from the previous point is deliberate: adjacent dioptres are
    # physically similar, so one correction step usually suffices. It is only a
    # start -- every point is re-armed, never carried over blindly.
    setting = adapt_exposure(
        set_exposure_ms,
        read_frame,
        start_ms=start_exposure_ms,
        floor_ms=floor_ms,
        ceiling_ms=ceiling_ms,
        full_scale=full_scale,
        discard=discard,
    )

    # Re-check with the averaged frame: averaging lowers noise, so the peak can
    # rise past the band the single-frame adaptation accepted. Backing off once
    # is cheaper than ranking a slightly clipped point as the winner.
    frame = _measure_frame(read_frame, frames)
    peak = robust_peak(frame)
    fraction = peak / full_scale if full_scale else 0.0
    exposure = setting.exposure_ms
    status = setting.status
    if fraction > ACCEPT_HIGH and exposure > floor_ms:
        exposure = max(floor_ms, exposure * TARGET_FRACTION / max(fraction, 1e-9))
        set_exposure_ms(exposure)
        for _ in range(int(discard)):
            read_frame()
        frame = _measure_frame(read_frame, frames)
        peak = robust_peak(frame)
        fraction = peak / full_scale if full_scale else 0.0
        status = "ok" if ACCEPT_LOW <= fraction <= ACCEPT_HIGH else "clipped"

    # 一次测量出**结构**判定: 光斑够不够亮、够不够集中、够不够窄。光路被挡住时
    # 这一步就已经能看出来 —— 那时每个点都只有读出噪声, 帧里没有任何局域结构。
    spot = assess_spot(frame)
    return FocusPoint(
        diopter_m1=float(diopter),
        focal_mm=focal_mm,
        exposure_ms=float(exposure),
        peak=peak,
        peak_fraction=fraction,
        peak_per_ms=peak / exposure if exposure > 0 else 0.0,
        fwhm_px=float(spot.fwhm_px),
        hollowness=float(spot.hollowness),
        status=status,
        spot=spot,
    )


def _parabolic_vertex(
    xs: Sequence[float], ys: Sequence[float]
) -> float | None:
    """Vertex of the parabola through three equally spaced points.

    Returns ``None`` when the middle point is not a maximum or the fit is
    degenerate, so the caller keeps its measured grid point instead of inventing
    an interpolation.
    """
    if len(xs) != 3 or len(ys) != 3:
        return None
    step = float(xs[1] - xs[0])
    if step == 0 or abs(float(xs[2] - xs[1]) - step) > 1e-9 * max(abs(step), 1.0):
        return None
    y0, y1, y2 = (float(v) for v in ys)
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12 or y1 < y0 or y1 < y2:
        return None
    offset = 0.5 * (y0 - y2) / denom
    if not (-1.0 <= offset <= 1.0):
        return None
    return float(xs[1] + offset * step)


def sweep_focus(
    *,
    apply_phase: Callable[[np.ndarray], None],
    read_frame: Callable[[], np.ndarray],
    set_exposure_ms: Callable[[float], float],
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
    radius_px: float = float(BEAM_RADIUS_PANEL),
    centre_px: tuple[float, float] = (SLM_PANEL_W / 2.0, SLM_PANEL_H / 2.0),
    coarse_points: int = DEFAULT_COARSE_POINTS,
    diopter_range: float = DEFAULT_DIOPTER_RANGE,
    diopters: Sequence[float] | None = None,
    refine_points: int = 7,
    start_exposure_ms: float = 0.1,
    floor_ms: float = FLOOR_MS,
    ceiling_ms: float = 20.0,
    frames: int = 3,
    discard: int = 2,
    full_scale: float | None = None,
    wavelength_nm: float = 1064.0,
    pixel_um: float = 8.0,
    max_extends: int = 2,
) -> FocusProbeResult:
    """Sweep Fresnel focal lengths and return the one that focuses the spot.

    Device-free: every device interaction is a callable, so the whole search runs
    against a synthetic bench in tests. The hardware wrapper is
    :func:`probe_session_focus`.

    Args:
        apply_phase: Display a raw-radian panel phase (settle is the callee's job).
        read_frame: One raw camera frame.
        set_exposure_ms: Set the exposure and return what the device accepted.
        diopters: Explicit sweep in m^-1. When given, ``coarse_points`` and
            ``diopter_range`` are ignored and no outward extension happens.
        frames: Frames averaged per point (the ranking uses the average).
        discard: Frames dropped after every exposure change.
        max_extends: How many times to extend the range when the winner sits on
            its edge (meaning the answer is outside).
    """
    notes: list[str] = []
    measured: dict[float, FocusPoint] = {}
    exposure_guess = min(max(float(start_exposure_ms), floor_ms), ceiling_ms)
    # One frame fixes the scale before anything is ranked. Integer frames carry
    # their own maximum; float frames do not, and this bench reports on 0-255.
    if full_scale is None:
        full_scale = frame_full_scale(read_frame())
    scale = float(full_scale)

    def measure(d: float) -> FocusPoint:
        nonlocal exposure_guess
        point = _measure_point(
            d,
            apply_phase=apply_phase,
            read_frame=read_frame,
            set_exposure_ms=set_exposure_ms,
            start_exposure_ms=exposure_guess,
            floor_ms=floor_ms,
            ceiling_ms=ceiling_ms,
            full_scale=scale,
            frames=frames,
            discard=discard,
            radius_px=radius_px,
            centre_px=centre_px,
            panel_shape=panel_shape,
            wavelength_nm=wavelength_nm,
            pixel_um=pixel_um,
        )
        exposure_guess = point.exposure_ms
        measured[round(float(d), 9)] = point
        return point

    if diopters is not None:
        grid = [float(d) for d in diopters]
        measured.update({round(d, 9): measure(d) for d in grid})
    else:
        half = int(coarse_points) // 2
        step = float(diopter_range) / max(half, 1)
        grid = [step * k for k in range(-half, half + 1)]
        for d in grid:
            measure(d)

        # Extend outward while the best point is the last one on either side:
        # that means the optimum is outside the scanned range, not inside it.
        for _ in range(int(max_extends)):
            usable = [p for p in measured.values() if p.status == "ok"] or list(
                measured.values()
            )
            best = max(usable, key=lambda p: p.peak_per_ms)
            edge_lo, edge_hi = min(grid), max(grid)
            if best.diopter_m1 >= edge_hi - 1e-9:
                new = [edge_hi + step * k for k in range(1, half + 1)]
                notes.append(
                    f"最优点在扫描右边界 {edge_hi:+.2f} m⁻¹，向右扩展到 "
                    f"{new[-1]:+.2f} m⁻¹"
                )
            elif best.diopter_m1 <= edge_lo + 1e-9:
                new = [edge_lo - step * k for k in range(1, half + 1)]
                notes.append(
                    f"最优点在扫描左边界 {edge_lo:+.2f} m⁻¹，向左扩展到 "
                    f"{new[-1]:+.2f} m⁻¹"
                )
            else:
                break
            grid = new
            for d in new:
                if round(d, 9) not in measured:
                    measure(d)

    usable = [p for p in measured.values() if p.status == "ok"]
    skipped = [p for p in measured.values() if p.status != "ok"]
    if skipped:
        notes.append(
            f"{len(skipped)} 个点落在可用曝光带之外"
            f"（clipped {sum(p.status == 'clipped' for p in skipped)} / "
            f"dim {sum(p.status == 'dim' for p in skipped)}），未参与排序"
        )

    best: FocusPoint | None = None
    best_diopter: float | None = None
    if usable:
        grid_sorted = sorted(usable, key=lambda p: p.diopter_m1)
        grid_best = max(grid_sorted, key=lambda p: p.peak_per_ms)
        best = grid_best
        best_diopter = grid_best.diopter_m1

        # Refine: a parabola through the winner and its neighbours gives a
        # sub-step estimate; it is only kept if a measurement there confirms it.
        idx = grid_sorted.index(grid_best)
        if 0 < idx < len(grid_sorted) - 1:
            trio = grid_sorted[idx - 1 : idx + 2]
            vertex = _parabolic_vertex(
                [p.diopter_m1 for p in trio], [p.peak_per_ms for p in trio]
            )
            if vertex is not None:
                refined = measure(vertex)
                if refined.status == "ok" and refined.peak_per_ms > best.peak_per_ms:
                    best, best_diopter = refined, refined.diopter_m1
                notes.append(
                    f"抛物线顶点 {vertex:+.3f} m⁻¹，四点复测后取 "
                    f"{best_diopter:+.3f} m⁻¹"
                )

        # A fine sweep around the winner, so "the best of 7 refined points" is
        # really the best and not just the middle of a coarse bracket.
        if int(refine_points) > 2 and best_diopter is not None:
            coarse_step = float(diopter_range) / max(int(coarse_points) // 2, 1)
            span = (
                float(diopter_range) / 10.0
                if diopters is not None
                else abs(coarse_step)
            )
            fine = np.linspace(
                best_diopter - span, best_diopter + span, int(refine_points)
            )
            for d in fine:
                key = round(float(d), 9)
                if key not in measured:
                    measure(float(d))
            fine_usable = [p for p in measured.values() if p.status == "ok"]
            if fine_usable:
                fine_best = max(fine_usable, key=lambda p: p.peak_per_ms)
                if fine_best.peak_per_ms > best.peak_per_ms:
                    best, best_diopter = fine_best, fine_best.diopter_m1

    # Linearity check at the winner. The whole ranking is peak-per-ms, which is
    # only valid if the sensor is linear in exposure. One extra reading at a
    # different exposure either confirms it or tells the operator to distrust the
    # pick. Halving is preferred; when the winner already sits near the floor
    # (bright bench) there is no room below, so it doubles instead.
    linearity: float | None = None
    factor: float | None = None
    if best is not None:
        if best.exposure_ms * 0.5 >= floor_ms:
            factor = 0.5
        elif best.exposure_ms * 2.0 <= ceiling_ms:
            factor = 2.0
    if best is not None and factor is not None:
        try:
            apply_phase(
                fresnel_panel(
                    best.focal_mm,
                    radius=int(round(radius_px)),
                    pupil_center=(
                        int(round(centre_px[0])),
                        int(round(centre_px[1])),
                    ),
                    panel_shape=panel_shape,
                    wavelength_nm=wavelength_nm,
                    pixel_um=pixel_um,
                )
            )
            alt_ms = best.exposure_ms * factor
            set_exposure_ms(alt_ms)
            for _ in range(int(discard)):
                read_frame()
            alt_frame = _measure_frame(read_frame, frames)
            alt_peak = robust_peak(alt_frame)
            if alt_peak > 0:
                linearity = (alt_peak / alt_ms) / best.peak_per_ms
                if not (0.8 <= linearity <= 1.25):
                    notes.append(
                        f"曝光线性度 {linearity:.2f} 偏离 1（0.8–1.25；"
                        f"在 ×{factor:g} 曝光下复核）；"
                        "peak/ms 排序可能被相机增益破坏，结论需人工复核"
                    )
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - device
            notes.append(f"线性度复核失败：{exc}")

    flat = measured.get(0.0)
    gain: float | None = None
    if best is not None and flat is not None and flat.peak_per_ms > 0:
        gain = best.peak_per_ms / flat.peak_per_ms

    # 光路受阻的判定: **每个**测点都是一张没有局域结构的画面。
    #
    # 为什么要求「每个点」而不是「最亮的那一个」: 只有当整条扫描线上都找不到光时
    # 才能说「光没到相机」。有一个点还能看到光斑, 就说明光路是通的, 只是焦面不在
    # 扫到的范围里 —— 那是另一件事 (退出码 2), 处置完全不同。
    blocked_reason = ""
    path_blocked = False
    if measured:
        verdicts = [p.spot.blocked for p in measured.values() if p.spot is not None]
        if verdicts and all(verdicts):
            path_blocked = True
            blocked_reason = BLOCKED_PATH_REASON
            notes.append(
                f"光路受阻: {len(measured)} 个测点全部没有局域光斑, "
                "画面停在读出噪声本底上"
            )

    return FocusProbeResult(
        points=tuple(sorted(measured.values(), key=lambda p: p.diopter_m1)),
        best=best,
        best_focal_mm=None if best is None else best.focal_mm,
        best_diopter_m1=best_diopter,
        gain_vs_flat=gain,
        linearity_ratio=linearity,
        range_m1=(
            (min(p.diopter_m1 for p in measured.values()),
             max(p.diopter_m1 for p in measured.values()))
            if measured
            else (0.0, 0.0)
        ),
        exposure_bounds_ms=(float(floor_ms), float(ceiling_ms)),
        notes=tuple(notes),
        path_blocked=path_blocked,
        blockage_reason=blocked_reason,
    )


# ---------------------------------------------------------------------------
# 硬件入口
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FocusProbeDeviceParams:
    """工厂 ``BenchDeviceParams`` 的结构化实现。

    ``BenchDeviceParams`` 是个 Protocol (结构化类型), 不能直接实例化 ——
    工厂只按属性取用, 所以给一个同形状的载体即可。

    ``video_mode`` 显式给 0 (memory 模式): DVI 模式的 ``open()`` 会挂死,
    驱动层明确禁止那条路。
    """

    cam_type: str
    cam_id: int
    exposure_time_ms: float
    cam_size: int
    slm_number: int
    slm_wavelength: int
    video_mode: int = 0


def _configure_sim_system(seed: int, far_field_padding: int) -> None:
    """把进程级的仿真系统重置到指定的种子与远场补零因子。

    ``reset_system`` 不接受 padding 参数, 所以 padding 是在**任何一次远场求值
    之前**改到新 system 上的 (那时 ``_far_field`` 与 ``_pad_buffer`` 还是
    ``None``, 改动安全)。已装的干扰项 (disturbance) 会被带过去。
    """
    from ao_shaping.drivers.sim.slm_pib_sim import get_system, reset_system

    existing = get_system()
    system = reset_system(seed=seed, disturbance=existing.disturbance)
    system.far_field_padding = int(far_field_padding)


def probe_session_focus(
    slm_number: int = 1,
    cam_type: str = "daheng",
    cam_id: int = 0,
    *,
    exposure_ms: float | None = None,
    radius_px: float | None = None,
    centre_px: tuple[float, float] | None = None,
    diopters: Sequence[float] | None = None,
    coarse_points: int = DEFAULT_COARSE_POINTS,
    diopter_range: float = DEFAULT_DIOPTER_RANGE,
    refine_points: int = 7,
    frames: int = 3,
    max_exposure_ms: float | None = None,
    wavelength_nm: int = 1064,
    debug_dir: Path | None = None,
    sim_seed: int = SIM_SEED,
    sim_far_field_padding: int = SIM_FAR_FIELD_PADDING,
) -> dict[str, Any]:
    """Open the bench, run :func:`sweep_focus`, and return JSON-able evidence.

    The panel is left showing a **plain** phase on exit, not the winner: every
    later run writes its own pattern, so leaving a lens in place would silently
    contaminate the next measurement. The winner is returned and stored by the
    caller, which is what actually applies it.

    The simulation backend's SLM comes from the bench factory
    (:func:`open_slm`), so ``cam_type "sim"`` runs the very same sweep on the
    digital twin. The simulation's winner is its **own** (geometry-decided)
    value, and the evidence says so: ``backend: "sim"`` plus a note.

    Failure never raises — the GUI calls this from a button — and comes back as
    ``{"error": ...}``, matching the radius probe's contract.
    """
    out: dict[str, Any] = {
        "slm_number": int(slm_number),
        "cam_type": cam_type,
        "cam_id": int(cam_id),
    }
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.factory import open_slm
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.utils.image.exposure_control import (
        configured_daheng_exposure_limit,
    )

    is_sim = str(cam_type).strip().lower() == "sim"
    if is_sim:
        # 仿真系统必须在**任何一次远场求值之前**就位 (种子 + 补零因子), 否则
        # 第一帧就已经用掉了默认系统的状态。
        _configure_sim_system(int(sim_seed), int(sim_far_field_padding))
        # 仿真产物必须自带「这是仿真给出的值」的标记, 否则它会被当成实测焦距
        # 引用。硬件分支**不加**这些键: 硬件那轮的证据文件字段集逐字不变。
        out["backend"] = "sim"
        out["result_kind"] = "simulation"
        out["sim_seed"] = int(sim_seed)
        out["sim_far_field_padding"] = int(sim_far_field_padding)

    ceiling = float(max_exposure_ms or 0.0) or 20.0
    configured = configured_daheng_exposure_limit()
    if configured > 0:
        ceiling = min(ceiling, configured)
    start = float(exposure_ms or 0.0) or 0.1
    radius = float(radius_px or BEAM_RADIUS_PANEL)
    centre = centre_px
    if centre is None:
        centre = (SLM_PANEL_W / 2.0, SLM_PANEL_H / 2.0)

    def apply_phase(slm: Any, phase: np.ndarray) -> None:
        # ``display_data`` rotates the memory slot itself and waits out the LCOS
        # flip; a fixed ``memory_number`` would be a firmware no-op.
        # 仿真面板没有翻转时间, 等待也没有物理含义 ⇒ 只把灰度图交下去。
        gray = slm.create_phase_from_array(phase)
        if is_sim:
            slm.display_data(gray)
        else:
            slm.display_data(gray, 0.4)

    device_params = _FocusProbeDeviceParams(
        cam_type=cam_type,
        cam_id=int(cam_id),
        exposure_time_ms=float(start),
        cam_size=0,
        slm_number=int(slm_number),
        slm_wavelength=int(wavelength_nm),
        video_mode=0,
    )
    try:
        # 面板: 仿真走后端工厂 (``cam_type="sim"`` ⇒ ``SimSLMPib``, 数字孪生
        # 面板), 硬件仍直接构造 Santec (memory 模式, video_mode=0)。
        #
        # 🔴 **硬件为什么不也走工厂**: ``tests/ao_shaping/tools/slm/
        # test_cli_argv_binding.py`` 的 R3 守卫把 ``Santec`` 这个**符号**换成一个
        # 没有 ``from_params`` 的函数, 用它拦截设备构造; 工厂走的是
        # ``Santec.from_params(params)``, 守卫拦不到, 反而会抛一个与该守卫无关的
        # AttributeError。所以硬件那条构造调用必须留在本文件里。
        if is_sim:
            slm_session = open_slm(device_params)
        else:
            slm_session = Santec(
                slm_number=int(slm_number), wavelength=int(wavelength_nm), video_mode=0
            )
        # 相机的构造与进入顺序逐字不变 (面板先开, 相机后开)。
        with slm_session as slm, create_camera(
            cam_type, int(cam_id), exposure_time_ms=start
        ) as cam:
            # Fact 2: the panel keeps the previous pattern, so start from a
            # *written* plain phase rather than whatever was on the LCOS.
            apply_phase(slm, np.zeros((SLM_PANEL_H, SLM_PANEL_W), dtype=np.float64))

            result = sweep_focus(
                apply_phase=lambda p: apply_phase(slm, p),
                read_frame=lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                set_exposure_ms=lambda ms: float(
                    cam.reset_exposure_time(float(ms)) or ms
                ),
                radius_px=radius,
                centre_px=(float(centre[0]), float(centre[1])),
                coarse_points=coarse_points,
                diopter_range=diopter_range,
                diopters=diopters,
                refine_points=refine_points,
                start_exposure_ms=start,
                ceiling_ms=ceiling,
                frames=frames,
                wavelength_nm=float(wavelength_nm),
            )
            # Restore a plain phase: see the docstring.
            apply_phase(slm, np.zeros((SLM_PANEL_H, SLM_PANEL_W), dtype=np.float64))
    except Exception as exc:  # 探测失败不该掀翻界面
        logger.exception("SLM {} 焦面探测失败", slm_number)
        out["error"] = str(exc)
        return out

    evidence = result.as_evidence()
    out.update(evidence)
    if is_sim:
        # 证据文件里必须能一眼看出这个「最佳 1/f」是仿真给的。
        out["notes"] = [*out.get("notes", []), SIM_RESULT_NOTE]
    out["radius_px"] = radius
    out["centre_panel_px"] = [float(centre[0]), float(centre[1])]
    target = Path(debug_dir) if debug_dir is not None else FOCUS_PROBE_DEBUG_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (target / f"slm{slm_number}_{stamp}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # 证据落盘失败不否定已完成的测量
        logger.warning("焦面探测证据写入 {} 失败: {}", target, exc)

    if result.best is None:
        if result.path_blocked:
            # 光路受阻与「没找到焦面」是两件事: 前者要查三段光路, 后者要重扫焦距。
            logger.error(
                "SLM {} 焦面探测：光路受阻 —— {}", slm_number, result.blockage_reason
            )
        else:
            logger.warning("SLM {} 焦面探测：没有任何点落在可用曝光带内", slm_number)
    else:
        logger.info(
            "SLM {} 焦面探测{}：最佳 1/f = {:+.3f} m⁻¹ (f = {:.0f} mm)，"
            "峰值/曝光 {:.0f} counts/ms，相对平场 ×{:.2f}",
            slm_number,
            " (仿真后端)" if is_sim else "",
            result.best_diopter_m1 or 0.0,
            result.best_focal_mm or 0.0,
            result.best.peak_per_ms,
            result.gain_vs_flat or 0.0,
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class FocusProbeParams:
    """CLI surface. Field order is the ``--help`` order (frozen by golden)."""

    slm_number: Annotated[
        int, option("--slm-number", help="SLM 设备编号 (默认 1)")
    ] = 1
    slm_wavelength: Annotated[
        int, option("--slm-wavelength", help="SLM 波长 nm (默认 1064)")
    ] = 1064
    cam_type: Annotated[
        str, option("--cam-type", help="相机类型 (daheng/miicam, 默认 daheng)")
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id", help="相机 ID (默认 0)")] = 0
    exposure_ms: Annotated[
        float,
        option("--exposure-ms", help="起始曝光 ms (默认 0.1，每个点会自动调整)"),
    ] = 0.1
    radius_px: Annotated[
        float,
        option("--radius-px", help=f"照明半径 面板px (默认 {BEAM_RADIUS_PANEL})"),
    ] = float(BEAM_RADIUS_PANEL)
    centre_x: Annotated[
        float, option("--centre-x", help="入射中心 x 面板px (默认面板中心)")
    ] = SLM_PANEL_W / 2.0
    centre_y: Annotated[
        float, option("--centre-y", help="入射中心 y 面板px (默认面板中心)")
    ] = SLM_PANEL_H / 2.0
    steps: Annotated[
        int, option("--steps", help="粗扫点数 (默认 25，含平场)")
    ] = DEFAULT_COARSE_POINTS
    diopter_range: Annotated[
        float,
        option("--diopter-range", help="粗扫 1/f 半宽 m⁻¹ (默认 6.0)"),
    ] = DEFAULT_DIOPTER_RANGE
    refine: Annotated[
        int, option("--refine", help="最优点附近的细扫点数 (默认 7)")
    ] = 7
    frames: Annotated[
        int, option("--frames", help="每点平均帧数 (默认 3)")
    ] = 3
    max_exposure_ms: Annotated[
        float, option("--max-exposure-ms", help="曝光上限 ms (默认 20)")
    ] = 20.0
    debug_dir: Annotated[
        str,
        option(
            "--debug-dir",
            help=(
                "证据文件落盘目录 (默认 data/debug/bench_settings/focus_probe)。"
                "下一轮把本轮的标定证据放进一个**专属目录**, 报告才不会读到"
                "另一次上机留下的证据"
            ),
        ),
    ] = ""


@click.command()
@with_params(FocusProbeParams, kw_name="params")
def main(params: FocusProbeParams) -> None:
    """在 plain 相位上扫描菲涅尔焦距，找让焦面回到相机的 1/f。"""
    result = probe_session_focus(
        slm_number=params.slm_number,
        cam_type=params.cam_type,
        cam_id=params.cam_id,
        exposure_ms=params.exposure_ms,
        radius_px=params.radius_px,
        centre_px=(params.centre_x, params.centre_y),
        coarse_points=params.steps,
        diopter_range=params.diopter_range,
        refine_points=params.refine,
        frames=params.frames,
        max_exposure_ms=params.max_exposure_ms,
        wavelength_nm=params.slm_wavelength,
        debug_dir=Path(params.debug_dir) if params.debug_dir else None,
    )
    if result.get("error"):
        logger.error("焦面探测失败: {}", result["error"])
        raise SystemExit(1)
    if result.get("best_focal_mm") is None:
        if result.get("path_blocked"):
            # 退出码 3 = 光路受阻, 与 2 (没找到焦面) 区分开: 前者要查
            # 光束→SLM 面板 → 2f 透镜 → CCD 三段光路。
            logger.error("焦面探测：光路受阻 —— {}", result.get("blockage_reason"))
            raise SystemExit(3)
        logger.error("焦面探测：没有任何点落在可用曝光带内，见证据文件")
        raise SystemExit(2)
    logger.info(
        "最佳 1/f = {:+.3f} m⁻¹ (f = {:.0f} mm)，相对平场峰值/曝光 ×{:.2f}",
        result["best_diopter_m1"],
        result["best_focal_mm"],
        result["gain_vs_flat"] or 0.0,
    )
    for note in result.get("notes", []):
        logger.info("  · {}", note)


if __name__ == "__main__":
    main()
