"""Bench geometry core: model-to-camera calibration from real captures.

Holds :class:`BenchGeometry`, its JSON round trip, and the speckle-correlation
solve :func:`calibrate_bench_geometry`.  The smooth-sweep route lives in
:mod:`ao_shaping.optimizer.wfless.bench_sweep_fit`; the shared static-aberration
fit lives in :mod:`ao_shaping.optimizer.wfless.bench_shared_aberration`.

本模块只做**测量**，不含任何 model-in-loop 优化循环。上游数学仍是
``algorithm/signal_processing/zernike_coefficient_optimizer.py``（唯一的物理预测
模型）；物理设备动作由 ``scripts/slm/model_in_loop_hw_runbook.py`` 的
``--stage sweep/calibrate`` 驱动，本模块只消费它的产物。

约束：全程不 import 任何硬件驱动。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.ndimage import zoom

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless.bench_geometry_common import (
    _MIN_PUPIL_SEMI_AXIS_PX,
    _pupil_amplitude,
    _spot_fwhm,
)
from ao_shaping.utils.slm.bench_constants import LASER_WAVELENGTH_M, SLM_PITCH_M

if TYPE_CHECKING:
    from ao_shaping.optimizer.wfless.bench_geometry_common import CalibrationRecord

#: Largest share of the camera window a measured flat spot may fill before the
#: geometry is refused (see :meth:`BenchGeometry.self_check_failures`). Measured on
#: this bench: 13.19 px of 300 (4.4%) for a geometry that shaped acceptably, against
#: 106.85 px of 300 (36%) for one fitted on an uncorrected defocused pupil -- both of
#: which cleared the correlation floor, at 0.848 and 0.831. The bound sits ~3.4x above
#: the usable value and ~2.4x below the broken one.
GEOMETRY_MAX_SPOT_WINDOW_FRACTION = 0.15

# Below this Pearson correlation between the model and a real capture, the
# speckle patterns do not correspond and the geometry solve has not converged on
# anything physical. Measured against a synthetic bench the correct geometry
# scores >0.95, so 0.75 leaves ample margin while still rejecting the
# "smooth near-flat pupil" case that cannot discriminate at all.
_MIN_CORRELATION = 0.75


# ---------------------------------------------------------------------------
# Bench geometry calibration (offline, from real captures)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchGeometry:
    """Model-to-camera calibration derived from real far-field captures.

    The digital twin hides three quantities that a real bench fixes by
    experiment: how much of the SLM panel the illuminated pupil covers, how wide
    that beam is, and how finely the far field has to be sampled before it can
    be compared with a camera frame. This record carries the answer so a
    hardware run does not have to re-derive it.

    Attributes:
        panel_disc_radius: Panel pixels from the panel centre to the edge of the
            region that was mapped onto the model grid.
        region: Model grid side length in pupil pixels.
        beam_waist_panel_px: Gaussian waist of the illumination in panel pixels.
        far_field_size: Zero-padding size the forward model needs for its far
            field to be sampled at the camera's pitch.
        camera_px_per_model_px: Far-field scale: camera pixels per model pixel.
        spot_fwhm_camera_px: Measured spot FWHM in camera pixels.
        spot_fwhm_model_px: Spot FWHM the model reproduces, in model pixels.
        correlation: Pearson correlation between the model and the measurement
            over the calibration records; the discriminative goodness metric.
            For ``method="sweep"`` this instead carries the defocus-response
            goodness of fit (1 - normalised RMS), because that method has no
            speckle to correlate against.
        method: How the geometry was obtained -- ``"speckle"`` (default) or
            ``"sweep"``.
        calibration_notes: Free-form provenance, e.g. the measured tilt slopes
            and the panel/camera axis mapping.
    """

    panel_disc_radius: int
    region: int
    beam_waist_panel_px: float
    far_field_size: int
    camera_px_per_model_px: float
    spot_fwhm_camera_px: float
    spot_fwhm_model_px: float
    correlation: float
    method: str = "speckle"
    calibration_notes: str = ""

    # ------------------------------------------------------------------
    # 序列化 —— 唯一真源 (issue #152)
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        """Serialise to the ``bench_geometry.json`` schema (pretty, indent=2)."""
        import json
        from dataclasses import asdict

        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json_text(cls, text: str) -> "BenchGeometry":
        """Inverse of :meth:`to_json`.

        Old files written before ``method`` existed simply omit the key; the
        dataclass default (``"speckle"``) then applies, so historical geometries
        keep loading unchanged.
        """
        import json

        return cls(**json.loads(text))

    @property
    def is_sweep(self) -> bool:
        """``True`` iff this geometry came from the sweep calibration route.

        Only a sweep geometry carries an independent goodness number
        (``correlation`` as defocus-response goodness); a speckle artefact is the
        abandoned ~0.07 route. Consumers should branch on this rather than
        re-string-comparing ``self.method == "sweep"``.
        """
        return self.method == "sweep"

    def self_check_failures(self, cam_size: int) -> list[str]:
        """Reasons this geometry should be refused despite a good correlation.

        The correlation floor alone does not catch a geometry that was fitted on a
        pupil phase carrying an uncorrected defocus: on this bench such a geometry
        scored 0.831 and was accepted, while one that shaped acceptably scored 0.848.
        Every textual self-diagnostic is shared between the two -- both carry
        ``waist NOT identifiable``-equivalent output and both report
        ``no usable phase correlation`` -- so none of them can be used to tell them
        apart, and neither can ``spot_fwhm_model_px``, which is 8.64 in both.

        What does separate them is the measured flat spot's share of the camera
        window: 13.19 px of 300 (4.4%) for the usable geometry against 106.85 px of
        300 (36%) for the defocused one. A diffraction-limited spot is a small
        fraction of the detector, so a spot filling a third of it means the beam was
        not focused *or* the scale is wrong -- and either way the geometry cannot
        drive shaping.

        Args:
            cam_size: Camera window edge in pixels, the same window the calibration
                measured in.

        Returns:
            Human-readable reasons, empty when the geometry vouches for itself.
        """
        if cam_size <= 0:
            raise ValueError(f"cam_size must be positive, got {cam_size}")
        failures: list[str] = []
        fwhm = float(self.spot_fwhm_camera_px)
        if not math.isfinite(fwhm) or fwhm <= 0.0:
            failures.append(f"measured spot FWHM is not a usable width: {fwhm}")
        else:
            share = fwhm / float(cam_size)
            if share > GEOMETRY_MAX_SPOT_WINDOW_FRACTION:
                failures.append(
                    f"flat spot FWHM {fwhm:.1f} px fills {100 * share:.0f}% of the "
                    f"{cam_size} px camera window (limit "
                    f"{100 * GEOMETRY_MAX_SPOT_WINDOW_FRACTION:.0f}%); the calibration "
                    "phase most likely carried an uncorrected defocus, or the "
                    "camera/model scale is wrong"
                )
        return failures

    def model_side_for_camera(self, camera_px: float) -> int:
        """Convert a target square side from camera pixels to model pixels.

        Args:
            camera_px: Desired target side in camera pixels.

        Returns:
            The equivalent side in model far-field pixels, at least 1.
        """
        return max(1, int(round(float(camera_px) / self.camera_px_per_model_px)))


def _model_window(
    far_field: np.ndarray, camera_shape: tuple[int, int], camera_px_per_model_px: float
) -> np.ndarray:
    """Crop the model's far field to the extent a camera frame covers.

    ``camera_px_per_model_px`` is how many camera pixels one model pixel spans,
    so a frame of ``n`` camera pixels covers ``n / camera_px_per_model_px`` model
    pixels.
    """
    rows = max(1, int(round(camera_shape[0] / camera_px_per_model_px)))
    cols = max(1, int(round(camera_shape[1] / camera_px_per_model_px)))
    rows, cols = min(rows, far_field.shape[0]), min(cols, far_field.shape[1])
    top = (far_field.shape[0] - rows) // 2
    left = (far_field.shape[1] - cols) // 2
    return far_field[top : top + rows, left : left + cols]


def load_bench_geometry(path: str | Path) -> BenchGeometry:
    """Load a ``bench_geometry.json`` from disk.

    The single entry point for reading the schema, so producer and consumer no
    longer each hand-roll ``json.loads`` + ``BenchGeometry(**...)``. See
    :meth:`BenchGeometry.from_json_text` for old-file (no ``method`` key)
    compatibility.
    """
    return BenchGeometry.from_json_text(Path(path).read_text(encoding="utf-8"))


def _cbg_measure_references(camera_pixel, far_field_size, focal_length, panel_span_px, records, region, slm_pitch, wavelength):
# Reference frame + spot width + optical scale derivation lifted verbatim
    measured = [np.asarray(r.image, dtype=np.float64) for r in records]
    widths = [_spot_fwhm(img) for img in measured]
    finite = [w for w in widths if np.isfinite(w) and w > 0]
    if not finite:
        raise ValueError("no record has a measurable spot (all zero or degenerate)")
    fwhm_meas = float(np.median(finite))
    reference_index = int(np.nanargmax(widths))
    reference = measured[reference_index]
    reference_phase = np.asarray(records[reference_index].phase, dtype=np.float64)

    # The model grid spans `panel_span_px` panel pixels, so its physical pitch is
    # that span times the SLM pitch divided by the number of samples. Using
    # `slm_pitch` here (as this did originally) assumes one model pixel per SLM
    # pixel, which is only true when region == panel_span_px.
    if panel_span_px is not None and float(panel_span_px) > 0:
        model_pitch = float(panel_span_px) * slm_pitch / float(region)
    else:
        model_pitch = slm_pitch
    p_fft = wavelength * focal_length / (far_field_size * model_pitch)
    # How many camera pixels one model pixel spans (1.85 at far_field_size=4096
    # for a 2.2 um camera and 8 um SLM pitch).
    camera_px_per_model_px = p_fft / camera_pixel
    # Model pixels -> panel pixels, for reporting sizes on the panel.
    model_to_panel = (float(panel_span_px) / float(region)) if panel_span_px else 1.0
    return camera_px_per_model_px, fwhm_meas, model_to_panel, reference, reference_phase


def _cbg_scan_seeds(region):
# Initial best sentinel + candidate lists lifted verbatim
    best: tuple[float, BenchGeometry] | None = None
    # (disc, waist) are searched jointly against the correlation. Matching a
    # spot *size* first was tried and abandoned: the FWHM of a speckle cut
    # through the global max is a noisy, non-monotonic function of the beam
    # waist, so a bisection on it converged to ~60% error, whereas the speckle
    # correlation peaks exactly when the pupil mapping is right.
    waist_grid_candidates = [
        region / float(2**k) for k in range(1, 7)
    ]  # region/2 .. region/64
    return best, waist_grid_candidates


def _cbg_finalize(best, model_to_panel, region):
# Saturation guard + geometry report lifted verbatim
    waist_grid_candidates_all = [region / float(2**k) for k in range(1, 7)]
    saturated = (
        best[1].beam_waist_panel_px
        >= waist_grid_candidates_all[0] * model_to_panel - 1e-9
    )
    if best[1].correlation < _MIN_CORRELATION or saturated:
        logger.warning(
            "bench geometry is NOT trustworthy: correlation {:.4f}, aperture {} px, "
            "w0 {:.1f} panel px. The pupil phase must vary across the illuminated "
            "area for this to work; Zernike runs built over a large radius leave "
            "the beam nearly flat and cannot pin the geometry. Also check that "
            "panel_span_px matches the panel box the probes were displayed on, and "
            "that the displayed phase actually covered the beam (the runbook "
            "writes one phase/CCD image per probe for exactly this).",
            best[1].correlation,
            best[1].panel_disc_radius,
            best[1].beam_waist_panel_px,
        )
    logger.info(
        "bench geometry: aperture={} px, w0={:.1f} panel px, {:.3f} cam px per "
        "model px, sigma {:.2f} cam px, correlation {:.4f}",
        best[1].panel_disc_radius,
        best[1].beam_waist_panel_px,
        best[1].camera_px_per_model_px,
        best[1].spot_fwhm_camera_px,
        best[1].correlation,
    )
    return 


def calibrate_bench_geometry(
    records: Sequence[CalibrationRecord],
    *,
    wavelength: float = LASER_WAVELENGTH_M,
    focal_length: float = 0.125,
    slm_pitch: float = SLM_PITCH_M,
    camera_pixel: float = 2.2e-6,
    region: int = 256,
    far_field_size: int = 4096,
    disc_candidates: Sequence[int] = (600, 450, 300, 250, 200, 150),
    panel_span_px: float | None = None,
) -> BenchGeometry:
    """Derive the model-to-camera geometry from real captures, offline.

    Two quantities are solved in sequence, both against the *measured* spot
    rather than a loss:

    1. **Beam waist.** The model's spot sigma scales as ``1 / w0``, so one probe
       evaluation gives the scale factor needed to match the measured sigma.
    2. **Panel extent.** For each candidate radius the panel phase is cropped to
       that disc, mapped onto the model grid, and the resulting far field is
       compared with the measurement by Pearson correlation.

    Correlation, not MSE, is the scoring metric on purpose: with a peaked spot
    the peak-normalised MSE is dominated by the spot's tail energy and reports
    ~0.007 for patterns that do not actually correspond, whereas the speckle
    correlation only peaks when the phase mapping is right.

    Args:
        records: Real captures, each with the applied panel ``phase`` and the
            measured far-field ``image``. Two or more distinct phases are needed
            for the speckle patterns to be comparable; a single record is
            accepted but scores less reliably.
        wavelength: Laser wavelength, metres.
        focal_length: 2f lens focal length, metres.
        slm_pitch: SLM pixel pitch, metres.
        camera_pixel: Camera pixel pitch, metres.
        region: Model grid side length in pupil pixels.
        far_field_size: Zero-padding size for the forward model.
        disc_candidates: Aperture semi-axes to try, in **model** pixels. Values
            above ``region / 2`` are clipped to it: the aperture is the disc
            inscribed in the model grid, because the runbook maps the model grid
            one-to-one onto the illuminated panel box. (Interpreting these as
            *panel* pixels, as earlier revisions did, made every candidate above
            ``region / 2`` collapse onto the same array, so the search could only
            ever report a saturated edge.)
        panel_span_px: Width of the panel region the model grid covers, in panel
            pixels (twice the runbook's ``--collect-disc``). **Required for real
            captures.** The model grid samples the pupil at
            ``panel_span_px / region`` times the SLM pitch, not at the SLM pitch
            itself; using ``slm_pitch`` directly makes the modelled aperture
            physically smaller than the real beam (2.05 mm vs 7.2 mm on this
            bench, a 3.5x error) and the predicted focal spot ~200x wider than
            the measured one, so no aperture can correlate.

    Returns:
        The best-scoring :class:`BenchGeometry`.

    Raises:
        ValueError: If ``records`` is empty or no disc radius yields a finite
            spot sigma.
    """
    if not records:
        raise ValueError("calibrate_bench_geometry needs at least one record")

    # Camera-side reference: sigma of the brightest real capture.
    camera_px_per_model_px, fwhm_meas, model_to_panel, reference, reference_phase = _cbg_measure_references(camera_pixel, far_field_size, focal_length, panel_span_px, records, region, slm_pitch, wavelength)

    def pupil(semi_axis_model_px: int) -> np.ndarray | None:
        """The pupil phase over the aperture, mapped onto the model grid.

        The aperture spans ``2 * semi_axis_model_px`` model pixels. How that maps
        onto the record's ``phase`` array depends on what the array represents:

        * ``region x region`` -- it is already the model grid (the runbook builds
          each probe on the model grid and scales it onto the panel box), so it
          is used unchanged;
        * anything else -- it is a panel-sized phase, and the aperture covers
          ``2 * semi_axis_model_px * model_to_panel`` panel pixels of it, which is
          cropped out and rescaled onto the grid.

        This used to always crop a centred sub-window and rescale, which
        double-counted the mapping in the first case and made every candidate at
        or above ``region / 2`` produce the same array -- a search that could
        only ever return its own edge.
        """
        a = int(min(int(semi_axis_model_px), region // 2))
        if a < _MIN_PUPIL_SEMI_AXIS_PX:
            return None
        span_panel = 2.0 * a * model_to_panel
        for phase in (reference_phase,):
            h, w = phase.shape
            if (h, w) == (region, region):
                return phase
            r_y, r_x = span_panel / 2.0, span_panel / 2.0
            y0, y1 = int(round(h / 2.0 - r_y)), int(round(h / 2.0 + r_y))
            x0, x1 = int(round(w / 2.0 - r_x)), int(round(w / 2.0 + r_x))
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                return None  # aperture does not fit; skip this candidate
            sub = phase[y0:y1, x0:x1]
            if sub.size == 0:
                return None
            return np.asarray(
                zoom(sub, (region / sub.shape[0], region / sub.shape[1]), order=1),
                dtype=np.float64,
            )
        return None

    best, waist_grid_candidates = _cbg_scan_seeds(region)
    for semi_axis in disc_candidates:
        semi_axis = int(min(int(semi_axis), region // 2))
        grid = pupil(semi_axis)
        if grid is None:
            continue
        opt = ZernikeCoefficientOptimizer(
            n_orders=1,
            region=region,
            dtype="float64",
            device="cpu",
            far_field_size=far_field_size,
        )
        zeros = np.zeros(opt.n_coefficients)
        for waist_grid in waist_grid_candidates:
            model = opt.forward_intensity(
                zeros, grid, _pupil_amplitude(region, waist_grid, semi_axis)
            )
            window = _model_window(model, reference.shape, camera_px_per_model_px)
            if window.shape != reference.shape:
                # Real frames are 250x248, not square, so the window is resampled
                # per axis rather than through the square-only _resize_grid.
                window = np.asarray(
                    zoom(
                        window,
                        (
                            reference.shape[0] / window.shape[0],
                            reference.shape[1] / window.shape[1],
                        ),
                        order=1,
                    ),
                    dtype=np.float64,
                )
                if window.shape != reference.shape:
                    continue
            a = window / (window.max() + 1e-30)
            b = reference / (reference.max() + 1e-30)
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            if not np.isfinite(corr):
                continue
            disc_px = int(round(semi_axis * model_to_panel))
            geo = BenchGeometry(
                panel_disc_radius=disc_px,
                region=region,
                beam_waist_panel_px=float(waist_grid * model_to_panel),
                far_field_size=far_field_size,
                camera_px_per_model_px=float(camera_px_per_model_px),
                spot_fwhm_camera_px=fwhm_meas,
                spot_fwhm_model_px=float(_spot_fwhm(model)),
                correlation=corr,
            )
            if best is None or geo.correlation > best[1].correlation:
                best = (corr, geo)

    if best is None:
        raise ValueError(
            "no candidate panel radius produced a usable model; check the record "
            "phase shapes and far_field_size"
        )
    # A low correlation, or a winner sitting on the edge of the search grid,
    # means the data cannot discriminate the geometry -- not that the geometry
    # was found. Returning the argmax silently would dress that up as a
    # calibration, so say it plainly.
    _cbg_finalize(best, model_to_panel, region)
    return best[1]

