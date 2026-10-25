"""Parity of :class:`ml.sim2f.optics.Sim2f` against the calibrated BenchTwin.

``drivers/sim/bench_twin.py`` is the repository's fitted 2f twin; this package's
forward model must agree with it, or the labels and the closed loop would be
computed on two different benches.  Everything here is *mapped by name* from
:class:`~ml.sim2f.config.Sim2fConfig` to
:class:`~ao_shaping.drivers.sim.bench_twin.BenchTwinConfig`, with ``noise_adu=0``
and ``errors=TwinErrorModel(**knobs)`` (``zero_order_eta`` has no twin
counterpart and is excluded; it is tested separately at the bottom).

Sampling convention
===================
``Sim2fConfig.sampling_ratio = 2.099`` is *cell size over pixel size*, so one
camera pixel covers ``1 / 2.099 = 0.476`` of a model sample: the resampling
matrix bins part of a sample per pixel (an interpolation, since the camera
oversamples the model).  ``BenchTwin`` does not resample at all -- it **crops**
``crop_px`` model samples and presents them as the camera window -- so a
like-for-like image comparison needs the two grids to coincide.  The parity
config does exactly that: ``ccd_px = grid`` with
``ccd_pixel_um = far_field_pitch`` gives ``sampling_ratio = 1`` (one camera pixel
= one model sample), and the twin's central ``grid x grid`` crop covers the same
interval, sample for sample.  At the reference pitch (2.2 um) the two windows
differ in extent by 5% and a comparison would measure the field-of-view mismatch
rather than the physics.

Normalisation
=============
``BenchTwin.frame`` renormalises **every** frame to ``peak_adu`` at its own peak,
i.e. it discards the absolute scale -- the same "renormalise per frame"
convention the calibrated bench explicitly rejects for photometry.  Images are
therefore compared peak-normalised, which is the only like-for-like form
available; the flat-referenced comparison is a pure scale factor away, and that
is pinned separately.

Measured (float64, three random Zernike phases, ``grid = 128``)
==============================================================
======================  ==========  ============
knobs                   flat frame  shaped frame
======================  ==========  ============
ideal (no knobs)        3.4e-16     1.9e-4
``defocus_m`` 10.08 mm  2.9e-5      3.9e-4
======================  ==========  ============

Both are far inside the 2e-3 / 5e-3 budgets.  The residual is the device LUT
round trip: ``BenchTwin`` always wraps the phase to ``[0, 2*pi)`` and quantises it
to 1023 levels, while this package skips the round trip when ``lut_gamma == 1`` --
that would only add a 6 mrad quantisation, and the ideal path must stay
bit-identical to "no panel at all".
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ao_shaping.drivers.sim.bench_twin import (
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)
from ml.sim2f.config import OpticsErrors, Sim2fConfig
from ml.sim2f.optics import Sim2f

#: Reference geometry; large enough that the parity numbers are the bench's,
#: small enough to run in a test.
BASE = Sim2fConfig()
#: ``sampling_ratio == 1``: one camera pixel is one model sample, so the twin's
#: central crop and this package's camera frame are the same interval.
PARITY = replace(BASE, ccd_px=BASE.grid, ccd_pixel_um=BASE.far_field_pitch_m * 1e6)

#: The fitted non-identity optical knob of the 2026-10-23 calibration.
DEFOCUS_M = 0.010084735


def _twin_config(cfg: Sim2fConfig) -> BenchTwinConfig:
    """Map a ``Sim2fConfig`` onto ``BenchTwinConfig`` field by field.

    Args:
        cfg: Bench configuration.

    Returns:
        The equivalent twin configuration (noise off, centre crop to ``grid``).
    """
    errors = cfg.errors
    return BenchTwinConfig(
        wavelength_m=cfg.wavelength_m,
        slm_pixel_um=cfg.panel_pixel_um,
        ccd_pixel_um=cfg.ccd_pixel_um,
        focal_length_m=cfg.focal_length_m,
        region=cfg.grid,
        region_span_px=cfg.region_span_px,
        far_field_padding=cfg.far_field_padding,
        beam_w0_px=cfg.beam_w0_px,
        center_crop=True,
        crop_px=None,  # -> region == grid
        peak_adu=1.0,
        noise_adu=0.0,
        seed=0,
        errors=TwinErrorModel(
            lut_gamma=errors.lut_gamma,
            lut_phase_max_rad=errors.lut_phase_max_rad,
            gray_levels=errors.gray_levels,
            slm_zernike=dict(errors.slm_zernike),
            slm_zernike_n_max=errors.slm_zernike_n_max,
            slm_roughness_sigma_rad=errors.slm_roughness_sigma_rad,
            slm_roughness_seed=errors.slm_roughness_seed,
            defocus_m=errors.defocus_m,
            tilt_x_rad=errors.tilt_x_rad,
            tilt_y_rad=errors.tilt_y_rad,
            lens_zernike=dict(errors.lens_zernike),
            lens_zernike_n_max=errors.lens_zernike_n_max,
        ),
    )


def _relative_l2(measured: np.ndarray, reference: np.ndarray) -> float:
    """``||measured - reference||_2 / ||reference||_2``.

    Args:
        measured: Image under test.
        reference: Reference image.

    Returns:
        Relative L2 distance.
    """
    difference = np.asarray(measured, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return float(np.linalg.norm(difference) / np.linalg.norm(reference))


def _phases(sim: Sim2f, count: int = 3) -> list[np.ndarray]:
    """Deterministic random Zernike phases, in raw radians.

    Args:
        sim: Bench the phases are built on.
        count: Number of phases.

    Returns:
        List of ``(grid, grid)`` float64 phase maps.
    """
    rng = np.random.default_rng(0)
    phases = []
    for _ in range(count):
        coefficients = rng.normal(0.0, 0.7, size=10)
        phases.append(
            sim.aberration(coefficients[None, :])[0].numpy().astype(np.float64)
        )
    return phases


@pytest.mark.parametrize("cfg", [BASE, PARITY], ids=["reference-pitch", "parity-pitch"])
def test_pitches_and_scales_match_the_twin(cfg: Sim2fConfig) -> None:
    """Every geometric scale is the twin's, to 1e-12 relative."""
    twin = BenchTwin(_twin_config(cfg))
    sim = Sim2f(cfg, device="cpu")

    assert sim.grid == twin.region
    assert sim.fft_n == twin.far_field_size
    assert cfg.fft_length == twin.far_field_size
    assert cfg.pupil_pitch_m == pytest.approx(twin.pupil_pitch_m, rel=1e-12)
    assert cfg.far_field_pitch_m == pytest.approx(twin.p_fft, rel=1e-12)
    assert cfg.ccd_pitch_m == pytest.approx(twin.ccd_pixel_m, rel=1e-12)
    assert cfg.sampling_ratio == pytest.approx(twin.camera_px_per_model_px(), rel=1e-12)
    if cfg.ccd_pixel_um == BASE.ccd_pixel_um:
        # The reference geometry's published numbers, as an anchor.
        assert cfg.pupil_pitch_m == pytest.approx(56.25e-6, rel=1e-12)
        assert cfg.far_field_pitch_m == pytest.approx(4.618055555555556e-6, rel=1e-9)
        assert cfg.sampling_ratio == pytest.approx(2.0991161616161618, rel=1e-9)
    else:
        # The parity geometry sits exactly on the twin's crop grid.
        assert cfg.sampling_ratio == pytest.approx(1.0, rel=1e-12)


def test_entrance_amplitude_matches_the_twin() -> None:
    """The Gaussian entrance beam is the twin's, sample for sample."""
    cfg = BASE
    twin = BenchTwin(_twin_config(cfg))
    sim = Sim2f(cfg, device="cpu", dtype=torch.float64)

    mine = sim.nominal_illumination[0, 0].numpy()
    theirs = twin.pupil_amplitude()
    assert mine.shape == theirs.shape
    assert float(np.abs(mine - theirs).max()) <= 1e-12
    assert float(mine.max()) == pytest.approx(1.0, rel=1e-12)


def test_images_match_with_ideal_optics() -> None:
    """Ideal bench: peak-normalised frames agree to 1.9e-4 (budget 2e-3)."""
    sim = Sim2f(PARITY, device="cpu", dtype=torch.float64)
    twin = BenchTwin(_twin_config(PARITY))
    zero = np.zeros((PARITY.grid, PARITY.grid), dtype=np.float64)

    flat_mine = sim.forward(zero[None]).numpy()[0, 0]
    flat_theirs = twin.frame(zero)
    assert _relative_l2(flat_mine / flat_mine.max(), flat_theirs / flat_theirs.max()) < 2e-3

    worst = 0.0
    for phase in _phases(sim):
        mine = sim.forward(phase[None]).numpy()[0, 0]
        theirs = twin.frame(phase)
        worst = max(worst, _relative_l2(mine / mine.max(), theirs / theirs.max()))
    assert worst < 2e-3


def test_images_match_with_the_calibrated_defocus() -> None:
    """The fitted 10.08 mm defocus agrees to 3.9e-4 (budget 5e-3)."""
    cfg = replace(PARITY, errors=OpticsErrors(defocus_m=DEFOCUS_M))
    sim = Sim2f(cfg, device="cpu", dtype=torch.float64)
    twin = BenchTwin(_twin_config(cfg))
    zero = np.zeros((cfg.grid, cfg.grid), dtype=np.float64)

    flat_mine = sim.forward(zero[None]).numpy()[0, 0]
    flat_theirs = twin.frame(zero)
    assert _relative_l2(flat_mine / flat_mine.max(), flat_theirs / flat_theirs.max()) < 5e-3

    worst = 0.0
    for phase in _phases(sim):
        mine = sim.forward(phase[None]).numpy()[0, 0]
        theirs = twin.frame(phase)
        worst = max(worst, _relative_l2(mine / mine.max(), theirs / theirs.max()))
    assert worst < 5e-3


def test_flat_reference_differs_only_by_the_twins_per_frame_peak() -> None:
    """``frame(phase)/frame(flat).max()`` is peak-normalised, not flat-referenced.

    ``BenchTwin.frame`` multiplies by ``peak_adu / cropped.max()``, so its
    "flat-normalised" image is the *shape* and the absolute scale is gone: a
    defocused frame looks as bright as an in-focus one.  This package keeps the
    flat reference (the bench needs it for photometry), so the two differ by
    exactly the peak ratio -- pinned here.
    """
    cfg = replace(PARITY, errors=OpticsErrors(defocus_m=DEFOCUS_M))
    sim = Sim2f(cfg, device="cpu", dtype=torch.float64)
    twin = BenchTwin(_twin_config(cfg))
    zero = np.zeros((cfg.grid, cfg.grid), dtype=np.float64)
    phase = _phases(sim)[0]

    flat_mine = sim.forward(zero[None]).numpy()[0, 0]
    flat_theirs = twin.frame(zero)
    mine = sim.forward(phase[None]).numpy()[0, 0] / flat_mine.max()
    theirs = twin.frame(phase) / flat_theirs.max()  # already peak-normalised

    scale = float(mine.max() / theirs.max())
    # The scale is not 1: a shaped frame can peak above the defocused flat frame
    # (the pupil phase partly refocuses), which is exactly the information the
    # twin's per-frame normalisation throws away.
    assert abs(scale - 1.0) > 1e-3
    assert _relative_l2(mine / scale, theirs) < 5e-3


def test_zero_order_has_no_twin_counterpart_but_behaves_as_documented() -> None:
    """``zero_order_eta = 1`` leaves only the unmodulated beam.

    ``TwinErrorModel`` has no 0-order term (``bench_twin`` predates the
    ``nonideal.ZeroOrderModel`` wiring), so this is checked against the physics
    directly: at ``eta = 1`` the pupil field collapses onto the incident
    amplitude, the phase cannot modulate anything, and the frame is the flat spot
    scaled by ``sqrt(1 - eta) = 0`` in the modulated channel.
    """
    # ``OpticsErrors`` caps eta below 1 (eta = 1 is a fully dead panel), so the
    # limit is taken at 0.999: the residual is the 3% amplitude leak of the
    # modulated channel, i.e. ``1 + 0.063 cos(phase)`` on top of the flat spot.
    cfg = replace(PARITY, errors=OpticsErrors(zero_order_eta=0.999))
    sim = Sim2f(cfg, device="cpu")
    zero = np.zeros((1, cfg.grid, cfg.grid), dtype=np.float32)
    # A smooth low-order Zernike phase: it keeps the light inside the camera
    # window, so the comparison is about the 0-order term, not about clipping.
    phase = _phases(sim)[0].astype(np.float32)

    def normalised(bench: Sim2f, command: np.ndarray) -> np.ndarray:
        frame = bench.forward(command[None])[0, 0].numpy()
        return frame / float(frame.max())

    flat = normalised(sim, zero[0])
    leaky = normalised(sim, phase)

    # Measured 1.3e-2: the modulated channel is 3% in amplitude, so the flat spot
    # only gains the 1 + 0.063 cos(phase) fringe.
    assert _relative_l2(leaky, flat) < 0.05
    # ... whereas without the 0-order term the same phase reshapes the frame
    # completely (measured 2.04, i.e. the flat frame is not even in the same
    # direction).
    full = Sim2f(PARITY, device="cpu")
    assert _relative_l2(normalised(full, phase), flat) > 0.5
    # The pupil split is an *energy* fraction: amplitudes are square roots.
    assert sim._zero_weights == pytest.approx((np.sqrt(0.001), np.sqrt(0.999)), rel=1e-9)
    partial = Sim2f(
        replace(cfg, errors=OpticsErrors(zero_order_eta=0.25)), device="cpu"
    )
    assert partial._zero_weights == pytest.approx((np.sqrt(0.75), 0.5), rel=1e-12)


