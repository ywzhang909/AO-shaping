"""TDD physics tests for the FourierGSNet simulation environment.

Verifies the 2f Fourier-bench physics model of ``SimFourierGSNetEnv``:

* the K-law — a blaze grating of period ``P_slm`` SLM pixels produces its
  +1 order exactly ``K_px / P_slm`` CCD pixels from the 0-order;
* the physical pixel envelope — a separable sinc whose first null sits at
  ``K_px * d_slm / d_eff`` CCD pixels;
* static aberration additivity — ``env.aberrations`` equals writing
  ``base + aberration`` directly through ``display_phase``, and a Noll-3
  tilt shifts the far-field centroid by ``c * K / (pi * R)`` pixels;
* the duck-typing contract the real ``fouriergsnet_optimize.py`` pipeline
  (SLMCCDCalibrator / SLMLUTCalibrator / ShapingSystem) relies on.

All tests are seedable, run on CPU with small K (2048), and use
``noise_enabled=False`` so the physics is deterministic.
"""
from __future__ import annotations

import numpy as np

from ao_shaping.drivers.sim.fouriergsnet_env import BeamParams, SimFourierGSNetEnv
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

K_SMALL = 2048
D_SLM = 8e-6
D_EFF_TEST = 4 * D_SLM  # envelope first null at K_px/4 = 512 px inside the band
PANEL_H, PANEL_W = 1920, 1200


def _blaze_phase(period_px: int) -> np.ndarray:
    """x-axis blaze grating phase (radians), replicating SLMCCDCalibrator._blaze."""
    return np.tile((2 * np.pi * np.arange(PANEL_W) / period_px), (PANEL_H, 1))


def _threshold_centroid(img: np.ndarray, thresh: float = 0.5) -> tuple[float, float]:
    """Intensity-weighted centroid of pixels above ``thresh * max``."""
    yy, xx = np.nonzero(img > thresh * img.max())
    w = img[yy, xx].astype(np.float64)
    return (float(np.average(yy, weights=w)), float(np.average(xx, weights=w)))


def test_k_law(seed: int = 0) -> None:
    """Grating period P_slm -> +1 order displacement == K_px / P_slm (within 3%)."""
    env = SimFourierGSNetEnv(K_px=K_SMALL, noise_enabled=False, seed=seed)
    center = K_SMALL / 2
    for period in (32, 64):
        env.slm.display_phase(_blaze_phase(period))
        frame = env.ccd.get_numpy_image(n_sample=1)
        _, col = _threshold_centroid(frame)
        disp = abs(col - center)
        expected = K_SMALL / period
        assert abs(disp - expected) / expected < 0.03, (
            f"period={period}: disp={disp:.3f}, expected={expected:.3f}"
        )


def test_sinc_envelope_null(seed: int = 0) -> None:
    """Flat phase: separable sinc envelope first null at K_px * d_slm / d_eff (within 5%)."""
    env = SimFourierGSNetEnv(
        K_px=K_SMALL,
        d_eff=D_EFF_TEST,
        beam=BeamParams(w0=1.0),  # delta-like beam -> far field ~ envelope
        noise_enabled=False,
        seed=seed,
    )
    env.slm.display_phase(np.zeros((PANEL_H, PANEL_W)))
    frame = env.ccd.get_numpy_image(n_sample=1)
    center = K_SMALL // 2
    profile = frame[center, :].astype(np.float64)
    null_offset = 20 + int(np.argmin(profile[center + 20:]))
    expected = K_SMALL * D_SLM / D_EFF_TEST  # 512 px
    assert abs(null_offset - expected) / expected < 0.05, (
        f"null at {null_offset} px, expected {expected} px"
    )


def test_static_aberration_additivity(seed: int = 0) -> None:
    """Static aberrations add to the displayed base phase (env path == direct path)."""
    region = 512
    zgen = ZernikeGenerator((region, region), radius=region / 2, n_orders=6)
    base = zgen.generate_polynomial({(2, 0): 0.3})  # small defocus
    tilt = zgen.generate_polynomial({(1, -1): 1.0})  # Noll 3 = (1,-1) tilt along rows
    r0, c0 = 960 - region // 2, 600 - region // 2

    env = SimFourierGSNetEnv(
        K_px=K_SMALL, beam=BeamParams(w0=100.0), noise_enabled=False, seed=seed
    )
    panel_base = np.zeros((PANEL_H, PANEL_W))
    panel_base[r0:r0 + region, c0:c0 + region] = base
    env.slm.display_phase(panel_base)
    env.aberrations = {3: 1.0}  # Noll 3
    frame_env = env.ccd.get_numpy_image(n_sample=1)

    env_ref = SimFourierGSNetEnv(
        K_px=K_SMALL, beam=BeamParams(w0=100.0), noise_enabled=False, seed=seed
    )
    panel_ref = np.zeros((PANEL_H, PANEL_W))
    panel_ref[r0:r0 + region, c0:c0 + region] = base + tilt
    env_ref.slm.display_phase(panel_ref)
    frame_ref = env_ref.ccd.get_numpy_image(n_sample=1)

    assert np.allclose(frame_env, frame_ref, rtol=1e-2)

    # tilt centroid shift: c * K / (pi * R) = 2.55 px at c=1, R=256, K=2048
    env_flat = SimFourierGSNetEnv(
        K_px=K_SMALL, beam=BeamParams(w0=100.0), noise_enabled=False, seed=seed
    )
    env_flat.slm.display_phase(np.zeros((PANEL_H, PANEL_W)))
    frame_flat = env_flat.ccd.get_numpy_image(n_sample=1)
    cy_env, _ = _threshold_centroid(frame_env)
    cy_flat, _ = _threshold_centroid(frame_flat)
    shift = cy_env - cy_flat
    expected = 1.0 * K_SMALL / (np.pi * (region / 2))
    assert abs(shift - expected) / expected < 0.2, (
        f"tilt shift {shift:.3f} px, expected {expected:.3f} px"
    )


def test_device_ducktyping_contract(seed: int = 0) -> None:
    """SimSLM/SimCCD satisfy the duck-typing contract of fouriergsnet_optimize.py."""
    env = SimFourierGSNetEnv(K_px=K_SMALL, seed=seed)
    slm = env.slm
    ccd = env.ccd

    assert slm.MEMORY_MODE_INTERNAL == 0
    assert slm.Panel_Res == (1920, 1200)

    ret = slm.display_phase(
        np.zeros((PANEL_H, PANEL_W)), wait_time_s=0.1, memory_number=3, memory_mode=0
    )
    assert isinstance(ret, int)
    assert len(slm.phase_calls) == 1
    call = slm.phase_calls[0]
    assert call["wait_time_s"] == 0.1
    assert call["memory_number"] == 3
    assert call["memory_mode"] == 0

    ret2 = slm.display_data(np.zeros((PANEL_H, PANEL_W), dtype=np.uint16), wait_time_s=0.1)
    assert isinstance(ret2, int)
    assert len(slm.data_calls) == 1

    assert isinstance(slm.get_displayed_memory_number(), int)

    frame = ccd.get_numpy_image(n_sample=1)
    assert frame.dtype == np.uint16
    assert frame.shape == (K_SMALL, K_SMALL)