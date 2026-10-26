"""Parameter-sweep regression tests for the telescope far-field link (issue #204).

Pins the DFT far-field engine in :mod:`ao_shaping.utils.wavefront.link_design`
against analytic diffraction theory across wavelength / aperture / focal length /
carrier / pattern-frequency sweeps. This is the fixed (CI) form of the manual
sweep in ``notebooks/telescope_link_spot_size_design.ipynb``.

Acceptance tolerance is 3 CCD pixels (7 um) everywhere, matching the notebook's
cell-by-cell assertions; the measured deviations are dominated by zero/peak
localisation on the pixel grid (<= 2 px), well inside tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.wavefront.link_design import (
    TelescopeLinkParams,
    airy_first_zero,
    carrier_phase,
    carrier_shift,
    ccd_to_far,
    circular_pupil,
    conversion_scale,
    far_field_dft,
    far_to_ccd,
    grating_factor,
    sinc_first_zero,
    square_pupil,
)

# Locked design point.
P = TelescopeLinkParams()
DX = P.exit_pupil_pixel          # 80 um pupil pixel
DU, N_OUT = 2.15e-6, 1024        # focal grid
F = P.effective_focal_length     # 2 m
LAM = P.wavelength               # 1064 nm
TOL = 3 * P.ccd_pixel            # 3 CCD px = 7 um

# Pre-built pupils (module scope, built once).
CIRC90 = circular_pupil(P.pupil_grid, DX, P.exit_pupil_diameter)
CIRC45 = circular_pupil(P.pupil_grid, DX, 0.045)
SQ30 = square_pupil(P.pupil_grid, DX, 30e-3)
SQ15 = square_pupil(P.pupil_grid, DX, 15e-3)
SQ5 = square_pupil(P.pupil_grid, DX, 5e-3)


# ---------- measurement helpers (same conventions as the sweep script) ----------

def _radial_first_zero(I, r_exp):
    row = I[N_OUT // 2, N_OUT // 2:]
    lo = int(0.6 * r_exp / DU)
    hi = min(int(1.25 * r_exp / DU), len(row) - 1)
    k = lo + int(np.argmin(row[lo:hi]))
    return k * DU


def _row_first_zero(I, r_exp, x0=0.0):
    row = I[N_OUT // 2]
    c = N_OUT // 2 + int(round(x0 / DU))
    lo = int(c + 0.6 * r_exp / DU)
    hi = min(int(c + 1.25 * r_exp / DU), N_OUT - 1)
    k = lo + int(np.argmin(row[lo:hi]))
    return (k - c) * DU


def _row_peak(I, r_exp, side=+1):
    row = I[N_OUT // 2]
    b_in = int(N_OUT // 2 + side * 0.5 * r_exp / DU)
    b_out = int(N_OUT // 2 + side * 1.8 * r_exp / DU)
    lo = max(0, min(b_in, b_out))
    hi = min(N_OUT - 1, max(b_in, b_out))
    if lo >= hi:
        lo, hi = 0, N_OUT - 1
    k = lo + int(np.argmax(row[lo:hi]))
    return (k - N_OUT // 2) * DU


def _centroid_offset(I):
    I_n = I / I.sum()
    yy, xxg = np.mgrid[0:N_OUT, 0:N_OUT]
    cx = (xxg * I_n).sum() - N_OUT // 2
    cy = (yy * I_n).sum() - N_OUT // 2
    return np.hypot(cx, cy) * DU


# ---------- S0 baseline + golden values ----------

def test_baseline_airy_zero_reproduces_notebook():
    I = np.abs(far_field_dft(CIRC90, DX, LAM, F)[0]) ** 2
    d0 = airy_first_zero(LAM, F, P.exit_pupil_diameter)
    assert abs(_radial_first_zero(I, d0) - d0) < TOL


def test_conversion_golden_values():
    assert abs(conversion_scale(P) - 500.0) < 1e-9
    w, px = far_to_ccd(0.10, P)
    assert abs(w - 200e-6) / 200e-6 < 1e-9
    assert abs(px - 90.9) < 0.05
    assert abs(ccd_to_far(1e-3, P) - 0.50) < 1e-9  # 1 mm CCD -> 50 cm far


def test_airy_zero_golden_value():
    d0 = airy_first_zero(LAM, F, P.exit_pupil_diameter)
    assert abs(d0 - 28.85e-6) / d0 < 1e-3
    assert abs(d0 / P.ccd_pixel - 13.1) < 0.1


def test_carrier_shift_golden_value():
    # Golden value uses the far-field carrier angle theta_far = theta_ramp / M_t
    # (2 mrad SLM side -> 0.2 mrad exit side): f_eff * theta_far = f_c * theta_ramp.
    off = carrier_shift(F, P.carrier_ramp / P.telescope_magnification)
    assert abs(off - 400e-6) / off < 1e-9
    assert abs(off / P.ccd_pixel - 181.8) < 0.1


def test_sampling_budget_golden_values():
    nu_max = P.telescope_magnification * (0.10 / P.target_distance) / (2 * LAM)
    period_px = 1.0 / nu_max / P.slm_pixel
    assert abs(nu_max - 470.6) < 1.0 and abs(period_px - 265.6) < 0.5
    nu_car = P.carrier_ramp / LAM
    per_car = 1.0 / nu_car / P.slm_pixel
    assert abs(nu_car - 1879.7) < 1.0 and abs(per_car - 66.5) < 0.5
    N90 = P.exit_pupil_diameter * (0.10 / P.target_distance) / LAM
    N100 = P.objective_diameter * (0.10 / P.target_distance) / LAM
    assert abs(N90 - 8.46) < 0.05 and abs(N100 - 9.40) < 0.05
    dz = LAM * P.target_distance ** 2 / P.exit_pupil_diameter ** 2
    assert abs(dz - 131.4) < 0.5


# ---------- S1 wavelength sweep ----------

@pytest.mark.parametrize("lam", [532e-9, 1064e-9, 1550e-9])
def test_wavelength_scaling(lam):
    I = np.abs(far_field_dft(CIRC90, DX, lam, F)[0]) ** 2
    d0 = airy_first_zero(lam, F, P.exit_pupil_diameter)
    assert abs(_radial_first_zero(I, d0) - d0) < TOL


# ---------- S2 aperture sweep ----------

def test_circular_45mm():
    I = np.abs(far_field_dft(CIRC45, DX, LAM, F)[0]) ** 2
    d0 = airy_first_zero(LAM, F, 0.045)
    assert abs(_radial_first_zero(I, d0) - d0) < TOL


@pytest.mark.parametrize("w,mask", [(30e-3, SQ30), (15e-3, SQ15), (5e-3, SQ5)])
def test_square_sinc_scaling(w, mask):
    I = np.abs(far_field_dft(mask, DX, LAM, F)[0]) ** 2
    d0 = sinc_first_zero(LAM, F, w)
    assert abs(_row_first_zero(I, d0) - d0) < TOL


# ---------- S3 focal-length sweep ----------

@pytest.mark.parametrize("f", [1.0, 4.0])
def test_focal_length_scaling(f):
    I = np.abs(far_field_dft(CIRC90, DX, LAM, f)[0]) ** 2
    d0 = airy_first_zero(LAM, f, P.exit_pupil_diameter)
    assert abs(_radial_first_zero(I, d0) - d0) < TOL


# ---------- S4 carrier sweep ----------

@pytest.mark.parametrize("th", [0.1e-3, 0.2e-3])
def test_carrier_shift_centroid(th):
    pupil = (SQ30 * carrier_phase(P.pupil_grid, DX, th, LAM)).astype(complex)
    I = np.abs(far_field_dft(pupil, DX, LAM, F)[0]) ** 2
    assert abs(_centroid_offset(I) - carrier_shift(F, th)) < TOL


def test_carrier_shift_peak_large_offset():
    th = 0.4e-3
    pupil = (SQ30 * carrier_phase(P.pupil_grid, DX, th, LAM)).astype(complex)
    I = np.abs(far_field_dft(pupil, DX, LAM, F)[0]) ** 2
    assert abs(abs(_row_peak(I, carrier_shift(F, th), +1)) - carrier_shift(F, th)) < TOL


def test_carrier_does_not_change_width():
    th = 0.2e-3
    pupil = (SQ30 * carrier_phase(P.pupil_grid, DX, th, LAM)).astype(complex)
    I = np.abs(far_field_dft(pupil, DX, LAM, F)[0]) ** 2
    w0 = sinc_first_zero(LAM, F, 30e-3)
    assert abs(_row_first_zero(I, w0, x0=carrier_shift(F, th)) - w0) < TOL


# ---------- S5 pattern-frequency sweep ----------

@pytest.mark.parametrize("nu", [47.0, 100.0, 200.0, 400.0])
def test_grating_order_position(nu):
    pupil = (CIRC90 * grating_factor(P.pupil_grid, DX, nu)).astype(complex)
    I = np.abs(far_field_dft(pupil, DX, LAM, F)[0]) ** 2
    pos = nu * LAM * F
    p1, p2 = abs(_row_peak(I, pos, +1)), abs(_row_peak(I, pos, -1))
    assert abs((p1 + p2) / 2.0 - pos) < TOL


# ---------- S6 sampling-budget invariants (no DFT) ----------

def test_focal_window_and_nyquist_budget():
    u_half = (N_OUT / 2) * DU
    assert abs(u_half - 1.1e-3) < 1e-6
    M = P.telescope_magnification
    # Content on the SLM plane (notebook presentation): carrier + 10 cm pattern.
    nu_car_slm = P.carrier_ramp / LAM                        # 1879.7 m^-1
    nu_pat_slm = M * (0.10 / P.target_distance) / (2 * LAM)  # 470.6 m^-1
    content_slm = nu_car_slm + nu_pat_slm                    # ~2350 m^-1
    # On the exit-pupil plane (the DFT's native plane) content is /M; the
    # content/Nyquist and content/window ratios are plane-independent.
    content_exit = content_slm / M                           # 235.0 m^-1
    nu_nyq_exit = 1.0 / (2 * DX)                             # 6250 m^-1
    assert abs(nu_nyq_exit - 6250.0) < 1e-9
    for lam in (532e-9, 1064e-9, 1550e-9):
        nu_win_exit = u_half / (lam * F)
        assert content_exit / nu_win_exit < 1.0              # ~45 % of window
        assert content_exit / nu_nyq_exit < 0.10             # ~3.8 % of Nyquist
    # SLM-plane Nyquist margin (the notebook's 53 px vs 2 px claim).
    nu_nyq_slm = 1.0 / (2 * P.slm_pixel)                     # 62500 m^-1
    assert content_slm / nu_nyq_slm < 0.10
    # Focal-plane scale is 2.3 % coarser than the CCD pixel; px readouts carry
    # that systematic bias — must stay within 3 % for the 3-px tolerance to hold.
    assert abs(1 - DU / P.ccd_pixel) < 0.03


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
