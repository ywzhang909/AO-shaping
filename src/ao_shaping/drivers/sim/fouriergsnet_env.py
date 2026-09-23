"""FourierGSNet simulation environment (SimSLM, SimCCD, SimFourierGSNetEnv).

Digital twin of the real 2f Fourier bench used by ``fouriergsnet_optimize.py``:
an SLM panel (Santec SLM200, 1920x1200) at the front focal plane of a lens
(f = 0.125 m) with a CCD camera at the back focal plane. The CCD far field is
the Fraunhofer diffraction pattern of the SLM field, sampled on the CCD pixel
lattice.

Physics model
-------------
* Fraunhofer propagation with the FFT convention of the real pipeline
  (``fouriergsnet_optimize.py:prop``)::

      E = fftshift(fft2(ifftshift(U), norm="ortho"))

  so the DC (0-order) sits at the frame centre.
* The FFT grid pitch is ``p_fft = lamb * f / (P * d_slm)`` where ``P`` is the
  padded grid size (next power of two >= max(K_px, beam region)). The CCD
  lattice is obtained by bilinearly resampling the P x P FFT intensity onto
  ``K_px x K_px`` pixels (``scipy.ndimage.zoom``, factor ``K_px / P``). This
  maps an FFT bin offset ``D`` to ``D * K_px / P`` CCD pixels, so a grating of
  period ``P_slm`` SLM pixels produces its +1 order exactly at
  ``K_px / P_slm`` pixels from the 0-order (the K-law).
* Physical pixel envelope: each SLM pixel has a finite active width
  ``d_eff``, so the far field is multiplied by a separable sinc envelope
  ``sinc(d_eff * u / (lamb * f))`` (first null at ``K_px * d_slm / d_eff``
  CCD pixels). The default ``d_eff = 7.8 um`` puts the first null at
  ~2100 px, outside the K=2048 band edge; the sinc-envelope test uses
  ``d_eff = 4 * d_slm`` so the null lands at ``K_px / 4 = 512 px`` inside
  the band.
* The beam is a Gaussian of width ``w0`` (default 250 px) on a
  ``region x region`` patch (default 512) centred at ``beam_center``
  (default panel centre (960, 600) in (row, col)). The patch is placed on
  the P x P grid so the beam centre lands on bin ``P / 2`` (DC at centre).
* Static aberrations (``env.aberrations``, Noll-index -> coefficient) are
  generated with ``ZernikeGenerator`` (radius = region / 2) and added to the
  displayed phase before the single gray quantization in ``render_intensity``
  — identical to writing ``base + aberration`` through ``display_phase``.
  Zernike polynomials are only defined inside the unit circle, so the
  aberration phase is NaN outside it and is zeroed there (``nan_to_num``).

Duck-typing contract (mirrors the real Santec SLM / MiiCam drivers)
-------------------------------------------------------------------
* ``slm.display_phase(phase_rad, wait_time_s=, memory_number=, memory_mode=) -> int``
* ``slm.display_data(gray_uint16, ...) -> int``
* ``slm.get_displayed_memory_number() -> int``
* ``ccd.get_numpy_image(n_sample=1) -> np.uint16``
* ``MEMORY_MODE_INTERNAL == 0``, ``Panel_Res == (1920, 1200)``

The gray depth is 255 (matching ``SLMLUTCalibrator.factory_2pi = 255.0``);
``display_phase`` stores the ideal (unquantized) phase and the quantization
happens once in ``render_intensity``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from scipy import ndimage

from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

# Hardware contract literals (mirror the real Santec SLM200 driver).
MEMORY_MODE_INTERNAL = 0
PANEL_RES = (1920, 1200)  # (width, height) — driver convention
PANEL_H, PANEL_W = 1920, 1200  # calibrator convention (h, w)

# Default optical bench parameters (2f Fourier bench).
LAMBDA = 1064e-9
D_SLM = 8e-6
D_CCD = 2.2e-6
F_LENS = 0.125
D_EFF = 7.8e-6


@dataclass
class BeamParams:
    """Gaussian beam parameters on the SLM panel.

    ``native`` selects the **native-pitch square-panel** geometry (default
    ``False`` keeps the historical 1920x1200 anisotropic panel). When ``True``
    the displayed model hologram sits at its **native pixel pitch** on a small
    square aperture (region x region) so that the model's FFT-based forward
    (``fouriergsnet_optimize.prop``) and the env's ``render_intensity`` FFT
    agree — this is what lets the closed loop actually shape a square.
    """

    region: int = 512
    w0: float = 250.0
    center: tuple[int, int] | None = None  # (row, col) on the panel; None = panel center
    native: bool = False


@dataclass
class CalibNoise:
    """Calibration-noise model injected into the rendered far field.

    ``deltaK_fraction`` perturbs Kx/Ky per-axis by a random fraction,
    ``center_offset_px`` shifts the whole frame by a random offset,
    ``rotation_deg`` rotates the frame, and ``scale`` multiplies K globally
    (deterministic).
    """

    deltaK_fraction: float = 0.0
    center_offset_px: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0


class SimSLM:
    """Simulated Santec SLM200 (duck-typed to the real driver contract)."""

    MEMORY_MODE_INTERNAL = 0
    Panel_Res = (1920, 1200)
    _max_gray = 255  # factory LUT: 2*pi at gray 255 (SLMLUTCalibrator.factory_2pi)

    def __init__(
        self,
        env: "SimFourierGSNetEnv",
        lut_nonlinearity: float = 0.0,
        amplitude_coupling: float = 0.0,
        panel_res: tuple[int, int] | None = None,
    ) -> None:
        self._env = env
        self.lut_nonlinearity = lut_nonlinearity
        self.amplitude_coupling = amplitude_coupling
        self.panel_h, self.panel_w = (
            (panel_res[0], panel_res[1]) if panel_res is not None else (PANEL_H, PANEL_W)
        )
        self._phase = np.zeros((self.panel_h, self.panel_w), dtype=np.float64)
        self.phase_calls: list[dict[str, Any]] = []
        self.data_calls: list[dict[str, Any]] = []
        self.displayed_memory_number = 0
        self._version = 0

    def display_phase(
        self,
        phase_rad: np.ndarray,
        wait_time_s: float = 0.0,
        memory_number: int | None = None,
        memory_mode: int = 0,
    ) -> int:
        """Store the ideal phase (radians); quantization happens at render time."""
        phase = np.asarray(phase_rad, dtype=np.float64)
        if phase.shape != (self.panel_h, self.panel_w):
            raise ValueError(
                f"display_phase expects ({self.panel_h}, {self.panel_w}) phase, got {phase.shape}"
            )
        self._phase = phase
        self.phase_calls.append(
            {
                "wait_time_s": float(wait_time_s),
                "memory_number": memory_number,
                "memory_mode": int(memory_mode),
            }
        )
        self._version += 1
        return 0

    def display_data(
        self,
        gray_uint16: np.ndarray,
        wait_time_s: float = 0.0,
        memory_number: int | None = None,
        memory_mode: int = 0,
    ) -> int:
        """Store a raw uint16 grayscale pattern (already quantized)."""
        gray = np.asarray(gray_uint16, dtype=np.uint16)
        if gray.shape != (self.panel_h, self.panel_w):
            raise ValueError(
                f"display_data expects ({self.panel_h}, {self.panel_w}) gray, got {gray.shape}"
            )
        self._phase = self._gray_to_phase(gray)
        self.data_calls.append(
            {
                "wait_time_s": float(wait_time_s),
                "memory_number": memory_number,
                "memory_mode": int(memory_mode),
            }
        )
        self._version += 1
        return 0

    def get_displayed_memory_number(self) -> int:
        return self.displayed_memory_number

    def _phase_to_gray(self, phase: np.ndarray) -> np.ndarray:
        return np.round(np.mod(phase, 2 * np.pi) / (2 * np.pi) * self._max_gray).astype(
            np.uint16
        )

    def _gray_to_phase(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float64) / self._max_gray
        return (g * 2 * np.pi + self.lut_nonlinearity * np.sin(2 * np.pi * g)).astype(
            np.float64
        )


class SimCCD:
    """Simulated CCD camera at the back focal plane (duck-typed to MiiCam)."""

    def __init__(
        self,
        env: "SimFourierGSNetEnv",
        peak_photons: float = 60000.0,
        read_noise_e: float = 0.0,
        noise_enabled: bool = True,
        seed: int = 0,
    ) -> None:
        self._env = env
        self.peak_photons = peak_photons
        self.read_noise_e = read_noise_e
        self.noise_enabled = noise_enabled
        self._rng = np.random.default_rng(seed)

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        """Render the far field and return a uint16 frame (K_px x K_px)."""
        intensity = self._env.render_intensity()
        intensity = intensity / intensity.max()
        if not self.noise_enabled:
            return np.round(intensity * self.peak_photons).astype(np.uint16)
        frame = np.zeros_like(intensity)
        for _ in range(int(n_sample)):
            photons = self._rng.poisson(intensity * self.peak_photons)
            if self.read_noise_e > 0:
                photons = photons + self._rng.normal(
                    0.0, self.read_noise_e, size=intensity.shape
                )
            frame += photons
        frame /= n_sample
        return np.clip(np.round(frame), 0, 65535).astype(np.uint16)


class SimFourierGSNetEnv:
    """Digital twin of the 2f Fourier bench (SLM -> lens -> CCD)."""

    def __init__(
        self,
        lamb: float = LAMBDA,
        d_slm: float = D_SLM,
        d_ccd: float = D_CCD,
        f: float = F_LENS,
        d_eff: float = D_EFF,
        K_px: int | None = None,
        beam: BeamParams | None = None,
        calib_noise: CalibNoise | None = None,
        peak_photons: float = 60000.0,
        read_noise_e: float = 0.0,
        noise_enabled: bool = True,
        seed: int = 0,
    ) -> None:
        self.lamb = lamb
        self.d_slm = d_slm
        self.d_ccd = d_ccd
        self.f = f
        self.d_eff = d_eff
        self.K_px = (
            int(round(lamb * f / (d_slm * d_ccd))) if K_px is None else int(K_px)
        )
        self.beam = beam if beam is not None else BeamParams()
        self.calib_noise = calib_noise if calib_noise is not None else CalibNoise()
        self._rng = np.random.default_rng(seed)
        self._native = self.beam.native
        # Native mode: center defaults to panel center; region must equal the
        # model hologram size (64) for a 1:1 native-pitch mapping.
        if self.beam.center is None:
            if self._native:
                self.beam.center = (int(self.beam.region) // 2, int(self.beam.region) // 2)
            else:
                self.beam.center = (960, 600)
        self.P = max(
            1 << (self.K_px - 1).bit_length(),
            1 << (self.beam.region - 1).bit_length(),
        )
        self.p_fft = lamb * f / (self.P * d_slm)
        # Native mode: square isotropic panel with pitch == region, so a 64x64
        # model hologram sits at native pixel pitch (no band-limiting upscale)
        # and the env FFT agrees with fouriergsnet_optimize.prop().
        if self._native:
            self.PANEL_H = self.PANEL_W = int(self.beam.region)
            self.slm = SimSLM(self, panel_res=(int(self.beam.region), int(self.beam.region)))
        else:
            self.PANEL_H, self.PANEL_W = PANEL_H, PANEL_W
            self.slm = SimSLM(self)
        self.ccd = SimCCD(
            self,
            peak_photons=peak_photons,
            read_noise_e=read_noise_e,
            noise_enabled=noise_enabled,
            seed=seed,
        )
        self.aberrations: dict[int, float] = {}
        self._zgen: ZernikeGenerator | None = None
        self._beam_amp = self._build_beam_amp()
        self._render_cache: np.ndarray | None = None
        self._render_key: tuple[Any, ...] | None = None
        # Time-varying turbulence (Ornstein-Uhlenbeck) — inactive by default.
        self._turb_rng: np.random.Generator | None = None
        self._turb_nolls: tuple[int, ...] = ()
        self._turb_sigma: float = 0.0
        self._turb_tau: float = 1.0
        self._turb_dt: float = 1.0
        self._turb_state: dict[int, float] = {}

    # -- time-varying turbulence (Ornstein-Uhlenbeck) -----------------------

    def configure_turbulence(
        self,
        seed: int = 42,
        sigma: float = 0.5,
        tau: float = 50.0,
        dt: float = 1.0,
        nolls: tuple[int, ...] = (4, 5, 6, 11, 13),
    ) -> None:
        """Enable a time-varying low-order aberration drift.

        Each configured Noll coefficient follows a mean-reverting
        Ornstein-Uhlenbeck process advanced one step per
        :meth:`advance_time` call (one closed-loop frame)::

            x <- x - (x / tau) * dt + sigma * sqrt(2 * dt / tau) * N(0, 1)

        The stationary distribution is ``N(0, sigma**2)``, so ``sigma`` is the
        stationary standard deviation in **radians** (same unit as
        ``self.aberrations``) and ``tau`` is the relaxation time in **frames**.
        Per-mode state is reset to ``0.0`` and a private seeded generator is
        created so trajectories are reproducible. ``self.aberrations`` is left
        untouched until the first :meth:`advance_time` step.
        """
        self._turb_rng = np.random.default_rng(seed)
        self._turb_nolls = tuple(int(n) for n in nolls)
        self._turb_sigma = float(sigma)
        self._turb_tau = float(tau)
        self._turb_dt = float(dt)
        self._turb_state = {noll: 0.0 for noll in self._turb_nolls}
        logger.debug(
            "Turbulence configured: seed={}, sigma={}, tau={}, dt={}, nolls={}",
            seed,
            self._turb_sigma,
            self._turb_tau,
            self._turb_dt,
            self._turb_nolls,
        )

    def advance_time(self) -> None:
        """Advance the OU turbulence by one frame and write ``self.aberrations``.

        No-op when turbulence is not configured (backward compatible). Applied
        coefficients are in **radians**, consistent with the static aberration
        semantics; values are not wrapped.
        """
        if self._turb_rng is None or not self._turb_nolls:
            return
        sigma = self._turb_sigma
        tau = self._turb_tau
        dt = self._turb_dt
        noise_scale = sigma * np.sqrt(2.0 * dt / tau)
        for noll in self._turb_nolls:
            x = self._turb_state.get(noll, 0.0)
            x = x - (x / tau) * dt + noise_scale * float(
                self._turb_rng.standard_normal()
            )
            self._turb_state[noll] = x
            self.aberrations[noll] = x

    @property
    def turbulence_active(self) -> bool:
        """True when :meth:`configure_turbulence` has enabled OU drift."""
        return self._turb_rng is not None and bool(self._turb_nolls)

    @property
    def turbulence_nolls(self) -> tuple[int, ...]:
        """Configured Noll indices (empty tuple when turbulence is inactive)."""
        return self._turb_nolls


    # -- public API ---------------------------------------------------------

    def render_intensity(self) -> np.ndarray:
        """Render the CCD far-field intensity (K_px x K_px, float64)."""
        key = (self.slm._version, tuple(sorted(self.aberrations.items())))
        if self._render_cache is not None and self._render_key == key:
            return self._render_cache
        P = self.P
        region = self.beam.region
        off = (P - region) // 2
        patch = self._extract_region(self.slm._phase)
        if self.aberrations:
            patch = patch + self._aberration_phase()
        patch = np.nan_to_num(patch, nan=0.0)
        U = np.zeros((P, P), dtype=np.complex128)
        U[off : off + region, off : off + region] = self._beam_amp * np.exp(1j * patch)
        E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U), norm="ortho"))
        u = (np.arange(P) - P / 2) * self.p_fft
        env = np.sinc(self.d_eff * u / (self.lamb * self.f))
        E = E * env[None, :] * env[:, None]
        I = np.abs(E) ** 2
        I = ndimage.zoom(I, self.K_px / P, order=1)
        I = self._apply_calib_noise(I)
        self._render_cache = I
        self._render_key = key
        return I

    def inject_calib_noise(self, calib: dict[str, Any]) -> dict[str, Any]:
        """Return a perturbed copy of a calibration dict (Kx/Ky/center/rotation)."""
        cn = self.calib_noise
        out = dict(calib)
        dkx = (
            self._rng.uniform(-cn.deltaK_fraction, cn.deltaK_fraction)
            if cn.deltaK_fraction > 0
            else 0.0
        )
        dky = (
            self._rng.uniform(-cn.deltaK_fraction, cn.deltaK_fraction)
            if cn.deltaK_fraction > 0
            else 0.0
        )
        out["Kx"] = out.get("Kx", self.K_px) * cn.scale * (1 + dkx)
        out["Ky"] = out.get("Ky", self.K_px) * cn.scale * (1 + dky)
        cx, cy = out.get("center", (self.K_px / 2, self.K_px / 2))
        dx = (
            self._rng.uniform(-cn.center_offset_px, cn.center_offset_px)
            if cn.center_offset_px > 0
            else 0.0
        )
        dy = (
            self._rng.uniform(-cn.center_offset_px, cn.center_offset_px)
            if cn.center_offset_px > 0
            else 0.0
        )
        out["center"] = (cx + dx, cy + dy)
        dr = (
            self._rng.uniform(-cn.rotation_deg, cn.rotation_deg)
            if cn.rotation_deg > 0
            else 0.0
        )
        out["rotation_deg"] = out.get("rotation_deg", 0.0) + dr
        return out

    # -- internals ----------------------------------------------------------

    def _build_beam_amp(self) -> np.ndarray:
        region = self.beam.region
        yy, xx = np.mgrid[0:region, 0:region]
        r2 = (yy - region / 2) ** 2 + (xx - region / 2) ** 2
        return np.exp(-r2 / (2 * self.beam.w0**2)).astype(np.float64)

    def _extract_region(self, panel_phase: np.ndarray) -> np.ndarray:
        region = self.beam.region
        ph, pw = self.slm.panel_h, self.slm.panel_w
        r0 = self.beam.center[0] - region // 2
        c0 = self.beam.center[1] - region // 2
        patch = np.zeros((region, region), dtype=np.float64)
        r_lo, r_hi = max(r0, 0), min(r0 + region, ph)
        c_lo, c_hi = max(c0, 0), min(c0 + region, pw)
        patch[r_lo - r0 : r_hi - r0, c_lo - c0 : c_hi - c0] = panel_phase[
            r_lo:r_hi, c_lo:c_hi
        ]
        return patch

    def _zernike_generator(self) -> ZernikeGenerator:
        if self._zgen is None:
            region = self.beam.region
            self._zgen = ZernikeGenerator((region, region), radius=region / 2, n_orders=6)
        return self._zgen

    def _aberration_phase(self) -> np.ndarray:
        z = self._zernike_generator()
        coeffs: dict[tuple[int, int], float] = {}
        for noll, c in self.aberrations.items():
            n, m = z.noll_to_nm(noll)
            coeffs[(n, m)] = c
        return z.generate_polynomial(coeffs)

    def _apply_calib_noise(self, I: np.ndarray) -> np.ndarray:
        cn = self.calib_noise
        dkx = (
            self._rng.uniform(-cn.deltaK_fraction, cn.deltaK_fraction)
            if cn.deltaK_fraction > 0
            else 0.0
        )
        dky = (
            self._rng.uniform(-cn.deltaK_fraction, cn.deltaK_fraction)
            if cn.deltaK_fraction > 0
            else 0.0
        )
        kx = self.K_px * cn.scale * (1 + dkx)
        ky = self.K_px * cn.scale * (1 + dky)
        if (kx, ky) != (self.K_px, self.K_px):
            I = ndimage.zoom(I, (ky / self.P, kx / self.P), order=1)
            I = self._crop_or_pad(I, self.K_px)
        if cn.rotation_deg != 0.0:
            I = ndimage.rotate(I, cn.rotation_deg, reshape=False, order=1)
        if cn.center_offset_px != 0.0:
            dy = (
                self._rng.uniform(-cn.center_offset_px, cn.center_offset_px)
                if cn.center_offset_px > 0
                else 0.0
            )
            dx = (
                self._rng.uniform(-cn.center_offset_px, cn.center_offset_px)
                if cn.center_offset_px > 0
                else 0.0
            )
            I = ndimage.shift(I, (dy, dx), order=1)
        return I

    @staticmethod
    def _crop_or_pad(I: np.ndarray, n: int) -> np.ndarray:
        h, w = I.shape
        if h == n and w == n:
            return I
        out = np.zeros((n, n), dtype=I.dtype)
        r0 = max((n - h) // 2, 0)
        c0 = max((n - w) // 2, 0)
        src_r0 = max((h - n) // 2, 0)
        src_c0 = max((w - n) // 2, 0)
        rr, cc = min(h, n), min(w, n)
        out[r0 : r0 + rr, c0 : c0 + cc] = I[src_r0 : src_r0 + rr, src_c0 : src_c0 + cc]
        return out