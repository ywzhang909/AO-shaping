"""Simulated 2f-Fourier SLM + CCD for running ``slm-pib`` without hardware.

The ``slm-pib`` optimizer (``optimizer/wfless/slm_zernike_pib.py``) drives a real
``Santec`` SLM through ``create_phase_from_array(phase_rad) -> uint16`` and
``display_data(gray, ...)`` and reads the far-field spot from a CCD. This module
provides a pure-numpy stand-in that reproduces that contract while actually
propagating light, so the SPGD search has a *real* optical feedback loop:

* :class:`SimSLMPib` — accepts a raw-radian phase, stores it, and on display
  computes the Fraunhofer far field of a Gaussian input beam modulated by the
  phase (``np.fft.fft2`` of the pupil field, centred). The 0-order spot sits at
  the image centre and its shape responds to the applied phase exactly as a 2f
  bench would.
* :class:`SimPibCCD` — a ``BaseCamera`` whose ``get_numpy_image`` simply returns
  the last far-field image produced by the sim SLM (plus a small Poisson-like
  shot-noise floor so the metric is not perfectly smooth).

The two are wired together through :class:`SimSLMPibSystem`, which also registers
a ``"sim"`` entry in the camera registry so ``create_camera("sim", ...)`` returns
a ``SimPibCCD`` bound to a shared system.

This is intentionally a *far-field FFT* model (the 2f Fourier bench in
``AGENTS.md``), not a near-field propagator: the SLM is at the front focal plane
and the CCD at the back focal plane, so the CCD image is the 2-D FFT of the SLM
pupil. That is exactly the regime ``slm-pib`` operates in.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.device_base import DeviceState

# SLM panel geometry (matches the real Santec SLM-200: 1920 x 1200, 10-bit).
SLM_W = 1920
SLM_H = 1200
SLM_BITS = 10
SLM_MAX_GRAY = (1 << SLM_BITS) - 1  # 1023
TWO_PI_GRAY = 993  # 2*pi in gray at 1064 nm (device dynamic value, see AGENTS.md)

# Input Gaussian beam radius (SLM pixels) — the waist of the flat-field beam
# incident on the SLM. Chosen so the beam comfortably fills the panel aperture.
BEAM_W0 = 400.0

# CCD far-field resolution (square). The 0-order spot lands at the image centre.
CCD_RES = (512, 512)


class SimPibSystem:
    """Shared optical state between the sim SLM and the sim CCD.

    Holds the *currently displayed* radian phase and the lazily-computed far
    field. A single instance is shared by the SLM (writer) and the CCD (reader)
    so the camera always sees the light that the SLM last displayed.
    """

    def __init__(
        self,
        slm_shape: tuple[int, int] = (SLM_H, SLM_W),
        ccd_res: tuple[int, int] = CCD_RES,
        beam_w0: float = BEAM_W0,
        noise_adu: float = 5.0,
        seed: int | None = None,
    ) -> None:
        self.slm_h, self.slm_w = slm_shape
        self.ccd_h, self.ccd_w = ccd_res
        self.beam_w0 = float(beam_w0)
        self.noise_adu = float(noise_adu)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._phase: np.ndarray = np.zeros((self.slm_h, self.slm_w), dtype=np.float64)
        self._far_field: np.ndarray | None = None
        self._gray: np.ndarray | None = None

    # --- SLM side --------------------------------------------------------

    def set_phase_rad(self, phase_rad: np.ndarray) -> None:
        """Store a raw-radian phase and mark the far field stale."""
        phase = np.asarray(phase_rad, dtype=np.float64)
        if phase.shape != (self.slm_h, self.slm_w):
            raise ValueError(
                f"phase shape {phase.shape} != SLM panel {(self.slm_h, self.slm_w)}"
            )
        with self._lock:
            self._phase = phase
            self._far_field = None  # stale until next display

    def set_gray(self, gray: np.ndarray) -> None:
        """Store a raw grayscale pattern (flat-gray path, no radian conversion)."""
        g = np.asarray(gray, dtype=np.uint16)
        if g.shape != (self.slm_h, self.slm_w):
            raise ValueError(f"gray shape {g.shape} != SLM panel {(self.slm_h, self.slm_w)}")
        with self._lock:
            self._gray = g
            self._far_field = None

    # --- CCD side --------------------------------------------------------

    def far_field(self) -> np.ndarray:
        """Compute (and cache) the far-field image of the displayed phase.

        The pupil field is ``A(r) * exp(i*phase)`` where ``A`` is the Gaussian
        input beam; the far field is ``|FFT(pupil)|^2`` with the zero-frequency
        component moved to the centre (``fftshift``). The 0-order spot is the
        image centre, matching the 2f-bench convention in ``AGENTS.md``.
        """
        with self._lock:
            if self._far_field is not None:
                return self._far_field.copy()
            phase = self._phase
            # Gaussian input beam amplitude on the SLM grid.
            yy, xx = np.mgrid[0 : self.slm_h, 0 : self.slm_w]
            cy, cx = self.slm_h / 2.0, self.slm_w / 2.0
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            amplitude = np.exp(-r2 / (2.0 * self.beam_w0**2))
            pupil = amplitude * np.exp(1j * phase)
            spectrum = np.fft.fftshift(np.abs(np.fft.fft2(pupil)) ** 2)
            # Normalise to a 0..255-ish grayscale so exposure/peak logic behaves
            # like a real camera (peak ~100 for the flat beam).
            peak = float(spectrum.max())
            if peak > 0:
                spectrum = spectrum * (100.0 / peak)
            self._far_field = spectrum.astype(np.float64)
            return self._far_field.copy()

    def far_field_noisy(self) -> np.ndarray:
        """Far field plus shot + read noise (returned as float, ~0..255 scale)."""
        img = self.far_field()
        if self.noise_adu > 0:
            shot = self._rng.poisson(np.clip(img, 0, None).astype(np.float64) / 10.0)
            img = img + (shot - img / 10.0) * (self.noise_adu / 10.0)
            img = img + self._rng.normal(0.0, self.noise_adu / 10.0, img.shape)
        return np.clip(img, 0.0, None)


# A single process-wide system so the registry-created CCD and the monkeypatched
# SLM share the same optical state.
_SYSTEM: SimPibSystem | None = None


def get_system(seed: int | None = None) -> SimPibSystem:
    """Return the process-wide :class:`SimPibSystem` (created on first use)."""
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = SimPibSystem(seed=seed)
    return _SYSTEM


def reset_system(seed: int | None = None) -> SimPibSystem:
    """Replace the process-wide system (used between matrix cells)."""
    global _SYSTEM
    _SYSTEM = SimPibSystem(seed=seed)
    return _SYSTEM


class SimSLMPib:
    """Simulated Santec SLM exposing the exact contract the optimizer uses.

    Only the methods ``slm_zernike_pib`` actually call are implemented:
    ``create_phase_from_array``, ``display_data``, ``set_grayscale``, ``open``,
    ``close``, ``is_connected``, ``__enter__``/``__exit__``.
    """

    def __init__(self, *args: Any, system: SimPibSystem | None = None, **kwargs: Any) -> None:
        self.system = system or get_system()
        self._open = False
        # Track the last displayed radian phase for reporting.
        self.last_phase_rad: np.ndarray | None = None

    # --- context manager / lifecycle -------------------------------------

    def open(self) -> None:
        self._open = True
        logger.debug("SimSLMPib opened (no hardware)")

    def close(self) -> None:
        self._open = False

    def is_connected(self) -> bool:
        return self._open

    def __enter__(self) -> "SimSLMPib":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- phase / gray API (matches Santec contract) ----------------------

    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        """Radian phase -> uint16 grayscale, AND arm the optical system.

        On the real Santec this only converts rad->gray (2*pi = ~993 gray) and
        leaves the panel untouched until ``display_data``. Here we additionally
        store the radian phase so the next ``far_field()`` reflects it. The
        returned grayscale is the radian->gray mapping (wrap + scale to 1023).
        """
        rad = np.asarray(phase_rad, dtype=np.float64)
        # Store the raw radian phase so the far field uses it.
        self.system.set_phase_rad(rad)
        self.last_phase_rad = rad.copy()
        # radian -> grayscale (2*pi maps to TWO_PI_GRAY), wrapping like the driver.
        gray = (np.mod(rad, 2.0 * np.pi) / (2.0 * np.pi) * TWO_PI_GRAY)
        return gray.astype(np.uint16)

    def display_data(
        self,
        gray: np.ndarray,
        memory_number: int | None = None,
        memory_mode: int | None = None,
    ) -> int:
        """Display a grayscale pattern.

        The optimizer always calls ``create_phase_from_array`` (which arms the
        phase) before ``display_data``, so by the time this runs the far field
        is already correct. A bare grayscale display (no preceding phase) is
        treated as a flat-phase pattern: its mean gray maps to a constant phase
        offset, which does not change the far-field *shape* (only the 0-order
        coupling), so we leave the stored phase as-is.
        """
        self._open = True
        # Force the far field to be (re)computed on next CCD read.
        self.system.set_gray(np.asarray(gray, dtype=np.uint16))
        return memory_number if memory_number is not None else 0

    def set_grayscale(self, gray: int | np.ndarray) -> None:
        """Set a uniform grayscale (used on exit to blank the panel)."""
        if isinstance(gray, (int, float)):
            arr = np.full((self.system.slm_h, self.system.slm_w), int(gray), dtype=np.uint16)
        else:
            arr = np.asarray(gray, dtype=np.uint16)
        self.system.set_gray(arr)


class SimPibCCD(BaseCamera):
    """Simulated CCD returning the sim SLM's far field.

    Wired to the process-wide :class:`SimPibSystem`. Registered as camera type
    ``"sim"`` so ``create_camera("sim", ...)`` yields an instance of this class.
    """

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 80.0,
        cam_size: int = 250,
        skip_sampling: bool = False,
        system: SimPibSystem | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=skip_sampling)
        self.cam_size = int(cam_size)
        self.system = system or get_system()
        self._open = False

    # --- BaseCamera interface -------------------------------------------

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def is_connected(self) -> bool:
        return self._open

    def initialize(self) -> None:
        self._open = True

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        """Return the current far field (with noise), cropped to ``cam_size``.

        The optimizer windows the CCD to ``cam_size`` around the spot; here we
        return the central ``cam_size x cam_size`` crop of the full far field so
        the spot (at the image centre) stays inside the window.
        """
        if not self._open:
            raise CameraError("SimPibCCD not opened")
        full = self.system.far_field_noisy()
        h, w = full.shape
        s = self.cam_size
        if s >= h or s >= w:
            img = full
        else:
            y0 = (h - s) // 2
            x0 = (w - s) // 2
            img = full[y0 : y0 + s, x0 : x0 + s]
        return img.astype(np.float64)

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (size, (0, 0))

    def reset_exposure_time(self, time_ms: float) -> float:
        self.exposure_time_ms = float(time_ms)
        return float(time_ms)

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        return True

    def get_auto_exposure_state(self) -> dict:
        return {"enabled": False, "mode": 1, "target": 0}

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        return True

    @staticmethod
    def get_cam_list() -> list:
        return ["sim"]

    # --- exposure compatibility helpers ---------------------------------

    @property
    def min_exposure_ms(self) -> float:
        return 0.011

    @property
    def max_exposure_ms(self) -> float:
        return 10_000.0

    def auto_exposure(
        self,
        target_max: float = 40.0,
        tolerance: float = 5.0,
        max_iterations: int = 20,
        n_sample: int = 1,
    ) -> np.ndarray:
        """Return the far field scaled so its peak ~= target_max (0-255 scale)."""
        img = self.get_numpy_image(n_sample=n_sample)
        peak = float(img.max())
        if peak > 0 and target_max > 0:
            img = img * (target_max / peak)
        return img


def register_sim_camera() -> None:
    """Register the ``"sim"`` camera type so ``create_camera("sim", ...)`` works."""
    from ao_shaping.drivers.ccd.common import register_camera

    try:
        register_camera("sim", SimPibCCD)
    except Exception as exc:  # already registered
        logger.debug("sim camera already registered: {}", exc)
