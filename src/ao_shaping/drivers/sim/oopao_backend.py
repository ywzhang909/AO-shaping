"""OOPAO-based turbulence phase-screen generator and angular-spectrum propagator.

Re-bases AO-shaping's simulation internals on the OOPAO library
(github.com/cheritier/OOPAO, ESO/LAM) instead of the hand-rolled FFT-spectrum
Kolmogorov path in :mod:`beam_backend`. Two primitives are ported from the
reference implementation (beaconless-ao-sim/physics):

* :class:`OopaoScreenBackend` — per-sample von-Karman phase screens via
  ``Atmosphere.generateNewPhaseScreen(seed)``, with per-slab r0 rescaling so the
  generated screens are statistically equivalent to the legacy aotools/FFT path.
* :class:`Propagator` (and the module-level :func:`propagate_asm`) —
  angular-spectrum propagation via OOPAO ``Atmosphere.ASM``, with a pure-numpy
  transcription fallback when OOPAO is unavailable.

OOPAO is imported through :mod:`ao_shaping.drivers.sim._oopao_compat`, which
shadows the broken upstream ``OOPAO/__init__.py`` (numpy 2.x incompatible) by
loading only the submodules we need. The OOPAO library is optional: when it is
not installed, :func:`_oopao_available` returns False and callers stay on the
numpy/legacy path (see ``AO_OOPAO_BACKEND`` in :mod:`beam_backend`).

Per-layer r0 calibration
------------------------
OOPAO's ``cn2`` bookkeeping divides the total Cn2 by ``max(altitude)``, which
does not correctly slice the path r0 into per-layer r0 (per-layer phase
variance comes out too strong). We therefore bypass it: each layer is
generated at the reference r0 ``_R0_REF_500 = 0.15`` (OOPAO's ``r0_def``, at
its hardcoded 500 nm convention) and amplitude-rescaled so its per-slab r0 is
*exactly* ``r0_slab = r0_path * n**(3/5)`` at the simulation wavelength — the
same per-slab r0 the legacy FFT path uses. Because the von-Karman PSD scales as
``r0**(-5/3)``, phase amplitude ~ ``r0**(-5/6)``; :func:`_rescale_for` applies
exactly that and nothing else.

Known limitation: no inner scale
--------------------------------
``Atmosphere.__init__`` has no ``l0`` parameter and ``generateNewPhaseScreen``
calls ``ft_sh_phase_screen`` without one, so OOPAO always generates with
``l0 = 1e-10`` — its inner-scale rolloff sits far above any grid Nyquist
frequency and is effectively absent. A configured ``l_min`` therefore cannot be
honoured on this backend. That only matters when ``l_min`` is large enough to
be grid-resolved; :func:`inner_scale_is_resolvable` detects that case and the
routing layer warns.
"""

from __future__ import annotations

import contextlib
import io
from functools import lru_cache
from typing import Any

import numpy as np

try:
    from ao_shaping.drivers.sim._oopao_compat import (
        Atmosphere,
        Source,
        Telescope,
    )

    _OOPAO_AVAILABLE = True
except Exception:  # pragma: no cover - OOPAO not installed
    _OOPAO_AVAILABLE = False

__all__ = [
    "OopaoScreenBackend",
    "Propagator",
    "make_screens",
    "propagate_asm",
    "rayleigh_range",
    "inner_scale_is_resolvable",
    "_oopao_available",
]

# Reference r0 used to generate the OOPAO layers before rescaling. Arbitrary
# (OOPAO's own r0_def); each layer's phase is amplitude-rescaled to the target
# per-slab r0 by _rescale_for(). No empirical normalization constant is needed:
# OOPAO's subharmonic-augmented screen already reproduces the analytic
# von-Karman variance for the r0 it was generated at.
_R0_REF_500 = 0.15

# OOPAO's Atmosphere hardcodes a 500 nm convention and cannot be given an inner
# scale. The legacy path's inner-scale rolloff has a characteristic frequency
# fm = 5.92 / (2*pi*l_min); it is only *resolvable* once that drops to the grid
# Nyquist frequency 1 / (2*dx). Below that, l_min is sub-pixel and physically
# irrelevant, so OOPAO ignoring it is harmless.
_L_MIN_RESOLVABLE_FACTOR = 5.92 / np.pi


def inner_scale_is_resolvable(l_min: float, pixel_size: float) -> bool:
    """Whether a configured ``l_min`` is coarse enough for the grid to resolve.

    The legacy generator rolls off high frequencies above
    ``fm = 5.92 / (2*pi*l_min)``. That rolloff only affects the represented
    spectrum once ``fm`` reaches the grid Nyquist frequency ``1/(2*dx)``, i.e.
    ``l_min >~ 5.92*dx/pi``. Below that threshold ``l_min`` is sub-pixel and
    this backend's inability to represent an inner scale is irrelevant.

    Args:
        l_min: Configured inner scale [m].
        pixel_size: Grid pixel pitch [m].

    Returns:
        True when the inner scale is coarse enough that the legacy path and
        OOPAO would legitimately disagree.
    """
    if l_min <= 0.0 or pixel_size <= 0.0:
        return False
    return float(l_min) > _L_MIN_RESOLVABLE_FACTOR * float(pixel_size)


def _oopao_available() -> bool:
    """Return True when the OOPAO library was imported successfully."""
    return _OOPAO_AVAILABLE


def compute_r0(lam: float, cn2: float, L: float) -> float:
    """Fried parameter ``r0`` for a single turbulent layer.

    ``r0 = (0.423 * k**2 * Cn2 * L)**(-3/5)``, with ``k = 2*pi/lam``.
    """
    k = 2.0 * np.pi / lam
    return float((0.423 * k**2 * cn2 * L) ** (-3.0 / 5.0))


def _rescale_for(r0_slab: float) -> float:
    """Amplitude rescale from OOPAO's reference screen to the target r0.

    OOPAO's ``layer.OPD`` is **phase in radians** at its hardcoded 500 nm
    convention, generated at ``r0=_R0_REF_500`` (note: ``generateNewPhaseScreen``
    overwrites the initial metre-valued OPD with a radian-valued screen, so the
    attribute named "OPD" on a layer is radians).

    Its std scales exactly as ``r0**(-5/6)`` (verified over a 10x r0 range:
    fitted exponent -0.8333 vs theory -5/6, normalization constant 0.619322
    constant to 6 digits). Rescaling to a target r0 is therefore *exact* and
    requires no extra factor:

    * **No wavelength factor.** The phase statistics at the simulation
      wavelength are fully determined by the Fried parameter, and ``r0_slab``
      is already computed at the simulation wavelength (``compute_r0`` uses
      ``k = 2*pi/lam``). Since ``r0 ∝ lam**(6/5)`` and phase-in-radians
      ∝ ``r0**(-5/6) ∝ 1/lam``, an explicit ``lam`` term would double-count and
      cancel the wavelength dependence entirely. (An earlier revision carried
      ``lam/_LAM_REF_500``; that factor exactly cancelled the r0 dependence and
      made this backend's phase std *wavelength-independent*, which is
      unphysical for a screen in radians.)
    * **No ``_CAL_REF`` division.** OOPAO's own normalization is the physically
      correct target: its subharmonic-augmented screen reproduces the analytic
      von-Karman variance, whereas the legacy plain-FFT path *under*-represents
      low-order power. Dividing by an empirical constant would discard that.
    * **No ``sqrt(1.03)`` factor.** That constant belongs to the piston-removed
      *pure Kolmogorov* (L0 -> inf) aperture variance; applying it while
      simulating a finite outer scale is a category error.

    Args:
        r0_slab: Target per-slab Fried parameter [m] at the simulation
            wavelength.
    """
    return (_R0_REF_500 / float(r0_slab)) ** (5.0 / 6.0)


class OopaoScreenBackend:
    """Generate per-sample turbulence screens via OOPAO ``Atmosphere``.

    Parameters
    ----------
    N : int
        Pupil-grid side length in pixels.
    dx : float
        Pixel scale in metres.
    Dscope : float
        Telescope diameter in metres (defines the OOPAO pupil).
    lam : float
        Simulation wavelength in metres.
    cn2 : float
        Cn2 in ``m**(-2/3)``.
    L : float
        Propagation path length in metres.
    L0 : float
        Outer scale in metres.
    n_screens : int
        Number of turbulence layers / screens.
    """

    def __init__(
        self,
        N: int,
        dx: float,
        Dscope: float,
        lam: float,
        cn2: float,
        L: float,
        L0: float,
        n_screens: int,
    ) -> None:
        if not _OOPAO_AVAILABLE:
            raise RuntimeError(
                "OOPAO is not available; install the oopao dependency to use "
                "OopaoScreenBackend"
            )
        self.N = int(N)
        self.n_screens = int(n_screens)
        self.lam = float(lam)

        # OOPAO prints banner tables (Telescope/Atmosphere setup) to stdout;
        # redirect them so the library stays silent in our logs/tests.
        with contextlib.redirect_stdout(io.StringIO()):
            self.tel = Telescope(
                resolution=self.N, diameter=float(Dscope), fov=0.0, samplingTime=0.001
            )

            self.src = Source(optBand="R", magnitude=0.0, display_properties=False)
            self.src * self.tel

            r0_path = compute_r0(self.lam, float(cn2), float(L))
            self.r0_slab = r0_path * self.n_screens ** (3.0 / 5.0)

            self._rescale = _rescale_for(self.r0_slab)

            n = self.n_screens
            self._altitudes = np.linspace(50.0, float(L) - 50.0, n).tolist()
            self._frac = [1.0 / n] * n

            self.atm = Atmosphere(
                self.tel,
                r0=_R0_REF_500,
                L0=float(L0),
                windSpeed=[10.0] * n,
                fractionalR0=self._frac,
                windDirection=[0.0] * n,
                altitude=self._altitudes,
                src=self.src,
            )
            self.atm.initializeAtmosphere(self.tel, compute_covariance=False)

    def make_screens(
        self, seed: int, r0_slab: float | None = None
    ) -> np.ndarray:
        """Draw ``n_screens`` fresh OOPAO screens for sample ``seed``.

        Parameters
        ----------
        seed : int
            Sample seed; OOPAO reseeds every layer with ``seed + i_layer``.
        r0_slab : float, optional
            Per-sample per-slab r0 [m]. When given, the reference-r0 OOPAO
            layers are rescaled to this target per-slab r0 instead of the
            constant-L ``self.r0_slab``. A constant amplitude rescale is a
            statistically exact r0 change (PSD shape in r/L0/l0 preserved).

        Returns
        -------
        np.ndarray
            ``(n_screens, N, N)`` float32 phase screens in radians, center-cropped
            from OOPAO's ``N+4``-pixel layers and rescaled to the target per-slab
            r0.
        """
        rescale = self._rescale
        if r0_slab is not None:
            rescale = _rescale_for(float(r0_slab))
        with contextlib.redirect_stdout(io.StringIO()):
            self.atm.generateNewPhaseScreen(seed=int(seed))
        out = np.empty((self.n_screens, self.N, self.N), dtype=np.float32)
        for i in range(self.n_screens):
            lay = getattr(self.atm, "layer_%d" % (i + 1))
            out[i] = (np.asarray(lay.OPD)[2:-2, 2:-2] * rescale).astype(
                np.float32
            )
        return out


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (used by beam_backend)
# ---------------------------------------------------------------------------


def make_screens(
    *,
    N: int,
    dx: float,
    Dscope: float,
    lam: float,
    cn2: float,
    L: float,
    L0: float,
    n_screens: int,
    seed: int,
    l0: float | None = None,
) -> np.ndarray:
    """Build an :class:`OopaoScreenBackend` and draw one screen batch.

    Thin convenience wrapper matching :func:`beam_backend.turbulence_phase`'s
    parameter surface; ``l0`` is accepted for API parity but unused (OOPAO's
    ``Atmosphere`` does not take an inner scale directly).
    """
    backend = _get_backend(
        N=N,
        dx=dx,
        Dscope=Dscope,
        lam=lam,
        cn2=cn2,
        L=L,
        L0=L0,
        n_screens=n_screens,
    )
    return backend.make_screens(seed=seed)


@lru_cache(maxsize=8)
def _get_backend(
    *,
    N: int,
    dx: float,
    Dscope: float,
    lam: float,
    cn2: float,
    L: float,
    L0: float,
    n_screens: int,
) -> OopaoScreenBackend:
    """Return a process-lifetime, config-keyed :class:`OopaoScreenBackend`.

    Cached because constructing OOPAO's Telescope + Atmosphere is expensive
    (~100 ms) and verbose; the seeds passed to ``make_screens`` fully determine
    the generated screens (verified deterministic per seed), so reusing one
    backend instance across samples is safe.
    """
    return OopaoScreenBackend(
        N=N,
        dx=dx,
        Dscope=Dscope,
        lam=lam,
        cn2=cn2,
        L=L,
        L0=L0,
        n_screens=n_screens,
    )


# ---------------------------------------------------------------------------
# Angular-spectrum propagation (OOPAO ASM + numpy fallback)
# ---------------------------------------------------------------------------


def rayleigh_range(w0: float, lam: float) -> float:
    """Return the Rayleigh range z_R = pi * w0**2 / lam."""
    return float(np.pi * w0**2 / lam)


_ASM_HOST: Any = None


def _asm_host() -> Any:
    """Lazily build a tiny OOPAO ``Atmosphere`` used only as an ASM host.

    ``ASM`` is a pure function; it does not depend on the host Atmosphere's
    resolution/layer configuration. A minimal telescope + source + single-layer
    atmosphere is used solely to host the angular-spectrum call (OOPAO requires
    the telescope to have a Source registered first, hence ``src * tel``).
    """
    global _ASM_HOST
    if _ASM_HOST is None:
        if not _OOPAO_AVAILABLE:
            raise RuntimeError("OOPAO is not available for the ASM host")
        with contextlib.redirect_stdout(io.StringIO()):
            tel = Telescope(resolution=8, diameter=1.0, fov=0.0, samplingTime=0.001)
            src = Source(optBand="R", magnitude=0.0, display_properties=False)
            _ = src * tel
            _ASM_HOST = Atmosphere(
                telescope=tel,
                r0=0.15,
                L0=30.0,
                windSpeed=[10.0],
                windDirection=[0.0],
                fractionalR0=[1.0],
                altitude=[0.0],
                elevation=90.0,
                angular_spectrum_propagation=True,
            )
    return _ASM_HOST


def propagate_asm(
    input_field: np.ndarray,
    *,
    lam: float,
    dx: float,
    z: float,
) -> np.ndarray:
    """OOPAO ``Atmosphere.ASM`` angular-spectrum propagation (pure function).

    Parameters
    ----------
    input_field : np.ndarray
        Input complex field (N, N).
    lam : float
        Wavelength (m).
    dx : float
        Input/output pixel pitch (m); input and output pitch are equal here.
    z : float
        Propagation distance (m); negative propagates in reverse.
    """
    if not _OOPAO_AVAILABLE:
        raise RuntimeError("OOPAO is not available; use the numpy backend")
    return _asm_host().ASM(input_field, lam, dx, dx, z)


def _asm_numpy(
    input_field: np.ndarray,
    wavelength: float,
    input_pitch: float,
    output_pitch: float,
    distance: float,
) -> np.ndarray:
    """Pure-numpy faithful transcription of OOPAO ``Atmosphere.ASM``.

    Only used as the fallback engine when OOPAO is unavailable, guaranteeing
    identical numerical behaviour.
    """
    if distance == 0:
        return input_field
    N = input_field.shape[0]
    k = 2.0 * np.pi / wavelength
    grid_dtype = (
        np.float64 if input_field.dtype in (np.complex128, np.float64) else np.float32
    )
    delta_f = 1.0 / (N * input_pitch)
    vals = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * delta_f
    fx, fy = np.meshgrid(vals, vals, copy=False)
    f_sq = fx**2 + fy**2
    vals_r = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * input_pitch
    x, y = np.meshgrid(vals_r, vals_r, copy=False)
    r_sq = x**2 + y**2
    m = output_pitch / input_pitch
    if m != 1.0:
        phase_1 = np.exp(1j * k / 2.0 * (1 - m) / distance * r_sq)
    else:
        phase_1 = 1.0
    phase_2 = np.exp(-1j * np.pi * wavelength * distance / m * f_sq)
    if m != 1.0:
        vals_out = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * output_pitch
        x_out, y_out = np.meshgrid(vals_out, vals_out, copy=False)
        r_out_sq = x_out**2 + y_out**2
        phase_3 = np.exp(1j * k / 2.0 * (m - 1) / (m * distance) * r_out_sq)
    else:
        phase_3 = 1.0
    field_freq = np.fft.fft2(np.fft.ifftshift(input_field * phase_1))
    field_filtered = np.fft.ifftshift(np.fft.fftshift(field_freq) * phase_2)
    field_out = np.fft.fftshift(np.fft.ifft2(field_filtered))
    return field_out * phase_3 / m


class Propagator:
    """Angular-spectrum field propagator built on OOPAO ``Atmosphere.ASM``.

    Parameters
    ----------
    N : int
        Grid size (square, N x N).
    dx : float
        Grid sample spacing (m).
    lam : float
        Wavelength (m).
    n_threads : int, optional
        Compatibility parameter (legacy FFTW thread count); unused by the
        OOPAO/numpy engines.
    engine : str, optional
        ``"oopao"`` (default): OOPAO ``Atmosphere.ASM`` kernel; falls back to
        ``"numpy"`` automatically when OOPAO is unavailable. ``"numpy"``: pure
        numpy transcription (bit-identical).
    dtype : np.dtype, optional
        Working complex dtype (default ``np.complex64``).
    """

    def __init__(
        self,
        N: int,
        dx: float,
        lam: float,
        n_threads: int = 1,
        engine: str = "oopao",
        dtype: np.dtype = np.dtype(np.complex64),
    ) -> None:
        self.N = N
        self.dx = dx
        self.lam = lam
        self.dtype = dtype

        if engine not in ("oopao", "numpy"):
            raise ValueError(f"unknown propagation engine: {engine!r}")
        if engine == "oopao" and not _OOPAO_AVAILABLE:
            engine = "numpy"
        self.engine = engine

    def _propagate_raw(self, E: np.ndarray, z: float) -> np.ndarray:
        if z == 0.0:
            return np.array(E, dtype=self.dtype, copy=True)
        if self.engine == "oopao":
            field = _asm_host().ASM(E, self.lam, self.dx, self.dx, z)
        else:
            field = _asm_numpy(E, self.lam, self.dx, self.dx, z)
        return np.asarray(field, dtype=self.dtype)

    def propagate(self, E: np.ndarray, z: float) -> np.ndarray:
        """Propagate a complex field by distance ``z`` (m)."""
        out = self._propagate_raw(E, z)
        return np.array(out, dtype=self.dtype, copy=True)

    def split_step(
        self,
        E_in: np.ndarray,
        screens: list[np.ndarray] | np.ndarray,
        dz: float,
    ) -> np.ndarray:
        """Symmetric split-step propagation through phase screens."""
        E = np.array(E_in, dtype=self.dtype, copy=True)
        for phi in screens:
            E = self.propagate(E, dz / 2.0)
            E = np.array(E, dtype=self.dtype, copy=True)
            E *= np.exp(1j * phi).astype(self.dtype)
            E = self.propagate(E, dz / 2.0)
        return E

    def angular_spectrum_intensity(self, E_in: np.ndarray, z: float) -> np.ndarray:
        """Return the propagated intensity ``|propagate(E_in, z)|**2``."""
        E = self.propagate(E_in, z)
        return (np.abs(E) ** 2).astype(np.float32)
