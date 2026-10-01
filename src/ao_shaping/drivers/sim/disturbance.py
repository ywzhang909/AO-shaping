"""Wavefront disturbance for the 2f-Fourier SLM simulation.

The ``slm-pib`` simulation originally propagated a *perfectly clean* wavefront:
:meth:`ao_shaping.drivers.sim.slm_pib_sim.SimPibSystem.far_field` used only the
SLM command phase, so the closed loop was never exposed to any wavefront
disturbance. This module supplies the missing disturbance -- **atmospheric
turbulence** plus a **thermal halo (热晕)** -- and exposes the static/dynamic
regimes the report needs to compare.

Two disturbance regimes
-----------------------
``static``
    One screen is generated on first use and then reused forever. This is the
    frozen-screen case, the analogue of the ``closed`` turbulence mode used by
    ``scripts/generate_oopao_impact_report.py``.
``dynamic``
    A fresh, independent screen is drawn on **every** optical evaluation -- the
    ``open``/sliding analogue. This models the *fully-decorrelated*
    ("white in time") limit. That limit is the correct asymptotic here: real
    atmospheric decorrelation is ~10-50 ms while one SPGD evaluation on this
    loop takes ~0.375 s, so consecutive evaluations are in fact uncorrelated.
    This is explicitly **not** a wind/advection model.

Both regimes are deterministic for a fixed ``seed``.

Amplitudes are reported *as measured*, never analytically
---------------------------------------------------------
``cn2`` and ``distance_m`` are **degenerate generator knobs**: the canonical
generator derives ``r0 = (0.423 * k**2 * cn2 * distance) ** (-3/5)``, which
depends only on their *product*. A single thin screen carries no propagation
physics, so there is no real kilometre-scale atmospheric path being modelled
here -- the bench is a ~0.3 m laboratory 2f-Fourier rig.

In addition, the repository's default (numpy) screen generator synthesises the
screen by FFT of a filtered white-noise field with **no subharmonic/low-frequency
compensation** (see ``src/ao_shaping/drivers/sim/AGENTS.md``, note 4). Because
``l_max`` (tens of metres) greatly exceeds the 15.36 mm aperture, most of the
von-Karman variance sits below the fundamental FFT frequency, so the measured
``sigma`` is a **lower bound** on the analytic variance for the same ``r0``.
Every amplitude this module reports is therefore measured from the produced
arrays via :meth:`SimDisturbance.stats`.

Raw-radian contract
-------------------
:meth:`SimDisturbance.phase` returns **raw, unwrapped radians** and never
applies ``mod 2*pi`` -- matching the repository rule that the only wrap point is
the SLM driver's radian -> grayscale conversion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.sim.beam_backend import make_beam_config, turbulence_phase
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

#: Valid :attr:`DisturbanceConfig.mode` values.
DISTURBANCE_MODES: tuple[str, ...] = ("none", "static", "dynamic")

#: Default Gaussian beam waist, in SLM pixels. Mirrors ``slm_pib_sim.BEAM_W0``
#: but is duplicated here on purpose: importing the optical-system module from
#: this one would create a cycle (``slm_pib_sim`` imports *this* module).
DEFAULT_BEAM_W0: float = 400.0

#: Amplitude below which the Gaussian pupil counts as un-illuminated when
#: normalising the thermal halo's peak-to-valley, so the normalisation is not
#: dominated by the numerically-noisy far wings.
_AMPLITUDE_FLOOR: float = 1e-3

#: Width of the smoothstep apodisation edge, as a fraction of the halo radius.
_APOD_EDGE_FRACTION: float = 0.35


@dataclass(frozen=True)
class DisturbanceConfig:
    """Configuration for :class:`SimDisturbance`.

    The panel geometry is deliberately *not* part of this config -- it is
    supplied at construction time so one config can be reused across grids.

    Attributes:
        mode: ``"none"``, ``"static"`` or ``"dynamic"``.
        cn2: Refractive-index structure constant. Degenerate with
            :attr:`distance_m`: only their product sets ``r0``.
        distance_m: Generator path-length knob, not a real propagation path on
            this bench (see the module docstring).
        l_max: Outer scale [m].
        l_min: Inner scale [m].
        wavelength_m: Wavelength [m].
        pixel_pitch_m: SLM pixel pitch [m]; sets the screen's physical extent so
            its pixels stay isotropic with the panel.
        thermal_halo_pv_waves: Peak-to-valley of the thermal-halo phase, in
            waves. ``0`` disables the halo.
        thermal_halo_radius_px: Radius over which the halo phase is retained, in
            SLM pixels. Values far beyond ``~2 * beam_w0`` are attenuated to
            invisibility by the Gaussian pupil amplitude (0.325 at 600 px and
            0.135 at 800 px for the default 400 px waist).
        halo_noll: ``(noll_index, coefficient)`` pairs fed to the canonical
            :func:`~ao_shaping.utils.wavefront.zernike_utils.generate_zernike_phase`.
            Negative coefficients give the negative thermal lens.
        halo_n_max: Maximum Zernike radial order for the halo basis.
        seed: Master seed; fully determines both regimes.
    """

    mode: str = "none"
    cn2: float = 2e-13
    distance_m: float = 500.0
    l_max: float = 30.0
    l_min: float = 2e-3
    wavelength_m: float = 1064e-9
    pixel_pitch_m: float = 8e-6
    thermal_halo_pv_waves: float = 0.30
    thermal_halo_radius_px: float = 600.0
    halo_noll: tuple[tuple[int, float], ...] = ((4, -1.0), (11, -1.0))
    halo_n_max: int = 4
    seed: int = 20261001

    def __post_init__(self) -> None:
        if self.mode not in DISTURBANCE_MODES:
            raise ValueError(
                f"unknown disturbance mode {self.mode!r}; expected one of {DISTURBANCE_MODES}"
            )
        if self.cn2 < 0.0:
            raise ValueError(f"cn2 must be >= 0, got {self.cn2}")
        if self.distance_m <= 0.0:
            raise ValueError(f"distance_m must be > 0, got {self.distance_m}")
        if self.pixel_pitch_m <= 0.0:
            raise ValueError(f"pixel_pitch_m must be > 0, got {self.pixel_pitch_m}")
        if self.l_min <= 0.0 or self.l_max <= 0.0:
            raise ValueError("l_min and l_max must be > 0")
        if self.thermal_halo_pv_waves < 0.0:
            raise ValueError(
                f"thermal_halo_pv_waves must be >= 0, got {self.thermal_halo_pv_waves}"
            )
        if self.thermal_halo_pv_waves > 0.0 and not self.halo_noll:
            raise ValueError("halo_noll must be non-empty when the thermal halo is enabled")


class SimDisturbance:
    """Turbulence + thermal-halo phase disturbance on an ``(h, w)`` panel grid.

    Memory is bounded on purpose. ``dynamic`` mode generates one full-resolution
    screen per optical evaluation and a 300-epoch run performs ~630 evaluations;
    retaining them (at ~18 MB each for a 1200x1920 float64 array) would cost
    several gigabytes. Only scalars are accumulated, and the frozen screen is
    the sole array kept alive -- in ``dynamic`` mode the screen is returned to
    the caller and never retained here.

    Args:
        config: The disturbance configuration.
        shape: Panel shape ``(h, w)`` -- e.g. the SLM's ``(1200, 1920)``.
        beam_w0: Gaussian beam waist in pixels, used only to mask the halo's
            peak-to-valley normalisation. Defaults to :data:`DEFAULT_BEAM_W0`.

    Raises:
        ValueError: On an unsupported configuration or an ``h > w`` panel.
    """

    def __init__(
        self,
        config: DisturbanceConfig,
        shape: tuple[int, int],
        *,
        beam_w0: float | None = None,
    ) -> None:
        self.config = config
        self.shape: tuple[int, int] = (int(shape[0]), int(shape[1]))
        self.beam_w0 = float(DEFAULT_BEAM_W0 if beam_w0 is None else beam_w0)
        if self.shape[0] > self.shape[1]:
            raise ValueError(
                f"panel shape {self.shape} is taller than it is wide; the turbulence "
                "generator is square and the screen is cropped from the wide axis, so "
                "h <= w is required"
            )

        self._halo = self._build_halo()
        self._mask = self._amplitude_mask()
        self._static_total: np.ndarray | None = None
        self._streak_count = 0
        self._turb_sigma_sum = 0.0
        self._last_total_std = 0.0
        self._calls = 0
        self._call_rms: list[float] = []
        self._call_streak_index: list[int] = []

        logger.debug(
            "SimDisturbance(mode={}, cn2={:g}, distance={:g}m, halo_pv={:g}waves, shape={})",
            config.mode,
            config.cn2,
            config.distance_m,
            config.thermal_halo_pv_waves,
            self.shape,
        )

    # --- construction helpers -------------------------------------------

    def _amplitude_mask(self) -> np.ndarray:
        """Boolean mask of where the Gaussian pupil is meaningfully illuminated."""
        h, w = self.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2
        return np.exp(-r2 / (2.0 * self.beam_w0**2)) > _AMPLITUDE_FLOOR

    def _build_halo(self) -> np.ndarray:
        """Deterministic thermal-halo phase, PV-normalised to the requested waves.

        The halo is a *steady-state* aberration (the negative thermal lens left
        in the beam path), so it is built once and added to every turbulence
        streak instead of being re-randomised.

        Construction follows the repository rule that all Zernike maths goes
        through the canonical API entry: the unit-coefficient phase comes from
        :func:`generate_zernike_phase` (linear in the coefficients), its
        peak-to-valley is measured inside the illuminated pupil, and the array is
        scaled linearly so PV equals ``thermal_halo_pv_waves * 2*pi`` radians.
        """
        h, w = self.shape
        if self.config.thermal_halo_pv_waves <= 0.0:
            return np.zeros(self.shape, dtype=np.float64)

        coefficients = {int(noll): float(amp) for noll, amp in self.config.halo_noll}
        unit = np.nan_to_num(
            np.asarray(
                generate_zernike_phase(
                    coefficients,
                    resolution=(w, h),
                    n_max=self.config.halo_n_max,
                ),
                dtype=np.float64,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.sqrt((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2)
        edge = max(1.0, self.config.thermal_halo_radius_px * _APOD_EDGE_FRACTION)
        t = np.clip(
            (self.config.thermal_halo_radius_px + edge - radius) / (2.0 * edge), 0.0, 1.0
        )
        unit = unit * (t * t * (3.0 - 2.0 * t))  # smoothstep apodisation

        mask = self._amplitude_mask()
        pv = float(np.ptp(unit[mask])) if np.any(mask) else 0.0
        if pv <= 0.0:
            logger.warning(
                "thermal halo has zero peak-to-valley after apodisation; using zeros"
            )
            return np.zeros(self.shape, dtype=np.float64)

        return unit * (self.config.thermal_halo_pv_waves * 2.0 * np.pi / pv)

    def _turbulence_streak(self, index: int) -> np.ndarray:
        """Generate the ``index``-th independent turbulence screen (uncached).

        Generated at the panel's native pixel pitch (``n_grid = w`` covering
        ``w * pixel_pitch``) and centre-cropped to ``h`` rows, so the screen has
        isotropic pixels, no anamorphic stretch and no tiling seam. A coarser
        grid was measured to lose ~41% of the phase amplitude.
        """
        h, w = self.shape
        if self.config.cn2 <= 0.0:
            return np.zeros(self.shape, dtype=np.float64)

        cfg = make_beam_config(
            n_grid=w,
            aperture_size=w * self.config.pixel_pitch_m,
            wavelength=self.config.wavelength_m,
            cn2=self.config.cn2,
            l_max=self.config.l_max,
            l_min=self.config.l_min,
            propagation_distance=self.config.distance_m,
        )
        full = np.asarray(
            turbulence_phase(
                cfg,
                cn2=self.config.cn2,
                l_max=self.config.l_max,
                l_min=self.config.l_min,
                propagation_distance=self.config.distance_m,
                rng=np.random.default_rng([self.config.seed, index]),
            ),
            dtype=np.float64,
        )
        start = (full.shape[0] - h) // 2
        return np.ascontiguousarray(full[start : start + h, :w])

    # --- public API ------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether a disturbance is being applied at all."""
        return self.config.mode != "none"

    @property
    def halo_phase(self) -> np.ndarray:
        """The deterministic thermal-halo phase array (raw radians)."""
        return self._halo

    def phase(self) -> np.ndarray:
        """Return the disturbance phase for the current optical evaluation.

        Returns:
            ``(h, w)`` float64 array of **raw, unwrapped radians**. ``"none"``
            returns zeros; ``"static"`` always returns the same frozen screen;
            ``"dynamic"`` returns a fresh, independent screen on every call.
        """
        self._calls += 1

        if self.config.mode == "none":
            self._call_rms.append(0.0)
            self._call_streak_index.append(0)
            return np.zeros(self.shape, dtype=np.float64)

        if self.config.mode == "static":
            if self._static_total is None:
                turbulence = self._turbulence_streak(0)
                self._turb_sigma_sum = float(turbulence[self._mask].std())
                self._static_total = turbulence + self._halo
                self._streak_count = 1
            self._call_rms.append(float(self._static_total.std()))
            self._call_streak_index.append(0)
            return self._static_total

        index = self._streak_count
        turbulence = self._turbulence_streak(index)
        total = turbulence + self._halo
        self._turb_sigma_sum += float(turbulence[self._mask].std())
        self._streak_count += 1
        self._last_total_std = float(total.std())
        self._call_rms.append(self._last_total_std)
        self._call_streak_index.append(index)
        return total

    def stats(self) -> dict[str, float]:
        """Measured disturbance amplitudes (never analytic values).

        Returns:
            Scalars only: ``sigma_turb_rad``, ``sigma_halo_rad``,
            ``sigma_total_rad``, ``streaks_used``, ``calls``, ``beam_w0``,
            ``panel_pixels``.
        """
        halo_std = float(self._halo[self._mask].std()) if np.any(self._mask) else 0.0

        if self.config.cn2 <= 0.0:
            turb_std = 0.0
            total_std = float(self._halo.std())
        elif self.config.mode == "dynamic":
            if self._streak_count:
                turb_std = self._turb_sigma_sum / self._streak_count
                total_std = (
                    float(np.mean(self._call_rms)) if self._call_rms else self._last_total_std
                )
            else:
                probe = self._turbulence_streak(0)
                turb_std = float(probe[self._mask].std())
                total_std = float((probe + self._halo).std())
        else:
            if self._static_total is None:
                turbulence = self._turbulence_streak(0)
                self._turb_sigma_sum = float(turbulence[self._mask].std())
                self._static_total = turbulence + self._halo
            turb_std = self._turb_sigma_sum
            total_std = float(self._static_total.std())

        return {
            "sigma_turb_rad": turb_std,
            "sigma_halo_rad": halo_std,
            "sigma_total_rad": total_std,
            "streaks_used": float(self._streak_count),
            "calls": float(self._calls),
            "beam_w0": float(self.beam_w0),
            "panel_pixels": float(self.shape[0] * self.shape[1]),
        }

    def trace(self) -> dict[str, list]:
        """Per-evaluation history: ``call_rms`` and ``call_streak_index``."""
        return {
            "call_rms": list(self._call_rms),
            "call_streak_index": list(self._call_streak_index),
        }

    def archive(self, *, factor: int = 8, max_count: int = 12) -> dict[str, np.ndarray]:
        """Decimated thumbnails of the distinct screens actually used.

        Screens are regenerated on demand from the same per-streak seed rather
        than retained, so archiving costs no memory during the run. For
        ``dynamic`` at most ``max_count`` evenly spaced streak indices are
        archived -- every ``call_rms`` entry is still recorded in full by
        :meth:`trace`, which is the actual evidence that the screen varied.

        Args:
            factor: Spatial decimation factor; the thumbnail is
                ``(h // factor, w // factor)``.
            max_count: Maximum number of distinct screens to archive.

        Returns:
            ``{"screens": (n, h//factor, w//factor) float32,
            "streak_indices": (n,) int32}``.
        """
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")

        h, w = self.shape
        thumb_shape = (h // factor, w // factor)

        if self.config.mode == "none":
            return {
                "screens": np.zeros((0, *thumb_shape), dtype=np.float32),
                "streak_indices": np.zeros((0,), dtype=np.int32),
            }

        if self.config.mode == "static":
            total = self._static_total
            if total is None:
                total = self._turbulence_streak(0) + self._halo
            return {
                "screens": np.asarray(total[::factor, ::factor], dtype=np.float32)[None, ...],
                "streak_indices": np.zeros((1,), dtype=np.int32),
            }

        count = self._streak_count
        if count == 0:
            return {
                "screens": np.zeros((0, *thumb_shape), dtype=np.float32),
                "streak_indices": np.zeros((0,), dtype=np.int32),
            }
        if count <= max_count:
            indices = list(range(count))
        else:
            indices = sorted(
                {int(i) for i in np.linspace(0, count - 1, max_count).round()}
            )
        screens = [
            np.asarray(
                (self._turbulence_streak(i) + self._halo)[::factor, ::factor],
                dtype=np.float32,
            )
            for i in indices
        ]
        return {
            "screens": np.stack(screens, axis=0),
            "streak_indices": np.asarray(indices, dtype=np.int32),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable config + measured stats (no arrays)."""
        config = asdict(self.config)
        config["halo_noll"] = [list(pair) for pair in self.config.halo_noll]
        return {
            "mode": self.config.mode,
            "enabled": self.enabled,
            "beam_w0": float(self.beam_w0),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "config": config,
            "measured": self.stats(),
        }
