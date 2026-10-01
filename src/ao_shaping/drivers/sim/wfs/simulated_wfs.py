"""Simulated Shack-Hartmann wavefront sensor built on OOPAO.

Why a slope model and not a focal-plane one
--------------------------------------------
A Shack-Hartmann sensor measures the **pupil-plane phase gradient**: a
microlens array images the pupil onto a detector and the spot displacement
encodes each subaperture's local tip/tilt. Driving ``wfs_measure(..., phase_in=)``
therefore measures exactly the quantity an AO loop needs.

Reusing the repository's existing focal-plane propagation instead (as an earlier
plan proposed) would have produced a *camera* image, not slopes — the two are not
interchangeable, and the optimizers consume slopes.

Fidelity
--------
The slopes come from OOPAO's lenslet FFT rather than a stub formula, and the
measurement obeys the invariants an AO loop depends on: **deterministic**
(repeated reads are bit-identical), **sign-symmetric**, and **exactly zero** on a
flat pupil.

The *Zernike coefficients* are far less faithful than those invariants, and
reports must not present them as ground truth. Measured, injecting known modes in
radians and reading ``get_zernike()`` back:

    isolated single modes   noll 2 tilt  24%     noll 4 defocus  1.4%     noll 11 spherical 32%
    four modes together     noll 2 tilt  83%     noll 4 defocus  0.9%     noll 7 coma 17%   noll 11 spherical 25%

Tilt alone recovers to 24% but only to 83% when co-injected with other modes, so
the dominant error is **inter-modal cross-talk**, not per-mode noise. That is a
genuine property of this sensor as modelled: the lenslet image is an intensity
(``|FFT|**2``), which is quadratic in phase, so cross-terms between modes cannot
be linearised away. Refining the sampling does not fix it -- raising
``n_pixel_per_subaperture`` from 8 to 32 and ``n_subap`` from 6 to 24 left
combined-mode error at 80-140%, and conditioning stayed benign (cond 2-6), which
rules out a rank deficiency.

Consequently this class is suitable for exercising control loops, checking
sign conventions and validating the micrometre/waves plumbing. It is **not** a
conforming wavefront reference: any report of RMS improvement or Strehl derived
from ``get_zernike()`` must be labelled as unvalidated. For trustworthy numbers,
calibrate against hardware the way ``zernike-matrix`` does.

Not modelled: microlens aberration, detector noise, and spot crosstalk.

Units (the historically dangerous part)
---------------------------------------
The WFS family returns Zernike coefficients in **micrometres** while corrections
are applied in **waves**. Mixing the two produced two real bugs (coefficients
inflated 1.88x; applied phase shrunk 2*pi = 6.28x). This sensor therefore keeps
radians internally and converts only at the boundary:

* ``get_wavefront()`` -> waves      = ``phase_rad / (2*pi)``
* ``get_zernike()``   -> micrometres = ``phase_rad * lambda_um / (2*pi)``

``um_to_waves()`` is hard-coded for 532 nm, so the default wavelength must be
532 nm for ``um_to_waves()`` then ``* 2*pi`` to recover the radians exactly. A
different wavelength would silently rescale the whole correction chain.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

from ao_shaping.drivers.sim import _oopao_compat as oopao
from ao_shaping.drivers.sim.dm_optics import SimDmOptics
from ao_shaping.drivers.sim.disturbance import DisturbanceConfig, SimDisturbance
from ao_shaping.drivers.wfs._registry import register_wfs
from ao_shaping.drivers.wfs.base import BaseWFS
from ao_shaping.utils.wavefront.zernike_utils import (
    generate_zernike_phase,
    list_zernike_modes,
)

_MATRIX_CACHE: dict[tuple, np.ndarray] = {}

#: Most recently constructed sensor. ``SimulateDM`` publishes voltages here so a
#: DM-driven loop actually moves the pupil the sensor measures. Process-wide for
#: the same reason ``slm_pib_sim.get_system()`` is: the DM and the sensor are
#: built by separate factories and never share a reference.
_ACTIVE_SENSOR: SimulatedWFS | None = None


def get_active_sim_wfs() -> SimulatedWFS | None:
    """Return the most recently constructed simulated sensor, if any."""
    return _ACTIVE_SENSOR


#: A Shack-Hartmann sensor is blind to piston: a subaperture's absolute phase
#: offset does not move its spot. ``list_zernike_modes`` starts at Noll 1
#: (piston), and fitting it against slopes leaves a near-null column that
#: ``pinv`` amplifies into a large spurious coefficient (measured: +0.50 rad for
#: a defocus pupil). Piston is therefore excluded from the fit basis.
_FIT_FIRST_MODE = 1


def _to_numpy(array: Any) -> np.ndarray:
    """CuPy -> NumPy. OOPAO's own ``to_numpy`` is broken on the installed CuPy."""
    try:
        import cupy as cp

        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except ImportError:
        pass
    return np.asarray(array)


@register_wfs("sim")
class SimulatedWFS(BaseWFS):
    """A simulated Shack-Hartmann WFS sharing the real driver's contract."""

    manufacturer = "Simulated"
    model = "SimulatedWFS"

    def __init__(
        self,
        resolution: int = 48,
        diameter: float = 0.30,
        n_subap: int = 6,
        n_pixel_per_subaperture: int = 8,
        wavelength_nm: int = 532,
        zernike_order: int = 4,
        disturbance_cn2: float = 0.0,
        disturbance_config: DisturbanceConfig | None = None,
        disturbance_seed: int = 20261001,
    ) -> None:
        super().__init__()
        if resolution % (n_subap * n_pixel_per_subaperture) != 0:
            raise ValueError(
                f"resolution {resolution} must be divisible by "
                f"n_subap * n_pixel_per_subaperture "
                f"({n_subap} * {n_pixel_per_subaperture})"
            )
        self.grid = int(resolution)
        self.diameter = float(diameter)
        self.n_subap = int(n_subap)
        self.n_pixel_per_subaperture = int(n_pixel_per_subaperture)
        self.wavelength_nm = int(wavelength_nm)
        self.zernike_order = int(zernike_order)

        # Thorlabs-facing surface. The optimizers and runners read these directly
        # (``wfs.num_spots_x``, ``wfs.mla_index``, ``wfs.serial_num`` ...), so a
        # simulated sensor has to answer them even though it has no MLA, no
        # serial number and no vendor DLL.
        self.num_spots_x = self.n_subap
        self.num_spots_y = self.n_subap
        self.mla_index = None
        self.serial_num = "SIM-WFS-0001"
        self.device_name = "Simulated Shack-Hartmann WFS"
        self.exposure_time = 0.0
        self.high_speed = True
        self.use_custom_ref = False
        self.pupil = np.ones((self.grid, self.grid), dtype=np.float64)
        # Subaperture pitch in pixels; the SH grid divides the pupil evenly.
        self.d_x = self.diameter / self.n_subap if self.n_subap else self.diameter

        self._phase_rad = np.zeros((self.grid, self.grid), dtype=float)
        # Sized to the sensor grid, not the SLM panel: `SimPibSystem` owns a
        # separate instance for the far field, and reusing that one would need
        # resampling 1200x1920 down to this grid on every measurement.
        self.dm_optics = SimDmOptics(slm_shape=(self.grid, self.grid))
        self._slopes: np.ndarray | None = None
        self._flux: np.ndarray | None = None
        self._raw: np.ndarray | None = None

        # OOPAO prints banner tables on construction; keep the library silent.
        with contextlib.redirect_stdout(io.StringIO()):
            self._tel = oopao.Telescope(
                resolution=self.grid,
                diameter=self.diameter,
                fov=0.0,
                samplingTime=0.001,
            )
            self._src = oopao.Source(
                optBand="R", magnitude=0.0, display_properties=False
            )
            self._src = self._src * self._tel
            self._wfs = oopao.ShackHartmann(
                nSubap=self.n_subap,
                telescope=self._tel,
                lightRatio=0.5,
                n_pixel_per_subaperture=self.n_pixel_per_subaperture,
            )
            self._wfs.relay(self._src)
        key = next(iter(self._wfs.sh_data))
        self._sh_data = self._wfs.sh_data[key]
        self._open = False
        self._set_state_ok()
        global _ACTIVE_SENSOR
        _ACTIVE_SENSOR = self
        # Optional aberration to correct. Without one the DM can only add
        # phase, so an RMS-minimising loop correctly keeps the flat command
        # and there is nothing to demonstrate.
        if disturbance_config is not None:
            self.disturbance = SimDisturbance(
                disturbance_config, (self.grid, self.grid)
            )
        elif disturbance_cn2 > 0.0:
            self.disturbance = SimDisturbance(
                DisturbanceConfig(mode="static", cn2=disturbance_cn2,
                                  seed=disturbance_seed),
                (self.grid, self.grid),
            )
        else:
            self.disturbance = None

    def _set_state_ok(self) -> None:
        from ao_shaping.drivers.device_base import DeviceState

        self._set_state(DeviceState.READY)

    # ---- configuration -------------------------------------------------

    def _n_modes(self, order: int) -> int:
        return len(list_zernike_modes(order))

    def _matrix_key(self, order: int) -> tuple:
        return (
            self.grid,
            self.diameter,
            self.n_subap,
            self.n_pixel_per_subaperture,
            self.wavelength_nm,
            int(order),
        )

    def _reconstructor(self, order: int) -> np.ndarray:
        """Pseudo-inverse mapping measured slopes -> Zernike radians.

        Calibrated numerically by pushing each Zernike mode through the sensor,
        which is the same procedure the ``zernike-matrix`` command performs on
        hardware. It is self-consistent by construction: no assumed sensitivity,
        no external calibration file, and it cannot drift from the forward model.
        """
        key = self._matrix_key(order)
        cached = _MATRIX_CACHE.get(key)
        if cached is not None:
            return cached

        n_modes = self._n_modes(order)
        columns = []
        with contextlib.redirect_stdout(io.StringIO()):
            for j in range(_FIT_FIRST_MODE, n_modes):
                phase = self._mode_phase(j, order)
                slopes = _to_numpy(
                    self._wfs.wfs_measure(
                        self._src, self._sh_data, phase_in=phase
                    )[0]
                )
                columns.append(slopes.ravel())
        matrix = np.stack(columns, axis=1)
        pseudo = np.linalg.pinv(matrix)
        _MATRIX_CACHE[key] = pseudo
        return pseudo

    def _mode_phase(self, index: int, order: int) -> np.ndarray:
        """Radian phase for one Noll mode, aperture-exterior filled with zero."""
        modes = list_zernike_modes(order)
        noll, _n, _m, _name = modes[index]
        phase = generate_zernike_phase(
            {noll: 1.0},
            resolution=(self.grid, self.grid),
            n_max=order,
            radius=self.grid / 2.4,
        )
        # The canonical generator returns NaN outside the aperture; the pupil
        # mask already excludes that region, so flatten it for the FFT.
        return np.nan_to_num(np.asarray(phase, dtype=float), nan=0.0)

    # ---- state ---------------------------------------------------------

    def set_pupil_phase(self, phase_rad: np.ndarray) -> None:
        """Inject the pupil-plane phase (radians) the sensor should measure."""
        arr = np.asarray(phase_rad, dtype=float)
        if arr.shape != (self.grid, self.grid):
            raise ValueError(
                f"pupil phase shape {arr.shape} != {(self.grid, self.grid)}"
            )
        self._phase_rad = np.nan_to_num(arr, nan=0.0)
        self._slopes = None

    def _pupil_phase(self) -> np.ndarray:
        """DM phase plus any explicitly injected phase, summed at evaluation.

        Following the ``SimDisturbance`` precedent, contributors are summed here
        rather than baked into ``_phase_rad``, so neither can be double-counted
        and the stored command stays exactly what the caller set.
        """
        total = self._phase_rad
        dm_phase = self.dm_optics.phase()
        if dm_phase.any():
            total = total + dm_phase
        if self.disturbance is not None:
            total = total + self.disturbance.phase()
        return total

    def take_slopes(self) -> tuple[np.ndarray, np.ndarray]:
        """Measure and return ``(dx, dy)`` slopes in the sensor's native units.

        OOPAO returns a row-blocked array: rows ``0:n_subap`` carry x-slopes and
        rows ``n_subap:`` carry y-slopes (verified with pure tilt probes).
        """
        if self._slopes is None:
            with contextlib.redirect_stdout(io.StringIO()):
                measured = self._wfs.wfs_measure(
                    self._src, self._sh_data, phase_in=self._pupil_phase()
                )
            self._slopes = _to_numpy(measured[0])
            self._flux = _to_numpy(measured[1])
            self._raw = _to_numpy(measured[2])
        n = self.n_subap
        return self._slopes[:n], self._slopes[n : 2 * n]

    def _zernike_radians(self, order: int) -> np.ndarray:
        dx, dy = self.take_slopes()
        slopes = np.concatenate([dx.ravel(), dy.ravel()])
        return self._reconstructor(order) @ slopes

    # ---- BaseWFS contract ----------------------------------------------

    def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> None:
        """Refresh the measurement. The simulated sensor is noise-free."""
        self._slopes = None
        self.take_slopes()

    def get_spots_statics(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Spot intensities and centroids from the lenslet-plane image."""
        self.take_slopes()
        assert self._raw is not None and self._flux is not None
        raw = np.asarray(self._raw, dtype=float)
        flux = np.asarray(self._flux, dtype=float).ravel()
        flat = raw.reshape(raw.shape[0], -1) if raw.ndim > 2 else raw
        with np.errstate(invalid="ignore", divide="ignore"):
            weight = np.where(flat > 0, flat, 0.0)
            cols = np.arange(weight.shape[1], dtype=float)
            cx = np.nansum(weight * cols, axis=1) / np.maximum(
                np.nansum(weight, axis=1), 1e-12
            )
        rows = np.arange(weight.shape[0], dtype=float)[:, None]
        cy = rows * np.ones_like(cx)[None, :]
        return flux, (cx, cy)

    def build_subaperture_mask(
        self,
        n_avg: int = 30,
        threshold_ratio: float = 0.3,
        edge_clip: int = 1,
        plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validity mask over the ``num_spots_x`` x ``num_spots_y`` subaperture grid.

        Returns ``(mask_bool, valid_indices_flat)``: the 2-D boolean mask and the
        flat indices of valid entries. Callers filter the slope vector with
        ``np.concatenate([mask.ravel(), mask.ravel()])`` against
        ``2 * num_spots_x * num_spots_y`` rows, so the mask must be sized by the
        subaperture grid and not by the flux array.

        It cannot be sized by flux: OOPAO reports flux on the 8x8 lenslet grid
        (64 entries at ``n_subap=6``) while the slopes span a 6x6 subaperture
        grid (36, doubled to 72 slopes). The two are different geometries.

        The model has no per-subaperture vignetting to detect, so every
        subaperture is valid apart from the requested edge clip.
        """
        nx, ny = self.num_spots_x, self.num_spots_y
        mask = np.ones((nx, ny), dtype=bool)
        if edge_clip and nx > 2 * edge_clip and ny > 2 * edge_clip:
            mask[:edge_clip, :] = False
            mask[-edge_clip:, :] = False
            mask[:, :edge_clip] = False
            mask[:, -edge_clip:] = False
        return mask, np.flatnonzero(mask.ravel())

    def get_spot_deviation(
        self, cancel_tile: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spot displacement from the reference, in the sensor's native units."""
        dx, dy = self.take_slopes()
        if cancel_tile and self.remove_tilt:
            dx = dx - float(dx.mean())
            dy = dy - float(dy.mean())
        return dx, dy

    def get_wavefront(self, cancel_tile: bool = False) -> tuple[np.ndarray, dict]:
        """Reconstructed wavefront in **waves**, plus summary statistics.

        The statistics keys mirror :meth:`ThorlabWFS.get_wavefront` exactly
        (``min``/``max``/``diff``/``mean``/``rms``/``wighted_rms``). The runner
        reads ``statics["wighted_rms"]`` to schedule its learning rate, so a
        narrower dict raises ``KeyError`` mid-optimization.
        """
        order = self.zernike_order
        fit_rad = self._zernike_radians(order)
        fitted = np.zeros((self.grid, self.grid), dtype=float)
        for k, j in enumerate(range(_FIT_FIRST_MODE, self._n_modes(order))):
            fitted += fit_rad[k] * self._mode_phase(j, order)
        wavefront = fitted / (2.0 * np.pi)
        # Piston is unmeasurable, so the residual is taken against the
        # piston-removed pupil; otherwise the DC offset dominates the RMS and
        # the number stops describing how well the modes were recovered.
        pupil = self._pupil_phase()
        residual = pupil - fitted - float(pupil.mean())
        stats = {
            "min": float(wavefront.min()),
            "max": float(wavefront.max()),
            "diff": float(wavefront.max() - wavefront.min()),
            "mean": float(wavefront.mean()),
            "rms": float(np.sqrt(np.mean(residual**2))),
            "wighted_rms": float(np.std(wavefront)),
        }
        return wavefront, stats

    def get_zernike(self, zernike_order: int = 10) -> np.ndarray:
        """Zernike coefficients in **micrometres**, Noll ordered.

        Micrometres are the historical WFS contract. Callers must convert with
        ``zernike_utils.um_to_waves()`` before doing anything in waves. Piston is
        reported as zero because the sensor cannot measure it.
        """
        fit_rad = self._zernike_radians(zernike_order)
        n_all = self._n_modes(zernike_order)
        z_rad = np.zeros(n_all, dtype=float)
        z_rad[_FIT_FIRST_MODE:] = fit_rad
        return z_rad * (self.wavelength_nm * 1e-3) / (2.0 * np.pi)

    # ---- Thorlabs-specific surface (no simulated counterpart) ----------

    def optimize_pupil(self) -> tuple[float, float, float, float]:
        """Return the configured pupil; there is nothing to optimise."""
        return (0.0, 0.0, float(self.diameter), float(self.diameter))

    def optimize_exposure_time_and_gain(self) -> tuple[float, float]:
        """The simulated sensor has neither exposure nor gain."""
        return (float(self.get_parameter_value("exposure_time_ms") or 0.0), 1.0)

    def save_user_ref(self, backup_dir: str | None = None) -> bool:
        """No user reference to persist; a no-op that reports success."""
        return True

    def load_user_ref(self, backup_path: str | None = None) -> bool:
        """No user reference to load; a no-op that reports success."""
        return True

    def create_default_user_ref(self) -> bool:
        return True

    def get_mla_name(self) -> str:
        """Name of the active microlens array.

        There is no MLA on a simulated sensor, but tooling prints this and
        writes it into report metadata, so it answers rather than raising.
        """
        return "SIM-MLA"

    def set_ref_plane(self, custom: bool) -> None:
        """Select the reference plane. No-op: the sim has no user .ref file."""
        self.use_custom_ref = bool(custom)

    # ---- Device lifecycle ----------------------------------------------

    def open(self) -> None:
        self._open = True
        self._set_state_ok()

    def close(self) -> None:
        self._open = False
        from ao_shaping.drivers.device_base import DeviceState

        self._set_state(DeviceState.DISCONNECTED)

    def is_connected(self) -> bool:
        return self._open

    def get_hardware_info(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "manufacturer": self.manufacturer,
            "grid": self.grid,
            "n_subap": self.n_subap,
            "wavelength_nm": self.wavelength_nm,
            "zernike_units": "um",
            "wavefront_units": "waves",
            "simulated": True,
        }


__all__ = ["SimulatedWFS"]
