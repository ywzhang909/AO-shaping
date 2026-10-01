"""DM voltage -> pupil phase coupling for the simulated optical bench.

Why this module exists
----------------------
``SimPibSystem.far_field()`` used to sum only the SLM command phase, so the DM
had no path into the optics at all. The ``pib`` and ``combined`` runners drive DM
*voltages*, so they ran to completion and wrote CSV files while the DM
influenced nothing -- the objective was pure noise. That is not "SPGD failed to
converge"; it is "the DM was absent from the model".

Influence model
---------------
Each actuator contributes a Gaussian phase bump. Real DM influence functions are
close to Gaussian and, being radially symmetric, are *separable* in x and y.
That separability is what makes this affordable: the phase is evaluated as one
matrix product over the DM's bounding box rather than as 64 full-panel images,
which would be ~1.2 GB at the real 1920x1200 panel size.

    phase(y, x) = sum_i  a_i * gy_i(y) * gx_i(x)

This is a fixed linear operator, hence linear in voltage -- a property the tests
pin, and the reason a reported "response" carries any meaning.

Physical device
---------------
This model stands in for the **NLight deformable mirror** (``drivers/dm/NLight.py``),
the bench's primary DM. Its actuator count and voltage limits are taken from that
driver's own class attributes rather than restated here, so the simulated device
cannot silently drift away from the hardware it mirrors.

One parameter has **no documented value in this repository**: the NLight DM's
optical-path stroke. ``stroke_um`` therefore stays explicit and defaults to a
placeholder that is reported as uncalibrated rather than passed off as a
manufacturer figure. Supply the datasheet (or measured) value for any run whose
absolute phase matters.

Units
-----
Voltages map to optical path difference in micrometres, then to phase in radians:

    opd_um    = stroke_um * v / v_max
    phase_rad = opd_um * 2*pi / (wavelength_nm * 1e-3)

Zero volts is flat, matching the real driver where 0 V sits mid-stroke. Because the
NLight range is asymmetric (``V_Min=-300``, ``V_Max=499``), the negative half
stroke is correspondingly shorter than the positive half.

Phase is returned as **raw unwrapped radians** and is never reduced mod 2*pi:
per the project-wide raw-only phase contract the single wrap point lives in the
SLM driver, and wrapping here would break the linearity the tests assert.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.NLight import NLight

#: Physical device this model mirrors. Actuator count and voltage limits are read
#: from it directly; only the un-documented stroke is a modelling parameter here.
MIRRORED_DEVICE = "NLight"

DEFAULT_WAVELENGTH_NM = 532.0

#: The NLight DM's optical-path stroke has **no documented value in this repo**.
#: This placeholder only sets coupling strength; ``__init__`` logs it as
#: uncalibrated so it is never mistaken for a datasheet figure.
UNCALIBRATED_STROKE_UM = 1.5

#: Fraction of the short panel axis spanned by the actuator array.
DEFAULT_SPAN_FRACTION = 0.6

#: Gaussian sigma as a fraction of the actuator pitch.
DEFAULT_SIGMA_FRACTION = 0.5


class SimDmOptics:
    """Maps the NLight DM's voltage vector onto a pupil phase map."""

    def __init__(
        self,
        n_actuators: int | None = None,
        slm_shape: tuple[int, int] = (1200, 1920),
        v_min: float | None = None,
        v_max: float | None = None,
        stroke_um: float = UNCALIBRATED_STROKE_UM,
        wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
        span_fraction: float = DEFAULT_SPAN_FRACTION,
        sigma_fraction: float = DEFAULT_SIGMA_FRACTION,
    ) -> None:
        # Default to the mirrored device's own declared specs so this model cannot
        # drift away from the hardware it stands in for.
        if n_actuators is None:
            n_actuators = NLight.DM_NUM
        if v_min is None:
            v_min = NLight.V_Min
        if v_max is None:
            v_max = NLight.V_Max
        if stroke_um == UNCALIBRATED_STROKE_UM:
            logger.warning(
                "{} DM stroke is undocumented in this repo; using placeholder "
                "stroke_um={} um. Treat absolute phase as uncalibrated.",
                MIRRORED_DEVICE,
                stroke_um,
            )
        side = int(round(np.sqrt(n_actuators)))
        if side * side != n_actuators:
            raise ValueError(
                f"n_actuators must be a perfect square for a grid layout, got {n_actuators}"
            )
        self.n_actuators = int(n_actuators)
        self.n_side = side
        self.slm_h, self.slm_w = int(slm_shape[0]), int(slm_shape[1])
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.stroke_um = float(stroke_um)
        self.wavelength_nm = float(wavelength_nm)

        span = DEFAULT_SPAN_FRACTION * min(self.slm_h, self.slm_w)
        self.pitch_px = span / (side - 1) if side > 1 else span
        self.sigma_px = max(self.pitch_px * sigma_fraction, 1.0)

        self.actuator_grid = self._build_grid(span)
        self._box = self._build_box()
        self._gy, self._gx = self._build_separable_basis()

        self._voltages = np.zeros(self.n_actuators, dtype=np.float64)
        self._opd = np.zeros(self.n_actuators, dtype=np.float64)
        self._phase = np.zeros((self.slm_h, self.slm_w), dtype=np.float64)
        self._dirty = True
        self._version = 0

    @property
    def version(self) -> int:
        """Monotonic counter bumped on every voltage change.

        ``SimPibSystem`` caches its far field, so without an explicit version the
        cache would keep serving the pre-DM image and the DM coupling would look
        like it had no effect.
        """
        return self._version

    @property
    def influence_radius_px(self) -> float:
        """Radius beyond which an actuator's bump has decayed to ~2%."""
        return 3.0 * self.sigma_px

    def _build_grid(self, span: float) -> np.ndarray:
        """Actuator centres in ``(row, col)`` pixels, row-major over the grid."""
        half = span / 2.0
        cy, cx = self.slm_h / 2.0, self.slm_w / 2.0
        rows = np.rint(np.linspace(cy - half, cy + half, self.n_side)).astype(int)
        cols = np.rint(np.linspace(cx - half, cx + half, self.n_side)).astype(int)
        rows = np.clip(rows, 0, self.slm_h - 1)
        cols = np.clip(cols, 0, self.slm_w - 1)
        yy, xx = np.meshgrid(rows, cols, indexing="ij")
        return np.column_stack((yy.ravel(), xx.ravel()))

    def _build_box(self) -> tuple[int, int, int, int]:
        pad = int(np.ceil(self.influence_radius_px))
        r0 = max(int(self.actuator_grid[:, 0].min()) - pad, 0)
        r1 = min(int(self.actuator_grid[:, 0].max()) + pad + 1, self.slm_h)
        c0 = max(int(self.actuator_grid[:, 1].min()) - pad, 0)
        c1 = min(int(self.actuator_grid[:, 1].max()) + pad + 1, self.slm_w)
        return r0, r1, c0, c1

    def _build_separable_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Factorised influence functions, shaped ``(n_actuators, box_len)``."""
        r0, r1, c0, c1 = self._box
        ys = np.arange(r0, r1, dtype=np.float64)[:, None]
        xs = np.arange(c0, c1, dtype=np.float64)[:, None]
        cy = self.actuator_grid[:, 0][None, :]
        cx = self.actuator_grid[:, 1][None, :]
        two_sigma_sq = 2.0 * self.sigma_px**2
        gy = np.exp(-((ys - cy) ** 2) / two_sigma_sq).T
        gx = np.exp(-((xs - cx) ** 2) / two_sigma_sq).T
        return gy, gx

    def set_voltages(self, voltages: np.ndarray) -> None:
        """Store the DM command, clipped to the DM's voltage range."""
        volts = np.asarray(voltages, dtype=np.float64).ravel()
        if volts.shape != (self.n_actuators,):
            raise ValueError(
                f"expected {self.n_actuators} voltages, got {volts.shape}"
            )
        self._voltages = np.clip(volts, self.v_min, self.v_max)
        self._opd = self.stroke_um * self._voltages / self.v_max
        self._dirty = True
        self._version += 1

    @property
    def voltages(self) -> np.ndarray:
        return self._voltages.copy()

    @property
    def opd_um(self) -> np.ndarray:
        """Per-actuator optical path difference in micrometres."""
        return self._opd.copy()

    def phase(self) -> np.ndarray:
        """Pupil phase in raw unwrapped radians, shaped like the SLM panel."""
        if self._dirty:
            r0, r1, c0, c1 = self._box
            block = (self._gy * self._opd[:, None]).T @ self._gx
            self._phase.fill(0.0)
            self._phase[r0:r1, c0:c1] = block * (2.0 * np.pi / (self.wavelength_nm * 1e-3))
            self._dirty = False
        return self._phase

    def reset(self) -> None:
        self._voltages = np.zeros(self.n_actuators, dtype=np.float64)
        self._opd = np.zeros(self.n_actuators, dtype=np.float64)
        self._dirty = True
        self._version += 1


__all__ = ["SimDmOptics"]
