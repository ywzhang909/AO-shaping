"""Strehl landscape over DM voltages for offline benchmark evaluation.

Wraps the offline physical simulation ``TraditionalAOSystem`` (turbulence
screen + DM influence functions + Fourier focal plane, see
``ao_shaping.drivers.sim.compat``) as a 64-D fitness landscape whose objective
is the on-axis **Strehl ratio** reported by the simulator.

One :meth:`StrehlLandscape.score` call == one device load (``set_dm_voltages``
+ ``observe``), so the benchmark's "load N" bookkeeping maps 1:1 to fitness
evaluations, matching the wavefront-sensorless hardware semantics where the
device can only hold one phase at a time.

The interface mirrors ``SimLandscape`` (``pib_sim_eval.py``) so the same
benchmark scaffold can consume either landscape:

* ``score(v) -> (strehl, strehl, 0.0)`` — the first component is the objective.
* ``render(v) -> uint16 far-field intensity image`` — for before/after spots.

Reproducibility contract: ``turbulence_phase()`` consumes the **global**
``numpy.random`` state (the ``rng`` argument is ignored upstream), so the seed
is applied in ``__post_init__`` *before* the ``TraditionalAOSystem`` is built.
Two landscapes with the same seed share the identical turbulence screen and
``_ideal_peak`` normalisation; ``reset()`` is never called, so the turbulence
is fixed for the whole benchmark run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ao_shaping.drivers.sim.compat import AOConfig, TraditionalAOSystem


@dataclass
class StrehlLandscape:
    """64-D DM-voltage landscape whose objective is the Strehl ratio.

    Args:
        seed: Random seed. Applied to the *global* numpy RNG before the
            turbulence screen is generated, so the landscape is deterministic
            for a given seed (turbulence is fixed for the whole run).
        n_grid: FFT grid size N of the simulation (256 like the hardware run,
            128 for smoke/unit tests).
        aperture: Circular aperture diameter L in metres.
        wavelength: Beam wavelength in metres.
        Cn2: Turbulence refractive-index structure constant (m^-2/3).
        L0 / l0: Outer/inner turbulence scale in metres.
        dm_actuators: DM actuator grid side length (8 -> 64-D voltage space).
        dm_stroke: DM stroke in metres (phase = v * stroke * 2*pi/lambda).
        propagation_distance: Turbulence propagation distance in metres.

    Attributes:
        ao: The wrapped ``TraditionalAOSystem`` (never ``reset()``).
        dim: Voltage-space dimension (``dm_actuators**2``).
        bounds: Per-axis voltage bounds ``(-1.0, 1.0)`` (clipped on set).
        init_v: Zero-voltage vector — the flat-DM starting point.
    """

    seed: int = 42
    n_grid: int = 256
    aperture: float = 0.1
    wavelength: float = 1550e-9
    Cn2: float = 1e-14
    L0: float = 10.0
    l0: float = 0.01
    dm_actuators: int = 8
    dm_stroke: float = 5e-6
    propagation_distance: float = 1000.0

    def __post_init__(self) -> None:
        # turbulence_phase() draws from the GLOBAL numpy RNG (beam_backend.py),
        # so seeding BEFORE construction fixes the turbulence screen.
        np.random.seed(self.seed)
        config = AOConfig(
            N=self.n_grid,
            L=self.aperture,
            wavelength=self.wavelength,
            Cn2=self.Cn2,
            L0=self.L0,
            l0=self.l0,
            dm_actuators=self.dm_actuators,
            dm_stroke=self.dm_stroke,
            propagation_distance=self.propagation_distance,
        )
        self.ao = TraditionalAOSystem(config=config)
        self.dim: int = self.ao.dm.total_actuators
        self.bounds: tuple[float, float] = (-1.0, 1.0)
        self.init_v: np.ndarray = np.zeros(self.dim, dtype=np.float64)

    def _coerce(self, voltages: np.ndarray) -> np.ndarray:
        v = np.asarray(voltages, dtype=np.float64)
        if v.shape != (self.dim,):
            msg = (
                f"expected {self.dim} DM voltages, got shape {v.shape}; "
                f"reshape to ({self.dim},)"
            )
            raise ValueError(msg)
        return v

    def score(self, voltages: np.ndarray) -> tuple[float, float, float]:
        """Strehl ratio at the given DM voltages (one device load).

        Returns:
            ``(strehl, strehl, 0.0)`` — the first component is the objective
            consumed by the benchmark; the tuple mirrors ``SimLandscape``.

        Fast path: the simulator's own Strehl (``observe()``) is exactly
        ``clip(_intensity.max() / _ideal_peak, 0.0, 1.0)`` (``compat.py``
        ``_strehl()``) plus a wavefront measurement we do not use (~40 % of
        the per-load cost). We replicate the identical clipped arithmetic over
        the cached float intensity, and fall back to ``observe()`` if the
        internals ever change shape.

        The clip is essential: the "ideal" reference is the far-field of the
        *Gaussian* pupil without turbulence/DM phase, which well-shaped DM
        phases can focus tighter than — the raw ratio routinely exceeds 1 at
        n_grid=256 (observed up to ~3.6), while ``observe()`` reports the
        clipped value. Without the clip the objective is not the simulator's
        Strehl and the benchmark numbers diverge from it.
        """
        v = self._coerce(voltages)
        self.ao.set_dm_voltages(v)
        self.ao.get_image()  # ensures the float _intensity is computed for v
        intensity = getattr(self.ao, "_intensity", None)
        if isinstance(intensity, np.ndarray):
            s = float(np.clip(intensity.max() / self.ao._ideal_peak, 0.0, 1.0))
        else:
            # internals changed shape — fall back to the simulator's own Strehl
            s = float(self.ao.observe()["strehl"])
        return s, s, 0.0

    def render(self, voltages: np.ndarray) -> np.ndarray:
        """Far-field intensity image (uint16) at the given DM voltages."""
        v = self._coerce(voltages)
        self.ao.set_dm_voltages(v)
        return self.ao.get_image()