"""Compatibility layer for legacy ao_shaping.sim module.

This module provides AOConfig and TraditionalAOSystem interfaces
that were available in the deleted ao_shaping.sim package, re-implemented
using sim.digitaltwin physics.

These are NOT device drivers - they are standalone simulation utilities
for optimizer scripts (sim_spgd.py, envs.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AOConfig:
    N: int = 256
    L: float = 0.1
    wavelength: float = 1550e-9
    Cn2: float = 1e-14
    L0: float = 10.0
    l0: float = 0.01
    dm_actuators: int = 8
    dm_stroke: float = 5e-6
    dm_infill: bool = True
    subapertures: int = 8
    pixel_scale: float = 0.5
    propagation_distance: float = 1000.0


class TraditionalAOSystem:
    def __init__(self, config: Optional[AOConfig] = None):
        self.config = config or AOConfig()
        cfg = self.config

        self.dm_voltages = np.zeros(cfg.dm_actuators ** 2)
        self._turbulence_phase: Optional[np.ndarray] = None

        self._wave: Any = None
        self._turb_screen: Any = None
        self._env: Any = None
        self._mask: Optional[np.ndarray] = None
        self._dm_surface: Optional[np.ndarray] = None
        self._phase_corrected: Optional[np.ndarray] = None
        self._intensity: Optional[np.ndarray] = None
        self._image: Optional[np.ndarray] = None

        self._init_components()

    def _init_components(self) -> None:
        cfg = self.config

        from sim.digitaltwin import base, screens, utilities

        self._dt_base = base
        self._dt_screens = screens
        self._dt_utils = utilities

        wave = base.Wave()
        wave.change_grid(cfg.N, cfg.L / cfg.N)
        wave.wavelength = cfg.wavelength
        wave.refractive = 1.0
        wave.wavefront = np.ones((cfg.N, cfg.N), dtype=complex)
        self._wave = wave

        env = base.Environment()
        env.Cn2 = cfg.Cn2
        env.L0 = cfg.L0
        env.l0 = cfg.l0
        self._env = env

        self._turb_screen = screens.TurbulentScreen(
            cfg.propagation_distance, env, harmonic=0
        )

        self._dpix = cfg.L / cfg.N
        self._aperture = cfg.L / 2

        x = np.linspace(-cfg.L / 2, cfg.L / 2, cfg.N)
        y = np.linspace(-cfg.L / 2, cfg.L / 2, cfg.N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)
        mask = (np.sign(self._aperture - R) + 1) / 2
        self._mask = mask

        self._focus_phase = -np.pi * R ** 2 / cfg.wavelength / cfg.propagation_distance

        if cfg.Cn2 > 0:
            self._turb_screen.out(wave)
            self._turbulence_phase = np.angle(wave.wavefront)
        else:
            self._turbulence_phase = np.zeros((cfg.N, cfg.N))

        self._init_dm_surface()

    def _init_dm_surface(self) -> None:
        cfg = self.config
        wave = self._wave

        x = np.linspace(-cfg.L / 2, cfg.L / 2, cfg.N)
        y = np.linspace(-cfg.L / 2, cfg.L / 2, cfg.N)
        X, Y = np.meshgrid(x, y)

        sigma = 0.8 / cfg.dm_actuators
        act_x = np.linspace(-0.9, 0.9, cfg.dm_actuators) * (cfg.L / 2)
        act_y = np.linspace(-0.9, 0.9, cfg.dm_actuators) * (cfg.L / 2)
        act_X, act_Y = np.meshgrid(act_x, act_y)

        inf_matrix = np.zeros((cfg.dm_actuators ** 2, cfg.N, cfg.N))
        for i, (ax, ay) in enumerate(
            zip(act_X.flatten(), act_Y.flatten())
        ):
            R2 = (X - ax) ** 2 + (Y - ay) ** 2
            inf_matrix[i] = np.exp(-R2 / (2 * sigma ** 2))

        self._inf_matrix = inf_matrix
        self._act_x = act_x
        self._act_y = act_y
        self._grid_x = x
        self._grid_y = y
        self._X = X
        self._Y = Y

        self._dm_surface = np.zeros((cfg.N, cfg.N))

    def set_dm_voltages(self, voltages: np.ndarray) -> None:
        self.dm_voltages = np.clip(voltages, -1, 1)
        self._dm_surface = np.tensordot(
            self.dm_voltages, self._inf_matrix, axes=1
        )
        self._intensity = None
        self._image = None

    @property
    def dm(self) -> "_DMProxy":
        return _DMProxy(self)

    @property
    def turbulence(self) -> "_TurbProxy":
        return _TurbProxy(self)

    @property
    def E_corrected(self) -> np.ndarray:
        return self._get_corrected_wave()
    
    @E_corrected.setter
    def E_corrected(self, value: np.ndarray) -> None:
        pass

    def _get_corrected_wave(self) -> np.ndarray:
        cfg = self.config
        wave = self._wave

        wave.wavefront = self._mask.astype(complex)
        wave.change_wf(phase=self._focus_phase)

        phase = self._focus_phase
        if self._turbulence_phase is not None:
            phase = phase + self._turbulence_phase

        phase = phase + self._dm_surface * 2 * np.pi / cfg.wavelength

        wave.change_wf(phase=phase)
        return wave.wavefront

    def _compute_image(self) -> np.ndarray:
        if self._intensity is None:
            E = self._get_corrected_wave()
            self._intensity = np.abs(E) ** 2

        img = self._intensity.copy()
        img = img / (np.max(img) + 1e-20) * 65535
        return img.astype(np.uint16)

    def get_image(self) -> np.ndarray:
        if self._image is None:
            self._image = self._compute_image()
        return self._image

    def measure_wavefront(self) -> np.ndarray:
        E = self._get_corrected_wave()
        intensity = np.abs(E) ** 2
        phase = np.angle(E)

        sub = self.config.subapertures
        sub_size = self.config.N // sub
        N = self.config.N

        x = np.arange(N)
        y = np.arange(N)
        X, Y = np.meshgrid(x, y)

        slopes_x: list[float] = []
        slopes_y: list[float] = []

        for i in range(sub):
            for j in range(sub):
                sl = slice(i * sub_size, (i + 1) * sub_size)
                mask = np.zeros((N, N), dtype=bool)
                mask[sl, sl] = True

                sub_I = intensity[mask]
                sub_X = X[mask]
                sub_Y = Y[mask]

                if np.sum(sub_I) < 1e-10:
                    slopes_x.append(0.0)
                    slopes_y.append(0.0)
                    continue

                cx = np.sum(sub_X * sub_I) / np.sum(sub_I)
                cy = np.sum(sub_Y * sub_I) / np.sum(sub_I)
                center_x = np.mean(sub_X)
                center_y = np.mean(sub_Y)

                slopes_x.append((cx - center_x) * self.config.pixel_scale)
                slopes_y.append((cy - center_y) * self.config.pixel_scale)

        return np.array(slopes_x + slopes_y)

    def reset(self) -> dict[str, Any]:
        self._intensity = None
        self._image = None

        img = self.get_image()
        slopes = self.measure_wavefront()

        phase = np.angle(self._get_corrected_wave())
        phase_rms = np.sqrt(np.mean(phase**2))
        strehl = float(np.exp(-phase_rms**2)) if phase_rms < 10 else 0.001

        return {
            "image": img,
            "slopes": slopes,
            "strehl": strehl,
            "power": float(np.sum(img)),
            "voltages": self.dm_voltages.copy(),
        }

    def step(self, action: np.ndarray) -> dict[str, Any]:
        new_voltages = self.dm_voltages + action
        self.set_dm_voltages(new_voltages)

        img = self.get_image()
        slopes = self.measure_wavefront()

        phase = np.angle(self._get_corrected_wave())
        phase_rms = np.sqrt(np.mean(phase**2))
        strehl = float(np.exp(-phase_rms**2)) if phase_rms < 10 else 0.001

        return {
            "image": img,
            "slopes": slopes,
            "strehl": strehl,
            "power": float(np.sum(img)),
            "voltages": self.dm_voltages.copy(),
        }


class _DMProxy:
    def __init__(self, ao: TraditionalAOSystem):
        self._ao = ao

    @property
    def total_actuators(self) -> int:
        cfg = self._ao.config
        return cfg.dm_actuators ** 2


class _TurbProxy:
    def __init__(self, ao: TraditionalAOSystem):
        self._ao = ao

    @property
    def phase_screen(self) -> np.ndarray:
        if self._ao._turbulence_phase is None:
            return np.zeros((self._ao.config.N, self._ao.config.N))
        return self._ao._turbulence_phase

    def get_phase_screen(self) -> np.ndarray:
        return self.phase_screen


__all__ = ["AOConfig", "TraditionalAOSystem"]
