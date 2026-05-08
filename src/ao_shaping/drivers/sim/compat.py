"""Compatibility layer for legacy ao_shaping.sim module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ao_shaping.drivers.sim.beam_backend import (
    focal_plane,
    gaussian_pupil,
    grid,
    make_beam_config,
    turbulence_phase,
)


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
    def __init__(self, config: AOConfig | None = None):
        self.config = config or AOConfig()
        cfg = self.config

        self.dm_voltages = np.zeros(cfg.dm_actuators**2, dtype=float)
        self._turbulence_phase: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._base_field: np.ndarray | None = None
        self._dm_surface: np.ndarray | None = None
        self._focal_field: np.ndarray | None = None
        self._intensity: np.ndarray | None = None
        self._image: np.ndarray | None = None
        self._wavefront_override: np.ndarray | None = None

        self._beam_cfg = make_beam_config(
            n_grid=cfg.N,
            aperture_size=cfg.L,
            wavelength=cfg.wavelength,
            cn2=cfg.Cn2,
            l_max=cfg.L0,
            l_min=cfg.l0,
            propagation_distance=cfg.propagation_distance,
        )
        self._init_components()

    def _init_components(self) -> None:
        cfg = self.config
        x, y = grid(self._beam_cfg)
        radius = np.sqrt(x**2 + y**2)
        self._mask = radius <= (cfg.L / 2)
        self._x = x
        self._y = y
        self._base_field = gaussian_pupil(self._beam_cfg, aperture_radius=cfg.L / 2)
        self._init_dm_surface()
        self._sample_turbulence_phase()
        ideal = self._propagate_field(
            self._base_field,
            dm_phase=np.zeros((cfg.N, cfg.N), dtype=float),
            turb_phase=np.zeros((cfg.N, cfg.N), dtype=float),
        )
        self._ideal_peak = float(np.max(np.abs(ideal) ** 2))

    def _sample_turbulence_phase(self) -> None:
        cfg = self.config
        self._turbulence_phase = turbulence_phase(
            self._beam_cfg,
            cn2=cfg.Cn2,
            l_max=cfg.L0,
            l_min=cfg.l0,
            propagation_distance=cfg.propagation_distance,
        )

    def _init_dm_surface(self) -> None:
        cfg = self.config
        x = self._x
        y = self._y

        sigma = 0.8 / cfg.dm_actuators * (cfg.L / 2)
        act_x = np.linspace(-0.9, 0.9, cfg.dm_actuators) * (cfg.L / 2)
        act_y = np.linspace(-0.9, 0.9, cfg.dm_actuators) * (cfg.L / 2)
        act_X, act_Y = np.meshgrid(act_x, act_y)

        inf_matrix = np.zeros((cfg.dm_actuators**2, cfg.N, cfg.N), dtype=float)
        for i, (ax, ay) in enumerate(zip(act_X.flatten(), act_Y.flatten())):
            r2 = (x - ax) ** 2 + (y - ay) ** 2
            inf_matrix[i] = np.exp(-r2 / (2 * sigma**2)) * self._mask

        self._inf_matrix = inf_matrix
        self._dm_surface = np.zeros((cfg.N, cfg.N), dtype=float)

    def _invalidate_cached_outputs(self) -> None:
        self._focal_field = None
        self._intensity = None
        self._image = None

    def set_dm_voltages(self, voltages: np.ndarray) -> None:
        self.dm_voltages = np.clip(np.asarray(voltages, dtype=float), -1.0, 1.0)
        self._dm_surface = np.tensordot(self.dm_voltages, self._inf_matrix, axes=1)
        self._wavefront_override = None
        self._invalidate_cached_outputs()

    @property
    def dm(self) -> _DMProxy:
        return _DMProxy(self)

    @property
    def turbulence(self) -> _TurbProxy:
        return _TurbProxy(self)

    @property
    def E_corrected(self) -> np.ndarray:
        return self._get_corrected_wave()

    @E_corrected.setter
    def E_corrected(self, value: np.ndarray) -> None:
        self._wavefront_override = np.asarray(value, dtype=np.complex128).copy()
        self._invalidate_cached_outputs()

    def _get_dm_phase(self) -> np.ndarray:
        return self._dm_surface * self.config.dm_stroke * (2 * np.pi / self.config.wavelength)

    def _get_corrected_wave(self) -> np.ndarray:
        if self._wavefront_override is not None:
            return self._wavefront_override

        total_phase = self._get_dm_phase()
        if self._turbulence_phase is not None:
            total_phase = total_phase + self._turbulence_phase
        return self._base_field * np.exp(1j * total_phase)

    def _propagate_field(
        self,
        pupil_field: np.ndarray,
        *,
        dm_phase: np.ndarray | None = None,
        turb_phase: np.ndarray | None = None,
    ) -> np.ndarray:
        phase = np.zeros_like(pupil_field, dtype=float)
        if dm_phase is not None:
            phase = phase + dm_phase
        if turb_phase is not None:
            phase = phase + turb_phase
        field = pupil_field * np.exp(1j * phase)
        return focal_plane(field, self._beam_cfg, self.config.propagation_distance)

    def _compute_image(self) -> np.ndarray:
        if self._intensity is None:
            self._focal_field = focal_plane(
                self._get_corrected_wave(),
                self._beam_cfg,
                self.config.propagation_distance,
            )
            self._intensity = np.abs(self._focal_field) ** 2

        image = self._intensity / max(float(np.max(self._intensity)), 1e-20)
        return np.round(image * 65535.0).astype(np.uint16)

    def get_image(self) -> np.ndarray:
        if self._image is None:
            self._image = self._compute_image()
        return self._image

    def _phase_rms(self) -> float:
        phase = np.angle(self._get_corrected_wave())
        masked = phase[self._mask]
        return float(np.sqrt(np.mean(masked**2))) if masked.size else 0.0

    def _strehl(self) -> float:
        if self._intensity is None:
            self._compute_image()
        peak = float(np.max(self._intensity))
        return float(np.clip(peak / max(self._ideal_peak, 1e-12), 0.0, 1.0))

    def measure_wavefront(self) -> np.ndarray:
        phase = np.angle(self._get_corrected_wave()) * self._mask
        grad_y, grad_x = np.gradient(phase)
        sub = self.config.subapertures
        sub_size = self.config.N // sub
        slopes_x: list[float] = []
        slopes_y: list[float] = []

        for i in range(sub):
            for j in range(sub):
                row_slice = slice(i * sub_size, (i + 1) * sub_size)
                col_slice = slice(j * sub_size, (j + 1) * sub_size)
                sub_mask = self._mask[row_slice, col_slice]
                if not np.any(sub_mask):
                    slopes_x.append(0.0)
                    slopes_y.append(0.0)
                    continue
                sx = grad_x[row_slice, col_slice][sub_mask]
                sy = grad_y[row_slice, col_slice][sub_mask]
                slopes_x.append(float(np.mean(sx) * self.config.pixel_scale))
                slopes_y.append(float(np.mean(sy) * self.config.pixel_scale))

        return np.array(slopes_x + slopes_y, dtype=np.float32)

    def _build_result(self) -> dict[str, Any]:
        image = self.get_image()
        return {
            "image": image,
            "slopes": self.measure_wavefront(),
            "strehl": self._strehl(),
            "power": float(np.sum(self._intensity)) if self._intensity is not None else float(np.sum(image)),
            "voltages": self.dm_voltages.copy(),
            "phase_rms": self._phase_rms(),
        }

    def observe(self) -> dict[str, Any]:
        return self._build_result()

    def reset(self) -> dict[str, Any]:
        self._sample_turbulence_phase()
        self._invalidate_cached_outputs()
        return self._build_result()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        self.set_dm_voltages(self.dm_voltages + np.asarray(action, dtype=float))
        return self._build_result()


class _DMProxy:
    def __init__(self, ao: TraditionalAOSystem):
        self._ao = ao

    @property
    def total_actuators(self) -> int:
        return self._ao.config.dm_actuators**2


class _TurbProxy:
    def __init__(self, ao: TraditionalAOSystem):
        self._ao = ao

    @property
    def phase_screen(self) -> np.ndarray:
        if self._ao._turbulence_phase is None:
            return np.zeros((self._ao.config.N, self._ao.config.N), dtype=float)
        return self._ao._turbulence_phase

    def get_phase_screen(self) -> np.ndarray:
        return self.phase_screen


__all__ = ["AOConfig", "TraditionalAOSystem"]
