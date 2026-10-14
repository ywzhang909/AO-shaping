"""Pupil-confined Zernike and freeform phase for GSNet inverse design.

The Zernike basis uses the project's canonical generator (Noll order, piston
excluded). Both components are raw radians; the SLM driver owns phase wrapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ao_shaping.utils.wavefront.fraunhofer import focal_intensity
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator, calc_n_zernike_terms


class HybridPupilPhase(nn.Module):
    """Trainable phase on a circular pupil at an explicit panel-grid position.

    ``center_xy`` and ``radius_px`` are in the output grid's pixel coordinates.
    The residual is interpolated from a small trainable grid, then masked to the
    pupil. This is a phase parameterization, not a learned camera forward model.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        center_xy: tuple[int, int],
        radius_px: int,
        *,
        n_max: int = 8,
        residual_grid: int = 16,
        base_phase: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        height, width = shape
        cx, cy = center_xy
        if n_max < 1 or residual_grid < 2 or radius_px < 2:
            raise ValueError("n_max >= 1, residual_grid >= 2 and radius_px >= 2 required")
        side = 2 * radius_px + 1
        x0, y0 = cx - radius_px, cy - radius_px
        if x0 < 0 or y0 < 0 or x0 + side > width or y0 + side > height:
            raise ValueError("pupil must fit entirely inside the output grid")

        generator = ZernikeGenerator((side, side), radius=radius_px, n_orders=n_max)
        n_terms = calc_n_zernike_terms(n_max)
        piston = generator.generate_noll(np.eye(n_terms, dtype=np.float64)[0])
        local_mask = np.isfinite(piston)
        mask = np.zeros(shape, dtype=np.float32)
        mask[y0 : y0 + side, x0 : x0 + side] = local_mask
        basis = np.zeros((n_terms - 1, height, width), dtype=np.float32)
        for index in range(1, n_terms):
            local = generator.generate_noll(np.eye(n_terms, dtype=np.float64)[index])
            basis[index - 1, y0 : y0 + side, x0 : x0 + side] = np.nan_to_num(local)

        if base_phase is None:
            base = np.zeros(shape, dtype=np.float32)
        else:
            base = np.asarray(base_phase, dtype=np.float32)
            if base.shape != shape or not np.isfinite(base).all():
                raise ValueError("base_phase must be finite and match output shape")
        self.register_buffer("pupil_mask", torch.from_numpy(mask))
        self.register_buffer("basis", torch.from_numpy(basis))
        self.register_buffer("base_phase", torch.from_numpy(base * mask))
        self._pupil_window = (y0, x0, side)
        self.coefficients = nn.Parameter(torch.zeros(n_terms - 1))
        self.residual = nn.Parameter(torch.zeros(1, 1, residual_grid, residual_grid))

    def forward(self) -> torch.Tensor:
        """Return one unwrapped phase map ``(H, W)`` in radians."""
        zernike = torch.einsum("k,khw->hw", self.coefficients, self.basis)
        y0, x0, side = self._pupil_window
        local_residual = F.interpolate(
            self.residual, size=(side, side), mode="bilinear", align_corners=False
        )[0, 0]
        freeform = F.pad(
            local_residual,
            (x0, self.pupil_mask.shape[1] - x0 - side, y0, self.pupil_mask.shape[0] - y0 - side),
        )
        return (self.base_phase + zernike + freeform) * self.pupil_mask


def far_field_intensity(source_intensity: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Differentiable ideal FFT intensity for a measured source-plane intensity."""
    if source_intensity.shape != phase.shape or source_intensity.ndim != 2:
        raise ValueError("source_intensity and phase must be matching 2D arrays")
    field = source_intensity.clamp_min(0).sqrt() * torch.exp(1j * phase)
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field))).abs().square()


@dataclass(frozen=True)
class CameraSampling:
    """Physical sampling of one square pupil grid onto a fixed CCD window."""

    aperture_size_m: float
    wavelength_m: float
    focal_length_m: float
    camera_pixel_m: float
    center_xy: tuple[float, float]
    far_field_padding: int = 3

    def __post_init__(self) -> None:
        dimensions = (
            self.aperture_size_m, self.wavelength_m,
            self.focal_length_m, self.camera_pixel_m, *self.center_xy,
        )
        if not np.isfinite(dimensions).all() or any(value <= 0 for value in dimensions[:4]):
            raise ValueError("optical dimensions must be finite and positive")
        if self.far_field_padding < 1:
            raise ValueError("far_field_padding must be >= 1")

    @property
    def far_field_pixel_m(self) -> float:
        return self.wavelength_m * self.focal_length_m / (
            self.aperture_size_m * self.far_field_padding
        )


def model_grid_aperture_size_m(
    grid_size: int,
    model_radius_px: float,
    panel_radius_px: float,
    panel_pixel_um: float,
) -> float:
    """Physical width of the full model grid from its mapped pupil radius."""
    values = (grid_size, model_radius_px, panel_radius_px, panel_pixel_um)
    if not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise ValueError("grid size, pupil radii and panel pixel size must be positive")
    return grid_size * panel_radius_px / model_radius_px * panel_pixel_um * 1e-6


def camera_far_field_intensity(
    source_intensity: torch.Tensor,
    phase: torch.Tensor,
    camera_shape: tuple[int, int],
    sampling: CameraSampling,
) -> torch.Tensor:
    """Differentiable ideal FFT sampled at physical CCD pixel centers."""
    if source_intensity.shape != phase.shape or phase.ndim != 2 or phase.shape[0] != phase.shape[1]:
        raise ValueError("source_intensity and phase must be matching square 2D arrays")
    height, width = camera_shape
    if height < 1 or width < 1:
        raise ValueError("camera_shape must be positive")
    field = source_intensity.clamp_min(0).sqrt() * torch.exp(1j * phase)
    size = phase.shape[0] * sampling.far_field_padding
    far = focal_intensity(field, size)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=phase.device, dtype=phase.dtype),
        torch.arange(width, device=phase.device, dtype=phase.dtype),
        indexing="ij",
    )
    cx, cy = sampling.center_xy
    scale = sampling.camera_pixel_m / sampling.far_field_pixel_m
    x = (size // 2 + (xx - cx) * scale) * (2.0 / max(size - 1, 1)) - 1.0
    y = (size // 2 + (yy - cy) * scale) * (2.0 / max(size - 1, 1)) - 1.0
    grid = torch.stack((x, y), dim=-1)[None]
    return F.grid_sample(
        far[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0]


def square_quality(intensity: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable GS-refine score, PIB and illuminated-target CV."""
    if intensity.shape != target.shape or intensity.ndim != 2:
        raise ValueError("intensity and target must be matching 2D arrays")
    inside = target > 0
    if not torch.any(inside):
        raise ValueError("target must contain illuminated pixels")
    roi = intensity[inside]
    pib = roi.sum() / intensity.sum().clamp_min(1e-12)
    cv = roi.std(unbiased=False) / roi.mean().clamp_min(1e-12)
    return 0.5 * pib + 0.5 / (1.0 + cv), pib, cv


@dataclass
class HybridRefinementResult:
    phase: np.ndarray
    initial_score: float
    best_score: float
    history: list[dict[str, float]]
    coefficients: np.ndarray
    residual: np.ndarray


class HybridPhaseOptimizer:
    """Refine a fixed GSNet prediction with Zernike-8 and a pupil residual.

    The base phase is frozen; only the two explicit correction components train.
    ``update`` takes one Adam step, and ``run`` returns the best state according
    to the same PIB/CV score used by hardware GS refine. The FFT is idealized,
    so this is an offline candidate generator rather than hardware acceptance.
    """

    def __init__(
        self,
        source_intensity: np.ndarray,
        target_intensity: np.ndarray,
        gsnet_phase: np.ndarray,
        *,
        center_xy: tuple[int, int],
        radius_px: int,
        n_max: int = 8,
        residual_grid: int = 16,
        lr: float = 0.05,
        train_residual: bool = True,
        camera_target_intensity: np.ndarray | None = None,
        camera_sampling: CameraSampling | None = None,
    ) -> None:
        source = np.asarray(source_intensity, dtype=np.float32)
        target = np.asarray(target_intensity, dtype=np.float32)
        base = np.asarray(gsnet_phase, dtype=np.float32)
        if source.ndim != 2 or source.shape != target.shape or source.shape != base.shape:
            raise ValueError("source, target and GSNet phase must be matching 2D arrays")
        if not all(np.isfinite(array).all() for array in (source, target, base)):
            raise ValueError("source, target and GSNet phase must be finite")
        if np.any(source < 0) or np.any(target < 0):
            raise ValueError("source and target intensity must be non-negative")
        if not np.any(source) or not np.any(target):
            raise ValueError("source and target intensity must contain light")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if (camera_target_intensity is None) != (camera_sampling is None):
            raise ValueError("camera_target_intensity and camera_sampling must be supplied together")
        self.source = torch.from_numpy(source)
        self.camera_sampling = camera_sampling
        if camera_target_intensity is None:
            self.target = torch.from_numpy(target)
        else:
            camera_target = np.asarray(camera_target_intensity, dtype=np.float32)
            if camera_target.ndim != 2 or not np.isfinite(camera_target).all():
                raise ValueError("camera target must be a finite 2D array")
            if np.any(camera_target < 0) or not np.any(camera_target):
                raise ValueError("camera target must contain non-negative light")
            self.target = torch.from_numpy(camera_target)
        self.phase_model = HybridPupilPhase(
            source.shape, center_xy, radius_px,
            n_max=n_max, residual_grid=residual_grid, base_phase=base,
        )
        self.phase_model.residual.requires_grad_(train_residual)
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.phase_model.parameters() if parameter.requires_grad),
            lr=lr,
        )
        self.history: list[dict[str, float]] = []
        self._steps_taken = 0

    @classmethod
    def from_gsnet(
        cls,
        model: nn.Module,
        source_intensity: np.ndarray,
        target_intensity: np.ndarray,
        *,
        center_xy: tuple[int, int],
        radius_px: int,
        n_max: int = 8,
        residual_grid: int = 16,
        lr: float = 0.05,
        train_residual: bool = True,
        camera_target_intensity: np.ndarray | None = None,
        camera_sampling: CameraSampling | None = None,
    ) -> HybridPhaseOptimizer:
        """Use a trained GSNet prediction as the frozen refinement base."""
        source = np.asarray(source_intensity, dtype=np.float32)
        target = np.asarray(target_intensity, dtype=np.float32)
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError("source and target must be matching 2D arrays")
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                predicted = model(
                    torch.from_numpy(source)[None, None].to(device),
                    torch.from_numpy(target)[None, None].to(device),
                )
            if predicted.shape != (1, 1, *source.shape):
                raise ValueError("GSNet must return one phase map matching the input grid")
            base = predicted[0, 0].detach().cpu().numpy()
        finally:
            model.train(was_training)
        return cls(
            source, target, base, center_xy=center_xy, radius_px=radius_px,
            n_max=n_max, residual_grid=residual_grid, lr=lr,
            train_residual=train_residual,
            camera_target_intensity=camera_target_intensity,
            camera_sampling=camera_sampling,
        )

    def _measure(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = self.phase_model()
        if self.camera_sampling is None:
            intensity = far_field_intensity(self.source, phase)
        else:
            intensity = camera_far_field_intensity(
                self.source, phase, tuple(self.target.shape), self.camera_sampling
            )
        return square_quality(intensity, self.target)

    def _record(self) -> float:
        with torch.no_grad():
            score, pib, cv = self._measure()
        self.history.append({
            "step": float(len(self.history)), "score": float(score),
            "pib": float(pib), "cv": float(cv),
        })
        return float(score)

    def update(self) -> np.ndarray:
        """Take one optimizer step and return the current phase in radians."""
        if not self.history:
            self._record()
        self.optimizer.zero_grad(set_to_none=True)
        score, _, _ = self._measure()
        (1.0 - score).backward()
        self.optimizer.step()
        self._steps_taken += 1
        self._record()
        return self.phase_model().detach().numpy().copy()

    def run(self, steps: int) -> HybridRefinementResult:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if self._steps_taken:
            raise RuntimeError("run must start before standalone update calls")
        initial_score = self._record() if not self.history else self.history[0]["score"]
        best_score = self.history[-1]["score"]
        best_phase = self.phase_model().detach().numpy().copy()
        best_coefficients = self.phase_model.coefficients.detach().numpy().copy()
        best_residual = self.phase_model.residual.detach().numpy().copy()
        for _ in range(steps):
            phase = self.update()
            score = self.history[-1]["score"]
            if score > best_score:
                best_score = score
                best_phase = phase
                best_coefficients = self.phase_model.coefficients.detach().numpy().copy()
                best_residual = self.phase_model.residual.detach().numpy().copy()
        return HybridRefinementResult(
            phase=best_phase, initial_score=initial_score, best_score=best_score,
            history=list(self.history), coefficients=best_coefficients,
            residual=best_residual,
        )
