"""Compact differentiable physical-model fitter; no hardware dependencies."""
from __future__ import annotations

import time
from loguru import logger
from datetime import datetime, timezone

import numpy as np

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    PANEL_PITCH_PX,
    PANEL_W,
    camera_sample_grids,
    defocus_phase,
    diopter_scale,
    gaussian_amplitude,
    irradiance_scale,
    panel_grid,
    sample_intensity,
    subpixel_offsets,
)


class PhysicalForwardFit:
    """Compact shared physical fit with fixed camera geometry.

    The panel geometry, Gaussian illumination, camera sampling grids and FFT
    irradiance scaling live in ``defocus_forward_core`` (issue #199 R2), shared
    bit-for-bit with ``FrozenDefocusForward``; this class keeps only the
    fitting-specific parts (trainable defocus / origin / amplitude, the
    control-grid Zernike basis, batching).
    """

    def __init__(self, phases, images, camera_xy, matrix, focus_diopter, train_mask,
                 amplitude_center, *, learn_amplitude=False, initial_defocus=None,
                 learn_residual=False, fft_size=None, pixel_samples=1,
                 basis_arrays=None, batch_size=None, device=None, diagnostic_callback=None):
        import torch
        from ao_shaping.utils.wavefront.fraunhofer import focal_intensity
        from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

        self.torch = torch
        self.propagate = focal_intensity
        # Default stays CPU: every existing caller (scripts, tests, other
        # fitters) keeps its numbers. A caller that has a device to spend passes
        # it explicitly -- see ``ao_shaping.tools.slm.transfer.device``.
        self._device = torch.device(device or "cpu")
        self.phases = torch.tensor(phases, dtype=torch.float32, device=self._device)
        size = self.phases.shape[-1]
        if self.phases.ndim != 3 or self.phases.shape[-2] != size or size < 8:
            raise ValueError("phases must be square batched grids of size >= 8")
        self.pitch = PANEL_PITCH_PX / size
        self.fft_size = fft_size or 4 * size
        if self.fft_size < size or self.fft_size % 2 or pixel_samples < 1:
            raise ValueError("require even fft_size >= phase grid, pixel_samples >= 1")
        self.pixel_samples = pixel_samples
        self.batch_size = batch_size or len(phases)
        self.diagnostic_callback = diagnostic_callback
        self.images = torch.tensor(images, dtype=torch.float32, device=self._device)
        self.xy = torch.tensor(camera_xy, dtype=torch.float32, device=self._device)
        self.inverse = torch.tensor(np.linalg.inv(matrix), dtype=torch.float32, device=self._device)
        self.train = torch.tensor(train_mask, dtype=torch.bool, device=self._device)
        self.z4 = torch.tensor(basis_arrays[4] if basis_arrays is not None else
                               np.nan_to_num(generate_zernike_phase({4: 1.0}, (size, size))),
                               dtype=torch.float32, device=self._device)
        # Z4 = sqrt(3)*(2*r^2 - 1); pupil half-span is 960 panel pixels.
        self.diopter_scale = diopter_scale()
        self.panel_x, self.panel_y, self.panel_mask = panel_grid(torch, size, device=self._device)
        d = -focus_diopter if initial_defocus is None else initial_defocus
        self.d = torch.nn.Parameter(torch.tensor(float(d), device=self._device))
        # Use one shared optical-axis intercept, not per-image fitted shifts.
        self.origin = torch.nn.Parameter(torch.tensor([1360., 1020.], device=self._device))
        self.center = torch.nn.Parameter(torch.tensor(amplitude_center, dtype=torch.float32,
                                                      device=self._device),
                                         requires_grad=learn_amplitude)
        self.log_width = torch.nn.Parameter(torch.log(torch.tensor([260., 260.], device=self._device)),
                                            requires_grad=learn_amplitude)
        params = [{"params": [self.d], "lr": .01}, {"params": [self.origin], "lr": 1.0}]
        if learn_amplitude:
            params += [{"params": [self.center], "lr": .5}, {"params": [self.log_width], "lr": .01}]
        self.residual_modes = list(range(5, 12)) if learn_residual else []
        self.residual = torch.nn.Parameter(torch.zeros(len(self.residual_modes), device=self._device),
                                           requires_grad=learn_residual)
        self.residual_basis = torch.tensor(np.stack([
            basis_arrays[j] if basis_arrays is not None else
            np.nan_to_num(generate_zernike_phase({j: 1.}, (size, size)))
            for j in self.residual_modes
        ]), dtype=torch.float32, device=self._device) if self.residual_modes else None
        if learn_residual:
            params += [{"params": [self.residual], "lr": .03}]
        self.optimizer = torch.optim.Adam(params)
        self.history = []
        self.iteration_records = []

    def prediction(self):
        torch = self.torch
        width = self.log_width.exp()
        amplitude = gaussian_amplitude(torch, self.panel_x, self.panel_y, self.center, width,
                                       self.panel_mask)
        if self.diagnostic_callback is not None:
            self.diagnostic_callback("input_amplitude", amplitude, None)
        grids = camera_sample_grids(
            torch, self.xy, self.origin, self.inverse,
            subpixel_offsets(torch, self.pixel_samples, device=self.xy.device),
            fft_size=self.fft_size, pitch=self.pitch)
        predictions = []
        for start in range(0, len(self.phases), self.batch_size):
            phase = defocus_phase(torch, self.phases[start:start+self.batch_size], self.d,
                                  self.diopter_scale, self.z4,
                                  residual_values=self.residual,
                                  residual_basis=self.residual_basis)
            field = amplitude * torch.exp(1j * phase)
            intensity = self.propagate(field, self.fft_size)
            # Preserve physical irradiance units when input pitch or FFT changes.
            intensity = intensity * irradiance_scale(self.pitch, self.fft_size)
            sampled = sample_intensity(torch, intensity, grids)
            if self.diagnostic_callback is not None:
                for name, response in (("panel_phase", phase), ("panel_field", field),
                                       ("propagated_intensity", intensity), ("ccd_integrated", sampled)):
                    self.diagnostic_callback(name, response, start)
            predictions.append(sampled)
        return torch.cat(predictions)

    def update(self):
        started = time.monotonic_ns()
        torch = self.torch
        self.optimizer.zero_grad()
        pred = self.prediction()[self.train]
        measured = self.images[self.train]
        # Shape fit with cosine similarity; shared absolute gain is estimated
        # ONLY on training samples after fitting, never separately on test data.
        p, t = pred.flatten(1), measured.flatten(1)
        p = p - p.mean(1, keepdim=True)
        t = t - t.mean(1, keepdim=True)
        loss = (1 - (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-8)).mean()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.d.clamp_(-2, 2)
            self.center.clamp_(0, PANEL_W)
            self.log_width.clamp_(float(np.log(80)), float(np.log(800)))
        self.history.append(float(loss.detach()))
        self.iteration_records.append(dict(iteration=len(self.history), loss=self.history[-1],
                                           timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                                           monotonic_start_ns=started, monotonic_end_ns=time.monotonic_ns()))
        return self.history[-1]

    def run(self, iterations):
        # Optical-axis coordinates are measured in pixels, use a larger LR.
        for index in range(iterations):
            loss = self.update()
            if index % 25 == 0:
                logger.info(f'fit {index}: loss={loss:.5f} d={self.d.item():.5f}')
        return self.prediction().detach().cpu().numpy()

