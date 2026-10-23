"""Compact differentiable physical-model fitter; no hardware dependencies."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np


class PhysicalForwardFit:
    """Nine-parameter shared physical fit with fixed camera geometry."""

    def __init__(self, phases, images, camera_xy, matrix, focus_diopter, train_mask,
                 amplitude_center, *, learn_amplitude=False, initial_defocus=None,
                 learn_residual=False, fft_size=None, pixel_samples=1,
                 basis_arrays=None, batch_size=None):
        import torch
        from ao_shaping.utils.wavefront.fraunhofer import focal_intensity
        from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

        self.torch = torch
        self.propagate = focal_intensity
        self.phases = torch.tensor(phases, dtype=torch.float32)
        size = self.phases.shape[-1]
        if self.phases.ndim != 3 or self.phases.shape[-2] != size or size < 8:
            raise ValueError("phases must be square batched grids of size >= 8")
        self.pitch = 1920. / size
        self.fft_size = fft_size or 4 * size
        if self.fft_size < size or self.fft_size % 2 or pixel_samples < 1:
            raise ValueError("require even fft_size >= phase grid, pixel_samples >= 1")
        self.pixel_samples = pixel_samples
        self.batch_size = batch_size or len(phases)
        self.images = torch.tensor(images, dtype=torch.float32)
        self.xy = torch.tensor(camera_xy, dtype=torch.float32)
        self.inverse = torch.tensor(np.linalg.inv(matrix), dtype=torch.float32)
        self.train = torch.tensor(train_mask, dtype=torch.bool)
        self.z4 = torch.tensor(basis_arrays[4] if basis_arrays is not None else
                               np.nan_to_num(generate_zernike_phase({4: 1.0}, (size, size))),
                               dtype=torch.float32)
        # Z4 = sqrt(3)*(2*r^2 - 1); pupil half-span is 960 panel pixels.
        self.diopter_scale = float(np.pi * (960 * 8e-6) ** 2 / (2 * np.sqrt(3) * 1064e-9))
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        self.panel_x = x.float() * self.pitch
        self.panel_y = y.float() * self.pitch
        self.panel_mask = (self.panel_y < 1200).float()
        d = -focus_diopter if initial_defocus is None else initial_defocus
        self.d = torch.nn.Parameter(torch.tensor(float(d)))
        # Use one shared optical-axis intercept, not per-image fitted shifts.
        self.origin = torch.nn.Parameter(torch.tensor([1360., 1020.]))
        self.center = torch.nn.Parameter(torch.tensor(amplitude_center, dtype=torch.float32),
                                         requires_grad=learn_amplitude)
        self.log_width = torch.nn.Parameter(torch.log(torch.tensor([260., 260.])),
                                            requires_grad=learn_amplitude)
        params = [{"params": [self.d], "lr": .01}, {"params": [self.origin], "lr": 1.0}]
        if learn_amplitude:
            params += [{"params": [self.center], "lr": .5}, {"params": [self.log_width], "lr": .01}]
        self.residual_modes = list(range(5, 12)) if learn_residual else []
        self.residual = torch.nn.Parameter(torch.zeros(len(self.residual_modes)), requires_grad=learn_residual)
        self.residual_basis = torch.tensor(np.stack([
            basis_arrays[j] if basis_arrays is not None else
            np.nan_to_num(generate_zernike_phase({j: 1.}, (size, size)))
            for j in self.residual_modes
        ]), dtype=torch.float32) if self.residual_modes else None
        if learn_residual:
            params += [{"params": [self.residual], "lr": .03}]
        self.optimizer = torch.optim.Adam(params)
        self.history = []
        self.iteration_records = []

    def prediction(self):
        torch = self.torch
        width = self.log_width.exp()
        amplitude = torch.exp(-((self.panel_x - self.center[0]) / width[0]) ** 2
                              -((self.panel_y - self.center[1]) / width[1]) ** 2) * self.panel_mask
        phase = self.phases + self.d * self.diopter_scale * self.z4
        if self.residual_basis is not None:
            phase = phase + torch.einsum("i,ijk->jk", self.residual, self.residual_basis)
        grids = []
        for iy in range(self.pixel_samples):
            for ix in range(self.pixel_samples):
                offset = self.xy.new_tensor([(ix+.5)/self.pixel_samples-.5,
                                             (iy+.5)/self.pixel_samples-.5])
                frequencies = (self.xy + offset - self.origin) @ self.inverse.T
                fft_indices = frequencies * (self.fft_size * self.pitch) + self.fft_size//2
                grids.append(2 * fft_indices / (self.fft_size-1) - 1)
        predictions = []
        for start in range(0, len(phase), self.batch_size):
            field = amplitude * torch.exp(1j * phase[start:start+self.batch_size])
            intensity = self.propagate(field, self.fft_size)
            # Preserve physical irradiance units when input pitch or FFT changes.
            intensity = intensity * (self.pitch**2 * self.fft_size / (15**2 * 512))**2
            sampled = sum(torch.nn.functional.grid_sample(
                intensity[:, None], grid[None].expand(len(intensity), -1, -1, -1),
                align_corners=True, padding_mode="zeros")[:, 0] for grid in grids) / len(grids)
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
            self.center.clamp_(0, 1920)
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
                print(f"fit {index}: loss={loss:.5f} d={self.d.item():.5f}", flush=True)
        return self.prediction().detach().numpy()

