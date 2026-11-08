"""Compact differentiable physical-model fitter; no hardware dependencies."""
from __future__ import annotations

import time
from loguru import logger
from datetime import datetime, timezone

import numpy as np

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    PANEL_PITCH_PX,
    PHASE_SIGN_SCOPES,
    scoped_command_phase,
    PANEL_W,
    camera_sample_grids,
    defocus_phase,
    diopter_scale,
    gaussian_amplitude,
    irradiance_scale,
    panel_grid,
    registration_sample_grid,
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
                 basis_arrays=None, batch_size=None, device=None, diagnostic_callback=None,
                 slm_phase_a=0., slm_phase_b=0., amplitude_residual_coeffs=(),
                 monotonic_ccd_h=(), monotonic_ccd_v_max=255., phase_sign=1.,
                 phase_sign_scope="command", focus_phase=None, carrier_phase=None,
                 registration_shift_ccd_px=None, learn_registration=False):
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
        # Must match ``FrozenDefocusForward``: the fit has to optimise under the
        # same propagation convention it will be exported with (issue #235).
        self.phase_sign = float(phase_sign)
        if self.phase_sign not in (1., -1.):
            raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign!r}")
        self.phase_sign_scope = str(phase_sign_scope)
        if self.phase_sign_scope not in PHASE_SIGN_SCOPES:
            raise ValueError(f"phase_sign_scope invalid: {phase_sign_scope!r}")
        # Named components (per row, optional). ``focus_phase`` is contained in ``phases``
        # for the paired dataset; it is used only by the ``aberration_focus`` scope.
        self.focus_phase = None if focus_phase is None else torch.as_tensor(
            np.asarray(focus_phase), dtype=torch.float32, device=self._device)
        self.carrier_phase = None if carrier_phase is None else torch.as_tensor(
            np.asarray(carrier_phase), dtype=torch.float32, device=self._device)
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
        # Optional global 2D registration (issue #237).  One global (dy, dx) in
        # CCD pixels; it moves the camera sampling correspondence only, never
        # the panel grid.  ``None`` disables it and leaves the historical
        # sampling grid untouched, so the baseline stays bit-for-bit identical.
        if registration_shift_ccd_px is None:
            if learn_registration:
                raise ValueError("learn_registration needs registration_shift_ccd_px "
                                 "to start from; a shift of its own to refine")
            self.registration_shift = None
        else:
            dy, dx = (float(v) for v in registration_shift_ccd_px)
            if not np.isfinite([dy, dx]).all():
                raise ValueError("registration_shift_ccd_px must be finite")
            self.registration_shift = torch.nn.Parameter(
                torch.tensor([dy, dx], dtype=torch.float32, device=self._device),
                requires_grad=bool(learn_registration))
            if learn_registration:
                params += [{"params": [self.registration_shift], "lr": .05}]
        self.optimizer = torch.optim.Adam(params)
        self.history = []
        self.iteration_records = []
        # --- optional, default-off physical-model extensions (report 20261027 §4) ---
        self._slm_phase = None
        self._amp_residual = None
        self._monotonic_ccd = None
        if slm_phase_a != 0. or slm_phase_b != 0.:
            from ao_shaping.algorithm.signal_processing.slm_phase_nonlinearity import (
                slm_phase_nonlinearity,
            )
            self._slm_phase = lambda phi: slm_phase_nonlinearity(phi, slm_phase_a, slm_phase_b)
        if amplitude_residual_coeffs:
            from ao_shaping.algorithm.signal_processing.amplitude_residual import (
                AmplitudeResidual,
            )
            model = AmplitudeResidual(grid_size=size,
                                      n_modes=int(round((len(amplitude_residual_coeffs) ** 0.5 - 1) / 2)))
            model.set_coeffs(amplitude_residual_coeffs)
            panel_x = self.panel_x.detach().cpu().numpy().astype(np.float64)
            panel_y = self.panel_y.detach().cpu().numpy().astype(np.float64)
            self._amp_residual = torch.as_tensor(model.evaluate(panel_x, panel_y),
                                                 dtype=torch.float32, device=self._device)
        if monotonic_ccd_h:
            from ao_shaping.algorithm.signal_processing.monotonic_response import (
                MonotonicResponse, MonotonicResponseConfig,
            )
            cfg = MonotonicResponseConfig(n_bins=len(monotonic_ccd_h), v_max=float(monotonic_ccd_v_max))
            self._monotonic_ccd = MonotonicResponse(cfg, np.asarray(monotonic_ccd_h, dtype=np.float64))

    @property
    def registration_value(self) -> "tuple[float, float] | None":
        """Fitted/current global registration ``(dy, dx)`` in CCD px, or None when disabled.

        Serialisation-friendly read of :attr:`registration_shift`, which is a plain
        tuple when registration is off in the fit and a (possibly trainable) tensor
        when it is on; checkpoint writers use this so a fitted shift survives a
        ``frozen_parameters.json`` round-trip (issue #237).
        """
        value = getattr(self, "registration_shift", None)
        if value is None:
            return None
        if self.torch.is_tensor(value):
            value = [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
        return (float(value[0]), float(value[1]))

    def prediction(self):
        torch = self.torch
        width = self.log_width.exp()
        amplitude = gaussian_amplitude(torch, self.panel_x, self.panel_y, self.center, width,
                                       self.panel_mask)
        # Optional Gaussian illumination amplitude residual (report 20261027 §4 item 3).
        if getattr(self, "_amp_residual", None) is not None:
            amplitude = amplitude * torch.exp(self._amp_residual)
        if self.diagnostic_callback is not None:
            self.diagnostic_callback("input_amplitude", amplitude, None)
        # Optional global 2D registration (issue #237): the camera sampling
        # points move, the panel grid does not.  Disabled (the default) leaves
        # the grid untouched, so the baseline stays bit-for-bit identical.
        xy = registration_sample_grid(torch, self.xy, self.registration_shift)
        grids = camera_sample_grids(
            torch, xy, self.origin, self.inverse,
            subpixel_offsets(torch, self.pixel_samples, device=self.xy.device),
            fft_size=self.fft_size, pitch=self.pitch)
        predictions = []
        for start in range(0, len(self.phases), self.batch_size):
            command = self.phases[start:start+self.batch_size]
            # The SLM response acts on commands, not static optical aberrations.
            # Keep this order identical to FrozenDefocusForward.predict: command
            # polarity, then the SLM nonlinearity, then the fitted defocus.
            if self.phase_sign < 0:
                sl = slice(start, start + self.batch_size)
                focus = None if self.focus_phase is None else self.focus_phase[sl]
                carrier = None if self.carrier_phase is None else self.carrier_phase[sl]
                command = scoped_command_phase(
                    command, self.panel_x, self.panel_y, phase_sign=self.phase_sign,
                    scope=self.phase_sign_scope, focus_component=focus, carrier_component=carrier)
            if getattr(self, "_slm_phase", None) is not None:
                command = self._slm_phase(command)
            phase = defocus_phase(torch, command, self.d,
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

