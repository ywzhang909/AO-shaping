"""Simulation-based backpropagation tests for the differentiable beam shaper.

This suite proves the far-field physics model (FFT) and the autograd path
are correct on synthetic targets — no hardware, CPU only. It complements
``test_differentiable_beam_optimizer.py`` (the class-API suite) and
deliberately does not repeat its validation / seed / early-stop cases.
``DifferentiableBeamOptimizer`` is the sole public API within the algorithm
package; the full loop is driven by the optimizer-layer function
``optimize_beam_shaping`` (the legacy one-shot ``differentiable_beam_optimize``
function was removed).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ao_shaping.algorithm import (
    DifferentiableBeamOptimizer,
    differentiable_far_field,
    far_field_intensity,
)
from ao_shaping.optimizer import optimize_beam_shaping
from ao_shaping.utils.image.targets import crop_resize_to_grid


def _gaussian_target(shape: tuple[int, int], sigma: float | None = None) -> np.ndarray:
    """A smooth 2D gaussian target intensity (peak-normalized to 1.0).

    Args:
        shape: ``(H, W)`` grid size.
        sigma: Gaussian sigma in pixels. Defaults to ``max(H, W) / 16``,
            matching the existing ``_gaussian_target`` convention.

    Returns:
        Float32 ``(H, W)`` array with peak value 1.0 at the grid center.
    """
    h, w = shape
    y, x = np.ogrid[-h // 2 : h // 2, -w // 2 : w // 2]
    if sigma is None:
        sigma = max(h, w) / 16.0
    r2 = x**2 + y**2
    g = np.exp(-r2 / (2.0 * sigma**2))
    return (g / g.max()).astype(np.float32)


def _square_target(shape: tuple[int, int], box_size: int) -> np.ndarray:
    """A uniform square target intensity (peak-normalized to 1.0).

    Args:
        shape: ``(H, W)`` grid size.
        box_size: Side length of the uniform square in pixels.

    Returns:
        Float32 ``(H, W)`` array with a centered ``box_size`` x ``box_size``
        block of ones and zeros elsewhere.
    """
    h, w = shape
    t = np.zeros(shape, dtype=np.float32)
    y0 = (h - box_size) // 2
    x0 = (w - box_size) // 2
    t[y0 : y0 + box_size, x0 : x0 + box_size] = 1.0
    return t


def _box_blur(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Box-blur a 2D array with a ``k`` x ``k`` kernel (edge-padded)."""
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.zeros_like(x)
    for i in range(k):
        for j in range(k):
            out += xp[i : i + x.shape[0], j : j + x.shape[1]]
    return out / (k * k)


def _random_target(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """A non-negative, smooth-ish random target intensity (peak-normalized).

    Generated as the square of twice-box-blurred Gaussian noise, so the
    target is non-negative with smooth low-frequency structure (no
    single-pixel spikes).

    Args:
        shape: ``(H, W)`` grid size.
        seed: RNG seed for reproducibility.

    Returns:
        Float32 ``(H, W)`` array with peak value 1.0.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape)
    smooth = _box_blur(_box_blur(noise, 5), 5)
    t = smooth**2
    return (t / t.max()).astype(np.float32)


class _MockSLM:
    """Records applied radian phases; exposes only ``write_phase``.

    This is the mock path of ``optimize_beam_shaping``'s ``_display_phase``
    (no ``create_phase_from_array`` attribute), so phases are recorded in
    radians exactly as the optimizer sends them.
    """

    def __init__(self, resolution: tuple[int, int]) -> None:
        self.resolution = resolution
        self.applied_phases: list[np.ndarray] = []
        self.current_phase: np.ndarray | None = None

    def write_phase(self, phase_rad: np.ndarray) -> None:
        phase = np.asarray(phase_rad, dtype=np.float32)
        assert phase.shape == self.resolution
        self.applied_phases.append(phase.copy())
        self.current_phase = phase.copy()


class _MockCCD:
    """Returns the model far-field intensity of the last applied SLM phase.

    The noiseless mock makes the measured image exactly the model's own
    forward pass, so the hardware loop's measured loss tracks the simulated
    loss (up to float32 round-trip noise) and the loop's per-iteration
    behavior is exercised without any real hardware.
    """

    def __init__(self, source_amplitude: np.ndarray, slm: _MockSLM) -> None:
        self.source_amplitude = source_amplitude
        self.slm = slm
        self.capture_count = 0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        self.capture_count += 1
        assert self.slm.current_phase is not None, "no phase applied before capture"
        phase = torch.from_numpy(self.slm.current_phase)
        i_far = far_field_intensity(self.source_amplitude, phase)
        return i_far.detach().cpu().numpy().astype(np.float32)


def _zero_order_peak(intensity: np.ndarray) -> tuple[int, int]:
    """Argmax of the 0-order (central) region of a far-field intensity map.

    The beam is centered, so the 0-order region is the central half of the
    fftshifted grid. Restricting the search here ignores residual speckle
    peaks in the outer region that a phase-only optimizer cannot fully
    suppress.

    Args:
        intensity: 2D far-field intensity map.

    Returns:
        ``(row, col)`` of the brightest pixel inside the central half.
    """
    h, w = intensity.shape
    region = intensity[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    r, c = np.unravel_index(np.argmax(region), region.shape)
    return (r + h // 4, c + w // 4)


class TestTargetBuilders:
    """The synthetic target builders must be peak-normalized and shaped."""

    def test_gaussian_target_peak_normalized_and_centered(self) -> None:
        """The gaussian target peaks at 1.0 at the grid center."""
        t = _gaussian_target((32, 32))
        assert t.shape == (32, 32)
        assert t.dtype == np.float32
        assert t.max() == 1.0
        assert t.min() >= 0.0
        assert np.unravel_index(np.argmax(t), t.shape) == (16, 16)

    def test_gaussian_target_symmetric(self) -> None:
        """A symmetric gaussian target is invariant under transpose."""
        t = _gaussian_target((32, 32))
        assert np.allclose(t, t.T, atol=1e-6)

    def test_square_target_uniform_centered_box(self) -> None:
        """The square target is a uniform centered box of ones."""
        t = _square_target((32, 32), 8)
        assert t.shape == (32, 32)
        assert t.max() == 1.0
        assert t.min() == 0.0
        assert np.all(t[12:20, 12:20] == 1.0)
        assert np.all(t[:12, :] == 0.0)
        assert np.all(t[20:, :] == 0.0)

    def test_random_target_nonnegative_peak_normalized(self) -> None:
        """The random target is non-negative, peak-normalized, and smooth."""
        t = _random_target((32, 32), seed=0)
        assert t.shape == (32, 32)
        assert t.dtype == np.float32
        assert t.max() == 1.0
        assert np.all(t >= 0.0)
        # Smooth-ish: more than one pixel above half-max (not a single spike).
        assert (t > 0.5).sum() > 1


class TestFarFieldModelMatchesNumpyFFT:
    """The differentiable FFT model must match the reference NumPy FFT."""

    def test_complex_field_matches_numpy_fft(self) -> None:
        """``differentiable_far_field`` equals ``fftshift(fft2(A * e^{i*phi}))``.

        Uses a random phase and an arbitrary non-uniform amplitude on a
        16x16 grid. Tolerance 1e-3 covers float32 FFT rounding: field values
        here are O(10), so the float32 error is ~1e-5 (measured 2.6e-6).
        """
        n = 16
        rng = np.random.default_rng(0)
        amp = (0.3 + 0.7 * rng.random((n, n))).astype(np.float32)
        phase_np = rng.uniform(0.0, 2.0 * np.pi, (n, n)).astype(np.float32)
        phase = torch.from_numpy(phase_np)
        e_far = differentiable_far_field(amp, phase)
        ref = np.fft.fftshift(
            np.fft.fft2(amp.astype(np.float64) * np.exp(1j * phase_np.astype(np.float64)))
        )
        assert np.allclose(e_far.detach().cpu().numpy(), ref, atol=1e-3)

    def test_intensity_matches_numpy_fft(self) -> None:
        """``far_field_intensity`` equals ``|fftshift(fft2(A * e^{i*phi}))|^2``.

        Same setup as the complex-field test; intensity values are O(100),
        so the float32 error is ~1e-4 (measured 9.5e-5), within atol 1e-3.
        """
        n = 16
        rng = np.random.default_rng(1)
        amp = (0.3 + 0.7 * rng.random((n, n))).astype(np.float32)
        phase_np = rng.uniform(0.0, 2.0 * np.pi, (n, n)).astype(np.float32)
        phase = torch.from_numpy(phase_np)
        i_far = far_field_intensity(amp, phase)
        ref = np.fft.fftshift(
            np.fft.fft2(amp.astype(np.float64) * np.exp(1j * phase_np.astype(np.float64)))
        )
        assert np.allclose(i_far.detach().cpu().numpy(), np.abs(ref) ** 2, atol=1e-3)


class TestDifferentiablePath:
    """The autograd path must be sound: grad_fn, shape, dtype, gradient."""

    def test_intensity_carries_grad_fn(self) -> None:
        """With ``requires_grad=True`` the intensity output links to phase."""
        amp = np.ones((16, 16), dtype=np.float32)
        phase = torch.zeros(16, 16, dtype=torch.float32, requires_grad=True)
        i_far = far_field_intensity(amp, phase)
        assert i_far.grad_fn is not None

    def test_output_shape_and_dtype(self) -> None:
        """Outputs keep the input shape and float32/complex64 dtype."""
        amp = np.ones((24, 20), dtype=np.float32)
        phase = torch.randn(24, 20, dtype=torch.float32)
        e_far = differentiable_far_field(amp, phase)
        i_far = far_field_intensity(amp, phase)
        assert e_far.shape == (24, 20)
        assert i_far.shape == (24, 20)
        assert e_far.dtype == torch.complex64
        assert i_far.dtype == torch.float32

    def test_gradient_flows_to_phase(self) -> None:
        """Backward through an MSE loss must populate ``phase.grad``.

        The loss is the MSE against a non-uniform gaussian target (a plain
        ``mean(i_far)`` would have ~zero gradient because total energy is
        conserved by the phase-only FFT).
        """
        amp = np.ones((16, 16), dtype=np.float32)
        phase = torch.randn(16, 16, dtype=torch.float32, requires_grad=True)
        target = torch.from_numpy(_gaussian_target((16, 16)))
        i_far = far_field_intensity(amp, phase)
        loss = torch.mean((i_far - target) ** 2)
        loss.backward()
        assert phase.grad is not None
        assert phase.grad.shape == phase.shape
        assert torch.isfinite(phase.grad).all()
        assert float(phase.grad.abs().sum()) > 0.0


class TestOptimizerLossDecreases:
    """For every target type the MSE loss must decrease over updates."""

    def test_loss_decreases_gaussian(self) -> None:
        """200 Adam steps on a gaussian target lower the MSE loss."""
        target = _gaussian_target((32, 32))
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=200, log_every=0
        )
        loss_history = [r["loss"] for r in res_list.history]
        assert loss_history[-1] < loss_history[0]

    def test_loss_decreases_square(self) -> None:
        """80 Adam steps on a square target lower the MSE loss."""
        target = _square_target((32, 32), 8)
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=80, log_every=0
        )
        loss_history = [r["loss"] for r in res_list.history]
        assert loss_history[-1] < loss_history[0]

    def test_loss_decreases_random(self) -> None:
        """80 Adam steps on a random target lower the MSE loss."""
        target = _random_target((32, 32), seed=0)
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=80, log_every=0
        )
        loss_history = [r["loss"] for r in res_list.history]
        assert loss_history[-1] < loss_history[0]


class TestPhaseShapesTheBeam:
    """The learned phase must have the target shape and actually shape it."""

    def test_phase_shape_matches_target(self) -> None:
        """The optimized phase has the same shape as the target grid."""
        target = _gaussian_target((32, 32))
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=50, log_every=0
        )
        best_iter, _ = res_list.get_best_iter()
        assert best_iter["best_phase"].shape == target.shape

    def test_narrow_gaussian_peak_lands_near_center(self) -> None:
        """A narrow gaussian target concentrates the far-field peak at center.

        The target's peak bin is the grid center (the beam is centered, so
        the 0-order region is the center of the fftshifted grid). After 300
        steps the far-field intensity's peak inside the 0-order region must
        sit within radius 2 of that bin, and the learned phase must be
        non-trivial (a flat phase would leave the energy spread out).
        """
        target = _gaussian_target((32, 32), sigma=1.0)
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=300, log_every=0
        )
        best_iter, _ = res_list.get_best_iter()
        phase = best_iter["best_phase"]
        amp = np.ones_like(target)
        i_far = far_field_intensity(amp, torch.from_numpy(phase))
        i_far_np = i_far.detach().cpu().numpy()
        peak = _zero_order_peak(i_far_np)
        center = (target.shape[0] // 2, target.shape[1] // 2)
        dist = np.hypot(peak[0] - center[0], peak[1] - center[1])
        assert dist <= 2.0
        assert np.std(phase) > 0.1


class TestConvergenceOnSymmetricGaussian:
    """On a symmetric gaussian the optimizer must converge to a low loss."""

    def test_final_loss_below_threshold_and_tail_monotonic(self) -> None:
        """Enough steps on a symmetric gaussian give a low final loss and a
        monotonically decreasing tail.

        Threshold 0.05: the MSE between the peak-normalized far-field
        intensity and the peak-normalized gaussian target (measured 0.0044
        at 300 steps). Epsilon 1e-6 absorbs float32 jitter in the Adam tail.
        """
        target = _gaussian_target((32, 32))
        res_list = optimize_beam_shaping(
            target_intensity=target, device="cpu", seed=0, epochs=300, log_every=0
        )
        loss_history = [r["loss"] for r in res_list.history]
        assert res_list.last["loss"] < 0.05
        tail = loss_history[-50:]
        eps = 1e-6
        assert all(tail[i + 1] <= tail[i] + eps for i in range(len(tail) - 1))


class TestHardwareClosedLoopSim:
    """Hardware-closed-loop mode with mock devices (no hardware needed).

    The mock CCD returns the model far-field of the last applied SLM phase,
    so the measured loss is the model's own forward-pass loss and must
    decrease exactly like the simulation-mode loss.
    """

    @staticmethod
    def _make_devices(shape: tuple[int, int]) -> tuple[np.ndarray, _MockSLM, _MockCCD]:
        amp = np.ones(shape, dtype=np.float32)
        slm = _MockSLM(shape)
        ccd = _MockCCD(amp, slm)
        return amp, slm, ccd

    def test_measured_loss_decreases_on_noiseless_mock(self) -> None:
        """200 hardware-loop steps on a gaussian target lower the measured
        loss (the mock CCD returns the model far field of the applied phase)."""
        target = _gaussian_target((32, 32))
        amp, slm, ccd = self._make_devices((32, 32))
        res_list = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=amp,
            lr=0.01,
            device="cpu",
            seed=0,
            epochs=200,
            log_every=0,
            progress=False,
            slm=slm,
            ccd=ccd,
            wait_time_s=0.0,
            discard_count=1,
        )
        loss_history = [r["loss"] for r in res_list.history]
        assert loss_history[-1] < loss_history[0]

    def test_measured_loss_decreases_square_target(self) -> None:
        """80 hardware-loop steps on a square target lower the measured loss."""
        target = _square_target((32, 32), 8)
        amp, slm, ccd = self._make_devices((32, 32))
        res_list = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=amp,
            lr=0.01,
            device="cpu",
            seed=0,
            epochs=80,
            log_every=0,
            progress=False,
            slm=slm,
            ccd=ccd,
            wait_time_s=0.0,
            discard_count=1,
        )
        loss_history = [r["loss"] for r in res_list.history]
        assert loss_history[-1] < loss_history[0]

    def test_measured_image_matches_model_far_field_of_applied_phase(self) -> None:
        """Each record's measured image is the model far field of the phase
        that was applied during that capture (apply -> measure ordering)."""
        target = _gaussian_target((32, 32))
        amp, slm, ccd = self._make_devices((32, 32))
        res_list = optimize_beam_shaping(
            target_intensity=target,
            source_amplitude=amp,
            lr=0.01,
            device="cpu",
            seed=0,
            epochs=5,
            log_every=0,
            progress=False,
            slm=slm,
            ccd=ccd,
            wait_time_s=0.0,
            discard_count=1,
        )
        for k, record in enumerate(res_list.history):
            applied = slm.applied_phases[k]
            expected = far_field_intensity(amp, torch.from_numpy(applied))
            expected_np = expected.detach().cpu().numpy().astype(np.float32)
            # Replicate the optimizer's crop_resize_to_grid: with the default
            # full-random init the far-field peak is off-center, so the raw
            # far field differs from the resized measured image.
            expected_resized = crop_resize_to_grid(
                expected_np, grid_h=target.shape[0], grid_w=target.shape[1]
            )
            assert np.allclose(record["measured_image"], expected_resized, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])