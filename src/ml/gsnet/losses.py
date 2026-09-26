"""Differentiable losses and metrics for the FourierGSNet beam-shaping model.

Single source of truth for the phase / far-field objectives shared by training
(:mod:`ml.gsnet.train`), evaluation (:mod:`ml.gsnet.evaluate`) and the hardware
``c_head`` fine-tuning entry point (``FourierGSNet.py``).

Every method is a ``staticmethod`` returning a scalar; nothing here owns state,
so the class is a namespace, not an instance-bearing object. The FFT physics
(``fft2`` → ``fftshift`` over the last two dims), the per-image unit-sum
normalization and the ``1e-12`` epsilon constants are byte-identical to the
original inline implementations.
"""

from __future__ import annotations

import torch

from ml.gsnet.model import angular_difference


class ShapingLosses:
    """Namespace of scalar phase / far-field objectives for FourierGSNet."""

    @staticmethod
    def angular_difference(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Shortest signed angular distance ``pred - target`` wrapped to (-π, π].

        Thin delegate to :func:`ml.gsnet.model.angular_difference` so the wrapped
        difference has exactly one implementation in the package.
        """
        return angular_difference(pred, target)

    @staticmethod
    def circular_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean squared shortest angular distance between two phase maps."""
        return torch.mean(ShapingLosses.angular_difference(pred, target) ** 2)

    @staticmethod
    def phase_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean absolute shortest angular distance between two phase maps."""
        return float(ShapingLosses.angular_difference(pred, target).abs().mean())

    @staticmethod
    def intensity_mse(
        source_amp: torch.Tensor,
        phase: torch.Tensor,
        target_intensity: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between normalized far-field intensity and normalized target.

        Applies the same physics as the model (FFT + fftshift), normalizes both
        distributions to unit sum, and returns the per-image mean squared
        difference. Scale-invariant so absolute beam power does not matter.

        Note: once the physics unrolling already matches the target (far_rmse
        ~1e-3), this term saturates near zero and carries no training signal —
        use :meth:`shaping_loss` as the shaping objective instead.
        """
        field = source_amp * torch.exp(1j * phase)
        far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
        intensity = torch.abs(far) ** 2
        pred_n = intensity / (intensity.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        tgt_n = target_intensity / (
            target_intensity.sum(dim=(-2, -1), keepdim=True) + 1e-12
        )
        return torch.mean((pred_n - tgt_n) ** 2)

    @staticmethod
    def shaping_loss(
        source_amp: torch.Tensor,
        phase: torch.Tensor,
        target_intensity: torch.Tensor,
    ) -> torch.Tensor:
        """1 - mean Pearson correlation between predicted and target far-field.

        Unlike :meth:`intensity_mse` — which saturates near zero the moment the
        physics amplitude exchange already matches the target — returns a loss on
        a O(1) scale (0 = perfect correlation) that stays differentiable well past
        the MSE saturation point. Scale-invariant: the predicted intensity has
        arbitrary absolute scale, so both distributions are centered and
        normalized per image before correlation.
        """
        field = source_amp * torch.exp(1j * phase)
        far = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
        intensity = torch.abs(far) ** 2
        pred = intensity.flatten(1)
        tgt = target_intensity.flatten(1)
        p = pred - pred.mean(dim=1, keepdim=True)
        t = tgt - tgt.mean(dim=1, keepdim=True)
        denom = torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-12
        corr = (p * t).sum(dim=1) / denom
        return torch.mean(1.0 - corr)

    @staticmethod
    def coefficient_l1(c_hat: torch.Tensor, c_gt: torch.Tensor) -> torch.Tensor:
        """Mean absolute error between predicted and ground-truth Zernike coefficients.

        Used to fine-tune the hardware ``c_head`` regressor (``FourierGSNet.py``),
        which predicts the upstream aberration coefficients in radians.
        """
        return (c_hat - c_gt).abs().mean()
