"""Composite physics-informed loss for inverse beam shaping.

The supervised term alone is degenerate: the far field loses the absolute phase,
so a whole one-parameter family of phases (any global piston) maps to the same
image.  The loss therefore combines

    L = w_sup * L_sup + w_phy * L_phy + w_smooth * L_smooth

* ``L_sup`` -- L1 between the predicted unit phasor and the commanded phasor.
  Anchoring the global phase is exactly what breaks the piston degeneracy.
* ``L_phy`` -- L1 between the predicted forward image and the camera image, both
  peak-normalised so the term measures *shape* and stays blind to the piston.
* ``L_smooth`` -- optional total-variation penalty on the predicted phasor
  (default weight 0: the phase-smoothness penalty was measured harmful, see
  ``ShapingConfig.w_smooth``).

All three are computed from the same forward model the bank exposes; nothing is
re-implemented here except the loss composition itself.
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from ml.shaping.config import ShapingConfig
from ml.shaping.models import unit_phasor
from ml.shaping.physics import ForwardBank

__all__ = ["ShapingLoss", "ShapingLossTerms"]


class ShapingLossTerms(TypedDict):
    """Every term :meth:`ShapingLoss.__call__` returns, with its real type.

    The four loss terms are differentiable scalars and ``physics_excluded`` is a
    plain count. Declaring them separately rather than as
    ``dict[str, Tensor | int]`` is what lets a caller write
    ``losses["total"].backward()`` without a type checker objecting that the
    value might be an ``int`` -- and it documents, at the return type, that the
    count must never enter the loss.
    """

    total: Tensor
    sup: Tensor
    phy: Tensor
    smooth: Tensor
    physics_excluded: int

#: Below this peak a frame is treated as empty when peak-normalising; guards
#: against dividing an all-dark (excluded) frame by zero.
_PEAK_EPS = 1e-8


def _total_variation(phasor: Tensor) -> Tensor:
    """Return the total variation of a ``(B, C, H, W)`` phasor tensor.

    Prefers the canonical ``tv`` helper from :mod:`ml.zernike.losses` when it is
    importable, otherwise falls back to ``torchmetrics`` -- the same underlying
    implementation -- so no TV kernel is duplicated in this repo.

    Args:
        phasor: A ``(B, C, H, W)`` tensor.

    Returns:
        A scalar total-variation penalty.
    """
    try:  # pragma: no cover - depends on module layout, both branches tested
        from ml.zernike.losses import tv as _tv  # type: ignore[attr-defined]
    except ImportError:
        from torchmetrics.functional.image import total_variation as _tv
    return _tv(phasor)


def _peak_normalise(image: Tensor) -> Tensor:
    """Divide each frame by its own peak so only shape is compared.

    Args:
        image: A ``(B, 1, H, W)`` intensity tensor.

    Returns:
        The tensor with every frame scaled so its maximum is ``1`` (all-dark
        frames stay zero).
    """
    peak = image.amax(dim=(-2, -1), keepdim=True)
    return image / torch.clamp(peak, min=_PEAK_EPS)


class ShapingLoss:
    """Physics-informed objective for the inverse-shaping network.

    Attributes:
        config: The :class:`~ml.shaping.config.ShapingConfig` supplying the term
            weights.
        bank: The :class:`~ml.shaping.physics.ForwardBank` used for ``L_phy``.
    """

    def __init__(self, config: ShapingConfig, bank: ForwardBank) -> None:
        """Store the weight config and the forward bank.

        Args:
            config: Loss weights (``w_sup`` / ``w_phy`` / ``w_smooth``).
            bank: Forward bank producing the camera-space prediction.
        """
        self.config = config
        self.bank = bank

    def __call__(
        self, pred_raw: Tensor, batch: dict[str, Any]
    ) -> ShapingLossTerms:
        """Compute every loss term for one batch.

        Args:
            pred_raw: Raw network output ``(B, 2, g, g)`` (pre-projection).
            batch: A collated shaping batch with ``phase_cos`` / ``phase_sin``
                ``(B, 1, g, g)``, ``image`` ``(B, 1, g, g)`` and ``fov_px``.

        Returns:
            A dict with keys ``total``, ``sup``, ``phy``, ``smooth`` and
            ``physics_excluded`` (the count of rows dropped from ``L_phy``
            because their ``fov_px`` has no calibrated forward model). ``total``,
            ``sup``, ``phy`` and ``smooth`` are scalar tensors; the physics term
            degrades to a finite ``0`` when *every* row is excluded, never NaN.

        Note:
            Every returned tensor lives on ``pred_raw``'s device. The forward
            bank may sit on a different device (it defaults to CUDA whenever one
            is available), so ``L_phy`` is computed on the bank's device and
            brought back here -- the caller's batch need not match the bank.
        """
        pred_phasor = unit_phasor(pred_raw)
        # The batch arrives on CPU (the DataLoader yields CPU tensors) while the
        # network -- and therefore `pred_raw` -- lives on the resolved device, so
        # the ground-truth phasor has to follow it. Without this the first
        # `F.l1_loss` raises "Expected all tensors to be on the same device".
        gt_phasor = torch.cat([batch["phase_cos"], batch["phase_sin"]], dim=1).to(
            device=pred_phasor.device, dtype=pred_phasor.dtype
        )

        sup = F.l1_loss(pred_phasor, gt_phasor)

        pred_img, excluded = self.bank.forward(pred_phasor, batch["fov_px"])
        valid = ~excluded
        physics_excluded = int(excluded.sum())

        if bool(valid.any()):
            pred_norm = _peak_normalise(pred_img)
            target = batch["image"].to(device=pred_img.device, dtype=pred_img.dtype)
            target_norm = _peak_normalise(target)
            per_sample = (pred_norm - target_norm).abs().mean(dim=(1, 2, 3))
            weights = valid.to(per_sample.dtype)
            phy = (per_sample * weights).sum() / weights.sum().clamp(min=1.0)
            # The bank may live on a different device than the network (it
            # defaults to CUDA whenever one is available, while the batch here
            # follows the caller). Bring the term back to the network's device
            # so `total` stays a single-device sum with `sup` / `smooth`.
            phy = phy.to(device=pred_phasor.device)
        else:
            phy = torch.zeros((), dtype=pred_phasor.dtype, device=pred_phasor.device)
            logger.debug(
                "ShapingLoss: all {} rows excluded from L_phy", excluded.numel()
            )

        if float(self.config.w_smooth) == 0.0:
            smooth = torch.zeros(
                (), dtype=pred_phasor.dtype, device=pred_phasor.device
            )
        else:
            smooth = _total_variation(pred_phasor)

        total = (
            float(self.config.w_sup) * sup
            + float(self.config.w_phy) * phy
            + float(self.config.w_smooth) * smooth
        )
        return {
            "total": total,
            "sup": sup,
            "phy": phy,
            "smooth": smooth,
            "physics_excluded": physics_excluded,
        }
