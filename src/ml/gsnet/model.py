"""FourierGSNet: deep-unrolled Gerchberg-Saxton network for beam shaping phase retrieval.

Implementation of the algorithm from:

    S. Yan, M. J. Holenderski, N. Meratnia,
    "Efficient Gerchberg–Saxton algorithm deep unrolling for phase retrieval
    with a complex forward path",
    Advanced Photonics Nexus 5(2):026005 (2026). DOI 10.1117/1.APN.5.2.026005

Core idea: the classical Gerchberg–Saxton (GS) iteration is *unrolled* into a
fixed-depth neural network. Each layer performs the physical amplitude-constraint
exchange (source plane → FFT → far plane → replace amplitude with ``sqrt(target)``
→ IFFT → source plane → replace amplitude with ``sqrt(source)``) and then a
learned CNN (`ConditionUNet`) refines the phase, conditioned on the complex
far-field produced by the physics step.

Once trained, the network predicts an SLM phase mask from
``(source_intensity, target_intensity)`` in a *single forward pass* — no
iterative optimisation at inference time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ml.gsnet.forward_ops import ForwardOperator


class ConditionUNet(nn.Module):
    """Compact U-Net with FiLM-style conditioning for GS-phase refinement.

    The network learns a *residual phase update* given the current source-plane
    phase and the requested far-field (target) intensity. The physics state
    (back-propagated constraint phase and the complex far-field real/imag parts)
    is injected as a conditioning signal via feature-wise linear modulation
    (FiLM) at each decoder stage.

    Args:
        base_channels: Channel width of the first encoder stage. The encoder
            grows ``base → 2*base → 4*base``, the bottleneck is ``4*base``.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError(f"base_channels must be >= 4, got {base_channels}")

        c1, c2, c3 = base_channels, 2 * base_channels, 4 * base_channels

        # --- Encoder (2 input channels: source phase + target intensity) ---
        self.down1 = self._conv_block(2, c1)
        self.down2 = self._conv_block(c1, c2, stride=2)
        self.down3 = self._conv_block(c2, c3, stride=2)
        self.bottleneck = self._conv_block(c3, c3)

        # --- Decoder (skip connections + FiLM conditioning) ---
        self.up2 = self._up_block(c3, c2)          # 2x upsample + merge w/ down2
        self.cond2 = FiLMLayer(3, c2)              # condition at scale 2
        self.up1 = self._up_block(c2, c1)          # 2x upsample + merge w/ down1
        self.cond1 = FiLMLayer(3, c1)              # condition at scale 1
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _conv_block(
        in_ch: int, out_ch: int, stride: int = 1,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Refine the phase map.

        Args:
            x: Input tensor ``(B, 2, H, W)`` — concat(source_phase, target_intensity).
            condition: Conditioning tensor ``(B, 3, H, W)`` —
                concat(pupil_phase, far_real, far_imag) from the physics step.

        Returns:
            Phase update ``(B, 1, H, W)`` (residual to add to the GS phase).
        """
        x1 = self.down1(x)                                  # (B, c1, H, W)
        x2 = self.down2(x1)                                 # (B, c2, H/2, W/2)
        x3 = self.down3(x2)                                 # (B, c3, H/4, W/4)
        xb = self.bottleneck(x3)                            # (B, c3, H/4, W/4)

        d2 = self.up2(xb)                                   # (B, c2, H/2, W/2)
        d2 = d2 + x2[:, : d2.shape[1]]                       # skip-connect (crop)
        d2 = self.cond2(d2, condition)                      # FiLM condition

        d1 = self.up1(d2)                                   # (B, c1, H, W)
        d1 = d1 + x1[:, : d1.shape[1]]                       # skip-connect
        d1 = self.cond1(d1, condition)                      # FiLM condition

        return self.out_conv(d1)                            # (B, 1, H, W)


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation from a (smaller) conditioning tensor.

    The condition is encoded to a per-channel scale/shift pair and applied as
    ``out = gamma * feat + beta``. The condition is spatially *global* (pooled),
    which keeps the injection cost flat and is well suited to conditioning on a
    physics-state tensor.
    """

    def __init__(self, cond_channels: int, feat_channels: int) -> None:
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.Conv2d(cond_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gamma_beta = nn.Linear(feat_channels, 2 * feat_channels)

    def forward(self, feat: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        cond_code = self.cond_net(condition).flatten(1)          # (B, feat)
        gamma, beta = self.gamma_beta(cond_code).chunk(2, dim=1)  # (B, feat) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


def _match_grid(tensor: torch.Tensor, side: int) -> torch.Tensor:
    """Bilinearly resample ``tensor`` to a ``side x side`` grid.

    Exact no-op when the tensor already has that side, so the same-grid path
    keeps its bit-identical arithmetic.

    Args:
        tensor: Tensor whose last two dims are the grid to resample.
        side: Target side length.

    Returns:
        The tensor on the target grid.
    """
    if tensor.shape[-1] == side:
        return tensor
    return torch.nn.functional.interpolate(
        tensor, size=(side, side), mode="bilinear", align_corners=False
    )


class FFTLayer(nn.Module):
    """One unrolled Gerchberg–Saxton iteration with a learned phase update.

    ⚠️ **The forward/backward pair here is deliberately FFT, not angular-spectrum
    propagation.** The canonical ASM lives in
    ``ao_shaping/algorithm/signal_processing/angular_spectrum.py`` and is a valid
    propagator — but it is *not* a drop-in for the Fourier-plane FFT on a 2f
    bench. Measured on the bench geometry (12 mm aperture, ``f = 0.125 m``,
    ``lambda = 532 nm``): the FFT peaks at the array centre with
    ``max|F| = 4057``, whereas ASM at ``z = f`` gives ``max|out| = 1.0066`` — the
    *input* modulus — and peaks at the **pupil** position. ASM returns the
    propagated beam; a 2f bench's Fourier plane is the **lens's** transform. A
    genuine Fraunhofer pattern via ASM needs ``z >> a**2/lambda = 271 m``.
    Boundary locked by
    ``tests/ao_shaping/algorithm/test_angular_spectrum.py::TestNotAFraunhoferSubstitute``.
    Note also that the forward and backward half-array shifts of an ASM
    propagator would have to cancel exactly for GS to stay self-consistent,
    which is why the shift convention is pinned rather than incidental.

    Each layer performs:

    1. **Forward physics**: build the source-plane field ``source_amp * e^{jφ}``
       and FFT to the far (Fourier) plane.
    2. **Far-plane amplitude constraint**: replace the Fourier magnitude with
       ``sqrt(target_intensity)``.
    3. **Backward physics**: IFFT back to the source plane.
    4. **Near-plane amplitude constraint**: replace the source magnitude with
       ``sqrt(source_intensity)`` (implicit in the next layer's field build).
    5. **Learned refinement**: a `ConditionUNet` maps
       ``(current phase, target intensity)`` → phase residual, conditioned on
       the physics state ``(constrained back-propagated phase, far real, far imag)``.

    The physics half is pluggable: with ``forward_op=None`` (default) the layer
    runs the legacy ``fftshift(fft2(·))`` / ``ifft2(ifftshift(·))`` chain
    **bit-identically to the historical implementation**; the shift convention
    above stays pinned.  With a `~ml.gsnet.forward_ops.ForwardOperator` given,
    the same amplitude-exchange runs through ``op.forward`` /
    ``op.back_propagate``.  See :mod:`ml.gsnet.forward_ops` for the pinned
    "legacy" vs "canonical" conventions — they differ by a checkerboard sign
    ``(-1)^(k1+k2)`` on the far field (``ifftshift`` before ``fft2``), so
    choosing one is a real numerical decision, not a cosmetic one.

    Args:
        base_channels: CNN width (see `ConditionUNet`).
        forward_op: Optional pluggable forward operator. ``None`` keeps the
            default legacy FFT path.
    """

    def __init__(
        self, base_channels: int = 32,
        forward_op: ForwardOperator | None = None,
    ) -> None:
        super().__init__()
        self.phase_net = ConditionUNet(base_channels)
        self._forward_op = forward_op

    def forward(
        self,
        phase: torch.Tensor,
        source_amp: torch.Tensor,
        target_amp: torch.Tensor,
    ) -> torch.Tensor:
        """Run one unrolled GS step.

        Args:
            phase: Current source-plane phase ``(B, 1, H, W)`` radians, on the
                **pupil** grid.
            source_amp: Source-plane amplitude ``(B, 1, H, W)``, pupil grid.
            target_amp: Far-field (target) amplitude ``(B, 1, S, S)``, on the
                **far-field** grid.  ``S`` may differ from ``H`` when the
                operator oversamples the far field; the historical same-grid
                regime has ``S == H``.

        Returns:
            Updated source-plane phase ``(B, 1, H, W)`` radians.
        """
        # ---- 1. Forward propagation to far field ----
        if self._forward_op is None:
            pupil_field = source_amp * torch.exp(1j * phase)
            far_field = torch.fft.fftshift(torch.fft.fft2(pupil_field), dim=(-2, -1))
            far_phase = torch.angle(far_field)
            torch.abs(far_field) ** 2

            # ---- 2. Far-plane amplitude constraint ----
            far_field_c = target_amp * torch.exp(1j * far_phase)

            # ---- 3. Back-propagate to source plane ----
            pupil_back = torch.fft.ifft2(
                torch.fft.ifftshift(far_field_c, dim=(-2, -1))
            )
        else:
            far_field = self._forward_op.forward(source_amp, phase)
            far_phase = torch.angle(far_field)
            far_field_c = target_amp * torch.exp(1j * far_phase)
            pupil_back = self._forward_op.back_propagate(far_field_c)
        pupil_phase = torch.angle(pupil_back)

        # ---- 5. Learned phase refinement ----
        # The CNN works on the PUPIL grid: the far-field tensors are the hint,
        # the phase is the answer, and the answer has to live on the pupil grid.
        # When the two grids coincide this is the historical expression, bit for
        # bit; when they differ the far-field parts are bilinearly resampled
        # down to the pupil grid.
        pupil_side = phase.shape[-1]
        cnn_input = torch.cat(
            [phase, _match_grid(target_amp**2, pupil_side)], dim=1
        )                                                       # (B, 2, H, W)
        condition = torch.cat(
            [
                pupil_phase,
                _match_grid(far_field_c.real, pupil_side),
                _match_grid(far_field_c.imag, pupil_side),
            ],
            dim=1,
        )                                                        # (B, 3, H, W)

        delta = self.phase_net(cnn_input, condition)
        return pupil_phase + delta


class FourierGSNet(nn.Module):
    """Deep-unrolled Gerchberg–Saxton network for beam-shaping phase retrieval.

    Stack of :math:`N` `FFTLayer` blocks. The initial phase is taken from the
    back-propagated target amplitude (a strong classical GS initialisation),
    then refined through the unrolled layers.

    Args:
        num_layers: Number of unrolled GS iterations (10 in the paper).
        base_channels: CNN width of each `FFTLayer`.
        shared_weights: Share one `FFTLayer` across all unrolled steps.
        forward_op: Optional pluggable forward operator (see
            :mod:`ml.gsnet.forward_ops`), forwarded to every `FFTLayer` and
            used for the GS warm start. ``None`` keeps the default legacy FFT
            path (bit-identical to the historical implementation).
    """

    def __init__(
        self, num_layers: int = 10, base_channels: int = 32,
        shared_weights: bool = False,
        forward_op: ForwardOperator | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.shared_weights = shared_weights
        self._forward_op = forward_op
        self.layers = nn.ModuleList(
            [
                FFTLayer(base_channels, forward_op)
                for _ in range(1 if shared_weights else num_layers)
            ]
        )

    def forward(
        self,
        source_intensity: torch.Tensor,
        target_intensity: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the SLM phase mask.

        Args:
            source_intensity: Source-plane intensity ``(B, 1, H, W)``,
                non-negative, on the **pupil** grid.
            target_intensity: Desired far-field intensity, non-negative.  On the
                **far-field** grid, which in the oversampled regime is larger
                than the pupil grid.

        Returns:
            Phase mask ``(B, 1, H, W)`` radians, unwrapped-free (mod 2π).
        """
        if source_intensity.ndim != 4 or target_intensity.ndim != 4:
            raise ValueError(
                "Expected 4D inputs (B, 1, H, W), got "
                f"{tuple(source_intensity.shape)} and {tuple(target_intensity.shape)}"
            )
        if self._forward_op is None:
            # Legacy path: one grid for both, exactly as before.
            if source_intensity.shape != target_intensity.shape:
                raise ValueError(
                    "source_intensity and target_intensity shapes must match: "
                    f"{tuple(source_intensity.shape)} vs {tuple(target_intensity.shape)}"
                )
        else:
            # The operator owns the two grid sides; check them when it publishes
            # them, so a mismatched training set fails before a device is touched.
            pupil_side = getattr(self._forward_op, "pupil_side", None)
            far_side = getattr(self._forward_op, "far_side", None)
            if pupil_side is not None and source_intensity.shape[-1] != pupil_side:
                raise ValueError(
                    f"source_intensity side {source_intensity.shape[-1]} does not "
                    f"match the operator's pupil grid {pupil_side}"
                )
            if far_side is not None and target_intensity.shape[-1] != far_side:
                raise ValueError(
                    f"target_intensity side {target_intensity.shape[-1]} does not "
                    f"match the operator's far-field grid {far_side}"
                )

        source_amp = torch.sqrt(source_intensity.clamp_min(0.0) + 1e-12)
        target_amp = torch.sqrt(target_intensity.clamp_min(0.0) + 1e-12)

        # Initial phase: back-propagated target (classical GS warm start).
        if self._forward_op is None:
            init_field = torch.fft.ifft2(
                torch.fft.ifftshift(target_amp, dim=(-2, -1))
            )
            phase = torch.angle(init_field)
        else:
            phase = self._forward_op.warm_start(target_amp)

        for step in range(self.num_layers):
            layer = self.layers[0] if self.shared_weights else self.layers[step]
            phase = layer(phase, source_amp, target_amp)

        return phase


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def angular_difference(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shortest signed angular distance ``pred - target`` wrapped to (-π, π]."""
    return torch.remainder(pred - target + math.pi, 2 * math.pi) - math.pi
