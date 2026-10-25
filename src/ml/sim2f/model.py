"""U-Net mapping a requested camera spot to the pupil phase that produces it.

The network predicts one thing only: the phase that turns the **aberration-free**
bench's flat spot into the requested target.  It never sees an aberration, so it
does not have to be retrained for a new optical path -- the bench state is
removed downstream by a Zernike estimate from probes plus a closed-loop refit.

Input and output are deliberately asymmetric:

* input ``(B, 2, ccd_px, ccd_px)`` -- channel 0 is the target frame (unit sum,
  as cached), channel 1 is the **static PSF context**: the bench's flat-phase
  frame, peak-normalised, computed once and stored as a buffer.  Without it the
  network cannot tell which angular scale the target lives on -- the same pixel
  pattern means a different phase on a bench with `fov_in_psf = 30` than on one
  with `fov_in_psf = 120` -- so it would have to infer the diffraction scale
  from the target's own blur, which is exactly the ambiguity the teacher's
  labels do not share.  The context frame must come from the **same**
  ``Sim2f`` configuration that produces the physics leg's frames: it carries
  the chain's own zero-order background and PSF shape, so a context from a
  different (e.g. idealised) chain would tell the network the wrong scale and
  the wrong pedestal.
* output ``(B, 2, grid, grid)`` unit phasor ``(cos, sin)``.  A wrapped angle has
  a ``+/- pi`` branch cut that a CNN cannot represent smoothly; the phasor form
  turns the global-phase degeneracy into a **rotation** rather than a
  discontinuity, which is also why :func:`ml.sim2f.losses.piston_free_phasor_l1`
  can remove it in closed form.

Normalisation choice: **GroupNorm**, not BatchNorm.  Batch sizes here are 4-8
and single-sample evaluation is the normal case (the evaluator runs one spot at
a time), where BatchNorm's batch statistics are both noisy and train/eval
inconsistent.

The aperture
------------
The phasor is set to ``(1, 0)`` outside the pupil disc so the field there is
undisturbed (a unit phasor, i.e. no phase, exactly what a dark panel does).
The disc radius comes from :attr:`Sim2fConfig.pupil_radius_px` -- the same
single property the simulator uses to build its aperture -- so the two cannot
disagree about the radius.  Only boundary cell inclusion could differ by half a
pixel, and those cells are unilluminated, so nothing physical depends on it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from ml.sim2f.config import Sim2fConfig, TrainConfig
from ml.sim2f._device import _resolve_device
from ml.sim2f.losses import as_frame, forward_image

__all__ = ["Sim2fUNet", "build_unet", "network_input"]


def _norm_groups(channels: int) -> int:
    """Largest group count ``<= 8`` that divides ``channels``.

    GroupNorm needs ``channels % groups == 0``; widths in this package are
    powers of two times ``width``, so 8 almost always divides, and the fallback
    keeps any valid ``width`` (e.g. 6) working rather than raising at build
    time.

    Args:
        channels: Channel count of the block.

    Returns:
        A divisor of ``channels`` no greater than 8.
    """
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1  # pragma: no cover - 1 always divides


class _DoubleConv(nn.Module):
    """``Conv3x3 - GroupNorm - SiLU`` twice; the U-Net's only block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block.

        Args:
            x: ``(B, in_channels, H, W)``.

        Returns:
            ``(B, out_channels, H, W)``.
        """
        return self.block(x)


def network_input(target: Tensor, context: Tensor) -> Tensor:
    """Assemble the network input ``[target, static PSF context]``.

    Args:
        target: ``(B, 1, ccd_px, ccd_px)`` (or ``(B, ccd_px, ccd_px)``) target
            frame, unit sum as cached.
        context: ``(1, 1, ccd_px, ccd_px)`` or ``(ccd_px, ccd_px)`` ideal
            flat-phase frame; broadcast over the batch.

    Returns:
        ``(B, 2, ccd_px, ccd_px)`` ready for :meth:`Sim2fUNet.forward`.

    Raises:
        ValueError: If the two frames disagree on the camera grid.
    """
    frames = as_frame(target)
    psf = as_frame(context).to(dtype=frames.dtype, device=frames.device)
    if psf.shape[-2:] != frames.shape[-2:]:
        raise ValueError(
            f"context grid {tuple(psf.shape[-2:])} != target grid "
            f"{tuple(frames.shape[-2:])}"
        )
    if psf.shape[0] == 1 and frames.shape[0] != 1:
        psf = psf.expand(frames.shape[0], -1, -1, -1)
    return torch.cat([frames, psf], dim=1)


class Sim2fUNet(nn.Module):
    """Encoder/decoder with skip connections, camera grid in, pupil phasor out.

    The decoder runs on the **camera** grid (that is where the input lives) and a
    final bilinear interpolation adapts its output to the pupil grid, so the
    common case ``ccd_px != grid`` needs no separate architecture.  The ratio
    must be an integer either way (see :meth:`__init__`): with ``grid = 2 *
    ccd_px`` the upsampling is an exact integer resample, and with ``ccd_px = 2
    * grid`` the phase is an exact 2x2 bin average, whereas an arbitrary
    non-integer ratio would resample the phase with a kernel whose width has no
    relation to the pupil sampling.

    Attributes:
        grid: Pupil grid side of the emitted phasor.
        ccd_px: Camera grid side of the input.
        depth: Number of pooling levels.
        pupil_mask: ``(1, 1, grid, grid)`` buffer, 1 inside the aperture.
        flat_psf: ``(1, 1, ccd_px, ccd_px)`` buffer, the peak-normalised
            flat-phase frame, or an empty tensor when no context was supplied.
    """

    def __init__(
        self,
        width: int = 32,
        depth: int = 5,
        in_channels: int = 2,
        *,
        grid: int | None = None,
        ccd_px: int | None = None,
        pupil_fill: float = 0.75,
        context: Tensor | None = None,
    ) -> None:
        """Build the network.

        Args:
            width: Channels in the first encoder stage; doubles per level.
            depth: Number of pooling levels.  ``ccd_px`` must be divisible by
                ``2 ** depth``.
            in_channels: Input channels (2 = target + PSF context).
            grid: Pupil grid side.  Defaults to ``ccd_px``.
            ccd_px: Camera grid side.  Defaults to the context's spatial size,
                then to ``grid``.
            pupil_fill: Aperture diameter as a fraction of the pupil grid, i.e.
                :attr:`Sim2fConfig.pupil_fill`.
            context: Ideal flat-phase frame.  Peak-normalised and stored as a
                non-persistent buffer; the checkpoint carries it explicitly, so
                it is not duplicated in ``state_dict``.

        Raises:
            ValueError: If the grids cannot be resolved, ``ccd_px`` is not
                divisible by ``2 ** depth``, or neither grid is an integer
                multiple of the other.
        """
        super().__init__()
        if ccd_px is None and context is not None:
            ccd_px = int(as_frame(context).shape[-1])
        if grid is None:
            grid = ccd_px
        if ccd_px is None:
            ccd_px = grid
        if grid is None or ccd_px is None:
            raise ValueError("Sim2fUNet needs grid and/or ccd_px (or a context tensor)")
        grid = int(grid)
        ccd_px = int(ccd_px)
        stride = 2 ** int(depth)
        if ccd_px % stride:
            raise ValueError(
                f"ccd_px ({ccd_px}) must be divisible by 2 ** depth ({stride}) for "
                f"{depth} pooling levels"
            )
        if grid % ccd_px and ccd_px % grid:
            raise ValueError(
                f"grid ({grid}) and ccd_px ({ccd_px}) must be integer multiples of "
                "each other so the camera/pupil resample stays exact"
            )
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")

        self.width = int(width)
        self.depth = int(depth)
        self.grid = grid
        self.ccd_px = ccd_px
        self.in_channels = int(in_channels)

        encoder_channels = [self.width * 2**level for level in range(self.depth)]
        bottleneck_channels = self.width * 2**self.depth

        self.encoder = nn.ModuleList()
        channels = self.in_channels
        for out_channels in encoder_channels:
            self.encoder.append(_DoubleConv(channels, out_channels))
            channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _DoubleConv(channels, bottleneck_channels)

        self.decoder = nn.ModuleList()
        channels = bottleneck_channels
        for out_channels in reversed(encoder_channels):
            self.decoder.append(_DoubleConv(channels + out_channels, out_channels))
            channels = out_channels

        # Two channels: real and imaginary parts of the pupil phasor.
        self.head = nn.Conv2d(channels, 2, kernel_size=1)

        self.register_buffer("pupil_mask", self._build_mask(label=grid, fill=pupil_fill))
        if context is None:
            self.register_buffer("flat_psf", torch.zeros(0), persistent=False)
        else:
            self.register_buffer(
                "flat_psf", self._normalise(context).detach(), persistent=False
            )

    @staticmethod
    def _build_mask(*, label: int, fill: float) -> Tensor:
        """Build the pupil disc used to park the phase outside the aperture.

        Args:
            label: Pupil grid side.
            fill: Aperture diameter as a fraction of ``grid``.

        Returns:
            ``(1, 1, label, label)`` with 1 inside the aperture.
        """
        radius = 0.5 * float(fill) * float(label)
        coordinates = torch.arange(label, dtype=torch.float32) + 0.5 - 0.5 * float(label)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        return (torch.hypot(yy, xx) <= radius).to(torch.float32)[None, None]

    @staticmethod
    def _normalise(context: Tensor) -> Tensor:
        """Peak-normalise a frame, keeping its batch/channel layout.

        Args:
            context: Any frame accepted by :func:`~ml.sim2f.losses.as_frame`.

        Returns:
            ``(1, 1, ccd_px, ccd_px)`` with maximum 1 (or all-zero if the frame
            was empty -- an empty context is then visibly wrong rather than
            NaN).
        """
        frame = as_frame(context)
        if frame.shape[0] != 1:
            frame = frame[:1]
        peak = frame.amax().clamp_min(1e-8)
        return frame / peak

    def n_parameters(self) -> int:
        """Number of trainable parameters (quoted in the run report).

        Returns:
            ``sum(p.numel() for p in self.parameters())``.
        """
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: Tensor) -> Tensor:
        """Map a target frame to the aberration-free-bench pupil phasor.

        Args:
            x: ``(B, in_channels, ccd_px, ccd_px)``, channel 0 the target frame
                and channel 1 the static PSF context (see
                :func:`network_input`).

        Returns:
            ``(B, 2, grid, grid)`` unit phasor.  Modulus is 1 everywhere: inside
            the pupil because of the explicit normalisation, outside because
            the phasor is parked at ``(1, 0)``.

        Raises:
            ValueError: If the input is not 4-D ``(B, in_channels, ccd_px,
                ccd_px)``.
        """
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Sim2fUNet expects (B, {self.in_channels}, {self.ccd_px}, "
                f"{self.ccd_px}), got shape {tuple(x.shape)}"
            )
        skips: list[Tensor] = []
        features = x
        for block in self.encoder:
            features = block(features)
            skips.append(features)
            features = self.pool(features)

        features = self.bottleneck(features)
        for block, skip in zip(self.decoder, reversed(skips), strict=True):
            features = F.interpolate(
                features, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = block(torch.cat([features, skip], dim=1))

        raw = self.head(features)
        if raw.shape[-2:] != (self.grid, self.grid):
            raw = F.interpolate(
                raw, size=(self.grid, self.grid), mode="bilinear", align_corners=False
            )

        # Unit phasor, then the aperture: outside it the field must be
        # undisturbed, so the phasor is exactly (1, 0) there.
        modulus = raw.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        raw = raw / modulus
        mask = self.pupil_mask
        real = raw[:, 0:1] * mask + (1.0 - mask)
        imag = raw[:, 1:2] * mask
        return torch.cat([real, imag], dim=1)


def build_unet(
    cfg: TrainConfig,
    sim_cfg: Sim2fConfig,
    *,
    device: str | torch.device = "auto",
    context: Tensor | None = None,
) -> Sim2fUNet:
    """Build the network for one bench, optionally computing the PSF context.

    Args:
        cfg: Architectural and training settings (``width``, ``depth``).
        sim_cfg: Bench geometry, giving ``grid``, ``ccd_px`` and ``pupil_fill``.
        device: ``"auto"``, ``"cuda"``, ``"cpu"`` or a device.
        context: Ideal flat-phase frame.  When ``None`` it is computed from the
            bench: ``Sim2f(sim_cfg).forward(zeros)`` -- the same chain and the
            same config the physics leg uses, so the two can never sit on
            different optics.

    Returns:
        The network, moved to ``device``.

    Raises:
        ValueError: If ``sim_cfg.ccd_px`` is not divisible by ``2 ** cfg.depth``.
    """
    stride = 2 ** int(cfg.depth)
    if sim_cfg.ccd_px % stride:
        raise ValueError(
            f"ccd_px ({sim_cfg.ccd_px}) must be divisible by 2 ** depth ({stride}); "
            f"lower --depth or change --ccd-px"
        )
    if context is None:
        # Imported here: `optics` pulls in torch on a CUDA-capable machine and
        # this module must stay importable without building a bench.
        from ml.sim2f.optics import Sim2f

        bench = Sim2f(sim_cfg, device="cpu")
        with torch.no_grad():
            zeros = torch.zeros(1, sim_cfg.grid, sim_cfg.grid)
            context = forward_image(bench, zeros).detach()
        logger.debug(
            "computed flat-phase context from the bench: ccd_px={} peak={:.3g}",
            sim_cfg.ccd_px,
            float(context.amax()),
        )
    model = Sim2fUNet(
        width=cfg.width,
        depth=cfg.depth,
        in_channels=2,
        grid=sim_cfg.grid,
        ccd_px=sim_cfg.ccd_px,
        pupil_fill=sim_cfg.pupil_fill,
        context=context,
    )
    logger.debug(
        "built Sim2fUNet width={} depth={} grid={} ccd_px={} params={}",
        cfg.width,
        cfg.depth,
        sim_cfg.grid,
        sim_cfg.ccd_px,
        model.n_parameters(),
    )
    return model.to(_resolve_device(device))
