"""Pluggable forward operators for the deep-unrolled GS network.

The unrolled GS physics step in :class:`ml.gsnet.model.FFTLayer` was hard-coded
to one FFT convention.  This module extracts that step behind a small protocol
so the network can be driven by different forward physics (ideal FFT vs. the
calibrated bench-2f propagator of issue #165) without touching the CNN
conditioning contract.

Layering: ``ml/`` may import ``ao_shaping.utils.*`` only — the bench-2f
2f simulation lives in :mod:`ao_shaping.drivers.sim.bench_twin`; the torch
mirror below is what the network differentiates through.

⚠️ The two ideal conventions are a **real numerical difference**, not a
cosmetic one.  ``ifftshift`` before ``fft2`` multiplies the spectrum by
``(-1)^(k1+k2)`` (for an even grid), so the canonical far field differs from
the legacy one by a checkerboard sign (plus the ``norm="ortho"`` factor).
Never mix a forward step from one convention with a back step from the other;
a GS iteration is only self-consistent when both halves share one convention.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import math

import numpy as np
import torch

from ao_shaping.utils.wavefront.fraunhofer import focal_field

__all__ = [
    "ForwardOperator",
    "IdealFFTOperator",
    "BenchTwinOperator",
    "make_forward_operator",
]


def make_forward_operator(spec: str | None, **kwargs: Any) -> ForwardOperator | None:
    """Build a forward operator from a string spec.

    Args:
        spec: ``None`` -> ``None`` (the caller keeps its own hard-coded default);
            ``"ideal-legacy"`` / ``"ideal-canonical"`` ->
            :class:`IdealFFTOperator` (``out_size`` goes through ``kwargs``);
            ``"bench-twin"`` -> :class:`BenchTwinOperator` (``twin`` goes
            through ``kwargs``).
        **kwargs: Forwarded to the selected class.

    Returns:
        The operator, or ``None`` for ``spec=None``.

    Raises:
        ValueError: On an unknown spec string.
    """
    if spec is None:
        return None
    if spec in ("ideal-legacy", "ideal-canonical"):
        return IdealFFTOperator(convention=spec.split("-", 1)[1], **kwargs)
    if spec == "bench-twin":
        return BenchTwinOperator(**kwargs)
    raise ValueError(f"unknown forward operator spec: {spec!r}")


def _centre_pad(field: torch.Tensor, out_size: int) -> torch.Tensor:
    """Centre-zero-pad the last two dims to ``out_size``.

    Same rule as :func:`ao_shaping.utils.wavefront.fraunhofer.focal_field`:
    a no-op when ``out_size <= n`` (never downsamples), and for an odd pad
    remainder the extra row/column goes to the right/bottom edge.

    Args:
        field: Tensor whose last two dims are the square grid to pad.
        out_size: Target side length.

    Returns:
        The padded tensor (the input itself when no padding is needed).
    """
    n = field.shape[-1]
    if out_size > n:
        half = (out_size - n) // 2
        field = torch.nn.functional.pad(
            field, (half, out_size - n - half, half, out_size - n - half)
        )
    return field


def _centre_crop(field: torch.Tensor, side: int) -> torch.Tensor:
    """Centre-crop the last two dims to ``side``.

    Exact no-op when the field is already that size (a slice, never a resize),
    so the same-grid path stays bit-identical.

    Args:
        field: Tensor whose last two dims are the grid to crop.
        side: Target side length.

    Returns:
        The cropped tensor.

    Raises:
        ValueError: If the field is smaller than ``side``.
    """
    n = field.shape[-1]
    if n == side:
        return field
    if n < side:
        raise ValueError(f"cannot crop a {n}-px field to {side} px")
    half = (n - side) // 2
    return field[..., half : half + side, half : half + side]


@runtime_checkable
class ForwardOperator(Protocol):
    """Physics contract for one unrolled GS iteration's forward/back pair.

    **Grid semantics** (this is the contract the network relies on):

    * ``forward`` takes the **pupil**-grid source amplitude and phase and
      returns the **far-field**-grid complex spectrum;
    * ``back_propagate`` takes a far-field-grid complex field and returns the
      **pupil**-grid complex field;
    * ``warm_start`` takes a far-field-grid target amplitude and returns the
      **pupil**-grid phase.

    The two grids are the same size in the historical same-grid regime and
    different in the oversampled regime, where the far field is the resolved
    one and the pupil is the coarse parameterisation.  An operator that knows
    its two grid sides exposes them as ``pupil_side`` / ``far_side``.
    """

    def forward(self, source_amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) source amplitude + (B,1,H,W) phase rad -> (B,1,S,S) far-field
        COMPLEX field (so the caller can take |.|^2 and the phase)."""
        ...

    def back_propagate(self, far_field: torch.Tensor) -> torch.Tensor:
        """(B,1,S,S) complex far field -> (B,1,H,W) complex pupil field."""
        ...

    def warm_start(self, target_amp: torch.Tensor) -> torch.Tensor:
        """(B,1,S,S) target amplitude -> (B,1,H,W) initial pupil phase rad."""
        ...


class IdealFFTOperator:
    """The textbook ideal-FFT far field, in TWO pinned conventions.

    ``convention="legacy"`` (DEFAULT) reproduces the CURRENT ``FFTLayer`` maths
    BIT-IDENTICALLY::

        forward        = fftshift(fft2(source_amp * exp(1j*phase)))   # no ifftshift, no ortho
        back_propagate = ifft2(ifftshift(far_field))
        warm_start     = angle(ifft2(ifftshift(target_amp)))

    ``convention="canonical"`` matches
    :func:`ao_shaping.utils.wavefront.fraunhofer.focal_field`::

        forward        = fftshift(fft2(ifftshift(source_amp * exp(1j*phase)), norm="ortho"))
        back_propagate = ifft2(ifftshift(far_field)) * (H*W)/(S*S)
        warm_start     = angle(ifft2(ifftshift(target_amp), norm="ortho"))

    The zero-pad to ``out_size`` is the same centred padding rule as
    ``focal_field``; the canonical forward literally *is* ``focal_field`` so the
    two cannot drift apart.

    The two conventions differ on the far field by a checkerboard sign
    ``(-1)^(k1+k2)`` (``ifftshift`` before ``fft2`` on an even grid) and the
    ``norm="ortho"`` scale — a real numerical difference, not a cosmetic one.

    In ``back_propagate``, ``(H, W)`` are the spatial dims of the far field
    actually received and ``(S, S)`` is this operator's ``out_size``: within the
    network the constrained far field lives on the model grid while the forward
    sampled the far field on the (larger, padded) ``out_size`` grid, and the
    ``norm="ortho"`` forward divides by ``sqrt(S*S)``, so the factor restores
    the pupil-plane scale.  The legacy convention applies no factor (today's
    un-scaled ``ifft2`` pair).

    **Pupil grid vs far-field grid.**  ``pupil_side`` is the side of the
    *pupil* grid the operator consumes and produces; ``out_size`` is the side
    of the *far-field* grid.  They are equal by default (the historical
    behaviour: the network's far field was a same-size FFT, so the
    diffraction-limited spot was sub-pixel and the pattern aliased).  Setting
    ``pupil_side < out_size`` is what lets the network see a **resolved** far
    field: the forward zero-pads the pupil to ``out_size`` and the backward
    step crops the centre back to ``pupil_side``.  The 2f bench needs that
    oversampling (8x and more) for its far field to be sampled at all.

    Args:
        out_size: Far-field grid side (zero-pad target of the forward).
        convention: ``"legacy"`` (default) or ``"canonical"``.
        pupil_side: Side of the pupil grid.  ``None`` (default) means
            ``out_size``, i.e. the historical same-grid behaviour.

    Raises:
        ValueError: On an unknown ``convention``, a non-positive ``out_size``,
            or a non-positive ``pupil_side``.
    """

    def __init__(
        self,
        out_size: int,
        *,
        convention: str = "legacy",
        pupil_side: int | None = None,
    ) -> None:
        if convention not in ("legacy", "canonical"):
            raise ValueError(
                f"convention must be 'legacy' or 'canonical', got {convention!r}"
            )
        if out_size < 1:
            raise ValueError(f"out_size must be >= 1, got {out_size}")
        if pupil_side is not None and pupil_side < 1:
            raise ValueError(f"pupil_side must be >= 1, got {pupil_side}")
        self.out_size = int(out_size)
        self.convention = convention
        self.pupil_side = int(out_size if pupil_side is None else pupil_side)

    @property
    def far_side(self) -> int:
        """Side of the far-field grid this operator produces."""
        return self.out_size

    def forward(self, source_amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Propagate the pupil field to the complex far field on ``out_size``."""
        pupil_field = source_amp * torch.exp(1j * phase)
        if self.convention == "canonical":
            # focal_field pads with the identical centred rule, then applies
            # fftshift(fft2(ifftshift(·, dim), norm="ortho"), dim).
            return focal_field(pupil_field, self.out_size)
        pupil_field = _centre_pad(pupil_field, self.out_size)
        return torch.fft.fftshift(torch.fft.fft2(pupil_field), dim=(-2, -1))

    def back_propagate(self, far_field: torch.Tensor) -> torch.Tensor:
        """Inverse-propagate a complex far field to the complex pupil field.

        The result is on the **pupil** grid (``pupil_side``), i.e. the centre
        window of the inverse transform when the far-field grid is larger.

        The two conventions are **exact inverse pairs** of their own forward:
        the legacy forward is an unnormalised ``fft2`` and its back is the
        unnormalised ``ifft2``; the canonical forward shifts the pupil before
        the transform (``ifftshift``) and normalises by the grid side
        (``norm="ortho"``), so its back has to shift back and undo the
        normalisation.  Only the phase of the result reaches the network, but an
        exact pair is what makes the round-trip testable — and it keeps the two
        operators in the same pupil frame.
        """
        if self.convention == "canonical":
            back = torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(far_field, dim=(-2, -1)),
                    dim=(-2, -1),
                    norm="ortho",
                ),
                dim=(-2, -1),
            )
        else:
            back = torch.fft.ifft2(torch.fft.ifftshift(far_field, dim=(-2, -1)))
        return _centre_crop(back, self.pupil_side)

    def warm_start(self, target_amp: torch.Tensor) -> torch.Tensor:
        """Classical GS initial phase from the target amplitude.

        The returned phase is on the **pupil** grid.
        """
        if self.convention == "canonical":
            init_field = torch.fft.ifft2(
                torch.fft.ifftshift(target_amp, dim=(-2, -1)), norm="ortho"
            )
        else:
            init_field = torch.fft.ifft2(
                torch.fft.ifftshift(target_amp, dim=(-2, -1))
            )
        return torch.angle(_centre_crop(init_field, self.pupil_side))


class BenchTwinOperator:
    """The canonical bench twin's chain, in torch, for use inside the network.

    :mod:`ao_shaping.drivers.sim.bench_twin` is the **canonical** 2f simulation
    and it is numpy (it is a simulation layer, not a differentiable model).  The
    network, however, has to differentiate through its own forward step.  This
    class mirrors the twin's chain in torch so the two speak the same physics:

    ``panel aberration -> LUT -> pupil field -> lens aberration -> centred
    zero-padded FFT -> |.|^2 -> centre crop -> peak normalise``

    Two deliberate differences from the twin, both documented here because they
    change what the network can learn:

    1. **The grey quantisation is not mirrored.**  The twin's LUT round trip goes
       through ``uint16`` grey levels, which has zero gradient almost
       everywhere — a torch mirror would contribute no gradient at all.  The
       continuous Gamma law is mirrored instead, so the LUT's *shape* (its
       non-linearity) is visible to the optimiser while the quantisation residual
       stays what it physically is: a staircase the reduced model cannot
       represent.
    2. **The focal-plane defocus and tilt are not mirrored.**  They act in the
       focal plane and are exactly what
       :mod:`ao_shaping.tools.slm.bench_twin_calibrate` fits; the network's
       reduced model leaves them to the closed loop's probe fit.

    Everything else — the panel Zernike + roughness, the lens Zernike, the
    illumination, the aperture, the padding factor and the centre crop — comes
    from the twin's own public API, so the two cannot drift apart.
    """

    def __init__(self, twin: Any) -> None:
        self._twin = twin
        region = int(twin.region)
        # Panel-side static aberration (Zernike + fixed-seed roughness): the twin
        # exposes it as a phase offset, so ask it for the zero-phase case.
        self._panel = torch.from_numpy(
            np.asarray(twin.apply_panel_aberration(np.zeros((region, region))), dtype=np.float64)
        ).to(torch.float32)
        # Lens-side wavefront error, applied on the pupil field.
        ones = np.ones((region, region), dtype=np.complex128)
        self._lens = torch.from_numpy(
            np.angle(np.asarray(twin.apply_lens_aberration(ones), dtype=np.complex128))
        ).to(torch.float32)
        self._amplitude = torch.from_numpy(
            np.asarray(twin.pupil_amplitude(), dtype=np.float64)
        ).to(torch.float32)
        self._levels = float(twin.errors.gray_levels)
        self._gamma = float(twin.errors.lut_gamma)
        self._phase_max = float(twin.errors.lut_phase_max_rad)
        self._table = twin.errors.lut_table
        self._far_size = int(twin.far_field_size)
        self._crop = int(twin.crop_side)
        self._region = region
        # 焦面（远场）两段：离焦（角谱传递函数）与倾斜（傅里叶移位）。
        self._p_fft = float(twin.p_fft)
        self._wavelength = float(twin.config.wavelength_m)
        self._focal_length = float(twin.config.focal_length_m)
        self._defocus = float(twin.errors.defocus_m)
        self._tilt_x = float(twin.errors.tilt_x_rad)
        self._tilt_y = float(twin.errors.tilt_y_rad)

    @property
    def pupil_side(self) -> int:
        """Side of the pupil grid."""
        return self._region

    @property
    def far_side(self) -> int:
        """Side of the far-field grid the operator produces (the twin's crop)."""
        return self._crop

    def _lut(self, phase: torch.Tensor) -> torch.Tensor:
        """The twin's LUT round trip, continuous (no grey quantisation)."""
        wrapped = torch.remainder(phase, 2.0 * torch.pi)
        gray = wrapped / (2.0 * torch.pi) * self._levels
        if self._table is not None:
            table = torch.from_numpy(
                np.asarray(self._table, dtype=np.float64)
            ).to(phase.dtype)
            index = gray.round().long().clamp(0, table.numel() - 1)
            return table[index]
        return self._phase_max * (gray / self._levels) ** self._gamma

    def _apply_defocus(self, field: torch.Tensor, dz: float | None = None) -> torch.Tensor:
        """相机离焦: 角谱传递函数作用在焦面复场上（孪生的同一写法）。

        ``H = exp(i·k·Δz·sqrt(1 - (λf_x)² - (λf_y)²))``，倏逝分量置 0。
        """
        if dz is None:
            dz = self._defocus
        if dz == 0.0:
            return field
        m = field.shape[-1]
        lam = self._wavelength
        k_wave = 2.0 * math.pi / lam
        freq = np.fft.fftfreq(m) / self._p_fft  # cycles/m
        fx = torch.from_numpy(freq).to(field.real.dtype)
        arg = 1.0 - (lam * fx)[None, :] ** 2 - (lam * fx)[:, None] ** 2
        propagating = arg > 0.0
        transfer = torch.zeros_like(arg, dtype=field.dtype)
        root = torch.sqrt(arg.clamp_min(0.0))
        transfer = torch.where(
            propagating,
            torch.exp(1j * k_wave * dz * root.to(field.dtype)),
            torch.zeros_like(root, dtype=field.dtype),
        )
        un = torch.fft.ifftshift(field, dim=(-2, -1))
        propagated = torch.fft.ifft2(
            torch.fft.fft2(un, norm="ortho") * transfer, norm="ortho"
        )
        return torch.fft.fftshift(propagated, dim=(-2, -1))

    def _apply_tilt(self, field: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
        """光路倾斜: 焦面图样平移 ``f · tilt``（傅里叶移位，孪生的同一写法）。"""
        dx = sign * self._focal_length * self._tilt_x
        dy = sign * self._focal_length * self._tilt_y
        if dx == 0.0 and dy == 0.0:
            return field
        m = field.shape[-1]
        freq = torch.from_numpy(np.fft.fftfreq(m) * m).to(field.real.dtype)
        shift = torch.exp(
            -2j
            * math.pi
            * (freq[:, None] * dy + freq[None, :] * dx)
            / (m * self._p_fft)
        ).to(field.dtype)
        un = torch.fft.ifftshift(field, dim=(-2, -1))
        shifted = torch.fft.ifft2(
            torch.fft.fft2(un, norm="ortho") * shift, norm="ortho"
        )
        return torch.fft.fftshift(shifted, dim=(-2, -1))

    def forward(self, source_amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Pupil phase -> complex far field on the crop grid.

        Args:
            source_amp: Unused — the calibrated illumination is part of the
                calibration and comes from the twin.
            phase: ``(B, 1, region, region)`` commanded phase in radians.

        Returns:
            ``(B, 1, crop, crop)`` complex far field, peak-normalised.
        """
        total = phase + self._panel.to(phase.device)
        actual = self._lut(total)
        amplitude = self._amplitude.to(phase.device)
        pupil = amplitude * torch.exp(1j * actual)
        pupil = pupil * torch.exp(1j * self._lens.to(phase.device))
        side = self._far_size
        buffer = torch.zeros(
            phase.shape[0], 1, side, side, dtype=pupil.dtype, device=phase.device
        )
        half = (side - self._region) // 2
        buffer[..., half : half + self._region, half : half + self._region] = pupil
        focal = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(buffer, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        focal = self._apply_defocus(focal)
        focal = self._apply_tilt(focal)
        crop = self._crop
        if crop < side:
            start = (side - crop) // 2
            focal = focal[..., start : start + crop, start : start + crop]
        peak = focal.abs().amax(dim=(-2, -1), keepdim=True)
        return focal / torch.where(peak > 0, peak, torch.ones_like(peak))

    def back_propagate(self, far_field: torch.Tensor) -> torch.Tensor:
        """Complex far field (crop grid) -> complex pupil field (pupil grid)."""
        side = self._far_size
        buffer = torch.zeros(
            far_field.shape[0],
            1,
            side,
            side,
            dtype=far_field.dtype,
            device=far_field.device,
        )
        crop = self._crop
        start = max(0, (side - crop) // 2)
        if start + crop <= side:
            buffer[..., start : start + crop, start : start + crop] = far_field
        else:
            keep = side - start
            buffer[..., start:side, start:side] = far_field[..., :keep, :keep]
        pupil = torch.fft.ifft2(torch.fft.ifftshift(buffer, dim=(-2, -1)), norm="ortho")
        # 逆变换把瞳面放回**角落**（正向是先 ifftshift 再 fft2），所以这里要
        # 再 fftshift 一次把它挪回中心，之后裁中心窗口才是瞳面。
        # 少了这一步，裁出来的是零 ⇒ 回代完全失效（GS 一步都不动）。
        pupil = torch.fft.fftshift(pupil, dim=(-2, -1))
        # 逆序撤销焦面两段：先反倾斜，再反离焦（两者在频域都是酉的）。
        pupil = self._apply_tilt(pupil, sign=-1.0)
        pupil = self._apply_defocus(pupil, -self._defocus)
        region = self._region
        half = (side - region) // 2
        return pupil[..., half : half + region, half : half + region]

    def warm_start(self, target_amp: torch.Tensor) -> torch.Tensor:
        """Classical GS initial phase from the target amplitude (pupil grid)."""
        return torch.angle(self.back_propagate(target_amp.to(torch.complex64)))
