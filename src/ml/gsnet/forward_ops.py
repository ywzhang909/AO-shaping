"""Pluggable forward operators for the deep-unrolled GS network.

The unrolled GS physics step in :class:`ml.gsnet.model.FFTLayer` was hard-coded
to one FFT convention.  This module extracts that step behind a small protocol
so the network can be driven by different forward physics (ideal FFT vs. the
calibrated bench-2f propagator of issue #165) without touching the CNN
conditioning contract.

Layering: ``ml/`` may import ``ao_shaping.utils.*`` only — the bench-2f
propagator lives in :mod:`ao_shaping.utils.wavefront.bench2f` and is referenced
exclusively through its public API.

⚠️ The two ideal conventions are a **real numerical difference**, not a
cosmetic one.  ``ifftshift`` before ``fft2`` multiplies the spectrum by
``(-1)^(k1+k2)`` (for an even grid), so the canonical far field differs from
the legacy one by a checkerboard sign (plus the ``norm="ortho"`` factor).
Never mix a forward step from one convention with a back step from the other;
a GS iteration is only self-consistent when both halves share one convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from ao_shaping.utils.wavefront.fraunhofer import focal_field

if TYPE_CHECKING:
    from ao_shaping.utils.wavefront.bench2f import Bench2FPropagator

__all__ = ["ForwardOperator", "IdealFFTOperator", "Bench2FOperator", "make_forward_operator"]


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


@runtime_checkable
class ForwardOperator(Protocol):
    """Physics contract for one unrolled GS iteration's forward/back pair."""

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

    Args:
        out_size: Far-field grid side (zero-pad target of the forward).
        convention: ``"legacy"`` (default) or ``"canonical"``.

    Raises:
        ValueError: On an unknown ``convention`` or a non-positive ``out_size``.
    """

    def __init__(self, out_size: int, *, convention: str = "legacy") -> None:
        if convention not in ("legacy", "canonical"):
            raise ValueError(
                f"convention must be 'legacy' or 'canonical', got {convention!r}"
            )
        if out_size < 1:
            raise ValueError(f"out_size must be >= 1, got {out_size}")
        self.out_size = int(out_size)
        self.convention = convention

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
        """Inverse-propagate a complex far field to the complex pupil field."""
        back = torch.fft.ifft2(torch.fft.ifftshift(far_field, dim=(-2, -1)))
        if self.convention == "legacy":
            return back
        scale = (far_field.shape[-2] * far_field.shape[-1]) / float(self.out_size**2)
        return back * scale

    def warm_start(self, target_amp: torch.Tensor) -> torch.Tensor:
        """Classical GS initial phase from the target amplitude."""
        if self.convention == "canonical":
            init_field = torch.fft.ifft2(
                torch.fft.ifftshift(target_amp, dim=(-2, -1)), norm="ortho"
            )
        else:
            init_field = torch.fft.ifft2(
                torch.fft.ifftshift(target_amp, dim=(-2, -1))
            )
        return torch.angle(init_field)


class Bench2FOperator:
    """The calibrated bench-2f forward, exposed through the ForwardOperator protocol.

    ``forward`` returns the propagator's complex far field: the **reduced**
    pupil field (propagator illumination × aperture × Zernike-reduced phase)
    pushed through the canonical centred zero-padded FFT, PEAK-NORMALISED —
    i.e. ``F / max|F|`` — so the network's far-plane amplitude constraint
    (``target_amp`` = ``sqrt(target_intensity)`` in peak units) keeps its
    semantics; the caller computes ``|F|^2`` for intensity.

    ``back_propagate`` delegates to the propagator's own canonical inverse
    (scaled by ``(region/S)**2``).  ``warm_start`` is the angle of the
    back-propagated target amplitude, centre-padded to the propagator's
    ``far_field_size`` first so the geometry matches.

    ``source_amp`` is accepted for protocol compatibility but deliberately NOT
    used to build the pupil field: the calibrated illumination × aperture
    profile is part of the calibration and lives inside the propagator.

    Args:
        propagator: A :class:`ao_shaping.utils.wavefront.bench2f.Bench2FPropagator`
            (duck-typed: needs ``geometry``, ``pupil_amplitude``, ``reduce_phase``,
            ``forward`` and ``back_propagate``).

    Raises:
        TypeError: If ``propagator`` lacks any required member.
    """

    def __init__(
        self,
        propagator: Bench2FPropagator,
        *,
        aberration: torch.Tensor | None = None,
    ) -> None:
        for member in ("geometry", "pupil_amplitude", "reduce_phase", "back_propagate"):
            if not hasattr(propagator, member):
                raise TypeError(
                    "propagator must expose "
                    f"{member!r} (Bench2FPropagator protocol)"
                )
        self._propagator = propagator
        self._aberration = aberration

    def forward(self, source_amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Complex, peak-normalised far field of the pupil field.

        ``source_amp`` is unused (see class docstring): the propagator's own
        illumination × aperture amplitude is the calibration.

        With no ``aberration`` the pupil phase is first projected onto the
        Zernike span (the calibrated model as a ~70-parameter model).  With an
        ``aberration`` the structure is the model-in-loop one instead — the
        **commanded** phase keeps its full per-pixel freedom and the bench's own
        aberration is added on top.
        """
        if self._aberration is None:
            total = self._propagator.reduce_phase(phase)
        else:
            total = self._propagator.reduce_phase(self._aberration) + phase
        amplitude = self._propagator.pupil_amplitude().to(total.device)
        pupil_field = amplitude * torch.exp(1j * total)
        far = focal_field(pupil_field, self._propagator.geometry.far_field_size)
        peak = far.abs().amax()
        if not bool(peak > 0):
            return far
        return far / peak

    def back_propagate(self, far_field: torch.Tensor) -> torch.Tensor:
        """(B,1,S,S) complex far field -> (B,1,region,region) complex pupil field."""
        return self._propagator.back_propagate(far_field)

    def warm_start(self, target_amp: torch.Tensor) -> torch.Tensor:
        """Initial pupil phase from the target amplitude via the propagator's back step."""
        padded = _centre_pad(target_amp, self._propagator.geometry.far_field_size)
        back = self._propagator.back_propagate(padded.to(torch.complex64))
        return torch.angle(back)


def make_forward_operator(spec: str | None, **kwargs) -> ForwardOperator | None:
    """Build a forward operator from a string spec.

    Args:
        spec: ``None`` -> ``None`` (the caller keeps its own hard-coded default);
            ``"ideal-legacy"`` / ``"ideal-canonical"`` -> :class:`IdealFFTOperator`
            (``out_size`` goes through ``kwargs``); ``"bench2f"`` ->
            :class:`Bench2FOperator` (``propagator`` goes through ``kwargs``).
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
    if spec == "bench2f":
        return Bench2FOperator(**kwargs)
    raise ValueError(
        "unknown forward-operator spec "
        f"{spec!r}; expected 'ideal-legacy', 'ideal-canonical', 'bench2f' or None"
    )
