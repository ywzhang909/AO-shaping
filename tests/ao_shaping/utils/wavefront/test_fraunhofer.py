"""Bitwise equivalence tests for the canonical Fraunhofer propagator.

The two historical call sites shipped byte-identical copies of the
centre-zero-pad + centred-FFT chain. These tests pin the extracted helper to
locally-inlined copies of BOTH original chains using ``torch.equal`` (bitwise,
never ``allclose``), so any behaviour drift fails loudly.
"""

from __future__ import annotations

import torch

from ao_shaping.utils.wavefront.fraunhofer import focal_field, focal_intensity


def _old_optimizer_chain(field: torch.Tensor, out_size: int) -> torch.Tensor:
    """Inlined pre-refactor ``_far_field_intensity`` tail (2-D input)."""
    pad = out_size
    if pad > field.shape[0]:
        half = (pad - field.shape[0]) // 2
        field = torch.nn.functional.pad(
            field.unsqueeze(0),
            (half, pad - field.shape[0] - half, half, pad - field.shape[1] - half),
        ).squeeze(0)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field), norm="ortho")
    )
    return spectrum.real**2 + spectrum.imag**2


def _old_models_chain(field: torch.Tensor, out_size: int) -> torch.Tensor:
    """Inlined pre-refactor ``_propagate`` body (4-D input)."""
    n = field.shape[-1]
    m = out_size
    if m > n:
        start = (m - n) // 2
        padded = torch.zeros(
            field.shape[:-2] + (m, m), dtype=field.dtype, device=field.device
        )
        padded[..., start : start + n, start : start + n] = field
        field = padded
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def test_matches_optimizer_chain_bitwise() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 8, dtype=torch.float64)
    for out_size in (8, 16, 80):  # n, 2n, 10n
        assert torch.equal(
            focal_intensity(x, out_size), _old_optimizer_chain(x, out_size)
        )


def test_matches_models_chain_bitwise() -> None:
    torch.manual_seed(1)
    x4 = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    for factor in (1, 2, 4, 10):
        out_size = 8 * factor
        assert torch.equal(focal_field(x4, out_size), _old_models_chain(x4, out_size))


def test_padding_odd_remainder_goes_right_and_bottom() -> None:
    torch.manual_seed(2)
    n = 8
    out_size = n + 3  # odd pad remainder
    d = out_size - n
    assert d % 2 == 1
    left, right = d // 2, d - d // 2
    # symmetric within one pixel (odd d means exactly one extra row/column)
    assert abs(left - right) <= 1
    assert right == left + 1  # the odd remainder lands right/bottom

    x = torch.randn(n, n, dtype=torch.float64)
    # Reference pad: first `half` rows/cols exactly zero, content at [left:left+n]
    padded = torch.nn.functional.pad(x, (left, right, left, right))
    assert padded.shape == (out_size, out_size)
    assert torch.equal(padded[..., :left, :], torch.zeros_like(padded[..., :left, :]))
    assert torch.equal(padded[..., :, :left], torch.zeros_like(padded[..., :, :left]))
    assert torch.equal(padded[..., left : left + n, left : left + n], x)
    # The helper's output is bitwise the FFT of that reference pad
    expect = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(padded, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    assert torch.equal(focal_field(x, out_size), expect)


def test_noop_when_out_size_equals_input() -> None:
    torch.manual_seed(3)
    n = 8
    x = torch.randn(n, n, dtype=torch.float64)
    # out_size == n means no padding: output is exactly the plain centred FFT
    expect = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    assert torch.equal(focal_field(x, n), expect)
    # ...and padding genuinely changes the sampling
    assert not torch.equal(focal_field(x, n), focal_field(x, n + 1))


def test_intensity_is_nonnegative_and_finite() -> None:
    torch.manual_seed(4)
    x = (
        torch.randn(16, 16, dtype=torch.float64)
        + 1j * torch.randn(16, 16, dtype=torch.float64)
    )
    intensity = focal_intensity(x, 32)
    assert bool(torch.all(intensity >= 0))
    assert bool(torch.all(torch.isfinite(intensity)))


def test_works_for_2d_and_4d_consistently() -> None:
    torch.manual_seed(5)
    g = 8
    x = torch.randn(g, g, dtype=torch.float64)
    x4 = x.unsqueeze(0).unsqueeze(0)
    out_size = 4 * g
    assert torch.equal(focal_field(x, out_size), focal_field(x4, out_size)[0, 0])
    assert torch.equal(focal_intensity(x, out_size), focal_intensity(x4, out_size)[0, 0])
