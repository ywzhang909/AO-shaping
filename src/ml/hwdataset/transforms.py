"""Pure array-to-array transforms for the hardware debug dataset.

This module is the transform half of :mod:`ml.hwdataset`. It is deliberately
free of any index, dataset, file or hardware knowledge: every function maps
arrays to arrays, so it can be unit-tested in isolation and reused by the
on-disk cache builder, the :class:`~ml.hwdataset.dataset.HwPhaseImageDataset`
and any offline analysis script.

Layering note
-------------
It reuses the canonical helpers in :mod:`ml.gsnet_debug.offline`
(``pupil_phase_to_grid``, ``farfield_to_grid``,
``reconstruct_pupil_phase_rad``) rather than reimplementing them: this repo has
been bitten repeatedly by duplicated Zernike/crop math drifting between copies.
Importing ``ao_shaping.runners`` from ``src/ml`` is a deliberate, reviewed
choice -- ``gsnet_offline`` imports only ``ao_shaping.utils.*``, and
``ao_shaping/runners/__init__.py`` uses a PEP-562 lazy ``__getattr__``, so there
is no import cycle.

Why a COHERENT block mean
-------------------------
The SLM phases recorded in ``data/debug`` are **wrapped to [0, 2*pi)**: the
observed maximum over a whole 1200x1920 panel is 6.2605, below
``2*pi = 6.2832``. An arithmetic mean of such a block is therefore biased --
on a test pattern it returns ~pi in *every* block and destroys the structure
entirely. The correct area downsample of a phase field averages the complex
phasor ``exp(1j*phi)`` and then takes the angle:

    z_block = block_mean(exp(1j * phi))
    phase   = arctan2(Im, Re)          # local mean phase
    contrast= hypot(Re, Im)            # per-cell coherence in [0, 1]

The public output is ``(cos_grid, sin_grid)`` rather than
``(angle, contrast)`` on purpose: ``arctan2`` has a branch cut at +/-pi that
produces a discontinuity in the input a convolutional network has to learn
around, whereas ``(Re, Im)`` is continuous. ``hypot(cos, sin)`` recovers the
coherence if a caller wants it.

Bench constants (see ``src/ao_shaping/drivers/AGENTS.md``)
----------------------------------------------------------
* Santec SLM-200 panel 1920x1200, 10-bit, @1064nm, ``2*pi = 993`` gray levels.
* Illuminated beam disc radius ~450 px centred at panel ``(x=960, y=600)``.
* Daheng MER2-507 CCD 2592x1944 ``uint8``, 2.2 um pixel.
* 2f Fourier bench: SLM front focus -> f=125 mm lens -> CCD back focus, so the
  camera frame is a spatial-frequency map of the pupil.

``DEFAULT_PANEL_RADIUS`` is 500 rather than the illumination radius 450 because
the commanded phase in the corpus reaches further than the beam: the widest
observed structure is radius ~479 px (``slm_pib_*``, whose sidecars declare
``zernike_radius=480``), which is 69% of the corpus by record count. 500 px is
the smallest round half-width containing everything measured, and the resulting
1000x1000 ROI sits inside the 1200x1920 panel (rows 100..1099, cols 460..1459).

Why ``/ 255`` is right for EVERY family
--------------------------------------
``_img`` is not uniformly ``uint8``: the corpus holds ``uint8`` (250x248, 250x250,
320x320, 400x400, 1944x2592), ``float32`` (64x64, 1944x2592) and ``float64``
(250x250) frames. The *dtype* is not the scale -- measured per-frame maxima
across the whole corpus are 255 (uint8), 239.6 (float32) and 100.9 (float64), all
on one 0-255 detector scale, and nothing anywhere exceeds 255. So dividing by
255 is the correct uniform normalisation and the ``[0, 1]`` clip never actually
fires. Treating the float frames as already-normalised (a natural mistake) would
compress them by up to 255x, and peak-normalising instead would throw away the
absolute brightness that encodes the exposure time.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from ml.gsnet_debug.offline import (
    farfield_to_grid,
    reconstruct_pupil_phase_rad,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "DEFAULT_PANEL_CENTER",
    "DEFAULT_PANEL_RADIUS",
    "DEFAULT_SLM_MAX_GRAY",
    "coherent_block_mean",
    "crop_panel_roi",
    "farfield_frame_to_grid",
    "freeform_grid_to_panel",
    "grayscale_to_phase_rad",
    "phase_to_grid",
    "zernike_coeffs_to_panel",
]

#: Grayscale level corresponding to a full 2*pi phase cycle on SLM#1 @1064nm.
#: The device queries this at runtime as ``_max_gray``; it is **not** recorded in
#: the debug dumps, so it is an explicit parameter wherever it is needed.
DEFAULT_SLM_MAX_GRAY: int = 993

#: Illuminated-beam centre on the panel, as ``(x, y)`` = ``(col, row)``.
DEFAULT_PANEL_CENTER: tuple[int, int] = (960, 600)

#: ROI half-width in panel pixels. See the module docstring for why 500 and not
#: the illumination radius 450.
DEFAULT_PANEL_RADIUS: float = 500.0

_TWO_PI = 2.0 * math.pi
#: Largest float32 strictly below 2*pi. ``2*pi - 1e-10`` rounds *up* to 2*pi in
#: float32 (the spacing near 6.28 is ~4.8e-7), which would let a full-scale gray
#: escape the documented ``[0, 2*pi)`` half-open range.
_TWO_PI_F32_MAX = float(np.nextafter(np.float32(_TWO_PI), np.float32(0.0)))


def grayscale_to_phase_rad(
    gray: NDArray[np.integer] | NDArray[np.floating],
    max_gray: int = DEFAULT_SLM_MAX_GRAY,
) -> NDArray[np.float32]:
    """Convert an SLM grayscale image to radians in ``[0, 2*pi)``.

    Exact inverse of the driver's single wrap point
    ``Santec.create_phase_from_array`` (radians -> ``mod(2*pi) / 2*pi * max_gray``).
    ``max_gray`` is a calibration constant that the device reports at runtime and
    that the debug dumps do **not** record, so it stays an explicit parameter
    rather than being inferred from the data.

    Args:
        gray: Grayscale image, integer or float, any shape.
        max_gray: Grayscale level corresponding to 2*pi. Must be >= 1.

    Returns:
        ``float32`` phase in ``[0, 2*pi)``, same shape as ``gray``. Non-finite
        input pixels are treated as 0 (a flat pixel).

    Raises:
        ValueError: If ``max_gray < 1``.
    """
    if int(max_gray) < 1:
        raise ValueError(f"max_gray must be >= 1, got {max_gray}")

    # One fused pass, not four. Measured on a real 1200x1920 uint16 panel:
    # asarray + nan_to_num + multiply + minimum separately costs 11.0 ms, and
    # uint16 cannot hold a non-finite value, so the finiteness guard is only
    # needed for the float-input case.
    scale = np.float32(_TWO_PI) / np.float32(max_gray)
    phase = np.multiply(gray, scale, dtype=np.float32)
    if not np.all(np.isfinite(phase)):
        np.nan_to_num(phase, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Gray above max_gray saturates just below a full cycle instead of wrapping
    # to 0: a wrap would fabricate a phase discontinuity the measurement never had.
    return np.minimum(phase, np.float32(_TWO_PI_F32_MAX))


def crop_panel_roi(
    phase: NDArray[np.floating],
    center: tuple[int, int],
    radius: float,
) -> NDArray[np.float32]:
    """Crop a square ROI of half-width ``radius`` centred on panel ``(x, y)``.

    The window is CLAMPED to the panel and never padded, so an ROI that hangs off
    an edge simply comes back smaller; a ``radius`` larger than the panel yields
    the whole panel. ``radius <= 0`` disables cropping.

    Cropping before the block mean is essential, not an optimisation: the beam
    is a ~500 px disc on a 1200x1920 panel, so a whole-panel downsample to 64
    cells would squeeze the entire pupil into about 5x7 of the 4096 output cells
    and dilute the rest with the flat-field background.

    Args:
        phase: Phase map ``(H, W)`` in radians.
        center: Panel centre as ``(x, y)`` = ``(col, row)``.
        radius: ROI half-width in pixels; ``<= 0`` means "no crop".

    Returns:
        C-contiguous ``float32`` array of shape ``(<= 2*radius, <= 2*radius)``,
        exactly ``(2*radius, 2*radius)`` when the ROI fits inside the panel.

    Raises:
        ValueError: If ``phase`` is not 2-D.
    """
    array = np.asarray(phase, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"phase must be a 2D array, got shape {array.shape}")

    if float(radius) <= 0.0:
        return np.ascontiguousarray(array)

    height, width = array.shape
    center_x, center_y = (int(center[0]), int(center[1]))
    half = int(math.floor(float(radius)))

    x0 = max(0, center_x - half)
    x1 = min(width, center_x + half)
    y0 = max(0, center_y - half)
    y1 = min(height, center_y + half)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"ROI centre {center} with radius {radius} does not intersect the "
            f"{height}x{width} panel"
        )
    return np.ascontiguousarray(array[y0:y1, x0:x1])


def _block_geometry(length: int, grid: int) -> tuple[int, int, int]:
    """Whole-block geometry for one axis, matching ``pupil_phase_to_grid``.

    The output has exactly ``grid`` cells along the axis, so the axis is split
    into ``grid`` blocks of ``ceil(length / grid)`` pixels. When ``length`` is
    not an exact multiple the input is zero-padded **up** to
    ``grid * ceil(length / grid)`` -- nothing is ever discarded.

    Args:
        length: Axis length in pixels.
        grid: Number of output cells along the axis.

    Returns:
        ``(block, total, pad)`` with ``block == ceil(length / grid)``,
        ``total == grid * block`` and ``pad == total - length``.
    """
    block = max(1, -(-int(length) // int(grid)))
    total = int(grid) * block
    return block, total, total - int(length)


def coherent_block_mean(
    phase_rad: NDArray[np.floating],
    grid: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Coherent (complex-phasor) area downsample of a phase map.

    Averages ``exp(1j*phi)`` over each block and returns its real and imaginary
    parts, so the caller gets a continuous 2-channel representation of the local
    mean phase instead of an ``arctan2`` with a branch cut. See the module
    docstring for why the arithmetic mean is wrong for a wrapped phase.

    The block geometry and the symmetric zero-pad-up rule are identical to
    :func:`ml.gsnet_debug.offline.pupil_phase_to_grid`; a unit test
    pins that equivalence. On the real 1200x1920 panel this runs in a few
    milliseconds at ``grid=64`` because ``torch`` evaluates ``cos``/``sin`` over
    the whole array with multiple threads.

    Non-finite input pixels are replaced by 0 before the trigonometric
    evaluation, so a NaN can never propagate into a training sample.

    Args:
        phase_rad: Phase map ``(H, W)`` in radians. Need not be wrapped.
        grid: Number of blocks along each axis, ``>= 1``.

    Returns:
        ``(cos_grid, sin_grid)``, each ``(grid, grid) float32``.
        ``hypot(cos_grid, sin_grid)`` is the per-cell coherence in ``[0, 1]`` and
        ``arctan2(sin_grid, cos_grid)`` is the local mean phase.

    Raises:
        ValueError: If ``grid < 1`` or ``phase_rad`` is not a non-empty 2-D array.
    """
    if int(grid) < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")

    array = np.asarray(phase_rad, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"phase_rad must be a non-empty 2D array, got shape {array.shape}"
        )
    # Guarded, not unconditional: nan_to_num always copies, which cost 2.2 ms of
    # the 3.2 ms total on a 1000x1000 ROI, while the finiteness probe is 0.13 ms.
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    height, width = array.shape
    block_h, total_h, pad_h = _block_geometry(height, grid)
    block_w, total_w, pad_w = _block_geometry(width, grid)
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    tensor = torch.from_numpy(array)
    # The block mean is linear, so mean(cos) and mean(sin) of a block equal the
    # real and imaginary parts of mean(exp(1j*phi)) -- no complex tensor needed.
    #
    # ORDER MATTERS: pad cos/sin, never the phase. Padding the phase with zeros
    # would make the filler evaluate to cos(0) = 1, i.e. a phasor of magnitude 1
    # pointing along +Re, which then leaks a spurious ~+0.42 DC term into every
    # border block (measured: 0.414 instead of -0.007 on a random phase).
    cos_blocked = torch.cos(tensor)
    sin_blocked = torch.sin(tensor)
    if pad_h or pad_w:
        cos_blocked = torch.nn.functional.pad(cos_blocked, pad)
        sin_blocked = torch.nn.functional.pad(sin_blocked, pad)
    cos_grid = cos_blocked.reshape(grid, block_h, grid, block_w).mean(dim=(1, 3))
    sin_grid = sin_blocked.reshape(grid, block_h, grid, block_w).mean(dim=(1, 3))

    return (
        np.ascontiguousarray(cos_grid.numpy(), dtype=np.float32),
        np.ascontiguousarray(sin_grid.numpy(), dtype=np.float32),
    )


def phase_to_grid(
    phase_rad: NDArray[np.floating],
    grid: int,
    *,
    center: tuple[int, int] = DEFAULT_PANEL_CENTER,
    radius: float = DEFAULT_PANEL_RADIUS,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """ROI-crop then coherently downsample a full-panel phase map.

    The order is deliberate: crop first, then block-average, so the illuminated
    pupil fills the output grid instead of occupying a few cells of it.

    Args:
        phase_rad: Full-panel phase map in radians.
        grid: Number of blocks along each axis.
        center: Panel centre as ``(x, y)``.
        radius: ROI half-width in panel pixels; ``<= 0`` uses the whole panel.

    Returns:
        ``(cos_grid, sin_grid)``, each ``(grid, grid) float32``.

    Raises:
        ValueError: If ``grid < 1``, ``phase_rad`` is not 2-D, or the ROI misses
            the panel entirely.
    """
    return coherent_block_mean(crop_panel_roi(phase_rad, center, radius), grid)


def zernike_coeffs_to_panel(
    coeffs: NDArray[np.floating],
    *,
    n_max: int,
    resolution: tuple[int, int],
    radius: float,
) -> NDArray[np.float32]:
    """Expand flat Noll-order Zernike coefficients into a panel phase map.

    Units contract: **radians**, Noll order (1-based), piston included. This
    traces to the hardware path, where ``slm_zernike_pib._zernike_to_phase``
    feeds the coefficient vector straight into the radian APIs; no
    ``um_to_waves`` / ``* 2*pi`` conversion applies (doing one is the unit bug
    documented in ``AGENTS.md``).

    A thin wrapper over
    :func:`ml.gsnet_debug.offline.reconstruct_pupil_phase_rad`, with
    one addition: the result is explicitly masked to zero outside the aperture.
    That helper only applies ``nan_to_num``, so pixels just outside the circular
    aperture keep a small non-zero value (measured up to 0.44 rad on a
    15-mode vector) even though its own docstring claims they are exactly zero.
    Those pixels lie inside the illuminated beam and would otherwise contribute a
    spurious phase ring to the coherent block mean.

    Args:
        coeffs: 1-D Zernike coefficient vector, length ``(n_max+1)(n_max+2)/2``.
        n_max: Maximum radial order.
        resolution: Panel resolution as ``(width, height)``.
        radius: Aperture radius in pixels, centred on the panel.

    Returns:
        ``float32`` panel phase map ``(height, width)``, exactly ``0.0`` outside
        the aperture.

    Raises:
        ValueError: Propagated from ``reconstruct_pupil_phase_rad`` when
            ``len(coeffs)`` disagrees with ``n_max``, or when ``radius < 1``.
    """
    flat = np.asarray(coeffs, dtype=np.float64).ravel()
    width, height = (int(resolution[0]), int(resolution[1]))
    aperture = float(radius)
    if aperture < 1.0:
        raise ValueError(f"radius must be >= 1, got {radius}")

    phase = np.asarray(
        reconstruct_pupil_phase_rad(
            flat,
            n_max=int(n_max),
            slm_width=width,
            slm_height=height,
            radius=aperture,
        ),
        dtype=np.float32,
    )
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

    # The Zernike basis is centred on the panel, matching DEFAULT_PANEL_CENTER.
    grid_y, grid_x = np.ogrid[:height, :width]
    inside = (grid_x - width / 2.0) ** 2 + (grid_y - height / 2.0) ** 2 <= aperture**2
    return np.ascontiguousarray(np.where(inside, phase, np.float32(0.0)))


def freeform_grid_to_panel(
    coeffs: NDArray[np.floating],
    *,
    grid: int,
    resolution: tuple[int, int],
) -> NDArray[np.float32]:
    """Upsample a flat ``grid*grid`` freeform phase to the full panel.

    Pixel replication (``np.kron``), byte-identical to the canonical
    :func:`ao_shaping.optimizer.wfless.slm_square_shaping._freeform_phase_radians`:
    every control cell becomes a solid ``ceil(H/grid) x ceil(W/grid)`` block. It
    is replication, not interpolation, and it is not masked to a disc -- the
    driver and the beam do that.

    Args:
        coeffs: Flat array of length ``grid*grid``, radians per cell.
        grid: Expected cell-grid side. Validated against ``len(coeffs)``; pass
            the inferred value if you only know the length.
        resolution: Panel resolution as ``(width, height)``.

    Returns:
        ``float32`` panel phase map ``(height, width)``.

    Raises:
        ValueError: If ``coeffs`` is empty, ``len(coeffs)`` is not a perfect
            square, or ``grid`` disagrees with ``sqrt(len(coeffs))``.
    """
    flat = np.asarray(coeffs, dtype=np.float32).ravel()
    if flat.size == 0:
        raise ValueError("coeffs must contain at least one cell")

    side = math.isqrt(flat.size)
    if side * side != flat.size:
        raise ValueError(
            f"len(coeffs)={flat.size} is not a perfect square; a freeform phase "
            f"grid must hold grid*grid cells"
        )
    if int(grid) != side:
        raise ValueError(
            f"grid={grid} disagrees with len(coeffs)={flat.size}, which implies "
            f"grid={side}"
        )

    width, height = (int(resolution[0]), int(resolution[1]))
    block_h = -(-height // side)
    block_w = -(-width // side)
    upsampled = np.kron(
        flat.reshape(side, side), np.ones((block_h, block_w), dtype=np.float32)
    )
    return np.ascontiguousarray(upsampled[:height, :width], dtype=np.float32)


def _anchored_window(frame: NDArray[np.floating], grid: int) -> NDArray[np.float32]:
    """Extract a ``grid x grid`` window centred on the 0-order spot, zero-padded.

    The 0-order is located by **global argmax** through the canonical
    :func:`~ao_shaping.utils.image.beam_metrics.zero_order_center` and clamped by
    :func:`~ao_shaping.utils.image.beam_metrics.clamp_center_to_frame`; on the 2f
    Fourier bench the optical axis lands at the frame centre only by luck.
    Outside the frame the window is zero-filled.

    This mirrors the private ``_fixed_window`` of
    :mod:`ml.gsnet_debug.offline`, which cannot be reused directly
    because every public entry point there peak-normalises -- exactly the
    transformation the absolute-intensity modes must not apply.

    Args:
        frame: 2-D finite frame.
        grid: Window side.

    Returns:
        ``(grid, grid) float32`` window.
    """
    from ao_shaping.utils.image.beam_metrics import (
        clamp_center_to_frame,
        zero_order_center,
    )

    height, width = frame.shape
    center_x, center_y = zero_order_center(frame, refine=False)
    center_x, center_y = clamp_center_to_frame(
        (center_x, center_y), (height, width), min(grid, height, width)
    )
    half = grid // 2
    top = int(center_y) - half
    left = int(center_x) - half
    out = np.zeros((grid, grid), dtype=np.float32)
    src_y0, src_x0 = max(top, 0), max(left, 0)
    src_y1, src_x1 = min(top + grid, height), min(left + grid, width)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[src_y0 - top : src_y1 - top, src_x0 - left : src_x1 - left] = frame[
            src_y0:src_y1, src_x0:src_x1
        ]
    return out


def farfield_frame_to_grid(
    img: NDArray[np.integer] | NDArray[np.floating],
    grid: int,
    *,
    mode: str = "abs255",
) -> NDArray[np.float32]:
    """Crop a far-field CCD frame to a ``grid x grid`` window on the 0-order spot.

    Args:
        img: Far-field frame, 2-D, any numeric dtype.
        grid: Output side length.
        mode: Scaling applied after the crop:
            ``"abs255"`` (default) divides by 255 and clips to ``[0, 1]``,
            **keeping absolute intensity** -- the default because the exposure
            time is a model *input*, so the target must retain the brightness
            that encodes it. Correct for every dtype in the corpus, including the
            ``float32`` / ``float64`` frames: see the module docstring.
            ``"peak"`` peak-normalises (delegating to
            :func:`~ml.gsnet_debug.offline.farfield_to_grid`), which
            discards absolute intensity. ``"raw"`` returns the float32 values
            unscaled.

    Returns:
        ``(grid, grid) float32`` window.

    Raises:
        ValueError: If ``grid < 1``, ``img`` is not a non-empty 2-D array, or
            ``mode`` is unknown.
    """
    if int(grid) < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    if mode not in ("abs255", "peak", "raw"):
        raise ValueError(
            f"mode must be one of 'abs255', 'peak', 'raw'; got {mode!r}"
        )

    frame = np.asarray(img)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"img must be a non-empty 2D array, got shape {frame.shape}")

    if mode == "peak":
        return np.asarray(
            farfield_to_grid(frame.astype(np.float64), grid), dtype=np.float32
        )

    values = np.asarray(frame, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "abs255":
        values = np.clip(values / np.float32(255.0), 0.0, 1.0)
    return _anchored_window(values, int(grid))
