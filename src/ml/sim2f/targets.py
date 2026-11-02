"""Target-shape families on the camera grid -- numpy only, no torch.

What a target is
================
A target is the far-field distribution the closed loop is asked to produce: a
float image on the camera grid with **unit sum** (all these shapes have the same
total energy, so the shaping problem is purely about *where* the energy goes) and
an **analytic support** mask, which is the region the shape covers.  The support
is a geometric fact, not a threshold on the image: :class:`TargetSample` builds
it by rasterising the shape's distance function, so a dim anti-aliased edge pixel
still belongs to the support if the shape covers any part of it.

Anti-aliasing
=============
Every shape is rasterised at 4x the camera grid and box-downsampled by
``sum / 16`` (the mean of the 16 sub-samples, i.e. the exact area fraction).  A
hard-edged analytic mask would make the teacher's L1 objective discontinuous in
the phase, and a rotated thin bar would alias into a dotted line.

Why the shapes are parameterised, not fixed
==========================================
Both the teacher (labels) and the network (target -> phase) are trained on a
*distribution* of targets, so the corpus must not collapse onto one square.  Each
family draws its own geometry (aspect, angle, gap, width) from the caller's
generator, and every number that was drawn is reported in
:attr:`TargetSample.geometry` so a cache entry can be described without
re-deriving it.

The feature floor (why a 2 px line is not a target)
===================================================
This bench cannot produce a feature thinner than its own point-spread function:
the calibrated twin's PSF FWHM is **16.0 CCD px / 35.2 um** (measured,
``report/shape_trials/sim_vs_hw/``).  A target whose thinnest bright feature is
below ``1.5 x FWHM`` is therefore not a shaping problem but a request for a delta
function, and the teacher measurably cannot solve it: with 800 iterations, a
``line`` drawn at 1.5-3.0 px (0.12 x FWHM) stalls at L1 1.62 / NCC 0.38, while
``gaussian`` reaches 0.029 (``ml.sim2f.losses.objective_terms``).  So every
family now (a) draws its thinnest dimension no smaller than
:data:`DEFAULT_MIN_FEATURE_PX` -- growing the *other* dimensions where the shape
would otherwise stop being that shape -- and (b) reports that dimension as
``geometry["thin_px"]``, re-measured from the geometry by
:func:`thin_feature_px`, which raises when it is below the floor.  There is no
silent resampling: a target that cannot reach the floor is an error, not a pair
in the corpus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ml.sim2f.config import TARGET_KINDS

__all__ = [
    "DEFAULT_MIN_FEATURE_PX",
    "REFERENCE_CCD_PX",
    "SUPERSAMPLE",
    "TARGET_KINDS",
    "TargetSample",
    "draw_target",
    "mask_target",
    "thin_feature_px",
]

#: Linear supersampling factor used to anti-alias every shape.
SUPERSAMPLE = 4

#: Shrink factor applied when a drawn shape would touch the frame border.
_SHRINK = 0.85

#: Maximum number of shrink attempts before giving up.
_MAX_ATTEMPTS = 24

#: Floor on the thinnest bright feature, camera pixels -- the same number as
#: :attr:`ml.sim2f.config.SynthConfig.min_feature_px`, restated here because this
#: module must stay usable without a config object (``test_targets`` asserts the
#: two agree).  1.5 x the measured PSF FWHM of 16.0 CCD px / 35.2 um on the
#: calibrated twin (``report/shape_trials/sim_vs_hw/``).
DEFAULT_MIN_FEATURE_PX = 24.0

#: Camera grid the floor above is calibrated on (``Sim2fConfig().ccd_px``).
#: Every target length in this module is a fraction of ``ccd_px``, so the *default*
#: floor is capped at ``DEFAULT_MIN_FEATURE_PX / REFERENCE_CCD_PX`` (9.375%) of a
#: smaller window: on a 64 px toy grid the unit tests draw on, a 24 px bar would
#: be the whole frame, and the guard would reject every family.  An explicit
#: ``min_feature_px`` is honoured verbatim -- the corpus path always passes the
#: configured value, so the reference 256 px window gets exactly 24 px.
REFERENCE_CCD_PX = 256

#: Gaussian FWHM per 1-sigma: ``2 * sqrt(2 * ln 2)``.
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

#: Minimum length / width ratio of the ``line`` family (a line must stay a line).
_LINE_MIN_ASPECT = 3.0

#: ``two_spot`` spot diameter as a fraction of the drawn size.
_TWO_SPOT_DIAMETER_FRAC = 0.44

#: ``gaussian`` FWHM as a fraction of the drawn size (``sigma = size / 3``).
_GAUSSIAN_FWHM_FRAC = _FWHM_PER_SIGMA / 3.0


def _resolve_floor(min_feature_px: float | None, ccd_px: int) -> float:
    """Resolve the feature floor for one draw, in camera pixels of this grid.

    Args:
        min_feature_px: Explicit floor in camera pixels, or ``None`` for the
            default (see :data:`REFERENCE_CCD_PX` for how the default is capped on
            windows smaller than the reference).
        ccd_px: Camera grid side, pixels.

    Returns:
        The floor to enforce, pixels.

    Raises:
        ValueError: If the floor is not finite / not positive, or does not fit
            the frame.
    """
    if min_feature_px is None:
        floor = min(DEFAULT_MIN_FEATURE_PX, DEFAULT_MIN_FEATURE_PX / REFERENCE_CCD_PX * ccd_px)
    else:
        floor = float(min_feature_px)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError(f"min_feature_px must be finite and > 0, got {min_feature_px!r}")
    if floor >= ccd_px / 2.0:
        raise ValueError(
            f"min_feature_px = {floor:.2f}px cannot fit a shape in a {ccd_px}px frame "
            "(it must stay below half the frame); lower min_feature_px or enlarge ccd_px"
        )
    return floor


@dataclass(frozen=True)
class TargetSample:
    """One requested far-field distribution.

    Attributes:
        intensity: ``(ccd_px, ccd_px)`` float32, non-negative, unit sum.
        support: ``(ccd_px, ccd_px)`` bool -- the analytic region the shape
            covers. Named support because it is what the teacher's encircled
            energy term integrates; it is *not* a threshold on ``intensity``.
        kind: One of :data:`ml.sim2f.config.TARGET_KINDS`, or ``"mask"`` for a
            user-supplied bitmap (:func:`mask_target` -- deliberately not a
            corpus family, since a cache stores ``kind`` as an integer index).
        geometry: The drawn geometry of this sample, in camera pixels / degrees,
            so a cache entry can be described without the generator.
    """

    intensity: np.ndarray
    support: np.ndarray
    kind: str
    geometry: dict[str, float]


def _block_sum(grid: np.ndarray) -> np.ndarray:
    """Box-downsample a ``SUPERSAMPLE``-times finer square grid by summing.

    Args:
        grid: ``(n * SUPERSAMPLE, n * SUPERSAMPLE)`` array.

    Returns:
        ``(n, n)`` array whose entries are SUMMER over each ``SUPERSAMPLE`` block.
    """
    rows, cols = grid.shape
    factor = SUPERSAMPLE
    return grid.reshape(rows // factor, factor, cols // factor, factor).sum(axis=(1, 3))


def _block_max(grid: np.ndarray) -> np.ndarray:
    """Box-pool a ``SUPERSAMPLE``-times finer boolean grid with logical OR.

    Args:
        grid: ``(n * SUPERSAMPLE, n * SUPERSAMPLE)`` boolean array.

    Returns:
        ``(n, n)`` boolean array, True where any sub-sample is covered.
    """
    rows, cols = grid.shape
    factor = SUPERSAMPLE
    return grid.reshape(rows // factor, factor, cols // factor, factor).any(axis=(1, 3))


def _rotated(x: np.ndarray, y: np.ndarray, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate centred coordinates by ``angle_rad``.

    Args:
        x: x coordinate offsets.
        y: y coordinate offsets.
        angle_rad: Rotation angle, radians.

    Returns:
        ``(u, v)`` rotated offsets.
    """
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    return x * cos + y * sin, -x * sin + y * cos


def _draw_params(
    kind: str, rng: np.random.Generator, *, size: float, min_feature_px: float
) -> dict[str, float]:
    """Draw the kind-specific geometry knobs (fixed order, for determinism).

    Args:
        kind: Shape family.
        rng: NumPy generator.
        size: Characteristic size already drawn for this sample, pixels -- the
            ``line`` width range depends on it so the bar cannot outgrow its own
            length.
        min_feature_px: Feature floor, pixels.

    Returns:
        Mapping of the drawn knobs this kind needs.
    """
    if kind == "rect":
        return {
            "aspect": float(rng.uniform(0.45, 0.8)),
            "angle_rad": float(rng.uniform(0.0, math.pi)),
        }
    if kind == "ring":
        return {"inner_frac": float(rng.uniform(0.55, 0.7))}
    if kind == "line":
        # The bar is thin and long by definition: its width is drawn at or above
        # the feature floor (a 2 px bar is 0.12 x PSF FWHM, i.e. a delta
        # function) and never above a third of the length already drawn.
        width_high = max(min_feature_px, min(1.6 * min_feature_px, size / _LINE_MIN_ASPECT))
        return {
            "width_px": float(rng.uniform(min_feature_px, width_high)),
            "angle_rad": float(rng.uniform(0.0, math.pi)),
        }
    if kind == "two_spot":
        return {"azimuth_rad": float(rng.uniform(0.0, 2.0 * math.pi))}
    return {}


def _shape_field(
    kind: str,
    params: dict[str, float],
    x: np.ndarray,
    y: np.ndarray,
    size: float,
    min_feature_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rasterise one shape at the super-sampled resolution.

    Coordinates are already centred on the shape.

    Every family's thinnest dimension is grown to at least ``min_feature_px``
    (and the *other* dimensions grown with it where the shape would otherwise
    stop being that shape); the reported geometry carries the resulting numbers,
    and :func:`thin_feature_px` re-measures them.

    Args:
        kind: Shape family.
        params: Knobs from :func:`_draw_params`.
        x: Centred x coordinates (super-sampled grid).
        y: Centred y coordinates (super-sampled grid).
        size: Characteristic size in camera pixels (the ``size_frac`` product).
        min_feature_px: Feature floor, camera pixels.

    Returns:
        ``(values, mask, geometry)`` -- the super-sampled intensity (float64), the
        geometric mask (bool) and the geometry of the drawn shape.
    """
    radius = np.hypot(x, y)
    if kind == "square":
        size = max(size, min_feature_px)
        half = size / 2.0
        inside = (np.abs(x) <= half) & (np.abs(y) <= half)
        return inside.astype(np.float64), inside, {"size_px": size, "edge_px": size}
    if kind == "rect":
        width = max(size, min_feature_px)
        # The thin side is the height; grow it to the floor but never past the
        # long side (that would stop being a rectangle).
        height = min(width, max(size * params["aspect"], min_feature_px))
        u, v = _rotated(x, y, params["angle_rad"])
        inside = (np.abs(u) <= width / 2.0) & (np.abs(v) <= height / 2.0)
        return (
            inside.astype(np.float64),
            inside,
            {
                "size_px": width,
                "width_px": width,
                "height_px": height,
                "aspect": height / width,
                "angle_deg": math.degrees(params["angle_rad"]),
            },
        )
    if kind == "disc":
        size = max(size, min_feature_px)
        arc = size / 2.0
        inside = radius <= arc
        return inside.astype(np.float64), inside, {"size_px": size, "radius_px": arc}
    if kind == "ring":
        # Keep a hole: the outer radius is at least 1.5 floors, so a wall of one
        # floor still leaves an inner radius of half a floor.
        outer = max(size / 2.0, 1.5 * min_feature_px)
        wall = min(outer, max(outer * (1.0 - params["inner_frac"]), min_feature_px))
        inner = outer - wall
        mid = 0.5 * (outer + inner)
        half_width = 0.5 * (outer - inner)
        inside = np.abs(radius - mid) <= half_width
        return (
            inside.astype(np.float64),
            inside,
            {
                "size_px": 2.0 * outer,
                "outer_radius_px": outer,
                "inner_radius_px": inner,
                "inner_frac": inner / outer,
                "wall_px": wall,
            },
        )
    if kind == "line":
        width = params["width_px"]
        length = max(size, _LINE_MIN_ASPECT * width)
        u, v = _rotated(x, y, params["angle_rad"])
        inside = (np.abs(u) <= length / 2.0) & (np.abs(v) <= width / 2.0)
        return (
            inside.astype(np.float64),
            inside,
            {
                "size_px": length,
                "length_px": length,
                "width_px": width,
                "angle_deg": math.degrees(params["angle_rad"]),
            },
        )
    if kind == "cross":
        # The arm thickness is the thin feature; the arms themselves are grown so
        # a thick arm is still an arm and not the whole square.
        size = max(size, 2.0 * min_feature_px)
        arm = min(size / 2.0, max(size / 4.0, min_feature_px))
        horizontal = (np.abs(x) <= size / 2.0) & (np.abs(y) <= arm / 2.0)
        vertical = (np.abs(x) <= arm / 2.0) & (np.abs(y) <= size / 2.0)
        inside = horizontal | vertical
        return (
            inside.astype(np.float64),
            inside,
            {
                "size_px": size,
                "arm_px": arm,
                "arm_width_px": arm,
                "arm_length_px": size,
            },
        )
    if kind == "two_spot":
        # The spot diameter is the thin feature; the separation scales with it.
        size = max(size, min_feature_px / _TWO_SPOT_DIAMETER_FRAC)
        spot_radius = 0.5 * _TWO_SPOT_DIAMETER_FRAC * size
        separation = 0.5 * size
        azimuth = params["azimuth_rad"]
        half = 0.5 * separation
        cx1 = half * math.cos(azimuth)
        cy1 = half * math.sin(azimuth)
        first = np.hypot(x - cx1, y - cy1) <= spot_radius
        second = np.hypot(x + cx1, y + cy1) <= spot_radius
        inside = first | second
        return (
            inside.astype(np.float64),
            inside,
            {
                "size_px": size,
                "radius_px": spot_radius,
                "diameter_px": 2.0 * spot_radius,
                "separation_px": separation,
                "gap_px": separation - 2.0 * spot_radius,
                "azimuth_deg": math.degrees(azimuth),
            },
        )
    if kind == "gaussian":
        # The thin feature of a smooth blob is its FWHM, not its support radius.
        size = max(size, min_feature_px / _GAUSSIAN_FWHM_FRAC)
        sigma = size / 3.0
        values = np.exp(-(radius**2) / (2.0 * sigma * sigma))
        inside = radius <= 1.5 * sigma
        return (
            values,
            inside,
            {
                "size_px": size,
                "sigma_px": sigma,
                "fwhm_px": _FWHM_PER_SIGMA * sigma,
                "radius_px": 1.5 * sigma,
            },
        )
    raise ValueError(f"unknown target kind {kind!r}; known: {list(TARGET_KINDS)}")


def thin_feature_px(
    kind: str, params: Mapping[str, float], *, floor_px: float | None = None
) -> float:
    """Measure a drawn family's thinnest bright feature from its geometry.

    This is a **measurement of the drawn geometry**, not a restatement of the draw
    range: it recomputes the thin dimension from the reported primitives (a bar's
    width, a rectangle's short side, an annulus' wall, a Gaussian's FWHM, ...), so
    a family that forgot to grow a dimension is caught here rather than shipped as
    an unsolvable pair.

    Args:
        kind: Shape family.
        params: The geometry reported by :func:`draw_target` /
            :func:`_shape_field` for one sample.
        floor_px: Feature floor to enforce, camera pixels (``None`` only
            measures).

    Returns:
        The thinnest feature, camera pixels.

    Raises:
        ValueError: If ``kind`` is unknown, the geometry lacks the keys this
            family needs, or the measured feature is below ``floor_px``.
    """
    try:
        if kind == "square":
            thin = float(params["edge_px"])
        elif kind == "rect":
            thin = min(float(params["width_px"]), float(params["height_px"]))
        elif kind in ("disc", "two_spot"):
            thin = 2.0 * float(params["radius_px"])
        elif kind == "ring":
            thin = float(params["outer_radius_px"]) - float(params["inner_radius_px"])
        elif kind == "cross":
            thin = float(params["arm_width_px"])
        elif kind == "line":
            thin = float(params["width_px"])
        elif kind == "gaussian":
            thin = _FWHM_PER_SIGMA * float(params["sigma_px"])
        else:
            raise ValueError(f"unknown target kind {kind!r}; known: {list(TARGET_KINDS)}")
    except KeyError as error:
        raise ValueError(
            f"{kind!r} geometry is missing {error.args[0]!r}; "
            f"expected the keys draw_target reports, got {sorted(params)}"
        ) from error

    if floor_px is not None and thin < float(floor_px):
        raise ValueError(
            f"unreachable {kind!r} target: its thinnest feature is {thin:.2f}px, "
            f"below the {float(floor_px):.2f}px floor "
            "(1.5 x the measured 16.0px PSF FWHM; report/shape_trials/sim_vs_hw/). "
            "This bench cannot shape a feature thinner than its own PSF: draw a "
            "larger geometry (size_frac / min_feature_px) or drop this family."
        )
    return thin


def _touches_border(mask: np.ndarray) -> bool:
    """Whether a super-sampled mask reaches the frame border.

    Args:
        mask: ``(n, n)`` boolean mask.

    Returns:
        True if any border sub-sample is covered.
    """
    return bool(
        mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def draw_target(
    kind: str,
    ccd_px: int,
    rng: np.random.Generator,
    *,
    size_frac: tuple[float, float] = (0.16, 0.44),
    center_px: tuple[float, float] = (0.0, 0.0),
    min_feature_px: float | None = None,
) -> TargetSample:
    """Draw one anti-aliased target of the requested family.

    Args:
        kind: One of :data:`ml.sim2f.config.TARGET_KINDS`.
        ccd_px: Camera grid side, in pixels.
        rng: NumPy generator. All randomness comes from it, so a fixed seed gives
            a byte-identical sample.
        size_frac: ``(min, max)`` characteristic size as a fraction of
            ``ccd_px``. The shape is shrunk (never grown) if it would touch the
            frame border, and grown to :data:`DEFAULT_MIN_FEATURE_PX` if that
            would leave its thin feature below the bench's resolution.
        center_px: Shape centre offset from the frame centre, camera pixels
            ``(x, y)``.
        min_feature_px: Floor on the thinnest bright feature, camera pixels.
            ``None`` uses :data:`DEFAULT_MIN_FEATURE_PX`, capped at that same
            fraction (9.375%) of a window smaller than
            :data:`REFERENCE_CCD_PX`; an explicit value is honoured verbatim
            (:attr:`ml.sim2f.config.SynthConfig.min_feature_px` on the corpus
            path).

    Returns:
        The drawn :class:`TargetSample`, whose ``geometry["thin_px"]`` is the
        measured thinnest feature that was checked against the floor.

    Raises:
        ValueError: If ``kind`` is unknown, the geometry is out of range, the
            floor cannot fit the frame, the drawn shape's thinnest feature is
            still below the floor (:func:`thin_feature_px`), or the rasterised
            shape fails a sanity check (non-finite, negative, not unit sum, empty
            support, or support outside the non-zero image).
    """
    if kind not in TARGET_KINDS:
        raise ValueError(f"unknown target kind {kind!r}; known: {list(TARGET_KINDS)}")
    if isinstance(ccd_px, bool) or not isinstance(ccd_px, (int, np.integer)):
        raise ValueError(f"ccd_px must be an int, got {ccd_px!r}")
    ccd_px = int(ccd_px)
    if ccd_px < 8:
        raise ValueError(f"ccd_px must be >= 8, got {ccd_px}")
    low, high = float(size_frac[0]), float(size_frac[1])
    if not (0.0 < low <= high <= 1.0):
        raise ValueError(f"size_frac must satisfy 0 < low <= high <= 1, got {size_frac!r}")
    floor_px = _resolve_floor(min_feature_px, ccd_px)
    center_x = float(center_px[0])
    center_y = float(center_px[1])
    if abs(center_x) >= ccd_px / 2.0 or abs(center_y) >= ccd_px / 2.0:
        raise ValueError(
            f"center_px {center_px!r} lies outside the {ccd_px}px frame"
        )

    # Sub-sample centres of each camera pixel, in camera pixels from the frame
    # centre (pixel k is centred on k - ccd_px / 2).
    n = ccd_px * SUPERSAMPLE
    axis = (np.arange(n, dtype=np.float64) + 0.5) / SUPERSAMPLE - ccd_px / 2.0
    grid_x, grid_y = np.meshgrid(axis, axis)

    size = float(rng.uniform(low, high)) * ccd_px
    params = _draw_params(kind, rng, size=size, min_feature_px=floor_px)
    values = mask = None
    geometry: dict[str, float] = {}
    for _ in range(_MAX_ATTEMPTS):
        values, mask, geometry = _shape_field(
            kind, params, grid_x - center_x, grid_y - center_y, size, floor_px
        )
        if not _touches_border(mask):
            break
        size *= _SHRINK
    else:  # pragma: no cover - only reachable for a degenerate centre
        raise ValueError(
            f"could not fit a {kind!r} target of size <= {size:.2f}px inside the "
            f"{ccd_px}px frame at centre {center_px!r} "
            f"(the {floor_px:.2f}px feature floor is part of the footprint)"
        )

    assert values is not None and mask is not None  # for type checkers only
    # The floor is a **measured** property of the drawn geometry: every family
    # reports its thinnest dimension, and this re-measures it rather than trusting
    # the draw range.  An unreachable target is an error, never a cached pair.
    thin_px = thin_feature_px(kind, geometry, floor_px=floor_px)

    intensity = _block_sum(values) / float(SUPERSAMPLE**2)
    support = _block_max(mask)

    if not np.isfinite(intensity).all():
        raise ValueError(f"{kind!r} target contains non-finite values")
    if float(intensity.min()) < 0.0:
        raise ValueError(f"{kind!r} target contains negative values")
    total = float(intensity.sum())
    if total <= 0.0:
        raise ValueError(f"{kind!r} target is empty")
    intensity = (intensity / total).astype(np.float32)

    support_count = int(support.sum())
    if support_count < 16:
        raise ValueError(
            f"{kind!r} target support has {support_count} pixels, need >= 16 "
            "(shape too small for the camera grid)"
        )
    if not bool((intensity[support] > 0.0).all()):
        raise ValueError(f"{kind!r} target support includes pixels the image does not cover")

    geometry = {
        **{key: float(value) for key, value in geometry.items()},
        "center_x_px": center_x,
        "center_y_px": center_y,
        "coverage_px": float(support_count),
        "thin_px": thin_px,
    }
    return TargetSample(
        intensity=intensity, support=support, kind=kind, geometry=geometry
    )


def _resample_bilinear(
    mask: np.ndarray, n: int, *, offset_x_px: float, offset_y_px: float, ccd_px: int
) -> np.ndarray:
    """Resample ``mask`` onto the ``(n, n)`` super-grid by bilinear interpolation.

    The mask's full extent maps onto the full ``ccd_px`` camera frame, corner to
    corner: row ``0`` of the super-grid samples the mask's first row and row
    ``n - 1`` its last, so content drawn at the bitmap's edge lands at the
    frame's edge (and is therefore caught by the border check below).  Sampling
    is bilinear, and coordinates the offset pushes outside the mask read as zero
    -- outside the bitmap there is nothing to sample, and replicating the edge
    would invent shape the caller did not draw.

    Args:
        mask: ``(h, w)`` non-negative source bitmap.
        n: Destination grid side (``ccd_px * SUPERSAMPLE``).
        offset_x_px: Centre shift in camera pixels (positive moves the shape
            towards larger columns).
        offset_y_px: Centre shift in camera pixels (positive moves the shape
            towards larger rows).
        ccd_px: Camera grid side, to convert the offset into frame fractions.

    Returns:
        ``(n, n)`` float64 resampled coverage.
    """
    h, w = mask.shape
    steps = np.arange(n, dtype=np.float64) / max(1, n - 1)
    src_x = (steps[None, :] - float(offset_x_px) / float(ccd_px)) * (w - 1)
    src_y = (steps[:, None] - float(offset_y_px) / float(ccd_px)) * (h - 1)

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    wx = src_x - x0
    wy = src_y - y0
    valid = (src_x >= 0.0) & (src_x <= w - 1.0) & (src_y >= 0.0) & (src_y <= h - 1.0)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)

    out = (
        (1.0 - wx) * (1.0 - wy) * mask[y0c, x0c]
        + wx * (1.0 - wy) * mask[y0c, x1c]
        + (1.0 - wx) * wy * mask[y1c, x0c]
        + wx * wy * mask[y1c, x1c]
    )
    return np.where(valid, out, 0.0)


def _median_run_length(support: np.ndarray) -> float:
    """Thinnest-feature estimate of a binary mask, in camera pixels.

    There is no analytic geometry to re-measure for a user-supplied bitmap, so
    the thinnest feature is estimated from the mask itself: the median length
    of the True runs along every row and every column.  The median (not the
    minimum) is deliberate: a single aliased stair-step pixel is not a feature,
    and a shape that is mostly thick with one thin neck is reported by its bulk
    -- the floor exists to keep the *target's* structure above the PSF, not to
    reject incidental single-pixel corners.

    Args:
        support: ``(n, n)`` boolean mask.

    Returns:
        Median run length, pixels; ``0.0`` for an empty mask.
    """
    runs: list[np.ndarray] = []
    for axis in (0, 1):
        lines = support if axis == 1 else support.T
        for line in lines:
            padded = np.concatenate(([False], line, [False]))
            edges = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            if starts.size:
                runs.append(ends - starts)
    if not runs:
        return 0.0
    return float(np.median(np.concatenate(runs)))


def mask_target(
    mask: np.ndarray,
    ccd_px: int,
    *,
    center_px: tuple[float, float] = (0.0, 0.0),
    min_feature_px: float | None = None,
) -> TargetSample:
    """Turn a user-supplied bitmap into an arbitrary-shape target.

    This is the entry point the parameterised families do not cover: any shape
    the caller can draw (a logo, a letter, a measured distribution) becomes a
    target, under the same contract as :func:`draw_target` -- unit sum, an
    anti-aliased intensity, an honest support, and the feature floor.

    Placement
    =========
    The bitmap's full extent maps onto the full ``ccd_px`` camera frame, so the
    caller controls size and position by where the shape sits in the bitmap
    (a shape drawn in the middle third of a square bitmap lands as a
    ``ccd_px / 3`` shape, centred).  ``center_px`` then shifts it in camera
    pixels.  Unlike :func:`draw_target` there is **no silent shrink**: a shape
    that would touch the frame border raises, because clipping a user's bitmap
    would silently change what was asked for.

    Values
    ======
    The bitmap need not be binary: values are a coverage/density, so a greyscale
    bitmap requests a greyscale far field (the intensity carries the grey
    levels, the support is wherever any coverage lands).

    Args:
        mask: ``(h, w)`` array-like, finite and non-negative, with some energy.
        ccd_px: Camera grid side, in pixels.
        center_px: ``(x, y)`` centre shift in camera pixels.
        min_feature_px: Floor on the thinnest bright feature, in camera pixels;
            ``None`` applies the default floor (see :func:`_resolve_floor`).

    Returns:
        The :class:`TargetSample` with ``kind="mask"``; ``geometry`` reports the
        placement, the coverage and the measured thinnest feature
        (``thin_px``, by :func:`_median_run_length`).

    Raises:
        ValueError: If the mask is not a finite non-negative 2-D array, has no
            energy, if ``ccd_px`` is not an int ``>= 8``, if the placed shape
            touches the frame border, or if its thinnest feature is below the
            floor.
    """
    if isinstance(ccd_px, bool) or not isinstance(ccd_px, (int, np.integer)):
        raise ValueError(f"ccd_px must be an int, got {ccd_px!r}")
    ccd_px = int(ccd_px)
    if ccd_px < 8:
        raise ValueError(f"ccd_px must be >= 8, got {ccd_px}")
    floor = _resolve_floor(min_feature_px, ccd_px)

    source = np.asarray(mask, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("mask contains non-finite values")
    if float(source.min()) < 0.0:
        raise ValueError("mask contains negative values")
    if float(source.max()) <= 0.0:
        raise ValueError("mask has no energy (max is zero)")
    source = source / float(source.max())

    center_x, center_y = float(center_px[0]), float(center_px[1])
    n = ccd_px * SUPERSAMPLE
    coverage = _resample_bilinear(
        source, n, offset_x_px=center_x, offset_y_px=center_y, ccd_px=ccd_px
    )

    mask_super = coverage > 0.0
    if _touches_border(mask_super):
        raise ValueError(
            "mask target touches the frame border; shrink or re-centre the shape "
            "in the bitmap (mask_target never clips silently)"
        )

    intensity = _block_sum(coverage) / float(SUPERSAMPLE**2)
    support = _block_max(mask_super)

    if not np.isfinite(intensity).all():
        raise ValueError("mask target contains non-finite values")
    total = float(intensity.sum())
    if total <= 0.0:
        raise ValueError("mask target is empty after resampling")
    intensity = (intensity / total).astype(np.float32)

    support_count = int(support.sum())
    if support_count < 16:
        raise ValueError(
            f"mask target support has {support_count} pixels, need >= 16 "
            "(shape too small for the camera grid)"
        )
    if not bool((intensity[support] > 0.0).all()):
        raise ValueError("mask target support includes pixels the image does not cover")

    thin_px = _median_run_length(support)
    if thin_px < floor:
        raise ValueError(
            f"mask target's thinnest feature is {thin_px:.1f}px < floor {floor:.1f}px "
            f"(the bench PSF is {DEFAULT_MIN_FEATURE_PX:.0f}px FWHM on the "
            f"{REFERENCE_CCD_PX}px reference window); coarsen the shape or lower "
            "min_feature_px"
        )

    geometry = {
        "center_x_px": center_x,
        "center_y_px": center_y,
        "coverage_px": float(support_count),
        "thin_px": thin_px,
        "mask_rows": float(source.shape[0]),
        "mask_cols": float(source.shape[1]),
    }
    return TargetSample(intensity=intensity, support=support, kind="mask", geometry=geometry)
