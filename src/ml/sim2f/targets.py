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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ml.sim2f.config import TARGET_KINDS

__all__ = ["TARGET_KINDS", "TargetSample", "draw_target"]

#: Linear supersampling factor used to anti-alias every shape.
SUPERSAMPLE = 4

#: Shrink factor applied when a drawn shape would touch the frame border.
_SHRINK = 0.85

#: Maximum number of shrink attempts before giving up.
_MAX_ATTEMPTS = 24


@dataclass(frozen=True)
class TargetSample:
    """One requested far-field distribution.

    Attributes:
        intensity: ``(ccd_px, ccd_px)`` float32, non-negative, unit sum.
        support: ``(ccd_px, ccd_px)`` bool -- the analytic region the shape
            covers. Named support because it is what the teacher's encircled
            energy term integrates; it is *not* a threshold on ``intensity``.
        kind: One of :data:`ml.sim2f.config.TARGET_KINDS`.
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


def _draw_params(kind: str, rng: np.random.Generator) -> dict[str, float]:
    """Draw the kind-specific geometry knobs (fixed order, for determinism).

    Args:
        kind: Shape family.
        rng: NumPy generator.

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
        return {
            "width_px": float(rng.uniform(1.5, 3.0)),
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
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rasterise one shape at the super-sampled resolution.

    Coordinates are already centred on the shape.

    Args:
        kind: Shape family.
        params: Knobs from :func:`_draw_params`.
        x: Centred x coordinates (super-sampled grid).
        y: Centred y coordinates (super-sampled grid).
        size: Characteristic size in camera pixels (the ``size_frac`` product).

    Returns:
        ``(values, mask, geometry)`` -- the super-sampled intensity (float64), the
        geometric mask (bool) and the geometry of the drawn shape.
    """
    radius = np.hypot(x, y)
    if kind == "square":
        half = size / 2.0
        inside = (np.abs(x) <= half) & (np.abs(y) <= half)
        return inside.astype(np.float64), inside, {"size_px": size, "edge_px": size}
    if kind == "rect":
        width = size
        height = size * params["aspect"]
        u, v = _rotated(x, y, params["angle_rad"])
        inside = (np.abs(u) <= width / 2.0) & (np.abs(v) <= height / 2.0)
        return (
            inside.astype(np.float64),
            inside,
            {
                "size_px": size,
                "width_px": width,
                "height_px": height,
                "aspect": params["aspect"],
                "angle_deg": math.degrees(params["angle_rad"]),
            },
        )
    if kind == "disc":
        arc = size / 2.0
        inside = radius <= arc
        return inside.astype(np.float64), inside, {"size_px": size, "radius_px": arc}
    if kind == "ring":
        outer = size / 2.0
        inner = outer * params["inner_frac"]
        mid = 0.5 * (outer + inner)
        half_width = 0.5 * (outer - inner)
        inside = np.abs(radius - mid) <= half_width
        return (
            inside.astype(np.float64),
            inside,
            {
                "size_px": size,
                "outer_radius_px": outer,
                "inner_radius_px": inner,
                "inner_frac": params["inner_frac"],
            },
        )
    if kind == "line":
        length = size
        width = params["width_px"]
        u, v = _rotated(x, y, params["angle_rad"])
        inside = (np.abs(u) <= length / 2.0) & (np.abs(v) <= width / 2.0)
        return (
            inside.astype(np.float64),
            inside,
            {
                "size_px": size,
                "length_px": length,
                "width_px": width,
                "angle_deg": math.degrees(params["angle_rad"]),
            },
        )
    if kind == "cross":
        arm = size / 4.0
        horizontal = (np.abs(x) <= size / 2.0) & (np.abs(y) <= arm / 2.0)
        vertical = (np.abs(x) <= arm / 2.0) & (np.abs(y) <= size / 2.0)
        inside = horizontal | vertical
        return (
            inside.astype(np.float64),
            inside,
            {"size_px": size, "arm_px": arm, "arm_width_px": arm},
        )
    if kind == "two_spot":
        spot_radius = 0.22 * size
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
                "separation_px": separation,
                "gap_px": separation - 2.0 * spot_radius,
                "azimuth_deg": math.degrees(azimuth),
            },
        )
    if kind == "gaussian":
        sigma = size / 3.0
        values = np.exp(-(radius**2) / (2.0 * sigma * sigma))
        inside = radius <= 1.5 * sigma
        return (
            values,
            inside,
            {"size_px": size, "sigma_px": sigma, "radius_px": 1.5 * sigma},
        )
    raise ValueError(f"unknown target kind {kind!r}; known: {list(TARGET_KINDS)}")


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
) -> TargetSample:
    """Draw one anti-aliased target of the requested family.

    Args:
        kind: One of :data:`ml.sim2f.config.TARGET_KINDS`.
        ccd_px: Camera grid side, in pixels.
        rng: NumPy generator. All randomness comes from it, so a fixed seed gives
            a byte-identical sample.
        size_frac: ``(min, max)`` characteristic size as a fraction of
            ``ccd_px``. The shape is shrunk (never grown) if it would touch the
            frame border.
        center_px: Shape centre offset from the frame centre, camera pixels
            ``(x, y)``.

    Returns:
        The drawn :class:`TargetSample`.

    Raises:
        ValueError: If ``kind`` is unknown, the geometry is out of range, or the
            rasterised shape fails a sanity check (non-finite, negative, not
            unit sum, empty support, or support outside the non-zero image).
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
    params = _draw_params(kind, rng)
    values = mask = None
    geometry: dict[str, float] = {}
    for _ in range(_MAX_ATTEMPTS):
        values, mask, geometry = _shape_field(
            kind, params, grid_x - center_x, grid_y - center_y, size
        )
        if not _touches_border(mask):
            break
        size *= _SHRINK
    else:  # pragma: no cover - only reachable for a degenerate centre
        raise ValueError(
            f"could not fit a {kind!r} target of size <= {size:.2f}px inside the "
            f"{ccd_px}px frame at centre {center_px!r}"
        )

    assert values is not None and mask is not None  # for type checkers only
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
    }
    return TargetSample(
        intensity=intensity, support=support, kind=kind, geometry=geometry
    )
