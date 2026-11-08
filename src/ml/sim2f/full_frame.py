"""Native full-sensor acquisition mixins for the sim2f instruments.

The instrument implementations and their device imports stay in :mod:`ml.sim2f.bench`.
This module contains only the ROI-independent full-frame API, so importing it never
loads a hardware SDK.  The hardware mixin expects its host to provide the camera,
panel-write, and exposure machinery, plus ``_benchmark_error`` and
``_build_panel_phase`` bound to the bench module's exception and phase builder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from torch import Tensor

#: Half-width of the box the 0-order centroid is integrated over, in sensor px.
#: The 2f 0-order is ~14 px FWHM, so +-8 px holds most of the energy while
#: staying small enough that a neighbouring diffraction order cannot leak in.
_ZERO_ORDER_BOX_PX = 8


def _half_max_box(
    data: np.ndarray, row: int, col: int, radius: int
) -> tuple[int, int, int, int]:
    """Half-open ``[r0, r1) x [c0, c1)`` box of ``radius`` around ``(row, col)``."""
    rows, cols = data.shape[0], data.shape[1]
    return (
        max(int(row) - int(radius), 0),
        min(int(row) + int(radius) + 1, rows),
        max(int(col) - int(radius), 0),
        min(int(col) + int(radius) + 1, cols),
    )


def box_centroid(
    frame: np.ndarray, *, row: int, col: int, radius: int = _ZERO_ORDER_BOX_PX,
    max_radius: int = 64,
) -> tuple[float, float]:
    """Half-maximum-weighted centroid of the 0-order around a ``row, col`` seed.

    ``argmax`` alone is not a 0-order locator (issue #235): one hot/defective
    pixel wins outright, and a clipped flat-top spot gives an integer answer
    that can sit a full pixel off the true centre.  Weighting only the pixels
    above the box's half-maximum level is insensitive to both -- an isolated
    hot pixel is one sample among hundreds, and a symmetric flat top averages
    to its geometric centre.

    The box grows (doubling, up to ``max_radius`` or the frame edge) while its
    edge still carries signal, so a spot wider than the initial box is not
    clipped and cannot pull the answer toward the seed.  Falls back to the seed
    when the box holds nothing above the local background (an all-zero frame, a
    saturated field, or a box containing only the hot pixel that seeded it).
    """
    data = np.asarray(frame, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"frame must be 2-D, got {data.shape}")
    if data.shape[0] < 3 or data.shape[1] < 3:
        return float(row), float(col)

    radius = max(int(radius), 1)
    r0, r1, c0, c1 = _half_max_box(data, row, col, radius)
    box = data[r0:r1, c0:c1]
    peak = float(box.max())
    border = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
    background = float(np.median(border))
    threshold = 0.5 * (peak + background)
    while radius < max_radius:
        at_edge = r0 == 0 and c0 == 0 and r1 == data.shape[0] and c1 == data.shape[1]
        if at_edge or float(border.max()) <= threshold:
            break
        radius *= 2
        r0, r1, c0, c1 = _half_max_box(data, row, col, radius)
        box = data[r0:r1, c0:c1]
        peak = float(box.max())
        border = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
        background = float(np.median(border))
        threshold = 0.5 * (peak + background)

    weights = np.clip(box - threshold, 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float(row), float(col)
    axis_rows = np.arange(r0, r1, dtype=np.float64)
    axis_cols = np.arange(c0, c1, dtype=np.float64)
    centroid_row = float((weights.sum(axis=1) * axis_rows).sum() / total)
    centroid_col = float((weights.sum(axis=0) * axis_cols).sum() / total)
    return centroid_row, centroid_col


class _FullFrameMetadataMixin:
    """JSON-safe native-frame metadata for the hardware instrument."""

    def _full_frame_metadata(self) -> dict[str, Any]:
        return {
            "native_shape": (
                None
                if self._native_shape is None
                else [int(v) for v in self._native_shape]
            ),
            "zero_order_xy": (
                None if self._zero_order_xy is None else list(self._zero_order_xy)
            ),
            "full_frame": self._full_frame,
        }


class SimFullFrameMixin:
    """Native-frame API of the calibrated-twin instrument."""

    @property
    def native_shape(self) -> tuple[int, int]:
        """The twin's "sensor" shape ``(height, width)``.

        The twin has no sensor: its read is already the ``ccd_px`` window, so the
        native shape *is* ``(ccd_px, ccd_px)``.  It exists so the hardware
        full-frame path and the dry run share one script, not because the twin
        models a larger detector.
        """
        ccd_px = int(self.bench.cfg.ccd_px)
        return (ccd_px, ccd_px)

    @property
    def zero_order_xy(self) -> tuple[int, int]:
        """The 0-order in "native" coordinates: the twin frame's centre."""
        height, width = self.native_shape
        return (width // 2, height // 2)

    @property
    def full_frame(self) -> bool:
        """Always ``True``: the twin's read is its whole (only) frame."""
        return True

    def enable_full_frame(self) -> tuple[int, int]:
        """No-op on the twin; returns :attr:`native_shape` for a shared script."""
        return self.native_shape

    def capture_full_frame(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """The twin's native frame -- identical to :meth:`capture` here."""
        return self.capture(phase_rad)


class HwFullFrameMixin(_FullFrameMetadataMixin):
    """Native-frame API layered on the existing hardware window machinery."""

    def _locate_zero_order(self) -> None:
        """Centre a ``ccd_px`` window on the 0-order, located by box centroid.

        The 0-order is the frame's global maximum in a 2f bench and is **not** at
        the geometric centre (measured 1441-1443, 705-706 against a centre of
        1344, 760), so geometry is never assumed.  ``argmax`` seeds the search
        (it can only be near the brightest spot) but the location itself comes
        from :func:`box_centroid`, which averages the half-maximum region and so
        ignores hot pixels and resolves sub-pixel centre offsets (issue #235).
        ``reset_window`` clamps the offset and returns the window it really
        applied, so the spot is re-located inside the window and one corrective
        move is made when the peak is more than 4 px off-centre.
        """
        full = np.asarray(self._cam.get_numpy_image(1, skip_first=True))
        if np.issubdtype(full.dtype, np.integer):
            self._full_scale = float(np.iinfo(full.dtype).max)
        self._native_shape = (int(full.shape[0]), int(full.shape[1]))
        row, col = np.unravel_index(int(np.argmax(full)), full.shape)
        row, col = box_centroid(full, row=int(row), col=int(col))
        size = (self._ccd_px, self._ccd_px)
        window, center = self._cam.reset_window((int(round(col)), int(round(row))), size)
        probe = np.asarray(self._cam.get_numpy_image(1, skip_first=True))
        peak_row, peak_col = np.unravel_index(int(np.argmax(probe)), probe.shape)
        peak_row, peak_col = box_centroid(probe, row=int(peak_row), col=int(peak_col))
        half_r, half_c = probe.shape[0] // 2, probe.shape[1] // 2
        if (
            abs(peak_row - half_r) > self._max_zero_order_offset_px
            or abs(peak_col - half_c) > self._max_zero_order_offset_px
        ):
            corrected = (
                int(center[0]) + (int(round(peak_col)) - half_c),
                int(center[1]) + (int(round(peak_row)) - half_r),
            )
            logger.warning(
                "0-order sits {}px off the window centre after reset_window; "
                "re-centring on {}",
                max(abs(peak_row - half_r), abs(peak_col - half_c)),
                corrected,
            )
            window, center = self._cam.reset_window(corrected, size)
        self._window = (int(window[0]), int(window[1]))
        self._center = (int(center[0]), int(center[1]))
        self._zero_order_xy = self._center
        logger.info(
            "CCD window {}x{} centred on the 0-order at {} (full-frame peak was "
            "{},{}, full scale {})",
            self._window[0],
            self._window[1],
            self._center,
            row,
            col,
            self._full_scale,
        )

    @property
    def native_shape(self) -> tuple[int, int]:
        """The camera's full sensor shape ``(height, width)``.

        Raises:
            BenchmarkError: If the initial full-frame grab has not happened --
                a caller must not guess a shape the device never reported.
        """
        if self._native_shape is None:
            raise self._benchmark_error(
                "native sensor shape is unknown until the bench opens"
            )
        return self._native_shape

    @property
    def zero_order_xy(self) -> tuple[int, int]:
        """The located 0-order in native-frame coordinates, pixels ``(x, y)``.

        Raises:
            BenchmarkError: If the bench has not located the 0-order yet.
        """
        if self._zero_order_xy is None:
            raise self._benchmark_error("0-order is unknown until the bench opens")
        return self._zero_order_xy

    @property
    def full_frame(self) -> bool:
        """Whether the ROI has been widened to the native sensor."""
        return self._full_frame

    def enable_full_frame(self) -> tuple[int, int]:
        """Widen the camera ROI to the native sensor, exactly once.

        The bench constructor already grabbed one native frame to locate the
        0-order, so the full shape is known without a second discovery pass.  This
        method restores that ROI and verifies the driver applied it: a silently
        clamped window would otherwise turn every "full" frame into a smaller crop.

        Idempotent.  It never touches the small-window path: :meth:`capture` keeps
        returning ``(ccd_px, ccd_px)`` when this has not been called, and its own
        shape check refuses a full frame.

        Returns:
            The native ``(height, width)`` shape now being read.

        Raises:
            BenchmarkError: If the driver's applied window is not the native size.
        """
        shape = self.native_shape
        if self._full_frame:
            return shape
        height, width = shape
        window, center = self._cam.reset_window(
            (width // 2, height // 2), (width, height)
        )
        applied = (int(window[0]), int(window[1]))
        if applied != (width, height):
            raise self._benchmark_error(
                f"camera applied a {applied[0]}x{applied[1]} window for the native "
                f"{width}x{height} sensor; refusing to call that a full frame"
            )
        self._window = applied
        self._center = (int(center[0]), int(center[1]))
        self._full_frame = True
        logger.info("CCD ROI widened to the native {}x{} sensor", height, width)
        return shape

    def capture_full_frame(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """Write the phase and read one native full-sensor frame.

        Same write path and exposure as :meth:`capture`; only the ROI differs.  The
        returned frame is float32 ``(native_height, native_width)`` and is what an
        offline window choice needs -- all ROI decisions happen after the session,
        so one command is never measured twice.

        Args:
            phase_rad: ``(grid, grid)`` radians.

        Returns:
            ``(native_height, native_width)`` float32 frame in detector counts.

        Raises:
            ValueError: If the camera returns a shape other than the native one.
        """
        expected = self.enable_full_frame()
        panel = self._build_panel_phase(
            phase_rad, self._panel_shape, self._radius_px, self._pupil_center_xy
        )
        self._display_phase(self._slm, panel, self._settle_s)
        frame = np.asarray(
            self._cam.get_numpy_image(self._n_sample, skip_first=True),
            dtype=np.float32,
        )
        if frame.shape != expected:
            raise ValueError(
                f"camera returned {frame.shape}, expected the native {expected}; "
                "the ROI changed mid-session"
            )
        return frame
