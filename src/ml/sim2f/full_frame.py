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
        """Centre a ``ccd_px`` window on the 0-order, located by ``argmax``.

        The 0-order is the frame's global maximum in a 2f bench and is **not** at
        the geometric centre (measured 1441-1443, 705-706 against a centre of
        1344, 760), so geometry is never assumed.  ``reset_window`` clamps the
        offset and returns the window it really applied, so the spot is
        re-located inside the window and one corrective move is made when the
        peak is more than 4 px off-centre.
        """
        full = np.asarray(self._cam.get_numpy_image(1, skip_first=True))
        if np.issubdtype(full.dtype, np.integer):
            self._full_scale = float(np.iinfo(full.dtype).max)
        self._native_shape = (int(full.shape[0]), int(full.shape[1]))
        row, col = np.unravel_index(int(np.argmax(full)), full.shape)
        size = (self._ccd_px, self._ccd_px)
        window, center = self._cam.reset_window((int(col), int(row)), size)
        probe = np.asarray(self._cam.get_numpy_image(1, skip_first=True))
        peak_row, peak_col = np.unravel_index(int(np.argmax(probe)), probe.shape)
        half_r, half_c = probe.shape[0] // 2, probe.shape[1] // 2
        if (
            abs(peak_row - half_r) > self._max_zero_order_offset_px
            or abs(peak_col - half_c) > self._max_zero_order_offset_px
        ):
            corrected = (
                int(center[0]) + (int(peak_col) - half_c),
                int(center[1]) + (int(peak_row) - half_r),
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
