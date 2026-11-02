"""Camera sub-window (ROI) support for small-sensor experiments.

A real CCD may be read out over a small region of interest: the *sensor* still
sees the whole far field, but only the window is measured.  The simulator has no
ROI concept of its own -- :meth:`ml.sim2f.optics.Sim2f.forward` always returns
the full ``ccd_px`` frame -- so a window is modelled here as a crop of that
frame, and :class:`WindowedBench` presents the cropped view through the same
bench protocol the fit machinery already expects.

Keeping the crop outside :mod:`ml.sim2f.optics` means the physics never changes:
a window only decides which pixels a measurement exposes, never how light
propagates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
import torch

__all__ = [
    "CameraWindow",
    "WindowedBench",
    "crop_measure",
]


def _as_pair(value: Sequence[float] | float, name: str) -> tuple[float, float]:
    """Normalise a scalar or 2-sequence to a float pair."""
    if np.isscalar(value):
        return float(value), float(value)  # type: ignore[arg-type]
    pair = tuple(float(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be a scalar or a 2-sequence, got {value!r}")
    return pair


@dataclass(frozen=True)
class CameraWindow:
    """A rectangular region of a square camera frame.

    The window is described in *camera pixels* of the full frame, with the
    origin at the top-left pixel and ``y`` growing downwards (the array layout).

    Attributes:
        center: ``(x, y)`` centre of the window, pixels.  Fractional values are
            rounded to the nearest pixel when the crop is taken.
        size: ``(width, height)`` of the window, pixels.  Must be square for the
            bench protocol; :class:`WindowedBench` enforces that.
        frame: ``(width, height)`` of the full frame.  ``None`` means "unknown",
            in which case :meth:`crop` clips against the array it is given.
    """

    center: tuple[float, float]
    size: tuple[int, int]
    frame: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        cx, cy = _as_pair(self.center, "center")
        w, h = _as_pair(self.size, "size")
        if w <= 0 or h <= 0:
            raise ValueError(f"window size must be positive, got {self.size!r}")
        if int(w) != w or int(h) != h:
            raise ValueError(f"window size must be whole pixels, got {self.size!r}")
        object.__setattr__(self, "center", (cx, cy))
        object.__setattr__(self, "size", (int(w), int(h)))
        if self.frame is not None:
            fw, fh = _as_pair(self.frame, "frame")
            if fw <= 0 or fh <= 0:
                raise ValueError(f"frame size must be positive, got {self.frame!r}")
            object.__setattr__(self, "frame", (int(fw), int(fh)))

    # -- geometry -------------------------------------------------------------
    @classmethod
    def centered(
        cls, size: int | tuple[int, int], frame: int | tuple[int, int]
    ) -> "CameraWindow":
        """A window of ``size`` at the centre of a ``frame``-sized sensor."""
        fw, fh = _as_pair(frame, "frame")
        w, h = _as_pair(size, "size")
        return cls(
            center=(fw / 2.0, fh / 2.0), size=(int(w), int(h)), frame=(int(fw), int(fh))
        )

    @property
    def width(self) -> int:
        """Window width, pixels."""
        return int(self.size[0])

    @property
    def height(self) -> int:
        """Window height, pixels."""
        return int(self.size[1])

    @property
    def square(self) -> bool:
        """Whether the window is square (required by the bench protocol)."""
        return self.width == self.height

    def offset(self, frame: tuple[int, int] | None = None) -> tuple[int, int]:
        """Top-left ``(x0, y0)`` of the crop, clipped to the frame.

        Args:
            frame: Override for the stored full-frame size.

        Returns:
            Integer ``(x0, y0)`` such that ``x0 >= 0`` and ``x0 + width`` fits
            the frame when the frame size is known.

        Raises:
            ValueError: If the frame size is known and the window does not fit.
        """
        full = frame if frame is not None else self.frame
        x0 = int(round(self.center[0] - self.width / 2.0))
        y0 = int(round(self.center[1] - self.height / 2.0))
        if full is None:
            return max(x0, 0), max(y0, 0)
        fw, fh = int(full[0]), int(full[1])
        if self.width > fw or self.height > fh:
            raise ValueError(f"window {self.size} does not fit a {fw}x{fh} frame")
        x0 = min(max(x0, 0), fw - self.width)
        y0 = min(max(y0, 0), fh - self.height)
        return x0, y0

    def crop(self, frame: Any) -> Any:
        """Crop the last two dimensions of ``frame`` to this window.

        Works for numpy arrays and torch tensors of any leading shape
        ``(..., H, W)``; the result keeps the leading dimensions.

        Args:
            frame: Array or tensor whose trailing two dimensions are ``(H, W)``.

        Returns:
            The cropped view/copy with shape ``(..., height, width)``.

        Raises:
            ValueError: If ``frame`` is smaller than the window.
        """
        height, width = int(frame.shape[-2]), int(frame.shape[-1])
        x0, y0 = self.offset((width, height))
        return frame[..., y0 : y0 + self.height, x0 : x0 + self.width]

    def paste(self, patch: Any, frame: Any | None = None, *, fill: float = 0.0) -> Any:
        """Place ``patch`` at this window's position on a full frame.

        Args:
            patch: ``(..., height, width)`` array or tensor.
            frame: Existing full frame to write into.  ``None`` allocates a
                zero (``fill``) frame of the stored ``frame`` size.
            fill: Background value used when allocating.

        Returns:
            A new full frame with the patch written in.

        Raises:
            ValueError: If neither ``frame`` nor a stored frame size is known,
                or the shapes do not agree.
        """
        if frame is None:
            if self.frame is None:
                raise ValueError("paste needs a frame or a stored frame size")
            out = np.full(
                (self.frame[1], self.frame[0]), fill, dtype=np.asarray(patch).dtype
            )
        else:
            out = frame
        x0, y0 = self.offset((int(out.shape[-1]), int(out.shape[-2])))
        if tuple(patch.shape[-2:]) != (self.height, self.width):
            raise ValueError(
                f"patch is {tuple(patch.shape[-2:])}, window is {self.size}"
            )
        out[..., y0 : y0 + self.height, x0 : x0 + self.width] = patch
        return out

    def mask(
        self, frame: int | tuple[int, int], *, dtype: Any = np.float32
    ) -> np.ndarray:
        """A full-frame 0/1 mask that is one inside the window.

        Args:
            frame: Full-frame size, ``(width, height)`` or a square int.
            dtype: Output dtype.

        Returns:
            ``(height, width)`` array on the given ``dtype``.
        """
        fw, fh = _as_pair(frame, "frame")
        out = np.zeros((int(fh), int(fw)), dtype=dtype)
        x0, y0 = self.offset((int(fw), int(fh)))
        out[y0 : y0 + self.height, x0 : x0 + self.width] = 1
        return out

    def coverage(self, frame: int | tuple[int, int]) -> float:
        """Fraction of the full frame this window covers.

        Args:
            frame: Full-frame size.

        Returns:
            ``height * width / (frame_h * frame_w)``.
        """
        fw, fh = _as_pair(frame, "frame")
        return (self.height * self.width) / (float(fh) * float(fw))

    def shifted(self, dx: float, dy: float) -> "CameraWindow":
        """A copy of this window moved by ``(dx, dy)`` pixels."""
        return replace(self, center=(self.center[0] + dx, self.center[1] + dy))

    def with_frame(self, frame: int | tuple[int, int]) -> "CameraWindow":
        """A copy of this window with a known full-frame size."""
        fw, fh = _as_pair(frame, "frame")
        return replace(self, frame=(int(fw), int(fh)))


class WindowedBench:
    """The bench protocol restricted to a :class:`CameraWindow`.

    ``forward`` returns the cropped frame; the fit machinery compares prediction
    and measurement on that same crop, so an aberration can be estimated from
    window pixels while the underlying physics stays full-frame.

    The wrapped bench is never modified: building the model prediction for the
    whole sensor is still one call to :meth:`full_forward`, which is what makes
    the out-of-window extrapolation testable.

    Attributes:
        bench: The wrapped bench (``grid``/``ccd_px``/``dtype``/``device``/
            ``cfg``/``aberration``/``forward``).
        window: The camera window, with ``frame`` filled in from the bench.
    """

    def __init__(self, bench: object, window: CameraWindow) -> None:
        """Wrap ``bench`` with ``window``.

        Args:
            bench: A :class:`~ml.sim2f.optics.Sim2f` or
                :class:`~ml.sim2f.calibrate.CalibratedBench`.
            window: The region of the bench's camera frame to expose.

        Raises:
            ValueError: If the window is not square or does not fit the bench.
        """
        ccd = int(bench.ccd_px)  # type: ignore[attr-defined]
        if not window.square:
            raise ValueError(
                f"the bench protocol needs a square window, got {window.size}"
            )
        window = window.with_frame((ccd, ccd))
        if window.width > ccd:
            raise ValueError(f"window {window.size} exceeds the {ccd}px bench frame")
        window.offset()  # validates the fit
        self._bench = bench
        self.window = window

    # -- the bench protocol ---------------------------------------------------
    @property
    def grid(self) -> int:
        """Model grid, pixels (unchanged by the window)."""
        return int(self._bench.grid)  # type: ignore[attr-defined]

    @property
    def ccd_px(self) -> int:
        """Camera window, pixels -- what the fit sees."""
        return self.window.width

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype of the wrapped bench."""
        return self._bench.dtype  # type: ignore[attr-defined]

    @property
    def device(self) -> torch.device:
        """Compute device of the wrapped bench."""
        return torch.device(self._bench.device)  # type: ignore[attr-defined]

    @property
    def cfg(self) -> Any:
        """The wrapped bench's geometry."""
        return self._bench.cfg  # type: ignore[attr-defined]

    @property
    def full_ccd_px(self) -> int:
        """Full sensor size, pixels."""
        return int(self._bench.ccd_px)  # type: ignore[attr-defined]

    def aberration(self, coeffs: Any) -> torch.Tensor:
        """Zernike map of a coefficient vector, on the wrapped bench's basis."""
        return self._bench.aberration(coeffs)  # type: ignore[attr-defined]

    def full_forward(self, phase: Any, **kwargs: Any) -> torch.Tensor:
        """The unwindowed forward pass (the model's whole-sensor prediction)."""
        return self._bench.forward(phase, **kwargs)  # type: ignore[attr-defined]

    def forward(self, phase: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass cropped to the window.

        Args:
            phase: ``(n, grid, grid)`` commanded radians.
            **kwargs: Passed through to the wrapped bench (``aberration``,
                ``illumination``, ``detector``, ...).

        Returns:
            ``(n, 1, window, window)`` intensity or detector frame.
        """
        return self.window.crop(self.full_forward(phase, **kwargs))


def crop_measure(
    measure: Callable[..., Any],
    window: CameraWindow,
) -> Callable[..., Any]:
    """Wrap a ``measure`` callable so it only returns the window.

    ``measure`` may return a frame or a ``(frame, extras)`` tuple; a tuple's
    frame is cropped and everything else is left alone.

    Args:
        measure: The full-frame measurement callable.
        window: The window to crop to.

    Returns:
        A callable with the same call signature whose frame is cropped.
    """

    def _measure(*args: Any, **kwargs: Any) -> Any:
        value = measure(*args, **kwargs)
        if isinstance(value, tuple):
            return (window.crop(value[0]), *value[1:])
        return window.crop(value)

    _measure.__name__ = getattr(measure, "__name__", "measure")
    _measure.__doc__ = getattr(measure, "__doc__", None)
    return _measure
