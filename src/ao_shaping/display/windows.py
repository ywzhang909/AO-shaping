from __future__ import annotations

from abc import ABC
from collections import namedtuple

import numpy as np
from loguru import logger

import importlib

from ao_shaping.display import Image2DFrame, VoltageFrame
from ao_shaping.display.frames import (
    BACKGROUND_COLOR,
    BaseFrame,
    to_display_uint8,
)

# Height (px) of the voltage bar-chart panel under the image.
VOLT_HEIGHT = 200

FrameInfo = namedtuple(
    "FrameInfo", ["name", "title", "frame", "kwargs"], defaults=[None, None, None, {}]
)


class BaseDisplay(ABC):
    def __init__(self, total_size) -> None:
        self.total_size = total_size

    def render(self, info: str = "") -> bool:
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        if info:
            pygame.display.set_caption(info)
        pygame.event.pump()
        pygame.display.update()
        return True

    def init_window(self) -> None:
        import pygame

        pygame.init()
        self.window = pygame.display.set_mode(self.total_size)

    def close(self) -> None:
        import pygame

        for frame in self._frames.values():
            frame.close()
        pygame.quit()

    def __enter__(self):
        self.init_window()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ImageVoltagesDisplay(BaseDisplay):
    """Live view: a CCD image with the bucket circle plus a voltage bar chart.

    This is the **single canonical implementation**. The former duplicate in
    ``ao_shaping.utils.image.display`` now imports this class, so both import
    paths resolve to the same object.

    ``render`` takes the voltage range per call (the DM min/max voltage), which is
    the signature the DM optimizers (``optimize_pib`` / combined PIB) use.
    """

    def __init__(
        self,
        img_size,
        volt_height=VOLT_HEIGHT,
        background_color=BACKGROUND_COLOR,
    ):
        img_w, img_h = int(img_size[0]), int(img_size[1])
        super().__init__((img_w, img_h + volt_height * 2))
        self.img_size = (img_w, img_h)
        self.volt_height = int(volt_height)
        self.background_color = background_color
        self._frames: dict = {}

    def render(self, img, volts, v_min, v_max, center, r, info="") -> bool:
        """Render the image (with bucket circle) and the voltage bars.

        Args:
            img: 2D image (any dtype; normalised for display).
            volts: coefficient/voltage vector drawn as bars.
            v_min / v_max: bar normalisation range.
            center / r: bucket circle centre and radius, in image coordinates.
            info: window caption.

        Returns:
            bool: ``False`` once the user closes the window.
        """
        import pygame

        img8 = to_display_uint8(img)
        img_h, img_w = img8.shape
        canvas = pygame.surfarray.make_surface(img8.transpose())
        pygame.draw.circle(
            canvas, (255, 0, 0), (int(center[0]), int(center[1])), max(1, int(r)), 1
        )
        self.window.blit(canvas, (0, 0))

        span = float(v_max) - float(v_min)
        if span == 0:
            span = 1.0
        plot_area = pygame.Rect(0, img_h, img_w, self.volt_height)
        self.window.fill(self.background_color, plot_area)
        bar_width = max(1, int(img_w / max(len(volts), 1)))
        y_base = int(img_h + self.volt_height)
        for i, v in enumerate(volts):
            norm = (float(v) - float(v_min)) / span
            if not np.isfinite(norm):
                continue  # NaN/inf coefficient: no bar
            norm_c = min(1.0, max(0.0, norm))
            color = (int(norm_c * 255), int((1 - norm_c) * 255), 0)
            x = int(i * bar_width)
            height = int(2 * float(v) * self.volt_height / span)
            pygame.draw.line(
                self.window, color, (x, y_base), (x, y_base - height), bar_width
            )

        return super().render(info)


class AutoDisplay(BaseDisplay):
    def __init__(
        self,
        frame_list: list[FrameInfo],
        frame_size=(300, 300),
        display_size=(1280, 720),
        margin=10,
        grid: tuple[int, int] | None = None,
    ) -> None:
        self.total_size = display_size
        self.frame_size = frame_size
        self.frame_list = frame_list
        self.margin = margin
        # Explicit (n_cols, n_rows) layout; None = derive from display/frame size.
        self.grid = grid

    def _resolve_grid(self, screen_w: int, screen_h: int) -> tuple[int, int]:
        """Return ``(n_cols, n_rows)`` for the frame list.

        ``grid`` (if provided) pins the layout exactly; otherwise the grid is
        derived from the display/frame sizes (legacy auto layout, which cannot
        always produce an exact grid — e.g. 2x2 for four frames).
        """
        frame_w, frame_h = self.frame_size
        if self.grid is not None:
            n_cols, n_rows = int(self.grid[0]), int(self.grid[1])
            if n_cols < 1 or n_rows < 1:
                raise ValueError(f"grid must be positive (cols, rows), got {self.grid}")
            if n_cols * n_rows < len(self.frame_list):
                raise ValueError(
                    f"grid {n_cols}x{n_rows} cannot hold {len(self.frame_list)} frames"
                )
            return n_cols, n_rows

        n_cols = screen_w // frame_w
        n_rows = len(self.frame_list) // n_cols + 1
        if n_rows * frame_h > screen_h:
            n_rows = screen_h // frame_h
            n_cols = len(self.frame_list) // n_rows + 1
        return n_cols, n_rows

    def init_window(self) -> None:
        import pygame

        super().init_window()
        screen_w, screen_h = self.total_size
        frame_w, frame_h = self.frame_size
        n_cols, n_rows = self._resolve_grid(screen_w, screen_h)
        self.n_cols, self.n_rows = n_cols, n_rows

        logger.info(f"AutoDisplay: {n_cols} x {n_rows} = {n_cols * n_rows} frames")
        total_size = (n_cols * frame_w, n_rows * frame_h)
        super().__init__(total_size)

        self._frames = {}
        for i, frame_info in enumerate(self.frame_list):
            name, title, frame_class_name = (
                frame_info.name,
                frame_info.title,
                frame_info.frame,
            )
            _row, _col = divmod(i, n_cols)
            _frame = self.__get_frame_by_name(frame_class_name)
            top = _row * (frame_h + self.margin)
            left = _col * (frame_w + self.margin)
            assert name not in self._frames, f"Frame name {name} is duplicated"
            self._frames[name] = _frame(
                window=self.window,
                render_pos=(top, left),
                frame_size=self.frame_size,
                title=title,
                **frame_info.kwargs,
            )

    def render(self, frame_data: dict[str, dict], info: str = "") -> bool:
        for name, frame in self._frames.items():
            frame.render(**frame_data.get(name))
        return super().render(info)

    @staticmethod
    def __get_frame_by_name(name: str) -> BaseFrame:
        module = importlib.import_module("ao_shaping.display.frames")
        return getattr(module, name)


class DisplayClosedError(RuntimeError):
    """Raised to abort an optimization loop when the pygame window is closed."""


class SlmZernikeDisplay(AutoDisplay):
    """Live pygame view for SLM-Zernike optimization.

    Composes the registered frames into a **2x2** four-panel window: the CCD frame
    with the bucket circle, the SLM phase currently being sent, the Zernike
    coefficient bars and a **metric curve** panel. The curve's x-axis spans the
    configured epoch count (``x = epoch / total_epochs``), so progress is shown
    in proportion to the search length rather than re-spread per point.

    Used as a context manager (inside the camera/SLM ``with`` block) so the
    window is always torn down, including on exceptions. ``update()`` returns
    ``False`` once the user closes the window; ``closed`` reflects that, and the
    optimizer raises :class:`DisplayClosedError` to stop the search.

    All frame input handling (dtype normalisation, bucket-circle scaling) is done
    by the frames themselves, so callers pass raw image/phase arrays and
    source-image coordinates.
    """

    # Default 2x2 panel size / window; the grid is pinned to (2, 2).
    DEFAULT_FRAME_SIZE = (620, 340)
    DEFAULT_DISPLAY_SIZE = (1280, 720)

    def __init__(
        self,
        zernike_clip: float = 5.0,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
        display_size: tuple[int, int] = DEFAULT_DISPLAY_SIZE,
        margin: int = 10,
        grid: tuple[int, int] = (2, 2),
        curve_title: str = "PIB curve",
        curve_y_range: tuple[float, float] | None = None,
        target_shape: str | None = None,
        target_size: float | None = None,
        target_aspect_ratio: float = 1.0,
    ) -> None:
        self.zernike_clip = float(zernike_clip)
        curve_y_min, curve_y_max = (
            curve_y_range if curve_y_range is not None else (None, None)
        )
        self.target_shape = target_shape
        self.target_size = target_size
        self.target_aspect_ratio = float(target_aspect_ratio)
        frames = [
            FrameInfo(
                "ccd",
                "CCD (target)",
                "Image2DWithBucketFrame",
                {
                    "target_shape": target_shape,
                    "target_size": target_size,
                    "target_aspect_ratio": target_aspect_ratio,
                },
            ),
            FrameInfo("phase", "SLM phase (sent)", "Image2DFrame", {}),
            FrameInfo(
                "coeff",
                "Zernike coeffs",
                "VoltageFrame",
                {"v_min": -self.zernike_clip, "v_max": self.zernike_clip},
            ),
            FrameInfo(
                "curve",
                curve_title,
                "EpochCurveFrame",
                {"y_min": curve_y_min, "y_max": curve_y_max},
            ),
        ]
        super().__init__(
            frames,
            frame_size=frame_size,
            display_size=display_size,
            margin=margin,
            grid=grid,
        )
        self._window_closed = False
        self._closed = False

    @property
    def closed(self) -> bool:
        """True once the user has closed the window (or :meth:`close` ran)."""
        return self._window_closed or self._closed

    def update(
        self,
        img: np.ndarray,
        phase: np.ndarray,
        coeffs: np.ndarray,
        center: tuple[int, int],
        r_bucket: float,
        info: str = "",
        value: float | None = None,
        epoch: int | None = None,
        total_epochs: int | None = None,
        target_size: float | None = None,
    ) -> bool:
        """Render one frame; returns ``False`` once the window has been closed.

        Args:
            img / phase / coeffs / center / r_bucket: panels' payloads.
            info: text shown as the window caption and the curve label.
            value: metric to append to the curve panel (``None`` = redraw only).
            epoch: x value for ``value``.
            total_epochs: x-axis span; the point is placed at
                ``epoch / total_epochs`` of the panel width.
            target_size: Override the target box size (pixels) for the current
                frame; ``None`` falls back to the constructor value.
        """
        if self.closed:
            return False

        import pygame

        _target_size = target_size if target_size is not None else self.target_size
        frame_data = {
            "ccd": {
                "img": img,
                "center": center,
                "r": r_bucket,
                "target_shape": self.target_shape,
                "target_size": _target_size,
                "target_aspect_ratio": self.target_aspect_ratio,
            },
            "phase": {"img": phase},
            "coeff": {"volts": np.asarray(coeffs, dtype=np.float64)},
            "curve": {
                "value": value,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "label": info,
            },
        }
        try:
            alive = self.render(frame_data, info=info)
        except pygame.error:
            alive = False
        if not alive:
            self._window_closed = True
        return bool(alive)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if getattr(self, "_frames", None):
            super().close()


def phase_for_display(phase: np.ndarray) -> np.ndarray:
    """Map signed radian phase onto ``[0, 1]`` so its structure is always visible.

    :func:`to_display_uint8` normalises by the **positive peak** only, which is
    correct for the non-negative panels (CCD intensity, predicted far field) but
    silently loses signed data: an all-negative phase maps to an empty panel, and
    a phase with one positive outlier maps to a single lit pixel out of thousands.
    Measured on this feature's own inputs -- an all-negative phase renders
    ``max=0`` (completely blank) and ``[-2, ..., -2, +3]`` renders exactly one
    non-zero pixel out of 4096.

    That is not hypothetical here: Step B's synthesised phase is whatever the
    solver converged to and carries no guarantee of being positive-dominant, so
    the one panel meant to answer "what phase is on the SLM?" could go blank while
    the run looked perfectly healthy.

    Scaling by the **peak-to-peak range** instead keeps every distinct value
    distinguishable. The absolute radian scale is not recoverable from the panel
    (it has no colour bar), so nothing that was readable is lost -- whereas a blank
    panel is actively misleading.

    A constant phase maps to zeros (uniform mid-grey once displayed), which is the
    honest rendering of "no structure".

    Args:
        phase: Raw unwrapped-radian phase, any shape.

    Returns:
        A ``float64`` array in ``[0, 1]``, safe for :func:`to_display_uint8`.
    """
    arr = np.asarray(phase, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    span = float(arr.max()) - lo
    if span <= 0.0:
        return np.zeros_like(arr)
    # ``to_display_uint8`` divides by the peak, and this is 1.0 by construction, so
    # the panel's own normalisation becomes a no-op rather than a second rescale.
    return (arr - lo) / span


class SlmModelInLoopDisplay(SlmZernikeDisplay):
    """Live pygame view for model-in-the-loop square shaping.

    **Shares** the four-panel window with :class:`SlmZernikeDisplay` -- the same
    2x2 geometry, the same ``FrameInfo`` panel descriptors, the same frame
    primitives, and the same ``closed`` / ``close()`` contract -- and swaps
    exactly one panel: where the PIB view draws the Zernike coefficient bars,
    this one draws the **forward model's predicted far field**.

    That single substitution is what makes this loop watchable. Step A fits one
    shared Zernike aberration by minimising
    ``loss(measured CCD, predicted far field)``, so showing the two side by side
    lets the fit be *seen* converging; no scalar in the history can show that.
    The fitted coefficients themselves stay in the recorder/CSV, which is the
    right place for numbers rather than a fourth picture.

    Panel contents:

    ``ccd``
        The measured camera frame with the square target outlined, in frozen
        optical-axis (source-image) coordinates.
    ``phase``
        The raw unwrapped-radian phase currently on the SLM panel.
    ``predicted``
        The forward model's far field for ``(coefficients, phase)``. Peak
        normalised per frame, so this is a *shape* comparison against ``ccd`` and
        not an absolute-brightness one -- the two live on different grids
        (camera pixels vs the padded model grid) and different radiometries.
    ``curve``
        The per-round metric log.

    The prediction panel renders blank while no prediction is available yet
    (the flat baseline is displayed before the model has any fitted
    aberration), rather than raising.
    """

    def __init__(
        self,
        frame_size: tuple[int, int] = SlmZernikeDisplay.DEFAULT_FRAME_SIZE,
        display_size: tuple[int, int] = SlmZernikeDisplay.DEFAULT_DISPLAY_SIZE,
        margin: int = 10,
        grid: tuple[int, int] = (2, 2),
        curve_title: str = "score curve",
        curve_y_range: tuple[float, float] | None = None,
        target_shape: str | None = "square",
        target_size: float | None = None,
        target_aspect_ratio: float = 1.0,
    ) -> None:
        # ``zernike_clip`` only parameterises the coefficient-bar panel this
        # layout does not have; the base constructor still requires a value.
        super().__init__(
            zernike_clip=1.0,
            frame_size=frame_size,
            display_size=display_size,
            margin=margin,
            grid=grid,
            curve_title=curve_title,
            curve_y_range=curve_y_range,
            target_shape=target_shape,
            target_size=target_size,
            target_aspect_ratio=target_aspect_ratio,
        )
        # Rebind the panel list AFTER the base constructor, which installs the
        # PIB layout. ``AutoDisplay.init_window`` instantiates the frames from
        # ``self.frame_list``, so replacing it here is what changes the window.
        self.frame_list = [
            FrameInfo(
                "ccd",
                "CCD (measured)",
                "Image2DWithBucketFrame",
                {
                    "target_shape": target_shape,
                    "target_size": target_size,
                    "target_aspect_ratio": target_aspect_ratio,
                },
            ),
            FrameInfo("phase", "SLM phase (sent)", "Image2DFrame", {}),
            FrameInfo("predicted", "Predicted far field", "Image2DFrame", {}),
            FrameInfo(
                "curve",
                curve_title,
                "EpochCurveFrame",
                {
                    "y_min": curve_y_range[0] if curve_y_range else None,
                    "y_max": curve_y_range[1] if curve_y_range else None,
                },
            ),
        ]

    def update(
        self,
        measured: np.ndarray,
        phase: np.ndarray,
        predicted: np.ndarray | None = None,
        center: tuple[int, int] = (0, 0),
        r: float = 0.0,
        info: str = "",
        value: float | None = None,
        epoch: int | None = None,
        total_epochs: int | None = None,
        target_size: float | None = None,
    ) -> bool:
        """Render one frame; returns ``False`` once the window has been closed.

        Args:
            measured: The camera frame just read, background-subtracted.
            phase: Raw unwrapped-radian phase on the SLM right now.
            predicted: The forward model's far field, or ``None`` to leave the
                panel blank.
            center / r: Target overlay position/radius in **source-image**
                coordinates; the frame class scales them to the zoomed surface.
            info: Window caption and curve label.
            value: Metric to append to the curve panel (``None`` redraws only,
                which is what the intra-round probe frames want).
            epoch: x value for ``value``.
            total_epochs: x-axis span, so the curve advances in proportion to the
                configured round count.
            target_size: Per-frame target box override in pixels; ``None`` falls
                back to the constructor value.
        """
        if self.closed:
            return False

        import pygame

        _target_size = target_size if target_size is not None else self.target_size
        frame_data = {
            "ccd": {
                "img": measured,
                "center": center,
                "r": r,
                "target_shape": self.target_shape,
                "target_size": _target_size,
                "target_aspect_ratio": self.target_aspect_ratio,
            },
            "phase": {"img": phase_for_display(phase)},
            "predicted": {
                # A blank panel rather than a stale or fabricated image: a
                # missing prediction must not read as a fitted result.
                "img": np.zeros((2, 2), dtype=np.float64)
                if predicted is None
                else predicted
            },
            "curve": {
                "value": value,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "label": info,
            },
        }
        try:
            alive = self.render(frame_data, info=info)
        except pygame.error:
            alive = False
        if not alive:
            self._window_closed = True
        return bool(alive)
