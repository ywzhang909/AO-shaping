from abc import ABC

import numpy as np
from loguru import logger
from scipy.ndimage import zoom

BACKGROUND_COLOR = (0, 0, 0)
Title_height = 36
DPI = 96

__frame_registry = {}

def get_frame_names():
    return __frame_registry.keys()

def get_frame(frame_name:str):
    return __frame_registry[frame_name]

def register_frame(frame_name:str):
    def decorator(cls):
        assert frame_name not in __frame_registry, f"Frame name {frame_name} already registered"
        __frame_registry[frame_name] = cls
        return cls
    return decorator

def to_display_uint8(img: np.ndarray) -> np.ndarray:
    """Normalise a 2D image/phase array to contiguous ``uint8`` (0-255) for pygame.

    Handles the dtypes the CCD/optimizer layers actually produce: ``uint8`` is
    passed through, ``uint16`` frames and ``float`` radian phases are scaled by
    their own peak, and NaN/inf are mapped to 0.

    This is a **display-only** transform: its output is never fed back to
    hardware, so it does not affect the raw-radian phase contract of the SLM
    pipeline.
    """
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(
            f"to_display_uint8 expects a 2D array, got shape {arr.shape}"
        )
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = np.nan_to_num(
        arr.astype(np.float64), copy=True, nan=0.0, posinf=0.0, neginf=0.0
    )
    peak = float(arr.max()) if arr.size else 0.0
    if peak > 0.0:
        arr = arr / peak * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


class BaseFrame(ABC):
    def __init__(self, window, render_pos, frame_size, title:str="") -> None:
        import pygame
        self.window = window
        self.top, self.left = render_pos
        self.width, self.height = frame_size
        self.title = title

        if title:
            self.top += Title_height
            self.height -= Title_height

        self.plot_area = pygame.Rect(self.left, self.top, self.width, self.height)

    def close(self):
        pass

    def _render_title(self):
        import pygame
        font = pygame.font.Font(None, Title_height-6)
        text = font.render(self.title, True, (0, 255, 255))
        self.window.blit(text, (self.left, self.top - Title_height))

    def render(self):
        self._render_title()

@register_frame("Image2D")
class Image2DFrame(BaseFrame):
    def render(self, img:np.ndarray):
        import pygame
        img = to_display_uint8(img)
        img_h, img_w = img.shape
        zoom_factors = (self.height / img_h, self.width / img_w)
        img = np.asarray(zoom(img, zoom_factors, mode='nearest'))
        img_surf = pygame.surfarray.make_surface(img.transpose())
        self.window.blit(img_surf, (self.left, self.top))
        super().render()


@register_frame("Image2DWithBucket")
class Image2DWithBucketFrame(BaseFrame):
    def render(self, img:np.ndarray, center:tuple[int,int], r:int):
        """Render a 2D image with the bucket circle overlaid.

        ``center`` / ``r`` are given in **source-image** coordinates and scaled
        to the (zoomed) surface internally, so callers never need to know the
        frame geometry.
        """
        import pygame
        img = to_display_uint8(img)
        img_h, img_w = img.shape
        zoom_factors = (self.height / img_h, self.width / img_w)
        img = np.asarray(zoom(img, zoom_factors, mode='nearest'))
        img_surf = pygame.surfarray.make_surface(img.transpose())
        scale_x = self.width / max(int(img_w), 1)
        scale_y = self.height / max(int(img_h), 1)
        _center = (int(round(center[0] * scale_x)), int(round(center[1] * scale_y)))
        _r = max(1, int(round(r * scale_x)))
        pygame.draw.circle(img_surf, (255, 0, 0), _center, _r, 3)
        self.window.blit(img_surf, (self.left, self.top))
        super().render()


@register_frame("Voltage")
class VoltageFrame(BaseFrame):
    def __init__(self, v_min:int=-300, v_max:int=500, background_color=BACKGROUND_COLOR, **kwargs) -> None:
        super().__init__(**kwargs)
        self.background_color = background_color
        self.v_min = v_min
        self.v_max = v_max

        max_hight_ratio, total_scale = self.height / max(abs(v_max),abs(v_min)), (self.v_max - self.v_min)
        self.v_norm = lambda v: (v - self.v_min) / total_scale
        self.v_hight = lambda v: int(v * max_hight_ratio)

    def render(self, volts):
        import pygame
        _volts = np.clip(volts, self.v_min, self.v_max)
        self.window.fill(self.background_color, self.plot_area)
        bar_width = int(self.width / len(_volts))
        for i,v in enumerate(_volts):
            normed_v = self.v_norm(v)
            color = (int(normed_v*255), int((1-normed_v)*255), 0)
            x = int(self.left + i * bar_width)
            y = int(self.top + self.height/2)
            height = self.v_hight(v)
            pygame.draw.line(self.window, color, (x, y), (x, y - height), bar_width)
        super().render()


@register_frame("Log")
class LogFrame(BaseFrame):
    Line_Coler = (0, 255, 0)

    def __init__(self, background_color=BACKGROUND_COLOR, **kwargs) -> None:
        super().__init__(**kwargs)
        self.background_color = background_color
        self.__recorder = []

    def render(self, value):
        import pygame
        self.window.fill(self.background_color, self.plot_area)
        self.__recorder.append(value)
        if len(self.__recorder) > 1:
            min_sum = min(self.__recorder)
            max_sum = max(self.__recorder)
            points = []
            num_points = len(self.__recorder)
            for i, sum_value in enumerate(self.__recorder):
                # 均匀分布 x 轴坐标
                x = self.left + int(i * (self.width / (num_points - 1)))
                y = self.top + self.height - int(
                    (sum_value - min_sum) / (max_sum - min_sum) * self.height
                ) if max_sum != min_sum else self.height // 2
                points.append((x, y))
            pygame.draw.lines(self.window, self.Line_Coler, False, points, 2)
        super().render()

    def close(self):
        self.__recorder.clear()

    @property
    def data(self):
        return self.__recorder


@register_frame("Text")
class TextFrame(BaseFrame):
    def __init__(self, font_size:int=0, background_color=BACKGROUND_COLOR, **kwargs) -> None:
        super().__init__(**kwargs)
        self.font_size = font_size
        self.background_color = background_color
        if self.font_size == 0:
            logger.info("Font size not specified, using dynamic font size")

    def render(self, text:str, font_size:int=0):
        import pygame
        self.window.fill(self.background_color, self.plot_area)

        lines = text.splitlines()
        if font_size != 0:
            self.font_size = font_size

        if self.font_size == 0:
            max_line_len = max(len(line) for line in lines)
            font_pixel = min(self.height // len(lines)-6, self.width // max_line_len-6)
            self.font_size = self.__font_pixel_to_pt(font_pixel)

        font = pygame.font.Font(None, self.font_size)
        for i, line in enumerate(lines):
            text = font.render(line, True, (0, 255, 255))
            self.window.blit(text, (self.left, self.top + i * (font.get_height()+3)))
        super().render()

    @staticmethod
    def __font_pt_to_pixel(pt:int):
        return int(pt * DPI / 72)

    @staticmethod
    def __font_pixel_to_pt(px:int):
        return int(px * 72 / DPI)


@register_frame("EpochCurve")
class EpochCurveFrame(BaseFrame):
    """Line plot of a per-epoch metric whose x-axis spans ``total_epochs``.

    Each render records (or updates) one point at ``x = epoch / total_epochs`` of
    the panel width, so the curve advances left-to-right in proportion to the
    configured epoch count instead of being re-spread as more points arrive.
    With ``total_epochs=None`` the x-axis falls back to even spacing over the
    observed history.

    The newest value is marked with a dot and the best (lowest) value with a
    second dot; an optional text label is drawn in the top-left corner.
    """

    LINE_COLOR = (0, 255, 0)
    POINT_COLOR = (255, 255, 0)
    BEST_COLOR = (255, 64, 64)
    TEXT_COLOR = (0, 255, 255)
    LABEL_FONT_SIZE = 18
    MAX_POINTS = 10000

    def __init__(
        self,
        y_min: float | None = None,
        y_max: float | None = None,
        y_pad: float = 0.05,
        background_color=BACKGROUND_COLOR,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.background_color = background_color
        self.y_min = y_min
        self.y_max = y_max
        self.y_pad = float(y_pad)
        self._epochs: list[float] = []
        self._values: list[float] = []
        self._label_font = None

    @property
    def data(self) -> dict:
        """Recorded history: ``{"epoch": [...], "value": [...]}``."""
        return {"epoch": list(self._epochs), "value": list(self._values)}

    def close(self):
        self._epochs.clear()
        self._values.clear()

    def _x_of_epoch(self, epoch: float, total_epochs: int | None, n_points: int) -> int:
        """Map an epoch to a pixel x inside the plot area.

        With ``total_epochs`` set the mapping is fixed: ``epoch / total_epochs``
        of the panel width, so the same epoch always lands on the same x. Without
        it, points are spread evenly over the observed history.
        """
        x0, x1 = self.left, self.left + self.width - 1
        if total_epochs and total_epochs > 0:
            frac = float(epoch) / float(total_epochs)
        else:
            frac = float(epoch) / max(n_points - 1, 1)
        return max(x0, min(x1, int(round(x0 + frac * (x1 - x0)))))

    def _y_range(self) -> tuple[float, float]:
        lo = self.y_min
        hi = self.y_max
        if self._values:
            if lo is None:
                lo = min(self._values)
            if hi is None:
                hi = max(self._values)
        if lo is None:
            lo = 0.0
        if hi is None:
            hi = 1.0
        if hi <= lo:
            hi = lo + 1.0
        pad = (hi - lo) * self.y_pad
        return (
            lo if self.y_min is not None else lo - pad,
            hi if self.y_max is not None else hi + pad,
        )

    def render(self, value=None, epoch=None, total_epochs=None, label: str = ""):
        """Draw the curve.

        Args:
            value: New metric value to record; ``None`` redraws without adding.
            epoch: Epoch (x) value for ``value``; defaults to the history length.
            total_epochs: When set, x = ``epoch / total_epochs`` x panel width.
            label: Optional text drawn in the top-left corner.
        """
        import pygame

        if value is not None:
            _epoch = float(epoch) if epoch is not None else float(len(self._epochs))
            if self._epochs and _epoch == self._epochs[-1]:
                # Re-render for the same epoch: replace, don't duplicate the point.
                self._values[-1] = float(value)
            else:
                self._epochs.append(_epoch)
                self._values.append(float(value))
                if len(self._epochs) > self.MAX_POINTS:
                    self._epochs.pop(0)
                    self._values.pop(0)

        self.window.fill(self.background_color, self.plot_area)

        if self._values:
            y_lo, y_hi = self._y_range()
            span = max(y_hi - y_lo, 1e-12)
            x0, x1 = self.left, self.left + self.width - 1
            y0, y1 = self.top, self.top + self.height - 1
            n = len(self._values)

            def x_of(e: float) -> int:
                return self._x_of_epoch(e, total_epochs, n)

            def y_of(v: float) -> int:
                y = int(round(y1 - (float(v) - y_lo) / span * (y1 - y0)))
                return max(y0, min(y1, y))

            points = [(x_of(e), y_of(v)) for e, v in zip(self._epochs, self._values)]
            if len(points) > 1:
                pygame.draw.lines(self.window, self.LINE_COLOR, False, points, 2)
            pygame.draw.circle(self.window, self.POINT_COLOR, points[-1], 3)

            best_idx = int(np.argmin(self._values))
            pygame.draw.circle(
                self.window,
                self.BEST_COLOR,
                (x_of(self._epochs[best_idx]), y_of(self._values[best_idx])),
                3,
            )

            if label:
                if self._label_font is None:
                    self._label_font = pygame.font.Font(None, self.LABEL_FONT_SIZE)
                self.window.blit(
                    self._label_font.render(label, True, self.TEXT_COLOR),
                    (self.left + 2, self.top + 2),
                )

        super().render()
