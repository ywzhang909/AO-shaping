"""Shared camera utilities: exposure/window dataclasses, camera-type registry
and camera-agnostic exposure helpers.

The camera registry lets callers obtain a concrete camera **instance by type and
id without opening it** (``create_camera("miicam", cam_id=0)``), which is the
counterpart of ``drivers/dm/_registry.create_dm`` for the CCD layer. Backends are
resolved lazily (``module:attr`` strings) so that a missing SDK only fails when
that specific type is requested, and so that importing this module never creates
an import cycle with the backend modules that import ``ExposureTime`` from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera


@dataclass
class ExposureTime:
    current: float
    _max: float = 100_000.0
    _min: float = 0.0
    unit = "ms"

    @classmethod
    def build(cls, time_str: str):
        """
        从字符串构建 ExposureTime 对象。

        参数:
        time_str (str): 形如 "50ms" 或 "0.05s" 的字符串，表示曝光时间。

        返回:
        ExposureTime: 构建的 ExposureTime 对象。
        """
        time_str = time_str.strip().lower()
        if time_str.endswith("ms"):
            current = float(time_str[:-2])
        elif time_str.endswith("s"):
            current = float(time_str[:-1]) * 1000
        else:
            raise ValueError("Invalid time string format. Use 'ms' or 's' suffix.")
        return cls(
            current=current,
            _max=current if current > 0 else 100_000.0,
            _min=0.0,
        )

    def __str__(self) -> str:
        return f"ExposureTime(current={self.current}{self.unit}, max={self.max}{self.unit}, min={self.min}{self.unit})"

    @property
    def ms(self):
        return self.current

    @ms.setter
    def ms(self, value):
        assert self.min <= value <= self.max, (
            f"Exposure time must be between {self.min}ms and {self.max}ms"
        )
        self.current = float(value)

    @property
    def s(self):
        return self.current / 1000

    @s.setter
    def s(self, value):
        self.ms = float(value * 1000)

    @property
    def max(self):
        assert self._max > 0, "Max exposure time must be set"
        return self._max

    @max.setter
    def max(self, value):
        assert value > 0, "Max exposure time must be positive"
        self._max = float(value)

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        assert value >= 0, "Min exposure time must be non-negative"
        self._min = float(value)


@dataclass
class WindowSize:
    width: int
    height: int
    max_width: int
    max_height: int

    inc: int


# ---------------------------------------------------------------------------
# Camera-type registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraSpec:
    """Registry entry describing one camera backend.

    Attributes:
        name: Registered type name (lower-case key).
        target: Lazy import target as ``"module:attr"``.
        accepted_kwargs: Constructor keyword allow-list; ``None`` forwards all
            keyword arguments unchanged.
    """

    name: str
    target: str
    accepted_kwargs: frozenset[str] | None = None


# Built-in backends. Targets are lazy ``module:attr`` strings so a missing SDK
# (e.g. gxipy for Daheng) only raises when that type is actually requested.
CAMERA_TYPES: dict[str, CameraSpec] = {
    "daheng": CameraSpec(
        "daheng",
        "ao_shaping.drivers.ccd.daheng:DahengCamera",
        frozenset({"cam_id", "exposure_time_ms", "skip_sampling"}),
    ),
    "miicam": CameraSpec(
        "miicam",
        "ao_shaping.drivers.ccd.miicam.driver:MIICamera",
        frozenset(
            {"cam_id", "exposure_time_ms", "skip_sampling", "bit_depth", "capture_mode"}
        ),
    ),
    "ffmpeg": CameraSpec(
        "ffmpeg",
        "ao_shaping.drivers.ccd.ffmpeg:FFmpegCamera",
        frozenset({"cam_id", "exposure_time_ms", "skip_sampling"}),
    ),
    "image_folder": CameraSpec(
        "image_folder",
        "ao_shaping.drivers.ccd.ffmpeg:ImageFolderCamera",
        frozenset({"cam_id", "exposure_time_ms", "skip_sampling"}),
    ),
}


def register_camera(
    name: str,
    target: str | type,
    accepted_kwargs: set[str] | frozenset[str] | None = None,
) -> None:
    """Register a camera backend under ``name`` (case-insensitive).

    Args:
        name: Type name used by :func:`create_camera` / :func:`list_camera_types`.
        target: Either a class or a lazy ``"module:attr"`` string. A class is
            converted to its ``module:qualname`` form so resolution stays lazy.
        accepted_kwargs: Optional constructor keyword allow-list. ``None`` (the
            default) forwards every keyword argument unchanged.

    Raises:
        TypeError: ``target`` is neither a class nor a string.
    """
    if isinstance(target, str):
        target_str = target
    elif isinstance(target, type):
        target_str = f"{target.__module__}:{target.__qualname__}"
    else:
        raise TypeError(
            f"target must be a class or 'module:attr' string, got {type(target)!r}"
        )

    CAMERA_TYPES[name.lower()] = CameraSpec(
        name.lower(),
        target_str,
        frozenset(accepted_kwargs) if accepted_kwargs is not None else None,
    )
    logger.debug("Registered camera type {!r} -> {}", name, target_str)


def list_camera_types() -> list[str]:
    """Return the sorted list of registered camera type names."""
    return sorted(CAMERA_TYPES.keys())


def _resolve_camera_class(target: str) -> type:
    module_path, sep, attr = target.partition(":")
    if not sep or not attr:
        raise ValueError(f"Invalid camera target {target!r}; expected 'module:attr'")
    module = import_module(module_path)
    return getattr(module, attr)


def create_camera(
    camera_type: str,
    cam_id: int | str = 0,
    exposure_time_ms: float = 20.0,
    **kwargs: Any,
) -> BaseCamera:
    """Create (but do **not** open) a camera instance by type and id.

    The returned driver is fully configured but not streaming: call
    ``cam.open()`` (or use it as a context manager) before capturing. This keeps
    construction side-effect free and symmetric with the DM registry's
    ``create_dm``.

    Args:
        camera_type: Registered camera type, e.g. ``"daheng"``, ``"miicam"``,
            ``"ffmpeg"`` or ``"image_folder"`` (case-insensitive).
        cam_id: Device index (``int``) or file/folder path/URL (``str``) for the
            file-based backends.
        exposure_time_ms: Initial exposure time in milliseconds.
        **kwargs: Backend-specific options. Only keys accepted by the backend are
            forwarded (e.g. ``bit_depth`` for ``miicam``).

    Returns:
        A ``BaseCamera`` subclass instance, not yet opened.

    Raises:
        ValueError: Unknown camera type.
        ImportError: The backend driver/SDK is unavailable.
    """
    key = str(camera_type).lower()
    spec = CAMERA_TYPES.get(key)
    if spec is None:
        raise ValueError(
            f"Unknown camera type: {camera_type!r}. Available: {list_camera_types()}"
        )

    kwargs = dict(kwargs)
    kwargs.setdefault("cam_id", cam_id)
    kwargs.setdefault("exposure_time_ms", exposure_time_ms)
    if spec.accepted_kwargs is not None:
        kwargs = {k: v for k, v in kwargs.items() if k in spec.accepted_kwargs}

    logger.debug("Creating camera type={} cam_id={}", key, cam_id)
    cls = _resolve_camera_class(spec.target)
    try:
        return cls(**kwargs)
    except NameError as exc:
        # Some backends (notably Daheng) swallow their SDK ImportError at module
        # import time, so an absent SDK surfaces only here as a NameError on the
        # undefined SDK handle. Normalise it so callers can uniformly catch
        # ImportError for "backend unavailable".
        raise ImportError(
            f"Camera backend {key!r} is unavailable (missing SDK): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Camera-agnostic exposure helpers
#
# Backends disagree on the exposure API: DahengCamera exposes an ``exposure_time``
# property (ms) and a native ``auto_exposure(target_max=...)`` (target 0-255),
# while MIICamera exposes an ``exposure_time_ms`` attribute and only
# ``reset_exposure_time(ms)`` (a Stop -> set -> Start cycle). These helpers let
# the optimizer layer treat both uniformly; all values are in **milliseconds**.
# ---------------------------------------------------------------------------
def get_camera_exposure_ms(cam: Any) -> float:
    """Read the current exposure time in ms from any supported backend.

    Raises:
        AttributeError: The camera exposes neither ``exposure_time_ms`` nor
            ``exposure_time``.
    """
    for attr in ("exposure_time_ms", "exposure_time"):
        value = getattr(cam, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    raise AttributeError(
        f"{type(cam).__name__} exposes no exposure time attribute "
        "(tried 'exposure_time_ms', 'exposure_time')"
    )


def get_camera_exposure_range(cam: Any) -> tuple[float, float]:
    """Return the ``(min_ms, max_ms)`` exposure range of a camera.

    Prefers the backend's own ``get_exposure_range()`` (DahengCamera: the device's
    real range, e.g. ``(0.02, 1000.0)``), then the MiiCam-style
    ``min_exposure_ms``/``max_exposure_ms`` properties, and only then the
    cross-driver fallback documented by ``BaseCamera`` (``0.011`` .. ``10000``).
    """
    # Prefer the backend's own range query (DahengCamera.get_exposure_range
    # returns the device's real [min, max] in ms — the properties below are
    # MiiCam-only and otherwise fall back to the generic range).
    getter = getattr(cam, "get_exposure_range", None)
    if callable(getter):
        rng = getter()
        if isinstance(rng, (tuple, list)) and len(rng) == 2 and rng[1] > rng[0]:
            return float(rng[0]), float(rng[1])

    lo = getattr(cam, "min_exposure_ms", None)
    hi = getattr(cam, "max_exposure_ms", None)
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and hi > lo:
        return float(lo), float(hi)
    return 0.011, 10_000.0


def set_camera_exposure_ms(cam: Any, time_ms: float) -> float:
    """Set the exposure time in ms, using the backend's preferred path.

    Uses ``reset_exposure_time`` when available (required by MIICAM's
    Stop -> set -> Start cycle, and equivalent for the other backends); otherwise
    writes the ``exposure_time_ms`` / ``exposure_time`` attribute directly.

    Returns:
        The exposure time that was requested, in ms.
    """
    reset = getattr(cam, "reset_exposure_time", None)
    if callable(reset):
        reset(float(time_ms))
        return float(time_ms)
    if hasattr(cam, "exposure_time_ms"):
        cam.exposure_time_ms = float(time_ms)
        return float(time_ms)
    if hasattr(cam, "exposure_time"):
        cam.exposure_time = float(time_ms)
        return float(time_ms)
    raise AttributeError(
        f"{type(cam).__name__} exposes no writable exposure time "
        "(tried 'reset_exposure_time', 'exposure_time_ms', 'exposure_time')"
    )


def auto_exposure(
    cam: Any,
    target_max: float,
    tolerance: float = 5.0,
    max_iterations: int = 20,
    n_sample: int = 1,
) -> np.ndarray:
    """Auto-adjust exposure to reach ``target_max`` peak brightness and return the frame.

    ``target_max`` / ``tolerance`` are on the **0-255 grayscale** scale, the
    unified contract implemented by both hardware backends
    (:meth:`DahengCamera.auto_exposure` and :meth:`MIICamera.auto_exposure`).
    When the backend provides a native ``auto_exposure`` it is delegated to;
    otherwise (ffmpeg / image-folder backends) a generic proportional loop is used.

    Args:
        cam: An opened camera driver.
        target_max: Target peak pixel value (0-255).
        tolerance: Absolute peak tolerance in grayscale counts (default 5).
        max_iterations: Maximum proportional adjustments for the generic loop.
        n_sample: Frames averaged per measurement.

    Returns:
        The captured image after the adjustment (or the best-effort last frame).
    """
    native = getattr(cam, "auto_exposure", None)
    if callable(native):
        return np.asarray(native(target_max=target_max, n_sample=n_sample))

    target = float(target_max)
    lo, hi = get_camera_exposure_range(cam)
    img = cam.get_numpy_image(n_sample)

    for _ in range(max(1, int(max_iterations))):
        peak = float(np.max(img))
        if peak <= 0.0 or target <= 0.0:
            break
        if abs(peak - target) <= tolerance:
            break
        current = get_camera_exposure_ms(cam)
        if current <= 0.0:
            break
        new_exp = float(np.clip(current * (target / peak), lo, hi))
        if abs(new_exp - current) < 1e-9:
            break
        set_camera_exposure_ms(cam, new_exp)
        img = cam.get_numpy_image(n_sample)

    return img
