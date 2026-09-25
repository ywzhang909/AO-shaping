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
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from loguru import logger

from ao_shaping.config import PATHS
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


class _CameraParams(Protocol):
    cam_type: str


def create_camera(
    camera_type: str | _CameraParams,
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
        camera_type: Registered camera type string (case-insensitive), or a
            parameter object exposing a ``cam_type`` attribute.
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
    if not isinstance(camera_type, str):
        params: Any = camera_type
        camera_type = getattr(params, "cam_type")
        cam_id = getattr(params, "cam_id", cam_id)
        exposure_time_ms = getattr(params, "exposure_time_ms", exposure_time_ms)
        if hasattr(params, "skip_sampling"):
            kwargs["skip_sampling"] = params.skip_sampling

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
def select_exposure_from_flats(
    serial: str | None = None,
    target_max: float = 200.0,
    config_root: Path | None = None,
) -> float | None:
    """Choose the optimal exposure (ms) from the saved flat-field frames.

    The bench's laser power is fixed, so the only code-side lever for SNR is the
    exposure. ``data/ccd_configs/ccd/<serial>/flat-exp-<ms>.jpg`` (the filename IS
    the exposure time) gives the measured ``max`` for a few exposures; fitting
    ``max = k * exposure + b`` yields the exposure that reaches ``target_max`` -
    derived OFFLINE, without touching the laser and without the (settling-prone)
    auto-exposure loop.

    Args:
        serial: camera serial (directory name). When ``None`` the single
            ``flat-exp-*`` directory found under ``config_root`` is used.
        target_max: desired peak grey level (keep below saturation, e.g. 200).
        config_root: defaults to ``<repo>/data/ccd_configs/ccd``.

    Returns:
        The exposure in ms, or ``None`` when no usable flat frames exist.
    """
    import re

    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover - numpy/PIL are project dependencies
        return None

    root = (
        Path(config_root)
        if config_root is not None
        else Path(PATHS.root_dir) / "ccd_configs" / "ccd"
    )
    base = root / serial if serial else None
    candidates = [base] if base is not None and base.is_dir() else []
    if not candidates:
        candidates = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    exposures: list[float] = []
    peaks: list[float] = []
    for folder in candidates:
        for path in sorted(folder.glob("flat-exp-*.jpg")):
            match = re.findall(r"(\d+(?:\.\d+)?)", path.stem)
            if not match:
                continue
            try:
                arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
            except OSError as exc:
                logger.warning(f"cannot read flat frame {path}: {exc}")
                continue
            exposures.append(float(match[0]))
            peaks.append(float(arr.max()))
    if len(exposures) < 2:
        return None
    xs = np.asarray(exposures, dtype=np.float64)
    ys = np.asarray(peaks, dtype=np.float64)
    slope, intercept = np.linalg.lstsq(
        np.vstack([xs, np.ones_like(xs)]).T, ys, rcond=None
    )[0]
    if not np.isfinite(slope) or slope <= 0.0:
        return None
    exposure = (float(target_max) - float(intercept)) / float(slope)
    exposure = float(min(max(exposure, 0.02), 1000.0))
    logger.info(
        "flat-field fit: max = {:.4f}*exp {:+.1f} over {} frames -> {:.1f}ms for "
        "target max {:.0f}",
        slope,
        intercept,
        len(exposures),
        exposure,
        target_max,
    )
    return exposure


def resolve_exposure_ms(
    cam: Any,
    target_max: float = 200.0,
    tolerance: float = 12.0,
    serial: str | None = None,
    max_iterations: int = 8,
    n_sample: int = 6,
) -> tuple[float, float]:
    """Pick an exposure that puts the peak near ``target_max``, without a laser change.

    The saved flat-field frames give only an **initial guess** (they can be stale:
    measured 12x dimmer than the live bench on 2026-09-21), so the guess is always
    validated on the live camera and then **bisected**: a frame above the target
    (including a saturated one, whose true peak is unknown) becomes the upper
    bound, a frame below it the lower bound. Stops when within ``tolerance``, on a
    black frame (the exposure is not the limit), or after ``max_iterations``.

    Returns:
        ``(exposure_ms, measured_peak)``. ``tolerance <= 0`` disables the loop and
        just applies the flat-field guess (or keeps the current exposure).
    """
    guess = select_exposure_from_flats(serial=serial, target_max=target_max)
    if guess is None:
        guess = get_camera_exposure_ms(cam)
        logger.info("no flat-field calibration - starting from {:.3f}ms", guess)
    guess = float(min(max(guess, 0.02), 1000.0))
    lo = 0.0
    hi: float | None = None
    peak = float("nan")
    for attempt in range(1, max(1, max_iterations) + 1):
        set_camera_exposure_ms(cam, guess)
        img = cam.get_numpy_image(max(1, n_sample))
        peak = float(np.max(img))
        logger.info(
            "exposure probe {}/{}: {:.3f}ms -> max={:.0f} (target {:.0f}+/-{:.0f})",
            attempt,
            max_iterations,
            guess,
            peak,
            target_max,
            tolerance,
        )
        if tolerance <= 0:
            break
        if peak <= 1.0:
            # Black frame: the exposure is not the limit - do not keep cranking it.
            logger.warning(
                "peak {:.0f} at {:.3f}ms is essentially black; leaving the exposure "
                "here (check the laser / beam path)",
                peak,
                guess,
            )
            break
        if abs(peak - target_max) <= tolerance:
            break
        if peak > target_max:
            hi = guess
        else:
            lo = guess
        if hi is not None:
            nxt = (lo + hi) / 2.0 if lo > 0.0 else hi / 2.0
        else:
            nxt = guess * target_max / peak
        nxt = float(min(max(nxt, 0.02), 1000.0))
        if abs(nxt - guess) < 1e-6:
            break
        guess = nxt
    return guess, peak


def get_camera_exposure_ms(cam: BaseCamera) -> float:
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


# ---------------------------------------------------------------------------
# Exposure-mode resolution + capture
#
# Consolidates the "fixed / auto / keep" exposure dispatch that previously
# appeared as ad-hoc if/elif/else blocks in every wfless optimizer
# (slm_zernike_pib.py, pib.py, slm_square_shaping.py, combined_optimizer.py).
# ---------------------------------------------------------------------------


def resolve_initial_exposure(
    exposure_time_ms: float, target_max_brightness: float
) -> tuple[str, float]:
    """Decide the initial exposure action (precedence: fixed > auto > keep).

    Returns one of:
        ``("fixed", ms)`` — use a fixed exposure time;
        ``("auto", target)`` — auto-expose to the target peak brightness;
        ``("keep", 0.0)`` — leave the exposure as-is (no auto-adjust).
    """
    if exposure_time_ms > 0:
        return ("fixed", float(exposure_time_ms))
    if target_max_brightness > 0:
        return ("auto", float(target_max_brightness))
    return ("keep", 0.0)


def capture_with_exposure(
    cam: Any,
    exposure_time_ms: float = 0.0,
    target_max_brightness: float = 0.0,
    n_sample: int = 1,
    auto_exposure_fn: Any | None = None,
    auto_n_sample: int | None = None,
) -> np.ndarray:
    """Set the exposure (fixed / auto / keep) and capture one image.

    Precedence: a positive ``exposure_time_ms`` wins; otherwise a positive
    ``target_max_brightness`` triggers auto-exposure; otherwise the camera's
    current exposure is left unchanged.

    Args:
        cam: An opened camera driver (exposes ``get_numpy_image`` and
            ``auto_exposure`` or the common helpers).
        exposure_time_ms: Fixed exposure in ms (0 = auto or keep).
        target_max_brightness: Auto-exposure target peak (0-255 grayscale; 0 = keep).
        n_sample: Frames averaged per fixed-exposure / keep capture.
        auto_exposure_fn: Override for the auto-exposure implementation. Defaults
            to :func:`auto_exposure` from this module. Pass ``cam.auto_exposure``
            to use the backend-native path directly.
        auto_n_sample: Frame sample count for the auto-exposure path (defaults
            to ``n_sample``). Useful when auto-exposure needs more frames than
            the fixed-exposure capture (e.g. 20 vs 1).

    Returns:
        The captured image as a NumPy array.
    """
    mode, value = resolve_initial_exposure(exposure_time_ms, target_max_brightness)
    if mode == "fixed":
        set_camera_exposure_ms(cam, value)
        return np.asarray(cam.get_numpy_image(max(1, n_sample)))
    if mode == "auto":
        fn = auto_exposure_fn if auto_exposure_fn is not None else auto_exposure
        return np.asarray(fn(cam, value, n_sample=auto_n_sample or n_sample))
    return np.asarray(cam.get_numpy_image(max(1, n_sample)))


def resample_on_saturation(
    img: np.ndarray,
    cam: Any,
    exposure_time_ms: float = 0.0,
    target_max_brightness: float = 0.0,
    auto_exposure_fn: Any | None = None,
    saturation_threshold: float = 255.0,
    n_sample: int = 1,
) -> np.ndarray:
    """Re-capture at a lower exposure if the image is saturated.

    Mirrors the guard found in every SPGD loop: when auto-exposure is active
    (``exposure_time_ms == 0``) and the peak hits ``saturation_threshold``,
    re-auto-expose to ``target_max_brightness`` (falling back to the original
    image data if auto-exposure is unavailable or the target is also 0).

    Args:
        img: The image to check (uint8 or float).
        cam: An opened camera driver.
        exposure_time_ms: Current fixed exposure (0 = auto mode, eligible for re-exposure).
        target_max_brightness: Auto-exposure target peak (0-255; 0 = use 220).
        auto_exposure_fn: Override for the auto-exposure implementation.
        saturation_threshold: Peak value considered saturated (default 255).
        n_sample: Frames averaged per re-capture.

    Returns:
        The (possibly resampled) image.
    """
    if exposure_time_ms > 0:
        return img
    if float(np.max(img)) < saturation_threshold:
        return img
    target = target_max_brightness if target_max_brightness > 0 else 220.0
    fn = auto_exposure_fn if auto_exposure_fn is not None else auto_exposure
    logger.info(
        "image saturated (peak={:.0f}); auto-re-exposing to target {:.0f}",
        float(np.max(img)),
        target,
    )
    return np.asarray(fn(cam, target, n_sample=max(1, n_sample)))
