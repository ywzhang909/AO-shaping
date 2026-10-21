from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.gui.slm import pyarrow_probe
from ao_shaping.utils.wavefront.pattern_helper import (
    PatternHelper,
    calc_blazed_grating_period,
)
from ao_shaping.utils.wavefront.zernike_calc import get_zernike_name, zernike_modes


class PatternControl(ABC):
    """Base class for a single pattern-type UI control.

    Each subclass carries its SLM context (id, wavelength, pixel pitch, panel
    size, bit depth) in ``__init__`` — ``render()`` builds the Streamlit
    widgets from these values (NOT from ``st.session_state``), and
    ``generate_phase_rad()`` turns the returned params into the raw radian
    phase image. ``slm_id`` prefixes all widget keys, so rendering several SLMs
    on one page never collides.

    A :class:`PatternHelper` may be injected via ``__init__(helper=...)`` so
    every pattern type on one SLM device shares a single helper (coordinate
    grids + Zernike generator caches) instead of building one per control per
    rerun; when omitted the control lazily builds and reuses its own.
    """

    #: Per-control widget defaults — subclasses declare every interactive
    #: widget's default value here; context-derived values (panel size,
    #: wavelength, pixel pitch, bit depth) are overridden in the subclass's own
    #: ``__init__`` after ``super().__init__()``.
    DEFAULTS: ClassVar[dict[str, Any]] = {}

    #: Per-control widget (min, max) ranges — keys mirror ``DEFAULTS``.
    RANGES: ClassVar[dict[str, Any]] = {}

    #: Widget-key suffixes (relative to ``f"slm{slm_id}_"``) whose default
    #: derives from live SLM context (wavelength or 2π max-grayscale). When the
    #: controller changes that context it pops these keys from
    #: st.session_state so the next rerun re-initializes the widgets — without
    #: this, Streamlit keeps the stale first-render value.
    CONTEXT_SYNCED_WIDGETS: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        slm_id: int = 0,
        wavelength: float = 1064.0,
        pixel_pitch_um: float = 8.0,
        width: int = 1920,
        height: int = 1200,
        bits: int = 10,
        max_gray: int | None = None,
        helper: PatternHelper | None = None,
    ) -> None:
        #: Shared per-device :class:`PatternHelper`. When injected, its panel
        #: geometry (resolution + bit depth) is authoritative and overrides
        #: ``width``/``height``/``bits``; every pattern generation reuses its
        #: cached coordinate grids and Zernike generators. When None one is
        #: built lazily on first use (see :meth:`_new_helper`).
        self._helper = helper
        if helper is not None:
            width, height = helper.resolution
            bits = helper.bits
        self.slm_id = int(slm_id)
        self.wavelength = float(wavelength)
        self.pixel_pitch_um = float(pixel_pitch_um)
        self.width = int(width)
        self.height = int(height)
        self.bits = int(bits)
        #: 2π max-grayscale: the driver's real device value (slm._max_gray,
        #: e.g. 993 @1064nm) when provided, else derived from bit depth.
        self.max_gray = int(max_gray) if max_gray is not None else (1 << self.bits) - 1
        # Instance specs: render() pulls every widget's value/min/max from
        # these (never from st.session_state, never falling back to min_value).
        self.defaults = dict(self.DEFAULTS)
        self.ranges = dict(self.RANGES)

    @property
    def prefix(self) -> str:
        """Streamlit widget-key prefix — unique per SLM id."""
        return f"slm{self.slm_id}"

    @property
    def default_radius(self) -> int:
        """Default aperture radius (half the short panel side)."""
        return min(self.width, self.height) // 2

    def _new_helper(self) -> PatternHelper:
        """Return the :class:`PatternHelper` used by this control.

        Reuses the instance injected via ``__init__(helper=...)`` (the
        controller keeps one shared helper per SLM device in session state),
        or lazily builds and caches one per control instance. Never a
        throwaway per call — the coordinate grids and Zernike generators stay
        reusable across ``render``/``generate_phase_rad`` invocations.
        """
        if self._helper is None:
            self._helper = PatternHelper((self.width, self.height), bits=self.bits)
        return self._helper

    @abstractmethod
    def render(self, prefix: str | None = None) -> dict[str, Any]:
        """Render pattern-specific controls, return params dict."""

    @abstractmethod
    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        """Compute the raw radian phase image (height, width), float64.

        The returned phase is UNWRAPPED (no mod-2π) and NOT converted to
        grayscale — the SLM driver applies radians→grayscale on hardware
        write. Binary patterns (棋盘格/二元光栅/达曼光栅) return exactly
        0.0/π. Flat phases use the ideal interpretation gray/max_gray·2π (the
        hardware write itself bypasses conversion — see :class:`FlatControl`).
        """

    def generate_phase_gray(
        self,
        params: dict[str, Any],
        slm: Santec | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        """Default: convert ``generate_phase_rad`` output via the SLM driver.

        Subclasses with a raw-gray contract (平场, 半半相位) override this
        method.
        """
        if slm is None:
            raise ValueError(f"{type(self).__name__} 生成灰度相位需要 slm 对象")
        return slm.create_phase_from_array(self.generate_phase_rad(params))


class FlatControl(PatternControl):
    """平场 — single grayscale value.

    RAW GRAY path: the Santec SLM has amplitude coupling at 1064nm, so the flat
    phase MUST be written as the literal uint16 gray value, bypassing
    ``create_phase_from_array`` (which would interpret it as radians and
    silently corrupt the value). ``generate_phase_rad`` only exists for API
    uniformity and encodes the ideal radian interpretation gray/max_gray·2π.
    """

    DEFAULTS: ClassVar[dict[str, Any]] = {"flat_gray": 0}
    RANGES: ClassVar[dict[str, Any]] = {"flat_gray": (0, 1023)}
    CONTEXT_SYNCED_WIDGETS: ClassVar[tuple[str, ...]] = ("flat_gray",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(flat_gray=0)
        self.ranges.update(flat_gray=(0, self.max_gray))

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "flat_gray": st.number_input(
                "灰度",
                min_value=self.ranges["flat_gray"][0],
                max_value=self.ranges["flat_gray"][1],
                value=self.defaults["flat_gray"],
                step=1,
                key=f"{prefix}_flat_gray",
            )
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        gray = int(params["flat_gray"])
        phase = gray / self.max_gray * 2.0 * np.pi
        return np.full((self.height, self.width), phase, dtype=np.float64)

    def generate_phase_gray(
        self,
        params: dict[str, Any],
        slm: Santec | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        gray = int(params["flat_gray"])
        return np.full((self.height, self.width), gray, dtype=np.uint16)


class LinearGratingControl(PatternControl):
    """线性光栅 — period and phase range."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "period": 64.0,
        "phase_range": 2.0,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "period": (1.0, 1000.0),
        "phase_range": (0.05, 2.0),
    }

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "period": st.number_input(
                "周期 (像素)",
                min_value=self.ranges["period"][0],
                max_value=self.ranges["period"][1],
                value=self.defaults["period"],
                step=1.0,
                key=f"{prefix}_linear_period",
            ),
            "phase_range": st.number_input(
                "相位范围 (π)",
                min_value=self.ranges["phase_range"][0],
                max_value=self.ranges["phase_range"][1],
                value=self.defaults["phase_range"],
                step=0.1,
                key=f"{prefix}_linear_phase_range",
                help="输入 N 表示 Nπ 弧度，发送时自动乘以 π",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        # phase_range is in units of π; multiply by π to get radians
        phase_range_rad = float(params["phase_range"]) * np.pi
        return self._new_helper().linear_grating(
            period=float(params["period"]),
            phase_range=phase_range_rad,
        )


class HologramGratingControl(PatternControl):
    """全息光栅 — period and phase range."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "period": 64.0,
        "phase_range": 2.0,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "period": (1.0, 1000.0),
        "phase_range": (0.05, 2.0),
    }

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "period": st.number_input(
                "周期 (像素)",
                min_value=self.ranges["period"][0],
                max_value=self.ranges["period"][1],
                value=self.defaults["period"],
                step=1.0,
                key=f"{prefix}_hologram_period",
            ),
            "phase_range": st.number_input(
                "相位范围 (π)",
                min_value=self.ranges["phase_range"][0],
                max_value=self.ranges["phase_range"][1],
                value=self.defaults["phase_range"],
                step=0.1,
                key=f"{prefix}_hologram_phase_range",
                help="输入 N 表示 Nπ 弧度，发送时自动乘以 π",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        # phase_range is in units of π; multiply by π to get radians
        phase_range_rad = float(params["phase_range"]) * np.pi
        return self._new_helper().hologram(
            period=float(params["period"]),
            phase_range=phase_range_rad,
        )


class BlazedGratingControl(PatternControl):
    """闪耀光栅 — with angle/direct mode and direction."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "period": 10.0,
        "phase_range": 2.0,
        "direction": "vertical",
        "angle_deg": 10.0,
        "calc_wl": 1064,
        "calc_pitch": 8.0,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "period": (1.0, 10000.0),
        "phase_range": (0.05, 2.0),
        "angle_deg": (0.1, 89.0),
        "calc_wl": (400, 1600),
        "calc_pitch": (0.1, 100.0),
    }
    CONTEXT_SYNCED_WIDGETS: ClassVar[tuple[str, ...]] = ("blazed_calc_wl",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(
            calc_wl=int(self.wavelength),
            calc_pitch=self.pixel_pitch_um,
        )

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        period_mode = st.radio(
            "周期设置方式",
            options=["direct", "angle"],
            format_func=lambda x: (
                "直接设置周期(像素)" if x == "direct" else "根据衍射角度和波长计算"
            ),
            key=f"{prefix}_blazed_period_mode",
            horizontal=True,
            label_visibility="collapsed",
        )

        if period_mode == "angle":
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                angle_deg = st.number_input(
                    "衍射角度 θ (度)",
                    min_value=self.ranges["angle_deg"][0],
                    max_value=self.ranges["angle_deg"][1],
                    value=self.defaults["angle_deg"],
                    step=0.5,
                    key=f"{prefix}_blazed_calc_angle",
                )
            with col_a2:
                calc_wl = st.number_input(
                    "波长 λ (nm)",
                    min_value=self.ranges["calc_wl"][0],
                    max_value=self.ranges["calc_wl"][1],
                    value=self.defaults["calc_wl"],
                    step=1,
                    key=f"{prefix}_blazed_calc_wl",
                )

            calc_pitch = st.number_input(
                "像素间距 (μm)",
                min_value=self.ranges["calc_pitch"][0],
                max_value=self.ranges["calc_pitch"][1],
                value=self.defaults["calc_pitch"],
                step=0.1,
                key=f"{prefix}_blazed_calc_pitch",
            )

            period_pixels = calc_blazed_grating_period(angle_deg, calc_wl, calc_pitch)
            # Deliberate widget-state write: keeps the direct-mode period widget
            # in sync with the freshly computed value (not a default read).
            st.session_state[f"{prefix}_blazed_period"] = period_pixels

            st.metric(
                "计算周期",
                f"{period_pixels:.1f} 像素",
                help=f"d = λ / sin(θ)，像素间距 {calc_pitch} μm",
            )
            period = period_pixels
        else:
            period = st.number_input(
                "周期 (像素)",
                min_value=self.ranges["period"][0],
                max_value=self.ranges["period"][1],
                value=self.defaults["period"],
                step=1.0,
                key=f"{prefix}_blazed_period",
            )

        phase_range = st.number_input(
            "相位范围 (π)",
            min_value=self.ranges["phase_range"][0],
            max_value=self.ranges["phase_range"][1],
            value=self.defaults["phase_range"],
            step=0.1,
            key=f"{prefix}_blazed_phase_range",
            help="输入 N 表示 Nπ 弧度，发送时自动乘以 π",
        )
        direction = st.selectbox(
            "光栅方向",
            options=["vertical", "horizontal"],
            format_func=lambda x: (
                "竖条纹（垂直）" if x == "vertical" else "横条纹（水平）"
            ),
            key=f"{prefix}_blazed_direction",
        )

        return {
            "period": period,
            "phase_range": phase_range,
            "direction": direction,
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        # phase_range is in units of π; multiply by π to get radians
        phase_range_rad = float(params["phase_range"]) * np.pi
        return self._new_helper().linear_grating(
            period=float(params["period"]),
            phase_range=phase_range_rad,
            direction=str(params["direction"]),
        )


class CircularGratingControl(PatternControl):
    """圆形光栅 — radius and phase range."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "radius": 64.0,
        "phase_range": 2.0,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "radius": (1.0, 2000.0),
        "phase_range": (0.05, 2.0),
    }

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "radius": st.number_input(
                "圆形周期半径 (像素)",
                min_value=self.ranges["radius"][0],
                max_value=self.ranges["radius"][1],
                value=self.defaults["radius"],
                step=10.0,
                key=f"{prefix}_circular_radius",
            ),
            "phase_range": st.number_input(
                "相位范围 (π)",
                min_value=self.ranges["phase_range"][0],
                max_value=self.ranges["phase_range"][1],
                value=self.defaults["phase_range"],
                step=0.1,
                key=f"{prefix}_circular_phase_range",
                help="输入 N 表示 Nπ 弧度，发送时自动乘以 π",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        # phase_range is in units of π; multiply by π to get radians
        phase_range_rad = float(params["phase_range"]) * np.pi
        return self._new_helper().circular_grating(
            radius=float(params["radius"]),
            phase_range=phase_range_rad,
        )


class LensControl(PatternControl):
    """透镜 — focal length, pixel pitch, lens radius."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "focal_length_mm": 300.0,
        "pixel_pitch_um": 8.0,
        "lens_radius": 600,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "focal_length_mm": (1.0, 100000.0),
        "pixel_pitch_um": (0.1, 100.0),
        "lens_radius": (1, 2000),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(
            pixel_pitch_um=self.pixel_pitch_um,
            lens_radius=self.default_radius,
        )

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "focal_length_mm": st.number_input(
                "焦距 (mm)",
                min_value=self.ranges["focal_length_mm"][0],
                max_value=self.ranges["focal_length_mm"][1],
                value=self.defaults["focal_length_mm"],
                step=10.0,
                key=f"{prefix}_lens_focal_length",
            ),
            "pixel_pitch_um": st.number_input(
                "像素间距 (um)",
                min_value=self.ranges["pixel_pitch_um"][0],
                max_value=self.ranges["pixel_pitch_um"][1],
                value=self.defaults["pixel_pitch_um"],
                step=0.1,
                key=f"{prefix}_lens_pixel_pitch",
            ),
            "lens_radius": st.number_input(
                "透镜半径 (像素)",
                min_value=self.ranges["lens_radius"][0],
                max_value=self.ranges["lens_radius"][1],
                value=self.defaults["lens_radius"],
                step=1,
                key=f"{prefix}_lens_radius",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        pitch = float(params.get("pixel_pitch_um", self.defaults["pixel_pitch_um"]))
        lens_radius = float(params.get("lens_radius", self.defaults["lens_radius"]))
        lens_radius = lens_radius if lens_radius > 0 else None
        return self._new_helper().lens(
            focal_length=float(params["focal_length_mm"]) * 1e-3,  # mm -> m
            wavelength=self.wavelength * 1e-9,  # nm -> m
            pixel_size=pitch * 1e-6,  # um -> m
            lens_radius=lens_radius,
        )


class CheckerboardControl(PatternControl):
    """棋盘格 — period (binary 0/π phase)."""

    DEFAULTS: ClassVar[dict[str, Any]] = {"period": 50}
    RANGES: ClassVar[dict[str, Any]] = {"period": (1, 1000)}

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "period": st.number_input(
                "棋盘格周期 (像素)",
                min_value=self.ranges["period"][0],
                max_value=self.ranges["period"][1],
                value=self.defaults["period"],
                step=1,
                key=f"{prefix}_checker_period",
            )
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        return self._new_helper().generate_checkerboard(period=int(params["period"]))


class BinaryGratingControl(PatternControl):
    """二元光栅 — a, b, direction (binary 0/π phase)."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "a": 2,
        "b": 3,
        "direction": "horizontal",
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "a": (1, 1000),
        "b": (1, 1000),
    }

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "a": st.number_input(
                "亮条纹宽度 a (像素)",
                min_value=self.ranges["a"][0],
                max_value=self.ranges["a"][1],
                value=self.defaults["a"],
                step=1,
                key=f"{prefix}_binary_a",
            ),
            "b": st.number_input(
                "暗条纹宽度 b (像素)",
                min_value=self.ranges["b"][0],
                max_value=self.ranges["b"][1],
                value=self.defaults["b"],
                step=1,
                key=f"{prefix}_binary_b",
            ),
            "direction": st.selectbox(
                "方向",
                options=["horizontal", "vertical"],
                format_func=lambda x: "水平" if x == "horizontal" else "垂直",
                key=f"{prefix}_binary_direction",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        return self._new_helper().generate_binary_grating(
            a=int(params["a"]),
            b=int(params["b"]),
            direction=str(params["direction"]),
        )


class MicrolensArrayControl(PatternControl):
    """微透镜阵列 — lens size, focal length, pixel pitch."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "lens_size": 16,
        "focal_length_mm": 300.0,
        "pixel_pitch_um": 8.0,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "lens_size": (8, 1000),
        "focal_length_mm": (1.0, 100000.0),
        "pixel_pitch_um": (0.1, 100.0),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(pixel_pitch_um=self.pixel_pitch_um)

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "lens_size": st.number_input(
                "微透镜尺寸 (像素)",
                min_value=self.ranges["lens_size"][0],
                max_value=self.ranges["lens_size"][1],
                value=self.defaults["lens_size"],
                step=1,
                key=f"{prefix}_microlens_size",
            ),
            "focal_length_mm": st.number_input(
                "焦距 (mm)",
                min_value=self.ranges["focal_length_mm"][0],
                max_value=self.ranges["focal_length_mm"][1],
                value=self.defaults["focal_length_mm"],
                step=1.0,
                key=f"{prefix}_microlens_focal_length",
            ),
            "pixel_pitch_um": st.number_input(
                "像素间距 (um)",
                min_value=self.ranges["pixel_pitch_um"][0],
                max_value=self.ranges["pixel_pitch_um"][1],
                value=self.defaults["pixel_pitch_um"],
                step=0.1,
                key=f"{prefix}_microlens_pixel_pitch",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        pitch = float(params.get("pixel_pitch_um", self.defaults["pixel_pitch_um"]))
        return self._new_helper().generate_microlens_array(
            lens_size=int(params["lens_size"]),
            focal_length=float(params["focal_length_mm"]) * 1e-3,
            wavelength=self.wavelength * 1e-9,
            pixel_size=pitch * 1e-6,
        )


class TurbulenceScreenControl(PatternControl):
    """湍流相位屏 — Cn², L, pixel pitch.

    由 Kolmogorov 谱生成相位屏, 每帧独立 (随机种子由 helper 内部管理)。
    """

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "Cn2": 1e-15,
        "L": 1000.0,
        "pixel_pitch_um": 8.0,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "Cn2": (1e-18, 1e-10),
        "L": (0.1, 1e6),
        "pixel_pitch_um": (0.1, 100.0),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(pixel_pitch_um=self.pixel_pitch_um)

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "Cn2": st.number_input(
                "Cn²",
                min_value=self.ranges["Cn2"][0],
                max_value=self.ranges["Cn2"][1],
                value=self.defaults["Cn2"],
                format="%.1e",
                key=f"{prefix}_turbulence_cn2",
            ),
            "L": st.number_input(
                "传播距离 L (m)",
                min_value=self.ranges["L"][0],
                max_value=self.ranges["L"][1],
                value=self.defaults["L"],
                step=10.0,
                key=f"{prefix}_turbulence_length",
            ),
            "pixel_pitch_um": st.number_input(
                "像素间距 (um)",
                min_value=self.ranges["pixel_pitch_um"][0],
                max_value=self.ranges["pixel_pitch_um"][1],
                value=self.defaults["pixel_pitch_um"],
                step=0.1,
                key=f"{prefix}_turbulence_pixel_pitch",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        helper = self._new_helper()
        pixel_scale = (
            float(params.get("pixel_pitch_um", self.defaults["pixel_pitch_um"])) * 1e-6
        )
        # Fried parameter r0 = (0.423·k²·Cn2·L)^(-3/5), k = 2π/λ
        k_wave = 2.0 * np.pi / (self.wavelength * 1e-9)
        Cn2 = float(params["Cn2"])
        L = float(params["L"])
        r0 = (0.423 * k_wave**2 * Cn2 * L) ** (-3.0 / 5.0)
        L0 = 10.0  # outer scale (m), typical atmospheric value
        helper.init_turbulence_screen(r0=r0, L0=L0, pixel_scale=pixel_scale)
        return helper.generate_turbulence_screen()


class VortexPhaseControl(PatternControl):
    """涡旋相位 — topological charge, wavelength, pixel pitch.

    Raw radians (phi = l·θ, no mod-2π, no wrap checkbox — the driver applies
    wrapping + wavefront correction + LUT consistently with all other phase
    patterns).
    """

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "topological_charge": 1,
        "wavelength_nm": 1064,
        "pixel_pitch_um": 8,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "topological_charge": (-10, 10),
        "wavelength_nm": (400, 1600),
        "pixel_pitch_um": (1.0, 100.0),
    }
    CONTEXT_SYNCED_WIDGETS: ClassVar[tuple[str, ...]] = ("vortex_wavelength",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(
            wavelength_nm=int(self.wavelength),
            pixel_pitch_um=self.pixel_pitch_um,
        )

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "topological_charge": st.number_input(
                "拓扑荷",
                min_value=self.ranges["topological_charge"][0],
                max_value=self.ranges["topological_charge"][1],
                value=self.defaults["topological_charge"],
                step=1,
                key=f"{prefix}_vortex_charge",
            ),
            "wavelength_nm": st.number_input(
                "波长 (nm)",
                min_value=self.ranges["wavelength_nm"][0],
                max_value=self.ranges["wavelength_nm"][1],
                value=self.defaults["wavelength_nm"],
                step=1,
                key=f"{prefix}_vortex_wavelength",
            ),
            "pixel_pitch_um": st.number_input(
                "像素间距 (um)",
                min_value=self.ranges["pixel_pitch_um"][0],
                max_value=self.ranges["pixel_pitch_um"][1],
                value=self.defaults["pixel_pitch_um"],
                step=1.0,
                key=f"{prefix}_vortex_pixel_pitch",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        wavelength_m = float(params["wavelength_nm"]) * 1e-9  # nm -> m
        pixel_pitch_m = (
            float(params.get("pixel_pitch_um", self.defaults["pixel_pitch_um"])) * 1e-6
        )  # um -> m
        return self._new_helper().generate_vortex(
            topological_charge=int(params["topological_charge"]),
            wavelength=wavelength_m,
            pixel_size=pixel_pitch_m,
        )


class ZernikeControl(PatternControl):
    """Zernike — n_max, radius, coefficient table."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "n_max": 5,
        "coefficients": {(0, 0): 0.0},
        "radius": 600,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "n_max": (1, 10),
        "radius": (1, 2000),
    }

    #: Bounds (radians) for the coefficient inputs — both editor variants share
    #: them; 峰值相位差远低于此范围, 仅防止误输入导致的失控幅度.
    COEFF_MIN: ClassVar[float] = -100.0
    COEFF_MAX: ClassVar[float] = 100.0

    #: Coefficients laid out per row in the pyarrow-free fallback editor.
    n_columns: ClassVar[int] = 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(radius=self.default_radius)

    def _render_coefficient_table(
        self, prefix: str, modes: list[tuple[int, int]]
    ) -> dict[tuple[int, int], float]:
        """Editable table (``st.data_editor``) — needs a working pyarrow."""
        pairs: list[dict[str, Any]] = [
            {
                "n": n,
                "m": m,
                "name": get_zernike_name(n, m) or f"n={n},m={m}",
                "coeff": 0.0,
            }
            for n, m in modes
        ]

        edited = st.data_editor(
            pairs,
            key=f"{prefix}_zernike_table",
            column_config={
                "n": st.column_config.NumberColumn("n", disabled=True, width="small"),
                "m": st.column_config.NumberColumn("m", disabled=True, width="small"),
                "name": st.column_config.TextColumn(
                    "名称", disabled=True, width="medium"
                ),
                "coeff": st.column_config.NumberColumn(
                    "系数",
                    step=0.001,
                ),
            },
            hide_index=True,
        )

        coefficients: dict[tuple[int, int], float] = {}
        for row in edited:
            coefficients[(int(row["n"]), int(row["m"]))] = float(row["coeff"])
        return coefficients

    def _render_coefficient_grid(
        self, prefix: str, modes: list[tuple[int, int]], reason: str = ""
    ) -> dict[tuple[int, int], float]:
        """pyarrow-free editor: one ``st.number_input`` per Zernike mode.

        Same widget-key prefix scheme as the table variant so switching between
        the two (pyarrow repaired / broken) never collides.
        """
        with st.expander("系数输入 (pyarrow 不可用)", expanded=True):
            st.warning(
                "pyarrow 不可用 — 「st.data_editor 系数表」已回退为逐项输入控件。"
                "修复 pyarrow 后可恢复系数表（pip install --force-reinstall "
                "--only-binary :all: pyarrow）。",
                icon="⚠️",
            )
            if reason:
                st.caption(f"pyarrow 导入失败原因: `{reason}`")
            st.caption("Zernike 系数单位为弧度 (raw, 未包裹), 默认 0.0。")

            coefficients: dict[tuple[int, int], float] = {}
            for start in range(0, len(modes), self.n_columns):
                columns = st.columns(self.n_columns)
                for column, (n, m) in zip(
                    columns, modes[start : start + self.n_columns], strict=True
                ):
                    with column:
                        coefficients[(n, m)] = float(
                            st.number_input(
                                f"{get_zernike_name(n, m)} (n={n}, m={m})",
                                min_value=self.COEFF_MIN,
                                max_value=self.COEFF_MAX,
                                value=0.0,
                                step=0.001,
                                format="%.3f",
                                key=f"{prefix}_zernike_coeff_{n}_{m}",
                            )
                        )
        return coefficients

    # def _render_debug_panel(
    #     self,
    #     n_max: int,
    #     mode_count: int,
    #     editor: str,
    #     editor_reason: str,
    # ) -> None:
    #     """Collapsed diagnostics panel for the Zernike editor selection.

    #     Rendered on every Zernike selection so a failure (or an unexpected
    #     editor backend) can be diagnosed from the UI alone — without hunting
    #     through the Streamlit server log.
    #     """
    #     with st.expander("🔧 Zernike debug 信息 (定位问题用)", expanded=False):
    #         st.markdown(
    #             f"- **选择结果**: 已渲染 `Zernike` 控制, n_max={n_max}, "
    #             f"{mode_count} 个模式"
    #         )
    #         st.markdown(f"- **编辑器路径**: `{editor}`")
    #         for label, value in pyarrow_probe.pyarrow_diagnostics():
    #             st.markdown(f"- **{label}**: `{value}`")
    #         if editor_reason:
    #             st.code(editor_reason, language="text")

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        n_max = st.number_input(
            "最大径向阶数 N",
            min_value=self.ranges["n_max"][0],
            max_value=self.ranges["n_max"][1],
            step=1,
            value=self.defaults["n_max"],
            key=f"{prefix}_zernike_n_max",
        )

        modes = zernike_modes(int(n_max))
        # Capability probe FIRST: st.data_editor importing a broken/mismatched
        # pyarrow mid-widget aborts the script run and wipes the rest of the
        # page (the "选择 Zernike 后页面自动退出" symptom), so pick the editor
        # variant before calling any of them.
        editor_available, editor_reason = pyarrow_probe.probe_pyarrow()
        editor = (
            "st.data_editor (系数表)"
            if editor_available
            else "st.number_input (逐项回退)"
        )
        logger.debug(
            "Zernike render: slm={} n_max={} modes={} editor={} pyarrow={} reason={}",
            self.slm_id,
            int(n_max),
            len(modes),
            editor,
            pyarrow_probe.pyarrow_version() or "不可用",
            editor_reason or "-",
        )

        if editor_available:
            try:
                coefficients = self._render_coefficient_table(prefix, modes)
            except ImportError as exc:
                # pyarrow went missing between the probe and the widget call
                # (or another lazily imported dependency of data_editor is
                # broken) — surface the real traceback in the UI and keep the
                # page usable instead of aborting the script run.
                editor_reason = f"{type(exc).__name__}: {exc}"
                editor = "st.number_input (逐项回退, 系数表渲染失败)"
                logger.exception("st.data_editor 渲染失败 — 回退逐项输入控件")
                st.error(
                    f"st.data_editor 渲染失败, 已回退为逐项输入控件: {editor_reason}"
                )
                st.exception(exc)
                coefficients = self._render_coefficient_grid(
                    prefix, modes, editor_reason
                )
        else:
            coefficients = self._render_coefficient_grid(prefix, modes, editor_reason)

        # self._render_debug_panel(int(n_max), len(modes), editor, editor_reason)

        return {
            "n_max": int(n_max),
            "coefficients": coefficients,
            "radius": float(
                st.number_input(
                    "孔径半径 (像素)",
                    value=self.defaults["radius"],
                    min_value=self.ranges["radius"][0],
                    max_value=self.ranges["radius"][1],
                    step=1,
                    key=f"{prefix}_zernike_radius",
                )
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        raw_coeffs = params.get("coefficients", self.defaults["coefficients"])
        coefficients: dict[tuple[int, int], float] | None = None
        if raw_coeffs is not None and isinstance(raw_coeffs, dict):
            coefficients = {
                k: float(v)
                for k, v in raw_coeffs.items()
                if isinstance(k, tuple) and isinstance(v, (int, float))
            }
        radius = float(params.get("radius", self.defaults["radius"]))
        n_max = int(params.get("n_max", self.defaults["n_max"]))
        # 原始弧度相位 (不 mod-2π, 不归一化) — 驱动 create_phase_from_array 负责
        # 弧度→灰度 (2π=1023 + 波前校正 + LUT), 保留系数绝对幅度 (2026-09)。
        return self._new_helper().generate_zernike_polynomial(
            coefficients=coefficients,
            radius=radius,
            n_max=n_max,
        )


class DammannGratingControl(PatternControl):
    """达曼光栅 — order and fill factor (binary 0/π phase)."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "order": 3,
        "fill_factor": 0.5,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "order": (2, 8),
        "fill_factor": (0.1, 1.0),
    }

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        return {
            "order": st.number_input(
                "衍射级数",
                min_value=self.ranges["order"][0],
                max_value=self.ranges["order"][1],
                value=self.defaults["order"],
                step=1,
                key=f"{prefix}_dammann_order",
            ),
            "fill_factor": st.slider(
                "填充因子",
                min_value=self.ranges["fill_factor"][0],
                max_value=self.ranges["fill_factor"][1],
                value=self.defaults["fill_factor"],
                step=0.1,
                key=f"{prefix}_dammann_fill_factor",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        order = int(params.get("order", self.defaults["order"]))
        fill_factor = float(params.get("fill_factor", self.defaults["fill_factor"]))
        return self._new_helper().generate_dammann_grating(
            order=order, fill_factor=fill_factor
        )


class HalfHalfPhaseControl(PatternControl):
    """半半相位 — flat gray + blazed grating on the other half.

    RAW GRAY half: the flat side bypasses radian conversion (amplitude
    coupling), the blaze side goes through ``slm.create_phase_from_array``.
    ``generate_phase_rad`` returns the equivalent composition in radians.
    """

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "flat_gray": 512,
        "split_direction": "左右",
        "period": 4.0,
        "phase_range": 2.0,
        "blaze_direction": "horizontal",
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "flat_gray": (0, 1023),
        "period": (1.0, 1000.0),
        "phase_range": (0.05, 2.0),
    }
    CONTEXT_SYNCED_WIDGETS: ClassVar[tuple[str, ...]] = ("halfhalf_flat_gray",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(flat_gray=self.max_gray // 2)
        self.ranges.update(flat_gray=(0, self.max_gray))

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        split = st.selectbox(
            "划分方向",
            options=["左右", "上下"],
            format_func=lambda x: (
                "左半平面+右半闪耀光栅" if x == "左右" else "上半平面+下半闪耀光栅"
            ),
            key=f"{prefix}_halfhalf_split",
        )

        params: dict[str, Any] = {
            "flat_gray": st.number_input(
                "平面灰度",
                min_value=self.ranges["flat_gray"][0],
                max_value=self.ranges["flat_gray"][1],
                value=self.defaults["flat_gray"],
                step=1,
                key=f"{prefix}_halfhalf_flat_gray",
            ),
            "split_direction": split,
            "period": st.number_input(
                "光栅周期 (像素)",
                min_value=self.ranges["period"][0],
                max_value=self.ranges["period"][1],
                value=self.defaults["period"],
                step=1.0,
                key=f"{prefix}_halfhalf_period",
            ),
            "phase_range": st.number_input(
                "相位范围 (π)",
                min_value=self.ranges["phase_range"][0],
                max_value=self.ranges["phase_range"][1],
                value=self.defaults["phase_range"],
                step=0.1,
                key=f"{prefix}_halfhalf_phase_range",
                help="输入 N 表示 Nπ 弧度，发送时自动乘以 π",
            ),
        }

        if split == "上下":
            params["blaze_direction"] = st.selectbox(
                "闪耀光栅方向",
                options=["horizontal", "vertical"],
                format_func=lambda x: (
                    "横条纹（水平）" if x == "horizontal" else "竖条纹（垂直）"
                ),
                key=f"{prefix}_halfhalf_blaze_dir",
            )
        else:
            st.caption("左右划分时闪耀光栅固定为竖条纹方向")
            params["blaze_direction"] = "vertical"

        return params

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        flat_gray = int(params["flat_gray"])
        period = float(params["period"])
        # phase_range is in units of π; multiply by π to get radians
        phase_range = float(params["phase_range"]) * np.pi
        split_dir = str(params["split_direction"])
        blaze_dir = str(params.get("blaze_direction", "vertical"))

        blaze_rad = self._new_helper().linear_grating(
            period=period,
            phase_range=phase_range,
            direction=blaze_dir,
        )
        flat_rad = np.full(
            (self.height, self.width),
            flat_gray / self.max_gray * 2.0 * np.pi,
            dtype=np.float64,
        )

        if split_dir == "左右":
            mid = self.width // 2
            return np.concatenate([flat_rad[:, :mid], blaze_rad[:, mid:]], axis=1)
        else:  # 上下
            mid = self.height // 2
            return np.concatenate([flat_rad[:mid, :], blaze_rad[mid:, :]], axis=0)

    def generate_phase_gray(
        self,
        params: dict[str, Any],
        slm: Santec | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        if slm is None:
            raise ValueError("半半相位 生成灰度相位需要 slm 对象")
        flat_gray = int(params["flat_gray"])
        period = float(params["period"])
        # phase_range is in units of π; multiply by π to get radians
        phase_range = float(params["phase_range"]) * np.pi
        split_dir = str(params["split_direction"])
        blaze_dir = str(params.get("blaze_direction", "vertical"))

        # Full-frame blazed grating: radians → grayscale via the SLM driver
        blaze_rad = self._new_helper().linear_grating(
            period=period,
            phase_range=phase_range,
            direction=blaze_dir,
        )
        blaze_gray = slm.create_phase_from_array(blaze_rad)

        # Flat half: direct uint16 grayscale (bypass radian conversion)
        flat_full = np.full((self.height, self.width), flat_gray, dtype=np.uint16)

        # Stitch: half flat + half blaze
        if split_dir == "左右":
            mid = self.width // 2
            return np.concatenate([flat_full[:, :mid], blaze_gray[:, mid:]], axis=1)
        else:  # 上下
            mid = self.height // 2
            return np.concatenate([flat_full[:mid, :], blaze_gray[mid:, :]], axis=0)


PATTERN_REGISTRY: dict[str, type[PatternControl]] = {
    "平场": FlatControl,
    "线性光栅": LinearGratingControl,
    "圆形光栅": CircularGratingControl,
    "透镜": LensControl,
    "全息光栅": HologramGratingControl,
    "闪耀光栅": BlazedGratingControl,
    "棋盘格": CheckerboardControl,
    "二元光栅": BinaryGratingControl,
    "微透镜阵列": MicrolensArrayControl,
    "湍流相位屏": TurbulenceScreenControl,
    "Zernike": ZernikeControl,
    "达曼光栅": DammannGratingControl,
    "涡旋相位": VortexPhaseControl,
    "半半相位": HalfHalfPhaseControl,
}


# ---------------------------------------------------------------------
# Phase generation functions
# ---------------------------------------------------------------------


#: Preview colormap choices shown in the SLM preview selectbox.
PREVIEW_COLORMAP_LABELS: tuple[str, ...] = (
    "相位色图 (twilight)",
    "灰度",
)
PREVIEW_COLORMAP_DEFAULT: str = PREVIEW_COLORMAP_LABELS[0]


def _phase_to_preview(
    phase_gray: np.ndarray,
    max_gray: int | None = None,
    colormap: str | None = None,
) -> np.ndarray:
    """Convert raw SLM grayscale (uint16) to an RGB preview image.

    ``colormap="相位色图 (twilight)"`` (default) maps normalized grayscale
    through the cyclic ``twilight`` colormap — smooth, hue-based rendering that
    keeps fine phase structure visible (0 ≈ 2π wrap at the same color).
    ``colormap="灰度"`` falls back to a plain grayscale normalization
    replicated over RGB. ``max_gray`` is the device's 2π maximum grayscale
    value (``slm._max_gray``); when ``None``, falls back to
    ``Santec.MAX_GRAYSCALE_VALUE`` (full 10-bit range).
    """
    if max_gray is None:
        max_gray = Santec.MAX_GRAYSCALE_VALUE
    max_gray = int(max(1, int(max_gray)))

    normalized = np.clip(
        np.asarray(phase_gray, dtype=np.float32) / float(max_gray),
        0.0,
        1.0,
    )
    if colormap == "灰度":
        gray = np.minimum(1.0, normalized * (1023 / max_gray))
        return np.stack([gray, gray, gray], axis=-1)

    import matplotlib

    cmap = matplotlib.colormaps["twilight"]
    rgba = cmap(normalized)  # float64 in [0,1]
    return np.asarray(rgba[:, :, :3], dtype=np.float32)


def refresh_phase_preview(slm_num: int) -> None:
    phase_key = f"slm{slm_num}_phase_preview"
    source_key = f"slm{slm_num}_phase_source"
    colormap_key = f"slm{slm_num}_preview_colormap"
    slm = st.session_state.get(f"slm{slm_num}")
    if slm is None:
        st.session_state[phase_key] = None
        st.session_state[source_key] = "暂无"
        return

    phase_gray, source = slm.get_displayed_phase()
    max_gray = getattr(slm, "_max_gray", Santec.MAX_GRAYSCALE_VALUE)
    colormap = st.session_state.get(colormap_key, PREVIEW_COLORMAP_DEFAULT)
    st.session_state[phase_key] = (
        None
        if phase_gray is None
        else _phase_to_preview(phase_gray, max_gray=max_gray, colormap=colormap)
    )
    st.session_state[source_key] = source


def _build_control(pattern_type: str, slm: Santec) -> PatternControl:
    """Construct the control for ``pattern_type`` from an SLM object's props."""
    cls = PATTERN_REGISTRY.get(pattern_type)
    if cls is None:
        raise ValueError(f"未知相位图类型: {pattern_type}")
    panel_res = getattr(slm, "Panel_Res", (1920, 1200))
    raw_max_gray = getattr(slm, "_max_gray", None)
    return cls(
        slm_id=int(getattr(slm, "slm_number", 0)),
        wavelength=float(getattr(slm, "wavelength", None) or 1064.0),
        pixel_pitch_um=float(getattr(slm, "Pitch_um", 8.0)),
        width=int(panel_res[0]),
        height=int(panel_res[1]),
        bits=int(getattr(slm, "Gray_Scale_bits", 10)),
        max_gray=int(raw_max_gray) if isinstance(raw_max_gray, int) else None,
    )


def generate_phase_gray(
    slm: Santec,
    pattern_type: str,
    params: dict[str, Any],
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> np.ndarray:
    """Generate phase pattern using SLM properties.

    Dispatcher: builds the matching :class:`PatternControl` from the SLM's own
    properties (resolution, pixel size, bit depth, wavelength, id) via
    ``__init__`` and delegates to its ``generate_phase_gray``.

    Args:
        slm: Connected SLM object.
        pattern_type: One of the supported pattern names (e.g. "平场", "Zernike").
        params: Pattern-specific parameters (see ``render_pattern_controls``).
        progress_cb: Optional callback ``fn(iteration, total, mse)`` forwarded to
        controls that iterate (currently none — closed-loop shaping lives in
        :mod:`ao_shaping.gui.slm.shaping_controls`).
    """
    control = _build_control(pattern_type, slm)
    return control.generate_phase_gray(params, slm=slm, progress_cb=progress_cb)
