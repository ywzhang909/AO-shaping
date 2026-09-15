from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

import numpy as np
import streamlit as st
from loguru import logger
from scipy.special import erf

from ao_shaping.algorithm.gerchberg_saxton import gerchberg_saxton
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.utils.beam_metrics import measure_spot_diameter_cam
from ao_shaping.utils.pattern_helper import (
    PatternHelper,
    calc_blazed_grating_period,
)
from ao_shaping.utils.targets import (
    build_square_target_amplitude,
    compute_square_side,
)
from ao_shaping.utils.zernike_calc import get_zernike_name


class PatternControl(ABC):
    """Base class for a single pattern-type UI control.

    Each subclass carries its SLM context (id, wavelength, pixel pitch, panel
    size, bit depth) in ``__init__`` — ``render()`` builds the Streamlit
    widgets from these values (NOT from ``st.session_state``), and
    ``generate_phase_rad()`` turns the returned params into the raw radian
    phase image. ``slm_id`` prefixes all widget keys, so rendering several SLMs
    on one page never collides.
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
    ) -> None:
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
        return PatternHelper((self.width, self.height), bits=self.bits)

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
        slm: SantecSLM200 | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        """Default: convert ``generate_phase_rad`` output via the SLM driver.

        Subclasses with a raw-gray contract (平场, 半半相位) or a bespoke
        hardware pipeline (GS方形整形) override this method.
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

    DEFAULTS: ClassVar[dict[str, Any]] = {"flat_gray": 512}
    RANGES: ClassVar[dict[str, Any]] = {"flat_gray": (0, 1023)}
    CONTEXT_SYNCED_WIDGETS: ClassVar[tuple[str, ...]] = ("flat_gray",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(flat_gray=self.max_gray // 2)
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
        slm: SantecSLM200 | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        gray = int(params["flat_gray"])
        return np.full((self.height, self.width), gray, dtype=np.uint16)


class LinearGratingControl(PatternControl):
    """线性光栅 — period and phase range."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "period": 64.0,
        "phase_range": float(2 * np.pi),
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "period": (1.0, 1000.0),
        "phase_range": (0.1, float(2 * np.pi)),
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
                "相位范围 (rad)",
                min_value=self.ranges["phase_range"][0],
                max_value=self.ranges["phase_range"][1],
                value=self.defaults["phase_range"],
                step=0.1,
                key=f"{prefix}_linear_phase_range",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        return self._new_helper().linear_grating(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
        )


class HologramGratingControl(PatternControl):
    """全息光栅 — period and phase range."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "period": 64.0,
        "phase_range": float(2 * np.pi),
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "period": (1.0, 1000.0),
        "phase_range": (0.1, float(2 * np.pi)),
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
                "相位范围 (rad)",
                min_value=self.ranges["phase_range"][0],
                max_value=self.ranges["phase_range"][1],
                value=self.defaults["phase_range"],
                step=0.1,
                key=f"{prefix}_hologram_phase_range",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        return self._new_helper().hologram(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
        )


class BlazedGratingControl(PatternControl):
    """闪耀光栅 — with angle/direct mode and direction."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "period": 10.0,
        "phase_range": float(2 * np.pi),
        "direction": "vertical",
        "angle_deg": 10.0,
        "calc_wl": 1064,
        "calc_pitch": 8.0,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "period": (1.0, 10000.0),
        "phase_range": (0.1, float(2 * np.pi)),
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
            "相位范围 (rad)",
            min_value=self.ranges["phase_range"][0],
            max_value=self.ranges["phase_range"][1],
            value=self.defaults["phase_range"],
            step=0.1,
            key=f"{prefix}_blazed_phase_range",
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
        return self._new_helper().linear_grating(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
            direction=str(params["direction"]),
        )


class CircularGratingControl(PatternControl):
    """圆形光栅 — radius and phase range."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "radius": 64.0,
        "phase_range": float(2 * np.pi),
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "radius": (1.0, 2000.0),
        "phase_range": (0.1, float(2 * np.pi)),
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
                "相位范围 (rad)",
                min_value=self.ranges["phase_range"][0],
                max_value=self.ranges["phase_range"][1],
                value=self.defaults["phase_range"],
                step=0.1,
                key=f"{prefix}_circular_phase_range",
            ),
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        return self._new_helper().circular_grating(
            radius=float(params["radius"]),
            phase_range=float(params["phase_range"]),
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.defaults.update(radius=self.default_radius)

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

        pairs: list[dict[str, Any]] = []
        for n in range(n_max + 1):
            for m in range(-n, n + 1):
                if (n - abs(m)) % 2 == 0:
                    default_val = 1.0 if n == 0 and m == 0 else 0.0
                    pairs.append(
                        {
                            "n": n,
                            "m": m,
                            "name": get_zernike_name(n, m) or f"n={n},m={m}",
                            "coeff": default_val,
                        }
                    )

        edited = st.data_editor(
            pairs,
            key=f"{prefix}_zernike_table",
            num_rows="fixed",
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
        "phase_range": float(2 * np.pi),
        "blaze_direction": "horizontal",
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "flat_gray": (0, 1023),
        "period": (1.0, 1000.0),
        "phase_range": (0.1, float(2 * np.pi)),
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
                "相位范围 (rad)",
                min_value=self.ranges["phase_range"][0],
                max_value=self.ranges["phase_range"][1],
                value=self.defaults["phase_range"],
                step=0.1,
                key=f"{prefix}_halfhalf_phase_range",
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
        phase_range = float(params["phase_range"])
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
        slm: SantecSLM200 | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        if slm is None:
            raise ValueError("半半相位 生成灰度相位需要 slm 对象")
        flat_gray = int(params["flat_gray"])
        period = float(params["period"])
        phase_range = float(params["phase_range"])
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


class GSSquareControl(PatternControl):
    """GS方形整形 — spot image upload, GS parameters.

    The GS pipeline is inherently SLM-contextual: it measures the uploaded
    far-field spot, sizes a square on the SLM grid, runs Gerchberg-Saxton, and
    (optionally) pushes live phase to the panel. ``generate_phase_rad`` /
    ``generate_phase_gray`` therefore require the SLM object.
    """

    #: Widget defaults used by render() (no st.session_state reads).
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "gs_factor": 1.5,
        "gs_focal_length_mm": 100.0,
        "gs_iterations": 100,
        "gs_energy": 0.90,
        "gs_p_cam": 0.0,
        "gs_live_display": False,
        "gs_live_interval": 1,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "gs_factor": (0.5, 5.0),
        "gs_focal_length_mm": (1.0, 5000.0),
        "gs_iterations": (1, 5000),
        "gs_energy": (0.1, 0.999),
        "gs_p_cam": (0.0, 100.0),
        "gs_live_interval": (1, 50),
    }

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        params: dict[str, Any] = {}

        uploaded = st.file_uploader(
            "上传远场光斑图片 (用于自动计算方形大小)",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key=f"{prefix}_gs_spot_image",
        )
        st.caption("GS 算法将输入光斑整形为方形；方形边长 = 光斑直径 × 尺寸因子")
        if uploaded is not None:
            try:
                params["intensity_cam"] = self._upload_to_intensity(uploaded)
            except Exception as e:
                st.error(f"图片解码失败: {e}")
                logger.exception(f"Failed to decode GS spot image: {e}")
        else:
            st.warning("请先上传远场光斑图片")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            params["gs_factor"] = st.number_input(
                "方形边长/光斑直径 因子",
                min_value=self.ranges["gs_factor"][0],
                max_value=self.ranges["gs_factor"][1],
                value=self.defaults["gs_factor"],
                step=0.1,
                key=f"{prefix}_gs_factor",
            )
        with col_g2:
            params["gs_focal_length_mm"] = st.number_input(
                "焦距 (mm)",
                min_value=self.ranges["gs_focal_length_mm"][0],
                max_value=self.ranges["gs_focal_length_mm"][1],
                value=self.defaults["gs_focal_length_mm"],
                step=1.0,
                key=f"{prefix}_gs_focal_mm",
            )

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            params["gs_iterations"] = st.number_input(
                "GS 迭代次数",
                min_value=self.ranges["gs_iterations"][0],
                max_value=self.ranges["gs_iterations"][1],
                value=self.defaults["gs_iterations"],
                step=1,
                key=f"{prefix}_gs_iterations",
            )
        with col_g4:
            params["gs_energy"] = st.number_input(
                "光斑能量占比 (0~1)",
                min_value=self.ranges["gs_energy"][0],
                max_value=self.ranges["gs_energy"][1],
                value=self.defaults["gs_energy"],
                step=0.05,
                key=f"{prefix}_gs_energy",
            )

        p_cam_input = st.number_input(
            "相机像素间距 (μm，留空使用 SLM 像素间距)",
            min_value=self.ranges["gs_p_cam"][0],
            max_value=self.ranges["gs_p_cam"][1],
            value=self.defaults["gs_p_cam"],
            step=0.1,
            key=f"{prefix}_gs_p_cam",
            help="输入相机像素间距(μm)。为 0 时默认与 SLM 像素间距一致。",
        )
        if p_cam_input > 0:
            params["gs_p_cam"] = p_cam_input * 1e-6

        params["gs_live_display"] = st.checkbox(
            "实时显示到 SLM（每轮迭代同步下发相位）",
            value=self.defaults["gs_live_display"],
            key=f"{prefix}_gs_live_display",
            help="勾选后 GS 每轮迭代都会把当前相位写入 SLM，方便实时观察整形收敛过程。\n"
            "写入会自动轮换内存槽，避免同槽重复写入被固件判为无操作。",
        )
        params["gs_live_interval"] = st.number_input(
            "实时显示间隔（每 N 轮迭代下发一次）",
            min_value=self.ranges["gs_live_interval"][0],
            max_value=self.ranges["gs_live_interval"][1],
            value=self.defaults["gs_live_interval"],
            step=1,
            key=f"{prefix}_gs_live_interval",
        )

        return params

    @staticmethod
    def _upload_to_intensity(uploaded: Any) -> Any:
        """Decode an uploaded image file into a 2D float intensity array."""
        from PIL import Image  # type: ignore[assignment]

        raw = uploaded.getvalue()
        img = Image.open(io.BytesIO(raw)).convert("L")
        import numpy as np

        return np.asarray(img, dtype=np.float64)

    def _require_intensity(self, params: dict[str, Any]) -> Any:
        intensity_cam = params.get("intensity_cam")
        if intensity_cam is None:
            raise ValueError("GS方形整形需要先上传远场光斑图片")
        return intensity_cam

    def generate_phase_rad(
        self,
        params: dict[str, Any],
        slm: SantecSLM200 | None = None,
    ) -> np.ndarray:
        """Raw radian GS phase — requires the SLM context (the pipeline
        measures the camera spot at panel resolution and may push live phase)."""
        if slm is None:
            raise ValueError("GS方形整形 generate_phase_rad 需要 slm 对象")
        intensity_cam = self._require_intensity(params)
        return _gs_square_phase_radians(
            slm,
            intensity_cam,
            factor=float(params.get("gs_factor", self.defaults["gs_factor"])),
            focal_length_m=(
                float(
                    params.get(
                        "gs_focal_length_mm", self.defaults["gs_focal_length_mm"]
                    )
                )
                * 1e-3
            ),
            iterations=int(params.get("gs_iterations", self.defaults["gs_iterations"])),
            energy=float(params.get("gs_energy", self.defaults["gs_energy"])),
            p_cam=params.get("gs_p_cam"),
            progress_cb=None,
            live_display=bool(
                params.get("gs_live_display", self.defaults["gs_live_display"])
            ),
            live_display_interval=int(
                params.get("gs_live_interval", self.defaults["gs_live_interval"])
            ),
        )

    def generate_phase_gray(
        self,
        params: dict[str, Any],
        slm: SantecSLM200 | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        if slm is None:
            raise ValueError("GS方形整形 生成灰度相位需要 slm 对象")
        intensity_cam = self._require_intensity(params)
        return generate_gs_square_phase(
            slm,
            intensity_cam,
            factor=float(params.get("gs_factor", self.defaults["gs_factor"])),
            focal_length_m=(
                float(
                    params.get(
                        "gs_focal_length_mm", self.defaults["gs_focal_length_mm"]
                    )
                )
                * 1e-3
            ),
            iterations=int(params.get("gs_iterations", self.defaults["gs_iterations"])),
            energy=float(params.get("gs_energy", self.defaults["gs_energy"])),
            p_cam=params.get("gs_p_cam"),
            progress_cb=progress_cb,
            live_display=bool(
                params.get("gs_live_display", self.defaults["gs_live_display"])
            ),
            live_display_interval=int(
                params.get("gs_live_interval", self.defaults["gs_live_interval"])
            ),
        )


class SteadyPhaseControl(PatternControl):
    """稳像法整形 — SPM + blaze params with Nyquist constraint check."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "focal_length_mm": 300.0,
        "waist_radius_um": 3600.0,
        "flat_top_half_length_um": 150.0,
        "blaze_period": 4.0,
        "blaze_angle_deg": 45.0,
    }
    RANGES: ClassVar[dict[str, Any]] = {
        "focal_length_mm": (10.0, 5000.0),
        "waist_radius_um": (100.0, 50000.0),
        "flat_top_half_length_um": (10.0, 5000.0),
        "blaze_period": (1.0, 1000.0),
        "blaze_angle_deg": (0.0, 360.0),
    }

    def render(self, prefix: str | None = None) -> dict[str, Any]:
        prefix = prefix or self.prefix
        st.caption(
            "稳像法 (Steady Phase Method) 生成方形平顶光束整形相位。"
            "参考: 翟中生等, 应用光学 2023, 44(4), 711-719。"
            "原理: 通过几何稳相法相位 φ(x,y) 将入射高斯光束整形为方形平顶光束, "
            "叠加闪耀光栅使平顶偏移到一级衍射位置, 与零级光空间分离。"
        )

        slm_wavelength_nm = self.wavelength
        slm_pixel_pitch_um = self.pixel_pitch_um

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            focal_length_mm = st.number_input(
                "傅里叶透镜焦距 f (mm)",
                min_value=self.ranges["focal_length_mm"][0],
                max_value=self.ranges["focal_length_mm"][1],
                value=self.defaults["focal_length_mm"],
                step=10.0,
                key=f"{prefix}_spm_focal_mm",
                help=(
                    "SLM 后方傅里叶透镜的焦距, 即 SLM 到 CCD 之间透镜的焦距。\n"
                    "单位: mm。典型值: 100~500 mm。\n"
                    "该透镜对 SLM 上的相位分布做傅里叶变换, "
                    "CCD 放在其后焦面上接收远场衍射图样。"
                ),
            )
        with col_s2:
            waist_radius_um = st.number_input(
                "高斯光束束腰半径 w₀ (μm)",
                min_value=self.ranges["waist_radius_um"][0],
                max_value=self.ranges["waist_radius_um"][1],
                value=self.defaults["waist_radius_um"],
                step=100.0,
                key=f"{prefix}_spm_waist_um",
                help=(
                    "入射到 SLM 表面的高斯光束 1/e² 束腰半径。\n"
                    "单位: μm。典型值: 1~5 mm (1000~5000 μm)。\n"
                    "测量方法: 用 CCD 采集光斑, 拟合高斯分布 I(r)=I₀·exp(-2r²/w₀²), "
                    "或测量光斑直径 D 后取 w₀ ≈ D/2 (强度降至 1/e² 处)。\n"
                    "注意: 此处是 SLM 面上的束腰, 非焦点处束腰。"
                ),
            )

        col_s3, col_s4 = st.columns(2)
        with col_s3:
            flat_top_half_length_um = st.number_input(
                "平顶光束半边长 L (μm)",
                min_value=self.ranges["flat_top_half_length_um"][0],
                max_value=self.ranges["flat_top_half_length_um"][1],
                value=self.defaults["flat_top_half_length_um"],
                step=10.0,
                key=f"{prefix}_spm_flat_top_um",
                help=(
                    "期望的方形平顶光束的半边长 (远场焦平面上)。\n"
                    "单位: μm。平顶光束总边长 = 2L。\n"
                    "⚠️ 奈奎斯特约束: L 必须满足 L < λ·f·w₀/(π·dx²), "
                    "否则相位梯度过大导致采样混叠, 整形质量下降。\n"
                    "参考代码中 f=300mm, w₀=3.6mm, λ=1064nm, dx=12.5μm 时, "
                    "L_max ≈ 184 μm。"
                ),
            )
        with col_s4:
            blaze_period = st.number_input(
                "闪耀光栅周期 T (像素)",
                min_value=self.ranges["blaze_period"][0],
                max_value=self.ranges["blaze_period"][1],
                value=self.defaults["blaze_period"],
                step=0.5,
                key=f"{prefix}_spm_blaze_period",
                help=(
                    "用于分离零级光的闪耀光栅周期 (单位: SLM 像素)。\n"
                    "偏转角 θ 满足 sin(θ) = λ/(T·dx)。\n"
                    "• T=4 像素: 偏转角较大, 平顶远离零级\n"
                    "• T=8 像素: 偏转角适中\n"
                    "• T=16+ 像素: 偏转角较小, 平顶靠近零级\n"
                    "较小的 T 值产生更大的偏转角, 但衍射效率可能降低。"
                ),
            )

        blaze_angle_deg = st.number_input(
            "闪耀光栅方向 θ (度)",
            min_value=self.ranges["blaze_angle_deg"][0],
            max_value=self.ranges["blaze_angle_deg"][1],
            value=self.defaults["blaze_angle_deg"],
            step=1.0,
            key=f"{prefix}_spm_blaze_angle",
            help=(
                "闪耀光栅的倾斜方向角 (单位: 度)。\n"
                "控制平顶光束相对于零级光的偏转方向。\n"
                "• 0°: 水平向右偏转\n"
                "• 45°: 右上方偏转 (对角线方向)\n"
                "• 90°: 垂直向上偏转\n"
                "• 180°: 水平向左偏转\n"
                "通常设为 45° 使平顶光束沿对角线方向偏移, "
                "与零级光有最大空间分离。"
            ),
        )

        wavelength_m = slm_wavelength_nm * 1e-9
        f_m = focal_length_mm * 1e-3
        w0_m = waist_radius_um * 1e-6
        dx_m = slm_pixel_pitch_um * 1e-6
        L_max_um = (wavelength_m * f_m * w0_m / (np.pi * dx_m**2)) * 1e6
        L_input_um = flat_top_half_length_um

        st.divider()
        st.caption("📊 约束检查")
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            st.metric(
                "奈奎斯特 L_max",
                f"{L_max_um:.1f} μm",
                help="L_max = λ·f·w₀/(π·dx²), 超过此值将产生采样混叠",
            )
        with col_check2:
            if L_input_um >= L_max_um:
                st.error(
                    f"⚠️ L={L_input_um:.0f}μm ≥ L_max={L_max_um:.1f}μm, "
                    "相位梯度过大, 将产生采样混叠!"
                )
            else:
                st.success(
                    f"✅ L={L_input_um:.0f}μm < L_max={L_max_um:.1f}μm, "
                    "满足奈奎斯特条件"
                )

        st.caption(
            f"当前参数: λ={slm_wavelength_nm}nm, "
            f"dx={slm_pixel_pitch_um}μm, "
            f"f={focal_length_mm:.0f}mm, "
            f"w₀={waist_radius_um:.0f}μm"
        )

        return {
            "focal_length_mm": focal_length_mm,
            "waist_radius_um": waist_radius_um,
            "flat_top_half_length_um": flat_top_half_length_um,
            "blaze_period": blaze_period,
            "blaze_angle_deg": blaze_angle_deg,
        }

    def generate_phase_rad(self, params: dict[str, Any]) -> np.ndarray:
        wavelength_m = self.wavelength * 1e-9  # nm -> m
        pixel_pitch_m = self.pixel_pitch_um * 1e-6  # μm -> m

        f = float(params["focal_length_mm"]) * 1e-3  # mm -> m
        w0 = float(params["waist_radius_um"]) * 1e-6  # μm -> m
        L = float(params["flat_top_half_length_um"]) * 1e-6  # μm -> m
        T = float(params["blaze_period"])  # pixels
        theta = float(params["blaze_angle_deg"]) * np.pi / 180  # deg -> rad

        # Create coordinate grids (in meters)
        x_m = (np.arange(self.width) - self.width // 2) * pixel_pitch_m
        y_m = (np.arange(self.height) - self.height // 2) * pixel_pitch_m
        X_m, Y_m = np.meshgrid(x_m, y_m)

        # 1D steady phase φ(t)
        def _steady_phase_1d(t, L, lam, f, w0):
            return (
                np.sqrt(np.pi)
                * L
                / (lam * f)
                * (
                    np.sqrt(np.pi) * t / (2 * w0) * erf(t / w0)
                    + 0.5 * np.exp(-(t**2) / w0**2)
                    - 0.5
                )
            )

        # SPM phase (separable 2D)
        phi_1d_x = _steady_phase_1d(x_m, L, wavelength_m, f, w0)
        phi_1d_y = _steady_phase_1d(y_m, L, wavelength_m, f, w0)
        phi_spm = phi_1d_y[:, None] + phi_1d_x[None, :]

        # Blazed grating phase (sawtooth, already wrapped to [0, 2π))
        px = np.arange(self.width) - self.width // 2
        py = np.arange(self.height) - self.height // 2
        PX, PY = np.meshgrid(px, py)
        phi_blaze = 2 * np.pi / T * np.mod(PX * np.cos(theta) + PY * np.sin(theta), T)

        # 原始弧度相位 (不 mod-2π — 驱动 create_phase_from_array 负责包裹;
        # 与旧实现 np.mod(phi_spm + phi_blaze, 2π) 的灰度输出字节一致)
        return phi_spm + phi_blaze


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
    "GS方形整形": GSSquareControl,
    "稳像法整形": SteadyPhaseControl,
}


# ---------------------------------------------------------------------
# Phase generation functions
# ---------------------------------------------------------------------


def _phase_to_preview(phase_gray: np.ndarray) -> np.ndarray:
    normalized = phase_gray.astype(np.float32) / max(
        SantecSLM200.MAX_GRAYSCALE_VALUE, 1
    )
    return np.clip(normalized, 0.0, 1.0)


def refresh_phase_preview(slm_num: int) -> None:
    phase_key = f"slm{slm_num}_phase_preview"
    source_key = f"slm{slm_num}_phase_source"
    slm = st.session_state.get(f"slm{slm_num}")
    if slm is None:
        st.session_state[phase_key] = None
        st.session_state[source_key] = "暂无"
        return

    phase_gray, source = slm.get_displayed_phase()
    st.session_state[phase_key] = (
        None if phase_gray is None else _phase_to_preview(phase_gray)
    )
    st.session_state[source_key] = source


def _gs_square_phase_radians(
    slm: SantecSLM200,
    intensity_cam: np.ndarray,
    factor: float = 1.5,
    focal_length_m: float = 0.1,
    iterations: int = 100,
    energy: float = 0.90,
    p_cam: float | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
    live_display: bool = False,
    live_display_interval: int = 1,
) -> np.ndarray:
    """Full GS beam-shaping pipeline returning the RAW radian phase.

    Measures the beam spot from ``intensity_cam``, auto-sizes a square target
    on the SLM grid, runs Gerchberg-Saxton at SLM resolution, and returns the
    raw (unwrapped, pre-grayscale) recovered phase.

    Args:
        slm: SLM object (reads resolution, pitch, bits, wavelength).
        intensity_cam: 2D far-field intensity image used to size the square.
        factor: Square-to-spot size factor (default 1.5).
        focal_length_m: Focal length / propagation distance in meters.
        iterations: GS iteration count (default 100).
        energy: Encircled-energy fraction for spot measurement (default 0.90).
        p_cam: Camera pixel pitch in meters; defaults to the SLM pitch.
        progress_cb: Optional callback ``fn(iteration, total, mse)`` invoked
            after each GS iteration (1-based iteration, total iterations,
            current mean-squared error). Enables live progress in the UI.
        live_display: When True, push the evolving phase to the SLM hardware on
            every iteration so the beam-shaping progress is visible in real
            time.  Writes rotate through memory slots automatically (the Santec
            firmware treats a repeat ``display_memory`` on the same slot as a
            no-op, so consecutive writes MUST target different slots).
            Uses auto-estimated pixel flip wait (wait_time_s=None) so the
            LCOS panel actually refreshes before the next iteration —
            0.0 would skip the wait and the display would never take effect.
        live_display_interval: Push to hardware every N-th iteration when
            ``live_display`` is True (default 1 = every iteration).

    Returns:
        float64 radian phase array with shape (height, width), unwrapped.
    """
    width = slm.Panel_Res[0]
    height = slm.Panel_Res[1]
    d_slm = float(slm.Pitch_um) * 1e-6  # um -> m
    p_cam_eff = d_slm if p_cam is None else float(p_cam)
    wavelength_nm = slm.wavelength
    assert wavelength_nm is not None, "SLM波长未设置，无法生成相位图"

    spot_d = measure_spot_diameter_cam(intensity_cam, energy=energy)
    side = compute_square_side(spot_d, factor=factor, p_cam=p_cam_eff, d_slm=d_slm)

    # Clamp the square so it fits on the SLM grid with a small margin.
    max_side = min(height, width) - 8
    if side > max_side:
        logger.warning(
            "Beam-derived square side {}px exceeds SLM grid, clamped to {}px",
            side,
            max_side,
        )
        side = max_side
    if side < 8:
        side = 8

    source = np.ones((height, width), dtype=np.float64)
    target = build_square_target_amplitude(height, width, side)

    logger.info(
        "GS square shaping: spot_d={:.1f}px, side={}px, f={:.3f}m, iters={}",
        spot_d,
        side,
        focal_length_m,
        iterations,
    )

    # Live hardware display — converts the current source-plane phase (radians)
    # to uint16 grayscale and pushes it to the SLM.  ``display_phase`` rotates
    # memory slots internally and auto-estimates pixel flip wait (wait_time_s=None)
    # so the LCOS panel actually refreshes before the next iteration.
    def _live_phase_cb(iteration: int, phase_rad: np.ndarray) -> None:
        if not live_display:
            return
        if live_display_interval > 1 and (iteration % live_display_interval) != 0:
            return
        try:
            # None = auto-estimate pixel flip wait so the LCOS panel
            # actually refreshes before the next iteration.
            # 0.0 would return immediately without waiting, causing
            # the display to never actually take effect.
            slm.display_phase(phase_rad, wait_time_s=None)
            # Refresh the phase preview so the UI reflects the latest
            # displayed pattern without waiting for the user to press the
            # manual refresh button.
            refresh_phase_preview(slm.slm_number)
            logger.debug(
                "GS live display: iteration {} phase 已发送到 SLM",
                iteration + 1,
            )
        except Exception as e:
            logger.warning(f"GS live display 失败 (iteration {iteration + 1}): {e}")

    result = gerchberg_saxton(
        source_amplitude=source,
        target_amplitude=target,
        iterations=iterations,
        cell_spacing=d_slm,
        distance=focal_length_m,
        wavelength=float(wavelength_nm) * 1e-9,
        progress_callback=(
            (lambda i, mse: progress_cb(i + 1, iterations, mse))
            if progress_cb is not None
            else None
        ),
        phase_callback=_live_phase_cb,
    )
    return result.phase


def generate_gs_square_phase(
    slm: SantecSLM200,
    intensity_cam: np.ndarray,
    factor: float = 1.5,
    focal_length_m: float = 0.1,
    iterations: int = 100,
    energy: float = 0.90,
    p_cam: float | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
    live_display: bool = False,
    live_display_interval: int = 1,
) -> np.ndarray:
    """Full GS beam-shaping pipeline: measure spot -> size square -> GS phase.

    Measures the beam spot from ``intensity_cam``, auto-sizes a square target
    on the SLM grid, runs Gerchberg-Saxton at SLM resolution, and returns the
    uint16 grayscale phase for the SLM. See :func:`_gs_square_phase_radians`
    for the parameter documentation.

    Returns:
        uint16 phase grayscale array with shape (height, width).
    """
    phase_rad = _gs_square_phase_radians(
        slm,
        intensity_cam,
        factor=factor,
        focal_length_m=focal_length_m,
        iterations=iterations,
        energy=energy,
        p_cam=p_cam,
        progress_cb=progress_cb,
        live_display=live_display,
        live_display_interval=live_display_interval,
    )
    return slm.create_phase_from_array(phase_rad)


def _build_control(pattern_type: str, slm: SantecSLM200) -> PatternControl:
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
    slm: SantecSLM200,
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
        pattern_type: One of the supported pattern names (e.g. "平场", "GS方形整形").
        params: Pattern-specific parameters (see ``render_pattern_controls``).
        progress_cb: Optional callback ``fn(iteration, total, mse)`` forwarded to
            :func:`generate_gs_square_phase` for the "GS方形整形" pattern.
    """
    control = _build_control(pattern_type, slm)
    return control.generate_phase_gray(params, slm=slm, progress_cb=progress_cb)
