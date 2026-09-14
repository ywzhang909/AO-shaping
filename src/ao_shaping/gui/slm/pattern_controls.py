from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Any, Callable

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

    Each subclass renders its own set of Streamlit widgets and returns
    a params dict consumed by :func:`generate_phase_gray`.
    """

    @abstractmethod
    def render(self, prefix: str) -> dict[str, Any]:
        """Render pattern-specific controls, return params dict."""


class FlatControl(PatternControl):
    """平场 — single grayscale value."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "flat_gray": st.number_input(
                "灰度",
                min_value=0,
                max_value=int(SantecSLM200.MAX_GRAYSCALE_VALUE),
                step=1,
                key=f"{prefix}_flat_gray",
            )
        }


class LinearGratingControl(PatternControl):
    """线性光栅 — period and phase range."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "period": st.number_input(
                "周期 (像素)",
                min_value=1.0,
                max_value=1000.0,
                step=1.0,
                key=f"{prefix}_linear_period",
            ),
            "phase_range": st.number_input(
                "相位范围 (rad)",
                min_value=0.1,
                max_value=float(2 * np.pi),
                step=0.1,
                key=f"{prefix}_linear_phase_range",
            ),
        }


class HologramGratingControl(PatternControl):
    """全息光栅 — period and phase range."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "period": st.number_input(
                "周期 (像素)",
                min_value=1.0,
                max_value=1000.0,
                step=1.0,
                key=f"{prefix}_hologram_period",
            ),
            "phase_range": st.number_input(
                "相位范围 (rad)",
                min_value=0.1,
                max_value=float(2 * np.pi),
                step=0.1,
                key=f"{prefix}_hologram_phase_range",
            ),
        }


class BlazedGratingControl(PatternControl):
    """闪耀光栅 — with angle/direct mode and direction."""

    def render(self, prefix: str) -> dict[str, Any]:
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
            default_wl = int(st.session_state.get(f"{prefix}_wavelength", 1064))
            default_pitch = float(st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0))

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                angle_deg = st.number_input(
                    "衍射角度 θ (度)",
                    min_value=0.1,
                    max_value=89.0,
                    value=10.0,
                    step=0.5,
                    key=f"{prefix}_blazed_calc_angle",
                )
            with col_a2:
                calc_wl = st.number_input(
                    "波长 λ (nm)",
                    min_value=400,
                    max_value=1600,
                    value=default_wl,
                    step=1,
                    key=f"{prefix}_blazed_calc_wl",
                )

            calc_pitch = st.number_input(
                "像素间距 (μm)",
                min_value=0.1,
                max_value=100.0,
                value=default_pitch,
                step=0.1,
                key=f"{prefix}_blazed_calc_pitch",
            )

            period_pixels = calc_blazed_grating_period(angle_deg, calc_wl, calc_pitch)
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
                min_value=1.0,
                max_value=10000.0,
                step=1.0,
                key=f"{prefix}_blazed_period",
            )

        phase_range = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
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


class CircularGratingControl(PatternControl):
    """圆形光栅 — radius and phase range."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "radius": st.number_input(
                "圆形周期半径 (像素)",
                min_value=1.0,
                max_value=2000.0,
                step=10.0,
                key=f"{prefix}_circular_radius",
            ),
            "phase_range": st.number_input(
                "相位范围 (rad)",
                min_value=0.1,
                max_value=float(2 * np.pi),
                step=0.1,
                key=f"{prefix}_circular_phase_range",
            ),
        }


class LensControl(PatternControl):
    """透镜 — focal length, pixel pitch, lens radius."""

    def render(self, prefix: str) -> dict[str, Any]:
        default_pitch = float(st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0))
        default_w = int(st.session_state.get(f"{prefix}_width", 1920))
        default_h = int(st.session_state.get(f"{prefix}_height", 1200))
        default_radius = min(default_w, default_h) // 2

        return {
            "focal_length_mm": st.number_input(
                "焦距 (mm)",
                min_value=1.0,
                max_value=100000.0,
                step=10.0,
                key=f"{prefix}_lens_focal_length",
            ),
            "pixel_pitch_um": st.number_input(
                "像素间距 (um)",
                min_value=0.1,
                max_value=100.0,
                value=default_pitch,
                step=0.1,
                key=f"{prefix}_lens_pixel_pitch",
            ),
            "lens_radius": st.number_input(
                "透镜半径 (像素)",
                min_value=1,
                max_value=2000,
                value=default_radius,
                step=1,
                key=f"{prefix}_lens_radius",
            ),
        }


class CheckerboardControl(PatternControl):
    """棋盘格 — period."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "period": st.number_input(
                "棋盘格周期 (像素)",
                min_value=1,
                max_value=1000,
                step=1,
                key=f"{prefix}_checker_period",
            )
        }


class BinaryGratingControl(PatternControl):
    """二元光栅 — a, b, direction."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "a": st.number_input(
                "亮条纹宽度 a (像素)",
                min_value=1,
                max_value=1000,
                step=1,
                key=f"{prefix}_binary_a",
            ),
            "b": st.number_input(
                "暗条纹宽度 b (像素)",
                min_value=1,
                max_value=1000,
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


class MicrolensArrayControl(PatternControl):
    """微透镜阵列 — lens size, focal length, pixel pitch."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "lens_size": st.number_input(
                "微透镜尺寸 (像素)",
                min_value=8,
                max_value=1000,
                step=1,
                key=f"{prefix}_microlens_size",
            ),
            "focal_length_mm": st.number_input(
                "焦距 (mm)",
                min_value=1.0,
                max_value=100000.0,
                step=1.0,
                key=f"{prefix}_microlens_focal_length",
            ),
            "pixel_pitch_um": st.number_input(
                "像素间距 (um)",
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                key=f"{prefix}_microlens_pixel_pitch",
            ),
        }


class TurbulenceScreenControl(PatternControl):
    """湍流相位屏 — Cn², L, pixel pitch."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "Cn2": st.number_input(
                "Cn²",
                min_value=1e-18,
                max_value=1e-10,
                format="%.1e",
                key=f"{prefix}_turbulence_cn2",
            ),
            "L": st.number_input(
                "传播距离 L (m)",
                min_value=0.1,
                max_value=1e6,
                step=10.0,
                key=f"{prefix}_turbulence_length",
            ),
            "pixel_pitch_um": st.number_input(
                "像素间距 (um)",
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                key=f"{prefix}_turbulence_pixel_pitch",
            ),
        }


class VortexPhaseControl(PatternControl):
    """涡旋相位 — topological charge, wavelength, pixel pitch, wrap phase."""

    def render(self, prefix: str) -> dict[str, Any]:
        default_wavelength = int(st.session_state.get(f"{prefix}_wavelength", 1064))
        default_pixel_pitch = float(
            st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0)
        )

        return {
            "topological_charge": st.number_input(
                "拓扑荷",
                min_value=-10,
                max_value=10,
                step=1,
                key=f"{prefix}_vortex_charge",
            ),
            "wavelength_nm": st.number_input(
                "波长 (nm)",
                value=default_wavelength,
                step=1,
                key=f"{prefix}_vortex_wavelength",
            ),
            "pixel_pitch_um": st.number_input(
                "像素间距 (um)",
                value=default_pixel_pitch,
                min_value=1,
                max_value=100,
                step=1,
                key=f"{prefix}_vortex_pixel_pitch",
            ),
            "wrap_phase": st.checkbox(
                "包裹相位",
                key=f"{prefix}_vortex_wrap_phase",
            ),
        }


class ZernikeControl(PatternControl):
    """Zernike — n_max, radius, coefficient table."""

    def render(self, prefix: str) -> dict[str, Any]:
        n_max = st.number_input(
            "最大径向阶数 N",
            min_value=1,
            max_value=10,
            step=1,
            value=5,
            key=f"{prefix}_zernike_n_max",
        )

        default_w = int(st.session_state.get(f"{prefix}_width", 1920))
        default_h = int(st.session_state.get(f"{prefix}_height", 1200))

        pairs: list[dict[str, Any]] = []
        for n in range(n_max + 1):
            for m in range(-n, n + 1):
                if (n - abs(m)) % 2 == 0:
                    default_val = 1.0 if n == 0 and m == 0 else 0.0
                    key = f"{prefix}_zernike_{n}_{m}"
                    pairs.append(
                        {
                            "n": n,
                            "m": m,
                            "name": get_zernike_name(n, m) or f"n={n},m={m}",
                            "coeff": float(st.session_state.get(key, default_val)),
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
                    value=min(default_w, default_h) // 2,
                    min_value=1,
                    max_value=2000,
                    step=1,
                    key=f"{prefix}_zernike_radius",
                )
            ),
        }


class DammannGratingControl(PatternControl):
    """达曼光栅 — order and fill factor."""

    def render(self, prefix: str) -> dict[str, Any]:
        return {
            "order": st.number_input(
                "衍射级数",
                min_value=2,
                max_value=8,
                step=1,
                key=f"{prefix}_dammann_order",
            ),
            "fill_factor": st.slider(
                "填充因子",
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                key=f"{prefix}_dammann_fill_factor",
            ),
        }


class HalfHalfPhaseControl(PatternControl):
    """半半相位 — flat gray, split direction, blaze params."""

    def render(self, prefix: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "flat_gray": st.number_input(
                "平面灰度",
                min_value=0,
                max_value=1024,
                step=1,
                key=f"{prefix}_halfhalf_flat_gray",
            ),
            "split_direction": st.selectbox(
                "划分方向",
                options=["左右", "上下"],
                format_func=lambda x: (
                    "左半平面+右半闪耀光栅" if x == "左右" else "上半平面+下半闪耀光栅"
                ),
                key=f"{prefix}_halfhalf_split",
            ),
            "period": st.number_input(
                "光栅周期 (像素)",
                min_value=1.0,
                max_value=1000.0,
                step=1.0,
                key=f"{prefix}_halfhalf_period",
            ),
            "phase_range": st.number_input(
                "相位范围 (rad)",
                min_value=0.1,
                max_value=float(2 * np.pi),
                step=0.1,
                key=f"{prefix}_halfhalf_phase_range",
            ),
        }

        split = st.session_state.get(f"{prefix}_halfhalf_split", "左右")
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


class GSSquareControl(PatternControl):
    """GS方形整形 — spot image upload, GS parameters."""

    def render(self, prefix: str) -> dict[str, Any]:
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
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                key=f"{prefix}_gs_factor",
            )
        with col_g2:
            params["gs_focal_length_mm"] = st.number_input(
                "焦距 (mm)",
                min_value=1.0,
                max_value=5000.0,
                value=100.0,
                step=1.0,
                key=f"{prefix}_gs_focal_mm",
            )

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            params["gs_iterations"] = st.number_input(
                "GS 迭代次数",
                min_value=1,
                max_value=5000,
                value=100,
                step=1,
                key=f"{prefix}_gs_iterations",
            )
        with col_g4:
            params["gs_energy"] = st.number_input(
                "光斑能量占比 (0~1)",
                min_value=0.1,
                max_value=0.999,
                value=0.90,
                step=0.05,
                key=f"{prefix}_gs_energy",
            )

        p_cam_input = st.number_input(
            "相机像素间距 (μm，留空使用 SLM 像素间距)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=f"{prefix}_gs_p_cam",
            help="输入相机像素间距(μm)。为 0 时默认与 SLM 像素间距一致。",
        )
        if p_cam_input > 0:
            params["gs_p_cam"] = p_cam_input * 1e-6

        params["gs_live_display"] = st.checkbox(
            "实时显示到 SLM（每轮迭代同步下发相位）",
            value=st.session_state.get(f"{prefix}_gs_live_display", False),
            key=f"{prefix}_gs_live_display",
            help="勾选后 GS 每轮迭代都会把当前相位写入 SLM，方便实时观察整形收敛过程。\n"
            "写入会自动轮换内存槽，避免同槽重复写入被固件判为无操作。",
        )
        params["gs_live_interval"] = st.number_input(
            "实时显示间隔（每 N 轮迭代下发一次）",
            min_value=1,
            max_value=50,
            value=int(st.session_state.get(f"{prefix}_gs_live_interval", 1)),
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


class SteadyPhaseControl(PatternControl):
    """稳像法整形 — SPM + blaze params with Nyquist constraint check."""

    def render(self, prefix: str) -> dict[str, Any]:
        st.caption(
            "稳像法 (Steady Phase Method) 生成方形平顶光束整形相位。"
            "参考: 翟中生等, 应用光学 2023, 44(4), 711-719。"
            "原理: 通过几何稳相法相位 φ(x,y) 将入射高斯光束整形为方形平顶光束, "
            "叠加闪耀光栅使平顶偏移到一级衍射位置, 与零级光空间分离。"
        )

        slm_wavelength_nm = float(st.session_state.get(f"{prefix}_wavelength", 1064))
        slm_pixel_pitch_um = float(
            st.session_state.get(f"{prefix}_pixel_pitch_um", 8.0)
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            focal_length_mm = st.number_input(
                "傅里叶透镜焦距 f (mm)",
                min_value=10.0,
                max_value=5000.0,
                value=300.0,
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
                min_value=100.0,
                max_value=50000.0,
                value=3600.0,
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
                min_value=10.0,
                max_value=5000.0,
                value=150.0,
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
                min_value=1.0,
                max_value=1000.0,
                value=4.0,
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
            min_value=0.0,
            max_value=360.0,
            value=45.0,
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
    uint16 grayscale phase for the SLM.

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
        uint16 phase grayscale array with shape (height, width).
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
    # to uint16 grayscale and pushes it to the SLM.  ``display_data`` rotates
    # memory slots internally and auto-estimates pixel flip wait (wait_time_s=None)
    # so the LCOS panel actually refreshes before the next iteration.
    def _live_phase_cb(iteration: int, phase_rad: np.ndarray) -> None:
        if not live_display:
            return
        if live_display_interval > 1 and (iteration % live_display_interval) != 0:
            return
        try:
            gray = slm.create_phase_from_array(phase_rad)
            # None = auto-estimate pixel flip wait so the LCOS panel
            # actually refreshes before the next iteration.
            # 0.0 would return immediately without waiting, causing
            # the display to never actually take effect.
            slm.display_data(gray, wait_time_s=None)
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
    return slm.create_phase_from_array(result.phase)


def generate_phase_gray(
    slm: SantecSLM200,
    pattern_type: str,
    params: dict[str, Any],
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> np.ndarray:
    """Generate phase pattern using SLM properties.

    Automatically reads resolution, pixel size, bit depth, and wavelength from
    the SLM object.

    Args:
        slm: Connected SLM object.
        pattern_type: One of the supported pattern names (e.g. "平场", "GS方形整形").
        params: Pattern-specific parameters (see ``render_pattern_controls``).
        progress_cb: Optional callback ``fn(iteration, total, mse)`` forwarded to
            :func:`generate_gs_square_phase` for the "GS方形整形" pattern.
    """
    # Get SLM properties (use pixel pitch for diffraction pattern geometry)
    width = slm.Panel_Res[0]
    height = slm.Panel_Res[1]
    pattern_pitch_um = slm.Pitch_um
    bits = slm.Gray_Scale_bits
    wavelength_nm = slm.wavelength
    assert wavelength_nm is not None, "SLM波长未设置，无法生成相位图"

    # Create unified pattern helper
    helper = PatternHelper((width, height), bits=bits)

    # Phase pattern generation mapping
    if pattern_type == "平场":
        _gray = int(params["flat_gray"])
        # Return raw grayscale uint16 — bypass create_phase_from_array which
        # interprets input as radians (mod 2π) and would map the gray value
        # through an unwanted radian-to-grayscale conversion.
        return np.full((height, width), _gray, dtype=np.uint16)
    if pattern_type == "线性光栅":
        phase_rad = helper.linear_grating(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
            wrap_phase=False,
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "圆形光栅":
        phase_rad = helper.circular_grating(
            radius=float(params["radius"]),
            phase_range=float(params["phase_range"]),
            wrap_phase=False,
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "透镜":
        # Use pixel pitch (center-to-center spacing) for diffraction geometry
        pattern_pitch = params.get("pixel_pitch_um", pattern_pitch_um)
        lens_radius = float(params.get("lens_radius", 0.0))
        lens_radius = lens_radius if lens_radius > 0 else None
        phase_rad = helper.lens(
            focal_length=float(params["focal_length_mm"]) * 1e-3,  # mm -> m
            wavelength=float(wavelength_nm) * 1e-9,  # nm -> m
            pixel_size=float(pattern_pitch) * 1e-6,  # um -> m
            lens_radius=lens_radius,
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "全息光栅":
        phase_rad = helper.hologram(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "棋盘格":
        return helper.generate_checkerboard(period=int(params["period"]))
    if pattern_type == "二元光栅":
        return helper.generate_binary_grating(
            a=int(params["a"]),
            b=int(params["b"]),
            direction=str(params["direction"]),
        )
    if pattern_type == "微透镜阵列":
        # Use pixel pitch (center-to-center spacing) for diffraction geometry
        pattern_pitch = params.get("pixel_pitch_um", pattern_pitch_um)
        return helper.generate_microlens_array(
            lens_size=int(params["lens_size"]),
            focal_length=float(params["focal_length_mm"]) * 1e-3,
            wavelength=float(wavelength_nm) * 1e-9,
            pixel_size=float(pattern_pitch) * 1e-6,
        )
    if pattern_type == "湍流相位屏":
        # Use pixel pitch (center-to-center spacing) for diffraction geometry
        pattern_pitch = params.get("pixel_pitch_um", pattern_pitch_um)
        pixel_scale = float(pattern_pitch) * 1e-6  # um -> m
        # Compute Fried parameter r0 from Cn2, L, and wavelength
        # r0 = (0.423 * k^2 * Cn2 * L)^(-3/5), k = 2π/λ
        k_wave = 2.0 * np.pi / (float(wavelength_nm) * 1e-9)
        Cn2 = float(params["Cn2"])
        L = float(params["L"])
        r0 = (0.423 * k_wave**2 * Cn2 * L) ** (-3.0 / 5.0)
        L0 = 10.0  # outer scale (m), typical atmospheric value
        helper.init_turbulence_screen(r0=r0, L0=L0, pixel_scale=pixel_scale)
        return helper.generate_turbulence_screen()
    if pattern_type == "Zernike":
        raw_coeffs = params.get("coefficients")
        coefficients: dict[tuple[int, int], float] | None = None
        if raw_coeffs is not None and isinstance(raw_coeffs, dict):
            coefficients = {
                k: float(v)
                for k, v in raw_coeffs.items()
                if isinstance(k, tuple) and isinstance(v, (int, float))
            }
        radius = float(params.get("radius", min(width, height) // 2))
        n_max = int(params.get("n_max", 6))
        return helper.generate_zernike_polynomial(
            coefficients=coefficients,
            radius=radius,
            n_max=n_max,
        )
    if pattern_type == "达曼光栅":
        order = int(params.get("order", 3))
        fill_factor = float(params.get("fill_factor", 0.5))
        return helper.generate_dammann_grating(order=order, fill_factor=fill_factor)
    if pattern_type == "涡旋相位":
        # Convert parameters to appropriate units
        wavelength_m = float(params["wavelength_nm"]) * 1e-9  # nm -> m
        pixel_pitch_m = (
            float(params.get("pixel_pitch_um", pattern_pitch_um)) * 1e-6
        )  # um -> m
        wrap_phase = bool(params.get("wrap_phase", True))
        topological_charge = int(params["topological_charge"])

        # Generate vortex phase (helper returns uint16 when wrap_phase=True,
        # radians when False).
        phase_gray = helper.generate_vortex(
            topological_charge=topological_charge,
            wavelength=wavelength_m,
            pixel_size=pixel_pitch_m,
            wrap_phase=wrap_phase,
        )

        # If wrap_phase=False, helper returns radians; convert to uint16
        if not wrap_phase:
            phase_wrapped = np.mod(phase_gray, 2 * np.pi)
            phase_gray = (phase_wrapped / (2 * np.pi) * (2**bits - 1)).astype(np.uint16)

        return phase_gray
    if pattern_type == "闪耀光栅":
        phase_rad = helper.linear_grating(
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
            wrap_phase=False,
            direction=str(params["direction"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "半半相位":
        flat_gray = int(params["flat_gray"])
        period = float(params["period"])
        phase_range = float(params["phase_range"])
        split_dir = str(params["split_direction"])
        blaze_dir = str(params.get("blaze_direction", "vertical"))

        # Generate full-frame blazed grating (radians → grayscale via SLM)
        blaze_rad = helper.linear_grating(
            period=period,
            phase_range=phase_range,
            wrap_phase=False,
            direction=blaze_dir,
        )
        blaze_gray = slm.create_phase_from_array(blaze_rad)

        # Generate flat half (direct uint16 grayscale — bypass radian conversion)
        flat_full = np.full((height, width), flat_gray, dtype=np.uint16)

        # Stitch: half flat + half blaze
        if split_dir == "左右":
            mid = width // 2
            return np.concatenate([flat_full[:, :mid], blaze_gray[:, mid:]], axis=1)
        else:  # 上下
            mid = height // 2
            return np.concatenate([flat_full[:mid, :], blaze_gray[mid:, :]], axis=0)
    if pattern_type == "GS方形整形":
        intensity_cam = params.get("intensity_cam")
        if intensity_cam is None:
            raise ValueError("GS方形整形需要先上传远场光斑图片")
        return generate_gs_square_phase(
            slm,
            intensity_cam,
            factor=float(params.get("gs_factor", 1.5)),
            focal_length_m=float(params["gs_focal_length_mm"]) * 1e-3,
            iterations=int(params.get("gs_iterations", 100)),
            energy=float(params.get("gs_energy", 0.90)),
            p_cam=params.get("gs_p_cam"),
            progress_cb=progress_cb,
            live_display=bool(params.get("gs_live_display", False)),
            live_display_interval=int(params.get("gs_live_interval", 1)),
        )
    if pattern_type == "稳像法整形":
        from scipy.special import erf

        # Get SLM parameters
        wavelength_m = float(wavelength_nm) * 1e-9  # nm -> m
        pixel_pitch_m = float(pattern_pitch_um) * 1e-6  # μm -> m

        # Get user parameters
        f = float(params["focal_length_mm"]) * 1e-3  # mm -> m
        w0 = float(params["waist_radius_um"]) * 1e-6  # μm -> m
        L = float(params["flat_top_half_length_um"]) * 1e-6  # μm -> m
        T = float(params["blaze_period"])  # pixels
        theta = float(params["blaze_angle_deg"]) * np.pi / 180  # degrees -> radians

        # Create coordinate grids (in meters)
        x_m = (np.arange(width) - width // 2) * pixel_pitch_m
        y_m = (np.arange(height) - height // 2) * pixel_pitch_m
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

        # Blazed grating phase
        px = np.arange(width) - width // 2
        py = np.arange(height) - height // 2
        PX, PY = np.meshgrid(px, py)
        phi_blaze = 2 * np.pi / T * np.mod(PX * np.cos(theta) + PY * np.sin(theta), T)

        # Combined phase
        phi_total = np.mod(phi_spm + phi_blaze, 2 * np.pi)

        return slm.create_phase_from_array(phi_total)
    raise ValueError(f"未知相位图类型: {pattern_type}")
