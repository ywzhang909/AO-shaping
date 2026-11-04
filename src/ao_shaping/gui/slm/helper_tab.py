"""``PatternHelper`` 相位生成器在独立 tab 下的直接暴露（直通 API，不建控件树）。

为什么单独成模块
----------------
``PatternHelper``（:mod:`ao_shaping.utils.wavefront.pattern_helper`）是纯相位
发生器，已被控制器经 ``multi_slm_controller`` 的 ``PATTERN_REGISTRY`` /
``PatternControl`` 控件树使用。那个控件树仍在「相位控制」tab 里，本模块**不碰**它，
而是把 ``PatternHelper`` 的生成方法**直接**暴露成一张「相位生成」tab：选图案类型 →
调参数 → 「生成预览」（``helper.to_uint16`` + ``st.image``）→ 「下发到 SLM」
（``slm.create_phase_from_array`` 弧度→灰度 + shift/矫正/LUT → ``slm.display_data``），
与控制器同一条下发路径，**不**二次施加平移。

设计契约
--------
* 入口 :func:`render_pattern_helper_tab`，父 ``main()`` 重构后的 tab 调用它。
* 纯/离线部分（类型注册表 :data:`PATTERN_TYPES`、参数规格 :data:`ParamSpec`、
  :func:`generate_phase_radians`、:data:`CONNECT_HINT`）与 ``st.*`` 调用隔离，
  可脱离 Streamlit 做单元测试。
* 遵循 gui/slm 三纪律：不调 ``st.divider()``；本模块不建 ``type="primary"`` 按钮；
  空状态文案一律点名下一步（侧重栏表格的「连接」：:data:`CONNECT_HINT`）。
* 每个 SLM 复用 ``session_state["slm{N}_pattern_helper"]`` 上的缓存
  （与控制器 ``_get_slm_helper`` 同一键、同一面板几何），不重复建坐标网格。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import streamlit as st
from loguru import logger

from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

# 侧重栏「SLM 设备」表格的引导语（与 multi_slm_controller.CONNECT_HINT 同文）。
CONNECT_HINT = "在左侧「SLM 设备」表格里点「连接」"

#: 未连接时的默认面板几何（与控制器 ``_get_slm_helper`` 的回退一致）。
_DEFAULT_PANEL_RES: tuple[int, int] = (1920, 1200)
_DEFAULT_BITS: int = 10

#: 每个 SLM 复用 / 重建的 PatternHelper 在 session_state 里的缓存键。
HELPER_CACHE_KEY = "slm{slm_num}_pattern_helper"


class ParamSpec:
    """单个图案参数的可渲染/可选配置。

    纯数据、无 ``st`` 依赖，供测试和 :func:`_render_params` 消费。
    """

    def __init__(
        self,
        name: str,
        label: str,
        kind: Literal["int", "float", "select", "seed"],
        default: float | int | str | None,
        *,
        min: float | int | None = None,
        max: float | int | None = None,
        step: float | None = None,
        options: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.label = label
        self.kind = kind
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.options = options


class MethodCall:
    """``PatternHelper`` 上的一次方法调用：``helper.<method>(**params)``。

    图案类型可含多步调用（如湍流先 ``init_turbulence_screen`` 再
    ``generate_turbulence_screen``）；最终输出取自**最后一次**调用的返回值。
    """

    def __init__(self, method: str, params: tuple[str, ...]) -> None:
        self.method = method
        self.params = params


class PatternTypeSpec:
    """一种图案类型的完整规格：显示名 + 一串对 helper 的方法调用。

    ``requires_init`` 为真表示生成前必须经过一次预初始化调用
    （当前仅湍流相位屏）。
    """

    def __init__(
        self,
        key: str,
        label: str,
        calls: tuple[MethodCall, ...],
    ) -> None:
        self.key = key
        self.label = label
        self.calls = calls

    @property
    def requires_init(self) -> bool:
        return len(self.calls) > 1

    @property
    def param_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for call in self.calls:
            for name in call.params:
                if name not in names:
                    names.append(name)
        return tuple(names)


#: 完整图案注册表：key → 规格。key 即 exposed 的图案类型标识。
PATTERN_TYPES: dict[str, PatternTypeSpec] = {
    "checkerboard": PatternTypeSpec(
        "checkerboard",
        "棋盘格",
        (MethodCall("generate_checkerboard", ("period",)),),
    ),
    "binary_grating": PatternTypeSpec(
        "binary_grating",
        "二元光栅",
        (
            MethodCall(
                "generate_binary_grating",
                ("a", "b", "direction"),
            ),
        ),
    ),
    "microlens_array": PatternTypeSpec(
        "microlens_array",
        "微透镜阵列",
        (
            MethodCall(
                "generate_microlens_array",
                ("lens_size", "focal_length", "wavelength", "pixel_size"),
            ),
        ),
    ),
    "turbulence": PatternTypeSpec(
        "turbulence",
        "湍流相位屏",
        (
            MethodCall(
                "init_turbulence_screen",
                ("r0", "L0", "pixel_scale", "random_seed"),
            ),
            MethodCall("generate_turbulence_screen", ()),
        ),
    ),
    "zernike": PatternTypeSpec(
        "zernike",
        "Zernike",
        (MethodCall("generate_zernike", ("n", "m", "amplitude")),),
    ),
    "focus": PatternTypeSpec(
        "focus",
        "聚焦",
        (MethodCall("generate_focus", ("focal_length", "lens_radius")),),
    ),
    "linear_grating": PatternTypeSpec(
        "linear_grating",
        "线性光栅",
        (
            MethodCall(
                "linear_grating",
                ("period", "phase_range", "direction"),
            ),
        ),
    ),
    "circular_grating": PatternTypeSpec(
        "circular_grating",
        "圆形光栅",
        (MethodCall("circular_grating", ("radius", "phase_range")),),
    ),
    "lens": PatternTypeSpec(
        "lens",
        "透镜",
        (MethodCall("lens", ("focal_length", "lens_radius")),),
    ),
    "vortex": PatternTypeSpec(
        "vortex",
        "涡旋相位",
        (MethodCall("generate_vortex", ("topological_charge",)),),
    ),
    "dammann": PatternTypeSpec(
        "dammann",
        "达曼光栅",
        (MethodCall("generate_dammann_grating", ("order", "fill_factor")),),
    ),
    "hologram": PatternTypeSpec(
        "hologram",
        "全息图",
        (
            MethodCall(
                "hologram",
                ("period", "phase_range", "direction"),
            ),
        ),
    ),
}

#: 每种图案类型的参数规格（按 key → 规格里的参数名对齐）。
PARAM_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "checkerboard": (ParamSpec("period", "周期 (像素)", "int", 100, min=1),),
    "binary_grating": (
        ParamSpec("a", "明条纹宽度", "int", 2, min=1),
        ParamSpec("b", "暗条纹宽度", "int", 3, min=1),
        ParamSpec(
            "direction",
            "方向",
            "select",
            "horizontal",
            options=("horizontal", "vertical"),
        ),
    ),
    "microlens_array": (
        ParamSpec("lens_size", "单透镜尺寸 (像素)", "int", 200, min=2),
        ParamSpec("focal_length", "焦距 (m)", "float", 0.1, min=1e-6, step=1e-3),
        ParamSpec("wavelength", "波长 (m)", "float", 532e-9, min=1e-9, step=1e-9),
        ParamSpec("pixel_size", "像素大小 (m)", "float", 8e-6, min=1e-9, step=1e-7),
    ),
    "turbulence": (
        ParamSpec("r0", "Fried 参数 r0 (m)", "float", 0.2, min=1e-3, step=1e-3),
        ParamSpec("L0", "外尺度 L0 (m)", "float", 25.0, min=0.1, step=0.5),
        ParamSpec(
            "pixel_scale", "每像素物理尺寸 (m)", "float", 8e-6, min=1e-9, step=1e-7
        ),
        ParamSpec("random_seed", "随机种子 (可选)", "seed", None, min=0),
    ),
    "zernike": (
        ParamSpec("n", "径向阶数 n", "int", 4, min=0, max=20),
        ParamSpec("m", "角向阶数 m", "int", 0, min=-20, max=20),
        ParamSpec("amplitude", "幅度", "float", 1.0, min=1e-3, step=0.1),
    ),
    "focus": (
        ParamSpec("focal_length", "焦距 (m)", "float", 0.5, min=1e-6, step=1e-3),
        ParamSpec("lens_radius", "透镜半径 (像素, 可选)", "float", None, min=1),
    ),
    "linear_grating": (
        ParamSpec("period", "周期 (像素)", "float", 100.0, min=0.1, step=1.0),
        ParamSpec("phase_range", "相位范围 (弧度)", "float", 2 * np.pi, min=1e-3),
        ParamSpec(
            "direction",
            "方向",
            "select",
            "vertical",
            options=("vertical", "horizontal"),
        ),
    ),
    "circular_grating": (
        ParamSpec("radius", "光栅半径 (像素)", "float", 100.0, min=0.1, step=1.0),
        ParamSpec("phase_range", "相位范围 (弧度)", "float", 2 * np.pi, min=1e-3),
    ),
    "lens": (
        ParamSpec("focal_length", "焦距 (m)", "float", 0.5, min=1e-6, step=1e-3),
        ParamSpec("lens_radius", "透镜半径 (像素, 可选)", "float", None, min=1),
    ),
    "vortex": (
        ParamSpec("topological_charge", "拓扑荷", "int", 1, min=-10, max=10, step=1),
    ),
    "dammann": (
        ParamSpec("order", "衍射级次", "int", 3, min=1, max=16),
        ParamSpec("fill_factor", "填充因子", "float", 0.5, min=0.0, max=1.0, step=0.05),
    ),
    "hologram": (
        ParamSpec("period", "周期 (像素)", "float", 100.0, min=0.1, step=1.0),
        ParamSpec("phase_range", "相位范围 (弧度)", "float", 2 * np.pi, min=1e-3),
        ParamSpec(
            "direction",
            "方向",
            "select",
            "vertical",
            options=("vertical", "horizontal"),
        ),
    ),
}


def _specs_for(key: str) -> tuple[ParamSpec, ...]:
    """返回图案类型 key 的参数规格（缺省空元组）。"""
    return PARAM_SPECS.get(key, ())


def generate_phase_radians(
    helper: PatternHelper,
    pattern_type: str,
    params: dict[str, Any],
) -> np.ndarray:
    """纯函数：按 ``pattern_type`` 的调用链在 ``helper`` 上生成弧度相位。

    校验 key 存在；逐条执行 :class:`MethodCall`，提前注入 ``params`` 里同名参数，
    缺失参数沿用各生成器的签名默认值。最终返回 helper 最后一次调用的
    ``float64 (height, width)`` 弧度数组。
    """
    spec = PATTERN_TYPES[pattern_type]
    phase: np.ndarray | None = None
    for call in spec.calls:
        kwargs = {name: params[name] for name in call.params if name in params}
        phase = getattr(helper, call.method)(**kwargs)
    assert phase is not None
    return phase


def _slm_panel_geometry(slm: Any) -> tuple[int, int, int]:
    """从 SLM 对象取面板几何，返回 ``(width, height, bits)``。

    与控制器 ``_build_control`` 一致：``Panel_Res`` 为 ``(width, height)``，
    位深取 ``Gray_Scale_bits``，缺省回退到 `_DEFAULT_PANEL_RES` / `_DEFAULT_BITS`。
    """
    panel_res = getattr(slm, "Panel_Res", _DEFAULT_PANEL_RES)
    width, height = int(panel_res[0]), int(panel_res[1])
    bits = int(getattr(slm, "Gray_Scale_bits", _DEFAULT_BITS))
    return width, height, bits


def _get_helper(slm_num: int, slm: Any) -> PatternHelper:
    """返回 SLM 的缓存 :class:`PatternHelper`（复用 ``session_state`` 缓存键）。

    与控制器 :func:`_get_slm_helper` 同键 ``slm{N}_pattern_helper``，面板几何或
    位深变化（例如重连不同设备）时重建。
    """
    width, height, bits = _slm_panel_geometry(slm)
    key = HELPER_CACHE_KEY.format(slm_num=slm_num)
    helper = st.session_state.get(key)
    if helper is None or helper.resolution != (width, height) or helper.bits != bits:
        helper = PatternHelper((width, height), bits=bits)
        st.session_state[key] = helper
    return helper


def _render_param(
    spec: ParamSpec,
    prefix: str,
) -> float | int | str | None:
    """渲染单个参数 widget，返回其当前值（随 Streamlit 默认逻辑读 session_state）。"""
    key = f"{prefix}_helper_{spec.name}"
    if spec.kind == "select":
        return st.selectbox(
            spec.label, options=spec.options, index=_index_of(spec), key=key
        )
    if spec.kind == "seed":
        value: Any = st.session_state.get(key, spec.default)
        return st.number_input(
            spec.label,
            value=value,
            min_value=spec.min,
            step=1,
            key=key,
        )
    if spec.kind == "int":
        return st.number_input(
            spec.label,
            value=int(spec.default) if spec.default is not None else None,
            min_value=spec.min,
            max_value=spec.max,
            step=int(spec.step) if spec.step is not None else 1,
            key=key,
        )
    return st.number_input(
        spec.label,
        value=float(spec.default) if spec.default is not None else None,
        min_value=spec.min,
        max_value=spec.max,
        step=spec.step,
        key=key,
    )


def _index_of(spec: ParamSpec) -> int | None:
    default = spec.default
    assert isinstance(default, str), "select 参数 default 必须是 str"
    if default in spec.options:
        return spec.options.index(default)
    return None


def _render_params(prefix: str, pattern_type: str) -> dict[str, Any]:
    """按注册表渲染 ``pattern_type`` 的全部参数 widget。"""
    specs = _specs_for(pattern_type)
    return {spec.name: _render_param(spec, prefix) for spec in specs}


def _display_phase(slm: Any, phase_radians: np.ndarray, slm_num: int) -> None:
    """把弧度相位经驱动下发到面板（与控制器同一条路径，不二次施加平移）。

    用 ``slm.create_phase_from_array`` 完成 弧度→灰度（内部含 shift + 矫正 +
    LUT），再 ``display_data`` 写入并立即显示 —— 平移只在这里应用一次。
    """
    phase_gray = slm.create_phase_from_array(phase_radians)
    slm.display_data(phase_gray)


def render_pattern_helper_tab(slm_num: int) -> None:
    """tab 入口：直接暴露 :class:`PatternHelper` 的相位生成器。

    Args:
        slm_num: SLM 序号，session_state 前缀 ``slm{slm_num}``。
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is None:
        st.info(f"尚未连接 SLM —— {CONNECT_HINT}, 连上后这里出现相位生成。")
        return

    helper = _get_helper(slm_num, slm)
    width, height, bits = _slm_panel_geometry(slm)
    st.caption(
        f"PatternHelper @ {width}×{height} ×{bits} bit（分辨率 {helper.resolution}）"
    )

    pattern_keys = list(PATTERN_TYPES)
    pattern_type: str = st.selectbox(
        "图案类型",
        options=pattern_keys,
        index=0,
        key=f"{prefix}_helper_pattern_type",
        format_func=lambda k: PATTERN_TYPES[k].label,
    )

    params = _render_params(prefix, pattern_type)

    generated: np.ndarray | None = None
    if st.button("生成预览", key=f"{prefix}_helper_preview_btn", type="secondary"):
        try:
            phase = generate_phase_radians(helper, pattern_type, params)
            if pattern_type == "turbulence":
                st.info("湍流相位屏已按当前 r0/L0 重新初始化解偶生成。")
            gray = helper.to_uint16(phase)
            st.image(
                gray,
                caption=f"图案「{PATTERN_TYPES[pattern_type].label}」预览",
                clamp=True,
            )
            generated = phase
        except Exception as e:
            st.error(f"生成相位失败: {e}")
            logger.exception(f"Pattern helper preview failed for SLM {slm_num}: {e}")

    if st.button("下发到 SLM", key=f"{prefix}_helper_display_btn", type="secondary"):
        phase = (
            generated
            if generated is not None
            else generate_phase_radians(helper, pattern_type, params)
        )
        try:
            _display_phase(slm, phase, slm_num)
            st.success(
                f"相位已下发到 SLM {slm_num} 并显示（灰度经 create_phase_from_array 编码）。"
            )
        except Exception as e:
            st.error(f"下发失败: {e} —— 请先回到「生成预览」确认参数后重试。")
            logger.exception(f"Pattern helper display failed for SLM {slm_num}: {e}")
