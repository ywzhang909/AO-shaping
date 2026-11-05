"""``PatternHelper`` 相位生成器的纯规格注册表（不 import Streamlit）。

为什么单独成模块
----------------
``PatternHelper`` 的生成方法原先只被「相位控制」tab 的控件树
（:mod:`ao_shaping.gui.slm.pattern_controls`）使用，控件树把「参数名 / 默认值 /
范围」散落在每个 ``PatternControl`` 子类里，并与 ``st.number_input`` 等控件绑定。
那个形态无法被桌面端（PySide6）复用，也无法脱离 Streamlit 做单元测试。

本模块把同一份信息抽成**纯数据**：``PATTERN_TYPES``（图案类型 → 方法调用链）与
``PARAM_SPECS``（参数名 → 控件描述），再加一个纯函数
:func:`generate_phase_radians`。它只依赖 ``numpy``，因此
:mod:`~ao_shaping.gui.slm.helper_tab`（Streamlit）与桌面页共享同一份规格，
不会各自漂移。

设计契约
--------
* ``ParamSpec.name`` **必须与方法形参同名**——它是直接按关键字转发给
  ``PatternHelper`` 方法的；显示用的中文单位写在 ``label`` 里。
* 单位换算只发生在 :func:`generate_phase_radians` 的转发层（``_coerce``），
  因为这些换算在 ``PatternHelper`` 里没有承载位置，而 UI 又不能直接让人填
  米/秒量级的小数。
* 生成器一律返回 **raw 未包裹弧度** 的 ``float64`` 面板，形状 ``(height, width)``；
  包裹与灰度化只发生在驱动层。
* 不使用可变默认值以外的模块级状态，``turbulence`` 是唯一需要先初始化的类型
  （``requires_init``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "MethodCall",
    "ParamSpec",
    "PatternTypeSpec",
    "PATTERN_TYPES",
    "PARAM_SPECS",
    "generate_phase_radians",
    "specs_for",
]


@dataclass(frozen=True)
class ParamSpec:
    """单个参数的控件描述。

    Attributes:
        name: 方法形参名，必须与 ``PatternHelper`` 目标方法的形参逐字一致。
        label: 中文显示名（含单位）。可选参数须在文案里写「可选」。
        kind: ``int`` / ``float`` / ``select`` / ``seed``。
        default: 默认值；``select`` 必须在 ``options`` 内。
        min: 下限（``int`` / ``float`` 必填，被测试强制）。
        max: 上限。
        step: 步进。
        options: ``select`` 的候选项。
    """

    name: str
    label: str
    kind: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MethodCall:
    """对 ``PatternHelper`` 方法的一次调用。"""

    method: str
    params: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PatternTypeSpec:
    """一个图案类型：显示名、调用链、参数名全集、是否需先初始化。"""

    label: str
    calls: tuple[MethodCall, ...]
    param_names: tuple[str, ...]
    requires_init: bool = False


# ---------------------------------------------------------------------------
# 单位换算：只按「方法形参名」查表，因此同名参数在所有图案里必须同义。
# focal_length 一律 mm，wavelength 一律 nm，pixel_size / pixel_scale 一律 um，
# phase_range 一律以 π 为单位（与 pattern_controls 的控件文案一致）。
# ---------------------------------------------------------------------------
_UNIT_SCALE: dict[str, float] = {
    "focal_length": 1e-3,  # mm -> m
    "wavelength": 1e-9,  # nm -> m
    "pixel_size": 1e-6,  # um -> m
    "pixel_scale": 1e-6,  # um -> m
    "phase_range": float(np.pi),  # 以 π 为单位 -> 弧度
}

#: 需要整数化的形参（其余按浮点转发；字符串原样透传）。
_INT_PARAMS: frozenset[str] = frozenset(
    {
        "period",
        "a",
        "b",
        "lens_size",
        "n",
        "m",
        "topological_charge",
        "order",
        "random_seed",
    }
)


def _coerce(name: str, value: Any) -> Any:
    """把 UI 读数换成方法需要的量纲（见 ``_UNIT_SCALE``）。"""
    if isinstance(value, str):
        return value
    if name in _INT_PARAMS:
        return int(value)
    return float(value) * _UNIT_SCALE.get(name, 1.0)


# ---------------------------------------------------------------------------
# 图案类型注册表
# ---------------------------------------------------------------------------
PATTERN_TYPES: dict[str, PatternTypeSpec] = {
    "linear_grating": PatternTypeSpec(
        label="线性光栅",
        calls=(MethodCall("linear_grating", ("period", "phase_range")),),
        param_names=("period", "phase_range"),
    ),
    "circular_grating": PatternTypeSpec(
        label="圆形光栅",
        calls=(MethodCall("circular_grating", ("radius", "phase_range")),),
        param_names=("radius", "phase_range"),
    ),
    "lens": PatternTypeSpec(
        label="透镜",
        calls=(
            MethodCall(
                "lens",
                ("focal_length", "wavelength", "pixel_size", "lens_radius"),
            ),
        ),
        param_names=("focal_length", "wavelength", "pixel_size", "lens_radius"),
    ),
    "checkerboard": PatternTypeSpec(
        label="棋盘格",
        calls=(MethodCall("generate_checkerboard", ("period",)),),
        param_names=("period",),
    ),
    "binary_grating": PatternTypeSpec(
        label="二元光栅",
        calls=(MethodCall("generate_binary_grating", ("a", "b", "direction")),),
        param_names=("a", "b", "direction"),
    ),
    "microlens_array": PatternTypeSpec(
        label="微透镜阵列",
        calls=(
            MethodCall(
                "generate_microlens_array",
                ("lens_size", "focal_length", "wavelength", "pixel_size"),
            ),
        ),
        param_names=("lens_size", "focal_length", "wavelength", "pixel_size"),
    ),
    "turbulence": PatternTypeSpec(
        label="湍流相位屏",
        calls=(
            MethodCall(
                "init_turbulence_screen",
                ("r0", "L0", "pixel_scale", "random_seed"),
            ),
            MethodCall("generate_turbulence_screen", ()),
        ),
        param_names=("r0", "L0", "pixel_scale", "random_seed"),
        requires_init=True,
    ),
    "zernike": PatternTypeSpec(
        label="Zernike",
        calls=(MethodCall("generate_zernike", ("n", "m", "amplitude", "radius")),),
        param_names=("n", "m", "amplitude", "radius"),
    ),
    "dammann": PatternTypeSpec(
        label="达曼光栅",
        calls=(MethodCall("generate_dammann_grating", ("order", "fill_factor")),),
        param_names=("order", "fill_factor"),
    ),
    "vortex": PatternTypeSpec(
        label="涡旋相位",
        calls=(
            MethodCall(
                "generate_vortex",
                ("topological_charge", "wavelength", "pixel_size"),
            ),
        ),
        param_names=("topological_charge", "wavelength", "pixel_size"),
    ),
    "focus": PatternTypeSpec(
        label="聚焦",
        calls=(
            MethodCall(
                "generate_focus",
                ("focal_length", "wavelength", "pixel_size", "lens_radius"),
            ),
        ),
        param_names=("focal_length", "wavelength", "pixel_size", "lens_radius"),
    ),
}


# ---------------------------------------------------------------------------
# 参数规格（默认值与范围取自 pattern_controls 的 DEFAULTS / RANGES）
# ---------------------------------------------------------------------------
PARAM_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "linear_grating": (
        ParamSpec("period", "周期 (像素)", "float", 64.0, 1.0, 1000.0, 1.0),
        ParamSpec("phase_range", "相位范围 (π)", "float", 2.0, 0.05, 2.0, 0.1),
    ),
    "circular_grating": (
        ParamSpec("radius", "圆形周期半径 (像素)", "float", 64.0, 1.0, 2000.0, 10.0),
        ParamSpec("phase_range", "相位范围 (π)", "float", 2.0, 0.05, 2.0, 0.1),
    ),
    "lens": (
        ParamSpec("focal_length", "焦距 (mm)", "float", 300.0, 1.0, 100000.0, 10.0),
        ParamSpec("wavelength", "波长 (nm)", "float", 1064.0, 400.0, 1600.0, 1.0),
        ParamSpec("pixel_size", "像素间距 (um)", "float", 8.0, 0.1, 100.0, 0.1),
        ParamSpec(
            "lens_radius",
            "透镜半径 (像素)（可选，留空=全孔径）",
            "float",
            None,
            1.0,
            2000.0,
            1.0,
        ),
    ),
    "checkerboard": (
        ParamSpec("period", "棋盘格周期 (像素)", "int", 50, 1, 1000, 1),
    ),
    "binary_grating": (
        ParamSpec("a", "亮条纹宽度 a (像素)", "int", 2, 1, 1000, 1),
        ParamSpec("b", "暗条纹宽度 b (像素)", "int", 3, 1, 1000, 1),
        ParamSpec(
            "direction",
            "方向",
            "select",
            "horizontal",
            options=("horizontal", "vertical"),
        ),
    ),
    "microlens_array": (
        ParamSpec("lens_size", "微透镜尺寸 (像素)", "int", 16, 8, 1000, 1),
        ParamSpec("focal_length", "焦距 (mm)", "float", 300.0, 1.0, 100000.0, 1.0),
        ParamSpec("wavelength", "波长 (nm)", "float", 1064.0, 400.0, 1600.0, 1.0),
        ParamSpec("pixel_size", "像素间距 (um)", "float", 8.0, 0.1, 100.0, 0.1),
    ),
    "turbulence": (
        ParamSpec("r0", "Fried 参数 r0 (m)", "float", 0.05, 1e-4, 10.0, 0.01),
        ParamSpec("L0", "外尺度 L0 (m)", "float", 10.0, 0.1, 10000.0, 1.0),
        ParamSpec("pixel_scale", "像素尺度 (um)", "float", 8.0, 0.1, 100.0, 0.1),
        ParamSpec("random_seed", "随机种子", "seed", None, 0),
    ),
    "zernike": (
        ParamSpec("n", "径向阶 n", "int", 2, 0, 10, 1),
        ParamSpec("m", "角向阶 m", "int", 0, -10, 10, 1),
        ParamSpec("amplitude", "幅度 (弧度)", "float", 1.0, -100.0, 100.0, 0.1),
        ParamSpec(
            "radius",
            "孔径半径 (像素)（可选，留空=自适应）",
            "float",
            None,
            1.0,
            2000.0,
            1.0,
        ),
    ),
    "dammann": (
        ParamSpec("order", "衍射级数", "int", 3, 2, 8, 1),
        ParamSpec("fill_factor", "填充因子", "float", 0.5, 0.1, 1.0, 0.1),
    ),
    "vortex": (
        ParamSpec("topological_charge", "拓扑荷", "int", 1, -10, 10, 1),
        ParamSpec("wavelength", "波长 (nm)", "float", 1064.0, 400.0, 1600.0, 1.0),
        ParamSpec("pixel_size", "像素间距 (um)", "float", 8.0, 1.0, 100.0, 1.0),
    ),
    "focus": (
        ParamSpec("focal_length", "焦距 (mm)", "float", 300.0, 1.0, 100000.0, 10.0),
        ParamSpec("wavelength", "波长 (nm)", "float", 1064.0, 400.0, 1600.0, 1.0),
        ParamSpec("pixel_size", "像素间距 (um)", "float", 8.0, 0.1, 100.0, 0.1),
        ParamSpec(
            "lens_radius",
            "透镜半径 (像素)（可选，留空=全孔径）",
            "float",
            None,
            1.0,
            2000.0,
            1.0,
        ),
    ),
}


def specs_for(pattern_type: str) -> tuple[ParamSpec, ...]:
    """返回图案类型的参数规格（未知 key 回退空元组，便于 UI 渲染空参数）。"""
    return PARAM_SPECS.get(pattern_type, ())


def generate_phase_radians(
    helper: Any,
    pattern_type: str,
    params: dict[str, Any],
) -> np.ndarray:
    """按注册表调用 ``helper`` 生成 raw 弧度相位。

    Args:
        helper: 已构造的 :class:`PatternHelper`。
        pattern_type: ``PATTERN_TYPES`` 的 key。
        params: 参数名 → UI 读数；``None`` 的项不转发（用方法自身默认）。

    Returns:
        ``float64`` 面板，形状 ``(height, width)``，未包裹、未归一化。

    Raises:
        KeyError: 未知 ``pattern_type``。
        RuntimeError: 调用链没有返回相位数组。
    """
    spec = PATTERN_TYPES[pattern_type]
    result: Any = None
    for call in spec.calls:
        kwargs = {
            name: _coerce(name, params[name])
            for name in call.params
            if params.get(name) is not None
        }
        result = getattr(helper, call.method)(**kwargs)
    if result is None:
        raise RuntimeError(f"图案 {pattern_type} 的调用链没有返回相位数组")
    return np.asarray(result, dtype=np.float64)
