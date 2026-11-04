"""``helper_tab`` 的离线单测：直通 PatternHelper API 的图案注册表、参数完整性、
纯生成 round-trip、``to_uint16`` 映射、以及 (无/有 SLM) 两态下 ``st`` 的替身渲染。

沿用 ``test_pattern_controls`` / ``test_multi_slm_controller_empty_states`` 的方法：
用最小记录式 ``st`` 替身经 ``monkeypatch.setattr(module, "st", stub)`` 注入，
不驱动真实 Streamlit。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ao_shaping.gui.slm import helper_tab as tab
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

SOURCE = Path(tab.__file__)

# 注册表里出现的图案类型 key —— 与 PatternHelper 的实际生成方法一一对应。
EXPOSED_KEYS = set(tab.PATTERN_TYPES)
#: 契约要求暴露的核心（含其方法签名与 PatternHelper 源码对齐的）key。
REQUIRED_CORE_KEYS = {
    "checkerboard",
    "binary_grating",
    "microlens_array",
    "zernike",
    "turbulence",
    "focus",
    "linear_grating",
    "circular_grating",
    "lens",
    "vortex",
    "dammann",
}

#: 这些生成器不依赖可变全局 / 可选依赖，可离线验证 round-trip。
_OFFLINE_GENERATORS = {
    "checkerboard",
    "binary_grating",
    "microlens_array",
    "focus",
    "linear_grating",
    "circular_grating",
    "lens",
    "vortex",
    "dammann",
}


class _St:
    """记录式 streamlit 替身：被调用过的 vtable + 记录的 / 值 / 按钮配置。"""

    def __init__(self, session_state: dict[str, Any]) -> None:
        self.session_state = session_state
        self.infos: list[str] = []
        self.captions: list[str] = []
        self.buttons: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self._selectbox_calls: list[dict[str, Any]] = []
        self._num_calls: list[dict[str, Any]] = []
        self._next_value = False

    def _bump(self) -> bool:
        # 用「第二个按钮才按下」模拟生成后再下发。
        self._next_value = not self._next_value
        return bool(self._next_value)

    def info(self, body: str, **kw: Any) -> None:
        self.infos.append(str(body))

    def caption(self, body: str, **kw: Any) -> None:
        self.captions.append(str(body))

    def button(self, label: str, **kw: Any) -> bool:
        self.buttons.append({"label": str(label), **kw})
        return False

    def image(self, *args: Any, **kw: Any) -> None:
        self.images.append({"args": args, "kw": kw})

    def error(self, body: str, **kw: Any) -> None:
        self.errors.append(str(body))

    def success(self, body: str, **kw: Any) -> None:
        self.successes.append(str(body))

    def selectbox(self, label: str, options: Any, **kw: Any) -> Any:
        self._selectbox_calls.append(
            {"label": str(label), "options": list(options), **kw}
        )
        return options[0] if options else None

    def number_input(self, label: str, **kw: Any) -> Any:
        self._num_calls.append({"label": str(label), **kw})
        return kw.get("value")


# ---------------------------------------------------------------------------
# 注册表完整性
# ---------------------------------------------------------------------------


def test_registry_exposes_required_core_keys() -> None:
    assert REQUIRED_CORE_KEYS <= EXPOSED_KEYS


def test_every_registry_key_has_a_param_spec() -> None:
    for key in EXPOSED_KEYS:
        assert key in tab.PARAM_SPECS, f"{key} 缺 param spec"
        assert tab.PATTERN_TYPES[key].label, f"{key} 缺显示名"


def test_every_param_spec_key_matches_a_registry_key() -> None:
    assert set(tab.PARAM_SPECS) == EXPOSED_KEYS


def test_param_names_match_declared_generator_signatures() -> None:
    """每个图案类型声明的参数名必须能被 PatternHelper 的方法调用接收。"""
    for key, spec in tab.PATTERN_TYPES.items():
        declared = {p.name for p in tab.PARAM_SPECS[key]}
        for call in spec.calls:
            method = getattr(PatternHelper, call.method)
            import inspect

            params = inspect.signature(method).parameters
            for name in call.params:
                assert name in params, f"{key}/{call.method} 参数 {name} 不在签名里"
                # 参数名必须来自注册表，且不得重复声明到别的调用
                assert name in declared, (
                    f"{key}/{call.method} 参数 {name} 未在 spec 声明"
                )
        assert declared == set(spec.param_names), f"{key} 的 spec 参数集与调用链不一致"


# ---------------------------------------------------------------------------
# 参数规格完整性
# ---------------------------------------------------------------------------


def _spec_for(key: str) -> dict[str, Any]:
    return {p.name: p for p in tab.PARAM_SPECS[key]}


def test_defaults_are_present_where_required() -> None:
    """除可选(标注「可选」或 seed)外，每个 spec 的 int/float 必须有 default 和范围。"""
    OPTIONAL_MARKERS = ("可选", "seed")
    for key, specs in tab.PARAM_SPECS.items():
        for p in specs:
            if p.kind == "seed":
                continue
            if p.kind in {"int", "float"} and any(
                m in p.label for m in OPTIONAL_MARKERS
            ):
                # 可选参数 default=None 是显式设计（生成器有内部默认）。
                continue
            if p.kind == "int" or p.kind == "float":
                assert p.default is not None, f"{key}.{p.name} 缺 default"
                assert p.min is not None, f"{key}.{p.name} 缺 min"
            elif p.kind == "select":
                assert p.default in p.options, f"{key}.{p.name} select 默认不在选项"


def test_turbulence_marked_requires_init() -> None:
    spec = tab.PATTERN_TYPES["turbulence"]
    assert spec.requires_init is True
    assert [c.method for c in spec.calls] == [
        "init_turbulence_screen",
        "generate_turbulence_screen",
    ]


def test_non_turbulence_do_not_require_init() -> None:
    for key, spec in tab.PATTERN_TYPES.items():
        if key != "turbulence":
            assert spec.requires_init is False, f"{key} 不应标记 requires_init"


# ---------------------------------------------------------------------------
# 生成 round-trip（离线、不需要 aotools）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(_OFFLINE_GENERATORS))
def test_offline_generator_round_trip(key: str) -> None:
    helper = PatternHelper((64, 64), bits=10)
    params = {p.name: p.default for p in tab.PARAM_SPECS[key] if p.default is not None}
    # 可选 radius 默认 None → 传 NaN 会破坏；这里用 spec 默认即可。
    phase = tab.generate_phase_radians(helper, key, params)
    assert isinstance(phase, np.ndarray)
    assert phase.dtype == np.float64
    # PatternHelper 输出 shape 为 (height, width)。
    assert phase.shape == (64, 64), f"{key} shape {phase.shape} != (64, 64)"


@pytest.mark.parametrize("key", sorted(["zernike", "microlens_array"]))
def test_generators_return_float64_panel_shape(key: str) -> None:
    """带可选依赖 / 需要显式参数的类型也要返回 float64 面板形状。"""
    helper = PatternHelper((64, 64), bits=10)
    params = {p.name: p.default for p in tab.PARAM_SPECS[key] if p.default is not None}
    phase = tab.generate_phase_radians(helper, key, params)
    assert phase.dtype == np.float64
    assert phase.shape == (64, 64)


# ---------------------------------------------------------------------------
# to_uint16 映射
# ---------------------------------------------------------------------------


def test_to_uint16_mapping() -> None:
    helper = PatternHelper((16, 16), bits=10)
    max_gray = helper._max_val  # 2**10 - 1 = 1023
    # to_uint16 先转 float32 再 mod 2π，故在 float32 域取「略小于 2π」，
    # 否则 float64 的 nextafter 经 float32 舍入会回到 2π → 折叠为 0。
    below_2pi = np.float32(2 * np.pi)
    half = np.float32(1.0)
    phase = np.array(
        [[0.0, float(np.nextafter(below_2pi, half)), np.pi]],
        dtype=np.float64,
    )
    gray = helper.to_uint16(phase)
    assert gray.dtype == np.uint16
    # to_uint16 折叠到 [0, 2π), 因此 0 → 0；略小于 2π → 接近满量程。
    assert int(gray[0, 0]) == 0
    assert int(gray[0, 1]) in (max_gray - 1, max_gray)
    assert 0 <= int(gray[0, 2]) <= max_gray


def test_helper_to_uint16_max_param_contract() -> None:
    """位深为 N 时「略小于 2π」映射到接近 2**N - 1。"""
    below_2pi = np.float32(2 * np.pi)
    half = np.float32(1.0)
    for bits in (8, 10, 12):
        helper = PatternHelper((8, 8), bits=bits)
        out = helper.to_uint16(np.array([[float(np.nextafter(below_2pi, half))]]))
        assert int(out[0, 0]) in (2**bits - 2, 2**bits - 1)


# ---------------------------------------------------------------------------
# 渲染 (st 替身)
# ---------------------------------------------------------------------------


def _stub_with_slm() -> MagicMock:
    slm = MagicMock()
    slm.Panel_Res = (64, 32)
    slm.Gray_Scale_bits = 10
    slm.create_phase_from_array.return_value = np.zeros((32, 64), dtype=np.uint16)
    return slm


def test_render_no_slm_mentions_connect_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _St({})
    monkeypatch.setattr(tab, "st", stub)
    tab.render_pattern_helper_tab(1)
    assert len(stub.infos) == 1
    assert tab.CONNECT_HINT in stub.infos[0]


def test_render_no_slm_registers_no_divider_or_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _St({})
    monkeypatch.setattr(tab, "st", stub)
    tab.render_pattern_helper_tab(1)
    assert stub.buttons == []
    assert stub.images == []
    assert not hasattr(stub, "divider")


def test_render_with_slm_registers_prefixed_keys_and_no_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _St({"slm1": _stub_with_slm()})
    monkeypatch.setattr(tab, "st", stub)
    tab.render_pattern_helper_tab(1)

    # 两个按钮都是 secondary、前缀一致。
    assert len(stub.buttons) == 2
    assert {b["label"] for b in stub.buttons} == {"生成预览", "下发到 SLM"}
    for b in stub.buttons:
        assert b.get("type") != "primary"
        assert b["key"].startswith("slm1_")

    # 图案类型 selectbox key 带前缀。
    assert any(call["key"].startswith("slm1_") for call in stub._selectbox_calls)


# ---------------------------------------------------------------------------
# 源码纪律：无 divider、无 primary
# ---------------------------------------------------------------------------


def _tree():
    text = SOURCE.read_text(encoding="utf-8")
    return ast.parse(text)


def test_module_source_uses_no_divider() -> None:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "divider", f"L{node.lineno} 出现 st.divider()"


def test_module_source_has_no_primary_button() -> None:
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords or []:
            if kw.arg == "type" and isinstance(kw.value, ast.Constant):
                assert kw.value.value != "primary", (
                    f"L{node.lineno} 出现 type='primary'"
                )
