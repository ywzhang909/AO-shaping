"""Issue #215 的控件级规则在 ``multi_slm_controller`` 上的守卫。

三条都是「读代码看不出来、改代码很容易改回去」的约定, 所以钉成测试:

* 规则 6（按钮分级）: 一个渲染函数里最多一个 ``type="primary"``。
  这一屏只有「从模式生成器生成相位」是主按钮, 侧栏只有「连接」是主按钮;
  任何新加的 primary 都必须先在这里改写预期, 让人停下来想一次。
* 规则 19（分隔线纪律）: 本模块不再有 ``st.divider()``。三层相位叠加
  用标题行区分, 横线只留给跨任务的边界。
* 规则 20（空状态引导）: 「未连接」不能是一句话的全部 —— 每条都指向
  侧栏的「SLM 设备」表格（:data:`CONNECT_HINT`）或点名一个具体控件。

为什么用 AST 而不是正则: 上面三条说的是「渲染函数里有没有这个关键字」,
正则会把注释里的同名文字也算进去, 而 ``str.replace`` 式的检查又查不出
藏在 f-string 里的引导语。
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ao_shaping.gui.slm import multi_slm_controller as gui

SOURCE = Path(gui.__file__)


class _St:
    """只记录 warning / info / title 的最小 streamlit 替身。"""

    def __init__(self, session_state: dict[str, Any]) -> None:
        self.session_state = session_state
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.titles: list[str] = []

    @contextmanager
    def _cm(self):
        yield self

    @property
    def sidebar(self):
        return self._cm()

    def set_page_config(self, **kwargs: Any) -> None:
        return None

    def title(self, body: str) -> None:
        self.titles.append(str(body))

    def warning(self, body: str, **kwargs: Any) -> None:
        self.warnings.append(str(body))

    def info(self, body: str, **kwargs: Any) -> None:
        self.infos.append(str(body))

    def columns(self, spec: Any, **kwargs: Any):
        count = spec if isinstance(spec, int) else len(spec)
        return tuple(self._cm() for _ in range(count))

    def write(self, body: Any, **kwargs: Any) -> None:
        return None

    def caption(self, body: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, body: Any, **kwargs: Any) -> None:
        return None

    def success(self, body: Any, **kwargs: Any) -> None:
        return None

    def error(self, body: Any, **kwargs: Any) -> None:
        return None


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _St:
    recorder = _St({})
    monkeypatch.setattr(gui, "st", recorder)
    return recorder


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 规则 20 —— 空状态说清下一步
# ---------------------------------------------------------------------------

def test_connect_hint_names_the_sidebar_control() -> None:
    """引导语必须点名操作员眼前那一列, 不能只说「请连接」。"""
    assert "「SLM 设备」" in gui.CONNECT_HINT
    assert "「连接」" in gui.CONNECT_HINT


def test_disconnected_status_says_where_to_connect(stub: _St) -> None:
    gui.display_slm_status(1)
    assert stub.warnings == [f"未连接 —— {gui.CONNECT_HINT}"]


def test_apply_correction_without_device_says_where_to_connect(stub: _St) -> None:
    gui._apply_uploaded_correction(1, None)
    assert len(stub.warnings) == 1
    assert gui.CONNECT_HINT in stub.warnings[0]


def test_clear_correction_without_device_says_where_to_connect(stub: _St) -> None:
    gui._clear_correction(1)
    assert len(stub.warnings) == 1
    assert gui.CONNECT_HINT in stub.warnings[0]


def test_save_phase_without_device_says_where_to_connect(stub: _St) -> None:
    gui._save_slm_phase_csv(1)
    assert len(stub.warnings) == 1
    assert gui.CONNECT_HINT in stub.warnings[0]


def test_save_config_without_device_says_where_to_connect(stub: _St) -> None:
    gui._save_slm_config(1, "slm1")
    assert len(stub.infos) == 1
    assert gui.CONNECT_HINT in stub.infos[0]


def test_start_toggle_without_slots_names_the_two_set_buttons(stub: _St) -> None:
    """「请先设置相位 A 和相位 B」要说出按哪两个按钮。"""
    gui._start_phase_toggle(1, "slm1")
    assert len(stub.warnings) == 1
    assert "「设为相位 A」" in stub.warnings[0]
    assert "「设为相位 B」" in stub.warnings[0]


def test_main_without_connection_says_where_to_connect(
    stub: _St, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一屏都没有连接时, 主区的空状态同样指向侧栏那一列。"""
    stub.session_state["available_slms"] = []
    monkeypatch.setattr(gui, "render_slm_sidebar", lambda: None)
    gui.main()
    assert stub.titles == ["SLM200 控制器"]
    assert len(stub.infos) == 1
    assert gui.CONNECT_HINT in stub.infos[0]


def test_every_disconnected_message_points_at_a_control() -> None:
    """扫一遍源码: 每条含「未连接」的 warning / info 都要给出下一步。"""
    offenders: list[str] = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in {
            "warning",
            "info",
        }:
            continue
        if not node.args:
            continue
        text = ast.unparse(node.args[0])
        if "未连接" not in text:
            continue
        if "CONNECT_HINT" not in text and "「" not in text:
            offenders.append(f"L{node.lineno}: {text}")
    assert offenders == [], (
        "空状态只说了「未连接」, 没说下一步点哪里 (issue #215 规则 20):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 规则 6 —— 每屏 ≤1 primary
# ---------------------------------------------------------------------------

def _primary_button_counts(tree: ast.Module) -> dict[str, int]:
    """每个顶层函数里 ``st.button(..., type="primary")`` 的个数。"""
    counts: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        total = 0
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "button":
                continue
            for keyword in call.keywords:
                if (
                    keyword.arg == "type"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "primary"
                ):
                    total += 1
        if total:
            counts[node.name] = total
    return counts


def test_only_one_primary_per_render_function() -> None:
    assert _primary_button_counts(_tree()) == {
        # 侧栏这一屏的起点: 没有连接就没有后面的任何操作。
        "render_slm_sidebar": 1,
        # 主区这一屏的目的动作: 生成并下发相位。
        "_render_phase_generate_section": 1,
    }


def test_phase_toggle_pair_is_not_primary() -> None:
    """设 A/B 是准备步骤, 排在「生成相位」之后, 不能再抢成 primary。"""
    tree = _tree()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_render_phase_toggle_section":
            primaries = [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "button"
                for keyword in call.keywords
                if keyword.arg == "type"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "primary"
            ]
            assert primaries == []
            return
    pytest.fail("_render_phase_toggle_section 不在模块顶层 (改名了?)")


# ---------------------------------------------------------------------------
# 规则 19 —— 分隔线纪律
# ---------------------------------------------------------------------------

def test_module_uses_no_dividers() -> None:
    """三层叠加用标题行区分后, 本模块不该再出现分隔线。"""
    dividers = [
        node.lineno
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "divider"
    ]
    assert dividers == [], f"st.divider() 应改为标题行或删掉: L{dividers}"


def test_overlay_layers_have_headings() -> None:
    """替代分隔线的是三层各自的小标题 —— 三层都必须在, 否则又变成一片糊。"""
    source = SOURCE.read_text(encoding="utf-8")
    for label in ("底相位", "载波（倾斜）", "离焦"):
        assert f'_overlay_layer_heading("{label}")' in source
