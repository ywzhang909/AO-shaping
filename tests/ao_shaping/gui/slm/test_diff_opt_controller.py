"""AppTest coverage for the dedicated differentiable-optimization GUI.

三件事必须锁住（都是实测踩过的）：

1. **不撞 widget key** —— 侧栏 ``render_slm_sidebar()`` 内部已经渲染「CCD参数设置」，
   主体再调一次 ``render_bench_settings()`` 会以 ``bench_cam_type`` 重复直接崩页
   （2026-10-27 实测 ``StreamlitDuplicateElementKey``）。
2. **无设备时给可执行提示**，不抛异常、不报 error。
3. **委托而非复刻** —— 参数区与运行都来自 ``shaping_controls.TransferMethod``
   （同名同 key、同一条带设备交还的 ``ShapingMethod.run``）。
"""

from __future__ import annotations

import inspect

import pytest
from streamlit.testing.v1 import AppTest

from ao_shaping.gui.slm import diff_opt_controller, shaping_controls
from ao_shaping.gui.slm.shaping_controls import TRANSFER_RECIPE, TransferMethod

APP = "src/ao_shaping/gui/slm/diff_opt_controller.py"


@pytest.fixture()
def page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


# --------------------------------------------------------------------------- #
# 1. 页面本身
# --------------------------------------------------------------------------- #
def test_page_renders_without_duplicate_widget_keys(page: AppTest) -> None:
    assert not page.exception
    # 「相机类型」只属于侧栏 bench 区块；主体不得再声明一份。
    assert [el.label for el in page.selectbox].count("相机类型") == 1


def test_page_has_no_error_mark(page: AppTest) -> None:
    assert not page.exception
    assert not page.error


def test_page_shows_connect_hint_without_a_device(page: AppTest) -> None:
    assert any("连接至少一个 SLM" in info.value for info in page.info)


# --------------------------------------------------------------------------- #
# 2. 委托契约（不建第二套渲染/编排）
# --------------------------------------------------------------------------- #
def test_page_module_delegates_instead_of_reimplementing() -> None:
    import ast

    tree = ast.parse(inspect.getsource(diff_opt_controller))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    # 光路控件只在侧栏出现一次：页面模块不得自己调 render_bench_settings。
    assert "render_bench_settings" not in imported
    assert "render_bench_settings" not in called
    # 编排走 ShapingMethod.run（设备交还 + 接受相位下发）。
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        for node in ast.walk(tree)
    )
    # 页面不得自己 import 迁移管线 —— 那是 execute() 的职责。
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    modules |= {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    }
    assert not any("tools.slm.transfer" in m for m in modules)


def test_page_uses_the_same_widget_keys_as_closed_loop_shaping() -> None:
    method = TransferMethod("slm1", 1)
    assert method.wkey("shapes") == "slm1_shp_transfer_shapes"
    assert method.wkey("blocks") == "slm1_shp_transfer_blocks"
    assert method.wkey("objective") == "slm1_shp_transfer_objective"


def test_transfer_method_is_the_canonical_recipe() -> None:
    assert TransferMethod.key == "transfer"
    assert TRANSFER_RECIPE["side"] > 0
    assert TRANSFER_RECIPE["blocks"] >= 1
    assert TransferMethod.applies_send_overlay is False
    assert TransferMethod.phase_carries_carrier is True


def test_validate_rejects_non_daheng_camera() -> None:
    method = TransferMethod("slm1", 1)
    error = method.validate({"cam_type": "sim", "shapes": ("square",)})
    assert error is not None and "daheng" in error


def test_shaping_controls_still_exports_what_the_page_uses() -> None:
    for name in ("TransferMethod", "render_bench_status", "snapshot_summary_line"):
        assert hasattr(shaping_controls, name)


# --------------------------------------------------------------------------- #
# 3. 参数区内容来自 TransferMethod（离线渲染，不经过 Streamlit 运行时）
# --------------------------------------------------------------------------- #
def test_transfer_method_renders_the_controls_the_page_shows() -> None:
    """参数区由适配器提供 —— 页面本身不新增任何专属控件。

    ``TransferMethod.render_inputs`` 需要一个 Streamlit 上下文，这里只锁
    契约面：``render_objective`` / ``render_budget`` / ``render_advanced``
    都在，且页面把三者串起来的顺序与 ``render_inputs`` 一致。
    """
    for name in ("render_objective", "render_budget", "render_advanced"):
        assert hasattr(TransferMethod, name)

    source = inspect.getsource(diff_opt_controller.render_parameters)
    order = [
        source.index(name)
        for name in ("render_inputs", "merge_bench_values", "hint_bench_calibration")
    ]
    assert order == sorted(order), "参数区顺序必须是 render_inputs → 并光路值 → 标定提示"

    run_source = inspect.getsource(diff_opt_controller.main)
    # 校验必须早于按钮，按钮必须早于 run。
    assert run_source.index("render_validation") < run_source.index(
        "render_run_controls"
    )
    assert run_source.index("render_run_controls") < run_source.index("method.run(")

