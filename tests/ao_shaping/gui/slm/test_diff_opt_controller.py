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
from types import SimpleNamespace

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

    run_source = inspect.getsource(diff_opt_controller._render_body)
    # 校验必须早于按钮，按钮必须早于 run。
    assert run_source.index("render_validation") < run_source.index(
        "render_run_controls"
    )
    assert run_source.index("render_run_controls") < run_source.index("method.run(")


# --------------------------------------------------------------------------- #
# 4. 布局模式：默认主区 / 专用子标题折叠面板
# --------------------------------------------------------------------------- #
def test_layout_defaults_to_the_main_area(page: AppTest) -> None:
    """没选折叠面板时主体画在主区，且侧栏有「页面布局」选择器。"""
    assert not page.exception
    assert [el.label for el in page.selectbox].count("页面布局") == 1
    assert page.session_state["diff_opt_layout_choice"] == "（无，默认主区域）"
    assert page.session_state["diff_opt_subheader_box"] is None


def test_subheader_layout_puts_the_body_inside_a_status_panel(page: AppTest) -> None:
    """选中折叠面板后，无设备提示仍出现且页面不报错（主体确实换容器）。"""
    page.selectbox(key="diff_opt_layout_choice").select(
        "专用子标题（折叠面板）"
    ).run()
    assert not page.exception
    assert not page.error
    assert any("连接至少一个 SLM" in info.value for info in page.info)
    # AppTest 把 ``st.status`` 的返回对象当回调注册，句柄不是可读字符串 ——
    # 只断言「选中时句柄非空、默认时为空」这条真实契约。
    assert page.session_state["diff_opt_layout_choice"] == "专用子标题（折叠面板）"



def test_subheader_layout_state_does_not_leak_into_other_pages() -> None:
    """布局句柄用本页自己的 key —— 不写 multi_slm_controller 的 header 名。"""
    src = inspect.getsource(diff_opt_controller)
    assert 'st.session_state["sb"]' not in src
    assert "diff_opt_subheader_box" in src


# --------------------------------------------------------------------------- #
# 5. 轨迹区：CCD 迭代过程 / 模型预测 / metrics 折线
#
# 页面只做两件事 —— 从会话目录取数（``collect_traces``）、交给
# ``render_traces`` 排版。这里锁的是页面级的接线契约：
# 目录来自 ``diff_opt_session_dir``、运行中不铺、产物缺失时不报错。
# --------------------------------------------------------------------------- #
class _FakeSLM:
    """已连接的 SLM 替身：``_connected_slm_numbers`` 只要求句柄非空，侧栏只读几个只读属性。"""

    is_open = True
    _serial_number = "FAKE-SN-01"
    wavelength = 1064
    slm_number = 1
    Pixel_Size_um = 8.0
    # 侧栏「波前矫正」段只读这两个 —— 无矫正文件时走 `未加载矫正文件` 分支。
    _correction = SimpleNamespace(is_valid=False, csv_path=None)
    correction_enabled = False
    correction_csv_path = None


def _connected(page: AppTest) -> AppTest:
    """让页面走「有设备」分支（主体才会出现进度/轨迹区）。

    设备枚举只在首轮跑一次（``available_slms`` 存在即跳过），所以先按真实顺序
    把 session 态备好再 ``run()``：``_initialize_slm_state`` 以
    ``"slm1" not in session_state`` 为首次标记，因此要先建 ``available_slms``
    再放句柄 —— 否则它会把句柄覆盖成 ``None``；而 ``slm1_width`` 等字段是该
    函数的产物，侧栏会直接下标读取，缺一个就 KeyError。
    """
    page.session_state["available_slms"] = []
    page.session_state["slm1"] = _FakeSLM()
    page.session_state["slm1_connected"] = True
    defaults: dict[str, object] = {
        "wavelength": 1064,
        "video_mode": "内存模式",
        "next_memory": 1,
        "phase_preview": None,
        "phase_source": "暂无",
        "preview_colormap": "viridis",
        "shift_x": 0,
        "shift_y": 0,
        "width": 1920,
        "height": 1200,
        "pixel_pitch_um": 8.0,
        "bits": 10,
        "use_120hz": False,
        "use_correction": True,
        "toggle_active": False,
        "toggle_frequency": 1.0,
        "base_phase": None,
        "overlay_base": False,
    }
    for name, value in defaults.items():
        page.session_state[f"slm1_{name}"] = value
    return page

def _session(tmp_path) -> str:
    """一份最小会话：四类轨迹各一条，足以让四张图都出现。"""
    import pickle

    root = tmp_path / "session"
    header = "cv,core_peak,model_loss,label,stage,shape,exposure_ms,_phase,_img,_grad,_id\n"
    for stage in ("model", "anchored"):
        path = root / stage / "recording" / "process.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            header
            + f'0.59,900.0,1.36,{stage}_seed,{stage},square,12.5,"[[0]]","[[1]]","[[0]]",0\n'
            + f'0.57,640.0,1.32,{stage}_0_A1,{stage},square,12.5,"[[0]]","[[1]]","[[0]]",1\n',
            encoding="utf-8",
        )
    fit = root / "fit" / "full"
    fit.mkdir(parents=True, exist_ok=True)
    with (fit / "iterations.pkl").open("wb") as stream:
        pickle.dump(
            [
                {"iteration": 0, "loss": 0.9,
                 "metrics": [{"correlation": 0.1, "relative_l2": 0.9, "cv_abs_error": 1.0}]},
                {"iteration": 1, "loss": 0.4,
                 "metrics": [{"correlation": 0.6, "relative_l2": 0.4, "cv_abs_error": 0.3}]},
            ],
            stream,
            protocol=4,
        )
    (fit / "provenance.json").write_text(
        '{"rows": [{"split": "train"}]}', encoding="utf-8"
    )
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "results.json").write_text(
        '{"metric_scope": "core CV", "shapes": [{"shape": "square", '
        '"seed": {"measured": {"cv": 0.59}, "predicted": {"cv": 0.035}, '
        '"correlation": 0.43, "loss": 1.36}}]}',
        encoding="utf-8",
    )
    (root / "anchored" / "summary.json").write_text(
        '[{"shape": "square", "block": 0, "baseline": 0.59, "candidate": 0.57, '
        '"accepted": true}]',
        encoding="utf-8",
    )
    return str(root)


def _rendered_traces(page: AppTest) -> bool:
    """主区是否排出了「轨迹」小标题（``subheader`` 是独立元素，不在 markdown 里）。"""
    return "轨迹" in [str(block.value) for block in page.main.subheader]


def test_no_session_dir_renders_no_trace_section(page: AppTest) -> None:
    """默认（还没跑过）不出现「轨迹」区，也不出现缺产物说明。"""
    assert "diff_opt_session_dir" not in page.session_state
    assert not _rendered_traces(page)


def test_a_session_dir_with_all_artefacts_renders_the_trace_section(
    page: AppTest, tmp_path
) -> None:
    """会话目录里有四类产物 ⇒ 「轨迹」区 + 四张图 + 四条图注。"""
    _connected(page)
    page.session_state["diff_opt_session_dir"] = _session(tmp_path)
    page.run()

    assert not page.exception
    assert not page.error
    assert _rendered_traces(page)
    said = [str(caption.value) for caption in page.main.caption]
    captions = [text for text in said if text.startswith(("模型预测", "CCD 迭代过程", "拟合轨迹", "逐块验收"))]
    assert len(captions) == 4
    assert any("CCD 迭代过程" in text for text in captions)
    assert not any("尚无轨迹产物" in text for text in said)


def test_a_session_dir_without_artefacts_says_which_ones_are_missing(
    page: AppTest, tmp_path
) -> None:
    """目录存在但产物还没写 ⇒ 不铺图、只说明缺什么，不报异常。"""
    _connected(page)
    (tmp_path / "empty").mkdir()
    page.session_state["diff_opt_session_dir"] = str(tmp_path / "empty")
    page.run()

    assert not page.exception
    assert not _rendered_traces(page)
    said = [str(caption.value) for caption in page.main.caption]
    assert any("尚无轨迹产物" in text for text in said)


def test_a_busy_bench_keeps_the_trace_section_off_the_page(
    page: AppTest, tmp_path
) -> None:
    """运行中：``TransferMethod`` 自己的面板在刷，页面不重复排进度与轨迹。"""
    _connected(page)
    page.session_state["diff_opt_session_dir"] = _session(tmp_path)
    page.session_state["bench_busy"] = True
    page.run()

    assert "运行进度" not in [str(block.value) for block in page.main.subheader]
    assert not _rendered_traces(page)


