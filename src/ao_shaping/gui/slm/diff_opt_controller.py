"""面向可微优化模型的专用 Streamlit 控制界面。

定位
----
本页是**迁移学习闭环整形**（``ao_shaping.tools.slm.transfer``，少参数物理预测
模型 + 当前光路重测几何 + 检查点迁移 + CCD 锚定 A–B–B–A 验收）的专用操作台：
单 SLM + 单相机，页面内容只有这条管线需要的控件。

与 ``multi_slm_controller.py`` 的关系
------------------------------------
**不重复实现**任何通用逻辑，全部 import 复用：

- 设备发现 / 连接 / 相机枚举 / bench 设置：``multi_slm_controller`` 的
  ``_initialize_slm_state`` / ``_refresh_device_list`` / ``render_slm_sidebar``。
- 参数渲染与运行编排：``shaping_controls.TransferMethod``。参数控件与
  ``render_closed_loop_shaping`` 里的**同名同 key**，运行走同一条
  ``ShapingMethod.run``（``shaping_session`` 装饰器 → 设备交还 → 下发接受相位），
  不在本页写第二套 orchestrator 或第二条 stage 序列。

合并掉的重复功能
----------------
- 侧栏「CCD参数设置」不再第二次调用 ``render_bench_settings()`` ——
  ``render_slm_sidebar()`` 内部已调用一次，重复调用会让 ``bench_cam_type``
  等 widget key 撞车直接崩页。
- 去掉多 SLM 页签、pattern 生成（Zernike / 湍流屏）、GS-refine 与 model-in-loop
  适配器选择、相位预览等与本管线无关的控件。
- 光路参数（相机 / 曝光 / 读出窗口 / 入射中心 / 几何）**只在侧栏**出现一次，
  主区通过 ``bench_settings.merge_bench_values`` 取回，不再铺第二份控件。

页面顺序（与 ``render_closed_loop_shaping`` 同一契约）
-----------------------------------------------------
    光路状态条 → 目标与预算 → 高级参数 → 校验区 → 运行 → 进度/结果

校验必须早于按钮，按钮才能 ``disabled``；运行中不重复排进度明细（
``TransferMethod`` 自己有一份实时面板）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ao_shaping.gui.slm import bench_settings
from ao_shaping.gui.slm.multi_slm_controller import (
    _initialize_slm_state,
    _refresh_device_list,
    render_slm_sidebar,
)
from ao_shaping.gui.slm.shaping_controls import (
    TransferMethod,
    render_bench_status,
    snapshot_summary_line,
)
from ao_shaping.gui.slm.transfer_progress import collect_progress, render_progress

#: 页面标题 / 侧栏品牌名。
PAGE_TITLE = "可微优化模型控制器"

#: 结果区在 ``st.session_state`` 里的键：最近一次运行写下的会话目录。
_SESSION_DIR_KEY = "diff_opt_session_dir"

#: 侧栏「页面布局」的折叠面板句柄名（本页自己的 key，不串到别的页面）。
_SUBHEADER_BOX_KEY = "diff_opt_subheader_box"

#: 侧栏「页面布局」选择器自己的 key。
_SUBHEADER_OPTION_KEY = "diff_opt_layout_choice"

#: 「默认主区域」选项的文案。
_PLAIN_LAYOUT = "（无，默认主区域）"


# --------------------------------------------------------------------------- #
# 渲染目标
# --------------------------------------------------------------------------- #
class _Layout:
    """页面渲染目标：普通模式 = 主区 ``st``；折叠面板模式 = ``st.status`` 内部。

    Streamlit 的渲染位置由调用点的**容器**决定，所以「在谁里面画」是一个显式
    参数而不是全局状态 —— 否则测试替身与其它页面会互相串味。

    本类只做两件事：``with layout:`` 进入该容器，以及把属性访问转给容器
    （``layout.columns`` / ``layout.button`` … 与 ``st`` 一致）。
    """

    __slots__ = ("_container", "_target")

    def __init__(self, target: Any, container: Any = None) -> None:
        self._target = target
        self._container = container

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __enter__(self) -> "_Layout":
        if self._container is not None:
            self._container.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._container is not None:
            self._container.__exit__(*exc)


def _layout() -> _Layout:
    """当前页面的渲染目标（普通主区，或侧栏选中的专用子标题面板）。"""
    box = st.session_state.get(_SUBHEADER_BOX_KEY)
    if box is not None:
        return _Layout(box)
    return _Layout(st)


# --------------------------------------------------------------------------- #
# 会话状态 / 连接判定
# --------------------------------------------------------------------------- #
def _ensure_state() -> None:
    """首次进入时枚举设备（与 ``multi_slm_controller.main`` 同一入口）。

    枚举会逐个探测 8 个 SLM 编号，**必须只做一次**：每轮 rerun 重扫等于每次点
    按钮都白等十几秒。``_initialize_slm_state`` 建出的状态就是「已扫过」的标记。
    """
    if "available_slms" not in st.session_state:
        _initialize_slm_state()
        _refresh_device_list()


def _connected_slm_numbers() -> list[int]:
    """已连接（有 handle 且状态为已连接）的 SLM 编号。"""
    return [
        n
        for n in (1, 2)
        if st.session_state.get(f"slm{n}_connected")
        and st.session_state.get(f"slm{n}") is not None
    ]


# --------------------------------------------------------------------------- #
# 主体各段
# --------------------------------------------------------------------------- #
def render_connection_hint() -> None:
    """没有任何已连接 SLM 时的页面主体。"""
    st.title(PAGE_TITLE)
    st.info("请在左侧连接至少一个 SLM")
    st.caption(
        "本页只需一台 SLM 与一台相机：光路参数在左侧「CCD参数设置」，"
        "标定按钮同在那里。"
    )


def render_status_header(slm_num: int, prefix: str) -> None:
    """标题 + 一句话说明 + 光路状态条（这条光路现在能不能信）。"""
    st.title(PAGE_TITLE)
    st.caption(
        f"SLM {slm_num} · 单相机 · 迁移学习闭环整形（物理预测模型）。与 CLI "
        "`python -m ao_shaping.tools.slm.transfer` 同一入口：先名义初始化 / "
        "迁移历史检查点，再用 CCD 锚定短块与 A–B–B–A 配对验收生成目标相位。"
    )
    status = render_bench_status(prefix, slm_num)
    if status:
        st.caption(status)
    manual = bench_settings.manual_fields(prefix)
    if manual:
        st.warning(
            "光路参数有手动覆盖，标定不会改动它们："
            + "；".join(bench_settings.field_hint(prefix, name) for name in manual)
        )


def render_parameters(method: TransferMethod, layout: _Layout) -> dict[str, Any]:
    """目标与预算 → 高级参数，并并入侧栏光路值（光路控件只在侧栏）。"""
    with layout:
        values = method.render_inputs()
    bench_settings.merge_bench_values(
        values, method.prefix, method.slm_num, method.key
    )
    method.hint_bench_calibration(values)
    return values


def render_validation(
    method: TransferMethod, values: dict[str, Any], layout: _Layout
) -> list[str]:
    """校验区必须在按钮之前，按钮才能 ``disabled``。"""
    errors: list[str] = []
    bench_error = bench_settings.validate_bench_inputs()
    if bench_error:
        errors.append(bench_error)
    method_error = method.validate(values)
    if method_error:
        errors.append(method_error)
    with layout:
        for error in errors:
            st.error(error)
    return errors


def render_run_controls(
    method: TransferMethod, errors: list[str], layout: _Layout
) -> tuple[bool, bool]:
    """调试开关 + 运行按钮。返回 ``(clicked, debug)``。"""
    with layout:
        debug = st.checkbox(
            "保存调试产物 (data/debug/…)",
            value=False,
            key=f"{method.prefix}_shp_debug",
            disabled=bool(errors),
            help="与 CLI --debug 同布局；迁移学习的证据目录不受它影响 —— "
            "该管线从第一阶段就写 journal/audit，没有「不保存」模式",
        )
        clicked = st.button(
            method.action,
            key=f"{method.prefix}_shp_run_{method.key}",
            type="primary",
            disabled=bool(errors) or bench_settings.bench_busy(),
            use_container_width=True,
        )
    return bool(clicked), bool(debug)


def render_session_progress() -> None:
    """最近一次运行的进度明细（读会话目录，不重算指标）。

    运行期间进度由 ``TransferMethod`` 自己的进度条 + 明细面板实时刷新；本函数
    只在**运行结束后**把同一条会话目录的最终状态再排一遍，所以运行中直接跳过，
    否则同一份明细会在页面上出现两次。
    """
    raw = st.session_state.get(_SESSION_DIR_KEY)
    if not raw or bench_settings.bench_busy():
        return
    st.subheader("运行进度")
    render_progress(collect_progress(Path(raw)), running=False)


# --------------------------------------------------------------------------- #
# 侧栏布局选择
# --------------------------------------------------------------------------- #
def _subheader_options() -> list[str]:
    """「页面布局」的选项：默认主区，或挂到专用子标题折叠面板。"""
    return [_PLAIN_LAYOUT, "专用子标题（折叠面板）"]


def render_subheader_selector() -> None:
    """侧栏「页面布局」：把主体挂进 ``st.status`` 折叠面板，或留在主区。

    实现与 ``multi_slm_controller`` 的专用子标题一致（``st.status`` +
    会话态句柄），但句柄存在**本页自己的 key** 上，不会串到别的页面。
    交互控件在折叠面板内照常可用。
    """
    options = _subheader_options()
    current = st.session_state.get(_SUBHEADER_OPTION_KEY, options[0])
    index = options.index(current) if current in options else 0
    choice = st.selectbox(
        "页面布局",
        options=options,
        index=index,
        key=_SUBHEADER_OPTION_KEY,
        help="折叠面板模式把主体收进一块可折叠的「专用子标题」容器；"
        "需要屏幕空间时用默认主区域",
    )
    if choice == _PLAIN_LAYOUT:
        st.session_state[_SUBHEADER_BOX_KEY] = None
        return
    st.session_state[_SUBHEADER_BOX_KEY] = st.status(
        f"{PAGE_TITLE}（专用子标题）", expanded=True
    )


# --------------------------------------------------------------------------- #
# 主体编排
# --------------------------------------------------------------------------- #
def _render_body(layout: _Layout) -> None:
    """主体内容：状态条 → 参数 → 校验 → 运行 → 进度。"""
    connected = _connected_slm_numbers()
    if not connected:
        with layout:
            render_connection_hint()
        return

    slm_num = connected[0]
    prefix = f"slm{slm_num}"

    with layout:
        render_status_header(slm_num, prefix)

    method = TransferMethod(prefix, slm_num)
    values = render_parameters(method, layout)
    errors = render_validation(method, values, layout)
    clicked, debug = render_run_controls(method, errors, layout)

    if clicked:
        # 计划目录要在 ``run`` 之前拿到：阶段中途失败时 ``run`` 返回 ``None``，
        # 事后无从得知证据写在哪，而失败阶段恰恰最需要看产物。
        planned = method.build_params(values).out
        st.session_state[_SESSION_DIR_KEY] = str(planned)
        method.run(values, debug=debug)
        return

    with layout:
        render_session_progress()
        if not bench_settings.effective_state(prefix):
            st.caption("尚无运行结果。")
        st.caption(snapshot_summary_line(prefix))


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ensure_state()

    with st.sidebar:
        # 「CCD参数设置」已在 render_slm_sidebar() 内部渲染一次 —— 再调
        # render_bench_settings() 会重复声明 bench_cam_type 等 key 让页面崩掉。
        render_slm_sidebar()
        render_subheader_selector()

    _render_body(_layout())


if __name__ == "__main__":
    main()
