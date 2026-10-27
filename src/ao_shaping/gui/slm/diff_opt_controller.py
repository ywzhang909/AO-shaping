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
- 侧栏 「CCD参数设置」不再第二次调用 ``render_bench_settings()`` ——
  ``render_slm_sidebar()`` 内部已调用一次，重复调用会让 ``bench_cam_type``
  等 widget key 撞车直接崩页。
- 去掉多 SLM 页签、pattern 生成（Zernike / 湍流屏）、GS-refine 与 model-in-loop
  适配器选择、相位预览等与本管线无关的控件。
- 光路参数（相机 / 曝光 / 读出窗口 / 入射中心 / 几何）**只在侧栏**出现一次，
  主区通过 ``bench_settings.merge_bench_values`` 取回，不再铺第二份控件。

页面顺序（与 ``render_closed_loop_shaping`` 同一契约）
-----------------------------------------------------
    光路状态条 → 目标与预算 → 高级参数 → 校验区 → 运行 → 进度/结果
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


def _ensure_state() -> None:
    """首次进入时枚举设备（与 ``multi_slm_controller.main`` 同一入口）。"""
    if "available_slms" not in st.session_state:
        _initialize_slm_state()
        _refresh_device_list()


def _connected_slm_numbers() -> list[int]:
    return [
        n
        for n in (1, 2)
        if st.session_state.get(f"slm{n}_connected")
        and st.session_state.get(f"slm{n}") is not None
    ]


def render_connection_hint() -> None:
    """没有任何已连接 SLM 时的页面主体。"""
    st.title(PAGE_TITLE)
    st.info("请在左侧连接至少一个 SLM")
    st.caption(
        "本页只需一台 SLM 与一台相机：光路参数在左侧「CCD参数设置」，"
        "标定按钮同在那里。"
    )


def render_header(slm_num: int) -> None:
    st.title(PAGE_TITLE)
    st.caption(
        f"SLM {slm_num} · 单相机 · 迁移学习闭环整形（物理预测模型）。与 CLI "
        "`python -m ao_shaping.tools.slm.transfer` 同一入口：先名义初始化 / "
        "迁移历史检查点，再用 CCD 锚定短块与 A–B–B–A 配对验收生成目标相位。"
    )


def render_path_status(prefix: str, slm_num: int) -> None:
    """光路状态条：这条光路现在能不能信（借用闭环整形的同一实现）。"""
    status = render_bench_status(prefix, slm_num)
    if status:
        st.caption(status)
    manual = bench_settings.manual_fields(prefix)
    if manual:
        st.warning(
            "光路参数有手动覆盖，标定不会改动它们："
            + "；".join(
                bench_settings.field_hint(prefix, name) for name in manual
            )
        )


def render_parameters(method: TransferMethod) -> dict[str, Any]:
    """目标与预算 → 高级参数，并并入侧栏光路值（唯一一份光路控件在侧栏）。"""
    values = method.render_inputs()
    bench_settings.merge_bench_values(values, method.prefix, method.slm_num, method.key)
    method.hint_bench_calibration(values)
    return values


def render_validation(
    method: TransferMethod, values: dict[str, Any]
) -> list[str]:
    """校验区必须在按钮之前，按钮才能 ``disabled``。"""
    errors: list[str] = []
    bench_error = bench_settings.validate_bench_inputs()
    if bench_error:
        errors.append(bench_error)
    method_error = method.validate(values)
    if method_error:
        errors.append(method_error)
    for error in errors:
        st.error(error)
    return errors


def render_run_controls(method: TransferMethod, errors: list[str]) -> tuple[bool, bool]:
    """调试开关 + 运行按钮。返回 ``(clicked, debug)``。"""
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
    """最近一次运行的进度明细（读会话目录，不重算指标）。"""
    raw = st.session_state.get(_SESSION_DIR_KEY)
    if not raw:
        return
    st.subheader("运行进度")
    render_progress(
        collect_progress(Path(raw)),
        running=bench_settings.bench_busy(),
    )


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
        # render_bench_settings() 会重复声明 bench_cam_type 等 key 并让页面崩掉。
        render_slm_sidebar()

    connected = _connected_slm_numbers()
    if not connected:
        render_connection_hint()
        return

    slm_num = connected[0]
    prefix = f"slm{slm_num}"
    render_header(slm_num)
    render_path_status(prefix, slm_num)

    method = TransferMethod(prefix, slm_num)
    values = render_parameters(method)
    errors = render_validation(method, values)
    clicked, debug = render_run_controls(method, errors)

    if clicked:
        method.run(values, debug=debug)
        # ``run`` 把接受相位写进 ``data/debug/slm_transfer/<stamp>``；目录由
        # ``prepare_config`` 前的 ``build_params`` 决定，这里只记录最近一次。
        planned = getattr(method, "_session_dir", None)
        if planned is not None:
            st.session_state[_SESSION_DIR_KEY] = str(planned)
        return

    render_session_progress()
    if not bench_settings.effective_state(prefix):
        st.caption("尚无运行结果。")
    st.caption(snapshot_summary_line(prefix))


if __name__ == "__main__":
    main()
