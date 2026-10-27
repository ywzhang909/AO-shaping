"""面向可微优化模型的专用 Streamlit 控制界面。

与 ``multi_slm_controller.py`` 的关系
------------------------------------
本模块**不重复实现**设备发现、SLM 打开/关闭、相机枚举、bench 设置等通用逻辑，
而是直接 import 它们（``render_slm_sidebar``、``bench_settings``、``phase_overlay``）。
本页只做一件事：把 ``shaping_controls.TransferMethod`` 的参数渲染与运行编排
提升为**页面主内容**，去掉多 SLM 切换、pattern 生成、GS/模型-in-loop 等与
可微优化无关的控件，形成一个**单 SLM + 单相机**的专用界面。

合并的重复功能
--------------
- 设备发现与连接：复用 ``multi_slm_controller._initialize_slm_state`` /
  ``_refresh_device_list`` / ``render_slm_sidebar``。
- bench 设置：复用 ``bench_settings.render_bench_settings``。
- 相位显示：复用 ``pattern_controls.refresh_phase_preview`` 与 ``display_phase``。
- 传输学习参数渲染：直接委托 ``TransferMethod.render_objective / render_budget /
  render_advanced``，保证与多 SLM 控制器里的同名控件**逐字一致**（同一份
  ``TRANSFER_RECIPE`` 默认值）。
- 运行编排：委托 ``TransferMethod.build_params / prepare_config / execute``，
  不在本页重写 stage 序列（避免第二套 orchestrator 漂移）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ao_shaping.gui.slm.multi_slm_controller import (
    _initialize_slm_state,
    _refresh_device_list,
    render_slm_sidebar,
)
from ao_shaping.gui.slm.bench_settings import render_bench_settings
from ao_shaping.gui.slm.shaping_controls import (
    TransferMethod,
)
from ao_shaping.gui.slm.transfer_progress import collect_progress, render_progress


def _ensure_state() -> None:
    if "available_slms" not in st.session_state:
        _initialize_slm_state()
        _refresh_device_list()


def main() -> None:
    st.set_page_config(
        page_title="可微优化模型控制器",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ensure_state()

    with st.sidebar:
        render_slm_sidebar()
        st.divider()
        render_bench_settings()

    connected = [
        n
        for n in (1, 2)
        if st.session_state.get(f"slm{n}_connected")
        and st.session_state.get(f"slm{n}") is not None
    ]
    if not connected:
        st.title("可微优化模型控制器")
        st.info("请在左侧连接至少一个 SLM")
        return

    slm_num = connected[0]
    st.title("可微优化模型控制器")
    st.caption(
        "单 SLM + 单相机的专用界面：少参数物理预测模型 + 当前光路重测几何 + "
        "检查点迁移 + CCD 锚定 A–B–B–A 配对验收。与 CLI "
        "`python -m ao_shaping.tools.slm.transfer` 同一入口。"
    )

    # A single TransferMethod instance owns all widgets + run state.
    method = TransferMethod(prefix=f"diff_opt_{slm_num}", slm_num=slm_num)

    # --- parameters ------------------------------------------------------
    values: dict[str, Any] = {}
    with st.expander("目标与预算", expanded=True):
        method.render_objective(values)
        method.render_budget(values)
    with st.expander("迁移来源与验收口径", expanded=False):
        method.render_advanced(values)

    # --- validation ------------------------------------------------------
    err = method.validate(values)
    if err:
        st.error(err)

    # --- run -------------------------------------------------------------
    col_run, col_status = st.columns([1, 3])
    with col_run:
        run_clicked = st.button(
            method.action,
            type="primary",
            disabled=bool(err),
            key=method.wkey("run"),
        )
    with col_status:
        if st.session_state.get(method.wkey("running")):
            st.info("运行中…页面会阻塞直到设备阶段结束")
        else:
            st.caption("运行结束后下方显示进度与结果")

    if run_clicked and not err:
        st.session_state[method.wkey("running")] = True
        try:
            params = method.build_params(values)
            config = method.make_config(params, debug=False)
            method.prepare_config(config, values)
            outcome = method.execute(config)
            st.session_state[method.wkey("outcome")] = outcome
            st.session_state[method.wkey("session_dir")] = str(params.out)
            st.success("运行完成")
        except Exception as exc:
            st.exception(exc)
        finally:
            st.session_state[method.wkey("running")] = False

    # --- progress + result -----------------------------------------------
    session_dir = st.session_state.get(method.wkey("session_dir"))
    if session_dir:
        st.subheader("运行进度")
        progress = collect_progress(Path(session_dir))
        render_progress(progress, running=st.session_state.get(method.wkey("running"), False))

    outcome = st.session_state.get(method.wkey("outcome"))
    if outcome is not None:
        st.subheader("运行结果")
        st.json(outcome if not hasattr(outcome, "__dict__") else outcome.__dict__)


if __name__ == "__main__":
    main()
