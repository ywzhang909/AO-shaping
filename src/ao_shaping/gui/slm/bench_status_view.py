"""光路与曝光状态条: 一句话说清「这条光路现在是什么状态」。

为什么要单独成模块
------------------
这一族只做**读数呈现**: 它把 ``bench_settings`` 里的标定状态、入射中心、相关度、
几何文件与曝光探测结果拼成给操作员看的一行字, 不参与优化、不改任何值。它与
``shaping_controls`` 的页面编排职责无关, 却长期占着那个文件的行数 —— 主文件已到
模块规模棘轮线 (``scripts/check_module_size.py``) 的基线, 因此整体搬到这里,
文案与判定逐字未改。
"""

from __future__ import annotations

import streamlit as st

from ao_shaping.gui.slm import bench_settings

__all__ = [
    "render_bench_status",
    "render_exposure_status",
]

def render_bench_status(prefix: str, slm_num: int) -> str | None:
    """状态条：一句话说清「这条光路现在是什么状态」。

    Three cases, and each one answers a question the operator actually has:
    not calibrated at all (the sidebar 标定 button has never run), calibrated with
    some fields hand-overridden, or calibrated and trusted.
    """
    detected = bench_settings.detected_state(prefix)
    manual = bench_settings.manual_fields(prefix)
    # One predicate, one implementation: ``is_calibrated`` also drives the
    # sidebar expander title, so the status bar cannot disagree with it.
    calibrated = bench_settings.is_calibrated(prefix)
    if not calibrated:
        return (
            f"光路 SLM {slm_num}：○ 未标定 —— {bench_settings.uncalibrated_note(prefix)}；"
            f"标定按钮在左侧「CCD参数设置」→「SLM {slm_num} · 光路几何与标定」"
        )
    parts = [f"光路 SLM {slm_num}：● 已标定 {detected.get('calibrated_at') or '-'}"]
    parts.append(
        f"入射中心 {bench_settings.bench_pupil_center(prefix, '未定位')} 探测值"
    )
    if detected.get("geometry_correlation") is not None:
        parts.append(f"相关度 {float(detected['geometry_correlation']):.4f}")
    parts.append(f"曝光 {detected.get('exposure_ms', '?')} ms")
    if detected.get("geometry_file"):
        parts.append(f"几何 {detected['geometry_file']}")
    line = " · ".join(parts)
    if manual:
        line += f" · {len(manual)} 项手动覆盖（{'、'.join(manual)}）"
    return line


def render_exposure_status(prefix: str) -> None:
    """L2 状态栏：会话曝光 + 「重测于 HH:MM:SS」+ 过期黄点 + 「重测」按钮。

    曝光是**会话级**量（单日有效，实测同一光路 1.5 h 后光强漂 ×1.35），所以实测
    记录超过 :data:`bench_settings.EXPOSURE_STALE_S` 时只点一个黄点。**绝不自动
    重测** —— 自动换掉操作员已写进实验记录的参数，比让他看见黄点自己点一下更糟；
    本函数也因此不排任何定时 rerun。值取控件**此刻**的值（L3），时刻取最近一次
    实测记录：手填之后两者不一致是真实状态，屏幕要能同时说清这两件事。
    """
    state = bench_settings.exposure_state(prefix)
    current = bench_settings.current_exposure_ms()
    if state["probed_at_text"]:
        line = f"会话曝光 {current:.3f} ms · 重测于 {state['probed_at_text']}"
    else:
        line = f"会话曝光 {current:.3f} ms · 无实测记录（手填值）"
    if state["stale"]:
        line += f" · 🟡 实测已过 {float(state['age_s']) / 3600:.1f} h，建议重测"
    # L1（光路指纹核对）的判定行就挂在曝光这一行下面：由 optical_session /
    # 指纹核对那条线给出结论后补一格，这里不预先造一个假值。
    text_col, button_col = st.columns([4, 1])
    with text_col:
        st.caption(line)
    with button_col:
        cam_type, cam_id = bench_settings.bench_camera()
        if st.button(
            "重测",
            key=f"{prefix}_retest_exposure",
            type="secondary",
            disabled=bench_settings.bench_busy() or cam_type != "daheng",
            use_container_width=True,
            help="按当前光路实测会话曝光（倍增阶梯找饱和上限 → 取一半）。"
            "只写会话曝光与取样时刻，不改其它参数",
        ):
            try:
                probed_ms, evidence = bench_settings.probe_session_exposure_ms(
                    cam_type, cam_id
                )
            except bench_settings.ExposureProbeFailure as exc:
                st.error(str(exc))
            else:
                bench_settings.record_exposure_probe("", probed_ms, evidence)
                bench_settings.queue_pending("", {"bench_exposure_ms": float(probed_ms)})
                st.rerun()
