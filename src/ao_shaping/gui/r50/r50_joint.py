"""R50 联合控制模块 (JC) — MicroDM 矩阵读取 / 连接 / 继电器 / 下发 / 编辑 / 可视化。

被 ``r50_tabs.render_tab_all_control`` 与
``r50_sidebar._sidebar_joint_connection`` 复用。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from ao_shaping.drivers.dm.MicroDM import MicroDM
from ao_shaping.gui.r50.r50_channel_select import (
    DM_NUM_ACTUATORS,
    GRID_SIZE,
    HW_VOLTAGE_MAX,
    HW_VOLTAGE_MIN,
    P,
    SINGLE_CHANNELS,
    jc_build_ip_index,
    jc_build_wiring_index,
    jc_matrix_to_flat,
    load_csv,
)
from ao_shaping.gui.r50.r50_connection import (
    SimulatedMicroDM,
    ping_reachable,
)
from ao_shaping.gui.r50.r50_debug import _debug_add_op


# =============================================================================
# 联合控制 (JC): 矩阵读取 / 连接 / 继电器 / 下发
# =============================================================================

def _jc_read_matrix_from_dm(
    dm: Any,
    pos_to_hw: dict[int, tuple[int, int]],
    ip_to_ctrl_idx: dict[int, int],
) -> np.ndarray:
    """从 MicroDM 读取当前电压并构建 36×36 矩阵。"""
    flat = dm.get_actuator_positions()
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    for physical_pos, (ip_suffix, payload_pos) in pos_to_hw.items():
        row = (physical_pos - 1) // GRID_SIZE
        col = (physical_pos - 1) % GRID_SIZE
        ctrl_idx = ip_to_ctrl_idx.get(ip_suffix)
        if ctrl_idx is not None:
            flat_idx = ctrl_idx * SINGLE_CHANNELS + (payload_pos - 1)
            if flat_idx < len(flat):
                matrix[row, col] = flat[flat_idx]
    return matrix


def _jc_connect() -> None:
    """连接 MicroDM (所有控制器, 支持仿真模式)。"""
    jc = f"{P}_jc"
    try:
        simulate = st.session_state.get(f"{P}_jc_simulate", False)
        csv_df = load_csv()
        if csv_df.empty:
            st.session_state[f"{jc}_connection_error"] = "1300-5-enriched.csv 加载失败"
            st.session_state[f"{jc}_connected"] = False
            return
        ip_suffixes = sorted(int(ip) for ip in csv_df["IP组"].unique())
        if simulate:
            dm: Any = SimulatedMicroDM(ips=ip_suffixes)
            dm.open()
            feedback_prefix = "🟡 [仿真] "
        else:
            dm = MicroDM(use_wiring_map=True)
            dm.open()
            feedback_prefix = ""
        st.session_state[f"{jc}_dm"] = dm
        st.session_state[f"{jc}_connected"] = True
        st.session_state[f"{jc}_relay_on"] = False
        st.session_state[f"{jc}_connection_error"] = ""
        st.session_state[f"{jc}_controller_count"] = len(dm._controllers)
        pos_to_hw = jc_build_wiring_index(csv_df)
        ip_to_ctrl_idx = jc_build_ip_index(csv_df)
        st.session_state[f"{jc}_pos_to_hw"] = pos_to_hw
        st.session_state[f"{jc}_ip_to_controller_idx"] = ip_to_ctrl_idx
        st.session_state[f"{jc}_sorted_ips"] = [f"192.168.0.{s}" for s in ip_suffixes]
        st.session_state[f"{jc}_dm_num"] = getattr(
            dm, "DM_Num", len(ip_suffixes) * SINGLE_CHANNELS
        )
        st.session_state[f"{jc}_matrix_init"] = True
        if not simulate:
            matrix = _jc_read_matrix_from_dm(dm, pos_to_hw, ip_to_ctrl_idx)
        else:
            matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        st.session_state[f"{jc}_matrix"] = matrix
        st.session_state[f"{jc}_applied_matrix"] = matrix.copy()
        st.session_state[f"{jc}_current_flat"] = matrix.flatten().copy()
        n_ctrl = st.session_state[f"{jc}_controller_count"]
        st.session_state[f"{jc}_feedback"] = f"{feedback_prefix}已连接 MicroDM: {n_ctrl} 个控制器"
        st.session_state[f"{jc}_feedback_type"] = "success"
        _debug_add_op("connect", f"joint ({n_ctrl} controllers)", "all")
        logger.info(f"MicroDM connected: {n_ctrl} controllers")
    except Exception as e:
        st.session_state[f"{jc}_connection_error"] = f"连接失败: {e}"
        st.session_state[f"{jc}_connected"] = False
        st.session_state[f"{jc}_dm"] = None
        st.session_state[f"{jc}_feedback"] = f"连接失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM connect failed: {e}")


def _jc_disconnect() -> None:
    """断开 MicroDM 连接 (先下电)。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is not None:
        try:
            if st.session_state.get(f"{jc}_relay_on", False):
                dm.set_relay_state(False)
            dm.close()
        except Exception as e:
            logger.warning(f"MicroDM disconnect warning: {e}")
    st.session_state[f"{jc}_dm"] = None
    st.session_state[f"{jc}_connected"] = False
    st.session_state[f"{jc}_relay_on"] = False
    st.session_state[f"{jc}_connection_error"] = ""
    st.session_state[f"{jc}_feedback"] = "已断开连接 (已先下电)"
    st.session_state[f"{jc}_feedback_type"] = "info"
    _debug_add_op("disconnect", "joint", "all")
    logger.info("MicroDM disconnected")


def _jc_set_relay(on: bool) -> None:
    """所有控制器继电器上下电。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is None:
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        dm.set_relay_state(on)
        st.session_state[f"{jc}_relay_on"] = on
        if on:
            st.session_state[f"{jc}_feedback"] = "✅ 所有控制器继电器已上电 (输出接通)"
            st.session_state[f"{jc}_feedback_type"] = "success"
            _debug_add_op("relay_on", "joint", "all")
        else:
            st.session_state[f"{jc}_feedback"] = "⏻ 所有控制器继电器已下电 (输出断开)"
            st.session_state[f"{jc}_feedback_type"] = "info"
            _debug_add_op("relay_off", "joint", "all")
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"继电器操作失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM relay failed: {e}")


def _jc_apply_matrix() -> None:
    """将当前 36×36 矩阵电压下发到所有控制器。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is None:
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    if not st.session_state.get(f"{jc}_relay_on", False):
        st.session_state[f"{jc}_feedback"] = "⚠️ 请先继电器上电后再下发电压"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        matrix = st.session_state[f"{jc}_matrix"]
        pos_to_hw = st.session_state[f"{jc}_pos_to_hw"]
        ip_to_ctrl = st.session_state[f"{jc}_ip_to_controller_idx"]
        dm_num = st.session_state[f"{jc}_dm_num"]
        flat = jc_matrix_to_flat(matrix, pos_to_hw, ip_to_ctrl, dm_num)
        dm.send_voltages(flat)
        st.session_state[f"{jc}_current_flat"] = flat.copy()
        st.session_state[f"{jc}_applied_matrix"] = matrix.copy()
        non_zero = np.count_nonzero(matrix)
        st.session_state[f"{jc}_feedback"] = (
            f"✅ 已下发 36×36 矩阵电压 (非零通道: {non_zero}/{DM_NUM_ACTUATORS})"
        )
        st.session_state[f"{jc}_feedback_type"] = "success"
        _debug_add_op("set_voltage", f"matrix {non_zero} non-zero channels", "all")
        logger.info(f"MicroDM voltage applied: {non_zero} non-zero channels")
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"电压下发失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM apply failed: {e}")


def _jc_reset_matrix() -> None:
    """将矩阵清零 (仅编辑缓冲区, 不下发到硬件)。"""
    jc = f"{P}_jc"
    st.session_state[f"{jc}_matrix"] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    st.session_state[f"{jc}_feedback"] = "矩阵已清零 (仅编辑缓冲区)"
    st.session_state[f"{jc}_feedback_type"] = "info"


def _jc_reset_to_applied() -> None:
    """将编辑缓冲区恢复为上次下发的矩阵 (edit-only, 不下发)。"""
    jc = f"{P}_jc"
    applied = st.session_state.get(f"{jc}_applied_matrix")
    if applied is None:
        st.session_state[f"{jc}_matrix"] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        st.session_state[f"{jc}_feedback"] = "无已下发矩阵，已重置为全零"
    else:
        st.session_state[f"{jc}_matrix"] = applied.copy()
        st.session_state[f"{jc}_feedback"] = "编辑缓冲区已恢复为上次下发状态"
    st.session_state[f"{jc}_feedback_type"] = "info"


def _jc_refresh_from_hardware() -> None:
    """从硬件读取当前电压并刷新矩阵显示。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is None:
        return
    if st.session_state.get(f"{P}_jc_simulate", False):
        st.session_state[f"{jc}_feedback"] = "仿真模式: 矩阵保持当前值"
        st.session_state[f"{jc}_feedback_type"] = "info"
        return
    try:
        pos_to_hw = st.session_state[f"{jc}_pos_to_hw"]
        ip_to_ctrl = st.session_state[f"{jc}_ip_to_controller_idx"]
        matrix = _jc_read_matrix_from_dm(dm, pos_to_hw, ip_to_ctrl)
        st.session_state[f"{jc}_matrix"] = matrix
        st.session_state[f"{jc}_applied_matrix"] = matrix.copy()
        st.session_state[f"{jc}_current_flat"] = matrix.flatten().copy()
        st.session_state[f"{jc}_feedback"] = "已从硬件刷新电压矩阵"
        st.session_state[f"{jc}_feedback_type"] = "info"
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"刷新失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"MicroDM refresh failed: {e}")


# =============================================================================
# 联合控制 (JC): 批量上下电 (Ping 测试)
# =============================================================================

def _jc_batch_power_on() -> None:
    """批量上电: 先 ping 测试所有控制器, 再继电器上电。"""
    jc = f"{P}_jc"
    dm = st.session_state.get(f"{jc}_dm")
    if dm is None:
        st.session_state[f"{jc}_feedback"] = "设备未连接"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    sorted_ips = st.session_state.get(f"{jc}_sorted_ips", [])
    if not sorted_ips:
        st.session_state[f"{jc}_feedback"] = "无控制器 IP 信息"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    simulate = st.session_state.get(f"{P}_jc_simulate", False)
    reachable: list[str] = []
    unreachable: list[str] = []
    for ip in sorted_ips:
        if simulate or ping_reachable(ip, timeout=1.0):
            reachable.append(ip)
        else:
            unreachable.append(ip)
    if not reachable:
        st.session_state[f"{jc}_feedback"] = f"❌ 所有控制器均不可达: {', '.join(unreachable)}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        return
    try:
        dm.set_relay_state(True)
        st.session_state[f"{jc}_relay_on"] = True
        if not unreachable:
            st.session_state[f"{jc}_feedback"] = (
                f"✅ 批量上电成功 ({len(reachable)} 个控制器全部可达)"
            )
            st.session_state[f"{jc}_feedback_type"] = "success"
        else:
            st.session_state[f"{jc}_feedback"] = (
                f"⚠️ 部分上电: {len(reachable)} 可达并已上电, "
                f"{len(unreachable)} 不可达 ({', '.join(unreachable)})"
            )
            st.session_state[f"{jc}_feedback_type"] = "warning"
    except Exception as e:
        st.session_state[f"{jc}_feedback"] = f"批量上电失败: {e}"
        st.session_state[f"{jc}_feedback_type"] = "error"
        logger.exception(f"Batch power on failed: {e}")


def _jc_batch_power_off() -> None:
    """批量下电: 所有控制器继电器下电。"""
    _jc_set_relay(False)


# =============================================================================
# 联合控制 (JC): 矩阵编辑
# =============================================================================

def _jc_set_cell(row: int, col: int, voltage: float) -> None:
    """设置矩阵中单个单元电压。"""
    jc = f"{P}_jc"
    matrix = st.session_state[f"{jc}_matrix"].copy()
    matrix[row, col] = voltage
    st.session_state[f"{jc}_matrix"] = matrix


def _jc_fill_row(row: int, voltage: float) -> None:
    """填充整行。"""
    jc = f"{P}_jc"
    matrix = st.session_state[f"{jc}_matrix"].copy()
    matrix[row, :] = voltage
    st.session_state[f"{jc}_matrix"] = matrix


def _jc_fill_col(col: int, voltage: float) -> None:
    """填充整列。"""
    jc = f"{P}_jc"
    matrix = st.session_state[f"{jc}_matrix"].copy()
    matrix[:, col] = voltage
    st.session_state[f"{jc}_matrix"] = matrix


def _jc_fill_rect(x1: int, y1: int, x2: int, y2: int, voltage: float) -> None:
    """填充矩形区域。"""
    jc = f"{P}_jc"
    matrix = st.session_state[f"{jc}_matrix"].copy()
    r1, r2 = min(y1, y2), max(y1, y2)
    c1, c2 = min(x1, x2), max(x1, x2)
    matrix[r1:r2 + 1, c1:c2 + 1] = voltage
    st.session_state[f"{jc}_matrix"] = matrix


def _jc_fill_all(voltage: float) -> None:
    """填充整个矩阵。"""
    jc = f"{P}_jc"
    matrix = np.full((GRID_SIZE, GRID_SIZE), voltage, dtype=np.float64)
    st.session_state[f"{jc}_matrix"] = matrix


# =============================================================================
# 联合控制 (JC): 可视化 (Streamlit 原生控件)
# =============================================================================

def _jc_colormap_image(matrix: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """将电压矩阵转换为彩色图像 (numpy, 无 matplotlib 依赖)。

    coolwarm 风格: 蓝色 (vmin) → 白色 (0) → 红色 (vmax)。
    """
    if vmax <= vmin:
        vmax = vmin + 1.0
    normalized = np.clip((matrix - vmin) / (vmax - vmin), 0, 1)
    h, w = matrix.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    mask_low = normalized <= 0.5
    mask_high = normalized > 0.5
    img[mask_low, 0] = normalized[mask_low] * 2.0
    img[mask_low, 1] = normalized[mask_low] * 2.0
    img[mask_low, 2] = 1.0
    img[mask_high, 0] = 1.0
    img[mask_high, 1] = 2.0 - normalized[mask_high] * 2.0
    img[mask_high, 2] = 2.0 - normalized[mask_high] * 2.0
    return img


def _jc_render_matrix_image(matrix: np.ndarray) -> None:
    """Streamlit ``st.image`` 显示彩色热力图。"""
    vmin = st.session_state.get(f"{P}_vmin", HW_VOLTAGE_MIN)
    vmax = st.session_state.get(f"{P}_vmax", HW_VOLTAGE_MAX)
    img = _jc_colormap_image(matrix, vmin, vmax)
    st.image(img, caption="36×36 电压分布 (蓝色低 · 红色高)", width='stretch')


def _jc_render_matrix_dataframe(matrix: np.ndarray) -> None:
    """分块显示带颜色的 36×36 数值矩阵 (6 块 × 6 列)。"""
    vmin = st.session_state.get(f"{P}_vmin", HW_VOLTAGE_MIN)
    vmax = st.session_state.get(f"{P}_vmax", HW_VOLTAGE_MAX)
    blocks = 6
    cols_per_block = 6
    for block_idx in range(blocks):
        start_col = block_idx * cols_per_block
        end_col = min(start_col + cols_per_block, GRID_SIZE)
        col_labels = [str(c + 1) for c in range(start_col, end_col)]
        df_block = pd.DataFrame(
            matrix[:, start_col:end_col],
            index=[f"行{r + 1}" for r in range(GRID_SIZE)],
            columns=col_labels,
        )
        with st.expander(f"📍 第 {start_col + 1}–{end_col} 列", expanded=(block_idx == 0)):
            col_config = {}
            for i, c in enumerate(col_labels):
                col_config[c] = st.column_config.NumberColumn(
                    label=c,
                    min_value=vmin,
                    max_value=vmax,
                    format="%.1f",
                )
            st.dataframe(
                df_block,
                column_config=col_config,
                height=min(36 * 35 + 40, 800),
                width='stretch',
            )


def _jc_render_profile(matrix: np.ndarray) -> None:
    """行/列均值剖面图表。"""
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    with st.container(border=True):
        st.markdown("##### 电压剖面")
        tab_r, tab_c = st.tabs(["📊 行均值", "📊 列均值"])
        with tab_r:
            df_row = pd.DataFrame(
                {"行号": list(range(1, GRID_SIZE + 1)), "均值 (V)": row_means}
            ).set_index("行号")
            st.bar_chart(df_row, height=200, width='stretch')
        with tab_c:
            df_col = pd.DataFrame(
                {"列号": list(range(1, GRID_SIZE + 1)), "均值 (V)": col_means}
            ).set_index("列号")
            st.bar_chart(df_col, height=200, width='stretch')


def _jc_render_stats(matrix: np.ndarray) -> None:
    """显示矩阵统计指标。"""
    vals = matrix.flatten()
    non_zero_count = np.count_nonzero(matrix)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("最小值", f"{np.min(vals):.1f} V")
    with col2:
        st.metric("最大值", f"{np.max(vals):.1f} V")
    with col3:
        st.metric("均值", f"{np.mean(vals):.1f} V")
    with col4:
        st.metric("标准差", f"{np.std(vals):.1f} V")
    with col5:
        st.metric("非零通道", f"{non_zero_count}/{DM_NUM_ACTUATORS}")


def _jc_render_styled_matrix(
    matrix: np.ndarray,
    applied: np.ndarray | None,
    vmin: float,
    vmax: float,
    chunk_start: int,
    chunk_end: int,
) -> None:
    """Render columns [chunk_start, chunk_end) with blue gradient + bold unsent cells."""
    n_cols = chunk_end - chunk_start
    if n_cols <= 0:
        return

    def _bg_gradient(val: float) -> str:
        if vmax <= vmin:
            ratio = 0.0
        else:
            ratio = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
        r = int(255 - (255 - 30) * ratio)
        g = int(255 - (255 - 100) * ratio)
        b = int(255 - (255 - 220) * ratio)
        return f"background-color: rgb({r},{g},{b})"

    chunk_data = matrix[:, chunk_start:chunk_end]
    df = pd.DataFrame(
        chunk_data,
        index=[str(r + 1) for r in range(GRID_SIZE)],
        columns=[str(c + 1) for c in range(chunk_start, chunk_end)],
    )

    styler = df.style.applymap(_bg_gradient)
    if applied is not None:
        applied_chunk = applied[:, chunk_start:chunk_end]

        def _row_bold(row: pd.Series) -> list[str]:
            row_idx = int(row.name) - 1
            styles = []
            for c_i, val in enumerate(row):
                col_idx = chunk_start + c_i
                if row_idx < applied_chunk.shape[0] and col_idx < applied_chunk.shape[1]:
                    if abs(float(val) - float(applied_chunk[row_idx, col_idx])) > 1e-9:
                        styles.append("font-weight: bold")
                    else:
                        styles.append("")
                else:
                    styles.append("")
            return styles

        styler = styler.apply(_row_bold, axis=1)

    styler = styler.format("{:.1f}")
    st.dataframe(styler, height=min(GRID_SIZE * 35 + 40, 800), width='stretch')
