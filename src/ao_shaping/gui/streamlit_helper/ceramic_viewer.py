"""
1300陶瓷单元查看工具 (Streamlit)

功能:
1. Tab 1 - 36×36 网格浏览: 查看任意陶瓷单元的 IP、序号、引脚位置
2. Tab 2 - IP+序号查询: 通过 IP+序号反查引脚编号和 36×36 位置

使用方式:
    streamlit run src/ao_shaping/gui/streamlit_helper/ceramic_viewer.py
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import pandas as pd
import streamlit as st

# =============================================================================
# Constants
# =============================================================================

GRID_SIZE = 36
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data"
ENRICHED_CSV = DATA_DIR / "1300-5-enriched.csv"

# =============================================================================
# Session State Initialization
# =============================================================================

def _initialize_state() -> None:
    """初始化 session_state 变量。"""
    st.session_state.setdefault("cv_selected_row", 0)
    st.session_state.setdefault("cv_selected_col", 0)
    st.session_state.setdefault("cv_edit_ip", 101)
    st.session_state.setdefault("cv_edit_seq", 0)
    st.session_state.setdefault("cv_lookup_ip", 101)
    st.session_state.setdefault("cv_lookup_seq", 0)
    st.session_state.setdefault("cv_lookup_result", None)
    st.session_state.setdefault("cv_data", None)
    st.session_state.setdefault("cv_error", "")
    st.session_state.setdefault("cv_confirm_save", False)
    st.session_state.setdefault("cv_conflict_info", None)
    st.session_state.setdefault("cv_save_feedback", "")


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data
def load_enriched_data() -> pd.DataFrame:
    """加载已处理的 1300 陶瓷单元数据。"""
    if not ENRICHED_CSV.exists():
        st.error(f"数据文件不存在: {ENRICHED_CSV}")
        st.info("请先运行 scripts/process_1300_data.py 生成数据文件")
        return pd.DataFrame()
    df = pd.read_csv(ENRICHED_CSV)
    return df


# =============================================================================
# Tab 1: 36×36 网格浏览
# =============================================================================

def _apply_edit(df: pd.DataFrame, row: int, col: int, new_ip: int, new_seq: int) -> None:
    """Apply IP组/序号 edit to the dataframe in place."""
    mask = (df["36×36行"] == row) & (df["36×36列"] == col)
    df.loc[mask, "IP组"] = new_ip
    df.loc[mask, "序号"] = new_seq


def _check_conflict(df: pd.DataFrame, row: int, col: int,
                    ip_val: int, seq_val: int) -> pd.DataFrame:
    """Return rows where (ip_val, seq_val) already exists at a different position."""
    return df[
        (df["IP组"] == ip_val)
        & (df["序号"] == seq_val)
        & ~((df["36×36行"] == row) & (df["36×36列"] == col))
    ]


def _render_conflict_section(df: pd.DataFrame, current_row: int, current_col: int) -> None:
    """Show conflict warning with confirm/cancel when a save conflict exists."""
    conflict = st.session_state.cv_conflict_info
    if not conflict:
        return

    st.divider()
    st.warning("⚠️ **检测到 IP组+序号 冲突**")

    for i, c in enumerate(conflict.get("conflicts", [])):
        st.markdown(
            f"- 单元格 **[{c['36×36行']}, {c['36×36列']}]** "
            f"(位置序号 {c['位置序号']}) — "
            f"已占用 IP组={c['IP组']}, 序号={c['序号']}"
        )

    col_cfm1, col_cfm2, _ = st.columns([1, 1, 4])
    with col_cfm1:
        if st.button("✅ 确认覆盖", type="primary", use_container_width=True, key="cv_confirm_overwrite"):
            _apply_edit(
                df,
                conflict["row"], conflict["col"],
                conflict["new_ip"], conflict["new_seq"],
            )
            st.session_state.cv_conflict_info = None
            st.session_state.cv_save_feedback = "已保存 (冲突覆盖)"
            st.rerun()
    with col_cfm2:
        if st.button("❌ 取消", use_container_width=True, key="cv_cancel_overwrite"):
            st.session_state.cv_conflict_info = None
            # Reset edit fields to original values
            orig = df[(df["36×36行"] == current_row) & (df["36×36列"] == current_col)]
            if not orig.empty:
                st.session_state.cv_edit_ip = int(orig.iloc[0]["IP组"])
                st.session_state.cv_edit_seq = int(orig.iloc[0]["序号"])
            st.rerun()


def _render_cell_editor(df: pd.DataFrame, sel_row: int, sel_col: int, position_num: int) -> None:
    """Show cell details + editable IP组/序号 + save button with conflict detection."""
    cell_data = df[(df["36×36行"] == sel_row) & (df["36×36列"] == sel_col)]
    if cell_data.empty:
        st.info(f"位置 [{sel_row}, {sel_col}] (序号 {position_num}) 无对应数据")
        return

    cell = cell_data.iloc[0]
    orig_ip = int(cell["IP组"])
    orig_seq = int(cell["序号"])
    orig_group = cell["组"]
    orig_pin = int(cell["引脚编号"])
    orig_conn = cell["连接器"]

    st.divider()
    st.markdown(f"##### 📍 单元格 [{sel_row}, {sel_col}] (位置序号 {position_num})")

    # Read-only info
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("所属组", orig_group)
    with meta_cols[1]:
        st.metric("引脚编号", orig_pin)
    with meta_cols[2]:
        st.metric("连接器", orig_conn)
    with meta_cols[3]:
        st.metric("位置序号", position_num)

    # Editable fields — use cell-dependent keys to avoid stale widget state
    st.markdown("**编辑 IP组 / 序号:**")
    edit_col1, edit_col2, edit_col3 = st.columns([1, 1, 1])

    key_suffix = f"_{sel_row}_{sel_col}"

    with edit_col1:
        new_ip = st.number_input("IP组", min_value=101, max_value=126,
                                  value=orig_ip, step=1,
                                  key=f"cv_ip{key_suffix}")
    with edit_col2:
        new_seq = st.number_input("序号", min_value=0, max_value=49,
                                   value=orig_seq, step=1,
                                   key=f"cv_seq{key_suffix}")

    # Feedback message
    fb = st.session_state.cv_save_feedback
    if fb:
        if "冲突" in fb:
            st.success(f"✅ {fb}")
        elif "失败" in fb:
            st.error(f"❌ {fb}")
        elif "已保存" in fb:
            st.success(f"✅ {fb}")
        st.session_state.cv_save_feedback = ""

    with edit_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        changed = (new_ip != orig_ip) or (new_seq != orig_seq)
        if st.button("💾 保存修改", type="primary", use_container_width=True,
                     disabled=not changed, key=f"cv_save{key_suffix}"):
            # Check conflicts
            conflicts = _check_conflict(df, sel_row, sel_col, new_ip, new_seq)
            if not conflicts.empty:
                st.session_state.cv_conflict_info = {
                    "row": sel_row, "col": sel_col,
                    "new_ip": new_ip, "new_seq": new_seq,
                    "conflicts": conflicts.to_dict("records"),
                }
                st.rerun()
            else:
                _apply_edit(df, sel_row, sel_col, new_ip, new_seq)
                st.session_state.cv_save_feedback = "已保存"
                st.rerun()

    if not changed:
        st.caption("值未改变，无需保存")
    elif st.session_state.cv_conflict_info is not None:
        st.caption("⚠️ 存在未解决的冲突，请在上方确认或取消")


def render_tab_grid() -> None:
    """渲染 Tab 1: 36×36 网格浏览。"""
    df = st.session_state.cv_data
    if df.empty:
        return

    st.markdown("##### 36×36 陶瓷单元网格")
    st.caption("点击网格行选择单元格，用列选择器精确定位列，下方可编辑 IP组/序号")

    # ---- Selection: grid click + position / row+col inputs ----
    col_sel1, col_sel2, col_sel3 = st.columns([1, 1, 1])
    with col_sel1:
        use_position = st.checkbox("按位置序号输入", value=False,
                                    help="勾选后用 1~1296 序号选择")
    with col_sel2:
        if st.button("🔄 重置选择", use_container_width=True, key="cv_reset_lookup"):
            st.session_state.cv_selected_row = 0
            st.session_state.cv_selected_col = 0
            st.session_state.cv_conflict_info = None
            st.session_state.cv_edit_ip = 101
            st.session_state.cv_edit_seq = 0
            st.rerun()
    with col_sel3:
        if st.button("🗑️ 放弃编辑", use_container_width=True):
            st.session_state.cv_conflict_info = None
            st.rerun()

    if use_position:
        position = st.number_input("位置序号", min_value=1, max_value=GRID_SIZE * GRID_SIZE,
                                    value=st.session_state.cv_selected_row * GRID_SIZE
                                    + st.session_state.cv_selected_col + 1,
                                    step=1, key="cv_pos_input")
        sel_row = (position - 1) // GRID_SIZE
        sel_col = (position - 1) % GRID_SIZE
    else:
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            sel_row = st.number_input("行 (0~35)", min_value=0, max_value=GRID_SIZE - 1,
                                       value=st.session_state.cv_selected_row, step=1,
                                       key="cv_row_input")
        with col_r2:
            sel_col = st.number_input("列 (0~35)", min_value=0, max_value=GRID_SIZE - 1,
                                       value=st.session_state.cv_selected_col, step=1,
                                       key="cv_col_input")

    st.session_state.cv_selected_row = sel_row
    st.session_state.cv_selected_col = sel_col
    position_num = sel_row * GRID_SIZE + sel_col + 1

    # ---- Conflict resolution section (shown when conflict detected) ----
    _render_conflict_section(df, sel_row, sel_col)

    # ---- Detail + Edit section ----
    _render_cell_editor(df, sel_row, sel_col, position_num)

    # ---- 36×36 网格 ----
    st.divider()
    st.markdown("##### 36×36 网格概览")

    col_grid_ctrl1, col_grid_ctrl2 = st.columns([2, 1])
    with col_grid_ctrl1:
        color_scheme = st.selectbox(
            "网格着色方案",
            ["无", "按IP组", "按所属组", "按序号", "按引脚编号"],
            index=0,
            key="cv_color_scheme",
        )
    with col_grid_ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 重置选择", use_container_width=True, key="cv_reset_grid"):
            st.session_state.cv_selected_row = 0
            st.session_state.cv_selected_col = 0
            st.rerun()

    # Build 36×36 grid: position numbers
    grid_matrix = df.pivot_table(
        index="36×36行", columns="36×36列", values="位置序号", aggfunc="first"
    ).values
    grid_df = pd.DataFrame(grid_matrix)
    grid_df.columns = [f"C{c}" for c in range(GRID_SIZE)]
    grid_df.index = [f"R{r}" for r in range(GRID_SIZE)]

    # ---- Build color lookup ----
    GROUP_COLORS = {
        "一组": "#4CAF50", "二组": "#2196F3",
        "三组": "#FF9800", "四组": "#9C27B0", "五组": "#F44336",
    }

    if color_scheme == "无":
        def _style_fn(val: int) -> str:
            if val == position_num:
                return "background-color: #ffeb3b; font-weight: bold; color: black"
            return ""
    elif color_scheme == "按所属组":
        pos_group = df.set_index("位置序号")["组"].to_dict()
        def _style_fn(val: int) -> str:
            g = pos_group.get(val)
            bg = GROUP_COLORS.get(g, "#ffffff")
            if val == position_num:
                return f"background-color: {bg}; border: 2px solid black; font-weight: bold"
            return f"background-color: {bg}; color: white"
    elif color_scheme == "按IP组":
        pos_ip = df.set_index("位置序号")["IP组"].to_dict()
        # Evenly spaced hues in HSV for 26 values (101-126)
        def _ip_color(ip_val: int) -> str:
            t = (ip_val - 101) / 25.0  # 0..1
            r, g, b = colorsys.hsv_to_rgb(t * 0.8, 0.6, 0.9)
            return f"rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})"
        def _style_fn(val: int) -> str:
            ip = pos_ip.get(val)
            bg = _ip_color(ip) if ip is not None else "#ffffff"
            if val == position_num:
                return f"background-color: {bg}; border: 2px solid black; font-weight: bold"
            return f"background-color: {bg}"
    elif color_scheme == "按序号":
        pos_seq = df.set_index("位置序号")["序号"].to_dict()
        def _style_fn(val: int) -> str:
            s = pos_seq.get(val, 0)
            intensity = s / 49.0
            g = int(200 - intensity * 180)
            b = int(200 - intensity * 120)
            bg = f"rgb({g},{b},{255 - int(intensity * 100)})"
            if val == position_num:
                return f"background-color: {bg}; border: 2px solid black; font-weight: bold"
            return f"background-color: {bg}"
    elif color_scheme == "按引脚编号":
        pos_pin = df.set_index("位置序号")["引脚编号"].to_dict()
        def _style_fn(val: int) -> str:
            p = pos_pin.get(val, 0)
            t = min(p / 316.0, 1.0)
            r = int(50 + t * 200)
            bg = f"rgb({r},{int(180*(1-t))},{int(80*(1-t))})"
            if val == position_num:
                return f"background-color: {bg}; border: 2px solid black; font-weight: bold"
            return f"background-color: {bg}"
    else:
        def _style_fn(val: int) -> str:
            return "background-color: #ffeb3b" if val == position_num else ""

    styled = grid_df.style.map(_style_fn)

    grid_event = st.dataframe(
        styled,
        height=520,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-cell",
        key="cv_grid_main",
    )

    # Capture click selection from grid — click highlights the cell in the grid,
    # updates session state, and reruns so the cell editor / number_inputs refresh.
    sel = grid_event.selection if grid_event else None
    if sel and sel.get("rows") and sel.get("columns"):
        clicked_row = sel["rows"][0]
        clicked_col = sel["columns"][0]
        if isinstance(clicked_row, str) and clicked_row.startswith("R"):
            clicked_row = int(clicked_row[1:])
        if isinstance(clicked_col, str) and clicked_col.startswith("C"):
            clicked_col = int(clicked_col[1:])
        if (0 <= clicked_row < GRID_SIZE and 0 <= clicked_col < GRID_SIZE
            and (clicked_row != st.session_state.cv_selected_row
                 or clicked_col != st.session_state.cv_selected_col)):
            st.session_state.cv_selected_row = clicked_row
            st.session_state.cv_selected_col = clicked_col
            # Force number_inputs to re-initialize from cv_selected_row/col
            st.session_state.pop("cv_row_input", None)
            st.session_state.pop("cv_col_input", None)
            st.session_state.pop("cv_pos_input", None)
            st.rerun()

    # ---- Legend ----
    def _swatch(bg: str, label: str) -> str:
        return f"<span style='display:inline-block;width:14px;height:14px;background:{bg};border-radius:3px;border:1px solid #999;vertical-align:middle'></span> <span style='font-size:0.85em'>{label}</span>"

    if color_scheme == "按所属组":
        st.markdown("**图例 — 所属组:**")
        leg_cols = st.columns(5)
        for i, (g, c) in enumerate(GROUP_COLORS.items()):
            with leg_cols[i]:
                st.markdown(_swatch(c, g), unsafe_allow_html=True)
    elif color_scheme == "按IP组":
        st.markdown("**图例 — IP组 (HSV 色调渐变):**")
        samples = [101, 106, 111, 116, 121, 126]
        leg_cols = st.columns(len(samples))
        for i, ip in enumerate(samples):
            t = (ip - 101) / 25.0
            r, g, b = colorsys.hsv_to_rgb(t * 0.8, 0.6, 0.9)
            bg = f"rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})"
            with leg_cols[i]:
                st.markdown(_swatch(bg, str(ip)), unsafe_allow_html=True)
    elif color_scheme == "按序号":
        st.markdown("**图例 — 序号 (蓝绿色阶):**")
        samples = [0, 12, 25, 37, 49]
        leg_cols = st.columns(len(samples))
        for i, s in enumerate(samples):
            intensity = s / 49.0
            gr = int(200 - intensity * 180)
            b = int(200 - intensity * 120)
            bg = f"rgb({gr},{b},{255 - int(intensity * 100)})"
            with leg_cols[i]:
                st.markdown(_swatch(bg, str(s)), unsafe_allow_html=True)
    elif color_scheme == "按引脚编号":
        st.markdown("**图例 — 引脚编号 (红色阶):**")
        samples = [1, 80, 158, 237, 316]
        leg_cols = st.columns(len(samples))
        for i, p in enumerate(samples):
            t = min(p / 316.0, 1.0)
            r = int(50 + t * 200)
            bg = f"rgb({r},{int(180*(1-t))},{int(80*(1-t))})"
            with leg_cols[i]:
                st.markdown(_swatch(bg, str(p)), unsafe_allow_html=True)
    elif color_scheme == "无":
        st.caption("黄色高亮 = 当前位置 (行 {}, 列 {}, 序号 {})".format(sel_row, sel_col, position_num))


# =============================================================================
# Tab 2: IP+序号查询
# =============================================================================

def lookup_by_ip_seq(df: pd.DataFrame, ip_group: int, seq: int) -> pd.DataFrame:
    """根据 IP 组和序号查询对应单元信息，可能返回多条。"""
    matches = df[(df["IP组"] == ip_group) & (df["序号"] == seq)]
    return matches


def render_tab_lookup() -> None:
    """渲染 Tab 2: IP+序号查询。"""
    df = st.session_state.cv_data
    if df.empty:
        return

    st.markdown("##### IP + 序号查询")
    st.caption("通过 IP 组号和序号查询对应的引脚编号和 36×36 网格位置")

    col_ip, col_seq = st.columns([1, 1])
    with col_ip:
        ip_group = st.number_input("IP组 (101~126)", min_value=101, max_value=126,
                                   value=st.session_state.cv_lookup_ip, step=1, key="cv_lookup_ip_input")
    with col_seq:
        seq = st.number_input("序号 (0~49)", min_value=0, max_value=49,
                              value=st.session_state.cv_lookup_seq, step=1, key="cv_lookup_seq_input")

    st.session_state.cv_lookup_ip = ip_group
    st.session_state.cv_lookup_seq = seq

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🔍 查询", type="primary", use_container_width=True, key="cv_lookup_btn"):
            result = lookup_by_ip_seq(df, ip_group, seq)
            st.session_state.cv_lookup_result = result

    # ---- 显示查询结果 ----
    result = st.session_state.cv_lookup_result
    if result is not None and not result.empty:
        # Check if the displayed result matches current query
        if len(result) > 0 and int(result.iloc[0]["IP组"]) == ip_group and int(result.iloc[0]["序号"]) == seq:
            st.divider()
            n_results = len(result)
            st.markdown(f"##### ✅ 查询结果 ({n_results} 条{'记录' if n_results > 1 else ''})")

            for i, (_, row_data) in enumerate(result.iterrows()):
                r = int(row_data["36×36行"])
                c = int(row_data["36×36列"])
                pos = r * GRID_SIZE + c + 1

                if n_results > 1:
                    st.markdown(f"**匹配 {i+1}:**")
                else:
                    st.markdown("")

                meta_c1, meta_c2, meta_c3 = st.columns(3)
                with meta_c1:
                    st.metric("36×36位置", f"[{r}, {c}]")
                with meta_c2:
                    st.metric("位置序号", pos)
                with meta_c3:
                    st.metric("所属组", row_data["组"])

                meta_c4, meta_c5 = st.columns(2)
                with meta_c4:
                    st.metric("引脚编号", int(row_data["引脚编号"]))
                with meta_c5:
                    st.metric("连接器", row_data["连接器"])

                if i < n_results - 1:
                    st.divider()
        else:
            st.info("请点击「查询」按钮查看结果")
    elif result is not None and result.empty:
        st.warning(f"未找到 IP组={ip_group}, 序号={seq} 的对应记录")
    else:
        st.info("请输入 IP 组和序号，点击查询")

        # Show available IPs
        st.divider()
        st.markdown("##### 可用 IP 组")
        ip_list = sorted(df["IP组"].unique())
        ip_cols = st.columns(7)
        for idx, ip in enumerate(ip_list):
            col_idx = idx % 7
            count = len(df[df["IP组"] == ip])
            with ip_cols[col_idx]:
                st.caption(f"**{int(ip)}** ({count}单元)")


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar() -> None:
    """渲染侧边栏信息。"""
    with st.sidebar:
        st.header("📊 数据概览")

        df = st.session_state.cv_data
        if not df.empty:
            st.metric("陶瓷单元总数", len(df))
            st.metric("网格大小", f"{GRID_SIZE} × {GRID_SIZE}")

            with st.container(border=True):
                st.markdown("##### 数据列说明")
                st.caption("**位置序号**: 1~1296 的排列位置")
                st.caption("**36×36行/列**: 在 36×36 网格中的行列坐标")
                st.caption("**IP组**: 单元所属 IP 组 (101~126)")
                st.caption("**序号**: 组内序号 (0~49)")
                st.caption("**所属组**: 对应输出线序表组别 (一组~五组)")
                st.caption("**引脚编号**: 机柜输出引脚编号")
                st.caption("**连接器**: 连接器标识")

            with st.container(border=True):
                st.markdown("##### 数据分布 — 所属组")
                for group_name in ["一组", "二组", "三组", "四组", "五组"]:
                    cnt = len(df[df["组"] == group_name])
                    st.caption(f"{group_name}: {cnt} 单元")

            with st.container(border=True):
                st.markdown("##### 数据分布 — IP组")
                ip_groups = sorted(df["IP组"].unique())
                ip_cols = st.columns(3)
                for idx, ip in enumerate(ip_groups):
                    cnt = len(df[df["IP组"] == int(ip)])
                    with ip_cols[idx % 3]:
                        st.caption(f"IP{int(ip):>3}: {cnt} 单元")

            with st.container(border=True):
                st.markdown("##### 数据源")
                st.caption(f"文件: `{ENRICHED_CSV.name}`")
                if ENRICHED_CSV.exists():
                    st.caption(f"大小: {ENRICHED_CSV.stat().st_size / 1024:.1f} KB")
        else:
            st.error("数据加载失败")
            st.info("运行 scripts/process_1300_data.py 生成数据")


# =============================================================================
# Main App
# =============================================================================

def main() -> None:
    """Streamlit 应用主入口。"""
    st.set_page_config(
        page_title="1300 陶瓷单元查看工具",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 1300 陶瓷单元查看工具")
    st.caption("Adaptive Optics 陶瓷驱动器 | 36×36 网格浏览 · IP+序号查询引脚关系")

    _initialize_state()

    # Load data
    if st.session_state.cv_data is None:
        st.session_state.cv_data = load_enriched_data()

    render_sidebar()

    tab1, tab2 = st.tabs(["36×36 网格浏览", "IP+序号查询"])

    with tab1:
        render_tab_grid()

    with tab2:
        render_tab_lookup()


if __name__ == "__main__":
    main()
