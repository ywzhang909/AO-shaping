"""
1300陶瓷单元查看工具 (Streamlit)

功能:
1. Tab 1 - 36×36 网格浏览: 查看任意陶瓷单元的 IP、序号、引脚位置
2. Tab 2 - IP+序号查询: 通过 IP+序号反查引脚编号和 36×36 位置

使用方式:
    streamlit run src/ao_shaping/gui/dm/ceramic_viewer.py
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Configuration
# =============================================================================


class Config:
    """应用常数配置。"""

    GRID_SIZE = 36
    IP_MIN = 101
    IP_MAX = 126
    SEQ_MIN = 0
    SEQ_MAX = 49
    PIN_MAX = 316
    DEFAULT_IP = 101
    DEFAULT_SEQ = 0
    COLOR_SCHEMES = ("无", "按IP组", "按所属组", "按序号", "按引脚编号")

    GROUP_COLORS = {
        "一组": "#4CAF50",
        "二组": "#2196F3",
        "三组": "#FF9800",
        "四组": "#9C27B0",
        "五组": "#F44336",
    }

    DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data"
    DEFAULT_CSV = "1300-5-enriched.csv"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CellInfo:
    """单个陶瓷单元的数据。"""

    row: int
    col: int
    position: int
    ip_group: int
    seq: int
    group: str
    pin: int
    connector: str

    @classmethod
    def from_df(cls, df: pd.DataFrame, row: int, col: int) -> CellInfo | None:
        """从 DataFrame 按 [row, col] 查找并构造 CellInfo。"""
        cell = df[(df["36×36行"] == row) & (df["36×36列"] == col)]
        if cell.empty:
            return None
        c = cell.iloc[0]
        pos = row * Config.GRID_SIZE + col + 1
        return cls(
            row=row,
            col=col,
            position=pos,
            ip_group=int(c["IP组"]),
            seq=int(c["序号"]),
            group=c["组"],
            pin=int(c["引脚编号"]),
            connector=c["连接器"],
        )


@dataclass
class ConflictInfo:
    """冲突信息，保存时检测到 IP组+序号 重复时使用。"""

    row: int
    col: int
    new_ip: int
    new_seq: int
    conflicts: list[dict]


# =============================================================================
# Session State Initialization
# =============================================================================


def _initialize_state() -> None:
    """初始化 session_state 变量。"""
    # 选择状态
    st.session_state.setdefault("cv_selected_row", 0)
    st.session_state.setdefault("cv_selected_col", 0)
    # 编辑状态
    st.session_state.setdefault("cv_edit_ip", Config.DEFAULT_IP)
    st.session_state.setdefault("cv_edit_seq", Config.DEFAULT_SEQ)
    st.session_state.setdefault("cv_conflict_info", None)
    st.session_state.setdefault("cv_save_feedback", "")
    # 查询状态
    st.session_state.setdefault("cv_lookup_ip", Config.DEFAULT_IP)
    st.session_state.setdefault("cv_lookup_seq", Config.DEFAULT_SEQ)
    st.session_state.setdefault("cv_lookup_result", None)
    st.session_state.setdefault("cv_lookup_positions", None)
    # 网格显示模式
    st.session_state.setdefault("cv_grid_display_mode", "位置序号")
    st.session_state.setdefault("cv_grid_color_mode", "不着色")
    # 高亮列表
    st.session_state.setdefault("cv_highlight_mode", "位置序号")
    st.session_state.setdefault("cv_highlight_input", "")
    st.session_state.setdefault("cv_highlight_positions", None)
    # 数据
    st.session_state.setdefault("cv_data", None)
    st.session_state.setdefault("cv_csv_filename", Config.DEFAULT_CSV)
    st.session_state.setdefault("cv_dirty", False)


# =============================================================================
# CSV Path Helpers
# =============================================================================


def _current_csv_path() -> Path:
    """返回当前配置的 CSV 完整路径。"""
    return Config.DATA_DIR / st.session_state.cv_csv_filename


# =============================================================================
# Data Loading
# =============================================================================


@st.cache_data
def load_enriched_data(csv_path: str) -> pd.DataFrame:
    """加载已处理的 1300 陶瓷单元数据。"""
    path = Path(csv_path)
    if not path.exists():
        st.error(f"数据文件不存在: {path}")
        st.info("请先运行 scripts/process_1300_data.py 生成数据文件")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


# =============================================================================
# Tab 1: 36×36 网格浏览
# =============================================================================


def _apply_edit(
    df: pd.DataFrame, row: int, col: int, new_ip: int, new_seq: int
) -> None:
    """Apply IP组/序号 edit to the dataframe in place and mark dirty."""
    mask = (df["36×36行"] == row) & (df["36×36列"] == col)
    df.loc[mask, "IP组"] = new_ip
    df.loc[mask, "序号"] = new_seq
    st.session_state.cv_dirty = True


def _check_conflict(
    df: pd.DataFrame, row: int, col: int, ip_val: int, seq_val: int
) -> pd.DataFrame:
    """Return rows where (ip_val, seq_val) already exists at a different position."""
    return df[
        (df["IP组"] == ip_val)
        & (df["序号"] == seq_val)
        & ~((df["36×36行"] == row) & (df["36×36列"] == col))
    ]


def _render_conflict_section(
    df: pd.DataFrame, current_row: int, current_col: int
) -> None:
    """Show conflict warning with confirm/cancel when a save conflict exists."""
    raw = st.session_state.cv_conflict_info
    if not raw:
        return
    conflict = ConflictInfo(**raw)

    st.divider()
    st.warning("⚠️ **检测到 IP组+序号 冲突**")

    for c in conflict.conflicts:
        st.markdown(
            f"- 单元格 **[{c['36×36行']}, {c['36×36列']}]** "
            f"(位置序号 {c['位置序号']}) — "
            f"已占用 IP组={c['IP组']}, 序号={c['序号']}"
        )

    col_cfm1, col_cfm2, _ = st.columns([1, 1, 4])
    with col_cfm1:
        if st.button(
            "✅ 确认覆盖",
            type="primary",
            use_container_width=True,
            key="cv_confirm_overwrite",
        ):
            _apply_edit(
                df, conflict.row, conflict.col, conflict.new_ip, conflict.new_seq
            )
            st.session_state.cv_conflict_info = None
            st.session_state.cv_save_feedback = "已保存 (冲突覆盖)"
            st.rerun()
    with col_cfm2:
        if st.button("❌ 取消", use_container_width=True, key="cv_cancel_overwrite"):
            st.session_state.cv_conflict_info = None
            orig = df[(df["36×36行"] == current_row) & (df["36×36列"] == current_col)]
            if not orig.empty:
                st.session_state.cv_edit_ip = int(orig.iloc[0]["IP组"])
                st.session_state.cv_edit_seq = int(orig.iloc[0]["序号"])
            st.rerun()


def _render_cell_editor(
    df: pd.DataFrame, sel_row: int, sel_col: int, position_num: int
) -> None:
    """Show cell details + editable IP组/序号 + save button with conflict detection."""
    info = CellInfo.from_df(df, sel_row, sel_col)
    if info is None:
        st.info(f"位置 [{sel_row}, {sel_col}] (序号 {position_num}) 无对应数据")
        return

    st.divider()
    st.markdown(f"##### 📍 单元格 [{info.row}, {info.col}] (位置序号 {info.position})")

    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("所属组", info.group)
    with meta_cols[1]:
        st.metric("引脚编号", info.pin)
    with meta_cols[2]:
        st.metric("连接器", info.connector)
    with meta_cols[3]:
        st.metric("位置序号", info.position)

    st.markdown("**编辑 IP组 / 序号:**")
    edit_col1, edit_col2, edit_col3 = st.columns([1, 1, 1])

    key_suffix = f"_{sel_row}_{sel_col}"

    with edit_col1:
        new_ip = st.number_input(
            "IP组",
            min_value=Config.IP_MIN,
            max_value=Config.IP_MAX,
            value=info.ip_group,
            step=1,
            key=f"cv_ip{key_suffix}",
        )
    with edit_col2:
        new_seq = st.number_input(
            "序号",
            min_value=Config.SEQ_MIN,
            max_value=Config.SEQ_MAX,
            value=info.seq,
            step=1,
            key=f"cv_seq{key_suffix}",
        )

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
        changed = (new_ip != info.ip_group) or (new_seq != info.seq)
        if st.button(
            "💾 保存修改",
            type="primary",
            use_container_width=True,
            disabled=not changed,
            key=f"cv_save{key_suffix}",
        ):
            conflicts = _check_conflict(df, sel_row, sel_col, new_ip, new_seq)
            if not conflicts.empty:
                st.session_state.cv_conflict_info = {
                    "row": sel_row,
                    "col": sel_col,
                    "new_ip": new_ip,
                    "new_seq": new_seq,
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


# ── Cached helpers for render_tab_grid ────────────────────────────────────


@st.cache_data
def _cv_grid_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute the 36×36 position-number matrix from raw data."""
    return df.pivot_table(
        index="36×36行", columns="36×36列", values="位置序号", aggfunc="first"
    ).values


@st.cache_data
def _cv_grid_display_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Build a 36×36 display DataFrame for the selected grid mode."""
    G = Config.GRID_SIZE
    pos_matrix = _cv_grid_matrix(df)
    display = np.empty_like(pos_matrix, dtype=object)

    pos_info = df.set_index("位置序号")
    for r in range(G):
        for c in range(G):
            pos = int(pos_matrix[r, c])
            if mode == "IP-序号":
                ip = int(pos_info.loc[pos, "IP组"])
                seq = int(pos_info.loc[pos, "序号"])
                display[r, c] = f"{ip}-{seq:02d}"
            elif mode == "组-针脚":
                group = str(pos_info.loc[pos, "组"])
                pin = int(pos_info.loc[pos, "引脚编号"])
                display[r, c] = f"{group}-{pin}"
            else:
                display[r, c] = str(pos)

    grid_df = pd.DataFrame(display)
    grid_df.columns = [f"C{c}" for c in range(G)]
    grid_df.index = [f"R{r}" for r in range(G)]
    return grid_df


@st.cache_data
def _cv_color_scheme_map(df: pd.DataFrame, mode: str) -> dict[int, str]:
    """Build position → background color map for the selected color mode."""
    if mode == "按IP组":
        pos_ip = df.set_index("位置序号")["IP组"].to_dict()
        ip_values = sorted(df["IP组"].unique())
        n_ip = len(ip_values)
        ip_color_map: dict[int, str] = {}
        for idx, ip in enumerate(ip_values):
            hue = idx / max(n_ip - 1, 1) * 0.85
            r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
            ip_color_map[int(ip)] = f"rgb({r * 255:.0f},{g * 255:.0f},{b * 255:.0f})"
        return {pos: ip_color_map.get(int(ip), "#ffffff") for pos, ip in pos_ip.items()}
    elif mode == "按组":
        pos_group = df.set_index("位置序号")["组"].to_dict()
        return {k: Config.GROUP_COLORS.get(v, "#ffffff") for k, v in pos_group.items()}
    return {}


def _apply_grid_color(
    grid_df: pd.DataFrame,
    df: pd.DataFrame,
    color_mode: str,
    highlight_positions: set[int] | None,
) -> pd.DataFrame:
    """Apply background colors to the grid DataFrame based on color mode and highlights."""
    if color_mode == "不着色" and not highlight_positions:
        return grid_df

    pos_matrix = _cv_grid_matrix(df)
    color_map = _cv_color_scheme_map(df, color_mode)

    styled = grid_df.copy()

    for r in range(Config.GRID_SIZE):
        for c in range(Config.GRID_SIZE):
            pos = int(pos_matrix[r, c])
            cell_value = styled.iat[r, c]
            bg = ""
            if highlight_positions and pos in highlight_positions:
                bg = "background-color: #FFD700; font-weight: bold"
            elif color_mode != "不着色" and pos in color_map:
                bg = f"background-color: {color_map[pos]}"
            styled.iat[r, c] = (
                f"<div style='{bg}'>{cell_value}</div>" if bg else cell_value
            )

    return styled


# ── Tab 1: Grid Renderer ────────────────────────────────────


@st.fragment()
def render_tab_grid() -> None:
    """渲染 Tab 1: 36×36 网格浏览 (fragment, 点击单元格不会 rerun 全应用)。"""
    df = st.session_state.cv_data
    if df.empty:
        return

    st.markdown("##### 36×36 陶瓷单元网格")
    st.caption("点击网格单元格选择，下方可编辑 IP组/序号")

    col_sel1, col_sel2, col_sel3 = st.columns([1, 1, 1])
    with col_sel1:
        use_position = st.checkbox(
            "按位置序号输入", value=False, help="勾选后用 1~1296 序号选择"
        )
    with col_sel2:
        if st.button("🔄 重置选择", use_container_width=True, key="cv_reset_lookup"):
            st.session_state.cv_selected_row = 0
            st.session_state.cv_selected_col = 0
            st.session_state.cv_conflict_info = None
            st.session_state.cv_edit_ip = Config.DEFAULT_IP
            st.session_state.cv_edit_seq = Config.DEFAULT_SEQ
            st.session_state.cv_grid_click_count += 1
            st.rerun()
    with col_sel3:
        if st.button("🗑️ 放弃编辑", use_container_width=True, key="cv_discard_grid"):
            st.session_state.cv_conflict_info = None
            st.rerun()

    G = Config.GRID_SIZE

    # ── Manual input callbacks ──────────────────────────────────────────
    if "cv_grid_click_count" not in st.session_state:
        st.session_state.cv_grid_click_count = 0

    def _on_row_input() -> None:
        key = f"cv_row_{st.session_state.cv_grid_click_count}"
        st.session_state.cv_selected_row = st.session_state[key]

    def _on_col_input() -> None:
        key = f"cv_col_{st.session_state.cv_grid_click_count}"
        st.session_state.cv_selected_col = st.session_state[key]

    def _on_pos_input() -> None:
        key = f"cv_pos_{st.session_state.cv_grid_click_count}"
        pos = st.session_state[key]
        st.session_state.cv_selected_row = (pos - 1) // G
        st.session_state.cv_selected_col = (pos - 1) % G

    _row_key = f"cv_row_{st.session_state.cv_grid_click_count}"
    _col_key = f"cv_col_{st.session_state.cv_grid_click_count}"
    _pos_key = f"cv_pos_{st.session_state.cv_grid_click_count}"

    if use_position:
        st.number_input(
            "位置序号",
            min_value=1,
            max_value=G * G,
            value=st.session_state.cv_selected_row * G
            + st.session_state.cv_selected_col
            + 1,
            step=1,
            key=_pos_key,
            on_change=_on_pos_input,
        )
    else:
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.number_input(
                "行 (0~35)",
                min_value=0,
                max_value=G - 1,
                value=st.session_state.cv_selected_row,
                step=1,
                key=_row_key,
                on_change=_on_row_input,
            )
        with col_r2:
            st.number_input(
                "列 (0~35)",
                min_value=0,
                max_value=G - 1,
                value=st.session_state.cv_selected_col,
                step=1,
                key=_col_key,
                on_change=_on_col_input,
            )

    sel_row = st.session_state.cv_selected_row
    sel_col = st.session_state.cv_selected_col
    position_num = sel_row * G + sel_col + 1

    st.divider()
    st.markdown("##### 36×36 网格概览")
    st.markdown(
        """
        <style>
        /* AG Grid 行选中：只有文字加粗，无背景色 */
        .ag-row-selected {
            font-weight: bold !important;
            background: none !important;
        }
        .ag-row-selected .ag-cell {
            font-weight: bold !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    col_grid_ctrl1, col_grid_ctrl2, col_grid_ctrl3, col_grid_ctrl4 = st.columns(
        [2, 1, 1, 1]
    )
    with col_grid_ctrl1:
        st.selectbox(
            "网格显示内容",
            ["位置序号", "IP-序号", "组-针脚"],
            index=["位置序号", "IP-序号", "组-针脚"].index(
                st.session_state.cv_grid_display_mode
            ),
            key="cv_grid_display_mode",
        )
    with col_grid_ctrl2:
        st.selectbox(
            "网格着色",
            ["不着色", "按IP组", "按组"],
            index=["不着色", "按IP组", "按组"].index(
                st.session_state.cv_grid_color_mode
            ),
            key="cv_grid_color_mode",
        )
    with col_grid_ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 重置选择", use_container_width=True, key="cv_reset_grid"):
            st.session_state.cv_selected_row = 0
            st.session_state.cv_selected_col = 0
            st.session_state.cv_grid_click_count += 1
            st.rerun()
    with col_grid_ctrl4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 导出 CSV", use_container_width=True, key="cv_export_csv"):
            export_df = _cv_grid_display_df(df, st.session_state.cv_grid_display_mode)
            csv_data = export_df.to_csv(index=True, encoding="utf-8-sig")
            st.download_button(
                label="下载 CSV",
                data=csv_data,
                file_name=f"grid_{st.session_state.cv_grid_display_mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="cv_download_csv",
            )

    grid_df = _cv_grid_display_df(df, st.session_state.cv_grid_display_mode)

    st.caption("点击表格行选择，或使用上方行/列输入框精确定位")

    highlight_positions = st.session_state.get("cv_lookup_positions")
    highlight_list = st.session_state.get("cv_highlight_positions")
    if highlight_positions and highlight_list:
        all_highlights = highlight_positions | highlight_list
    elif highlight_positions:
        all_highlights = highlight_positions
    elif highlight_list:
        all_highlights = highlight_list
    else:
        all_highlights = None

    color_mode = st.session_state.cv_grid_color_mode
    styled_grid_df = _apply_grid_color(grid_df, df, color_mode, all_highlights)

    if (
        all_highlights
        or color_mode != "不着色"
        or st.session_state.cv_grid_display_mode != "位置序号"
    ):
        selected = st.dataframe(
            styled_grid_df,
            use_container_width=True,
            height=600,
            on_select="rerun",
            selection_mode="single-row",
        )
    else:
        selected = st.dataframe(
            grid_df,
            use_container_width=True,
            height=600,
            on_select="rerun",
            selection_mode="single-row",
        )

    if selected and hasattr(selected, "selection") and selected.selection.rows:
        clicked_row_idx = selected.selection.rows[0]
        if 0 <= clicked_row_idx < G:
            st.session_state.cv_selected_row = clicked_row_idx
            st.rerun()

    sel_row = st.session_state.cv_selected_row
    sel_col = st.session_state.cv_selected_col
    position_num = sel_row * G + sel_col + 1

    _render_conflict_section(df, sel_row, sel_col)
    _render_cell_editor(df, sel_row, sel_col, position_num)


# =============================================================================
# Tab 2: IP+序号查询
# =============================================================================


def lookup_by_ip_seq(df: pd.DataFrame, ip_group: int, seq: int) -> pd.DataFrame:
    """根据 IP 组和序号查询对应单元信息，可能返回多条。"""
    return df[(df["IP组"] == ip_group) & (df["序号"] == seq)]


def parse_highlight_input(
    df: pd.DataFrame, raw_input: str, mode: str
) -> set[int] | None:
    """Parse highlight input string into a set of position numbers."""
    if not raw_input.strip():
        return None
    pos_info = df.set_index("位置序号")
    positions: set[int] = set()
    for token in raw_input.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if mode == "位置序号":
                pos = int(token)
                if 1 <= pos <= Config.GRID_SIZE * Config.GRID_SIZE:
                    positions.add(pos)
            elif mode == "IP-序号":
                parts = token.split("-")
                if len(parts) == 2:
                    ip = int(parts[0])
                    seq = int(parts[1])
                    matched = df[(df["IP组"] == ip) & (df["序号"] == seq)]
                    for _, row_data in matched.iterrows():
                        positions.add(int(row_data["位置序号"]))
            elif mode == "组-针脚":
                parts = token.split("-")
                if len(parts) == 2:
                    group = parts[0]
                    pin = int(parts[1])
                    matched = df[(df["组"] == group) & (df["引脚编号"] == pin)]
                    for _, row_data in matched.iterrows():
                        positions.add(int(row_data["位置序号"]))
        except (ValueError, KeyError):
            continue
    return positions if positions else None


def render_tab_lookup() -> None:
    """渲染 Tab 2: IP+序号查询。"""
    df = st.session_state.cv_data
    if df.empty:
        return

    st.markdown("##### IP + 序号查询")
    st.caption("通过 IP 组号和序号查询对应的引脚编号和 36×36 网格位置")

    col_ip, col_seq = st.columns([1, 1])
    with col_ip:
        ip_group = st.number_input(
            "IP组 (101~126)",
            min_value=Config.IP_MIN,
            max_value=Config.IP_MAX,
            value=st.session_state.cv_lookup_ip,
            step=1,
            key="cv_lookup_ip_input",
        )
    with col_seq:
        seq = st.number_input(
            "序号 (0~49)",
            min_value=Config.SEQ_MIN,
            max_value=Config.SEQ_MAX,
            value=st.session_state.cv_lookup_seq,
            step=1,
            key="cv_lookup_seq_input",
        )

    st.session_state.cv_lookup_ip = ip_group
    st.session_state.cv_lookup_seq = seq

    col_btn, clear_btn, _ = st.columns([1, 1, 2])
    with col_btn:
        if st.button(
            "🔍 查询", type="primary", use_container_width=True, key="cv_lookup_btn"
        ):
            result = lookup_by_ip_seq(df, ip_group, seq)
            st.session_state.cv_lookup_result = result
            if not result.empty:
                highlight_positions = set()
                for _, row_data in result.iterrows():
                    r = int(row_data["36×36行"])
                    c = int(row_data["36×36列"])
                    pos = r * Config.GRID_SIZE + c + 1
                    highlight_positions.add(pos)
                st.session_state.cv_lookup_positions = highlight_positions
            else:
                st.session_state.cv_lookup_positions = set()
    with clear_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ 清除高亮", use_container_width=True, key="cv_clear_highlight"):
            st.session_state.cv_lookup_positions = None
            st.session_state.cv_lookup_result = None
            st.rerun()

    result = st.session_state.cv_lookup_result
    if result is not None and not result.empty:
        if (
            len(result) > 0
            and int(result.iloc[0]["IP组"]) == ip_group
            and int(result.iloc[0]["序号"]) == seq
        ):
            st.divider()
            n_results = len(result)
            st.markdown(
                f"##### ✅ 查询结果 ({n_results} 条{'记录' if n_results > 1 else ''})"
            )

            for i, (_, row_data) in enumerate(result.iterrows()):
                r = int(row_data["36×36行"])
                c = int(row_data["36×36列"])
                pos = r * Config.GRID_SIZE + c + 1

                if n_results > 1:
                    st.markdown(f"**匹配 {i + 1}:**")
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


def _save_to_csv() -> None:
    """将当前 DataFrame 保存到 CSV，成功后清理 dirty 标记。"""
    df = st.session_state.cv_data
    if df.empty:
        return
    path = _current_csv_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        st.session_state.cv_dirty = False
        st.success(f"✅ 已保存到 `{path}`")
    except Exception as e:
        st.error(f"❌ 保存失败: {e}")


def _reload_data() -> None:
    """重新加载 CSV 数据，清除缓存。"""
    load_enriched_data.clear()
    csv_path_str = str(_current_csv_path())
    st.session_state.cv_data = load_enriched_data(csv_path_str)


def render_sidebar() -> None:
    """渲染侧边栏信息。"""
    with st.sidebar:
        st.header("📊 数据概览")

        df = st.session_state.cv_data
        if not df.empty:
            st.metric("陶瓷单元总数", len(df))
            st.metric("网格大小", f"{Config.GRID_SIZE} × {Config.GRID_SIZE}")

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
                st.markdown("##### 高亮列表")
                st.caption("输入值列表，在表格中高亮显示对应单元格")

                highlight_mode = st.selectbox(
                    "输入模式",
                    ["位置序号", "IP-序号", "组-针脚"],
                    index=["位置序号", "IP-序号", "组-针脚"].index(
                        st.session_state.cv_highlight_mode
                    ),
                    key="cv_highlight_mode",
                )

                highlight_input = st.text_area(
                    "输入列表（逗号分隔）",
                    value=st.session_state.cv_highlight_input,
                    height=100,
                    key="cv_highlight_input",
                    placeholder="例如：\n1, 5, 10\n或 101-00, 101-01\n或 一组-1, 二组-2",
                )

                col_hl1, col_hl2 = st.columns([1, 1])
                with col_hl1:
                    if st.button(
                        "🔍 应用高亮",
                        type="primary",
                        use_container_width=True,
                        key="cv_apply_highlight",
                    ):
                        positions = parse_highlight_input(
                            df,
                            st.session_state.cv_highlight_input,
                            st.session_state.cv_highlight_mode,
                        )
                        st.session_state.cv_highlight_positions = positions
                        st.rerun()
                with col_hl2:
                    if st.button(
                        "🗑️ 清除高亮",
                        use_container_width=True,
                        key="cv_clear_highlight_list",
                    ):
                        st.session_state.cv_highlight_positions = None
                        st.session_state.cv_highlight_input = ""
                        st.rerun()

                current_hl = st.session_state.cv_highlight_positions
                if current_hl:
                    st.caption(f"当前高亮 {len(current_hl)} 个单元格")
                else:
                    st.caption("暂无高亮")

            with st.container(border=True):
                st.markdown("##### 数据源配置")
                prev_filename = st.session_state.cv_csv_filename
                new_filename = st.text_input(
                    "CSV 文件名",
                    value=prev_filename,
                    key="cv_csv_name_input",
                    help="文件名 (位于 data/ 目录下)",
                )
                if new_filename != prev_filename:
                    st.session_state.cv_csv_filename = new_filename
                    _reload_data()
                    st.rerun()

                path = _current_csv_path()
                st.caption(f"完整路径: `{path}`")
                if path.exists():
                    st.caption(f"文件大小: {path.stat().st_size / 1024:.1f} KB")

                col_save1, col_save2 = st.columns([1, 1])
                with col_save1:
                    dirty = st.session_state.cv_dirty
                    if st.button(
                        "💾 保存到 CSV",
                        type="primary" if dirty else "secondary",
                        use_container_width=True,
                        disabled=not dirty,
                        key="cv_save_csv",
                    ):
                        _save_to_csv()
                        st.rerun()
                with col_save2:
                    if st.button(
                        "🔄 重新加载", use_container_width=True, key="cv_reload_data"
                    ):
                        _reload_data()
                        st.rerun()

                if dirty:
                    st.warning("⚠️ 有未保存的修改")
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

    if st.session_state.cv_data is None:
        _reload_data()

    render_sidebar()

    tab1, tab2 = st.tabs(["36×36 网格浏览", "IP+序号查询"])

    with tab1:
        render_tab_grid()

    with tab2:
        render_tab_lookup()


if __name__ == "__main__":
    main()
