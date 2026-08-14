"""R50 controller UI — channel selection base module (pure logic + shared widgets).

Owns: configuration constants, ChannelInfo/GroupDef/ChannelSelection models,
CSV wiring index (1300-5-enriched.csv), joint-control flat-array mapping,
session-state helpers, feedback helpers, and the shared channel selector widget.

No top-level streamlit import: the module stays importable and testable in a
plain numpy/pandas environment (streamlit imported lazily where needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loguru import logger

from ao_shaping.drivers.dm.MicroDM import MAX_CHANNELS, VOLTAGE_MAX, VOLTAGE_MIN
from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT


# =============================================================================
# Configuration Constants
# =============================================================================


@dataclass(frozen=True)
class Cfg:
    """Application configuration constants."""

    # session_state key prefix
    PREFIX: str = "r50c"
    # single controller (hardware limits from drivers.dm.MicroDM)
    SINGLE_CHANNELS: int = MAX_CHANNELS
    DEFAULT_PORT: int = 10101
    # voltage hardware limits (never exceeded)
    HW_VOLTAGE_MIN: float = VOLTAGE_MIN
    HW_VOLTAGE_MAX: float = VOLTAGE_MAX
    # joint control 36x36 grid
    GRID_SIZE: int = 36
    DM_NUM_ACTUATORS: int = 96  # computed via __post_init__
    # UI refresh interval (s)
    REFRESH_INTERVAL: float = 0.15
    # debug
    DEBUG_LOG_MAX: int = 300
    DEBUG_TCP_HOST: str = "127.0.0.1"
    DEBUG_TCP_PORT: int = 9999
    # CSV data source
    CSV_PATH: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "1300-5-enriched.csv"
    )

    def __post_init__(self) -> None:
        # Can't modify frozen dataclass, so use object.__setattr__
        object.__setattr__(self, "DM_NUM_ACTUATORS", self.GRID_SIZE * self.GRID_SIZE)


CFG = Cfg()
P = CFG.PREFIX

# Module-level aliases for CFG (single source of truth remains CFG)
GRID_SIZE = CFG.GRID_SIZE
SINGLE_CHANNELS = CFG.SINGLE_CHANNELS
HW_VOLTAGE_MIN = CFG.HW_VOLTAGE_MIN
HW_VOLTAGE_MAX = CFG.HW_VOLTAGE_MAX
REFRESH_INTERVAL = CFG.REFRESH_INTERVAL
DEBUG_LOG_MAX = CFG.DEBUG_LOG_MAX
DEBUG_HOST = CFG.DEBUG_TCP_HOST
DEBUG_PORT = CFG.DEBUG_TCP_PORT
DM_NUM_ACTUATORS = CFG.DM_NUM_ACTUATORS


def key(name: str) -> str:
    """Session-state key helper: '{PREFIX}_{name}'."""
    return f"{P}_{name}"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ChannelInfo:
    """Physical unit info from 1300-5-enriched.csv."""

    ip_suffix: int
    payload_position: int  # 1-based
    physical_position: int  # 1-based 36x36 position
    group: str
    needle_id: int
    physical_label: str

    @property
    def ip(self) -> str:
        return f"192.168.0.{self.ip_suffix}"

    @property
    def description(self) -> str:
        desc = f"ch{self.payload_position}"
        if self.needle_id:
            desc += f" 针脚#{self.needle_id}"
        if self.physical_label:
            desc += f" ({self.physical_label})"
        desc += f" [{self.ip}]"
        return desc

    def short_info(self) -> str:
        """Brief needle info for channel labels: '一组 针脚#277 (3-3-1)'."""
        if self.needle_id:
            return f"{self.group} 针脚#{self.needle_id} ({self.physical_label})"
        return f"{self.group} ({self.physical_label})"


@dataclass
class GroupDef:
    """A named group of channels from CSV, keyed by controller IP suffix."""

    name: str
    channels_by_ip: dict[int, list[ChannelInfo]]

    @property
    def total_channels(self) -> int:
        return sum(len(chs) for chs in self.channels_by_ip.values())

    @property
    def all_payload_positions(self) -> list[int]:
        return sorted(
            ch.payload_position
            for chs in self.channels_by_ip.values()
            for ch in chs
        )


@dataclass
class ChannelSelection:
    """Shared channel selection: all channels (all_mode) or an explicit list."""

    all_mode: bool = False
    channels: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.all_mode and not self.channels

    def select_all(self, total: int) -> None:
        """Select every channel 0..total-1 (does NOT set all_mode)."""
        self.all_mode = False
        self.channels = list(range(total))

    def invert(self, total: int) -> None:
        """Invert the selection within 0..total-1."""
        selected = set(self.channels)
        self.channels = [i for i in range(total) if i not in selected]

    def normalized(self, total: int) -> list[int]:
        """Effective channels: all when all_mode, else sorted-deduped list."""
        if self.all_mode:
            return list(range(total))
        return sorted({int(c) for c in self.channels})


# =============================================================================
# CSV Wiring Index (1300-5-enriched.csv)
# =============================================================================


def load_csv() -> pd.DataFrame:
    """Load 1300-5-enriched.csv, returns empty DataFrame on failure."""
    if not CFG.CSV_PATH.exists():
        logger.warning(f"CSV not found: {CFG.CSV_PATH}")
        return pd.DataFrame()
    try:
        return pd.read_csv(CFG.CSV_PATH)
    except Exception as e:
        logger.warning(f"Failed to load CSV: {e}")
        return pd.DataFrame()


def row_to_channel_info(row: pd.Series) -> ChannelInfo:
    """Convert a CSV row to ChannelInfo."""
    ip_suffix = int(row["IP组"])
    pp = int(row["序号"]) + 1  # 0-based -> 1-based
    r = int(row["36×36行"])
    c = int(row["36×36列"])
    pos = r * CFG.GRID_SIZE + c + 1  # 1-based
    return ChannelInfo(
        ip_suffix=ip_suffix,
        payload_position=pp,
        physical_position=pos,
        group=str(row["组"]),
        needle_id=int(row["引脚编号"]),
        physical_label=str(row["连接器"]),
    )


def build_csv_index(df: pd.DataFrame | None = None) -> dict[tuple[int, int], ChannelInfo]:
    """Build (ip_suffix, payload_position) -> ChannelInfo from CSV."""
    if df is None:
        df = load_csv()
    index: dict[tuple[int, int], ChannelInfo] = {}
    if df.empty:
        return index
    try:
        for _, row in df.iterrows():
            ci = row_to_channel_info(row)
            key = (ci.ip_suffix, ci.payload_position)
            if key in index:
                continue
            index[key] = ci
    except Exception as e:
        logger.warning(f"Failed to build CSV index: {e}")
    return index


def get_wiring_index() -> dict[tuple[int, int], ChannelInfo]:
    """Lazy-loaded CSV index, cached in function attribute."""
    if not hasattr(get_wiring_index, "_cache"):
        get_wiring_index._cache = build_csv_index()  # type: ignore[attr-defined]
    return get_wiring_index._cache  # type: ignore[attr-defined]


def get_channel_info(ip_suffix: int, channel: int) -> ChannelInfo | None:
    """Look up ChannelInfo for a 0-based channel of controller ip_suffix."""
    idx = get_wiring_index()
    if not idx:
        return None
    return idx.get((int(ip_suffix), channel + 1))


def channel_label(ip_suffix: int, ch: int) -> str:
    """Formatted channel label with needle info for multiselect."""
    info = get_channel_info(ip_suffix, ch)
    return f"{ch} | {info.short_info()}" if info else str(ch)


def build_groups(df: pd.DataFrame | None = None) -> dict[str, GroupDef]:
    """Group channels by CSV 组 column, keyed by controller IP suffix."""
    if df is None:
        df = load_csv()
    groups: dict[str, dict[int, list[ChannelInfo]]] = {}
    if df.empty:
        return {}
    try:
        for _, row in df.iterrows():
            ci = row_to_channel_info(row)
            groups.setdefault(ci.group, {}).setdefault(ci.ip_suffix, []).append(ci)
    except Exception as e:
        logger.warning(f"Failed to build CSV groups: {e}")
        return {}
    return {name: GroupDef(name, by_ip) for name, by_ip in groups.items()}


def build_all_units(df: pd.DataFrame | None = None) -> list[ChannelInfo]:
    """All physical units, deduped by (ip_suffix, payload_position), sorted
    by (group, ip_suffix, payload_position) — same ordering as the legacy UI."""
    if df is None:
        df = load_csv()
    if df.empty:
        return []
    seen: set[tuple[int, int]] = set()
    all_units: list[ChannelInfo] = []
    try:
        for _, row in df.iterrows():
            ci = row_to_channel_info(row)
            key = (ci.ip_suffix, ci.payload_position)
            if key in seen:
                continue
            seen.add(key)
            all_units.append(ci)
    except Exception as e:
        logger.warning(f"Failed to build CSV units: {e}")
        return []
    all_units.sort(key=lambda u: (u.group, u.ip_suffix, u.payload_position))
    return all_units


# =============================================================================
# Joint Control: Matrix <-> Flat Array Conversion
# =============================================================================


def jc_build_wiring_index(df: pd.DataFrame | None = None) -> dict[int, tuple[int, int]]:
    """physical_position (1-based) -> (ip_suffix, payload_position 1-based)."""
    if df is None:
        df = load_csv()
    pos_to_hw: dict[int, tuple[int, int]] = {}
    if df.empty:
        return pos_to_hw
    try:
        for _, row in df.iterrows():
            ip_s = int(row["IP组"])
            pp = int(row["序号"]) + 1
            r = int(row["36×36行"])
            c = int(row["36×36列"])
            pos = r * GRID_SIZE + c + 1
            pos_to_hw[pos] = (ip_s, pp)
    except Exception as e:
        logger.warning(f"Failed to build joint control CSV index: {e}")
    return pos_to_hw


def jc_build_ip_index(df: pd.DataFrame | None = None) -> dict[int, int]:
    """ip_suffix -> controller_index (sorted IP order, flat-array layout)."""
    if df is None:
        df = load_csv()
    if df.empty:
        return {}
    sorted_ips = sorted(df["IP组"].unique())
    return {int(ip): idx for idx, ip in enumerate(sorted_ips)}


def jc_matrix_to_flat(
    matrix: np.ndarray,
    pos_to_hw: dict[int, tuple[int, int]],
    ip_to_ctrl_idx: dict[int, int],
    dm_num: int,
) -> np.ndarray:
    """Convert a 36x36 matrix to the flat array expected by MicroDM.send_voltages().

    Flat array order: controller[0]->vs[0:50], controller[1]->vs[50:100], ...
    """
    flat = np.zeros(dm_num, dtype=np.float64)
    n_cols = matrix.shape[1] if matrix.ndim == 2 else GRID_SIZE
    for physical_pos, (ip_suffix, payload_pos) in pos_to_hw.items():
        row = (physical_pos - 1) // n_cols
        col = (physical_pos - 1) % n_cols
        voltage = matrix[row, col]
        ctrl_idx = ip_to_ctrl_idx.get(ip_suffix)
        if ctrl_idx is not None:
            flat_idx = ctrl_idx * SINGLE_CHANNELS + (payload_pos - 1)
            if flat_idx < dm_num:
                flat[flat_idx] = voltage
    return flat


# =============================================================================
# Session State & Feedback Helpers (lazy streamlit import)
# =============================================================================


def ensure_session_defaults(defaults: dict[str, Any]) -> None:
    """Set missing session_state keys (no-op for keys already present)."""
    import streamlit as st

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def set_feedback(key_prefix: str, msg: str, ftype: str = "success") -> None:
    """Persist a feedback message under '{key_prefix}_feedback'.

    ftype must be one of streamlit's message method names:
    success / info / warning / error.
    """
    import streamlit as st

    st.session_state[f"{key_prefix}_feedback"] = {"msg": msg, "type": ftype}


def show_feedback(key_prefix: str) -> None:
    """Render the persisted feedback message (idempotent, clears after show)."""
    import streamlit as st

    fb = st.session_state.get(f"{key_prefix}_feedback")
    if fb and fb.get("msg"):
        getattr(st, fb.get("type", "info"))(fb["msg"])


# =============================================================================
# Shared Channel Selector Widget
# =============================================================================


def render_channel_selector(
    key_prefix: str,
    ip_suffix: int | None = None,
    total: int = SINGLE_CHANNELS,
) -> ChannelSelection:
    """Shared channel selector: 全选 checkbox + 反选 button + multiselect.

    One state {all_mode, channels} under '{key_prefix}_all_mode' /
    '{key_prefix}_channels', shared by all send forms of this key_prefix.
    """
    import streamlit as st

    all_mode = st.checkbox(
        "全选",
        value=bool(st.session_state.get(f"{key_prefix}_all_mode", False)),
        key=f"{key_prefix}_all_mode",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("反选", key=f"{key_prefix}_invert_btn", use_container_width=True):
            sel = ChannelSelection(
                all_mode=all_mode,
                channels=list(st.session_state.get(f"{key_prefix}_channels", [])),
            )
            sel.invert(total)
            st.session_state[f"{key_prefix}_channels"] = sel.channels
    with col_info:
        n_selected = len(st.session_state.get(f"{key_prefix}_channels", []))
        st.caption(f"已选 {n_selected} / {total} 通道" + ("（全选模式）" if all_mode else ""))

    channels = st.multiselect(
        "选择通道",
        options=list(range(total)),
        default=list(st.session_state.get(f"{key_prefix}_channels", [])),
        format_func=(lambda ch: channel_label(int(ip_suffix), ch))
        if ip_suffix is not None
        else None,
        key=f"{key_prefix}_channels",
    )
    return ChannelSelection(all_mode=all_mode, channels=channels)
