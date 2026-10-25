"""Shared ceramic-cell data layer for the dm and r50 viewers.

Both viewers are the same Streamlit application over the 1300-cell
ceramic corpus: they load the same enriched CSV, look a cell up by
``[row, col]``, detect ``(IP, seq)`` conflicts on save, and write the
frame back. Those data operations existed twice -- byte-identical
bodies with formatting drift between the copies -- so they live here
once.

The viewer-specific parts stay in the viewers: ``Config`` (each adds
its own constants), ``ConflictInfo`` (different field sets), the
session-state wiring, and every hardware-control section.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

__all__ = [
    "CellInfo",
    "check_conflict",
    "load_enriched_data",
    "write_summary_csv",
]


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
    def from_df(
        cls, df: pd.DataFrame, row: int, col: int, grid_size: int
    ) -> CellInfo | None:
        """从 DataFrame 按 [row, col] 查找并构造 CellInfo。

        ``grid_size`` is the viewer's ``Config.GRID_SIZE`` -- the two
        viewers' ``Config`` classes differ in other constants, so the
        grid is a parameter rather than an import.
        """
        cell = df[(df["36×36行"] == row) & (df["36×36列"] == col)]
        if cell.empty:
            return None
        c = cell.iloc[0]
        pos = row * grid_size + col + 1
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


@st.cache_data
def load_enriched_data(csv_path: str) -> pd.DataFrame:
    """加载已处理的 1300 陶瓷单元数据。"""
    path = Path(csv_path)
    if not path.exists():
        st.error(f"数据文件不存在: {path}")
        st.info("请先运行 scripts/dm/process_1300_data.py 生成数据文件")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def check_conflict(
    df: pd.DataFrame, row: int, col: int, ip_val: int, seq_val: int
) -> pd.DataFrame:
    """Return rows where (ip_val, seq_val) already exists at a different position."""
    return df[
        (df["IP组"] == ip_val)
        & (df["序号"] == seq_val)
        & ~((df["36×36行"] == row) & (df["36×36列"] == col))
    ]


def write_summary_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path``; parent directories are created.

    Session state, dirty flags and Streamlit messages stay with the
    caller -- this is only the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
