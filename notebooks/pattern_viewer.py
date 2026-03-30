"""Phase Pattern Viewer for AO-Shaping

Interactive visualization of phase patterns from data/pattern/ directory.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Page config
st.set_page_config(
    page_title="Phase Pattern Viewer",
    page_icon="📊",
    layout="wide",
)

# Constants
PATTERN_DIR = Path("data/pattern")


def load_pattern(filepath: Path) -> tuple[np.ndarray, str]:
    """Load CSV pattern and return data array with bit depth info."""
    df = pd.read_csv(filepath, index_col=0)
    data = df.values.astype(float)

    # Detect bit depth
    max_val = data.max()
    if max_val > 255:
        bit_depth = "10-bit"
    else:
        bit_depth = "8-bit"

    return data, bit_depth


def get_pattern_files() -> list[str]:
    """Get all CSV files in pattern directory."""
    pattern_files = sorted(glob.glob(str(PATTERN_DIR / "*.csv")))
    return [Path(f).name for f in pattern_files]


def main() -> None:
    st.title("🔬 Phase Pattern Viewer")
    st.caption("AO-Shaping - Adaptive Optics Beam Shaping")

    # Sidebar: File selection
    with st.sidebar:
        st.header("📁 Pattern Selection")

        pattern_files = get_pattern_files()

        if not pattern_files:
            st.error("No CSV files found in data/pattern/")
            return

        selected_file = st.selectbox(
            "Select a pattern:",
            options=pattern_files,
            index=0,
        )

        st.divider()
        st.caption(f"Total patterns: {len(pattern_files)}")

    if not selected_file:
        st.info("👈 Select a pattern from the sidebar")
        return

    # Load and display pattern
    filepath = PATTERN_DIR / selected_file

    try:
        data, bit_depth = load_pattern(filepath)
    except Exception as e:
        st.error(f"Failed to load pattern: {e}")
        return

    # Stats row
    st.subheader(f"📊 {selected_file}")

    col1, col2, col3, col4, col5 = st.columns(5)

    stats = {
        "Shape": f"{data.shape[0]} × {data.shape[1]}",
        "Min": f"{data.min():.1f}",
        "Max": f"{data.max():.1f}",
        "Mean": f"{data.mean():.1f}",
        "Std": f"{data.std():.1f}",
    }

    for col, (label, value) in zip([col1, col2, col3, col4, col5], stats.items()):
        col.metric(label=label, value=value)

    st.caption(f"Bit depth: {bit_depth}")

    # Plotly heatmap
    st.divider()
    st.subheader("🗺️ Phase Distribution")

    # Determine color range based on bit depth
    if bit_depth == "10-bit":
        zmin, zmax = 0, 1023
        colorscale = "Viridis"
    else:
        zmin, zmax = 0, 255
        colorscale = "Viridis"

    fig = px.imshow(
        data,
        color_continuous_scale=colorscale,
        zmin=zmin,
        zmax=zmax,
        aspect="auto",
        origin="lower",
        labels=dict(x="X (pixel)", y="Y (pixel)", color="Phase"),
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        colorbar=dict(
            title="Phase",
            tickvals=[zmin, zmax // 4, zmax // 2, 3 * zmax // 4, zmax],
            ticktext=[
                f"{zmin}",
                f"{zmax // 4}",
                f"{zmax // 2}",
                f"{3 * zmax // 4}",
                f"{zmax}",
            ],
        ),
    )

    fig.update_xaxes(showgrid=False, ticks="")
    fig.update_yaxes(showgrid=False, ticks="")

    st.plotly_chart(fig, use_container_width=True)

    # Additional info
    with st.expander("📋 Pattern Information"):
        st.write(f"**File:** `{selected_file}`")
        st.write(f"**Path:** `{filepath.absolute()}`")
        st.write(f"**Dimensions:** {data.shape[0]} rows × {data.shape[1]} columns")
        st.write(f"**Data type:** {data.dtype}")
        st.write(f"**Memory:** {data.nbytes / 1024:.1f} KB")


if __name__ == "__main__":
    main()
