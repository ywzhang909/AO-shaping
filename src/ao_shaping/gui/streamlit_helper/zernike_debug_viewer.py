"""
Zernike Response Matrix Debug Data Viewer (Streamlit)

Features:
1. Load debug data saved during zernike-matrix calibration with --debug flag
2. Browse by SLM Zernike mode (mode_index)
3. View SLM phase image (with shift applied, normalized to gray)
4. Browse by cycle and sample (plus/minus perturbation)
5. Visualize WFS deviation (x, y) heatmaps
6. Visualize raw WFS Zernike coefficients (before averaging)
7. Compare across cycles/samples for same mode

Usage:
    streamlit run src/ao_shaping/gui/streamlit_helper/zernike_debug_viewer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _get_zernike_name(n_max: int) -> dict[int, str]:
    names = {
        0: "Piston (Z1)",
        1: "Tip (Z2)",
        2: "Tilt (Z3)",
        3: "Defocus (Z4)",
        4: "Astig 45° (Z5)",
        5: "Astig 0° (Z6)",
        6: "Coma Y (Z7)",
        7: "Coma X (Z8)",
        8: "Trefoil Y (Z9)",
        9: "Trefoil X (Z10)",
        10: "Spherical (Z11)",
    }
    return {i: names.get(i, f"Z{i+1}") for i in range(n_max)}


def _initialize_state() -> None:
    if "zrm_debug_dir" not in st.session_state:
        st.session_state.zrm_debug_dir = None

    if "zrm_debug_data" not in st.session_state:
        st.session_state.zrm_debug_data = None


def _load_debug_data(debug_dir: Path) -> dict:
    data = {
        "modes": {},
        "available_cycles": set(),
        "available_signs": set(),
        "shift_x": 0,
        "shift_y": 0,
    }

    mode_dirs = sorted(debug_dir.glob("mode_*"))
    for mode_dir in mode_dirs:
        mode_name = mode_dir.name
        try:
            mode_idx = int(mode_name.replace("mode_", ""))
        except ValueError:
            continue

        data["modes"][mode_idx] = {
            "path": mode_dir,
            "cycles": {},
        }

        cycle_dirs = sorted(mode_dir.glob("cycle_*"))
        for cycle_dir in cycle_dirs:
            try:
                cycle_idx = int(cycle_dir.name.replace("cycle_", ""))
            except ValueError:
                continue

            data["available_cycles"].add(cycle_idx)
            data["modes"][mode_idx]["cycles"][cycle_idx] = {
                "path": cycle_dir,
                "signs": {},
            }

            for sign_dir in cycle_dir.iterdir():
                if sign_dir.is_dir() and sign_dir.name in ["plus", "minus"]:
                    sign = sign_dir.name
                    data["available_signs"].add(sign)

                    samples = {}
                    for sample_file in sign_dir.glob("sample_*.npy"):
                        sample_name = sample_file.stem
                        try:
                            sample_idx = int(sample_name.split("_")[1])
                            samples[sample_idx] = sample_file
                        except (ValueError, IndexError):
                            continue

                    data["modes"][mode_idx]["cycles"][cycle_idx]["signs"][sign] = {
                        "path": sign_dir,
                        "samples": samples,
                    }

                    meta_file = sign_dir / "sample_000_meta.json"
                    if meta_file.exists():
                        with open(meta_file) as f:
                            meta = json.load(f)
                            data["shift_x"] = meta.get("shift_x", 0)
                            data["shift_y"] = meta.get("shift_y", 0)

    data["available_cycles"] = sorted(data["available_cycles"])
    data["available_signs"] = sorted(data["available_signs"])

    return data


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Zernike Debug Viewer")

        st.session_state.zrm_debug_dir = st.text_input(
            "Debug Data Directory",
            value=st.session_state.zrm_debug_dir or "",
            help="Path to debug data folder (e.g., data/zernike_response_matrix/debug_20260101_120000)",
        )

        if st.session_state.zrm_debug_dir:
            debug_path = Path(st.session_state.zrm_debug_dir)
            if debug_path.exists():
                if st.button("Load Debug Data", type="primary"):
                    st.session_state.zrm_debug_data = _load_debug_data(debug_path)
                    st.success(f"Loaded {len(st.session_state.zrm_debug_data.get('modes', {}))} modes")
            else:
                st.error(f"Directory not found: {debug_path}")


def _render_main() -> None:
    st.title("Zernike Response Matrix Debug Viewer")

    if st.session_state.zrm_debug_data is None:
        st.info("Please enter the debug data directory in the sidebar and click 'Load Debug Data'")
        return

    data = st.session_state.zrm_debug_data

    if not data.get("modes"):
        st.warning("No mode data found in the debug directory")
        return

    available_modes = sorted(data["modes"].keys())
    n_modes = len(available_modes)

    st.divider()
    st.subheader("Mode Selection")
    mode_idx = st.selectbox(
        "Select SLM Zernike Mode",
        options=available_modes,
        format_func=lambda x: f"Mode {x} (Z{x+1})",
    )

    zernike_names = _get_zernike_name(12)
    st.caption(f"**{zernike_names.get(mode_idx, f'Z{mode_idx+1}')}**")

    st.divider()
    st.subheader("Cycle & Sample Selection")

    col1, col2 = st.columns(2)
    with col1:
        if data["available_cycles"]:
            cycle_idx = st.selectbox(
                "Select Cycle",
                options=data["available_cycles"],
                format_func=lambda x: f"Cycle {x}",
            )
        else:
            cycle_idx = None
            st.warning("No cycle data")

    with col2:
        if data["available_signs"]:
            sign = st.selectbox(
                "Select Perturbation",
                options=data["available_signs"],
                format_func=lambda x: "Positive (+)" if x == "plus" else "Negative (-)",
            )
        else:
            sign = None
            st.warning("No sign data")

    if mode_idx in data["modes"] and cycle_idx is not None and sign is not None:
        mode_data = data["modes"][mode_idx]
        if cycle_idx in mode_data["cycles"]:
            cycle_data = mode_data["cycles"][cycle_idx]
            if sign in cycle_data["signs"]:
                sign_data = cycle_data["signs"][sign]
                samples = sign_data["samples"]

                if samples:
                    sample_idx = st.selectbox(
                        "Select Sample",
                        options=sorted(samples.keys()),
                        format_func=lambda x: f"Sample {x}",
                    )

                    st.divider()
                    st.subheader(f"Mode {mode_idx} - Cycle {cycle_idx} - {sign} - Sample {sample_idx}")

                    sample_dir = sign_data["path"]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Shift X", data["shift_x"])
                    with col2:
                        st.metric("Shift Y", data["shift_y"])

                    st.divider()
                    st.subheader("SLM Phase")

                    phase_file = sample_dir / f"sample_{sample_idx:03d}_slm_phase.npy"
                    if phase_file.exists():
                        phase = np.load(phase_file)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Shape", f"{phase.shape}")
                        with col2:
                            st.metric("Min", f"{phase.min()}")
                        with col3:
                            st.metric("Max", f"{phase.max()}")

                        fig = px.imshow(
                            phase,
                            color_continuous_scale="gray",
                            aspect="equal",
                            title=f"SLM Phase (Mode {mode_idx}, {sign})",
                            labels={"x": "X (pixels)", "y": "Y (pixels)", "color": "Gray Value (0-1023)"},
                        )
                        fig.update_layout(coloraxis_colorbar=dict(title="Gray Value (0-1023)"))
                        fig.update_yaxes(scaleanchor="x", scaleratio=1)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Phase file not found: {phase_file}")

                    st.divider()
                    st.subheader("WFS Deviation")

                    dev_x_file = sample_dir / f"sample_{sample_idx:03d}_deviation_x.npy"
                    dev_y_file = sample_dir / f"sample_{sample_idx:03d}_deviation_y.npy"

                    if dev_x_file.exists() and dev_y_file.exists():
                        dev_x = np.load(dev_x_file)
                        dev_y = np.load(dev_y_file)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Deviation X Shape", f"{dev_x.shape}")
                            st.metric("Deviation X Range", f"{dev_x.min():.3f} ~ {dev_x.max():.3f}")
                        with col2:
                            st.metric("Deviation Y Shape", f"{dev_y.shape}")
                            st.metric("Deviation Y Range", f"{dev_y.min():.3f} ~ {dev_y.max():.3f}")

                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Deviation X", "Deviation Y"),
                            shared_yaxes=True,
                        )
                        fig.add_trace(
                            go.Heatmap(z=dev_x, colorscale="RdBu_r", colorbar=dict(title="deviation (μm)")),
                            row=1, col=1,
                        )
                        fig.add_trace(
                            go.Heatmap(z=dev_y, colorscale="RdBu_r", colorbar=dict(title="deviation (μm)")),
                            row=1, col=2,
                        )
                        fig.update_xaxes(title_text="X", row=1, col=1)
                        fig.update_yaxes(title_text="Y", row=1, col=1)
                        fig.update_xaxes(title_text="X", row=1, col=2)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Deviation data not available")

                    st.divider()
                    st.subheader("WFS Zernike Coefficients")

                    zernike_file = sample_dir / f"sample_{sample_idx:03d}_zernike_coeffs.npy"
                    if zernike_file.exists():
                        zernike_coeffs = np.load(zernike_file)

                        st.metric("Coefficients Shape", f"{zernike_coeffs.shape}")

                        zernike_labels = [zernike_names.get(i, f"Z{i+1}") for i in range(len(zernike_coeffs))]
                        fig = go.Figure(data=go.Bar(x=zernike_labels, y=zernike_coeffs))
                        fig.update_layout(
                            title="Raw Zernike Coefficients (before averaging)",
                            xaxis_title="Zernike Mode",
                            yaxis_title="Coefficient (μm)",
                            height=350,
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Zernike coefficients not available")

                    st.divider()
                    st.subheader("Compare Across Samples")

                    if len(samples) > 1:
                        compare_sample = st.selectbox(
                            "Select sample to compare",
                            options=sorted(samples.keys()),
                            index=1 if len(samples) > 1 else 0,
                        )

                        comp_dir = sign_data["path"]
                        comp_zernike_file = comp_dir / f"sample_{compare_sample:03d}_zernike_coeffs.npy"

                        if comp_zernike_file.exists():
                            comp_zernike = np.load(comp_zernike_file)

                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                name=f"Sample {sample_idx}",
                                x=zernike_labels,
                                y=zernike_coeffs,
                            ))
                            fig.add_trace(go.Bar(
                                name=f"Sample {compare_sample}",
                                x=zernike_labels,
                                y=comp_zernike,
                            ))
                            fig.update_layout(
                                title=f"Compare Sample {sample_idx} vs Sample {compare_sample}",
                                xaxis_title="Zernike Mode",
                                yaxis_title="Coefficient (μm)",
                                barmode="group",
                                height=350,
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, width='stretch')
                else:
                    st.text("Samples Not found")

def main():
    st.set_page_config(
        page_title="Zernike Debug Viewer",
        page_icon="🔬",
        layout="wide",
    )

    _initialize_state()
    _render_sidebar()
    _render_main()


if __name__ == "__main__":
    main()