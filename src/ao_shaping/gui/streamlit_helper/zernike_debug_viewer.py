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
8. Visualize the calibrated response matrix (from .h5 file)

Usage:
    streamlit run src/ao_shaping/gui/streamlit_helper/zernike_debug_viewer.py
"""

from __future__ import annotations

import json
import sys
import json
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT

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

    if "zrm_result" not in st.session_state:
        st.session_state.zrm_result = None

    if "zrm_result_path" not in st.session_state:
        st.session_state.zrm_result_path = None


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


def _find_result_file(debug_dir: Path) -> Path | None:
    """Find the corresponding .h5 result file from debug directory.
    
    Args:
        debug_dir: Path to debug directory (e.g., data/zernike_response_matrix/debug_20260525_113803)
        
    Returns:
        Path to result .h5 file, or None if not found
    """
    # Debug dir is: {output_path}/debug_{timestamp}
    # So output_path is debug_dir.parent
    output_base = debug_dir.parent
    
    # Look for .h5 files in the output_base directory
    h5_files = list(output_base.glob("*.h5"))
    
    if not h5_files:
        # Also check for magnitude-suffixed files
        h5_files = list(output_base.glob("*_mag*.h5"))
        
    if not h5_files:
        return None
        
    # If multiple files, return the most recently modified one
    return max(h5_files, key=lambda p: p.stat().st_mtime)


def _load_result_file(result_path: Path):
    """Load ZernikeResponseMatrixResult from .h5 file.
    
    Args:
        result_path: Path to .h5 file
        
    Returns:
        ZernikeResponseMatrixResult object, or None if failed
    """
    try:
        # Try to import the necessary class
        sys.path.insert(0, str(SRC_ROOT))
        from ao_shaping.optimizer.wf.zernike_response_matrix import load_zernike_response_matrix
        
        result = load_zernike_response_matrix(result_path)
        return result
    except Exception as e:
        st.error(f"Failed to load result file {result_path}: {e}")
        return None


def _compute_response_matrix_from_debug(debug_data: dict, magnitude: float):
    """Compute response matrix from debug data.
    
    Args:
        debug_data: Dictionary containing debug data structure
        magnitude: Perturbation magnitude used (in wavelengths)
        
    Returns:
        ZernikeResponseMatrixResult object, or None if failed
    """
    try:
        # Import necessary classes and functions
        sys.path.insert(0, str(SRC_ROOT))
        from ao_shaping.optimizer.wf.zernike_response_matrix import ZernikeResponseMatrixResult
        from ao_shaping.utils.matrix_utils import calc_n_zernike_terms, compute_pinv, compute_lstsq
        import numpy as np
        from datetime import datetime
        
        modes = debug_data.get("modes", {})
        if not modes:
            st.error("No mode data found")
            return None
            
        # Determine the actual n_max from the data
        # We need to infer this from the coefficient arrays we find
        max_coeff_len = 0
        for mode_idx, mode_data in modes.items():
            for cycle_idx, cycle_data in mode_data.get("cycles", {}).items():
                for sign in ["plus", "minus"]:
                    sign_data = cycle_data.get("signs", {}).get(sign)
                    if not sign_data:
                        continue
                    samples = sign_data.get("samples", {})
                    for sample_idx, sample_file in samples.items():
                        zernike_file = sign_data["path"] / f"sample_{sample_idx:03d}_zernike_coeffs.npy"
                        if zernike_file.exists():
                            coeffs = np.load(zernike_file)
                            max_coeff_len = max(max_coeff_len, len(coeffs))
        
        if max_coeff_len == 0:
            st.error("No coefficient data found")
            return None
            
        # Estimate n_max from coefficient length
        # The coefficient array length corresponds to the highest Zernike term measured
        # For Noll ordering: term 0=piston, 1=tip, 2=tilt, etc.
        # If we have coefficients up to index L-1, then n_max = L-1
        n_max_measured = max_coeff_len - 1
        
        # But we need to account for excluded terms
        # From the debug data collection, it seems all terms are saved
        # Let's check what the actual excluded_piston/excluded_tip_tilt settings were
        # We can infer this from the first meta file we find
        
        excluded_piston = True   # Default assumption
        excluded_tip_tilt = False # Default assumption
        
        # Try to get actual settings from metadata
        for mode_idx, mode_data in modes.items():
            for cycle_idx, cycle_data in mode_data.get("cycles", {}).items():
                for sign in ["plus", "minus"]:
                    sign_data = cycle_data.get("signs", {}).get(sign)
                    if sign_data:
                        samples = sign_data.get("samples", {})
                        if samples:
                            first_sample = sorted(samples.keys())[0]
                            meta_file = sign_data["path"] / f"sample_{first_sample:03d}_meta.json"
                            if meta_file.exists():
                                import json
                                with open(meta_file) as f:
                                    meta = json.load(f)
                                # Meta doesn't directly contain exclusion info, 
                                # but we can check if coefficient 0 (piston) is always zero
                                # or look at the pattern
                                break
                        break
                break
            break
        
        # For now, let's determine n_slm_terms from the mode indices we have
        mode_indices = sorted(modes.keys())
        if not mode_indices:
            st.error("No valid mode indices found")
            return None
            
        max_mode_idx = max(mode_indices)
        # Assuming we have continuous mode indices starting from 0
        n_slm_terms = max_mode_idx + 1
        
        # For WFS terms, we need to know what order was used
        # Let's assume it's the same as what we measured in coefficients
        n_wfs_terms = max_coeff_len  # All measured terms
        
        # But we need to subtract excluded terms if any
        n_remove = (1 if excluded_piston else 0) + (2 if excluded_tip_tilt else 0)
        n_slm_terms_for_matrix = n_slm_terms - n_remove
        n_wfs_terms_for_matrix = n_wfs_terms - n_remove
        
        if n_slm_terms_for_matrix <= 0 or n_wfs_terms_for_matrix <= 0:
            st.error("Invalid number of terms after exclusions")
            return None
        
        # Initialize matrices
        response_matrix = np.zeros((n_wfs_terms_for_matrix, n_slm_terms_for_matrix), dtype=np.float64)
        variance_matrix = np.zeros((n_wfs_terms_for_matrix, n_slm_terms_for_matrix), dtype=np.float64)
        
        # Process each SLM mode (column of response matrix)
        for col_idx, slm_mode_idx in enumerate(mode_indices):
            if col_idx >= n_slm_terms_for_matrix:
                continue  # Skip if beyond expected terms
                
            mode_data = modes[slm_mode_idx]
            
            # Collect data from all cycles and samples for this SLM mode
            mode_responses = []  # List of response vectors (each is WFS coefficients)
            
            for cycle_idx, cycle_data in mode_data.get("cycles", {}).items():
                for sign in ["plus", "minus"]:
                    sign_data = cycle_data.get("signs", {}).get(sign)
                    if not sign_data:
                        continue
                        
                    samples = sign_data.get("samples", {})
                    for sample_idx, sample_file in samples.items():
                        # Load zernike coefficients for this sample (THIS IS THE MEASURED OUTPUT)
                        zernike_file = sign_data["path"] / f"sample_{sample_idx:03d}_zernike_coeffs.npy"
                        if zernike_file.exists():
                            coeffs = np.load(zernike_file)
                            mode_responses.append(coeffs)
            
            if mode_responses:
                # Convert to array and compute statistics
                responses_array = np.array(mode_responses)  # Shape: (n_measurements, n_coeffs)
                
                # Compute mean and variance across measurements for each WFS coefficient
                mean_response = np.mean(responses_array, axis=0)   # Shape: (n_coeffs,)
                variance_response = np.var(responses_array, axis=0) # Shape: (n_coeffs,)
                
                # Now we need to map this to our response matrix
                # The response_matrix[row, col] = response of WFS mode 'row' when SLM mode 'col' is excited
                
                # We need to exclude piston and tip/tilt from both rows and columns if they were excluded
                row_start = 0
                col_start = 0
                if excluded_piston:
                    row_start += 1
                    col_start += 1
                if excluded_tip_tilt:
                    row_start += 2
                    col_start += 2
                
                # Copy the relevant portion of mean_response to the response matrix
                # We take WFS coefficients [row_start:] and map to matrix rows [0:]
                # We take SLM mode col_idx and map to matrix column [col_idx - col_start]
                # But we need to be careful about bounds
                
                wfs_start_idx = row_start
                wfs_end_idx = min(wfs_start_idx + n_wfs_terms_for_matrix, len(mean_response))
                slm_end_idx = min(col_idx + 1, len(mode_indices))  # This is tricky
                
                # Actually, let's think differently:
                # For SLM mode index `slm_mode_idx`, we want to fill column `col_idx` in the matrix
                # The values come from mean_response[wfs_start_idx:wfs_start_idx + n_wfs_terms_for_matrix]
                
                n_to_copy = min(n_wfs_terms_for_matrix, len(mean_response) - wfs_start_idx)
                if n_to_copy > 0:
                    response_matrix[0:n_to_copy, col_idx] = mean_response[wfs_start_idx:wfs_start_idx + n_to_copy]
                    variance_matrix[0:n_to_copy, col_idx] = variance_response[wfs_start_idx:wfs_start_idx + n_to_copy]
        
        # Compute derived properties
        mean_variance = float(np.mean(variance_matrix)) if variance_matrix.size > 0 else 0.0
        max_variance = float(np.max(variance_matrix)) if variance_matrix.size > 0 else 0.0
        
        # Try to compute pseudo-inverse if we have enough data
        pinv_matrix = None
        lstsq_matrix = None
        if response_matrix.size > 0 and min(response_matrix.shape) > 0:
            try:
                pinv_matrix = np.linalg.pinv(response_matrix)
                # For consistency with the original code, we might want to use SVD-based pinv
                # But numpy's pinv is fine for now
                # Try to compute condition number
                if np.linalg.matrix_rank(response_matrix) == min(response_matrix.shape):
                    try:
                        # Condition number = ||A|| * ||A^+||
                        cond_num = np.linalg.norm(response_matrix, ord=2) * np.linalg.norm(pinv_matrix, ord=2)
                    except:
                        cond_num = None
                else:
                    cond_num = None  # Singular matrix
            except:
                pinv_matrix = None
                lstsq_matrix = None
        
        # Estimate n_averages and n_cycles from data
        n_averages_est = 1
        n_cycles_est = 1
        for mode_idx, mode_data in modes.items():
            cycle_count = len(mode_data.get("cycles", {}))
            if cycle_count > n_cycles_est:
                n_cycles_est = cycle_count
            for cycle_idx, cycle_data in mode_data.get("cycles", {}).items():
                sample_count = len(cycle_data.get("signs", {}).get("plus", {}).get("samples", {}))
                if sample_count > n_averages_est:
                    n_averages_est = sample_count
        
        # Create result object
        result = ZernikeResponseMatrixResult(
            matrix=response_matrix,
            variance_matrix=variance_matrix,
            n_max=n_max_measured,  # This is approximate
            magnitude=magnitude,
            wavelength_nm=1064,  # Default, could try to extract from somewhere
            n_averages=max(1, n_averages_est),
            n_cycles=max(1, n_cycles_est),
            timestamp=datetime.now().isoformat(),
            excluded_piston=excluded_piston,
            excluded_tip_tilt=excluded_tip_tilt,
            pinv_matrix=pinv_matrix,
            lstsq_matrix=lstsq_matrix,
        )
        
        # Try to set wavelength from SLM info if available
        # We could look for SLM info in metadata or file names
        
        return result
        
    except Exception as e:
        st.error(f"Failed to compute response matrix: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Zernike Debug Viewer")

        st.session_state.zrm_debug_dir = st.text_input(
            "Debug Data Directory",
            value=st.session_state.zrm_debug_dir or "",
            help="Path to debug data folder (e.g., data/zernike_response_matrix/debug_20260101_120000)",
        )

        # Magnitude input for response matrix computation
        st.session_state.zrm_magnitude = st.number_input(
            "Perturbation Magnitude (λ)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.01,
            help="Magnitude of Zernike perturbation used during calibration (in wavelengths)"
        )

        if st.session_state.zrm_debug_dir:
            debug_path = Path(st.session_state.zrm_debug_dir)
            if debug_path.exists():
                if st.button("Load Debug Data", type="primary"):
                    with st.spinner("Loading debug data..."):
                        st.session_state.zrm_debug_data = _load_debug_data(debug_path)
                        st.success(f"Loaded {len(st.session_state.zrm_debug_data.get('modes', {}))} modes")
                    
                    # Try to load corresponding result file
                    result_path = _find_result_file(debug_path)
                    if result_path:
                        with st.spinner("Loading calibration result..."):
                            result = _load_result_file(result_path)
                            if result:
                                st.session_state.zrm_result = result
                                st.session_state.zrm_result_path = result_path
                                st.success(f"Loaded result from: {result_path.name}")
                            else:
                                st.session_state.zrm_result = None
                                st.session_state.zrm_result_path = None
                    else:
                        st.warning("No corresponding .h5 result file found")
                        st.session_state.zrm_result = None
                        st.session_state.zrm_result_path = None
                        
                # Button to compute response matrix from debug data
                if st.session_state.zrm_debug_data is not None:
                    if st.button("Compute Response Matrix from Debug Data", type="secondary"):
                        with st.spinner("Computing response matrix from debug data..."):
                            result = _compute_response_matrix_from_debug(
                                st.session_state.zrm_debug_data,
                                st.session_state.zrm_magnitude
                            )
                            if result:
                                st.session_state.zrm_computed_result = result
                                st.success("Response matrix computed successfully!")
                            else:
                                st.session_state.zrm_computed_result = None
                                st.error("Failed to compute response matrix")
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
                    
                    st.subheader("Compare Across Samples")

                    if len(samples) > 1:
                        # Collect zernike coefficients from all samples
                        all_zernike_coeffs = []
                        sample_indices = sorted(samples.keys())
                        
                        for samp_idx in sample_indices:
                            samp_file = sign_data["path"] / f"sample_{samp_idx:03d}_zernike_coeffs.npy"
                            if samp_file.exists():
                                samp_coeffs = np.load(samp_file)
                                all_zernike_coeffs.append(samp_coeffs)
                        
                        if all_zernike_coeffs:
                            # Convert to numpy array for statistical calculations
                            all_zernike_coeffs = np.array(all_zernike_coeffs)  # shape: (n_samples, n_coeffs)
                            
                            # Calculate mean and standard deviation across samples
                            mean_coeffs = np.mean(all_zernike_coeffs, axis=0)
                            std_coeffs = np.std(all_zernike_coeffs, axis=0)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                name='Mean Coefficients',
                                x=zernike_labels,
                                y=mean_coeffs,
                                error_y=dict(type='data', array=std_coeffs, visible=True)
                            ))
                            fig.update_layout(
                                title=f"Zernike Coefficients Across {len(sample_indices)} Samples (Mean ± Std)",
                                xaxis_title="Zernike Mode",
                                yaxis_title="Coefficient (μm)",
                                height=350,
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.warning("No zernike coefficient files found for samples")
                    elif len(samples) == 1:
                        st.info("Only one sample available. Need at least 2 samples to compute statistics.")

                    # Add response matrix visualization if available
                    st.divider()
                    st.subheader("Response Matrix Visualization")
                    
                    # Check for loaded result file first, then computed result
                    result_to_show = None
                    result_source = ""
                    
                    if st.session_state.zrm_result is not None:
                        result_to_show = st.session_state.zrm_result
                        result_source = "Loaded from .h5 file"
                    elif hasattr(st.session_state, 'zrm_computed_result') and st.session_state.zrm_computed_result is not None:
                        result_to_show = st.session_state.zrm_computed_result
                        result_source = "Computed from debug data"
                    
                    if result_to_show is not None:
                        result = result_to_show
                        
                        # Show basic info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Matrix Shape", f"{result.matrix.shape}")
                        with col2:
                            st.metric("Mean Variance", f"{result.mean_variance:.6f}")
                        with col3:
                            if result.condition_number is not None:
                                st.metric("Condition Number", f"{result.condition_number:.2e}")
                            else:
                                st.metric("Condition Number", "N/A")
                        
                        st.caption(f"Source: {result_source}")
                        
                        # Response matrix heatmap
                        st.write("**Response Matrix** (SLM Zernike → WFS Zernike)")
                        fig_response = go.Figure(data=go.Heatmap(
                            z=result.matrix.T,  # Transpose for conventional display (input on x, output on y)
                            colorscale="RdBu_r",
                            colorbar=dict(title="Response")
                        ))
                        fig_response.update_layout(
                            title=f"Response Matrix (n_max={result.n_max})",
                            xaxis_title="SLM Zernike Mode Index",
                            yaxis_title="WFS Zernike Mode Index",
                            height=400,
                        )
                        fig_response.update_xaxes(tickangle=45)
                        fig_response.update_yaxes(tickangle=45)
                        st.plotly_chart(fig_response, width='stretch')
                        
                        # Variance matrix heatmap
                        st.write("**Variance Matrix** (Measurement Uncertainty)")
                        fig_variance = go.Figure(data=go.Heatmap(
                            z=result.variance_matrix.T,  # Transpose for consistent display
                            colorscale="YlOrRd",
                            colorbar=dict(title="Variance")
                        ))
                        fig_variance.update_layout(
                            title=f"Variance Matrix (mean={result.mean_variance:.6f})",
                            xaxis_title="SLM Zernike Mode Index",
                            yaxis_title="WFS Zernike Mode Index",
                            height=400,
                        )
                        fig_variance.update_xaxes(tickangle=45)
                        fig_variance.update_yaxes(tickangle=45)
                        st.plotly_chart(fig_variance, width='stretch')
                        
                        # Show which modes were excluded
                        excluded_info = []
                        if result.excluded_piston:
                            excluded_info.append("Piston (Z1)")
                        if result.excluded_tip_tilt:
                            excluded_info.append("Tip/Tilt (Z2, Z3)")
                        
                        if excluded_info:
                            st.info(f"Excluded modes: {', '.join(excluded_info)}")
                        else:
                            st.info("No modes excluded")
                            
                        # Show magnitude info
                        st.info(f"Perturbation magnitude used: {result.magnitude}λ")
                    else:
                        st.info("No calibration result available. Load debug data to load .h5 file or click 'Compute Response Matrix from Debug Data'.")
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