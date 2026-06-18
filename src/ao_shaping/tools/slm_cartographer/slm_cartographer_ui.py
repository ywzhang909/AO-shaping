"""SLM Hartmann Calibrator - Streamlit UI

Interactive GUI for SLM Hartmann-Shack calibration and dynamic aberration correction.
Based on the paper: "From Hartmann Spots to SLM Phase Maps"

Run:
    streamlit run src/ao_shaping/tools/slm_cartographer/slm_cartographer_ui.py

Modules:
    1. Center Cosine Pattern - Generate SLM phase patterns
    2. Hartmann Capture   - WFS spotfield measurements
    3. Fourier Reconstruction - Phase from spot displacements
    4. LUT Calibration    - Phase-grayscale lookup table
    5. Dynamic Compensation - Closed-loop aberration correction
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from loguru import logger

# Add project root for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Import calibration package
from ao_shaping.tools.slm_cartographer import (
    LUTCalibrationConfig,
    LUTCalibrationResult,
    PhaseGrayscaleLUT,
    CompensationConfig,
    CompensationResult,
    DynamicCompensator,
    CosinePatternConfig,
    generate_center_cosine_pattern,
    generate_traditional_gradient_pattern,
    FourierReconstructionConfig,
    FourierWavefrontReconstructor,
    HartmannCaptureConfig,
    HartmannMeasurement,
)
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.wfs.thorlab_wfs import ThorlabWFS, MlaRes


def _init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "cart_slm": None,
        "cart_slm_connected": False,
        "cart_wfs": None,
        "cart_wfs_connected": False,
        "cart_lut_result": None,
        "cart_compensation_result": None,
        "cart_lut": None,
        "cart_lut_file": None,
        "cart_storage_dir": "data/slm_cartographer",
        "cart_wavelength": 532,
        "cart_slm_number": 1,
        "cart_mla": "768",
        "cart_pupil_diameter": 3.0,
        "cart_cosine_radius": 40.0,
        "cart_n_averages": 10,
        "cart_grayscale_min": 0,
        "cart_grayscale_max": 1023,
        "cart_step": 16,
        "cart_running": False,
        "cart_progress": {"percent": 0, "message": "", "gs": None},
        "cart_pattern_preview": None,
        "cart_last_measurement": None,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _connect_slm() -> bool:
    """Connect to Santec SLM."""
    try:
        if (
            st.session_state.cart_slm_connected
            and st.session_state.cart_slm is not None
        ):
            try:
                st.session_state.cart_slm.close()
            except Exception:
                pass

        slm = SantecSLM200(
            slm_number=st.session_state.cart_slm_number,
            wavelength=st.session_state.cart_wavelength,
            video_mode=0,
        )
        slm.open()
        st.session_state.cart_slm = slm
        st.session_state.cart_slm_connected = True
        st.success(
            f"SLM #{st.session_state.cart_slm_number} connected "
            f"(wavelength={slm.wavelength}nm, "
            f"res={slm.Panel_Res})"
        )
        return True
    except Exception as e:
        st.error(f"SLM connection failed: {e}")
        logger.exception(f"SLM connection failed: {e}")
        return False


def _disconnect_slm() -> None:
    """Disconnect from SLM."""
    slm = st.session_state.cart_slm
    try:
        if slm is not None:
            slm.close()
    except Exception as e:
        st.error(f"SLM disconnect failed: {e}")
    finally:
        st.session_state.cart_slm = None
        st.session_state.cart_slm_connected = False
        st.success("SLM disconnected")


def _connect_wfs() -> bool:
    """Connect to Thorlabs WFS."""
    try:
        if (
            st.session_state.cart_wfs_connected
            and st.session_state.cart_wfs is not None
        ):
            try:
                st.session_state.cart_wfs.close()
            except Exception:
                pass

        wfs = ThorlabWFS(
            mla_index=MlaRes.from_str(st.session_state.cart_mla, default=MlaRes.Res768),
            exposure_time=0.0,
            pupil_diameter=st.session_state.cart_pupil_diameter,
        )
        wfs.open()
        st.session_state.cart_wfs = wfs
        st.session_state.cart_wfs_connected = True
        st.success(
            f"WFS connected: {wfs.device_name} "
            f"(S/N: {wfs.serial_num}, "
            f"spots: {wfs.num_spots_x}×{wfs.num_spots_y})"
        )
        return True
    except Exception as e:
        st.error(f"WFS connection failed: {e}")
        logger.exception(f"WFS connection failed: {e}")
        return False


def _disconnect_wfs() -> None:
    """Disconnect from WFS."""
    wfs = st.session_state.cart_wfs
    try:
        if wfs is not None:
            wfs.close()
    except Exception as e:
        st.error(f"WFS disconnect failed: {e}")
    finally:
        st.session_state.cart_wfs = None
        st.session_state.cart_wfs_connected = False
        st.success("WFS disconnected")


def _generate_preview() -> np.ndarray:
    """Generate and cache cosine pattern preview."""
    if st.session_state.cart_pattern_preview is None or st.session_state.get(
        "_pattern_dirty", True
    ):
        config = CosinePatternConfig(
            center_x=960,
            center_y=540,
            radius_pixels=st.session_state.cart_cosine_radius,
            max_phase_2pi=1.0,
        )
        pattern = generate_center_cosine_pattern(
            config=config,
            output_resolution=(1920, 1200),
        )
        st.session_state.cart_pattern_preview = pattern
        st.session_state._pattern_dirty = False
    return st.session_state.cart_pattern_preview


# ==================== Module 1: Pattern Preview ====================


def render_pattern_module() -> None:
    """Render the center cosine pattern preview module."""
    st.header("1. Phase Pattern Preview")
    st.markdown(
        "Generate and inspect the center cosine grayscale pattern used in calibration.\n"
        "This pattern reduces pixel crosstalk compared to traditional gradient patterns."
    )

    col1, col2 = st.columns(2)
    with col1:
        radius = st.slider(
            "Pattern Radius (pixels)",
            min_value=10,
            max_value=200,
            value=int(st.session_state.cart_cosine_radius),
            key="cart_cosine_radius_slider",
        )
        st.session_state.cart_cosine_radius = float(radius)

    with col2:
        show_gradient = st.checkbox("Show comparison: gradient pattern", value=False)

    st.session_state._pattern_dirty = True
    cos_pattern = _generate_preview()
    h, w = cos_pattern.shape

    # Display pattern as image
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.subheader("Center Cosine Pattern")
        st.image(
            cos_pattern,
            clamp=True,
            caption=f"Radius={radius}px, max_gs={cos_pattern.max()}",
            width=None,
        )
        st.caption(
            f"Shape: {cos_pattern.shape}, range: [{cos_pattern.min()}, {cos_pattern.max()}]"
        )

    with fig_col2:
        if show_gradient:
            grad_pattern = generate_traditional_gradient_pattern(
                output_resolution=(w, h),
                direction="x",
            )
            st.subheader("Radial Gradient Pattern (Comparison)")
            st.image(
                grad_pattern,
                clamp=True,
                caption=f"Gradient, R={radius}px",
                width=None,
            )
        else:
            st.subheader("Pattern profile at center row")
            center_row = cos_pattern[h // 2, :]
            profile_chart = {
                "x (pixel)": list(range(len(center_row))),
                "grayscale": center_row.tolist(),
            }
            st.line_chart(profile_chart, x="x (pixel)", y="grayscale")

    # Pattern statistics
    st.caption(
        f"Max at center: {int(cos_pattern[h // 2, w // 2])}, "
        f"sum={int(cos_pattern.sum())}, nonzero_pixels={int(np.count_nonzero(cos_pattern))}"
    )


# ==================== Module 2: LUT Calibration ====================


def _lut_calibration_worker(
    config: LUTCalibrationConfig,
    progress_queue: list[dict],
) -> LUTCalibrationResult | None:
    """Background worker for LUT calibration."""
    try:
        slm = st.session_state.cart_slm
        wfs = st.session_state.cart_wfs

        if slm is None or not slm.is_open:
            raise RuntimeError("SLM not connected")
        if wfs is None or not wfs.is_connected():
            raise RuntimeError("WFS not connected")

        def progress_cb(current: int, total: int, gs: int, msg: str) -> None:
            progress_queue.append(
                {
                    "percent": current / max(total, 1) * 100,
                    "message": msg,
                    "gs": gs,
                }
            )

        calibrator = PhaseGrayscaleLUT(
            slm=slm,
            wfs=wfs,
            config=config,
            storage_dir=str(
                Path(st.session_state.cart_storage_dir) / "lut_calibration"
            ),
        )
        result = calibrator.calibrate(progress_callback=progress_cb)
        return result

    except Exception as e:
        logger.exception(f"LUT calibration failed: {e}")
        progress_queue.append({"status": "error", "message": str(e)})
        return None


def render_lut_calibration_module() -> None:
    """Render the LUT calibration module."""
    st.header("2. Phase-Grayscale LUT Calibration")
    st.markdown(
        "Build a lookup table mapping SLM grayscale → measured phase.\n"
        "Uses the center cosine pattern to minimize pixel crosstalk in the response curve."
    )

    if (
        not st.session_state.cart_slm_connected
        or not st.session_state.cart_wfs_connected
    ):
        st.warning("Connect both SLM and WFS first (see sidebar).")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        gs_min = st.number_input(
            "Grayscale Min",
            0,
            1023,
            st.session_state.cart_grayscale_min,
            step=16,
            key="cart_gs_min",
        )
        st.session_state.cart_grayscale_min = gs_min
    with col2:
        gs_max = st.number_input(
            "Grayscale Max",
            0,
            1023,
            st.session_state.cart_grayscale_max,
            step=16,
            key="cart_gs_max",
        )
        st.session_state.cart_grayscale_max = gs_max
    with col3:
        step = st.number_input(
            "Step", 1, 100, st.session_state.cart_step, key="cart_step_input"
        )
        st.session_state.cart_step = step

    n_averages = st.number_input(
        "Frames to average per step",
        1,
        30,
        st.session_state.cart_n_averages,
        key="cart_n_avg",
    )
    st.session_state.cart_n_averages = n_averages

    progress_bar = st.progress(0, text="Ready")
    status_text = st.empty()

    if st.button(
        "Start LUT Calibration",
        type="primary",
        disabled=st.session_state.cart_running,
    ):
        st.session_state.cart_running = True
        st.session_state.cart_progress = {
            "percent": 0,
            "message": "Starting...",
            "gs": None,
        }

        config = LUTCalibrationConfig(
            slm_wavelength_nm=st.session_state.cart_wavelength,
            wavelength_nm=float(st.session_state.cart_wavelength),
            grayscale_range=(gs_min, gs_max),
            step=step,
            n_averages=n_averages,
            use_center_cosine=True,
            cosine_radius_px=st.session_state.cart_cosine_radius,
            output_dir=str(Path(st.session_state.cart_storage_dir) / "lut_calibration"),
            slm_number=st.session_state.cart_slm_number,
            mla_resolution=st.session_state.cart_mla,
        )

        progress_queue: list[dict] = []

        import threading

        thread = threading.Thread(
            target=_lut_calibration_worker,
            args=(config, progress_queue),
            daemon=True,
        )
        thread.start()
        st.rerun()

    if st.session_state.cart_running:
        if (
            progress_queue
            and "status" in progress_queue[-1]
            and progress_queue[-1]["status"] == "error"
        ):
            status_text.error(progress_queue[-1]["message"])
            st.session_state.cart_running = False
        else:
            latest = progress_queue[-1] if progress_queue else {}
            pct = latest.get("percent", 0)
            msg = latest.get("message", "Running...")
            progress_bar.progress(min(pct / 100.0, 1.0), text=msg)
            status_text.info(f"Current: gs={latest.get('gs', '?')}, {msg}")

        if thread and not thread.is_alive():
            st.session_state.cart_running = False
            progress_bar.progress(1.0, text="Complete!")
            st.rerun()
        else:
            time.sleep(0.3)
            st.rerun()

    if st.session_state.cart_lut_result is not None:
        result = st.session_state.cart_lut_result
        st.divider()
        st.subheader("Calibration Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Phase", f"{result.max_phase_2pi:.3f} × 2π")
        with col2:
            st.metric("Peak Grayscale", result.peak_grayscale)
        with col3:
            st.metric("Data Points", len(result.grayscale_values))

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            gs_vals = result.grayscale_values
            phases_2pi = result.measured_phases_2pi
            ax.plot(gs_vals, phases_2pi, "b-o", linewidth=2, label="Measured")
            if result.fit_coefficients:
                gs_fit = np.linspace(min(gs_vals), max(gs_vals), 200)
                poly = np.poly1d(result.fit_coefficients)
                phases_fit = poly(gs_fit) / (2.0 * np.pi)
                ax.plot(gs_fit, phases_fit, "r--", label="3rd-order fit")
            ax.axhline(y=result.max_phase_2pi, color="g", linestyle=":", label="Peak")
            ax.set_xlabel("Grayscale Value")
            ax.set_ylabel("Phase (×2π)")
            ax.set_title("SSLM Phase-Grayscale Response")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        except ImportError:
            st.warning("matplotlib not available")

        if st.button("Save LUT"):
            save_path = Path(st.session_state.cart_storage_dir) / "lut"
            save_path.mkdir(exist_ok=True)
            result.save(
                save_path
                / f"lut_{st.session_state.cart_wavelength}nm_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            st.session_state.cart_lut_file = str(save_path)
            st.success(f"LUT saved to {save_path}")


# ==================== Module 3: Dynamic Compensation ====================


def render_compensation_module() -> None:
    """Render the dynamic aberration compensation module."""
    st.header("3. Dynamic Aberration Compensation")
    st.markdown(
        "**Closed-loop compensation workflow:**\n"
        "1. Measure dynamic aberration with flat (zero-grayscale) SLM\n"
        "2. Invert the measured aberration → compensation phase\n"
        "3. Compute compensation grayscale via inverse LUT\n"
        "4. Apply to SLM and verify RMS reduction"
    )

    if (
        not st.session_state.cart_slm_connected
        or not st.session_state.cart_wfs_connected
    ):
        st.warning("Connect both SLM and WFS first.")
        return

    if st.session_state.cart_lut is None:
        st.warning("Complete LUT calibration first (Module 2).")
        return

    col1, col2 = st.columns(2)
    with col1:
        n_iters = st.number_input(
            "Correction iterations",
            1,
            10,
            1,
            key="cart_comp_iters",
        )
    with col2:
        converge_thresh = st.number_input(
            "Convergence threshold (waves RMS)",
            0.01,
            0.5,
            0.05,
            step=0.01,
            key="cart_conv_thresh",
        )

    comp_config = CompensationConfig(
        slm_wavelength_nm=st.session_state.cart_wavelength,
        n_averages=st.session_state.cart_n_averages,
        n_correction_iterations=n_iters,
        convergence_threshold_rms_waves=converge_thresh,
        pupil_diameter_mm=st.session_state.cart_pupil_diameter,
        mla_resolution=st.session_state.cart_mla,
        slm_number=st.session_state.cart_slm_number,
        cosine_radius_px=st.session_state.cart_cosine_radius,
    )

    if st.button("Run Compensation", type="primary"):
        st.info("Running dynamic aberration compensation...")

        slm = st.session_state.cart_slm
        wfs = st.session_state.cart_wfs

        compensator = DynamicCompensator(
            slm=slm,
            wfs=wfs,
            config=comp_config,
            lut=st.session_state.cart_lut,
            storage_dir=str(Path(st.session_state.cart_storage_dir) / "compensation"),
        )

        result = compensator.compensate_once(max_iterations=n_iters)
        st.session_state.cart_compensation_result = result

    if st.session_state.cart_compensation_result is not None:
        result = st.session_state.cart_compensation_result
        st.divider()
        st.subheader("Compensation Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Initial RMS",
                f"{result.initial_wavefront_rms:.4f} waves",
            )
        with col2:
            st.metric(
                "Final RMS",
                f"{result.final_wavefront_rms:.4f} waves",
            )
        with col3:
            improvement = result.initial_wavefront_rms - result.final_wavefront_rms
            st.metric(
                "RMS Improvement",
                f"{improvement:.4f} waves",
                delta=f"{-improvement / result.initial_wavefront_rms * 100:.1f}%"
                if result.initial_wavefront_rms
                else "N/A",
            )

        st.caption(
            f"Converged: {result.converged}, iterations: {result.iterations_used}"
        )

        try:
            import matplotlib.pyplot as plt

            if result.measured_aberration is not None:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                vmax = max(
                    abs(np.nanmin(result.measured_aberration)),
                    abs(np.nanmax(result.measured_aberration)),
                )
                im0 = axes[0].imshow(
                    result.measured_aberration,
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax,
                )
                axes[0].set_title("Initial Aberration")
                axes[0].axis("off")
                fig.colorbar(im0, ax=axes[0], fraction=0.02)

                if result.compensation_grayscale is not None:
                    im1 = axes[1].imshow(
                        result.compensation_grayscale,
                        cmap="gray",
                    )
                    axes[1].set_title("Compensation Grayscale")
                    axes[1].axis("off")
                    fig.colorbar(im1, ax=axes[1], fraction=0.02)

                plt.tight_layout()
                st.pyplot(fig)
        except ImportError:
            st.warning("matplotlib not available")

        if st.button("Save Compensation Result"):
            save_path = Path(st.session_state.cart_storage_dir) / "compensation"
            save_path.mkdir(parents=True, exist_ok=True)
            with open(
                save_path / f"compensation_{time.strftime('%Y%m%d_%H%M%S')}.json",
                "w",
                encoding="utf-8",
            ) as f:
                import json

                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            st.success(f"Saved to {save_path}")


# ==================== Sidebar ====================


def render_sidebar() -> None:
    """Render sidebar with device connections and settings."""
    with st.sidebar:
        st.header("SLM Hartmann Calibrator")

        st.subheader("Storage")
        st.session_state.cart_storage_dir = st.text_input(
            "Storage Directory",
            value=st.session_state.cart_storage_dir,
            key="cart_storage_input",
        )

        st.divider()
        st.subheader("Optics Parameters")
        st.session_state.cart_wavelength = st.number_input(
            "Wavelength (nm)",
            450,
            1600,
            st.session_state.cart_wavelength,
            key="cart_wavelength_input",
        )
        st.session_state.cart_pupil_diameter = st.number_input(
            "Pupil Diameter (mm)",
            0.5,
            10.0,
            st.session_state.cart_pupil_diameter,
            step=0.5,
            key="cart_pupil_input",
        )

        st.divider()
        st.subheader("SLM Connection")
        st.session_state.cart_slm_number = st.number_input(
            "SLM Number",
            1,
            8,
            st.session_state.cart_slm_number,
            key="cart_slm_number_input",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if not st.session_state.cart_slm_connected:
                if st.button("Connect SLM", type="primary", key="cart_conn_slm"):
                    _connect_slm()
            else:
                st.success("SLM connected")
        with col_b:
            if st.session_state.cart_slm_connected:
                if st.button("Disconnect", key="cart_disc_slm"):
                    _disconnect_slm()

        st.divider()
        st.subheader("WFS Connection")
        st.session_state.cart_mla = st.selectbox(
            "MLA Resolution",
            ["320", "512", "768", "1024", "1280"],
            index=2,
            key="cart_mla_input",
        )
        col_c, col_d = st.columns(2)
        with col_c:
            if not st.session_state.cart_wfs_connected:
                if st.button("Connect WFS", type="primary", key="cart_conn_wfs"):
                    _connect_wfs()
            else:
                st.success("WFS connected")
        with col_d:
            if st.session_state.cart_wfs_connected:
                if st.button("Disconnect", key="cart_disc_wfs"):
                    _disconnect_wfs()

        st.divider()
        st.subheader("Calibration Parameters")
        st.session_state.cart_cosine_radius = st.slider(
            "Cosine Pattern Radius (px)",
            min_value=10,
            max_value=200,
            value=int(st.session_state.cart_cosine_radius),
            key="cart_cosine_radius_sidebar",
        )
        st.session_state.cart_n_averages = st.slider(
            "Frames to Average",
            1,
            30,
            st.session_state.cart_n_averages,
            key="cart_n_avg_sidebar",
        )


# ==================== Main App ====================


def main() -> None:
    """Main entry point for SLM Hartmann Calibrator."""
    st.set_page_config(
        page_title="SLM Hartmann Calibrator",
        page_icon=":telescope:",
        layout="wide",
    )

    # Initialize Zustand-like state
    _init_session_state()

    # Sidebar with device connections
    render_sidebar()

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Phase Pattern",
            "LUT Calibration",
            "Compensation",
            "Data / Log",
        ]
    )

    with tab1:
        render_pattern_module()

    with tab2:
        render_lut_calibration_module()

    with tab3:
        render_compensation_module()

    with tab4:
        st.header("Calibration Data")
        storage = Path(st.session_state.cart_storage_dir)
        if storage.exists():
            json_files = sorted(storage.rglob("*.json"), reverse=True)[:20]
            if json_files:
                for f in json_files:
                    st.caption(f"📄 {f.relative_to(storage)}")
            else:
                st.info("No data files found yet.")
        else:
            st.info("Storage directory will be created on first run.")


if __name__ == "__main__":
    main()
