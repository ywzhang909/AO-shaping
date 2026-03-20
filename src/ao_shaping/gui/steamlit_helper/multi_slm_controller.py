import streamlit as st
import numpy as np
from pathlib import Path
import sys
from loguru import logger

# Add the src directory to the path when running this file directly via Streamlit.
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.slm.slm_pattern_helper import PatternHelper, SLMPatternHelper

# Initialize SLM pattern helper
SLM_WIDTH = 1920
SLM_HEIGHT = 1080
pattern_helper = SLMPatternHelper()
gray_pattern_helper = PatternHelper((SLM_WIDTH, SLM_HEIGHT), bits=SantecSLM200.Gray_Scale_bits)


def _initialize_slm_state() -> None:
    if "slm1" not in st.session_state:
        st.session_state.slm1 = None
        st.session_state.slm1_connected = False
        st.session_state.slm1_wavelength = 1064
        st.session_state.slm1_video_mode = 0
        st.session_state.slm1_next_memory = 1
        st.session_state.slm1_phase_preview = None
        st.session_state.slm1_phase_source = "暂无"

    if "slm2" not in st.session_state:
        st.session_state.slm2 = None
        st.session_state.slm2_connected = False
        st.session_state.slm2_wavelength = 1064
        st.session_state.slm2_video_mode = 0
        st.session_state.slm2_next_memory = 1
        st.session_state.slm2_phase_preview = None
        st.session_state.slm2_phase_source = "暂无"


def _phase_to_preview(phase_gray: np.ndarray) -> np.ndarray:
    normalized = phase_gray.astype(np.float32) / max(SantecSLM200.MAX_GRAYSCALE_VALUE, 1)
    return np.clip(normalized, 0.0, 1.0)


def refresh_phase_preview(slm_num: int) -> None:
    phase_key = f"slm{slm_num}_phase_preview"
    source_key = f"slm{slm_num}_phase_source"
    slm = st.session_state.get(f"slm{slm_num}")
    if slm is None:
        st.session_state[phase_key] = None
        st.session_state[source_key] = "暂无"
        return

    phase_gray, source = slm.get_displayed_phase()
    st.session_state[phase_key] = None if phase_gray is None else _phase_to_preview(phase_gray)
    st.session_state[source_key] = source


def render_phase_preview(slm_num: int) -> None:
    preview = st.session_state.get(f"slm{slm_num}_phase_preview")
    source = st.session_state.get(f"slm{slm_num}_phase_source", "暂无")
    st.caption(f"当前显示来源: {source}")
    if preview is None:
        st.info("当前无法精确展示相位预览。只有通过本页面下发并缓存过的相位，才能保证预览与设备显示一致。")
        return

    st.image(
        preview,
        caption=f"SLM {slm_num} 当前显示相位预览",
        clamp=True,
        use_container_width=True,
    )


def render_pattern_controls(slm_num: int) -> tuple[str, dict[str, int | float | str]]:
    prefix = f"slm{slm_num}"
    pattern_type = st.selectbox(
        "选择相位图类型",
        options=[
            "平场",
            "线性光栅",
            "圆形光栅",
            "透镜",
            "全息光栅",
            "棋盘格",
            "二元光栅",
            "微透镜阵列",
            "湍流相位屏",
            "Zernike",
        ],
        key=f"{prefix}_pattern_type",
    )

    params: dict[str, int | float | str] = {}

    if pattern_type in {"线性光栅", "全息光栅"}:
        params["period"] = st.number_input(
            "周期 (像素)",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
            key=f"{prefix}_{pattern_type}_period",
        )
        params["phase_range"] = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
            value=float(2 * np.pi),
            key=f"{prefix}_{pattern_type}_phase_range",
        )
    elif pattern_type == "圆形光栅":
        params["radius"] = st.number_input(
            "圆形周期半径 (像素)",
            min_value=1.0,
            max_value=2000.0,
            value=300.0,
            step=10.0,
            key=f"{prefix}_circular_radius",
        )
        params["phase_range"] = st.number_input(
            "相位范围 (rad)",
            min_value=0.1,
            max_value=float(2 * np.pi),
            value=float(2 * np.pi),
            key=f"{prefix}_circular_phase_range",
        )
    elif pattern_type == "透镜":
        params["focal_length_mm"] = st.number_input(
            "焦距 (mm)",
            min_value=1.0,
            max_value=100000.0,
            value=1000.0,
            step=10.0,
            key=f"{prefix}_lens_focal_length",
        )
        params["pixel_size_um"] = st.number_input(
            "像素尺寸 (um)",
            min_value=0.1,
            max_value=100.0,
            value=8.0,
            step=0.1,
            key=f"{prefix}_lens_pixel_size",
        )
    elif pattern_type == "棋盘格":
        params["period"] = st.number_input(
            "棋盘格周期 (像素)",
            min_value=1,
            max_value=1000,
            value=100,
            step=1,
            key=f"{prefix}_checker_period",
        )
    elif pattern_type == "二元光栅":
        params["a"] = st.number_input(
            "亮条纹宽度 a (像素)",
            min_value=1,
            max_value=1000,
            value=2,
            step=1,
            key=f"{prefix}_binary_a",
        )
        params["b"] = st.number_input(
            "暗条纹宽度 b (像素)",
            min_value=1,
            max_value=1000,
            value=3,
            step=1,
            key=f"{prefix}_binary_b",
        )
        params["direction"] = st.selectbox(
            "方向",
            options=["horizontal", "vertical"],
            format_func=lambda x: "水平" if x == "horizontal" else "垂直",
            key=f"{prefix}_binary_direction",
        )
    elif pattern_type == "微透镜阵列":
        params["lens_size"] = st.number_input(
            "微透镜尺寸 (像素)",
            min_value=8,
            max_value=1000,
            value=200,
            step=1,
            key=f"{prefix}_microlens_size",
        )
        params["focal_length_mm"] = st.number_input(
            "焦距 (mm)",
            min_value=1.0,
            max_value=100000.0,
            value=100.0,
            step=1.0,
            key=f"{prefix}_microlens_focal_length",
        )
        params["pixel_size_um"] = st.number_input(
            "像素尺寸 (um)",
            min_value=0.1,
            max_value=100.0,
            value=8.0,
            step=0.1,
            key=f"{prefix}_microlens_pixel_size",
        )
    elif pattern_type == "湍流相位屏":
        params["Cn2"] = st.number_input(
            "Cn²",
            min_value=1e-18,
            max_value=1e-10,
            value=1e-14,
            format="%.1e",
            key=f"{prefix}_turbulence_cn2",
        )
        params["L"] = st.number_input(
            "传播距离 L (m)",
            min_value=0.1,
            max_value=1e6,
            value=1000.0,
            step=10.0,
            key=f"{prefix}_turbulence_length",
        )
        params["pixel_size_um"] = st.number_input(
            "像素尺寸 (um)",
            min_value=0.1,
            max_value=100.0,
            value=8.0,
            step=0.1,
            key=f"{prefix}_turbulence_pixel_size",
        )
    elif pattern_type == "Zernike":
        params["n"] = st.number_input(
            "径向阶数 n",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
            key=f"{prefix}_zernike_n",
        )
        params["m"] = st.number_input(
            "角向阶数 m",
            min_value=-20,
            max_value=20,
            value=0,
            step=1,
            key=f"{prefix}_zernike_m",
        )
        params["amplitude"] = st.number_input(
            "振幅 (单位: 波长)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key=f"{prefix}_zernike_amplitude",
        )
        params["radius"] = st.number_input(
            "孔径半径 (像素)",
            min_value=1,
            max_value=2000,
            value=min(SLM_WIDTH, SLM_HEIGHT) // 2,
            step=1,
            key=f"{prefix}_zernike_radius",
        )

    return pattern_type, params


def generate_phase_gray(
    slm: SantecSLM200,
    wavelength_nm: int,
    pattern_type: str,
    params: dict[str, int | float | str],
) -> np.ndarray:
    if pattern_type == "平场":
        phase_rad = np.zeros((SLM_HEIGHT, SLM_WIDTH))
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "线性光栅":
        phase_rad = pattern_helper.linear_grating(
            width=SLM_WIDTH,
            height=SLM_HEIGHT,
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "圆形光栅":
        phase_rad = pattern_helper.circular_grating(
            width=SLM_WIDTH,
            height=SLM_HEIGHT,
            radius=float(params["radius"]),
            phase_range=float(params["phase_range"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "透镜":
        phase_rad = pattern_helper.lens(
            width=SLM_WIDTH,
            height=SLM_HEIGHT,
            focal_length=float(params["focal_length_mm"]),
            wavelength=wavelength_nm,
            pixel_size=float(params["pixel_size_um"]) * 1e-6,
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "全息光栅":
        phase_rad = pattern_helper.hologram(
            width=SLM_WIDTH,
            height=SLM_HEIGHT,
            period=float(params["period"]),
            phase_range=float(params["phase_range"]),
        )
        return slm.create_phase_from_array(phase_rad)
    if pattern_type == "棋盘格":
        return gray_pattern_helper.generate_checkerboard(period=int(params["period"]))
    if pattern_type == "二元光栅":
        return gray_pattern_helper.generate_binary_grating(
            a=int(params["a"]),
            b=int(params["b"]),
            direction=str(params["direction"]),
        )
    if pattern_type == "微透镜阵列":
        return gray_pattern_helper.generate_microlens_array(
            lens_size=int(params["lens_size"]),
            focal_length=float(params["focal_length_mm"]) * 1e-3,
            wavelength=wavelength_nm * 1e-9,
            pixel_size=float(params["pixel_size_um"]) * 1e-6,
        )
    if pattern_type == "湍流相位屏":
        return gray_pattern_helper.generate_turbulence_screen(
            Cn2=float(params["Cn2"]),
            L=float(params["L"]),
            wavelength=wavelength_nm * 1e-9,
            pixel_size=float(params["pixel_size_um"]) * 1e-6,
        )
    if pattern_type == "Zernike":
        return gray_pattern_helper.generate_zernike(
            n=int(params["n"]),
            m=int(params["m"]),
            amplitude=float(params["amplitude"]),
            radius=float(params["radius"]),
        )
    raise ValueError(f"未知相位图类型: {pattern_type}")


def main():
    st.title("双SLM200控制器")
    _initialize_slm_state()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("SLM 1 设置")
        slm1_conn_button = st.button("连接 SLM 1", key="slm1_connect")
        slm1_disc_button = st.button("断开 SLM 1", key="slm1_disconnect")
        
        if slm1_conn_button:
            connect_slm(1)
        if slm1_disc_button:
            disconnect_slm(1)
        
        # Only show wavelength and mode settings if connected
        if st.session_state.slm1_connected:
            st.number_input(
                "波长 (nm)", 
                min_value=450, 
                max_value=1600, 
                value=st.session_state.slm1_wavelength,
                key="slm1_wavelength"
            )
            if st.button("设置波长", key="slm1_set_wl"):
                set_wavelength(1)
            
            st.selectbox(
                "视频模式",
                options=[0, 1],
                format_func=lambda x: "内存模式" if x == 0 else "DVI模式",
                index=st.session_state.slm1_video_mode,
                key="slm1_video_mode"
            )
            if st.button("设置模式", key="slm1_set_mode"):
                set_video_mode(1, st.session_state.slm1_video_mode)
        
        st.divider()
        
        st.header("SLM 2 设置")
        slm2_conn_button = st.button("连接 SLM 2", key="slm2_connect")
        slm2_disc_button = st.button("断开 SLM 2", key="slm2_disconnect")
        
        if slm2_conn_button:
            connect_slm(2)
        if slm2_disc_button:
            disconnect_slm(2)
        
        if st.session_state.slm2_connected:
            st.number_input(
                "波长 (nm)", 
                min_value=450, 
                max_value=1600, 
                value=st.session_state.slm2_wavelength,
                key="slm2_wavelength"
            )
            if st.button("设置波长", key="slm2_set_wl"):
                set_wavelength(2)
            
            st.selectbox(
                "视频模式",
                options=[0, 1],
                format_func=lambda x: "内存模式" if x == 0 else "DVI模式",
                index=st.session_state.slm2_video_mode,
                key="slm2_video_mode"
            )
            if st.button("设置模式", key="slm2_set_mode"):
                set_video_mode(2, st.session_state.slm2_video_mode)
    
    # Main area: Phase control for each SLM
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("SLM 1 相位控制")
        display_slm_status(1)
        if st.session_state.slm1_connected:
            if st.button("刷新当前显示相位", key="slm1_refresh_phase"):
                refresh_phase_preview(1)
            render_phase_preview(1)
            slm1_phase_control()
    
    with col2:
        st.header("SLM 2 相位控制")
        display_slm_status(2)
        if st.session_state.slm2_connected:
            if st.button("刷新当前显示相位", key="slm2_refresh_phase"):
                refresh_phase_preview(2)
            render_phase_preview(2)
            slm2_phase_control()

def connect_slm(slm_num: int):
    """Connect to the specified SLM"""
    try:
        if slm_num == 1:
            slm = SantecSLM200(slm_number=1, wavelength=st.session_state.slm1_wavelength, video_mode=st.session_state.slm1_video_mode)
            slm.open()
            st.session_state.slm1 = slm
            st.session_state.slm1_connected = True
            # Update wavelength and mode from the object (in case they changed during open)
            st.session_state.slm1_wavelength = slm.wavelength
            st.session_state.slm1_video_mode = slm.video_mode
            st.session_state.slm1_phase_preview = None
            st.session_state.slm1_phase_source = "连接成功，等待读取或下发相位"
            st.success(f"SLM {slm_num} 连接成功")
        else:
            slm = SantecSLM200(slm_number=2, wavelength=st.session_state.slm2_wavelength, video_mode=st.session_state.slm2_video_mode)
            slm.open()
            st.session_state.slm2 = slm
            st.session_state.slm2_connected = True
            st.session_state.slm2_wavelength = slm.wavelength
            st.session_state.slm2_video_mode = slm.video_mode
            st.session_state.slm2_phase_preview = None
            st.session_state.slm2_phase_source = "连接成功，等待读取或下发相位"
            st.success(f"SLM {slm_num} 连接成功")
    except Exception as e:
        st.error(f"SLM {slm_num} 连接失败: {e}")
        logger.error(f"Failed to connect SLM {slm_num}: {e}")

def disconnect_slm(slm_num: int):
    """Disconnect from the specified SLM"""
    try:
        if slm_num == 1 and st.session_state.slm1 is not None:
            st.session_state.slm1.close()
            st.session_state.slm1 = None
            st.session_state.slm1_connected = False
            st.session_state.slm1_phase_preview = None
            st.session_state.slm1_phase_source = "暂无"
            st.success(f"SLM {slm_num} 已断开")
        elif slm_num == 2 and st.session_state.slm2 is not None:
            st.session_state.slm2.close()
            st.session_state.slm2 = None
            st.session_state.slm2_connected = False
            st.session_state.slm2_phase_preview = None
            st.session_state.slm2_phase_source = "暂无"
            st.success(f"SLM {slm_num} 已断开")
    except Exception as e:
        st.error(f"SLM {slm_num} 断开失败: {e}")
        logger.error(f"Failed to disconnect SLM {slm_num}: {e}")

def set_wavelength(slm_num: int):
    """Set wavelength for the specified SLM"""
    try:
        if slm_num == 1 and st.session_state.slm1 is not None:
            st.session_state.slm1.set_wavelength(st.session_state.slm1_wavelength)
            st.success(f"SLM 1 波长设置为 {st.session_state.slm1_wavelength} nm")
        elif slm_num == 2 and st.session_state.slm2 is not None:
            st.session_state.slm2.set_wavelength(st.session_state.slm2_wavelength)
            st.success(f"SLM 2 波长设置为 {st.session_state.slm2_wavelength} nm")
    except Exception as e:
        st.error(f"设置波长失败: {e}")
        logger.error(f"Failed to set wavelength for SLM {slm_num}: {e}")

def set_video_mode(slm_num: int, mode: int):
    """Set video mode for the specified SLM"""
    try:
        if slm_num == 1 and st.session_state.slm1 is not None:
            st.session_state.slm1._set_memory_mode(mode)
            st.session_state.slm1_video_mode = mode
            st.success(f"SLM 1 模式设置为 {'内存模式' if mode == 0 else 'DVI模式'}")
        elif slm_num == 2 and st.session_state.slm2 is not None:
            st.session_state.slm2._set_memory_mode(mode)
            st.session_state.slm2_video_mode = mode
            st.success(f"SLM 2 模式设置为 {'内存模式' if mode == 0 else 'DVI模式'}")
    except Exception as e:
        st.error(f"设置模式失败: {e}")
        logger.error(f"Failed to set video mode for SLM {slm_num}: {e}")

def display_slm_status(slm_num: int):
    """Display the current status of the specified SLM"""
    if slm_num == 1:
        if st.session_state.slm1_connected and st.session_state.slm1 is not None:
            slm = st.session_state.slm1
            st.write(f"**状态**: 已连接")
            st.write(f"**波长**: {slm.wavelength} nm")
            st.write(f"**模式**: {'内存模式' if slm.video_mode == 0 else 'DVI模式'}")
            st.write(f"**下一个内存槽**: {st.session_state.slm1_next_memory}")
        else:
            st.write("**状态**: 未连接")
    else:
        if st.session_state.slm2_connected and st.session_state.slm2 is not None:
            slm = st.session_state.slm2
            st.write(f"**状态**: 已连接")
            st.write(f"**波长**: {slm.wavelength} nm")
            st.write(f"**模式**: {'内存模式' if slm.video_mode == 0 else 'DVI模式'}")
            st.write(f"**下一个内存槽**: {st.session_state.slm2_next_memory}")
        else:
            st.write("**状态**: 未连接")

def slm1_phase_control():
    """Phase control UI for SLM 1"""
    st.subheader("设置相位")
    pattern_type, params = render_pattern_controls(1)
    
    if st.button("从模式生成器生成相位", key="slm1_gen_pattern"):
        try:
            phase_gray = generate_phase_gray(
                st.session_state.slm1,
                st.session_state.slm1_wavelength,
                pattern_type,
                params,
            )
            
            # Write to next memory slot and immediately display
            mem_slot = st.session_state.slm1_next_memory
            st.session_state.slm1.write_phase(phase_gray, memory_number=mem_slot)
            st.session_state.slm1.display_memory(mem_slot)
            refresh_phase_preview(1)
            
            # Update next memory slot (avoid using same slot consecutively)
            st.session_state.slm1_next_memory = (st.session_state.slm1_next_memory % 128) + 1
            
            st.success(f"相位已写入内存槽 {mem_slot} 并显示")
        except Exception as e:
            st.error(f"生成或显示相位失败: {e}")
            logger.error(f"Failed to generate/display phase for SLM 1: {e}")
    
    # Option to load from CSV
    uploaded_file = st.file_uploader("上传CSV相位文件", type=["csv"], key="slm1_csv")
    if uploaded_file is not None:
        if st.button("从CSV加载相位", key="slm1_load_csv"):
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp_slm1_phase.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load phase from CSV
                phase_gray = st.session_state.slm1.load_phase_from_csv(temp_path)
                
                # Write to next memory slot and immediately display
                mem_slot = st.session_state.slm1_next_memory
                st.session_state.slm1.write_phase(phase_gray, memory_number=mem_slot)
                st.session_state.slm1.display_memory(mem_slot)
                refresh_phase_preview(1)
                
                # Update next memory slot
                st.session_state.slm1_next_memory = (st.session_state.slm1_next_memory % 128) + 1
                
                st.success(f"相位已从CSV加载到内存槽 {mem_slot} 并显示")
                
                # Clean up temp file
                temp_path.unlink()
            except Exception as e:
                st.error(f"加载CSV相位失败: {e}")
                logger.error(f"Failed to load CSV phase for SLM 1: {e}")

def slm2_phase_control():
    """Phase control UI for SLM 2"""
    st.subheader("设置相位")
    pattern_type, params = render_pattern_controls(2)
    
    if st.button("从模式生成器生成相位", key="slm2_gen_pattern"):
        try:
            phase_gray = generate_phase_gray(
                st.session_state.slm2,
                st.session_state.slm2_wavelength,
                pattern_type,
                params,
            )
            
            # Write to next memory slot and immediately display
            mem_slot = st.session_state.slm2_next_memory
            st.session_state.slm2.write_phase(phase_gray, memory_number=mem_slot)
            st.session_state.slm2.display_memory(mem_slot)
            refresh_phase_preview(2)
            
            # Update next memory slot (avoid using same slot consecutively)
            st.session_state.slm2_next_memory = (st.session_state.slm2_next_memory % 128) + 1
            
            st.success(f"相位已写入内存槽 {mem_slot} 并显示")
        except Exception as e:
            st.error(f"生成或显示相位失败: {e}")
            logger.error(f"Failed to generate/display phase for SLM 2: {e}")
    
    # Option to load from CSV
    uploaded_file = st.file_uploader("上传CSV相位文件", type=["csv"], key="slm2_csv")
    if uploaded_file is not None:
        if st.button("从CSV加载相位", key="slm2_load_csv"):
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp_slm2_phase.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load phase from CSV
                phase_gray = st.session_state.slm2.load_phase_from_csv(temp_path)
                
                # Write to next memory slot and immediately display
                mem_slot = st.session_state.slm2_next_memory
                st.session_state.slm2.write_phase(phase_gray, memory_number=mem_slot)
                st.session_state.slm2.display_memory(mem_slot)
                refresh_phase_preview(2)
                
                # Update next memory slot
                st.session_state.slm2_next_memory = (st.session_state.slm2_next_memory % 128) + 1
                
                st.success(f"相位已从CSV加载到内存槽 {mem_slot} 并显示")
                
                # Clean up temp file
                temp_path.unlink()
            except Exception as e:
                st.error(f"加载CSV相位失败: {e}")
                logger.error(f"Failed to load CSV phase for SLM 2: {e}")

if __name__ == "__main__":
    main()
