import streamlit as st
import numpy as np
from pathlib import Path
import sys
from loguru import logger

# Add the project root to the path to import drivers
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from src.ao_shaping.drivers.slm.slm_pattern_helper import SLMPatternHelper

# Initialize SLM pattern helper
pattern_helper = SLMPatternHelper()

def main():
    st.title("双SLM200控制器")
    
    # Initialize session state for SLMs
    if 'slm1' not in st.session_state:
        st.session_state.slm1 = None
        st.session_state.slm1_connected = False
        st.session_state.slm1_wavelength = 1064
        st.session_state.slm1_video_mode = 0  # 0: Memory, 1: DVI
        st.session_state.slm1_next_memory = 1  # Next memory slot to use for SLM1
    
    if 'slm2' not in st.session_state:
        st.session_state.slm2 = None
        st.session_state.slm2_connected = False
        st.session_state.slm2_wavelength = 1064
        st.session_state.slm2_video_mode = 0
        st.session_state.slm2_next_memory = 1
    
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
            st.session_state.slm1_wavelength = st.number_input(
                "波长 (nm)", 
                min_value=450, 
                max_value=1600, 
                value=st.session_state.slm1_wavelength,
                key="slm1_wavelength"
            )
            if st.button("设置波长", key="slm1_set_wl"):
                set_wavelength(1)
            
            slm1_video_mode = st.selectbox(
                "视频模式",
                options=[0, 1],
                format_func=lambda x: "内存模式" if x == 0 else "DVI模式",
                index=st.session_state.slm1_video_mode,
                key="slm1_video_mode"
            )
            if st.button("设置模式", key="slm1_set_mode"):
                set_video_mode(1, slm1_video_mode)
        
        st.divider()
        
        st.header("SLM 2 设置")
        slm2_conn_button = st.button("连接 SLM 2", key="slm2_connect")
        slm2_disc_button = st.button("断开 SLM 2", key="slm2_disconnect")
        
        if slm2_conn_button:
            connect_slm(2)
        if slm2_disc_button:
            disconnect_slm(2)
        
        if st.session_state.slm2_connected:
            st.session_state.slm2_wavelength = st.number_input(
                "波长 (nm)", 
                min_value=450, 
                max_value=1600, 
                value=st.session_state.slm2_wavelength,
                key="slm2_wavelength"
            )
            if st.button("设置波长", key="slm2_set_wl"):
                set_wavelength(2)
            
            slm2_video_mode = st.selectbox(
                "视频模式",
                options=[0, 1],
                format_func=lambda x: "内存模式" if x == 0 else "DVI模式",
                index=st.session_state.slm2_video_mode,
                key="slm2_video_mode"
            )
            if st.button("设置模式", key="slm2_set_mode"):
                set_video_mode(2, slm2_video_mode)
    
    # Main area: Phase control for each SLM
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("SLM 1 相位控制")
        display_slm_status(1)
        if st.session_state.slm1_connected:
            slm1_phase_control()
    
    with col2:
        st.header("SLM 2 相位控制")
        display_slm_status(2)
        if st.session_state.slm2_connected:
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
            st.success(f"SLM {slm_num} 连接成功")
        else:
            slm = SantecSLM200(slm_number=2, wavelength=st.session_state.slm2_wavelength, video_mode=st.session_state.slm2_video_mode)
            slm.open()
            st.session_state.slm2 = slm
            st.session_state.slm2_connected = True
            st.session_state.slm2_wavelength = slm.wavelength
            st.session_state.slm2_video_mode = slm.video_mode
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
            st.success(f"SLM {slm_num} 已断开")
        elif slm_num == 2 and st.session_state.slm2 is not None:
            st.session_state.slm2.close()
            st.session_state.slm2 = None
            st.session_state.slm2_connected = False
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
    
    # Option to use pattern helper
    pattern_type = st.selectbox(
        "选择相位图类型",
        options=["平场", "线性光栅", "圆形光栅", "透镜", "全息光栅"],
        key="slm1_pattern_type"
    )
    
    if st.button("从模式生成器生成相位", key="slm1_gen_pattern"):
        try:
            # Generate pattern using helper
            if pattern_type == "平场":
                phase_rad = np.zeros((1080, 1920))
            elif pattern_type == "线性光栅":
                phase_rad = pattern_helper.linear_grating(
                    width=1920, height=1080, period=50, phase_range=2*np.pi
                )
            elif pattern_type == "圆形光栅":
                phase_rad = pattern_helper.circular_grating(
                    width=1920, height=1080, radius=300, phase_range=2*np.pi
                )
            elif pattern_type == "透镜":
                phase_rad = pattern_helper.lens(
                    width=1920, height=1080, focal_length=1000, wavelength=st.session_state.slm1_wavelength
                )
            elif pattern_type == "全息光栅":
                phase_rad = pattern_helper.hologram(
                    width=1920, height=1080, period=50, phase_range=2*np.pi
                )
            
            # Convert to grayscale
            phase_gray = st.session_state.slm1.create_phase_from_array(phase_rad)
            
            # Write to next memory slot and immediately display
            mem_slot = st.session_state.slm1_next_memory
            st.session_state.slm1.write_phase(phase_gray, memory_number=mem_slot)
            st.session_state.slm1.display_memory(mem_slot)
            
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
    
    # Option to use pattern helper
    pattern_type = st.selectbox(
        "选择相位图类型",
        options=["平场", "线性光栅", "圆形光栅", "透镜", "全息光栅"],
        key="slm2_pattern_type"
    )
    
    if st.button("从模式生成器生成相位", key="slm2_gen_pattern"):
        try:
            # Generate pattern using helper
            if pattern_type == "平场":
                phase_rad = np.zeros((1080, 1920))
            elif pattern_type == "线性光栅":
                phase_rad = pattern_helper.linear_grating(
                    width=1920, height=1080, period=50, phase_range=2*np.pi
                )
            elif pattern_type == "圆形光栅":
                phase_rad = pattern_helper.circular_grating(
                    width=1920, height=1080, radius=300, phase_range=2*np.pi
                )
            elif pattern_type == "透镜":
                phase_rad = pattern_helper.lens(
                    width=1920, height=1080, focal_length=1000, wavelength=st.session_state.slm2_wavelength
                )
            elif pattern_type == "全息光栅":
                phase_rad = pattern_helper.hologram(
                    width=1920, height=1080, period=50, phase_range=2*np.pi
                )
            
            # Convert to grayscale
            phase_gray = st.session_state.slm2.create_phase_from_array(phase_rad)
            
            # Write to next memory slot and immediately display
            mem_slot = st.session_state.slm2_next_memory
            st.session_state.slm2.write_phase(phase_gray, memory_number=mem_slot)
            st.session_state.slm2.display_memory(mem_slot)
            
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