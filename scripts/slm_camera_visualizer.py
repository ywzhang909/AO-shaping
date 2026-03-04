"""
SLM + Camera 实时可视化测试工具

功能:
- 手动设置单个或多个相位图案 (参数生成或加载CSV)
- 实时显示相机画面
- 相机控制: 曝光时间、自动曝光开关

使用方法:
    streamlit run scripts/slm_camera_visualizer.py

要求:
    - SLM 硬件 (Santec SLM-200)
    - 相机硬件 (MIICAM 4100 或 Daheng)
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import time

# 导入驱动
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.ccd.miicam import CameraStreamManager as MIICAMCamera
from ao_shaping.drivers.ccd.daheng import CameraStreamManager as DahengCamera


# ============== 相位图案生成函数 ==============

SLM_RESOLUTION = (1920, 1200)  # SLM 分辨率
SLM_BITS = 10
SLM_MAX_VAL = 2**SLM_BITS - 1  # 1023


def generate_blazed_grating(
    period: int = 20, direction: str = "horizontal"
) -> np.ndarray:
    """生成闪耀光栅"""
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL

    if direction == "horizontal":
        y = np.arange(height)
        grating = (y % period) / period * max_val
        img = np.tile(grating[:, np.newaxis], (1, width))
    else:
        x = np.arange(width)
        grating = (x % period) / period * max_val
        img = np.tile(grating[np.newaxis, :], (height, 1))

    return img.astype(np.uint16)


def generate_focus(
    focal_length: float = 0.5, wavelength: float = 532e-9, pixel_size: float = 8e-6
) -> np.ndarray:
    """生成聚焦相位 (抛物面)"""
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL

    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2

    phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    return img


def generate_checkerboard(period: int = 100) -> np.ndarray:
    """生成棋盘格"""
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL

    y = np.arange(height) // period
    x = np.arange(width) // period
    X, Y = np.meshgrid(x, y)

    checker = (X + Y) % 2
    img = (checker * max_val).astype(np.uint16)

    return img


def generate_binary_grating(
    period: int = 8, direction: str = "horizontal"
) -> np.ndarray:
    """生成二元光栅 (01光栅)"""
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL // 2

    if direction == "horizontal":
        y = np.arange(height)
        grating = np.where(y % period < period // 2, 0, max_val)
        img = np.tile(grating[:, np.newaxis], (1, width))
    else:
        x = np.arange(width)
        grating = np.where(x % period < period // 2, 0, max_val)
        img = np.tile(grating[np.newaxis, :], (height, 1))

    return img.astype(np.uint16)


def generate_vortex(topological_charge: int = 1) -> np.ndarray:
    """生成涡旋光束相位"""
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL

    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)

    theta = np.arctan2(Y, X)
    phase = topological_charge * theta
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    return img


def generate_zernike(n: int = 4, m: int = 0, amplitude: float = 2.0) -> np.ndarray:
    """生成Zernike多项式相位"""
    from scipy.special import factorial

    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL

    radius = min(height, width) // 2

    x = (np.arange(width) - width // 2) / radius
    y = (np.arange(height) - height // 2) / radius
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    mask = R <= 1.0

    def zernike_radial(n, m, r):
        R_arr = np.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            coef = ((-1) ** k * factorial(n - k)) / (
                factorial(k)
                * factorial((n + abs(m)) // 2 - k)
                * factorial((n - abs(m)) // 2 - k)
            )
            R_arr += coef * r ** (n - 2 * k)
        return R_arr

    if m >= 0:
        Z = zernike_radial(n, m, R) * np.cos(m * Theta)
    else:
        Z = zernike_radial(n, -m, R) * np.sin(-m * Theta)

    Z = Z * mask
    phase = Z * amplitude * 2 * np.pi
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    return img


def load_phase_csv(file_path: str) -> np.ndarray:
    """加载CSV格式的相位图案"""
    with open(file_path, "r") as f:
        header = f.readline().strip().split(",")
        cols = len(header) - 1

    data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=range(1, cols + 1))
    return data.astype(np.uint16)


def resize_to_slm(img: np.ndarray) -> np.ndarray:
    """将图像调整到SLM分辨率"""
    from scipy.ndimage import zoom

    target_height, target_width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]

    if img.shape[0] == target_height and img.shape[1] == target_width:
        return img

    zoom_y = target_height / img.shape[0]
    zoom_x = target_width / img.shape[1]
    img_scaled = zoom(img, (zoom_y, zoom_x), order=1)

    return img_scaled.astype(np.uint16)


# ============== Streamlit 应用 ==============

# 页面配置
st.set_page_config(
    page_title="SLM + Camera 可视化测试",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 SLM + Camera 实时可视化测试")

# 初始化 session state
if "slm_connected" not in st.session_state:
    st.session_state.slm_connected = False
if "camera_connected" not in st.session_state:
    st.session_state.camera_connected = False
if "camera" not in st.session_state:
    st.session_state.camera = None
if "slm" not in st.session_state:
    st.session_state.slm = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# ============== 侧边栏 - 设备连接 ==============
st.sidebar.header("🔌 设备连接")

# SLM 连接
st.sidebar.subheader("SLM 设置")
slm_connected = st.sidebar.checkbox("连接 SLM", value=st.session_state.slm_connected)

if slm_connected != st.session_state.slm_connected:
    if slm_connected:
        try:
            st.session_state.slm = SantecSLM200(
                slm_number=1,
                wavelength=1064,
                phase_range=200,
                video_mode=0,
            )
            st.session_state.slm.open()
            st.session_state.slm_connected = True
            st.sidebar.success("✅ SLM 已连接")
        except Exception as e:
            st.sidebar.error(f"❌ SLM 连接失败: {e}")
            st.session_state.slm_connected = False
    else:
        if st.session_state.slm:
            st.session_state.slm.close()
            st.session_state.slm = None
        st.session_state.slm_connected = False
        st.sidebar.info("ℹ️ SLM 已断开")

# 相机选择和连接
st.sidebar.subheader("相机设置")
camera_type = st.sidebar.selectbox("相机类型", ["MIICAM 4100", "Daheng"], index=0)

camera_connected = st.sidebar.checkbox(
    "连接相机", value=st.session_state.camera_connected
)

if camera_connected != st.session_state.camera_connected:
    if camera_connected:
        try:
            if camera_type == "MIICAM 4100":
                st.session_state.camera = MIICAMCamera(cam_id=0, exposure_time_ms=20)
            else:
                st.session_state.camera = DahengCamera(cam_id=0, exposure_time_ms=20)
            st.session_state.camera.initialize()
            st.session_state.camera_connected = True
            st.sidebar.success("✅ 相机已连接")
        except Exception as e:
            st.sidebar.error(f"❌ 相机连接失败: {e}")
            st.session_state.camera_connected = False
    else:
        if st.session_state.camera:
            st.session_state.camera.close()
            st.session_state.camera = None
        st.session_state.camera_connected = False
        st.sidebar.info("ℹ️ 相机已断开")

# ============== 主界面布局 ==============

# 创建两个主要列: SLM控制和相机显示
col1, col2 = st.columns([1, 2])

# ============== SLM 控制面板 ==============
with col1:
    st.subheader("🎛️ SLM 相位控制")

    # 相位来源选择
    phase_source = st.radio(
        "相位来源",
        ["参数生成", "加载 CSV"],
        horizontal=True,
    )

    phase_array = None

    if phase_source == "参数生成":
        # 相位类型选择
        phase_type = st.selectbox(
            "相位类型",
            [
                "闪耀光栅 (Blazed Grating)",
                "聚焦透镜 (Focus)",
                "棋盘格 (Checkerboard)",
                "二元光栅 (Binary Grating)",
                "涡旋光束 (Vortex)",
                "Zernike 多项式",
                "清空 (全零)",
            ],
        )

        # 根据相位类型显示对应参数
        if "闪耀光栅" in phase_type:
            period = st.slider("周期 (像素)", 4, 100, 20)
            direction = st.selectbox("方向", ["horizontal", "vertical"], 0)
            phase_array = generate_blazed_grating(period=period, direction=direction)

        elif "聚焦" in phase_type:
            focal_length = st.number_input(
                "焦距 (m)", value=0.5, min_value=0.01, max_value=10.0, step=0.01
            )
            wavelength = st.number_input(
                "波长 (nm)", value=532, min_value=450, max_value=1600, step=1
            )
            phase_array = generate_focus(
                focal_length=focal_length, wavelength=wavelength * 1e-9
            )

        elif "棋盘格" in phase_type:
            period = st.slider("周期 (像素)", 10, 200, 100)
            phase_array = generate_checkerboard(period=period)

        elif "二元光栅" in phase_type:
            period = st.slider("周期 (像素)", 4, 50, 8)
            direction = st.selectbox("方向", ["horizontal", "vertical"], 0)
            phase_array = generate_binary_grating(period=period, direction=direction)

        elif "涡旋" in phase_type:
            charge = st.slider("拓扑荷", 1, 10, 1)
            phase_array = generate_vortex(topological_charge=charge)

        elif "Zernike" in phase_type:
            n = st.selectbox("径向阶数 n", [1, 2, 3, 4, 5, 6, 7, 8], 3)
            m = st.selectbox("角向阶数 m", [-n, -n + 2, -n + 4, n], 0)
            amplitude = st.slider("振幅 (波长)", 0.5, 5.0, 2.0)
            phase_array = generate_zernike(n=n, m=m, amplitude=amplitude)

        elif "清空" in phase_type:
            phase_array = np.zeros(
                (SLM_RESOLUTION[1], SLM_RESOLUTION[0]), dtype=np.uint16
            )

    else:  # 加载 CSV
        csv_file = st.file_uploader("选择 CSV 文件", type=["csv"])

        if csv_file:
            try:
                # 保存上传的文件到临时位置
                temp_path = Path(f"/tmp/{csv_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(csv_file.getbuffer())

                # 加载并调整大小
                loaded = load_phase_csv(str(temp_path))
                phase_array = resize_to_slm(loaded)
                st.success(f"已加载: {csv_file.name}, 原始尺寸: {loaded.shape}")
            except Exception as e:
                st.error(f"加载失败: {e}")

    # 发送到 SLM 按钮
    if phase_array is not None and st.session_state.slm_connected:
        if st.button("🚀 发送到 SLM", type="primary"):
            try:
                st.session_state.slm.write_phase(phase_array, memory_number=1)
                st.session_state.slm.display_memory(1)
                st.success("✅ 已发送到 SLM")
            except Exception as e:
                st.error(f"❌ 发送失败: {e}")

    # 显示当前相位图案预览
    if phase_array is not None:
        st.image(
            phase_array,
            caption="相位图案预览",
            clamp=True,
            channels="GRAY",
        )
        st.caption(f"尺寸: {phase_array.shape}, 最大值: {phase_array.max()}")

# ============== 相机显示和控制 ==============
with col2:
    st.subheader("📷 相机实时显示")

    if not st.session_state.camera_connected:
        st.info("请先连接相机")
    else:
        # 相机控制
        with st.expander("📸 相机控制", expanded=True):
            # 曝光控制
            col_expo1, col_expo2 = st.columns(2)

            with col_expo1:
                # 手动曝光时间
                expo_time = st.number_input(
                    "曝光时间 (ms)",
                    min_value=1,
                    max_value=1000,
                    value=20,
                    step=1,
                )
                if st.button("设置曝光"):
                    try:
                        st.session_state.camera.reset_exposure_time(expo_time)
                        st.success(f"曝光时间设为 {expo_time} ms")
                    except Exception as e:
                        st.error(f"设置失败: {e}")

            with col_expo2:
                # 自动曝光
                auto_expo_enabled = st.checkbox("启用自动曝光", value=False)

                if auto_expo_enabled:
                    try:
                        st.session_state.camera.enable_auto_exposure(
                            enable=True, mode=1
                        )

                        target_val = st.slider("目标亮度", 16, 220, 120)

                        if st.button("设置目标亮度"):
                            st.session_state.camera.set_auto_exposure_target(
                                target=target_val
                            )
                            st.success(f"目标亮度设为 {target_val}")

                        # 显示自动曝光状态
                        state = st.session_state.camera.get_auto_exposure_state()
                        st.caption(f"状态: {state}")
                    except Exception as e:
                        st.error(f"自动曝光设置失败: {e}")
                else:
                    try:
                        st.session_state.camera.enable_auto_exposure(
                            enable=False, mode=0
                        )
                    except:
                        pass

        # 自动刷新控制
        auto_refresh = st.checkbox("自动刷新", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh

        refresh_interval = st.slider("刷新间隔 (秒)", 0.5, 5.0, 1.0)

        # 拍照按钮
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            n_sample = st.number_input("采样次数", 1, 10, 1)

        with col_btn2:
            if st.button("📸 拍照"):
                try:
                    img = st.session_state.camera.get_numpy_image(n_sample=n_sample)
                    st.session_state.last_image = img
                    st.success(f"已拍摄, 尺寸: {img.shape}")
                except Exception as e:
                    st.error(f"拍照失败: {e}")

        with col_btn3:
            if st.button("🔄 刷新显示"):
                pass

        # 显示图像
        if "last_image" in st.session_state:
            img = st.session_state.last_image

            # 显示图像统计
            st.caption(
                f"图像统计: 最大={img.max()}, 最小={img.min()}, 平均={img.mean():.1f}"
            )

            # 显示图像
            st.image(
                img,
                caption="相机画面",
                clamp=True,
                channels="GRAY",
            )
        else:
            st.info("点击「拍照」或启用「自动刷新」显示图像")

        # 自动刷新循环
        if auto_refresh:
            # 使用 placeholder 进行实时更新
            placeholder = st.empty()

            try:
                img = st.session_state.camera.get_numpy_image(n_sample=1)
                st.session_state.last_image = img

                # 显示
                with placeholder.container():
                    st.caption(
                        f"自动刷新中... 图像统计: 最大={img.max()}, 最小={img.min()}, 平均={img.mean():.1f}"
                    )
                    st.image(
                        img, caption="相机画面 (自动刷新)", clamp=True, channels="GRAY"
                    )

            except Exception as e:
                st.error(f"自动刷新失败: {e}")

            # JavaScript 定时刷新
            time.sleep(refresh_interval)
            st.rerun()

# ============== 底部信息 ==============
st.divider()
st.caption("🔬 AO-shaping SLM + Camera 可视化测试工具")
st.caption("SLM: Santec SLM-200 | 相机: MIICAM 4100 / Daheng")
