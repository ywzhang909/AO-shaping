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

import ctypes
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Any

import streamlit as st
import numpy as np
from loguru import logger

# 导入驱动
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.ccd.miicam import CameraStreamManager as MIICAMCamera
from ao_shaping.drivers.ccd.daheng import CameraStreamManager as DahengCamera
from ao_shaping.utils.phase_patterns import (
    SLM_RESOLUTION,
    generate_blazed_grating,
    generate_focus,
    generate_checkerboard,
    generate_binary_grating,
    generate_vortex,
    generate_zernike,
    resize_to_slm,
    load_phase_csv,
)


class SLMVisualizer:
    def __init__(self):
        self.slm = None
        self.camera = None
        self.phase_array = None
        self.last_image = None
        self.auto_refresh = False
        self.refresh_interval = 1.0
        self.n_sample = 1

    def verify_slm_display(self) -> Dict[str, Any]:
        """验证SLM显示状态"""
        result = {
            "success": False,
            "message": "",
            "display_memory": None,
        }

        try:
            # 直接从session state获取SLM对象
            slm = st.session_state.get("slm_obj", None)
            if slm is None:
                result["message"] = "SLM未连接"
                return result

            mem_num = ctypes.c_ulong(0)
            ret = slm._slm.SLM_Ctrl_ReadDS(slm.slm_number, ctypes.byref(mem_num))

            if ret == 0:
                result["success"] = True
                result["display_memory"] = mem_num.value
                result["message"] = f"SLM正在显示内存 #{mem_num.value}"
            else:
                result["message"] = f"读取显示状态失败, 错误码: {ret}"

        except Exception as e:
            result["message"] = f"验证失败: {e}"

        return result

    def generate_phase_pattern(self, phase_type: str, **kwargs) -> Optional[np.ndarray]:
        """生成相位图案"""
        try:
            if phase_type == "闪耀光栅":
                return generate_blazed_grating(
                    kwargs.get("period", 20),
                    kwargs.get("direction", "horizontal"),
                )
            elif phase_type == "聚焦透镜":
                return generate_focus(
                    focal_length=kwargs.get("focal_length", 0.5),
                    wavelength=kwargs.get("wavelength", 532e-9),
                )
            elif phase_type == "棋盘格":
                return generate_checkerboard(kwargs.get("period", 100))
            elif phase_type == "二元光栅":
                return generate_binary_grating(
                    kwargs.get("period", 8),
                    kwargs.get("direction", "horizontal"),
                )
            elif phase_type == "涡旋光束":
                return generate_vortex(kwargs.get("charge", 1))
            elif phase_type == "Zernike 多项式":
                return generate_zernike(
                    n=kwargs.get("n", 4),
                    m=kwargs.get("m", 0),
                    amplitude=kwargs.get("amplitude", 2.0),
                )
            elif phase_type == "清空":
                return np.zeros((1200, 1920), dtype=np.uint16)
        except Exception as e:
            st.error(f"相位生成失败: {e}")
            return None

    def send_to_slm(
        self, phase_array: np.ndarray, memory_number: Optional[int] = None
    ) -> bool:
        """发送相位图案到SLM

        Args:
            phase_array: 相位数据数组
            memory_number: 内存编号（2-15循环），默认自动递增
        """
        # 使用循环内存编号 (2-15)
        if memory_number is None:
            current_num: int = st.session_state.slm_memory_number
            # 递增并在 2-15 范围内循环
            next_num = current_num + 1
            if next_num > 15:
                next_num = 2
            st.session_state.slm_memory_number = next_num
            memory_number = current_num

        # 确保 memory_number 是 int 类型
        mem_num: int = memory_number  # type: ignore[assignment]

        try:
            # 直接从session state获取SLM对象（避免实例变量丢失问题）
            slm = st.session_state.get("slm_obj", None)

            if slm is None:
                st.error("SLM未连接")
                return False

            # 使用session state中的对象
            self.slm = slm

            # 计算预期灰度范围 (固定 2π = 1023)
            max_grayscale = 1023
            actual_max = phase_array.max()
            actual_min = phase_array.min()

            st.caption(
                f"相位数据: shape={phase_array.shape}, dtype={phase_array.dtype}, "
                f"max={actual_max}, min={actual_min}, "
                f"相位范围=0~2π (2π对应灰度{max_grayscale})"
            )

            # 警告如果灰度值超出当前相位范围的有效范围
            if actual_max > max_grayscale:
                st.warning(
                    f"⚠️ 相位最大灰度值 {actual_max} 超过当前相位范围允许的最大值 {max_grayscale}"
                )

            # 写入相位到SLM内存
            self.slm.write_phase(phase_array, memory_number=mem_num)
            st.info(f"相位数据已写入内存 #{mem_num}")

            # 显示相位
            self.slm.display_memory(mem_num)

            # 验证显示状态
            verify_result = self.verify_slm_display()
            if verify_result["success"]:
                # 验证显示的内存编号与预期一致
                if verify_result["display_memory"] == mem_num:
                    st.success(f"✅ {verify_result['message']}")
                    return True
                else:
                    st.warning(
                        f"⚠️ SLM 显示内存 #{verify_result['display_memory']} 与预期 #{mem_num} 不符"
                    )
                    return False
            else:
                st.warning(f"⚠️ 发送完成但验证失败: {verify_result['message']}")
                return False

        except Exception as e:
            st.error(f"❌ 发送失败: {e}")
            return False

    def take_photo(self, n_sample: int = 1) -> Optional[np.ndarray]:
        """拍摄照片"""
        try:
            # 直接从session state获取相机对象
            camera = st.session_state.get("camera_obj", None)
            if camera is None:
                st.error("相机未连接")
                return None

            img = camera.get_numpy_image(n_sample=n_sample)
            self.last_image = img
            st.success(f"✨ 已拍摄, 尺寸: {img.shape}")
            return img
        except Exception as e:
            st.error(f"❌ 拍照失败: {e}")
            return None

    def refresh_display(self, silent: bool = False) -> Optional[np.ndarray]:
        """刷新显示

        Args:
            silent: 如果为True，则不显示成功消息（用于自动刷新减少闪烁）
        """
        try:
            # 直接从session state获取相机对象
            camera = st.session_state.get("camera_obj", None)
            if camera is None:
                if not silent:
                    st.error("相机未连接")
                return None

            img = camera.get_numpy_image(n_sample=self.n_sample)
            self.last_image = img
            if not silent:
                st.success(f"🔄 已刷新, 尺寸: {img.shape}")
            return img
        except Exception as e:
            if not silent:
                st.error(f"❌ 刷新失败: {e}")
            logger.error(f"刷新显示失败: {e}")
            return None

    def set_exposure(self, expo_time: float) -> bool:
        """设置曝光时间"""
        try:
            # 直接从session state获取相机对象
            camera = st.session_state.get("camera_obj", None)
            if camera is None:
                st.error("相机未连接")
                return False

            # 将float转换为int (SLM SDK可能要求int)
            camera.reset_exposure_time(int(round(expo_time)))
            st.success(f"✨ 曝光时间设为 {expo_time} ms")
            return True
        except Exception as e:
            st.error(f"❌ 设置失败: {e}")
            return False

    def set_auto_exposure(self, enable: bool, target_val: Optional[int] = None) -> bool:
        """设置自动曝光"""
        try:
            # 直接从session state获取相机对象
            camera = st.session_state.get("camera_obj", None)
            if camera is None:
                st.error("相机未连接")
                return False

            camera.enable_auto_exposure(enable=enable, mode=1)
            if target_val is not None:
                camera.set_auto_exposure_target(target=target_val)
                st.success(f"✨ 目标亮度设为 {target_val}")
            return True
        except Exception as e:
            st.error(f"❌ 自动曝光设置失败: {e}")
            return False

    def run(self):
        """运行Streamlit应用"""
        # 页面配置
        st.set_page_config(
            page_title="SLM + Camera 实时可视化测试",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🧪 SLM + Camera 实时可视化测试")

        # 初始化 session state
        if "slm_connected" not in st.session_state:
            st.session_state.slm_connected = False
        if "camera_connected" not in st.session_state:
            st.session_state.camera_connected = False
        if "auto_refresh" not in st.session_state:
            st.session_state.auto_refresh = False
        if "camera_placeholder" not in st.session_state:
            st.session_state.camera_placeholder = None
        if "last_frame_time" not in st.session_state:
            st.session_state.last_frame_time = 0
        # 初始化内存编号 (2-15 循环)
        if "slm_memory_number" not in st.session_state:
            st.session_state.slm_memory_number = 2
        # 初始化聚焦参数
        if "focus_focal_length_mm" not in st.session_state:
            st.session_state.focus_focal_length_mm = 500
        if "focus_wavelength" not in st.session_state:
            st.session_state.focus_wavelength = 1064

        # ============== 侧边栏 - 设备连接 ==============
        st.sidebar.header("🔌 设备连接")

        # SLM 连接
        st.sidebar.subheader("SLM 设置")

        # SLM 参数设置
        slm_wavelength = st.sidebar.number_input(
            "波长 (nm)", min_value=450, max_value=1600, value=1064, step=1
        )

        slm_connected = st.sidebar.checkbox(
            "连接 SLM", value=st.session_state.slm_connected
        )

        if slm_connected != st.session_state.slm_connected:
            if slm_connected:
                try:
                    slm_obj = SantecSLM200(
                        slm_number=1,
                        wavelength=slm_wavelength,
                        video_mode=0,
                    )
                    slm_obj.open()
                    # 注意：不调用 set_wavelength()，保持SLM现有设置
                    # 将SLM对象存储在session state中以保持持久化
                    st.session_state.slm_obj = slm_obj
                    st.session_state.slm_connected = True
                    st.sidebar.success(f"✅ SLM 已连接 (保持现有设置)")
                except Exception as e:
                    st.sidebar.error(f"❌ SLM 连接失败: {e}")
                    st.session_state.slm_connected = False
                    if "slm_obj" in st.session_state:
                        del st.session_state.slm_obj
            else:
                if "slm_obj" in st.session_state and st.session_state.slm_obj:
                    st.session_state.slm_obj.close()
                    del st.session_state.slm_obj
                st.session_state.slm_connected = False
                st.sidebar.info("ℹ️ SLM 已断开")

        # 从session state恢复SLM对象到实例变量
        if "slm_obj" in st.session_state:
            self.slm = st.session_state.slm_obj
            st.sidebar.caption(f"📝 SLM对象已从session state恢复: {self.slm}")
        else:
            self.slm = None
            st.sidebar.caption("📝 session state中没有SLM对象")

        # 相机选择和连接
        st.sidebar.subheader("相机设置")
        camera_type = st.sidebar.selectbox(
            "相机类型", ["MIICAM 4100", "Daheng"], index=0
        )

        camera_connected = st.sidebar.checkbox(
            "连接相机", value=st.session_state.camera_connected
        )

        if camera_connected != st.session_state.camera_connected:
            if camera_connected:
                try:
                    if camera_type == "MIICAM 4100":
                        camera_obj = MIICAMCamera(cam_id=0, exposure_time_ms=20)
                    else:
                        camera_obj = DahengCamera(cam_id=0, exposure_time_ms=20)
                    camera_obj.initialize()
                    # 将相机对象存储在session state中以保持持久化
                    st.session_state.camera_obj = camera_obj
                    st.session_state.camera_connected = True
                    st.sidebar.success("✅ 相机已连接")
                except Exception as e:
                    st.sidebar.error(f"❌ 相机连接失败: {e}")
                    st.session_state.camera_connected = False
                    if "camera_obj" in st.session_state:
                        del st.session_state.camera_obj
            else:
                if "camera_obj" in st.session_state and st.session_state.camera_obj:
                    st.session_state.camera_obj.close()
                    del st.session_state.camera_obj
                st.session_state.camera_connected = False
                st.session_state.camera_placeholder = None  # 重置占位符
                st.session_state.last_frame_time = 0
                st.sidebar.info("ℹ️ 相机已断开")

        # 从session state恢复相机对象到实例变量
        if "camera_obj" in st.session_state:
            self.camera = st.session_state.camera_obj
        else:
            self.camera = None

        # ============== 主界面布局 ==============

        # 创建两个主要列: SLM控制和相机显示
        col1, col2 = st.columns([1, 2])

        # ============== SLM 控制面板 ==============
        with col1:
            st.subheader("🎮 SLM 相位控制")

            # 相位来源选择
            phase_source = st.radio(
                "相位来源",
                ["参数生成", "加载 CSV"],
                horizontal=True,
            )

            self.phase_array = None

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
                    self.phase_array = self.generate_phase_pattern(
                        "闪耀光栅", period=period, direction=direction
                    )
                    # 发送到 SLM 按钮
                    if self.phase_array is not None and st.session_state.slm_connected:
                        if st.button("🚀 发送到 SLM", type="primary", key="btn_blazed"):
                            self.send_to_slm(self.phase_array)

                elif "聚焦" in phase_type:
                    # 焦距：滑块 + 手动输入 (单位: mm)，使用 session state 联动
                    col_fl1, col_fl2 = st.columns([2, 1])
                    with col_fl1:
                        focal_length_mm = st.slider(
                            "焦距 (mm)",
                            min_value=10,
                            max_value=10000,
                            value=st.session_state.focus_focal_length_mm,
                            key="focus_slider",
                        )
                    with col_fl2:
                        focal_length_manual = st.number_input(
                            "手动 (mm)",
                            min_value=10,
                            max_value=10000,
                            value=st.session_state.focus_focal_length_mm,
                            step=10,
                            key="focus_manual",
                        )

                    # 同步滑块和手动输入
                    if focal_length_mm != st.session_state.focus_focal_length_mm:
                        st.session_state.focus_focal_length_mm = focal_length_mm
                        st.rerun()
                    if focal_length_manual != st.session_state.focus_focal_length_mm:
                        st.session_state.focus_focal_length_mm = focal_length_manual
                        st.rerun()

                    # 使用 session state 中的值
                    focal_length_mm = st.session_state.focus_focal_length_mm

                    # 波长：使用 session state，默认值为 sidebar 中的波长
                    wavelength = st.number_input(
                        "波长 (nm)",
                        min_value=450,
                        max_value=1600,
                        value=st.session_state.get("focus_wavelength", slm_wavelength),
                        step=1,
                        key="focus_wavelength_input",
                    )
                    if wavelength != st.session_state.focus_wavelength:
                        st.session_state.focus_wavelength = wavelength
                        st.rerun()

                    self.phase_array = self.generate_phase_pattern(
                        "聚焦透镜",
                        focal_length=focal_length_mm / 1000.0,  # mm 转 m
                        wavelength=wavelength * 1e-9,
                    )
                    # 发送到 SLM 按钮
                    if self.phase_array is not None and st.session_state.slm_connected:
                        if st.button("🚀 发送到 SLM", type="primary", key="btn_focus"):
                            self.send_to_slm(self.phase_array)

                elif "棋盘格" in phase_type:
                    period = st.slider("周期 (像素)", 10, 200, 100)
                    self.phase_array = self.generate_phase_pattern(
                        "棋盘格", period=period
                    )
                    # 发送到 SLM 按钮
                    if self.phase_array is not None and st.session_state.slm_connected:
                        if st.button(
                            "🚀 发送到 SLM", type="primary", key="btn_checker"
                        ):
                            self.send_to_slm(self.phase_array)

                elif "二元光栅" in phase_type:
                    period = st.slider("周期 (像素)", 4, 50, 8)
                    direction = st.selectbox("方向", ["horizontal", "vertical"], 0)
                    self.phase_array = self.generate_phase_pattern(
                        "二元光栅", period=period, direction=direction
                    )
                    # 发送到 SLM 按钮
                    if self.phase_array is not None and st.session_state.slm_connected:
                        if st.button("🚀 发送到 SLM", type="primary", key="btn_binary"):
                            self.send_to_slm(self.phase_array)

                elif "涡旋光束" in phase_type:
                    charge = st.slider("拓扑荷", 1, 10, 1)
                    self.phase_array = self.generate_phase_pattern(
                        "涡旋光束", charge=charge
                    )
                    # 发送到 SLM 按钮
                    if self.phase_array is not None and st.session_state.slm_connected:
                        if st.button("🚀 发送到 SLM", type="primary", key="btn_vortex"):
                            self.send_to_slm(self.phase_array)

                elif "Zernike" in phase_type:
                    n = st.selectbox("径向阶数 n", [1, 2, 3, 4, 5, 6, 7, 8], 3)
                    m = st.selectbox("角向阶数 m", [-n, -n + 2, -n + 4, n], 0)
                    amplitude = st.slider("振幅 (波长)", 0.5, 5.0, 2.0)
                    self.phase_array = self.generate_phase_pattern(
                        "Zernike 多项式", n=n, m=m, amplitude=amplitude
                    )
                    # 发送到 SLM 按钮
                    if self.phase_array is not None and st.session_state.slm_connected:
                        if st.button(
                            "🚀 发送到 SLM", type="primary", key="btn_zernike"
                        ):
                            self.send_to_slm(self.phase_array)

                elif "清空" in phase_type:
                    self.phase_array = np.zeros(
                        (SLM_RESOLUTION[1], SLM_RESOLUTION[0]), dtype=np.uint16
                    )
                    # 发送到 SLM 按钮
                    if self.phase_array is not None and st.session_state.slm_connected:
                        if st.button("🚀 发送到 SLM", type="primary", key="btn_clear"):
                            self.send_to_slm(self.phase_array)

            else:  # 加载 CSV
                csv_file = st.file_uploader("选择 CSV 文件", type=["csv"])

                if csv_file:
                    try:
                        # 保存上传的文件到临时位置 (使用系统temp目录)
                        temp_dir = tempfile.gettempdir()
                        safe_filename = Path(csv_file.name).name
                        temp_path = Path(temp_dir) / safe_filename
                        with open(temp_path, "wb") as f:
                            f.write(csv_file.getbuffer())

                        # 加载并调整大小
                        loaded = load_phase_csv(str(temp_path))
                        self.phase_array = resize_to_slm(loaded)
                        st.success(
                            f"✅ 已加载: {csv_file.name}, 原始尺寸: {loaded.shape}"
                        )
                        # 发送到 SLM 按钮
                        if (
                            self.phase_array is not None
                            and st.session_state.slm_connected
                        ):
                            if st.button(
                                "🚀 发送到 SLM", type="primary", key="btn_csv"
                            ):
                                self.send_to_slm(self.phase_array)
                    except Exception as e:
                        st.error(f"❌ 加载失败: {e}")

            # 显示当前相位图案预览
            if self.phase_array is not None:
                try:
                    st.image(
                        self.phase_array,
                        caption="相位图案预览",
                        clamp=True,
                        channels="GRAY",
                    )
                except Exception as e:
                    st.warning(f"⚠️ 预览图显示失败: {e}")
                st.caption(
                    f"尺寸: {self.phase_array.shape}, 最大值: {self.phase_array.max()}"
                )

        # ============== 相机显示和控制 ==============
        with col2:
            st.subheader("📸 相机实时显示")

            if not st.session_state.camera_connected:
                st.info("先连接相机")
            else:
                # 相机控制
                with st.expander("📸 相机控制", expanded=True):
                    # 露光控制
                    col_expo1, col_expo2 = st.columns(2)

                    with col_expo1:
                        # 手动露光时间
                        expo_time = st.number_input(
                            "露光时间 (ms)",
                            min_value=0.011,
                            max_value=2.0,
                            value=1.0,
                            step=0.001,
                        )
                        if st.button("设置露光"):
                            self.set_exposure(expo_time)

                    with col_expo2:
                        # 自动露光
                        auto_expo_enabled = st.checkbox("启用自动露光", value=False)

                        if auto_expo_enabled:
                            try:
                                self.set_auto_exposure(True)

                                target_val = st.slider("目标亮度", 16, 220, 120)

                                if st.button("设置目标亮度"):
                                    self.set_auto_exposure(True, target_val)

                                # 显示自动露光状态
                                state = "正在自动露光..."
                                st.caption(f"状态: {state}")
                            except Exception as e:
                                st.error(f"❌ 自动露光设置失败: {e}")
                        else:
                            try:
                                self.set_auto_exposure(False)
                            except Exception as e:
                                logger.debug(f"Failed to disable auto exposure: {e}")

                # 自动刷新控制
                auto_refresh = st.checkbox(
                    "自动刷新", value=st.session_state.auto_refresh
                )
                st.session_state.auto_refresh = auto_refresh

                refresh_interval = st.slider("刷新间隔 (秒)", 0.5, 5.0, 1.0)

                # 拍照按钮
                col_btn1, col_btn2, col_btn3 = st.columns(3)

                with col_btn1:
                    self.n_sample = st.number_input("采样次数", 1, 10, 1)

                with col_btn2:
                    if st.button("📸 拍照"):
                        self.take_photo(self.n_sample)

                with col_btn3:
                    if st.button("🔄 刷新显示"):
                        self.refresh_display()

                # 图像显示区域 - 使用占位符避免闪烁
                if st.session_state.camera_placeholder is None:
                    st.session_state.camera_placeholder = st.empty()

                # 显示图像
                if self.last_image is not None:
                    # 使用占位符更新图像，避免整个页面重绘
                    with st.session_state.camera_placeholder.container():
                        # 显示图像统计
                        st.caption(
                            f"图像统计: 最大={self.last_image.max()}, 最小={self.last_image.min()}, 平均={self.last_image.mean():.1f}"
                        )
                        # 显示图像 - 使用固定宽度避免布局抖动
                        st.image(
                            self.last_image,
                            caption="相机画面",
                            clamp=True,
                            channels="GRAY",
                            use_container_width=True,
                        )
                else:
                    # 自动刷新时不显示提示信息
                    if not auto_refresh:
                        with st.session_state.camera_placeholder.container():
                            st.info("点击「拍照」或启用「自动刷新」显示图像")

                # 自动刷新循环 - 使用非阻塞方式
                if auto_refresh:
                    current_time = time.time()
                    elapsed = current_time - st.session_state.last_frame_time

                    # 只在达到刷新间隔时更新
                    if elapsed >= refresh_interval:
                        try:
                            img = self.refresh_display(silent=True)
                            if img is not None:
                                # 更新 last_image 以便在非自动刷新时也能显示
                                self.last_image = img
                                st.session_state.last_frame_time = current_time
                                # 直接更新占位符内容，不触发全页面rerun
                                with st.session_state.camera_placeholder.container():
                                    st.caption(
                                        f"自动刷新中... 图像统计: 最大={img.max()}, 最小={img.min()}, 平均={img.mean():.1f}"
                                    )
                                    st.image(
                                        img,
                                        caption="相机画面 (自动刷新)",
                                        clamp=True,
                                        channels="GRAY",
                                        use_container_width=True,
                                    )
                        except Exception as e:
                            st.error(f"❌ 自动刷新失败: {e}")

                    # 使用 st.rerun() 但保持占位符状态
                    time.sleep(0.05)  # 短暂休眠避免CPU占用过高
                    st.rerun()

        # ============== 底部信息 ==============
        st.divider()
        st.caption("🧪 AO-shaping SLM + Camera 可视化测试工具")
        st.caption("SLM: Santec SLM-200 | 相机: MIICAM 4100 / Daheng")


if __name__ == "__main__":
    app = SLMVisualizer()
    app.run()
