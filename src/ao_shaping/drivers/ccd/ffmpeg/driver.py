"""面向视频流与虚拟相机的 FFmpeg/OpenCV 相机驱动。

本驱动借助 OpenCV 的 FFmpeg 后端提供相机接口兼容。支持:
- 视频文件 (本地 .mp4、.avi 等)
- RTSP/RTMP/HTTP 流
- 虚拟相机设备 (例如 OBS virtual camera)
- 摄像头设备 (在 Windows 上经 DirectShow)
- 按时间戳命名的图像文件目录 (虚拟 CCD)

注意: 曝光时间控制是仿真的 —— 它控制的是内部帧处理延迟, 而非真实硬件曝光。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.utils.io.file import logger
from ao_shaping.utils.io.timestamp import TimestampParser


class FFmpegCameraError(CameraError):
    """FFmpeg 相机错误异常。"""

    pass


class FFmpegCamera(BaseCamera):
    """基于 FFmpeg/OpenCV 的相机驱动。

    通过 OpenCV 的 FFmpeg 后端提供相机接口兼容。支持视频文件、网络流与虚拟
    相机设备。

    参数:
        cam_id: 相机标识 —— 可以是:
            - 整数: 设备索引 (0 为默认摄像头, 1 为第二台相机等)
            - 字符串: 文件路径或流 URL (如 "video.mp4"、"rtsp://...")
        exposure_time_ms: 仿真的曝光时间, 单位毫秒。
            注意: 这是仿真参数 —— 真实曝光取决于视频源/流, 无法被控制。
        skip_sampling: 为 True 时跳帧以降低采集速率。
            为 False 则逐帧等待 (更慢, 但不丢帧)。

    示例:
        # 打开一个视频文件
        with FFmpegCamera(cam_id="test_video.mp4") as cam:
            img = cam.get_numpy_image()

        # 打开一个网络流
        with FFmpegCamera(cam_id="rtsp://192.168.1.100:8554 live stream") as cam:
            img = cam.get_numpy_image()

        # 打开虚拟相机 (Windows)
        with FFmpegCamera(cam_id=1) as cam:  # OBS Virtual Camera 通常显示为设备 1
            img = cam.get_numpy_image()
    """

    def __init__(
        self,
        cam_id: int | str = 0,
        exposure_time_ms: float = 20.0,
        skip_sampling: bool = False,
    ):
        """初始化 FFmpeg 相机配置。

        参数:
            cam_id: 相机设备索引、文件路径或流 URL。
            exposure_time_ms: 初始曝光时间, 单位毫秒 (仿真)。
            skip_sampling: 是否跳帧以加快采集。
        """
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self._cap = None
        self._stream_url = None
        self._is_file = False
        self._fps = 30.0
        self._frame_count = 0
        self._total_frames = 0

    def __enter__(self) -> FFmpegCamera:
        """上下文管理器入口 —— 初始化相机。"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """上下文管理器出口 —— 清理相机资源。"""
        self.close()

    def initialize(self) -> None:
        """初始化 FFmpeg 相机 / 视频流。

        本方法:
        1. 关闭此前已打开的相机。
        2. 按 cam_id 打开指定的视频源。
        3. 读取视频属性 (分辨率、fps)。
        4. 判断源是文件、流还是设备。

        异常:
            ConnectionAbortedError: 相机/视频源打不开。
            FFmpegCameraError: 读不到视频属性。
        """
        # 关闭此前已打开的相机
        self.close()

        import cv2

        # 处理不同的源类型
        if isinstance(self.cam_id, int):
            # 设备索引 (摄像头、虚拟相机)
            self._cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
            self._stream_url = f"device_{self.cam_id}"
            self._is_file = False
        elif isinstance(self.cam_id, str):
            # 判断是 URL 还是文件路径
            if self.cam_id.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                # 网络流
                self._cap = cv2.VideoCapture(self.cam_id)
                self._stream_url = self.cam_id
                self._is_file = False
            else:
                # 本地视频文件
                self._cap = cv2.VideoCapture(self.cam_id)
                self._stream_url = self.cam_id
                self._is_file = True
        else:
            raise FFmpegCameraError(
                f"Invalid cam_id type: {type(self.cam_id)}. "
                "Expected int (device index) or str (file path/URL)."
            )

        # 检查是否成功打开
        if not self._cap or not self._cap.isOpened():
            error_info = f"Failed to open video source: {self.cam_id}"
            logger.error(error_info)
            raise ConnectionAbortedError(error_info)

        # 取视频属性
        self.cam_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

        # 取总帧数 (只对视频文件有效, 流不行)
        if self._is_file:
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self._total_frames = 0

        # 由源生成序列号
        self._sn = f"FFmpeg_{self.cam_id}"

        # 建立坐标网格
        self.xv, self.yv = self._get_grid(self.cam_width, self.cam_height)

        logger.info(
            f"Open FFmpeg camera {self.cam_id} success. "
            f"width={self.cam_width}, height={self.cam_height}, fps={self._fps}"
        )

    def open(self) -> None:
        """打开相机设备 (initialize 的别名)。"""
        self.initialize()

    def close(self) -> None:
        """关闭相机设备并释放资源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.cam_width = 0
        self.cam_height = 0
        self._frame_count = 0
        logger.info(f"Closed FFmpeg camera: {self.cam_id}")

    def reset_exposure_time(self, time_ms: float) -> float:
        """设置仿真的曝光时间。

        注意: 这是仿真参数 —— 真实曝光取决于视频源/流, 无法经 FFmpeg 控制。
        该设置只是给帧采集之间加一点延迟。

        参数:
            time_ms: 新的曝光时间, 单位毫秒。

        返回:
            float: 所设定的曝光时间。

        异常:
            AssertionError: 相机未初始化。
        """
        assert self._cap is not None and self._cap.isOpened(), "camera not initialized"
        self.exposure_time_ms = max(0.011, float(time_ms))
        logger.warning(
            f"[FFmpegCamera] Exposure time is SIMULATED only - adds {self.exposure_time_ms}ms "
            f"delay between captures. Actual exposure depends on video source, not controllable."
        )
        return self.exposure_time_ms

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """重设视频 ROI (FFmpeg 不支持)。

        FFmpeg 不支持感兴趣区域。返回当前尺寸。

        参数:
            center: 忽略 (不支持)。
            size: 忽略 (不支持)。

        返回:
            ((width, height), (center_x, center_y))。

        异常:
            AssertionError: 相机未初始化。
        """
        assert self._cap is not None and self._cap.isOpened(), "camera not initialized"
        # FFmpeg 不支持 ROI —— 返回当前尺寸
        center_x = self.cam_width // 2
        center_y = self.cam_height // 2
        return (self.cam_width, self.cam_height), (center_x, center_y)

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """从 FFmpeg 视频源采集图像, 可选做平均。

        参数:
            n_sample: 平均的采样数。必须 > 0。
            skip_first: 是否跳过第一帧 (对流常常不稳定)。

        返回:
            np.ndarray: 采集到的图像, 为 uint8 数组。

        异常:
            AssertionError: n_sample 不是正数, 或相机未初始化。
            FFmpegCameraError: 采集帧失败。
        """
        assert self._cap is not None and self._cap.isOpened(), "camera not initialized"
        assert n_sample > 0, "Sample count must be > 0"

        import cv2

        # 按需跳过第一帧
        if skip_first:
            ret, frame = self._cap.read()
            if not ret:
                raise FFmpegCameraError(f"Failed to capture frame from {self.cam_id}")
            self._frame_count += 1

        # 采集帧
        frames = []
        for _ in range(n_sample):
            ret, frame = self._cap.read()
            if not ret:
                # 对视频文件, 循环回到开头
                if self._is_file and self._total_frames > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_count = 0
                    ret, frame = self._cap.read()
                    if not ret:
                        raise FFmpegCameraError(
                            f"Failed to capture frame from {self.cam_id}"
                        )
                else:
                    raise FFmpegCameraError(
                        f"Failed to capture frame from {self.cam_id}. "
                        "Stream may have ended."
                    )

            self._frame_count += 1

            # 需要时把 BGR 转成灰度
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

            # 仿真的曝光时间延迟
            if self.exposure_time_ms > 0:
                time.sleep(self.exposure_time_ms / 1000.0)

        # 计算平均
        avg_img = np.mean(frames, axis=0)

        # 跳采样: 丢帧以降低采集速率
        if self.skip_sampling and self._is_file:
            # skip_sampling 为 True 时跳掉一半的帧
            skip_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
            if skip_count > 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, skip_count)

        return avg_img.astype(np.uint8)

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """启用或关闭自动曝光。

        注意: FFmpeg 源不支持。返回 False。

        参数:
            enable: 忽略。
            mode: 忽略。

        返回:
            bool: 始终返回 False (不支持)。
        """
        logger.warning("Auto exposure not supported for FFmpeg sources")
        return False

    def set_auto_exposure_target(self, target: int) -> int:
        """设置自动曝光的目标亮度。

        注意: FFmpeg 源不支持。

        参数:
            target: 忽略。

        返回:
            int: 返回 0 (不支持)。

        异常:
            NotImplementedError: 始终抛出 (不支持)。
        """
        raise NotImplementedError("Auto exposure not supported for FFmpeg sources")

    def get_auto_exposure_state(self) -> dict:
        """获取当前自动曝光状态。

        返回:
            dict: 始终指示禁用状态。
        """
        return {
            "enabled": False,
            "mode": 0,
            "target": 0,
        }

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """设置自动曝光的时间与增益范围。

        注意: FFmpeg 源不支持。

        参数:
            max_time_ms: 忽略。
            min_time_ms: 忽略。
            max_gain: 忽略。
            min_gain: 忽略。

        返回:
            bool: 始终返回 False (不支持)。
        """
        logger.warning("Auto exposure range not supported for FFmpeg sources")
        return False

    @staticmethod
    def get_cam_list() -> list:
        """获取可用视频采集设备列表。

        注意: 这是尽力而为的列表。在 Windows 上, 它可能只能探测到 DirectShow
        设备。视频文件与流不会被枚举。

        返回:
            可尝试的设备索引 (0-9) 列表。
        """
        import cv2

        devices = []

        # 尝试在 Windows 上探测 DirectShow 设备
        if hasattr(cv2, "CAP_DSHOW"):
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap is not None and cap.isOpened():
                        devices.append(i)
                        cap.release()
                except Exception:
                    pass
        else:
            # 尝试默认后端
            for i in range(5):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap is not None and cap.isOpened():
                        devices.append(i)
                        cap.release()
                except Exception:
                    pass

        return devices


class ImageFolderCamera(BaseCamera):
    """从按时间戳命名的文件目录读取图像的相机驱动。

    本驱动通过从一个包含图像文件的目录中顺序读取来模拟 CCD 相机, 这些文件名带
    时间戳 (例如 1700000000000.png、1700000000050.png)。

    适用于:
    - 回放已录制的图像序列
    - 用已保存的数据测试优化流程
    - 以录制数据做虚拟 CCD 仿真

    参数:
        cam_id: 含图像文件的目录路径。
        exposure_time_ms: 读帧之间的延迟 (仿真曝光)。
            注意: 这是仿真参数 —— 真实的采集时序取决于文件时间戳, 无法控制。
        skip_sampling: 为 True 时读取时每隔一帧跳一张。
            为 False 则按顺序读每一帧。

    示例:
        # 打开一个带时间戳图像的目录
        with ImageFolderCamera(cam_id="/path/to/images") as cam:
            img = cam.get_numpy_image(n_sample=5)

        # 遍历图像 (类视频回放)
        with ImageFolderCamera(cam_id="recordings/20240101") as cam:
            for _ in range(100):
                img = cam.get_numpy_image()
                # 处理 img...
    """

    # 支持的图像扩展名
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}

    def __init__(
        self,
        cam_id: str = "",
        exposure_time_ms: float = 20.0,
        skip_sampling: bool = False,
    ):
        """初始化图片文件夹相机。

        参数:
            cam_id: 含图像文件的目录路径。
            exposure_time_ms: 两次读取之间的延迟, 单位毫秒。
            skip_sampling: 是否每隔一帧跳一张。
        """
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self._folder_path: Path | None = None
        self._image_files: list[Path] = []
        self._current_index: int = 0
        self._last_image: np.ndarray | None = None

    def __enter__(self) -> ImageFolderCamera:
        """上下文管理器入口 —— 初始化相机。"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """上下文管理器出口 —— 清理资源。"""
        self.close()

    def initialize(self) -> None:
        """通过扫描目录中的图像文件来初始化。

        本方法:
        1. 关闭此前已打开的目录。
        2. 校验目录路径存在。
        3. 扫描受支持的图像文件。
        4. 按文件名排序 (通常是时间戳)。
        5. 读第一张图以得到尺寸。

        异常:
            ConnectionAbortedError: 目录不存在或没找到图像。
            FFmpegCameraError: 读不出图像尺寸。
        """
        # 关闭此前已打开的目录
        self.close()

        # 把 cam_id 转成 Path
        self._folder_path = Path(self.cam_id)

        # 校验目录存在
        if not self._folder_path.exists():
            raise ConnectionAbortedError(f"Image folder not found: {self.cam_id}")

        if not self._folder_path.is_dir():
            raise ConnectionAbortedError(f"Path is not a directory: {self.cam_id}")

        # 扫描图像文件
        self._image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            self._image_files.extend(self._folder_path.glob(f"*{ext}"))
            self._image_files.extend(self._folder_path.glob(f"*{ext.upper()}"))

        if not self._image_files:
            raise ConnectionAbortedError(
                f"No image files found in folder: {self.cam_id}. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        # 用 TimestampParser 按时间戳排序
        self._image_files = TimestampParser().sort_files(self._image_files)
        self._current_index = 0

        # 读第一张图以得到尺寸
        first_img = self._read_image(self._image_files[0])
        if first_img is None:
            raise FFmpegCameraError(
                f"Failed to read first image: {self._image_files[0]}"
            )

        self.cam_width = first_img.shape[1]
        self.cam_height = first_img.shape[0]
        self._last_image = first_img

        # 由目录名生成序列号
        self._sn = f"ImageFolder_{self._folder_path.name}"

        # 建立坐标网格
        self.xv, self.yv = self._get_grid(self.cam_width, self.cam_height)

        logger.info(
            f"Open ImageFolder camera {self.cam_id} success. "
            f"Found {len(self._image_files)} images, "
            f"width={self.cam_width}, height={self.cam_height}"
        )

    def _read_image(self, image_path: Path) -> np.ndarray | None:
        """读取单个图像文件。

        参数:
            image_path: 图像文件路径。

        返回:
            np.ndarray: 灰度图像; 读取失败时返回 None。
        """
        import cv2

        try:
            # 特殊处理 .npy 文件
            if image_path.suffix.lower() == ".npy":
                img = np.load(image_path)
            else:
                # 用 OpenCV 读取
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None

            return img
        except Exception as e:
            logger.warning(f"Error reading {image_path}: {e}")
            return None

    def open(self) -> None:
        """打开图像目录 (initialize 的别名)。"""
        self.initialize()

    def close(self) -> None:
        """关闭相机并释放资源。"""
        self._image_files = []
        self._current_index = 0
        self._last_image = None
        self.cam_width = 0
        self.cam_height = 0
        logger.info(f"Closed ImageFolder camera: {self.cam_id}")

    def reset_exposure_time(self, time_ms: float) -> float:
        """设置仿真的曝光时间 (读帧之间的延迟)。

        注意: 这是仿真参数 —— 只给读帧之间加延迟。
        真实时序取决于文件时间戳, 无法控制。

        参数:
            time_ms: 新的延迟时间, 单位毫秒。

        返回:
            float: 所设定的延迟时间。

        异常:
            AssertionError: 相机未初始化。
        """
        assert self._folder_path is not None, "camera not initialized"
        self.exposure_time_ms = max(0.0, float(time_ms))
        logger.warning(
            f"[ImageFolderCamera] Delay is SIMULATED only - adds {self.exposure_time_ms}ms "
            f"delay between reads. Actual timing depends on file timestamps."
        )
        return self.exposure_time_ms

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """重设图像 ROI (不支持)。

        图片文件夹相机不支持 ROI。返回当前尺寸。

        参数:
            center: 忽略 (不支持)。
            size: 忽略 (不支持)。

        返回:
            ((width, height), (center_x, center_y))。

        异常:
            AssertionError: 相机未初始化。
        """
        assert self._folder_path is not None, "camera not initialized"
        center_x = self.cam_width // 2
        center_y = self.cam_height // 2
        return (self.cam_width, self.cam_height), (center_x, center_y)

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """从目录读取图像, 可选做平均。

        参数:
            n_sample: 平均的采样数。必须 > 0。
            skip_first: 是否跳过第一张图 (常常不稳定)。

        返回:
            np.ndarray: 采集到的图像, 为 uint8 数组。

        异常:
            AssertionError: n_sample 不是正数, 或相机未初始化。
            FFmpegCameraError: 读图失败。
        """
        assert self._folder_path is not None, "camera not initialized"
        assert n_sample > 0, "Sample count must be > 0"

        # 按需跳过第一张图
        if skip_first:
            self._advance_index()
            if self._current_index >= len(self._image_files):
                # 循环回到开头
                self._current_index = 0

        # 读取所请求张数的图像
        frames = []
        for _ in range(n_sample):
            if self._current_index >= len(self._image_files):
                # 连续回放时循环回到开头
                self._current_index = 0

            img = self._read_image(self._image_files[self._current_index])
            if img is None:
                raise FFmpegCameraError(
                    f"Failed to read image: {self._image_files[self._current_index]}"
                )

            frames.append(img)
            self._last_image = img
            self._advance_index()

            # 仿真的曝光时间延迟
            if self.exposure_time_ms > 0:
                time.sleep(self.exposure_time_ms / 1000.0)

        # 计算平均
        avg_img = np.mean(frames, axis=0)
        return avg_img.astype(np.uint8)

    def _advance_index(self) -> None:
        """前进当前索引, 并处理 skip_sampling。"""
        if self.skip_sampling:
            # 每隔一张跳一张
            self._current_index += 2
        else:
            self._current_index += 1

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """启用或关闭自动曝光。

        注意: 图片文件夹不支持。返回 False。

        参数:
            enable: 忽略。
            mode: 忽略。

        返回:
            bool: 始终返回 False (不支持)。
        """
        logger.warning("Auto exposure not supported for ImageFolder sources")
        return False

    def set_auto_exposure_target(self, target: int) -> int:
        """设置自动曝光的目标亮度。

        注意: 图片文件夹不支持。

        参数:
            target: 忽略。

        返回:
            int: 返回 0 (不支持)。

        异常:
            NotImplementedError: 始终抛出。
        """
        raise NotImplementedError("Auto exposure not supported for ImageFolder sources")

    def get_auto_exposure_state(self) -> dict:
        """获取当前自动曝光状态。

        返回:
            dict: 始终指示禁用状态。
        """
        return {
            "enabled": False,
            "mode": 0,
            "target": 0,
        }

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """设置自动曝光的时间与增益范围。

        注意: 图片文件夹不支持。

        参数:
            max_time_ms: 忽略。
            min_time_ms: 忽略。
            max_gain: 忽略。
            min_gain: 忽略。

        返回:
            bool: 始终返回 False (不支持)。
        """
        logger.warning("Auto exposure range not supported for ImageFolder sources")
        return False

    @staticmethod
    def get_cam_list() -> list:
        """获取可用图像目录列表 (不适用)。

        返回:
            空列表 —— 图像目录按路径指定, 不做枚举。
        """
        logger.warning(
            "ImageFolderCamera.get_cam_list() not applicable - specify folder path directly"
        )
        return []
