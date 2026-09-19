# -*- coding: utf-8 -*-
"""SLM 标定工具包 (单文件合并版)

本模块由三个来源合并而成 (2026-09 重构):
  1. ``drivers/slm/slm_calibration.py`` — SLM 相位-灰度响应标定
     (闪耀光栅法 / 零级比值法 / 干涉法 / 衍射效率法, 含自动曝光)
  2. ``tools/slm/slm_shift_calib.py`` — SLM defocus 平移标定
     (shift_x / shift_y, WFS tip/tilt 零点法)
  3. ``tools/slm/calibration.py`` (原) — SLM+CCD 一体化标定工具包
     (几何标定: 装配 -> 光束位置 -> K/旋转 -> 验证; LUT 标定已废弃)

标定光路

激光器(单纵模, 线偏振)
   │  偏振方向对准 SLM 液晶取向
   ▼
扩束准直镜组 ──► 光斑均匀照亮 SLM 有效区(左右半屏照度均匀是 LUT 前提)
   │
   ▼
┌─────────────────────────────────────────────┐
│ 相位 SLM (Santec SLM-200, 反射式 LCOS)        │
│                                              │
│  ┌─────────────┬─────────────┐               │
│  │  左半屏       │   右半屏      │  ← LUT标定:   │
│  │  (灰度 g)    │  (灰度 g_ref)│   两半不同灰度  │
│  ├─────────────┴─────────────┤               │
│  │   几何标定: 满屏闪耀光栅      │   ← x/y 方向   │
│  │   (周期 16/20/24/32 px)     │    周期变化    │
│  └───────────────────────────┘               │
└─────────────────────────────────────────────┘
   │ 反射光
   ▼
傅里叶透镜 L1 (焦距 f)
   │
   ▼
L1 后焦面 (频谱面) ◄── CCD 放在这里
   ┌──────────────────────────┐
   │  · 0级光斑(直反光)        │ ← 位置=质心标定;
   │    └─ 内部含干涉条纹(Λ≈K/960px)│   LUT标定发生在这里
   │  · +1级光斑(K/P 位移)  · -1级 │ ← 几何标定测这两个
   └──────────────────────────┘

注意一个细节：LUT 标定时不要加挡 0 级的针孔（整形运行时才需要），几何标定反而要避开 0 级（代码里用 3×FWHM 排除盘 + 闪耀移到 +1 级）。

包含的标定器:
  SLMCCDCalibrator          装配辅助 + 光束位置 + 几何标定 + 效果验证 (几何 CLI 唯一流程)
  SLMLUTCalibrator          [DEPRECATED] 灰度-相位 LUT 标定(自参考干涉法, legacy 8-bit)
                             canonical 灰度↔相位 LUT 由 slm-lut 管线提供
                             (tools/slm/slm_lut_runner.py + utils/slm_lut.py, 输出
                             lut_forward.csv/lut_inverse.csv, 经 Santec.load_lut 加载);
                             SLMLUTCalibrator 仅保留供参考/单测, 已移出 CLI.
  SantecCalibrator          闪耀光栅法/零级比值法标定器 (相位-灰度响应)
  InterferometerCalibrator  干涉法标定器
  DiffractionEfficiencyCalibrator  衍射效率法标定器
  AutoExposureController    自动曝光控制器

几何标定完整流程(全部可独立调用):
  align()              装配辅助: 0级/±1级入视场检查与定心引导
  find_beam_on_slm()   刀口扫描测光束在SLM面板的中心/尺寸
  calibrate()          几何标定: center/Kx/Ky/rotation/crop_side
  verify()             几何验证: 质心漂移/预测残差/能量集中度 (PASS/FAIL)

defocus 平移标定 (shift_x / shift_y) — WFS tip/tilt 零点法:
  图案平移 ``(sx, sy)`` 后光束感受到 ``P(b − s + ξ)`` (``b`` = 光束光轴在 SLM 坐标中的
  偏移, 未知)。对 defocus ``P = D·u²`` 有局部梯度 ``∝ 2D(b − s)`` → **WFS 读出的
  tip/tilt 关于 shift 线性, 零点即 ``s = b``**(图案中心与光束对齐)。
  判据用**相对纯平的"附加"倾斜** ``‖z_tilt(defocus@shift) − z_tilt(flat)‖`` —— 系统
  本身有静态倾斜 (实测 flat 下 tilt ≈ −0.14λ), 绝对归零是错的判据。
  关键约束 (2026-09-15 实测踩坑):
    1. 幅度 A 必须足够大 (默认 20 rad): A=2 时响应被 ``(r_beam/R)²`` 压制到噪声级。
    2. Zernike 半径 R 必须 > 光束半径: 实测光束在 SLM 上半径 ≈200px (≈1.6mm)。
       R=200 响应最强但一平移就裁切光束; 默认 R=600 可平移 ±400px 不裁切。
       光束尺寸可由 ``--radius-scan`` 诊断 (R 扫描取 Δdefocus 最大者)。
    3. shift 限制 ±500: defocus 盘中心 ``(960+sx, 600+sy)`` 超出 1920×1200 面板后
       光束几乎看不到图案, 数据无效。
    4. SLM 轴 ↔ WFS 轴存在 90° 交换: 本机实测 SLM-x → WFS Noll3 (y-tilt),
       SLM-y → WFS Noll2 (x-tip)。故判据取轴无关的 ``‖Δz_tilt‖``。
  实测结果 (SLM#22030102 + WFS M01219666, 532nm): ``shift_x=106, shift_y=40``
  (附加倾斜 0.854λ → 0.0245λ, 降低 97.1%)。

CLI (双入口):
  python -m ao_shaping.tools.slm.calibration                          # 几何标定 (装配->光束->几何->验证)
  python -m ao_shaping.tools.slm.calibration --skip-align --skip-beam # 只重新标定
  python -m ao_shaping.tools.slm.calibration --verify-only calib.npz  # 只验证已有标定
  python -m ao_shaping.tools.slm.calibration shift                     # defocus 平移标定
  python -m ao_shaping.tools.slm.calibration shift --defocus-a 30 --radius-scan
  python -m ao_shaping.tools.slm.calibration shift --no-save           # 只测不写 config
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch

import click
import numpy as np
from loguru import logger
from scipy.interpolate import UnivariateSpline

from ao_shaping.drivers.ccd import DahengCamera
from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.wfs import ThorlabWFS
from ao_shaping.tools.slm.slm_scan_analysis import clamp_shift, parabolic_min
from ao_shaping.tools.slm.slm_zernike_common import make_phase, measure_tilt_defocus
from ao_shaping.utils.pattern_helper import PatternHelper

SETTLE_S = 0.2
CAMERA_SAMPLES = 10
PANEL_H, PANEL_W = 1200, 1920
DEFAULT_CONFIG = Path("data/slm_configs/slm")
DEFAULT_OUTPUT = Path("data/calibration/defocus_shift_calib.json")

# 安全阈值
MAX_EXPOSURE_MS = 7.0
DEFAULT_EXPOSURE_MS = 4.0
DEFAULT_SHIFT_LIMIT = 500


# =====================================================================
# SLM 相位-灰度响应标定 (闪耀光栅法/零级比值法/干涉法/衍射效率法)
# =====================================================================
def lrange(a, b, step):
    return list(range(a, b, step))


@dataclass
class CalibrationResult:
    """标定结果数据类

    Attributes:
        grayscale_2pi: 2π相位对应的灰度值
        grayscale_values: 扫描的灰度值数组
        intensities: 对应的光强数组
        wavelength_nm: 标定波长（nm）
        slm_model: SLM型号
        timestamp: 标定时间戳
        metadata: 其他元数据
    """

    grayscale_2pi: int
    grayscale_values: list[int]
    intensities: np.ndarray
    wavelength_nm: int
    slm_model: str
    timestamp: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        return {
            "grayscale_2pi": self.grayscale_2pi,
            "grayscale_values": self.grayscale_values,
            "intensities": self.intensities.tolist(),
            "wavelength_nm": self.wavelength_nm,
            "slm_model": self.slm_model,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationResult":
        """从字典创建标定结果"""
        return cls(
            grayscale_2pi=data["grayscale_2pi"],
            grayscale_values=data["grayscale_values"],
            intensities=np.array(data["intensities"]),
            wavelength_nm=data["wavelength_nm"],
            slm_model=data["slm_model"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """保存标定结果到JSON文件

        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"标定结果已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CalibrationResult":
        """从JSON文件加载标定结果

        Args:
            filepath: 文件路径

        Returns:
            CalibrationResult实例
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"标定文件不存在: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"已从 {filepath} 加载标定结果")
        return cls.from_dict(data)


class SLMProtocol(Protocol):
    """SLM设备协议接口

    定义标定所需的SLM操作接口。
    任何实现这些方法的SLM驱动都可以使用标定功能。
    """

    @property
    def width(self) -> int:
        """SLM宽度（像素）"""
        ...

    @property
    def height(self) -> int:
        """SLM高度（像素）"""
        ...

    @property
    def wavelength(self) -> int:
        """当前波长（nm）"""
        ...

    def write_phase(self, phase: np.ndarray, memory_number: int = 1) -> None:
        """写入相位数据到SLM内存

        Args:
            phase: 相位数据（灰度值数组）
            memory_number: 内存编号
        """
        ...

    def display_memory(self, memory_number: int) -> None:
        """显示指定内存的相位图

        Args:
            memory_number: 内存编号
        """
        ...

    def set_grayscale(self, gs: int) -> None:
        """设置均匀灰度值

        Args:
            gs: 灰度值
        """
        ...


class CameraProtocol(Protocol):
    """相机设备协议接口

    定义标定所需的相机操作接口。
    """

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        """获取图像

        Args:
            n_sample: 采样次数
            skip_first: 是否跳过第一帧

        Returns:
            图像数组
        """
        ...


class CameraWithExposureProtocol(CameraProtocol):
    """带曝光控制的相机协议接口

    扩展CameraProtocol，添加曝光控制功能。
    """

    def reset_exposure_time(self, time_ms: float) -> float:
        """重置曝光时间

        Args:
            time_ms: 曝光时间（毫秒）

        Returns:
            实际设置的曝光时间
        """
        ...


class CalibrationMethod(Enum):
    """标定方法枚举"""

    BLAZED_GRATING = "blazed_grating"  # 闪耀光栅法
    INTERFEROMETER = "interferometer"  # 干涉法
    DIFFRACTION_EFFICIENCY = "diffraction_efficiency"  # 衍射效率法
    TWIN_BEAM = "twin_beam"  # 双光束干涉法


class AutoExposureController:
    """自动曝光控制器

    用于在标定过程中自动调整相机曝光时间，确保测量在合理范围内。

    工作原理：
    1. 初始拍摄一张图像，检测最大灰度值
    2. 如果饱和（>250），降低曝光时间
    3. 如果信号过弱（<50），增加曝光时间
    4. 重复直到达到目标范围
    """

    def __init__(
        self,
        camera: CameraWithExposureProtocol,
        target_min: int = 80,
        target_max: int = 220,
        min_exposure: float = 1.0,
        max_exposure: float = 1000.0,
        max_iterations: int = 5,
    ):
        """初始化自动曝光控制器

        Args:
            camera: 相机实例
            target_min: 目标最小灰度值
            target_max: 目标最大灰度值
            min_exposure: 最小曝光时间（ms）
            max_exposure: 最大曝光时间（ms）
            max_iterations: 最大迭代次数
        """
        self.camera = camera
        self.target_min = target_min
        self.target_max = target_max
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.max_iterations = max_iterations

        self._current_exposure = getattr(camera, "exposure_time_ms", 50.0)

    def auto_adjust(self, n_samples: int = 3) -> float:
        """自动调整曝光时间

        Args:
            n_samples: 采样次数

        Returns:
            最终设置的曝光时间
        """
        logger.info("开始自动曝光调整...")

        for iteration in range(self.max_iterations):
            # 获取图像
            img = self.camera.get_numpy_image(n_sample=n_samples)
            max_val = np.max(img)
            mean_val = np.mean(img)

            logger.debug(
                f"迭代 {iteration + 1}: 最大灰度={max_val:.0f}, 平均灰度={mean_val:.1f}"
            )

            # 检查是否在目标范围内
            if self.target_min <= max_val <= self.target_max:
                logger.info(
                    f"曝光调整完成: 曝光时间={self._current_exposure}ms, "
                    f"最大灰度={max_val:.0f}"
                )
                return self._current_exposure

            # 计算新的曝光时间
            if max_val > self.target_max:
                # 饱和，降低曝光
                ratio = self.target_max / max_val
                new_exposure = float(
                    self._current_exposure * ratio * 0.8
                )  # 额外降低20%确保安全
            elif max_val < self.target_min:
                # 信号过弱，增加曝光
                ratio = self.target_min / max_val if max_val > 0 else 2.0
                new_exposure = float(
                    self._current_exposure * ratio * 1.2
                )  # 额外增加20%确保足够
            else:
                break

            # 限制曝光时间范围
            new_exposure = max(self.min_exposure, min(self.max_exposure, new_exposure))

            if new_exposure == self._current_exposure:
                logger.warning("曝光时间已达到极限值")
                break

            # 应用新的曝光时间
            if hasattr(self.camera, "reset_exposure_time"):
                self._current_exposure = self.camera.reset_exposure_time(new_exposure)
            else:
                logger.warning("相机不支持曝光时间调整")
                break

            time.sleep(0.1)  # 等待相机稳定

        logger.warning(
            f"自动曝光调整达到最大迭代次数，当前曝光时间={self._current_exposure}ms"
        )
        return self._current_exposure

    def get_optimal_exposure_for_signal(self, signal_func, *args, **kwargs) -> float:
        """针对特定信号获取最佳曝光时间

        Args:
            signal_func: 获取信号的函数
            *args, **kwargs: 传递给signal_func的参数

        Returns:
            最佳曝光时间
        """
        logger.info("为特定信号调整曝光...")

        for iteration in range(self.max_iterations):
            # 获取信号
            signal = signal_func(*args, **kwargs)

            if isinstance(signal, (int, float)):
                signal_value = signal
            else:
                signal_value = np.max(signal) if hasattr(signal, "__len__") else signal

            # 检查范围
            if self.target_min * 2 <= signal_value <= self.target_max * 2:
                break

            # 调整曝光
            if signal_value > self.target_max * 2:
                ratio = (self.target_max * 2) / signal_value
                new_exposure = float(self._current_exposure * ratio * 0.8)
            elif signal_value < self.target_min * 2:
                ratio = (
                    (self.target_min * 2) / signal_value if signal_value > 0 else 2.0
                )
                new_exposure = float(self._current_exposure * ratio * 1.2)
            else:
                break

            new_exposure = max(self.min_exposure, min(self.max_exposure, new_exposure))

            if hasattr(self.camera, "reset_exposure_time"):
                self._current_exposure = self.camera.reset_exposure_time(new_exposure)

            time.sleep(0.1)

        return self._current_exposure


class SLMCalibratorBase(ABC):
    """SLM标定器基类

    使用闪耀光栅法进行SLM相位-灰度响应标定。

    标定流程：
    1. 在SLM上显示不同灰度深度的闪耀光栅
    2. 用相机测量衍射光斑强度
    3. 找到最大衍射效率对应的灰度值
    4. 该灰度值即为2π相位对应的灰度值
    """

    def __init__(
        self,
        slm: SLMProtocol,
        camera: CameraWithExposureProtocol,
        grating_period: int = 8,
        roi_center: tuple[int, int] | None = None,
        roi_size: tuple[int, int] = (100, 100),
    ):
        """初始化标定器

        Args:
            slm: SLM设备实例
            camera: 相机设备实例
            grating_period: 闪耀光栅周期（像素），默认为8
            roi_center: 感兴趣区域中心坐标，默认为图像中心
            roi_size: 感兴趣区域大小 (width, height)
        """
        self.slm = slm
        self.camera = camera
        self.grating_period = grating_period
        self.roi_size = roi_size

        # ROI中心默认为图像中心
        if roi_center is None:
            # 假设相机图像中心，实际使用时需要根据相机图像设置
            self.roi_center = None  # 延迟设置
        else:
            self.roi_center = roi_center

        # 标定结果
        self._result: CalibrationResult | None = None

    @property
    def result(self) -> CalibrationResult | None:
        """获取标定结果"""
        return self._result

    def create_blazed_grating(
        self, grayscale_depth: int, direction: str = "horizontal"
    ) -> np.ndarray:
        """创建闪耀光栅相位图

        创建一个线性相位梯度的闪耀光栅图案。

        Args:
            grayscale_depth: 相位深度（灰度值0-1023）
            direction: 光栅方向，'horizontal' 或 'vertical'

        Returns:
            相位图数组（uint16）
        """
        width = self.slm.width if hasattr(self.slm, "width") else 1920
        height = self.slm.height if hasattr(self.slm, "height") else 1080

        # 创建坐标网格
        if direction == "horizontal":
            # 水平方向光栅（相位沿x方向变化）
            x = np.arange(width)
            phase = (x / self.grating_period) * grayscale_depth
            phase = np.tile(phase, (height, 1))
        else:
            # 垂直方向光栅
            y = np.arange(height)
            phase = (y / self.grating_period) * grayscale_depth
            phase = np.tile(phase.reshape(-1, 1), (1, width))

        # 取模并转换为灰度值
        phase = np.mod(phase, grayscale_depth + 1)
        phase = np.clip(phase, 0, 1023).astype(np.uint16)

        return phase

    def measure_diffraction_efficiency(
        self, grayscale_depth: int, n_samples: int = 3, memory_number: int = 1
    ) -> float:
        """测量指定灰度深度的衍射效率

        Args:
            grayscale_depth: 相位深度（灰度值）
            n_samples: 采样次数
            memory_number: SLM内存编号

        Returns:
            衍射光斑的平均强度
        """
        # 创建并显示闪耀光栅
        grating = self.create_blazed_grating(grayscale_depth)
        self.slm.display_data(grating, memory_number=memory_number)

        # 等待SLM响应
        time.sleep(0.1)

        # 获取图像
        img = self.camera.get_numpy_image(n_sample=n_samples)

        # 计算ROI内的平均强度
        intensity = self._calculate_roi_intensity(img)

        return intensity

    def measure_zero_order_ratio(
        self, grayscale_depth: int, n_samples: int = 3, memory_number: int = 1
    ) -> float:
        """测量零级光强比值作为衍射效率

        通过比较全0相位（无光栅）和光栅相位图案下的零级光强，
        计算衍射效率。这种方法可以消除光源功率波动的影响。

        原理：
        - 全0相位时，光直接通过（零级）
        - 光栅相位时，部分光被衍射到一级，零级光强降低
        - 衍射效率 = I(光栅) / I(全0)

        Args:
            grayscale_depth: 相位深度（灰度值）
            n_samples: 采样次数
            memory_number: SLM内存编号

        Returns:
            零级光强比值（0-1之间）
        """
        # 测量全0相位时的零级光强（作为参考）
        self.slm.set_grayscale(0)
        time.sleep(0.1)
        ref_img = self.camera.get_numpy_image(n_sample=n_samples)
        ref_intensity = self._calculate_roi_intensity(ref_img)

        # 测量光栅相位时的零级光强
        grating = self.create_blazed_grating(grayscale_depth)
        self.slm.display_data(grating, memory_number=memory_number)
        time.sleep(0.1)

        grating_img = self.camera.get_numpy_image(n_sample=n_samples)
        grating_intensity = self._calculate_roi_intensity(grating_img)

        # 计算比值
        if ref_intensity > 0:
            ratio = grating_intensity / ref_intensity
        else:
            logger.warning("参考光强为零，无法计算比值")
            ratio = 0.0

        return ratio

    def _calculate_roi_intensity(self, img: np.ndarray) -> float:
        """计算ROI区域内的平均强度

        Args:
            img: 图像数组

        Returns:
            ROI内平均强度
        """
        if self.roi_center is None:
            # 自动寻找最亮点作为ROI中心
            center = np.unravel_index(np.argmax(img), img.shape)
            self.roi_center = (int(center[1]), int(center[0]))  # (x, y)
            logger.info(f"自动检测到光斑中心: {self.roi_center}")

        cx, cy = self.roi_center
        w, h = self.roi_size

        # 计算ROI边界
        x1 = max(0, cx - w // 2)
        x2 = min(img.shape[1], cx + w // 2)
        y1 = max(0, cy - h // 2)
        y2 = min(img.shape[0], cy + h // 2)

        # 提取ROI并计算平均强度
        roi = img[y1:y2, x1:x2]
        return float(np.mean(roi))

    def calibrate(
        self,
        grayscale_range: tuple[int, int] = (100, 1023),
        step: int = 10,
        n_samples: int = 3,
        fine_search: bool = True,
        fine_step: int = 2,
        fine_range: int = 50,
    ) -> CalibrationResult:
        """执行标定

        扫描灰度值范围，找到最大衍射效率对应的灰度值。

        Args:
            grayscale_range: 灰度值扫描范围 (min, max)
            step: 扫描步长
            n_samples: 每个点的采样次数
            fine_search: 是否进行精细搜索
            fine_step: 精细搜索步长
            fine_range: 精细搜索范围（在粗搜索结果两侧）

        Returns:
            CalibrationResult: 标定结果
        """
        logger.info("开始SLM标定...")

        # 粗搜索
        grayscale_values = lrange(grayscale_range[0], grayscale_range[1] + 1, step)
        intensities = []

        logger.info(f"粗搜索范围: {grayscale_range}, 步长: {step}")

        for gs in grayscale_values:
            intensity = self.measure_diffraction_efficiency(gs, n_samples)
            intensities.append(intensity)
            logger.debug(f"灰度值: {gs}, 强度: {intensity:.2f}")

        intensities = np.array(intensities)

        # 找到最大值
        max_idx = np.argmax(intensities)
        best_grayscale = grayscale_values[max_idx]
        logger.info(f"粗搜索结果: 最佳灰度值 = {best_grayscale}")

        # 精细搜索
        if fine_search:
            fine_min = max(grayscale_range[0], best_grayscale - fine_range)
            fine_max = min(grayscale_range[1], best_grayscale + fine_range)

            fine_grayscale_values = lrange(fine_min, fine_max + 1, fine_step)
            fine_intensities = []

            logger.info(f"精细搜索范围: ({fine_min}, {fine_max}), 步长: {fine_step}")

            for gs in fine_grayscale_values:
                intensity = self.measure_diffraction_efficiency(gs, n_samples)
                fine_intensities.append(intensity)
                logger.debug(f"精细搜索 - 灰度值: {gs}, 强度: {intensity:.2f}")

            fine_intensities = np.array(fine_intensities)

            # 合并结果
            all_grayscale_values = np.concatenate(
                [grayscale_values, fine_grayscale_values]
            )
            all_intensities = np.concatenate([intensities, fine_intensities])

            # 找到最终最佳值
            final_max_idx = np.argmax(all_intensities)
            best_grayscale = int(all_grayscale_values[final_max_idx])

            # 排序以便保存
            sort_idx = np.argsort(all_grayscale_values)
            grayscale_values = all_grayscale_values[sort_idx].tolist()
            intensities = all_intensities[sort_idx]

        logger.info(f"标定完成: 2π相位对应灰度值 = {best_grayscale}")

        # 创建标定结果
        self._result = CalibrationResult(
            grayscale_2pi=best_grayscale,
            grayscale_values=grayscale_values,
            intensities=intensities,
            wavelength_nm=self.slm.wavelength if hasattr(self.slm, "wavelength") else 0,
            slm_model=self._get_slm_model(),
            timestamp=datetime.now().isoformat(),
            metadata={
                "grating_period": self.grating_period,
                "roi_center": self.roi_center,
                "roi_size": self.roi_size,
                "n_samples": n_samples,
            },
        )

        return self._result

    @abstractmethod
    def _get_slm_model(self) -> str:
        """获取SLM型号"""
        pass

    def save_calibration(self, filepath: Union[str, Path]) -> None:
        """保存标定结果

        Args:
            filepath: 保存路径
        """
        if self._result is None:
            raise RuntimeError("没有可保存的标定结果，请先执行标定")
        self._result.save(filepath)

    def load_calibration(self, filepath: Union[str, Path]) -> CalibrationResult:
        """加载标定结果

        Args:
            filepath: 文件路径

        Returns:
            CalibrationResult: 标定结果
        """
        self._result = CalibrationResult.load(filepath)
        return self._result


class SantecCalibrator(SLMCalibratorBase):
    """Santec SLM-200 专用标定器

    针对Santec SLM-200的闪耀光栅标定实现。
    """

    def __init__(
        self,
        slm,  # Santec实例
        camera: CameraWithExposureProtocol,
        grating_period: int = 8,
        roi_center: tuple[int, int] | None = None,
        roi_size: tuple[int, int] = (100, 100),
    ):
        """初始化Santec SLM-200标定器

        Args:
            slm: Santec实例
            camera: 相机设备实例
            grating_period: 闪耀光栅周期（像素）
            roi_center: ROI中心坐标
            roi_size: ROI大小
        """
        super().__init__(
            slm=slm,
            camera=camera,
            grating_period=grating_period,
            roi_center=roi_center,
            roi_size=roi_size,
        )

        # Santec SLM-200的分辨率
        self._width = 1920
        self._height = 1080

    def create_blazed_grating(
        self, grayscale_depth: int, direction: str = "horizontal"
    ) -> np.ndarray:
        """创建闪耀光栅相位图（Santec SLM-200专用）

        Args:
            grayscale_depth: 相位深度（灰度值0-1023）
            direction: 光栅方向

        Returns:
            相位图数组（uint16），shape为(1080, 1920)
        """
        # Santec SLM-200分辨率: 1920x1080
        width, height = self._width, self._height

        # 创建坐标网格
        x = np.arange(width)

        if direction == "horizontal":
            # 水平方向光栅
            phase = (x / self.grating_period) * grayscale_depth
            phase = np.tile(phase, (height, 1))
        else:
            # 垂直方向光栅
            y = np.arange(height)
            phase = (y / self.grating_period) * grayscale_depth
            phase = np.tile(phase.reshape(-1, 1), (1, width))

        # 取模并转换为灰度值
        phase = np.mod(phase, grayscale_depth + 1)
        phase = np.clip(phase, 0, 1023).astype(np.uint16)

        return phase

    def calibrate_with_background(
        self,
        grayscale_range: tuple[int, int] = (100, 1023),
        step: int = 10,
        n_samples: int = 3,
        fine_search: bool = True,
        fine_step: int = 2,
        fine_range: int = 50,
        measure_background: bool = True,
    ) -> CalibrationResult:
        """带背景测量的标定

        先测量背景光强，然后从衍射效率中扣除背景。

        Args:
            grayscale_range: 灰度值扫描范围
            step: 扫描步长
            n_samples: 采样次数
            fine_search: 是否精细搜索
            fine_step: 精细搜索步长
            fine_range: 精细搜索范围
            measure_background: 是否测量背景

        Returns:
            CalibrationResult: 标定结果
        """
        # 测量背景（显示均匀灰度0）
        background_intensity = 0.0
        if measure_background:
            logger.info("测量背景光强...")
            self.slm.set_grayscale(0)
            time.sleep(0.2)
            bg_img = self.camera.get_numpy_image(n_sample=n_samples)
            background_intensity = self._calculate_roi_intensity(bg_img)
            logger.info(f"背景强度: {background_intensity:.2f}")

        # 执行标定
        result = self.calibrate(
            grayscale_range=grayscale_range,
            step=step,
            n_samples=n_samples,
            fine_search=fine_search,
            fine_step=fine_step,
            fine_range=fine_range,
        )

        # 扣除背景
        if measure_background and background_intensity > 0:
            corrected_intensities = result.intensities - background_intensity
            corrected_intensities = np.maximum(corrected_intensities, 0)

            # 更新结果
            result.intensities = corrected_intensities
            result.metadata["background_intensity"] = background_intensity

        return result

    def calibrate_with_zero_order_ratio(
        self,
        grayscale_range: tuple[int, int] = (100, 1023),
        step: int = 10,
        n_samples: int = 3,
        fine_search: bool = True,
        fine_step: int = 2,
        fine_range: int = 50,
    ) -> CalibrationResult:
        """使用零级光强比值法标定

        通过比较全0相位和光栅相位下的零级光强比值来确定2π相位。
        这种方法可以消除光源功率波动的影响，测量更稳定。

        原理：
        - 全0相位时：零级光强 = I₀（参考）
        - 光栅相位时：零级光强 = I₁
        - 当相位深度 = 2π时，一级衍射效率最高，零级光强最低
        - 通过扫描找到零级光强最低的点，即为2π相位

        Args:
            grayscale_range: 灰度值扫描范围
            step: 扫描步长
            n_samples: 采样次数
            fine_search: 是否精细搜索
            fine_step: 精细搜索步长
            fine_range: 精细搜索范围

        Returns:
            CalibrationResult: 标定结果
        """
        logger.info("开始零级光强比值法标定...")

        # 先测量一次全0相位的参考光强
        logger.info("测量全0相位参考光强...")
        self.slm.set_grayscale(0)
        time.sleep(0.2)
        ref_imgs = [self.camera.get_numpy_image(n_sample=n_samples) for _ in range(3)]
        ref_intensity = np.mean(
            [self._calculate_roi_intensity(img) for img in ref_imgs]
        )
        logger.info(f"参考光强 (全0相位): {ref_intensity:.2f}")

        # 粗搜索
        grayscale_values = lrange(grayscale_range[0], grayscale_range[1] + 1, step)
        ratios = []

        logger.info(f"粗搜索范围: {grayscale_range}, 步长: {step}")

        for gs in grayscale_values:
            ratio = self.measure_zero_order_ratio(gs, n_samples)
            ratios.append(ratio)
            logger.debug(f"灰度值: {gs}, 零级比值: {ratio:.4f}")

        ratios = np.array(ratios)

        # 找到最小值（零级光强最低点，即衍射效率最高）
        min_idx = np.argmin(ratios)
        best_grayscale = grayscale_values[min_idx]
        logger.info(
            f"粗搜索结果: 最佳灰度值 = {best_grayscale}, 最小比值 = {ratios[min_idx]:.4f}"
        )

        # 精细搜索
        if fine_search:
            fine_min = max(grayscale_range[0], best_grayscale - fine_range)
            fine_max = min(grayscale_range[1], best_grayscale + fine_range)

            fine_grayscale_values = lrange(fine_min, fine_max + 1, fine_step)
            fine_ratios = []

            logger.info(f"精细搜索范围: ({fine_min}, {fine_max}), 步长: {fine_step}")

            for gs in fine_grayscale_values:
                ratio = self.measure_zero_order_ratio(gs, n_samples)
                fine_ratios.append(ratio)
                logger.debug(f"精细搜索 - 灰度值: {gs}, 零级比值: {ratio:.4f}")

            fine_ratios = np.array(fine_ratios)

            # 合并结果
            all_grayscale_values = np.concatenate(
                [grayscale_values, fine_grayscale_values]
            )
            all_ratios = np.concatenate([ratios, fine_ratios])

            # 找到最终最小值
            final_min_idx = np.argmin(all_ratios)
            best_grayscale = int(all_grayscale_values[final_min_idx])

            # 排序以便保存
            sort_idx = np.argsort(all_grayscale_values)
            grayscale_values = all_grayscale_values[sort_idx].tolist()
            ratios = all_ratios[sort_idx]

        logger.info(f"零级光强比值法标定完成: 2π相位对应灰度值 = {best_grayscale}")

        # 创建标定结果
        self._result = CalibrationResult(
            grayscale_2pi=best_grayscale,
            grayscale_values=grayscale_values,
            intensities=ratios,
            wavelength_nm=self.slm.wavelength if hasattr(self.slm, "wavelength") else 0,
            slm_model=self._get_slm_model(),
            timestamp=datetime.now().isoformat(),
            metadata={
                "method": "zero_order_ratio",
                "reference_intensity": ref_intensity,
                "grating_period": self.grating_period,
                "roi_center": self.roi_center,
                "roi_size": self.roi_size,
                "n_samples": n_samples,
            },
        )

        return self._result

    def _get_slm_model(self) -> str:
        """获取SLM型号"""
        return "Santec SLM-200"

    def calibrate_with_auto_exposure(
        self,
        grayscale_range: tuple[int, int] = (100, 1023),
        step: int = 10,
        n_samples: int = 3,
        fine_search: bool = True,
        fine_step: int = 2,
        fine_range: int = 50,
        measure_background: bool = True,
        auto_exposure: bool = True,
        target_min: int = 80,
        target_max: int = 220,
    ) -> CalibrationResult:
        """带自动曝光的标定

        在标定开始前自动调整相机曝光时间，确保测量在合理范围内。

        Args:
            grayscale_range: 灰度值扫描范围
            step: 扫描步长
            n_samples: 采样次数
            fine_search: 是否精细搜索
            fine_step: 精细搜索步长
            fine_range: 精细搜索范围
            measure_background: 是否测量背景
            auto_exposure: 是否启用自动曝光
            target_min: 自动曝光目标最小值
            target_max: 自动曝光目标最大值

        Returns:
            CalibrationResult: 标定结果
        """
        # 自动曝光调整
        if auto_exposure and hasattr(self.camera, "reset_exposure_time"):
            logger.info("启用自动曝光调整...")
            auto_expo_ctrl = AutoExposureController(
                camera=self.camera, target_min=target_min, target_max=target_max
            )
            # 先显示一个测试图案来调整曝光
            test_grating = self.create_blazed_grating(512)  # 中等灰度值测试
            self.slm.display_data(test_grating, memory_number=1)
            time.sleep(0.2)

            optimal_exposure = auto_expo_ctrl.auto_adjust(n_samples=3)
            logger.info(f"自动曝光调整完成，最佳曝光时间: {optimal_exposure}ms")

        # 执行带背景测量的标定
        return self.calibrate_with_background(
            grayscale_range=grayscale_range,
            step=step,
            n_samples=n_samples,
            fine_search=fine_search,
            fine_step=fine_step,
            fine_range=fine_range,
            measure_background=measure_background,
        )


class InterferometerCalibrator(SLMCalibratorBase):
    """干涉法标定器

    使用干涉图样测量SLM的相位-灰度响应。

    原理：
    1. 将SLM输出光与参考光进行干涉
    2. 在SLM上显示不同灰度值的均匀图案
    3. 观察干涉条纹的移动或相位变化
    4. 从干涉条纹变化计算相位-灰度响应

    注意：此方法需要马赫-泽德干涉仪或类似干涉装置。
    """

    def __init__(
        self,
        slm: SLMProtocol,
        camera: CameraWithExposureProtocol,
        roi_center: tuple[int, int] | None = None,
        roi_size: tuple[int, int] = (100, 100),
    ):
        """初始化干涉法标定器

        Args:
            slm: SLM设备实例
            camera: 相机设备实例
            roi_center: ROI中心坐标
            roi_size: ROI大小
        """
        super().__init__(
            slm=slm,
            camera=camera,
            grating_period=1,  # 不使用光栅
            roi_center=roi_center,
            roi_size=roi_size,
        )

    def measure_phase_from_interference(
        self, grayscale: int, reference_grayscale: int = 0, n_samples: int = 3
    ) -> float:
        """从干涉图样测量相位变化

        Args:
            grayscale: 当前灰度值
            reference_grayscale: 参考灰度值
            n_samples: 采样次数

        Returns:
            相位变化（弧度）
        """
        # 设置SLM为测试灰度值
        self.slm.set_grayscale(grayscale)
        time.sleep(0.1)
        img1 = self.camera.get_numpy_image(n_sample=n_samples)

        # 设置SLM为参考灰度值
        self.slm.set_grayscale(reference_grayscale)
        time.sleep(0.1)
        img2 = self.camera.get_numpy_image(n_sample=n_samples)

        # 计算相位差（简化版，实际需要更复杂的相位提取算法）
        # 这里使用强度差分作为简化
        diff = np.mean(img1) - np.mean(img2)

        # 转换为相位（假设线性响应）
        return float(diff * np.pi / 128.0)  # 简化转换

    def calibrate(
        self,
        grayscale_range: tuple[int, int] = (0, 1023),
        step: int = 32,
        n_samples: int = 3,
        reference_grayscale: int = 0,
    ) -> CalibrationResult:
        """执行干涉法标定

        Args:
            grayscale_range: 灰度值扫描范围
            step: 扫描步长
            n_samples: 采样次数
            reference_grayscale: 参考灰度值

        Returns:
            CalibrationResult: 标定结果
        """
        logger.info("开始干涉法标定...")

        grayscale_values = lrange(grayscale_range[0], grayscale_range[1] + 1, step)
        phase_values = []

        for gs in grayscale_values:
            phase = self.measure_phase_from_interference(
                gs, reference_grayscale, n_samples
            )
            phase_values.append(phase)
            logger.debug(f"灰度值: {gs}, 相位: {phase:.4f} rad")

        phase_values = np.array(phase_values)

        # 找到2π相位对应的灰度值
        # 拟合相位-灰度曲线，找到斜率
        valid_idx = phase_values > 0
        if np.sum(valid_idx) > 1:
            valid_gs = np.array(grayscale_values)[valid_idx]
            valid_phase = phase_values[valid_idx]
            slope = np.polyfit(valid_gs, valid_phase, 1)[0]

            if slope > 0:
                grayscale_2pi = int(2 * np.pi / slope)
            else:
                grayscale_2pi = int(grayscale_values[-1])
        else:
            grayscale_2pi = int(grayscale_values[-1])

        logger.info(f"干涉法标定完成: 2π相位对应灰度值 = {grayscale_2pi}")

        self._result = CalibrationResult(
            grayscale_2pi=min(grayscale_2pi, 1023),
            grayscale_values=grayscale_values,
            intensities=phase_values,
            wavelength_nm=self.slm.wavelength if hasattr(self.slm, "wavelength") else 0,
            slm_model=self._get_slm_model(),
            timestamp=datetime.now().isoformat(),
            metadata={
                "method": "interferometer",
                "roi_center": self.roi_center,
                "roi_size": self.roi_size,
            },
        )

        return self._result

    def _get_slm_model(self) -> str:
        """获取SLM型号"""
        return "SLM (Interferometer)"


class DiffractionEfficiencyCalibrator(SLMCalibratorBase):
    """衍射效率法标定器

    通过测量不同灰度值下的一级衍射效率来确定相位响应。

    原理：
    1. 在SLM上显示周期性光栅结构
    2. 测量一级衍射光的强度
    3. 根据衍射效率与相位深度的关系（贝塞尔函数）
    4. 拟合得到相位-灰度响应曲线

    衍射效率公式：η₁ = (2π * J₁(φ) / φ)²
    其中φ是相位深度，J₁是第一类贝塞尔函数
    """

    def __init__(
        self,
        slm: SLMProtocol,
        camera: CameraWithExposureProtocol,
        grating_period: int = 16,
        roi_center: tuple[int, int] | None = None,
        roi_size: tuple[int, int] = (50, 50),
    ):
        """初始化衍射效率法标定器

        Args:
            slm: SLM设备实例
            camera: 相机设备实例
            grating_period: 光栅周期（像素）
            roi_center: 一级衍射光斑中心坐标
            roi_size: ROI大小
        """
        super().__init__(
            slm=slm,
            camera=camera,
            grating_period=grating_period,
            roi_center=roi_center,
            roi_size=roi_size,
        )

    def _calculate_diffraction_efficiency(
        self, phase_depth: float, order: int = 1
    ) -> float:
        """计算理论衍射效率

        使用贝塞尔函数计算一级衍射效率。

        Args:
            phase_depth: 相位深度（弧度）
            order: 衍射级次

        Returns:
            衍射效率
        """
        from scipy.special import jn

        if abs(phase_depth) < 1e-10:
            return 0.0

        # 一级衍射效率
        j_val = jn(order, phase_depth)
        efficiency = (2 * j_val / phase_depth) ** 2
        return efficiency

    def calibrate(
        self,
        grayscale_range: tuple[int, int] = (50, 1023),
        step: int = 10,
        n_samples: int = 3,
        fine_search: bool = True,
        fine_step: int = 2,
        fine_range: int = 50,
    ) -> CalibrationResult:
        """执行衍射效率法标定

        Args:
            grayscale_range: 灰度值扫描范围
            step: 扫描步长
            n_samples: 采样次数
            fine_search: 是否精细搜索
            fine_step: 精细搜索步长
            fine_range: 精细搜索范围

        Returns:
            CalibrationResult: 标定结果
        """
        logger.info("开始衍射效率法标定...")

        # 测量零级（透射光）作为参考
        self.slm.set_grayscale(0)
        time.sleep(0.1)
        zero_order_img = self.camera.get_numpy_image(n_sample=n_samples)
        zero_order_intensity = self._calculate_roi_intensity(zero_order_img)
        logger.info(f"零级衍射强度: {zero_order_intensity:.2f}")

        # 粗搜索
        grayscale_values = lrange(grayscale_range[0], grayscale_range[1] + 1, step)
        efficiencies = []

        logger.info(f"粗搜索范围: {grayscale_range}, 步长: {step}")

        for gs in grayscale_values:
            intensity = self.measure_diffraction_efficiency(gs, n_samples)
            # 计算相对衍射效率
            if zero_order_intensity > 0:
                efficiency = intensity / zero_order_intensity
            else:
                efficiency = 0
            efficiencies.append(efficiency)
            logger.debug(f"灰度值: {gs}, 衍射效率: {efficiency:.4f}")

        efficiencies = np.array(efficiencies)

        # 找到最大效率点
        max_idx = np.argmax(efficiencies)
        best_grayscale = grayscale_values[max_idx]
        logger.info(f"粗搜索结果: 最佳灰度值 = {best_grayscale}")

        # 精细搜索
        if fine_search:
            fine_min = max(grayscale_range[0], best_grayscale - fine_range)
            fine_max = min(grayscale_range[1], best_grayscale + fine_range)

            fine_grayscale_values = lrange(fine_min, fine_max + 1, fine_step)
            fine_efficiencies = []

            for gs in fine_grayscale_values:
                intensity = self.measure_diffraction_efficiency(gs, n_samples)
                efficiency = (
                    intensity / zero_order_intensity if zero_order_intensity > 0 else 0
                )
                fine_efficiencies.append(efficiency)

            # 合并结果
            all_grayscale = np.concatenate([grayscale_values, fine_grayscale_values])
            all_efficiencies = np.concatenate([efficiencies, fine_efficiencies])

            final_max_idx = np.argmax(all_efficiencies)
            best_grayscale = int(all_grayscale[final_max_idx])

            # 排序
            sort_idx = np.argsort(all_grayscale)
            grayscale_values = all_grayscale[sort_idx].tolist()
            efficiencies = all_efficiencies[sort_idx]

        logger.info(f"衍射效率法标定完成: 2π相位对应灰度值 = {best_grayscale}")

        self._result = CalibrationResult(
            grayscale_2pi=best_grayscale,
            grayscale_values=grayscale_values,
            intensities=efficiencies,
            wavelength_nm=self.slm.wavelength if hasattr(self.slm, "wavelength") else 0,
            slm_model=self._get_slm_model(),
            timestamp=datetime.now().isoformat(),
            metadata={
                "method": "diffraction_efficiency",
                "grating_period": self.grating_period,
                "zero_order_intensity": zero_order_intensity,
                "roi_center": self.roi_center,
                "roi_size": self.roi_size,
            },
        )

        return self._result

    def _get_slm_model(self) -> str:
        """获取SLM型号"""
        return "SLM (Diffraction Efficiency)"


def create_calibration_curve(
    grayscale_values: np.ndarray, intensities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """创建标定曲线

    对标定数据进行多项式拟合，生成平滑的标定曲线。

    Args:
        grayscale_values: 灰度值数组
        intensities: 强度数组

    Returns:
        Tuple[np.ndarray, np.ndarray]: (拟合后的灰度值, 拟合后的强度)
    """
    # 使用样条插值平滑曲线（如果scipy可用）

    spline = UnivariateSpline(
        grayscale_values, intensities, s=len(grayscale_values) * 10
    )
    fit_intensities = np.array(spline(grayscale_values))

    return grayscale_values, fit_intensities


def plot_calibration_result(
    result: CalibrationResult, save_path: Path | None = None
) -> None:
    """绘制标定结果曲线

    Args:
        result: 标定结果
        save_path: 图片保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制原始数据
        ax.plot(
            result.grayscale_values,
            result.intensities,
            "b-",
            label="衍射效率",
            linewidth=2,
        )

        # 标记2π点
        ax.axvline(
            x=result.grayscale_2pi,
            color="r",
            linestyle="--",
            label=f"2π相位 = {result.grayscale_2pi}",
        )

        ax.set_xlabel("灰度值", fontsize=12)
        ax.set_ylabel("衍射光强 (a.u.)", fontsize=12)
        ax.set_title(
            f"SLM标定结果 - {result.slm_model}\n波长: {result.wavelength_nm}nm",
            fontsize=14,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"标定曲线已保存到: {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib未安装，无法绘制标定曲线")


# 便捷函数
def calibrate_santec(
    slm,
    camera,
    wavelength: int = 1064,
    grating_period: int = 8,
    output_dir: Path | None = None,
) -> CalibrationResult:
    """Santec SLM-200 快速标定函数

    Args:
        slm: Santec实例
        camera: 相机实例
        wavelength: 工作波长（nm）
        grating_period: 光栅周期
        output_dir: 输出目录

    Returns:
        CalibrationResult: 标定结果
    """
    calibrator = SantecCalibrator(
        slm=slm, camera=camera, grating_period=grating_period
    )

    result = calibrator.calibrate_with_background()

    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result.save(output_dir / f"slm_calibration_{wavelength}nm_{timestamp}.json")

    return result


# =====================================================================
# SLM defocus 平移标定 (shift_x / shift_y) — WFS tip/tilt 零点法
# =====================================================================
def diagnose_radius(
    slm: Santec,
    wfs: ThorlabWFS,
    ph: PatternHelper,
    radii: tuple[float, ...],
    amplitude: float,
    n_avg: int,
) -> list[dict]:
    """扫描 Zernike 半径, 找 WFS defocus 响应最大者 (≈ 光束半径)."""
    results: list[dict] = []
    for r in radii:
        phase = make_phase(ph, {(2, 0): amplitude}, r, n_max=5)
        slm.set_shift(0, 0)
        slm.display_phase(phase, wait_time_s=0.5)
        time.sleep(0.25)
        zt, zd = measure_tilt_defocus(wfs, n_avg=n_avg)
        rec = {"radius": r, "amplitude": amplitude,
               "defocus": None if zd is None else zd}
        if zt is not None:
            rec.update({"tip": float(zt[0]), "tilt": float(zt[1])})
        results.append(rec)
        click.echo(
            f"  R={r:6.0f}px A={amplitude:5.1f} → defocus="
            f"{'n/a' if zd is None else f'{zd:+.4f}'}λ"
        )
    valid = [r for r in results if r["defocus"] is not None]
    if valid:
        best = max(valid, key=lambda r: abs(r["defocus"]))
        click.echo(f"  → 最强响应 R={best['radius']:.0f}px (≈ 光束半径)")
    return results


@click.command()
@click.option("--slm-number", type=int, default=1, show_default=True, help="SLM 设备编号")
@click.option("--slm-wavelength", type=int, default=532, show_default=True, help="SLM 波长 nm")
@click.option("--wfs-exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS,
              show_default=True, help=f"WFS 曝光 ms (必须 ≤ {MAX_EXPOSURE_MS})")
@click.option("--zernike-radius", type=float, default=600.0, show_default=True,
              help="defocus Zernike 半径 px (必须 > 光束半径, 默认 600)")
@click.option("--defocus-a", type=float, default=20.0, show_default=True,
              help="defocus 幅度 rad (太小则响应淹没在噪声中)")
@click.option("--shift-limit", type=int, default=DEFAULT_SHIFT_LIMIT, show_default=True,
              help="shift 绝对值上限 (防 defocus 盘推出面板)")
@click.option("--coarse-step", type=int, default=100, show_default=True, help="粗扫步长 px")
@click.option("--coarse-half", type=int, default=300, show_default=True, help="粗扫半宽 px")
@click.option("--fine-half", type=int, default=40, show_default=True, help="细扫半宽 px")
@click.option("--iterations", type=int, default=2, show_default=True, help="x/y 迭代轮数")
@click.option("--n-avg-scan", type=int, default=3, show_default=True, help="扫描帧平均次数")
@click.option("--n-avg-verify", type=int, default=5, show_default=True, help="校验帧平均次数")
@click.option("--improve-ratio", type=float, default=0.5, show_default=True,
              help="质量门控: 附加倾斜需降低到该比例以下才写入 config")
@click.option("--radius-scan", is_flag=True, default=False,
              help="先诊断光束半径 (扫 Zernike R), 不做标定")
@click.option("--no-save", is_flag=True, default=False,
              help="只测量, 不写入 SLM config (dry run)")
@click.option("-o", "--output", default=str(DEFAULT_OUTPUT), show_default=True,
              help="标定报告 JSON 路径")
def main_shift_calib(
    slm_number: int,
    slm_wavelength: int,
    wfs_exposure_ms: float,
    zernike_radius: float,
    defocus_a: float,
    shift_limit: int,
    coarse_step: int,
    coarse_half: int,
    fine_half: int,
    iterations: int,
    n_avg_scan: int,
    n_avg_verify: int,
    improve_ratio: float,
    radius_scan: bool,
    no_save: bool,
    output: str,
) -> int:
    """SLM defocus 平移标定 — WFS tip/tilt 零点法."""
    if wfs_exposure_ms > MAX_EXPOSURE_MS:
        raise click.BadParameter(
            f"WFS 曝光 {wfs_exposure_ms}ms 超过安全上限 {MAX_EXPOSURE_MS}ms"
        )
    if zernike_radius <= 0:
        raise click.BadParameter("--zernike-radius 必须 > 0")

    click.echo("=" * 72)
    click.echo("[SLM defocus shift 标定] WFS tip/tilt 零点法")
    click.echo("=" * 72)

    slm = Santec(slm_number=slm_number, wavelength=slm_wavelength, video_mode=0)
    wfs = ThorlabWFS(exposure_time=wfs_exposure_ms, use_custom_ref=False)
    ph = PatternHelper(resolution=(PANEL_W, PANEL_H))

    report: dict = {
        "zernike_radius": zernike_radius,
        "defocus_a": defocus_a,
        "method": "minimize ||z_tilt(defocus@shift) - z_tilt(flat)||",
        "evals": [],
    }
    ok = False
    sx_star, sy_star = 0.0, 0.0
    original: tuple[int, int] | None = None

    try:
        slm.open()
        wl, max_gray = slm.get_wavelength_info()
        original = (slm.shift_x, slm.shift_y)
        report["original_shift"] = list(original)
        click.echo(f"[OK] SLM open: serial={slm._serial_number}, {wl}nm, "
                   f"2π gray={max_gray}, 当前 shift={original}")

        wfs.open()
        exp = float(wfs.exposure_time)
        click.echo(f"[OK] WFS open: serial={wfs.serial_num}, exposure={exp:.4f}ms")
        assert exp <= MAX_EXPOSURE_MS, f"曝光 {exp}ms 超过 {MAX_EXPOSURE_MS}ms"

        # pupil 必须自动获取并显式写回 (optimize_pupil 只计算不设置)
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
        click.echo(f"[OK] pupil: center=({cx:.3f},{cy:.3f})mm, "
                   f"diameter=({dx:.3f},{dy:.3f})mm")
        report["pupil"] = [cx, cy, dx, dy]

        flat = np.full((PANEL_H, PANEL_W), 0, dtype=np.uint16)

        if radius_scan:
            click.echo("\n[诊断] 扫描 Zernike 半径 → WFS defocus 响应")
            report["radius_diagnostic"] = diagnose_radius(
                slm, wfs, ph, (120.0, 200.0, 300.0, 450.0, 600.0), defocus_a, n_avg_scan
            )
            slm.display_data(flat, wait_time_s=0.5)
            _write_report(output, report)
            return 0

        # 基线: 纯平下的静态倾斜 (判据基准)
        slm.set_shift(0, 0)
        slm.display_data(flat, wait_time_s=0.5)
        time.sleep(0.3)
        base_tilt, base_def = measure_tilt_defocus(wfs, n_avg=n_avg_verify)
        if base_tilt is None:
            raise RuntimeError("基线测量失败: 无有效 zernike")
        click.echo(f"[BASE] 纯平: tip={base_tilt[0]:+.4f}λ tilt={base_tilt[1]:+.4f}λ "
                   f"(defocus={base_def:+.4f}λ) ← 判据为相对此值的附加倾斜")
        report["baseline"] = {"tip": float(base_tilt[0]), "tilt": float(base_tilt[1])}

        phase_rad = make_phase(ph, {(2, 0): defocus_a}, zernike_radius, n_max=5)
        click.echo(f"[INFO] defocus R={zernike_radius:.0f}px A={defocus_a}rad, "
                   f"range=[{phase_rad.min():.1f},{phase_rad.max():.1f}] rad")

        def evaluate(sx: int, sy: int, n_avg: int, tag: str) -> float | None:
            slm.set_shift(clamp_shift(sx, shift_limit), clamp_shift(sy, shift_limit))
            slm.display_phase(phase_rad, wait_time_s=0.5)
            time.sleep(0.25)
            zt, zd = measure_tilt_defocus(wfs, n_avg=n_avg)
            if zt is None:
                logger.warning("{} shift=({},{}) 无有效 zernike", tag, sx, sy)
                return None
            added = zt - base_tilt
            norm = float(np.linalg.norm(added))
            report["evals"].append({
                "tag": tag, "sx": int(sx), "sy": int(sy),
                "tip": float(zt[0]), "tilt": float(zt[1]),
                "added_tip": float(added[0]), "added_tilt": float(added[1]),
                "added_norm": norm, "defocus": zd,
            })
            click.echo(f"[EVAL] {tag:14s} shift=({sx:5d},{sy:5d}) "
                       f"Δtip={added[0]:+.4f} Δtilt={added[1]:+.4f} "
                       f"‖Δ‖={norm:.4f} defocus={zd:+.4f}")
            return norm

        def scan_axis(axis: str, other: int, values, tag: str):
            best_v, best_m, pts = None, np.inf, []
            for v in values:
                sx = int(v) if axis == "x" else other
                sy = int(v) if axis == "y" else other
                m = evaluate(sx, sy, n_avg_scan, tag)
                if m is None:
                    continue
                pts.append((float(v), m))
                if m < best_m:
                    best_m, best_v = m, float(v)
            return best_v, best_m, pts

        coarse = list(range(-coarse_half, coarse_half + 1, coarse_step))

        for it in range(1, iterations + 1):
            click.echo("\n" + "=" * 72)
            click.echo(f"[迭代 {it}/{iterations}]")
            click.echo("=" * 72)

            bx, bm, _ = scan_axis("x", clamp_shift(sy_star, shift_limit), coarse, f"it{it}-x-coarse")
            if bx is None:
                raise RuntimeError("X 粗扫无有效点")
            fine_x = [bx + d for d in np.linspace(-fine_half, fine_half, 5)]
            click.echo(f"[X 粗扫] 最优 sx={bx:.0f} (‖Δ‖={bm:.4f}) → 细扫 "
                       f"{[round(v) for v in fine_x]}")
            bx2, bm2, pts2 = scan_axis("x", clamp_shift(sy_star, shift_limit), fine_x,
                                       f"it{it}-x-fine")
            if bx2 is not None:
                px = parabolic_min(sorted(pts2))
                sx_star = float(clamp_shift(px if px is not None else bx2, shift_limit))
                click.echo(f"[X] → sx*={sx_star:.1f} (‖Δ‖={bm2:.4f})")

            by, bmy, _ = scan_axis("y", clamp_shift(sx_star, shift_limit), coarse, f"it{it}-y-coarse")
            if by is None:
                raise RuntimeError("Y 粗扫无有效点")
            fine_y = [by + d for d in np.linspace(-fine_half, fine_half, 5)]
            click.echo(f"[Y 粗扫] 最优 sy={by:.0f} (‖Δ‖={bmy:.4f}) → 细扫 "
                       f"{[round(v) for v in fine_y]}")
            by2, bmy2, ptsy2 = scan_axis("y", clamp_shift(sx_star, shift_limit), fine_y,
                                         f"it{it}-y-fine")
            if by2 is not None:
                py = parabolic_min(sorted(ptsy2))
                sy_star = float(clamp_shift(py if py is not None else by2, shift_limit))
                click.echo(f"[Y] → sy*={sy_star:.1f} (‖Δ‖={bmy2:.4f})")

        # 校验: 标定 shift vs (0,0)
        click.echo("\n" + "=" * 72)
        click.echo(f"[VERIFY] shift=({sx_star:.0f},{sy_star:.0f}) vs (0,0)")
        click.echo("=" * 72)
        m_star = evaluate(int(round(sx_star)), int(round(sy_star)), n_avg_verify, "verify")
        m_zero = evaluate(0, 0, n_avg_scan, "verify-zero")
        if m_star is None or m_zero is None:
            raise RuntimeError("校验测量失败")
        ratio = m_star / m_zero if m_zero > 0 else 1.0
        click.echo(f"[改善] ‖Δz_tilt‖ {m_zero:.4f}λ → {m_star:.4f}λ "
                   f"({100 * (1 - ratio):.1f}% 降低)")
        report["verify"] = {"shift": [sx_star, sy_star], "added_norm": m_star,
                            "zero_shift_added_norm": m_zero, "improvement_ratio": ratio}

        quality_ok = m_star < m_zero and ratio < improve_ratio
        report["quality_ok"] = bool(quality_ok)
        click.echo(f"[GATE] 比值={ratio:.3f} (阈值 {improve_ratio}) → "
                   f"{'通过' if quality_ok else '不通过'}")

        if quality_ok and not no_save:
            slm.set_shift(int(round(sx_star)), int(round(sy_star)))
            slm.save_config()
            click.echo(f"[OK] config 已写入: shift_x={slm.shift_x}, shift_y={slm.shift_y}")
            report["saved_shift"] = [slm.shift_x, slm.shift_y]
            ok = True
        elif quality_ok and no_save:
            click.echo(f"[INFO] --no-save: 标定通过但未写入 config "
                       f"(建议 shift=({sx_star:.0f},{sy_star:.0f}))")
            report["saved_shift"] = None
            # close() 会自动 save_config() → 必须先把 shift 还原, 否则 dry run 仍会改配置
            if original is not None:
                slm.set_shift(*original)
                click.echo(f"[INFO] --no-save: 已还原 shift={original} (防 close() 自动保存)")
            ok = True
        else:
            if original is not None:
                slm.set_shift(*original)
                slm.save_config()
                click.echo(f"[WARN] 质量门控不通过 → 恢复原始 shift={original}")
                report["saved_shift"] = list(original)

        slm.display_data(flat, wait_time_s=0.5)

    except AssertionError as e:
        click.echo(f"[FAIL] 断言失败: {e}")
    except Exception as e:
        logger.exception("标定失败")
        click.echo(f"[FAIL] {type(e).__name__}: {e}")
        # 异常路径也恢复原始 shift, 避免 close() 自动保存垃圾值
        if original is not None:
            try:
                slm.set_shift(*original)
                slm.save_config()
                click.echo(f"[INFO] 已恢复原始 shift={original}")
            except Exception as e2:
                logger.warning("恢复原始 shift 失败: {}", e2)
    finally:
        for dev, name in ((wfs, "WFS"), (slm, "SLM")):
            try:
                dev.close()
                click.echo(f"[INFO] {name} close")
            except Exception as e:
                logger.warning("{} close: {}", name, e)

    _write_report(output, report)
    click.echo("=" * 72)
    click.echo(f"[{'ALL PASS' if ok else 'SOME FAILURES'}] "
               f"final_shift=({sx_star:.0f},{sy_star:.0f})")
    click.echo("=" * 72)
    return 0 if ok else 1


def _write_report(output: str, report: dict) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"[INFO] 标定报告: {path}")


# =====================================================================
# 几何标定: 装配 -> 光束位置 -> K/旋转 -> 验证
# =====================================================================
class SLMCCDCalibrator:
    """SLM+CCD 几何标定工具. 硬件无关: 传入 slm/ccd 与 acquire 回调.

    Parameters
    ----------
    slm          具有 display_phase(radian phase) 的SLM驱动 (如 Santec)
    ccd          相机驱动
    acquire      callable(ccd) -> ndarray  采图回调 (建议含平均帧与超时保护)
    panel_res    (h, w) SLM面板分辨率, 如 Santec.Panel_Res
    settle_s     SLM显示后稳定等待时间
    """

    def __init__(self, slm: Santec, ccd: DahengCamera, settle_s: float = SETTLE_S):
        self.slm = slm
        self.ccd = ccd
        self.panel_res = slm.Panel_Res[::-1]
        self.settle_s = settle_s
        self.calib: dict | None = None

    # ------------------------------------------------------------ 底层工具
    @staticmethod
    def _moments(img: np.ndarray, exclude=None, thresh_frac: float = 0.15):
        img = np.asarray(img, np.float64)
        if exclude is not None:
            (cy, cx), r = exclude
            yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
            img = img.copy()
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0.0
        thr = img.max() * thresh_frac
        m = img >= thr
        if int(m.sum()) < 5:
            raise RuntimeError("未找到光斑(阈值内像素<5): 检查曝光/衰减/0级遮挡")
        yy, xx = np.nonzero(m)
        wv = img[m]
        return np.array([(yy * wv).sum() / wv.sum(), (xx * wv).sum() / wv.sum()])

    @staticmethod
    def _fwhm1d(profile: np.ndarray) -> float:
        peak = profile.max()
        if peak <= 0:
            return 0.0
        above = np.nonzero(profile >= peak / 2)[0]
        return float(above[-1] - above[0] + 1) if len(above) >= 2 else 0.0

    def spot_fwhm(self, img: np.ndarray, c) -> float:
        cy, cx = int(round(c[0])), int(round(c[1]))
        h, w = img.shape
        col = img[:, min(max(cx, 0), w - 1)]
        row = img[min(max(cy, 0), h - 1), :]
        return 0.5 * (self._fwhm1d(col) + self._fwhm1d(row))

    def _show(self, phase: np.ndarray):
        self.slm.display_phase(phase)
        time.sleep(self.settle_s)

    @staticmethod
    def _blaze(period_px: float, axis: str, panel_res) -> np.ndarray:
        h, w = panel_res
        if axis == "x":
            ramp = 2 * np.pi * np.arange(w) / period_px
            return np.tile(ramp, (h, 1)).astype(np.float32)
        ramp = 2 * np.pi * np.arange(h) / period_px
        return np.tile(ramp[:, None], (1, w)).astype(np.float32)

    # ------------------------------------------------------------ 阶段1: 装配辅助
    def align(
        self,
        max_rounds: int = 10,
        center_tol_px: float = 30.0,
        blaze_period: float = 24.0,
        window_margin_factor: float = 4.0,
        period_candidates: tuple[int, ...] | None = None,
        min_window_side: int = 128,
    ) -> dict:
        """软件化装配辅助: 用相机开窗(ROI)自动把0级与+1级框入视场并定心.

        阶段A 全幅检测  flat相位下0级质心c0与FWHM f0.
        阶段B 周期选择  按 period_candidates 升序测±1级位移, 首个"0级/+1级
                       在视场(边缘余量 margin=margin_factor*f0)且窗口
                       (2*(half+margin), clamp>=min_window_side)不超传感器"
                       的周期胜出; 全部不满足 -> 恢复全幅返回失败.
        阶段C 窗口定心  每轮把窗口中心移到0级(全幅坐标, 边缘clamp), 窗口大小
                       固定(sx/sy 按阶段B的half与margin). 收敛判据:
                       窗口内0级 - 窗口返回中心 <= center_tol_px.
        阶段D 收尾      成功: 保留窗口, 写 calib keys; 失败: 恢复全幅,
                       清空align keys, align_ok=False.

        写入 calib:
          align_ok             bool
          align_period         float 选中的闪耀周期
          align_center_full    (cy, cx) 0级全幅像素坐标
          align_window_center_xy (cx, cy) 最终窗口中心(驱动(x,y)序)
          align_window_size_xy (sx, sy)  最终窗口尺寸(驱动序)
        (align_window_* 存中心/尺寸而非offset: 真实驱动不返回offset,
         offset由 reset_window 从(center,size)确定性推导.)
        返回装配报告 dict(ok, reason, period, rounds, ...).
        """
        default_cands: tuple[int, ...] = (16, 24, 32, 48, 64, 96)
        if period_candidates is not None:
            cands = tuple(int(p) for p in period_candidates)
        elif int(round(blaze_period)) in default_cands:
            cands = default_cands
        else:
            cands = (int(round(blaze_period)),) + default_cands

        report: dict = {"rounds": [], "ok": False, "period": None, "reason": "unknown"}

        def restore_full() -> None:
            try:
                self.ccd.reset_window((0, 0), (0, 0))
            except Exception:
                logger.warning("恢复全幅窗口失败")

        try:
            # ---------- 阶段A: 全幅检测 ----------
            self._show(np.zeros(self.panel_res, np.float32))
            F0 = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
            H, W = F0.shape
            c0 = self._moments(F0)
            f0 = max(self.spot_fwhm(F0, c0), 3.0)
            margin = window_margin_factor * f0
            logger.info(
                "align 阶段A: 帧={}x{} 0级=({:.1f},{:.1f}) FWHM≈{:.1f}px",
                W,
                H,
                c0[0],
                c0[1],
                f0,
            )

            # ---------- 阶段B: 周期选择 ----------
            period: float = 0.0
            half_x = half_y = 0.0
            for P in cands:
                self._show(self._blaze(P, "x", self.panel_res))
                s_x = self._moments(
                    np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64),
                    exclude=(c0, 3.0 * f0),
                )
                self._show(self._blaze(P, "y", self.panel_res))
                s_y = self._moments(
                    np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64),
                    exclude=(c0, 3.0 * f0),
                )
                half_x = max(abs(s_x[1] - c0[1]), 0.0)
                half_y = max(abs(s_y[0] - c0[0]), 0.0)
                sx = max(2.0 * (half_x + margin), float(min_window_side))
                sy = max(2.0 * (half_y + margin), float(min_window_side))
                in_fov = {
                    "0级": (
                        margin <= c0[0] < H - margin and margin <= c0[1] < W - margin
                    ),
                    "+1级x": (
                        margin <= s_x[0] < H - margin
                        and margin <= s_x[1] < W - margin
                    ),
                    "+1级y": (
                        margin <= s_y[0] < H - margin
                        and margin <= s_y[1] < W - margin
                    ),
                }
                fits = sx <= W - 1 and sy <= H - 1
                report["rounds"].append(
                    dict(
                        stage="B",
                        period=float(P),
                        spots={"0级": c0, "+1级x": s_x, "+1级y": s_y},
                        window_size=(sx, sy),
                        in_fov=in_fov,
                    )
                )
                logger.info(
                    "align 阶段B P={:2d}px: +1级x=({:.0f},{:.0f}) +1级y=({:.0f},{:.0f}) "
                    "窗口={:.0f}x{:.0f} {}",
                    P,
                    s_x[0],
                    s_x[1],
                    s_y[0],
                    s_y[1],
                    sx,
                    sy,
                    "采用" if (all(in_fov.values()) and fits) else "跳过",
                )
                if all(in_fov.values()) and fits:
                    period = float(P)
                    break
            if period == 0.0:
                report["reason"] = "no_period"
                logger.warning(
                    "align 阶段B: 无周期满足(级次出视场/窗口超传感器), 恢复全幅"
                )
                restore_full()
                self._finalize_align(report, ok=False)
                return report

            # ---------- 阶段C: 窗口定心 ----------
            sx = min(
                max(2.0 * (half_x + margin), float(min_window_side)), float(W - 1)
            )
            sy = min(
                max(2.0 * (half_y + margin), float(min_window_side)), float(H - 1)
            )
            win_offset = (0, 0)  # 驱动(x,y)序
            returned_rc0 = np.array([0.0, 0.0])
            last_center_full = np.array([c0[0], c0[1]])  # (cy,cx)
            window_center: tuple[float, float] = (0.0, 0.0)
            window_size: tuple[float, float] = (sx, sy)
            first = True
            for rnd in range(max_rounds):
                self._show(np.zeros(self.panel_res, np.float32))
                F = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
                c0_win = self._moments(F)
                if not first and np.linalg.norm(c0_win - returned_rc0) <= center_tol_px:
                    last_center_full = np.array(
                        [win_offset[1] + c0_win[0], win_offset[0] + c0_win[1]]
                    )
                    logger.info(
                        "align 阶段C 第{}轮: 0级窗口内偏移=({:.1f},{:.1f})px "
                        "<= tol {:.1f}px, 收敛",
                        rnd,
                        c0_win[0] - returned_rc0[0],
                        c0_win[1] - returned_rc0[1],
                        center_tol_px,
                    )
                    break
                cx_full = win_offset[0] + c0_win[1]
                cy_full = win_offset[1] + c0_win[0]
                last_center_full = np.array([cy_full, cx_full])
                cx_c = float(np.clip(cx_full, sx / 2.0, W - 1 - sx / 2.0))
                cy_c = float(np.clip(cy_full, sy / 2.0, H - 1 - sy / 2.0))
                returned = self.ccd.reset_window(
                    (int(round(cx_c)), int(round(cy_c))),
                    (int(round(sx)), int(round(sy))),
                )
                window_size = (float(returned[0][0]), float(returned[0][1]))
                window_center = (cx_c, cy_c)
                win_offset = (
                    int(round(cx_c)) - returned[1][0],
                    int(round(cy_c)) - returned[1][1],
                )
                returned_rc0 = np.array([returned[1][1], returned[1][0]])  # (cy,cx)
                report["rounds"].append(
                    dict(
                        stage="C",
                        round=rnd,
                        window_size=window_size,
                        window_center=window_center,
                        window_offset=win_offset,
                        c0_window=c0_win,
                        residual=c0_win - returned_rc0,
                    )
                )
                first = False
            else:
                report["reason"] = "not_converged"
                logger.warning("align 阶段C: {}轮未收敛, 恢复全幅", max_rounds)
                restore_full()
                self._finalize_align(report, ok=False)
                return report

            # ---------- 阶段D: 收尾 (成功, 保留窗口) ----------
            report["ok"] = True
            report["period"] = period
            report["center_full"] = last_center_full
            report["window_center"] = window_center
            report["window_size"] = window_size
            self._finalize_align(report, ok=True)
            logger.info(
                "align 完成: 周期={:.0f}px 0级全幅=({:.0f},{:.0f}) "
                "窗口 中心=({:.0f},{:.0f}) 尺寸={:.0f}x{:.0f}px",
                period,
                last_center_full[0],
                last_center_full[1],
                window_center[0],
                window_center[1],
                window_size[0],
                window_size[1],
            )
            return report
        except RuntimeError as e:
            report["reason"] = "no_spot"
            logger.warning("align 失败(找不到光斑): {}", e)
            restore_full()
            self._finalize_align(report, ok=False)
            return report

    def _finalize_align(self, report: dict, ok: bool) -> None:
        """把 align 结果写入/清理 self.calib 的 align_* keys."""
        if self.calib is None:
            self.calib = {}
        self.calib["align_ok"] = bool(ok)
        self.calib["align_period"] = report.get("period")
        if ok:
            self.calib["align_center_full"] = np.asarray(
                report["center_full"], np.float64
            )
            self.calib["align_window_center_xy"] = np.asarray(
                report["window_center"], np.float64
            )
            self.calib["align_window_size_xy"] = np.asarray(
                report["window_size"], np.float64
            )
        else:
            for k in (
                "align_center_full",
                "align_window_center_xy",
                "align_window_size_xy",
            ):
                self.calib.pop(k, None)

    def apply_stored_window(self) -> bool:
        """把 calib 中保存的 align 窗口应用到相机 (center/size 形式恢复).

        驱动 reset_window 由 (center, size) 确定性推导 offset, 因此恢复窗口
        无需保存 offset. 无窗口信息或窗口不合法/超传感器时返回 False
        (相机保持现状, 不主动复位).
        """
        if self.calib is None:
            return False
        size = self.calib.get("align_window_size_xy")
        center = self.calib.get("align_window_center_xy")
        if size is None or center is None:
            return False
        sx, sy = int(round(size[0])), int(round(size[1]))
        if sx <= 0 or sy <= 0:
            return False
        try:
            self.ccd.reset_window(
                (int(round(center[0])), int(round(center[1]))), (sx, sy)
            )
        except AssertionError:
            logger.warning(
                "存储窗口超出传感器范围, 保持全幅: size={} center={}", size, center
            )
            return False
        logger.info(
            "已应用存储窗口: 中心=({:.0f},{:.0f}) 尺寸={}x{}",
            center[0],
            center[1],
            sx,
            sy,
        )
        return True

    # ------------------------------------------------------------ 阶段1.5: 光束在SLM上的位置
    def find_beam_on_slm(
        self, n_scan: int = 25, checker_period: int = 8, window: int | None = None
    ) -> dict:
        """刀口扫描: 测光束在SLM面板上的中心(beam_center)与宽度(beam_sigma), 单位panel px.
        图案: 分割位置s一侧为0相位(全通), 另一侧为0/pi棋盘(散射走0级);
        0级功率 P(s) = 光束截面累积分布 -> 中心=50% crossing, sigma=(84%-16%)/2.
        结果写入 self.calib['beam_center']=(cy,cx), ['beam_sigma']=(sy,sx)."""
        self._show(np.zeros(self.panel_res, np.float32))
        F0 = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
        c0 = self._moments(F0)
        h, w = self.panel_res
        win = int(window or 6 * max(self.spot_fwhm(F0, c0), 5))

        def power(frame: np.ndarray) -> float:
            cy, cx = int(round(c0[0])), int(round(c0[1]))
            hh, ww = frame.shape
            y0, x0 = cy - win // 2, cx - win // 2
            pt, pb = max(0, -y0), max(0, y0 + win - hh)
            pl, pr = max(0, -x0), max(0, x0 + win - ww)
            p = frame[max(0, y0) : min(hh, y0 + win), max(0, x0) : min(ww, x0 + win)]
            if pt or pb or pl or pr:
                p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
            return float(p.sum())

        def scan_pattern(s: int, axis: str) -> np.ndarray:
            yy, xx = np.mgrid[0:h, 0:w]
            clear = (xx < s) if axis == "x" else (yy < s)
            checker = ((xx // checker_period + yy // checker_period) % 2) * np.pi
            return np.where(clear, 0.0, checker).astype(np.float32)

        def scan(axis: str):
            H = w if axis == "x" else h
            ss = np.linspace(0, H, n_scan)
            P = []
            for s in ss:
                self._show(scan_pattern(int(round(s)), axis))
                P.append(
                    power(
                        np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
                    )
                )
            P = np.asarray(P)
            Pn = (P - P.min()) / (np.ptp(P) + 1e-12)
            c50 = float(np.interp(0.50, Pn, ss))
            sigma = float(np.interp(0.84, Pn, ss) - np.interp(0.16, Pn, ss)) / 2.0
            return c50, max(sigma, 1.0)

        cx, sx = scan("x")
        cy, sy = scan("y")
        if self.calib is None:
            self.calib = {}
        self.calib["beam_center"] = np.array([cy, cx])
        self.calib["beam_sigma"] = np.array([sy, sx])
        off_y, off_x = cy - h / 2, cx - w / 2
        logger.info(
            "SLM面板上的光束: 中心=({:.0f},{:.0f}) 偏移面板中心=({:+.0f},{:+.0f})px "
            "sigma=({:.0f},{:.0f})px 直径(1/e²)≈{:.0f}px",
            cy,
            cx,
            off_y,
            off_x,
            sy,
            sx,
            math.sqrt(2) * (sy + sx),
        )
        if abs(off_y) > 0.1 * h or abs(off_x) > 0.1 * w:
            logger.warning(
                "光束偏离面板中心>10%, 建议装配调节(见align()引导)或后续图案均以beam_center为原点"
            )
        return dict(
            beam_center=self.calib["beam_center"], beam_sigma=self.calib["beam_sigma"]
        )

    def place_on_panel(self, pattern_nx: np.ndarray, N: int = 64) -> np.ndarray:
        """把N×N弧度相位放到面板, 以beam_center(而非面板几何中心)为原点.
        模型里的图案坐标(0,0) ≡ 光束轴线 ≡ 0级质心."""
        import torch
        import torch.nn.functional as F

        if self.calib is None or "beam_center" not in self.calib:
            raise RuntimeError("先find_beam_on_slm()")
        h, w = self.panel_res
        full = F.interpolate(
            torch.from_numpy(np.asarray(pattern_nx, np.float32))[None, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        cy, cx = self.calib["beam_center"]
        tx, ty = 2 * (cx - w / 2) / w, 2 * (cy - h / 2) / h
        theta = torch.tensor([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=torch.float32)[
            None
        ]
        grid = F.affine_grid(theta, full.shape, align_corners=False)
        return (
            F.grid_sample(full, grid, align_corners=False, padding_mode="zeros")[0, 0]
            .numpy()
            .astype(np.float32)
        )

    # ------------------------------------------------------------ 阶段2: 几何标定
    def calibrate(
        self, periods=(16, 24, 32), exclude_radius_factor: float = 3.0
    ) -> dict:
        """自动标定 center/Kx/Ky/rotation, 结果存 self.calib 并返回.
        保留已有 beam_center/beam_sigma(阶段1.5)不丢失."""
        self._show(np.zeros(self.panel_res, np.float32))
        F0 = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
        c0 = self._moments(F0)
        f0 = max(self.spot_fwhm(F0, c0), 3.0)
        logger.info("0级质心=({:.1f}, {:.1f})  FWHM≈{:.1f}px", c0[0], c0[1], f0)

        K, u, resid = {}, {}, {}
        for axis in ("x", "y"):
            disps, invs = [], []
            for P in periods:
                self._show(self._blaze(P, axis, self.panel_res))
                F = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
                d = self._moments(F, exclude=(c0, exclude_radius_factor * f0)) - c0
                disps.append(d)
                invs.append(1.0 / P)
                logger.info(
                    "  {}光栅 P={:2d}px -> 位移=({:+7.1f}, {:+7.1f})px",
                    axis,
                    P,
                    d[0],
                    d[1],
                )
            disps = np.asarray(disps)
            invs = np.asarray(invs)
            u0 = disps.sum(axis=0)
            nrm = np.linalg.norm(u0)
            if nrm < 1e-6:
                raise RuntimeError(f"标定失败: {axis}方向光栅无可测位移")
            u0 /= nrm
            K[axis] = float((disps @ u0 @ invs) / (invs @ invs))
            u[axis] = u0
            resid[axis] = float(np.abs(disps - np.outer(invs * K[axis], u0)).max())
            logger.info(
                "  {}轴: K={:.1f}px·SLMpx  线性拟合残差={:.1f}px",
                axis,
                K[axis],
                resid[axis],
            )

        rotation_deg = (
            math.degrees(math.acos(float(np.clip(u["x"] @ u["y"], -1, 1)))) - 90.0
        )
        mirror = bool(u["x"][1] < 0)
        prev = self.calib or {}
        self.calib = dict(
            center=c0,
            Kx=K["x"],
            Ky=K["y"],
            rotation_deg=float(rotation_deg),
            fwhm0=float(f0),
            mirror=mirror,
            resid_x=resid["x"],
            resid_y=resid["y"],
            crop_side=int(round(max(K["x"], K["y"]) * 1.15)),
        )
        for k_src in (  # 保留光束位置与对准结果
            "beam_center",
            "beam_sigma",
            "align_ok",
            "align_period",
            "align_center_full",
            "align_window_center_xy",
            "align_window_size_xy",
        ):
            if k_src in prev:
                self.calib[k_src] = prev[k_src]
        logger.info(
            "标定完成: Kx={:.1f} Ky={:.1f} rot={:+.2f}° crop={}px mirror={}",
            self.calib["Kx"],
            self.calib["Ky"],
            self.calib["rotation_deg"],
            self.calib["crop_side"],
            mirror,
        )
        if mirror:
            logger.warning("检测到x方向镜像, 检查光路奇次反射或翻转SLM坐标")
        if max(resid.values()) > 0.05 * min(K.values()):
            logger.warning("拟合残差偏大(>5%K): 0级污染? 曝光不足? 建议verify()复核")
        return self.calib

    def save(self, path) -> None:
        if self.calib is None:
            raise RuntimeError("无标定结果, 先calibrate()或load()")
        np.savez(path, **self.calib)
        logger.info("标定已保存: {}", path)

    def load(self, path) -> dict:
        self.calib = dict(np.load(path, allow_pickle=True))
        logger.info(
            "加载标定: Kx={:.1f} Ky={:.1f} rot={:+.2f}°",
            self.calib["Kx"],
            self.calib["Ky"],
            self.calib["rotation_deg"],
        )
        return self.calib

    # ------------------------------------------------------------ 阶段3: 几何验证
    def verify(
        self,
        test_period: float = 20.0,
        drift_tol_px: float = 3.0,
        pred_tol_px: float = 5.0,
    ) -> dict:
        """定量验证标定效果. 返回报告 dict(pass, items).
        1) 0级质心漂移(重复性) 2) ±1级位置预测残差 3) 工作区能量集中度."""
        if self.calib is None:
            raise RuntimeError("先calibrate()或load()")
        c0_ref = np.asarray(self.calib["center"], np.float64)
        Kx, Ky = self.calib["Kx"], self.calib["Ky"]
        report: dict = {"items": {}}

        self._show(np.zeros(self.panel_res, np.float32))
        c0 = self._moments(
            np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
        )
        drift = float(np.linalg.norm(c0 - c0_ref))
        report["items"]["center_drift_px"] = drift

        errors = {}
        for axis, K in (("x", Kx), ("y", Ky)):
            self._show(self._blaze(test_period, axis, self.panel_res))
            F = np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
            c_meas = self._moments(F, exclude=(c0, 3.0 * max(self.calib["fwhm0"], 3.0)))
            direction = np.array([0.0, 1.0]) if axis == "x" else np.array([1.0, 0.0])
            if abs(self.calib["rotation_deg"]) > 0.3:
                th = math.radians(self.calib["rotation_deg"])
                R = np.array(
                    [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
                )
                direction = R @ direction
            c_pred = c0 + direction * (K / test_period)
            errors[axis] = float(np.linalg.norm(c_meas - c_pred))
        report["items"]["pred_err_px"] = errors

        patch = self.workzone(
            np.asarray(self.ccd.get_numpy_image(CAMERA_SAMPLES), np.float64)
        )
        In = patch / max(patch.sum(), 1e-12)
        yy, xx = np.mgrid[0 : In.shape[0], 0 : In.shape[1]]
        for frac, tag in ((0.25, "中心1/2区域"), (0.0625, "中心1/4区域")):
            r = math.sqrt(frac) * In.shape[0] / 2.0
            m = (yy - In.shape[0] / 2) ** 2 + (xx - In.shape[1] / 2) ** 2 <= r * r
            report["items"][f"encircled_{tag}"] = float(In[m].sum())

        ok = drift < drift_tol_px and max(errors.values()) < pred_tol_px
        report["pass"] = bool(ok)
        logger.info(
            "几何验证: 质心漂移={:.2f}px(限{:.1f})  预测残差 x={:.2f}/y={:.2f}px(限{:.1f})  {}",
            drift,
            drift_tol_px,
            errors["x"],
            errors["y"],
            pred_tol_px,
            "通过 ✓" if ok else "未通过 ✗",
        )
        for k, v in report["items"].items():
            if k.startswith("encircled"):
                logger.info("  {}: {:.3f}", k, v)
        if not ok:
            logger.warning("验证未通过: 检查装配松动/温度漂移, 或重新calibrate()")
        return report

    # ------------------------------------------------------------ 运行时接口
    def workzone(self, img: np.ndarray, N: int = 64) -> np.ndarray:
        """按标定参数取工作区: 定心(calibrated center) -> 纠旋 -> 重采样N×N.
        所有后续帧(闭环采图同理)都必须走这个函数, 保证与模型网格严格对齐."""
        if self.calib is None:
            raise RuntimeError("先calibrate()或load()")
        import torch
        import torch.nn.functional as F

        cy, cx = self.calib["center"]
        side = int(self.calib["crop_side"])
        h, w = img.shape
        y0, x0 = int(round(cy)) - side // 2, int(round(cx)) - side // 2
        pt, pb = max(0, -y0), max(0, y0 + side - h)
        pl, pr = max(0, -x0), max(0, x0 + side - w)
        p = img[max(0, y0) : min(h, y0 + side), max(0, x0) : min(w, x0 + side)]
        if pt or pb or pl or pr:
            p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
        t = torch.from_numpy(p.astype(np.float32))[None, None]
        if abs(self.calib["rotation_deg"]) > 0.3:
            th = math.radians(-self.calib["rotation_deg"])  # 若旋转方向反了, 去掉负号
            rot = torch.tensor(
                [[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0]],
                dtype=torch.float32,
            )[None]
            grid = F.affine_grid(rot, t.shape, align_corners=False)
            t = F.grid_sample(t, grid, align_corners=False, padding_mode="border")
        return (
            F.interpolate(t, size=(N, N), mode="bilinear", align_corners=False)[0, 0]
            .numpy()
            .astype(np.float64)
        )

    def gauss_amp_from_farfield(
        self, flat_frame: np.ndarray, N: int = 64, w0_override: float | None = None
    ) -> torch.Tensor:
        """由flat相位远场高斯拟合估计源面高斯幅值(近似).
        有更准的近场测量时直接传 w0_override 或替换此函数."""
        import torch

        if w0_override is not None:
            w0 = w0_override
        else:
            fine = self.workzone(flat_frame, 256)
            In = fine / max(fine.sum(), 1e-12)
            fwhm_n = self.spot_fwhm(In, np.array(In.shape) / 2.0) / 256.0
            w0 = float(
                np.clip(
                    2 * math.sqrt(2 * math.log(2)) / (math.pi * max(fwhm_n, 1e-3)),
                    0.2,
                    1.5,
                )
            )
            logger.info(
                "远场FWHM={:.3f}(归一化) -> 源面高斯束腰 w0≈{:.3f}(可调)", fwhm_n, w0
            )
        t = torch.linspace(-1, 1, N)
        y, x = torch.meshgrid(t, t, indexing="ij")
        A = torch.exp(-(x * x + y * y) / w0**2)
        return A / A.amax()

    def target_half_to_ccd_px(self, half_norm: float) -> float:
        """归一化半宽 -> CCD像素边长(核对目标尺寸): side_px = 2*half*K/2."""
        if self.calib is None:
            raise RuntimeError("先calibrate()或load()")
        return float(half_norm) * max(self.calib["Kx"], self.calib["Ky"])


# =====================================================================
class SLMLUTCalibrator:
    """[DEPRECATED] SLM 灰度-相位 LUT 标定(自参考干涉法, legacy 8-bit).

    请改用 canonical 灰度↔相位 LUT 管线: ``tools/slm/slm_lut_runner.py`` +
    ``utils/slm_lut.py`` (输出供 ``Santec.load_lut`` 加载)。本类仅为参考/
    研究/单测保留, 已从 CLI 移除; 直接使用会收到 DeprecationWarning。

    与几何标定相互独立, 可任意顺序执行.

    Parameters
    ----------
    slm         SLM驱动 (display_phase 接收弧度矩阵并转换为灰度)
    ccd, acquire 相机与采图回调
    panel_res   (h, w)
    calib       几何标定 dict(可选): 提供 0级质心与 beam_center,
                条纹窗口更准且半屏分割对称; 无则用全帧质心/面板中心
    factory_2pi 保留用于 legacy ``_split_pattern`` 纯函数; 硬件标定与显示均使用 raw uint16 灰度.
    """

    def __init__(
        self,
        slm: Santec,
        ccd: DahengCamera,
        calib: dict | None = None,
        settle_s: float = SETTLE_S,
        factory_2pi: float = 255.0,
        window: int = 384,
    ):
        warnings.warn(
            "SLMLUTCalibrator is deprecated; use the slm-lut pipeline "
            "(tools/slm/slm_lut_runner.py + utils/slm_lut.py) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.slm = slm
        self.ccd = ccd
        self.panel_res = slm.Panel_Res[::-1]
        self.calib = calib
        self.settle_s = settle_s
        self.factory_2pi = factory_2pi
        self.window = window
        self.result: dict | None = None

    # ------------------------------------------------------------ 图案与条纹分析
    def _split_pattern(self, g_left: float, g_right: float) -> np.ndarray:
        """Return the legacy radian half-panel pattern used by pure tests."""
        h, w = self.panel_res
        bc = None
        if self.calib is not None:
            bc = self.calib.get("beam_center", None)
        cx = int(round(bc[1])) if bc is not None else w // 2
        pat = np.empty((h, w), np.float32)
        pat[:, :cx] = g_left * (2 * np.pi / self.factory_2pi)
        pat[:, cx:] = g_right * (2 * np.pi / self.factory_2pi)
        return pat

    def _split_gray_pattern(self, g_left: float, g_right: float) -> np.ndarray:
        """Return a raw uint16 half-panel pattern for SLM display."""
        h, w = self.panel_res
        bc = None
        if self.calib is not None:
            bc = self.calib.get("beam_center", None)
        cx = int(round(bc[1])) if bc is not None else w // 2
        pat = np.zeros((h, w), dtype=np.uint16)
        pat[:, :cx] = np.clip(np.rint(g_left), 0, np.iinfo(np.uint16).max)
        pat[:, cx:] = np.clip(np.rint(g_right), 0, np.iinfo(np.uint16).max)
        return pat

    def _center(self, img: np.ndarray):
        if self.calib is not None:
            return np.asarray(self.calib["center"], np.float64)
        thr = img.max() * 0.15
        m = img >= thr
        yy, xx = np.nonzero(m)
        return np.array([yy.mean(), xx.mean()])

    def _fringe_phase(self, frame: np.ndarray):
        """取中心窗口, 行平均成一维, Hann窗+FFT主峰相位. 自动找条纹频率."""
        img = np.asarray(frame, np.float64)
        cy, cx = self._center(img)
        h, w = img.shape
        half = self.window // 2
        y0, x0 = int(round(cy)) - half, int(round(cx)) - half
        pt, pb = max(0, -y0), max(0, y0 + self.window - h)
        pl, pr = max(0, -x0), max(0, x0 + self.window - w)
        p = img[
            max(0, y0) : min(h, y0 + self.window), max(0, x0) : min(w, x0 + self.window)
        ]
        if pt or pb or pl or pr:
            p = np.pad(p, ((pt, pb), (pl, pr)), mode="edge")
        prof = p.mean(axis=0)
        prof = prof - prof.mean()
        spec = np.fft.rfft(prof * np.hanning(len(prof)))
        mag = np.abs(spec)
        mag[:3] = 0
        k = int(np.argmax(mag))
        period = len(prof) / k if k > 0 else np.inf
        if not (3.0 <= period <= len(prof) / 8):
            logger.warning(
                "条纹周期={:.1f}px 异常(期望3~{:.0f}px), 检查窗口/半屏分割",
                period,
                len(prof) / 8,
            )
        return float(np.angle(spec[k])), k, float(period)

    # ------------------------------------------------------------ 标定主流程
    def calibrate(self, grays=None, g_ref: float = 128.0, drift_correct: bool = True):
        """全灰度扫描, 返回并缓存 dict(gray, phase_of_gray, gray_of_phase, phase_range)."""
        if grays is None:
            grays = np.linspace(0, 255, 64)
        grays = np.asarray(grays, np.float64)

        def psi(g):
            self.slm.display_data(self._split_gray_pattern(g, g_ref))
            time.sleep(self.settle_s)
            ph, k, period = self._fringe_phase(self.ccd.get_numpy_image(CAMERA_SAMPLES))
            return ph

        logger.info("LUT标定: 参考灰度 g_ref={:.0f}, {}个灰度点 ...", g_ref, len(grays))
        psi_ref0 = psi(g_ref)
        phases = np.array([psi(g) for g in grays])
        psi_ref1 = psi(g_ref) if drift_correct else psi_ref0
        frac = np.linspace(0, 1, len(grays))
        phases = phases - (psi_ref0 + frac * (psi_ref1 - psi_ref0))
        curve = np.unwrap(phases - phases[np.argmin(np.abs(grays - g_ref))])
        curve = np.maximum.accumulate(curve)
        k = 5
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        smooth = np.convolve(curve, ker, mode="same")
        smooth[:k] = curve[:k]
        smooth[-k:] = curve[-k:]

        phase_grid = np.linspace(smooth.min(), smooth.max(), 256)
        gray_of_phase = np.interp(phase_grid, smooth, grays)
        self.result = dict(
            gray=grays,
            phase_of_gray=smooth,
            gray_of_phase=gray_of_phase,
            phase_range=float(smooth.max() - smooth.min()),
            g_ref=g_ref,
        )
        logger.info(
            "LUT标定完成: 调制深度={:.2f} rad ({:.2f}×2π)",
            self.result["phase_range"],
            self.result["phase_range"] / (2 * np.pi),
        )
        if self.result["phase_range"] < 1.8 * np.pi:
            logger.warning(
                "调制深度不足2π({:.2f}rad), 相位会量化失真, 考虑换波长/型号",
                self.result["phase_range"],
            )
        return self.result

    def save(self, path) -> None:
        if self.result is None:
            raise RuntimeError("先calibrate()或load()")
        np.savez(path, **self.result)
        logger.info("LUT已保存: {}", path)

    def load(self, path) -> dict:
        self.result = dict(np.load(path, allow_pickle=True))
        logger.info("加载LUT: 调制深度={:.2f} rad", float(self.result["phase_range"]))
        return self.result

    # ------------------------------------------------------------ 标定显示通道
    def phase2gray(self, phase: np.ndarray) -> np.ndarray:
        """弧度相位矩阵 -> 灰度图 (查标定LUT, 自动按调制深度缩放)"""
        if self.result is None:
            raise RuntimeError("先calibrate()或load()")
        ph = np.asarray(phase, np.float64)
        depth = float(self.result["phase_range"])
        ph01 = np.mod(ph, depth) / depth
        gray = np.interp(
            ph01,
            np.linspace(0, 1, 256),
            np.asarray(self.result["gray_of_phase"], np.float64),
        )
        return gray.astype(np.float32)

    def display_phase(self, phase_nx: np.ndarray):
        """标定后的相位显示: N×N弧度相位 -> 上采样 -> 查LUT -> raw uint16 灰度."""
        import torch
        import torch.nn.functional as F

        full = F.interpolate(
            torch.from_numpy(np.asarray(phase_nx, np.float32))[None, None],
            size=self.panel_res,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        gray = self.phase2gray(full)
        self.slm.display_data(gray.astype(np.uint16))
        time.sleep(self.settle_s)

    # ------------------------------------------------------------ LUT验证
    def verify(self, n_test: int = 9) -> dict:
        """用标定LUT显示 n_test 个目标相位(0~深度), 测残余条纹相位, 报告std."""
        if self.result is None:
            raise RuntimeError("先calibrate()或load()")
        depth = float(self.result["phase_range"])
        targets = np.linspace(0, 0.9 * depth, n_test)
        res = []
        for t in targets:
            gray_test = float(
                np.interp(
                    t / depth,
                    np.linspace(0, 1, 256),
                    np.asarray(self.result["gray_of_phase"], np.float64),
                )
            )
            self.slm.display_data(
                self._split_gray_pattern(gray_test, self.result["g_ref"])
            )
            time.sleep(self.settle_s)
            ph, _, _ = self._fringe_phase(self.ccd.get_numpy_image(CAMERA_SAMPLES))
            res.append((t, ph))
        res = np.asarray(res)
        err = np.unwrap(res[:, 1] - res[0, 1]) - (res[:, 0] - res[0, 0])
        std = float(err.std())
        report = dict(
            err_rad=err.tolist(), residual_std_rad=std, pass_=bool(std < 0.05)
        )
        logger.info(
            "LUT验证: 残差std={:.3f} rad ({:.1f}°)  {}",
            std,
            np.degrees(std),
            "通过 ✓" if report["pass_"] else "未通过 ✗",
        )
        return report


# =====================================================================
# CLI: 全流程 装配 -> 光束位置 -> 几何标定 -> 几何验证 (LUT 已移出, 见 slm-lut)
# =====================================================================
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--out-calib",
    "out_calib",
    default="calib.npz",
    show_default=True,
    help="几何标定输出路径",
)
@click.option(
    "--calib",
    "calib_path",
    default=None,
    help="已有几何标定文件(verify-only时作为输入)",
)
@click.option("--skip-align", is_flag=True, help="跳过装配辅助")
@click.option("--skip-beam", is_flag=True, help="跳过光束位置测量(沿用已有beam_center)")
@click.option(
    "--verify-only", is_flag=True, help="只验证已有几何标定(需 --calib)"
)
@click.option(
    "--align-margin",
    default=4.0,
    show_default=True,
    help="align 窗口边缘余量倍数(×FWHM)",
)
@click.option(
    "--align-min-window",
    default=128,
    show_default=True,
    help="align 最小窗口边长(px)",
)
@click.option("--exposure-ms", default=1.2, show_default=True, help="CCD曝光(ms)")
@click.option("--settle-s", default=0.2, show_default=True, help="SLM显示稳定等待(s)")
def main(
    out_calib,
    calib_path,
    skip_align,
    skip_beam,
    verify_only,
    align_margin,
    align_min_window,
    exposure_ms,
    settle_s,
):
    """SLM+CCD 几何标定工具: 装配 -> 光束位置 -> 几何标定 -> 验证."""
    with Santec() as slm, DahengCamera(exposure_time_ms=exposure_ms) as ccd:
        geo = SLMCCDCalibrator(slm, ccd, settle_s=settle_s)

        # ---------- 仅验证 ----------
        if verify_only:
            if not calib_path:
                raise click.UsageError("--verify-only 需要 --calib")
            geo.load(calib_path)
            geo.verify()
            return

        # ---------- 几何标定 ----------
        if calib_path and skip_align and skip_beam:
            geo.load(calib_path)  # 以已有标定为底, 增量复标
        if not skip_align:
            geo.align(
                window_margin_factor=align_margin, min_window_side=align_min_window
            )
        if not skip_beam:
            geo.find_beam_on_slm()
        geo.calibrate()
        geo.save(out_calib)
        geo.verify()
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shift":
        main_shift_calib.main(args=sys.argv[2:], prog_name="ao_shaping.tools.slm.calibration shift")
    else:
        main.main(args=sys.argv[1:], prog_name="ao_shaping.tools.slm.calibration")