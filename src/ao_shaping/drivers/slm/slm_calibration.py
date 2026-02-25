"""SLM闪耀光栅标定模块

使用闪耀光栅法实现对SLM的相位-灰度响应标定。
通过在SLM上显示闪耀光栅图案，测量衍射效率来确定灰度值与相位的对应关系。

闪耀光栅法原理：
1. 在SLM上显示线性相位梯度（闪耀光栅）
2. 光栅将入射光衍射到特定方向
3. 当相位深度为2π时，衍射效率最高
4. 通过扫描灰度值找到最大衍射效率点，确定2π对应的灰度值
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Protocol, Union

import numpy as np
from loguru import logger

from scipy.interpolate import UnivariateSpline

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
            'grayscale_2pi': self.grayscale_2pi,
            'grayscale_values': self.grayscale_values,
            'intensities': self.intensities.tolist(),
            'wavelength_nm': self.wavelength_nm,
            'slm_model': self.slm_model,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationResult':
        """从字典创建标定结果"""
        return cls(
            grayscale_2pi=data['grayscale_2pi'],
            grayscale_values=data['grayscale_values'],
            intensities=np.array(data['intensities']),
            wavelength_nm=data['wavelength_nm'],
            slm_model=data['slm_model'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """保存标定结果到JSON文件
        
        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"标定结果已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'CalibrationResult':
        """从JSON文件加载标定结果
        
        Args:
            filepath: 文件路径
            
        Returns:
            CalibrationResult实例
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"标定文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
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
        camera: CameraProtocol,
        grating_period: int = 8,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_size: Tuple[int, int] = (100, 100)
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
        self._result: Optional[CalibrationResult] = None
    
    @property
    def result(self) -> Optional[CalibrationResult]:
        """获取标定结果"""
        return self._result
    
    def create_blazed_grating(
        self,
        grayscale_depth: int,
        direction: str = 'horizontal'
    ) -> np.ndarray:
        """创建闪耀光栅相位图
        
        创建一个线性相位梯度的闪耀光栅图案。
        
        Args:
            grayscale_depth: 相位深度（灰度值0-1023）
            direction: 光栅方向，'horizontal' 或 'vertical'
            
        Returns:
            相位图数组（uint16）
        """
        width = self.slm.width if hasattr(self.slm, 'width') else 1920
        height = self.slm.height if hasattr(self.slm, 'height') else 1080
        
        # 创建坐标网格
        if direction == 'horizontal':
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
        self,
        grayscale_depth: int,
        n_samples: int = 3,
        memory_number: int = 1
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
        self.slm.write_phase(grating, memory_number=memory_number)
        self.slm.display_memory(memory_number)
        
        # 等待SLM响应
        time.sleep(0.1)
        
        # 获取图像
        img = self.camera.get_numpy_image(n_sample=n_samples)
        
        # 计算ROI内的平均强度
        intensity = self._calculate_roi_intensity(img)
        
        return intensity
    
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
        grayscale_range: Tuple[int, int] = (100, 1023),
        step: int = 10,
        n_samples: int = 3,
        fine_search: bool = True,
        fine_step: int = 2,
        fine_range: int = 50
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
        grayscale_values = lrange(
            grayscale_range[0], 
            grayscale_range[1] + 1, 
            step
        )
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
            all_grayscale_values = np.concatenate([grayscale_values, fine_grayscale_values])
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
            wavelength_nm=self.slm.wavelength if hasattr(self.slm, 'wavelength') else 0,
            slm_model=self._get_slm_model(),
            timestamp=datetime.now().isoformat(),
            metadata={
                'grating_period': self.grating_period,
                'roi_center': self.roi_center,
                'roi_size': self.roi_size,
                'n_samples': n_samples
            }
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


class SantecSLM200Calibrator(SLMCalibratorBase):
    """Santec SLM-200 专用标定器
    
    针对Santec SLM-200的闪耀光栅标定实现。
    """
    
    def __init__(
        self,
        slm,  # SantecSLM200实例
        camera: CameraProtocol,
        grating_period: int = 8,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_size: Tuple[int, int] = (100, 100)
    ):
        """初始化Santec SLM-200标定器
        
        Args:
            slm: SantecSLM200实例
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
            roi_size=roi_size
        )
        
        # Santec SLM-200的分辨率
        self._width = 1920
        self._height = 1080
    
    def create_blazed_grating(
        self,
        grayscale_depth: int,
        direction: str = 'horizontal'
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
        
        if direction == 'horizontal':
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
        grayscale_range: Tuple[int, int] = (100, 1023),
        step: int = 10,
        n_samples: int = 3,
        fine_search: bool = True,
        fine_step: int = 2,
        fine_range: int = 50,
        measure_background: bool = True
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
            fine_range=fine_range
        )
        
        # 扣除背景
        if measure_background and background_intensity > 0:
            corrected_intensities = result.intensities - background_intensity
            corrected_intensities = np.maximum(corrected_intensities, 0)
            
            # 更新结果
            result.intensities = corrected_intensities
            result.metadata['background_intensity'] = background_intensity
        
        return result
    
    def _get_slm_model(self) -> str:
        """获取SLM型号"""
        return "Santec SLM-200"


def create_calibration_curve(
    grayscale_values: np.ndarray,
    intensities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """创建标定曲线
    
    对标定数据进行多项式拟合，生成平滑的标定曲线。
    
    Args:
        grayscale_values: 灰度值数组
        intensities: 强度数组
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (拟合后的灰度值, 拟合后的强度)
    """
    # 使用样条插值平滑曲线（如果scipy可用）

    spline = UnivariateSpline(grayscale_values, intensities, s=len(grayscale_values) * 10)
    fit_intensities = np.array(spline(grayscale_values))

    return grayscale_values, fit_intensities


def plot_calibration_result(result: CalibrationResult, save_path: Optional[Path] = None) -> None:
    """绘制标定结果曲线
    
    Args:
        result: 标定结果
        save_path: 图片保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制原始数据
        ax.plot(result.grayscale_values, result.intensities, 'b-', 
                label='衍射效率', linewidth=2)
        
        # 标记2π点
        ax.axvline(x=result.grayscale_2pi, color='r', linestyle='--', 
                   label=f'2π相位 = {result.grayscale_2pi}')
        
        ax.set_xlabel('灰度值', fontsize=12)
        ax.set_ylabel('衍射光强 (a.u.)', fontsize=12)
        ax.set_title(f'SLM标定结果 - {result.slm_model}\n'
                     f'波长: {result.wavelength_nm}nm', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"标定曲线已保存到: {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib未安装，无法绘制标定曲线")


# 便捷函数
def calibrate_santec_slm200(
    slm,
    camera,
    wavelength: int = 1064,
    grating_period: int = 8,
    output_dir: Optional[Path] = None
) -> CalibrationResult:
    """Santec SLM-200 快速标定函数
    
    Args:
        slm: SantecSLM200实例
        camera: 相机实例
        wavelength: 工作波长（nm）
        grating_period: 光栅周期
        output_dir: 输出目录
        
    Returns:
        CalibrationResult: 标定结果
    """
    calibrator = SantecSLM200Calibrator(
        slm=slm,
        camera=camera,
        grating_period=grating_period
    )
    
    result = calibrator.calibrate_with_background()
    
    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result.save(output_dir / f'slm_calibration_{wavelength}nm_{timestamp}.json')
    
    return result


if __name__ == '__main__':
    # 示例用法
    print("SLM闪耀光栅标定模块")
    print("=" * 50)
    print("""
使用示例:

    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
    from ao_shaping.drivers.slm.slm_calibration import SantecSLM200Calibrator
    from ao_shaping.drivers.ccd.daheng import CameraStreamManager
    
    # 连接设备
    with SantecSLM200(slm_number=1) as slm:
        slm.set_wavelength(1064, 200)
        
        with CameraStreamManager(cam_id=0, exposure_time_ms=50) as camera:
            # 创建标定器
            calibrator = SantecSLM200Calibrator(
                slm=slm,
                camera=camera,
                grating_period=8
            )
            
            # 执行标定
            result = calibrator.calibrate_with_background()
            
            # 保存结果
            calibrator.save_calibration('calibration_result.json')
            
            # 绘制曲线
            plot_calibration_result(result)
    """)
