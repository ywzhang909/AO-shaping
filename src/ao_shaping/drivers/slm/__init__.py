"""SLM (Spatial Light Modulator) 驱动模块

提供空间光调制器设备的驱动支持。

目前支持:
- Santec SLM-200系列（SDK原生）
- SLM闪耀光栅标定

Example:
    >>> from ao_shaping.drivers.slm import SantecSLM200
    >>> 
    >>> with SantecSLM200(slm_number=1, wavelength=1064) as slm:
    ...     # 加载相位数据
    ...     phase = np.zeros((1080, 1920), dtype=np.uint16)
    ...     # 写入并显示
    ...     slm.write_phase(phase, memory_number=1)
    ...     slm.display_memory(1)


Calibration Example:
    >>> from ao_shaping.drivers.slm import SantecSLM200Calibrator, plot_calibration_result
    >>> 
    >>> calibrator = SantecSLM200Calibrator(slm=slm, camera=camera)
    >>> result = calibrator.calibrate_with_background()
    >>> plot_calibration_result(result)
"""

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200, SantecSLM200Error
from ao_shaping.drivers.slm.zernike_slm import ZernikeSLM, ZernikeSLMError

__all__ = [
    "SantecSLM200",
    "SantecSLM200Error",
    "ZernikeSLM",
    "ZernikeSLMError",
]

# Note: PatternHelper is in ao_shaping.utils.pattern_helper



# 导入标定模块
from ao_shaping.drivers.slm.slm_calibration import (
    SLMCalibratorBase,
    SantecSLM200Calibrator,
    CalibrationResult,
    plot_calibration_result,
    calibrate_santec_slm200,
)
__all__ += [
    "CalibrationResult",
    "SLMCalibratorBase",
    "SantecSLM200Calibrator",
    "calibrate_santec_slm200",
    "plot_calibration_result",
]
