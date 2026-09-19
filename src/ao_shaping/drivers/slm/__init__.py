"""SLM (Spatial Light Modulator) 驱动模块

提供空间光调制器设备的驱动支持。

目前支持:
- Santec 系列 (SLM-200/SLM-300 等, SDK 原生)
- SLM闪耀光栅标定

Example:
    >>> from ao_shaping.drivers.slm import Santec
    >>>
    >>> with Santec(slm_number=1, wavelength=1064) as slm:
    ...     # 加载相位数据
    ...     phase = np.zeros((1080, 1920), dtype=np.uint16)
    ...     # 写入并显示
    ...     slm.display_data(phase, memory_number=1)
    ...     slm._display_memory(1)


Calibration Example:
    >>> from ao_shaping.tools.slm.calibration import SantecCalibrator, plot_calibration_result
    >>>
    >>> calibrator = SantecCalibrator(slm=slm, camera=camera)
    >>> result = calibrator.calibrate_with_background()
    >>> plot_calibration_result(result)
"""

from ao_shaping.drivers.slm.santec import Santec, SantecError
from ao_shaping.drivers.slm.zernike_slm import ZernikeSLM, ZernikeSLMError

__all__ = [
    "Santec",
    "SantecError",
    "ZernikeSLM",
    "ZernikeSLMError",
]

# Note: PatternHelper is in ao_shaping.utils.pattern_helper