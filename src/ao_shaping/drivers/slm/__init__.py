"""SLM (Spatial Light Modulator) 驱动模块

提供空间光调制器设备的驱动支持。

目前支持:
- Santec SLM-200系列（SDK原生）
- Santec SLM-200系列（PyVISA兼容）

Example:
    >>> from ao_shaping.drivers.slm import SantecSLM200
    >>> 
    >>> with SantecSLM200(slm_number=1, wavelength=1064) as slm:
    ...     # 加载相位数据
    ...     phase = np.zeros((1080, 1920), dtype=np.uint16)
    ...     # 写入并显示
    ...     slm.write_phase(phase, memory_number=1)
    ...     slm.display_memory(1)

PyVISA Example:
    >>> from ao_shaping.drivers.slm import SantecSLM200Visa
    >>> 
    >>> with SantecSLM200Visa('SLM::1::INSTR') as slm:
    ...     slm.write('WAVELENGTH 1064')
    ...     slm.query('*IDN?')
"""

from .santec_slm200 import SantecSLM200, SantecSLM200Error

__all__ = ["SantecSLM200", "SantecSLM200Error"]

# 可选导入 PyVISA 兼容类
try:
    from .santec_slm200_visa import SantecSLM200Visa, create_slm_visa_instrument
    __all__ += ["SantecSLM200Visa", "create_slm_visa_instrument"]
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"SantecSLM200Visa not available: {e}")
