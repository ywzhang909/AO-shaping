"""SLM (Spatial Light Modulator) 驱动模块

提供空间光调制器设备的驱动支持。

目前支持:
- Santec 系列 (SLM-200/SLM-300 等, SDK 原生)
- SLM闪耀光栅标定

🔴 **采集实验开始前先断言驱动板健康**
------------------------------------------------------------------------------
Santec 控制器上有驱动板 (Drive board) 与选板 (Option board) 两块板, 各有一个
ID, 二者属于**同一台**设备。2026-10-22 实测确认的静默失效: 驱动板不应答、
只剩选板应答时, 驱动会**静默回退**到选板 ID 当设备身份, 于是整轮实验在面板
根本不调制 (离焦响应落在噪声级) 的状态下跑完, 数据全无效却没有任何报错,
还会被误读成光路故障。

因此实验流程必须显式断言:

    from ao_shaping.drivers.slm import Santec, SLMDriveBoardError

    with Santec(slm_number=1) as slm:
        slm.require_drive_board()   # 驱动板不应答 → 抛 SLMDriveBoardError
        ...                          # 之后才开始采集

    # 或者只查不抛:
    status = slm.check_drive_board()   # DriveBoardStatus (只读, 不改设备状态)
    print(slm.serial_source())         # "driveboard" / "driveboard_readsd"
                                       # / "optionboard_fallback" / None

``Santec.open()`` 本身**不抛** (保守策略, 不打断既有流程), 只记录 warning
级日志与可查询的健康状态 (``slm.drive_board_status()``)。

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
from __future__ import annotations

from ao_shaping.drivers.slm.santec import (
    DriveBoardStatus,
    Santec,
    SantecError,
    SerialSource,
    SLMDriveBoardError,
)
from ao_shaping.drivers.slm.zernike_slm import ZernikeSLM, ZernikeSLMError

__all__ = [
    "DriveBoardStatus",
    "Santec",
    "SantecError",
    "SerialSource",
    "SLMDriveBoardError",
    "ZernikeSLM",
    "ZernikeSLMError",
]

# 注意: PatternHelper 位于 ao_shaping.utils.pattern_helper
