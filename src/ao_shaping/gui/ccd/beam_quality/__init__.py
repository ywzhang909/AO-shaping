"""光束质量分析 (PySide6): MiiCam 实时画面 + 峰值趋势 + 剖面拟合。

模块划分:

* :mod:`~.analyzer`      逐帧数值分析 (纯 numpy/scipy, 可单测)
* :mod:`~.frame_source`  帧源抽象 (MiiCam 回调 / 离屏仿真), 最新帧信箱背压
* :mod:`~.trend_buffer`  趋势环形缓冲 + CSV 导出 (纯 python)
* :mod:`~.trend_widget`  峰值趋势折线图
* :mod:`~.video_widget`  画面 + 质心/椭圆叠加
* :mod:`~.profile_widget` X/Y 剖面与高斯拟合
* :mod:`~.app`           主窗口与 ``main()``

入口::

    python -m ao_shaping.gui.ccd.beam_quality
"""

from __future__ import annotations

from ao_shaping.gui.ccd.beam_quality.app import BeamQualityApp, main

__all__ = ["BeamQualityApp", "main"]
