"""桌面控制台 (PySide6): 光束质量分析 + SLM 迁移学习闭环整形。

模块划分:

* :mod:`~.analyzer`      逐帧数值分析 (纯 numpy/scipy, 可单测)
* :mod:`~.frame_source`  帧源抽象 (MiiCam 回调 / 离屏仿真), 最新帧信箱背压
* :mod:`~.trend_buffer`  趋势环形缓冲 + CSV 导出 (纯 python)
* :mod:`~.trend_widget`  峰值趋势折线图
* :mod:`~.video_widget`  画面 + 质心/椭圆叠加
* :mod:`~.profile_widget` X/Y 剖面与高斯拟合
* :mod:`~.app`           光束质量主窗口 (:class:`BeamQualityApp`)
* :mod:`~.shell`         多页外壳 (:class:`DesktopShellApp`), 左侧导航 + 页面栈;
  页 0 嵌入 :class:`BeamQualityApp`, 页 1 惰性创建 SLM 迁移学习页

入口::

    python -m ao_shaping.gui.ccd.beam_quality          # 多页外壳 (默认)
    python -m ao_shaping.gui.ccd.beam_quality.shell    # 同上

单页仍可直接用: ``BeamQualityApp()``。
"""

from __future__ import annotations

from ao_shaping.gui.ccd.beam_quality.app import BeamQualityApp

__all__ = ["BeamQualityApp", "main"]


def main(argv: list[str] | None = None) -> int:
    """启动多页桌面外壳 (与 ``shell.main`` 同一入口, 惰性 import)。"""
    from ao_shaping.gui.ccd.beam_quality.shell import main as shell_main

    return shell_main(argv)
