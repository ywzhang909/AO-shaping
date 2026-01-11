"""
AO系统结果显示面板

集成图像显示、波前可视化和数值指标
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSplitter, QFrame
)
from PySide6.QtCore import Qt

from .image_display import ImageDisplay
from .wavefront_display import WavefrontDisplay


class ResultsPanel(QWidget):
    """
    结果显示面板

    显示仿真结果：图像、波前和性能指标
    """

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)

        # 创建分割器
        splitter = QSplitter(Qt.Vertical)

        # 上半部分：图像和波前显示
        display_widget = QWidget()
        display_layout = QHBoxLayout(display_widget)

        # 图像显示
        image_group = QGroupBox("相机图像")
        image_layout = QVBoxLayout(image_group)
        self.image_display = ImageDisplay()
        image_layout.addWidget(self.image_display)
        display_layout.addWidget(image_group)

        # 波前显示
        wavefront_group = QGroupBox("波前")
        wavefront_layout = QVBoxLayout(wavefront_group)
        self.wavefront_display = WavefrontDisplay()
        wavefront_layout.addWidget(self.wavefront_display)
        display_layout.addWidget(wavefront_group)

        splitter.addWidget(display_widget)

        # 下半部分：指标显示
        metrics_group = QGroupBox("性能指标")
        metrics_layout = QVBoxLayout(metrics_group)
        self.create_metrics_display(metrics_layout)
        splitter.addWidget(metrics_group)

        # 设置分割器比例
        splitter.setSizes([600, 200])

        layout.addWidget(splitter)

    def create_metrics_display(self, layout):
        """创建指标显示"""
        # 创建网格布局来显示多个指标
        metrics_widget = QWidget()
        metrics_layout = QHBoxLayout(metrics_widget)

        # Strehl比
        strehl_group = QVBoxLayout()
        strehl_label = QLabel("Strehl比:")
        self.strehl_value = QLabel("--")
        self.strehl_value.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        strehl_group.addWidget(strehl_label)
        strehl_group.addWidget(self.strehl_value)
        strehl_group.addStretch()
        metrics_layout.addLayout(strehl_group)

        # 分隔线
        line1 = QFrame()
        line1.setFrameShape(QFrame.VLine)
        line1.setFrameShadow(QFrame.Sunken)
        metrics_layout.addWidget(line1)

        # 总功率
        power_group = QVBoxLayout()
        power_label = QLabel("总功率:")
        self.power_value = QLabel("--")
        self.power_value.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
        power_group.addWidget(power_label)
        power_group.addWidget(self.power_value)
        power_group.addStretch()
        metrics_layout.addLayout(power_group)

        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.VLine)
        line2.setFrameShadow(QFrame.Sunken)
        metrics_layout.addWidget(line2)

        # RMS
        rms_group = QVBoxLayout()
        rms_label = QLabel("波前RMS:")
        self.rms_value = QLabel("--")
        self.rms_value.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
        rms_group.addWidget(rms_label)
        rms_group.addWidget(self.rms_value)
        rms_group.addStretch()
        metrics_layout.addLayout(rms_group)

        # 分隔线
        line3 = QFrame()
        line3.setFrameShape(QFrame.VLine)
        line3.setFrameShadow(QFrame.Sunken)
        metrics_layout.addWidget(line3)

        # PV
        pv_group = QVBoxLayout()
        pv_label = QLabel("波前PV:")
        self.pv_value = QLabel("--")
        self.pv_value.setStyleSheet("font-size: 14px; font-weight: bold; color: purple;")
        pv_group.addWidget(pv_label)
        pv_group.addWidget(self.pv_value)
        pv_group.addStretch()
        metrics_layout.addLayout(pv_group)

        metrics_layout.addStretch()
        layout.addWidget(metrics_widget)

    def update_image(self, image: np.ndarray):
        """更新图像显示"""
        self.image_display.update_image(image)

    def update_wavefront(self, slopes: np.ndarray):
        """更新波前显示"""
        self.wavefront_display.update_slopes(slopes)

    def update_metrics(self, strehl: float = None, power: float = None,
                      rms: float = None, pv: float = None):
        """更新性能指标"""
        if strehl is not None:
            self.strehl_value.setText(".3f")
        if power is not None:
            self.power_value.setText(".2e")
        if rms is not None:
            self.rms_value.setText(".2e")
        if pv is not None:
            self.pv_value.setText(".2e")</content>
</xai:function_call">结果显示面板已创建。现在创建图像显示组件，使用matplotlib嵌入到PySide6中。让我创建image_display.py文件。