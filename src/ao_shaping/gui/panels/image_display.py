"""
图像显示组件

使用matplotlib显示相机图像，支持缩放和颜色映射
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QCheckBox
)
from PySide6.QtCore import Qt


class ImageDisplay(QWidget):
    """
    图像显示组件

    显示相机图像，支持颜色映射和缩放
    """

    def __init__(self):
        super().__init__()
        self.image_data = None
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)

        # 创建matplotlib图形
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

        # 初始显示
        self.ax.text(0.5, 0.5, '无图像数据',
                    transform=self.ax.transAxes,
                    ha='center', va='center',
                    fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        layout.addWidget(self.canvas)

        # 控制面板
        control_layout = QHBoxLayout()

        # 颜色映射选择
        cmap_layout = QVBoxLayout()
        cmap_label = QLabel("颜色映射:")
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            'viridis', 'plasma', 'inferno', 'magma',
            'gray', 'hot', 'cool', 'jet'
        ])
        self.cmap_combo.setCurrentText('gray')
        self.cmap_combo.currentTextChanged.connect(self.update_display)
        cmap_layout.addWidget(cmap_label)
        cmap_layout.addWidget(self.cmap_combo)
        control_layout.addLayout(cmap_layout)

        # 缩放控制
        scale_layout = QVBoxLayout()
        scale_label = QLabel("缩放:")
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(50, 200)  # 50% 到 200%
        self.scale_slider.setValue(100)
        self.scale_slider.setTickInterval(25)
        self.scale_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_slider.valueChanged.connect(self.update_display)
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_slider)
        control_layout.addLayout(scale_layout)

        # 对数缩放
        log_layout = QVBoxLayout()
        log_label = QLabel("对数显示:")
        self.log_check = QCheckBox()
        self.log_check.stateChanged.connect(self.update_display)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_check)
        control_layout.addLayout(log_layout)

        control_layout.addStretch()
        layout.addLayout(control_layout)

    def update_image(self, image: np.ndarray):
        """更新显示的图像"""
        self.image_data = image.astype(float)
        self.update_display()

    def update_display(self):
        """更新图像显示"""
        if self.image_data is None:
            return

        self.ax.clear()

        # 获取显示参数
        cmap = self.cmap_combo.currentText()
        scale = self.scale_slider.value() / 100.0
        use_log = self.log_check.isChecked()

        # 准备图像数据
        display_data = self.image_data.copy()

        if use_log:
            # 对数缩放，避免零值
            display_data = np.log10(display_data - display_data.min() + 1e-10)

        # 显示图像
        im = self.ax.imshow(display_data,
                           cmap=cmap,
                           origin='lower',
                           extent=[0, display_data.shape[1] * scale,
                                  0, display_data.shape[0] * scale],
                           aspect='equal')

        # 添加颜色条
        self.figure.colorbar(im, ax=self.ax, shrink=0.8)

        # 设置标题
        self.ax.set_title(f'相机图像 ({display_data.shape[0]}×{display_data.shape[1]})')

        # 刷新画布
        self.canvas.draw()

    def get_image_stats(self) -> dict:
        """获取图像统计信息"""
        if self.image_data is None:
            return {}

        return {
            'shape': self.image_data.shape,
            'min': float(np.min(self.image_data)),
            'max': float(np.max(self.image_data)),
            'mean': float(np.mean(self.image_data)),
            'std': float(np.std(self.image_data))
        }</content>
</xai:function_call">图像显示组件已创建。现在创建波前可视化组件，显示波前斜率。让我创建wavefront_display.py文件。