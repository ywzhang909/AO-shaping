"""
波前可视化组件

显示波前传感器测量的斜率数据
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton
)
from PySide6.QtCore import Qt


class WavefrontDisplay(QWidget):
    """
    波前显示组件

    显示波前斜率，支持不同可视化模式
    """

    def __init__(self):
        super().__init__()
        self.slopes_data = None
        self.subapertures = 8  # 默认值
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)

        # 创建matplotlib图形
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

        # 初始显示
        self.ax.text(0.5, 0.5, '无波前数据',
                    transform=self.ax.transAxes,
                    ha='center', va='center',
                    fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')

        layout.addWidget(self.canvas)

        # 控制面板
        control_layout = QHBoxLayout()

        # 显示模式选择
        mode_layout = QVBoxLayout()
        mode_label = QLabel("显示模式:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            '向量场', 'X斜率', 'Y斜率', '斜率幅度'
        ])
        self.mode_combo.setCurrentText('向量场')
        self.mode_combo.currentTextChanged.connect(self.update_display)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        control_layout.addLayout(mode_layout)

        # 自动缩放按钮
        auto_scale_btn = QPushButton("自动缩放")
        auto_scale_btn.clicked.connect(self.auto_scale)
        control_layout.addWidget(auto_scale_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

    def update_slopes(self, slopes: np.ndarray):
        """更新斜率数据"""
        self.slopes_data = slopes.copy()
        self.update_display()

    def update_display(self):
        """更新波前显示"""
        if self.slopes_data is None:
            return

        self.ax.clear()

        mode = self.mode_combo.currentText()

        if mode == '向量场':
            self.plot_vector_field()
        elif mode == 'X斜率':
            self.plot_slope_component(0)
        elif mode == 'Y斜率':
            self.plot_slope_component(1)
        elif mode == '斜率幅度':
            self.plot_slope_magnitude()

        # 设置标题
        self.ax.set_title(f'波前斜率 - {mode}')
        self.ax.set_xlabel('X 位置')
        self.ax.set_ylabel('Y 位置')

        # 刷新画布
        self.canvas.draw()

    def plot_vector_field(self):
        """绘制向量场"""
        num_subaps = self.subapertures ** 2
        x_slopes = self.slopes_data[:num_subaps]
        y_slopes = self.slopes_data[num_subaps:]

        # 创建子孔径位置网格
        x_pos = []
        y_pos = []

        for i in range(self.subapertures):
            for j in range(self.subapertures):
                x_pos.append(j + 0.5)  # 子孔径中心X坐标
                y_pos.append(i + 0.5)  # 子孔径中心Y坐标

        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)

        # 绘制向量场
        self.ax.quiver(x_pos, y_pos, x_slopes, y_slopes,
                      angles='xy', scale_units='xy', scale=1,
                      color='blue', alpha=0.7)

        # 设置轴范围
        self.ax.set_xlim(-0.5, self.subapertures + 0.5)
        self.ax.set_ylim(-0.5, self.subapertures + 0.5)
        self.ax.set_aspect('equal')

        # 添加网格
        self.ax.grid(True, alpha=0.3)

    def plot_slope_component(self, component: int):
        """绘制斜率分量"""
        num_subaps = self.subapertures ** 2
        if component == 0:
            slopes = self.slopes_data[:num_subaps]
            title_comp = 'X'
        else:
            slopes = self.slopes_data[num_subaps:]
            title_comp = 'Y'

        # 重塑为网格
        slope_grid = slopes.reshape((self.subapertures, self.subapertures))

        # 显示为图像
        im = self.ax.imshow(slope_grid, origin='lower', cmap='RdBu_r',
                           extent=[0, self.subapertures, 0, self.subapertures],
                           aspect='equal')

        # 添加颜色条
        self.figure.colorbar(im, ax=self.ax, shrink=0.8)

        self.ax.set_xlim(0, self.subapertures)
        self.ax.set_ylim(0, self.subapertures)

    def plot_slope_magnitude(self):
        """绘制斜率幅度"""
        num_subaps = self.subapertures ** 2
        x_slopes = self.slopes_data[:num_subaps]
        y_slopes = self.slopes_data[num_subaps:]

        magnitudes = np.sqrt(x_slopes**2 + y_slopes**2)

        # 重塑为网格
        mag_grid = magnitudes.reshape((self.subapertures, self.subapertures))

        # 显示为图像
        im = self.ax.imshow(mag_grid, origin='lower', cmap='viridis',
                           extent=[0, self.subapertures, 0, self.subapertures],
                           aspect='equal')

        # 添加颜色条
        self.figure.colorbar(im, ax=self.ax, shrink=0.8)

        self.ax.set_xlim(0, self.subapertures)
        self.ax.set_ylim(0, self.subapertures)

    def auto_scale(self):
        """自动调整显示范围"""
        if self.slopes_data is None:
            return

        mode = self.mode_combo.currentText()

        if mode in ['X斜率', 'Y斜率', '斜率幅度']:
            # 对于图像模式，颜色映射已经自动缩放
            pass
        elif mode == '向量场':
            # 对于向量场，可以调整向量长度
            pass

        self.update_display()

    def set_subapertures(self, subapertures: int):
        """设置子孔径数量"""
        self.subapertures = subapertures
        if self.slopes_data is not None:
            self.update_display()</content>
</xai:function_call">波前可视化组件已创建。现在创建工作线程类，用于处理仿真计算，避免GUI冻结。让我创建simulation_worker.py文件。