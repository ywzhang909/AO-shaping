import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget


class VisualizationPanel(QWidget):
    """可视化面板类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 创建电压历史图表
        self.voltage_history_fig = Figure(figsize=(5, 4), dpi=100)
        self.voltage_history_canvas = FigureCanvas(self.voltage_history_fig)
        self.voltage_history_ax = self.voltage_history_fig.add_subplot(111)
        self.voltage_history_ax.set_title("电压历史")
        self.voltage_history_ax.set_xlabel("单元索引")
        self.voltage_history_ax.set_ylabel("电压值")
        self.tab_widget.addTab(self.voltage_history_canvas, "电压历史")
        
        # 创建电压热力图
        self.voltage_heatmap_fig = Figure(figsize=(5, 4), dpi=100)
        self.voltage_heatmap_canvas = FigureCanvas(self.voltage_heatmap_fig)
        self.voltage_heatmap_ax = self.voltage_heatmap_fig.add_subplot(111)
        self.voltage_heatmap_ax.set_title("电压分布")
        self.tab_widget.addTab(self.voltage_heatmap_canvas, "电压分布")
        
        # 创建RMS历史图表
        self.rms_history_fig = Figure(figsize=(5, 4), dpi=100)
        self.rms_history_canvas = FigureCanvas(self.rms_history_fig)
        self.rms_history_ax = self.rms_history_fig.add_subplot(111)
        self.rms_history_ax.set_title("RMS历史")
        self.rms_history_ax.set_xlabel("迭代次数")
        self.rms_history_ax.set_ylabel("RMS值")
        self.tab_widget.addTab(self.rms_history_canvas, "RMS历史")
        
        # 创建PIB历史图表
        self.pib_history_fig = Figure(figsize=(5, 4), dpi=100)
        self.pib_history_canvas = FigureCanvas(self.pib_history_fig)
        self.pib_history_ax = self.pib_history_fig.add_subplot(111)
        self.pib_history_ax.set_title("PIB历史")
        self.pib_history_ax.set_xlabel("迭代次数")
        self.pib_history_ax.set_ylabel("PIB值")
        self.tab_widget.addTab(self.pib_history_canvas, "PIB历史")
        
        # 初始化数据存储
        self.voltage_history = []  # 存储电压历史数据
        self.rms_history = []      # 存储RMS历史数据
        self.pib_history = []      # 存储PIB历史数据
        
    def update_plots(self, voltages):
        """更新所有图表"""
        # 更新电压历史图表
        self.update_voltage_history_plot(voltages)
        
        # 更新电压热力图
        self.update_voltage_heatmap(voltages)
        
        # 添加当前电压到历史记录
        self.voltage_history.append(voltages.copy())
        
        # 限制历史记录长度以提高性能
        if len(self.voltage_history) > 100:
            self.voltage_history.pop(0)
            
    def update_voltage_history_plot(self, voltages):
        """更新电压历史图表"""
        self.voltage_history_ax.clear()
        self.voltage_history_ax.bar(range(len(voltages)), voltages)
        self.voltage_history_ax.set_title("当前电压分布")
        self.voltage_history_ax.set_xlabel("单元索引")
        self.voltage_history_ax.set_ylabel("电压值")
        self.voltage_history_ax.set_ylim(-1.1, 1.1)
        self.voltage_history_canvas.draw()
        
    def update_voltage_heatmap(self, voltages):
        """更新电压热力图"""
        self.voltage_heatmap_ax.clear()
        
        # 将64个电压值重塑为8x8矩阵
        voltage_matrix = np.array(voltages).reshape(8, 8)
        
        # 显示热力图
        im = self.voltage_heatmap_ax.imshow(voltage_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        self.voltage_heatmap_ax.set_title('电压分布热力图')
        
        # 添加颜色条
        self.voltage_heatmap_fig.colorbar(im, ax=self.voltage_heatmap_ax)
        
        # 添加数值标注
        for i in range(8):
            for j in range(8):
                self.voltage_heatmap_ax.text(j, i, f'{voltage_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=6)
        
        self.voltage_heatmap_canvas.draw()
        
    def add_rms_value(self, rms_value):
        """添加RMS值到历史记录"""
        self.rms_history.append(rms_value)
        
        # 更新RMS历史图表
        self.rms_history_ax.clear()
        self.rms_history_ax.plot(self.rms_history)
        self.rms_history_ax.set_title("RMS历史")
        self.rms_history_ax.set_xlabel("迭代次数")
        self.rms_history_ax.set_ylabel("RMS值")
        self.rms_history_canvas.draw()
        
        # 限制历史记录长度
        if len(self.rms_history) > 1000:
            self.rms_history.pop(0)
            
    def add_pib_value(self, pib_value):
        """添加PIB值到历史记录"""
        self.pib_history.append(pib_value)
        
        # 更新PIB历史图表
        self.pib_history_ax.clear()
        self.pib_history_ax.plot(self.pib_history)
        self.pib_history_ax.set_title("PIB历史")
        self.pib_history_ax.set_xlabel("迭代次数")
        self.pib_history_ax.set_ylabel("PIB值")
        self.pib_history_canvas.draw()
        
        # 限制历史记录长度
        if len(self.pib_history) > 1000:
            self.pib_history.pop(0)