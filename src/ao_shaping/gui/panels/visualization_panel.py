import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QGraphicsView, 
                              QGraphicsScene, QGraphicsRectItem, QLabel, QSizePolicy,
                              QHBoxLayout, QFrame)
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtGui import QColor, QBrush, QPen, QFont, QPainter
from PySide6.QtCore import Qt, QRectF


class VisualizationPanel(QWidget):
    """可视化面板类 - 使用Qt原生控件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.update_pending = False  # 标记是否有待处理的更新
        self.last_voltages = None    # 存储最后一次电压数据
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 创建电压历史图表（柱状图）
        self.voltage_history_view = QGraphicsView()
        self.voltage_history_scene = QGraphicsScene()
        self.voltage_history_view.setScene(self.voltage_history_scene)
        self.voltage_history_view.setRenderHint(QPainter.Antialiasing)
        self.tab_widget.addTab(self.voltage_history_view, "电压历史")
        
        # 创建电压热力图
        self.voltage_heatmap_view = QGraphicsView()
        self.voltage_heatmap_scene = QGraphicsScene()
        self.voltage_heatmap_view.setScene(self.voltage_heatmap_scene)
        self.voltage_heatmap_view.setRenderHint(QPainter.Antialiasing)
        self.tab_widget.addTab(self.voltage_heatmap_view, "电压分布")
        
        # 创建RMS历史图表（折线图）
        self.rms_history_chart = QChart()
        self.rms_history_series = QLineSeries()
        self.rms_history_chart.addSeries(self.rms_history_series)
        self.rms_history_chart.setTitle("RMS历史")
        self.rms_history_chart.createDefaultAxes()
        self.rms_history_chart.legend().hide()
        
        # 设置轴标签
        self.rms_history_x_axis = QValueAxis()
        self.rms_history_x_axis.setTitleText("迭代次数")
        self.rms_history_chart.setAxisX(self.rms_history_x_axis, self.rms_history_series)
        
        self.rms_history_y_axis = QValueAxis()
        self.rms_history_y_axis.setTitleText("RMS值")
        self.rms_history_chart.setAxisY(self.rms_history_y_axis, self.rms_history_series)
        
        self.rms_history_view = QChartView(self.rms_history_chart)
        self.rms_history_view.setRenderHint(QPainter.Antialiasing)
        self.tab_widget.addTab(self.rms_history_view, "RMS历史")
        
        # 创建PIB历史图表（折线图）
        self.pib_history_chart = QChart()
        self.pib_history_series = QLineSeries()
        self.pib_history_chart.addSeries(self.pib_history_series)
        self.pib_history_chart.setTitle("PIB历史")
        self.pib_history_chart.createDefaultAxes()
        self.pib_history_chart.legend().hide()
        
        # 设置轴标签
        self.pib_history_x_axis = QValueAxis()
        self.pib_history_x_axis.setTitleText("迭代次数")
        self.pib_history_chart.setAxisX(self.pib_history_x_axis, self.pib_history_series)
        
        self.pib_history_y_axis = QValueAxis()
        self.pib_history_y_axis.setTitleText("PIB值")
        self.pib_history_chart.setAxisY(self.pib_history_y_axis, self.pib_history_series)
        
        self.pib_history_view = QChartView(self.pib_history_chart)
        self.pib_history_view.setRenderHint(QPainter.Antialiasing)
        self.tab_widget.addTab(self.pib_history_view, "PIB历史")
        
        # 初始化数据存储
        self.voltage_history = []  # 存储电压历史数据
        self.rms_history = []      # 存储RMS历史数据
        self.pib_history = []      # 存储PIB历史数据
        
        # 连接选项卡切换信号
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
    def on_tab_changed(self, index):
        """处理选项卡切换事件"""
        # 当选项卡切换时，如果有待处理的更新，则执行更新
        if self.update_pending and self.last_voltages is not None:
            self.update_plots_immediately(self.last_voltages)
            self.update_pending = False
            self.last_voltages = None
            
    def update_plots(self, voltages):
        """更新所有图表"""
        # 检查当前是否在正确的选项卡上
        current_tab_index = self.tab_widget.currentIndex()
        
        # 如果当前选项卡不是电压历史或电压分布，则推迟更新
        if current_tab_index not in [0, 1]:  # 0: 电压历史, 1: 电压分布
            self.update_pending = True
            self.last_voltages = voltages.copy()
            return
            
        # 立即更新图表
        self.update_plots_immediately(voltages)
        
        # 添加当前电压到历史记录
        self.voltage_history.append(voltages.copy())
        
        # 限制历史记录长度以提高性能
        if len(self.voltage_history) > 100:
            self.voltage_history.pop(0)
            
    def update_plots_immediately(self, voltages):
        """立即更新所有图表"""
        # 更新电压历史图表
        self.update_voltage_history_plot(voltages)
        
        # 更新电压热力图
        self.update_voltage_heatmap(voltages)
            
    def update_voltage_history_plot(self, voltages):
        """更新电压历史图表（使用Qt Graphics View）"""
        # 清除之前的图形
        self.voltage_history_scene.clear()
        
        if not voltages:
            return
            
        # 设置场景大小
        scene_width = 800
        scene_height = 400
        self.voltage_history_scene.setSceneRect(0, 0, scene_width, scene_height)
        
        # 计算柱状图参数
        bar_width = scene_width / len(voltages) * 0.8
        bar_spacing = scene_width / len(voltages) * 0.2
        max_height = scene_height * 0.8
        
        # 绘制标题
        title = self.voltage_history_scene.addText("当前电压分布", QFont("Arial", 12, QFont.Bold))
        title.setPos(scene_width / 2 - title.boundingRect().width() / 2, 10)
        
        # 绘制Y轴标签
        y_axis_label = self.voltage_history_scene.addText("电压值", QFont("Arial", 10))
        y_axis_label.setPos(10, scene_height / 2 - y_axis_label.boundingRect().height() / 2)
        
        # 绘制X轴标签
        x_axis_label = self.voltage_history_scene.addText("单元索引", QFont("Arial", 10))
        x_axis_label.setPos(scene_width / 2 - x_axis_label.boundingRect().width() / 2, scene_height - 20)
        
        # 绘制柱状图
        for i, voltage in enumerate(voltages):
            # 计算柱子高度（将-1到1映射到0到max_height）
            bar_height = abs(voltage) * max_height / 2
            x_pos = i * (bar_width + bar_spacing) + bar_spacing / 2 + 50
            
            # 确定颜色（负值为蓝色，正值为红色，0为绿色）
            if voltage > 0:
                color = QColor(255, 0, 0, 150)  # 红色
            elif voltage < 0:
                color = QColor(0, 0, 255, 150)  # 蓝色
            else:
                color = QColor(0, 255, 0, 150)  # 绿色
            
            # 创建柱子
            if voltage >= 0:
                # 正值从中心向上绘制
                bar = self.voltage_history_scene.addRect(
                    x_pos, 
                    scene_height / 2 - bar_height, 
                    bar_width, 
                    bar_height, 
                    QPen(Qt.black, 1), 
                    QBrush(color)
                )
            else:
                # 负值从中心向下绘制
                bar = self.voltage_history_scene.addRect(
                    x_pos, 
                    scene_height / 2, 
                    bar_width, 
                    bar_height, 
                    QPen(Qt.black, 1), 
                    QBrush(color)
                )
            
            # 每隔一定间隔添加标签
            if i % 8 == 0:
                label = self.voltage_history_scene.addText(str(i), QFont("Arial", 8))
                label.setPos(x_pos + bar_width / 2 - label.boundingRect().width() / 2, scene_height / 2 + 10)
        
        # 绘制中心线
        center_line = self.voltage_history_scene.addLine(50, scene_height / 2, scene_width - 50, scene_height / 2, QPen(Qt.black, 1, Qt.DashLine))
        
        # 绘制Y轴刻度线和标签
        for i in range(-1, 2):
            y_pos = scene_height / 2 - i * max_height / 2
            line = self.voltage_history_scene.addLine(45, y_pos, 50, y_pos, QPen(Qt.black, 1))
            label = self.voltage_history_scene.addText(str(i), QFont("Arial", 8))
            label.setPos(10, y_pos - label.boundingRect().height() / 2)
        
    def update_voltage_heatmap(self, voltages):
        """更新电压热力图（使用Qt Graphics View）"""
        # 清除之前的图形
        self.voltage_heatmap_scene.clear()
        
        if not voltages or len(voltages) != 64:
            return
            
        # 设置场景大小
        scene_size = 400
        self.voltage_heatmap_scene.setSceneRect(0, 0, scene_size, scene_size)
        
        # 计算单元格大小
        cell_size = scene_size / 8
        
        # 将64个电压值重塑为8x8矩阵
        voltage_matrix = np.array(voltages).reshape(8, 8)
        
        # 绘制标题
        title = self.voltage_heatmap_scene.addText("电压分布热力图", QFont("Arial", 12, QFont.Bold))
        title.setPos(scene_size / 2 - title.boundingRect().width() / 2, 10)
        
        # 绘制热力图
        for i in range(8):
            for j in range(8):
                voltage = voltage_matrix[i, j]
                
                # 根据电压值计算颜色（从蓝到红，0为白色）
                if voltage > 0:
                    # 正值：从白色到红色
                    red = 255
                    green = int(255 * (1 - voltage))
                    blue = int(255 * (1 - voltage))
                else:
                    # 负值或0：从蓝色到白色
                    red = int(255 * (1 - abs(voltage)))
                    green = int(255 * (1 - abs(voltage)))
                    blue = 255
                
                color = QColor(red, green, blue)
                
                # 创建单元格矩形
                x_pos = j * cell_size + scene_size * 0.1
                y_pos = i * cell_size + 50
                rect = self.voltage_heatmap_scene.addRect(
                    x_pos, 
                    y_pos, 
                    cell_size, 
                    cell_size, 
                    QPen(Qt.black, 1), 
                    QBrush(color)
                )
                
                # 添加数值标注
                text = self.voltage_heatmap_scene.addText(f'{voltage:.2f}', QFont("Arial", 6))
                text.setDefaultTextColor(Qt.black)
                text.setPos(
                    x_pos + cell_size / 2 - text.boundingRect().width() / 2,
                    y_pos + cell_size / 2 - text.boundingRect().height() / 2
                )
        
    def add_rms_value(self, rms_value):
        """添加RMS值到历史记录"""
        self.rms_history.append(rms_value)
        
        # 更新RMS历史图表
        # 清除之前的数据点
        self.rms_history_series.clear()
        
        # 添加所有数据点
        for i, value in enumerate(self.rms_history):
            self.rms_history_series.append(i, value)
        
        # 更新坐标轴范围
        if self.rms_history:
            self.rms_history_x_axis.setRange(0, len(self.rms_history) - 1)
            min_val = min(self.rms_history)
            max_val = max(self.rms_history)
            # 添加一些边距
            margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            self.rms_history_y_axis.setRange(min_val - margin, max_val + margin)
        
        # 限制历史记录长度
        if len(self.rms_history) > 1000:
            self.rms_history.pop(0)
            
    def add_pib_value(self, pib_value):
        """添加PIB值到历史记录"""
        self.pib_history.append(pib_value)
        
        # 更新PIB历史图表
        # 清除之前的数据点
        self.pib_history_series.clear()
        
        # 添加所有数据点
        for i, value in enumerate(self.pib_history):
            self.pib_history_series.append(i, value)
        
        # 更新坐标轴范围
        if self.pib_history:
            self.pib_history_x_axis.setRange(0, len(self.pib_history) - 1)
            min_val = min(self.pib_history)
            max_val = max(self.pib_history)
            # 添加一些边距
            margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            self.pib_history_y_axis.setRange(min_val - margin, max_val + margin)
        
        # 限制历史记录长度
        if len(self.pib_history) > 1000:
            self.pib_history.pop(0)