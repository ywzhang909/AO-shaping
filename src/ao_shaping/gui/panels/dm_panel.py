from typing import Literal

from PySide6.QtWidgets import (
    QWidget, QLabel
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush

import numpy as np

from ao_shaping.utils import logger


class DMCircularUnit(QLabel):
    """变形镜圆形单元类"""
    
    valueChanged = Signal(int, float)  # 单元索引, 新值
    
    def __init__(self, unit_index: int, x: float, y: float, parent=None):
        super().__init__(parent)
        self.unit_index = unit_index
        self.x_coord = x
        self.y_coord = y
        self.value = 0.0  # 默认初始值为0
        
        # 设置按钮属性
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText(str(unit_index))
        self.setFont(QFont("Arial", 8))
        
        # 启用鼠标事件
        self.setMouseTracking(True)
        
        # 设置固定大小
        self.setFixedSize(40, 40)
        
        # 更新颜色显示
        self.update_color()
        
    def setValue(self, value: float):
        """设置按钮的数值并更新颜色"""
        # 限制值在-1到1之间
        self.value = max(-1.0, min(1.0, value))
        self.update_color()
        # 发出信号通知数值改变
        self.valueChanged.emit(self.unit_index, self.value)
        
    def getValue(self) -> float:
        """获取按钮的当前数值"""
        return self.value
        
    def update_color(self):
        """根据当前数值更新按钮颜色"""
        # 根据数值计算颜色
        if self.value == 0:
            # 0为绿色
            color = QColor(0, 255, 0)
        elif self.value > 0:
            # 0到1之间，从绿色渐变到红色
            # 红色分量随着值增加而增加，绿色分量随着值增加而减少
            red = int(255 * self.value)
            green = int(255 * (1 - self.value))
            color = QColor(red, green, 0)
        else:  # self.value < 0
            # -1到0之间，从绿色渐变到蓝色
            # 蓝色分量随着值减小而增加，绿色分量随着值减小而减少
            blue = int(255 * abs(self.value))
            green = int(255 * (1 - abs(self.value)))
            color = QColor(0, green, blue)
            
        # 设置背景颜色
        palette = self.palette()
        palette.setColor(self.backgroundRole(), color)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 左键点击：增加值
            new_value = self.value + 0.1
            self.setValue(new_value)
        elif event.button() == Qt.MouseButton.RightButton:
            # 右键点击：减少值
            new_value = self.value - 0.1
            self.setValue(new_value)
        super().mousePressEvent(event)
        
    def paintEvent(self, event):
        """自定义绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制圆形背景
        rect = self.rect()
        size = min(rect.width(), rect.height())
        diameter = size - 4
        x = (rect.width() - diameter) // 2
        y = (rect.height() - diameter) // 2
        
        # 根据数值计算颜色
        if self.value == 0:
            color = QColor(0, 255, 0)
        elif self.value > 0:
            red = int(255 * self.value)
            green = int(255 * (1 - self.value))
            color = QColor(red, green, 0)
        else:
            blue = int(255 * abs(self.value))
            green = int(255 * (1 - abs(self.value)))
            color = QColor(0, green, blue)
        
        # 绘制圆形
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor("black"), 1))
        painter.drawEllipse(x, y, diameter, diameter)
        
        # 绘制文本
        painter.setPen(QPen(QColor("black"), 1))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self.unit_index))


class DMPanel(QWidget):
    """变形镜面板类"""
    
    valueChanged = Signal(int, float)  # 单元索引, 新值
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buttons = []
        self.values = [0.0] * 64  # 每个单元的数值列表
        self.unit_positions = []  # 存储单元位置信息
        self.init_ui()
        self.create_buttons()
        
    def init_ui(self):
        """初始化界面"""
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(400, 400)

    @classmethod
    def from_unit_positions(cls, unit_positions: list[tuple[int, float, float]]):
        """从单元位置创建面板"""
        panel = cls()
        panel.unit_positions = unit_positions
        panel.create_buttons()
        return panel
    
    @classmethod
    def from_layout_file_to_panel(cls, layout_file: str):
        """从布局文件创建面板
        
        Parameters
        ----------
        layout_file : str
            布局文件路径
            
        Returns
        -------
        DMPanel
            创建的面板实例
        """
        unit_positions = cls.__load_unit_positions(layout_file)
        return cls.from_unit_positions(unit_positions)
    
    @classmethod
    def create_default_panel(cls, layout: Literal["grid", "circular"] = "grid", num_actuators_x: int = 8, num_actuators_y: int = 8):
        """创建默认面板
        
        Parameters
        ----------
        layout : Literal["grid", "circular"], optional
            布局类型，默认是"grid"
        num_actuators_x : int, optional
            X方向上的单元数量，默认是8
        num_actuators_y : int, optional
            Y方向上的单元数量，默认是8
            
        Returns
        -------
        DMPanel
            创建的面板实例
        """
        unit_positions = cls.__generate_actuator_positions(layout, num_actuators_x, num_actuators_y, True)
        return cls.from_unit_positions(unit_positions)

    def create_buttons(self):
        """创建所有单元按钮"""
        # 清除现有的按钮
        for button in self.buttons:
            button.deleteLater()
        self.buttons.clear()
        
        # 创建新的按钮
        for unit_index, x, y in self.unit_positions:
            button = DMCircularUnit(unit_index, x, y)
            button.setValue(self.values[unit_index - 1])  # 设置初始值
            button.valueChanged.connect(self.on_button_value_changed)
            self.buttons.append(button)
            
        # 布局所有按钮
        self.layout_buttons()
            
    def layout_buttons(self):
        """根据面板大小布局按钮"""
        if not self.unit_positions or not self.buttons:
            logger.warning("没有单元位置信息或按钮")
            return
            
        # 获取面板尺寸
        panel_width = self.width()
        panel_height = self.height()
        
        if panel_width <= 0 or panel_height <= 0:
            return
            
        # 计算坐标范围
        min_x = min(pos[1] for pos in self.unit_positions)
        max_x = max(pos[1] for pos in self.unit_positions)
        min_y = min(pos[2] for pos in self.unit_positions)
        max_y = max(pos[2] for pos in self.unit_positions)
        
        # 添加边距
        margin = 20
        coord_width = max_x - min_x
        coord_height = max_y - min_y
        
        # 避免除零错误
        if coord_width == 0:
            coord_width = 1
        if coord_height == 0:
            coord_height = 1
            
        # 计算缩放因子
        scale_x = (panel_width - 2 * margin) / coord_width
        scale_y = (panel_height - 2 * margin) / coord_height
        scale = min(scale_x, scale_y, 20)  # 限制最大按钮大小
        
        # 计算按钮大小
        button_size = max(15, int(scale * 0.8))  # 最小15像素
        
        # 布局所有按钮
        for button, (unit_index, x, y) in zip(self.buttons, self.unit_positions):
            # 计算按钮位置
            rel_x = (x - min_x) * scale_x
            rel_y = (y - min_y) * scale_y
            
            button_x = int(margin + rel_x - button_size / 2)
            button_y = int(margin + rel_y - button_size / 2)
            
            # 设置按钮位置和大小
            button.setParent(self)
            button.setGeometry(button_x, button_y, button_size, button_size)
            button.show()
            
    def resizeEvent(self, event):
        """处理面板大小调整事件"""
        super().resizeEvent(event)
        self.layout_buttons()
        
    def on_button_value_changed(self, unit_index: int, value: float):
        """处理按钮数值改变事件"""
        if 1 <= unit_index <= 64:
            self.values[unit_index - 1] = value
            # 通知主窗口数值已更改
            self.valueChanged.emit(unit_index, value)
            print(f"单元 {unit_index} 的值更新为: {value:.2f}")
            
    def get_values(self) -> list[float]:
        """获取所有单元的当前数值"""
        return self.values.copy()
        
    def set_values(self, values: list[float]):
        """设置所有单元的数值"""
        if len(values) != 64:
            raise ValueError("数值列表长度必须为64")
            
        self.values = values.copy()
        for i, button in enumerate(self.buttons):
            if i < len(values):
                button.setValue(values[i])
                
    def reset_values(self):
        """重置所有单元的数值"""
        self.values = [0.0] * 64
        # 直接设置所有按钮的值而不触发单独的更新
        for button in self.buttons:
            button.value = 0.0
        # 一次性更新所有按钮的颜色显示
        for button in self.buttons:
            button.update_color()
        # 刷新整个面板以减少重绘次数
        self.update()
        
    def set_unit_positions(self, unit_positions: list[tuple[int, float, float]]):
        """设置所有单元的位置"""
        self.unit_positions.clear()
        self.unit_positions.extend(unit_positions)
        self.layout_buttons()
        
    @staticmethod
    def __generate_actuator_positions(arrangement_type: Literal["grid", "circular"], num_actuators_x, num_actuators_y, extend_outside):
        if arrangement_type == "grid":
            if extend_outside:
                min_val, max_val = -1.2, 1.2
            else:
                min_val, max_val = -1, 1
            actuator_x_positions = np.linspace(min_val, max_val, num_actuators_x)
            actuator_y_positions = np.linspace(min_val, max_val, num_actuators_y)
            actuator_positions = [
                (enum, xx, yy) for enum, xx in enumerate(actuator_x_positions) for yy in actuator_y_positions
            ]
        elif arrangement_type == "circular":
            if extend_outside:
                max_radius = 1.2
            else:
                max_radius = 1.0
            rings = np.linspace(0.0, max_radius, num_actuators_y)
            angles = np.linspace(0, 2*np.pi, num_actuators_x, endpoint=False)
            actuator_positions = []
            for enum, r in enumerate(rings):
                for a in angles:
                    actuator_positions.append((enum, r*np.cos(a), r*np.sin(a)))
        else:
            raise ValueError("Unknown arrangement_type.")
        return actuator_positions
    
    @staticmethod
    def __load_unit_positions(layout_file):
        """从文件加载单元位置信息"""
        unit_positions = []
        with open(layout_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        unit_index = int(parts[0])
                        x_coord = float(parts[1])
                        y_coord = float(parts[2])
                        unit_positions.append((unit_index, x_coord, y_coord))
                        
        logger.info(f"成功加载 {len(unit_positions)} 个单元位置信息")
        return unit_positions