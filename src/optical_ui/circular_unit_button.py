from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

# 定义常量
MIN_VALUE = -1.0
MAX_VALUE = 1.0
DELTA = 0.1


class CircularUnitButton(QLabel):
    """自定义圆形按钮类，表示一个光学单元"""
    
    # 定义信号，当数值改变时发出
    valueChanged = Signal(int, float)  # 单元索引, 新值
    
    def __init__(self, unit_index: int, x: float, y: float, parent=None):
        super().__init__(parent)
        self.unit_index = unit_index
        self.x_coord = x
        self.y_coord = y
        self.value = 0.0  # 默认初始值为0
        
        # 设置按钮属性
        self.setAlignment(Qt.AlignCenter)
        self.setText(str(unit_index))
        self.setFont(QFont("Arial", 8))
        
        # 启用鼠标事件
        self.setMouseTracking(True)
        
    def setValue(self, value: float):
        """设置按钮的数值并更新颜色"""
        # 确保值在有效范围内
        self.value = max(MIN_VALUE, min(MAX_VALUE, value))
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
            
        # 应用颜色作为背景
        palette = self.palette()
        palette.setColor(self.backgroundRole(), color)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            # 左键点击：增加值
            new_value = self.value + DELTA
            self.setValue(new_value)
        elif event.button() == Qt.RightButton:
            # 右键点击：减少值
            new_value = self.value - DELTA
            self.setValue(new_value)
        super().mousePressEvent(event)
        
    def resizeEvent(self, event):
        """处理大小调整事件"""
        super().resizeEvent(event)
        # 确保按钮保持圆形
        size = min(self.width(), self.height())
        self.resize(size, size)