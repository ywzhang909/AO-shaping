from typing import List
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtGui import QPainter, QColor, QFont, QPixmap


class ImagePanel(QWidget):
    """numpy二维图片展示面板类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_data = np.zeros((8, 8))  # 初始化8x8的零矩阵
        self.pixmap = None
        self.setMinimumSize(400, 400)
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标题
        title_label = QLabel('数值矩阵可视化')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # 设置策略以填充可用空间
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def update_image(self, values: List[float] = None):
        """更新图像显示"""
        if values is not None and len(values) == 64:
            # 将一维数组重塑为8x8矩阵
            self.image_data = np.array(values).reshape(8, 8)
            
        # 重新绘制图像
        self.update()
        
    def paintEvent(self, event):
        """绘制事件处理"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取绘制区域
        rect = self.rect()
        # 减去标题和其他边距的空间
        content_rect = QRect(rect.left() + 20, rect.top() + 50, 
                           rect.width() - 40, rect.height() - 70)
        
        if content_rect.width() <= 0 or content_rect.height() <= 0:
            return
            
        # 计算单元格大小
        cell_width = content_rect.width() // 8
        cell_height = content_rect.height() // 8
        cell_size = min(cell_width, cell_height)
        
        # 居中绘制
        total_width = cell_size * 8
        total_height = cell_size * 8
        start_x = content_rect.left() + (content_rect.width() - total_width) // 2
        start_y = content_rect.top() + (content_rect.height() - total_height) // 2
        
        # 绘制8x8矩阵
        for i in range(8):
            for j in range(8):
                # 计算颜色基于数值 (-1 到 1)
                value = self.image_data[i, j]
                color = self._get_color_from_value(value)
                
                # 绘制单元格
                cell_rect = QRect(start_x + j * cell_size, 
                                 start_y + i * cell_size, 
                                 cell_size, cell_size)
                painter.fillRect(cell_rect, color)
                
                # 绘制边框
                painter.setPen(QColor(100, 100, 100))
                painter.drawRect(cell_rect)
                
                # 绘制数值文本
                painter.setPen(QColor(0, 0, 0))
                painter.setFont(QFont("Arial", max(6, cell_size // 6)))
                text = f"{value:.2f}"
                text_rect = painter.boundingRect(cell_rect, Qt.AlignCenter, text)
                painter.drawText(cell_rect, Qt.AlignCenter, text)
                
        # 绘制标题
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        title_text = "8x8 数值矩阵"
        title_rect = QRect(start_x, start_y - 30, total_width, 20)
        painter.drawText(title_rect, Qt.AlignCenter, title_text)
        
    def _get_color_from_value(self, value):
        """根据数值获取颜色，仿照matplotlib的RdYlBu_r色彩映射"""
        # 限制值在-1到1之间
        value = max(-1.0, min(1.0, value))
        
        # RdYlBu_r 颜色映射近似实现
        # 蓝色 (-1.0) -> 白色 (0.0) -> 红色 (1.0)
        if value < 0:
            # 蓝色到白色的过渡
            factor = (value + 1.0)  # 0到1
            red = int(255 * factor)
            green = int(255 * factor)
            blue = int(255 * (0.5 + 0.5 * factor))  # 更亮的蓝色
        else:
            # 白色到红色的过渡
            factor = value  # 0到1
            red = int(255)
            green = int(255 * (1.0 - factor))
            blue = int(255 * (1.0 - factor))
            
        return QColor(red, green, blue)
        
    def sizeHint(self):
        """返回推荐大小"""
        return QSize(500, 500)