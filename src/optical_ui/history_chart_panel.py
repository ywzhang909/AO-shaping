from typing import Dict, List
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtGui import QPainter, QColor, QFont, QPen


class HistoryChartPanel(QWidget):
    """历史数据图表面板类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_data: Dict[int, List[float]] = {}  # 存储每个单元的历史数据
        self.setMinimumSize(400, 300)
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标题
        title_label = QLabel('单元数值变化历史')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # 设置策略以填充可用空间
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def add_data_point(self, unit_index: int, value: float):
        """添加数据点"""
        if unit_index not in self.history_data:
            self.history_data[unit_index] = []
            
        self.history_data[unit_index].append(value)
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
            
        # 查找当前选中的单元（默认为第一个）
        selected_unit = 1
        if self.history_data:
            selected_unit = next(iter(self.history_data))
            
        # 获取选定单元的数据
        data = self.history_data.get(selected_unit, [])
        
        # 绘制图表标题
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        title_text = f'单元 #{selected_unit} 数值变化历史'
        title_rect = QRect(content_rect.left(), content_rect.top() - 30, 
                          content_rect.width(), 20)
        painter.drawText(title_rect, Qt.AlignCenter, title_text)
        
        # 绘制坐标轴标签
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 8))
        
        # Y轴标签
        painter.drawText(content_rect.left() - 15, content_rect.top() + 10, "1.0")
        painter.drawText(content_rect.left() - 15, content_rect.center().y(), "0.0")
        painter.drawText(content_rect.left() - 15, content_rect.bottom() - 10, "-1.0")
        
        # X轴标签
        painter.drawText(content_rect.left(), content_rect.bottom() + 5, "时间")
        
        # 绘制网格
        painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.DashLine))
        # 水平网格线
        for i in range(5):
            y = content_rect.top() + i * content_rect.height() // 4
            painter.drawLine(content_rect.left(), y, content_rect.right(), y)
            
        # 垂直网格线（如果有数据）
        if data:
            max_points = min(len(data), 10)  # 最多显示10个点
            for i in range(max_points + 1):
                x = content_rect.left() + i * content_rect.width() // max_points
                painter.drawLine(x, content_rect.top(), x, content_rect.bottom())
        
        # 绘制数据线
        if data:
            painter.setPen(QPen(QColor(0, 200, 0), 2))  # 绿色线条
            
            # 如果数据点超过10个，只显示最近的10个
            display_data = data[-10:] if len(data) > 10 else data
            max_points = len(display_data)
            
            if max_points > 1:
                # 计算点坐标
                points = []
                for i, value in enumerate(display_data):
                    # X坐标
                    x = content_rect.left() + i * content_rect.width() // (max_points - 1) \
                        if max_points > 1 else content_rect.center().x()
                    
                    # Y坐标 (-1.0 to 1.0) 映射到 (bottom to top)
                    y = content_rect.bottom() - (value + 1.0) * content_rect.height() / 2.0
                    
                    points.append((x, y))
                
                # 绘制连线
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    painter.drawLine(x1, y1, x2, y2)
                    
                # 绘制数据点
                painter.setBrush(QColor(0, 200, 0))
                for x, y in points:
                    painter.drawEllipse(x - 3, y - 3, 6, 6)
        
        # 绘制坐标轴
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawLine(content_rect.left(), content_rect.top(), 
                        content_rect.left(), content_rect.bottom())  # Y轴
        painter.drawLine(content_rect.left(), content_rect.bottom(), 
                        content_rect.right(), content_rect.bottom())  # X轴
                        
    def sizeHint(self):
        """返回推荐大小"""
        return QSize(500, 400)