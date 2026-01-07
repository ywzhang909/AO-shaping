from typing import List
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QScrollArea, QHBoxLayout
from PySide6.QtCore import Qt, Signal

from .circular_unit_button import MIN_VALUE, MAX_VALUE


class ValueInputPanel(QWidget):
    """数值输入面板类"""
    
    # 定义信号，当数值改变时发出
    valueChanged = Signal(int, float)  # 单元索引, 新值
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.input_fields: List[QLineEdit] = []
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标题
        title_label = QLabel('单元数值控制面板')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # 创建输入框网格
        grid_widget = QWidget()
        grid_layout = QVBoxLayout(grid_widget)
        
        # 每行10个输入框，共7行
        for row in range(7):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            # 计算该行的起始和结束索引
            start_index = row * 10
            end_index = min(start_index + 10, 64)
            
            for index in range(start_index, end_index):
                # 创建标签和输入框
                label = QLabel(f"{index+1:02d}:")
                line_edit = QLineEdit("0.00")
                line_edit.setFixedWidth(50)
                line_edit.setObjectName(f"value_input_{index}")
                
                # 使用闭包捕获当前索引
                def make_callback(idx):
                    return lambda: self.on_value_changed(idx, self.input_fields[idx].text())
                
                # 连接信号槽
                line_edit.returnPressed.connect(make_callback(index))
                
                # 保存引用
                self.input_fields.append(line_edit)
                
                # 添加到行布局
                row_layout.addWidget(label)
                row_layout.addWidget(line_edit)
                
            # 添加弹性空间
            row_layout.addStretch()
            grid_layout.addWidget(row_widget)
            
        # 创建滚动区域包装网格
        scroll_area = QScrollArea()
        scroll_area.setWidget(grid_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        layout.addWidget(scroll_area)
        
    def on_value_changed(self, unit_index: int, text: str):
        """处理输入框数值改变事件"""
        try:
            value = float(text)
            # 限制数值范围
            value = max(MIN_VALUE, min(MAX_VALUE, value))
            
            # 更新输入框显示
            if unit_index < len(self.input_fields):
                self.input_fields[unit_index].setText(f"{value:.2f}")
                
            # 发出信号
            self.valueChanged.emit(unit_index + 1, value)  # unit_index + 1 因为单元索引从1开始
        except ValueError:
            # 如果输入无效，恢复为当前值
            if unit_index < len(self.input_fields):
                current_value = 0.0  # 默认值
                self.input_fields[unit_index].setText(f"{current_value:.2f}")
                
    def update_values(self, values: List[float]):
        """更新所有输入框的值"""
        for i, value in enumerate(values):
            if i < len(self.input_fields):
                self.input_fields[i].setText(f"{value:.2f}")