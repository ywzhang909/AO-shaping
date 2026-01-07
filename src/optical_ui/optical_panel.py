from pathlib import Path
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QResizeEvent

from optical_ui.circular_unit_button import CircularUnitButton
DELTA = 0.1

class OpticalPanel(QWidget):
    """光学面板类，管理所有单元按钮"""
    
    # 定义信号，当数值改变时发出
    valueChanged = Signal(int, float)  # 单元索引, 新值
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buttons: list[CircularUnitButton] = []
        self.values: list[float] = [0.0] * 64  # 每个单元的数值列表
        self.unit_positions = []  # 存储单元位置信息
        self.initUI()
        self.load_unit_positions()
        self.create_buttons()
        
    def initUI(self):
        """初始化界面"""
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        
    def load_unit_positions(self):
        """从文件加载单元位置信息"""
        try:
            # 构建文件路径
            file_path = Path(__file__).parent.parent.parent / "res" / "layouts" / "XuWeiDM64_UnitsPos.txt"
            
            # 清空旧的位置信息
            self.unit_positions.clear()
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            unit_index = int(parts[0])
                            x_coord = float(parts[1])
                            y_coord = float(parts[2])
                            self.unit_positions.append((unit_index, x_coord, y_coord))
                            
            print(f"成功加载 {len(self.unit_positions)} 个单元位置信息")
        except Exception as e:
            print(f"加载单元位置信息失败: {e}")
            # 如果加载失败，使用默认位置
            self.unit_positions = [(i, 50.0 + (i % 8) * 10, 50.0 + (i // 8) * 10) for i in range(1, 65)]
            
    def create_buttons(self):
        """创建所有单元按钮"""
        # 清除现有的按钮
        for button in self.buttons:
            button.deleteLater()
        self.buttons.clear()
        
        # 创建新的按钮
        for unit_index, x, y in self.unit_positions:
            button = CircularUnitButton(unit_index, x, y)
            button.setValue(self.values[unit_index - 1])  # 设置初始值
            button.valueChanged.connect(self.on_button_value_changed)
            self.buttons.append(button)
        
        # 布局所有按钮
        self.layout_buttons()
            
    def layout_buttons(self):
        """根据面板大小布局按钮"""
        if not self.unit_positions or not self.buttons:
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
            
    def resizeEvent(self, event: QResizeEvent):
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