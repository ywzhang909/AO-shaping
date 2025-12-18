from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QScrollArea, QSplitter, QTabWidget
from PySide6.QtCore import Qt

from .optical_panel import OpticalPanel
from .value_input_panel import ValueInputPanel
from .history_chart_panel import HistoryChartPanel
from .image_panel import ImagePanel


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('光学单元控制面板')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标题
        title_label = QLabel('光学单元控制系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 创建垂直分割器用于主面板和图表
        vertical_splitter = QSplitter(Qt.Vertical)
        
        # 创建水平分割器用于放置光学面板和数值输入面板
        horizontal_splitter = QSplitter(Qt.Horizontal)
        
        # 创建光学面板
        self.optical_panel = OpticalPanel()
        self.optical_panel.valueChanged.connect(self.on_optical_value_changed)
        
        # 创建滚动区域包装光学面板
        optical_scroll = QScrollArea()
        optical_scroll.setWidget(self.optical_panel)
        optical_scroll.setWidgetResizable(True)
        
        # 创建数值输入面板
        self.value_input_panel = ValueInputPanel()
        self.value_input_panel.valueChanged.connect(self.on_input_value_changed)
        
        # 创建滚动区域包装数值输入面板
        input_scroll = QScrollArea()
        input_scroll.setWidget(self.value_input_panel)
        input_scroll.setWidgetResizable(True)
        
        # 添加到水平分割器
        horizontal_splitter.addWidget(optical_scroll)
        horizontal_splitter.addWidget(input_scroll)
        
        # 设置初始大小比例
        horizontal_splitter.setSizes([700, 300])
        
        # 创建图表面板
        self.chart_panel = HistoryChartPanel()
        
        # 创建图像面板
        self.image_panel = ImagePanel()
        
        # 创建选项卡控件来包含图表和图像面板
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.chart_panel, "数值变化历史")
        self.tab_widget.addTab(self.image_panel, "数值矩阵可视化")
        
        # 创建滚动区域包装选项卡控件
        tab_scroll = QScrollArea()
        tab_scroll.setWidget(self.tab_widget)
        tab_scroll.setWidgetResizable(True)
        
        # 添加到垂直分割器
        vertical_splitter.addWidget(horizontal_splitter)
        vertical_splitter.addWidget(tab_scroll)
        
        # 设置初始大小比例
        vertical_splitter.setSizes([600, 200])
        
        main_layout.addWidget(vertical_splitter)
        
        # 创建状态栏
        self.statusBar().showMessage('就绪')
        
        # 初始化数值显示
        self.update_input_values()
        
    def on_optical_value_changed(self, unit_index: int, value: float):
        """处理光学面板数值改变事件"""
        # 更新输入面板中的对应值
        if 1 <= unit_index <= 64:
            index = unit_index - 1
            if index < len(self.value_input_panel.input_fields):
                self.value_input_panel.input_fields[index].setText(f"{value:.2f}")
                
            # 更新图表（只跟踪第一个单元）
            if unit_index == 1:
                self.chart_panel.add_data_point(unit_index, value)
                
            # 更新图像面板
            values = self.optical_panel.get_values()
            self.image_panel.update_image(values)
                
    def on_input_value_changed(self, unit_index: int, value: float):
        """处理输入面板数值改变事件"""
        # 更新光学面板中的对应按钮值
        if 1 <= unit_index <= 64:
            index = unit_index - 1
            if index < len(self.optical_panel.buttons):
                self.optical_panel.buttons[index].setValue(value)
                
            # 更新图表（只跟踪第一个单元）
            if unit_index == 1:
                self.chart_panel.add_data_point(unit_index, value)
                
            # 更新图像面板
            values = self.optical_panel.get_values()
            self.image_panel.update_image(values)
                
    def update_input_values(self):
        """更新输入面板的数值显示"""
        values = self.optical_panel.get_values()
        self.value_input_panel.update_values(values)
        
        # 初始化第一个单元的历史数据
        if values:
            self.chart_panel.add_data_point(1, values[0])
            
        # 更新图像面板
        self.image_panel.update_image(values)
        
    def resizeEvent(self, event):
        """处理窗口大小调整事件"""
        super().resizeEvent(event)
        # 重新布局光学面板中的按钮
        self.optical_panel.layout_buttons()