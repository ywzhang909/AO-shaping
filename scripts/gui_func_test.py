import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QLineEdit, QSlider, QGroupBox, QTabWidget,
                               QGridLayout)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SGDAlgorithm:
    """SGD算法实现类"""
    def __init__(self, target_coeffs, degree=3, learning_rate=0.01):
        self.target_coeffs = np.array(target_coeffs)
        self.degree = degree
        self.learning_rate = learning_rate
        self.current_coeffs = np.random.randn(degree+1) * 0.1  # 初始化系数
        self.x_data = np.linspace(-10, 10, 100)
        self.y_target = self.evaluate_polynomial(self.x_data, self.target_coeffs)
        self.iteration_count = 0
    
    def evaluate_polynomial(self, x, coeffs):
        """计算多项式的值"""
        result = np.zeros_like(x)
        for i, coeff in enumerate(coeffs):
            result += coeff * (x ** i)
        return result
    
    def compute_gradient(self, x, y_true, y_pred):
        """计算梯度"""
        n = len(x)
        gradients = np.zeros(len(self.current_coeffs))
        
        for i in range(len(self.current_coeffs)):
            # 计算损失函数对第i个系数的偏导数
            grad = 2/n * np.sum((y_pred - y_true) * (x ** i))
            gradients[i] = grad
            
        return gradients
    
    def step(self):
        """执行一次SGD更新"""
        y_pred = self.evaluate_polynomial(self.x_data, self.current_coeffs)
        gradients = self.compute_gradient(self.x_data, self.y_target, y_pred)
        self.current_coeffs -= self.learning_rate * gradients
        
        # 计算RMS误差
        rms_error = np.sqrt(np.mean((y_pred - self.y_target) ** 2))
        self.iteration_count += 1
        
        return self.current_coeffs.copy(), rms_error, self.iteration_count

class CoefficientsBarChart(FigureCanvas):
    """系数条形图组件"""
    def __init__(self):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)
        
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
    
    def update_plot(self, current_coeffs, target_coeffs):
        """更新条形图"""
        self.ax.clear()
        
        x_pos = np.arange(len(current_coeffs))
        bar_width = 0.35
        
        # 绘制当前系数和目标系数的对比
        bars1 = self.ax.bar(x_pos - bar_width/2, current_coeffs, bar_width, 
                           label='当前系数', alpha=0.7, color='blue')
        bars2 = self.ax.bar(x_pos + bar_width/2, target_coeffs, bar_width, 
                           label='目标系数', alpha=0.7, color='orange')
        
        self.ax.set_xlabel('系数索引')
        self.ax.set_ylabel('系数值')
        self.ax.set_title('多项式系数比较')
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        self.fig.tight_layout()
        self.draw()

class RMSErrorPlot(FigureCanvas):
    """RMS误差折线图组件"""
    def __init__(self):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)
        
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
    
    def update_plot(self, rms_history):
        """更新RMS误差折线图"""
        self.ax.clear()
        
        if len(rms_history) > 0:
            self.ax.plot(range(len(rms_history)), rms_history, 
                        label='RMS误差', color='red', linewidth=2)
            self.ax.set_xlabel('迭代次数')
            self.ax.set_ylabel('RMS误差')
            self.ax.set_title('RMS误差收敛曲线')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.6)
        else:
            self.ax.text(0.5, 0.5, '等待数据...', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=14)
            self.ax.set_title('RMS误差收敛曲线')
        
        self.fig.tight_layout()
        self.draw()

class ControlPanel(QWidget):
    """控制面板组件"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 目标函数系数设置组
        target_group = QGroupBox("目标函数系数")
        target_layout = QGridLayout(target_group)
        
        self.target_coeff_inputs = []
        for i in range(6):  # 支持最多6次多项式
            row = i // 2
            col = i % 2
            h_layout = QHBoxLayout()
            label = QLabel(f"Coeff {i} (x^{i}):")
            line_edit = QLineEdit("0.0")
            h_layout.addWidget(label)
            h_layout.addWidget(line_edit)
            target_layout.addLayout(h_layout, row, col)
            self.target_coeff_inputs.append(line_edit)
        
        layout.addWidget(target_group)
        
        # SGD参数设置组
        sgd_group = QGroupBox("SGD参数设置")
        sgd_layout = QGridLayout(sgd_group)
        
        # 学习率
        lr_label = QLabel("学习率:")
        self.lr_input = QLineEdit("0.0001")
        sgd_layout.addWidget(lr_label, 0, 0)
        sgd_layout.addWidget(self.lr_input, 0, 1)
        
        # 多项式阶数
        degree_label = QLabel("多项式阶数:")
        self.degree_slider = QSlider(Qt.Horizontal)
        self.degree_slider.setMinimum(1)
        self.degree_slider.setMaximum(5)
        self.degree_slider.setValue(3)
        self.degree_value_label = QLabel("3")
        sgd_layout.addWidget(degree_label, 1, 0)
        sgd_layout.addWidget(self.degree_slider, 1, 1)
        sgd_layout.addWidget(self.degree_value_label, 1, 2)
        
        self.degree_slider.valueChanged.connect(
            lambda v: self.degree_value_label.setText(str(v)))
        
        layout.addWidget(sgd_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始拟合")
        self.stop_button = QPushButton("停止拟合")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        
        # 进度信息
        progress_group = QGroupBox("进度信息")
        progress_layout = QVBoxLayout(progress_group)
        
        self.iteration_label = QLabel("迭代次数: 0")
        self.error_label = QLabel("当前RMS误差: 0.0")
        progress_layout.addWidget(self.iteration_label)
        progress_layout.addWidget(self.error_label)
        
        layout.addWidget(progress_group)
        
        layout.addStretch()
        self.setLayout(layout)

class VisualizationPanel(QWidget):
    """可视化面板组件"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 系数对比图
        coeff_group = QGroupBox("多项式系数对比")
        coeff_layout = QVBoxLayout(coeff_group)
        self.coeff_chart = CoefficientsBarChart()
        coeff_layout.addWidget(self.coeff_chart)
        
        # RMS误差图
        error_group = QGroupBox("RMS误差收敛曲线")
        error_layout = QVBoxLayout(error_group)
        self.error_plot = RMSErrorPlot()
        error_layout.addWidget(self.error_plot)
        
        layout.addWidget(coeff_group)
        layout.addWidget(error_group)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("SGD多项式拟合 - 优化版")
        self.setGeometry(100, 100, 1200, 800)
        
        # 中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 创建选项卡控件
        tab_widget = QTabWidget()
        
        # 控制面板
        self.control_panel = ControlPanel()
        tab_widget.addTab(self.control_panel, "控制面板")
        
        # 可视化面板
        self.viz_panel = VisualizationPanel()
        tab_widget.addTab(self.viz_panel, "可视化")
        
        # 将选项卡添加到主布局
        main_layout.addWidget(tab_widget)
        
        # 初始化SGD拟合器
        self.sgd_fitter = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_fitting)
        
        # 数据记录
        self.rms_history = []
        
        # 连接信号
        self.control_panel.start_button.clicked.connect(self.start_fitting)
        self.control_panel.stop_button.clicked.connect(self.stop_fitting)
        
        # 设置初始值
        self.setup_initial_values()
    
    def setup_initial_values(self):
        """设置初始值"""
        self.control_panel.target_coeff_inputs[0].setText("1.0")   # 常数项
        self.control_panel.target_coeff_inputs[1].setText("0.5")   # x项
        self.control_panel.target_coeff_inputs[2].setText("-0.2")  # x^2项
        self.control_panel.target_coeff_inputs[3].setText("0.1")   # x^3项
    
    def start_fitting(self):
        """开始拟合"""
        try:
            # 获取目标函数系数
            target_coeffs = []
            degree = self.control_panel.degree_slider.value()
            
            for i in range(degree + 1):
                value = float(self.control_panel.target_coeff_inputs[i].text())
                target_coeffs.append(value)
            
            # 获取学习率
            learning_rate = float(self.control_panel.lr_input.text())
            
            # 创建SGD拟合器
            self.sgd_fitter = SGDAlgorithm(target_coeffs, degree, learning_rate)
            
            # 重置数据
            self.rms_history = []
            
            # 启动定时器
            self.timer.start(50)  # 每50毫秒更新一次
            self.control_panel.start_button.setEnabled(False)
            self.control_panel.stop_button.setEnabled(True)
            
        except ValueError:
            print("输入参数错误，请检查数值格式")
    
    def stop_fitting(self):
        """停止拟合"""
        self.timer.stop()
        self.control_panel.start_button.setEnabled(True)
        self.control_panel.stop_button.setEnabled(False)
    
    def update_fitting(self):
        """更新拟合过程"""
        if self.sgd_fitter:
            current_coeffs, rms_error, iteration_count = self.sgd_fitter.step()
            
            # 更新历史记录
            self.rms_history.append(rms_error)
            
            # 更新UI
            self.control_panel.iteration_label.setText(f"迭代次数: {iteration_count}")
            self.control_panel.error_label.setText(f"当前RMS误差: {rms_error:.6f}")
            
            # 更新可视化
            self.viz_panel.coeff_chart.update_plot(current_coeffs, self.sgd_fitter.target_coeffs)
            self.viz_panel.error_plot.update_plot(self.rms_history)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())



