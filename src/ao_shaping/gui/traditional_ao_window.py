import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter, QStatusBar,
    QMessageBox, QProgressBar, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QLabel, QPushButton, QTextEdit
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtGui import QFont

from .panels.visualization_panel import VisualizationPanel
from ..sim.devices import TraditionalAOSystem, AOConfig


class TraditionalAOConfigPanel(QWidget):
    """传统AO系统配置面板"""

    # 信号定义
    configApplied = Signal(object)  # AOConfig对象

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)

        # 标题
        title_label = QLabel('传统AO系统配置')
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)

        # 网格参数组
        grid_group = QGroupBox("网格参数")
        grid_form = QFormLayout(grid_group)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(64, 1024)
        self.n_spin.setValue(256)
        self.n_spin.setSingleStep(64)
        grid_form.addRow("网格点数 N:", self.n_spin)

        self.l_spin = QDoubleSpinBox()
        self.l_spin.setRange(0.01, 1.0)
        self.l_spin.setValue(0.1)
        self.l_spin.setSingleStep(0.01)
        grid_form.addRow("物理孔径大小 L (m):", self.l_spin)

        layout.addWidget(grid_group)

        # 光源参数组
        light_source_group = QGroupBox("光源参数")
        light_source_form = QFormLayout(light_source_group)

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(1e-9, 10e-6)
        self.wavelength_spin.setValue(1550e-9)
        self.wavelength_spin.setSingleStep(100e-9)
        self.wavelength_spin.setDecimals(12)  # 显示更多小数位
        light_source_form.addRow("波长 (m):", self.wavelength_spin)

        layout.addWidget(light_source_group)

        # 大气参数组
        atmosphere_group = QGroupBox("大气参数")
        atmosphere_form = QFormLayout(atmosphere_group)

        self.cn2_spin = QDoubleSpinBox()
        self.cn2_spin.setRange(0, 1e-10)
        self.cn2_spin.setValue(1e-14)
        self.cn2_spin.setSingleStep(1e-15)
        self.cn2_spin.setDecimals(15)  # 显示更多小数位
        atmosphere_form.addRow("折射率结构常数 Cn²:", self.cn2_spin)

        self.l0_spin = QDoubleSpinBox()
        self.l0_spin.setRange(1.0, 100.0)
        self.l0_spin.setValue(10.0)
        self.l0_spin.setSingleStep(1.0)
        atmosphere_form.addRow("外尺度 L₀ (m):", self.l0_spin)

        self.l0_inner_spin = QDoubleSpinBox()
        self.l0_inner_spin.setRange(0.001, 1.0)
        self.l0_inner_spin.setValue(0.01)
        self.l0_inner_spin.setSingleStep(0.001)
        atmosphere_form.addRow("内尺度 l₀ (m):", self.l0_inner_spin)

        layout.addWidget(atmosphere_group)

        # DM参数组
        dm_group = QGroupBox("变形镜参数")
        dm_form = QFormLayout(dm_group)

        self.dm_actuators_spin = QSpinBox()
        self.dm_actuators_spin.setRange(4, 16)
        self.dm_actuators_spin.setValue(8)
        dm_form.addRow("DM致动器数量:", self.dm_actuators_spin)

        self.dm_stroke_spin = QDoubleSpinBox()
        self.dm_stroke_spin.setRange(1e-6, 50e-6)
        self.dm_stroke_spin.setValue(5e-6)
        self.dm_stroke_spin.setSingleStep(1e-6)
        self.dm_stroke_spin.setDecimals(9)  # 显示纳米级精度
        dm_form.addRow("DM行程 (m):", self.dm_stroke_spin)

        layout.addWidget(dm_group)

        # WFS参数组
        wfs_group = QGroupBox("波前传感器参数")
        wfs_form = QFormLayout(wfs_group)

        self.subapertures_spin = QSpinBox()
        self.subapertures_spin.setRange(4, 16)
        self.subapertures_spin.setValue(8)
        wfs_form.addRow("子孔径数量:", self.subapertures_spin)

        self.pixel_scale_spin = QDoubleSpinBox()
        self.pixel_scale_spin.setRange(0.1, 2.0)
        self.pixel_scale_spin.setValue(0.5)
        self.pixel_scale_spin.setSingleStep(0.1)
        wfs_form.addRow("像素比例:", self.pixel_scale_spin)

        layout.addWidget(wfs_group)

        # 传播参数组
        propagation_group = QGroupBox("传播参数")
        propagation_form = QFormLayout(propagation_group)

        self.propagation_distance_spin = QDoubleSpinBox()
        self.propagation_distance_spin.setRange(100.0, 10000.0)
        self.propagation_distance_spin.setValue(1000.0)
        self.propagation_distance_spin.setSingleStep(100.0)
        propagation_form.addRow("传播距离 (m):", self.propagation_distance_spin)

        layout.addWidget(propagation_group)

        # 操作按钮
        buttons_layout = QHBoxLayout()

        self.apply_button = QPushButton("应用配置")
        self.apply_button.clicked.connect(self.on_apply_clicked)
        buttons_layout.addWidget(self.apply_button)

        self.reset_button = QPushButton("重置为默认")
        self.reset_button.clicked.connect(self.on_reset_clicked)
        buttons_layout.addWidget(self.reset_button)

        layout.addLayout(buttons_layout)

        # 添加伸缩空间
        layout.addStretch()

    def on_apply_clicked(self):
        """应用配置"""
        # 发送配置更新信号
        config = self.get_config()
        self.configApplied.emit(config)

    def on_reset_clicked(self):
        """重置为默认配置"""
        self.set_default_config()

    def get_config(self) -> AOConfig:
        """获取当前配置"""
        return AOConfig(
            N=self.n_spin.value(),
            L=self.l_spin.value(),
            wavelength=self.wavelength_spin.value(),
            Cn2=self.cn2_spin.value(),
            L0=self.l0_spin.value(),
            l0=self.l0_inner_spin.value(),
            dm_actuators=self.dm_actuators_spin.value(),
            dm_stroke=self.dm_stroke_spin.value(),
            subapertures=self.subapertures_spin.value(),
            pixel_scale=self.pixel_scale_spin.value(),
            propagation_distance=self.propagation_distance_spin.value()
        )

    def set_config(self, config: AOConfig):
        """设置配置"""
        self.n_spin.setValue(config.N)
        self.l_spin.setValue(config.L)
        self.wavelength_spin.setValue(config.wavelength)
        self.cn2_spin.setValue(config.Cn2)
        self.l0_spin.setValue(config.L0)
        self.l0_inner_spin.setValue(config.l0)
        self.dm_actuators_spin.setValue(config.dm_actuators)
        self.dm_stroke_spin.setValue(config.dm_stroke)
        self.subapertures_spin.setValue(config.subapertures)
        self.pixel_scale_spin.setValue(config.pixel_scale)
        self.propagation_distance_spin.setValue(config.propagation_distance)

    def set_default_config(self):
        """设置默认配置"""
        default_config = AOConfig()
        self.set_config(default_config)


class TraditionalAOSimulationThread(QThread):
    """传统AO系统仿真线程"""

    simulationUpdated = Signal(dict)
    simulationFinished = Signal(dict)
    simulationError = Signal(str)

    def __init__(self, ao_system, num_steps=100):
        super().__init__()
        self.ao_system = ao_system
        self.num_steps = num_steps
        self.is_running = False

    def run(self):
        """运行仿真"""
        try:
            self.is_running = True

            # 重置系统
            initial_state = self.ao_system.reset()

            # 发送初始状态
            self.simulationUpdated.emit({
                'step': 0,
                'strehl': initial_state['strehl'],
                'image': initial_state['image'],
                'slopes': initial_state['slopes'],
                'voltages': initial_state['voltages']
            })

            # 运行仿真步骤
            for step in range(1, self.num_steps + 1):
                if not self.is_running:
                    break

                # 生成随机动作（用于演示）
                action = np.random.normal(0, 0.1, self.ao_system.dm.total_actuators)

                # 执行一步
                result = self.ao_system.step(action)

                # 发送更新
                self.simulationUpdated.emit({
                    'step': step,
                    'strehl': result['strehl'],
                    'image': result['image'],
                    'slopes': result['slopes'],
                    'voltages': result['voltages']
                })

                # 短暂延迟以观察结果
                QThread.msleep(100)

            if self.is_running:
                final_state = self.ao_system.reset()
                self.simulationFinished.emit(final_state)

        except Exception as e:
            self.simulationError.emit(str(e))

    def stop(self):
        """停止仿真"""
        self.is_running = False


class TraditionalAOWindow(QMainWindow):
    """传统AO系统主窗口"""

    def __init__(self):
        super().__init__()
        self.ao_system = None
        self.simulation_thread = None
        self.is_running = False

        # 创建默认配置
        self.config = AOConfig()
        self.create_ao_system()

        self.init_ui()
        self.connect_signals()
        self.update_button_states()

    def create_ao_system(self):
        """创建AO系统"""
        try:
            self.ao_system = TraditionalAOSystem(self.config)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建AO系统失败: {str(e)}")
            self.ao_system = None

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('传统AO系统可视化控制系统')
        self.setGeometry(100, 100, 1400, 900)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)

        # 创建左侧控制面板
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 配置面板
        self.config_panel = TraditionalAOConfigPanel()
        left_layout.addWidget(self.config_panel)

        # 控制按钮面板
        control_group = QGroupBox("仿真控制")
        control_layout = QVBoxLayout(control_group)

        self.start_button = QPushButton("开始仿真")
        self.start_button.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止仿真")
        self.stop_button.clicked.connect(self.stop_simulation)
        control_layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("重置系统")
        self.reset_button.clicked.connect(self.reset_system)
        control_layout.addWidget(self.reset_button)

        left_layout.addWidget(control_group)

        # 状态显示面板
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)

        left_layout.addWidget(status_group)

        splitter.addWidget(left_widget)

        # 创建右侧显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 创建Strehl比图表
        self.strehl_chart = QChart()
        self.strehl_series = QLineSeries()
        self.strehl_chart.addSeries(self.strehl_series)
        self.strehl_chart.setTitle("Strehl比历史")
        self.strehl_chart.createDefaultAxes()
        self.strehl_chart.legend().hide()

        self.strehl_x_axis = QValueAxis()
        self.strehl_x_axis.setTitleText("仿真步骤")
        self.strehl_chart.setAxisX(self.strehl_x_axis, self.strehl_series)

        self.strehl_y_axis = QValueAxis()
        self.strehl_y_axis.setTitleText("Strehl比")
        self.strehl_y_axis.setRange(0, 1)
        self.strehl_chart.setAxisY(self.strehl_y_axis, self.strehl_series)

        self.strehl_view = QChartView(self.strehl_chart)
        self.strehl_view.setMaximumHeight(300)
        right_layout.addWidget(self.strehl_view)

        # 创建可视化面板
        self.visualization_panel = VisualizationPanel()
        right_layout.addWidget(self.visualization_panel)

        splitter.addWidget(right_widget)
        splitter.setSizes([400, 1000])

        main_layout.addWidget(splitter)

        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪')

        # 初始化数据存储
        self.strehl_history = []
        self.step_count = 0

    def connect_signals(self):
        """连接信号"""
        # 配置面板信号
        self.config_panel.configApplied.connect(self.on_config_applied)

    def update_button_states(self):
        """更新按钮状态"""
        self.start_button.setEnabled(not self.is_running and self.ao_system is not None)
        self.stop_button.setEnabled(self.is_running)
        self.reset_button.setEnabled(not self.is_running)

    def on_config_applied(self, config: AOConfig):
        """处理配置应用"""
        self.config = config
        self.create_ao_system()
        self.update_button_states()
        self.status_bar.showMessage('配置已应用')

        # 更新状态显示
        self.update_status_display()

    def update_status_display(self):
        """更新状态显示"""
        if not self.ao_system:
            self.status_text.setPlainText("AO系统未初始化")
            return

        status_info = f"""传统AO系统状态:

网格参数:
  网格点数: {self.config.N}
  孔径大小: {self.config.L:.3f} m
  波长: {self.config.wavelength*1e9:.1f} nm

大气参数:
  Cn²: {self.config.Cn2:.2e}
  外尺度 L₀: {self.config.L0:.1f} m
  内尺度 l₀: {self.config.l0:.3f} m

变形镜参数:
  致动器数量: {self.config.dm_actuators}x{self.config.dm_actuators} = {self.config.dm_actuators**2}
  行程: {self.config.dm_stroke*1e6:.1f} μm

波前传感器参数:
  子孔径数量: {self.config.subapertures}x{self.config.subapertures} = {self.config.subapertures**2}
  像素比例: {self.config.pixel_scale}

传播参数:
  传播距离: {self.config.propagation_distance:.0f} m
"""
        self.status_text.setPlainText(status_info)

    def start_simulation(self):
        """开始仿真"""
        if self.is_running or not self.ao_system:
            return

        self.is_running = True
        self.update_button_states()

        # 清除历史数据
        self.strehl_history.clear()
        self.strehl_series.clear()
        self.step_count = 0

        # 创建仿真线程
        self.simulation_thread = TraditionalAOSimulationThread(self.ao_system, num_steps=100)

        # 连接信号
        self.simulation_thread.simulationUpdated.connect(self.on_simulation_updated)
        self.simulation_thread.simulationFinished.connect(self.on_simulation_finished)
        self.simulation_thread.simulationError.connect(self.on_simulation_error)

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定模式
        self.status_bar.showMessage('正在运行仿真...')

        # 启动仿真
        self.simulation_thread.start()

    def stop_simulation(self):
        """停止仿真"""
        if not self.is_running:
            return

        self.is_running = False
        self.update_button_states()

        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.wait()

        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('仿真已停止')

    def reset_system(self):
        """重置系统"""
        self.is_running = False
        self.update_button_states()

        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.wait()

        if self.ao_system:
            try:
                state = self.ao_system.reset()
                self.on_simulation_updated({
                    'step': 0,
                    'strehl': state['strehl'],
                    'image': state['image'],
                    'slopes': state['slopes'],
                    'voltages': state['voltages']
                })
            except Exception as e:
                QMessageBox.critical(self, "错误", f"重置系统失败: {str(e)}")

        # 清除历史数据
        self.strehl_history.clear()
        self.strehl_series.clear()
        self.step_count = 0

        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('系统已重置')

    def on_simulation_updated(self, data: dict):
        """处理仿真更新"""
        step = data.get('step', 0)
        strehl = data.get('strehl', 0.0)
        voltages = data.get('voltages', [])

        # 更新Strehl比历史
        self.strehl_history.append(strehl)
        self.strehl_series.append(step, strehl)

        # 更新坐标轴范围
        if self.strehl_history:
            self.strehl_x_axis.setRange(0, max(step, 10))
            min_strehl = min(self.strehl_history)
            max_strehl = max(self.strehl_history)
            margin = (max_strehl - min_strehl) * 0.1 if max_strehl != min_strehl else 0.1
            self.strehl_y_axis.setRange(max(0, min_strehl - margin), min(1, max_strehl + margin))

        # 更新可视化面板
        if voltages:
            self.visualization_panel.update_plots(np.array(voltages))

        # 更新状态栏
        self.status_bar.showMessage(f'仿真步骤: {step}, Strehl比: {strehl:.4f}')

    def on_simulation_finished(self, final_state: dict):
        """处理仿真完成"""
        self.is_running = False
        self.update_button_states()
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('仿真完成')

        QMessageBox.information(self, "完成", "传统AO系统仿真已完成")

    def on_simulation_error(self, error: str):
        """处理仿真错误"""
        self.is_running = False
        self.update_button_states()
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('仿真出错')

        QMessageBox.critical(self, "仿真错误", f"仿真过程中发生错误: {error}")

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        self.stop_simulation()
        event.accept()