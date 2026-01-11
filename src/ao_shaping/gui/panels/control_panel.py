"""
AO系统控制面板

提供DM电压手动控制和仿真操作按钮
"""

import numpy as np
from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSlider, QLabel, QPushButton, QScrollArea, QFrame,
    QDoubleSpinBox
)
from PySide6.QtCore import Signal, Qt


class ControlPanel(QWidget):
    """
    控制面板

    提供DM电压控制和仿真操作
    """

    # 信号
    reset_requested = Signal()
    step_requested = Signal()
    voltages_changed = Signal(list)  # voltages: List[float]

    def __init__(self):
        super().__init__()
        self.num_actuators = 8  # 默认值
        self.voltage_sliders: List[QSlider] = []
        self.voltage_spins: List[QDoubleSpinBox] = []
        self.voltage_labels: List[QLabel] = []

        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)

        # DM控制组
        dm_group = self.create_dm_control_group()
        layout.addWidget(dm_group)

        # 操作按钮组
        operation_group = self.create_operation_group()
        layout.addWidget(operation_group)

        layout.addStretch()

    def create_dm_control_group(self) -> QGroupBox:
        """创建DM控制组"""
        group = QGroupBox("DM电压控制")
        layout = QVBoxLayout(group)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.dm_layout = QFormLayout(scroll_widget)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)

        layout.addWidget(scroll_area)

        # 批量控制按钮
        batch_layout = QHBoxLayout()

        zero_all_btn = QPushButton("全部置零")
        zero_all_btn.clicked.connect(self.zero_all_voltages)
        batch_layout.addWidget(zero_all_btn)

        randomize_btn = QPushButton("随机化")
        randomize_btn.clicked.connect(self.randomize_voltages)
        batch_layout.addWidget(randomize_btn)

        layout.addLayout(batch_layout)

        return group

    def create_operation_group(self) -> QGroupBox:
        """创建操作按钮组"""
        group = QGroupBox("仿真操作")
        layout = QVBoxLayout(group)

        # 重置按钮
        reset_btn = QPushButton("重置系统")
        reset_btn.clicked.connect(self.reset_requested.emit)
        layout.addWidget(reset_btn)

        # 单步执行按钮
        step_btn = QPushButton("单步执行")
        step_btn.clicked.connect(self.step_requested.emit)
        layout.addWidget(step_btn)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 自动运行控制（预留）
        auto_label = QLabel("自动运行功能待实现")
        auto_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(auto_label)

        layout.addStretch()

        return group

    def set_num_actuators(self, num_actuators: int):
        """设置致动器数量"""
        if num_actuators == self.num_actuators:
            return

        self.num_actuators = num_actuators
        self.update_dm_controls()

    def update_dm_controls(self):
        """更新DM控制控件"""
        # 清除现有控件
        for slider in self.voltage_sliders:
            slider.deleteLater()
        for spin in self.voltage_spins:
            spin.deleteLater()
        for label in self.voltage_labels:
            label.deleteLater()

        self.voltage_sliders.clear()
        self.voltage_spins.clear()
        self.voltage_labels.clear()

        # 重新创建控件
        for i in range(self.num_actuators):
            self.create_actuator_control(i)

        # 发出初始电压变化信号（全部为0）
        self.emit_voltages_changed()

    def create_actuator_control(self, index: int):
        """创建单个致动器控制"""
        # 创建水平布局
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)

        # 标签
        label = QLabel(f"Act {index+1}:")
        label.setMinimumWidth(60)
        control_layout.addWidget(label)
        self.voltage_labels.append(label)

        # 滑块
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-100, 100)  # -1.0 到 1.0，对应电压范围
        slider.setValue(0)
        slider.setMinimumWidth(100)
        slider.valueChanged.connect(lambda v, idx=index: self.on_slider_changed(idx, v))
        control_layout.addWidget(slider)
        self.voltage_sliders.append(slider)

        # 数值输入框
        spin = QDoubleSpinBox()
        spin.setRange(-1.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(3)
        spin.setValue(0.0)
        spin.setMaximumWidth(80)
        spin.valueChanged.connect(lambda v, idx=index: self.on_spin_changed(idx, v))
        control_layout.addWidget(spin)
        self.voltage_spins.append(spin)

        # 添加到布局
        self.dm_layout.addRow(control_widget)

    def on_slider_changed(self, index: int, value: int):
        """滑块值变化处理"""
        voltage = value / 100.0  # 转换为 -1.0 到 1.0
        self.voltage_spins[index].blockSignals(True)
        self.voltage_spins[index].setValue(voltage)
        self.voltage_spins[index].blockSignals(False)
        self.emit_voltages_changed()

    def on_spin_changed(self, index: int, voltage: float):
        """数值输入框变化处理"""
        slider_value = int(voltage * 100)  # 转换为 -100 到 100
        self.voltage_sliders[index].blockSignals(True)
        self.voltage_sliders[index].setValue(slider_value)
        self.voltage_sliders[index].blockSignals(False)
        self.emit_voltages_changed()

    def emit_voltages_changed(self):
        """发出电压变化信号"""
        voltages = [spin.value() for spin in self.voltage_spins]
        self.voltages_changed.emit(voltages)

    def zero_all_voltages(self):
        """将所有电压置零"""
        for slider in self.voltage_sliders:
            slider.setValue(0)
        # emit_voltages_changed 会在滑块变化时自动调用

    def randomize_voltages(self):
        """随机化所有电压"""
        for slider in self.voltage_sliders:
            random_value = np.random.randint(-50, 51)  # -0.5 到 0.5
            slider.setValue(random_value)
        # emit_voltages_changed 会在滑块变化时自动调用

    def get_voltages(self) -> List[float]:
        """获取当前电压值"""
        return [spin.value() for spin in self.voltage_spins]

    def set_voltages(self, voltages: List[float]):
        """设置电压值"""
        if len(voltages) != len(self.voltage_spins):
            return

        for i, voltage in enumerate(voltages):
            self.voltage_spins[i].blockSignals(True)
            self.voltage_sliders[i].blockSignals(True)

            self.voltage_spins[i].setValue(voltage)
            self.voltage_sliders[i].setValue(int(voltage * 100))

            self.voltage_spins[i].blockSignals(False)
            self.voltage_sliders[i].blockSignals(False)</content>
</xai:function_call">控制面板已创建。现在创建结果显示面板，它需要集成图像显示和波前显示，并显示数值指标。让我创建results_panel.py文件。