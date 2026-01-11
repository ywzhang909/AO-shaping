"""
仿真工作线程

处理AO系统仿真计算，避免阻塞GUI
"""

import numpy as np
from typing import Optional, Dict, Any
from PySide6.QtCore import QThread, Signal

from ...sim.devices import TraditionalAOSystem, AOConfig


class SimulationWorker(QThread):
    """
    仿真工作线程

    在后台执行AO系统仿真计算
    """

    # 信号定义
    simulation_finished = Signal(dict)  # 仿真结果
    error_occurred = Signal(str)        # 错误信息
    progress_updated = Signal(str)      # 进度信息

    def __init__(self):
        super().__init__()
        self.ao_system: Optional[TraditionalAOSystem] = None
        self.config: Optional[AOConfig] = None
        self.operation = None
        self.parameters = {}

    def set_system(self, ao_system: TraditionalAOSystem):
        """设置AO系统实例"""
        self.ao_system = ao_system

    def set_config(self, config: AOConfig):
        """设置配置"""
        self.config = config

    def initialize_system(self):
        """初始化系统操作"""
        self.operation = 'initialize'

    def reset_system(self):
        """重置系统操作"""
        self.operation = 'reset'

    def set_voltages(self, voltages: np.ndarray):
        """设置电压操作"""
        self.operation = 'set_voltages'
        self.parameters['voltages'] = voltages

    def step_simulation(self, action: Optional[np.ndarray] = None):
        """单步仿真操作"""
        self.operation = 'step'
        if action is not None:
            self.parameters['action'] = action

    def run(self):
        """执行线程任务"""
        try:
            if self.ao_system is None:
                raise ValueError("AO系统未初始化")

            if self.operation == 'initialize':
                self._initialize_system()
            elif self.operation == 'reset':
                self._reset_system()
            elif self.operation == 'set_voltages':
                self._set_voltages()
            elif self.operation == 'step':
                self._step_simulation()
            else:
                raise ValueError(f"未知操作: {self.operation}")

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _initialize_system(self):
        """初始化AO系统"""
        self.progress_updated.emit("正在初始化AO系统...")

        if self.config is None:
            raise ValueError("配置未设置")

        # 创建新的AO系统实例
        self.ao_system = TraditionalAOSystem(self.config)

        self.progress_updated.emit("AO系统初始化完成")

        # 获取初始结果
        result = self.ao_system.reset()
        self.simulation_finished.emit(result)

    def _reset_system(self):
        """重置系统"""
        self.progress_updated.emit("正在重置系统...")

        result = self.ao_system.reset()

        self.progress_updated.emit("系统重置完成")
        self.simulation_finished.emit(result)

    def _set_voltages(self):
        """设置DM电压"""
        voltages = self.parameters.get('voltages')
        if voltages is None:
            raise ValueError("未提供电压参数")

        self.progress_updated.emit("正在设置DM电压...")

        self.ao_system.set_dm_voltages(voltages)

        # 获取更新后的结果
        image = self.ao_system.get_image()
        slopes = self.ao_system.measure_wavefront()

        # 计算指标
        strehl = self._calculate_strehl()

        result = {
            'image': image,
            'slopes': slopes,
            'strehl': strehl,
            'power': np.sum(image),
            'voltages': self.ao_system.dm_voltages.copy()
        }

        self.progress_updated.emit("DM电压设置完成")
        self.simulation_finished.emit(result)

    def _step_simulation(self):
        """执行单步仿真"""
        action = self.parameters.get('action', np.zeros(self.ao_system.dm.total_actuators))

        self.progress_updated.emit("正在执行仿真步骤...")

        result = self.ao_system.step(action)

        self.progress_updated.emit("仿真步骤完成")
        self.simulation_finished.emit(result)

    def _calculate_strehl(self) -> float:
        """计算Strehl比"""
        if self.ao_system is None:
            return 0.0

        # 使用step方法计算Strehl比（零动作）
        result = self.ao_system.step(np.zeros(self.ao_system.dm.total_actuators))
        return result['strehl']

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        if self.ao_system is None:
            return {}

        return {
            'config': self.config.__dict__ if self.config else {},
            'dm_actuators': self.ao_system.dm.total_actuators,
            'wfs_subapertures': self.ao_system.wfs.total_subapertures,
            'pupil_mask_shape': self.ao_system.pupil_mask.shape
        }</content>
</xai:function_call">工作线程类已创建。现在需要更新主窗口以集成所有组件，并处理配置变化时更新控制面板的致动器数量。让我更新main_window.py文件。