import numpy as np
from typing import List, Dict, Any
import subprocess
import time
from PySide6.QtCore import QObject, Signal, QThread


class SimulationWorker(QObject):
    """模拟工作进程类"""
    
    finished = Signal()
    progress = Signal(list)
    error = Signal(str)
    
    def __init__(self, algorithm: str, parameters: Dict[str, Any]):
        super().__init__()
        self.algorithm = algorithm
        self.parameters = parameters
        self._is_running = False
        self.process = None
        
    def start(self):
        """开始运行模拟"""
        self._is_running = True
        try:
            self._run_simulation()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
            
    def stop(self):
        """停止运行"""
        self._is_running = False
        if self.process and self.process.poll() is None:
            # 如果进程仍在运行，则终止它
            try:
                self.process.terminate()
                self.process.wait(timeout=5)  # 等待最多5秒
            except subprocess.TimeoutExpired:
                # 如果进程没有响应，强制杀死
                self.process.kill()
                self.process.wait()
            
    def _run_simulation(self):
        """运行模拟算法"""
        # 模拟算法执行过程
        iteration = 0
        max_iterations = self.parameters.get("epochs", 100)
        
        # 初始化电压值
        voltages = np.zeros(64)
        
        while self._is_running and iteration < max_iterations:
            # 模拟算法执行
            if self.algorithm == "wf":
                # 波前优化算法模拟
                voltages = self._simulate_wf_algorithm(voltages)
            elif self.algorithm == "pib":
                # 轴向光束优化算法模拟
                voltages = self._simulate_pib_algorithm(voltages)
            elif self.algorithm == "combine":
                # 组合优化算法模拟
                voltages = self._simulate_combine_algorithm(voltages)
            elif self.algorithm == "bayes-opt":
                # 贝叶斯优化算法模拟
                voltages = self._simulate_bayes_opt_algorithm(voltages)
            elif self.algorithm == "heuristic":
                # 启发式搜索算法模拟
                voltages = self._simulate_heuristic_algorithm(voltages)
                
            # 发送进度更新
            self.progress.emit(voltages.tolist())
            
            iteration += 1
            time.sleep(0.1)  # 控制模拟速度
            
    def _simulate_wf_algorithm(self, voltages):
        """模拟波前优化算法"""
        # 简单的模拟实现，实际应用中需要复杂的波前计算
        # 这里只是随机微调电压值
        noise = np.random.normal(0, 0.01, 64)
        return np.clip(voltages + noise, -1.0, 1.0)
        
    def _simulate_pib_algorithm(self, voltages):
        """模拟轴向光束优化算法"""
        # 简单的模拟实现
        noise = np.random.normal(0, 0.01, 64)
        return np.clip(voltages + noise, -1.0, 1.0)
        
    def _simulate_combine_algorithm(self, voltages):
        """模拟组合优化算法"""
        # 简单的模拟实现
        noise = np.random.normal(0, 0.01, 64)
        return np.clip(voltages + noise, -1.0, 1.0)
        
    def _simulate_bayes_opt_algorithm(self, voltages):
        """模拟贝叶斯优化算法"""
        # 简单的模拟实现
        noise = np.random.normal(0, 0.01, 64)
        return np.clip(voltages + noise, -1.0, 1.0)
        
    def _simulate_heuristic_algorithm(self, voltages):
        """模拟启发式搜索算法"""
        # 简单的模拟实现
        noise = np.random.normal(0, 0.01, 64)
        return np.clip(voltages + noise, -1.0, 1.0)


class SimulationManager(QObject):
    """模拟管理器类"""
    
    # 定义信号
    simulationUpdated = Signal(list)  # 发送电压列表
    simulationFinished = Signal()
    simulationError = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.voltages = np.zeros(64)  # 64个变形镜单元的电压值
        self.worker = None
        self.thread = None
        self.algorithm = "wf"  # 默认算法
        self.parameters = {}   # 算法参数
        self.history = []      # 历史记录
        self.iteration = 0     # 迭代次数
        
    def set_voltage(self, unit_index: int, value: float):
        """设置指定单元的电压值"""
        if 1 <= unit_index <= 64:
            self.voltages[unit_index - 1] = max(-1.0, min(1.0, value))
            
    def get_voltages(self) -> List[float]:
        """获取所有单元的电压值"""
        return self.voltages.tolist()
        
    def set_algorithm(self, algorithm: str):
        """设置要使用的算法"""
        self.algorithm = algorithm
        
    def set_parameters(self, parameters: Dict[str, Any]):
        """设置算法参数"""
        self.parameters = parameters.copy()
        
    def start_simulation(self):
        """启动模拟"""
        # 停止任何正在进行的模拟
        self.stop_simulation()
        
        # 创建工作进程
        self.worker = SimulationWorker(self.algorithm, self.parameters)
        self.thread = QThread()
        
        # 移动工作进程到后台线程
        self.worker.moveToThread(self.thread)
        
        # 连接信号
        self.thread.started.connect(self.worker.start)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.progress.connect(self.on_simulation_updated)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # 启动线程
        self.thread.start()
            
    def stop_simulation(self):
        """停止模拟"""
        if self.worker:
            self.worker.stop()
            
    def reset(self):
        """重置模拟"""
        self.stop_simulation()
        self.voltages = np.zeros(64)
        self.history = []
        self.iteration = 0
        
    def on_simulation_updated(self, voltages: list):
        """处理模拟更新"""
        self.voltages = np.array(voltages)
        self.simulationUpdated.emit(voltages)
        
    def on_simulation_finished(self):
        """处理模拟完成"""
        self.simulationFinished.emit()
        
    def on_simulation_error(self, error: str):
        """处理模拟错误"""
        self.simulationError.emit(error)
        
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.history.copy()
        
    def cleanup(self):
        """清理资源"""
        self.stop_simulation()