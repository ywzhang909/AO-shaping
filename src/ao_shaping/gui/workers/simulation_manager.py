import numpy as np
from typing import List, Dict, Any
import threading
import time
from PySide6.QtCore import QObject, Signal

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.utils.file import get_init_V_by_rms


class SimulationManager(QObject):
    """PIB优化管理器类"""
    
    # 定义信号
    simulationUpdated = Signal(list)  # 发送电压列表
    simulationFinished = Signal(object)  # 传递Recorder对象
    simulationError = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.voltages = np.zeros(64)  # 64个变形镜单元的电压值
        self.is_running = False
        self.simulation_thread = None
        self.algorithm = "wf"  # 默认算法
        self.parameters = {}   # 算法参数
        self.history = []      # 历史记录
        self.iteration = 0     # 迭代次数
        self.recorder = None   # 优化记录器
        
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
        """启动PIB优化"""
        if not self.is_running:
            self.is_running = True
            self.iteration = 0
            self.simulation_thread = threading.Thread(target=self._run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
    def stop_simulation(self):
        """停止优化"""
        self.is_running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2)  # 等待最多2秒
            
    def reset(self):
        """重置优化"""
        self.stop_simulation()
        self.voltages = np.zeros(64)
        self.history = []
        self.iteration = 0
        self.recorder = None
        
    def _run_simulation(self):
        """运行PIB优化的后台线程函数"""
        try:
            self._run_simulation_loop()
                
        except Exception as e:
            self.simulationError.emit(str(e))
        finally:
            self.is_running = False
            self.simulationFinished.emit(self.recorder)
        
    def _run_simulation_loop(self):
        """运行模拟循环（用于非PIB算法）"""
        max_iterations = int(self.parameters.get("epochs", 1000))  # 获取设置的迭代次数，默认1000

        algorithm_mapping = {
            "pib": self._run_pib_optimization,
            "wf": self._simulate_wf_algorithm,
            "combine": self._simulate_combine_algorithm,
            "bayes-opt": self._simulate_bayes_opt_algorithm,
            "heuristic": self._simulate_heuristic_algorithm,
        } 
        
        while self.is_running and self.iteration < max_iterations:
            # 模拟算法执行
            if self.algorithm in algorithm_mapping:
                # 调用映射中的函数
                algorithm_mapping[self.algorithm]()
            else:
                # 未知算法，使用默认模拟
                raise ValueError(f"未知的优化算法: {self.algorithm}")
                
            # 记录历史
            self.history.append({
                "iteration": self.iteration,
                "voltages": self.voltages.copy(),
                "timestamp": time.time()
            })
            
            # 限制历史记录长度
            if len(self.history) > 1000:
                self.history.pop(0)
                
            # 发送更新信号
            self.simulationUpdated.emit(self.voltages.tolist())
            
            self.iteration += 1
            time.sleep(0.1)  # 控制模拟速度
            
        # 如果是因为达到迭代次数而停止，发送完成信号
        if self.iteration >= max_iterations:
            self.is_running = False
            self.simulationFinished.emit(None)

    def _run_pib_optimization(self):
        """运行PIB优化算法"""      
        # 创建回调函数用于实时更新
        def progress_callback(current_epoch, total_epochs, voltages):
            """进度回调函数"""
            if not self.is_running:
                return  # 直接返回表示停止优化
                
            # 更新内部状态
            self.voltages = np.array(voltages)
            self.iteration = current_epoch
            
            # 发送更新信号
            self.simulationUpdated.emit(self.voltages.tolist())
            
            # 记录历史
            self.history.append({
                "iteration": current_epoch,
                "voltages": self.voltages.copy(),
                "timestamp": time.time()
            })
            
            # 限制历史记录长度
            if len(self.history) > 1000:
                self.history.pop(0)
        
        # TODO: 实现PIB优化算法
            
        # 发送最终更新信号
        self.simulationUpdated.emit(self.voltages.tolist())
            
    def _simulate_wf_algorithm(self):
        """模拟波前优化算法"""
        # 简单的模拟实现，实际应用中需要复杂的波前计算
        # 这里只是随机微调电压值
        noise = np.random.normal(0, 0.01, 64)
        self.voltages = np.clip(self.voltages + noise, -1.0, 1.0)
        
    def _simulate_combine_algorithm(self):
        """模拟组合优化算法"""
        # 简单的模拟实现
        noise = np.random.normal(0, 0.01, 64)
        self.voltages = np.clip(self.voltages + noise, -1.0, 1.0)
        
    def _simulate_bayes_opt_algorithm(self):
        """模拟贝叶斯优化算法"""
        # 简单的模拟实现
        noise = np.random.normal(0, 0.01, 64)
        self.voltages = np.clip(self.voltages + noise, -1.0, 1.0)
        
    def _simulate_heuristic_algorithm(self):
        """模拟启发式搜索算法"""
        # 简单的模拟实现
        noise = np.random.normal(0, 0.01, 64)
        self.voltages = np.clip(self.voltages + noise, -1.0, 1.0)
        
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.history.copy()
        
    def cleanup(self):
        """清理资源"""
        self.stop_simulation()