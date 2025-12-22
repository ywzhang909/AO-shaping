import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from PySide6.QtCore import QObject, Signal, QThread

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.utils.file import get_init_V_by_rms


class PIBOptimizerWorker(QObject):
    """PIB优化器工作线程类"""
    
    # 定义信号
    progressUpdated = Signal(dict)  # 进度更新信号
    optimizationFinished = Signal(object)  # 优化完成信号，传递Recorder对象
    optimizationError = Signal(str)  # 错误信号
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        初始化PIB优化器工作线程
        
        Args:
            parameters (Dict[str, Any]): 优化参数
        """
        super().__init__()
        self.parameters = parameters.copy()
        self.is_running = False
        self.recorder = None
        
    def start_optimization(self):
        """开始优化"""
        try:
            self.is_running = True
            
            # 处理初始电压
            load_file = self.parameters.get('load_file', 'rms')
            if load_file.lower() == 'rms':
                init_v = get_init_V_by_rms()
            elif load_file:
                last_v = np.loadtxt(load_file)
                init_v = last_v.tolist()
            else:
                init_v = []
                
            # 设置变形镜单元掩码
            dm_unit_mask = np.ones(64, dtype=bool)
            dm_unit_mask[0] = False  # 禁用第一个单元
            
            # 调用optimize_pib函数
            self.recorder = optimize_pib(
                center=self.parameters.get('center', 'mass'),
                epochs=int(self.parameters.get('epochs', 4000)),
                r_bucket=float(self.parameters.get('r_bucket', 0)),
                delta=float(self.parameters.get('delta', 2)),
                lr=float(self.parameters.get('lr', 0)),
                exposure_time_ms=int(self.parameters.get('exposure_time_ms', 60)),
                shrink_iter=int(self.parameters.get('shrink_iter', 300)),
                shrink_ratio=float(self.parameters.get('shrink_ratio', 0.8)),
                cam_id=self.parameters.get('cam_id', 0),
                show=False,  # GUI中不显示
                init_v=init_v,
                cam_size=int(self.parameters.get('cam_size', 200)),
                target_max_brightness=int(self.parameters.get('target_max_brightness', 90)),
                dm_unit_mask=dm_unit_mask,
                dm_neibor_diff=400,
                dm_max_voltage=300,
                dm_min_voltage=-200,
            )
            
            # 发送完成信号
            self.optimizationFinished.emit(self.recorder)
            
        except Exception as e:
            self.optimizationError.emit(str(e))
        finally:
            self.is_running = False
            
    def stop_optimization(self):
        """停止优化"""
        self.is_running = False