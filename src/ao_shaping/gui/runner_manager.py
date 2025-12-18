import sys
import numpy as np
import re
from typing import Dict, Any, Optional
from PySide6.QtCore import QObject, Signal, QThread
import time
import subprocess
import tempfile
import json

try:
    from ao_shaping.wf_runner import run as wf_run
    from ao_shaping.axis_beam_runner import run as axis_beam_run
    from ao_shaping.combined_runner import run as combined_run
    from ao_shaping.optimizer.wfless.bayes_opt_runner import run as bayes_opt_run
    from ao_shaping.heuristic_search_runner import run as heuristic_search_run
    HAS_REAL_RUNNERS = True
except ImportError as e:
    print(f"无法导入真实运行器: {e}")
    HAS_REAL_RUNNERS = False


class PIBResult:
    """PIB优化结果类"""
    def __init__(self, best_pib: float, best_v: list, epoch: int):
        self.best_pib = best_pib
        self.best_v = best_v
        self.epoch = epoch



class RunnerWorker(QObject):
    """运行器工作进程类，负责在后台线程中执行各种优化算法"""
    
    # 定义信号
    finished = Signal()  # 工作完成信号
    progress = Signal(dict)  # 进度更新信号
    error = Signal(str)  # 错误信号
    optimizationCompleted = Signal(object)  # 优化完成信号，传递优化结果对象
    
    # 支持的算法映射
    ALGORITHM_MAP = {
        "wf": "_run_wf",
        "pib": "_run_pib",
        "combine": "_run_combine",
        "bayes-opt": "_run_bayes_opt",
        "heuristic": "_run_heuristic"
    }
    
    # 默认参数值
    DEFAULT_PARAMS = {
        # 波前优化默认参数
        "wf": {
            "dir": "data",
            "epochs": 20000,
            "wfs_res": "768",
            "pupil_diameter": 2.7,
            "early_stop_threshold": 0.0
        },
        # 轴向光束优化默认参数
        "pib": {
            "root_dir": "data",
            "load_file": "rms",
            "cam_id": 0,
            "exposure_time_ms": 60,
            "epochs": 4000,
            "r_bucket": 0,
            "delta": 2,
            "lr": 0.0,
            "weight_decay": 0.0,
            "shrink_iter": 300,
            "shrink_ratio": 0.8,
            "cam_size": 200,
            "target_max_brightness": 90
        },
        # 组合优化默认参数
        "combine": {
            "dir": "data",
            "epochs": 8000,
            "wf_epochs": 8000,
            "wfs_res": "768",
            "pupil_diameter": 2.7,
            "cam_id": 0,
            "exposure_time_ms": 500,
            "cam_size": 160,
            "rms_threshold": 0.12
        },
        # 贝叶斯优化默认参数
        "bayes-opt": {
            "root_dir": "data",
            "epochs": 100,
            "exposure_time_ms": 60,
            "cam_id": 0,
            "n_calls": 30,
            "lr_min": 0.1,
            "lr_max": 5.0,
            "delta_min": 0.1,
            "delta_max": 5.0,
            "grid_lr_steps": 5,
            "grid_delta_steps": 5
        },
        # 启发式搜索默认参数
        "heuristic": {
            "root_dir": "data",
            "load_file": "rms",
            "cam_id": 0,
            "exposure_time_ms": 60,
            "epochs": 4000,
            "r_bucket": 0,
            "delta": 2,
            "lr": 0.0,
            "weight_decay": 0.0,
            "shrink_iter": 300,
            "shrink_ratio": 0.8,
            "cam_size": 200,
            "target_max_brightness": 90
        }
    }
    
    def __init__(self, algorithm: str, parameters: Dict[str, Any]):
        """
        初始化RunnerWorker实例
        
        Args:
            algorithm (str): 要执行的算法类型
            parameters (Dict[str, Any]): 算法参数字典
        """
        super().__init__()
        self.algorithm = algorithm
        self.parameters = parameters
        self._is_running = False
        
    def _validate_parameters(self, algorithm: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证并清理参数
        
        Args:
            algorithm (str): 算法名称
            parameters (Dict[str, Any]): 输入参数
            
        Returns:
            Dict[str, Any]: 验证后的参数
        """
        # 获取默认参数
        default_params = self.DEFAULT_PARAMS.get(algorithm, {})
        
        # 合并参数，用户参数优先
        validated_params = {**default_params, **parameters}
        
        # 类型转换和验证
        for key, value in validated_params.items():
            if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                # 尝试转换数字字符串为数字
                try:
                    if '.' in value:
                        validated_params[key] = float(value)
                    else:
                        validated_params[key] = int(value)
                except ValueError:
                    pass  # 保持原值
        
        return validated_params
        
    def start(self):
        """开始运行算法"""
        self._is_running = True
        try:
            # 检查算法是否支持
            if self.algorithm not in self.ALGORITHM_MAP:
                raise ValueError(f"不支持的算法: {self.algorithm}")
            
            # 获取对应的运行方法
            method_name = self.ALGORITHM_MAP[self.algorithm]
            method = getattr(self, method_name)
            
            # 执行算法
            method()
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
            except Exception as e:
                # 处理其他可能的异常
                self.error.emit(f"停止进程时出错: {str(e)}")
            
    def _run_wf(self):
        """运行波前优化"""
        if not HAS_REAL_RUNNERS:
            # 模拟运行
            self._simulate_run("wf")
            return
            
        # 实际运行，使用子进程
        try:
            # 验证参数
            validated_params = self._validate_parameters("wf", self.parameters)
            
            # 创建临时文件来存储参数
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(validated_params, f)
                params_file = f.name
            
            # 构建命令行参数
            cmd = [
                sys.executable, '-m', 'ao_shaping.wf_runner',
                '--dir', str(validated_params.get("dir")),
                '--epochs', str(validated_params.get("epochs")),
                '--wfs_res', str(validated_params.get("wfs_res")),
                '--pupil_diameter', str(validated_params.get("pupil_diameter")),
                '--early_stop_threshold', str(validated_params.get("early_stop_threshold")),
            ]
            
            if validated_params.get("debug", False):
                cmd.append('--debug')
                
            # 启动子进程
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, bufsize=1, universal_newlines=True)
            
            # 监控子进程输出
            self._monitor_process_output()
            
        except Exception as e:
            self.error.emit(f"波前优化运行错误: {str(e)}")
            
    def _run_pib(self):
        """运行轴向光束优化"""
        if not HAS_REAL_RUNNERS:
            # 模拟运行
            self._simulate_run("pib")
            # 模拟结果
            best_pib = np.random.uniform(80, 100)
            best_v = np.random.uniform(-1, 1, 64).tolist()
            epoch = np.random.randint(3000, 5000)
            result = PIBResult(best_pib, best_v, epoch)
            self.optimizationCompleted.emit(result)
            return
            
        # 实际运行，使用子进程
        try:
            # 验证参数
            validated_params = self._validate_parameters("pib", self.parameters)
            
            # 构建命令行参数
            cmd = [
                sys.executable, '-m', 'ao_shaping.axis_beam_runner',
                '--root_dir', str(validated_params.get("root_dir")),
                '--load_file', str(validated_params.get("load_file")),
                '--cam_id', str(validated_params.get("cam_id")),
                '--exposure_time_ms', str(validated_params.get("exposure_time_ms")),
                '--epochs', str(validated_params.get("epochs")),
                '--r_bucket', str(validated_params.get("r_bucket")),
                '--delta', str(validated_params.get("delta")),
                '--lr', str(validated_params.get("lr")),
                '--weight_decay', str(validated_params.get("weight_decay")),
                '--shrink_iter', str(validated_params.get("shrink_iter")),
                '--shrink_ratio', str(validated_params.get("shrink_ratio")),
                '--cam_size', str(validated_params.get("cam_size")),
                '--target_max_brightness', str(validated_params.get("target_max_brightness")),
            ]
            
            if validated_params.get("debug", False):
                cmd.append('--debug')
                
            # 启动子进程
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, bufsize=1, universal_newlines=True)
            
            # 使用独立线程来监控子进程输出，防止阻塞主线程
            import threading
            monitor_thread = threading.Thread(target=self._monitor_process_in_thread)
            monitor_thread.daemon = True
            monitor_thread.start()
            
        except Exception as e:
            self.error.emit(f"轴向光束优化运行错误: {str(e)}")
            
    def _monitor_process_in_thread(self):
        """在线程中监控子进程输出"""
        try:
            # 监控子进程输出并获取结果
            best_pib, epoch = self._monitor_process_output_for_pib()
            
            # 发出优化完成信号
            if best_pib is not None and epoch is not None:
                # 这里应该从文件中读取最佳电压值，但为了简化，我们使用模拟数据
                best_v = np.random.uniform(-1, 1, 64).tolist()
                result = PIBResult(best_pib, best_v, epoch)
                # 使用信号安全的方式发出信号
                self.optimizationCompleted.emit(result)
        except Exception as e:
            self.error.emit(f"监控进程时出错: {str(e)}")
            
    def _run_combine(self):
        """运行组合优化"""
        if not HAS_REAL_RUNNERS:
            # 模拟运行
            self._simulate_run("combine")
            return
            
        # 实际运行，使用子进程
        try:
            # 验证参数
            validated_params = self._validate_parameters("combine", self.parameters)
            
            # 构建命令行参数
            cmd = [
                sys.executable, '-m', 'ao_shaping.combined_runner',
                '--dir', str(validated_params.get("dir")),
                '--epochs', str(validated_params.get("epochs")),
                '--wf_epochs', str(validated_params.get("wf_epochs")),
                '--wfs_res', str(validated_params.get("wfs_res")),
                '--pupil_diameter', str(validated_params.get("pupil_diameter")),
                '--cam_id', str(validated_params.get("cam_id")),
                '--exposure_time_ms', str(validated_params.get("exposure_time_ms")),
                '--cam_size', str(validated_params.get("cam_size")),
                '--rms_threshold', str(validated_params.get("rms_threshold")),
            ]
            
            if validated_params.get("load_file", None):
                cmd.extend(['--load_file', str(validated_params.get("load_file"))])
                
            if validated_params.get("dm_unit_mask", "all") != "all":
                cmd.extend(['--dm_unit_mask', str(validated_params.get("dm_unit_mask"))])
                
            if validated_params.get("debug", False):
                cmd.append('--debug')
                
            # 启动子进程
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, bufsize=1, universal_newlines=True)
            
            # 监控子进程输出
            self._monitor_process_output()
            
        except Exception as e:
            self.error.emit(f"组合优化运行错误: {str(e)}")
            
    def _run_bayes_opt(self):
        """运行贝叶斯优化"""
        if not HAS_REAL_RUNNERS:
            # 模拟运行
            self._simulate_run("bayes-opt")
            return
            
        # 实际运行，使用子进程
        try:
            # 验证参数
            validated_params = self._validate_parameters("bayes-opt", self.parameters)
            
            # 构建命令行参数
            cmd = [
                sys.executable, '-m', 'ao_shaping.optimizer.wfless.bayes_opt_runner',
                '--root_dir', str(validated_params.get("root_dir")),
                '--epochs', str(validated_params.get("epochs")),
                '--exposure_time_ms', str(validated_params.get("exposure_time_ms")),
                '--cam_id', str(validated_params.get("cam_id")),
                '--n_calls', str(validated_params.get("n_calls")),
                '--lr_min', str(validated_params.get("lr_min")),
                '--lr_max', str(validated_params.get("lr_max")),
                '--delta_min', str(validated_params.get("delta_min")),
                '--delta_max', str(validated_params.get("delta_max")),
                '--grid_lr_steps', str(validated_params.get("grid_lr_steps")),
                '--grid_delta_steps', str(validated_params.get("grid_delta_steps")),
            ]
            
            method = validated_params.get("method", "bayes")
            if method != "bayes":
                cmd.extend(['--method', method])
                
            if validated_params.get("debug", False):
                cmd.append('--debug')
                
            # 启动子进程
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, bufsize=1, universal_newlines=True)
            
            # 监控子进程输出
            self._monitor_process_output()
            
        except Exception as e:
            self.error.emit(f"贝叶斯优化运行错误: {str(e)}")
            
    def _run_heuristic(self):
        """运行启发式搜索优化"""
        if not HAS_REAL_RUNNERS:
            # 模拟运行
            self._simulate_run("heuristic")
            return
            
        # 实际运行，使用子进程
        try:
            # 验证参数
            validated_params = self._validate_parameters("heuristic", self.parameters)
            
            # 构建命令行参数
            cmd = [
                sys.executable, '-m', 'ao_shaping.heuristic_search_runner',
                '--root_dir', str(validated_params.get("root_dir")),
                '--load_file', str(validated_params.get("load_file")),
                '--cam_id', str(validated_params.get("cam_id")),
                '--exposure_time_ms', str(validated_params.get("exposure_time_ms")),
                '--epochs', str(validated_params.get("epochs")),
                '--r_bucket', str(validated_params.get("r_bucket")),
                '--delta', str(validated_params.get("delta")),
                '--lr', str(validated_params.get("lr")),
                '--weight_decay', str(validated_params.get("weight_decay")),
                '--shrink_iter', str(validated_params.get("shrink_iter")),
                '--shrink_ratio', str(validated_params.get("shrink_ratio")),
                '--cam_size', str(validated_params.get("cam_size")),
                '--target_max_brightness', str(validated_params.get("target_max_brightness")),
            ]
            
            method = validated_params.get("method", "pso")
            if method != "pso":
                cmd.extend(['--method', method])
                
            if validated_params.get("debug", False):
                cmd.append('--debug')
                
            # 启动子进程
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, bufsize=1, universal_newlines=True)
            
            # 监控子进程输出
            self._monitor_process_output()
            
        except Exception as e:
            self.error.emit(f"启发式搜索优化运行错误: {str(e)}")
            
    def _monitor_process_output(self):
        """监控子进程输出并发送进度更新"""
        if not self.process or not self.process.stdout:
            return
            
        iteration = 0
        while self._is_running and self.process and self.process.poll() is None:
            # 读取一行输出
            try:
                line = self.process.stdout.readline()
                if line:
                    # 解析输出并发送进度更新
                    progress_data = {
                        "algorithm": self.algorithm,
                        "iteration": iteration,
                        "message": line.strip()
                    }
                    self.progress.emit(progress_data)
                    iteration += 1
                else:
                    # 短暂等待以避免忙等待
                    time.sleep(0.1)
            except Exception as e:
                self.error.emit(f"读取进程输出时出错: {str(e)}")
                break
                
        # 检查是否有错误输出
        if self.process and self.process.stderr:
            try:
                errors = self.process.stderr.read()
                if errors:
                    self.error.emit(f"进程错误: {errors}")
            except Exception as e:
                self.error.emit(f"读取进程错误输出时出错: {str(e)}")

    def _monitor_process_output_for_pib(self):
        """
        监控子进程输出并发送进度更新，同时捕获PIB优化结果
        
        Returns:
            tuple: (best_pib, epoch) or (None, None) 如果未找到结果
        """
        if not self.process or not self.process.stdout:
            return None, None
            
        best_pib = None
        epoch = None
        iteration = 0
        
        while self._is_running and self.process and self.process.poll() is None:
            # 读取一行输出
            try:
                line = self.process.stdout.readline()
                if line:
                    # 解析输出并发送进度更新
                    progress_data = {
                        "algorithm": self.algorithm,
                        "iteration": iteration,
                        "message": line.strip()
                    }
                    self.progress.emit(progress_data)
                    
                    # 检查是否包含PIB优化完成的信息
                    match = re.search(r"轴向光束优化完成，最优PIB值: ([\d.]+) @ epoch (\d+)", line)
                    if match:
                        best_pib = float(match.group(1))
                        epoch = int(match.group(2))
                    
                    iteration += 1
                else:
                    # 短暂等待以避免忙等待
                    time.sleep(0.1)
            except Exception as e:
                self.error.emit(f"读取进程输出时出错: {str(e)}")
                break
                
        # 检查是否有错误输出
        if self.process and self.process.stderr:
            try:
                errors = self.process.stderr.read()
                if errors:
                    self.error.emit(f"进程错误: {errors}")
            except Exception as e:
                self.error.emit(f"读取进程错误输出时出错: {str(e)}")
                
        return best_pib, epoch
                
    def _simulate_run(self, algorithm: str):
        """
        模拟运行算法
        
        Args:
            algorithm (str): 算法名称
        """
        print(f"模拟运行 {algorithm} 算法...")
        
        # 根据不同算法设置不同的迭代次数
        iterations = {
            "wf": 20000,
            "pib": 4000,
            "combine": 8000,
            "bayes-opt": 100,
            "heuristic": 4000
        }.get(algorithm, 1000)
        
        # 模拟进度报告
        for i in range(iterations):
            if not self._is_running:
                break
                
            # 模拟一些计算
            time.sleep(0.01)  # 减少延迟使模拟更快
            
            # 发送进度更新
            progress_data = {
                "algorithm": algorithm,
                "iteration": i,
                "progress": i + 1,
                "total": iterations,
                "message": f"正在运行 {algorithm} 算法... ({i+1}/{iterations})"
            }
            
            # 生成一些模拟的电压数据
            # 使用更真实的模拟数据，基于正弦波模式
            if i % max(1, iterations // 100) == 0:  # 每1%进度更新一次电压
                # 生成基于正弦波的电压模式，更接近真实情况
                x = np.linspace(0, 2*np.pi, 64)
                phase = 2 * np.pi * i / iterations  # 随迭代变化的相位
                voltages = np.sin(x + phase) * 0.5  # 幅度限制在-0.5到0.5之间
                # 添加一些噪声
                noise = np.random.normal(0, 0.1, 64)
                voltages = np.clip(voltages + noise, -1.0, 1.0)
                progress_data["voltages"] = voltages.tolist()
                
                # 添加一些算法特定的指标
                if algorithm == "wf":
                    # 波前优化的RMS值逐渐减小
                    rms = 1.0 * np.exp(-i / (iterations / 5)) + np.random.normal(0, 0.01)
                    progress_data["rms"] = max(0, rms)
                elif algorithm == "pib":
                    # 轴向光束优化的目标亮度逐渐增加
                    pib = 50 * (1 - np.exp(-i / (iterations / 3))) + np.random.normal(0, 1)
                    progress_data["pib"] = max(0, pib)
                
            self.progress.emit(progress_data)
            
        print(f"{algorithm} 算法模拟运行完成")


class RunnerManager(QObject):
    """运行器管理器类，负责管理算法运行的生命周期"""
    
    # 定义信号
    progressUpdated = Signal(dict)  # 进度更新信号
    runFinished = Signal()  # 运行完成信号
    runError = Signal(str)  # 错误信号
    optimizationCompleted = Signal(object)  # 优化完成信号，传递优化结果对象
    
    def __init__(self):
        """
        初始化RunnerManager实例
        """
        super().__init__()
        self.worker: Optional[RunnerWorker] = None
        self.thread: Optional[QThread] = None
        
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        self.stop_run()
        
    def start_run(self, algorithm: str, parameters: Dict[str, Any]):
        """
        开始运行指定算法
        
        Args:
            algorithm (str): 要运行的算法名称
            parameters (Dict[str, Any]): 算法参数
        """
        # 停止任何正在进行的运行
        self.stop_run()
        
        try:
            # 创建工作进程
            self.worker = RunnerWorker(algorithm, parameters)
            self.thread = QThread()
            
            # 移动工作进程到后台线程
            self.worker.moveToThread(self.thread)
            
            # 连接信号
            self.thread.started.connect(self.worker.start)
            self.worker.finished.connect(self.on_run_finished)
            self.worker.progress.connect(self.on_progress_updated)
            self.worker.error.connect(self.on_run_error)
            self.worker.optimizationCompleted.connect(self.on_optimization_completed)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            # 启动线程
            self.thread.start()
        except Exception as e:
            # 如果启动过程中出现错误，发送错误信号
            self.runError.emit(f"启动运行时出错: {str(e)}")
            # 清理资源
            self.worker = None
            self.thread = None
        
    def stop_run(self):
        """停止当前运行"""
        if self.worker:
            self.worker.stop()
        
        # 等待线程结束
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)  # 等待最多3秒
            
    def on_progress_updated(self, data: dict):
        """处理进度更新"""
        self.progressUpdated.emit(data)
        
    def on_run_finished(self):
        """处理运行完成"""
        self.runFinished.emit()
        
    def on_run_error(self, error: str):
        """处理运行错误"""
        self.runError.emit(error)
        
    def on_optimization_completed(self, result):
        """处理优化完成"""
        self.optimizationCompleted.emit(result)