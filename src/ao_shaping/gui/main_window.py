import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter, QStatusBar,
    QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, QTimer

from .panels.dm_panel import DMPanel
from .panels.visualization_panel import VisualizationPanel
from .panels.control_panel import ControlPanel
from .workers.runner_manager import RunnerManager
from .workers.simulation_manager import SimulationManager


class MainWindow(QMainWindow):
    """AO Shaping 主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.runner_manager = RunnerManager()
        self.simulation_manager = SimulationManager()
        self.is_running = False  # 添加运行状态标志
        self.init_ui()
        self.connect_signals()
        self.update_button_states()  # 初始化按钮状态

    def update_button_states(self):
        """更新按钮状态"""
        # 在控制面板中设置按钮的启用/禁用状态
        if hasattr(self, 'control_panel'):
            self.control_panel.start_button.setEnabled(not self.is_running)
            self.control_panel.stop_button.setEnabled(self.is_running)
            self.control_panel.reset_button.setEnabled(not self.is_running)

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('AO Shaping 可视化控制系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 创建左侧控制面板
        self.control_panel = ControlPanel(None)  # 暂时不传递模拟管理器
        splitter.addWidget(self.control_panel)
        
        # 创建右侧显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 创建变形镜面板
        self.dm_panel = DMPanel()
        self.dm_panel.valueChanged.connect(self.on_dm_value_changed)
        right_layout.addWidget(self.dm_panel)
        
        # 创建可视化面板
        self.visualization_panel = VisualizationPanel()
        right_layout.addWidget(self.visualization_panel)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])  # 设置初始大小比例
        
        main_layout.addWidget(splitter)
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪')
        
    def connect_signals(self):
        """连接信号"""
        # 连接控制面板信号
        self.control_panel.startRequested.connect(self.start_run)
        self.control_panel.stopRequested.connect(self.stop_run)
        self.control_panel.resetRequested.connect(self.reset_system)
        
        # 连接运行器管理器信号
        self.runner_manager.progressUpdated.connect(self.on_progress_updated)
        self.runner_manager.runFinished.connect(self.on_run_finished)
        self.runner_manager.runError.connect(self.on_run_error)
        self.runner_manager.optimizationCompleted.connect(self.on_optimization_completed)
        
        # 连接模拟管理器信号
        self.simulation_manager.simulationUpdated.connect(self.on_simulation_updated)
        self.simulation_manager.simulationFinished.connect(self.on_simulation_finished)
        self.simulation_manager.simulationError.connect(self.on_simulation_error)
        
    def on_dm_value_changed(self, unit_index: int, value: float):
        """处理变形镜单元值变化事件"""
        # 更新模拟管理器中的电压值
        self.simulation_manager.set_voltage(unit_index, value)
        # 更新可视化
        self.update_visualization()
        
    def start_run(self):
        """启动运行"""
        if self.is_running:
            return
            
        self.is_running = True
        self.update_button_states()  # 更新按钮状态
        try:
            # 获取参数
            params = self.control_panel.get_parameters()
            
            # 获取算法名称
            algorithm_map = {
                "波前优化 (wf)": "wf",
                "轴向光束优化 (pib)": "pib",
                "组合优化 (combine)": "combine",
                "贝叶斯优化 (bayes-opt)": "bayes-opt",
                "启发式搜索 (heuristic)": "heuristic"
            }
            algorithm = algorithm_map.get(params["algorithm"], "wf")
            
            # 检查是否启用实时优化模式且算法为PIB
            realtime_optimization = params.get("realtime_optimization", True)
            
            if realtime_optimization and algorithm == "pib":
                # 使用RunnerManager运行实际的PIB优化
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # 设置为不确定模式
                self.status_bar.showMessage(f'正在运行 {algorithm} 算法...')
                
                # 启动实际运行
                self.runner_manager.start_run(algorithm, params)
            else:
                # 使用SimulationManager运行模拟或非实时优化
                self.simulation_manager.set_algorithm(algorithm)
                self.simulation_manager.set_parameters(params)
                
                # 显示进度条
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # 设置为不确定模式
                self.status_bar.showMessage(f'正在运行 {algorithm} 算法...')
                
                # 启动模拟
                self.simulation_manager.start_simulation()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动运行时发生错误: {str(e)}")
            
    def stop_run(self):
        """停止运行"""
        self.is_running = False
        self.update_button_states()  # 更新按钮状态
        
        # 使用QTimer来异步停止运行，避免阻塞UI线程
        QTimer.singleShot(0, self._async_stop_run)
        
    def _async_stop_run(self):
        """异步停止运行"""
        try:
            self.runner_manager.stop_run()
            self.simulation_manager.stop_simulation()
        except Exception as e:
            print(f"停止运行时发生错误: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage('运行已停止')
        
    def reset_system(self):
        """重置系统"""
        self.is_running = False  # 重置运行状态
        self.update_button_states()  # 更新按钮状态
        self.simulation_manager.reset()
        self.dm_panel.reset_values()
        self.status_bar.showMessage('系统已重置')
        self.update_visualization()
        
    def on_progress_updated(self, data: dict):
        """处理进度更新"""
        # 更新状态栏消息
        if "message" in data:
            self.status_bar.showMessage(data["message"])
            
        # 更新电压显示
        if "voltages" in data:
            self.dm_panel.set_values(data["voltages"])
            self.visualization_panel.update_plots(data["voltages"])
            
        # 更新RMS或PIB历史（如果有）
        if "rms" in data:
            self.visualization_panel.add_rms_value(data["rms"])
        elif "pib" in data:
            self.visualization_panel.add_pib_value(data["pib"])
            
    def on_run_finished(self):
        """处理运行完成"""
        self.is_running = False
        self.update_button_states()  # 更新按钮状态
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('运行完成')
        QMessageBox.information(self, "完成", "算法运行已完成")
        
    def on_run_error(self, error: str):
        """处理运行错误"""
        self.is_running = False
        self.update_button_states()  # 更新按钮状态
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('运行出错')
        QMessageBox.critical(self, "错误", f"运行过程中发生错误: {error}")
        
    def on_optimization_completed(self, result):
        """处理优化完成"""
        self.is_running = False
        self.update_button_states()  # 更新按钮状态
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('优化完成')
        
        # 显示优化结果
        if hasattr(result, 'best_pib'):
            # 显示最佳PIB值
            self.status_bar.showMessage(f'PIB优化完成，最佳PIB值: {result.best_pib:.4f}')
            
            # 更新变形镜面板和可视化面板的最佳电压值
            if hasattr(result, 'best_v'):
                self.dm_panel.set_values(result.best_v)
                self.visualization_panel.update_plots(result.best_v)
                
            # 显示完成消息框
            QMessageBox.information(self, "完成", f"PIB优化已完成，最佳PIB值: {result.best_pib:.4f}")
        else:
            # 如果没有特定的结果对象，显示通用完成消息
            QMessageBox.information(self, "完成", "优化已完成")
        
    def on_simulation_updated(self, voltages: list):
        """处理模拟更新"""
        # 更新变形镜面板
        self.dm_panel.set_values(voltages)
        
        # 更新可视化面板
        self.visualization_panel.update_plots(voltages)
        
        # 更新状态栏
        self.status_bar.showMessage(f'模拟运行中... 迭代次数: {self.simulation_manager.iteration}')
        
    def on_simulation_finished(self, recorder=None):
        """处理模拟完成"""
        self.is_running = False
        self.update_button_states()  # 更新按钮状态
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('运行完成')
        
        # 如果是PIB优化且有Recorder对象，则显示优化结果
        if recorder is not None:
            # 显示最佳PIB值
            if hasattr(recorder, 'best_pib'):
                self.status_bar.showMessage(f'PIB优化完成，最佳PIB值: {recorder.best_pib:.4f}')
                
            # 更新可视化面板
            if hasattr(recorder, 'best_v'):
                self.dm_panel.set_values(recorder.best_v)
                self.visualization_panel.update_plots(recorder.best_v)
                
            # 显示完成消息
            QMessageBox.information(self, "完成", f"PIB优化已完成，最佳PIB值: {recorder.best_pib:.4f}")
        else:
            self.status_bar.showMessage('模拟完成')
        
    def on_simulation_error(self, error: str):
        """处理模拟错误"""
        self.is_running = False
        self.update_button_states()  # 更新按钮状态
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage('模拟出错')
        QMessageBox.critical(self, "模拟错误", f"模拟过程中发生错误: {error}")
        
    def update_visualization(self):
        """更新可视化显示"""
        # 获取当前电压值
        voltages = self.dm_panel.get_values()
        
        # 更新可视化面板
        self.visualization_panel.update_plots(voltages)
        
    def closeEvent(self, event):
        """处理窗口关闭事件"""
        try:
            # 停止所有运行的进程
            self.runner_manager.stop_run()
            self.simulation_manager.stop_simulation()
            self.simulation_manager.cleanup()
        except RuntimeError:
            # 忽略Qt对象已被删除的运行时错误
            pass
        except Exception as e:
            # 打印其他异常但不中断关闭过程
            print(f"关闭时出现错误: {str(e)}")
        
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
