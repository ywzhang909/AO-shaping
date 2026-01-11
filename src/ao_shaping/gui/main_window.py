"""
AO系统仿真GUI主窗口

提供完整的AO系统仿真控制和可视化界面
"""

import sys
import json
import os
from typing import Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStatusBar, QMenuBar, QMenu, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction

from .panels.config_panel import ConfigPanel
from .panels.control_panel import ControlPanel
from .panels.results_panel import ResultsPanel
from .workers.simulation_worker import SimulationWorker
from ..sim.devices import TraditionalAOSystem, AOConfig


class MainWindow(QMainWindow):
    """
    AO系统仿真GUI主窗口

    包含配置面板、控制面板和结果显示面板
    """

    def __init__(self):
        super().__init__()
        self.ao_system: Optional[TraditionalAOSystem] = None
        self.config = AOConfig()
        self.worker = SimulationWorker()

        self.init_ui()
        self.init_system()
        self.setup_connections()
        self.setup_worker_connections()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("AO系统仿真控制台")
        self.setGeometry(100, 100, 1400, 900)

        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建分割器
        self.splitter = QSplitter(Qt.Horizontal)

        # 创建面板
        self.config_panel = ConfigPanel(self.config)
        self.control_panel = ControlPanel()
        self.results_panel = ResultsPanel()

        # 添加面板到分割器
        self.splitter.addWidget(self.config_panel)
        self.splitter.addWidget(self.control_panel)
        self.splitter.addWidget(self.results_panel)

        # 设置分割器比例
        self.splitter.setSizes([300, 300, 800])

        main_layout.addWidget(self.splitter)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建状态栏
        self.create_status_bar()

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        # 加载配置
        load_config_action = QAction("加载配置(&L)", self)
        load_config_action.triggered.connect(self.load_config)
        file_menu.addAction(load_config_action)

        # 保存配置
        save_config_action = QAction("保存配置(&S)", self)
        save_config_action.triggered.connect(self.save_config)
        file_menu.addAction(save_config_action)

        file_menu.addSeparator()

        # 退出
        exit_action = QAction("退出(&X)", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")

        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

        # 添加永久显示的组件
        self.strehl_label = self.status_bar.addPermanentWidget(QWidget())
        self.strehl_label.setText("Strehl比: --")

    def init_system(self):
        """初始化AO系统"""
        # 设置工作线程配置
        self.worker.set_config(self.config)

        # 启动初始化
        if not self.worker.isRunning():
            self.worker.initialize_system()
            self.worker.start()

    def setup_connections(self):
        """设置信号连接"""
        # 配置面板变化
        self.config_panel.config_changed.connect(self.on_config_changed)

        # 控制面板操作
        self.control_panel.reset_requested.connect(self.reset_system)
        self.control_panel.step_requested.connect(self.step_simulation)
        self.control_panel.voltages_changed.connect(self.on_voltages_changed)

    def setup_worker_connections(self):
        """设置工作线程信号连接"""
        self.worker.simulation_finished.connect(self.on_simulation_finished)
        self.worker.error_occurred.connect(self.on_simulation_error)
        self.worker.progress_updated.connect(self.on_progress_updated)

    def on_config_changed(self, new_config: AOConfig):
        """配置变化处理"""
        self.config = new_config

        # 更新控制面板的致动器数量
        self.control_panel.set_num_actuators(new_config.dm_actuators)

        # 更新波前显示的子孔径数量
        self.results_panel.wavefront_display.set_subapertures(new_config.subapertures)

        # 使用工作线程重新初始化系统
        if not self.worker.isRunning():
            self.worker.set_config(new_config)
            self.worker.initialize_system()
            self.worker.start()
            self.status_bar.showMessage("正在重新初始化系统...")

    def on_voltages_changed(self, voltages):
        """DM电压变化处理"""
        if self.ao_system and not self.worker.isRunning():
            self.worker.set_voltages(np.array(voltages))
            self.worker.start()

    def reset_system(self):
        """重置系统"""
        if self.ao_system and not self.worker.isRunning():
            self.worker.reset_system()
            self.worker.start()

    def step_simulation(self):
        """执行单步仿真"""
        if self.ao_system and not self.worker.isRunning():
            # 使用零动作进行单步
            action = np.zeros(self.ao_system.dm.total_actuators)
            self.worker.step_simulation(action)
            self.worker.start()

    def on_simulation_finished(self, result: dict):
        """仿真完成处理"""
        try:
            # 更新AO系统引用
            if hasattr(self.worker, 'ao_system') and self.worker.ao_system:
                self.ao_system = self.worker.ao_system

            # 更新显示
            if 'image' in result:
                self.results_panel.update_image(result['image'])
            if 'slopes' in result:
                self.results_panel.update_wavefront(result['slopes'])

            # 更新指标
            metrics = {}
            if 'strehl' in result:
                metrics['strehl'] = result['strehl']
                self.strehl_label.setText(".3f")
            if 'power' in result:
                metrics['power'] = result['power']
            if 'voltages' in result:
                # 更新控制面板的电压显示
                self.control_panel.set_voltages(result['voltages'])

            self.results_panel.update_metrics(**metrics)

            self.status_bar.showMessage("仿真完成")

        except Exception as e:
            print(f"处理仿真结果时出错: {e}")

    def on_simulation_error(self, error_msg: str):
        """仿真错误处理"""
        QMessageBox.critical(self, "仿真错误", f"仿真过程中发生错误:\n{error_msg}")
        self.status_bar.showMessage("仿真失败")

    def on_progress_updated(self, message: str):
        """进度更新处理"""
        self.status_bar.showMessage(message)

    def load_config(self):
        """加载配置"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "加载配置", "", "JSON文件 (*.json);;所有文件 (*)"
            )

            if not file_path:
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 创建AOConfig对象
            new_config = AOConfig(**config_data)

            # 更新配置面板
            self.config_panel.load_config_from_data(config_data)

            # 触发配置变化
            self.on_config_changed(new_config)

            self.status_bar.showMessage(f"配置已加载: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"加载配置文件失败:\n{str(e)}")

    def save_config(self):
        """保存配置"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存配置", "ao_config.json", "JSON文件 (*.json);;所有文件 (*)"
            )

            if not file_path:
                return

            # 获取当前配置数据
            config_data = {
                'N': self.config.N,
                'L': self.config.L,
                'wavelength': self.config.wavelength,
                'Cn2': self.config.Cn2,
                'L0': self.config.L0,
                'l0': self.config.l0,
                'dm_actuators': self.config.dm_actuators,
                'dm_stroke': self.config.dm_stroke,
                'dm_infill': self.config.dm_infill,
                'subapertures': self.config.subapertures,
                'pixel_scale': self.config.pixel_scale,
                'propagation_distance': self.config.propagation_distance
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            self.status_bar.showMessage(f"配置已保存: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"保存配置文件失败:\n{str(e)}")

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于 AO 系统仿真",
            "AO-Shaping 自适应光学系统仿真控制台\n\n"
            "基于 PySide6 和 matplotlib 开发\n"
            "支持传统 AO 系统仿真和可视化"
        )

    def closeEvent(self, event):
        """关闭事件处理"""
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要退出应用程序吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 停止工作线程
            if self.worker.isRunning():
                self.worker.quit()
                self.worker.wait(3000)  # 等待最多3秒
            event.accept()
        else:
            event.ignore()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("AO系统仿真控制台")
    app.setApplicationVersion("1.0.0")

    # 设置样式
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()</content>
</xai:function_call">PySide6主窗口类已创建。现在创建配置面板，它需要根据AOConfig的数据类型选择合适的控件。让我创建config_panel.py文件。