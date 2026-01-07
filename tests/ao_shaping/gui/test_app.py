#!/usr/bin/env python3
"""
测试AO Shaping GUI应用程序
"""

import sys


try:
    from PySide6.QtWidgets import QApplication
    from ao_shaping.gui.main_window import MainWindow
    print("成功导入PySide6和主窗口模块")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def test_imports():
    """测试导入"""
    try:
        from ao_shaping.gui.dm_panel import DMPanel
        from ao_shaping.gui.visualization_panel import VisualizationPanel
        from ao_shaping.gui.control_panel import ControlPanel
        from ao_shaping.gui.runner_manager import RunnerManager
        from ao_shaping.gui.simulation_manager import SimulationManager
        print("所有模块导入成功")
        return True
    except ImportError as e:
        print(f"模块导入失败: {e}")
        return False

def test_main_window():
    """测试主窗口创建"""
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        print("主窗口创建成功")
        return True
    except Exception as e:
        print(f"主窗口创建失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试AO Shaping GUI应用程序...")
    
    # 测试导入
    if not test_imports():
        sys.exit(1)
        
    # 测试主窗口
    if not test_main_window():
        sys.exit(1)
        
    print("所有测试通过!")