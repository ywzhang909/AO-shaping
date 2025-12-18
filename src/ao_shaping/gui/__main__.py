"""
AO Shaping GUI 应用程序入口点
"""

import sys

from PySide6.QtWidgets import QApplication
from .main_window import MainWindow


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()