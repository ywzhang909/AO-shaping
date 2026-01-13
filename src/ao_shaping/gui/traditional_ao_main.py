import sys
from PySide6.QtWidgets import QApplication
from ao_shaping.gui.traditional_ao_window import TraditionalAOWindow


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = TraditionalAOWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()