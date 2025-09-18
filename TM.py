# main.py
import sys, datetime, pickle, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIntValidator
from serial_port_fsm import SerialPortFSM

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FSM 位置调试助手')
        self.resize(900, 500)
        self.ser = SerialPortFSM()
        self.build_ui()
        # 加载上次保存的位置
        self.load_last_position()
        self.timer = QTimer(self, timeout=self.on_timer)
        self.timer.start(50)

    # -------------------- 界面 --------------------
    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        lo = QHBoxLayout(central)

        # 左：串口控制
        left = QFormLayout()
        self.cmb_port = QComboBox()
        self.cmb_port.addItems(self.ser.list_port() or ['COM_Mock'])
        self.cmb_baud = QComboBox(); self.cmb_baud.addItems(['9600','115200']); self.cmb_baud.setCurrentText('115200')
        self.btn_open = QPushButton('打开串口')
        self.btn_open.setCheckable(True)
        self.btn_open.toggled.connect(self.on_open)
        left.addRow('串口', self.cmb_port)
        left.addRow('波特率', self.cmb_baud)
        left.addRow(self.btn_open)
        lo.addLayout(left)

        # 中：报文显示
        mid = QVBoxLayout()
        self.txt_log = QTextEdit(readOnly=True)
        self.txt_log.setFont(QFont('Consolas', 9))
        mid.addWidget(self.txt_log)
        lo.addLayout(mid, stretch=2)

        # 右：快速发送
        right = QVBoxLayout()
        self.slider_x = QSlider(Qt.Horizontal); self.slider_x.setRange(-1000, 1000); self.slider_x.setValue(0)
        self.slider_y = QSlider(Qt.Horizontal); self.slider_y.setRange(-1000, 1000); self.slider_y.setValue(0)
        self.lab_x = QLabel('X: 0.00 mm'); self.lab_y = QLabel('Y: 0.00 mm')
        # 添加输入框
        self.txt_x = QLineEdit('0'); self.txt_x.setValidator(QIntValidator(-1000, 1000))
        self.txt_y = QLineEdit('0'); self.txt_y.setValidator(QIntValidator(-1000, 1000))
        
        self.slider_x.valueChanged.connect(self.update_x_label)
        self.slider_y.valueChanged.connect(self.update_y_label)
        # 连接信号：滑块变化时更新输入框（使用具名函数）
        self.slider_x.valueChanged.connect(self.update_x_text)
        self.slider_y.valueChanged.connect(self.update_y_text)
        # 连接信号：输入框变化时更新滑块
        self.txt_x.textChanged.connect(self.on_x_text_changed)
        self.txt_y.textChanged.connect(self.on_y_text_changed)
        
        self.btn_send = QPushButton('下发位置')
        self.btn_send.clicked.connect(self.on_send)
        right.addWidget(self.lab_x)
        right.addWidget(self.slider_x)
        right.addWidget(self.txt_x)
        right.addWidget(self.lab_y)
        right.addWidget(self.slider_y)
        right.addWidget(self.txt_y)
        right.addWidget(self.btn_send)
        right.addStretch()
        lo.addLayout(right)

    # -------------------- 辅助方法 --------------------
    def update_x_label(self, value):
        self.lab_x.setText(f'X: {value/10:.2f} mm')
    
    def update_y_label(self, value):
        self.lab_y.setText(f'Y: {value/10:.2f} mm')
    
    def update_x_text(self, value):
        self.txt_x.setText(str(value))
    
    def update_y_text(self, value):
        self.txt_y.setText(str(value))
        
    # 保存当前位置到文件
    def save_position(self, x, y):
        try:
            with open('last_position.pkl', 'wb') as f:
                pickle.dump((x, y), f)
            self.log(f'已保存位置: X={x:.2f}, Y={y:.2f}')
        except Exception as e:
            self.log(f'保存位置失败: {str(e)}')
            
    # 从文件加载上次保存的位置
    def load_last_position(self):
        try:
            if os.path.exists('last_position.pkl'):
                with open('last_position.pkl', 'rb') as f:
                    x_value, y_value = pickle.load(f)
                    # 转换为滑块值（乘以10）
                    x_int = int(x_value * 10)
                    y_int = int(y_value * 10)
                    # 确保值在有效范围内
                    x_int = max(-1000, min(1000, x_int))
                    y_int = max(-1000, min(1000, y_int))
                    # 更新滑块和输入框
                    self.slider_x.setValue(x_int)
                    self.slider_y.setValue(y_int)
                    self.txt_x.setText(str(x_int))
                    self.txt_y.setText(str(y_int))
                    self.log(f'加载上次位置: X={x_value:.2f}, Y={y_value:.2f}')
        except Exception as e:
            self.log(f'加载位置失败: {str(e)}')

    # -------------------- 槽 --------------------
    def on_open(self, chk):
        if chk:
            self.ser.ser.port = self.cmb_port.currentText()
            self.ser.ser.baudrate = int(self.cmb_baud.currentText())
            try:
                ok = self.ser.open()
                if ok:
                    self.btn_open.setText('关闭串口')
                    self.log('串口已打开')
            except Exception as e:
                self.log('打开串口失败：' + str(e))
                self.btn_open.setChecked(False)
        else:
            self.ser.close()
            self.btn_open.setText('打开串口')
            self.log('串口已关闭')

    # 修改方法：处理X输入框变化
    def on_x_text_changed(self, text):
        try:
            if text:  # 确保文本不为空
                value = int(text)
                # 限制在有效范围内
                value = max(-1000, min(1000, value))
                # 临时断开连接避免循环更新
                self.slider_x.valueChanged.disconnect(self.update_x_text)
                self.slider_x.setValue(value)
                # 重新连接
                self.slider_x.valueChanged.connect(self.update_x_text)
        except ValueError:
            pass

    # 修改方法：处理Y输入框变化
    def on_y_text_changed(self, text):
        try:
            if text:  # 确保文本不为空
                value = int(text)
                # 限制在有效范围内
                value = max(-1000, min(1000, value))
                # 临时断开连接避免循环更新
                self.slider_y.valueChanged.disconnect(self.update_y_text)
                self.slider_y.setValue(value)
                # 重新连接
                self.slider_y.valueChanged.connect(self.update_y_text)
        except ValueError:
            pass

    def on_send(self):
        x, y = self.slider_x.value()/10, self.slider_y.value()/10
        self.ser.send_pos(x, y)
        self.log(f'下发  X={x:.2f}  Y={y:.2f}')
        # 保存发送的位置
        self.save_position(x, y)

    def on_timer(self):
        data = self.ser.get_rx()
        if data:
            self.log('回读 ' + data.hex(' ').upper())

    # -------------------- 日志 --------------------
    def log(self, txt):
        t = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.txt_log.append(f'[{t}]  {txt}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWin()
    w.show()
    sys.exit(app.exec_())