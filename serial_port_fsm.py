# serial_port_fsm.py
import struct, serial, serial.tools.list_ports, threading, queue

class SerialPortFSM:
    """与 Qt 解耦，普通线程跑收发；外部 push/pop 数据即可"""
    MAX, MIN = 1510.0, -1510.0
    FRAME_LEN = 13

    def __init__(self, port=None, baud=115200):
        self.ser = serial.Serial(timeout=0.5)
        self.ser.port = port
        self.ser.baudrate = baud
        self.rx_que = queue.Queue()
        self.tx_que = queue.Queue()
        self.worker = None
        self.running = False

    def __enter__(self):
        self.open()
        return self
    
    def __close__(self, exc_type, exc_val, exc_tb):
        self.close()
    # -------------------- 线程生命周期 --------------------
    def open(self):
        if self.ser.is_open: return True
        self.ser.open()
        self.running = True
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        return self.ser.is_open

    def close(self):
        self.running = False
        if self.worker: self.worker.join(timeout=1)
        self.ser.close()

    # -------------------- 业务 API --------------------
    def send_pos(self, x: float, y: float):
        """外部唯一需要调用的接口：发位置"""
        self.tx_que.put(self._build_frame(x, y))

    def get_rx(self):
        """非阻塞弹一条最新回读"""
        return self.rx_que.get_nowait() if not self.rx_que.empty() else b''

    # -------------------- 内部 --------------------
    def _loop(self):
        while self.running and self.ser.is_open:
            # 发
            while not self.tx_que.empty():
                self.ser.write(self.tx_que.get())
            # 收
            if self.ser.in_waiting:
                data = self.ser.read(self.ser.in_waiting)
                self.rx_que.put(data)

    @staticmethod
    def _build_frame(x: float, y: float):
        buf = bytearray(13)
        buf[0] = 0x7E
        buf[1] = 0xE7
        buf[2] = 0x01
        # 限幅 + 0.05 量化
        x = max(SerialPortFSM.MIN, min(SerialPortFSM.MAX, x))
        y = max(SerialPortFSM.MIN, min(SerialPortFSM.MAX, y))
        ix, iy = int(round(x / 0.05)), int(round(y / 0.05))
        buf[3:7] = struct.pack('>hh', ix, iy)   # 大端 2×int16
        checksum = (~(sum(buf[2:12])) & 0xFF)
        buf[12] = checksum
        return buf

    @staticmethod
    def list_port():
        return [p.device for p in serial.tools.list_ports.comports()]
    
    
if __name__ == '__main__':
    print(SerialPortFSM.list_port())