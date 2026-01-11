# serial_port_fsm.py
import serial
import serial.tools.list_ports
import threading
import queue

class SerialPortFSM:
    MAX, MIN = 1510.0, -1510.0
    FRAME_LEN = 13

    def __init__(self, port=None, baud=115200):
        self.ser = serial.Serial(timeout=0.5)
        if port:
            self.ser.port = port
        else:
            available_port = self.list_port()
            assert available_port, "No available serial port found."
            self.ser.port = available_port[0]
        self.ser.baudrate = baud
        self.ser.bytesize = serial.EIGHTBITS
        self.ser.parity = serial.PARITY_NONE    
        self.ser.stopbits = serial.STOPBITS_TWO
        
        self.rx_que = queue.Queue()
        self.tx_que = queue.Queue()
        self.worker = None
        self.running = False

    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    # -------------------- 线程生命周期 --------------------
    def open(self):
        if self.ser.is_open:
            return True
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
    def send(self, x:float, y:float):
        frame = self._build_frame(x, y)
        self.ser.write(frame)
        return frame

    def send_in_queue(self, x: float, y: float):
        frame = self._build_frame(x, y)
        self.tx_que.put(frame)
        return frame

    def get_rx(self):
        """非阻塞弹一条最新回读"""
        return self.rx_que.get_nowait() if not self.rx_que.empty() else b''
    
    def wait_rx(self, timeout: float = 0.5):
        """阻塞等待一条回读"""
        return self.rx_que.get(timeout=timeout)

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
    def pack_position_xy(position_x: float, position_y: float) -> bytes:
        """
        与 C++ 代码 bit-wise 一致：
            int16_t xInt = (int16_t)std::round(PositionX / 0.05);
            写入高 8 位、低 8 位
        返回 4 字节 bytes：X高 X低 Y高 Y低
        """
        # 量化 + 四舍五入，与 C++ 保持相同溢出行为
        x_int = int(round(position_x / 0.05))
        y_int = int(round(position_y / 0.05))
        buf = bytearray(4)
        # 将 int16 转换为两个字节（大端序：高8位在前）
        buf[0] = (x_int >> 8) & 0xFF   # X轴高8位
        buf[1] = x_int & 0xFF          # X轴低8位
        buf[2] = (y_int >> 8) & 0xFF   # Y轴高8位
        buf[3] = y_int & 0xFF          # Y轴低8位
        return buf
    
    @staticmethod
    def _build_frame(x: float, y: float):
        buf = bytearray(13)
        buf[0] = 0x7E
        buf[1] = 0xE7
        buf[2] = 0x01
        # 限幅 + 0.05 量化
        x = max(min(x, SerialPortFSM.MAX), SerialPortFSM.MIN)
        y = max(min(y, SerialPortFSM.MAX), SerialPortFSM.MIN)
        # Pack position data in big-endian format to match C++ implementation
        buf[3:7] = SerialPortFSM.pack_position_xy(x, y)
        checksum = (~(sum(buf[2:12])) & 0xFF)
        buf[12] = checksum
        return buf
    
    @staticmethod
    def bin_frame_to_pos(frame: bytes) -> tuple[float, float]:
        """
        解析 13 字节帧，返回 (x, y) 坐标
        假设 frame包含 X 和 Y 的 int16 大端序编码
        """
        if len(frame) != 4:
            raise ValueError(f"Frame must be 4 bytes long")
        
        # 提取 X 和 Y 的 int16 编码（大端序）
        x_int = (frame[0] << 8) | frame[1]
        y_int = (frame[2] << 8) | frame[3]
        
        # 转换为浮点数（假设 0.05 分辨率）
        x = x_int * 0.05
        y = y_int * 0.05
        
        return x, y

    @staticmethod
    def list_port():
        return [p.device for p in serial.tools.list_ports.comports()]
    

def __run():
    with SerialPortFSM() as sp:
        sp.send_pos(80, 160)
        print(sp.get_rx())
    
def __test_frame_generation():
    """Test function to verify frame generation matches C++ implementation"""
    # Test case 1: Normal values
    frame = SerialPortFSM._build_frame(8.0, 16.0)
    print("Frame for (8.0, 16.0):", " ".join(f"{b:02X}" for b in frame))
    
    # Test case 2: Boundary values
    frame = SerialPortFSM._build_frame(SerialPortFSM.MAX, SerialPortFSM.MIN)
    print(f"Frame for ({SerialPortFSM.MAX}, {SerialPortFSM.MIN}):", " ".join(f"{b:02X}" for b in frame))
    
    # Test case 3: Values that require rounding
    frame = SerialPortFSM._build_frame(8.025, 16.025)  # Should round to 8.05 and 16.05
    print("Frame for (8.025, 16.025):", " ".join(f"{b:02X}" for b in frame))

if __name__ == '__main__':
    __run()
