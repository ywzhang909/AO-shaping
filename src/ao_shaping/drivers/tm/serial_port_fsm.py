# serial_port_fsm.py
import serial
import serial.tools.list_ports
import threading
import queue

from ao_shaping.utils import logger

class SerialPortFSM:
    MAX, MIN = 1510.0, -1510.0
    FRAME_LEN = 13

    def __init__(self, port=None, baud=2000000):
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

    # -------------------- 数据验证相关方法 --------------------
    def validate_received_data(self, data: bytes) -> tuple[bool, int]:
        """
        验证接收到的数据的完整性
        对应C++: bool SerialPortFSM::validateReceivedData(const QByteArray &data, quint8 &calculatedChecksum)
        
        Args:
            data: 接收到的字节数组
            
        Returns:
            Tuple[bool, int]: (是否有效, 计算出的校验和)
        """
        # 第一步：校验数据长度（原逻辑：至少10字节）
        if len(data) < 10:
            print(f"数据长度不足，校验失败（实际{len(data)}字节，预期≥10字节）")
            return False, 0

        # 第二步：计算校验和（原逻辑：索引2~8累加后取反）
        calculated_checksum = 0
        for i in range(2, 9):  # 索引2到8（包含）
            calculated_checksum += data[i]
        calculated_checksum = (~calculated_checksum) & 0xFF

        # 第三步：比对校验和（接收的校验位在索引9）
        received_checksum = data[9]
        if calculated_checksum != received_checksum:
            logger.error(f"校验和不匹配：计算值0x{calculated_checksum:02X}，接收值0x{received_checksum:02X}")
            return False, calculated_checksum

        return True, calculated_checksum

    def parse_position_data(self, data: bytes) -> tuple[bool, float, float]:
        """
        将原始字节解析为X/Y坐标
        对应C++: bool SerialPortFSM::parsePositionData(const QByteArray &data, float &x, float &y)
        
        Args:
            data: 包含位置信息的字节数组
            
        Returns:
            Tuple[bool, float, float]: (是否成功解析, X坐标, Y坐标)
        """
        # 检查数据长度是否足够
        if len(data) < 7:  # 至少需要7个字节来获取坐标信息（索引3-6）
            logger.error(f"数据长度不足，无法解析位置信息（实际{len(data)}字节，预期≥7字节）")
            return False, 0.0, 0.0

        # 解析X坐标（原逻辑：索引3=高8位，索引4=低8位，乘以0.05）
        actual_position_x_hex = (data[3] << 8) | data[4]
        # 将无符号整数转换为有符号整数（处理负数情况）
        if actual_position_x_hex >= 32768:  # 0x8000
            actual_position_x_hex -= 65536  # 转换为-32768到32767的范围
        x = float(actual_position_x_hex) * 0.05

        # 解析Y坐标（原逻辑：索引5=高8位，索引6=低8位，乘以0.05）
        actual_position_y_hex = (data[5] << 8) | data[6]
        # 将无符号整数转换为有符号整数（处理负数情况）
        if actual_position_y_hex >= 32768:  # 0x8000
            actual_position_y_hex -= 65536  # 转换为-32768到32767的范围
        y = float(actual_position_y_hex) * 0.05

        # 可选：校验坐标是否在合理范围（避免异常值）
        if x < self.MIN or x > self.MAX or y < self.MIN or y > self.MAX:
            logger.error(f"解析的位置数据超出范围：X={x}，Y={y}（范围{self.MIN}~{self.MAX}）")
            return False, x, y

        return True, x, y

    # -------------------- 线程生命周期 --------------------
    def open(self):
        if self.ser.is_open:
            return True
        self.ser.open()
        self.running = True
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        logger.info(f"串口 {self.ser.port} 已打开，波特率 {self.ser.baudrate}")
        return self.ser.is_open

    def close(self):
        self.running = False
        if self.worker:
            self.worker.join(timeout=1)
        self.ser.close()
        logger.info(f"串口 {self.ser.port} 已关闭")

    # -------------------- 业务 API --------------------
    def send(self, x:float, y:float):
        frame = self._build_frame(x, y)
        self.ser.write(frame)
        logger.info(f"发送位置命令：X={x:.3f}, Y={y:.3f}")
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
    def _pack_position_xy(position_x: float, position_y: float) -> bytes:
        """
        与 C++ 代码 bit-wise 一致：
            int16_t xInt = (int16_t)std::round(PositionX / 0.05);
            写入高 8 位、低 8 位
        返回 4 字节 bytes：X高 X低 Y高 Y低
        """
        # 量化 + 四舍五入，与 C++ 保持相同溢出行为
        x_int = int(round(position_x / 0.05))
        y_int = int(round(position_y / 0.05))
        # 确保值在int16范围内 (-32768 到 32767)
        x_int = max(-32768, min(32767, x_int))
        y_int = max(-32768, min(32767, y_int))
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
        buf[3:7] = SerialPortFSM._pack_position_xy(x, y)
        checksum = (~(sum(buf[2:12])) & 0xFF)
        buf[12] = checksum
        return buf

    @staticmethod
    def list_port():
        return [p.device for p in serial.tools.list_ports.comports()]
