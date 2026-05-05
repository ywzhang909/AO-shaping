import numpy as np
import time
from .base import DM

class SimulateDM(DM):
    channel: int = 64
    n_actuators: int = 64
    disabled_actuators: list[int] = []
    
    v_min, v_max = -300, 499

    def __init__(self, max_iter_diff=20, max_neibor_diff=0, keep_when_exit=True, noise_level=0.01):
        super().__init__()
        self.units_adj_mat = self._load_adj_txt()
        self.__last_v = np.zeros(self.channel)
        self.max_iter_diff = max_iter_diff
        self.max_neibor_diff = max_neibor_diff
        self.__keep_when_exit = keep_when_exit
        self.noise_level = noise_level
        self.hv_state = False
        self.deformation_history = []
        self.voltage_history = []
        # 模拟变形模型参数
        self.deformation_model = np.eye(self.channel) * 0.01

    def open(self) -> None:
        self.initialize()
        print("Simulated DM initialized successfully")

    def close(self) -> None:
        if not self.__keep_when_exit:
            self.reset_all()
            self.set_hv(False)
            print("Simulated DM turned off")
        print("Simulated DM connection closed")

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        cmd = np.clip(cmd, -1, 1)
        return (cmd + 1) * (self.v_max - self.v_min) / 2 + self.v_min

    def send(self, cmd):
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        raise ValueError("Unsupported command type. Expected numpy array of voltages.")

    def get_actuator_positions(self):
        # 生成模拟的致动器位置
        x = np.linspace(0, 10, int(np.sqrt(self.channel)))
        y = np.linspace(0, 10, int(np.sqrt(self.channel)))
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def initialize(self) -> None:
        self.set_hv(hv=True)
        self.reset_all()

    def reset_all(self):
        self.send_voltages(np.zeros(self.channel), 0.01)
        self.__last_v = np.zeros_like(self.__last_v)
        time.sleep(0.5)
        return 0

    def send_voltages(self, vs: np.ndarray, wait_time_s=0.001):
        vs = np.clip(vs, self.v_min, self.v_max)
        # 添加噪声模拟
        noisy_vs = vs + np.random.normal(0, self.noise_level, size=vs.shape)
        # 应用电压限制逻辑
        __gap = noisy_vs - self.__last_v
        if self.max_iter_diff > 0:
            _direction = np.sign(__gap)
            _abs_gap = np.abs(__gap)
            while _abs_gap.any():
                _step = np.minimum(_abs_gap, self.max_iter_diff)
                self.__last_v += _direction * _step
                _abs_gap -= _step
                time.sleep(wait_time_s / 10)
        else:
            self.__last_v = noisy_vs
        # 记录电压历史
        self.voltage_history.append(self.__last_v.copy())
        # 计算模拟变形量 (电压到变形的转换)
        deformation = self._voltage_to_deformation(self.__last_v)
        self.deformation_history.append(deformation)
        time.sleep(wait_time_s)
        return self.__last_v

    def set_hv(self, hv: bool = True):
        self.hv_state = hv
        time.sleep(0.5)
        return 0

    def get_hv_state(self):
        return self.hv_state

    def _voltage_to_deformation(self, voltages):
        # 简单的线性模型转换电压到变形量
        deformation = np.dot(self.deformation_model, voltages)
        # 添加变形噪声
        deformation += np.random.normal(0, self.noise_level * 0.1, size=deformation.shape)
        return deformation

    @staticmethod
    def _load_adj_txt():
        # 加载邻接矩阵或使用默认结构
        try:
            return np.loadtxt('data/dm_adj.txt')
        except FileNotFoundError:
            # 如果没有邻接矩阵文件，创建一个简单的网格结构
            size = int(np.sqrt(64))
            adj = np.zeros((64, 64), dtype=int)
            for i in range(64):
                row, col = i // size, i % size
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        j = nr * size + nc
                        adj[i, j] = 1
            return adj

    def get_deformation_history(self):
        return np.array(self.deformation_history)

    def get_voltage_history(self):
        return np.array(self.voltage_history)

    def clear_history(self):
        self.deformation_history = []
        self.voltage_history = []