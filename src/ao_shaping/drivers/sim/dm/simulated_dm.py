"""用于无硬件测试与开发的模拟 DM。

本模块提供一个通用的模拟 DM, 通道数、噪声仿真与变形建模均可配置。

Attributes:
    v_min: 最小电压 (-300)。
    v_max: 最大电压 (499)。
"""

from typing import Any

import numpy as np

from ao_shaping.drivers.dm._adjacency import load_adjacency
from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.model.quantities import DmCommands


@register_dm("sim")
class SimulateDM(DM):
    """通用模拟变形镜。

    模拟一台致动器数量、电压上下限、噪声与变形建模均可配置的变形镜。便于在没有
    硬件的情况下测试优化算法。

    Attributes:
        channel: 通道数 (默认 64)。
        n_actuators: 致动器数 (默认 64)。
        disabled_actuators: 被禁用致动器的索引列表。
        v_min: 最小电压。
        v_max: 最大电压。
    """

    channel: int = 64
    n_actuators: int = 64
    disabled_actuators: list[int] = []

    #: 基类上的规范名称。下面的小写 ``v_min``/``v_max`` 曾是唯一定义的那两个, 于是
    #: 基类默认的 ``+/-inf`` 透了出来, 任何把指令裁剪到 DM 量程的调用都裁了个空。
    V_Min: float = -300.0
    V_Max: float = 499.0

    v_min: int = -300
    v_max: int = 499

    @property
    def DM_NUM(self) -> int:
        """致动器数, 拼写与 ``DM`` 基类一致。

        ``DM.DM_Num`` 会转发到 ``self.DM_NUM``; 缺了这个, 该类在每次访问 ``DM_Num``
        时都抛 ``AttributeError``。此前一直没被发现, 是因为那时该类型尚未注册 ——
        ``@register_dm("sim")`` 才让这个缺口变得可达。
        """
        return self.n_actuators

    def __init__(
        self,
        max_iter_diff: int = 20,
        max_neibor_diff: int = 0,
        keep_when_exit: bool = True,
        noise_level: float = 0.01,
    ):
        """初始化模拟 DM。

        Args:
            max_iter_diff: 每轮迭代允许的最大电压变化量。
            max_neibor_diff: 邻居之间允许的最大电压差。
            keep_when_exit: 退出时是否保留电压。
            noise_level: 电压仿真的噪声水平。
        """
        super().__init__()
        self.units_adj_mat = self._load_adj_txt()
        self.__last_v = np.zeros(self.channel)
        self.max_iter_diff = max_iter_diff
        self.max_neibor_diff = max_neibor_diff
        self.__keep_when_exit = keep_when_exit
        self.noise_level = noise_level
        self.hv_state = False
        self.deformation_history: list[np.ndarray] = []
        self.voltage_history: list[np.ndarray] = []
        # 模拟变形模型的参数
        self.deformation_model = np.eye(self.channel) * 0.01

    def open(self) -> None:
        """打开仿真连接并初始化。"""
        self.initialize()
        print("Simulated DM initialized successfully")

    def close(self) -> None:
        """关闭仿真连接。"""
        if not self.__keep_when_exit:
            self.reset_all()
            self.set_hv(False)
            print("Simulated DM turned off")
        print("Simulated DM connection closed")

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """把归一化指令变换到电压量程。"""
        cmd = np.clip(cmd, -1, 1)
        return (cmd + 1) * (self.v_max - self.v_min) / 2 + self.v_min

    def send(self, cmd: np.ndarray) -> np.ndarray:
        """向模拟 DM 发送指令。"""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        raise ValueError("Unsupported command type. Expected numpy array of voltages.")

    def get_actuator_positions(self) -> np.ndarray:
        """以网格布局返回模拟致动器位置。"""
        x = np.linspace(0, 10, int(np.sqrt(self.channel)))
        y = np.linspace(0, 10, int(np.sqrt(self.channel)))
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def initialize(self) -> None:
        """初始化 DM: 打开高压并把电压清零。"""
        self.set_hv(hv=True)
        self.reset_all()

    def reset_all(self) -> int:
        """把所有通道电压重置为零。"""
        self.send_voltages(np.zeros(self.channel), 0.01)
        self.__last_v = np.zeros_like(self.__last_v)
        return 0

    def send_voltages(self, vs: DmCommands | np.ndarray, wait_time_s: float = 0.001) -> DmCommands | np.ndarray:
        """向模拟 DM 发送电压, 含噪声与速率限制。

        Args:
            vs: 待发送的电压数组。
            wait_time_s: 每个电压步长的等待时间。

        Returns:
            仿真后的当前电压数组。
        """
        typed = isinstance(vs, DmCommands)
        if typed:
            if vs.n_actuators != self.channel or vs.range_min < self.v_min or vs.range_max > self.v_max:
                raise ValueError("DmCommands actuator count or voltage range does not match this DM")
            values = vs.voltages
        else:
            values = np.asarray(vs, dtype=np.float64)
        if values.shape != (self.channel,):
            raise ValueError(f"Expected {self.channel} voltages, got {values.shape}")
        vs = np.clip(values, self.v_min, self.v_max)
        # 加入噪声以模拟真实硬件
        noisy_vs = vs + np.random.normal(0, self.noise_level, size=vs.shape)
        # 施加电压速率限制逻辑
        __gap = noisy_vs - self.__last_v
        if self.max_iter_diff > 0:
            _direction = np.sign(__gap)
            _abs_gap = np.abs(__gap)
            while _abs_gap.any():
                _step = np.minimum(_abs_gap, self.max_iter_diff)
                self.__last_v += _direction * _step
                _abs_gap -= _step
        else:
            self.__last_v = noisy_vs

        # 记录电压历史
        self.voltage_history.append(self.__last_v.copy())
        # 计算模拟变形量 (电压 → 变形)
        deformation = self._voltage_to_deformation(self.__last_v)
        self.deformation_history.append(deformation)
        self._publish_to_optics(self.__last_v)
        if typed:
            return DmCommands(self.__last_v, self.v_min, self.v_max, self.channel)
        return self.__last_v

    def _publish_to_optics(self, voltages: np.ndarray) -> None:
        """把实际达到的电压推给共享光学模型的 DM 相位。

        缺了这一步, DM 对 ``SimPibSystem`` 就是不可见的: 它的 ``far_field``
        只累加 SLM 命令相位, 于是 DM 驱动的环路 (``pib``、``combined``)
        一路跑完, 而电压变化什么都没改变。

        这里做延迟 import, 因为 ``drivers/sim/__init__`` 是**先** import 本模块、
        **再** import ``slm_pib_sim``, 模块级 import 会解析到半初始化的包上。

        发布的是**实际达到**的电压而非请求值, 这样光学侧看到的就是驱动真正施加的、
        经过速率限制且带噪声的状态。
        """
        from loguru import logger

        from ao_shaping.drivers.sim.slm_pib_sim import get_system

        try:
            optics = get_system().dm_optics
        except (ImportError, AttributeError) as exc:
            logger.warning("sim DM not coupled into the optical model: {}", exc)
            return
        if optics.n_actuators != self.channel:
            logger.warning(
                "sim DM has {} actuators but the optical model expects {}; "
                "DM voltages will not reach the pupil",
                self.channel,
                optics.n_actuators,
            )
            return
        optics.set_voltages(voltages)
        self._publish_to_wfs(voltages)

    def _publish_to_wfs(self, voltages: np.ndarray) -> None:
        """在有活跃的模拟传感器时, 也把电压推给它。

        远场模型与 WFS 模型是两个各自独立的光学状态: 传感器测的是瞳面而非远场,
        因此只发布到 ``SimPibSystem`` 会让 ``wf``/``rms-zernike`` 一直读到平 pupil,
        无论 DM 怎么动都报零 RMS。

        没有模拟传感器时静默跳过 —— 硬件路径, 或只用远场模型时即是。
        """
        from ao_shaping.drivers.sim.wfs.simulated_wfs import get_active_sim_wfs

        sensor = get_active_sim_wfs()
        if sensor is None or sensor.dm_optics.n_actuators != self.channel:
            return
        sensor.dm_optics.set_voltages(voltages)

    def set_hv(self, hv: bool = True) -> int:
        """设置高压状态。"""
        self.hv_state = hv
        return 0

    def get_hv_state(self) -> bool:
        """获取当前高压状态。"""
        return self.hv_state

    def _voltage_to_deformation(self, voltages: np.ndarray) -> np.ndarray:
        """用线性模型把电压换算为变形量。

        Args:
            voltages: 输入电压数组。

        Returns:
            变形量数组。
        """
        deformation = np.dot(self.deformation_model, voltages)
        # 加入变形噪声
        deformation += np.random.normal(0, self.noise_level * 0.1, size=deformation.shape)
        return deformation

    @staticmethod
    def _load_adj_txt() -> np.ndarray:
        """邻接矩阵, 与真实 DM 驱动共用。

        取代了本加载器的一份逐字节重复实现 —— 那份去读一个相对于 CWD 的
        ``data/dm_adj.txt``, 并自带一套回退网格。
        """
        return load_adjacency()


    def get_deformation_history(self) -> np.ndarray:
        """以数组形式返回变形历史。

        Returns:
            随时间的变形值数组。
        """
        return np.array(self.deformation_history)

    def get_voltage_history(self) -> np.ndarray:
        """以数组形式返回电压历史。

        Returns:
            随时间的电压值数组。
        """
        return np.array(self.voltage_history)

    def clear_history(self) -> None:
        """清空变形历史与电压历史。"""
        self.deformation_history = []
        self.voltage_history = []
