from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from ao_shaping.model.quantities import DmCommands


class DM(ABC):
    """变形镜驱动的抽象基类。

    所有 DM 类型的统一接口:
    - 电压范围型 DM (NLight、MicroDM): send_voltages, 可选斜坡
    - 相位型 DM (ZernikeDM、HadamardDM): send 系数 → 相位图案

    安全模式 (默认开启): 启用时, send_voltages 会自动从当前电压斜坡到目标值,
    每步变化量不超过 max_neibor_diff。
    """

    DM_NUM: int

    # 电压范围 —— 子类按硬件上限覆写
    V_Min: float = float("-inf")
    V_Max: float = float("inf")

    # 邻居电压安全步长 —— 适用时由子类覆写
    max_neibor_diff: float = float("inf")

    def __init__(self, safety_mode: bool = True) -> None:
        """以可选的安全模式初始化 DM。

        Args:
            safety_mode: 为 True (默认) 时, send_voltages 会自动从当前电压斜坡到
                目标值, 每步变化量不超过 max_neibor_diff。设为 False 则直接施加电压。
        """
        self._safety_mode = safety_mode
        dm_num = getattr(self, "DM_NUM", getattr(self, "DM_Num", 0))
        self._last_voltages: np.ndarray = np.zeros(dm_num)

    @property
    def default_dm_unit_mask(self) -> np.ndarray:
        """默认的活动致动器掩码 (True = 活动)。"""
        return np.ones(self.DM_NUM, dtype=bool)

    # ---- 抽象接口 ----

    @classmethod
    def is_reachable(cls) -> bool:
        """检查该 DM 类型的硬件在网络上是否可达。

        若至少有一台该类型的设备能联系上则返回 True。
        子类必须以类型专属的发现逻辑覆写本方法。
        """
        return False

    @abstractmethod
    def transform(self, cmd) -> np.ndarray:
        """把归一化命令转换为设备专属的取值。"""
        ...

    @abstractmethod
    def open(self) -> None:
        """打开与 DM 的连接。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭与 DM 的连接。"""
        ...

    @abstractmethod
    def get_actuator_positions(self) -> np.ndarray:
        """获取当前致动器取值。"""
        ...

    # ---- 默认实现 ----

    def send(self, cmd) -> np.ndarray:
        """向 DM 发送命令。数组形式委托给 send_voltages。"""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        raise ValueError(f"Unsupported command type: {type(cmd)}")

    def is_connected(self) -> bool:
        """检查 DM 是否已连接。"""
        return False

    def get_hardware_info(self) -> dict:
        """获取硬件专属信息。"""
        return {
            "type": type(self).__name__,
            "DM_NUM": self.DM_NUM,
            "safety_mode": self._safety_mode,
        }

    @property
    def DM_Num(self) -> int:
        """DM_NUM 的别名 (向后兼容)。"""
        return self.DM_NUM

    @property
    def min_voltage(self) -> float:
        """以 ``min_voltage`` 拼写暴露的电压下界。

        ``V_Min`` 是规范名称, 但 ``optimizer/combined_optimizer.py``
        会用 ``dm.min_voltage`` / ``dm.max_voltage`` 截断其更新量, 而它只在
        自己构造的 DM 上设置过这两个属性。因此注入 DM 的 runner (``combined``
        就是这样做的) 在每个驱动上都会撞上 ``AttributeError``, 硬件驱动也不例外。
        在这里做别名可让该边界无论 DM 如何构造都可用。
        """
        return self.V_Min

    @property
    def max_voltage(self) -> float:
        """以 ``max_voltage`` 拼写暴露的电压上界。参见 :attr:`min_voltage`。"""
        return self.V_Max

    # ---- 电压变换 (通用) ----

    def transform_voltage(self, cmd: np.ndarray) -> np.ndarray:
        """把归一化命令 [-1, 1] 转换为电压范围 [V_Min, V_Max]。

        这是电压范围型 DM 通用的电压变换。
        子类可覆写以实现自定义的变换逻辑。
        """
        cmd = np.clip(np.asarray(cmd, dtype=np.float64), -1.0, 1.0)
        return (cmd + 1.0) * (self.V_Max - self.V_Min) / 2.0 + self.V_Min

    # ---- 电压斜坡 (安全模式) ----

    def _ramp_voltages(
        self, target: np.ndarray, step_size: float | None = None
    ) -> np.ndarray:
        """从当前电压以受限步长斜坡到目标值。

        每一步中任何通道的变化量都不超过 step_size (默认为
        max_neibor_diff)。这可避免可能损坏 DM 的电压突跳。

        有硬件专属斜坡机制的子类 (例如 NLight 的 iter-diff)
        应覆写本方法。

        Args:
            target: 目标电压数组。
            step_size: 单步内每个通道的最大变化量。默认为 max_neibor_diff。

        Returns:
            最终施加的电压数组。
        """
        if step_size is None:
            step_size = self.max_neibor_diff

        if step_size <= 0 or not self._safety_mode:
            return self._apply_voltages(target)

        current = self._last_voltages.copy()

        # 步长为无穷 = 无需斜坡, 直接施加
        if not np.isfinite(step_size):
            result = self._apply_voltages(target)
            self._last_voltages = target.copy()
            return result

        while np.any(np.abs(target - current) > 1e-10):
            gap = target - current
            step = np.minimum(np.abs(gap), step_size)
            intermediate = current + np.sign(gap) * step
            self._apply_voltages(intermediate)
            current = intermediate

        self._last_voltages = target.copy()
        return self._last_voltages

    def _apply_voltages(self, vs: np.ndarray) -> np.ndarray:
        """向硬件施加电压。子类必须覆写。

        这是真正把电压发到设备上的底层方法, 由 _ramp_voltages 每步调用。

        Args:
            vs: 已截断的电压数组。

        Returns:
            施加后的电压数组。
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _apply_voltages"
        )

    # ---- 公开的发送接口 ----

    def send_voltages(self, vs: DmCommands | np.ndarray, wait_time_s: float = 0.001) -> DmCommands | np.ndarray:
        """向所有通道发送电压数组, 可选安全斜坡。

        safety_mode 为 True (默认) 时, 电压会从当前状态斜坡到目标值,
        每步变化量不超过 max_neibor_diff。

        Args:
            vs: 所有逻辑通道的电压数组。
            wait_time_s: 发送后休眠时间 (硬件稳定时间)。

        Returns:
            施加后的电压数组。
        """
        typed = isinstance(vs, DmCommands)
        if typed:
            if vs.n_actuators != self.DM_Num or vs.range_min < self.V_Min or vs.range_max > self.V_Max:
                raise ValueError("DmCommands actuator count or voltage range does not match this DM")
            values = vs.voltages
        else:
            values = np.asarray(vs, dtype=np.float64)
        result = self._ramp_voltages(np.clip(values, self.V_Min, self.V_Max))
        time.sleep(wait_time_s)
        if typed:
            return DmCommands(result, self.V_Min, self.V_Max, self.DM_Num)
        return result

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """设置单个通道的电压。"""
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_channel_voltage"
        )

    def set_all_voltage_by_arr(self, voltages: np.ndarray) -> None:
        """按数组设置所有通道的电压。"""
        self.send_voltages(voltages)

    def check_dm_unit_grad_safe(self, vs: np.ndarray) -> bool:
        """检查邻居电压差是否在安全限值内。"""
        return True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
