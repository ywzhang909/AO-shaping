from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from ao_shaping.model.quantities import DmCommands


class BaseDM(ABC):
    """变形镜驱动的抽象基类, 定义所有 DM 的公共接口与通用实现。

    所有 DM 类型的统一接口:
    - 电压范围型 DM (NLight、MicroDM): send_voltages, 可选斜坡
    - 相位型 DM (ZernikeDM、HadamardDM): send 系数 → 相位图案

    安全模式 (默认开启): 启用时, send_voltages 会自动从当前电压斜坡到目标值,
    每步变化量不超过 max_neibor_diff。

    本类是 DM 驱动的唯一基类与实现载体; 所有驱动直接继承本类。
    """


    #: 致动器 (模式) 总数。基类只声明存在性, 值由各子类按硬件实际给出。
    #: 标注为 ``Any`` 而非 ``int``: 现有子类既有普通属性覆写 (MicroDM/NLight/
    #: SimMicroDM 的类常量), 也有 ``@property`` 覆写 (HadamardDM/ZernikeDM/
    #: SimulateDM 从生成器派生)。pyright 的覆写检查要求基类声明能同时接受这两种
    #: 形式, 任何具体类型都会在其中一个方向上报 ``reportIncompatibleVariableOverride``
    #: / ``reportIncompatibleMethodOverride``; ``Any`` 是唯一能让两者并存的声明。
    DM_NUM: Any

    # 电压范围 —— 子类按硬件上限覆写 (同理用 ``Any``, 原因见 ``DM_NUM``)
    V_Min: Any = float("-inf")
    V_Max: Any = float("inf")

    # 邻居电压安全步长 —— 适用时由子类覆写 (同理用 ``Any``)
    max_neibor_diff: Any = float("inf")

    #: ``DM_NUM`` 的向后兼容别名。**只声明, 不给类级默认值**: 值在 ``__init__`` 里
    #: 按 ``DM_NUM`` 回填成实例属性, 因此 ``hasattr(BaseDM, "DM_Num")`` 仍是 False
    #: —— 与旧的 ``@property`` 版本一样"未实例化就没有可用值", 不会用一个假的 ``0``
    #: 骗过 ``config._resolve_class_attribute`` 那类"类级读到非数值就实例化再读"的分派。
    #: 声明本身是给 pyright 的: 子类以普通属性 (``MicroDM.DM_Num: int = 1521``) 覆写时
    #: 不再与基类 ``@property`` 冲突 (``reportIncompatibleMethodOverride``)。
    DM_Num: Any

    def __init__(self, safety_mode: bool = True) -> None:
        """以可选的安全模式初始化 DM。

        Args:
            safety_mode: 为 True (默认) 时, send_voltages 会自动从当前电压斜坡到
                目标值, 每步变化量不超过 max_neibor_diff。设为 False 则直接施加电压。
        """
        self._safety_mode = safety_mode
        dm_num = getattr(self, "DM_NUM", getattr(self, "DM_Num", 0))
        # ``DM_Num`` 是普通类属性 (基类声明), 而基类 ``@property`` 与子类普通
        # 属性覆写互不兼容 (pyright)。这里在 ``__init__`` 回填, 保持
        # ``self.DM_Num == self.DM_NUM`` 的向后兼容别名语义。
        self.DM_Num = dm_num
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

    def transform(self, cmd: Any) -> npt.NDArray[np.floating]:
        """把归一化命令转换为设备专属的取值。

        默认实现就是电压型 DM 的标准映射: 委托 :meth:`transform_voltage`,
        把 ``[-1, 1]`` 线性映射到 ``[V_Min, V_Max]``。因此电压型子类
        (``MicroDM`` / ``R50Controller`` / ``AsyncMicroDM`` / ``NLight``)
        不需要再逐个转发一遍; 位置型/相位型子类 (``HadamardDM``、
        ``ZernikeDM``、仿真 DM) 必须覆写为各自的图案生成逻辑。
        """
        return self.transform_voltage(cmd)

    @abstractmethod
    def open(self) -> None:
        """打开与 DM 的连接。

        失败时必须抛出异常并携带具体信息 (设备标识、地址、底层错误),
        不得静默返回 ``False``。驱动专属异常 (如
        :class:`~ao_shaping.drivers.dm.micro.driver.MicroDMConnectionError`)
        或 ``ConnectionError`` 均可。
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭与 DM 的连接。"""
        ...

    @abstractmethod
    def get_actuator_positions(self) -> np.ndarray:
        """获取当前致动器取值。

        Returns:
            电压型 DM 子类返回软件回显 (可为 :class:`DmCommands`,
            它是 ``np.ndarray`` 的子类); 位置型 (如网格布局) 子类返回
            各自的 ndarray 布局。
        """
        ...

    # ---- 默认实现 ----

    def send(self, cmd: npt.ArrayLike) -> DmCommands:
        """向 DM 发送**电压**命令。数组形式委托给 send_voltages。

        本方法是电压型 DM 的下发入口, 不是"任意命令"的通用入口: 相位型 DM
        (``HadamardDM`` / ``ZernikeDM``) 不下发电压, 要用它们覆写的
        :meth:`transform` 拿灰度相位图。对相位型 DM 调用本方法会一路走到
        ``_apply_voltages`` 并抛 ``NotImplementedError`` (刻意让它响亮失败,
        而不是静默返回一张图案)。

        Returns:
            本次下发的电压命令 (:class:`DmCommands`)。它是软件回显, 不是
            硬件实测读数。
        """
        if isinstance(cmd, (np.ndarray, list, tuple)):
            return self.send_voltages(cmd)
        raise ValueError(f"Unsupported command type: {type(cmd)}")

    def is_connected(self) -> bool:
        """检查 DM 是否已连接。"""
        return False

    def get_hardware_info(self) -> dict[str, Any]:
        """获取硬件专属信息。"""
        return {
            "type": type(self).__name__,
            "DM_NUM": self.DM_NUM,
            "safety_mode": self._safety_mode,
        }

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

    def transform_voltage(self, cmd: npt.ArrayLike) -> npt.NDArray[np.floating]:
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
            # ``max_neibor_diff`` 在基类上标注为 ``Any`` (子类既有普通属性也有
            # ``@property`` 覆写, 见类体注释), 这里显式收成 ``float`` 以恢复
            # ``step_size`` 的窄化 —— 否则后续的比较与 ``np.isfinite`` 会被
            # 判为可能是 ``None``。运行时值本来就是数值, 转换无行为变化。
            step_size = float(self.max_neibor_diff)

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

    def send_voltages(
        self, vs: DmCommands | npt.ArrayLike, wait_time_s: float = 0.001
    ) -> DmCommands:
        """向所有通道发送电压数组, 可选安全斜坡。

        safety_mode 为 True (默认) 时, 电压会从当前状态斜坡到目标值,
        每步变化量不超过 max_neibor_diff。

        Args:
            vs: 所有逻辑通道的电压数组; 传 :class:`DmCommands` 时会先校验其
                致动器数量与电压范围是否与本 DM 匹配。
            wait_time_s: 发送后休眠时间 (硬件稳定时间)。

        Returns:
            本次下发的电压命令 (:class:`DmCommands`)。它是软件回显, 不是
            硬件实测读数。
        """
        if isinstance(vs, DmCommands):
            if (
                vs.n_actuators != self.DM_Num
                or vs.range_min < self.V_Min
                or vs.range_max > self.V_Max
            ):
                raise ValueError(
                    "DmCommands actuator count or voltage range does not match this DM"
                )
            values = vs.voltages
        else:
            values = np.asarray(vs, dtype=np.float64)
        result = self._ramp_voltages(np.clip(values, self.V_Min, self.V_Max))
        time.sleep(wait_time_s)
        return DmCommands(result, self.V_Min, self.V_Max, self.DM_Num)

    def set_channel_voltage(self, channel: int, voltage: float) -> bool:
        """设置单个通道的电压。

        Returns:
            成功返回 ``True``。契约统一为 ``bool``: 完整 DM 驱动失败时抛
            异常; 控制器级子类 (如 R50Controller) 失败时返回 ``False``。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_channel_voltage"
        )

    def set_all_voltage_by_arr(self, voltages: npt.ArrayLike) -> DmCommands:
        """按数组设置所有通道的电压。"""
        return self.send_voltages(voltages)

    def check_dm_unit_grad_safe(self, vs: np.ndarray) -> bool:
        """检查邻居电压差是否在安全限值内。"""
        return True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
