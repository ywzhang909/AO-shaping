"""仿真光学台架中 DM 电压 → 瞳孔相位的耦合。

本模块为何存在
--------------
``SimPibSystem.far_field()`` 过去只累加 SLM 命令相位, DM 根本没有进入光学的通路。
``pib`` 与 ``combined`` runner 驱动的是 DM *电压*, 于是它们一路跑完并写出 CSV 文件,
而 DM 却什么都没影响 —— 目标函数纯粹是噪声。这不是"SPGD 不收敛", 而是"模型里没有 DM"。

影响力模型
----------
每个致动器贡献一个高斯相位凸包。真实 DM 的影响力函数与高斯很接近, 且因径向对称而在
x 与 y 上是*可分离*的。正是这种可分离性让实现变得可负担: 相位是在 DM 包围盒上一次
矩阵乘求出来的, 而不是 64 张整面板图 —— 后者在真实 1920x1200 面板尺寸下要 ~1.2 GB。

    phase(y, x) = sum_i  a_i * gy_i(y) * gx_i(x)

这是一个固定的线性算子, 因而对电压是线性的 —— 测试锁定的正是这条性质, 它也是所报
"响应"能有意义的原因。

对应的物理设备
---------------
本模型代替的是 **NLight 变形镜** (``drivers/dm/nlight/driver.py``), 即本台架的主 DM。
它的致动器数与电压上下限直接取自那个驱动自身的类属性, 而不在此重述, 以免仿真设备
悄悄地偏离它所镜像的硬件。

有一个参数**在本仓库里没有记载值**: NLight DM 的光程行程。因此 ``stroke_um`` 保持
显式传入, 默认取一个占位值, 并被报告为未标定, 而不是冒充厂商数据。凡是绝对相位有意义
的运行, 都请提供数据手册 (或实测) 的值。

单位
----
电压先映射为以微米计的光程差, 再映射为以弧度计的相位:

    opd_um    = stroke_um * v / v_max
    phase_rad = opd_um * 2*pi / (wavelength_nm * 1e-3)

零伏即平场, 与真实驱动一致 (那里 0 V 位于行程中点)。由于 NLight 的量程不对称
(``V_Min=-300``、``V_Max=499``), 负半程行程相应地比正半程短。

相位以 **raw 未包裹弧度** 返回, 且永不取 mod 2*pi: 按全项目的 raw-only 相位契约,
唯一的 wrap 点在 SLM 驱动里, 在此处包裹会破坏测试所断言的线性。
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.nlight import NLight

#: 本模型所镜像的物理设备。致动器数与电压上下限直接从中读取; 只有未记载的行程
#: 是这里的建模参数。
MIRRORED_DEVICE = "NLight"

DEFAULT_WAVELENGTH_NM = 532.0

#: NLight DM 的光程行程**在本仓库里没有记载值**。
#: 这个占位值只用来设定耦合强度; ``__init__`` 会把它记为未标定,
#: 以免被误当成数据手册上的数值。
UNCALIBRATED_STROKE_UM = 1.5

#: 致动器阵列跨越面板短轴的比例。
DEFAULT_SPAN_FRACTION = 0.6

#: 高斯 sigma 占致动器间距的比例。
DEFAULT_SIGMA_FRACTION = 0.5


class SimDmOptics:
    """把 NLight DM 的电压向量映射为瞳孔相位图。"""

    def __init__(
        self,
        n_actuators: int | None = None,
        slm_shape: tuple[int, int] = (1200, 1920),
        v_min: float | None = None,
        v_max: float | None = None,
        stroke_um: float = UNCALIBRATED_STROKE_UM,
        wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
        span_fraction: float = DEFAULT_SPAN_FRACTION,
        sigma_fraction: float = DEFAULT_SIGMA_FRACTION,
    ) -> None:
        # 默认取所镜像设备自己声明的规格, 以免本模型偏离它所代表的硬件。
        if n_actuators is None:
            n_actuators = NLight.DM_NUM
        if v_min is None:
            v_min = NLight.V_Min
        if v_max is None:
            v_max = NLight.V_Max
        if stroke_um == UNCALIBRATED_STROKE_UM:
            logger.warning(
                "{} DM stroke is undocumented in this repo; using placeholder "
                "stroke_um={} um. Treat absolute phase as uncalibrated.",
                MIRRORED_DEVICE,
                stroke_um,
            )
        side = int(round(np.sqrt(n_actuators)))
        if side * side != n_actuators:
            raise ValueError(
                f"n_actuators must be a perfect square for a grid layout, got {n_actuators}"
            )
        self.n_actuators = int(n_actuators)
        self.n_side = side
        self.slm_h, self.slm_w = int(slm_shape[0]), int(slm_shape[1])
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.stroke_um = float(stroke_um)
        self.wavelength_nm = float(wavelength_nm)

        span = DEFAULT_SPAN_FRACTION * min(self.slm_h, self.slm_w)
        self.pitch_px = span / (side - 1) if side > 1 else span
        self.sigma_px = max(self.pitch_px * sigma_fraction, 1.0)

        self.actuator_grid = self._build_grid(span)
        self._box = self._build_box()
        self._gy, self._gx = self._build_separable_basis()

        self._voltages = np.zeros(self.n_actuators, dtype=np.float64)
        self._opd = np.zeros(self.n_actuators, dtype=np.float64)
        self._phase = np.zeros((self.slm_h, self.slm_w), dtype=np.float64)
        self._dirty = True
        self._version = 0

    @property
    def version(self) -> int:
        """每次电压变化都会自增的单调计数器。

        ``SimPibSystem`` 会缓存远场, 所以若没有显式版本号, 缓存会一直供应 DM 作用
        之前的那张图, DM 耦合看起来就像毫无效果。
        """
        return self._version

    @property
    def influence_radius_px(self) -> float:
        """致动器凸包衰减到 ~2% 之外的半径。"""
        return 3.0 * self.sigma_px

    def _build_grid(self, span: float) -> np.ndarray:
        """致动器中心的 ``(行, 列)`` 像素坐标, 按网格行优先排列。"""
        half = span / 2.0
        cy, cx = self.slm_h / 2.0, self.slm_w / 2.0
        rows = np.rint(np.linspace(cy - half, cy + half, self.n_side)).astype(int)
        cols = np.rint(np.linspace(cx - half, cx + half, self.n_side)).astype(int)
        rows = np.clip(rows, 0, self.slm_h - 1)
        cols = np.clip(cols, 0, self.slm_w - 1)
        yy, xx = np.meshgrid(rows, cols, indexing="ij")
        return np.column_stack((yy.ravel(), xx.ravel()))

    def _build_box(self) -> tuple[int, int, int, int]:
        pad = int(np.ceil(self.influence_radius_px))
        r0 = max(int(self.actuator_grid[:, 0].min()) - pad, 0)
        r1 = min(int(self.actuator_grid[:, 0].max()) + pad + 1, self.slm_h)
        c0 = max(int(self.actuator_grid[:, 1].min()) - pad, 0)
        c1 = min(int(self.actuator_grid[:, 1].max()) + pad + 1, self.slm_w)
        return r0, r1, c0, c1

    def _build_separable_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """分解形式的影响力函数, 形状为 ``(n_actuators, box_len)``。"""
        r0, r1, c0, c1 = self._box
        ys = np.arange(r0, r1, dtype=np.float64)[:, None]
        xs = np.arange(c0, c1, dtype=np.float64)[:, None]
        cy = self.actuator_grid[:, 0][None, :]
        cx = self.actuator_grid[:, 1][None, :]
        two_sigma_sq = 2.0 * self.sigma_px**2
        gy = np.exp(-((ys - cy) ** 2) / two_sigma_sq).T
        gx = np.exp(-((xs - cx) ** 2) / two_sigma_sq).T
        return gy, gx

    def set_voltages(self, voltages: np.ndarray) -> None:
        """保存 DM 指令, 并裁剪到 DM 的电压量程。"""
        volts = np.asarray(voltages, dtype=np.float64).ravel()
        if volts.shape != (self.n_actuators,):
            raise ValueError(f"expected {self.n_actuators} voltages, got {volts.shape}")
        self._voltages = np.clip(volts, self.v_min, self.v_max)
        self._opd = self.stroke_um * self._voltages / self.v_max
        self._dirty = True
        self._version += 1

    @property
    def voltages(self) -> np.ndarray:
        return self._voltages.copy()

    @property
    def opd_um(self) -> np.ndarray:
        """逐致动器的光程差, 单位微米。"""
        return self._opd.copy()

    def phase(self) -> np.ndarray:
        """raw 未包裹弧度的瞳孔相位, 形状与 SLM 面板一致。"""
        if self._dirty:
            r0, r1, c0, c1 = self._box
            block = (self._gy * self._opd[:, None]).T @ self._gx
            self._phase.fill(0.0)
            self._phase[r0:r1, c0:c1] = block * (
                2.0 * np.pi / (self.wavelength_nm * 1e-3)
            )
            self._dirty = False
        return self._phase

    def reset(self) -> None:
        self._voltages = np.zeros(self.n_actuators, dtype=np.float64)
        self._opd = np.zeros(self.n_actuators, dtype=np.float64)
        self._dirty = True
        self._version += 1


__all__ = ["SimDmOptics"]
