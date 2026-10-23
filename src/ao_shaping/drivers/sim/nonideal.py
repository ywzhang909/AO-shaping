"""数字孪生的「器件级非理想物理」封装 (V2.0 升级路线)。

报告 ``report/shape_trials/sim_vs_hw/sim_vs_hw_20261022.md`` 第 7 节把当前孪生缺的
物理效应列成了 7 条; 下面的分析把其中的 5 条映射成可参数化的模型:

* **零级光 (η)** —— 真实 LCoS 瞳面是双分量 ``U = U_in·[η + (1−η)e^{iφ}]``;
  仿真的瞳面只有一路光 ⇒ 「框外功率占比」的来源构成不对 (实测含零级晕, 仿真只有
  边瓣与焦外弥散)。见 :class:`ZeroOrderModel`。
* **SLM LUT 非线性 + 灰度量化** —— 仿真原来假设理想的「弧度→灰度→远场」线性映射
  (``two_pi_gray = 993``), 没有 Gamma/LUT 曲线、没有量化误差 ⇒ CV 比实测低 ~40%。
  见 :class:`SlmLutModel`。
* **探测器非理想性 (PRNU / 暗电流 FPN)** —— 真实 sCMOS 有像素响应非均匀 (PRNU) 与
  固定模式噪声 (FPN); 仿真只有逐像元高斯白噪声 ⇒ 噪声地板的 σ(CV)/σ(复合分数)
  比实测小 60%+。见 :class:`CameraNonIdealModel`。

三个模型全部**默认 identity** (= 理想器件), 于是把旋钮置空时, 任何消费方的结果都与
改动前**逐位相同** —— 这是本仓库孪生旋钮的一贯纪律 (默认 = 既有行为)。

拟合入口
--------
每个模型都为「由一组标定实验反推参数」留好了位置:

* ``η`` 由一张**零级光斑** (未加相位) 的能量占比反推 (通常 5%~15%)。
* ``SlmLutModel`` 由一张实测 LUT 表 (灰度→弧度) 直接灌入; 没有表时由
  ``gamma`` 拟合 (对一组已知灰度测出的实际相位做幂律回归)。
* ``CameraNonIdealModel.prnu_sigma`` 由平场/散斑的**空间相关**残差标准差估计;
  ``dark_fpn_sigma`` 由暗帧的逐像元标准差估计。

本模块**纯 numpy**, import 期无任何副作用, 不 import 任何硬件驱动包, 也不触发
``matplotlib`` (与 ``slm_pib_sim`` 在纯净解释器里 import 的约束一致)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

#: 设备动态: 1064 nm 下 2π 弧度对应的灰度 (见 ``slm_pib_sim.TWO_PI_GRAY``)。
#: 这里独立定义一份, 避免与 ``slm_pib_sim`` 形成循环 import。
SLM_TWO_PI_GRAY = 993


@dataclass(frozen=True)
class SlmLutModel:
    """SLM 的 LUT 非线性 + 灰度量化往返模型。

    弧度→灰度的前向映射与 :meth:`SimSLMPib.create_phase_from_array` 共用同一套约定
    (``gray = round(mod(相位, 2π) / 2π · two_pi_gray)``), 灰度→相位的反查:

    * 给了 ``lut_table`` → 直接查表 (表长 = ``two_pi_gray + 1``);
    * 否则 → ``phase_max_rad · (gray / two_pi_gray) ** gamma``。

    ``gamma == 1`` 且无 ``lut_table`` 时, 往返**只引入灰度量化误差** (SLM 本就是
    2π 周期器件, 没有额外非线性); 这让仿真的 CV 自然上升到与实测接近的水平。

    Attributes:
        gamma: 灰度→相位的 Gamma 指数 (1 = 理想线性)。
        phase_max_rad: 满灰度对应的最大相位 (rad); 默认 2π。
        two_pi_gray: 2π 弧度对应的灰度 (设备动态值, 默认 993)。
        lut_table: 可选的实测 LUT (灰度→弧度), 长度 ``two_pi_gray + 1``; 给了就
            **优先于** ``gamma``。
    """

    gamma: float = 1.0
    phase_max_rad: float = 2.0 * math.pi
    two_pi_gray: int = SLM_TWO_PI_GRAY
    lut_table: np.ndarray | None = None

    def is_identity(self) -> bool:
        """是否只引入灰度量化、不引入额外非线性相位。

        ``True`` (γ=1 且无表) 时 ``actual_phase`` 仍会经过 2π/993 的灰度量化
        (与「完全不经 LUT」的 ``slm_lut=None`` 略有差别); 它只表示没有 LUT 非线性。
        ⚠️ 在孪生里「是否走 LUT 往返」由 ``slm_lut is not None`` 决定, **不是**由
        本方法决定 —— 显式给一个 γ=1 的 LUT 也会量化。
        """
        has_table = self.lut_table is not None
        return (not has_table) and abs(float(self.gamma) - 1.0) < 1e-12

    def actual_phase(self, phase_rad: np.ndarray) -> np.ndarray:
        """命令相位 → SLM 实际显示出的相位 (弧度), 含 LUT 非线性与灰度量化。

        Args:
            phase_rad: raw 弧度相位 (面板尺寸, 不必已 wrap)。

        Returns:
            实际相位 (与输入同形; 落在 ``[0, phase_max_rad)`` 或经 LUT 表取值)。

        Raises:
            ValueError: LUT 表长度与 ``two_pi_gray + 1`` 不符。
        """
        wrapped = np.mod(np.asarray(phase_rad, dtype=np.float64), 2.0 * np.pi)
        gmax = int(self.two_pi_gray)
        gray = np.round(wrapped / (2.0 * np.pi) * float(gmax)).astype(np.int64)
        gray = np.clip(gray, 0, gmax)
        table = self.lut_table
        if table is not None:
            table_arr = np.asarray(table, dtype=np.float64)
            if table_arr.shape[0] != gmax + 1:
                raise ValueError(
                    "lut_table must have length two_pi_gray + 1 "
                    f"({gmax + 1}), got {table_arr.shape[0]}"
                )
            return table_arr[gray]
        gamma = float(self.gamma)
        frac = gray.astype(np.float64) / float(gmax)
        return float(self.phase_max_rad) * (frac**gamma)


@dataclass(frozen=True)
class ZeroOrderModel:
    """SLM 瞳面的零级 (未调制) 光分量。

    真实 LCoS 存在填充因子与液晶响应不完全的问题, 会产生零级光 (``η`` 分量): 实际
    瞳面复振幅 ``U_slm = U_in·[η + (1−η)e^{iφ}]``。其中 ``√η·U_in`` 是**未调制**的
    常数项 ⇒ 它的傅里叶变换是照明孔径的 Airy 斑 (落在 0 级), 在远场上形成一片弥散
    本底/晕; ``√(1−η)·U_in·e^{iφ}`` 才是被 GS 相位调制的衍射级。

    仿真原来只有调制项 (``η = 0``) ⇒ 没有零级晕, 「框外功率占比」的来源构成不对。

    Attributes:
        eta: 零级光能量占比 (0~1); 0 = 理想单分量器件。
    """

    eta: float = 0.0

    def is_identity(self) -> bool:
        """是否不注入零级分量 (``eta <= 0`` ⇒ 与理想单分量逐位相同)。"""
        return float(self.eta) <= 0.0

    def pupil_weights(self) -> tuple[float, float]:
        """返回 ``(调制项权重, 零级项权重)`` = ``(√(1−η), √η)``。

        瞳面复振幅 = ``调制权重 · U_in·e^{iφ} + 零级权重 · U_in``。
        """
        e = float(self.eta)
        return math.sqrt(max(0.0, 1.0 - e)), math.sqrt(e)


@dataclass(frozen=True)
class CameraNonIdealModel:
    """探测器的空间相关非理想性: PRNU 与暗电流 FPN。

    仿真原来的噪声是逐像元高斯白噪声 + 8-bit 量化, 没有空间相关性。真实 sCMOS 有:

    * **PRNU** (像素响应非均匀): 一张固定的乘性噪声图
      ``M(x,y) ~ N(1, prnu_sigma²)``, 每帧乘以它 ⇒ 当光斑在视场内微小抖动或光强
      微调时, CV/复合分数的计算出现**系统性**波动 (白噪声模拟不出)。
    * **暗电流 FPN** (固定模式噪声): 暗电平不再是一个全局常数, 而是每像素一个固定
      偏置 ``D(x,y) = dark_level + N(0, dark_fpn_sigma²)`` ⇒ 固定模式噪声直接进
      CV 计算。

    Attributes:
        prnu_sigma: PRNU 乘性图的标准差 (1%~2% 量级); 0 = 无 PRNU。
        dark_fpn_sigma: 暗电平固定模式噪声的标准差 (counts); 0 = 暗电平是全局常数。
        seed: 生成固定 PRNU/暗场图的 RNG 种子 (固定 ⇒ 可复现)。
    """

    prnu_sigma: float = 0.0
    dark_fpn_sigma: float = 0.0
    seed: int = 0

    def is_identity(self) -> bool:
        """是否什么都不注入 (⇒ 与既有探测器模型逐位相同)。"""
        return float(self.prnu_sigma) <= 0.0 and float(self.dark_fpn_sigma) <= 0.0

    def prnu_map(self, shape: tuple[int, int]) -> np.ndarray:
        """固定乘性 PRNU 图 ``M ~ N(1, prnu_sigma²)`` (与输入同形)。"""
        sigma = float(self.prnu_sigma)
        if sigma <= 0.0:
            return np.ones(shape, dtype=np.float64)
        rng = np.random.default_rng(int(self.seed) + 0x51A3)
        return rng.normal(1.0, sigma, size=shape).astype(np.float64)

    def dark_map(self, shape: tuple[int, int], base_dark: float) -> np.ndarray:
        """固定暗场图 ``D(x,y) = base_dark + N(0, dark_fpn_sigma²)``。

        Args:
            shape: 帧形状 (远场窗口)。
            base_dark: 全局暗电平 (counts), 来自 ``SimNoiseSpec.dark_level_counts``。

        Returns:
            逐像元暗电平图 (与输入同形)。
        """
        sigma = float(self.dark_fpn_sigma)
        base = float(base_dark)
        if sigma <= 0.0:
            return np.full(shape, base, dtype=np.float64)
        rng = np.random.default_rng(int(self.seed) + 0xDA17)
        return (base + rng.normal(0.0, sigma, size=shape)).astype(np.float64)


def build_nonideal_bundle(
    *,
    zero_order_eta: float = 0.0,
    slm_lut: SlmLutModel | None = None,
    prnu_sigma: float = 0.0,
    dark_fpn_sigma: float = 0.0,
    dark_map_seed: int = 0,
    slm_lut_gamma: float | None = None,
    slm_lut_table: np.ndarray | None = None,
) -> dict[str, Any]:
    """把一组「标定实验反推出的参数」收成一个字典, 便于透传给 system 构造器。

    这是「由 ``sim_twin_params`` 文件读回一组非理想参数」的统一入口; 各采集脚本的
    ``configure_sim_system`` 与 ``compare_sim_vs_hw`` 都走它。

    Args:
        zero_order_eta: 零级光能量占比。
        slm_lut: 现成的 LUT 模型; 给了就优先。
        prnu_sigma: PRNU 乘性图标准差。
        dark_fpn_sigma: 暗电流 FPN 标准差。
        dark_map_seed: PRNU/暗场图种子。
        slm_lut_gamma: 没给 ``slm_lut`` 时, 用这个 gamma 现建一个 (None = 不建)。
        slm_lut_table: 没给 ``slm_lut`` 时, 用这个实测表现建一个 (None = 不建)。

    Returns:
        ``{"zero_order_eta": ..., "slm_lut": ..., "nonideal_camera": ...}``
        (未启用的部分给 ``None``/0, system 默认即理想器件)。
    """
    lut: SlmLutModel | None = slm_lut
    if lut is None and (slm_lut_gamma is not None or slm_lut_table is not None):
        lut = SlmLutModel(
            gamma=float(slm_lut_gamma) if slm_lut_gamma is not None else 1.0,
            lut_table=slm_lut_table,
        )
    camera = CameraNonIdealModel(
        prnu_sigma=float(prnu_sigma),
        dark_fpn_sigma=float(dark_fpn_sigma),
        seed=int(dark_map_seed),
    )
    return {
        "zero_order_eta": float(zero_order_eta),
        "slm_lut": lut,
        "nonideal_camera": camera if not camera.is_identity() else None,
    }


__all__ = [
    "SLM_TWO_PI_GRAY",
    "SlmLutModel",
    "ZeroOrderModel",
    "CameraNonIdealModel",
    "build_nonideal_bundle",
]
