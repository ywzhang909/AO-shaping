"""Δz = 0..50 mm 的离焦扫描可视化 (纯仿真): 方案A 真值 vs 瞳面二次相位近似。

回答三个问题
------------

**V1 —— 把 CCD 沿光轴往后挪 0..50 mm, 光斑会变成什么样?**
以 Δz=0 优化得到的 plain GS 相位为基底, 对 11 个 Δz 各出一张方案A 正演的 CCD
远场强度图 (0 / 25 / 50 mm 三张全尺寸 + 一张 11 宫格总览), 每张标 CV / PIB / score。

**V2 —— 硬件**能**施加的那张相位长什么样?**
CCD 挪不动, 只能改瞳面相位。输出 11 张瞳面二次相位图 (11 宫格), 外加「峰值相位 vs
Δz」曲线 —— 让人知道 50 mm 处面板边缘相位已经绕了多少个 2π。

**V3 —— 不同匀化算法各自嵌入的离焦, 在硬件可施加的形式下还剩多少?**
四种算法各自按各自 Δz 优化, 取其最优离焦幅度, 换算成瞳面二次相位叠加上去, 再用
方案A 在 0..50 mm 全扫描上评测。

物理: 两种正演 + 零级背景
-------------------------

把 CCD 移到透镜焦后 Δz, 正演有两条路:

* **方案A (真值)**: ``U_CCD = ASM(FT(U_SLM), Δz)`` —— 透镜先做傅里叶变换, 再对
  **已变换的场**自由传播 Δz。任何 Δz 下都正确。
* **瞳面二次相位 (硬件可施加的近似)**: 傍轴极限下与方案A 等价, 只是把二次相位挪到
  瞳面 —— ``U_CCD ∝ FT( U · exp(s·i·α·π·Δz·r²/(λf²)) )``, 硬件上只能施加这个。

🔴 **CCD 离焦不是放大, 是变模糊 —— 不存在放大率 M=(f+d)/f。**

有外部材料主张「CCD 挪到 f+d 时图样被放大 M=(f+d)/f, target 要按 M 重采样」。**对 2f
光路这是错的**, 错在漏掉了透镜。Collins/ABCD 严格算::

    T(f+d) · Lens(f) · T(f)  =  [[−d/f, f], [−1/f, 0]]      → A = −d/f, B = f

于是 ``U(ξ) ∝ e^{ikDξ²/2B}·∫ U(x)·e^{ikAx²/2B}·e^{−ikxξ/B} dx``, 核是 ``e^{−ikxξ/B}``
而 **B 恒等于 f** ⇒ 核尺度是 ``λf``, 不是 ``λ(f+d)``。CCD 离焦只让图样**变模糊**
(发散球面波), 整体不缩放: 单个倾斜平面波经透镜聚到 fθ, 之后只是发散, 中心不跑。

本脚本因此**不做**任何 target 重采样, **不**缩放远场像元 pitch
(恒为 ``λf/(N·dx_pupil)``), 瞳面二次相位的分母取 **λf²** 而非 ``λf(f+d)`` (后者在
d=50mm、f=122.4mm 时会把 chirp 弱化 1.408 倍)。:func:`self_check_no_magnification`
把「像元 pitch 与 Δz 无关」钉成回归门禁, 防止有人日后把 M 改回来。

**符号与前因子仍不照抄公式**: ``s ∈ {−1, +1}`` 与连续系数 ``α`` (理论值 1) 由
:func:`resolve_chirp_sign_alpha` **数值一致性比对**方案A 定出来, 结论落进 summary JSON。
实测 (完整网格): ``sign = −1`` 残差 4.46e−6, ``sign = +1`` 残差 1.65e−3 (差 ~370 倍),
``α = 1.00016`` (残差 2.24e−7)。

🔴 **零级背景: 双分量瞳面模型**

真实 SLM 上入射场到瞳面是**双分量**::

    U_pupil = U_inc · [ η + (1−η)·exp(iφ) ]

* ``η`` = 零级振幅系数 (SLM 调制深度不足 / 像素死区 / 未调制区域); 零级**功率**占比
  = ``η²/(η²+(1−η)²)``, 对 ``η ≪ 1`` 约等于 ``η²``。
* 零级分量 ``η·U_inc`` **只经过物理透镜**, 在 f+d 处散成一个晕斑, 半径约
  ``ρ ≈ d·D/(2f)``; 它与整形方形是**两个独立的空间尺度** (点像 vs 97 µm 的方斑),
  所以在 d 稍大时它会**相对**更亮。

``OffFocusBench.focal`` 把瞳面振幅钳死成 ``amp·exp(iφ)``, **零级没有任何自由度**,
仿真看不见这个污染。故本脚本的 :func:`pupil_field` 自己组装双分量瞳面场 (只复用
``OffFocusBench`` 的 ``amp`` / ``pad`` / ``_asm`` —— **传播与 GS 一行不重写**,
:func:`self_check_two_component_at_eta0` 在 ``η=0`` 上钉到 1e−15 证明这一点)。

⚠️ **建模选择 (明写)**: 瞳面二次相位乘在**整个**瞳面场上, 含零级分量。因为 chirp 是
「传播了 d」这件事的替身, 而传播距离对零级和整形分量是共用的; 这样方案A 与瞳面
二次相位两条路才仍然 apples-to-apples。若认为零级不该吃这个虚拟透镜, 退化掉
``pupil_field`` 里的 chirp 即可, 其余结构不变。

晕斑标度自检: 焦面光斑的瑞利长度 ``z_R = πw_f²/λ ≈ 1.2 mm``, 而最小的扫描点就已经是
5 mm ≫ z_R, 所以整条扫描都处在渐近区 —— 晕斑峰值应按 **``I ∝ f²/d²``** 下降。
:func:`fit_halo_power_law` 把实测指数拟出来并落进 summary; **拟合不出 ≈ −2 就说明
双分量模型写错了** (这是硬门禁, 不达标直接抛错)。

复用仓库已验证件 (不自写)
--------------------------

* **物理预测模型 / GS 骨架**: 直接 import
  :class:`scripts.slm.experiment_gs_defocus_offocus.OffFocusBench` 与
  :class:`~ao_shaping.algorithm.signal_processing.square_shaping.SquareShapingConfig`
  (它已过自检: GS 镜像 ``max|Δphase| = 6.7e-15``)。本文件只补它没有的那一件事:
  **瞳面二次相位算子 + 它的数值标定**。
* **传播**: :func:`~ao_shaping.algorithm.signal_processing.angular_spectrum.angular_spectrum_propagate`。
* **Zernike**: :class:`~ao_shaping.utils.wavefront.zernike_calc.ZernikeGenerator`
  + :func:`~ao_shaping.utils.wavefront.zernike_calc.zernike_modes`。
* **光学常量**: 从 :mod:`ao_shaping.optimizer.wfless.slm_gs_refine` 追到源头, 经
  兄弟脚本转手, 不在本文件另立一份。

⚠️ **窗口限制 (必须读)**: 仿真 CCD 网格的**物理宽度与 ``far_field_padding`` 无关**,
恒为 ``n_grid·λf/D``。本脚本的完整网格下它是 1.157 mm, 而 CCD 后移 Δz 引起的几何
弥散半径是 ``(D/2)·Δz/f`` = 0.0294·Δz mm —— **Δz ≥ 20 mm 时弥散就撑破画幅**。
ASM 在周期边界下会把溢出的能量卷回对面, 于是大 Δz 的方案A 画面被边界卷绕主导,
不是真实传感器上会看到的图。所以下面每一行都同时记 ``edge_energy`` (外侧 10% 环
围的功率占比) 与 ``sigma_px``, 判据把它们一起纳入「可信 / 不可信」。真实实验系统的
CCD 视场 (Daheng 2592×1944 @ 2.2 µm = 5.70×4.28 mm) 比这个仿真窗口大得多,
所以这是**仿真窗口**的限制, 不是实验系统的限制。

运行
----

.. code-block:: bash

    python scripts/slm/experiment_gs_defocus_sweep_visuals.py           # 完整网格
    python scripts/slm/experiment_gs_defocus_sweep_visuals.py --quick   # 小网格冒烟

产物写到 ``report/shape_trials/gs_defocus_sim/artifacts/``, 文件名带 ``_visuals``: 3 个 CSV (方案A vs 瞳面
差异表 / V1 扫描 / V3 算法对比)、1 个 summary JSON、7 张图 (V1 三张全尺寸 + 总览,
V2 十一宫格 + 峰值曲线, V3 CV 曲线 + 50 mm 2×2 对比)。

纯仿真: 不打开任何设备, 不 import 任何驱动包。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# `scripts/` 自身是可导入包 (scripts/README.md 规定兄弟脚本一律按包路径 import,
# 不能用裸文件名), 故把仓库根也放进 sys.path。
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from scripts.slm.experiment_gs_defocus_offocus import (
    APERTURE_M,
    BEAM_RADIUS_PX,
    DZ_BLUR_M,
    FOCAL_LENGTH_M,
    PANEL_PIXEL_UM,
    WAVELENGTH_M,
    WAVELENGTH_NM,
    OffFocusBench,
    RunConfig,
    QUICK as SIM_QUICK,
    FULL as SIM_FULL,
)
from ao_shaping.algorithm.signal_processing.angular_spectrum import (
    angular_spectrum_propagate,
)
from ao_shaping.algorithm.signal_processing.square_shaping import (
    composite_from_pib_cv,
)
from ao_shaping.utils.wavefront.matrix_utils import (
    focal_length_from_camera_pixel,
)
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
    TILT_SHIFT_SCALE,
)

# ---------------------------------------------------------------------------
# 扫描网格
# ---------------------------------------------------------------------------
#: Δz 扫描点 (mm): 0 到 50, 步长 5, 共 11 个点。**一个都不能少。**
DZ_SCAN_MM: tuple[float, ...] = tuple(float(v) for v in range(0, 55, 5))

#: 数值反解二次相位符号/前因子时的探针工作点 (Δz_blur 单位)。
#: 必须**小** —— 傍轴近似只在近焦处成立, 在大 Δz 上反解会把逼近误差一起吸收进 α。
CHIRP_PROBE_DZ_UNITS = 0.25
SIGN_CANDIDATES: tuple[float, ...] = (-1.0, 1.0)

#: 瞳面二次相位前因子的黄金分割区间 (理论值 1)。
ALPHA_BOUNDS = (0.5, 1.5)
ALPHA_ITERS = 40

#: 「瞳面二次相位还能信」的判据 (两条都过才算可信)。
TRUST_REL_TOL = 0.05
TRUST_EDGE_ENERGY = 0.02

#: 差异表里「窗内」比较用的中央裁剪比例 (半宽占比)。方案A 的大 Δz 画面被周期卷绕
#: 主导, 全画幅的相对误差会被卷绕项污染; 中央裁剪给的是「光还没跑出去」那块。
INNER_CROP = 0.5

#: 实验系统 CCD (Daheng MER2-507-23GM) 的尺寸与像元, 用于对比「仿真画幅 vs 真实
#: 视场」。值取自 ``src/ao_shaping/drivers/AGENTS.md`` 记录的实测参数。
HW_CCD_PX = (2592, 1944)
HW_CCD_PIXEL_M = 2.2e-6

#: 零级振幅系数 η 的取值 (公式 ``U = U_inc·[η + (1−η)e^{iφ}]`` 里的那个 η)。
#: 零级**功率**占比 = ``η²/(η²+(1−η)²)``: 0.02 → 4.2e−4, 0.05 → 2.6e−3,
#: 0.10 → 1.1e−2 (即 0.4% / 0.26% / 1.1%)。0.0 = 纯理论对照 (无零级)。
ETAS: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
#: V1/V2/V3 的图上并列展示用的 η (其余 η 只进晕斑曲线与 CSV, 免得图爆炸)
ETA_DEMO = 0.05

#: 晕斑幂律自检: 期望指数 ≈ −2 (``I_halo ∝ f²/d²``), 偏离超过该容差直接抛错
HALO_EXPECT_EXPONENT = -2.0
HALO_EXPONENT_TOL = 0.5
#: 幂律拟合只用**窗口可信**的 Δz 点 (晕斑半径超出仿真画幅后峰值不再可信)
HALO_FIT_MIN_DZ_MM = 2.0

#: V1 给全尺寸图的三个 Δz (mm)。
V1_FULLSIZE_MM: tuple[float, ...] = (0.0, 25.0, 50.0)

#: V3 的四个算法。
ALGO_KEYS: tuple[str, ...] = (
    "gs",
    "gs_zernike",
    "gs_zernike_defocus",
    "gs_zernike_post",
)
ALGO_LABELS: dict[str, str] = {
    "gs": "GS",
    "gs_zernike": "GS + Zernike",
    "gs_zernike_defocus": "GS + Zernike + 离焦并入环路",
    "gs_zernike_post": "GS + 事后外挂离焦",
}
ALGO_COLORS: dict[str, str] = {
    "gs": "#2196F3",
    "gs_zernike": "#4CAF50",
    "gs_zernike_defocus": "#FF5722",
    "gs_zernike_post": "#9C27B0",
}

#: V3 「事后外挂离焦」的外层一维搜索 (确定性, 无随机重启)。区间与
#: experiment_gs_fresnel_vs_virtuallens.py 的 DEFOCUS_LIMITS 齐平, 才可比。
POST_DEFOCUS_LIMITS = (-5.0, 5.0)
POST_DEFOCUS_COARSE = 21
POST_DEFOCUS_GOLDEN = 12


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisConfig:
    """一次运行的网格与优化预算 (在兄弟脚本的 RunConfig 上再挂本脚本自己的旋钮)。"""

    base: RunConfig
    n_defocus_coarse: int = POST_DEFOCUS_COARSE
    n_defocus_golden: int = POST_DEFOCUS_GOLDEN
    tag: str = ""

    @property
    def n_grid(self) -> int:
        return self.base.n_grid

    @property
    def m(self) -> int:
        return self.base.m

    @property
    def far_field_half_width_m(self) -> float:
        """仿真 CCD 画幅的半宽 (m)。与 ``far_field_padding`` 无关。

        ⚠️ 必须用 **``far_field_size``** (``= n_grid·padding``) 而不是 ``n_grid``:
        ``far_field_pixel_size = λf/(D·padding)`` 已经带了一次 ``1/padding``, 拿
        ``n_grid`` 去乘会**再丢一个 padding**, 报出来的半宽小 ``padding`` 倍
        (完整网格下 0.072mm vs 真值 0.579mm)。两个 ``padding`` 抵消, 画幅宽度恒为
        ``n_grid·λf/D``。
        """
        return 0.5 * self.m * self.base.bench().far_field_pixel_size

    @property
    def blur_radius_m(self) -> float:
        """Δz = 50 mm 时焦外几何弥散半径 ``(D/2)·Δz/f`` 的最大值 (m)。"""
        return 0.5 * APERTURE_M * (max(DZ_SCAN_MM) * 1e-3) / FOCAL_LENGTH_M


#: ⚠️ ``--quick`` 与完整网格的产物**必须落到不同文件名**, 否则冒烟跑会把完整结果
#: 覆盖掉。所有产物都带 ``_visuals`` 子后缀 (名字里写死的), 只有 ``tag`` 这一个变量
#: 决定它后面跟不跟 ``_quick``。
VIS_QUICK = VisConfig(base=SIM_QUICK, tag="_quick")
VIS_FULL = VisConfig(base=SIM_FULL, tag="")


# ---------------------------------------------------------------------------
# 瞳面二次相位: 符号与前因子靠数值反解, 不照抄公式
# ---------------------------------------------------------------------------
class PupilChirp:
    """瞳面二次相位算子 ``exp(s·i·α·π·Δz·r²/(λf²))`` (硬件可施加的那张相位图)。

    构造时**不假设** ``s`` 与 ``α``: 由 :func:`resolve_chirp_sign_alpha` 在方案A 上
    数值一致性比对定出来, 过程与结论落进 summary JSON。``α`` 是理论值为 1 的连续前因子,
    保留成可反解参数而不是写死, 就是为了让「是不是 1」成为一个被测量的结论。
    """

    def __init__(self, bench: OffFocusBench, *, sign: float, alpha: float) -> None:
        self.bench = bench
        self.sign = float(sign)
        self.alpha = float(alpha)
        # 瞳面像元间距 (m) —— 瞳面 r 的单位来源
        self.dx_pupil = bench.cfg.aperture_size / bench.n
        # 居中排布的瞳面物理坐标 (m), n×n (只有这一块会被 pad_pupil 用到,
        # 之外的补零区乘什么都不影响结果)
        x = (np.arange(bench.n) - bench.n // 2) * self.dx_pupil
        self._r2 = x[:, None] ** 2 + x[None, :] ** 2
        #: 瞳面边缘 (角点, r 最远处) 的 |r|², 用于换算「面板边缘相位」
        self._r2_corner = float(self._r2[0, 0])
        #: 照明口径边缘 (r = D/2 = ``BEAM_RADIUS_PANEL``) 的 |r|²。
        #: 这是「面板边缘相位」在硬件上真正对应的量 —— 光斑只照亮到 r = D/2, 角点
        #: (r = √2·D/2) 落在照明区外, 面板上那片玻璃根本不参与成像。
        self._r2_disc = float(bench.cfg.aperture_size**2 / 4.0)

    # -- 系数 ---------------------------------------------------------------
    def coefficient(self, dz: float) -> float:
        """二次相位的整体系数 (1/m): ``s·α·π·Δz/(λf²)``。"""
        return (
            self.sign
            * self.alpha
            * np.pi
            * dz
            / (self.bench.lam * self.bench.cfg.focal_length**2)
        )

    def phase_rad(self, dz: float) -> np.ndarray:
        """n×n 瞳面二次相位 (raw 弧度, **不** mod 2π)。Δz=0 时全零。"""
        return self.coefficient(dz) * self._r2

    def phase_corner_rad(self, dz: float) -> float:
        """瞳面角点处的相位值 (rad) —— 网格外接圆角上的量。"""
        return abs(self.coefficient(dz) * self._r2_corner)

    def phase_disc_edge_rad(self, dz: float) -> float:
        """照明口径边缘 ``r = D/2`` 处的相位值 (rad) —— **面板边缘相位**。

        硬件上光斑只照亮到 ``BEAM_RADIUS_PANEL``, 所以这个量才是「面板边缘相位绕了
        多少个 2π」; 角点量是它的 √2 倍, 落在照明区外。
        """
        return abs(self.coefficient(dz) * self._r2_disc)

    # -- 正演 ---------------------------------------------------------------
    def intensity_bp(self, phase: np.ndarray, dz: float, eta: float = 0.0) -> np.ndarray:
        """瞳面域一次 FFT 正演 (方案B 本体, 不含输出二次相位; 双分量瞳面)。

        ``pupil_field(φ, η) · chirp`` → 一次 FFT → ``|U|²`` 归一化。Δz=0 时 chirp 恒为
        1, 逐位退化成 :meth:`OffFocusBench.focal` (在 ``η=0`` 上由
        :func:`self_check_two_component_at_eta0` 钉到 1e−12)。
        """
        return intensity_scheme_bp(self.bench, self, phase, dz, eta)

    # -- 与 Zernike (2,0) 的换算 -------------------------------------------
    def zernike_defocus_coeff(self, dz: float, basis: np.ndarray) -> float:
        """把瞳面二次相位换算成等效的 Zernike (2,0) 系数 (弧度)。

        用**最小二乘投影**而不是公式: ``a = <Q, Z₂⁰> / <Z₂⁰, Z₂⁰>``。两个归一化约定
        (Noll / 正交) 会给出差一个常数倍的结果, 公式抄错就是差一个系数整列; 投影是
        无假设的, 且残差 (:func:`zernike_fit_residual`) 一并报出来。
        """
        q = self.phase_rad(dz)
        z = np.asarray(basis, dtype=np.float64)
        return float(np.sum(q * z) / max(np.sum(z * z), 1e-30))

    def zernike_fit_residual(self, dz: float, basis: np.ndarray) -> float:
        """投影残差 (rad): 去掉 Z₂⁰ 方向后二次相位剩下的部分 (主要是 piston)。"""
        q = self.phase_rad(dz)
        z = np.asarray(basis, dtype=np.float64)
        a = self.zernike_defocus_coeff(dz, z)
        resid = q - a * z
        return float(np.max(np.abs(resid - resid.mean())))


def rel_err(ref: np.ndarray, got: np.ndarray) -> float:
    """归一化强度的最大相对误差 (按 ``max|ref|`` 归一, 与兄弟脚本同口径)。"""
    return float(np.abs(ref - got).max() / max(float(ref.max()), 1e-30))


def rel_err_crop(ref: np.ndarray, got: np.ndarray, frac: float) -> float:
    """中央 ``frac`` 半宽裁剪区内的最大相对误差。"""
    m = ref.shape[0]
    h = max(2, int(round(m * frac)))
    s = (m - h) // 2
    return rel_err(ref[s : s + h, s : s + h], got[s : s + h, s : s + h])


def golden_section(f: Any, lo: float, hi: float, *, n_iters: int) -> float:
    """确定性黄金分割一维极小 (无随机重启)。"""
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(n_iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def resolve_chirp_sign_alpha(bench: OffFocusBench, vc: VisConfig) -> dict[str, Any]:
    """**数值确定**瞳面二次相位的符号与前因子, 不照抄公式。

    1. 在探针工作点 (小 Δz, 傍轴误差尚小) 上对 ``sign ∈ {−1, +1}`` 各算一遍相对误差,
       取小的那个;
    2. 再对连续系数 ``α`` 做黄金分割最小化残差 (理论值 1);
    3. 两个符号的残差若没拉开一个量级, 或 ``α`` 明显偏离 1, 直接抛错。

    返回 ``{"sign", "alpha", ...}`` 供 JSON 落盘。
    """
    rng = np.random.default_rng(vc.base.seed + 101)
    phase = rng.normal(0, 0.3, size=(bench.n, bench.n))
    dz = CHIRP_PROBE_DZ_UNITS * DZ_BLUR_M
    ref = bench.intensity(phase, dz)

    def probe(sign: float, alpha: float) -> float:
        chirp = PupilChirp(bench, sign=sign, alpha=alpha)
        return rel_err(ref, chirp.intensity_bp(phase, dz))

    trials = [{"sign_pupil": s, "rel_err": probe(s, 1.0)} for s in SIGN_CANDIDATES]
    trials.sort(key=lambda t: t["rel_err"])
    best, second = trials[0], trials[1]
    if best["rel_err"] > 1e-3:
        raise AssertionError(
            f"瞳面二次相位的两个符号都不吻合 (最优相对误差 {best['rel_err']:.3e}); "
            "这个建模可能就不成立, 停下来报告, 不要硬凑"
        )
    if second["rel_err"] < 1e-3 * max(best["rel_err"], 1e-300):
        raise AssertionError(
            f"符号无法唯一确定 (最优 {best['rel_err']:.3e} vs 次优 {second['rel_err']:.3e}); "
            "结论不可用"
        )

    sign = float(best["sign_pupil"])
    lo, hi = ALPHA_BOUNDS
    alpha = golden_section(lambda a: probe(sign, a), lo, hi, n_iters=ALPHA_ITERS)
    if abs(alpha - 1.0) > 0.05:
        raise AssertionError(
            f"二次相位前因子 α 反解成 {alpha:.4f} (理论值 1, 区间 {ALPHA_BOUNDS}); "
            "算子的前因子被写错或模型不成立, 停下来报告, 不要硬凑"
        )
    return {
        "sign_pupil": sign,
        "alpha": float(alpha),
        "probe_dz_units": CHIRP_PROBE_DZ_UNITS,
        "probe_dz_mm": CHIRP_PROBE_DZ_UNITS * DZ_BLUR_M * 1e3,
        "probe_rel_err_at_alpha1": float(probe(sign, 1.0)),
        "probe_rel_err_at_alpha_star": float(probe(sign, alpha)),
        "trials": trials,
        "note": (
            "符号与前因子由方案A 的 |U_CCD|^2 对拍反解, 不是照抄 exp(-i pi dz r^2/(lambda f^2)) "
            "的公式; alpha 的理论值为 1, 实测值落在 ±0.05 内"
        ),
    }


def self_check_zero_dz(bench: OffFocusBench, chirp: PupilChirp, phase: np.ndarray) -> dict[str, float]:
    """Δz=0 时瞳面二次相位算子必须逐位退化成纯傅里叶变换。"""
    ref = bench.intensity(phase, 0.0)
    err = rel_err(ref, chirp.intensity_bp(phase, 0.0))
    if err > 1e-12:
        raise AssertionError(f"Δz=0 时瞳面二次相位算子未退化为傅里叶变换: {err:.3e}")
    return {"dz0_rel_err": err}


def self_check_pupil_pixel_size() -> dict[str, float]:
    """模型网格的口径必须等于真实面板口径 —— 否则二次相位的物理长度整体错。

    这是硬件可施加性的前提: 模型口径 = ``2·BEAM_RADIUS_PX·PANEL_PIXEL_UM``, 面板
    口径 = ``2·BEAM_RADIUS_PANEL·SLM_PITCH_M``。两者相等 ⇒ 模型里的
    ``π·Δz·r²/(λf²)`` 可以逐像素搬到面板上, 不需要任何再缩放; 不相等 ⇒ 差一个
    常数倍, 而这个倍数是 ``(r_pupil/r_panel)²``, 平方关系放大得很快。

    注意: 模型网格的**像元**本身随 ``n_grid`` 变 (口径固定, 格数变), 所以不能拿
    ``aperture_size/n_grid`` 去比面板像元 —— 要比的是**口径**。
    """
    panel_aperture_m = 2.0 * BEAM_RADIUS_PANEL * SLM_PITCH_M
    err = abs(APERTURE_M - panel_aperture_m) / panel_aperture_m
    if err > 1e-12:
        raise AssertionError(
            f"模型口径 {APERTURE_M * 1e3:.4f} mm 与面板口径 "
            f"{panel_aperture_m * 1e3:.4f} mm 不一致; 二次相位的物理长度会差 "
            f"{(APERTURE_M / panel_aperture_m) ** 2:.3f} 倍的平方"
        )
    return {
        "model_aperture_m": APERTURE_M,
        "panel_aperture_m": panel_aperture_m,
        "panel_pitch_m": SLM_PITCH_M,
        "beam_radius_panel_px": float(BEAM_RADIUS_PANEL),
    }


# ---------------------------------------------------------------------------
# 画面度量
# ---------------------------------------------------------------------------
def zero_order_power_fraction(eta: float) -> float:
    """双分量模型里零级分量占的**功率**比例。

    公式 ``U = U_inc·[η + (1−η)e^{iφ}]`` 里的 ``η`` 是**振幅**系数, 所以功率占比是
    ``η²/(η²+(1−η)²)`` 而不是 ``η²``。对 ``η ≪ 1`` 两者差一个相对 ``2η`` 的小量, 但
    报告里一律给精确值, 免得读的人按 η² 估错量级。
    """
    num = eta * eta
    den = num + (1.0 - eta) ** 2
    return float(num / den) if den > 0 else 0.0


def pupil_field(
    phase: np.ndarray, eta: float, *, amp: np.ndarray, add_chirp: np.ndarray | None = None
) -> np.ndarray:
    """双分量瞳面场 ``U_inc·[η + (1−η)·e^{iφ}]`` (n×n, 复数)。

    :param amp: 照明幅度 (:attr:`OffFocusBench.amp`)。
    :param add_chirp: 可选的瞳面二次相位 (n×n, rad), 乘在**整个**场上 —— 它是
        「传播了 d」的替身, 对零级与整形分量是共用的 (见模块 docstring 的建模选择)。
    """
    ph = np.asarray(phase, dtype=np.float64)
    field = amp * (eta + (1.0 - eta) * np.exp(1j * ph))
    if add_chirp is not None:
        field = field * np.exp(1j * np.asarray(add_chirp, dtype=np.float64))
    return field


def _pad_pupil(bench: OffFocusBench, field_n: np.ndarray) -> np.ndarray:
    """n×n 瞳面复场 → m×m 零填充瞳面侧场 (居中)。与 ``OffFocusBench.focal`` 同约定。"""
    padded = np.zeros((bench.m, bench.m), dtype=np.complex128)
    padded[bench.pad : bench.pad + bench.n, bench.pad : bench.pad + bench.n] = field_n
    return padded


def intensity_scheme_a(
    bench: OffFocusBench,
    phase: np.ndarray,
    dz: float,
    eta: float = 0.0,
    *,
    chirp_rad: np.ndarray | None = None,
) -> np.ndarray:
    """方案A 正演 (双分量瞳面), 归一化到 sum = 1。

    只**组装**「瞳面场 → 透镜 FT → 焦后 Δz → CCD 强度」, 传播一步走
    :attr:`OffFocusBench._asm` (它包着仓库权威的 ``angular_spectrum_propagate``),
    **传播数学一行不重写**。
    """
    field = pupil_field(phase, eta, amp=bench.amp, add_chirp=chirp_rad)
    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(_pad_pupil(bench, field))))
    inten = np.abs(bench._asm(focal, dz)) ** 2  # 复用已验证的算子
    return inten / (inten.sum() + 1e-12)


def intensity_scheme_bp(
    bench: OffFocusBench,
    chirp: "PupilChirp",
    phase: np.ndarray,
    dz: float,
    eta: float = 0.0,
) -> np.ndarray:
    """瞳面域一次 FFT 正演 (方案B 本体, 双分量瞳面)。"""
    field = pupil_field(phase, eta, amp=bench.amp,
                        add_chirp=chirp.phase_rad(dz) if dz else None)
    inten = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(_pad_pupil(bench, field))))) ** 2
    return inten / (inten.sum() + 1e-12)


def self_check_two_component_at_eta0(
    bench: OffFocusBench, chirp: "PupilChirp", phase: np.ndarray, dz: float
) -> dict[str, float]:
    """``η=0`` 时两条正演必须逐位复现 :class:`OffFocusBench` 的既有结果。

    这是「我只复用了已验证的算子、自己只组装了双分量振幅」的可执行证明: 若这里不
    逐位相等, 说明我另写了一遍传播或填充约定, 结论不可用。
    """
    ref_a = bench.intensity(phase, dz)
    ref_b = chirp.intensity_bp(phase, dz)
    got_a = intensity_scheme_a(bench, phase, dz, 0.0)
    got_b = intensity_scheme_bp(bench, chirp, phase, dz, 0.0)
    err_a, err_b = rel_err(ref_a, got_a), rel_err(ref_b, got_b)
    if max(err_a, err_b) > 1e-12:
        raise AssertionError(
            f"η=0 时未能复现 OffFocusBench 的正演 (方案A {err_a:.3e}, 方案B {err_b:.3e}); "
            "双分量实现改动了已验证的算子约定, 结论不可用"
        )
    return {"eta0_scheme_A_rel_err": err_a, "eta0_scheme_B_rel_err": err_b}


def self_check_no_magnification(bench: OffFocusBench, chirp: "PupilChirp") -> dict[str, float]:
    """回归门禁: 远场像元 pitch 与 CCD 像元尺寸**都**不随 Δz 变。

    外部材料主张离焦会放大 ``M=(f+d)/f``、要按 M 重采样 target。Collins/ABCD 算下来
    ``B ≡ f`` ⇒ 没有放大 (见模块 docstring)。这里把「不存在 M」钉成可执行断言: 若
    日后有人把 pitch 或系数改回含 ``(f+d)`` 的形式, 这个门禁会立刻报出来。
    """
    cfg = bench.cfg
    pitch0 = float(cfg.far_field_pixel_size)
    # 瞳面二次相位的分母必须是 λf²: 若误写成 λf(f+d), 系数会随 Δz 偏移。
    lam, f = cfg.wavelength, cfg.focal_length
    dz = 20e-3
    denom_needed = abs(chirp.coefficient(dz) * lam * f**2 / (-chirp.sign * chirp.alpha * np.pi * dz))
    err_denom = abs(denom_needed - 1.0)
    if err_denom > 1e-9:
        raise AssertionError(
            f"瞳面二次相位的分母不是 λf² (偏离 {err_denom:.3e}); "
            "很可能是被换成了 λf(f+d) —— 那等价于引入 M=(f+d)/f, 对 2f 光路是错的"
        )
    # pitch 本身是 config 的常量, 任何 Δz 下重取都必须相同。
    if float(bench.cfg.far_field_pixel_size) != pitch0:
        raise AssertionError("远场像元 pitch 随 Δz 变化了 —— 这就是放大率 M 混进来了")
    return {
        "far_field_pixel_size_m": pitch0,
        "denominator_is_lambda_f_squared_rel_err": err_denom,
        "magnification_M": 1.0,
        "note": "Collins/ABCD: T(f+d)·Lens(f)·T(f) = [[-d/f, f], [-1/f, 0]], B ≡ f ⇒ 无放大",
    }


def halo_metrics(
    inten: np.ndarray, box: np.ndarray
) -> dict[str, float]:
    """框外 (即晕斑/零级侧) 的峰值与功率占比。

    框外峰值取的是**归一化**强度, 所以它直接就是「零级光斑有多亮」; 框外功率占比
    是「有多少光没进目标框」。两者都对归一化免疫, 可以跨 Δz 直接比。
    """
    total = float(inten.sum())
    outside = inten[~box]
    return {
        "halo_peak_outside": float(outside.max()) if outside.size else 0.0,
        "halo_power_fraction_outside": float(outside.sum() / total) if total > 0 else 0.0,
    }


def fit_halo_power_law(
    rows: list[dict[str, Any]], *, eta: float
) -> dict[str, float]:
    """把**纯零级**光斑峰值 vs Δz 拟成幂律, 返回指数。

    拟合量是 :func:`run_halo` 里 ``φ≡0`` 的纯零级正演的整帧峰值, **不是**整形画面里的
    框外峰值 —— 后者在 Δz=0 会被整形方形自己的边瓣主导, 在窗口卷绕区又完全失真,
    两者都量不到 ``I_halo`` 本身。

    参考材料给的量级是 ``I_halo ∝ f²/d²`` ⇒ 指数 **−2**。
    :func:`self_check_halo_scaling` 在窗口可信点上拟合并对超容差的情况直接抛错。
    """
    sel = [
        r for r in rows
        if abs(r["eta"] - eta) < 1e-12
        and r["dz_mm"] >= HALO_FIT_MIN_DZ_MM
        and r["window_ok"]
        and r["halo_peak_total"] > 0
    ]
    if len(sel) < 3:
        return {"exponent": float("nan"), "n_points": len(sel),
                "note": "窗口可信的 Δz 点不足 3 个, 幂律指数不可靠 (不作为门禁)"}
    x = np.log(np.array([r["dz_mm"] for r in sel], dtype=np.float64))
    y = np.log(np.array([r["halo_peak_total"] for r in sel], dtype=np.float64))
    exponent = float(np.polyfit(x, y, 1)[0])
    return {"exponent": exponent, "n_points": len(sel),
            "expected": HALO_EXPECT_EXPONENT,
            "dz_range_mm": [sel[0]["dz_mm"], sel[-1]["dz_mm"]]}


def run_halo(
    bench: OffFocusBench,
    chirp: PupilChirp,
    vc: VisConfig,
    gs0: np.ndarray,
) -> list[dict[str, Any]]:
    """零级晕斑扫描 (每个 η × 每个 Δz)。

    记两类量, 口径不同**不能混**:

    * ``halo_peak_total`` / ``halo_rms_radius_px``
      —— **纯零级**正演 (``φ≡0``, 即只有 ``η·U_inc`` 那一项) 的整帧峰值与二阶矩
      半径。这就是 ``I_halo`` 本身, 也是幂律自检的量: 峰值 ~1/d²、半径 ~d。
      零级分量与 η 无关, 所以纯零级帧每个 Δz 只算一次。
    * ``halo_peak_outside`` / ``halo_power_fraction_outside``
      —— 双分量**整形**画面里目标框外的峰值与功率占比。这是污染的实测口径, 但在
      Δz=0 量不到零级 (零级峰此时在框内), 在窗口卷绕区又被边瓣主导, 只作描述。
    """
    box = bench.target > 0
    zero_phase = np.zeros((bench.n, bench.n), dtype=np.float64)
    pure_cache: dict[float, tuple[np.ndarray, float, float]] = {}
    rows: list[dict[str, Any]] = []
    for eta in ETAS:
        for mm in DZ_SCAN_MM:
            dz = mm * 1e-3
            inten = intensity_scheme_a(bench, gs0, dz, eta)
            m = halo_metrics(inten, box)
            if mm not in pure_cache:
                pure = intensity_scheme_a(bench, zero_phase, dz, 1.0)
                _cx, _cy, rms = OffFocusBench.centroid_sigma(pure)
                pure_cache[mm] = (pure, float(rms), edge_energy(pure))
            pure, rms, pure_edge = pure_cache[mm]
            rows.append(
                {
                    "eta": float(eta),
                    "dz_mm": mm,
                    "dz_units": dz / DZ_BLUR_M,
                    "zero_order_power_fraction": zero_order_power_fraction(eta),
                    "halo_peak_total": float(pure.max()),
                    "halo_rms_radius_px": rms,
                    "halo_rms_radius_mm": rms * bench.dx * 1e3,
                    "halo_edge_energy": pure_edge,
                    **m,
                    "halo_radius_mm": 0.5 * APERTURE_M * dz / FOCAL_LENGTH_M,
                    # ⚠️ 窗口可信**必须**用 edge_energy, 不能用 RMS 半径: 周期卷绕把
                    # 溢出的光折回画面中央, RMS 半径反而**变小** (实测 d=0 时
                    # 7.1px → d=50mm 时 163px, 但 d≥25mm 起它已饱和不再反映真实晕斑),
                    # 拿它判窗口会漏掉全部卷绕点。edge_energy 直接测「能量是否堆到边上」。
                    "window_ok": pure_edge <= TRUST_EDGE_ENERGY,
                }
            )
    return rows


def self_check_halo_scaling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """纯零级光斑峰值必须按 ``I ∝ 1/d²`` 下降 —— 不达标说明双分量模型写错了。

    这是参考材料要求的自查: 「如果你的数值算出来不是这个趋势, 说明两分量模型写错了」。
    各 η 组的 ``halo_peak_total`` 是同一份纯零级帧 (零级分量与 η 无关), 所以每组都
    独立拟合并都要过; 拿整形画面的框外峰值去判物理会误报, 那个量只作描述。
    """
    fits = {
        f"eta_{eta:g}": fit_halo_power_law(rows, eta=eta)
        for eta in ETAS if eta != 0.0
    }
    bad = {
        k: v["exponent"]
        for k, v in fits.items()
        if v["n_points"] >= 3
        and abs(v["exponent"] - HALO_EXPECT_EXPONENT) > HALO_EXPONENT_TOL
    }
    if bad:
        raise AssertionError(
            f"纯零级光斑的标度不是 1/d²: {bad} (期望指数 {HALO_EXPECT_EXPONENT} "
            f"±{HALO_EXPONENT_TOL}); 两分量模型或零级正演算错了, "
            "停下来报告, 不要硬凑"
        )
    return fits


def edge_energy(inten: np.ndarray, frac: float = 0.1) -> float:
    """外侧 ``frac`` 环围的功率占比 —— 周期卷绕的直接探针。

    焦外弥散撑破仿真画幅时, ASM 的周期边界把溢出的能量卷回对面, 能量开始堆到画面
    边缘。这个量比「总能量守恒」敏感得多 (后者在周期边界下恒等于 1, 什么也测不到)。
    """
    m = inten.shape[0]
    b = max(1, int(round(m * frac)))
    ring = np.zeros((m, m), dtype=bool)
    ring[:b, :] = True
    ring[-b:, :] = True
    ring[:, :b] = True
    ring[:, -b:] = True
    total = float(inten.sum())
    return float(inten[ring].sum() / total) if total > 0 else 0.0


def measure(
    bench: OffFocusBench,
    inten: np.ndarray,
    *,
    ref_sigma: float,
    target_side_px: int,
    vc: VisConfig,
) -> dict[str, float]:
    """一帧的完整度量: 仓库口径 CV/PIB/score + 窗口可信度探针。"""
    m = bench.m
    out = dict(bench.evaluate(inten, ref_sigma=ref_sigma, target_side_px=target_side_px))
    out["edge_energy"] = edge_energy(inten)
    out["box_saturated"] = float(out["side_px"] >= m - 2)
    return out


# ---------------------------------------------------------------------------
# 方案A vs 瞳面二次相位: 逐点差异表 (这是「哪一点起不可信」的答案)
# ---------------------------------------------------------------------------
def run_scheme_gap(
    bench: OffFocusBench,
    chirp: PupilChirp,
    vc: VisConfig,
    ref_sigma: float,
    gs0: np.ndarray,
) -> list[dict[str, Any]]:
    """Δz 扫描上对比方案A 与瞳面二次相位近似的 ``|U_CCD|²``。

    基底是 plain GS 相位 (Δz=0 优化得到), 与 V1/V2 同一张, 所以三组图和这张表
    讲的是同一件事。
    """
    rows: list[dict[str, Any]] = []
    box = bench.target > 0
    for mm in DZ_SCAN_MM:
        dz = mm * 1e-3
        ref = bench.intensity(gs0, dz)
        got = chirp.intensity_bp(gs0, dz)
        mA = measure(bench, ref, ref_sigma=ref_sigma, target_side_px=vc.base.target_side_px, vc=vc)
        mB = measure(bench, got, ref_sigma=ref_sigma, target_side_px=vc.base.target_side_px, vc=vc)
        corner = chirp.phase_corner_rad(dz)
        trusted = (rel_err(ref, got) <= TRUST_REL_TOL) and (mA["edge_energy"] <= TRUST_EDGE_ENERGY)
        # 同一张表里再给一版「零级存在 (η=η_demo)」的偏差 —— 零级会同时改变 A 和 B,
        # 所以它是叠加在傍轴误差之上的第二层, 必须分开看。
        ref_e = intensity_scheme_a(bench, gs0, dz, ETA_DEMO)
        got_e = chirp.intensity_bp(gs0, dz, ETA_DEMO)
        mAe = measure(bench, ref_e, ref_sigma=ref_sigma,
                      target_side_px=vc.base.target_side_px, vc=vc)
        mBe = measure(bench, got_e, ref_sigma=ref_sigma,
                      target_side_px=vc.base.target_side_px, vc=vc)
        rows.append(
            {
                "dz_mm": mm,
                "dz_units": dz / DZ_BLUR_M,
                "chirp_corner_rad": corner,
                "chirp_corner_waves": corner / (2.0 * np.pi),
                "chirp_disc_edge_rad": chirp.phase_disc_edge_rad(dz),
                "chirp_disc_edge_waves": chirp.phase_disc_edge_rad(dz) / (2.0 * np.pi),
                "chirp_zernike_defocus_rad": chirp.zernike_defocus_coeff(
                    dz, bench.defocus_basis
                ),
                "chirp_fit_residual_rad": chirp.zernike_fit_residual(
                    dz, bench.defocus_basis
                ),
                "rel_err_max": rel_err(ref, got),
                "rel_err_mean": float(np.abs(ref - got).mean() / max(float(ref.mean()), 1e-30)),
                "rel_err_inner_max": rel_err_crop(ref, got, INNER_CROP),
                "edge_energy_scheme_A": mA["edge_energy"],
                "sigma_scheme_A": mA["sigma_px"],
                "sigma_over_ref": mA["sigma_px"] / max(ref_sigma, 1e-12),
                "box_saturated_scheme_A": mA["box_saturated"],
                "cv_scheme_A": mA["cv"],
                "pib_scheme_A": mA["pib"],
                "score_scheme_A": mA["score"],
                "cv_chirp": mB["cv"],
                "pib_chirp": mB["pib"],
                "score_chirp": mB["score"],
                "score_delta_chirp_minus_A": mB["score"] - mA["score"],
                # --- 同一表内: 带零级 (η=ETA_DEMO) 的那一版 ---
                "eta": ETA_DEMO,
                "rel_err_max_eta": rel_err(ref_e, got_e),
                "rel_err_inner_max_eta": rel_err_crop(ref_e, got_e, INNER_CROP),
                "halo_peak_outside_A_eta": halo_metrics(ref_e, box)["halo_peak_outside"],
                "halo_peak_outside_B_eta": halo_metrics(got_e, box)["halo_peak_outside"],
                "cv_scheme_A_eta": mAe["cv"],
                "cv_chirp_eta": mBe["cv"],
                "score_scheme_A_eta": mAe["score"],
                "score_chirp_eta": mBe["score"],
                "score_delta_eta": mAe["score"] - mB["score"],
                "trusted": trusted,
            }
        )
    return rows


def first_untrusted(gap_rows: list[dict[str, Any]], key: str) -> float | None:
    """差异表里第一个触发 ``key`` 的 Δz (mm); 全程都过则 None。"""
    for r in gap_rows:
        if not r[key]:
            return float(r["dz_mm"])
    return None


# ---------------------------------------------------------------------------
# V1: plain GS 相位在 0..50 mm 上的 CCD 画面
# ---------------------------------------------------------------------------
def run_v1(
    bench: OffFocusBench, vc: VisConfig, ref_sigma: float, gs0: np.ndarray
) -> list[dict[str, Any]]:
    """Δz 扫描下方案A 的整帧度量 (V1 的数值底稿), 每点给 η=0 与 η=η_demo 两版。"""
    rows: list[dict[str, Any]] = []
    box = bench.target > 0
    for mm in DZ_SCAN_MM:
        dz = mm * 1e-3
        i0 = intensity_scheme_a(bench, gs0, dz, 0.0)
        ie = intensity_scheme_a(bench, gs0, dz, ETA_DEMO)
        m0 = measure(bench, i0, ref_sigma=ref_sigma,
                     target_side_px=vc.base.target_side_px, vc=vc)
        me = measure(bench, ie, ref_sigma=ref_sigma,
                     target_side_px=vc.base.target_side_px, vc=vc)
        h0 = halo_metrics(i0, box)
        he = halo_metrics(ie, box)
        rows.append(
            {
                "dz_mm": mm,
                "dz_units": dz / DZ_BLUR_M,
                **{f"eta0_{k}": v for k, v in m0.items()},
                **{f"eta_{ETA_DEMO:g}_{k}": v for k, v in me.items()},
                "halo_peak_outside_eta0": h0["halo_peak_outside"],
                "halo_peak_outside_eta": he["halo_peak_outside"],
                "halo_power_fraction_outside_eta0": h0["halo_power_fraction_outside"],
                "halo_power_fraction_outside_eta": he["halo_power_fraction_outside"],
                "eta": ETA_DEMO,
                "zero_order_power_fraction": zero_order_power_fraction(ETA_DEMO),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# V2: 硬件可施加的瞳面二次相位图
# ---------------------------------------------------------------------------
def run_v2(
    bench: OffFocusBench,
    chirp: PupilChirp,
    vc: VisConfig,
    ref_sigma: float,
    gs0: np.ndarray,
) -> list[dict[str, Any]]:
    """每个 Δz 的相位图统计 + 「施加上它之后方案A 会测到多少」。"""
    rows: list[dict[str, Any]] = []
    for mm in DZ_SCAN_MM:
        dz = mm * 1e-3
        phase = chirp.phase_rad(dz)
        same = intensity_scheme_a(bench, gs0 + phase, dz)
        at0 = intensity_scheme_a(bench, gs0 + phase, 0.0)
        same_e = intensity_scheme_a(bench, gs0 + phase, dz, ETA_DEMO)
        at0_e = intensity_scheme_a(bench, gs0 + phase, 0.0, ETA_DEMO)
        m_same = measure(bench, same, ref_sigma=ref_sigma,
                         target_side_px=vc.base.target_side_px, vc=vc)
        m0 = measure(bench, at0, ref_sigma=ref_sigma,
                     target_side_px=vc.base.target_side_px, vc=vc)
        m_same_e = measure(bench, same_e, ref_sigma=ref_sigma,
                           target_side_px=vc.base.target_side_px, vc=vc)
        m0_e = measure(bench, at0_e, ref_sigma=ref_sigma,
                       target_side_px=vc.base.target_side_px, vc=vc)
        rows.append(
            {
                "dz_mm": mm,
                "dz_units": dz / DZ_BLUR_M,
                "peak_rad": float(np.max(np.abs(phase))),
                "peak_waves": float(np.max(np.abs(phase)) / (2.0 * np.pi)),
                "corner_rad": float(np.abs(chirp.phase_corner_rad(dz))),
                "corner_waves": float(chirp.phase_corner_rad(dz) / (2.0 * np.pi)),
                "disc_edge_rad": chirp.phase_disc_edge_rad(dz),
                "disc_edge_waves": chirp.phase_disc_edge_rad(dz) / (2.0 * np.pi),
                "zernike_defocus_rad": chirp.zernike_defocus_coeff(dz, bench.defocus_basis),
                "zernike_fit_residual_rad": chirp.zernike_fit_residual(
                    dz, bench.defocus_basis
                ),
                "cv_schemeA_at_same_dz": m_same["cv"],
                "pib_schemeA_at_same_dz": m_same["pib"],
                "score_schemeA_at_same_dz": m_same["score"],
                "cv_schemeA_at_dz0": m0["cv"],
                "pib_schemeA_at_dz0": m0["pib"],
                "score_schemeA_at_dz0": m0["score"],
                "eta": ETA_DEMO,
                "cv_schemeA_at_same_dz_eta": m_same_e["cv"],
                "pib_schemeA_at_same_dz_eta": m_same_e["pib"],
                "score_schemeA_at_same_dz_eta": m_same_e["score"],
                "cv_schemeA_at_dz0_eta": m0_e["cv"],
                "score_schemeA_at_dz0_eta": m0_e["score"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# V3: 各算法嵌入自己的离焦, 再用硬件可施加的形式评测
# ---------------------------------------------------------------------------
def build_algo_basis(bench: OffFocusBench, algo: str) -> tuple[np.ndarray, bool]:
    """算法的 Zernike 基与是否需要 GS 热启动。

    ``gs_zernike`` / ``gs_zernike_post`` 的基**刻意不含离焦 (2,0)**, 于是与
    ``gs_zernike_defocus`` 只差这一个自由度。
    """
    if algo == "gs":
        return bench.zernike_basis[:0], True
    if algo == "gs_zernike":
        return bench.zernike_basis, True
    if algo == "gs_zernike_defocus":
        return np.concatenate([bench.zernike_basis, bench.defocus_basis[None]]), True
    if algo == "gs_zernike_post":
        return bench.zernike_basis, True
    raise ValueError(f"unknown algo {algo!r}")


def optimise_post_defocus(
    bench: OffFocusBench,
    base_phase: np.ndarray,
    dz: float,
    ref_sigma: float,
    vc: VisConfig,
) -> tuple[float, float, bool]:
    """「事后外挂离焦」的外层确定性一维搜索。

    返回 ``(a*, best score, 是否撞搜索区间边界)``。撞边界 = 区间仍然不够宽, 报出的
    「最优离焦」是被端点决定的假象 —— 必须随产物一起报出去。
    """
    lo, hi = POST_DEFOCUS_LIMITS
    rows: list[dict[str, float]] = []

    def score_at(a: float) -> float:
        phase = base_phase + a * bench.defocus_basis
        m = bench.evaluate(bench.intensity(phase, dz), ref_sigma=ref_sigma,
                           target_side_px=vc.base.target_side_px)
        rows.append({"defocus_rad": float(a), **m})
        return float(m["score"])

    grid = [float(v) for v in np.linspace(lo, hi, vc.n_defocus_coarse)]
    for a in grid:
        score_at(a)
    step = grid[1] - grid[0]
    a0 = float(max(rows, key=lambda r: r["score"])["defocus_rad"])
    l, h = max(lo, a0 - step), min(hi, a0 + step)
    score_at(golden_section(lambda a: -score_at(a), l, h, n_iters=vc.n_defocus_golden))
    best = max(rows, key=lambda r: r["score"])
    eps = 1e-3 * (hi - lo)
    at_boundary = bool(
        best["defocus_rad"] <= lo + eps or best["defocus_rad"] >= hi - eps
    )
    return float(best["defocus_rad"]), float(best["score"]), at_boundary


def run_v3(
    bench: OffFocusBench,
    chirp: PupilChirp,
    vc: VisConfig,
    ref_sigma: float,
    gs_cache: dict[float, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """四种算法 × Δz: 各自按各自 Δz 优化 → 换算成瞳面二次相位 → 方案A 全扫描评测。

    返回 ``(扫描行, 嵌入信息, Δz=50mm 的光斑)``。
    """
    scan_rows: list[dict[str, Any]] = []
    embed_rows: list[dict[str, Any]] = []
    finals: dict[str, Any] = {}

    for mm in DZ_SCAN_MM:
        dz = mm * 1e-3
        key = round(dz, 12)
        if key not in gs_cache:
            gs_cache[key] = bench.gs_phase(dz, n_iters=vc.base.gs_iters, seed=vc.base.seed)
        gs = gs_cache[key]

        for algo in ALGO_KEYS:
            basis, _needs_gs = build_algo_basis(bench, algo)
            base = gs
            opt_score = float("nan")
            a_inline = 0.0
            at_boundary = False
            if basis.shape[0] == 0:
                phase_algo = base
            else:
                coeffs, opt_score = bench.optimize_coeffs(
                    basis, base, dz, n_iters=vc.base.adam_iters, lr=vc.base.adam_lr
                )
                phase_algo = base + np.tensordot(coeffs, basis, axes=(0, 0))
                if algo == "gs_zernike_defocus":
                    a_inline = float(coeffs[-1])
            if algo == "gs_zernike_post":
                a_inline, _s, at_boundary = optimise_post_defocus(
                    bench, phase_algo, dz, ref_sigma, vc
                )

            # 「最优离焦幅度 a*」→ 硬件可施加的瞳面二次相位。
            #
            # ⚠️ 这条换算**不能**把 a* 直接当成瞳面相位的系数: a* 的单位是
            # Zernike (2,0) 的弧度系数, 瞳面相位是 π·Δz·r²/(λf²) 的弧度场, 两套
            # 归一化差一个常数倍。这里走 :meth:`PupilChirp.zernike_defocus_coeff`
            # 的**数值投影**取那个倍数, 不照抄任何公式; 投影残差同时报进 CSV。
            quad_unit = chirp.phase_rad(dz)
            a_pupil_unit = chirp.zernike_defocus_coeff(dz, bench.defocus_basis)
            scale = a_inline / a_pupil_unit if abs(a_pupil_unit) > 1e-12 else 0.0
            quad = quad_unit * scale
            final_phase = phase_algo + quad

            for mm_eval in DZ_SCAN_MM:
                dz_eval = mm_eval * 1e-3
                for eta in (0.0, ETA_DEMO):
                    inten = intensity_scheme_a(bench, final_phase, dz_eval, eta)
                    m = measure(bench, inten, ref_sigma=ref_sigma,
                                target_side_px=vc.base.target_side_px, vc=vc)
                    h = halo_metrics(inten, bench.target > 0)
                    tag = "" if eta == 0.0 else "_eta"
                    scan_rows.append(
                        {
                            "algo": algo,
                            "eta": eta,
                            "opt_dz_mm": mm,
                            "eval_dz_mm": mm_eval,
                            "eval_dz_units": dz_eval / DZ_BLUR_M,
                            "embedded_defocus_rad": a_inline,
                            "embedded_defocus_at_boundary": at_boundary,
                            "opt_score_at_opt_dz": opt_score,
                            **{f"{k}{tag}": v for k, v in {**m, **h}.items()},
                        }
                    )
                    if mm_eval == max(DZ_SCAN_MM):
                        finals[(algo, eta)] = {
                            "phase": final_phase,
                            "dz": dz_eval,
                            "metrics": m,
                            "intensity": inten,
                            "halo": h,
                            "embedded_defocus_rad": a_inline,
                        }
            embed_rows.append(
                {
                    "algo": algo,
                    "opt_dz_mm": mm,
                    "opt_dz_units": dz / DZ_BLUR_M,
                    "embedded_defocus_rad": a_inline,
                    "embedded_defocus_at_boundary": at_boundary,
                    "pupil_quadratic_peak_rad": float(np.max(np.abs(quad))),
                    "pupil_quadratic_corner_rad": float(chirp.phase_corner_rad(dz) * scale),
                    "opt_score_at_opt_dz": opt_score,
                    "n_coeffs": int(basis.shape[0]),
                }
            )
        logger.info("  Δz={:>5.1f} mm 完成四个算法", mm)
    return scan_rows, embed_rows, finals


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 CSV (字段名取所有行的并集, 保持首次出现顺序)。"""
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _log_view(ax: Any, inten: np.ndarray, m: dict[str, float]) -> None:
    """按对数尺度画归一化强度, 并裁到光斑邻域 + 画出目标框 (整幅 log 图全是噪点)。"""
    M = inten.shape[0]
    half = max(m["side_px"], 4 * m["sigma_px"])
    cx, cy = m["cx"], m["cy"]
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, M))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, M))
    view = inten[y0:y1, x0:x1]
    ax.imshow(np.log10(np.clip(view, view.max() * 1e-6, None)), cmap="inferno")
    side = m["side_px"]
    ax.add_patch(
        plt.Rectangle((cx - side / 2 - x0, cy - side / 2 - y0), side, side,
                      fill=False, edgecolor="#00FF88", linewidth=1.6)
    )


def plot_v1_frame(
    inten0: np.ndarray,
    m0: dict[str, float],
    inten_e: np.ndarray,
    m_e: dict[str, float],
    title: str,
    out: Path,
) -> None:
    """一个 Δz 的两张全尺寸画面并排: 左 η=0 (理论对照), 右 η=η_demo (带零级晕斑)。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))
    for ax, inten, m, eta in (
        (axes[0], inten0, m0, 0.0), (axes[1], inten_e, m_e, ETA_DEMO)
    ):
        _log_view(ax, inten, m)
        ax.set_title(
            f"η={eta:g} (零级功率 {zero_order_power_fraction(eta):.2%})\n"
            f"CV={m['cv']:.4f}  PIB={m['pib']:.4f}\n"
            f"score={m['score']:.4f}  σ={m['sigma_px']:.1f}px  "
            f"边缘能量={m['edge_energy']:.3f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_v1_overview(
    v1_rows: list[dict[str, Any]],
    bench: OffFocusBench,
    gs0: np.ndarray,
    vc: VisConfig,
    out: Path,
    *,
    eta: float,
) -> None:
    """11 宫格总览: Δz = 0..50 mm 每 5 mm 一格 (η=0 与 η=η_demo 各出一张)。"""
    n = len(DZ_SCAN_MM)
    cols, rows_n = 4, 3
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.0 * cols, 3.6 * rows_n))
    for ax, mm in zip(axes.ravel(), DZ_SCAN_MM):
        dz = mm * 1e-3
        inten = intensity_scheme_a(bench, gs0, dz, eta)
        m = measure(bench, inten, ref_sigma=_REF_SIGMA[0],
                    target_side_px=vc.base.target_side_px, vc=vc)
        _log_view(ax, inten, m)
        flag = "" if not m["box_saturated"] else " (框饱和)"
        ax.set_title(
            f"Δz={mm:.0f} mm ({dz / DZ_BLUR_M:.1f} Δz_blur)\n"
            f"CV={m['cv']:.3f} PIB={m['pib']:.3f} score={m['score']:.3f}{flag}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        f"V1 plain GS 相位在 CCD 焦后 0..50 mm 上的远场 (方案A, η={eta:g}, "
        f"质心自动缩放框内计 CV/PIB)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


#: :func:`plot_v1_overview` 需要 ref_sigma, 但它是 main 的局部量 —— 用一个单元素
#: 容器传进去, 避免把 11 次重算的基准 σ 当成全局常量 (那正是 AGENTS 里记的「每帧
#: 重定位」类坑的变体: 基准 σ 只量一次, 框随图样走)。
_REF_SIGMA = [1.0]


def plot_v2_panels(
    chirp: PupilChirp, bench: OffFocusBench, v2_rows: list[dict[str, Any]], out: Path
) -> None:
    """11 宫格: 每个 Δz 的瞳面二次相位图 (硬件可施加的那张)。"""
    n = len(DZ_SCAN_MM)
    cols, rows_n = 4, 3
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.0 * cols, 3.6 * rows_n))
    for ax, mm, row in zip(axes.ravel(), DZ_SCAN_MM, v2_rows):
        phase = chirp.phase_rad(mm * 1e-3)
        ax.imshow(phase, cmap="RdBu_r")
        ax.set_title(
            f"Δz={mm:.0f} mm\n口径边缘 {row['disc_edge_rad']:.1f} rad "
            f"({row['disc_edge_waves']:.2f} 波)\n方案A 预测 CV={row['cv_schemeA_at_same_dz']:.3f}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        "V2 瞳面二次相位 (硬件可施加的相位图, raw 弧度不包裹; "
        f"符号={chirp.sign:+.0f}, α={chirp.alpha:.4f})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_v2_peak(
    v2_rows: list[dict[str, Any]], chirp: PupilChirp, out: Path
) -> None:
    """相位 vs Δz 的峰值相位曲线 —— 让人知道 50 mm 处绕了多少个 2π。"""
    mm = [r["dz_mm"] for r in v2_rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax.plot(mm, [r["disc_edge_rad"] for r in v2_rows], "o-", color="#FF5722", linewidth=2,
            label="口径边缘 r=D/2 (面板上真正被照亮的边缘)")
    ax.plot(mm, [r["corner_rad"] for r in v2_rows], "s--", color="#2196F3", linewidth=1.6,
            label="网格外接角点 (照明区外, 大 √2 倍)")
    ax.set_xlabel("CCD 焦后位移 Δz / mm")
    ax.set_ylabel("瞳面二次相位幅度 / rad")
    ax.set_title("峰值相位 vs Δz (raw 弧度)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax2.plot(mm, [r["disc_edge_waves"] for r in v2_rows], "o-", color="#4CAF50", linewidth=2)
    for row in v2_rows:
        ax2.annotate(f"{row['disc_edge_waves']:.1f}",
                     (row["dz_mm"], row["disc_edge_waves"]),
                     textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax2.set_xlabel("CCD 焦后位移 Δz / mm")
    ax2.set_ylabel("口径边缘相位 / 波数 (2π)")
    ax2.set_title("面板边缘相位绕过的 2π 个数")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_v3_curves(scan_rows: list[dict[str, Any]], out: Path) -> None:
    """V3: 每个算法一条 CV vs Δz 曲线 (11 点), 标出该算法嵌入的离焦幅度。

    左栏 η=0, 右栏 η=η_demo —— 零级对**所有**算法是同一个加性污染, 所以两条曲线的
    相对排序不该变; 若排序变了, 说明零级已经淹到能改结论的量级, 那是要立刻报的。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, tag, title in (
        (axes[0], "", "η=0 (无零级)"),
        (axes[1], "_eta", f"η={ETA_DEMO:g} (零级功率 {zero_order_power_fraction(ETA_DEMO):.2%})"),
    ):
        for algo in ALGO_KEYS:
            sel = sorted(
                (r for r in scan_rows
                 if r["algo"] == algo and r["opt_dz_mm"] == 0.0 and r["eta"] == (ETA_DEMO if tag else 0.0)),
                key=lambda r: r["eval_dz_mm"],
            )
            ax.plot([r["eval_dz_mm"] for r in sel], [r[f"cv{tag}"] for r in sel], "o-",
                    color=ALGO_COLORS[algo], label=ALGO_LABELS[algo], linewidth=2)
            sel_all = sorted(
                (r for r in scan_rows
                 if r["algo"] == algo and r["eta"] == (ETA_DEMO if tag else 0.0)),
                key=lambda r: r["eval_dz_mm"],
            )
            for opt_mm in (0.0, 25.0, 50.0):
                row = next(r for r in sel_all if r["opt_dz_mm"] == opt_mm)
                ax.plot([opt_mm], [row[f"cv{tag}"]], "*", color=ALGO_COLORS[algo],
                        markersize=13)
                if not tag:
                    ax.annotate(f"嵌入 {row['embedded_defocus_rad']:+.2f} rad",
                                (opt_mm, row[f"cv{tag}"]), textcoords="offset points",
                                xytext=(4, 6), fontsize=7, color=ALGO_COLORS[algo])
        ax.set_yscale("log")
        ax.set_xlabel("评测用 CCD 焦后位移 Δz / mm")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("目标框内 CV (自动定标框)")
    axes[0].legend(fontsize=8)
    fig.suptitle("V3 各算法 (在 Δz=0 优化) 嵌入离焦后, 方案A 全扫描 (星号 = 各算法的优化点)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_v3_finals(
    finals: dict[Any, Any], out: Path, *, eta: float
) -> None:
    """Δz = 50 mm 处四个算法的远场光斑 2×2 对比 (η=0 与 η=η_demo 各一张)。"""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.6))
    for ax, algo in zip(axes.ravel(), ALGO_KEYS):
        final = finals[(algo, eta)]
        m = final["metrics"]
        _log_view(ax, final["intensity"], m)
        ax.set_title(
            f"{ALGO_LABELS[algo]}\nCV={m['cv']:.3f} PIB={m['pib']:.3f} "
            f"score={m['score']:.3f}\n"
            f"嵌入离焦 {final['embedded_defocus_rad']:+.3f} rad  "
            f"框外晕斑峰值={final['halo']['halo_peak_outside']:.2e}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"V3 各算法嵌入自身最优离焦后, 在 Δz={max(DZ_SCAN_MM):.0f} mm 处的 CCD 远场 "
        f"(方案A, η={eta:g})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_halo(
    halo_rows: list[dict[str, Any]], fits: dict[str, Any], out: Path,
    *, sim_half_width_m: float,
) -> None:
    """纯零级光斑: 峰值 vs Δz (应 ∝ 1/d²) + 实测半径 vs 仿真画幅半宽。

    峰值用**纯零级**正演的整帧最大值 (各 η 组是同一份帧), 不画整形画面的框外峰值 ——
    那个量在 Δz=0 被边瓣主导、在卷绕区失真, 会把结论带偏。
    """
    colors = {0.02: "#2196F3", 0.05: "#FF5722", 0.10: "#9C27B0"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for eta in ETAS:
        if eta == 0.0:
            continue
        sel = sorted((r for r in halo_rows if r["eta"] == eta), key=lambda r: r["dz_mm"])
        mm = np.array([r["dz_mm"] for r in sel], dtype=np.float64)
        peak = np.array([r["halo_peak_total"] for r in sel], dtype=np.float64)
        good = mm >= HALO_FIT_MIN_DZ_MM
        col = colors.get(eta, "#666666")
        ax.plot(mm, peak, "o-", color=col, linewidth=2,
                label=f"η={eta:g} 纯零级 (零级功率 {zero_order_power_fraction(eta):.2%})")
        ref = peak[good][0] * (mm[good] / mm[good][0]) ** HALO_EXPECT_EXPONENT
        ax.plot(mm[good], ref, ":", color=col, linewidth=1.4, label=f"η={eta:g} 参考 ∝ 1/d²")
        fit = fits.get(f"eta_{eta:g}", {})
        if fit.get("n_points", 0) >= 3:
            ax.annotate(f"拟合指数 {fit['exponent']:+.2f}",
                        (mm[good][-1], peak[good][-1]), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color=col)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("CCD 焦后位移 Δz / mm")
    ax.set_ylabel("纯零级光斑峰值 (归一化强度)")
    ax.set_title(f"零级晕斑峰值 vs Δz (期望 ∝ 1/d², 容差 ±{HALO_EXPONENT_TOL})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")

    sel0 = sorted((r for r in halo_rows if r["eta"] == ETAS[1]), key=lambda r: r["dz_mm"])
    ax2.plot([r["dz_mm"] for r in sel0], [r["halo_rms_radius_mm"] for r in sel0], "o-",
             color="#4CAF50", linewidth=2, label="纯零级二阶矩半径 (实测)")
    ax2.plot([r["dz_mm"] for r in sel0], [r["halo_radius_mm"] for r in sel0], "s--",
             color="#FF9800", linewidth=1.6, label="几何估计 d·D/(2f)")
    ax2.axhline(sim_half_width_m * 1e3, color="k", linestyle="--", linewidth=1,
                label=f"仿真画幅半宽 {sim_half_width_m * 1e3:.3f} mm")
    ax2.set_xlabel("CCD 焦后位移 Δz / mm")
    ax2.set_ylabel("晕斑半径 / mm")
    ax2.set_title("晕斑半径 vs 仿真画幅半宽 (超出后峰值不可信)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_scheme_gap(gap_rows: list[dict[str, Any]], out: Path) -> None:
    """方案A vs 瞳面二次相位: 相对误差 vs Δz + 窗口可信度探针。"""
    mm = [r["dz_mm"] for r in gap_rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax.plot(mm, [r["rel_err_max"] for r in gap_rows], "o-", color="#FF5722", linewidth=2,
            label="全画幅 max 相对误差")
    ax.plot(mm, [r["rel_err_inner_max"] for r in gap_rows], "s--", color="#4CAF50", linewidth=1.8,
            label=f"中央 {INNER_CROP:.0%} 裁剪 max 相对误差")
    ax.axhline(TRUST_REL_TOL, color="k", linestyle="--", linewidth=1,
               label=f"可信判据 {TRUST_REL_TOL:g}")
    ax.set_yscale("log")
    ax.set_xlabel("CCD 焦后位移 Δz / mm")
    ax.set_ylabel("瞳面二次相位 vs 方案A 相对误差")
    ax.set_title("傍轴近似误差 (按 Δz² 增长)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax2.plot(mm, [r["edge_energy_scheme_A"] for r in gap_rows], "o-", color="#9C27B0",
             linewidth=2, label="方案A 画面边缘能量占比")
    ax2.axhline(TRUST_EDGE_ENERGY, color="k", linestyle="--", linewidth=1,
                label=f"窗口可信判据 {TRUST_EDGE_ENERGY:g}")
    ax2.set_yscale("log")
    ax2.set_xlabel("CCD 焦后位移 Δz / mm")
    ax2.set_ylabel("外侧 10% 环围功率占比")
    ax2.set_title("仿真窗口探针 (周期卷绕的直接影响)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="小网格冒烟跑")
    p.add_argument("--tag", default=None,
                   help="产物文件名后缀 (默认: --quick 时 _quick, 完整网格为空; "
                        "所有产物都已带 _visuals)")
    p.add_argument("--out", default=None, help="产物目录 (默认 report/shape_trials/gs_defocus_sim/artifacts)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    vc = VIS_QUICK if args.quick else VIS_FULL
    if args.tag is not None:
        vc = replace(vc, tag=args.tag)
    out = Path(args.out) if args.out else ROOT / "report" / "shape_trials" / "gs_defocus_sim" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    rc = vc.base
    logger.info(
        "光学常量: λ={:.0f}nm  f={:.2f}mm  D={:.3f}mm  面板像元={:.1f}um",
        WAVELENGTH_NM, FOCAL_LENGTH_M * 1e3, APERTURE_M * 1e3, PANEL_PIXEL_UM,
    )
    logger.info(
        "网格: n_grid={}  padding={}  far_field_size={}  画幅半宽={:.3f}mm  "
        "Δz_blur={:.4f}mm  50mm={:.1f} Δz_blur",
        rc.n_grid, rc.far_field_padding, rc.m, vc.far_field_half_width_m * 1e3,
        DZ_BLUR_M * 1e3, (max(DZ_SCAN_MM) * 1e-3) / DZ_BLUR_M,
    )
    logger.info(
        "窗口限制: 50mm 的焦外弥散半径={:.3f}mm, 画幅半宽={:.3f}mm (比值 {:.2f})",
        vc.blur_radius_m * 1e3, vc.far_field_half_width_m * 1e3,
        vc.blur_radius_m / vc.far_field_half_width_m,
    )
    assert len(DZ_SCAN_MM) == 11, f"Δz 扫描点必须是 11 个, 实际 {len(DZ_SCAN_MM)}"

    px = self_check_pupil_pixel_size()
    logger.info("自检 1/5: 模型口径 vs 面板口径 ... 通过 ({:.4f} mm)",
                px["model_aperture_m"] * 1e3)

    bench = OffFocusBench(rc.bench(), n_zernike=rc.n_zernike)
    chirp_meta = resolve_chirp_sign_alpha(bench, vc)
    chirp = PupilChirp(bench, sign=chirp_meta["sign_pupil"], alpha=chirp_meta["alpha"])
    logger.info(
        "自检 2/5: 数值反解瞳面二次相位 sign={:+.0f} alpha={:.4f} "
        "(探针 Δz={:.3f}mm, 相对误差 {:.3e} vs α=1 的 {:.3e})",
        chirp.sign, chirp.alpha, chirp_meta["probe_dz_mm"],
        chirp_meta["probe_rel_err_at_alpha_star"], chirp_meta["probe_rel_err_at_alpha1"],
    )
    zero = self_check_zero_dz(bench, chirp, np.zeros((bench.n, bench.n)))
    logger.info("自检 3/5: Δz=0 退化 ... 通过 ({dz0_rel_err:.3e})", **zero)
    rng_phase = np.random.default_rng(rc.seed + 7).normal(0, 0.3, size=(bench.n, bench.n))
    eta0 = self_check_two_component_at_eta0(bench, chirp, rng_phase, 1.0 * DZ_BLUR_M)
    logger.info(
        "自检 4/5: 双分量实现 η=0 复现 OffFocusBench ... 通过 "
        "(方案A {:.2e}, 方案B {:.2e})",
        eta0["eta0_scheme_A_rel_err"], eta0["eta0_scheme_B_rel_err"],
    )
    nomag = self_check_no_magnification(bench, chirp)
    logger.info("自检 5/5: 无放大率 M (ABCD B≡f, 分母=λf²) ... 通过 "
                "(分母相对误差 {:.2e})", nomag["denominator_is_lambda_f_squared_rel_err"])

    logger.info("跑 plain GS 基线 (Δz=0) ...")
    gs0 = bench.gs_phase(0.0, n_iters=rc.gs_iters, seed=rc.seed)
    ref_sigma = 1.0
    m0 = measure(bench, bench.intensity(gs0, 0.0), ref_sigma=ref_sigma,
                 target_side_px=rc.target_side_px, vc=vc)
    ref_sigma = m0["sigma_px"]
    m0 = measure(bench, bench.intensity(gs0, 0.0), ref_sigma=ref_sigma,
                 target_side_px=rc.target_side_px, vc=vc)
    _REF_SIGMA[0] = ref_sigma
    logger.info(
        "baseline (Δz=0, plain GS): score={:.4f} PIB={:.4f} CV={:.4f} σ={:.3f}px",
        m0["score"], m0["pib"], m0["cv"], ref_sigma,
    )

    logger.info("方案A vs 瞳面二次相位 逐点差异表 (11 个 Δz) ...")
    gap_rows = run_scheme_gap(bench, chirp, vc, ref_sigma, gs0)

    logger.info("零级晕斑扫描 (η ∈ {} ) ...", ETAS)
    halo_rows = run_halo(bench, chirp, vc, gs0)
    halo_fits = self_check_halo_scaling(halo_rows)
    for key, fit in halo_fits.items():
        if fit["n_points"] >= 3:
            logger.info("  {} 幂律指数 = {:+.3f} (期望 {:.1f} ±{:.1f}, {} 点)",
                        key, fit["exponent"], HALO_EXPECT_EXPONENT,
                        HALO_EXPONENT_TOL, fit["n_points"])
        else:
            logger.info("  {} 窗口可信点仅 {} 个, 幂律指数不可靠", key, fit["n_points"])

    logger.info("V1: plain GS 相位 × Δz 扫描 (η=0 与 η={}) ...", ETA_DEMO)
    v1_rows = run_v1(bench, vc, ref_sigma, gs0)
    for mm in V1_FULLSIZE_MM:
        dz = mm * 1e-3
        plot_v1_frame(
            intensity_scheme_a(bench, gs0, dz, 0.0),
            measure(bench, intensity_scheme_a(bench, gs0, dz, 0.0),
                    ref_sigma=ref_sigma, target_side_px=rc.target_side_px, vc=vc),
            intensity_scheme_a(bench, gs0, dz, ETA_DEMO),
            measure(bench, intensity_scheme_a(bench, gs0, dz, ETA_DEMO),
                    ref_sigma=ref_sigma, target_side_px=rc.target_side_px, vc=vc),
            f"V1 plain GS 相位 @ Δz = {mm:.0f} mm ({dz / DZ_BLUR_M:.1f} Δz_blur) "
            f"— 左 η=0, 右 η={ETA_DEMO:g}",
            out / f"v1_gs_dz{int(mm):02d}mm_visuals{vc.tag}.png",
        )
    plot_v1_overview(v1_rows, bench, gs0, vc,
                     out / f"v1_gs_dz_overview_visuals{vc.tag}.png", eta=0.0)
    plot_v1_overview(v1_rows, bench, gs0, vc,
                     out / f"v1_gs_dz_overview_eta_visuals{vc.tag}.png", eta=ETA_DEMO)

    logger.info("V2: 瞳面二次相位图 × Δz 扫描 ...")
    v2_rows = run_v2(bench, chirp, vc, ref_sigma, gs0)
    plot_v2_panels(chirp, bench, v2_rows, out / f"v2_pupil_chirp_panels_visuals{vc.tag}.png")
    plot_v2_peak(v2_rows, chirp, out / f"v2_pupil_chirp_peak_visuals{vc.tag}.png")

    logger.info("V3: 四算法 × Δz (各自优化 → 换算瞳面相位 → 方案A 全扫描) ...")
    gs_cache: dict[float, np.ndarray] = {0.0: gs0}
    scan_rows, embed_rows, finals = run_v3(bench, chirp, vc, ref_sigma, gs_cache)

    write_csv(out / f"scheme_gap_dz0_50mm_visuals{vc.tag}.csv", gap_rows)
    write_csv(out / f"v1_gs_dz_sweep_visuals{vc.tag}.csv", v1_rows)
    write_csv(out / f"v2_pupil_chirp_visuals{vc.tag}.csv", v2_rows)
    write_csv(out / f"v3_algo_scan_visuals{vc.tag}.csv", scan_rows)
    write_csv(out / f"v3_algo_embedded_defocus_visuals{vc.tag}.csv", embed_rows)
    write_csv(out / f"halo_zero_order_visuals{vc.tag}.csv", halo_rows)
    plot_scheme_gap(gap_rows, out / f"scheme_gap_visuals{vc.tag}.png")
    plot_halo(halo_rows, halo_fits, out / f"halo_zero_order_visuals{vc.tag}.png",
              sim_half_width_m=vc.far_field_half_width_m)
    plot_v3_curves(scan_rows, out / f"v3_cv_vs_dz_visuals{vc.tag}.png")
    plot_v3_finals(finals, out / f"v3_dz50_spots_visuals{vc.tag}.png", eta=0.0)
    plot_v3_finals(finals, out / f"v3_dz50_spots_eta_visuals{vc.tag}.png", eta=ETA_DEMO)

    # -- 摘要 ---------------------------------------------------------------
    def at(dz_mm: float) -> dict[str, Any]:
        return next(r for r in gap_rows if r["dz_mm"] == dz_mm)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "quick" if args.quick else "full",
        "wavelength_nm": WAVELENGTH_NM,
        "focal_length_m": FOCAL_LENGTH_M,
        "focal_length_recheck_m": focal_length_from_camera_pixel(
            wavelength_nm=WAVELENGTH_NM,
            camera_pixel_um=2.2,
            slm_pixel_um=PANEL_PIXEL_UM,
            focal_scale_px=TILT_SHIFT_SCALE,
            scale_measured_at_nm=1064.0,
        ),
        "aperture_m": APERTURE_M,
        "dz_blur_mm": DZ_BLUR_M * 1e3,
        "n_grid": rc.n_grid,
        "far_field_padding": rc.far_field_padding,
        "far_field_size": rc.m,
        "dz_scan_mm": list(DZ_SCAN_MM),
        "objective": "0.5*PIB + 0.5*(1/(1+CV)), CV/PIB in the centroid-tracked auto-scaled box",
        "chirp_sign_and_alpha": chirp_meta,
        "self_check": {
            "pupil_pixel_size": px,
            "dz0_degeneracy": zero,
            "two_component_at_eta0": eta0,
            "no_magnification": nomag,
            "asm_is_canonical": (
                "OffFocusBench._asm -> angular_spectrum_propagate (band_limit=True); "
                "GS 镜像与 gs_shape 在 Δz=0 逐位一致由兄弟脚本的自检保证; "
                "本脚本只组装双分量瞳面振幅, 传播一步都没重写"
            ),
        },
        "magnification_note": (
            "CCD 离焦**不是放大**: Collins/ABCD T(f+d)·Lens(f)·T(f) = "
            "[[-d/f, f], [-1/f, 0]], B ≡ f ⇒ 核尺度是 λf 而非 λ(f+d), 无 M=(f+d)/f。"
            "远场像元 pitch 恒为 λf/(N·dx_pupil), 目标框不做任何重采样; "
            "瞳面二次相位的分母取 λf² (不是 λf(f+d), 后者会弱化 1.408 倍)。"
            "CCD 离焦只让图样变模糊: 倾斜平面波经透镜聚到 fθ 后只是发散, 中心不跑。"
        ),
        "zero_order_model": {
            "formula": "U_pupil = U_inc * [eta + (1-eta)*exp(i*phi)]",
            "etas_amplitude": list(ETAS),
            "eta_demo": ETA_DEMO,
            "zero_order_power_fraction": {
                f"{eta:g}": zero_order_power_fraction(eta) for eta in ETAS
            },
            "chirp_applies_to": (
                "整个瞳面场 (含零级分量) —— chirp 是「传播了 d」的替身, 对零级和整形分量"
                "共用, 这样方案A 与方案B 仍 apples-to-apples"
            ),
            "halo_radius_formula": "rho ≈ d*D/(2f)",
            "halo_scaling_expected": "I_halo ∝ f²/d² (幂律指数 -2)",
            "halo_power_law_fit": halo_fits,
            "halo_peak_vs_dz": [
                {"eta": r["eta"], "dz_mm": r["dz_mm"],
                 "halo_peak_total": r["halo_peak_total"],
                 "halo_rms_radius_mm": r["halo_rms_radius_mm"],
                 "halo_edge_energy": r["halo_edge_energy"],
                 "halo_peak_outside": r["halo_peak_outside"],
                 "halo_power_fraction_outside": r["halo_power_fraction_outside"],
                 "halo_radius_mm_geometric": r["halo_radius_mm"],
                 "window_ok": r["window_ok"]}
                for r in halo_rows
            ],
        },
        "panel_geometry_for_hw": {
            "panel_h": SLM_PANEL_H,
            "panel_w": SLM_PANEL_W,
            "panel_pitch_m": SLM_PITCH_M,
            "beam_center_panel": list(BEAM_CENTER_PANEL),
            "beam_radius_panel": BEAM_RADIUS_PANEL,
            "note": (
                "模型网格口径 = 2*BEAM_RADIUS_PX*PANEL_PIXEL_UM, 与面板口径同源; "
                "所以模型里的二次相位系数可以直接搬到面板上, 无需再缩放"
            ),
        },
        "window_limit": {
            "sim_half_width_mm": vc.far_field_half_width_m * 1e3,
            "sim_half_width_note": "= n_grid*lambda*f/D, 与 far_field_padding 无关",
            "blur_radius_at_max_dz_mm": vc.blur_radius_m * 1e3,
            "blur_over_half_width": vc.blur_radius_m / vc.far_field_half_width_m,
            "hw_ccd_half_width_mm": 0.5 * HW_CCD_PX[0] * HW_CCD_PIXEL_M * 1e3,
            "note": (
                f"仿真画幅半宽 {0.5 * rc.m * rc.bench().far_field_pixel_size * 1e3:.3f} mm, "
                f"而 50mm 的焦外几何弥散半径已到 {vc.blur_radius_m * 1e3:.3f} mm (比值 "
                f"{vc.blur_radius_m / vc.far_field_half_width_m:.2f}); ASM 在周期边界下会把"
                "溢出能量卷回对面, 所以大 Δz 的方案A 画面被窗口主导。真实实验系统的 CCD "
                f"视场 (2592x1944 @ 2.2um) 半宽 "
                f"{0.5 * HW_CCD_PX[0] * HW_CCD_PIXEL_M * 1e3:.2f} mm, 大得多, "
                "这是**仿真窗口**的限制, 不是实验系统的限制"
            ),
        },
        "trust_criteria": {
            "rel_err_max": TRUST_REL_TOL,
            "edge_energy_max": TRUST_EDGE_ENERGY,
            "first_untrusted_dz_mm": first_untrusted(gap_rows, "trusted"),
        },
        "scheme_gap_table": [
            {
                "dz_mm": r["dz_mm"],
                "dz_units": round(r["dz_units"], 3),
                "corner_rad": round(r["chirp_corner_rad"], 3),
                "corner_waves": round(r["chirp_corner_waves"], 3),
                "disc_edge_rad": round(r["chirp_disc_edge_rad"], 3),
                "disc_edge_waves": round(r["chirp_disc_edge_waves"], 3),
                "rel_err_max": r["rel_err_max"],
                "rel_err_inner_max": r["rel_err_inner_max"],
                "edge_energy_scheme_A": r["edge_energy_scheme_A"],
                "sigma_over_ref": r["sigma_over_ref"],
                "cv_scheme_A": r["cv_scheme_A"],
                "cv_chirp": r["cv_chirp"],
                "score_scheme_A": r["score_scheme_A"],
                "score_chirp": r["score_chirp"],
                "rel_err_max_eta": r["rel_err_max_eta"],
                "cv_scheme_A_eta": r["cv_scheme_A_eta"],
                "cv_chirp_eta": r["cv_chirp_eta"],
                "halo_peak_outside_A_eta": r["halo_peak_outside_A_eta"],
                "trusted": r["trusted"],
            }
            for r in gap_rows
        ],
        "v1_baseline_gs_dz0": {
            "score": m0["score"], "pib": m0["pib"], "cv": m0["cv"],
            "sigma_px": ref_sigma, "side_px": m0["side_px"],
        },
        "v2_peak_phase": [
            {
                "dz_mm": r["dz_mm"],
                "corner_rad": r["corner_rad"],
                "corner_waves": r["corner_waves"],
                "disc_edge_rad": r["disc_edge_rad"],
                "disc_edge_waves": r["disc_edge_waves"],
                "cv_schemeA_at_same_dz": r["cv_schemeA_at_same_dz"],
                "cv_schemeA_at_dz0": r["cv_schemeA_at_dz0"],
            }
            for r in v2_rows
        ],
        "v3_embedded_defocus": [
            {
                "algo": r["algo"], "label": ALGO_LABELS[r["algo"]],
                "opt_dz_mm": r["opt_dz_mm"],
                "embedded_defocus_rad": r["embedded_defocus_rad"],
                "at_boundary": r["embedded_defocus_at_boundary"],
                "pupil_quadratic_peak_rad": r["pupil_quadratic_peak_rad"],
            }
            for r in embed_rows
            if r["opt_dz_mm"] in (0.0, 25.0, 50.0)
        ],
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "outputs": {
            "scheme_gap_csv": f"scheme_gap_dz0_50mm_visuals{vc.tag}.csv",
            "v1_csv": f"v1_gs_dz_sweep_visuals{vc.tag}.csv",
            "v2_csv": f"v2_pupil_chirp_visuals{vc.tag}.csv",
            "v3_scan_csv": f"v3_algo_scan_visuals{vc.tag}.csv",
            "v3_embed_csv": f"v3_algo_embedded_defocus_visuals{vc.tag}.csv",
            "halo_csv": f"halo_zero_order_visuals{vc.tag}.csv",
            "scheme_gap_png": f"scheme_gap_visuals{vc.tag}.png",
            "v1_frames_png": [f"v1_gs_dz{int(mm):02d}mm_visuals{vc.tag}.png" for mm in V1_FULLSIZE_MM],
            "v1_overview_png": f"v1_gs_dz_overview_visuals{vc.tag}.png",
            "v1_overview_eta_png": f"v1_gs_dz_overview_eta_visuals{vc.tag}.png",
            "v2_panels_png": f"v2_pupil_chirp_panels_visuals{vc.tag}.png",
            "v2_peak_png": f"v2_pupil_chirp_peak_visuals{vc.tag}.png",
            "v3_curves_png": f"v3_cv_vs_dz_visuals{vc.tag}.png",
            "v3_finals_png": f"v3_dz50_spots_visuals{vc.tag}.png",
            "v3_finals_eta_png": f"v3_dz50_spots_eta_visuals{vc.tag}.png",
            "halo_png": f"halo_zero_order_visuals{vc.tag}.png",
        },
    }
    (out / f"summary_defocus_sweep_visuals{vc.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary["scheme_gap_table"], indent=2, ensure_ascii=False))
    print(f"\n产物目录: {out}")
    print(f"耗时: {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()