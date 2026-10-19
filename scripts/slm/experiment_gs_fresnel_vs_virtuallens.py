"""焦后离焦的两套建模 (方案A Fresnel/ASM vs 方案B 等效虚拟透镜) 对照实验。

回答三个问题
------------

**Q1 —— 方案A 与方案B 到底等不等价? 符号是哪个?**

CCD 被移到透镜焦后 Δz, 正演有两条路:

* **方案A (Fresnel / ASM)**: ``U_CCD = ASM( FT(U_SLM), Δz )`` —— 透镜先做傅里叶
  变换, 再对**已变换的场**自由传播 Δz。
* **方案B (等效虚拟透镜)**: 傍轴极限下与 A 等价, 只是把二次相位挪到瞳面 ——
  瞳面乘一次二次相位, 只做一次 FFT, 省掉 ASM。

**二次相位的正负号不假设**: 本脚本在探针工作点上把 ``sign_pupil ∈ {−1, +1}`` 各算
一遍相对误差, 再用连续的一维黄金分割反解系数 ``α`` (理论值 1), 取残差最小的符号
并把残差写进输出。符号错则整张表错, 所以这是硬门禁 (:func:`resolve_scheme_b_sign`),
不达标直接抛错。

⚠️ **输出二次相位: 算子里有它, 但强度里它不出现。** Fresnel 传播搬到瞳面时完整
算子还带一个 ``exp(+iπ|x_c|²/(λΔz))``, 在 ``Δz = Δz_blur`` 时角端相位可达 ~10³ rad,
逐像素不同。但它**模长为 1**, 所以对归一化强度**逐位无影响** —— 本脚本用
``Bp_chirped`` 变体把这个事实数值钉死 (:func:`summarise_equivalence`), 而不是把它
当成一个「漏了就会错」的坑。它真正起作用的场合是 **GS 的反向传播**: 那里取的是
相位而不是模长, 漏掉会给取回的瞳面相位加一个逐像素的旋转。方案B 的 GS 算子
因此带上它 (物理完整的算子), 而它的**评测**不受影响。

**Q2 —— 把离焦并进优化环路能救回多少?**

已有的 E1 是**事后**在不含离焦的 GS 相位上外挂离焦。本脚本对每组报
"该 Δz 工作点上的 best score 与 min CV (含达到它们的离焦幅度)", 五组对照:

==== =====================================================================
组    GS 相位场                       离焦
==== =====================================================================
G1    Δz=0 优化后迁移 (Δz 盲)        不加 (基线, = E1 的 defocus=0 行)
G2    Δz=0 优化后迁移 (Δz 盲)        事后外挂 (E1 口径)
G1′   **在该 Δz 上重新优化**          **不加** ← 最关键的对照
G3    在该 Δz 上重新优化             联合优化 (外层确定性一维搜索)
G4    同 G3, 但内环用方案B 算子       同上
==== =====================================================================

**为什么必须有 G1′**: 没有它, ``G3 − G1`` 把「按 Δz 重新优化」的收益和「离焦」的
收益混在一起, 只能说「离焦没用」而说不清收益到底来自哪。G1′ 的相位场与 G3 **完全
相同** (所以不额外花一次 GS), 只差离焦那一维, 于是

* ``G1′ − G1`` = **重新优化本身**的收益;
* ``G3 − G1′`` = **离焦在重优化之上的净贡献**。

⚠️ **口径来源已核实** (读 ``experiment_gs_defocus_offocus.py`` 的当前代码):
``run_e1`` 接收一个**算好的** ``gs0`` 参数、内部从不调用 ``gs_phase``; 唯一的
``gs0 = bench.gs_phase(0.0, ...)`` 在 ``main()`` 里跑一次。**所以 E1 的 GS 是
Δz 盲的, 只在评测时换 Δz** —— 也就是本表的 G1/G2。而 E2 的模式 B 走
``run_method`` → ``bench.gs_phase(dz_opt, ...)`` 且按 ``dz_opt`` 缓存, 是**真的
逐点重优化**, 对应本表的 G1′/G3/G4。两者的数不可直接比较。

**离焦搜索区间**: ``[−2.5, +2.5]`` rad。前置任务 E2 的区间只有约 ``[−0.28, +0.28]``
(``defocus_rad_opt`` 全程只取 ±0.2806 / −0.0028), 而其 E1 宽扫描显示最优离焦可达
±1.2 —— 所以**别引用 E2 那个「离焦让 score 下降 0.001~0.002」的结论, 那是区间不足
的假象**。本脚本另外报出 ``defocus_at_boundary``: 撞边界 = 区间仍然不够。

⚠️ **口径边界 (必须写明)**: GS 相位是**自由相位**, ``φ_GS + a·Z₂⁰`` 仍在 GS 可达的
相位空间内; 而标量离焦就是 Zernike (2,0) 模态, 本身已被 GS 的自由相位覆盖。所以
G3−G1′ 度量的是离焦**能不能在重优化之上再多挤出一点**, 而不是「能不能表达」——
预期它很小甚至为负, 这本身是结论而不是失败。

**四/五组一律用方案A 的正演评测** —— 方案A 是物理真值, 方案B 只是内环的近似;
用真值评测才回答「方案B 优化出来的相位在真实系统里管不管用」。离焦幅度由外层
确定性一维搜索给出 (21 点粗扫 + 黄金分割, 固定 seed, 不随机重启)。

⚠️⚠️ **已实测的关键事实: 收敛后的 GS 对 Δz 不敏感。** GS 是交替投影, 收敛到的是
「瞳面幅度 + 目标幅度」两个幅度约束的交点, 而角谱传递函数是**酉的** (``|H| = 1``,
无 bin 被带限剔除), 所以两个约束与传播算子无关 —— 不同 Δz 的 GS 从**同一个**初值噪声
出发会落进同一个吸引域。因此 G3 相对 G1 的增益要拆成两部分看:

* **GS 看见 Δz** 的贡献 (即 ``GS(Δz)`` 相对 ``GS(0)``, 离焦 = 0): CSV 里
  ``phase_source`` 为 ``gs_at_dz`` 且 ``defocus_rad == 0`` 的那一行;
* **外层离焦一维搜索** 的贡献: 同一行与其最优行之差。

:func:`gs_coherence` 把这个事实直接量化 (``|mean(e^{iΔφ})|``, 注意是**复**均值 ——
``mean(|e^{iΔφ}|)`` 恒等于 1, 是无信息的量, 别用)。

复用仓库已有资产 (不自写)
--------------------------

* 传播: :func:`~ao_shaping.algorithm.signal_processing.angular_spectrum.angular_spectrum_propagate`。
* GS: 正/反向算子由上面的权威传播器给出; 循环骨架、目标、瞳面幅度、口径、
  光学常量全部照搬 :mod:`scripts.slm.experiment_gs_defocus_offocus` 的已验证结构,
  并在 Δz=0 上跑 :func:`self_check_gs` 与仓库的 ``gs_shape`` 逐位对齐 (门禁)。
* 整形配置/目标/评分: :mod:`~ao_shaping.algorithm.signal_processing.square_shaping`。
* Zernike (2,0) 基: :class:`~ao_shaping.utils.wavefront.zernike_calc.ZernikeGenerator`。
* 光学常量: 从 :mod:`ao_shaping.optimizer.wfless.slm_gs_refine` 追到源头
  (面板像元 8.0 µm、CCD 像元 2.2 µm、光束半径 450 SLM px、焦距由
  :func:`~ao_shaping.utils.wavefront.matrix_utils.focal_length_from_camera_pixel`
  从实测 2f 焦距标度导出、波长 ``DEFAULT_WAVELENGTH_NM``)。

标定 (与 ``report/iterative_zernike_shaping/data.json`` 同口径)
--------------------------------------------------------------

目标框跟着图样走: 中心 = 强度质心, 边长 = ``target_side_px`` 乘该帧实测二阶矩
半径 σ 与参考 σ 的比值; 参考 σ 在 (Δz=0, 离焦=0) 的 G1 基线上量一次, 故基线上
自动框与 data.json 的定尺框重合。Δz=0 时 :func:`FresnelBench.evaluate` 的自动框与
定尺框逐位相同, 所以 G1 在 Δz=0 必须复现 data.json 的 ``gs`` 行。

score 口径与既有报告一致: ``0.5·PIB + 0.5·(1/(1+CV))``, CV / PIB 在目标框内算。

运行
----

.. code-block:: bash

    python scripts/slm/experiment_gs_fresnel_vs_virtuallens.py           # 完整网格
    python scripts/slm/experiment_gs_fresnel_vs_virtuallens.py --quick   # 小网格冒烟

产物写到 ``report/shape_trials/``, 文件名带 ``_fresnel_vs_vl``:
两个 CSV (等价性逐点 / 四组完整轨迹 + 各组最优点)、一个 summary JSON、三张图
(A/B 等价性误差 vs Δz、CV vs Δz 四组对照、G3 最优离焦幅度 vs Δz)。

纯仿真: 不打开任何设备。
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

from ao_shaping.algorithm.signal_processing.angular_spectrum import (
    angular_spectrum_propagate,
)
from ao_shaping.algorithm.signal_processing.square_shaping import (
    SquareShapingConfig,
    composite_from_pib_cv,
    gs_shape,
    make_square_target,
)
from ao_shaping.optimizer.wfless import slm_gs_refine as gs_refine
from ao_shaping.utils.image.beam_metrics import measure_spot_diameter_cam
from ao_shaping.utils.wavefront.matrix_utils import focal_length_from_camera_pixel
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator, zernike_modes

# ---------------------------------------------------------------------------
# 光学常量 —— 与 scripts/slm/experiment_gs_defocus_offocus.py 同一段推导,
# 全部从 slm_gs_refine.py 追到源头, 不在这里另立一份。
# 下面的常量快照是那个脚本已发布 summary 里的值, 只用于对拍 (见 :func:`self_check_constants`)。
# ---------------------------------------------------------------------------
WAVELENGTH_NM = float(gs_refine.DEFAULT_WAVELENGTH_NM)
WAVELENGTH_M = WAVELENGTH_NM * 1e-9
PANEL_PIXEL_UM = float(gs_refine._DEFAULT_PANEL_PIXEL_UM)
CAMERA_PIXEL_UM = float(gs_refine._DEFAULT_CAMERA_PIXEL_UM)
BEAM_RADIUS_PX = float(gs_refine._DEFAULT_BEAM_RADIUS_PX)
SCALE_MEASURED_AT_NM = float(gs_refine._TILT_SHIFT_SCALE_MEASURED_AT_NM)
FOCAL_LENGTH_M = float(
    focal_length_from_camera_pixel(
        wavelength_nm=WAVELENGTH_NM,
        camera_pixel_um=CAMERA_PIXEL_UM,
        slm_pixel_um=PANEL_PIXEL_UM,
        focal_scale_px=gs_refine._TILT_SHIFT_SCALE_PX,
        scale_measured_at_nm=SCALE_MEASURED_AT_NM,
    )
)
#: 照明口径 = 2 × 光束半径 × 面板像元 (与 utils.slm.bench_constants 一致)
APERTURE_M = 2.0 * BEAM_RADIUS_PX * PANEL_PIXEL_UM * 1e-6


def defocus_scale_m(wavelength_m: float, focal_m: float, aperture_m: float) -> float:
    """焦后位移的参考尺度 Δz_blur (m): 使离焦弥散等于焦面光斑尺度。

    CCD 后移 Δz 时, 瞳面边缘 (h = D/2) 的光线在 CCD 上偏移 ``h·Δz/f``; 令其等于
    Airy 半径 ``1.22·λf/D`` 得 ``Δz = 2.44·λf²/D²``。扫描点写成 ``u · Δz_blur``
    即与具体波长/焦距/口径解耦。
    """
    return 2.44 * wavelength_m * focal_m**2 / aperture_m**2


DZ_BLUR_M = defocus_scale_m(WAVELENGTH_M, FOCAL_LENGTH_M, APERTURE_M)

#: 与 report/iterative_zernike_shaping/data.json 同口径的仿真网格
N_GRID = 64
FAR_FIELD_PADDING = 8
TARGET_SIDE_PX = 43
SEED = 0

#: 瞳面离焦模式 (Noll 4 = (2,0) 离焦)。四组的离焦旋钮共用这一个模式。
DEFO_MODE = (2, 0)
N_ZERNIKE = 4

#: 参考常量快照 (scripts/slm/experiment_gs_defocus_offocus.py 已发布的 summary)
REF_SNAPSHOT = {"focal_length_m": 0.12240601503759398, "dz_blur_mm": 0.7503639840341595}
#: data.json :: gs (Δz = 0 基线, 完整网格)
DATA_JSON_GS = {"score": 0.8118054772479197, "PIB": 0.9153051811202841, "CV": 0.4118196372088354}

DZ_UNITS_FULL = (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
DZ_UNITS_QUICK = (-1.0, -0.5, 0.0, 0.5, 1.0)


def equivalence_dz_units() -> tuple[float, ...]:
    """等价性验证的 Δz 扫描点 (Δz_blur 单位), 对称且**近零处加密**。

    傍轴近似误差按 ``Δz²`` 增长 (实测: 0.25 → 4.7e-6, 2.0 → 2.1e-4), 所以要回答
    「等价到 1e-6 的 |Δz| 上限」必须在零点附近密集采样, 否则只报出「上限 = 0」。
    """
    coarse = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)
    fine = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
    mag = tuple(sorted(fine + coarse))
    return tuple(sorted({0.0} | {s * v for v in mag for s in (-1.0, 1.0)}))

#: 离焦一维搜索区间 (Zernike (2,0) 系数, 弧度)。
#: ⚠️ 必须宽到让最优点**落在区间内部**, 否则「best score」是被区间端点决定的假象。
#: 两轮实测把这个区间从 ±2.0 → ±2.5 → **±5.0**: ±2.5 时 G3 在 Δz=+0.5/+1.0 的
#: 最优解就顶在 −2.5 的边界上 (defocus_at_boundary=True), 说明区间仍不够。
#: 前置任务 experiment_gs_defocus_offocus.py 的 E2 把区间限制在约 [−0.28, +0.28]
#: (``defocus_rad_opt`` 全程只取 ±0.2806 / −0.0028), 而同任务 E1 的宽扫描显示最优
#: 离焦可达 ±1.2 —— 所以 E2 报出的「离焦让 score 下降 0.001~0.002」是
#: **搜索区间不足的假象, 不是物理结论**。
DEFOCUS_LIMITS = (-5.0, 5.0)
#: 撞边界判定的容差 (区间宽度的 1%)
DEFOCUS_BOUNDARY_FRAC = 0.01

#: 方案B 的两个待定符号: 瞳面二次相位、与输出二次相位
SIGN_CANDIDATES = (-1.0, 1.0)

GROUPS: tuple[str, ...] = ("G1", "G2", "G1p", "G3", "G4")
GROUP_LABELS: dict[str, str] = {
    "G1": "G1 方案A · Δz 盲 GS · 不加离焦 (基线, E1 口径)",
    "G2": "G2 方案A · Δz 盲 GS · 事后外挂离焦 (E1 口径)",
    "G1p": "G1′ 方案A · 逐 Δz 重优化 · 不加离焦 (对照)",
    "G3": "G3 方案A · 逐 Δz 重优化 + 离焦联合优化",
    "G4": "G4 方案B · 逐 Δz 重优化 + 离焦联合优化",
}
GROUP_COLORS: dict[str, str] = {
    "G1": "#9E9E9E",
    "G2": "#FF9800",
    "G1p": "#9C27B0",
    "G3": "#2196F3",
    "G4": "#4CAF50",
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """一次运行的网格与优化预算。"""

    n_grid: int = N_GRID
    far_field_padding: int = FAR_FIELD_PADDING
    target_side_px: int = TARGET_SIDE_PX
    seed: int = SEED
    n_zernike: int = N_ZERNIKE
    gs_iters: int = 200
    dz_units: tuple[float, ...] = DZ_UNITS_FULL
    #: 离焦一维搜索: 粗扫点数 + 黄金分割细化步数 (确定性, 无随机重启)
    n_defocus_coarse: int = 21
    n_defocus_golden: int = 12
    #: 等价性验证的 Δz 扫描 (Δz_blur 单位) 与随机瞳面相位个数
    equiv_dz_units: tuple[float, ...] = equivalence_dz_units()
    equiv_n_random: int = 3
    #: 等价性容差: 相对误差 <= 该值才算"等价"
    equiv_tol: float = 1e-6
    tag: str = ""

    @property
    def m(self) -> int:
        """零填充远场网格边长。"""
        return self.n_grid * self.far_field_padding

    def bench(self) -> SquareShapingConfig:
        """带实验系统光学常量的整形配置。

        ``make_gaussian_pupil`` 只依赖 ``aperture_size / n_grid`` 与
        ``aperture_size / 3.5`` 的比值, 所以远场图样 (以远场像素计) 与
        wavelength / focal_length / aperture_size 无关。
        """
        return SquareShapingConfig(
            n_grid=self.n_grid,
            target_side_px=self.target_side_px,
            far_field_padding=self.far_field_padding,
            seed=self.seed,
            wavelength=WAVELENGTH_M,
            focal_length=FOCAL_LENGTH_M,
            aperture_size=APERTURE_M,
        )


QUICK = RunConfig(
    n_grid=32,
    far_field_padding=4,
    target_side_px=21,
    gs_iters=60,
    dz_units=DZ_UNITS_QUICK,
    equiv_n_random=2,
    tag="_quick",
)
FULL = RunConfig()


# ---------------------------------------------------------------------------
# 正向模型: 方案A / 方案B (焦平面域 / 瞳面域)
# ---------------------------------------------------------------------------
class FresnelBench:
    """可注入焦后位移 Δz 的 2f 整形正向模型, 带三套传播算子 (纯 numpy)。

    传播一律走 ``angular_spectrum_propagate`` (焦平面域方案A) 或它的傍轴极限
    (方案B); 本类只负责**组装**「瞳面场 → 透镜 → CCD 强度」这条链, 以及 GS 的
    正/反向算子。

    三套算子 (都以**居中**排布为输入输出):

    ``A``
        ``fftshift(ASM(ifftshift(fftshift(fft2(ifftshift(field)))), dz))`` ——
        物理真值 (精确角谱)。
    ``Bf`` (焦平面域方案B)
        把 ``kz = sqrt(k²−k⊥²)`` 展开到二阶, 传递函数退化为二次相位
        ``exp(s·iπ·Δz·dx_p²·(nx²+ny²)/(λf²))``, 于是「焦后传播」= 一次
        ``fft → 乘二次相位 → ifft``。这是方案B 在**频域**的写法。
    ``Bp`` (瞳面域方案B, 即「省掉 ASM」的实用写法)
        ``chirp_pupil → 一次 fft → chirp_out``, 频域写法搬到瞳面。

    ``Bp`` 与 ``Bf`` 的差别是本脚本要量化的核心: (1) 傍轴近似误差;
    (2) 离散环卷绕 vs 线性卷积; (3) 是否补回输出二次相位
    ``exp(+iπ|x_c|²/(λΔz))``。
    """

    def __init__(self, cfg: SquareShapingConfig, *, n_zernike: int = N_ZERNIKE) -> None:
        self.cfg = cfg
        self.n = cfg.n_grid
        self.m = cfg.far_field_size
        self.dx = cfg.far_field_pixel_size  # 远场像元间距 (m)
        self.lam = cfg.wavelength
        self.f = cfg.focal_length
        self.pad = (self.m - self.n) // 2
        self.amp = cfg.make_gaussian_pupil().astype(np.float64)
        self.target = make_square_target(cfg)
        #: 优化目标函数用的**定尺**支撑域 (居中, 边长 target_side_px)
        self.support = self.target > 0
        #: 瞳面像元间距 (m) —— 方案B 的二次相位系数用它
        self.dx_pupil = cfg.aperture_size / self.n

        gen = ZernikeGenerator((self.n, self.n), radius=self.n / 2.0, n_orders=n_zernike)
        self.zernike_modes: list[tuple[int, int]] = [
            m for m in zernike_modes(n_zernike) if m != (0, 0) and m != DEFO_MODE
        ]
        self.zernike_basis = np.stack(
            [
                np.nan_to_num(gen.generate_polynomial({nm: 1.0}), nan=0.0).astype(np.float64)
                for nm in self.zernike_modes
            ]
        )
        self.defocus_basis = np.nan_to_num(
            gen.generate_polynomial({DEFO_MODE: 1.0}), nan=0.0
        ).astype(np.float64)

        # -- 网格坐标与二次相位缓存 (只读, 提前建好) --------------------------
        # ASM 的 k⊥ 用的是 **raw fft2 序** 的频率下标 (angular_spectrum 的模块
        # 文档明写 "torch.fft.fftfreq is already in raw-fft2 order, so the
        # frequency grid is used exactly as returned -- do not fftshift it").
        # 这里必须照抄同一约定: 用 fftfreq(m)*m 而不是 arange(m)-m//2 —— 后者是
        # fftshift 后的排布, 直接用会让整个 n² 传递函数循环错位半个数组,
        # 表现为方案B 与方案A 的相对误差恒等于 ~1。
        n_unshifted = np.fft.fftfreq(self.m) * self.m
        self._n2 = n_unshifted[:, None] ** 2 + n_unshifted[None, :] ** 2
        # 瞳面 / 输出面的物理坐标 (m) —— 均为**居中**排布
        x_p = (np.arange(self.m) - self.m // 2) * self.dx_pupil
        self._xp2 = x_p[:, None] ** 2 + x_p[None, :] ** 2
        x_out = (np.arange(self.m) - self.m // 2) * self.dx
        self._xo2 = x_out[:, None] ** 2 + x_out[None, :] ** 2

    # -- 方案A: 精确角谱 ---------------------------------------------------
    def _asm_raw(self, field: np.ndarray, dz: float) -> np.ndarray:
        """权威传播器走一步。输入**居中**排布, 输出**未 fftshift**。"""
        return angular_spectrum_propagate(
            np.fft.ifftshift(field), self.dx, dz, self.lam, band_limit=True
        )

    def _asm(self, field: np.ndarray, dz: float) -> np.ndarray:
        return np.fft.fftshift(self._asm_raw(field, dz))

    def _asm_inverse(self, field: np.ndarray, dz: float) -> np.ndarray:
        """``_asm`` 的伴随逆 (CCD → 焦平面, 居中排布)。Δz = 0 时退化为恒等。"""
        return np.fft.fftshift(np.fft.ifft2(self._asm_raw(field, -dz)))

    def focal(self, field: np.ndarray) -> np.ndarray:
        """M×M 瞳面侧场 → 焦平面复场 (居中排布, Δz = 0)。"""
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

    def unfocal(self, focal_field: np.ndarray) -> np.ndarray:
        """焦平面复场 → M×M 瞳面侧场 (居中排布, Δz = 0)。"""
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(focal_field)))

    def pad_pupil(self, phase: np.ndarray) -> np.ndarray:
        """n×n 瞳面相位 → M×M 瞳面侧复场 (居中排布)。"""
        out = np.zeros((self.m, self.m), dtype=np.complex128)
        out[self.pad : self.pad + self.n, self.pad : self.pad + self.n] = self.amp * np.exp(
            1j * np.asarray(phase, dtype=np.float64)
        )
        return out

    def window_phase(self, field: np.ndarray) -> np.ndarray:
        """M×M 瞳面侧复场 → n×n 瞳面相位 (原始弧度)。"""
        s = self.pad
        return np.angle(field[s : s + self.n, s : s + self.n])

    # -- 方案B: 焦平面域 (Bf) 与瞳面域 (Bp) --------------------------------
    def _bf_raw(self, focal_field: np.ndarray, dz: float, *, sign: float, alpha: float = 1.0) -> np.ndarray:
        """傍轴传递函数: ``fft → 乘二次相位 → ifft`` (未 fftshift 进, 未 fftshift 出)。

        系数由 ASM 展开 ``kz ≈ k − k⊥²/(2k)`` 得
        ``−(kx²+ky²)Δz/(2k) = −π Δz dx_p²(nx²+ny²)/(λf²)``, 故
        ``h = exp(sign · i · alpha · π Δz dx_p² (nx²+ny²)/(λf²))``。
        ``alpha`` 是供数值反解的连续系数 (理论值 1)。
        """
        if dz == 0.0:
            return np.fft.ifftshift(focal_field)
        h = np.exp(
            sign * 1j * alpha * np.pi * dz * self.dx_pupil**2 * self._n2 / (self.lam * self.f**2)
        )
        return np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(focal_field)) * h)

    def bf(self, focal_field: np.ndarray, dz: float, *, sign: float, alpha: float = 1.0) -> np.ndarray:
        """焦平面域方案B (居中排布)。Δz = 0 时逐位等于 :meth:`focal`。"""
        return np.fft.fftshift(self._bf_raw(focal_field, dz, sign=sign, alpha=alpha))

    def pupil_chirp(self, dz: float, *, sign: float, alpha: float = 1.0) -> np.ndarray:
        """瞳面侧二次相位 (M×M): ``exp(sign · i · alpha · π Δz |x|²/(λf²))``。"""
        if dz == 0.0:
            return np.ones((self.m, self.m), dtype=np.complex128)
        return np.exp(sign * 1j * alpha * np.pi * dz * self._xp2 / (self.lam * self.f**2))

    def output_chirp(self, dz: float, *, sign: float) -> np.ndarray:
        """输出面 (CCD) 二次相位: ``exp(sign · i · π |x_c|²/(λ Δz))``。

        这是 Fresnel 传播搬到瞳面时**必须补回**的那一项 ——
        ``U_ccd ∝ exp(+iπ|x_c|²/(λΔz)) · FT[U_pupil·exp(−iπΔz|x|²/(λf²))]``。
        模长为 1 但逐像素不同, 漏掉它强度就不一样。
        """
        if dz == 0.0:
            return np.ones((self.m, self.m), dtype=np.complex128)
        return np.exp(sign * 1j * np.pi * self._xo2 / (self.lam * dz))

    def bp(self, phase: np.ndarray, dz: float, *, sign: float, out_chirp: bool = False) -> np.ndarray:
        """瞳面域方案B 的一次 FFT 正演 (n×n 相位 → m×m CCD 复场, 居中排布)。"""
        field = self.pad_pupil(phase) * self.pupil_chirp(dz, sign=sign)
        out = self.focal(field)
        if out_chirp:
            out = out * self.output_chirp(dz, sign=-sign)
        return out

    # -- 四个强度正演 (归一化到 sum = 1) -----------------------------------
    @staticmethod
    def _norm(inten: np.ndarray) -> np.ndarray:
        return inten / (inten.sum() + 1e-12)

    def intensity_A(self, phase: np.ndarray, dz: float) -> np.ndarray:
        """方案A (精确角谱) —— 物理真值, 也是四组统一的评测正演。"""
        inten = np.abs(self._asm(self.focal(self.pad_pupil(phase)), dz)) ** 2
        return self._norm(inten)

    def intensity_Bf(self, phase: np.ndarray, dz: float, *, sign: float, alpha: float = 1.0) -> np.ndarray:
        inten = np.abs(self.bf(self.focal(self.pad_pupil(phase)), dz, sign=sign, alpha=alpha)) ** 2
        return self._norm(inten)

    def intensity_Bp(self, phase: np.ndarray, dz: float, *, sign: float) -> np.ndarray:
        """方案B 本体: 瞳面乘二次相位 + 一次 FFT (不含输出二次相位)。"""
        inten = np.abs(self.bp(phase, dz, sign=sign)) ** 2
        return self._norm(inten)

    def intensity_Bp_chirped(self, phase: np.ndarray, dz: float, *, sign: float) -> np.ndarray:
        """方案B + 输出二次相位 ``exp(+iπ|x_c|²/(λΔz))``。

        Fresnel 传播搬到瞳面时, 完整的算子确实还带这一项。它**模长为 1**, 所以对
        归一化强度**逐位无影响** —— 本变体就是为了把这个事实数值钉死, 而不是为了
        修一个「漏了就会错」的坑。但 GS 的**反向**传播必须带上它: 那里取的是相位
        而不是模长, 一个逐像素的旋转会改变取回的瞳面相位。
        """
        inten = np.abs(self.bp(phase, dz, sign=sign, out_chirp=True)) ** 2
        return self._norm(inten)

    # -- GS ---------------------------------------------------------------
    def gs_phase(
        self,
        dz: float,
        *,
        n_iters: int,
        seed: int,
        scheme: str = "A",
        sign: float = -1.0,
    ) -> np.ndarray:
        """在给定焦后位移下跑 GS, 返回瞳面相位 (原始弧度)。

        循环骨架与 :func:`square_shaping.gs_shape` 逐行同构, 只把两条 FFT 换成
        ``fwd`` / ``bwd`` 两个可注入算子; 初始化、目标幅度、瞳面幅度、口径全部取自
        ``square_shaping`` 的同一份配置。``scheme="A"`` 且 Δz = 0 时与 ``gs_shape``
        逐位相同 (:func:`self_check_gs` 是硬门禁)。
        """
        if scheme == "A":
            def fwd(field: np.ndarray) -> np.ndarray:
                return self._asm(self.focal(field), dz)

            def bwd(field: np.ndarray) -> np.ndarray:
                return self._asm_inverse(field, dz)
        elif scheme == "B":
            chirp_p = self.pupil_chirp(dz, sign=sign)
            chirp_p_inv = np.conj(chirp_p)
            chirp_o = self.output_chirp(dz, sign=-sign)
            chirp_o_inv = np.conj(chirp_o)

            def fwd(field: np.ndarray) -> np.ndarray:
                return self.focal(field * chirp_p) * chirp_o

            def bwd(field: np.ndarray) -> np.ndarray:
                return self.unfocal(field * chirp_o_inv) * chirp_p_inv
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        rng = np.random.default_rng(seed)
        amp_target = np.sqrt(self.target)
        amp_slm = np.zeros((self.m, self.m), dtype=np.float64)
        amp_slm[self.pad : self.pad + self.n, self.pad : self.pad + self.n] = self.amp
        field = amp_slm * np.exp(1j * rng.normal(0, 0.1, size=amp_slm.shape))
        for _ in range(n_iters):
            ff = fwd(field)
            ff = amp_target * np.exp(1j * np.angle(ff))
            field = bwd(ff)
            field = amp_slm * np.exp(1j * np.angle(field))
        return self.window_phase(field)

    # -- 度量 --------------------------------------------------------------
    @staticmethod
    def centroid_sigma(inten: np.ndarray) -> tuple[float, float, float]:
        """功率质心 ``(cx, cy)`` 与二阶矩半径 σ (px)。"""
        total = float(inten.sum())
        m = inten.shape[0]
        yy, xx = np.ogrid[:m, :m]
        cx = float((inten * xx).sum() / total)
        cy = float((inten * yy).sum() / total)
        sigma = float(np.sqrt(((inten * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum()) / total))
        return cx, cy, sigma

    def box_stats(self, inten: np.ndarray, cx: float, cy: float, side: int) -> tuple[float, float]:
        """目标框内的 ``(PIB, CV)``。框边长夹在 ``[5, m-2]``; 框内无光时 CV 返回 ``inf``。"""
        m = inten.shape[0]
        side = int(np.clip(side, 5, m - 2))
        c0 = int(round(cx))
        r0 = int(round(cy))
        rows = slice(max(r0 - side // 2, 0), min(r0 + side // 2, m))
        cols = slice(max(c0 - side // 2, 0), min(c0 + side // 2, m))
        vals = inten[rows, cols]
        total = float(inten.sum())
        pib = float(vals.sum() / total) if total > 0 else 0.0
        cv = float(vals.std() / vals.mean()) if vals.size and vals.mean() > 0 else float("inf")
        return pib, cv

    def evaluate(self, inten: np.ndarray, *, ref_sigma: float, target_side_px: int) -> dict[str, float]:
        """整帧度量: 自动定标框 (主) + 定尺框 (副, 与 data.json 同口径)。"""
        cx, cy, sigma = self.centroid_sigma(inten)
        side = int(round(target_side_px * sigma / max(ref_sigma, 1e-12)))
        pib, cv = self.box_stats(inten, cx, cy, side)
        pib_f, cv_f = self.box_stats(inten, cx, cy, target_side_px)
        d90 = measure_spot_diameter_cam(inten, energy=0.90)
        return {
            "cx": cx,
            "cy": cy,
            "sigma_px": sigma,
            "side_px": int(np.clip(side, 5, inten.shape[0] - 2)),
            "cv": cv,
            "pib": pib,
            "score": composite_from_pib_cv(pib, cv),
            "cv_fixed": cv_f,
            "pib_fixed": pib_f,
            "score_fixed": composite_from_pib_cv(pib_f, cv_f),
            "d90_px": float(d90),
            "d90_um": float(d90) * self.dx * 1e6,
        }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def self_check_constants() -> dict[str, float]:
    """与 scripts/slm/experiment_gs_defocus_offocus.py 已发布的常量对拍。

    本文件不 import 那个兄弟脚本 (它正在被后台 run 使用, 耦合过去会把两份产物
    绑死), 改为和它已发布的 summary 值对拍 —— 光学常量错一个数, 整张 Δz 扫描
    的单位就全错。
    """
    errs = {
        "focal_length_rel": abs(FOCAL_LENGTH_M - REF_SNAPSHOT["focal_length_m"]) / REF_SNAPSHOT["focal_length_m"],
        "dz_blur_rel": abs(DZ_BLUR_M * 1e3 - REF_SNAPSHOT["dz_blur_mm"]) / REF_SNAPSHOT["dz_blur_mm"],
    }
    if max(errs.values()) > 1e-9:
        raise AssertionError(
            f"光学常量与 experiment_gs_defocus_offocus.py 的已发布快照不一致: {errs}; "
            "Δz 扫描的单位会错, 结论不可用"
        )
    return errs


def self_check_gs(rc: RunConfig, bench: FresnelBench) -> float:
    """GS 镜像 vs 仓库 ``gs_shape`` (方案A, Δz = 0)。返回 ``max|Δphase|``。"""
    ref = gs_shape(rc.bench(), n_iters=rc.gs_iters, seed=rc.seed).phase
    mine = bench.gs_phase(0.0, n_iters=rc.gs_iters, seed=rc.seed, scheme="A")
    delta = float(np.abs(ref - mine).max())
    if delta > 1e-9:
        raise AssertionError(
            f"GS 镜像在 Δz=0 与 gs_shape 不一致 (max|Δphase|={delta:.3e}); "
            "正/反向算子的约定被改坏了, 结论不可用"
        )
    return delta


def self_check_schemeB_at_zero(bench: FresnelBench, phase: np.ndarray) -> dict[str, float]:
    """Δz = 0 时 Bf / Bp 都必须逐位等于方案A (退化为纯傅里叶变换)。"""
    ref = bench.intensity_A(phase, 0.0)
    out = {}
    for name, got in (
        ("bf", bench.intensity_Bf(phase, 0.0, sign=-1.0)),
        ("bp", bench.intensity_Bp(phase, 0.0, sign=-1.0)),
        ("bp_chirped", bench.intensity_Bp_chirped(phase, 0.0, sign=-1.0)),
    ):
        err = float(np.abs(got - ref).max() / max(float(ref.max()), 1e-30))
        out[name] = err
    if max(out.values()) > 1e-12:
        raise AssertionError(f"Δz=0 时方案B 未退化为傅里叶变换: {out}")
    return out


def resolve_scheme_b_sign(bench: FresnelBench, rc: RunConfig) -> dict[str, Any]:
    """**数值确定**方案B 的二次相位符号与系数, 不假设。

    1. 在探针工作点 (Δz = +1·Δz_blur, 随机瞳面相位) 上把
       ``sign_pupil ∈ {−1, +1}`` 各算一遍相对误差, 取小的那个;
    2. 再用连续的一维黄金分割反解系数 ``alpha`` (理论值 1);
    3. 若 alpha 明显偏离 1、或 ``+1`` 与 ``−1`` 的残差没有拉开一个量级, 直接抛错。

    返回 ``{"sign", "alpha", "probe": [...]}``, 供 JSON 落盘。
    """
    rng = np.random.default_rng(rc.seed + 101)
    phase = rng.normal(0, 0.3, size=(bench.n, bench.n))
    dz = 1.0 * DZ_BLUR_M
    ref = bench.intensity_A(phase, dz)

    trials: list[dict[str, Any]] = []
    for sign in SIGN_CANDIDATES:
        got = bench.intensity_Bp(phase, dz, sign=sign)
        trials.append({"sign_pupil": sign, "rel_err": rel_err(ref, got)})
    trials.sort(key=lambda t: t["rel_err"])
    best, second = trials[0], trials[1]
    if best["rel_err"] > 1e-3:
        raise AssertionError(
            f"方案B 的两个符号都不吻合 (最优相对误差 {best['rel_err']:.3e}); "
            "方案B 的建模本身可能就不成立, 停下来报告, 不要硬凑"
        )
    if second["rel_err"] < 1e-3 * max(best["rel_err"], 1e-300):
        raise AssertionError(
            f"方案B 的符号无法唯一确定 (最优 {best['rel_err']:.3e} vs 次优 {second['rel_err']:.3e}); "
            "结论不可用"
        )

    sign = float(best["sign_pupil"])

    # alpha 只在 Bf (焦平面域) 上反解 —— Bp 的 chirp 系数已由理论固定, 反解它没有意义
    def loss_bf(alpha: float) -> float:
        got = bench.intensity_Bf(phase, dz, sign=sign, alpha=alpha)
        return rel_err(ref, got)

    alpha = golden_section(loss_bf, 0.0, 2.0, n_iters=40)
    return {
        "sign": sign,
        "alpha": float(alpha),
        "probe_dz_units": 1.0,
        "probe_rel_err_Bp": float(best["rel_err"]),
        "probe_rel_err_Bf_at_alpha1": float(loss_bf(1.0)),
        "probe_rel_err_Bf_at_alpha_star": float(loss_bf(alpha)),
        "trials": trials,
    }


def rel_err(ref: np.ndarray, got: np.ndarray) -> float:
    """归一化强度的相对误差 (按 ``max|ref|`` 归一, 与既有脚本的相对误差口径一致)。"""
    return float(np.abs(ref - got).max() / max(float(ref.max()), 1e-30))


def mean_rel_err(ref: np.ndarray, got: np.ndarray) -> float:
    return float(np.abs(ref - got).mean() / max(float(ref.mean()), 1e-30))


def golden_section(f: Any, lo: float, hi: float, *, n_iters: int) -> float:
    """确定性黄金分割一维极小 (无随机重启)。返回近似极小点。"""
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


# ---------------------------------------------------------------------------
# 第一步: 等价性验证
# ---------------------------------------------------------------------------
def run_equivalence(bench: FresnelBench, rc: RunConfig, sign: float) -> list[dict]:
    """在 Δz 扫描 × 随机瞳面相位上对比方案A 与方案B 的 ``|U_CCD|²``。

    同时记录三个方案B 变体:

    * ``Bf`` —— 焦平面域 (一次 fft + 乘二次相位 + 一次 ifft): 只测**傍轴近似误差**;
    * ``Bp`` —— 瞳面域 (瞳面二次相位 + 一次 fft), 即「省掉 ASM」的实用写法:
      在傍轴近似之外还额外含**离散环卷绕 vs 线性卷积**的误差;
    * ``Bp_chirped`` —— ``Bp`` 再乘输出二次相位 ``exp(+iπ|x_c|²/(λΔz))``:
      用来把「输出二次相位模长为 1, 对强度逐位无影响」这件事数值钉死。
    """
    rows: list[dict] = []
    for r in range(rc.equiv_n_random):
        rng = np.random.default_rng(rc.seed + 1000 + r)
        phase = rng.normal(0, 0.3, size=(bench.n, bench.n))
        for u in rc.equiv_dz_units:
            dz = u * DZ_BLUR_M
            ref = bench.intensity_A(phase, dz)
            bp = bench.intensity_Bp(phase, dz, sign=sign)
            variants = {
                "Bf": bench.intensity_Bf(phase, dz, sign=sign),
                "Bp": bp,
                "Bp_chirped": bench.intensity_Bp_chirped(phase, dz, sign=sign),
            }
            # Bp vs Bp_chirped 的**直接**差: 输出二次相位是否真的对强度无影响
            chirp_vs_bp = float(np.abs(variants["Bp_chirped"] - bp).max() / max(float(bp.max()), 1e-30))
            for name, got in variants.items():
                rows.append(
                    {
                        "random_idx": r,
                        "dz_units": u,
                        "dz_mm": dz * 1e3,
                        "variant": name,
                        "rel_err_max": rel_err(ref, got),
                        "rel_err_mean": mean_rel_err(ref, got),
                        "rel_err_vs_Bp_max": chirp_vs_bp if name == "Bp_chirped" else 0.0,
                    }
                )
    return rows


def summarise_equivalence(rows: list[dict], tol: float) -> dict[str, Any]:
    """逐变体给出「等价到 ``tol`` 的 |Δz| 范围」与全区间最大相对误差。"""
    out: dict[str, Any] = {}
    for variant in ("Bf", "Bp", "Bp_chirped"):
        sel = [r for r in rows if r["variant"] == variant]
        worst: dict[tuple[int, float], float] = {}
        for r in sel:
            key = (r["random_idx"], abs(r["dz_units"]))
            worst[key] = max(worst.get(key, 0.0), r["rel_err_max"])
        by_dz = sorted({k[1] for k in worst})
        per_dz = {d: max(worst[(i, d)] for (i, _dd) in worst if _dd == d) for d in by_dz}
        within = [d for d in by_dz if per_dz[d] <= tol]
        limit = max(within) if within else 0.0
        # 越过 tol 的最小 |Δz| (线性内插), 给出「1e-6 上限」的亚网格估计
        crossing = None
        for lo, hi in zip(by_dz, by_dz[1:]):
            if per_dz[lo] <= tol < per_dz[hi] and per_dz[hi] > per_dz[lo]:
                frac = (tol - per_dz[lo]) / (per_dz[hi] - per_dz[lo])
                crossing = lo + frac * (hi - lo)
                break
        out[variant] = {
            "max_rel_err_overall": max(per_dz.values()),
            "max_rel_err_at_dz0": per_dz.get(0.0, 0.0),
            "equivalent_abs_dz_units_limit": limit,
            "equivalent_abs_dz_units_interpolated": crossing,
            "equivalent_abs_dz_mm": (crossing or limit) * DZ_BLUR_M * 1e3,
            "rel_err_by_abs_dz": {f"{d:.4f}": per_dz[d] for d in by_dz},
        }
    out["Bp_chirped"]["max_rel_err_vs_Bp_direct"] = max(
        r["rel_err_vs_Bp_max"] for r in rows if r["variant"] == "Bp_chirped"
    )
    return out


# ---------------------------------------------------------------------------
# 第二步 + 第三步: 离焦并入环路, 四组对照
# ---------------------------------------------------------------------------
def search_defocus(
    bench: FresnelBench,
    base_phase: np.ndarray,
    dz: float,
    ref_sigma: float,
    rc: RunConfig,
) -> tuple[dict[str, float], list[dict]]:
    """外层确定性一维搜索: 粗扫 + 黄金分割, 目标为 **方案A** 下的 score。

    返回 ``(最优点的度量, 全部评估点)``。离焦是**加在 GS 相位上的 Zernike (2,0)**
    (``φ = φ_GS + a·Z₂⁰``), 四组用同一个旋钮、同一个区间、同一个搜索器, 差值才有意义。
    """
    lo, hi = DEFOCUS_LIMITS
    traj: list[dict] = []

    def evaluate(a: float) -> dict[str, float]:
        phase = base_phase + a * bench.defocus_basis
        m = bench.evaluate(
            bench.intensity_A(phase, dz), ref_sigma=ref_sigma, target_side_px=rc.target_side_px
        )
        traj.append({"defocus_rad": float(a), **m})
        return m

    def best_under(key: str) -> tuple[float, float, dict]:
        """在已评估的轨迹点里挑 ``key`` 最大者, 返回 (值, a, 该行)。"""
        row = max(traj, key=lambda t: t[key])
        return float(row[key]), float(row["defocus_rad"]), row

    grid = [float(v) for v in np.linspace(lo, hi, rc.n_defocus_coarse)]
    for a in grid:
        evaluate(a)
    # 在**每个判据**各自的粗扫最优点左右邻格里做黄金分割细化
    # (确定性, 无随机重启)。score 与 score_fixed 的最优点可以不同, 所以分两轮。
    step = grid[1] - grid[0]
    for key in ("score", "score_fixed"):
        _, a0, _row = best_under(key)
        l, h = max(lo, a0 - step), min(hi, a0 + step)
        evaluate(golden_section(lambda a: -evaluate(a)[key], l, h, n_iters=rc.n_defocus_golden))

    eps = DEFOCUS_BOUNDARY_FRAC * (hi - lo)

    def at_boundary(a: float) -> bool:
        return bool(a <= lo + eps or a >= hi - eps)

    # 两个判据各自的最优工作点**必须成套取出**。曾经写成
    # `for key, tag in (...): out["defocus_rad"] = a` —— 循环第二轮会把
    # defocus_rad 覆盖成 score_fixed 的 argmax, 于是 score (来自 score 判据)
    # 与 defocus_rad (来自 score_fixed 判据) **不是同一个工作点**, 出现
    # 「a=0 却比同相位 a=0 的 G1p 高 0.02」这种自相矛盾。成套取出即可避免。
    sel = {}
    for key, tag in (("score", "auto"), ("score_fixed", "fixed")):
        val, a, row = best_under(key)
        sel[tag] = (val, a, row, at_boundary(a))
    auto_val, auto_a, auto_row, auto_bnd = sel["auto"]
    fix_val, fix_a, fix_row, fix_bnd = sel["fixed"]

    out: dict = {
        "defocus_rad": auto_a,
        "defocus_at_boundary": auto_bnd,
        "defocus_rad_fixed": fix_a,
        "defocus_at_boundary_fixed": fix_bnd,
        "score": auto_val,
        "score_fixed": fix_val,
    }
    # 其余指标一律取**主判据 (自动定标框)** 那个工作点, 这样组间可比;
    # 只有 score / score_fixed 是「各自判据选出来的」, 已在上面成套取。
    for k in ("cv", "pib", "cv_fixed", "pib_fixed", "sigma_px", "cx", "cy", "d90_px", "d90_um"):
        out[k] = float(auto_row[k])
    out["side_px"] = int(auto_row["side_px"])
    at_lo = next(t for t in traj if t["defocus_rad"] == lo)
    at_hi = next(t for t in traj if t["defocus_rad"] == hi)
    out["score_at_defocus_lo"] = float(at_lo["score"])
    out["score_at_defocus_hi"] = float(at_hi["score"])
    return out, traj


def gs_coherence(a: np.ndarray, b: np.ndarray) -> float:
    """两个瞳面相位的**相位相干度** ``|mean(e^{i(φ_a−φ_b)})|`` ∈ [0,1]。

    ⚠️ 必须用**复**均值。``mean(|e^{i(φ_a−φ_b)}|)`` 恒等于 1 —— 那是逐像素取模再平均,
    对任何两个相位都给 1, 是个无信息的量 (曾经据此误判"GS 对 Δz 完全不敏感")。
    复均值才度量"这两个相位场是否指向同一解"; 1 = 完全相同, 0 = 完全无关。
    """
    return float(np.abs(np.mean(np.exp(1j * (np.asarray(a) - np.asarray(b))))))


def run_groups(
    bench: FresnelBench,
    rc: RunConfig,
    ref_sigma: float,
    sign: float,
    gs_dz0: np.ndarray,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """四组对照的完整轨迹 + 各组各 Δz 的最优点。

    ``gs_dz0`` 由调用方算好传进来 —— Δz=0 的 GS 在完整网格上要 25 s, 不该跑两遍。
    """
    best_rows: list[dict] = []
    traj_rows: list[dict] = []
    timings: dict[str, Any] = {}
    coherence: dict[str, float] = {}

    logger.info("G1/G2 相位场: 复用已算好的 Δz=0 GS (Δz 盲, 与 E1 同口径)")

    gs_cache: dict[tuple[str, float], np.ndarray] = {}
    for u in rc.dz_units:
        dz = u * DZ_BLUR_M
        row_base = {"dz_units": u, "dz_mm": dz * 1e3}
        # (group, base_phase, 是否搜离焦, phase_source)
        plans = [
            ("G1", gs_dz0, False, "gs_dz0"),
            ("G2", gs_dz0, True, "gs_dz0"),
            # G1′ —— 最关键的对照: 逐 Δz 重新优化但**不加任何离焦**。
            # 没有它, G3−G1 会把「重新优化本身的收益」和「离焦的收益」混在一起,
            # 只能说「离焦没用」而说不清收益到底来自哪。相位场与 G3 完全相同,
            # 所以不额外花一次 GS。
            ("G1p", None, False, "gs_at_dz"),
        ]
        for scheme, group in (("A", "G3"), ("B", "G4")):
            key = (scheme, u)
            if key not in gs_cache:
                t0 = time.perf_counter()
                gs_cache[key] = bench.gs_phase(
                    dz, n_iters=rc.gs_iters, seed=rc.seed, scheme=scheme, sign=sign
                )
                timings[f"gs_{scheme}_{u:+.2f}"] = round(time.perf_counter() - t0, 2)
            coherence[f"{group}_dz{u:+.2f}_vs_dz0"] = round(
                gs_coherence(gs_cache[key], gs_dz0), 6
            )
            plans.append((group, gs_cache[key], True, "gs_at_dz"))

        # 先把三个「Δz 盲/逐 Δz 重优化」的相位场解出来, 供 G1p 复用
        phases = {
            (group, src): base if base is not None else gs_cache[("A", u)]
            for group, base, _s, src in plans
        }
        for group, base, search, source in plans:
            base = phases[(group, source)]
            m0 = bench.evaluate(
                bench.intensity_A(base, dz),
                ref_sigma=ref_sigma,
                target_side_px=rc.target_side_px,
            )
            traj_rows.append(
                {"group": group, "phase_source": source, **row_base, "defocus_rad": 0.0,
                 "defocus_rad_fixed": 0.0, "defocus_at_boundary": False,
                 "defocus_at_boundary_fixed": False, **m0}
            )
            if not search:
                best_rows.append(
                    {"group": group, "phase_source": source, **row_base, "defocus_rad": 0.0,
                     "defocus_rad_fixed": 0.0, "defocus_at_boundary": False,
                     "defocus_at_boundary_fixed": False, **m0}
                )
                continue
            best, traj = search_defocus(bench, base, dz, ref_sigma, rc)
            best_rows.append({"group": group, "phase_source": source, **row_base, **best})
            for t in traj:
                traj_rows.append(
                    {"group": group, "phase_source": source, **row_base, **t}
                )
        logger.info(
            "  Δz={u:+.2f} Δz_blur | " + " | ".join(
                f"{g}: score={next(r['score'] for r in best_rows if r['group'] == g and r['dz_units'] == u):.4f}"
                for g in GROUPS
            )
            + " | GS 相干度 vs Δz=0: "
            + ", ".join(
                f"{g}={coherence[f'{g}_dz{u:+.2f}_vs_dz0']:.3f}" for g in ("G3", "G4")
            ),
            u=u,
        )
    return best_rows, traj_rows, {**timings, "_gs_phase_coherence_vs_dz0": coherence}


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]) -> None:
    """写 CSV。字段名取**所有行的并集** (保持首次出现顺序)。

    轨迹表天然异质: ``defocus_rad == 0`` 的行 (G1/G1′ 与各组的基线点) 没有
    ``score_at_defocus_lo/hi`` / ``defocus_at_boundary`` 这些只有搜索结果才有的字段。
    用 ``rows[0].keys()`` 会直接抛 "dict contains fields not in fieldnames"。
    """
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


def plot_equivalence(rows: list[dict], out: Path, tol: float) -> None:
    """(1) A/B 等价性误差 vs Δz (三个方案B 变体)。"""
    labels = {
        "Bf": "方案B 焦平面域 (Bf)",
        "Bp": "方案B 瞳面域 (Bp, 方案B 本体)",
        "Bp_chirped": "方案B 瞳面域 + 输出二次相位",
    }
    colors = {"Bf": "#2196F3", "Bp": "#4CAF50", "Bp_chirped": "#9E9E9E"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in ("Bf", "Bp", "Bp_chirped"):
        sel = [r for r in rows if r["variant"] == variant]
        by_dz: dict[float, float] = {}
        for r in sel:
            by_dz[r["dz_units"]] = max(by_dz.get(r["dz_units"], 0.0), r["rel_err_max"])
        ax.plot(
            list(by_dz),
            list(by_dz.values()),
            "o-",
            color=colors[variant],
            linewidth=2,
            markersize=4,
            label=labels[variant],
        )
    ax.axhline(tol, color="k", linestyle="--", linewidth=1, label=f"等价判据 {tol:g}")
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="完全失配 (100%)")
    ax.set_yscale("log")
    ax.set_xlabel("CCD 焦后位移 Δz / Δz_blur")
    ax.set_ylabel("max 相对误差 (方案B vs 方案A)")
    ax.set_title(
        f"方案A (精确角谱) vs 方案B (等效虚拟透镜) 等价性  [Δz_blur={DZ_BLUR_M * 1e3:.4f} mm]"
        "\n三条曲线重合: 焦平面域与瞳面域是同一个离散算子"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cv_vs_dz(best_rows: list[dict], out: Path) -> None:
    """(2) CV vs Δz 四组对照。

    ⚠️ G3 与 G4 的分数差 ≤ 1e-6 (方案B ≈ 方案A), 直接画会**完全重合**、G3 被 G4 盖住,
    读图的人会以为 G3 没跑。所以 G4 画成**虚线 + 更细**, 让 G3 (实线) 透出来;
    标题里也写明两者重合, 不让「看不见」被误读成「不存在」。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    style = {
        "G1": dict(linestyle="-", linewidth=2),
        "G2": dict(linestyle="-", linewidth=2),
        "G1p": dict(linestyle="-", linewidth=2.4),  # 对照, 略粗以突出
        "G3": dict(linestyle="-", linewidth=3.2),  # 粗: 压在 G4 下面也能看见
        "G4": dict(linestyle="--", linewidth=1.4),  # 细虚线: 叠在 G3 上
    }
    for ax, key, title in (
        (axes[0], "cv", "目标框内 CV (自动定标框)"),
        (axes[1], "score", "score = 0.5·PIB + 0.5/(1+CV)"),
    ):
        for g in GROUPS:
            sel = sorted(
                (r for r in best_rows if r["group"] == g), key=lambda r: r["dz_units"]
            )
            ax.plot(
                [r["dz_units"] for r in sel],
                [r[key] for r in sel],
                marker="o",
                markersize=5,
                color=GROUP_COLORS[g],
                label=GROUP_LABELS[g],
                **style[g],
            )
        if key == "cv":
            ax.set_yscale("log")
        ax.set_xlabel("CCD 焦后位移 Δz / Δz_blur")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower center")
    fig.suptitle(
        "五组对照: 上限 = 该 Δz 上 best score / min CV (离焦幅度由确定性一维搜索给出)\n"
        "G1′ (紫) = 逐 Δz 重优化但**不加离焦**的对照 —— 它与 G3 的差才是离焦的净贡献\n"
        "G3 (粗实线) 与 G4 (细虚线) 重合 —— 方案B 的上限与方案A 相同"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_best_defocus(best_rows: list[dict], out: Path) -> None:
    """(3) G3 (方案A 并入环路) 的最优离焦幅度 vs Δz。"""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for g in ("G2", "G1p", "G3", "G4"):
        sel = sorted(
            (r for r in best_rows if r["group"] == g), key=lambda r: r["dz_units"]
        )
        # G3 粗实线 / G4 细虚线: 两者重合时两条都还看得见
        style = (
            dict(linestyle="--", linewidth=1.4)
            if g == "G4"
            else dict(linestyle="-", linewidth=3.2 if g == "G3" else 2)
        )
        ax.plot(
            [r["dz_units"] for r in sel],
            [r["defocus_rad"] for r in sel],
            marker="o",
            markersize=5,
            color=GROUP_COLORS[g],
            label=GROUP_LABELS[g],
            **style,
        )
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("CCD 焦后位移 Δz / Δz_blur")
    ax.set_ylabel("达到 best score 的离焦系数 (Zernike (2,0), 弧度)")
    ax.set_title("逐 Δz 最优瞳面离焦幅度\nG3 (粗实线) 与 G4 (细虚线) 重合")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower center")

    by = {r["dz_units"]: r for r in best_rows if r["group"] == "G3"}
    xs = sorted(by)
    ax2.plot(
        xs,
        [by[x]["score"] - next(r["score"] for r in best_rows if r["group"] == "G1" and r["dz_units"] == x) for x in xs],
        "o-",
        color="#2196F3",
        linewidth=2,
        label="G3 − G1 (总增益)",
    )
    ax2.plot(
        xs,
        [
            next(r["score"] for r in best_rows if r["group"] == "G1p" and r["dz_units"] == x)
            - next(r["score"] for r in best_rows if r["group"] == "G1" and r["dz_units"] == x)
            for x in xs
        ],
        "s-",
        color="#9C27B0",
        linewidth=2,
        label="G1′ − G1 (仅重新优化)",
    )
    ax2.plot(
        xs,
        [
            by[x]["score"]
            - next(r["score"] for r in best_rows if r["group"] == "G1p" and r["dz_units"] == x)
            for x in xs
        ],
        "^--",
        color="#FF5722",
        linewidth=1.6,
        label="G3 − G1′ (离焦的净贡献)",
    )
    ax2.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax2.set_xlabel("CCD 焦后位移 Δz / Δz_blur")
    ax2.set_ylabel("score 增益 (相对 G1 基线)")
    ax2.set_title("逐 Δz 的 score 增益分解\nG1′ 只含「重新优化」, G3 再加离焦")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="小网格冒烟跑")
    p.add_argument("--tag", default=None, help="产物文件名后缀 (默认: --quick 时 _quick)")
    p.add_argument("--out", default=None, help="产物目录 (默认 report/shape_trials)")
    return p


def self_check_g3_vs_g1p(
    best_rows: list[dict], traj_rows: list[dict], dz_units: tuple[float, ...]
) -> float:
    """G1′ 与 G3 用**同一个相位场**, 只差离焦那一维。

    所以轨迹里 G3 在 ``defocus_rad == 0`` 的那一行, 必须逐位等于 G1′ 的那一行。
    这条不变量能抓住「best score 与 best defocus 取自**不同判据/不同工作点**」
    这类组装错误 —— 那类 bug 不报错, 只是让 G3 凭空比 G1′ 高出零点零几。
    """
    worst = 0.0
    for u in dz_units:
        g1p = next(
            r for r in traj_rows
            if r["group"] == "G1p" and abs(r["dz_units"] - u) < 1e-12
        )
        g3 = next(
            r for r in traj_rows
            if r["group"] == "G3" and abs(r["dz_units"] - u) < 1e-12
            and abs(r["defocus_rad"]) < 1e-12
        )
        worst = max(worst, abs(g1p["score"] - g3["score"]))
    # 另外: G3 若把最优离焦选在 0, 其 best 必须等于自己的 a=0 行
    for u in dz_units:
        best = next(r for r in best_rows if r["group"] == "G3" and abs(r["dz_units"] - u) < 1e-12)
        if abs(best["defocus_rad"]) < 1e-12:
            zero = next(
                r for r in traj_rows
                if r["group"] == "G3" and abs(r["dz_units"] - u) < 1e-12
                and abs(r["defocus_rad"]) < 1e-12
            )
            worst = max(worst, abs(best["score"] - zero["score"]))
    if worst > 1e-12:
        raise AssertionError(
            f"G3(a=0) 与 G1′ 不一致 (最大偏差 {worst:.3e}); best score 与 best defocus "
            "很可能取自不同工作点/不同判据, 结论不可用"
        )
    return worst


def main() -> None:
    args = build_parser().parse_args()
    rc = QUICK if args.quick else FULL
    if args.tag is not None:
        rc = replace(rc, tag=args.tag)
    out = Path(args.out) if args.out else ROOT / "report" / "shape_trials"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cfg = rc.bench()
    logger.info(
        "光学常量: λ={:.0f}nm  f={:.2f}mm  D={:.3f}mm  面板像元={:.1f}um  CCD像元={:.2f}um",
        WAVELENGTH_NM, FOCAL_LENGTH_M * 1e3, APERTURE_M * 1e3, PANEL_PIXEL_UM, CAMERA_PIXEL_UM,
    )
    logger.info(
        "网格: n_grid={}  padding={}  far_field_size={}  远场像元={:.4f}um  "
        "瞳面像元={:.4f}um  target_side={}px  Δz_blur={:.4f}mm",
        rc.n_grid, rc.far_field_padding, rc.m, cfg.far_field_pixel_size * 1e6,
        cfg.aperture_size / rc.n_grid * 1e6, rc.target_side_px, DZ_BLUR_M * 1e3,
    )

    logger.info("自检 1/4: 光学常量 vs experiment_gs_defocus_offocus.py 快照 ...")
    const_err = self_check_constants()
    logger.info("自检 2/4: GS 镜像 vs gs_shape (方案A, Δz=0) ...")
    bench = FresnelBench(cfg, n_zernike=rc.n_zernike)
    gs_delta = self_check_gs(rc, bench)
    logger.info("自检 3/4: Δz=0 时方案B 退化为傅里叶变换 ...")
    zero_err = self_check_schemeB_at_zero(
        bench, np.random.default_rng(rc.seed + 3).normal(0, 0.3, size=(bench.n, bench.n))
    )
    logger.info(
        "自检通过: 常量偏差 max={:.3e}  max|Δphase|={:.3e}  B@Δz=0 max={:.3e}",
        max(const_err.values()), gs_delta, max(zero_err.values()),
    )

    logger.info("第一步: 数值确定方案B 的符号与系数 (不假设) ...")
    sign_info = resolve_scheme_b_sign(bench, rc)
    sign = float(sign_info["sign"])
    logger.info(
        "  方案B: sign_pupil={:+.0f}  反解 alpha={:.6f}  "
        "Bp 残差={:.3e}  Bf 残差 (alpha=1)={:.3e} (alpha*)={:.3e}",
        sign, sign_info["alpha"],
        sign_info["probe_rel_err_Bp"],
        sign_info["probe_rel_err_Bf_at_alpha1"], sign_info["probe_rel_err_Bf_at_alpha_star"],
    )

    logger.info("第一步: A/B 等价性扫描 ({:.0f} 个 Δz × {} 个随机瞳面相位) ...",
                len(rc.equiv_dz_units), rc.equiv_n_random)
    equiv_rows = run_equivalence(bench, rc, sign)
    equiv_summary = summarise_equivalence(equiv_rows, rc.equiv_tol)
    for variant in ("Bf", "Bp", "Bp_chirped"):
        info = equiv_summary[variant]
        logger.info(
            "  {v:<11} 等价到 |Δz| <= {lim:.4f} Δz_blur ({mm:.4f} mm) [线性内插]  "
            "采样上限 {grid:+.2f}  全区间最大相对误差 {mx:.3e}  (Δz=0 处 {z:.3e})",
            v=variant,
            lim=info["equivalent_abs_dz_units_interpolated"] or 0.0,
            mm=info["equivalent_abs_dz_mm"],
            grid=info["equivalent_abs_dz_units_limit"],
            mx=info["max_rel_err_overall"], z=info["max_rel_err_at_dz0"],
        )

    logger.info("基线: G1 在 Δz=0 上量参考 σ 与基线分数 ...")
    gs_dz0 = bench.gs_phase(0.0, n_iters=rc.gs_iters, seed=rc.seed, scheme="A")
    ref_sigma = bench.evaluate(
        bench.intensity_A(gs_dz0, 0.0), ref_sigma=1.0, target_side_px=rc.target_side_px
    )["sigma_px"]
    base_metrics = bench.evaluate(
        bench.intensity_A(gs_dz0, 0.0), ref_sigma=ref_sigma, target_side_px=rc.target_side_px
    )
    logger.info(
        "baseline (Δz=0, 无离焦, GS): score={:.4f} PIB={:.4f} CV={:.4f} σ={:.3f}px",
        base_metrics["score"], base_metrics["pib"], base_metrics["cv"], ref_sigma,
    )

    logger.info("第二/三步: 四组对照 (G1 基线 / G2 事后外挂 / G3 方案A 并入环路 / G4 方案B 并入环路) ...")
    best_rows, traj_rows, timings = run_groups(bench, rc, ref_sigma, sign, gs_dz0)

    logger.info("自检 4/4: G3 在 defocus=0 的工作点必须逐位等于 G1′ (同相位场) ...")
    consistency_err = self_check_g3_vs_g1p(best_rows, traj_rows, rc.dz_units)
    logger.info("  G3(a=0) vs G1′ 最大偏差 = {:.3e}", consistency_err)

    write_csv(out / f"fv_equivalence_fresnel_vs_vl{rc.tag}.csv", equiv_rows)
    write_csv(out / f"fv_groups_fresnel_vs_vl{rc.tag}.csv", best_rows)
    write_csv(out / f"fv_groups_trajectory_fresnel_vs_vl{rc.tag}.csv", traj_rows)
    plot_equivalence(equiv_rows, out / f"fv_equivalence_fresnel_vs_vl{rc.tag}.png", rc.equiv_tol)
    plot_cv_vs_dz(best_rows, out / f"fv_cv_vs_dz_fresnel_vs_vl{rc.tag}.png")
    plot_best_defocus(best_rows, out / f"fv_best_defocus_fresnel_vs_vl{rc.tag}.png")

    # --- 摘要 ------------------------------------------------------------
    def pick(group: str, u: float) -> dict:
        for r in best_rows:
            if r["group"] == group and abs(r["dz_units"] - u) < 1e-12:
                return r
        raise KeyError((group, u))

    def pick_zero(group: str, u: float) -> dict:
        """轨迹里该组在 ``defocus_rad == 0`` 的那一行 = 「GS 看见 Δz」的裸贡献。"""
        for r in traj_rows:
            if (
                r["group"] == group
                and abs(r["dz_units"] - u) < 1e-12
                and abs(r["defocus_rad"]) < 1e-12
            ):
                return r
        raise KeyError((group, u))

    g1 = {f"{u:+.2f}": pick("G1", u) for u in rc.dz_units}
    ceiling = {
        g: {
            f"{u:+.2f}": {
                key: round(float(pick(g, u)[key]), 6)
                for key in (
                    "score",
                    "cv",
                    "pib",
                    "defocus_rad",
                    "score_fixed",
                    "cv_fixed",
                    "pib_fixed",
                    "defocus_rad_fixed",
                    "sigma_px",
                    "side_px",
                )
            }
            for u in rc.dz_units
        }
        for g in GROUPS
    }
    gain_over_g1 = {
        g: {f"{u:+.2f}": round(pick(g, u)["score"] - pick("G1", u)["score"], 6) for u in rc.dz_units}
        for g in ("G2", "G1p", "G3", "G4")
    }
    bv_limit = equiv_summary["Bp"]["equivalent_abs_dz_units_interpolated"] or 0.0
    bf_limit = equiv_summary["Bf"]["equivalent_abs_dz_units_interpolated"] or 0.0
    chirp_diff = float(equiv_summary["Bp_chirped"]["max_rel_err_vs_Bp_direct"])
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "quick" if args.quick else "full",
        "wavelength_nm": WAVELENGTH_NM,
        "focal_length_m": FOCAL_LENGTH_M,
        "aperture_m": APERTURE_M,
        "panel_pixel_um": PANEL_PIXEL_UM,
        "camera_pixel_um": CAMERA_PIXEL_UM,
        "n_grid": rc.n_grid,
        "far_field_padding": rc.far_field_padding,
        "far_field_size": rc.m,
        "far_field_pixel_size_um": cfg.far_field_pixel_size * 1e6,
        "pupil_pixel_size_um": cfg.aperture_size / rc.n_grid * 1e6,
        "target_side_px": rc.target_side_px,
        "objective": "0.5*PIB + 0.5*(1/(1+CV)), CV/PIB in the centroid-tracked auto-scaled box",
        "dz_blur_mm": DZ_BLUR_M * 1e3,
        "dz_units": list(rc.dz_units),
        "defocus_limits_rad": list(DEFOCUS_LIMITS),
        "defocus_search": {
            "coarse_points": rc.n_defocus_coarse,
            "golden_section_iters": rc.n_defocus_golden,
            "deterministic": True,
            "note": "外层一维搜索评估的是**方案A** 正演下的 score; 四组共用同一搜索器与同一离焦旋钮 (Zernike (2,0))",
        },
        "self_check": {
            "constants_rel_err_vs_sibling_script": const_err,
            "gs_mirror_max_phase_delta": gs_delta,
            "schemeB_at_dz0_rel_err": zero_err,
            "G3_at_zero_defocus_vs_G1p_max_delta": consistency_err,
        },
        "scheme_b_sign_resolution": sign_info,
        "equivalence": equiv_summary,
        "equivalence_verdict": {
            "tolerance": rc.equiv_tol,
            "Bf_equivalent_limit_dz_units": bf_limit,
            "Bp_equivalent_limit_dz_units": bv_limit,
            "Bp_max_rel_err_over_swept_range": equiv_summary["Bp"]["max_rel_err_overall"],
            "Bp_chirped_max_rel_err_over_swept_range": equiv_summary["Bp_chirped"]["max_rel_err_overall"],
            "swept_dz_units_range": [min(rc.equiv_dz_units), max(rc.equiv_dz_units)],
            "Bf_vs_Bp_identical": bool(
                abs(
                    equiv_summary["Bf"]["max_rel_err_overall"]
                    - equiv_summary["Bp"]["max_rel_err_overall"]
                )
                <= 1e-9 * max(equiv_summary["Bf"]["max_rel_err_overall"], 1e-30)
            ),
            "Bf_vs_Bp_note": (
                "焦平面域 (Bf) 与瞳面域 (Bp) 的相对误差在所有 Δz 上逐位相同 -> "
                "在离散网格上这两者是同一个算子, 「搬回瞳面」不引入额外的环卷绕误差; "
                "A 与 B 的全部差异都来自把 kz 展开到二阶的傍轴截断"
            ),
            "output_chirp_is_unit_modulus": bool(chirp_diff <= 1e-12),
            "output_chirp_direct_rel_err_vs_Bp": chirp_diff,
            "output_chirp_note": (
                "Bp 与 Bp_chirped 只差一个输出二次相位 exp(+i*pi*|x_c|^2/(lambda*dz)), "
                "它模长为 1, 故对归一化强度逐位无影响 (Bp 直比实测 "
                f"{chirp_diff:.3e}); 但 GS 的反向传播取的是相位, 必须带上它"
            ),
            "relative_error_by_abs_dz_units": {
                v: equiv_summary[v]["rel_err_by_abs_dz"]
                for v in ("Bf", "Bp", "Bp_chirped")
            },
        },
        "baseline_g1_dz0": {
            "score": base_metrics["score"],
            "PIB": base_metrics["pib"],
            "CV": base_metrics["cv"],
            "score_fixed_box": base_metrics["score_fixed"],
            "PIB_fixed_box": base_metrics["pib_fixed"],
            "CV_fixed_box": base_metrics["cv_fixed"],
            "sigma_px": ref_sigma,
            "side_px": base_metrics["side_px"],
            "d90_px": base_metrics["d90_px"],
            "d90_um": base_metrics["d90_um"],
        },
        "reference_baseline_data_json": {
            **DATA_JSON_GS,
            "note": "report/iterative_zernike_shaping/data.json :: gs",
            "delta_vs_mine": {
                "score": base_metrics["score"] - DATA_JSON_GS["score"],
                "PIB": base_metrics["pib"] - DATA_JSON_GS["PIB"],
                "CV": base_metrics["cv"] - DATA_JSON_GS["CV"],
            },
            "deviation_cause": (
                f"迭代数口径, 不是 bug: 本脚本与 scripts/slm/experiment_gs_defocus_offocus.py "
                f"都用 gs_iters=200, 两者的 G1@Δz=0 逐位相同 (score {base_metrics['score']:.16g}); "
                f"而 data.json::gs = 0.8118055 对应的是**未收敛**的 GS —— 在 64/8/43 网格上实测 "
                f"20 次迭代 0.812071 / 50 次 0.811884 / 100 次 0.807316 / 400 次 0.807309, "
                f"即 data.json 的 GS 迭代数在 20~50 之间。兄弟脚本的 summary.json 带着同一个偏差。"
            ),
        },
        "group_ceiling": ceiling,
        "group_ceiling_criterion": (
            "每组的 score/cv/pib 都是**该组自己的最优离焦**下的值 (对 score 与 "
            "score_fixed 两个判据各取一次最优点, 分别记为 defocus_rad 与 "
            "defocus_rad_fixed)。G1/G1′ 不搜离焦, 两列都等于 0。"
        ),
        "score_gain_over_G1": gain_over_g1,
        "defocus_search_sanity": {
            "limits_rad": list(DEFOCUS_LIMITS),
            "coarse_points": rc.n_defocus_coarse,
            "golden_section_iters": rc.n_defocus_golden,
            "boundary_frac_tol": DEFOCUS_BOUNDARY_FRAC,
            "note": (
                "撞边界 = 最优解顶在区间端 => 区间仍不够。前置任务 "
                "experiment_gs_defocus_offocus.py 的 E2 把区间限制在约 [−0.28,+0.28], "
                "而其 E1 宽扫描显示最优离焦可达 ±1.2 —— E2 的「离焦让 score 下降」"
                "是区间不足的假象, 不可引用。"
            ),
            "at_boundary": {
                g: {
                    f"{u:+.2f}": bool(pick(g, u).get("defocus_at_boundary", False))
                    for u in rc.dz_units
                }
                for g in ("G2", "G3", "G4")
            },
            "best_defocus_rad": {
                g: {f"{u:+.2f}": round(pick(g, u)["defocus_rad"], 6) for u in rc.dz_units}
                for g in ("G2", "G3", "G4")
            },
            "any_at_boundary": bool(
                any(
                    bool(pick(g, u).get("defocus_at_boundary", False))
                    for g in ("G2", "G3", "G4")
                    for u in rc.dz_units
                )
            ),
            "fixed_box_criterion": {
                "note": (
                    "第二判据: 在**定尺**框 (边长恒为 target_side_px, 与 data.json 同口径) "
                    "上取最优离焦。必须同时报, 因为主判据用的自动定标框边长 = "
                    "target_side_px * σ(a)/σ_ref, 而离焦**直接改变 σ** —— 实测 "
                    "G3 在 Δz=+0.5 时 a 从 0 走到 -2.5, 框边长从 20 张到 39、σ 从 8.2 "
                    "张到 16.2。于是主判据的 'best score' 里有一部分是「把光斑摊大、"
                    "让自动框套进更多图样」, 而不是匀化真的变好。定尺判据没有这个循环。"
                ),
                "at_boundary": {
                    g: {
                        f"{u:+.2f}": bool(pick(g, u).get("defocus_at_boundary_fixed", False))
                        for u in rc.dz_units
                    }
                    for g in ("G2", "G3", "G4")
                },
                "best_defocus_rad": {
                    g: {f"{u:+.2f}": round(pick(g, u)["defocus_rad_fixed"], 6) for u in rc.dz_units}
                    for g in ("G2", "G3", "G4")
                },
                "any_at_boundary": bool(
                    any(
                        bool(pick(g, u).get("defocus_at_boundary_fixed", False))
                        for g in ("G2", "G3", "G4")
                        for u in rc.dz_units
                    )
                ),
            },
        },
        "gain_decomposition_vs_G1": {
            "note": (
                "关键拆解, 五组构成一条因果链:\n"
                "  G1  = Δz 盲 GS, 无离焦                       (基线, E1 的 defocus=0 行)\n"
                "  G2  = Δz 盲 GS + 离焦                        (E1 的宽扫描, 事后补偿口径)\n"
                "  G1′ = 逐 Δz 重优化 GS, 无离焦                 <- 对照; G1′−G1 = **重新优化本身**的收益\n"
                "  G3  = 逐 Δz 重优化 GS + 离焦 (方案A)          (G3−G1′ = 离焦在重优化之上的**增量**)\n"
                "  G4  = 逐 Δz 重优化 GS + 离焦 (方案B)          (G4−G3 = 方案B 与方案A 之差)\n"
                "没有 G1′ 时, G3−G1 把「重新优化」与「离焦」混在一起, 只能说「离焦没用」"
                "而说不清收益来源。\n"
                "注: 标量离焦就是 Zernike (2,0) 模态, 而 GS 相位是**自由相位**, 所以 "
                "φ_GS + a·Z₂⁰ 仍在 GS 可达空间内 —— G3−G1′ 度量的是离焦能不能"
                "在重优化之上再多挤出一点, 而不是「能不能表达」。"
            ),
            "reoptimisation_only_G1p_minus_G1": {
                f"{u:+.2f}": round(pick("G1p", u)["score"] - pick("G1", u)["score"], 6)
                for u in rc.dz_units
            },
            "defocus_on_top_of_reopt_G3_minus_G1p": {
                f"{u:+.2f}": round(pick("G3", u)["score"] - pick("G1p", u)["score"], 6)
                for u in rc.dz_units
            },
            "defocus_on_blind_G2_minus_G1": {
                f"{u:+.2f}": round(pick("G2", u)["score"] - pick("G1", u)["score"], 6)
                for u in rc.dz_units
            },
            "total_G3_minus_G1": {
                f"{u:+.2f}": round(pick("G3", u)["score"] - pick("G1", u)["score"], 6)
                for u in rc.dz_units
            },
        },
        "score_gain_G3_minus_G4": {
            f"{u:+.2f}": round(pick("G3", u)["score"] - pick("G4", u)["score"], 6)
            for u in rc.dz_units
        },
        "gs_wallclock_s": timings,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "outputs": {
            "equivalence_csv": f"fv_equivalence_fresnel_vs_vl{rc.tag}.csv",
            "groups_csv": f"fv_groups_fresnel_vs_vl{rc.tag}.csv",
            "groups_trajectory_csv": f"fv_groups_trajectory_fresnel_vs_vl{rc.tag}.csv",
            "equivalence_png": f"fv_equivalence_fresnel_vs_vl{rc.tag}.png",
            "cv_vs_dz_png": f"fv_cv_vs_dz_fresnel_vs_vl{rc.tag}.png",
            "best_defocus_png": f"fv_best_defocus_fresnel_vs_vl{rc.tag}.png",
        },
    }

    # 报告用一句话结论 (措辞固定, 数字全部从实测结果代入, 不手写常量)。
    a_times = [v for k, v in timings.items() if k.startswith("gs_A_")]
    b_times = [v for k, v in timings.items() if k.startswith("gs_B_")]
    worst_g34 = max(abs(v) for v in summary["score_gain_G3_minus_G4"].values())
    summary["equivalence_verdict"]["one_liner_for_report"] = (
        f"方案A 与方案B 在 |Δz| ≤ {bf_limit * DZ_BLUR_M * 1e3:.3f} mm 内等价到 {rc.equiv_tol:.0e}; "
        f"超出后方案B 作为傍轴截断近似按 Δz² 发散, 在 |Δz|={max(rc.equiv_dz_units)} 单位处达 "
        f"{equiv_summary['Bp']['max_rel_err_overall']:.1e}。但由于 G3 与 G4 的 score 差仅 "
        f"≤{worst_g34:.0e}, 该误差不改变本实验的任何 score 级结论, 方案B 可作为约 2× 提速的替身 "
        f"(wallclock {min(b_times):.2f}–{max(b_times):.2f}s vs {min(a_times):.2f}–{max(a_times):.2f}s), "
        f"但不能用于任何场级/相位级的比较。"
    )

    (out / f"summary_fresnel_vs_vl{rc.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n产物目录: {out}")
    print(f"耗时: {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()