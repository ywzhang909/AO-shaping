"""零级背景 + 自适应加权 GS: 在**双分量瞳面模型**下重扫 CCD 焦后位移 Δz。

要回答的问题
------------

现有的 2f 整形模型把瞳面写成 ``field = gauss · exp(iφ)`` —— 振幅被钳死, **零级没有
自由度**。真实 SLM 上总有调制深度不足 / 像素死区 / 未调制区域带来的一部分光原样
透过 (``η``), 它只经过物理透镜, 在 ``f + d`` 处散成一团晕斑::

    U_pupil = U_inc · [ η + (1 - η) · exp(iφ) ]

本文件把 η 做成显式参数并扫 Δz × η, 对照五个方案:

======  ==================================================  =========================
 组     说明
======  ==================================================  =========================
 H0     plain GS, η=0 —— 复现现有基线 (Δz=0 处必须对上 data.json 的 gs 行)
 H1     plain GS, η>0, **不做**相干扣除
 H2     plain GS, η>0, **做**相干扣除
 H3     加权 GS, η>0, 做相干扣除
 H4     H3 + 小倾斜把整形光斑连同目标框一起移开轴上的零级晕斑
======  ==================================================  =========================

模型要点 (照做, 不自行改动光学)
------------------------------

* 正演一律**方案 A**: ``U_CCD = ASM(FT(U_pupil), d)`` —— 透镜先做傅里叶变换,
  再对**已变换的场**自由传播 d。**没有放大率**: Collins/ABCD 给出 B=f 恒定, 所以
  远场像元 pitch 恒为 ``λf/(N·dx_pupil)``, **不**按 ``(f+d)/f`` 重采样目标。
* 瞳面二次相位 (傍轴等价式) 是 ``exp(−iπ·d·r²/(λf²))``, 分母是 **λf²**。
* 零级分量 ``η·U_inc`` 与信号光走**同一个** ``focal()`` + ``_asm()`` 算子 —— 它
  只是不承载 ``exp(iφ)``, 但同样吃那一次焦后传播 (FT 与 ASM 都是线性的, 所以把
  两分量写进同一个瞳面复场再走一遍链, 零级自动获得同样的传播)。
* 晕斑峰值应当满足 ``I_halo ∝ f²/d²`` (Δz 越大越弱越散)。脚本启动时自检这一点,
  不满足直接抛错。
* **瞳面约束写成 SLM 真实流形**, 不是 ``|field| = amp``: 把回代场分解出信号相量
  ``e^{iφ} = (field/amp − η)/(1 − η)``, 再重建 ``amp·(η + (1−η)e^{iφ})``。η=0 时
  它逐位退化成 plain GS (口径内 ``max|Δphase| = 5.8e-15``, 启动时自检)。

H4 的做法 (「小倾斜移开晕斑」怎么落地)
----------------------------------------

零级分量**不承载** ``exp(iφ)``, 所以给 SLM 加线性相位倾斜只会挪动**整形光斑**,
零级晕斑仍留在光轴上。H4 因此反过来做: 把**目标框**整体平移出晕斑半径, GS 的解
自然带一个对应的线性相位倾斜 —— 整形光斑落到离轴的框里, 轴上的晕斑留在框外。
所需瞳孔倾斜幅度 ``θ_rad = 2π·Δx/(λf)`` (Δx 为远场位移), 每行的 ``offset_px``
与 ``tilt_fits_window`` 都进 CSV; 半窗放不下时位移**封顶**并把该行标成 False,
不在结果里假装成功。H4 的靶区改用**定尺框**度量 (不再跟着强度质心跑), 否则远处
的晕斑会把框撑大、把质心拽偏 —— 那是测量口径的锅, 不是 H4 的。

度量口径 (2026-10-19 修正, 三层并列)
------------------------------------

1. **主口径 = 固定物理尺度框** (``score``/``cv``/``pib`` 及 ``*_cont``/``*_bl``):
   边长恒为 ``target_side_px``, 框心用**窗口化质心**在信号上定位一次后**冻结**
   (:func:`anchor_box_center`), 同一口径的所有帧共用。**结论只准用这一层。**
2. **自适应框** (``*_adapt``): 质心跟踪 + σ 缩放边长。**只作对照** —— 框跟着光斑
   一起变大会把「测量区域变大」当成「均匀性变好」(本调查里这个陷阱出现过三次)。
3. **data.json 居中定尺框** (``*_box``): 不锚在信号上 (H4 的晕斑就在它中心),
   所以它**对任何口径都恒可计算、永不 NaN**; Δz=0/η=0 上逐位复现基线。

⚠️ **上一版的扫描路径 bug (已修, 且加了永久断言)**: ``_row`` 曾经一律用 H4 的倾斜
偏移定位度量框, 于是 H0~H3 (无倾斜) 的框被整体挪出信号 —— Δz=0 时挪 23 px, 框里
剩不到 0.4% 的光, PIB 从 **0.9153** 塌到 **0.0034** (270×); 同时 ``pib_box``/
``score_box`` 在非零偏移分支里被写成 ``NaN``。修复 = 每行用自己的口径
(:attr:`GsSetup.offset_px`) + 框心锚在信号上。永久断言 = :func:`self_check_scan_baseline`
走**扫描自己的代码路径**在 Δz=0/η=0 上核对 data.json, 容差 1e-6, 不成立直接 raise。

⚠️ **回代算子在大 Δz 下不闭合** (继承自 :class:`OffFocusBench`): 带限 ASM 不是酉
算子, ``_asm_inverse`` 用 ``H(−dz)=conj(H(dz))`` 代替伴随, 只在 Δz=0 精确往返。
闭合误差逐 Δz 量出来记进 ``summary.self_check.adjoint_closure`` —— 它对 H1~H4 一视
同仁, 组间比较仍公平, 但「扣除后的回代」在大 Δz 上要打折看。

复用仓库已验证件 (不自写传播/GS/Zernike)
----------------------------------------

* 传播: :func:`~ao_shaping.algorithm.signal_processing.angular_spectrum.angular_spectrum_propagate`
* 正向模型 / GS 骨架 / 光学常量 / 度量: **import 复用**
  :mod:`scripts.slm.experiment_gs_defocus_offocus` 的 :class:`OffFocusBench`
  (本文件只 :meth:`TwoComponentBench.gs_phase_eta` 重写了投影链, 其余全部继承)。
* 加权 / HIO 的现成参数与更新式: :class:`~ao_shaping.algorithm.signal_processing.square_shaping.UniformSquareShaper`
  的 ``_weighted_step`` —— 本文件照抄那 4 行公式 (``gs_gamma`` / ``gs_weight_clip``
  的用法与之逐位一致), 参数默认值取自
  :class:`~ao_shaping.optimizer.wfless.slm_gs_refine.SlmGsRefineConfig`
  (``gs_gamma=0.5``, ``gs_weight_clip=4.0``, ``gs_beta=0.9``, ``gs_hio_fraction=0.7``)。
* 目标 / 度量: :func:`~ao_shaping.algorithm.signal_processing.square_shaping.make_square_target`
  + :func:`~ao_shaping.algorithm.signal_processing.square_shaping.compute_bench_metrics`。

⚠️ **带限 ASM 在大 Δz 上让 GS 内环失效** (权威件的默认值, 本实验把它关掉)
--------------------------------------------------------------------------

``angular_spectrum_propagate`` 默认 ``band_limit=True`` (Voelz & Roggemann cutoff)。
那个 cut 在焦平面上**有损且不可逆**: 高频 bin 被置零, 能量在**正向一步**就漏掉
(Δz=50 mm 实测漏 0.7%~22%, 随入射场谱宽而变), 反向一步补不回来 —— 于是
``fwd(+z)`` 与 ``bwd(−z)`` **不构成伴随对**, 交替投影解的不是自洽问题。

往返误差 ``max|u − bwd(fwd(u))| / max|u|`` (二次相位瞳面场, 逐 Δz):

=================  =======  =======  =======  =======
``band_limit``     0 mm     10 mm    30 mm    50 mm
=================  =======  =======  =======  =======
``True`` (权威默认) 8e-16   2.4e-01  8.6e-01  9.4e-01
``False`` (本实验)  8e-16   6.5e-16  6.8e-16  6.8e-16
=================  =======  =======  =======  =======

⚠️ **上表的数字不能单独引用。** 破坏程度**强依赖场的谱内容**: cutoff
``f_max = 1/(λ√(1+(2z/B)²))`` 随 z 收紧, 宽带场先被削掉的远多于准单频场。同一个
传播器、同一个 Δz=50 mm, 实测能量损失从 **0.7%** (准单频) 到 **80%** (宽带)
都有。``measure_band_limit_damage`` 用两个探针 (本实验实际场 / 宽带随机相位场)
把这个跨度**测出来**写进 summary —— 换实验系统请重测, 别抄本文件的数字。

**不改 ``src/``** —— 权威件那个默认值有物理理由。只在本实验里把它变成可配参数
(:class:`TwoComponentBench` 的 ``band_limit``), **Δz>0 一律关**: 本实验远场网格
已零填充 (padding=8 ⇒ 512²/2048²), 残余混叠可控; 而关掉后传递函数模长恒为 1、
传播可逆、内环自洽。Δz=0 时开与关**逐位相同** (``exp(i·kz·0)=1`` 且 cut 全通),
所以基线与 ``data.json`` 的核对不受影响。启动时的 ``round_trip_relerr`` 自检
(≤1e-12, Δz=0..50 mm) 会挡住同类问题再溜进来; ``--band-limit`` 可复现带限那一版,
每行 CSV 的 ``*_bl`` 列给出同一组解在两种正演下的指标差 —— **那个差就是数值
误差带**, 组间差异若落在带子里就不算差异。

⚠️ **场窗口是被钉死的 —— padding 救不了回绕**
-------------------------------------------

``far_field_pixel = λf/(D·padding)`` 只随 padding 变, 而窗口宽
``W = n_grid·λf/D`` **只随 n_grid 变**。加大 padding 只是把像元切细、把网格变大,
**窗口一动不动** —— 实测 ``n_grid=64`` 下 padding=8 (m=512) 与 padding=32
(m=2048) 的半窗都是 **578.8 µm**, 晕斑回绕阈值完全一样。要装下更远的晕斑只能
**加瞳面过采样 n_grid**: 128→半窗 1158 µm, 256→2315 µm, 512→4631 µm。

晕斑 1/e² 半径 (焦平面腰 ``λf/(πw0) = 20.1 µm``, z_R = 1.49 mm) 在整形网格半窗
(578.8 µm) 上占的比例: Δz=0/5/10/20/30/40/50 mm → 0.04/0.15/**0.29**/**0.58**/
0.84/1.01/1.10。超过 ``HALO_WRAP_RATIO_LIMIT = 0.5`` 的点被标记 ``trusted=False``
并在校验区间图上涂红 —— 那些 CV/PIB 是 **FFT 周期性把边缘能量从对面卷回靶区**的
数值伪影, 不是「能量掉出探测器」的物理。``trustworthy_dz`` 块逐 Δz 给出该比例,
并算出覆盖当前最大 Δz 所需的 ``n_grid``。

所以晕斑峰值用一张**更宽的网格** (halo 网格, 只加瞳面过采样, 远场像元 pitch 一根
标尺不变) 单独量, 整形扫描仍在 64 网格上跑 (H0 才能对上 data.json)。两件事都记进
summary。

运行
----

.. code-block:: bash

    python scripts/slm/experiment_gs_improved.py --quick    # 小网格冒烟 (含 data.json 基线核对)
    python scripts/slm/experiment_gs_improved.py             # 完整扫描

产物 (全部带 ``_improved`` 后缀) 写到 ``report/shape_trials/``:
CSV / summary.json + 四张图。纯仿真: 不打开任何设备。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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
    compute_bench_metrics,
    composite_from_pib_cv,
    composite_score,
    gs_shape,
    make_square_target,
)
from ao_shaping.optimizer.wfless import slm_gs_refine as gs_refine
from ao_shaping.utils.image.beam_metrics import measure_spot_diameter_cam

# ---------------------------------------------------------------------------
# 复用兄弟实验的正向模型 / 光学常量 (不重写)
# ---------------------------------------------------------------------------
def _load_sibling() -> Any:
    """import ``scripts/slm/experiment_gs_defocus_offocus.py`` 里的已验证件。

    直接 import 而不是复制: 传播约定 (居中/未居中)、光学常量、GS 自检都在那边,
    复制一份就是本仓库反复被咬的「重复实现漂移」。
    """
    path = ROOT / "scripts" / "slm" / "experiment_gs_defocus_offocus.py"
    spec = importlib.util.spec_from_file_location("_exp_gs_defocus_offocus", path)
    if spec is None or spec.loader is None:  # pragma: no cover - 打包异常
        raise RuntimeError(f"cannot load sibling module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


defo = _load_sibling()
OffFocusBench = defo.OffFocusBench

# 光学常量全部来自兄弟实验 (追到 slm_gs_refine / matrix_utils 的源头)
WAVELENGTH_M = float(defo.WAVELENGTH_M)
FOCAL_LENGTH_M = float(defo.FOCAL_LENGTH_M)
APERTURE_M = float(defo.APERTURE_M)
PANEL_PIXEL_UM = float(defo.PANEL_PIXEL_UM)
BEAM_RADIUS_PX = float(defo.BEAM_RADIUS_PX)

#: SLM 位深: 10 bit, 2π 对应 1023 级灰度 (与实测 ``2π=993`` 同量级, 取满量程)。
SLM_PHASE_LEVELS = 1023

#: 加权 GS 参数默认值 —— 取自 SlmGsRefineConfig, 不在这里另立一份。
GS_GAMMA = float(gs_refine.SlmGsRefineConfig().gs_gamma)
GS_WEIGHT_CLIP = float(gs_refine.SlmGsRefineConfig().gs_weight_clip)
GS_BETA = float(gs_refine.SlmGsRefineConfig().gs_beta)
GS_HIO_FRACTION = float(gs_refine.SlmGsRefineConfig().gs_hio_fraction)

#: data.json 的 gs 行 (H0 必须复现它)
DATA_JSON_GS = {
    "score": 0.8118054772479197,
    "PIB": 0.9153051811202841,
    "CV": 0.4118196372088354,
}

#: 对照组
GROUPS: tuple[str, ...] = ("H0", "H1", "H2", "H3", "H4")
GROUP_LABELS: dict[str, str] = {
    "H0": "H0 plain GS, η=0 (基线)",
    "H1": "H1 plain GS, η>0, 不扣除",
    "H2": "H2 plain GS, η>0, 相干扣除",
    "H3": "H3 加权 GS, η>0, 相干扣除",
    "H4": "H4 H3 + 小倾斜移开晕斑",
}

#: 参考材料里「选 d」的判据: 晕斑峰值须低于目标区平均强度的 GAMMA_REF 倍。
GAMMA_REF = 1.0
GAMMA_LADDER: tuple[float, ...] = (1.0, 0.3, 0.1)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """一次运行的网格与预算。

    ``n_grid`` / ``far_field_padding`` / ``target_side_px`` / ``seed`` 必须与
    ``report/iterative_zernike_shaping/data.json`` 同口径 —— H0 的分数完全由这
    四项决定, 换了就对不上基线。
    """

    n_grid: int
    far_field_padding: int
    target_side_px: int
    seed: int
    gs_iters: int
    #: data.json 基线核对专用的迭代数 (gs_shape 用的就是 200)
    baseline_iters: int = 200
    dz_mm: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
    eta_values: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    #: 晕斑诊断用的更宽网格 (n_grid 越大窗口越宽, 见模块 docstring 的窗口公式)
    halo_n_grid: int = 256
    #: 权威传播器的 Voelz & Roggemann 带限。**默认关** —— 带限有损且不可逆,
    #: 正/反向不构成伴随对, GS 内环解的不是自洽问题 (见 TwoComponentBench
    #: 类 docstring 的实测表)。Δz=0 时开与关逐位相同, 所以基线不受影响。
    band_limit: bool = False
    tag: str = ""

    @property
    def m(self) -> int:
        return self.n_grid * self.far_field_padding

    def bench_cfg(self, n_grid: int | None = None) -> SquareShapingConfig:
        """整形用的配置 (光学常量与 data.json 同口径)。"""
        return SquareShapingConfig(
            n_grid=self.n_grid if n_grid is None else n_grid,
            target_side_px=self.target_side_px,
            far_field_padding=self.far_field_padding,
            seed=self.seed,
            wavelength=WAVELENGTH_M,
            focal_length=FOCAL_LENGTH_M,
            aperture_size=APERTURE_M,
        )


QUICK = RunConfig(
    # n_grid 决定**窗口宽** W=n_grid·λf/D (与 padding 无关)。n_grid=32 ⇒ 半窗只有
    # 289 µm, 零级晕斑在 Δz>0 全部超窗回绕 → 指标退化成数值伪影 (实测 PIB≈0)。
    # 所以 quick 也必须保住 n_grid=64 的窗口, 靠**减小 padding** 省算力 (512²→256²),
    # target_side 随像元变粗同步缩放以保持物理靶区不变。
    n_grid=64,
    far_field_padding=4,
    target_side_px=21,
    seed=0,
    gs_iters=60,
    dz_mm=(0.0, 5.0, 10.0, 20.0),
    eta_values=(0.0, 0.02, 0.05, 0.10),
    halo_n_grid=192,
    tag="_improved_quick",
)
FULL = RunConfig(
    n_grid=64,
    far_field_padding=8,
    target_side_px=43,
    seed=0,
    gs_iters=200,
    tag="_improved",
)


@dataclass(frozen=True)
class GsSetup:
    """一次 GS 运行的口径 (组 → 具体参数)。"""

    group: str
    eta: float
    weighted: bool
    subtract: bool
    #: 目标框 (与整形光斑一起) 相对画框中心的偏移 (px, (dy, dx))
    offset_px: tuple[int, int] = (0, 0)
    warm_start: bool = False


def setup_for(group: str, eta: float, offset_px: tuple[int, int]) -> GsSetup:
    """组名 → 具体 GS 口径。H0 只在 η=0 上有意义 (它是基线)。"""
    if group == "H0":
        return GsSetup("H0", 0.0, weighted=False, subtract=False)
    if group == "H1":
        return GsSetup(group, eta, weighted=False, subtract=False)
    if group == "H2":
        return GsSetup(group, eta, weighted=False, subtract=True)
    if group == "H3":
        return GsSetup(group, eta, weighted=True, subtract=True)
    if group == "H4":
        return GsSetup(group, eta, weighted=True, subtract=True, offset_px=offset_px)
    raise ValueError(f"unknown group {group!r}")


# ---------------------------------------------------------------------------
# 相位量化 (SLM 10 bit)
# ---------------------------------------------------------------------------
def quantize_phase(
    phase: np.ndarray, mask: np.ndarray, levels: int = SLM_PHASE_LEVELS
) -> np.ndarray:
    """把瞳面相位包裹到 [0, 2π) 并量化到位深 (2π = ``levels`` 级灰度)。

    只有 ``mask`` (瞳面口径内) 被量化; 口径外恒为 0 —— 那里没有像素, 量化一个
    不存在的相位没有意义, 而且会把 NaN 传染进来。
    """
    out = np.zeros_like(phase)
    wrapped = np.mod(phase[mask], 2.0 * math.pi)
    gray = np.mod(np.round(wrapped / (2.0 * math.pi) * levels), levels)
    out[mask] = gray / levels * 2.0 * math.pi
    return out


# ---------------------------------------------------------------------------
# 双分量瞳面模型
# ---------------------------------------------------------------------------
class TwoComponentBench(OffFocusBench):
    """在 :class:`OffFocusBench` 上加零级分量、SLM 流形约束与可配的带限。

    度量 (:meth:`evaluate` / :meth:`box_stats`)、光学常量与 GS 的投影骨架一律
    继承; 本类只回答两件事: 「瞳面里多一个 η 分量之后 GS 该怎么投影」, 以及
    「权威传播器的带限要不要开」。

    ⚠️ **为什么把 ``band_limit`` 提成参数、且默认关**
    --------------------------------------------------
    :func:`~ao_shaping.algorithm.signal_processing.angular_spectrum.angular_spectrum_propagate`
    默认 ``band_limit=True`` (Voelz & Roggemann cutoff)。那个 cut 在焦平面上
    **有损且不可逆**: 高频 bin 被置零, 能量在**正向一步**就漏掉 (实测 Δz=50 mm
    漏 0.7%~22%, 随入射场谱宽而变), 反向一步再也补不回来 —— 于是
    ``fwd(z)`` 与 ``bwd(−z)`` **不构成伴随对**, GS 内环是在解一个不自洽的问题。
    往返误差实测 (64² 瞳面 + 二次相位, dx=8 µm):

    ==============  ======  =======  =======  =======
    ``band_limit``  0 mm   10 mm    30 mm    50 mm
    ==============  ======  =======  =======  =======
    ``True``        8e-16  2.4e-01  8.6e-01  9.4e-01
    ``False``       8e-16  6.5e-16  6.8e-16  6.8e-16
    ==============  ======  =======  =======  =======

    关掉带限后传递函数模长恒为 1, 传播可逆, 内环自洽。本实验的远场网格已经零填充
    (padding=8 ⇒ 512²/2048²), 残余混叠可控, 故 **Δz>0 一律默认 ``band_limit=False``**。
    Δz=0 时两者逐位相同 (``exp(i·kz·0)=1`` 且 cut 全通), 所以基线与 ``data.json``
    的核对不受影响 —— 由 :func:`self_check_eta0` 与 ``round_trip_relerr`` 兜底。

    **不改 ``src/``**: 权威件本身的默认值是有物理理由的, 只在**本实验**里把它
    变成可配项。``--band-limit`` 可以随时复现带限那一版。
    """

    def __init__(
        self,
        cfg: SquareShapingConfig,
        *,
        band_limit: bool = False,
        n_zernike: int = defo.N_ZERNIKE,
    ) -> None:
        super().__init__(cfg, n_zernike=n_zernike)
        self.band_limit = bool(band_limit)

    # -- 正/反向传播算子 (只把带限做成可配, 公式一行不改) --------------------
    def _asm_raw(
        self, field: np.ndarray, dz: float, band_limit: bool | None = None
    ) -> np.ndarray:
        """权威传播器走一步。输入**居中**排布, 输出**未 fftshift** (DC 在 index 0)。"""
        return angular_spectrum_propagate(
            np.fft.ifftshift(field),
            self.dx,
            dz,
            self.lam,
            band_limit=self.band_limit if band_limit is None else bool(band_limit),
        )

    def _asm(
        self, field: np.ndarray, dz: float, band_limit: bool | None = None
    ) -> np.ndarray:
        """焦平面 → CCD 复场 (输入输出都是**居中**排布)。"""
        return np.fft.fftshift(self._asm_raw(field, dz, band_limit))

    def _asm_inverse(
        self, field: np.ndarray, dz: float, band_limit: bool | None = None
    ) -> np.ndarray:
        """``_asm`` 的伴随逆 (CCD → 瞳面, 居中排布)。

        用 ``H(−dz) = conj(H(dz))`` 代替真正的伴随。**只有传播器无歧时这才是
        伴随** —— 带限一开, 高频 bin 在正向被永久置零, 往返闭不上 (见类 docstring
        的实测表)。:func:`measure_round_trip` 逐 Δz 量这个闭合误差。
        """
        return np.fft.fftshift(
            np.fft.ifft2(self._asm_raw(field, -dz, band_limit))
        )

    def pupil_field(self, phase: np.ndarray, eta: float) -> np.ndarray:
        """``U_inc·[η + (1−η)·exp(iφ)]``, 已补零到远场网格 (居中排布)。

        η=0 时逐位等价于父类的 ``amp·exp(iφ)`` —— ``0.0 + 1.0·z`` 与 ``1.0·z``
        在 IEEE754 下都是精确运算, 所以基线不受影响 (由 :func:`self_check_eta0`
        逐位验证)。
        """
        n = self.n
        inner = self.amp * (eta + (1.0 - eta) * np.exp(1j * np.asarray(phase, np.float64)))
        padded = np.zeros((self.m, self.m), dtype=np.complex128)
        padded[self.pad : self.pad + n, self.pad : self.pad + n] = inner
        return padded

    def zero_order_focal(self, eta: float) -> np.ndarray:
        """零级分量 ``η·U_inc`` 在**焦平面**的复场 (居中排布)。"""
        n = self.n
        padded = np.zeros((self.m, self.m), dtype=np.complex128)
        padded[self.pad : self.pad + n, self.pad : self.pad + n] = (eta * self.amp).astype(
            np.complex128
        )
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))

    def ccd_field_eta(
        self, phase: np.ndarray, eta: float, dz: float, band_limit: bool | None = None
    ) -> np.ndarray:
        """瞳面相位 + η → CCD 复场 (居中排布)。"""
        return self._asm(
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.pupil_field(phase, eta)))),
            dz,
            band_limit,
        )

    def intensity_eta(
        self, phase: np.ndarray, eta: float, dz: float, band_limit: bool | None = None
    ) -> np.ndarray:
        """CCD 强度, 归一化到 sum = 1。"""
        inten = np.abs(self.ccd_field_eta(phase, eta, dz, band_limit)) ** 2
        return inten / (inten.sum() + 1e-12)

    def zero_order_ccd(
        self, eta: float, dz: float, band_limit: bool | None = None
    ) -> np.ndarray:
        """零级分量在 CCD 的复场 ``b0`` —— 相干扣除要减掉的那个量。

        FT 与 ASM 都是线性的, 所以「两分量一起走同一条链」和「零级单独走一遍
        同一条链」给出同一个 ``b0``; 零级只是不承载 ``exp(iφ)``。
        """
        return self._asm(self.zero_order_focal(eta), dz, band_limit)

    # -- 目标 ---------------------------------------------------------------
    @staticmethod
    def shifted_target(cfg: SquareShapingConfig, offset_px: tuple[int, int]) -> np.ndarray:
        """把方形目标整体平移 ``offset_px`` (dy, dx)。

        与 :func:`make_square_target` 的构造逐字相同, 只是方框中心从 ``m//2``
        挪到 ``m//2 + offset``。GS 的解会自带一个线性相位倾斜, 于是整形光斑连同
        目标框一起移开轴上的零级晕斑 (零级不承载 SLM 相位, 不跟着走)。
        """
        base = make_square_target(cfg)
        if offset_px == (0, 0):
            return base
        out = np.zeros_like(base)
        dy, dx = offset_px
        cy = base.shape[0] // 2 + dy
        cx = base.shape[1] // 2 + dx
        side = int(cfg.target_side_px)
        out[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2] = 1.0
        return out / out.sum()

    # -- GS ----------------------------------------------------------------
    def gs_phase_eta(
        self,
        dz: float,
        *,
        eta: float,
        n_iters: int,
        seed: int,
        weighted: bool = False,
        subtract: bool = False,
        offset_px: tuple[int, int] = (0, 0),
        warm_phase: np.ndarray | None = None,
        quantize: bool = True,
        gamma: float = GS_GAMMA,
        weight_clip: float = GS_WEIGHT_CLIP,
        record_history: bool = False,
        history_every: int = 10,
    ) -> dict[str, Any]:
        """双分量瞳面模型下的 GS, 返回相位 (量化版 + 连续版) 与逐轮分数。

        与 :meth:`OffFocusBench.gs_phase` 的循环骨架逐行同构 (Δz=0、η=0、
        ``weighted=False``、``subtract=False``、不量化时逐位相同), 三处改动:

        1. **目标面约束前的相干扣除** (``subtract=True``): ``U -= b0``。实测里
           ``b0`` 只能从模型取, 硬件上要单独标定 (关掉 SLM 调制, 拍一张)。
        2. **加权反馈** (``weighted=True``): 照抄
           :meth:`UniformSquareShaper._weighted_step` 的权重更新式与裁剪/归一化,
           参数与之逐位一致 (``gamma`` 用的是**振幅**比, 等价于强度比的 γ/2 次幂)。
        3. **瞳面约束写成 SLM 流形**而不是 ``|field| = amp``: 从回代场里解出信号
           相量再重建 ``amp·(η + (1−η)e^{iφ})``。写成 ``|field| = amp`` 等于把
           零级也一起优化掉了 —— 那正是本模型要排除的。

        ``record_history`` 会在循环里**每 ``history_every`` 轮**额外算一次正演来
        记分数 (每次正演 ≈ 1/3 个迭代的代价), 所以只用来做收敛速度对照, 主扫描
        不开。
        """
        target = self.shifted_target(self.cfg, offset_px)
        rng = np.random.default_rng(seed)

        amp_target = np.sqrt(target)
        amp_slm = np.zeros((self.m, self.m), dtype=np.float64)
        amp_slm[self.pad : self.pad + self.n, self.pad : self.pad + self.n] = self.amp
        support = amp_target > 0
        # 能量守恒: 目标幅度重标到与**瞳面**总能量一致 (UniformSquareShaper 的做法)。
        # plain GS 里这个常数因子被 exp(i∠) 抵消, 加权分支里它只影响权重更新的
        # 一个全局 γ 次幂 (随后被 W 的均值归一化吸收), 所以各组口径一致。
        rescale = math.sqrt(
            self.m * self.m * float(np.sum(amp_slm**2)) / max(float(np.sum(amp_target**2)), 1e-30)
        )
        amp_target = amp_target * rescale

        if warm_phase is not None:
            field = self.pupil_field(np.asarray(warm_phase, np.float64), eta)
        else:
            field = amp_slm * np.exp(1j * rng.normal(0, 0.1, size=amp_slm.shape))

        f_zero = self.zero_order_focal(eta) if (eta > 0.0 or subtract) else None
        u_zero = self._asm(f_zero, dz) if f_zero is not None else None

        weights = np.ones_like(amp_target)
        aperture = self.amp > 0
        phi = np.zeros((self.n, self.n), dtype=np.float64)
        phi_cont = np.zeros_like(phi)
        history: list[float] = []
        hist_iter: list[int] = []

        s = self.pad
        for it in range(n_iters):
            ccd = self._asm(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field))), dz)
            if subtract and u_zero is not None:
                ccd = ccd - u_zero
            if weighted:
                ratio = np.ones_like(amp_target)
                ratio[support] = (
                    amp_target[support] / (np.abs(ccd)[support] + 1e-12)
                ) ** gamma
                weights = np.clip(weights * ratio, 1.0 / weight_clip, weight_clip)
                weights /= max(float(weights[support].mean()), 1e-12)
                amp_c = weights * amp_target
            else:
                amp_c = amp_target
            ccd = amp_c * np.exp(1j * np.angle(ccd))
            if subtract and u_zero is not None:
                # 回代时把零级**加回 CCD 场**: 扣除只发生在「目标面约束之前」。
                # 回到瞳面的场必须仍是含零级的完整场, 下面的流形投影才拿得到 φ。
                # (零级与信号走同一条 fft→asm 链, 所以 `_asm_inverse` 天然把它带回来。)
                ccd = ccd + u_zero
            field = self._asm_inverse(ccd, dz)

            # 瞳面流形投影: 从回代场里解出信号相量 e^{iφ} = (field/amp − η)/(1 − η)。
            # amp 在口径**外**恒为 0 (Gaussian 被口径截断), 所以必须用 where= 只在
            # 口径内除 —— 直接整块相除会得到 inf/NaN, 角度一旦是 NaN 就再也回不来。
            inner = np.zeros((self.n, self.n), dtype=np.complex128)
            np.divide(
                field[s : s + self.n, s : s + self.n],
                self.amp,
                out=inner,
                where=aperture,
            )
            phi_cont = np.angle((inner - eta) / (1.0 - eta))
            phi_cont[~aperture] = 0.0
            phi = quantize_phase(phi_cont, aperture) if quantize else phi_cont
            field = self.pupil_field(phi, eta)

            if record_history and it % max(history_every, 1) == 0:
                inten = self.intensity_eta(phi, eta, dz)
                history.append(self._score_in_box(inten, support))
                hist_iter.append(it)

        return {
            "phase": phi,
            "phase_cont": phi_cont,
            "history": history,
            "history_iters": hist_iter,
        }

    def _score_in_box(self, inten: np.ndarray, support: np.ndarray) -> float:
        """当前帧在**定尺**支撑域上的 composite score (与 data.json 同口径)。"""
        vals = inten[support]
        if vals.size == 0 or vals.mean() <= 0:
            return 0.0
        return composite_from_pib_cv(
            float(vals.sum()), float(vals.std() / vals.mean())
        )


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def self_check_eta0(bench: TwoComponentBench, n_iters: int, seed: int) -> float:
    """η=0 的双分量 GS 与仓库 :func:`gs_shape` 逐位相同 (口径内)。返回 ``max|Δphase|``。

    **只在瞳面口径内比较**: 口径外的 Gaussian 幅度被截断成**精确 0**, 那里的相位
    是 ``angle(0)`` 的 ±0/±π 垃圾 (``gs_phase`` 与本文件的取法不同), 但它既不会被
    显示 (``pupil_field`` 在口径外恒为 0) 也不参与任何计算。口径外不同 = 正常。
    """
    ref = gs_shape(bench.cfg, n_iters=n_iters, seed=seed).phase
    mine = bench.gs_phase_eta(
        0.0, eta=0.0, n_iters=n_iters, seed=seed, quantize=False
    )["phase_cont"]
    mask = bench.amp > 0
    delta = float(np.abs(ref[mask] - mine[mask]).max())
    if delta > 1e-9:
        raise AssertionError(
            f"η=0 的双分量 GS 与 gs_shape 在口径内不一致 (max|Δphase|={delta:.3e}); "
            "流形约束写错了, H0 对不上基线"
        )
    return delta


def self_check_scan_baseline(tolerance: float = 1e-6) -> dict[str, Any]:
    """**扫描路径**在 Δz=0、η=0 上必须逐位复现 data.json 的 gs 行 (永久断言)。

    这条断言挡的是上一版那个「度量框放到 H4 倾斜偏移上」的 bug: 框一旦离开信号,
    框内 PIB 从 **0.9153** 塌到 **0.0034** (270×), 而 ``pib_box`` 在非零偏移分支里
    直接写 ``NaN``。这里故意**走扫描自己的代码路径** (``TwoComponentBench.gs_phase_eta``
    → :func:`anchor_box_center` → ``bench.box_stats``), 而不是另调一次 ``gs_shape``:
    只有这样才能挡住「框放歪 / 度量口径被改坏」这类改动 —— 另算一遍 ``gs_shape``
    永远是对的, 挡不住任何东西。

    口径固定为 data.json 的 (n_grid=64/padding=8/target_side=43/seed=0, 200 轮,
    **连续相位**), 与 ``--quick`` 无关 —— 基线核对不随冒烟跑缩水。
    """
    cfg = SquareShapingConfig(
        n_grid=64,
        target_side_px=43,
        far_field_padding=8,
        seed=0,
        wavelength=WAVELENGTH_M,
        focal_length=FOCAL_LENGTH_M,
        aperture_size=APERTURE_M,
    )
    bench = TwoComponentBench(cfg, band_limit=False)
    out = bench.gs_phase_eta(0.0, eta=0.0, n_iters=200, seed=0, quantize=False)
    inten = bench.intensity_eta(out["phase_cont"], 0.0, 0.0)
    center, centroid = anchor_box_center(
        inten, (cfg.far_field_size // 2, cfg.far_field_size // 2), radius=anchor_radius(43)
    )
    pib, cv = bench.box_stats(inten, center[1], center[0], 43)
    score = composite_from_pib_cv(pib, cv)
    measured = {"score": float(score), "PIB": float(pib), "CV": float(cv)}
    delta = {k: measured[k] - DATA_JSON_GS[k] for k in measured}
    worst = max(abs(v) for v in delta.values())
    if worst > tolerance:
        raise AssertionError(
            "扫描路径在 Δz=0、η=0 上没有复现 data.json 基线: "
            f"max|Δ|={worst:.3e} > {tolerance:.0e} "
            f"(measured {measured}, reference {DATA_JSON_GS})。"
            "最可能的原因: 度量框没有锚在信号上 (框放歪了) —— "
            "检查 anchor_box_center / GsSetup.offset_px 是否被按组正确传入"
        )
    return {
        "measured": measured,
        "reference_datajson_gs": DATA_JSON_GS,
        "delta": delta,
        "max_abs_delta": float(worst),
        "tolerance": tolerance,
        "box_center_px": [int(center[0]), int(center[1])],
        "box_side_px": 43,
        "signal_centroid_px": [round(centroid[0], 4), round(centroid[1], 4)],
        "phase": "continuous (quantize=False), i.e. the data.json baseline caliber",
        "note": (
            "走扫描自己的代码路径 (gs_phase_eta + anchor_box_center + box_stats); "
            "框放歪 / 度量口径被改坏都会在这里 raise"
        ),
    }


def self_check_halo_monotonic(
    rows: list[dict], spread_tol: float = 1.15
) -> dict[str, Any]:
    """晕斑峰值必须随 Δz 单调下降, 且在渐近区满足 ``I_halo·Δz² = const``。

    两条量分别回答两个问题:

    * ``worst_adjacent_ratio`` —— **最小**相邻比 ``I(dz_{k+1})/I(dz_k)``。它就是
      「最陡的那一步」, 取值落在 0~1 之间。它**不是**「1/d² 的系数」: 0→10 mm
      这一步落在菲涅耳高斯的过渡区 ``1/(1+(Δz/z_R)²)`` 内 (实测 z_R ≈ 1.5 mm),
      天然比渐近的 1/Δz² 陡得多; 之后相邻比就回到 ``(dz_k/dz_{k+1})²``。
    * ``max_I_dz2_spread`` —— 渐近区里 ``I·Δz²`` 的 max/min。理论值 1; 实测 ~1.01
      (带限关掉)。>1.15 说明有别的机制在压平/抬高峰值 (带限漏能、晕斑大过场
      窗口被卷回), 那就不是「两分量模型错了」, 是数值口径坏了 —— 照样 raise。
    """
    by_eta: dict[float, list[dict]] = {}
    for r in rows:
        by_eta.setdefault(float(r["eta"]), []).append(r)
    worst = 1.0
    spreads: list[float] = []
    for _eta, group in by_eta.items():
        group.sort(key=lambda r: r["dz_mm"])
        peaks = [float(r["i_halo_peak"]) for r in group]
        for a, b in zip(peaks, peaks[1:]):
            worst = min(worst, b / a if a > 0 else 0.0)
        scaled = [
            p * (max(float(r["dz_mm"]), 1e-9) * 1e-3) ** 2 for p, r in zip(peaks, group)
        ]
        tail = scaled[2:]
        if tail and max(tail) > 0:
            spreads.append(float(max(tail) / min(tail)))
    spread = max(spreads or [1.0])
    if worst >= 1.0:
        raise AssertionError(
            f"零级晕斑峰值没有随 Δz 单调下降 (最小相邻比 = {worst:.4f} ≥ 1); "
            "双分量模型写错了 (I_halo 应 ∝ f²/d²), 不继续跑"
        )
    if spread > spread_tol:
        raise AssertionError(
            f"渐近区 I_halo·Δz² 的展布 = {spread:.3f} > {spread_tol}; "
            "1/d² 律不成立。通常是带限漏能或晕斑大过场窗口被卷回 —— "
            "先查 band_limit 与 halo_w_over_halfwindow, 不要继续跑"
        )
    return {
        "worst_adjacent_ratio": worst,
        "worst_adjacent_ratio_means": "min_k I(dz_{k+1})/I(dz_k), 越接近 1 越平",
        "max_I_dz2_spread": spread,
        "max_I_dz2_spread_tol": spread_tol,
        "max_I_dz2_spread_means": "max/min of I_halo·Δz² over the asymptotic tail (Δz ≥ 3rd point)",
    }


# ---------------------------------------------------------------------------
# 晕斑诊断 (宽网格)
# ---------------------------------------------------------------------------
def measure_halo(rc: RunConfig) -> tuple[list[dict], TwoComponentBench]:
    """在**宽网格**上量零级晕斑: 峰值、1/e² 半径、与目标区平均强度之比。

    为什么要宽网格: 见模块 docstring —— ``W = n_grid·λf/D``。整形网格 (n_grid=64,
    padding=8) 的半窗只有 579 µm, 而 Δz≥40 mm 的晕斑 1/e² 半径已超过它, 会被
    FFT 的周期性**卷回**网格中央、把峰值趋势压平。halo 网格只加**瞳面过采样**,
    **远场像元 pitch 与光学常量完全不变** (``far_field_pixel = λf/(D·padding)``,
    与 n_grid 无关), 所以量到的仍是同一套物理、同一根标尺, 只是画框更宽。

    峰值同时报带限开/关两版 (``:eta0`` 之外的后缀 ``_bl``), 差值就是这条曲线上的
    数值误差带。
    """
    bench = TwoComponentBench(rc.bench_cfg(n_grid=rc.halo_n_grid), band_limit=rc.band_limit)
    rows: list[dict] = []
    half_um = bench.m * bench.dx * 1e6 / 2.0
    for eta in rc.eta_values:
        if eta <= 0.0:
            continue
        for dz_mm in rc.dz_mm:
            entry: dict[str, Any] = {"eta": eta, "dz_mm": dz_mm}
            w0_um = None
            for bl in (rc.band_limit, True):
                suffix = "" if bl == rc.band_limit else "_bl"
                u = bench.zero_order_ccd(eta, dz_mm * 1e-3, band_limit=bl)
                inten = np.abs(u) ** 2
                inten = inten / inten.sum()
                _, _, rms = bench.centroid_sigma(inten)
                # centroid_sigma 给的是 **2D RMS 半径** R = √2·σ_axis; 高斯强度
                # I = I0·exp(−r²/(2σ_axis²)) 的 1/e² 半径是 2σ_axis = √2·R。
                w_1e2_um = math.sqrt(2.0) * rms * bench.dx * 1e6
                entry[f"i_halo_peak{suffix}"] = float(inten.max()) * eta**2
                entry[f"halo_w_1e2_px{suffix}"] = float(math.sqrt(2.0) * rms)
                entry[f"halo_w_1e2_um{suffix}"] = float(w_1e2_um)
                entry[f"halo_w_over_halfwindow{suffix}"] = float(w_1e2_um / half_um)
                if not bl and dz_mm == 0.0:
                    w0_um = w_1e2_um
            entry["half_window_um"] = float(half_um)
            rows.append(entry)
    # 目标区平均强度 (H0 基线归一化强度在目标框内的均值) 作为比较基准
    ref = measure_reference_box(rc)
    for r in rows:
        r["i_box_mean"] = ref["i_box_mean"]
        r["halo_peak_over_box_mean"] = r["i_halo_peak"] / max(ref["i_box_mean"], 1e-30)
        if "_bl" in r:
            r["halo_peak_over_box_mean_bl"] = r["i_halo_peak_bl"] / max(
                ref["i_box_mean"], 1e-30
            )
    z_rayleigh_mm = measure_z_rayleigh_mm(rows)
    ref_peak: dict[float, float] = {}
    for r in rows:
        if r["dz_mm"] == 0.0:
            ref_peak[float(r["eta"])] = float(r["i_halo_peak"])
    shape_half_um = (
        rc.m * rc.bench_cfg().far_field_pixel_size * 1e6 / 2.0
    )  # 整形网格的半窗 —— 晕斑最终要落在**这张**网格里
    for r in rows:
        r["z_rayleigh_mm"] = z_rayleigh_mm
        base = max(ref_peak.get(float(r["eta"]), 0.0), 1e-300)
        r["i_halo_ratio_measured"] = float(r["i_halo_peak"]) / base
        pred = (
            _fresnel_gaussian_ratio(r["dz_mm"] / z_rayleigh_mm)
            if z_rayleigh_mm
            else float("nan")
        )
        r["i_halo_ratio_predicted"] = pred
        r["i_halo_rel_dev"] = (
            float(r["i_halo_ratio_measured"] / pred) if pred > 0 else float("nan")
        )
        # 晕斑跑出**整形网格**半窗 ⇒ FFT 周期性把它从对面卷回靶区, 该点不可信
        w_shape = r["halo_w_1e2_um"]
        ratio_shape = w_shape / shape_half_um
        r["halo_w_over_shape_halfwindow"] = float(ratio_shape)
        r["trusted"] = bool(ratio_shape <= HALO_WRAP_RATIO_LIMIT)
    return rows, bench


def measure_z_rayleigh_mm(rows: list[dict]) -> float | None:
    """从晕斑 1/e² 半径随 Δz 的增长反推瑞利范围 ``z_R`` (mm)。

    高斯束 ``w(Δz) = w0·sqrt(1+(Δz/z_R)²)`` ⇒ ``Δz = z_R·sqrt((w/w0)²−1)``。
    只用**最小 η** 那一组 (晕斑形状与 η 无关, 只有幅值差一个 η²), 取各点中位数,
    单点损坏 (晕斑大过场窗口被卷回) 不会带偏它。
    """
    eta0 = min(float(r["eta"]) for r in rows)
    sel = sorted(
        (r for r in rows if float(r["eta"]) == eta0), key=lambda r: r["dz_mm"]
    )
    w0 = next((r["halo_w_1e2_um"] for r in sel if r["dz_mm"] == 0.0), None)
    if not w0:
        return None
    est = [
        r["dz_mm"] / math.sqrt(r["halo_w_1e2_um"] ** 2 / w0**2 - 1.0)
        for r in sel
        if r["dz_mm"] > 0.0 and r["halo_w_1e2_um"] > w0
    ]
    return float(np.median(est)) if est else None


def _fresnel_gaussian_ratio(dz_over_zr: float) -> float:
    """相对 Δz=0 的峰值比 ``1/(1+(Δz/z_R)²)`` —— 晕斑衰减的解析预言。

    这条曲线在 ``Δz ≪ z_R`` 近乎平坦, 在 ``Δz ≫ z_R`` 渐近到 ``(z_R/Δz)²``
    (即任务里的 ``I_halo ∝ f²/d²``)。**它就是「0→10 mm 那一步为什么掉到
    1.6%」的答案** —— 那一段还没进渐近区。
    """
    return 1.0 / (1.0 + float(dz_over_zr) ** 2)


def measure_reference_box(rc: RunConfig) -> dict[str, float]:
    """η=0、Δz=0 的 plain GS 基线上, 目标框内的平均强度 (sum=1 归一)。"""
    cfg = rc.bench_cfg()
    bench = TwoComponentBench(cfg, band_limit=rc.band_limit)
    phase = bench.gs_phase_eta(
        0.0, eta=0.0, n_iters=rc.baseline_iters, seed=rc.seed, quantize=False
    )["phase_cont"]
    inten = bench.intensity_eta(phase, 0.0, 0.0)
    vals = inten[make_square_target(cfg) > 0]
    return {
        "i_box_mean": float(vals.mean()),
        "i_box_sum": float(vals.sum()),
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------
def measure_round_trip(bench: TwoComponentBench, rc: RunConfig) -> list[dict]:
    """往返闭合误差: 瞳面场走 ``+d`` 再走 ``−d``, 逐 Δz 与逐带限开关。

    入射场取**带二次相位的瞳面场** ``exp(−iπ·d·r²/(λf²))`` —— 任务给的傍轴
    等价式, 谱宽足够铺到 Nyquist, 所以带限一开就会立刻露出损失。用一个常数场或
    一个纯 DC 场测是测不出来的: 它的谱全在零频, 带限根本碰不到。

    带限打开时这个数会在 |Δz|≳10 mm 爆到 1e-1 量级 (能量在**正向**就漏, 见
    :class:`TwoComponentBench` 类 docstring), 关掉则全程 ~1e-15。
    """
    n, s = bench.n, bench.pad
    dx_pupil = bench.cfg.aperture_size / n
    yy, xx = np.mgrid[-n / 2 : n / 2, -n / 2 : n / 2].astype(np.float64)
    r2 = (xx * dx_pupil) ** 2 + (yy * dx_pupil) ** 2
    rows: list[dict] = []
    for dz_mm in rc.dz_mm:
        dz = dz_mm * 1e-3
        pupil = np.zeros((bench.m, bench.m), dtype=np.complex128)
        pupil[s : s + n, s : s + n] = bench.amp * np.exp(
            -1j * math.pi * max(dz, 1e-12) * r2 / (bench.lam * bench.cfg.focal_length**2)
        )
        scale = float(np.abs(pupil).max())
        focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        entry: dict[str, Any] = {"dz_mm": dz_mm}
        for bl in (False, True):
            ccd = bench._asm(focal, dz, bl)
            back = bench._asm_inverse(ccd, dz, bl)
            entry[f"round_trip_relerr_bl{int(bl)}"] = float(
                np.abs(back - pupil).max() / scale
            )
            entry[f"energy_ratio_bl{int(bl)}"] = float(
                (np.abs(ccd) ** 2).sum() / max((np.abs(pupil) ** 2).sum(), 1e-30)
            )
        entry["round_trip_relerr"] = entry[f"round_trip_relerr_bl{int(bench.band_limit)}"]
        rows.append(entry)
    return rows


def self_check_round_trip(
    rows: list[dict], band_limit: bool, tolerance: float = 1e-12
) -> dict[str, Any]:
    """配置的传播器必须在全程可逆, 否则 GS 内环无意义。"""
    key = f"round_trip_relerr_bl{int(band_limit)}"
    worst = max(r[key] for r in rows)
    worst_bl = max(r["round_trip_relerr_bl1"] for r in rows)
    if worst > tolerance:
        raise AssertionError(
            f"传播器往返不闭合: max relerr = {worst:.3e} > {tolerance:.0e} "
            f"(Δz=0..50 mm)。带限 ASM 有损且不可逆, 正/反向不构成伴随对, "
            "GS 内环解的不是自洽问题 —— 请 band_limit=False, 不要继续跑"
        )
    return {
        "configured_band_limit": bool(band_limit),
        "round_trip_relerr_max": worst,
        "round_trip_relerr_max_band_limited": worst_bl,
        "tolerance": tolerance,
        "note": (
            "带限一开, 往返误差随 Δz 爆到 1e-1 量级 (能量在正向一步就漏); "
            "关掉后全程 ~1e-15。|Δz|≳10 mm 时带限 ASM 的 GS 内环不可信"
        ),
        "by_dz": rows,
    }


#: 晕斑 1/e² 半径 / 场半窗 超过它就算被 FFT 周期性卷回污染, 该 Δz 的 CV/PIB 不可信。
HALO_WRAP_RATIO_LIMIT = 0.5


def measure_band_limit_damage(rc: RunConfig) -> dict[str, Any]:
    """带限在大 Δz 上的破坏程度 —— **强依赖场的谱内容与网格尺度**, 故扫多个探针取跨度。

    Voelz & Roggemann 的 ``f_max = 1/(λ√(1+(2z/B)²))`` 里 ``B`` 是**传播网格宽度**,
    所以损失由两件事决定: (a) 场的谱在 ``f_max`` 之上的功率有多少, (b) ``B`` 有多大
    (= 网格边长 × 像元)。宽带场 + 小窗 = 最惨。

    探针 (全部用权威传播器, 只开带限):

    * ``coarse_pupil_grid``: 64²、dx=8 µm 的瞳面网格 —— 窄谱 + 窗大的一端;
    * ``far_field_truncated_pupil``: 整形网格上 GS 内环真正在用的场 (有硬口径);
    * ``far_field_smooth_gaussian``: 同网格、无硬口径的窄谱场;
    * ``far_field_random_phase``: 同网格、随机相位 —— 最宽的一端。

    每行同时报 ``frac_power_above_cut``, 让人直接看到机制而不只是结果。
    **不要引用单一数字** —— 换实验系统请重测。
    """
    lam = WAVELENGTH_M
    cfg = rc.bench_cfg()
    n, m = cfg.n_grid, cfg.far_field_size
    dx_ff = cfg.far_field_pixel_size
    s = (m - n) // 2
    d_ap = cfg.aperture_size
    yy, xx = np.mgrid[-n / 2 : n / 2, -n / 2 : n / 2] * (d_ap / n)
    r = np.sqrt(xx**2 + yy**2)
    w0 = d_ap / 3.5
    smooth = np.exp(-(r**2) / w0**2)
    truncated = smooth * (r <= d_ap / 2.0)
    rng = np.random.default_rng(0)

    def to_focal(pupil_field: np.ndarray) -> np.ndarray:
        block = np.zeros((m, m), dtype=np.complex128)
        block[s : s + n, s : s + n] = pupil_field
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(block)))

    n_c, dx_c = 64, 8e-6
    yyc, xxc = np.mgrid[-n_c / 2 : n_c / 2, -n_c / 2 : n_c / 2] * (d_ap / n_c)
    coarse = np.exp(-(xxc**2 + yyc**2) / w0**2)

    probes = [
        ("coarse_pupil_grid", dx_c, coarse.astype(np.complex128)),
        ("far_field_truncated_pupil", dx_ff, to_focal(truncated)),
        ("far_field_smooth_gaussian", dx_ff, to_focal(smooth)),
        ("far_field_random_phase", dx_ff,
         to_focal(truncated * np.exp(1j * rng.normal(0, 0.5, (n, n))))),
    ]
    rows: list[dict] = []
    for name, dx, field in probes:
        shape = field.shape[0]
        freq = np.fft.fftfreq(shape, d=dx)
        fy, fx = np.meshgrid(freq, freq, indexing="ij")
        fr = np.hypot(fx, fy)
        spec = np.abs(np.fft.fft2(np.fft.ifftshift(field))) ** 2
        spec = spec / max(spec.sum(), 1e-300)
        e0 = float((np.abs(field) ** 2).sum())
        for dz_mm in rc.dz_mm:
            z = dz_mm * 1e-3
            f_max = 1.0 / (lam * math.sqrt(1.0 + (2.0 * z / (shape * dx)) ** 2))
            out = angular_spectrum_propagate(field, dx, z, lam, band_limit=True)
            e1 = float((np.abs(out) ** 2).sum())
            rows.append(
                {
                    "probe": name,
                    "dz_mm": dz_mm,
                    "grid": shape,
                    "window_um": shape * dx * 1e6,
                    "f_max_per_m": f_max,
                    "frac_power_above_cut": float(spec[fr > f_max].sum()),
                    "energy_loss_pct": 100.0 * (1.0 - e1 / max(e0, 1e-30)),
                }
            )
    worst = [r for r in rows if r["dz_mm"] == max(rc.dz_mm)]
    losses = [r["energy_loss_pct"] for r in worst]
    return {
        "note": (
            "破坏程度强依赖场的谱内容**与网格尺度** —— Δz={dz:g} mm 实测能量损失跨度 "
            "{lo:.2f}% ~ {hi:.2f}% (同一权威传播器, 只开带限, {n} 个探针)。"
            "不要引用单一数字, 先测你自己那个场的谱。"
        ).format(lo=min(losses), hi=max(losses), dz=max(rc.dz_mm), n=len(probes)),
        "loss_span_pct": [float(min(losses)), float(max(losses))],
        "probes": [p[0] for p in probes],
        "rows": rows,
    }


def measure_trustworthy_dz(
    halo: list[dict], half_window_um: float, n_grid_hint: int
) -> dict[str, Any]:
    """逐 Δz 判可信度, 并算出**当前网格能覆盖到哪个 Δz**。

    判据: 晕斑 1/e² 半径占场半窗的比例 ``w/half_window``。FFT 是周期的, 晕斑一旦
    超过半窗, 边缘的能量会从对面卷回靶区, CV/PIB 变成数值伪影 —— 那不是
    「能量掉出探测器」的物理, 是采样不够。超过 ``HALO_WRAP_RATIO_LIMIT`` 即标记。

    **能覆盖到哪只由 ``n_grid`` 决定**: 窗口宽 ``W = n_grid·λf/D`` 与 padding
    无关 —— 加大 padding 只把像元切细、把网格变大, 窗口一动不动 (实测
    ``n_grid=64`` 下 padding 8 与 padding 32 的半窗都是 578.8 µm)。要装下更远的
    晕斑必须**加瞳面过采样 n_grid**。
    """
    per_dz: list[dict] = []
    for dz in sorted({float(r["dz_mm"]) for r in halo}):
        sel = [r for r in halo if abs(float(r["dz_mm"]) - dz) < 1e-12]
        # 注意比的是**整形网格**的半窗 —— 晕斑最终要落进那张网格, 不是 halo 网格
        ratio = max(float(r["halo_w_over_shape_halfwindow"]) for r in sel)
        per_dz.append(
            {
                "dz_mm": dz,
                "halo_radius_over_halfwindow": ratio,
                "trusted": bool(ratio <= HALO_WRAP_RATIO_LIMIT),
            }
        )
    trusted = [r["dz_mm"] for r in per_dz if r["trusted"]]
    dz_max = max(rc_dz(per_dz))
    w_at_max = max(
        float(r["halo_w_1e2_um"]) for r in halo if abs(float(r["dz_mm"]) - dz_max) < 1e-12
    )
    lam, f, d = WAVELENGTH_M, FOCAL_LENGTH_M, APERTURE_M
    # half_window_m >= w/LIMIT, 而 half_window_m = n_grid·λf/(2D)
    need_half_m = (w_at_max / HALO_WRAP_RATIO_LIMIT) * 1e-6
    n_needed = int(math.ceil(need_half_m * 2.0 * d / (lam * f)))
    return {
        "half_window_um": half_window_um,
        "ratio_limit": HALO_WRAP_RATIO_LIMIT,
        "per_dz": per_dz,
        "trusted_dz_mm": trusted,
        "first_untrusted_dz_mm": (
            next((r["dz_mm"] for r in per_dz if not r["trusted"]), None)
        ),
        "window_formula": "W = n_grid * lambda * f / D   (padding 无关)",
        "halo_radius_at_dz_max_um": w_at_max,
        "n_grid_needed_for_dz_max": n_needed,
        "far_field_size_needed_at_padding8": n_needed * 8,
        "note": (
            f"当前整形网格 (n_grid={n_grid_hint}) 半窗 {half_window_um:.0f} µm, "
            f"覆盖到 Δz≤{max(trusted) if trusted else 0:g} mm, "
            f"首个不可信点在 Δz={next((r['dz_mm'] for r in per_dz if not r['trusted']), None)} mm。"
            f"要覆盖 Δz={dz_max:g} mm 需要 n_grid≈{n_needed} "
            f"(padding=8 ⇒ far_field {n_needed * 8}²)。"
            "padding 加大没用 —— 窗口只随 n_grid 变。"
        ),
    }


def rc_dz(per_dz: list[dict]) -> list[float]:
    return [r["dz_mm"] for r in per_dz]


def offset_for_tilt(
    halo_row: dict | None, rc: RunConfig
) -> tuple[tuple[int, int], bool]:
    """H4 的框偏移: 把目标框挪到零级晕斑 1/e² 半径之外。

    需要 ``side/2 + K·w_halo + margin`` 像素的位移; 若半窗放不下就**封顶**并
    返回 ``False``, 报告里据实写「该 Δz 下 H4 已无法把晕斑挪出靶区」。
    """
    side = rc.target_side_px
    half_window = rc.m // 2 - side // 2 - 3
    if half_window <= 0:
        return (0, 0), False
    if halo_row is None:
        return (0, 0), True
    need = side / 2.0 + 1.6 * float(halo_row["halo_w_1e2_px"]) + 3.0
    disp = int(math.ceil(need))
    if disp > half_window:
        return (half_window, 0), False
    return (disp, 0), True


def anchor_radius(side_px: int) -> int:
    """信号定位窗口的半径 (px): 目标框半边长 + 3。

    必须**小于**目标框心到晕斑的距离 (:func:`offset_for_tilt` 给的
    ``side/2 + 1.6·w_halo + 3``), 否则窗口会碰到轴上的零级晕斑, 把框心往晕斑拽。
    H4 的偏移恒 > 本半径, 所以晕斑 (它留在光轴上) 一定落在窗口外。
    """
    return int(side_px) // 2 + 3


def anchor_box_center(
    inten: np.ndarray, target_center: tuple[int, int], radius: int
) -> tuple[tuple[int, int], tuple[float, float]]:
    """在 ``target_center`` 附近 ``radius`` 像素的**窗口**内量信号质心, 返回冻结的整数框心。

    返回 ``((row, col), (centroid_row, centroid_col))`` —— 前者是真正用于度量的整数
    框心, 后者是亚像素质心 (QC)。

    **只在 ``(口径, Δz)`` 上定位一次** (见 :func:`run_groups`), 同一口径的所有帧
    (不同 η、带限/无带限正演、量化/连续相位) 共用它。**绝不在度量循环里重定位**:
    类散斑场的 ``argmax`` 在 ~1e-3 扰动下会跳到另一颗晶粒上, 滚动框让 CV/PIB 不连续
    (仓库明令的坑)。

    亚像素质心**不移动**整数框: 实测基线信号质心 (255.4996, 255.4999), 直接取整会把
    框整体挪 1 px (score 0.8118 → 0.8073, 0.5% 差) —— 那是测量噪声不是信号跑偏。
    所以只有 |质心 − 目标框心| ≥ 1 px 才移动框; 那才说明信号真的不在靶上 (例如 H4
    的倾斜没把光斑移到位)。

    **晕斑被排除在窗口之外**: 窗口半径取 :func:`anchor_radius` (目标框半边长 + 3),
    框心取**目标**位置 —— H4 把目标框整体移出晕斑半径 (见 :func:`offset_for_tilt`,
    偏移恒 > 窗口半径), 于是轴上的零级晕斑落在窗口外, 不会被当成信号。H0~H3 无倾斜,
    晕斑与信号同在画框中心 (物理上就重叠), 窗口覆盖的仍是信号本身。
    """
    m = inten.shape[0]
    r0, c0 = int(target_center[0]), int(target_center[1])
    r_lo, r_hi = max(r0 - radius, 0), min(r0 + radius + 1, m)
    c_lo, c_hi = max(c0 - radius, 0), min(c0 + radius + 1, m)
    win = inten[r_lo:r_hi, c_lo:c_hi]
    tot = float(win.sum())
    if tot <= 0.0 or win.size == 0:
        return (r0, c0), (float("nan"), float("nan"))
    rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
    centroid_r = float((win * rr).sum() / tot)
    centroid_c = float((win * cc).sum() / tot)
    dr = 0 if abs(centroid_r - r0) < 1.0 else int(round(centroid_r - r0))
    dc = 0 if abs(centroid_c - c0) < 1.0 else int(round(centroid_c - c0))
    return (r0 + dr, c0 + dc), (centroid_r, centroid_c)


def get_anchor(
    anchors: dict[tuple[tuple[int, int], float], tuple[tuple[int, int], tuple[float, float]]],
    bench: TwoComponentBench,
    rc: RunConfig,
    offset_px: tuple[int, int],
    dz_mm: float,
    inten_ref: np.ndarray,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """取 (或首次定位) 该 ``(口径, Δz)`` 的冻结框心。口径 = ``offset_px``。"""
    key = (tuple(int(v) for v in offset_px), float(dz_mm))
    if key not in anchors:
        m = rc.bench_cfg().far_field_size
        target_center = (m // 2 + int(offset_px[0]), m // 2 + int(offset_px[1]))
        anchors[key] = anchor_box_center(
            inten_ref, target_center, anchor_radius(rc.target_side_px)
        )
    return anchors[key]


def run_groups(
    bench: TwoComponentBench, rc: RunConfig, ref_sigma: float, halo: list[dict]
) -> tuple[list[dict], dict[tuple[float, float], np.ndarray], dict]:
    """H0~H4 × Δz × η 的主扫描。

    顺带把 H2 的相位按 ``(Δz, η)`` 缓存下来给热启动对照当初值 —— 那正是
    「上一组解」, 重跑一遍只是把同一条轨迹换个名字再算一次。

    ⚠️ **上一版的扫描路径 bug (已修)**: 度量框的位置曾经一律取 H4 的倾斜偏移, 于是
    H0~H3 的框被整体挪出信号 (Δz=0 时挪 23 px), 框里剩不到 0.4% 的光 → PIB 从
    0.915 塌到 0.0034 (270×), 而 ``pib_box``/``score_box`` 在非零偏移分支里直接写
    ``NaN``。现在**每行用自己的口径** (:attr:`GsSetup.offset_px`), 框心锚在信号上
    (见 :func:`anchor_box_center`), ``*_box`` 恒可计算。
    """
    halo_by_dz = {float(r["dz_mm"]): r for r in halo}
    rows: list[dict] = []
    h2_cache: dict[tuple[float, float], np.ndarray] = {}
    anchors: dict[tuple[tuple[int, int], float], tuple] = {}
    etas = [e for e in rc.eta_values if e > 0.0]
    for dz_mm in rc.dz_mm:
        dz = dz_mm * 1e-3
        offset, fits = offset_for_tilt(halo_by_dz.get(dz_mm), rc)
        halo_ratio = halo_by_dz.get(dz_mm, {}).get("halo_w_over_shape_halfwindow")
        for group in GROUPS:
            group_etas = [0.0] if group == "H0" else etas
            for eta in group_etas:
                setup = setup_for(group, eta, offset)
                out = bench.gs_phase_eta(
                    dz,
                    eta=setup.eta,
                    n_iters=rc.gs_iters,
                    seed=rc.seed,
                    weighted=setup.weighted,
                    subtract=setup.subtract,
                    offset_px=setup.offset_px,
                    quantize=True,
                )
                if group == "H2":
                    h2_cache[(dz_mm, eta)] = out["phase"]
                # 信号定位基准帧 = 本行的量化相位在**配置正演**下的强度
                inten_ref = bench.intensity_eta(out["phase"], setup.eta, dz)
                center, anchor_centroid = get_anchor(
                    anchors, bench, rc, setup.offset_px, dz_mm, inten_ref
                )
                rows.append(
                    _row(
                        bench,
                        rc,
                        setup,
                        dz_mm,
                        fits,
                        out["phase"],
                        out["phase_cont"],
                        center,
                        anchor_centroid,
                        ref_sigma,
                        halo_ratio=halo_ratio,
                        warm=False,
                    )
                )
                logger.info(
                    "  {g:<3} η={e:<5.2f} Δz={d:>5.1f}mm  score={s:.4f} "
                    "CV={cv:.4f} PIB={p:.4f}  act={a:.4f}  box@({cy},{cx})",
                    g=group,
                    e=eta,
                    d=dz_mm,
                    s=rows[-1]["score"],
                    cv=rows[-1]["cv"],
                    p=rows[-1]["pib"],
                    a=rows[-1]["score_adapt"],
                    cy=center[0],
                    cx=center[1],
                )
    return rows, h2_cache, anchors


def _row(
    bench: TwoComponentBench,
    rc: RunConfig,
    setup: GsSetup,
    dz_mm: float,
    fits: bool,
    phase: np.ndarray,
    phase_cont: np.ndarray,
    center: tuple[int, int],
    anchor_centroid: tuple[float, float],
    ref_sigma: float,
    *,
    warm: bool,
    halo_ratio: float | None = None,
    group: str | None = None,
) -> dict:
    """一次 GS 运行的完整指标行。

    度量分三层, 全部写进同一行:

    * **主口径 ``score``/``cv``/``pib``** = **固定物理尺度框** (边长恒为
      ``target_side_px``, 框心锚在信号上并冻结)。``*_cont`` 用连续相位,
      ``*_bl`` 用带限正演 (数值误差带)。**结论只准用这三列。**
    * **``*_adapt``** = 自适应框 (质心跟踪 + σ 缩放边长)。**只作对照** —— 框跟着
      光斑一起变大会把「测量区域变大」当成「均匀性变好」, 本调查里这个陷阱出现过
      三次 (见模块 docstring 与 summary 的 ``metric_convention``)。
    * **``*_box``** = data.json 的**居中定尺框** (``make_square_target`` 口径),
      连续相位。它与前两层不同, 因为它**不锚在信号上** (H4 的晕斑就在它中心),
      但它对任何口径都恒可计算, 且 Δz=0/η=0 上逐位复现基线 —— 这就是永久性的
      「扫描路径没被改坏」判据 (由 :func:`self_check_scan_baseline` 主动 raise)。
    """
    dz = dz_mm * 1e-3
    cfg = rc.bench_cfg()
    group = setup.group if group is None else group
    side = int(rc.target_side_px)
    crow, ccol = int(center[0]), int(center[1])
    m = cfg.far_field_size
    target_center = (m // 2 + int(setup.offset_px[0]), m // 2 + int(setup.offset_px[1]))
    inten_ref = bench.intensity_eta(phase, setup.eta, dz)
    cx_f, cy_f, sigma = bench.centroid_sigma(inten_ref)
    ac_r = float(anchor_centroid[0])
    ac_c = float(anchor_centroid[1])
    out: dict[str, Any] = {
        "group": group,
        "group_label": GROUP_LABELS[group],
        "eta": setup.eta,
        "dz_mm": dz_mm,
        "offset_px": [int(setup.offset_px[0]), int(setup.offset_px[1])],
        "tilt_fits_window": fits,
        "warm_start": warm,
        "band_limit": bench.band_limit,
        # 信号定位 (QC): 冻结框心 vs 目标框心 vs **窗口内**亚像素质心 vs 全帧质心
        "signal_center_px": [crow, ccol],
        "target_center_px": [int(target_center[0]), int(target_center[1])],
        "signal_centroid_px": [round(ac_r, 4), round(ac_c, 4)],
        "signal_anchor_shift_px": [crow - int(target_center[0]), ccol - int(target_center[1])],
        "signal_anchor_ok": bool(
            math.hypot(ac_r - target_center[0], ac_c - target_center[1]) <= 3.0
        ),
        "frame_centroid_px": [round(cy_f, 4), round(cx_f, 4)],
        "sigma_px": float(sigma),
    }
    for tag, ph in (("", phase), ("_cont", phase_cont)):
        # ``*_bl`` 是**同一组解**在带限正演下的分数: 两者之差 = 这条指标的数值
        # 误差带 (带限在 Δz≳10 mm 上漏 0.7%~22%, 与入射场谱宽有关)。
        # 注意它**不是**「带限版 GS 解的分数」—— 那要整条轨迹重跑一遍。
        for suffix, bl in (("_bl", True), ("", bench.band_limit)):
            inten = bench.intensity_eta(ph, setup.eta, dz, band_limit=bl)
            pib, cv = bench.box_stats(inten, ccol, crow, side)
            out[f"score{tag}{suffix}"] = composite_from_pib_cv(pib, cv)
            out[f"cv{tag}{suffix}"] = cv
            out[f"pib{tag}{suffix}"] = pib
            out[f"d90_px{tag}{suffix}"] = float(measure_spot_diameter_cam(inten, energy=0.90))
    # 自适应框 (质心跟踪 + σ 缩放边长) —— 只作对照, 不作结论
    adapt = bench.evaluate(inten_ref, ref_sigma=ref_sigma, target_side_px=side)
    out["score_adapt"] = adapt["score"]
    out["cv_adapt"] = adapt["cv"]
    out["pib_adapt"] = adapt["pib"]
    out["side_adapt_px"] = adapt["side_px"]
    # data.json 居中定尺框口径 (连续相位) —— 恒可计算, 永不 NaN
    dj = compute_bench_metrics(
        bench.intensity_eta(phase_cont, setup.eta, dz, band_limit=bench.band_limit),
        make_square_target(cfg),
    )
    out["score_box"] = composite_score(dj)
    out["cv_box"] = dj["CV"]
    out["pib_box"] = dj["PIB"]
    if halo_ratio is not None:
        out["halo_w_over_shape_halfwindow"] = float(halo_ratio)
        out["trusted"] = bool(float(halo_ratio) <= HALO_WRAP_RATIO_LIMIT)
    return out


def run_warm_start(
    bench: TwoComponentBench,
    rc: RunConfig,
    ref_sigma: float,
    h2_phases: dict[tuple[float, float], np.ndarray],
    halo: list[dict],
    anchors: dict,
) -> tuple[list[dict], list[dict]]:
    """热启动 (用同 (Δz, η) 的 H2 解) vs 随机初值, 比较收敛速度。

    热启动源取主扫描里已经跑过的 H2 相位 (同 η、同 Δz、同目标框), 不重跑一遍
    plain GS —— 那只是把同一条轨迹换个名字再算一次。度量框与主扫描共用同一份
    ``anchors`` (同口径 ⇒ 同框心), 保证两组数字可比。
    """
    halo_by_dz = {float(r["dz_mm"]): r for r in halo}
    warm_rows: list[dict] = []
    traj: list[dict] = []
    for dz_mm in rc.dz_mm:
        dz = dz_mm * 1e-3
        offset, fits = offset_for_tilt(halo_by_dz.get(dz_mm), rc)
        halo_ratio = halo_by_dz.get(dz_mm, {}).get("halo_w_over_shape_halfwindow")
        for eta in [e for e in rc.eta_values if e > 0.0]:
            setup = GsSetup("H3", eta, weighted=True, subtract=True, offset_px=offset)
            rnd = bench.gs_phase_eta(
                dz,
                eta=eta,
                n_iters=rc.gs_iters,
                seed=rc.seed,
                weighted=True,
                subtract=True,
                offset_px=offset,
                quantize=True,
                record_history=True,
            )
            warm = bench.gs_phase_eta(
                dz,
                eta=eta,
                n_iters=rc.gs_iters,
                seed=rc.seed,
                weighted=True,
                subtract=True,
                offset_px=offset,
                quantize=True,
                warm_phase=h2_phases[(dz_mm, eta)],
                record_history=True,
            )
            inten_ref = bench.intensity_eta(warm["phase"], setup.eta, dz)
            center, anchor_centroid = get_anchor(
                anchors, bench, rc, setup.offset_px, dz_mm, inten_ref
            )
            row = _row(
                bench,
                rc,
                setup,
                dz_mm,
                fits,
                warm["phase"],
                warm["phase_cont"],
                center,
                anchor_centroid,
                ref_sigma,
                warm=True,
                halo_ratio=halo_ratio,
            )
            row["score_random"] = _last(rnd["history"])
            row["score_warm"] = _last(warm["history"])
            row["iters_to_95pct_warm"] = _iters_to_fraction(warm["history"], 0.95)
            row["iters_to_95pct_random"] = _iters_to_fraction(rnd["history"], 0.95)
            row["iters_to_99pct_warm"] = _iters_to_fraction(warm["history"], 0.99)
            row["iters_to_99pct_random"] = _iters_to_fraction(rnd["history"], 0.99)
            warm_rows.append(row)
            for label, out in (("random", rnd), ("warm", warm)):
                for it, v in zip(out["history_iters"], out["history"]):
                    traj.append(
                        {
                            "group": "H3",
                            "eta": eta,
                            "dz_mm": dz_mm,
                            "init": label,
                            "iter": it,
                            "score": v,
                        }
                    )
            logger.info(
                "  热启动 η={:.2f} Δz={:.0f}mm  warm={:.4f} random={:.4f}  "
                "95%@warm={} 95%@random={}",
                eta,
                dz_mm,
                row["score_warm"],
                row["score_random"],
                row["iters_to_95pct_warm"],
                row["iters_to_95pct_random"],
            )
    return warm_rows, traj


def _last(history: list[float]) -> float:
    return float(history[-1]) if history else float("nan")


def _iters_to_fraction(history: list[float], frac: float) -> int:
    """达到最终分数 ``frac`` 倍所需的迭代数 (收敛速度的单一数字)。"""
    if not history:
        return -1
    target = frac * max(history[-1], 0.0)
    for i, v in enumerate(history):
        if v >= target:
            return i
    return len(history)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(rc: RunConfig, rows: list[dict], trusted_dz: list[float] | None = None) -> dict:
    """每组的天花板: best score / min CV / 达到它的参数。

    ``trusted_dz`` 给定时只在该 Δz 集合上统计 —— 晕斑超窗回绕的点 (见
    :func:`measure_trustworthy_dz`) 的 CV/PIB 是 FFT 周期性的数值伪影, 混进
    天花板只会被伪影抬高。
    """
    out: dict[str, Any] = {}
    pool = rows
    if trusted_dz is not None:
        pool = [r for r in rows if any(abs(r["dz_mm"] - d) < 1e-12 for d in trusted_dz)]
    for group in GROUPS:
        sel = [r for r in pool if r["group"] == group]
        if not sel:
            continue
        best = max(sel, key=lambda r: r["score"])
        lowest_cv = min(sel, key=lambda r: r["cv"])
        out[group] = {
            "label": GROUP_LABELS[group],
            "n_runs": len(sel),
            "best_score": best["score"],
            "best_score_eta": best["eta"],
            "best_score_dz_mm": best["dz_mm"],
            "best_score_cv": best["cv"],
            "best_score_pib": best["pib"],
            "best_score_dcv_band": best["cv"] - best["cv_bl"],
            "min_cv": lowest_cv["cv"],
            "min_cv_eta": lowest_cv["eta"],
            "min_cv_dz_mm": lowest_cv["dz_mm"],
            "min_cv_score": lowest_cv["score"],
        }
    return out


def band_limit_band(rows: list[dict]) -> list[dict]:
    """逐 Δz 的数值不确定度带: 同一组解在带限/无带限正演下的 CV 与 score 之差。

    带限一开, 大 Δz 上漏掉几个百分点的能量, 指标就跟着漂。所以**任何单报一个
    数的对比都要先看这一列** —— 组间差若落在带子里就不算差异。
    """
    out: list[dict] = []
    for dz in sorted({float(r["dz_mm"]) for r in rows}):
        sel = [r for r in rows if abs(float(r["dz_mm"]) - dz) < 1e-12]
        out.append(
            {
                "dz_mm": dz,
                "n_rows": len(sel),
                "max_abs_dcv": max(abs(r["cv_bl"] - r["cv"]) for r in sel),
                "max_rel_dcv": max(
                    abs(r["cv_bl"] - r["cv"]) / max(abs(r["cv"]), 1e-12) for r in sel
                ),
                "max_abs_dscore": max(abs(r["score_bl"] - r["score"]) for r in sel),
            }
        )
    return out


def subtraction_benefit(rows: list[dict]) -> list[dict]:
    """逐 ``(Δz, η)`` 的**零级相干扣除收益** (固定物理尺度框为主口径)。

    * ``dh2_minus_h1_*`` = H2 − H1 —— 只多一步**相干扣除** (同为 plain GS);
    * ``dh3_minus_h1_*`` = H3 − H1 —— 加权 GS **且**扣除 (相对纯 GS 的总收益)。

    ``score``/``cv``/``pib`` 三列与自适应框版 (``h*_score_adapt``) 并列。
    **结论只准用固定框版** (见 summary 的 ``metric_convention``)。
    """
    out: list[dict] = []
    etas = sorted({float(r["eta"]) for r in rows if float(r["eta"]) > 0.0})
    for dz in sorted({float(r["dz_mm"]) for r in rows}):
        for eta in etas:
            sel = {
                r["group"]: r
                for r in rows
                if abs(float(r["dz_mm"]) - dz) < 1e-12
                and abs(float(r["eta"]) - eta) < 1e-12
            }
            if not {"H1", "H2", "H3"} <= set(sel):
                continue
            h1, h2, h3 = sel["H1"], sel["H2"], sel["H3"]
            out.append(
                {
                    "dz_mm": dz,
                    "eta": eta,
                    "trusted": bool(h1.get("trusted", False)),
                    "halo_w_over_shape_halfwindow": h1.get(
                        "halo_w_over_shape_halfwindow"
                    ),
                    "h1_score": h1["score"],
                    "h2_score": h2["score"],
                    "h3_score": h3["score"],
                    "h1_cv": h1["cv"],
                    "h2_cv": h2["cv"],
                    "h3_cv": h3["cv"],
                    "h1_pib": h1["pib"],
                    "h2_pib": h2["pib"],
                    "h3_pib": h3["pib"],
                    "dh2_minus_h1_score": h2["score"] - h1["score"],
                    "dh2_minus_h1_cv": h2["cv"] - h1["cv"],
                    "dh2_minus_h1_pib": h2["pib"] - h1["pib"],
                    "dh3_minus_h1_score": h3["score"] - h1["score"],
                    "dh3_minus_h1_cv": h3["cv"] - h1["cv"],
                    "dh3_minus_h1_pib": h3["pib"] - h1["pib"],
                    "h1_score_adapt": h1["score_adapt"],
                    "h2_score_adapt": h2["score_adapt"],
                    "h3_score_adapt": h3["score_adapt"],
                }
            )
    return out


def hardware_advice(rows: list[dict], rc: RunConfig) -> dict[str, Any]:
    """给硬件实验的一条明确建议: **什么 Δz、什么 η 下值得做零级相干扣除**。

    判据取**固定物理尺度框**上的 ``score = 0.5·PIB + 0.5/(1+CV)`` 增益
    (``H2 − H1``): 增益为正说明「多拍一张零级参考图 (关调制) 并把它从回代场里
    减掉」是划算的。同时报 CV / PIB 各自的增益方向 —— 它们常常反向 (扣除抬 PIB
    但压不平 CV, 因为扣除只作用在 GS 的目标面约束之前)。

    ``trusted`` 直接沿用晕斑回绕判据 (:func:`measure_trustworthy_dz`): Δz 大到晕斑
    1/e² 半径超过整形网格半窗时, 该点的 CV/PIB 是 FFT 周期性卷回的伪影, 不参与建议。
    """
    benefit = subtraction_benefit(rows)
    per_dz: dict[float, list[dict]] = {}
    for b in benefit:
        per_dz.setdefault(float(b["dz_mm"]), []).append(b)
    by_dz: dict[str, Any] = {}
    for dz, bs in sorted(per_dz.items()):
        bs = sorted(bs, key=lambda x: x["eta"])
        by_dz[f"{dz:g}mm"] = {
            "trusted": bool(bs[0]["trusted"]),
            "eta_where_subtraction_raises_score": [
                x["eta"] for x in bs if x["dh2_minus_h1_score"] > 0
            ],
            "eta_where_subtraction_lowers_cv": [
                x["eta"] for x in bs if x["dh2_minus_h1_cv"] < 0
            ],
            "eta_where_subtraction_raises_pib": [
                x["eta"] for x in bs if x["dh2_minus_h1_pib"] > 0
            ],
            "best_dscore_h2_minus_h1": max(x["dh2_minus_h1_score"] for x in bs),
            "worst_dscore_h2_minus_h1": min(x["dh2_minus_h1_score"] for x in bs),
            "best_dscore_h3_minus_h1": max(x["dh3_minus_h1_score"] for x in bs),
        }
    trusted_dz = [d for d, v in by_dz.items() if v["trusted"]]
    # 「扣除在多大 η 以上开始必要」= 在**可信** Δz 上, 扣除不再给出正 score 增益的 η 之下
    beneficial = [
        x["eta"]
        for b in benefit
        if b["trusted"] and b["dh2_minus_h1_score"] > 0
        for x in [b]
    ]
    return {
        "conclusion_rule": (
            "建议只用**固定物理尺度框**上的 score 增益 (H2−H1) 判定; "
            "自适应框版单列, 不作依据。"
        ),
        "by_dz": by_dz,
        "trusted_dz": trusted_dz,
        "eta_where_subtraction_beneficial_on_trusted_dz": sorted(set(beneficial)),
        "caliber_caveat": (
            f"本建议来自 {rc.n_grid}² 瞳面 / padding={rc.far_field_padding} / "
            f"target_side={rc.target_side_px}px / gs_iters={rc.gs_iters} 的口径; "
            "换实验系统的焦距/口径/像素请重跑 (远场像元与窗口都会变)。"
        ),
    }


def dz_choice_table(halo: list[dict]) -> dict[str, Any]:
    """参考材料那个「选 d」的公式的验证表。

    判据: 晕斑峰值 ``< γ · 目标区平均强度``。本表给出每个 (η, Δz) 的实测比值
    ``I_halo_peak / I_box_mean``, 以及落在 ``γ = 1.0 / 0.3 / 0.1`` 哪一档。
    """
    table: dict[str, Any] = {"gamma_ladder": list(GAMMA_LADDER), "by_eta": {}}
    for eta in sorted({float(r["eta"]) for r in halo}):
        rows = sorted(
            (r for r in halo if abs(r["eta"] - eta) < 1e-12),
            key=lambda r: r["dz_mm"],
        )
        ratios = [float(r["halo_peak_over_box_mean"]) for r in rows]
        entry: dict[str, Any] = {
            "dz_mm": [r["dz_mm"] for r in rows],
            "ratio": ratios,
            "i_halo_peak": [float(r["i_halo_peak"]) for r in rows],
            "halo_w_1e2_um": [float(r["halo_w_1e2_um"]) for r in rows],
        }
        for g in GAMMA_LADDER:
            ok = [r["dz_mm"] for r, v in zip(rows, ratios) if v < g]
            entry[f"dz_below_gamma_{g}"] = ok
        table["by_eta"][f"{eta:g}"] = entry
    return table


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_halo(halo: list[dict], out: Path, rc: RunConfig) -> None:
    """零级晕斑峰值 vs Δz (含菲涅耳高斯解析预言), 附 1/e² 半径与场窗口的比较。"""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    etas = sorted({float(r["eta"]) for r in halo})
    for eta in etas:
        rows = sorted((r for r in halo if r["eta"] == eta), key=lambda r: r["dz_mm"])
        x = [max(r["dz_mm"], 1e-3) for r in rows]
        axes[0].plot(
            x,
            [r["i_halo_peak"] for r in rows],
            "o-",
            label=f"η={eta:g}",
            linewidth=2,
        )
        axes[1].plot(x, [r["halo_w_1e2_um"] for r in rows], "o-", label=f"η={eta:g}")
        axes[2].plot(
            x, [r["halo_peak_over_box_mean"] for r in rows], "o-", label=f"η={eta:g}"
        )
    # 晕斑超出整形网格半窗 ⇒ 被 FFT 周期性卷回, 该 Δz 的指标是数值伪影
    shape_half_um = rc.m * rc.bench_cfg().far_field_pixel_size * 1e6 / 2.0
    eta0 = sorted(
        (r for r in halo if float(r["eta"]) == etas[0]), key=lambda r: r["dz_mm"]
    )
    bad = [r["dz_mm"] for r in eta0 if not r["trusted"]]
    if bad:
        for ax in axes:
            ax.axvspan(
                min(bad),
                max(r["dz_mm"] for r in eta0) * 1.15,
                color="red",
                alpha=0.12,
                label="晕斑回绕区 (不可信)",
            )
    first = eta0
    peak0 = next(r["i_halo_peak"] for r in first if r["dz_mm"] == 0.0)
    zr = first[0].get("z_rayleigh_mm") or float("nan")
    axes[0].plot(
        [r["dz_mm"] for r in first],
        [peak0 * _fresnel_gaussian_ratio(r["dz_mm"] / zr) for r in first],
        "k-",
        linewidth=1.6,
        label=f"菲涅耳高斯 1/(1+(Δz/z_R)²), z_R={zr:.2f}mm",
    )
    axes[0].plot(
        [r["dz_mm"] for r in first],
        [r["i_halo_peak_bl"] for r in first],
        "k--",
        linewidth=1.2,
        label="带限正演 (数值误差带)",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("CCD 焦后位移 Δz (mm)")
    axes[0].set_ylabel("零级晕斑峰值 (sum=1 归一 × η²)")
    axes[0].set_title("晕斑峰值单调下降 (I ∝ f²/d²)")
    axes[2].axhline(GAMMA_REF, color="k", linestyle="--", linewidth=1.2,
                    label=f"γ={GAMMA_REF:g}")
    for g in GAMMA_LADDER[1:]:
        axes[2].axhline(g, color="gray", linestyle=":", linewidth=1.0,
                        label=f"γ={g:g}")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("CCD 焦后位移 Δz (mm)")
    axes[2].set_ylabel("晕斑峰值 / 目标区平均强度")
    axes[2].set_title("参考材料的选 d 判据")
    axes[1].axhline(
        shape_half_um,
        color="k",
        linestyle="--",
        linewidth=1.2,
        label=f"整形网格半窗 {shape_half_um:.0f}µm",
    )
    axes[1].axhline(
        first[0]["half_window_um"],
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"halo 网格半窗 {first[0]['half_window_um']:.0f}µm",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("CCD 焦后位移 Δz (mm)")
    axes[1].set_ylabel("晕斑 1/e² 半径 (µm)")
    axes[1].set_title("晕斑 vs 整形网格半窗 (超出即被 FFT 卷回)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"零级晕斑诊断 (halo 网格 n_grid={rc.halo_n_grid}, λ={defo.WAVELENGTH_NM:.0f}nm "
        f"f={FOCAL_LENGTH_M * 1e3:.1f}mm D={APERTURE_M * 1e3:.2f}mm)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_subtraction(rows: list[dict], out: Path) -> None:
    """扣除前后 CV (H1 vs H2), 每个 η 一条曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    etas = sorted({r["eta"] for r in rows if r["group"] == "H2"})
    for eta in etas:
        for ax, group, title in (
            (axes[0], "H1", "H1 不做相干扣除"),
            (axes[1], "H2", "H2 做相干扣除"),
        ):
            sel = sorted(
                (r for r in rows if r["group"] == group and r["eta"] == eta),
                key=lambda r: r["dz_mm"],
            )
            ax.plot(
                [r["dz_mm"] for r in sel],
                [r["cv"] for r in sel],
                "o-",
                label=f"η={eta:g}",
                linewidth=2,
            )
            ax.set_title(title)
            ax.set_yscale("log")
            ax.set_xlabel("CCD 焦后位移 Δz (mm)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
    axes[0].set_ylabel("目标框内 CV")
    fig.suptitle("相干扣除对均匀性的影响 (量化后的 SLM 相位)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ceilings(summary: dict, out: Path) -> None:
    """各组天花板: best score 与 min CV。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = [g for g in GROUPS if g in summary]
    axes[0].bar(labels, [summary[g]["best_score"] for g in labels],
                color=["#9E9E9E", "#FF5722", "#4CAF50", "#2196F3", "#673AB7"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("best score")
    axes[0].set_title("各组天花板 (best score)")
    axes[1].bar(labels, [summary[g]["min_cv"] for g in labels],
                color=["#9E9E9E", "#FF5722", "#4CAF50", "#2196F3", "#673AB7"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("min CV")
    axes[1].set_title("各组天花板 (min CV)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    for g in labels:
        s = summary[g]
        axes[0].annotate(
            f"η={s['best_score_eta']:g}\nΔz={s['best_score_dz_mm']:g}mm",
            (labels.index(g), s["best_score"]),
            textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8,
        )
    fig.suptitle("H0~H4 对照: 扣除与加权 GS 的天花板")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_groups_vs_dz(rows: list[dict], out: Path) -> None:
    """score vs Δz, 每组一条曲线 (取该 η 下最好的一组 η 展示)。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    colors = {"H0": "#9E9E9E", "H1": "#FF5722", "H2": "#4CAF50",
              "H3": "#2196F3", "H4": "#673AB7"}
    for group in GROUPS:
        sel = [r for r in rows if r["group"] == group]
        if not sel:
            continue
        eta = max({r["eta"] for r in sel})
        sub = sorted((r for r in sel if r["eta"] == eta), key=lambda r: r["dz_mm"])
        ax.plot(
            [r["dz_mm"] for r in sub], [r["score"] for r in sub], "o-",
            color=colors[group], label=f"{group} (η={eta:g})", linewidth=2,
        )
    ax.set_xlabel("CCD 焦后位移 Δz (mm)")
    ax.set_ylabel("score = 0.5·PIB + 0.5/(1+CV)")
    ax.set_title("各组 score vs Δz (量化相位)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="小网格冒烟跑")
    p.add_argument("--tag", default=None, help="产物文件名后缀 (默认 _improved)")
    p.add_argument("--out", default=None, help="产物目录 (默认 report/shape_trials)")
    p.add_argument(
        "--band-limit",
        action="store_true",
        help=(
            "打开权威传播器的 Voelz & Roggemann 带限 (默认关)。带限有损且不可逆, "
            "正/反向不构成伴随对 —— 打开会在 round_trip_relerr 自检里直接 raise"
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    rc = QUICK if args.quick else FULL
    if args.tag is not None:
        rc = replace(rc, tag=args.tag)
    rc = replace(rc, band_limit=bool(args.band_limit))
    out = Path(args.out) if args.out else ROOT / "report" / "shape_trials"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    logger.info(
        "光学常量: λ={:.0f}nm  f={:.2f}mm  D={:.3f}mm  面板像元={:.1f}um",
        defo.WAVELENGTH_NM, FOCAL_LENGTH_M * 1e3, APERTURE_M * 1e3, PANEL_PIXEL_UM,
    )
    logger.info(
        "整形网格: n_grid={}  padding={}  m={}  远场像元={:.4f}um  窗口={:.4f}mm "
        "(半窗 {:.1f}um)  target_side={}px  gs_iters={}",
        rc.n_grid, rc.far_field_padding, rc.m,
        rc.bench_cfg().far_field_pixel_size * 1e6,
        rc.m * rc.bench_cfg().far_field_pixel_size * 1e3,
        rc.m * rc.bench_cfg().far_field_pixel_size * 1e6 / 2,
        rc.target_side_px, rc.gs_iters,
    )
    logger.info(
        "带限 band_limit={} (权威件默认值是 True, 本实验 Δz>0 一律关 —— "
        "带限有损且不可逆, 见 round_trip_relerr 自检)",
        rc.band_limit,
    )

    # --- 0. data.json 基线核对 (永远是 data.json 的口径, 不随 --quick 变) -----
    base_cfg = SquareShapingConfig(
        n_grid=64,
        target_side_px=43,
        far_field_padding=8,
        seed=0,
        wavelength=WAVELENGTH_M,
        focal_length=FOCAL_LENGTH_M,
        aperture_size=APERTURE_M,
    )
    base_bench = TwoComponentBench(base_cfg, band_limit=rc.band_limit)
    logger.info("基线核对: η=0/Δz=0/gs_iters={} vs data.json :: gs ...", rc.baseline_iters)
    gs_delta = self_check_eta0(base_bench, rc.baseline_iters, 0)
    base_phase = base_bench.gs_phase_eta(
        0.0, eta=0.0, n_iters=rc.baseline_iters, seed=0, quantize=False
    )["phase_cont"]
    base_inten = base_bench.intensity_eta(base_phase, 0.0, 0.0)
    base_m = compute_bench_metrics(base_inten, make_square_target(base_cfg))
    base_score = composite_score(base_m)
    base_sigma = base_bench.centroid_sigma(base_inten)[2]
    base_delta = {
        "score": base_score - DATA_JSON_GS["score"],
        "PIB": base_m["PIB"] - DATA_JSON_GS["PIB"],
        "CV": base_m["CV"] - DATA_JSON_GS["CV"],
    }
    logger.info(
        "基线: score={:.6f} PIB={:.6f} CV={:.6f}  (data.json: {:.6f}/{:.6f}/{:.6f})",
        base_score, base_m["PIB"], base_m["CV"],
        DATA_JSON_GS["score"], DATA_JSON_GS["PIB"], DATA_JSON_GS["CV"],
    )
    logger.info("η=0 与 gs_shape 逐位一致: max|Δphase|={:.3e}", gs_delta)
    # 硬断言: 基线核对不是日志而是闸门 —— 度量口径 (目标框/归一化) 一旦被改坏,
    # 这里直接 raise, 不用等到扫描跑完才从数字上看出不对。
    if max(abs(v) for v in base_delta.values()) > 1e-6:
        raise AssertionError(
            "data.json 基线核对失败: max|Δ|={:.3e} > 1e-6 (measured {}/{}/{} vs {}). "
            "度量口径 (目标框、归一化、正演) 被改坏了".format(
                max(abs(v) for v in base_delta.values()),
                base_score, base_m["PIB"], base_m["CV"], DATA_JSON_GS,
            )
        )
    logger.info("扫描路径基线断言 (Δz=0/η=0, 自己那条代码路径) ...")
    scan_baseline = self_check_scan_baseline(tolerance=1e-6)
    logger.info(
        "扫描路径 ✓ score={:.6f} PIB={:.6f} CV={:.6f}  框心={} 侧={}px  质心={}",
        scan_baseline["measured"]["score"], scan_baseline["measured"]["PIB"],
        scan_baseline["measured"]["CV"], scan_baseline["box_center_px"],
        scan_baseline["box_side_px"], scan_baseline["signal_centroid_px"],
    )

    # --- 1. 晕斑诊断 (宽网格) ------------------------------------------------
    logger.info("零级晕斑诊断 (halo 网格 n_grid={}) ...", rc.halo_n_grid)
    halo, halo_bench = measure_halo(rc)
    halo_check = self_check_halo_monotonic(halo)
    logger.info(
        "晕斑峰值单调下降 ✓ 最小相邻比={:.4f}  I·Δz² 相对展布={:.3f}  "
        "z_R={:.3f}mm",
        halo_check["worst_adjacent_ratio"],
        halo_check["max_I_dz2_spread"],
        halo[0]["z_rayleigh_mm"] or float("nan"),
    )

    # --- 2. 主扫描 ----------------------------------------------------------
    cfg = rc.bench_cfg()
    bench = TwoComponentBench(cfg, band_limit=rc.band_limit)
    # ref_sigma 必须量在**当前网格自己的像素单位**上: box 的边长按
    # ``side = target_side_px · σ/ref_sigma`` 定, 而 σ 是像素数。混用另一张网格
    # (padding 不同 ⇒ 像元宽度差一倍) 的 σ 会把框定错一半 —— 表现为 PIB 塌到 0。
    ref_sigma = bench.centroid_sigma(
        bench.intensity_eta(
            bench.gs_phase_eta(
                0.0, eta=0.0, n_iters=rc.baseline_iters, seed=rc.seed, quantize=False
            )["phase_cont"],
            0.0,
            0.0,
        )
    )[2]
    rt_rows = measure_round_trip(bench, rc)
    round_trip = self_check_round_trip(rt_rows, rc.band_limit)
    logger.info(
        "传播器往返闭合: 配置({}) max relerr={:.2e}  |  带限版 max relerr={:.2e}",
        "band_limit=True" if rc.band_limit else "band_limit=False",
        round_trip["round_trip_relerr_max"],
        round_trip["round_trip_relerr_max_band_limited"],
    )
    damage = measure_band_limit_damage(rc)
    logger.info("带限破坏程度 (Δz={:g}mm): 能量损失跨度 {:.1f}% ~ {:.1f}%",
                max(rc.dz_mm), *damage["loss_span_pct"])
    shape_half_um = rc.m * cfg.far_field_pixel_size * 1e6 / 2.0
    trustworthy = measure_trustworthy_dz(halo, shape_half_um, rc.n_grid)
    logger.info("晕斑回绕: {}", trustworthy["note"])
    logger.info("  逐 Δz w/整形网格半窗: {}",
                {f"{r['dz_mm']:g}mm": round(r["halo_radius_over_halfwindow"], 2)
                 for r in trustworthy["per_dz"]})
    logger.info("主扫描: H0~H4 × Δz {} × η {} ...", rc.dz_mm, rc.eta_values)
    rows, h2_cache, anchors = run_groups(bench, rc, ref_sigma, halo)
    ceilings = summarize(rc, rows)
    ceilings_trusted = summarize(rc, rows, trustworthy["trusted_dz_mm"])

    # --- 3. 热启动 vs 随机 ---------------------------------------------------
    logger.info("热启动 vs 随机初值 ...")
    warm_rows, traj = run_warm_start(bench, rc, ref_sigma, h2_cache, halo, anchors)

    # --- 4. 产物 ------------------------------------------------------------
    dz_choice = dz_choice_table(halo)
    benefit = subtraction_benefit(rows)
    advice = hardware_advice(rows, rc)
    write_csv(out / f"imp_groups{rc.tag}.csv", rows)
    write_csv(out / f"imp_subtraction{rc.tag}.csv", benefit)
    write_csv(out / f"imp_halo{rc.tag}.csv", halo)
    write_csv(out / f"imp_warm_start{rc.tag}.csv", warm_rows)
    write_csv(out / f"imp_warm_trajectory{rc.tag}.csv", traj)
    plot_halo(halo, out / f"imp_halo_peak_vs_dz{rc.tag}.png", rc)
    plot_subtraction(rows, out / f"imp_cv_before_after{rc.tag}.png")
    plot_ceilings(ceilings_trusted, out / f"imp_group_ceilings{rc.tag}.png")
    plot_groups_vs_dz(rows, out / f"imp_groups_vs_dz{rc.tag}.png")

    h0_row = next(
        (r for r in rows if r["group"] == "H0" and abs(r["dz_mm"]) < 1e-12),
        None,
    )
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "quick" if args.quick else "full",
        # --- 顶层: 传播器可逆性 (GS 内环成立的前提) -------------------------
        "round_trip_relerr": round_trip["round_trip_relerr_max"],
        "energy_ratio": {
            f"{r['dz_mm']:g}mm": {
                "configured": r["energy_ratio_bl0"],
                "band_limited": r["energy_ratio_bl1"],
                "round_trip_relerr_configured": r["round_trip_relerr_bl0"],
                "round_trip_relerr_band_limited": r["round_trip_relerr_bl1"],
            }
            for r in rt_rows
        },
        "band_limit_damage": damage,
        "trustworthy_dz": trustworthy,
        "optics": {
            "wavelength_nm": defo.WAVELENGTH_NM,
            "focal_length_m": FOCAL_LENGTH_M,
            "aperture_m": APERTURE_M,
            "panel_pixel_um": PANEL_PIXEL_UM,
            "far_field_pixel_um": cfg.far_field_pixel_size * 1e6,
        },
        "grid": {
            "n_grid": rc.n_grid,
            "far_field_padding": rc.far_field_padding,
            "far_field_size": rc.m,
            "window_mm": rc.m * cfg.far_field_pixel_size * 1e3,
            "half_window_um": rc.m * cfg.far_field_pixel_size * 1e6 / 2,
            "halo_n_grid": rc.halo_n_grid,
            "halo_window_um": halo[0]["half_window_um"] * 2 if halo else None,
            "target_side_px": rc.target_side_px,
            "gs_iters": rc.gs_iters,
            "slm_phase_levels": SLM_PHASE_LEVELS,
        },
        "band_limit": {
            "configured": rc.band_limit,
            "authoritative_default": True,
            "conclusion": (
                "|Δz| ≳ 10 mm 时带限 ASM 的 GS 内环不可信: Voelz & Roggemann cut 在"
                "焦平面上有损且不可逆 (高频 bin 被置零, 能量在正向一步就漏), 正/反向"
                "不构成伴随对, 交替投影解的不是自洽问题。Δz>0 一律 band_limit=False; "
                "Δz=0 时开与关逐位相同, 所以基线与 data.json 核对不受影响。"
                "破坏程度**强依赖场的谱内容**, 跨度见 band_limit_damage —— "
                "先测你自己那个场的谱, 不要引用本文件的单一数字。"
                "若必须开带限, 只能加大**瞳面过采样 n_grid** (加大 padding 只加远场"
                "过采样, 窗口一动不动 —— 见 trustworthy_dz.window_formula)。"
            ),
            "numeric_band_note": (
                "CSV 里每行的 *_bl 列是同一组解在带限正演下的分数, 与无带限列之差"
                "即该指标的数值误差带; 它不是带限版 GS 解的分数 (那要整条轨迹重跑)"
            ),
        },
        "dz_mm": list(rc.dz_mm),
        "eta_values": list(rc.eta_values),
        "gs_params": {
            "gamma": GS_GAMMA,
            "weight_clip": GS_WEIGHT_CLIP,
            "beta": GS_BETA,
            "hio_fraction": GS_HIO_FRACTION,
            "note": (
                "gamma/weight_clip/beta/hio_fraction 取自 "
                "SlmGsRefineConfig 默认值; 本文件的加权更新式照抄 "
                "UniformSquareShaper._weighted_step (振幅比, 等价强度比 γ/2 次幂)"
            ),
        },
        "self_check": {
            "eta0_vs_gs_shape_max_phase_delta": gs_delta,
            "scan_path_baseline": scan_baseline,
            "round_trip": round_trip,
            "halo_monotonic": halo_check,
        },
        "metric_convention": {
            "primary_fixed_box": (
                "score/cv/pib (及 *_cont / *_bl) = **固定物理尺度框**: 边长恒为 "
                "target_side_px, 框心用质心/窗口定位一次后**冻结** (anchor_box_center), "
                "不随帧重定位。H4 的框心落在信号上, 轴上的零级晕斑在框外。"
            ),
            "adaptive_box_reference_only": (
                "score_adapt/cv_adapt/pib_adapt = 自适应框 (强度质心 + σ 缩放边长)。"
                "**只作对照, 不作结论**: 框跟着光斑一起变大会把「测量区域变大」当成"
                "「均匀性变好」—— 本调查里这个陷阱出现过三次 (side 43→104 px, "
                "σ 涨 9.06×, 于是「最优离焦」被推到 ±3.87 rad 这种非物理值)。"
            ),
            "datajson_box": (
                "score_box/cv_box/pib_box = data.json 的**居中定尺框** "
                "(make_square_target 口径), 连续相位。它不锚在信号上 (H4 的晕斑就在"
                "它中心), 所以恒可计算、永不 NaN; Δz=0/η=0 上逐位复现基线, 由 "
                "self_check.scan_path_baseline 主动 raise 兜底。"
            ),
            "conclusion_rule": (
                "所有组间比较只准用**固定框**版 (score/cv/pib)。自适应框与 data.json 框"
                "只用来暴露测量口径的影响。"
            ),
        },
        "baseline_datajson_check": {
            "measured": {
                "score": base_score,
                "PIB": base_m["PIB"],
                "CV": base_m["CV"],
            },
            "reference_datajson_gs": DATA_JSON_GS,
            "delta": base_delta,
            "tolerance": 1e-6,
            "asserted": True,
            "h0_row_from_scan": (
                None
                if h0_row is None
                else {
                    "score_box": h0_row["score_box"],
                    "cv_box": h0_row["cv_box"],
                    "pib_box": h0_row["pib_box"],
                    "score": h0_row["score"],
                    "cv": h0_row["cv"],
                    "pib": h0_row["pib"],
                    "score_cont": h0_row["score_cont"],
                    "cv_cont": h0_row["cv_cont"],
                    "pib_cont": h0_row["pib_cont"],
                    "signal_center_px": h0_row["signal_center_px"],
                    "note": (
                        "扫描行 (量化相位, 与 --quick 无关的 data.json 口径只在 "
                        "self_check.scan_path_baseline 里核对)。*_box 列是连续相位 + "
                        "居中定尺框, 与基线同口径; score/cv/pib 是量化相位 + 锚在信号上"
                        "的定尺框 (H0 在 Δz=0 上二者只差量化)。"
                    ),
                }
            ),
        },
        "group_ceilings": ceilings,
        "group_ceilings_trusted_dz_only": ceilings_trusted,
        "subtraction_benefit": benefit,
        "hardware_advice": advice,
        "band_limit_metric_band": band_limit_band(rows),
        "dz_choice": dz_choice,
        "halo_rows": halo,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "outputs": {
            "groups_csv": f"imp_groups{rc.tag}.csv",
            "subtraction_csv": f"imp_subtraction{rc.tag}.csv",
            "halo_csv": f"imp_halo{rc.tag}.csv",
            "warm_start_csv": f"imp_warm_start{rc.tag}.csv",
            "warm_trajectory_csv": f"imp_warm_trajectory{rc.tag}.csv",
            "halo_png": f"imp_halo_peak_vs_dz{rc.tag}.png",
            "cv_png": f"imp_cv_before_after{rc.tag}.png",
            "ceilings_png": f"imp_group_ceilings{rc.tag}.png",
            "groups_png": f"imp_groups_vs_dz{rc.tag}.png",
        },
    }
    (out / f"summary_gs_improved{rc.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(
        {k: summary[k] for k in
         ("mode", "round_trip_relerr", "energy_ratio", "band_limit_damage",
          "trustworthy_dz", "metric_convention", "self_check",
          "baseline_datajson_check", "group_ceilings",
          "group_ceilings_trusted_dz_only", "subtraction_benefit",
          "hardware_advice", "band_limit_metric_band", "dz_choice")},
        indent=2, ensure_ascii=False,
    ))
    print(f"\n产物目录: {out}")
    print(f"耗时: {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()