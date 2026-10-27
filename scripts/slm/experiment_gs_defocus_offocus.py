"""离焦双旋钮的纯仿真实验: 瞳面离焦 (Zernike (2,0)) 与 CCD 焦后位移 Δz。

回答两个问题
------------

**E1 —— 补充离焦能不能救回匀化指标?**
在 CCD 焦后位移 Δz 固定时, 扫**瞳面**离焦幅度 (Zernike (2,0), 弧度), 看
CV / PIB / score 与 90% 环围直径怎么变。含 Δz = 0 与离焦 = 0 两个对照。

**E2 —— 叠加不同匀化方法的效果, A/B 两模式**

* 模式 A (鲁棒性): 只在 Δz = 0 下优化匀化, 再拿到各个 Δz 上评测 → 退化曲线;
* 模式 B (可达): 在**每个** Δz 下各自优化匀化, 再在该 Δz 评测 → 该工作点上限。

A 与 B 在同一 Δz 上的差, 连同 "GS+Zernike+瞳面离焦" 是否优于 "GS+Zernike",
就是「离焦能不能被匀化补偿」的答案。

物理建模 (最容易搞错的一点)
--------------------------

把 CCD 移到焦平面后面**不是** ``ASM(pupil_field, f + Δz)``。透镜已经完成了傅里叶
变换, 焦后移动是**对已变换的场再自由传播 Δz**::

    pupil_field = gauss * exp(1j * (base_phase + zernike_phase))          # 原始弧度
    F           = fftshift(fft2(ifftshift(pad_to_far_field(pupil))))       # 透镜 FT, 复数
    U_ccd       = fftshift(ASM(ifftshift(F), dx_far_m, dz, wavelength_m))  # 焦后 Δz
    intensity   = |U_ccd|**2, 归一化到 sum = 1

两个离焦是**正交**的旋钮:

* **瞳面离焦** (Zernike (2,0), 弧度) ⇒ 傅里叶缩放定理 ⇒ 缩放/稀释远场图样;
* **CCD 焦后 Δz** ⇒ 对焦面图样做菲涅耳模糊。

复用仓库已有资产 (不自写)
--------------------------

* 传播: :func:`~ao_shaping.algorithm.signal_processing.angular_spectrum.angular_spectrum_propagate`
  (numpy, 未 fftshift 的权威传播器)。**不自己写传播。**
* GS: 正/反向算子由上面的权威传播器给出; 目标、瞳面幅度、口径全部取自
  :mod:`~ao_shaping.algorithm.signal_processing.square_shaping`。本文件的
  :meth:`OffFocusBench.gs_phase` 与 ``square_shaping.gs_shape`` 的循环逐行同构,
  唯一差别是把 ``fftshift(fft2(ifftshift(·)))`` 换成可注入的正/反向算子 ——
  ``gs_shape`` 把传播算子写死了, 无法注入 Δz。**Δz = 0 时 ASM 恒等**
  (``exp(i*kz*0) = 1`` 且带限不丢任何 bin), 所以两条路径逐位相同;
  脚本启动时跑一次自检并打印 ``max|Δphase|``, 不达标直接抛错。
* Zernike 数学: :class:`~ao_shaping.utils.wavefront.zernike_calc.ZernikeGenerator`
  + :func:`~ao_shaping.utils.wavefront.zernike_calc.zernike_modes`。不自写
  Noll↔(n,m) 查表, 也不自写多项式求值。
* 梯度优化器: ``ao_shaping.algorithm.gradient.adam.Base.create("adam")``。
* 光学常量: 从 :mod:`ao_shaping.optimizer.wfless.slm_gs_refine` 追到源头
  (面板像元 8.0 µm、CCD 像元 2.2 µm、光束半径 450 SLM px、焦距由
  :func:`~ao_shaping.utils.wavefront.matrix_utils.focal_length_from_camera_pixel`
  从实测 2f 焦距标度导出、波长 ``DEFAULT_WAVELENGTH_NM``)。

标定 (与 ``report/iterative_zernike_shaping/data.json`` 交叉核对)
--------------------------------------------------------------

目标框**跟着图样走**: 中心 = 强度质心, 边长 = ``target_side_px`` 乘以该帧实测
二阶矩半径 σ 与参考 σ 的比值。参考 σ 在 (Δz = 0, 离焦 = 0) 的 GS 基线上量一次,
因此基线上自动框边长恰为 ``target_side_px``。data.json 的框是**定尺框、框心取画框
中心 (m//2, m//2)**: 全网格基线上图样质心在 (255.5, 255.5), 与 m//2 = 256 差半个
像素, 整数化后两框差 1 px → 基线 score 两种口径差 ~0.5% (实测 score −0.0045 /
CV +0.010 / PIB −0.0042, 换光学常量本身不影响任何像素口径, 逐位验证)。
summary 因此同时记录 data.json 口径的仓库 canonical 结果
(``baseline_gs_dz0_datajson_box``, 经 ``compute_bench_metrics`` +
``make_square_target``), 以及定边长 (质心居中) 框的 ``*_fixed`` 列, 便于交叉核对。

score 口径与既有报告一致: ``0.5 * PIB + 0.5 * (1/(1 + CV))``, CV / PIB 在目标框内算。

运行
----

.. code-block:: bash

    python scripts/slm/experiment_gs_defocus_offocus.py           # 完整网格
    python scripts/slm/experiment_gs_defocus_offocus.py --quick   # 小网格冒烟

产物写到 ``report/shape_trials/gs_defocus_sim/artifacts/``: 两个 CSV (原始数据)、summary.json、
一张 E2 对比图 (CV vs Δz, 每方法一条曲线, 模式 A/B 分开)、一张 E1 热力图。

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

from ao_shaping.algorithm.gradient.adam import Base as GradientBase
from ao_shaping.algorithm.signal_processing.angular_spectrum import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_torch,
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
from ao_shaping.utils.wavefront.matrix_utils import focal_length_from_camera_pixel
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator, zernike_modes

# ---------------------------------------------------------------------------
# 光学常量 —— 全部从 slm_gs_refine.py 追到源头, 不在这里另立一份
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

#: 与 report/iterative_zernike_shaping/data.json 同口径的仿真网格
N_GRID = 64
FAR_FIELD_PADDING = 8
TARGET_SIDE_PX = 43
SEED = 0

#: 瞳面离焦模式 (Noll 4 = (2,0) 离焦)。所有离焦都用这一个模式, 保证是同一个旋钮。
DEFO_MODE = (2, 0)
N_ZERNIKE = 4


def defocus_scale_m(wavelength_m: float, focal_m: float, aperture_m: float) -> float:
    """焦后位移的参考尺度 Δz_blur (m): 使离焦弥散等于焦面光斑尺度。

    CCD 后移 Δz 时, 瞳面边缘 (h = D/2) 的光线在 CCD 上偏移 ``h·Δz/f``; 令其等于
    Airy 半径 ``1.22·λf/D`` 得 ``Δz = 2.44·λf²/D²``。扫描点写成 ``u · Δz_blur``
    即与具体波长/焦距/口径解耦 —— 换实验系统时 u 这一串数字不用重挑。
    """
    return 2.44 * wavelength_m * focal_m**2 / aperture_m**2


DZ_BLUR_M = defocus_scale_m(WAVELENGTH_M, FOCAL_LENGTH_M, APERTURE_M)

#: Δz 扫描点 (单位 Δz_blur)。0 = 焦面, 作为对照。
DZ_UNITS_FULL = (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
#: 瞳面离焦扫描点 (Zernike (2,0) 系数, 弧度), 对称区间含 0。
DEFOCUS_RAD_FULL = tuple(float(v) for v in np.linspace(-2.0, 2.0, 11))
DZ_UNITS_QUICK = (-1.0, -0.5, 0.0, 0.5, 1.0)
DEFOCUS_RAD_QUICK = (-2.0, -1.0, 0.0, 1.0, 2.0)

METHODS: tuple[str, ...] = ("gs", "zernike", "gs_zernike", "gs_zernike_defocus")
METHOD_LABELS: dict[str, str] = {
    "gs": "GS",
    "zernike": "Zernike",
    "gs_zernike": "GS+Zernike",
    "gs_zernike_defocus": "GS+Zernike+瞳面离焦",
}
METHOD_COLORS: dict[str, str] = {
    "gs": "#2196F3",
    "zernike": "#9E9E9E",
    "gs_zernike": "#4CAF50",
    "gs_zernike_defocus": "#FF5722",
}

#: torch 只在优化循环里惰性用到 (见 :meth:`OffFocusBench._torch`)
_torch_module: Any = None


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
    adam_iters: int = 200
    adam_lr: float = 0.05
    dz_units: tuple[float, ...] = DZ_UNITS_FULL
    defocus_rad: tuple[float, ...] = DEFOCUS_RAD_FULL
    tag: str = ""

    @property
    def m(self) -> int:
        """零填充远场网格边长。"""
        return self.n_grid * self.far_field_padding

    def bench(self) -> SquareShapingConfig:
        """带实验系统光学常量的整形配置。

        ⚠️ ``make_gaussian_pupil`` 只依赖 ``aperture_size / n_grid`` 与
        ``aperture_size / 3.5`` 的比值, 所以**远场图样 (以远场像素计) 与
        wavelength / focal_length / aperture_size 无关** —— 换成本实验系统的
        实测常量后, 远场图样与 ``ShapingBenchConfig`` 默认值 (532 nm / 125 mm /
        12 mm) 逐位相同, 只有「一个远场像素等于多少米」变了。这正是能和
        ``data.json`` 直接对比的原因。
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
    adam_iters=60,
    dz_units=DZ_UNITS_QUICK,
    defocus_rad=DEFOCUS_RAD_QUICK,
    tag="_quick",
)
FULL = RunConfig()


# ---------------------------------------------------------------------------
# 物理预测模型 + 优化 + 度量
# ---------------------------------------------------------------------------
class OffFocusBench:
    """可注入焦后位移 Δz 的 2f 整形物理预测模型 (纯 numpy)。

    传播一律走 ``angular_spectrum_propagate``; 本类只负责**组装**
    「瞳面场 → 透镜 FT → 焦后 Δz → CCD 强度」这条链, 以及 GS 的正/反向算子。
    """

    def __init__(self, cfg: SquareShapingConfig, *, n_zernike: int = N_ZERNIKE) -> None:
        self.cfg = cfg
        self.n = cfg.n_grid
        self.m = cfg.far_field_size
        self.dx = cfg.far_field_pixel_size  # 远场像元间距 (m)
        self.lam = cfg.wavelength
        self.pad = (self.m - self.n) // 2
        self.amp = cfg.make_gaussian_pupil().astype(np.float64)
        self.target = make_square_target(cfg)
        #: 优化目标函数用的**定尺**支撑域 (居中, 边长 target_side_px)。
        #: 与 ``IterativeZernikeShapingOptimizer._score`` 同约定: 不跟随 argmax
        #: 滚动 —— 类散斑场里 ~1e-3 的变化就足以让 argmax 跳到另一颗晶粒上,
        #: 滚动框让目标函数不连续。
        self.support = self.target > 0

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

        # torch 侧的常量 (只读, 提前建好避免每次进优化循环重建)
        try:
            import torch as _t
        except ImportError:  # pragma: no cover - 仅优化循环需要 torch
            self.amp_t = self.support_t = None  # type: ignore[assignment]
        else:
            self.amp_t = _t.as_tensor(self.amp, dtype=_t.float64)
            self.support_t = _t.as_tensor(self.support)

    # -- 正/反向传播算子 ----------------------------------------------------
    def _asm_raw(self, field: np.ndarray, dz: float) -> np.ndarray:
        """权威传播器走一步。输入**居中**排布, 输出**未 fftshift** (DC 在 index 0)。"""
        return angular_spectrum_propagate(
            np.fft.ifftshift(field), self.dx, dz, self.lam, band_limit=True
        )

    def _asm(self, field: np.ndarray, dz: float) -> np.ndarray:
        """焦平面 → CCD 复场 (输入输出都是**居中**排布)。"""
        return np.fft.fftshift(self._asm_raw(field, dz))

    def _asm_inverse(self, field: np.ndarray, dz: float) -> np.ndarray:
        """``_asm`` 的伴随逆 (CCD → 焦平面, 居中排布)。

        ASM 的传递函数在未 fftshift 的频域里是 ``exp(i·kz·dz)`` 且带限掩模关于 z
        偶对称, 故 ``H(-dz) = conj(H(dz))``, 于是把 z 取负就是它的转置共轭。
        Δz = 0 时退化为恒等 —— 这正是自检里 GS 镜像能与 ``gs_shape`` 逐位相同的原因。
        """
        return np.fft.fftshift(np.fft.ifft2(self._asm_raw(field, -dz)))

    # -- 物理预测模型 ----------------------------------------------------------
    def focal(self, phase: np.ndarray) -> np.ndarray:
        """瞳面相位 → 焦平面复场 (居中排布, Δz = 0)。"""
        padded = np.zeros((self.m, self.m), dtype=np.complex128)
        padded[self.pad : self.pad + self.n, self.pad : self.pad + self.n] = (
            self.amp * np.exp(1j * np.asarray(phase, dtype=np.float64))
        )
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))

    def ccd_field(self, phase: np.ndarray, dz: float) -> np.ndarray:
        """瞳面相位 → CCD 复场 (居中排布): 透镜 FT 之后再自由传播 dz。"""
        return self._asm(self.focal(phase), dz)

    def intensity(self, phase: np.ndarray, dz: float) -> np.ndarray:
        """CCD 强度, 归一化到 sum = 1。"""
        inten = np.abs(self.ccd_field(phase, dz)) ** 2
        return inten / (inten.sum() + 1e-12)

    # -- GS ----------------------------------------------------------------
    def gs_phase(self, dz: float, *, n_iters: int, seed: int) -> np.ndarray:
        """在给定焦后位移下跑 GS, 返回瞳面相位 (原始弧度)。

        与 ``square_shaping.gs_shape`` 的循环逐行同构, 只把两条 FFT 换成
        ``_asm`` / ``_asm_inverse``; 初始化、目标幅度、瞳面幅度、口径全部取自
        ``square_shaping`` 的同一份配置。Δz = 0 时与 ``gs_shape`` 逐位相同。
        """
        rng = np.random.default_rng(seed)
        amp_target = np.sqrt(self.target)
        amp_slm = np.zeros((self.m, self.m), dtype=np.float64)
        amp_slm[self.pad : self.pad + self.n, self.pad : self.pad + self.n] = self.amp
        field = amp_slm * np.exp(1j * rng.normal(0, 0.1, size=amp_slm.shape))
        for _ in range(n_iters):
            # 正向: 瞳面 → 透镜 FT → 焦后 Δz → CCD。Δz = 0 时第二步退化为恒等,
            # 于是整条链恰是 square_shaping.gs_shape 的 fftshift(fft2(ifftshift(·)))。
            ff = self._asm(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field))), dz)
            ff = amp_target * np.exp(1j * np.angle(ff))
            # 反向: CCD → 焦后 −Δz → 瞳面。Δz = 0 时即 fftshift(ifft2(ifftshift(·)))。
            field = self._asm_inverse(ff, dz)
            field = amp_slm * np.exp(1j * np.angle(field))
        s = self.pad
        return np.angle(field)[s : s + self.n, s : s + self.n]

    # -- 目标函数与梯度 ---------------------------------------------------
    def _torch(self) -> Any:
        """惰性导入 torch (只有优化循环需要它; 报告用的正演始终走 numpy)。"""
        global _torch_module
        if _torch_module is None:
            try:
                import torch as _t
            except ImportError as exc:  # pragma: no cover - 依赖缺失时的显式指引
                raise RuntimeError(
                    "Zernike 系数的梯度优化需要 torch (仓库的 'ml' 依赖组): "
                    "uv sync --group ml"
                ) from exc
            _torch_module = _t
        return _torch_module

    def torch_intensity(
        self,
        coeffs: Any,
        basis_t: Any,
        base_t: Any,
        dz: float,
    ) -> Any:
        """与 :meth:`intensity` 同构的 torch 正演 (可微)。

        传播走仓库的 torch 版 :func:`angular_spectrum_propagate_torch` —— 它与
        numpy 版是同一套公式的孪生实现 (随机复场上两者 max|Δ| = 8.6e-17)。
        **不复现第三份传播数学**。

        ⚠️ 梯度用 torch autograd, 不手写伴随。伴随链里有三处极易漏掉的归一化
        (未归一化 ``fft2`` 的转置共轭是 ``N·ifft2``; score 吃的是归一化强度;
        ``∂PIB/∂Î = [j∈S] − PIB`` 而非 ``[j∈S]``), 漏掉任何一处都**不报错**,
        只会让 Adam 停在一个看似合理的分数上。autograd 没有这个风险,
        :func:`self_check_gradient` 再兜一层底。
        """
        t = self._torch()
        phase = base_t + t.einsum("k,kij->ij", coeffs, basis_t)
        padded = t.zeros((self.m, self.m), dtype=t.complex128)
        padded[self.pad : self.pad + self.n, self.pad : self.pad + self.n] = (
            self.amp_t * t.exp(1j * phase.to(t.complex128))
        )
        focal = t.fft.fftshift(t.fft.fft2(t.fft.ifftshift(padded)))
        u = t.fft.fftshift(
            angular_spectrum_propagate_torch(
                t.fft.ifftshift(focal), self.dx, dz, self.lam, band_limit=True
            )
        )
        inten = u.real**2 + u.imag**2
        return inten / inten.sum()

    def optimize_coeffs(
        self,
        basis: np.ndarray,
        base_phase: np.ndarray,
        dz: float,
        *,
        n_iters: int,
        lr: float,
    ) -> tuple[np.ndarray, float]:
        """在定尺支撑域上最大化 score, 返回 (最优系数, 最优 score)。

        ``phase = base_phase + Σ_k coeffs[k] · basis[k]``, 系数从 0 起步。更新器用
        仓库的 ``Adam`` (``c ← c + Adam.update(dJ/dc)``, 即**升** score)。
        """
        t = self._torch()
        basis_t = t.as_tensor(basis, dtype=t.float64)
        base_t = t.as_tensor(base_phase, dtype=t.float64)
        c = t.zeros(basis.shape[0], dtype=t.float64, requires_grad=True)
        opt = GradientBase.create("adam", dim=basis.shape[0], lr=lr)
        best_c = c.detach().clone()
        best_score = -np.inf
        for _ in range(n_iters):
            vals = self.torch_intensity(c, basis_t, base_t, dz)[self.support_t]
            cv = vals.std(unbiased=False) / vals.mean()
            score = 0.5 * vals.sum() + 0.5 / (1.0 + cv)
            (grad,) = t.autograd.grad(score, c)
            value = float(score.detach())
            with t.no_grad():
                c += t.from_numpy(opt.update(grad.detach().numpy().astype(np.float64)))
            if value > best_score:
                best_c = c.detach().clone()
                best_score = value
        return best_c.numpy().copy(), best_score

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
        """目标框内的 ``(PIB, CV)``。

        框边长 ``side`` 被夹在 ``[5, m-2]``; 中心取最近整数像素。框内无光时
        CV 返回 ``inf`` (与 ``square_shaping.uniformity_cv`` 一致)。
        """
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

    def evaluate(self, inten: np.ndarray, *, ref_sigma: float, target_side_px: int) -> dict:
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
# 自检: Δz = 0 时本文件的 GS 必须与仓库的 gs_shape 逐位相同
# ---------------------------------------------------------------------------
def self_check_gs(rc: RunConfig) -> float:
    """GS 镜像 vs 仓库 ``gs_shape`` (Δz = 0)。返回 ``max|Δphase|``。"""
    cfg = rc.bench()
    bench = OffFocusBench(cfg, n_zernike=rc.n_zernike)
    ref = gs_shape(cfg, n_iters=rc.gs_iters, seed=rc.seed).phase
    mine = bench.gs_phase(0.0, n_iters=rc.gs_iters, seed=rc.seed)
    delta = float(np.abs(ref - mine).max())
    if delta > 1e-9:
        raise AssertionError(
            f"GS 镜像在 Δz=0 与 gs_shape 不一致 (max|Δphase|={delta:.3e}); "
            "正/反向算子的约定被改坏了, 结论不可用"
        )
    return delta


def self_check_gradient(bench: OffFocusBench, dz: float, basis: np.ndarray) -> float:
    """torch autograd 的梯度 vs numpy 正演上的中心差分。返回最大相对误差。"""
    torch = bench._torch()
    rng = np.random.default_rng(7)
    c = rng.normal(0, 0.2, size=basis.shape[0])
    base = np.zeros((bench.n, bench.n))

    def score_at(coef: np.ndarray) -> float:
        phase = base + np.tensordot(coef, basis, axes=(0, 0))
        inten = bench.intensity(phase, dz)
        vals = inten[bench.support]
        return composite_from_pib_cv(
            float(vals.sum()), float(vals.std() / vals.mean())
        )

    basis_t = torch.as_tensor(basis, dtype=torch.float64)
    base_t = torch.as_tensor(base, dtype=torch.float64)
    ct = torch.tensor(c, dtype=torch.float64, requires_grad=True)
    vals = bench.torch_intensity(ct, basis_t, base_t, dz)[bench.support_t]
    score = 0.5 * vals.sum() + 0.5 / (1.0 + vals.std(unbiased=False) / vals.mean())
    (analytic,) = torch.autograd.grad(score, ct)
    analytic = analytic.detach().numpy().copy()

    numeric = np.empty_like(c)
    for i in range(c.size):
        step = np.zeros_like(c)
        step[i] = 1e-6
        numeric[i] = (score_at(c + step) - score_at(c - step)) / 2e-6
    err = float(np.abs(analytic - numeric).max() / max(np.abs(numeric).max(), 1e-30))
    if err > 1e-4:
        raise AssertionError(f"梯度自检失败: 相对误差 {err:.3e} > 1e-4")
    return err


def self_check_torch_vs_numpy(bench: OffFocusBench, dz: float) -> float:
    """torch 正演 vs numpy 正演 (报告值用的是 numpy, 优化用的是 torch)。"""
    torch = bench._torch()
    rng = np.random.default_rng(11)
    phase = rng.normal(0, 0.3, size=(bench.n, bench.n))
    ref = bench.intensity(phase, dz)
    basis_t = torch.zeros((0, bench.n, bench.n), dtype=torch.float64)
    base_t = torch.as_tensor(phase, dtype=torch.float64)
    got = (
        bench.torch_intensity(
            torch.zeros(0, dtype=torch.float64), basis_t, base_t, dz
        )
        .detach()
        .numpy()
    )
    err = float(np.abs(ref - got).max() / max(float(ref.max()), 1e-30))
    if err > 1e-10:
        raise AssertionError(f"torch/numpy 正演不一致: 相对误差 {err:.3e} > 1e-10")
    return err


# ---------------------------------------------------------------------------
# 方法
# ---------------------------------------------------------------------------
def method_basis(bench: OffFocusBench, method: str) -> tuple[np.ndarray, bool]:
    """该方法的 Zernike 基与是否需要 GS 热启动。

    ``zernike`` / ``gs_zernike`` 的基**刻意不含离焦** (2,0), 这样
    ``gs_zernike_defocus`` 多出来的那一个自由度就是**离焦本身**, 两个方法之间
    只差这一个旋钮, 差值才有意义。
    """
    if method == "gs":
        return bench.zernike_basis[:0], True
    if method == "zernike":
        return bench.zernike_basis, False
    if method == "gs_zernike":
        return bench.zernike_basis, True
    if method == "gs_zernike_defocus":
        return np.concatenate([bench.zernike_basis, bench.defocus_basis[None]]), True
    raise ValueError(f"unknown method {method!r}")


def run_method(
    bench: OffFocusBench,
    method: str,
    dz_opt: float,
    *,
    gs_cache: dict[float, np.ndarray],
    rc: RunConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """在给定 Δz 上跑一个方法, 返回 (瞳面相位, info)。"""
    basis, needs_gs = method_basis(bench, method)
    base = np.zeros((bench.n, bench.n))
    if needs_gs:
        if dz_opt not in gs_cache:
            gs_cache[dz_opt] = bench.gs_phase(
                dz_opt, n_iters=rc.gs_iters, seed=rc.seed
            )
        base = gs_cache[dz_opt]
    if basis.shape[0] == 0:  # 纯 GS, 无系数
        return base, {"defocus_rad": 0.0, "opt_score": float("nan"), "n_coeffs": 0}
    coeffs, opt_score = bench.optimize_coeffs(
        basis,
        base,
        dz_opt,
        n_iters=rc.adam_iters,
        lr=rc.adam_lr,
    )
    phase = base + np.tensordot(coeffs, basis, axes=(0, 0))
    defocus_rad = float(coeffs[-1]) if method == "gs_zernike_defocus" else 0.0
    return phase, {
        "defocus_rad": defocus_rad,
        "opt_score": float(opt_score),
        "n_coeffs": int(basis.shape[0]),
    }


# ---------------------------------------------------------------------------
# 实验
# ---------------------------------------------------------------------------
def run_e1(bench: OffFocusBench, rc: RunConfig, ref_sigma: float, gs0: np.ndarray) -> list[dict]:
    """E1: 瞳面离焦幅度 × 焦后 Δz 的二维扫描 (基线 GS 相位)。"""
    rows: list[dict] = []
    for u in rc.dz_units:
        dz = u * DZ_BLUR_M
        for a in rc.defocus_rad:
            phase = gs0 + a * bench.defocus_basis
            m = bench.evaluate(bench.intensity(phase, dz), ref_sigma=ref_sigma,
                               target_side_px=rc.target_side_px)
            rows.append({"dz_units": u, "dz_mm": dz * 1e3, "defocus_rad": a, **m})
    return rows


def run_e2(bench: OffFocusBench, rc: RunConfig, ref_sigma: float) -> list[dict]:
    """E2: 四种方法 × Δz, 模式 A (Δz=0 优化后迁移) 与模式 B (逐点重优化)。"""
    gs_cache: dict[float, np.ndarray] = {}
    phases: dict[tuple[str, float], np.ndarray] = {}
    infos: dict[tuple[str, float], dict] = {}
    rows: list[dict] = []

    plans = [("A", method, 0.0) for method in METHODS]
    plans += [("B", method, u) for u in rc.dz_units for method in METHODS]
    for mode, method, u in plans:
        dz = u * DZ_BLUR_M
        key = (method, dz)
        if key not in phases:
            phases[key], infos[key] = run_method(bench, method, dz, gs_cache=gs_cache, rc=rc)
        opt_dz = 0.0 if mode == "A" else dz
        for u_eval in rc.dz_units:
            dz_eval = u_eval * DZ_BLUR_M
            m = bench.evaluate(
                bench.intensity(phases[key], dz_eval),
                ref_sigma=ref_sigma,
                target_side_px=rc.target_side_px,
            )
            rows.append(
                {
                    "method": method,
                    "mode": mode,
                    "opt_dz_units": opt_dz / DZ_BLUR_M,
                    "eval_dz_units": u_eval,
                    "dz_mm": dz_eval * 1e3,
                    "defocus_rad_opt": infos[key]["defocus_rad"],
                    "n_coeffs": infos[key]["n_coeffs"],
                    **m,
                }
            )
        logger.info(
            "  [{mode}] {label:<22} opt Δz={u:+.2f} Δz_blur  优化点 score={s:.4f}",
            mode=mode,
            label=METHOD_LABELS[method],
            u=opt_dz / DZ_BLUR_M,
            s=infos[key]["opt_score"],
        )
    return rows


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


def plot_e2(rows: list[dict], out: Path, rc: RunConfig) -> None:
    """CV vs Δz: 每方法一条曲线, 模式 A / B 分开两栏。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, mode, title in zip(axes, ("A", "B"), ("模式 A (Δz=0 优化后迁移)", "模式 B (逐点重优化)")):
        for method in METHODS:
            # 模式 B 的行是 (优化点 × 评估点) 矩阵; 曲线只画对角 (逐点重优化),
            # 否则 5 条不同优化点的曲线会叠成一张乱线。
            sel = [
                r
                for r in rows
                if r["mode"] == mode
                and r["method"] == method
                and (mode == "A" or abs(r["opt_dz_units"] - r["eval_dz_units"]) < 1e-12)
            ]
            sel.sort(key=lambda r: r["eval_dz_units"])
            ax.plot(
                [r["eval_dz_units"] for r in sel],
                [r["cv"] for r in sel],
                "o-",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=2,
            )
        ax.set_yscale("log")
        ax.set_xlabel("CCD 焦后位移 Δz / Δz_blur")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("目标框内 CV (自动定标框)")
    axes[0].legend(fontsize=9)
    fig.suptitle(
        f"E2 匀化方法对比 (Δz_blur = {DZ_BLUR_M * 1e3:.4f} mm, "
        f"λ={WAVELENGTH_NM:.0f}nm f={FOCAL_LENGTH_M * 1e3:.1f}mm D={APERTURE_M * 1e3:.2f}mm)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_e1(rows: list[dict], out: Path, rc: RunConfig) -> None:
    """E1 热力图: 瞳面离焦幅度 × Δz 的 CV, 外加每行的最优离焦。"""
    grid = np.full((len(rc.dz_units), len(rc.defocus_rad)), np.nan)
    for r in rows:
        i = rc.dz_units.index(r["dz_units"])
        j = rc.defocus_rad.index(r["defocus_rad"])
        grid[i, j] = r["cv"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    im = ax.imshow(
        np.log10(np.clip(grid, 1e-6, None)),
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=[min(rc.defocus_rad), max(rc.defocus_rad),
                min(rc.dz_units), max(rc.dz_units)],
    )
    ax.set_xlabel("瞳面离焦系数 (Zernike (2,0), 弧度)")
    ax.set_ylabel("CCD 焦后位移 Δz / Δz_blur")
    ax.set_title("E1 CV 热力图 (log10)")
    fig.colorbar(im, ax=ax, label="log10(CV)")

    best_j = np.nanargmin(grid, axis=1)
    ax2.plot(rc.dz_units, np.array(rc.defocus_rad)[best_j], "o-", color="#FF5722", linewidth=2)
    ax2.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
    ax2.set_xlabel("CCD 焦后位移 Δz / Δz_blur")
    ax2.set_ylabel("该 Δz 下 CV 最小的离焦系数 (弧度)")
    ax2.set_title("E1 逐 Δz 的最优瞳面离焦")
    ax2.grid(alpha=0.3)
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
    p.add_argument("--out", default=None, help="产物目录 (默认 report/shape_trials/gs_defocus_sim/artifacts)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    rc = QUICK if args.quick else FULL
    if args.tag is not None:
        rc = replace(rc, tag=args.tag)
    out = Path(args.out) if args.out else ROOT / "report" / "shape_trials" / "gs_defocus_sim" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cfg = rc.bench()
    logger.info(
        "光学常量: λ={:.0f}nm  f={:.2f}mm  D={:.3f}mm  面板像元={:.1f}um  CCD像元={:.2f}um",
        WAVELENGTH_NM, FOCAL_LENGTH_M * 1e3, APERTURE_M * 1e3,
        PANEL_PIXEL_UM, CAMERA_PIXEL_UM,
    )
    logger.info(
        "网格: n_grid={}  padding={}  far_field_size={}  远场像元={:.4f}um  "
        "target_side={}px  Δz_blur={:.4f}mm",
        rc.n_grid, rc.far_field_padding, rc.m, cfg.far_field_pixel_size * 1e6,
        rc.target_side_px, DZ_BLUR_M * 1e3,
    )

    logger.info("自检 1/3: GS 镜像 vs gs_shape (Δz=0) ...")
    gs_delta = self_check_gs(rc)
    bench = OffFocusBench(cfg, n_zernike=rc.n_zernike)
    logger.info("自检 2/3: torch 正演 vs numpy 正演 ...")
    twin_err = self_check_torch_vs_numpy(bench, 1.0 * DZ_BLUR_M)
    logger.info("自检 3/3: autograd 梯度 vs 中心差分 ...")
    grad_err = self_check_gradient(bench, 1.0 * DZ_BLUR_M, bench.zernike_basis)
    logger.info(
        "自检通过: max|Δphase|={:.3e}  torch/numpy={:.3e}  梯度相对误差={:.3e}",
        gs_delta, twin_err, grad_err,
    )

    logger.info("跑 GS 基线 (Δz=0) ...")
    gs0 = bench.gs_phase(0.0, n_iters=rc.gs_iters, seed=rc.seed)
    I0 = bench.intensity(gs0, 0.0)
    base_metrics = bench.evaluate(I0, ref_sigma=1.0, target_side_px=rc.target_side_px)
    ref_sigma = base_metrics["sigma_px"]
    base_metrics = bench.evaluate(I0, ref_sigma=ref_sigma, target_side_px=rc.target_side_px)
    # data.json 口径: 定尺框, 框心 = 画框中心 (m//2, m//2) (仓库 canonical 函数)。
    # 上面的自动框基线框心 = 强度质心 (512 帧上为 255.5), 与 m//2 差半个像素 ——
    # 边长一致、位置差 1 px, 产生 ~0.5% 的分数差 (见模块 docstring 标定节)。
    datajson_box_metrics = compute_bench_metrics(I0, make_square_target(cfg))
    logger.info(
        "baseline (Δz=0, 无离焦, GS): score={:.4f} PIB={:.4f} CV={:.4f} σ={:.3f}px",
        base_metrics["score"], base_metrics["pib"], base_metrics["cv"], ref_sigma,
    )

    logger.info("E1: 离焦幅度 × Δz 扫描 ...")
    e1 = run_e1(bench, rc, ref_sigma, gs0)
    logger.info("E2: 四方法 × Δz, 模式 A / B ...")
    e2 = run_e2(bench, rc, ref_sigma)

    write_csv(out / f"e1_defocus_sweep{rc.tag}.csv", e1)
    write_csv(out / f"e2_methods{rc.tag}.csv", e2)
    plot_e1(e1, out / f"e1_cv_heatmap{rc.tag}.png", rc)
    plot_e2(e2, out / f"e2_cv_vs_dz{rc.tag}.png", rc)

    # --- 摘要 ------------------------------------------------------------
    def pick(mode: str, method: str, u: float, opt_u: float) -> dict:
        """按 (mode, method, 评估点, 优化点) 取唯一一行。

        模式 B 的行是完整的 (优化点 × 评估点) 迁移矩阵 —— 只按评估点取行会拿到
        第一个优化点 (最小 Δz) 的行, 把「Δz=−1 优化的相位在 Δz=0 的分数」误当成
        「Δz=0 的逐点重优化」。故必须同时匹配 ``opt_dz_units``:
        A−B 差距与自由度增益都只取对角 (优化点 == 评估点)。
        """
        for r in e2:
            if r["mode"] != mode or r["method"] != method:
                continue
            if abs(r["eval_dz_units"] - u) > 1e-12:
                continue
            if abs(r["opt_dz_units"] - opt_u) > 1e-12:
                continue
            return r
        raise KeyError((mode, method, u, opt_u))

    gaps = {
        f"{u:+.2f}": round(
            pick("B", "gs_zernike", u, u)["score"] - pick("A", "gs_zernike", u, 0.0)["score"], 6
        )
        for u in rc.dz_units
    }
    defocus_gain = {
        f"{u:+.2f}": round(
            pick("B", "gs_zernike_defocus", u, u)["score"]
            - pick("B", "gs_zernike", u, u)["score"],
            6,
        )
        for u in rc.dz_units
    }
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
        "target_side_px": rc.target_side_px,
        "objective": "0.5*PIB + 0.5*(1/(1+CV)), CV/PIB in the centroid-tracked auto-scaled box",
        "dz_blur_mm": DZ_BLUR_M * 1e3,
        "dz_units": list(rc.dz_units),
        "defocus_rad": list(rc.defocus_rad),
        "self_check": {
            "gs_mirror_max_phase_delta": gs_delta,
            "torch_vs_numpy_rel_err": twin_err,
            "autograd_vs_central_difference_rel_err": grad_err,
        },
        "baseline_gs_dz0": {
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
        "baseline_gs_dz0_datajson_box": {
            "score": composite_score(datajson_box_metrics),
            "PIB": datajson_box_metrics["PIB"],
            "CV": datajson_box_metrics["CV"],
            "note": (
                "data.json 口径 (定尺框, 框心 = 画框中心 m//2); 上面的主基线是质心跟踪自动框, "
                "框心差半个像素 → 分数差 ~0.5%"
            ),
        },
        "reference_baseline_data_json": {
            "score": 0.8118054772479197,
            "PIB": 0.9153051811202841,
            "CV": 0.4118196372088354,
            "note": "report/iterative_zernike_shaping/data.json :: gs",
        },
        "e2_score_gap_B_minus_A_gs_zernike": gaps,
        "e2_score_gain_defocus_over_plain_gs_zernike": defocus_gain,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "outputs": {
            "e1_csv": f"e1_defocus_sweep{rc.tag}.csv",
            "e2_csv": f"e2_methods{rc.tag}.csv",
            "e1_png": f"e1_cv_heatmap{rc.tag}.png",
            "e2_png": f"e2_cv_vs_dz{rc.tag}.png",
        },
    }
    (out / f"summary{rc.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"\n产物目录: {out}")
    logger.info(f"耗时: {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()