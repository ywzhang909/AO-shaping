"""FourierGSNet 仿真环境 (SimSLM / SimCCD / SimFourierGSNetEnv)。

真实 2f 傅里叶台架 (SLM 前焦面 → f=125mm 透镜 → CCD 后焦面) 的数字孪生:
SLM 面板按 Santec SLM200 的 1920×1200 建, CCD 远场即 SLM 场的夫琅禾费衍射图样,
按 CCD 像元格采样。

物理模型
---------
* 夫琅禾费传播采用归一化 FFT 约定 (0 级落在画面正中)::

      E = fftshift(fft2(ifftshift(U), norm="ortho"))

* FFT 网格步长 ``p_fft = lamb * f / (P * d_slm)``, ``P`` 是补零后的网格边长
  (≥ max(K_px, 光束 region) 的最小 2 的幂)。CCD 画面由 P×P 强度图双线性重采样
  到 ``K_px × K_px`` (``scipy.ndimage.zoom``, 系数 ``K_px / P``): FFT bin 偏移 ``D``
  映射到 ``D * K_px / P`` 个 CCD 像素, 故周期 ``P_slm`` 个 SLM 像素的光栅, 其 +1
  级恰好落在距 0 级 ``K_px / P_slm`` 像素处 —— 即 **K 定律**。
* 像素包络: 每个 SLM 像元有有限有效宽度 ``d_eff``, 故远场乘一个可分离 sinc 包络
  ``sinc(d_eff * u / (lamb * f))``, 首零点位于 ``K_px * d_slm / d_eff`` 个 CCD 像素处。
  默认 ``d_eff = 7.8 um`` 把首零点推到 ~2100 px, 落在 K=2048 的带外; sinc 包络测试用
  ``d_eff = 4 * d_slm``, 使零点落在带内 ``K_px / 4 = 512 px``。
* 光束是宽度 ``w0`` (默认 250 px) 的高斯, 铺在 ``region × region`` (默认 512) 的
  面板块上, 以 ``beam_center`` (默认面板中心 (960, 600), (行, 列)) 为中心; 该块被放到
  P×P 网格的 ``P / 2`` 位置, 使 0 级居中。
* 静态像差 (``env.aberrations``, Noll 索引 → 系数) 由 ``ZernikeGenerator``
  (半径 = region / 2) 生成, 在 ``render_intensity`` 的**唯一一次**灰度量化之前叠加到
  已显示相位上 —— 与直接向 ``display_phase`` 写 ``base + aberration`` 完全等价。
  Zernike 多项式只在单位圆内有定义, 故圆外为 NaN, 用 ``nan_to_num`` 置零。

鸭子类型契约 (对齐真实 Santec SLM / MiiCam 驱动)
------------------------------------------------
* ``slm.display_phase(phase_rad, wait_time_s=, memory_number=, memory_mode=) -> int``
* ``slm.display_data(gray_uint16, ...) -> int``
* ``slm.get_displayed_memory_number() -> int``
* ``ccd.get_numpy_image(n_sample=1) -> np.uint16``
* ``MEMORY_MODE_INTERNAL == 0``, ``Panel_Res == (1920, 1200)``

灰度深度 255 (对应 ``SLMLUTCalibrator.factory_2pi = 255.0``): ``display_phase`` 只存
理想 (未量化) 相位, 量化统一发生在 ``render_intensity``。

与其他核心模块的关系
--------------------
本模块是 **纯仿真层**, 不 import 任何硬件驱动, 因此可被离线训练/评估直接消费:

* 上游物理口径唯一: FFT 约定与 K 定律同时被
  ``algorithm/signal_processing/zernike_coefficient_optimizer.py`` 的正向模型采用,
  两边逐位可比 —— 这是 model-in-the-loop 拟合结果可直接与注入真值比较的前提。
* 下游主要消费者:
  - ``optimizer/wfless/model_in_loop_shaping.py`` — 仅仿真, 用 ``BeamParams(native=True)``
    让面板网格就是模型网格, 环路内不引入任何重采样;
  - ``scripts/fouriergsnet_sim_train.py`` — 场景矩阵离线训练 (该脚本依赖的独立 CLI
    ``fouriergsnet_optimize.py`` 已于 f19dced 从仓库根删除, 脚本目前不可运行);
  - 物理回归测试 ``tests/ao_shaping/drivers/sim/test_sim_fouriergsnet.py`` 与湍流测试
    ``test_sim_fouriergsnet_turbulence.py``。
* 与 ``drivers/sim/slm_pib_sim.py`` 的区别: 后者是 SLM-PIB 族的台架孪生并注册为
  ``--cam_type sim`` 的相机, 本模块**不注册**任何设备类型, 只能被显式构造。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from scipy import ndimage

from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

# 硬件契约字面量 (对齐真实 Santec SLM200 驱动)。
MEMORY_MODE_INTERNAL = 0
PANEL_RES = (1920, 1200)  # (宽, 高) — 驱动约定
PANEL_H, PANEL_W = 1920, 1200  # 标定器约定 (h, w)

# 2f 傅里叶台架的默认光学参数。
LAMBDA = 1064e-9
D_SLM = 8e-6
D_CCD = 2.2e-6
F_LENS = 0.125
D_EFF = 7.8e-6


@dataclass
class BeamParams:
    """SLM 面板上的高斯光束参数。

    ``native`` 选择**原生节距方形面板**几何 (默认 ``False`` 保留历史的 1920×1200
    各向异性面板)。置 ``True`` 时, 显示的模型全息图以**原生像元节距**铺在
    ``region × region`` 的小方形孔径上, 使模型的 FFT 正向与本环境的
    ``render_intensity`` FFT 逐位一致 —— 这正是闭环能真正整形出方形的原因。
    消费方见 ``optimizer/wfless/model_in_loop_shaping.py``。
    """

    region: int = 512
    w0: float = 250.0
    center: tuple[int, int] | None = None  # 面板上的 (行, 列); None = 面板中心
    native: bool = False


@dataclass
class CalibNoise:
    """注入到渲染远场中的标定噪声模型。

    ``deltaK_fraction`` 按轴给 Kx/Ky 加一个随机比例扰动, ``center_offset_px`` 整帧
    平移随机偏移, ``rotation_deg`` 旋转画面, ``scale`` 全局缩放 K (确定性项)。
    """

    deltaK_fraction: float = 0.0
    center_offset_px: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0


class SimSLM:
    """模拟 Santec SLM200 (鸭子类型对齐真实驱动契约)。

    与 ``slm_pib_sim.SimSLMPib`` 的区别: 本类不参与设备注册, 只服务
    ``SimFourierGSNetEnv``, 因此 ``Panel_Res`` 固定为 1920×1200 (native 模式除外)。
    """

    MEMORY_MODE_INTERNAL = 0
    Panel_Res = (1920, 1200)
    _max_gray = 255  # 工厂 LUT: 2π 对应灰度 255 (SLMLUTCalibrator.factory_2pi)

    def __init__(
        self,
        env: "SimFourierGSNetEnv",
        lut_nonlinearity: float = 0.0,
        amplitude_coupling: float = 0.0,
        panel_res: tuple[int, int] | None = None,
    ) -> None:
        self._env = env
        self.lut_nonlinearity = lut_nonlinearity
        self.amplitude_coupling = amplitude_coupling
        self.panel_h, self.panel_w = (
            (panel_res[0], panel_res[1]) if panel_res is not None else (PANEL_H, PANEL_W)
        )
        self._phase = np.zeros((self.panel_h, self.panel_w), dtype=np.float64)
        self.phase_calls: list[dict[str, Any]] = []
        self.data_calls: list[dict[str, Any]] = []
        self.displayed_memory_number = 0
        self._version = 0

    def display_phase(
        self,
        phase_rad: np.ndarray,
        wait_time_s: float = 0.0,
        memory_number: int | None = None,
        memory_mode: int = 0,
    ) -> int:
        """保存理想相位 (弧度); 量化推迟到渲染时统一做一次。"""
        phase = np.asarray(phase_rad, dtype=np.float64)
        if phase.shape != (self.panel_h, self.panel_w):
            raise ValueError(
                f"display_phase expects ({self.panel_h}, {self.panel_w}) phase, got {phase.shape}"
            )
        self._phase = phase
        self.phase_calls.append(
            {
                "wait_time_s": float(wait_time_s),
                "memory_number": memory_number,
                "memory_mode": int(memory_mode),
            }
        )
        self._version += 1
        return 0

    def display_data(
        self,
        gray_uint16: np.ndarray,
        wait_time_s: float = 0.0,
        memory_number: int | None = None,
        memory_mode: int = 0,
    ) -> int:
        """保存已量化的 uint16 原始灰度图案。"""
        gray = np.asarray(gray_uint16, dtype=np.uint16)
        if gray.shape != (self.panel_h, self.panel_w):
            raise ValueError(
                f"display_data expects ({self.panel_h}, {self.panel_w}) gray, got {gray.shape}"
            )
        self._phase = self._gray_to_phase(gray)
        self.data_calls.append(
            {
                "wait_time_s": float(wait_time_s),
                "memory_number": memory_number,
                "memory_mode": int(memory_mode),
            }
        )
        self._version += 1
        return 0

    def get_displayed_memory_number(self) -> int:
        return self.displayed_memory_number

    def _phase_to_gray(self, phase: np.ndarray) -> np.ndarray:
        return np.round(np.mod(phase, 2 * np.pi) / (2 * np.pi) * self._max_gray).astype(
            np.uint16
        )

    def _gray_to_phase(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float64) / self._max_gray
        return (g * 2 * np.pi + self.lut_nonlinearity * np.sin(2 * np.pi * g)).astype(
            np.float64
        )


class SimCCD:
    """模拟置于后焦面的 CCD (鸭子类型对齐 MiiCam)。"""

    def __init__(
        self,
        env: "SimFourierGSNetEnv",
        peak_photons: float = 60000.0,
        read_noise_e: float = 0.0,
        noise_enabled: bool = True,
        seed: int = 0,
    ) -> None:
        self._env = env
        self.peak_photons = peak_photons
        self.read_noise_e = read_noise_e
        self.noise_enabled = noise_enabled
        self._rng = np.random.default_rng(seed)

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        """渲染远场并返回 uint16 画面 (K_px × K_px)。"""
        intensity = self._env.render_intensity()
        intensity = intensity / intensity.max()
        if not self.noise_enabled:
            return np.round(intensity * self.peak_photons).astype(np.uint16)
        frame = np.zeros_like(intensity)
        for _ in range(int(n_sample)):
            photons = self._rng.poisson(intensity * self.peak_photons)
            if self.read_noise_e > 0:
                photons = photons + self._rng.normal(
                    0.0, self.read_noise_e, size=intensity.shape
                )
            frame += photons
        frame /= n_sample
        return np.clip(np.round(frame), 0, 65535).astype(np.uint16)


class SimFourierGSNetEnv:
    """2f 傅里叶台架 (SLM → 透镜 → CCD) 的数字孪生。"""

    def __init__(
        self,
        lamb: float = LAMBDA,
        d_slm: float = D_SLM,
        d_ccd: float = D_CCD,
        f: float = F_LENS,
        d_eff: float = D_EFF,
        K_px: int | None = None,
        beam: BeamParams | None = None,
        calib_noise: CalibNoise | None = None,
        peak_photons: float = 60000.0,
        read_noise_e: float = 0.0,
        noise_enabled: bool = True,
        seed: int = 0,
    ) -> None:
        self.lamb = lamb
        self.d_slm = d_slm
        self.d_ccd = d_ccd
        self.f = f
        self.d_eff = d_eff
        self.K_px = (
            int(round(lamb * f / (d_slm * d_ccd))) if K_px is None else int(K_px)
        )
        self.beam = beam if beam is not None else BeamParams()
        self.calib_noise = calib_noise if calib_noise is not None else CalibNoise()
        self._rng = np.random.default_rng(seed)
        self._native = self.beam.native
        # native 模式: 中心默认取面板中心, 且 region 必须等于模型全息图边长 (64),
        # 才能保证 1:1 原生节距映射。
        if self.beam.center is None:
            if self._native:
                self.beam.center = (int(self.beam.region) // 2, int(self.beam.region) // 2)
            else:
                self.beam.center = (960, 600)
        self.P = max(
            1 << (self.K_px - 1).bit_length(),
            1 << (self.beam.region - 1).bit_length(),
        )
        self.p_fft = lamb * f / (self.P * d_slm)
        # native 模式: 节距等于 region 的方形各向同性面板, 于是 64×64 的模型全息图
        # 坐在原生像元节距上 (不做带限上采样), 环境 FFT 与模型正向完全一致。
        if self._native:
            self.PANEL_H = self.PANEL_W = int(self.beam.region)
            self.slm = SimSLM(self, panel_res=(int(self.beam.region), int(self.beam.region)))
        else:
            self.PANEL_H, self.PANEL_W = PANEL_H, PANEL_W
            self.slm = SimSLM(self)
        self.ccd = SimCCD(
            self,
            peak_photons=peak_photons,
            read_noise_e=read_noise_e,
            noise_enabled=noise_enabled,
            seed=seed,
        )
        self.aberrations: dict[int, float] = {}
        self._zgen: ZernikeGenerator | None = None
        self._beam_amp = self._build_beam_amp()
        self._render_cache: np.ndarray | None = None
        self._render_key: tuple[Any, ...] | None = None
        # 时变湍流 (Ornstein-Uhlenbeck) —— 默认不激活。
        self._turb_rng: np.random.Generator | None = None
        self._turb_nolls: tuple[int, ...] = ()
        self._turb_sigma: float = 0.0
        self._turb_tau: float = 1.0
        self._turb_dt: float = 1.0
        self._turb_state: dict[int, float] = {}

    # -- 时变湍流 (Ornstein-Uhlenbeck) ---------------------------------------

    def configure_turbulence(
        self,
        seed: int = 42,
        sigma: float = 0.5,
        tau: float = 50.0,
        dt: float = 1.0,
        nolls: tuple[int, ...] = (4, 5, 6, 11, 13),
    ) -> None:
        """开启低阶像差的时变漂移。

        每个被配置的 Noll 系数走一条均值回归的 Ornstein-Uhlenbeck 过程, 每次
        :meth:`advance_time` (即一个闭环帧) 推进一步::

            x <- x - (x / tau) * dt + sigma * sqrt(2 * dt / tau) * N(0, 1)

        平稳分布为 ``N(0, sigma**2)``, 故 ``sigma`` 是以**弧度**为单位的平稳标准差
        (与 ``self.aberrations`` 同单位), ``tau`` 是以**帧**为单位的弛豫时间。
        逐模式状态重置为 ``0.0`` 并新建私有带种子生成器, 使轨迹可复现;
        ``self.aberrations`` 在第一次 :meth:`advance_time` 之前保持不动。
        """
        self._turb_rng = np.random.default_rng(seed)
        self._turb_nolls = tuple(int(n) for n in nolls)
        self._turb_sigma = float(sigma)
        self._turb_tau = float(tau)
        self._turb_dt = float(dt)
        self._turb_state = {noll: 0.0 for noll in self._turb_nolls}
        logger.debug(
            "Turbulence configured: seed={}, sigma={}, tau={}, dt={}, nolls={}",
            seed,
            self._turb_sigma,
            self._turb_tau,
            self._turb_dt,
            self._turb_nolls,
        )

    def advance_time(self) -> None:
        """把 OU 湍流推进一帧并写入 ``self.aberrations``。

        未配置湍流时是空操作 (向后兼容)。写入的系数单位为**弧度**, 与静态像差
        语义一致; 不做 mod 2π 包裹。
        """
        if self._turb_rng is None or not self._turb_nolls:
            return
        sigma = self._turb_sigma
        tau = self._turb_tau
        dt = self._turb_dt
        noise_scale = sigma * np.sqrt(2.0 * dt / tau)
        for noll in self._turb_nolls:
            x = self._turb_state.get(noll, 0.0)
            x = x - (x / tau) * dt + noise_scale * float(
                self._turb_rng.standard_normal()
            )
            self._turb_state[noll] = x
            self.aberrations[noll] = x

    @property
    def turbulence_active(self) -> bool:
        """``configure_turbulence`` 是否已开启 OU 漂移。"""
        return self._turb_rng is not None and bool(self._turb_nolls)

    @property
    def turbulence_nolls(self) -> tuple[int, ...]:
        """已配置的 Noll 索引 (湍流未激活时为空元组)。"""
        return self._turb_nolls


    # -- 公开 API ---------------------------------------------------------

    def render_intensity(self) -> np.ndarray:
        """渲染 CCD 远场强度 (K_px × K_px, float64)。

        渲染键 = (SLM 版本号, 排序后的像差字典), 因此连续两次读取同一相位会命中
        缓存 —— 这是 ``advance_time`` 之外的第二条"内容未变"快路径。
        """
        key = (self.slm._version, tuple(sorted(self.aberrations.items())))
        if self._render_cache is not None and self._render_key == key:
            return self._render_cache
        P = self.P
        region = self.beam.region
        off = (P - region) // 2
        patch = self._extract_region(self.slm._phase)
        if self.aberrations:
            patch = patch + self._aberration_phase()
        patch = np.nan_to_num(patch, nan=0.0)
        U = np.zeros((P, P), dtype=np.complex128)
        U[off : off + region, off : off + region] = self._beam_amp * np.exp(1j * patch)
        E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U), norm="ortho"))
        u = (np.arange(P) - P / 2) * self.p_fft
        env = np.sinc(self.d_eff * u / (self.lamb * self.f))
        E = E * env[None, :] * env[:, None]
        I = np.abs(E) ** 2
        I = ndimage.zoom(I, self.K_px / P, order=1)
        I = self._apply_calib_noise(I)
        self._render_cache = I
        self._render_key = key
        return I

    def inject_calib_noise(self, calib: dict[str, Any]) -> dict[str, Any]:
        """返回标定字典 (Kx/Ky/center/rotation) 的扰动副本。"""
        cn = self.calib_noise
        out = dict(calib)
        dkx = (
            self._rng.uniform(-cn.deltaK_fraction, cn.deltaK_fraction)
            if cn.deltaK_fraction > 0
            else 0.0
        )
        dky = (
            self._rng.uniform(-cn.deltaK_fraction, cn.deltaK_fraction)
            if cn.deltaK_fraction > 0
            else 0.0
        )
        out["Kx"] = out.get("Kx", self.K_px) * cn.scale * (1 + dkx)
        out["Ky"] = out.get("Ky", self.K_px) * cn.scale * (1 + dky)
        cx, cy = out.get("center", (self.K_px / 2, self.K_px / 2))
        dx = (
            self._rng.uniform(-cn.center_offset_px, cn.center_offset_px)
            if cn.center_offset_px > 0
            else 0.0
        )
        dy = (
            self._rng.uniform(-cn.center_offset_px, cn.center_offset_px)
            if cn.center_offset_px > 0
            else 0.0
        )
        out["center"] = (cx + dx, cy + dy)
        dr = (
            self._rng.uniform(-cn.rotation_deg, cn.rotation_deg)
            if cn.rotation_deg > 0
            else 0.0
        )
        out["rotation_deg"] = out.get("rotation_deg", 0.0) + dr
        return out

    # -- 内部实现 ----------------------------------------------------------

    def _build_beam_amp(self) -> np.ndarray:
        region = self.beam.region
        yy, xx = np.mgrid[0:region, 0:region]
        r2 = (yy - region / 2) ** 2 + (xx - region / 2) ** 2
        return np.exp(-r2 / (2 * self.beam.w0**2)).astype(np.float64)

    def _extract_region(self, panel_phase: np.ndarray) -> np.ndarray:
        region = self.beam.region
        ph, pw = self.slm.panel_h, self.slm.panel_w
        r0 = self.beam.center[0] - region // 2
        c0 = self.beam.center[1] - region // 2
        patch = np.zeros((region, region), dtype=np.float64)
        r_lo, r_hi = max(r0, 0), min(r0 + region, ph)
        c_lo, c_hi = max(c0, 0), min(c0 + region, pw)
        patch[r_lo - r0 : r_hi - r0, c_lo - c0 : c_hi - c0] = panel_phase[
            r_lo:r_hi, c_lo:c_hi
        ]
        return patch

    def _zernike_generator(self) -> ZernikeGenerator:
        if self._zgen is None:
            region = self.beam.region
            self._zgen = ZernikeGenerator((region, region), radius=region / 2, n_orders=6)
        return self._zgen

    def _aberration_phase(self) -> np.ndarray:
        z = self._zernike_generator()
        coeffs: dict[tuple[int, int], float] = {}
        for noll, c in self.aberrations.items():
            n, m = z.noll_to_nm(noll)
            coeffs[(n, m)] = c
        return z.generate_polynomial(coeffs)

    def _apply_calib_noise(self, I: np.ndarray) -> np.ndarray:
        cn = self.calib_noise
        dkx = (
            self._rng.uniform(-cn.deltaK_fraction, cn.deltaK_fraction)
            if cn.deltaK_fraction > 0
            else 0.0
        )
        dky = (
            self._rng.uniform(-cn.deltaK_fraction, cn.deltaK_fraction)
            if cn.deltaK_fraction > 0
            else 0.0
        )
        kx = self.K_px * cn.scale * (1 + dkx)
        ky = self.K_px * cn.scale * (1 + dky)
        if (kx, ky) != (self.K_px, self.K_px):
            I = ndimage.zoom(I, (ky / self.P, kx / self.P), order=1)
            I = self._crop_or_pad(I, self.K_px)
        if cn.rotation_deg != 0.0:
            I = ndimage.rotate(I, cn.rotation_deg, reshape=False, order=1)
        if cn.center_offset_px != 0.0:
            dy = (
                self._rng.uniform(-cn.center_offset_px, cn.center_offset_px)
                if cn.center_offset_px > 0
                else 0.0
            )
            dx = (
                self._rng.uniform(-cn.center_offset_px, cn.center_offset_px)
                if cn.center_offset_px > 0
                else 0.0
            )
            I = ndimage.shift(I, (dy, dx), order=1)
        return I

    @staticmethod
    def _crop_or_pad(I: np.ndarray, n: int) -> np.ndarray:
        h, w = I.shape
        if h == n and w == n:
            return I
        out = np.zeros((n, n), dtype=I.dtype)
        r0 = max((n - h) // 2, 0)
        c0 = max((n - w) // 2, 0)
        src_r0 = max((h - n) // 2, 0)
        src_c0 = max((w - n) // 2, 0)
        rr, cc = min(h, n), min(w, n)
        out[r0 : r0 + rr, c0 : c0 + cc] = I[src_r0 : src_r0 + rr, src_c0 : src_c0 + cc]
        return out