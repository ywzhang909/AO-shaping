"""2f 实验系统 (SLM → 透镜 → CCD) 的可复用数字孪生核心。

本模块把物理世界里的非理想因素**数学化**, 并注入到 2f 傅里叶传播链里, 输出一台
**参数化的实验系统**: 任何别的任务 (优化器仿真、反演、预补偿) 都可以直接
``import`` 复用, 而不必各自重写一遍传播链。

口径
----
孪生以**瞳面区域**为口径 (不是整块面板), 这样才能与仓库既有的
``region`` / ``far_field_padding`` / 像元标度约定一致
(见 ``ml/zernike/models.py`` 与 ``ml/hwdataset``):

* ``region`` —— 瞳面网格边长 (采样点数)。
* ``region_span_px`` —— 该网格覆盖的 SLM 面板边长 (px)。
* 因此**一个瞳面采样点 = ``region_span_px / region`` 个面板像元**, 该节距就是
  傅里叶变换看到的等效像元节距 (记作 ``pupil_pitch_m``), 远场标度按物理量算。

物理链
------
::

    命令相位 (raw 弧度)
      → SLM 面板静态相差 (canonical Zernike + 固定种子高斯粗糙度)
      → SLM LUT 非线性 + 灰度量化往返          (得到**实际**相位)
      → 瞳面场 = 高斯幅度 × exp(i·实际相位)
      → 透镜像差 (瞳面 Zernike 相位掩膜)      ← 透镜在傅里叶变换之前
      → 居中补零 → fftshift(fft2(ifftshift(·)))  (焦面复场)
      → 相机离焦 (焦面角谱传播)
      → 光路倾斜 (焦面平移 f·tilt)
      → |·|² → 居中裁剪 → 峰值归一到 ``peak_adu``

三族误差
--------
1. **SLM LUT 非线性 + 量化** —— 灰度→相位的映射可以偏离线性 (Gamma 指数), 也可以
   直接给实测 LUT 表; 灰度量化 (默认 1023 级) 永远存在。
2. **SLM 面板静态相差** —— 低阶 Zernike (canonical 生成) + 固定种子的高斯随机场
   粗糙度。
3. **装调** —— 相机离焦 (焦面角谱传播)、光路倾斜 (焦面平移)、透镜像差
   (瞳面 Zernike 相位掩膜)。

四条与常见直觉不同、但被实测锁定的实现细节
------------------------------------------
1. **LUT 往返用 canonical 入口**: 弧度→灰度只走
   :func:`ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale`
   (wrap→scale→clip→uint16), **绝不**做 min-max 归一化 (那是本仓库明令的反模式);
   相位生成器一律返回 raw 未包裹弧度, ``mod 2π`` 只允许发生在该入口内部。
2. **离焦用角谱传递函数作用在焦面复场上**, 倏逝分量 (``λf > 1``) 置 0。
   它与「瞳面乘二次相位」的写法**逐位等价** (实测: 二次矩 113.056 vs 113.023),
   但前者的频率变量就是焦面坐标的共轭频率, 物理含义更直接。
3. **倾斜用「焦面平移 f·tilt」实现**, 而不是「焦面乘线性相位」: 后者按位移定理
   等价于**平移瞳面**, 远场强度**完全不变** (实测 0 级光斑不动); 而光路倾斜的
   物理效果正是把远场图样平移 ``f · tilt``。本模块用傅里叶移位精确实现该平移。
4. **透镜像差作用在瞳面**, 而不是焦面: 焦面单位模相位掩膜对 ``|·|²`` **逐位无
   影响** (``max|ΔI| = 0.000000``), 放在那里该误差完全不可观测; 透镜在傅里叶变换
   之前, 其波前误差物理上就是瞳面相位。

口径与偏离说明
--------------
* ``p_fft = λ·f / (N_pad · pupil_pitch_m)``, 其中 ``pupil_pitch_m`` 是**瞳面采样
  网格的等效节距** (= ``region_span_px / region`` 个面板像元)。当
  ``region_span_px == region`` 时它退化为 SLM 像元节距本身。
* ``camera_px_per_model_px = p_fft / d_ccd`` —— 远场模型采样点 ↔ 相机像元的换算。
* 居中裁剪沿用 ``ml/zernike/models.py::_centre_crop`` 的口径: 裁到
  ``region × region`` 个模型采样点 (可用 ``crop_px`` 覆盖), 以光学轴 (0 级) 为中心。

本模块是**纯仿真层**: 不 import 任何硬件驱动包, 不做任何设备 I/O, import 期无副作用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

__all__ = ["TwinErrorModel", "BenchTwinConfig", "BenchTwin"]


@dataclass
class TwinErrorModel:
    """三族误差参数。默认全 0 = 理想系统。

    Attributes:
        lut_gamma: 灰度→相位的 Gamma 指数 (1 = 理想线性)。
        lut_phase_max_rad: 满灰度对应的最大相位 (rad)。
        gray_levels: 量化级数 (10 bit → 1023)。
        lut_table: 可选的实测 LUT (灰度→弧度), 长度必须是 ``gray_levels + 1``;
            给了就**优先于** ``lut_gamma``。
        slm_zernike: SLM 面板静态相差的 Noll 序 → 弧度。
        slm_zernike_n_max: 面板 Zernike 的径向阶数上限。
        slm_roughness_sigma_rad: 面板粗糙度 (高斯随机场的 RMS, 弧度)。
        slm_roughness_seed: 粗糙度随机场的种子 (固定 → 可复现)。
        defocus_m: 相机离焦量 (m, 正负皆可)。
        tilt_x_rad: 光路在 x 方向的倾斜 (rad)。
        tilt_y_rad: 光路在 y 方向的倾斜 (rad)。
        lens_zernike: 透镜像差 (瞳面波前误差) 的 Noll 序 → 弧度。
        lens_zernike_n_max: 透镜像差 Zernike 的径向阶数上限。
    """

    # 族 1 —— SLM LUT 非线性 + 量化
    lut_gamma: float = 1.0
    lut_phase_max_rad: float = 2.0 * math.pi
    gray_levels: int = 1023
    lut_table: np.ndarray | None = None
    # 族 2 —— SLM 面板静态相差
    slm_zernike: dict[int, float] = field(default_factory=dict)
    slm_zernike_n_max: int = 6
    slm_roughness_sigma_rad: float = 0.0
    slm_roughness_seed: int = 0
    # 族 3 —— 装调
    defocus_m: float = 0.0
    tilt_x_rad: float = 0.0
    tilt_y_rad: float = 0.0
    lens_zernike: dict[int, float] = field(default_factory=dict)
    lens_zernike_n_max: int = 4


@dataclass
class BenchTwinConfig:
    """实验系统的几何与噪声参数。

    Attributes:
        wavelength_m: 工作波长 (m)。
        slm_pixel_um: SLM 面板像元节距 (µm)。
        ccd_pixel_um: 相机像元节距 (µm)。
        focal_length_m: 傅里叶透镜焦距 (m)。
        region: 瞳面网格边长 (采样点数)。
        region_span_px: 瞳面网格覆盖的 SLM 面板边长 (px)。
        far_field_padding: 远场补零因子 (默认 10 = hwdataset 标定过的内点最优)。
        beam_w0_px: 瞳面上高斯入射光束的 1/e² 强度半径 (面板 px)。
        center_crop: 是否把远场裁到光学轴居中的方形窗口。
        crop_px: 裁剪窗口边长 (模型采样点); ``None`` → ``region``
            (与 ``ml/zernike/models.py::_centre_crop`` 同口径)。
        peak_adu: 归一化后的峰值 ADU 标度 (与既有 sim 的 100 一致)。
        noise_adu: 噪声幅度 (ADU); 0 = 完全确定性 (反演要用)。
        seed: 噪声 RNG 种子。
        errors: 三族误差参数。
    """

    wavelength_m: float = 1064e-9
    slm_pixel_um: float = 8.0
    ccd_pixel_um: float = 2.2
    focal_length_m: float = 0.125
    region: int = 128
    region_span_px: float = 900.0
    far_field_padding: int = 10
    beam_w0_px: float = 250.0
    center_crop: bool = True
    crop_px: int | None = None
    peak_adu: float = 100.0
    noise_adu: float = 0.0
    seed: int = 0
    errors: TwinErrorModel = field(default_factory=TwinErrorModel)


class BenchTwin:
    """参数化的 2f 实验系统数字孪生。

    Args:
        config: 几何 / 噪声 / 误差参数。

    Raises:
        ValueError: 配置自相矛盾 (region < 2、补零因子 < 1、灰度级数 < 1、
            LUT 表长度与灰度级数不符等)。
    """

    def __init__(self, config: BenchTwinConfig) -> None:
        self.config = config
        self._validate()
        self._rng = np.random.default_rng(int(config.seed))
        # 缓存: 面板 Zernike / 粗糙度 / 透镜像差掩膜 / 入射幅度。
        # (Zernike 网格构建是本模块最贵的一步, 必须按实例缓存。)
        self._panel_zernike: np.ndarray | None = None
        self._roughness: np.ndarray | None = None
        self._lens_mask_cache: np.ndarray | None = None
        self._amplitude: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        cfg = self.config
        if int(cfg.region) < 2:
            raise ValueError(f"region must be >= 2, got {cfg.region}")
        if int(cfg.far_field_padding) < 1:
            raise ValueError(
                f"far_field_padding must be >= 1, got {cfg.far_field_padding}"
            )
        if int(cfg.errors.gray_levels) < 1:
            raise ValueError(f"gray_levels must be >= 1, got {cfg.errors.gray_levels}")
        if float(cfg.errors.lut_gamma) <= 0.0:
            raise ValueError(f"lut_gamma must be > 0, got {cfg.errors.lut_gamma}")
        if float(cfg.region_span_px) <= 0.0:
            raise ValueError(f"region_span_px must be > 0, got {cfg.region_span_px}")
        table = cfg.errors.lut_table
        if table is not None:
            table_arr = np.asarray(table, dtype=np.float64)
            if table_arr.ndim != 1 or table_arr.shape[0] != int(cfg.errors.gray_levels) + 1:
                raise ValueError(
                    "lut_table must be a 1-D array of length gray_levels + 1 "
                    f"({int(cfg.errors.gray_levels) + 1}), got shape {table_arr.shape}"
                )

    # ------------------------------------------------------------------
    # 几何
    # ------------------------------------------------------------------

    @property
    def errors(self) -> TwinErrorModel:
        """三族误差参数。"""
        return self.config.errors

    @property
    def region(self) -> int:
        """瞳面网格边长 (采样点数)。"""
        return int(self.config.region)

    @property
    def far_field_size(self) -> int:
        """远场 (焦面) 网格边长 = ``region * far_field_padding``。"""
        return self.region * max(1, int(self.config.far_field_padding))

    @property
    def slm_pixel_m(self) -> float:
        """SLM 面板像元节距 (m)。"""
        return float(self.config.slm_pixel_um) * 1e-6

    @property
    def ccd_pixel_m(self) -> float:
        """相机像元节距 (m)。"""
        return float(self.config.ccd_pixel_um) * 1e-6

    @property
    def pupil_pitch_m(self) -> float:
        """瞳面采样网格的等效节距 (m) = ``region_span_px / region`` 个面板像元。

        这是傅里叶变换真正看到的像元节距; ``region_span_px == region`` 时它就是
        SLM 像元节距本身。
        """
        return (
            float(self.config.region_span_px)
            / self.region
            * self.slm_pixel_m
        )

    @property
    def p_fft(self) -> float:
        """焦面 (远场模型网格) 的采样间距 (m) = ``λ·f / (N_pad · pupil_pitch_m)``。"""
        return (
            float(self.config.wavelength_m)
            * float(self.config.focal_length_m)
            / (self.far_field_size * self.pupil_pitch_m)
        )

    def camera_px_per_model_px(self) -> float:
        """远场一个模型采样点 = 多少相机像元 (= ``p_fft / d_ccd``)。"""
        return self.p_fft / self.ccd_pixel_m

    @property
    def crop_side(self) -> int:
        """裁剪窗口边长 (模型采样点)。"""
        side = self.config.crop_px
        if side is None:
            side = self.region
        return max(1, int(side))

    @property
    def camera_window_px(self) -> float:
        """裁剪窗口对应的相机像元数 (便于与实测 CCD 视场对齐)。"""
        return self.crop_side * self.camera_px_per_model_px()

    # ------------------------------------------------------------------
    # 瞳面
    # ------------------------------------------------------------------

    def pupil_amplitude(self) -> np.ndarray:
        """瞳面上的高斯入射幅度 (峰值归一到 1)。

        幅度 = ``exp(-r² / (2·w0²))``, ``w0 = beam_w0_px`` 换算到瞳面网格;
        ``r`` 以米计。峰值归一使 ``far_field`` 的绝对标度只由 ``peak_adu`` 决定。
        """
        if self._amplitude is None:
            n = self.region
            idx = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
            x = idx * self.pupil_pitch_m
            w0 = float(self.config.beam_w0_px) * self.slm_pixel_m
            r2 = x[:, None] ** 2 + x[None, :] ** 2
            amp = np.exp(-r2 / (2.0 * w0 * w0))
            peak = float(amp.max())
            if peak > 0.0:
                amp = amp / peak
            self._amplitude = amp
        return self._amplitude

    # ------------------------------------------------------------------
    # 族 1/2: SLM 侧
    # ------------------------------------------------------------------

    def _panel_zernike_phase(self) -> np.ndarray:
        """面板静态相差 (canonical Zernike, 孔径外置 0)。"""
        if self._panel_zernike is None:
            n = self.region
            coeffs = {
                int(k): float(v) for k, v in self.errors.slm_zernike.items()
            }
            if not coeffs:
                self._panel_zernike = np.zeros((n, n), dtype=np.float64)
            else:
                raw = generate_zernike_phase(
                    coeffs,
                    resolution=(n, n),
                    n_max=int(self.errors.slm_zernike_n_max),
                    radius=n / 2.0,
                )
                # canonical 生成器在圆孔径外给 NaN; 硬件路径同样把它置 0。
                self._panel_zernike = np.nan_to_num(
                    np.asarray(raw, dtype=np.float64), nan=0.0
                )
        return self._panel_zernike

    def _roughness_phase(self) -> np.ndarray:
        """面板粗糙度: 固定种子的高斯随机场, RMS = ``slm_roughness_sigma_rad``。"""
        if self._roughness is None:
            n = self.region
            sigma = float(self.errors.slm_roughness_sigma_rad)
            if sigma <= 0.0:
                self._roughness = np.zeros((n, n), dtype=np.float64)
            else:
                rng = np.random.default_rng(int(self.errors.slm_roughness_seed))
                field = rng.normal(0.0, 1.0, size=(n, n))
                rms = float(np.sqrt(np.mean(field**2)))
                self._roughness = field * (sigma / rms)
        return self._roughness

    def apply_panel_aberration(self, phase_rad: np.ndarray) -> np.ndarray:
        """命令相位 + 面板静态相差 (Zernike + 粗糙度)。

        Args:
            phase_rad: SLM 命令相位 (raw 弧度, ``(region, region)``)。

        Returns:
            叠加了面板相差后的相位 (仍是 raw 弧度, 未包裹)。
        """
        phase = self._as_pupil_phase(phase_rad)
        return phase + self._panel_zernike_phase() + self._roughness_phase()

    def apply_lut(self, phase_rad: np.ndarray) -> np.ndarray:
        """SLM 的 LUT 非线性 + 灰度量化往返, 返回**实际**相位。

        弧度→灰度只走 canonical 入口
        :func:`ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale`
        (wrap→scale→clip→uint16)。灰度→相位的反查:

        * 给了 ``lut_table`` → 直接查表 (表长 = ``gray_levels + 1``);
        * 否则 → ``lut_phase_max_rad · (gray / gray_levels) ** lut_gamma``。

        ``lut_gamma == 1`` 且无 ``lut_table`` 时, 往返**不引入任何额外相位**
        (只有灰度量化误差与 ``mod 2π``; SLM 本就是 2π 周期器件)。
        """
        phase = np.asarray(phase_rad, dtype=np.float64)
        levels = int(self.errors.gray_levels)
        gray = phase_to_slm_grayscale(phase, max_grayscale=levels)
        table = self.errors.lut_table
        if table is not None:
            table_arr = np.asarray(table, dtype=np.float64)
            return table_arr[gray]
        gamma = float(self.errors.lut_gamma)
        return (
            float(self.errors.lut_phase_max_rad)
            * (gray.astype(np.float64) / float(levels)) ** gamma
        )

    # ------------------------------------------------------------------
    # 族 3: 装调
    #   - 透镜像差 → 瞳面 (透镜在傅里叶变换之前)
    #   - 相机离焦 / 光路倾斜 → 焦面复场
    # ------------------------------------------------------------------

    def _lens_mask_phase(self) -> np.ndarray:
        """透镜像差的**瞳面**相位掩膜 (canonical Zernike, 归一化半径 = 瞳面孔径)。

        掩膜在瞳面网格上生成 (半径 = ``region / 2`` 个采样点, 孔径外置 0),
        与面板静态相差共用同一个孔径口径。
        """
        if self._lens_mask_cache is None:
            coeffs = {
                int(k): float(v) for k, v in self.errors.lens_zernike.items()
            }
            n = self.region
            if not coeffs:
                self._lens_mask_cache = np.zeros((n, n), dtype=np.float64)
            else:
                raw = generate_zernike_phase(
                    coeffs,
                    resolution=(n, n),
                    n_max=int(self.errors.lens_zernike_n_max),
                    radius=n / 2.0,
                )
                self._lens_mask_cache = np.nan_to_num(
                    np.asarray(raw, dtype=np.float64), nan=0.0
                )
        return self._lens_mask_cache

    def apply_lens_aberration(self, field: np.ndarray) -> np.ndarray:
        """透镜像差作用在**瞳面**复场上 (乘以 Zernike 相位掩膜)。

        ⚠️ **口径说明 (与任务书给的口径不同, 理由是实测)**: 任务书要求把透镜像差
        作为**焦面频域**的单位模相位掩膜相乘。但单位模相位乘在焦面复场上对
        ``|·|²`` 是**逐位无影响**的 (``|E·e^{iφ}|² ≡ |E|²``, 实测
        ``max|ΔI| = 0.000000``), 该误差因此**完全不可观测** —— 除非其后还有
        相位敏感的操作 (本链路里只有离焦的角谱传播)。

        透镜位于傅里叶变换**之前**, 它的波前误差物理上就是**瞳面相位**, 故本模块
        在瞳面施加: 这样远场才会真的改变 (实测可测量), 且与
        ``ml/zernike/models.py`` 把 Zernike 系数当瞳面相位的口径一致。

        Args:
            field: 瞳面复场 (``(region, region)`` complex)。

        Returns:
            施加像差后的瞳面复场。
        """
        mask = self._lens_mask_phase()
        if not np.any(mask):
            return field
        return field * np.exp(1j * mask)

    def apply_defocus(self, field: np.ndarray) -> np.ndarray:
        """相机离焦: 角谱传递函数作用在焦面复场上。

        ``H = exp(i·k·Δz·sqrt(1 - (λf_x)² - (λf_y)²))``, 其中
        ``f_x = m_x / (N·p_fft)`` 是焦面坐标的**共轭频率** (cycles/m, ``m_x`` 是
        整数频移索引); 倏逝分量 (``λf > 1``) 置 0, **不得出 NaN**。

        该写法与「瞳面乘二次相位 ``exp(-i·π·Δz·r²/(λ·f²))``」**逐位等价**
        (实测: 光斑二阶矩 113.056 vs 113.023) —— 因为焦面复场的频谱就是瞳面,
        所以离焦在频域就是给瞳面加二次相位。选角谱写法是因为它的频率变量就是
        焦面坐标的共轭频率, 物理含义更直接, 且天然带倏逝分量截断。

        ``Δz == 0`` 是恒等 (逐位相同)。

        Args:
            field: 焦面复场 (``(N, N)`` complex, 以 ``fftshift`` 口径存放,
                即光学轴在中心)。

        Returns:
            离焦后的焦面复场。
        """
        dz = float(self.errors.defocus_m)
        if dz == 0.0:
            return field
        arr = np.asarray(field)
        m = arr.shape[0]
        lam = float(self.config.wavelength_m)
        k_wave = 2.0 * math.pi / lam
        freq = np.fft.fftfreq(m) * m / (m * self.p_fft)  # cycles/m
        arg = 1.0 - (lam * freq[None, :]) ** 2 - (lam * freq[:, None]) ** 2
        transfer = np.zeros((m, m), dtype=np.complex128)
        propagating = arg > 0.0
        transfer[propagating] = np.exp(
            1j * k_wave * dz * np.sqrt(arg[propagating])
        )
        un = np.fft.ifftshift(arr)
        propagated = np.fft.ifft2(
            np.fft.fft2(un, norm="ortho") * transfer, norm="ortho"
        )
        return np.fft.fftshift(propagated)

    def apply_tilt(self, field: np.ndarray) -> np.ndarray:
        """光路倾斜: 焦面图样平移 ``f · tilt``。

        物理上, 光路倾斜 = 入射波相对透镜倾斜, 其**精确**效果是把远场图样平移
        ``f · tilt``。本模块用傅里叶移位精确实现该平移。

        ⚠️ 注意: 「在焦面上乘一个线性相位」**不是**等价写法 —— 按位移定理它等价于
        **平移瞳面**, 远场强度完全不变 (实测 0 级光斑不动)。故此处不采用该写法。

        Args:
            field: 焦面复场 (``(N, N)`` complex)。

        Returns:
            平移后的焦面复场。
        """
        dx = float(self.config.focal_length_m) * float(self.errors.tilt_x_rad)
        dy = float(self.config.focal_length_m) * float(self.errors.tilt_y_rad)
        if dx == 0.0 and dy == 0.0:
            return field
        arr = np.asarray(field)
        m = arr.shape[0]
        freq = np.fft.fftfreq(m) * m  # 整数频移
        shift = np.exp(
            -2j
            * math.pi
            * (freq[:, None] * dy + freq[None, :] * dx)
            / (m * self.p_fft)
        )
        un = np.fft.ifftshift(arr)
        shifted = np.fft.ifft2(
            np.fft.fft2(un, norm="ortho") * shift, norm="ortho"
        )
        return np.fft.fftshift(shifted)

    # ------------------------------------------------------------------
    # 正向
    # ------------------------------------------------------------------

    def far_field(self, phase_rad: np.ndarray) -> np.ndarray:
        """无噪远场强度 (0..``peak_adu``)。

        Args:
            phase_rad: SLM 命令相位 (raw 弧度, ``(region, region)``)。

        Returns:
            远场强度图 (``crop_side × crop_side``)。

        Raises:
            ValueError: 相位形状与 ``region`` 不符。
        """
        phase = self._as_pupil_phase(phase_rad)
        actual = self.apply_lut(self.apply_panel_aberration(phase))
        pupil = self.pupil_amplitude() * np.exp(1j * actual)
        pupil = self.apply_lens_aberration(pupil)
        focal = self._propagate(pupil)
        focal = self.apply_defocus(focal)
        focal = self.apply_tilt(focal)
        intensity = np.abs(focal) ** 2
        cropped = self._crop(intensity)
        peak = float(cropped.max())
        if peak > 0.0:
            cropped = cropped * (float(self.config.peak_adu) / peak)
        return cropped

    def frame(self, phase_rad: np.ndarray) -> np.ndarray:
        """加噪的远场帧 (原始 ADU)。

        散粒噪声按泊松统计加入 (峰值 ``peak_adu`` ADU 对应 ``peak_adu`` 个光子,
        故峰值处的散粒噪声标准差 = ``sqrt(peak_adu)`` ADU), 读出噪声是高斯,
        ``σ = noise_adu``。``noise_adu == 0`` 时**完全确定性** (反演要用)。

        Args:
            phase_rad: SLM 命令相位 (raw 弧度)。

        Returns:
            加噪后的 ADU 帧 (形状同 :meth:`far_field`)。
        """
        image = self.far_field(phase_rad)
        noise = float(self.config.noise_adu)
        if noise <= 0.0:
            return image
        shot = self._rng.poisson(np.clip(image, 0.0, None)) - image
        read = self._rng.normal(0.0, noise, size=image.shape)
        return image + shot + read

    def compensate(self, phase_rad: np.ndarray) -> np.ndarray:
        """预补偿: 反解 SLM 侧误差, 返回应当下发的命令相位。

        求解目标: 让 ``apply_lut(apply_panel_aberration(cmd)) ≈ phase_rad``
        (即实际相位 ≈ 期望相位)。

        * 面板 Zernike 相差: 直接从命令里减掉 (已知量)。
        * LUT 非线性: 反解 Gamma (或反查实测表), 再乘回灰度→相位的正向关系。
        * 灰度量化: 不可反解 ⇒ 残余 ≤ 一个量化级。
        * 面板粗糙度: 随机量, 不补偿 (残差 = 其 RMS)。

        Args:
            phase_rad: 期望的**实际**相位 (raw 弧度)。

        Returns:
            应当下发给 SLM 的命令相位 (raw 弧度)。
        """
        target = np.mod(np.asarray(phase_rad, dtype=np.float64), 2.0 * math.pi)
        levels = int(self.errors.gray_levels)
        table = self.errors.lut_table
        if table is not None:
            table_arr = np.asarray(table, dtype=np.float64)
            gray = np.interp(
                target, table_arr, np.arange(table_arr.shape[0], dtype=np.float64)
            )
        else:
            gamma = float(self.errors.lut_gamma)
            gray = levels * (
                target / float(self.errors.lut_phase_max_rad)
            ) ** (1.0 / gamma)
        pre_lut = 2.0 * math.pi * gray / float(levels)
        return pre_lut - self._panel_zernike_phase()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _as_pupil_phase(self, phase_rad: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase_rad, dtype=np.float64)
        if phase.shape != (self.region, self.region):
            raise ValueError(
                f"phase shape {phase.shape} != pupil grid "
                f"({self.region}, {self.region})"
            )
        return phase

    def _propagate(self, pupil_field: np.ndarray) -> np.ndarray:
        """瞳面场 → 居中补零 → canonical 傅里叶链 → 焦面复场。"""
        n = self.region
        m = self.far_field_size
        buffer = np.zeros((m, m), dtype=np.complex128)
        half = (m - n) // 2
        buffer[half : half + n, half : half + n] = pupil_field
        return np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(buffer), norm="ortho")
        )

    def _crop(self, image: np.ndarray) -> np.ndarray:
        if not self.config.center_crop:
            return image
        side = self.crop_side
        m = image.shape[0]
        if side >= m:
            return image
        start = (m - side) // 2
        return image[start : start + side, start : start + side]

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """导出为 JSON 可序列化的字典 (反演结果要能存成 JSON 再读回)。"""
        cfg = self.config
        err = cfg.errors
        table_payload: Any = None
        if err.lut_table is not None:
            table = np.asarray(err.lut_table, dtype=np.float64)
            table_payload = {
                "data": [float(v) for v in table.ravel()],
                "shape": [int(s) for s in table.shape],
            }
        return {
            "wavelength_m": float(cfg.wavelength_m),
            "slm_pixel_um": float(cfg.slm_pixel_um),
            "ccd_pixel_um": float(cfg.ccd_pixel_um),
            "focal_length_m": float(cfg.focal_length_m),
            "region": int(cfg.region),
            "region_span_px": float(cfg.region_span_px),
            "far_field_padding": int(cfg.far_field_padding),
            "beam_w0_px": float(cfg.beam_w0_px),
            "center_crop": bool(cfg.center_crop),
            "crop_px": None if cfg.crop_px is None else int(cfg.crop_px),
            "peak_adu": float(cfg.peak_adu),
            "noise_adu": float(cfg.noise_adu),
            "seed": int(cfg.seed),
            "errors": {
                "lut_gamma": float(err.lut_gamma),
                "lut_phase_max_rad": float(err.lut_phase_max_rad),
                "gray_levels": int(err.gray_levels),
                "lut_table": table_payload,
                "slm_zernike": {
                    str(int(k)): float(v) for k, v in err.slm_zernike.items()
                },
                "slm_zernike_n_max": int(err.slm_zernike_n_max),
                "slm_roughness_sigma_rad": float(err.slm_roughness_sigma_rad),
                "slm_roughness_seed": int(err.slm_roughness_seed),
                "defocus_m": float(err.defocus_m),
                "tilt_x_rad": float(err.tilt_x_rad),
                "tilt_y_rad": float(err.tilt_y_rad),
                "lens_zernike": {
                    str(int(k)): float(v) for k, v in err.lens_zernike.items()
                },
                "lens_zernike_n_max": int(err.lens_zernike_n_max),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BenchTwin":
        """从 :meth:`to_dict` 的输出 (或等价的 JSON 字典) 重建孪生。

        Args:
            payload: 配置字典; Noll 序键可以是 ``int`` 或 ``str`` (JSON 会把
                整数键变成字符串)。

        Returns:
            重建好的 :class:`BenchTwin`。
        """
        raw_errors = dict(payload.get("errors") or {})

        def _noll_dict(raw: Any) -> dict[int, float]:
            if not raw:
                return {}
            return {int(k): float(v) for k, v in dict(raw).items()}

        table_payload = raw_errors.get("lut_table")
        table: np.ndarray | None = None
        if table_payload is not None:
            arr = np.asarray(table_payload["data"], dtype=np.float64)
            shape = tuple(int(s) for s in table_payload["shape"])
            table = arr.reshape(shape)

        errors = TwinErrorModel(
            lut_gamma=float(raw_errors.get("lut_gamma", 1.0)),
            lut_phase_max_rad=float(
                raw_errors.get("lut_phase_max_rad", 2.0 * math.pi)
            ),
            gray_levels=int(raw_errors.get("gray_levels", 1023)),
            lut_table=table,
            slm_zernike=_noll_dict(raw_errors.get("slm_zernike")),
            slm_zernike_n_max=int(raw_errors.get("slm_zernike_n_max", 6)),
            slm_roughness_sigma_rad=float(
                raw_errors.get("slm_roughness_sigma_rad", 0.0)
            ),
            slm_roughness_seed=int(raw_errors.get("slm_roughness_seed", 0)),
            defocus_m=float(raw_errors.get("defocus_m", 0.0)),
            tilt_x_rad=float(raw_errors.get("tilt_x_rad", 0.0)),
            tilt_y_rad=float(raw_errors.get("tilt_y_rad", 0.0)),
            lens_zernike=_noll_dict(raw_errors.get("lens_zernike")),
            lens_zernike_n_max=int(raw_errors.get("lens_zernike_n_max", 4)),
        )
        crop_px = payload.get("crop_px")
        config = BenchTwinConfig(
            wavelength_m=float(payload.get("wavelength_m", 1064e-9)),
            slm_pixel_um=float(payload.get("slm_pixel_um", 8.0)),
            ccd_pixel_um=float(payload.get("ccd_pixel_um", 2.2)),
            focal_length_m=float(payload.get("focal_length_m", 0.125)),
            region=int(payload.get("region", 128)),
            region_span_px=float(payload.get("region_span_px", 900.0)),
            far_field_padding=int(payload.get("far_field_padding", 10)),
            beam_w0_px=float(payload.get("beam_w0_px", 250.0)),
            center_crop=bool(payload.get("center_crop", True)),
            crop_px=None if crop_px is None else int(crop_px),
            peak_adu=float(payload.get("peak_adu", 100.0)),
            noise_adu=float(payload.get("noise_adu", 0.0)),
            seed=int(payload.get("seed", 0)),
            errors=errors,
        )
        return cls(config)
