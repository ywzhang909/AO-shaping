"""2f 傅里叶实验系统的数字孪生: 面板灰度 → 远场 CCD 帧 (无硬件依赖)。

本模块给 :mod:`ao_shaping.tools.slm.calibration.diffraction_efficiency.core`
的分析链路提供一个**纯 numpy 的仿真后端**: 下发一张面板灰度图, 拿回一帧远场
画面, 于是 "图案生成 → 阶次识别 → 能量积分 → 效率解算 → 相位响应拟合"
整条路可以在没有实验系统的情况下被完整走一遍。仿真验的是**同一条**代码路径,
不是另一条平行实现。

模型了什么
----------
* **入射光**: 瞳面 (pupil) 圆盘 (半径 ``pupil_radius_px`` 面板像元) 上的高斯
  束 (束腰 ``w0_px`` 面板像元), 幅度峰归一为 1; 圆盘之外为 0 (硬孔径)。
* **器件**: 面板每个像元带相位 ``gray_to_phase(gray)`` —— 即器件**本征**的
  灰度→相位律 ``depth_max_rad * (gray / gray_levels) ** gamma``。器件真值
  (``depth_max_rad`` / ``gamma``) **故意只写在配置里**, 分析侧拿不到, 拟合
  出来之后才与真值对账 —— 这样拟合链路才是被验证过的, 而不是被喂了答案。
* **传播**: 傅里叶透镜把瞳面 (pupil) 变成远场, 数值上就是一次二维 FFT。
* **探测器**: 暗电平 (加性背景 (杂散光) 项) + 读出噪声 + 满量程截断。

⚠️ **本文件不复刻驱动的 弧度→灰度 管线** (``Santec.create_phase_from_array``):
那个函数是**被测对象**的一部分 (波前矫正图 / 相位 LUT / 平移 都在里面), 用它
来生成仿真图案就等于把待测的链路一起"证明"了。这里刻意自包含: 只依赖
numpy / scipy / loguru, **不导入任何设备驱动**。

远场尺度约定 (ccd_px_per_sample)
---------------------------------
几何标定给出的 K 是 "面板每像元对应多少 CCD 像元" (CCD px / panel px)。
远场的频率采样间隔是 ``1 / (N * block)`` cycles per panel px, 其中

* ``N = region * padding`` —— 零填充后的 FFT 边长 (padding 只加密度, 不加
  视场, 所以频率间隔是 ``1 / (N * block)`` 而不是 ``1 / (region * block)``);
* ``block = 2 * pupil_radius_px / region`` —— 一个瞳面采样点覆盖多少面板像元。

于是 **1 cycle per panel px 对应 K 个 CCD 像元**, 远场相邻采样点的间距是::

    ccd_px_per_sample = K / (padding * 2 * pupil_radius_px)

默认配置: ``K = 7546.4``、``padding = 8``、``2R = 960`` ⇒ 0.98260 CCD px/采样点。
零级落在远场数组的正中心, 取一个以它为中心、边长
``round(crop_ccd_px / ccd_px_per_sample)`` 个采样点的窗口, 再重采样到像素间距
**恰好 1 CCD px** 的输出画幅 (边长 ``round(crop_ccd_px)``)。

复场子采样 (为什么要平均复数相量, 而不是平均相位)
---------------------------------------------------
瞳面采样点比面板像元粗 (``block`` 个面板像元合成 1 个瞳面采样点)。所以下采样
必须在**复场**上做: 每个面板像元带相位 ``φ(x, y)``, 一个采样点的值是它所覆盖
的 ``block × block`` 个像元的复数相量的**平均**::

    field = mean_over_block( exp(i * gray_to_phase(gray_block)) )

如果先对相位取算术平均再取指数, 被卷绕的相位会互相抵消 (每个像元都偏向 ~π),
结构被抹平; 先取指数再平均才是物理上正确的亚像元平均, 也让灰度→相位的
非线性表现得像真器件。

器件非理想 (本工具的"被测对象"本身就不完美)
------------------------------------------------
* **填充因子** ``fill_factor`` (0, 1]: 真实 LCoS 像元只有一部分面积能加相位,
  其余是死区 (固定相位 ``fill_background_rad``)。每个像元的复振幅于是是
  ``ff·exp(iφ) + (1-ff)·exp(iφ_bg)``。死区那一份是**常量** (没有空间调制),
  所以它只进零级 —— 与工具拟合里的**泄漏项** ℓ 是同一类东西 (未调制的光直漏
  零级)。硬边矩形填充因子还会给每阶乘一个 ``sinc(ff·m/P)`` 包络, 并把死区的光
  送进像元衍射包络 (中心瓣正好叠在零级上); 本模型把两者的零级贡献合并进混合项,
  是它的零阶近似。
* **相位 LUT** ``lut_table``: 逐灰度的 灰度→相位 表 (弧度)。给了它就取代幂律
  ``gamma`` —— 真实器件出厂标定出来的就是这种表, 幂律只是理想化。
* **面板静态相差** ``panel_zernike`` + ``roughness_sigma_rad``: 面板自身的
  Zernike 相差 (canonical Noll 序, 弧度) 与表面粗糙度 (固定种子的高斯随机场)。
* **离焦** ``defocus_coeff``: 光学离焦 (相机不在后焦面时)。本实验系统实测过
  这个坑: 离焦时光斑从 FWHM 15px 涨到 115px, 而效率积分 ROI 是按几何间距
  取的 —— 于是读数被截断, 效率系统性偏低。

面板 x 向 → CCD 行向 (轴向映射)
-------------------------------
本实验系统的 SLM 面板轴与传感器轴**交叉**: 面板的 x 向 (列) 在 CCD 上表现为
**行**方向 (行号变化), 面板的 y 向 (行) 表现为 CCD 的列向。几何标定里的 K 只给
"面板每像元对应多少 CCD 像元", **不给指向**。

⚠️ 本仓库以前的一个标定工具在算 ±1 级排除盘时, 把 x 光栅的**列**分量当成
列向半宽、y 光栅的**行**分量当成行向半宽 —— 两行正好互换, 排除盘小了约 50 倍,
±1 级永远落在 ROI 之外, 报出来的是"未通过"而不是真实偏差。所以这里的映射写成
显式的 ``panel_x_to_row`` 开关, 并在代码里用一次转置把它落实: 面板 x 光栅的
±1 级因此沿 **CCD 行** 排开 (只沿一个轴, 绝不沿对角线)。

噪声模型
--------
``frame = dark_level + peak_adu * intensity / reference + read_noise * randn``,
其中 ``reference`` 是**平相位 (全灰 0) 零级峰值强度**, 每个实例**只算一次**
(惰性), 因此扫描曲线上的通量下降是真实的 (深度越大, 光被推离零级, 零级变暗),
而不是被逐帧归一化抹掉。噪声由 ``np.random.default_rng(seed)`` 逐帧抽取,
序列确定可复现。帧最后截断到 ``[0, full_scale]``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.ndimage import map_coordinates, rotate

__all__ = ["SimConfig", "SimulatedBench"]


@dataclass(frozen=True)
class SimConfig:
    """数字孪生的全部参数。

    Attributes:
        region: 瞳面 (pupil) 采样边长 (采样点)。必须整除 ``2*pupil_radius_px``。
        panel_width_px: 面板宽度 (像元)。
        panel_height_px: 面板高度 (像元)。
        pupil_radius_px: 瞳面 (pupil) 半径 (面板像元), 圆盘以面板中心为中心。
        w0_px: 高斯入射束腰 (面板像元)。
        padding: FFT 零填充倍数 (只加远场采样密度, 不加视场)。
        crop_ccd_px: 输出 CCD 画幅边长 (CCD 像素)。
        kx: 几何标定 K (CCD px per panel px), 对应面板 x 光栅。
        ky: 几何标定 K, 对应面板 y 光栅。
        rotation_deg: 远场画面绕零级的旋转 (度)。
        panel_x_to_row: True = 面板 x 向在 CCD 上表现为行方向 (本实验系统的
            实测事实); False = 面板 x 向表现为 CCD 列方向。
        peak_adu: 平相位时零级峰值 (ADU)。
        full_scale: 探测器满量程 (ADU)。
        read_noise_adu: 读出噪声标准差 (ADU)。
        dark_level_adu: 暗电平 (加性背景 (杂散光) 项, ADU)。
        defocus_coeff: 瞳面二次相位系数 (rad / panel px^2); > 0 = 离焦。
        gray_levels: 器件灰度级数 (10 bit = 1023)。
        depth_max_rad: 器件真实 2π 相位深度 (真值, 分析侧不可见)。
        gamma: 器件 灰度→相位 幂律指数 (真值, 分析侧不可见)。
        fill_factor: 像元**有效面积占比** (0, 1]。小于 1 时每个像元只有
            ``fill_factor`` 的面积带调制相位, 余下是死区 (固定相位
            ``fill_background_rad``) —— 见模块 docstring "填充因子" 一节。
        fill_background_rad: 死区的相位 (弧度)。
        lut_table: 显式 灰度→相位 查找表 (弧度), 长度必须是
            ``gray_levels + 1``。给了它就**取代**幂律 ``gamma`` —— 这是真实
            器件 (出厂标定过的相位 LUT) 的样子, 幂律只是它的理想化。
        panel_zernike: 面板静态相差 (Noll 序 → 系数, 弧度)。
        panel_zernike_n_max: 面板静态相差的最大径向阶数。
        roughness_sigma_rad: 面板粗糙度 RMS (弧度)。
        roughness_seed: 粗糙度随机场的种子 (固定, 可复现)。
        seed: 随机数种子 (噪声序列)。
    """

    region: int = 240
    panel_width_px: int = 1920
    panel_height_px: int = 1200
    pupil_radius_px: int = 480
    w0_px: float = 240.0
    padding: int = 8
    crop_ccd_px: float = 1400.0
    kx: float = 7546.4
    ky: float = 7495.9
    rotation_deg: float = 0.12
    panel_x_to_row: bool = True
    peak_adu: float = 18000.0
    full_scale: float = 65535.0
    read_noise_adu: float = 25.0
    dark_level_adu: float = 20.0
    defocus_coeff: float = 0.0
    gray_levels: int = 1023
    depth_max_rad: float = 6.283185307179586
    gamma: float = 1.0
    fill_factor: float = 1.0
    fill_background_rad: float = 0.0
    lut_table: tuple[float, ...] | None = None
    panel_zernike: dict[int, float] | None = None
    panel_zernike_n_max: int = 7
    roughness_sigma_rad: float = 0.0
    roughness_seed: int = 0
    seed: int = 0


class SimulatedBench:
    """2f 实验系统的数字孪生: 面板灰度 → 远场 CCD 帧。

    用法::

        bench = SimulatedBench(SimConfig())
        gray = ...                       # 面板灰度 (uint16, panel_shape)
        frame = bench.capture(gray)      # (frame_shape) float64, 带噪声

    同一份代码既能喂 ``core.blaze_gray`` 生成的图案, 也能喂任意自定义图案;
    真实器件的灰度→相位律 (``depth_max_rad`` / ``gamma``) 藏在配置里, 分析侧
    只能从衍射效率反推。
    """

    def __init__(self, config: SimConfig) -> None:
        """按配置建立数字孪生, 并校验几何自洽性。

        Args:
            config: 数字孪生参数。

        Raises:
            ValueError: 配置不自洽 (采样格不整除面板、面板装不下瞳面、深度
                超过 2π 等)。
        """
        cfg = config
        region = int(cfg.region)
        padding = int(cfg.padding)
        radius = int(cfg.pupil_radius_px)
        panel_h = int(cfg.panel_height_px)
        panel_w = int(cfg.panel_width_px)

        if region < 1:
            raise ValueError(f"region 必须 >= 1, 得到 {region}")
        if padding < 1:
            raise ValueError(f"padding 必须 >= 1, 得到 {padding}")
        if radius < 1:
            raise ValueError(f"pupil_radius_px 必须 >= 1, 得到 {radius}")
        if (2 * radius) % region != 0:
            raise ValueError(
                f"2*pupil_radius_px 必须能被 region 整除, "
                f"得到 2*{radius}={2 * radius} 与 region={region}"
            )
        if int(cfg.gray_levels) < 1:
            raise ValueError(f"gray_levels 必须 >= 1, 得到 {cfg.gray_levels}")
        depth = float(cfg.depth_max_rad)
        if depth < 0.0 or depth > 2.0 * math.pi + 1e-9:
            raise ValueError(
                f"depth_max_rad 必须在 [0, 2π] 内 (配置不做 mod 2π 包裹), 得到 {depth}"
            )
        if float(cfg.gamma) <= 0.0:
            raise ValueError(f"gamma 必须为正, 得到 {cfg.gamma}")
        if float(cfg.w0_px) <= 0.0:
            raise ValueError(f"w0_px 必须为正, 得到 {cfg.w0_px}")
        if panel_w < 2 * radius or panel_h < 2 * radius:
            raise ValueError(
                f"面板 ({panel_w}x{panel_h}) 装不下半径 {radius} 的瞳面 (pupil) 圆盘"
            )
        if float(cfg.crop_ccd_px) < 8.0:
            raise ValueError(f"crop_ccd_px 必须 >= 8, 得到 {cfg.crop_ccd_px}")
        if not 0.0 < float(cfg.fill_factor) <= 1.0:
            raise ValueError(
                f"fill_factor 必须在 (0, 1] 内, 得到 {cfg.fill_factor}"
            )
        if float(cfg.roughness_sigma_rad) < 0.0:
            raise ValueError(
                f"roughness_sigma_rad 不能为负, 得到 {cfg.roughness_sigma_rad}"
            )
        lut = cfg.lut_table
        if lut is not None:
            table = np.asarray(lut, dtype=np.float64)
            if table.ndim != 1 or table.shape[0] != int(cfg.gray_levels) + 1:
                raise ValueError(
                    "lut_table 必须是长度 gray_levels+1 的一维数组, 得到 "
                    f"shape={table.shape} (gray_levels={cfg.gray_levels})"
                )
            if not np.all(np.isfinite(table)):
                raise ValueError("lut_table 含非有限值")
            if float(table.min()) < 0.0 or float(table.max()) > 2.0 * math.pi + 1e-9:
                raise ValueError(
                    "lut_table 的相位必须在 [0, 2π] 内, 得到 "
                    f"[{table.min()}, {table.max()}]"
                )

        k = float(cfg.kx) if bool(cfg.panel_x_to_row) else float(cfg.ky)
        if k <= 0.0:
            raise ValueError(f"几何标定 K 必须为正, 得到 {k}")

        self._config = cfg
        self._region = region
        self._padding = padding
        self._radius = radius
        self._block = (2 * radius) // region
        self._panel_shape = (panel_h, panel_w)
        self._w0 = float(cfg.w0_px)
        self._defocus_coeff = float(cfg.defocus_coeff)
        self._gray_levels = int(cfg.gray_levels)
        self._depth_max_rad = depth
        self._gamma = float(cfg.gamma)
        self._peak_adu = float(cfg.peak_adu)
        self._full_scale = float(cfg.full_scale)
        self._read_noise = float(cfg.read_noise_adu)
        self._dark_level = float(cfg.dark_level_adu)
        self._rotation_deg = float(cfg.rotation_deg)
        self._panel_x_to_row = bool(cfg.panel_x_to_row)
        self._k = k
        self._fill_factor = float(cfg.fill_factor)
        self._fill_background = np.exp(1j * float(cfg.fill_background_rad))
        self._lut_table = (
            None
            if cfg.lut_table is None
            else np.asarray(cfg.lut_table, dtype=np.float64)
        )
        self._panel_zernike = (
            None if cfg.panel_zernike is None else dict(cfg.panel_zernike)
        )
        self._panel_zernike_n_max = int(cfg.panel_zernike_n_max)
        self._roughness_sigma = float(cfg.roughness_sigma_rad)
        self._roughness_seed = int(cfg.roughness_seed)

        # 瞳面 (pupil) 圆盘以面板中心为中心, 取其外接方框作为裁剪窗口。
        self._crop_row0 = int(round(panel_h / 2.0 - radius))
        self._crop_col0 = int(round(panel_w / 2.0 - radius))
        if (
            self._crop_row0 < 0
            or self._crop_row0 + 2 * radius > panel_h
            or self._crop_col0 < 0
            or self._crop_col0 + 2 * radius > panel_w
        ):
            raise ValueError(
                f"瞳面 (pupil) 圆盘中心 ({panel_w / 2.0}, {panel_h / 2.0}) "
                f"半径 {radius} 越出面板 ({panel_w}x{panel_h})"
            )

        # 远场几何: N = region*padding 的零填充 FFT, 频率间隔 1/(N*block) cycles/panel px。
        self._far_field_size = region * padding
        self._ccd_px_per_sample = k / (padding * (2 * radius))
        self._crop_samples = int(round(float(cfg.crop_ccd_px) / self._ccd_px_per_sample))
        if self._crop_samples < 8:
            raise ValueError(
                f"远场裁剪窗口至少要 8 个采样点, 得到 {self._crop_samples} "
                f"(crop_ccd_px={cfg.crop_ccd_px}, ccd_px_per_sample="
                f"{self._ccd_px_per_sample:.6f})"
            )
        if self._crop_samples > self._far_field_size:
            raise ValueError(
                f"远场裁剪窗口 {self._crop_samples} 超出 FFT 画幅 "
                f"{self._far_field_size}, 请减小 crop_ccd_px 或增大 padding"
            )
        self._crop_start = self._far_field_size // 2 - self._crop_samples // 2
        self._frame_side = int(round(float(cfg.crop_ccd_px)))
        if self._frame_side < 8:
            raise ValueError(f"输出画幅边长必须 >= 8, 得到 {self._frame_side}")

        self._rng = np.random.default_rng(int(cfg.seed))
        self._amplitude: np.ndarray | None = None
        self._reference_peak: float | None = None
        self._aberration: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------

    @property
    def config(self) -> SimConfig:
        """原始配置。"""
        return self._config

    @property
    def panel_shape(self) -> tuple[int, int]:
        """面板尺寸 ``(height, width)`` (像元)。"""
        return self._panel_shape

    @property
    def frame_shape(self) -> tuple[int, int]:
        """输出 CCD 帧尺寸 ``(height, width)`` (像元)。"""
        return (self._frame_side, self._frame_side)

    @property
    def ccd_px_per_sample(self) -> float:
        """远场相邻采样点对应的 CCD 像素间距。

        ``K / (padding * 2 * pupil_radius_px)``: 远场频率间隔是
        ``1 / (N * block)`` cycles per panel px, 而 1 cycle per panel px 对应
        ``K`` 个 CCD 像元。
        """
        return self._ccd_px_per_sample

    # ------------------------------------------------------------------
    # 器件模型
    # ------------------------------------------------------------------

    def gray_to_phase(self, gray: np.ndarray) -> np.ndarray:
        """器件本征灰度→相位 (弧度), 逐元素。

        两种器件模型, 由配置决定:

        * **幂律** (默认): ``phase = depth_max_rad * (gray/gray_levels) ** gamma``。
        * **查找表** (``lut_table`` 给了): 逐灰度查表, 值就是相位 (弧度)。
          真实器件出厂标定出来的相位 LUT 就是这个样子 —— 幂律只是它的理想化,
          而"理想化能不能代替真表"正是本工具要回答的问题之一。

        注意: **不做 mod 2π 包裹** —— 配置要求 ``depth_max_rad <= 2π``, 器件
        的相位包裹行为属于被测量的对象, 不是这里可以替它做主的。

        Args:
            gray: 灰度值 (任意形状, 会被截断到 ``[0, gray_levels]``)。

        Returns:
            弧度相位 (float64, 与输入同形)。
        """
        values = np.clip(np.asarray(gray, dtype=np.float64), 0.0, float(self._gray_levels))
        if self._lut_table is not None:
            index = np.rint(values).astype(np.int64)
            return self._lut_table[index]
        return self._depth_max_rad * np.power(values / float(self._gray_levels), self._gamma)

    # ------------------------------------------------------------------
    # 远场
    # ------------------------------------------------------------------

    def far_field(self, phase_rad: np.ndarray) -> np.ndarray:
        """瞳面相位 → 无噪远场强度。

        Args:
            phase_rad: 瞳面 (pupil) 网格上的相位 (弧度, ``region × region``)。

        Returns:
            输出画幅的远场强度 (float64, ``frame_shape``)。

        Raises:
            ValueError: 相位图尺寸不是 ``region × region``。
        """
        phase = np.asarray(phase_rad, dtype=np.float64)
        if phase.shape != (self._region, self._region):
            raise ValueError(
                f"瞳面相位尺寸必须是 ({self._region}, {self._region}), 得到 {phase.shape}"
            )
        pupil_field = self._fill_factor_mix(np.exp(1j * phase))
        field = self._pupil_amplitude() * pupil_field * np.exp(
            1j * (self._defocus_phase() + self._panel_aberration_phase())
        )
        return self._far_field_array(field)

    def capture(self, gray: np.ndarray) -> np.ndarray:
        """下发面板灰度 → 带噪声的 CCD 帧。

        Args:
            gray: 面板灰度图 (``panel_shape``, uint16 或浮点)。

        Returns:
            CCD 帧 (float64, ``frame_shape``), 已截断到 ``[0, full_scale]``。

        Raises:
            ValueError: 灰度图尺寸与 ``panel_shape`` 不符。
        """
        array = np.asarray(gray)
        if array.shape != self._panel_shape:
            raise ValueError(
                f"灰度图尺寸必须是 {self._panel_shape}, 得到 {array.shape}"
            )
        pupil_field = self._pupil_complex_field(array)
        intensity = self._far_field_array(pupil_field)
        reference = self._reference_peak_value()
        signal = self._peak_adu * intensity / reference
        noise = self._read_noise * self._rng.standard_normal(
            (self._frame_side, self._frame_side)
        )
        frame = self._dark_level + signal + noise
        return np.clip(frame, 0.0, self._full_scale)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _pupil_offsets(self) -> np.ndarray:
        """瞳面采样点相对瞳面中心的偏移 (面板像元), 长度 ``region``。

        采样点取每个 ``block`` 块的中心: ``-R + (i + 0.5) * block``。
        """
        index = np.arange(self._region, dtype=np.float64)
        return (index + 0.5) * self._block - self._radius

    def _pupil_amplitude(self) -> np.ndarray:
        """瞳面幅度: 高斯束 × 圆形硬孔径, 峰归一为 1 (``region × region``)。"""
        if self._amplitude is None:
            offsets = self._pupil_offsets()
            dy = offsets[:, None]
            dx = offsets[None, :]
            r2 = dx * dx + dy * dy
            amplitude = np.exp(-r2 / (2.0 * self._w0 * self._w0))
            amplitude[r2 > self._radius * self._radius] = 0.0
            peak = float(amplitude.max())
            self._amplitude = amplitude / peak if peak > 0.0 else amplitude
        return self._amplitude

    def _defocus_phase(self) -> np.ndarray:
        """瞳面二次相位 (rad / panel px^2 × r²), ``region × region``。"""
        offsets = self._pupil_offsets()
        dy = offsets[:, None]
        dx = offsets[None, :]
        r2 = dx * dx + dy * dy
        return self._defocus_coeff * r2

    def _fill_factor_mix(self, phasors: np.ndarray) -> np.ndarray:
        """按填充因子把"调制区"与"死区"的复场混合。

        真实 LCoS 像元只有一部分面积能加相位, 其余是死区 (固定相位反射)。
        于是每个像元的复振幅是

            ff·exp(iφ) + (1-ff)·exp(iφ_bg)

        死区那一份是**常量** (没有空间调制), 所以它只进零级 —— 也就是
        "未调制的光直漏零级", 与工具拟合里的**泄漏项** ℓ 是同一类东西。
        真实硬边矩形填充因子还会给每一阶乘一个 ``sinc(ff·m/P)`` 包络, 并把
        死区的光送进像元衍射包络 (其中心瓣正好叠在零级上)。本模型把两者的
        零级贡献合并进这个混合项, 是它的零阶近似。
        """
        if self._fill_factor >= 1.0:
            return phasors
        return self._fill_factor * phasors + (1.0 - self._fill_factor) * (
            self._fill_background
        )

    def _panel_aberration_phase(self) -> np.ndarray:
        """面板静态相差 (Zernike + 粗糙度), ``region × region`` (弧度)。

        Zernike 用仓库 canonical 入口
        :func:`ao_shaping.utils.wavefront.zernike_utils.generate_zernike_phase`
        (Noll 序, 系数单位弧度, 圆孔外给 NaN → 这里置 0); 粗糙度是固定种子的
        高斯随机场, RMS = ``roughness_sigma_rad``。
        """
        if self._aberration is not None:
            return self._aberration
        phase = np.zeros((self._region, self._region), dtype=np.float64)
        if self._panel_zernike:
            from ao_shaping.utils.wavefront.zernike_utils import (
                generate_zernike_phase,
            )

            raw = generate_zernike_phase(
                self._panel_zernike,
                resolution=(self._region, self._region),
                n_max=self._panel_zernike_n_max,
                radius=self._region / 2.0,
            )
            phase += np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
        if self._roughness_sigma > 0.0:
            rng = np.random.default_rng(self._roughness_seed)
            field = rng.normal(0.0, 1.0, size=(self._region, self._region))
            rms = float(np.sqrt(np.mean(field**2)))
            phase += self._roughness_sigma * field / rms
        self._aberration = phase
        return phase

    def _pupil_complex_field(self, gray: np.ndarray) -> np.ndarray:
        """面板灰度 → 瞳面复场 (亚像元复数平均 + 填充因子 + 离焦 + 面板相差)。

        先把瞳面 (pupil) 窗口从面板上裁下来, 再把每个 ``block × block`` 面板像元
        块的复数相量取平均 —— 见模块 docstring "复场子采样" 一节; 之后再叠上
        填充因子混合、离焦与面板静态相差。
        """
        size = 2 * self._radius
        window = np.asarray(gray, dtype=np.float64)[
            self._crop_row0 : self._crop_row0 + size,
            self._crop_col0 : self._crop_col0 + size,
        ]
        phase = self.gray_to_phase(window)
        block = self._block
        reshaped = phase.reshape(self._region, block, self._region, block)
        phasors = np.exp(1j * reshaped).mean(axis=(1, 3))
        pupil_field = self._fill_factor_mix(phasors)
        return (
            self._pupil_amplitude()
            * pupil_field
            * np.exp(1j * (self._defocus_phase() + self._panel_aberration_phase()))
        )

    def _far_field_array(self, pupil_field: np.ndarray) -> np.ndarray:
        """瞳面复场 → 输出画幅的远场强度 (无噪)。

        流程: 零填充 FFT → fftshift → (必要时) 转置实现轴向映射 → 以零级为中心
        裁剪 → 重采样到 1 CCD px 像素间距 → 按 ``rotation_deg`` 绕中心旋转。
        """
        size = self._far_field_size
        spectrum = np.fft.fftshift(np.fft.fft2(pupil_field, s=(size, size)))
        if self._panel_x_to_row:
            # 面板 x 向 (列) 在 CCD 上表现为行方向: FFT 输出的行 = 面板 y、
            # 列 = 面板 x, 转置后行 = 面板 x, 于是 x 光栅的 ±1 级沿 CCD 行排开。
            spectrum = spectrum.T
        start = self._crop_start
        window = spectrum[
            start : start + self._crop_samples, start : start + self._crop_samples
        ]
        intensity = np.abs(window) ** 2

        side = self._frame_side
        axis = np.arange(side, dtype=np.float64)
        # 输出像素 j → 远场采样坐标; 零级正好落在 j = side//2 上。
        position = (axis - side // 2) / self._ccd_px_per_sample + self._crop_samples // 2
        rows, cols = np.meshgrid(position, position, indexing="ij")
        frame = map_coordinates(
            intensity, (rows, cols), order=1, mode="constant", cval=0.0
        )
        if abs(self._rotation_deg) > 0.0:
            # 旋转的是**画面**, 因此面板坐标轴在画面上看起来转过了相反的角度。
            frame = rotate(
                frame,
                self._rotation_deg,
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
            )
        return np.asarray(frame, dtype=np.float64)

    def _reference_peak_value(self) -> float:
        """平相位 (全灰 0) 零级峰值强度 —— 惰性计算, 每实例一次。"""
        if self._reference_peak is None:
            flat = self._pupil_amplitude() * np.exp(1j * self._defocus_phase())
            size = self._far_field_size
            spectrum = np.fft.fftshift(np.fft.fft2(flat, s=(size, size)))
            self._reference_peak = float(np.abs(spectrum).max() ** 2)
            logger.debug(
                "平相位零级峰值强度 = {:.6e}, ccd_px_per_sample = {:.6f}".format(
                    self._reference_peak, self._ccd_px_per_sample
                )
            )
        return self._reference_peak
