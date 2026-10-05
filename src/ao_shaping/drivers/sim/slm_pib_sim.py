"""用于无硬件跑 ``slm-pib`` 的模拟 2f 傅里叶 SLM + CCD。

``slm-pib`` 优化器 (``optimizer/wfless/slm_zernike_pib.py``) 驱动一台真实的 ``Santec``
SLM, 方式是 ``create_phase_from_array(phase_rad) -> uint16`` 与 ``display_data(gray, ...)``,
并从一台 CCD 上读取远场光斑。本模块提供的是一个纯 numpy 的替身, 它复现该契约的同时
真的传播光, 于是 SPGD 搜索有了一个*真实*的光学反馈环路:

* :class:`SimSLMPib` —— 接受 raw 弧度相位、把它存下来, 并在显示时计算被该相位调制的
  高斯入射光束的夫琅禾费远场 (瞳孔场的 ``np.fft.fft2``, 居中)。0 级光斑落在画面正中,
  其形状对所施加相位的响应与一台 2f 台架完全一致。
* :class:`SimPibCCD` —— 一个 ``BaseCamera``, 其 ``get_numpy_image`` 直接返回仿真 SLM
  最近一次产出的远场图像 (再加一点类 Poisson 的散粒噪声地板, 使指标不会完全光滑)。

两者经 :class:`SimSLMPibSystem` 连在一起; 它同时在相机注册表里注册一个 ``"sim"``
条目, 使 ``create_camera("sim", ...)`` 返回一个绑定到共享 system 的 ``SimPibCCD``。

这有意是一个*远场 FFT* 模型 (即 ``AGENTS.md`` 里的 2f 傅里叶台架), 而不是近场传播器:
SLM 在前焦面、CCD 在后焦面, 因此 CCD 图像就是 SLM 瞳孔的二维 FFT。这恰好是
``slm-pib`` 所工作的机制。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.device_base import DeviceState
from ao_shaping.drivers.sim.disturbance import SimDisturbance
from ao_shaping.drivers.sim.dm_optics import SimDmOptics
from ao_shaping.drivers.slm.santec.slm200_constants import GRAY_SCALE_BITS

try:  # scipy 的 pocketfft 是多线程的, 而补零后的变换需要它。
    from scipy import fft as _scipy_fft
except ImportError:  # pragma: no cover - numpy 总是存在
    _scipy_fft = None

# SLM 面板几何 (与真实 Santec SLM-200 一致: 1920 x 1200, 10-bit)。
SLM_W = 1920
SLM_H = 1200
SLM_BITS = 10
SLM_MAX_GRAY = (1 << SLM_BITS) - 1  # 1023
TWO_PI_GRAY = 993  # 1064 nm 下的 2*pi 灰度 (设备动态值, 见 AGENTS.md)

# 入射高斯光束半径 (SLM 像素) —— 即打在 SLM 上的平场光束的光腰。取值使光束能宽裕地
# 填满面板口径。
BEAM_W0 = 400.0

# CCD 远场分辨率 (正方形)。0 级光斑落在画面正中。
CCD_RES = (512, 512)

# 在 ``fft2`` 之前施加到瞳孔上的补零因子。远场像元间距由该变换固定, 因此不补零的 FFT
# 会把 0 级光斑放到 ~1.8 px FWHM —— 横跨它的采样几乎只有一个, 于是每个功率比指标都要
# 去除一个在数值上根本没被分辨的峰值。补零把这个间距按该因子缩小, 这在数字上等价于
# 真实台架通过缩小相机 ROI 获得的更长有效焦距 (裁剪*同一*数组只会放大它, 不增加任何
# 采样)。取 4 时光斑横跨 ~7 px。
FAR_FIELD_PADDING = 4

# 补零 FFT 之后保留的居中窗口的边长, 单位远场像素。它同时限定了缓存数组以及
# ``far_field_noisy`` 施加在上面的散粒/读出噪声; 必须 >= 相机窗口 (``cam_size``)。
FAR_FIELD_WINDOW = 1024


def _forward_fft2(field: np.ndarray) -> np.ndarray:
    """二维正向 FFT, 可用时走 scipy 的多线程核。"""
    if _scipy_fft is not None:
        return _scipy_fft.fft2(field, workers=-1)
    return np.fft.fft2(field)


class SimPibSystem:
    """仿真 SLM 与仿真 CCD 之间共享的光学状态。

    持有*当前显示的*弧度相位与惰性计算的远场。SLM (写方) 与 CCD (读方) 共用同一个
    实例, 因此相机总能看到的正是 SLM 最后显示的那束光。
    """

    def __init__(
        self,
        slm_shape: tuple[int, int] = (SLM_H, SLM_W),
        ccd_res: tuple[int, int] = CCD_RES,
        beam_w0: float = BEAM_W0,
        noise_adu: float = 5.0,
        seed: int | None = None,
        *,
        disturbance: SimDisturbance | None = None,
        dm_optics: SimDmOptics | None = None,
        far_field_padding: int = FAR_FIELD_PADDING,
        far_field_window: int = FAR_FIELD_WINDOW,
    ) -> None:
        """构造共享的光学状态。

        Args:
            slm_shape: SLM 面板形状 ``(h, w)``。
            ccd_res: 远场分辨率。
            beam_w0: 高斯入射光束腰, 单位 SLM 像素。
            noise_adu: :meth:`far_field_noisy` 加入的散粒/读出噪声地板。
            seed: 噪声模型的 RNG 种子。
            disturbance: 可选的湍流 + 热晕模型。仅关键字传入且默认 ``None``, 这样
                每个现有调用方都保持原先无干扰的行为。它的相位屏在 :meth:`far_field`
                内部被加到 SLM 命令相位上; ``self._phase`` 从不被修改, 因此干扰不会
                被重复计入。
            dm_optics: 可选的 DM 电压 → 相位耦合。与干扰一样, 它在求值时被累加,
                从不烘进 ``self._phase``。它默认是一个平的、零电压的实例而非 ``None``,
                于是 DM 驱动的 runner 在构造上就已耦合, 不会静默退化成什么都没驱动。
        """
        # 延迟 import: config.DM_N_ACTUATORS 是经一次活的 DM 可达性探测解析的,
        # 所以在模块作用域 import 它会阻塞在 socket 上。
        from ao_shaping.config import DM_N_ACTUATORS

        self.slm_h, self.slm_w = slm_shape
        self.ccd_h, self.ccd_w = ccd_res
        self.beam_w0 = float(beam_w0)
        self.noise_adu = float(noise_adu)
        self.disturbance = disturbance
        self.far_field_padding = max(1, int(far_field_padding))
        self.far_field_window = max(1, int(far_field_window))
        self.dm_optics = dm_optics or SimDmOptics(
            n_actuators=DM_N_ACTUATORS, slm_shape=slm_shape
        )
        self._dm_version = self.dm_optics.version
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._phase: np.ndarray = np.zeros((self.slm_h, self.slm_w), dtype=np.float64)
        self._far_field: np.ndarray | None = None
        self._pad_buffer: np.ndarray | None = None
        self._gray: np.ndarray | None = None

    # --- SLM 侧 --------------------------------------------------------

    def set_phase_rad(self, phase_rad: np.ndarray) -> None:
        """保存一份 raw 弧度相位, 并把远场标记为过期。"""
        phase = np.asarray(phase_rad, dtype=np.float64)
        if phase.shape != (self.slm_h, self.slm_w):
            raise ValueError(
                f"phase shape {phase.shape} != SLM panel {(self.slm_h, self.slm_w)}"
            )
        with self._lock:
            self._phase = phase
            self._far_field = None  # 直到下次显示前都是过期的

    def set_gray(self, gray: np.ndarray) -> None:
        """保存原始灰度图样 (平场灰度路径, 不做弧度换算)。"""
        g = np.asarray(gray, dtype=np.uint16)
        if g.shape != (self.slm_h, self.slm_w):
            raise ValueError(f"gray shape {g.shape} != SLM panel {(self.slm_h, self.slm_w)}")
        with self._lock:
            self._gray = g
            self._far_field = None

    # --- CCD 侧 --------------------------------------------------------

    def far_field(self) -> np.ndarray:
        """计算 (并缓存) 所显示相位的远场图像。

        瞳孔场是 ``A(r) * exp(i*phase)``, 其中 ``A`` 是高斯入射光束; 远场是
        ``|FFT(pupil)|^2``, 并把零频分量移到中心 (``fftshift``)。0 级光斑即画面中心,
        与 ``AGENTS.md`` 里的 2f 台架约定一致。
        """
        with self._lock:
            # DM 是一个独立的光学状态, 所以电压变化必须让缓存失效。没有这一步,
            # 缓存会一直供应 DM 作用之前的那张图, 于是 DM 驱动的环路读起来就像
            # DM 什么都没做。
            if self.dm_optics.version != self._dm_version:
                self._far_field = None
                self._dm_version = self.dm_optics.version
            if self._far_field is not None:
                return self._far_field.copy()
            phase = self._phase
            # 干扰与 SLM 命令共处同一个瞳面, 所以它在傅里叶变换之前就被加进去。
            # 上面的提前返回保证每次真实光学求值只消耗它一次。`self._phase` 保持为
            # 纯命令 —— 把干扰烘进去会被重复计入。
            if self.disturbance is not None:
                phase = phase + self.disturbance.phase()
            # DM 同理: 在此累加, 从不烘进 `self._phase`。
            dm_phase = self.dm_optics.phase()
            if dm_phase.any():
                phase = phase + dm_phase
            # SLM 网格上的高斯入射光束幅度。
            yy, xx = np.mgrid[0 : self.slm_h, 0 : self.slm_w]
            cy, cx = self.slm_h / 2.0, self.slm_w / 2.0
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            amplitude = np.exp(-r2 / (2.0 * self.beam_w0**2))
            pupil = amplitude * np.exp(1j * phase)
            pad = self.far_field_padding
            if pad > 1:
                shape = (self.slm_h * pad, self.slm_w * pad)
                buffer = self._pad_buffer
                if buffer is None or buffer.shape != shape:
                    buffer = np.zeros(shape, dtype=np.complex128)
                    self._pad_buffer = buffer
                buffer[: self.slm_h, : self.slm_w] = pupil
                padded = buffer
            else:
                padded = pupil
            spectrum = np.fft.fftshift(np.abs(_forward_fft2(padded)) ** 2)
            window = min(self.far_field_window, spectrum.shape[0], spectrum.shape[1])
            y0 = (spectrum.shape[0] - window) // 2
            x0 = (spectrum.shape[1] - window) // 2
            spectrum = spectrum[y0 : y0 + window, x0 : x0 + window]
            # 归一化到 ~0..255 的灰度范围, 使曝光/峰值逻辑的行为像真实相机
            # (平场光束下峰值 ~100)。
            peak = float(spectrum.max())
            if peak > 0:
                spectrum = spectrum * (100.0 / peak)
            self._far_field = spectrum.astype(np.float64)
            return self._far_field.copy()

    def far_field_noisy(self) -> np.ndarray:
        """远场加上散粒噪声 + 读出噪声, 以原始 ADU 帧返回。

        结果刻意**不**在零处裁剪。``far_field`` 是 ``|FFT|**2``, 本身已非负, 因此这里
        每一个负值都只可能来自读出噪声。在零处裁剪会把这种对称噪声整流成一个与像素数
        成正比的 DC 地板: 光斑在一幅 1200x1920 的画面里只占 ~150 px 信号, 而该地板实测
        约为信号的 ~3000x。于是每个功率比指标除的都是那个地板而不是光, 这正是
        ``pib``/``combined`` 停在 ~0.0145、其 SPGD 梯度估计全是噪声的原因。

        真实传感器的地板位于其裁剪阈值*之下*, 只在量化时被裁剪一次, 因此原始帧可以
        略低于黑电平; 去掉该偏移靠的是暗帧扣除, 而不是逐帧 clip。
        """
        img = self.far_field()
        if self.noise_adu > 0:
            shot = self._rng.poisson(np.clip(img, 0, None).astype(np.float64) / 10.0)
            img = img + (shot - img / 10.0) * (self.noise_adu / 10.0)
            img = img + self._rng.normal(0.0, self.noise_adu / 10.0, img.shape)
        return img


# 一个进程级的 system, 让注册表创建的 CCD 与被 monkeypatch 的 SLM 共享同一份光学状态。
_SYSTEM: SimPibSystem | None = None


def get_system(
    seed: int | None = None, *, disturbance: SimDisturbance | None = None
) -> SimPibSystem:
    """返回进程级的 :class:`SimPibSystem` (首次使用时创建)。

    ``disturbance`` 只在 system 于此处被创建时才生效; 已存在的 system 原样返回
    (用 :func:`reset_system` 来替换它)。
    """
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = SimPibSystem(seed=seed, disturbance=disturbance)
    return _SYSTEM


def reset_system(
    seed: int | None = None, *, disturbance: SimDisturbance | None = None
) -> SimPibSystem:
    """替换进程级的 system (用于矩阵单元格之间)。"""
    global _SYSTEM
    _SYSTEM = SimPibSystem(seed=seed, disturbance=disturbance)
    return _SYSTEM


class SimSLMPib:
    """模拟 Santec SLM, 暴露出优化器实际使用的那个精确契约。

    只实现 ``slm_zernike_pib`` 真正会调用的那些方法:
    ``create_phase_from_array``、``display_data``、``set_grayscale``、``open``、
    ``close``、``is_connected``、``__enter__``/``__exit__``。
    """

    #: 镜像 ``Santec.Gray_Scale_bits``; ``optimize_slm_square`` 在给它的
    #: ``PatternHelper`` 定尺寸时会读它。缺了它, ``spgd-square`` 的仿真运行会在
    #: 中途抛 ``AttributeError``。
    Gray_Scale_bits: int = GRAY_SCALE_BITS

    def __init__(self, *args: Any, system: SimPibSystem | None = None, **kwargs: Any) -> None:
        self.system = system or get_system()
        self._open = False
        # 跟踪最后显示的弧度相位, 供报告使用。
        self.last_phase_rad: np.ndarray | None = None

    @classmethod
    def from_params(cls, params: Any, **overrides: Any) -> "SimSLMPib":
        """从驱动参数对象构造 (dataclass API 对齐)。

        ``optimize_slm_zernike_pib`` 是通过 ``Santec.from_params(config.slm)`` 而非
        ``Santec(...)`` 构造 SLM 的, 所以这个仿真替身必须暴露同一个 classmethod, 否则
        仿真 CLI 路径会抛 ``AttributeError: type object 'SimSLMPib' has no attribute
        'from_params'``。这里镜像真实驱动中那种"``getattr`` 带默认值"的取值方式;
        ``slm_number`` 被记录下来只为对齐/调试, 仿真并无物理面板身份。
        """
        kwargs: dict[str, Any] = {
            "slm_number": getattr(params, "slm_number", 1),
            "wavelength": getattr(
                params, "slm_wavelength", getattr(params, "wavelength", None)
            ),
            "shift_x": getattr(params, "shift_x", None),
            "shift_y": getattr(params, "shift_y", None),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # --- 上下文管理器 / 生命周期 -------------------------------------

    def open(self) -> None:
        self._open = True
        logger.debug("SimSLMPib opened (no hardware)")

    def close(self) -> None:
        self._open = False

    def is_connected(self) -> bool:
        return self._open

    def __enter__(self) -> "SimSLMPib":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- 相位 / 灰度 API (对齐 Santec 契约) --------------------------

    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        """弧度相位 → uint16 灰度, 并同时给光学系统上膛。

        在真实 Santec 上这一步只做弧度→灰度换算 (2*pi = ~993 灰度), 在
        ``display_data`` 之前不动面板。而这里额外把弧度相位存下来, 使下一次
        ``far_field()`` 会反映它。返回的灰度就是弧度→灰度的映射 (wrap 后缩放到 1023)。
        """
        rad = np.asarray(phase_rad, dtype=np.float64)
        # 存下原始弧度相位, 供远场使用。
        self.system.set_phase_rad(rad)
        self.last_phase_rad = rad.copy()
        # 弧度 → 灰度 (2*pi 映射到 TWO_PI_GRAY), wrap 方式与驱动一致。
        gray = (np.mod(rad, 2.0 * np.pi) / (2.0 * np.pi) * TWO_PI_GRAY)
        return gray.astype(np.uint16)

    def display_data(
        self,
        gray: np.ndarray,
        memory_number: int | None = None,
        memory_mode: int | None = None,
    ) -> int:
        """显示一幅灰度图样。

        优化器总是先调用 ``create_phase_from_array`` (它给相位上膛) 再调
        ``display_data``, 所以到这里时远场已经是对的。一次裸的灰度显示 (前面没有相位)
        被当作平场相位图样处理: 它的平均灰度映射到一个常数相位偏移, 而后者不改变远场的
        *形状* (只改变 0 级耦合), 所以我们把存下的相位原样保留。
        """
        self._open = True
        # 强制下一次 CCD 读取时 (重新) 计算远场。
        self.system.set_gray(np.asarray(gray, dtype=np.uint16))
        return memory_number if memory_number is not None else 0

    def set_grayscale(self, gray: int | np.ndarray) -> None:
        """设置均匀灰度 (退出时用来把面板涂黑)。"""
        if isinstance(gray, (int, float)):
            arr = np.full((self.system.slm_h, self.system.slm_w), int(gray), dtype=np.uint16)
        else:
            arr = np.asarray(gray, dtype=np.uint16)
        self.system.set_gray(arr)


class SimPibCCD(BaseCamera):
    """返回仿真 SLM 远场的模拟 CCD。

    接到进程级的 :class:`SimPibSystem` 上。注册为相机类型 ``"sim"``, 因此
    ``create_camera("sim", ...)`` 会产出本类的实例。
    """

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 80.0,
        cam_size: int = 250,
        skip_sampling: bool = False,
        system: SimPibSystem | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=skip_sampling)
        self.cam_size = int(cam_size)
        self.system = system or get_system()
        self._open = False

    # --- BaseCamera 接口 -------------------------------------------

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def is_connected(self) -> bool:
        return self._open

    def initialize(self) -> None:
        self._open = True

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        """返回当前远场 (含噪声), 裁剪到 ``cam_size``。

        优化器把 CCD 在光斑周围开窗到 ``cam_size``; 这里返回完整远场中央的
        ``cam_size x cam_size`` 裁剪, 使光斑 (位于画面中心) 留在窗口内。
        """
        if not self._open:
            raise CameraError("SimPibCCD not opened")
        full = self.system.far_field_noisy()
        h, w = full.shape
        s = self.cam_size
        if s >= h or s >= w:
            img = full
        else:
            y0 = (h - s) // 2
            x0 = (w - s) // 2
            img = full[y0 : y0 + s, x0 : x0 + s]
        return img.astype(np.float64)

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (size, (0, 0))

    def reset_exposure_time(self, time_ms: float) -> float:
        self.exposure_time_ms = float(time_ms)
        return float(time_ms)

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        return True

    def get_auto_exposure_state(self) -> dict:
        return {"enabled": False, "mode": 1, "target": 0}

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        return True

    @staticmethod
    def get_cam_list() -> list:
        return ["sim"]

    # --- 曝光兼容辅助 ---------------------------------------------

    @property
    def min_exposure_ms(self) -> float:
        return 0.011

    @property
    def max_exposure_ms(self) -> float:
        return 10_000.0

    def auto_exposure(
        self,
        target_max: float = 40.0,
        tolerance: float = 5.0,
        max_iterations: int = 20,
        n_sample: int = 1,
    ) -> np.ndarray:
        """返回缩放后峰值 ~= target_max 的远场 (0-255 量纲)。"""
        img = self.get_numpy_image(n_sample=n_sample)
        peak = float(img.max())
        if peak > 0 and target_max > 0:
            img = img * (target_max / peak)
        return img


def register_sim_camera() -> None:
    """注册 ``"sim"`` 相机类型, 使 ``create_camera("sim", ...)`` 可用。"""
    from ao_shaping.drivers.ccd.common import register_camera

    try:
        register_camera("sim", SimPibCCD)
    except Exception as exc:  # 已注册过
        logger.debug("sim camera already registered: {}", exc)


# 在此处注册 (而不只是放在 ``register_sim_camera()`` 里) 使 ``--cam_type sim`` 对每一个
# 消费者都可用。此前它恰好只有两个调用点 (slm-gsnet runner 的 ``_maybe_sim_patch`` 与
# ``scripts/slm_pib_sim_run.py`` 那套脚手架), 因此文档里写的命令
# ``main.py slm-pib spgd --cam_type sim`` 会死于
# ``ValueError: Unknown camera type: 'sim'``。持有仿真远场本就意味着持有读取它的仿真
# 相机, 所以 import 本模块就是把两者绑在一起的自然位置。
register_sim_camera()
