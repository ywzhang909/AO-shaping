"""用于无硬件跑 ``slm-pib`` 的模拟 2f 傅里叶 SLM + CCD。

``slm-pib`` 优化器 (``optimizer/wfless/slm_zernike_pib.py``) 驱动一台真实的 ``Santec``
SLM, 方式是 ``create_phase_from_array(phase_rad) -> uint16`` 与 ``display_data(gray, ...)``,
并从一台 CCD 上读取远场光斑。本模块提供的是一个纯 numpy 的替身, 它复现该契约的同时
真的传播光, 于是 SPGD 搜索有了一个*真实*的光学反馈环路:

* :class:`SimSLMPib` —— 接受 raw 弧度相位、把它存下来, 并在显示时计算被该相位调制的
  高斯入射光束的夫琅禾费远场 (瞳面场的 ``np.fft.fft2``, 居中)。0 级光斑落在画面正中,
  其形状对所施加相位的响应与一台 2f 实验系统完全一致。
* :class:`SimPibCCD` —— 一个 ``BaseCamera``, 其 ``get_numpy_image`` 直接返回仿真 SLM
  最近一次产出的远场图像 (再加一点类 Poisson 的散粒噪声地板, 使指标不会完全光滑)。

两者经 :class:`SimPibSystem` 连在一起; 它同时在相机注册表里注册一个 ``"sim"``
条目, 使 ``create_camera("sim", ...)`` 返回一个绑定到共享 system 的 ``SimPibCCD``。

这有意是一个*远场 FFT* 模型 (即 ``AGENTS.md`` 里的 2f 傅里叶实验系统), 而不是近场传播器:
SLM 在前焦面、CCD 在后焦面, 因此 CCD 图像就是 SLM 瞳面场的二维 FFT。这恰好是
``slm-pib`` 所工作的机制。

硬件契约 (2026-10-22 补齐)
--------------------------
本模块的仿真替身必须能**原样**跑那些在真实硬件上出过报告的采集脚本, 所以它对齐
硬件驱动的三个调用形态:

1. ``display_data(gray, wait_time_s)`` —— 硬件驱动的第二个**位置**参数就是
   ``wait_time_s`` (见 :meth:`Santec.display_data`), 而
   ``tools/kernels/bench_kernels.py`` 的 ``acquire_settled`` / ``capture_settled`` /
   ``display_and_average`` 正是这么调的。仿真没有面板松弛时间, 故该参数被**接受并忽略**
   (真实硬件的松弛由 ``bench_kernels`` 的稳定性判据处理, 仿真里面板是理想瞬时响应)。
2. ``set_phase_overlay(phase_rad | None)`` —— 弧度制**叠加层** (离焦等), 在
   :meth:`SimSLMPib.create_phase_from_array` 内部、**弧度→灰度之前**相加; ``None`` 清除。
   语义与硬件 :meth:`Santec.set_phase_overlay` 一致 (叠加后的相位**同时**进入远场,
   所以仿真的离焦响应是真的让光斑变糊, 不是假的)。
3. 尺寸不一致时**告警并跳过**, 不抛异常 (与硬件 ``create_phase_from_array`` 同)。

口径旋钮 (默认 = 本模块既有行为)
--------------------------------
:class:`SimPibSystem` 把真实实验系统的口径做成了可覆盖的旋钮, 默认值全部等于本模块
改动前的行为 (既有调用方的结果逐位不变):

============================  ==================================  ==========  ==========================================
旋钮                          默认值                              单位        含义
============================  ==================================  ==========  ==========================================
``slm_pixel_um``              :data:`SLM_PIXEL_UM` (8.0)         µm         面板像元节距
``wavelength_nm``             :data:`WAVELENGTH_NM` (1064.0)     nm         工作波长
``ccd_pixel_um``              :data:`CCD_PIXEL_UM` (2.2)         µm         相机像元节距
``focal_length_m``            :data:`FOCAL_LENGTH_M` (0.125)     m          傅里叶透镜标称焦距
``two_pi_gray``               :data:`TWO_PI_GRAY` (993)           灰度       2π 弧度对应的灰度 (设备动态值)
``beam_w0``                   :data:`BEAM_W0` (400.0)            面板 px    入射高斯光束 1/e² 强度半径
``cam_window_px``             :data:`CAM_WINDOW_PX` (250)        px         相机 ROI 窗口边长
``far_field_padding``         :data:`FAR_FIELD_PADDING` (4)      –          远场 FFT 补零因子 (定角标度)
``far_field_window``          :data:`FAR_FIELD_WINDOW` (1024)    px         FFT 之后保留的居中窗口
``peak_counts_per_ms``        ``None`` (走旧的相对强度口径)      counts     单位峰值分数每毫秒的 counts
``noise_spec``                ``None`` (走旧的 ADU 噪声模型)     –          counts 口径的探测器噪声模型
``panel_aberration``          ``None`` (理想面板)                –          面板静态相差 (Zernike + 粗糙度)
============================  ==================================  ==========  ==========================================

⚠️ **远场补零因子 = 角标度**: 远场一个采样点的物理尺寸是 ``λ·f / (N_pad · p)``, 而本模块
的约定是「**一个远场采样点 = 一个相机像元**」。故只有当 ``N_pad = λ·f / (d_cam · p)``
时角标度才与真实实验系统一致: 本实验系统实测 ``λf/(p·d_cam) = 7557`` (相机 px / 面板 px,
实测 Kx = 7546.4 / Ky = 7495.9), 面板 1920 列 ⇒ ``far_field_padding = 7557/1920 = 3.94``
⇒ **:data:`FAR_FIELD_PADDING` = 4 就是复现实测角标度的那个值**。
`scripts/slm/run_gs_refine_zernike_sweep.py` 用的 ``--sim-far-field-padding`` 默认
**16** 是一个 4× **放大**的显示口径 (同一个远场、像素更细), 不是另一套角标度 ——
引用该脚本的仿真结论时必须说明用的是哪个 padding。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.sim.disturbance import SimDisturbance
from ao_shaping.drivers.sim.dm_optics import SimDmOptics
from ao_shaping.drivers.slm.santec.slm200_constants import GRAY_SCALE_BITS

try:  # scipy 的 pocketfft 是多线程的, 而补零后的变换需要它。
    from scipy import fft as _scipy_fft
except ImportError:  # pragma: no cover - numpy 总是存在
    _scipy_fft = None

# ---------------------------------------------------------------------------
# 口径常量区 (所有可覆盖旋钮的默认值集中在这里, 别散落在代码里)
# ---------------------------------------------------------------------------

# SLM 面板几何 (与真实 Santec SLM-200 一致: 1920 x 1200, 10-bit)。
SLM_W = 1920
SLM_H = 1200
SLM_BITS = 10
SLM_MAX_GRAY = (1 << SLM_BITS) - 1  # 1023
TWO_PI_GRAY = 993  # 1064 nm 下的 2*pi 灰度 (设备动态值, 见 AGENTS.md)

# 面板像元节距 (µm) —— 真实 SLM#1 (22030108) 为 8 µm。
SLM_PIXEL_UM = 8.0

# 工作波长 (nm) —— 本实验系统 1064 nm。
WAVELENGTH_NM = 1064.0

# 相机像元节距 (µm) —— Daheng MER2-507-23GM 为 2.2 µm。
CCD_PIXEL_UM = 2.2

# 傅里叶透镜标称焦距 (m) —— 2f 光路的 f = 125 mm。
FOCAL_LENGTH_M = 0.125

# 入射高斯光束半径 (SLM 像素) —— 即打在 SLM 上的平场光束的 1/e² **强度** 半径。
# 取值使光束能宽裕地填满面板口径; 与实测焦面矫正后的平场光斑 FWHM ~14 px 相符
# (理想 2f: FWHM = 2√(2ln2)·λf/(π·w0) / d_cam = 14.2 px)。
BEAM_W0 = 400.0

# CCD 远场分辨率 (正方形)。0 级光斑落在画面正中。
CCD_RES = (512, 512)

# 在 ``fft2`` 之前施加到瞳面上的补零因子。远场像元间距由该变换固定, 因此不补零的 FFT
# 会把 0 级光斑放到 ~1.8 px FWHM —— 横跨它的采样几乎只有一个, 于是每个功率比指标都要
# 去除一个在数值上根本没被分辨的峰值。补零把这个间距按该因子缩小, 这在数字上等价于
# 真实实验系统通过缩小相机 ROI 获得的更长有效焦距 (裁剪*同一*数组只会放大它, 不增加任何
# 采样)。取 4 时光斑横跨 ~7 px。
#
# 🔴 **4 是复现实测角标度的那个值** (见模块 docstring「远场补零因子 = 角标度」):
# 远场一个采样点的物理尺寸 = λ·f/(N_pad·p); 本模块的约定是「一个远场采样点 = 一个相机
# 像元」, 于是 N_pad 必须 = λf/(d_cam·p) = 7557 (实测 Kx = 7546.4), 而面板 1920 列 ⇒
# padding = 3.94 ≈ 4。
FAR_FIELD_PADDING = 4

# 补零 FFT 之后保留的居中窗口的边长, 单位远场像素。它同时限定了缓存数组以及
# ``far_field_noisy`` 施加在上面的散粒/读出噪声; 必须 >= 相机窗口 (``cam_size``)。
FAR_FIELD_WINDOW = 1024

# 相机 ROI 窗口边长 (px) —— 真实实验系统用的是 250 px。
CAM_WINDOW_PX = 250


def _forward_fft2(field: np.ndarray) -> np.ndarray:
    """二维正向 FFT, 可用时走 scipy 的多线程核。"""
    if _scipy_fft is not None:
        return _scipy_fft.fft2(field, workers=-1)
    return np.fft.fft2(field)


@dataclass(frozen=True)
class SimPanelAberration:
    """面板静态相差 (Zernike 系数 + 粗糙度)。

    真实面板不是理想相位板: 实测平场光斑比理想 2f 宽得多 (~40-48 px vs ~6-8 px),
    说明面板自带静态相差与粗糙度。本类是这个效应的**注入点**, 默认全零 = 理想面板
    (即本模块改动前的行为)。

    Attributes:
        zernike_rad: Noll 序 → 弧度。键是 1-based Noll 序 (与
            :func:`ao_shaping.utils.wavefront.zernike_utils.parse_zernike_coefficients`
            一致), 值是**弧度**。
        zernike_n_max: 面板 Zernike 的径向阶数上限。
        roughness_sigma_rad: 面板粗糙度 (高斯随机场的 RMS, 弧度)。
        roughness_seed: 粗糙度随机场的种子 (固定 → 可复现)。
    """

    zernike_rad: Mapping[int, float] = field(default_factory=dict)
    zernike_n_max: int = 6
    roughness_sigma_rad: float = 0.0
    roughness_seed: int = 0

    def is_empty(self) -> bool:
        """是否什么也不注入 (⇒ 与不传本类逐位等价)。"""
        return not self.zernike_rad and float(self.roughness_sigma_rad) <= 0.0


@dataclass(frozen=True)
class SimNoiseSpec:
    """counts 口径的探测器噪声模型。

    只在 :class:`SimPibSystem` 开启 counts 口径 (``peak_counts_per_ms`` 非 ``None``)
    时生效; 默认 ``None`` ⇒ 走本模块既有的 ADU 噪声模型 (``noise_adu``), 既有调用方
    的结果逐位不变。

    Attributes:
        read_sigma_counts: 逐像元读出噪声的高斯标准差 (counts)。
        shot_k: 散粒噪声系数 —— ``σ_shot = shot_k · √signal_counts``。
        quantize_bits: 量化位深 (例如 8 ⇒ uint8, 帧被取整并裁剪到 ``[0, 255]``);
            ``None`` = 不量化。
    """

    read_sigma_counts: float = 0.0
    shot_k: float = 0.0
    quantize_bits: int | None = None


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
        dm_optics: Any = None,
        far_field_padding: int = FAR_FIELD_PADDING,
        far_field_window: int = FAR_FIELD_WINDOW,
        slm_pixel_um: float = SLM_PIXEL_UM,
        wavelength_nm: float = WAVELENGTH_NM,
        ccd_pixel_um: float = CCD_PIXEL_UM,
        focal_length_m: float = FOCAL_LENGTH_M,
        two_pi_gray: int = TWO_PI_GRAY,
        cam_window_px: int = CAM_WINDOW_PX,
        peak_counts_per_ms: float | None = None,
        noise_spec: SimNoiseSpec | None = None,
        panel_aberration: SimPanelAberration | None = None,
    ) -> None:
        """构造共享的光学状态。

        Args:
            slm_shape: SLM 面板形状 ``(h, w)``。
            ccd_res: 远场分辨率。
            beam_w0: 高斯入射光束的 1/e² **强度** 半径, 单位 SLM 像素。
            noise_adu: :meth:`far_field_noisy` 加入的散粒/读出噪声地板 (ADU 口径)。
            seed: 噪声模型的 RNG 种子。
            disturbance: 可选的湍流 + 热晕模型。仅关键字传入且默认 ``None``, 这样
                每个现有调用方都保持原先无干扰的行为。它的相位屏在 :meth:`far_field`
                内部被加到 SLM 命令相位上; ``self._phase`` 从不被修改, 因此干扰不会
                被重复计入。
            dm_optics: 可选的 DM 电压 → 瞳面相位耦合实例 (默认自建一个)。
            far_field_padding: 远场 FFT 的补零因子 (定角标度, 见模块 docstring)。
            far_field_window: FFT 之后保留的居中窗口边长 (远场像素)。
            slm_pixel_um: 面板像元节距 (µm) —— 只用于**报告**远场的物理角标度。
            wavelength_nm: 工作波长 (nm) —— 同上。
            ccd_pixel_um: 相机像元节距 (µm) —— 同上。
            focal_length_m: 傅里叶透镜标称焦距 (m) —— 同上。
            two_pi_gray: 2π 弧度对应的灰度值 (设备动态值, 1064 nm 下是 993)。
            cam_window_px: 相机 ROI 窗口边长 (px); 仿真相机的 ``cam_size`` 默认取它。
            peak_counts_per_ms: **绝对光度**换算 —— 单位峰值分数每毫秒曝光的 counts。
                ``None`` (默认) ⇒ 走旧的相对强度口径 (远场峰值归一到 100, 与曝光无关),
                既有调用方逐位不变。非 ``None`` ⇒ 相机读到的是 counts 帧:
                ``counts = 峰值分数 · peak_counts_per_ms · 曝光(ms)``。
            noise_spec: counts 口径的探测器噪声模型; 只在 counts 口径下生效。
            panel_aberration: 面板静态相差 (Zernike + 粗糙度); 默认 ``None`` = 理想面板。
        """
        if noise_spec is not None and peak_counts_per_ms is None:
            raise ValueError(
                "noise_spec (counts 口径) 需要 peak_counts_per_ms 同时给出: "
                "counts 口径的噪声必须知道 counts 的标度"
            )
        self.slm_h, self.slm_w = slm_shape
        self.ccd_h, self.ccd_w = ccd_res
        self.beam_w0 = float(beam_w0)
        self.noise_adu = float(noise_adu)
        self.disturbance = disturbance
        self.far_field_padding = max(1, int(far_field_padding))
        self.far_field_window = max(1, int(far_field_window))
        self.slm_pixel_um = float(slm_pixel_um)
        self.wavelength_nm = float(wavelength_nm)
        self.ccd_pixel_um = float(ccd_pixel_um)
        self.focal_length_m = float(focal_length_m)
        self.two_pi_gray = int(two_pi_gray)
        self.cam_window_px = max(1, int(cam_window_px))
        self.peak_counts_per_ms = (
            None if peak_counts_per_ms is None else float(peak_counts_per_ms)
        )
        self.noise_spec = noise_spec
        self.panel_aberration = panel_aberration
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._phase: np.ndarray = np.zeros((self.slm_h, self.slm_w), dtype=np.float64)
        self._far_field: np.ndarray | None = None
        self._pad_buffer: np.ndarray | None = None
        self._amplitude: np.ndarray | None = None
        self._gray: np.ndarray | None = None
        self._panel_aberration_phase: np.ndarray | None = None
        # DM 电压 → 瞳面相位耦合 (见 dm_optics.py)。未注入时使用与 SLM 面板
        # 同尺寸的默认实例, 使 ``SimulateDM`` (``create_dm("sim")``) 的
        # ``_publish_to_optics`` 与 DM 驱动的优化环路 (``pib``/``combined``)
        # 能找到 ``system.dm_optics``。
        self.dm_optics = dm_optics if dm_optics is not None else SimDmOptics(
            slm_shape=(self.slm_h, self.slm_w)
        )
        self._dm_version: int = 0

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

    # --- 几何 (只用于报告/换算, 不改变光学计算本身) --------------------

    @property
    def far_field_pitch_um(self) -> tuple[float, float]:
        """远场一个采样点的物理尺寸 (µm), 返回 ``(行, 列)``。

        ``p_pitch = λ·f / (N_pad · p_panel)`` —— 补零后 FFT 网格的采样间距。它是
        **角标度** 的唯一来源: 本模块的约定是「一个远场采样点 = 一个相机像元」,
        所以当 ``p_pitch == ccd_pixel_um`` 时角标度与真实实验系统一致
        (见模块 docstring 的推导)。
        """
        lam = self.wavelength_nm * 1e-9
        pitch = self.slm_pixel_um * 1e-6
        n_rows = self.slm_h * self.far_field_padding
        n_cols = self.slm_w * self.far_field_padding
        row = lam * self.focal_length_m / (n_rows * pitch) * 1e6
        col = lam * self.focal_length_m / (n_cols * pitch) * 1e6
        return float(row), float(col)

    @property
    def camera_px_per_model_px(self) -> tuple[float, float]:
        """一个远场采样点等于多少相机像元, 返回 ``(行, 列)``。

        默认口径下 = ``(1.57, 0.98)`` —— 即**列**方向与真实实验系统一致 (面板是
        1920×1200 的长方形, 单一补零因子无法同时对齐两轴)。
        """
        row_um, col_um = self.far_field_pitch_um
        return float(row_um / self.ccd_pixel_um), float(col_um / self.ccd_pixel_um)

    @property
    def counts_mode(self) -> bool:
        """是否走 counts 口径 (绝对光度)。

        ``True`` ⇒ 相机读到的是 counts 帧 (峰值随曝光线性变化); ``False`` ⇒ 走旧的
        相对强度口径 (远场峰值恒为 100, 与曝光无关)。
        """
        return self.peak_counts_per_ms is not None

    def far_field(self) -> np.ndarray:
        """计算 (并缓存) 所显示相位的远场图像。

        瞳面场是 ``A(r) * exp(i*phase)``, 其中 ``A`` 是高斯入射光束; 远场是
        ``|FFT(pupil)|^2``, 并把零频分量移到中心 (``fftshift``)。0 级光斑即画面中心,
        与 ``AGENTS.md`` 里的 2f 实验系统约定一致。
        """
        with self._lock:
            # DM 电压每次变化都会自增 ``dm_optics.version``; 版本变化说明瞳面相位
            # 已变, 缓存的远场 (基于旧电压) 必须失效, 否则 DM 耦合看起来毫无效果。
            dm_ver = self.dm_optics.version
            if dm_ver != self._dm_version:
                self._dm_version = dm_ver
                self._far_field = None
            if self._far_field is not None:
                return self._far_field.copy()
            phase = self._phase
            # 干扰与 SLM 命令共处同一个瞳面, 所以它在傅里叶变换之前就被加进去。
            # 上面的提前返回保证每次真实光学求值只消耗它一次。`self._phase` 保持为
            # 纯命令 —— 把干扰烘进去会被重复计入。
            if self.disturbance is not None:
                phase = phase + self.disturbance.phase()
            # DM 相位 (电压 → 瞳面相位) 与干扰同一约定: 在求值时叠加进瞳面相位,
            # 绝不烘进 ``self._phase``, 避免 SLM 命令与 DM 贡献重复计入。
            phase = phase + self.dm_optics.phase()
            # 面板静态相差 (Zernike + 粗糙度) 同样是**面板的属性**, 在瞳面里相加。
            # 同一约定: 求值时才加, ``self._phase`` 保持纯命令。
            panel_phase = self._panel_aberration_field()
            if panel_phase is not None:
                phase = phase + panel_phase
            # SLM 网格上的高斯入射光束幅度 (静态几何 ⇒ 懒缓存, 见下)。
            amplitude = self._amplitude_field()
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

    def _amplitude_field(self) -> np.ndarray:
        """高斯入射光束幅度 (峰值 1), 只依赖静态几何 (``slm_h/w`` + ``beam_w0``)。

        懒缓存: 每次远场求值都重算 ``mgrid`` + ``exp`` 会白白 churn 4 个
        2.3M 元素临时数组 (~74 MB/次, 1920x1200 面板), 闭环里这是 FFT 之外
        最大的单笔浪费。
        """
        if self._amplitude is None:
            yy, xx = np.mgrid[0 : self.slm_h, 0 : self.slm_w]
            cy, cx = self.slm_h / 2.0, self.slm_w / 2.0
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            self._amplitude = np.exp(-r2 / (2.0 * self.beam_w0**2))
        return self._amplitude

    def _panel_aberration_field(self) -> np.ndarray | None:
        """面板静态相差 (Zernike + 粗糙度) 的瞳面相位, ``None`` = 理想面板。

        Zernike 用 canonical 入口 :func:`generate_zernike_phase` 生成 (不自写第二套
        Zernike 数学); 孔径外是 NaN, 按硬件路径置 0。粗糙度是固定种子的高斯随机场,
        RMS = ``roughness_sigma_rad``。

        ⚠️ canonical 入口是**函数内延迟 import** 的: ``utils.wavefront`` 的包初始化
        会拉起 ``aotools`` → ``matplotlib``, 而本模块在 import 期就注册 ``"sim"``
        相机 (必须在一个**纯净**解释器里也能 import —— 见
        ``tests/ao_shaping/drivers/sim/test_sim_camera_registration.py``)。
        """
        # 延迟 import: 见 docstring 的说明 (import 期不得拉起 matplotlib)。
        from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

        spec = self.panel_aberration
        if spec is None or spec.is_empty():
            return None
        if self._panel_aberration_phase is not None:
            return self._panel_aberration_phase
        coeffs = {int(k): float(v) for k, v in spec.zernike_rad.items()}
        if coeffs:
            raw = generate_zernike_phase(
                coeffs,
                resolution=(self.slm_w, self.slm_h),
                n_max=int(spec.zernike_n_max),
                radius=min(self.slm_w, self.slm_h) / 2.0,
            )
            zernike = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
        else:
            zernike = np.zeros((self.slm_h, self.slm_w), dtype=np.float64)
        sigma = float(spec.roughness_sigma_rad)
        if sigma > 0:
            rng = np.random.default_rng(int(spec.roughness_seed))
            field = rng.normal(0.0, 1.0, size=(self.slm_h, self.slm_w))
            rms = float(np.sqrt(np.mean(field**2)))
            roughness = field * (sigma / rms)
        else:
            roughness = np.zeros((self.slm_h, self.slm_w), dtype=np.float64)
        self._panel_aberration_phase = zernike + roughness
        return self._panel_aberration_phase

    def far_field_noisy(self) -> np.ndarray:
        """远场加上散粒噪声 + 读出噪声, 以原始 ADU 帧返回。

        结果刻意**不**在零处裁剪。``far_field`` 是 ``|FFT|**2``, 本身已非负, 因此这里
        每一个负值都只可能来自读出噪声。在零处裁剪会把这种对称噪声整流成一个与像素数
        成正比的 DC 地板: 光斑在一幅 1200x1920 的画面里只占 ~150 px 信号, 而该地板实测
        约为信号的 ~3000x。于是每个功率比指标除的都是那个地板而不是光, 这正是
        ``pib``/``combined`` 停在 ~0.0145、其 SPGD 梯度估计全是噪声的原因。

        真实传感器的地板位于其裁剪阈值*之下*, 只在量化时被裁剪一次, 因此原始帧可以
        略低于黑电平; 去掉该偏移靠的是暗帧扣除, 而不是逐帧 clip。

        ⚠️ 这是**旧的 ADU 口径**噪声模型。counts 口径 (绝对光度) 走
        :meth:`counts_frame`。
        """
        img = self.far_field()
        if self.noise_adu > 0:
            shot = self._rng.poisson(np.clip(img, 0, None).astype(np.float64) / 10.0)
            img = img + (shot - img / 10.0) * (self.noise_adu / 10.0)
            img = img + self._rng.normal(0.0, self.noise_adu / 10.0, img.shape)
        return img

    def counts_frame(self, exposure_ms: float) -> np.ndarray:
        """远场 → **counts** 帧 (绝对光度 + 可选的 counts 噪声)。

        换算: ``counts = 峰值分数 · peak_counts_per_ms · 曝光(ms)``, 其中峰值分数是
        ``far_field() / 100`` (远场峰值恒被归一到 100)。故曝光翻倍 ⇒ counts 翻倍,
        这与真实相机的线性响应一致。

        未开启 counts 口径 (``peak_counts_per_ms is None``) 时**回落到**
        :meth:`far_field_noisy` (旧的相对强度口径), 于是既有调用方逐位不变。

        Args:
            exposure_ms: 曝光时间 (ms)。

        Returns:
            counts 帧 (形状同 :meth:`far_field`)。

        Raises:
            ValueError: ``exposure_ms`` 为负。
        """
        exposure = float(exposure_ms)
        if exposure < 0.0:
            raise ValueError(f"exposure_ms must be >= 0, got {exposure_ms!r}")
        if self.peak_counts_per_ms is None:
            return self.far_field_noisy()
        fraction = self.far_field() / 100.0
        counts = fraction * float(self.peak_counts_per_ms) * exposure
        spec = self.noise_spec
        if spec is not None:
            if spec.shot_k > 0:
                sigma = spec.shot_k * np.sqrt(np.clip(counts, 0.0, None))
                counts = counts + self._rng.normal(0.0, 1.0, counts.shape) * sigma
            if spec.read_sigma_counts > 0:
                counts = counts + self._rng.normal(
                    0.0, spec.read_sigma_counts, counts.shape
                )
            if spec.quantize_bits is not None:
                levels = (1 << int(spec.quantize_bits)) - 1
                counts = np.clip(np.round(counts), 0.0, float(levels))
        return counts


# 一个进程级的 system, 让注册表创建的 CCD 与被 monkeypatch 的 SLM 共享同一份光学状态。
_SYSTEM: SimPibSystem | None = None


def get_system(
    seed: int | None = None,
    *,
    disturbance: SimDisturbance | None = None,
    far_field_padding: int | None = None,
    far_field_window: int | None = None,
    slm_pixel_um: float | None = None,
    wavelength_nm: float | None = None,
    ccd_pixel_um: float | None = None,
    focal_length_m: float | None = None,
    two_pi_gray: int | None = None,
    cam_window_px: int | None = None,
    peak_counts_per_ms: float | None = None,
    noise_spec: SimNoiseSpec | None = None,
    panel_aberration: SimPanelAberration | None = None,
) -> SimPibSystem:
    """返回进程级的 :class:`SimPibSystem` (首次使用时创建)。

    ``disturbance`` 与其余口径旋钮**只在 system 于此处被创建时**生效; 已存在的
    system 原样返回 (用 :func:`reset_system` 来替换它)。

    口径旋钮的默认值 = :class:`SimPibSystem` 的默认值 (= 本模块改动前的行为);
    传 ``None`` 即「用默认值」。

    Args:
        seed: 噪声模型的 RNG 种子。
        disturbance: 可选的湍流 + 热晕模型。
        far_field_padding: 远场 FFT 补零因子 (定角标度)。
        far_field_window: FFT 之后保留的居中窗口边长 (远场像素)。
        slm_pixel_um: 面板像元节距 (µm)。
        wavelength_nm: 工作波长 (nm)。
        ccd_pixel_um: 相机像元节距 (µm)。
        focal_length_m: 傅里叶透镜标称焦距 (m)。
        two_pi_gray: 2π 弧度对应的灰度值。
        cam_window_px: 相机 ROI 窗口边长 (px)。
        peak_counts_per_ms: 绝对光度换算 (counts/(峰值分数·ms)); ``None`` = 旧口径。
        noise_spec: counts 口径的探测器噪声模型。
        panel_aberration: 面板静态相差 (Zernike + 粗糙度)。

    Returns:
        进程级的 system。
    """
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = SimPibSystem(
            seed=seed,
            disturbance=disturbance,
            **{
                key: value
                for key, value in (
                    ("far_field_padding", far_field_padding),
                    ("far_field_window", far_field_window),
                    ("slm_pixel_um", slm_pixel_um),
                    ("wavelength_nm", wavelength_nm),
                    ("ccd_pixel_um", ccd_pixel_um),
                    ("focal_length_m", focal_length_m),
                    ("two_pi_gray", two_pi_gray),
                    ("cam_window_px", cam_window_px),
                    ("peak_counts_per_ms", peak_counts_per_ms),
                    ("noise_spec", noise_spec),
                    ("panel_aberration", panel_aberration),
                )
                if value is not None
            },
        )
    return _SYSTEM


def reset_system(
    seed: int | None = None,
    *,
    disturbance: SimDisturbance | None = None,
    far_field_padding: int | None = None,
    far_field_window: int | None = None,
    slm_pixel_um: float | None = None,
    wavelength_nm: float | None = None,
    ccd_pixel_um: float | None = None,
    focal_length_m: float | None = None,
    two_pi_gray: int | None = None,
    cam_window_px: int | None = None,
    peak_counts_per_ms: float | None = None,
    noise_spec: SimNoiseSpec | None = None,
    panel_aberration: SimPanelAberration | None = None,
) -> SimPibSystem:
    """替换进程级的 system (用于矩阵单元格之间)。

    口径旋钮的语义与 :func:`get_system` 相同 (``None`` = 用默认值)。
    """
    global _SYSTEM
    _SYSTEM = SimPibSystem(
        seed=seed,
        disturbance=disturbance,
        **{
            key: value
            for key, value in (
                ("far_field_padding", far_field_padding),
                ("far_field_window", far_field_window),
                ("slm_pixel_um", slm_pixel_um),
                ("wavelength_nm", wavelength_nm),
                ("ccd_pixel_um", ccd_pixel_um),
                ("focal_length_m", focal_length_m),
                ("two_pi_gray", two_pi_gray),
                ("cam_window_px", cam_window_px),
                ("peak_counts_per_ms", peak_counts_per_ms),
                ("noise_spec", noise_spec),
                ("panel_aberration", panel_aberration),
            )
            if value is not None
        },
    )
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
        # 弧度制相位叠加层 (离焦等) —— 语义同硬件 ``Santec._phase_overlay``。
        self._phase_overlay: np.ndarray | None = None
        # 跟踪最后显示的弧度相位, 供报告使用。
        self.last_phase_rad: np.ndarray | None = None
        # 叠加层合成**之后**真正上屏的相位 (= 命令 + 叠加)。
        self.last_applied_phase_rad: np.ndarray | None = None

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

    def set_phase_overlay(self, phase_rad: np.ndarray | None) -> None:
        """设置弧度制相位叠加层 (离焦等); ``None`` 清除。

        语义与硬件 :meth:`Santec.set_phase_overlay` 一致: 叠加层是**弧度制**的, 在
        :meth:`create_phase_from_array` 内部、**弧度→灰度之前**相加。因此

        * 送进面板的灰度 = ``mod(命令 + 叠加, 2π) / 2π · 2π灰度``;
        * 远场看到的也是 ``命令 + 叠加`` —— 离焦/焦面补偿在仿真里**真的**让光斑变糊,
          不是假的。

        尺寸不匹配时**告警并跳过** (不抛异常), 与硬件 ``create_phase_from_array``
        的处理一致: 叠加是修正层, 不该卡住核心发送路径。

        Args:
            phase_rad: 弧度制叠加层 (面板尺寸); ``None`` 清除。
        """
        if phase_rad is None:
            self._phase_overlay = None
            return
        # 存副本: 调用方之后修改输入数组不能静默改变驱动上的叠加层。
        self._phase_overlay = np.array(phase_rad, dtype=np.float64, copy=True)

    def get_phase_overlay(self) -> np.ndarray | None:
        """返回当前弧度制相位叠加层的副本, 无叠加时返回 ``None``。"""
        if self._phase_overlay is None:
            return None
        return self._phase_overlay.copy()

    def create_phase_from_array(
        self, phase_rad: np.ndarray, max_grayscale: int | None = None
    ) -> np.ndarray:
        """弧度相位 → uint16 灰度, 并同时给光学系统上膛。

        在真实 Santec 上这一步只做弧度→灰度换算 (2*pi = ~993 灰度), 在
        ``display_data`` 之前不动面板。而这里额外把弧度相位存下来, 使下一次
        ``far_field()`` 会反映它。

        叠加层 (:meth:`set_phase_overlay`) 在**弧度→灰度之前**相加, 且相加后的
        相位**同时**进入远场 (与硬件语义一致)。

        Args:
            phase_rad: raw 弧度相位 (面板尺寸)。
            max_grayscale: 2π 弧度对应的灰度值; ``None`` ⇒ 用 system 的
                ``two_pi_gray`` (1064 nm 下是设备实测的 993)。

        Returns:
            灰度相位数组 (uint16)。
        """
        rad = np.asarray(phase_rad, dtype=np.float64)
        total = rad
        overlay = self._phase_overlay
        if overlay is not None:
            if overlay.shape == rad.shape:
                # 相位空间叠加: 先于弧度→灰度换算。
                total = rad + overlay
            else:
                logger.warning(
                    "相位叠加层尺寸 {} 与面板 {} 不一致，已跳过叠加"
                    "（请检查离焦/底相位设置）",
                    overlay.shape,
                    rad.shape,
                )
        # 存下**合成后**的弧度相位, 供远场使用 (离焦必须在远场里看得见)。
        self.system.set_phase_rad(total)
        self.last_phase_rad = rad.copy()
        self.last_applied_phase_rad = total.copy()
        # 弧度 → 灰度 (2*pi 映射到 2π 灰度), wrap 方式与驱动一致。
        two_pi_gray = (
            float(self.system.two_pi_gray) if max_grayscale is None else float(max_grayscale)
        )
        gray = (np.mod(total, 2.0 * np.pi) / (2.0 * np.pi) * two_pi_gray)
        return gray.astype(np.uint16)

    def display_data(
        self,
        gray: np.ndarray,
        memory_number: int | None = None,
        memory_mode: int | None = None,
        wait_time_s: float | None = None,
    ) -> int:
        """显示一幅灰度图样。

        优化器总是先调用 ``create_phase_from_array`` (它给相位上膛) 再调
        ``display_data``, 所以到这里时远场已经是对的。一次裸的灰度显示 (前面没有相位)
        被当作平场相位图样处理: 它的平均灰度映射到一个常数相位偏移, 而后者不改变远场的
        *形状* (只改变 0 级耦合), 所以我们把存下的相位原样保留。

        参数形态与硬件驱动 (``Santec.display_data``) 对齐, 以便同一份采集脚本在硬件
        与仿真下都能跑:

        * ``wait_time_s`` —— **接受并忽略**: 仿真没有松弛时间 (面板是理想瞬时响应),
          真实硬件的松弛由 ``tools/kernels/bench_kernels.py`` 的稳定性判据处理
          (``acquire_settled`` / ``capture_settled`` / ``display_and_average``)。
        * ``memory_number`` / ``memory_mode`` —— 仿真没有内存槽, 同样**接受并忽略**
          (返回的仍是 ``memory_number``, 以免影响既有调用方)。

        ⚠️ **位置参数的兼容**: 硬件驱动的第二个**位置**参数是 ``wait_time_s``,
        而 ``bench_kernels`` 正是 ``slm.display_data(gray, wait_time_s)`` 这么调的。
        所以当第二个位置参数是**非整数** (即它其实是等待时间而不是槽号) 时, 本方法
        把它当作 ``wait_time_s`` 处理 —— 与硬件的语义一致。

        Args:
            gray: 灰度图样 (面板尺寸)。
            memory_number: 内存槽号 (仿真忽略)。
            memory_mode: 内存模式 (仿真忽略)。
            wait_time_s: 显示后的等待时间 (秒, 仿真忽略)。

        Returns:
            ``memory_number`` (未给则 0) —— 与既有行为一致。
        """
        # 位置参数兼容: display_data(gray, wait_time_s) —— 硬件的第二个位置参数是
        # wait_time_s, 故非整数的第二实参按等待时间解释。
        if (
            memory_number is not None
            and wait_time_s is None
            and not isinstance(memory_number, (int, np.integer))
        ):
            wait_time_s = float(memory_number)
            memory_number = None
        self._open = True
        # 强制下一次 CCD 读取时 (重新) 计算远场。
        self.system.set_gray(np.asarray(gray, dtype=np.uint16))
        return int(memory_number) if memory_number is not None else 0

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
        cam_size: int | None = None,
        skip_sampling: bool = False,
        system: SimPibSystem | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=skip_sampling)
        self.system = system or get_system()
        # ROI 窗口边长: 未显式给定时取 system 的口径旋钮 (默认 250 px)。
        self.cam_size = (
            int(cam_size) if cam_size is not None else int(self.system.cam_window_px)
        )
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

        counts 口径 (system 的 ``peak_counts_per_ms`` 非 ``None``) 下, 返回的是
        **counts** 帧 —— 峰值随曝光线性变化; 否则是旧的相对强度口径 (峰值恒 100)。
        """
        if not self._open:
            raise CameraError("SimPibCCD not opened")
        if self.system.counts_mode:
            full = self.system.counts_frame(self.exposure_time_ms)
        else:
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


# 在此处注册 (而不只是放在 ``register_sim_camera()`` 里) 使 ``--cam-type sim`` 对每一个
# 消费者都可用。此前它恰好只有两个调用点 (slm-gsnet runner 的 ``_maybe_sim_patch`` 与
# ``scripts/slm/slm_pib_sim_run.py`` 那套脚手架), 因此文档里写的命令
# ``main.py slm-pib spgd --cam-type sim`` 会死于
# ``ValueError: Unknown camera type: 'sim'``。持有仿真远场本就意味着持有读取它的仿真
# 相机, 所以 import 本模块就是把两者绑在一起的自然位置。
register_sim_camera()
