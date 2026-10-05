"""基于 OOPAO 的湍流相位屏生成器与角谱传播器。

把 AO-shaping 的仿真内核从 :mod:`beam_backend` 里手写的 FFT 谱 Kolmogorov 路径,
改为基于 OOPAO 库 (github.com/cheritier/OOPAO, ESO/LAM)。从参考实现
(beaconless-ao-sim/physics) 移植了两个原语:

* :class:`OopaoScreenBackend` —— 逐采样的 von-Karman 相位屏, 经
  ``Atmosphere.generateNewPhaseScreen(seed)`` 生成, 并做逐 slab 的 r0 重缩放, 使
  生成的相位屏在统计上等价于 legacy 的 aotools/FFT 路径。
* :class:`Propagator` (以及模块级的 :func:`propagate_asm`) —— 经 OOPAO
  ``Atmosphere.ASM`` 做角谱传播, 并在 OOPAO 不可用时回落到纯 numpy 转写实现。

OOPAO 经 :mod:`ao_shaping.drivers.sim._oopao_compat` 导入, 该模块只加载我们需要的
子模块, 从而屏蔽掉上游那份坏掉的 ``OOPAO/__init__.py`` (不兼容 numpy 2.x)。OOPAO 库
是可选的: 未安装时 :func:`_oopao_available` 返回 False, 调用方继续走 numpy/legacy
路径 (见 :mod:`beam_backend` 中的 ``AO_OOPAO_BACKEND``)。

逐层 r0 标定
------------
OOPAO 的 ``cn2`` 记账方式是把总 Cn2 除以 ``max(altitude)``, 这并不能正确地把路径
r0 切成逐层 r0 (逐层相位方差会偏强)。因此我们绕开它: 每一层都在参考 r0
``_R0_REF_500 = 0.15`` (OOPAO 自己的 ``r0_def``, 在其写死的 500 nm 约定下) 上生成,
再做幅度重缩放, 使其逐 slab 的 r0 在仿真波长处*恰好*等于
``r0_slab = r0_path * n**(3/5)`` —— 与 legacy FFT 路径所用的逐 slab r0 相同。由于
von-Karman 的 PSD 按 ``r0**(-5/3)`` 标度, 相位幅度 ~ ``r0**(-5/6)``;
:func:`_rescale_for` 施加的正是这个指数, 且别无他物。

已知局限: 没有内尺度
---------------------
``Atmosphere.__init__`` 没有 ``l0`` 参数, ``generateNewPhaseScreen`` 调用
``ft_sh_phase_screen`` 时也不传, 所以 OOPAO 恒用 ``l0 = 1e-10`` —— 它的内尺度滚降
远在任何网格 Nyquist 频率之上, 实际上并不存在。因此所配置的 ``l_min`` 在本后端上
无法兑现。这只在 ``l_min`` 粗到能被网格分辨时才有影响;
:func:`inner_scale_is_resolvable` 判定该情形, 由路由层发出警告。
"""

from __future__ import annotations

import contextlib
import io
from functools import lru_cache
from typing import Any

import numpy as np

try:
    from ao_shaping.drivers.sim._oopao_compat import (
        Atmosphere,
        Source,
        Telescope,
    )

    _OOPAO_AVAILABLE = True
except Exception:  # pragma: no cover - OOPAO not installed
    _OOPAO_AVAILABLE = False

__all__ = [
    "OopaoScreenBackend",
    "Propagator",
    "make_screens",
    "propagate_asm",
    "rayleigh_range",
    "inner_scale_is_resolvable",
    "_oopao_available",
]

# 重缩放之前用于生成 OOPAO 各层的参考 r0。是任选的 (OOPAO 自己的 r0_def); 每一层的
# 相位由 _rescale_for() 幅度重缩放到目标的逐 slab r0。不需要经验归一化常数: OOPAO 那张
# 加了次谐波的相位屏, 对它生成时所用的 r0 已经复现了解析 von-Karman 方差。
_R0_REF_500 = 0.15

# OOPAO 的 Atmosphere 写死了 500 nm 约定, 且无法给定内尺度。legacy 路径的内尺度滚降
# 特征频率为 fm = 5.92 / (2*pi*l_min); 只有当它降到网格 Nyquist 频率 1 / (2*dx) 时才
# *可分辨*。在此之下 l_min 是亚像素的、物理上无关紧要, 所以 OOPAO 忽略它无害。
_L_MIN_RESOLVABLE_FACTOR = 5.92 / np.pi


def inner_scale_is_resolvable(l_min: float, pixel_size: float) -> bool:
    """所配置的 ``l_min`` 是否粗到能被网格分辨。

    legacy 生成器在 ``fm = 5.92 / (2*pi*l_min)`` 以上做高频滚降。该滚降只有在 ``fm``
    达到网格 Nyquist 频率 ``1/(2*dx)``、即 ``l_min >~ 5.92*dx/pi`` 时才影响所表示的
    谱。低于该阈值时 ``l_min`` 是亚像素的, 本后端无法表示内尺度这件事也就无关紧要。

    Args:
        l_min: 配置的内尺度 [m]。
        pixel_size: 网格像元间距 [m]。

    Returns:
        True 表示内尺度足够粗, 以致 legacy 路径与 OOPAO 确实会合理地产生分歧。
    """
    if l_min <= 0.0 or pixel_size <= 0.0:
        return False
    return float(l_min) > _L_MIN_RESOLVABLE_FACTOR * float(pixel_size)


def _oopao_available() -> bool:
    """OOPAO 库导入成功时返回 True。"""
    return _OOPAO_AVAILABLE


def compute_r0(lam: float, cn2: float, L: float) -> float:
    """单层湍流的 Fried 参数 ``r0``。

    ``r0 = (0.423 * k**2 * Cn2 * L)**(-3/5)``, 其中 ``k = 2*pi/lam``。
    """
    k = 2.0 * np.pi / lam
    return float((0.423 * k**2 * cn2 * L) ** (-3.0 / 5.0))


def _rescale_for(r0_slab: float) -> float:
    """从 OOPAO 的参考相位屏重缩放到目标 r0 的幅度因子。

    OOPAO 的 ``layer.OPD`` 是其写死 500 nm 约定下的**弧度相位**, 生成时的 r0 是
    ``r0=_R0_REF_500`` (注意: ``generateNewPhaseScreen`` 会用一张弧度值的相位屏
    覆写掉初值那个以米为单位的 OPD, 所以层上名为 "OPD" 的属性是弧度)。

    它的 std 严格按 ``r0**(-5/6)`` 标度 (在 10 倍 r0 区间上验证过: 拟合指数 -0.8333
    对理论值 -5/6, 归一化常数 0.619322 在 6 位数字上恒定)。因此重缩放到目标 r0 是
    *精确的*, 不需要任何额外因子:

    * **不含波长因子。** 仿真波长处的相位统计完全由 Fried 参数决定, 而 ``r0_slab``
      本身就是在仿真波长处算出来的 (``compute_r0`` 用 ``k = 2*pi/lam``)。由于
      ``r0 ∝ lam**(6/5)`` 且弧度相位 ∝ ``r0**(-5/6) ∝ 1/lam``, 显式的 ``lam`` 项会
      重复计入并把波长依赖完全抵消。(早先某个版本带着 ``lam/_LAM_REF_500``; 那个因子
      恰好抵消了 r0 依赖, 使本后端的相位 std *与波长无关*, 而对一张以弧度计的相位屏
      来说这不符合物理。)
    * **不做 ``_CAL_REF`` 除法。** OOPAO 自己的归一化就是物理上正确的目标: 它那张加了
      次谐波的相位屏复现了解析 von-Karman 方差, 而 legacy 的纯 FFT 路径则*低*估了
      低阶功率。除以一个经验常数会把那部分丢掉。
    * **没有 ``sqrt(1.03)`` 因子。** 该常数属于去 piston 的*纯 Kolmogorov*
      (L0 -> inf) 口径方差; 在模拟有限外尺度时施加它属于范畴错误。

    Args:
        r0_slab: 仿真波长处的目标逐 slab Fried 参数 [m]。
    """
    return (_R0_REF_500 / float(r0_slab)) ** (5.0 / 6.0)


class OopaoScreenBackend:
    """经 OOPAO ``Atmosphere`` 生成逐采样湍流相位屏。

    Parameters
    ----------
    N : int
        瞳孔网格边长, 单位像素。
    dx : float
        像元尺度, 单位米。
    Dscope : float
        望远镜口径, 单位米 (定义 OOPAO 的瞳孔)。
    lam : float
        仿真波长, 单位米。
    cn2 : float
        Cn2, 单位 ``m**(-2/3)``。
    L : float
        传播路径长度, 单位米。
    L0 : float
        外尺度, 单位米。
    n_screens : int
        湍流层/相位屏的数目。
    """

    def __init__(
        self,
        N: int,
        dx: float,
        Dscope: float,
        lam: float,
        cn2: float,
        L: float,
        L0: float,
        n_screens: int,
    ) -> None:
        if not _OOPAO_AVAILABLE:
            raise RuntimeError(
                "OOPAO is not available; install the oopao dependency to use "
                "OopaoScreenBackend"
            )
        self.N = int(N)
        self.n_screens = int(n_screens)
        self.lam = float(lam)

        # OOPAO 会往 stdout 打横幅表格 (Telescope/Atmosphere 的初始化信息);
        # 把它们重定向掉, 让该库在我们的日志/测试里保持安静。
        with contextlib.redirect_stdout(io.StringIO()):
            self.tel = Telescope(
                resolution=self.N, diameter=float(Dscope), fov=0.0, samplingTime=0.001
            )

            self.src = Source(optBand="R", magnitude=0.0, display_properties=False)
            self.src * self.tel

            r0_path = compute_r0(self.lam, float(cn2), float(L))
            self.r0_slab = r0_path * self.n_screens ** (3.0 / 5.0)

            self._rescale = _rescale_for(self.r0_slab)

            n = self.n_screens
            self._altitudes = np.linspace(50.0, float(L) - 50.0, n).tolist()
            self._frac = [1.0 / n] * n

            self.atm = Atmosphere(
                self.tel,
                r0=_R0_REF_500,
                L0=float(L0),
                windSpeed=[10.0] * n,
                fractionalR0=self._frac,
                windDirection=[0.0] * n,
                altitude=self._altitudes,
                src=self.src,
            )
            self.atm.initializeAtmosphere(self.tel, compute_covariance=False)

    def make_screens(
        self, seed: int, r0_slab: float | None = None
    ) -> np.ndarray:
        """为采样 ``seed`` 抽取 ``n_screens`` 张全新的 OOPAO 相位屏。

        Parameters
        ----------
        seed : int
            采样种子; OOPAO 会用 ``seed + i_layer`` 给每一层重新播种。
        r0_slab : float, optional
            逐采样的逐 slab r0 [m]。给出时, 参考 r0 的 OOPAO 各层会被重缩放到这个
            目标逐 slab r0, 而不是常量 L 的 ``self.r0_slab``。恒定的幅度重缩放在统计上
            等价于一次 r0 变更 (PSD 形状中的 r/L0/l0 保持不变)。

        Returns
        -------
        np.ndarray
            ``(n_screens, N, N)`` float32 弧度相位屏, 从 OOPAO 的 ``N+4`` 像素层居中
            裁剪而来, 并重缩放到目标逐 slab r0。
        """
        rescale = self._rescale
        if r0_slab is not None:
            rescale = _rescale_for(float(r0_slab))
        with contextlib.redirect_stdout(io.StringIO()):
            self.atm.generateNewPhaseScreen(seed=int(seed))
        out = np.empty((self.n_screens, self.N, self.N), dtype=np.float32)
        for i in range(self.n_screens):
            lay = getattr(self.atm, "layer_%d" % (i + 1))
            out[i] = (np.asarray(lay.OPD)[2:-2, 2:-2] * rescale).astype(
                np.float32
            )
        return out


# ---------------------------------------------------------------------------
# 模块级便捷包装 (供 beam_backend 使用)
# ---------------------------------------------------------------------------


def make_screens(
    *,
    N: int,
    dx: float,
    Dscope: float,
    lam: float,
    cn2: float,
    L: float,
    L0: float,
    n_screens: int,
    seed: int,
    l0: float | None = None,
) -> np.ndarray:
    """构造 :class:`OopaoScreenBackend` 并抽一批相位屏。

    一个与 :func:`beam_backend.turbulence_phase` 参数面完全对应的轻量包装; ``l0``
    为 API 对齐而接受, 但并未使用 (OOPAO 的 ``Atmosphere`` 不直接接受内尺度)。
    """
    backend = _get_backend(
        N=N,
        dx=dx,
        Dscope=Dscope,
        lam=lam,
        cn2=cn2,
        L=L,
        L0=L0,
        n_screens=n_screens,
    )
    return backend.make_screens(seed=seed)


@lru_cache(maxsize=8)
def _get_backend(
    *,
    N: int,
    dx: float,
    Dscope: float,
    lam: float,
    cn2: float,
    L: float,
    L0: float,
    n_screens: int,
) -> OopaoScreenBackend:
    """返回一个进程内存活、按配置为键的 :class:`OopaoScreenBackend`。

    做缓存是因为构造 OOPAO 的 Telescope + Atmosphere 既贵 (~100 ms) 又啰嗦;
    传给 ``make_screens`` 的种子完全决定所生成的相位屏 (已验证每个种子确定性), 所以
    跨采样复用同一个后端实例是安全的。
    """
    return OopaoScreenBackend(
        N=N,
        dx=dx,
        Dscope=Dscope,
        lam=lam,
        cn2=cn2,
        L=L,
        L0=L0,
        n_screens=n_screens,
    )


# ---------------------------------------------------------------------------
# 角谱传播 (OOPAO ASM + numpy 回退)
# ---------------------------------------------------------------------------


def rayleigh_range(w0: float, lam: float) -> float:
    """返回 Rayleigh 范围 z_R = pi * w0**2 / lam。"""
    return float(np.pi * w0**2 / lam)


_ASM_HOST: Any = None


def _asm_host() -> Any:
    """惰性构造一个仅用作 ASM 宿主的迷你 OOPAO ``Atmosphere``。

    ``ASM`` 是纯函数; 它不依赖宿主 Atmosphere 的分辨率/分层配置。这里用一个最小的
    望远镜 + 光源 + 单层大气, 唯一目的就是承载那次角谱调用 (OOPAO 要求望远镜先注册
    一个 Source, 因此有 ``src * tel``)。
    """
    global _ASM_HOST
    if _ASM_HOST is None:
        if not _OOPAO_AVAILABLE:
            raise RuntimeError("OOPAO is not available for the ASM host")
        with contextlib.redirect_stdout(io.StringIO()):
            tel = Telescope(resolution=8, diameter=1.0, fov=0.0, samplingTime=0.001)
            src = Source(optBand="R", magnitude=0.0, display_properties=False)
            _ = src * tel
            _ASM_HOST = Atmosphere(
                telescope=tel,
                r0=0.15,
                L0=30.0,
                windSpeed=[10.0],
                windDirection=[0.0],
                fractionalR0=[1.0],
                altitude=[0.0],
                elevation=90.0,
                angular_spectrum_propagation=True,
            )
    return _ASM_HOST


def propagate_asm(
    input_field: np.ndarray,
    *,
    lam: float,
    dx: float,
    z: float,
) -> np.ndarray:
    """OOPAO ``Atmosphere.ASM`` 角谱传播 (纯函数)。

    Parameters
    ----------
    input_field : np.ndarray
        输入复场 (N, N)。
    lam : float
        波长 (m)。
    dx : float
        输入/输出像元间距 (m); 此处输入与输出间距相等。
    z : float
        传播距离 (m); 负值表示反向传播。
    """
    if not _OOPAO_AVAILABLE:
        raise RuntimeError("OOPAO is not available; use the numpy backend")
    return _asm_host().ASM(input_field, lam, dx, dx, z)


def _asm_numpy(
    input_field: np.ndarray,
    wavelength: float,
    input_pitch: float,
    output_pitch: float,
    distance: float,
) -> np.ndarray:
    """OOPAO ``Atmosphere.ASM`` 的纯 numpy 忠实转写。

    仅在 OOPAO 不可用时作为回退引擎, 保证数值行为完全一致。
    """
    if distance == 0:
        return input_field
    N = input_field.shape[0]
    k = 2.0 * np.pi / wavelength
    grid_dtype = (
        np.float64 if input_field.dtype in (np.complex128, np.float64) else np.float32
    )
    delta_f = 1.0 / (N * input_pitch)
    vals = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * delta_f
    fx, fy = np.meshgrid(vals, vals, copy=False)
    f_sq = fx**2 + fy**2
    vals_r = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * input_pitch
    x, y = np.meshgrid(vals_r, vals_r, copy=False)
    r_sq = x**2 + y**2
    m = output_pitch / input_pitch
    if m != 1.0:
        phase_1 = np.exp(1j * k / 2.0 * (1 - m) / distance * r_sq)
    else:
        phase_1 = 1.0
    phase_2 = np.exp(-1j * np.pi * wavelength * distance / m * f_sq)
    if m != 1.0:
        vals_out = np.arange(-N / 2.0, N / 2.0, dtype=grid_dtype) * output_pitch
        x_out, y_out = np.meshgrid(vals_out, vals_out, copy=False)
        r_out_sq = x_out**2 + y_out**2
        phase_3 = np.exp(1j * k / 2.0 * (m - 1) / (m * distance) * r_out_sq)
    else:
        phase_3 = 1.0
    field_freq = np.fft.fft2(np.fft.ifftshift(input_field * phase_1))
    field_filtered = np.fft.ifftshift(np.fft.fftshift(field_freq) * phase_2)
    field_out = np.fft.fftshift(np.fft.ifft2(field_filtered))
    return field_out * phase_3 / m


class Propagator:
    """基于 OOPAO ``Atmosphere.ASM`` 的角谱场传播器。

    Parameters
    ----------
    N : int
        网格边长 (正方形, N x N)。
    dx : float
        网格采样间距 (m)。
    lam : float
        波长 (m)。
    n_threads : int, optional
        兼容参数 (legacy 的 FFTW 线程数); OOPAO/numpy 引擎都不用它。
    engine : str, optional
        ``"oopao"`` (默认): OOPAO ``Atmosphere.ASM`` 核; OOPAO 不可用时自动回落到
        ``"numpy"``。``"numpy"``: 纯 numpy 转写 (逐位相同)。
    dtype : np.dtype, optional
        工作复数 dtype (默认 ``np.complex64``)。
    """

    def __init__(
        self,
        N: int,
        dx: float,
        lam: float,
        n_threads: int = 1,
        engine: str = "oopao",
        dtype: np.dtype = np.dtype(np.complex64),
    ) -> None:
        self.N = N
        self.dx = dx
        self.lam = lam
        self.dtype = dtype

        if engine not in ("oopao", "numpy"):
            raise ValueError(f"unknown propagation engine: {engine!r}")
        if engine == "oopao" and not _OOPAO_AVAILABLE:
            engine = "numpy"
        self.engine = engine

    def _propagate_raw(self, E: np.ndarray, z: float) -> np.ndarray:
        if z == 0.0:
            return np.array(E, dtype=self.dtype, copy=True)
        if self.engine == "oopao":
            field = _asm_host().ASM(E, self.lam, self.dx, self.dx, z)
        else:
            field = _asm_numpy(E, self.lam, self.dx, self.dx, z)
        return np.asarray(field, dtype=self.dtype)

    def propagate(self, E: np.ndarray, z: float) -> np.ndarray:
        """把复场传播距离 ``z`` (m)。"""
        out = self._propagate_raw(E, z)
        return np.array(out, dtype=self.dtype, copy=True)

    def split_step(
        self,
        E_in: np.ndarray,
        screens: list[np.ndarray] | np.ndarray,
        dz: float,
    ) -> np.ndarray:
        """穿过相位屏的对称分步传播。"""
        E = np.array(E_in, dtype=self.dtype, copy=True)
        for phi in screens:
            E = self.propagate(E, dz / 2.0)
            E = np.array(E, dtype=self.dtype, copy=True)
            E *= np.exp(1j * phi).astype(self.dtype)
            E = self.propagate(E, dz / 2.0)
        return E

    def angular_spectrum_intensity(self, E_in: np.ndarray, z: float) -> np.ndarray:
        """返回传播后的强度 ``|propagate(E_in, z)|**2``。"""
        E = self.propagate(E_in, z)
        return (np.abs(E) ** 2).astype(np.float32)
