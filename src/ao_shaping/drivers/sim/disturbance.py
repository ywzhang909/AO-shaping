"""2f 傅里叶 SLM 仿真的波前干扰。

``slm-pib`` 的仿真原本传播的是一份*完全干净*的波前:
:meth:`ao_shaping.drivers.sim.slm_pib_sim.SimPibSystem.far_field` 只用了 SLM 命令相位,
因此闭环从未暴露在任何波前干扰之下。本模块补上缺失的干扰 —— **大气湍流** 加上
**热晕 (热晕)** —— 并提供报告需要对比的 static/dynamic 两种机制。

两种干扰机制
-------------
``static``
    首次使用时生成一张相位屏, 之后永远复用。这就是冻结屏的情形, 对应
    ``scripts/generate_oopao_impact_report.py`` 所用的 ``closed`` 湍流模式。
``dynamic``
    **每一次**光学求值都抽一张全新的独立相位屏 —— 即 ``open``/滑动的对应物。
    它建模的是*完全去相关* ("时间上白") 极限。该极限在此是正确的渐近: 真实大气的
    去相关时间约 10-50 ms, 而本环路上一次 SPGD 求值要 ~0.375 s, 因此相邻两次求值
    实际上是不相关的。这明确**不是**风场/平流模型。

两种机制在 ``seed`` 固定时都是确定性的。

幅度一律*按实测*报告, 绝不给解析值
--------------------------------------
``cn2`` 与 ``distance_m`` 是**退化的生成器旋钮**: 规范生成器推出的是
``r0 = (0.423 * k**2 * cn2 * distance) ** (-3/5)``, 它只依赖二者的*乘积*。
单张薄屏不携带任何传播物理, 所以这里并没有在建模真实的公里级大气光路 —— 本台架是
一台 ~0.3 m 的实验室 2f 傅里叶装置。

此外, 本仓库默认的 (numpy) 相位屏生成器是用滤波后的白噪声场做 FFT 合成相位屏的,
**没有**次谐波/低频补偿 (见 ``src/ao_shaping/drivers/sim/AGENTS.md`` 注 4)。由于
``l_max`` (几十米) 远大于 15.36 mm 口径, von-Karman 方差的大部分落在基频 FFT 频率
以下, 因此实测 ``sigma`` 是同一 ``r0`` 下解析方差的一个**下界**。所以本模块报告的
每个幅度都是经 :meth:`SimDisturbance.stats` 从实际产出的数组上量出来的。

raw 弧度契约
-------------
:meth:`SimDisturbance.phase` 返回 **raw 未包裹弧度**, 从不施加 ``mod 2*pi`` ——
这与仓库规则一致: 唯一的 wrap 点是 SLM 驱动里弧度 → 灰度的转换。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.sim.beam_backend import make_beam_config, turbulence_phase
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

#: 合法的 :attr:`DisturbanceConfig.mode` 取值。
DISTURBANCE_MODES: tuple[str, ...] = ("none", "static", "dynamic")

#: 默认高斯光束腰, 单位 SLM 像素。与 ``slm_pib_sim.BEAM_W0`` 一致, 但这里**有意**
#: 重复一份: 从本模块去 import 光学系统模块会形成循环 (``slm_pib_sim`` 反过来
#: import 的是*本*模块)。
DEFAULT_BEAM_W0: float = 400.0

#: 在归一化热晕的峰谷值时, 低于该幅度的高斯瞳孔就算未被照亮, 以免归一化被数值
#: 噪声较大的远翼主导。
_AMPLITUDE_FLOOR: float = 1e-3

#: smoothstep 渐晕边的宽度, 占光晕半径的比例。
_APOD_EDGE_FRACTION: float = 0.35


@dataclass(frozen=True)
class DisturbanceConfig:
    """:class:`SimDisturbance` 的配置。

    面板几何刻意*不*属于这份配置 —— 它在构造时传入, 这样同一份配置可跨网格复用。

    Attributes:
        mode: ``"none"``、``"static"`` 或 ``"dynamic"``。
        cn2: 折射率结构常数。与 :attr:`distance_m` 退化: 只有二者的乘积决定 ``r0``。
        distance_m: 生成器的路径长度旋钮, 而不是本台架上的真实传播路径
            (见模块 docstring)。
        l_max: 外尺度 [m]。
        l_min: 内尺度 [m]。
        wavelength_m: 波长 [m]。
        pixel_pitch_m: SLM 像元间距 [m]; 它决定相位屏的物理范围, 使相位屏的像元与
            面板保持各向同性。
        thermal_halo_pv_waves: 热晕相位的峰谷值, 单位 waves。``0`` 表示禁用光晕。
        thermal_halo_radius_px: 保留光晕相位的半径, 单位 SLM 像素。远大于
            ``~2 * beam_w0`` 的取值会被高斯瞳孔幅度衰减到不可见 (在默认 400 px 腰
            半径下, 600 px 处为 0.325、800 px 处为 0.135)。
        halo_noll: 喂给规范入口
            :func:`~ao_shaping.utils.wavefront.zernike_utils.generate_zernike_phase`
            的 ``(noll_index, coefficient)`` 对。负系数给出负热透镜。
        halo_n_max: 光晕基的 Zernike 最大径向阶数。
        seed: 主种子; 完全决定两种机制。
    """

    mode: str = "none"
    cn2: float = 2e-13
    distance_m: float = 500.0
    l_max: float = 30.0
    l_min: float = 2e-3
    wavelength_m: float = 1064e-9
    pixel_pitch_m: float = 8e-6
    thermal_halo_pv_waves: float = 0.30
    thermal_halo_radius_px: float = 600.0
    halo_noll: tuple[tuple[int, float], ...] = ((4, -1.0), (11, -1.0))
    halo_n_max: int = 4
    seed: int = 20261001

    def __post_init__(self) -> None:
        if self.mode not in DISTURBANCE_MODES:
            raise ValueError(
                f"unknown disturbance mode {self.mode!r}; expected one of {DISTURBANCE_MODES}"
            )
        if self.cn2 < 0.0:
            raise ValueError(f"cn2 must be >= 0, got {self.cn2}")
        if self.distance_m <= 0.0:
            raise ValueError(f"distance_m must be > 0, got {self.distance_m}")
        if self.pixel_pitch_m <= 0.0:
            raise ValueError(f"pixel_pitch_m must be > 0, got {self.pixel_pitch_m}")
        if self.l_min <= 0.0 or self.l_max <= 0.0:
            raise ValueError("l_min and l_max must be > 0")
        if self.thermal_halo_pv_waves < 0.0:
            raise ValueError(
                f"thermal_halo_pv_waves must be >= 0, got {self.thermal_halo_pv_waves}"
            )
        if self.thermal_halo_pv_waves > 0.0 and not self.halo_noll:
            raise ValueError("halo_noll must be non-empty when the thermal halo is enabled")


class SimDisturbance:
    """在 ``(h, w)`` 面板网格上的湍流 + 热晕相位干扰。

    内存是有意受限的。``dynamic`` 模式每次光学求值都会生成一张全分辨率相位屏,
    而 300 个 epoch 的运行要做 ~630 次求值; 把它们都留着 (1200x1920 float64 数组
    每张约 18 MB) 要吃掉好几 GB。这里只累加标量, 唯一长期存活的数组是那张冻结屏 ——
    在 ``dynamic`` 模式下相位屏交还给调用方, 从不在此留存。

    Args:
        config: 干扰配置。
        shape: 面板形状 ``(h, w)`` —— 例如 SLM 的 ``(1200, 1920)``。
        beam_w0: 高斯光束腰, 单位像素, 只用于掩模光晕的峰谷值归一化。
            默认取 :data:`DEFAULT_BEAM_W0`。

    Raises:
        ValueError: 配置不支持, 或面板 ``h > w``。
    """

    def __init__(
        self,
        config: DisturbanceConfig,
        shape: tuple[int, int],
        *,
        beam_w0: float | None = None,
    ) -> None:
        self.config = config
        self.shape: tuple[int, int] = (int(shape[0]), int(shape[1]))
        self.beam_w0 = float(DEFAULT_BEAM_W0 if beam_w0 is None else beam_w0)
        if self.shape[0] > self.shape[1]:
            raise ValueError(
                f"panel shape {self.shape} is taller than it is wide; the turbulence "
                "generator is square and the screen is cropped from the wide axis, so "
                "h <= w is required"
            )

        self._halo = self._build_halo()
        self._mask = self._amplitude_mask()
        self._static_total: np.ndarray | None = None
        self._streak_count = 0
        self._turb_sigma_sum = 0.0
        self._last_total_std = 0.0
        self._calls = 0
        self._call_rms: list[float] = []
        self._call_streak_index: list[int] = []

        logger.debug(
            "SimDisturbance(mode={}, cn2={:g}, distance={:g}m, halo_pv={:g}waves, shape={})",
            config.mode,
            config.cn2,
            config.distance_m,
            config.thermal_halo_pv_waves,
            self.shape,
        )

    # --- 构造辅助 ---------------------------------------------------

    def _amplitude_mask(self) -> np.ndarray:
        """高斯瞳孔被有效照亮处的布尔掩模。"""
        h, w = self.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2
        return np.exp(-r2 / (2.0 * self.beam_w0**2)) > _AMPLITUDE_FLOOR

    def _build_halo(self) -> np.ndarray:
        """确定性的热晕相位, 峰谷值归一化到请求的 waves 数。

        光晕是一个*稳态*像差 (留在光路上的负热透镜), 因此只构造一次并加到每一条湍流
        上, 而不重新随机化。

        构造过程遵循仓库规则: 所有 Zernike 数学都走规范 API 入口 —— 单位系数的相位
        来自 :func:`generate_zernike_phase` (对系数线性), 其峰谷值在被照亮的瞳孔内
        量出, 再对数组做线性缩放, 使 PV 等于 ``thermal_halo_pv_waves * 2*pi`` 弧度。
        """
        h, w = self.shape
        if self.config.thermal_halo_pv_waves <= 0.0:
            return np.zeros(self.shape, dtype=np.float64)

        coefficients = {int(noll): float(amp) for noll, amp in self.config.halo_noll}
        unit = np.nan_to_num(
            np.asarray(
                generate_zernike_phase(
                    coefficients,
                    resolution=(w, h),
                    n_max=self.config.halo_n_max,
                ),
                dtype=np.float64,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.sqrt((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2)
        edge = max(1.0, self.config.thermal_halo_radius_px * _APOD_EDGE_FRACTION)
        t = np.clip(
            (self.config.thermal_halo_radius_px + edge - radius) / (2.0 * edge), 0.0, 1.0
        )
        unit = unit * (t * t * (3.0 - 2.0 * t))  # smoothstep 渐晕

        mask = self._amplitude_mask()
        pv = float(np.ptp(unit[mask])) if np.any(mask) else 0.0
        if pv <= 0.0:
            logger.warning(
                "thermal halo has zero peak-to-valley after apodisation; using zeros"
            )
            return np.zeros(self.shape, dtype=np.float64)

        return unit * (self.config.thermal_halo_pv_waves * 2.0 * np.pi / pv)

    def _turbulence_streak(self, index: int) -> np.ndarray:
        """生成第 ``index`` 张独立湍流相位屏 (不做缓存)。

        按面板的原生像元间距生成 (``n_grid = w`` 覆盖 ``w * pixel_pitch``), 再**居中
        裁剪**到 ``h`` 行, 使相位屏像元各向同性、无各向异性拉伸、无拼接缝。实测更粗的
        网格会丢失约 41% 的相位幅度。
        """
        h, w = self.shape
        if self.config.cn2 <= 0.0:
            return np.zeros(self.shape, dtype=np.float64)

        cfg = make_beam_config(
            n_grid=w,
            aperture_size=w * self.config.pixel_pitch_m,
            wavelength=self.config.wavelength_m,
            cn2=self.config.cn2,
            l_max=self.config.l_max,
            l_min=self.config.l_min,
            propagation_distance=self.config.distance_m,
        )
        full = np.asarray(
            turbulence_phase(
                cfg,
                cn2=self.config.cn2,
                l_max=self.config.l_max,
                l_min=self.config.l_min,
                propagation_distance=self.config.distance_m,
                rng=np.random.default_rng([self.config.seed, index]),
            ),
            dtype=np.float64,
        )
        start = (full.shape[0] - h) // 2
        return np.ascontiguousarray(full[start : start + h, :w])

    # --- 公开 API ------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """是否正在施加干扰。"""
        return self.config.mode != "none"

    @property
    def halo_phase(self) -> np.ndarray:
        """确定性的热晕相位数组 (raw 弧度)。"""
        return self._halo

    def phase(self) -> np.ndarray:
        """返回当前光学求值所用的干扰相位。

        Returns:
            **raw 未包裹弧度** 的 ``(h, w)`` float64 数组。``"none"`` 返回全零;
            ``"static"`` 始终返回同一张冻结屏; ``"dynamic"`` 每次调用都返回一张
            全新的独立相位屏。
        """
        self._calls += 1

        if self.config.mode == "none":
            self._call_rms.append(0.0)
            self._call_streak_index.append(0)
            return np.zeros(self.shape, dtype=np.float64)

        if self.config.mode == "static":
            if self._static_total is None:
                turbulence = self._turbulence_streak(0)
                self._turb_sigma_sum = float(turbulence[self._mask].std())
                self._static_total = turbulence + self._halo
                self._streak_count = 1
            self._call_rms.append(float(self._static_total.std()))
            self._call_streak_index.append(0)
            return self._static_total

        index = self._streak_count
        turbulence = self._turbulence_streak(index)
        total = turbulence + self._halo
        self._turb_sigma_sum += float(turbulence[self._mask].std())
        self._streak_count += 1
        self._last_total_std = float(total.std())
        self._call_rms.append(self._last_total_std)
        self._call_streak_index.append(index)
        return total

    def stats(self) -> dict[str, float]:
        """实测的干扰幅度 (绝不给解析值)。

        Returns:
            只含标量: ``sigma_turb_rad``、``sigma_halo_rad``、
            ``sigma_total_rad``、``streaks_used``、``calls``、``beam_w0``、
            ``panel_pixels``。
        """
        halo_std = float(self._halo[self._mask].std()) if np.any(self._mask) else 0.0

        if self.config.cn2 <= 0.0:
            turb_std = 0.0
            total_std = float(self._halo.std())
        elif self.config.mode == "dynamic":
            if self._streak_count:
                turb_std = self._turb_sigma_sum / self._streak_count
                total_std = (
                    float(np.mean(self._call_rms)) if self._call_rms else self._last_total_std
                )
            else:
                probe = self._turbulence_streak(0)
                turb_std = float(probe[self._mask].std())
                total_std = float((probe + self._halo).std())
        else:
            if self._static_total is None:
                turbulence = self._turbulence_streak(0)
                self._turb_sigma_sum = float(turbulence[self._mask].std())
                self._static_total = turbulence + self._halo
            turb_std = self._turb_sigma_sum
            total_std = float(self._static_total.std())

        return {
            "sigma_turb_rad": turb_std,
            "sigma_halo_rad": halo_std,
            "sigma_total_rad": total_std,
            "streaks_used": float(self._streak_count),
            "calls": float(self._calls),
            "beam_w0": float(self.beam_w0),
            "panel_pixels": float(self.shape[0] * self.shape[1]),
        }

    def trace(self) -> dict[str, list]:
        """逐次求值的历史: ``call_rms`` 与 ``call_streak_index``。"""
        return {
            "call_rms": list(self._call_rms),
            "call_streak_index": list(self._call_streak_index),
        }

    def archive(self, *, factor: int = 8, max_count: int = 12) -> dict[str, np.ndarray]:
        """实际用到的那些不同相位屏的抽稀缩略图。

        相位屏是按需从各自的 streak 种子重新生成的, 而非留存, 因此归档在运行期间不占
        内存。对 ``dynamic`` 最多归档 ``max_count`` 个等间隔的 streak 索引 ——
        :meth:`trace` 仍会完整记录每一条 ``call_rms``, 那才是相位屏确实在变的证据。

        Args:
            factor: 空间抽稀因子; 缩略图为 ``(h // factor, w // factor)``。
            max_count: 最多归档多少张不同相位屏。

        Returns:
            ``{"screens": (n, h//factor, w//factor) float32,
            "streak_indices": (n,) int32}``。
        """
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")

        h, w = self.shape
        thumb_shape = (h // factor, w // factor)

        if self.config.mode == "none":
            return {
                "screens": np.zeros((0, *thumb_shape), dtype=np.float32),
                "streak_indices": np.zeros((0,), dtype=np.int32),
            }

        if self.config.mode == "static":
            total = self._static_total
            if total is None:
                total = self._turbulence_streak(0) + self._halo
            return {
                "screens": np.asarray(total[::factor, ::factor], dtype=np.float32)[None, ...],
                "streak_indices": np.zeros((1,), dtype=np.int32),
            }

        count = self._streak_count
        if count == 0:
            return {
                "screens": np.zeros((0, *thumb_shape), dtype=np.float32),
                "streak_indices": np.zeros((0,), dtype=np.int32),
            }
        if count <= max_count:
            indices = list(range(count))
        else:
            indices = sorted(
                {int(i) for i in np.linspace(0, count - 1, max_count).round()}
            )
        screens = [
            np.asarray(
                (self._turbulence_streak(i) + self._halo)[::factor, ::factor],
                dtype=np.float32,
            )
            for i in indices
        ]
        return {
            "screens": np.stack(screens, axis=0),
            "streak_indices": np.asarray(indices, dtype=np.int32),
        }

    def to_dict(self) -> dict[str, Any]:
        """可 JSON 序列化的配置 + 实测统计 (不含数组)。"""
        config = asdict(self.config)
        config["halo_noll"] = [list(pair) for pair in self.config.halo_noll]
        return {
            "mode": self.config.mode,
            "enabled": self.enabled,
            "beam_w0": float(self.beam_w0),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "config": config,
            "measured": self.stats(),
        }
