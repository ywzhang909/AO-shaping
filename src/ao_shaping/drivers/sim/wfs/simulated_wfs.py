"""基于 OOPAO 的模拟 Shack-Hartmann 波前传感器。

为何用 slope 模型而非焦平面模型
--------------------------------
Shack-Hartmann 传感器测量的是**瞳面相位梯度**: 微透镜阵列把瞳孔成像到探测器上, 光斑
位移编码各子孔径的局部 tip/tilt。因此驱动 ``wfs_measure(..., phase_in=)`` 所测得的,
恰好就是 AO 环路所需的量。

改而复用本仓库已有的焦平面传播 (早先某个方案曾这么提议) 会产出*相机*图像而不是
slope —— 二者不可互换, 而优化器消费的是 slope。

保真度
------
slope 来自 OOPAO 的微透镜 FFT 而非某个凑数公式, 且该测量满足 AO 环路所依赖的不变量:
**确定性** (重复读取逐位相同)、**符号对称**, 且在平 pupil 上**恰好为零**。

*Zernike 系数*远不如这些不变量保真, 报告绝不应把它们当作真值。实测: 以弧度注入已知
模式, 再读回 ``get_zernike()``:

    isolated single modes   noll 2 tilt  24%     noll 4 defocus  1.4%     noll 11 spherical 32%
    four modes together     noll 2 tilt  83%     noll 4 defocus  0.9%     noll 7 coma 17%   noll 11 spherical 25%

tilt 单独注入只恢复到 24%, 与其他模式同注入却只恢复到 83%, 因此主误差是**模式间串扰**
(cross-talk), 而非逐模式噪声。这是该传感器在如此建模下的真实性质: 微透镜图像是一张
强度图 (``|FFT|**2``), 它对相位是二次的, 所以模式之间的交叉项无法被线性化掉。细化采样
并不能修好这一点 —— 把 ``n_pixel_per_subaperture`` 从 8 提到 32、``n_subap`` 从 6 提到
24 之后, 组合模式误差仍在 80-140%, 且条件数始终良性 (cond 2-6), 这排除了秩亏。

因此本类适合用来跑控制环、检验符号约定、验证 µm/waves 这条单位链。它**不是**一个合格的
波前参考: 任何由 ``get_zernike()`` 推出的 RMS 改善率或 Strehl 都必须标注为**未验证**。
要可信的数值, 请像 ``zernike-matrix`` 那样对硬件做标定。

未建模: 微透镜像差、探测器噪声, 以及光斑串扰。

单位 (历史上危险的那一部分)
-----------------------------
WFS 这一族以**微米**返回 Zernike 系数, 而矫正是以 **waves** 施加的。两者混用造成过两个
真实 bug (系数被放大 1.88x; 施加的相位缩小 2*pi = 6.28x)。因此本传感器内部始终保持
弧度, 只在边界处换算:

* ``get_wavefront()`` -> waves      = ``phase_rad / (2*pi)``
* ``get_zernike()``   -> micrometres = ``phase_rad * lambda_um / (2*pi)``

``um_to_waves()`` 是写死 532 nm 的, 所以默认波长必须是 532 nm, 否则 ``um_to_waves()``
再 ``* 2*pi`` 无法还原弧度。换一个波长会静默地重新缩放整条校正链。
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

from ao_shaping.drivers.sim import _oopao_compat as oopao
from ao_shaping.drivers.sim.dm_optics import SimDmOptics
from ao_shaping.drivers.sim.disturbance import DisturbanceConfig, SimDisturbance
from ao_shaping.drivers.wfs._registry import register_wfs
from ao_shaping.drivers.wfs.base import BaseWFS
from ao_shaping.utils.wavefront.zernike_utils import (
    generate_zernike_phase,
    list_zernike_modes,
)

_MATRIX_CACHE: dict[tuple, np.ndarray] = {}

#: 最近构造的传感器。``SimulateDM`` 把电压发布到这里, 于是 DM 驱动的环路才真的能移动
#: 传感器所测的 pupil。之所以是进程级的, 理由与 ``slm_pib_sim.get_system()`` 相同: DM 与
#: 传感器由彼此独立的工厂构造, 从不共享引用。
_ACTIVE_SENSOR: SimulatedWFS | None = None


def get_active_sim_wfs() -> SimulatedWFS | None:
    """返回最近构造的模拟传感器, 若有。"""
    return _ACTIVE_SENSOR


#: Shack-Hartmann 传感器对 piston 是盲的: 子孔径的绝对相位偏移不会移动它的光斑。
#: ``list_zernike_modes`` 从 Noll 1 (piston) 起, 把它一起拟合会留下一列近零空间向量,
#: 而 ``pinv`` 会把它放大成一个巨大的伪系数 (实测: 对纯离焦瞳孔为 +0.50 rad)。因此
#: piston 被排除出拟合基。
_FIT_FIRST_MODE = 1


def _to_numpy(array: Any) -> np.ndarray:
    """CuPy → NumPy。OOPAO 自带的 ``to_numpy`` 在已安装的 CuPy 上是坏的。"""
    try:
        import cupy as cp

        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except ImportError:
        pass
    return np.asarray(array)


@register_wfs("sim")
class SimulatedWFS(BaseWFS):
    """与真实驱动共用契约的模拟 Shack-Hartmann WFS。"""

    manufacturer = "Simulated"
    model = "SimulatedWFS"

    def __init__(
        self,
        resolution: int = 48,
        diameter: float = 0.30,
        n_subap: int = 6,
        n_pixel_per_subaperture: int = 8,
        wavelength_nm: int = 532,
        zernike_order: int = 4,
        disturbance_cn2: float = 0.0,
        disturbance_config: DisturbanceConfig | None = None,
        disturbance_seed: int = 20261001,
    ) -> None:
        super().__init__()
        if resolution % (n_subap * n_pixel_per_subaperture) != 0:
            raise ValueError(
                f"resolution {resolution} must be divisible by "
                f"n_subap * n_pixel_per_subaperture "
                f"({n_subap} * {n_pixel_per_subaperture})"
            )
        self.grid = int(resolution)
        self.diameter = float(diameter)
        self.n_subap = int(n_subap)
        self.n_pixel_per_subaperture = int(n_pixel_per_subaperture)
        self.wavelength_nm = int(wavelength_nm)
        self.zernike_order = int(zernike_order)

        # 面向 Thorlabs 的接口面。优化器与 runner 直接读取这些成员
        # (``wfs.num_spots_x``、``wfs.mla_index``、``wfs.serial_num`` ...), 所以一个
        # 仿真传感器即便没有 MLA、没有序列号、没有厂商 DLL, 也必须回答它们。
        self.num_spots_x = self.n_subap
        self.num_spots_y = self.n_subap
        self.mla_index = None
        self.serial_num = "SIM-WFS-0001"
        self.device_name = "Simulated Shack-Hartmann WFS"
        self.exposure_time = 0.0
        self.high_speed = True
        self.use_custom_ref = False
        self.pupil = np.ones((self.grid, self.grid), dtype=np.float64)
        # 子孔径间距, 单位像素; SH 网格把瞳孔均分。
        self.d_x = self.diameter / self.n_subap if self.n_subap else self.diameter

        self._phase_rad = np.zeros((self.grid, self.grid), dtype=float)
        # 按传感器网格而非 SLM 面板定尺寸: `SimPibSystem` 为远场另有一份实例,
        # 复用那一份会需要在每次测量时把 1200x1920 重采样到这个网格。
        self.dm_optics = SimDmOptics(slm_shape=(self.grid, self.grid))
        self._slopes: np.ndarray | None = None
        self._flux: np.ndarray | None = None
        self._raw: np.ndarray | None = None

        # OOPAO 在构造时会打横幅表格; 让该库保持安静。
        with contextlib.redirect_stdout(io.StringIO()):
            self._tel = oopao.Telescope(
                resolution=self.grid,
                diameter=self.diameter,
                fov=0.0,
                samplingTime=0.001,
            )
            self._src = oopao.Source(
                optBand="R", magnitude=0.0, display_properties=False
            )
            self._src = self._src * self._tel
            self._wfs = oopao.ShackHartmann(
                nSubap=self.n_subap,
                telescope=self._tel,
                lightRatio=0.5,
                n_pixel_per_subaperture=self.n_pixel_per_subaperture,
            )
            self._wfs.relay(self._src)
        key = next(iter(self._wfs.sh_data))
        self._sh_data = self._wfs.sh_data[key]
        self._open = False
        self._set_state_ok()
        global _ACTIVE_SENSOR
        _ACTIVE_SENSOR = self
        # 待校正的可选像差。没有它时 DM 只能*增加*相位, 因此一个最小化 RMS 的环路
        # 正确地保留平场命令, 也就没什么可演示的了。
        if disturbance_config is not None:
            self.disturbance = SimDisturbance(
                disturbance_config, (self.grid, self.grid)
            )
        elif disturbance_cn2 > 0.0:
            self.disturbance = SimDisturbance(
                DisturbanceConfig(mode="static", cn2=disturbance_cn2,
                                  seed=disturbance_seed),
                (self.grid, self.grid),
            )
        else:
            self.disturbance = None

    def _set_state_ok(self) -> None:
        from ao_shaping.drivers.device_base import DeviceState

        self._set_state(DeviceState.READY)

    # ---- 配置 ---------------------------------------------------------

    def _n_modes(self, order: int) -> int:
        return len(list_zernike_modes(order))

    def _matrix_key(self, order: int) -> tuple:
        return (
            self.grid,
            self.diameter,
            self.n_subap,
            self.n_pixel_per_subaperture,
            self.wavelength_nm,
            int(order),
        )

    def _reconstructor(self, order: int) -> np.ndarray:
        """把实测 slope 映射到 Zernike 弧度的伪逆。

        通过把每个 Zernike 模式推过传感器来数值标定, 这与 ``zernike-matrix`` 命令在硬件
        上所执行的流程相同。它在构造上就是自洽的: 不假设任何灵敏度, 不依赖外部标定文件,
        且不可能与正向模型发生漂移。
        """
        key = self._matrix_key(order)
        cached = _MATRIX_CACHE.get(key)
        if cached is not None:
            return cached

        n_modes = self._n_modes(order)
        columns = []
        with contextlib.redirect_stdout(io.StringIO()):
            for j in range(_FIT_FIRST_MODE, n_modes):
                phase = self._mode_phase(j, order)
                slopes = _to_numpy(
                    self._wfs.wfs_measure(
                        self._src, self._sh_data, phase_in=phase
                    )[0]
                )
                columns.append(slopes.ravel())
        matrix = np.stack(columns, axis=1)
        pseudo = np.linalg.pinv(matrix)
        _MATRIX_CACHE[key] = pseudo
        return pseudo

    def _mode_phase(self, index: int, order: int) -> np.ndarray:
        """单个 Noll 模式的弧度相位, 口径外填零。"""
        modes = list_zernike_modes(order)
        noll, _n, _m, _name = modes[index]
        phase = generate_zernike_phase(
            {noll: 1.0},
            resolution=(self.grid, self.grid),
            n_max=order,
            radius=self.grid / 2.4,
        )
        # 规范生成器在口径外返回 NaN; 瞳孔掩模本就排除了那一块, 因此把它压平供 FFT 使用。
        return np.nan_to_num(np.asarray(phase, dtype=float), nan=0.0)

    # ---- 状态 ---------------------------------------------------------

    def set_pupil_phase(self, phase_rad: np.ndarray) -> None:
        """注入传感器应当测量的瞳面相位 (弧度)。"""
        arr = np.asarray(phase_rad, dtype=float)
        if arr.shape != (self.grid, self.grid):
            raise ValueError(
                f"pupil phase shape {arr.shape} != {(self.grid, self.grid)}"
            )
        self._phase_rad = np.nan_to_num(arr, nan=0.0)
        self._slopes = None

    def _pupil_phase(self) -> np.ndarray:
        """DM 相位加上任何显式注入的相位, 在求值时累加。

        沿用 ``SimDisturbance`` 的先例: 各贡献者在此处累加而不是烘进 ``_phase_rad``,
        于是两者都不会被重复计入, 而存下的命令恰好就是调用方设的那个。
        """
        total = self._phase_rad
        dm_phase = self.dm_optics.phase()
        if dm_phase.any():
            total = total + dm_phase
        if self.disturbance is not None:
            total = total + self.disturbance.phase()
        return total

    def take_slopes(self) -> tuple[np.ndarray, np.ndarray]:
        """测量并返回传感器原生单位下的 ``(dx, dy)`` slope。

        OOPAO 返回的是一个按行分块的数组: 第 ``0:n_subap`` 行携带 x-slope, 第
        ``n_subap:`` 行携带 y-slope (用纯倾斜探针实测确认)。
        """
        if self._slopes is None:
            with contextlib.redirect_stdout(io.StringIO()):
                measured = self._wfs.wfs_measure(
                    self._src, self._sh_data, phase_in=self._pupil_phase()
                )
            self._slopes = _to_numpy(measured[0])
            self._flux = _to_numpy(measured[1])
            self._raw = _to_numpy(measured[2])
        n = self.n_subap
        return self._slopes[:n], self._slopes[n : 2 * n]

    def _zernike_radians(self, order: int) -> np.ndarray:
        dx, dy = self.take_slopes()
        slopes = np.concatenate([dx.ravel(), dy.ravel()])
        return self._reconstructor(order) @ slopes

    # ---- BaseWFS 契约 -------------------------------------------------

    def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> None:
        """刷新测量。仿真传感器无噪声。"""
        self._slopes = None
        self.take_slopes()

    def get_spots_statics(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """来自微透镜平面图像的光斑强度与质心。"""
        self.take_slopes()
        assert self._raw is not None and self._flux is not None
        raw = np.asarray(self._raw, dtype=float)
        flux = np.asarray(self._flux, dtype=float).ravel()
        flat = raw.reshape(raw.shape[0], -1) if raw.ndim > 2 else raw
        with np.errstate(invalid="ignore", divide="ignore"):
            weight = np.where(flat > 0, flat, 0.0)
            cols = np.arange(weight.shape[1], dtype=float)
            cx = np.nansum(weight * cols, axis=1) / np.maximum(
                np.nansum(weight, axis=1), 1e-12
            )
        rows = np.arange(weight.shape[0], dtype=float)[:, None]
        cy = rows * np.ones_like(cx)[None, :]
        return flux, (cx, cy)

    def build_subaperture_mask(
        self,
        n_avg: int = 30,
        threshold_ratio: float = 0.3,
        edge_clip: int = 1,
        plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """"num_spots_x" x ``num_spots_y`` 子孔径网格上的有效性掩模。

        返回 ``(mask_bool, valid_indices_flat)``: 二维布尔掩模以及有效条目的扁平索引。
        调用方会用 ``np.concatenate([mask.ravel(), mask.ravel()])`` 过滤 slope 向量并与
        ``2 * num_spots_x * num_spots_y`` 行相对照, 因此该掩模必须按子孔径网格定尺寸,
        而不是按 flux 数组。

        它不可能按 flux 定尺寸: OOPAO 把 flux 报在 8x8 微透镜网格上 (``n_subap=6`` 时
        64 项), 而 slope 跨的是 6x6 子孔径网格 (36 项, 展开为 72 个 slope)。两者是不同
        的几何。

        该模型没有逐子孔径的渐晕可供检测, 所以除所请求的边缘裁剪之外, 每个子孔径都有效。
        """
        nx, ny = self.num_spots_x, self.num_spots_y
        mask = np.ones((nx, ny), dtype=bool)
        if edge_clip and nx > 2 * edge_clip and ny > 2 * edge_clip:
            mask[:edge_clip, :] = False
            mask[-edge_clip:, :] = False
            mask[:, :edge_clip] = False
            mask[:, -edge_clip:] = False
        return mask, np.flatnonzero(mask.ravel())

    def get_spot_deviation(
        self, cancel_tile: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """相对参考的光斑位移, 单位为传感器原生单位。"""
        dx, dy = self.take_slopes()
        if cancel_tile and self.remove_tilt:
            dx = dx - float(dx.mean())
            dy = dy - float(dy.mean())
        return dx, dy

    def get_wavefront(self, cancel_tile: bool = False) -> tuple[np.ndarray, dict]:
        """以 **waves** 表示的重建波前, 以及汇总统计量。

        统计量的键与 :meth:`ThorlabWFS.get_wavefront` 完全一致
        (``min``/``max``/``diff``/``mean``/``rms``/``wighted_rms``)。runner 会读
        ``statics["wighted_rms"]`` 来调度学习率, 所以字典少一个键就会在优化中途抛
        ``KeyError``。
        """
        order = self.zernike_order
        fit_rad = self._zernike_radians(order)
        fitted = np.zeros((self.grid, self.grid), dtype=float)
        for k, j in enumerate(range(_FIT_FIRST_MODE, self._n_modes(order))):
            fitted += fit_rad[k] * self._mode_phase(j, order)
        wavefront = fitted / (2.0 * np.pi)
        # piston 不可测, 因此残差是相对去掉 piston 的 pupil 取的; 否则那个 DC 偏移会
        # 主导 RMS, 那个数就不再描述模式被恢复了多少。
        pupil = self._pupil_phase()
        residual = pupil - fitted - float(pupil.mean())
        stats = {
            "min": float(wavefront.min()),
            "max": float(wavefront.max()),
            "diff": float(wavefront.max() - wavefront.min()),
            "mean": float(wavefront.mean()),
            "rms": float(np.sqrt(np.mean(residual**2))),
            "wighted_rms": float(np.std(wavefront)),
        }
        return wavefront, stats

    def get_zernike(self, zernike_order: int = 10) -> np.ndarray:
        """以**微米**表示、按 Noll 排序的 Zernike 系数。

        微米是 WFS 的历史契约。调用方在做任何 waves 相关的运算之前, 必须先用
        ``zernike_utils.um_to_waves()`` 换算。piston 报为零, 因为传感器测不到它。
        """
        fit_rad = self._zernike_radians(zernike_order)
        n_all = self._n_modes(zernike_order)
        z_rad = np.zeros(n_all, dtype=float)
        z_rad[_FIT_FIRST_MODE:] = fit_rad
        return z_rad * (self.wavelength_nm * 1e-3) / (2.0 * np.pi)

    # ---- Thorlabs 专有接口面 (仿真侧无对应物) ------------------------

    def optimize_pupil(self) -> tuple[float, float, float, float]:
        """返回所配置的瞳孔; 没有可优化的东西。"""
        return (0.0, 0.0, float(self.diameter), float(self.diameter))

    def optimize_exposure_time_and_gain(self) -> tuple[float, float]:
        """仿真传感器既无曝光也无增益。"""
        return (float(self.get_parameter_value("exposure_time_ms") or 0.0), 1.0)

    def save_user_ref(self, backup_dir: str | None = None) -> bool:
        """没有需要持久化的用户参考; 一个报告成功的空操作。"""
        return True

    def load_user_ref(self, backup_path: str | None = None) -> bool:
        """没有需要加载的用户参考; 一个报告成功的空操作。"""
        return True

    def create_default_user_ref(self) -> bool:
        return True

    def get_mla_name(self) -> str:
        """当前微透镜阵列的名字。

        仿真传感器上没有 MLA, 但工具链会打印它、并把它写进报告元数据, 所以这里
        如实回答而不是抛异常。
        """
        return "SIM-MLA"

    def set_ref_plane(self, custom: bool) -> None:
        """选择参考面。空操作: 仿真没有用户 .ref 文件。"""
        self.use_custom_ref = bool(custom)

    # ---- 设备生命周期 ------------------------------------------------

    def open(self) -> None:
        self._open = True
        self._set_state_ok()

    def close(self) -> None:
        self._open = False
        from ao_shaping.drivers.device_base import DeviceState

        self._set_state(DeviceState.DISCONNECTED)

    def is_connected(self) -> bool:
        return self._open

    def get_hardware_info(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "manufacturer": self.manufacturer,
            "grid": self.grid,
            "n_subap": self.n_subap,
            "wavelength_nm": self.wavelength_nm,
            "zernike_units": "um",
            "wavefront_units": "waves",
            "simulated": True,
        }


__all__ = ["SimulatedWFS"]
