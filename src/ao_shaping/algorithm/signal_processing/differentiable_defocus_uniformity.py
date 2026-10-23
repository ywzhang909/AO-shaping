"""离焦光路下的可微均匀度优化（冻结物理前向 + 固定掩膜 CV）。

把**已拟合的冻结物理前向模型**装成可微 torch 模块，并在其上提供均匀度目标与一个
类式优化器。约束来自 `report/shape_trials/defocus_differentiable_test_plan_20261023.md`
§3–§5：前向数学与检查点拟合口径逐项一致（实测复现存档预测相对 L1 9.976e-08）、
面板相位始终 raw 未包裹弧度、损失用固定掩膜内的 CV（不用全图峰值归一化 MSE）、
相位增量按照明加权 RMS 限制、测量锚定必须用同一完整模型的 Jacobian。

torch 在函数内延迟导入；``algorithm`` 层不 import 驱动/tools。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer

PANEL_H, PANEL_W = 1200, 1920
PANEL_PITCH_PX = 1920.0
WAVELENGTH_M = 1064e-9
PANEL_ILLUMINATION_CEILING_PX = 1200.0


def _torch():
    """返回 ``torch``；缺失时给出安装提示。"""
    try:
        import torch as _t
    except ImportError:  # pragma: no cover - 环境问题
        raise ImportError("differentiable_defocus_uniformity 需要 PyTorch: uv sync --extra ml") from None
    return _t


@dataclass(frozen=True)
class FrozenDefocusParameters:
    """检查点 ``result.json["parameters"]`` 的冻结参数。

    ``ramp_matrix`` 来自上游拟合 JSON（``config.json["fit"]`` 指向的文件），是
    **面板像元 → 相机像元**的线性映射；前向用它的逆把相机采样点映射回频域网格。
    它不在 ``result.json`` 里，必须由调用方一并提供。
    """

    defocus_diopter: float
    origin_xy: tuple[float, float]
    amplitude_center_xy: tuple[float, float]
    amplitude_width_xy: tuple[float, float]
    residual_noll_radians: Mapping[int, float]
    gain: float
    background: float
    sigma_ccd_px: float
    ramp_matrix: tuple[tuple[float, float], tuple[float, float]]
    fft_size: int = 1024
    pixel_samples: int = 3
    grid_size: int = 128

    @classmethod
    def from_dicts(cls, parameters: Mapping[str, Any], ramp_matrix: Sequence[Sequence[float]], *,
                   fft_size: int = 1024, pixel_samples: int = 3, grid_size: int = 128) -> "FrozenDefocusParameters":
        residual = {int(k): float(v) for k, v in dict(parameters["residual_noll_radians"]).items()}
        matrix = tuple(tuple(float(v) for v in row) for row in ramp_matrix)
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError("ramp_matrix must be 2x2")
        return cls(
            defocus_diopter=float(parameters["defocus_diopter"]),
            origin_xy=tuple(float(v) for v in parameters["origin"]),
            amplitude_center_xy=tuple(float(v) for v in parameters["amplitude_center"]),
            amplitude_width_xy=tuple(float(v) for v in parameters["amplitude_width"]),
            residual_noll_radians=residual,
            gain=float(parameters["gain"]),
            background=float(parameters["background"]),
            sigma_ccd_px=float(parameters["sigma_ccd_px"]),
            ramp_matrix=matrix,
            fft_size=int(fft_size), pixel_samples=int(pixel_samples), grid_size=int(grid_size),
        )

    @property
    def diopter_scale(self) -> float:
        """Zernike Noll 4 单位系数对应的屈光度标度（与拟合脚本同一公式）。"""
        return float(np.pi * (960 * 8e-6) ** 2 / (2 * np.sqrt(3) * WAVELENGTH_M))

    def to_dict(self) -> dict[str, Any]:
        return dict(defocus_diopter=self.defocus_diopter, origin=list(self.origin_xy),
                    amplitude_center=list(self.amplitude_center_xy),
                    amplitude_width=list(self.amplitude_width_xy),
                    residual_noll_radians={str(k): v for k, v in self.residual_noll_radians.items()},
                    gain=self.gain, background=self.background, sigma_ccd_px=self.sigma_ccd_px,
                    ramp_matrix=[list(r) for r in self.ramp_matrix], fft_size=self.fft_size,
                    pixel_samples=self.pixel_samples, grid_size=self.grid_size)


def load_frozen_parameters(checkpoint_dir: str | Path, *, fft_size: int = 1024,
                           pixel_samples: int = 3) -> FrozenDefocusParameters:
    """从 refit 检查点目录读出冻结参数（沿 ``config.json`` 取上游 ``ramp_matrix``）。

    Raises:
        FileNotFoundError: 链上任一环缺失。
        KeyError: 参数缺字段。
    """
    ckpt = Path(checkpoint_dir)
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    sampling_cfg = json.loads((Path(cfg["source"]) / "config.json").read_text(encoding="utf-8"))
    upstream = json.loads(Path(sampling_cfg["fit"]).read_text(encoding="utf-8"))
    result = json.loads((ckpt / "result.json").read_text(encoding="utf-8"))
    return FrozenDefocusParameters.from_dicts(result["parameters"], upstream["ramp_matrix"],
                                              fft_size=fft_size, pixel_samples=pixel_samples)


class FrozenDefocusForward:
    """冻结物理前向：128 控制网格相位 → 相机采样网格上的强度。

    数学与 ``PhysicalForwardFit.prediction()`` + refit 后处理（高斯展宽 / 增益 /
    加性背景）逐项一致；类自身不持有可训练参数，唯一叶子是传入的相位。
    """

    def __init__(self, params: FrozenDefocusParameters, camera_xy: npt.NDArray[np.floating],
                 *, device: str | None = None) -> None:
        torch = _torch()
        xy = np.asarray(camera_xy, dtype=np.float64)
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError(f"camera_xy must be (H, W, 2), got {xy.shape}")
        self.params = params
        self.camera_xy = xy
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        size = int(params.grid_size)
        self.pitch = PANEL_PITCH_PX / size
        y, x = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                              torch.arange(size, dtype=torch.float32), indexing="ij")
        self.panel_x = (x * self.pitch).to(self._device)
        self.panel_y = (y * self.pitch).to(self._device)
        self.panel_mask = (self.panel_y < PANEL_ILLUMINATION_CEILING_PX).float()
        center = torch.tensor(params.amplitude_center_xy, dtype=torch.float32, device=self._device)
        width = torch.tensor(params.amplitude_width_xy, dtype=torch.float32, device=self._device)
        self.amplitude = (torch.exp(-((self.panel_x - center[0]) / width[0]) ** 2
                                    - ((self.panel_y - center[1]) / width[1]) ** 2)
                          * self.panel_mask)
        self.z4 = self._mode_basis(4)
        self.residual_modes = sorted(int(k) for k in params.residual_noll_radians)
        self.residual_basis = (torch.stack([self._mode_basis(j) for j in self.residual_modes])
                               if self.residual_modes else None)
        self.residual_values = (torch.tensor([float(params.residual_noll_radians[j]) for j in self.residual_modes],
                                             dtype=torch.float32, device=self._device)
                                if self.residual_modes else None)
        self.origin = torch.tensor(params.origin_xy, dtype=torch.float32, device=self._device)
        inverse = np.linalg.inv(np.asarray(params.ramp_matrix, dtype=np.float64))
        self.inverse = torch.tensor(inverse, dtype=torch.float32, device=self._device)
        self._xy_t = torch.tensor(xy, dtype=torch.float32, device=self._device)
        self._grid_offsets = []
        for iy in range(params.pixel_samples):
            for ix in range(params.pixel_samples):
                offset = ((ix + 0.5) / params.pixel_samples - 0.5,
                          (iy + 0.5) / params.pixel_samples - 0.5)
                self._grid_offsets.append(torch.tensor(offset, dtype=torch.float32, device=self._device))
        self._blaze_kernel = self._build_blaze_kernel(params.sigma_ccd_px)

    def _mode_basis(self, noll: int):
        """单个 Noll 模式在该控制网格上的基（物理中心 952.5 面板像元）。"""
        torch = _torch()
        from scipy.ndimage import map_coordinates

        from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

        size = int(self.params.grid_size)
        full = np.nan_to_num(generate_zernike_phase({noll: 1.0}, (1920, 1920)))
        yy, xx = np.mgrid[:size, :size] * (PANEL_PITCH_PX / size)
        sampled = map_coordinates(full, [yy + 7, xx + 7], order=1, mode="constant").astype(np.float32)
        return torch.tensor(sampled, dtype=torch.float32, device=self._device)

    def _build_blaze_kernel(self, sigma_px: float):
        torch = _torch()
        if sigma_px <= 0:
            return None
        grid = torch.arange(-12, 13, dtype=torch.float32, device=self._device)
        kernel = torch.exp(-0.5 * (grid / (sigma_px / 2.0)) ** 2)
        kernel = kernel / kernel.sum()
        return (kernel[:, None] * kernel[None, :])[None, None]

    def total_phase_grid(self, phase: Any, *, focus_phase: Any = None, carrier_phase: Any = None):
        """``phi_panel = phi + phi_focus + phi_carrier``（raw 弧度，不包裹）。"""
        torch = _torch()

        def _coerce(value):
            if value is None:
                return None
            return value.to(self._device) if torch.is_tensor(value) else torch.as_tensor(
                value, dtype=torch.float32, device=self._device)

        total = _coerce(phase)
        for extra in (focus_phase, carrier_phase):
            coerced = _coerce(extra)
            if coerced is not None:
                total = total + coerced
        return total

    def predict(self, phase: Any, *, focus_phase: Any = None, carrier_phase: Any = None):
        """返回相机采样网格上的强度（对 ``phase`` 可微）。"""
        torch = _torch()
        from ao_shaping.utils.wavefront.fraunhofer import focal_intensity

        total = self.total_phase_grid(phase, focus_phase=focus_phase, carrier_phase=carrier_phase)
        if total.ndim == 2:
            total = total[None]
        p = self.params
        phase_all = total + p.defocus_diopter * p.diopter_scale * self.z4
        if self.residual_basis is not None:
            phase_all = phase_all + torch.einsum("i,ijk->jk", self.residual_values, self.residual_basis)
        field = self.amplitude[None] * torch.exp(1j * phase_all)
        intensity = focal_intensity(field, p.fft_size)
        intensity = intensity * (self.pitch ** 2 * p.fft_size / (15 ** 2 * 512)) ** 2
        grids = []
        for offset in self._grid_offsets:
            frequencies = (self._xy_t + offset - self.origin) @ self.inverse.T
            indices = frequencies * (p.fft_size * self.pitch) + p.fft_size // 2
            grids.append(2 * indices / (p.fft_size - 1) - 1)
        sampled = sum(torch.nn.functional.grid_sample(
            intensity[:, None], grid[None].expand(len(intensity), -1, -1, -1),
            align_corners=True, padding_mode="zeros")[:, 0] for grid in grids) / len(grids)
        if self._blaze_kernel is not None:
            sampled = torch.nn.functional.conv2d(sampled[:, None], self._blaze_kernel, padding=12)[:, 0]
        return sampled * float(p.gain) + float(p.background)

    def illumination_weights(self) -> npt.NDArray[np.floating]:
        """照明加权用的权重 ``A²``（归一化到和为一）。"""
        weights = (self.amplitude.detach().cpu().numpy().astype(np.float64)) ** 2
        total = weights.sum()
        return weights / total if total > 0 else weights

    def sample_grid_index(self, x_ccd: float, y_ccd: float) -> tuple[int, int]:
        """相机坐标 (x, y) → 采样网格下标 (col, row)。"""
        xy = self.camera_xy
        col = int(round((x_ccd - xy[0, 0, 0]) / (xy[0, 1, 0] - xy[0, 0, 0])))
        row = int(round((y_ccd - xy[0, 0, 1]) / (xy[1, 0, 1] - xy[0, 0, 1])))
        if not (0 <= row < xy.shape[0] and 0 <= col < xy.shape[1]):
            raise ValueError(f"camera point ({x_ccd}, {y_ccd}) outside sampled region")
        return col, row

    def sample_step_ccd(self) -> tuple[float, float]:
        xy = self.camera_xy
        return (float(xy[0, 1, 0] - xy[0, 0, 0]), float(xy[1, 0, 1] - xy[0, 0, 1]))


@dataclass
class UniformityObjective:
    """固定掩膜内的 CV 目标（+ 可选形状项与相位平滑项）。

    ``mask_box`` 是完整方框、``mask_core`` 是内缩核心区，两者都在采样网格上且
    必须在优化前冻结；排序以 ``mask_core`` 为准。
    """

    mask_box: npt.NDArray[np.bool_]
    mask_core: npt.NDArray[np.bool_]
    w_shape: float = 0.0
    w_phase: float = 0.01
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.mask_box = np.asarray(self.mask_box, dtype=bool)
        self.mask_core = np.asarray(self.mask_core, dtype=bool)
        if self.mask_box.shape != self.mask_core.shape:
            raise ValueError("mask_box and mask_core must share shape")
        if not self.mask_core.any():
            raise ValueError("mask_core is empty")
        if self.w_shape < 0 or self.w_phase < 0:
            raise ValueError("weights must be non-negative")

    def cv(self, image, mask=None):
        """掩膜内变异系数平方（对 ``image`` 可微）。"""
        torch = _torch()
        mask_t = torch.as_tensor(mask if mask is not None else self.mask_core, device=image.device)
        values = image[mask_t]
        mu = values.mean().clamp_min(self.eps)
        return ((values / mu - 1.0) ** 2).mean()

    def phase_smoothness(self, delta_phase):
        """``mean(1 - cos(相邻差))``，两轴平均。"""
        torch = _torch()
        dy = torch.cos(delta_phase[1:, :] - delta_phase[:-1, :])
        dx = torch.cos(delta_phase[:, 1:] - delta_phase[:, :-1])
        return 0.5 * ((1 - dy).mean() + (1 - dx).mean())

    def evaluate(self, image, delta_phase, *, shape_term=None):
        """返回 (总损失, 分量字典)。"""
        l_cv = self.cv(image)
        l_phase = self.phase_smoothness(delta_phase)
        l_shape = shape_term if shape_term is not None else 0.0
        total = l_cv + self.w_shape * l_shape + self.w_phase * l_phase
        return total, dict(l_cv=float(l_cv.detach()), l_phase=float(l_phase.detach()),
                           l_shape=float(l_shape.detach()) if hasattr(l_shape, "detach") else float(l_shape))


def build_masks(shape: tuple[int, int], centre_ij: tuple[int, int], box_px: int,
                inner_shrink_px: int, step_ccd_px: tuple[float, float]) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """按 CCD 像元尺寸构造方框/核心掩膜（冻结用一次）。"""
    col, row = centre_ij
    half_step = 0.5 * (step_ccd_px[0] + step_ccd_px[1])
    half = max(1, int(round(box_px / 2.0 / half_step)))
    inner = max(1, half - int(round(inner_shrink_px / half_step)))
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    box = (np.abs(xx - col) <= half) & (np.abs(yy - row) <= half)
    core = (np.abs(xx - col) <= inner) & (np.abs(yy - row) <= inner)
    return box, core


@dataclass
class CoordinateTransform:
    """面板↔相机的**实测**线性映射（带方向标记）。

    ``rows`` 的第 i 行描述「面板轴 i 的斜坡把光斑挪到相机哪个方向」，由 chirp-tilt
    标定（面板各轴一条已知周期斜坡）实测；``sign`` 为该轴符号。仓库里「面板 x 斜坡
    挪相机 y」的旧记法**不可假定** —— 主报告 §10 要求现场标定，本类以标定结果为准。
    """

    rows: tuple[tuple[float, float], tuple[float, float]] = ((8083.34, 0.17), (1.08, -8136.20))
    sign: tuple[float, float] = (1.0, -1.0)
    source: str = "default"

    @classmethod
    def from_axis_map(cls, axis_map: Mapping[str, Any], *, source: str = "chirp_tilt") -> "CoordinateTransform":
        """从 ``run_config.json`` 的 ``chirp_tilt_axis_map`` 构造。"""
        entries = [axis_map[k] for k in sorted(axis_map) if k.startswith("panel_axis")]
        if len(entries) != 2:
            raise ValueError("axis map must contain two panel axes")
        rows = []
        sign = []
        for entry in entries:
            vector = np.asarray([float(v) for v in entry["shift_vector_px"]], dtype=np.float64)
            slope = float(entry.get("slope_px_per_inv_period", 0.0))
            magnitude = float(np.hypot(*vector)) or abs(slope) or 1.0
            rows.append(tuple((vector / magnitude * (1.0 if slope >= 0 else -1.0)).tolist()))
            sign.append(1.0 if slope >= 0 else -1.0)
        return cls(rows=tuple(rows), sign=tuple(sign), source=source)

    def matrix(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.rows, dtype=np.float64)

    def to_camera(self, panel_vector: Sequence[float]) -> npt.NDArray[np.float64]:
        return self.matrix() @ np.asarray(panel_vector, dtype=np.float64)

    def to_panel(self, camera_vector: Sequence[float]) -> npt.NDArray[np.float64]:
        return np.linalg.solve(self.matrix(), np.asarray(camera_vector, dtype=np.float64))

    def compare_with_ramp_matrix(self, ramp_matrix: Sequence[Sequence[float]], *,
                                 tolerance: float = 0.35) -> dict[str, Any]:
        """核对现场标定与拟合 ``ramp_matrix`` 的方向（**不抛错**），返回结构化判定。

        量级只作参考：两套标定单位不同（实测是 px per 1/period，拟合是 px per 单位
        系数），不可直接相除，因此量级**不参与判定**。
        """
        fitted = np.asarray(ramp_matrix, dtype=np.float64)
        measured = np.abs(self.matrix())
        fitted_abs = np.abs(fitted)
        measured_axes = np.argmax(measured, axis=1).tolist()
        fitted_axes = np.argmax(fitted_abs, axis=1).tolist()
        same_axes = measured_axes == fitted_axes
        ratio = (measured.max(axis=1) / np.maximum(fitted_abs.max(axis=1), 1e-9)).tolist()
        measured_signs = np.sign(self.matrix()[np.arange(2), measured_axes]).tolist()
        fitted_signs = np.sign(fitted[np.arange(2), fitted_axes]).tolist()
        same_signs = measured_signs == fitted_signs
        verdict = ("axis map agrees with ramp_matrix" if (same_axes and same_signs)
                   else "axis map DISAGREES with ramp_matrix"
                        + (" (possible 90-degree swap)" if not same_axes else "")
                        + (" (sign mismatch)" if same_axes and not same_signs else ""))
        return dict(same_axes=same_axes, same_signs=same_signs,
                    within_scale=all(abs(r - 1.0) <= tolerance for r in ratio),
                    measured_axes=measured_axes, fitted_axes=fitted_axes,
                    measured_signs=measured_signs, fitted_signs=fitted_signs,
                    scale_ratio=ratio, scale_comparable=False, verdict=verdict, source=self.source)

    def check_against_ramp_matrix(self, ramp_matrix: Sequence[Sequence[float]], *,
                                  tolerance: float = 0.35) -> None:
        """同 :meth:`compare_with_ramp_matrix`，方向不一致时抛 ``ValueError``。"""
        verdict = self.compare_with_ramp_matrix(ramp_matrix, tolerance=tolerance)
        if not (verdict["same_axes"] and verdict["same_signs"]):
            raise ValueError(verdict["verdict"])


class DifferentiableUniformityOptimizer(IterativeOptimizer):
    """在**冻结**物理前向上做可微均匀度优化（类式 API：``update()`` 一步）。

    一步顺序（方案 §5）：合成总相位 → 前向 → 损失（模型或实测锚定）→ 反传 → Adam →
    把增量投影到照明加权 RMS ≤ cap → 去 piston → 去整体倾斜 → 记录。
    """

    def __init__(self, forward: FrozenDefocusForward, objective: UniformityObjective,
                 seed_phase: npt.NDArray[np.floating], *, phi_focus: npt.NDArray[np.floating] | None = None,
                 phi_carrier: npt.NDArray[np.floating] | None = None, lr: float = 0.01,
                 step_rms_cap: float = 0.1, device: str | None = None) -> None:
        torch = _torch()
        seed = np.asarray(seed_phase, dtype=np.float32)
        if seed.ndim != 2 or seed.shape[0] != seed.shape[1]:
            raise ValueError(f"seed_phase must be a square 2D grid, got {seed.shape}")
        if seed.shape[0] != forward.params.grid_size:
            raise ValueError(f"seed_phase size {seed.shape[0]} != model grid {forward.params.grid_size}")
        if lr <= 0 or step_rms_cap <= 0:
            raise ValueError("lr and step_rms_cap must be positive")
        self.forward = forward
        self.objective = objective
        self._device = torch.device(device or forward._device.type)
        self._seed = torch.tensor(seed, dtype=torch.float32, device=self._device)
        self._delta = torch.zeros_like(self._seed, requires_grad=True)
        self._accepted = torch.zeros_like(self._seed)
        self._focus = None if phi_focus is None else torch.as_tensor(phi_focus, dtype=torch.float32, device=self._device)
        self._carrier = None if phi_carrier is None else torch.as_tensor(phi_carrier, dtype=torch.float32, device=self._device)
        self._opt = torch.optim.Adam([self._delta], lr=lr)
        self._lr = float(lr)
        self._cap = float(step_rms_cap)
        self._weights = torch.tensor(forward.illumination_weights(), dtype=torch.float32, device=self._device)
        super().__init__(max_iterations=1)
        self._step = 0
        self._history: list[dict[str, Any]] = []
        self._converged = False

    @property
    def step(self) -> int:
        return self._step

    @property
    def delta_phase(self) -> npt.NDArray[np.floating]:
        return self._delta.detach().cpu().numpy()

    @property
    def accepted_phase(self) -> npt.NDArray[np.floating]:
        return self._accepted.detach().cpu().numpy()

    @property
    def current_phase(self) -> npt.NDArray[np.floating]:
        """总相位（seed + delta + focus + carrier），raw 弧度。"""
        torch = _torch()
        with torch.no_grad():
            return self.forward.total_phase_grid(self._seed + self._delta, focus_phase=self._focus,
                                                 carrier_phase=self._carrier).detach().cpu().numpy()

    @property
    def loss_history(self) -> list[dict[str, Any]]:
        return self._history

    @property
    def last_loss(self) -> float | None:
        return self._history[-1]["loss"] if self._history else None

    @property
    def converged(self) -> bool:
        return self._converged

    def weighted_rms(self, values) -> float:
        torch = _torch()
        v = values if torch.is_tensor(values) else torch.as_tensor(values, device=self._device)
        v = v.detach()
        return float(torch.sqrt((self._weights * v ** 2).sum() / self._weights.sum().clamp_min(1e-12)))

    def _project_increment(self) -> float:
        """把 ``delta - accepted`` 缩到照明加权 RMS ≤ cap；返回**投影前**的 RMS。"""
        torch = _torch()
        with torch.no_grad():
            increment = self._delta - self._accepted
            rms = float(torch.sqrt((self._weights * increment ** 2).sum() / self._weights.sum().clamp_min(1e-12)))
            if rms > self._cap and rms > 0:
                self._delta.copy_(self._accepted + increment * (self._cap / rms))
            return rms

    def _remove_piston_and_tilt(self) -> None:
        """去 piston + 去整体倾斜（倾角按照明加权最小二乘拟合，两轴各一次）。"""
        torch = _torch()
        size = self._delta.shape[0]
        grid = torch.arange(size, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            weights = self._weights
            self._delta.sub_((weights * self._delta).sum() / weights.sum().clamp_min(1e-12))
            for axis in (0, 1):
                dim = 1 if axis == 0 else 0
                w_axis = weights.sum(dim=dim)
                profile = (self._delta * weights).sum(dim=dim) / w_axis.clamp_min(1e-12)
                centre = (w_axis * grid).sum() / w_axis.sum().clamp_min(1e-12)
                centred = grid - centre
                slope = ((w_axis * centred * profile).sum()
                         / (w_axis * centred ** 2).sum().clamp_min(1e-12))
                self._delta.sub_(slope * centred[:, None] if axis == 0 else slope * centred[None, :])

    def update(self, measured_intensity: npt.NDArray[np.floating] | None = None) -> npt.NDArray[np.floating]:
        """一步：合成 → 前向 → 损失 → 反传 → Adam → 投影 → 去 piston/倾斜 → 记录。"""
        torch = _torch()
        if self._converged:
            return self.current_phase
        self._opt.zero_grad(set_to_none=True)
        prediction = self.forward.predict(self._seed + self._delta, focus_phase=self._focus,
                                          carrier_phase=self._carrier)
        if prediction.ndim == 3:
            prediction = prediction[0]
        if measured_intensity is None:
            loss, terms = self.objective.evaluate(prediction, self._delta)
            loss.backward()
            mode = "model"
        else:
            measured = torch.as_tensor(np.asarray(measured_intensity, dtype=np.float32), device=self._device)
            if measured.shape != prediction.shape:
                raise ValueError(f"measured_intensity shape {tuple(measured.shape)} != {tuple(prediction.shape)}")
            measured = measured.detach().clone().requires_grad_(True)
            loss_meas, terms = self.objective.evaluate(measured, self._delta.detach())
            loss_meas.backward()
            prediction.backward(gradient=measured.grad)
            loss = loss_meas
            mode = "measured"
        grad_norm = float(self._delta.grad.norm()) if self._delta.grad is not None else 0.0
        self._opt.step()
        rms_before = self._project_increment()
        self._remove_piston_and_tilt()
        self._step += 1
        record = dict(step=self._step, mode=mode, loss=float(loss.detach()), gradient_norm=grad_norm,
                      increment_rms_before_projection=rms_before,
                      increment_rms=self.weighted_rms(self._delta - self._accepted),
                      phase_rms=float(self._delta.detach().norm() / np.sqrt(self._delta.numel())),
                      **terms)
        self._history.append(record)
        self._record(record["loss"])
        return self.current_phase

    def accept(self) -> npt.NDArray[np.floating]:
        """把当前 delta 记为已接受（增量限制的参照随之推进）。"""
        torch = _torch()
        with torch.no_grad():
            self._accepted.copy_(self._delta)
        return self.accepted_phase
