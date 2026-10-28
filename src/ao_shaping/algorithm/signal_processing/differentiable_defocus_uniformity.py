"""离焦光路下的可微均匀度优化（冻结物理预测 + 固定掩膜 CV）。

把**已拟合的冻结物理预测模型**装成可微 torch 模块，并在其上提供均匀度目标与一个
类式优化器。约束来自 `report/shape_trials/defocus_differentiable_test_plan_20261023.md`
§3–§5：前向数学与检查点拟合口径逐项一致（实测复现存档预测相对 L1 9.976e-08）、
面板相位始终 raw 未包裹弧度、损失用固定掩膜内的 CV（不用全图峰值归一化 MSE）、
相位增量按照明加权 RMS 限制、测量锚定必须用同一完整模型的 Jacobian。

torch 在函数内延迟导入；``algorithm`` 层不 import 驱动/tools。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    PANEL_PITCH_PX,
    axial_plane_correction,
    camera_sample_grids,
    defocus_phase,
    diopter_scale,
    gaussian_amplitude,
    irradiance_scale,
    panel_grid,
    sample_intensity,
    subpixel_offsets,
)
from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer
from ao_shaping.utils.image.ranks import NDIM_SINGLE_CHANNEL

#: Side length of the panel-ramp matrix: it maps ``(x, y)`` panel slopes to ``(dx, dy)``
#: camera shifts, so both the row count and each row's length are this same number. It is a
#: matrix arity -- deliberately not the array-rank constant, which counts axes of an image
#: plane and says nothing about a 2x2 table.
_RAMP_MATRIX_SIDE = 2


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
    bias_counts: float = 0.
    slm_axial_offset_m: float = 0.
    axial_reference_curvature_diopter: float | None = None
    # --- optional, default-off physical-model extensions (report 20261027 §4) ---
    # All default to the identity so the historical linear baseline is preserved
    # bit-for-bit; the next hardware round fits these coefficients.
    slm_phase_a: float = 0.          # φ_eff = φ + a·sin φ + b·sin 2φ (SLM nonlinearity)
    slm_phase_b: float = 0.
    amplitude_residual_coeffs: tuple[float, ...] = ()  # flat Fourier coeffs (low-freq)
    monotonic_ccd_h: tuple[float, ...] = ()            # LUT log-derivative on [0, v_max]
    monotonic_ccd_v_max: float = 255.                  # response table upper bound

    @classmethod
    def from_dicts(cls, parameters: Mapping[str, Any], ramp_matrix: Sequence[Sequence[float]], *,
                   fft_size: int = 1024, pixel_samples: int = 3, grid_size: int = 128) -> "FrozenDefocusParameters":
        residual = {int(k): float(v) for k, v in dict(parameters["residual_noll_radians"]).items()}
        matrix = tuple(tuple(float(v) for v in row) for row in ramp_matrix)
        if len(matrix) != _RAMP_MATRIX_SIDE or any(len(row) != _RAMP_MATRIX_SIDE for row in matrix):
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
            bias_counts=float(parameters.get("bias_counts", 0.)),
            slm_axial_offset_m=float(parameters.get("slm_axial_offset_m", 0.)),
            axial_reference_curvature_diopter=(float(parameters["axial_reference_curvature_diopter"])
                if parameters.get("axial_reference_curvature_diopter") is not None else None),
            slm_phase_a=float(parameters.get("slm_phase_a", 0.)),
            slm_phase_b=float(parameters.get("slm_phase_b", 0.)),
            amplitude_residual_coeffs=tuple(float(c) for c in parameters.get("amplitude_residual_coeffs", ())),
            monotonic_ccd_h=tuple(float(h) for h in parameters.get("monotonic_ccd_h", ())),
            monotonic_ccd_v_max=float(parameters.get("monotonic_ccd_v_max", 255.)),
        )

    @property
    def diopter_scale(self) -> float:
        """Zernike Noll 4 单位系数对应的屈光度标度（共享前向核心同一公式）。"""
        return diopter_scale()

    def to_dict(self) -> dict[str, Any]:
        return dict(defocus_diopter=self.defocus_diopter, origin=list(self.origin_xy),
                    amplitude_center=list(self.amplitude_center_xy),
                    amplitude_width=list(self.amplitude_width_xy),
                    residual_noll_radians={str(k): v for k, v in self.residual_noll_radians.items()},
                    gain=self.gain, background=self.background, sigma_ccd_px=self.sigma_ccd_px,
                    ramp_matrix=[list(r) for r in self.ramp_matrix], fft_size=self.fft_size,
                    pixel_samples=self.pixel_samples, grid_size=self.grid_size, bias_counts=self.bias_counts,
                    slm_axial_offset_m=self.slm_axial_offset_m,
                    axial_reference_curvature_diopter=self.axial_reference_curvature_diopter,
                     slm_phase_a=self.slm_phase_a, slm_phase_b=self.slm_phase_b,
                     amplitude_residual_coeffs=tuple(self.amplitude_residual_coeffs),
                     monotonic_ccd_h=tuple(self.monotonic_ccd_h),
                     monotonic_ccd_v_max=self.monotonic_ccd_v_max)

    def effective_parameters(self):
        """Absorb input-plane offset into the identifiable intensity parameters.

        Use before fitting fresh geometry or synthesizing a seed so an axial
        offset is never applied twice. The original checkpoint remains intact.
        """
        if self.slm_axial_offset_m == 0:
            return self
        reference = (self.defocus_diopter if self.axial_reference_curvature_diopter is None
                     else self.axial_reference_curvature_diopter)
        d, scale = axial_plane_correction(reference, self.slm_axial_offset_m)
        return replace(self, defocus_diopter=self.defocus_diopter+d-reference, gain=self.gain/scale**2,
                       ramp_matrix=tuple(tuple(v*scale for v in row) for row in self.ramp_matrix),
                       slm_axial_offset_m=0., axial_reference_curvature_diopter=d)


def load_frozen_parameters(checkpoint_dir: str | Path, *, fft_size: int = 1024,
                           pixel_samples: int = 3) -> FrozenDefocusParameters:
    """从 refit 检查点目录读出冻结参数（沿 ``config.json`` 取上游 ``ramp_matrix``）。

    Raises:
        FileNotFoundError: 链上任一环缺失。
        KeyError: 参数缺字段。
    """
    ckpt = Path(checkpoint_dir)
    standalone = ckpt if ckpt.is_file() else ckpt / "frozen_parameters.json"
    if standalone.is_file():
        parameters = json.loads(standalone.read_text(encoding="utf-8"))
        return FrozenDefocusParameters.from_dicts(
            parameters, parameters["ramp_matrix"],
            fft_size=int(parameters.get("fft_size", fft_size)),
            pixel_samples=int(parameters.get("pixel_samples", pixel_samples)),
            grid_size=int(parameters.get("grid_size", 128)),
        )
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    sampling_cfg = json.loads((Path(cfg["source"]) / "config.json").read_text(encoding="utf-8"))
    upstream = json.loads(Path(sampling_cfg["fit"]).read_text(encoding="utf-8"))
    result = json.loads((ckpt / "result.json").read_text(encoding="utf-8"))
    return FrozenDefocusParameters.from_dicts(result["parameters"], upstream["ramp_matrix"],
                                              fft_size=fft_size, pixel_samples=pixel_samples)


class FrozenDefocusForward:
    """冻结物理预测：128 控制网格相位 → 相机采样网格上的强度。

    数学与 ``PhysicalForwardFit.prediction()`` + refit 后处理（高斯展宽 / 增益 /
    加性背景）逐项一致；两者共用的面板几何、高斯幅度、相机采样网格与 FFT
    辐照度标度已抽到 ``defocus_forward_core``（issue #199 R2），本类只保留
    冻结侧特有的部分（1920² 上采样 Zernike 基、blaze/增益/背景后处理）。
    类自身不持有可训练参数，唯一叶子是传入的相位。
    """

    def __init__(self, params: FrozenDefocusParameters, camera_xy: npt.NDArray[np.floating],
                 *, device: str | None = None) -> None:
        torch = _torch()
        xy = np.asarray(camera_xy, dtype=np.float64)
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError(f"camera_xy must be (H, W, 2), got {xy.shape}")
        params = params.effective_parameters()
        self.params = params
        self.effective_defocus, self.axial_scale = axial_plane_correction(
            params.defocus_diopter, params.slm_axial_offset_m)
        self.exposure_ms = None
        self.camera_xy = xy
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        size = int(params.grid_size)
        self.pitch = PANEL_PITCH_PX / size
        self.panel_x, self.panel_y, self.panel_mask = panel_grid(torch, size, device=self._device)
        center = torch.tensor(params.amplitude_center_xy, dtype=torch.float32, device=self._device)
        width = torch.tensor(params.amplitude_width_xy, dtype=torch.float32, device=self._device)
        self.amplitude = gaussian_amplitude(torch, self.panel_x, self.panel_y, center, width,
                                            self.panel_mask)
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
        self._grid_offsets = subpixel_offsets(torch, params.pixel_samples, device=self._device)
        self._blaze_kernel = self._build_blaze_kernel(params.sigma_ccd_px)
        # --- optional, default-off physical-model extensions (report 20261027 §4) ---
        # Built only when the checkpoint actually carries non-identity coefficients,
        # so the historical linear baseline stays bit-for-bit identical.
        self._slm_phase = self._build_slm_phase_model(params)
        self._amp_residual = self._build_amplitude_residual(params)
        self._monotonic_ccd = self._build_monotonic_ccd_model(params)

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

    def _build_slm_phase_model(self, params: "FrozenDefocusParameters"):
        """Bounded periodic SLM phase nonlinearity (identity when a=b=0)."""
        if params.slm_phase_a == 0. and params.slm_phase_b == 0.:
            return None
        from ao_shaping.algorithm.signal_processing.slm_phase_nonlinearity import (
            slm_phase_nonlinearity,
        )
        return lambda phi: slm_phase_nonlinearity(phi, params.slm_phase_a, params.slm_phase_b)

    def _build_amplitude_residual(self, params: "FrozenDefocusParameters"):
        """Low-frequency illumination amplitude residual (identity when empty)."""
        if not params.amplitude_residual_coeffs:
            return None
        from ao_shaping.algorithm.signal_processing.amplitude_residual import (
            AmplitudeResidual,
        )
        model = AmplitudeResidual(grid_size=int(params.grid_size),
                                  n_modes=int(round((len(params.amplitude_residual_coeffs) ** 0.5 - 1) / 2)))
        model.set_coeffs(params.amplitude_residual_coeffs)
        panel_x = self.panel_x.detach().cpu().numpy().astype(np.float64)
        panel_y = self.panel_y.detach().cpu().numpy().astype(np.float64)
        torch = _torch()
        residual = torch.as_tensor(model.evaluate(panel_x, panel_y),
                                  dtype=torch.float32, device=self._device)
        return residual  # (grid, grid) additive log residual, applied via exp

    def _build_monotonic_ccd_model(self, params: "FrozenDefocusParameters"):
        """Monotonic CCD observation-chain response (identity when h empty)."""
        if not params.monotonic_ccd_h:
            return None
        from ao_shaping.algorithm.signal_processing.monotonic_response import (
            MonotonicResponse,
            MonotonicResponseConfig,
        )
        n = len(params.monotonic_ccd_h)
        cfg = MonotonicResponseConfig(n_bins=n, v_max=float(params.monotonic_ccd_v_max))
        return MonotonicResponse(cfg, np.asarray(params.monotonic_ccd_h, dtype=np.float64))

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

    def predict(self, phase: Any, *, focus_phase: Any = None, carrier_phase: Any = None, exposure_ms: Any = None):
        """返回相机采样网格上的强度（对 ``phase`` 可微）。"""
        torch = _torch()
        from ao_shaping.utils.wavefront.fraunhofer import focal_intensity

        total = self.total_phase_grid(phase, focus_phase=focus_phase, carrier_phase=carrier_phase)
        if total.ndim == NDIM_SINGLE_CHANNEL:
            total = total[None]
        p = self.params
        # Optional SLM command -> effective phase nonlinearity (report 20261027 §4 item 2).
        if self._slm_phase is not None:
            total = self._slm_phase(total)
        phase_all = defocus_phase(torch, total, self.effective_defocus, p.diopter_scale, self.z4,
                                  residual_values=self.residual_values,
                                  residual_basis=self.residual_basis)
        # Optional Gaussian illumination amplitude residual (report 20261027 §4 item 3).
        amplitude = self.amplitude[None]
        if self._amp_residual is not None:
            amplitude = amplitude * torch.exp(self._amp_residual[None])
        field = amplitude * torch.exp(1j * phase_all)
        intensity = focal_intensity(field, p.fft_size)
        intensity = intensity * irradiance_scale(self.pitch, p.fft_size) / self.axial_scale**2
        grids = camera_sample_grids(torch, self._xy_t, self.origin, self.inverse / self.axial_scale,
                                    self._grid_offsets, fft_size=p.fft_size, pitch=self.pitch)
        sampled = sample_intensity(torch, intensity, grids)
        if self._blaze_kernel is not None:
            sampled = torch.nn.functional.conv2d(sampled[:, None], self._blaze_kernel, padding=12)[:, 0]
        result = sampled * float(p.gain) + float(p.background)
        if p.bias_counts != 0:
            exposure = exposure_ms if exposure_ms is not None else self.exposure_ms
            if exposure is None:
                raise ValueError("bias_counts checkpoint requires exposure_ms")
            exposure = torch.as_tensor(exposure, dtype=result.dtype, device=result.device)
            if not bool(torch.isfinite(exposure).all()) or bool((exposure <= 0).any()):
                raise ValueError("exposure_ms must be finite and positive")
            if exposure.ndim == 1 and exposure.shape[0] == result.shape[0]:
                exposure = exposure[:, None, None]
            elif exposure.ndim != 0:
                raise ValueError("exposure_ms must be scalar or one value per phase")
            result = result + float(p.bias_counts)/exposure
        # Optional monotonic CCD observation-chain response (report 20261027 §4 item 1).
        if self._monotonic_ccd is not None:
            result = self._monotonic_ccd.forward(result)
        return result

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
        """掩膜内变异系数**平方** (对 ``image`` 可微)。

        定义 = :func:`~ao_shaping.utils.image.beam_metrics.masked_cv` 的平方
        (issue #199 R3 两层结构: numpy 原语是唯一口径, 本方法是保持可微的
        torch 薄封装; 一致性契约见
        ``tests/ao_shaping/algorithm/test_metric_layers_agree.py``)。均值经
        ``clamp_min(eps)`` 防护, 因此均值低于 ``eps`` 时与原语的 ``0.0``
        退化策略**有意**不同 —— 优化器需要有限的梯度。
        """
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


def recentre_window(centre_xy: tuple[float, float], *, half_span_px: float = 120.0,
                    step_px: float = 2.0) -> npt.NDArray[np.floating]:
    """以**实测零点**为中心重建模型采样窗（``camera_xy`` 的 (H, W, 2) 布局）。

    拟合数据集里的窗心对应采集当天的 U 支零点；换会话后零点会漂（2026-10-23 实测
    漂了约 72 px），沿用旧窗会让零点贴边、背景与 EE 分母不对称。评估与重拟合前
    都应以本次实测零点重建。

    Args:
        centre_xy: 本次实测零点 ``(x, y)``，相机像元。
        half_span_px: 单边跨度（121 点 × 2 px ⇒ 120）。
        step_px: 采样步长（与拟合口径一致，默认 2 px）。

    Returns:
        形状 ``(2*n+1, 2*n+1, 2)`` 的数组，``[..., 0]`` 是 x、``[..., 1]`` 是 y。
    """
    count = int(round(half_span_px / step_px))
    offsets = np.arange(-count, count + 1, dtype=np.float64) * step_px
    xs = float(centre_xy[0]) + offsets
    ys = float(centre_xy[1]) + offsets
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx, yy], axis=-1)


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
    """在**冻结**物理预测上做可微均匀度优化（类式 API：``update()`` 一步）。

    一步顺序（方案 §5）：合成总相位 → 前向 → 损失（模型或实测锚定）→ 反传 → Adam →
    把增量投影到照明加权 RMS ≤ cap → 去 piston → 去整体倾斜 → 记录。
    """

    def __init__(self, forward: FrozenDefocusForward, objective: UniformityObjective,
                 seed_phase: npt.NDArray[np.floating], *, phi_focus: npt.NDArray[np.floating] | None = None,
                 phi_carrier: npt.NDArray[np.floating] | None = None, lr: float = 0.01,
                 step_rms_cap: float = 0.1, device: str | None = None) -> None:
        torch = _torch()
        seed = np.asarray(seed_phase, dtype=np.float32)
        if seed.ndim != NDIM_SINGLE_CHANNEL or seed.shape[0] != seed.shape[1]:
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
