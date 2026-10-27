"""Fixed-camera multishape objectives, GS initialization and measured anchors.

Pure numerical helpers; no device or tools imports.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates


def target_at(xy, shape, centre, side):
    """Sample the canonical shape raster in physical CCD coordinates."""
    from ao_shaping.utils.image.target.patterns import create_target_shape

    if not np.isfinite(side) or side <= 0:
        raise ValueError("target side must be finite and positive")
    extent = side * (2 if shape == "rectangle" else 1)
    size = max(512, 2 * int(np.ceil((extent + 8) / 2)))
    middle = size / 2
    raster = create_target_shape(shape, (size, size), side=side, radius_ratio=side/size,
                                 aspect_ratio=2., center=(middle, middle))
    return map_coordinates(raster, [xy[..., 1]-centre[1]+middle, xy[..., 0]-centre[0]+middle],
                           order=1, mode="constant")


class ShapeObjective:
    """CV plus shape fidelity and window energy, on one frozen camera grid.

    ``cv2`` 是 :func:`~ao_shaping.utils.image.beam_metrics.masked_cv` 的**平方**
    (issue #199 R3: numpy 指标层 + torch 薄封装两层; 两层的一致性由
    ``tests/ao_shaping/algorithm/test_metric_layers_agree.py`` 钉住)。
    """

    def __init__(self, target, *, cv_only=False, formation=False):
        self.cv_only = bool(cv_only)
        self.formation = bool(formation)
        if self.cv_only and self.formation:
            raise ValueError("CV-only and formation objectives are mutually exclusive")
        self.target = np.asarray(target, dtype=np.float32)
        self.box = self.target > .5
        self.core = binary_erosion(self.box, iterations=2)
        if not self.core.any():
            raise ValueError("empty target core")

    def evaluate(self, image, delta_phase):
        import torch

        target = torch.as_tensor(self.target, device=image.device)
        core = image[torch.as_tensor(self.core, device=image.device)]
        # CV^2 == masked_cv(image, self.core)**2 (R3 两层结构); clamp_min 让
        # 均值趋零时损失有限、梯度不炸 (与 numpy 原语的 0.0 退化策略有意不同)。
        cv2 = ((core/core.mean().clamp_min(1e-9)-1)**2).mean()
        total = image.sum().clamp_min(1e-9)
        energy = image[torch.as_tensor(self.box, device=image.device)].sum()/total
        shape_loss = ((image/total-target/target.sum())**2).sum()*target.sum()
        smooth = .5*((1-torch.cos(delta_phase[1:]-delta_phase[:-1])).mean()
                     +(1-torch.cos(delta_phase[:, 1:]-delta_phase[:, :-1])).mean())
        if self.cv_only:
            loss = cv2+.001*smooth
        elif self.formation:
            loss = .05*cv2+8*(1-energy)+shape_loss+.001*smooth
        else:
            loss = cv2+2*(1-energy)+shape_loss+.001*smooth
        return loss, dict(l_cv=float(cv2.detach()), ee_roi=float(energy.detach()),
                          shape_loss=float(shape_loss.detach()))


def image_metrics(image, objective):
    """实测帧在目标区域上的指标 (issue #199 R3: 委托 beam_metrics 原语)。

    本函数曾是 "框内 CV + ROI 能量" 的四份内联口径之一; 现 CV/EE 走
    :func:`~ao_shaping.utils.image.beam_metrics.masked_cv` /
    :func:`~ao_shaping.utils.image.beam_metrics.masked_energy_fraction`。
    ``cv`` 的退化策略保持 ``None`` (transfer 的种子判定把"无功率"与"完全
    均匀"区分开), 与原语的 ``0.0`` 不同 —— 这是调用点契约, 不是指标定义。
    """
    from ao_shaping.utils.image.beam_metrics import masked_cv, masked_energy_fraction

    frame = np.asarray(image)
    core_values = frame[objective.core]
    mean = float(core_values.mean())
    return dict(cv=masked_cv(frame, objective.core) if mean > 0 else None, mean=mean,
                ee_roi=masked_energy_fraction(frame, objective.box, floor=1e-9),
                target_peak=float(frame[objective.box].max()))


def anchored_intensity(measured, prediction, baseline_prediction):
    """Use measured irradiance at the reference and the frozen local model change."""
    measured, prediction, baseline_prediction = (np.asarray(x, dtype=np.float32) for x in
                                                (measured, prediction, baseline_prediction))
    if measured.shape != prediction.shape or measured.shape != baseline_prediction.shape:
        raise ValueError("anchor and predictions must use the identical camera grid")
    return np.maximum(measured + prediction - baseline_prediction, 1e-8)



def fft_grid_target(forward, centre, side, shape):
    """Sample the shape target on the forward model's own FFT grid (CCD pixels).

    The FFT grid is the centred spectrum layout the model propagates to
    (``fftshift(fft2(ifftshift(field)))``), so index ``[i, j]`` of the returned
    array is the far-field sample at that spectral position. Every seed
    construction (GS, geometric transport, back-projection) must address the
    same grid, so this is the single place the mapping from spectrum index to
    CCD pixel is written down.
    """
    p = forward.params
    n = p.fft_size
    yy, xx = np.mgrid[:n, :n]
    frequency = np.stack((xx - n // 2, yy - n // 2), axis=-1) / (n * forward.pitch)
    camera_xy = frequency @ np.asarray(p.ramp_matrix).T + np.asarray(p.origin_xy)
    return target_at(camera_xy, shape, centre, side)


def gs_seed(forward, centre, side, shape):
    """Construct the target on FFT coordinates using the full measured matrix.

    Every tensor read out of ``forward`` goes through ``detach().cpu()`` first:
    the forward model may live on CUDA (``TransferSession.device``), and a
    ``.numpy()`` on a CUDA tensor raises. On CPU this is a no-op, so the numbers
    are unchanged.
    """
    from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton

    p = forward.params
    n, g = p.fft_size, p.grid_size
    half = (n - g) // 2
    target = fft_grid_target(forward, centre, side, shape)
    amplitude = np.zeros((n, n))
    amplitude[half:half + g, half:half + g] = forward.amplitude.detach().cpu().numpy()
    result = gerchberg_saxton(amplitude, np.sqrt(target), iterations=80, propagation="fft")
    static = p.defocus_diopter * p.diopter_scale * forward.z4.detach().cpu().numpy()
    if forward.residual_basis is not None:
        static = static + np.einsum("i,ijk->jk", forward.residual_values.detach().cpu().numpy(),
                                    forward.residual_basis.detach().cpu().numpy())
    return result.phase[half:half + g, half:half + g] - static

