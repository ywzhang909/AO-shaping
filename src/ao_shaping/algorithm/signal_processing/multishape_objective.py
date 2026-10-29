"""Fixed-camera multishape objectives, GS initialization and measured anchors.

Pure numerical helpers; no device or tools imports.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates

#: Loss weights per objective mode (issue #213 item 4c). These are the values
#: that were hard-coded inline before 2026-10-29; making them a named dict is
#: behaviour-preserving. ``weight_schedule.staged_weights`` overrides a subset
#: per block -- only the optimisation path moves, the acceptance criteria stay
#: frozen (``report/shape_trials/multishape_scale/next_round_20261029.md`` §1.1).
#: Keys absent from a mode are not used by that mode's loss, and ``set_weights``
#: rejects them so a typo can never be a silent no-op.
DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "cv": {"cv2": 1.0, "smooth": .001},
    "shape": {"cv2": 1.0, "energy": 2.0, "shape": 1.0, "smooth": .001},
    "formation": {"cv2": .05, "energy": 8.0, "shape": 1.0, "smooth": .001,
                  "partition": .5, "peripheral": .5, "floor": 1.0, "boundary": .3},
}


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

    **formation 模式** (report 20261027 §3): 不仅压低中心亮点，还加入目标内
    分区覆盖、外围带能量、净信号下限与边界误差，避免"CV 降了但目标没填满"的
    假阳性。新增分量在 ``evaluate`` 返回的字典中以 ``l_partition`` /
    ``l_peripheral`` / ``l_floor`` / ``l_boundary`` 键暴露，便于诊断。
    """

    def __init__(self, target, *, cv_only=False, formation=False,
                 n_partitions=4, peripheral_px=3, signal_floor=0.01):
        self.cv_only = bool(cv_only)
        self.formation = bool(formation)
        if self.cv_only and self.formation:
            raise ValueError("CV-only and formation objectives are mutually exclusive")
        #: Which weight set ``set_weights`` validates against.
        self.mode = "cv" if self.cv_only else ("formation" if self.formation else "shape")
        #: Active loss weights; ``constant`` unless a schedule overrides them.
        self.weights = dict(DEFAULT_WEIGHTS[self.mode])
        self.target = np.asarray(target, dtype=np.float32)
        self.box = self.target > .5
        self.core = binary_erosion(self.box, iterations=2)
        if not self.core.any():
            raise ValueError("empty target core")
        self.n_partitions = int(n_partitions)
        self.peripheral_px = int(peripheral_px)
        self.signal_floor = float(signal_floor)
        # Precompute partition masks (target divided into n×n sub-regions).
        self._partition_masks = self._build_partitions()
        # Peripheral band: pixels within peripheral_px of the target boundary.
        self._peripheral_mask = self._build_peripheral()
        # Boundary mask: pixels on the target edge (for boundary error).
        self._boundary_mask = self._build_boundary()

    def _build_partitions(self):
        """Divide the target box into n×n sub-regions; return list of bool masks."""
        ys, xs = np.where(self.box)
        if len(ys) == 0:
            return []
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        masks = []
        for pi in range(self.n_partitions):
            for pj in range(self.n_partitions):
                y_lo = y_min + (y_max - y_min) * pi // self.n_partitions
                y_hi = y_min + (y_max - y_min) * (pi + 1) // self.n_partitions
                x_lo = x_min + (x_max - x_min) * pj // self.n_partitions
                x_hi = x_min + (x_max - x_min) * (pj + 1) // self.n_partitions
                m = self.box.copy()
                m[:y_lo, :] = False
                m[y_hi:, :] = False
                m[:, :x_lo] = False
                m[:, x_hi:] = False
                if m.any():
                    masks.append(m)
        return masks

    def _build_peripheral(self):
        """Pixels within peripheral_px of the target boundary (inside or outside)."""
        from scipy.ndimage import binary_dilation, binary_erosion
        inner = binary_erosion(self.box, iterations=self.peripheral_px)
        outer = binary_dilation(self.box, iterations=self.peripheral_px)
        return outer & ~inner

    def _build_boundary(self):
        """Pixels on the target edge (box minus eroded box)."""
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(self.box, iterations=1)
        return self.box & ~eroded

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Replace the active loss weights for this objective's mode.

        Only the keys ``DEFAULT_WEIGHTS[self.mode]`` already defines may be set;
        an unknown key raises instead of being silently ignored, because a
        mistyped weight would otherwise look like "the schedule did nothing".
        """
        unknown = sorted(set(weights) - set(DEFAULT_WEIGHTS[self.mode]))
        if unknown:
            raise ValueError(f"unknown weight(s) for mode {self.mode!r}: {unknown}; "
                             f"allowed: {sorted(DEFAULT_WEIGHTS[self.mode])}")
        merged = dict(DEFAULT_WEIGHTS[self.mode])
        merged.update({key: float(value) for key, value in weights.items()})
        self.weights = merged

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
        # Formation-specific terms (report 20261027 §3: avoid only suppressing hot spots).
        l_partition = torch.tensor(0.0, device=image.device)
        l_peripheral = torch.tensor(0.0, device=image.device)
        l_floor = torch.tensor(0.0, device=image.device)
        l_boundary = torch.tensor(0.0, device=image.device)
        if self.formation:
            # Partition coverage: penalize imbalance across sub-regions.
            if self._partition_masks:
                partition_energies = []
                for pm in self._partition_masks:
                    pm_t = torch.as_tensor(pm, device=image.device)
                    pe = image[pm_t].sum() / total.clamp_min(1e-9)
                    partition_energies.append(pe)
                if partition_energies:
                    pe_tensor = torch.stack(partition_energies)
                    l_partition = pe_tensor.std() if pe_tensor.numel() > 1 else torch.tensor(0.0, device=image.device)
            # Peripheral band energy: ensure target edges are filled.
            if self._peripheral_mask.any():
                pm_t = torch.as_tensor(self._peripheral_mask, device=image.device)
                peripheral_e = image[pm_t].sum() / total.clamp_min(1e-9)
                l_peripheral = torch.relu(0.1 - peripheral_e)
            # Net signal floor: ensure minimum signal level in target.
            box_mean = image[torch.as_tensor(self.box, device=image.device)].mean()
            l_floor = torch.relu(self.signal_floor - box_mean)
            # Boundary error: penalize intensity variation at target boundary.
            if self._boundary_mask.any():
                bm_t = torch.as_tensor(self._boundary_mask, device=image.device)
                boundary_vals = image[bm_t]
                if boundary_vals.numel() > 0:
                    l_boundary = boundary_vals.std() / boundary_vals.mean().clamp_min(1e-9)
        w = self.weights
        # One expression for all modes: a mode simply leaves the keys it does
        # not define at weight 0 (``cv`` mode has no energy/shape key). The
        # constants are byte-for-byte the ones that used to be inline here.
        loss = (w["cv2"]*cv2
                + w.get("energy", 0.0)*(1-energy)
                + w.get("shape", 0.0)*shape_loss
                + w["smooth"]*smooth
                + w.get("partition", 0.0)*l_partition
                + w.get("peripheral", 0.0)*l_peripheral
                + w.get("floor", 0.0)*l_floor
                + w.get("boundary", 0.0)*l_boundary)
        return loss, dict(l_cv=float(cv2.detach()), ee_roi=float(energy.detach()),
                          shape_loss=float(shape_loss.detach()),
                          l_partition=float(l_partition.detach()) if hasattr(l_partition, "detach") else float(l_partition),
                          l_peripheral=float(l_peripheral.detach()) if hasattr(l_peripheral, "detach") else float(l_peripheral),
                          l_floor=float(l_floor.detach()) if hasattr(l_floor, "detach") else float(l_floor),
                          l_boundary=float(l_boundary.detach()) if hasattr(l_boundary, "detach") else float(l_boundary))


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