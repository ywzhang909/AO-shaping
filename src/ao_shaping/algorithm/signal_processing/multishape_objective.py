"""Fixed-camera multishape objectives, GS initialization and measured anchors.

Pure numerical helpers; no device or tools imports. The one import here,
``PANEL_PITCH_PX``, comes from :mod:`...defocus_forward_core`, which is itself
constants-only (it re-exports the ``bench_constants`` leaf), so this module stays
importable without torch, a device SDK or ``tools/``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    PANEL_PITCH_PX,
    SLM_PANEL_H,
    SLM_PANEL_W,
)

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


def target_at(xy, shape, centre, side, aspect_ratio: float = 2.0):
    """Sample the canonical shape raster in physical CCD coordinates.

    ``aspect_ratio`` only matters for an axis-aligned ``"rectangle"``: it is the
    long : short width ratio the raster is drawn with, and the sampling extent
    covers the **long** axis (``extent = side * aspect_ratio``). For every other
    shape the extent stays ``side`` regardless of the value passed, so callers
    that only ever pass the four positional arguments (``aspect_ratio`` keeps its
    2.0 default) observe no behaviour change.
    """
    from ao_shaping.utils.image.target.patterns import create_target_shape

    if not np.isfinite(side) or side <= 0:
        raise ValueError("目标边长必须是有限正数")
    extent = side * (aspect_ratio if shape == "rectangle" else 1)
    size = max(512, 2 * int(np.ceil((extent + 8) / 2)))
    middle = size / 2
    raster = create_target_shape(shape, (size, size), side=side, radius_ratio=side/size,
                                 aspect_ratio=aspect_ratio, center=(middle, middle))
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
            raise ValueError("仅 CV 目标与形成目标互斥")
        #: Which weight set ``set_weights`` validates against.
        self.mode = "cv" if self.cv_only else ("formation" if self.formation else "shape")
        #: Active loss weights; ``constant`` unless a schedule overrides them.
        self.weights = dict(DEFAULT_WEIGHTS[self.mode])
        self.target = np.asarray(target, dtype=np.float32)
        self.box = self.target > .5
        self.core = binary_erosion(self.box, iterations=2)
        if not self.core.any():
            raise ValueError("目标核心为空")
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
            raise ValueError(f"模式 {self.mode!r} 的未知权重：{unknown}；"
                             f"允许：{sorted(DEFAULT_WEIGHTS[self.mode])}")
        merged = dict(DEFAULT_WEIGHTS[self.mode])
        merged.update({key: float(value) for key, value in weights.items()})
        self.weights = merged

    def _flat_index(self, name: str, mask, device):
        """Flat row-major index of ``mask``'s True cells on ``device``, cached.

        ``image.flatten()[idx]`` gathers the same cells in the same row-major
        order as the boolean-mask indexing it replaces, so every sum/mean is
        bit-identical; but it avoids the ``nonzero`` host sync per call. The
        cache is keyed by content, so an external in-place edit of ``self.core``
        / ``self.box`` (done by callers) is picked up, never served stale.
        """
        import torch
        arr = np.asarray(mask, dtype=bool)
        cache = self.__dict__.setdefault("_index_cache", {})
        hit = cache.get(name)
        if hit is None or hit[0].shape != arr.shape or not np.array_equal(hit[0], arr) or hit[1] != str(device):
            flat = torch.as_tensor(np.flatnonzero(arr), dtype=torch.long, device=device)
            hit = (arr.copy(), str(device), flat)
            cache[name] = hit
        return hit[2]

    def _target_tensor(self, device):
        import torch
        cache = self.__dict__.setdefault("_target_cache", {})
        hit = cache.get(str(device))
        if hit is None or not np.array_equal(hit[0], self.target):
            hit = (self.target.copy(), torch.as_tensor(self.target, device=device))
            cache[str(device)] = hit
        return hit[1]

    def evaluate(self, image, delta_phase):
        import torch
        dev = image.device
        target = self._target_tensor(dev)
        # Each gather takes its OWN ``reshape(-1)`` view: the view node has a
        # single consumer, so its backward contributes to ``image.grad``
        # separately -- the same one-edge-per-gather structure the boolean-mask
        # indexing it replaces had.  A shared flat view would merge the
        # gradients of two gathers before they reach ``image`` and change the
        # accumulation order, i.e. the last bits of the total gradient.
        core = image.reshape(-1)[self._flat_index("core", self.core, dev)]
        # CV^2 == masked_cv(image, self.core)**2 (R3 两层结构); clamp_min 让
        # 均值趋零时损失有限、梯度不炸 (与 numpy 原语的 0.0 退化策略有意不同)。
        cv2 = ((core/core.mean().clamp_min(1e-9)-1)**2).mean()
        total = image.sum().clamp_min(1e-9)
        box_idx = self._flat_index("box", self.box, dev)
        energy = image.reshape(-1)[box_idx].sum()/total
        shape_loss = ((image/total-target/target.sum())**2).sum()*target.sum()
        smooth = .5*((1-torch.cos(delta_phase[1:]-delta_phase[:-1])).mean()
                    +(1-torch.cos(delta_phase[:, 1:]-delta_phase[:, :-1])).mean())
        # Formation-specific terms (report 20261027 §3: avoid only suppressing hot spots).
        zero = torch.zeros((), device=dev)
        l_partition = zero
        l_peripheral = zero
        l_floor = zero
        l_boundary = zero
        if self.formation:
            # Partition coverage: penalize imbalance across sub-regions.
            if self._partition_masks:
                partition_energies = []
                for i, pm in enumerate(self._partition_masks):
                    pe = image.reshape(-1)[self._flat_index(f"partition{i}", pm, dev)].sum() / total.clamp_min(1e-9)
                    partition_energies.append(pe)
                if partition_energies:
                    pe_tensor = torch.stack(partition_energies)
                    l_partition = pe_tensor.std() if pe_tensor.numel() > 1 else zero
            # Peripheral band energy: ensure target edges are filled.
            if self._peripheral_mask.any():
                peripheral_e = image.reshape(-1)[self._flat_index("peripheral", self._peripheral_mask, dev)].sum() / total.clamp_min(1e-9)
                l_peripheral = torch.relu(0.1 - peripheral_e)
            # Net signal floor: ensure minimum signal level in target.
            box_mean = image.reshape(-1)[box_idx].mean()
            l_floor = torch.relu(self.signal_floor - box_mean)
            # Boundary error: penalize intensity variation at target boundary.
            if self._boundary_mask.any():
                boundary_vals = image.reshape(-1)[self._flat_index("boundary", self._boundary_mask, dev)]
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
        # One host transfer for all diagnostics (was 7 separate syncs).
        scalars = torch.stack([cv2.detach(), energy.detach(), shape_loss.detach(),
                               l_partition.detach(), l_peripheral.detach(),
                               l_floor.detach(), l_boundary.detach()]).tolist()
        return loss, dict(l_cv=scalars[0], ee_roi=scalars[1], shape_loss=scalars[2],
                          l_partition=scalars[3], l_peripheral=scalars[4],
                          l_floor=scalars[5], l_boundary=scalars[6])


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
                target_peak=float(frame[objective.box].max()),
                target_pearson=target_pearson(frame, objective.target))


def target_pearson(image, target):
    """实测帧与目标掩膜的 Pearson 相关 (整形相似度, 无量纲)。

    为什么要整窗而不是只在 ``objective.box`` 内算: 方斑目标的掩膜在框内恒为
    1, 框内方差为 0, Pearson 在那里**无定义**; 三角/矩形目标同理, 掩膜在框内
    也只有 0/1 两档。真正要回答的是「实测远场像不像这个形状」——落在目标之外
    的光同样属于答案, 所以相关在全窗口 (相机窗口 = 目标掩膜同一网格) 上算。

    返回 ``None`` 而不是 0.0: 全暗帧 (无光) 的方差为 0, 相关无定义, 与
    ``cv=None`` 的退化约定一致 —— 0.0 会被读成「测了, 但完全不像」。

    归一化不改变本值 (Pearson 对仿射缩放不变), 保留归一化是给读者把它与叠加
    图对照时口径一致; 因此本函数不对输入做任何截断或归一化。
    """
    frame = np.asarray(image, dtype=np.float64)
    ref = np.asarray(target, dtype=np.float64)
    if frame.shape != ref.shape:
        raise ValueError("图像与目标必须共用同一相机窗网格")
    if frame.size < 2 or frame.std() == 0 or ref.std() == 0:
        return None
    return float(np.corrcoef(frame.ravel(), ref.ravel())[0, 1])


def anchored_intensity(measured, prediction, baseline_prediction):
    """Use measured irradiance at the reference and the frozen local model change."""
    measured, prediction, baseline_prediction = (np.asarray(x, dtype=np.float32) for x in
                                                (measured, prediction, baseline_prediction))
    if measured.shape != prediction.shape or measured.shape != baseline_prediction.shape:
        raise ValueError("锚点与预测必须使用同一相机网格")
    return np.maximum(measured + prediction - baseline_prediction, 1e-8)



#: 可达性判据里给目标留的边界余量（px）。与 :func:`target_at` 加在光栅上的
#: ``extent + 8`` 是同一段余量，:mod:`...transfer.preflight` 的采样窗留白同值。
FFT_WINDOW_MARGIN_PX: float = 8.0

#: 采集网格下限（与 ``TransferSession.validate`` 的 ``capture_grid < 16`` 同义）。
CAPTURE_GRID_MIN: int = 16


def _fft_reach_px(grid_size: float, slope_px: float) -> float:
    """载波能把目标推到离零级多远（px）—— 可达公式的唯一一份实现。

    模型传播到以零级为中心的频谱窗 ``[-fmax, fmax]``，其中 ``fmax = 0.5 / pitch``、
    ``pitch = PANEL_PITCH_PX / grid_size``；载波斜率 ``slope_px``（每单位归一化频率
    的位移像素，实测即 ``TILT_SHIFT_SCALE``）在 ``fmax`` 处给出的位移就是可达半径。
    ``fft_size`` 不进来：它只改频谱采样密度，不改覆盖范围（这正是
    :func:`assert_target_within_fft_window` 忽略它的原因）。

    输入不合法（非有限、≤0）时返回 ``0.0``（= 什么都够不着），因为这条公式也用于
    操作员可见的提示文案，显示路径不许抛。
    """
    grid, slope = float(grid_size), abs(float(slope_px))
    if not (math.isfinite(grid) and math.isfinite(slope)) or grid <= 0 or slope <= 0:
        return 0.0
    return slope * (0.5 / (PANEL_PITCH_PX / grid))


def fft_max_side_px(grid_size: float, slope_px: float, *,
                    margin_px: float = FFT_WINDOW_MARGIN_PX,
                    offset_px: float = 0.0) -> float:
    """网格 ``grid_size`` 下仍放得进 FFT 可达窗的最大目标**边长**（px）。

    ``offset_px`` 是目标中心离零级的载波位移（见 :func:`carrier_offset_px`）：位移
    吃掉一段可达半径，剩下的才是目标能占的半跨。形状按包围盒算，矩形另乘长/短比
    （:func:`required_capture_grid` 负责这件事）。
    """
    room = _fft_reach_px(grid_size, slope_px) - float(margin_px) - float(offset_px)
    return max(0.0, 2.0 * room)


def min_calibrated_grid_px(half_extent_px: float, slope_px: float, *,
                           margin_px: float = FFT_WINDOW_MARGIN_PX,
                           offset_px: float = 0.0) -> float:
    """:func:`_fft_reach_px` 的反向：要够到 ``half_extent_px``，网格至少要多大（连续值）。

    返回值没有取到合法档（见 :func:`capture_grid_ladder`），只用于「至少 N」这类文案；
    真正要下发给采集的网格请用 :func:`required_capture_grid`。够不着（斜率非正）时
    返回 ``inf``。
    """
    reach = float(half_extent_px) + float(margin_px) + float(offset_px)
    slope = abs(float(slope_px))
    if not math.isfinite(reach) or reach <= 0 or slope <= 0:
        return 0.0
    return 2.0 * PANEL_PITCH_PX * reach / slope


def carrier_offset_px(period: float, slope_px: float) -> float:
    """载波周期 ``period``（panel px）给出的位移（px），即 ``slope / period``。

    与 ``bench_constants`` 的定义一致：2π 斜坡周期 P ⇒ 光斑移动
    ``TILT_SHIFT_SCALE / P`` 像素。周期非法时返回 ``0.0``（= 目标留在零级）。
    """
    p, slope = float(period), abs(float(slope_px))
    if not (math.isfinite(p) and math.isfinite(slope)) or p <= 0 or slope <= 0:
        return 0.0
    return slope / p


def capture_grid_ladder(panel_px: int = SLM_PANEL_W, rows_px: int = SLM_PANEL_H,
                        min_grid: int = CAPTURE_GRID_MIN) -> tuple[int, ...]:
    """能整铺面板的合法采集网格（升序）。

    规则与 ``TransferSession.validate`` 同源：``panel_px % grid == 0`` 且
    ``rows_px % (panel_px // grid) == 0``（单元格必须是整数像素且行向也铺得尽）。
    测试 ``test_transfer_reach.py`` 把它和 ``workflow`` 的检查逐档钉住。
    """
    return tuple(grid for grid in range(max(1, int(min_grid)), int(panel_px) + 1)
                 if panel_px % grid == 0 and rows_px % (panel_px // grid) == 0)


def target_extent_px(side: float, *, shapes: Sequence[str] = ("square",),
                     aspect_ratio: float = 2.0) -> float:
    """目标的包围盒边长（px）—— 可达判据用的**保守**边长，全工程只这一份规则。

    选了矩形时取长边 ``side * aspect_ratio``（与 :func:`assert_target_within_fft_window`
    同口径），否则取 ``side``。非法输入返回 ``0.0``（显示路径不许抛）。
    """
    side = float(side)
    if not math.isfinite(side) or side <= 0:
        return 0.0
    extent = side * (float(aspect_ratio) if "rectangle" in set(shapes) else 1.0)
    return extent if math.isfinite(extent) else 0.0


def required_capture_grid(side: float, slope_px: float, *,
                          shapes: Sequence[str] = ("square",),
                          aspect_ratio: float = 2.0,
                          period: float | None = None,
                          offset_px: float | None = None,
                          margin_px: float = FFT_WINDOW_MARGIN_PX,
                          grids: Sequence[int] | None = None) -> int | None:
    """本次目标够得着的最小**合法**采集网格；任何合法网格都够不着时返回 ``None``。

    取最挑剔的形状（:func:`target_extent_px`），位移取 ``offset_px``（实测位移），没给
    则按 ``period`` 外推，都没有则按目标留在零级算。判据就是 :func:`fft_max_side_px`，
    与硬保护 :func:`assert_target_within_fft_window` 用的是同一条公式。
    """
    extent = target_extent_px(side, shapes=shapes, aspect_ratio=aspect_ratio)
    if extent <= 0:
        return None
    if offset_px is None:
        offset_px = carrier_offset_px(period or 0.0, slope_px)
    ladder = tuple(grids) if grids is not None else capture_grid_ladder()
    for grid in ladder:
        if extent <= fft_max_side_px(grid, slope_px, margin_px=margin_px,
                                     offset_px=float(offset_px)) + 1e-9:
            return int(grid)
    return None


def assert_target_within_fft_window(forward, centre, side, shape,
                                    aspect_ratio: float = 2.0) -> None:
    """Raise ``ValueError`` if the target extent exceeds the model's FFT window.

    The model propagates to a centred spectrum window covering frequencies
    ``[-fmax, fmax]`` per axis, where ``fmax = 1 / (2 * pitch)`` and
    ``pitch = PANEL_PITCH_PX / grid_size``. That physical extent depends on the
    control-grid size but **not** on ``fft_size`` — the FFT size only changes
    the spectral sampling density, never the covered span. A target whose
    bounding box maps outside this window is clipped by :func:`target_at` into a
    full-window mask (all-ones when it fully contains the grid), so any seed
    built from it chases a shape the forward model cannot produce. Fail fast.

    The check is deliberately conservative: it tests the enclosing square of the
    shape raster (``centre ± extent/2`` per axis, ``extent = side * aspect_ratio``
    for a ``"rectangle"`` else ``side``), so it never wrongly accepts a target
    that overflows the window; for a triangle the true bounding box is smaller.

    A model that exposes no reach geometry (``pitch`` / ``ramp_matrix`` /
    ``origin_xy``) cannot be checked, so the guard is a no-op there — every real
    :class:`FrozenDefocusForward` carries all three.
    """
    pitch = getattr(forward, "pitch", None)
    params = getattr(forward, "params", None)
    if pitch is None or params is None:
        return
    ramp = getattr(params, "ramp_matrix", None)
    origin_xy = getattr(params, "origin_xy", None)
    if ramp is None or origin_xy is None:
        return
    fmax = 0.5 / float(pitch)
    half = side * (aspect_ratio if shape == "rectangle" else 1.0) / 2.0
    inv_ramp = np.linalg.inv(np.asarray(ramp, dtype=float))
    corner = np.asarray(centre, dtype=float)
    origin = np.asarray(origin_xy, dtype=float)
    worst = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            freq = (corner + np.array([sx * half, sy * half]) - origin) @ inv_ramp
            worst = max(worst, float(np.abs(freq).max()))
    if worst > fmax:
        raise ValueError(_fft_window_error(pitch, ramp, centre, origin, side, shape,
                                           aspect_ratio, fmax, worst))


def _fft_window_error(pitch, ramp, centre, origin, side, shape, aspect_ratio,
                      fmax, worst) -> str:
    """硬保护的报错文案：判据之外再给出「这个网格最多放得下多大」和「要多宽的网格」。

    数字来自 :func:`_fft_reach_px` 那一份公式（这里用**实测**斜坡与**实测**零点偏移），
    所以操作员在读到报错的同一行就能改参数，不用再自己反推。任何一步算不出数都退回
    不带数字的原句 —— 报错路径本身绝不能再抛。
    """
    head = (
        f"目标边长 {side:g} px（形状 {shape!r}）超过物理预测模型的 FFT 网格可达窗口："
        f"该网格的频率覆盖只到 ±{fmax:.4f}，但目标角点已达 ±{worst:.4f}。"
    )
    fallback = "请减小目标边长，或改用更细的控制网格。"
    try:
        slope = float(np.max(np.linalg.norm(np.asarray(ramp, dtype=float), axis=1)))
        offset = float(np.max(np.abs(np.asarray(centre, dtype=float)
                                     - np.asarray(origin, dtype=float))))
        grid = PANEL_PITCH_PX / float(pitch)
        reach = _fft_reach_px(grid, slope)
        limit = fft_max_side_px(grid, slope, offset_px=offset)
        needed = required_capture_grid(side, slope, shapes=(shape,),
                                       aspect_ratio=aspect_ratio, offset_px=offset)
    except Exception:  # 显示路径兜底，绝不掩盖原始判据
        return head + fallback
    numbers = (f"该网格下目标围绕零级的载频偏移达 ±{reach:.0f} px，"
               f"故居中目标边长至多 {limit:.0f} px。")
    if needed is None:
        widest = capture_grid_ladder()[-1]
        numbers += (f"最大 {widest} 的采集网格也达不到 {side:g} px；"
                    f"请把目标边长保持在 "
                     f"{fft_max_side_px(widest, slope, offset_px=offset):.0f} px 以下。")
    else:
        numbers += f"请使用至少 {needed} 的采集网格。"
    return head + numbers + fallback


def fft_grid_target(forward, centre, side, shape):
    """Sample the shape target on the forward model's own FFT grid (CCD pixels).

    The FFT grid is the centred spectrum layout the model propagates to
    (``fftshift(fft2(ifftshift(field)))``), so index ``[i, j]`` of the returned
    array is the far-field sample at that spectral position. Every seed
    construction (GS, geometric transport, back-projection) must address the
    same grid, so this is the single place the mapping from spectrum index to
    CCD pixel is written down.
    """
    assert_target_within_fft_window(forward, centre, side, shape)
    p = forward.params
    n = p.fft_size
    yy, xx = np.mgrid[:n, :n]
    frequency = np.stack((xx - n // 2, yy - n // 2), axis=-1) / (n * forward.pitch)
    camera_xy = frequency @ np.asarray(p.ramp_matrix).T + np.asarray(p.origin_xy)
    return target_at(camera_xy, shape, centre, side)


_GS_SEED_ITERS = 80


def gs_seed(forward, centre, side, shape, *, device: str | None = None):
    """Construct the target on FFT coordinates using the full measured matrix.

    Every tensor read out of ``forward`` goes through ``detach().cpu()`` first:
    the forward model may live on CUDA (``TransferSession.device``), and a
    ``.numpy()`` on a CUDA tensor raises. On CPU this is a no-op, so the numbers
    are unchanged.

    ``device`` selects where the Gerchberg-Saxton loop runs:

    * ``None`` / ``"auto"`` -- ``"cuda"`` when ``torch.cuda.is_available()``
      else ``"cpu"`` (the production default);
    * ``"cuda"`` -- a torch-CUDA ``fft``-mode loop (this machine ~34x faster
      once warm);
    * ``"cpu"`` -- the numpy ``gerchberg_saxton`` path (the historical
      numbers, bit-for-bit, for reproducibility pins).

    The CUDA path reproduces the numpy path to ~1e-11 relative (the resulting
    CV is identical) because both run the same double-precision ``fft``-mode GS
    iterations; it exists to remove the one CPU-heavy step (``_GS_SEED_ITERS``
    FFT pairs on the ``fft_size`` grid) from an otherwise-CUDA pipeline.
    """
    from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton

    p = forward.params
    n, g = p.fft_size, p.grid_size
    half = (n - g) // 2
    target = fft_grid_target(forward, centre, side, shape)
    amplitude = np.zeros((n, n))
    amplitude[half:half + g, half:half + g] = forward.amplitude.detach().cpu().numpy()
    if _gs_seed_use_cuda(device):
        phase = _gs_seed_cuda(amplitude, np.sqrt(np.clip(target, 0.0, None)), n, g, half)
    else:
        result = gerchberg_saxton(amplitude, np.sqrt(target), iterations=_GS_SEED_ITERS,
                                  propagation="fft")
        phase = result.phase[half:half + g, half:half + g]
    static = p.defocus_diopter * p.diopter_scale * forward.z4.detach().cpu().numpy()
    if forward.residual_basis is not None:
        static = static + np.einsum("i,ijk->jk", forward.residual_values.detach().cpu().numpy(),
                                    forward.residual_basis.detach().cpu().numpy())
    return phase - static


def _gs_seed_use_cuda(device: str | None) -> bool:
    if device is not None and device != "auto":
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"device must be 'cpu', 'cuda', 'auto' or None, got {device!r}")
        return device == "cuda"
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gs_seed_cuda(amplitude, target_sqrt, n, g, half):
    """torch-CUDA ``fft``-mode GS loop; identical math to the numpy path."""
    import torch
    dev = torch.device("cuda")
    src = torch.as_tensor(amplitude, dtype=torch.complex128, device=dev)
    tgt = torch.as_tensor(target_sqrt, dtype=torch.complex128, device=dev)
    A = torch.fft.ifft2(torch.fft.ifftshift(tgt))
    for _ in range(_GS_SEED_ITERS):
        B = src * torch.exp(1j * torch.angle(A))
        C = torch.fft.fftshift(torch.fft.fft2(B))
        D = tgt * torch.exp(1j * torch.angle(C))
        A = torch.fft.ifft2(torch.fft.ifftshift(D))
    return torch.angle(A)[half:half + g, half:half + g].detach().cpu().numpy()