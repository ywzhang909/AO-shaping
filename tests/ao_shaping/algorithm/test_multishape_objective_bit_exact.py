"""``ShapeObjective.evaluate`` must stay bit-identical to its boolean-mask form.

2026-11-10 的调度优化把 ``evaluate`` 的三处热点换掉了：掩膜索引从
``image[bool_mask]``（内部 ``nonzero`` → 主机同步）改成 ``image.reshape(-1)[idx]``
（扁平索引），7 个诊断标量从逐个 ``float()``（7 次主机同步）合并成一次
``torch.stack([...]).tolist()``。两者只该省同步，不该改数值。

这里把**改动前**的实现逐字复制成参考（``_reference_evaluate``），钉住三条：

* 损失与 7 个诊断标量逐位相等（``==``，不是 ``approx``）；
* 对图像的梯度逐位相等；
* 外部**原地**改写 ``objective.core`` / ``objective.box`` 后仍然逐位相等——
  扁平索引按内容失效，缓存不得返回过期掩膜。

梯度那条不是白给的：布尔掩膜索引的 backward（``masked_scatter``）与扁平索引的
backward（``index_add``）若共用同一个 reshape 视图，两个 gather 的梯度会先在视图里
合并再回流，累加顺序一变，总梯度的末位就变。所以每个 gather 必须各自
``reshape``（单消费者节点），回流结构才与布尔掩膜版一一对应。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective

H = W = 96


def _target() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (((yy - 48) ** 2 + (xx - 48) ** 2) <= 30 ** 2).astype(np.float32)


def _image() -> torch.Tensor:
    rng = np.random.default_rng(20261110)
    im = rng.random((H, W), dtype=np.float32)
    im[40:56, 40:56] += 3.0
    return torch.as_tensor(im)


def _delta() -> torch.Tensor:
    rng = np.random.default_rng(4242)
    return torch.as_tensor(rng.standard_normal((32, 32)).astype(np.float32) * 0.01)


def _reference_evaluate(objective, image, delta_phase):
    """改动前（2026-11-10 之前）的 ``evaluate``，逐字复制。

    取自 ``multishape_objective.py`` 在 ``297408ec`` 之前的版本：布尔掩膜索引、
    每次调用新建 target 张量、7 个诊断标量各自 ``float()``。
    """
    import torch
    target = torch.as_tensor(objective.target, device=image.device)
    core = image[torch.as_tensor(objective.core, device=image.device)]
    # CV^2 == masked_cv(image, self.core)**2 (R3 两层结构); clamp_min 让
    # 均值趋零时损失有限、梯度不炸 (与 numpy 原语的 0.0 退化策略有意不同)。
    cv2 = ((core/core.mean().clamp_min(1e-9)-1)**2).mean()
    total = image.sum().clamp_min(1e-9)
    energy = image[torch.as_tensor(objective.box, device=image.device)].sum()/total
    shape_loss = ((image/total-target/target.sum())**2).sum()*target.sum()
    smooth = .5*((1-torch.cos(delta_phase[1:]-delta_phase[:-1])).mean()
                +(1-torch.cos(delta_phase[:, 1:]-delta_phase[:, :-1])).mean())
    # Formation-specific terms (report 20261027 §3: avoid only suppressing hot spots).
    l_partition = torch.tensor(0.0, device=image.device)
    l_peripheral = torch.tensor(0.0, device=image.device)
    l_floor = torch.tensor(0.0, device=image.device)
    l_boundary = torch.tensor(0.0, device=image.device)
    if objective.formation:
        # Partition coverage: penalize imbalance across sub-regions.
        if objective._partition_masks:
            partition_energies = []
            for pm in objective._partition_masks:
                pm_t = torch.as_tensor(pm, device=image.device)
                pe = image[pm_t].sum() / total.clamp_min(1e-9)
                partition_energies.append(pe)
            if partition_energies:
                pe_tensor = torch.stack(partition_energies)
                l_partition = pe_tensor.std() if pe_tensor.numel() > 1 else torch.tensor(0.0, device=image.device)
        # Peripheral band energy: ensure target edges are filled.
        if objective._peripheral_mask.any():
            pm_t = torch.as_tensor(objective._peripheral_mask, device=image.device)
            peripheral_e = image[pm_t].sum() / total.clamp_min(1e-9)
            l_peripheral = torch.relu(0.1 - peripheral_e)
        # Net signal floor: ensure minimum signal level in target.
        box_mean = image[torch.as_tensor(objective.box, device=image.device)].mean()
        l_floor = torch.relu(objective.signal_floor - box_mean)
        # Boundary error: penalize intensity variation at target boundary.
        if objective._boundary_mask.any():
            bm_t = torch.as_tensor(objective._boundary_mask, device=image.device)
            boundary_vals = image[bm_t]
            if boundary_vals.numel() > 0:
                l_boundary = boundary_vals.std() / boundary_vals.mean().clamp_min(1e-9)
    w = objective.weights
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


def _objective(mode: str) -> ShapeObjective:
    if mode == "cv":
        return ShapeObjective(_target(), cv_only=True)
    if mode == "formation":
        return ShapeObjective(_target(), formation=True)
    return ShapeObjective(_target())


@pytest.mark.parametrize("mode", ["cv", "shape", "formation"])
def test_evaluate_is_bit_identical_to_boolean_mask_reference(mode: str) -> None:
    """损失、诊断标量与梯度必须与布尔掩膜版逐位相等。"""
    objective = _objective(mode)
    image = _image()
    delta = _delta()

    loss_ref, diag_ref = _reference_evaluate(objective, image, delta)
    loss_new, diag_new = objective.evaluate(image, delta)

    assert float(loss_new) == float(loss_ref)
    assert sorted(diag_new) == sorted(diag_ref)
    for key in diag_ref:
        assert diag_new[key] == diag_ref[key], key

    img_ref = image.clone().requires_grad_(True)
    img_new = image.clone().requires_grad_(True)
    _reference_evaluate(objective, img_ref, delta)[0].backward()
    objective.evaluate(img_new, delta)[0].backward()
    assert torch.equal(img_new.grad, img_ref.grad)
    assert float(img_new.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("mode", ["cv", "formation"])
def test_in_place_mask_edit_invalidates_the_cached_index(mode: str) -> None:
    """外部原地改写 ``core`` / ``box`` 后，结果必须跟参考（读新掩膜）一致。"""
    objective = _objective(mode)
    image = _image()
    delta = _delta()

    objective.core &= ~np.zeros_like(objective.core)  # no-op copy, keeps dtype/shape
    radius = np.hypot(*np.mgrid[0:H, 0:W][::-1])
    objective.core &= radius > 20.0
    objective.box = objective.box & (radius < 55.0)

    loss_ref, diag_ref = _reference_evaluate(objective, image, delta)
    loss_new, diag_new = objective.evaluate(image, delta)

    assert float(loss_new) == float(loss_ref)
    for key in diag_ref:
        assert diag_new[key] == diag_ref[key], key
