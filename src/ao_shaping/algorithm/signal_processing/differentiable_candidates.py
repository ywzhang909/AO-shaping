"""K 个候选并行前向：``DifferentiableUniformityOptimizer`` 的批量分支。

为什么单独成模块
----------------
父模块 ``differentiable_defocus_uniformity.py`` 已在模块规模棘轮
(``scripts/check_module_size.py``) 的基线上，新增的批量代码必须落在新文件。
批量分支要读写父类的 ``_seed`` / ``_weights`` / ``_opt`` / ``_history`` 等属性，
反向 import 父模块会成环，所以做成 mixin 由父类继承。

K=1 不走这里
------------
父类保留原来的二维代码路径（参数仍是 2D ``_delta``，运算顺序不变），批量路径只在
``n_candidates > 1`` 时接管 ``update`` / ``accept``。对外契约仍是**单相位**：``delta_phase``
/ ``current_phase`` / ``accepted_phase`` / ``last_candidate_prediction`` 返回 best 候选的
二维值（父类里的 ``_delta`` / ``_accepted`` 是每步与 best 候选同步的二维镜像）。

前向为什么不用 ``torch.vmap``
-----------------------------
:meth:`FrozenDefocusForward.predict` 对 ``(B, H, H)`` 输入本来就是批安全的
(``total[None]`` 不触发，``sample_intensity`` 与 ``conv2d`` 都保 batch 维)，直接把
``(K, H, H)`` 传进去就是一次批前向。vmap 会把 ``interpolated_pupil_field`` /
``focal_intensity`` 的二维契约变成 vmap 内部规则，既没有必要也更容易踩到不支持的操作。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm._torch_support import require_torch


def _torch() -> Any:
    """返回 ``torch``；缺失时给出与父模块一致的安装提示。"""
    return require_torch(
        "differentiable_defocus_uniformity 需要 PyTorch: uv sync --extra ml"
    )


def remove_piston_and_tilt(delta: Any, weights: Any) -> None:
    """原地去 piston + 去整体倾斜（倾角按照明加权最小二乘拟合，两轴各一次）。

    作用在 ``delta`` 的**最后两维**上，因此 2D ``(H, W)`` 与批 ``(K, H, W)`` 共用同一段
    代码；二维输入的各项归约与拆出前逐字相同（同一个表达式、同一个张量形状），
    所以 K=1 的数值路径不变。

    Args:
        delta: 面板相位张量 ``(H, W)`` 或 ``(K, H, W)``，原地修改。
        weights: 照明权重 ``(H, W)``（做分母前有 ``clamp_min`` 防护）。
    """
    torch = _torch()
    with torch.no_grad():
        denominator = weights.sum().clamp_min(1e-12)
        if delta.ndim == 2:
            delta.sub_((weights * delta).sum() / denominator)
        else:
            delta.sub_((weights * delta).sum(dim=(-2, -1), keepdim=True) / denominator)
        height, width = delta.shape[-2], delta.shape[-1]
        grid_y = torch.arange(height, dtype=torch.float32, device=delta.device)
        grid_x = torch.arange(width, dtype=torch.float32, device=delta.device)
        # 行向 (axis 0)：按行加权拟合 ``slope * y``，沿列方向不变。
        w_rows = weights.sum(dim=-1)
        profile = (delta * weights).sum(dim=-1) / w_rows.clamp_min(1e-12)
        centre = (w_rows * grid_y).sum() / w_rows.sum().clamp_min(1e-12)
        centred = grid_y - centre
        slope = (w_rows * centred * profile).sum(dim=-1) / (
            w_rows * centred ** 2
        ).sum().clamp_min(1e-12)
        delta.sub_(slope[..., None, None] * centred[:, None])
        # 列向 (axis 1)：按列加权拟合 ``slope * x``，沿行方向不变。
        w_cols = weights.sum(dim=-2)
        profile = (delta * weights).sum(dim=-2) / w_cols.clamp_min(1e-12)
        centre = (w_cols * grid_x).sum() / w_cols.sum().clamp_min(1e-12)
        centred = grid_x - centre
        slope = (w_cols * centred * profile).sum(dim=-1) / (
            w_cols * centred ** 2
        ).sum().clamp_min(1e-12)
        delta.sub_(slope[..., None, None] * centred[None, :])


class CandidateBatchMixin:
    """``n_candidates > 1`` 的批量前向分支；K=1 由父类原始路径处理。

    父类 ``__init__`` 在创建 ``_delta`` / ``_accepted`` 之后调用
    :meth:`_setup_candidates`，并把 Adam 参数换成 :meth:`_adam_params`；``update`` /
    ``accept`` 在 ``_deltas is not None`` 时分别委派给 :meth:`_update_candidates` /
    :meth:`_accept_candidates`。

    记账口径：

    * 每个候选有自己的 ``delta`` 与自己的 ``accepted`` 行（增量 cap 各自计），所以
      一个候选被接受不会把其他候选的增量重置回同一个参照。
    * ``accept()`` 只把 **best** 候选写进二维 ``_accepted``（对外契约），best 行的
      ``_accepted_rows`` 同步推进；其他候选保持原样继续走。
    * anchored 模式下 ``measured`` / ``baseline_prediction`` 都是 ``(H, W)``，
      靠广播作用于 K 个候选（``baseline`` 就是 best 候选的锚点前向）。
    """

    _n_candidates: int = 1
    _deltas: Any = None
    _accepted_rows: Any = None
    _best_index: int = 0

    def _setup_candidates(self, n_candidates: int, step_rms_cap: float) -> None:
        """校验候选数并建立批量参数；``n_candidates == 1`` 时退回父类的二维路径。

        K>1 的候选从**不同**的确定性初始增量出发（同一固定 RNG 种子，幅度
        ``0.02 * step_rms_cap``）：同种子同梯度时 K 个候选会逐位相同，批量前向就退化成
        K 份一样的计算，测不出候选间差异，``argmin`` 也没有意义。K=1 不经过这里。
        """
        torch = _torch()
        if isinstance(n_candidates, bool):
            raise ValueError(f"n_candidates must be a positive integer, got {n_candidates!r}")
        try:
            count = int(n_candidates)
        except (TypeError, ValueError):
            raise ValueError(
                f"n_candidates must be a positive integer, got {n_candidates!r}"
            ) from None
        if count != n_candidates:
            raise ValueError(f"n_candidates must be a positive integer, got {n_candidates!r}")
        if count < 1:
            raise ValueError(f"n_candidates must be >= 1, got {count}")
        self._n_candidates = count
        if count == 1:
            self._deltas = None
            self._accepted_rows = None
            return
        shape = (count, *tuple(self._seed.shape))
        generator = torch.Generator().manual_seed(0x5EED)
        start = torch.randn(shape, generator=generator, dtype=torch.float32)
        self._deltas = (start * (0.02 * float(step_rms_cap))).to(self._device).requires_grad_(True)
        self._accepted_rows = torch.zeros_like(self._deltas)
        # 二维镜像同步到候选 0，与父类刚建立的 ``_delta`` / ``_accepted`` 同一契约。
        with torch.no_grad():
            self._delta.copy_(self._deltas[0])
            self._accepted.copy_(self._accepted_rows[0])

    def _adam_params(self) -> list[Any]:
        """交给 ``torch.optim.Adam`` 的参数：K=1 是原二维张量，K>1 是批张量。"""
        return [self._deltas if self._deltas is not None else self._delta]

    @property
    def n_candidates(self) -> int:
        """候选数（默认 1）。"""
        return self._n_candidates

    @property
    def candidate_index(self) -> int:
        """最近一次 :meth:`update` 选出的 best 候选下标（K=1 恒为 0）。

        ``loss_history`` 每条的 ``candidate`` 与之一致：按候选损失取 argmin。
        """
        return self._best_index

    def _increment_rms(self, increments: Any) -> Any:
        """逐候选的照明加权 RMS（沿最后两维归约，返回 ``(K,)``）。"""
        torch = _torch()
        scale = self._weights.sum().clamp_min(1e-12)
        return torch.sqrt((self._weights * increments ** 2).sum(dim=(-2, -1)) / scale)

    @property
    def candidate_increment_rms(self) -> list[float]:
        """每个候选 ``delta - accepted`` 的照明加权 RMS（K=1 返回单元素 list）。"""
        with _torch().no_grad():
            if self._deltas is None:
                return [self.weighted_rms(self._delta - self._accepted)]
            return [float(value) for value in self._increment_rms(self._deltas - self._accepted_rows)]

    def _project_increment_batch(self) -> list[float]:
        """逐候选把 ``delta - accepted`` 缩到照明加权 RMS ≤ cap；返回投影前 RMS。"""
        torch = _torch()
        with torch.no_grad():
            increment = self._deltas - self._accepted_rows
            rms = self._increment_rms(increment)
            scale = torch.where(
                (rms > self._cap) & (rms > 0),
                self._cap / rms.clamp_min(1e-20),
                torch.ones_like(rms),
            )
            self._deltas.copy_(self._accepted_rows + increment * scale[:, None, None])
            return [float(value) for value in rms]

    def _accept_candidates(self) -> npt.NDArray[np.floating]:
        """把 best 候选记为已接受；其他候选的 ``accepted`` 行保持不变。"""
        with _torch().no_grad():
            best = self._best_index
            self._accepted_rows[best].copy_(self._deltas[best])
            self._accepted.copy_(self._deltas[best])
        self._anchor_pred = None  # 锚点输入变了，缓存的前向作废
        return self.accepted_phase

    def _as_2d(self, image: Any, device: Any) -> Any:
        """把锚点/候选前向统一成 ``(H, W)``（兼容 ``(1, H, W)``）。"""
        torch = _torch()
        value = image if torch.is_tensor(image) else torch.as_tensor(
            np.asarray(image, dtype=np.float32), device=device
        )
        value = value.detach().to(device)
        return value[0] if value.ndim == 3 else value

    def _update_candidates(self, measured_intensity: npt.NDArray[np.floating] | None = None):
        """一步（K>1）：批前向 → 逐候选损失求和 → **一次**反传 → Adam → 逐候选投影/去倾斜。"""
        torch = _torch()
        if self._converged:
            return self.current_phase
        self._candidate_pred = None
        self._opt.zero_grad(set_to_none=True)
        count = self._n_candidates
        penalty = None
        if self._spectral_weight > 0:
            from ao_shaping.algorithm.signal_processing.spectral_regulariser import (
                spectral_penalty,
            )

            penalty = sum(
                self._spectral_weight * spectral_penalty(self._deltas[index], self._spectral_mask)
                for index in range(count)
            )
        # 一次批前向：(K, H, H) 进，(K, H, W) 出。``total[None]`` 只对二维输入触发。
        prediction = self.forward.predict(
            self._seed[None] + self._deltas,
            focus_phase=self._focus,
            carrier_phase=self._carrier,
        )
        if prediction.ndim != 3 or prediction.shape[0] != count:
            raise ValueError(
                f"forward.predict must return ({count}, H, W) for {count} candidates, "
                f"got {tuple(prediction.shape)}"
            )
        mode = "model"
        if measured_intensity is not None:
            measured = torch.as_tensor(
                np.asarray(measured_intensity, dtype=np.float32), device=self._device
            ).detach()
            if measured.shape != prediction.shape[1:]:
                raise ValueError(
                    f"measured_intensity shape {tuple(measured.shape)} "
                    f"!= {tuple(prediction.shape[1:])}"
                )
            # 锚定局部模型：I(phi) = I_anchor + F(phi) - F(seed+accepted)。
            # 残差对 delta 是常数，梯度只经冻结模型；measured / baseline 是 (H, W)，
            # 靠广播作用于 K 个候选。
            baseline_prediction = self._as_2d(self._anchor_prediction(), self._device)
            prediction = (measured[None] + prediction - baseline_prediction[None]).clamp_min(1e-8)
            mode = "anchored"
        parts = [
            self.objective.evaluate(prediction[index], self._deltas[index])
            for index in range(count)
        ]
        per_candidate = [float(part[0].detach()) for part in parts]
        total = parts[0][0]
        for part in parts[1:]:
            total = total + part[0]
        if penalty is not None:
            total = total + penalty
        total.backward()
        grad_norm = (
            float(self._deltas.grad.norm()) if self._deltas.grad is not None else 0.0
        )
        raw_step = -self._lr * self._deltas.grad.detach().clone()
        self._opt.step()
        # 与 K=1 同序：先逐候选去 piston/倾斜（对准方向），再逐候选投影（只缩放幅度到
        # 各自 cap）。批量投影逐候选归约，不会把 K 个增量混成一个全局 RMS。
        remove_piston_and_tilt(self._deltas, self._weights)
        rms_before = self._project_increment_batch()
        best = min(range(count), key=per_candidate.__getitem__)
        terms = parts[best][1]
        if penalty is not None:
            terms = {**terms, "l_spectral": float(penalty.detach())}
        with torch.no_grad():
            anchor_prediction = self._anchor_prediction()
            final = self.forward.predict(
                self._seed[None] + self._deltas,
                focus_phase=self._focus,
                carrier_phase=self._carrier,
            )
            if final.ndim != 3 or final.shape[0] != count:
                raise ValueError(
                    f"forward.predict must return ({count}, H, W) for {count} candidates, "
                    f"got {tuple(final.shape)}"
                )
            best_prediction = final[best]
            anchor_loss, _ = self.objective.evaluate(anchor_prediction, self._accepted)
            candidate_loss, _ = self.objective.evaluate(best_prediction, self._deltas[best])
            if penalty is not None:
                anchor_loss, candidate_loss = anchor_loss + penalty, candidate_loss + penalty
            final_step = self._deltas[best].detach() - self._accepted
            denom = raw_step[best].norm().clamp_min(1e-20) * final_step.norm().clamp_min(1e-20)
            step_cosine = float((raw_step[best] * final_step).sum() / denom)
            record_extra = dict(
                anchor_model_loss=float(anchor_loss),
                candidate_model_loss=float(candidate_loss),
                candidate_model_loss_change=float(candidate_loss - anchor_loss),
                raw_to_final_step_cosine=step_cosine,
            )
            if self._focus is None and self._carrier is None:
                self._candidate_pred = best_prediction.detach().cpu().numpy()
            # 对外契约仍是单相位：二维镜像同步到 best 候选。
            self._delta.copy_(self._deltas[best])
            self._accepted.copy_(self._accepted_rows[best])
        self._best_index = best
        self._step += 1
        best_delta = self._deltas[best]
        record = dict(
            step=self._step,
            mode=mode,
            loss=float(total.detach()),
            gradient_norm=grad_norm,
            increment_rms_before_projection=rms_before[best],
            increment_rms=self.candidate_increment_rms[best],
            phase_rms=float(best_delta.detach().norm() / np.sqrt(best_delta.numel())),
            **terms,
            **record_extra,
            loss_best=per_candidate[best],
            loss_per_candidate=list(per_candidate),
            candidate=best,
        )
        self._history.append(record)
        self._record(record["loss"])
        return self.current_phase
