"""不确定性采样探针选择器 (主动学习选探针)。

闭环整形的 Step A 用一组强随机探针把正向模型的像差系数重新拟合一遍。固定
随机探针并不保证激励到当前系数后验里**最不确定**的方向: 一轮随机相位可能
把灵敏度都花在已经被良好辨识的像差上, 而对残差方向几乎无信息。本类把探针
相位 ``p`` 当作可微变量, 沿"系数不确定性经物理正向算子传播到远场的预测方差"
``U(p)`` 做梯度上升, 于是选出的探针专门放大模型当前最没把握的像差方向。

度量 (无需训练, 只用正向算子本身):

    U(p) = (1/K) Σ_k ‖ far(c* + δ_k, p) − far(c*, p) ‖²,   δ_k ~ N(0, Σ)

``c*`` 是当前最优系数, ``far`` 是 :class:`ZernikeCoefficientOptimizer` 的可微
远场 (峰值归一化)。系数后验协方差 ``Σ`` 由正向算子的 Fisher 信息

    F = Σ_p J_pᵀ J_p,   J_p = ∂ far(c*, p) / ∂ c

经岭正则求逆得到 (``Σ = (F + λI)⁻¹``)。``F`` 小的方向 —— 即当前探针集没有
激励到的像差 —— 在 ``Σ`` 里方差大, 于是 ``U(p)`` 奖励能激励它们的探针。首轮
没有探针时退化为各向同性先验 ``Σ = prior_std² · I``, 此时 ``U(p)`` 正比于
远场对系数扰动的总灵敏度 (``‖J_p‖_F²``), 仍是一个有意义的"信息探针"判据。

``mode="minimize"`` 只用于**反向验证**: 最小化 ``U`` 会把探针推向对系数不确
定性最"盲"的退化点 (近零方差), 用来证明该度量确实在度量不确定性, 而不是在
追随机噪声。生产路径恒用 ``maximize``。

约束 (每步上升后投影, 保证探针仍是可下发的物理相位):
    * 幅度上限: 标准差超过 ``spread_cap`` 时整体**回推** (缩放, 非截断),
      与 :func:`slm_model_in_loop._probe_phase` 的"推到不截断"一致。
    * 幅度下限: 标准差低于 ``spread_floor`` 时整体放大, 防止塌成平场
      (近零探针是盲探针, 正是 minimize 模式的退化点)。
    * 带限 / 光滑性: 高斯低通, 匹配硬件上 ``smooth=True`` 的探针。
    * 多样性: 惩罚与本轮已选探针的余弦相似度, 防止多个探针塌到同一点。

本类是 :mod:`ao_shaping.algorithm` 的类式优化器约定下的**状态类**: 构造一次,
由优化器层的驱动循环反复调用 :meth:`update` 逐步上升。循环逻辑 (日志、早停、
best 追踪、逐探针验收) 属于优化器层, 不在本类内。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy import ndimage

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import torch

    from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
        ZernikeCoefficientOptimizer,
    )


_MODES = ("maximize", "minimize")


def _torch():
    """返回 ``torch`` 模块; 缺失时抛出 :class:`ImportError`。"""
    try:
        import torch as _t
    except ImportError:
        raise ImportError(
            "uncertainty_probe_selector requires PyTorch. "
            "Install with: uv sync --group ml"
        ) from None
    return _t


class UncertaintyProbeSelector:
    """把探针相位推向"系数不确定性预测方差"极值的可微选择器。

    Attributes:
        region: 探针相位网格边长 (瞳面网格)。
        n_coefficients: 拟合的 Zernike 系数个数。
        mode: ``"maximize"`` (生产) 或 ``"minimize"`` (反向验证)。
        uncertainty_history: 每次 :meth:`update` 记录的 ``U(p)`` 值。
    """

    def __init__(
        self,
        optimizer: "ZernikeCoefficientOptimizer",
        *,
        seed: int = 0,
        n_samples: int = 6,
        ridge: float = 1e-3,
        prior_std: float = 1.0,
        spread_cap: float = 4.0,
        spread_floor: float = 3.0,
        bandlimit_sigma: float = 1.5,
        diversity_weight: float = 0.5,
        mode: str = "maximize",
        lr: float = 0.3,
        fd_eps: float = 1e-3,
    ) -> None:
        """校验输入并建立全部状态 (先验协方差、MC 样本、初始探针)。

        Args:
            optimizer: 提供可微远场 :meth:`far_field_tensor` 与当前系数
                :attr:`coefficients` 的 Step A 优化器; 其源振幅必须已设置
                (``set_source_amplitude``), 因为 ``far_field_tensor`` 缺省
                取优化器内部存的那份。
            seed: 复现种子, 决定 MC 系数样本与初始探针。
            n_samples: 预测方差的蒙特卡洛系数样本数 ``K``。
            ridge: Fisher 求逆的岭正则 ``λ`` (>=0)。
            prior_std: 首轮各向同性系数先验标准差 (弧度), 用于
                ``Σ = prior_std² · I``; 调用 :meth:`update_covariance` 后
                被 Fisher 后验替换。
            spread_cap: 探针相位标准差上限 (弧度), 超出则整体回推。
            spread_floor: 探针相位标准差下限 (弧度), 低于则整体放大。
            bandlimit_sigma: 高斯低通 sigma (像素); ``<=0`` 关闭带限。
            diversity_weight: 多样性惩罚权重 (>=0)。
            mode: ``"maximize"`` 或 ``"minimize"``。
            lr: 归一化梯度步长 (每步相位最多移动的弧度数)。
            fd_eps: Fisher 有限差分对系数的扰动 (弧度)。

        Raises:
            ValueError: 任一参数非法 (维度、形状、取值范围) 时。
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if float(ridge) < 0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        if float(prior_std) <= 0:
            raise ValueError(f"prior_std must be > 0, got {prior_std}")
        if not (0 < float(spread_floor) <= float(spread_cap)):
            raise ValueError(
                f"need 0 < spread_floor <= spread_cap, got "
                f"floor={spread_floor} cap={spread_cap}"
            )
        if float(lr) <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if float(diversity_weight) < 0:
            raise ValueError(f"diversity_weight must be >= 0, got {diversity_weight}")
        if float(fd_eps) <= 0:
            raise ValueError(f"fd_eps must be > 0, got {fd_eps}")

        torch = _torch()
        self._torch = torch

        self._opt = optimizer
        self._region = int(optimizer.region)
        self._n_coeffs = int(optimizer.n_coefficients)
        if self._n_coeffs < 1:
            raise ValueError("optimizer exposes no coefficients")
        self._dev = optimizer.coefficient_tensor.device

        self._seed = int(seed)
        self._n_samples = int(n_samples)
        self._ridge = float(ridge)
        self._prior_std = float(prior_std)
        self._spread_cap = float(spread_cap)
        self._spread_floor = float(spread_floor)
        self._bandlimit_sigma = float(bandlimit_sigma)
        self._diversity_weight = float(diversity_weight)
        self._mode = mode
        self._lr = float(lr)
        self._fd_eps = float(fd_eps)

        self._rng = np.random.default_rng(self._seed)
        self._uncertainty_history: list[float] = []

        # 当前最优系数 (numpy, 弧度) 与各向同性先验协方差。
        self._c_star = np.asarray(optimizer.coefficients, dtype=np.float64).copy()
        self._sigma = (self._prior_std ** 2) * np.eye(self._n_coeffs)
        self._deltas = self._sample_deltas()

        # 候选探针相位: 一个 requires_grad 的叶子张量, 初始为带限随机探针。
        self._phase = self._new_phase_leaf()
        # 本轮已提交探针 (展平、detach), 供多样性惩罚。
        self._chosen: list["torch.Tensor"] = []

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------
    @property
    def region(self) -> int:
        """探针相位网格边长。"""
        return self._region

    @property
    def n_coefficients(self) -> int:
        """系数维度。"""
        return self._n_coeffs

    @property
    def mode(self) -> str:
        """``"maximize"`` 或 ``"minimize"``。"""
        return self._mode

    @property
    def uncertainty_history(self) -> list[float]:
        """每次 :meth:`update` 记录的 ``U(p)``。"""
        return self._uncertainty_history

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def current_phase(self) -> npt.NDArray[np.float64]:
        """当前候选探针相位 (弧度, ``(region, region)``), 已满足全部约束。"""
        return self._phase.detach().cpu().numpy().astype(np.float64)

    def current_uncertainty(self) -> float:
        """当前候选探针的预测方差 ``U(p)`` (不建图)。"""
        with self._torch.no_grad():
            return float(self._predictive_variance(self._phase))

    def update(self) -> npt.NDArray[np.float64]:
        """沿 ``U(p)`` 做**一步**受约束上升 (或 minimize 模式下一步下降)。

        Returns:
            更新后的候选探针相位 (弧度, ``(region, region)``)。
        """
        torch = self._torch
        self._phase.grad = None
        obj = self._predictive_variance(self._phase) - self._diversity_weight * self._diversity_penalty(self._phase)
        grad = torch.autograd.grad(obj, self._phase)[0]

        # ``U`` 对相位的斜率量级极小且随 far-field 尺度漂移: 原始梯度会把步长
        # 绑到目标尺度上。除以峰值后 ``lr`` 成为尺度无关的"每步弧度预算"。
        peak = grad.abs().max()
        if peak > 0:
            grad = grad / peak

        sign = 1.0 if self._mode == "maximize" else -1.0
        with torch.no_grad():
            self._phase += self._lr * sign * grad
        self._apply_constraints()
        self._uncertainty_history.append(self.current_uncertainty())
        return self.current_phase()

    def commit(self) -> None:
        """把当前候选登记为本轮已选探针, 并为下一个探针重播种候选相位。

        驱动在测得该探针的帧、确认采纳后调用。登记后多样性惩罚会把它计入,
        下一个 :meth:`update` 序列因此被推向与已选探针不同的方向。
        """
        flat = self._phase.detach().reshape(-1).clone()
        self._chosen.append(flat)
        self._phase = self._new_phase_leaf()

    def update_covariance(
        self, probe_phases: npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]
    ) -> None:
        """用刚测过的一批探针重算系数后验协方差 ``Σ = (F + λI)⁻¹``。

        Fisher ``F = Σ_p J_pᵀ J_p`` 用对系数的中心有限差分在当前 ``c*`` 处求
        (峰值归一化远场)。调用后重采样 MC 系数样本 ``δ_k``。

        Args:
            probe_phases: 一个或多个 ``(region, region)`` 探针相位 (弧度)。

        Raises:
            ValueError: 探针形状不对或为空时。
        """
        if isinstance(probe_phases, np.ndarray) and probe_phases.ndim == 2:
            probes = [probe_phases]
        else:
            probes = list(probe_phases)  # type: ignore[arg-type]
        if not probes:
            raise ValueError("update_covariance needs at least one probe phase")

        self._c_star = np.asarray(self._opt.coefficients, dtype=np.float64).copy()
        fisher = np.zeros((self._n_coeffs, self._n_coeffs), dtype=np.float64)
        for probe in probes:
            fisher += self._probe_fisher(np.asarray(probe, dtype=np.float64))
        reg = fisher + self._ridge * np.eye(self._n_coeffs)
        try:
            self._sigma = np.linalg.inv(reg)
        except np.linalg.LinAlgError:
            logger.warning("Fisher posterior singular; falling back to isotropic prior")
            self._sigma = (self._prior_std ** 2) * np.eye(self._n_coeffs)
        self._deltas = self._sample_deltas()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _predictive_variance(self, phase_leaf: "torch.Tensor") -> "torch.Tensor":
        """``U(p) = (1/K) Σ_k mean‖far(c*+δ_k,p) − far(c*,p)‖²`` (挂在图上)。"""
        base = self._opt.far_field_tensor(self._c_star, phase_leaf)
        base = base / (base.max() + 1e-8)
        acc = 0.0
        for delta in self._deltas:
            pert = self._opt.far_field_tensor(
                self._c_star + delta.detach().cpu().numpy().astype(np.float64),
                phase_leaf,
            )
            pert = pert / (pert.max() + 1e-8)
            acc = acc + ((pert - base) ** 2).mean()
        return acc / float(self._n_samples)

    def _diversity_penalty(self, phase_leaf: "torch.Tensor") -> "torch.Tensor":
        """已选探针上的平方余弦相似度之和 (越大越该被惩罚)。"""
        if not self._chosen or self._diversity_weight == 0:
            return self._torch.zeros(())
        flat = phase_leaf.reshape(-1)
        norm = flat.norm() + 1e-9
        unit = flat / norm
        pen = self._torch.zeros(())
        for q in self._chosen:
            qn = q / (q.norm() + 1e-9)
            pen = pen + (unit * qn).sum() ** 2
        return pen

    def _probe_fisher(self, probe: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """单个探针处 ``JᵀJ``, ``J = ∂ far(c*, probe) / ∂ c`` (中心差分)。"""
        n = self._n_coeffs
        cols = []
        for k in range(n):
            cp = self._c_star.copy()
            cm = self._c_star.copy()
            cp[k] += self._fd_eps
            cm[k] -= self._fd_eps
            fp = self._opt.forward_intensity(cp, probe)
            fm = self._opt.forward_intensity(cm, probe)
            fp = fp / (float(np.max(fp)) + 1e-8)
            fm = fm / (float(np.max(fm)) + 1e-8)
            cols.append((fp - fm) / (2.0 * self._fd_eps))
        jac = np.stack(cols, axis=1).reshape(-1, n)  # (ff*ff, n)
        return jac.T @ jac

    def _sample_deltas(self) -> "torch.Tensor":
        """从当前 ``Σ`` 采 ``K`` 个系数扰动 (特征分解保证半正定)。"""
        sigma = 0.5 * (self._sigma + self._sigma.T)
        w, v = np.linalg.eigh(sigma)
        w = np.clip(w, 0.0, None)
        chol = v @ np.diag(np.sqrt(w))  # Σ = chol cholᵀ
        raw = self._rng.standard_normal((self._n_samples, self._n_coeffs)) @ chol.T
        return self._torch.from_numpy(
            np.ascontiguousarray(raw, dtype=np.float64)
        ).to(self._dev)

    def _new_phase_leaf(self) -> "torch.Tensor":
        """带限随机探针, 作为 requires_grad 叶子张量。"""
        p = self._rng.standard_normal((self._region, self._region))
        if self._bandlimit_sigma > 0:
            p = ndimage.gaussian_filter(p, sigma=self._bandlimit_sigma)
        std = float(p.std())
        if std <= 1e-9:
            p = self._rng.standard_normal((self._region, self._region))
            std = float(p.std())
        target = 0.5 * (self._spread_cap + self._spread_floor)
        p = p * (target / std)
        leaf = self._torch.from_numpy(
            np.ascontiguousarray(p, dtype=np.float64)
        ).to(self._dev)
        leaf.requires_grad_(True)
        return leaf

    def _apply_constraints(self) -> None:
        """对候选相位施加带限 + 幅度上下限 (回推/放大, 非截断)。"""
        p = self._phase.detach().cpu().numpy().astype(np.float64)
        if self._bandlimit_sigma > 0:
            p = ndimage.gaussian_filter(p, sigma=self._bandlimit_sigma)
        std = float(p.std())
        if std <= 1e-9:
            p = self._rng.standard_normal((self._region, self._region))
            if self._bandlimit_sigma > 0:
                p = ndimage.gaussian_filter(p, sigma=self._bandlimit_sigma)
            std = float(p.std())
            if std <= 1e-9:
                std = 1.0
        if std > self._spread_cap:
            p = p * (self._spread_cap / std)
        elif std < self._spread_floor:
            p = p * (self._spread_floor / std)
        self._phase.data = self._torch.from_numpy(
            np.ascontiguousarray(p, dtype=np.float64)
        ).to(self._dev)
