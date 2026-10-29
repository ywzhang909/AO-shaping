"""逐块 Jacobian 方向校验与它驱动的信任域收缩（免设备）。

为什么要做
----------
anchored 模式的前提是**冻结**的物理预测模型给出的梯度方向大致正确。这个前提
不成立时, 拟合损失照样下降而实测指标不动: 2026-10 的既有实测是重拟合
``loss 0.93 -> 0.36`` 而配对 Δ≈0, 2026-10-29 的 200 px 方斑 lane 更直接 ——
四个末相位里三个的 CV 比种子**更差** (``dCV = +0.158 … +0.170``)。

每个 CCD 校验块本来就已经采了两端: ``A1`` (相位 = ``best``) 与 ``B`` (相位 =
``candidate``)。它们的**差分**是一次免费的方性探针 —— 模型预测
``pred(candidate) - pred(best)``, 实验系统实测 ``B - A1``。两者的相关性回答
"这一块走的这一步方向, 实验系统上存在吗", 代价只是一次 ``forward.predict``
(本机实测约 14 ms), 不需要额外采集。

拿到答案之后做什么
------------------
本模块只做两件事: **报告**一致性, 以及**调度**下一步的 ``phase_rms_cap``。
它绝不改变接受/拒绝判定 —— 判定始终是 CCD 实测的 (``accepted`` 仍由实测 CV
决定)。一个方向被否定的块, 只要实测确实更好就照样接受; 收缩的是**下一块**的
步长上限, 让闭环在模型说谎期间不再继续迈大步。

阈值从哪来
----------
``low=0.5`` / ``high=0.85`` 是先验起点, 不是标定值: 数字孪生
(:mod:`ao_shaping.drivers.sim.bench_twin`) 上可以先注入已知的 Jacobian 误差
再回放, 用同一口径标定 (见 ``report/shape_trials/multishape_scale`` 的下一步
方案)。方向一致性是**相对**量, 换设备不需要重标定, 换控制分辨率才需要。

本模块纯 numpy, 不 import torch / 设备层, 因此可以在无设备的机器上单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DEFAULT_HIGH",
    "DEFAULT_LOW",
    "DEFAULT_MIN_SIGNAL",
    "Agreement",
    "TrustRegionSchedule",
    "direction_agreement",
]

#: 相关性低于此值即认为本块的模型方向被实验系统否定。
DEFAULT_LOW = 0.50
#: 相关性高于此值即认为模型方向得到实验系统支持。
DEFAULT_HIGH = 0.85
#: 实测差分相对锚定帧均值的 RMS 低于此值 = 相机噪声, 不下结论。
DEFAULT_MIN_SIGNAL = 2e-3


@dataclass(frozen=True)
class Agreement:
    """一次方向校验的结果。``verdict`` 取值见 :func:`direction_agreement`。"""

    correlation: float
    relative_l2: float
    predicted_rms: float
    measured_rms: float
    verdict: str

    def as_dict(self) -> dict:
        return dict(correlation=self.correlation, relative_l2=self.relative_l2,
                    predicted_rms=self.predicted_rms, measured_rms=self.measured_rms,
                    verdict=self.verdict)


def direction_agreement(predicted_anchor, predicted_candidate, measured_anchor,
                        measured_candidate, *, low: float = DEFAULT_LOW,
                        high: float = DEFAULT_HIGH,
                        min_signal: float = DEFAULT_MIN_SIGNAL) -> Agreement:
    """比较模型预测差分与实测差分的方向一致性。

    四个入参都是**同一网格**上的强度图: 前两个是模型预测, 后两个是实测。
    相关性用 Pearson 系数 (对增益/曝光不敏感); ``relative_l2`` 用两者各自
    归一化后的差的相对 L2, 因此也与绝对亮度无关。

    ``verdict``:

    - ``"insufficient_signal"`` — 实测差分的 RMS 相对锚定帧均值太小, 这是噪声,
      既不能支持也不能否定 (下一步的 cap 保持不变);
    - ``"contradict"`` — 相关性 < ``low``;
    - ``"agree"`` — 相关性 >= ``high``;
    - ``"weak"`` — 介于两者之间。
    """
    pa = np.asarray(predicted_anchor, dtype=np.float64)
    pc = np.asarray(predicted_candidate, dtype=np.float64)
    ma = np.asarray(measured_anchor, dtype=np.float64)
    mc = np.asarray(measured_candidate, dtype=np.float64)
    if not (pa.shape == pc.shape == ma.shape == mc.shape):
        raise ValueError(f"shape mismatch: predicted {pa.shape}/{pc.shape}, measured {ma.shape}/{mc.shape}")

    predicted = (pc - pa).ravel()
    measured = (mc - ma).ravel()
    predicted_rms = float(np.sqrt(np.mean(predicted ** 2)))
    measured_rms = float(np.sqrt(np.mean(measured ** 2)))
    reference = float(np.mean(np.abs(ma))) + 1e-12

    if measured_rms / reference < min_signal:
        return Agreement(correlation=float("nan"), relative_l2=float("nan"),
                         predicted_rms=predicted_rms, measured_rms=measured_rms,
                         verdict="insufficient_signal")

    predicted_centred = predicted - predicted.mean()
    measured_centred = measured - measured.mean()
    p_norm = float(np.linalg.norm(predicted_centred))
    m_norm = float(np.linalg.norm(measured_centred))
    if p_norm < 1e-12 or m_norm < 1e-12:
        return Agreement(correlation=float("nan"), relative_l2=float("nan"),
                         predicted_rms=predicted_rms, measured_rms=measured_rms,
                         verdict="insufficient_signal")

    correlation = float(np.dot(predicted_centred, measured_centred) / (p_norm * m_norm))
    relative_l2 = float(np.linalg.norm(predicted / p_norm - measured / m_norm))
    verdict = ("agree" if correlation >= high
               else "contradict" if correlation < low else "weak")
    return Agreement(correlation=correlation, relative_l2=relative_l2,
                     predicted_rms=predicted_rms, measured_rms=measured_rms,
                     verdict=verdict)


@dataclass
class TrustRegionSchedule:
    """按方向一致性收缩 / 恢复 ``step_rms_cap`` 的调度器。

    设计取意与
    :class:`ao_shaping.optimizer.wfless.adaptive_trust_region.TrustRegionPolicy`
    一致 —— **增长靠证据, 收缩靠一次否定** —— 但这里的证据不是"接受/拒绝"
    (那已经由 ``phase_lr`` 调度), 而是"模型方向与实测方向是否一致"。两者正交:
    一个块可以实测接受但方向被否定, 这时 ``phase_lr`` 不变而 cap 减半。

    ``ceiling`` 默认等于初始 cap: 收缩后最多恢复到本 stage 配置的上限, 不会
    因为连续一致而超出配方。
    """

    cap: float
    floor: float = 0.02
    ceiling: float | None = None
    shrink: float = 0.5
    grow: float = 1.25
    low: float = DEFAULT_LOW
    high: float = DEFAULT_HIGH
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cap <= 0:
            raise ValueError("cap must be positive")
        if self.floor <= 0 or self.floor > self.cap:
            raise ValueError("floor must be positive and <= cap")
        if self.ceiling is None:
            self.ceiling = float(self.cap)
        if self.ceiling < self.floor:
            raise ValueError("ceiling must be >= floor")
        if not 0 < self.shrink < 1:
            raise ValueError("shrink must be in (0, 1)")
        if self.grow <= 1:
            raise ValueError("grow must be > 1")

    def observe(self, agreement: Agreement) -> float:
        """吸收一次方向校验结果, 返回**下一块**应使用的 cap。"""
        before = self.cap
        if agreement.verdict == "agree":
            self.cap = min(float(self.ceiling), self.cap * self.grow)
        elif agreement.verdict == "contradict":
            self.cap = max(self.floor, self.cap * self.shrink)
        # weak / insufficient_signal: 证据不足, 不动。
        self.history.append(dict(verdict=agreement.verdict,
                                 correlation=agreement.correlation,
                                 cap_before=before, cap_after=self.cap))
        return self.cap
