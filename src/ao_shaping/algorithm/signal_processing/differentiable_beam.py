"""基于 FFT 远场模型的可微分 (反向传播) 光束整形。

本模块实现 SLM 相位恢复的 "backprop" 算法。它不去做交替投影的
Gerchberg-Saxton 循环, 而是把相位图形 ``φ`` 当作一个*可学习*张量,
通过可微的角谱模型 (一次傅里叶变换) 对远场强度损失关于 ``φ`` 求导,
再用 Adam 优化器更新 ``φ``。

模型
-----
在 SLM (源) 面, 复光场为

    E_near = A * exp(i * φ)

其中 ``A`` 是 (已知、通常均匀的) 照明振幅, ``φ`` 是我们优化的相位。
远场被建模为傅里叶变换

    E_far = fftshift(fft2(E_near))

远场强度为 ``I_far = |E_far|²``。损失是 ``I_far`` 与 (归一化的) 目标强度
之间的 MSE。梯度从损失经 FFT 流向 ``φ``。

优化过程中 ``φ`` 保持未包裹 (不做 mod-2π), 以保证数值稳定;
只有在转换阶段才包裹到 SLM 的灰度范围
(见 ``beam_shaping_utils.phase_to_slm_grayscale``)。

优化循环以有状态类 (:class:`DifferentiableBeamOptimizer`) 暴露,
提供 ``__init__`` (校验 + 状态) 与 ``update()`` (一步)。
带日志、进度条、逐步历史与最优相位跟踪的完整循环位于 optimizer 层, 即
``ao_shaping.optimizer.wfless.differentiable_beam.optimize_beam_shaping``
(驱动 ``update()`` 的 pib 式顶层函数)。旧的一次性函数
``differentiable_beam_optimize`` 已被移除; 该类是 algorithm 包内唯一的公开 API。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer
from ao_shaping.utils.slm.phase_display import DEFAULT_WAVELENGTH

if TYPE_CHECKING:  # pragma: no cover – type-only imports
    import torch


def _torch():
    """返回 ``torch`` 模块; 缺失时抛出 :class:`ImportError`。

    Returns:
        ``torch`` 顶层模块。

    Raises:
        ImportError: 若未安装 PyTorch。
    """
    try:
        import torch as _t
    except ImportError:
        raise ImportError(
            "differentiable_beam requires PyTorch. Install with: uv sync --extra ml"
        ) from None
    return _t


def _to_tensor(x: npt.NDArray[np.floating], device: torch.device) -> torch.Tensor:
    """把实数二维 numpy 数组转成 ``device`` 上的 float32 torch 张量。"""
    torch = _torch()
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def differentiable_far_field(
    amplitude: npt.NDArray[np.floating],
    phase: torch.Tensor,
) -> torch.Tensor:
    """由源面振幅与相位算出可微的远场复振幅。

    Args:
        amplitude: 实数二维源面振幅 (numpy, 例如全 1)。
        phase: 二维相位张量, 单位弧度 (必须是叶子张量或已接入计算图的张量;
            梯度就是关于它计算的)。

    Returns:
        二维复张量: 远场复光场 ``fftshift(fft2(A * exp(i * phase)))``。

    Note:
        ``fftshift`` 与复指数都可微, 因此返回的光场带有回连到 ``phase``
        的 grad_fn。
    """
    torch = _torch()
    amp = _to_tensor(amplitude, phase.device)
    complex_field = amp * torch.exp(1j * phase)
    fft = torch.fft.fft2(complex_field)
    return torch.fft.fftshift(fft)


def far_field_intensity(
    amplitude: npt.NDArray[np.floating],
    phase: torch.Tensor,
) -> torch.Tensor:
    """可微的远场*强度* ``|E_far|²``。

    Args:
        amplitude: 实数二维源面振幅 (numpy)。
        phase: 二维相位张量, 单位弧度。

    Returns:
        形状为 ``(H, W)`` 的二维 float 张量, 存放远场强度。
    """
    e_far = differentiable_far_field(amplitude, phase)
    return e_far.real**2 + e_far.imag**2


class DifferentiableBeamOptimizer(IterativeOptimizer):
    """优化 SLM 相位图以匹配目标远场强度。

    用 PyTorch autograd 配合 Adam 优化器。损失是当前相位产生的
    (峰值归一化的) 远场强度与归一化目标之间的 MSE。

    该优化器是有状态的: 构造一次, 然后反复调用 :meth:`update` 以手动控制。
    完整循环 (日志、进度条、提前停止、最优相位跟踪) 由 optimizer 层的函数
    ``ao_shaping.optimizer.wfless.differentiable_beam.optimize_beam_shaping`` 提供。

    Attributes:
        step: 迄今为止执行的优化步数。
        device: 优化所在的 torch 设备。
        phase_tensor: 实时的相位参数张量 (requires_grad)。
        current_phase: 当前相位的 numpy 副本 (已 detach), 单位弧度。
        loss_history: 每一步优化对应的损失值。
        last_loss: 最后一步的损失, 首次更新之前为 ``None``。
        converged: 是否满足提前停止条件。
    """

    def __init__(
        self,
        target_intensity: npt.NDArray[np.floating],
        source_amplitude: npt.NDArray[np.floating] | None = None,
        lr: float = 0.01,
        init_phase: npt.NDArray[np.floating] | None = None,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        """初始化优化器并校验所有输入。

        Args:
            target_intensity: 二维目标远场*强度*图 (非负)。
                比较之前会做峰值归一化, 使最大值为 1.0,
                所以目标的绝对尺度无关紧要。
            source_amplitude: 二维源面 (SLM) 照明振幅。
                若为 ``None``, 则使用与目标同形状的全 1 均匀振幅。
            lr: Adam 学习率。
            init_phase: 初始相位 (弧度)。若为 ``None``, 则随机抽取一个
                小相位 (由 ``seed`` 播种以保证可复现)。
            device: torch 设备字符串 (``"cuda"``、``"cpu"``)。若为 ``None``,
                则可用时用 CUDA, 否则用 CPU。
            seed: 随机初始相位的可选 RNG 种子。

        Raises:
            ValueError: 若目标不是二维、含负值, 或源振幅 / 初始相位的形状
                与目标形状不匹配。
        """
        torch = _torch()
        target = np.asarray(target_intensity)
        if target.ndim != 2:
            raise ValueError(f"target_intensity must be 2D, got {target.ndim}D")
        if np.any(target < 0):
            raise ValueError("target_intensity must be non-negative")

        # 把目标做峰值归一化, 使绝对尺度无关紧要。
        # (峰值归一化是相位恢复里的标准做法; 能量归一化在 float32 下会
        # 下溢, 因为 FFT 把能量集中到了少数像素上。)
        t = target.astype(np.float32)
        t_max = float(t.max())
        if t_max > 0:
            t = t / t_max

        if source_amplitude is None:
            source_amplitude = np.ones_like(t)
        else:
            source_amplitude = np.asarray(source_amplitude, dtype=np.float32)
            if source_amplitude.shape != t.shape:
                raise ValueError(
                    f"source_amplitude shape {source_amplitude.shape} "
                    f"must match target shape {t.shape}"
                )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        # 播种 RNG, 使随机初始化可复现。
        if seed is not None:
            rng = np.random.default_rng(seed)
            init_phase_np = rng.random(t.shape, dtype=np.float32) * (2 * np.pi)
        else:
            init_phase_np = np.random.random(t.shape).astype(np.float32) * (2 * np.pi)

        if init_phase is not None:
            init_phase_np = np.asarray(init_phase, dtype=np.float32)
            if init_phase_np.shape != t.shape:
                raise ValueError(
                    f"init_phase shape {init_phase_np.shape} "
                    f"must match target shape {t.shape}"
                )

        phase = torch.from_numpy(init_phase_np).to(dev)
        phase.requires_grad_(True)

        self._target_t = torch.from_numpy(t).to(dev)
        self._source_amplitude = source_amplitude
        self._phase = phase
        self._opt = torch.optim.Adam([phase], lr=lr)
        self._lr = lr
        self._dev = dev
        self._seed = seed
        # 初始化基类的迭代记账。``max_iterations=1`` 让基类的
        # ``is_converged`` (预算耗尽) 检查保持惰性: 真正的收敛是本类自己的
        # 提前停止标志 ``_converged`` (见 ``converged`` 属性),
        # 所以基类的 ``run()``/``is_converged`` 预算只是一道无效的保险,
        # 而非主要的停止条件。
        super().__init__(max_iterations=1)
        self._step = 0
        self._loss_history: list[float] = []
        self._converged = False

    @property
    def step(self) -> int:
        """迄今为止执行的优化步数。"""
        return self._step

    @property
    def device(self) -> str:
        """优化所在的 torch 设备。"""
        return str(self._dev)

    @property
    def phase_tensor(self) -> torch.Tensor:
        """实时的相位参数张量 (requires_grad)。"""
        return self._phase

    @property
    def current_phase(self) -> npt.NDArray[np.floating]:
        """当前相位的 numpy 副本 (已 detach), 单位弧度。"""
        return self._phase.detach().cpu().numpy()

    @property
    def loss_history(self) -> list[float]:
        """每一步优化对应的损失值。"""
        return self._loss_history

    @property
    def last_loss(self) -> float | None:
        """最后一步的损失, 首次更新之前为 ``None``。"""
        return self._loss_history[-1] if self._loss_history else None

    @property
    def converged(self) -> bool:
        """是否满足提前停止条件。"""
        return self._converged

    def _intensity_loss(self, i_tensor: torch.Tensor) -> torch.Tensor:
        """峰值归一化强度图与目标之间的 MSE。

        全类使用的损失约定: 把强度峰值归一化到 1.0 (与归一化后的目标匹配),
        再取均方差。float32 下的峰值归一化避免了能量归一化 (sum) 在
        FFT 把能量集中到少数像素时所出现的下溢。

        Args:
            i_tensor: 形状为 ``(H, W)`` 的二维强度张量。

        Returns:
            标量 MSE 损失张量 (对 ``i_tensor`` 可微)。
        """
        torch = _torch()
        i_norm = i_tensor / (i_tensor.max() + 1e-8)
        return torch.mean((i_norm - self._target_t) ** 2)

    def loss_value(self) -> torch.Tensor:
        """计算当前的 MSE 损失张量 (不做 backward)。

        Returns:
            当前相位的峰值归一化远场强度与归一化目标之间的 MSE。
        """
        i_far = far_field_intensity(self._source_amplitude, self._phase)
        return self._intensity_loss(i_far)

    def update(self, measured_intensity: npt.NDArray[np.floating] | None = None) -> npt.NDArray[np.floating]:
        """执行一步反向传播并返回下一个相位。

        当 ``measured_intensity=None`` (默认) 时, 这一步是经典的仿真步:
        损失由模型对当前相位自身算出的远场强度得到, 再经相位张量反向传播。

        传入 ``measured_intensity`` 数组时, 这一步是*测量锚定*的
        (硬件闭环): 损失在实测远场图像处求值, 其关于强度的梯度 (``∂ℓ/∂I``)
        也在该实测图像处计算, 再把该梯度作为模型对当前相位的远场强度的
        上游梯度 —— 即模型 Jacobian ``∂I_sim/∂φ`` 被在实测图像处求值的
        ``∂ℓ/∂I`` 加权。随后用内部的 Adam 步更新相位。
        记录的损失是实测损失。

        若优化器已经收敛, 这一步为空操作, 直接返回当前相位而不推进。

        Args:
            measured_intensity: 可选的二维实测远场强度图
                (例如重采样到目标网格的 CCD 帧)。比较之前会做峰值归一化,
                与目标的归一化方式一致。

        Returns:
            更新后的相位, 以 numpy 数组返回 (已 detach), 单位弧度。

        Raises:
            ValueError: 若 ``measured_intensity`` 不是二维, 或其形状与目标形状
                不匹配。
        """
        torch = _torch()
        if self._converged:
            return self.current_phase
        self._opt.zero_grad(set_to_none=True)

        if measured_intensity is None:
            loss = self.loss_value()
            loss.backward()
            self._opt.step()
            self._step += 1
            self._loss_history.append(float(loss.detach().cpu()))
            self._record(float(loss.detach().cpu()))
            return self.current_phase

        # 测量锚定的一步: 在实测图像处对损失求值。
        measured = np.asarray(measured_intensity, dtype=np.float32)
        if measured.ndim != 2:
            raise ValueError(f"measured_intensity must be 2D, got {measured.ndim}D")
        if measured.shape != tuple(self._target_t.shape):
            raise ValueError(
                f"measured_intensity shape {measured.shape} must match "
                f"target shape {tuple(self._target_t.shape)}"
            )
        i_meas = torch.from_numpy(measured).to(self._dev)
        i_meas.requires_grad_(True)
        loss_meas = self._intensity_loss(i_meas)
        loss_meas.backward()  # -> i_meas.grad = 在实测图像处求值的 ∂ℓ/∂I

        # 用实测图像处的损失梯度给模型 Jacobian ∂I_sim/∂φ 加权:
        # 相位梯度由此锚定在真实硬件图像上。
        i_sim = far_field_intensity(self._source_amplitude, self._phase)
        i_sim.backward(gradient=i_meas.grad)

        self._opt.step()
        self._step += 1
        self._loss_history.append(float(loss_meas.detach().cpu()))
        self._record(float(loss_meas.detach().cpu()))
        return self.current_phase
