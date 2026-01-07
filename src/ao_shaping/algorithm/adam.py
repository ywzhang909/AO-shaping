from typing import Literal

import numpy as np

from abc import ABC, abstractmethod

def learning_schedule(
    lr, epoch, epochs, method: Literal["static", "cosin", "exp", "linear"] = "static"
):
    if method == "static":
        return lr
    # 余弦退火
    elif method == "cosin":
        lr = lr * np.cos(np.pi * epoch / epochs) + 1e-6
        return lr
    # 指数衰减
    elif method == "exp":
        lr = lr * np.exp(-epoch / epochs) + 1e-6
        return lr
    # 线性衰减
    elif method == "linear":
        lr = lr * (1 - epoch / epochs) + 1e-6
        return lr
    else:
        raise ValueError("method must be static, cosin, exp or linear")


class Base(ABC):
    def __init__(self, dim:int, lr=1.0):
        self.dim = dim
        self.lr = lr
        self.t:int = 0
        
    @abstractmethod
    def update(self, grad:np.ndarray) -> np.ndarray:
        pass


class SGD(Base):
    
    def __init__(self, dim:int, lr=1.0):
        self.dim = dim
        self.lr = lr
        self.t:int = 0
        
    def update(self, grad:np.ndarray):
        self.t += 1
        return self.lr * grad


class Adam(Base):
    """
    使用 EMA 来估计二阶矩。这意味着它会遗忘早期的梯度信息。这使得 Adam 的自适应性更强，可以快速适应梯度的局部变化。
    """
    
    def __init__(self, dim:int, lr=1.0, beta1 = 0.9, beta2 = 0.99):
        self.dim = dim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.t:int = 0
        
    def update(self, grad:np.ndarray):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    def scale_momentum(self, scaler:float):
        self.m *= scaler
        self.v *= scaler**2
    
class AdamW(Adam):
    def __init__(self, dim:int, lr=1.0, beta1 = 0.9, beta2 = 0.99, weight_decay=1e-2):
        super().__init__(dim, lr, beta1, beta2)
        self.weight_decay = weight_decay
        
    def update(self, grad:np.ndarray):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + 1e-8) + self.weight_decay * self.lr * self.m


class AdaMOD(Adam):
    """
    AdaMod 是一个基于 Adam 的新的深度学习优化器，但它提供了自动warmup heuristic和长期学习率缓冲。 
    从最初的测试来看，AdaMod 是top 5的优化器，很容易击败或超过普通的 Adam，且对学习率超参数不那么敏感，训练曲线更平滑，不需要warmup模式。
    
    Pros:
    AdaMod保持了自适应学习率自身的指数长期平均值，并在整个训练过程中用这个值来clip任何过高的适应率。 
    结果改善了收敛性，不需要warmup，对实际学习率选择的敏感性较低。 记忆的程度由一个新的参数 Beta3控制。
    
    Cons:
    虽然AdaMod通常比普通的Adam表现更好，但是在更长的训练条件下，SGDM 仍然可能比AdaMod表现更好。
    
    """
    def __init__(self, dim:int, lr=1.0, beta1 = 0.9, beta2 = 0.99, beta3 = 0.9995, **kwargs):
        super().__init__(dim, lr, beta1, beta2)
        self.beta3 = beta3
        self.s = 0.0
        
    def update(self, grad:np.ndarray):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        gamma = self.lr / (np.sqrt(v_hat) + 1e-8)
        self.s = self.beta3 * self.s + (1 - self.beta3) * gamma
        learning_rate = np.where(gamma<self.s, gamma, self.s)
        return learning_rate * m_hat
    

class Muno(Base):
    """
    Muno 优化器是一种结合了动量和自适应学习率的优化算法。
    它通过维护梯度的指数移动平均和梯度平方的指数移动平均来动态调整学习率，
    同时引入了额外的机制来稳定训练过程。
    """
    
    def __init__(self, dim:int, lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8, amsgrad=False):
        """
        初始化 Muno 优化器
        
        Args:
            dim: 参数维度
            lr: 初始学习率
            beta1: 动量项的指数衰减率
            beta2: 梯度平方项的指数衰减率
            eps: 数值稳定性常数
            amsgrad: 是否使用 AMSGrad 变体
        """
        super().__init__(dim, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        
        # 初始化动量和梯度平方的累积变量
        self.m = np.zeros(self.dim, dtype=np.float32)  # 动量
        self.v = np.zeros(self.dim, dtype=np.float32)  # 梯度平方的累积
        self.v_max = np.zeros(self.dim, dtype=np.float32)  # AMSGrad 中的最大梯度平方
        
    def update(self, grad:np.ndarray):
        """
        更新参数
        
        Args:
            grad: 当前梯度
            
        Returns:
            更新步长
        """
        self.t += 1
        
        # 更新动量（一阶矩估计）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # 更新梯度平方的累积（二阶矩估计）
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        
        # 偏差修正
        m_hat = self.m / (1 - self.beta1**self.t)
        
        if self.amsgrad:
            # AMSGrad: 维护历史最大值
            self.v_max = np.maximum(self.v_max, self.v)
            v_hat = self.v_max / (1 - self.beta2**self.t)
        else:
            # 标准 Muno
            v_hat = self.v / (1 - self.beta2**self.t)
        
        # 计算更新步长
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return step


class MunoW(Muno):
    """
    带权重衰减的 Muno 优化器 (MunoW)
    """
    
    def __init__(self, dim:int, lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8, 
                 weight_decay=1e-2, amsgrad=False):
        """
        初始化 MunoW 优化器
        
        Args:
            dim: 参数维度
            lr: 初始学习率
            beta1: 动量项的指数衰减率
            beta2: 梯度平方项的指数衰减率
            eps: 数值稳定性常数
            weight_decay: 权重衰减系数
            amsgrad: 是否使用 AMSGrad 变体
        """
        super().__init__(dim, lr, beta1, beta2, eps, amsgrad)
        self.weight_decay = weight_decay
        
    def update(self, grad:np.ndarray):
        """
        更新参数
        
        Args:
            grad: 当前梯度
            
        Returns:
            更新步长
        """
        self.t += 1
        
        # 更新动量（一阶矩估计）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # 更新梯度平方的累积（二阶矩估计）
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        
        # 偏差修正
        m_hat = self.m / (1 - self.beta1**self.t)
        
        if self.amsgrad:
            # AMSGrad: 维护历史最大值
            self.v_max = np.maximum(self.v_max, self.v)
            v_hat = self.v_max / (1 - self.beta2**self.t)
        else:
            # 标准 MunoW
            v_hat = self.v / (1 - self.beta2**self.t)
        
        # 计算更新步长，包含权重衰减
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * self.lr * self.m
        return step


def zeropower_via_newtonschulz5(G, steps: int = 5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a 
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose 
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at 
    zero even beyond the point where the iteration no longer converges all the way to one everywhere 
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T 
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model 
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2, "G must be at least 2-dimensional"
    
    # Coefficients for quintic iteration
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Work with a copy of G in float32
    X = G.astype(np.float32)
    
    # Transpose if needed (when rows > columns)
    transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = np.swapaxes(X, -2, -1)
        transposed = True
    
    # Ensure spectral norm is at most 1
    norm = np.linalg.norm(X, axis=(-2, -1), keepdims=True)
    X = X / (norm + 1e-7)
    
    # Perform the NS iterations
    for _ in range(steps):
        A = np.matmul(X, np.swapaxes(X, -2, -1))
        B = b * A + c * np.matmul(A, A)  # quintic computation
        X = a * X + np.matmul(B, X)
    
    # Transpose back if needed
    if transposed:
        X = np.swapaxes(X, -2, -1)
    
    return X


def muon_update(grad, momentum_buffer, beta=0.95, ns_steps=5, nesterov=True):
    """
    Muon update function that applies momentum and Newton-Schulz orthogonalization
    
    Args:
        grad: Current gradient
        momentum_buffer: Momentum buffer
        beta: Momentum coefficient
        ns_steps: Number of Newton-Schulz steps
        nesterov: Whether to use Nesterov momentum
    """
    # Update momentum buffer
    momentum_buffer = beta * momentum_buffer + (1 - beta) * grad
    
    # Apply Nesterov momentum or standard momentum
    if nesterov:
        update = grad * (1 - beta) + momentum_buffer * beta
    else:
        update = momentum_buffer
    
    # For convolutional filters (4D), reshape to 2D
    original_shape = update.shape
    if update.ndim == 4:
        update = update.reshape(len(update), -1)
    
    # Apply Newton-Schulz orthogonalization
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    
    # Rescale based on dimension ratio
    scale = max(1, update.shape[-2] / update.shape[-1]) ** 0.5
    update = update * scale
    
    # Reshape back to original shape if needed
    if update.shape != original_shape:
        update = update.reshape(original_shape)
    
    return update, momentum_buffer


class Muon(Base):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration.
    
    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    """
    
    def __init__(self, dim: int, lr=0.02, weight_decay=0, momentum=0.95, ns_steps=5):
        """
        Initialize Muon optimizer
        
        Args:
            dim: Parameter dimension
            lr: Learning rate
            weight_decay: Weight decay coefficient
            momentum: Momentum coefficient
            ns_steps: Number of Newton-Schulz steps
        """
        super().__init__(dim, lr)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.ns_steps = ns_steps
        
        # Initialize momentum buffer
        self.momentum_buffer = np.zeros(dim, dtype=np.float32)
        
    def update(self, grad: np.ndarray):
        """
        Update parameters using Muon optimization
        
        Args:
            grad: Current gradient
            
        Returns:
            Update step
        """
        self.t += 1
        
        # Apply weight decay
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.momentum_buffer
        
        # Apply Muon update
        update, self.momentum_buffer = muon_update(
            grad, self.momentum_buffer, 
            beta=self.momentum, 
            ns_steps=self.ns_steps
        )
        
        # Scale by learning rate
        return -self.lr * update


class AdamNS(Base):
    """
    Adam optimizer with Newton-Schulz orthogonalization post-processing
    """
    
    def __init__(self, dim: int, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, ns_steps=5):
        """
        Initialize AdamNS optimizer
        
        Args:
            dim: Parameter dimension
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            ns_steps: Number of Newton-Schulz steps for orthogonalization
        """
        super().__init__(dim, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.ns_steps = ns_steps
        
        # Initialize buffers
        self.buf1 = np.zeros(dim, dtype=np.float32)  # First moment estimate
        self.buf2 = np.zeros(dim, dtype=np.float32)  # Second moment estimate
        
    def adam_update(self, grad):
        """
        Standard Adam update
        """
        # Update biased first moment estimate
        self.buf1 = self.beta1 * self.buf1 + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        self.buf2 = self.beta2 * self.buf2 + (1 - self.beta2) * grad**2
        
        # Compute bias-corrected first moment estimate
        buf1c = self.buf1 / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        buf2c = self.buf2 / (1 - self.beta2**self.t)
        
        # Compute update
        return buf1c / (np.sqrt(buf2c) + self.eps)
        
    def update(self, grad: np.ndarray):
        """
        Update parameters using Adam with Newton-Schulz orthogonalization
        
        Args:
            grad: Current gradient
            
        Returns:
            Update step
        """
        self.t += 1
        
        # Standard Adam update
        update = self.adam_update(grad)
        
        # For higher dimensional parameters, apply Newton-Schulz orthogonalization
        if update.ndim >= 2:
            original_shape = update.shape
            # Reshape to 2D if needed
            if update.ndim > 2:
                update = update.reshape(-1, update.shape[-1])
            
            # Apply Newton-Schulz orthogonalization
            update = zeropower_via_newtonschulz5(update, steps=self.ns_steps)
            
            # Reshape back
            if update.shape != original_shape:
                update = update.reshape(original_shape)
        
        return self.lr * update