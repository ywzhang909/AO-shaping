import numpy as np
from numba import njit, float32

from abc import ABC, abstractmethod

class Base(ABC):
    def __init__(self, dim:int, lr=1.0):
        self.dim = dim
        self.lr = lr
        self.t:int = 0
        
    @abstractmethod
    def update(self, grad:np.ndarray):
        pass


class SGD(Base):
    
    def __init__(self, dim:int, lr=1.0):
        self.dim = dim
        self.lr = lr
        self.m = np.zeros(self.dim, dtype=np.float32)
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
    
