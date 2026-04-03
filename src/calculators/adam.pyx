# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, cos, exp
from typing import Literal

cnp.import_array()

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.float64_t DTYPE_DOUBLE_t

def learning_schedule(
    DTYPE_t lr, 
    int epoch, 
    int epochs, 
    method: Literal["static", "cosin", "exp", "linear"] = "static"
):
    if method == "static":
        return lr
    # 余弦退火
    elif method == "cosin":
        lr = lr * cos(3.141592653589793 * epoch / epochs) + 1e-6
        return lr
    # 指数衰减
    elif method == "exp":
        lr = <DTYPE_t>(lr * exp(<DTYPE_DOUBLE_t>(-epoch / epochs)) + 1e-6)
        return lr
    # 线性衰减
    elif method == "linear":
        lr = lr * (1 - epoch / epochs) + 1e-6
        return lr
    else:
        raise ValueError("method must be static, cosin, exp or linear")


cdef class Base:
    cdef public int dim
    cdef public DTYPE_DOUBLE_t lr
    cdef public int t
    
    def __init__(self, int dim, DTYPE_DOUBLE_t lr=1.0):
        self.dim = dim
        self.lr = lr
        self.t = 0
        
    cpdef update(self, cnp.ndarray[DTYPE_t, ndim=1] grad):
        pass


cdef class SGD(Base):
    cdef object m  # Use object instead of buffer type

    def __init__(self, int dim, DTYPE_DOUBLE_t lr=1.0):
        self.dim = dim
        self.lr = lr
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.t = 0

    cpdef update(self, cnp.ndarray[DTYPE_t, ndim=1] grad):
        self.t += 1
        return self.lr * grad


cdef class Adam(Base):
    """
    使用 EMA 来估计二阶矩。这意味着它会遗忘早期的梯度信息。这使得 Adam 的自适应性更强，可以快速适应梯度的局部变化。
    """
    cdef public DTYPE_DOUBLE_t beta1
    cdef public DTYPE_DOUBLE_t beta2
    cdef object m  # Use object instead of buffer type
    cdef object v  # Use object instead of buffer type

    def __init__(self, int dim, DTYPE_DOUBLE_t lr=1.0, DTYPE_DOUBLE_t beta1 = 0.9, DTYPE_DOUBLE_t beta2 = 0.99):
        self.dim = dim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.t = 0

    cpdef update(self, cnp.ndarray[DTYPE_t, ndim=1] grad):
        cdef DTYPE_DOUBLE_t m_hat, v_hat
        cdef int i
        cdef cnp.float32_t[:] m_view = self.m  # Use memoryview for performance
        cdef cnp.float32_t[:] v_view = self.v  # Use memoryview for performance
        cdef cnp.float32_t[:] grad_view = grad  # Use memoryview for performance

        self.t += 1
        for i in range(self.dim):
            m_view[i] = self.beta1 * m_view[i] + (1 - self.beta1) * grad_view[i]
            v_view[i] = self.beta2 * v_view[i] + (1 - self.beta2) * grad_view[i] * grad_view[i]

        m_hat = 1.0 / (1 - self.beta1**self.t)
        v_hat = 1.0 / (1 - self.beta2**self.t)

        cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.empty(self.dim, dtype=np.float32)
        cdef cnp.float32_t[:] result_view = result
        for i in range(self.dim):
            result_view[i] = self.lr * (m_view[i] * m_hat) / (sqrt(v_view[i] * v_hat) + 1e-8)
        return result
    
cdef class AdamW(Adam):
    cdef public DTYPE_DOUBLE_t weight_decay
    
    def __init__(self, int dim, DTYPE_DOUBLE_t lr=1.0, DTYPE_DOUBLE_t beta1 = 0.9, DTYPE_DOUBLE_t beta2 = 0.99, DTYPE_DOUBLE_t weight_decay=1e-2):
        super().__init__(dim, lr, beta1, beta2)
        self.weight_decay = weight_decay
        
    cpdef update(self, cnp.ndarray[DTYPE_t, ndim=1] grad):
        cdef DTYPE_DOUBLE_t m_hat, v_hat
        cdef int i
        cdef cnp.float32_t[:] m_view = self.m  # Use memoryview for performance
        cdef cnp.float32_t[:] v_view = self.v  # Use memoryview for performance
        cdef cnp.float32_t[:] grad_view = grad  # Use memoryview for performance

        self.t += 1
        for i in range(self.dim):
            m_view[i] = self.beta1 * m_view[i] + (1 - self.beta1) * grad_view[i]
            v_view[i] = self.beta2 * v_view[i] + (1 - self.beta2) * grad_view[i] * grad_view[i]

        m_hat = 1.0 / (1 - self.beta1**self.t)
        v_hat = 1.0 / (1 - self.beta2**self.t)

        cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.empty(self.dim, dtype=np.float32)
        cdef cnp.float32_t[:] result_view = result
        for i in range(self.dim):
            result_view[i] = self.lr * (m_view[i] * m_hat) / (sqrt(v_view[i] * v_hat) + 1e-8) + self.weight_decay * self.lr * m_view[i]
        return result


cdef class AdaMOD(Adam):
    """
    AdaMod 是一个基于 Adam 的新的深度学习优化器，但它提供了自动warmup heuristic和长期学习率缓冲。 
    从最初的测试来看，AdaMod 是top 5的优化器，很容易击败或超过普通的 Adam，且对学习率超参数不那么敏感，训练曲线更平滑，不需要warmup模式。
    
    Pros:
    AdaMod保持了自适应学习率自身的指数长期平均值，并在整个训练过程中用这个值来clip任何过高的适应率。 
    结果改善了收敛性，不需要warmup，对实际学习率选择的敏感性较低。 记忆的程度由一个新的参数 Beta3控制。
    
    Cons:
    虽然AdaMod通常比普通的Adam表现更好，但是在更长的训练条件下，SGDM 仍然可能比AdaMod表现更好。
    
    """
    cdef public DTYPE_DOUBLE_t beta3
    cdef public DTYPE_DOUBLE_t s
    
    def __init__(self, int dim, DTYPE_DOUBLE_t lr=1.0, DTYPE_DOUBLE_t beta1 = 0.9, DTYPE_DOUBLE_t beta2 = 0.99, DTYPE_DOUBLE_t beta3 = 0.9995):
        super().__init__(dim, lr, beta1, beta2)
        self.beta3 = beta3
        self.s = <DTYPE_DOUBLE_t>0.0
        
    cpdef update(self, cnp.ndarray[DTYPE_t, ndim=1] grad):
        cdef DTYPE_DOUBLE_t m_hat, v_hat, gamma, learning_rate
        cdef int i
        cdef cnp.float32_t[:] m_view = self.m  # Use memoryview for performance
        cdef cnp.float32_t[:] v_view = self.v  # Use memoryview for performance
        cdef cnp.float32_t[:] grad_view = grad  # Use memoryview for performance

        self.t += 1
        for i in range(self.dim):
            m_view[i] = self.beta1 * m_view[i] + (1 - self.beta1) * grad_view[i]
            v_view[i] = self.beta2 * v_view[i] + (1 - self.beta2) * grad_view[i] * grad_view[i]

        m_hat = 1.0 / (1 - self.beta1 ** self.t)
        v_hat = 1.0 / (1 - self.beta2 ** self.t)

        cdef cnp.ndarray[DTYPE_t, ndim=1] gammas = np.empty(self.dim, dtype=np.float32)
        cdef cnp.float32_t[:] gammas_view = gammas
        for i in range(self.dim):
            gammas_view[i] = self.lr / (sqrt(v_view[i] * v_hat) + 1e-8)

        gamma = np.mean(gammas)  # 简化计算
        self.s = self.beta3 * self.s + (1 - self.beta3) * gamma
        learning_rate = gamma if gamma < self.s else self.s

        cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.empty(self.dim, dtype=np.float32)
        cdef cnp.float32_t[:] result_view = result
        for i in range(self.dim):
            result_view[i] = learning_rate * m_view[i] * m_hat
        return result
