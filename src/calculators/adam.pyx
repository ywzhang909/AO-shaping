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
        
    cpdef update(self, grad):
        pass


cdef class SGD(Base):
    cdef object m  # Use object instead of buffer type

    def __init__(self, int dim, DTYPE_DOUBLE_t lr=1.0):
        self.dim = dim
        self.lr = lr
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.t = 0

    cpdef update(self, grad):
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

    cpdef update(self, grad):
        cdef DTYPE_DOUBLE_t m_hat, v_hat
        cdef int i
        cdef cnp.float32_t[:] m_view = self.m
        cdef cnp.float32_t[:] v_view = self.v
        cdef cnp.float32_t[:] grad_view = grad

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
        
    cpdef update(self, grad):
        cdef DTYPE_DOUBLE_t m_hat, v_hat
        cdef int i
        cdef cnp.float32_t[:] m_view = self.m
        cdef cnp.float32_t[:] v_view = self.v
        cdef cnp.float32_t[:] grad_view = grad

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
        
    cpdef update(self, grad):
        cdef DTYPE_DOUBLE_t m_hat, v_hat, gamma, learning_rate
        cdef int i
        cdef cnp.float32_t[:] m_view = self.m
        cdef cnp.float32_t[:] v_view = self.v
        cdef cnp.float32_t[:] grad_view = grad

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

        gamma = np.mean(gammas)
        self.s = self.beta3 * self.s + (1 - self.beta3) * gamma
        learning_rate = gamma if gamma < self.s else self.s

        cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.empty(self.dim, dtype=np.float32)
        cdef cnp.float32_t[:] result_view = result
        for i in range(self.dim):
            result_view[i] = learning_rate * m_view[i] * m_hat
        return result


cdef cnp.ndarray[DTYPE_t, ndim=2] zeropower_via_newtonschulz5(cnp.ndarray[DTYPE_t, ndim=2] G, int steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2, "G must be at least 2-dimensional"

    # Coefficients for quintic iteration
    cdef DTYPE_DOUBLE_t a = 3.4445
    cdef DTYPE_DOUBLE_t b = -4.7750
    cdef DTYPE_DOUBLE_t c = 2.0315

    # Work with a copy of G in float32
    cdef cnp.ndarray[DTYPE_t, ndim=2] X = G.astype(np.float32)

    # Transpose if needed (when rows > columns)
    cdef bint transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = np.swapaxes(X, -2, -1)
        transposed = True

    # Ensure spectral norm is at most 1
    cdef cnp.ndarray[DTYPE_t, ndim=2] norm = np.linalg.norm(X, axis=(-2, -1), keepdims=True)
    X = X / (norm + 1e-7)

    # Perform the NS iterations
    cdef int k
    cdef cnp.ndarray[DTYPE_t, ndim=2] A, B
    for k in range(steps):
        A = np.matmul(X, np.swapaxes(X, -2, -1))
        B = b * A + c * np.matmul(A, A)  # quintic computation
        X = a * X + np.matmul(B, X)

    # Transpose back if needed
    if transposed:
        X = np.swapaxes(X, -2, -1)

    return X


cdef class Muno(Base):
    """
    Muno 优化器是一种结合了动量和自适应学习率的优化算法。
    它通过维护梯度的指数移动平均和梯度平方的指数移动平均来动态调整学习率，
    同时引入了额外的机制来稳定训练过程。
    """

    cdef public DTYPE_DOUBLE_t beta1
    cdef public DTYPE_DOUBLE_t beta2
    cdef public DTYPE_DOUBLE_t eps
    cdef public bint amsgrad
    cdef object m
    cdef object v
    cdef object v_max

    def __init__(
        self, int dim, DTYPE_DOUBLE_t lr=1.0, DTYPE_DOUBLE_t beta1=0.9, DTYPE_DOUBLE_t beta2=0.999, 
        DTYPE_DOUBLE_t eps=1e-8, bint amsgrad=False
    ):
        super().__init__(dim, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad

        # 初始化动量和梯度平方的累积变量
        self.m = np.zeros(self.dim, dtype=np.float32)  # 动量
        self.v = np.zeros(self.dim, dtype=np.float32)  # 梯度平方的累积
        self.v_max = np.zeros(self.dim, dtype=np.float32)  # AMSGrad 中的最大梯度平方

    cpdef update(self, grad):
        """
        更新参数

        Args:
            grad: 当前梯度

        Returns:
            更新步长
        """
        cdef DTYPE_DOUBLE_t m_hat, v_hat
        cdef int i
        cdef cnp.float32_t[:] m_view = self.m
        cdef cnp.float32_t[:] v_view = self.v
        cdef cnp.float32_t[:] v_max_view = self.v_max
        cdef cnp.float32_t[:] grad_view = grad

        self.t += 1

        # 更新动量(一阶矩估计)
        for i in range(self.dim):
            m_view[i] = self.beta1 * m_view[i] + (1 - self.beta1) * grad_view[i]

        # 更新梯度平方的累积（二阶矩估计）
        for i in range(self.dim):
            v_view[i] = self.beta2 * v_view[i] + (1 - self.beta2) * grad_view[i] * grad_view[i]

        # 偏差修正
        m_hat = 1.0 / (1 - self.beta1**self.t)

        if self.amsgrad:
            # AMSGrad: 维护历史最大值
            for i in range(self.dim):
                v_max_view[i] = v_max_view[i] if v_max_view[i] > v_view[i] else v_view[i]
            v_hat = 1.0 / (1 - self.beta2**self.t)
        else:
            # 标准 Muno
            v_hat = 1.0 / (1 - self.beta2**self.t)

        cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.empty(self.dim, dtype=np.float32)
        cdef cnp.float32_t[:] result_view = result
        for i in range(self.dim):
            if self.amsgrad:
                result_view[i] = self.lr * m_view[i] * m_hat / (sqrt(v_max_view[i] * v_hat) + self.eps)
            else:
                result_view[i] = self.lr * m_view[i] * m_hat / (sqrt(v_view[i] * v_hat) + self.eps)
        return result


cdef class MunoW(Muno):
    """
    带权重衰减的 Muno 优化器 (MunoW)
    """

    cdef public DTYPE_DOUBLE_t weight_decay

    def __init__(
        self,
        int dim,
        DTYPE_DOUBLE_t lr=1.0,
        DTYPE_DOUBLE_t beta1=0.9,
        DTYPE_DOUBLE_t beta2=0.999,
        DTYPE_DOUBLE_t eps=1e-8,
        DTYPE_DOUBLE_t weight_decay=1e-2,
        bint amsgrad=False,
    ):
        super().__init__(dim, lr, beta1, beta2, eps, amsgrad)
        self.weight_decay = weight_decay

    cpdef update(self, grad):
        """
        更新参数

        Args:
            grad: 当前梯度

        Returns:
            更新步长
        """
        cdef DTYPE_DOUBLE_t m_hat, v_hat
        cdef int i
        cdef cnp.float32_t[:] m_view = self.m
        cdef cnp.float32_t[:] v_view = self.v
        cdef cnp.float32_t[:] v_max_view = self.v_max
        cdef cnp.float32_t[:] grad_view = grad

        self.t += 1

        # 更新动量（一阶矩估计）
        for i in range(self.dim):
            m_view[i] = self.beta1 * m_view[i] + (1 - self.beta1) * grad_view[i]

        # 更新梯度平方的累积（二阶矩估计）
        for i in range(self.dim):
            v_view[i] = self.beta2 * v_view[i] + (1 - self.beta2) * grad_view[i] * grad_view[i]

        # 偏差修正
        m_hat = 1.0 / (1 - self.beta1**self.t)

        if self.amsgrad:
            # AMSGrad: 维护历史最大值
            for i in range(self.dim):
                v_max_view[i] = v_max_view[i] if v_max_view[i] > v_view[i] else v_view[i]
            v_hat = 1.0 / (1 - self.beta2**self.t)
        else:
            # 标准 MunoW
            v_hat = 1.0 / (1 - self.beta2**self.t)

        cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.empty(self.dim, dtype=np.float32)
        cdef cnp.float32_t[:] result_view = result
        for i in range(self.dim):
            if self.amsgrad:
                result_view[i] = (
                    self.lr * m_view[i] * m_hat / (sqrt(v_max_view[i] * v_hat) + self.eps)
                    + self.weight_decay * self.lr * m_view[i]
                )
            else:
                result_view[i] = (
                    self.lr * m_view[i] * m_hat / (sqrt(v_view[i] * v_hat) + self.eps)
                    + self.weight_decay * self.lr * m_view[i]
                )
        return result


cdef cnp.ndarray[DTYPE_t, ndim=2] zeropower_via_newtonschulz5_cython(cnp.ndarray[DTYPE_t, ndim=2] G, int steps=5):
    """
    Newton-Schulz iteration - pure Cython implementation for performance
    """
    cdef DTYPE_DOUBLE_t a = 3.4445
    cdef DTYPE_DOUBLE_t b = -4.7750
    cdef DTYPE_DOUBLE_t c_val = 2.0315
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] X = G.astype(np.float32)
    cdef bint transposed = False
    cdef int rows = X.shape[0]
    cdef int cols = X.shape[1]
    cdef DTYPE_DOUBLE_t norm_val = 0.0
    cdef int i, j, m
    cdef DTYPE_DOUBLE_t row_norm, sum_val
    cdef DTYPE_DOUBLE_t temp_norm
    
    if rows > cols:
        X = np.swapaxes(X, 0, 1)
        transposed = True
        rows, cols = cols, rows
    
    # Ensure spectral norm is at most 1
    for i in range(rows):
        row_norm = 0.0
        for j in range(cols):
            row_norm += X[i, j] * X[i, j]
        temp_norm = row_norm ** 0.5
        if temp_norm > norm_val:
            norm_val = temp_norm
    
    if norm_val > 1e-7:
        for i in range(rows):
            for j in range(cols):
                X[i, j] = X[i, j] / (norm_val + 1e-7)
    
    # Perform NS iterations
    cdef cnp.ndarray[DTYPE_t, ndim=2] A = np.empty((rows, rows), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=2] B = np.empty((rows, rows), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_new = np.empty((rows, cols), dtype=np.float32)
    cdef int k
    
    for k in range(steps):
        # A = X @ X.T
        for i in range(rows):
            for j in range(rows):
                sum_val = 0.0
                for m in range(cols):
                    sum_val += X[i, m] * X[j, m]
                A[i, j] = sum_val
        
        # B = b * A + c * A @ A
        for i in range(rows):
            for j in range(rows):
                sum_val = 0.0
                for m in range(rows):
                    sum_val += A[i, m] * A[m, j]
                B[i, j] = b * A[i, j] + c_val * sum_val
        
        # X = a * X + B @ X
        for i in range(rows):
            for j in range(cols):
                sum_val = 0.0
                for m in range(rows):
                    sum_val += B[i, m] * X[m, j]
                X_new[i, j] = a * X[i, j] + sum_val
        X = X_new
    
    if transposed:
        X = np.swapaxes(X, 0, 1)
    
    return X


cdef class Muon(Base):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    """

    cdef public DTYPE_DOUBLE_t weight_decay
    cdef public DTYPE_DOUBLE_t momentum
    cdef public int ns_steps
    cdef object momentum_buffer

    def __init__(self, int dim, DTYPE_DOUBLE_t lr=0.02, DTYPE_DOUBLE_t weight_decay=0, DTYPE_DOUBLE_t momentum=0.95, int ns_steps=5):
        super().__init__(dim, lr)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.ns_steps = ns_steps

        # Initialize momentum buffer
        self.momentum_buffer = np.zeros(dim, dtype=np.float32)

    cpdef update(self, grad):
        """
        Update parameters using Muon optimization

        Args:
            grad: Current gradient

        Returns:
            Update step
        """
        cdef cnp.ndarray[DTYPE_t, ndim=1] update
        cdef int i
        cdef cnp.float32_t[:] momentum_buffer_view = self.momentum_buffer
        cdef cnp.float32_t[:] grad_view = grad

        self.t += 1

        # Apply weight decay
        if self.weight_decay > 0:
            for i in range(self.dim):
                grad_view[i] = grad_view[i] + self.weight_decay * momentum_buffer_view[i]

        # Apply Muon update - first update momentum buffer
        for i in range(self.dim):
            momentum_buffer_view[i] = self.momentum * momentum_buffer_view[i] + (1 - self.momentum) * grad_view[i]

        # Apply Nesterov momentum
        update = np.empty(self.dim, dtype=np.float32)
        cdef cnp.float32_t[:] update_view = update
        for i in range(self.dim):
            update_view[i] = grad_view[i] * (1 - self.momentum) + momentum_buffer_view[i] * self.momentum

        # For 1D vectors, we reshape to 2D for Newton-Schulz
        cdef cnp.ndarray[DTYPE_t, ndim=2] update_2d = update.reshape(1, self.dim)
        update_2d = zeropower_via_newtonschulz5_cython(update_2d, steps=self.ns_steps)
        
        # Rescale based on dimension ratio
        cdef DTYPE_DOUBLE_t scale = max(1, 1.0 / self.dim) ** 0.5
        for i in range(self.dim):
            update_view[i] = update_2d[0, i] * scale

        # Scale by learning rate
        for i in range(self.dim):
            update_view[i] = -self.lr * update_view[i]

        return update


cdef class AdamNS(Base):
    """
    Adam optimizer with Newton-Schulz orthogonalization post-processing
    """

    cdef public DTYPE_DOUBLE_t beta1
    cdef public DTYPE_DOUBLE_t beta2
    cdef public DTYPE_DOUBLE_t eps
    cdef public int ns_steps
    cdef object buf1
    cdef object buf2

    def __init__(self, int dim, DTYPE_DOUBLE_t lr=1e-3, DTYPE_DOUBLE_t beta1=0.9, DTYPE_DOUBLE_t beta2=0.999, DTYPE_DOUBLE_t eps=1e-8, int ns_steps=5):
        super().__init__(dim, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ns_steps = ns_steps

        # Initialize buffers
        self.buf1 = np.zeros(dim, dtype=np.float32)  # First moment estimate
        self.buf2 = np.zeros(dim, dtype=np.float32)  # Second moment estimate

    cpdef update(self, grad):
        """
        Update parameters using Adam with Newton-Schulz orthogonalization

        Args:
            grad: Current gradient (1D or higher)

        Returns:
            Update step
        """
        cdef int i
        cdef DTYPE_DOUBLE_t beta1_t, beta2_t
        cdef DTYPE_DOUBLE_t buf1c, buf2c
        cdef int ndim = grad.ndim
        cdef int dim = self.dim
        cdef cnp.float32_t[:] buf1_view = self.buf1
        cdef cnp.float32_t[:] buf2_view = self.buf2
        cdef cnp.float32_t[:] grad_view
        cdef cnp.ndarray[DTYPE_t, ndim=1] result
        cdef cnp.float32_t[:] result_view
        cdef cnp.ndarray[DTYPE_t, ndim=1] update
        cdef cnp.float32_t[:] update_view
        cdef cnp.ndarray[DTYPE_t, ndim=2] update_2d
        cdef cnp.ndarray[DTYPE_t, ndim=1] grad_flat
        
        self.t += 1

        # Handle 1D case (standard Adam)
        if ndim == 1:
            grad_view = grad
            result = np.empty(dim, dtype=np.float32)
            result_view = result

            for i in range(dim):
                buf1_view[i] = self.beta1 * buf1_view[i] + (1 - self.beta1) * grad_view[i]

            for i in range(dim):
                buf2_view[i] = self.beta2 * buf2_view[i] + (1 - self.beta2) * grad_view[i] * grad_view[i]

            beta1_t = self.beta1 ** self.t
            beta2_t = self.beta2 ** self.t
            
            for i in range(dim):
                buf1c = buf1_view[i] / (1 - beta1_t)
                buf2c = buf2_view[i] / (1 - beta2_t)
                result_view[i] = self.lr * buf1c / (sqrt(buf2c) + self.eps)

            return result
        else:
            # For 2D+ inputs, flatten and apply NS orthogonalization
            grad_flat = grad.ravel()
            grad_view = grad_flat
            update = np.empty(dim, dtype=np.float32)
            update_view = update

            for i in range(dim):
                buf1_view[i] = self.beta1 * buf1_view[i] + (1 - self.beta1) * grad_view[i]

            for i in range(dim):
                buf2_view[i] = self.beta2 * buf2_view[i] + (1 - self.beta2) * grad_view[i] * grad_view[i]

            beta1_t = self.beta1 ** self.t
            beta2_t = self.beta2 ** self.t
            
            for i in range(dim):
                buf1c = buf1_view[i] / (1 - beta1_t)
                buf2c = buf2_view[i] / (1 - beta2_t)
                update_view[i] = buf1c / (sqrt(buf2c) + self.eps)

            # Apply Newton-Schulz orthogonalization for higher dimensional params
            update_2d = update.reshape(1, dim)
            update_2d = zeropower_via_newtonschulz5(update_2d, steps=self.ns_steps)
            
            result = np.empty(dim, dtype=np.float32)
            result_view = result
            for i in range(dim):
                result_view[i] = self.lr * update_2d[0, i]

            return result
