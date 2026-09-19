import numpy as np
from scipy.sparse import dia_matrix
from numpy.linalg import svd, lstsq

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def zernike_piston_tilt(N):
    """返回 piston+x tilt+y tilt 的 3 个正交基，形状 (3,(N+1)**2)"""
    x = np.linspace(-1,1,N+1)
    X,Y = np.meshgrid(x,x)
    Z0 = np.ones_like(X).ravel() / np.sqrt((N+1)**2)          # piston
    Z1 = X.ravel() / np.sqrt(np.sum(X*X))                     # x-tilt
    Z2 = Y.ravel() / np.sqrt(np.sum(Y*Y))                     # y-tilt
    return np.vstack([Z0,Z1,Z2])                              # 3×K

def build_D_vectorized(N):
    """向量化构造稀疏差分矩阵 D，形状 (2*N*N, (N+1)*(N+1))"""
    M = N * N
    K = (N+1)*(N+1)
    j, i = np.divmod(np.arange(M), N)          # 子孔径坐标
    base = j*(N+1) + i                         # 节点 (i,j) 线性索引
    n1 = base                                  # (i,j)
    n2 = base + 1                              # (i+1,j)
    n3 = base + (N+1)                          # (i,j+1)
    n4 = base + (N+1) + 1                      # (i+1,j+1)

    # sx 部分：前 M 行
    row_sx = np.repeat(np.arange(M), 4)
    col_sx = np.stack([n1, n2, n3, n4], axis=0).ravel('F')
    data_sx = np.tile([-0.5, 0.5, -0.5, 0.5], M)

    # sy 部分：后 M 行
    row_sy = np.repeat(np.arange(M) + M, 4)
    col_sy = col_sx
    data_sy = np.tile([-0.5, -0.5, 0.5, 0.5], M)

    D = coo_matrix((np.r_[data_sx, data_sy],
                    (np.r_[row_sx, row_sy],
                     np.r_[col_sx, col_sy])), shape=(2*M, K))
    return D.tocsr()

def reconstruct_wavefront(sx, sy, remove_piston_tilt=True):
    """
    参数
    ----
    sx, sy : ndarray, shape=(N,N)  子孔径 x/y 斜率
    remove_piston_tilt : bool       是否去掉 piston+tilt

    返回
    ----
    phi   : 波前节点，shape=(N+1,N+1)
    rms   : 波前 RMS
    """
    N = sx.shape[0]
    M = N*N
    K = (N+1)*(N+1)

    # 1. 斜率拉向量
    s = np.concatenate([sx.ravel(), sy.ravel()])   # 2M×1

    # 2. 向量化构造 D（稀疏）
    D = build_D_vectorized(N)                      # 2M×K 稀疏 CSR

    # 3. 最小二乘解 φ
    phi_vec, *_ = lstsq(D.toarray(), s, rcond=None)  # K×1

    # 4. 去掉 piston+tilt
    if remove_piston_tilt:
        Z = zernike_piston_tilt(N)          # 3×K
        coeffs = Z @ phi_vec
        phi_vec -= Z.T @ coeffs

    # 5. 还原二维 & RMS
    phi = phi_vec.reshape(N+1, N+1)
    rms = float(np.std(phi))
    return phi, rms

# ---------------- 构造 2-D 拉普拉斯矩阵 ----------------
def laplacian_2d(N, boundary='neumann'):
    """
    返回 (N*N, N*N) 的稀疏拉普拉斯矩阵 L，Neumann 边界（默认）
    边界处理：用一阶镜像，使边界导数=0
    """
    e = np.ones(N)
    diag0 = -4*e
    diag1 = np.concatenate([e[:-1], [0]]) + np.concatenate([[0], e[:-1]])
    # Neumann：边界镜像 => 对角线±1 处系数减半
    diag1[0]   = 1
    diag1[-1]  = 1
    data = [diag0, diag1, diag1, e, e]
    offsets = [0, -1, 1, -N, N]
    L = dia_matrix((data, offsets), shape=(N*N, N*N))
    # 把 L 转成 CSR 方便后续 SVD
    return L.tocsr()

# ---------------- 斜率 → 右端项 ρ ----------------
def slope_to_rhs(sx, sy):
    """
    由 sx,sy 构造泊松右端项 ρ = div(s) = dx(sx) + dy(sy)
    边界用一阶前向/后向差分，兼容 Neumann
    """
    N = sx.shape[0]
    dx = np.zeros((N, N))
    dy = np.zeros((N, N))
    # 内部中心差
    dx[:, 1:-1] = (sx[:, 1:-1] - sx[:, :-2])
    dy[1:-1, :] = (sy[1:-1, :] - sy[:-2, :])
    # 边界：前向/后向一阶
    dx[:, 0]  = sx[:, 0]
    dx[:, -1] = -sx[:, -2]
    dy[0, :]  = sy[0, :]
    dy[-1, :] = -sy[-2, :]
    rho = (dx + dy).ravel()
    # 保证均值 0（Neumann 可解条件）
    rho -= rho.mean()
    return rho

# ---------------- SVD 求解 L·φ = ρ ----------------
def poisson_svd(L, rho, rcond=1e-12):
    """
    L  : (K,K) 稀疏矩阵，K=N*N
    rho: (K,)  右端项
    返回: φ (K,)  最小范数解（已去掉 piston）
    """
    # 把稀疏矩阵稠密化才能直接 SVD（小规模够用；超大矩阵再换稀疏 SVD）
    L_dense = L.toarray()
    U, s, Vt = svd(L_dense, full_matrices=False)
    # 截断零空间
    mask = s > rcond*s.max()
    inv_s = np.zeros_like(s)
    inv_s[mask] = 1.0 / s[mask]
    # 最小范数解: φ = V^T @ S^+ @ U^T @ rho
    phi = Vt.T @ (inv_s * (U.T @ rho))
    return phi

# ---------------- 主接口 ----------------
def reconstruct_poisson_svd(sx, sy, remove_tilt=True):
    """
    sx, sy: (N,N) 子孔径斜率
    返回: wavefront (N,N), rms
    """
    N = sx.shape[0]
    L = laplacian_2d(N, boundary='neumann')
    rho = slope_to_rhs(sx, sy)
    phi_vec = poisson_svd(L, rho)
    phi = phi_vec.reshape(N, N)

    if remove_tilt:
        # 去掉 piston+tilt（同 zernike 前 3 项）
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        basis = [np.ones_like(X), X, Y]
        coeff = [np.sum(phi*b)/np.sum(b*b) for b in basis]
        for c, b in zip(coeff, basis):
            phi -= c*b

    rms = float(np.std(phi))
    return phi, rms


# ---------------- demo ----------------
if __name__ == "__main__":
    N = 32
    # -------------- 伪造：X tilt + 球面 + 噪声 --------------
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    # 1. X 倾斜（只随 x 线性变化）
    tx = 0.5 * X

    # 2. 球面项（r²）
    sphere = 0.4 * (X**2 + Y**2)

    # 3. 高频随机噪声
    noise = 0.05 * np.random.randn(N, N)

    # 合成真实波前
    true_phi = tx + sphere + noise
    # 数值差分得到斜率
    sx = 0.5*(np.roll(true_phi, -1, axis=1) - np.roll(true_phi, 1, axis=1))[:, 1:-1]
    sy = 0.5*(np.roll(true_phi, -1, axis=0) - np.roll(true_phi, 1, axis=0))[1:-1, :]
    sx = sx[1:-1, :]              # 内部 N-2×N-2，补零到 N×N
    sy = sy[:, 1:-1]
    sx = np.pad(sx, 1, mode='constant')
    sy = np.pad(sy, 1, mode='constant')

    phi, rms = reconstruct_wavefront(sx, sy, True)
    phi_center = 0.25*(phi[:-1,:-1] + phi[1:,:-1] + phi[:-1,1:] + phi[1:,1:])

    plt.figure(figsize=(9,3))
    plt.subplot(231)
    plt.imshow(true_phi)
    plt.title('true (tilt+sphere+noise)')
    plt.subplot(232)
    plt.imshow(phi)
    plt.title('reconstructed')
    plt.subplot(233)
    plt.imshow(true_phi-phi_center, vmin=-0.05, vmax=0.05)
    plt.title("Recovered RMS = %.3f" % rms)
    plt.subplot(234)
    plt.imshow(sx)
    plt.title("sx")
    plt.subplot(235)
    plt.imshow(sy)
    plt.title("sy")
    plt.show()
