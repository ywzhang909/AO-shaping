"""
Zernike Response Matrix Analyzer

分析 SLM Zernike 命令 → WFS Zernike 响应的校准数据。
支持两种数据格式:
  1. HDF5 (.h5) 文件: data/zernike_response_matrix_mag0.h5
  2. 目录数据: data/zernike_response_matrix/ (含 .json/.npy/.h5)

参考 ao_shaping.optimizer.wf.zernike_response_matrix 的数据结构。
"""

# %%
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import SymLogNorm

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# %%
# ============================================================
# 1. 加载数据
# ============================================================

# --- 方式A: 加载 HDF5 文件 ---
h5_path = Path("data/zernike_response_matrix_mag0.h5")
if h5_path.exists():
    print(f"=== 加载 H5 文件: {h5_path} ===")
    with h5py.File(h5_path, "r") as f:
        print("顶层 keys:", list(f.keys()))
        matrix = f["matrix"][:]           # (64, 42) DM actuator x Zernike
        variance_matrix = f["variance_matrix"][:]  # (64, 42)
        subaperture_mask = f["subaperture_mask"][:] if "subaperture_mask" in f else None
        deviation_matrix = (
            f["deviation_response_matrix"][:] if "deviation_response_matrix" in f else None
        )
        print(f"matrix shape: {matrix.shape}")
        print(f"variance_matrix shape: {variance_matrix.shape}")
        if subaperture_mask is not None:
            print(f"subaperture_mask shape: {subaperture_mask.shape}, valid: {subaperture_mask.sum()}/{subaperture_mask.size}")
        if deviation_matrix is not None:
            print(f"deviation_response_matrix shape: {deviation_matrix.shape}")

        # 读取 metadata attrs
        meta = f["metadata"] if "metadata" in f else None
        if meta is not None:
            print("\n--- Metadata ---")
            for k in meta.attrs:
                print(f"  {k}: {meta.attrs[k]}")

# --- 方式B: 加载目录数据 ---
dir_path = Path("data/zernike_response_matrix")
if dir_path.exists() and dir_path.is_dir():
    print(f"\n=== 加载目录数据: {dir_path} ===")
    files = list(dir_path.glob("*"))
    print("文件列表:", [f.name for f in files])

    # 读取 JSON 配置
    json_path = dir_path / "zernike_response_matrix.json"
    if json_path.exists():
        with open(json_path) as f:
            cfg = json.load(f)
        print("\n--- JSON 配置 ---")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

    # 读取 .npy 文件
    npy_response = dir_path / "zernike_response_matrix.response.npy"
    npy_variance = dir_path / "zernike_response_matrix.variance.npy"
    if npy_response.exists():
        matrix_dir = np.load(npy_response)
        print(f"\nresponse.npy shape: {matrix_dir.shape}")
    if npy_variance.exists():
        variance_dir = np.load(npy_variance)
        print(f"variance.npy shape: {variance_dir.shape}")

    # 读取 .h5 文件 (如果存在)
    h5_in_dir = next(dir_path.glob("*.h5"), None)
    if h5_in_dir:
        print(f"\n--- 目录内 H5: {h5_in_dir.name} ---")
        with h5py.File(h5_in_dir, "r") as f:
            print("keys:", list(f.keys()))

# %%
# ============================================================
# 2. 基本统计
# ============================================================

if "matrix" in dir():
    print("=== 响应矩阵统计 ===")
    print(f"Shape: {matrix.shape}  (WFS terms × SLM terms)")
    print(f"Min: {matrix.min():.4f}, Max: {matrix.max():.4f}, Mean: {matrix.mean():.4f}")
    print(f"Non-zero elements: {np.count_nonzero(matrix)}/{matrix.size}")

if "variance_matrix" in dir():
    print(f"\n=== 方差矩阵统计 ===")
    print(f"Mean variance: {variance_matrix.mean():.6f}")
    print(f"Max variance: {variance_matrix.max():.6f}")
    print(f"Stable modes (var < 0.01): {(variance_matrix.mean(axis=0) < 0.01).sum()}/{matrix.shape[1]}")

# %%
# ============================================================
# 3. 响应矩阵热图 (主图)
# ============================================================

if "matrix" in dir():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 响应矩阵
    ax = axes[0]
    vmax = np.max(np.abs(matrix)) * 0.8
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("SLM Zernike Mode Index")
    ax.set_ylabel("DM Actuator Index")
    ax.set_title("Response Matrix\n(SLM Zernike → DM Response)")
    plt.colorbar(im, ax=ax, label="Response [a.u.]")

    # 右: 方差矩阵 (log scale)
    ax = axes[1]
    im2 = ax.imshow(
        variance_matrix,
        aspect="auto",
        cmap="YlOrRd",
        norm=SymLogNorm(linthresh=variance_matrix.mean() * 0.1, vmin=0),
    )
    ax.set_xlabel("SLM Zernike Mode Index")
    ax.set_ylabel("DM Actuator Index")
    ax.set_title("Variance Matrix\n(Measurement Stability)")
    plt.colorbar(im2, ax=ax, label="Variance")
    plt.tight_layout()
    plt.show()

# %%
# ============================================================
# 4. 逐列分析 (每个 Zernike 模式的响应质量)
# ============================================================

if "matrix" in dir():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    n_modes = matrix.shape[1]
    modes = np.arange(n_modes)

    # 4a. 每列的 L2 范数 (总体响应强度)
    col_norms = np.linalg.norm(matrix, axis=0)
    ax = axes[0, 0]
    ax.bar(modes, col_norms, color="steelblue", alpha=0.8)
    ax.set_xlabel("SLM Mode Index")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Response Amplitude per Mode")
    ax.grid(True, alpha=0.3)

    # 4b. 每列平均方差 (稳定性)
    col_var = np.mean(variance_matrix, axis=0)
    ax = axes[0, 1]
    ax.bar(modes, col_var, color="tomato", alpha=0.8)
    ax.set_xlabel("SLM Mode Index")
    ax.set_ylabel("Mean Variance")
    ax.set_title("Measurement Stability per Mode")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 4c. 信噪比 (响应幅度 / 平均方差)
    snr = col_norms / (col_var + 1e-12)
    ax = axes[1, 0]
    ax.bar(modes, snr, color="seagreen", alpha=0.8)
    ax.set_xlabel("SLM Mode Index")
    ax.set_ylabel("SNR (L2 norm / variance)")
    ax.set_title("Signal-to-Noise Ratio per Mode")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 4d. 响应向量相关性 (相邻模式的内积)
    corr = []
    for i in range(1, n_modes):
        c = np.dot(matrix[:, i], matrix[:, i - 1])
        c /= np.linalg.norm(matrix[:, i]) * np.linalg.norm(matrix[:, i - 1]) + 1e-12
        corr.append(c)
    ax = axes[1, 1]
    ax.plot(range(1, n_modes), corr, "o-", color="purple", alpha=0.8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Mode Pair (i, i-1)")
    ax.set_ylabel("Correlation")
    ax.set_title("Orthogonality Check\n(Adjacent Mode Correlation)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
# ============================================================
# 5. 子孔径掩膜可视化
# ============================================================

if "subaperture_mask" in dir() and subaperture_mask is not None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(subaperture_mask, cmap="gray")
    ax.set_title(
        f"Valid Subaperture Mask\n"
        f"valid={subaperture_mask.sum()}/{subaperture_mask.size} "
        f"({subaperture_mask.sum()/subaperture_mask.size*100:.1f}%)"
    )
    ax.set_xlabel("Subaperture X")
    ax.set_ylabel("Subaperture Y")
    plt.tight_layout()
    plt.show()
else:
    print("No subaperture_mask found in data.")

# %%
# ============================================================
# 6. 偏离响应矩阵 (deviation_response_matrix) 分析
# ============================================================

if "deviation_matrix" in dir() and deviation_matrix is not None:
    n_spots = deviation_matrix.shape[0] // 2  # dev_x + dev_y
    rows = int(np.sqrt(n_spots))
    cols = (n_spots + rows - 1) // rows
    n_rows_actual = int(np.sqrt(n_spots))
    n_cols_actual = int(np.ceil(n_spots / n_rows_actual))

    fig, axes = plt.subplots(
        nrows=2, ncols=n_cols_actual, figsize=(2.5 * n_cols_actual, 4)
    )
    fig.suptitle("Spot Deviation Response (first 8 modes)", fontsize=12)

    for m in range(min(8, deviation_matrix.shape[1])):
        r, c = m // n_cols_actual, m % n_cols_actual
        ax = axes[r, c]
        dev = deviation_matrix[:, m]
        n_dev = len(dev) // 2
        dev_x = dev[:n_dev].reshape(27, -1) if n_dev == 27 * (deviation_matrix.shape[0] // 2 // 27) else dev[:n_dev]
        dev_y = dev[n_dev:2 * n_dev] if len(dev) > n_dev else dev[n_dev:]

        v = np.max(np.abs(dev)) + 1e-12
        ax.imshow(dev_x.reshape(-1, n_dev // n_rows_actual) if len(dev_x) > 0 else np.zeros((27, 27)),
                  cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"Mode {m}")
        ax.axis("off")

    # Hide unused subplots
    total = 2 * n_cols_actual
    for idx in range(min(8, deviation_matrix.shape[1]), total):
        r, c = idx // n_cols_actual, idx % n_cols_actual
        if r < 2:
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No deviation_response_matrix found in data.")

# %%
# ============================================================
# 7. SVD 奇异值分析 (矩阵条件数)
# ============================================================

if "matrix" in dir():
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    cond = s[0] / s[-1] if s[-1] > 0 else np.inf

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(s, "o-", color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_title(f"SVD Singular Values\nCondition Number = {cond:.2e}")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    explained = np.cumsum(s**2) / np.sum(s**2)
    ax.plot(explained, "o-", color="seagreen")
    ax.axhline(0.95, color="red", linestyle="--", label="95% energy")
    ax.axhline(0.99, color="orange", linestyle="--", label="99% energy")
    n95 = np.searchsorted(explained, 0.95) + 1
    n99 = np.searchsorted(explained, 0.99) + 1
    ax.set_xlabel("Number of Modes")
    ax.set_ylabel("Cumulative Energy Ratio")
    ax.set_title(f"Energy Concentration\n95%: {n95} modes, 99%: {n99} modes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
# ============================================================
# 8. 目录数据 (.npy) 对比分析 (如果有)
# ============================================================

if "npy_response" in dir() and npy_response.exists():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 8a. H5 vs NPY 矩阵差异
    diff = matrix_dir - matrix if "matrix" in dir() and matrix_dir.shape == matrix.shape else None

    ax = axes[0]
    im = ax.imshow(matrix_dir, aspect="auto", cmap="RdBu_r")
    ax.set_title(f"NPY Response Matrix\nshape={matrix_dir.shape}")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(np.abs(matrix_dir), aspect="auto", cmap="viridis")
    ax.set_title("Absolute Response")
    plt.colorbar(im, ax=ax)

    if diff is not None:
        ax = axes[2]
        v = np.max(np.abs(diff))
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title("H5 vs NPY Difference")
        plt.colorbar(im, ax=ax)
    else:
        axes[2].text(
            0.5, 0.5,
            f"NPY shape: {matrix_dir.shape}\nH5 shape: {matrix.shape if 'matrix' in dir() else 'N/A'}",
            ha="center", va="center", fontsize=11,
        )
        axes[2].set_title("Shape Comparison")

    plt.tight_layout()
    plt.show()

# %%
# ============================================================
# 9. Zernike 模式命名映射 & 单模式响应剖面图
# ============================================================

from ao_shaping.utils.zernike_calc import ZernikeGenerator, ZERNIKE_NAMES

# %%
# 模式标签辅助函数
def get_mode_label(idx: int) -> str:
    """获取模式标签 (简化版)"""
    offset = 3
    noll = idx + offset + 1
    mapping = {
        4: "Defocus", 5: "Astig 45", 6: "Astig 0",
        7: "Coma Y", 8: "Coma X", 9: "Trefoil Y",
        10: "Trefoil X", 11: "Spherical", 12: "Sec Astig 45",
    }
    return mapping.get(noll, f"Z{noll}")

if "matrix" in dir():
    n_max = 10
    gen = ZernikeGenerator(resolution=(512, 512), n_orders=n_max)

    # 生成模式名称列表
    n_total = gen._cart.nk  # 总Zernike项数
    mode_names = []
    for j in range(matrix.shape[1]):
        # 注意: 排除piston/tip-tilt后的偏移
        # 根据 noll_offset = (1 if excluded_piston else 0) + (2 if excluded_tip_tilt else 0)
        # 假设 excluded_piston=True, excluded_tip_tilt=True → offset=3
        offset = 3  # 这个需要根据实际数据调整
        noll_idx = j + offset + 1  # Noll index (1-based)
        n, m = gen.noll_to_nm(noll_idx)
        name = ZERNIKE_NAMES.get((n, m), f"Z{noll_idx}")
        mode_names.append(f"{name} (n={n},m={m})")

    print("=== Zernike Mode Names ===")
    for i, name in enumerate(mode_names[: min(20, len(mode_names))]):
        print(f"  Mode {i}: {name}")
    if len(mode_names) > 20:
        print(f"  ... ({len(mode_names) - 20} more modes)")

# %%
# ============================================================
# 10. Zernike 模式命名 + 单模式响应剖面图
# ============================================================

n_max = 10
gen = ZernikeGenerator(resolution=(512, 512), n_orders=n_max)

n_total = gen._cart.nk
mode_names = []
for j in range(matrix.shape[1]):
    offset = 3  # excluded_piston=True, excluded_tip_tilt=True → skip 3 terms
    noll_idx = j + offset + 1
    n, m = gen.noll_to_nm(noll_idx)
    name = ZERNIKE_NAMES.get((n, m), f"Z{noll_idx}")
    mode_names.append(f"{name} (n={n},m={m})")

print("=== Zernike Mode Names ===")
for i, name in enumerate(mode_names[: min(20, len(mode_names))]):
    print(f"  Mode {i}: {name}")
if len(mode_names) > 20:
    print(f"  ... ({len(mode_names) - 20} more modes)")

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for idx in range(min(12, matrix.shape[1])):
    ax = axes[idx]
    col = matrix[:, idx]
    ax.plot(col, ".-", lw=0.8, color="steelblue")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"Mode {idx}\n{get_mode_label(idx)}", fontsize=9)
    ax.grid(True, alpha=0.3)

for idx in range(12, len(axes)):
    axes[idx].axis("off")

plt.suptitle("Response Vector per Zernike Mode (DM Actuator Index)", fontsize=12)
plt.tight_layout()
plt.show()


# %%
# ============================================================
# 11. 综合统计报告
# ============================================================

if "matrix" in dir():
    print("=" * 50)
    print("       Zernike Response Matrix Summary")
    print("=" * 50)
    print(f"  数据源: {h5_path if h5_path.exists() else dir_path}")
    print(f"  矩阵维度: {matrix.shape[0]} × {matrix.shape[1]}")
    print(f"  响应范围: [{matrix.min():.4f}, {matrix.max():.4f}]")
    print(f"  平均响应: {matrix.mean():.4f} ± {matrix.std():.4f}")
    print(f"  平均方差: {variance_matrix.mean():.6f}")
    print(f"  条件数:   {cond:.2e}")
    print(f"  能量95%:  {n95} modes")
    print(f"  能量99%:  {n99} modes")
    print(f"  SVD 有效秩: {(s / s[0] > 1e-6).sum()}/{len(s)}")
    print("=" * 50)