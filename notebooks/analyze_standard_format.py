"""
Standard Format Analyzer for Zernike Response Matrix
Analyzes .json, .npy, and .h5 files in the standard format.
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import SymLogNorm

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

def analyze_standard_format(base_path):
    """Analyze standard format files (.json, .npy, .h5).
    
    Args:
        base_path: Path to the zernike_response_matrix directory
        
    Returns:
        tuple: (matrix, variance_matrix, metadata_dict) or (None, None, None) if not found
    """
    print(f"=== Analyzing Standard Format in: {base_path} ===")
    
    # Initialize return values
    matrix = None
    variance_matrix = None
    metadata = {}
    
    # Check for standard files
    json_path = base_path / "zernike_response_matrix.json"
    npy_response = base_path / "zernike_response_matrix.response.npy"
    npy_variance = base_path / "zernike_response_matrix.variance.npy"
    h5_path = base_path / "zernike_response_matrix_mag0.h5"
    
    # Load JSON configuration if available
    if json_path.exists():
        print(f"Loading JSON config: {json_path}")
        with open(json_path) as f:
            metadata = json.load(f)
        print("JSON config loaded")
        # Show first few items
        for k, v in list(metadata.items())[:5]:
            print(f"  {k}: {v}")
    
    # Load .npy files if available
    if npy_response.exists():
        print(f"Loading response matrix: {npy_response}")
        matrix = np.load(npy_response)
        print(f"Response matrix shape: {matrix.shape}")
        
    if npy_variance.exists():
        print(f"Loading variance matrix: {npy_variance}")
        variance_matrix = np.load(npy_variance)
        print(f"Variance matrix shape: {variance_matrix.shape}")
    
    # Load .h5 file if available (and if we don't already have matrix data)
    if h5_path.exists() and matrix is None:
        print(f"Loading H5 file: {h5_path}")
        try:
            with h5py.File(h5_path, "r") as f:
                print(f"H5 keys: {list(f.keys())}")
                if "matrix" in f:
                    matrix = f["matrix"][:]
                    print(f"H5 matrix shape: {matrix.shape}")
                if "variance_matrix" in f and variance_matrix is None:
                    variance_matrix = f["variance_matrix"][:]
                    print(f"H5 variance matrix shape: {variance_matrix.shape}")
                
                # Extract metadata from H5
                if "metadata" in f:
                    meta = f["metadata"]
                    for k in meta.attrs:
                        metadata[k] = meta.attrs[k]
                    print("H5 metadata loaded")
        except Exception as e:
            print(f"Error loading H5 file: {e}")
    
    return matrix, variance_matrix, metadata

def visualize_standard_data(matrix, variance_matrix, metadata=None):
    """Create visualizations for standard format data.
    
    Args:
        matrix: Response matrix (WFS terms × SLM terms)
        variance_matrix: Variance matrix (same shape as matrix)
        metadata: Optional metadata dictionary
    """
    if matrix is None:
        print("No matrix data available for visualization")
        return
        
    print("\n=== Creating Visualizations ===")
    
    # Basic statistics
    print("=== Response Matrix Statistics ===")
    print(f"Shape: {matrix.shape}  (WFS terms × SLM terms)")
    print(f"Min: {matrix.min():.4f}, Max: {matrix.max():.4f}, Mean: {matrix.mean():.4f}")
    print(f"Non-zero elements: {np.count_nonzero(matrix)}/{matrix.size}")
    
    if variance_matrix is not None:
        print(f"\n=== Variance Matrix Statistics ===")
        print(f"Mean variance: {variance_matrix.mean():.6f}")
        print(f"Max variance: {variance_matrix.max():.6f}")
        stable_modes = (variance_matrix.mean(axis=0) < 0.01).sum()
        print(f"Stable modes (var < 0.01): {stable_modes}/{matrix.shape[1]}")
    
    # Visualizations
    if variance_matrix is not None:
        # Response and variance heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Response matrix
        ax = axes[0]
        vmax = np.max(np.abs(matrix)) * 0.8
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="YlOrRd",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("SLM Zernike Mode Index")
        ax.set_ylabel("WFS Component Index")
        ax.set_title("Response Matrix\n(SLM Zernike → WFS Response)")
        plt.colorbar(im, ax=ax, label="Response [a.u.]")
        
        # Variance matrix
        ax = axes[1]
        im2 = ax.imshow(
            variance_matrix,
            aspect="auto",
            cmap="YlOrRd",
        )
        ax.set_xlabel("SLM Zernike Mode Index")
        ax.set_ylabel("WFS Component Index")
        ax.set_title("Variance Matrix\n(Measurement Stability)")
        plt.colorbar(im2, ax=ax, label="Variance")
        plt.tight_layout()
        plt.show()
    else:
        # Only response matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = np.max(np.abs(matrix)) * 0.8
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="YlOrRd",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("SLM Zernike Mode Index")
        ax.set_ylabel("WFS Component Index")
        ax.set_title("Response Matrix\n(SLM Zernike → WFS Response)")
        plt.colorbar(im, ax=ax, label="Response [a.u.]")
        plt.tight_layout()
        plt.show()
    
    # Column analysis if we have both matrices
    if matrix is not None and variance_matrix is not None:
        print("\n=== Column Analysis ===")
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        n_modes = matrix.shape[1]
        modes = np.arange(n_modes)
        
        # 4a. L2 norms (response strength)
        col_norms = np.linalg.norm(matrix, axis=0)
        ax = axes[0, 0]
        ax.bar(modes, col_norms, color="steelblue", alpha=0.8)
        ax.set_xlabel("SLM Mode Index")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Response Amplitude per Mode")
        ax.grid(True, alpha=0.3)
        
        # 4b. Average variance (stability)
        col_var = np.mean(variance_matrix, axis=0)
        ax = axes[0, 1]
        ax.bar(modes, col_var, color="tomato", alpha=0.8)
        ax.set_xlabel("SLM Mode Index")
        ax.set_ylabel("Mean Variance")
        ax.set_title("Measurement Stability per Mode")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        
        # 4c. Signal-to-noise ratio
        snr = col_norms / (col_var + 1e-12)
        ax = axes[1, 0]
        ax.bar(modes, snr, color="seagreen", alpha=0.8)
        ax.set_xlabel("SLM Mode Index")
        ax.set_ylabel("SNR (L2 norm / variance)")
        ax.set_title("Signal-to-Noise Ratio per Mode")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        
        # 4d. Adjacent mode correlation (orthogonality check)
        corr = []
        for i in range(1, n_modes):
            c = np.dot(matrix[:, i], matrix[:, i-1])
            c /= (np.linalg.norm(matrix[:, i]) * np.linalg.norm(matrix[:, i-1]) + 1e-12)
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
    
    # SVD analysis
    if matrix is not None:
        print("\n=== SVD Analysis ===")
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
        
        return cond, n95, n99, s
    
    return None, None, None, None

def compare_formats(matrix, variance_matrix, h5_path=None):
    """Compare NPY and H5 formats if both are available.
    
    Args:
        matrix: NPY format matrix
        variance_matrix: NPY format variance matrix
        h5_path: Path to H5 file for comparison
    """
    if matrix is None or variance_matrix is None or h5_path is None or not h5_path.exists():
        print("Skipping format comparison - missing required data")
        return
        
    print("\n=== Format Comparison (NPY vs H5) ===")
    try:
        with h5py.File(h5_path, "r") as f:
            matrix_h5 = f["matrix"][:]
            variance_h5 = f["variance_matrix"][:] if "variance_matrix" in f else None
        
        print(f"NPY matrix shape: {matrix.shape}")
        print(f"H5 matrix shape: {matrix_h5.shape}")
        
        if matrix.shape == matrix_h5.shape:
            diff = matrix - matrix_h5
            print(f"Matrix difference - Min: {diff.min():.6f}, Max: {diff.max():.6f}, Mean: {diff.mean():.6f}")
            
            if variance_h5 is not None and variance_matrix.shape == variance_h5.shape:
                var_diff = variance_matrix - variance_h5
                print(f"Variance difference - Min: {var_diff.min():.6f}, Max: {var_diff.max():.6f}, Mean: {var_diff.mean():.6f}")
            else:
                print("Variance matrices not compatible for comparison")
        else:
            print("Matrix shapes incompatible for direct comparison")
            
    except Exception as e:
        print(f"Error during format comparison: {e}")

if __name__ == "__main__":
    # When run directly, analyze the standard format in the expected location
    base_path = Path("../data/zernike_response_matrix")
    if not base_path.exists():
        base_path = Path("data/zernike_response_matrix")
    
    matrix, variance_matrix, metadata = analyze_standard_format(base_path)
    visualize_standard_data(matrix, variance_matrix, metadata)
    
    # Try format comparison if H5 file exists
    h5_path = base_path / "zernike_response_matrix_mag0.h5"
    compare_formats(matrix, variance_matrix, h5_path)