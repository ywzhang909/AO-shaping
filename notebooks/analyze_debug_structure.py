"""
Debug Directory Analyzer for Zernike Response Matrix
Analyzes the debug directory structure with mode_NNN/cycle_M/plus/minus/ organization.
"""

# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


def glob_sorted_samples(directory, suffix):
    """Get sorted list of sample files with given suffix in directory.
    
    Args:
        directory: Path to search in
        suffix: File suffix pattern (e.g., 'deviation_x.npy')
        
    Returns:
        Sorted list of matching file paths
    """
    return sorted(list(directory.glob(f"sample_*_{suffix}")))


def load_samples(file_list):
    """Load numpy arrays from list of file paths.
    
    Args:
        file_list: List of file paths to load
        
    Returns:
        List of loaded numpy arrays
    """
    return [np.load(f) for f in file_list]


def compute_median_diff(plus_data, minus_data):
    """Compute median of plus/minus arrays and return their difference.
    
    Args:
        plus_data: List of numpy arrays from plus samples
        minus_data: List of numpy arrays from minus samples
        
    Returns:
        plus_median - minus_median
    """
    plus_array = np.stack(plus_data, axis=0)
    minus_array = np.stack(minus_data, axis=0)
    return np.median(plus_array, axis=0) - np.median(minus_array, axis=0)


def compute_variance_sum(plus_data, minus_data):
    """Compute sum of variances from plus and minus data.
    
    Args:
        plus_data: List of numpy arrays from plus samples
        minus_data: List of numpy arrays from minus samples
        
    Returns:
        Sum of variances: Var(plus) + Var(minus)
    """
    plus_array = np.stack(plus_data, axis=0)
    minus_array = np.stack(minus_data, axis=0)
    return np.var(plus_array, axis=0) + np.var(minus_array, axis=0)


def load_debug_data(base_path):
    """Load raw sample data from debug directory structure.
    
    Args:
        base_path: Path to the zernike_response_matrix directory
        
    Returns:
        tuple: (raw_data, info) where raw_data is dict mapping mode_idx to 
               cycle data, each containing 'plus'/'minus' arrays for dev_x, dev_y, zernike
    """
    print(f"=== Loading Debug Directory Structure in: {base_path} ===")
    
    info = {}
    raw_data = {}
    
    # Find debug directories
    debug_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("debug_")]
    
    if not debug_dirs:
        print("No debug directories found")
        return None, info
    
    latest_debug = sorted(debug_dirs)[-1]
    print(f"Using debug directory: {latest_debug.name}")
    info['debug_directory'] = latest_debug.name
    
    mode_dirs = [d for d in latest_debug.iterdir() if d.is_dir() and d.name.startswith("mode_")]
    if not mode_dirs:
        print("No mode directories found in debug directory")
        return None, info
    
    n_modes = len(mode_dirs)
    print(f"Found {n_modes} mode directories")
    info['n_modes'] = n_modes
    
    first_mode = mode_dirs[0]
    cycle_dirs = [d for d in first_mode.iterdir() if d.is_dir() and d.name.startswith("cycle_")]
    if not cycle_dirs:
        print("No cycle directories found in first mode")
        return None, info
    
    first_cycle = cycle_dirs[0]
    plus_dir = first_cycle / "plus"
    minus_dir = first_cycle / "minus"
    
    if not (plus_dir.exists() and minus_dir.exists()):
        print("Plus/minus directories not found in first mode cycle")
        return None, info
    
    # Get sample count
    plus_samples = glob_sorted_samples(plus_dir, "slm_phase.npy")
    n_samples_per_cycle = len(plus_samples)
    print(f"Samples per cycle per mode: {n_samples_per_cycle}")
    info['samples_per_cycle'] = n_samples_per_cycle
    
    if not plus_samples:
        print("No sample files found")
        return None, info
    
    # Load dimension info
    try:
        sample_phase = np.load(glob_sorted_samples(plus_dir, "slm_phase.npy")[0])
        info['slm_phase_shape'] = sample_phase.shape
        
        sample_zernike_coeffs = np.load(glob_sorted_samples(plus_dir, "zernike_coeffs.npy")[0])
        info['zernike_coeffs_length'] = len(sample_zernike_coeffs)
        
        nonzero_indices = np.nonzero(sample_zernike_coeffs)[0]
        if len(nonzero_indices) > 0:
            info['active_zernike_indices'] = nonzero_indices.tolist()
            info['active_zernike_values'] = sample_zernike_coeffs[nonzero_indices].tolist()
        else:
            info['active_zernike_indices'] = []
            info['active_zernike_values'] = []
        
        dev_x = np.load(glob_sorted_samples(plus_dir, "deviation_x.npy")[0])
        info['wfs_deviation_length'] = len(dev_x)
        
        cycle_dirs_all = sorted([d for d in first_mode.iterdir() if d.is_dir() and d.name.startswith("cycle_")])
        info['n_cycles'] = len(cycle_dirs_all)
        
        # Load raw data for each mode and cycle
        print("Loading raw sample data...")
        for mode_idx, mode_dir in enumerate(mode_dirs):
            if mode_idx % 10 == 0:
                print(f"  Loading mode {mode_idx}/{n_modes}")
            
            mode_cycles = []
            cycle_dirs = sorted([d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("cycle_")])
            
            for cycle_dir in cycle_dirs:
                plus_cycle_dir = cycle_dir / "plus"
                minus_cycle_dir = cycle_dir / "minus"
                
                if not (plus_cycle_dir.exists() and minus_cycle_dir.exists()):
                    continue
                
                # Load plus files
                plus_dev_x = load_samples(glob_sorted_samples(plus_cycle_dir, "deviation_x.npy"))
                plus_dev_y = load_samples(glob_sorted_samples(plus_cycle_dir, "deviation_y.npy"))
                plus_zernike = load_samples(glob_sorted_samples(plus_cycle_dir, "zernike_coeffs.npy"))
                
                # Load minus files
                minus_dev_x = load_samples(glob_sorted_samples(minus_cycle_dir, "deviation_x.npy"))
                minus_dev_y = load_samples(glob_sorted_samples(minus_cycle_dir, "deviation_y.npy"))
                minus_zernike = load_samples(glob_sorted_samples(minus_cycle_dir, "zernike_coeffs.npy"))
                
                if plus_dev_x and minus_dev_x:
                    mode_cycles.append({
                        'plus': {'dev_x': plus_dev_x, 'dev_y': plus_dev_y, 'zernike': plus_zernike},
                        'minus': {'dev_x': minus_dev_x, 'dev_y': minus_dev_y, 'zernike': minus_zernike}
                    })
            
            if mode_cycles:
                raw_data[mode_idx] = mode_cycles
        
        print(f"Loaded raw data for {len(raw_data)} modes")
        
    except Exception as e:
        print(f"Error loading debug data: {e}")
        import traceback
        traceback.print_exc()
        return None, info
    
    return raw_data, info


def compute_statistics(raw_data, n_dev):
    """Compute response and variance matrices from raw sample data.
    
    Args:
        raw_data: Dictionary from load_debug_data mapping mode_idx to cycle data
        n_dev: WFS deviation vector length (for reference)
        
    Returns:
        tuple: (response_matrix, variance_matrix) both shaped (2*n_dev, n_modes)
    """
    if raw_data is None:
        return None, None
    
    print("\nComputing statistics from raw data...")
    
    all_mode_dev_medians = []
    all_mode_dev_variances = []
    
    for mode_idx in sorted(raw_data.keys()):
        mode_cycles = raw_data[mode_idx]
        cycle_dev_medians = []
        cycle_dev_variances = []
        
        for cycle_data in mode_cycles:
            plus_data = cycle_data['plus']
            minus_data = cycle_data['minus']
            
            # Compute median differences for dev_x and dev_y
            dev_x_diff = compute_median_diff(plus_data['dev_x'], minus_data['dev_x'])
            dev_y_diff = compute_median_diff(plus_data['dev_y'], minus_data['dev_y'])
            cycle_dev_medians.append(np.concatenate([dev_x_diff, dev_y_diff]))
            
            # Compute variance sum
            cycle_dev_variances.append(
                compute_variance_sum(plus_data['dev_x'], minus_data['dev_x']) +
                compute_variance_sum(plus_data['dev_y'], minus_data['dev_y'])
            )
        
        if cycle_dev_medians:
            all_mode_dev_medians.append(np.mean(np.array(cycle_dev_medians), axis=0))
            if cycle_dev_variances:
                all_mode_dev_variances.append(np.mean(np.array(cycle_dev_variances), axis=0))
    
    if all_mode_dev_medians:
        response_matrix = np.array(all_mode_dev_medians).T
        variance_matrix = np.array(all_mode_dev_variances).T
        print(f"Computed matrices: {response_matrix.shape}")
        return response_matrix, variance_matrix
    
    return None, None


def analyze_debug_structure(base_path):
    """Analyze debug directory structure - main entry point.
    
    This is a convenience wrapper that loads data and computes statistics.
    
    Args:
        base_path: Path to the zernike_response_matrix directory
        
    Returns:
        tuple: (matrix, variance_matrix, info_dict)
    """
    raw_data, info = load_debug_data(base_path)
    n_dev = info.get('wfs_deviation_length', 0)
    matrix, variance_matrix = compute_statistics(raw_data, n_dev)
    
    if matrix is not None and variance_matrix is not None:
        info['matrix_shape'] = matrix.shape
        info['matrix_min'] = float(matrix.min())
        info['matrix_max'] = float(matrix.max())
        info['matrix_mean'] = float(matrix.mean())
        info['matrix_std'] = float(matrix.std())
        info['variance_mean'] = float(variance_matrix.mean())
    
    return matrix, variance_matrix, info


def visualize_debug_data(matrix, variance_matrix, info=None):
    """Create visualizations for debug directory data.
    
    Args:
        matrix: Response matrix (WFS terms × SLM terms)
        variance_matrix: Variance matrix (same shape as matrix)
        info: Optional information dictionary from analysis
    """
    if matrix is None:
        print("No matrix data available for visualization")
        return
        
    print("\n=== Creating Visualizations ===")
    
    # Visualizations
    if variance_matrix is not None:
        # Response and variance heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Response matrix
        ax = axes[0]
        vmax = np.max(np.abs(matrix)) * 0.8
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("SLM Zernike Mode Index")
        ax.set_ylabel("WFS Component Index")
        ax.set_title("Response Matrix\n(SLM Zernike → WFS Response)")
        plt.colorbar(im, ax=ax, label="Response [a.u.]")
        
        # Variance matrix
        ax = axes[1]
        im2 = ax.imshow(variance_matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("SLM Zernike Mode Index")
        ax.set_ylabel("WFS Component Index")
        ax.set_title("Variance Matrix\n(Measurement Stability)")
        plt.colorbar(im2, ax=ax, label="Variance")
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = np.max(np.abs(matrix)) * 0.8
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("SLM Zernike Mode Index")
        ax.set_ylabel("WFS Component Index")
        ax.set_title("Response Matrix\n(SLM Zernike → WFS Response)")
        plt.colorbar(im, ax=ax, label="Response [a.u.]")
        plt.tight_layout()
        plt.show()
    
    # Column analysis
    if matrix is not None and variance_matrix is not None:
        print("\n=== Column Analysis ===")
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        n_modes = matrix.shape[1]
        modes = np.arange(n_modes)
        
        # L2 norms
        col_norms = np.linalg.norm(matrix, axis=0)
        ax = axes[0, 0]
        ax.bar(modes, col_norms, color="steelblue", alpha=0.8)
        ax.set_xlabel("SLM Mode Index")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Response Amplitude per Mode")
        ax.grid(True, alpha=0.3)
        
        # Average variance
        col_var = np.mean(variance_matrix, axis=0)
        ax = axes[0, 1]
        ax.bar(modes, col_var, color="tomato", alpha=0.8)
        ax.set_xlabel("SLM Mode Index")
        ax.set_ylabel("Mean Variance")
        ax.set_title("Measurement Stability per Mode")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        
        # SNR
        snr = col_norms / (col_var + 1e-12)
        ax = axes[1, 0]
        ax.bar(modes, snr, color="seagreen", alpha=0.8)
        ax.set_xlabel("SLM Mode Index")
        ax.set_ylabel("SNR (L2 norm / variance)")
        ax.set_title("Signal-to-Noise Ratio per Mode")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        
        # Orthogonality
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


def print_summary(matrix, variance_matrix, info=None):
    """Print a summary of the analysis results.
    
    Args:
        matrix: Response matrix
        variance_matrix: Variance matrix
        info: Information dictionary from analysis
    """
    print("\n" + "="*50)
    print("       Zernike Response Matrix Summary")
    print("="*50)
    
    if info:
        if 'debug_directory' in info:
            print(f"  Debug directory: {info['debug_directory']}")
        if 'n_modes' in info:
            print(f"  Number of modes: {info['n_modes']}")
        if 'samples_per_cycle' in info:
            print(f"  Samples per cycle: {info['samples_per_cycle']}")
        if 'wfs_deviation_length' in info:
            print(f"  WFS deviation length: {info['wfs_deviation_length']} (each direction)")
        if 'active_zernike_indices' in info and info['active_zernike_indices']:
            print(f"  Active Zernike modes: {info['active_zernike_indices'][:5]}")
    
    if matrix is not None:
        print(f"  Matrix dimensions: {matrix.shape[0]} × {matrix.shape[1]}")
        print(f"  Response range: [{matrix.min():.4f}, {matrix.max():.4f}]")
        print(f"  Average response: {matrix.mean():.4f} ± {matrix.std():.4f}")
    
    if variance_matrix is not None:
        print(f"  Average variance: {variance_matrix.mean():.6f}")
    
    if matrix is not None and matrix.size > 0:
        try:
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            if s[-1] > 0:
                cond = s[0] / s[-1]
                print(f"  Condition number: {cond:.2e}")
            
            explained = np.cumsum(s**2) / np.sum(s**2)
            n95 = np.searchsorted(explained, 0.95) + 1
            n99 = np.searchsorted(explained, 0.99) + 1
            effective_rank = np.sum(s / s[0] > 1e-6)
            print(f"  Energy 95%: {n95} modes")
            print(f"  Energy 99%: {n99} modes")
            print(f"  SVD effective rank: {effective_rank}/{len(s)}")
        except Exception as e:
            print(f"  Could not calculate SVD metrics: {e}")
    
    print("="*50)


if __name__ == "__main__":
    base_path = Path("../data/zernike_response_matrix")
    if not base_path.exists():
        base_path = Path("data/zernike_response_matrix")
    
    matrix, variance_matrix, info = analyze_debug_structure(base_path)
    visualize_debug_data(matrix, variance_matrix, info)
    print_summary(matrix, variance_matrix, info)