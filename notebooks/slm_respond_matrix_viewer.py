"""
Main Script for Zernike Response Matrix Analysis
This script chooses the appropriate analyzer based on available data format.
"""

import sys
from pathlib import Path

def main():
    print("=== Zernike Response Matrix Analyzer ===")
    print("This script analyzes Zernike response matrix data from either:")
    print("1. Standard format (.json, .npy, .h5 files)")
    print("2. Debug directory structure (mode_NNN/cycle_M/plus/minus/)")
    print()
    
    # Try to import the analyzers
    try:
        from analyze_standard_format import analyze_standard_format, visualize_standard_data, compare_formats
        from analyze_debug_structure import analyze_debug_structure, visualize_debug_data, print_summary
        print("✓ Analyzer modules loaded successfully")
    except ImportError as e:
        print(f"✗ Error importing analyzer modules: {e}")
        print("Make sure both analyzer files are in the same directory")
        return
    
    # Set up paths
    base_path = Path("../data/zernike_response_matrix")
    if not base_path.exists():
        base_path = Path("data/zernike_response_matrix")
    
    print(f"Checking directory: {base_path.absolute()}")
    print(f"Directory exists: {base_path.exists()}")
    print()
    
    # First, try to analyze standard format
    print("--- Attempting Standard Format Analysis ---")
    matrix, variance_matrix, metadata = analyze_standard_format(base_path)
    
    standard_format_success = matrix is not None
    
    if standard_format_success:
        print("\n✓ Standard format analysis completed")
        visualize_standard_data(matrix, variance_matrix, metadata)
        
        # Try format comparison if H5 file exists
        h5_path = base_path / "zernike_response_matrix_mag0.h5"
        compare_formats(matrix, variance_matrix, h5_path)
        
        print_summary(matrix, variance_matrix, {'metadata': metadata} if metadata else None)
    else:
        print("✗ No standard format data found")
        
        # Try debug directory structure
        print("\n--- Attempting Debug Directory Analysis ---")
        matrix, variance_matrix, info = analyze_debug_structure(base_path)
        
        if matrix is not None:
            print("✓ Debug directory analysis completed")
            visualize_debug_data(matrix, variance_matrix, info)
            print_summary(matrix, variance_matrix, info)
        else:
            print("✗ No data found in either format")
            print("\nPlease check that:")
            print("1. Standard files exist (.json, .npy, .h5) in data/zernike_response_matrix/")
            print("2. Or debug directories exist (debug_*) in data/zernike_response_matrix/")
            return
    
    print("\n=== Analysis Complete ===")

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
        if 'metadata' in info:
            print(f"  Data source: Standard format with metadata")
            meta = info['metadata']
            if isinstance(meta, dict):
                for k, v in list(meta.items())[:3]:  # Show first 3 metadata items
                    print(f"    {k}: {v}")
            else:
                print(f"    Metadata: {meta}")
        else:
            print(f"  Data source: Debug directory structure")
            if 'debug_directory' in info:
                print(f"  Debug directory: {info['debug_directory']}")
            if 'n_modes' in info:
                print(f"  Number of modes: {info['n_modes']}")
            if 'samples_per_cycle' in info:
                print(f"  Samples per cycle: {info['samples_per_cycle']}")
            if 'wfs_deviation_length' in info:
                print(f"  WFS deviation length: {info['wfs_deviation_length']} (each direction)")
    
    if matrix is not None:
        print(f"  Matrix dimensions: {matrix.shape[0]} × {matrix.shape[1]}")
        print(f"  Response range: [{matrix.min():.4f}, {matrix.max():.4f}]")
        print(f"  Average response: {matrix.mean():.4f} ± {matrix.std():.4f}")
    
    if variance_matrix is not None:
        print(f"  Average variance: {variance_matrix.mean():.6f}")
    
    # Try to calculate SVD metrics if possible
    if matrix is not None and matrix.size > 0:
        try:
            import numpy as np
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
    main()