# Hysteretic Deformable Mirror Simulation in Python

This is a Python implementation of the hysteretic deformable mirror simulation originally written in MATLAB. The simulation models the behavior of a deformable mirror with hysteresis effects using Preisach operators.

## Overview

The simulation consists of several modules that replicate the functionality of the original MATLAB code:

1. `create_electrodes.py` - Creates the electrode configuration for the deformable mirror
2. `compute_influence_matrix.py` - Computes the influence matrix that describes how each electrode affects the mirror surface
3. `mat_utils.py` - Utility functions for matrix operations
4. `zernike.py` - Implementation of Zernike polynomials
5. `create_hdm_matrices.py` - Creates the HDM (Hysteric Deformable Mirror) matrices
6. `wavefront_reconstruction.py` - Performs wavefront reconstruction using Zernike polynomials
7. `create_preisachs.py` - Creates Preisach operators to model hysteresis
8. `simulate_hdm_control.py` - Simulates the control of the hysteretic deformable mirror
9. `main.py` - Main module that integrates all components

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the complete simulation:
```
python main.py
```

Or run individual steps:
```
python main.py --step electrodes
python main.py --step wavefront
python main.py --step preisach
```

## Modules Description

### create_electrodes.py
Creates a grid of electrodes on a circular mirror surface. The electrodes are arranged in a square grid pattern.

### compute_influence_matrix.py
Computes the influence matrix that describes how each electrode deforms the mirror surface. This involves numerical integration to calculate the influence functions.

### mat_utils.py
Provides utility functions for converting between matrix and vector representations with masking.

### zernike.py
Implements Zernike polynomials, which are used to describe wavefront aberrations.

### create_hdm_matrices.py
Creates the matrices needed for HDM simulation, including scaling of influence functions and computation of HDM matrices.

### wavefront_reconstruction.py
Performs wavefront reconstruction using Zernike polynomials. This includes generating random wavefronts and fitting them with Zernike polynomials.

### create_preisachs.py
Creates Preisach operators to model the hysteresis behavior of the actuators. The Preisach model is a classical approach to modeling hysteresis.

### simulate_hdm_control.py
Simulates the control of the hysteretic deformable mirror, including initialization, resetting, and iterative control loops.

### main.py
Main module that orchestrates the entire simulation workflow.

## Dependencies

- NumPy: For numerical computations
- SciPy: For scientific computing utilities
- Matplotlib: For plotting (used in electrode visualization)

## Notes

- The Python implementation closely follows the structure and algorithms of the original MATLAB code
- Some computationally intensive parts (like influence matrix computation) may take significant time to run
- The simulation uses a 5x5 grid of electrodes by default
- Zernike polynomials are used to represent wavefront aberrations

## References

This simulation is based on the work described in:
- [Hysteretic deformable mirror project](https://research.rug.nl/en/publications/high-pixel-number-deformable-mirror-concept-utilizing-piezoelectr-3)
- [PhD dissertation on multi-loop hysteresis and recursive remnant control](https://research.rug.nl/en/publications/multi-loop-hysteresis-and-recursive-remnant-control)