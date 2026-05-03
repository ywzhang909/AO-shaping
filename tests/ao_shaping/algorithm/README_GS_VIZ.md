# GS Algorithm Visualization Tests

This directory contains tests for the Gerchberg-Saxton algorithm that generate animated GIFs showing the iteration process.

## Generated GIF Files

After running the tests, you will find animated GIFs in the `logs/gs_viz/` directory:

### Main Test GIFs
- **gaussian**: GS convergence for Gaussian target shape
- **circle**: GS convergence for circular target shape  
- **square**: GS convergence for square target shape
- **annular**: GS convergence for annular (ring) target shape
- **convergence**: Extended 50-iteration convergence demonstration

### Comparison GIFs
Located in `logs/gs_viz/comparison/`:
- Side-by-side comparison of different target shapes
- Shows how different target patterns affect convergence

### High Resolution GIFs
Located in `logs/gs_viz/high_res/`:
- 256x256 resolution animations
- Better visual quality for presentations

## Running the Tests

### Run all visualization tests:
```bash
python -m pytest tests/ao_shaping/algorithm/test_gs_viz.py -v
```

### Run specific shape test:
```bash
python -m pytest tests/ao_shaping/algorithm/test_gs_viz.py -k "gaussian" -v
```

### Run convergence demonstration:
```bash
python -m pytest tests/ao_shaping/algorithm/test_gs_viz.py::TestGSVizualization::test_gs_convergence_animation -v
```

## GIF Animation Content

Each GIF shows three panels:

1. **Left Panel - Phase Pattern**: 
   - Shows the SLM phase pattern (0 to 2π)
   - Color-coded using HSV colormap
   - Evolves as the algorithm converges

2. **Middle Panel - Target Amplitude**:
   - The desired target intensity pattern
   - Remains constant throughout
   - Shows what we're trying to achieve

3. **Right Panel - Reconstructed Amplitude**:
   - The actual reconstructed intensity
   - Starts from diffused pattern
   - Converges toward target
   - Shows current MSE (Mean Squared Error)

## Animation Parameters

- **Frame Rate**: 5-10 FPS (configurable)
- **Frame Skip**: Every 2nd frame saved (configurable)
- **Resolution**: 128x128 (default) or 256x256 (high-res)
- **Format**: Animated GIF with infinite loop

## Visualization Utilities

The `gs_visualization.py` module provides:

### Main Functions

#### `gerchberg_saxton_with_visualization()`
Run GS algorithm and automatically save animation:

```python
from ao_shaping.utils.gs_visualization import gerchberg_saxton_with_visualization

result = gerchberg_saxton_with_visualization(
    source_amplitude=source,
    target_amplitude=target,
    iterations=50,
    output_dir=Path("logs/my_experiment"),
    save_animation=True,
    animation_fps=10,
    skip_frames=2,
)

print(f"Animation saved to: {result.animation_path}")
```

#### `GSVizCallback` Class
Collect states manually for custom animations:

```python
from ao_shaping.utils.gs_visualization import GSVizCallback

callback = GSVizCallback(source_amplitude)

# During GS iterations, manually add states
callback.add_state(phase, amplitude, error)

# Save animation when done
callback.save_animation(
    target_amplitude=target,
    output_path=Path("logs/custom.gif")
)
```

## Example Output

Typical GIF file sizes:
- 128x128, 30 iterations: ~3-4 MB
- 256x256, 40 iterations: ~5-8 MB
- High FPS (10+): Larger files

## Dependencies

- matplotlib (for rendering)
- Pillow (PIL) for GIF creation
- numpy for array operations

## Notes

- GIFs are saved with timestamps to avoid overwriting
- Tests automatically create the `logs/gs_viz/` directory
- All tests use pure simulation (no hardware required)
- Animations are optimized for file size
