---
title: optimizer_greedy（贪婪局部搜索 Zernike 优化）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [optimizer,zernike,greedy]
source: src/ao_shaping/optimizer/wf/greedy_zernike.py:73
---

## 签名

```python
optimizer_greedy(epochs, n_init, n_directions, perturbation_scale, init_z, pupil_center, pupil_diameter, early_stop_threshold, wavelength, shift_x, shift_y, n_max, … (+7 个)) → `Recorder`
```

## docstring（原文）

```
Optimize wavefront RMS using Greedy Local Search with SLM Zernike control.

    This function implements a greedy optimizer that:
    1. Initializes N random starting positions and picks the best one
    2. At each iteration, samples n random perturbation directions
    3. Evaluates all n perturbations + current position (n+1 candidates)
    4. Moves to the best candidate
    5. Repeats until convergence or max epochs

    Args:
        epochs: Maximum number of optimization iterations.
        n_init: Number of random initial positions to sample.
        n_directions: Number of random perturbation directions per iteration.
        perturbation_scale: Scale factor for perturbation magnitudes.
        init_z: Initial Zernike coefficients. Can be:
            - dict: {(n, m): value} form
            - Sequence: Noll-ordered coefficients
            - None: starts from random initialization
        pupil_center: WFS pupil center (x, y).
        pupil_diameter: WFS pupil diameter.
        early_stop_threshold: Stop if RMS drops below this threshold.
        wavelength: SLM wavelength in nm.
        shift_x: SLM X shift in pixels.
        shift_y: SLM Y shift in pixels.
        n_max: Maximum Zernike radial order.
        wfs_res: WFS resolution.
        remove_tilt: Remove tilt in WFS wavefront measurement.
        slm_number: SLM device number (1-8).
        slm_wavelength: Override SLM wavelength (deprecated, use wavelength).
        algorithm: ``"spgd"`` (the greedy local search below, default) or a
            black-box heuristic (``ga``/``pso``/``sa``/``hc``/``rs``/``cem``/``de``).
        pop_size: Population size for population-based heuristics
            (ga/pso/cem/de); ignored by SA/HC/RS and by ``"spgd"``.

    Returns:
        Recorder: Optimization history with RMS and coefficients.
```

## 直接调用（出边）

- `ao_shaping.algorithm.heuristic.search.heuristic_algorithm_choices`
- `_generate_random_direction`
- [[functions/run-heuristic-search|`run_heuristic_search`]]
- `ao_shaping.drivers.slm.ZernikeSLM` *(未解析/外部)*
- [[functions/resolve-wfs|`resolve_wfs`]]
- `ao_shaping.utils.Recorder` *(未解析/外部)*
- [[functions/matrix-utils|`calc_n_zernike_terms`]]
- [[functions/zernike-calc|`zernike_modes`]]

## 谁调用它（入边）

- [[commands/greedy-zernike|`greedy_zernike_run`]]

## 相关页面

- [[entry/cli]]
