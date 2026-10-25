---
title: optimizer_ga（遗传算法 Zernike 优化）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [optimizer,zernike,ga]
source: src/ao_shaping/optimizer/wf/ga_zernike.py:138
---

## 签名

```python
optimizer_ga(n_generations, population_size, crossover_prob, mutation_prob, tournament_size, elite_count, init_z, pupil_center, pupil_diameter, early_stop_threshold, wavelength, shift_x, … (+7 个)) → `Recorder`
```

## docstring（原文）

```
Optimize wavefront RMS using Genetic Algorithm with SLM Zernike control.

    This function uses a Genetic Algorithm to optimize Zernike coefficients
    displayed on an SLM, minimizing the wavefront RMS measured by a WFS.

    Args:
        n_generations: Number of GA generations to run.
        population_size: Number of individuals in the population.
        crossover_prob: Probability of crossover between parents.
        mutation_prob: Probability of mutating each gene.
        tournament_size: Number of individuals in tournament selection.
        elite_count: Number of top individuals to preserve each generation.
        init_z: Initial Zernike coefficients. Can be:
            - dict: {(n, m): value} form
            - Sequence: Noll-ordered coefficients
            - None: starts from zeros
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

    Returns:
        Recorder: Optimization history with RMS and coefficients.
```

## 直接调用（出边）

- `_gaussian_mutation`
- `_tournament_selection`
- `_blend_crossover`
- `ao_shaping.drivers.slm.ZernikeSLM` *(未解析/外部)*
- [[functions/resolve-wfs|`resolve_wfs`]]
- `ao_shaping.utils.Recorder` *(未解析/外部)*
- [[functions/matrix-utils|`calc_n_zernike_terms`]]
- [[functions/zernike-calc|`zernike_modes`]]

## 谁调用它（入边）

- [[commands/ga-zernike|`run`]]

## 相关页面

- [[entry/cli]]
