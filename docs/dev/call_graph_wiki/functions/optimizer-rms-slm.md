---
title: optimizer_rms_slm（SLM Zernike RMS 优化，AdaMOD/AdamW）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [optimizer,zernike,slm]
source: src/ao_shaping/optimizer/wf/rms_by_zernike.py:439
---

## 签名

```python
optimizer_rms_slm(epochs, init_z, lr, delta, early_stop_threshold, wavelength, shift_x, shift_y, n_max, wfs_res, remove_tilt, wfs_exposure_time, … (+26 个))
```

## docstring（原文）

```
Optimize wavefront RMS using SLM with Zernike coefficient control.

    This function uses AdaMOD or AdamW algorithm to optimize Zernike coefficients
    displayed on an SLM, minimizing the wavefront RMS measured by a WFS.

    Args:
        epochs: Number of optimization iterations.
        init_z: Initial Zernike coefficients. Can be:
            - dict: {(n, m): value} form
            - np.ndarray: Noll-ordered coefficients
            - None: starts from zeros
        lr: Initial learning rate.
        delta: Initial perturbation delta.
        early_stop_threshold: Stop if RMS drops below this threshold.
        wavelength: SLM wavelength in nm.
        shift_x: SLM X shift in pixels.
        shift_y: SLM Y shift in pixels.
        n_max: Maximum Zernike radial order.
        wfs_res: WFS resolution.
        remove_tilt: Remove tilt in WFS wavefront measurement.
        slm_number: SLM device number (1-8).
        slm_wavelength: Override SLM wavelength (deprecated, use wavelength).
        n_init_positions: Number of random initial positions to try (0 = disabled).
        init_range: Range for random initialization.
        lr_schedule: Learning rate schedule type ("static", "cosine", "exp", "linear").
        lr_min: Minimum learning rate for scheduling.
        delta_schedule: Delta schedule type ("static", "cosine", "exp", "linear").
        delta_min: Minimum delta for scheduling.
        optimizer_type: Optimizer type ("adamod" or "adamw").
        beta1: Adam beta1 parameter for momentum.
        weight_decay: AdamW weight decay parameter.
        mini_batch: Number of SPGD perturbations to average per gradient estimate.
        gradient_clip: Maximum absolute gradient value (0 to disable).
        stagnation_patience: Epochs without improvement before delta boost.
        stagnation_delta_boost: Multiplier for delta boost on stagnation.
        freeze_high_order_threshold: RMS threshold to freeze high-order modes.
        early_stop_window: Sliding window size for early stopping validation.
        early_stop_min_epochs: Minimum epochs before early stopping can trigger.
        early_stop_patience: Consecutive non-improvement windows before stopping.
        n_frames: Number of WFS images to average per measurement.
        algorithm: Search algorithm: ``"spgd"`` (gradient loop, default) or a
            black-box heuristic (``ga``/``pso``/``sa``/``hc``/``rs``/``cem``/``de``).
        pop_size: Population size for population-based heuristics
            (ga/pso/cem/de); ignored by SA/HC/RS and by ``"spgd"``.

    Returns:
        Recorder: Optimization history with RMS and coefficients.
```

## 直接调用（出边）

- `ao_shaping.algorithm.heuristic.search.heuristic_algorithm_choices`
- [[functions/matrix-utils|`calc_n_zernike_terms`]]
- `_get_perturb_weights`
- `ao_shaping.algorithm.gradient.adam.AdaMOD` *(未解析/外部)*
- `ao_shaping.algorithm.gradient.adam.AdamW` *(未解析/外部)*
- [[functions/run-heuristic-search|`run_heuristic_search`]]
- `ao_shaping.drivers.slm.ZernikeSLM` *(未解析/外部)*
- [[functions/resolve-wfs|`resolve_wfs`]]
- [[functions/spgd-gradient|`spgd_gradient`]]
- `ao_shaping.utils.Recorder` *(未解析/外部)*
- [[functions/zernike-calc|`zernike_modes`]]
- `compute_mini_batch_gradient`
- `early_stopping_check`
- `get_delta_schedule`
- `get_lr_schedule`

## 谁调用它（入边）

- [[commands/rms-zernike|`run`]]

## 相关页面

- [[entry/cli]]
