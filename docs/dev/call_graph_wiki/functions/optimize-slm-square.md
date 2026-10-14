---
title: optimize_slm_square（SLM 方形远场 SPGD 优化）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [optimizer,shaping,square,spgd]
source: src/ao_shaping/optimizer/wfless/slm_square_shaping.py:1129
---

## 签名

```python
optimize_slm_square(center, epochs, config, n_max, target_side, target_mean_brightness, side_factor, delta, lr, exposure_time_ms, cam_id, cam_type, … (+28 个)) → `Recorder`
```

## docstring（原文）

```
Optimize square beam uniformity using SLM with Zernike coefficient control.

    This function uses the SPGD (Stochastic Parallel Gradient Descent) algorithm
    to optimize Zernike coefficients displayed on an SLM, minimizing the
    coefficient of variation (CV) of the intensity within a target square region.

    Args:
        center: Center position for the square target. Can be None (auto-detect
            original-spot centroid, same as "shape"), "shape" (argmax-anchored
            smart detection — argmax anchor + centroid refinement on flat cores),
            "centroid_thresh" (brightness centroid, threshold=0.1×max),
            "max" (brightest pixel / argmax), "mass" (full-image centroid,
            stray-light sensitive), or tuple (x, y) for a fixed center
            (not re-detected after exposure adjustment).
        epochs: Number of optimization iterations.
        n_max: Maximum Zernike radial order. Controls the number of modes.
        target_side: Side length of the target square in pixels.
            Mutually exclusive with target_mean_brightness (specify only one).
            If 0 and target_mean_brightness is also 0, auto-computed from spot
            size × side_factor.
        target_mean_brightness: Target mean brightness (average pixel value /
            gray level) of the square. Mutually exclusive with target_side.
            When positive, the side length is auto-derived by energy
            conservation: side = sqrt(total_brightness / target_mean_brightness),
            where total_brightness is the integrated intensity of the initial
            camera image.
        side_factor: Multiplier for auto-computed side length (default 1.5).
        delta: Perturbation amplitude for Zernike coefficients.
        lr: Learning rate. If 0, auto-adjusted via learning_schedule.
        exposure_time_ms: Camera exposure time in ms. If 0, auto-exposure.
        cam_id: Camera device ID.
        show: Whether to display images during optimization.
        init_c: Initial Zernike coefficients. If None, starts from zeros.
        cam_size: Camera window size.
        target_max_brightness: Target max brightness for auto-exposure.
        slm_number: SLM device number (1-8).
        slm_wavelength: SLM wavelength in nm (0 = ask the device).
        optimizer_type: Optimizer type (adam/adamod/sgd/muno).
        random_seed: Random seed for reproducibility.
        w_uniformity: Weight for uniformity (CV) in quality score.
        w_efficiency: Weight for encircled energy in quality score.
        w_aspect: Weight for aspect ratio in quality score.
        basis: Parameterisation basis ("zernike" or "freeform").
        phase_grid: Freeform grid side (dim = grid^2).
        zernike_radius: Zernike aperture radius in pixels. Defaults to
            min(SLM_HEIGHT, SLM_WIDTH) / 2 = 600, matching the GUI
            (multi_slm_controller.py Zernike branch).
        zernike_mask: 0/1 binary array selecting which Zernike Noll modes
            participate in SPGD. Length must be ≥ 5; Noll indices 1-3 (piston,
            tip, tilt) are forced to 0. When provided, overrides the hardcoded
            ZERNIKE_ACTIVE_MODES ((2,0),(4,0)). E.g. [0,0,0,1,0,0,0,0,0,0,0,1]
            activates only defocus (2,0) and spherical (4,0).
        rotation_search_deg: SLM↔相机相对旋转搜索范围(度, 0~360)。0=不启用
            旋转校正(默认); >0 时旋转角作为额外 SPGD 自由度, 在
            [-range/2, +range/2] 内搜索, 每轮下发相位图按当前角旋转补偿。
            起始值 0°, 扰动步长固定 1°。
        objective: Square-shaping objective, one of
            :data:`SQUARE_OBJECTIVE_CHOICES`. ``"quality"`` (default) is the
            historical combined score; ``"pearson"`` is the FourierGSNet
            ``1 - Pearson`` correlation loss, negated to higher-is-better by
            :func:`square_objective_score`.
        adaptive: Enable the adaptive SPGD closed loop (default False). When on,
            the quality weights are scheduled from
            :data:`AdaptiveSpgdConfig.stage_weights` — a
            (w_uniformity, w_efficiency, w_aspect) triple — switching at epoch
            thirds, and ``w_uniformity`` is temporarily boosted while the
            best-CV improvement stays below ``min_improve_frac`` over the last
            ``stagnate_win`` epochs. When off, behavior is identical to previous
            releases.
        stagnate_win: Stagnation-detection window in epochs (adaptive only).
        stagnate_boost: Multiplicative boost applied to ``w_uniformity`` while
            stagnating; 1.0 disables the boost (adaptive only).
        min_improve_frac: Stagnation trigger — best-CV improvement below this
            fraction over the window counts as stagnation (adaptive only).
        **kwargs: Additional optimizer parameters.

    Returns:
        Recorder: Optimization history recorder.
```

## 直接调用（出边）

- `ao_shaping.drivers.ccd.common.get_camera_exposure_ms`
- `square_objective_score`
- `square_aspect_ratio`
- `square_encircled_energy`
- `square_peak_to_background_ratio`
- `square_uniformity_cost`
- `_locate_square_center`
- `ao_shaping.drivers.ccd.common.capture_with_exposure`
- `learning_schedule`
- `_freeform_phase_radians`
- `_map_init_to_active_modes`
- `_slm_cls`
- [[functions/run-heuristic-search|`run_heuristic_search`]]
- [[functions/create-camera|`create_camera`]]
- `ao_shaping.drivers.ccd.common.resample_on_saturation`
- `ao_shaping.optimizer.constants.create_optimizer`
- `ao_shaping.optimizer.wfless.adaptive_spgd_core.AdaptiveSpgdConfig` *(未解析/外部)*
- `ao_shaping.optimizer.wfless.adaptive_spgd_core.get_stage_weights`
- `ao_shaping.optimizer.wfless.adaptive_spgd_core.should_boost_stagnation`
- `ao_shaping.utils.Recorder` *(未解析/外部)*

## 谁调用它（入边）

- [[commands/spgd-square|`square`]]

## 相关页面

- [[entry/cli]]
