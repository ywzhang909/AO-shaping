---
title: calibrate_dm_response_matrix（DM 响应矩阵标定内核）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [calibration,dm,matrix]
source: src/ao_shaping/optimizer/wf/dm_response_matrix.py:491
---

## 签名

```python
calibrate_dm_response_matrix(dm, wfs, disturb_voltage, n_cycles, n_averages, wait_time, compute_inverses, verbose, dm_unit_mask, cancel_tile, subaperture_mask, mask_n_avg, … (+7 个)) → `DMResponseMatrixResult`
```

## docstring（原文）

```
Calibrate DM actuator response matrix.

    Builds the response matrix by measuring each valid actuator's contribution
    to WFS spot deviations using a push-pull voltage perturbation method.

    The response matrix shape is (n_slopes, n_actuators_valid) where n_slopes
    is the number of filtered slope measurements (2 * valid subapertures if
    subaperture_mask is provided, else 2 * num_spots_x * num_spots_y).

    Args:
        dm: NLight DM instance (must be open).
        wfs: WFSManager instance (must be initialized).
        disturb_voltage: Voltage perturbation. If None/0 and auto_optimize=True,
            auto-optimizes per actuator.
        n_cycles: Push-pull cycles per actuator (N).
        n_averages: WFS readings per polarity per cycle (M).
        wait_time: Seconds after voltage apply.
        compute_inverses: If True, compute SVD pseudoinverse and least-squares inverse.
        verbose: If True, show tqdm progress bar.
        dm_unit_mask: Boolean mask of valid actuators (True = measure).
            If None, uses dm.default_dm_unit_mask. Actuator 0 is typically disabled.
        cancel_tile: If True, remove tip/tilt from WFS deviations.
        subaperture_mask: Pre-computed 2D bool mask of valid subapertures.
            If None, builds automatically via wfs.build_subaperture_mask().
        mask_n_avg: Frames to average for auto-built subaperture mask.
        mask_threshold_ratio: Intensity threshold ratio for auto-built mask.
        mask_edge_clip: Edge clip for auto-built mask.
        auto_optimize_voltage: If True and disturb_voltage is None/0, auto-optimize
            per-actuator voltage for best linearity.
        optimize_n_avg: WFS readings per voltage during optimization.
        debug_data_callback: Optional callback for raw measurement data.
            Signature depends on measure_actuator_response.
        mode: Calibration strategy: "sequential" (one actuator at a time) or
            "hadamard" (simultaneous Hadamard-pattern push-pull sweeps).
        hadamard_order: Hadamard matrix order for mode="hadamard". If None,
            uses the smallest power of two >= n_valid.

    Returns:
        DMResponseMatrixResult with response matrix, variance matrix, inverses, and metadata.
```

## 直接调用（出边）

- `_build_device_config`
- `_decode_hadamard_responses`
- `_hadamard_matrix`
- `_next_hadamard_order`
- `_optimize_perturbation_voltage`
- `ao_shaping.utils.wavefront.matrix_utils.compute_lstsq`
- [[functions/matrix-utils|`compute_pinv`]]
- `ao_shaping.utils.wavefront.wfs_utils.flatten_slopes`
- `measure_actuator_response`

## 谁调用它（入边）

- [[commands/dm-matrix|`run`]]

## 相关页面

- [[entry/cli]]
