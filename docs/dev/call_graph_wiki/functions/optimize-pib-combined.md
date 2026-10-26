---
title: optimize_pib（AdaMOD 综合 PIB + 自适应桶收缩）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [optimizer,pib,adapmod]
source: src/ao_shaping/optimizer/combined_optimizer.py:83
---

## 签名

```python
optimize_pib(center, epochs, r_bucket, delta, lr, exposure_time_ms, cam_type, shrink_iter, shrink_ratio, cam_id, show, init_v, … (+8 个)) → `Recorder`
```

## docstring（原文）

```
Optimize PIB (Power in Bucket) with AdaMOD and adaptive bucket radius.

    This optimizer opens the camera and DM, then runs a disturbance-based
    gradient loop that dynamically shrinks the bucket radius as the spot
    becomes more concentrated.

    Args:
        center: Spot center. Pass a ``(x, y)`` tuple, ``"mass"``, ``"max"``,
            ``"shape"``, or ``None`` for auto-detection.
        epochs: Optimisation iterations.
        r_bucket: Initial bucket radius. ``0`` = auto-detect.
        delta: Disturbance magnitude.
        lr: AdaMOD learning rate. ``0`` = auto-schedule.
        exposure_time_ms: Camera exposure (ms). ``0`` = auto.
        shrink_iter: Shrink bucket every N steps. ``0`` = disable.
        shrink_ratio: Bucket shrink factor per shrink step.
        cam_id: Camera device ID.
        show: Show live Pygame display.
        init_v: Initial DM voltage vector.
        cam_size: Camera region-of-interest size.
        target_max_brightness: Target max pixel brightness. ``0`` = disable auto-exposure.
        dm_unit_mask: Boolean mask for enabled DM actuators.
        dm_neibor_diff: Max neighbour voltage difference safety check.
        dm_max_voltage: DM max voltage limit.
        dm_min_voltage: DM min voltage limit.
        dm: DM instance. If ``None``, a default NlightDM is created.
        **kwargs: Additional kwargs forwarded to the AdaMOD constructor.

    Returns:
        ``Recorder`` with full per-epoch history (J, pib, voltages, etc.).
```

## 直接调用（出边）

- [[functions/spots|`pib_ratio_mask`]]
- [[functions/spots|`radius`]]
- `ao_shaping.drivers.ccd.common.get_camera_exposure_ms`
- [[functions/spots|`power_in_bucket_mask`]]
- `learning_schedule`
- `ao_shaping.algorithm.gradient.adam.AdaMOD` *(未解析/外部)*
- `ao_shaping.drivers.ccd.common.resample_on_saturation`
- `ao_shaping.drivers.dm._registry.get_dm_registry`
- `ao_shaping.optimizer.spid_pib_loop.open_pib_bench`
- `ao_shaping.optimizer.spid_pib_loop.spid_pib_epoch`
- `ao_shaping.utils.ImageVoltagesDisplay` *(未解析/外部)*
- `ao_shaping.utils.Recorder` *(未解析/外部)*

## 谁调用它（入边）

- [[commands/combined|`run`]]

## 相似函数（自动）

与本页函数结构/职责相近的函数（相似度得分，来自 `build_similarity.py`；含义与阈值见 [[relations/similar-functions|相似函数总览]]）：

- 0.384 — [[functions/optimize-pib-wfless|`optimize_pib`]]（同名 `optimize_pib`；共享 9 个被调函数（调用结构高度相似））


## 相关页面

- [[entry/cli]]
