---
title: optimizer_rms_dm（DM 电压 SPGD 波前 RMS 优化）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [optimizer,wavefront,spgd]
source: src/ao_shaping/optimizer/wf/rms.py:45
---

## 签名

```python
optimizer_rms_dm(epochs, wfs_res, init_v, pupil_center, pupil_diameter, early_stop_threshold, dm, wfs_type, disturbance_cn2, lr, delta) → `Recorder`
```

## docstring（原文）

```
Wavefront RMS optimizer using SPGD with a deformable mirror.

    Args:
        epochs: Number of optimization iterations.
        wfs_res: WFS resolution (e.g. "512", "768", "1024").
        init_v: Initial voltage array. If empty, starts from zeros.
        pupil_center: WFS pupil center coordinates.
        pupil_diameter: WFS pupil diameter.
        early_stop_threshold: RMS threshold for early stopping.
        dm: Deformable mirror instance (must be open). Required.
        wfs_type: Registered sensor type, resolved via the WFS registry. Defaults
            to ``"thorlab"``, preserving the previous hard-coded construction.
        disturbance_cn2: Turbulence strength for the simulated sensor. ``0``
            injects no aberration, so an RMS-minimising loop correctly
            keeps the flat DM command and nothing is demonstrated.
        lr: Override the schedule's learning rate. ``None`` keeps the
            hardware-calibrated auto-schedule.
        delta: Override the SPGD perturbation amplitude, in volts. The
            auto-schedule targets a specific hardware voltage-to-phase
            scale; against a simulated DM it perturbs by ~2.6% of the
            objective, too little for the gradient estimate to register.

    Returns:
        Recorder with optimization history.
```

## 直接调用（出边）

- `schedule_lr_delta`
- `ao_shaping.algorithm.gradient.adam.AdaMOD` *(未解析/外部)*
- [[functions/resolve-wfs|`resolve_wfs`]]
- [[functions/spgd-gradient|`spgd_gradient`]]
- `ao_shaping.utils.Recorder` *(未解析/外部)*

## 谁调用它（入边）

- [[commands/pipeline|`run`]]
- [[commands/wf|`run`]]

## 相关页面

- [[entry/cli]]
