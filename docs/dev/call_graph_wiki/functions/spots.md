---
title: centroid / radius / pib_ratio_mask（光斑量测）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [image,spots,utils]
source: src/ao_shaping/utils/image/spots_calc.py:234
---

## 签名

```python
centroid(intensity, moment, threshold, return_float) → `tuple[float | int, float | int]`
```

## docstring（原文）

```
光强的质心位置

    :param intensity: 强度分布
    :param moment: 阶
    :return center_x, center_y: 光强的质心
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- `ao_shaping.utils.image.hardware_utils.resolve_spot_center`

## 相关页面

- [[entry/cli]]
