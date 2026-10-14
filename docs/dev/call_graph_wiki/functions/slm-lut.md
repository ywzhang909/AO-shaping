---
title: build_inverse_lut / invert_depth_scan / depth_pattern（SLM LUT 数学）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [slm,lut,utils]
source: src/ao_shaping/utils/slm/slm_lut.py:248
---

## 签名

```python
build_inverse_lut(phi, g_values, n) → `tuple[np.ndarray, np.ndarray]`
```

## docstring（原文）

```
Build an inverse LUT (phase -> gray).

    Uses a uniform target grid ``t`` in ``[0, 2*pi]`` (``n`` points) and sets
    ``gray_lut[i] = g_values[argmin(|phi - t[i]|)]`` (first occurrence on ties).

    Args:
        phi: Measured phase array (must be sorted ascending).
        g_values: Gray values parallel to ``phi``.
        n: Number of points in the target grid.

    Returns:
        Tuple ``(t_grid, gray_lut)`` where ``t_grid`` is the float target grid
        of size ``n`` and ``gray_lut`` is a uint16 index array of size ``n``.

    Raises:
        ValueError: If ``phi`` is not sorted ascending.
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- [[commands/slm-lut|`run`]]

## 相关页面

- [[entry/cli]]
