---
title: zernike_modes / calc_n_zernike_terms（Zernike 模式与阶数）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [zernike,wavefront,utils]
source: src/ao_shaping/utils/wavefront/zernike_calc.py:90
---

## 签名

```python
zernike_modes(n_max) → `list[tuple[int, int]]`
```

## docstring（原文）

```
Valid (n, m) index pairs up to radial order ``n_max``.

    Parity rule: ``n - |m|`` must be even (m steps by 2 for each n).
    This is the inverse of :func:`calc_n_zernike_terms` — given the radial
    order it enumerates every (n, m) pair in the standard Noll ordering
    (n ascending, m ascending within each n).

    Pure Zernike math — no GUI / no session_state dependency, so it belongs
    in :mod:`ao_shaping.utils.zernike_calc` alongside
    :func:`get_zernike_name` and :func:`calc_n_zernike_terms`.

    Args:
        n_max: Maximum Zernike radial order. Coerced to ``int`` so a widget
            string value (e.g. ``"5"``) is accepted transparently.

    Returns:
        List of ``(n, m)`` tuples. Length equals
        :func:`calc_n_zernike_terms` ``= (n_max+1)(n_max+2)//2``.
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- [[functions/optimizer-ga|`optimizer_ga`]]
- [[functions/optimizer-greedy|`optimizer_greedy`]]
- [[functions/optimizer-rms-slm|`optimizer_rms_slm`]]

## 相关页面

- [[entry/cli]]
