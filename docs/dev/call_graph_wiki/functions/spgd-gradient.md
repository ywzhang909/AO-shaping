---
title: spgd_gradient（SPGD 梯度估计，float64 防溢出）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [algorithm,spgd,gradient]
source: src/ao_shaping/optimizer/spgd.py:40
---

## 签名

```python
spgd_gradient(pos_obj, neg_obj, disturb, maximize) → `np.ndarray`
```

## docstring（原文）

```
Gradient to pass to ``optimizer.update()`` when using ``param - update``.

    Args:
        pos_obj: Objective measured at ``param + disturb``.
        neg_obj: Objective measured at ``param - disturb``.
        disturb: The signed perturbation vector used for that evaluation pair.
        maximize: ``True`` if the objective must be maximised, ``False`` to
            minimise. Keyword-only on purpose: every call site must state its
            objective direction explicitly.

    Returns:
        The gradient in the descent frame expected by ``Base.update()``.

    The subtraction is performed in float64. Bucket/energy objectives are
    unsigned sums, so ``pos - neg`` wraps on underflow and ``* -1`` raises
    ``OverflowError`` for uint64 inputs if the cast is omitted.
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- `ao_shaping.optimizer.spid_pib_loop.spid_pib_epoch`
- [[functions/optimizer-rms-dm|`optimizer_rms_dm`]]
- [[functions/optimizer-rms-slm|`optimizer_rms_slm`]]

## 相关页面

- [[entry/cli]]
