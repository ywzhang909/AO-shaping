---
title: compute_pinv / compute_lstsq / safe_pinv / effective_cond（矩阵伪逆与条件数）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [matrix,utils,calibration]
source: src/ao_shaping/utils/wavefront/matrix_utils.py:12
---

## 签名

```python
compute_pinv(matrix, rcond) → `np.ndarray`
```

## docstring（原文）

```
Compute SVD pseudoinverse.

    Args:
        matrix: Input matrix of shape (m, n).
        rcond: Singular value truncation threshold.

    Returns:
        Pseudoinverse matrix of shape (n, m).
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- [[functions/calibrate-dm-matrix|`calibrate_dm_response_matrix`]]

## 相关页面

- [[entry/cli]]
