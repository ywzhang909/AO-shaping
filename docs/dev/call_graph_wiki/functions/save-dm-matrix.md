---
title: save_dm_response_matrix（DM 矩阵落盘）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [calibration,io]
source: src/ao_shaping/optimizer/wf/dm_response_matrix.py:814
---

## 签名

```python
save_dm_response_matrix(result, path, include_inverses) → `None`
```

## docstring（原文）

```
Save DM response matrix to HDF5 file with full metadata.

    HDF5 structure:
    - Datasets: matrix, variance_matrix, subaperture_mask (optional),
      pinv_matrix (optional), lstsq_matrix (optional)
    - Metadata group: stores scalar attrs and JSON-serialized fields

    Args:
        result: Calibration result to save.
        path: Save path. If suffix is not '.h5', appends '.h5'.
        include_inverses: If True, include pinv_matrix and lstsq_matrix.
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- [[commands/dm-matrix|`run`]]

## 相关页面

- [[entry/cli]]
