---
title: save_zernike_response_matrix（Zernike 矩阵落盘）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [calibration,io]
source: src/ao_shaping/optimizer/wf/zernike_response_matrix.py:624
---

## 签名

```python
save_zernike_response_matrix(result, path, include_inverses) → `None`
```

## docstring（原文）

```
Save response matrix to HDF5 file with full metadata.

    Args:
        result: Calibration result
        path: Save path (with .h5 extension or as base path)
        include_inverses: Whether to include inverse matrices
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- [[commands/zernike-matrix|`run`]]

## 相关页面

- [[entry/cli]]
