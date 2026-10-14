---
title: get_debug_mode / resolve_debug（DEBUG 判定）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [cli,debug,utils]
source: src/ao_shaping/utils/io/cli_helpers.py:29
---

## 签名

```python
get_debug_mode(—) → `bool`
```

## docstring（原文）

```
从环境变量读取DEBUG模式

    Returns:
        bool: DEBUG环境变量为1/true/yes时返回True，否则返回False
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- [[entry/cli|`cli`]]
- [[commands/dm-matrix|`run`]]
- [[commands/pipeline|`run`]]
- [[commands/wf|`run`]]
- [[commands/slm-model-in-loop|`run`]]
- [[commands/ga-zernike|`run`]]

## 相关页面

- [[entry/cli]]
