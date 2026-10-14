---
title: runner_common（命令共享工具：center 解析 / 配置构建 / 结果保存）
created: 2026-10-13
updated: 2026-10-13
type: class
tags: [runner,utils]
source: src/ao_shaping/runners/runner_common.py:1366
---

## 签名

```python
parse_center(raw) → `tuple[int, int] | str | None`
```

## docstring（原文）

```
Normalize the ``--center`` option to the optimizer's accepted form.
```

## 直接调用（出边）

- （叶子函数，无仓库内调用）

## 谁调用它（入边）

- [[commands/spgd-square|`square`]]

## 相关页面

- [[entry/cli]]
