---
title: gen_date_dir / make_date_dir / 时间戳工具（目录与命名）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [io,utils,time]
source: src/ao_shaping/utils/io/file.py:159
---

## 签名

```python
gen_date_dir(base_dir)
```

## docstring（原文）

```
Timestamped run directory, second resolution.

    See :func:`~ao_shaping.utils.io.timestamp.make_date_dir` for why
    :func:`~ao_shaping.utils.io.cli_helpers.create_save_dir` keeps a coarser
    stamp rather than being folded into this one.
```

## 直接调用（出边）

- [[functions/path-dirs|`make_date_dir`]]

## 谁调用它（入边）

- [[commands/pib|`run`]]
- [[commands/combined|`run`]]

## 相关页面

- [[entry/cli]]
