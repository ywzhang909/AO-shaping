---
title: resolve_dm（DM 类型解析与实例创建）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [driver,dm,registry]
source: src/ao_shaping/drivers/dm/_registry.py:230
---

## 签名

```python
resolve_dm(dm_type, **kwargs) → `DM`
```

## docstring（原文）

```
解析 DM 类型并创建 DM 实例。

    对应 ``wf`` / ``pipeline`` / ``pib`` / ``combined`` / ``dm-matrix``
    这些 runner 共用的 DM 选择块: 显式的 ``dm_type`` 会被转小写后直接使用;
    否则探测可达的 DM 类型并选取唯一可达的那一个, 对零个/多个候选分别报错。

    Args:
        dm_type: 显式的 DM 类型名, 传 ``None`` 则自动检测。
        **kwargs: 转发给 ``create_dm`` 的额外构造函数 kwargs
            (例如 ``keep_when_exit``、``max_neibor_diff``、
            ``dm_neibor_diff``)。

    Returns:
        创建好的 DM 实例。

    Raises:
        RuntimeError: 没有可达的 DM 时, 或 ``dm_type`` 为 ``None``
            却有多个 DM 可达时。
```

## 直接调用（出边）

- `_ensure_sim_dms_bound`
- `create_dm`
- `list_reachable_dm_types`

## 谁调用它（入边）

- [[commands/dm-matrix|`run`]]
- [[commands/pib|`run`]]
- [[commands/combined|`run`]]
- [[commands/pipeline|`run`]]
- [[commands/wf|`run`]]

## 相关页面

- [[entry/cli]]
