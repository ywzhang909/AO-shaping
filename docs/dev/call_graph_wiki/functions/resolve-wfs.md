---
title: resolve_wfs（WFS 类型解析与实例创建）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [driver,wfs,registry]
source: src/ao_shaping/drivers/wfs/_registry.py:147
---

## 签名

```python
resolve_wfs(wfs_type, **kwargs) → `BaseWFS`
```

## docstring（原文）

```
解析传感器类型并创建实例。

    ``wfs_type=None`` 时给出 Thorlab, 与该接入点存在之前 runner 调用点的行为一致,
    因此把它们迁移到 ``resolve_wfs`` 是行为保持的。
```

## 直接调用（出边）

- `create_wfs`

## 谁调用它（入边）

- [[functions/optimizer-rms-dm|`optimizer_rms_dm`]]
- [[functions/optimizer-ga|`optimizer_ga`]]
- [[functions/optimizer-greedy|`optimizer_greedy`]]
- [[functions/optimizer-rms-slm|`optimizer_rms_slm`]]
- [[commands/zernike-matrix|`run`]]

## 相关页面

- [[entry/cli]]
