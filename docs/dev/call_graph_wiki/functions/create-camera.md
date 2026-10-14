---
title: create_camera（CCD 相机实例工厂）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [driver,ccd,factory]
source: src/ao_shaping/drivers/ccd/common.py:221
---

## 签名

```python
create_camera(camera_type, cam_id, exposure_time_ms, **kwargs) → `BaseCamera`
```

## docstring（原文）

```
Create (but do **not** open) a camera instance by type and id.

    The returned driver is fully configured but not streaming: call
    ``cam.open()`` (or use it as a context manager) before capturing. This keeps
    construction side-effect free and symmetric with the DM registry's
    ``create_dm``.

    Args:
        camera_type: Registered camera type string (case-insensitive), or a
            parameter object exposing a ``cam_type`` attribute.
        cam_id: Device index (``int``) or file/folder path/URL (``str``) for the
            file-based backends.
        exposure_time_ms: Initial exposure time in milliseconds.
        **kwargs: Backend-specific options. Only keys accepted by the backend are
            forwarded (e.g. ``bit_depth`` for ``miicam``).

    Returns:
        A ``BaseCamera`` subclass instance, not yet opened.

    Raises:
        ValueError: Unknown camera type.
        ImportError: The backend driver/SDK is unavailable.
```

## 直接调用（出边）

- `_resolve_camera_class`
- `list_camera_types`

## 谁调用它（入边）

- [[functions/optimize-slm-square|`optimize_slm_square`]]

## 相关页面

- [[entry/cli]]
