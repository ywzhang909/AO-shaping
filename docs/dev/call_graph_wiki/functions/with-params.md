---
title: with_params（dataclass → Click options 装饰器）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [cli,utils,params]
source: src/ao_shaping/utils/cli/params.py:244
---

## 签名

```python
with_params(arg_class, kw_name) → `Any`
```

## docstring（原文）

```
Attach the click options of ``arg_class`` to a click command.

    Every ``Annotated[..., option(...)]`` field becomes a click option (help
    text order == dataclass declaration order). Fields marked with
    :class:`ClickGroup` additionally contribute their nested options, which are
    re-assembled into the nested instance before delivery. The wrapped command
    receives ``kw_name=arg_class(**provided_fields)`` — one keyword argument per
    decorator, containing a fully-populated parameter instance.
```

## 直接调用（出边）

- `_collect_click_annotations`
- `_copy_delayed_call`
- `_patch_click_types`
- `_patch_defaults`
- `_patch_names`

## 谁调用它（入边）

- [[commands/pib|`run`]]
- [[commands/wf|`run`]]
- [[commands/zernike-matrix|`run`]]
- [[commands/hadamard-matrix|`hadamard_matrix_run`]]
- [[commands/dm-matrix|`run`]]
- [[commands/combined|`run`]]
- [[commands/pipeline|`run`]]
- [[commands/rms-zernike|`run`]]
- [[commands/greedy-zernike|`greedy_zernike_run`]]
- [[commands/ga-zernike|`run`]]
- [[commands/full-voltage|`full_voltage_run`]]
- [[commands/alt-voltage|`run`]]
- [[commands/slm-gs-refine|`run`]]
- [[commands/slm-gsnet|`run`]]
- [[commands/slm-model-in-loop|`run`]]
- [[commands/slm-pib|`run`]]
- [[commands/closed-loop|`closed_loop_run`]]
- [[commands/slm-lut|`run`]]
- [[commands/slm-diagnose|`main`]]

## 相关页面

- [[entry/cli]]
