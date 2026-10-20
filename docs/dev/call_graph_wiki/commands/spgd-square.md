---
title: spgd-square（SLM 方形远场均匀度整形（SPGD））
created: 2026-10-13
updated: 2026-10-13
type: command
tags: [command]
source: src/ao_shaping/tools/slm/panel/diagnose.py:314
---

## 一句话

SLM 方形远场均匀度整形（SPGD）

## 入口函数

`tools.slm.panel.diagnose.main`（`src/ao_shaping/tools/slm/panel/diagnose.py:314`）

```python
main(params)
```

## docstring（原文节选）

```
SLM 硬件自检: 逐级定位是否存在"面板不调制光"类故障。
```

## 调用链分析（自动）

**硬件**：（无直接硬件，纯计算/仿真）

## 直接调用（1 级，按调用次数）

- [[functions/with-params|`with_params`]]
- `step_freezing`
- `step_linearity`
- `step_modulation`

## 谁调用它

- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `spgd-square`）

## 相关页面

- [[entry/cli]]
