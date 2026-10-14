---
title: main.py（仓库根入口）
created: 2026-10-13
updated: 2026-10-13
type: entry
tags: [entry,cli]
source: main.py
---

## 职责

仓库根入口文件，7 行。唯一工作：

```python
from ao_shaping.main import cli

if __name__ == "__main__":
    cli()
```

把 `python main.py …` 的全部调用转交给 [[entry/cli|`ao_shaping.main.cli`]]。真正的命令注册、全局选项、调试模式切换都在 [[entry/cli]] 内完成。

## 调用链

```
python main.py <command> [opts]
  └─ cli()                     ao_shaping/main.py:82
     └─ click 分发 20 个 add_command 注册的 runner（见 [[entry/cli]]）
```

## 关联

- [[entry/cli]]
- 全部命令页见 [[index|目录]] 的「CLI 命令」一节
