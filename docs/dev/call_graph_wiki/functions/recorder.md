---
title: Recorder（优化过程记录器）
created: 2026-10-13
updated: 2026-10-29
type: class
tags: [io, recorder, utils]
source: src/ao_shaping/utils/io/file.py:472
---

# Recorder（优化过程记录器）

## 职责

逐行记录优化过程的运行记录器：每个优化器在每次迭代（或每个采样）后 `append`
一行 dict（含 `mark` 标记列），最终汇总为 DataFrame，`get_best_*` 系列按指定列取最优行。各命令页"结果/调试落盘"段里的 `save_recorder_debug_artifacts` 就是消费 `Recorder`
产物的落盘函数（见 [[functions/debug-artifacts]]）。

## 结构（`src/ao_shaping/utils/io/file.py:472`）

```python
class Recorder:
    def __init__(self, mark: str = "J", mode: Literal["max", "min"] = "max")
    def append(self, record: dict)                    # 追加一行（自动补 _id）
    def postprocess_feature(self, feature_name, func, column="")  # 注册派生特征列
    def _ensure_postprocess_applied(self, df) -> DataFrame
```

- `mark`：记录行的标记列名（断言 `mark in record`），用于区分不同优化器/阶段的行。
- `mode`：`"max"`/`"min"`，决定 `get_best_*` 取最大还是最小。
- `history`：`list[dict]`，每行一个 `_id` 自增。
- `postprocess_feature`：注册 `func(row_dict) -> value`，落盘/导出时自动补算派生列，
  失败时 `logger.warning` 并置 `None`（单列失败不中断整表）。

## 谁使用它

- 各优化器（[[functions/optimizer-rms-dm]]、[[functions/optimize-pib-wfless]]、
  [[functions/optimize-slm-square]] 等）在迭代循环内 `rec.append({...})`。
- [[functions/debug-artifacts|`save_recorder_debug_artifacts`]] 在 `--debug` 时把
  `Recorder.history` 导出为 pkl/csv。

## 相关页面

- [[functions/debug-artifacts]]
- [[functions/path-dirs]]
- [[entry/cli]]
