# Task B2: 三个代码 bug（#140 + #143 问题1/问题2）

仓库: /tmp/ao_taskB2 (branch fix/taskB2-bugs, 基于 e95eb173)
venv: .venv（符号链接）

## 红线
- 无硬件，全 offline。#140 的修复必须 sim/offline 可验证。
- 每个 issue 独立 commit，message 带 issue 号。
- 术语规范：新增/改动的中文 docstring/注释必须遵守 `docs/terminology_glossary.md`（瞳面/实验系统/杂散光 等，见 AGENTS.md 头部）。

## Issue #140: slm_gs_refine 细化循环 best_phase 保存的是胜出的随机探针
现状（gh issue 140）：`src/ao_shaping/optimizer/wfless/slm_gs_refine.py:871-930` 主循环每轮只测 `dof+delta` 与 `dof-delta`，`best_phase = measured_phase`（胜出的 ±delta 探针，:923-925），落盘相位带 ±0.35 rad 随机偏移，与"收敛迭代点 dof"语义不符。
修法（issue 的两条建议都要做）：
1. 每轮补测收敛点：记录未扰动 `dof` 的分数（在现有测量预算内，若每轮只允许 2 次测量，则改为记录 `max(pos, neg)` 的同时把 `dof` 的分数也记入 Recorder/log，分开两列，区分"优化器推进"与"探针选择偏差"）。
2. `best_phase` 改为收敛点：`best_phase = phase_of(dof)`（dof 是本轮迭代点；若担心 dof 未实测，用该轮已测的 max(pos,neg) 分数作为 dof 的保守下界来更新 best 判定，落盘相位取 dof 对应的相位图）。
- 注意 confirm-best 机制（重测历史最优）保持原语义。
- 加 offline 测试：构造 stub 测量函数（正负探针分数确定、dof 分数确定），断言落盘的 best_phase 对应 dof 而非 ±delta 探针（相位数组相等断言）。

## Issue #143 问题1: str(None) 生成 `none` 文件名
现状：hwdataset index 的 JSON dump 被误命名成文件 `none`，某处 `str(None)` 当路径。
修法：
1. 在 `src/ml/hwdataset/index.py`（或相关 dump 代码，用 grep 找 `str(` + 可能 None 的路径变量；`git log -S` 搜 `none` 文件名生成点）定位写出点：找出哪个变量可能是 None。
2. 路径变量为 None 时显式 raise（`raise ValueError(...)` 说明哪个字段为 None），不要落成 `none` 文件。
3. 回归测试：mock 一个会让该变量为 None 的输入，断言 raise 且不生成 `none` 文件。

## Issue #143 问题2: sweep 空结果覆盖非空文件
现状：`report/loss_defects/padding_sweep.json` 曾被并发 sweep 跑成 `{}`，无防护。
修法：
1. 找 sweep/批量写出 JSON 的公共写出点（`report/loss_defects/` 相关脚本 + 任何 `json.dump` 到已存在非空文件的写出路径；优先收敛到一个 helper，如 `scripts/_common/` 下）。
2. guard：结果为空（空 dict/空 list）**且** 目标文件已存在 **且** size>0 时，拒绝覆盖并 `logger.warning`（记录被保护的文件名）。
3. 加一个拒绝覆盖的测试：先写非空文件，再用空结果调写出，断言文件内容不变 + warning 被记录。

## 验证
```
.venv/bin/python -m pytest tests/ao_shaping/optimizer/wfless -q
.venv/bin/python -m pytest tests/ao_shaping/ml -q   （或 hwdataset 对应子集）
.venv/bin/python -m ruff check src scripts tests --select F401,F811,F821,F841
```
报告：每个 issue 的 commit sha + 新增测试名 + 验证输出尾部。
