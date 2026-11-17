---
name: ponytail
version: 1.0.0
description: "Lazy senior-dev mode for AO-shaping code: the smallest complete change that fully works, plus a lean audit (what to delete, merge, split) that reconciles with this repo's function-size ratchet and shape guards. Use on any AO-shaping coding task (writing, fixing, refactoring, reviewing, dead-code cleanup, dependency choice) and when the user says 'ponytail', 'be lazy', 'simplest solution', 'yagni', or complains about over-engineering / bloat. Levels: lite, full (default), ultra."
---

# Ponytail（懒资深开发模式）— AO-shaping 定制版

> 核心一句话：**最好的代码是没写出来的代码。** 用最少的新代码解决整个问题；多余的每一行，以后都要读、测、修。
> 来源：[`DietrichGebert/ponytail`](https://github.com/DietrichGebert/ponytail)（MIT，"lazy senior dev mode"）。原文规则见 [references/original-ponytail.md](references/original-ponytail.md)。
> 本 skill = ponytail 通用规则 + 本仓库（AO-shaping）的落地约束：死代码判定协议、行数棘轮/形状守卫协同、WIP 护栏。

**优先级（冲突时从左到右）：correct > safe > 扛得住负载 > 有测试 > 快 > lean。**
lean 不是第 7 位可选项——每多一行以后都要读、测、修。但在前 5 项没满足前，不为了瘦而瘦。

**会话级生效**：进入 ponytail 后整轮会话遵守，直到用户说 "stop ponytail" / "normal mode"。
**强度**：`lite`（只交付所求，更小选项一句话点出让用户选）/ `full`（下面全部，默认）/ `ultra`（先质疑需求，凡需求不支撑的部分直接顶回去）。

---

## 0. 动手前（写代码之前，必做）

1. 读任务 + 它要碰的代码。**列出改动必须触达的每一处**：调用方、测试、fixture、配置、导出（`__all__`/`__init__.py` re-export）。
2. 评估改动对用户的影响：会销毁或暴露什么数据？会让哪些调用方失效？
   - **这是"范围"。没人要的功能、选项、灵活性，不属于范围。**
3. 本仓库特有：改动是否落在 `module_size_baseline.json` 的棘轮条目里、是否命中 `test_issue205_function_shape.py` 的形状守卫？若是，拆分/删除后**必须同步重钉基线**（见 §4 漂移重钉）。
4. 当前工作树是否有**另一个代理的未提交 WIP**？有则**不碰、不补推**（见 §5 护栏）。

---

## 1. 最小的完整改动（七级梯子，取第一个能完全工作的）

1. **要不要存在？** 没人要的功能/选项/灵活性，跳过并用一句话点明。模糊需求（"给我做个 X"）交付能完成核心工作的最小版本。
2. **本仓库已有？** 现成的 helper / 组件 / 服务 / 模式 → 按周边代码的用法直接用。
3. **标准库 / 平台特性？** 有就用（除非项目自造了同类）。**项目内组件优先于原生 widget。**
4. **已安装的依赖？** 用它。绝不为了几行代码加依赖。
5. **一行能搞定且读者一眼看懂？** 就一行。
6. **否则：能工作的最小代码。**

**纪律（lazy 是对方案，不是对改动本身）：**

- **改动要完整**：任务需要的所有部分都要做完，包括你改坏了的调用方、测试、fixture。
- 不写没人要的类型转换、option、config、boilerplate、"以后再说"代码。值保持平台给的原样。**删除优先于新增。** 保持代码库已有的结构：层、接口、约定。
- 已知所有必须触达之处后，**最短的可工作 diff 胜出**。需要解码的一行不算短。
- 注释只写代码表达不出来的 why，一行。
- **改 bug：动笔前先 grep 被改函数的所有调用方**，然后在共享代码里把根因修一次。
- 移动或合并的代码**保留它的错误处理与校验**。
- 大小相同的选项之间，取在边界情况上正确的那个。
- **没检查的 lazy 代码是半成品**：新增的非平凡逻辑（分支、循环、解析器、金额或安全、整个新脚本/程序）留一个小的测试或基于 assert 的自检。平凡改动可以不留。
- 有已知上限的捷径用注释标记：`shortcut: <上限>, <何时升级>`（欠账台账见 §3）。

**永不削减**：信任边界的校验、防数据丢失的错误处理、安全、可访问性、真机需要的标定、用户明确要的。

---

## 2. 瘦（lean）审计：删 / 并 / 拆

对"该不该存在、能不能更小"的代码，按六类判：

| 动作 | 判据 | 本仓库实例 |
|---|---|---|
| **delete** 删 | 死代码、无人用的 option/flag/config、投机性功能 | 孤儿 Mixin 模块、全树 0 引用的顶层函数 |
| **reuse** 并 | 两个 helper 做同一件事（留一个，指明路径） | 近孪生函数，`difflib` 行级 ratio ≥ 0.4 |
| **stdlib/native** | 标准库/平台已能做；为几行引入一个依赖 | 用 stdlib 替掉薄 wrapper |
| **yagni** | 只有一个实现的接口、只有一个产品的工厂、只透传的 wrapper | 删掉无消费者的抽象 |
| **merge** 合 | 必须一起改的近拷贝 | 见 [references/dead-code-audit.md](references/dead-code-audit.md) 孪生判定 |
| **split** 拆 | 一个函数/类做几件不相干的事，难读难测 | **按职责拆，不按行数拆**；绝不为变短而造只服务于缩短的 helper |

**拆的边界（与本仓库守卫协同，关键）：**
- 本仓库有 `scripts/check_module_size.py` + `module_size_baseline.json`（**模块级行数棘轮**）和 `test_issue205_function_shape.py`（**函数级 span/嵌套守卫**）。
- ponytail 说"按职责拆，不按行数拆"——但本仓库的棘轮是**防回长**的锁。所以：
  - 拆分/删除**导致基线条目行数下降**时，必须把 `module_size_baseline.json` 对应值**重钉到实测**（b162 的"漂移重钉"教训：45e06487 净 −765 行后 6 处 wiki + 4 处 live-doc 行号锚全漂移，必须重钉）。
  - 不要把"按行数拆成更短"当成目标；把"每个函数只做一件事 + 棘轮反映新基线"当目标。
  - 形状守卫的天花板是**实测值 + 余量**；不要在锁里悄悄抬高基线（用户 2026-11-14 明确否决过擅自 886→989）。

---

## 3. 欠账台账（shortcut: 标记）

每个**有意为之的** ponytail 捷径都用注释标记，格式 `shortcut: <上限>, <何时升级>`。
这使"推迟"不会悄悄变永久。收割欠账：

```bash
git grep --no-index --exclude-standard -nE '(#|//|/[*]) ?(shortcut|ponytail):' src/ scripts/
```

每行 = 一条台账：`<file>:<line>, <简化了什么>  上限: <命名的上限>  升级: <触发重访的条件>`。
**标 no-trigger**：任何没写升级路径的标记，是那种会悄悄腐烂的。

---

## 4. 漂移重钉（本仓库强制，ponytail 原文没有，是 b162 的活教训）

**删了行 / 拆了函数后，三处必须一起更新，缺一即 CI 红或锚失效：**

1. `scripts/_common/module_size_baseline.json` —— 对应模块行数重钉到实测（`check_module_size.py` 只允许降不允许涨）。
2. `tests/ao_shaping/test_issue205_function_shape.py` —— 被拆函数的 span/嵌套天花板按新函数集重钉。
3. 行号锚文档：`docs/dev/call_graph_wiki/**`、`docs/**`、`TODO.md` 里指向被改文件的 `:NN` 行号锚。
   - 大改动后跑 `git log --oneline -3 -- <file>` 找拆分 commit，对照 `docs/dev/call_graph_wiki/` 与 `docs/slm/` 的 `file.py:NN` 引用，漂移的一并重钉。
   - 纯删行（无函数签名变化）一般只影响 1 和 3，不影响 2。

**验证顺序**：改代码 → `ruff check` → 形状/约定守卫（`test_issue205_function_shape.py`、`test_conventions.py`）→ 受影响模块的逐文件测试（`AO_TEST_PER_FILE=1`）→ 棘轮 `check_module_size.py` 绿。

---

## 5. 护栏（AO-shaping 特有，ponytail 原文没有）

1. **不碰 WIP**：工作树里若有另一代理的未提交改动（如 `src/ml/gsnet/`、`src/ml/sim2f/` 的在制工作），**不碰、不补推、不 rebase 进它的文件**。ponytail 的"最短 diff"不适用于"顺手把别人的 WIP 提交了"。
2. **不抬基线**：模块尺寸/函数形状的棘轮天花板，不许自行抬高；超限先做"过度设计评估"，压不回才呈"重钉 + 理由"求批准。
3. **CLI 契约冻结**：`main.py` 的 `add_command` 名 = `EXPECTED_COMMANDS` = `cli_help_golden.json` 顶层 key = `main.py` docstring 命令表，四源必须一致。删一个 runner / param dataclass 前，确认这四个面与 `docs/user/cli-reference.md` 不产生幽灵命令。
4. **死代码判定必须过全树 grep**：本仓库 import 图看不见 `register_*_lazy("name", "ao_shaping...driver", "Class")` 字符串注册、`python -m`/`streamlit run`/`click` 的入口组合、`AGENTS.md` 与厂商文档的约定面、迁移与重构意图。**纯 import 图在本仓库大面积误报"死"。** 完整协议见 [references/dead-code-audit.md](references/dead-code-audit.md)。
5. **每轮扫描范围**：新增 commit 逐条审（拆分是否合法、有无引入死符号）+ 全库 `ponytail_lean_scan.py`。`scripts/` 的问题只列报告不修（用户 2026-10-27 规矩）；`src/` 的简单问题与重构类直接修，真正难的/有争议的才开 issue。

---

## 6. 报告纪律（ponytail 审计的输出格式）

- **每个发现必须有具体 case**（此 grep 0 引用 / 此输入→此错误结果）。无 case，无发现。
- 发现四段式：**是什么 / 问题 / 修法（最小，优先删代码）/ 跳过后果**。
- 分组：**Must fix / Should fix / Nice to have**，跨组统一编号（用户可说"修 2 和 5"）。
- `shortcut: <上限>, <升级触发>` 标记 = 决策，不是发现（除非负载已越过上限）。
- 结尾：`Verdict` 一行 + `Lean: -N 行可能` + `Not checked` 一行。
- 无发现：`What this repo does:` + `Healthy. Nothing to fix.` + 一行说明查了什么。

**无风格品味、无 "consider"、无模糊担忧。**

---

## 7. 配套脚本

```bash
# 死函数 / 未 import 模块候选 / 同名孪生候选（全库 ~35s）
python3 libs/skills/ponytail/scripts/ponytail_lean_scan.py <repo_root>
```

输出四组 JSON 候选（`dead_strict` / `dead_maybe` / `unimported_candidates` / `twin_name_candidates`），**全部是"待复核"，不是结论**。
- `dead_strict`：全树 0 命中 → 仍要过 §5 的全树 grep 协议（含字符串注册/入口组合/约定面）才能判死。
- `unimported_candidates`：import 图未命中 → **绝大多数在本仓库是活的**（入口脚本、字符串注册、延迟 import、脚本直接 import），无第 2 步复核不得入报告。
- `twin_name_candidates`：跨 ≥3 个 src 模块的同名顶层函数 → 需 `difflib` 行级 ratio ≥ 0.4 才报孪生。

脚本与判定协议细节见 [references/dead-code-audit.md](references/dead-code-audit.md)。

---

## 诚实边界（ponytail-gain 的诚实边界，移植）

ponytail 基准（5.0 版，39 任务 × 5 run，无技能对照）声称 −53% 代码 / −41% 时间 / −26% 成本 / −45% token，"需要测试的逻辑 98% 带测试 vs 无技能 68%"。**这些是基准均值，不是本仓库。**
**绝不在活仓库里印"你这里省了 X 行/token"的数字**——没写出来的版本从未被写出，活仓库里没有可减的真实基线。唯一的仓库级实数是 `/ponytail-debt`（计数的欠账台账）与 `ponytail_lean_scan.py` 的 `Lean: -N 行` 候选。
