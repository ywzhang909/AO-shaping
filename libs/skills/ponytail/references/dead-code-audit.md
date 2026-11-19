# 死代码 / 孪生 / 未 import 模块 判定协议（AO-shaping 实测版）

来源：`ponytail` 的 lean 审计（correct > safe > load > tested > fast > lean）在本仓库的落地。
本文件自包含——cron 子代理与扫描脚本不依赖 Hermes 技能库，按此协议独立复算即可。
配套脚本：`libs/skills/ponytail/scripts/ponytail_lean_scan.py <repo_root>`（全库 ~35s，输出 4 组**候选**，非结论）。

> **本仓库最大的坑**：纯 import 图（`ponytail_lean_scan.py` 的 `unimported_candidates`）
> 在 AO-shaping **大面积误报"死"**。原因见下"误报来源"。任何未过全树 grep 的清单
> **不得**进入报告。

## 误报来源（为什么 import 图不可信）

1. **字符串惰性注册**：`register_*_lazy("name", "ao_shaping...driver", "Class")`——
   nlight driver 就靠字符串活，import 图完全看不见。
2. **入口组合**：`python -m ao_shaping.xxx`、`streamlit run gui/...`、`click` 命令树
   （`main.py` 的 `_LAZY_RUNNERS`/`_IMPORT_ATTRS`）——模块级 `__main__`/`def main` 不被任何
   模块 import 但活着。
3. **延迟 import**：函数体内 `from ao_shaping.x import y`，level-0 import 图只到顶层。
4. **约定面**：`AGENTS.md`、`docs/`、厂商文档里命名的符号（`DeviceBusyError` 在
   `docs/drivers_architecture.md` 是驱动错误层级的一部分，属公开 API 面，不是死）。
5. **迁移/重构意图**：`TODO.md` 写明"计划迁到 tools/slm/"的 runner，正确动作是迁移
   不是删除。
6. **`__init__.py` re-export / `__all__`**：符号经包 `__init__` 再导出。
7. **测试常量表 / fixture**：文件路径或符号名出现在 set/list 常量里
   （gready_cam 在 `test_spgd_convention` 的豁免集合里，"未被 import"但活着）。

## 死代码检测三步（第 2 步是全部误报的来源，不可省）

1. **AST 候选**（脚本自动）：`src/` 顶层 `def`/`class`（排除 `__init__.py`）。
   候选条件 = 全树标识符计数（除本文件）= 0 且本文件额外使用 ≤ 1。
   用 `libs/skills/ponytail/scripts/ponytail_lean_scan.py`。
2. **全树 `grep -w` 复核**（用 ripgrep / `search_files`，**勿用 `grep -rn`**——
   含 `runs/data` 大数据目录的仓库会 >180s 超时）。必须覆盖以下字符串/惰性形态：
   - `register_*_lazy("name", ...)` 字符串注册
   - 函数内 / 延迟 `from ... import ...`
   - `__init__.py` re-export、`__all__`
   - 测试常量表（路径/符号名在 set/list）
   - docs/TODO 提及（不算代码引用，但说明**有保留意图**——读上下文再判）
   - 非 .py 文件：`*.md *.json *.ipynb *.yml`（`rg -w <name> -g '!*.py'`）
   - **`src/ tests/ scripts/` 之外的 .py**：`runs/`、`notebooks/`、`_scripts/`、
     `report/`、`main.py`（`rg -w <name> runs/ notebooks/ _scripts/ report/ main.py`）
3. **分级**：
   - **strict**（全树除 def 行 0 命中）→ 仍要过第 2 步的字符串/入口/约定面复核后才能报死。
   - **maybe**（本文件 1 处额外出现，可能是 docstring 自述）→ 需人眼。
   - 报告**只报 strict 且第 2 步通过**的。

**2026-11-18 实测**：宽松条件 847 个候选 → strict 30 个 → 过全树 grep 后，
多数被 docs/TODO/`register_*_lazy`/入口组合/约定面救活，真死是少数。
死代码**必须整条链路确认后单独开条目删**（先立失败测试 → 删 → 全量分片复验），
不顺手删。

## 未 import 模块：清单 = 误报候选

`ponytail_lean_scan.py` 的 `unimported_candidates`（import 图 level-0 未命中）
**全部**要逐个复核。2026-11-18 实测 52 个 "unimported" 里绝大多数是活的：
- 入口脚本（`__main__`/argparse/`def main`）
- `scripts/` 直接 import 的库模块（`oopao_backend`）
- 延迟 import（`wandb_logger`）
- 字符串注册（nlight driver）

**没有第 2 步的 unimported 清单不能出现在报告里。**

## 孪生检测（同名 ≠ 孪生）

- 跨 ≥3 个 src 模块的同名顶层函数只是**候选**（排除 `main`/`run`/下划线开头）。
- 用 `ast.get_source_segment` 取函数源码，`difflib.SequenceMatcher(None, a.splitlines(), b.splitlines()).ratio()`。
- 阈值：**≥0.4 才报**。
  - ≥0.4：ceramic_viewer 两份 = 0.47 → 报（但先查 git 是否已抽共享层，见下）。
  - <0.2 是同名巧合：`render_sidebar` ×7 = 0.06–0.17 → **不报**（7 个不同 GUI 面板）。
- **报前查 git 历史是否成对变动**：`git log --oneline -- fileA fileB`，
  看 commit 是否交替出现。若已抽共享层（如 `gui/ceramic_data.py`），
  剩余是各自 UI，**不"必须一起改"，不合并**（ponytail 规则：merge 只针对
  "必须一起改"的近孪生）。

## 拆分孤儿产物（orphaned leaf）

- `git log --oneline -3 -- <file>` 找"拆出/拆分为叶子模块"的 refactor commit。
- 若逻辑主体已搬回父类（对照：函数签名 + docstring 逐字相同，
  如 `AutoExposureMixin.auto_exposure` vs `DahengCamera.auto_exposure`），
  叶子文件即孤儿，**整文件可删**。
- 2026-11-18 实测：`0772202b` 拆出的 3 个 Mixin 文件全 0 引用 = 真孤儿（已删，−621 行）。

## 与行数棘轮 / 形状守卫的协同（本仓库特有）

ponytail 说"按职责拆，不按行数拆"，但本仓库有防回长的锁：
- `scripts/check_module_size.py` + `module_size_baseline.json`（模块级行数，只降不升）
- `tests/ao_shaping/test_issue205_function_shape.py`（函数级 span/嵌套天花板）

删行/拆函数后**三处必须一起更新**，缺一 CI 红或锚失效：
1. `module_size_baseline.json` 对应模块重钉到实测（用 `wc -l` 或 `check_module_size.py` 现值）。
2. `test_issue205_function_shape.py` 被拆函数的 span/嵌套天花板按新函数集重钉。
3. 行号锚文档：`docs/dev/call_graph_wiki/**`、`docs/slm/**`、`TODO.md` 里
   指向被改文件的 `:NN` 行号锚（`git log` 找拆分 commit，对照 `file.py:NN` 引用）。

**教训（b162 "漂移重钉"）**：`45e06487` 净 −765 行后 6 处 wiki + 4 处 live-doc
行号锚全漂移，必须逐处重钉。纯删行（无签名变化）一般只影响 1 和 3。

## 工具坑（2026-11-18 实测）

- 全树 `grep -rn` 在含 `runs/data` 大数据目录的仓库 **>180s 超时**
  → 用 ripgrep（`search_files`，自动忽略 .gitignore）或 `rg -w <name> -g '!*.py'` 限定。
- Python AST 死代码扫描：O(N²) 逐文件 `str.count` 在 25 万行仓库超时
  → 单次 regex 预计算 per-file `Counter`（~35s 完成），
  或 `terminal(background=True, notify_on_complete=True)` 跑。
- `find | xargs wc -l | tail -1` 只拿到 xargs **最后一批**的 total（xargs 分批）
  → 用 `find -print0 | xargs -0 cat | wc -l`。
- `wc -c`/`ls` 输出含 `.venv`（18k py 文件）会污染统计
  → 一切 find 先排除 `.venv` `__pycache__` `.git`。
- `git grep --no-index` 收割 `shortcut:` 欠账时加 `--exclude-standard`，
  跳过 .gitignore 忽略的目录。
