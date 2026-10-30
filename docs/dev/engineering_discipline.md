# 工程纪律与质量门禁 (ub-mode)

> 外迁自根 `AGENTS.md`。


任何改动（含 AI 代理）按三段推进，不跳段：

1. **先能跑（Make it Work）**——交付能解决问题的最小改动，不为"将来可能"预埋。
2. **再写对（Make it Right）**——用下面 5 条纪律和确定性门禁把改动写对，再谈扩展。
3. **守习惯（Practice）**——留行动日志、守护上下文、小步有序提交。

### 动手前五问（Make it Work 阶段）

- **一个决定一个位置**（Single Source of Truth）：同一常量/函数/格式/schema 只允许一份
  真源。第二份实现出现前先 grep 全仓消费者——已登记的 SSoT（`zernike_utils`、
  `beam_metrics`、`eval_stats` 等）不得在 runner/tools/GUI 内重写。
- **先减后加**（Subtract Before You Add）：动手前先把能删的删掉（死代码、过期脚本、
  被收编的重钉）。本仓 `scripts/` 的动作顺序固定为：先删过期，再迁复用，最后才谈拆。
- **攻前提**（Attack the Premise）：当两个共享同一前提的修复都卡在同一个门禁上时，
  把那个前提用一句话写下来，先消掉它，再谈下一个修复。不要在同一前提上反复打补丁
  （本仓典型：把"大文件行数"当前提去加 baseline 白名单，而非删被塞进去的重复 SSoT）。
- **数据形状先于逻辑**：先把持久化格式/入参类型/单位约定定对，再写分支逻辑。
  数据在入口处校验一次，内部信任它（Boundary Discipline）。
- **以可检查的终局为准**（Outcome-Oriented）：瞄准能验证的终态；中间允许断点，
  但断点必须计划过、可回滚（rebase/commit 粒度保证）。

### 写对的五条纪律（Make it Right 阶段）

- **修根因**：复现它，改根因。一个只把崩溃藏起来的守卫（`try/except: pass`、
  `# noqa` 压提示）不是修复。
- **测行为，不测实现**：测试像用户一样调用代码、检查用户能看到的输出。如果一个测试
  在"把所有函数改成 `return None` 后"仍能通过，就扔掉它。
- **幂等**：操作跑两次、或中途崩溃后重跑，要落到同一终态。CI 分片、`--mark-pushed`、
  报告重生成都必须幂等。
- **可验证单元**：一次只完成一个小单元并验证它，再开下一个；不要"先改完再统一测"。
- **类型纪律**：非法值不可被写出；外部数据在入口一次性解析；不压编译器（不 `# type: ignore`
  掩盖真问题）。

### 确定性质量门禁（本地，按窗口跑）

`ruff` 之外，对本窗口新增/改动的 `src/**` 用 ub-mode 三件套做结构性门禁。三者均
确定性（非 LLM），输出可复现，作为"需定方案/高风险"判定的客观证据：

| 门禁 | 工具 | 命令（在仓库根） | 判据 |
|------|------|------------------|------|
| 复杂度/测试 | `crapper`（CRAP 分数） | `/opt/data/.venvs/ubmode/bin/crapper --no-coverage src/ao_shaping/...` | 每函数 `CRAP = CC²·(1-cov)³ + CC`；`>30` 必拆（拆函数 / 补测试 / 两者），`5–30` 值得关注 |
| 重复 | `dryer`（结构 Jaccard） | `/opt/data/.venvs/ubmode/bin/dryer src/ao_shaping/...` | 相似度 `>=0.82` 的函数对列入重复清单，按"逐字比对确认真重复"处置（见 5+1 扫描 ③ 维） |
| 变异 | `mutator`（mutation testing） | `/opt/data/.venvs/ubmode/bin/mutator src/ao_shaping/... --test-command "..."` | 幸存者 mutant = 测试没覆盖到的行为；"覆盖未覆盖、杀死幸存者、删死代码" |

> 说明：`crapper`/`dryer`/`mutator` 是**本地辅助门禁**（不在 GitHub Actions 硬门禁内，
> 见 `docs/dev/ci_testing.md` 的 CI 现状节），用于在评审/扫描时给出客观的"该拆/该删/
> 该补测试"证据，替代纯主观判断。`crapper --no-coverage` 只算复杂度（不依赖覆盖率
> 报告，最省事）；需要 CRAP 全值时加 `--use-existing-coverage` 并先跑一次
> `./run_pytest.sh -- --cov=ao_shaping`。工具装在持久卷 `/opt/data/.venvs/ubmode`
> （`uv venv` + `git+https://github.com/unclebob/{crapper,dryer,mutator}.git`）。

