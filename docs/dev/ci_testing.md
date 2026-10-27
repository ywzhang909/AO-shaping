# CI 与测试策略

> 本文件从根 [AGENTS.md](../AGENTS.md) 的 `Build/Lint/Test Commands` / `MISSING INFRASTRUCTURE` 节外迁（2026-10-27）。

## 常规测试默认排除（pytest addopts）

### 常规测试默认排除（pytest addopts，2026-10-16；2026-10-24 按实测撤销 drivers 排除；2026-10-26 按实测撤销 AppTest 排除）

`pyproject.toml` 的 `addopts` 默认 `--ignore` 一类**会阻塞**（挂死、无输出）的测试，
容器里的 `pytest` / `pytest tests` 不再踩到它们（从仓库根目录运行）：

| 排除 | 原因 |
|---|---|
| `tests/ao_shaping/ml` | 65 GB hwdataset 重 I/O + torch：多进程/mmap 曾把整树 `pytest tests` 挂死 |
| ~~`tests/ao_shaping/drivers`~~ **已撤销** | **原排除整棵 drivers 树过度保守一个量级，且造成了真实的可见性事故**：`drivers/ccd` 曾烂到 **6 failed + 1 error** 而 CI 毫无察觉（`29001826` 修）。逐域实测后全部恢复进发现集：`ccd` 204/25、`sim` 394/2（63.33 s）、`wfs` 110/11、`slm` 158/127（7.95 s）、`adc` 49（2.84 s）、`hardware` 37 skipped（0.06 s）、`dm` 139/5（**40.22 s**）、根文件 13/9；整棵 `drivers` 单 session **1067 passed / 216 skipped / 124.43 s** |
| ~~`tests/ao_shaping/gui/slm/test_zernike_abort_repro.py`~~ **已撤销** | 原排除理由是「唯一跑**真 `AppTest`** 的测试，需网页自动化配合，否则挂死」—— **该理由不成立**：它是**无浏览器**的 AppTest，跑真实 GUI 函数（`ZernikeControl.render` / `render_pattern_controls`），2026-10-26 实测 **2.06 s / 3 passed**。它当初是「复现」Zernike 切换时的 abort，而现在的代码已不会触发。作为**唯一**端到端渲染真实 GUI 运行时的测试，排除它等于白丢覆盖率 |

🔴 **`drivers/` 的 `--hardware` 门必须在 `drivers/conftest.py`**：跳过钩子原先只存在于
`drivers/slm/conftest.py` 与 `drivers/hardware/conftest.py`，而 `pytest_collection_modifyitems`
是**会话级**钩子 ⇒ 某条 `@pytest.mark.hardware` 测试是否被跳过，取决于**同会话里还收集了哪棵无关子树**
（`pytest drivers/dm drivers/slm` 会跳过，单独 `pytest drivers/dm` 不跳过）。后果不是观感问题：
`dm` 域曾因 `TestMicroDMReport` 的标记无人执行，fixture 把 26 个不可达控制器逐个扫到底 ——
**每测试 260 s 的 setup，520 s 占掉该域 560 s 的绝大部分**。新增 `drivers/conftest.py`
（复用 `drivers/_hardware_gate.py` 单源）后 `dm` 变 **40.22 s / 0 failed**。
⇒ 判"某域慢/挂起"必须先用 `--durations` 归因，不要按域整体归因。

🔴 **本地剪除与 CI 收集必须同向，但两份清单不必相等**：`pyproject.toml` 的 `--ignore` 只剪**本地默认**的
发现集（当前仅 `tests/ao_shaping/ml`：65 GB 语料 + torch 多进程曾把全量运行挂死），而
CI 按 `.github/workflows/pytest.yml` 的 `run_shard` 目标**显式收集**这些区域。两侧清单曾被要求
逐路径相等（脚本从 scratch 目录启动 pytest，pyproject 的**相对**排除在那种 CWD 下匹配不到任何
东西），该等式在 CI 开始有意收集 `ml` 之后即作废：强行相等就等于把整区 `--ignore` 塞回 harness，
让 CI 重新对该树变盲——那正是 2026-10-24 那次漂移与 TODO.md 8.9 items 102–109 的成因。
现由 `tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py::test_every_test_area_is_sharded`
逐文件钉住（`tests/ao_shaping` 下每个测试模块都必须落在某条 `run_shard` 目标里，被本地剪除的 `ml`
树因而仍由 CI 收集）；harness 自身的 `EXCLUDES` 由同文件的
`tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py::test_no_harness_ignore_blanks_a_whole_test_area`
与其同伴守卫逐条审（不得悬空、必须带理由、不得整区豁免、解析器不得空洞）。

⚠️ `--ignore` 只作用于 **discovery**（从 `testpaths` 发现的裸 `pytest`）。2026-10-24 实测：
裸 `pytest` 收集 **9293**；被排除的是 `ml` **1415** + AppTest **3**（两者亦实测）。
撤销 drivers 排除前为 **8011**，即整棵 drivers 树原本藏了 **1286** 个测试。

## 缺失基础设施 / CI 现状

## MISSING INFRASTRUCTURE

- **CI**: 2026-10-15 曾建立 ruff 硬门禁（`.github/workflows/ruff.yml`，push + PR 触发，
  `ruff check .`，issue #132 决策 B）与参考级 `pylint.yml`（`continue-on-error: true`）；
  2026-10-19 `.github/` 曾整体移出 git 跟踪（本机保留）；
  **2026-10-21 恢复入库**（`1d33d078`，现 `.github/workflows/ci.yml`：ruff 硬门禁
  + best-effort pytest 作业 `continue-on-error`），新克隆重新自带 CI 工作流 ——
  本地 `ruff check .` 仍须保持全绿。
  ✅ **2026-10-24：pytest 已成硬门禁**（`.github/workflows/pytest.yml`，push + PR）：
  一次镜像构建 + 分片（2026-10-24 建立时为 **8 片**；2026-10-27 由 TODO items 108–109 把 `ml`
  区域补齐、新增 `ml-gsnet-debug-offline` 与 `ml-gsnet-debug-rest` 后现 **11 片**），每片一个 pytest 进程，`AO_CPUS=4 / AO_MEMORY=6g`
  （ubuntu-latest 4 vCPU / 16 GB，留头寸）。`ci.yml` 里那个整目录单进程的
  `continue-on-error` 作业已删除 —— 它在 runner 上**不可能通过**（实测无目标参数的
  `run_pytest_ci.sh` 在 16g 上限下 34% 处 SIGKILL，exit 137）。分片清单只列**目录**、
  不列文件（文件清单会在新增测试时静默失盖），由
  `tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py` 两向钉住：
  每个顶层测试区必须被某片覆盖、每个分片目标必须存在、且至少要有一个阻塞分片
  （否则"永远红不了的作业"等于没有门禁）。`optimizer` 片用 `AO_TEST_PER_FILE=1`
  （见下），实测覆盖率与单进程一致（1259 passed + 3 skipped == 1262 collected）。
   **2026-10-25 起 8 片全部 blocking，无 advisory**：最后一片 `scripts` 的名义原因是
   wt-190 的 `_fmt` 去重 —— 该去重**已入库**（`741ea0e5` 是 master 祖先），但 `273181e6`
   又**重新引入**一份本地 `def _fmt`，守卫在 master 上**仍红**；本账本 §8.9 第 44 项把该
   副本改走 `scripts/_common` 的 `fmt_general` 后实测整片全绿
   （**2423 passed / 64 skipped / 0 failed**，`AO_CPUS=4 / AO_MEMORY=6g`），当日转 blocking。
   `runners-gui-misc` 曾因 `slm-gsnet --help` 基线红而挂 advisory（wt-193 在办），
  **2026-10-25 已转 blocking**：基线由上游 `c5cd68a9` 重做后实测整片全绿
  （1149 passed / 30 skipped / 0 failed，`AO_CPUS=4 / AO_MEMORY=6g`）。
  ⚠️ 门禁是 **fail-closed** 的：镜像构建失败会让整个作业变红，这是有意取舍
  （宁可红也不允许"没跑测试却绿"）。
  仍缺：类型检查门禁（mypy / pyright 未配置）。
  注：本机工作树还残留两个未跟踪的 `ruff.yml`/`pylint.yml`（2026-10-19 移出跟踪的
  本地遗留，现已被 `ci.yml` 取代）—— 属本机状态，勿提交。
  ⚠️ **pytest 门禁必须分片跑，不能单进程整目录跑。** 2026-10-20 实测：sim 后端
  每次测量做 **7680×4800 complex128 FFT**（面板 1920×1200 × padding 4 = 36.9 M 点，
  单缓冲 0.59 GB，一次测量含 spectrum/`abs**2`/`fftshift` 副本约 2 GB 瞬时分配），
  而 numpy/Python 分配器不把已释放内存还给 OS —— 整目录单进程跑到几百个测量时
  工作集涨到 **20–27 GB**（63.7 GB 机器上页文件开始换页）。两个这样的 pytest
  同时跑就会互相拖成「假挂起」（实测进程 99% user CPU、页错误 0、可用内存 36 GB
  —— 是分配抖动，不是死锁）。分片（按文件）跑则每个文件都在秒级～分钟级完成，
  全部文件合计约 20–25 分钟。
  ⇒ 该口径已进机器：`scripts/build/run_pytest_ci.sh` 的 `AO_TEST_PER_FILE=1` 就是它，
  **每文件一个 pytest 进程**，清单由 `--collect-only` 现生成（自动遵守 EXCLUDES，不会因
  新增文件而静默漏测），**收集到 0 个文件即 FATAL** —— 一片「什么都没跑却绿」比红更糟。
  ⚠️ 另：**别把「日志里点数停在某个位置」当成挂起** —— `Tee-Object`/`Select-Object`
  管道会缓冲，实测「停在 216」的那次运行其实以 28 点/分钟正常推进，点数是缓冲
  量化出来的假象。判据要看进程 CPU 与内存，而不是日志末尾的点数。
- **Linting**: ruff 已配置（`ruff.toml`，`select = [E, F, RUF100, B023, PLR0124]`，
  `libs/**` 与
  `notebooks/**` 显式排除，守卫见
  `tests/ao_shaping/ml/test_train_cli_runnable.py::test_ruff_excludes_nothing_under_src`）；
  mypy / flake8 未配置。
  🔴 **ruff 版本只有一把**：`ci.yml`、`ruff.yml` 与 `pyproject.toml` 的 dev 组下限必须
  同为一个**精确版**（现 `0.15.12`），由
  `tests/ao_shaping/test_conventions.py::test_all_ruff_pins_agree_and_are_exact` 钉住。
  此前两处不一致（`ci.yml==0.11.10` vs `ruff.yml>=0.15.12`）实测**不等价**：`PLC0415`
  在 0.11.10 是 preview 规则（**0** 命中，ruff 自己会打印 "Selection `PLC0415` has no
  effect because preview is not enabled"），在 0.15.12 已稳定（**1476** 命中）；
  `SLF001` 差 2 条。硬门禁不得写 `>=`（新版本能在两次推送之间无代码变化地把门禁翻红）。
- **machine-local 开发工具配置**: `.mcp.json`、`.pre-commit-config.yaml`、`opencode.jsonc`、`pyrightconfig.json`、`skills-lock.json` 已移出 git 跟踪并加入根 `.gitignore`（2026-10-19，仅保留在开发者本机，不再入库）；同日 7 个 machine-local 目录整体移出跟踪并忽略：`.auto_doc_review`/`.claude`/`.codegraph`/`.slim`/`.streamlit`/`.vscode`/`_scripts`（本地副本保留，不随仓库分发）。**注意 `.github/` 不在其列**：它 2026-10-19 曾一并移出，2026-10-21 已恢复入库（见上文 CI 条，`1d33d078`），根 `.gitignore` 中 `.github` 的忽略行同日删除。
- **No requirements.txt**: Only `pyproject.toml` and `uv.lock`

Consider adding: 类型检查门禁（mypy / pyright 未配置）。pytest 硬门禁已于
2026-10-24 入库（见上文 CI 条），ruff 门禁已入库，无需重新提交工作流文件。
