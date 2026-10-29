# 多形状整形 campaign UI (multishape_campaign_ui.py) — 软件需求规格说明书 (SRS)

## 1. 概述

### 1.1 文档目的

本文件定义 `src/ao_shaping/gui/slm/multishape_campaign_ui.py` 的功能需求、非功能需求、
接口规范与关键设计约束。该页面是
`report/shape_trials/multishape_scale/multishape_scale_live_20261028.md` §3.2 那次实验
（4 形状 × 4 阶段 = **16 段 / 112 块**）的操作面：**主要输入只有目标形状与尺寸**，其余
硬件与算法参数由自动辅助层派生，并逐行标注来源。

立此页面的理由是那一次实验留下的两个**结构性**失败模式，不是"人粗心"：

| 失败模式 | 表现 | 本页的对策 |
|---|---|---|
| 参数漂移 | 逐条 lane 手敲 PowerShell，同一批参数在两次运行之间悄悄不同 | 参数由 `campaign.derive_settings` 从「形状 + 尺寸」派生，人只改自己想改的，且改过立刻标「手工」 |
| 来源丢失 | 事后无法回答"这个值是哪来的" | 每个参数带 `FieldSource`（实测 / 派生 / 默认 / 手工）与一句说明，② 的参数表逐行着色 |

### 1.2 启动方式

```bash
streamlit run src/ao_shaping/gui/slm/multishape_campaign_ui.py
```

独立 `streamlit run` 入口：**不**挂进 `multi_slm_controller.main()`。后者已被
`diff_opt_controller` 单向 import，反向挂载会闭合成环。

### 1.3 依赖模块

| 模块 | 职责 |
|------|------|
| `ao_shaping.tools.slm.transfer.campaign` | 形状 + 尺寸 → 全参数派生（`derive_settings`）、来源标注（`FieldSource`/`Setting`）、矩阵与预算（`campaign_matrix`/`matrix_totals`/`estimate_seconds`/`estimate_disk_gb`）、`SIZE_WINDOWS` |
| `ao_shaping.tools.slm.transfer.campaign_progress` | campaign 目录的唯一进度读取器（`collect_campaign_progress`、`StageRow`/`BlockRow`、`COMPLETE_STATUSES`、`MISSING`） |
| `ao_shaping.tools.slm.transfer.preflight` | `check_zero_overlap`（零级剔除占比）、`WINDOW_MARGIN_PX`、`MIN_SIDE_PX` |
| `ao_shaping.tools.slm.transfer.stages` | `STAGE_RECIPES` 四阶段配方（单一事实源） |
| `ao_shaping.tools.slm.transfer.light_guard` | `OpticalPathInvalid`（`exit_code = 3`、`reason`、`cause`、`report()`） |
| `ao_shaping.gui.slm.precheck` | ⑤ 的六项检查与门禁（`build_check_card`/`render_check_card`/`CheckCard.blocking`） |
| `ao_shaping.gui.slm.transfer_progress` | `SHAPE_LABELS`（canonical）等展示件 |
| `ao_shaping.gui.slm.bench_settings` | 侧栏实验系统配置的只读取值（`bench_value`/`merge_bench_values`/`bench_busy`/`widget_default`） |
| `ao_shaping.runners.slm.transfer_shapes_runner.execute_campaign` | 真正跑 campaign 的可调用入口（与 CLI 同一实现） |
| `ao_shaping.utils.image.target.shapes.TARGET_SHAPE_LABELS` | 形状中文名唯一来源 |

### 1.4 用户画像

- 光学实验操作员：选形状与尺寸，看检查卡，点开始，读逐块进度
- 复盘者：事后挑一个 campaign 目录，离线看逐段 / 逐块读数（⑦）

### 1.5 与 CLI 的关系

本页**不**重写运行逻辑：开跑时把参数快照交
`transfer_shapes_runner.execute_campaign(**kwargs)`，与
`python src/ao_shaping/main.py slm-transfer-shapes ...` 走同一份实现，因此
`tests/ao_shaping/runners/test_transfer_shapes_runner.py` 的 CLI 契约测试对本页同样成立。

---

## 2. 功能需求

页面固定七段，顺序即操作顺序：

| 段 | 标题 | 无设备时可用 |
|---|---|---|
| ① | 目标（唯一必填） | 是 |
| ② | 自动辅助设置 | 是 |
| ③ | 计划预览 | 是 |
| ④ | 模型目录（自动匹配） | 是 |
| ⑤ | 运行前检查 | 是 |
| ⑥ | 运行 / 逐块进度 | 部分（未连接 SLM 时「开始」禁用） |
| ⑦ | 历史会话 | 是 |

### 2.1 ① 目标（唯一必填）

- FR-1：形状多选，选项为 `square` / `triangle` / `star` / `circle`（campaign 只用这四种；
  `rectangle` 从未进入 campaign，故不提供）。
- FR-2：边长（px）预设 `(1200, 600, 200)`，另可自填 `SIDE_CUSTOM`（"其他"）。
- FR-3：形状中文名一律经 `SHAPE_LABELS.get(name, name)`；字典里没有的名字原样显示
  （`SHAPE_LABELS` 只是展示层，不得当作校验集合）。
- FR-4：一个形状都不选时，②–⑥ 不渲染，只留 ⑦（历史浏览与本次选择无关）。

### 2.2 ② 自动辅助设置（参数分层）

- FR-5：参数表由 `CampaignSettings.settings` 生成，一行一个参数：名称 / 值 / 来源徽标 / 说明。
- FR-6：来源四类，配色固定：**实测**（本会话实测，绿）、**派生**（由形状 + 尺寸或报告
  口径推出，蓝）、**默认**（runner 的 CLI 默认，灰）、**手工**（人改过，黄，说明里必须记下
  被替换掉的原值）。
- FR-7：只暴露几个值得人改的旋钮（采样窗半跨 `window_span`、载波周期 `period`、探针
  采样网格 `capture_grid`）；其余只读展示。手工值存 `msc_overrides`。
- FR-8：「按尺寸重新推导（清空全部手工覆盖）」按钮一次性清空 `msc_overrides`，表回到全派生。
- FR-9：派生表**只**含 `campaign.RUNNER_CLI_DEFAULTS` 与形状/尺寸能推出来的字段；
  `model_dir` / `resume` 不在表里，由 ④ 与 ⑥ 单独给出，开跑前显式补齐——少了它们
  `execute_campaign` 会安静地取 click 默认（从零重训 + 重跑已完成段），两种都是
  "看起来在续跑、其实白跑几小时"。

### 2.3 ③ 计划预览

- FR-10：按当前形状集合列出 `campaign_matrix` 的行（形状 × 阶段 → 块数、位深、updates），
  合计 `矩阵 totals`（默认 16 段 / 112 块）。
- FR-11：给出时间与磁盘估算（`estimate_seconds` / `estimate_disk_gb`，含 `ESTIMATE_SPREAD`
  区间），并注明 **capture / fit / 重放不计入**该估算。
- FR-12：尺寸 → 采样窗半跨取 `SIZE_WINDOWS`：200 → 420、600 → 620、1200 → 900（px）；
  手工覆盖时表里标「手工」并记原值。

### 2.4 ④ 模型目录（自动匹配）

- FR-13：列出候选物理预测模型目录，默认选中最新一份（`index=1 if models else 0`，
  第 0 项是"从零训练"）。
- FR-14：可手填路径；填了就用填的，否则用下拉选中项。结果存 `msc_model`
  （`None` = 从零训练）。
- FR-15：选目录时不打开相机（只用 `Path` 与本地文件）。

### 2.5 ⑤ 运行前检查（六项，红停 + 黄勾选）

| key | 名称 | 判据 |
|---|---|---|
| `geometry` | 几何自洽 | 零级剔除圆盘（半径 `--zero-radius` = 60 px）∩ 目标方框 / 目标面积；采样窗判据：目标半边长 + `WINDOW_MARGIN_PX`（8 px）≤ `--window-span` |
| `disk` | 磁盘预算 | 计划产物 GB 与所在盘的可用空间 |
| `time` | 时间预算 | 估算秒数与设备占用预算（缺省 `BUDGET_HOURS = 8 h`，超只 `warn`） |
| `model` | 模型新鲜度 | 模型目录的产物时间与本次预期口径 |
| `saturation` | 饱和预估 | 历史 campaign 的 `peak / full_scale` 峰值比 |
| `exposure` | 曝光收敛史 | 历史 campaign 的曝光阶梯是否收敛 |

- FR-16：每项状态为 `pass` / `warn` / `stop` / `unknown` 之一，并附判据与出处（报告节号或
  常量名），让人能自己复核，不要求信任页面。
- FR-17：**缺信息一律 `unknown`，不静默通过**。`unknown` 计入黄项，需勾选才能开跑。
- FR-18：`stop`（红）是硬门禁；`warn` / `unknown`（黄）需逐条勾选 `msc_ack_<check>` 才放行。
- FR-19：检查卡**不碰** `verify_light`：那要活相机（`read_frame` + `set_exposure_ms`），
  塞进去就毁掉"没设备也能先看一眼"这条前提。活体判定由 runner 在开跑后给出，经 ⑥ 显示。
- FR-20：几何阈值 5 % / 15 % 取自规划报告 §2 勘误；三个实测锚点为
  s1200 → 0.7854 %、s600 → 3.1416 %、s200 → 5.6057 %（`preflight.check_zero_overlap` 解析计算，
  与报告同口径）。

### 2.6 ⑥ 运行 / 逐块进度

- FR-21：「开始 campaign」受**四重门禁**：检查卡无红停项、黄项已勾选、已连接 SLM、
  `bench_settings.bench_busy()` 为假；本页已有一份在跑时同样禁用。禁用原因逐条列出。
- FR-22：开跑前把参数快照算好（主线程），线程体只收普通参数：
  `out` / `kwargs` / `stop_event` / `out_q`，**不碰** `st.session_state`。
- FR-23：进度以"重读 campaign 目录"为主（`collect_campaign_progress` 容忍半成品：读到
  一半的 `manifest.json`、只写了块没写 `summary.json` 都不抛），刷新节奏
  `@st.fragment(run_every=2.0)`（一块 24–40 s，2 s 足够）。
- FR-24：队列只走三种终态消息：`("done", summary)`、`("error", payload)`、
  `("optical_path", {"reason","cause","report"})`。**先抽队列、再看 `stop_event`**：
  worker 的 `finally` 无论如何都会 set 它，顺序反过来会把"已完成"的那次回传跳过。
- FR-25：进度分母取**计划**（`campaign_matrix`/`matrix_totals`），不取读取器：读取器只
  数目录里已存在的段，跑到一半的目录会显示成"4/4 段已完成"。计划形状的优先级为
  journal 的 `plan.shapes` → 页面当前选择 → 目录里见到的形状。
- FR-26：段状态表列：形状 / 阶段 / 状态 / 块 / 段首 CV / 段末 CV / 核心峰值 / 曝光 ms；
  当前段的逐块表列：块 / CV / 饱和 px / 核心峰值 / 曝光 ms / 验收 / 说明。
- FR-27：`验收` 读 `summary.json` 的 `paired_decision`（`manifest.json` 里没有），由读取器
  按块号 join；页面**不**做第二遍 join。
- FR-28：「当前段」一行同时给首块/末块 CV、曝光、饱和，与报告 §3.2 口径一致——CV 必须
  带曝光与饱和一起读。块数分母取该阶段的**计划**块数（不能拿"已写下的块"当分母，否则
  这一行永远显示 `4/4`）。
- FR-29：两张表交给 `st.dataframe` 前，缺值（`MISSING` = `"—"`）必须转 `None`：同列混排
  float 与字符串会让 arrow 序列化抛 `ArrowInvalid`，整张表**不显示**（栈只进日志），
  而触发条件恰是"某段还没有块读数"——刚开始 / 被 `stopped` / 读到半截产物。
- FR-30：段状态透传：`complete` / `reused` 计为已完成，`running` / `stopped` / 任何新状态
  原样显示，读取器不造假块。

### 2.7 停止与断点续跑

- FR-31：「停止（当前块结束后）」只 set `stop_event`；单块（~24–40 s）是原子事务，不可
  中断。停止是**段粒度**协作停止，journal 记 `stopped`。
- FR-32：「从断点续跑」开关只在 `--out` 里已有 `journal.json` 时出现，默认勾上，并列出
  本次会跳过的段（口径 = `COMPLETE_STATUSES`，`stopped` 不算）。
- FR-33：该开关落到 runner 的 `--resume`：`manifest.json` 已存在的段不再进段、journal 记
  `reused`。**不带**它时已存在的段目录会抛 `FileExistsError`，所以页面写的是
  「取消勾选只适用于换一个新的 `--out`」，而不是「重跑一遍」。

### 2.8 光路无光（`OpticalPathInvalid`）

- FR-34：`OpticalPathInvalid`（`exit_code = 3`）在实验系统上是**预期**状态，不是缺陷。
- FR-35：库内不转 `SystemExit`：映射只发生在 CLI 边界（`runner.run()`）。`execute_campaign`
  让异常上抛，GUI worker 接住并转发 `(reason, cause, report())`。
- FR-36：⑥ 印 `report()` **原文**（含曝光阶梯实测数字，不重排、不摘要），并按 `cause` 给
  处置方向：`blocked`（峰值不随曝光变化）⇒ 先去看光路；`exposure_ceiling`（会话曝光上限
  把阶梯截短）⇒ 先调 `AO_DAHENG_MAX_EXPOSURE_MS`。两者处置方向相反。

### 2.9 ⑦ 历史会话

- FR-37：列出 `data/debug/slm_transfer/**` 下写有 `journal.json` 的目录（默认选第一个），
  用同一套段表 / 块表离线复盘。
- FR-38：目录被删过/挪过时先清 widget 键 `msc_history_choice`，否则 `selectbox` 会因
  state 不在 options 里抛异常。

---

## 3. 接口规范

### 3.1 页面 → runner 的参数快照

`_runner_kwargs(settings, slm_number, model_dir, resume)` = `settings.kwargs()` +
`slm_number` + `model_dir`（`Path | None`）+ `resume`（`bool`）。键名一律照 click 参数名
（`--model-dir` → `model_dir`）。`tests/ao_shaping/gui/slm/test_multishape_campaign_ui.py`
把这张快照的每个键对着 `slm-transfer-shapes` 的参数表核一遍，拼错一个就在测试期暴露，
而不是第一次真跑时抛 `TypeError`。

### 3.2 Session State

| 键 | 类型 | 含义 |
|---|---|---|
| `msc_side_preset` / `msc_side_custom` | int / float | 边长预设 / 自填 |
| `msc_overrides` | dict[str, Any] | 手工覆盖的参数（按参数名） |
| `msc_overrides_reset` | bool | 一次性重置信号（消费后清） |
| `msc_out_side` | float | `msc_out_text` 上次对应的边长（换尺寸才重写输出目录） |
| `msc_out_text` | str | 输出目录（默认随形状 + 尺寸生成） |
| `msc_model_choice` / `msc_model_custom` | str | 模型目录下拉 / 手填 |
| `msc_model` | str \| None | 生效的模型目录（None = 从零训练） |
| `msc_budget_hours` | float | 设备占用预算（h） |
| `msc_acks` | dict[str, bool] | 黄项勾选状态 |
| `msc_ack_<check>` | bool | 单个黄项的勾选框（六项之一） |
| `msc_resume` / `msc_resume_dir` | bool / str | 断点续跑开关 / 它对应的目录 |
| `msc_run` | `_CampaignRun` | 运行体（目录、线程、`stop_event`、`out_q`、`shapes`） |
| `msc_summary` / `msc_error` / `msc_optical_path` | dict | 三种终态回传 |
| `msc_history_choice` | str | ⑦ 选中的 campaign 目录 |
| `msc_start` / `msc_stop` / `msc_reset_overrides` | bool | 按钮状态键 |

---

## 4. 非功能需求

- NFR-1：**离线可用**。①–⑤ 与 ⑦ 在无设备时全部可渲染；未连接 SLM 时「开始」禁用并说明原因。
- NFR-2：线程安全。线程体不读写 `st.session_state`；进度靠重读目录，终态靠队列。
  违反这条只会在页面上表现为竞态，不会报错。
- NFR-3：不静默失败。缺信息 → `unknown`；表格缺值 → `None`（不是 `"—"`）；`st.dataframe`
  序列化失败不得只留日志。
- NFR-4：不重复实现。参数派生、进度读取、零级几何、形状标签、阶段配方各只有一份来源
  （`campaign` / `campaign_progress` / `preflight` / `TARGET_SHAPE_LABELS` / `stages`），
  本页只做渲染与编排。
- NFR-5：术语。文档与界面文案遵守 `docs/terminology_glossary.md`：`实验系统`（非"台架"）、
  `对比择优`（非 `bake-off`），不使用"实拍/真机"。

---

## 5. 本版明确不做

| 项 | 说明 |
|---|---|
| M 系列实验系统指纹比对 | 见 issue #214 的后续条目；当前只在 ⑤ 用历史 campaign 做饱和 / 曝光两项推断 |
| 独立的历史浏览器（跨 campaign 检索） | ⑦ 只做单目录复盘，检索仍靠 `report/` 与脚本 |
| `cost_table`（逐形状的价格/工时表） | 预算只给总时间与总磁盘 |
| 四工位并行 | 本页一次只跑一份 campaign；`bench_busy()` 是互斥判据 |

---

## 6. 验证

| 层次 | 证据 |
|---|---|
| 纯函数 | `tests/ao_shaping/tools/slm/transfer/test_campaign_settings.py`（派生 / 来源标签 / 覆盖 / 矩阵 / 预算 / 计划脚本同源）、`test_campaign_progress.py`、`test_zero_overlap.py` |
| 页面 | `tests/ao_shaping/gui/slm/test_multishape_campaign_ui.py`（含离线渲染七段、计划分母、形状标签兜底、光路无光横幅、arrow 缺值回归） |
| 检查卡 | `tests/ao_shaping/gui/slm/test_precheck.py`（六项 × 四种状态的判据与门禁） |
| runner | `tests/ao_shaping/runners/test_transfer_shapes_runner.py`（退出码 3、无 traceback、参数表对齐） |
| 真实产物 | `data/debug/slm_transfer/multishape_scale_20261028/s1200`：16 段 / 112 块、square 四段 CV 与 `peak_full`（255 / 3923 / 4095 / 3465）与报告 §3.2 逐位一致 |
