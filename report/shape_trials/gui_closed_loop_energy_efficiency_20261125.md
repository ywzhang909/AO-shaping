# GUI 闭环整形「完成整形后」能量利用率 vs SLM 衍射效率 (2026-11-25，2026-11-26 修订)

## 摘要

为实测闭环整形页面（`ClosedLoopShapingPage`）新增**完成态能量利用率报告**：闭环结束后，
在结果卡片中展示当前（最优）整形结果对 CCD 总能量的利用率，并与可选的 **SLM 衍射效率
参照**（探针 `summary.json` 实测值）对比，给出「目标窗口拿到了多少可用光」。全程只读、
不重新采集、不触发任何设备动作。

2026-11-26 按用户口径修订效率定义：**SLM 衍射效率 = 从平相位 0 级可以转移走的能量份额**
`η = E_blaze / F0`（`F0` 是平相位帧零级窗口能量）；不再用 `lobe.ratios[blaze_order]`
（其分母是闪耀帧三窗之和，会把效率抬高：同一探针 39.78% vs 19.27%）。同时用三组历史
数据算了倾斜与整形各自的利用率，结论见「历史数据计算」：倾斜在探针口径下 19.27%，
在同次效率匹配口径下 54.21%；整形在同次效率匹配口径下 **−12.46%**（目标方框内反而比
平相位少），该次判定 `undeterminable`，只能当趋势参考。

2026-11-26 追加 **A/B–ABBA 闭环实测判定**（详见对应章节）：在 0.12 ms 非饱和、同曝光口径下，
以 `paired_decision(metric="cv")` 对 `model640`(A) vs `refit_forward`(B) 做单变量配对，
四帧 A→B→B→A 全部未饱和，CV 改善 0.0623 > 漂移 0.0360，判 **`accepted=True`** → 采用
`refit_forward`。这是 CCD 实测闭环收益判定，与前面的历史利用率趋势参考相互独立。

## 任务项

| 项 | 状态 | 说明 |
|---|---|---|
| 后端纯函数 `shaping_efficiency_report` / `load_slm_diffraction_efficiency` | 完成 | `src/ao_shaping/gui/slm/closed_loop_shaping.py`，18 项测试锁定 |
| 2026-11-26 新效率公式 `flat_zero_transfer_fraction(energy, flat_zero_energy) = η` | 完成 | 同一模块，`__all__` 已登记；探针自带 `blaze_energy_share` 时直接采信并交叉校验 |
| 页面参照输入 + 结果卡片 + 完成态刷新 | 完成 | `closed_loop_shaping_page.py`，设计师车道实施，主 agent 复审文案与线程纪律 |
| 页面离线测试 | 完成 | `test_closed_loop_shaping_page.py`，夹具同步新 schema |
| 后端首次提交 | **完成** | `d1c80cf9`（本地 master，**未推送**），+224 行、0 删除，仅 2 文件 |
| 2026-11-26 公式修订 + 历史数据分析脚本 | **未提交** | 公式 2 文件 + 新分析脚本 + 本报告；等页面持有车道提交后再一并处理 |
| 页面提交 | **挂起** | 与并行会话改动同文件 hunk 交错，整文件让渡给持有车道（见「提交状态」） |
| 报告溯源登记 | **挂起** | `scripts/_common/provenance.py` 被另一车道整体 CRLF 改写，按整文件让渡规程不碰 |
| A/B–ABBA 闭环实测判定（CCD） | **完成** | `model640`(A) vs `refit_forward`(B)，`paired_decision(cv)` 判定 `accepted=True`，见「A/B–ABBA 闭环实测判定」 |

## 指标定义（2026-11-26 修订）

- **能量利用率** = 最优轮 `window_fraction`：目标窗口能量 / **CCD 全帧**能量。
  直接复用该轮已落账的诊断量，**不重算、不重新采集**。
- **最优轮**：按 `LoopSummary.best_score` 匹配（对应 `best_phase`）；无 best_score 时
  退化为实测分数最低且带 `window_fraction` 的轮次；无任何有效轮次时两个值均为
  `None`，页面显示占位文案而非 0%。
- **SLM 衍射效率（新定义）** = 从平相位 0 级能转移走的能量份额：

  ```
  η = E_blaze / F0
  F0 = relative_efficiency.baseline_zero_window_counts   # 平相位帧零级窗口暗减积分
  E_blaze = lobe.energies[闪耀级]                          # 闪耀帧闪耀级窗口暗减积分
  ```

  探针自带 `relative_efficiency.blaze_energy_share` 时直接采信（同一公式，探针自己算的），
  并仍用上式校验基准；缺失时按上式复算。闪耀级取自
  `relative_efficiency.blaze_order`（新）/ `lobe.orders.blaze_order`（旧），两者都缺时取
  ±1 级中较强者。**不使用** `lobe.ratios[blaze_order]`（分母是闪耀后三窗之和 S₃）。
  效率非正、> 1 或缺基准时直接报错，页面宽泛捕获，只丢对比项、不丢利用率。
- **相对 SLM 效率** = 利用率 ÷ 衍射效率。⚠ 两者分母不同（**CCD 全帧** vs **平相位零级
  窗口**），只作粗对照，**不设硬判定**，可在页面文案中注明口径。

## 历史数据计算：倾斜与整形分别能利用多少

新增只读脚本 `scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py`，
不碰设备、不改数据，重跑命令：

```
.venv\Scripts\python.exe scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py
```

### 1. 倾斜（闪耀光栅）：探针口径 `η = E_blaze / F0`

| 探针 | 有效曝光 | 级次间距 | F0 (counts) | E_blaze (counts) | η (F0 口径) | 零级残余/F0 | 三窗未计入/F0 | 旧口径 `ratios` |
|---|---|---|---|---|---|---|---|---|
| `probe_real_20261123_gate`（当前参照） | 0.02 ms | 471 px | 16349.7 | 3151.3 | **19.27%** | 29.18% | 51.54% | 39.78% |
| `probe_real_20261121` | 5 ms（两帧峰值均 255，饱和） | 81 px | 291215.5 | 4735.5 | 1.63% | 76.00% | 20.96% | 2.06% |
| `probe_real_20261121_0p1ms` | 0.1 ms（±1 级能量全为 0） | 81 px | 32172.9 | 0.0 | 0.00% | 28.58% | 71.42% | 0.00% |

- 页面参照值取 `probe_real_20261123_gate`：**η = 19.27%**（闪耀级 `minus_one`，灰度 993）；
  这与既有报告 `report/slm/diffraction_efficiency/hw_blaze_20261023/report.md` 补充节的
  `blaze_energy_share = 19.27%` 完全一致。
- 19.27% 应视为**下界**：窗口半径 19.5 px 对 FWHM 13 px 的光斑存在截断，三窗合计仅覆盖
  F0 的 48.45%，其余 51.54% 落在窗口之外（高阶、散射，以及 0.02 ms 低信号下的量化）。
- 另外两份探针不可用：`probe_real_20261121` 两帧都顶到满量程，`0.1ms` 那份没有形成闪耀
  级。它们说明**曝光/灰度契约没对齐时该口径不成立**，不能拿来横比。

### 2. 同一实验内的倾斜 vs 整形（`efficiency_match`，12-bit，曝光 0.04 ms）

数据：`data/debug/efficiency_match/20261123_050735Z/analysis.json`；口径来自
`scripts/slm/efficiency_match_analyze_20261123.py`：
`η_d = G1/F0`（倾斜，R1=60 px 圆盘），`η_s = (S − S0)/F0`（整形，方框 61.2 px，
S0 为平相位同框积分），`F0` 为平相位 R0=48 px 圆盘积分；级次间距 59.1 px。

| 轮 | η_d = G1/F0（倾斜） | η_s = (S−S0)/F0（整形） | d = η_s − η_d |
|---|---|---|---|
| 1:blaze#0 / 1:shaping#0 | 54.01% | −10.25% | −0.6426 |
| 1:blaze#1 / 1:shaping#1 | 55.28% | −10.82% | −0.6610 |
| 2:blaze#0 / 2:shaping#0 | 53.02% | −10.48% | −0.6350 |
| 2:blaze#1 / 2:shaping#1 | 54.52% | −18.30% | −0.7282 |
| **均值** | **54.21%**（sd 0.0106） | **−12.46%**（sd 0.0552） | −0.6667（CI95 [−0.7068, −0.6388]） |

- 倾斜在这套口径下可把平相位零级圆盘能量的 **54.21%** 送到 +1 级（镜像侧泄漏约
  0.86×，见 `EM_REPORT.md`），是本次唯一确认「搬得动」的通道。
- 整形相位（来源是 `differentiable_opt/20261117_022658/phase_final.npy` 的最近邻块映射）
  在平相位零级位置方框内只留下 F0 的 **84.81%**（逐帧 87.02/86.46/86.79/78.98%），
  而平相位同框是 **97.27%**；两者之差即 η_s = −12.46%。**整形没有净增益**：能量被搬出
  方框，但方框内结果比平相位还少，不能把 84.81% 当成「转移到整形图案的份额」。
- 判定为 `undeterminable`，原因：`out_of_range_energy_check_background_or_saturation`
  （三窗能量和越界）、`g1_group_sd_exceeds_limit`（G1 两轮组内离散 > 0.03；另外 η_s 的
  sd 0.0552 也超 0.03，η_d 的 sd 0.0106 则通过）、`drift_rounds_gt_1`（4 轮 > 1）。
  F0 = 649699.2 counts (sd 0.0) 只来自 6 帧平场中**唯一未被漂移标记的 1 帧**，质心漂移
  3.12 / 2.47 px；因此上述百分比只能当趋势，不做验收。

### 3. 方形整形扫描 `pib`（整帧口径，2026-10-23）

| 批次 | side_px | pib 范围 | 饱和帧 |
|---|---|---|---|
| `20261023_gs_square`（20 ms） | 100–1800 | 43.53%–96.88% | 1/18（side=100） |
| `20261023_atomic_full`（180 ms） | 100–1700 | 6.70%–95.75% | 17/17 |

`pib` = 方框内能量 / **整帧**能量，方框中心与零级重合；它只说明「整帧能量有多少落进零级
附近的方框」，**不是**「从 0 级转移到图案的比例」，且 180 ms 批全部饱和，不可跨曝光横比。

### 4. 口径结论

| 路径 | 分母 | 历史值 | 能否直接互比 |
|---|---|---|---|
| 倾斜（探针窗口） | 平相位零级窗口 F0 | **19.27%** | 页面参照值；窗口截断，下界 |
| 倾斜（效率匹配圆盘） | 平相位 R0=48 px 圆盘 | **54.21%** | 与上者分母不同，不可相减 |
| 整形（效率匹配方框） | 同一 F0 圆盘 | **−12.46%** | 与上者**同一次实验同分母**，可直接对比 |
| 闭环 GUI `window_fraction` / `pib` | CCD 整帧 | 无历史落账 | 与三者都不可相除 |

一句话：**在同一份效率匹配数据里，倾斜能搬走约 54.2% 的平相位零级能量，整形反而比平相位
少 12.5 个百分点；探针口径下倾斜的可转移份额是 19.27%（下界）。**闭环页面的利用率与这些
效率分母不同，卡片里的「相对 SLM 效率」只作粗对照。

## A/B–ABBA 闭环实测判定（CCD，2026-11-26）

前面「历史数据计算」只是趋势参考，**不构成 `model640` vs `refit_forward` 的闭环收益判定**。
按冻结口径做单变量配对 A/B–ABBA 实测，A 臂 = `model640` 闭环 best
（`ab_validate_base/square_best.npy`），B 臂 = `refit_forward`
（`ab_validate_refit/square_best.npy`），同一设备会话内按 **A → B → B → A** 依次回放相位、
CCD 各采一帧，以交错序列抵消漂移。主判据为
`paired_decision(metric="cv")`，容差严格用默认 `rel_tol=0, abs_tol=0`；
`rel_l2` / `ee_roi` 只报告、不参与判定。

**设备占用与释放**：独占 **SLM #1 + Daheng MER2-507-23GM (NIR) CCD**，段间物理冷却 60 s
（覆盖原 ≥200 s 规则，用户确认）。回放外壳 `replay_closed_loop_abba_20261124.py` 经
`device_lock.json` 加锁，19:59:27 开相机/19:59:28 开 SLM，19:59:35 四帧采完，`finally` 释放；
`released.json` 记 `camera_closed=true`、`slm_closed=true`、
`released_at=2026-11-26T11:59:35+00:00`。当前无设备进程占用。

**曝光有效性**：首轮用驱动默认 20 ms，四帧 `peak=4095` 全饱和，判定作废；根因是原脚本 CLI
无曝光入口。给回放脚本补 `--exposure-ms`，经 `set_camera_exposure_ms` 在 `cam.open()` 后显式
下发并回读生效值。重跑 `summary.json` 记 `exposure_ms_requested=0.12`、
`exposure_ms_effective=0.12`，四帧 `peak=2019–2138`（12-bit 满量程 4095，未饱和），判定有效。

四帧实测（camera-pixel core mask `abba_core_mask_camera.npz`，shape `(1944,2592)`、1369 个
True，与 analysis-grid decimated CV 1.2299 对齐）：

| 级 | 臂 | CV | mean | peak | saturated |
|---|---|---|---|---|---|
| 0 | A | 1.897016 | 150.38 | 2059 | false |
| 1 | B | 1.862118 | 151.28 | 2091 | false |
| 2 | B | 1.843373 | 148.22 | 2019 | false |
| 3 | A | 1.932998 | 152.11 | 2138 | false |

`paired_decision(metric="cv", rel_tol=0, abs_tol=0)`（A 为基线、B 为候选，CV 越低越均匀）：

| 量 | 值 | 含义 |
|---|---|---|
| baseline（A1/A2 均值） | 1.915007 | `model640` 闭环 |
| candidate（B1/B2 均值） | 1.852745 | `refit_forward` |
| drift | 0.035982 | A 两帧自身差，交错抵消后残差 |
| tolerance | 0.0 | 严格默认容差，未放水 |
| required = baseline − candidate | 0.062261 | B 相对 A 的 CV 改善量 |
| drift + tolerance | 0.035982 | 需超过的门槛 |
| **accepted** | **True** | 0.062261 > 0.035982，改善超出漂移 |

**结论**：在 0.12 ms 非饱和、同曝光口径下，`refit_forward` 闭环的 core CV（1.853）低于
`model640`（1.915），改善 0.0623 大于 ABBA 漂移 0.0360，`paired_decision(cv)` 判
**`accepted=True`** → 采用 `refit_forward`。证据真源：
`data/debug/slm_transfer/ab_session_20261124/ab_replay_square_retry2/{summary.json,released.json,device_lock.json}`
及 `captures/{0_A,1_B,2_B,3_A}.npz`。注意 CV 衡量的是光斑均匀度，非能量利用率；本判定回答
「哪个闭环更均匀」，不改写前面利用率/效率的历史趋势结论。

## 文件改动

| 文件 | 车道 | 内容 |
|---|---|---|
| `src/ao_shaping/gui/slm/closed_loop_shaping.py` | 本任务 | `import json`、`__all__` 三项、三个纯函数（含新 η 公式） |
| `tests/ao_shaping/gui/slm/test_closed_loop_efficiency_report.py` | 本任务 | 新建，18 用例（公式/负分子/非法基准/两版 schema/自动选级/>1 拒绝/选轮/对比） |
| `src/ao_shaping/gui/slm/closed_loop_shaping_page.py` | 本任务（设计师） | 参照输入行 + `_build_efficiency_result` 卡片 + `_on_finished` 里 `_show_shaping_efficiency` + 失败/清场重置 |
| `tests/ao_shaping/gui/slm/test_closed_loop_shaping_page.py` | 本任务（设计师） | 离屏用例；`_probe_summary` 夹具同步新 schema |
| `scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py` | 本任务 | 只读历史数据复算三口径（新） |
| `scripts/slm/replay_closed_loop_abba_20261124.py` | 本任务 | 补 `--exposure-ms` 显式曝光入口（下发 + 回读写 summary），修复默认 20ms 饱和 |
| 本报告 | 本任务 | 定义修订 + 历史计算 + A/B–ABBA 闭环实测判定 |

## 验证结果（离线，未占设备）

```
.venv\Scripts\python.exe -m pytest tests\ao_shaping\gui\slm\test_closed_loop_efficiency_report.py \
    tests\ao_shaping\gui\slm\test_closed_loop_shaping_page.py -q --import-mode=importlib
→ 38 passed（18 后端 + 20 页面）

.venv\Scripts\python.exe -m ruff check <4 个代码文件>                     → All checks passed
BOM 检查                                                                   → 5 个文件均 no-bom
.venv\Scripts\python.exe scripts/_common/term_unify_20261016.py --dry-run → 本任务文件 0 命中
```

此前基线（公式修订前）另有 `test_closed_loop_shaping.py` 33 passed，确认新增未破坏既有语义；
本次公式改动限于纯函数与探针加载，页面夹具已同步，待页面车道合入后须重跑一次。

ABBA 回放外壳补丁（离线门禁）与实测判读：

```
.venv\Scripts\python.exe -m ruff check scripts\slm\replay_closed_loop_abba_20261124.py → All checks passed
.venv\Scripts\python.exe -m py_compile scripts\slm\replay_closed_loop_abba_20261124.py → exit 0
BOM 检查（ABBA 脚本）                                                              → no-bom, LF
无 --execute 干跑                                                                  → 拒绝开设备 (exit 2)
实跑判读 C:\Users\zhangh\AppData\Local\Temp\opencode\abba_decide.py             → accepted=True, 四帧 unsaturated
```

## 提交状态：首次后端已提交，公式修订与页面让渡待处理

2026-11-26 追加：本报告、效率复算脚本、A/B–ABBA 判定章节以及 ABBA 回放脚本的
`--exposure-ms` 补丁已纳入本车道提交（见下方 git 记录）；A/B 判定证据落在
`data/debug/slm_transfer/ab_session_20261124/ab_replay_square_retry2/`（数据目录，
不入库）。溯源登记与页面改动仍按整文件让渡规程等对方车道落地后处理。

实施期间共享工作树中另一会话正在对同一模块族做大改（移除「目标保持 / target-lock」
功能：`closed_loop_shaping.py` 约 410 行删除、页文件重写、`test_closed_loop_shaping.py`
修改、`test_drift_feedback.py` / `test_target_lock*.py` 删除）。本任务的 hunk 与其交错。

**首次后端已提交 `d1c80cf9`**（本地 master，未推送）：模块内本任务的 3 处改动均为纯插入
且 HEAD 锚点唯一，按 playbook 用「HEAD 行 + 本线 hunk → `git hash-object -w` →
`git update-index --cacheinfo` → 普通 `git commit`（不带路径限定）」方式拆分提交。
提交内容 +224 / −0，仅 2 文件；`grid_spacing_px` 等对方将删的符号在提交中保持原样，
工作树中对方车道的改动全数未动。

```
git show d1c80cf9 --stat
 src/ao_shaping/gui/slm/closed_loop_shaping.py      |  78 ++++++++++++
 .../gui/slm/test_closed_loop_efficiency_report.py  | 146 +++++++++++++++++++++
 2 files changed, 224 insertions(+)
```

**2026-11-26 公式修订未提交**：只涉及 `closed_loop_shaping.py`（新定义/采信探针值/
docstring 口径说明）与 `test_closed_loop_efficiency_report.py`（新增 2 用例）；新分析脚本与
本报告为未跟踪/待提交文件；`scripts/_common/provenance.py` 被另一车道整体 CRLF 改写，
本任务按整文件让渡规程不碰，待其提交后补登记。

**页面未提交**：`closed_loop_shaping_page.py` 被对方车道**整体重写**，本任务的 UI 改动
叠在其新结构上，hunk 无法干净分离，按 §3 整文件让渡给持有车道；页面测试
`test_closed_loop_shaping_page.py` 同理（本轮仅同步了夹具）。

### 过程中的坑（供后续车道复用）

- `git commit -- <paths>` 提交的是**工作树**内容而非索引，会把对方 532 行删除裹挟进去
  （本次误发为 `e4b2d2d3`，已 `git reset --soft HEAD~1` 回退并重提，未推送）。索引拆分
  后必须用**不带路径限定**的普通 `git commit`。
- 用 `git hash-object -w` 构造 blob 时须字节级对齐：`import json` 的插入位置（空行后、
  `import threading` 前）、函数块后的 2 个 PEP8 空行都要与 HEAD/工作树一致，否则
  「工作树 vs 索引」diff 会出现假 hunk。判据：`git diff -U0 -- <file>` 中不再出现本线
  新增符号即为干净。
- PowerShell `git show ... > file` 写出的是 UTF-16，会让 `py_compile` 误报；须用 Python
  `open(..., 'wb').write(stdout)` 落字节。

### 快照与备份

- 安全快照（未触碰工作树）：`git stash create` = `053ae1326d22dc6c36d786aadeb07ce6e3431c22`
  - `053ae132:src/ao_shaping/gui/slm/closed_loop_shaping.py` = `43e7d627`
  - `053ae132:src/ao_shaping/gui/slm/closed_loop_shaping_page.py` = `3527063d`
- 新测试备份：`C:\Users\zhangh\AppData\Local\Temp\opencode\test_closed_loop_efficiency_report.py.bak`
  （SHA256 `11B56C89…C27`；该文件已随 `d1c80cf9` 入库）

**后续**：等目标保持移除车道提交后，把本任务的页面改动（参照输入行 + `_build_efficiency_result`
+ `_show_shaping_efficiency` / `_render_efficiency_report` / `_read_efficiency_reference` /
`_reset_efficiency_result` + 页面用例）合入其新结构，再跑上述同一组测试复核。

## 参数与阈值

- 参照文件默认**空**：留空时只显示能量利用率，不做对比。
- 参照路径由操作员用浏览按钮选择探针 `summary.json`（样例：
  `data/diffraction_efficiency/probe_real_20261123_gate/summary.json`，效率 ≈ 0.1927，
  闪耀级 `minus_one`，灰度 993）。
- 完成态与开跑前（`_clear_rounds` / `_on_failed`）都会整段重写结果区，两场连跑不残留
  上一场数字。
