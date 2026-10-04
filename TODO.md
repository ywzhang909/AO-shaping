# TODO — AO-Shaping 待办总账

> 生成时间：**2026-10-01**（扫描 + 逐项验证）
> 扫描范围：`docs/**/*.md`、`src/**`、`scripts/**`、`tests/**` 中的 `TODO.md`、
> `TODO/FIXME/XXX/HACK/待办/未实现/待确认` 代码注释、以及各报告文档末尾的"下一步/建议"。
> **每项都已对照当前代码核实**，确认仍存在才收录；已修复项移到 §5。
> 排除：`.kilo/worktrees/`、`.venv/`、`libs/OOPAO`、`drivers/ccd/_miicam_sdk`（厂商 SDK）。

> **增量更新 2026-10-02**（ABBA 漂移对消落地）：新增 H-19（ABBA 待实机验收）、
> 改写 H-9（方形路径 ABBA 现已有可复用参考实现）、**推翻并重写 H-14**（1000× 全扫描
> 证明无任何 delta 收敛，瓶颈是慢漂移而非 δ）、§5 新增 ABBA 一行。其余条目未复核。

> **增量更新 2026-10-03**（第 1 批、第 2 批 + 第 3 批的 R-20/R-26 落地，见 §5）：
> R-23/R-24/R-30/R-31/R-40/F-2/F-6/F-7/F-8、**R-1~R-4 / R-9~R-11 / R-35**、
> 以及第 3 批的 **R-20（硬阻断）/ R-21 / R-22 / R-25 / R-26** 已完成；**全部经变异测试验证可捕获回归**，且
> `git stash` 暂存全部改动后复跑确认 **23 个既有失败一个不多一个不少**（§5.10）。
> 期间发现三处 TODO 未记的问题（已修，见 §5.2 与 §4）：
> ① `slm_zernike_shaping.py` 是 `slm_zernike_pib.py` 的**同源副本**，R-1~R-4 四条
> bug **两边都有**，原条目只记了一边；
> ② `recorder` 列不对称不止 `w_ee`/`ee_term`，`optimizer` 列同样只记在 `_row0`；
> ③ `slm_zernike_pib.py` 与 `slm_zernike_shaping.py` 的同名常量
> `ZERNIKE_APERTURE_RADIUS` **值不同**（300 vs 600）⇒ 新增 **X-3**。
> ④ 工作区的 `src/ml/zernike/models.py` 存在**语法损坏**，打挂整条
> `import ao_shaping.runners` 导入链（已修，但建议单独提交复核）。

## 来源清单

| 来源文档 | 提出日期 | 主题 | 处置 |
|---|---|---|---|
| `docs/TODO.md` | 2026-09-25 | `slm_zernike_pib.py` P0–P3 | → §1 / §2 |
| `docs/refactor/TODO.md` | 2026-10-01 | `utils/`+`scripts/`+`tools/` 架构重构 | → §2 |
| `src/ao_shaping/tools/slm/TODO.md` | 2026-10-01 | `tools/slm/` 22 个探针 | → §1 / §3 |
| `docs/fouriergsnet_pipeline/TODO.md` | 2026-10-01 | FourierGSNet 真机验证 | → §1 |
| `docs/fouriergsnet_pipeline/hardware_run_20261001.md` | 2026-10-01 | 真机跑出的 10 项代码级问题 | → §1 |
| `docs/slm/report2.md` §4 | 2026-09-16 | Zernike 矫正后续动作 | → §1 |
| `docs/slm_pib_bench/report.md` | 2026-09-30 | slm-pib 台架下一步 | → §1 |
| `docs/slm/bench_calibration_20261001.md` | 2026-10-01 | 稳定化判据 | → §1 |
| 代码注释 ×4 | 2026-09-17 ~ 09-25 | runner 待测清单 | → §1 / §3 |
| `docs/issues_report.md` §10/§11 | 2026-05-26 | 架构建议 + 新增扫描 | → §3 |
| `OBJECTIVE_TARGET_SHAPE_MERGE_PLAN.md` | 2026-09-26 | objective/target_shape 合并 | → §5（已落地） |

---

## 1. 需硬件（设备在线才能推进）

### 1.1 runner 待上设备实测清单

| # | 项 | 位置 | 提出 |
|---|---|---|---|
| H-1 | `dm-matrix` sequential/hadamard 双模式标定内核离线通过（122 passed）但**尚未上设备实测**。待跑：sequential 重跑、hadamard 自动序/显式序、闭环验证、report3 | `runners/dm_matrix_runner.py:12,72` | 2026-09-17 |
| H-2 | `zernike-matrix` 2026-09-16 重写后的标定内核**尚未上设备实测**。待跑：重标矩阵、离线矫正 RMS >20%、`export_correction_csv` 产物、新矩阵闭环、`--n-magnitudes` 线性度 | `runners/slm/zernike_matrix_runner.py:52,1169` | 2026-09-18 |
| H-19 | **ABBA 漂移对消已落地但未实机验收。** 4 个提交（`e9d1769` 实现 / `139a461` CLI / `796d3ce` 回归测试 / `1d6329e` 文档），离线全绿（`wfless` 414 passed、ruff 干净、LSP 无诊断、CLI→config→采样符号端到端断言通过、变异测试证明能捕获回归）。**尚未上设备**：`slm-pib spgd --abba-sampling`。待跑 A/B（同 `delta`/`seed`/`n_max=9`，ABBA 开 vs 关）判据 `dec>0.55` 且 `late_gain≥10%`。⚠️ 注意 `delta=0.0005` 无梯度信号（见 H-14），**别在该档验收** —— 应在 `0.05` 这类有信号但被漂移污染的档位做对照。2026-10-02 尝试时 Daheng `get_cam_list()` 返回 `[]`（设备离线） | `slm_zernike_pib.py:271,511,1355` | 2026-10-02 |

### 1.2 FourierGSNet 真机验证（`docs/fouriergsnet_pipeline/`）

| # | 项 | 提出 |
|---|---|---|
| H-3 | §0 前置条件：设备在线 + 激光开 + **无残留 `python*` 进程** + 看门狗硬超时（SLM memory-mode `open()` 与首次 capture 是挂起高危点） | 2026-10-01 |
| H-4 | §2 smoke test 结果区仍 `_待填_`（5 步：Daheng 枚举 → 开相机采帧 → 开 SLM #1 → SLM 开后再采帧 → 关设备，末尾须打印 `SMOKE_OK`） | 2026-10-01 |
| H-5 | §3 DM/WFS 实际 IP 与网段（上次 `192.168.0.101–126` 全段不可达），逐个 ping 并记录 | 2026-10-01 |
| H-6 | §4 `slm-gsnet spgd` 真机光路验证验收（0 级 `argmax` 定位、框内能量与 CV 可测改善、产物落盘） | 2026-10-01 |

### 1.3 真机跑出的代码级问题（修完才能复测）

> 2026-10-01 真机 `slm-gsnet spgd` 实测结论是**随机游走**（`dec=0.487`，末态比初态差 35.9%，
> EE 流失 6 倍，0 阶峰值 225→17）。以下 6 项**必须先改**再复测。
> 来源：`hardware_run_20261001.md` §5–6。

| # | 项 | 位置 | 提出 |
|---|---|---|---|
| H-7 | 方形路径**无 encircled-energy guard**，任何 `w_efficiency` 权重都会被"散光换 CV"击败 → 复用 `slm-pib` 的 `max_roi_energy_loss` | `slm_square_shaping.py` `ObjectiveParamsSquare` | 2026-10-01 |
| H-8 | 随机初始化是满幅 `uniform(-π,π)`，与 docstring 声称的 "small init" **矛盾**，且满幅起点 EE 反而最高 → 改平场起步或 ±0.1 rad | `slm_gsnet_runner.py:1223` | 2026-10-01 |
| H-9 | 方形路径**无 ABBA / 漂移对消**。单帧采样 `CAM_SAMPLE_ITER = 1`，三个采集点 `:1685`（单次评估）、`:1841`/`:1867`（`+`/`-` 差分）全是相邻两帧。**2026-10-02 起本项成本大幅下降**：`slm-pib` 路径已落地同款回文采样（`_spgd_capture_signs` @ `slm_zernike_pib.py:271`、`abba_sampling` @ `:511`、循环重写 @ `:1355`），语义与门控处理可直接照搬；台架探针 `slm_snr_probe.abba_signal` 是更早的参考实现 | `slm_square_shaping.py:136,1685,1841,1867` | 2026-10-01（2026-10-02 补参考实现） |
| H-10 | `--exposure_time_ms` 默认 **80.0 ms**，本台架近饱和基线 ~0.02 ms → 默认值必然饱和。且传该参数会**关闭自动曝光**，使饱和保护成死代码（仅 `exposure_time_ms == 0` 触发） | `runner_common.py:1154` | 2026-10-01 |
| H-11 | 焦距标定常数自相矛盾：文档同写 `132940/P` 与 `7600/P`（差 17.5×）；实测 7400 与一阶 `7557/P` 仅差 1–2% | `slm_bench_probe.py:78` / 文档 | 2026-10-01 |
| H-12 | `lr=0` 时 `learning_schedule()` **静默覆盖 `--delta`**；`_param_scale` freeform=1.0 而 Zernike=0.1，故 Zernike 调好的 δ 不通用；"freeform per-pixel" 名不副实（`np.kron` 分块，24×24 → 80×50 px/block） | `slm_square_shaping.py:1596-1605,1163/1170`、`_freeform_phase_radians` | 2026-10-01 |
| H-13 | 可达目标需重新定义：本台架 80 px staircase 只能影响**大尺度**结构（≥100 px 大 ROI、压低斑径、提 Strehl），刻 50 px 平顶方块不可达 | `hardware_run_20261001.md` §6.5 | 2026-10-01 |
| H-14 | 🔴 **原结论「可用区间 ≈0.1」已被 1000× 全扫描推翻（2026-10-02）。** 实机 8 档 `0.0005 / 0.001 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5`（`n_max=9`，200 epochs，`pearson`，`lr=0.5`，320px 窗，1.2ms 曝光，每档带 `--debug`）**无一收敛**，`recommended=None`；全部 `dec` 落在 0.43–0.56 ≈ 随机游走。逐档：`0.0005` 完全无移动（`first==final==0.5308`，信号在噪声底下，漂移对消也救不回）、`0.001` `late +0.1%`、`0.02` `dec 0.56 / late +8.5%`、`0.05` `dec 0.53 / late +21.5%`、`0.1` `dec 0.43 / late −16.7% / guard 31.4%`、`0.2` `dec 0.52 / guard 0%`、`0.3` `guard 0.7%`、`0.5` `guard 84.1%`。⇒ **瓶颈不是 δ 而是慢漂移污染 `J(+d)−J(−d)`**，先走 H-19 / H-9 的漂移对消路线，再回来扫 δ。⚠️ 旧记录「`0.2` 触发亮度折叠门」**与实测矛盾**（`0.2` guard 实为 0%，真正折叠的是 `0.1`(31.4%) 与 `0.5`(84.1%)），该条系单点观测误判 | `docs/slm_pib_bench/delta_scan.md`、`report.md` §6 | 2026-09-30（2026-10-02 修订） |
| H-15 | 驱动级 `get_camera_exposure_ms` 回读恒为 3.0、`auto_exposure` settle 滞后 | `drivers/ccd/common.py` | 2026-09-30 |

### 1.4 标定常数与物理量待复核

| # | 项 | 提出 |
|---|---|---|
| H-16 | CCD 像元尺寸权威值 **2.2 µm**（用户确认），此前按 datasheet 假设 4.8 µm → 2f 衍射尺度 K 应由 **5021 改为 10954**（×2.18）。受影响：`test_slm_diagnose.py:_DIFFRACTION_SCALE_PX`（当前仍断言 `5021.0`）、`test_slm_calibration_windowed.py:K`、AGENTS.md 偏移表（P64 78→170px, P32 157→342px, P96 52→114px, P40 126→274px）。**整形目标函数/靶框/束腰是像素域计算，不受影响。** 待硬件光栅偏移实测复核后统一更新 | 2026-09-30 |
| H-17 | 用 `zernike-matrix` 重标响应矩阵（旧矩阵在归一化下测得，幅度维度无意义）；查 §3.3 非线性（`phase_gray.npy` 对比请求 vs 实屏）；多轮迭代改 `leak>0` 或降 `gain` 抑过冲；提高 `n_avg` 覆盖大幅度段；波前矫正 CSV 换 532nm 版本（当前 520nm） | 2026-09-16 |
| H-18 | SLM 稳定化**不能用固定等待**：`SLM_RESPONSE_TIME_S = 0.3` 是驱动自报值，实测面板弛豫远慢于此 → 改用"连续两次读数一致"判据（`slm_gs_refine` 已有 `settle-wait-s`）。实测噪声 0.67 counts RMS/px | 2026-10-01 |

---

## 2. 离线代码重构（无需硬件，但需先补特征测试）

### 2.1 `slm_zernike_pib.py` —— 行为 bug 优先（源自 `docs/TODO.md`，2026-09-25）

> ⚠️ 该文件已从 2179 行重构到 **1592 行**，`calc_objective` 已抽成
> `ShapingObjective.__call__`。**原文档行号全部失效**，但 bug 本身逐条存活。
> 🔴 **`slm_zernike_shaping.py` 是本文件的同源副本**（同样 1600 行量级、同样
> `_row0`/`_log_row`/`learning_schedule`/饱和分支结构）。**改本节任何一条都必须同时改
> 副本**，否则测试可能只覆盖一边 —— R-1~R-4 四条 P0 实测**两边都有**。

| # | 项 | 当前证据 | 提出 |
|---|---|---|---|
| R-5 P1 | 目标函数 if/elif 链 → Objective 类族 + `GuardedObjective`（闭包已提取成函数名，仍是 if 分发 + `objective_mode` 符号 + `last_terms` nonlocal） | `slm_zernike_pib.py:1300-1368` | 2026-09-25 |
| R-6 P1 | 两个搜索分支共享 `BenchSession`（"clip→相位→display→sleep→采图→饱和→算目标" 序列重复，饱和策略还不一致） | SPGD ~`:1419` / heuristic `:1190` | 2026-09-25 |
| R-7 P1 | 巨型函数拆编排器：~900 行 → `prepare_geometry` 返回**不可变 dataclass**（字段区分 `window_center_full_frame` / `reference_center_window_local`）+ `_run_spgd`/`_run_heuristic` | `slm_zernike_pib.py:760-1700` | 2026-09-25 |
| R-8 P1 | 硬件安全 try/finally：任何异常（相机掉线、越界、KeyboardInterrupt）都让 SLM 停在随机相位 | `slm_zernike_pib.py:1060` | 2026-09-25 |
| R-9 P2 | 300px 光阑踩坑文档**挂错常量** | ✅ **已完成**（`pib` 早已正确）。实测：note 现正确挂在 `ZERNIKE_APERTURE_RADIUS = 300.0`（`slm_zernike_pib.py:195`）之后。**本轮只修了 `slm_zernike_shaping.py`** —— 那里 docstring 是 `TARGET_BOX_WAIST_FACTOR` 之后的**裸字符串**（不是 docstring、不可达），且与 `:186-188` 的注释**互相矛盾**（一个说该用 600、一个说 600 会让修正静默失效且是硬件实测）→ §5.20 | 2026-09-25 |
| R-10 P2 | 死代码 `gauss_center`（零生产调用，可删） | ✅ **早已完成**（本轮实测确认）：全仓 grep 只剩 `TODO.md` / `docs/TODO.md` 的记录行，`src/` 无定义、无调用；专属测试也已删（`test_slm_zernike_pib_shape.py` 里剩下的 `"gaussian"` 是 target_shape 取值，无关） | 2026-09-25 |
| R-11 P2 | ~~常量替换字面量~~ | ✅ **早已完成**（本轮实测确认）：两侧都有 `ZERNIKE_CLIP = 5.0` 与 `IMPROVE_EPS = 1e-4`，全文件再无裸 `5.0` / `1e-4` clip 字面量（除常量定义自身）。⚠️ 顺带纠正：`GUARD_PENALTY` **不在** `drivers/ccd/common.py`，而在 `utils/image/target/objective.py:33`（随 ObjectiveSpec 抽取时搬的）；且 `metrics.py:298,384,396` 另有 3 处裸 `1e3` **语义不同、故意不统一** → §5.21 | 2026-09-25 |
| R-12 P2 | `_update_dynamic_weights` → `AdaptiveWeights` dataclass | ✅ **原定理由已不存在**（§5.27）：该函数**早已搬进共享叶子** `utils/image/target/objective.py:36`，两个引擎各自 import **同一份**（`slm_zernike_pib.py:93`），**副本漂移的风险已经消失**。剩下的裸 dict `setdefault` + 2/3-tuple 联合返回是**已测试的既定契约**（`test_slm_zernike_rms_pib.py:128+`，且 docstring 写明两元组分支"byte-identical to the previous pair"）⇒ 改成 dataclass 是**纯 churn**，本轮不做 | 2026-09-25 |
| R-13 P2 | `_create_optimizer` 的 `inspect.signature` 创可贴 | ✅ **已按实测修**（§5.25），但**未**引入 `OptimizerConfig`：真正的问题是**静默吞参数**，不是签名不够显式。实测 `SGD` 签名只有 `(self, dim, lr)` ⇒ `momentum`/`weight_decay`/`ns_steps` 被无声丢弃；且带 `**kwargs` 的类永远收不到 kwargs ⇒ `**config.kwargs` 逃生口是死的 | 2026-09-25 |
| R-14 P2 | `_metric_panel` 每 epoch 全量指标 → 加 `panel_every_n` 开关 | ⚠️ **符号已搬家**，原 `_metric_panel` 不存在，面板现在是共享叶子里的 `ShapingObjective.metric_panel()`，每轮在 `slm_zernike_pib.py:1091` 被调。**未做**：跳过的那几轮 `m_*` 列该留空还是沿用上一轮，是**产品决策**（报告生成器与 NaN 判据测试都读这些列），且需真实 run 对照 → §5.26 | 2026-09-25 |
| R-15 P2 | `_apply_best_on_exit` 往 Recorder 挂属性 → 显式 `RawReport` 字段 | ⚠️ **部分完成，`RawReport` 不该做**（§5.26）：该值是**整轮一个标量**，而 `Recorder.append` 取每轮 record 键的并集⇒ 做成列要么不出现、要么只挂在最后一轮，且 `Recorder` 没有"轮次元数据"schema。已改成**显式直接赋值 + 写明理由 + 加测试**（唯一消费者是 `scripts/compare_shape_objectives.py:258`）| 2026-09-25 |
| R-16 P2 | `SLM_WIDTH/HEIGHT` 与驱动 `Panel_Res` 重复 | ✅ **已完成**（§5.22）：两侧改为 `SLM_WIDTH, SLM_HEIGHT = PANEL_RES`（值实测一致 (1920,1200)），**保留常量名**（有测试 import），只改值的来源；并加 AST 守卫禁止再写回字面量 | 2026-09-25 |
| R-17 P2 | 日志 f-string/`{}` 占位符混用 | ✅ **已完成**（§5.23）：9 处 f-string 日志全改为 loguru 惰性格式化参数，并加 AST 守卫（变异验证过） | 2026-09-25 |
| R-18 P3 | 离线 GS 作闭环初值 `--init-gs`。⚠️ **`gs_warm_start` 已在别处落地**（`slm_gs_refine.py`、`iterative_zernike_shaping.py` + `slm_gs_refine_runner`）→ 本项改为"接入已有实现"或删掉 `:739` 的陈旧注释 | `slm_zernike_pib.py:739` | 2026-09-25 |
| R-19 P3 | 补 sim 台架对称 BenchSession（`sim` 相机后端已有 2f-Fourier），使 R-1~R-17 可无硬件回归 | — | 2026-09-25 |

### 2.2 `utils/` + `scripts/` + `tools/` 架构重构（源自 `docs/refactor/TODO.md`，2026-10-01）

**原前置阻断项（🔴）已于 2026-10-03 完成 → §5.4。** 原文：`utils/` **不是**安全的叶子 ——
`utils/wavefront/pattern_helper.py:20` 裸 `import aotools`（无 try/except），
`utils/__init__.py` eager 拉起 4 个子包 + 88 个 eager 名字（无 PEP 562 `__getattr__`）
⇒ 未装 `aotools` 的机器上 `import ao_shaping.utils` 直接失败。**现已全部修复并有测试钉住。**

| # | 项 | 类型 | 提出 |
|---|---|---|---|
| R-27 | `runner_common.py` 的 CLI 机制 → 新叶子模块（**零 `ao_shaping` 导入**）；`runners/__init__.py` 改真 lazy | ✅ 两半均完成 → §5.11（机制部分已在 R-36 落地，路径按实测改为 `utils/cli_params.py` 而非 TODO 写的 `utils/io/cli_params.py`） | 重构 | 2026-10-01 |
| R-28 | utils 内部去重：D1/D2 ✅ 已合并；D3/D4/D5 ⚠️ **实测判定不可合并**，改为钉特征测试 | → §5.10 | 重构 | 2026-10-01 |
| R-29 | ~~`utils/image/display.py:19` 从 `io/handler.py` 导入 `Register`，AGENTS.md 把方向说反了~~ | ❌ **前提不成立，作废**（2026-10-04 实测）：① 导入方向确实是 `utils/image/display.py` → `utils/io/handler.py`（`:19`），`Register` 定义在 `handler.py:1`，**与 TODO 描述一致**；② 实测 **AGENTS.md 里没有任何关于该方向的表述**（grep `Register`/`handler.py` 无命中），"把方向说反了"**无从谈起**；③ `display.py:12-14` 的注释方向也**是对的**。真正待办是"把 `utils/image/display.py` 搬到 `display/`"（AGENTS 反模式表已如实跟踪，含 `utils/__init__.py:78/80/81` 的三个 re-export），**不在本 TODO 项范围内** | 文档 | 2026-10-01 |
| R-30 | 修 `pyproject.toml` 加 `pythonpath = ["src", "scripts"]`（一行修好 11 个脆弱脚本在 pytest/IDE 下的导入）；清空 `tools/slm/__init__.py` eager 再导出（保留 docstring，**只能清空不能删文件**） | 修复 | 2026-10-01 |
| R-31 | `cartographer/test_smoke.py` 从 `src/` 迁到 `tests/`（现永不被收集）；修 2 处输出路径违规（`generate_cython_optimizer_report.py` 写 `docs/` 根、`generate_centroid_test_visualization.py` 写进 `scripts/reports/`） | 清理 | 2026-10-01 |
| R-32 | 加约定测试（孤儿检测 + `python -m` 一致性 + utils 分层守卫 + **测试写已提交 docs** 守卫），**warn-only + baseline** | ✅ 4 条守卫全部落地，2 条零 baseline（树本来就干净）→ §5.12 | 重构 | 2026-10-01 |
| R-33 | `micro_dm_image_collect.py` → `with_params(MicroDMParams)`；删 7 个驱动内部符号导入与手写 `_resolve_ips`；同 PR 内启用已有的 `R50Controller.__enter__/__exit__`（全仓库零使用）。⚠️ 必须先钉死 Micro-DM 磁盘布局（承重：`find_cell_image` + 4 个 `md_img_*` 脚本依赖） | 重构 | 2026-10-01 |
| R-34 | 清理被 git 跟踪的 `scripts/tuning_devices/stdWavefront/` **66 个 .txt（~57 MB）**；`train_data_collect.py` 与 `micro_dm_image_collect.py` 均 **0 测试** | 清理 | 2026-10-01 |

### 2.3 `tools/slm/` CLI 层重构（源自 `src/ao_shaping/tools/slm/TODO.md`，2026-10-01）

**只动 CLI 层，不合并命令体**（settle/slot/dark-frame 是位置敏感步骤，挪动会产生看似可信的错数据）。

| # | 项 | 当前实测 | 提出 |
|---|---|---|---|
| R-37 | Step 2/3：逐个迁移 19 个可执行探针到 `with_params` 机制。**原计划「抽共享 dataclass 到 `params.py`」已实测证伪并取消**（见 §5.15）：19 个探针 / **287 个声明 flag** 中**只有 1 个**（`--settle-extra-s` ×3）能原样共享。改为**每个探针自带 dataclass，default/help/type 全部留在本地不动** | ✅ **已完成 15/15 个 Click 探针**（§5.15）；4 个 argparse 探针拆出为 R-42 | 2026-10-01 |
| R-42 | **4 个 argparse 探针改 click**（`slm_abba_probe` / `slm_drift_probe` / `slm_floor_probe` / `slm_zernike_sweep_probe`）。实测三个非机械迁移障碍：① `main(argv) -> int` + `raise SystemExit(main())`，click command 不接 argv；② `test_slm_abba_probe.py:541/560` **直接绑定 `probe._parse_args(...)`**（断言默认值 + 断言非法 `--cam-type nikon` 报错），改 click 就得删掉 `_parse_args` 并重写这些测试；③ `--help` 格式从 argparse 变 click | 同左 | 2026-10-04 |
| R-38 | ~~canonical 采用率过低~~ | ❌ **2026-10-04 实测后按原描述拒做**（见 §5.19）。两条理由：(1) `phase_to_slm_grayscale(phase, slm=...)` **就是** `slm.create_phase_from_array(...)`（`phase_display.py:72-73` 直接委托），那"7 处直调"**行为完全等价**，不是缺陷；(2) `zero_order_center` 返回 **`(x, y)`**，而裸 `np.unravel_index(np.argmax(...))` 解包成 **`(y, x)`** ⇒ **替换不是机械操作**，照抄会静默转置每个 ROI 中心 | 2026-10-01 |
| R-39 | 曝光默认值 7 种并存（0.02/0.03/1.1/1.2/2.0/3.0/4.0 ms）；内存槽轮换 3 种写法（驱动自动 / 自建 `SlotRotator` / 手工 `current_slot`） | 同左 | 2026-10-01 |
| R-41 | flag 拼写分裂（2026-10-04 实测）：`--cam-type` **9** 个探针 vs `--camera-type` **2** 个（`slm_diagnose` / `slm_lut_runner`）；另有 `--output` **8** vs `--out` **4**、`--slm-wavelength` **15** vs `--wavelength` **3**。建议保留现有拼写不破坏习惯用法 | ✅ **已完成**（§5.18）：两种拼法都**保留**，并加**契约测试**把意图写死 —— 不只是"golden 顺带钉住"，而是明确断言「两套都在、且没有任何探针同时暴露两套」 | 2026-10-01 |

---

## 3. 离线缺陷修复 / 文档（无需硬件）

| # | 项 | 位置 | 提出 |
|---|---|---|---|
| F-1 | ~~**内存槽固件 no-op 违规**~~ | ✅ 已修 → §5.16。`:383` 曾在 `for i in range(max_iter)` 里反复 `apply_compensation(comp_gs, memory_slot=2)` → 固件把已显示槽当 no-op，**第 2..N 次迭代全是空操作，LCOS 不刷新**。⚠️ 原记录另两条**实测为不成立**：`memory_mode` 在 `display_data` 签名里**本就默认 `MEMORY_MODE_INTERNAL`**（`driver.py:1197`），不传不是"漏"；`:272` docstring 的 "1-128" 与驱动 `_display_memory` 的校验区间一致，也不是错 | `tools/slm/cartographer/dynamic_compensation.py:263,383` | 2026-10-01 |
| F-3 | `--display/--no-display` 的 help 写"暂未实现"，需确认补实现还是删选项 | ✅ **无需改动**（2026-10-04 实测）：**两个同名 flag 状态不同，且各自 help 都是对的** —— `DmMatrixRunnerParams.display`（`runner_common.py:1689`）只 `click.echo("Note: --display mode is not yet implemented...")`⇒ help 标"暂未实现"**准确**；`HadamardMatrixRunnerParams.display`（`:1743`）在 `zernike_matrix_runner.py:1203` **真的构造 `ZernikeCalibrationDisplay`** ⇒ 它的 help 不带caveat 也准确 | 2026-09-25 |
| F-4 | `src/ml/` 移入 `src/ao_shaping/ml/` 并更新所有引用（`docs/issues_report.md` §10.3，待评估至今） | `src/ml/` | 2026-05-26 |
| F-5 | `docs/issues_report.md` §11 的代码规范整改：`print()` 替代 loguru（原文 82 处）、宽泛 `except`、配置项分散、大文件拆分（~22 个）、冗余 `__main__` 入口（32 处）、`__future__` 覆盖率（29 个文件）。⚠️ **原文数字已过期，实施前需重新扫描** | 全仓 | 2026-05-26 |
| ~~F-9~~ | ~~`repeat_shape_objectives.py` 加进度显示（用户要求）~~ | ✅ **部分已存在 + 补全局计数** → §5.28。原有 `rep {rep}/{repeats}` 只报**单 variant 内**进度；各 variant 耗时差异大，操作者无法从日志判断整体到哪一步 → 新增 `run i/N` 全局计数 | `scripts/repeat_shape_objectives.py` | 2026-09-30 |
| **F-10** | 🔴 **`sim/AGENTS.md`「已知约束」第 4 条描述的代码改写从未落地**：该条声称 `_rescale_for` 已改为**只** `(_R0_REF_500/r0_slab)**(5/6)`，并称已移除 `lam/_LAM_REF_500`、`/_CAL_REF`、`*sqrt(1.03)`。**三者至今仍在** `oopao_backend.py:96,101-103`（`_CAL_REF = 0.6191` 在 `:71`）。连带第 3 条的实测常数 1.068/2.628 **不可复现** —— 真实值是 **5.428 / 13.354**（与 `docs/oopao_impact/report.md:62-63` 一致）。**先落地改写并重跑 `generate_oopao_impact_report.py`，或回退那两条。** | `oopao_backend.py:88-103` + `sim/AGENTS.md` 第 3/4 条 | 2026-10-01 |
| **F-11** | 🔴 **SLM 序列号三路冲突**：`drivers/AGENTS.md:158` 与 `docs/slm/bench_calibration_20261001.md` 记 SLM#1 = **22030108**（@1064nm，2π=993）；`drivers/slm/AGENTS.md:114,139` 记 **22030102**（@532nm，2π=998）；`docs/slm/report2.md` / `report3.md` / `zernike_linearity/linearity.md` 记 **23020026**（@532nm）。三者或为两台设备。**引用前必须确认，并回写 `drivers/AGENTS.md` 硬件表**（Daheng CCD `FJB24112232` 已于 2026-10-01 补录进该表） | `drivers/AGENTS.md` 硬件事实表 | 2026-10-01 |
| **F-12** | **焦面标定常数三方不一致**：`AGENTS.md:697` 写 `5021/Λ`（对应 3.31 µm 像元）；`docs/slm/model_in_loop_bench_calibration.md` / `README.md:517` 写 7400–7600（对应 2.2 µm 像元）；`docs/slm_pib_heuristic_hw/report.md:159` 主张改 **10954**。⚠️ **2.2 µm 像元推得 ~7557 而非 10954，故该主张本身也待复核**。H-11 只覆盖了 132940 vs 7600，**未覆盖此三方冲突** | `slm_diagnose.py:54`、`slm_lut_runner.py:38`、`slm_bench_probe.py:78`、两处测试 | 2026-10-01 |
| **F-13** | ~~`strehl()` 被当物理 Strehl 比~~ | ✅ **早已基本处理完**（2026-10-04 实测）：`strehl()` 的 docstring 已自述 "Normalized overlap (**Strehl-like**)" 并写明是余弦相似度；`beam_shaping_papers.md:30` 也已定义为"归一化重叠"；`zotero_objectives/README.md:524-529` 已列命名冲突与建议。⚠️ 原记录里"`beam_shaping_benchmark.py` 消费其输出并称 Strehl"**已不成立** —— 该文件现在**完全没有** overlap / strehl / cosine 引用。**本轮只补最后一处裸列名**：表头 `Strehl` → `Strehl†` 并加脚注 → §5.24 | `slm_shaping_bench.py:309` | 2026-10-01 |
| **F-14** 报告生成写在 `algorithm/` 层 | ✅ 违反反模式「report generation MUST live in `scripts/`」。写出器移到 `scripts/generate_beam_shaping_benchmark_report.py`；`algorithm/` 侧 `run_benchmark`/`run_benchmark_suite` **不再接受 `output_dir`** → §5.13 |
| **F-15** `docs/beam_shaping_benchmark_metrics.md` 被 README 当权威链接，实际是 1 行 smoke 残留 |✅ 9 单元权威网格已生成并**提交**（~100 s 全离线），README 改指真产物；1 行残留**删除** → §5.13 |

---

## 3.1 文档层已完成的订正（2026-10-01 复核，19 个文件）

已就地订正（加显式修订标记 / 改错值 / 删死链），**不需再动**：

| 文件 | 订正内容 |
|---|---|
| `docs/slm_pib_shape_50px/report.md` | 目标方向写反（`shape` 是 max）→ 已加警示 + 改正解读；11 处反斜杠死链 |
| `docs/zernike_farfield_sim/report.md` | §6 结论 2 与自己的 §5.1 表格矛盾（n=4 实为退化**更慢**）→ 改写 |
| `docs/slm_pib_heuristic_hw/report.md` | 4 处内部矛盾（加载预算 30→82-116、初始值 -34→-1.21、曝光"安全上限"作废、附录 A/B ROI 不可比）；待办 ③ 已完成 |
| `docs/diff_beam/README.md` | runner 与 `diff-beam` 命令已删除 → 归档标注 + 5 条路径订正 |
| `docs/slm/slm_shaping_diff/readme.md` | `diff-shaping` 命令已删除 → 归档标注 + 3 条路径订正 |
| `docs/slm/slm_pattern_helper.md` | Noll 表与 canonical **相反** → 按 `list_zernike_modes()` 实输出改正；2 处会 `TypeError` 的签名；11 个不存在的插图 |
| `docs/slm/slm_gui_manual.md` | §7 图案表把 5 个控制类标成「原始 uint16」**与代码相反**（违反 raw-radians 契约）→ 改正 + 补漏 `GSSquareGaussianControl` |
| `docs/slm/slm_square_spgd/README.md` | B2/B3 已修复（`_zernike_to_uint16` 已删）；3 个已移除命令 |
| `docs/slm/report2.md` | 12 处漂移行号/路径（`L1369→L1457` 等）；`_apply_shift` 已不存在；SLM 序列号存疑标注 |
| `docs/slm/report3.md` / `daily_2026-09-16.md` | `python -m` 路径补 `slm/` 段 |
| `docs/slm/daily_2026-09-08.md` | 状态表 3 项「待修复」其实早已修完；**P1「面板不调制」结论已被 RETRACTED** |
| `docs/slm/model_in_loop_bench_calibration.md` | `132940/P` → `7600/P`（与自身另外三处差 17.5×）；相机序列号自相矛盾；曝光默认值已被取代 |
| `docs/slm/align_windowed.md` | `AGENTS.md L592` → `:667`（原行号是代码围栏） |
| `docs/beam_shaping/papers/literature_survey.md` | 5 个已删除 runner、`utils/beam_metrics` 旧路径、DOI 漏首位 `0` |
| `docs/oopao_impact/report.md` | 加「**本报告是对的，`sim/AGENTS.md` 是错的**」警示（F-10）；POSIX 路径 |
| `docs/fouriergsnet_pipeline/report.md` | `phi.detach()` bug **已修复**（原文仍写「待修」）；7 处漂移行号 |
| `docs/fouriergsnet_sim/report.md` / `oopao_vs_numpy` | POSIX 路径；`optimizer/rl/envs/fouriergsnet_env.py` **从未存在**；`K` vs `K_UNROLL` 消歧 |
| `docs/slm_pib_bench/report.md` + `EXPERIMENT_REPORT.md` | 死链；δ=0.2 折叠门两个分母（45/60 vs 144/200）未对账；run 总数 51/44 未对账；两个 `### 3.4` 重编号为 3.4–3.7；悬空的 `§3.3.1` |
| `docs/issues_report.md` + `AGENTS.md`/`README.md`/`sim/AGENTS.md`/`drivers/AGENTS.md` | 见 §3 F-10~F-12 及各文件内修订标记 |

**未改动**（复核后确认干净或仍成立）：`docs/heuristic_pib/`、`docs/strehl_benchmark/`、
`docs/beam_shaping/papers/beam_shaping_papers.md`、`docs/fouriergsnet_pipeline/offline_training/`、
`docs/slm_pib_sim/report.md`、`docs/slm/zernike_response_matrix_report/`、`docs/slm/bench_probe/`、
`docs/slm/bench_calibration_20261001.md`（最新）、`docs/slm/zernike_linearity/linearity.md`、
`docs/oopao_vs_numpy/report.md`（数值部分）、`docs/zotero_objectives/README.md`（仅 1 条路径）。

---

## 4. 两处原文档结论需要修正（照做会出事）

| # | 原结论 | 实际情况 | 提出 |
|---|---|---|---|
| X-1 | `docs/refactor/TODO.md` #16：「`utils/image/resample.py` **0 导入** → 删模块并修 README」 | ❌ **现在有 1 个生产导入方**：`FourierGSNet.py:42` `from ao_shaping.utils.image.resample import resample_to_grid`（用于 `:345`、`:419`）。**照原文删会直接打断 FourierGSNet** | 2026-10-01 |
| X-2 | `tools/slm/TODO.md` D2：「`calibration.py:2182` min-max 归一化 → 尺度无关反模式（系数 ×1 与 ×4 输出字节相同）」 | ❌ **误报**。该处 `P` 是**实测功率-扫描位置曲线**，归一化到 [0,1] 后用 `np.interp` 找 50%/84%/16% 交点是标准做法，功能正确。它**不是** `PatternHelper._zernike_to_uint16` / `ZernikeDM.generate_phase` 那类系数归一化 | 2026-10-01 |
| **X-3** | ~~两个 runner 的 `ZERNIKE_APERTURE_RADIUS` 默认值不同：`slm_zernike_pib.py` = **300.0**，`slm_zernike_shaping.py` = `min(PANEL_RES)/2.0` = **600**，README 记 **600**~~ | ✅ **2026-10-04 解开，原结论的前提是错的**（见 §5.17）。实测：`_zernike_to_phase` 两份**逐字节相同**，但读各自的模块常量 ⇒ pib=**300**、shaping=**600**（import 实测，非读码）。**但 `slm_zernike_shaping` 没有任何生产导入方**（只有测试 + 自己的 legacy `__main__`）：`slm-pib`→`slm_zernike_pib`、`rms-zernike`→**`optimizer.wf.rms_by_zernike`**（不是 shaping）、`delta_explorer`→`slm_zernike_pib`。所以**不存在"换个 runner 就换光阑"的生产风险**，也**不需要硬件对比**：语料 `test_gsnet_dataset.py:328-333` 钉的 **300** 与 pib 一致，shaping 的 600 是死代码 | 2026-10-03 |

---

## 5. 已完成（从待办中移除，供追溯）

| 原项 | 落地情况 |
|---|---|
| `docs/TODO.md` P2-6：文件尾 argparse `__main__` 块 | ✅ 已删。文件 1592 行结束于 `return recorder`，全文件无 `__main__` / `argparse`；`main.py` Click 是唯一入口 |
| `docs/issues_report.md` §1.2：`config.py` 依赖具体硬件 | ✅ 已改为 `drivers.dm._registry.get_dm_registry()` + `list_reachable_types()`，不再直接 import `NLight` |
| `docs/issues_report.md` §10.4：删空目录 `drivers/sim/wfs/` | ✅ 目录已有内容（`simulated_wfs.py`） |
| `docs/slm/report2.md` §2.3：`ZernikeDM.generate_phase` min-max 归一化 | ✅ 2026-09-16 已移除，`zernike_dm.py:90,117-118` 注明；系数即弧度 |
| `OBJECTIVE_TARGET_SHAPE_MERGE_PLAN.md`（2026-09-26）：`objective`/`target_shape` 合并 | ✅ 已落地为 `runner_common.py:622` 的 `ObjectiveTarget`（`target_shape` 成为 objective 下级字段） |
| `docs/refactor/TODO.md` §一：`tools/` 是中间层、`scripts/` 是同级 | ✅ 架构结论已采纳为文档，不再是待办 |
| `README.md` `--wfs_type` / `--disturbance-cn2` / `--lr` / `--delta` CLI 文档（2026-10-01 merge 引入） | ✅ 已补文档 |
| **slm-pib ABBA 漂移对消（2026-10-02）** | ✅ 提交 `e9d1769`（`_spgd_capture_signs` + `SlmZernikePibConfig.abba_sampling` + 采集循环改为按符号序列驱动，fold 门/基线 EMA/饱和检查/符号均值全部覆盖每一帧）、`139a461`（`--abba-sampling` + `_build_slm_pib_config` 透传）、`796d3ce`（执行真实 epoch 循环的回归测试，断言每轮采集 2 vs 4 帧 + `_pos_c` 绑定）、`1d6329e`（README）。**opt-in，默认 `False` ⇒ 2 帧路径逐字节不变。** 过程中修掉一个 subagent 引入的**首轮必崩 bug**（4 元组切 `[2:]` 得 2 元组却解包 3 名 ⇒ `ValueError`，默认模式同样崩），并补上缺失的循环级测试（原有 12 个测试只测符号助手与漂移代数，不执行循环，故漏检）；变异测试确认可捕获。⚠️ **实机验收仍未做 → 转 H-19** |

### 5.1 第 1 批（纯收益，零行为风险）—— 2026-10-03

| 原项 | 落地情况 |
|---|---|
| **R-23** 删 `utils/slm_utils.py` | ✅ 已删（483 B，0 导入的 `sys.modules` 别名，docstring 理由"测试重置 `_last_slm_slot`"已过期）。全仓库仅 3 处提及，两处是 TODO 文档 |
| **R-24** 补 `tools/micro_dm/__init__.py` + 修失效 docstring | ✅ 新建带 docstring 的包 `__init__`（明确不做 eager re-export 及原因）。**实测 docstring 有 13 处**（TODO 记 12）`python -m ao_shaping.tools.micro_dm_image_collect`，全部改为真实路径 `.micro_dm.`；`python src/...` 那行也改。新增 `test_tools_package_facade.py` 锁定"docstring 不得再出现旧路径" |
| **R-30** `pyproject` pythonpath + `tools/slm/__init__.py` 惰性 | ✅ `pythonpath = ["src", ".", "scripts"]`（实测 `scripts/` 下有 2 个脚本靠同级 import：`generate_diff_shaping_report`、`md_img_diff_centroid`）。`tools/slm/__init__.py` 的 **60+ 个 eager re-export 全部清空**为 `__all__ = []`（实测**零生产代码**从该 facade 导入，4 处命中全是子模块 import）。`tools/micro_dm/__init__.py` 同规则 |
| **R-31** `cartographer/test_smoke.py` 迁到 `tests/` + 修 2 处输出路径 | ⚠️ **不是"迁移"，是"删除 + 补差"**：`tests/ao_shaping/tools/slm/test_slm_cartographer_pure.py` 已**完全覆盖** 5 个 smoke 用例中的 4 个且断言更严 ⇒ 迁移只会造成重复覆盖。已删 `src/.../test_smoke.py`，把**唯一未覆盖**的 `CompensationResult`/`CompensationConfig` 往返测试（含 ndarray→list 展平）补进 pure 测试。输出路径：`generate_cython_optimizer_report.py` `docs/` → **`docs/benchmarks/`**（git 跟踪产物已随之移动，README 链接同步）；`generate_centroid_test_visualization.py` → **`docs/centroid_test_visualization/`**，并删除 `scripts/reports/` 整目录与 `scripts/README.md` 的 `### reports/` 小节 |
| **R-40** `tools/slm/__init__.py` docstring | ✅ 重写为分组索引。**实测漏 10 个模块**（TODO 记 6）：除 TODO 列出的 6 个，还有 `slm_abba_probe` / `slm_bench_metrics` / `slm_drift_probe` / `slm_floor_probe`。LUT canonical 路径 `utils/slm_lut.py` → **`utils/slm/slm_lut.py`**（已与 `drivers/slm/santec/driver.py:1674` 交叉核实） |
| **F-2** README `slm-diagnose` 选项表 | ✅ 表格 `--cam-type` → `--camera-type`（与 `slm_diagnose.py:256` 及同页警告块/示例统一） |
| **F-6** 清理陈旧 `.pyc` | ✅ 删 5 个（`slm_shift_calib` / `slm_zernike_report` / `_slm_fix_wavelength` / `_slm_health_check` / `_slm_reboot_wavelength`） |
| **F-7** `scripts/README.md` 补 8 个脚本 | ✅ 8/8 补齐（`## Report Generation Scripts` 与 `## Verification Scripts`），选项/默认值均**读源码核实**而非推测。同时修 `generate_centroid_test_visualization.py` 的陈旧路径、`generate_cython_optimizer_report.py` 的输出路径、删 `### reports/` 小节。验证：8 个标题全部存在、`scripts/` 下 0 处 `scripts/reports` 引用 |
| **F-8** 脚本 CJK 字体 + `--help` | ✅ **9 个**脚本补 `font.sans-serif` / `axes.unicode_minus`（初扫 11 个命中里 2 个是误报：`generate_beam_shaping_papers_report.py` / `generate_zernike_farfield_sim_report.py` 已用 `rcParams["font.family"]`）。**`--help` 部分：0 个手写解析器** —— 65 个脚本无一使用 `add_help=False` 或自写 `sys.argv` 解析器，所有带参入口都是 `argparse`/`click`（自带 `-h/--help`），故该项本就满足，无需改动。`scripts/*.py` 的 diff **只有 CJK 块 + 2 处输出路径**，无任何数值/科学行为改动 |

### 5.2 第 2 批 P0（止真 bug）—— 2026-10-03

> 🔴 **共同发现：`slm_zernike_shaping.py` 是 `slm_zernike_pib.py` 的同源副本**，R-1~R-4
> **四条 bug 两边都有**，原条目只记了一边。全部两侧同改。

| 原项 | 落地情况 |
|---|---|
| **R-1 P0** 能量守卫惩罚被绕过 | ✅ `ObjectiveResult` 新增**必填**字段 `tracking`（"best" 跟踪值），`raw()` 返回值从 2 元组扩为 `(j, ratio, tracking)`，`__call__` 对 `j` 与 `tracking` **施加同一个** `GUARD_PENALTY`；`ShapingObjective.tracking_value()` **整体删除**，6 个调用点改读 `res.tracking`。`GUARD_PENALTY = 1e3` 提取为 `drivers/ccd/common.py` 常量并经 `utils/image/target` 导出（顺带完成 R-11 的一半）。`ratio` 明确**不**惩罚（离线报告要靠它统计并排除 guard 行）。测试：新增 4 条 guard×tracking 用例 + 1 条"无守卫时 tracking 必须等于旧 `tracking_value` 返回值"的**前后一致性**断言 + 1 条"`tracking` 必须必填否则 `TypeError`"（防止默认值把 R-1 悄悄放回来）。**变异验证：撤掉惩罚 ⇒ 3 条失败** |
| **R-2 P0** `learning_schedule` 坐标系 | ✅ 两侧都改为 `radius(init_img, center=reference_center, energy=0.8)`（窗口局部光斑位），与已修的 `r_bucket` 块和 SPGD 分支的 `pos_center` 一致。**实测危害比"读数错"更严重**：`radius()` 对越界中心**不抛异常、静默返回 0.0**（实测窗口局部 14.53 vs 全帧 (633,934) 得 0.0）⇒ 自动调度一直吃的是"零半径"。测试：1 条钉住"越界中心静默返回 0"这个陷阱本身，1 条 AST 守卫断言**两个模块**的 `learning_schedule(radius(...))` 都不再锚在裸 `center` 上。**变异验证：改回 `center` ⇒ 守卫失败并报出行号** |
| **R-3 P0** 饱和判定硬编码 255 | ✅ 新增 `full_scale(img)` / `is_saturated(img)` 到 `drivers/ccd/common.py`（**紧邻既有的 `resample_on_saturation`**，即饱和逻辑的天然归属），经 `utils/image/target` 同层导出；4 个优化器模块（`slm_zernike_pib` / `slm_zernike_shaping` / `slm_square_shaping` / `pib` / `combined_optimizer` 共 6 处调用点）统一改用之。整数 dtype 读 `np.iinfo(dtype).max`（uint8→255、uint16→65535）；**浮点 dtype 用 255.0 而非 1.0** —— 语料实测 uint8/float32/float64 帧**逐帧最大值全 ≤ 255，同一个 0–255 探测器量纲**，当成已归一化会压缩最多 255×。`resample_on_saturation` 的 `saturation_threshold` 默认值 `255.0` → `None`（按 dtype 派生）。测试：uint8/uint16/float32/float64 各钉"恰在满量程触发、差 1 不触发"，外加一条**显式记录旧字面量会误判 uint16**（`300 >= 255` 但 `300` 离 65535 很远）。**变异验证：退回硬编码 255 ⇒ 3 条失败** |
| **R-4 P0** `_log_row` 漏列 | ✅ 两侧 `_log_row` 补齐 `w_ee` / `ee_term`（6 项齐全）。**额外发现 `optimizer` 列也有同一类不对称**（`_row0` 有、`_log_row` 无）⇒ 一并补齐。测试：`test_rms_pib_adaptive_columns_are_present_on_every_row[spgd|ga]` + `..._are_not_nan_on_later_rows`（NaN 判据）+ `test_both_engines_log_the_full_adaptive_column_set`（**参数化两个模块**）。⚠️ 最后这条是必需的：原有 sim 测试从 `slm_zernike_shaping` 导入 `optimize_slm_zernike_pib`，**只覆盖副本一侧** —— 我第一版测试因此在变异 `slm_zernike_pib.py` 时**假通过**，补上参数化后两个模块各自都能被捕获 |
| **附带修复：`src/ml/zernike/models.py` 语法损坏** | ✅ 工作区里该文件有 2 行被压到第 0 列（`observable:` / 缩进错乱的 `normalization:`），`ast.parse` 直接 `IndentationError`。它经 `ao_shaping.runners.__init__` → `slm_gsnet_runner` → `gsnet_train` → `gsnet_dataset` → `ml` 的导入链**打挂整条 `import ao_shaping.runners`**，任何 import `runner_common` 的测试都无法收集（不止 `ml` 的测试）。按同文件 docstring 与 `__post_init__` 校验逻辑恢复为 `observable="intensity"` / `normalization="peak"`，`ast.parse` 通过、29 个 sim 测试恢复收集。⚠️ **不属于 TODO 任何条目，是工作区既有损坏**，建议单独提交并复核 `ml/hwdataset` 那批 WIP |
| **R-9 P2** 300px 光阑注释挂错常量 | ✅ 两侧都把那段硬件坑说明从 `TARGET_BOX_WAIST_FACTOR` 之后的**裸字符串字面量**（不是 docstring、不可达）搬回 `ZERNIKE_APERTURE_RADIUS` 正下方，真正变成它自己的 docstring |
| **R-10 P2** 死代码 `gauss_center` | ⚠️ **不是"零调用"——有 2 个专属测试**（`test_slm_zernike_pib_shape.py`），但零生产调用。彻底删：两侧 `gauss_center` 定义（各 50 行）+ 那 2 个只测死代码的测试 + import。它与 `spots_calc` 现有任何函数**不重复**（`centroid` 带阈值、`center_of_brightness` 无背景扣除），但既无生产调用方，就不占 utils 位置 |
| **R-11 P2** 字面量收口 | ✅ 两侧 `np.clip(..., -5.0, 5.0)` 共 5 处 → `-ZERNIKE_CLIP`；`±1e-4` 共 12 处（best 比较 + 退出决策）→ 新增常量 `IMPROVE_EPS = 1e-4`（带注释说明为何要 epsilon：测量抖动不得改写 best，退出决策也不得在噪声上翻转）。守卫惩罚 `1e3` 已在 R-1 落地为 `GUARD_PENALTY`。`ruff check` 全干净 |

**⚠️ 副本漂移（R-1~R-11 之外的额外发现，尚未处理）**：`slm_zernike_pib.py` 与
`slm_zernike_shaping.py` 是近乎逐行的副本，且**同名常量已经不同**
`ZERNIKE_APERTURE_RADIUS`：pib = `300.0`，shaping = `min(PANEL_RES)/2.0` = **600**。
README 记 `--zernike_radius` 默认 **600**。两者都是生产入口
（`slm-pib` / `rms-zernike`），默认值不同意味着**同一硬件上台架的 Zernike 光阑
取决于走了哪个 runner**，而 300 vs 600 正是 R-9 那段注释里"只有内一半落在光束上
⇒ 修正静默无效"的分界。**需硬件判定哪个正确**，暂记为 **X-3**（见 §4）。

---

### 5.3 第 2 批剩余 —— 2026-10-03 状态

| 项 | 状态 |
|---|---|
| **R-35** `tools/slm/` CLI 特征测试（Step 0 硬前置） | ✅ **已完成**（下方详录）→ R-36→R-41 解锁 |
| ~~R-9 / R-10 / R-11~~ | ✅ 完成 → §5.2 |

#### R-35 落地细节（`tests/ao_shaping/runners/test_cli_contract_freeze.py`，37 passed）

钉住 5 类可观测行为，全部离线（`--help` 不碰硬件）：

1. **命令清单**：19 个注册名冻结（README 记 19，一致）。
2. **`--dm_type` 选项列表**：⚠️ **实测只有 5 个命令暴露它**（`wf` / `pib` / `pipeline` /
   `combined` / `dm-matrix`），**TODO 记的 8 个不准**。冻结了各自的 choice 元组，并**单独
   钉住 `dm-matrix` 比其余 4 个少一项**（缺 `asyn_micro`）—— 这不是 bug 而是
   `runner_common.py:137-152` 的 `DM_TYPES_PRE_ASYN_MICRO` 刻意快照，为的是
   `dm_matrix_runner` 的 help 与注册前逐字节一致。另加一条"任何命令**新增** `--dm_type`
   都算行为变更"的守卫。
3. **`--help` 全文 golden**：19 条命令的完整 help 冻结进
   `tests/ao_shaping/runners/cli_help_golden.json`（45 KB）。有意改动后用
   `AO_CLI_CONTRACT_UPDATE=1` 重生成。
4. **同一命令内不得有重复 option flag** —— 重复正是"两个 dataclass 抢同一个 flag、
   click 静默丢掉一个"（R-37 迁移 22 个探针时最可能犯的错）的表现。实测当前 0 重复。
5. **`pupil_center` 的「二元组注解 + callback」耦合**：注解必须保留
   `tuple[float, float]` 那一支、option 必须挂 `parse_tuple`、默认值必须仍是字符串
   `"(0,0)"`，并**行为化**钉住 `3,4` / `(3, 4)` / `-1.5,2.25` / `577,655` 四种写法以及
   `mass`/`max`/`shape` 三个 `-c auto` 关键字的透传。另加一条说明**为什么删 callback 会
   静默出错而非崩**：下游做 `pupil_center[0]`，没有 callback 时 `"(0,0)"[0] == "("`。

**变异验证**：① 把 golden 里一个字符 `Usage:` → `Usage :` ⇒ 对应 help 比对失败；
② 让 `DM_TYPES` 去掉 `asyn_micro` ⇒ 4 条 choice 冻结 + 1 条 help golden 同时失败。

⚠️ **第一版 golden 有测试污染，已修**：`--dm_type` 的 choice 列表来自**活的 DM 注册表**，
全量套件里别的测试先注册了 DM 类型 ⇒ `dm-matrix` 的选项比裸跑时多一项 ⇒ golden 与
choice 冻结双双变红。已改为：① choice 只断言"**声明的 7 个都在、且顺序前缀不变**"
（真实内容由 `DM_TYPES_PRE_ASYN_MICRO` 的接线断言单独钉）；② golden 里**只**把
`--dm_type` 那一行的方括号内容归一化为 `[<DM_TYPES>]`，其余选项列表（`--wfs_type` /
`--mode` / `--mla-index` …）**仍逐字节冻结**。已复测：干净跑与"先跑污染测试再跑"两种
顺序均全绿。

### 5.4 R-20（第 3 批硬阻断）—— 2026-10-03 完成

> R-20 已在 §2.2 移除；落地记录如下。它是 R-21/R-22/R-25/R-27/R-28 的共同前置。

| 原项 | 落地情况 |
|---|---|
| **R-20 前置** `pattern_helper` 的 `aotools` 改惰性 + `utils/__init__.py` PEP 562 | ✅ 两处根因都修了。**先写测试复现**：在子进程里用 `sys.meta_path` 拦截器屏蔽 `aotools`，修复前 5 条用例**全红**（`ImportError: No module named 'aotools'` 直接抛在 `import ao_shaping.utils` 上）。① `pattern_helper.py:20` 的裸 `from aotools...` 移进 `TYPE_CHECKING`（仅供 `self._turbulence_screen` 的注解，`from __future__ import annotations` 下不需运行期求值）+ `init_turbulence_screen()` 体内延迟导入 —— **失败模式从 import 期移到调用期**，且只在真要造 Kolmogorov 相屏时才需要。② `utils/__init__.py` 的 **16 处模块作用域 import / 88 个 eager 名字**改为 PEP 562：`_LAZY_EXPORTS: dict[str, str]`（名字 → `"module:attr"`）+ `__getattr__` + `__dir__`。`__getattr__` 内部**必须写 `globals()[name] = value` 缓存**（否则后续 `from ao_shaping.utils import x` 会重新绑定全局、彻底绕过惰性）—— 与 `drivers/_lazy.py` 的同一条红线一致 |

**"不要保留兼容"的边界如何划**：`__all__` **一个名字都没删**（R-20 原文只要求"只延迟，绝不删名字"）。
已用**逐对象比对**证明等价：新旧版本各跑一遍探针，把 `__all__` 每个名字解析成
`module:qualname` 后对比 ⇒ **85 vs 85，无增无减，`resolved-object differences: NONE`**。
其中两个易错点已确认原样保留：

- `calc_n_zernike_terms` 原文件里**同时**来自 `matrix_utils`（line 98）和
  `zernike_calc as calc_n_zernike_terms_zern`（line 128）—— **两个名字都在**，
  且 `calc_n_zernike_terms` 仍解析到 `matrix_utils`（因为 `zernike_calc` 自己也是
  从 `matrix_utils` 转手的，同一个对象）。
- `logger` 保持 eager（`configure_error_logging` 在模块作用域用它，且 loguru 无可选依赖）。

**测试**（`tests/ao_shaping/utils/test_utils_import_is_sdk_free.py`，7 passed）：
子进程屏蔽 `aotools` 后 ① `import ao_shaping.utils` 仍成功 ② 公共面 ≥87 个名字且
`PatternHelper` 可用 ③ `PatternHelper((64,64))` **构造成功**、只有
`init_turbulence_screen()` 才抛 `ImportError`（证明失败已移到调用期）；另加两条静态
守卫（`pattern_helper` 模块作用域不得出现 `aotools`、`utils/__init__.py` 模块作用域
不得出现 `from ao_shaping.utils...`）与两条表面守卫（`__all__` 不得缩到 85 以下、
未知名字仍抛 `AttributeError`）。

**回归**：`utils` + `algorithm` + `model` + `tools` + `drivers/ccd` + `gui` + `scripts`
+ CLI 契约 = **2247 passed**，只剩 6 个既有的 `gx` 失败（见 §5.10）。

### 5.5 R-26（第 3 批）—— 2026-10-03 完成

| 原项 | 落地情况 |
|---|---|
| **R-26** `generate_strehl_benchmark_report.py:442` 硬编码兄弟产物 → 改 CLI 参数 | ✅ 两处硬编码都改成参数：`load_pib_summary(pib_csv)` 与 `main(*, out_dir, pib_csv)`；新增 CLI `--pib-summary`（默认 `PIB_SUMMARY` 常量）与 `--out-dir`。**重点不是"加个参数"，而是把静默降级变成可见事实**：原实现在文件缺失/列不对/读失败时只 `return None`，报告照样渲染、照样看起来完整，只留一条 log —— 这正是 TODO 说的"**磁盘上已有的报告可能就是错的**"。现在该分支写入 markdown 的是 `**Cross-benchmark comparison SKIPPED** — no usable PIB summary at <解析后的路径>`，并告诉读者用 `--pib-summary` 修。报告是**要提交进仓库**的，所以新增 `_rel()` 把路径渲染成**仓库相对 POSIX**（绝不写机器绝对路径），并有测试钉住 |

**测试**（`tests/ao_shaping/scripts/test_strehl_report_pib_summary.py`，12 passed）：
路径确实是参数（`inspect.signature` 断言）/`main` 三个参数都在/缺文件·缺列·正常三条
`load_pib_summary` 分支/跳过时报告里**必须出现 SKIPPED + 具体文件名 + 修复提示**/
正常时表格仍在/`_rel` 的相对路径与 POSIX 分隔符/`--help` 确实暴露两个新 flag。
**变异验证**：把提示改回原来那句不带路径的斜体说明 ⇒ 2 条失败。

### 5.6 R-25（第 3 批）—— 2026-10-03 完成

| 原项 | 落地情况 |
|---|---|
| **R-25** 建 `scripts/_common/` 抽出报告/绘图助手 | ✅ 建了 `scripts/_common/`（并把 `scripts/` 变成真包，`__init__.py` 到位），抽出 **9 个助手**、迁移 **10 个生成器**。TODO 原文列的是 4 组（`iters_to_threshold`/`format_iters`、`_savefig`、`_markdown_table`、OOPAO 5 助手） |

**🔴 与 TODO 记载不符之处（实测，据此改了做法）**：TODO 说 `_fmt` 只有两份且
"统一取 gsnet 行为视为修复"。实测 `_fmt` 有 **5 份**，而且**只有 2 份真的该合并**：

| 脚本 | 原 `_fmt` 语义 | 处置 |
|---|---|---|
| `generate_gsnet_offline_report` | `np.isnan` + 极小值走科学计数 | **基准** → `fmt_metric` |
| `generate_fouriergsnet_sim_report` | `math.isnan`，**无极小值分支** ⇒ `1e-7` 渲染成 `0.0000` | 合到 `fmt_metric`（这就是 TODO 指定的那处修复） |
| `generate_oopao_{vs_numpy,impact}_report` | `0`→`"0"`；`<1e-3` 或 `>=1e5` 走科学计数；默认 6 位 | 独立保留 → `fmt_ratio` |
| `generate_slm_pib_online_report` | `format(float(v), ".4g")`，**不特判 None/NaN** | 独立保留 → `fmt_general` |
| `generate_slm_pib_rms_pib_report` | `f"{v:.4f}" if v==v else "—"`，**None 会抛 TypeError**、NaN 用破折号 | 独立保留 → `fmt_fixed` |
| `generate_shape_objective_comparison` | `format(v, "+.4f")`，非有限值 → `"n/a"` | 独立保留 → `fmt_signed` |

**为什么不能一股脑合并**（这是本项最关键的判断）：把后 4 份塞进 `fmt_metric` 会
**静默改写已经提交的报告**。实测证据：170 组「旧实现 vs 新实现」探针，
按 TODO 原方案统一后 **28 组不同**，例如 `0.5`→`0.5000`、`1e+05`→`100000`、
`nan`→`-`（旧为 `"nan"`/`"—"`，甚至直接 `TypeError`）。
改成 5 个具名助手后复测：**170 组里 168 组逐字节相同，唯一 2 组差异就是
TODO 指定的那处极小值修复**（`0.0000` → `1.000e-07`）。

**顺带发现并修掉**：`generate_shape_objective_comparison.py` 带着一个 **UTF-8 BOM**
（HEAD 里没有，工作区有），任何用纯 `utf-8` 解码的工具都会
`SyntaxError: invalid non-printable character U+FEFF`。已剥掉，并让新测试用
`utf-8-sig` 读取，这样即使将来又混入 BOM，扫描也不会被它绊倒。

**测试**（`tests/ao_shaping/scripts/`，214 passed）：
- `test_common_helpers.py`：9 个助手逐分支钉死，期望值全部**从迁移前的实现里实测**
  记录（不是照着新代码抄的）；其中 `fmt_metric` 的 2 组差异单独写成"修复"断言。
- `test_common_helpers_not_reintroduced.py`：① 扫描 `scripts/*.py`，任何
  `def _fmt` / `_savefig` / `_markdown_table` / `iters_to_threshold` / `format_iters`
  **重新出现即失败**；② 10 个生成器必须 `from scripts._common import ...`；
  ③ 每个都要有仓库根 bootstrap；④ `scripts/__init__.py` 必须存在；
  ⑤ **import 每一个生成器的模块作用域**（证明导入链真的通）。

⚠️ **写这个守卫时踩过一次坑并已纠正**：最初用 `python scripts/<name>.py --help`
当冒烟测试，结果这些生成器**大多根本没有 argparse**，于是 `--help` 被忽略、
**整个报告真的跑了起来**，把 `docs/heuristic_pib/*.png`、`docs/wfs/*` 覆盖，
还新建了 `docs/slm_differential_shaping/`。已全部回滚
（`git checkout docs/heuristic_pib docs/wfs` + 删除误建目录），
测试改成"import 模块而不执行"。**教训：这些生成器的"无参即运行"必须当成副作用对待。**

**回归**：`tests/ao_shaping/scripts` 214 passed。

### 5.7 R-22（第 3 批）—— 2026-10-03 完成

| 原项 | 落地情况 |
|---|---|
| **R-22** `utils/hardware_utils.py` → `utils/image/hardware_utils.py` 迁移收尾 | ✅ **先做了 TODO 要求的 grep**（见下方"前置核查"），然后把**全部 11 处旧路径导入**改到 canonical 路径并**删掉 16 行 `sys.modules` 别名**（`src/ao_shaping/utils/hardware_utils.py`）。6 个 src：`beam_shaping_utils` / `differentiable_beam` / `micro_dm_image_collect` / `phase_capture` / `slm_diagnose` / `slm_lut_runner`；3 个 test：`test_slm_diagnose_hardware` / `test_auto_camera_finders` / `test_hardware_utils`。**不留兼容 shim** —— 旧路径现在 `ImportError` |

**前置核查（TODO 原文警告的"各有两个家，移动会静默撕裂"）**：实测结论与 TODO 的
担忧**不一致**，据此调整了做法：

| TODO 的说法 | 实测 |
|---|---|
| `flat_gray` 有两个家 | ✅ 确实有：`tools/slm/slm_zernike_common.py:352` 与 `utils/slm_phase.py:12`。但**两个都不在 `hardware_utils.py` 里**（该模块根本不导出 `flat_gray`），所以移动 `hardware_utils` 不会牵动它们。这属于 R-28 的去重范围，未在本项处理 |
| `um_to_waves` 有两个家 | ❌ **只有一个**：`utils/wavefront/zernike_utils.py:93`（`UM_TO_WAVES` 常量也在同处）。TODO 记录已过期 |
| 迁移"半途（6 旧 / 7 新）"，身份被 `assert legacy is canonical` 锁定 | ✅ 属实，但那条断言的**理由已经不成立**：别名 docstring 说"测试通过旧名重置 `_frames_dir` / `_frame_counter`，所以旧路径必须解析到同一模块对象"。一旦所有导入方都走 canonical 路径，`import ao_shaping.utils.image.hardware_utils` 本身就绑定**同一个模块对象**，全局照样改到真身上 ⇒ 别名存在的唯一理由消失 |

**测试**：`test_image_subpackage.py` 整份重写，从"断言别名是同一个对象"翻成
**"断言别名不存在"**：canonical 可导入 / 旧路径 `ImportError` / **物理文件已删** /
静态守卫"任何 `src/**.py` 都不许再出现旧路径"。`test_hardware_utils.py` 的
`from ao_shaping.utils import hardware_utils` 改为
`from ao_shaping.utils.image import hardware_utils`（否则删掉子模块后
`from ao_shaping.utils import hardware_utils` 不再有任何东西绑定该属性）。

**变异验证**：① 把别名文件重新写回去 ⇒ 2 条失败（`DID NOT RAISE` + 文件仍在）；
② 把某个 src 文件的导入改回旧路径 ⇒ 静态守卫失败。

**回归**：`utils` 800 passed；`algorithm`+`tools`+`model` 1010 passed；
`wfless`+`runners`+`scripts` 1040 passed（只剩 §5.14 那个既有失败）。

### 5.8 R-21（第 3 批）—— 2026-10-03 完成

| 原项 | 落地情况 |
|---|---|
| **R-21** `utils/image/gs_visualization.py`（390 行）→ `display/` | ✅ 物理移动到 `src/ao_shaping/display/gs_visualization.py`，并从 `ao_shaping.display` 正式导出（`GSVizCallback` / `create_gs_iteration_frame` / `save_frames_as_gif` / `render_gs_animation` / `gerchberg_saxton_with_visualization`）。**不留兼容 shim** —— 旧路径 `ao_shaping.utils.image.gs_visualization` 现在直接 `ImportError`，并有测试钉住这一点（留 shim 就等于又造两个家）。唯一导入方 `tests/ao_shaping/algorithm/test_gs_viz.py` 与 `README_GS_VIZ.md` 的示例路径已同步。`utils/__init__.py` 的包 docstring 也不再把 `gs_visualization` 列在 `image` 下。同步更新了 `AGENTS.md` 的 "Pygame/viz code inside utils/" 反模式行（该行原本同时点名 `utils/image/display.py`，**那一半仍开放**）与 `docs/issues_report.md` §10 的路径漂移记录 |

**顺带核实**（不是本项要求，但确认了原 TODO 的一句话）：`gerchberg_saxton_with_visualization`
**确实是驱动 canonical 循环而非复刻** —— 它在函数体内延迟
`from ao_shaping.algorithm.signal_processing.gerchberg_saxton import ...`，
`display/` 模块作用域对 `algorithm/` 零依赖。已写成两条测试钉住（延迟导入存在 +
模块作用域无 `algorithm` import），否则渲染层与算法层会静默分叉。

**测试**（`tests/ao_shaping/display/test_display_layering.py`，10 passed）：
文件确实在 `display/`、**旧位置没有残留副本**、`display.__all__` 导出齐全、
旧路径 `ImportError`、docstring 自述归属、canonical GS 被驱动、
模块作用域无 `algorithm` 导入。

### 5.9 R-36（第 3 批）—— 2026-10-03 完成

| 原项 | 落地情况 |
|---|---|
| **R-36** Step 1：`runner_common.py` 的 click 机制 → 新叶子模块（**禁止任何 `ao_shaping.*` 导入**） | ✅ 抽出 `src/ao_shaping/utils/cli_params.py`（203 行），`runner_common.py` 减 **212 行**（2241 → 2030）。**零 `ao_shaping` 导入**，机制段只依赖 stdlib + `click` |

**为什么必须真抽出去、而不是留个 re-export**：`tools/slm/params.py`（R-37）要和
`runners/` 用**同一套** `Annotated[..., option(...)]` 约定。若机制留在
`runners/runner_common.py`，`tools/` 就得 import `runners/` —— 而 `runners/`
在 import 期就会拉起 `drivers/`（`DM_TYPES = list_dm_types()` 还要
`import ao_shaping.drivers.dm.asyn_micro_dm` 触发注册副作用）。那会让
「台架探针」在**没有设备**的导入路径上就碰硬件包，正好违反 R-20 刚立的
惰性契约。

**不变式写进了 docstring，并由 AST 测试兜住**（不是 grep）：允许的 import 只有
stdlib + `click`。三条守卫覆盖不同退化方式：

| 测试 | 挡住什么 |
|---|---|
| `test_leaf_imports_nothing_from_ao_shaping` | 直接 `from ao_shaping.x import y` |
| `test_leaf_imports_only_stdlib_and_click` | 任何新的第三方依赖（须先在 docstring 记理由） |
| `test_no_deferred_ao_shaping_import_can_hide_in_a_function` | **函数体内的延迟 import** —— grep 看不见，AST 看得见 |
| `test_mechanism_is_not_duplicated_anywhere_else` | 在 `runners/` 或 `tools/` 里**重新定义** `with_params` / `option` / `_DelayedCall` … |

最后一条是**变异测试**：已注入一次 `from ao_shaping.config import …` 验证，
4 个测试同时失败（其中 3 个正是上面三条）。第二条的判据特意区分「import」与
「定义」—— 12 个 runner import `with_params` 是正常消费，不算重复。

**回归**（R-35 golden 是本项的安全网）：

| 项 | 结果 |
|---|---|
| `tests/ao_shaping/runners/cli_help_golden.json` | **逐字节未变**（19 个命令的 `--help` 全文） |
| `tests/ao_shaping/runners/test_cli_contract_freeze.py` | 38 passed |
| `tests/ao_shaping/utils/test_cli_params_leaf.py`（新增） | 8 passed |
| runners + utils + tools + optimizer + display + scripts + model + algorithm + gui | **3400 passed / 56 skipped / 1 既有失败** |

**顺带修掉的两个真问题**：

1. **`get_type_hints` 导入来源搞错了**（我第一版写成 `from dataclasses import
   get_type_hints`）。`ruff` **不报**——它只查「导入了但没用」，查不出「从错误的
   模块导入」。是 import 时的 `ImportError` 抓到的。这正好说明为什么 R-36 需要
   自己的测试而不是只靠 lint。
2. **私有名不该跨模块 re-export**：`_collect_click_annotations` 原本被
   `tests/.../test_gsnet_train.py` 从 `runner_common` 取。已让测试直接 import
   叶子模块，`runner_common` 不再导出私有名。

⚠️ **顺带发现，未修**（不在 R-36 范围，单独记）：
`src/ao_shaping/optimizer/wfless/gready_cam.py:38`
`np.loadtxt('data\dm_adj.txt')` —— `'\d'` 是非法转义（`SyntaxWarning`），
按AGENTS.md 的「不要把 `\\U`/`\\u` 路径粘进代码」同族。这是模块级执行、
且依赖 CWD 相对路径，属于 H-* 硬件/环境问题，留待定。

### 5.10 R-28（第 3 批）—— 2026-10-03 完成

| 原项 | 落地情况 |
|---|---|
| **R-28** D1 日期格式化（2+2 份逐字节相同） | ✅ 4 个公开名全部保留，实现收敛到 `utils/io/timestamp.py` 一个 |
| **R-28** D2 日期目录 | ✅ `gen_date_dir` 加 `fmt` 参数，`create_save_dir` 复用它，**但两者粒度故意不同** |
| **R-28** D3 argmax→(x,y) 光斑定位（11 处 / 9 份内联） | ⚠️ **判定不可合并**，改为钉特征测试 |
| **R-28** D5 `scipy.zoom` vs 手写双线性 | ⚠️ **判定不可合并**，改为钉特征测试 |
| **R-28** D4 max 归一化（6+ 处） | ⚠️ **判定不可合并** —— 实测 3 种行为，见下 |

#### D1/D2 —— 合并了，因为能证明行为一致

`utils/io/timestamp.py` 现在是唯一实现（stdlib-only，不 import 任何 `ao_shaping`）。
**刻意没有**放在 `utils/io/file.py`：那个模块 import pandas + matplotlib，而
`cli_helpers` 被一堆 runner import，走它会把两个重依赖拖进每一条路径。

实测 4 份逐字节相同：

| 重复 | 收敛到 |
|---|---|
| `cli_helpers.get_timestamp_str()` == `file.gen_date_str()` | `format_ts()` |
| `cli_helpers.get_date_dir_name()` == `file.py:140` == `file.py:167`（两处内联 `%Y%m%d`） | `format_ts(fmt=DATE_FMT)` |
| `cli_helpers.create_save_dir()` == `file.gen_date_dir()` | `make_date_dir()` |

⚠️ **D2 的两个目录函数粒度不同，这是实测结论不是遗漏**：

| | 产物 | 语义 |
|---|---|---|
| `gen_date_dir(base)` | `base/<YYYYMMDD_HHMMSS>` | **每轮一个目录** |
| `create_save_dir(base, subdir)` | `base/subdir/<YYYYMMDD>` | **每天一个目录** |

合并要么把同一天的多次 run 挤进同一目录，要么重命名所有既有产物树。所以
`gen_date_dir` 加了 `fmt` 参数、`create_save_dir` 传 `DATE_FMT`，**结构共享、
行为逐字节不变**（两个方向的幂等性、递归建父目录都单独钉了）。

4 个公开名一个都没删：`utils.__all__` 必须保持 85 项（R-20 的测试硬断言），
且 `gen_date_str` 就有 6 个调用点。去重的断言写成「任一模块里不得再出现裸
`strftime`」，而不是「两个函数今天返回相同字符串」—— 后者拦不住第二份实现。

#### D3 —— 判定**不可**合并（暗帧语义分叉）

四种「最亮像素 → 坐标」的实现，前三个是同一函数的三种后端（200 组随机帧 +
高斯光斑**逐位相同**，两者都返回 Python `int` 而非 numpy `int`）：

```
center_of_brightness        numpy  -> (x, y)  裸 argmax
center_of_brightness_cupy   cupy   -> (x, y)  裸 argmax
center_of_brightness_numba  numba  -> (x, y)  手写算术
zero_order_center           numpy  -> (x, y)  + 暗帧守卫
```

第四个**不可互换**，差别就是全部意义所在。暗帧上：

```
center_of_brightness(np.zeros((5, 5)))  ->  (0, 0)   左上角
zero_order_center(np.zeros((5, 5)))     ->  (2, 2)   (w//2, h//2)
```

裸 `argmax` 在全零帧返回 0，于是「中心」变成角落。这正是本仓台架笔记那条
「暗帧不可用裸 argmax 定位光斑」的由来，也是 `zero_order_center` 加守卫的原因。
两种行为现在都被钉死，**任何方向上的误合并都会失败**。

顺带钉住一个「读起来像 bug」的事实：`refine` 是**局部**质心，窗口
`±max(min(h,w)//20, 8)`，**不是**「找到真实峰值」—— 20 行外的光斑它看不见。

#### D5 —— 判定**不可**合并（实测差 O(peak)，非舍入）

随机 `[0,1)` 场上的实测：

| 比例 | `max\|A - B\|` | `max\|A - C\|` |
|---|---|---|
| 64→32 | 2.2e-16 | 4.5e-01 |
| 250→50 | **0.0** | 9.0e-01 |
| 248→64 | 2.2e-16 | 8.4e-01 |
| 1200→64 | 3.3e-16 | 8.5e-01 |
| 64→248 | **3.8e-01** | 9.8e-01 |
| 37→111 | **3.5e-01** | 4.0e-01 |

A = `target/ccd.py` 手写 `_resize_bilinear`，B = `zoom(order=1, grid_mode=True,
mode="grid-constant")`，C = `zoom(order=1)`（scipy 默认，也是本仓**多数**调用点）。

- A 与 B **降采样**时只到机器精度（2e-16~3e-16），**只有 5:1 那档逐位相同**
  ——那是唯一两套坐标映射都落在整数像素上的比例。
- A 与 B **升采样**时差 0.35~0.38 peak，而**升采样正是 SLM 面板的路径**。
- 两者都不等于 C。

约定差异用**线性斜坡**读出（任何正确插值都精确复现仿射函数，所以返回值就是
它取的源坐标）：

```
250 -> 50, 输出索引 0 处的值
  A  2.0    半像素中心
  B  2.0    与 A 相同
  C  0.0    align-corners
```

即 **scipy 默认是那个异类**，而默认恰是本仓多数。合并任一组合都会让每个 SLM
面板像素、每个远场裁剪块移动最多 90% peak。测试断言的是**发散**而不是相等，
失败信息里写明原因，让下一次尝试**响亮地失败**而不是悄悄改掉光学。

另记两种行为以备后查：`gsnet_offline` 的 block mean 是**故意**的低通（噪声场上
与双线性差 ~0.43 全量程）；`phase_wrap` / `dynamic_compensation` 用的 `order=3`
三次样条会振铃到零以下（硬边上 min = **-0.198**），而相位没有这个量纲。

**修正自己一次错误结论**：中途我一度把 scipy 的 `grid_mode` 语义记反了，是
8→4 的手算探针纠正的。斜坡探针是自验证的（仿射场），所以结论不依赖对 scipy
内部实现的推测。


#### D4 —— 判定**不可**合并（暗帧 NaN + 三种行为）

TODO 说「6+ 处」，实测**正好 6 处** `x / x.max()`（`gui/slm/pattern_controls.py`、
`drivers/sim/fouriergsnet_env.py`、`drivers/sim/slm_shaping_bench.py` ×4），
同时仓库里**已经有 2 个 canonical helper** 而这 6 处都没用。三者按输入实测：

| 输入 | `x / x.max()`（6 处） | `normalize_pattern`（peak，默认） | `normalize_01`（min-max） |
|---|---|---|---|
| 全零暗帧 | **NaN + RuntimeWarning** | 0.0 | 0.0 |
| 峰值为负 | 1.0 | **-1.0**（原样返回） | 0.0 |
| 含 NaN | NaN | 有限（先 `nan_to_num`） | **NaN 传播** |
| 常量 5.0 | 1.0 | 1.0 | **全零** |
| int32 输入 | → float64 | → float32 | → float64 |

**第一行是要害**：暗帧除以自身最大值 = 0/0，所以那 6 处恰好在本仓台架笔记
明令「不可信任」的输入上**制造 NaN**，而 `normalize_pattern` 返回干净的零。
这是**性质不同**，不是舍入。

**第二行是 helper 自身的契约矛盾**：`normalize_pattern` 文档写「归一化到 [0,1]」，
但峰值为非正时**原样返回** —— 既没归一化也不在 [0,1] 内。作为守卫是合理的，
作为文档是错的。

**顺带修掉一个说谎的 docstring**：`normalize_01` 声称「委托至
`normalize_pattern`」。它**没有**委托，是内联重写的 min-max；而
`normalize_pattern` 默认 mode 是 `"peak"`，所以**真委托也会算出另一个函数**。
两处都错。已替换为实测对照表 —— 因为这个差别从签名看不出来：
`[[10,11],[12,14]]` 在 `normalize_01` 下拉伸到 `[0,1]`，在
`normalize_pattern` 下 `min` 仍是 0.71。

另钉住 `normalize_pattern` 两个 mode 是**不同目标**而非优劣之分：常量数组下
`peak` 给 1.0、`sum` 给 1/16，所以合并必须**选一个**而不能取平均。

**survey agent 跑了 2h33m 未收敛，已 cancel**，D4 由我自己用定向 grep + 实测完成。
（教训：这类「数一数有几处重复」的任务，(a) 本身就很小，(b) 探索型 agent 容易
在大范围 grep 上打转，不如给明确 pattern 自己查。）

**顺带发现，未修（归入 R-32）**：
`tests/ao_shaping/drivers/wfs/test_wfs_report.py` 是个**会写已提交报告**的测试
（`docs/wfs/wfs_report.md` + `001_simulated_wfs.png`）。跑到它就会弄脏工作树 ——
本次已 `git checkout` 回滚。与 R-25 的 `--help` 陷阱同属一类副作用。

### 5.11 R-27（第 3 批）—— 2026-10-03 完成

| 原项 | 落地情况 |
|---|---|
| **R-27** `runner_common.py` 的 CLI 机制 → 新叶子模块（零 `ao_shaping` 导入） | ✅ **已在 R-36 完成**（§5.9），此处只补路径差异说明 |
| **R-27** `runners/__init__.py` 改真 lazy | ✅ 删掉 15 行 eager import，`__getattr__` 从死代码变成真路径 |

#### 路径与 TODO 不一致（据实调整）

TODO 写的是 `utils/io/cli_params.py`，实际落地在 **`utils/cli_params.py`**。
理由：`utils/io/` 的定位是「I/O 与配置」，而这个模块是 CLI **参数绑定机制**，
塞进去会让 `io/` 的职责漂移；且它 stdlib-only + click，放根级更直白。
R-36 的测试锁的是「零 `ao_shaping` 导入」这个**不变式**，不是路径，所以换位置
不影响已验证的性质。

#### `runners/__init__.py` 的 docstring 原本在说谎

它写明自己存在的理由是「让每个 runner 能 `python -m` 直接跑而**不触发**
`RuntimeWarning: found in sys.modules`」。但文件顶部有 **15 行 eager import**
（`from ... import run as ...`），于是每个名字在 `__getattr__` 可能被调用之前
就已经进了 `globals()` —— **惰性路径是不可达的死代码，而它承诺避免的告警一直在**：

```
RuntimeWarning: 'ao_shaping.runners.slm_pib_runner' found in sys.modules after
import of package 'ao_shaping.runners', but prior to execution of
'ao_shaping.runners.slm_pib_runner'; this may result in unpredictable behaviour
```

删掉 eager 块后的实测（`import ao_shaping.runners`）：

| | `ao_shaping` 模块数 |
|---|---|
| 裸 `import ao_shaping` | 128 |
| `import ao_shaping.runners` **修复前** | **172** |
| `import ao_shaping.runners` **修复后** | **129** |
| `import ao_shaping.utils.cli_params`（R-36 叶子，基准） | 129 |

即现在只比裸包多 1 个，而不是自称的 0。`python -m` 在三种入口形态下
（含两个在子包里的）全部 **0 条 RuntimeWarning**。

#### ⚠️ 删掉 eager import 后立刻暴露一个它一直在掩盖的 bug

`slm_gsnet_run` **既不在 `__all__` 也不在 `_LAZY_RUNNERS`**，但 `main.py` 会
`from ao_shaping.runners import slm_gsnet_run` —— 它能解析**只因为**第 30 行
eager import 顺手带了它。也就是说 `slm-gsnet` 命令的注册依赖的是一个**巧合**，
不是一份声明。

**所以新测试没有停在「自洽」上**。我先写了"`__all__` == `_LAZY_RUNNERS`"这条不变式，
它**通过了**，而 `main.py` 离崩溃只差一行 —— 因为漏掉的名字**两个列表里都没有**。
于是补了 `test_every_name_main_imports_is_resolvable`：**解析 `main.py` 真实的
import 列表**，要求其中每个名字都被导出、被映射、且真的能取到可调用对象。
这才是能抓住本次 bug 的检查。

另外钉住：`globals()[name] = value` 的回写（没有它每次访问都会重入
`__getattr__` 重新 import）；以及一条 AST 断言 —— 模块作用域不得 import 任何
runner，让 eager 块无法悄悄回来。

**回归**：R-35 的 `--help` golden **逐字节未变**（19 个命令），这才是本次的关键 ——
CLI 注册必须完全一致，而底下的 import 图变小。

### 5.12 R-32（第 3 批）—— 2026-10-03 完成

| 原项 | 落地情况 |
|---|---|
| **R-32** 孤儿检测（`src/` 里的 test 文件） | ✅ 当前 **0 个**（R-31 已删掉唯一那个） |
| **R-32** `python -m` 一致性 | ⚠️ 实测发现 **11 处失效路径**，已修 → §5.12.1 |
| **R-32** utils 分层守卫 | ✅ 模块作用域引用高层 **0 处**（14 处全在 `TYPE_CHECKING`/函数内） |
| **R-32** warn-only + baseline 起步 | ✅ 两条守卫零 baseline；另两条各需 2 / 1 条**带理由**的白名单 |

⚠️ **`warn-only + baseline` 的实际形态与 TODO 设想的不同**：实测发现树在「孤儿检测」
和「分层」两条上**本来就干净**，所以不需要 baseline —— 直接硬失败才是有用的形态。
另外两条各有 1~2 处**确实是文档正确**（见 §5.12.1），才需要白名单。TODO 预期
「先 warn-only 攒 baseline」的情况没有出现，因为 R-24/R-31 已经把前两项修掉了。

#### §5.12.1 `python -m` 一致性 —— 11 处失效路径（已修）

runners 重组进 `micro_drive/` 与 `slm/` 子包后，**文档里的模块路径没跟着改**：

| 文件 | 处数 | 失效路径 | 真实路径 |
|---|---|---|---|
| `micro_drive/alt_voltage_runner.py` | **5** | `runners.alt_voltage_runner` | `runners.micro_drive.alt_voltage_runner` |
| `micro_drive/full_voltage_runner.py` | **4** | `runners.full_voltage_runner` | `runners.micro_drive.full_voltage_runner` |
| `scripts/generate_zernike_response_matrix_report.py` | 1 | `runners.zernike_matrix_runner` | `runners.slm.zernike_matrix_runner` |
| `tools/slm/cartographer/__init__.py` | 1 | `ao_shaping.tools.slm.cartographer`（**包无 `__main__`**） | `...cartographer.slm_cartographer_ui` |

⚠️ **第一处最严重**：那条路径是被 `md.append(...)` **写进生成报告里**的，
所以源码改对了，**产物里的错路径还在**，用户照着复制仍然失败。

⚠️ **第四处是「文档形式上对、实际跑不了」**：包没有 `__main__.py` 就不可能被
`python -m` 执行；真正的入口是 `slm_cartographer_ui` 子模块。

**2 条合法不解析，走白名单（各带理由）**：
- `utils/io/cli_helpers.py` 里的 `python -m ao_shaping.runners...` 是散文式 glob；
- `scripts/generate_gsnet_offline_report.py` 明说 `gsnet_train` 是库模块、
  **该调用方式无效** —— 是**文档在正确地报错**。

**历史文档按路径排除**：带日期的日报、已封存的报告**本来就应该**记录当时的路径，
改它们等于篡改历史；`TODO.md` 引用失效路径是**故意的**（那是在描述这个缺陷）。

#### ⚠️ 写这条守卫时的过程失误（两条守卫互相抓到对方的 bug）

- 我先用 PowerShell `Select-String` 定位失效路径，它报的 `README.md:35` 与
  Python 扫描**行号与内容都对不上**（文件混合行尾）。**行数/内容不一致时以
  Python 扫描为准**，PowerShell 那次的输出基本不可用。
- 我给 `full_voltage` 等写的替换断言用了 `==` 精确匹配，结果 `alt_voltage_runner.py`
  实际有 **5 处**而非扫描看到的 4 处 —— 断言当场失败，比事后发现好。
- 写「docs 写入守卫」的第二条时抓到第一条的 bug：它把
  `TestReport("miicam", device_dir="docs/miicam_simulation")` 判成写
  `docs/miicam`，因为**显式 `device_dir` 覆盖已经决定目录了，设备名不该再兜底**。
  两条守卫现在互相一致，白名单才可信。

#### 第四条守卫：测试会写已提交报告（本轮新发现）

跑套件时 `docs/wfs/wfs_report.md` + `001_simulated_wfs.png` 被改写 —— 本轮已回滚
**两次**（R-25 一次、本轮一次）。根因比单个文件大：

- `tests/ao_shaping/utils/test_report.py::TestReport.__init__` 在**构造函数里**
  就 `mkdir` 并把目标指向 `docs/<device>/<device>_report.md` —— **早于任何 skip**；
- `hardware` marker 虽然声明了，但 `pyproject` 的 `addopts` **没有** `-m "not hardware"`。

⇒ 一次普通 `pytest` 会收集并执行**正是那些会改写已提交文件的测试**：
`docs/slm`(82 个已跟踪文件)、`docs/slm-200`(9)、`docs/wfs`(2)、`docs/miicam`(2)。

🔴 **刻意没有**给 `addopts` 加 `-m "not hardware"`：该 marker 覆盖 **8 个文件 158 个
测试函数**，其中可能确有无需设备即可通过者，默认 deselect 有**静默削减覆盖率**的
风险。这是拥有该套件的人的决策，不该由一个清理任务顺手改掉。

改为强制一条更窄的不变式：**写已跟踪 docs 目录的测试必须带 `hardware` marker**，
以便可被过滤。`test_miicam_simulation_report.py` 白名单 + 理由（目标是
**未被 git 跟踪**的 `docs/miicam_simulation/`，且不需要设备）。

### 5.13 F-14 / F-15（第 3 批）—— 2026-10-04 完成

| 原项 | 落地情况 |
|---|---|
| **F-14** 报告生成写在 `algorithm/` 层 | ✅ 违反反模式「report generation MUST live in `scripts/`」 |
| **F-15** README 指向 1 行 smoke 残留，权威网格被 gitignore | ✅ 权威 9 单元网格已生成并提交，README 改指真产物 |

**这两项是同一个问题的两面**：F-14 是「谁写」，F-15 是「写到哪」。先修 F-14
（写出器搬到 `scripts/`），才使 F-15 可修 —— 因为权威网格必须能重新生成才有意义。

#### F-14：`algorithm/` 侧现在**完全没有写盘路径**

`beam_shaping_benchmark.py` 原有 3 个函数写 CSV/MD/GIF。现在
`run_benchmark` / `run_benchmark_suite` **不再接受 `output_dir`**，
`algorithm/` 里已搜不到 `to_csv` / `write_text` / `savefig` / `mkdir`。

三个**非 I/O** 的助手留在生产侧（它们是计算，不是序列化）：

| 助手 | 为什么留下 |
|---|---|
| `build_gif_frames(target, simulated)` | 只在内存里造 PIL 帧，不写文件 |
| `to_dataframe(rows)` | 它是 `run_benchmark_suite` **返回值**里的 DataFrame 投影，返回类型属于 API |
| `HPRINT_KEYS` | 列契约 |

序列化全部移到 `scripts/generate_beam_shaping_benchmark_report.py`；
`run_device_less_full.py` 因被文档引用而保留为转发壳。

#### F-15：权威产物从「不可用」变成「已提交」

| | 修前 | 修后 |
|---|---|---|
| README 指向 | `docs/beam_shaping_benchmark_metrics.md`（**1 行 smoke 残留**） | `docs/benchmarks/device_less_full/beam_shaping_benchmark_metrics.md`（**9 单元**） |
| 权威 9 行网格 | 被 `.gitignore` 排除，**本 checkout 不存在** | **已提交**（~100 KB，含 6 个 GIF） |

⚠️ 那个残留文件**自己就带着 2026-10-01 的警告横幅**描述了这个问题 ——
前一轮 review 诊断出来了但没动，只是加了横幅。已**删除**而不是留着：
一个只有一行表格、名字听起来很权威的文件比没有更糟；它的诊断内容现在在
生成报告的头部、本 commit message 和 git history 里。

🔴 **刻意没提交的**：同目录的 `.csv`。`.gitignore` 有**全局** `*.csv` 规则
（`:7`），且**当前 `docs/` 下 0 个 CSV 被跟踪** —— 提交它们等于替全仓改政策，
超出 F-15 的授权。报告头部写明了 CSV 的去向与再生成方式。

#### 顺带纠正一个**不实的老说法**

旧 `run_device_less_full.py` docstring 写「gs/spgd-sim 在无设备下面积达标」。
实测**不是**：

| 算法 | 面积检查通过 |
|---|---|
| `gs` | **3/3** |
| `backprop` | **1/3**（只过 square） |
| `spgd-sim` | **0/3** |

`spgd-sim` 的实测面积塌到 **~1 px**、`uniformity_cv` **16–34**，无设备下它的
均匀性/能量数字描述的是噪声。这句话已写进**生成报告的头部**，这样表格单独被
读到时也不会被误读；并明确要求**先看 `area_met` 再比较任意两行**。
没有为了让报告好看而调整任何 CSV/GIF。

**新增第 5 条约定守卫**：`algorithm/` 下不得出现
`to_csv` / `write_text` / `savefig` / `mkdir` / `.save(`，且
`beam_shaping_benchmark` 不得再出现 `output_dir`。已用植入 `to_csv` 变异验证。

### 5.14 全量套件基线（2026-10-03 实测，非本轮引入）

按目录分块跑（`tests/ao_shaping`），**单块崩潰不影响其余块计数**：

| 目录 | 结果 |
|---|---|
| `algorithm` / `model` / `optimizer` / `tools` / `utils` / `gui` / `scripts` | ✅ **2945 passed**，0 failed |
| `drivers/ccd` | 138 passed，**6 failed** |
| `drivers/dm` | 134 passed，**5 failed** |
| `drivers/sim` | 209 passed，**5 failed** |
| `runners` | 529 passed，**1 failed** |
| `ml` | 409 passed，**3 failed** |
| `drivers/wfs` | 💥 原生 `access violation`（CPython 3.13 GC × tqdm monitor 线程，dump 停在 `Garbage-collecting`，非逻辑错误） |

**23 个失败全部是既有问题**，已用 `git stash push` 把本轮**全部**改动暂存后重跑同一组
文件验证：**基线恰好也是这 23 个、一模一样**。分类：

| 数量 | 根因 |
|---|---|
| 6（`daheng/test_reset_window.py`） | `NameError: name 'gx' is not defined` —— `gxipy` 未安装（`libs/gxipy` 不在 `sys.path`） |
| 13（`dm/test_adjacency_loading` 5 + `sim/test_sim_camera_registration` 5 + `drivers/test_lazy_driver_loading` 3） | 这些用 `subprocess` 起新解释器验"导入与 CWD 无关"，本机 Winsock 坏了：`import asyncio` → `_overlapped` → `OSError: [WinError 10106]` |
| 1（`runners/test_slm_pib_runner_debug.py::test_debug_artifacts_pkl_exports_array_fields`） | 断言 `_phase` 不在导出键集里，但实际在（`_save_debug_artifacts` 未剔 `_phase`） |
| 3（`ml/zernike/test_metrics.py`） | `assert nan == 1.0` —— 落在**未提交的 `src/ml/` WIP** 里（`hwdataset`/`ZernikeAmpModel` 那批），非本轮范围 |

⚠️ 全量单进程跑会在 ~32% 崩（同一个 GC/tqdm 问题），**按目录分块跑即可完整计数**；
`gui` 单独跑 191 passed，但混在全量里会触发那次崩溃。

---

### 5.15 R-37 —— 「抽共享 dataclass」被实测证伪（2026-10-04）

**结论先行：R-37 原设计的 Step 2/3（把 recurring flag 抽成 `tools/slm/params.py` 共享 dataclass）
在物理上不可实现，已取消。**改为**只迁移声明机制**——每个探针自带 dataclass，
`default` / `help` / `type` 全部留在本地不动。

#### 实测数据（R-37 step 0，19 个可执行探针）

| 项 | 实测值 | TODO 原记录 |
|---|---|---|
| 可执行探针 | **19** | 22 |
| Click / argparse | **15 / 4** | 15 / 1 |
| 声明 flag 总数 | **287** | ~200 |
| 用 `with_params` 的 | **0** | 0 |
| 直接构造 `Santec(...)` | 20 处 | 17 文件 / 18 处 |

#### 为什么共享不可行（两条独立的硬约束）

**① `cli_params` 的 dataclass 字段是 CLI default 的唯一来源**（`utils/cli_params.py:142`
`_patch_defaults`：`option()` 里传 `default=` 直接 `raise TypeError`）。
⇒ 共享 dataclass 每个字段**只能有一个 default**，没有 per-consumer override 机制。

**② `help=` 也写在 `option()` 里**，同样只能有一份 ⇒ 共享组还会强制统一 help 文本。

于是逐一核对「同名 flag 是否 type+default+help 三者全同」：

| flag | 出现探针数 | 不同签名数 | default 分布 |
|---|---|---|---|
| `--slm-wavelength` | 15 | **6** | **1064（11 个）vs 532（6 个）** |
| `--exposure-ms` | 15 | **12** | 0.02 / 0.03 / 0.4 / 0.8 / 1.0 / 1.2 / 2.0 / 3.0 … |
| `--slm-number` | 18 | 6 | 全是 `1`（差在 `type=` 与中英文 help） |
| `--output` | 8 | 8 | 4 个不同目录 |
| `--zernike-radius` | 6 | 6 | 300 / 450 / 600 / `BEAM_RADIUS_PANEL` / `None` |
| `--settle-extra-s` | 3 | **1** | ✅ **唯一真正可共享的** |

**全包 287 个 flag 里只有 `--settle-extra-s` 一个能原样共享。**

#### 「按物理台架拆分再共享」也救不了（第二轮实测）

先按 `--slm-wavelength` 把探针分成两台架再核对：

| 台架 | 探针数 | 可共享 | 被阻塞 |
|---|---|---|---|
| 1064 nm（远场 CCD） | 13 | **0** | 28 |
| 532 nm（WFS 通道） | 5 | **1** | 12 |
| 无该 flag（`slm_exposure_check`） | 1 | 0 | 0 |

阻塞项**主要不是 default，而是 help 文本与 `show_default` 漂移**：
`--slm-number` ×13 default 全是 `1` 却有 5 种签名；`--period-ref` ×2 default 全是 `'64'` 仍有 2 种签名。

#### 真正的性质：这不是「重复逻辑」，是「各自标定」

532 nm 那 6 个探针恰好就是 WFS 通道工具
（`calibration` / `slm_wfs_probe` / `slm_wfs_reference` / `slm_zernike_correction` /
`slm_zernike_response`，外加 `slm_zernike_sweep_probe` 用 1064）——
**波长不是随手写的默认值，而是台架的物理属性**。
`--exposure-ms` 的 12 种取值同理：每个探针的曝光是对着自己那台架单独标定的。

⚠️ **如果按原计划抽共享组，flag 名守卫生效、物理默认值被改写**：
WFS 探针被静默改成 1064（或反之），而这种改动**不会触发任何测试**——
它长得像一个纯重构。建 `Slm532Params` / `Slm1064Params` 只会得到一个**没有使用者的抽象**。

#### Step 0 护栏（本次真正的产出，先于任何迁移落地）

`tests/ao_shaping/tools/slm/test_probe_flags.py`（41 passed, **0.20 s**）
+ `probe_help_golden.json`：

* **AST 扫描器必须同时认得两种声明形态**，否则它会在每个「已迁移」文件上误报：
  - 迁移前：`@click.option("--x", type=int, default=1, help=...)`
  - 迁移后：`x: Annotated[int, option("--x", help=...)] = 1`
  实现上按**被调用名**匹配（`option` / `add_argument`），不看语法位置。
  ⚠️ 只认装饰器形态的扫描器会在迁移后报告「该探针一个 flag 都没有了」。
* 第三种形态也要认：`calibration.py:2711` 用
  `@click.command(context_settings=dict(help_option_names=["-h", "--help"]))`
  覆盖 click 内建 help —— 注意 `help_option_names` 挂在**内层 `dict(...)`** 上，
  不是 `click.command(...)` 上，按 callee 名匹配会漏。
* click 的配对布尔 `"--display/--no-display"` 是**一个字符串两个 flag**，
  必须按 `/` 拆开各自计数，否则重写成 `"--display", "--no-display"` 会「看起来没变」。
* **为什么默认守卫不比对渲染后的 `--help`**：19 个子进程各付一次 ~28 s
  `import ao_shaping` ⇒ **549 s**。这个价位的守卫只会被关掉。
  渲染文本留在 golden 里，用 `AO_PROBE_HELP_UPDATE=1` 走 opt-in 深检，
  且**只允许顺序变化**（click 的 `__click_params__` 是逆序累积，R-35 已记录同一效应）。
* 变异测试：植入一个多余 flag / 删掉一个已迁移的 flag，守卫均如期失败。

**首例迁移 `slm_exposure_check.py`（5 flag）的 `--help` 逐字节未变。**

#### 落地结果：15/15 个 Click 探针，`--help` 逐字节未变

全部 15 个 Click 探针改完，**每一个的 `--help` 与迁移前逐字节相同**（不只是 flag 集合相同）。
`calibration.py` 最险：它有**两个命令**、26 个 flag、16 个参数、
62 处引用分布在 228 行 body 里，其中 `-o, --output` 与 `--help` 覆盖都保留。

**三道守卫**（`tests/ao_shaping/tools/slm/test_probe_flags.py`，**79 passed / 0.44 s**）：

| 守卫 | 抓什么 | 为什么 flag 名守卫生效不了 |
|---|---|---|
| ① 声明 flag 集合（AST） | flag 丢了/改名/重复/两种拼写 | — |
| ② **函数体内 string literal 计数 + sha256** | dict key / Recorder kwarg / log 格式被改写 | `{"csv_path": params.csv_path}` 仍是合法 Python，flag 名一个没少 |
| ③ **`option()` 内禁止 `default=`**（静态） | import 期 `TypeError` | 声明全都正确、flag 名一个不少，只有 `--help` 跑不起来 |

守卫 ② 和 ③ 都是**实测逼出来的**，不是预防性设计：

* ② 抓到 `slm_zernike_correction` 的 dict key `"pred_z_norm"` 被改成 `"_z_norm"`。
  只看 flag 名的守卫对这类损坏完全失明。
* ③ 抓到两个已"通过全部 flag 断言"的探针在 `option()` 里带了 `default=`，
  `cli_params._patch_defaults` 直接 `raise TypeError` ⇒ **模块 import 就崩**。
  这个错误只有跑 `--help` 才看得见，而一个要 28 s 子进程才报警的守卫，实践中只会被关掉。
  改成静态检查后 **0.44 s 就能抓**。

⚠️ **body 改写必须用 AST 位置、且按 UTF-8 字节算偏移**。两次踩坑：
`ast` 的 `col_offset` 是**字节**偏移，按字符数算会在含中文的行上错位
（`f"WFS 曝光 {wfs_exposure_ms}ms"` → `{wfs_params.wfs_exposure_ms}`）；
只删 `default=` 的**值**会留下 `default=`（`expected argument value expression`）；
删 `, default=None` 时若把「前一个元素」当成位置参数，会连带删掉 `-o` 而丢掉 `--output`。
**只有 `ast.Name` 节点才是真引用** —— kwarg 名 (`Santec(slm_number=...)`) 和
字符串字面量都不是，所以位置法天然避开它们，而字符串里的 `{...}` 插值**是** Name 节点、正该改。

#### 顺带修掉一个真实的坏 flag（不是迁移造成的）

`slm_zernike_sweep_probe.py` 声明了 `ap.add_argument("--save-frames/--no-save-frames", default=True)`。
**argparse 没有 `/` 语法**，所以它只注册了一个字面长选项：

| 命令 | 迁移前 | 删除后 |
|---|---|---|
| `--save-frames` | rc=2 `expected one argument` | rc=2 `unrecognized arguments` |
| `--no-save-frames` | rc=2 `unrecognized arguments` | 同左 |

且 `args.save_frames` **全模块从未被读取**（`args` 只按属性显式取用，没有 `**vars(args)`），
落盘是无条件的 `save_recorder_debug_artifacts(...)` ⇒ 这个 flag **什么都没控制**。
所以是**删掉**而不是"修好"：修好会造出一个声称能控制存帧、实际仍什么都不做的 flag。
两个拼法迁移前就都是 rc=2，**没有任何能用的命令被改变**。
flag 总数因此 287 → 285（help 可见 289 → 287）。

#### 已知遗留（非本轮引入，`HEAD` 里就有）

* `slm_zernike_response.py:311` **F541** f-string 无占位符
* `slm_drift_probe.py:518` **F601** 字典 key `"exposure_ms"` 重复 —— 这个像是真 bug，值得单独查

#### 全量回归（分块，单进程全量会在 ~32% 崩）

| 目录 | 结果 |
|---|---|
| `tests/ao_shaping/tools` | **452 passed** / 4 skipped |
| `tests/ao_shaping/utils` | **892 passed** |
| `tests/ao_shaping/runners` | 541 passed / 1 **既有失败** |
| `tests/ao_shaping/algorithm` | **561 passed** |
| `tests/ao_shaping/optimizer` | **581 passed** / 2 skipped |
| `tests/ao_shaping/model` / `display` / `gui` | 76 / 38 / 191 passed |
| `tests/ao_shaping/scripts` + `test_conventions.py` | **287 passed** |

⚠️ 混在一个进程里跑 `tests/ao_shaping` 根目录会出现 3～5 个失败，且**每次集合都不一样**
（735 vs 741 个用例）—— 这是 §5.14 已记录的跨用例干扰，不是本次改动：
`test_optimize_uniform_spot_phase.py` 只 import `ml.zernike.models` 与
`ao_shaping.utils.image.beam_metrics`（都不在本次 diff 内），单独跑 **23 passed**。

---

### 5.16 F-1 —— 补偿闭环把内存槽写死，迭代 2..N 全是固件 no-op（2026-10-04）

`compensate_once` 的循环里每一轮都调用
`apply_compensation(comp_gs, memory_slot=2)`（原 `:383`），而
`apply_compensation` 把它透传给 `display_data(..., memory_number=2)`。

**后果不是"少刷一次"，而是整个闭环失效**：固件把"对正在显示的同一槽再写"
视为 no-op —— 数据写进去了，**但 LCOS 不刷新**。于是第 2..N 轮量到的
**还是第 1 轮那幅画面**，只是叠上了新算出的相位数据。循环会照常跑完、
照常写 `per_iteration`、照常打印 `RMS 0.58 -> 0.50`，但那个改善里
没有任何一次来自面板真的换图。**它看起来完全正常。**

驱动其实已经在提醒这件事：`_display_memory` 检测到同一槽连写会
`logger.warning("...固件视为 no-op, LCOS 面板不会刷新...")`
（`driver.py:1140-1148`），所以旧代码跑起来日志里本该有告警。

**修法：不传 `memory_number`。** `display_data` 在 `memory_number is None`
时**自己轮换**，并且**额外跳过当前正在显示的槽**（`driver.py:1230-1249`），
所以 `open()` 之后第一次写也不会撞上别的进程留在屏上的图案。
这正是 `tools/slm/README.md` 写的台架纪律。

`apply_compensation` 的 `memory_slot` 参数**整个删掉**（全仓只有 `:383`
一个调用方，无外部依赖），并改为返回驱动选中的槽位号以便日志追踪。

#### 顺带纠正原记录里两条不成立的指控

* 「`:276` 漏 `memory_mode=MEMORY_MODE_INTERNAL`」—— **不成立**。
  `display_data(..., memory_mode: int = MEMORY_MODE_INTERNAL)`（`driver.py:1197`）
  本来就是默认值，不传是**对的**，传了反而多一个真相来源。
* 「`:272` docstring 写 "1-128" 与实际不符」—— **不成立**。
  `1-128` 与 `_display_memory` 的 `MEMORY_NUMBER_MIN/MAX` 校验区间一致。
  （真正需要区分的是**校验区间 1-128** 与**轮换建议区间 2-125** `SLOT_MIN/MAX`，
  两者不同，但这属于文档可以补一句，不是违规。）

#### 测试（纯离线，`DynamicCompensator` 的 slm/wfs 是注入的）

`tests/ao_shaping/tools/slm/test_cartographer_slot_rotation.py`，**7 passed**：
* `memory_number` 恒为 `None`（**回归本体**）
* `memory_mode` 也不传（锁住上面那条纠正）
* 返回值 == 驱动选中的槽
* SLM 未 open 时拒绝写入且不记录调用
* **3 轮迭代每一轮都不钉槽**，且 `iterations_used == 3`、`per_iteration_data == 3`
  —— 后者防止"迭代被悄悄丢掉"让前一条**空过**
* **相邻两次写入落在不同槽** —— 把 `memory_number is None` 重新挂回那个物理事实：
  固件对同槽写入 no-op，所以断言槽真的**移动了**

**变异验证**：把 `memory_number=2` 塞回去 → 3 个测试如期失败。

回归：`tests/ao_shaping/tools` + `test_conventions.py` **473 passed / 4 skipped**。

---

### 5.17 `slm_zernike_shaping.py` 副本盘点 + 修掉一处会静默归零的桶半径（2026-10-04）

#### (a) 副本的真实性质：**没有生产导入方**

全仓 grep（`from|import` 形式，非字符串）实测：

| 入口 | 实际用的模块 |
|---|---|
| `slm-pib`（`runners/slm_pib_runner.py:15`） | `wfless.slm_zernike_pib` ✅ |
| `delta_explorer`（`tools/slm/delta_explorer.py:49`） | `wfless.slm_zernike_pib` ✅ |
| `rms-zernike`（`runners/slm/rms_zernike_runner.py:12`） | **`wf.rms_by_zernike`** ❌ 不是 shaping |
| `slm_zernike_shaping` | **零生产导入方**（只有 3 个测试 + 自己的 legacy `__main__`） |

⚠️ `slm_zernike_shaping` 在仓库里出现 174 次，但**绝大多数是数据 family 名**
（`slm_zernike_shaping_rms_pib_<ts>.pkl`），不是模块引用。
X-3 说的"两者都是生产入口（`slm-pib` / `rms-zernike`）"**不成立**。

**这条改变了合并的风险评估**：既然 shaping 无生产入口，它可以按
`slm_shaping_bench.py` 的既有做法直接变成 **re-export shim**，
不会打断任何生产路径。

#### (b) 已逐字节核对的重叠（AST 提取函数体后程序化比对，非目测）

| 符号 | pib | shaping | 结论 |
|---|---|---|---|
| `_create_optimizer` | 183-191 | 170-178 | **逐字节相同** |
| `_default_camera` / `_default_slm` | 382-386 / 389-393 | 370-374 / 377-381 | **逐字节相同** |
| `_display` | 231-249 | 219-237 | **逐字节相同** |
| `learning_schedule` | 272-375 | 260-363 | **逐字节相同（104 行）** |
| `_zernike_to_phase` | 252-269 | 240-257 | **函数体逐字节相同，但读不同常量 ⇒ 行为不同** |
| `SlmZernikePibConfig` | 397-475 | 385-447 | 分叉（pib 多 5 个鲁棒字段，shaping 多 `debug`/`w_outside`） |
| `optimize_slm_zernike_pib` | 543-1596 | 517-1451 | **分叉 273 行** |

`_zernike_to_phase` 正是 `slm_shaping_bench.py` 那个回归的同一形状：
**函数体一样，行为由模块常量决定**。import 实测 pib=300.0 / shaping=600.0。

#### (c) 已修：桶半径静默归零（shaping，pib 早已正确）

`shaping:1363` 的收缩块把 **full-frame** 的 `center` 传给了窗口内图像的 `radius()`：

```python
power_radio = radius(pos_img, center=center, energy=0.8)   # 错
_pr      = power_radio * shrink_ratio        # → 0.0
r_bucket = min(_r, _pr, _init_r)             # → 0.0   桶半径归零
```

* `center` 在该作用域是 `reset_window` 返回的**全画幅**中心（`:737`），
  `pos_img` 是**重新开窗后**的帧 —— 正是 AGENTS.md 已登记的反模式
  「Full-frame centre leaking into window-local metrics」。
* `radius()` 对越界中心**不报错，直接返回 `0.0`**（实测 `(125,125)`→62.5，`(577,655)`→0.0），
  所以**没有任何异常**，`r_bucket` 直接归零，而 objective 之后读的就是这个 0。
* 同一文件在 `:844` 和 `:924` 已经写明"用 `reference_center`（窗口内），**不要**用 `center`"——
  收缩块是唯一漏掉的一处。
* pib 在同一位置用的是窗口内重定位的 `pos_center`（`:1413` 定义，`:1501` 使用），**一直是对的**。

**改为** `center=reference_center`（与同文件另外两处 `radius()` 调用一致）。

**为什么原有守卫没抓到**：`test_slm_zernike_pib_robustness.py` 的守卫
只遍历 **`learning_schedule` 调用节点内部的 `radius`**，而这一处在 epoch 循环里
**直接调用 `radius`**，不在任何 `learning_schedule` 里。
**已补一条守卫：遍历两个模块里所有 `radius(...)` 调用**，禁止裸 `center=`。
变异验证：塞回 `center=center` → 新守卫如期失败。
`test_slm_zernike_pib_robustness.py` **29 passed**；
`tests/ao_shaping/optimizer/wfless` **428 passed / 1 skipped**。

#### (d) 尚未处理（留待后续，避免与本轮混淆）

* `optimize_slm_zernike_pib` 两份分叉 273 行，**不能当一次 drop-in 合并**。
  其中 pib 独有：fold gate / noise gate / ABBA / 逐评估重定位中心 / `n_eval_frames`；
  shaping 独有：`debug` 产物与 `_debug_report`、`w_outside`（`rmse_out` 靠它）。
  建议形状：pib 为唯一实现，shaping 变 re-export，能力差异用 config 字段保留。
* `pib.py` 里可能还有**第三份** `learning_schedule`/`_create_optimizer`
  （`test_pib_helpers.py` 测的是它），**未核对**。
* `_display_shape` 的 objective 列表**已经漂移**：pib 缺 `"rmse_out"`
  ⇒ `rmse_out` 在 pib 里实际是失效的。**这是一个独立 bug，本轮未修。**

---

### 5.18 R-41 —— 两种 flag 拼法都保留，并加契约测试（2026-10-04）

R-37 的 golden 已经**隐含**钉住了所有 flag 名，所以 R-41 其实是"已经被覆盖"的。
但**隐式覆盖不够**：下一次有人做"顺手统一拼写"的重构时，
golden 会以"我只是想清理一下"的名义被重新生成，意图就此消失。

所以补了**显式契约测试**（`test_probe_flags.py`，现 **82 passed / 0.45 s**）：

* `--cam-type` **9** 个探针、`--camera-type` **2** 个（`slm_diagnose` / `slm_lut_runner`）
  —— 断言**两套都在**，且**没有任何探针同时暴露两套**
  （同时暴露是 CLI 分叉，不是别名，必须显式决定）。
* `--output` **8** vs `--out` **4**、`--slm-wavelength` **15** vs `--wavelength` **3**
  —— 同样断言。

计数写死在测试里，所以任何拼写侧的改动都会**先在这里失败**，
逼作者说明这是有意为之还是顺手改的。

---

### 5.19 R-38 实测后**拒做**：「采用率过低」本身不是缺陷（2026-10-04）

原条目把"canonical helper 采用率低"当成待修的问题。实测后**按原描述不动**，
因为两条前提都不成立。

#### (1) `phase_to_slm_grayscale` vs 直调 `create_phase_from_array`：行为等价

`utils/slm/phase_display.py:72-73`：

```python
if slm is not None:
    return slm.create_phase_from_array(phase, max_grayscale=max_grayscale)
```

传活跃 SLM 时**它就是**一次转发。全仓 17 个文件 46 处 `create_phase_from_array(`
里，**没有一处把 uint16 灰度图传进去**（逐个看过实参：`phase_rad` / `ramp_panel(...)`
/ Zernike 相位，都是弧度）⇒ **不存在"灰度值被当弧度静默损坏"这个风险**。
改写这 7 处纯属 churn。

#### (2) `zero_order_center` **不是** `np.argmax` 的 drop-in

`beam_metrics.py:414`：

* 返回 **`(x, y)`**（项目约定）；
* 默认 `refine=True`，会在 argmax 锚点邻域内**再做一次质心细化**。

而裸写法 `py, px = np.unravel_index(np.argmax(frame), frame.shape)` 得到 **`(y, x)`**。
**两者坐标序相反** —— 照抄会把每个 ROI 中心静默转置。
再叠加"质心细化会把坐标挪动几像素"，直接替换还会改变所有半径/ROI 数值。

#### (3) 逐处分类：多数"裸 argmax"是对的

`tools/slm/` 共 **18 处** argmax，按用途分三类：

| 类别 | 处数 | 该不该换 |
|---|---|---|
| **一维曲线峰值**（强度/效率/相位带、`argmax(eta)` 等） | ~10 | ❌ 本来就不是定位光斑，换了反而错 |
| **相关峰**（`slm_bench_probe.py:176` `argmax(corr)`） | 1 | ❌ 相关峰定位是另一个问题 |
| **二维光斑定位** | ~7 | ⚠️ 需逐处判断，**不能批量替换** |

其中 `slm_diagnose.py:90` 特别确认过：它所在的 `peak_and_bucket()`
**函数名与 docstring 都写明"帧内全局最大"**，并返回 `frame[py, px]` 作为
**峰值**。换成带质心细化的 `zero_order_center`，`py, px` 可能不再指向峰，
**这个函数会直接坏掉**。README 里"暗帧不可用裸 argmax"针对的是
`measure_flat_reference` 那种**无光**帧，不是这里的有光平场/光栅帧。

⚠️ 真正与 README 那条铁律相关的 `measure_flat_reference` 属
`slm_bench_probe.py` / `slm_drift_probe.py`，本轮**未在无硬件条件下判定**，
留给有硬件时确认。

**结论**：R-38 剩余候选都需要**逐处判断 + 硬件确认**，不属于
"不需要设备就能确认修改是否正确"的范畴，故留在待办、不进本轮。

---

### 5.20 R-9 —— 把 300px 光阑的说明挂回它真正讲的那个常量（2026-10-04）

`slm_zernike_pib.py` 早已正确：那段硬件踩坑说明就挂在
`ZERNIKE_APERTURE_RADIUS = 300.0`（`:195`）正下方。

**`slm_zernike_shaping.py` 才是坏的那份**，而且比"挂错常量"更糟 —— 同一个常量
上挂着**两段互相矛盾**的注释：

| 位置 | 主张 |
|---|---|
| `:186-188`（正确挂在常量上） | 300 太窄，本 runner 应当用 600，否则"生成的 Zernike 基只有 GUI 的 1/4 面积" |
| `:203-212`（**裸字符串**，挂在 `TARGET_BOX_WAIST_FACTOR` 之后） | 本台架光束半径只有 ~300 px；用 600 光阑则只有内一半落在光束上，每个模式在照明区几乎**恒定**，而恒定相位不改变远场 ⇒ **修正静默失效**（**硬件实测**：平场与 2 rad 离焦无法区分，直到匹配） |

第二段是**硬件实测**结论，第一段是推理。而且 `:203-212` 是普通字符串表达式，
**不是 docstring**，所以它谁都没在解释 —— 读者只会以为 `TARGET_BOX_WAIST_FACTOR`
的说明。

**做法**：把这段硬件实测说明搬回 `ZERNIKE_APERTURE_RADIUS` 名下，
并把 `:186-188` 那段改写成"此处存在未决矛盾"而不是继续断言 600 是对的。
**没有改常量值** —— 值归 X-3 决定（§5.17：本模块无生产导入方、语料钉 300），
不静默改动。

---

### 5.21 两处 `1e3` **故意不统一** —— 别顺手去重（2026-10-04）

R-11 实测**已完成**（`ZERNIKE_CLIP` / `IMPROVE_EPS` 两侧都在，裸字面量已清）。
顺带纠正一条**过期的位置记录**：TODO 说 `GUARD_PENALTY` 定义在
`drivers/ccd/common.py`，**实际在 `utils/image/target/objective.py:33`**
（它随 `ObjectiveSpec` 抽取一起搬进了叶子）。

但仓库里还有 3 处裸 `1e3`（`metrics.py:298,384,396`），
**它们与 `GUARD_PENALTY` 不是同一个用法，不能合并**：

| | 用法 | 含义 |
|---|---|---|
| `objective.py:773-775` | `j + sign * GUARD_PENALTY` | **相对偏移**：无论 `j` 多大，惩罚后一定比 `j` 差 |
| `metrics.py:298/384/396` | `return 1e3, 0.0` | **绝对值**：注释声称"far worse than any valid state"，但这只在**没有任何合法目标超过 1e3** 时成立 |

`GUARD_PENALTY` 之所以做成偏移量，恰恰就是为了**不依赖目标的取值范围**。
绝对值那种写法在范围变大时会静默失效 —— 这是它更脆弱的地方。

⚠️ **本轮不修**：改绝对值会改变真实数据上的目标函数取值，
可能影响优化轨迹与最优解，**必须用硬件/基准对照**才能确认，
不属于"不需要设备就能确认修改是否正确"的范畴。

---

### 5.23 R-17 —— 日志改回 loguru 惰性格式化，并加 AST 守卫（2026-10-04）

9 处 `logger.xxx(f"...")` 改为 `logger.xxx("{}", value)`。这不是洁癖：
f-string 在**调用前**就完成插值，所以**日志级别关掉时消息照样拼**，
而这正是用结构化日志而不是 `print` 的唯一理由。

⚠️ **文本搜索漏了 3 处，AST 守卫一上来就抓到**：那几处是
```python
logger.debug(
    f"Initial Image Max brightness: {np.max(init_img)} "
    f"@ {get_camera_exposure_ms(cam)}ms"
)
```
`logger.debug(f"` 不在同一行，正则 `logger\.\w+\(f"` 匹配不到。
**这是本轮第三次"文本搜索给出假阴性、AST 给出真值"**（前两次是
`default=` 在 `option()` 里、以及 `radius(center=center)`），
也是把守卫写成 AST 而不是正则的第三次收获。

守卫只匹配 `logger.<level>(...)` 且首参是 `ast.JoinedStr`，
因此不会误伤那些"为了别的目的先拼字符串"的地方。

变异验证：塞回一处 f-string 日志 → 守卫如期失败。

`tests/ao_shaping/optimizer/wfless` **434 passed / 1 skipped**。

> 顺带说明：`slm_model_in_loop.py` 有一个 F821（`Recorder` 未定义），
> 但该文件**不在 HEAD 里、且是未跟踪状态** ⇒ 属于并行的 WIP，不是本轮引入。

---

### 5.24 F-13 —— `Strehl` 列名加脚注（2026-10-04）

实测发现这条**早已基本处理完**，原记录有两处过时：

* `strehl()` 的 docstring 已自述 `Normalized overlap (**Strehl-like**)`，
  并明确写了是均值中心化后的余弦相似度 ⇒ 函数侧早就诚实了；
* `beam_shaping_papers.md:30` 早已定义 `- **Strehl**: 归一化重叠（余弦相似度）`；
* `zotero_objectives/README.md:524-529` 早已列出命名冲突与建议；
* ⚠️ 原记录说 **`beam_shaping_benchmark.py`** 消费其输出并称 Strehl ——
  实测该文件**完全没有** `overlap` / `strehl` / `cosine` 引用，**该前提已不成立**。

**唯一真正误导的残留**是表格列名：表头只有裸 `Strehl`，读者若不往下翻到
定义行，会把它当物理 Strehl 比读。

改动：列名 `Strehl` → `Strehl†`，并在定义处补脚注，写明
① 它是去均值余弦相似度、② 真 Strehl 比是 `峰值强度 / 理想峰值强度`、
③ 列名保留只为与既有表格对齐但**不可当 Strehl 读**、④ 指向 zotero README 的决策记录。

**没有改函数名**：`strehl` 是既有 API，重命名会波及调用方与已提交产物，
属于需要一并决策的事，留在 `zotero_objectives/README.md` 的待决项里。

---

### 5.26 R-14 / R-15 —— 一个符号搬家了，一个方案不该做（2026-10-04）

#### R-14：`_metric_panel` 已不存在，问题搬家且**需要产品决策**

原条目指向 `slm_zernike_pib.py:1544` 的 `_metric_panel`。实测**该函数已不存在**：
面板现在是共享叶子里的 `ShapingObjective.metric_panel()`
（`utils/image/target/objective.py:780`），每轮在 `slm_zernike_pib.py:1091`
被调用一次（另有 `:1031` 在 init 帧上跑一次）。
**"每 epoch 全量指标"这个担忧仍然成立**，但符号与行号都失效了。

**未做，原因是它不是机械改动**：跳过的那几轮，`m_*` 列该**留空**还是
**沿用上一轮**？
* 留空 ⇒ `m_ee` 等字段变 NaN，而报告生成器与
  `test_rms_pib_adaptive_columns_are_not_nan_on_later_rows`（**NaN 判据**）
  都读这些列；
* 沿用 ⇒ 面板描述的是**上一帧**的远场，而代码里明确写着
  "The recorded panel/metrics describe the POSITIVE frame" ⇒ **指标会说谎**。

这是**产品决策**（每 N 轮记一次全量面板是否可接受），且要真实 run 对照，
不属于"不用硬件就能确认"的范畴。

#### R-15：`RawReport` 字段**不该做**，但契约应该写死

原条目要求把 `setattr(recorder, "energy_loss_violations", ...)` 改成
"显式 `RawReport` 字段"。实测**这个方案形状不对**：

`energy_loss_violations` 是**整轮一个标量**，而 `Recorder.append`
（`utils/io/file.py:388`）取**每轮 record 键的并集** ⇒ 做成"列"
要么根本不出现，要么只挂在**最后一轮**那一行上；`Recorder` 也**没有**
"轮次元数据"的 schema（只有 `mark`/`mode`/`history`/`_all_columns`）。

所以**保留属性**是对的机制，改成：
1. **直接赋值**而非 `setattr`（意图明确，不靠读者推断）；
2. **写明为什么是属性而不是列**（避免下次有人"顺手"改成列）；
3. **加测试钉住**：断言两侧都不用 `setattr`、都直接赋值，
   并断言唯一消费者 `scripts/compare_shape_objectives.py:258`
   的 `getattr(rec, "energy_loss_violations", 0)` 读法成立。

#### 过程失误：同一个错误犯两次

两次改坏文件，**同一个原因**：`oldString` 从无缩进的 `def` 开头，
它在**缩进后的行里作为子串匹配成功**，于是**静默吃掉 4 空格缩进**。
第二次是"优化 complete 日志"时踩的（`logger.info(` 那次）。

两次都是 `ruff` 先报出来、而不是测试 —— 说明**语法层的破坏必须先过 ruff**。
修法：不再硬编码缩进，而是**从被替换的那一行读取缩进**再重建。

---

### 5.27 R-12 —— 副本早就合并了，剩下的只是"形状不好看"（2026-10-04）

原条目把 `_update_dynamic_weights` 的问题定义为"两个引擎各有一份、会漂移"，
建议改成 `AdaptiveWeights` dataclass。实测：**前半句已经不成立**。

`_update_dynamic_weights` 现在**只有一份**，在共享叶子
`utils/image/target/objective.py:36`，两个引擎各自 import 同一份
（`slm_zernike_pib.py:93`），`slm_zernike_shaping.py` 同理；
`slm_zernike_pib.py:180` 的注释也写明了这次搬迁
（"likewise now live in `utils.image.targets` next to the `ShapingObjective`
that calls them"）。

**所以 R-12 真正想消灭的风险（副本漂移）已经消失。**

剩下的两点确实存在：裸 `dict` 上的 `state.setdefault(...)`，
以及变长返回 `tuple[float, float] | tuple[float, float, float]`。
但它们**不是缺陷，是被测试钉住的既定契约**：
* `test_slm_zernike_rms_pib.py:128+` 覆盖首调 50/50、PIB 改善抬高权重、RMS 改善抬高权重等行为；
* docstring 明确写了 `ee is None`（两项）与给了 `ee`（三项）两种分支，
  且两项分支"byte-identical to the previous `(w_pib, w_rms)` pair"——
  这是 R-4 有意保留的兼容面。

改成 dataclass 会同时动共享叶子 + 两个引擎 + R-4 测试，
**行为收益为零**，属于纯 churn，故本轮不做。

> 这是本轮第 6 次出现"条目描述的问题已被后续重构解决"的模式
> （R-9/R-10/R-11 已完成，X-3 前提不成立，R-38 前提不成立，R-29 前提不成立，
> R-14 符号已搬家）。**结论：§2.1 的行号与描述需要一次系统性重扫**，
> 否则后续每轮都会重走这些已经清掉的路。

---

### 5.28 F-9 —— 进度显示本来就有，但缺"全局计数"（2026-10-04）

原条目只写"加进度显示"，实测**已存在**：`repeat_shape_objectives.py:137` 在
`variant x rep` 双层循环里逐次打 `=== {slug} rep {rep}/{repeats} ===`。
所以本条不是"从零加"，而是判断已有日志够不够。

**不够，缺的是全局位置。** 各 variant 的耗时差异很大（启发式 vs SPGD、
不同 objective 的收敛轮数），而 `rep 1/1` 这种尾行**在整轮扫描的任何位置都可能出现**：
第 1 个 variant 的第 1 次运行和最后 1 个 variant 的第 1 次运行打印出的字符串完全一样。
操作者无法回答"现在到第几个了、还剩多少"。

**改动**（`scripts/repeat_shape_objectives.py`）：

* 抽出纯函数 `_progress_label(...)`，把三层计数拼成一行：
  `=== run {i}/{N} | variant {v}/{V} '{slug}' | rep {r}/{R} ===`；
* `main` 里 `variants = list(cso.VARIANTS)` 先物化，`n_runs = len(variants) * args.repeats`，
  循环内 `run_idx += 1` 后再打日志；
* 循环前多打一行 `sweep matrix = V variants x R repeats = N runs`，让 N 在第一次运行前就可见。

`variants` 物化是必需的：原来 `enumerate(cso.VARIANTS)` 直接消费可迭代对象，
新增的 `len()` 需要能重复求长度。

**离线验证**（`tests/ao_shaping/scripts/test_repeat_shape_objectives_progress.py`，3 例）：
标签格式、2x3 矩阵走完恰好铺满 `1..6` 且末行为 `run 6/6`、
以及一条**源码守卫**（`n_runs` 必须由 `len(variants) * args.repeats` 得出，
且 `run_idx += 1` 必须出现在日志调用之前 —— 否则首 run 报 `0/N`、末 run 报 `N-1/N`）。
两处变异（删掉自增、把总数改成 `len(variants)`）均被捕获。

> 顺带记录：`tests/ao_shaping/scripts/` 全目录跑下来有**一个既有失败**，与本条无关 ——
> `test_common_helpers_not_reintroduced.py::test_migrated_generator_import_actually_resolves[generate_oopao_impact_report.py]`
> 因 `ModuleNotFoundError: No module named 'gymnasium'` 失败（`gymnasium` 是 `rl` 可选依赖组，
> 本环境未装），即该 generator 的 import 链会拖进 `optimizer/rl/envs.py`。留给 F-10 一并看。

---

## 6. 建议执行顺序

| 批次 | 内容 | 前置 |
|---|---|---|
| ~~**第 1 批（纯收益，零行为风险）**~~ | ~~R-23、R-24、R-30、R-31、F-2、F-6、F-7、F-8、R-40~~ ✅ **2026-10-03 全部完成 → §5.1** | — |
| **第 1.5 批（文档/常量收口，先定事实再改代码）** | F-10（OOPAO 改写 + 重跑报告）、F-11（SLM 序列号）、F-12（标定常数三方）、F-13（`strehl()` 命名） | 需设备/一次扫描 |
| **第 2 批（止真 bug，需先补特征测试）** | ✅ **R-1~R-4、R-9~R-11、R-35 全部完成（均经变异验证）→ §5.2 / §5.3** | — |
| **第 3 批（架构重构）** | ~~R-20~~ ✅ §5.4、~~R-21~~ ✅ §5.8、~~R-22~~ ✅ §5.7、~~R-25~~ ✅ §5.6、~~R-26~~ ✅ §5.5；~~R-36~~ ✅ §5.9；~~R-27~~ ✅ §5.11、~~R-28~~ ✅ §5.10；~~R-32~~ ✅ §5.12；~~F-14~~ ✅ §5.13、~~F-15~~ ✅ §5.13；**剩余 R-37→R-38→R-39→R-41** | ~~R-20 先行~~ ✅ 已满足；R-36 有 R-35 golden 兜底 |
| **第 4 批（内部重构）** | R-5~R-8、R-12~R-17、R-19 | ~~R-1~R-4 完成~~ ✅ 已满足 |
| **硬件轨道（并行）** | F-1 → H-7~H-13 → **H-19**（与 H-9 合并做：方形路径复用 PIB 的 ABBA 参考实现）→ H-14 复扫 → H-15/H-16 → H-1/H-2 → H-3~H-6 → H-17/H-18 | 设备在线 |

> **下一步建议（离线，无需设备）**：
> ① **R-36**（把 `runner_common.py:118-323` 的 click 机制抽成零 `ao_shaping` 导入的叶子
> 模块）现在有 R-35 的 19 条 `--help` golden 兜底，是第 3 批里风险最低的一步；
> ② 反模式表里 `utils/image/display.py` 仍未搬（`ImageVoltagesDisplay` /
> `plot_funcs` / `VOLT_HEIGHT` 的 re-export 要一起改，是 `utils/image/` 最后一块渲染代码）；
> ③ **R-36** 之后是 R-37（22 个探针迁到 `tools/slm/params.py`）—— R-35 的 golden 已就位。