# TODO — AO-Shaping 待办索引

> **2026-10-06 拆分**：本文件曾经是 1822 行的工程账本，其中 **87% 是已完成档案**。
> 现在按「待办 vs 证据」拆成两处：
>
> | 内容 | 位置 |
> |---|---|
> | **未完成项（43 条）** | GitHub issue，标题带原 ID（`[H-14] ...` / `[R-44] ...`） |
> | **已完成档案（§3.1 / §4 / §5，1625 行）** | [`docs/dev/todo_archive.md`](docs/dev/todo_archive.md) |
>
> 拆分的判据不是篇幅，是**用途**：未完成项需要 assignee / 阻塞关系 / 检索
> （§6 的硬件轨道写明 *H-19 与 H-9 合并做、H-14 复扫必须在 H-19 之后* ——
> 这在 issue 的 blocked-by 关系里是原生的，一张 markdown 表格表达不了）；
> 而档案是**决策证据**，它的作用是阻止未来的重复动作（详见
> [archive 头部](docs/dev/todo_archive.md#为什么要分开)）。
>
> **⚠️ 两者没有合并进 `README.md`。** README 是用户手册且被
> `tests/ao_shaping/test_conventions.py` 归为 `_LIVE_DOCS`
> （其中每个 `python -m` 目标必须可解析），而本账本按 ID 引用当时的**失效路径**
> （那是在描述缺陷，不是给出可执行指令）⇒ 合并会直接打红该 guard。

---

## 怎么用这个文件

- **找一件要做的事** → 下面三节的表格 → 点进对应 issue。
- **想知道某条结论为什么是这样** → `docs/dev/todo_archive.md`（§ 编号与旧账本一致）。
- **全仓按 ID 的引用**（`AGENTS.md`、`docs/issues_report.md`、`report/**`、
  代码注释、测试 docstring 里的 `TODO.md R-20` / `H-11` / `R-37` …）
  指的是**内容**，不是行号 ⇒ 本表与 archive 合起来仍能解析它们。
  已完成项的 ID 直接引 archive 对应小节。

**新增待办**：开 issue，标题带 ID 前缀（本文件的编号规则：`H-` 需硬件、
`R-` 离线重构、`F-` 缺陷/文档、`X-` 原结论修正），并在本表加一行。
**不要**再往本文件堆完成记录 —— 那是 archive 的职责，且它已经 1740 行了。

---

## 1. 需硬件（设备在线才能推进）

### 1.1 runner 待上设备实测清单

| ID | 摘要 | issue |
|---|---|---|
| H-1 | `dm-matrix` sequential/hadamard 双模式标定内核离线通过（122 passed）但未上设备实测 | [#18](https://github.com/ywzhang909/AO-shaping/issues/18) |
| H-2 | `zernike-matrix` 2026-09-16 重写后的标定内核未上设备实测 | [#19](https://github.com/ywzhang909/AO-shaping/issues/19) |
| H-19 | **ABBA 漂移对消已落地但未实验验收**（判据 `dec>0.55` 且 `late_gain≥10%`） | [#36](https://github.com/ywzhang909/AO-shaping/issues/36) |
| H-20 | `spgd-square` 合并成 `slm_shaping_runner` 后未完成实验验证光学回归 | [#37](https://github.com/ywzhang909/AO-shaping/issues/37) |
| H-21 | `spgd-square` 的 `--delta` 语义变了（空操作 → 可钉住），需实验系统 A/B | [#38](https://github.com/ywzhang909/AO-shaping/issues/38) |
| H-22 | `slm-pib` 合并后未完成实验验证回归（新增一次 `patch_sim_pib_shaping`） | [#39](https://github.com/ywzhang909/AO-shaping/issues/39) |
 | H-23 | ✅ **已实施（2026-10-14）**：三处协调改动全部落地。①`slm_shaping_runner.py::_square_json_payload` 增记 `active_modes`（用 `_active_zernike_modes(n_max, mask)`）+ `n_terms_total`；②`index.py` 增 `_Verdict.active_modes`/`HwRecordRef.active_modes` + **规则 4b**（规则 5 之前，仅当 sidecar 声明且 `len(_c)==len(active)` 才收），`_CACHE_VERSION` 2→3 且缓存 encode/decode 都带上该字段；③`transforms.py::expand_active_modes_to_noll` 用 canonical `noll_indices` 把活动模式散射回全长 Noll 向量，`records.py` 在 `zernike_coeffs_to_panel` 前调用。实测散射后 panel 与「手搭全长向量」**逐位一致**；**无 sidecar 声明的旧语料仍按 `ODD_COEFFICIENT_LENGTH` 排除**（不追溯重解释）。新增 20 测试（`tests/ao_shaping/ml/hwdataset/test_active_modes.py`），`hwdataset/` 380 passed | [#40](https://github.com/ywzhang909/AO-shaping/issues/40)（**本 commit `Closes`**） |
| H-24 | 方形家族进 `hwdataset` 会换 `fov_px` ⇒ `far_field_padding` 标定必须重扫 | [#41](https://github.com/ywzhang909/AO-shaping/issues/41) |
| **H-32** | **`--w-loggrad` 结构项已落地但未实验验收**。离线全绿（`utils/` 992、`wfless/` 626、sim 端到端 `J≠shape`）；6 类候选里只落地这 1 类，2 类早已存在、3 类否决（附实测数字）⇒ [`report/pib_loss_terms/README.md`](report/pib_loss_terms/README.md)。**⚠️ 它不能修复不收敛**（H-14 已证瓶颈是慢漂移），故默认 `0.0`。验收请在 **δ=0.05** 这类「有信号但被漂移污染」的档位做，**勿用 0.0005**（梯度信号本身在噪声下）。**待开 issue** | — |
| **H-33** | 🔴 **GS → Zernike 拟合 → SPGD 匀化阶数扫描：代码已就绪并合入 `banckend`，但「多少阶 Zernike 才能复现 GS 整形」只能在实验系统回答**。采集 [`scripts/slm/run_gs_refine_zernike_sweep.py`](scripts/slm/run_gs_refine_zernike_sweep.py)（Stage A 跑一次 `slm_gs_refine` 留全历史 → Stage B 按 `n_max` 拟合 GS 最优相位 → Stage C 在系数上跑 SPGD，逐阶独立记录）；计划 **H0~H12** 见 [`report/shape_trials/gs_refine/gs_refine_zernike_hardware_plan.md`](./report/shape_trials/gs_refine/gs_refine_zernike_hardware_plan.md)。**仿真已确立（硬件无关）**：拟合残差随 `n_max` 单调不增（8 阶 0.0719→0.0385 rad，−46.5%）、`n_max=25`（351 模）不触发 `ZERNIKE_CLIP`、`--lr 0` 复现步长拒绝失效、能量守卫非瓶颈。**仿真答不了本条**：数字孪生 GS 前向模型未标定 ⇒ `best_phase`≈平场（`max\|c\|` 各阶恒 0.0142 rad），且阶数间**不可排序**（跨阶 PIB 极差 0.0046 / CV 极差 0.0039 ≈ 单轮 SPGD 抖动 0.0010~0.0035）⇒ 报告据此**拒绝**给出「最优阶数」。**两个 P0 硬门**：① GS 对比择优 —— GS 打不过平场即**停止**，先查 `--camera-pixel-um` / `--beam-radius-px` / 面板配准；② 噪声地板 σ —— 它是后续一切判据的分母，σ 与阶数差同量级则排序**物理上不可判**，须如实报告。核心科学项 = **H6**（Zernike 在**远场**复现了多少 GS 效果，`fit_rms` 只是相位空间代理）。⚠️ `fit_rms` **只在同一次 run 内跨阶可比**（换 `--n-max` / `--max-energy-loss` 会改变 Stage A 相位 0.008~0.012 rad，根因未查明）。**已知未做**：SPGD 预算放大（E1）在仿真下需 300 epoch 被砍，「不改善是预算不足还是物理上限」未答 → 列为 H-7 项 | — 待开 issue（正文已提交于 `report/shape_trials/gs_refine/gs_refine_zernike_hardware_plan.issue.md`；**API 被网络中间人拦截，无法从本机创建**，见 H-14 下方同类记录） |

> ⚠️ **H 编号说明（2026-10-19 复核）**：上两条原为 H-25/H-26，已重编为 **H-32/H-33**。
> 2026-10-16 的硬件验收批次在 GitHub 上**已占用** H-25~H-31 号段
> （#137=H-25、#139=H-26、#141=H-27、#145=H-28、#146=H-29、#147=H-30、#159=H-31），
> 旧号若保留会与新批次 issue 撞号。H-32 及以上目前空闲，新条目从 H-32 起编。

### 1.2 FourierGSNet 实验系统验证（`report/fouriergsnet_pipeline/`）

> 📋 2026-10-12: 结合 `slm-gs-refine`（从未实验运行）与 gsnet 系列报告的
> **下一步尝试方案**见 [`report/fouriergsnet_pipeline/next_attempt_plan.md`](report/fouriergsnet_pipeline/next_attempt_plan.md)
> （A 离线修复 → B 探针 → C1 对比择优 → C2 refine 首跑 → C3 gsnet 复测 → C4 FourierGSNet run）。

| ID | 摘要 | issue |
|---|---|---|
| H-3 | §0 前置：设备在线 + 激光开 + 无残留 `python*` + 看门狗硬超时 | [#20](https://github.com/ywzhang909/AO-shaping/issues/20) |
| H-4 | §2 smoke test 结果区仍 `_待填_`（5 步，末尾须打印 `SMOKE_OK`） | [#21](https://github.com/ywzhang909/AO-shaping/issues/21) |
| H-5 | §3 DM/WFS 实际 IP 与网段（上次 `192.168.0.101–126` 全段不可达） | [#22](https://github.com/ywzhang909/AO-shaping/issues/22) |
| H-6 | `slm-gsnet spgd` 实验系统光路验证验收（0 阶 `argmax` / 框内能量与 CV / 产物落盘） | [#23](https://github.com/ywzhang909/AO-shaping/issues/23) |

### 1.3 实验系统跑出的代码级问题（修完才能复测）

> 2026-10-01 实验系统 `slm-gsnet spgd` 实测结论是**随机游走**
> （`dec=0.487`，末态比初态差 35.9%，EE 流失 6 倍，0 阶峰值 225→17）。
> 以下 **6 项必须先改**再复测。来源：`report/fouriergsnet_pipeline/hardware_run_20261001.md` §5–6。

| ID | 摘要 | issue |
|---|---|---|
| H-7 | 方形路径**无 encircled-energy guard** → 任何 `w_efficiency` 权重都会被「散光换 CV」击败 | [#24](https://github.com/ywzhang909/AO-shaping/issues/24) |
| H-8 | 随机初始化是满幅 `uniform(-π,π)`，与 docstring 的 "small init" **矛盾**，且满幅起点 EE 最高 | [#25](https://github.com/ywzhang909/AO-shaping/issues/25) |
| H-9 | 方形路径**无 ABBA / 漂移对消**（三个采集点全是相邻两帧）。**2026-10-02 起成本大降**：`slm-pib` 已有可照搬的同款实现 | [#26](https://github.com/ywzhang909/AO-shaping/issues/26) |
| H-10 | `--exposure-time-ms` 默认 **80.0 ms**（本实验系统近饱和基线 ~0.02 ms），且传它会**关掉饱和保护**使其成死代码 | [#27](https://github.com/ywzhang909/AO-shaping/issues/27) |
| H-11 | 焦距标定常数自相矛盾：文档同写 `132940/P` 与 `7600/P`（**差 17.5×**） | [#28](https://github.com/ywzhang909/AO-shaping/issues/28) |
| H-12 | `lr=0` 时 `learning_schedule()` **静默覆盖 `--delta`**；`_param_scale` 两基不同；"freeform per-pixel" 名不副实（`np.kron` 分块） | [#29](https://github.com/ywzhang909/AO-shaping/issues/29) |
| H-13 | 可达目标需重新定义：本实验系统 80 px staircase 只能影响**大尺度**结构，刻 50 px 平顶方块不可达 | [#30](https://github.com/ywzhang909/AO-shaping/issues/30) |
| H-14 | 🔴 **原结论「可用区间 ≈0.1」已被 1000× 全扫描推翻**（8 档无一收敛，`dec` 全在 0.43–0.56）⇒ 瓶颈是**慢漂移**不是 δ | [#31](https://github.com/ywzhang909/AO-shaping/issues/31) |
| H-15 | 驱动级 `get_camera_exposure_ms` 回读恒为 3.0、`auto_exposure` settle 滞后 | [#32](https://github.com/ywzhang909/AO-shaping/issues/32) |

### 1.4 标定常数与物理量待复核

| ID | 摘要 | issue |
|---|---|---|
| H-16 | CCD 像元尺寸权威值 **2.2 µm** ⇒ `K` 应 5021→10954（×2.18）。⚠️ 但 10954 本身待复核，见 F-12 | [#33](https://github.com/ywzhang909/AO-shaping/issues/33) |
| H-17 | 用 `zernike-matrix` 重标响应矩阵（旧矩阵在归一化下测得，幅度维度无意义）+ 非线性/迭代/`n_avg`/532nm CSV | [#34](https://github.com/ywzhang909/AO-shaping/issues/34) |
| H-18 | SLM 稳定化**不能用固定等待**（`SLM_RESPONSE_TIME_S = 0.3` 是驱动自报值）⇒ 改「连续两次读数一致」 | [#35](https://github.com/ywzhang909/AO-shaping/issues/35) |

---

## 2. 离线代码重构（无需硬件，但需先补特征测试）

### 2.1 `slm_zernike_pib.py` —— 行为 bug 优先

> ⚠️ 该文件已从 2179 行大幅重构（现 **1720 行**，#135-A `3b8cfc9c` 移除重复 `TargetShape` 后），`calc_objective` 已抽成
> `ShapingObjective.__call__` ⇒ **旧账本行号全部失效**，但 bug 逐条存活。
> 🔴→✅ **`slm_zernike_shaping.py` 已不再是同源副本**（2026-10-15，X-4/#74，PR #129
> `ee96acfe` 合入）：该 1600 行分叉（`_row0`/`_log_row`/`learning_schedule`/饱和分支
> 结构 + `w_outside` 能量项 + `__main__` fork demo）已并入 `slm_zernike_pib.py`
> （`w_outside`/调试产物/ABBA 等均收敛至此）；其最后的 **44 行弃用重导出 shim**
> 也已删除（2026-10-20，#138 —— 实测**零生产导入方**、零测试导入方，仅一处
> 按模块字符串枚举的测试引用，已随之更新）⇒ 下面各条改动**只需改
> `slm_zernike_pib.py` 一处**，历史「改一处必须同时改副本」的约束已失效。
> 副本盘点见 [archive §5.17](docs/dev/todo_archive.md)。

| ID | 摘要 | issue |
|---|---|---|
| R-5 P1 | 目标函数 if/elif 链 → `Objective` 类族 + `GuardedObjective` | [#42](https://github.com/ywzhang909/AO-shaping/issues/42) |
| R-6 P1 | 两个搜索分支共享 `BenchSession`（序列重复，**饱和策略还不一致**） | [#43](https://github.com/ywzhang909/AO-shaping/issues/43) |
| R-7 P1 | 巨型函数拆编排器：~900 行 → `prepare_geometry`（**不可变 dataclass**，字段必须区分 `window_center_full_frame` / `reference_center_window_local`）+ `_run_spgd` / `_run_heuristic` | [#44](https://github.com/ywzhang909/AO-shaping/issues/44) |
| R-8 P1 | 硬件安全 try/finally：任何异常（相机掉线、越界、`KeyboardInterrupt`）都让 SLM 停在随机相位 | [#45](https://github.com/ywzhang909/AO-shaping/issues/45) |
| ~~R-13~~ | ✅ **收敛完成**（#46）：实测**不是 5 份相同副本，而是 4 种不同实现** —— 只有 `slm_square_shaping` 与 `pib` 逐字节相同（9 行）；`slm_zernike_pib` + `slm_zernike_shaping` **早已是「响亮失败」版**（检测 `accepts_var_keyword`、转发 `**kwargs`、并 warning 列出被丢参数），而 `pib` / `slm_square_shaping` / `sim_spgd` **仍静默吞参**（且永远不转发给声明 `**kwargs` 的 callee，即 `**config.kwargs` 逃生口失效）。canonical 落在 `optimizer/constants.py::create_optimizer`，紧邻它本就该集中管理的 `OPTIMIZER_MAP`。🎁 **额外发现**：3 个文件各自还定义了一份 `OPTIMIZER_MAP`，与 canonical **内容完全相同**（仅差类型标注）→ 一并改为导入。守卫 `test_optimizer_factory_single_home.py`（12 项）除结构检查外**锁定行为**：丢参必须被报告、声明 `**kwargs` 的 callee 必须收到传参 | ✅ |
| R-14 P2 | `metric_panel` 每 epoch 全量指标 → 加 `panel_every_n`。⚠️ **前置是产品决策**（跳过轮次留空 vs 沿用上一轮） | [#47](https://github.com/ywzhang909/AO-shaping/issues/47) |
| R-18 P3 | ⚠️ **2026-10-06 复核：原描述已过期，改为需设计决策**。①「删陈旧注释」**已完成** —— `slm_zernike_pib.py` 与死分叉里 `--init-gs` / `gs_warm` / `gerchberg` **零痕迹**。②「接入已有实现」**无物可接** —— `gs_warm_start` 只是 `runner_common.py:2442` 的**布尔 CLI flag**；GS 相位由 `_gs_pupil_phase`（`slm_gs_refine.py:573`，内调 `gs_shape` 于 `:626`）生成（2026-10-22 重钉，行号随代码漂移），但该函数签名绑死 `SlmGsRefineConfig` + `radius_px`/`target_side_cam`，slm-pib 用的是 `SlmZernikePibConfig`，**无法直接复用**。③ 真正的障碍：slm-pib 每轮由 `_c` 重建面板相位，注入全面板 GS 相位会破坏该不变式；改为拟合 Zernike 系数则对**方形**目标受 AGENTS.md「n≤4 无法合成方形远场」限制 | [#48](https://github.com/ywzhang909/AO-shaping/issues/48)（**仍开放**，待决策） |
| R-19 P3 | 🟡 注入机制已统一（archive §5.34），**`BenchSession` 本体未做** ⇒ R-5~R-8 仍待办。本条是那四项的跟踪父项 | [#49](https://github.com/ywzhang909/AO-shaping/issues/49) |

### 2.2 `utils/` + `scripts/` + `tools/` 架构重构

**原前置阻断项（🔴 `utils/` 不是安全的叶子）已于 2026-10-03 完成 → archive §5.4。**

| ID | 摘要 | issue |
|---|---|---|
| R-29 | ✅ 已完成：`utils/image/display.py` 已搬到 `display/image_display.py`（commit `a9236e9`，2026-10-09），`utils/__init__.py` 的 `ImageVoltagesDisplay` / `plot_funcs` / `VOLT_HEIGHT` … re-export 已改指 `ao_shaping.display.image_display`；旧路径 `ao_shaping.utils.image.display` 现抛 `ImportError`（`test_utils_leaf_no_rendering_import.py::test_old_display_path_raises` 锁定） | [#50](https://github.com/ywzhang909/AO-shaping/issues/50) |
| ~~R-33~~ | ✅ **已由上游实现**（2026-10-14 核实）：`@with_params(MicroDMImageCollectParams, kw_name="params")` 已接线（`micro_dm_image_collect.py:465-467`，docstring 自述 "R-33: migrated to with_params"）；手写 `_resolve_ips` 已删除，`resolve_ips` 现住 `drivers/dm/micro/wiring_map.py` 并在 `tests/ao_shaping/drivers/dm/test_micro_resolve_ips.py` 有测试。**issue 的「7 个驱动内部符号导入」判据已失效**：实测剩余 6 个 driver 导入（`BaseCamera` / `MAX_CHANNELS` / `VOLTAGE_MAX` / `VOLTAGE_MIN` / `R50Controller` / `resolve_ips`）**全部在各自 `__all__` 内**，即公开再导出，driver-internal 导入数为 **0** | [#51](https://github.com/ywzhang909/AO-shaping/issues/51) |
| ~~R-39~~ | ✅ **实测后判定无需改代码**（2026-10-14）：两个子项都**不可离线收口**，issue 的指令照做会引入危害。**曝光默认值**：issue 自己写明「不能一刀切成同一个值 —— 那是实验系统相关物理事实，不是笔误」，需逐探针确认可用区间 = 硬件。**内存槽轮换**：实测为**两份不同策略**而非重复实现 —— 驱动 `santec/driver.py::SlotRotator` **随机**取槽且从面板 `get_displayed_memory_number()` 读当前显示槽（跨进程有效），`wfless/slm_zernike_primitives.py::SlotRotator` **顺序**递增、无设备引用（可复现）；合并会静默改变实验系统写入的物理槽位，正是 issue §2.3 警告的「看似可信的错数据」。第三种写法 `phase_capture.py` 的 `% 128 + 1` 覆盖 1..128（其余路径 2..125），属实验系统事实，未动。**已用测试钉死该差异**：`tests/ao_shaping/tools/slm/test_slot_rotation_policies.py`（6 项），使「同名两份 = 重复」不再看起来像显然的清理目标 | [#52](https://github.com/ywzhang909/AO-shaping/issues/52)（**不关闭**：仍需硬件确认） |
| ~~R-43~~ | ✅ **已补**（2026-10-14）：新增 `tests/ao_shaping/tools/micro_dm/test_micro_dm_image_collect.py`（**38 项**，含参数化展开），覆盖既有 CLI / 磁盘布局两个测试都到不了的「解析 → 下发」这段纯逻辑：`_parse_channels`（all 大小写/空白、单通道不被 `all` 前缀吞掉、越界与畸形各自 exit 1、上界含端点）、`_resolve_ip_port`（`10000+末段` 规则、显式端口优先、末段 0 合法、畸形/越界各自报错、**有效 IP 永不漏出 `port=None`**）、`_voltage_array`（仅目标通道带电压、`ch=None` 全归位、home 值生效、首末通道可达）、`_send_voltage`/`_home_channel`（**开关两侧都跑**，断言互斥不双发）。⚠️ 顺带修正一个**我自己的坏测试**：原打算构造输入去撞 `无法解析端口` 分支，实测该分支**不可达**（IP 先被完整校验，`controller_tcp_port` 不可能返回 None），故改为断言可观测不变式而非伪造错误路径 | [#53](https://github.com/ywzhang909/AO-shaping/issues/53) |
| ~~R-44~~ | ✅ **已修**（commit `7284dc1f`，2026-10-07）：`cosin` 分支原为 `lr * cos(pi*e/E) + 1e-6` —— 是**半波**不是退火，`cos` 在 `E/2` 变号使 `1e-6` 地板失效，后半程喂给优化器 `-lr + 1e-6`（满幅反向，不是小步）。改为教科书退火 `1e-6 + (lr-1e-6)*(1+cos(pi*e/E))/2`，与 `optimizer.wf.rms_by_zernike.cosine_annealing_lr(lr_min=1e-6)` **逐式相同**：epoch 0 = `lr`、`epoch==epochs` = 地板、单调不增、恒正。测试 `test_adam.py`（29 项）钉死该契约（ positivity / monotone / floor），旧「钉住负值」测试已替换。⚠️ `optimizer/wfless/*` 的同名 `learning_schedule(power_radius, …) -> (lr, delta)` 是**另一个函数**（SPGD 自适应），不在本条范围 | [#54](https://github.com/ywzhang909/AO-shaping/issues/54) |

> **旧账本 §2.2 的 R-30 / R-31 两行是陈旧的** —— 它们实际已于 2026-10-03 完成
> （见 [archive §5.1](docs/dev/todo_archive.md)），但表格未同步更新。**不要**为它们开 issue。

### 2.3 `tools/slm/` CLI 层重构

**只动 CLI 层，不合并命令体** —— settle/slot/dark-frame 是位置敏感步骤，
挪动会产生**看似可信的错数据**。

| ID | 摘要 | issue |
|---|---|---|
| — | ~~R-37 迁 19 个探针 / R-42 迁 4 个 argparse 探针 / R-38 拒做 / R-41 保留两种 flag 拼写~~ | ✅ 全部完成 → archive §5.15 / §5.31 / §5.19 / §5.18 |

### 2.4 `src/sim/` 顶层仿真包迁移

> **起因**: `drivers/sim/` 把仿真设备层深埋在硬件 drivers 内, 违反「硬件包不得
> import 仿真包」的分层原则 (F-10 #57); 同时 13 个根目录散 .py 文件组织混乱,
> internal import 路径冗长 (`ao_shaping.drivers.sim.*`).
> **方案**: 提升为顶层 `src/sim/` 包 (与 `ml/` 同级), 并按功能重组目录结构.
>
> **⚠️ 命名历史 (2026-10-19 复核)**: 顶层 `src/sim/` 这个路径名 2026-03-11 曾被一个
> **独立**的 `digitaltwin` 仿真框架短暂占用 (commit `e1c1d638` 创建, 2026-03-22 commit
> `bbe73d8c` 整体删除, 现无 `src/sim/` 目录). 那个 `digitaltwin` 框架与本节的 `drivers/sim/`
> 提升方案**无关** (前者是独立数字孪生实现, 后者是把 `drivers/sim/` 27 个 .py 提为顶层包,
> 从未实施). 该方案若执行, 将复用 `src/sim/` 这个已腾空的路径名, 勿与已废弃的 `digitaltwin` 混淆.

**迁移内容 (2 大类):**

| 来源 | 目标 | 说明 |
|---|---|---|
| `drivers/sim/` **27 个 .py** | `sim/` 顶层包 | 设备模拟层 (约 5,000 LOC) |
| `optimizer/wfless/strehl_sim_eval.py` | `sim/optimizer/strehl_eval.py` | `StrehlLandscape`, 无 optimizer 依赖 |
| `optimizer/wfless/sim_spgd.py` | **保留** `optimizer/wfless/sim_spgd.py` | ❌ 依赖 `optimizer.constants.create_optimizer`, 违反 sim→optimizer 禁止原则 |
| `optimizer/rl/envs.py` 中的 3 个类 | `sim/rl/turbulence_env.py` | `SimTurbulenceAOEnv`, `_BaseSimAOEnv`, `StaticAberrationAOEnv` |
| `optimizer/rl/envs.py` 中余下类 | **保留** `envs.py` | `LaserCastEnv`, `TraditionalAOEnv` (非 sim) |
| `utils/slm/gs_debug_optics.py` | **保留** | `grayscale_to_radians` 仍被 scripts/tests 使用; `idealized_far_field` **也未弃用**（2026-10-25 实测：47 行模块内 `deprecat`、`弃用`、`legacy` 一类标记命中 **0**，且被 `tests/ao_shaping/utils/test_gs_debug_optics.py:18-24` 真实调用并断言、被 `scripts/slm/generate_gs_refine_fwdmodel_report.py:280` 报告正文引用）⇒ 它是**仅测试与报告引用的 numpy 查看辅助**，不是死代码；原括注 `(AGANTS anti-pattern)` 两处皆误：拼写应为 AGENTS，且 `AGENTS.md` 里**没有**关于该符号的反模式条目（全文仅在结构表把它当模块名列出）|

**新包结构 (根目录散文件 → 子目录归类):**

| 原位置 (驱动层) | 新位置 | 归类理由 |
|---|---|---|
| `drivers/sim/base.py` | `sim/base.py` | 基类, 留在根 |
| `drivers/sim/compat.py` | `sim/compat.py` | `AOConfig`/`TraditionalAOSystem`, 留在根 |
| `drivers/sim/_oopao_compat.py` | `sim/_oopao_compat.py` | OOPAO shim, 留在根 |
| `drivers/sim/beam_backend.py` | `sim/physics/beam.py` | 光学传播 + GS 算法 |
| `drivers/sim/beam_simulation.py` | `sim/physics/beam.py` | **合并** (adapter 模式, 见 archive §7.1) |
| `drivers/sim/wave.py` | `sim/physics/wave.py` | 波传播器 |
| `drivers/sim/disturbance.py` | `sim/physics/disturbance.py` | 干扰/湍流 |
| `drivers/sim/dm_optics.py` | `sim/physics/dm_optics.py` | DM 光学耦合 |
| `drivers/sim/oopao_backend.py` | `sim/oopao/backend.py` | OOPAO 后端 |
| `drivers/sim/slm_shaping_bench.py` | `sim/bench/shaping.py` | SLM 整形台 |
| `drivers/sim/slm_pib_sim.py` | `sim/bench/pib.py` | SimSLMPib + register_sim_camera |
| `drivers/sim/sim_bench_patch.py` | `sim/bench/patch.py` | runner patch helper |
| `drivers/sim/fouriergsnet_env.py` | `sim/bench/fouriergsnet.py` | FourierGSNet env |
| `drivers/sim/atmos/` | `sim/device/atmos/` | 设备模拟器 |
| `drivers/sim/ccd/` | `sim/device/ccd/` | 设备模拟器 |
| `drivers/sim/dm/` | `sim/device/dm/` | 设备模拟器 |
| `drivers/sim/laser/` | `sim/device/laser/` | 设备模拟器 |
| `drivers/sim/optics/` | `sim/device/optics/` | 设备模拟器 |
| `drivers/sim/wfs/` | `sim/device/wfs/` | 设备模拟器 |

**依赖方向 (迁移后):**
```
sim.* ──→ ao_shaping.drivers.* (设备基类, 单向: sim → hardware base)
sim.* ──→ ao_shaping.utils.* / algorithm.* / model.* (单向)
ao_shaping.drivers.* ─lazy─→ sim.* (PEP 562 __getattr__, 现有机制)
ao_shaping.optimizer.* ─func-local─→ sim.* (现有模式)
ao_shaping.runners.* ─func-local─→ sim.* (现有模式)
ao_shaping/__init__.py ─lazy/shim─→ sim.*
```

> **硬约束**: `sim/` **不得** import `ao_shaping.optimizer.*` — `sim_spgd.py` 因依赖 `optimizer.constants.create_optimizer` 无法移动。`optimizer.constants` 仅 import `ao_shaping.algorithm.gradient.adam`, 如需解耦可考虑将 `create_optimizer` 下沉到 `algorithm/gradient/factory.py` (R-46 #62 前置).

**关键更改清单:**

1. `pyproject.toml`: `include = ["ao_shaping*", "ml*", "sim*"]`
2. `drivers/__init__.py` `_LAZY_BACKENDS`: `ao_shaping.drivers.sim.dm` → `sim.device.dm.simulated_dm`
3. `drivers/dm/__init__.py` + `_registry.py`: `ao_shaping.drivers.sim.dm` → `sim.device.dm.*`
4. `ao_shaping/__init__.py`: 直接 `from ao_shaping.drivers.sim.*` 改为 lazy/shim
5. 38 src + 14 scripts + 20 tests: `sed s/ao_shaping.drivers.sim/sim/g` (注意 `sim.atmos/` → `sim/device/atmos/`)
6. **shim 包** `src/ao_shaping/drivers/sim/__init__.py` (6 个月过渡, DeprecationWarning)

---

## 3. 离线缺陷修复 / 文档（无需硬件）

| ID | 摘要 | issue |
|---|---|---|
| ~~F-4~~ | ✅ **决策已定（2026-10-14）：保留 `src/ml/` 为独立顶层包，不迁移**（依赖方向是 ml→主包，torch 可选性靠 PEP-562 惰性隔离已验证，迁移 155 文件零收益）。`gsnet_runner.py` 顶层 `from ml...` 属 GSNet 专用 runner、被 `_LAZY_RUNNERS` 惰性加载，`import ao_shaping.runners` 不拖 torch，**无需改动** | [#55](https://github.com/ywzhang909/AO-shaping/issues/55) |
| F-5 | `docs/issues_report.md` §11 代码规范整改。**已重新扫描（2026-10-04），原文数字低估 5～14 倍** ⇒ 建议按切片做。**切片 1 已完成（2026-10-14）：`src/ao_shaping` + `scripts` 41 个文件补齐 `from __future__ import annotations`（tests 67 与 `__init__.py` 39 有意跳过）** → archive §5.36；余 3 片（print→loguru / 宽泛 except / 大文件）未动 | [#56](https://github.com/ywzhang909/AO-shaping/issues/56) |
| F-10 | ✅ 已解决 (2026-10-13)：单因子改写其实**已于 2026-10-02 落地** (commit `284fbb5`)，`sim/AGENTS.md` 第 3/4 条与代码已对齐 (第 4 条改为「已落地」)。`report/oopao_impact/oopao_impact_report/report.md` 的 1.068/2.628 是改写前旧公式 (commit `5e12fa2`) 的实测值，随公式变更现已不可复现，须重跑生成器更新 | [#57](https://github.com/ywzhang909/AO-shaping/issues/57) |
| F-11 | ✅ **已收口 (2026-10-25)**：三路序列号对应**两台 SLM**（用户 2026-10-05 定论"实验室有 532/1064 两台"，关闭 #58 时定调"保留配置文件默认行为即可"）：SLM#1 = **22030108 @1064nm 2π=993**（唯一读数，权威）；532nm 台 = **23020026 @532nm 2π=998**（2026-09-16 起各报告一致）；**22030102** 为 532nm 台 09-15 标定 session 的早期读数（`daily_2026-09-15.md` §9 明文"⚠️ 与前几轮 22030102 不同"，09-16 起统一 23020026），2π=998 两读数一致系 532nm 台属性。结论已回写 `drivers/AGENTS.md` 硬件表（新增 SLM#2 行 + 引用规则）；代码侧 `slm-gs-refine --slm_wavelength` 默认 0=询问设备波长保持不变，满足定调。532nm 台物理序列号的最终设备端核对（needs-hardware）不再阻塞，如需对 532nm 台做设备端序列号复核可另开 issue | [#58](https://github.com/ywzhang909/AO-shaping/issues/58) |
| F-12 | 🔴 **焦面标定常数三方不一致**：`5021/Λ` / 7400–7600 / 主张 10954。⚠️ 2.2 µm 像元推得 ~7557 而非 10954 ⇒ **三方无一是已验证的** | [#59](https://github.com/ywzhang909/AO-shaping/issues/59) |
| ~~F-14~~ | ✅ **已修**（2026-10-14）：**30 → 0 个 `F821`**，`ruff.toml` 的 `exclude = ["src/ml/**"]` 已删除。根因比原判断更严重 —— 不只是 `DataLoader`：`build_gan_models`（`ml.train` 3 处引用）**在全部提交历史里从未被定义过**；`build_model` 在 `40d65af` 删除后**接线被遗忘**（`_zernike_regression_removed` 定义了却零调用）；另有 `d1645486` 把 coeffs 模式整块误粘进了 `_train_phase_mode`。`ao-train`（`pyproject.toml` 注册的 console script）三条命令**全部**因 `NameError` 不可运行，且零测试覆盖。证据与守卫见 `tests/ao_shaping/ml/test_train_cli_runnable.py` | — 待开 issue（API 被网络拦截，无法从本机创建） |

> **F-12 未定案前不要单独改 H-16 的 `K`** —— 否则只是把一个未验证的数从一处搬到另外两处。
> **F-14 与 §8 的 R-63 互为前置**：F-14 是 R-63 第 4 步（放开 `src/ml/**`）必须先清掉的真实缺陷。
> **F-14 已于 2026-10-14 完成 ⇒ §8.6 第 4 步的前置已清**，且 `exclude` 已实际删除。
> ⚠️ 但 §8.6 的**顺序仍然有效**：下一步（§6「第 1.7 批」）是先做 R-63 的**第 1–3 步**
> （代码层 noqa + 配置分层），**最后**才轮到第 4 步；RUF100 仍不得先跑。

---

## 4. 原文档结论需修正（照做会出事）→ 已结案

| ID | 原结论 | 实际情况 |
|---|---|---|
| X-1 | `TODO.md` #16：「`utils/image/resample.py` **0 导入** → 删模块」 | ✅ **前提已消失** —— 现在有 1 个生产导入方（`FourierGSNet.py:42`），照原文删会打断 FourierGSNet。**且 `TODO.md` 本身已不存在**（已并入本账本） |
| ~~X-2~~ | ✅ **已加显式标记**（commit `f4b9afee`，2026-10-14）：`tools/slm/TODO.md` D2 的 min-max 是**误报** —— 那处 `P` 是实测功率-扫描位置曲线（`power()` 返回 `float(p.sum())`），归一化**强度轴**后 `np.interp` 找 50/84/16% CDF 分位点是标准做法，与相位/尺度无关性无关。陷阱在**代码**里（读 `calibration.py` 的人看不到 markdown 论证），故把标记**就地**加在 `calibration/calibration.py` 的 `find_beam_on_slm`（原引 `:2182` 行号已漂移，真身在 `:2204`）：一句不变量 + 指向 D2/archive §4 X-2。纯注释，行为不变（`test_slm_calibration_{pure,windowed}` 17 项绿） | [#60](https://github.com/ywzhang909/AO-shaping/issues/60) |
| X-3 | 两个 runner 的 `ZERNIKE_APERTURE_RADIUS` 不同（300 vs 600） | ✅ **2026-10-04 解开，原结论前提是错的** —— `slm_zernike_shaping` 无任何生产导入方（`rms-zernike` 走 `optimizer.wf.rms_by_zernike`），所以**不存在「换个 runner 就换光阑」的生产风险**，600 是死代码。详见 archive §5.17 |

---

## 5. 已完成 → [`docs/dev/todo_archive.md`](docs/dev/todo_archive.md)

**87% 的原账本在这里**（1625 行正文 + 42 行头部说明，§ 编号不变）。它不是变更日志的礼貌性副本，
每一条都是**一次决策的证据**，作用是**阻止未来的重复动作** ——
例如「D3/D4/D5 判定不可合并」附实测数值（`zoom` 与手写双线性在升采样时差
0.35~0.38 peak）、「抽共享 dataclass 被实测证伪」、「两处 `1e3` 故意不统一」。

## 6. 建议执行顺序

| 批次 | 内容 |
|---|---|
| ~~第 1 批（纯收益，零行为风险）~~ | ~~R-23、R-24、R-30、R-31、F-2、F-6、F-7、F-8、R-40~~ ✅ 2026-10-03 → archive §5.1 |
| **第 1.5 批（文档/常量收口，先定事实再改代码）** | **F-12** → **F-11** → **F-10**（#59 → #58 → #57）。**F-12 是 H-11 / H-16 的前置**，未定案前不要改 `K` |
| ~~第 1.7 批 首项（lint 抑制层，纯离线，见 §8）~~ | ~~**F-14**（`src/ml/` 30 个 F821，含必然 `NameError`）~~ ✅ 2026-10-14 |
| **第 1.7 批 剩余（lint 抑制层）** | **R-63**（73 条 noqa 分类 + `ruff.toml` 分层，**严格按 §8.6 的 5 步顺序**：F-14 已完成 ⇒ 第 4 步的排除已删除，可从第 1 步继续，但 RUF100 仍不得先跑）→ **R-64**（实测 **BLE001 518 处**（2026-10-25 重钉，原 2026-10-24 记 522：`61fa586a` 把 `test_spots_calc.py` 4 处 CuPy `except:pass` 重构为 `pytest.skip`，522 减 4；原 514；原估 312）+ **`src/` assert 当运行期护栏**：2026-10-25 重钉，现树 `src/` `assert` 共 **36**（`drivers/` 5 + 其外 31，全为本地不变式，按 §8.10 保留）；原记 99/111 与"含硬件安全联锁"系各簇收口**前**口径，实测设备状态类 `assert` 现 **0**（守卫见 §8.9 第 13 项）。✅ WFS 曝光安全上限簇已收口（`e6f80254`）＋ **CCD「设备已打开」簇已收口**（`de5b9ccd`，`BaseCamera._ensure_open()` 抛 `CameraError`，`drivers/ccd` 设备状态 `assert` 实测归 0），详见 §8.9） |
| ~~第 2 批（止真 bug）~~ | ~~R-1~R-4、R-9~R-11、R-35~~ ✅ → archive §5.2 / §5.3 |
| ~~第 3 批（架构重构）~~ | ~~R-20、R-21、R-22、R-25、R-26、R-27、R-28、R-32、R-36、R-37、R-41、R-42、F-14、F-15~~ ✅ → archive §5.4–§5.13 |
| **第 3 批剩余（离线）** | ~~**R-29**~~ ✅ #50 ⇒ ~~**R-33**~~ ✅ #51（上游已实现，判据已失效）⇒ ~~**R-43**~~ ✅ #53（补 38 项测试）⇒ ~~**R-39**~~ ✅ 实测判定无需改代码（见 §2.2；#52 仍开放待硬件） |
| **第 3.5 批（函数归属，2026-10-06 新增，见 §7）** | ~~**R-45**（#61）⇒ 误报，已结案~~ ⇒ **R-58**（#77，叶子层唯一的真违规）⇒ ~~**R-46**（#62 optimizer→tools，`gaussian_grid` 下沉 `utils/wavefront/beam_amplitude`）~~ ✅（`57ba939c`）⇒ ~~**R-47**（#63 面板几何下沉）~~ ✅（`78a9d3d7`）⇒ ~~**R-48**（#64 帧分析下沉）~~ ✅（`4d068c8f`）⇒ ~~**R-49**（#65 新建 `utils/math`）~~ ✅（`252d81e1`）⇒ ~~**R-50**（#66 Zernike 第三套 API 层）~~ ✅（重复实现已在 `531fc1fb`/`6d5e0f6d` 收口为委托；本仓补数值特征测试 `tests/ao_shaping/utils/wavefront/test_zernike_single_home.py`，见 §7 R-50 行） ⇒ ~~**R-51**（#67 重复收口）~~ ✅（`290ee51`，第三项 `power_bucket` 拆 R-59/#78）⇒ ~~**R-52**（#68 报告脚本参数化）~~ ✅（`3a5b23f`）⇒ ~~**R-53**（#69 `compute_metrics` 改名）~~ ✅（`3aece31`）⇒ ~~**R-54**（#70 runner 重复）~~ ✅（误报 not-planned）⇒ ~~**R-55**（#71 叶子层 matplotlib）~~ ✅ ⇒ **R-56**（#72 optimizer 内渲染面）🟡部分（item 42-45 pygame 状态机 + 47-48 开放；~~无 `__main__` 守卫的 3 文件拆 R-61~~ ⇒ ~~**R-61**（#81）~~ ✅（误报：AST 复核证伪，3 文件均有 `__main__` 守卫 + 渲染 import 已函数局部，`test_rl_no_module_rendering_import.py` 钉死））。**X-4**（#74 死分叉单向移植 ~80 LOC）待做；**F-13**（#73）已结案 |
| **第 4 批（内部重构，需先补特征测试）** | **R-19**（#49，父项）⇒ **R-5 / R-6 / R-7 / R-8**（#42–#45）；**R-14**（#47，需先做产品决策）；**R-13**（#46）；~~**R-44**（#54）~~ ✅（`7284dc1f` 已修，见 §2.2）；**R-18**（#48，接入已有实现或删注释） |
| **纯离线但要实验系统数据** | ~~**F-4**~~ ✅ 已决策（#55，2026-10-14：保留 `src/ml/` 不迁移）；**F-5**（#56，先切片）；~~**H-23**~~ ✅（`8a756060` 已实施，`Closes` #40）；~~**X-2**（#60，加误报标记）~~ ✅（`f4b9afee` 已加代码标记） |
| **硬件轨道（并行，需设备在线）** | **H-7~H-13**（复测前必修）⇒ **H-19 + H-9 合并做**（ABBA 对消，`dec>0.`55` 且 `late_gain≥10%`）⇒ **H-14 复扫 δ** ⇒ **H-15 / H-16 / F-12**（曝光回读 + 标定常数）⇒ **H-1 / H-2** ⇒ **H-3~H-6**（FourierGSNet）⇒ **H-17 / H-18**。**H-20 / H-21 / H-22** 独立于上述轨道（合并回归 + `--delta` A/B），**H-24** 需先有实验系统 `--debug` 产物。**H-33**（GS→Zernike→SPGD 阶数扫描）同样独立成轨：它的两个 P0 门（GS bake-off、噪声地板 σ）与上述任何一条都无依赖，但 **P0-1 若失败须先查 F-12 / H-16 的标定常数** —— 未定案前不要单独改 `K` |

---


## 7. 函数归属审计（2026-10-06）

> 一次 4 路并行审计（plotting 越界 / `utils` 叶子违规 / 重复实现 / `tools` 包边界）
> + 1 次 `slm_zernike` 分叉评审，产出 **49 个条目**。
> **详细清单已全部搬迁为 GitHub issue**（下表），本节只保留索引 —— 符合本文件
> 「新增待办：开 issue，标题带 ID 前缀…并在本表加一行」的约定。
> 执行顺序见 §6「第 3.5 批」。

| ID | issue | 摘要 | 波次 / 风险 |
|---|---|---|---|
| ~~R-45~~ | [#61](https://github.com/ywzhang909/AO-shaping/issues/61) | ❌ **误报，已作 not-planned 结案**：14 处经 AST 核实**全部**是 README 明文许可的 `TYPE_CHECKING` / 函数内延迟导入。真正的 1 处见 R-58 | ✅ |
| ~~R-46~~ | [#62](https://github.com/ywzhang909/AO-shaping/issues/62) | `optimizer` 反向 import `tools/` 探针模块（`slm_model_in_loop` 两处） | ✅ B 🟢 |
| ~~R-47~~ | [#63](https://github.com/ywzhang909/AO-shaping/issues/63) | ✅ **已收口**（commit `78a9d3d7`）：`phase_to_panel` + 5 个面板常量（`SLM_PANEL_W/H`、`SLM_PITCH_M`、`BEAM_CENTER_PANEL`、`BEAM_RADIUS_PANEL`）从 `tools/kernels/bench_kernels` 下沉到叶子 `utils/slm/`（常量入 `bench_constants`，函数入新建 `panel_geometry`）。`bench_kernels` re-export 全部 6 名（留在 `__all__`）⇒ 13+ importer 与 monkeypatch 测试零改动；删除随之失效的 `scipy.ndimage.zoom` 导入。**命名陷阱**已在 `panel_geometry` 模块 docstring 交叉引用：`phase_to_panel` 是**空间放置**（model-grid 弧度→面板尺寸弧度，盘外置零），**不是** `utils/slm/phase_display.phase_to_slm_grayscale`（弧度→uint16 灰度）——两个变换不可合并。**test-first**：`test_panel_geometry.py` 先钉死数值行为（面板形状/dtype、非方 ValueError、单位 zoom 恒等放置、盘外置零、常数盘保持、偏心裁剪在界内），迁移后改指叶子 + 加 re-export **同一性**校验 + AST 叶子守卫；`test_bench_constants_leaf.py` 加钉 5 常量精确值。容器 blast-radius **333 passed**。src 改动 167 行（<200）⇒ 直推 `banckend` + `Closes #63` | ✅ |
| ~~R-48~~ | [#64](https://github.com/ywzhang909/AO-shaping/issues/64) | ✅ **已收口**（commit `4d068c8f`）：13 个纯数组符号从 `tools/kernels/bench_kernels` 下沉到叶子 `utils/image/frame_analysis`（`SpotMeasurement`/`smooth_frame`/`estimate_shift`/`measure_spot`/`core_fraction`/`finite_clip`/`finite_median_subtract`/`crop_around_zero_order`/`crop_roi`/`roi_l2`/`flat_to_flat_floor`/`_as_2d_float` + `despike_frame`）。`despike_frame` 与 `beam_metrics` 的**逐字节重复**已去重（单一来源）。`bench_kernels` re-export 全部公开名 ⇒ ~15 个 importer 零改动。`_spot_fwhm`（`measure_spot` 依赖）原为 `model_in_loop_shaping` 私有 ⇒ 提为 `frame_analysis.spot_fwhm`，`model_in_loop_shaping` 保留 `_spot_fwhm` 别名（4 内部调用 + 2 测试 importer 不动）。`frame_analysis` 保持叶子（仅 numpy + `beam_metrics`）：其 `measure_spot` 调**本地** `spot_fwhm`，不回指 optimizer（无循环、无分层违规）。守卫 `test_frame_analysis.py` 钉死 re-export 同一性 / despike 单源 / 别名。容器验证：blast-radius 17 文件 **686 passed**（3 失败为 `slm_panel_locate` golden 陈旧，已在 HEAD `a8ed482b` 复现证明**先于本改动**，与 #64 无关） | ✅ |
| ~~R-49~~ | [#65](https://github.com/ywzhang909/AO-shaping/issues/65) | ✅ **已收口**（commit `252d81e1`）：4 个纯 numpy 符号从 `tools/kernels/bench_kernels` 下沉到新建叶子 `utils/math`（`fitting.py`: `fit_linear_slope`；`stability.py`: `settle_time_s`/`snr_vs_averages`/`exposure_monotonicity`）。`bench_kernels` re-export 四个名字 ⇒ tilt/floor/drift 探针消费者与现有行为测试零改动；`Sequence` 死导入已删。`utils/math` 保持叶子（仅 stdlib + numpy）。守卫 `test_math_leaf.py` 钉死 AST 叶子契约 / re-export 同一性 / canonical 路径行为 smoke。容器验证：blast-radius 5 文件 **117 passed** | ✅ |
| R-50 | [#66](https://github.com/ywzhang909/AO-shaping/issues/66) | ⚠️ **2026-10-23 复核：本行 ②③ 已过时，"待决策"的前提已不存在**。①（2026-10-06 结论不变）`PatternHelper.generate_zernike_polynomial:550` 是薄适配器（孔径外置0 + `nan_to_num` + raw 弧度），被 ~15 文件依赖 ⇒ 不可删；其"唯一入口"不实声称已订正（`82998b9`）。② ✅ **重复实现已收口**：`beam_simulation.py:74` 的手写多项式由 `531fc1fb`（2026-10-14）删除，`6d5e0f6d`（2026-10-17，A-3）以**委托** canonical `zernike_radial` 的形式补回（现 `beam_simulation.py:72-99`）⇒ 今天不存在"两份不可互换"；本行的 18 个尺度因子 / 7 个真不同多项式（最大绝对差 3.243）只在 `531fc1fb~1` 的历史版本之间成立。③ ❌ **原"零生产调用方"声称是错的**：`optimizer/rl/envs.py:842`（`StaticAberrationAOEnv._sample_disturbance`，经 `sac_train.py:49` 实例化、`_BaseSimAOEnv.reset` envs.py:640 调用）一直在调用 `bs.zernike_polynomial` ⇒ `531fc1fb` 以"全仓零调用点"删函数后，**`531fc1fb`→`6d5e0f6d` 之间 RL 环境 `reset()` 必然 `AttributeError`**（无人发现，因为 RL 测试只断 shape 且 `importorskip("gymnasium")`）。④ 这次换实现把径向从教科书未归一化换成 aotools `RZern` ⇒ **`envs.py` 的 RL 相位量纲已被静默改动，当时零测试阻力**：`zernike_radial` 仅有的 3 个测试只断言 `shape`，`beam_simulation` 在全仓测试里**没有任何引用**。✅ 已补数值特征测试 `tests/ao_shaping/utils/wavefront/test_zernike_single_home.py`（6 项，在默认发现集内 ⇒ CI 会跑）：委托逐位相同（`rtol=0, atol=0`）、**教科书 × 约定尺度**（`sqrt(n+1)` / `sqrt(2(n+1))`，实测比值逐模式恒定 ⇒ 本行"仅差尺度"的口径对已探模式成立）、瞳边缘归一化指纹、Noll 20 模式半瞳数值、`m<0` 选 sin、`normalize_rho` 特征钉。**遗留待查（不作断言）**：`normalize_rho` 除以**网格最远点**而非瞳半径 ⇒ 圆瞳边缘 `rho < 1`（特征测试已钉现状并注明"不代表认可"），是否有意需 RL 侧确认 | ✅ 已收口（含历史缺陷记录 + 数值守卫） |
| ~~R-51~~ | [#67](https://github.com/ywzhang909/AO-shaping/issues/67) | ✅ **2/3 已收口**（`290ee51`）：`flat_gray` 委托 `utils/slm_phase`（写死副本是「面板尺寸一变就静默不一致」）；`create_target_mask` 两份**去docstring 后逐字节相同**，改为由 `algorithm/…/differentiable_shaping` re-export `utils/image/target/patterns`（utils 在 algorithm 之下，方向正确）。❌ 第三项 `power_bucket` **删除与委托皆不可** → 拆出 R-59（#78） | ✅ |
| ~~R-52~~ | [#68](https://github.com/ywzhang909/AO-shaping/issues/68) | ✅ **2/3 收口**（`3a5b23f`）：`heuristic_pib` 与 `strehl_benchmark` 两份 `plot_summary_bars` 完全同构（只差 metric key / xlabel / title）→ 提为 `scripts/_common.plot_summary_bars`，其余常量全部钉死为两份原本的取值，**两个调用点 PNG 逐字节一致已验证**（21101B / 23372B），不重写任何已提交图。第三份 `generate_slm_pib_rms_pib_report` **刻意不合**：它按行数缩放图高、xlim 由数据动态导出、dpi=110 且无 `bbox_inches`、**有返回值**，强行合并等于把一套契约强加给两者（与 `_common` 五个 formatter 刻意不合并同理） | ✅ |
| ~~R-53~~ | [#69](https://github.com/ywzhang909/AO-shaping/issues/69) | ✅ **改名消歧完成**（`3aece31`）：三份 `compute_metrics` 是**三种真实契约**、合并会破坏至少两个调用方，故只改名不合并 —— `compute_beam_metrics`（dict: mse/correlation/efficiency，canonical，20+ 调用方）· `compute_bench_metrics`（dict: center/zero_order_margin_px）· `compute_cv_ee`（tuple: CV,EE）。60 处 / 16 文件，**无任何文件同时引用两者**，故按文件定向改名安全；docstring / AGENTS.md / 两处导航型 report README 同步更新。新增守卫 `test_compute_metrics_renamed.py`：裸名再现、三名缺失、两契约塌缩为同一 docstring 均判失败 | ✅ |
| ~~R-54~~ | [#70](https://github.com/ywzhang909/AO-shaping/issues/70) | ❌ **误报，`not planned` 结案**：两种口径都量过 —— **≥6 行的相同块 0 处**（0% of `gsnet_runner.py`）；7 个同名函数**函数体全部不同**（`_build_square_config` 68 vs 46、`_execute` 19 vs 43 行…）。差异是**契约级**：`ObjectiveParamsSquare` + `SlmGsnetConfig` vs `CameraParamsPib`/`SlmParamsPib` + 位置传参、freeform vs Zernike、目标函数不同。收敛需先统一 param 类型（设计变更 + 硬件复测），不是 copy-paste 抽取。⚠️ 本 issue 的硬约束（settle/槽轮换/dark-frame 位置敏感）正说明**无真实重复时不该动手** | ✅ |
| ~~R-55~~ | [#71](https://github.com/ywzhang909/AO-shaping/issues/71) | ✅ **item 1 修复，item 2/3 原非违规**：`utils/io/file.py` 的模块级 `import matplotlib.pyplot` 已下沉到 4 个出图函数内（`Recorder.save_best` / `Recorder.plot` / `_save_data_mode_debug_artifacts` / `save_optimization_debug_artifacts`），不引入新包依赖故无循环风险（该文件第 727 行已有 `display` 的延迟导入）。❌ item 2/3（`spots_calc.disp` / `hardware_utils.save_frame_png`）**本来就是函数内延迟导入**，不违反叶子层规则 —— issue 对它们的抱怨是「渲染函数混进纯计算模块」，属另一件事。新增守卫 `test_utils_leaf_no_rendering_import.py`（含 #50 的 `display.py`  carve-out）。⚠️ 诚实说明：**契约 R-20（`import ao_shaping.utils` 不拖入 matplotlib）由 `ao_shaping/__init__.py` 破坏，非 `utils/`**（实测所有 delta = 0） | ✅ |
| ~~R-56~~ | [#72](https://github.com/ywzhang909/AO-shaping/issues/72) | ✅ **item 46 可下沉部分完成**：`sac_common` / `pib` / `pib_sim_eval` / `gready_cam` / `adc_dm_adam` 共 5 文件的模块级 matplotlib/pygame 已下沉到实际使用它的函数（`pib_sim_eval`、`sac_common` 的 `matplotlib.use("Agg")` 全局副作用随同搬走，不可与 pyplot 拆分）。⚠️ **issue 的 item 46 指令「下沉进 `__main__`」对 6 个文件中的 5 个是错的** —— 原拆 R-61 的理由（`lr.py` / `lr_wfs.py` / `rl_wfs.py` 在**模块级**用 plt 且**无 `if __name__` 守卫**）经 AST 复核（2026-10-14）**证伪**：三文件**都有** `if __name__ == "__main__":` 守卫（`lr.py:149` / `lr_wfs.py:322` / `rl_wfs.py:348`），全部 `matplotlib.pyplot` import 都是**函数内局部**（`_on_step` 内）或在守卫内，模块体**零**渲染 import、**零**裸 plot 调用。唯一的模块级副作用是刻意的 `os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'`（OpenMP/MKL 规避，非渲染面，勿动）。→ **R-61（#81）误报，已结案**，状态由 `tests/ao_shaping/optimizer/rl/test_rl_no_module_rendering_import.py` 钉死。item 42-45（pygame 状态机迁移）与 47-48 仍开放 | 🟡 |
| F-13 | [#73](https://github.com/ywzhang909/AO-shaping/issues/73) | ~~`pyarrow_probe` 两份~~ **已结案**：`scripts/` 副本零导入方且与 `gui/slm` 那份**归一化行尾后逐字节相同**，已删 | ✅ 619d2d0/待本轮 |
| X-4 | [#74](https://github.com/ywzhang909/AO-shaping/issues/74) | 死分叉结论修正：**单向移植 ~80 LOC，不是删除** | F 🟡 |
| R-29 | [#50](https://github.com/ywzhang909/AO-shaping/issues/50) | ✅ 已完成：`utils/image/display.py` → `display/image_display.py`（commit `a9236e9`，2026-10-09），re-export 已改指 `ao_shaping.display.image_display`，无兼容 shim | E ✅ |
| ~~R-58~~ | [#77](https://github.com/ywzhang909/AO-shaping/issues/77) | ❌ **误报，已 `not planned` 结案**：`phase_display.py` 的模块级 `PhaseMap` 导入**增量 0 个 `drivers.*` 模块**（原判据的 60 个其实来自父包 `ao_shaping/__init__.py` 的 20 处 eager 再导出）；`model/field.py` 唯一的 `drivers` 导入已在函数内延迟（`model/field.py:43`，注释说明是为规避循环）；`model/` 亦不在 utils 的禁止清单内，且 README 已把 `phase_to_slm_grayscale()` 接受 `PhaseMap` 列为既定契约 | ✅ |
| ~~R-59~~ | [#78](https://github.com/ywzhang909/AO-shaping/issues/78) | `power_bucket` 两份**归一化语义不同**（`wave.py` 裸桶内和 vs `spots_calc` 的 `weighted`/`use_dpix_scaling`），**委托会改数值、删除会打断活测试** → 用户拍板**方案 3 改名消歧**（`612f8cad`，2026-10-16）：canonical `spots_calc.power_bucket`（dpix 缩放+weighted）保留原名，`wave.py` 裸求和版改名 `power_bucket_sum`，两处 docstring 标注量纲差异，不改数值行为；`docs/simulation.md` 同步改名 | ✅ |
| ~~R-57~~ | — （已落地，无需 issue） | 峰峰值安全已由软惩罚改为**可行性门**：CLI `--max-peak`（相机计数，默认 `0`=关闭），超限则**不提交该轮更新**，镜像 fold 门（**完全不调** `optimizer.update`，以免用无据的编辑污染 Adam 动量；不追加 `_diff_history`、不刷新基线、不 best-track），违规轮诚实记录 `gate="peak"` 并从 "updates applied" 扣除。**⚠️ 它不是投影也不是安全联锁**：±d 帧在被检查前已出光、提交点要到下一轮才显示（滞后一轮）、只约束相机窗内计数、且 `exposure_time_ms=0` 的自动曝光会掩盖它（该组合下运行器显式告警）。真激光安全仍须带外硬件联锁（IEC 60825-1）。依据见 `report/pib_loss_terms/README.md` §4.1 | E ✅ |

| ~~R-62~~ | [#83](https://github.com/ywzhang909/AO-shaping/issues/83) | ✅ **收敛完成**：3 处 best-frame PNG saver 提为 `display/frames.py::save_best_image(img, out_path, *, title, marker_side=None, hide_axes=False)`（从 `display` **子模块**导入，避免拉进 `windows.py` 的 pygame —— 实测导入后 pygame 未加载）。差异全部参数化而非抹平：**红色十字**（方形 runner 的目标方斑半对角参考）、`axis("off")`、文件名（`best_square.png` vs `best_far_field.png`）、各 runner 自己的标题指标。第三份原用 `plt.savefig`，统一为 `fig.savefig`（新建 figure 上等价，且不依赖 pyplot 隐式当前图）。文件名与标题不变 ⇒ 不改写任何已提交报告 | ✅ |
| R-46 | [#62](https://github.com/ywzhang909/AO-shaping/issues/62) | ✅ **已收口**。item 1（#63/R-47 `78a9d3d7`）+ item 2（`capture_settled` 移入 `bench_kernels`）此前已完成。本轮收口最后一条模块级 `optimizer→tools` 边：`gaussian_grid`（纯 numpy）从 `tools/kernels/bench_kernels.py` 下沉到叶子 `utils/wavefront/beam_amplitude.py`（新建，仅 import numpy），`bench_kernels` re-export（`a is b` 已验证，`__all__` 不变），`slm_model_in_loop` 改从叶子 import。`capture_settled` 是**函数内延迟 import 的设备内核**（非模块级边），按 `slm_model_in_loop.py` 原注释「last remaining optimizer->tools edge」判定为已接受边界，不动。守卫：新建 `tests/ao_shaping/utils/wavefront/test_beam_amplitude.py`（AST 叶子守卫 + re-export 同一性 + 数值不变量，含 odd-region 浮点孔径 `region/2.0`（阈值 56.25）vs 整型 `region//2`（49）的边界锁定，防 R-51 合并时漂移）；更新 `test_slm_model_in_loop.py::test_declared_module_level_dependencies_are_the_documented_ones` 依赖 pin（`bench_kernels`→`beam_amplitude`）并加负断言防边复活。⚠️ `gaussian_grid` 与 `model_in_loop_shaping._pupil_amplitude`/`amplitude` 是**三份非字节相同**的近重复（孔径半径 float vs int-clamp），合并属行为敏感，归 **R-51（#67）** | ✅ |
### 7.1 护栏：以下**不是**待办，**不得**当作重复去「修」

这批是审计的**假阳性**，没有对应 issue（因为不是待办）。记在这里只为阻止未来的
重复动作 —— 当初审计差点把它们当成缺陷改掉。

| 项 | 为什么不是重复 |
|---|---|
| `far_field_intensity`：`algorithm/signal_processing/differentiable_beam.py:98` vs `scripts/.../generate_zernike_farfield_sim_report.py:62` | 一个 torch 可微、一个 numpy 2f-Fourier，**签名与语义都不同** ⇒ 纯命名撞车 |
| `apply_lens` / `propagate`：`drivers/sim/beam_backend.py:174,179` | 该处**委派**给 `beam_simulation.py:60` ⇒ adapter 模式，不是复制 |
| `utils/wavefront/matrix_utils.py` 的 `calc_n_zernike_terms` | **有 docstring 声明的 re-export shim** ⇒ 保留 |
| `prepare_roi_frame`：`tools/slm/slm_abba_probe.py:169` vs `slm_floor_probe.py:106` | **用户钦定的有意分歧**（`finite_clip` vs `finite_median_subtract`） |
| `compute_metrics` 三份 | 三种真实契约 ⇒ 改**名**不合并，见 R-53 |
| `main` / `run`（各 entrypoint） | 合法 |
| ~~`gui/slm/pyarrow_probe.py` vs `src/ao_shaping/gui/slm/pyarrow_probe.py`~~ | **曾是误判**：原文写「已漂移（各 106 行但 SHA256 不同）」，实测**两份各 130 行、归一化行尾后 SHA256 完全相同**（`9d4a448b78d7028f`），差异**仅 CRLF vs LF**。副本已删（F-13） |

> 上述行号是 2026-10-06 审计时的快照。**动手前必须重新核对** —— 用户工作区当时有
> 大量未提交改动，行号大概率已漂移。

## 8. lint 抑制层审计（2026-10-11）

> **起因**：排查「unbounded pytest fan-out」时顺带翻到 `ruff.toml`，发现它的
> `ignore` 列表里有 `F401` / `E402`，而 `src/ml/**` 被整体 `exclude`。
> 于是 **73 条 `# noqa` 里几乎没一条在压制真实告警**（`RUF100` 实测 33 条确认为死），
> 但同一套配置把 **30 个 `F821` 未定义名**藏了起来 —— 那不是风格问题，是必然抛出的 `NameError`。
>
> **方法**：全量 `rg "noqa" src/` + `ruff check --extend-select RUF100`
> + `ruff check --isolated --select F,E,BLE,SLF,S,A`（绕过本仓配置量真实爆炸半径）
> + 4 路并行 agent 逐点分类。**结论全部经 CLI 实测；未改动任何代码。**

| ID | 摘要 | issue | 波次 |
|---|---|---|---|
| R-63 | ruff 抑制层清理：73 条 noqa 分类 + `ruff.toml` 分层重构（见 §8.3 / §8.6） | ✅ **第 1–4 步已完成**（`41cacbd2` step 1 → `8aa88fa2` step 2a → `84214f7f` step 2b；step 3 RUF100 常驻、step 4 `exclude` 已删并有护栏钉死），全局 `ignore` 现仅剩 RUF00x / isort / pyupgrade / E501 / F841 / F811 / PLC0415 / E731 / E741。**仅剩第 5 步**（`BLE001`/`SLF001`/`S301`），前置 = R-64。GitHub 出站仍被拦截 ⇒ 未能开对应 issue | A 🟡（第 5 步待 R-64） |
| R-64 | ruff 抑制层的下一片：`assert` 当运行期护栏（`python -O` 下整条消失）+ 无标注宽异常 | ⚠️ **2026-10-23 实测重量（原估数全部偏低/过时）**：**BLE001 518 条**（2026-10-25 重钉至 518，原 2026-10-24 记 522：`61fa586a` 把 `test_spots_calc.py` 4 处 CuPy `except:pass` 重构为 `pytest.skip`，4 处全在 `tests/utils`，`src/` 纯静默仍 67；原记 514 与 `src/` 355 均已过时；按 handler 行为分 **272 记日志后吞 / 152 其它 / 91 纯静默（`pass`、`continue` 且无 raise，`src/` 占 67）/ 3 重抛**，分簇见 §8.9 第 5 项），**`src/` `assert`**：2026-10-25 重钉现树共 **36**（原表 99/111 系各簇收口**前**口径），`except:` 裸捕获 **0**，`except BaseException` 5 处（2026-10-25 逐处判定：**3 处线程边界捕获禁改**且已机械钉住、1 处退出清理已复核保留、1 处本就重抛 —— 见 §8.9 第 14 项；原记「§8.7 明文禁改 2 处」是当时口径）。**已做**：#174 系列（`009e621d`/`bb64c162`/`2ceb288f`）收口 import 守卫与 `drivers/ccd/common.py`+`nlight` 的 assert→raise；本仓 `e6f80254` 收口 **WFS 曝光安全上限** 簇（6 处 assert→显式 raise、常量 3 处声明→1 处、解除与 `MIICamera.MAX_EXPOSURE_MS` 的 1400 倍同名冲突、加 AST 守卫防复发）；**CCD「设备已打开」簇已收口**（`de5b9ccd`：37 处设备状态 `assert` 收口到 `BaseCamera._ensure_open()`/`_device_open()` 抛 `CameraError`，`drivers/ccd` 实测归 0，AST 守卫 `test_ccd_drivers_never_assert_on_device_state` 钉在 CI 默认发现集内；`29001826` 把 SDK 缺失时的 `NameError`/`AttributeError` 改为 `_require_sdk()` 抛 `CameraError`）。**剩余**：~~cartographer 13 处删、optimizer 7 处删~~ **两簇均已收口**（`e4868b54` 补 `ThorlabWFS._ensure_open` 机制 + `3ba720ec` 删调用侧，实测见 §8.9）；**~40 处内部不变式保留 `assert`**（§8.10）；实测 `src/` `assert` 共 **36**、设备状态类 **0** ⇒ `assert` 半项已收口（§8.9 第 13 项守卫）；BLE001 518 条中 silent 已逐处判定完毕（§8.9 第 5-10 项），盲吞计数钉成棘轮（基线 61，第 11 项）。⚠️ 三条前置**已全部收口**：改名（`e6f80254`）；`spots_calc` 3 处实测**不是待办**（实现与测试契约互相一致，见 §8.9 第 1 项）；`DeviceParameter` **否决"把 `validate()` 接到写路径"**（实测该写缝在生产路径不可达，且接线会把硬 `raise` 降级为静默失败、并把合法哨兵 `exposure_time_ms=0.0` 判成越界），改为修自相矛盾的注册声明 + GUI 重复边界漂移（见 §8.9 第 3 项）。⇒ 原新开放项 `gui/dm/micro_dm_ui.py` 后台线程触碰 `st.session_state`（§8.9 第 4 项，原 `:642-657`）**已收口**（`987ed669`；函数现 `:783`）。GitHub 出站仍被拦截 ⇒ 未能开 issue | A 🟡（WFS 簇 + CCD 簇已收口，其余分簇推进） |

### 8.1 根因：压制发生在**配置层**，不在代码层

| 事实 | 证据 |
|---|---|
| `select = ["E","F"]` ⇒ `BLE001` / `SLF001` / `S301` / `A002` **根本没启用** | `ruff.toml:6-9` |
| 全局 `ignore` 含 `F401`、`E402` ⇒ 对应 30 条 noqa 压制的是**已被忽略**的规则 | `ruff.toml:29,32` |
| `exclude = ["src/ml/**"]`（注释写的是 "legacy code with syntax errors"） | `ruff.toml:12` |
| **实测 33 条 noqa 为死** —— 恰好等于 `src/` 里非 `src/ml/` 的**全部** | `ruff check src/ --extend-select RUF100` |
| 另 42 条在 `src/ml/` 内，ruff **根本看不到** | 同上 |

⇒ **结论要反转**：把 noqa 当"压制静态检查"是不准确的 —— 它们几乎全是装饰品。
真正的问题是 `ignore` / `exclude` 让**规则失效**，于是真实缺陷（§3 的 F-14）无人看见。

### 8.2 爆炸半径（`--isolated` 绕过本仓配置后，`src/` 全树）

| 规则 | 命中 | 其中在 `src/ml/` |
|---|---|---|
| `F401` 未用导入 | 155 | 11 |
| `BLE001` 盲 except | 299 | 12 |
| `SLF001` 私有成员访问 | 129 | 0 |
| `F841` 未用变量 | 71 | 4 |
| `S110` try-except-pass | 60 | 0 |
| `E402` 非顶部导入 | 44 | 18 |
| **`F821` 未定义名** | **30** | **30（全在排除区）** |
| `E741` / `A002` / `F811` | 13 / 5 / 9 | 0 |

⚠️ **不能一次性开**。`F401` 那 155 条里绝大多数是包级 `__init__.py` 再导出
（`ao_shaping/__init__.py`、`optimizer/__init__.py`、`gui/slm/__init__.py` …），
必须先补 `__all__` 再开，否则一次性 155 条噪声 → 按 §8.6 分批。

### 8.3 73 条 noqa 的分类（agent 逐点核对）

| 类别 | 条数 | 处置 |
|---|---|---|
| **真缺陷** | 1 | `tools/slm/slm_zernike_sweep_probe.py:44` —— re-export 块里 `core_fraction` / `display_and_average` 确实未被使用 ⇒ **删这两个名字**。✅ 已删（commit `41cacbd2`，2026-10-23，R-63 step 1）：模块现位于 `tools/zernike/sweep_probe.py`，两个名字连同已失效的 `(capture_settled re-export)` noqa 一并移除，docstring 交叉引用改指 `bench_kernels` 规范路径 |
| **noqa 已多余**（别名其实被用了） | 0 | 2026-10-21 复核：原 3 处（`utils/wavefront/matrix_utils.py:51`、`drivers/sim/beam_backend.py:209`、`optimizer/wfless/pib.py:95`）已由 `1faaabb5` 全部删除 ⇒ 本类已清空。⚠️ 勿误删 `pib.py` 现有 3 条 `# noqa: F401`（`:23`/`:24`/`:25`）：那是**必需**的再导出豁免（该文件无 `__all__`，去掉即报 F401），且 `pib._camera_cls` 被 `tests/ao_shaping/runners/test_sensorless_sim_params.py:132-133` 断言身份（行号 2026-10-23 重钉，跟进 `2ceb288f` 顶部 `_convergence_diagnostics` import 块下移 +4，原 `:19`/`:20`/`:21`）|
| **应改为正式再导出** | 1 | `display/image_display.py:15` ⇒ 补 `__all__`，让 F401 不再是问题。⚠️ 不能改成 `import as` 冗余别名 —— `tests/ao_shaping/display/test_image_voltages_display.py` 有身份断言（`a is b`）。✅ 已补 `__all__`（commit `41cacbd2`，2026-10-23，R-63 step 1），re-export 由 blanket `# noqa: F401` 改为声明式公开面，身份断言测试 4 passed |
| **合法副作用/能力探测，保留 noqa** | 8 | `drivers/dm/__init__.py:26,27`（驱动注册；`tests/ao_shaping/runners/test_cli_contract_freeze.py` 依赖）、`drivers/dm/_registry.py:77`（惰性 sim 导入以避环）、`optimizer/wfless/slm_gs_refine.py:757`（注册 sim 相机；`test_sim_camera_registration.py` 依赖）、`gui/slm/pyarrow_probe.py:39,60`（能力探测）、`tools/slm/slm_phase_response.py:47`（Agg 先于 pyplot） |
| **全域 `E402` ignore 是错的工具** | — | ✅ **已完成（`84214f7f`，2026-10-23，R-63 step 2b）**。原估 **435 条** 偏低：按项目配置实测 **879 条**（`scripts/` 709 / `tests/` 128 / `src/` 42），且在两个被钉住的 ruff（ci.yml `0.11.10`、ruff.yml `>=0.15.12`）下逐字节相同 ⇒ 不是版本产物。已按 §8.6 第 1–2 步执行：step 2a（`8aa88fa2`）用**纯行位置移动**修掉 40 处普通错位（11 个文件的模块级语句多重集与 HEAD **AST 相同**，逐语句 `ast.dump` 比对，故代码内容/注释/相对 import 顺序均未变），step 2b 把余下 839 条按**成因**分组写进 `lint.per-file-ignores`（22 条具名豁免 + `scripts/*` + `__init__.py`），全局 `ignore` 里已无 `E402` ⇒ `src/`、`tests/` 从此真的被检查。 |
| ↳ **E402 剩余豁免的可修性（2026-10-23 探测，两个 ruff 版本结论一致）** | — | ruff 的 E402 **豁免**：直接 `sys.path.insert(...)`、`os.environ[k] = ...`、`os.environ.setdefault(...)`、**带守卫的** `if k not in sys.modules: sys.modules[k] = ...`；**不豁免**：**无守卫的** `sys.modules[k] = MagicMock()`、`warnings.filterwarnings(...)`、以及**任何中间赋值**（`ROOT = Path(...)`、`torch = pytest.importorskip(...)`）。⇒ 现有豁免多数是被**赋值**触发，而非 bootstrap 本身。两条零风险收口路径已写进 `ruff.toml` 注释备查：① 去掉绑定形式（`pytest.importorskip("torch")` 单独一行 + `import torch`）可退掉 `tests/ao_shaping/ml/*` 48 条；② 把路径内联（不留 `ROOT =`）可退掉 bootstrap 类。**未在本刀做**：要改 ~40 文件且无正确性收益。**永不该动**：`tools/kernels`-类**导入普查测试**（`tests/ao_shaping/optimizer/wfless/test_slm_model_in_loop.py` 围绕自身 import 采样 `sys.modules`，移动 import 会改变被测对象）。 |
| **matplotlib Agg 那批多半不需要** | ~18 | ruff 的 E402 **已豁免** `sys.path` / `os.environ` 修改与 `matplotlib.use()` 激活（`is_matplotlib_activation`，PR #9094 修 #9091）⇒ `src/ml/shaping/viz.py:50-55`、`src/ml/gsnet_debug/train.py:43-66` 的 noqa 即使开 E402 也很可能仍不需要，**先验证再删** |

### 8.4 `BLE001` 31 条：遵守规范 = **收窄类型**，而非删 noqa

⚠️ 先纠正两条常见误解：

* **`except Exception` 不捕获 `KeyboardInterrupt` / `SystemExit`**（二者继承 `BaseException`）
  ⇒ 遥测类 `except Exception` 对 Ctrl-C 是**安全**的，不必收窄。
* 真风险只有 **2 处 `except BaseException`**（⚠️ 2026-10-25 重测：现树 **5 处** = `src/` 3 + `scripts/` 2，
  逐处判定见 §8.9 第 14 项；本行的「2 处」是本节写作时的口径，`transfer/workflow.py` 与 `scripts/` 两处未计入），而其中
  `utils/image/hardware_utils.py:627`（原钉 `:628`，已按 grep 实测重钉）**必须保持 `BaseException`**：它跑在 daemon
  工作线程里，目的就是把工作线程的任何异常搬到调用线程重抛。收窄成 `Exception`
  会让工作线程的 `KeyboardInterrupt`/`SystemExit` 静默消失，`call_with_timeout`
  改为返回 `None`（静默行为变更）。它**不影响**主线程 Ctrl-C（`join` 可中断）。

| 处置 | 站点 | 收窄到 |
|---|---|---|
| **保持 `BaseException` + 补注释** | `utils/image/hardware_utils.py:627` | 不改，仅把线程边界语义写进注释；已机械钉住：`test_conventions.py::test_thread_boundary_exception_capture_keeps_base_exception` |
| **最高价值：收窄** | `drivers/slm/santec/driver.py:1184` | `SantecError` —— 房内既有写法：`slm_gs_refine.py:780` `except SantecError as exc: if exc.code != 1: raise` |
| 收窄 | `gui/r50/r50_channel_select.py:360` | `(ValueError, KeyError, TypeError)`（pandas `int()` / 列访问） |
| 收窄 | `tools/slm/slm_zernike_common.py:114,129,155,162,166` | `(SantecError, WfsError, AttributeError)` |
| 收窄 | `runners/slm/zernike_matrix_runner.py:652,759,765`（旧行号 733/774/840 已漂移，2026-10-24 复核） | `(WfsError, RuntimeError, ValueError)`。🔴 原建议含 `AssertionError`，现**禁止**：`92dbe716` 的守卫 `test_production_code_never_catches_assertion_error` 机器禁止 `src/`+`scripts/` 捕获 `AssertionError`（`python -O` 下 assert 消失 ⇒ 处理器变死代码；且 assert→raise 转换会让按 `AssertionError` 写的处理器静默失配）；该文件现 3 处均为裸 `except Exception`，未含 `AssertionError` ⇒ 收窄时不得引入 |
| 收窄（teardown / close） | `runners/slm/zernike_matrix_runner.py:1376`、`slm_gs_refine.py:804`、`slm_model_in_loop.py:1679` | `(DeviceError, OSError, RuntimeError)` |
| **必须保持宽** | 遥测 12 条（`ml/zernike/train_coeff.py:512,524,547,581,591`、`ml/zernike/train_amp.py:1301,1329`、`ml/shaping/train.py:680,741,767`、`ml/gsnet_debug/train.py:737,855`） | wandb 从网络/文件系统/序列化多路抛错；且 `except Exception` 不吞 Ctrl-C ⇒ 保留 noqa + 理由 |
| **必须保持宽** | `zernike_matrix_runner.py:846` | 包住**调用方传入的任意回调**，无法枚举 |
| 边缘（建议保留） | `zernike_matrix_runner.py:1448`（整条多设备流水线）、`:1323`（显示回调） | 收窄会变脆 |
| **改动前须单独复核 → 已复核，保留** | `optimizer/wfless/slm_zernike_pib.py:1355`（`BaseException`，在 `_apply_best_on_exit()` 退出清理里；原钉 `:1285`/`:1297` 两处已漂移，2026-10-25 实测**只剩 1 处**） | 已逐行读：处理器把异常**类型与消息都写进日志**，再强制 `set_grayscale(0)` 回安全平场，内层兜底 `except Exception` 记 "SLM state unknown" ⇒ 不重抛但**不静默**；宽捕获是**必需**的（收窄后 Ctrl-C 走不到安全态）。无需改动 |

⚠️ **ruff 没有「故意宽 catch」的开关**：`lint.flake8-blind-except.allow-raise`
**不存在**（官方 settings 无该节；GitHub 全仓 grep 无结果）。唯一相关选项是
`lint.logger-objects`（用 `exc_info` 记到已登记 logger 对象上不算盲吞）。
⇒ 对确实必须宽的站点只有两条路：**收窄类型**，或保留**带理由的定点** `# noqa`。

### 8.5 noqa 之外：更大的两片（R-64）

| 项 | 数量 | 说明 |
|---|---|---|
| **`src/` 无标注的 `except Exception`** | **312** | 上表 31 条带 noqa 的只占 **9%**；主流写法是**完全不标注**，而 `BLE001` 不在 `select` ⇒ 这 312 条从未被任何工具看见。热点：`gui/zernike/zernike_response_matrix_ui.py` 19、`gui/slm/multi_slm_controller.py` 19、`gui/ccd/ccd_analyzer.py` 18、`drivers/ccd/miicam/driver.py` 16 |
| **`src/` 把 `assert` 当运行期护栏** | **111**（37 文件） | `drivers/ccd/miicam/driver.py` 21、`daheng/driver.py` 18、`ffmpeg/driver.py` 8。含**设备已打开断言、曝光范围检查、安全检查** ⇒ `python -O` 下**全部消失**。对硬件安全护栏这是正确性问题，不是风格问题 |
| `# type: ignore` | `src/` 44 · `scripts/` 7 · `tests/` 53 | 仅 4 处无理由：`utils/image/target/patterns.py:233`、`tools/slm/slm_train_data_collect.py:183,848,849` |
| 其它 lint 配置 | — | `pyrightconfig.json` 存在但 `typeCheckingMode: "basic"`（宽松）；**无** mypy / flake8 / pylint / bandit；`.github/workflows` 于 2026-10-15 建立 ruff 硬门禁 + pylint 参考级，2026-10-19 曾整体移出跟踪，**2026-10-21 恢复入库**（现 `ci.yml`：ruff 硬门禁 + best-effort pytest，本地 `ruff check .` 仍须全绿）；`.pre-commit-config.yaml` 只有 ruff + ruff-format |

### 8.6 `ruff.toml` 重构顺序（**必须按序**）

> ⚠️ **顺序错了会误删**：`RUF100` 对「规则未启用」的 noqa 同样报 `non-enabled`
> **且可自动修复**。若先跑 RUF100 再开规则，`BLE001`/`SLF001`/`S301` 那 40 条
> 会被删掉，等规则真正启用时就变成裸奔。故顺序为：

1. **先改代码** —— §8.3 的 1 条真缺陷 + 3 条多余 noqa；§8.4 的收窄项
   ✅ 真缺陷与 `__all__` 项已做（`41cacbd2`）；E402 普通错位 40 处已做（`8aa88fa2`，逐文件 AST 证伪「仅移动」）。§8.4 的 `BLE001` 收窄进行中（#174：`009e621d`/`bb64c162`/`2ceb288f`）。
2. **再改配置** —— 从全局 `ignore` 移除 `F401`、`E402`；改用 `lint.per-file-ignores`
   给 `scripts/*` 加 `E402`（保住 435 条 bootstrap 约定）、给 `__init__.py` 加 `F401`
   （AGENTS.md 的再导出 shim 约定）
   ✅ 两步都已落地：`F401` 已移出（见 §8.2），`E402` 已移出（`84214f7f`，与 22 条具名豁免**同一提交**入库，故 §8.7「瞬间爆红」那条没有发生）。⚠️ 「435 条」估算作废：实测 **879 条**（scripts 709 / tests 128 / src 42），见 §8.3 同主题行。
3. **然后**清理剩余死 noqa：`ruff check src/ --extend-select RUF100 --fix`
   —— **必须 `--extend-select`**；用 `--select RUF100` 会替换整条选择链导致全量误报
   ✅ 已入库为**常驻规则**（`select` 含 `RUF100`，864 条死 noqa 已在先前一轮清除并逐文件 AST 验证）；本轮 `--extend-select RUF100` 命中 0。不得改回一次性 `--fix` 扫描。
4. **最后**处理 `exclude = ["src/ml/**"]`：先清 F-14，再逐族放开（建议先只放开 `F` 族）
   ✅ F-14 已清（2026-10-14），`exclude` 已删，并由 `test_ruff_excludes_nothing_under_src` 钉死不得回归。
5. `BLE001` / `SLF001` / `S301` **暂不整族启用** —— 等 §8.4 收窄完成，否则一次性 299+129+6 条
   🔴 仍开放，**前置 = R-64**。启用前必须先按项目配置实测真实命中数（`--extend-select BLE001` 等），**不得按估数决策**——§8.3 的 435→879 就是估数失准的实例。2026-10-23 已实测：本行原写的 "299+129+6" 低估约 5×，真实命中 **BLE001 514 / SLF001 1050 / S301 63**；且 `PLC0415` 在 `0.11.10` 是 preview（0 命中）、在 `0.15.12` 已稳定（1476 命中），所以**任何 `--extend-select` 测量都必须写明 ruff 版本**（详见 §8.13 末表）。

### 8.7 护栏：以下**不是**待办，**不得**当作「顺手清理」去做

| 项 | 为什么不能 |
|---|---|
| 不要为「让 lint 变绿」给 `src/ml/` 加 noqa | 那 30 个 F821 是真缺陷，加 noqa 等于把 `NameError` 藏得更深 |
| 不要把 `E402` 从 `ignore` 直接挪到 `select` 而不给 `scripts/*` 开 `per-file-ignores` | 会让 `scripts/*` 的 709 条 bootstrap 约定瞬间爆红（E402 实测总数 879 = scripts 709 / tests 128 / src 42，见 §8.3；原估 435 已作废，2026-10-23 随 R-63 step 2b 重钉） |
| 不要先跑 RUF100 再开规则 | 见 §8.6 第 3 步，会删掉将来必需的 noqa |
| 不要用 `# ruff: enable` 复活被 ignore 的规则 | 官方明确：range suppression **不能**启用"未被 select 的规则" |
| 不要把 `except BaseException`（`hardware_utils.py:627`，以及两处同构的线程边界捕获 `scripts/slm/diff_beam_capture_flat.py:68`、`scripts/slm/hw_gs_defocus_probe.py:1071`）改成 `Exception` | 会造成静默行为变更（工作线程的 Ctrl-C 消失，调用方把挂起当成成功），已推理见 §8.4；已机械钉住：`test_thread_boundary_exception_capture_keeps_base_exception`（见 §8.9 第 14 项） |

### 8.8 一处不一致（非缺陷，但值得知道）

`tools/slm/slm_zernike_common.py:382`（`measure_zernike`）有 `except Exception`
却**没有** noqa，而同文件 `:114-166` 五处都带。说明 noqa 是**逐点手工加的**，
不反映「哪些是故意宽的」这一信息 ⇒ 收窄时应以**代码语义**为准，不要以
「有没有 noqa」为准。

### 8.9 `assert` 当运行期护栏（111 处）→ 按层收口

> Python 文档明确：`assert` 用于**内部不变量**，**不得**用于校验外部输入；
> 且 `python -O` 会**整条移除**（连同被保护的表达式）。因此下列检查在 `-O` 下全部消失。

**关键事实：正确的范式本仓库已经有了，不需要引入新概念。**

| | 现状 | 判定 |
|---|---|---|
| **SLM（范式）** | `Santec._ensure_open()`（`santec/driver.py:1807`，**18 处**调用）+ `ZernikeSLM._ensure_open()`（`zernike_slm.py:100`，**5 处**）—— 单一入口、`raise RuntimeError` | ✅ 照抄这个 |
| CCD（原反范式） | ✅ **已收口**：`assert self.cam, "camera not initialized"` 曾在 `miicam/driver.py` 重复 **18 处**、`daheng/driver.py` **12 处**（+1 处 `camera not found`）、`ffmpeg/driver.py` **6 处**（两种写法：`_cap.isOpened()` 与 `_folder_path`）| 现全部走 `BaseCamera._ensure_open()`（`base.py:77`）/ `_device_open()`，抛 `CameraError`；ffmpeg 覆写 `_device_open()`，录制源前置另设 `_ensure_recording()` |

| 簇 | 条数 | 现状 | 应收口到 |
|---|---|---|---|
| CCD 设备已打开 | ~~**37**~~ 实测 **37**（`miicam` 18 + `daheng` 12 + **1** `camera not found` + `ffmpeg` 3 `_cap` + 3 `_folder_path`；原表 daheng 记 13 是把 `camera not found` 计入） | ✅ **已收口**（`BaseCamera._ensure_open()` / `_device_open()` 抛 `CameraError`；`ffmpeg` 的 `_folder_path` 是**录制源前置**而非设备状态 ⇒ 单独 `_ensure_recording()`，不与设备就绪混为一谈；`thorlab` 式参数断言 2 处改 `ValueError`） | 照 SLM 范式，`BaseCamera._ensure_open()` 抛 `CameraError`（调用方已捕获它）。守卫落在 **CI 默认发现集内** 的 `test_conventions.py::test_ccd_drivers_never_assert_on_device_state`（AST 扫 `drivers/ccd/**`，任何 `assert self.cam/_cap/_folder_path` 复活即失败）—— 落点原因：当时 `addopts --ignore` 把 `tests/ao_shaping/drivers/**` 排除在发现集外，守卫若放那里等于不设（该 drivers 排除已于 2026-10-24 撤销，守卫保留在现位置） |
| WFS 曝光**安全上限** | ~~5~~ **6**（实测：`scripts/zernike_bench/generate_zernike_wfs_report.py:381` 也是同一写法，原行数漏计） | ✅ **机制已修 + 已单源 + 已改名**（`e6f80254`，2026-10-23）。原状：`MAX_EXPOSURE_MS = 7.0` 在 `tools/zernike/common.py:40`、`tools/wfs/wfs_probe.py:44`、`tools/slm/calibration/calibration.py:127` **声明 3 处**（原表把路径写作 `slm_wfs_probe.py` / `slm_zernike_common.py`，2026-10-12 重组后已移动），6 个脚本各查**两遍**：CLI 层 `if…raise`（策略，保留）+ 内层 `assert`（机制，`-O` 下整条消失）。被 `assert` 守护的是 `exp = float(wfs.exposure_time)` —— **打开后从硬件读回的实际曝光**（上一次会话可能遗留更大值），属外部状态 ⇒ 正是 Python 文档所说"不得用 assert 校验外部输入"。现统一到 `tools/zernike/common.py` 的 `WFS_EXPOSURE_SAFETY_MAX_MS` + `require_wfs_exposure_within_safety_ceiling()`（显式 `ValueError`；`wfs_probe.py:159` 那个吞掉它的 `except AssertionError` 已确认由 `:161` 的 `except Exception` 等价接管，控制流不变），并加 AST 守卫防复发 | ⚠️ **原计划"移进 `ThorlabWFS.exposure_time` setter"这一步与本表 §8.10 的红线冲突，建议撤销**：setter（`thorlab_wfs.py:1774-1776`）**已经**用显式 raise 校验**硬件边界** `EXP_TIME_LOW/HIGH = 0.002/86`（与设备无关的事实）；而 `7.0` 是**随光路/激光功率变化的策略上限**，写进驱动就是把策略伪装成机制（与 R-57 对 `--max_peak` 的结论同形）。要做的是把它变成**可配置**（类比 `AO_DAHENG_MAX_EXPOSURE_MS` 的按会话设上限），而不是搬进驱动 |
| 曝光/范围校验（**层已对，机制错**）| ~~9~~ **0 剩余**（2026-10-23 实测） | ✅ **已完成**（#174：`009e621d`/`bb64c162`/`2ceb288f`）。实测 `drivers/ccd/common.py`、`drivers/dm/nlight/driver.py`、`drivers/dm/nlight/sdk.py` 的 `assert` 计数均为 **0**，`thorlab_wfs.py:1774` 注释即写明 "Explicit raise, not `assert`"。⇒ 本行原列的 `common.py:66,81,86,95` / `thorlab_wfs.py:1811` / `nlight/driver.py:147,148` 全部过时，勿重复处理。**曾列出的 WFS `assert`（另一类，非本行）已全部收口**（`e4868b54`）：`thorlab_wfs.py:446`（`device_in_use`）、`:1074`/`:1184`（高速模式前置）、`zernike_order <= 10` 参数边界 ⇒ 全部改显式 `raise WfsError`（消息逐字保留），`ThorlabWFS` 类内 `assert` 实测归 **0**；`:468` 已是 `or` 表达式非断言 | 就地改 `raise`（原计划）——**该子项已结** |
| NLight 原生连接 | ~~1~~ **0 剩余（本行过时）** | ✅ **早已修**（`1faaabb5` "make runtime guards survive python -O"）。实测 `dm/nlight/sdk.py:62-66` 现在就是显式 `if not self._dll.GetConnection2(): raise DeviceNotFoundError("device connection error.")`，注释也写明 "Explicit raise, not `assert`"。⇒ 本行原写的 `assert self._dll.GetConnection2()` 已不存在，勿重复处理 | 已结 |
| TM 串口存在性 | **1**（`tm/serial_port_fsm.py:36` `assert available_port, "No available serial port found."`） | ✅ **已修**（见`96447536`）。这是 `drivers/` 里**最后一处**设备状态 `assert`：`list_port()` 返回空是**外部环境**条件（没插串口适配器），`-O` 剥掉断言后下一行 `available_port[0]` 变成莫名的 `IndexError`。现改 `raise DeviceNotFoundError`（复用 `device_base.py:558`，与 NLight 同一异常词汇） | 外部条件在驱动入口显式 raise；已纳入下面的全树守卫 |
| 参数校验（调用方输入） | ~~10~~ 实测 **7 在 `drivers/ccd`** + 1 内部 | ✅ **已修**（见`96447536`）：`n_sample > 0`（`daheng:528`、`ffmpeg:245`、`ffmpeg:644`、`miicam:822`）、`tolerance > 0`（`daheng:585`、`miicam:573`）、offset ≥0（`miicam:770`）全部改**显式 `raise ValueError`**（消息逐字保留）。⚠️ 原行两处不准：(a) 条数不是 10；(b) 它暗示其余在相机驱动 —— 实测 `display/frames.py:27`、`display/windows.py:210` 是**重复注册检查**、`utils/io/file.py:479/592/946` 是 Recorder **内部不变式**，都不是调用方数值，**不在本簇**。另 `daheng` 的 `width_val/height_val is not None` 是从 SDK **读回**的内部状态 ⇒ 保留 `assert`（进 allowlist） | 相机侧入口统一 `raise ValueError`。测试契约同步：`test_frame_averaging.py::test_nonpositive_sample_count_rejected` 原用 `pytest.raises(AssertionError, match="Sample count")` 把异常类型**锁成契约**，已改为 `pytest.raises(ValueError, ...)`（先改契约再改实现，与 §8.9 前置 1 同法） |
| 全树守卫（新增） | — | ✅ `test_conventions.py::test_drivers_only_assert_internal_invariants`：`drivers/**` 里的 `assert` 集合必须**恰好等于** 5 条 allowlist（`ccd/daheng width_val/height_val`、`dm/micro` 2 条、`sim/wfs` 1 条、`slm/santec max_grayscale`）。比较是**双向**的：新增 `assert` 失败，allowlist 里残留已消失的条目也失败（陈旧 allowlist = 守卫悄悄失效，正是本仓账本反复踩的坑）。键用 `路径 + ast.unparse(表达式)` 而非行号 ⇒ 上方插代码不会把合法断言挤出去，新断言也不能靠行号漂移蒙混 | 先补机制再删调用侧；`drivers/` 的 `assert` 从此只允许内部不变量 |
| WFS 设备就绪（**机制原本缺失**） | 实测 **0 → 已补**（`e4868b54`） | ✅ **已收口**：`ThorlabWFS` 此前**完全没有**设备就绪校验 —— `grep '_ensure_open\|_require_sdk' src/ao_shaping/drivers/wfs/` 命中 **0**。现已新增 `ThorlabWFS._ensure_open()`，**23** 个 DLL 入口的首行调用；类内 `assert` 归 0。⚠️ 该守卫只校验**句柄非空**，**不能**用 `is_connected()`（那要求 `READY`）：`open()` 在 `:520` 置 `READY` **之前**就要调 `select_mla`/`set_ref_plane`/`optimize_exposure_time_and_gain`/`exposure_time`/`high_speed`/`pupil`（`:508-518`，仍在 `CONNECTING` 态），READY 判据会让 `open()` 拒绝自己 | 关键实测：空句柄**不会**可靠报错 —— `select_mla` 丢弃全部返回码、`take_image` 自动曝光分支忽略错误码并仍置 `_image_captured = True` ⇒ 未打开的调用**返回垃圾数值而非异常**。守卫必须是**方法**而非内联 `if self._instrument_handle.value == 0`：`test_wfs_new.py:486` 用 `ThorlabWFS.set_ref_plane.__get__(wfs)` 把真方法绑到 `MagicMock`（其 `_instrument_handle = 1` 是普通 int），内联 `.value` 会 `AttributeError`，而 `self._ensure_open()` 被 mock 掉。`SimulatedWFS(BaseWFS)` 且全仓无类继承 `ThorlabWFS` ⇒ 守卫到不了仿真/假对象。守卫 `test_conventions.py::test_wfs_driver_guards_every_dll_entry_point`（AST：任何触碰 `_instrument_handle` 的公开方法首行必须是守卫；并禁 `ThorlabWFS` 内出现 `assert`）|
| cartographer 设备前置 | ~~13~~ 实测包内 `assert` 共 **14**，本簇 **10** | ✅ **已收口**（`e4868b54` 补机制 + `3ba720ec` 删调用侧）。删 10 处：设备前置 **9**（`slm.is_open` 3 + `wfs.is_connected()` 6）+ `assert self.slm.Panel_Res is not None` **1**（`Panel_Res = PANEL_RES = (1920, 1200)` 是**类常量**，`santec/driver.py:398`/`slm200_constants.py:23` ⇒ 对真实驱动永不为假，只有假对象能触发）。保留 **4** 处**内部不变式**（`self.lut` ×2、`self._reference_saved`、`self._last_measurement`）—— 正是 §8.10"保留 `assert`"那一行。实测 `_reference_saved = True` 只在 `:218`/`:323` 两处写且各自行前都写 `_last_measurement` ⇒ `:254` 与 `:249` 逻辑冗余，但保留作类型收窄 | 先补机制再删调用侧（见上一行：原判定的"驱动的 `_ensure_open` 已覆盖"对 `ThorlabWFS` **不成立**）。守卫 `test_conventions.py::test_cartographer_never_asserts_on_device_state`（AST 扫 `tools/slm/cartographer/**`）；**非空验证**：同一规则扫改前树命中 **10**、改后 **0** |
| optimizer 设备非空 | ~~7~~ 实测此形态仅 **1**（`slm_zernike_pib.py:814`） | ✅ **已删**。⚠️ 原表 "7" 是**审计当时 `optimizer/` 目录的 `assert` 总数**（`0026f583` 处确为 7 条），其中本形态只有 **2**：`slm_zernike_pib.py:721` 与分叉文件 `slm_zernike_shaping.py:675`（后者已由 `ee96acfe` X-4/#74 删除）。其余 5 条（`best_q`、`_pattern_helper`、`zernike_radius`、`_rot_clip`、DM 矩阵形状）不是设备句柄 ⇒ 归"编程不变式 保留"行 | 契约保证：`create_camera`（`drivers/ccd/common.py:274 return cls(**kwargs)`，否则 `:261` `ValueError` / `:280` `ImportError`）与 `Santec.from_params`（`santec/driver.py:410 return cls(**kwargs)`）**永不返回 None**；全部 `__enter__` 返回 `self`（`ccd/base.py:75`、`daheng:154`、`miicam:208`、`santec:504`、`SimSLMPib:967`）；被断言的名字来自 `with cam_ctx as cam, slm_ctx as slm` 的 `as` 绑定，**不是参数** —— 调用方无法注入 None（API 只收 dataclass）。同目录 `slm_square_shaping.py:1331-1341` 早就是同一写法且**无**此断言。9 个 `test_slm_zernike_pib*` 文件实测 **112 passed** |
| 调用方窄处理器与异常类型脱钩（**由本账本的转换动作引出**） | 实测 `src/` + `scripts/` 共 **9** 处 `except AssertionError`：其中 **3 处是元组写法** `except (RuntimeError, ValueError, AssertionError)`，用 `grep 'except AssertionError'` **完全看不见** | ✅ **已清 8 处，1 处具名豁免**（见`92dbe716`）。两处是**我自己前面提交造成的真回归**：① `96447536` 把 `miicam:770` 的 offset 断言改 `ValueError` 后，`calibration.py::apply_stored_window` 的窄处理器不再命中 ⇒ 越界存储窗口**不再按 docstring 承诺降级为全幅**，而是把异常穿出标定流程；② `e4868b54` 把高速模式断言改 `raise WfsError` 后，`wfs_probe.py:126` 的处理器失配 ⇒ "需要关闭 high speed mode" 的可执行提示消失、退化成 `except Exception` 的整段 traceback。另 1 处是既有缺陷：`scripts/tuning_devices/dm_unit_compute.py` 的生成器主循环用 `assert running` + `except AssertionError: pass` 作为**退出机制**，`python -O` 下 `assert` 消失 ⇒ 生成器永不终止、`finally` 里的 `wfs.close()` 永不执行（现改显式 `if not running: return`）。豁免：`src/ml/hwdataset/dataset.py:735` 捕获的是 **torch 自己**CUDA 探测抛的 `AssertionError`（契约是降级为 `pin_memory=False`），与本仓 `assert` 无关 | 生产代码一律**捕获被调用方真正抛出的类型**。守卫 `test_conventions.py::test_production_code_never_catches_assertion_error`（AST 走 `ExceptHandler`，**元组里每个名字都看**；`src/` + `scripts/` 全域；allowlist 用 `路径 + 排序后的异常名元组` 作键 ⇒ 行号漂移挤不掉、新增躲不过、改动了处理器则条目变陈旧同样失败；`tests/` 有意豁免 —— 那里 `pytest.raises(AssertionError)` 是把**内部不变量**钉成契约，属"保留 assert"行）。回归测试**先红后绿**：`test_slm_calibration_windowed.py::test_apply_stored_window_out_of_range_degrades` 在修复前实测 `E ValueError: Window center position: (100, 100) must be within image…` 穿出，修复后 13 passed |
| 编程不变式 | ~40 | `assert img.ndim == 2`、`assert best_q is not None`、`assert G.ndim >= 2` 等 | **保留 `assert`**（内部不变量，外部输入到不了）。⚠️ 本行一度被我的编辑误删（2026-10-24，替换而非插入）—— 它是全表唯一写明"哪些**不该**改"的行，删掉等于把 campaign 的边界条件丢掉，已原文恢复。`drivers/` 内的不变式 `assert` 现由 allowlist **精确钉死为 5 条**（`test_drivers_only_assert_internal_invariants` 双向比较），不再是估算数 |

🔴 **同一坑的第二次，且这次是机器拦住的**：我在动手前用 `grep -rn 'except AssertionError' src/ scripts/` 得到 **0 命中**，据此以为没有窄处理器；AST 守卫一跑就报出 3 条元组写法。⇒ 判"不存在某类引用"**不得依赖单次 grep**：文本匹配看不见 `except (A, B)` 这种同一语义的不同写法。本会话已两次踩中（前一次是漏看 `test_frame_averaging.py:69` 的 `pytest.raises(AssertionError)`），区别是这次把规则写成了测试而非靠人复查。

> 🔴 **`--max_peak` 不要动。** 它在 **optimizer**（`slm_zernike_pib.py:1585-1605`），不在驱动（行号 2026-10-20 复核，原 `:1456-1474`；`1ce4717a` 常量收敛后整体 −1）；
> 且其 CLI 帮助与 archive R-57 **明文声明它不是安全联锁**（"NOT a safety interlock"，
> 只拒绝提交下一轮更新，光已出）。把它搬进驱动等于把**策略**伪装成**机制** ——
> 与 TODO 里 R-57 的结论直接冲突。

**三个必须先解决的前置**

1. ✅ **本条原判定有误，已更正（2026-10-24 实测）**：原文写 `spots_calc.py:77,119,133`
   "不许批量转换"。实测该文件**只有 3 条 `assert`，位于 61 / 103 / 117**（原行号过时），
   且三条都是 `assert img.ndim == 2` —— 这正是本表"编程不变式 **保留 `assert`**"那一行
   自己举的例子（原文即写作 `assert img.ndim == 2`）。⇒ **本项不是待办**：实现与
   `test_spots_calc.py::test_crop_3d_image_raises_assertion`（现 `:102-107`，对 `crop`
   与 `crop_numba` 各断一次）互相一致，无需改动。
   🔴 但它暗示的风险"**异常类型被当成契约**"是真的，且方向相反 —— 危险不在测试里
   `pytest.raises(AssertionError)`，而在**生产代码的 `except AssertionError`**：见下新增行。
2. ✅ **`MAX_EXPOSURE_MS` 命名冲突（同名差 1400 倍）已解除**（`e6f80254`）：工具层 `7.0`（WFS 激光安全上限）改名 `WFS_EXPOSURE_SAFETY_MAX_MS` 并收敛到**一处声明**；`drivers/ccd/miicam/driver.py:133` `MAX_EXPOSURE_MS = 10000.0`（相机自身硬件上限）**保持原名原值不动**（2026-10-25 重钉，`7532c8cb` 在 `MIN_EXPOSURE_MS` 上方插入一行，原 `:132`）（驱动层不在本次改动范围，且语义不同）。实测 `src/` + `scripts/` 下除 miicam 外 `MAX_EXPOSURE_MS` 命中数 **0**（改前 30），并由新测试 `tests/ao_shaping/tools/wfs/test_exposure_safety_ceiling.py` 钉住"两者是不同量"。
3. ✅ **`DeviceParameter` 登记表：结论是"不要接线"，改修「自相矛盾的声明」**（`950e2e9d`）：
   实测比原判断更强 —— `set_parameter_value` / `get_parameter_value` 在 `gui/`、
   `runners/`、`tools/` 下调用数 **0**；唯一走校验的写缝 `sync_from_twin`
   （`device_base.py:519`）的整条消费链（`get_all_twin_states` /
   `sync_from_twin_states` / `get_twin_snapshot`）在 `src`+`tests`+`scripts` 全树
   调用数也是 **0** ⇒ 该写缝**在生产路径上不可达**，不只是"硬件写路径不生效"。
   ⇒ **明确否决"把 `validate()` 接到写路径"**（原方案的暗示方向）：`validate()` 越界只
   `return False` + 打日志（`device_base.py:77-81`），而驱动一律硬 `raise` ——
   接线等于把强校验降级成静默失败；而且它会拒绝驱动自己文档化的合法哨兵
   `exposure_time_ms = 0.0`（自动曝光），直接把 `open()` 的正常路径判成越界。
   真正修的是**重复声明漂移**：全扫 `src/` 下 **62** 处 `register_parameter`，
   **1 处自相矛盾** —— `ThorlabWFS` 注册 `default_value=0.0` 却写
   `min_value=EXP_TIME_LOW`(0.002)，自己的默认值落在自己声明的下限之外；
   驱动 `open()` 的实际接受域是 `{0.0} ∪ [EXP_TIME_LOW, EXP_TIME_HIGH]`（`:467-474`
   把非零越界值重置回 0.0）⇒ 注册下限改 `0.0` 使声明与哨兵一致，硬件下限仍**唯一**来自
   `EXP_TIME_LOW`。`BaseWFS` 一侧原本就诚实（`min_value=0.0` + "0 selects auto"）。
   同簇第二处：`gui/dm/micro_dm_ui.py:40-41` 把驱动常量 `-20.0/120.0` 重打一遍（2026-10-24 重钉至实际 `:40-41`，`987ed669` 顶部 import 块 +2 行），
   并且**已经漂了** —— 正弦振幅控件 `max_value=140.0` 超硬件上限 20 V，而驱动静默
   截断所以无人报错；现取 `SINE_AMP_MAX = (VOLTAGE_MAX - VOLTAGE_MIN)/2`：
   `offset±amp` 须同时落界内 ⇒ 两式相加得 `2·amp ≤ 峰峰值`，故 70.0 是**精确**上限，
   更大的振幅只能命中截断警告。写法与 `gui/r50/r50_channel_select.py:41-42`
   （`HW_VOLTAGE_MIN: float = VOLTAGE_MIN`，经 PEP-562 惰性再导出）一致。
   两条守卫：`test_registered_parameter_defaults_satisfy_declared_bounds`（**无**
   allowlist —— 默认值落在自己值域外没有合法情形）、
   `test_gui_never_redeclares_driver_voltage_bounds`。⚠️ 后者首版**漏报** `-20.0`
   （负字面量在 AST 里是 `UnaryOp(USub, Constant)` 而非 `Constant`），靠"把守卫跑在
   改前树上"的非空验证抓到 ⇒ 加 `_numeric_literal`；与"grep 漏 `except (…, AssertionError)`
   元组形式"是同一类只看一种拼写的盲区，守卫的**非空性**必须实测而不是假定。
4. ✅ **`gui/` 后台线程触碰 `st.session_state` 已收口**（`987ed669`）：实测**不是单点而是 5 处** ——
   `gui/dm/micro_dm_ui.py` 的 `_matrix_continuous_loop` / `_single_fixed_loop` /
   `_single_periodic_loop`，`gui/zernike/zernike_response_matrix_ui.py::_run_calibration_thread`，
   `gui/slm/slm_calibration_ui.py::_calibration_worker`。最重的三个 dm 循环把 `session_state`
   的 running 标志**当循环条件每 tick 轮询**、并每 tick 往历史 deque 追加（竞态窗口最大）；
   而 `disconnect_dm` **根本没停**矩阵线程 ⇒ 它继续驱动已关闭的设备，直到异常把它绊倒。
   违规还是**间接**的：`_matrix_continuous_loop` 自己不写 `session_state`，它经
   `set_tab2_feedback` 写；两个单单元循环经 `_single_send_v` 写 ⇒ **只扫线程体自身属性
   访问的守卫会把这三处全判干净**。改法与 `gui/r50/r50_voltage_send.py` 的
   `run_loop/start_loop/stop_loop` **同构但不跨设备域 import**：`_start_bg_loop` 在主线程建
   `Event`+`Queue` 并把句柄存进 `session_state`；线程体只收 ctrl / 参数快照 / `Event` / `Queue`；
   停止用 `Event` 而非被轮询的布尔标志；线程在 `finally` 里 set 自己的 Event 表示自退，
   **运行标志只由主线程清**；样本与错误走队列，由 fragment 在 rerun 时排空（tab1 的
   `run_every=0.15` 就是排空点）。用户可见行为不变：Streamlit 本来就**不会**因线程写
   `session_state` 而 rerun，旧代码的反馈同样只在下次 rerun 才出现。
   顺带堵掉两处**新模型下才会显形**的生命周期坑：设备为 `None` 时原代码直接 `return`，
   而运行标志改由主线程持有后，早退不 set Event 会把「运行中」指示灯**永久卡住**
   ⇒ 改为回传错误并在 `finally` set Event。
   守卫 `test_gui_thread_bodies_never_reach_session_state`：`Thread(target=…)` 的体内
   **沿同模块调用图做到不动点**（否则间接违规漏检），allowlist **两向**。剩余 2 处
   （zernike/slm 只在收尾与出错时触碰，窗口小得多）当时作为**已登记欠债**进 allowlist，
   改好却不删条目同样判失败。✅ **这两处现已收口（`07f25d4d`）**：`_GUI_THREAD_SS_ALLOWLIST`
   已清空 ⇒ 该守卫从此**零例外**；同时补上这两个 worker 原本**完全没有**的覆盖
   （`test_zernike_calibration_thread_boundary.py` 5 项 + `test_slm_calibration_thread_boundary.py`
   6 项，假设备、无硬件），其中 `slm_cal_camera` 是**线程体内读 session_state 的设备句柄**，
   一并改为 spawn 期传入。⚠️ 清空时若逐字删掉两条会留下 `= {}` —— 那是**空 dict 不是空 set**，
   守卫里 `allowlist - found` 会 `TypeError`；已写成 `set()`。
   非空性实测：把同一守卫跑在改前的 `950e2e9d` 上 ⇒
   `new` 恰为那 3 个 dm 循环且 `stale=[]`（顺带证明 2 条 allowlist 当时确实成立，
   不是凭空写的）；把**清空后**的守卫跑在改前的 `408593a2` 上 ⇒ `new` 恰为 zernike/slm
   这 2 处且 `stale=[]`。新增 `tests/ao_shaping/gui/dm/test_micro_dm_threads.py`（13 项，
   纯逻辑无硬件）：把 `st.session_state` 换成「一碰就炸」的哨兵后跑遍三条循环；
   其中 `test_bomb_actually_trips_on_session_state_access` 钉**哨兵本身有效** ——
   否则「从不触碰」只是一句空话。⚠️ 该文件此前**没有任何测试 import 过 `micro_dm_ui`**。

5. ✅ **BLE001「514 处宽异常」不是清扫目标 —— 按行为分簇后只有极少数是真缺陷**（2026-10-24）：
   复测 **522** 条（`gui` 157（silent 27）/ `tools` 75（15）/ `drivers` 61（19）/ `tests/drivers` 36（10）/
   `runners` 20（2）/ `optimizer` 15（3）/ `utils` 9（0）/ `scripts` 零散）。按 handler 行为：
   **272 记日志后吞 / 152 其它 / 95 纯静默（`pass`、`continue` 且无 raise）/ 3 重抛**
   （`ruff --select BLE001 --output-format=json` 定位 + AST 复核 handler 体）。
   ⇒ 判据不是条数，而是「**这个吞掉是否让一个错数字或一次丢掉的测量不可归因**」。
   `drivers` 簇 19 处 silent **逐处读上下文**后判定 **17 处是正确的**：miicam 恢复流程里 `Stop()`/`Close()`/`FLUSH`
   的尽力清理（`:460`、`:532` 重试到上限后**照旧 raise**）、ffmpeg 设备枚举把失败当**控制流**（该索引无设备）、
   两处 socket/writer close。**把它们改成 raise 会破坏正在工作的恢复逻辑** ⇒ 明确不动
   （§8.10「别把策略伪装成机制」的**反向**情形：这里机制本就对，缺的是归因）。
   真缺陷 2 处：`drivers/ccd/miicam/driver.py:887/905` 在 **SDK 帧回调** `_frame_callback` 内 `except Exception: pass`
   ⇒ **丢帧零痕迹**，下游唯一症状是 `get_callback_frame` 超时返回 `None` 且无从归因（与 §8.9 反复出现的
   「指标静默读出 0/空」同类）。修法：**不抛**（在厂商 SDK 线程上解栈会毁掉回调线程），改 `_note_frame_loss()`
   —— **计数 + 限流日志**（首条与其后每 100 条 WARNING，其余 DEBUG；30 fps 不限流即 1800 条/分钟），
   计数器**每次进入回调会话归零**（否则跨会话累计使限流误判、丢帧数不可信）。
   新 `tests/ao_shaping/drivers/ccd/test_miicam_frame_callback.py`（6 项，无硬件）：SDK 缺席时模块 `miicam = None`，
   而 `_frame_callback` 在 try **之前**就读 `miicam.MIICAM_EVENT_IMAGE` ⇒ 必须桩模块属性。钉住：不抛、计数==1、
   首条 WARNING 含异常原因与 `nEvent=`、**两个分支都插桩**（只修一处会留半个洞）、100 次失败 ⇒ 计数 100 而
   WARNING 仅 2 条（钉限流）、**成功帧不计数**（否则计数器虚高会说谎）、进入会话归零。
   非空性实测：同一测试跑在改前树 ⇒ `AttributeError: no attribute '_callback_error_count'`。
   回归：`drivers/ccd` + `test_conventions` **241 passed / 26 skipped / 0 failed**。
   ⚠️ **未加新 AST 守卫**：行为测试比静态守卫更硬（守卫只看拼写，看不到「是否计数 / 是否限流」）；
   本轮已新增 8 个守卫，不再膨胀。分簇逐片见 §8.9 第 5-7 项。

6. ✅ **BLE001 `gui` 片（原列 27 处 silent）逐处读上下文 ⇒ 26 处正确「不动」；删 1 处死代码；并暴露本判据的盲区**（2026-10-24）：
   26 处的正确性理由分五类，与 `drivers` 簇同源：**清理陈旧的尽力 close**（`multi_slm_controller.py:639`
   自带注释 "Ignore errors when cleaning up stale state"、`slm_calibration_ui.py:133/179` 重连前先关旧设备、
   `r50_single.py:110`）、**重试循环**（`r50_voltage_send.py:111` `_send_with_recovery`：吞单次失败后重试，
   整体成败由返回值交给调用方）、**上报动作本身的失败**（`ccd_analyzer.py:292/314` 是包住
   `raw_frame_queue.put(("error", …))` 的**内层**护栏 —— 外层已把错误送进队列；`r50_voltage_send.py:395/402/420`
   同理包住 `feedback_q.put`）、**探测降级**（`bench_settings.py:1390/1399` `detect_camera_pixel_um` 回落
   `KNOWN_CAMERA_PIXEL_UM` 型号表，外层打 debug 并显式 `return None`，注释已写明"探测失败不是错误"）、
   **清理进度文件**（`zernike_response_matrix_ui.py:833`）。⇒ **一处都没改成 raise**。
   唯一真缺陷不是"吞掉测量"而是**死代码**：`gui/zernike/zernike_debug_viewer.py:392-404` 在
   `pinv_matrix = np.linalg.pinv(...)` 之后算 `np.linalg.matrix_rank(...)` 判分支、分支里算
   `‖A‖₂ · ‖A⁺‖₂`，**结果从不赋值**，`else` 也是 `pass` ⇒ 两个分支都不决定任何事。
   代价是实的：`matrix_rank` 与两个 `ord=2` 的 `np.linalg.norm` **各自要做一次 SVD**，所以每次从调试数据
   算响应矩阵都**白做两次 O(n³) SVD** 再丢掉。已删（连同其误导性注释）。
   🔴 **我最初的定性是错的，被实测否证**：先看 `:801-803` 确实显示 `result.condition_number`，
   于是判"缺赋值 ⇒ 一个 UI 诊断量永远显示不出来"。实测两头都不成立 —— `:435` 已把 `pinv_matrix=pinv_matrix`
   传进 result，而 `condition_number` 是 `ZernikeResponseMatrixResult` 上的 **property**，自己算
   `np.linalg.cond(self.matrix)`（`optimizer/wf/zernike_response_matrix.py:111`），根本不需要那个局部表达式。
   ⇒ 规则：**判"缺赋值"之前必须先查该值的消费点是不是已经由别处（尤其 property/派生字段）承担**；
   本 campaign 这是第二次"初判被实测推翻"（前一次见 §8.9 前置 1 的 `spots_calc`）。
   ⚠️ **判据盲区（比删掉这处更值钱）**：本片的分簇只看 **handler 行为**（`pass`/`continue` 且无 raise），
   于是 AST 结构复核在同一函数里另抓到 **2 处被丢弃的三元表达式**（`:382-384`
   `float(np.mean(variance_matrix)) if … else 0.0` ×2，注释写着 "Compute derived properties"）——
   它们没有任何 `except`，**永远不会进 BLE001 计数**。同样确认为冗余：`mean_variance`/`max_variance`
   也是 result 上的 property，UI 直接读 `result.mean_variance`（`:786/:826`），已一并删。
   ⇒ 对 UI/指标类文件，除 handler 之外还要过一遍「**表达式语句的结果被丢弃**」（`ast.Expr` 且非 `Call`，
   排除 docstring）—— 本轮用 `/tmp/struct.py` 手工做，未固化为守卫（只 2 例，先记账不膨胀）。
   验证：该文件**测试覆盖为零**（`git grep -ln zernike_debug_viewer -- tests` 空）⇒ 不能只靠回归；
   改后在 CI 镜像里 import 并用 `inspect.getsource` 断言 `pinv` 仍在、被弃的 SVD 与两条三元已消失。
   回归 `optimizer/wf` + `gui` **566 passed / 23 skipped / 0 failed**（`condition_number`/`variance`
   property 的测试都在 `optimizer/wf`，证明显示路径未被我碰坏）。
   计数变化：该文件 BLE001 **4 → 3**，`gui` silent **27 → 26**。
   下一片：`tools` 15（**已做**，见下第 7 项）→ `tests/ao_shaping/drivers` 10（未做）。

7. ✅ **BLE001 `tools` 片（15 处 silent）⇒ 14 处正确不动、1 处真缺陷；并挖出一个更严重的系统性缺陷：loguru 的 `%`-参数被静默丢弃**（2026-10-24）：
   14 处分五类且**全部不动**：`main()` 收尾的尽力 close（`panel/diagnose.py:403/408`、`panel/phase_response.py:585/590`、
   `wfs/wfs_reference.py:309`、`wfs/zernike_correction.py:713`、`wfs/zernike_response.py:554`、
   `calibration/train_data_collect.py:970/975/980`（2026-10-25 二次重钉：`9bfc7f18` 在 `on_record` 处加计数块使整体下移 **+14** 行 ⇒ grep 实测 `:970/975/980`；上一次重钉按插入行数推成 `:971/976/981`，**差一** ⇒ 规则：`file:line` 一律 **grep 实测**，不得由插入行数推算））、重连前先关旧设备（`cartographer/slm_cartographer_ui.py:99/145`）、
   **可选依赖探测**（`train_data_collect.py:187`，下一行就是 `return _RECOGNISED_FAMILIES` 回落）、
   **有注释的基底外 Noll 索引跳过**（`panel/focus_preflight.py:553`；`gen.fit` 只返回自身基底的项 ⇒ 该跳过实际不可达）。
   真缺陷 1 处：`train_data_collect.py:683-687` 用 `except Exception: pass` 包住 `on_record(dict(record))`。
   判据成立的**关键实测**是落盘路径：`history` 只由 `on_record` 喂（`:926`），而 `_flush(history)` 是**唯一**写 pkl 的路
   （`batch.history = list(rows)` → `save_recorder_debug_artifacts`），循环里那份 `recorder.append(rec)` **从不进写入器**
   ⇒ 吞掉一次回调失败 = **一条已采集记录从语料中消失**，还顺带挪动 `len(history) >= flush_every` 的分批。
   修法**不改策略**（观测者失败不该毁掉整轮硬件采集），只补**归因 + 计数**：首条与每 100 条 WARNING（带 `p.index`），
   生成器收尾再报总数 —— 与 §8.9 第 5 项 miicam 丢帧同一处理（不抛、计数、限流）。
   🔴 **我自己的第一版补丁犯了同一类错，是运行时日志抓住的**：新写的
   `logger.warning("…(第 %d 条, 累计 %d 条): %s…", idx, n, e)` 在 loguru 下**参数全被丢弃**
   （loguru 用 `str.format` 插值，不是 stdlib 的 `%`），等于"修不可归因"却产出一条**没有数字的**归因日志。
   证据是测试捕获的既有日志：`零阶定位（冻结）：(%.1f, %.1f)`。
   系统性测量（AST 逐调用，写明 ruff 无关）：**既有 19 处** `logger.*(含 %-转换符的消息, *args)` 分布在 3 个 loguru 模块
   （`train_data_collect` 16 / `gui/slm/bench_settings` 2 / `utils/image/target/objective` 1），加我新写的 2 处
   ⇒ 本次共改 **21 处**，一律转 `{}` 槽位且**保留精度**（`%.1f` → `{:.1f}`），并审计**槽位数 == 参数位数：0 处失配**。
   ⚠️ **不得一刀切**：`gui/utils/phase_overlay.py` 用的是 **stdlib `logging`**（`:56`），那里 `%` 才是正确写法
   ⇒ 守卫按「该文件是否 `from loguru import logger`」过滤，**不用 allowlist**；
   `tools/ccd/exposure_check.py:148` 是 `{:.1%}` 配字面 `%`（本已正确），检测器以「消息含 `{` 即跳过」绕开。
   守卫 `test_conventions.py::test_loguru_calls_never_pass_percent_style_arguments`（AST：`logger.<level>` + 首参为
   含 %-转换符的字符串常量 + **另带参数** + 消息不含 `{`；`logger.log` 的消息在 `args[1]` 而非 `args[0]`）。
   **非空性实测**：同一守卫跑在改前树 ⇒ 报 **19 处**并失败；改后 ⇒ 32 passed / 1 skipped。
   ⚠️ **上一行是守卫健壮性不足的第一版，已当场纠正（§8.10「守卫必须只声称它真正强制的东西」的反例）**：
   守卫名 `…_never_pass_percent_style_arguments` 是**全称**声称，而 helper 只扫 `SRC/` ⇒ 名字比机制宽。
   已把扫描域扩到 `src` + `tests` + `scripts`（别名导入也算：子串检查能命中
   `from loguru import logger as loguru_logger`，实测 `tests/ao_shaping/runners/test_runner_common.py:19`）。
   扩域**立刻抓到 1 处新的既有缺陷**：`scripts/slm/hw_differentiable_phase_opt.py:2628`
   ⇒ 该类缺陷的既有总数应为 **20**（`src` 19 + `scripts` 1），上文"19 处"是**当时的 src 口径**，保留不改。
   ⚠️ 这一处还暴露批量转换脚本的第二个坑：它含**符号标志** `%+.2f`，脚本式的 `%.Nf`→`{:.Nf}` 映射会
   **把 `+` 抹平**（变成 `{:.2f}`，正号消失）⇒ 手工转 `{:+.2f}`，并用 `str.format` **渲染样本实测**
   保号成立（`rot=+0.50°`）。扩域后守卫**先红后绿**：红时恰好只报这 1 处，改后 32 passed / 1 skipped。
   行为测试 `TestOnRecordFailureIsAttributable`（2 项，无硬件，复用本文件既有 `_FakeCamera/_FakeSlm/build_plan`）：
   钉「回调失败**不中断**采集（yield 记录数 == plan 数）」与「失败**必须计数上报**（收尾行含总条数，逐条行按 1/100 限流）」。
   非空性：改前 ⇒ `4 records dropped with zero attribution` 失败、另一项通过（后者正是"策略未变"的证词）。
   ⚠️ 本片踩到的两个**自身工具坑**（都会复发，记下）：① `run_pytest_ci.sh` 把不带 `-` 的参数一律当**路径**，
   故 `-k NAME` 会让 `-k` 悬空报 `expected one argument` ⇒ 必须写单 token `-k=NAME`；
   ② 守卫里我写的 `f"… '{}' …"` 中 `{}` 是**空表达式** ⇒ f-string SyntaxError，本地漏跑 `ast.parse` 就被容器在
   collection 阶段拦下 ⇒ **改完测试文件先在宿主机 AST/ruff，再进容器**。
   回归：`tools/slm/test_slm_train_data_collect` + `utils/image` + `gui/slm` **371 passed / 0 failed**；
   `test_conventions` **32 passed / 1 skipped**。术语门禁 `合计: 0 个文件, 0 行`。
   **BLE001 分簇进度**：已**逐处读过**的簇 = `src/drivers` 19→17、`src/gui` 27→26、`src/tools` 15→14、
   `tests/ao_shaping/drivers` 10→9（本行写于第 8 项之前，`tests` 片见下）。
   2026-10-24 全仓复测（`src`+`tests`+`scripts`，AST 判 silent）：**silent 共 90**，
   分 `src/` 63（drivers/ccd 15、tools/slm 11、gui/ccd 9、gui/slm 9、gui/r50 6、drivers/dm 2、
   optimizer/wfless 2、tools/wfs 3、runners 2…）、`tests/` 16（drivers/dm 7、utils/test_spots_calc.py 4、
   gui/zernike 2…）、`scripts/` 11（probe_optical_path_validity.py 2…）。
   ⇒ ~~**尚未逐处过**：`tests/` 非 drivers 的 7 处、`scripts/` 11 处、`src/runners` 2 处、`src/optimizer` 3 处~~
     **本行写于第 9/10 项之前，四个片已全部逐处过完**（2026-10-25 用**守卫自己的规则** `_blind_silent_handlers`
     重测**残留**：`src/` **61** —— 与棘轮基线逐字相等，口径自证一致；`tests/` 11 = drivers 9 + 非 drivers 2；
     `scripts/` 11；合计 **83**。分片：`src/runners` 1 + `src/optimizer` 2 = 3（第 9 项：5 处里 2 处真缺陷已修）、
     `tests/` 非 drivers 2（第 10 项：7 处里 5 处已修、2 处判定为只记录不动 —— ⚠️ 该判定的**前提是该文件整模块跳过**，前提已由第 15 项撤除，现非 drivers 残留 **0**）、`scripts/` 11（第 10 项：全部合法）。

8. ✅ **BLE001 `tests/ao_shaping/drivers` 片（10 处 silent）⇒ 9 处正确不动、1 处是「假绿+假证据」缺陷**（2026-10-24）：
   9 处不动的理由：fixture/工装的尽力 close（`ccd/test_exposure_states.py:99`、
   `test_hardware_integration.py:569`、`dm/multi_port_server.py:78/84/133`、`dm/port_listener.py:91/99/160`）。
   ⚠️ `multi_port_server.py:128`（吞掉 recv 循环异常）**实测不是假绿**：消费方
   `dm/test_micro_dm_multi_controller.py:83-88` 逐条断言记录内容（`HEADER`/`FOOTER`/`0x09`/载荷长度），
   丢消息会以断言失败或 `IndexError` 暴露 ⇒ 只缺归因，不缺判定 ⇒ 与 `src/drivers` 簇同判据，不动。
   **真缺陷**：`dm/test_micro_dm_report.py` 的 `dm` fixture 先试硬件、失败后**静默**回落 `SimMicroDM`，
   而该类 `pytestmark = pytest.mark.hardware`、它写的是**入库文档** `docs/micro-dm/micro-dm_report.md`
   （标题 "Micro DM Hardware Test Report"）⇒ 硬件**存在但打不开**（设备被占 / IP 错 / SDK 缺）时测试照样绿，
   报告把**仿真数字当设备验证结论**。这是"错数字不可归因"判据里后果最重的一类：产物本身是设备证据，
   且落在 `test_only_the_harness_or_hardware_tests_write_tracked_docs` 守卫的入库文档上。
   ⚠️ **两条静默路，旧结构只可能报告其中一条**：① 构造/`open()` 抛异常（被 `except: pass` 吞）；
   ② `open()` 成功但 `is_connected()` 为假 ⇒ 直接落到 `return SimMicroDM()`，**根本没有异常**，
   任何 `except` 里的补丁都覆盖不到它。
   修法（**策略不变**：仍允许回落，工装要在无硬件机器上可用）：回落逻辑抽成模块级 `_open_dm()`
   （fixture 只委托；顺带让这段风险逻辑能被离线测试直接驱动），两条路都写 `reason` 并在收尾打**一条**
   `logger.warning`；新增 `_backend_note(dm)`，把**实际服务的设备类**写进报告正文
   （仿真回退时明写 "NOT device evidence"）⇒ 产物**自我描述**，不再靠人回忆 fixture 干了什么。
   测试 4 项（模块级、不带 `hardware` 标记 ⇒ CI 默认可跑）：`test_sim_fallback_is_announced_on_both_routes`
   （两条路各跑一次）+ `_backend_note` 两个分支。
   **非空性实测**：把**改前 fixture 体逐字移植**成改前树里的 `_open_dm()` ⇒ 2 失败
   （`assert any("falling back to SimMicroDM" …)`）+ 2 通过（后两项与 fixture 体无关）；改后整文件
   **5 passed / 2 skipped**（skip 的 2 项要 `--hardware`）。
   🔴 **本片我自己制造、又当场抓住的坑，比修复本身更值钱**：测试里用
   `monkeypatch.setattr("tests.ao_shaping.drivers.dm.test_micro_dm_report.MicroDM", cls)` 这种**点号字符串目标**，
   在**命名空间包**下绑到了同一文件的**第二次 import** ⇒ "离线"单元测试**真的去连了硬件**：日志
   `Controller[24] 192.168.0.124 connection failed`（28 个控制器逐个超时）、`No controllers connected`，
   墙钟 **522 s**（`drivers` 是**阻塞分片** ⇒ CI 里白等 8 分钟），而且两个用例是**因为错误的原因**通过/失败。
   修法：目标改 `sys.modules[__name__]`（与本测试同一 globals，必然命中），并加**自证断言**
   `hardware_cls.instances == 1`（补丁没生效就直接失败，而不是悄悄退化成硬件测试）。
   改后同一组测试 **1.30 s**、整文件 **1.57 s** ⇒ **522 s 全是我这一处造成的**，不是既有仿真测试的开销
   （我一开始把它误记成既有开销，是靠改后计时才否证的）。
    ⇒ 新规则（已进本行，未固化成守卫）：**硬件包附近的"离线"测试必须断言桩对象确实被调用**，
    不能只断言返回值 —— 否则补丁失效会静默变成一次真硬件访问。

9. ✅ **BLE001 `src/runners` + `src/optimizer` 片（5 处 silent）⇒ 2 处真缺陷；并修掉本账本自己守卫的「扫描盲区」（§8.10 第二次）**（2026-10-24）：
   3 处不动：`runners/slm/zernike_matrix_runner.py:1377`（`finally` 里的尽力 `dev.close()`）、
   `optimizer/wfless/slm_square_shaping.py:1807/2183`（包住 `display_ctx.update(...)` —— 显示刷新失败**不该**中断优化闭环）。
   **真缺陷两处，都在闭环热路径上**：
   ① `runners/micro_drive/voltage_runner.py:_adc_worker` 用 `except Exception: pass` 吞掉 ADC 读取失败。
   不可归因的关键实测：消费端 `:381`（2026-10-25 重钉，`a05f7e3f` 在 ADC 读取路径插入退避块 +22 行，原 `:359`）只 `logger.debug("Drained {} ADC samples this cycle", …)` —— 只数**收到多少**，
   不数**丢了多少** ⇒ 设备掉线时记录的 `(时间戳, 电压)` 序列出现缺口而**整轮仍绿**。
   修法（策略不变：在工作线程里抛会**静默**终结采集）：计数 + 首条与每 100 条 WARNING + 线程收尾一条总数。
   ⚠️ **顺带实测到热自旋**：失败路径立即返回，20 ms 窗口内**空转 40,800 次**（≈2M 次/秒），
   一边猛敲已失效的 DAQ、一边和电压更新循环抢 CPU ⇒ 加 `_ADC_READ_RETRY_S = 0.005` 的**仅错误路径**退避
   （成功路径的节拍由 DAQ 采样窗自身决定，**不动**）。
   ② `optimizer/combined_optimizer.py:352`（2026-10-25 重钉：该行在 `9bfc7f18` 时代即 `:352`，原误记 `:353`）吞掉 `dm.send_voltages(zeros)` —— 这是 `finally` 里的**安全复位**。
   这里**不能抛**（会掩盖结束本轮的那个异常），但**静默更糟**：变形镜物理上停在最后一组（可能很大的）电压，
   下一个人站在实验系统前**无从得知**。改成 `logger.error`。
   证据分层（按可验证性选，不是一律写测试）：① 有离线行为测试 —— `adc` 是注入参数且模块离线可导入
   （`TYPE_CHECKING` + `try/except ImportError` 的 `ADC_AVAILABLE`），新 `tests/ao_shaping/runners/test_voltage_adc_worker.py`
   5 项钉「失败不终结采集 / 失败被计数上报 / 不热自旋（`calls < 50`）/ 限流行数 = `1 + n//100` / 恢复后仍入队且仍告警」。
   ② 无处可跑行为测试 —— `optimize_pib` 要开相机与 DM，无离线工装 ⇒ 新 `tests/ao_shaping/optimizer/test_combined_optimizer_voltage_reset.py`
   用 AST 钉「每个被守卫的 `send_voltages` 处理器必须记日志」，且**先断言该守卫结构存在**（结构挪走就失败，不让它空过）。
   **非空性实测**：两项新测试跑在改前树 ⇒ **5 失败 / 2 通过**，而通过的 2 项恰是「策略未变」那两项
   （采集不中断、守卫结构存在）—— 红绿分布本身即证据。
   🔴 **本片最大的收获是本账本自己的守卫漏扫**：`utils/__init__.py:32` 把 loguru 的 `logger` **再导出**，
   而 9bfc7f18 落地的守卫按字面 `from loguru import logger` 过滤 ⇒ **20 个模块从未被扫**（含 `combined_optimizer`）。
   即：同一提交里我把扫描域从 `src` 扩到 `src+tests+scripts`，却仍按**一种写法**认 logger —— 守卫的作用域比它的名字窄，
   正是 §8.10 第一次实例的同类复犯。已改为「直接导入 **或** 从 `ao_shaping.*` 再导出，且不含 `logging.getLogger(`」，
   并加 `test_the_loguru_guard_follows_the_logger_not_one_spelling_of_its_import` **把边界的两侧都钉住**
   （再导出模块必须算进、stdlib `logging` 模块必须排除，否则守卫会把正确代码判成缺陷；且该测试先断言前提仍成立）。
   实测：全仓**没有**同时混用两种 logger 的文件 ⇒ 不会整文件漏扫；扩域后重扫三棵树 ⇒ **dropped-arg 归 0**
   （该类缺陷全仓既有总数 20 = `src` 19 + `scripts` 1，已全部修完）。
   ⚠️ 本片我自己犯的两个错，都被工具抓住：① ADC 测试的限流公式写成 `1 + (n-1)//100`，在 n 为 100 整数倍时**差一**
   （实际规则是第 1 条 + 每 100 条 ⇒ `1 + n//100`）—— 容器实测 `40,800` 次失败时暴露；
   ② 我把整棵 `tests/ao_shaping/optimizer` 在**单进程** 10 g 下执行 ⇒ **exit 137（SIGKILL）且无 summary**，
   极易误读成测试挂死；这不是代码缺陷，是**我的调用方式错** —— 该目录必须按 CI 的方式 `AO_TEST_PER_FILE=1`，
   改后 **1271 passed / 3 skipped / exit 0**。
   ⚠️ 另一次"先怀疑后读上下文"的否证：`voltage_runner.py:90-91` 的 `HW_VOLTAGE_MIN/MAX = -20.0/120.0`
   看着像我在 `gui/` 收口过的**重复边界漂移**，读 `:87-89` 才发现明文写着同步路径与异步路径**有意不统一**
   （异步用驱动自己的 `VOLTAGE_MIN/MAX`）⇒ 不动。本会话这是第 4 次"模式匹配出的缺陷被上下文否证"。
   **BLE001 分簇状态**：`src/` 各簇已**全部逐处读过**（drivers 19→17、gui 27→26、tools 15→14、runners+optimizer 5→3 保留）；
   `tests/` 与 `scripts/` 见下第 10 项。

10. ✅ **BLE001 收尾两片：`tests/` 非 drivers 7 处 ⇒ 5 处修、2 处只记录不动；`scripts/` 11 处 ⇒ 全部合法**（2026-10-25）：
    判据在这两棵树**换了一把尺子**：`tests/` 里吞异常的真正危害是**空过的绿**（阻塞分片里尤其贵），
    `scripts/` 里则是**报告表格里少一行/少一个数**（报告是结论的唯一事实源）。
    **修掉的 5 处**：
    ① `utils/test_spots_calc.py` 4 处：CuPy 奇偶校验断言写在 `try: if CUPY_AVAILABLE: … except Exception: pass
    # Skip if cupy not working` 里 ⇒ 把**可用性**与**正确性**混为一谈：无 GPU 时断言根本不执行却记为 passed
    （绿而无内容），而**装了却坏掉**的 CuPy 抛错同样被吞 ⇒ 真奇偶缺陷不可见。改为本仓既有的
    「硬件跳过模式」`pytest.skip(CUPY_UNAVAILABLE_REASON)`（可用性 ⇒ **显式跳过**，可见），断言不再包 `try`
    （正确性 ⇒ 坏 GPU 直接失败）。⚠️ 顺带发现 `CUPY_UNAVAILABLE_REASON`（区分"未安装/初始化失败/无 CUDA 设备"
    三种原因）**定义了却从未被引用** —— 我第一版硬编码了自己的跳过文案四遍，已改为复用该单一事实源。
    实测账目：这两文件从 `N passed` 变 **65 passed / 4 skipped**，跳过原因逐条打印
    （`CuPy is not installed in the active Python environment (No module named 'cupy')`）。
    ② `optimizer/wfless/test_slm_zernike_rmse.py:157`：`except Exception: pass` 本身是**有据可依的通过条件**
    （"验证在开硬件之前跑完，能走到硬件错即算通过"），**不是**误吞；真正的问题在**配置构造写在 `try` 内**
    ⇒ `SlmZernikePibConfig` / `CameraParamsPib` 字段改名会抛 `TypeError` 并被当作"已到硬件"吞掉，
    测试在**传入非法配置**的情况下依旧绿。修法是把配置**提到 `try` 外**并把两个 handler 合成一个带断言的
    （顺带该点不再是静默吞）。⚠️ 这条风险不是假想：`SlmZernikePibConfig` 正是本会话收敛过的 dataclass API。
    **变异实测（决定性）**：把 `center=` 改成 `ceners=` 后 —— 旧版 **1 passed（盲）**，新版
    **1 failed：`TypeError: SlmZernikePibConfig.__init__() got an unexpected keyword argument 'ceners'`（有牙）**。
    **只记录不动的 2 处**：`gui/zernike/test_zernike_response_matrix_ui.py:423/657` —— 该文件
    `:19 pytestmark = pytest.mark.skip(...)` **整模块硬跳过**（记录当时的状态；现已由第 15 项撤除），不构成 CI 空绿；且其真正的弱点**不是吞异常**而是
    `if mock_calibrate.called:` 这个**条件断言**（mock 未被调用时不做任何断言却通过）。**不改**：这类重度 mock 的
    Streamlit 测试我跑不起来，改它本身就是把它弄坏的方式 ⇒ 记为**已知弱点**，要动必须先让模块可跑。
    ⚠️ 本判定的**前提是该文件整模块跳过**（"跑不起来"即指此）；第 15 项实测该前提为假（无任何 AppTest、全 mock、1.7 s 跑完），
    跳过已撤，此 2 处随复活**一并改严**（`assert_called_once`、去吞异常）⇒ 本条「只记录不动」判定**已失效**，见第 15 项。
    **`scripts/` 11 处全部合法**，分四类：`finally` 尽力清理（`set_phase_overlay(None)` ×3、
    `display_data(zeros)`、`dev.close()`、`wfs.close()`）；**搜索循环里跳过读不到的候选**
    （`generate_fouriergsnet_sim_report.py:181` 遍历各候选目录找含所需键的 `config.json` —— 跳过坏候选是**控制流**，
    与 §8.9 第 5 项 ffmpeg 枚举同判据）；**逐条跳过损坏产物**（`generate_bench_twin_report.py:348`，注释已写明
    "落盘产物损坏时只丢这一项"）；**可选设备元数据**（`probe_optical_path_validity.py:1351/1356`）。
    ⚠️ 我先前怀疑 `generate_inverse_shaping_model_report.py:325` 的 `except: continue` 会**悄悄缩小发布表格的样本集**
    —— **实测否证**：行字典同时发布 `"n": len(norms)` 与 `"records": len(records)`，跳过会表现为 `n < records`，
    **归因本来就有** ⇒ 不动。另实测 `cam_exposure_range_ms` / `cam_hardware` 在两处赋值之外**全文件零引用**，
    且该探针的 verdict 分支（`:930-957`）走的是设备互斥锁、不读它们 ⇒ 只影响描述性元数据，不影响结论。
    **BLE001 全仓终账（实测，非减法推算）**：silent（`pass`/`continue` 且无 raise）由 **90 → 83**
    （`src` 63→61、`tests` 16→11、`scripts` 11 不变）；`src`/`tests`/`scripts` **三棵树已全部逐处判定完毕**。
    回归：`tests/ao_shaping/utils` 按 CI 的 `AO_TEST_PER_FILE=1` 跑 **1326 passed / 4 skipped / exit 0**，
    `tests/ao_shaping/optimizer` 同法 **1261 passed / 3 skipped / exit 0**。

11. ✅ **BLE001 分簇的结论已被钉成守卫：`test_conventions.py::test_blind_silent_exception_handlers_in_src_do_not_grow`（棘轮，基线 61）**（2026-10-25）：
    五片逐个判读的价值会随下一次随手 `except Exception: pass` 蒸发 ⇒ 加**只增不减**的棘轮。
    - **基线与守卫用同一条 AST 规则自测**（不依赖 ruff）：`src/` **61**；同时独立复测
      `tests/` 11、`scripts/` 11 ⇒ 三棵树合计 **83**，与 §8.10 之前用 ruff-`BLE001` 定位 + AST 复核得到的
      **83 逐树一致** —— 两个互不相同的检测器给出同一数，规则本身才算定义清楚（守卫不得用它算不出来的量做基线）。
    - **计数棘轮，不用 `path:line` allowlist**：本会话已两次为陈旧行号重钉（`977b663c` 甚至**按插入行数推算**而**差一**，
      实测应为 `:970/975/980`）⇒ 行号 pin 会随任何无关插入腐烂，计数不会；失败时打印全量 offender 列表保证仍可 diff。
    - **作用域只 `src/`**（守卫名与断言同宽，§8.10 的正面执行）：`tests/`/`scripts/` 里的尽力 close 是合法工装噪声，
      并进来会让硬门禁狼来了。基线**双向**：若计数下降也必须显式改小，不让它变成"没人再推导过的数字"。
    - **非空性实测**：临时树里注入一个 `except Exception: pass` ⇒ 守卫失败
      `62 > baseline 61` 并点名注入处；不注入 ⇒ 34 passed / 1 skipped（既有 ruff-not-on-PATH 跳过）。
    ⚠️ **本片差点产出一个"跑过但什么都没断言"的验证——正是本账本整条 campaign 的论点本身**：
    我第一次非空性验证往 `src/ao_shaping/utils/cli_params.py` 注入，而该文件**早已迁到 `utils/cli/params.py`**
    ⇒ 注入脚本 `FileNotFoundError` 退出，但后续 pytest **仍照跑**，于是在**未改动的树**上得到
    `1 passed` —— 一个"证明守卫有牙"的验证其实什么都没验。修法：验证脚本的副作用必须**先自证**
    （本次改为 `grep -c` 确认注入落地 + `&&` 串接使失败即停），再让被验证对象跑。
    规则：**任何"证明某检查非空"的步骤，必须先断言该检查的输入确实变了**；否则红色与绿色都不承载信息。

12. ✅ **BLE001 收尾时顺带量到的另一类"检查了但什么都没断言"：`tests/` 里 30 个「全部断言都在 `if` 里」的测试 ⇒ 修 3 处（逐处差分证明），其余按类保留不钉**（2026-10-25）：
    这是吞异常在 `tests/` 里的同构体：不是"失败不可归因"，而是**通过不承载信息**。
    AST 测法（与 §8.9 第 11 项不同的一条独立规则，别混用）：一个 `test_*` 函数里的**断言事件**
    = `Assert` 节点 + `assert_*`（mock）+ `pytest.raises/fail/xfail`；若**每一个**断言事件都落在某个
    `If` 之下 ⇒ 该测试可以在前提不成立时零断言通过。全仓命中 **30**。
    分布（前几）：`drivers/slm/test_slm_hardware.py` 5（硬件门控，CI 不产生绿）、
    `gui/slm/test_pattern_controls.py` 3、`gui/zernike/test_zernike_response_matrix_ui.py` 2（记录当时整模块
    `pytest.mark.skip` 硬跳过；⚠️ 该跳过已由第 15 项撤除，此 2 处条件断言已随复活改严）、`optimizer/wfless/test_slm_model_in_loop.py` 2、
    `scripts/test_build_hw_sweep_plan.py` 2，其余各 1。
    **修掉 3 处，每处都跑"旧盲 / 新有牙"差分**（改前测试与改后测试对**同一变异**的反应）：
    ① `display/test_display_layering.py::test_no_module_scope_algorithm_import`：AST 扫描型守卫，
    全部断言挂在 `isinstance(node, ast.Import*)` 上 ⇒ **若模块一个顶层 import 都没有，扫描什么也不看就通过**。
    加前提断言（顶层 import 列表非空）。差分：把该文件 6 个顶层 import 剥掉（AST 复测确认
    `top-level imports after mutation: 0` 后才允许跑）⇒ 旧版 `1 passed`（盲），新版失败并点名
    "declares no module-scope import: the scan below would pass without examining anything"。
    ② `optimizer/wfless/test_slm_zernike_primitives.py::test_module_defines_no_module_level_mutable_state`：
    名字声称"无模块级可变状态"，机制**只匹配 `ast.Dict`** ⇒ 模块级 list / set / 推导式 / `dict()` 之类
    同样会造成两个 importer 在一个进程里共享槽位轮换状态，却能全身而绿。扩到 Dict/List/Set +
    ListComp/SetComp/DictComp + `dict/list/set/deque/defaultdict/Counter/OrderedDict()`；
    先实测真实模块**命中 0** ⇒ 扩宽不会把既有代码判成缺陷。差分：注入模块级 list + set ⇒
    旧版 `2 passed`（盲），新版点名 `_INJECTED_LIST_STATE = <List> at line 189; _INJECTED_SET_STATE = <Set> at line 190`。
    ③ `optimizer/wfless/test_slm_model_in_loop.py`：**实测发现该测试自写下起就没断言过任何东西** ——
    它的循环是 `for row in history[1:]: if not row["accepted"]: assert row["reason"]`，而这份小仿真
    **一条 rejected 轮次都不产生**（前提永假）。改名
    `test_every_history_row_names_a_machine_readable_reason`，把检查改为对**全部** history 行无条件成立
    （另加一条一致性检查：被拒的行不得标成 `"accepted"`）；理由写在 docstring 里：
    拒绝分支本身已由 `acceptance_verdict` 的直接测试覆盖（§该文件 `:811-893` 十余项），端到端唯一能证明的是
    **verdict 的 reason 确实落到记录行上**。差分：把模块里 `reason=` 全部置空 ⇒ 旧版 `2 passed`（盲），
    新版失败并给出 `round 1 recorded reason=''`。⚠️ 注意这不是"加个断言让它更严"，而是
    **原测试的通过条件恒真**；`optimizer` 是阻塞分片，所以修复必须同时证明它**不会把既有绿变红**（先测 0 命中 / 真实 159 passed）。
    **其余 ~27 处不钉棘轮**（与第 11 项相反的判定，写清楚免得下次重议）：其中多数是**循环内逐元素断言**
    （测试自己构造的数据上 `if isinstance(...)` 过滤，正常写法），且 5 处在硬跳过模块里；
    "条件断言"这条规则的假阳性压力远大于"盲吞异常"，计数基线会把合法写法一起钉死。
    `gui/zernike` 两处维持第 10 项判定：真弱点是 `if mock_calibrate.called:` 这类条件断言，
    但重度 mock 的 Streamlit 测试我跑不起来 ⇒ 只记录不动。
    ⚠️ 此「维持」与「跑不起来」的前提（整模块跳过）已由第 15 项实测推翻，2 处条件断言已随复活
    **一并改严且仍全绿** ⇒ 本句对 `gui/zernike` 的判定**已失效**，以第 15 项为准。
    回归：三个被改文件 **159 passed / 0 failed**；旧测试名在全仓**无其它引用**（改名安全）。

13. ✅ **R-64 的另一半（"`assert` 当运行期护栏"）经实测已收口**：现树 `src/` 共 36 条 `assert`，设备状态类 0 ⇒ 钉守卫防回归（原记 99/111 与"含硬件安全联锁"均为各簇收口前的口径）（2026-10-25）：
    - **AST 全量实测**（不是 grep）：`src/` `assert` 共 **36** = `drivers/` **5** + 其外 **31**。
      `drivers/` 那 5 条与既有双向 allowlist `_DRIVERS_ASSERT_ALLOWLIST` **逐字一致**，且全是
      `width_val is not None` / `isinstance(result, np.ndarray)` / `max_grayscale is not None` 一类**本地不变式**。
      其外 31 条**逐条读过**：`img.ndim == 2`、`G.ndim >= 2`、`type(x) is type(y)`、`self.lut is not None`、
      `out.shape == panel_phase.shape` … **全部是本函数刚算出的值的不变式，没有一条断言设备生命周期状态**
      ⇒ §8.7 表里"含设备已打开断言、曝光范围检查、安全检查"在**当前树**上不成立：那些联锁原本就在 `drivers/`，
      并已由 #174 系列 + CCD `_ensure_open()` + WFS `_ensure_open()` + NLight 显式 `raise` + cartographer 各簇收口。
    - ⚠️ **`python -O` 可达性实测：全仓（`pyproject` scripts、CI workflow、docs、Streamlit 启动、源码）
      没有任何 `-O` / `PYTHONOPTIMIZE` 使用** ⇒ "被 `-O` 整条删除"在**本仓当前不是活风险**。
      诚实记法：本守卫是**回归保险**，不是修现存缺陷。但它不依赖 `-O`：`assert` 抛的异常类型
      调用方**无法窄化**，正是 `92dbe716` 追过的 `except AssertionError` 静默失配那一类 —— 这条论证与 `-O` 无关。
    - **新守卫** `test_conventions.py::test_no_assert_on_device_state_outside_the_drivers_tree`：
      AST 扫 `src/` 全树，跳过 `ao_shaping/drivers/`（那里由 `test_drivers_only_assert_internal_invariants` 双向钉住），
      按**设备生命周期标识符**词表命中（`is_open` / `is_connected` / `is_ready` / `_instrument_handle` / `_dll` /
      `_handle` / `_device_open` / `_ensure_open` / `_sdk` / `_miicam` / `_session`）。基线 **0 ⇒ 没有 allowlist 可腐烂**。
      词表有意**窄**：不做"生产代码一律禁 `assert`"的全域禁令 —— 那会重开 §8.10 已判的"内部不变式**保留** `assert`"
      （改 `raise` 只增加运行开销与 traceback 噪声，零安全收益）。
    - **非空性实测**（并按第 11 项规则先自证注入落地）：临时树里往 `optimizer/combined_optimizer.py`
      注入 `assert dev.is_open()` ⇒ 守卫失败并点名 `combined_optimizer.py:369`；未注入 ⇒ 35 passed / 1 skipped
      （既有 ruff-not-on-PATH 跳过）。基线是用**守卫自己的规则**在同一棵树上重测得到（31 / 0），
      不借用别条规则的数 —— 第 11 项那条教训在这里直接省掉了一次"守卫跑的是另一套口径"的坑。
    - 结论：**R-64 的 `assert` 半项关闭**；§1.7 批与 R-64 行里两处过期数（99 / 111）按"重钉"就地更正，
      §8.7 那张带日期的 2026-10-23 测量表**保留原数**（它记录的是当时口径）。

14. ✅ **`except BaseException` 全量复核收口：实测 5 处，逐处判定正当，并把「线程边界不得收窄」钉成守卫**（2026-10-25）：
    - **AST 实测 5 处**（不是 grep 计数）：`optimizer/wfless/slm_zernike_pib.py:1355`、
      `tools/slm/transfer/workflow.py:78`、`utils/image/hardware_utils.py:627`、
      `scripts/slm/diff_beam_capture_flat.py:68`、`scripts/slm/hw_gs_defocus_probe.py:1071`
      ⇒ §8.4 的「真风险只有 2 处」与 R-64 行的「禁改 2 处」均为当时口径，已就地重钉。
    - **逐处判定：没有一处需要改**（本项初始假设「能在硬件运动中途静默吞掉中断」被实测推翻）：
      `workflow.py:78` 记录状态后 `raise`（本就重抛）；`hardware_utils.py:627`、
      `diff_beam_capture_flat.py:68`、`hw_gs_defocus_probe.py:1071` 是三处**同构的线程边界捕获**，
      调用方确定重抛（实测 `:77 raise error[0]`、`:1082 raise box["error"]`）——
      它们正是把「原生 SDK 调用无限挂起」变成**有界且可归因**的机制
      （后者文档还引用了仓库里那次 900 s 强杀记录）；
      `slm_zernike_pib.py:1355` 在 `_apply_best_on_exit()` 退出清理里：异常类型与消息都写进日志，
      再强制 `set_grayscale(0)` 回安全平场，内层兜底 `except Exception` 记 "SLM state unknown"
      ⇒ 不重抛但**不静默**，且宽捕获是必需的：收窄后 Ctrl-C 走不到安全态。
    - **新守卫** `test_conventions.py::test_thread_boundary_exception_capture_keeps_base_exception`：
      AST 扫 `src` + `scripts` + `tests`，命中「起线程 + 把捕获的异常存进容器 + 同子树内重抛该容器」的处理器，
      要求捕获类型必须是 `BaseException`。基线 3 处全部合规 ⇒ **违规数 0**；并断言「至少扫到 1 处」自证非空。
    - ⚠️ **规则形状差点写错**：第一版只认 `raise <name>`，实测 3 处只看见 2 处 —— 漏掉
      `raise box["error"]` 这种**下标**形式。就此落守卫会**悄悄少钉一处**却按名字宣称全覆盖
      （§8.10 那条红线我第三次差点踩）。
    - ⚠️ **差分第一次无效**：临时树从 `HEAD` 切出，而新守卫还在工作区未提交 ⇒ 树里没有该测试，
      输出只是 `36 deselected` 而非失败。教训：**差分必须把被测的新代码一并带进临时树**。
      补上后实测：把 `hardware_utils.py:627` 改成 `except Exception` ⇒ 守卫失败并点名
      `hardware_utils.py:627: Exception`；未改动 ⇒ 36 passed / 1 skipped（既有 ruff-not-on-PATH 跳过）。
    - 宿主 ruff 在落容器**之前**拦下一个真错：标识符写成 `re-raised`，连字符被解析成 `re - raised =`，
      报 `invalid-syntax: Invalid assignment target` —— 省一次容器运行。

15. ✅ **`gui/zernike` UI 测试的整模块硬跳过已撤除：22 条全部复活（其中 15 条本来就该跑），并顺带清掉该文件里 2 处盲吞 + 2 处条件断言**（2026-10-25）：
    - **跳过理由与文件自身矛盾**（这正是它长期无人质疑的原因）：`pytestmark = pytest.mark.skip(reason="Requires Streamlit runtime and live SLM/WFS hardware")`，
      但实测该文件**没有任何 `AppTest`**、自带 `MockSessionState`、`st.button` 等被 `patch.object`，
      且被测的 `calibrate_zernike_response_matrix` 在测试里就是 `MagicMock` ⇒ 既不需 Streamlit 运行时，也不需设备。
      **实测证据**：撤除跳过行后整文件 **1.7 s** 结束、无挂起（挂起风险实测不存在）；撤跳过但**不改任何断言**时 22 条里 **15 条直接通过**。
    - **7 条失败全部是测试侧腐烂，不是代码缺陷** —— 每条都先用 UI 现值核实，才改期望（绝不反向）：
      标签 `"SLM Shift X"` 实际是 `"SLM Shift X (像素)"`（UI `:1082`；测试把引号两端都锚进源码文本，故永不匹配）；
      `zrm_wfs_exp_time` 默认 `0.0` 实际 `0.01`（UI `:138` 带注释 "Valid default within [0.002, 86]"，即有意选择）；
      子标题 `WFS设置` 实际 `WFS参数`（UI `:965`）；`plot_placeholder = st.empty()` 在 UI 里**根本不存在**
      （全文件 `placeholder` 零命中，只有 `:758-759` 的 `progress_bar` / `status_text`）⇒ 断言的是幻影，**删断言而非改名迁就**；
      `_progress_callback(data)` 实际签名是 `_progress_callback(progress_file, progress_data)`（UI `:639`），
      三处调用点全改（`:521`/`:689`/`:706` —— 第三处在 `error_data` 分支，第一轮读取窗口没覆盖到）；
      `zrm_slm_number=1` 是喂错键：`connect_slm` 读的是 `zrm_slm_selection`（`:237`，且 `:108-109` 初始化为 1），
      `zrm_slm_number` 只是连接后回填的实际编号（`:255`）⇒ 已专门查过生产路径 `slm_number` 不会为 `None`，不是 UI 缺陷。
    - ⚠️ **这批测试的断言机制本身是「源码文本 grep」**（`open(ui).read()` 后 `assert '"…"' in content`）：
      测的是**字符串是否出现**而不是行为，所以 UI 一改措辞它就腐烂；加上整模块跳过 ⇒ 腐烂完全不可见。
      本项只把期望对齐现值并删幻影断言，**没有**重写成行为断言（那是单独一刀，仍开放）。
    - **随复活一并修掉两处「检查了但什么都没断言」**（第 12 项那一类；此前正因为整模块跳过才被判「只记录不动」）：
      `test_calibrate_receives_parameters` 与 `test_threading_for_calibration` 原本是
      `try: render_calibrate_mode()` + `except Exception: pass`，再把**全部**断言挂在 `if mock_*.called:` 之下
      ⇒ 结构上永远不会失败。改成 `assert_called_once()` 并去掉吞异常之后**仍然通过**
      （实测 `-k=calibrate_receives_parameters` 1 passed；整文件 22 passed）⇒ 校准确被调用一次，
      线程确有 `target` / `daemon=True` / `args`（UI `:1225` 确实起线程），掩码从来不存在。
    - **计数联动**（用棘轮守卫自己的规则 `_blind_silent_handlers` 重测，口径与第 11 项一致）：
      `tests/` 盲吞 **11 → 9**（非 drivers **2 → 0**），全仓 silent **83 → 81**；`src/` **61 不变** ⇒ 棘轮基线不受影响。
      上面第 7-10 项那段「非 drivers 2 = 只记录不动」已就地标注其前提已撤。
    - **同类普查（只记数，理由未验）**：`tests/` 仍有 **9 条模块级跳过** —— `drivers/test_tm.py`（COM3）、
      `drivers/slm/test_slm_miicam.py`、`drivers/test_driver.py`、`drivers/ccd/test_miicam.py`、
      `ao/test_combine.py`、`ao/test_wfless.py`、`ao/test_wf.py`、`display/test_windows.py`（pygame 显示），
      以及 `algorithm/test_differentiable_defocus_uniformity.py` 的 `skipif`（缺 checkpoint，条件式，合法）。
      除本项这条之外**都没验证过理由是否成立** —— 本项的教训是「理由必须可实测」，不是「凡跳过都可疑」。

16. ✅ **「源码文本 grep」断言类普查：实测 113 处 / 26 文件，其中只有一个文件算真腐烂**（2026-10-25）：
    - **普查口径是 AST 而不是全文 grep**：只算 `assert <字面量> in <变量>` 且该变量确实绑自
      `read()` / `read_text()`（跟着赋值走）⇒ 全 `tests/` **113 处 / 26 个文件**。
    - ⚠️ **第一版分类错了，错在「文件里有没有 `.py` 字面量」**：那样会得出「101 处是源码 grep」的**假结论**。
      报告生成类测试**同时**引用生成器脚本路径**和**生成的 markdown，两种扩展名共存 ⇒
      扩展名存在性**说不出断言读的到底是哪一样**。正确判据是**断言比较的那个变量绑自哪里**，
      再加上字面量的种类（措辞 / 标识符 / 产物内容）。
    - 更正后的实测三分（逐文件把断言字面量抽出来看，不猜）：
      1. **产物内容断言 —— 合法，产物就是被测物**：`test_generate_slm_pib_sim_report_disturbance.py` 17 处
         （断言的是生成的表格单元与实测数字）、`test_generate_capability_report.py`、
         `test_summarize_optics_change_blocks.py` 等 ⇒ 这类措辞一改就是**产品变了**，不是测试腐烂。
      2. **机制不变式守卫 —— 合法，与 `test_conventions.py` 同族**：`test_term_gate_enforced.py`
         （断言 CI 里必须出现 `term_unify_20261016.py --check`）、`test_utils_import_is_sdk_free.py`
         （`__getattr__` / `_LAZY_EXPORTS`）、`test_no_historical_bench_defaults.py`、`test_report_provenance.py`
         ⇒ 断言的是**标识符与命令**，是有意的源码检查。
      3. **措辞脆弱 —— 这才是真腐烂，且只有第 15 项那个文件**：
         `tests/ao_shaping/gui/zernike/test_zernike_response_matrix_ui.py` 26 处把**会话键**（`zrm_shift_x`）
         与**用户可见标签**（`"SLM Shift X (像素)"`、`WFS参数`）混在同一机制里，而标签按设计就会变
         ⇒ 第 15 项已把期望对齐 UI 现值；**机制本身（源码 grep 代替行为断言）仍是开放的一刀**，未动。
    - **不落守卫**（有意）：普查结论是「26 个文件里只有 1 个属于腐烂」，任何覆盖全 `tests/` 的禁令都会把
      上面 1、2 两类合法用法直接判成违规 —— §8.10 那条「不得宣称多于实测」在**普查驱动**的工作上同样适用。

17. ✅ **删掉 `tests/ao_shaping/ao/` 里 3 个「永远不可能失败」的硬件演示：0 断言 + 整模块跳过 + 调用已过期 API**（2026-10-25）：
    - **实测三件套**：`test_wf.py` 39 行 / `test_wfless.py` 51 行 / `test_combine.py` 93 行，
      `assert` 计数 **0 / 0 / 0**，且各自还调 `plt.show()` ⇒ 它们不是测试，是**在硬件上跑图的脚本**
      （AGENTS 早写明硬件编排归 `tools/`、实验叙述归 `report/`、测试一律 mock-first）。
    - **其中一条即便接上设备也会炸**：`test_wf.py:13` 调 `optimizer_rms_dm(init_v=…, epochs=20_000)`，
      而现签名 `rms.py:50` 为 `dm: DM | None = None`、`:90` 明确 "dm parameter is required" ⇒ 那条整模块
      跳过把这笔**已过期调用**一并掩掉了（`test_combine.py:25` 同形）。
    - **删除不丢覆盖**（逐 API 实测）：`optimizer_rms_dm` 由 `optimizer/wf/test_rms_sim_seam.py`（9 断言）覆盖；
      `optimize_pib` 由 6 处覆盖（`test_spid_pib_path_characterization.py` 15、`test_sensorless_sim_params.py` 15、
      `test_pib_objective_sign.py` 10、`test_spgd_convention.py` 8、`test_pib_objective_columns.py` 5、
      `test_combined_optimizer_voltage_reset.py` 3），最后那个也正是 `combined_optimizer` 的行为测试。
    - **同目录的 `test_respond_matrix.py` 保留**：它不是整模块 `pytestmark`，而是在第 12 行按
      `AO_RUN_HARDWARE=1` **逐测试**放行（含 2 条断言）⇒ 是正当的硬件门，不是掩码。
      目录未清空 ⇒ `pytest.yml:107` 的 `tests/ao_shaping/ao` 分片路径仍然有效，**无需改 CI**。
    - **容器实测**：删后 `tests/ao_shaping/ao` + 分片覆盖守卫 `test_pytest_workflow_shards_cover_tree.py`
      + `test_conventions.py` 合跑 **41 passed / 2 skipped**，两条 skip 都已知且正当
      （硬件门 `test_respond_matrix.py:12`、`ruff not on PATH`）。
    - ⚠️ **顺带记一条我自己的测量错误（发现即弃用，未据其下任何结论）**：普查「无断言测试」时把
      `any(<生成器> for d in [fn.decorator_list])` 写成对**生成器对象**取真值 ⇒ 恒为真，过滤器等于不存在，
      输出把 fixture、`__init__`、桩类全列成「测试」。本项结论改用逐文件 `grep -c assert` 加签名与调用点核对**重测**得到。
      另：`pyproject` 里「忽略 `tests/ao_shaping/drivers`」已不存在 —— 现行是 `drivers/conftest.py` +
      `_hardware_gate.py` 用 `hardware` 标记门控（注释见 `:25-31`），属**有意替换**而非回归；
      我按旧账本以为是漂移，实测后才纠正（第 13 项「重钉要 grep 实测、不要靠算术」在这里再次生效）。

18. ✅ **「无断言测试」普查：实测 107 处 / 38 文件，但这一类**不成立**——已否决「批量补断言」**（2026-10-25）：
    - **口径**：AST，只取模块顶层 `test_*` 与 `Test*` 类方法（不数嵌套辅助函数，不数 `conftest` fixture）；
      断言判定包含 `assert`、`np.testing.assert_*`、`self.assert*`、`pytest.raises`、`raise AssertionError`。
      ⚠️ 第一版口径漏了 numpy 的 `assert_allclose` 一族与 bare `raises(`，先把 **265 处**误判成缺陷 ——
      比第 17 项那次 `any(<生成器>)` 恒真更隐蔽，因为那个数字看着**很合理**。
    - **更正后 107 处 / 38 文件**；再按「这些函数实际在做什么」分（同一函数可多标签）：
      仅 `print` 52、无任何可测痕迹 52、写产物 9、设备报告工装 8、交互绘图 6、动画 3。
    - **否决依据是两条实测反例，不是推断**：
      1. `drivers/ccd/test_miicam_frame_callback.py::test_failing_pull_does_not_raise_out_of_the_callback`
         0 断言，而它本身就是**「不得抛出」契约**（docstring：SDK 线程不能 unwind，这正是处理器存在的原因）——
         处理器一旦抛出 pytest 当场失败；同文件相邻两测各有 5 条与 1 条断言 ⇒ 该文件是**有意分层**。
      2. `drivers/dm/test_micro_dm_report.py` 里被判「无断言」的只是**生成报告的几个方法**，
         文件本身在 `:507`、`:526-538` 用 6 条断言把「服务后端必须写进报告」钉住（第 8 项那一刀的守卫）。
    - ⇒ **`assert` 数量不是缺陷信号**。第 17 项那三个文件之所以删，靠的是**合取**：整模块跳过 +
      调用已过期 API（接上设备也会 `raise`）+ 收尾 `plt.show()` 的绘图脚本 + 目录不镜像任何 `src/` 包 +
      覆盖已在别处实测存在（9/15/10/8/5/3 条断言），而不是「没有 assert」。
    - **决定不做**：给 107 处批量补断言 —— 既没有验收判据，也没有设备可对照，属越界改动（minimal-diff 红线）。

19. ✅ **`gui/zernike` UI 测试的「源码文本 grep」机制已换成 AST 结构断言：该文件 26 → 3 处，全仓 113 → 90**（2026-10-25，第 15/16 项留下的那一刀）：
    - **改的是断言机制，不是期望值**：7 个测试（`test_shift_inputs_in_slm_settings`、`test_wfs_settings_section_exists`、
      `test_mla_res_options`、三个 checkbox 测试、`test_progress_bar_placeholder_exists`）原先清一色是
      `open(ui).read()` + `assert "<字面量>" in content`；现在用模块级 `_widget_call(key)` 在 UI 的 **AST** 上找
      「结果落进 `st.session_state.<key>` 的那个 widget 调用」，再断言 widget 种类与标签。
    - 新助手必须处理**两跳绑定**：曝光时间不是直接落键，而是 `new_exp = st.number_input(...)`（UI `:1010`）
      之后才 `st.session_state.zrm_wfs_exp_time = new_exp`（`:1024`）⇒ 只认直接赋值会误报「没有 widget」。
    - **顺手杀掉最脆的一条**：`assert "options=[0, 1, 2, 3, 4]" in content` 钉的是**源码排版**
      （列表字面量的空格写法），任何人重排代码都会让它变红；现在断言 `selectbox` 的 `options` 有 5 项、
      且 `format_func` 字典把 5 个索引映射到 `1280x1024` … `320x320`（UI `:968-970`）。
    - **差分实测（三处改动，全部点名）**：临时树里改 shift 标签 ⇒ `zrm_shift_x label should be
      'SLM Shift X (像素)', got 'SLM 平移 X'`；删掉曝光时间的落键赋值 ⇒ `zrm_wfs_exp_time should be
      driven by a widget`；`options` 砍到 4 项 ⇒ `the selectbox must offer exactly 5 MLA resolutions`。
      未改动时整文件 **22 passed**；差分树**同时带上了未提交的新测试**（第 14 项那条教训在这里直接省下一次空跑）。
    - 🔴 **严格性升级的硬证据**：在「删掉曝光 widget 落键」那棵树上，字符串 `zrm_wfs_exp_time`
      **仍然出现在 UI 源码里** ⇒ 旧断言 `assert "zrm_wfs_exp_time" in content` 会**照样通过**，
      而那个输入控件其实已经不存在 —— 这就是本类断言的假通过模式，也是第 16 项只把这个文件单列为真腐烂的原因。
    - **残留 3 处**（同文件 `test_polling_displays_updates`）未转：它断言的是轮询展示的文字，
      要驱动 `_render_calibration_polling()` 的读取循环才有行为等价物，不在本刀范围。
      ⚠️ **本行的判断错了**（同日更正，见第 20 项）：不需要驱动那条循环，**按函数取作用域**就够了 ⇒ 3 处已转完。

20. ✅ **该文件最后 3 处源码 grep 已转完（26 → 0），并删掉一段「搭完台就 `pass`」的死工装**（2026-10-25）：
    - `test_polling_displays_updates` 原 68 行：**约 45 行 mock 脚手架**（临时 JSON、`st.empty` 三连
      `side_effect`、`patch time.sleep` / `Path.exists` / `builtins.open`）跑完之后里面只有一句 `pass`，
      真正的断言只有其后 **3 条读源码文本**的 `in content` ⇒ 整套 mock 从未被断言引用。
    - 现在用 `_function(name)` 把 AST 断言**限定到目标函数体内**：`st.session_state.get("zrm_progress_file")`
      （UI `:777`）确有读、JSON 解码按 `load` / `loads` **两种拼法都接受**（钉行为不钉写法）、
      以及对进度条调用 `.progress(...)`（UI `:789`）。
    - **差分两条各自点名**：① 只在 `_render_calibration_polling` 内部把键名 `zrm_progress_file` 改成
      `zrm_prog` ⇒ 失败 `polling must read the progress-file path out of session state`；**同时** UI 文本里
      该字符串**仍然存在**（`:178/:182/:1210/:1241`）⇒ 旧断言会**照样通过**，这是本类假通过的直接证据。
      ② 把 `progress_bar.progress(` 改成 `update(` ⇒ 失败 `polling must update the progress bar`。
    - **全仓口径同步**（与第 16 项同一条 AST 规则）：`assert <字面量> in <读文件变量>` 由 **113 → 87**
      （26 → 25 个文件），其中本文件已**清零**。

21. ✅ **两个"任何地方都永不执行"的设备测试模块改由 `--hardware` 门控（26 测 / 57 断言复活）；另两个刻意不动，各有证据**（2026-10-25，续第 15 项那张只记了数的普查表）：
    - **为什么 blanket skip 本身是缺陷**：`pytest.mark.skip` **不可覆盖** ⇒ 连设备会话里显式要跑也跑不到；
      而 `pytest.mark.hardware` 由 `drivers/conftest.py` → `_hardware_gate.skip_marked_hardware_tests` 控制
      （没有 `--hardware` 才加 skip）。改的是**可达性**，不是 CI 结果。
    - **改的两块**：`drivers/test_tm.py`（6 测 / 11 断言）、`drivers/ccd/test_miicam.py`（20 测 / 46 断言）。
    - **差分两侧都实测**：① 默认跑 ⇒ `SKIPPED [6] … test_tm.py: need --hardware option to run` +
      `SKIPPED [20] … ccd/test_miicam.py: 同上`，共 **26 skipped**、**无一执行**（不碰设备，CI 与改前完全一致，
      只是跳过由守卫施加而非 blanket skip）；② 加 `--hardware` ⇒ `test_port_list` **真的执行并通过**
      （`1 passed, 5 deselected`）⇒ 标记不是装饰。
    - **先验 API 现时性再放行**（第 17 项教训：永久跳过的模块会静默腐烂）：`TM` 其实是别名
      （`tm/__init__.py:9` = `SerialPortFSM as TM`），构造签名 `(port, baud)` 对上 `TM("COM3")`，
      `__enter__`/`__exit__` 都在 ⇒ `with TM("COM3")` 合法；四个 helper 测点名的
      `_pack_position_xy` / `_build_frame` / `list_port` / `parse_position_data` / `validate_received_data` /
      `wait_rx` **逐个存在于现类**。`test_miicam.py` 的 `MIICamera` 导入、构造关键字、实例方法亦逐条对上。
    - ⚠️ **刻意保留 blanket skip 的两块**：
      1. `drivers/slm/test_slm_miicam.py:258` 断言 `slm.phase_range == 200`，而 `phase_range`
         **只在** `drivers/sim/optics/simulated_slm.py:73` 定义，`Santec` 上没有 ⇒ 启用即 `AttributeError`；
         且那段 `try` 只捕 `ImportError`（`:265`），掩不住它。要放行得先由设备侧确认正确属性。
      2. `drivers/test_driver.py`：实例解析在该文件里**取到 0 个设备实例**（设备经工厂/注册表而来），
         等于方法核对没做 ⇒ "未验"不等于"通过"，不放行。
    - **普查表的两处更正**：`display/test_windows.py` 共 6 行且**没有任何 `def test_`**（0 测 0 断言）
      ⇒ 它是占位文件而非"被跳过的测试"；`algorithm/test_differentiable_defocus_uniformity.py` 是**条件** `skipif`
      （缺 checkpoint）⇒ 合法。这两类都不是第 17/21 项那种掩码。
    - **守卫联动已确认**：`test_conventions.py::test_only_the_harness_or_hardware_tests_write_tracked_docs`
      的方向是"改写 tracked docs 的用例必须带 `hardware` 标记"；两块 `report_refs=0`（不碰 `TestReport` 工装）
      ⇒ 增加标记只会更满足它。实测合跑 **41 passed / 1 skipped**（既有 `ruff not on PATH` 跳过）。

22. ✅ **「死 mock 工装」普查：两条硬信号实测为 0 ⇒ 第 20 项删掉的那一处是孤例，不是类**（2026-10-25）：
    - **只取三条无歧义信号**（刻意不测"接了 mock 但从不校验"，见下）：
      ① mock 局部变量赋值后**全函数再无读取**（含属性读）⇒ **0 处**；
      ② `with patch(...)` 的 body **只有 `pass` / `...`** ⇒ **0 处**；
      ③ `@patch` 装饰器注入的参数**从未被引用** ⇒ **3 处**。
    - **结论：不落守卫、不改代码**。第 20 项那种「搭完台只留一句 `pass`」是**孤例**；
      而且 ③ 那 3 处**全是误报**，不是缺陷。
    - ⚠️ **这 3 处不得清理**：`drivers/ccd/daheng/test_constants.py:65`、`:81`、
      `drivers/ccd/test_auto_exposure.py:86` 全部是
      `@patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)` ——
      `create=True` 的用意就是**在没装厂商 SDK 的机器上凭空造出那个符号**，测试本体自然不必点它的名字。
      删掉参数会直接让这三测在缺 SDK 的环境炸掉。且它们**都有真断言**
      （`result is True` / `isinstance(result, np.ndarray)` + `max == 50` / `assert_called_once_with(...)`）⇒ 有信号。
    - **口径边界（诚实记）**：能自动判的只有"完全无人引用"这类硬惰性；而第 20 项真正的形状是
      "mock 都接上了、却没有任何一条断言引用它们"——这个**不能**自动化：一旦按"无断言"筛，
      就重新踩回第 18 项的误报（`test_failing_pull_does_not_raise_out_of_the_callback` 那类
      「不得抛出」契约本就没有断言）。所以这类判断**保留人工**，不做普查式批量处理。

23. ✅ **AGENTS 反模式表两处过期条目已更正，并把「叶子层不得绑 SDK」钉成守卫**（2026-10-25）：
    - **先记本次的测量教训**：判断 `sim.py` 有没有消费者时我用了不带 `-E` 的 `git grep`，
      BRE 下裸 `|` 是**字面字符** ⇒ 返回 0 命中，我据此差点写下「无 importer」。
      同一模式改 ERE 重跑得 3 命中，**正对照实测 BRE=0 / ERE=3** ⇒ 0 命中不等于不存在，
      必须先证明过滤器本身能命中已知存在的东西。
    - **`utils/wavefront/vi.py` 那一行是幻影**：该文件早在 `bbb3820b` 被删（-188 行 ctypes VISA），
      `git ls-files` 已无任何 `vi.py`；现存 ctypes 只在 `drivers/`（正当的 SDK 绑定）。
      而该行仍写「remove it or move it out of utils」⇒ 指着不存在的文件让后来人修。
      改为**保留不变式并机械钉住**：新守卫
      `test_conventions.py::test_the_utils_leaf_layer_never_binds_a_native_sdk`
      扫 `src/ao_shaping/utils/**` 的 `import ctypes` / `from ctypes ...`；实测基线 **0**
      ⇒ 空基线、没有 allowlist 会腐烂。**差分**：临时树里（同时带上未提交的新守卫，第 14 项的教训）
      往 `utils/wavefront/zernike_calc.py` 注入一行 `import ctypes` ⇒ 守卫失败并点名该文件；
      不注入时 `test_conventions.py` **37 passed / 1 skipped**。
    - **「Vector-beam demo `sim.py` 应搬去 `scripts/`」这条已过期，照做反而会打断导入**：
      `14b1cda`（#90/#128，见 §8.12 台账）已把 6 个库文件的 `__main__` demo 搬成独立脚本 ——
      `sim.py`→`scripts/signal_processing/demo_wavefront_sim.py`、
      `wavefront.py`→`scripts/signal_processing/demo_wavefront_reconstruction.py`。
      实测 `sim.py` 只剩库代码（`numpy` / `matplotlib.colors` / `loguru`，**无** `__main__`、
      `argparse`、`plt.show`），而它**唯一的消费者**恰是那个 demo
      （取 `VectorAdaptiveOptics` / `VectorBeamPropagator` / `calculate_stokes_rgb` /
      `create_gaussian_field`）⇒ 把模块本身搬去 `scripts/` 反而会断；
      `wavefront.py` 另有 2 个消费者（demo + `tests/ao_shaping/algorithm/test_wavefront.py`）。
    - **顺手清掉搬移留下的两处悬挂横幅**：`sim.py:187` 与 `wavefront.py:163` 各是
      **文件最后一行**的段落标题（`# ==== 6. 主程序 ====`、`# ---- 演示 ----`），其后**零行内容**
      ⇒ demo 早已搬走、标题成孤儿。删后 `py_compile` 通过、**末尾恰好一个换行**、
      demo 所需 4 个符号仍可解析，且 `tests/ao_shaping/algorithm` + 两条脚本导入守卫合跑
      **556 passed** ⇒ 行为零变化（实测，非假设）。
    - **改 `AGENTS.md` 前已确认它不被任何测试解析**：4 处命中全是 docstring 里的散文引用，
      `test_report_provenance.py` 只是把 `report/AGENTS.md` 当作**布局规范**而跳过溯源块检查。
      改后 `test_conventions.py` + 分片覆盖守卫合跑 **42 passed / 1 skipped**。
    - **门禁口径两处更正**（本轮实测得出，防止将来误报）：
      1. 「逐行 `**` 奇偶」这条自检**本身是错的**：`**强调**` 跨软换行在 CommonMark 里合法，
         且行内代码里的 glob（`utils/**`）同样含星号对。改成**按段落块**（连续非空行）统计、
         并先剥掉行内代码 ⇒ 实测 **149 个块中未配对 0 个**；旧的逐行口径会虚报 12 行，
         照着它改文反而会破坏本来正确的措辞。
      2. 双重夹注 grep 会**自指命中** `AGENTS.md:24` —— 该行正是把禁写形态**引出来当作规则**，
         与 `HEAD` 逐字相同、非本次引入 ⇒ 该检查的作用域须排除这条定义行；
         本次新增的 AGENTS 行实测 0 命中。

24. ✅ **`AGENTS.md` 的地图性引用逐条实测：5 处指向不存在的文件/包，已全部重钉；另有两件事刻意不做**（2026-10-25）：
    - **为什么单独一刀**：`AGENTS.md` 是人类与 AI 工具共用的**入口地图**，指向不存在的文件就是布置
      **幻影任务** —— 本项开头我差点去"删"一个 2026-09-20 就已不存在的 `vi.py`（第 23 项）。
    - **已重钉的 5 处**（每条都由整树 basename 搜索 + `ls` + `git log --diff-filter=D` 实测得出）：
      1. `tools/slm/slm_diagnose.py`（`:1111`、`:1159`）⇒ 真身 `tools/slm/panel/diagnose.py`；
      2. `utils/slm/pattern_helper.py`（`:1135`）⇒ 真身 `utils/wavefront/pattern_helper.py`；
      3. `tools/slm/slm_zernike_common.um_to_waves()`（`:1118`）⇒ canonical 是
         `utils/wavefront/zernike_utils.py` 的 `um_to_waves`（`:93`），`tools/zernike/common.py:36`
         只是**带注释声明的 re-export** ⇒ 旧写法不仅指错文件，还**违反本仓单一事实源**；
      4. `diff_shaping_runner.py`（`:1115`）⇒ 该 runner 已在 `fbef1926` 删除；条目里的挂起结论**原样保留**，
         只把指针改到仍然存在的 `report/slm/slm_shaping_diff/readme.md` 与回归测试
         `tests/ao_shaping/runners/test_diff_shaping_runner.py`；
      5. 结构表 `:266` 的「仍开放: `display.py`」⇒ 实测该文件已不在 `utils/image/`（搬到
         `display/image_display.py`，`a9236e9`），而同文件 `:1130` 已记为**搬完** ⇒ 两处自相矛盾，
         已改为完成态并注明旧路径无 shim、由 `test_utils_leaf_no_rendering_import.py` 钉住。
    - **顺带按 `ls` 补齐结构表**：`utils/wavefront/` 行漏列 3 个真实模块（`pattern_helper` /
      `beam_amplitude` / `fraunhofer`），`utils/slm/` 行却列了它**没有**的 `pattern_helper` ⇒ 两行重写。
    - **`:1135` 的前提也已不成立**（AST 实测）：该文件模块级只有 `__future__` / `typing` / `numpy` /
      `utils.wavefront.phase_unwrap:8` / `utils.wavefront.zernike_calc:13`，`algorithm` 全部延迟到函数内
      （`:20/:36/:92/:98/:837/:854/:871/:890/:903`）⇒ 红线**保留**，状态改记**已合规**，并点名既有守卫
      `test_utils_has_no_module_scope_import_of_a_higher_layer`（其非空性由
      `test_the_deferred_forms_still_exist_so_the_guard_is_not_vacuous` 守住，现 14 处延迟形式）。
    - ⚠️ **刻意不做之一：不把根 `AGENTS.md` 纳入 `_LIVE_DOCS`**。实测 `AGENTS.md:249` 与 `:721` 含字面
      占位符 `python -m ao_shaping.runners.xxx` ⇒ 纳入后
      `test_documented_python_dash_m_entry_points_resolve` 会因**占位符**变红，而不是因缺陷变红。
    - ⚠️ **刻意不做之二：不加「反引号路径必须全部解析」的守卫**。本轮三次普查证明该记号语言**无界**：
      公式（`(w//2, h//2)`）、调用签名（`...um_to_waves()`）、裸基名（`records.py` 实际在 `ml/hwdataset/`）、
      glob（`scripts/generate_*_report.py`）与目录 token（`utils/`）混在同一列；朴素扫描得
      **解析 5 / 未解析 45** —— 那是在量我的正则，不是量树。真要做须先定**路径语法**并复用
      `_HISTORICAL_DOCS` 的「记录 vs 指令」分法；**宁可不落，也不落一个专产误报的守卫**（§8.10 红线）。
    - ⚠️ **本轮我自己踩到的 4 个测量错误**（都被正对照当场抓住，记下防再犯）：
      ① `git grep` 不带 `-E` 时裸 `|` 是字面字符 ⇒ 「无消费者」是假结论（同模式 BRE 0 / ERE 3）；
      ② 逐行 `**` 奇偶会误报**合法的跨行强调**与代码内 glob（12 处假阳）⇒ 改成按**段落块**统计才准
      （真实缺陷 0 处，见第 23 项）；③ 双引号内的反引号被 bash **命令替换**，得到的计数 `1` 是垃圾值；
      ④ 分类器把非 `.py` 的 token 削成扩展名（`README.md` → `md`）⇒ 真实存在的文件被判 DEAD。
      规则：**任何 0 命中 / N 命中的普查结论，必须用同一模式跑一条正对照**，并写明方言（BRE/ERE）与引号方式。
    - **改动自身的验证**：9 个新指针全部在树中解析 OK；编辑过的行**列数与原表头一致**（结构表 4 个 `|`、
      反模式表 3 个 `|`）；表格行奇数强调 **0 处**；术语门禁 `合计: 0 个文件, 0 行`；
      `test_conventions.py` + 分片覆盖守卫合跑 **42 passed / 1 skipped**。

25. ✅ **一条"已弃用"标签在动手前被实测推翻：`idealized_far_field` 是活代码，不是死代码**（2026-10-25）：
    - **为什么单独立项**：`:174` 那行写着「`idealized_far_field` 已弃用 (AGANTS anti-pattern)」，我据此**准备删它**；
      普查在动手前把它推翻 ⇒ 记录这条，是因为**没有代码内标记支撑的状态标签会直接导致错误动作**。
    - **推翻它的实测证据**：① 该模块 47 行里 `deprecat\|弃用\|已废弃\|no longer used\|legacy` 命中 **0**，
      docstring 只说明"是归一化 FFT 图、不是标定过的 CCD 预测"（划边界，不是弃用声明）；
      ② `tests/ao_shaping/utils/test_gs_debug_optics.py:18 test_far_field_uses_off_center_pupil`
      在 `:20`、`:24` **实际调用并断言**它 ⇒ 删掉就是打断一个活测试；
      ③ `scripts/slm/generate_gs_refine_fwdmodel_report.py:280` 的报告正文把它当**对照口径**引用。
    - **原括注两处皆误**：`AGANTS` 是 `AGENTS` 的拼写错误；且 `AGENTS.md` 全文**并无**关于该符号的反模式条目
      （只在结构表把 `gs_debug_optics` 作为模块名列出）⇒ 该行引用的是一个**不存在的条目**。
    - **单一事实源核查（结论：不违例）**：canonical 远场在 `utils/wavefront/fraunhofer.py`
      （`focal_field` / `focal_intensity` / `interpolated_pupil_*`，torch 可微，7+ 个 src/scripts 模块引用），
      而 `idealized_far_field` 是 **numpy + 峰值归一化 + 偏心圆瞳**的查看实现 ⇒ 栈与语义都不同，
      不属于"重复实现同一数学"那一类（第 34 项 / Zernike 收敛那类才是）。
    - **处置**：`:174` 只改**理由**、保留 **保留** 裁决；"要不要真的弃用 / 挪成报告专用 helper"**留给设备侧决定**——
      它涉及"该 helper 的语义是否仍要作为对照口径"，无设备无从验证等价性 ⇒ 本轮不做（避免把口径判断当重构做）。
    - **顺带一条普查纪律**（与第 23、24 项同源）：判断"死代码"须同时看 **import 面**与**断言面**——
      只看 `git grep 名字` 的命中数会漏掉"仅被测试引用"这一类，而这一类删了就是**削测试**。

    - **本轮新增一条自检口径**：表格行的列数须与**该块表头**一致（**此基准已被第 26 项推翻**：真值是**分隔行**，表头本身可能就是坏的那一行；下面括号里的 `10 vs 5` / `6 vs 5` 就是按当时那条表头基准数的），且只数**未转义**的 `|`
      （GFM 里代码段内的裸 `|` 照样切列）。按此扫 `TODO.md` 的 31 个表块：本轮**引入的破坏 = 0**
      —— 先误引入 1 处（在表格单元的代码段里写了 `a|b|c`），正是被这条自检当场抓出并修掉的。
      另有 **2 处既存破坏**（`:304` 10 列 vs 表头 5；§8.12 表一行 6 vs 5，`HEAD` 里即在）：
      **刻意不顺手改** —— 属既存缺陷、非本轮动因，且两表均为带日期的历史记录，改动须另立条目；
      发现先登记于此，避免下一轮把它们误算成新引入的破坏。
      **后续兑现（2026-10-25，第 26 项）**：这两处就是第 26 项「手维护文档 5 处」里的 ②（`:304` 两行挤在同一物理行 ⇒ 拆回两行）与 ③（§8.12 表的留档说明占了第五格 ⇒ 并回第四格）⇒ 上面「改动须另立条目」的门槛已满足。
      **原判保留**：不在无关轮次里顺手改是对的，这里只补去向，免得下一位读者以为这两行还欠着。

26. ✅ **markdown 表格单元里的裸 `|` 会切列：手维护文档 5 处已修，生成报告 24 处登记待修（→ 已由第 27 项清零）**（2026-10-25）：
    - **普查口径（基准是分隔行，不是表头）**：以 `|---|` 分隔行的列数为真值。理由实测：
      `report/shape_trials/hw_defocus_zernike/` 三份报告里坏的是**表头本身**（`max|c| (rad)` 把表头
      切成 14 格，而分隔行与数据行一致为 12 格）—— 拿表头当基准会把**责任行**判反。
      只报"写入格数 > 声明格数"的行；**少于**声明是合法的（GFM 补齐短行、行尾 `|` 可省略），刻意不报。
    - **手维护文档 5 处，全部修掉，三种不同成因**：
      ① 行内代码里的裸 `|` **照样切列**：`docs/architecture.md` 的 `|E|`、`docs/dev/todo_archive.md` 的
        `[spgd|ga]`、`scripts/README.md` 的 `--protocol(file|objective)` ⇒ 转义成 `\|`；
      ② `TODO.md` 里**两行挤在同一物理行**（`… | ✅ || ~~R-57~~ | …`，缺换行）⇒ 拆回两行；
      ③ `TODO.md` 四列表把留档说明放成了**第五格** ⇒ 并回第四格（不是转义问题）。
      ⇒ **计数是探测器，不是诊断**：同一个"格数超出"症状至少三种成因，每处都必须读原文再决定改法。
    - **生成报告 24 处 / 11 份刻意不改 `.md`**：它们由 `scripts/_common/provenance.py::REPORTS` 映射的
      **8 个生成脚本**产出，手改会在下次重生成时被覆盖。正确修法在**生成侧**，且实测**全仓没有转义约定**
      （`replace("|"` 零命中；仅有的两个 `_row*` 定义是 CSV 取行与排序键，不是 markdown 发射器；
      8 个脚本一律裸 `f"| {a} | {b} |"`）⇒ 需先立一个共享的行发射/转义 helper 再逐脚本接线；而
      `hw_defocus_zernike` 系列与 `test_miicam_auto_exposure_report.py`（`-m hardware`）重生成要碰设备数据
      ⇒ **本轮不做**，登记为后续单元（先修 helper + 接线，数据可离线的那批先重生成）。
    - **新增守卫** `test_markdown_table_rows_never_write_more_cells_than_declared`
      （`tests/ao_shaping/test_conventions.py`）：覆盖全部 260 个 `.md` / **1487 个表块**，
      围栏代码块先清空（仓库在围栏里写表格样例，不清就会报渲染中根本不存在的缺陷）。
      **两个桶**：手维护文件空基线（必须 0）；生成报告按 **24** 计数棘轮（只降不升）。
      桶的划分**复用 `provenance.REPORTS`**（生成 vs `script=HANDWRITTEN`），不按目录猜。
    - **同一症状的第三种成因（本轮新发现，不是转义问题）**：`report/models_analysis/`
      的表**分隔行短一列**（表头与数据行都写 8 格，`|---|` 只声明 7 格）—— 这不是错位，
      **GFM 根本不把整块当表格**（分隔行与表头列数必须相等）。修法是**给分隔行补一列**，
      转义任何单元都是错的 ⇒ 单靠"单元里有没有裸 `|`"判因由会走偏，必须读分隔行。
    - **反方向漏检 ⇒ 表头改双向校验**：`report/slm_pib_bench/` 是 9 格表头配 **10 格**分隔行，
      旧规则只报"行数 > 分隔行"，结构上**看不见**这种形状。现规则：表头 `!=` 分隔行即报；
      数据行只在 `>` 时报（GFM 补齐短行、行尾 `|` 可省，**少于**是合法的，不能报）。
    - **普查工具自身的两个坑（都值一节）**：
      ① 判"生成 vs 手维护"最初按目录 `report/` 一刀切 ⇒ 把 `script=HANDWRITTEN` 的
        **手维护报告**（`experiment_summary_20261022.md`）永久豁免了，而它确有切列缺陷。
        ⇒ 分类必须复用仓库已有的单一事实源 `provenance.REPORTS`。
      ② 解析扁平字面量映射时用**固定 400 字符窗口**取 `script=` 字段，窗口**溢到下一条**
        （`models_report/report.md` 是生成的，却因邻居 `reward_analysis.md` 的
        `script=HANDWRITTEN` 被判成手维护，连带 4 处误进"必须为 0"的桶）⇒ 改为从键位置起
        **搜该条目自己的第一个** `script=` 字段。判据：分类数须等于条目数（实测 85 + 54 = 139）。
    - **守卫自身的三条教训（比修 5 处更值钱）**：
      ① 首版把 `_unfenced_md_lines` 的三元写反（`"" if open_fence is None else line`）⇒ 把**围栏外**的
        真表格全部清空，扫到 **0 个表块**。若反空转断言写成 `blocks > 0`，这个守卫会**永远空过还显示通过**；
        因为按实测 609 留了 `blocks > 400`，当场判红。⇒ **计数型守卫要断言"扫到了量级"，不是"扫到了东西"**。
      ② 差异证明用**对 `HEAD` 内容跑同一段代码**（`git archive HEAD | tar -x -C /tmp/head_tree` 后 exec 守卫
        helper）：工作树 **0**，`HEAD` 树 **5**（正是修掉的那 5 处）。比人工注入更硬——证明守卫抓的是
        **真实存在过的缺陷类**，不是我为它造的样本。
      ③ 棘轮基线的注释必须写清算术（23 + 新查出的 1，且与旧规则的 24 **不可比**）；
        写成"24, down from 24"就是一条**看起来自洽但为假**的注释——正是本轮在别处清理的那类东西。

27. ✅ **表格切列的生成侧接线（三段做完）：共享发射器 + 两处分隔行错列 + 手写行字面量 + 探针渲染点，棘轮 24→19→7→0**（2026-10-25）：
    - **共享发射器**：`scripts/_common/__init__.py` 的 `markdown_table`（全仓 87 处调用）此前用 `str(cell)`
      裸拼接，任何含 `|` 的单元都自我切列。新增 `escape_md_cell`（**幂等**：已有的 `\|` 不动；无 `|` 时短路），
      表头与单元都过它。`generate_hw_defocus_zernike_report.py` 的 14 处表格因此**已在源头**修好
      （`max|c| (rad)` 现发射为 `max\|c\| (rad)`），但它**已落盘**的三份报告文本仍需同步。
    - **两处不是转义，而是分隔行列数错**（第 26 项预告的"第三种成因"）：`scripts/rl/generate_models_report.py:703`
      给 8 列表头发射 7 格分隔行（连带 4 行判超），`scripts/slm/generate_pib_bench_report.py:462` 给 9 列表头
      发射 10 格分隔行（连带 1 行判超）。**只改分隔行那一行**即消 5 处；产物同步改（实测两表现在
      header = delimiter = data row）。⇒ 棘轮从 24 掉到 19 的步长不是"一行一处"，注释里写清算术。
    - **发射路径普查（决定剩下 19 处怎么修）**：8 个受影响生成脚本里只有 `generate_hw_defocus_zernike_report.py`(14)
      与 `generate_zernike_amp_report.py`(7) 用 `markdown_table`；其余 6 个（inverse_design / oopao_vs_numpy /
      asm_vs_fraunhofer / farfield_sim / pib_bench / models）一律**手写行字符串** ⇒ 逐个转义其字面量。
      另：miicam 报告的 `.md` 由 `scripts/ccd/miicam_auto_exposure_probe.py:868` 写（`tests/.../test_miicam_auto_exposure_report.py`
      只是驱动硬件），接线点在**探针脚本**里，不在测试里。
    - **一处产物内容是手加的（已漂移）**：`report/zernike_phase2amp/zernike_amp_report/report.md:697` 的 piston 行
      在**当前**生成脚本里没有任何发射点（`可学习参数中排除` 全仓零命中；脚本只在 `:699` 发射同物理的**项目符号**，
      该符号在产物 `:40`）。⇒ **重生成会删掉这一整行**。这一处只能改产物文本，同时登记：手加内容已与生成脚本漂移。
    - **验收口径（本轮不重生成任何报告）**：实验记录重生成会动数字，属设备侧判断 ⇒ 只跑**重扫**（守卫计数）
      与 helper 单测；`markdown_table` 的既有 characterization 测试用无 `|` 的单元，故契约兼容、逐字不变。
    - **第二阶段（本轮已做）：棘轮 19 → 7**。清掉 12 处：8 处是**在生成脚本里转义字面量**
      （inverse_design 3、oopao_vs_numpy 1、asm_vs_fraunhofer 3、zernike_farfield_sim 1），这样
      **将来重生成**出来的文本与已改好的产物一致；另 3 处是 hw_defocus 已落盘产物（其发射器
      `markdown_table` 已在源头修好），1 处是 zernike_amp 手加行（无发射器，只能改产物文本）。
      实测产物与生成脚本**逐段一致**：生成源码里是 `\|\|` 一层 Python 转义、产物里是 `\|`，
      **差一层**——我第一次的"一致性校验"忘了折这一层而全报 MISMATCH，错在**校验脚本**不在文件。
    - **本轮自己引入、又被自己抓到的错（比修 12 处更值钱）**：批量转义用了 `splitlines(keepends=True)`，
      行尾 `\n` 让**收尾的结构 `|`** 看起来"右边不是空格"⇒ 被判成内容并转义。后果分两种：表头立刻判红
      （表头必须与分隔行**恰好相等**），数据行只是**少写一格**——GFM 补齐短行、守卫**故意不报**
      ⇒ 同批 12 行里 11 行是**静默损坏**。两条结论：① 机械改写先**剥离行尾终止符**再判定；
      ② "少写格数合法"这条豁免让**单靠 arity 规则看不见丢掉收尾竖线的行** ⇒ 守卫补一条**零容忍**的
      外层竖线检查（行首/行尾以 `\|` 起止即报；实测全仓 0 处，故空基线）。
    - **撤销纪律**：那 8 个产物文件在本轮之前未被改过 ⇒ 坏改动用 `git checkout -- <8 files>` 干净回退后
      重做，不 stack 补丁、不用 stash。前提：动手前先确认这些文件在当前改动集合里的状态。
    - **第三阶段（本轮收尾）：棘轮 7 → 0，两类 arity 全部归零**。miicam 的 7 处接在**渲染点**：
      `_write_report` 的判定表只对**单元**套 `escape_md_cell`，**不**在造文本的 `record()`
      （`:792`/`:319`）上套 —— 同一 `detail` 会原样进 `results.json`，往数据产物里塞反斜杠就是污染
      JSON ⇒ **转义属渲染关注点，不属数据关注点**（新增第三条单测专门锁住"渲染不改 `results`"）。
    - **为什么不必上相机也能验**：探针驱动硬件，但 `_write_report(path, results, config)` 是**纯
      dict→markdown** 函数、模块级只 import stdlib+numpy+loguru ⇒ 三个**无设备**单测（单元转义 /
      表块各行 arity 相等 / 渲染后 `results` 不被改）即足以证明接线正确。落盘的 7 行按**同一条规则**
      手改，且**逐行断言** `写入格数 == 分隔行声明`（就是上一段那个"少写一格看不见"的坑的防线）。
    - **诚实登记没做的事**：**没有**在相机上重生成 miicam 报告。产物与生成器一致由"同一转义规则 +
      纯函数单测"保证，不是由一次真跑保证；真跑一次的验收点在设备侧。

28. ✅ **死符号普查：76 条线索，本轮一条不删 —— 普查看不见「入口组合」与「约定面」**（2026-10-25）：
    - **口径**：`src/**/*.py` 的 439 个模块（跳过 `__init__.py` 再导出层）中的**模块级**符号，同时满足
      ① 本模块内零自用；② 全仓 1230 个 `.py` 里零外部引用（`ast.Name` / `ast.Attribute` / `ast.alias` 三面）；
      ③ 不在该模块 `__all__`；④ 不由注册型装饰器（`click.command|group` / `fixture` / `abstractmethod` /
      `property` / `dataclass` …）绑定 ⇒ **76 条**。再要求字符串字面量与 md/yml/json/ipynb 全文里也**不出现**
      ⇒ **49 条**。
    - **普查工具自身的两个坑（都已修，记下来免得重踩）**：
      ① 首版 `self_use = refs_in(...) - 1` **过减**：`FunctionDef`/`ClassDef` 的名字是**字符串字段**、
        不生成 `ast.Name` 节点，只有 `Assign` 目标才生成 ⇒ 只在自身模块被调用一次的函数会被误判成
        「零自用」。现只对 `const` 减 1。
      ② 首版按「候选 × 文件 × 节点」三重遍历 ⇒ **15 分钟超时**。改为**先一遍 `ast.walk` 建全局名字索引**、
        过滤后再对存活者做字符串扫描。⇒ 普查脚本的**候选数与耗时都必须可复现**。
    - **为什么本轮不删：清单头部就不是该删的东西**（逐条实测，非推断）：
      - `drivers/device_base.py DeviceBusyError`：`AGENTS.md` 驱动错误层级的一部分，属**公开 API 面**；
      - `drivers/slm/santec/_slm_win.py` 的 `FLAGS_COLOR_*` / `SLM_BS` / `SLM_ER` / `SHORT` / `UCHAR`：
        厂商 SDK 的**镜像常量**（驱动协议面，命中的是厂商文档而非本仓调用）；
      - `runners/slm/offset_runner.py search_offset_by_defocus`：同文件
        `search_offset_by_defocus_with_optimization` 在 `:519` 被该模块自己的入口调用、`:494` 给出
        `python -m ao_shaping.runners.slm.offset_runner` 用法，且 `AGENTS.md` 写明该 runner **未注册、
        计划迁到 `tools/slm/`** ⇒ 正确动作是**迁移**，不是删除；
      - `runners/slm/gsnet_runner.py _FreeformSquareDisplay`（134 行私有类，标识符面与字符串面**全仓零命中**）：
        最强的一条真候选，但该文件落在 `pytest.yml` 的 **advisory 分片 `runners-gui-misc`**，其
        `slm-gsnet --help` golden 正由 **wt-193** 修 ⇒ 在途改动，**不顺手删**；
      - `src/ml/train.py train_model`（232 行）：全仓仅剩**文档串**提到它，活路径是 `train()` /
        `_train_phase_mode()` / `_train_coeffs_mode()` ⇒ 确属遗留，但它是 ML 训练入口，**归 ML 侧定夺**；
      - `gui/r50/r50_channel_select.py` 的 `render_channel_selector` / `show_feedback` /
        `ensure_session_defaults` 与 `gui/r50/r50_joint.py` 的 4 个 `_jc_*`：Streamlit 的 UI 靠**入口组合**
        驱动，且 `_jc_*` 前缀像是**在途重构**的中间态 ⇒ 要 UI 侧确认。
    - **可复用结论**：AST 普查只覆盖 **import 面 / 断言面 / 字符串面**，看不见三样东西——
      ① `python -m` 与 `streamlit run` 的**入口组合**；② 写在 `AGENTS.md` 与厂商文档里的**约定面**；
      ③ **迁移与重构意图**。故 76 条是**待业主确认的线索清单**，不是删除清单。本轮**未改任何代码**；
      真要动某一条，按「先立失败测试 → 删 → 全量分片复验」单独开条目。

29. ✅ **CLI 清单与代码对齐：命令数 20→19、幽灵命令 `spgd-square`、`main.py` 自述的三个不存在命令**（2026-10-25）：
    - **四源基准（现已一致，自检脚本按差集断言相等）**：`main.py` 里 `cli.add_command(...)` 的名字 = **19**；
      `tests/ao_shaping/runners/test_cli_contract_freeze.py::EXPECTED_COMMANDS` = **19**；
      `tests/ao_shaping/runners/cli_help_golden.json` 的顶层 key = **19**；`main.py` docstring 命令表 = **19**。
    - **修掉的六处偏差**：
      ① `AGENTS.md` 写"main.py 注册 20 个命令"，实为 **19**（多的那 1 就是 ② 的幽灵）；
      ② `spgd-square` 被当**顶层命令**写满 8 处（CLI 树 / bash 示例 / `WHERE TO LOOK` / 策略↔runner
        对照表 / 三层架构图 / 两条"未注册 Runner"行 / Entry Points 注释）—— 它其实是 `slm-pib`
        **命令组**的子命令（`run.add_command(square, name="square")`），正确写法 `slm-pib square`；
      ③ `main.py` **自己的 docstring** 列了三个**不存在**的命令 `gs` / `gs-square` / `diff-beam`
        （`add_command` 与 `EXPECTED_COMMANDS` 均零命中），又漏列 9 个真命令；Examples 里的
        `--debug gs …` 同样不可跑 ⇒ 已改为 `slm-pib spgd`。照 `--help` 抄的人第一次就会失败；
      ④ `hadamard_matrix_runner` 在"未注册"注里仍标「正在注册」⇒ 实为**已注册**（已进入
        `EXPECTED_COMMANDS` 与 help golden），移出该行；`slm_offset_runner` 仍未注册，保留；
      ⑤ `ao-train = ml.train:cli`（`pyproject.toml [project.scripts]` 里**唯一**随包安装的入口）
        原先在文档中**完全没写**，已补进 Entry Points；
      ⑥ 命令树补全 19 条并标出**真实模块与 attr**（读自 `runners/__init__.py` 的 `_LAZY_RUNNERS` /
        `_IMPORT_ATTRS`，不猜）：`zernike-matrix` 与 `closed-loop` 同模块、`ga-`/`greedy-zernike` 同模块、
        `alt-`/`full-voltage` 同模块、`dm-`/`hadamard-matrix` 同模块 ⇒ **一个模块导出多个命令时必须显式
        给 attr**，否则落到默认 attr `run` 会取回**错误的命令**。
    - **一次自我纠正（比修六处更值钱）**：我第一轮把 `combined` / `hadamard-matrix` 写成 `main.py:113` /
      `:120` 这样的**行号锚**，而**同一轮**扩写 docstring 让注册块整体下移 7 行 ⇒ 引用**当场失效**。
      已全部换成**符号锚**（如 `cli.add_command(combined_run, name="combined")`），并加两条自检：
      文档引用的锚必须在 `main.py` 里**逐字存在**；`main.py:<数字>` 型引用必须**清零**。
      理由与守卫基线同一条：**行号会腐，符号不会**（守卫只用计数、不用 `path:line` 白名单）。
    - **验收**：容器 **725 passed / 1 skipped**；唯一红是
      `test_cli_contract_freeze.py::test_help_text_matches_golden` 里 `slm-gsnet` 的 usage 形状
      （`[COMMAND]` 可选），它**在干净 `HEAD` 上同样复现**，且两份 help golden 都**不含组级帮助**
      （`cli_help_golden.json` 无裸 `main` key；`_cli_help_golden.json` 零个裸 key；我的 docstring
      文本在 golden 值中零命中）⇒ 属 wt-193 在 `runners-gui-misc` **advisory** 分片的既存红，不动。
    - **已兑现（同日下一轮）：四源核对升级为常驻守卫**，落在 `tests/ao_shaping/test_conventions.py`：
      `test_cli_command_inventory_agrees_across_its_five_sources` 与
      `test_agents_md_never_pins_main_py_by_line_number`。口径最终是**五源**（`main.py` 的
      `cli.add_command`、`EXPECTED_COMMANDS`、help golden 的**裸 key**、`main.py` docstring 命令表、
      `AGENTS.md` 命令树），实测各 **19** 且取差集相等，以注册面为基准。
      当时担心的子命令行歧义这样解决：命令树**只取分支符后的第一个 token**，子命令写在同行括号内
      （`slm-pib … (click 组: spgd / heuristic / square)`）因而不会被误当顶层命令。
    - **守卫的两条自身防线（都有实测）**：① `_MIN_CLI_COMMANDS = 8` 的**反空转下限**——把
      `main.py` docstring 的段名改成 `Sub-commands` 让解析器返回空集后，守卫报
      `parsed to almost nothing: ['main.py docstring']` 而不是"空集 == 空集"的**假通过**；
      ② `_docstring_commands` 里的 `seen_entry` 标志：docstring 的段名会**折行**（续行在 0 缩进），
      没有这个标志解析器会直接返回空集 ⇒ 同样落入假通过。③ 另两处注入分别被点名：
      把 `AGENTS.md` 的数字改成 20 ⇒ `states 20 … registers 19`；补一条 `main.py:106` ⇒
      `1 absolute main.py line citation(s)`。**未触发过的守卫等于没有守卫**。

30. ✅ **把 `AGENTS.md` 纳入 `python -m` 入口守卫：根文档此前是唯一不被核查的活文档**（2026-10-25）：
    - **缺口**：`_LIVE_DOCS` 覆盖 `README.md` + `src/**.md` + `scripts/*.md` + `docs/*.md`，**唯独不含根
      `AGENTS.md`** —— 而它是本仓阅读量最大的操作文档。故第 23、24、29 项里它的失效引用**全靠人工发现**，
      没有任何守卫兜底。本轮补上（不是重写第 23/24 项，而是让它不再复发）。
    - **三处必须一起放宽，缺一处就是假通过**：
      ① `_LIVE_DOCS` 加入根 `AGENTS.md`；
      ② `_module_inventory()` 原来只走 `src/ao_shaping`，而 `src/ml` 是**独立顶层包**
        （`python -m ml.hwdataset.inspect`）⇒ 只放宽①会让每个**合法**的 `ml` 入口被判"无此模块"；
      ③ `_PY_M` / `_MD_M` 的前缀由 `ao_shaping[\w.]*` 放宽为 `(?:ao_shaping|ml)[\w.]*`。
      顺序上 ② 先于 ③：正则与清单覆盖面不一致时，守卫会**误报**而非漏报。
    - **不扩白名单，改文档**：`AGENTS.md` 两处 `python -m ao_shaping.runners.xxx` 本就**跑不通**
      （没有名为 `xxx` 的模块）。守卫已豁免 `...<name>` 模板形式，故把占位符改成 `<name>` 与既有约定一致，
      **而不是**把 `ao_shaping.runners.xxx` 塞进 `_ALLOWED_UNRESOLVED`（那张表的语义是"确实坏的"）。
    - **放宽前先算代价**（先测后改）：对全部活文档 + `.py` 文档串跑一遍解析，**未解析项恰为 1**
      （`ao_shaping.runners`，来自 `utils/io/cli_helpers.py` 把 `python -m ao_shaping.runners...` 当
      "某个 runner"的通配说法，已在 `_ALLOWED_UNRESOLVED` 里有理由）⇒ 基线**不必增长**，
      `set(problems) == set(_ALLOWED_UNRESOLVED)` 仍成立。顺带暴露：`AGENTS.md` 里 4 条 `python -m ml.*`
      与 `README.md` 里若干条**一直没人核查**。
    - **验收**：容器 `test_conventions.py` **40 passed / 1 skipped**（含两条入口守卫与基线双向核查）。
      **咬合证明**（注入到 scratch worktree，且先断言放宽真在副本里，否则"通过"只说明守卫不存在）：
      往 `AGENTS.md` 注入 `python -m ml.probe_does_not_exist` 与 `ao_shaping.runners.probe_does_not_exist`
      ⇒ `documented python -m targets that do not run: {...(no such module)...}` 且基线守卫并报
      `newly broken: [...]`，**2 failed**。两条新覆盖轴各自都能被抓住。
    - **本轮没做**（避免把口径判断当重构做）：`_HISTORICAL_DOCS` 仍把 `docs/dev/`、`report/` 等排除在外，
      这是**有意**的（历史文档里的旧路径是记录，不是错误指引）；要不要给历史文档单独一条"只警告不失败"的
      检查，属口径决定，另议。

31. ✅ **CHANGELOG 的提交引用有个单字符笔误；补「引用的哈希必须可解析」守卫**（2026-10-25）：
    - **缺陷**：`CHANGELOG.md:18` 的去重/重构批次列了 6 个哈希，其中 `fdc0696c` 在本仓对象库里
      **不存在**，其余 5 个都能解析。该行文字列的六个条目（#87/#154/#153/#152/#91/#130）里
      **只有 #154 没有任何哈希覆盖**；而按 `fdc0` 消歧只给出两个对象——一个 tree，和提交
      `fdc0696f`「lift the T1/T2 composite into a shared _common.shaping_score (#154)」
      ⇒ 末位 `c`→`f` 的一字笔误。**引用文字与批次条目互相印证才认得出 intended commit**；已改（1 文件 1 行）。
    - **为何只在 CHANGELOG.md 上守卫**：同一条法扫全仓 `*.md` 得 185 个候选、**16 个假阳性**，
      分四类——日期、设备序列号（`22030102`）、上游 OOPAO 版本号（OOPAO 是独立 editable 安装，
      `oopao = { path = "libs/OOPAO" }`，其提交本就不在本仓对象库）、以及测试记录里的期望校验和前缀。
      全仓规则要把这四类豁免硬编码，那就变成**测自己的正则**而不是测仓库。
    - **按上下文排除，不按数字形状**：CHANGELOG 里唯一的非提交哈希串都是 **markdown 链接标签**
      （方括号内是反引号包的 8 位数字，圆括号内是该报告的日期文件名），指文件不指提交。
      ⚠️ **不得**用「全是数字 ⇒ 日期」这种形状豁免：本仓历史里 **37 个提交的短哈希全是数字**
      （`91209176` 即其一且可解析），形状豁免会放走真笔误。
    - **反空转下限** `_MIN_CHANGELOG_CITATIONS = 50`（实测 71）：解析不足就判失败。理由不是洁癖——
      markdown 表格守卫差点带着「扫了 0 个表格块」上线，正是这类计数守卫最典型的失效方式。
    - **守卫自身的诊断缺陷（比原缺陷更值得记）**：首版用 `git rev-parse --disambiguate <tok>`
      （两个 argv 参数）判「前缀不唯一」。git **不接受**该写法（须 `--disambiguate=<tok>`），
      于是 rev-parse 把未知选项**连同该 token 本身**回显进 stdout，我的 `startswith` 过滤让
      笔误哈希**匹配了自己**，把「对象不存在」误报成「指向非提交对象」——**一个专职诊断的守卫
      给出了错误诊断**，会把下一个人引向「把哈希写长」而不是「哈希不存在」。
      现改为**单次** `git cat-file -t`，歧义短哈希由 **stderr** 区分
      （`error: short object ID … is ambiguous`），三支诊断各对应一种修法。
      ⚠️ 顺带记一条假线索：容器 git 2.47.3 与宿主 2.43.0 的「行为差异」是我用了两种命令形式造成的，
      不是版本差异——**先核对命令是否同一条，再谈版本**。
    - **咬合证明（四条，每条先断言改动已落地，否则「通过」只说明守卫没跑）**：恢复笔误 →
      `no such object`；把引用换成某 tree 的 40 位哈希 → `names a tree, not a commit`；
      换成真实歧义前缀 `224ab5d` → `prefix not unique`；把 CHANGELOG 截到 4 行 → `below the floor`。
      真实树 `test_conventions.py` **41 passed / 1 skipped**，ruff 全绿，引用普查 **71 引用 / 0 失配**。

32. ✅ **`path:行号` 引用普查：结论是「不建守卫」，并记下一次连错六次的仪表化过程**（2026-10-25）：
    - **动机**：本会话我自己两次被行号引用坑到（写 `main.py:106`/`:113` 后改 docstring 就让引用失效，
      第 29 项因此禁了 `main.py` 的行号钉）。于是想给「引用的行号超过文件当前长度」这条**可机械判定**的
      子集建守卫。普查后**放弃**。
    - **精确形人口太少**：只接受根相对精确形（`src/`、`tests/`、`scripts/`、`docs/`… 直接命中
      `git ls-files`），扫 31 份活文档得 **6 条**：4 条行号在范围内、**0 条越界**、2 条路径已迁走
      （`utils/wfs_utils.py`、`utils/pattern_helper.py`）。下限时限只能设 4，一改排版就碎；
      按本仓既有纪律（`_ALLOWED_UNRESOLVED` 注释：**没有待豁免的真实声明就别加豁免**）不值得建。
    - **那 2 条也不是缺陷**：`docs/issues_report.md` 开头就声明「**本报告已严重过时**…§1~§3 引用的
      行号都需要重新扫描，实施前请勿直接引用」，且**同一个表格格里已自注新路径**
      （「2026-10-01：该文件现已迁至 `utils/wavefront/wfs_utils.py`，行号已漂移」）。
      快照 + 就地更正 ⇒ 不动它，也不因此把它加进 `_HISTORICAL_DOCS`：`_is_historical()` 的唯一消费者是
      `_unresolved()`（`python -m` 守卫），加豁免**今天不改变任何判定**，只会遮蔽将来的真实声明。
    - **其余形态不可判定**：宽松语法扫到 235 条，绝大多数是**裸文件名或相对目录形**
      （如 `drivers/INTERFACE_DOCS.md` 写 `ccd/base.py`、AGENTS.md 先给全路径后再写 `transforms.py:440`）。
      判它们要么靠 basename 唯一性猜（本仓 `base.py`/`common.py`/`driver.py` 多处同名），要么靠
      「相对本文档目录」这条**仓库里从未写明**的约定。猜测型守卫产出假阳性，**比没有守卫更糟**。
    - **仪表化连错六次（本条的主要价值在这里）**——每一次都让普查数字一度失真，全部靠
      「比例不合理 ⇒ 先怀疑仪表」拦下：
      ① `st=*** MISSING ***` 被 bash 解析成「赋值 + 命令」，打印 17 行 shell 错误却被读成「17 个缺失」；
      ② 探针里 `subprocess.run` **漏 `cwd=`**，容器 git 全部 `rc=128 not a git repository`；
      ③ `git rev-parse --disambiguate <tok>` 拆成两个 argv（git 要 `--disambiguate=<tok>`），
        rev-parse 把未知选项**与 token 本身**回显进 stdout，`startswith` 过滤让笔误哈希匹配自己 →
        **已在上一项上线的守卫给出了错误诊断**（已改为单次 `git cat-file -t`，歧义看 stderr）；
      ④ **通过分支不计数**：解析成功、行号在范围内的引用不 increment，于是「共 235 条」里
        一类引用整体消失（`root-path` 桶根本不存在）——计数守卫必须同时计通过项；
      ⑤ `git ls-files` 对非 ASCII 路径**加引号转义**，2951 条里 9 条不是路径（须 `-z`）；
      ⑥ `next((p for p in PREFIXES if (p + rel) in tset), None)` 返回**前缀**而非 `p + rel`，
        于是 `read_text` 打在目录上抛 `IsADirectoryError`。
      最开始的 **211/279「缺陷」**就是 ①④⑤⑥ 合成出来的假结果；若我当时直接照单修文档，
      就会把一批**正确的简写引用**改错——这正是本仓反复记录的「先证明仪表可信，再相信它的数字」。
    - **仍开放的这条**（不做不等于没问题）：行内容漂移（行号存在但指向的代码已改写）无法机械判定，
      它比「行号越界」更常见也更隐蔽；目前只有 `main.py` 一处的**禁钉行号**规则（第 29 项）在防它。

33. ✅ **让生成脚本真正拥有 zernike_amp 报告的手写附录；并记明「生成报告手写段落」普查为何不可用**（2026-10-25）：
    - **缺陷**：`report/zernike_phase2amp/zernike_amp_report/report.md` 由
      `scripts/ml/generate_zernike_amp_report.py` **整文件覆写**（写入点在 `main()` 末尾），
      但产物里有 **75 行脚本从不产出的手写内容**：`## 附录 A: 名词解释 (Glossary)`
      （产物 679–703，2078 字节 / 21 行表格）与 `### U-Net 对比模型`（43–92，50 行）。
      `git grep` 证实这些字句**不在任何被跟踪源文件里** ⇒ 唯一副本存于一个会被机器覆写的文件。
      第 27 项当时只记下其中 **piston 一行**，实际是**两节**。
    - **只修可机械验证的那一半**：附录位于产物**末尾**，拼接点唯一 ⇒ 移进生成脚本成模块级 raw
      字符串常量 `GLOSSARY_APPENDIX`，写入点改为 `write_text(text + GLOSSARY_APPENDIX)`，且放在
      「synthesis 为空则删 §10 并重编号」之后，不与 `## 11.`/`## 12.` 的替换相互作用。
      ⚠️ **必须 raw**：表格带着第 31 项留下的 `\|` 单元格转义与 LaTeX 花括号 `e^{iφ₀}`，
      塞进 `build_report` 那个巨型 f-string 会被当作格式域吃掉。产物**一字未改**（它本来就对），
      脚本 +37 行；字节级用 AST 取常量证 `artifact.endswith(appendix) = True`（2084 字节 / 21 行）。
    - **不修的另一半写进代码而非忘掉**：`### U-Net 对比模型` 在文档**中部**，其中部拼接点无法验证
      —— 生成脚本的输入 `logs/zernike_amp_sweep.json`、`logs/zernike_amp_final/summary.json` 被
      `.gitignore:41` 的 `logs*` 排除（`git ls-files logs/` 仅 1 条），从克隆根本跑不起来。
      盲插 1228 行的 f-string 风险大于收益 ⇒ 以 `KNOWN_UNEMITTED_HEADINGS` 做**棘轮**登记
      （只可缩不可涨；擅自删基线条目会立刻失败，因为标题仍不在脚本中）。
    - **顺带查实的溯源缺口**（属口径决定，未在本轮改）：该报告 provenance 块写有
      `复现命令: python scripts/ml/generate_zernike_amp_report.py` 与 `运行环境: 离线`，
      但**输入不在仓库**（11 张图已提交，JSON 未提交）⇒ 「可复现」只对当时那台机器成立。
      要不要提交这类输入、或给 provenance 增加 `inputs=` 并显式声明缺失，另议。
    - **为何没有「所有生成报告」的守卫**：标题-字面量这条**代理不成立**。扫 85 份生成报告得
      **346 个「孤儿标题」/ 1019 标题**，样例 `### 6.1` 与 `### gaussian__defocus__off` 全是
      **由数据在循环里拼出的标题**，永远不会以字面量出现在脚本里。守卫只在
      「单报告 ↔ 单脚本」这种对上成立（就是本条），且必须显式承认标题只是代理。
    - **本轮仪表化又错四次**（延续「比例不合理 ⇒ 先怀疑仪表」）：
      ⑦ `REPORTS` 是**带注解赋值**，我只查 `ast.Assign` ⇒ 取到 `None` 却差点继续统计；
      ⑧ `REPORTS` 的值是 **`ReportProvenance(...)` 调用**而非 dict ⇒ `script` 字段全缺，
        `REPO / ""` 退化成仓库根并抛 `IsADirectoryError`；「85 生成 / 54 手写」是修好后才成立的数；
      ⑨ 新测试的三个用例把 **fixture 当自由变量**引用（报 `FixtureFunctionDefinition has no len()`），
        三条断言**根本没执行**，当时只有第四条是真失败 —— 「测试失败」不等于「差分成立」；
      ⑩ 行核对扫了**全文**的粗体首格行，把别的表的行当成缺失。
      另：一次在 heredoc 里手写三引号定界符导致 `SyntaxError` —— 自此**含引号/反斜杠的编辑一律
      先写成文件再执行**（本轮的脚本改动与差分驱动都是这么做的）。
    - **验收**：新测试 `tests/ao_shaping/scripts/test_zernike_amp_report_appendix.py` **5 passed**。
      更大范围 `tests/ao_shaping/scripts/` + `test_conventions.py`：**2463 passed / 65 skipped / 1 failed**，
      唯一红是 wt-190 的 `_fmt` 复用守卫，已在**干净 HEAD worktree**（我的三个文件不存在）复现
      ⇒ 与本轮无关，不抢修。
      **差分三条**：只改产物 → 2 failed；只改脚本常量 → 同样 2 failed（**两个方向的漂移都被抓到**）；
      只往产物手加一节 → 棘轮抓到并点名新增标题。写点未接线那条差分在改脚本**之前**已取到。

34. ✅ **给「生成报告以无人能发射的尾节收尾」建守卫：11 份产物的尾节只存在于产物里**（2026-10-25）：
    - **缺陷**：provenance 登记的 85 份生成报告中，**11 份的最后一节**（696/441/713/503/267/569/526/
      415/163/259/326 字节）在其 owner 脚本与**所有**被跟踪的 `scripts/`、`src/`、`tests/` 源文件里
      都找不到对应字句，而 owner 用 `write_text` **整文件覆写**该产物 ⇒ 下次重新生成就删掉这节内容。
    - **判据必须是内容级，不能按 `provenance.script` 核对**：该字段记的是**复现入口**，不是写盘者。
      反例：`report/miicam_auto_exposure/miicam_auto_exposure_probe/report.md` 登记的是
      `tests/ao_shaping/drivers/ccd/test_miicam_auto_exposure_report.py`（它 subprocess 驱动探针），
      而 `## 运行后状态` 实际由 `scripts/ccd/miicam_auto_exposure_probe.py`（`lines.append` 处）发射。
      按 script 字段核对会把这一节误判为孤儿（本轮 13 条候选里就有它）。
    - **标题只是代理，所以只扫最后一节**：先前按「全部标题字面量」扫 85 份得 **346/1019 条假孤儿**
      （`### 6.1` 之类带运行名的标题、`### gaussian__defocus__off` 都是循环里由数据拼出来的，
      永远不会以字面量出现在脚本中）。扫**尾节**（附录 / 结论 / 复现）既覆盖本类又避开数据拼标题。
    - **不搬回生成器，改为棘轮登记**：这 11 份的 owner 在本环境跑不起来 —— 两个要 `data/debug`
      （`scripts/build/stage_test_data.sh` 的 `EXCLUDE_RE` 明确排除 `debug`，且所需 `summary_snr.json`
      在暂存树与本机语料里实测**均为 0**），其余 `environment=硬件`。把正文移进无法执行的生成器，
      只是把一个不可验证换成另一个不可验证。故用 `_UNEMITTED_FINAL_SECTIONS` 基线：
      **出现新孤儿即失败；基线条目若已不再是孤儿也即失败**（修好必须删条目，不许基线悄悄腐烂）。
    - **两个反空转下限**：emitter 语料 > 500（实测 1274）、扫到的生成报告 ≥ 60（实测 85）。
      另加**跨解析器互校**：新 `_report_owners()`（AST 读 `ReportProvenance(...)` 调用）与第 26 项的
      正则版 `_generated_report_paths()` 给出的集合必须相等 —— 两种读法一旦漂移就失败
      （本仓已两次被重复实现的漂移咬到）。
    - **写守卫时就发现的自指陷阱**（未上线）：基线字典逐字引用了那 11 个标题，若 emitter 语料包含
      本测试文件自身，则每条基线都会「被记录它的守卫发射」，`stale` 一侧全部失效 ⇒ 语料排除 `__file__`。
    - **验收**：容器 `test_conventions.py` **42 passed / 1 skipped**（新增一条守卫）；
      `tests/ao_shaping/scripts/` **2422 passed / 1 failed**，唯一红是 wt-190 的 `_fmt` 复用守卫
      （今日已在干净 HEAD worktree 复现，该片为 advisory、blocking=0），与本轮无关。
      **差分两条**（scratch worktree 里先拷入改过的守卫，否则「通过」只说明守卫不存在）：
      给某生成报告追加一个新尾节 → `1 failed … end on a section no emitter can reproduce` 并点名该产物；
      把某基线节的标题加进其 owner → `1 failed … no longer in that state`（证明棘轮两侧都咬）。
    - **仍开放**：把这 11 节真正搬回各自生成器，只能在能执行该生成器的机器上做（硬件会话或本地
      `data/debug`）；`REPORTS` 没有「实际写盘者」字段，若要机器化审计需加字段，属 schema 决定
      （本轮只在 `note` 里补了人工说明，未改语义）。

35. ✅ **修掉活文档里 16 处指向不存在路径的链接，并加「相对链接必须落在 git 索引里」守卫**（2026-10-25）：
    - **两类缺陷**：① `docs/GUI/<area>/<ui>/` 下 12 处链接**爬出仓库根** —— 这类文档距根 4 层，
      链接却写了 5 或 6 个 `../`，且标签漏了 `src/` 前缀；其中 `ao_shaping/utils/image/spots_calc/`
      还把**模块文件**写成包目录（被跟踪的是 `src/ao_shaping/utils/image/spots_calc.py`）。
      ② 4 处 `](../logs/mamba_curriculum_20260323_121023/report.md)`（`docs/ml_training.md` 3 处、
      `src/ao_shaping/drivers/sim/AGENTS.md` 396 行 1 处）——**标签里写的正是已提交的报告路径**，
      目标却指向被 gitignore 的 `logs/`：作者本机路径习惯泄漏进文档。
    - **判据必须是 git 索引，不能是磁盘 `exists()`**：`logs/`、`data/` 在作者机器上存在、在**任何新克隆
      里都不存在**，只有按 `git ls-files` 解析才抓得到第 ② 类。`data/md_test/` 既未跟踪、本机也不存在
      ⇒ 无物可链，改为行内代码（保留路径文字，去掉必然失效的链接）。
    - **复用既有机器，不另写解析器**：围栏屏蔽用 `_unfenced_md_lines()`（它保留原始行号），
      活/历史分界用 `_is_historical()` —— 于是 `report/**` 与 `docs/dev|slm|wfs|daheng|…` 合理落在射程外：
      带日期的实验记录引用本机 `data/debug/` 产物是记录本身，不是错误指引（与第 32、33 项同一口径）。
    - **顺序即正确性（本轮两次自伤）**：先剥行内代码再判围栏，会把 ```` ```mermaid ```` 的前两个反引号
      当成空代码跨度吃掉，围栏判定随即失效、**图内文字被当正文链接扫**，行号全错位（一度报
      `sim/AGENTS.md:352`，实际在 396）。正确顺序：先屏蔽围栏，再把代码跨度**原地置空**
      （保长度、保偏移），行号才可信。
    - **第二处顺序坑（已回滚重做）**：`read_text`/`write_text` 带通用换行转换，第一版据此整文件重写，
      在 6 个文件的每个**收尾围栏前插入空行**；暴露信号是 `git diff --numstat` 出现 7/1 这类
      **不对称**计数。改为 `read_bytes`/`write_bytes` + `splitlines(keepends=True)` 分段替换，
      最终 diff 为严格 1:1 行替换（14 文件、16 处）。
    - **反空转下限** `_MIN_LIVE_DOC_LINKS = 200`：解析不到足够目标即判失败 —— 计数守卫必须**同时计通过项**
      （第 32 项的教训：不计数就无法区分「干净」与「没扫到」）。
    - **验收（两侧差分，数目精确对上）**：修好后守卫通过；把**改过的守卫**拷进**未修的 HEAD worktree**
      ⇒ 报 **恰好 16 条**，逐条即本轮修掉的那些，且行号正确（396）。全量 `test_conventions.py`
      **43 passed / 1 skipped**（新增一条守卫）。
    - **仍开放**：`report/**` 内指向 `data/debug/` 的链接按上述分界不入本守卫；若要机器化审计
      「实际写盘者」，需给 `REPORTS` 增字段（第 34 项已记），属 schema 决定。

36. ✅ **兑现「落地即转 blocking」：`runners-gui-misc` 转阻塞；并记一次「不能建守卫」的反例**（2026-10-25）：
    - **由头**：上游 `c5cd68a9` 重做了 `cli_help_golden.json`（即 wt-193 在办的 `slm-gsnet --help`
      基线）。第 29 项当时写的是「两者落地后即转 blocking」—— 于是这一半到期了，**兑现而不是留着**。
    - **翻牌前测的是整片，不是单个文件**：`runners-gui-misc` 的六个目标（`runners` `gui` `model`
      `display` `report` `ao`）在 CI 资源档 `AO_CPUS=4 / AO_MEMORY=6g` 下实测
      **1149 passed / 30 skipped / 0 failed（34 s）**。只凭 `test_cli_contract_freeze.py`
      38 passed 就翻牌**不成立** —— 该片还盖着另外五个目录。
    - **顺手排掉一个更严重的怀疑**：该片显式传 `tests/ao_shaping/gui`，而 `pyproject.toml` 的
      `--ignore` 只作用于 discovery、显式路径不受影响 ⇒ 会不会把唯一跑真 `AppTest` 的
      `test_zernike_abort_repro.py` 收进来挂死？**不会**：`scripts/build/run_pytest_ci.sh`
      自己注入 `--ignore`（180-181 行），而 CI 调的就是这个 harness，故 CI 与本机同覆盖。
    - **改动**：`run_shard "runners-gui-misc" 0 0` → `0 1`（现 **7 blocking / 1 advisory**）。
      `scripts` 片仍 advisory —— 其红是 wt-190 的 `_fmt` 复用守卫，今日已在**干净 HEAD worktree**
      复现，尚未落地。同时改掉两处**翻牌后会撒谎的说明**：workflow 注释块与
      AGENTS.md「两片暂为 advisory」。
    - **验收**：容器 `test_pytest_workflow_shards_cover_tree.py` + `test_conventions.py` +
      `test_cli_contract_freeze.py` **86 passed / 1 skipped**；术语门禁 `合计: 0`；arity 0/1489；
      ruff 全绿。三处锚点各**只匹配一次**（`once()` 断言），所以不存在文档与配置半更新。
    - **命名一处盲区（不假装已修）**：第 33 项的附录守卫 `test_zernike_amp_report_appendix.py`
      落在 `tests/ao_shaping/scripts/`，正是**仍 advisory** 的那片 ⇒ 它目前只能报警、不能拦停。
      第 34、35 项的守卫在 `test_conventions.py`（root-guards，blocking）才有牙；
      `scripts` 片转 blocking 时前者一并生效。
    - **反例：phantom 命令名这条不能建守卫**。上游那条 commit 修了 5 处把 `spgd-square`
      当顶层命令的引用，我本想据此扩守卫，普查后**放弃**：`spgd-square` / `gs-square` /
      `diff-beam` 在跟踪文件里的剩余出现（115 / 11 / 23 条）几乎全是**刻意保留的废弃说明**
      —— `AGENTS.md`「无 `spgd-square` 顶层命令」、`README.md`「已从 CLI 中移除」、
      `docs/user/cli-reference.md` 的「历史沿革」。机械判据分不清「声称它是命令」与
      「声明它已不存在」，硬建就会去改坏正确的历史说明。
      ⚠️ 而且我自己先把 `hadamard-matrix` 误列进 phantom 名单 —— 它**确实已注册**
      （`main.py` 19 条之一），差点据此立一条把合法引用判成缺陷的守卫。可判定的那一半
      第 30 项的五源互校已经钉住（golden 19 键、其中无 `spgd-square`，与 `main.py` 注册集一致）。
    - 另记一次仪表化自伤：grep 模式里的**反引号放在双引号内**会触发命令替换
      （`寻找匹配的 `` EOF`）。自此含反引号/反斜杠的一律先写成文件再执行。
37. ✅ **入口调用路径全是断的：启动器、被启动模块自己的 docstring、脚本索引三处都在教同一条不存在的路径；补「`python <路径>.py` 必须在 git 索引里」守卫**（2026-10-25）：
    - **由头**：`scripts/ml/train_ml.ps1` 跑的是 `python src/ao_shaping/ml/train.py`，而**该文件不存在**
      （包是 `src/ml/`，console script `ao-train = ml.train:cli`）。更糟的是这条死路径的**源头是
      被启动模块自己**：`src/ml/train.py` 的 docstring 五行用法全写 `uv run python src/ao_shaping/ml/train.py`，
      `scripts/README.md` 又引用这个启动器 —— 文档把缺陷当规范传给下一位读者。
    - **启动器其实是三个 bug 叠在一起**，不止路径写错：(1) 文件路径不存在；(2) `Set-Location $PSScriptRoot`
      把 CWD 设在 `scripts/ml/`，于是 `PYTHONPATH = "src;libs"` 这个**相对**值解析到
      `scripts/ml/src`；(3) 同一 CWD 让 `--data-dir` / `--output-dir` 的相对默认值落在错的树里。
      修法：`Set-Location "$PSScriptRoot\..\.."`（回到仓库根，`src;libs` 遂成立）+ 改用
      与 CWD 无关的 `python -m ml.train $args`。
    - **两侧证明（容器）**：修好的调用形态 `PYTHONPATH=src python -m ml.train --help`（仓库根）
      **rc=0** 并打印 `Usage: python -m ml.train [OPTIONS] COMMAND [ARGS]...`；旧的 file-path 形态
      **rc=2** `can't open file '/workspace/scripts/ml/sr…'`。⚠️ **诚实边界**：PowerShell 本身没执行过
      （本机无 Windows 也无 `pwsh`），验证的是**调用语义**而非那层 shell。
    - **顺带修的两处同族陈旧引用**：`scripts/README.md` 里 `src/ao_shaping/tools/slm/slm_zernike_sweep_probe.py`
      → 实际是 `src/ao_shaping/tools/zernike/sweep_probe.py`；`docs/paper_gs_refine_experiment.md`
      复现命令 `python scripts/replay_gs_refine_hw.py` → `python scripts/slm/replay_gs_refine_hw.py`。
      另修 `scripts/_common/provenance.py` 两条 `note` 里的陈旧路径，并重跑
      `scripts/sync_report_provenance.py`（更新 2 处 header，`--check` 报 `check: clean`）。
    - **守卫** `test_python_invocation_paths_in_docs_and_docstrings_are_tracked`（补第 30 项
      `python -m <模块>` 守卫的姊妹缺口 —— 文件路径形态此前无人核查）：语法
      `(?:^|[\s(>])(?:uv run )?python3?[ \t]+((?:src|scripts|tests|libs)/[\w./-]+\.py)(?![\w.])`，
      扫**跟踪 .py 的模块 docstring** 与**非历史活文档**，零容忍。
      ⚠️ 右侧 `(?![\w.])` 是**必需**的：去掉它 `src/calculators/gerchberg_saxton.pyx`（已跟踪的
      Cython 模块）会被当成 `…/gerchberg_saxton.py` 报成缺陷 —— 即守卫把真实文件判成错误。
    - **先 dry-run 再决定豁免**：本普查先前手工排除了 `CHANGELOG.md`，但守卫用的是仓库自己的
      `_is_historical()`，而它**不豁免** CHANGELOG ⇒ 有可能一落地就红。于是从守卫文件里
      `ast` 读出 `_HISTORICAL_DOCS` 原样 dry-run（避免我的 dry-run 与上线守卫在范围上漂移）：
      **533 条调用声明 / 0 缺陷** ⇒ 不需要凭空发明豁免。反空转下限取 `_MIN_INVOCATION_CLAIMS = 400`。
    - **验收（两侧差分，数目精确对上）**：把**改过的守卫**拷进**未修的 HEAD worktree**
      ⇒ 报**恰好 7 条**：5 条来自 `src/ml/train.py` 的 docstring、2 条来自活文档
      （`scripts/README.md`、`docs/paper_gs_refine_experiment.md`）—— 一次运行同时证明
      **docstring 与文档两个面都有牙**；修后工作树 **0 条**。再注入一条全新假路径
      ⇒ **8 条**且指名，证明它不是在复述历史。全量 `test_conventions.py` **44 passed / 1 skipped**。
    - **顺手挖出一个未来会炸的语法错误**：新守卫 `ast.parse` 全部 **1238** 个跟踪 .py，
      于是暴露 `tests/ao_shaping/drivers/ccd/test_exposure_states.py:9` 的 docstring 里
      Windows 路径 `.venv\Scripts\python.exe …` 含 `\S` / `\p` —— **invalid escape sequence**
      （3.13 报 SyntaxWarning，未来 Python 会变 SyntaxError），且它会随 root-guards 片每次运行
      污染 CI 输出，被误读成守卫自身的问题。修法只有**一个字符**：docstring 加 `r` 前缀。
      机器化确认**文本逐字符不变**（`docstring text identical: True`，1849 字符），复扫
      1238 个文件 ⇒ **emitters 0**，容器守卫运行不再有 `warnings summary`。
    - **再记三次仪表化自伤**：(1) `ast.Assign` 的属性是 **`targets`（复数）**，我写了 `.target`
      （那是 `AnnAssign` 的）⇒ dry-run 首跑 `AttributeError`；(2) shell 里 `echo` 想输出含
      `'S'` 的字面量，引号套叠造成 `寻找匹配的 '' 时遇到了未预期的 EOF`，**吞掉了同一条命令里
      后面的两段**；(3) 沙箱无 `git` ⇒ 要比对 HEAD 版本必须先在 host 侧 `git show … > /tmp/…` 再读。
    - **仍开放（命名，不假装已修）**：`scripts/ml/generate_fouriergsnet_pipeline_report.py`
      与其注册产物不一致 —— 脚本引用一个在 `baefba2d` 删掉的测试文件与一个在 `f19dced5`
      删掉的 CLI；默认 `-o report/fouriergsnet_pipeline` 并**不写**注册表里的产物路径；
      已提交产物里有 **22 行**生成器从不发射。该生成器纯 stdlib、可离线跑，所以这条产物
      **能**端到端复现，但重新生成会删掉手写结论（属第 33 项那一类取舍）⇒ 需先裁定归属再动，另立一项。
38. ✅ **让 fouriergsnet 报告的生成器与已提交产物逐字节一致；给「离线可重跑」的生成报告建端到端复现守卫**（2026-10-25）：
    - **由头**：第 37 项末尾登记的未决单元。三条可测缺陷：(D1) 生成器默认 `-o report/fouriergsnet_pipeline`
      写的是 `report/fouriergsnet_pipeline/report.md`（**未被跟踪**），而注册产物在
      `report/fouriergsnet_pipeline/fouriergsnet_pipeline_report/report.md` ⇒ provenance 的
      **复现命令写不出它自己登记的那份产物**；(D2) 生成器 docstring 以现在时断言已删除的文件存在；
      (D3) 产物比生成器多 26 行、少 11 行。
    - **裁定方向与第 33 项相反：这次产物才是权威。** 那 26 行里是 2026-10-01 的**复核注释**：
      `phi.cpu().numpy()` 这个 bug 已在源头修成 `phi.detach()`、8 处行号已漂移、`SETTLE_S` 现为 0.2；
      而生成器仍把它写成「修复方向: `phi.detach().cpu().numpy()`」。**照第 33 项把手写段搬进生成器，
      就会用过期文本覆盖一份已更正的记录。** 所以改成生成器逐字重发产物正文。
    - **被测主体确已删除（普查核实，不是移动）**：`fouriergsnet_optimize.py` 删于 `f19dced5`；
      `test_sim_fouriergsnet_pipeline.py` 与 `tests/ao_shaping/runners/test_fouriergsnet_optimize.py`
      删于 `baefba2d`；索引里无同名后继。报告因此是 2026-09-22 的历史记录，§1/§4/§5 的路径按当时状态
      保留。我在 §4 补一条 **2026-10-25** 的 `⚠️` 说明（沿用该报告已有的 2026-10-01 `⚠️` 注记惯例）
      声明这些路径不是现存代码。
    - **生成时间必须冻结**：原生成器用 `datetime.now()` ⇒ 重跑会把 `生成时间: 2026-09-22 12:13:32`
      改成运行当天，既篡改「记录何时写下」又让逐字节复现永远不可能。改为把正文（含该行）逐字内嵌、
      取消 `.format` ⇒ 守卫**不需要任何豁免**，比较就是纯字节相等。
    - **新守卫** `test_frozen_reports_reproduce_their_committed_artifact`（`test_conventions.py`，
      root-guards 片，blocking）：把生成器跑到 `tmp_path`（`-o` 传**绝对路径**，pathlib 遇绝对操作数会
      替换 ⇒ 绝不写进仓库），与产物剥掉 provenance 块后的正文比对。**登记前提**：该生成器只 import
      标准库、正文是字面量（不读 `data/`、`logs/`）。其余 84 份生成报告做不到，所以
      `_FROZEN_REPRODUCIBLE_REPORTS` 刻意显式且短。
    - **差分（三段，方向明确）**：先上守卫再修 ⇒ **红**，报「产物独有 26 行 / 生成器独有 11 行」
      —— 失败归因于真实缺陷，而不是我注入的假缺陷；修后工作区**绿**；把一致状态三件套（守卫 +
      生成器 + 产物）拷进 HEAD worktree 仍**绿**；再在产物尾部手写加一节 ⇒ **红**，报
      「产物独有 4 行 / 生成器独有 0 行」—— 正是当初造成该缺陷的那个回归方向。
    - **同步修掉的**：`scripts/README.md` 里 `-o report/fouriergsnet_pipeline` 那条示例（指向未被跟踪的
      杂散路径）删除，"Documents the 5 integration tests … that drive the REAL standalone …"
      由现在时改过去时并点名两个删除 commit；`provenance.py` 该条 `note` 增「冻结：正文与生成时间
      逐字重发；被测脚本/测试已删除」，重跑 sync（`headers updated: 1`、`--check: clean`），并**在换块
      之后重新验证字节相等**（5469 字符，True）。
    - **普查负结果（不建守卫）**：试过把守卫扩成「活文档里**全限定**的 `src|scripts|tests|libs/….py`
      路径引用必须在 git 索引里」——484 个 token 中 6 个未跟踪，逐个裁定后**只有 1 个**是真缺陷
      （`scripts/README.md:2930`，本项已修），其余 5 个都合法：`docs/drivers_architecture.md:822` 的
      `newbrand.py` 是「如何新增驱动」的**假设文件名**；`src/calculators/CYLON_CONVERSION_SUMMARY.md:9`
      的 `gerchberg_saxton.py` 讲的正是**它已变成 `.pyx`**；`docs/issues_report.md` 三处是登记当时的路径。
      机器判据分不清「假设」「已改名的历史记录」与「笔误」⇒ 与第 32、36 项同类，不建守卫。
    - **再记三次仪表化自伤**：(1) `_UNEMITTED_FINAL_SECTIONS` 与 `REPORTS` 都是**带注解赋值**，
      我**第三次**只匹配 `ast.Assign`，于是 dry-run 报 `entries = 0` 的**假绿灯**（0 缺陷看着像干净）——
      只因我把「必须 ~139」写进打印才没被骗；(2) `except ValueError → "<<sentinel>>"` 把 `HANDWRITTEN`
      变成**字符串**，于是 `isinstance(script, str)` 把 54 条手写条目误计成生成条目（真实是 85/54）；
      (3) 我从 `provenance:end` 切分产物时把 **H1 标题划进了 provenance 块**，生成的模板丢了标题 ⇒
      字节比较 False；若沿用「两种推导各测一次」的老写法就会漏掉。教训：**同一量只允许一种推导式**。
    - **仍开放**：该报告 §1/§5 与 `scripts/README.md` 里的 `L814` / `L938` 一类**行号**引用，按第 32 项
      结论仍不可判定，保留原样；`report/**` 内指向已删除代码的路径按第 35 项的历史分界不入守卫。
39. ✅ **`.env` 里有一个从未被任何代码读取的配置键；补「声明的环境变量必须被跟踪代码读取」守卫（守卫自己差点替它背书）**（2026-10-25）：
    - **普查**：`.env` 与 `.env.example` 共声明 6 个键，5 个确有读者（`Far_Cam_ID` / `Near_Cam_ID` /
      `IDEAL_SPOT_RADIUS` / `PATH` / `PYTHONPATH`），**`CENTER=577,655` 没有任何代码读取**：带引号的字面量
      `"CENTER"` 在全仓跟踪文件里出现 0 次，`git log -S'"CENTER"' -- '*.py'` 也为空 ⇒ 它不是重构残留，是
      **生来就死**（在 `546bb610`「unify config and CLI helpers」中加入）。分析中心真正的来源一直是 runner
      选项 `-c/--center`（`src/ao_shaping/runners/runner_common.py:240`）。
    - **为什么值得拦**：这两个文件是仓库的配置契约，`AGENTS.md` 还把该 env 块**镜像了两遍**。读者照文档设一个
      没人读的键，得到的不是行为而是沉默 —— 一个「看着权威却无效」的键比缺这个键更坏。
    - **语义前提（先核实，别建在假设上）**：全仓**没有 dotenv**，且 `.github` 与 `scripts/build` 里**没有任何
      步骤加载 `.env`** ⇒ 这些键是经编辑器终端环境进入 Python 的。所以「没有代码读」等价于「设了也无效果」，
      不是 CI 另有注入通道造成的假阳性。
    - **守卫** `test_declared_dotenv_keys_are_read_by_tracked_code`（`tests/ao_shaping/test_conventions.py`，
      root-guards 片，**blocking**）：解析两个 `.env*` 的**非注释**赋值行，要求每个键在跟踪代码里以带引号
      字面量出现。语料只收 `*.py *.pyx *.sh *.ps1 *.yml *.yaml *.toml *.cfg *.ini`，**故意不收 `.json`** ——
      `report/**/plan_*.json` 会回显 `AO_DAHENG_MAX_EXPOSURE_MS` 这类别名，收进来等于让**产物给缺陷作证**。
      `PYTHONPATH` / `PATH` 走显式豁免（读它们的是解释器、OS 与工具链，不是本仓库代码）。反空转下限
      `_MIN_ENV_KEYS_DECLARED = 4`（实测 5–6）、`_MIN_ENV_FILES_SCANNED = 1000`（实测 1256）。
    - **守卫差点自证清白（第 31 类自伤）**：第一版在**仍声明 `CENTER` 的 HEAD worktree** 上报 **GREEN**。
      临时探针把中间量打出来才看清：`key=CENTER hits=1 ['tests/ao_shaping/test_conventions.py']` —— 全语料里
      唯一的 `"CENTER"` 命中是**这条守卫自己的注释**。搜索一个名字，会被「提到这个名字的文件」满足 ⇒
      修法是**只把本模块从命中里剔除**（`self_rel`），**不是**把整个 `tests/` 剔除（`Far_Cam_ID` 的合法消费者
      里就有 `tests/ao_shaping/test_config.py`）。这是第 34 项「基线字典自引用」的同一形状。
    - **差分（三段，归因不靠注入）**：HEAD worktree（仍声明 `CENTER`）⇒ **红**，点名
      `CENTER (declared in .env)`；修好的工作区 ⇒ **绿**；HEAD 上再注入一个假键 ⇒ **红**且同时点名 `CENTER`
      与 `PHANTOM_KNOB_XYZ`。第一段就是真实缺陷，无需我制造。
    - **改动**：`CENTER=577,655` 从 `.env`、`.env.example` 删除（连上面那行说明注释一并删，否则文件仍在描述
      一个不存在的键）；`AGENTS.md` 两处镜像改为**指针**而非静默删除（英文块
      `CENTER is not an env knob … comes from -c/--center`，中文块「`CENTER` 不是环境变量：分析中心由 runner
      选项 `-c/--center` 给出」），沿用该文件对 `MAX_GRAYSCALE_VALUE` 已有的「不再是环境变量」写法，防止再次
      被加回来。改后全仓**再无任何 `CENTER=` 声明**：唯一残留的 `CENTER=` 出现是在守卫自己的注释里引述这个
      键名 —— 那正是第 31 类自指的来源，所以守卫必须剔除本模块。
    - **再记两次仪表化自伤**：(a) 我把一条 `git grep` 输出经 `| head -14` 截断后当**完整**结论读，据此宣称
      「`AO_DAHENG_MAX_EXPOSURE_MS` 不在任何 `.py` 里」—— 它在 `src/ao_shaping/utils/image/exposure_control.py:25`
      以常量 `DAHENG_EXPOSURE_LIMIT_ENV` 定义、由 `configured_daheng_exposure_limit()` 读取，跨 40+ 处使用。
      **截断的普查当完整普查读会凭空造出缺陷**，与「0 命中当干净」是对称的两种错法；(b)
      `git grep -n -- 'pat' -- pathspec` 里**第二个 `--` 不是分隔符**，pathspec 被当成了模式，返回空被我读成
      「无匹配」—— 那是**假阴性**，正例控制当场作废。另：把守卫改紧时我误删缩进层级、又复制出一行重复的
      `for`，`IndentationError` 让三次差分**全部无输出**；若非我核对「必须先看到 passed/failed 行」，这会被读成
      「没有缺陷」。**过滤后的 grep 只有在 collection 被证明发生后才有意义。**
    - **仍欠（复测，不假装兑现）**：第 36 项的 `scripts` 片转 blocking 仍未兑现。本轮在 master 上整片复测
      **2422 passed / 1 failed**，唯一失败仍是
      `test_no_generator_redefines_an_extracted_helper[_fmt]`（wt-190 在办，其产物一致性属它的范围，不抢）。
      ⚠️ `refactor/190-fmt-dedup` 的分支尖端是 master 的祖先，但这**只说明该分支陈旧**，不说明修复已落地 ——
      判据是守卫的颜色，不是祖先关系。第 33 项的附录守卫仍在该片内，故今日仍只报警、不能拦停；本项守卫在
      root-guards（blocking），所以从落地起就有牙。
40. ✅ **脚本索引里 5 条复现命令仍指向 `569ea514` 重组前的旧路径；补「活文档 shell 代码块里的脚本路径必须在 git 索引里」守卫**（2026-10-25）：
    - **由头**：`569ea514`（group the 139 flat scripts by what they process）把平铺脚本按处理对象归入子包。
      章节标题只写**裸文件名**，所以标题照样解析得到；但 **Usage 代码块里的完整路径没跟着改** ⇒ 五条命令
      照抄即失败。其中一处正是第 39 项刚修过本体的 `scripts/ml/train_ml.ps1` —— 本体的死路径修好了，
      文档里教它的那条调用路径仍是旧的 `.\scripts\train_ml.ps1`。
    - **五处已修**（`scripts/README.md`，各按其代码块自身的分隔符风格）：58 `.\scripts\build.ps1` →
      `scripts\build\build.ps1`；71 `.\scripts\build_cython.bat` → `scripts\build\…`；
      100 `scripts/setup_pytorch.sh` → `scripts/build/…`；318 `.\scripts\train_ml.ps1` → `scripts\ml\…`；
      3642 `scripts\wfs_driver_hw_test.py` → `scripts\wfs\…`。
    - **守卫** `test_reproduce_commands_in_live_docs_name_tracked_scripts`（`tests/ao_shaping/test_conventions.py`，
      root-guards 片，blocking）：只扫**带 shell 语言标签**（bash / sh / shell / powershell / ps1 / cmd /
      batch / console）代码块内、以 `scripts|src` 开头且以 `.py .sh .ps1 .bat` 结尾的路径，要求把反斜杠
      归一化后落在 git 索引里；含 `*` `<` `>` 的模板名跳过。零容忍，反空转下限
      `_MIN_CMD_SCRIPT_CLAIMS = 200`（实测 297）。
    - **范围为何不含正文**（与第 32、36 项同一堵墙）：把活文档的正文与代码块合起来普查，416 条路径声明里
      6 条不解析，但**其中 5 条是合法的** —— `docs/issues_report.md:27/29/101` 是登记当时确实正确的旧路径
      （日期化记录），`docs/drivers_architecture.md:822` 的 `src/ao_shaping/drivers/ccd/newbrand.py` 是
      「如何新增驱动」的**假设文件名**。把它们纳入守卫等于逼人篡改正确记录 ⇒ 本守卫只管**可执行声明**
      （shell 代码块里的路径），正文一概不管。
    - **一条被自己测量否证的假设（没有按错理由改代码）**：我原以为第 39 项的 `_INVOCATION` 漏掉这五条是
      因为它只认正斜杠，于是把语法放宽到 `[/\\]` 重测：**claims=238 / offenders=0，与放宽前完全一致** ⇒
      假设作废，**没有**据此去改 `_INVOCATION`。真正原因是 3642 行写的是
      `.venv\Scripts\python.exe scripts\…`：**解释器本身是个路径**，`python[ \t]+` 在那里根本不成匹配。
      因此新守卫刻意**不看解释器怎么拼**，只看代码块里的路径。
    - **两次仪表化自伤（都是先前踩过的同形坑）**：(1) 我第一版左边界写成 `(?<![\w/\\])`，**把前导反斜杠
      排除在外**，于是 `.\scripts\…` 这类 Windows 形式全体隐身：语料 297→292、缺陷 5→2 —— 一条用来抓
      「路径写错」的守卫自己漏掉三类缺陷，而且它当时**显示为通过**。是并排比较两种语法才暴露的。
      (2) 代码块语言标签用了非捕获 `(?:…)`，却又调 `open_match.group(1)` ⇒ `IndexError: no such group`；
      上线前先在容器外用 stdlib 探针跑一遍才发现，改法是**不记语言名**，只保留 `in_fence / shell_fence`
      两个状态。另记：探针里我把 `[\w-]` 写进**非 raw** 的字典键，触发 `SyntaxWarning: invalid escape
      sequence` —— 与同日修掉的 `test_exposure_states.py` 属同一类错误。
    - **顺带核实：`scripts/build/build.ps1` 本身已半废弃（不擅自「修」）**：其第 3 步的 Nuitka 入口
      `src/ao_shaping/wf-less/DM_cam.py` 已在 `44867a71`（连同 `wf-less` 目录）删除，首行又硬编码
      `d:\workspace\AO-shaping`，而 `.gitignore:38` 至今仍在忽略 `DM_cam.*`。我**没有**替它挑一个新的打包
      入口 —— 那是在猜产品意图（该应用是被有意当作 obsolete 删除的）。改为在 `scripts/README.md` 里加
      **日期化 ⚠️ 说明**：整脚本只有在仓库恰好位于那个绝对路径时才能运行，第 3 步在任何机器上都不可执行，
      恢复它得先决定入口应指向何处。⚠️ 我第一版说明写的是「前两步仍然有效」，核对首行后判定**不实**
      （硬编码路径使第 1 步就失败）并改写 —— 文档里的能力声明同样要经得起逐行核对。
    - **差分（三段，归因不靠注入）**：守卫先上线、文档不动 ⇒ **红**，点名 5 条并各附「同名文件在索引里的
      位置」；改好文档 ⇒ **绿**（1 passed / 47 deselected）；把一致状态（守卫 + README + 文档）拷进 HEAD
      worktree ⇒ **绿**；再往 `docs/user/cli-reference.md` 的 bash 块注入一条假命令 ⇒ **红**，点名
      `scripts/no_such_tool_never_existed.sh`（并提示该 basename 无任何跟踪文件）。
    - **验收**：容器 `test_conventions.py` 全量 + provenance + shard 覆盖 **614 passed / 1 skipped**
      （skip 是沙箱里没有 ruff 在 PATH，已由仓库根 `ruff check .` 单独通过）；术语门禁
      `合计: 0 个文件, 0 行`；新增行无禁用词；`scripts/README.md` 强调配对与 HEAD 持平（1→1）；ruff 全绿。
41. ✅ **三处「自己把自己短路」的布尔条件（含一条永不会失败的断言）；补 AST 守卫**（2026-10-25）：
    - **类别**：`x or True` 与 `x and False` 里的**支配常量**会把同一表达式里其余部分全部丢掉 —— 写成像检查、
      实则永不为假。只拦支配常量这一侧：`x or False` / `x and True` 只是冗余，不会让任何检查失效。
    - **普查**：跟踪的 `.py`/`.pyx` 共 1238 个可解析；守卫范围限定 `src/ scripts/ tests/` 共 **1223** 个
      （`libs/` 是第三方 SDK，不由我们改，与 `ruff.toml` 同界），命中 **3 处**。
    - **(1) 报告生成器的死指标** `scripts/oopao/generate_asm_vs_fraunhofer_report.py:127-130`：
      `gs_shift_is_exact_ifftshift = bool(np.allclose(...) or True)`，原注释自己承认「实际要比的是两个构造器
      之间」，于是把比较结果丢掉。**不截断的 grep 显示全仓只有这一行定义它，没有任何消费者**（模板与产物都不
      引用）⇒ 既空转又是死代码，删这 4 行即可，产物一字不动。该脚本会**计算并重写**产物，所以我**刻意没有
      运行它**：跑一次就会把带日期与当日 GPU 名的记录覆盖掉。
    - **(2) 硬件工具里的惯性控制流** `src/ao_shaping/tools/slm/cartographer/hartmann_capture.py:264`：
      `if wavefront_device is not None or True:` 的块体内**已经**做了回落
      （`wavefront_device if wavefront_device else self._wfs`），所以这个门只是无效包装。删门并把 2 条语句
      反缩进，**用 AST 证明等价**（`if` 体的 `ast.dump` 与反缩进后代码逐条相同，且 `orelse` 为空）。
      ⚠️ `tests/` 里**没有任何一处引用 `HartmannCapture` / `hartmann_capture`** —— 这一处不能靠测试取证；
      但 `cartographer/__init__.py:29` 在 import 期就导入该模块，故整条 `tests/ao_shaping/tools/slm`
      （**788 passed / 4 skipped**）都在走它的导入路径。
    - **(3) 最坏的一类：永远不会失败的断言** `tests/ao_shaping/tools/slm/test_beam_extent_sweep.py:184`：
      `assert result.radius_px == result.fall_hi_px - result.fall_lo_px >> 1 or True` —— 优先级使它是
      `radius_px == ((hi-lo) >> 1)`，再被 `or True` 整个中和，**该断言无论实现怎样都会通过**。而**紧邻的下一行**
      就是正确断言 `radius_px == pytest.approx(width / 2.0)` ⇒ 删掉被中和的重复断言（覆盖率不减；把它「修好」
      只会留下第二条同义断言）。
    - **新守卫** `test_no_boolean_short_circuit_discards_its_own_check`（`tests/ao_shaping/test_conventions.py`，
      root-guards 片，blocking）：`ast.walk` 找 `BoolOp`，任一操作数是支配常量即命中；**解析失败不再静默跳过**，
      改为收集并断言为空 —— 否则语料会在无声中缩小，而下面那个计数正是在替语料大小说话。下限
      `_MIN_TAUTOLOGY_FILES = 1000`（实测 1223）。守卫不会命中自己：它提到的 `or True` 只存在于 docstring 与
      注释里（不是 `BoolOp` 节点），自身的 `and`/`any(...)` 也不含支配常量。
    - **差分**：守卫先上线、代码不动 ⇒ **红**，点名 3 处（**没有一处是我注入的**）；三处修完 ⇒ **绿**
      （`test_conventions.py` 全量 + 被改测试 **58 passed / 1 skipped**）。
    - **两条负结果（同样入账）**：(a) 第 40 项之后我普查「哪些注册生成器把报告正文写成字面量模板」，
      65 个生成器里只有 **2** 个如此，另一个 `generate_asm_vs_fraunhofer_report.py` 报出的「漂移 11/12 行」
      其实是**我自己的比较越过了格式化边界**（模板里是 `{torch_vs_numpy:.3e}` 占位，产物里是已填的实测值）——
      不是缺陷；这同时**证明**第 38 项的 `_FROZEN_REPRODUCIBLE_REPORTS` 只有 1 个成员是完备的，而非登记不足
      （其余 63 个生成器从 `data/`、`logs/` 组装正文，逐字节复现根本不可得）。(b) 我一度认定产物第 73 行
      「实测两者相差恰好一个 `ifftshift`（`max|Δ| = 0.000e+00`）」自相矛盾；读生成器代码后确认该指标是
      **扣除 `ifftshift` 之后**的残差（`|canonical_prop − ifftshift(old)|.max()`），0 正是该声明的证据 ——
      是我的怀疑不成立，声明本身成立。
    - **再记仪表化自伤四处**：(1) 早先「该指标无人引用」的结论出自 `| head -8` **截断**的 grep（第 28 类旧错
      重犯），删代码前改跑不截断版本才确认；(2) `ast.True` 不是节点类 ⇒ SyntaxError，另有非捕获 `(?:…)` 上调
      `.group(1)` ⇒ `IndexError: no such group`；(3) 修复脚本**连着两次**死在自己的辅助逻辑上（一次把 body
      起点设在门那一行导致 body 为空，一次对**类方法内部**带缩进的片段直接 `ast.parse` ⇒ `IndentationError`，
      须先 `textwrap.dedent`）—— 两次都在写盘**之前**抛出，`git diff --stat` 证明没有留下半成品；
      (4) 教训同上：**同一量只允许一种推导式**，且写盘类脚本要在改动前把断言全部前置。
42. ✅ **优化器层 4 个可变默认参数（`[]` 在 def 期只求值一次、被所有调用共享）；补 AST 守卫 —— 这类 ruff 看不见，因为 B006 在未安装的 bugbear 里**（2026-10-25）：
    - **普查**：限定 `src/ scripts/ tests/` 的 **1223** 个 `.py`（跟踪的 `.py` 共 1238，另 15 个在 `libs/` 等
      范围外），命中 **4** 处，且全部在
      `optimizer/` 层 —— 而这一层的契约恰恰是「同一进程内两次 run 不得共享状态」，所以类别选得中。
    - **定性是潜在，不是现行**：逐函数 AST 检查显示这些参数在体内**只有 load**（没有 `.append`/`update`
      之类变异，也没有重绑定），且所有静态可见调用点都**显式传参**。另有两处经 `**kwargs` 转发，静态无法证明，
      因此我按「无法证明」记录，而不是宣称「从未被使用」。
    - **改法用的是仓库既有约定，不是我发明的风格**：`optimizer_rms_dm` 与 `optimize_pib` 的 `init_v` 改为
      `... | None = None`，依据是 `optimizer/combined_optimizer.py:95` 早已写成
      `init_v: list[float] | None = None`，且下游 `optimizer/spid_pib_loop.py:224` 本来就写
      `if init_v is None or len(init_v) == 0:` —— 在该路径上 `None` 与空**同义**；`rms.py` 侧的判据是
      `if not init_v:`，`None` 与 `[]` 同假。`grid_search_lr_delta` 的两个扫描列表改**元组**（体内只用
      `len()` 与 `for`，元组等价且不可变）。
    - **新守卫** `test_no_mutable_default_arguments`（`tests/ao_shaping/test_conventions.py`，root-guards 片，
      blocking）：遍历函数/lambda 的 `defaults` 与 `kw_defaults`，命中 `List/Dict/Set` 字面量以及
      `list()/dict()/set()`；**`field(default_factory=...)` 不命中** —— 它正是为规避本缺陷而存在的写法。
      解析失败**收集并断言为空**，不再静默跳过（否则下面的计数就替一个被悄悄缩小的语料背书）。扫描范围**复用**
      第 41 项的 `_TAUTOLOGY_SCOPE`：一份定义而不是两份 —— 本仓已被 `_fmt` 那种「两处实现各自漂移」咬过。
      下限 `_MIN_MUTABLE_DEFAULT_FILES = 1000`（实测 1223）。
    - **差分**：守卫先上线、代码不动 ⇒ **红**，逐条点名 4 处（**没有一处是我注入的**），失败消息里直接给出三种
      合法改法；修完 ⇒ **绿**（conventions 全量与 8 个受影响测试文件同跑 **103 passed / 1 skipped**）。
    - **已知边界，量过再声明**：数组类默认（`np.zeros(64)` 同样只求值一次并共享）不在本守卫内。普查
      `np./torch./cp.` 开头的默认只有 **1** 处 —— `src/ao_shaping/drivers/sim/oopao_backend.py:452` 的
      `np.dtype(np.complex64)`，而 dtype 对象**不可变**，不是缺陷。故该边界之内目前没有真实样本，不是我把类别
      封死了。
    - **三条被数值否证的候选守卫**（记下数字，免得将来重蹈）：(a) 把第 41 项的短路守卫扩到「常量条件」会命中
      **20** 处 `while True:`，全是有意的轮询/事件循环 ⇒ 20 个假缺陷；(b) `except X: pass` 有 **111** 处，多数是
      可选后端的刻意兜底 ⇒ 硬建守卫等于逼出 111 次意图裁定，不建；(c) `assert (tuple)` **0** 处、裸 `except:`
      **0** 处（后者已由 ruff 选中的 `E` 里的 E722 覆盖）—— 归零的两类直接不建，避免重复劳动。
    - **另记一次测量教训（先归因错了一次）**：三次宽范围容器跑都断在进度行、**没有 summary 行**。我最初写成
      「撞上墙上时钟」—— **不对**：补测退出码是 **137 = SIGKILL**（真超时应当是 124），即在
      `AO_CPUS=8 / AO_MEMORY=8g` 档位下被 cgroup 杀掉。再用**不含本次改动**的 HEAD worktree 重跑同一条命令，
      同样 137 断在 41%（并核对 `init_v ... | None = None` 计数为 0，确认改动确实不在场）⇒ 这是**既有环境
      特性**，不是本次回归，也不是该目录的缺陷。结论：这种档位下宽范围跑给不出判据，正确做法是把证据打在
      变更的爆炸半径上 —— 只跑受影响的 9 个文件，**33 s** 即得 103 passed / 1 skipped。另注：
      `pyproject.toml` 已因同类原因把 `tests/ao_shaping/ml` 排除出裸跑默认，`optimizer/` 没有这层保护。
43. ✅ **两处 `pytest.raises(Exception)` 让测试接受任何异常；补守卫（并记下四条被普查否证的候选类）**（2026-10-25）：
    - **普查数值**：`tests/` 的 **507** 个 `.py` 里共有 `pytest.raises(...)` 调用点 **978** 处，其中**裸 `Exception`
      只有 2 处**。本仓惯用**具名**异常（643 `ValueError`、55 `RuntimeError`、42 `SystemExit`、37 `TypeError`、
      12 `click.BadParameter`），所以这 2 处是离群点，不是一种需要保留的风格。裸 `Exception` 的问题是
      **任何异常都算通过**：被测代码因重构抛 `TypeError`、因改名抛 `AttributeError`，测试照样绿。
    - **修法都有出处，不是猜的**：
      * `tests/ao_shaping/utils/test_shaping_objective.py:612` → `dataclasses.FrozenInstanceError`。依据是
        `utils/image/target/objective.py:323` 的 `@dataclass(frozen=True)`（且该类 docstring 就写着结果是不可变快照）。
        这条**离线可执行**：与 `test_conventions.py` 同跑 **107 passed / 2 skipped**（两处 skip 分别是硬件门控与
        沙箱里没有 ruff 在 PATH）。
      * `tests/ao_shaping/drivers/ccd/test_exposure_states.py:105` → `click.BadParameter`。依据是
        `utils/cli/exposure.py:111` 的 `def _reject(...) -> click.BadParameter`，且 `resolve_exposure` 的
        `Raises:` 段明写该类型、所有拒绝路径都是 `raise _reject(...)`。⚠️ 这条测试本身是**硬件门控**（同文件
        L45-49 模块级 skip），本机无法执行断言 ⇒ 我改为**取证于同参数同调用的离线测试**
        `tests/ao_shaping/runners/test_exposure_cli_token.py:127-128`（`resolve_exposure("0", param_hint=HINT)`
        包在 `pytest.raises(click.BadParameter)` 里），该文件与 `utils/test_cli_helpers.py` 同跑 **65 passed**。
        同时把 `import click` 放在门控**之上**（L41），这样 collect 阶段就会解析它 —— 容器里该模块按预期
        SKIPPED 而**收集成功**，说明依赖不是哑弹。
    - **新守卫** `test_pytest_raises_names_a_specific_exception`（`tests/ao_shaping/test_conventions.py`，
      root-guards 片，blocking）：AST 扫 `tests/`，匹配 `pytest.raises(...)` 且**位置参数是裸 `Exception` /
      `BaseException` 名字**；`(ValueError, TypeError)` 这类具名元组不命中。解析失败**收集并断言为空**
      （静默跳过会让下面那个计数替一个被悄悄缩小的语料背书）。下限 `_MIN_PYTEST_RAISES_CALLS = 400`
      （实测 978 / 507）。守卫不会命中自己：docstring 与注释里出现的 `pytest.raises(Exception)` 是字符串常量，
      不是 `Call` 节点。
    - **差分**：守卫先上线、测试不动 ⇒ **红**，逐条点名那 2 处（**没有一处是我注入的**；`50 deselected` 同时
      证明守卫确实被收集而非空跑）；两处修完 ⇒ **绿**。
    - **四条被普查否证的候选守卫**（数字在此，免得将来重建成假缺陷发生器）：(a) 「类内重复定义 ⇒ 方法被遮蔽」
      命中 12 处，**全是 `@property` / `@xxx.setter` 配对** —— 合法惯用法，不建；(b) 「只含 docstring / `pass`
      的空测试」**0** 处；(c) 「类级可变字面量」24 处几乎全是**约定或只读常量**（Gymnasium 环境的 `metadata`、
      `ctypes.Structure._fields_`、测试类的 `_enums_to_check` / `NOLLS` 等），硬建会一次产生约 24 个假缺陷，不建；
      (d) 「`assert` 被自己的 `except` 吞掉」**0** 处（`except Exception: pass` 虽有 111 处，但那是第 42 项已记的
      可选后端兜底问题，与断言无关）。
    - **再记两次自伤**：(1) 我用**下标**编辑文件：先插入一行 `import click`，随后仍按**旧下标**写 `L[104]` ——
      那一行此时已经变成函数 docstring，于是把 docstring 覆盖成 `with` 行，制造出两个叠在一起的 `with`
      （`IndentationError`）。处置：`git checkout --` 回到 HEAD（并 grep 计数确认第 39 项的 `r"""` 修复仍在，
      不是被回滚掉），再**只用内容替换 + `count == 1` 断言**重做。教训：**插入之后任何下标定位都会错位**，
      改文件一律按内容而非行号。(2) `echo '... site B's test ...'` 单引号里带撇号，命令在引号处
      「寻找匹配的 `' 时遇到了未预期的 EOF」被截断（本会话第三次同类）—— echo 里不得出现撇号与反引号。
    - **已知边界（不假装类别已封死）**：若某处**确实故意**接受任何异常（只断言「必须抛错」），零容忍会挡住它；
      目前这样的用例为 0，真要豁免得引入显式标记，届时再定，不预先开口子。

44. ✅ **`_fmt` 副本回潮：wt-190 的去重已入库，`273181e6` 又写回一份本地实现 —— 改走 `scripts/_common`，并把 `scripts` 分片由 advisory 转 blocking（结清第 36/39/43 项三次记欠）**（2026-10-25）：
    - **判据是守卫的颜色，不是祖先关系（第 39 项那条教训再次生效）**：`refactor/190-fmt-dedup` 尖端
      `741ea0e5` 已是 master 祖先 ⇒ 去重**确实已入库**；但
      `test_no_generator_redefines_an_extracted_helper[_fmt]` 在 master 上**仍红**。红因不是「别人没办完」，
      而是新功能提交 `273181e6`（D 块离线汇总器）在 `scripts/slm/summarize_optics_change_blocks.py:386`
      **重新引入**一份本地 `def _fmt`。也就是说：三次记欠等的条件早已满足，拦住它的是**一次回潮的重复实现**
      —— 恰好是该守卫存在的理由本身。
    - **改法沿 wt-190 既有约定，不另立第三套**：删除本地定义，六个调用点改走
      `fmt_general(v, ".5f", missing_none=_NA)`；`from scripts._common import fmt_general` 落在 sys.path
      bootstrap **之下**（与 `spgd_drift_cancel_loop.py:100` 同形；`scripts/*` 对 `E402` 是 per-file 豁免，
      无需 `noqa`）。
    - **为什么是 `fmt_general` 而不是 `fmt_metric` / `fmt_fixed`（选错就等于制造漂移）**：本地语义是
      `None → "—"`（em dash，`_NA`）加 `.5f`。`fmt_metric` 把 `None` 渲染成 `-`（连字符）且对整数与大量级
      做紧凑化；`fmt_fixed` 只处理 NaN，遇 `None` 直接 `TypeError`。只有 `fmt_general` 同时可配 `spec` 与
      `missing_none` —— 后者正是 wt-190 为第三份退役副本加的开关。
    - **等价性是量出来的，不是推的**：17 个**定义域内**取值（float / int / `bool` / `np.float16|32|64` /
      `np.int64` / `None` / NaN / ±inf）新旧**逐字节一致，0 处不同**。另有 4 个**定义域外**输入行为不同：
      `"0.5"` 旧抛 `ValueError` 新给 `0.50000`，`"abc"` / `[1.0]` / `{}` 旧抛新原样透传。六个调用点只会取
      `cv_core` 均值或 `None`，故这是**健壮性差异而非输出差异** —— 按实际口径记录，不宣称全域一致。
    - **验证落在渲染口径上**：该脚本自己的测试读 `blocks.md` 全文（L208 / 229 / 252 / 267 / 315 / 340），
      em dash 与 5 位小数单元格都在断言范围内 ⇒ **62 passed** 是**渲染之后**的一致，不只是等价性探针。
    - **有意不注册进 `MIGRATED`（已知边界，不假装完备）**：该片另一条
      `test_migrated_generator_bootstraps_the_repo_root` 要求**字面**
      `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`，而本脚本用 `for` 循环逐个
      `sys.path.insert(0, _text)`（四个候选路径）⇒ 注册会**把当前绿的测试改红**，还得顺手重构别人的
      bootstrap，越界。红守卫由 `EXTRACTED` 驱动、扫**全部** `scripts/**`，与 `MIGRATED` 无关，
      故删副本本身就已让守卫转绿。
    - **一次被否证的候选守卫（`ruff` 已拥有的东西不再自建）**：「字典字面量里重复常量键（一份静默覆盖
      另一份）」在 1238 个跟踪 `.py` 里 **0 命中**；探针同时证实 `ruff` 的 **F601** 已覆盖该类别
      （`select = [E, F]` 且全库绿 ⇒ 类别既在零也已被强制）。自建守卫就是 `_fmt` 教训换装：同一条规则的
      第二份实现会自由漂移。
    - **翻转与测量**：`run_shard "scripts" 0 0` → `0 1`，`pytest.yml` 注释块与 `AGENTS.md` 同步改写为
      「2026-10-25 起 8 片全部 blocking」。数字按 **CI 自己的包络**取（不是我第一次跑的 8 CPU / 8g）：
      `AO_CPUS=4 / AO_MEMORY=6g` 下整片 **2423 passed / 64 skipped / 0 failed**；修前同片是
      **2422 passed / 1 failed**，+1 恰是这条守卫翻绿。改完**解析该文件的守卫**也复跑：
      `test_pytest_workflow_shards_cover_tree.py` **5 passed**（每片一行、不用续行的契约仍成立，
      8 条 `run_shard` 全部被解析）。
    - **这笔翻转的真实收益**：`scripts` 片转 blocking 之后，此前**只报不拦**的第 33 项附录守卫
      （`tests/ao_shaping/scripts/test_zernike_amp_report_appendix.py`）与该片全部守卫**首次具备 CI 牙齿**。
    - **已知未证（不假装门禁已验证）**：`pytest.yml` **从未在真实 runner 上执行过**（GitHub HTTPS 出口被封），
      故本次翻转是**前瞻性配置**；本地容器全绿只是必要条件，其拦截效果要等首次真实 CI 运行才算证实。
    - **再记三次自伤（全是本会话反复踩的旧坑）**：(1) 等价性探针里 `safe(fn, v)` 没有 `**kwargs`，
      `missing_none=` 直接 `TypeError` ⇒ 前两轮「验证」其实一条都没比对成功，我却在叙述里当作已验；
      (2) 探针放在 `/tmp` 被 `/tmp/struct.py` **遮蔽标准库 `struct`**（Python 把脚本所在目录置于
      `sys.path[0]`），numpy 导入链整体崩掉 ⇒ 处置是**挪到干净目录**，不改名他人文件；
      (3) 又一次在 heredoc / echo 里混引号，触发「寻找匹配的 `' 时遇到了未预期的 EOF」，并把同一命令里
      要跑的容器测试一并带走。教训照旧：**要验证就把脚本写成文件**，别在 shell 里嵌 Python。

45. ✅ **`update()` 的「必须返回下一步」这半句被写了两次、强制了零次；补形状判定的 AST 守卫（并把一次被否证的普查反过来变成守卫的价值）**（2026-10-25）：
    - **契约的出处不是我编的**：`AGENTS.md`「Class-based Optimizer Convention」与
      `src/ao_shaping/algorithm/README.md` 都写 `update()` 执行一步并**返回**下一步状态。
      前半句到处在用，后半句**没有任何守卫**；而全仓 **36 处**调用点写成
      `update = optimizer.update(gradient)` —— 一个忘了 `return` 的具体实现会把 `None`
      无声递给这 36 处，直到下游某次下标才炸。
    - **普查数字（范围 `algorithm/` + `optimizer/`）**：**89** 个文件、**20** 个 `update`
      方法、**17** 个有值返回、**3** 个接口桩、**现行违规 0**。所以这条守卫是**预防性**的，
      不是修 bug —— 按第 42 项同一口径如实标注，不冒充战绩。
    - **它的价值恰好来自一次被否证的普查**：我先测「赋值给 `None` 返回型方法」，命中 **64** 处
      却**全是假阳性**（`np.sort` / `np.append` 返回数组，`x = opt.update(g)` 是本仓约定）。
      也就是说：**同那 36 处**调用点让按方法名反着查毫无用处，却正好让「正向查有没有返回」成立。
      否证的不是方向，是谓词的方向。
    - **豁免按形状，不按血统**：三个桩各自是 `algorithm/gradient/adam.py:115 Base.update`
      （`@abstractmethod`，体是 `pass`）、`algorithm/signal_processing/iterative_base.py:58
      IterativeOptimizer.update`（`@abstractmethod`，体是 `...`）、
      `optimizer/wfless/model_in_loop_shaping.py:760 _StepAOptimizer.update`（`Protocol` 成员，
      体是 `...`）。**不写「类继承 `ABC` 即豁免」** —— ABC 里完全可以有一个具体的
      `update()`，按祖先放行正是真实违规藏身的地方；也不用名字白名单。
    - **两个仪表陷阱被写死进守卫**：(1) `ast.walk(fn)` 会**下钻进嵌套闭包**，里面一个
      `return` 就能替一个不返回的外层 `update()` 顶罪 ⇒ 守卫用自己的栈，遇
      `FunctionDef` / `AsyncFunctionDef` / `Lambda` **不下钻**；(2) `ast.Ellipsis` 在
      Python 3.13 上**已弃用**（普查当场打出 `DeprecationWarning: … removed in Python 3.14`）
      ⇒ 桩判定比较 `ast.Constant` 且其 `value is Ellipsis`。
    - **防空转的下限**：`_MIN_UPDATE_SCOPE_FILES = 60`（实测 89）、`_MIN_UPDATE_METHODS = 15`
      （实测 20），解析失败**收集并断言为空**。守卫只扫两个已知目录，因此用 `rglob` 而不是
      `git ls-files`（与第 42/43 项不同，这是**有意**的选择，见下条）。
    - **差分：不改动任何被跟踪文件**（`rglob` 的副产品）。探针是作用域内一个**未跟踪**文件，
      内含两个类：具体的 `_ConcreteNoReturn.update()`（无返回）与同文件的
      `@abstractmethod` 桩。结果**只点名具体那一个**、桩不被点名 ⇒ 一次跑完同时验出
      「找得到」与「豁免不误伤」两半；删掉探针 ⇒ 复绿；全套 root-guards 由
      **50 passed → 51 passed**（skip 数不变），收集数 +1 证明它确是被收集的用例而非死函数。
    - **本轮一并否证、不再重建的候选类**（数字留在此）：
      (a) `field(default=<可变字面量>)` 在 1238 个 `.py` / 450 个 dataclass 里 **0 命中**，
      探针也证实 `ruff` **不管**它，但仍**不建守卫**：常见写法 `x: list = []` 在 dataclass
      里**类创建时就抛 `ValueError`**（语言自己拦），剩下这一种冷门拼写在仓里一次都没出现，
      纯推测性守卫不值一次收集成本；
      (b) `.format()` / `%` 的占位符与实参不匹配 —— `ruff` **已拥有**（F524 缺参、F522 多余命名
      实参，`%` 侧 F504/F507），全库绿 ⇒ 类别既在零也已被强制；
      (c) 丢弃型变异 `d.get(k, []).append(…)` 与链式 `or`（`if m == "a" or "b":`）各 **0 命中**，
      且检测器由**正对照**证明是活的（不是空扫）；
      (d) 相位自包裹 `np.mod(…, 2π)` 在 `src/` 有 **20** 处，但**该类别在 AST 层不可判定** ——
      契约禁的是**生成器**自包裹，而驱动、仿真后端、显示渲染器和 `phase_wrap.py` 本身都合法包裹，
      守卫会第一时间命中那个唯一合法包裹点自身 ⇒ 不建；
      (e) WFS 的 µm 契约：三处 `get_zernike` 定义**都**在 docstring 里写明 µm；仿真侧算的是
      `z_rad * (wavelength_nm * 1e-3) / (2π)`（弧度→waves→µm，正是 `um_to_waves` 的逆），
      硬件侧从 `WFS_ZernikeLsf` 填的是 `zernike_um` ⇒ **孪生与硬件一致**，无缺陷。
    - **再记四处自伤（三处是旧坑的第四次变形）**：(1) 我断言「硬件驱动的 docstring 没写单位」，
      依据是 `git grep -A10` 的**截断窗口**——单位其实在 docstring 的「注意」段里，是整段读取的
      AST 普查把我证伪的；这是「截断冒充证据」在本会话的第三次（前两次：`| head -8`、空输出）；
      (2) 我把路径猜成 `drivers/sim/simulated_wfs.py`（真实是 `drivers/sim/wfs/simulated_wfs.py`），
      空输出差点被我读成「没有实现」——**路径错配的空结果不是证据**；
      (3) 我未测量就断言「本文件没有 `E501` 豁免」，实测 `E501` 在 `ruff.toml` 的**全局 `ignore`**
      里且全仓**没有 `line-length` 设置**（94 列探针直接通过）——对工具行为的断言要先探针；
      (4) 两处计数仪表错：`git grep -c` 在**零命中时不打印任何东西**，于是我的
      `um_to_waves=<次数>` 列显示空白而不是 0；账本加粗配平检查先按**空行**切段（条目正文里根本没有
      空行 ⇒ 48 行只算出「1 段」），改成按 `- ` 切段后 `BULLET = r"^- \*\*"` 又因**缩进**对不上，
       连既有的第 42/43 项都报出 0 个 bullet。两处都只因数字看着不对才被抓住。

46. ✅ **`ThorlabWFS` 自己重算了 `calc_n_zernike_terms`，而它的基类 docstring 明写「只保留唯一一份实现」——删副本 + 按「名字与形状」判定的委托守卫（并修掉一处描述已被删除之对手的 docstring）**（2026-10-25）：
    - **缺陷本体**：`drivers/wfs/thorlab_wfs.py:1490` 的 `@staticmethod calc_n_zernike_terms` 直接返回
      `(n + 1) * (n + 2) // 2 + 1`；而它的基类 `drivers/wfs/base.py:66` 早已写成 canonical + 1 的
      **委托**，docstring 明写「这里只保留唯一一份实现，避免两者在无人察觉的情况下各自漂移」，
      `drivers/wfs/AGENTS.md` 的「`calc_n_zernike_terms` 的 piston 偏移」一节又把该 `+1` 关系钉给
      `test_base_wfs.py`。⇒ 子类是**第三份**实现：与基类恒等，却各写一遍。
    - **成因是收口没走到子类**：覆盖来自 `588415a4`（早期），基类的单一实现声明来自 `3e63680f`（较晚）。
    - **孪生与硬件此前靠巧合一致**：`SimulatedWFS` 不覆盖（走基类委托），硬件驱动自带公式 —— 两边数值
      相同纯属巧合，改基类而不动驱动就会分裂。删掉覆盖之后**结构上**只剩一份实现。
    - **删除是行为保持的（量过，不是推断）**：两处签名与装饰完全相同（`(n: int) -> int`，均
      `@staticmethod`），基类体 `canonical(n) + 1` 与被删表达式恒等；调用点
      `self.calc_n_zernike_terms(zernike_order)`（`thorlab_wfs.py:1530`）与
      `ThorlabWFS.calc_n_zernike_terms(n)`（`test_base_wfs.py:166` / `:174`）都按继承解析。
      钉该关系的测试 **131 passed**；全部消费者（`optimizer/wf` + `tools/wfs` + `drivers/wfs` +
      `drivers/sim`）**640 passed / 14 skipped**（skip 全是硬件与可选 `rl` 组）。
    - **新守卫 `test_term_count_entry_point_is_not_reimplemented`**（`tests/ao_shaping/test_conventions.py`，
      root-guards 片，blocking）：谓词是**名字 + 形状**，不是白名单 —— canonical 之外任何
      **定义** `calc_n_zernike_terms` 处，体内必须**调用**一个以该名结尾的函数（向后兼容 re-export 的
      形状正是如此，`base.py` 与 `matrix_utils.py` 都这样通过），重算算术的副本过不了。并且**断言
      canonical 自身仍存在**（入口被改名或搬走时守卫必须失败，而不是空过），解析失败收集后断言为空；
      下限 `_MIN_TERM_COUNT_SCOPE_FILES = 600`（实测 716）、`_MIN_TERM_COUNT_DEFS = 3`（实测 3）。
    - **差分**：探针是作用域内一个未跟踪文件，体内重算表达式 ⇒ 守卫**只点名探针**，同名的
      `base.py` / `matrix_utils.py` **不被点名** —— 一次同时证明「找得到违规」与「谓词按形状而非白名单」；
      删探针 ⇒ 复绿；`test_conventions.py` 由 **51 → 52 passed**（skip 不变）。
    - **两处刻意保留的作用域边界（不假装类别已封死）**：(1) 三角数表达式 `(x+1)*(x+2)//2` 全仓 **22** 处
      （canonical 之外 21），但**按公式建守卫会一次造出十几个假缺陷** ——
      `scripts/slm/hw_gs_square_scans.py:709-710` 是拿该公式**校验 canonical** 的断言，报告生成器里则
      是表格算术；(2) `tests/ao_shaping/optimizer/wf/test_zernike_response_matrix.py:10` 有一份本地
      重算（循环式）—— **测试里的独立算例按本会话口径不算第二份生产实现**，故作用域排除 `tests/`。
    - **一处我自己的前提被证伪（记下来免得重犯）**：原本打算守卫「全仓禁止 `import aotools`」，实测
      `aotools>=1.0.7` 是**已声明依赖**（`pyproject.toml:76`），且 `src/` 里有 **3 处合法使用**
      （PSF、湍流相位屏）；红线禁的是**重复实现 Zernike 数学**（原文点名的是 `RZern` 与 `noll2nm` 表），
      从来不是使用 aotools ⇒ 照原方案上线会在 3 处正常代码上变红。
    - **顺手修掉的过期 docstring**：`zernike_calc.py` 的 `ZernikeGenerator.noll_to_nm` 曾警告「本约定与
      `optimizer/wf/ga_zernike.py`、`optimizer/wf/rms_by_zernike.py` 里的 hardcoded lookup table 不同」——
      实测这两个文件对 `noll|NOLL|(2,0)` **0 命中**，且都 `from ...zernike_calc import zernike_modes`
      （`ga_zernike.py:35`、`rms_by_zernike.py:42`），而 `noll_to_nm_legacy` 在代码里**已不存在**
      （只在 `AGENTS.md` 与两份文档里作为历史记录留存）⇒ 参考实现自己在描述一个**已被删除的对手约定**，
      改为陈述单一事实源。
    - **再记一处仪表自伤（本会话第五类）**：我用 `ast.walk` 统计「重复定义」时**不记录作用域**，于是
      `zernike_calc.py:118` 的模块级 `noll_to_nm` 与 `:366` 的 `ZernikeGenerator.noll_to_nm` **方法**
      被算成两处 ⇒ 我一度准备报「canonical 模块内部存在遮蔽式重复定义」。真相是两个命名空间，无遮蔽。
      教训：**定义类普查必须同时记录所属作用域**，否则「MULTIPLE」这个标记是在撒谎。
    - **明确不动、需要人来判断的物理问题（不在最小修复范围）**：
      `runners/slm/zernike_matrix_runner.py:184/633` 用 **canonical 版**（`calc_n_zernike_terms(10)` = 66）
      作为 WFS 项数，而 `optimizer/wf/zernike_response_matrix.py:197/432/502` 用
      **`wfs.calc_n_zernike_terms(...)`**（67，含 piston）—— 两处读起来像同一个 WFS 边界却差一项 piston；
      两个数值**都已有测试各自钉住**，且它紧邻硬件系数向量的尺寸（正是当年 1.88× 单位 bug 的那条边界）
      ⇒ 这需要设备侧口径的判断，不是我能靠最小改动替它决定的。

47. ✅ **`calc_n_zernike_terms` 的 `+1` 被多处文档讲成「piston 项」，实测是 DLL 1 基填充留下的 0 号空槽 —— 改掉 10 处措辞（含我上一条提交里自己抄来的那句），孪生与驱动差一格的分歧记为待决**（2026-10-25）：
    - **触发点是我自己**：第 46 项守卫的注释里我写了「canonical 不含 piston / WFS 的 `+1` 是
      slope 拟合带上的那一项」—— 那是**照抄** `base.py` 与 `drivers/wfs/AGENTS.md` 的说法，
      没有验证。⇒ 本会话第六次仪表自伤，形态是**把文档当事实**。
    - **实测（容器、无硬件，四个量并列）**：
      | order | canonical | `BaseWFS` | `ThorlabWFS` | 孪生 `_n_modes` |
      |---|---:|---:|---:|---:|
      | 0 | 1 | 2 | 2 | 1 |
      | 2 | 6 | 7 | 7 | 6 |
      | 4 | 15 | 16 | 16 | 15 |
      | 10 | 66 | 67 | 67 | 66 |
    - **「piston 说」站不住（三条仓库内独立证据）**：(1) canonical **自己的 docstring** 写着
      `Number of Zernike terms (including piston)`（`zernike_calc.py:85`）；(2) 消费者把同一公式
      标为 `# 含 piston`（`tools/wfs/zernike_response.py:430`）；(3) 驱动多出的那一格被写明
      **不使用** —— 「返回 67 长数组，DLL 填 `coeff[1..66]`（index 0 未用）」
      （`tools/wfs/zernike_response.py:15`、`tools/wfs/zernike_correction.py:25`、
      `tools/zernike/common.py:8`），而 `zernike_response.py:219` 正是据此切 `w = w[1:]`。
      ⇒ `+1` 属于**数组长度**层面的 1 基填充空槽，不是第 67 个系数。
    - **危险的不是注释而是测试名字**：`test_base_wfs.py` 的 `test_canonical_excludes_piston` /
      `test_wfs_counts_include_piston` / `test_offset_is_exactly_piston` 断言的**数值关系全对**，
      错的是名字讲的故事 —— 名字会被复制、会被当作复用依据。⇒ 只改名
      （`test_canonical_count_is_the_term_total` / `test_wfs_count_is_terms_plus_one_slot` /
      `test_offset_is_exactly_one_slot`），断言一字未动 ⇒ 同容器 **38 passed**。
    - **改动面**：4 个文件 / **10** 处措辞 —— `drivers/wfs/base.py` 不变量 docstring（1）、
      `drivers/wfs/AGENTS.md` 该小节标题与正文（1）、`test_base_wfs.py`（5：模块 docstring 1 +
      类 docstring 1 + 测试名 3）、`test_conventions.py`（3：我上一条提交的守卫注释 2 + 失败消息 1）。
      替换按**内容锚点 + `count == 1` 断言**执行，命中数不为 1 就在写盘前中止（本会话早期按行号
      改文件咬过一次）。
    - **顺带证实第 46 项的删除**：上表里 `BaseWFS` 与 `ThorlabWFS` 现在**逐阶相等**（2 / 7 / 16 / 67），
      这正是「只剩一份实现」的可观测后果。
    - **未被我说服、也没有被我改掉的发现（留给设备侧口径）**：孪生 `_n_modes(10) = 66`（0 基、
      无空槽），驱动缓冲区 67（1 基、0 号空槽）⇒ 两侧对 `index 0` 的**含义**不一致，风险是具体的：
      `w[1:]` 在硬件上丢掉空槽，在孪生上丢掉**真的 0 号模式**。我没有统一它 —— 统一到哪一侧取决于
      DLL 与既有落盘语料的口径，属硬件侧决定而非最小修复；已作为 ⚠️ 待决写进
      `drivers/wfs/AGENTS.md` 该小节末尾。
    - **再记一处探针仪表错（第七处）**：第一版探针把实例方法 `SimulatedWFS._n_modes` 当**未绑定函数**
      调用（少了 `self`），`TypeError` 落在**断言**行上 ⇒ 表格**一行未印**就 FAIL，我差点把那个
      「F」当作结论。处置：改成**只打印不断言**的探针，第二版以 `SimulatedWFS._n_modes(None, order)`
      忠实读取（该方法不使用 `self`），跑完当场删除。教训：**测不出数据的 F 不是证据**；
      先让数据打印出来，再决定要不要断言。

48. ✅ **驱动把未初始化内存交给调用方，经由一个「文档说不使用」的数组槽位 —— `get_zernike` 的返回缓冲改 `zeros`；4 处同类点明确不动，且不建守卫**（2026-10-25）：
    - **缺陷本体**：`ThorlabWFS.get_zernike`（`drivers/wfs/thorlab_wfs.py:1520`）用 `np.empty` 分配
      **返回给调用方**的系数缓冲，而 DLL 只填 `coeff[1..N]`、`index 0` 不写 —— 这一点仓库内三处文档一致
      （`tools/wfs/zernike_response.py:15`、`tools/wfs/zernike_correction.py:25`、
      `tools/zernike/common.py:8`）。默认路径 `excluded_piston=True` 会 `response[start_idx:]` 丢掉 0 号槽
      （`optimizer/wf/zernike_response_matrix.py:359-366`），所以**默认不发作**；但 `excluded_piston=False`
      是**受支持的参数**，那条路径就把未写入内存当作「测到的 piston」交出去 —— 而
      `zernike_response.py:27` 早已写明 piston 不可矫正也不该出现非零假值 ⇒ 假值**看起来合理**，不会报错。
    - **改法**：两处分配改 `np.zeros`（`:1520` 的返回缓冲、`:1536` 错误路径的 `np.empty_like`），并留一行
      不变量注释说明「DLL 只填 `coeff[1..N]`，0 号槽是否定义取决于这里的初始化；孪生侧对同一数组也用
      zeros」。成本 67 个 float，可忽略；**没有**碰图像路径。
    - **纠正我自己上一轮的判断**：我曾说错误路径「会把未初始化内存交出去」。实测
      `handle_error(err)` 默认 `no_raise=False` 会 **raise**（`thorlab_wfs.py:632-633`），因此那行
      `return` 今天是**死代码**。改它的理由是「别留隐患」（若将来有人传 `no_raise=True` 就变成活的），
      **不是**修一个现行 bug —— 按现行 bug 记账就是虚报。
    - **明确不动的 4 处**（普查数字留在此，防止无据改动）：`drivers/sim/oopao_backend.py:214
      make_screens → out`、`thorlab_wfs.py:1185 get_spotfiled_image → image_buf`
      （**每帧 ~2.5 MB**，无据置零等于在相机热路径上加一次 memset）、`thorlab_wfs.py:1212
      get_spots_statics → spots_center_x / spots_center_y / spots_intensities`、`thorlab_wfs.py:1380
      get_wavefront → wavefront`（返回的是重排后的二维波前，原始 `MAX_SPOTS` 缓冲是否被完整写入未证实）。
      判据缺口：**这些缓冲是被 DLL 完整写入，还是只写 `n_spots` 前缀再由调用方截断** —— 在 AST/静态层
      **不可判定**，需要 DLL 契约或硬件实测。
    - **因此不建守卫**：按「驱动返回 `np.empty` 分配的缓冲」建零容忍守卫，会在上述 **4** 处**未证实**的点
      变红；用棘轮把它们登记进去等于把「未验证」洗成「已接受」。精确谓词（该槽位**永远**不会被写）
      静态不可判定 ⇒ 留作**具名开放项**，不预先建仪表。
    - **验证**：`parses OK` + ruff 干净；四个消费方目录（`optimizer/wf` + `tools/wfs` + `drivers/wfs` +
      `drivers/sim`）**640 passed / 14 skipped**，与改动前**逐位同数**；术语门禁 0 文件、
      `sync_report_provenance.py --check` clean、双夹注仅 `AGENTS.md` 与术语表两行许可。
      ⚠️ **诚实边界**：`get_zernike` 只有硬件路径能执行（需要 `_lib` 与 DLL），本次修复**没有离线测试
      覆盖** —— 能证明的只是「不破坏既有契约、不引入回归」，不能证明「那条假 piston 已被消除」，
      后者要等硬件会话。
    - **再记一次自伤（第八处）**：批量替换脚本的第二条锚点我写成 `handle_error(err)`，实际是
      `self.handle_error(err)` ⇒ `count == 0` 断言在**写盘之前**中止，文件一字未改（`git status` 证实）。
      这正是本会话把「先断言、后写盘」写进脚本的原因；连带一个更阴的坑：随后我跑的 `parses OK` 与 `ruff`
      是对**未改动的文件**跑的，差点被当成改动的验证 —— **验证必须在确认改动已落地之后跑**。

49. ✅ **驱动的 Zernike 数组被标成「Noll 1976」，而手册与实测都说是顺序 m 枚举 —— 改 4 处标签、撤回我自己写错的一句、并否证一个按散文关键词建守卫的候选**（2026-10-25）：
    - **仓库对自己的硬件契约是分裂的**：工具与 runner 层六处写法正确（`tools/zernike/common.py:6`、
      `tools/wfs/zernike_response.py:13`、`tools/wfs/zernike_correction.py:291`、
      `runners/slm/zernike_matrix_runner.py:34/947/1314` 均写「顺序 m 枚举，非标准 Noll 1976」），
      而驱动与设备文档写反（`thorlab_wfs.py:1507`「系数为 Noll 1976 约定」、
      `drivers/wfs/AGENTS.md:128`「Noll 1976 1-based 前 66 项」）。
    - **判据是双重确认的，不是我推的**（`common.py:6-19` 本就点名「驱动 docstring 不准确」）：
      官方手册 `roCMm` "derived from Zernike coefficient **Z[5]**"、
      `arrayZernikeUm` "indices [1..66] are used instead of [0..65]"；实测加载 (2,0)→[5]、(2,2)→[6]、
      (3,1)→[9]、(4,0)→[13]，响应矩阵 14/14 强对角。另外该标签**自相矛盾**：同一行还写
      `RoC = coeff[5]`，而标准 Noll 的 5 号是 (2,-2) 像散 —— 只有顺序 m 才让 5 号等于 defocus。
    - **孪生与驱动差两件事**（容器实测，无硬件；`SimulatedWFS` 可离线构造，这是它比驱动好在的地方）：
      | 注入模式 | 孪生落点 | 文档中的 DLL 槽位 |
      |---|---:|---:|
      | Noll 2 = (1,1) | 1 | 3 |
      | Noll 3 = (1,-1) | 2 | 2 |
      | Noll 4 = (2,0) defocus | **3** | **5** |
      | Noll 5 = (2,-2) | 4 | 4 |
      | Noll 6 = (2,2) | 5 | 6 |

      长度也不同：@n=4 孪生 15 / 驱动缓冲区 16；@n=10 孪生 66 / 驱动 67。
      ⇒ 为驱动写的 `w[1:]`、`w[5]` 用到孪生上会取到**另一个模式**，且不报错。
      孪生自己的测试也钉着 0 基（`test_simulated_wfs.py:143` 注入 Noll 4 断言 `argmax == 3`）。
    - **撤回我自己写错的一句**（第 47 项里我刚加进 AGENTS.md 的那句）：我写「runner 直接
      `um_to_waves(z0[1:])`，没有走 `tools/zernike/common.py` 那个修索引的入口」—— 实测
      `zernike_matrix_runner.py:94` 正是 `from ao_shaping.tools.zernike.common import (… measure_zernike …)`，
      而 `measure_zernike`（`common.py:389-406`）返回的就是原始 67 长数组 ⇒ 那里的 `[1:]` 是**正确**用法。
      我上一轮还怀疑那里有 off-by-one，同样不成立。**先把要断言的东西 grep 一遍再写进文档**，这一次救了我。
    - **实际改动 4 处**：`thorlab_wfs.py` 的 `get_zernike` 注意段（标签改为顺序 m，保留 RoC 与 µm 事实，
      并指回权威模块）；`drivers/wfs/AGENTS.md:128` 表行；`common.py:19` —— 我改完驱动之后它自己变成了
      **过期描述**（仍在说「驱动 docstring 不准确」），改为记录已改正；`simulated_wfs.py` 的
      `get_zernike` docstring 原文只说「按 Noll 排序」，读者会默认它与驱动一致，现明写两侧排序与长度不同
      并指向未决条目。
    - **一个被否证的守卫候选**（谓词：散文里 `Noll` 与 驱动/`coeff[`/DLL 同现即违规）：活树 9 处命中里
      多数是**正确表述** —— `thorlab_wfs.py:1510`「标准 Noll 里 5 号是 (2,-2) 像散」、
      `simulated_wfs.py:400`「本仿真按 0 基 Noll 序」、`zernike_matrix_runner.py:533/630`
      「66 = Noll 总数」（讲的是**项数**，与枚举无关）、`common.py:20`（我新写的改正说明）；
      另有 `zernike_response.py:590` 已经写了 `(非 Noll)`，是我的否定词表漏了拼写。
      ⇒ 该谓词会把刚写正确的句子标成违规，属**假缺陷发生器**，不建。散文里的枚举约定**不可判定**。
    - **留给设备侧的两处，我没有动**：(1) 孪生 0 基 Noll 与驱动 1 基顺序 m 统一到哪一侧 —— 需要 DLL 契约与
      既有落盘语料的口径；(2) `drivers/slm/AGENTS.md:131` 的实测记录「SLM-x 平移驱动 WFS Noll3 (y-tilt)，
      SLM-y 平移驱动 WFS Noll2 (x-tip)」—— 按 Noll 是 2=(1,1)/3=(1,-1)，按 DLL 槽位是 [2]=(1,-1)/[3]=(1,1)，
      **标签直接决定哪个轴对应哪个通道**，我手上没有判据说它当时用的是哪一套，故只记录、不改写实测结论。
    - **仪表自伤两处**：(1) 普查脚本读 `/tmp/tracked.z`，而那份清单**已被我在上一轮末尾删掉** ⇒
      `FileNotFoundError` —— 缺失输入不等于「0 命中」（本会话第 9 处）；(2) 我在两条 grep 结果下写了
      「^ empty = …」而输出**根本不空**（`from ao_shaping.tools.zernike.common import (` 就在下一行）——
      **给尚未看到的结果预先写解说，本身就在制造假结论**。
    - **验证**：容器把 `test_conventions.py` + `drivers/wfs` + `drivers/sim` + `tools/wfs` + `optimizer/wf`
      合在同一次运行 ⇒ **692 passed / 15 skipped**（skip 全是硬件与可选 `rl` 组）；术语门禁 0 文件、
      `ruff check .` 全绿、`sync_report_provenance.py --check` clean、双夹注仅 `AGENTS.md` 与术语表两行。
      ⚠️ 诚实边界：枚举正确性依据是**手册引文 + 既有实测**，本次**没有重新上实验系统** —— 「14/14 强对角」
      是引用，不是本次复现。

### 8.10 `assert` → 驱动层：三条护栏

| 项 | 为什么不能 |
|---|---|
| 不要把 ~40 条内部不变式一并改成 `raise` | 它们校验的是本函数刚算出来的值，外部到不了；改 `raise` 只会抬高运行开销并让 traceback 变噪声 |
| 不要动 `spots_calc` 那 3 条 | 被测试锁成 `AssertionError` 契约（见 8.9 前置 1） |
| 不要把 `--max_peak` 搬进驱动 | 它是策略不是机制，且已明文声明不是联锁（见 8.9） |

### 8.12 `__main__` demo blocks in library modules (#90 / R-68) — 2026-10-14 实测

AST 全量扫描（`src/**`，排除 `__pycache__`）：**10 个** ≥10 行的 `if __name__ == "__main__":` 块（issue 说 7）。逐个核实结果一致：

| 文件 | 块行数 | `__main__` 守卫 | 裸模块级语句 |
|---|---|---|---|
| `optimizer/wfless/slm_zernike_shaping.py` | ~~119~~ | 1 | **0**（← 2026-10-20 该文件已整体删除（#138），本行为 2026-10-14 的历史测量留档） |
| `optimizer/wfless/slm_square_shaping.py` | 113 | 1 | **0** |
| `optimizer/rl/rl_wfs.py` / `lr_wfs.py` / `lr.py` | 64 / 60 / 58 | 各 1 | 各 **0** |
| `algorithm/signal_processing/sim.py` / `wavefront.py` | 48 / 45 | 各 1 | 各 **0** |
| `optimizer/wfless/bayes_opt_lr_delta.py` | 33 | 1 | **0** |
| `optimizer/wfless/adc_dm_adam.py` | 29 | 1 | **0** |
| `optimizer/wf/interaction_matrix.py` | 21 | 1 | **0** |

⚠️ **判定：#90 是整洁性问题，不是正确性问题。** 与 #81（R-61）性质不同 —— #81 的三个
`rl/` 文件**根本没有守卫**，`import` 即训练+弹图；此处 10 个文件**全部已有守卫**且
**零裸模块级语句**，即 `import` 不会执行任何 demo 代码。因此「库模块 import 不应有副作用」
这条红线**并未被违反**，不应按 #81 的紧急度处理。

剩余动作只是把守卫块内的 demo 内容**搬到 `scripts/`**，属搬移而非修复。最大两块
（119 / 113 行）另有既有裁决：`slm_zernike_shaping.py` 的 fork `__main__` 见 #74
（判「删，不移植」）。**未实施** —— 纯搬移无行为收益，且需逐块核对是否有其它代码依赖。

> ✅ **2026-10-15 已实施 6/10（`14b1cda`，#128/#90 决策）**：上表 10 个文件中 6 个库文件的
> `__main__` 块已搬至独立 demo 脚本（docstring 注明源模块）：
> `slm_square_shaping.py`→`scripts/slm/demo_square_shaping.py`、
> `bayes_opt_lr_delta.py`→`scripts/slm/demo_bayes_lr_delta.py`、
> `adc_dm_adam.py`→`scripts/dm/demo_adc_dm_adam.py`、
> `interaction_matrix.py`→`scripts/wfs/demo_interaction_matrix.py`、
> `sim.py`→`scripts/signal_processing/demo_wavefront_sim.py`、
> `wavefront.py`→`scripts/signal_processing/demo_wavefront_reconstruction.py`；
> 顺带修 2 个被 `__main__` 掩盖的 bug（adc_dm_adam F821 `sample_num` / sim.py numpy-2
> `ptp()`）。`slm_zernike_shaping.py` 的 fork 块已按 #74 裁决删除（`ee96acfe`，删不移植）。
> **剩余**：仅 `rl/{rl_wfs,lr_wfs,lr}.py` 3 个块（64/61/59 行）—— 涉及 RL 训练循环，
> 搬移前仍需逐块核对依赖。


### 8.13 2026-10-23：容器门禁 7 处红清到全绿（顺带挖出一个真·单一事实源违例）

起点是 4 处**长期红**、先前被记作「master 既有、与本次改动无关」的容器失败；收尾后连同 #173/#176
新引入的 3 处，共 **7 处全部清掉**。**教训：「既有失败」不是免检理由** —— 其中 1 处拦着的是一件真事。

| 红点 | 真因 | 处置 | commit |
|---|---|---|---|
| `test_device_mutex::test_the_old_names_are_the_same_objects` | 🔴 **兼容再导出被本地定义覆盖**：`run_hw_sweep_groups.py` 既 `from ...device_mutex import`（:228）又在下文**自己再定义同名** 7 个（`probe_devices` / `format_probe_line` / `resolve_probe_cmd` / `OWN_PIDS_ENV` / `PROBE_CMD_ENV` / `PROBE_IDLE` / `PROBE_NOTE_PREFIX`）。来由：驱动先有自己的实现（`251b2328`），权威版后落（`ff46321d`），接导入时**没删旧定义**（`6f3aff24`）。6 个逐字节等价，**`probe_devices` 已漂移**：本地版返回 `ProbeResult(status=PROBE_IDLE)`，权威版返回 `source=PROBE_SOURCE_SDK` | 删 7 处残留定义（135 行，纯删除）；漂移当前**不可观测**（`format_probe_line` 不把 `source` 写进协议行，父进程在 `device_mutex.py:1191` 重新盖章），但兼容入口从此不再跟随权威实现，且权威 docstring 的第三条硬约束（返回前释放句柄，否则下次开锁 -10003）对旧脚本失效 | `8e451110` |
| `test_cli_argv_binding::test_santec_and_camera_call_sites_are_stable` | 设备互斥特性给 census 加了 1 Santec + 2 camera（`kernels/device_mutex.py`；其中一条是 module-docstring 用例，按既定口径计入） | 沿既有溯源链**显式**重钉 30→31 / 28→30，并补两份逐文件字典条目 | `55b1187f` |
| `test_device_mutex` 两处 PID 树测试 | 测试用 `os.getppid()` 写持有者元数据；容器里 pytest 常是 PID 1 ⇒ `getppid()==0`（父在命名空间外），「持有者是我的父进程」这一前提在测试进程内**不存在**（实现本身是命名空间安全的：进程表 `parent=0` 即 `break`，`is_held_by_me` 对 `pid<=0` 直接 False） | 把该条挪进**真子进程**里判（其父进程就是测试进程 ⇒ 命名空间内、确定可验）；并补上用例名里写着、**原本根本没测**的后代条；祖先条保留条件判断并写明为何在容器里是空真 | `15da456e` |
| `test_common_helpers_not_reintroduced[_fmt]` | `compare_sim_vs_hw.py` 的 `def _fmt` **不是**五个既有格式化器的副本，而是**第六个**：`None→—`、`±inf→∞/-∞`、`g` 规格。17 个探针值实测，五个没有一个逐字节相同（`fmt_general` 同 `g` 规格但输出 `inf`/`None`，`fmt_metric` 输出扁平 `-`） | 提取为 `scripts._common.fmt_sigfig`（进 `__all__`），59 处调用点改名；等价性用「被删函数体 vs 新函数」在探针集与 `digits=3..6` 上逐字节证明。同时把该生成器纳入 `MIGRATED` 契约（要求具名内联 repo-root bootstrap —— 也正是 ruff E402 豁免的那种写法），防止再漂 | `b6ee43f1` |
| `test_no_script_hardcodes_a_machine_absolute_path` | `scripts/slm/fix_report_links.py` 写死 `D:/Projects/TIFO/...`；且未登记 `scripts/README.md`、无引用、就地改写报告 | **删除**（不是修）：链接完整性已由 `test_report_relative_links_resolve` 机器化守护，留着一条更坏的改链接路径是隐患 | `395d7623` |
| `test_report_relative_links_resolve[gs_issue_live_20261015.md]` | 正是上面那个脚本的幂等守卫只跳过 `../figures/`、漏了已正确的 `figures/` ⇒ 15 条链接变成 `../figures/figures/…`。正确目标是 `report/shape_trials/figures/`（由 `generate_shape_trial_profiles.py:105 FIG_DIR` 决定；`gs_refine/figures/` 下有**同名但字节不同**的另一套，靠猜必错） | 15 条改回 `../figures/…` | `395d7623` |

| `test_gs_refine_target_shapes::test_default_gs_phase_is_bit_identical_to_the_pre_change_golden` | 🔴 **与生俱红**（2026-10-24 查明，非本轮改动引入）：写下该测试的提交 `d2d0c2b0`（10-23 20:52）当场就是 **7 failed / 38 passed**；到 merge `1de0104c` 只剩它一条还红。期望金标 `c94410b9…` 自写入起**从未被改动过**，而当前实际值 `ebd0b18e…` **在全历史中从未出现过** ⇒ 不是「重新基线又被回退」，是**金标当初就对不上出货代码**。**根因：仓库没有 pytest CI**（AGENTS.md MISSING INFRASTRUCTURE 已自陈），所以一条与生俱红的测试能安静躺一天以上无人知 | ✅ **已修（`cff3a49c`）**：跨提交重算证明**数值从来没变** —— 同一配置在 `6a5e1067`、`6a5e1067^`、`d2d0c2b0`、merge `1de0104c` 与当前 master 上**全是** `ebd0b18e…`；再扫 **2000 组** `(n_grid, far_field_padding, target_side_px, n_iters, seed)` 组合，**无一**产出 `c94410b9` ⇒ 旧字面量**无法从本仓库任何状态复现**（抄错或凭空写的），**不是回归**。据此把金标重钉为 `ebd0b18e…`，并把「加三个新字段没有改变默认数值」的依据从「来历不明的常量」换成「**跨提交重算相等**」；测试改名 `test_default_gs_phase_matches_the_pinned_golden`（原名的 `pre_change` 暗示了它从来没有的来历），docstring 记下证据与可直接重算的代码。该文件现 **45 passed** | `cff3a49c` |

> **2026-10-24 全量门禁实测**（默认发现集，不带 `--hardware`）：`AO_MEMORY=48g` 下
> **8953 passed / 363 skipped / 3 failed**，2185 s。⚠️ 顺带测出容器脚本默认的
> `AO_MEMORY=16g` **跑不完全量**：在 34% 处被 SIGKILL（**exit 137**，不是测试失败，
> 现场没有任何 summary）⇒ 全量必须抬到 48g（宿主 751 G，无压力）。
> 三条红**在 merge `1de0104c` 上当时就已红**（本轮在同一提交上重跑验证），与本轮改动无关，
> 且全是**自家防重复守卫拦住新代码**：
> ① `test_help_text_matches_golden` —— `slm-gsnet` 的 `--help` 由 `COMMAND [ARGS]`
> 变 `[COMMAND] [ARGS]`（group 子命令变可选），金标未重钉 ⇒ **wt-193 在办，勿抢**；
> ② `test_no_generator_redefines_an_extracted_helper[_fmt]` —— `scripts/slm/summarize_optics_change_blocks.py`
> 把 `_fmt` 又写了一遍（#173 已收敛为 `scripts/_common` 五个具名格式化器）⇒ **wt-190 在办，勿抢**；
> ③ `test_no_bare_compute_metrics_is_defined_anywhere` —— `scripts/slm/sweep_target_size.py`
> 重新定义了裸 `compute_metrics`（#69 当年正因三个不同契约同名才改名）⇒ **已修（`fca1ed88`）**：
> 改名 **`compute_size_metrics`**（与同文件的 `SizeMetrics` / `best_balance_size` 同一词汇）。
> ⚠️ 判过"是不是三个 canonical 之一的副本"——**不是**：它吃**预建掩膜 + 基线帧**
> (`arr, base_arr, box, core, zero_disk, ref_mask, noise_floor, k`)，输出带 `fill` / `valid`
> 防退化门限，且**空核心区显式记 `1e3` 最差值**；而 `compute_bench_metrics` 吃
> `(intensity, target, center=)`、`uniformity_cv` 在空支撑域返回 **`+inf`**（口径不同）。
> 它写的 `sqrt(mean((v/m-1)²))` 与 `std/mean` 是**同一量的另一种写法**（非重复实现）。
> ⇒ 按守卫自己的指示「用 canonical 或**另起显式名**」走改名，不删不重接。
> def + 3 处调用全改；守卫 5 passed；脚本 `--help` 仍可跑。
> 顺带发现（**不改**）：`report/zotero_objectives/README.md:518` 仍引 `compute_metrics` L350 ——
> 那是 #69 改名**前**的 `square_shaping` 函数，报告是当时实验的记录，代码行号与旧名不追改。

> 🔴 **惰性契约的「磁盘半边」此前零执行**（本轮收口）。README/AGENTS 写「`import ao_shaping`
> 及其子包不得加载原生 SDK、不得做磁盘 I/O」：SDK 半边有机制（惰性 `_load_sdk`），
> **磁盘半边一个守卫都没有**，而且一直在被违反 —— `DeviceConfigManager.__init__`
> （`utils/io/file.py:1145`）既 `mkdir(parents=True)` 又读 `defaults.json`，而它由**五个模块级
> 单例**在 import 期实例化（`thorlab_wfs.py:193`、`santec/driver.py:276`、`daheng/driver.py:46`、
> `micro/driver.py:96`、`nlight/driver.py:76`）；再叠上 `drivers/__init__.py:28` 急切
> `from ...wfs import ThorlabWFS` ⇒ **任何** `ao_shaping.drivers.*` 的 import 都会建
> `data/wfs_configs/wfs`。后果不是理论：只读挂载下 `import ao_shaping.drivers.sim.slm_shaping_bench`
> 直接 `OSError: [Errno 30] Read-only file system: '/workspace/data/wfs_configs'` —— 这条 import
> 根本不可用（本轮给 GS 金标做跨提交重算时就是这么撞上的）。
> 修法最小且**不改语义**：`__init__` 不碰磁盘；全类唯一 mkdir 挪到写盘点
> （`save_config` 建 `device_config_dir`、`save_default_config` 建 `config_dir`）。⚠️ 那次
> `__init__` mkdir 是**承重**的 —— `save_config` 自己从不建目录，**直接删会让每次保存都
> FileNotFoundError**（所以是"挪走"不是"删掉"）。`defaults.json` 改为首次访问时加载
> （`_get_default_config()`；`save_default_config` 后把缓存置 `None` 作废）。
> 守卫 `tests/ao_shaping/utils/test_import_makes_no_disk_writes.py`（10 项）跑在**子进程**里并加
> `-B`：同进程内模块早已被 import，检查会假绿；字节码缓存本身也是写。判定范围限 repo 与 scratch ——
> 实测第一版把 **matplotlib 在 HOME 下建自己的缓存**也算违规（那不是本契约，也不是我们能修的），
> 故按 §8.10「机制只声明它真正执行的策略」收窄并在 docstring 写明边界。
> 非空性实测：同一守卫跑在改前树 `fca1ed88` ⇒ `VIOLATION … mkdir('/workspace/data/wfs_configs/wfs')`。
> 容器：drivers+utils+gui+tools+optimizer **5059 passed / 248 skipped / 0 失败**（48g，25m58s）。

> 🔴 **CI 夹具 staging 会把夹具自己埋掉**（已修）。`scripts/build/run_pytest_ci.sh` 原样是
> `rm -rf "$SCRATCH/data" 2>/dev/null || true` + `cp -r "$SEED_DATA" "$SCRATCH/data"`。两处都**静默**错：
> 容器以 root 运行 ⇒ 上一轮在 `scratch/data` 里留下 root 属主目录 ⇒ 普通用户 `rm -rf` **必然**失败，
> 而 `|| true` 把失败吞了 ⇒ 紧接着的 `cp -r src dst` 在**仍存在**的 dst 里**嵌套**出
> `data/ao-testdata/` ⇒ `/workspace/data/` 顶层看不到任何夹具。表现极具误导性：8 个 `gui/r50`
> 测试报 `CSV not found: /workspace/data/1300-5-enriched.csv`，与被测代码毫无关系（本轮先被误怀疑成
> 自己刚改的 `utils/io/file.py`，查到挂载层才定位）。
> 修：删不掉就用一次性 root 容器删，再不行**大声 FATAL 退出**，绝不带污染 stage 继续跑；拷贝改为
> `cp -a "$SEED_DATA/." "$SCRATCH/data/"`（拷**内容**，与 dst 是否存在无关）；再加顶层条目数校验
> （seed 与 staged 不等、或为 0 ⇒ FATAL）。⚠️ 顺带记：`AO_TEST_DATA_REFRESH` **默认就是 1**，
> 传 `=0` 会完全跳过 staging 而复用上一轮的 `scratch/data` —— stage 坏过一次后，后续所有 `=0`
> 的运行都继续吃坏数据（本轮前半段就是这么连着红的）。
> 守卫 `tests/ao_shaping/scripts/test_ci_scratch_staging.py`（5 项，纯 shell 语义、不需 docker）：
> 先**复现** `cp -r src dst` 在 dst 已存在时嵌套，再证 `cp -a src/. dst/` 幂等，最后钉住 harness
> 用的是安全形式且清理失败不被吞。端到端实测：用 root 容器往 `scratch/data` 造一个 root 属主目录
> （正是原致坏条件）再 `REFRESH=1` ⇒ 无嵌套、root 残留被清、夹具在顶层、seed 9 项 == staged 9 项、
> `test_r50_connection.py` **36 passed**（同一条件下旧逻辑产生 8 处假红）。

> 🔴 **pytest 门禁缺失（本轮补上）**：`ci.yml` 里那个 `continue-on-error` 的整目录单进程
> 作业**不可能通过** —— 实测无目标的 `run_pytest_ci.sh` 在 `--memory=16g` 下于 34% 处被
> SIGKILL（**exit 137，无任何 summary**），而 ubuntu-latest 总共才 16 GB。这与 2026-10-20
> 已记录的根因一致（sim 后端每次测量做 7680×4800 complex128 FFT，分配器不还内存）：本轮
> 把它**定位到具体片** —— `optimizer/wfless` 整域单进程在 **6g 与 12g 下都死**，而它 55 个
> 文件**每一个单独跑都在 4g 内通过** ⇒ 症状是进程内跨文件累积，不是某个巨型用例，所以
> **抬上限无效**（12g 也死）。按手工清单分组会被"新增文件静默漏测"反噬，故给 harness 加
> `AO_TEST_PER_FILE=1`：文件清单由 `--collect-only` 现生成（天然遵守 EXCLUDES），
> **每文件一个 pytest 进程**；实测覆盖与单进程**逐项相等**（wfless 1072 passed + 2 skipped
> == 1074 collected；整个 optimizer 1259 + 3 == 1262），代价约多几分钟。
> 新工作流 `.github/workflows/pytest.yml`：一次镜像构建 + 8 片，`AO_CPUS=4 / AO_MEMORY=6g`
> （留头寸），**6 片 blocking / 2 片 advisory**（`scripts` 的 `_fmt` 归 wt-190，
> `runners-gui-misc` 的 `slm-gsnet --help` 金标归 wt-193，两者落地即转 blocking）。
> 分片清单只列**目录**。守卫 `test_pytest_workflow_shards_cover_tree.py`（5 项，正则解析
> 而非 YAML —— 本仓不依赖 PyYAML，跑不起来的守卫不是守卫）**两向**钉：每个顶层测试区必须被
> 覆盖 / 每个分片目标必须存在 / 至少要有一个 blocking 片。⚠️ 它当场抓出我手写的清单**漏了
> `tests/ao_shaping/test_config.py`（81 项）**，还抓出我一度把分片写成 8 个文件的显式清单。
> 非空性实测：删掉一行 `run_shard` ⇒ 守卫报出被删片覆盖的全部区域名。
> ⚠️ 顺带修掉 harness 两个自找 bug：(a) 我把 `PYTEST_BASE` 里含 `python -m pytest` 直接喂给
> 容器内循环 ⇒ 变成 `python -m pytest python -m pytest …`，pytest 把 `python`/`-m`/`pytest`
> 当测试目标 ⇒ "no tests collected"；解释器与选项已拆成两个数组。(b) 首版 `sed -n "s/::.*//"`
> 缺 `p` ⇒ 收集到 **0 个文件却 exit 0**（假绿），已加 `p` 并补**空清单即 FATAL**；
> `--collect-only` 给的是**相对 rootdir** 的路径而容器 CWD 是 scratch ⇒ 必须重新锚定绝对路径。
> 本机验证：YAML 可解析、内联 shell `bash -n` 通过、stub 回放证明 blocking 片失败会
> 让作业红而 advisory 片不会；45 项 CI/门禁相关守卫全绿。⚠️ 未做：Actions 真实运行
> （本机 GitHub HTTPS 出站被拦），故工作流是按上述实测**构建**的，非 CI 实测通过。

**门禁状态（推送树 `b6ee43f1`）**：`ruff check .` 在 `0.11.10` 与 `0.15.12` 下均 `All checks passed!`；
`--extend-select RUF100` 干净；术语门禁 `合计: 0 个文件, 0 行`。

✅ **曾发现（2026-10-23，非当时改动引入，已证明；现已修）**：
`tests/ao_shaping/drivers/ccd/` 在**未含任何本轮改动**的同一提交上就是 **6 failed + 1 error** —— `daheng/test_reset_exposure_time.py` 5 条 + `test_miicam_auto_exposure_report.py` 1 条，外加 `test_miicam_report.py` 的 setup error。根因：测试仍按 `daheng_driver.gx.GxFrameStatusList.SUCCESS` 取**模块级 SDK 属性**，而惰性加载契约已把 SDK 收进 `_load_sdk()` ⇒ `AttributeError: module ... has no attribute 'gx'`。因为 `addopts --ignore=tests/ao_shaping/drivers` 把它排除在**发现集**之外，CI 从来没看过这组测试的状态。证明方法（可复现）：`git worktree add --detach /home/ws/wt-verify HEAD` 后在同一个容器 runner 里跑同一条命令，得到**完全相同**的 6+1（本轮改动未提交时 HEAD 即改前状态）。⇒ **已修**（见下方"CCD SDK 缺失"条目）：测试改为注入假 SDK 句柄（不恢复模块级 `gx`，那会破坏惰性加载红线），并把**真需要相机**的两条报告测试改为按驱动自身的可用性标志跳过。

**CCD「SDK 缺失 → 看不懂的崩溃」已修**：`drivers/ccd/daheng/driver.py` 顶层 `try: import gxipy as gx / except: logger.error(...)` 在失败分支里**根本没有绑定 `gx`** ⇒ 无 SDK 的机器上 `get_numpy_image` 在 `gx.GxFrameStatusList.SUCCESS` 处抛 `NameError`；`miicam/driver.py` 绑了 `None`，于是在 `miicam.Miicam()` 处抛 `AttributeError: 'NoneType' object has no attribute 'Miicam'`。两者现都：失败分支显式绑定（daheng 另存 `_SDK_IMPORT_ERROR` 供报错），并在 `initialize()` 第一行经**单一** `_require_sdk()` 抛 `CameraError`（带可行动信息：需要 `libs/gxipy` 在 PYTHONPATH 上）。`try/except` 本身不动 ⇒ 惰性加载红线保持不破。测试侧：`test_reset_exposure_time.py` 5 条改为用 `monkeypatch.setattr(driver, "gx", fake_gx)` 注入假 SDK 令牌（不再依赖真实 SDK 是否存在），`test_miicam_report.py` / `test_miicam_auto_exposure_report.py` 按 `miicam is None` / `_MIICAM_AVAILABLE` 跳过（沿用 `requires_oopao` 与 `test_wfs_report.py` 既有写法，未删任何断言）。实测 `tests/ao_shaping/drivers/ccd`：**198 passed / 25 skipped / 0 failed**（修前 189 passed / 23 skipped / **6 failed + 1 error**）。

✅ **发现集已按实测彻底取消 drivers 排除（`7d10f7d4` 起，`a8e1423d` 收口）**：这组测试腐烂之所以**长期不可见**，是因为 `addopts` 与 `run_pytest_ci.sh` 都把整棵 `tests/ao_shaping/drivers` 排除在发现集之外。现两处的 drivers 排除**已全部撤销**，各域实测**绿且快**：`ccd` 204/25、`sim` 394/2（63.33 s）、`wfs` 110/11、`slm` 158/127（7.95 s）、`adc` 49（2.84 s）、`hardware` 37 skipped（0.06 s）、`dm` 139/5（**40.22 s**）、根文件 13/9。裸 `pytest` 收集数 **8011 → 9147 → 9293**（实测；9293 = 9147 + 本会话新增 2 条门禁测试 + `dm` 144）。整棵 `drivers` 单 session **1067 passed / 216 skipped / 124.43 s**。

🔴 **上一提交把 `dm` 的成因写错了，此处更正**：我写的是"2 个测试在无控制器应答时**失败而非跳过**，修好 fixture 的跳过分级是前置条件"。实测（`--durations`）两条真因都不同：
1. 那 2 条失败其实是 **`TestReport` 写 `docs/micro-dm/` 撞上只读挂载**的 `OSError: [Errno 30]`（`test_micro_dm_report.py:56` / `:331`），与"会不会跳过"无关 —— 它们在 `--hardware` 下写这份入库报告本就是设计意图（`docs/micro-dm/*` 已在 `.gitignore`）。
2. 真正的缺陷是 **`hardware` 标记在 `dm/` 下无人执行**：跳过钩子只存在于 `drivers/slm/conftest.py` 与 `drivers/hardware/conftest.py`，而 `pytest_collection_modifyitems` 是**会话级**钩子 —— 于是 `pytest drivers/dm drivers/slm` 会跳过、`pytest drivers/dm` 单独跑就不跳过。**同一条测试的跳过与否取决于同会话里还收集了哪棵无关子树**。后果不是观感问题：`TestMicroDMReport` 的 fixture 因此把 26 个不可达控制器（192.168.0.101-126）逐个扫到底，**每个测试 260 s 的 setup，两条合计 520 s，占掉该域 560 s 的绝大部分**。

⇒ 修法是**把门放到测试真正所在的层**：新增 `tests/ao_shaping/drivers/conftest.py`（复用上一提交建的 `_hardware_gate` 单源），使 `--hardware` 门对整个 `drivers/` 一致生效。实测 `dm` 由 **560.73 s / 2 failed** 变为 **40.22 s / 0 failed**（`SKIPPED [2] ... need --hardware option to run`）。剩下的 27 s 是 `test_dm.py::TestDMRegistry::test_list_reachable_types`，它**通过**而非挂起。

⚠️ 方法论教训（已第三次踩同一类坑）：**"慢/挂起"要先用 `--durations` 归因**，不要按域整体归因。`dm` 曾被当作"硬件驱动天然慢所以排除"，实际 93% 的成本来自**一个未被执行的标记**，一次 conftest 就消掉了。

🔴 顺带挖出**第二份排除清单**：`scripts/build/run_pytest_ci.sh` 的 `EXCLUDES` 把同样的路径重述成
绝对路径（脚本从 scratch 目录启动 pytest，pyproject 的**相对**排除在那种 CWD 下匹配不到任何东西）。
本次改动第一次提交后实测发现**收窄完全没生效**——pyproject 已改、脚本仍写整棵 `drivers`。
现由 `test_conventions.py::test_pytest_excludes_match_across_configs` 钉住两处集合相等。
⚠️ 实测纠正一处旧文档说法：`--ignore` **从不**裁剪显式点名的路径（`pytest tests/ao_shaping/drivers/dm`
在被排除时仍收集 144 个），故原表"显式指定不受影响"的说法成立但机制与直觉不同。

✅ 本条已结（见上一段）：`drivers/dm` 的 2 条失败真因是只读挂载下 `TestReport` 写 `docs/`，且该域的
560 s 来自未被执行的 `hardware` 标记；两者已由 `drivers/conftest.py` 一并解决，`dm` 已进发现集
（40.22 s / 139 passed / 5 skipped）。

🔴 **同一收窄顺带暴露了 CI 侧的启动期崩溃**：`drivers/hardware/conftest.py` 与
`drivers/slm/conftest.py` **各自**注册 `--hardware` —— 前者无条件、后者包
`try/except ValueError`。pytest 按**传入路径的顺序**加载 conftest ⇒ 参数顺序决定崩不崩：
`pytest drivers/slm drivers/hardware` 在启动期就 `ValueError: option names {'--hardware'} already added`，
而从 `tests` 走字母序（`hardware` 在前）恰好躲过。把默认排除从整棵 `drivers` 收窄到
`drivers/dm` 后，两棵子树第一次进同一 session，才把它逼出来。现两处都走
`tests/ao_shaping/drivers/_hardware_gate.py`（选项、help、跳过规则**单源**，重复注册幂等吞掉）。
实测两种顺序结果**完全一致**（158 passed / 164 skipped）；`--hardware` 仍可解析；
整棵 `drivers`（dm 被排除）单 session **928 passed / 211 skipped / 86.42 s**。

⚠️ **一条核验教训（关于流程，不是代码）**：改 `n_sample` 的异常类型之前，我用
`grep -rn 'pytest.raises(AssertionError)' tests/ao_shaping/drivers/` 得到 **0 命中**，
据此判断"没有测试把异常类型锁成契约"；实际 `test_frame_averaging.py:69` 正是这样一条。
是**直接读测试源码**才拦下来的（否则会推上一个红灯）。⇒ 断言"不存在契约锁定"必须用
**两个独立工具交叉验证**（grep 工具 + 直接读文件）；单次 bash grep 的假阴性不足以支撑删除。

✅ **已收口（见`5d07ecd2`）**：两个 ruff 工作流原先钉版不一致 —— `ci.yml` 用 `uv tool install ruff==0.11.10`，而
`ruff.yml` 用 `pip install "ruff>=0.15.12"`，两者都跑 `ruff check .` 且都是硬门禁。现三处统一到**精确版 `0.15.12`**
（`ci.yml` / `ruff.yml` / `pyproject.toml` dev 组下限），方向取**上调**而非下调：`ci.yml` 原来的 `0.11.10` 在
dev 组下限 `>=0.15.12` **之下**，等于 CI 跑一个本机开发者根本不会装到的 ruff。硬门禁禁止 `>=`
（新版本能在两次推送之间无代码变化地翻红）。由
`test_conventions.py::test_all_ruff_pins_agree_and_are_exact` 钉住：三处各**恰好一个**版本、版本必须相同、
两个工作流必须是 `==`（"恰好一个"是刻意的 —— 被注释掉的旧钉版正是陈旧版本存活并被复制回来的途径）。
**已实测两版真的不等价**（下表保留为证据，勿当作待办）：

| 规则 | 0.11.10 | 0.15.12 | 说明 |
|---|---|---|---|
| `PLC0415` | **0** | **1476** | 0.11.10 里它是 **preview** 规则（`ruff rule PLC0415` ⇒ "This rule is in preview… `--preview` flag is required"，并打印 "Selection `PLC0415` has no effect because preview is not enabled"）；0.15.12 已稳定。⇒ 全局 `ignore` 里那条 `"PLC0415"` 在 0.11.10 是**空操作**，在 0.15.12 是**实际压住 1476 条** |
| `SLF001` | 1050 | 1048 | 0.11.10 多报 2 条（`scripts/slm/generate_bench_twin_report.py:443/444` 的 `_original` / `_record`） |
| `BLE001` / `S301` / `E731` / `E741` / `F841` / `F811` / `PLR0913` / `PLR2004` | 514 / 63 / 7 / 35 / 39 / 17 / 469 / 1132 | 同左 | 逐条一致 |

两条被这次测量纠正的旧说法：① §8.6 第 5 步的 "一次性 299+129+6 条" 严重低估 ⇒ 实测量级是
**BLE001 514 / SLF001 1050 / S301 63**（约 5×）；② 「CLI `--select` 会不会压过配置 `ignore`」——
实测**两版都会压过 `ignore`**（`--select F841` 在两版都报出 39 条，而 `F841` 就在 `ignore` 里），
但**不会**压过 `per-file-ignores`（flip 后 `--select E402` 在两版都是 0）。此前记的「0.11.10 上
`--select` 不能覆盖 ignore」是误判：那个 0 来自 per-file 豁免，不是 ignore 优先级。

建议单独一刀统一版本：先按项目配置在两版下重测**全部**已启用与候选规则的命中数（上表就是起点），
再决定钉哪一个；不统一之前，任何 `--extend-select` 测量都必须写明用的哪个版本。


## 来源清单

| 来源文档 | 提出日期 | 主题 | 现落点 |
|---|---|---|---|
| `TODO.md` | 2026-09-25 | `slm_zernike_pib.py` P0–P3 | 文件已删，内容 → §1 / §2 + archive §5.2/§5.3 |
| `TODO.md` | 2026-10-01 | `utils/`+`scripts/`+`tools/` 架构重构 | 文件已删，内容 → §2 + archive §5.4–§5.13 |
| `src/ao_shaping/tools/slm/TODO.md` | 2026-10-01 | `tools/slm/` 22 个探针 | **文件仍在** ⇒ §2.3 + X-2 |
| `report/fouriergsnet_pipeline/TODO.md` | 2026-10-01 | FourierGSNet 实验系统验证 | 文件仍在 ⇒ §1.2 |
| `report/slm/report2.md` §4 | 2026-09-16 | Zernike 矫正后续动作 | → §1.4 |
| `report/slm_pib_bench/pib_bench_report/report.md` | 2026-09-30 | slm-pib 实验系统下一步 | → §1.3 |
| `report/slm/bench/bench_calibration_20261001.md` | 2026-10-01 | 稳定化判据 | → H-18 |
| 代码注释 ×4 | 2026-09-17 ~ 09-25 | runner 待测清单 | → §1.1 |
| `docs/issues_report.md` §10/§11 | 2026-05-26 | 架构建议 + 新增扫描 | → F-4 / F-5 |
| `OBJECTIVE_TARGET_SHAPE_MERGE_PLAN.md` | 2026-09-26 | objective/target_shape 合并 | ✅ 已落地 → archive §5 |
| 代码结构审计（4 路并行 + 1 次分叉评审） | 2026-10-06 | 函数归属 / 重复实现 / 叶子层违规 | → **§7**（issue #50、#61–#74、#77、#78） |

**扫描范围**：`docs/**/*.md`、`src/**`、`scripts/**`、`tests/**` 中的 `TODO.md`、
`TODO/FIXME/XXX/HACK/待办/未实现/待确认` 代码注释、以及各报告文档末尾的
「下一步/建议」。**每项都已对照当前代码核实**，确认仍存在才收录。
排除：`.kilo/worktrees/`、`.venv/`、`libs/OOPAO`、`drivers/ccd/_miicam_sdk`（厂商 SDK）。

**路径订正 2026-10-05**：实验报告已从 `docs/` 迁到仓库根的 `report/`
（见 [`report/README.md`](report/README.md)）。本账本正文中所有指向报告的
`docs/<topic>/...` 引用已随之改写为 `report/<topic>/...`；
archive §4 中复述 2026-10-01 那次误覆盖事故的段落**保持原路径不变** ——
那里记录的是当时实际被写入的位置。
