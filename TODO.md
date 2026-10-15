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
**不要**再往本文件堆完成记录 —— 那是 archive 的职责，且它已经 1625 行了。

---

## 1. 需硬件（设备在线才能推进）

### 1.1 runner 待上设备实测清单

| ID | 摘要 | issue |
|---|---|---|
| H-1 | `dm-matrix` sequential/hadamard 双模式标定内核离线通过（122 passed）但未上设备实测 | [#18](https://github.com/ywzhang909/AO-shaping/issues/18) |
| H-2 | `zernike-matrix` 2026-09-16 重写后的标定内核未上设备实测 | [#19](https://github.com/ywzhang909/AO-shaping/issues/19) |
| H-19 | **ABBA 漂移对消已落地但未实机验收**（判据 `dec>0.55` 且 `late_gain≥10%`） | [#36](https://github.com/ywzhang909/AO-shaping/issues/36) |
| H-20 | `spgd-square` 合并成 `slm_shaping_runner` 后未上真机光学回归 | [#37](https://github.com/ywzhang909/AO-shaping/issues/37) |
| H-21 | `spgd-square` 的 `--delta` 语义变了（空操作 → 可钉住），需真机 A/B | [#38](https://github.com/ywzhang909/AO-shaping/issues/38) |
| H-22 | `slm-pib` 合并后未上真机回归（新增一次 `patch_sim_pib_shaping`） | [#39](https://github.com/ywzhang909/AO-shaping/issues/39) |
 | H-23 | ✅ **已实施（2026-10-14）**：三处协调改动全部落地。①`slm_shaping_runner.py::_square_json_payload` 增记 `active_modes`（用 `_active_zernike_modes(n_max, mask)`）+ `n_terms_total`；②`index.py` 增 `_Verdict.active_modes`/`HwRecordRef.active_modes` + **规则 4b**（规则 5 之前，仅当 sidecar 声明且 `len(_c)==len(active)` 才收），`_CACHE_VERSION` 2→3 且缓存 encode/decode 都带上该字段；③`transforms.py::expand_active_modes_to_noll` 用 canonical `noll_indices` 把活动模式散射回全长 Noll 向量，`records.py` 在 `zernike_coeffs_to_panel` 前调用。实测散射后 panel 与「手搭全长向量」**逐位一致**；**无 sidecar 声明的旧语料仍按 `ODD_COEFFICIENT_LENGTH` 排除**（不追溯重解释）。新增 20 测试（`tests/ao_shaping/ml/hwdataset/test_active_modes.py`），`hwdataset/` 380 passed | [#40](https://github.com/ywzhang909/AO-shaping/issues/40)（**本 commit `Closes`**） |
| H-24 | 方形家族进 `hwdataset` 会换 `fov_px` ⇒ `far_field_padding` 标定必须重扫 | [#41](https://github.com/ywzhang909/AO-shaping/issues/41) |
| **H-25** | **`--w-loggrad` 结构项已落地但未实机验收**。离线全绿（`utils/` 992、`wfless/` 626、sim 端到端 `J≠shape`）；6 类候选里只落地这 1 类，2 类早已存在、3 类否决（附实测数字）⇒ [`report/pib_loss_terms/README.md`](report/pib_loss_terms/README.md)。**⚠️ 它不能修复不收敛**（H-14 已证瓶颈是慢漂移），故默认 `0.0`。验收请在 **δ=0.05** 这类「有信号但被漂移污染」的档位做，**勿用 0.0005**（梯度信号本身在噪声下）。**待开 issue** | — |

### 1.2 FourierGSNet 真机验证（`report/fouriergsnet_pipeline/`）

> 📋 2026-10-12: 结合 `slm-gs-refine`（从未真机运行）与 gsnet 系列报告的
> **下一步尝试方案**见 [`report/fouriergsnet_pipeline/next_attempt_plan.md`](report/fouriergsnet_pipeline/next_attempt_plan.md)
> （A 离线修复 → B 探针 → C1 bake-off → C2 refine 首跑 → C3 gsnet 复测 → C4 FourierGSNet run）。

| ID | 摘要 | issue |
|---|---|---|
| H-3 | §0 前置：设备在线 + 激光开 + 无残留 `python*` + 看门狗硬超时 | [#20](https://github.com/ywzhang909/AO-shaping/issues/20) |
| H-4 | §2 smoke test 结果区仍 `_待填_`（5 步，末尾须打印 `SMOKE_OK`） | [#21](https://github.com/ywzhang909/AO-shaping/issues/21) |
| H-5 | §3 DM/WFS 实际 IP 与网段（上次 `192.168.0.101–126` 全段不可达） | [#22](https://github.com/ywzhang909/AO-shaping/issues/22) |
| H-6 | `slm-gsnet spgd` 真机光路验证验收（0 阶 `argmax` / 框内能量与 CV / 产物落盘） | [#23](https://github.com/ywzhang909/AO-shaping/issues/23) |

### 1.3 真机跑出的代码级问题（修完才能复测）

> 2026-10-01 真机 `slm-gsnet spgd` 实测结论是**随机游走**
> （`dec=0.487`，末态比初态差 35.9%，EE 流失 6 倍，0 阶峰值 225→17）。
> 以下 **6 项必须先改**再复测。来源：`report/fouriergsnet_pipeline/hardware_run_20261001.md` §5–6。

| ID | 摘要 | issue |
|---|---|---|
| H-7 | 方形路径**无 encircled-energy guard** → 任何 `w_efficiency` 权重都会被「散光换 CV」击败 | [#24](https://github.com/ywzhang909/AO-shaping/issues/24) |
| H-8 | 随机初始化是满幅 `uniform(-π,π)`，与 docstring 的 "small init" **矛盾**，且满幅起点 EE 最高 | [#25](https://github.com/ywzhang909/AO-shaping/issues/25) |
| H-9 | 方形路径**无 ABBA / 漂移对消**（三个采集点全是相邻两帧）。**2026-10-02 起成本大降**：`slm-pib` 已有可照搬的同款实现 | [#26](https://github.com/ywzhang909/AO-shaping/issues/26) |
| H-10 | `--exposure-time-ms` 默认 **80.0 ms**（本台架近饱和基线 ~0.02 ms），且传它会**关掉饱和保护**使其成死代码 | [#27](https://github.com/ywzhang909/AO-shaping/issues/27) |
| H-11 | 焦距标定常数自相矛盾：文档同写 `132940/P` 与 `7600/P`（**差 17.5×**） | [#28](https://github.com/ywzhang909/AO-shaping/issues/28) |
| H-12 | `lr=0` 时 `learning_schedule()` **静默覆盖 `--delta`**；`_param_scale` 两基不同；"freeform per-pixel" 名不副实（`np.kron` 分块） | [#29](https://github.com/ywzhang909/AO-shaping/issues/29) |
| H-13 | 可达目标需重新定义：本台架 80 px staircase 只能影响**大尺度**结构，刻 50 px 平顶方块不可达 | [#30](https://github.com/ywzhang909/AO-shaping/issues/30) |
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

> ⚠️ 该文件已从 2179 行重构到 **1592 行**，`calc_objective` 已抽成
> `ShapingObjective.__call__` ⇒ **旧账本行号全部失效**，但 bug 逐条存活。
> 🔴→✅ **`slm_zernique_shaping.py` 已不再是同源副本**（2026-10-15，X-4/#74，PR #129
> `ee96acfe` 合入）：该 1600 行分叉（`_row0`/`_log_row`/`learning_schedule`/饱和分支
> 结构 + `w_outside` 能量项 + `__main__` fork demo）已并入 `slm_zernique_pib.py`
> （`w_outside`/调试产物/ABBA 等均收敛至此），`slm_zernique_shaping.py` 现仅为
> **44 行弃用重导出 shim**（`__all__` = `SLM_WIDTH/HEIGHT/RESOLUTION`、`ObjectiveSpec`、
> `Santec`、`SlmZerniquePibConfig`、`optimize_slm_zernique_pib`），仅供旧 import
> 路径解析，**不含任何可独立修改的实现** ⇒ 下面各条改动**只需改 `slm_zernique_pib.py`
> 一处**，历史「改一处必须同时改副本」的约束已失效。副本盘点见
> [archive §5.17](docs/dev/todo_archive.md)。

| ID | 摘要 | issue |
|---|---|---|
| R-5 P1 | 目标函数 if/elif 链 → `Objective` 类族 + `GuardedObjective` | [#42](https://github.com/ywzhang909/AO-shaping/issues/42) |
| R-6 P1 | 两个搜索分支共享 `BenchSession`（序列重复，**饱和策略还不一致**） | [#43](https://github.com/ywzhang909/AO-shaping/issues/43) |
| R-7 P1 | 巨型函数拆编排器：~900 行 → `prepare_geometry`（**不可变 dataclass**，字段必须区分 `window_center_full_frame` / `reference_center_window_local`）+ `_run_spgd` / `_run_heuristic` | [#44](https://github.com/ywzhang909/AO-shaping/issues/44) |
| R-8 P1 | 硬件安全 try/finally：任何异常（相机掉线、越界、`KeyboardInterrupt`）都让 SLM 停在随机相位 | [#45](https://github.com/ywzhang909/AO-shaping/issues/45) |
| ~~R-13~~ | ✅ **收敛完成**（#46）：实测**不是 5 份相同副本，而是 4 种不同实现** —— 只有 `slm_square_shaping` 与 `pib` 逐字节相同（9 行）；`slm_zernike_pib` + `slm_zernike_shaping` **早已是「响亮失败」版**（检测 `accepts_var_keyword`、转发 `**kwargs`、并 warning 列出被丢参数），而 `pib` / `slm_square_shaping` / `sim_spgd` **仍静默吞参**（且永远不转发给声明 `**kwargs` 的 callee，即 `**config.kwargs` 逃生口失效）。canonical 落在 `optimizer/constants.py::create_optimizer`，紧邻它本就该集中管理的 `OPTIMIZER_MAP`。🎁 **额外发现**：3 个文件各自还定义了一份 `OPTIMIZER_MAP`，与 canonical **内容完全相同**（仅差类型标注）→ 一并改为导入。守卫 `test_optimizer_factory_single_home.py`（12 项）除结构检查外**锁定行为**：丢参必须被报告、声明 `**kwargs` 的 callee 必须收到传参 | ✅ |
| R-14 P2 | `metric_panel` 每 epoch 全量指标 → 加 `panel_every_n`。⚠️ **前置是产品决策**（跳过轮次留空 vs 沿用上一轮） | [#47](https://github.com/ywzhang909/AO-shaping/issues/47) |
| R-18 P3 | ⚠️ **2026-10-06 复核：原描述已过期，改为需设计决策**。①「删陈旧注释」**已完成** —— `slm_zernike_pib.py` 与死分叉里 `--init-gs` / `gs_warm` / `gerchberg` **零痕迹**。②「接入已有实现」**无物可接** —— `gs_warm_start` 只是 `runner_common.py:2100` 的**布尔 CLI flag**，GS 相位是 `slm_gs_refine.py:598` 内联调 `gs_shape`，**没有可复用函数**。③ 真正的障碍：slm-pib 每轮由 `_c` 重建面板相位，注入全面板 GS 相位会破坏该不变式；改为拟合 Zernike 系数则对**方形**目标受 AGENTS.md「n≤4 无法合成方形远场」限制 | [#48](https://github.com/ywzhang909/AO-shaping/issues/48)（**仍开放**，待决策） |
| R-19 P3 | 🟡 注入机制已统一（archive §5.34），**`BenchSession` 本体未做** ⇒ R-5~R-8 仍待办。本条是那四项的跟踪父项 | [#49](https://github.com/ywzhang909/AO-shaping/issues/49) |

### 2.2 `utils/` + `scripts/` + `tools/` 架构重构

**原前置阻断项（🔴 `utils/` 不是安全的叶子）已于 2026-10-03 完成 → archive §5.4。**

| ID | 摘要 | issue |
|---|---|---|
| R-29 | ✅ 已完成：`utils/image/display.py` 已搬到 `display/image_display.py`（commit `a9236e9`，2026-10-09），`utils/__init__.py` 的 `ImageVoltagesDisplay` / `plot_funcs` / `VOLT_HEIGHT` … re-export 已改指 `ao_shaping.display.image_display`；旧路径 `ao_shaping.utils.image.display` 现抛 `ImportError`（`test_utils_leaf_no_rendering_import.py::test_old_display_path_raises` 锁定） | [#50](https://github.com/ywzhang909/AO-shaping/issues/50) |
| ~~R-33~~ | ✅ **已由上游实现**（2026-10-14 核实）：`@with_params(MicroDMImageCollectParams, kw_name="params")` 已接线（`micro_dm_image_collect.py:465-467`，docstring 自述 "R-33: migrated to with_params"）；手写 `_resolve_ips` 已删除，`resolve_ips` 现住 `drivers/dm/micro/wiring_map.py` 并在 `tests/ao_shaping/drivers/dm/test_micro_resolve_ips.py` 有测试。**issue 的「7 个驱动内部符号导入」判据已失效**：实测剩余 6 个 driver 导入（`BaseCamera` / `MAX_CHANNELS` / `VOLTAGE_MAX` / `VOLTAGE_MIN` / `R50Controller` / `resolve_ips`）**全部在各自 `__all__` 内**，即公开再导出，driver-internal 导入数为 **0** | [#51](https://github.com/ywzhang909/AO-shaping/issues/51) |
| ~~R-39~~ | ✅ **实测后判定无需改代码**（2026-10-14）：两个子项都**不可离线收口**，issue 的指令照做会引入危害。**曝光默认值**：issue 自己写明「不能一刀切成同一个值 —— 那是台架相关物理事实，不是笔误」，需逐探针确认可用区间 = 硬件。**内存槽轮换**：实测为**两份不同策略**而非重复实现 —— 驱动 `santec/driver.py::SlotRotator` **随机**取槽且从面板 `get_displayed_memory_number()` 读当前显示槽（跨进程有效），`wfless/slm_zernike_primitives.py::SlotRotator` **顺序**递增、无设备引用（可复现）；合并会静默改变台架写入的物理槽位，正是 issue §2.3 警告的「看似可信的错数据」。第三种写法 `phase_capture.py` 的 `% 128 + 1` 覆盖 1..128（其余路径 2..125），属台架事实，未动。**已用测试钉死该差异**：`tests/ao_shaping/tools/slm/test_slot_rotation_policies.py`（6 项），使「同名两份 = 重复」不再看起来像显然的清理目标 | [#52](https://github.com/ywzhang909/AO-shaping/issues/52)（**不关闭**：仍需硬件确认） |
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

**迁移内容 (2 大类):**

| 来源 | 目标 | 说明 |
|---|---|---|
| `drivers/sim/` **27 个 .py** | `sim/` 顶层包 | 设备模拟层 (约 5,000 LOC) |
| `optimizer/wfless/strehl_sim_eval.py` | `sim/optimizer/strehl_eval.py` | `StrehlLandscape`, 无 optimizer 依赖 |
| `optimizer/wfless/sim_spgd.py` | **保留** `optimizer/wfless/sim_spgd.py` | ❌ 依赖 `optimizer.constants.create_optimizer`, 违反 sim→optimizer 禁止原则 |
| `optimizer/rl/envs.py` 中的 3 个类 | `sim/rl/turbulence_env.py` | `SimTurbulenceAOEnv`, `_BaseSimAOEnv`, `StaticAberrationAOEnv` |
| `optimizer/rl/envs.py` 中余下类 | **保留** `envs.py` | `LaserCastEnv`, `TraditionalAOEnv` (非 sim) |
| `utils/slm/gs_debug_optics.py` | **保留** | `grayscale_to_radians` 仍被 scripts/tests 使用; `idealized_far_field` 已弃用 (AGANTS anti-pattern) |

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
| F-10 | ✅ 已解决 (2026-10-13)：单因子改写其实**已于 2026-10-02 落地** (commit `284fbb5`)，`sim/AGENTS.md` 第 3/4 条与代码已对齐 (第 4 条改为「已落地」)。`report/oopao_impact/report.md` 的 1.068/2.628 是改写前旧公式 (commit `5e12fa2`) 的实测值，随公式变更现已不可复现，须重跑生成器更新 | [#57](https://github.com/ywzhang909/AO-shaping/issues/57) |
| F-11 | 🔴 **SLM 序列号三路冲突**：22030108 / 22030102 / 23020026 ⇒ 引用前必须确认并回写 `drivers/AGENTS.md` 硬件表 | [#58](https://github.com/ywzhang909/AO-shaping/issues/58) |
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
| X-1 | `docs/refactor/TODO.md` #16：「`utils/image/resample.py` **0 导入** → 删模块」 | ✅ **前提已消失** —— 现在有 1 个生产导入方（`FourierGSNet.py:42`），照原文删会打断 FourierGSNet。**且 `docs/refactor/TODO.md` 本身已不存在**（已并入本账本） |
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
| **第 1.7 批 剩余（lint 抑制层）** | **R-63**（73 条 noqa 分类 + `ruff.toml` 分层，**严格按 §8.6 的 5 步顺序**：F-14 已完成 ⇒ 第 4 步的排除已删除，可从第 1 步继续，但 RUF100 仍不得先跑）→ **R-64**（312 处无标注 `except Exception` + 111 处 `assert` 护栏；后者含硬件安全检查，`python -O` 下会消失，**应先于任何硬件复测处理**） |
| ~~第 2 批（止真 bug）~~ | ~~R-1~R-4、R-9~R-11、R-35~~ ✅ → archive §5.2 / §5.3 |
| ~~第 3 批（架构重构）~~ | ~~R-20、R-21、R-22、R-25、R-26、R-27、R-28、R-32、R-36、R-37、R-41、R-42、F-14、F-15~~ ✅ → archive §5.4–§5.13 |
| **第 3 批剩余（离线）** | ~~**R-29**~~ ✅ #50 ⇒ ~~**R-33**~~ ✅ #51（上游已实现，判据已失效）⇒ ~~**R-43**~~ ✅ #53（补 38 项测试）⇒ ~~**R-39**~~ ✅ 实测判定无需改代码（见 §2.2；#52 仍开放待硬件） |
| **第 3.5 批（函数归属，2026-10-06 新增，见 §7）** | ~~**R-45**（#61）⇒ 误报，已结案~~ ⇒ **R-58**（#77，叶子层唯一的真违规）⇒ ~~**R-46**（#62 optimizer→tools，`gaussian_grid` 下沉 `utils/wavefront/beam_amplitude`）~~ ✅（`57ba939c`）⇒ ~~**R-47**（#63 面板几何下沉）~~ ✅（`78a9d3d7`）⇒ ~~**R-48**（#64 帧分析下沉）~~ ✅（`4d068c8f`）⇒ ~~**R-49**（#65 新建 `utils/math`）~~ ✅（`252d81e1`）⇒ **R-50**（#66 Zernike 第三套 API 层）🔴待决策 ⇒ ~~**R-51**（#67 重复收口）~~ ✅（`290ee51`，第三项 `power_bucket` 拆 R-59/#78）⇒ ~~**R-52**（#68 报告脚本参数化）~~ ✅（`3a5b23f`）⇒ ~~**R-53**（#69 `compute_metrics` 改名）~~ ✅（`3aece31`）⇒ ~~**R-54**（#70 runner 重复）~~ ✅（误报 not-planned）⇒ ~~**R-55**（#71 叶子层 matplotlib）~~ ✅ ⇒ **R-56**（#72 optimizer 内渲染面）🟡部分（item 42-45 pygame 状态机 + 47-48 开放；~~无 `__main__` 守卫的 3 文件拆 R-61~~ ⇒ ~~**R-61**（#81）~~ ✅（误报：AST 复核证伪，3 文件均有 `__main__` 守卫 + 渲染 import 已函数局部，`test_rl_no_module_rendering_import.py` 钉死））。**X-4**（#74 死分叉单向移植 ~80 LOC）待做；**F-13**（#73）已结案 |
| **第 4 批（内部重构，需先补特征测试）** | **R-19**（#49，父项）⇒ **R-5 / R-6 / R-7 / R-8**（#42–#45）；**R-14**（#47，需先做产品决策）；**R-13**（#46）；~~**R-44**（#54）~~ ✅（`7284dc1f` 已修，见 §2.2）；**R-18**（#48，接入已有实现或删注释） |
| **纯离线但要台架数据** | ~~**F-4**~~ ✅ 已决策（#55，2026-10-14：保留 `src/ml/` 不迁移）；**F-5**（#56，先切片）；~~**H-23**~~ ✅（`8a756060` 已实施，`Closes` #40）；~~**X-2**（#60，加误报标记）~~ ✅（`f4b9afee` 已加代码标记） |
| **硬件轨道（并行，需设备在线）** | **H-7~H-13**（复测前必修）⇒ **H-19 + H-9 合并做**（ABBA 对消，`dec>0.`55` 且 `late_gain≥10%`）⇒ **H-14 复扫 δ** ⇒ **H-15 / H-16 / F-12**（曝光回读 + 标定常数）⇒ **H-1 / H-2** ⇒ **H-3~H-6**（FourierGSNet）⇒ **H-17 / H-18**。**H-20 / H-21 / H-22** 独立于上述轨道（合并回归 + `--delta` A/B），**H-24** 需先有真机 `--debug` 产物 |

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
| R-50 | [#66](https://github.com/ywzhang909/AO-shaping/issues/66) | ⚠️ **2026-10-06 部分修复，保持开放**。①「第三套 API 层」前提**不成立**：`PatternHelper.generate_zernike_polynomial:550` **已委托** canonical，是薄适配器（孔径外置0 + `nan_to_num` + raw 弧度），被~15 文件依赖 ⇒ 不可删；已订正其 docstring 的「唯一入口」不实声称（`82998b9`）。② 🔴 **两份实现不可互换**：`beam_simulation.py:74` 手写多项式 vs canonical `zernike_calc.py:197 zernike_radial` —— 25 个模式中 18 个仅差**逐模式尺度因子**（归一化不同），**7 个（n≥3 奇数阶）是真正不同的多项式**，最大绝对差 3.243 ⇒ 直接换用会静默改变 `optimizer/rl/envs.py:840` 的 RL 相位。③ `zernike_radial` **零生产调用方**，仅 3 个测试且只覆盖一致的 `(0,0)/(1,1)/(2,0)` ⇒ 它才是可疑的一方 | D 🔴 待决策 |
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
| R-59 | [#78](https://github.com/ywzhang909/AO-shaping/issues/78) | `power_bucket` 两份**归一化语义不同**（`wave.py` 裸桶内和 vs `spots_calc` 的 `weighted`/`use_dpix_scaling`），**委托会改数值、删除会打断活测试** → 需决策，非机械收口 | D 🔴 || ~~R-57~~ | — （已落地，无需 issue） | 峰峰值安全已由软惩罚改为**可行性门**：CLI `--max-peak`（相机计数，默认 `0`=关闭），超限则**不提交该轮更新**，镜像 fold 门（**完全不调** `optimizer.update`，以免用无据的编辑污染 Adam 动量；不追加 `_diff_history`、不刷新基线、不 best-track），违规轮诚实记录 `gate="peak"` 并从 "updates applied" 扣除。**⚠️ 它不是投影也不是安全联锁**：±d 帧在被检查前已出光、提交点要到下一轮才显示（滞后一轮）、只约束相机窗内计数、且 `exposure_time_ms=0` 的自动曝光会掩盖它（该组合下运行器显式告警）。真激光安全仍须带外硬件联锁（IEC 60825-1）。依据见 `report/pib_loss_terms/README.md` §4.1 | E ✅ |

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
| ~~`gui/slm/pyarrow_probe.py` vs `scripts/pyarrow_probe.py`~~ | **曾是误判**：原文写「已漂移（各 106 行但 SHA256 不同）」，实测**两份各 130 行、归一化行尾后 SHA256 完全相同**（`9d4a448b78d7028f`），差异**仅 CRLF vs LF**。副本已删（F-13） |

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
| R-63 | ruff 抑制层清理：73 条 noqa 分类 + `ruff.toml` 分层重构（见 §8.3 / §8.6） | — 待开 issue | A 🟢 |
| R-64 | `src/` 还有 **312 处无标注 `except Exception`**（带 noqa 的只占 9%）+ **111 处 `assert` 当运行期护栏**（`python -O` 下消失，含硬件安全检查） | — 待开 issue | A 🟢 |

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
| **真缺陷** | 1 | `tools/slm/slm_zernike_sweep_probe.py:44` —— re-export 块里 `core_fraction` / `display_and_average` 确实未被使用 ⇒ **删这两个名字** |
| **noqa 已多余**（别名其实被用了） | 3 | `utils/wavefront/matrix_utils.py:51`（用在 `:58`）、`drivers/sim/beam_backend.py:209`（用在 `:212`）、`optimizer/wfless/pib.py:95`（用在 L525/L545）⇒ 直接删 noqa |
| **应改为正式再导出** | 1 | `display/image_display.py:15` ⇒ 补 `__all__`，让 F401 不再是问题。⚠️ 不能改成 `import as` 冗余别名 —— `tests/ao_shaping/display/test_image_voltages_display.py` 有身份断言（`a is b`） |
| **合法副作用/能力探测，保留 noqa** | 8 | `drivers/dm/__init__.py:26,27`（驱动注册；`tests/ao_shaping/runners/test_cli_contract_freeze.py` 依赖）、`drivers/dm/_registry.py:77`（惰性 sim 导入以避环）、`optimizer/wfless/slm_gs_refine.py:703`（注册 sim 相机；`test_sim_camera_registration.py` 依赖）、`gui/slm/pyarrow_probe.py:39,60`（能力探测）、`tools/slm/slm_phase_response.py:47`（Agg 先于 pyplot） |
| **全域 `E402` ignore 是错的工具** | — | `scripts/` 的 **435 条** E402 是仓库**明文约定**的 `sys.path` bootstrap ⇒ 正确做法是 `lint.per-file-ignores` 给 `scripts/*` 加 E402，而不是全局 ignore，这样 `src/` 与 `tests/` 才能被检查 |
| **matplotlib Agg 那批多半不需要** | ~18 | ruff 的 E402 **已豁免** `sys.path` / `os.environ` 修改与 `matplotlib.use()` 激活（`is_matplotlib_activation`，PR #9094 修 #9091）⇒ `src/ml/shaping/viz.py:50-55`、`src/ml/gsnet_debug/train.py:43-66` 的 noqa 即使开 E402 也很可能仍不需要，**先验证再删** |

### 8.4 `BLE001` 31 条：遵守规范 = **收窄类型**，而非删 noqa

⚠️ 先纠正两条常见误解：

* **`except Exception` 不捕获 `KeyboardInterrupt` / `SystemExit`**（二者继承 `BaseException`）
  ⇒ 遥测类 `except Exception` 对 Ctrl-C 是**安全**的，不必收窄。
* 真风险只有 **2 处 `except BaseException`**，而其中
  `utils/image/hardware_utils.py:628` **必须保持 `BaseException`**：它跑在 daemon
  工作线程里，目的就是把工作线程的任何异常搬到调用线程重抛。收窄成 `Exception`
  会让工作线程的 `KeyboardInterrupt`/`SystemExit` 静默消失，`call_with_timeout`
  改为返回 `None`（静默行为变更）。它**不影响**主线程 Ctrl-C（`join` 可中断）。

| 处置 | 站点 | 收窄到 |
|---|---|---|
| **保持 `BaseException` + 补注释** | `utils/image/hardware_utils.py:628` | 不改，仅把线程边界语义写进注释 |
| **最高价值：收窄** | `drivers/slm/santec/driver.py:1184` | `SantecError` —— 房内既有写法：`slm_gs_refine.py:720` `except SantecError as exc: if exc.code != 1: raise` |
| 收窄 | `gui/r50/r50_channel_select.py:360` | `(ValueError, KeyError, TypeError)`（pandas `int()` / 列访问） |
| 收窄 | `tools/slm/slm_zernike_common.py:114,129,155,162,166` | `(SantecError, WfsError, AttributeError)` |
| 收窄 | `runners/slm/zernike_matrix_runner.py:733,774,840` | `(WfsError, RuntimeError, ValueError, AssertionError)` |
| 收窄（teardown / close） | `zernike_matrix_runner.py:1458`、`slm_gs_refine.py:745`、`slm_model_in_loop.py:880` | `(DeviceError, OSError, RuntimeError)` |
| **必须保持宽** | 遥测 12 条（`ml/zernike/train_coeff.py:512,524,547,581,591`、`ml/zernike/train_amp.py:1301,1329`、`ml/shaping/train.py:680,741,767`、`ml/gsnet_debug/train.py:737,855`） | wandb 从网络/文件系统/序列化多路抛错；且 `except Exception` 不吞 Ctrl-C ⇒ 保留 noqa + 理由 |
| **必须保持宽** | `zernike_matrix_runner.py:846` | 包住**调用方传入的任意回调**，无法枚举 |
| 边缘（建议保留） | `zernike_matrix_runner.py:1448`（整条多设备流水线）、`:1323`（显示回调） | 收窄会变脆 |
| **改动前须单独复核** | `optimizer/wfless/slm_zernike_pib.py:1182`（`BaseException`，在 finally 清理里）、`:1194` | 唯一可能真吞 `KeyboardInterrupt` 的站点；须先确认清理后是否重抛 |

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
| 其它 lint 配置 | — | `pyrightconfig.json` 存在但 `typeCheckingMode: "basic"`（宽松）；**无** mypy / flake8 / pylint / bandit；`.github/workflows` 缺失；`.pre-commit-config.yaml` 只有 ruff + ruff-format |

### 8.6 `ruff.toml` 重构顺序（**必须按序**）

> ⚠️ **顺序错了会误删**：`RUF100` 对「规则未启用」的 noqa 同样报 `non-enabled`
> **且可自动修复**。若先跑 RUF100 再开规则，`BLE001`/`SLF001`/`S301` 那 40 条
> 会被删掉，等规则真正启用时就变成裸奔。故顺序为：

1. **先改代码** —— §8.3 的 1 条真缺陷 + 3 条多余 noqa；§8.4 的收窄项
2. **再改配置** —— 从全局 `ignore` 移除 `F401`、`E402`；改用 `lint.per-file-ignores`
   给 `scripts/*` 加 `E402`（保住 435 条 bootstrap 约定）、给 `__init__.py` 加 `F401`
   （AGENTS.md 的再导出 shim 约定）
3. **然后**清理剩余死 noqa：`ruff check src/ --extend-select RUF100 --fix`
   —— **必须 `--extend-select`**；用 `--select RUF100` 会替换整条选择链导致全量误报
4. **最后**处理 `exclude = ["src/ml/**"]`：先清 F-14，再逐族放开（建议先只放开 `F` 族）
5. `BLE001` / `SLF001` / `S301` **暂不整族启用** —— 等 §8.4 收窄完成，否则一次性 299+129+6 条

### 8.7 护栏：以下**不是**待办，**不得**当作「顺手清理」去做

| 项 | 为什么不能 |
|---|---|
| 不要为「让 lint 变绿」给 `src/ml/` 加 noqa | 那 30 个 F821 是真缺陷，加 noqa 等于把 `NameError` 藏得更深 |
| 不要把 `E402` 从 `ignore` 直接挪到 `select` 而不给 `scripts/*` 开 `per-file-ignores` | 会让 435 条 bootstrap 约定瞬间爆红 |
| 不要先跑 RUF100 再开规则 | 见 §8.6 第 3 步，会删掉将来必需的 noqa |
| 不要用 `# ruff: enable` 复活被 ignore 的规则 | 官方明确：range suppression **不能**启用"未被 select 的规则" |
| 不要把 `except BaseException`（`hardware_utils.py:628`）改成 `Exception` | 会造成静默行为变更，已推理见 §8.4 |

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
| **CCD（反范式）** | `assert self.cam, "camera not initialized"` 在 `miicam/driver.py` 重复 **18 处**、`daheng/driver.py` **13 处**、`ffmpeg/driver.py` **6 处**（两种写法）| ❌ 收口到 `BaseCamera._ensure_open()`，抛 `CameraError` |

| 簇 | 条数 | 现状 | 应收口到 |
|---|---|---|---|
| CCD 设备已打开 | **37** | 见上 | `BaseCamera._ensure_open()`，照抄 SLM 范式 |
| WFS 曝光**安全上限** | 5 | `MAX_EXPOSURE_MS = 7.0` 在 `tools/slm/` **声明 3 处**（`calibration.py:127`、`slm_wfs_probe.py:44`、`slm_zernike_common.py:38`）、5 个脚本各查**两遍**（CLI 层 `if…raise` + 内层 `assert`，后者 `-O` 下失效）| 移进 `ThorlabWFS.exposure_time` setter（`thorlab_wfs.py:1801-1818` **已在此校验硬件边界** `EXP_TIME_LOW/HIGH`），安全上限与硬件边界并列 |
| 曝光/范围校验（**层已对，机制错**）| 9 | `drivers/ccd/common.py:66,81,86,95`、`thorlab_wfs.py:1811`、`nlight/driver.py:147,148` 用 `assert` | 就地改 `raise`（`ExposureTime` 越界 → `CameraError`；`EXP_TIME_*` → `WfsError`；DM 梯度 → `DeviceError`）。**位置不用动，只换机制** |
| NLight 原生连接 | 1 | `dm/nlight/sdk.py:62` `assert self._dll.GetConnection2()` | `raise DeviceNotFoundError`（`device_base.py:558`）—— 否则 `-O` 下对空句柄调用原生函数 |
| 参数校验 | 10 | `n_sample > 0` / `tolerance > 0` / `width and height` / offset ≥0，散在 3 个相机驱动 | 相机侧入口统一 `raise ValueError`；调用侧的重复项删 |
| cartographer 设备前置 | 13 | `assert self.slm.is_open` / `assert self.wfs.is_connected()` | **删** —— 驱动的 `_ensure_open` 已覆盖；`is_connected()` 是**询问**，不是断言 |
| optimizer 设备非空 | 7 | `assert cam is not None and slm is not None`（`slm_zernike_pib.py:721` 等）| **删** —— 已由 `create_camera` / `Santec.from_params` 的返回值契约保证 |
| 编程不变式 | ~40 | `assert img.ndim == 2`、`assert best_q is not None`、`assert G.ndim >= 2` 等 | **保留 `assert`**（内部不变量，外部输入到不了）|

> 🔴 **`--max_peak` 不要动。** 它在 **optimizer**（`slm_zernike_pib.py:1456-1474`），不在驱动；
> 且其 CLI 帮助与 archive R-57 **明文声明它不是安全联锁**（"NOT a safety interlock"，
> 只拒绝提交下一轮更新，光已出）。把它搬进驱动等于把**策略**伪装成**机制** ——
> 与 TODO 里 R-57 的结论直接冲突。

**三个必须先解决的前置**

1. ⚠️ **`spots_calc.py:77,119,133` 不许批量转换** ——
   `tests/ao_shaping/utils/test_spots_calc.py:104`（`test_crop_3d_image_raises_assertion`）
   用 `pytest.raises(AssertionError)` 把 `assert` **锁成了公开契约**。要改必须先改测试契约。
2. 🔴 **`MAX_EXPOSURE_MS` 命名冲突，同名差 1400 倍**：工具层 `7.0`（WFS 激光安全上限）
   vs `drivers/ccd/miicam/driver.py:132` `MAX_EXPOSURE_MS = 10000.0`（相机自身硬件上限）。
   合并前必须先改名（如 `WFS_EXPOSURE_SAFETY_MAX_MS`），否则移动过程中必然误用。
3. 💡 **`DeviceParameter` 的边界是"已声明但未接线"**：WFS 已用
   `register_parameter` 登记了 `exposure_time_ms` 的 `EXP_TIME_LOW/HIGH`，
   但 `set_parameter_value`（`device_base.py:354`，内部 `:375` 调 `validate()`）
   **只被** `sync_from_twin`（`:519`）、`mock_devices.py`、`simulated_laser.py` 调用
   ⇒ **硬件写路径上从不生效**。所以不是"要新建校验层"，而是**把已有的声明接到写路径上**。

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
| `optimizer/wfless/slm_zernike_shaping.py` | 119 | 1 | **0** |
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


## 来源清单

| 来源文档 | 提出日期 | 主题 | 现落点 |
|---|---|---|---|
| `docs/TODO.md` | 2026-09-25 | `slm_zernike_pib.py` P0–P3 | 文件已删，内容 → §1 / §2 + archive §5.2/§5.3 |
| `docs/refactor/TODO.md` | 2026-10-01 | `utils/`+`scripts/`+`tools/` 架构重构 | 文件已删，内容 → §2 + archive §5.4–§5.13 |
| `src/ao_shaping/tools/slm/TODO.md` | 2026-10-01 | `tools/slm/` 22 个探针 | **文件仍在** ⇒ §2.3 + X-2 |
| `report/fouriergsnet_pipeline/TODO.md` | 2026-10-01 | FourierGSNet 真机验证 | 文件仍在 ⇒ §1.2 |
| `report/slm/report2.md` §4 | 2026-09-16 | Zernike 矫正后续动作 | → §1.4 |
| `report/slm_pib_bench/report.md` | 2026-09-30 | slm-pib 台架下一步 | → §1.3 |
| `report/slm/bench_calibration_20261001.md` | 2026-10-01 | 稳定化判据 | → H-18 |
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
