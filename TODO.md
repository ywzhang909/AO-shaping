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
| H-23 | 🟡 **纯离线可修**：方形家族 `--debug` 产物只有 freeform 基能进 ML 语料，zernike 基 0% | [#40](https://github.com/ywzhang909/AO-shaping/issues/40) |
| H-24 | 方形家族进 `hwdataset` 会换 `fov_px` ⇒ `far_field_padding` 标定必须重扫 | [#41](https://github.com/ywzhang909/AO-shaping/issues/41) |
| **H-25** | **`--w_loggrad` 结构项已落地但未实机验收**。离线全绿（`utils/` 992、`wfless/` 626、sim 端到端 `J≠shape`）；6 类候选里只落地这 1 类，2 类早已存在、3 类否决（附实测数字）⇒ [`report/pib_loss_terms/README.md`](report/pib_loss_terms/README.md)。**⚠️ 它不能修复不收敛**（H-14 已证瓶颈是慢漂移），故默认 `0.0`。验收请在 **δ=0.05** 这类「有信号但被漂移污染」的档位做，**勿用 0.0005**（梯度信号本身在噪声下）。**待开 issue** | — |

### 1.2 FourierGSNet 真机验证（`report/fouriergsnet_pipeline/`）

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
| H-10 | `--exposure_time_ms` 默认 **80.0 ms**（本台架近饱和基线 ~0.02 ms），且传它会**关掉饱和保护**使其成死代码 | [#27](https://github.com/ywzhang909/AO-shaping/issues/27) |
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
> 🔴 **`slm_zernike_shaping.py` 是本文件的同源副本**（同样 1600 行量级、同样
> `_row0`/`_log_row`/`learning_schedule`/饱和分支结构）⇒ **改下面任何一条都必须同时改
> 副本**，否则测试可能只覆盖一边（历史上 R-1~R-4 四条 P0 实测**两边都有**）。
> 副本盘点见 [archive §5.17](docs/dev/todo_archive.md)。

| ID | 摘要 | issue |
|---|---|---|
| R-5 P1 | 目标函数 if/elif 链 → `Objective` 类族 + `GuardedObjective` | [#42](https://github.com/ywzhang909/AO-shaping/issues/42) |
| R-6 P1 | 两个搜索分支共享 `BenchSession`（序列重复，**饱和策略还不一致**） | [#43](https://github.com/ywzhang909/AO-shaping/issues/43) |
| R-7 P1 | 巨型函数拆编排器：~900 行 → `prepare_geometry`（**不可变 dataclass**，字段必须区分 `window_center_full_frame` / `reference_center_window_local`）+ `_run_spgd` / `_run_heuristic` | [#44](https://github.com/ywzhang909/AO-shaping/issues/44) |
| R-8 P1 | 硬件安全 try/finally：任何异常（相机掉线、越界、`KeyboardInterrupt`）都让 SLM 停在随机相位 | [#45](https://github.com/ywzhang909/AO-shaping/issues/45) |
| R-13 P2 | `_create_optimizer` 的 `inspect.signature` 创可贴**已按实测修**，但真问题是**静默吞参数**：`SGD` 只收 `(dim, lr)` ⇒ `**config.kwargs` 逃生口是**死的** | [#46](https://github.com/ywzhang909/AO-shaping/issues/46) |
| R-14 P2 | `metric_panel` 每 epoch 全量指标 → 加 `panel_every_n`。⚠️ **前置是产品决策**（跳过轮次留空 vs 沿用上一轮） | [#47](https://github.com/ywzhang909/AO-shaping/issues/47) |
| R-18 P3 | 离线 GS 作闭环初值 `--init-gs`：**`gs_warm_start` 已在别处落地** ⇒ 接入已有实现，或删掉陈旧注释 | [#48](https://github.com/ywzhang909/AO-shaping/issues/48) |
| R-19 P3 | 🟡 注入机制已统一（archive §5.34），**`BenchSession` 本体未做** ⇒ R-5~R-8 仍待办。本条是那四项的跟踪父项 | [#49](https://github.com/ywzhang909/AO-shaping/issues/49) |

### 2.2 `utils/` + `scripts/` + `tools/` 架构重构

**原前置阻断项（🔴 `utils/` 不是安全的叶子）已于 2026-10-03 完成 → archive §5.4。**

| ID | 摘要 | issue |
|---|---|---|
| R-29 | 作废条目的遗留：把 `utils/image/display.py` 搬到 `display/`（`utils/image/` 最后一块渲染代码，含 `utils/__init__.py:78/80/81` 三个 re-export） | [#50](https://github.com/ywzhang909/AO-shaping/issues/50) |
| R-33 | `micro_dm_image_collect.py` → `with_params(MicroDMParams)`；删 7 个驱动内部符号导入与手写 `_resolve_ips`；启用 `R50Controller.__enter__/__exit__`。⚠️ 必须先钉死 Micro-DM 磁盘布局 | [#51](https://github.com/ywzhang909/AO-shaping/issues/51) |
| R-39 | 曝光默认值 7 种并存（0.02/0.03/1.1/1.2/2.0/3.0/4.0 ms）；内存槽轮换 3 种写法。⚠️ 只动 CLI 层，**不合并命令体** | [#52](https://github.com/ywzhang909/AO-shaping/issues/52) |
| R-43 | 补 `micro_dm_image_collect.py` 测试（🟡 `train_data_collect.py` 一半已完成 → archive §5.32） | [#53](https://github.com/ywzhang909/AO-shaping/issues/53) |
| R-44 | 🔴 `algorithm.learning_schedule` 的 `cosin` 分支在 `epoch > epochs/2` 后返回**负学习率**（实测 `-3.09e-4`）。已用测试钉住现状，改它会动到所有调用者轨迹 | [#54](https://github.com/ywzhang909/AO-shaping/issues/54) |

> **旧账本 §2.2 的 R-30 / R-31 两行是陈旧的** —— 它们实际已于 2026-10-03 完成
> （见 [archive §5.1](docs/dev/todo_archive.md)），但表格未同步更新。**不要**为它们开 issue。

### 2.3 `tools/slm/` CLI 层重构

**只动 CLI 层，不合并命令体** —— settle/slot/dark-frame 是位置敏感步骤，
挪动会产生**看似可信的错数据**。

| ID | 摘要 | issue |
|---|---|---|
| — | ~~R-37 迁 19 个探针 / R-42 迁 4 个 argparse 探针 / R-38 拒做 / R-41 保留两种 flag 拼写~~ | ✅ 全部完成 → archive §5.15 / §5.31 / §5.19 / §5.18 |

---

## 3. 离线缺陷修复 / 文档（无需硬件）

| ID | 摘要 | issue |
|---|---|---|
| F-4 | `src/ml/` 移入 `src/ao_shaping/ml/` 并更新所有引用（2026-05-26 提出，**决策本身还没做**） | [#55](https://github.com/ywzhang909/AO-shaping/issues/55) |
| F-5 | `docs/issues_report.md` §11 代码规范整改。**已重新扫描（2026-10-04），原文数字低估 5～14 倍** ⇒ 建议按切片做 | [#56](https://github.com/ywzhang909/AO-shaping/issues/56) |
| F-10 | 🔴 `sim/AGENTS.md`「已知约束」第 4 条声称的代码改写**从未落地** ⇒ 连带第 3 条的实测常数 1.068/2.628 **不可复现**（真实值 5.428/13.354） | [#57](https://github.com/ywzhang909/AO-shaping/issues/57) |
| F-11 | 🔴 **SLM 序列号三路冲突**：22030108 / 22030102 / 23020026 ⇒ 引用前必须确认并回写 `drivers/AGENTS.md` 硬件表 | [#58](https://github.com/ywzhang909/AO-shaping/issues/58) |
| F-12 | 🔴 **焦面标定常数三方不一致**：`5021/Λ` / 7400–7600 / 主张 10954。⚠️ 2.2 µm 像元推得 ~7557 而非 10954 ⇒ **三方无一是已验证的** | [#59](https://github.com/ywzhang909/AO-shaping/issues/59) |

> **F-12 未定案前不要单独改 H-16 的 `K`** —— 否则只是把一个未验证的数从一处搬到另外两处。

---

## 4. 原文档结论需修正（照做会出事）→ 已结案

| ID | 原结论 | 实际情况 |
|---|---|---|
| X-1 | `docs/refactor/TODO.md` #16：「`utils/image/resample.py` **0 导入** → 删模块」 | ✅ **前提已消失** —— 现在有 1 个生产导入方（`FourierGSNet.py:42`），照原文删会打断 FourierGSNet。**且 `docs/refactor/TODO.md` 本身已不存在**（已并入本账本） |
| X-2 | `tools/slm/TODO.md` D2：`calibration.py:2182` 的 min-max 是尺度无关反模式 | ❌ **误报** —— 那处 `P` 是实测功率-扫描位置曲线，归一化后 `np.interp` 找交点是标准做法。**但该源文件仍存在，陷阱仍可踩到** ⇒ 需加显式标记 → [#60](https://github.com/ywzhang909/AO-shaping/issues/60) |
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
| ~~第 2 批（止真 bug）~~ | ~~R-1~R-4、R-9~R-11、R-35~~ ✅ → archive §5.2 / §5.3 |
| ~~第 3 批（架构重构）~~ | ~~R-20、R-21、R-22、R-25、R-26、R-27、R-28、R-32、R-36、R-37、R-41、R-42、F-14、F-15~~ ✅ → archive §5.4–§5.13 |
| **第 3 批剩余（离线）** | **R-29**（#50，`display.py` 搬 `display/`）⇒ **R-33 / R-43**（同一文件，#51 / #53）⇒ **R-39**（#52） |
| **第 3.5 批（函数归属，2026-10-06 新增，见 §7）** | **R-45**（#61 叶子层 14 处）⇒ **R-46**（#62 optimizer→tools）⇒ **R-47**（#63 面板几何下沉）⇒ **R-48**（#64 帧分析下沉）⇒ **R-49**（#65 新建 `utils/math`）⇒ **R-50**（#66 Zernike 第三套 API 层）⇒ **R-51**（#67 重复收口）⇒ **R-52**（#68 报告脚本参数化）⇒ **R-53**（#69 `compute_metrics` 改名）⇒ **R-54**（#70 runner 重复）⇒ **R-55**（#71 叶子层 matplotlib）⇒ **R-56**（#72 optimizer 内渲染面）。**X-4**（#74 死分叉单向移植）与 **F-13**（#73 先 diff）是两个前置决策项，建议先做 |
| **第 4 批（内部重构，需先补特征测试）** | **R-19**（#49，父项）⇒ **R-5 / R-6 / R-7 / R-8**（#42–#45）；**R-14**（#47，需先做产品决策）；**R-13**（#46）；**R-44**（#54，需先决定是否动所有调用者轨迹）；**R-18**（#48，接入已有实现或删注释） |
| **纯离线但要台架数据** | **F-4**（#55，先做「`ml` 算不算主包」的决策）；**F-5**（#56，先切片）；**H-23**（#40，修 `hwdataset` 索引）；**X-2**（#60，加误报标记） |
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
| R-45 | [#61](https://github.com/ywzhang909/AO-shaping/issues/61) | `utils/` 叶子层 14 处反向依赖（`hardware_utils` / `pattern_helper` / `wfs_utils`） | A 🟢 |
| R-46 | [#62](https://github.com/ywzhang909/AO-shaping/issues/62) | `optimizer` 反向 import `tools/` 探针模块（`slm_model_in_loop` 两处） | B 🟢 |
| R-47 | [#63](https://github.com/ywzhang909/AO-shaping/issues/63) | 面板几何下沉 `utils/slm/`（**`phase_to_panel` 归属纠正 + 命名陷阱**） | C 🟡 |
| R-48 | [#64](https://github.com/ywzhang909/AO-shaping/issues/64) | 帧分析下沉 `utils/image/`（13 个纯数组符号） | C 🟡 |
| R-49 | [#65](https://github.com/ywzhang909/AO-shaping/issues/65) | 新建 `utils/math/`（拟合 + 稳定性判据） | C 🟡 |
| R-50 | [#66](https://github.com/ywzhang909/AO-shaping/issues/66) | Zernike **第三套 API 层**收敛（`beam_simulation` / `PatternHelper`） | D 🟡 |
| R-51 | [#67](https://github.com/ywzhang909/AO-shaping/issues/67) | 重复收口：`flat_gray` / `power_bucket` / `create_target_mask` | D 🟡 |
| R-52 | [#68](https://github.com/ywzhang909/AO-shaping/issues/68) | 报告脚本 `plot_summary_bars` 三份相同实现参数化 | D 🟡 |
| R-53 | [#69](https://github.com/ywzhang909/AO-shaping/issues/69) | `compute_metrics` 三份同名不同契约 → **改名不合并** | D 🟡 |
| R-54 | [#70](https://github.com/ywzhang909/AO-shaping/issues/70) | 两个 SLM runner 大块重复（只抽共用段，不动 CLI flag） | D 🟡 |
| R-55 | [#71](https://github.com/ywzhang909/AO-shaping/issues/71) | `utils/io/file.py` 等叶子层模块级 import matplotlib | E 🟡 |
| R-56 | [#72](https://github.com/ywzhang909/AO-shaping/issues/72) | `optimizer/` 内 9 处 pygame / matplotlib 渲染面 | E 🔴→🟡 |
| F-13 | [#73](https://github.com/ywzhang909/AO-shaping/issues/73) | `pyarrow_probe` 两份**已漂移**（哈希不同），先 diff 再定去留 | 护栏 |
| X-4 | [#74](https://github.com/ywzhang909/AO-shaping/issues/74) | 死分叉结论修正：**单向移植 ~80 LOC，不是删除** | F 🟡 |
| R-29 | [#50](https://github.com/ywzhang909/AO-shaping/issues/50) | `utils/image/display.py` → `display/`（既有 issue，本次只补了 `Register` 循环导入裁决） | E 🟡 |
| ~~R-57~~ | — （已落地，无需 issue） | 峰峰值安全已由软惩罚改为**可行性门**：CLI `--max_peak`（相机计数，默认 `0`=关闭），超限则**不提交该轮更新**，镜像 fold 门（**完全不调** `optimizer.update`，以免用无据的编辑污染 Adam 动量；不追加 `_diff_history`、不刷新基线、不 best-track），违规轮诚实记录 `gate="peak"` 并从 "updates applied" 扣除。**⚠️ 它不是投影也不是安全联锁**：±d 帧在被检查前已出光、提交点要到下一轮才显示（滞后一轮）、只约束相机窗内计数、且 `exposure_time_ms=0` 的自动曝光会掩盖它（该组合下运行器显式告警）。真激光安全仍须带外硬件联锁（IEC 60825-1）。依据见 `report/pib_loss_terms/README.md` §4.1 | E ✅ |

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
| `gui/slm/pyarrow_probe.py` vs `scripts/pyarrow_probe.py` | **已漂移**（各 106 行但 SHA256 不同）⇒ 见 F-13，不可按「相同副本」直接删 |

> 上述行号是 2026-10-06 审计时的快照。**动手前必须重新核对** —— 用户工作区当时有
> 大量未提交改动，行号大概率已漂移。

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
| 代码结构审计（4 路并行 + 1 次分叉评审） | 2026-10-06 | 函数归属 / 重复实现 / 叶子层违规 | → **§7**（issue #50、#61–#74） |

**扫描范围**：`docs/**/*.md`、`src/**`、`scripts/**`、`tests/**` 中的 `TODO.md`、
`TODO/FIXME/XXX/HACK/待办/未实现/待确认` 代码注释、以及各报告文档末尾的
「下一步/建议」。**每项都已对照当前代码核实**，确认仍存在才收录。
排除：`.kilo/worktrees/`、`.venv/`、`libs/OOPAO`、`drivers/ccd/_miicam_sdk`（厂商 SDK）。

**路径订正 2026-10-05**：实验报告已从 `docs/` 迁到仓库根的 `report/`
（见 [`report/README.md`](report/README.md)）。本账本正文中所有指向报告的
`docs/<topic>/...` 引用已随之改写为 `report/<topic>/...`；
archive §4 中复述 2026-10-01 那次误覆盖事故的段落**保持原路径不变** ——
那里记录的是当时实际被写入的位置。
