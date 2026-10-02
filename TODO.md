# TODO — AO-Shaping 待办总账

> 生成时间：**2026-10-01**（扫描 + 逐项验证）
> 扫描范围：`docs/**/*.md`、`src/**`、`scripts/**`、`tests/**` 中的 `TODO.md`、
> `TODO/FIXME/XXX/HACK/待办/未实现/待确认` 代码注释、以及各报告文档末尾的"下一步/建议"。
> **每项都已对照当前代码核实**，确认仍存在才收录；已修复项移到 §5。
> 排除：`.kilo/worktrees/`、`.venv/`、`libs/OOPAO`、`drivers/ccd/_miicam_sdk`（厂商 SDK）。

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
| H-9 | 单帧采样（`CAM_SAMPLE_ITER = 1`），无 ABBA / 漂移对消。台架已备 `slm_snr_probe.abba_signal` 但**未接** → 单帧差分 SNR 撑不住 576 DOF | `slm_square_shaping.py:136` | 2026-10-01 |
| H-10 | `--exposure_time_ms` 默认 **80.0 ms**，本台架近饱和基线 ~0.02 ms → 默认值必然饱和。且传该参数会**关闭自动曝光**，使饱和保护成死代码（仅 `exposure_time_ms == 0` 触发） | `runner_common.py:1154` | 2026-10-01 |
| H-11 | 焦距标定常数自相矛盾：文档同写 `132940/P` 与 `7600/P`（差 17.5×）；实测 7400 与一阶 `7557/P` 仅差 1–2% | `slm_bench_probe.py:78` / 文档 | 2026-10-01 |
| H-12 | `lr=0` 时 `learning_schedule()` **静默覆盖 `--delta`**；`_param_scale` freeform=1.0 而 Zernike=0.1，故 Zernike 调好的 δ 不通用；"freeform per-pixel" 名不副实（`np.kron` 分块，24×24 → 80×50 px/block） | `slm_square_shaping.py:1596-1605,1163/1170`、`_freeform_phase_radians` | 2026-10-01 |
| H-13 | 可达目标需重新定义：本台架 80 px staircase 只能影响**大尺度**结构（≥100 px 大 ROI、压低斑径、提 Strehl），刻 50 px 平顶方块不可达 | `hardware_run_20261001.md` §6.5 | 2026-10-01 |
| H-14 | `--delta` 未被 `learning_schedule` 覆盖时可用区间实测为 `≈0.1`（`n_max=9`）；`0.2` 触发亮度折叠门，`<0.001` 95% 迭代被门控。噪声地板单日波动 21× → 每次须重跑 SNR 扫描选 δ | `docs/slm_pib_bench/report.md` §6 | 2026-09-30 |
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
> `ShapingObjective.__call__`、`ideal_pib_ratio` 已改名 `tracking_value`。
> **原文档行号全部失效**，但 bug 本身逐条存活。

| # | 项 | 当前证据 | 提出 |
|---|---|---|---|
| R-1 **P0** | **能量守卫惩罚被绕过**（真 bug）。`ShapingObjective.__call__` 返回含 `±1e3` 惩罚的 `res.j`，但 `tracking_value()` 在 `objective == "pib"` 时**从 `img` 重算桶比、完全丢弃惩罚**。三处 `best_*` 都走它（`slm_zernike_pib.py:992/1212/1470`），退出时 `:1097` 把该未惩罚值写回 SLM → 硬件上 SLM 会留在守卫不允许的状态。修法：`__call__` 返回三元组 `(J_eff, ratio, headline)`，守卫触发时一并惩罚 headline | `utils/image/target/objective.py:717-755` | 2026-09-25 |
| R-2 **P0** | `learning_schedule(radius(init_img, center=center, energy=0.8))` 传的 `center` 是 `reset_window` 返回的**全帧坐标**，而 `init_img` 已是窗口局部图 —— 与已修的 `r_bucket` 同类坐标系 bug。**SPGD 分支已修**（`:1497` 用 `pos_center`），heuristic/init 这处仍错 | `slm_zernike_pib.py:980-985` | 2026-09-25 |
| R-3 **P0** | 饱和判定硬编码 `255`，且两分支**互不一致**（heuristic `>=255`、SPGD `==255`），都假设 uint8 → 统一 `np.iinfo(img.dtype).max` 并抽 `is_saturated(img)` | `slm_zernike_pib.py:1190,1419` | 2026-09-25 |
| R-4 **P0** | `_log_row` 漏记 `w_ee`/`ee_term`（`_row0` 记 6 项，`_log_row` 只记 4 项，`_w_ee` 已解包但没写）→ DataFrame 首行有值、其余全 NaN。**部分已修**（4/6） | `slm_zernike_pib.py:1017-1025,1076-1082` | 2026-09-25 |
| R-5 P1 | 目标函数 if/elif 链 → Objective 类族 + `GuardedObjective`（闭包已提取成函数名，仍是 if 分发 + `objective_mode` 符号 + `last_terms` nonlocal） | `slm_zernike_pib.py:1300-1368` | 2026-09-25 |
| R-6 P1 | 两个搜索分支共享 `BenchSession`（"clip→相位→display→sleep→采图→饱和→算目标" 序列重复，饱和策略还不一致） | SPGD ~`:1419` / heuristic `:1190` | 2026-09-25 |
| R-7 P1 | 巨型函数拆编排器：~900 行 → `prepare_geometry` 返回**不可变 dataclass**（字段区分 `window_center_full_frame` / `reference_center_window_local`）+ `_run_spgd`/`_run_heuristic` | `slm_zernike_pib.py:760-1700` | 2026-09-25 |
| R-8 P1 | 硬件安全 try/finally：任何异常（相机掉线、越界、KeyboardInterrupt）都让 SLM 停在随机相位 | `slm_zernike_pib.py:1060` | 2026-09-25 |
| R-9 P2 | 300px 光阑踩坑文档**挂错常量**：文字在 `TARGET_BOX_WAIST_FACTOR`（`:252`）后且是字符串字面量（不是 docstring、不可达），真正该注释的 `ZERNIKE_APERTURE_RADIUS = 300.0`（`:239`）无任何说明 | `slm_zernike_pib.py:239,252` | 2026-09-25 |
| R-10 P2 | 死代码 `gauss_center`（零生产调用，可删）。**新增发现**：`slm_zernike_shaping.py:154` 还有第二份副本，而那份是生产代码 | `slm_zernike_pib.py:167-217` | 2026-09-25 |
| R-11 P2 | 常量替换字面量（`-5.0/5.0` clip ×6、守卫惩罚 `1e3`、`1e-4`）→ `GUARD_PENALTY` / `IMPROVE_EPS` / 复用 `ZERNIKE_CLIP` | 多处 | 2026-09-25 |
| R-12 P2 | `_update_dynamic_weights` → `AdaptiveWeights` dataclass（现为裸 dict setdefault + 2/3-tuple 联合返回），顺带收口 R-4 | `slm_zernike_pib.py:251-375` | 2026-09-25 |
| R-13 P2 | `_create_optimizer` 的 `inspect.signature` 创可贴 → 显式 `OptimizerConfig` | `slm_zernike_pib.py:422-430` | 2026-09-25 |
| R-14 P2 | `_metric_panel` 每 epoch 全量六套指标 → 加 `panel_every_n: int = 1` 开关 | `slm_zernike_pib.py:1544` | 2026-09-25 |
| R-15 P2 | `_apply_best_on_exit` 往 Recorder 挂属性 → 显式 `RawReport` 字段 | `slm_zernike_pib.py:~1584` | 2026-09-25 |
| R-16 P2 | `SLM_WIDTH/HEIGHT` 与驱动 `Panel_Res` 重复 → 读驱动常量 | `slm_zernike_pib.py:129-131` | 2026-09-25 |
| R-17 P2 | 日志 f-string/`{}` 占位符混用 → 统一 | `slm_zernike_pib.py` 多处 | 2026-09-25 |
| R-18 P3 | 离线 GS 作闭环初值 `--init-gs`。⚠️ **`gs_warm_start` 已在别处落地**（`slm_gs_refine.py`、`iterative_zernike_shaping.py` + `slm_gs_refine_runner`）→ 本项改为"接入已有实现"或删掉 `:739` 的陈旧注释 | `slm_zernike_pib.py:739` | 2026-09-25 |
| R-19 P3 | 补 sim 台架对称 BenchSession（`sim` 相机后端已有 2f-Fourier），使 R-1~R-17 可无硬件回归 | — | 2026-09-25 |

### 2.2 `utils/` + `scripts/` + `tools/` 架构重构（源自 `docs/refactor/TODO.md`，2026-10-01）

**前置阻断项（🔴 必须最先做）**：`utils/` **不是**安全的叶子。
`utils/wavefront/pattern_helper.py:20` 裸 `import aotools`（无 try/except），
`utils/__init__.py:17-27` eager 拉起 4 个子包 + 98 个 eager 名字（无 PEP 562 `__getattr__`）。
⇒ 未装 `aotools` 的机器上 `import ao_shaping.utils` 直接失败。

| # | 项 | 类型 | 提出 |
|---|---|---|---|
| R-20 **前置** | `pattern_helper.py` 的 `aotools` 改惰性/受保护导入；`utils/__init__.py` 98 个 eager 名字 → PEP 562 `__getattr__`（**只延迟，绝不删名字**）。验证：失败模式从 import 期移到调用期，需专门测试 | 重构 | 2026-10-01 |
| R-21 | `utils/image/gs_visualization.py`（390 行）→ `display/`。同时是 test-only orphan，且**违反仓库自己的反模式** | 重构 | 2026-10-01 |
| R-22 | `utils/hardware_utils.py` → `utils/image/hardware_utils.py` 迁移收尾。**旧路径仍有 6 src + 2 test + 2 根目录脚本 = 10 处**，新路径 5 src + 2 test = 7 处。⚠️ 迁移前必须 grep `flat_gray` / `um_to_waves`（各有两个家，移动会静默撕裂） | 重构 | 2026-10-01 |
| R-23 | 删 `utils/slm_utils.py`（**0 导入**，docstring 理由"测试重置 `slm_utils._last_slm_slot`"已过期） | 清理 | 2026-10-01 |
| R-24 | 补 `tools/micro_dm/__init__.py`（**不存在**，`find_packages()` 式打包会静默丢弃）；修 **12 处**已失效的 `python -m ao_shaping.tools.micro_dm_image_collect` docstring（真实路径带 `.micro_dm.`） | 修复 | 2026-10-01 |
| R-25 | 建 `scripts/_common/` 抽出报告/绘图助手：`iters_to_threshold`/`format_iters`、`_savefig`、`_markdown_table`、OOPAO 5 助手。⚠️ **`_fmt` 已漂移**（一份用 `math.isnan` 会把 1e-7 渲染成 `0.0000`，另一份用 `np.isnan`）→ 统一取 gsnet 行为视为**修复** | 重构 | 2026-10-01 |
| R-26 | `generate_strehl_benchmark_report.py:442` 硬编码兄弟产物 `docs/heuristic_pib/summary.csv` → 改 CLI 参数。**当前就在静默降级**（缺文件只打一条提示），磁盘上已有的报告可能就是错的 | 修复 | 2026-10-01 |
| R-27 | `runner_common.py` 的 CLI 机制 → `utils/io/cli_params.py`（零 `ao_shaping` 导入，结构上不可能成环）；`runners/__init__.py` 改真 lazy | 重构 | 2026-10-01 |
| R-28 | utils 内部去重：日期格式化 D1（2+2 份逐字节相同）、日期目录 D2、max 归一化 D4（6+ 处）、argmax→(x,y) 光斑定位 D3（11 处 / 9 份内联）。⚠️ D5（scipy.zoom vs 手写双线性）是**唯一会改变数值结果**的去重，必须先钉数值特征测试 | 重构 | 2026-10-01 |
| R-29 | `utils/image/display.py:19` 从 `io/handler.py` 导入 `Register`，AGENTS.md 把方向说反了 → **改文档，不迁移 `Register`** | 文档 | 2026-10-01 |
| R-30 | 修 `pyproject.toml` 加 `pythonpath = ["src", "scripts"]`（一行修好 11 个脆弱脚本在 pytest/IDE 下的导入）；清空 `tools/slm/__init__.py` eager 再导出（保留 docstring，**只能清空不能删文件**） | 修复 | 2026-10-01 |
| R-31 | `cartographer/test_smoke.py` 从 `src/` 迁到 `tests/`（现永不被收集）；修 2 处输出路径违规（`generate_cython_optimizer_report.py` 写 `docs/` 根、`generate_centroid_test_visualization.py` 写进 `scripts/reports/`） | 清理 | 2026-10-01 |
| R-32 | 加约定测试（孤儿检测 + `python -m` 一致性 + utils 分层守卫），**warn-only + baseline 起步**。⚠️ 必须在 R-24 之后做，否则 `python -m` 测试会红 | 重构 | 2026-10-01 |
| R-33 | `micro_dm_image_collect.py` → `with_params(MicroDMParams)`；删 7 个驱动内部符号导入与手写 `_resolve_ips`；同 PR 内启用已有的 `R50Controller.__enter__/__exit__`（全仓库零使用）。⚠️ 必须先钉死 Micro-DM 磁盘布局（承重：`find_cell_image` + 4 个 `md_img_*` 脚本依赖） | 重构 | 2026-10-01 |
| R-34 | 清理被 git 跟踪的 `scripts/tuning_devices/stdWavefront/` **66 个 .txt（~57 MB）**；`train_data_collect.py` 与 `micro_dm_image_collect.py` 均 **0 测试** | 清理 | 2026-10-01 |

### 2.3 `tools/slm/` CLI 层重构（源自 `src/ao_shaping/tools/slm/TODO.md`，2026-10-01）

**只动 CLI 层，不合并命令体**（settle/slot/dark-frame 是位置敏感步骤，挪动会产生看似可信的错数据）。

| # | 项 | 当前实测 | 提出 |
|---|---|---|---|
| R-35 | **Step 0（前置，不可跳过）**：冻结每个命令的 `--help` 全文、固定非默认 argv 下的 dataclass repr、8 个命令的 `--dm_type` 选项列表；断言"固定 argv 下无字段等于默认值"（防跨 `with_params` 字段名静默冲突）；钉住 `pupil_center` 的「二元联合 + callback」耦合（改注解为裸 `str` 且删 callback → **导入即炸**） | — | 2026-10-01 |
| R-36 | Step 1：`runner_common.py:118-323` 的 click 机制 → 新叶子模块（**禁止任何 `ao_shaping.*` 导入**，写进 docstring 不变式） | 机制段零 `ao_shaping` 导入，是干净叶子 | 2026-10-01 |
| R-37 | Step 2/3：逐族迁移 22 个探针到 `tools/slm/params.py`。实测 **15 个手写 `@click.option`**（~200 个 flag）、1 个 argparse、**0 个用 `with_params`**；**17 个文件直接构造 `Santec(...)`**（18 处） | 同左 | 2026-10-01 |
| R-38 | canonical 采用率过低：19 个构造 SLM 的文件里 **只有 1 个**用 `zero_order_center`（`slm_snr_probe.py`），其余裸 `np.argmax`；`phase_to_slm_grayscale` 也**只有 1 个**文件用，另有 **7 处**直调 `create_phase_from_array` | 同左 | 2026-10-01 |
| R-39 | 曝光默认值 7 种并存（0.02/0.03/1.1/1.2/2.0/3.0/4.0 ms）；内存槽轮换 3 种写法（驱动自动 / 自建 `SlotRotator` / 手工 `current_slot`） | 同左 | 2026-10-01 |
| R-40 | `tools/slm/__init__.py` docstring 漏 6 个模块：`slm_zernike_response` / `slm_zernike_correction` / `slm_zernike_common`（fan-in 10，被 `zernike_matrix_runner.py:95,111` 与 `gui/slm/slm_calibration_ui.py:36` 依赖）/ `slm_wfs_probe` / `slm_wfs_reference` / `delta_explorer`；且仍声称 LUT canonical 在 `utils/slm_lut.py`（实际 `utils/slm/slm_lut.py`） | 同左 | 2026-10-01 |
| R-41 | flag 拼写分裂：`--cam-type`（6 个探针）vs `--camera-type`（`slm_diagnose` / `slm_lut_runner`）。建议保留现有拼写不破坏习惯用法 | 同左 | 2026-10-01 |

---

## 3. 离线缺陷修复 / 文档（无需硬件）

| # | 项 | 位置 | 提出 |
|---|---|---|---|
| F-1 | **内存槽固件 no-op 违规**：`:383` 在 `for i in range(max_iter)` 里反复 `apply_compensation(comp_gs, memory_slot=2)` → 固件把已显示槽当 no-op，**第 2..N 次迭代全是空操作，LCOS 不刷新**。且 `:266` 默认值写死 2、`:272` docstring 写 "1-128"、`:276` 漏 `memory_mode=MEMORY_MODE_INTERNAL`（不同于 canonical `_display()`） | `tools/slm/cartographer/dynamic_compensation.py:266,272,276,383` | 2026-10-01 |
| F-2 | **README 选项表自相矛盾**：`slm-diagnose` 选项表写 `--cam-type`（`README.md:439`），而同页警告块（`:449`）与示例（`:455`）写 `--camera-type`。**代码实际是 `--camera-type`**（`slm_diagnose.py:256`）→ 只有表格是错的，照抄表格直接报未知选项 | `README.md:439,449,455` | 2026-10-01 |
| F-3 | `--display/--no-display` 选项的 help 写"暂未实现"——需确认是补实现还是删选项 | `runner_common.py:1795` | 2026-09-25 |
| F-4 | `src/ml/` 移入 `src/ao_shaping/ml/` 并更新所有引用（`docs/issues_report.md` §10.3，待评估至今） | `src/ml/` | 2026-05-26 |
| F-5 | `docs/issues_report.md` §11 的代码规范整改：`print()` 替代 loguru（原文 82 处）、宽泛 `except`、配置项分散、大文件拆分（~22 个）、冗余 `__main__` 入口（32 处）、`__future__` 覆盖率（29 个文件）。⚠️ **原文数字已过期，实施前需重新扫描** | 全仓 | 2026-05-26 |
| F-6 | 清理 `__pycache__` 里 5 个已删模块的陈旧 `.pyc`（`_slm_fix_wavelength` / `_slm_health_check` / `_slm_reboot_wavelength` / `slm_shift_calib` / `slm_zernike_report`） | `tools/slm/__pycache__/` | 2026-10-01 |
| F-7 | `scripts/README.md` 未收录 8 个脚本：`generate_slm_pib_online_report.py` / `generate_slm_pib_rms_pib_report.py` / `generate_beam_shaping_papers_report.py` / `generate_cython_optimizer_report.py` / `repeat_shape_objectives.py` / `explore_delta.py` / `objective_rep_logging.py` / `pyarrow_probe.py` | `scripts/` | 2026-10-01 |
| F-8 | 17 个脚本输出中文但**缺 CJK 字体 rcParams**，其中 ~5 个把中文写进图/markdown → 豆腐块。6 个手写 `cli(...)` 无 `--help` | `scripts/` | 2026-10-01 |
| F-9 | `repeat_shape_objectives.py` 加进度显示（用户要求） | `scripts/` | 2026-09-30 |
| **F-10** | 🔴 **`sim/AGENTS.md`「已知约束」第 4 条描述的代码改写从未落地**：该条声称 `_rescale_for` 已改为**只** `(_R0_REF_500/r0_slab)**(5/6)`，并称已移除 `lam/_LAM_REF_500`、`/_CAL_REF`、`*sqrt(1.03)`。**三者至今仍在** `oopao_backend.py:96,101-103`（`_CAL_REF = 0.6191` 在 `:71`）。连带第 3 条的实测常数 1.068/2.628 **不可复现** —— 真实值是 **5.428 / 13.354**（与 `docs/oopao_impact/report.md:62-63` 一致）。**先落地改写并重跑 `generate_oopao_impact_report.py`，或回退那两条。** | `oopao_backend.py:88-103` + `sim/AGENTS.md` 第 3/4 条 | 2026-10-01 |
| **F-11** | 🔴 **SLM 序列号三路冲突**：`drivers/AGENTS.md:158` 与 `docs/slm/bench_calibration_20261001.md` 记 SLM#1 = **22030108**（@1064nm，2π=993）；`drivers/slm/AGENTS.md:114,139` 记 **22030102**（@532nm，2π=998）；`docs/slm/report2.md` / `report3.md` / `zernike_linearity/linearity.md` 记 **23020026**（@532nm）。三者或为两台设备。**引用前必须确认，并回写 `drivers/AGENTS.md` 硬件表**（Daheng CCD `FJB24112232` 已于 2026-10-01 补录进该表） | `drivers/AGENTS.md` 硬件事实表 | 2026-10-01 |
| **F-12** | **焦面标定常数三方不一致**：`AGENTS.md:697` 写 `5021/Λ`（对应 3.31 µm 像元）；`docs/slm/model_in_loop_bench_calibration.md` / `README.md:517` 写 7400–7600（对应 2.2 µm 像元）；`docs/slm_pib_heuristic_hw/report.md:159` 主张改 **10954**。⚠️ **2.2 µm 像元推得 ~7557 而非 10954，故该主张本身也待复核**。H-11 只覆盖了 132940 vs 7600，**未覆盖此三方冲突** | `slm_diagnose.py:54`、`slm_lut_runner.py:38`、`slm_bench_probe.py:78`、两处测试 | 2026-10-01 |
| **F-13** | `strehl()` 是**去均值余弦相似度**，不是物理 Strehl 比（`slm_shaping_bench.py:309-323`）。但 `beam_shaping_benchmark.py` 与 `docs/beam_shaping/papers/beam_shaping_papers.md:30` 消费它的输出并称 "Strehl" ⇒ **docs 树里所有 "Strehl" 数字实为归一化重叠**。`docs/zotero_objectives/README.md:526-528` 已标记「需决策」但未修 | `slm_shaping_bench.py:309` | 2026-10-01 |
| **F-14** | 报告生成写在 `algorithm/` 层（`signal_processing/beam_shaping_benchmark.py:543,562` 写 CSV/MD）违反 `AGENTS.md` 反模式红线「report generation MUST live in `scripts/`」 | `algorithm/signal_processing/beam_shaping_benchmark.py` | 2026-10-01 |
| **F-15** | `docs/beam_shaping_benchmark_metrics.md` 被 `README.md:1813` 当权威链接，实际是 **1 行 smoke 残留**；权威 9 行网格在 `docs/benchmarks/device_less_full/`（被 gitignore，本 checkout 无）。两边都不可用 | `README.md:1813` | 2026-10-01 |

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

---

## 6. 建议执行顺序

| 批次 | 内容 | 前置 |
|---|---|---|
| **第 1 批（纯收益，零行为风险）** | R-23、R-24、R-30、R-31、F-2、F-6、F-7、F-8、R-40 | 无 |
| **第 1.5 批（文档/常量收口，先定事实再改代码）** | F-10（OOPAO 改写 + 重跑报告）、F-11（SLM 序列号）、F-12（标定常数三方）、F-13（`strehl()` 命名） | 需设备/一次扫描 |
| **第 2 批（止真 bug，需先补特征测试）** | R-1、R-2、R-3、R-4、R-9、R-10、R-11、R-35 | R-35 先行 |
| **第 3 批（架构重构）** | R-20（前置阻断）→ R-21、R-22、R-25、R-26、R-27、R-28、R-32、R-36→R-37→R-38→R-39→R-41、F-14、F-15 | R-20 先行 |
| **第 4 批（内部重构）** | R-5~R-8、R-12~R-17、R-19 | R-1~R-4 完成 |
| **硬件轨道（并行）** | F-1 → H-7~H-13 → H-14/H-15/H-16 → H-1/H-2 → H-3~H-6 → H-17/H-18 | 设备在线 |