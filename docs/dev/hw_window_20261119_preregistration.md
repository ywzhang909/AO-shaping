# 全包硬件窗口：验收判据预注册（2026-11-19）

<!-- 本文件在任何本窗口新采集数据产生之前写成，用于冻结评分口径，防止事后选阈值。 -->

## 0. 冻结元信息（写本文件时）

| 条目 | 值 |
|---|---|
| 预注册写入时刻（UTC） | 2026-11-19T11:04:55Z |
| 依据用户授权 | 本窗口采集范围「全包一次窗口」；阈值口径「配对改善 + 报绝对值」 |
| git HEAD | `adcdb50bc5652477551e748297c68f73c284f77a` |
| `validate_transfer_shapes_hardware.py` SHA256(前16) | `19947B714C8A3986` |
| `replay_transfer_shapes_hardware.py` SHA256(前16) | `25DB72439891A98E` |
| `run_transfer_window_20261119.py` SHA256(前16) | `A406D6220E1A3FDC` |
| `run_issue238_period_sweep_20261119.py` SHA256(前16) | `4CA2762FB25ABAB2`（原 `25508186A97DD926`，见重签记录） |
| `scripts/experiments/_window.py` SHA256(前16) | `808BFAAB0976526D`（初签未列，见重签记录） |
| `protocol.py` SHA256(前16) | `49CBA3A62DADC8ED`（初签与重签一致，评分口径未变） |
| 写文件时设备状态 | `device_lock.json` 不存在；未占用 SLM / CCD / 激光 |

> 上表哈希只标识评分所用的冻结脚本版本；若本窗口使用不同版本，须在此登记新哈希并重签，否则评分作废。

### 0.1 重签记录（2026-11-19T11:19:41Z，早于本窗口任何新数据）

- 复核时刻 `2026-11-19T11:19:41Z`，`git rev-parse HEAD` 仍为 `adcdb50bc5652477551e748297c68f73c284f77a`；`device_lock.json` 仍不存在，未采任何本窗口数据。
- `run_issue238_period_sweep_20261119.py` 由 `25508186A97DD926` 变为 `4CA2762FB25ABAB2`。经核读源码（第 12–24、264 行），改动**仅为锁协议修复**：改用 `lock_mode="segment"`（窗口不持锁、每个周期子段自己 acquire/release bench 锁），修复 11-19 `run3_period_sweep/period32` 的「窗口持锁 + 子段 `--wait-until-idle 0`」父子互锁失败。**不含任何评分/指标逻辑**。
- `scripts/experiments/_window.py`（`808BFAAB0976526D`）为并行会话改动；它是采集信封，只负责等锁 / 段间冷却 / 建独立 `--out` / 释放与证据落盘，**不参与任何指标计算**，初签时未列入，此处补登记。**不含任何评分/指标逻辑**。
- 评分口径的唯一实现 `protocol.py::target_intensity_rmse` 哈希仍为 `49CBA3A62DADC8ED`，与初签逐位一致；§2 全部判据（2.3 配对主门、2.4 绝对值仅报告、2.5 归因边界）**不变**。
- 结论：两处变更均属设备占用/锁协议层，不改变评分口径。按 §0 登记新哈希并重签后，本窗口可用上表当前版本评分；若执行前哈希再次变化，须再次重签。

## 1. 本窗口范围（用户授权）

- **执行**：S0 六步光路探测 + 曝光标定；Transfer K=4（及 K=1 同窗对照）；`ccd_mode light` / `ccd_mode none`；#238 二值光栅 32/48 px 周期扫描（run3）；目标形状 measured-vs-target 配对采集（带锁 `validate --ccd-check` + 带锁 `replay`）。
- **不执行**：S7（#247 零级闭环，D2 未放行）；S5/S6（P3/P4 探针，D1 未放行）。
- **风险接受声明**：项目记录 `transfer_mil_hardware_plan_20261118.md` §10–§11 与 `agent_notes_20261119.md` 结论 7 明确「目标形状 CCD 验收暂不放行」（静态观测模型基选错、仅在已删除 `mil` 分支实现、未入 master）。本窗口对目标形状段按**用户明确风险接受**取数；不据此升默认、不合并、不宣称验收通过。

## 2. 评分判据（冻结，看到本窗口数据后不得修改）

### 2.1 数据来源与排除项

- 只用**本窗口新采**的、带唯一 run 目录与完整 provenance 的 measured-vs-target 配对帧。
- **明确排除、不参与评分**：所有历史数据（`20261024` / `1027` / `1028` / `1107` / `1111` / `1116` / `1118` 及 11-19 早先的 `issue238_binary_phase_20261119`、`transfer_window_20261119` 失败段）。历史重算只作背景，不作验收。
- 饱和判据：`mean_frame >= full_scale * 250/255`（`full_scale` 按生效位深取值）。任一帧饱和的配对**剔除并计数**，不参与通过率分母。

### 2.2 指标与实现（口径冻结）

- 用生产函数 `ao_shaping.tools.slm.transfer.protocol.target_intensity_rmse` 在冻结 grid/window 上算 `rel_l2 = ||m-t||/||t||` 与 `energy_fraction`（整窗各自归一化后相减）；辅助报 `cv` / `ee_roi` / `pearson`（`image_metrics`）。
- **背景 floor**：当次**同曝光暗场**实测得到的冻结值；历史 replay 的 `floor=0` 口径不合规，禁止沿用。floor 的**来源方法在此冻结**，具体数值现场读回后填入 §3 并留证。
- 逐帧重复离散度作为**噪声地板**：同一相位、同一曝光的重复帧逐帧 `rel_l2` 极差（`rel_l2_frame_spread`）；`energy_fraction` 同理取其逐帧极差。

### 2.3 配对改善通过判据（唯一主门）

对每个有效（非饱和）seed→final 配对，同时满足才算该对通过：

1. `Δrel_l2 = rel_l2(seed) − rel_l2(final) > max(noise_floor_rel_l2(seed), noise_floor_rel_l2(final))`；
2. `Δefrac = energy_fraction(final) − energy_fraction(seed) > max(noise_floor_efrac(seed), noise_floor_efrac(final))`。

**窗口级通过**：有效配对数 ≥ 5，且通过配对占比 > 50%（严格多数）。

### 2.4 绝对值：仅报告，不设通过线

- `rel_l2` / `energy_fraction` / `cv` / `ee_roi` / `pearson` 的绝对值与分布**如实报告**，用于判断绝对保真度距离目标还有多远。
- **本窗口不设任何绝对通过阈值**（不采用 core CV≤0.10、rel_l2≤0.6、efrac≥0.5 等线）；绝对门须由后续独立窗口预注册，不得由本窗口数据反推。

### 2.5 归因边界（写死，防止过度宣称）

- 上面 2.3 若通过，只说明**端到端整形相位优化相对 seed 有实测改善**，不指向任何单一 issue 子项。
- 五个 OPEN issue（#234/#235/#236/#238/#247）在本窗口后**仍保持 OPEN**，除非存在「同一相位、仅切换单一子项开关」的 A/B 配对并单独满足 2.3。
- #238 二值光栅周期扫描是本窗口唯一可能隔离单一 issue 的段；其判据独立按 `issue238_binary_gray_capture_plan_20261119.md` 执行，且若给出 g₂π 工作值，须与设备回读 993 对照而非取代。
- 任何结果都**不构成**升默认、合并或「验收通过」的依据。

## 3. 现场填写（方法已在 §2 冻结，值须现场读回后填入并留时间戳）

| 条目 | 当次值 | 来源 / 时间戳 |
|---|---|---|
| 本窗口唯一 run 根目录 | 待填 | 执行时新建、确认不存在 |
| 同曝光暗场 floor（每档曝光） | 待填 | 当次暗场采集 |
| zero-order center (CCD px) / exclusion radius | 待填 | 当次标定 |
| 目标形状 / 边长（须 ≤ window_span 可容纳的 424 px；11-19 失败段用了 1200 px） | 待填 | 当次几何配对 |
| target center / window span / step / core mask | 待填 | 当次冻结 |
| CCD 型号 / ID / 生效位深 / full scale / 原生帧尺寸 | 待填 | 当次读回 |
| 曝光固定值 / 硬件范围 / 操作上限 / 读回值 | 待填 | 当次读回 |
| frames / settle / ABBA 是否启用及理由 | 待填 | 当次决定 |
| dataset / checkpoint 路径与 SHA256 | 待填 | 当次冻结 |
| 设备授权 / run 时间窗 / 操作者 | 待填 | 用户/操作者 |

## 4. 执行与设备纪律

- 接设备前后按 `docs/dev/hardware_discipline.md` 与 `libs/skills/ao-optical-path-calibration/SKILL.md` 在回复中逐条报告：占用设备、做了什么、何时释放、是否正常关闭。
- 六步光路探测（设备在线 / 光路有效 / 曝光不超量程 / SLM 相位真在调制 / A=0 焦点 / 位深）任一项失败即止步。
- 段间冷却 ≥200 s（背对背 SLM open/close 触发驱动板假死 `ReadSU=-1000`）；单进程占用，不并行第二设备进程。
- 11-19 早先 `transfer_window_20261119/segments/k4.log` 的 `--side 1200` 超窗错误须在 S1 前修正为 ≤424 px 几何，不得沿用。
- 评分在设备下线后离线进行（§2 口径）；占用期不做分析。

## 5. 复算可追溯

判据 2.3/2.4 用 `C:/Users/zhangh/AppData/Local/Temp/opencode/hist_traj.py` 的同源生产函数逐帧重算，本窗口另写 `hw_window_20261119_score.json` 落每次配对的 `rel_l2/efrac/noise_floor/sat/pass` 明细。历史脚本本轮已逐位复现（JSON 哈希 `FEB310A351E79AAA6AEF608DB21AAAEC909A0767C0646A8C4C1B09EFF847EDB1`）。
