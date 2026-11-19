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
| `run_transfer_window_20261119.py` SHA256(前16) | `79E76137B6DB4065`（原 `A406D6220E1A3FDC`，见 §0.2 重签记录） |
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

### 0.2 重签记录（2026-11-19T12:58:52Z，早于本窗口任何新数据）

- 复核时刻 `2026-11-19T12:58:52Z`，`device_lock.json` 不存在，未采任何本窗口数据（transfer 唯一尝试 `transfer_window_20261119` 在 `validate()` 即失败，未生成任何 k4/light/none 数据）。
- `run_transfer_window_20261119.py` 由 `A406D6220E1A3FDC` 变为 `79E76137B6DB4065`。改动**仅为参数/几何入口**，不触碰评分逻辑：
  1. `--side` 默认由 `1200` 改为 `None`（不给即沿用 transfer 默认 `80`）；`1200` 在默认采样窗半跨 `220 px` 下必然超窗（`2*(220-8)=424`），是 11-19 k4 失败的唯一原因。
  2. 新增 `--window-span` / `--capture-grid` / `--control-grid` 透传，以及 dry-run 阶段按 `2*(window-span-8)` 提前拒绝超窗 `--side` 的守卫；只影响 argv 构造与预检，不改变任何指标计算。
- 评分口径实现 `protocol.py::target_intensity_rmse` 哈希仍为 `49CBA3A62DADC8ED`，§2 判据（2.3 配对主门 / 2.4 绝对值仅报告 / 2.5 归因边界）**不变**；`validate_transfer_shapes_hardware.py`、`replay_transfer_shapes_hardware.py` 哈希未变。
- 结论：该变更属采集入口层，不改评分口径。按 §0 登记新哈希并重签后可用其评分；执行前请再次复核此哈希。


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
| 本窗口唯一 run 根目录 | S0：`data/experiments/optical_probe_20261119/s0_window_2pi_fix_20261119/20261119_123642Z`；S1 光路复核：`data/experiments/optical_probe_20261119/s1_laser_on_recheck_20261119/20261119_132412Z`；目标形状配对：`data/experiments/hw_window_20261119/validate_pairs`（replay 待新建 `.../replay_pairs`） | 2026-11-19 20:36 / 21:24 / 21:36 CST 新建，执行前均确认不存在 |
| 同曝光暗场 floor（每档曝光） | **已采并冻结**：0.08 ms 档 `floor=14.9375 counts`（12-bit 暗场均值图空间中位数，协议 `_robust_background` 口径）；附 0.02/0.04/0.06/0.1/0.12/1/10 ms 全档，均 12-bit/full_scale=4095、16 帧；详见 `report/shape_trials/dark_frame_record_20261119.md` | 本次 laser-off 采集 2026-11-19 20:58:37→20:59:49 CST；`data/ccd_configs/ccd/FJB24112232/darks/dark-exp-0.08ms.npz` SHA256(前16)=`608825B8A5AE957D`；manifest 快照 SHA256=`A0F63BF9FC0D092128CA2FCF16B8C7D220DD0FC84BF8DF03F3BAAF1ADF1A3109` |
| zero-order center (CCD px) / exclusion radius | **已冻结**：`zero_xy=(1226, 1105)`（S1 当次 0.08 ms 双帧 argmax (1226,1105)/(1227,1105)，与 S0 (1227,1106) 一致）；`zero_radius=0`（目标由 ramp predicted 落在 y≈988，距零级约 117 px，不靠零级盘排除；如 seed 帧显示零级串扰则中止并记录，不靠事后调参补救） | S1 `data/experiments/optical_probe_20261119/s1_laser_on_recheck_20261119/20261119_132412Z/exposure_0.08ms.npz` 逐帧 argmax，2026-11-19 21:26 CST |
| 目标形状 / 边长（须 ≤ window_span 可容纳的 424 px；11-19 失败段用了 1200 px） | **已冻结**：shape=`square`，`side=80` px（保守取值，目标核 x[1188,1268]/y[948,1028] 全在采样窗内，远离窗边；不追大尺寸，先看配对改善是否成立） | 操作者 2026-11-19 21:34 CST 拍板；`side≤2*(220-8)=424` |
| target center / window span / step / core mask | **已冻结**：`target_xy=(1228, 988)`（=zero+ramp[:,0]/period，fit ramp[:,0]=(155.97,-7489.12)，period=64 → shift≈(2.4,-117.0)）；`window_span=220`、`window_step=2`、`core_margin` 不显式给（沿用默认 2 sampling cells）；`period=64`，`zero_radius=0`。target_xy 由 fit checkpoint ramp predicted，**以 validate 的 square_seed 帧现场验证**（判据见下一行 §3.1 修正） | fit checkpoint ramp + S1 zero_xy；操作者 2026-11-19 21:34 CST 拍板 period=64 |
| CCD 型号 / ID / 生效位深 / full scale / 原生帧尺寸 | Daheng MER2-507-23GM（sn `FJB24112232`），cam_id=0，12-bit，`full_scale=4095`，原生 2592×1944 | S0 `optical_path_probe.json` `config` + `device`，2026-11-19T20:38Z |
| 曝光固定值 / 硬件范围 / 操作上限 / 读回值 | 建议 0.08 ms（峰值 2401 counts / 149.5 gray，落在 100–200 目标带）；硬件范围 [0.02, 1000] ms；可用窗 0.02–0.12 ms；峰值线性 r²=0.987，光通量 26298 counts/ms | S0 `optical_path_probe.json` `ladder_summary`，2026-11-19T20:38Z |
| frames / settle / ABBA 是否启用及理由 | S0：每档 frames=2、离焦每态 5、settle=0.5 s；目标形状段：**frames=3**（与 transfer 默认一致），**ABBA 关闭、启用 `--ccd-check --mode anchored`**（anchored sequential CCD 判定；避免 4 次采集带来的漂移与零级串扰，manifest 每 shape 只留 1 个 `square_seed` + 1 个 `square_final`），`--fixed-exposure-ms 0.08`（与暗场 floor 14.9375 counts、S1 未饱和读数同档） | S0/S1 读回 + 操作者 2026-11-19 21:34 CST 确认 |
| dataset / checkpoint 路径与 SHA256 | **已冻结**：checkpoint=`data/debug/slm_transfer/axis_mismatch_validation_20261118/fit/frozen_parameters.json`，SHA256=`44E7EBF1DFC73C6AA3D4FAF40B98573E54CC8F5F92092FDB64F89944223D721F`；dataset=`.../capture/manifest.json`，SHA256=`3D6A734EFD9158211A1004D7D13570E27BAB228FE8775DB63B5C5359C197C14E`。操作者确认：该 fit 是 11-18 失败段 provenance 对应的拟合输出、为当前可定位的最新 full run，且 S0/S1 光路读数与之一致（光路未动）；历史 provenance 的 `checkpoint` 字段指向 `initial/`（拟合输入），本窗口按操作者 21:20 拍板改用 `fit/`（拟合输出，含 gain/defocus/bias）。历史数据仍**不参与评分**，仅提供冻结物理参数 | 操作者 2026-11-19 21:20 CST 拍板（fit）；SHA256 21:34 CST 计算；checklist §6 provenance 核对 |
| 设备授权 / run 时间窗 / 操作者 | 用户授权「全包一次窗口」+「立即开始」；S0 运行窗 20:36–20:38 CST，设备正常释放；操作者=用户 | 用户消息 + 设备占用报告 |
| SLM 物理 2π 灰度（新增，来自本次修正） | **993**（设备回读 `Santec.get_wavelength_info()[1]`，非编码标度 1023） | S0 `optical_path_record.json` `devices.slm.two_pi_grayscale`，2026-11-19T20:38Z |

> §3 回填说明（2026-11-19 20:40 CST）：以上仅回填本次 S0 已客观读回的字段；需现场几何标定/暗场采集/操作者确认的字段明确标「仍缺」，未回填前不得进入对应采集段。回填不改动 §1 范围、§2 冻结判据与 §4 纪律。

### 3.1 几何验证判据统计量的修正（2026-11-19 21:52 CST，**在看到本批数据之后**）

> **这是事后修正，必须按事后修正来读。** §3 原写的现场安全阀是「目标亮斑 argmax 落在 `(1228,988)`±10 px」。该判据在本批 `validate_pairs` 上按字面**不通过**（seed 偏 12.0–15.8 px、final 偏 20.0–39.7 px）。修正后改用**能量质心**，并同时公开原判据的失败读数，不允许只引用修正后的通过结论。

- **为什么原判据不成立**：`argmax` 是单像素极值，在散斑场内逐帧跳动 —— final 三帧的 argmax x 在 1197→1198→1208、y 在 983→1013→989 之间跳，而同一三帧的能量质心稳定在 `(1228.0,986.0)`±0.5 px。它衡量的是「最亮那一个像素在哪」，不是「目标能量中心在哪」。
- **修正后的判据**：以 `floor=14.9375 counts` 扣除后，在**预测目标周围的能量质心**（r≤45 圆盘，等价检验亦用 model `core` mask）上判断，容差 ≤10 px。本批实测：seed `(1229.6,983.5)/(1229.7,983.5)/(1229.5,983.9)` → 偏 4.4–4.8 px；final `(1228.2,985.7)/(1227.7,986.0)/(1228.1,986.4)` → 偏 1.6–2.3 px。**通过**。
- **稳健性证据（随修正一并留档，供事后复核判据是否被放松）**：质心随窗口半径 r=20/30/40/60/80 稳定在预测位置 ±5 px 内；只有 r=120 才被零级（`y≈1105`）拖到 `y≈1018`，说明该半径已跨入零级，而非目标漂移。故 r≤45 的取法不是为凑通过而选的窗口。
- **一并披露的负面读数（不得隐去）**：本批全局最亮点始终是零级（约 `(1225,1105)`，峰值 1053–1319 counts），目标区峰值仅 109–182 counts；零级盘（r≤30）占扣背景后总能量 11.4–13.4%，目标盘（r≤40）占 10.8–13.0%；`core_energy/total` seed 0.403–0.411 → final 0.338–0.350，即 **final 的目标区能量占比低于 seed**。同时 `ee_roi` 0.1391→0.1241、`target_pearson` 0.3894→0.3052 均下降（仅 `cv` 0.3271→0.1728 改善）。
- **效力边界**：本修正只决定「本批帧是否进入评分」，**不改动 §2 的评分主门**（配对改善 `rel_l2` / `energy_fraction` + 绝对值仅报告 + §2.5 归因边界）。通过本安全阀不等于验收通过，也不改变任何 OPEN issue 状态。

## 4. 执行与设备纪律

- 接设备前后按 `docs/dev/hardware_discipline.md` 与 `libs/skills/ao-optical-path-calibration/SKILL.md` 在回复中逐条报告：占用设备、做了什么、何时释放、是否正常关闭。
- 六步光路探测（设备在线 / 光路有效 / 曝光不超量程 / SLM 相位真在调制 / A=0 焦点 / 位深）任一项失败即止步。
- 段间冷却 ≥200 s（背对背 SLM open/close 触发驱动板假死 `ReadSU=-1000`）；单进程占用，不并行第二设备进程。
- 11-19 早先 `transfer_window_20261119/segments/k4.log` 的 `--side 1200` 超窗错误须在 S1 前修正为 ≤424 px 几何，不得沿用。
- 评分在设备下线后离线进行（§2 口径）；占用期不做分析。

## 5. 复算可追溯

判据 2.3/2.4 用 `C:/Users/zhangh/AppData/Local/Temp/opencode/hist_traj.py` 的同源生产函数逐帧重算，本窗口另写 `hw_window_20261119_score.json` 落每次配对的 `rel_l2/efrac/noise_floor/sat/pass` 明细。历史脚本本轮已逐位复现（JSON 哈希 `FEB310A351E79AAA6AEF608DB21AAAEC909A0767C0646A8C4C1B09EFF847EDB1`）。

## 6. §2.3 主门实测结果（离线评分，2026-11-19 追加）

> 本节是**结果追加**，不是判据修改。判据/指标/floor 仍严格按 §2.2–§2.5；离线评分在设备下线后进行（§4）。结论为**主门不通过且方向为负**，不构成验收通过、升默认或合并依据。过程性辅助读数已在 §3.1 披露；本节补的是 §2.3 主门本身（`rel_l2` / `energy_fraction` 相对噪声地板）的计算值。

- **输入**：`data/experiments/hw_window_20261119/validate_pairs/`（6 capture × 3 帧；12-bit；`0.08 ms`；0 饱和；含 `square_seed` 与 `square_final` 各一）。
- **产物**：`data/experiments/hw_window_20261119/hw_window_20261119_score.json`（`data/*` 被 gitignore，不入库）；SHA256=`B92934B649DEA78620F51EB873F7D1092B828B76716B9847B79773E24D34E8C9`。
- **评分脚本**（离线、纯生产函数、只读原始帧）：`C:/Users/zhangh/AppData/Local/Temp/opencode/hw_window_score_20261119.py`，SHA256=`496BDCBC0620172E4A8CC1EB9DFDD1B6A400BE1E5C97D375D3ED2768D16C065B`。
- **复现**：`.venv\Scripts\python.exe C:/Users/zhangh/AppData/Local/Temp/opencode/hw_window_score_20261119.py`。
- **配方自校验**：脚本对每 capture 以 `raw.mean(0)[y, x]`（`(y,x)=camera_xy`，与 `transfer/artifacts.py::audit_run` 一致）算 `cv/ee_roi/target_pearson`，与 `manifest.json` 逐项最大绝对差 = **0.0**，证明评分配方与生产 validate 完全一致。

### 6.1 逐 capture（3 帧均值，`floor=14.9375 counts`）

| label | rel_l2 | energy_fraction | cv | ee_roi | target_pearson | rel_l2 帧极差 | efrac 帧极差 |
|---|---|---|---|---|---|---|---|
| square_seed  | 1.15748 | 0.45557 | 0.32709 | 0.13914 | 0.38938 | 0.01170 | 0.00574 |
| square_0_A1  | 1.18980 | 0.45044 | 0.32949 | 0.13827 | 0.37470 | 0.04296 | 0.01246 |
| square_0_B   | 1.25386 | 0.41963 | 0.24541 | 0.13128 | 0.33488 | 0.03659 | 0.00541 |
| square_1_A1  | 1.26406 | 0.41395 | 0.24879 | 0.13052 | 0.32826 | 0.01119 | 0.00635 |
| square_1_B   | 1.28990 | 0.39248 | 0.17472 | 0.12278 | 0.30722 | 0.04114 | 0.01035 |
| square_final | 1.29579 | 0.39191 | 0.17276 | 0.12410 | 0.30521 | 0.04243 | 0.01456 |

（全部 capture `sat_px=0`。`rel_l2 帧极差`/`efrac 帧极差` 即 §2.2 的逐帧噪声地板 `noise_floor_*`。）

### 6.2 §2.3 配对主门

| 配对 | Δrel_l2 = rel_l2(A)−rel_l2(B) | 噪声地板 | rel_l2 通过 | Δefrac = efrac(B)−efrac(A) | 噪声地板 | efrac 通过 | 结论 |
|---|---|---|---|---|---|---|---|
| seed → final | **−0.1383** | 0.0424 | 否（方向为负，超地板） | **−0.0637** | 0.0146 | 否（方向为负，超地板） | 未通过 |
| 0_A1 → 0_B | **−0.0641** | 0.0430 | 否（方向为负，超地板） | **−0.0308** | 0.0125 | 否（方向为负，超地板） | 未通过 |
| 1_A1 → 1_B | −0.0258 | 0.0411 | 否（在噪声内，方向为负） | **−0.0215** | 0.0103 | 否（方向为负，超地板） | 未通过 |

- **符号约定**：`Δrel_l2>0` 表示 final 更接近目标；`Δefrac>0` 表示 final 把更多能量放入目标核。上表三对**两项均为负**——即 final 相对其配对起点在实测上**更远离目标形状、目标区能量更少**，而非"无改善"。
- **seed→final 这对**的 `rel_l2` 与 `energy_fraction` 退化量均**超过**各自噪声地板的 3 倍以上（0.1383/0.0424≈3.3；0.0637/0.0146≈4.4），是稳健的负向变化，不是帧噪声。

### 6.3 窗口级判定（§2.3）

- 有效 seed→final 配对数 = **1**（< 预注册要求的 5）→ 窗口级状态 `insufficient_pairs`；且该唯一配对两项主指标均**未通过**、方向为负。
- 故本窗口无法给出"配对改善"结论；即使后续补到 ≥5 对，当前这对的负向读数也要求先解释"优化为何把能量推离目标核"，再谈通过。

### 6.4 辅助指标的相反方向（§2.4 仅报告）

- seed→final：`cv` 0.3271→0.1728（目标核更平，**唯一改善**）；但 `ee_roi` 0.1391→0.1241、`target_pearson` 0.3894→0.3052 **均下降**，`core_energy/total` 亦由 0.403–0.411 落到 0.338–0.350（见 §3.1）。
- 机理一致：优化把目标核内亮度压平（CV 降），但把能量更多留在了全域最亮点——约 `(1225,1105)` 的**零级**（峰值 1053–1319 counts），目标区峰值仅 109–182 counts。即"CV 降了但目标没被填对"，与 §2.2 冻结 `rel_l2/energy_fraction` 的负向结论一致，不构成任何算法收益。

### 6.5 replay 状态与边界

- `replay_transfer_shapes_hardware.py` 本身**必开设备**（无 model-only 分支），本轮无硬件授权 → **未执行**，`.../replay_pairs` **未新建**。
- §2.3 主门评分**不需要 replay**：`validate_pairs` 已分别采集 `square_seed` 与 `square_final` 的相位帧，评分直接在已采帧上完成。replay 只作相位序列复现性复核，留待下次获批窗口。
- **归因边界（§2.5 重申）**：上述只说明本窗口 seed→final 端到端在实测上**相对目标变差**，不指向任何单一 issue 子项；#234/#235/#236/#238/#247 仍保持 OPEN，不改 LUT/默认、不升默认、不合并。
