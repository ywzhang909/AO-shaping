# #238 二值光栅破简并采集计划（2026-11-19）

> 状态：**16 px run1 与 run4 的 32/48 px 均已完成离线拟合**；两周期 fold 估计约 947–953，均低于设备回读 993，但整段束功率漂移 13–32%、模型残差与 ±1 极小分裂仍较大，不能据此改写 LUT。具体判读见 §6.5。
> 本文件保留实验设计；任何新一轮设备占用仍须由用户在回复中明确批准。
> 关联：[当日记录](agent_notes_20261119.md) · [slm_lut §10](../slm/slm_lut.md) · GitHub #238。

## 1. 为什么需要这次采集

#238 第一轮灰度探测（`data/debug/issue238_gray_phase_20261119/run1/`）确认：单组闪耀光栅效率中 γ（灰度→相位非线性）、g₂π 与条纹对比度强简并，峰值法结论已撤回；跨灰度幂律只能给 938±11、CI[917,961]，不是工作值。工作值取设备回读 993。串扰判据因 cell 128 px / gap 64 px 使像素核跨不过间隙而不可判定。

破简并办法：改用**对称二值相位光栅**，逐点读出绝对相位 φ(g)，其第二零点直接给出 g₂π 且与 γ 无关。离线验证已通过（e 域直接拟合无偏、SD<1 灰度；993/938 分离 16σ、938/885 分离 11σ），本次采集只为在实验系统上取得对应实测数据。

## 2. 设备与占用纪律

- 设备：Santec SLM #1（1064 nm，memory 模式）+ Daheng CCD #0 + 激光。默认不启 WFS/DM。
- 全程单进程占用，不并行任何设备进程；占用开始与释放在回复中明确报告时间戳。
- 禁止背靠背 SLM open/close；写入前确认驱动板在线（历史 ReadSU=-1000 / Open=-10002 假死需用户手动重启，不反复重试）。
- 实配前先跑短参考探测：`--wait-until-idle`、光路有效、曝光不超量程、SLM 相位真在调制四项通过后再进正式序列；任一项失败即止步。
- 采集结束写相位归零、关 SLM、释放 CCD、释放 device_lock；复核锁状态。

## 3. 采集参数

- 图案：对称二值相位光栅（相位只有 0 与 φ 两级，占空比 1:1），φ 按灰度 g 扫描；与现有 ramp 探针分开，避免模式混淆。
- 灰度点：覆盖 g=0..1023，重点密集采 885 / 938 / 993 / 1023 及其邻域；邻域步长不大于 8 灰度，保证第二零点定位。
- 每点帧数：≥20 帧（离线验证的 20 帧平均已给出 SD<1 灰度）；帧序随机化以抑制漂移。
- 曝光：以 run1 的 0.04 ms 为起点（Mono12 峰值约 2107 < 饱和线 3685），按自动曝光判据确认不饱和、非欠曝；不得沿用 8-bit 量纲。
- 读取：Mono12 全幅原始帧，不裁剪；同时记录 0 级与 ±1 级功率（分离相位深度与可见度衰减）。
- 输出：**新目录** `data/debug/issue238_gray_phase_20261119/run2_binary/`，不得复用 run1；写入前确认目录不存在。
- 记录：同步写 `optical_path_record.json`（保留 run1 既有值，不覆盖）、device_log、plan、summary（n_patterns / 帧数 / error 字段）。

## 4. 分析口径（执行前冻结）

- 主拟合：绝对效率 e 域最小二乘，`η₀ = cos²(φ/2)`、`η₁ = (4/π²) sin²(φ/2)`；**禁止**用 `cos(phi) = (e0n - A)/B` 且 `e0n = η₀/(η₀+2η₁)` 的归一化反演。
- 判据：第二零点给 g₂π；检验 993/938/885 两两分离是否 ≥10σ；报告 φ(g) 与 γ 的估计及置信区间。
- 分支处理：先确认 g > g₂π 后是饱和还是缠绕，据此决定全点拟合或仅首支拟合；不得两者择优取好看的结果。
- 串扰不参与本轮判定；如需重测，另立探针几何（缩小 cell / 增大 gap），不得用现有 128/64 数据下结论。

## 5. 交付与后续

- 采集完成后：写 `summary.json` 与分析结果到 run2_binary；更新当日记录与 #238 issue 评论；根据实测 g₂π 决定是否更新 `docs/slm/slm_lut.md` §10 与 `AGENTS.md` 指针。
- 若驱动板假死或光路/曝光自检失败：只记录失败原因，不反复重试、不改判据；保持 issue OPEN。

## 6. 执行状态与修订（2026-11-19 实测后追加）

- **run1 已执行**：`data/debug/issue238_binary_phase_20261119/run1/`，对称二值光栅 **16 px** 周期、50% 占空、48 灰度点（粗步 32 + 864..1023 步 8）、**16 帧/点**（777 帧，Mono12，峰值 2048 无饱和，漂移 −0.56%，`error=null`，设备已干净释放）。此为设计第 3 节初版，尚未加周期维。
- **§4「主拟合 = 绝对效率 e 域最小二乘」经实测证伪为误设定**：理想 full-curve e 拟合给 g₂π≈898、γ≈1.17、rms 0.11，与设备回读 993 及折叠结构矛盾。正确口径为 **γ-无关的折叠/地标估计**（η₁ 第二零点/极小对称轴），理想 e 拟合**降级为诊断量**，不得作为标定值。§4 主拟合一条以本节为准。
- **run1 结论（合格版）**：折叠 g₂π≈**962.7**（帧 bootstrap CI [959,966]，仅帧噪声）；地标峰（φ=π）≈448；条件 γ≈0.906（系统范围 [0.884,0.920]）；±1 极小分裂 +1≈981 / −1≈952 → 系统包络 **≈[952,981]**，故报告 **g₂π≈964，±13 系统**。简并仅在地标模型内打破，非模型无关绝对标定。885/1023 仍被排除；blaze 幂律 CI[917,961] 与二值区间重叠，**不能宣称决定性排除**。
- **新增瓶颈 = 周期，不是帧数**：16 px 下像素响应模糊使 e₀ 地板偏高、±1 极小分裂，系统 ±13 由此而来。**加大周期（32/48 px）是唯一有新增价值的实验增量**；采集脚本已支持 `--period-px`，无需改代码。

### 6.1 32 px 首次尝试（run2_period32）— 中止；完整复采见 §6.4

- **采集中止**：`data/debug/issue238_binary_phase_20261119/run2_period32/`，对称二值光栅 **32 px** 周期、50% 占空、**20 帧/点**、计划 51 图案 / 969 帧；实际至 **725 帧（gray 0..928，36 个灰度层）**时 SLM 写入报 `SantecError: 写入相位数据到内存#41失败（错误码 -10032, USB其他错误）` 中止，耗时 529 s。finally 已执行：SLM 关闭 ok / 相机关闭 ok / 锁释放 ok，平场归零失败（同 USB 错误），`release.json.error` 已记录；无残留进程。
- **曝光选点缺陷已修**：首轮直接正式序列在 `0000_flat_r00` 处 6 像素过曝（bracket 把 0.16 ms 选成「最亮不饱和」，贴 0.9 饱和度线、零裕度）。已把选点改为 `choose_exposure`：在「峰值 ≤ 0.7×满量程」的可用档里取最亮，留漂移裕度；配 4 项单测。修正后 32 px 运行到 USB 中止为止未再出现饱和。
- **部分数据不可用于收紧**：诊断拟合 `fit_partial.json` 给出 +1/−1 顶点都被钉在 **928.0（数据边缘）**、折叠 **916 [916,928]**（较 run1 低 ≈46 灰度、边缘受限）——因缺 gray 936..1023（±1 各自极小值 ~951/981 与 2π 回归段），**明确退化**。故 32 px 结论不成立，**g₂π 仍以 run1（16 px 完整序列）为准 = 964 ±13**。
- **当时的后续计划（已由 run4 完成）**：在健康设备会话用 `--period-px 32 --frames-per-level 20` 重跑完整序列（覆盖至 1023）；run4 的完成数据见 §6.4，离线判读见 §6.5。若未来仍需复采，开跑前先验证 SLM 健康（fresh open + 状态/读写自检）；-10032 若复现按已知假死流程手动重启控制器，不反复重试。

### 6.2 run3 周期扫描（32/48 px）— 采集脚本设计（已于 run4 成功执行，见 §6.4）

> 本节记录的 `--run-dir .../run3_period_sweep` 是**初版 dry-run/首次尝试路径**；run3 因抢锁失败、
> 无数据且目录不可复用，实际成功采集在 `run4_period_sweep/`（§6.4）。下列参数设计仍有效。

- **脚本**：`scripts/experiments/run_issue238_period_sweep_20261119.py`（薄封装 `scripts/slm/issue238_gray_phase_probes.py --mode binary`）。
  包一层的理由：探针脚本自带锁与等待，但**一个窗口里连跑 32/48 px 两个周期**需要
  「串行 + 段间冷却 ≥200 s」（背对背 SLM open/close 会触发驱动板假死）；手敲两次会漏掉其中之一。
  **注意（2026-11-19 修订）**：探针自带加锁，因此信封**不能**同时持锁——锁由每个子段自己
  acquire/release，窗口只负责串行、冷却与逐段锁状态留证（首次错误的「窗口持锁」见 §6.3）。
  信封实现见 `scripts/experiments/_window.py`。
- **数据落盘**：`data/experiments/issue238_binary_phase_20261119/run3_period_sweep/period32/`、
  `period48/`（**新根目录 `data/experiments/`**；run1/run2 留在 `data/debug/` 不动，报告与脚本按原路径引用）。
- **默认参数**：`--frames-per-level 20`、`--periods-px 32 48`、`--cooldown-s 200`、`--wait-until-idle 600`；
  曝光仍走探针的 `choose_exposure`（峰值 ≤0.7×满量程取最亮，已修零裕度缺陷）。
- **dry-run**（不建目录、不碰设备）：
  `.venv\Scripts\python.exe scripts/experiments/run_issue238_period_sweep_20261119.py --run-dir data/experiments/issue238_binary_phase_20261119/run3_period_sweep --exposure-bracket 0.02 0.04 0.08 0.16`
- **状态**：✅ **已在 run4 成功执行**（§6.4）；此处原「未采集」为设计初稿状态，仅保留设计参数。
  前置 = 健康设备会话（fresh open + SLM 状态/读写自检通过）+ 用户授权；
  32 px 每周期约 969 帧 / ≈10 min，两周期含冷却约 25 min。判读仍按 §4/§6.1 的折叠/地标口径，
  脚本内**不含**任何拟合（采集与判读分离）。

### 6.3 run3 首次尝试（失败于锁协议，未采一帧）

- **窗口目录**：`data/experiments/issue238_binary_phase_20261119/run3_period_sweep/`，创建于
  `2026-11-19T07:28:35Z`，整体 `EXIT=1`。**不可复用**（信封 `exist_ok=False`），重试须换新目录。
- **失败点**：`segments/period32.log` — 子脚本 `scripts/slm/issue238_gray_phase_probes.py:629`
  的 `acquire_device_lock` 总是尝试 `lock.acquire()`；`--wait-until-idle 0` 只表示「立即失败」，
  **不是**仓库别处假定的「不等不锁」。而父窗口按设计已持同一把 `DeviceLock`，于是子段在
  `0.4 s` 内报 `ERROR: bench device lock stayed busy; refusing to start`，`exit_code=2`、
  `out_exists=false`。**没有打开 SLM/CCD，没有生成任何帧数据**。
- **释放**：`released.json` `released_at=2026-11-19T07:29:04Z`、`lock_released=true`、`error=null`，
  复查 `holder=None`。此次失败是参数/封装缺陷，不是设备故障，不构成 §6.1 的 USB `-10032` 复现。
- **修复要求（已在 2026-11-19 实现并离线验证；无硬件回归见 §6.4）**：自带加锁的采集子脚本**不能**与「父窗口持锁」同时使用；
  正确分工是——子段自己 acquire/release（父窗口转发真实等待时长），父窗口只负责串行与
  段间 ≥200 s 冷却，并在每段结束回读锁状态留证。实现：`scripts/experiments/_window.py` 新增
  `LOCK_MODE_WINDOW`/`LOCK_MODE_SEGMENT`、只读 `open_lock_probe()`/`lock_probe()`、逐段 `lock_after`。
  离线证据：`pytest tests/ao_shaping/scripts/test_experiment_window_lock_mode.py
  tests/ao_shaping/scripts/test_issue238_gray_phase_probes.py -q` → 27 passed；`ruff`、
  `py_compile` 通过；dry-run exit 0 且不建目录、不导入设备模块。
- **run3 状态**：该目录未采集（仅确认封装缺陷）；修复后的重试已在 §6.4 的全新 run4 目录完成。→ **已在 §6.4 的 run4 中完成。**

### 6.4 周期扫描（32/48 px）— run4 已采集完成

- **修复后重跑**：`data/experiments/issue238_binary_phase_20261119/run4_period_sweep/`（全新目录，未复用 run3）。
  `lock_mode="segment"`：窗口不持锁，`period32`/`period48` 子段各自 acquire/release，段间真实冷却 200 s
  （`window.json` 的 `cooldowns[].holder_samples` 全为 null）。窗口 `WINDOW_EXIT=0`，两段均 `status=ok`。
- **period32**：`969` 帧（`960 binary` + `9 flat` 曝光 bracket），`n_patterns=51`；bracket 4 档全可用
  （0.02/0.04/0.08/0.16 ms → 峰值 607/1039/1791/3506），选 `0.08 ms`；**0 饱和数据帧**。耗时 1259.9 s。
- **period48**：`969` 帧（`960 binary` + `9 flat`），`n_patterns=51`。曝光 bracket `0.16 ms` 档
  （`flat`, index=3）峰值 4095、**64 像素饱和**，按 `analysis_policy`「only unsaturated bracket steps are
  usable」判 `usable=false` 并排除；选 `0.08 ms`（峰值 2496）承载正式数据。**960 个 binary 帧全部无饱和**，
  故该段按设计完成，不触发中止。耗时 1438.7 s。
- **设备释放与锁**：两段 `release.json` 均 `slm_flat_gray0=ok / slm_closed=ok / cam_closed=ok /
  lock_released=ok / error=null`；释放时间 period32 `11:51:21Z`、period48 `12:18:39Z`。
  `released.json`：`window_held_lock=false`、`final_state.status=free`、`holder=null`（`12:18:40Z`）；
  窗口结束后独立复核 `DeviceLock.default().holder()=None`（`12:19:13Z`），无 issue238 残留进程。
- **2π 口径记录**：`optical_path_record.json` 保留 `two_pi_grayscale=993`（设备物理 2π 回读）与编码
  `g_ref=1023` 的区分；`summary.json` 记 `device_two_pi_gray_readback=993`、`g_ref=1023`、
  `two_pi_gray_true="pending offline fit"`，符合 `two_pi_grayscale_source_decision_20261119.md`。
- **状态**：✅ 两段采集完成、设备与锁均已正常释放；离线拟合已完成，判读见 §6.5。

### 6.5 run4 离线拟合判读（2026-11-19）

- **复算命令**（纯离线；输出只写 Temp，不改原始 run）：
  `.venv\Scripts\python.exe scripts/slm/issue238_binary_fit.py --run data/experiments/issue238_binary_phase_20261119/run4_period_sweep/period32 --out C:/Users/zhangh/AppData/Local/Temp/opencode/issue238_period32_fit_audit.json --bootstrap 500`
  ；period48 同命令将 run 与输出文件名替换为 `period48`。生产脚本输出的 JSON/Markdown 审计结果保存在 Temp。
- **折叠 / 地标主估计**：period32 `g₂π=952.6`，frame-bootstrap 95% CI `[948,956]`；±1 顶点 `959.6 / 939.8`，系统包络 `[940,960]`。period48 `g₂π=946.9`，frame-bootstrap 95% CI `[945,950]`；±1 顶点 `944.7 / 954.4`，系统包络 `[945,954]`。设备回读均为 `993`，估值差约 `40–46` 灰度；两周期点估计相差 `5.7` 灰度。bootstrap CI 只反映逐帧噪声，不含系统误差。
- **拟合失配与漂移**：理想整曲线 e-fit（只作诊断）给 period32 `888.9`、`rms(e0/e1)=0.2359/0.1180`；period48 `890.9`、`0.1948/0.0974`，不可作为 2π 标定。两段束功率跨扫漂移分别 `+31.53%`、`−13.33%`；±1 极小分裂仍分别约 `19.9` 与 `9.7` 灰度。现有数据提示设备回读 `993` 与光学响应地标存在未解释差异，但本次不满足稳定性/系统误差闭合条件，**不能改 LUT、不能宣称已标定出新的 2π 灰度，也不能据此推进逐像元 LUT/串扰核实现**。
- **issue #238 判定更新**：新数据强化了“灰度—相位标尺存在未解释系统误差”这一问题的真实性；像素级 LUT 与 3×3/5×5 串扰核的算法收益依然未证明。维持 **有效、未完成、OPEN**；不改默认。
- **缺失数据与下一次有效采集设计**：离线先联合分析周期相关的像素模糊/可见度/静态分量，预先冻结选值规则与系统不确定度口径，禁止事后择优；若模型仍无法解释周期趋势，再申请新鲜设备采集。下一次扫描加入短间隔平场锚点、固定灰度重复与伪随机灰度顺序，预注册可接受漂移及拒收阈值；高灰度尾部（约 900–1023）加密并重复，按 sweep/block 留一验证。继续保存完整 Mono12 帧、设备 2π 回读、曝光/饱和/锁释放 provenance。串扰须另用小 cell（如 8–16 px）与窄 gap（0–4 px）的独立 3×3/5×5 几何探针；现有 cell=128/gap=64 不能判决。任何新设备采集仍需另行明确批准。

### 6.6 run4 已有数据联合诊断计划（2026-11-20；纯离线，尚未执行）

**目的与边界**：检查全幅图像是否与 period 相关的空间响应变化相容，并诊断固定孔径拟合中的可见度/背景偏差；本分析不把“模糊、可见度衰减、静态分量”视为可由现有两段数据分别识别的物理机制，不用于更新 LUT、冻结 2π 灰度或宣称像素串扰已定标。现有 `issue238_binary_fit.py` 仅做固定半径 30 px 孔径积分和理想二值光栅能量拟合，不实现这里的联合模型。

**只读输入**：
- `data/experiments/issue238_binary_phase_20261119/run4_period_sweep/period32/` 与 `period48/` 下的 `plan.json`、`summary.json`、`frames.jsonl` 及逐帧 NPZ（原始 Mono12 全幅 `1944×2592`）。每段计划 48 个灰度、每灰度 20 帧；period48 的 0.16 ms bracket 中 64 个饱和像素按原记录排除，不得混入正式 0.08 ms 帧分析。
- 逐帧索引以 `frames.jsonl` 的 `gray`、`period_px`、`frame_file`、`frame_shape`、`frame_dtype`、曝光、时间戳及饱和字段核验；原始数据、清单与 `optical_path_record.json` 一律保持只读。`data/debug/issue238_gray_phase_20261119/run1/` 是 period16 blaze 探针，非同款二值光栅，不作为 run4 的第三个周期点。

**分析流程（实现前冻结，脚本/结果先写 Temp）**：
1. 生成输入清单与哈希；验证两段帧数、shape/dtype、周期/灰度标签、正式曝光、饱和记录和采集顺序。按时间戳确认每个灰度的 20 帧及灰度扫序，禁止把同灰度相邻帧随机拆入训练/验证。
2. 对每帧用同一、预先固定的规则定位 0、±1 级，提取各级完整局部二维轮廓、固定孔径积分，以及远离衍射级的背景区域统计；不得根据拟合结果删帧、改 ROI 或挑灰度。将图像对齐、裁窗、背景扣除等参数与失败规则写入配置和结果。
3. 比较受限的嵌套诊断模型：M0 为共享空间轮廓/响应，M1 允许 period32/48 有不同的空间宽度或低维轮廓项；两者均显式容纳帧级乘性强度/慢漂移项及背景/静态加性项，并将归一化可见度指标作为观测量而非预设成因。报告参数相关性、profile/边界、残差空间图及 aperture/background 敏感性；复杂模型拟合内误差更低不算证据。
4. 验证与不确定度以连续时间/灰度块为单位做留块预测和 block bootstrap；不得按帧随机拆分或把 20 帧当 20 个独立硬件重复。由于 period32/48 各自只有一个分时段 sweep，period 与采集时段/漂移完全不可独立解混；即使留块预测改善，也仅报告“与周期相关的空间响应项相容”。
5. 记录不可辨识项：两周期点不足以验证周期函数；`flat` 是照明下零灰度帧、不是遮光暗场；光路记录缺相机 gain、pixel pitch 等若干常数。因此不能单凭拟合把空间宽度解释为绝对 PSF/像素模糊，或把低频项解释为独立静态偏置/串扰。

**产物与门槛**：第一轮审计 JSON/图表放 `C:/Users/zhangh/AppData/Local/Temp/opencode/issue238_run4_joint_diagnostic_20261120/`，所有命令显式指定 Temp 输出，禁止沿用 `--out` 默认写回 run 目录。只有在输入完整性检查通过、按时间块验证集仍有稳定预测改善、且背景/ROI/漂移敏感性下参数不贴边且方向一致时，才写永久诊断报告；报告放 `report/shape_trials/`，须在 `scripts/_common/provenance.py::REPORTS` 登记精确脚本/复现命令、离线+硬件环境与数据来源，再执行 provenance sync/check 和对应测试。若这些条件未满足，结论记为“当前 run4 无法区分机制”，按 §6.5 的交错低漂移新采计划补数据；设备采集仍须用户另行明确批准。

### 6.7 执行结果：第一阶段（清单核验 + 固定 ROI 描述量）已执行；M0/M1 未实现，永久报告门槛未触发（2026-11-20）

**脚本与命令**（纯离线；未 import 硬件模块、未碰 `DeviceLock`、未写回原始 run）：
- `scripts/slm/issue238_run4_spatial_diagnostic_20261120.py`（SHA256 `791fc5d48dc33b841356bb92f74ce6c7dbc2fd962a4d58a7d182cc6d26f78374`，89,704 bytes）
- 自测：`--self-test` **6/6 通过**（几何/测光、清单正例与三类篡改、临时目录端到端、verdict 抑制、`--out` 守卫、NPZ `allow_pickle=False`）；`ruff check` clean；`py_compile` exit 0；无 BOM。
- 正式运行：`.venv\Scripts\python.exe scripts/slm/issue238_run4_spatial_diagnostic_20261120.py --out C:/Users/zhangh/AppData/Local/Temp/opencode/issue238_run4_joint_diagnostic_20261120/diagnostic.json`
- 产物：同目录 `diagnostic.json`（SHA256 `a93db533ae41dc6a6f9ec611bde8c17aa82f496283457cd7b8775484412bde34`，4,838,353 bytes ≈ 4.8 MB）+ `run.log`。**未写入 `report/`**，原因见下。

**输入完整性（两段均通过）**：period32/period48 各 `n_analyzed=960`、`levels=48`、`frames_failed=0`、`analysis_complete=true`；逐帧检查项（`frame_shape` / `npz_scalars` / `meta_json` / `peak_recomputed` / `saturation_recomputed` / 曝光 / 采集顺序 / 行数）全部 ok，`manifest` 与 `summary` 一致。两段均流式逐帧处理全幅 `1944×2592` uint16（单段约 80–85 s，首轮计时；r50 修复只在末尾累计、不触及逐帧主路径，重跑耗时同量级；峰值内存未超限）。

**描述量：孔径捕获比例随灰度变化（两段形状一致）**。`capture_small` = r=30 px 圆盘净能量 / r=120 px 圆盘净能量，`capture_large` 为 r=60 px 版本；背景为每帧全幅远场中位台阶（`bg_pedestal`），局部环仅作 `*_altbg` 敏感性对照。抽点（`+1` 级，`capture_small`）：

| gray | 0 | 128 | 256 | 384 | 448 | 512 | 640 | 768 | 896 | 1023 |
|---|---|---|---|---|---|---|---|---|---|---|
| period32 | 0.038 | 0.663 | 0.825 | 0.860 | 0.861 | 0.850 | 0.829 | 0.703 | 0.377 | 0.433 |
| period48 | 0.052 | 0.676 | 0.833 | 0.854 | 0.853 | 0.848 | 0.809 | 0.667 | 0.287 | 0.453 |

两段都在 gray≈384–512 达到峰值、向 0 与 1023 两端回落，是二值相位光栅衍射效率随编码灰度变化的定性形状；**本报告不对其作机制归因**。period32/period48 的 `zero.capture_small` 分别中位 0.882 / 0.868，`zero.rms_radius_px` 17.100 / 17.087 px（差 `−0.111`，CI `[−0.306, 0.128]`，跨 0）。

**跨周期差异（块内连续灰度块 bootstrap，400 次，seed=238，6 块 × 8 灰度）**：

| 指标 | period32 | period48 | Δ(32−48) | 95% CI |
|---|---|---|---|---|
| `zero.capture_small` | 0.8817 | 0.8681 | +0.0148 | [+0.0046, +0.0220] |
| `zero.capture_large` | 0.9608 | 0.9452 | +0.0181 | [+0.0027, +0.0224] |
| `plus1.capture_small` | 0.5417 | 0.4808 | +0.0050 | [−0.0050, +0.0592] |
| `plus1.capture_large` | 0.6382 | 0.5996 | +0.0145 | [−0.0012, +0.0467] |
| `minus1.capture_small` | 0.5406 | 0.4123 | +0.1015 | [+0.0411, +0.1548] |
| `minus1.capture_large` | 0.6494 | 0.5211 | +0.1072 | [+0.0463, +0.1604] |
| `zero.profile_l1` | — | — | 0.0559 | [0.0533, 0.0675] |
| `plus1.profile_l1` | — | — | 0.1155 | [0.0758, 0.1860] |
| `minus1.profile_l1` | — | — | 0.2676 | [0.1228, 0.3341] |

**判读边界（必须与数字一起读）**：
- period 与采集时段、束功率慢漂移**完全共线**（两段各只有一个顺序 sweep），故上述 Δ **不能归因于 period**；§6.5 已记两段束功率跨扫漂移 `+31.53%` / `−13.33%`，与 Δ 的符号/量级同阶。
- `−1` 级位置由 `+1` 点对称给出（§6.6 冻结规则），其 Δ 大于 `+1` 也可能是定位/ROI 差异的投影，不能读成物理不对称。
- `r50_px`/`r90_px` 端点**仍不作结论量**，但原因已与初版不同：初版对未截断的 `patch − bg_pedestal`（含负值）求累计，累计曲线非单调、`np.interp` 给出无意义值（曾出现 period48 `minus1.r50=93.4 px` ≫ 同帧 `rms_radius=25.6 px`）；本轮已改为对 `clip(·,0,None)` 求累计，并在自测加「`r50 ≤ r90 ≤ aperture_r`」硬检查后整段重跑，端点现单调且落在孔径内（`zero.r50` 9.97/10.18、`plus1.r50` 80.0/89.4、`minus1.r50` 78.0/95.7 px，均 ≤ 120 px）。但把背景残差截断为非负会在半径大处累积正的背景噪声，把 `r90` 抬到大孔径边缘（period32 `plus1`/`minus1` 的 `r90` 中位最高达 119.7 px），这与同帧 `capture_small=0.86`（r=30 净能量已占 86%）自相矛盾——说明 `r50/r90` 反映的是「截断后正的背景残留」，不是真实 50%/90% 能量半径。故本报告只用有符号圆盘净能量 `capture_small`/`capture_large` 作比较量，`r50/r90` 仅留在 JSON 作原始记录，**不得引用、不进入任何结论**。
- 20 帧/灰度是**块内重复**，不是 20 次独立硬件重复；bootstrap 单位是连续灰度块，未按帧随机拆分。

**为什么没写永久报告（§6.6 门槛未满足）**：脚本显式 `stage="phase1_only"`、`algorithm_benefit_verdict=null`、`verdict_emitted=false`，`m0_m1.implemented=false`。M0/M1 嵌套模型（共享轮廓 + period 相关低维项 + 帧级乘性漂移 + 加性静态项）**未实现**，因此不存在“按时间块验证集的稳定预测改善”这一放行证据；单看上面的描述量 Δ 不满足 §6.6 的“参数不贴边且方向一致”要求，更不构成机制证据。按 §6.6 的约定，此时结论记为**“当前 run4 无法区分机制”**，补数据需求同 §6.5（每周期至少两个独立时段、遮光暗场 + 参考功率监测），且任何采集须用户另行批准。

**对 #238 判定的影响：无变化**。仍为“有效、收益未证明，仍不可定标”：不更新 `docs/slm/slm_lut.md` 工作值、不冻结物理 2π 灰度、不宣称串扰已定标。本轮唯一增量是——run4 全幅原始帧的清单/曝光/饱和一致性得到独立机器核验（960/960 ×2，零丢帧），以及固定 ROI 描述量有了可复现的基线数值。

### 6.8 功率参考（ABBA）与回文周期序已落地（离线，2026-11-20）

针对 §6.7 “period 与束功率漂移共线”这一判读边界，给既有采集链加了**可选**漂移解耦能力（默认关闭，历史 plan 逐字节不变）：

- **采集脚本 `scripts/slm/issue238_gray_phase_probes.py`**：新增 `--power-reference {off,abba}`（默认 `off`）、`--power-reference-frames N`（默认 2）。`abba` 时每个 sweep block 前后各插 `N` 帧 `flat`(φ=0) 参考帧（`role=pre/post`、`block_index`、`ref_slot`、单调 UTC 时间戳、固定曝光、同 uint16+sha256）。**全程单一曝光**，出现第二个曝光立即中止并写 `power_reference_abort.json`（`abort` + 原因 + 观测曝光集）后正常释放。`summary.json` 仅在启用时增 `power_reference` 段：每块 pre/post 原始总计数、`drift_ratio_post_over_pre`、逐帧 `scale`、`descriptive_only: true`、`dark_subtracted: false`；**不在脚本内计算或声明新 g₂π**。
- **估计器（纯函数，可 import）** `interpolate_power_scale(frame_times, ref_times, ref_totals)`：`r_j = ref_total_j / mean(所有 ref_total)`，帧时刻按括号参考线性插值，返回 1.0 = 全程平均功率；区间外**常数保持**最近一侧比值（不做线性外推，避免无括号处放大噪声）。
- **窗口封装 `scripts/experiments/run_issue238_period_sweep_20261119.py`**：新增 `--period-order {as-given,palindrome}`（默认 `as-given`）、`--power-reference`、`--power-reference-frames`。`palindrome` 把周期展开为如 `32,48,64,64,48,32`（奇数长度中心周期单份），使 period 与漂移可分离；`validate()` 拒绝 <2 个不同周期的回文、`abba` 下 `--power-reference-frames < 1`；`off` 不转发两旗标，plan-only 无副作用。
- **验证（离线）**：`test_issue238_power_reference_20261120.py` 14 passed；连带既有 `test_issue238_gray_phase_probes.py` 合计 30 passed；`py_compile`/`ruff`/两脚本 `--help` 全 0；`off` 计划把唯一 `timestamp_utc` 归一后与改前**文本逐字相等**；plan-only 不建目录；无 BOM。`graphify update .` 已跑。
- **仍未做**：尚未按新口径采数；`power_reference` 只是描述性记录，正式漂移校正与 g₂π 重估须离线完成，且以“每周期≥2 独立时段 + 遮光暗场 + ABBA 参考”为前提。任何采集须用户另行明确批准。

**对 #238 判定的影响：无变化**。仍 OPEN、不升默认、不冻结物理映射；本轮增量是“下一次采集已具备可分离 period/漂移的机制”，而非任何物理结论。
