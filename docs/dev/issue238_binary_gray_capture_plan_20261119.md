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
