# Agent 注意事项（2026-11-19 当日修改）

> 按日追加；过期条目迁入 `docs/dev/` 对应专题。

## #238 灰度探测：2π 工作值、光学拟合简并与反演公式纠错

- **设备占用/释放**：Santec SLM #1（1064 nm，memory 模式）+ Daheng CCD #0；09:52:38 开、09:55:10 释放，全程单进程、无并行设备进程，相位归零与设备关闭正常。本轮仅记录既得数据，未再操作设备。
- **数据**：`data/debug/issue238_gray_phase_20261119/run1/`，155 帧 Mono12、n_patterns=76、error=null、曝光 0.04 ms（峰值 2107 < 8-bit 饱和线 3685）。该数据为 12-bit，与 11/18 的 8-bit 量纲不可直接比。

### 结论 1：2π 灰度工作值取设备回读 993

- 1064 nm 下设备相位表回读 phase=206 ⇒ 2.06π ⇒ **993 灰度**；这是工作值，不是 radian→gray 标尺上限 1023。设备会拒绝更小的值（196/200 均吸附回 206），因此不能强设为 1023。零级地板在相位 ≥ 2.06π 时饱和于约 2.86%（ff≈0.854、η±1_max≈97.1%），与设备值自洽。

### 结论 2：光学拟合在 γ–g₂π–对比度 简并下只能给区间

- 跨灰度幂律给出 938±11、CI[917,961]；885 与 1023 在模型内被强排斥。**此前峰值法结论不成立，正式撤回**，不得再作为判据引用。
- 简并根因：γ（灰度→相位非线性）、g₂π（2π 对应灰度）与条纹对比度在单组闪耀光栅效率数据中同时进入拟合，三者不可分辨；只测 +1 级或只用峰值位置无法把它们拆开。

### 结论 3：相位反演公式曾写错（勿重犯）

- **错误做法**：用 `cos(phi) = (e0n - A) / B` 反演，其中 `e0n` 是按实测级次和归一化的 `η₀/(η₀+2η₁)`。
- **错在哪里**：归一化分母 `η₀+2η₁` 本身随灰度 g 变化，`e0n` 不是绝对效率，A/B 定标失去物理意义；在这组归一化量上，`cos(phi)` 的线性关系不成立，反演会把归一化造成的漂移误当成相位响应变化，从而加重 γ–g₂π–对比度 简并（表现为只能给区间、峰值法失效）。
- **正确做法**：对绝对级次效率直接拟合。对称二值相位光栅的精确傅里叶结果为 `η₀ = cos²(φ/2)`、`η₁ = (4/π²) sin²(φ/2)`；用绝对 η₀（而非归一化 e0n）做 e 域最小二乘，φ(g) 无偏，第二零点直接给出 g₂π 且与 γ 无关。

### 结论 4：串扰判「未检出 / 不可判定」

- 该次探针 cell 128 px / gap 64 px，3×3、5×5 像素核跨不过间隙，观测被稀释约 16×。因此现有数据**不能**用来宣称无串扰；结论只能写「未检出、不可判定」，后续需换探针几何重测。

### 结论 5：破简并方案（用户授权 B，离线已验证）

- 改用对称二值相位光栅，逐点读出绝对相位 φ(g)，第二零点直接定 g₂π，与 γ 无关；多帧平均抑制散粒噪声。
- 离线 e 域直接拟合无偏、SD<1 灰度：真值 847→847.0、885→884.9、938→938.0、993→992.9、1023→1023.1（20 帧平均）；993 vs 938 分离 16σ、938 vs 885 分离 11σ，足以破简并。验证脚本：`C:\Users\zhangh\AppData\Local\Temp\opencode\binary_validate238{,b,c,d}.py`。
- 采集脚本 `scripts/slm/issue238_gray_phase_probes.py` 已新增 `binary` 模式（`binary_gray` / `binary_sweep_levels` / `_build_binary_plan` / CLI 参数），尚未提交；v2 二值光栅采集尚未执行。新采集须用独立 `--out`，不得复用 run1 目录。
- 设备行为待确认：g > g₂π 后是饱和还是缠绕，决定用全点拟合还是仅首支拟合。

### 结论 6：二值光栅实测修订（10:39–10:50 UTC）

> 本节修订上方结论 5 中“逐点绝对相位 / SD<1 / 足以破简并”的理想仿真外推；那不是实测结论。

- **设备占用/释放**：Santec SLM #1 + Daheng CCD #0 + 1064 nm 激光，10:39 左右开始、10:50:12 UTC 释放；单进程，无并行设备进程；SLM 灰度归零、SLM/CCD 关闭、lock 释放均 `ok`，`error=null`。未再占用设备。
- **数据**：`data/debug/issue238_binary_phase_20261119/run1/`，777 帧 / 48 灰度 × 16，Mono12，0.04 ms；曝光峰值 2048，无饱和；扫前后 0 级漂移 −0.56%。
- 对称二值光栅数据支持 **g₂π≈964 灰度**，但只应报告约 **±13 灰度系统 allowance**（由 +1/−1 级 minima 分离决定，约 951–978），不能把仅含帧噪声的 fold bootstrap CI [959,966] 当作总不确定度。平均级次折叠 962.6、e1 极小抛物线 962.4、两级各自极小位置中点 964.5、设备码线性外推 964.1 相互吻合；但 +1 和 −1 单级极小分别约 978 和 951，显示真实灰度相关不对称。
- 条件性 landmark 模型给 γ≈0.91；以 g₂π≈963 预测灰度 993 相位 2.057π，与设备回读 2.06π 自洽。**仅能说二值光栅 landmark 模型下 γ–g₂π 简并显著缓解，不可宣称 model-independent 绝对相位标定**。
- 理想全曲线 e 域拟合 g₂π≈898，残差 rms≈0.18，拒绝为校准估计；含 0 级 pedestal 的全曲线拟合 g₂π≈1009 虽 rms≈0.009，但错过实测首级 minima（预测在约 1009、观测在约 960），同样不采纳。二者反映全曲线模型对未建模的 0 级背景/级次不对称敏感。
- 二值中心值 964 与既有 blaze 拟合 938、CI[917,961] 有中心差异但区间仍重叠，不得声称 blaze 区间已被决定性排除；885 与 1023 仍受现有 blaze 与二值数据共同反对。更大周期探针只在需要更强、model-independent 的精密标定前再采。

### 结论 7：mil 分支成果迁入 master 并退役（离线，未占用设备）

- 算法验证包（脚本 / 权威 JSON / NPZ / 权重 / 探针数据 / 术语 T-77 / 设备锁证据 / 残差基热启动用例）已逐路径迁入 master，权威数值以原始 JSON/NPZ 为准；适配到 master API 的诊断脚本重跑输出带 `*_master_api` 后缀。provenance `--check` 为 clean。
- 未并入的两处 mil 实验脚手架（`physical_forward_fit.residual_modes`、`fit.py` 的 blur/static/sigma_ccd 观测模型）已原样归档为 `report/shape_trials/physical_refine/mil_only_experimental_20261119.patch`，随后删除 `.worktrees/mil` 与 `mil` 分支；mil 独有提交 `2e39dd25`（provenance REPORTS 条目错位修复）在 master 工作树中已具备等价内容。
- 新增可选入口 `scripts/slm/refit_transfer_forward.py`（train-only 拟合、calibration 排除、test 单次评分、`--out` 不覆盖、`--dry-run` 不落盘），冒烟 1 步：B test corr 0.7228→0.7417。**不改** optimizer 默认更新规则。
- 边界未变：目标形状 CCD 验收**仍不放行**。此后已补齐代码侧缺口（见结论 9）：measured-vs-target RMSE 指标与带锁 replay 外壳均已实现并测试，但真实 CCD 实测/阈值预注册/独立证据仍缺。

## 真实暗场采集、扣暗无效与暗帧配置库（用户仅授权一次采集，硬件已停）

- **设备占用/释放**：Daheng CCD #0，仅相机（无 SLM / 不动激光，用户确认激光已关），三档曝光各 16 帧；采集完成后相机正常关闭、bench DeviceLock 已释放；此后未再占用任何设备。原始帧落在 `.worktrees/direction-model-class-20261118/data/debug/darks_20261119_laseroff/`（`dark_0.024.npy` / `dark_0.043.npy` / `dark_0.047.npy`，各约 38 MB）。这是本轮**唯一**一次被授权的硬件采集，之后暂停主动采集，重测曝光线性度 / 重采 pass 均须另行授权。
- **暗场实测**：三档（设备回读曝光 0.024 / 0.043 / 0.047 ms）全帧均值 0.2666 / 0.3055 / 0.3054 counts，峰值 1.9 / 2.0 / 2.0 counts —— 暗场近零，随曝光基本不变（读出噪声底）。
- **扣暗不能救线性度**：扣暗后阶梯峰值 `[129.8, 193.2, 196.9]`，相邻比值 `[1.489, 1.019]`；0.043→0.047 ms 峰值比 1.019 远小于曝光比 1.093（高曝光档响应滚降到约 200 counts 平台）。因此 `non_monotonic` 结论不变，不按原计划重跑原判据。若强行要求翻转这段单调性，需对所有档统一扣除约 `b=143.7` counts 才能把 193.2/196.9 拉回递增 —— 与暗场实测（≈0.3 counts）矛盾，物理不成立，故否决。
- **结论归属**：高曝光档滚降是传感器/前端在短曝光下的真实响应平台，不是暗场偏置；曝光线性度修复（pass2 固定曝光 + 逐帧回读，提交 `df0ea76d`）与双门评估脚本（`f9220430`）均未推送。**双门未联合通过前不放行锚定闭环**；零步绝对强度门无预注册数值门槛（0.25 仅为诊断线）。验证帧不得用于配准或强度门参考。
- **暗帧配置库（新增，离线）**：`drivers/ccd/dark_frames.py` 按序列号分目录存暗帧配置，`<CCD_CONFIG_DIR>/ccd/<serial>/darks/{manifest.json, dark-exp-<expo>ms.npz}`，manifest (`ao_shaping.ccd.dark_frames/v1`) 为唯一索引，曝光存设备回读真值，默认 `rel_tol=0.02` 严格匹配、不跨曝光回退（`allow_nearest=True` 须显式）。`BaseCamera` 新增 `serial_number` / `load_dark_frames()` / `get_dark_frame()` 入口（无序列号报错/返回 None，空集合法）。采集 CLI `tools/ccd/capture_darks.py`（多曝光 + `--no-hw` + 设备锁 + 曝光回读）与离线测试同步落地。配置数据属运行时资产（`.npz` / `data/*` 被 gitignore），默认不入库。
- **三档真实暗帧已离线导入配置库**（无设备参与）：以实测序列号 `FJB24112232` 为键写入
  `data/ccd_configs/ccd/FJB24112232/darks/`（`frames=16`、`bit_depth=8`、`full_scale=255`，
  阵列取 16 帧均值），`BaseCamera.load_dark_frames()` 与 `get_dark_frame(0.043)` 已实测可载入；
  0.030 ms 严格匹配返回 `None`（不跨曝光回退），`allow_nearest=True` 才退到 0.043。
  原始 `.npy` 与本次导入内容逐位同源，仅保留作证据。


### 结论 8：32 px 二值光栅尝试（run2）— SLM USB 中止、部分数据退化（离线判读 + 中止记录）

- **设备占用/释放**：SLM #1 + CCD #0 + 1064 nm 激光，单进程；run2 于 11:35 起采，至第 725/969 帧（gray 0..928）时 SLM 写入报 `SantecError: 写入相位数据到内存#41失败（错误码 -10032, USB其他错误）` 中止（耗时 529 s）。finally 已执行：SLM 关闭 ok / 相机关闭 ok / 锁释放 ok，平场归零失败（同 USB 错误）；`release.json.error` 已记录，无残留进程。此后未再占用设备。
- **曝光选点缺陷（已修，须记住）**：原 `choose_exposure` 取「最亮不饱和」档，贴 0.9 饱和度线零裕度——bracket 0.16 ms 峰值 3519 判不饱和，正式平场重拍 3712、6 像素越线。改为在「峰值 ≤ 0.7×满量程」的可用档里取最亮（`PREFERRED_PEAK_FRACTION=0.7`），留激光暖机/热点漂移裕度；配 4 项单测。任何新采集不得再选贴饱和线的档。
- **部分数据退化、结论不成立**：run2 只到 gray=928，缺 936..1023（±1 各自极小 ~951/981 与 2π 回归段）。`fit_partial.json` 给出 +1/−1 顶点都被钉在 928.0（数据边缘）、折叠 916 [916,928]（较 run1 低 ≈46 灰度）——边缘受限、明确退化。故 **32 px 未收紧 g₂π，工作值仍取 run1 = 964 ±13**。
- **下一步（需健康设备会话，非本轮）**：`--period-px 32 --frames-per-level 20` 重跑**完整**序列覆盖到 1023；开跑前先 fresh open + 状态/读写自检，-10032 若复现按已知假死流程手动重启控制器，不反复重试。计划见 `issue238_binary_gray_capture_plan_20261119.md` §6.1。

### 结论 9：补齐目标形状验收的代码侧缺口（离线，未占用设备）

- **measured-vs-target intensity RMSE**：`protocol.target_intensity_rmse`（整窗两侧各自归一化后相减，背景 `floor` 先扣、`None` 表示无能量——与 cv=None 同约定），`replay_phases` 在每条记录写入 `target_rmse`（grid 不一致时为 `None`，不在采集中断）。冻结契约 `TARGET_INTENSITY_KEYS = (rmse, rel_l2, energy_fraction)`；8 项单测（radiometry 不变性、目标外能量惩罚、背景扣除、无能量/grid 拒绝等）通过。
- **带锁 replay 外壳**：新增 `scripts/slm/replay_transfer_shapes_hardware.py`，复用验证器外壳的锁协议；消费 `--execute` 门禁（不转发给 replay 模块）、强制 `--out` 与 `--exposures` 显式、等空闲+持锁贯穿重放+finally 释放+写 `device_lock.json`；无 `--execute` 拒绝且不导入设备模块。16 项 mock 测试通过，真实 `--help`/拒绝路径实际执行通过；**`--execute` 从未执行**。
- 验收计划 `target_shape_ccd_acceptance_plan_20261118.md` 与主报告 §11.3 已更新：定义 RMSE 归一化/背景/阈值边界（阈值须现场预注册，`rel_l2` 上限 + `energy_fraction` 下限，冻结后只评分一次）；明确**代码就绪 ≠ 验收通过**。provenance `--check` clean、相关报告定向 8 项通过；`protocol` 单测 51 项、transfer/replay 汇总 522 passed + 1 skipped。
- 边界不变：无任何目标形状 CCD 实测、无阈值预注册、无独立证据，**不放行**；全部改动未提交、未推送。

### 结论 10：把两份计划的实验增量落成一等窗口脚本（未占用设备）

- **新增 `scripts/experiments/`**：`_window.py`（共用信封：`wait_until_idle` → 段间冷却默认 200 s → 每段独立 `--out` → 写 `released.json`/`window.json`；锁归属由 `lock_mode` 决定：**初版持锁贯穿全窗口**，后因抢锁失败改为 `segment` 子段各自取放，见结论 11 与 `issue238_binary_gray_capture_plan_20261119.md` §6.3/§6.4）、`run_transfer_window_20261119.py`（`k1`/`k4`/`light`/`none`，仍默认 `window` 持锁）、`run_issue238_period_sweep_20261119.py`（`period32`/`period48`）。后者必须整目录全新跑，因为 `run2_period32` 在 725/969 帧 USB `-10032` 中止、数据退化，不得复用目录。
- **为什么要包一层**：`python -m ao_shaping.tools.slm.transfer run` 既不拿 bench `DeviceLock`、也没有 `--wait-until-idle`；背对背 SLM open/close 会触发驱动板假死 `ReadSU=-1000`。
- **门禁教训（记住）**：`scripts/README.md` 属 *live text*，被 `test_quoted_source_paths_in_live_text_exist_in_the_git_index` 与 `test_python_invocation_paths_in_docs_and_docstrings_are_tracked` 扫描；反引号里写**未进 git 索引**的脚本路径（例如别的会话新建、还没提交的 `scripts/slm/*.py`）会新增一条 offender，改成只写目录级路径即可。
- **术语门禁教训**：`scripts/_common/term_unify_20261016.py` 的词替换规则会把已成句的行改成病句（一个词被替换成包含它的更长词，或整词被换成动词后句子不通）。新写的中文注释只能改写措辞，不能照搬替换结果；自查办法是把目标文件逐行喂给 `process_line`，比对是否有变化。本次 3 个新脚本与 §7/§13 增补均已 term-clean。
- **base 既存失败（非本任务回归）**：`test_conventions.py` 16 项、term gate 2 项，offender 全在 `.worktrees/`、`data/debug/`、`docs/dev/agent_notes_*` 与各计划原文行。
- **纪律不变**：新采集统一落 `data/experiments/`，历史 `data/debug/...` 原位不动；全部改动 `--execute` 门控，本轮无 `--execute`、未占用 SLM/CCD/激光。

### 结论 11：P=32/48 周期扫描（run4）已采集并完成离线拟合（判读见 issue238 采集计划 §6.5）

- **修复后重跑**：`data/experiments/issue238_binary_phase_20261119/run4_period_sweep/`（全新目录，不复用 run3）。`lock_mode="segment"` —— 窗口不持锁，`period32`/`period48` 子段各自 acquire/release，段间真实冷却 200 s（`window.json` 的 `cooldowns[].holder_samples` 全为 null）。窗口 `WINDOW_EXIT=0`，两段均 `status=ok`。
- **设备占用/释放**：Santec SLM #1（1064 nm，memory）+ Daheng CCD #0；`11:30:21Z` 起，`period32` 于 `11:51:21Z` 释放、`period48` 于 `12:18:39Z` 释放。两段 `release.json` 均 `slm_flat_gray0=ok / slm_closed=ok / cam_closed=ok / lock_released=ok / error=null`。窗口结束后独立复核 `DeviceLock.default().holder()=None`（`12:19:13Z`），`released.json` `final_state.status=free`、`holder=null`（`12:18:40Z`），无 issue238 残留进程。
- **数据量**：两段各 `969` 帧（`960 binary` + `9 flat` 曝光 bracket），`n_patterns=51`、`bit_depth` 断言 12/12、`error` 无。
- **period32**：bracket 4 档全可用（0.02/0.04/0.08/0.16 ms → 峰值 607/1039/1791/3506），选 `0.08 ms`；0 饱和数据帧。耗时 1259.9 s。
- **period48**：bracket `0.16 ms` 档（`flat`, index=3）峰值 4095、**64 像素饱和**，按 `analysis_policy`「only unsaturated bracket steps are usable」判 `usable=false` 排除；选 `0.08 ms`（峰值 2496）承载正式数据，**960 binary 帧全部无饱和**，故**不触发中止**、按设计完成。耗时 1438.7 s。
- **2π 口径**：`optical_path_record.json` 保留 `two_pi_grayscale=993`（设备物理 2π 回读）与编码 `g_ref=1023` 的区分；`summary.json` 记 `device_two_pi_gray_readback=993`、`g_ref=1023`、`two_pi_gray_true="pending offline fit"`。符合 `two_pi_grayscale_source_decision_20261119.md`。
- **离线拟合已完成**（生产脚本 `scripts/slm/issue238_binary_fit.py`，bootstrap=500；判读详见 `issue238_binary_gray_capture_plan_20261119.md` §6.5，审计 JSON 在 Temp）：period32 fold `g₂π=952.6`、CI [948,956]，period48 `g₂π=946.9`、CI [945,950]；两周期点估计相差 5.7 灰度，均低于设备回读 993。束功率漂移 +31.53%/−13.33%、e-fit 残差高、±1 极小仍系统分裂，CI 只含逐帧噪声。因此只记为「存在未解释系统差异」，**不改 LUT、不宣告新 2π 标定、不推进串扰核**；run1(16 px) 的 964±13 只作历史对照，新证据既不收紧也不替代它。`#238` 保持 OPEN。

## Transfer 模型资格复核（2026-11-19；硬件暂缓）

- **冻结解析模型 `FrozenDefocusForward` 的方向门失败是模型类问题**：已有噪声预算、7×7 交叉匹配、参数扫描与第二 pass 证据排除了噪声、索引错位及样本不足；方向响应的主导 PC1 符号/幅度不匹配，失败数在模型变体中封顶。详见 `agent_notes_20261118.md` §“方向门失败机制归因” L44-L72。
- **原强度数值不能判为通过**：`relL2=0.355` 是 in-sample 零步结果、无 held-out；`adaptation.py` 的 0.25 是 escalation heuristic，不是预注册放行门。因而原所谓“双门互斥”只描述现有冻结模型及不完整强度判据下的结果，不能推广成模型类普遍结论，也不能据此放行。
- **`LocalResponseModel` 是诊断性替代模型，不是闭环放行证据**：隔离 worktree 报告 `direction_model_class_20261118.md` §0 报告形式 OOF 12/14、median 0.8378；两折互为数学镜像，独立证据仅 6/7 模式跨 pass 可复现，Z11 未复现。held-out 强度相关 0.9885–0.9964，但未缩放 relL2=0.217–0.261 跨越 0.25 启发式，且该阈值未预注册；报告也指出跨 pass 强度标度有采集侧不自洽。因此不得把 12/14 或相关系数当作锚定闭环放行结论。
- **独立轴失配假设为阴性**：合并计划 §2 的 D2′ 验证集 `axis − baseline` r² 差为 −0.00348，95% CI [−0.00824,+0.00160] 含 0；不得再把物理 xy 轴失配写作方向失败解释。
- **1200 px 计划参数边界**：历史几何的离线 `plan` 与 640 网格导出只证明参数/几何可行，不证明 checkpoint 有科学资格、也不证明实验质量。
- **下一步（先纯离线；transfer 硬件验证仍暂停）**：强度门 `δ=0.20` 与联合规则已预注册；仍需锁定具体模型定义/权重、训练/验证拆分与脚本哈希，再只用归档数据复核 `LocalResponseModel` 的拆分和曝光配对。现有两 pass 不能制造 14 个真正独立 OOF 对；若需确认模型泛化，须先有符合本预注册的数据方案。双门/模型资格路径未澄清前，transfer `light/none` 继续暂停，不以工程验证混同科学验收。

### Transfer 强度门预注册（2026-11-19；δ 已冻结）

> 完整方案：[`transfer_strength_gate_preregistration_20261119.md`](transfer_strength_gate_preregistration_20261119.md)。
> 用户授权自行调研并设定相对合理值后，δ 冻结为 `0.20`；依据与证据边界见方案 §6。冻结方案正文 SHA-256：`DF91919FBCE88CD5059478BCBA8C978050E4B60A0DDF88CB5AB825B38DC1F790`。

- **主指标**：固定目标 ROI 与 mask 上，预测零步绝对强度图对独立实测图的未缩放 `relL2`；不允许事后拟合全局增益、重新中心化或挑选曝光。Pearson 相关仅作空间形状诊断，不替代绝对强度误差。
- **测量有效性先决条件**：模型/权重、ROI/mask、曝光档与 `exposure_ms` 记账、暗场处理、位深/满量程、settle 判据、脚本哈希在验证数据采集前锁定。任一 pass 曝光记录与实测不一致、饱和或 settle 失败，该 pass 标为 `not assessable`，不得从该 pass 宣告通过/失败。
- **独立性与汇总**：冻结后的模型只评估新采集的 hold-out；两个方向互换的 cross-fit 不重复计数。每个独立 pass 分开计算 `relL2`，采用最差 pass 而非合并平均，避免一个 pass 的好结果掩盖另一 pass 的失败；重复帧只用于估计该 pass 的测量不确定度，不当成独立实验单位。
- **暂定联合规则**：方向门沿用已预定的 10/14，但仅对锁定模型的真实独立 hold-out 对计数；强度门要求所有有效独立 pass 的 `relL2 ≤ δ`。两门均通过且所有测量有效性前置条件通过才可放行；否则分别报告 fail 或 `not assessable`。本规则不放行未经单独计划的锚定闭环/混合模式外推。
- **强度门已预注册，尚未用于评价既有数据**：`δ=0.20` 是按用户授权自行探测后设定的工程容差，以本项目 `light` 相对 `full` 的 CV 退化 `<20% relative` 应用要求作为同量级依据（非数学换算）。CV 与像素级 `relL2` 不等价；既有约 8% 跨 pass 差异不是本门 `relL2` 噪声分布。历史重复帧 scale-free `rel_l2` 极差 `0.0001–0.0011` (`historical_measured_shape_metrics_reanalysis_20261119.md:32,48`) 也与本门未缩放口径不同，不能直接校准 δ。首次新验证须披露本门口径下的重复性；若测量有效性/重复性不足，报告 `not assessable`，不得事后放宽 δ。详情见 `transfer_strength_gate_preregistration_20261119.md` §6。



### 结论 9：12-bit 同曝光暗场补采与 floor 冻结（用户授权立即采集，激光已关）

- **设备占用/释放**：Daheng CCD#0（`FJB24112232`），仅相机、12-bit（`full_scale=4095`），未占用 SLM、未动激光（用户已关）；占用 20:58:37→20:59:49 CST（持 `DeviceLock`），`with create_camera` 退出正常 `close()`、锁释放、无残留进程。采集 CLI `capture_darks.py` 本次已加 `--bit-depth` 并做设备回读校验。
- **归档**：8 档曝光（0.02/0.04/0.06/0.08/0.1/0.12/1/10 ms）各 16 帧，落 `data/ccd_configs/ccd/FJB24112232/darks/`；全档 12-bit/4095、peak≤39 counts（激光关成立）、偏置≈15 counts、读出噪声 σ≈0.9、暗电流斜率 0.079 counts/ms 可忽略。0.08 ms 档 SHA256(前16)=`608825B8A5AE957D`。
- **floor 冻结**：0.08 ms 暗场均值图空间中位数 `14.9375 counts`（协议 `_robust_background` 口径）已回填预注册 §3；历史 `floor=0` 不合规、已弃用。正式记录与逐文件 SHA256 见 `report/shape_trials/dark_frame_record_20261119.md`。
- **发现一项潜在缺陷（本窗口不修）**：manifest 现混存 3 条旧 8-bit（0.024/0.043/0.047 ms）与 8 条 12-bit 条目，`match()` 不按位深过滤且文件名不含位深 → 同曝光不同位深会互覆/误配。经核 0.08 ms 与最近 8-bit（0.043 ms）差 0.037 ms ≫ `rel_tol=0.02` 容差，**不影响本窗口冻结档**；为遵守预注册 §0「哈希漂移即作废」，本窗口内不改共享驱动代码，列为窗口后任务项。

### 结论 12：目标形状配对采集完成，现场几何验证的统计量判断错了（argmax → 能量质心）

- **设备占用/释放**：Santec SLM #1（1064 nm，memory）+ Daheng CCD #0（`FJB24112232`），1064 nm 激光由用户现场开启、脚本只读光斑不控制激光；单进程无并行；`21:36:06→21:38:09 CST`；`validate --ccd-check` 退出码 0，`21:38:04` SLM `close()` 成功、相机释放、wrapper `finally` 释放锁（`data/device_lock.json` 已不存在）、无残留采集进程。
- **采集落盘**：`data/experiments/hw_window_20261119/validate_pairs`，6 条 manifest 记录（`square_seed` + 2 block 的 A1/B + `square_final`），全 12-bit / 0.08 ms / 无饱和（`saturation_pixels=0`）。config 与预注册 §3 逐项一致：`checkpoint_sha256=44e7ebf1…`、`zero_xy=[1226,1105]`、`target_xy=[1228,988]`、`period=64`、`side=80`、`window_span=220`、`window_step=2`、`zero_radius=0`、`bit_depth=12`、`abba=false`、`acceptance=sequential_ccd`。
- **配对指标（仅记录，未评分）**：`square_seed` cv=0.3271 / ee_roi=0.1391 / target_pearson=0.3894；`square_final` cv=0.1728 / ee_roi=0.1241 / target_pearson=0.3052。**CV 下降但 ee_roi 与 target_pearson 同时下降**，评分时须如实并列报告，不得只取 CV。
- **现场几何验证：我预注册的 argmax 判据不成立（判据统计量选错，不是数据不达标）**。预注册 §3 写的是「目标亮斑 argmax 落在 (1228,988)±10 px」。实测 argmax 距预测：seed 12.0/15.8/14.2 px、final 31.4/39.7/20.0 px，按字面超标。但同一批帧的能量质心（floor=14.9375 扣除后）：
  - 目标盘（r≤45 @ 预测中心）质心 seed `(1229.6,983.5)/(1229.7,983.5)/(1229.5,983.9)` → 距预测 **4.4–4.8 px**；final `(1228.2,985.7)/(1227.7,986.0)/(1228.1,986.4)` → **1.6–2.3 px**。
  - 在 model `core` mask（1332 px）上取质心同结论：seed d≈4.4–4.8 px、final d≈1.5–2.0 px。
  - **稳健性**：质心随窗口半径 r=20/30/40/60/80 完全稳定（都在预测 ±5 px 内）；只有 r=120 才被零级拖到 y≈1018，说明目标与零级分离充分。
  - **argmax 不可用作中心估计**：final 三帧 argmax x 在 1197→1198→1208 之间跳、y 在 983→1013→989 之间跳，而同一三帧质心稳定在 (1228,986)。散斑场内单像素峰位本就逐帧抖动。
  - **零级与目标共存且目标不是最亮点**：全局 argmax 始终是零级（约 (1225,1105)，峰值 1053–1319 counts），目标区峰值仅 109–182 counts；零级盘（r≤30）占扣除 floor 后总能量 11.4–13.4%，目标盘（r≤40）占 10.8–13.0%，`core_energy/total` seed 0.403–0.411 → final 0.338–0.350。
  - **结论**：目标能量确实打在 ramp 预测位置（≤5 px），模型→硬件几何对齐成立；argmax 判据是统计量选择错误。已在预注册 §3 登记为「看到数据后修正判据统计量」，并注明这是事前未预见的修正、不是判据放松。
- **不得据此宣称的事**：本批帧只通过了「目标打到位」的现场安全阀，**没有**通过 §2 评分主门（配对改善 `rel_l2`/`energy_fraction` 尚未算），也没有改变五个 OPEN issue（#234/#235/#236/#238/#247）的状态。

### 结论 13：强度门评估器 P0 实现 + P1 归档候选复核（离线，未占用设备）

- **P0（完成）**：新增 `src/ao_shaping/tools/slm/transfer/strength_gate.py`——`masked_absolute_rel_l2`（掩膜内**未缩放**、可扣 DC 的 `relL2`，空掩膜走 `max(·,1e-9)` 分支）+ `window_zero_mask`（窗口网格零级盘掩膜，对 `window_step` 不变）+ `STRENGTH_GATE_DELTA=0.20` + `gate_rel_l2`（把暗场/bias 的 counts/ms 口径收成唯一表达式）；离线 runner `scripts/slm/evaluate_strength_gate_20261119.py`（**不打开任何设备**，按 split 取最差 pass，写 checkpoint/dataset/script SHA256 与预注册 §3 六项 validity）；15 个单测。`pytest …test_strength_gate.py -q --import-mode=importlib` → 15 passed；`ruff check` 三文件 → 全绿。SHA256（终版）：`strength_gate.py`=`71626026b023241766617094469360741bb349b165c2484eb4517d0f16b57fe8`、`runner`=`e39fb4d26cbc3cd24510659f8e1925c8f127a6315b5e83537e80aea3884d2254`。
- **暗场量纲 bug 修复（本轮自查发现）**：`dark_rate` 由 `frames/exposure` 算出、已是 counts/ms，旧代码 `measured - dark_rate * exposure` 会少减 `1/exposure` 倍（0.051 ms 处约 20 倍），且被零暗帧冒烟掩盖。修法：抽出 `gate_rel_l2(pred, meas, mask, bias_rate=, dark_rate=)`，runner 与单测共用；补 2 个量纲 pin 测试。另把两条静默容错改成显式门槛：暗帧只有原始 counts 且无 `exposure_ms` → `DarkUnavailable`（不再当 counts/ms）；暗帧记录曝光 ≠ probe 曝光 → `not_assessable`。冒烟证据：`dark-exp-0.024ms.npz` → `carries no exposure_ms`；合成 0.024 ms 暗 vs 0.051 ms probe → 曝光不匹配判 `not_assessable`；合成同曝光零暗（接线验证，非证据）→ 12/12 行算出 gate、worst `relL2=1.0362`、verdict `fail`。
- **P1（完成，结论 `not assessable`）**：在 `replicate_a` / `extend_b` 的 refit checkpoint 上（`--span 160`）复现归档 test 全窗 `relL2` 0.723238 / 0.505481（差 ~5.7e-9 / ~1.6e-8），但**无匹配暗帧** ⇒ 两候选 verdict 均为 `not_assessable`（exit 2）。证据：`data/debug/slm_transfer/strength_gate_eval_20261119/{replicate_a,extend_b}.json`。`axis_mismatch_validation_20261118/fit`（grid 320）未记录 `span`，未跑数值，仅记暗场不匹配。
- **两处口径发现（未改冻结阈值）**：① 模型原点须取该 dataset 的 `reference_start` **实测点**才复现归档分数；用 checkpoint 内 fitted `origin` 得 `0.664707`。runner 默认 `--origin measured` 并在证据中记录。② checkpoint 存 `gain=47531.96 / background=0 / bias=0.679406`，与产生归档分数的 train 重拟合 radiometry（`37969.895 / 0.7326 / 0.7307`）**不一致**；直接吃 checkpoint gain 得 `0.827`。故日后冻结该模型须以 train 重拟合 radiometry 或与 `refit_provenance` 一致导出为准。
- **暗场事实更正（重要）**：归档 probe 为 **Mono8 8-bit**（`_optical_probe.json: bit_depth=8`, `camera_format=Mono8`，`light_check` peak 207≤255 佐证）；暗场库 `data/ccd_configs/ccd/FJB24112232/darks/` 有 8-bit（0.024/0.043/0.047 ms）与 12-bit（0.02–0.12 ms + 长曝光），但**无暗帧落在 probe hold-out 曝光档（`replicate_a`=0.051 / `extend_b`=0.028 ms）**，且预注册无冻结插值规则 ⇒ `not assessable` 成立。此前"归档完全无暗场"的说法不准确，已更正为"暗场存在但曝光/位深不匹配、不可事后补"。
- **诊断（不参与判定）**：`replicate_a` test 逐行 `rel_l2_median_fallback` 在 0.46–0.93，即便完美扣 DC 也远高于 δ=0.20；模型前向误差本身偏大，暗场不是唯一缺口。
- **设备纪律**：本轮纯离线只读 + 离线运行 runner，未占用 SLM/CCD/激光；设备锁 `holder=None`；最近真实设备操作为 #238 run4（释放证据 12:18:40Z）。
- **下一步**：P2/P3（合规暗场 + 带独立 holdout 的新采集 + 可能多轮模型迭代）待用户拍板；transfer `light/none` 继续暂停。计划细节见 `transfer_upstream_qualification_plan_20261119.md`。

### 结论 14：Beam Quality GUI 倾斜目标越界的早期诊断与保护（纯离线）

- **根因**：GUI 自动派生采样窗只按目标边长留白，未计入倾斜载波位移；启动前只能用名义 `TILT_SHIFT_SCALE` 估算，无法覆盖拟合 `ramp_matrix` 与名义值不一致的情况。实测几何检查虽计算了目标窗口，却只对 `side_offset_overflow` 警告，未对目标 mask 实际触及采样窗边界作与最终 `validate` 相同的硬判定，导致用户在后续整形阶段才遇到 `target intersects camera window boundary`。
- **修复**：`transfer_desktop/params.py` 的自动派生半跨现在按 `carrier_offset_px(period, TILT_SHIFT_SCALE)` 计入名义位移；上限取 `DEFAULT_SYSTEM.camera_shape` 短轴一半（当前大恒 MER2-507 为 972 px）。`transfer/preflight.py::check_geometry` 对目标 mask 触及任一采样窗边界新增 error，与 `validate` 硬保护同判据；越界时在实际整形前用现有提示要求调大周期或减小边长。窗内但侧向余量不足的 `side_offset_overflow` 仍保留 warning。
- **离线验证**：GUI 参数派生、实测几何预检及 workflow 整形前拦截均有回归测试；定向命令 `python -m pytest tests/ao_shaping/tools/slm/test_transfer_preflight.py tests/ao_shaping/gui/slm/test_transfer_desktop_params.py tests/ao_shaping/tools/slm/test_transfer_workflow.py tests/ao_shaping/tools/slm/transfer/test_capture_grid_auto.py -q` → 151 passed；`ruff check` 涉及 5 个源码/测试文件通过。未占用硬件；未提交/推送。
