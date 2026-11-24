# Agent 当日记录 2026-11-23（未提交工作树改动；本轮占用设备实测一次）

设备状态：用户确认激光在线。SLM #1 USB 已恢复（FTDI FT601 Bridge 枚举 OK），大恒
MER2-507-23GM 可连。本轮占用设备完成一次实验系统衍射效率探针验证，验证曝光预检与质量门禁
在实验系统上的实际行为，跑完正常关闭、释放台面锁。

## 1. 实测：曝光门禁实验系统验证

**占用**：SLM #1（1064 nm，驱动板 `22030108`）+ 大恒 CCD cam 0；激光由用户开启。
**命令**：

```bash
$env:PYTHONPATH = "D:\Projects\TIFO\AO-shaping\libs"; python -m ao_shaping.tools.slm.calibration.diffraction_efficiency probe \
  --slm-number 1 --slm-wavelength 1064 --cam-type daheng --exposure-ms 0.1 \
  --period-px 16 --axis x --separation-px 471 --lobe-axis y \
  -o data/diffraction_efficiency/probe_real_20261123_gate
```

> 注意：`gxipy` 的 import 需要把**包父目录** `libs` 放进 `PYTHONPATH`（`libs\gxipy` 会导致
> `ModuleNotFoundError`，因包内用 `from gxipy.gxiapi import *` 绝对导入）。

**结果**：`quality.qualified=true`，CLI 正常退出（0），产物落盘
`data/diffraction_efficiency/probe_real_20261123_gate/{summary.json,summary.csv}`。

- **曝光预检**：请求 0.1 ms → 阶梯收敛到 **0.02 ms**（`reads=9`，`status=ok`），平场峰值
  104 counts / 255 = **40.8%**，落在接受带 25%–80% 内（目标 45%）。两帧（平场/闪耀）共用
  此曝光；暗帧 `dark_exposure_ms=0.02`、16 帧。
- **饱和统计**：平场与闪耀两帧 `clipped_px=0`，**无像素顶到满量程**——与 20261121 的
  255/255 过曝帧（作废）形成对照，门禁在实验系统上按预期拦截了过曝场景（该次 0.1 ms 请求被
  预检压到 0.02 ms 后即无截断）。
- **级次结果（相对指标，未定标，非绝对衍射效率）**：
  - 0 级窗口能量占比 0.6022；**−1 级 0.3978**（闪耀级次）；+1 级 0.0（弱侧泄漏 floor，
    非共轭级次，符合预期）。
  - `relative_efficiency.blaze_energy_share = 0.1927`（相对平场零级窗口能量）。
  - 0 级位置 (1106, 1229)，光斑 FWHM 13.0 px，ROI 半径 19.5 px（按 FWHM 定标）。
- **几何自检**：给定间距 471 px，实测 −1 级偏移 −471.39 px，`separation_check.consistent=true`。

**释放**：SLM #1 成功关闭，台面互斥锁已释放，CCD 随进程退出释放；无异常、无二次占用。

## 2. 结论

- 曝光预检 + 质量门禁在实验系统上端到端验证通过：过曝场景不再流入正式计算，未过门禁时 CLI
  会非零退出，产物仍落盘可复核。
- 复刻 20261121 参数（0.1 ms 请求、gray 993、period 16、axis x）并显式给标定参数
  （separation 471 / lobe-axis y）后，测得结果稳定、几何自检一致，可作为该探针的
  「带门禁」基准运行。

## 3. 待办 / 未提交状态

- 本次及 20261121–1122 的探针重构改动均**未提交**，工作树存在其他会话暂存的无关文件；
  未经明确授权不提交、不推送。

## 4. 效率匹配窗口：主采集完成（20261123_050735Z）

**占用**：SLM #1（1064 nm，`22030108`）+ 大恒 MER2-507-23GM cam 0 + 1064 nm 激光；
采集结束 `release.json` 记 `slm_flat_gray0=ok / slm_closed=true / camera_closed=true /
lock_released=true / closed_normally=true`，`released.json` 双写一致，无驱动板假死。

**两次六步探测（采集前门）**：

| 曝光 ms | 降功率前峰值 | 降功率后峰值 |
|---:|---:|---:|
| 0.02 | 2009 | 1409 |
| 0.03 | 3608 | 2404 |
| 0.04 | 3761 | 2462 |
| 0.05 | **4095 饱和** | 3688 |
| 0.06 | — | 3278 |

- 第一次（`efficiency_match_20261123`，05:00Z）verdict = **low_snr**：0.05 ms 平场 0 级已顶满
  量程（饱和线 3891），无可用窗口——冻结曝光已不可用。
- 用户降激光功率后第二次（`..._ladder2`，05:05Z）verdict = **ok**，可用窗口
  0.02–0.06 ms，噪声底 20.9 counts。
- **采集前修订**：曝光 0.05 → **0.04 ms**（峰值 2462，满量程 60%，距饱和 37%；0.03 仅高 2.4%
  但无同档暗帧）；暗帧按用户「复用之前采集的暗帧」决定，用库内
  `dark-exp-0.04ms.npz`（12-bit / 16 帧 / mean 14.97），`rel_tol=0.02` 精确命中。已回填
  `hw_window_20261123_efficiency_match_prereg.md` §1/§3.4/§4.3 与计划回填节第 5、6 条，
  修订时点在**任何采集帧下发之前**。

**主采集**：`python scripts/slm/efficiency_match_capture_20261123.py --execute
--out data/debug/efficiency_match/20261123_050735Z --exposure-ms 0.04`，`EXIT=0`，
13:07:35–13:09:00 本地。产物 16 帧全采（`raw/` 14 张图像 + 1 张共享库暗帧），
`config.exposure.readback_ms=0.04`、`all_frames_same_exposure=true`、
`dark_source=camera_library`、每帧 `saturated_pixels=0`；各相位峰值：flat
2017/2217/2151/2231/2183/2068，blaze 1039/1178/1138/1054，shaping 1237/1203/1234/1142
（均 < 饱和线 3891，最高 2231 = 满量程 54%）。

**踩坑（已修）**：分析器原 `_resolve_raw_path` 依赖 journal 的 `raw_file` 字段，而采集端 journal
不写该字段、只给 `frame_index`，raw 命名是 `{frame_index:04d}_{id}.npy`；其兜底 glob 的序号
位置还写反了（`*_[0-9][0-9][0-9][0-9]_{id}.npy`），导致真实数据上 14 帧全被跳过。
修复方向：优先用 `frame_index` 拼名，兜底 glob 改为 `[0-9][0-9][0-9][0-9]_{id}.npy`，
`dark_r2` 回落 base_id 解析到共享 `dark_library_*.npy`。教训：离线合成数据用 `raw_file` 字段
掩盖了真实 journal 的字段差异，端到端产物验证不可省。

**离线分析（已出结论）**：`analysis.json` + `EM_REPORT.md` 已落盘。verdict =
**undeterminable**，reason = `out_of_range_energy_check_background_or_saturation`；触发门
`energy_out_of_range`、`group_sd`、`drift_rounds`（其余 false）。ηs 逐帧为负（−0.103…−0.183，
即 S < S0），ηd ≈ 0.530–0.553，`|mean d|=0.667`，G1 组内 sd 0.0106，ηs sd 0.0552，
F0 跨帧相对变化 4.5–9.4%、质心位移最大 3.12 px，drift_rounds=4 > 1。按预注册口径：
不可判定门优先级高于 matched/mismatched，越界部分**不作物理解释**，不外推、不改判据；
不据此写标定、不判定任何模型解禁。修复仅有分析器路径解析与暗帧去重，未放宽任何阈值，
43 个用例通过、self-test 四场景 PASS、ruff clean（fixer 回报，已核对 analysis.json 门快照）。

## 5. #247 波形判别窗口：采集 + 离线分析完成（20261123_051615Z / hw_window_20261123_issue247_waveform_probe）

**扫描结论**：`docs/dev/` 下未完成采集的实验计划仅此项——预注册写于今日 09:16，60 帧从未
下发；效率匹配、动态解耦 T4/T5、#238 run1/run4、#235、P2a/P2b 均已有落盘数据。

**设备占用与释放**：13:16:16–13:17:xx 本地六步探测（session `issue247_waveform_20261123`，
证据 `data/debug/optical_path_probe/issue247_waveform_20261123/20261123_051615Z/`，
`verdict=ok`，可用曝光 0.03–0.06 ms，噪声底 20.9 counts）；13:18:31–13:19:18 主采集。
设备为 SLM #1（1064 nm）+ 大恒 MER2-507-23GM cam 0 + 1064 nm 激光，一次 open/close；
`released.json` 记 `slm_flat_gray0=ok / slm_closed=true / camera_closed=true /
lock_released=true / closed_normally=true / status=ok / error=null`，无残留 bench 进程。

**采集前修订**：预注册 §1 冻结 0.05 ms；当前功率下距 §4 饱和门仅 0–17% 余量（20 分钟前
同档 3688 counts 已越门），故改 **0.04 ms**（本轮实测 2199，外推最坏 2954 < 3685.5），
修订写进预注册 §8、在下发任何图样之前完成，判据/结论枚举/帧数一律未动。

**采集**：`python scripts/slm/issue247_geometry_capture_20261120.py --plan waveform --execute
--out data/debug/hw_window_20261123_issue247_waveform_probe --exposure-ms 0.04 --settle-s 0.8
--saturation-fraction 0.9`，`EXIT=0`，15 图样 × 4 帧 = 60 帧全采、`saturation_count=0`、
`status=ok`、`error=null`。

**离线**：`waveform_probe_analysis.json` + `WAVEFORM_PROBE_REPORT.md` 已出，verdict =
`patch_position_specific`；flat 漂移 1.75 px < 2 px 门、无饱和；`axis_gate`、`scale_gate`、
`patch_gate` 未通过。原始帧独立核验（argmax + 离轴亮斑质心，非分析器路径）：x 族 ±1 级距
0 级约 ±94 px 沿 CCD 列；y 族 ±1 级距 0 级约 ±64 px **也沿 CCD 列**（角度差 ≈0.26°），
`blaze_patch_x` 则沿行偏移 ≈86 px——与分析器测量一致，说明不是分析器坐标读错，而是本窗口
实测的面板轴响应异常。`patch_position_specific` 是预注册优先级标签（patch 门先判定），
但 axis/scale 门同时未过，不能解读为「只有 patch 异常」。分析器 `--self-test` 3 场景 PASS、
exit 0。结论仅作记录：不写任何标定、不改映射/相位 LUT、不改 #234/#236、不动 transfer
`light/none`、不发布 #238。

---

## 零设备审计与 P3/P4 重路由（2026-11-23 下午）

**#247 映射异常的零设备定位（原始帧重定位）**：对 11-23（60 帧）/ 11-21（44 帧）/ 11-21b（60 帧）
帧重定位（扣 0 级 PSF、ROI 由预测位置导出、150 px 全峰表）：

- 全幅 1D 图样（**逐行完全相同 fy≡0**：ramp_x / grating_x / tilt_x）落在 **CCD 列轴**，其幅值
  = K·N（K=6.344 px/条纹、K×1200=7613、34 瓣 rms 1.06 px）与 2f 解析（6.2975）和 A 映射（6.4177）
  一致 ⇒ **错的只是轴向，不是标度**；全幅 1D y 族（fy≠0）落列轴正确；全幅 2D `blaze2d` 与局部
  patch/ladder 映射正常（`blaze2d` 独立复核残差 ≤1.8 px）。
- **11-23 的 `scale_gate.y` fail（0.664）是假象**：`grating_y_p256` 真 ±1 在 **29.4–30.4 px**，
  恰落在分析器 `core_exclude=34.96 px` 盲区内；11-23 记的 88.89 px 是 **±3 级**（3×29.42≈88.3；
  ±2 因 50% 占空比严格为零；±3/±1 强度比 0.094–0.102 ≈ 理论 1/9）。
- ⇒ **A 映射（2D / 局部内容类）可冻结**；全幅 1D fy≡0 类**不得**用 A 映射校验。根因只能落在
  SDK/固件 `SLM_Disp_Data`（`libs/SLM_DLL_ver.2.51`，仓库外）或面板 10-bit 内部存储路径
  （仓库软件链已逐段验证 1:1 无转置无重采样）⇒ **零设备不可定级**，需设备侧内容相关判别实验。

**门审计（零设备，新脚本，ruff 过）**：

- `scripts/slm/audit_direction_gate_resolvability_20261123.py` → 方向门可分辨上限
  （pass1↔pass2 实测差分 Pearson）Z5 .927 / Z6 .857 / **Z7 −.315** / Z8 .976 / Z9 .784 /
  Z10 .893 / Z11 .853 ⇒ **6/7 模可分辨**；Z7 两 pass 反号、不可分辨。
- `scripts/slm/audit_strength_gate_floor_20261123.py` → 8-bit 预言机地板逐 holdout 0.191–0.315、
  **中位 0.244 > δ=0.20** ⇒ 11-20 强度门（worst 0.8849）**降级 not_assessable**（δ 不放宽）；
  且 11-20 数据为 Mono8（违反 Mono12 纪律）。

**P3/P4 重路由（council 裁决 2026-11-23）**：「可能被整段删去」的前提已撤回——D1 的前提被
`next_experiment_plan_and_capture_20261120.md` §5.3 撤回，且 11-20 执行的是上游资格轨
P3a/P3b（≠ S5 的 176 帧结构化探针）。P3 = **P3-1 零设备离线有限差分 Jacobian**（立即做）
+ P3-2 176 帧**暂缓（非删除**，重启前置 = G1 映射闭合 + P3-1 复审）；P4 门控 = G1+G3+G4，
强度判决另需 G2（测量地板 ≤0.10）。已写入 `docs/dev/hw_plan_consolidated_20261118.md`
（§2 P3/P4 行、§3 G1–G4、§4 S5/S6 暂缓、§7.4）。

**下轮实验窗口预注册（草案，待批准）**：`docs/dev/hw_window_20261123_attribution_prereg.md`
（S0 光路探测 → S1a A 映射闭合 / **S1b 内容相关判别（决定性，4 帧）** → S2 辐射+地板 →
S3 方向门重判（3 pass × 14 对含 Z7）→ S4 patch 位置相位），89 帧、≤95 min、一次 open/close、Mono12。

**P3-1 有限差分审计（`scripts/slm/audit_p3_finite_difference_20261123.py`，零设备、ruff 过）**：

- 78 行大样本（Z4–Z11 ×{0.10,0.20} rad）方向门 pooled **2/16**（0.10 档 2/8、0.20 档 0/8；
  仅 Z5–Z11 为 1/14），比 14 对小样本 4/14 **更差**；16 对信号全部充足（rms/anchor 0.20–0.75）。
- **口径敏感性**：现协议 `phase_sign=−1` → 2/16；`phase_sign=+1` → 0/16；残差取反 → 0/16；
  生产拟合几何 → 0/16。而 `p3_fit_20261120/frozen_parameters.json` **无 `phase_sign` 字段**（默认 +1）
  ⇒ 方向门判决对极性/几何约定**不可复现**，S3 必须先冻结三项口径并报敏感性。
- Jacobian（Δ=5%|c|，7 列 Z5–Z11）：可辨识率 **0.143**（1/7，仅 Z11）、列相关 cond **48.8**；
  **同一 P0 配方在本台复算得 50.4 / 0.143** ⇒ P0 的 `0.0 / 17425` 主要是几何口径产物，不可相减。
  实测侧 6/7 列两两耦合 |r| ≥ 0.76–0.92、对角/非对角比 0.88/0.80 < 3 ⇒ 矩阵门不过，
  ⇒ 「三档 zernike_block」在该相位族上无法用结构化探针判可辨识性。

**#235 P1 附属判据复核（`background_median` 差 40 倍，零设备）**：

- 计划文本（`hw_direction_test_plan_20261117.md:68-69`）写「`background_median` 与历史同量级
  （~15 counts，8-bit / 差分路径）」，实测 `data/debug/issue235_center_20261117/summary.json` 为
  **0.4000**。根因是**指标定义不同**：脚本 `issue235_beam_centre_differential_20261117.py:259`
  取 `bg = float(np.median(flat))`，即**整帧中位数**；光斑 FWHM 几何 24.2 px，视场内 99% 像素未照明，
  故该值就是真实暗电平。
- 全 `data/debug` 语料同类记录区间 **0.2–0.4 counts**（最小 `issue235_beam_center/diff_20261117_170647_v2`
  0.2、`opt_path_diag/run_20261110_230000` 0.375、本段 0.4），**无一条 ~15 counts 记录**
  ⇒ 计划里的 15 counts 无历史依据，属文本口径错误而非器件异常。
- 光斑邻域确有底座，但整帧中位数看不见：`pedestal.retained_fraction` = 0.477 / 0.447 / 0.405
  （窗 6 / 12 / 24 px，全遮挡 ÷ 全开）⇒ 若 15 counts 指光斑邻域底座则量级相容，但**不可用整帧中位数验证**。
- 结论：**口径差异，非异常，#235 的 P1 附属判据不构成阻塞**。有效部分：`flat_peak_counts` 163.4
  < 0.98×255=249.9、`saturated=false`（曝光 0.05 ms）通过；底座以 `pedestal` 记录在案。

**判读口径提醒**：`scripts/slm/direction_holdout_gate_20261119.py` 对 `insufficient_signal`
（NaN 相关性）记 `passed=False` 并计入分母，与计划文档「不得当失败计入分母」冲突；本轮 11-20
数据 7 模信号均充足故未影响该次判决，新判读须用修正口径。

## 轴路由离线重判（2026-11-23，零设备，lane ses_df88f323fffeG28ssQnGKK4TfI）

**背景**：11-23 已把 #247 映射异常定位为「全幅 1D（逐行相同 fy≡0）图样被路由到 CCD 列轴」。
本次在 11-20 P3 门控数据（14 对 + 78 行）上离线检验该假设能否解释 4/14 / 2/16。

**裁决：假设在 11-20 门控数据上不被支持**（零设备，未开任何设备）：

- 全部轴变体 0/14：V1 实测转置 / V2 模型转置 / V3 180° / V4 转置+取负 / V5 转置+平移
  （V5 1/14，系自由平移搜索伪影，跨模式无关对 null 对照 51/84 ≥0.50 已否定）。
- **fy≡0 前提类为空**：被探测的 Z5–Z11 全部 row/col 梯度能量比 0.600–1.667，
  无一为行恒定图案（Noll 组仅 Z2/Z3 纯 tilt 属该类，未被探测）。
- **12/12 ramp 标定行路由正确**：ramp_x→row、ramp_y→col，幅度比 0.993–1.007；
  角 ~179.5° 是 phase_sign=−1 下的符号约定（+1 时 0.3–0.9° 平行），非轴交换。
- 强模式（Z5/Z6/Z8/Z10）位移方向 meas/model 一致（0.5–10°），无 ~90° 特征。
- 4/14 的更可能成因（按证据强度）：① ~8 px 原点/配准偏移（拟合原点 1235.08/1104.64 vs
  门控椭圆中心 1227/1104；改用拟合原点 4/14→7/14，仍 <10/14）；② 差分形状/幅度失配
  （meas 幅度 3.9–8.6×、PSF 更宽）；③ Z7 低信噪比（两 pass 复现性 −0.315）；
  ④ checkpoint 时序（p3_fit 用晚于两 pass 的 zernike 数据重拟合）+ phase_sign 张力。
- **路径相关性**：x 族（grating_x/tilt_x）落 CCD 列轴（±95.6 px）只在 11-21/11-23
  issue247 采集路径成立；11-20 transfer 路径 12/12 ramp 轴向正常 ⇒ 路由行为与采集链相关。
  双链图样轴约定核对（T5–T8）进行中：T5 零设备生成两链名义 x/y 图样比对 panel 梯度轴、
  T6 transfer 链显示路径转置核对、T7 y 族约定表、T8 裁决（显示路径差异 / 命名互换 / 分析伪影）。

**对预注册的影响**：S1b「内容相关判别」的图样集与判据措辞**待 T5–T8 定稿**；S1a/判读口径不变。
中间产物在 C:/Users/zhangh/AppData/Local/Temp/opencode/issue247_axis_*.{json,txt,md}。
主 agent 已抽验 lane 数字：归档 direction_gate.json 逐对 Pearson（Z5 .570/Z6 .066/Z7 −.183/
Z8 .797，pooled 4/14）与 lane V0 复现一致；frozen_parameters.json 确认无 phase_sign 键、
拟合原点 (1235.08, 1104.64) 与门控椭圆中心差 ~8 px。

**补记（2026-11-23 夜，上游 5d7c8e05 回写后文档对账）**：上文「T5–T8 进行中 / 图样集待其
定稿」为 19:14 轴路由重判时的实时状态；同晚 22:27 `5d7c8e05` 已把 T5–T8 定稿（T8 改判
「内容类相关」）并定稿 S1b 图样集（2×2 因子设计）——预注册 `hw_window_20261123_attribution_prereg.md`
§0.2 已随该 commit 更新为「已定稿」，但 §7.1 S1b 标题行与 §9 条目仍残留「进行中」措辞，
与 §0.2 自相矛盾。已按 5d7c8e05 为最新事实，将两处残留同步为「已定稿」（图样集 2×2 见
预注册 §2.1），S1a/判读口径不变。
