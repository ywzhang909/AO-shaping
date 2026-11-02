# 超视场开环预测、D 最优主动探测与物理约束残差拟合（仿真，2026-11-01）

<!-- provenance:start -->
> **生成脚本**: [`scripts/ml/sim2f_fov_extrapolation.py`](../../../scripts/ml/sim2f_fov_extrapolation.py)
> **复现命令**: `python scripts/ml/sim2f_fov_extrapolation.py --out report/shape_trials/fov_generalization`
> **数据/关联脚本**: [`scripts/ml/sim2f_probe_active.py`](../../../scripts/ml/sim2f_probe_active.py)
> **数据/关联脚本**: [`scripts/ml/sim2f_pinn_extrapolation.py`](../../../scripts/ml/sim2f_pinn_extrapolation.py)
> **数据/关联脚本**: [`scripts/ml/sim2f_fov_hw.py`](../../../scripts/ml/sim2f_fov_hw.py)
> **数据/关联脚本**: [`scripts/ml/sim2f_fov_hw_session.py`](../../../scripts/ml/sim2f_fov_hw_session.py)
> **数据/关联脚本**: [`src/ml/sim2f/full_frame.py`](../../../src/ml/sim2f/full_frame.py)
> **数据/关联脚本**: [`src/ml/sim2f/window.py`](../../../src/ml/sim2f/window.py)
> **数据/关联脚本**: [`src/ml/sim2f/extrapolate.py`](../../../src/ml/sim2f/extrapolate.py)
> **数据/关联脚本**: [`src/ml/sim2f/probe_active.py`](../../../src/ml/sim2f/probe_active.py)
> **数据/关联脚本**: [`src/ml/sim2f/pinn.py`](../../../src/ml/sim2f/pinn.py)
> **运行环境**: 离线（带像差与粗糙度的仿真器件 + 冻结标称模型；器件与模型不是同一对象）
> **说明**: 小视场相机下的超视场开环预测、D 最优主动探测、物理约束残差拟合三路仿真：窗内拟合 + 独立确认相位验收门，未确认一律回退零像差；逐 trial 明细与主动探测/物理残差原始产物在 artifacts/。
<!-- provenance:end -->

> 本报告的数字全部来自离线仿真，**不含任何实验测量**（相机不可达，实验用仿真器件 + 冻结标称模型）。器件与模型是**两个独立对象**：器件 = 带像差与粗糙度的 `Sim2f`，模型 = 同几何但无误差的 `Sim2f`。所有结论只对这一对器件-模型有效；迁移到真实设备前必须重跑 `scripts/ml/sim2f_fov_hw.py` 的标定分支。

---

## 0. 口径、非可比声明与文件索引

### 0.1 三个通道回答三个不同问题

| 通道 | 问题 | 主指标 | 原始产物 |
|---|---|---|---|
| **Lane A** 超视场开环预测 | 只看相机窗内，能否一次开环预测出窗外形状 | 窗外相对 L1 `l1_out`、NCC、能量包络 `d86_err_px` | `artifacts/trials.csv`, `artifacts/summary.json` |
| **Lane B** D 最优主动探测 | 固定探针预算下，置信度选相位是否比固定 Noll 银行恢复像差更快 | 系数 RMSE vs 真值 | `artifacts/probe/runs.json`, `artifacts/probe/summary.json` |
| **Lane C** 物理约束残差拟合 | 只在窗内拟合的残差，能否泛化到全幅 | `l1_out` 分形状、`ncc_out`、`ee_out_err` | `artifacts/pinn/runs.json`, `artifacts/pinn/summary.json` |

三者**不能互相替代**：Lane A 测的是「拟合像差后开环预测」，Lane B 测的是「如何用更少探针把像差拟合出来」，Lane C 测的是「给拟合加物理先验是否改善泛化」。把 Lane A 的 `validated` 当作 Lane C 的 `base` 来比，或者把 Lane C 的 `data_only` 当作 Lane A 的 `windowed_fit` 来比，都是错误的。

### 0.2 历史报告不可直接比较

历史形状试验（`report/shape_trials/mask_closed_loop/`、`blackbox_path/`、`axial_move_sim/`）的数字**不可**与本报告直接比较，理由：

1. **真值不同**：历史试验的真值是标定孪生/现场测量；本报告的真值是仿真器件的全幅强度，且器件像差是 Noll 4/6/9/14 四项 + 粗糙度，不是现场数据。
2. **指标不同**：历史试验用 `EE`、`CV`、`accept` 轮次；本报告用 `l1_out`、`ncc_out`、`d86_err_px`、系数 RMSE。
3. **窗口不同**：历史试验无显式小窗概念；本报告故意用 96×96 / 64×64 窗口模拟超视场。
4. **验收门不同**：历史试验的验收门是现场 `valid` 位深门限；本报告用独立确认相位的 ratio 门。

任何"本报告比历史试验好/差 X%"的表述都是无意义的。历史数字只在各自报告里引用。

### 0.3 文件索引

| 路径 | 用途 |
|---|---|
| `artifacts/trials.csv` | Lane A 逐 trial 明细（96 行） |
| `artifacts/summary.json` | Lane A 聚合 + 确认门 + 诊断 |
| `artifacts/trials.json` | Lane A 逐 trial 完整记录（96 条：窗口、arm、确认 ratio、全部指标、拟合系数） |
| `figures/fig_fov_cases.png` | Lane A 典型帧：真值 / 预测 / 有符号误差 |
| `figures/fig_fov_error.png` | Lane A 每窗每组 `l1_all/l1_out` 与 `ncc_out` |
| `artifacts/probe/` | Lane B 原始产物（`runs.json`、`summary.json`、`trials.csv`） |
| `figures/fig_probe_budget.png` | Lane B 预算-误差曲线与逐 seed 比值 |
| `artifacts/pinn/` | Lane C 原始产物（含 `coeffs.npz`、`loss_curves.png`、`fig_pinn_cases.png`） |

硬件会话脚本：[`scripts/ml/sim2f_fov_hw_session.py`](../../../scripts/ml/sim2f_fov_hw_session.py)；
全画幅接口实现：[`src/ml/sim2f/full_frame.py`](../../../src/ml/sim2f/full_frame.py)。执行方案见 §7。

---

## 1. 共用器件、模型与协议

### 1.1 器件与模型

- **器件** `Sim2f(Sim2fConfig(grid=128, ccd_px=256), errors=OpticsErrors(...))`：SLM 像差 `[(4,0.40),(6,-0.32),(9,0.26),(14,0.18)]` rad + 粗糙度 σ=0.06 rad（seed 7）。
- **模型** `Sim2f(Sim2fConfig(grid=128, ccd_px=256))`：同几何、零误差。两个对象，**永远不是同一个**。
- **Oracle** = 把器件面板相位最小二乘投影到模型的 Noll 2..11 基上（投影残差 RMS 0.0825 rad）。它**不是**完美校正：器件的粗糙度（σ=0.06 rad）与 Noll ≥12 的高阶项（Noll 14 被器件使用）不在模型基内，所以 oracle 是上界，不是真值。

### 1.2 相机窗口

`CameraWindow` / `WindowedBench` / `crop_measure` 只做**后处理**：全幅帧先算，再裁剪。仿真层没有 ROI 概念。窗口：

| 窗口 | 中心 (px) | 尺寸 | 覆盖率 |
|---|---|---|---|
| `96x96@128,108` | (128,108) | 96×96 | 14.06% |
| `64x64@128,108` | (128,108) | 64×64 | 6.25% |
| `96x96@128,128` | (128,128) | 96×96 | 14.06% |

故意选小窗，模拟目标光斑超出 CCD 靶面的场景。窗口中心故意偏离帧中心 (128,108) 而非 (128,128)，模拟真实相机装调偏差。

### 1.3 Lane A 协议（冻结，不再改）

- **一次辨识，零指令相位**：探针银行（Noll 4..11）在零指令相位下推，拟合器件的静态像差。**每窗只辨识一次**；拒绝"每阶段重辨识"，因为拟合会吸收指令相位。
- **系数冻结**：辨识出的系数对所有测试相位（random / oversized / tiny / defocus，各 2 个）保持不变；每个测试帧都是**单次开环预测**，没有闭环。
- **验收门**：独立确认相位（从自有 RNG 流抽取、与任何被评分案例不同）下的窗内相对 L1，拟合后 ≤ 未校正模型的 `CONFIRM_RATIO = 1.0` 倍。未通过 → 回退零像差（`nominal`）。
- **拟合模式**：`probe`（固定 Noll 4..11）+ `FitConfig` 默认；不改 `fit.py`。

---

## 2. Lane A — 超视场开环预测

### 2.1 结果：确认门

| 窗口 | confirm_l1（拟合后） | confirm_l1（未校正） | ratio | 门 (1.0×) | 0.5× | 1.0× | 1.5× |
|---|---|---|---|---|---|---|---|
| `96x96@128,108` | 0.2337 | 0.1335 | **1.751** | 拒绝 | 拒绝 | 拒绝 | 拒绝 |
| `64x64@128,108` | 0.0399 | 0.1536 | **0.260** | 接受 | 拒绝 | 接受 | 接受 |
| `96x96@128,128` | 0.2387 | 0.1355 | **1.761** | 拒绝 | 拒绝 | 拒绝 | 拒绝 |

解读：

- **64×64 窗口辨识成功**，且在 1.0×/1.5× 门下接受；0.5× 门下拒绝（确认帧改善不够激进）。
- **两个 96×96 窗口均被拒绝**，且 ratio ≈ 1.75–1.76 > 1，拟合后确认帧**比不校正还差**。这正是"低残差 ≠ 正确像差分支"的证据：`FitResult.ok=True` 在三窗都成立，但只有 64×64 的分支是对的。
- ratio 门设为 1.0 是**故意的保守选择**：宁可回退到零像差，也不把符号翻转的像差送进开环预测。

### 2.2 结果：预测误差（per 窗 × per 形状组）

`l1_out` = 窗外像素的相对 L1（`sum|pred-truth| / sum(truth)`）；`ncc_out` = 窗外 NCC；`d86_err_px` = 能量包络直径误差。

#### 窗口 `96x96@128,108`（被拒绝 → validated 回退到 nominal）

| 形状组 | nominal l1_out | windowed_fit l1_out | validated l1_out | oracle l1_out | nominal ncc_out | windowed_fit ncc_out |
|---|---|---|---|---|---|---|
| random | 0.160 | 0.271 | 0.160 | 0.028 | 0.986 | 0.964 |
| oversized | 0.037 | 0.063 | 0.037 | 0.019 | 0.999 | 0.996 |
| tiny | 0.841 | 0.851 | 0.841 | 0.616 | 0.992 | 0.917 |
| defocus | 0.025 | 0.033 | 0.025 | 0.019 | 0.997 | 0.995 |

#### 窗口 `64x64@128,108`（被接受 → validated 使用拟合像差）

| 形状组 | nominal l1_out | windowed_fit l1_out | validated l1_out | oracle l1_out |
|---|---|---|---|---|
| random | 0.120 | 0.049 | 0.049 | 0.015 |
| oversized | 0.036 | 0.024 | 0.024 | 0.019 |
| tiny | 0.168 | 0.083 | 0.083 | 0.015 |
| defocus | 0.027 | 0.022 | 0.022 | 0.019 |

#### 窗口 `96x96@128,128`（被拒绝 → validated 回退到 nominal）

| 形状组 | nominal l1_out | windowed_fit l1_out | validated l1_out | oracle l1_out |
|---|---|---|---|---|
| random | 0.160 | 0.235 | 0.160 | 0.028 |
| oversized | 0.037 | 0.058 | 0.037 | 0.019 |
| tiny | 0.841 | 0.851 | 0.841 | 0.616 |
| defocus | 0.025 | 0.032 | 0.025 | 0.019 |

解读：

- **接受窗（64×64）**：拟合像差把窗外误差压到 nominal 的 **1/2 ～ 1/5**（random 0.120→0.049，tiny 0.168→0.083），且 `d86_err_px` 同步改善（random 0.403→0.588 例外，包络略变差）。
- **拒绝窗（96×96）**：未经验收的 `windowed_fit` 在所有形状上都**比 nominal 差**（random 0.160→0.271），而 `validated` 因回退而**从不比 nominal 差**。验收门的价值就在这里：它把"拟合把事情搞砸"的 6 个 case 全部拦下。
- **tiny 形状组**：`l1_out` 数值大（0.17–0.85）但 `ncc_out` ≥ 0.92，说明误差主要在振幅/包络，不在结构。tiny 的窗外能量占比小，`l1_out` 接近噪声比，解读时以 `ncc_out` 为准。（Lane A 的 `trials.csv` 未记录窗外能量分母，该判断依据 `ncc_out` 与 `l1_out` 的不一致推断；Lane C 在同类形状上实测 `out_energy_frac` 为 3.7e-5，见 §4.2。下一轮应把该分母补进 Lane A 产物。）
- **oracle** 在所有窗/形状上都是上界（random 0.015–0.028），未被任何 arm 达到；残差主要来自器件粗糙度与 Noll ≥12 高阶项，这两者都不在模型基内。

### 2.3 失败模式：符号翻转分支（`ident_shot_l1`）

被拒绝的拟合不是"残差大"——`FitResult.ok=True`、窗内残差 0.0083——而是**像差符号反了**。诊断字段：

| 窗口 | fit（窗内残差） | opposite（符号翻转残差） | zero（零像差残差） |
|---|---|---|---|
| `96x96@128,108` | 0.0083 | 0.0400 | 0.0517 |
| `96x96@128,128` | 0.0084 | 0.0401 | 0.0518 |

拟合残差（0.0083）比零像差残差（0.0517）低 6×，但确认帧比零像差差 1.75×——拟合把器件像差解释成了一个**符号相反、幅度相近**的相位，窗内拟合得很好、窗外全错。`oracle` 的投影系数（Noll 4 = +0.226、Noll 6 = −0.181、Noll 9 = +0.109）与器件真值（Noll 4 = +0.40、Noll 6 = −0.32、Noll 9 = +0.26）同号，而被拒绝拟合的系数反号。

教训：**窗内残差不是验收标准**；必须用独立测量（确认相位）做分支选择。这也是本报告把 `validated` 设计成"未通过回退 nominal"的原因。

### 2.4 图

- `figures/fig_fov_cases.png`：每个窗的真值 / 预测 / 有符号误差三联图，标题含拟合残差与确认门结果。
- `figures/fig_fov_error.png`：每窗每组的 `l1_all`、`l1_out`、`ncc_out` 柱状图。

---

## 3. Lane B — D 最优主动探测

### 3.1 协议

- 器件与模型：同 §1.1 的分离原则（每 seed 新建两个独立 `Sim2f`）。5 seeds；每 seed 重抽 Noll 4..11 系数（每模随机方位角，整矢量 RMS 均匀取自 `AberrationConfig.rms_rad = [0.05, 0.45]` rad），粗糙度 σ=0.05 rad、seed 跟随。
- 两策略跑**同一代码路径** `probe_active`：固定银行 `probe_noll(cfg)` vs 置信度选模（D-optimality on `trace(cov)`），主动振幅自适应 ±50%（`[0.5,1.5]×` 配置振幅 0.35 rad）。
- 预算：2 / 4 / 6 / 8 / 10 次探针（每次 2 帧：±）。两个策略各跑到 `max(budgets)` 一次，较短的预算读作前缀。
- 主指标 `coeff_rmse_basis`：把该 seed 画出的器件像差重新投影到**拟合器使用的 pupil（瞳面） 基**（Noll 4..11），再与拟合系数逐模比较。**这是主指标**，因为 `OpticsErrors.slm_zernike` 按整块面板半径归一，而拟合按 pupil（瞳面） 半径归一，直接比原始系数会把这个共同的半径约定误差算进两个变体。
- 辅指标（仅对照）`coeff_rmse`：与画出的 `slm_zernike` 原始系数比较。两个变体共享同一个约 0.4× 的半径约定偏移，所以它**掩盖**真实差异，不能作为胜负判据。
- 其他诊断：`trace(cov)`、条件数、`sigma2`、`mean_confidence`、重复选模次数、选模/振幅序列。

### 3.2 结果：主动选模在同等预算下**更差**

| 预算 | fixed `coeff_rmse_basis` | active `coeff_rmse_basis` | active/fixed | 重复选模 (fixed/active) | mean confidence (fixed/active) |
|---|---|---|---|---|---|
| 2 | 0.0131 ± 0.0036 | 0.0175 ± 0.0046 | **1.417 ± 0.537** | 0.0 / 0.0 | 1.18e-4 / 1.18e-4 |
| 4 | 0.0124 ± 0.0038 | 0.0147 ± 0.0052 | **1.169 ± 0.067** | 0.0 / 1.4 | 9.34e-5 / 7.14e-5 |
| 6 | 0.0123 ± 0.0038 | 0.0147 ± 0.0050 | **1.183 ± 0.093** | 0.0 / 3.0 | 8.63e-5 / 5.96e-5 |
| 8 | 0.0117 ± 0.0039 | 0.0148 ± 0.0050 | **1.259 ± 0.087** | 0.0 / 4.4 | 6.51e-5 / 5.16e-5 |
| 10 | 0.0124 ± 0.0043 | 0.0146 ± 0.0051 | **1.175 ± 0.021** | 2.0 / 6.0 | 5.52e-5 / 4.63e-5 |

（mean ± std，n=5 seeds；值为 rad。`coeff_rmse_basis` 越小越好。）

**逐 seed 无例外**：在 5 个 budget 上，active 的 `coeff_rmse_basis` 均劣于 fixed，`active_better_seeds = 0/5`（预算 2 的比值范围 1.006–2.313，预算 4–10 为 1.073–1.365）。因此这不是"平均被一个坏 seed 拉动"，而是同向的劣化。

**退化机制（可直接从产物读出）**：active 在预算 4 之后开始**重复已探过的模式**，到预算 10 平均有 6.0 次重复（fixed 为 2.0 次，只因为 8 模银行在预算 10 时绕回开头）。典型 seed 0 的 active 序列是 `[8, 4, 4, 11, 4, 4, 4, 11, 7, 4]`——第 2 次之后几乎只在 Noll 4 上反复推。重复一次已测模式的相位不增加新的设计方向；同时 `mean_confidence` 反而比 fixed 更低，说明这些重复没有转化为系数精度。

**与辅指标的冲突必须明说**：若按 `coeff_rmse`（原始系数）看，active/fixed 为 1.014 / 0.992 / 0.990 / 0.987 / 0.992，似乎"active 在 4–10 预算略好 1–2%"。这是**假象**：两个变体共享约 0.4× 的半径约定偏移（例如 seed 0 的 `rms_rad` 0.267 对 `rms_basis_rad` 0.122），该偏移主导了原始系数误差，把两个变体的差异压到 1–2%。主指标 `coeff_rmse_basis` 扣掉了这个共同偏移，结果就是 active 差 17–42%。**本报告以 `coeff_rmse_basis` 判定：主动选模在本实现下失败。**

### 3.3 这是实现问题还是方法问题？

本报告只能证明前者所在的层面：**被测的是 `src/ml/sim2f/probe_active.py` 当前实现的 D-optimality**，不是"D-optimality 方法本身不可行"。产物给出的失败特征是选模退化（重复已测模式），而不是信息量不足或测量噪声。可能原因（未在本轮改动/验证）：

1. `select_probe` 用**对角近似**（只看 `cov[j,j]`）计算 trace 缩减，忽略模间相关；一旦某模后验方差被压小，其余模的边缘方差可能仍更小，导致反复选同一模。
2. `coeff_rmse_basis` 的投影基与拟合器的 `n_modes=10` 不完全一致（拟合输出 10 维，评分只取 Noll 4..11 的 8 维），可能让"未被评分覆盖"的模式吸收能量；fixed 与 active 受此影响不同。
3. 主动振幅自适应只放大不缩小，重复推同一模时每次仍用 0.35 rad，未形成新的独立观测。

本轮**没有**为追平 fixed 而修改模块或调参——这是预先声明的负面结果。要把"主动选模是否有价值"回答清楚，下一轮应先修 1（full-covariance D-optimality 或显式禁止重复），再重跑同一协议。

### 3.4 图

- `figures/fig_probe_budget.png`：(a) `coeff_rmse_basis` vs 预算，fixed vs active，mean ± std；(b) 逐 seed 的 active/fixed 比值 vs 预算，1.0 水平线。
- 原始产物：`artifacts/probe/runs.json`、`summary.json`、`trials.csv`；脚本自带的诊断图在 `artifacts/probe/curves.png` 与 `artifacts/probe/fig_probes.png`（不作为正文结论图）。

---

## 4. Lane C — 物理约束残差拟合

### 4.1 协议

- 小窗 96×96（14.06% 覆盖率）拟合残差相位 `delta`，评分全幅（窗口外的像素拟合器从未见过）。
- 九变体：`base`（仅 stage-1 探针拟合，无残差）、`data_only`（窗内 L1 + TV，`physics=False`）、`pinn`（全部物理项）、`band_only`、`support_only`、`pinn_support_1.0/0.1/0.01`（support 权重敏感性）、`oracle`。
- 物理项（`pinn.py`）：孔径支撑（aperture support，**绝对**惩罚）、Zernike 带限（Noll 4..21，**绝对**惩罚）、曝光线性、TV 平滑（`ResidualPhase` 上）。修正前的比值型惩罚已废弃。
- `pinn_support_1.0` 与 `pinn` 是**故意重复**（数值逐位相同），用来证明敏感性扫的就是 support 项本身。
- 3 seeds × 12 训练相位 × 200 epochs，grid 128 / ccd 256，runtime 92.9 s。

### 4.2 关键口径：`l1_out` 只在 held_out 上可读

`l1_out = sum|pred−truth| / sum(truth)` 定义在窗外像素上。若窗外真实能量本身接近 0，这个比值就是噪声比，不是误差。

| 相位 | `out_energy_frac` | `l1_out` 分母 | 可读？ |
|---|---|---|---|
| held_out | **6.56%** | 59.77 | **是** |
| oversized | 9.18e-4 | 0.839 | 否（比值在 0.8 计数上） |
| tiny | 3.74e-5 | 0.034 | 否 |

`summary.json["arms"]` 里仍保留跨相位池化均值（`pinn` 4.86）以便对照，但**池化均值由不可读的 tiny 相位主导，不得引用**。下表按相位分列。

### 4.3 结果：held_out（唯一 `l1_out` 可读的相位）

| arm | l1_out | ncc_out | l1_in | ee_out_err | centroid_err_px |
|---|---|---|---|---|---|
| base | 0.4489 ± 0.0000 | 0.9501 | 0.2317 | 1.51e-2 | 0.194 |
| **data_only** | **0.0329 ± 0.0011** | 0.9998 | 0.0109 | 1.51e-4 | 0.033 |
| pinn | 0.0622 ± 0.0029 | 0.9992 | 0.0166 | 7.03e-4 | 0.050 |
| band_only | 0.0358 ± 0.0045 | 0.9997 | 0.0121 | 5.54e-4 | 0.031 |
| support_only | 0.0531 ± 0.0030 | 0.9994 | 0.0144 | 2.55e-4 | 0.052 |
| pinn_support_1.0 | 0.0622 ± 0.0029 | 0.9992 | 0.0166 | 7.03e-4 | 0.050 |
| pinn_support_0.1 | 0.0432 ± 0.0031 | 0.9996 | 0.0125 | 4.32e-4 | 0.040 |
| pinn_support_0.01 | 0.0339 ± 0.0014 | 0.9998 | 0.0107 | 8.07e-5 | 0.038 |
| oracle | 0.0206 ± 0.0000 | 0.9999 | 0.0049 | 1.10e-4 | 0.019 |

### 4.4 结果：oversized（`l1_out` 不可读，以 `ncc_out` 为准）

| arm | ncc_out | l1_in | ee_out_err | centroid_err_px | （l1_out 仅供参考） |
|---|---|---|---|---|---|
| base | 0.1552 | 0.4564 | 6.07e-4 | 0.027 | 0.904 |
| **data_only** | **0.9871** | 0.0143 | 1.09e-5 | 0.021 | 0.325 |
| pinn | 0.9349 | 0.0242 | 4.34e-4 | 0.036 | 0.838 |
| band_only | 0.9877 | 0.0154 | 3.40e-5 | 0.031 | 0.322 |
| support_only | 0.9616 | 0.0226 | 3.39e-4 | 0.041 | 0.663 |
| pinn_support_1.0 | 0.9349 | 0.0242 | 4.34e-4 | 0.036 | 0.838 |
| pinn_support_0.1 | 0.9759 | 0.0213 | 1.23e-4 | 0.030 | 0.441 |
| pinn_support_0.01 | 0.9859 | 0.0160 | 3.46e-5 | 0.024 | 0.333 |
| oracle | 0.9973 | 0.0031 | 4.88e-5 | 0.014 | 0.187 |

### 4.5 结论

1. **残差拟合本身是有效的**。held_out 上 `data_only` 把窗外 `l1_out` 从 0.4489 压到 **0.0329（13.6×）**，`ncc_out` 0.9501→0.9998，`ee_out_err` 1.5e-2→1.5e-4。oversized 上 `ncc_out` 0.1552→0.9871。**只在窗内拟合的残差确实能泛化到全幅**——这是本报告最强的正面结论。
2. **本报告的物理项没有提升泛化，反而有害**，且有害程度随 support 权重单调：held_out 相对 `data_only`，support=1.0 时 **+89%**（9.6σ），0.1 时 +31%（3.2σ），0.01 时 +3%（0.54σ，不可区分）。oversized 同序：+158% / +36% / +2.7%（0.24σ）。
3. **band 项近似中性**：held_out 0.0358 vs 0.0329（+9%），oversized `ncc_out` 0.9877 vs 0.9871（略好）。真正有害的是 **support 项**。
4. **`pinn_support_0.01` 与 `data_only` 在 seed 波动内不可区分**（0.54σ / 0.24σ）。也就是说：support 权重降到 0.01 等于关掉它，而 `data + band + smooth` 本身已经做到了物理正则项想做的事。
5. **诚实说明**：在 `l1_out` 上挑权重就是**用评价指标做选择**，所以"最佳权重 0.01"的读数是乐观的。本报告不据此提出推荐权重，只报告单调趋势。
6. `oracle` 在所有相位上都是上界（held_out 0.0206），未被任何 arm 达到；差距来自器件粗糙度与 Noll ≥12 高阶项，二者不在模型基内。

### 4.6 图

`artifacts/pinn/fig_pinn_cases.png`、`artifacts/pinn/loss_curves.png`（均在 `artifacts/pinn/`，为 Lane C 的诊断图，正文不直接引用为结论图）。

---

## 5. 三路联合结论

1. **超视场开环预测可行，但必须配独立确认门**。64×64 窗口（6.25% 覆盖）下，拟合像差把窗外误差压到 nominal 的 1/2–1/5；96×96 窗口下拟合分支符号翻转，未经验收的 `windowed_fit` 比 nominal 差 1.7×，验收门把 6 个坏 case 全部拦下并回退到 nominal。**窗内残差不是验收标准**。
2. **D 最优主动探测在本实现下失败**。同预算下 `coeff_rmse_basis` 的 active/fixed 比为 1.17–1.42，5 个 seed × 5 个预算全部劣于固定 Noll 银行；退化机制是重复选已测模式（预算 10 时平均 6.0 次重复）。原始系数指标看似 active 好 1–2%，但那是两个变体共享的半径约定偏移，不能作为判据。见 §3。下一轮应改用 full-covariance 准则或显式禁止重复，再重跑同一协议。
3. **物理约束残差拟合：残差拟合本身有效，但本报告的 support 项在当前权重下损害泛化**。只在窗内拟合的残差把 held_out 的窗外 `l1_out` 压到 1/13.6、`ncc_out` 0.9501→0.9998——泛化成立。但 support 项权重 1.0 时反而比 `data_only` 差 89%（9.6σ），权重降到 0.01 才与 `data_only` 不可区分。结论是**support 惩罚需要重新参数化（硬约束或按残差能量归一）**，不是放弃 PINN。
4. **oracle 是上界不是真值**：器件粗糙度与 Noll ≥12 高阶项不在模型基内，所有 arm 都到不了 oracle。
5. **指标本身有适用域**：`l1_out` 只在窗外能量占比可观时可读。Lane C 实测 held_out 6.56%（可读）、oversized 9.18e-4、tiny 3.74e-5（均不可读）。凡引用 `l1_out` 必须同时给出该相位的窗外能量占比，否则就是在引用噪声比。

## 6. 历史对比与非可比性

见 §0.2。本报告不与 `mask_closed_loop` / `blackbox_path` / `axial_move_sim` 的数字做任何定量比较。

## 7. 硬件执行方案

接口已落地为 `scripts/ml/sim2f_fov_hw_session.py`（会话编排）、
`src/ml/sim2f/bench.py`（`HwInstrument` 公共入口）与
`src/ml/sim2f/full_frame.py`（`enable_full_frame()` /
`capture_full_frame()` 实现）。方案只使用**一台相机、原生全画幅、一次采集一帧**：
ROI 与窗口裁切全部留到离线阶段，不在采集中途改窗口。

### 7.1 阶段划分

| 阶段 | 是否接触硬件 | 作用 | 主要产物 |
|---|---|---|---|
| `plan` | 否 | 生成确定性的相位/命令清单；同一参数两次运行字节一致 | `plan.json`、`phases/*.npy` |
| `preflight` | 否 | 检查计划、待采项、会话与标定文件是否就绪 | `preflight.json` |
| `capture` | 是 | 每条命令读一帧原生全画幅；按 item 原子写 manifest | `capture_manifest.json`、`frames/*.npy` |
| `finalize` | 否 | 以 0 级为中心裁切到分析窗口，写标准录制 | `session.json`、`process_data/` |
| `calibrate` | 否 | 从已 finalize 的录制拟合 bench 状态 | `calibration/calibration.json`、`.npz` |
| `analyze` | 否 | 全画幅预测与窗口内外指标；窗外不可读时写 `null` | `analysis/summary.json` |

`capture` 是唯一打开 SLM/相机的阶段；其余阶段只读文件。`--dry-run` 用
`SimInstrument` 走同一条 capture/resume/finalize 代码路径，是管线自检，
**不是**对真实 bench 的结论。

### 7.2 推荐执行顺序（两遍法）

第一遍只做标定，第二遍带标定生成拟合形状：

```bash
# 1) 标定会话：低阶屏幕 + Noll 4–11 ±探针 + 固定形状
python scripts/ml/sim2f_fov_hw_session.py plan \
  --out data/debug/sim2f_fov_calib \
  --screens 24 --repeats 2 \
  --targets ring cross frame two_spot two_spot_wide
python scripts/ml/sim2f_fov_hw_session.py preflight --session data/debug/sim2f_fov_calib
python scripts/ml/sim2f_fov_hw_session.py capture --session data/debug/sim2f_fov_calib
python scripts/ml/sim2f_fov_hw_session.py finalize --session data/debug/sim2f_fov_calib
python scripts/ml/sim2f_fov_hw_session.py calibrate \
  --session data/debug/sim2f_fov_calib --out data/debug/sim2f_fov_calib/calibration

# 2) 生产会话：同一计划，但带标定，额外生成 fitted 形状
python scripts/ml/sim2f_fov_hw_session.py plan \
  --out data/debug/sim2f_fov \
  --screens 24 --repeats 2 \
  --targets ring cross frame two_spot two_spot_wide \
  --calibration data/debug/sim2f_fov_calib/calibration
python scripts/ml/sim2f_fov_hw_session.py preflight --session data/debug/sim2f_fov
python scripts/ml/sim2f_fov_hw_session.py capture --session data/debug/sim2f_fov
python scripts/ml/sim2f_fov_hw_session.py finalize --session data/debug/sim2f_fov
python scripts/ml/sim2f_fov_hw_session.py analyze \
  --session data/debug/sim2f_fov --out data/debug/sim2f_fov/analysis \
  --calibration data/debug/sim2f_fov_calib/calibration
```

若相机不可达，先用 `capture --session data/debug/sim2f_fov --dry-run` 验证
plan/resume/finalize 管线；该路径不导入相机或 SLM SDK。

### 7.3 采集与恢复规则

- 物理会话的规范存储根为 `data/debug`：本次标定与生产数据已完整复制到
  `data/debug/sim2f_fov_calib`（609 个文件）和 `data/debug/sim2f_fov`
  （727 个文件），并同步生产计划与分析摘要中的标定路径。旧 `runs/` 副本
  暂保留作迁移前快照，后续恢复与新增采集使用 `data/debug` 路径。
- `plan` 的默认规模为 24 个标定屏幕、16 个 ±Noll 4–11 探针、5 种形状 × 2
  次重复，并在每个探针/形状后插入 flat；实际条目数见 `plan.json`。
- `capture` 对每个待采 item 写 float32 原生帧与原子 manifest，记录
  `phase_sha256` / `frame_sha256` / `panel_sha256`、曝光与时间戳。
- 中断后重跑同一条 `capture` 命令即可恢复：SHA256 有效的 item 跳过，
  缺失或损坏的 item 重采；不会因为半写文件而整场重来。
- `finalize` 以采集时记录的 0 级坐标为中心裁切 `window_px`（默认 256），
  生成仓库标准 `session.json` + `process_data/`；重复运行幂等。
- `calibrate` 拒绝 `session.json` 与当前 manifest 不一致的过期会话，必须
  先重新 `finalize`。其输出的 `ok=false` 表示标定未通过泛化验收，不得把该
  标定用于生产；零像差 dry-run 出现此拒绝属于预期的自检行为。
- `analyze` 的 `l1_out` / `ncc_out` 只在窗外能量可读时给出数值，否则为
  `null`，并在 `not_implemented["outside_window_metrics"]` 说明原因，不用
  噪声比伪造结论。

### 7.4 报告输入契约

硬件会话只作为**原始数据**，不直接改写本报告的仿真数字。需要把硬件结果
写入报告时，必须同时给出：`plan.json`、`capture_manifest.json`、
`session.json`、`calibration/calibration.json` 与 `analysis/summary.json`
的路径，并注明 `window_px`、0 级坐标、曝光和是否使用标定。跨会话、跨窗口
或跨器件的数字不得与本报告 §2–§4 的离线 surrogate 直接比较。

**已验证硬件发现**——全画幅原生帧 `(1944×2592)` 与分析窗口预设
`(1944×1944)` 形状不一致，通过 `scripts/ml/sim2f_fov_hw_session.py` 中的
中心裁剪修复：每帧围绕 `zero_order_xy` 以 `full_px = min(native)` 为边长的
正方形裁剪，使模型预测与实测在 0 级周围对齐。这保留了窗外指标的对称意义，
仅丢弃关于 0 级对称的左右远翼。修复前分析会在整幅 NCC ≈ 0.06 处崩溃
（模型整幅窗 512×4.62 µm 比原生整幅 1944×2.2 µm 窄，外圈像素无模型采样），
修复后所有 120 项完成，fit `ok=True`，Lane B 权重保持不变。

## 8. 文献（已从原页核验）

**Lane A（超视场 / 外推）**

- Cho, E.-S. et al. (2026). "Graph-based modeling of optical system enables adaptive optics on dynamic samples with self-calibration." *iScience* 29, 116769. DOI `10.1016/j.isci.2026.116769`. —— 图模型 + 自标定，支持"器件与模型分离"的设定。
- Latychevskaia, T., Chushkin, Y. & Fink, H.-W. (2016). "Resolution enhancement by extrapolation of coherent diffraction images." *J. Microsc.* 264, 3–13. DOI `10.1111/jmi.12408`. —— 相干衍射图像外推的定量极限，支撑"窗外预测有极限"的结论。
- Véran, J.-P. et al. (1997). "Estimation of the adaptive optics long-exposure point spread function using control loop data." *Proc. SPIE* 3126, 81. DOI `10.1117/12.279066`. —— 控制环数据估计 PSF，支撑"用残差/控制数据做预测"的路线。
- Ho, C.-J. et al. (2024). "A Differentiable Wave Optics Model for End-to-End Computational Imaging System Optimization." arXiv `2412.09774`. —— 可微波光学模型，支撑"单次开环预测"的可微性。
- Landman, R. et al. (2022). "Joint optimization of wavefront sensing and reconstruction with automatic differentiation." arXiv `2209.05904`. —— 传感与重建联合优化，支撑 Lane A 的"拟合 + 预测"一体化。

**Lane B（主动探测）**

- Zhang, B., Booth, M. J. & Hu, Q. (2026). "Information-guided optimization of image-based sensorless adaptive optics methods." *Opt. Express* 34, 33074. DOI `10.1364/OE.609619`. —— 信息引导的无波前传感器 AO，直接支撑 D-optimality 选模。
- Taras, A. K., Haffert, S. Y. & Desdoigts, L. (2026). "Differentiable design of the PIAA-ZWFS." arXiv `2606.28136`. —— 可微波前传感器设计。
- Booth, M. J. (2006). "Wave front sensor-less adaptive optics: a model-based approach using sphere packings." *Opt. Express* 14, 1339. DOI `10.1364/OE.14.001339`. —— 模型化无波前传感器 AO 基线。
- Fan, et al. (2017). "Cramér–Rao lower bound analysis of phase diversity for sparse aperture optical systems." *Appl. Opt.* 56, 2563. DOI `10.1364/AO.56.002563`. —— CRLB，支撑"协方差/置信度"的理论下界。

**Lane C（PINN / 物理约束）**

- Tu, H., Liu, H., Pan, T., Xie, W., Ma, Z., Zhang, F., Xu, P., Wu, L., Xu, O., Xu, Y., Qin, Y. & Xu, Y. (2025). "Deep empirical neural network for optical phase retrieval over a scattering medium." *Nat. Commun.* 16, 1369. DOI `10.1038/s41467-025-56522-5`（PMC11799312）. —— 物理增强网络在无解析模型系统上的失效与经验模型替代，支撑"物理项要按可实现约束写"。
- Hoidn, O., Mishra, A. & Mehta, A. (2023). "Physics-informed neural networks for imaging." *Sci. Rep.* DOI `10.1038/s41598-023-48351-7`.
- Liu, X., Hu, P., Luo, W., Zhang, J., Zhang, F. & Su, H. (2025). "Physics-informed deep learning for accurate and efficient wavefront sensing in adaptive optics." *Opt. Commun.* 596, 132458. DOI `10.1016/j.optcom.2025.132458`. —— PINN 用于波前传感，支撑"物理约束残差"路线。
- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2017). "Physics Informed Deep Learning (Part I)." arXiv `1711.10561`. —— PINN 原始定义。

### 8.1 文献空白（经核验）

经检索，**没有**已发表工作同时满足：(1) 向 CCD 未观测像素外推、(2) 探针按置信度主动选择、(3) 以物理约束锚定残差拟合。三者分别有文献，但组合无先例——本报告是该组合的首次离线验证。

## 9. 限制与下一步

1. §2–§4 为离线结果，§7 已完成标定与生产采集且 fit `ok=True`；生产 open-loop 指标不等同于任意形状实测闭环验收。后续闭环必须固定同块曝光，保存候选相位、原生 CCD 帧及逐轮接受/拒绝记录；单目标旧脚本 `sim2f_fov_hw.py` 仅保留作兼容入口。
2. 器件像差固定为 Noll 4/6/9/14；其他像差谱可能改变 64×64 vs 96×96 的排序。
3. Lane C 的物理项权重是手设的（support=1.0, band=1.0, smooth=0.05）；support 项权重 1.0 时有害（held_out +89%，9.6σ），降到 0.01 才与 `data_only` 不可区分。下一步应把 support 改为**硬约束**（残差在孔径外强制为 0，而不是惩罚）或按残差能量归一，并做验证集选择——不能在本报告的 `l1_out` 上挑权重，那是用评价指标做选择。
4. `tiny` 与 `oversized`（Lane C）的 `l1_out` 是噪声比（窗外能量占比 3.7e-5 / 9.2e-4）；报告与验收一律以 `ncc_out`/`ee_out_err` 为准。Lane A 的 `tiny` 组同理，但 Lane A 的 `trials.csv` **未记录窗外能量分母**，下一轮应补上该字段。
5. Lane A 每窗只辨识一次、`n=2` 每形状组；Lane B 5 seeds、Lane C 3 seeds。统计功效有限，趋势判断以 σ 倍数为准，不以点估计排名。真实实验应增加相位数与重复。Lane B 的负面结果只针对当前 `select_probe` 实现，不代表 D-optimality 方法本身失效。
6. Lane B 的 `n_modes=10` 而评分只用 Noll 4..11 的 8 维，未评分模式可能吸收能量；这一点在下一轮必须统一，否则主动/固定两个变体的差异仍可能混入基不匹配。

## 附录 A. 复现命令

```bash
# Lane A
python scripts/ml/sim2f_fov_extrapolation.py --out report/shape_trials/fov_generalization
# Lane B（产物路径写死；不要加 --out）
python scripts/ml/sim2f_probe_active.py --seeds 5 --budgets 2 4 6 8 10
# Lane C
python scripts/ml/sim2f_pinn_extrapolation.py --out report/shape_trials/fov_generalization/artifacts/pinn
# 硬件会话 dry-run（不触碰设备；只验证 plan/resume/finalize 管线）
python scripts/ml/sim2f_fov_hw_session.py plan --out data/debug/sim2f_fov --screens 1 --repeats 2 --targets ring
python scripts/ml/sim2f_fov_hw_session.py capture --session data/debug/sim2f_fov --dry-run
python scripts/ml/sim2f_fov_hw_session.py finalize --session data/debug/sim2f_fov
# 门禁
python -m pytest tests/ao_shaping/ml/sim2f/test_window.py tests/ao_shaping/ml/sim2f/test_extrapolate.py tests/ao_shaping/ml/sim2f/test_probe_active.py tests/ao_shaping/ml/sim2f/test_pinn.py tests/ao_shaping/scripts/test_sim2f_fov_hw_session.py -q
python -m ruff check scripts/ml/sim2f_fov_extrapolation.py scripts/ml/sim2f_probe_active.py scripts/ml/sim2f_pinn_extrapolation.py scripts/ml/sim2f_fov_hw_session.py
python -m ruff format --check scripts/ml/sim2f_fov_extrapolation.py scripts/ml/sim2f_probe_active.py scripts/ml/sim2f_pinn_extrapolation.py scripts/ml/sim2f_fov_hw_session.py
python scripts/_common/term_unify_20261016.py   # 应报 0 个文件
python scripts/sync_report_provenance.py
```

## 附录 B. 原始产物契约

- `artifacts/trials.csv`：列 `window,case,phase,arm,fit_applied,confirm_l1,confirm_ratio,l1_all,l1_out,l1_in,ee_out_err,centroid_err_px,d86_err_px,ncc_out`。
- `artifacts/summary.json`：`meta`（含 `confirmation` 与 `ident_shot_l1`）、`aggregate`（per `window|group|arm`）、`commands`。
- `artifacts/probe/summary.json`：per `(strategy,budget)` 聚合 + per-seed 明细。
- `artifacts/probe/trials.csv`：Lane B 逐 `(seed,strategy,budget)` 明细，列
  `seed,strategy,budget,coeff_rmse,coeff_rmse_basis,trace,condition,rank,sigma2,mean_confidence,repeats,selected_modes`。
- `artifacts/probe/runs.json`：Lane B 每个 seed 的完整记录（含 `coeffs` 与 `amplitudes`）。
- `artifacts/pinn/summary.json`：per arm 聚合 + per-phase 聚合 + stage1 + runtime。
- `artifacts/pinn/runs.json`：per (seed, arm, phase) 明细，含 `metrics` 与 `l1_start/l1_end`。
- `artifacts/pinn/coeffs.npz`：拟合残差系数。
