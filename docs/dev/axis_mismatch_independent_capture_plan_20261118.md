# XY 轴失配独立数据采集与回验计划（2026-11-18）

## 1 目的与结论边界

上一轮结论（`data/debug/deswivel_forward_inverse_20261118/axis_three_arms/`）建立在
`data/debug/slm_transfer/20261117_202916/fit/inputs.npz` 上：baseline 的 `ramp_matrix`
就是在这 24 帧会话上拟合的，因而对同一批数据有结构性优势。所以上一轮只能否定
“把标定角/比例直接替换 baseline 矩阵”这一做法，不能证明或否定物理上的 xy 轴失配。

本计划只做一件事：**采一批从未参与任何拟合的新会话数据**，用同一套三个变体（baseline /
corrected / axis）在同一份数据、同一划分、同一超参下重跑，检验
“保持 +x 方位、不消旋，仅修正 xy 正交性与尺度失配”能否在**未见数据**上改善前向预测与
逆向优化。

本文件与配套脚本默认**不启动采集、不触碰设备**。只有设备窗口到位、六步光路探测通过、
并在回复中完成设备报告后，才允许由操作者执行带 `--execute` 的采集命令。

## 2 正确的数据管线（修正上一版）

上一版计划误指向 `capture_defocus_forward_dataset.py` + `fit_defocus_forward_dataset.py`。
`sanity 核对`表明 `compare_deswivel_forward_inverse_20261118.py` 读取的
`inputs.npz{phases, measured, camera_xy, train}` 与 `provenance.json{rows, ramp_matrix,
ramp_samples}` 由 **transfer 管线**产出，不是 defocus-forward 管线。正确链路是：

1. 采集：`python -m ao_shaping.tools.slm.transfer.capture` → 会话目录写
   `manifest.json`、`sample_*.npz`（`model_phase/raw_frames/panel_phase/panel_gray`）、
   `config.json`（含 `script_sha256`）、`_optical_probe.json`、`.capture_done`。
2. 冷启动检查点：`python -m ao_shaping.tools.slm.transfer.bootstrap` →
   `initial/frozen_parameters.json`（`grid_size` 由数据决定）。
3. 前向拟合：`python -m ao_shaping.tools.slm.transfer.fit` → `fit/inputs.npz`、
   `fit/provenance.json`、`fit/result.json`、`fit/iteration_*.npz`、
   `fit/frozen_parameters.json`。**这一步才是 `inputs.npz` 的唯一产出者**。
4. 三个变体回验：`python scripts/slm/compare_deswivel_forward_inverse_20261118.py --fit-dir <session>/fit`。

实证依据（本仓可复核）：
`src/ao_shaping/tools/slm/transfer/fit.py:321` 写 `inputs.npz`；
`fit.py:307` 的 `provenance` 含 `rows / ramp_matrix / ramp_samples`；
归档会话 `20261117_202916/capture/config.json` 为 `grid=320, count=24, frames=3,
seed=102319, save_dataset=false`，`fit/frozen_parameters.json` 为 `grid_size=320,
fft_size=1024, pixel_samples=3, ramp_matrix=[[165.614…,7555.30…],[-7539.29…,180.087…]]`。

## 3 采集设计（实验前冻结）

| 项 | 值 | 理由 |
|---|---|---|
| 会话目录 | `data/debug/slm_transfer/<新时间戳>/` | 全新，禁止复用/覆盖 |
| `--grid` | 320 | 必须与 warm-start 检查点 `grid_size` 一致，否则 `assemble_components` 直接报错 |
| `--count` | 24 | 复现归档的 12/12 划分（`index % 6 >= 3` 为 validation） |
| `--frames` | 3 | 与归档一致，fit 对原始帧取均值 |
| `--seed` | 20261118（新） | `probe_grid(index, size, seed+index)` 生成全新相位，绝不与任何历史拟合重叠 |
| `--exposure-max-steps` | 12 | 与归档一致 |
| `--save-dataset` | 开 | 保留 `sample_*.npz` 以便离线复核/重拟，不依赖内存路径 |
| `--ramp-periods` | 64 128（默认） | 校准 ramp 的 period，决定几何矩阵 |

**划分（预注册，采集前写死）**：probe 行按 `index` 排列；`index % 6 >= 3` → validation。
第 0..23 号 probe 中：
- train：`0,1,2,6,7,8,12,13,14,18,19,20`
- validation：`3,4,5,9,10,11,15,16,17,21,22,23`

三族（zernike/smooth/bandlimited，`index % 3`）在两个 split 中都出现，family 不是划分依据；
划分只按预先固定的 `index % 6`，**采集后不得按表现挑选或改动**。

**独立性来自哪里（关键）**：
- baseline 的 `ramp_matrix` 由**本会话的 calibration 行**（`reference_*` 与 `ramp_*`）推导，
  不参与任何 probe 的 train/validation 评估 → 消除了上一轮“baseline 在 eval 帧上拟合”的优势；
- 评估只用本会话的 validation probes，这些相位从未进入任何历史拟合；
- **修正参数冻结**：`corrected` 的 `panel_to_ccd_deg` 与 `scale_ratio` 取自
  `data/debug/issue247_analysis_20261117/issue247_analysis.json`（2026-11-17 标定），
  在本次评估中**不重新估计**。这才是“把先前假设拿到新数据上检验”的独立测试。

可选二次分析（非本计划主判据）：用本会话 ramp 重新估计角/比例后再跑一遍，回答
“每次上新标定是否有收益”，但它不再是独立验证，需单列。

## 4 三个变体定义与公平性

三个变体共用同一 `inputs.npz`（相位、实测、`camera_xy`、`train` 掩膜）与同一检查点，唯一差异是
`ramp_matrix`：

- `baseline`：`provenance.ramp_matrix`（本会话 ramp 拟合值）。
- `corrected`：`corrected_ramp_matrix(baseline, angle_deg=-88.54399725809812, scale_ratio=1.0859)`，
  列 0 落在标定角、列 1 严格正交、几何平均尺度守恒、y/x=1.0859。
- `axis`：`axis_corrected_ramp_matrix(baseline, scale_ratio=1.0859)`，
  保持 baseline 的 +x 方位、两轴正交、套 y/x=1.0859（**不消旋**）。

公平性铁律：
- 绝不单独旋转实测强度数组；`camera_xy` 与实测强度保持原始采样关系；
- 三个变体的 `--seed/--iterations/--patience/--min-delta/--inverse-*/--device` 全部相同；
- 目标、掩膜、初相位、逆向优化器与步数由脚本同一份代码生成。

## 5 预注册指标与判定规则

**主判据（前向，只在 validation probes 上）**：per-phase `correlation`、`relative L2`、
`nrmse`、`r²`。以每个 phase 为配对单位，报告 `corrected − baseline` 与 `axis − baseline`
的均值差与 bootstrap 95% CI。

**次判据（逆向）**：`DifferentiableUniformityOptimizer` 的 CV start/final/improvement
（同变体样本内）；再做“每个优化相位 × 每个物理预测模型”的 3×3 交叉评估，区分前向保真度与相位质量。

**判定规则（先写死，采集后不得改）**：
- 仅当 `axis − baseline` 在主判据（如 validation `r²` 或 `correlation`）上改善、且 bootstrap 95% CI
  不含 0、同时逆向交叉评估不恶化，才判定“不消旋的轴失配修正有收益”。
- 若 CI 覆盖 0 或方向相反 → 结论为“证据不足/无收益”，不得用训练集或单个 CV 数值宣布胜出。
- `corrected`（消旋）作为对照；上一轮已显示其不如 baseline，本轮用于看新数据是否复现。

## 6 实验前检查（任一失败即停，来自光路标定 skill 与硬件纪律）

1. 设备空闲：`DeviceLock.default().holder()` 为 `None`，无另一进程占用 SLM/CCD/激光；
   采集期不得并行启动第二个硬件任务（`--wait-until-idle` 语义）。
2. 六步光路探测（`libs/skills/ao-optical-path-calibration/SKILL.md`）：设备在线 / 光路有效 /
   曝光不超量程 / SLM 相位自检 / A=0 焦点确认 / SLM 确实在调制，产出本次
   `optical_path_record.json`。
3. **驱动板门禁**：transfer `capture` 自身**不调用** `require_drive_board_entry`，只在
   `tools/slm/transfer/open_loop.py` 里接线。采集前必须自行确认 SLM #1 驱动板应答
   （否则整批测量无效），不能仅凭 `open()` 成功就认为在调制。
4. 相机**全幅采集**，不做物理开窗；需要缩小分析范围只允许后处理裁切。
5. 采集前后在回复中报告：占用设备、做了什么、何时释放、是否正常关闭。

## 7 命令序列

### 7.1 先写预注册（离线、无设备）

配套脚本 `scripts/slm/plan_axis_mismatch_validation_20261118.py` 只读、不连设备：
它枚举 production `capture.probes` 的精确计划、校验新会话目录是否为空、
检查点 `grid_size` 是否与 `--grid` 一致，写出 `probe_plan.json`（schema
`axis_mismatch_validation/1`），并打印下列命令。

```powershell
python scripts/slm/plan_axis_mismatch_validation_20261118.py `
  --out-root data/debug/slm_transfer `
  --session axis_mismatch_validation_20261118 `
  --seed 20261118
```

> 本次已在 2026-11-18 离线生成
> `data/debug/slm_transfer/axis_mismatch_validation_20261118/probe_plan.json`
> （`plan_fingerprint` 前 16 位 `6334aeba36f06976`）。核对结果：34 条 probe，其中
> calibration 10、train 12、validation 12；train 索引 `[0,1,2,6,7,8,12,13,14,18,19,20]`、
> validation 索引 `[3,4,5,9,10,11,15,16,17,21,22,23]`，与 §3 完全一致；三族在两侧各 4 条。
> 参考检查点 `grid_size=320` 与 `--grid 320` 一致；冻结假设
> `panel_to_ccd_deg=-88.54399725809812`、`scale_ratio=1.085894931194651` 已写入
> `frozen_hypothesis`。脚本导入 `capture.probes` 但不拉入 `ao_shaping.drivers`
> （实测 `sys.modules` 无驱动模块、无 `gxipy`），生成过程未触碰设备。
> 实验前若要改计划，必须带 `--force` 重写（指纹变更会被记录）；采集开始后不得改动预注册。

### 7.2 设备窗口内执行（操作者 + 显式 `--execute`）

下列命令中 `<S>` = `data/debug/slm_transfer/axis_mismatch_validation_20261118`。
每个 `--out` 目录必须全新（`capture`/`bootstrap`/`fit` 均用 `exist_ok=False`）。

```powershell
# 1) 采集（占用 SLM #1 + Daheng CCD #0；结束自动 blank + 关设备）
python -m ao_shaping.tools.slm.transfer.capture --out <S>/capture --execute `
  --grid 320 --count 24 --frames 3 --seed 20261118 --exposure-max-steps 12 --save-dataset

# 2) 本会话冷启动检查点
python -m ao_shaping.tools.slm.transfer.bootstrap --dataset <S>/capture --out <S>/initial

# 3) 前向拟合（产出 fit/inputs.npz + provenance.json；cuda 可换 CPU）
python -m ao_shaping.tools.slm.transfer.fit --dataset <S>/capture `
  --checkpoint <S>/initial/frozen_parameters.json --out <S>/fit `
  --iterations 200 --span 120 --adaptation full --device cuda
```

### 7.3 三个变体回验（离线、无设备；可在设备释放后进行）

```powershell
python scripts/slm/compare_deswivel_forward_inverse_20261118.py `
  --fit-dir <S>/fit `
  --calibration data/debug/issue247_analysis_20261117/issue247_analysis.json `
  --out-root data/debug/deswivel_forward_inverse_20261118 `
  --run-name independent_20261118 `
  --device cuda --iterations 200 --patience 8 --min-delta 1e-4 --inverse-steps 60
```

```powershell
# 配对 bootstrap 95% CI（§5 判定规则要求；只读 report.json，不重训）
python scripts/slm/paired_bootstrap_deswivel_20261118.py `
  --run-dir data/debug/deswivel_forward_inverse_20261118/independent_20261118
```

说明：`--calibration` 固定传入归档 issue247 JSON（冻结修正参数，见 §3）。脚本的
bit-identical guard 复现的是**本次** fit 的最优 `iteration_*.npz`（`result.json` 的
`iteration/loss`），因此新数据下 guard 依然成立，**不需要** `--allow-guard-mismatch`；
只有当有人手工改过 fit 输出时才用该开关。`compare` 只输出均值差，per-phase 配对
bootstrap CI 由 `paired_bootstrap_deswivel_20261118.py` 从同一 `report.json` 计算，
写 `<run-dir>/paired_bootstrap_heldout.json`。

## 8 审查包（采后一次性归档，便于离线复跑）

- `<S>/capture/`：`config.json`（`script_sha256`、`exposure_seed_policy`）、`manifest.json`、
  `capture_records.pkl`、`_optical_probe.json`、`_optical_probe_flat.npz`、`.capture_done`、
  全部 `sample_*.npz`；
- `<S>/initial/`：`frozen_parameters.json`、`provenance.json`；
- `<S>/fit/`：`inputs.npz`、`provenance.json`、`result.json`、`frozen_parameters.json`、
  `iteration_*.npz`、`iterations.pkl`；
- `probe_plan.json`（预注册）；
- `optical_path_record.json`（本次光路标定）；
- 三个变体 `report.json` / `inverse.npz`（`compare` 输出）。

## 9 执行记录（2026-11-18 已采集并回验）

设备：Santec SLM #1（驱动板 `22030108`，内存模式，未断电重启）+ 大恒
MER2-507-23GM NIR（Mono8，8 bit，0.02–1000 ms）。WFS、DM 未使用。激光随光路常开。

| 段 | 命令 | 起止（本地） | 退出 | 释放 |
|---|---|---|---|---|
| 光路预检 | `probe_optical_path_validity.py --session axis_mismatch_20261118 --wait-until-idle 600` | 09:21:23–09:22:57 | 0（`verdict=ok`） | finally 放锁，`optical_path_record.json` 已写 |
| SLM 调制自检 | `diag_checker_cells_sweep.py --cells-list 24 --exposure-ms 0.05 --execute` | 09:28:27–09:29 | 0 | `release.json` 四项全 `ok` |
| 独立采集 | `transfer.capture --out <S>/capture --execute --grid 320 --count 24 --frames 3 --seed 20261118 --exposure-max-steps 12 --save-dataset` | 09:30:1x–09:32:15 | 0 | 进程结束，锁 `holder=None` |
| 冷启动 | `transfer.bootstrap --dataset <S>/capture --out <S>/initial` | 09:33 | 0 | 无设备 |
| 前向拟合 | `transfer.fit … --iterations 200 --span 120 --adaptation full --device cuda` | 09:33–09:34 | 0（early-stop 171，loss 0.191718） | 无设备 |
| 三变体回验 | `compare_deswivel_forward_inverse_20261118.py … --run-name independent_20261118` | 09:35–09:36 | 0（guard 通过） | 无设备 |
| 配对 CI | `paired_bootstrap_deswivel_20261118.py --run-dir …` | 09:37 | 0 | 无设备 |

采后审计（`manifest.json`，34 行）：

- 划分与预注册逐行一致：calibration = `reference_start` + 8 条 ramp + `reference_end`，
  train = `probe 0,1,2,6,7,8,12,13,14,18,19,20`，validation =
  `probe 3,4,5,9,10,11,15,16,17,21,22,23`；`fit/provenance.json rows=24`，
  held-out 12 帧未被静默削减。
- 曝光：`status` 全 `ok`，无 `raw.max() >= 250`；峰值 78–199 counts（8 bit），
  未触发饱和，`saturation_pixels=0`。探针帧统一 0.043 ms，参考帧 0.035 / 0.043 ms。
- 参考漂移：`reference_start` 光斑 (1225.86, 1104.06) → `reference_end` (1225.75, 1103.85)，
  位移 0.11 / 0.20 px；同曝光亮度 3171 → 4000 counts/ms（激光起亮漂移，由
  `radiometry=exposure` 的共享增益吸收；三个变体同帧同权重，不影响变体间比较）。
- 标签无重复，`sample_*.npz` 34 个与 manifest 行数一致，`.capture_done` 记
  `early_stopped=false`。
- `capture` 本身**不写** `released.json`：释放证据为进程退出码 0、`.capture_done` 落盘与
  `DeviceLock.default().holder() is None`。

## 10 结果与判定

前向（held-out 12 帧，按帧配对）：

| 变体 | correlation | r² | relative L2 | best iter / loss |
|---|---|---|---|---|
| baseline | **0.84352** | **0.67354** | **0.52912** | 171 / 0.191718 |
| corrected（消旋） | 0.84115 | 0.66998 | 0.53247 | 200 / 0.193291 |
| axis（不消旋、正交化 + 尺度） | 0.84115 | 0.67006 | 0.53244 | 200 / 0.193123 |

配对 bootstrap（20000 次，seed 20261118，`paired_bootstrap_heldout.json`）：

- `axis − baseline` correlation：−0.00237，CI95 [−0.00520, +0.00057]，含 0；
- `axis − baseline` r²：−0.00348，CI95 [−0.00824, +0.00160]，含 0；
- `axis − baseline` relative L2：+0.00332，CI95 [−0.00142, +0.00791]，含 0；
- `corrected − baseline` 同向且同样含 0（correlation −0.00237，r² −0.00355）。

逆向（同一目标，60 步）：CV final baseline 0.83303 / corrected 0.82624 / axis 0.82605，
improvement 0.11498 / 0.11426 / 0.11431；3×3 交叉评估中 baseline 前向对三个变体的相位都给出
最低 CV（0.8330 / 0.8345 / 0.8345），axis 前向最低（0.8267 / 0.8261 / 0.8260），差异来自
前向几何而非优化相位质量。

**判定（按 §5 预注册规则，采集后未改）**：`axis − baseline` 在主判据上的点估计为**负**
（correlation / r² 更低，relative L2 更高），且三个指标的 bootstrap 95% CI 全部覆盖 0。
→ **证据不足，不消旋的轴失配修正没有可检出收益**；不得用训练集指标或单个 CV 值宣布胜出。

**边界**：baseline 变体用的是**本会话自己** ramp 标定出的矩阵
（`[[155.97, 7570.39], [−7489.12, 196.04]]`），所以这次比较回答的是“要不要把该矩阵正交化
并按 1.0859 改 y/x 尺度”，而不是“物理上是否存在 xy 轴失配”。后者需要独立的轴失配测量
（旋转标定板或正交光栅双向扫描），不能由本批数据推断。

## 11 管线回归基线

- 归档会话 `20261117_202916` 的 `fit/inputs.npz`、`provenance.json`、`result.json` 齐全，可
  作为管线回归基线；新会话不得复用其 `capture/` 目录（必须 `exist_ok=False` 新目录）。
