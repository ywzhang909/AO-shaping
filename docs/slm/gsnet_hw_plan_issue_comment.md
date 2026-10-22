## 实验方案（把设备占用压到最短）+ 两个新工具（准备就绪，issue #165 续）

接上一条评论（实测操作手册）。这一条给两件事：**怎么排才能少占设备**，以及为此新增的
两个**全程不打开设备**的工具。

完整方案在仓库里：[`docs/slm/gsnet_hw_plan_min_time.md`](https://github.com/ywzhang909/AO-shaping/blob/feat/165-gsnet-2f-calib-sim/docs/slm/gsnet_hw_plan_min_time.md)（分支 `feat/165-gsnet-2f-calib-sim`）。

### 一、结论先行：两段会话，合计 ≈ 10–15 分钟设备时间

| 会话 | 内容 | 采集次数 | 相机帧 | 计划墙钟 | 硬上限 |
|---|---|---|---|---|---|
| **S1** | 光功率门 + 灰度扫描 + 一次 MIL 采集（含几何预检） | ≈ 87 | ≈ 475 | **4–6 min** | 12 min |
| **离线块** | 拟合孪生 → 导出/迁移 → 验证门 → 报告 | 0 | 0 | 小时级（不占设备） | — |
| **S2** | 闭环 A/B：3 对交错、每臂一次 MIL | ≈ 138 | ≈ 950 | **5–8 min** | 20 min |

三条判断依据：

1. **设备时间由「采集次数」决定，不由脚本个数决定。** 一次 MIL 采集只有 42 次
   （几何预检 8 + 平场 1 + 3 轮 × 11），A/B 六次就是 138 次；拟合孪生、迁移、
   导出相位、双口径统计全部是纯计算。
2. **只有两次真正需要独占设备**：采一次数据、跑六次 A/B。其余都能推到离线块。
3. **吞掉设备时间的从来不是测量，是重跑与挂起**（见第六节）。

### 二、单次采集的开销口径（把假设摊开）

| 项 | 值 | 来源 |
|---|---|---|
| 显式等待 | 0.5 s | `slm_model_in_loop.py:807` `settle_wait_s=0.5` |
| 丢弃帧 | 3 | `bench_kernels.py:452` |
| 稳定判据 | 连续两次（峰值, 质心）相对一致 ≤ 2 %，最多等 6 s | `bench_kernels.py:453-455` |
| 平均帧 | 4 | `slm_model_in_loop.py:790` `n_eval_frames=4` |

⇒ **单次采集 ≈ 7 帧 + 0.5 s 面板沉降**。曝光在 0.02–0.4 ms 量级时帧本身可忽略，
**每帧成本实际由 0.5 s 显式等待 + SLM 写盘决定**。面板黏滞时（等满 6 s）
S1-3 最坏 42 × 6 s ≈ 4 min —— 这就是上限写 12 min 的原因。

### 三、会话 S1：测量（顺序是设计出来的，前两步能一票否决）

```bash
# S1-0 环境自检（不开设备）
Get-Process python*                                   # 无残留进程占 SLM 的 USB
$env:AO_DAHENG_MAX_EXPOSURE_MS = "0.4"

# S1-1 光功率门（≈ 5–9 帧，< 30 s）
python scripts/slm/diagnose_light_level.py --cam-type daheng --cam-id 0 \
    --slm-number 1 --slm-wavelength 1064 --exposures 0.02,0.05,0.1,0.2,0.4

# S1-2 灰度扫描（40 点 × 4 帧 = 160 帧，≈ 40 s）
python scripts/slm/validate_flat_phase_gray.py --scan --gray-step 25 \
    --exposure-ms <S1-1 定下的档位> --cam-type daheng --metric bucket \
    --csv data/slm_gray_response_daheng.csv

# S1-3 一次 MIL 采集（42 次采集 ≈ 290 帧，≈ 2–3 min）
python src/ao_shaping/main.py --debug slm-model-in-loop \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --exposure-time-ms <现定> --region 128 --far-field-padding 8 \
    --cam-size 300 --target-side 100 \
    --n-rounds 3 --probe-count 8 --holdout-probe-count 2 \
    --step-a-iterations 40 --step-b-iterations 200 --dir data
```

**S1-3 一次采集供四个下游**（这是省时间的关键）：

| 产物 | 路径 | 消费者 |
|---|---|---|
| 几何预检 | `preflight/geometry.json`（含实测 `camera_px_per_model_px`） | `fit_bench_twin.py` 自动读取 |
| 探针对 | `probes/*.npz`（`phase` / `model_grid` / 帧） | 拟合孪生 + 迁移训练集 + 验证门 |
| 源面强度 | `source_amplitude.npy` | 迁移与导出的网络的 source 输入 |
| 平场与整形帧 | `preflight/` 与 `probes/` | 报告口径核对 |

**fail-fast**：几何求解一旦 `correlation < 0.5`（`--min-geometry-correlation`）
就中止 ⇒ **坏光路 1 分钟内止损**。

### 四、离线块：0 帧、0 设备时间（小时级）

```bash
python scripts/slm/fit_slm_amplitude_coupling.py \
    --csv data/slm_gray_response_daheng.csv --out data/slm_coupling.json
python scripts/slm/fit_bench_twin.py --from-dir data/debug/slm_model_in_loop/<ts> \
    --coupling-config data/slm_coupling.json --out data/bench_twin/<ts>.json
python scripts/slm/export_gsnet_phase.py --checkpoint <ckpt> \
    --twin-config data/bench_twin/<ts>.json --forward calibrated ...
python scripts/slm/gsnet_transfer.py --checkpoint <ckpt> \
    --from-dir data/debug/slm_model_in_loop/<ts> \
    --twin-config data/bench_twin/<ts>.json \
    --freeze freeze-decoder --gate --epochs 20 --lr 1e-4
```

判据沿用仿真侧结论，不新造阈值：`phase_loss` 对照随机猜测水平 π²/3 ≈ 3.290；
`--gate` 判决「候选不如预训练权重就丢弃」。

### 五、会话 S2：闭环 A/B

```bash
# 先出计划，不打开设备
python scripts/slm/gsnet_init_hw_ab.py --plan \
    --gsnet-phase data/gsnet_transfer/<ts>/phase.npy --pairs 3

# 正式跑；timeout 必须从 5400 s 压到 600 s
python scripts/slm/gsnet_init_hw_ab.py \
    --gsnet-phase data/gsnet_transfer/<ts>/phase.npy --pairs 3 --timeout-s 600 \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --exposure-time-ms <现定> --target-side 100 --cam-size 300 \
    --region 128 --far-field-padding 8
```

协议不变（交错重放、第 i 对奇偶翻转、每次 MIL 前过光功率门）。统计口径：
报**配对均值差 + p 值 + `d_z`**，并写明 `min_attainable_pvalue(3) = 0.25`
⇒ 3 对**结构上**到不了 p < 0.05，这轮只报效应量与方向。

### 六、省时规则（按「省下的分钟/条」排序）

| # | 规则 | 省下 | 依据 |
|---|---|---|---|
| 1 | 一律 `gsnet_transfer --from-dir`，**不用 `--hardware`** | 一次重复采集 ≈ 20–28 次 ≈ 2–3 min | `--hardware` 会自己再开一次 MIL 采集 |
| 2 | `--timeout-s` 5400 → **600** | 单次挂起从 90 min 降到 10 min（6 stage 最坏 9 h → 1 h） | `gsnet_init_hw_ab.py:410` |
| 3 | 确认门 1 阈值已是网络初值单独的 **2.5 rad** | 否则 6 次 A/B 全被判 `trust_region_rejected`，8 分钟零信息 | 闭环自身最优相位 rms 1.678 rad > 0.5 rad |
| 4 | 一次采集供四个下游 | 省掉整段独立「采集会话」 | 见第三节 |
| 5 | fail-fast 顺序：光功率门 → 几何预检 → 灰扫 → 采集轮 | 坏光路 1 min 止损 | `min_geometry_correlation=0.5` |
| 6 | 只跑 3 对（6 次） | 4 对只把 p 下限从 0.25 降到 0.125 | 3 对已够报效应量 |
| 7 | A/B 前 `--plan` 空转一次 | 省掉因参数写错而重开的会话 | `--plan` 不打开设备 |
| 8 | 双口径 CV、参数扫描用已落盘帧**离线**算 | 零设备时间 | 见下面第八节 |
| 9 | 会话中不训练、不拟合、不出报告 | 避免设备被 CPU 任务拖住 | — |
| 10 | 跑前查残留 python 进程；**绝不**试 `video_mode=1` | 避免 SLM 挂死导致物理断电 | 仓库反模式记录 |

### 七、止损表（出现即停，不要在设备上试错）

| 现象 | 判断 | 动作 |
|---|---|---|
| 光功率门不过 | 没有光 / 光路堵 | 停 S1，查光路与快门 |
| 几何预检 `correlation < 0.5` | 探针没被分辨出来 | 停 S1；提 `--probe-spread` 或加 `--n-calibration-probes` 后重开一次 |
| 帧里找不到目标方框 | ROI 没锚在光斑上（0 阶不在画幅中心） | 用 `argmax` 锚定，别用画幅中心 |
| 单次采集 > 4 s | 面板黏滞 | 放弃本会话，物理断电复位后再来 |
| 每臂都是 `holdout_correlation_rejected` | 探针幅度在噪声底之下 / 几何没标定 | 停 S2，回离线块；不要靠加 run 次数硬凑 |
| 每臂都是 `trust_region_rejected` | 门 1 又回到绝对相位口径 | 停 S2，改配置 —— 这不是数据问题 |

### 八、两个新工具（**全程不打开设备**）

**（1）`scripts/slm/gsnet_sim_dryrun.py` —— 全链离线预演器**

把整条链逐段跑一遍并逐阶段给判决。我实跑的结果（退出码 0，共 ~74 s）：

| 阶段 | 状态 | 秒 | 判据 |
|---|---|---|---|
| `sim_mil` | OK | 22.6 | 仿真 MIL 链路验证通过，4 个产物齐全 |
| `probe_fixture` | OK | 6.1 | 12 条探针记录，形状与实验系统一致 |
| `twin_fit` | OK | 28.3 | 孪生拟合验证通过，标度来自 `preflight/geometry.json`（实测标度口径） |
| `coupling` | OK | 3.5 | 合成 CSV 自洽：周期 993.0、c=0.1200（真值 0.12）、ψ=0.700（真值 0.7） |
| `export` | **BLOCKED** | — | 没有预训练 checkpoint |
| `transfer` | **BLOCKED** | — | 同上 |
| `ab_plan` | OK | 8.7 | A/B 计划渲染成功，交错顺序与奇偶翻转校验通过（`--plan` 不打开设备） |
| `dual_aperture` | OK | 2.5 | 双口径 CV 验证通过：两个口径各一行，自检已做 |

依赖阶段失败 ⇒ 不复用旧产物；没执行 ⇒ 只有磁盘上已有产物才复用，否则 BLOCKED ——
**绝不静默跳过**。报告末尾给出「仍然必须在设备上执行」的清单。

**（2）`scripts/slm/dual_aperture_cv.py` —— 双口径 CV 离线后处理**

补上 issue 的 P1 并行项 (9)：CV **同时**报告两个口径，且每个数字都带口径标注。

* 口径 B / `target_full`（现行闭环口径）：目标方框 `std/mean`；
* 口径 A / `plateau_inner`（文献对标口径，Zhou 2010 的「小平坦区 rms」）：
  内层小平坦区 `rms/mean`；
* 两行不合并，CSV/JSON/MD 都带 `aperture` 与 `aperture_definition` 列。

两个实现上的要点：

* **光学轴没有落盘**（运行时把它冻在内存里）⇒ 只能从平场帧用仓库既有的
  `zero_order_center` 重算；找不到平场帧就**退出码 2**，**绝不**回退到画幅
  几何中心（0 阶不在画幅中心）。
* **自检**：重算口径 B 与该次运行记录的 `uniformity_cv` 比对
  （`cv_check_rel_delta`），`--strict` 超容差即退出码 1 —— 这是口径/坐标系
  搞错时的报警器（预演里这个偏差是 6.2e-10）。

### 九、预演器查出来的两个事实（会改变怎么做）

1. **仿真路径不写 `probes/*.npz`、也不写 `preflight/geometry.json`**（探针记录
   只在实验系统模式落盘）⇒ **拟合孪生不能吃仿真运行目录**，必须在实验系统上
   采到的目录上做。预演器因此自己造一份形状一致的 fixture，把这条离线链
   真正走通。
2. **没有预训练 checkpoint 时 `export` / `transfer` 判 BLOCKED（不是失败）** ——
   这正是当前工作树的真实状态（`data/` 下无 `data/gsnet_pretrain`）。

### 十、当前唯一的前置阻塞项

**预训练 checkpoint**（本工作树缺失）。没有它 ⇒ S2 进不去。
离线块先跑一次预训练（`--samples 2000 --epochs 24` 量级）即可解开。

### 十一、验证状态

* 新增测试 **21 passed**；连同四个脚本守卫测试 **689 passed / 45 skipped**
  （跳过项是「该脚本不用 `parents[N]`」的历史跳过，与新文件无关）。
* `ruff check` 新增/改动的 4 个文件：**全绿**。术语校验：**0 文件 0 行**。
* 预演器实跑：8 阶段 6 OK / 2 BLOCKED，退出码 0。
* **零设备触碰**：预演器与双口径工具都不 import 任何硬件驱动。

### 十二、还需要在实验系统上回答的（更新）

1. **耦合的实测 `c` 与 `ψ`**（S1-2）—— 决定孪生那条物理是不是真的。
   ⚠️ 预演里那组数是**合成 CSV 的自洽检查**，不是实测。
2. **拟合前向在真实帧上的保真度**（S1-3 之后）—— 仿真里校准前向 corr 0.9752、
   Zernike 基线 0.7318、理想 FFT 0.6302；实验系统上这组数还没有。
3. **闭环 A/B 的真实判决**（S2）—— P1 条款的第一次真实执行。
4. **实测的 `target_camera` 与 `run_config.json` 的 `target_side` 是否相等** ——
   双口径工具的自检依赖它；不等时自检会报一个非零偏差，那不是坐标系 bug。

> ⚠️ 报告口径提醒：**不要把闭环的最终分数说成网络的贡献。** 这一轮要回答的是
> 「在真实光路上，一个初值能不能帮闭环跳出局部最优」，不是「网络相位本身好不好」。
