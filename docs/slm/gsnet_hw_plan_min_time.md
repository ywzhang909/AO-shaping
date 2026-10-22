# GSNet 2f 迁移 + 闭环 A/B：最小设备占用实验方案（issue #165 实测部分）

> **这份文档回答一件事**：issue #165 里「必须在实验系统上做」的部分，怎么排成
> **两段短会话、合计 10–15 分钟设备占用**，并且每一分钟都花在能改变结论的地方。
>
> 它不回答算法为什么这样设计（见
> [`report/gsnet_2f_calibration/report.md`](../../report/gsnet_2f_calibration/report.md)
> 与 [`report/gsnet_aberration_transfer/report.md`](../../report/gsnet_aberration_transfer/report.md)），
> 也不替代逐条操作手册（见 [`gsnet_transfer_hw_runbook.md`](gsnet_transfer_hw_runbook.md)）。
> 本方案是**在它之上做减法**：去掉重复采集、把能离线的全部推出会话、给每个可能挂起的
> 命令加硬上限。

---

## 0. 结论先行：两段会话，合计 ≈ 10–15 分钟

| 会话 | 内容 | 采集次数 | 相机帧 | SLM 写 | 计划墙钟 | 硬上限 |
|---|---|---|---|---|---|---|
| **S1** | 光功率门 + 灰度扫描 + 一次 MIL 采集（含几何预检） | ≈ 87 | ≈ 475 | ≈ 90 | **4–6 min** | 12 min |
| **离线块** | 拟合孪生 → 导出/迁移 → 验证门 → 报告 | 0 | 0 | 0 | 小时级（不占设备） | — |
| **S2** | 闭环 A/B：3 对交错、每臂一次 MIL | ≈ 138 | ≈ 950 | ≈ 120 | **5–8 min** | 20 min |

三条判断依据：

1. **设备时间由「采集次数」决定，不由脚本个数决定。** 一次 MIL 采集只有 42 次
   （`n_calibration_probes=8` + 平场 1 + 3 轮 × 11），A/B 六次就是 138 次；
   拟合孪生、迁移、导出相位、双口径统计全部是纯计算。
2. **只有两次真正需要独占设备**：采一次数据、跑六次 A/B。其余全部可以推到离线块，
   在别的机器上跑（`fit_bench_twin` 强制 `--from-dir`，`gsnet_transfer` 支持
   `--from-dir`，`gsnet_init_hw_ab --plan` 根本不打开设备）。
3. **吞掉设备时间的从来不是测量，是重跑与挂起。** `gsnet_transfer --hardware`
   会在会话里**再采一遍**数据；`gsnet_init_hw_ab --timeout-s` 默认
   **5400 s**（`scripts/slm/gsnet_init_hw_ab.py:410`），一次挂起就顶 90 分钟 × 6 个
   stage。这两条是本次方案主要砍掉的对象。

---

## 1. 帧数与时长的计算口径（先把假设摊开）

单次采集的真实开销由 `capture_settled` 决定
（`src/ao_shaping/tools/kernels/bench_kernels.py:446-485`）：

| 项 | 值 | 来源 |
|---|---|---|
| 显式等待 | 0.5 s | `slm_model_in_loop.py:807` `settle_wait_s=0.5` |
| 丢弃帧 | 3 | `bench_kernels.py:452` `n_discard=3` |
| 稳定判据 | 连续两次 (峰值, 质心) 相对一致 ≤2 %，最多等 6 s | `bench_kernels.py:453-455` |
| 平均帧 | 4 | `slm_model_in_loop.py:790` `n_eval_frames=4`（CLI：`--n-eval-frames`，`runner_common.py:3164`） |

⇒ **单次采集 ≈ 3 + 4 = 7 帧 + 0.5 s 面板沉降**。曝光档在 0.02–0.4 ms 量级时，
帧本身可忽略，**每帧成本实际由 0.5 s 显式等待 + SLM 写盘决定**。

**因此本方案按 0.6–1.2 s/采集做预算**（含写盘）。若面板黏滞严重、频繁等满
6 s，阶段 2 的最坏情形是 42 × 6 s ≈ 4 min——这就是把上限写成「12 min」的原因。

---

## 2. 会话 S1：测量（≈ 87 次采集 / ≈ 475 帧 / 4–6 min / 上限 12 min）

**顺序是设计的，不是随意的：前两步都是「便宜且能一票否决」的门。**

```bash
# ---- S1-0 环境自检（不开设备，0 帧）----
Get-Process python*        # Windows：必须无残留进程占着 SLM 的 USB
$env:AO_DAHENG_MAX_EXPOSURE_MS = "0.4"   # 会话内现定，不要留空

# ---- S1-1 光功率门（≈ 5–9 帧，< 30 s）----
python scripts/slm/diagnose_light_level.py --cam-type daheng --cam-id 0 \
    --slm-number 1 --slm-wavelength 1064 --exposures 0.02,0.05,0.1,0.2,0.4
# 不过就必须停：没有光就没有任何可整形的东西，后面全是白烧

# ---- S1-2 灰度扫描（40 点 × 4 帧 = 160 帧，≈ 40 s）----
python scripts/slm/validate_flat_phase_gray.py --scan --gray-step 25 \
    --exposure-ms <S1-1 定下的档位> --cam-type daheng --metric bucket \
    --csv data/slm_gray_response_daheng.csv
# 40 点 = range(0, 993, 25)（validate_flat_phase_gray.py:184）
# 每点 1 次写 + 3 丢弃 + 1 采集；--gray-step 50 会漏掉周期峰谷，不要图省事放大

# ---- S1-3 一次 MIL 采集（42 次采集 ≈ 290 帧，≈ 2–3 min）----
python src/ao_shaping/main.py --debug slm-model-in-loop \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --exposure-time-ms <现定> --region 128 --far-field-padding 8 \
    --cam-size 300 --target-side 100 \
    --n-rounds 3 --probe-count 8 --holdout-probe-count 2 \
    --step-a-iterations 40 --step-b-iterations 200 --dir data
```

**S1-3 是本方案的核心复用点：一次采集同时产出四个下游要的东西。**

| 产物 | 路径 | 消费者 |
|---|---|---|
| 几何预检 | `preflight/geometry.json`（含实测 `camera_px_per_model_px`） | `fit_bench_twin.py:350-382` 自动读取 |
| 探针对 | `probes/*.npz`（`phase` / `model_grid` / 帧） | 拟合孪生 + 迁移训练集 + 验证门 |
| 源面强度 | `source_amplitude.npy` | 迁移与导出的网络的 source 输入 |
| 平场与整形帧 | `preflight/flat.png` 等 | 报告与口径核对 |

**顺序上的 fail-fast 保证**：`--n-calibration-probes` 默认 8
（`slm_model_in_loop.py:709`），几何求解一旦 `correlation < --min-geometry-correlation`
（默认 0.5）就中止。⇒ **坏光路在 1 分钟内止损，而不是烧完一整轮**。

**S1 可选的减法**（只在有明确理由时做）：

- 漂移探针只跑 12 帧而不是 24（`drift_period_s` 不变）——只是为了确认「这段会话内
  功率没有跳变」，不是重复 8 分钟漂移普查。
- 随机相位可分辨幅度探针（abba）只在打算把 `--probe-spread` 压到默认 4.0 以下时
  才跑；不打算压就跳过。
- 曝光档位若信任 `AO_DAHENG_MAX_EXPOSURE_MS` 的自动探测（倍增阶梯到饱和），
  可以不跑曝光阶梯，但那一步本身就是 9 档 × 2 次重复，便宜，建议留着。

---

## 3. 离线块：0 帧、0 设备时间（小时级）

全部在任意一台机器上跑；**会话 S1 结束就可以放设备**。

```bash
# 3-0 拟合耦合（S1-2 的 CSV），得到 c 与 psi
python scripts/slm/fit_slm_amplitude_coupling.py \
    --csv data/slm_gray_response_daheng.csv --out data/slm_coupling.json

# 3-1 反演这台设备的孪生（唯一入口；此前没有任何脚本调用过 fit_twin）
python scripts/slm/fit_bench_twin.py --from-dir data/debug/slm_model_in_loop/<ts> \
    --coupling-config data/slm_coupling.json --out data/bench_twin/<ts>.json
# 默认 maxiter 12 / popsize 8 / maxiter-fine 60（fit_bench_twin.py:164-170），分钟级

# 3-2 用拟合前向导出相位（P0 的最后一个离线检查点）
python scripts/slm/export_gsnet_phase.py \
    --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
    --twin-config data/bench_twin/<ts>.json --forward calibrated \
    --source data/debug/slm_model_in_loop/<ts>/source_amplitude.npy \
    --target <目标远场 .npy>

# 3-3 迁移（离线读 S1-3 的探针，不碰设备）
python scripts/slm/gsnet_transfer.py \
    --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
    --from-dir data/debug/slm_model_in_loop/<ts> \
    --twin-config data/bench_twin/<ts>.json \
    --freeze freeze-decoder --gate --epochs 20 --lr 1e-4 \
    --out data/gsnet_transfer/<ts>
```

判据（沿用仿真结论，不要自己发明阈值）：

| 检查点 | 判据 | 不通过怎么办 |
|---|---|---|
| 拟合孪生 | 拟合前后残差、`correlation`；几何标度须来自 `preflight/geometry.json` | 标度读不到就别继续——按标称焦距算出来的标度差 2.54 倍，系数全部不可信 |
| 迁移 `phase_loss` | 首末值对照随机猜测水平 π²/3 ≈ 3.290 | 贴着 3.290 = 数据量不够；**不要**据此宣布路线不通 |
| `--gate` | 微调后验证集 Pearson 不低于预训练权重 | 候选丢弃，退回预训练权重 |

> ⚠️ **换前向本身有代价**：预训练 checkpoint 内部是理想前向，换成拟合孪生后
> 迁移前分数会掉。若掉得厉害，正确做法是**先在离线块用拟合前向短训一次再迁移**，
> 而不是带着掉分的权重直接进 A/B。这一步仍在离线块内，不占设备。

> ⚠️ **本工作树 `data/` 下没有预训练 checkpoint**（已核实：无 `data/gsnet_pretrain`）。
> 会话前必须先在离线块跑一次预训练（`--samples 2000 --epochs 24` 量级，
> 32 进程生成约 100 样本/秒）。**没有 checkpoint 就没有 S2**，这是前置阻塞项。

---

## 4. 会话 S2：闭环 A/B（≈ 138 次采集 / ≈ 950 帧 / 5–8 min / 上限 20 min）

```bash
# S2-0 先出计划，不打开设备（0 帧）——确认交错顺序与臂参数
python scripts/slm/gsnet_init_hw_ab.py --plan \
    --gsnet-phase data/gsnet_transfer/<ts>/phase.npy --pairs 3

# S2-1 正式跑；timeout 必须从 5400 s 压到 600 s
python scripts/slm/gsnet_init_hw_ab.py \
    --gsnet-phase data/gsnet_transfer/<ts>/phase.npy --pairs 3 \
    --timeout-s 600 \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --exposure-time-ms <现定> --target-side 100 --cam-size 300 \
    --region 128 --far-field-padding 8
```

**协议是脚本已实现的，不要自己改**（`scripts/slm/gsnet_init_hw_ab.py:147-160`）：
两臂交错、第 i 对先后按奇偶翻转、每次 MIL 前都过光功率门
（`:250-259`，门不过记 `blocked_by_light_gate` 并跳过该 stage）。

**成本拆解**：每次 MIL 默认 `n_rounds=1 / probe_count=6 / holdout_probe_count=2`
（`:419-422`）⇒ 每 run = 8 预检 + 1 平场 + 9 = **18 次采集**，加上每 run 一次光功率门
（5 档曝光）≈ 23 次。6 个 run ⇒ ≈ 138 次采集 ≈ 950 帧。

**统计口径（先说清，避免事后挑数）**：

- 报**配对均值差 + p 值 + `d_z`**，并显式写出 `min_attainable_pvalue(3) = 2/2³ = 0.25`
  ⇒ 3 对**结构上**到不了 p < 0.05，这次只报效应量与方向；
- 单轮 best 分数不得作为可复现增益声明（本实验系统分数呈双峰）；
- 每臂的 `init_from_gsnet_status` 分布本身就是结果（门 1/2/3 各挡掉多少）；
- **跨会话数字不比较**（8 分钟内功率可漂 21 %）。

**为什么 A/B 不和 S1 合并**：合并会让 A/B 落在会话末段、光功率已漂移的时段；
而且 S2 必须等离线块产出 `phase.npy`。分两段后每段都短，漂移窗口小。

---

## 5. 省时规则（按「省下的分钟/条」排序）

| # | 规则 | 省下 | 依据 |
|---|---|---|---|
| 1 | 一律用 `gsnet_transfer --from-dir`，**不用 `--hardware`** | 一次重复采集 ≈ 20–28 次 ≈ 2–3 min | `gsnet_transfer.py:440-446` 的 `--hardware` 会自己再开一次 MIL 采集 |
| 2 | `--timeout-s` 5400 → **600** | 单次挂起从 90 min 降到 10 min（6 stage 最坏 9 h → 1 h） | `gsnet_init_hw_ab.py:410` |
| 3 | 会话前确认门 1 阈值已是**网络初值单独的 2.5 rad** | 否则 6 次 A/B 全被判 `trust_region_rejected`，8 分钟零信息 | 闭环自身最优相位 rms 1.678 rad > 0.5 rad |
| 4 | 一次采集供四个下游（几何/探针/source/平场） | 省掉整段「采集」独立会话 | 见 §2 表 |
| 5 | fail-fast 顺序：光功率门 → 几何预检 → 灰扫 → 采集轮 | 坏光路 1 min 止损 | `min_geometry_correlation=0.5` |
| 6 | 只跑 3 对（6 次），不上 4 对以上 | 4 对 = 多 2 次 run ≈ +2–3 min，而 p 下限只从 0.25 降到 0.125 | 3 对已够报效应量 |
| 7 | A/B 前 `--plan` 空转一次 | 省掉一次因参数写错而重开的会话 | `--plan` 不打开设备 |
| 8 | 双口径 CV、多帧平均参数扫描用**已落盘帧离线算** | 零设备时间 | 帧已存盘；`--n-eval-frames` 可在重跑时再调 |
| 9 | 会话中不训练、不拟合、不出报告 | 避免设备被 CPU 任务拖住 | — |
| 10 | 跑前查残留 python 进程；**绝不**试 `video_mode=1` | 避免 SLM 挂死导致物理断电 | 仓库反模式记录 |

---

## 6. 止损表（出现即停，不要在设备上试错）

| 现象 | 判断 | 动作 |
|---|---|---|
| 光功率门不过（各档峰值同量级或全黑） | 没有光 / 光路堵 | 停 S1，查光路与快门 |
| 几何预检 `correlation < 0.5` | 探针没被分辨出来 | 停 S1；提 `--probe-spread` 或加 `--n-calibration-probes` 后重开一次 |
| 帧里找不到目标方框 | ROI 没锚在光斑上（0 阶不在画幅中心） | 用 `argmax` 锚定，别用画幅中心 |
| 单次采集耗时 > 4 s（日志里 `max_wait_s` 被打满） | 面板黏滞 | 直接放弃本会话，物理断电复位后再来；硬撑着跑只会拿到一堆不稳的帧 |
| 每次 A/B 都是 `holdout_correlation_rejected` | 探针幅度在噪声底之下 / 几何没标定 | 停 S2，回离线块；不要靠加 run 次数硬凑 |
| 每臂都是 `trust_region_rejected` | 门 1 又回到绝对相位口径 | 停 S2，改配置——这不是数据问题 |

---

## 7. 会话前必须做完的离线欠账（否则会话白跑）

1. **预训练 checkpoint**（本工作树缺失）。没有它，S2 无从谈起。
2. **全链离线预演** —— ✅ 已交付：`scripts/slm/gsnet_sim_dryrun.py`
   把 8 个阶段（仿真 MIL → 探针 fixture → 拟合孪生 → 耦合拟合 → 导出相位 →
   迁移 → A/B 计划 → 双口径 CV）逐段跑一遍，**全程不打开设备**，逐阶段给
   OK / BLOCKED / FAILED 判决，末尾给出「仍然必须在设备上执行」的清单：

   ```bash
   python scripts/slm/gsnet_sim_dryrun.py
   ```

   两个它替我们查出来的事实：① **仿真路径不落 `probes/*.npz`、也不落
   `preflight/geometry.json`**（探针记录只在硬件路径落盘）⇒ 拟合孪生**不能**直接吃
   仿真运行目录，所以脚本自己造一份形状一致的 fixture；② 无 checkpoint 时
   `export` / `transfer` 判 **BLOCKED**（不是失败）——这正是当前工作树的真实状态。
3. **门 1 阈值核对**：`--init-from-gsnet-trust-region-c-l2` 默认 2.5 rad 生效
   （`runner_common.py:2903`）。
4. **双口径 CV 落实现** —— ✅ 已交付：`scripts/slm/dual_aperture_cv.py`
   读已落盘的远场帧，**离线**同时给出两个口径，且每个数字都带口径标注：

   ```bash
   python scripts/slm/dual_aperture_cv.py --from-dir data/debug/slm_model_in_loop/<ts> \
       --csv cv.csv --json cv.json --md cv.md
   ```

   * 口径 B（`target_full`，现行闭环口径）：目标方框 `std/mean`；
   * 口径 A（`plateau_inner`，文献对标口径）：内层小平坦区 `rms/mean`；
   * 自检：重算口径 B 与该次运行记录的 `uniformity_cv` 比对
     （`cv_check_rel_delta`），`--strict` 超容差即退出码 1 —— 这是口径/坐标系
     搞错时的报警器；
   * 光学轴**没有落盘**，只能从平场帧用仓库既有的 `zero_order_center` 重算；
     找不到平场帧就退出码 2，**绝不**回退到画幅几何中心。
5. **报告骨架**：先在 `report/` 建好待填表，避免会话后回忆口径。

---

## 8. 会话记录清单（写报告用，直接对应 issue 验收条款）

- S1：曝光档位区间、噪声底、几何 `camera_px_per_model_px` 实测值、灰扫 40 点 CSV、
  拟合出的 `c` 与 `psi`（周期是否 ≈ 993）；
- 孪生拟合：`maxiter`/`popsize`、拟合前后残差、拟合 JSON 路径；
- 迁移：`phase_loss` 首末值 vs 3.290、`shaping_loss` 首末值、`--gate` 判决、
  可训练参数数；
- A/B：每臂 `init_from_gsnet_status`、配对均值差、`d_z`、`min_attainable_pvalue`、
  两臂 CV/EE/复合分数、`--timeout-s` 实际值、每 run 秒数；
- 环境：相机型号与序列号、SLM 控制器编号、波长、曝光、`geometry.json` 内容。

> ⚠️ 报告口径：**不要把闭环最终分数说成网络的贡献**。这一轮要回答的是
> 「在真实光路上，一个初值能不能帮闭环跳出局部最优」，不是「网络相位本身好不好」。
