# 正向模型闭环整形 (slm-model-in-loop)

> **生成脚本**: [`scripts/slm/generate_model_in_loop_report.py`](../../scripts/slm/generate_model_in_loop_report.py)
> **复现命令**: `python scripts/slm/generate_model_in_loop_report.py`
> **运行环境**: 离线 (纯源码静态分析 + 可选 torch 实测; 不开设备)
> **详细算法文档**: [`docs/slm/model_in_loop_algorithm.md`](model_in_loop_algorithm.md)

## 1. 这条命令做什么

每轮交替两步，用实测数据在闭环里持续修正正向模型，再用修正后的模型合成相位：

1. **Step A —— 拟合正向模型**：显示若干个**强随机探针**相位，把每个探针的远场实测帧与正向模型的预测对比，重拟合**一个共享的 Zernike 像差向量**。
2. **Step B —— 合成方形**：**冻结**刚拟合出的像差，优化**全像素 SLM 相位**逼近方形目标，下发实测；本轮相位成为下一轮的热启动。

两个守卫 (§6) 抑制两步相互耦合振荡；最终相位只有**实测优于平场基线**才提交。

## 2. 调用关系

```mermaid
graph TD
    CLI["main.py · slm-model-in-loop"]
    RUN["runners/slm/model_in_loop_runner.py · run()"]
    OPT["optimizer/wfless/slm_model_in_loop.py · optimize_slm_model_in_loop(config)"]
    MATH["optimizer/wfless/model_in_loop_shaping.py · 共享数学 (孪生/硬件复用)"]
    ZCO["algorithm/signal_processing/ · ZernikeCoefficientOptimizer (Adam)"]
    CCD["drivers/ccd/ · DahengCamera/MIICamera"]
    SLM["drivers/slm/santec.py · Santec"]
    
    CLI --> RUN
    RUN --> OPT
    OPT --> MATH
    MATH --> ZCO
    OPT --> CCD
    OPT --> SLM
```

## 3. 计算时序 (每轮)

```
硬件会话预检 (未指定 --geometry-file)
  └─ 全板随机相位盘 + 成对平场 → 当前入射位置；漂移超限则拒绝
  └─ 当前画面探测曝光 → sweep 斜坡/Zernike 几何 → bench_geometry.json
     └─ 相关度 < min_geometry_correlation → 中止，不提交成形相位

平场基线 (浅绿)
  └─ display(flat) → measure() → _metrics_at → _quality → score_before

第 t 轮 Step A (蓝)
  ├─ probe_count 次:
  │   ├─ _probe_phase(region, probe_spread, seed) → 零均值高斯瞳面相位
   │   ├─ display(probe) + measure() → prepare_frame
  │   └─ _to_model_grid() 搬到模型网格
  └─ step_a_iterations 次 Adam 更新 (轮流喂全部探针，同一 Adam 状态)
     └─ trust_region_clamp → clamped

第 t 轮 Step B (粉)
  ├─ step_b_iterations 次全像素相位 Adam (梯度穿过解析 FFT)
   ├─ display(shaped) + measure() → prepare_frame → _metrics_at → _quality
  └─ acceptance_verdict(loss, score)
     ├─ accept: c←clamped, phase←shaped, lr 复原, streak=0
     └─ reject: lr×damp_lr_factor, 探针+escalate, streak+=1
        └─ streak ≥ max_rejection_streak → 中止 (退出码 2)
```

## 4. 关键设计点

### 为什么必须用探针 (Step A)
平瞳面聚焦成近 δ 函数，归一化强度对光滑低阶像差几乎不响应 —— 像差不可辨识。
探针相位 std > ~3 rad 时信号才越过 16-bit 量化本底两个数量级，拟合才良态。

### Step A 所有探针喂同一个 Adam 状态
焦面强度是瞳面相位的非凸函数，单个探针留下大量驻点。轮流喂多个独立探针提供消掉驻点所需的多样性。
`update` 连续 30 步无改善会锁存 (`PLATEAU_PATIENCE`)，无东西清锁存 → re-arm 必须绕**当前**系数重建优化器。

### Step B 只在最后测一次
Step B 是纯计算 (可微前向模型过 FFT)，硬件上不可微的只有「真实测量」。
实测只在末尾做一次，交给验收测试。因此 `step_b_iterations=600` 不产生 600 次设备往返。

### 冻结 Noll 1,2,3 (piston/tip/tilt)
它们**无法从远场强度辨识**：piston 常数无贡献；tip/tilt 只是搬 0 阶位置。
放开实测后果：56% 系数范数转移到这些简并方向而不带来收益。

### 孪生与硬件共享数学
`model_in_loop_shaping.py` 含 `calibrate_bench_geometry` / `_probe_phase` / `_fit_aberration_at_probes` / `shape_phase_with_frozen_aberration` / `simulate_iterative_shaping`。
`--cam-type sim` 只替换 `Bench` 实现 (`_SimBench` vs `_HardwareBench`)，数学逐位相同。

## 5. 主要选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-r, --n-rounds` | 迭代轮数 (拟合→整形→复测) | 6 |
| `--probe-count` / `--probe-spread` | 每轮探针数 / 探针相位 std (rad) | 8 / 4.0 |
| `--step-a-iterations` / `--step-a-lr` | Step A Adam 步数/学习率 | 80 / 0.05 |
| `--n-orders` | 拟合 Zernike 阶数 | 10 |
| `--step-b-iterations` / `--step-b-lr` | Step B Adam 步数/学习率 | 600 / 0.05 |
| `--frozen-modes` | 冻结的 1-based Noll 序号 | 1,2,3 |
| `--region` / `--far-field-padding` | 正向模型瞳面网格 / 远场补零倍数 | 64 / 8 |
| `--pupil-oversampling` | 瞳面双线性过采样 (Step A/B 子像素插值, 放大瞳面后按匹配远场采样再裁回 `--region`, 视场不变只提采样率) | 1 (可选 1/2/4) |
| `--panel-span-px` / `--pupil-center` | 探针覆盖面板宽度 / 光斑面板坐标 | 900 / (960,600) |
| `--camera-pixel-um` / `--slm-pixel-um` / `--focal-length-m` | 光路模型测量锚点 | 2.2 / 8.0 / 0.0 |
| `--target-side` | 目标方形边长 (相机像素) | 60 |
| `--w-efficiency` / `--w-uniformity` | 质量分权重。**w-efficiency 必须非零** | 0.6 / 0.4 |
| `--max-cv-regression` | 已验收成形相位之间允许的最大 CV 绝对增量 | 0.02 |
| `--min-geometry-correlation` | 几何 对比择优 阈值；低于直接中止 (退出码 2) | 0.5 |
| `--warm-start/--no-warm-start` | 热启动 | True |
| `--early-stop-score` / `--save-best-image` | 早停分数 / 保存最优图 | - |
| `--cam-type [daheng\|miicam\|sim]` / `--cam-id` / `--exposure-time-ms` / `--cam-size` | 相机配置 | - |
| `--slm-number` / `--slm-wavelength` / `--device` / `--dtype` / `--seed` | SLM/设备/种子 | - |

## 6. 逐轮守卫

### 守卫 1：Trust Region
逐轮系数变化 L2 范数被 `trust_region_c_l2` 截断。

### 守卫 2：逐轮验收
`acceptance_verdict(loss_before, loss_after, score_before, score_after)`：
- 拟合 loss 变差、或实测综合分变差 (容差 `acceptance_score_eps=0.001`) → reject
- 连续 reject ≥ `max_rejection_streak=3` → 中止 (实验系统落在模型之外)

对匀化任务，另以 `max_cv_regression=0.02` 限制相对最佳已验收成形相位的 CV 回退。
首轮仍由最终平场 对比择优 验收；之后即使 EE 提升使综合分微升，只要 CV
增加超过容差，就拒绝该轮并保留先前相位。容差只覆盖测量抖动，
`shape_iou_after` 仍须单独检查，CV 下降不等于方斑轮廓形成。

## 7. 会话曝光

`--exposure-time-ms 0` 从 Daheng 的最小曝光 `0.02 ms` 打开相机，
随后定位、几何 sweep 与优化器分别从当前画面探测曝光上限，并检查设备回读。
本次 2026-10-16 光路以该自动路径完成实测闭环，探测上限约 `0.20 ms`。
显式曝光仅用于已在**当前**光路测定的复现实验，必须仍受本次画面探测限制。

- **不要将历史 `1.5 ms` 当作当前光路推荐值**；它在本次会话会过曝。
- 上限取自 `AO_DAHENG_MAX_EXPOSURE_MS`（缺省或 `0` 时由画面探测饱和点）。
- 历史曝光数值**不可跨光路复用**：换激光功率或中性密度片后必须重测。
- 不同曝光下的 CCD 帧只有经过该曝光段线性化验证后才能直接比较。

## 8. 仿真模式

```bash
python src/ao_shaping/main.py slm-model-in-loop --cam-type sim -r 2
```

## 9. 硬件示例

```bash
python src/ao_shaping/main.py slm-model-in-loop \
    --cam-type daheng --cam-id 0 --exposure-time-ms 0 \
    --max-exposure-ms 0 --slm-number 1 --slm-wavelength 1064 \
    --panel-span-px 900 --camera-pixel-um 2.2 \
    --region 64 --far-field-padding 3 \
    --geometry-file data/debug/slm_model_in_loop/20261013_105911/bench_geometry.json \
    --holdout-probe-count 4 --step-b-iterations 800 -r 1
```

### 9.1 几何文件 (`--geometry-file`) 与 padding 换算

缺省不提供 `--geometry-file` 时，硬件 runner 每次会话自动做全板入射定位、
画面曝光探测和 sweep 几何标定，并把证据保存在该次 debug 目录的
`panel_scan/` 和 `session_geometry/`。若光路在扫描期间突变，漂移守卫会中止；
重新运行会重新探测。每轮 Step A 再用当前相位探针拟合相差。这个流程无需
手工更新历史中心、曝光或几何文件。

显式 `--geometry-file` 用于固定几何复现实验，它指向 `scripts/slm/model_in_loop_hw_runbook.py --stage all` 的 sweep
标定产物 (`bench_geometry.json`)。**只有 `region` 必须与本次运行一致**；远场补零倍数
(`--far-field-padding`) 可以不同 —— 一个远场格宽为 `λf/(P·d_model)`，故
`camera_px_per_model_px` 与 `P` 成反比，优化器会按 `P` 把它换算到本次运行的网格
(实测: `far_field=192` (region 64 / padding 3) 标定的 `2.7955` 换算到 `far_field=512`
(padding 8) 得 `1.048`，与解析估计 `1.028` 相差约 2%)。`region` 不一致仍会被拒绝：
瞳面网格本身不同，任何换算都修不好。**因此 §9 里 `--far-field-padding 3` 不再是必须项**，
用默认 8 也可以。

## 10. 异常时的独立表征探针

硬件 CLI 的自动会话标定已经覆盖位置、曝光和几何探测。需要进一步分析
漂移、相机噪声或 ABBA 稳定性时，按下列顺序运行独立诊断入口
（曝光值按当前光路选择），不要求每次正常自动会话前重复运行。

```bash
python -m ao_shaping.tools.ccd.drift_probe --exposure-ms <当前曝光>
python -m ao_shaping.tools.ccd.floor_probe --exposure-ms <当前曝光>
python -m ao_shaping.tools.zernike.abba_probe --exposure-ms <当前曝光>
```

见 [`pre_run_characterization.md`](pre_run_characterization.md)。

若实验系统无法被正向模型描述 (对比择优 失败)，改用 `slm-gs-refine`。

## 11. 输出契约

- `data/slm_model_in_loop/<日期>/`：逐轮历史 CSV、`best_phase.npy` (**raw 未包裹弧度**)、`bench_geometry.json` (拟合出的几何，供复现)、可选最优远场图 PNG
- `data/debug/slm_model_in_loop/<ts>/slm_model_in_loop_<ts>.pkl`：**每次运行都写**，不依赖 `--debug`。结构 `{epoch: record}`，带 `_epoch` 索引键，配 `.json` sidecar。行内只有标量 (由测试锁定)，逐帧图像不在其中；最优帧另存 PNG。

## 12. 终止状态

跨光路的一次性配置、自动运行、逐项验收与失败恢复见
[`model_in_loop_handoff.md`](model_in_loop_handoff.md)。自动脚本调用本 CLI 的
现有会话探测，不复用历史几何；三个独立诊断探针用于深入排查异常，
不要求在每次自动会话前重复执行。

| 状态 | 含义 |
|------|------|
| `completed` | 至少一轮被接受 (或正确地保留了平场) |
| `aborted_unidentifiable` | 几何标定从未达到要求的相关度 |
| `aborted_rejection_streak` | 连续 reject 达上限 —— 实验系统落在模型之外 |

退出码 `2` = 几何 对比择优 失败或持续落在模型之外，该轮**没有**提交任何相位。
