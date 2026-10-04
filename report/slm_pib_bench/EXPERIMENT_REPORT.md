# SLM 整形目标函数实验报告 —— Pearson 迁移与参数标定

<!-- provenance:start -->
> **生成脚本**: [`scripts/generate_pib_bench_report.py`](../../scripts/generate_pib_bench_report.py)
> **复现命令**: `python scripts/generate_pib_bench_report.py`
> **数据/关联脚本**: [`scripts/generate_shape_objective_comparison.py`](../../scripts/generate_shape_objective_comparison.py)
> **数据/关联脚本**: [`scripts/measure_shape_sensitivity.py`](../../scripts/measure_shape_sensitivity.py)
> **运行环境**: 离线
> **说明**: 台架验收总报告；含 Pearson 迁移与参数标定两节
<!-- provenance:end -->

**日期**: 2026-09-29 ~ 09-30
**台架**: Santec SLM-200 (SN 22030108, 1920×1200, 10-bit, 内存模式, @1064nm 2π=993 灰度)
+ 大恒 MER2-507-23GM NIR (SN FJB24112232, 2592×1944, IP 192.168.0.11)
**软件**: `slm-pib` / `slm-gsnet` (SPGD + 启发式), FourierGSNet `shaping_loss`
**数据来源**: `data/debug/slm_pib_*` (44 个搜索 run)、`data/debug/slm_pib_online/*` (3 次 SNR 扫描 + 8 次 smoke)

> 本文是**结论汇总**。所有数字均可由离线脚本复现:
> - 台架验收报告: `report/slm_pib_bench/report.md` (`scripts/generate_pib_bench_report.py`)
> - 离线三目标对比: `report/slm_pib_online/objective_comparison.md`
> - 实机三目标对比: `report/slm_pib_online_hw/objective_comparison.md`
>
> 注: 后两个目录被 `.gitignore:112` 排除, 未入库; 需时用对应脚本重新生成。

---

## 1. 实验目标

1. 将 FourierGSNet 的 `1 - Pearson` 损失迁移到硬件整形通路 (方形 + PIB), 并与既有手工目标对比。
2. 回答 `1 - Pearson` 能否**替换**现有目标函数。
3. 标定 SPGD 可用的扰动幅度 `delta`, 解释历史 smoke run 为何"跑不动"。

---

## 2. 交付内容

| 类别 | 内容 |
|---|---|
| 目标函数 | `pearson_shape_metric` (NumPy canonical) + 方形/PIB 通路贯通; `pearson` 已入 `GUARDED_OBJECTIVES` |
| 硬件路径 | `slm-gsnet --objective {quality,pearson}`; `slm-pib --objective pearson` |
| 溯源 | debug JSON sidecar 现记录 objective / cam_type / cam_id / exposure / zernike_radius / 搜索参数 |
| 参数测量 | `ao_shaping.tools.slm.slm_snr_probe` —— **设备实例由参数传入**, 不构造设备 |
| 报告 | 离线台架验收报告 + 三目标对比 (离线 & 实机) |
| 测试 | Pearson 指标/对齐 (24) + Torch 对齐 (6) + 方形搜索 sim (11) + SNR 探针 (33) + 报告回归 (13) |

---

## 3. 实验结论

### 3.1 `1 - Pearson` 能优化, 但**排序不可靠** → 维持 RISKY

在同一固定 anchor (recorder row 0) 下重算三个目标, 比较逐帧 Spearman 秩相关。
Pearson 是 loss (越小越好), 故与"越大越好"分数**负相关**才算一致。

**实机 (4 run, delta=0.1, 60 帧/-run):**

| run | ρ(square, pearson) | ρ(metrics, pearson) | 对照 ρ(square, metrics) |
|---|---|---|---|
| 161710 | **+0.405** | +0.566 | +0.956 |
| 161856 | **−0.529** | −0.267 | +0.926 |
| 163533 | **+0.201** | +0.476 | +0.915 |
| shape 162049 | −0.001 | +0.292 | +0.884 |

- 仅 **2/4** run 符号正确, mean |ρ| = **0.442**;
- 对照组 (两个 composite 之间) **4/4** 正确, mean ρ = **+0.920** → 帧集与 anchor 无问题, 弱点只在 Pearson。
- **离线 6 run 独立复现同一结论**: 2/6 符号正确, mean |rho| = 0.355, 对照 +0.932。

**结论**: Pearson 作为**优化方向**有效 (loss 实测降 30%), 但作为**帧排序信号**太弱且逐 run 变号,
不能当收敛判据。定位为**显式可选的附加目标**, 保留能量门, 不替换 `quality`/`shape`。

### 3.2 ⚠️ 修正: 「sustained」曾被端点噪声欺骗 —— 搜索实际是随机游走

> **本节在 2026-09-30 复测后推翻重写。** 原文用 `final vs first` 作为
> "sustained 改善" 判据, 结论是 `delta≈0.1` 可用 (sustained +18.8%)。
> **该结论错误**, 下表为同参数复测结果。

**为什么 `sustained` 会骗人**: 轨迹若是随机游走, 末帧落在低点纯属运气。
`frac_decreasing ≈ 0.5` (涨跌各半) 才是关键判据, 而它始终 ≈ 0.5:

| n_max | delta | accepted/200 | frac_decreasing | 前 1/3 均值 | 后 1/3 均值 | **late_gain** | final-vs-first |
|---|---|---|---|---|---|---|---|
| 9 | 0.02 | 6/200 | 0.56 | 0.5315 | 0.4861 | +8.5% | **+24.1%** ← 端点假象 |
| 9 | 0.05 | 7/200 | 0.47 | 0.5071 | 0.4675 | +7.8% | +12.8% |
| 9 | 0.05 | 7/200 | 0.49 | 0.5187 | 0.5155 | +0.6% | +18.9% ← 端点假象 |
| 9 | 0.10 | 7/200 | 0.48 | 0.4730 | 0.4256 | +10.0% | +18.8% |
| 1 | 0.05 | 3/200 | 0.48 | 0.5310 | 0.5416 | **−2.0%** | −4.7% |

`delta=0.02` 的轨迹最能说明问题:
`0.529 → 0.400 → 0.522 → 0.527 → 0.533 → 0.534 → 0.513 → 0.526 → 0.527 → 0.456 → 0.402`
—— 先掉到 0.400 又回到 0.53, 最后又落在 0.402。头尾一比就是 "+24%", 但全程没有下降趋势。

**修正后的结论**: 在 (delta ∈ [0.02, 0.1]) × (n_max ∈ {1, 9}) 的全部组合下,
`frac_decreasing ≈ 0.47–0.56`、`late_gain ∈ [−2%, +10%]`, 即
**没有任何配置能让 SPGD 可靠收敛 —— 搜索在做随机游走**。
`late_gain` 比 `final-vs-first` 稳健得多, 后续判断应以前者为准。

**归因**: 95%+ 的迭代被噪声门拒绝 (accepted 仅 3–9/200)。真正瓶颈
**不是 delta 也不是 n_max**, 而是**每步 SPGD 的信噪比 ≈ 1**。
SPGD 逐帧取 `pos`/`neg` 两次测量求差, 而台架噪声是**慢漂移** (§3.4),
相邻两次采样之间的漂移与梯度同量级, 差分后残留即为噪声。

### 3.3 自由度稀释: SNR 层面成立, 但**不是**实机失效的原因

`snr_sweep` 在同 delta 下测得单模式 SNR 2.25 vs 54 自由度 SNR 1.34, 稀释确实存在。
但实机把 n_max 从 9 降到 1 (54 → 3 自由度, 稀释应大幅减轻) 后
**表现反而更差** (accepted 3/200, late_gain −2.0%)。

因此: **稀释能解释 SNR 数值, 不能解释实机为什么不收敛**。
主导因素是慢漂移污染梯度估计 (§3.4), 而非信号被摊薄到各模式。

### 3.4 噪声是慢漂移, 不是散粒噪声 —— 这是真正的瓶颈

| 观测 | 含义 |
|---|---|
| 同相位 40 帧平均**未**降低 σ (2.8e-3 vs 10 帧 1.7e-3) | 加帧数无法降噪 |
| 三次扫描 σ 相差 **21×** (3.7e-4 → 1.7e-5) | 噪声地板非常数, 逐次实测 |
| 降低 DOF 无效 | 瓶颈是共模漂移, 不是信号稀释 |

SPGD 的 `J(+δ) − J(−δ)` 是**两次相邻采样**之差, 漂移在其间近似恒定 → 直接进入梯度估计。
有效的缓解手段是**回文采样** (`+ − − +`), 使共模漂移在差分中抵消;
`abba_signal` (§5) 已实现并单测, 但**尚未接入 SPGD 主循环** —— 这是最值得做的下一步改进。

### 3.5 噪声地板不是台架常数 —— SNR 必须每次实测

同一天、同一光斑位置三次扫描:

| 扫描 | 中心 (x, y) | σ_J | 单模式 SNR@0.0005 | 判定 |
|---|---|---|---|---|
| 08:58 | (665, 1031) | 3.71e-4 | 1.35 | 不可用 |
| 09:11 | (664, 1029) | 8.18e-5 | 2.85 | 可用 |
| 22:09 | (664, 1029) | 1.72e-5 | 3.25 | 强 |

σ 相差 **21×**。且注意: 单模式 SNR 从 1.35 改善到 3.25 的同时, 同 delta 的 smoke 采纳率
反而从 26/31 降到 5/31 —— **单模式 SNR 不能预测 SPGD 实际采纳率**, 进一步印证本节结论
(采纳率取决于多模式信号 vs 逐步噪声门, 而非单模式 SNR)。
> ⚠️ 2026-10-01 修正：原文此处引用「§3.3.1」，但 §3.3 **没有子章节**，引用悬空；
> 上文两个 `### 3.4` 已重编号为 3.4/3.5/3.6/3.7。

### 3.6 能量门频繁触发 = 搜索原地踏步 (排查首查项)

44 个 run 中, 6 个出现过 guard 惩罚行 (J=1e3 哨兵), 其 sustained 改善中位数
**−11.3%**; 而 38 个无 guard 的 run 中位数为 **+0.4%**。被能量门拦下的迭代不更新系数,
门一旦频繁触发, 调 delta 也无济于事 —— **先看 guard 计数, 再看 SNR/delta**。

### 3.7 其他实机事实

- 0 阶光斑在 **(x=673, y=1027)**, **不是**帧中心 (1296, 972) → 必须用 `argmax` 定位, 不可用几何中心。
- 曝光 1.2 ms 全帧下 0 阶稳定 ±1 px, 峰 58 / 均值 0.25 (真实局部光斑, 非均匀背景)。
- 无 DVI 挂起; 内存模式开 SLM 约 1.7–3.5 s; 写相位需轮换不同内存槽。
- `create_camera(..., cam_size=N)` **不会**开窗 (只有显式 `reset_window` 才开)。离线 SNR 探针
  曾因此把 50 px ROI 放在 2592×1944 全帧上, 测得 `usable_deltas()=[]` 而实机同时
  在跑出真实改善 —— 已加入 `score_fn` / `window` 参数与 `roi_fraction` 诊断 (§5)。

---

## 4. 结论汇总 (一句话版)

| 结论 | 依据 |
|---|---|
| `1 - Pearson` **不替换**现有目标, 仅作附加选项 | 实机 2/4 符号正确, 对照组 4/4 正确 |
| **当前台架 SPGD 无法可靠收敛** (随机游走) | `frac_decreasing≈0.5`, `late_gain∈[−2%,+10%]`, accepted 3–9/200 |
| 主因是**慢漂移污染梯度**, 非 delta 或 n_max | 降 DOF 18× 反而更差; 加帧数无效 |
| 判据必须用 **`frac_decreasing` / `late_gain`**, 不能用 final-vs-first | 随机游走下末帧位置是运气 (§3.2) |
| `delta` 过大触发折叠门 | delta=0.2 → 144/200 被折断门拒绝 |
| 噪底非常数, 单日波动 21× | 三次扫描 σ 3.7e-4 → 1.7e-5 |
| 排查先看 guard 计数 | guard run sustained 中位数 −11.3% |
| 降噪只能靠漂移对消 (`+ − − +`) | 40 帧平均反而更差 |

> **⚠️ 与本文早期版本的差异**: 早期结论「delta≈0.1 可用 (sustained +18.8%)」已被 §3.2 推翻 ——
> 那是随机游走的端点假象。真实情况是**所有测试组合都不收敛**, 瓶颈是漂移。

---

## 5. 参数测量工具 (设备无关)

噪底 / ΔJ / SNR 逻辑已抽到 `ao_shaping.tools.slm.slm_snr_probe`, **设备实例由参数传入**,
不构造任何设备、不 import 驱动、不读 `.env` —— Daheng / MiiCam / 仿真相机 / fake 通用:

```python
from ao_shaping.tools.slm.slm_snr_probe import snr_sweep
with create_camera('daheng', cam_id=0, exposure_time_ms=1.2) as cam, \
     Santec(slm_number=1, wavelength=1064) as slm:
    r = snr_sweep(cam, slm, n_max=9, radius=480.0)
    print(r.sigma, r.multi_snrs, r.usable_deltas())   # 按多模式 SNR 给可用 delta
```

| 符号 | 作用 |
|---|---|
| `snr_sweep` | 噪底 + 单模式/多模式逐 delta SNR (设备实例传入) |
| `measure_noise_floor` | 固定相位下 σ (含慢漂移) |
| `abba_signal` | `+ - - +` 回文对消漂移的 ΔJ |
| `snr_verdict` | 阈值 → strong / usable / unusable |
| `SnrSweepResult.usable_deltas` | **按多模式 SNR** 给出的可用 delta |

消费者: `scripts/measure_shape_sensitivity.py` 与硬件门控测试
`tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py` 均委托该工具, 不会再各自漂移。
离线可测: `tests/ao_shaping/tools/test_slm_snr_probe.py` (33 例, 纯 fake 设备)。

---

## 6. 下一步建议

### 6.1 设备上线后 (按优先级)

1. **重标定 delta** (最高优先):
   ```powershell
   $env:AO_RUN_HARDWARE=1
   uv run pytest tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py -v -s
   ```
   读 `multi_snrs` 列 → 参考 `usable_deltas()`。**注意: 2026-09-30 实测该指标与
   优化成败不吻合** (它报 `[]` 而实机同时在跑出改善), 故只能作参考, 不能作唯一依据。
2. **改用收敛判据验收**: 每个候选 delta 跑 ≥200 epoch, 用
   `frac_decreasing` (应显著 > 0.5) 与 `late_gain` (前 1/3 vs 后 1/3 均值, 应明显为正)
   判定, **不要**用 `final vs first`。
3. **最高优先: 把回文采样接入 SPGD 主循环**。当前 `J(+δ) − J(−δ)` 取相邻两次采样,
   慢漂移直接进入梯度估计, 这是 95% 迭代被门控的根因。`abba_signal` 已实现并单测
   (`+ − − +` 可精确对消线性漂移), 但主循环仍是单次正负采样。这是投入产出比最高的改动。

### 6.2 工具 / 流程改进

4. **`_gate` 导出到 debug pkl**: 目前 gate 统计只在 smoke 产物 (直接 pickle Recorder) 中可得,
   CLI 搜索 run 的 pkl 未含 `_gate`, 离线无法复盘门控行为。**已修复部分**:
   sidecar 现记录 `n_max` / `delta` / `lr` / `epochs` / `n_eval_frames` / `noise_gate_k`
   / `fold_ratio` / `optimizer_type` (此前 `n_max` 读错对象被静默丢弃, 而它正是决定
   稀释程度的变量)。
5. **给报告分析工具加收敛判据**: `generate_pib_bench_report.py` 目前用
   `final vs first` 作 sustained, 在随机游走下会给出端点假象, 应补
   `frac_decreasing` 与 `late_gain`。
6. **三目标对比报告考虑入库**: `report/slm_pib_online*` 被 `.gitignore:112` 排除,
   实机结论目前只能靠重跑脚本复现; 建议放行或改存 `report/slm_pib_bench/` 同级。
7. **若必须满足 `delta<0.001`**: 降 `n_max` 已被实测证伪 (54→3 自由度反而更差);
   在漂移问题解决前, 任何 delta 都难以稳定收敛。

### 6.3 目标函数方向 (若要继续推进 Pearson)

7. 当前 Pearson 的问题是**秩相关弱且变号**, 而非数值错误 (Torch/NumPy 已 1e-9 对齐)。
   可尝试: 混合目标 (`w·composite + (1-w)·pearson`) 而非纯 Pearson; 始终保留能量门。
8. 方形 `slm-gsnet` 路径**仍无能量门** (PIB 已有) —— 若要在方形用 Pearson, 需先补 guard,
   否则可能复现 AGENTS.md 记录的 bare `-CV` 塌陷 (EE→0.002)。

---

## 7. 复现方式

```powershell
$env:PYTHONPATH = "src;libs"

# 离线台架验收报告 (不需硬件)
python scripts/generate_pib_bench_report.py

# 实机 SNR / smoke (需设备; 结果写入 data/debug/slm_pib_online/<ts>/)
$env:AO_RUN_HARDWARE = "1"
uv run pytest tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py -v -s

# 离线 SNR 扫描 (需设备)
python scripts/measure_shape_sensitivity.py --deltas 0.002,0.01,0.1

# 三目标排序对比
python scripts/generate_shape_objective_comparison.py --root data/debug/slm_pib_hw_20260929 -o report/slm_pib_online_hw
```
