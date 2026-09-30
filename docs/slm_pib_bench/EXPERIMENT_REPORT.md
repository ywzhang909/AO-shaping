# SLM 整形目标函数实验报告 —— Pearson 迁移与参数标定

**日期**: 2026-09-29 ~ 09-30
**台架**: Santec SLM-200 (SN 22030108, 1920×1200, 10-bit, 内存模式, @1064nm 2π=993 灰度)
+ 大恒 MER2-507-23GM NIR (SN FJB24112232, 2592×1944, IP 192.168.0.11)
**软件**: `slm-pib` / `slm-gsnet` (SPGD + 启发式), FourierGSNet `shaping_loss`
**数据来源**: `data/debug/slm_pib_*` (44 个搜索 run)、`data/debug/slm_pib_online/*` (3 次 SNR 扫描 + 8 次 smoke)

> 本文是**结论汇总**。所有数字均可由离线脚本复现:
> - 台架验收报告: `docs/slm_pib_bench/report.md` (`scripts/generate_pib_bench_report.py`)
> - 离线三目标对比: `docs/slm_pib_online/objective_comparison.md`
> - 实机三目标对比: `docs/slm_pib_online_hw/objective_comparison.md`
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

### 3.2 `delta` 不是越小越稳 —— 关键在信噪比能否被"看见"

SPGD 只有在能看见自己梯度时才能优化 (`SNR = ΔJ_signal/σ_noise ≳ 2~3`)。
实测 (n_max=9 → 54 自由度, Pearson, 同台架):

| delta | epochs | best 改善 | **sustained 改善** | 判读 |
|---|---|---|---|---|
| 0.002 | 201 | +24.8% | **+0.2%** | best 好看, 但末帧回到起点 → 漂移 |
| 0.01 | 201 | +28.9% | +3.0% | 仍以漂移为主 |
| **0.1** | 61 | +30.4% | **+24.4%** | ✅ 真实优化 |
| 0.1 (重复) | 61 | +29.0% | +14.6% | ✅ 可复现 |
| 0.5 | 41 | +3.3% | +1.4% | 过大, 折叠门大量拒绝 |

`shape` 目标同趋势: delta=0.0005 时 best +32.5% 但 sustained 仅 +0.8%; delta=0.1 时 sustained +16.0%。

> **判据**: 只有 **sustained** 才是真优化。best − sustained 的巨大差值 = 噪声漂移。
> 因此 `delta<0.001` 在 n_max=9 下**不可用** —— 不是约束荒谬, 而是信号被淹没。

### 3.3 两个可分离的成因

1. **信噪比被自由度稀释.** SPGD 对所有模式施加随机 ±1 扰动, 单模式探测只动一个模式。
   实测同一 `delta=0.0005`: 单模式 SNR **2.25** (可用) vs 54 自由度 SNR **1.34** (不可用) —— 
   这正是小 delta 跑不动的直接原因。`snr_sweep` 因此同时给出两列, 且
   `usable_deltas()` 刻意按**多模式**过滤 (单模式列会高估 SPGD 实际能分辨的信号)。
2. **噪声是慢漂移, 不是散粒噪声.** 同相位 40 帧平均**未**降低 σ (2.8e-3 vs 10 帧 1.7e-3),
   所以加帧数无效; 必须靠 `+ - - +` 回文序列对消漂移 (见 §5 工具)。

### 3.4 噪声地板不是台架常数 —— SNR 必须每次实测

同一天、同一光斑位置三次扫描:

| 扫描 | 中心 (x, y) | σ_J | 单模式 SNR@0.0005 | 判定 |
|---|---|---|---|---|
| 08:58 | (665, 1031) | 3.71e-4 | 1.35 | 不可用 |
| 09:11 | (664, 1029) | 8.18e-5 | 2.85 | 可用 |
| 22:09 | (664, 1029) | 1.72e-5 | 3.25 | 强 |

σ 相差 **21×**。且注意: 单模式 SNR 从 1.35 改善到 3.25 的同时, 同 delta 的 smoke 采纳率
反而从 26/31 降到 5/31 —— **单模式 SNR 不能预测 SPGD 实际采纳率**, 进一步印证 §3.3.1
(采纳率取决于多模式信号 vs 逐步噪声门, 而非单模式 SNR)。

### 3.5 能量门频繁触发 = 搜索原地踏步 (排查首查项)

44 个 run 中, 6 个出现过 guard 惩罚行 (J=1e3 哨兵), 其 sustained 改善中位数
**−11.3%**; 而 38 个无 guard 的 run 中位数为 **+0.4%**。被能量门拦下的迭代不更新系数,
门一旦频繁触发, 调 delta 也无济于事 —— **先看 guard 计数, 再看 SNR/delta**。

### 3.6 其他实机事实

- 0 阶光斑在 **(x=673, y=1027)**, **不是**帧中心 (1296, 972) → 必须用 `argmax` 定位, 不可用几何中心。
- 曝光 1.2 ms 全帧下 0 阶稳定 ±1 px, 峰 58 / 均值 0.25 (真实局部光斑, 非均匀背景)。
- 无 DVI 挂起; 内存模式开 SLM 约 3.5 s; 写相位需轮换不同内存槽。

---

## 4. 结论汇总 (一句话版)

| 结论 | 依据 |
|---|---|
| `1 - Pearson` **不替换**现有目标, 仅作附加选项 | 实机 2/4 符号正确, 对照组 4/4 正确 |
| 但它**能优化** (loss −30%), 可显式选用 | delta=0.1 时 sustained +24% |
| `delta` 必须按**实测多模式 SNR** 选, 不能沿用历史值 | 噪底单日波动 21× |
| `delta<0.001` 在 n_max=9 下不可用 | 多模式 SNR ≈ 1.3, ~95% 迭代被门控 |
| `delta≈0.1` 是当前可用区间; `0.2+` 触发折叠门 | sustained 曲线 + fold 计数 |
| 降噪只能靠漂移对消, 不能靠加帧 | 40 帧平均反而更差 |
| 排查先看 guard 计数 | guard run sustained 中位数 −11.3% |

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
   读 `multi_snrs` 列 → `usable_deltas()` → 作为 `--delta`。**不要沿用本文数字** (噪底会变)。
2. **确认可用区间**: 用选定 delta 跑 `n_max=9` 的 `pearson` 与 `shape` 各 ≥200 epoch,
   要求 **sustained > 5%** 且 guard 计数低。
3. **若必须满足 `delta<0.001`**: 只能显著降低 `n_max` (自由度越少, 稀释越轻);
   或改用单模式/少模式扰动策略 (与"全模式 SPGD"是不同算法, 需重新评估)。

### 6.2 工具 / 流程改进

4. **让 `slm-pib` 的 delta 默认随 `n_max` 自适应**, 或在启动时做一次自动 SNR 扫描
   (类似 `rms-zernike` 的 delta 自动检测), 避免固定默认值在不同 `n_max` 下失效。
5. **`_gate` 导出到 debug pkl**: 目前 gate 统计只在 smoke 产物 (直接 pickle Recorder) 中可得,
   CLI 搜索 run 的 pkl 未含 `_gate`, 离线无法复盘门控行为。
6. **三目标对比报告考虑入库**: `docs/slm_pib_online*` 被 `.gitignore:112` 排除,
   实机结论目前只能靠重跑脚本复现; 建议放行或改存 `docs/slm_pib_bench/` 同级。

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
python scripts/generate_shape_objective_comparison.py --root data/debug/slm_pib_hw_20260929 -o docs/slm_pib_online_hw
```
