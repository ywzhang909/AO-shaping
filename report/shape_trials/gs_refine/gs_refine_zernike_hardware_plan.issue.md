# GS → Zernike → SPGD 匀化实验：阶数扫描已就绪，请求实验执行

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 硬件
> **说明**: GS → Zernike → SPGD 匀化实验阶数扫描的 GitHub issue 正文（实验测试请求, 非实验结论）; 交付拟合模块/扫描驱动/离线报告/8 阶仿真扫描报告 + 实验测试计划 H0~H12, 记录仿真已确立的硬件无关结论与仿真天花板、合并上游 #140 的基线变化、复现性边界
<!-- provenance:end -->

## 背景

已有的 GS-refine 整形链路能跑出方斑，但**没有回答"GS 逐像素相位能不能被低阶 Zernike 近似、然后只用
Zernike 系数做 SPGD 细化"**。本次交付把这条链路完整实现并在仿真中验证，实验测试计划已写好。

## 交付内容

| 类型 | 文件 |
|---|---|
| 拟合模块 | `src/ao_shaping/optimizer/wfless/gs_refine_zernike_fit.py` |
| 三阶段扫描驱动（sim + 硬件同一条码） | `scripts/slm/run_gs_refine_zernike_sweep.py` |
| 离线报告生成 | `scripts/slm/generate_gs_refine_zernike_report.py` |
| 8 阶仿真扫描报告 + 3 图 | `report/shape_trials/gs_refine_zernike/gs_refine_zernike_uniformity.md` |
| **实验测试计划 H0~H12** | `report/shape_trials/gs_refine/gs_refine_zernike_hardware_plan.md` |

流水线：`Stage A` 跑一次 GS-refine 并记录完整历史 → `Stage B` 把 GS 最优相位按
`n_max` 拟合成 Noll Zernike 系数 → `Stage C` 在系数上跑 SPGD，每个阶数独立记录。

## 仿真结论

**站得住的（与硬件无关）**
- Stage-B 拟合残差**随 `n_max` 单调不增**：8 阶扫描 0.0719 → 0.0385 rad（−46.5%，多 341 个模）。
  嵌套最小二乘基的性质，已用特征化测试钉住。
- `n_max=25`（351 模）不触发 `ZERNIKE_CLIP = 5.0`。
- `--lr 0` 复现已记载的步长拒绝失效（5/5 阶 `spgd_improved=false`）。
- 能量守卫在仿真里**不是瓶颈**（关掉结果不变）。

**站不住的（这就是为什么要实验）**
- 数字孪生是纯 FFT，而 `_derive_target_side` 用的是实验系统模型（`camera_pixel_um=2.2`、焦距来自实测
  tilt-shift scale），角尺度不匹配 ⇒ GS 预矫正在仿真里**打不过平场**，送进拟合的相位 ≈ 平场
  （`max|c|` 各阶恒为 0.0142 rad，拟合几乎没抓到结构）。
- **阶数之间在仿真里不可排序**：PIB 跨阶极差 0.0046、CV 极差 0.0039，与单轮 SPGD 抖动
  0.0010~0.0035 同量级。任何"最优阶数"都是在读噪声，报告因此**故意不给**该结论。

实验系统上 GS 相位是真实且高度不均匀的（剔零级 **CV ≈ 0.723**，面板梯度 ≈ 0.052 rad/px），
与仿真的"≈平场"是完全不同的 regime——这正是必须实验的理由。

## 复现性边界（已在报告局限中记录）

- 完全相同的命令行在 12 路并发与低负载下**逐位一致**（9 位小数全同）⇒ Stage A 本身确定性良好。
- 但 `--n-max` 列表或 `--max-energy-loss` 不同时，Stage A 的最优相位会整体改变（同一 `n_max`
  跨 run 的 `fit_rms` 相差 0.008~0.012 rad），而这两个参数都在 Stage A **之后**才被读取。
  **根因未查明。**
- 结论：`fit_rms` **只在同一次 run 内跨阶可比**，跨 run 绝对值比较无效。

## 合并上游的副作用（已处理）

上游 issue #140（`540d51b2`）把 GS 最优相位从"获胜的 ±delta 探针"改成"干净的 dof 收敛点"：

```python
-  best_phase = measured_phase   # 带 ±delta rad 随机扰动
+  best_phase = phase_dof        # 未扰动的收敛点
```

`fit_rms(n_max=4)` 因此从 0.2349 → 0.0826 rad、`max|c|` 0.05 → 0.017。报告已据此重生成。
**所有基线都是 #140 之后的**，不要与 2026-10-16 之前的数字直接比较。

## 请求

请按 `gs_refine_zernike_hardware_plan.md` 执行，其中两个 `P0` 门是硬阻塞：

- **H1 GS 对比择优**：若 GS 在本实验系统仍打不过平场，**立即停止**——先查 `--camera-pixel-um` /
  `--beam-radius-px` / 面板配准，不要继续烧 Stage C 的机时。
- **H2 噪声地板**：给出 CV/PIB/EE 的重复测量 σ。它是后续所有判据的分母；若 σ 与阶数间差异
  同量级，则 H9 的阶数排序**物理上不可判**，必须如实报告而非硬排。

核心科学项是 **H6**（Zernike 拟合在**远场**复现了多少 GS 整形效果）——`fit_rms` 只是相位空间的代理。

```bash
python scripts/slm/run_gs_refine_zernike_sweep.py \
  --cam-type daheng --cam-id 0 --exposure-time-ms 0.120 \
  --cam-size 256 --beam-radius-px 450 --slm-number 1 \
  --epochs 400 --gs-iters 200 --spgd-epochs 200 \
  --n-max 4 6 8 11 15 --lr 0.01 --delta 0.02 --max-energy-loss 0.6 --seed 0 --save-images
```

## 验证

- 680 项定向测试通过（新增 + `slm_zernike_pib` + `beam_metrics` + provenance）。
- 全量：13 failed / 5682 passed；其中 5 个"新增"失败已验证**在 `origin/banckend` 上同样失败**
  （golden digest 完全一致 `95ebbfec…→107aeb12…`），是上游改了代码没重生成 golden，
  **与本次合并无关**。
- ruff 仅剩上游本来就红的 2 条（`gen_xsec_v2.py`）。
- `sync_report_provenance.py --check` clean（107 条登记）。

## 已知未做

- **E1（SPGD 预算放大到 60 epoch）被砍**：仿真下需 300 epoch，并发下不切实际。
  "SPGD 不改善是预算不足还是物理上限"这个问题**未答**，已列为实验项 H7。
- 仿真 GS 物理预测模型标定未做（已确认 OOM：GS bench FFT 网格会到 7200²–14400²）。
