# `hw_defocus_zernike/` 的输入契约（按轮归档）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/hw_gs_defocus_probe.py`](../../../../scripts/slm/hw_gs_defocus_probe.py)
> **数据/关联脚本**: [`scripts/slm/run_gs_refine_zernike_sweep.py`](../../../../scripts/slm/run_gs_refine_zernike_sweep.py)
> **运行环境**: 硬件
> **说明**: 本系列的**输入契约**归档 (按轮): 两轮各自的分组记录 JSON/CSV、泽尼克阶数扫描的 summary.json 与 spgd_nmax*.csv、以及 before/after PNG (报告里 16 张泽尼克图的**直接来源**)、焦面标定证据 JSON; 逐帧 .npz 与相位 .npy 体积大不入库, 仍在 data/debug/。**第二轮那 11 个组的原始记录已被 10-21 15:57-16:01 第三轮首次尝试覆盖**, 归档的是现存的那一份, 详见正文
<!-- provenance:end -->

本目录是这个**报告系列**的输入契约：上游实验脚本
（`scripts/slm/hw_gs_defocus_probe.py`、`scripts/slm/run_gs_refine_zernike_sweep.py`、
`src/ao_shaping/tools/slm/panel/focus_probe.py`）落盘的原始记录，按**轮**归档，
与报告同目录（`report/AGENTS.md` §4）。

| 子目录 | 是哪一轮 | 里面是什么 |
|--------|----------|------------|
| `round1_20261020/` | 第一轮（2026-10-20） | `hw_groups_20261020/`（离焦组、噪声组、远场补拍组）、`hw_zernike_sweep_20261020b/`（泽尼克阶数扫描）、`focus_calibration/`（焦面标定证据） |
| `round2_20261021/` | 第二轮（2026-10-21） | `hw_groups_20261021/`、`hw_zernike_sweep_20261021/`、`focus_calibration/` |
| `plan_20261021.json` | 第二轮 | 下一轮（26 组）的采集计划，与 `round2_20261021/hw_groups_20261021/plan.json` 是同一份 |
| `plan_20261021_r3.json` | 第三轮 | 第三轮（27 组）的采集计划，**由 `scripts/slm/build_hw_sweep_plan.py` 生成**（不是手写） |

每个组目录里是三件套：`measurements.json` / `measurements.csv` / `run_config.json`；
泽尼克阶数组里是 `summary.json` / `spgd_nmax*.csv`，外加 `before_nmax*.png` / `after_nmax*.png` ——
这 16 张 PNG 就是报告里 16 张 `hw_zernike_{before,after}_nmax*.png` 图的**直接来源**。

对应的两份报告：[第一轮 `hw_defocus_zernike_20261020.md`](../hw_defocus_zernike_20261020.md)、
[第二轮 `hw_defocus_zernike_20261021.md`](../hw_defocus_zernike_20261021.md)。

## ⚠ 第二轮那 11 个组的原始记录已被第三轮的失败尝试覆盖

**`round2_20261021/hw_groups_20261021/` 下 11 个组目录（3 个离焦/噪声组
`dz_noise_hw_gs` / `dz_scan_hw_gs` / `dz_scan_model_gs` + 8 个远场补拍组
`zernike_farfield_nmax{3,4,6,8,11,15,20,25}`）里的 JSON/CSV，是 2026-10-21
15:57–16:01 那次**第三轮首次尝试**写进去的，不是第二轮原始数据**。**

第二轮采集时这 11 个组写的是同一批目录，第三轮首次尝试又写了一遍，把它们顶掉了；
第二轮那一份已经不在磁盘上。所以：

- 本目录归档的是**现存**的那一份，**不是**第二轮原始数据 —— 不要拿它复核第二轮正文的数字；
- 同目录下 `groups_manifest.json` / `manifest.json`（16:01）同样是第三轮那次写的；
  `plan.json`（15:23）是第二轮自己的采集计划；
- `zernike_stage_a_1/`（15:56）是第三轮的 Stage A 第一块，**不是**第二轮的；
  第二轮自己的 Stage A 在 `zernike_stage_a/`（13:40）；
- **第二轮自己的泽尼克阶数扫描是完好的**：`hw_zernike_sweep_20261021/zernike_order_nmax{3,4,6,8,11,15,20,25}/`
  里的 `summary.json` / `spgd_nmax*.csv` / `before_nmax*.png` / `after_nmax*.png`
  时间戳是 13:43–14:12，正是第二轮那一轮 —— 第二轮报告里的阶数结论与那 16 张图都出自这里。

第一轮 `round1_20261020/` 下的东西**全部**是第一轮自己的（时间戳 2026-10-20 11:19–11:45），
未被后来的任何一次采集覆盖。

## 第三轮的采集计划是**脚本生成**的

`plan_20261021_r3.json`（27 组）由 `scripts/slm/build_hw_sweep_plan.py` 生成，
不是手写。它与第二轮那份手写计划的三处差异：

- **新数据根**：分组根 `data/debug/hw_groups_20261021_r3/`、泽尼克扫描根
  `data/debug/hw_zernike_sweep_20261021_r3/`。第二轮与上一轮共用旧根，
  于是第三轮的离焦/远场组把第二轮那 11 个组的 `measurements.json` 顶掉了
  （见上文「已被第三轮的失败尝试覆盖」一节）。
- **块数 6 → 7**：每块墙钟预算 540 s，实测一块只跑出 ~39 轮（请求 45 轮），
  6 × 39 ≈ 234 轮够不到 250 轮的总目标。39 是墙钟截断的**结果**，
  每块**请求**的轮数仍是 45。
- **离焦组与远场组引用的相位文件改指新根**。上一轮那里写的是**旧根**，
  于是逐阶组失败后离焦组静默用了上一轮遗留的相位文件而不报错 ——
  生成器的自检现在会拒绝这种计划。

生成命令（会先自检，不过就报错退出、不写文件）：

```bash
python scripts/slm/build_hw_sweep_plan.py \
    --out report/shape_trials/hw_defocus_zernike/artifacts/plan_20261021_r3.json
```

## 不在库里的东西

以下产物**仍然只在** `data/debug/`（根 `.gitignore` 忽略 `data/*`，所以不入库），
体积或用途上不适合进版本库：

| 产物 | 大小 | 为什么不入库 |
|------|------|--------------|
| `phase_nmax*.npy` / `best_phase_nmax*.npy` | 18 MB / 个 | 逐点相位，报告正文只引用它们的拟合结果 |
| `fit_coefficients_nmax*.npy` | < 2 KB | 内容已在 `summary.json` 里 |
| `stage_a.npz` | 19–56 MB | Stage A 的续跑状态 |
| `frame_*.npz` | 逐帧原始画面 | 离线重算光斑量的输入；报告里的 FWHM / 环形度 / 质心都由它重算 |
| `hw_summary.png` | 85–160 KB / 组 | 上游探针自己的汇总截图，报告不引用 |
| `baseline_flat.png` / `baseline_gs_freeform.png` | 12–28 KB | 扫描脚本的基线截图，报告不引用 |

**逐帧 `.npz` 与相位 `.npy` 不入库 ⇒ 报告里的图无法从本目录单独重建**，
必须回到 `data/debug/` 的那一轮采集目录（各轮的采集目录见上文表格）。
