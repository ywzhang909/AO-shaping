# report/ 子目录索引（外迁自 report/AGENTS.md §2.1）

### 2.0 目录布局模板（外迁自 report/AGENTS.md §2）

一份报告 = 一个子目录，报告与其独占资产同住，不平铺在 `report/<topic>/` 根下：

```
report/<topic>/<系列名>/
├── <报告名>.md
├── figures/       # 正文引用的图
├── images/        # 插图/动画
├── artifacts/     # 上游脚本原始产物 (CSV/JSON)，输入契约
└── assets/        # 现场抓取的原始资产
```

### 2.1 已确立的子目录（`report/shape_trials/`）

| 子目录 | 生成脚本 | 报告 |
|--------|----------|------|
| `transfer_uncertainty/` | `scripts/slm/transfer_uncertainty_experiment.py` | `transfer_uncertainty_20261104.md` |
| `gs_defocus_sim/` | `scripts/slm/generate_gs_defocus_sim_report.py` | `gs_defocus_sim_20261019.md` |
| `hw_defocus_zernike/` | `scripts/slm/generate_hw_defocus_zernike_report.py` | `hw_defocus_zernike_20261020.md`、`hw_defocus_zernike_20261021.md`、`hw_defocus_zernike_20261022.md` |
| `sim_vs_hw/` | `scripts/slm/compare_sim_vs_hw.py` | `sim_vs_hw_20261022.md` |
| `gs_golden_reference/` | `scripts/slm/generate_gs_golden_reference.py` | `gs_golden_reference.md` |
| `gs_refine_fwdmodel/` | `scripts/slm/generate_gs_refine_fwdmodel_report.py` | `gs_refine_fwdmodel_calibration.md` |
| `gs_refine_zernike/` | `scripts/slm/generate_gs_refine_zernike_report.py` | `gs_refine_zernike_uniformity.md` |
| `model_in_loop_cv/` | `scripts/slm/generate_model_in_loop_cv_report.py` | `model_in_loop_cv.md` |
| `online_recal_efficiency/` | `scripts/slm/generate_online_recal_efficiency_report.py` | `online_recal_efficiency.md` |
| `gui_closed_loop_shaping/` | `scripts/generate_gui_shaping_hw_report.py` | `gui_closed_loop_shaping_20261014.md` |
| `fast_recal_adapter/` | `scripts/ml/exp_fast_recal_adapter_report.py` | `fast_recal_adapter.md` |
| `sim_pretrain_transfer/` | `scripts/slm/generate_sim_pretrain_transfer_report.py` | `sim_pretrain_transfer_20261024.md` |
| `multishape1200/` | `scripts/slm/generate_multishape1200_report.py` | `multishape1200_20261027.md` |
| `transfer_square1200/` | `scripts/slm/generate_transfer_square1200_report.py` | `transfer_square1200_20261027.md` |
| `gs_refine/` | 手工撰写（GS 系列） | `gs_issue_live_20261015.md` 等 9 份 |
| `model_in_loop/` | 手工撰写（MIL 系列） | `model_in_loop_live_20261013.md` 等 9 份 |
| `physical_refine/` | 手工撰写（`sim2f_fov_hw_session.py` / `sim2f_fov_hw_refine.py`） | `physical_refine_plan_20261102.md`、`physical_refine_results_20261102.md` |
| `mask_closed_loop/` | `scripts/ml/sim2f_shape_trials.py` | `mask_closed_loop_20261031.md` |
| `fov_generalization/` | `scripts/ml/sim2f_fov_extrapolation.py` | `fov_generalization_20261101.md` |

同一系列在**同一天**追加轮次时，文件名保留日期并加 `_rN` 轮次后缀（如 `hw_defocus_zernike_20261022_r2.md` 表示 10-22 当天的第二轮）；跨天追加则直接换日期（`hw_defocus_zernike_20261020.md` → `..._20261021.md` → `..._20261022.md`）。生成脚本从系列前缀之后的**整段后缀**推导图名戳（`_20261022`、`_20261022_r2`），所以同系列的多份报告不会互相覆盖图。

**每一张图都要带这个戳** —— 逐组图族（远场拼版 / 光斑截面 / 泽尼克逐阶 / 离焦响应）与聚合图一视同仁，否则同系列的两份报告仍会互相覆盖对方的逐组图。同一系列里每份报告**独占**自己的图；正文已定稿、只想把图换回自己那一轮时，用生成器的 `--figures-only` 重画，不要重新生成正文（重新生成会让正文出现与那一轮事实不符的叙述）。

### 2.2 新增报告时

1. **先定归属**：找到生成它的脚本（`scripts/` 下）。同脚本 → 已有子目录；新脚本 → **新建子目录**，目录名 = 脚本语义名（去掉 `generate_` / `experiment_` 前缀）。
2. 报告放子目录根，资产放子目录的 `figures/` / `images/`。
3. 若是**手工撰写**的报告，归入对应的**实验系列**子目录（按报告名的语义前缀），不要另起新目录。
4. **跨系列的总结/计划**留在 `report/<topic>/` 根（如 `shape_trials/README.md`、`experiment_summary_*.md`）。

