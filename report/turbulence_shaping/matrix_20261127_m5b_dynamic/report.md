# 湍流整形对比择优矩阵 (epochs=120, cam=256, n_max=8, side=120, pop=12)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_turb_shaping_matrix.py`](../../../scripts/slm/run_turb_shaping_matrix.py)
> **复现命令**: `python scripts/slm/run_turb_shaping_matrix.py --variants model_in_loop --turb-mode dynamic --epochs 120 --n-max 8 --cam-size 256 --out-root data/turb_matrix_m5b_dyn --report-root report/turbulence_shaping/matrix_20261127_m5b_dynamic`
> **数据/关联脚本**: [`scripts/slm/generate_turbulence_shaping_report.py`](../../../scripts/slm/generate_turbulence_shaping_report.py)
> **运行环境**: 离线（数字孪生 sim 设备，无需 SLM / CCD / 激光）
> **说明**: M5b dynamic 边界三格 (--turb-mode 覆盖轴): MIL 在时变屏下仍收敛出优于平场的终值 (终局对比择优三档 0.729/0.752/0.732 胜 0.665/0.666/0.667), 但逐轮验收大面积翻盘 —— accepted 2-4/120 (static 同槽 5-8/120), streak 峰值 78-110 贴放宽保险; dynamic 单元指标本身逐屏抽样, 与 static 数字不可跨协议直比。适用域结论: 准静态屏是 MIL 工作域, dynamic 为其验收边界; 终判维持
<!-- provenance:end -->

| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |
|---|---|---|---|---|
| model_in_loop | 1e-14 | 0.0635 | 2013 | [report.md](model_in_loop_cn21e-14/report.md) |
| model_in_loop | 2e-13 | 0.2832 | 2021 | [report.md](model_in_loop_cn22e-13/report.md) |
| model_in_loop | 2e-12 | 0.8942 | 1997 | [report.md](model_in_loop_cn22e-12/report.md) |

## 口径

- 屏协议 (本次实跑逐变体取值, 缺省协议见 cell_plan, `--turb-mode` 可覆盖): model_in_loop=`dynamic`;
- 幅度**实测** (stats), 收敛快慢/终值对比见各单元 GIF 与 metrics.png; 本矩阵为**对比择优**; MIL 的孪生湍流通道自 M5b 入列 (`_SimBench` disturbance 注入; GS-refine 通道见 matrix_20261121_m5a; #253 注记);
- 种子: dist_seed = 20261121 + 冻结网格序号 (``slot_of(variant, cn2)``; 子扫掠与基线逐单元配对)。
