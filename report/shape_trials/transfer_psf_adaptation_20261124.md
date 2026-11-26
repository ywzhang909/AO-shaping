# Transfer 物理预测模型静态 CCD PSF 适配验证报告

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线 CPU 拟合与评估（拟合与评估阶段不打开 SLM、CCD 或激光控制器，数据由先前会话采集）
> **说明**: 数值由冻结评估脚本计算，图由同一评估 JSON 直接绘制。评估脚本 `data/debug/slm_transfer/20261124_transfer_psf_v1/evaluate_psf.py`，SHA256 `828bf746757daad246ecaa8333d29e429e04bfa0e708a4b09908aff35f892a79`；绘图脚本 `data/debug/slm_transfer/20261124_transfer_psf_v1/make_psf_figures.py`，SHA256 `90d29c7d07d68d4c716a4112ef08649f1afad639870110ce8ad82eb6c6113a41`；复现命令 `python data/debug/slm_transfer/20261124_transfer_psf_v1/evaluate_psf.py`；训练命令见 §7。预注册 `docs/dev/transfer_psf_adaptation_prereg_20261124.md`。本轮只检验单一共享、静态、各向同性、有界 CCD 高斯 PSF；原 capture validation 门与独立 192 px capture 主门均通过，**支持将静态 PSF 适配作为显式 opt-in**；默认 `learn_psf=False` 保持不变。人工撰写。
<!-- provenance:end -->

> 类型：实测、离线拟合与独立 held-out 验证报告
> 日期：2026-11-24
> 分支：`feat/transfer-geometry-calibration-20261124`
> 总结论：**两道冻结门全部通过；静态 CCD PSF 适配支持为显式 opt-in，默认保持关闭。**

## 摘要

上一轮有界几何残差 `G = I + 0.05 * tanh(raw)` 未通过验证，`learn_geometry=False` 保持默认。随后仅对原 capture 的预测数组做离线残差诊断（未读取独立会话的 probe 图像/模型指标），发现主要残差不是几何偏差：对预测做约 2–3 CCD 像素的各向同性高斯展宽可明显降低 relative-L2，现有冻结 checkpoint 的 `sigma_ccd_px=0.0` 未建模该展宽。本轮因此在冻结前向的 CCD 模糊阶段引入**一个**可训练、有界的静态高斯 PSF，不扩大几何自由度，不拟合空间增益场，不引入逐图样自由参数。

核心结果：

1. **原 capture validation 门通过。** 6 个 validation probe 的 relative-L2 均值由 A 的 `0.60243929` 降至 B 的 `0.43018529`，相对改善 `28.5928%`（要求 `≥15%`）；20,000 次 phase-level paired bootstrap 的绝对改善 95% CI 为 `(0.087282, 0.276039)`，下界 `> 0`；correlation 均值变化 `+0.094949`（要求 `≥ -0.01`）。
2. **独立 192 px capture 主门通过。** 仅用预注册的 `probe_003/004/005`：relative-L2 均值由 `0.52978986` 降至 `0.41696567`，相对改善 `21.2960%`（要求 `≥10%`）；bootstrap CI `(0.040656, 0.193446)`，下界 `> 0`；correlation 均值变化 `+0.085852`；单行最差为 `probe_003` 改善 `10.5553%`，无任何一行恶化（要求每行恶化 `≤20%`）。
3. **学习值落在诊断范围。** B 导出 `sigma_ccd_px=4.33202982`（按 `FrozenDefocusForward._build_blaze_kernel` 约定，实际高斯标准差 `2.16601491 CCD px`），与诊断建议的 2–3 px 一致，且 A/B 的跨会话标量 gain 几乎相同（`0.86212446` vs `0.86402129`），故改善不是 session 增益重标定造成。
4. **描述性一致但样本有限。** 独立 capture 全部 6 个 probe 全部改善（相对改善 `10.47%–43.46%`），但主门只有 3 行、来自同一会话、同一批 ramp-only probe；结论只支持“静态各向同性 PSF 这一单项”，不宣称已消除旋转、倾斜、SLM 相位非线性、串扰、暗场杂散光或 CCD 响应非线性。

## 1. 问题与假设

`FrozenDefocusForward` 的冻结前向在光学预测后做一次 CCD 模糊卷积（`_build_blaze_kernel`，25×25、归一化、实际高斯标准差 `sigma_ccd_px / 2`）。现有 M3 checkpoint 导出 `sigma_ccd_px=0.0`、`pixel_samples=3`，即完全无 CCD 模糊；fit 只使用该固定核，不学习 sigma。离线诊断显示这低估了实测光斑的展宽。

本轮假设：设备与装调引入的一个**共享、静态、各向同性**高斯展宽可由单个有界参数描述，在 25×25 核内、以 `sigma_ccd_px / 2` 为实际标准差建模，即可显著降低 transfer 正向预测误差，且该改善能跨会话泛化到独立采集。

## 2. 预注册与数据角色

预注册 [`docs/dev/transfer_psf_adaptation_prereg_20261124.md`](../../docs/dev/transfer_psf_adaptation_prereg_20261124.md) 在任何独立 capture probe 图像/模型指标计算前写入，并冻结：

| 数据 | 角色 | 用途 |
| --- | --- | --- |
| 原 capture `20261124_transfer_geometry_v1` | 建模与内部验收 | ramp 估计矩阵；6 个 `split=train` probe 拟合 B；6 个 `split=validation` probe 只做一次内部验收 |
| `arm_A` | 冻结对照 A | 原 `learn_geometry=False`、PSF 关闭、iteration 67 的结果与预测，不改选 |
| `arm_psf` | 候选 B | 与 A 同数据/预算/选择规则，仅新增 `learn_psf=True` |
| 独立 capture `20261124_transfer_geometry_period192_holdout_v1` | 最终 held-out | 主检验仅 `probe_003/004/005`；`probe_000/001/002` 与全部 6 行仅描述 |

冻结门见预注册 §5：validation 门（改善 `≥15%`、CI 下界 `>0`、corr 变化 `≥ -0.01`）与独立主门（改善 `≥10%`、CI 下界 `>0`、corr 变化 `≥ -0.01`、单行恶化 `≤20%`）。跨会话主规则 `origin_mode="session"`，即 `origin_new = measured_new + (origin_fit - measured_old)`；只在独立 capture 的 `split=train` 行上重拟合一个标量 gain，不改变 background、bias、PSF、residual、matrix 或 origin。

## 3. 模型改动

仅改动三处，默认路径逐位不变：

- `PhysicalForwardFit.__init__` 新增 `learn_psf=False`、`initial_psf_sigma_ccd_px=1.0`、`psf_sigma_max_ccd_px=8.0`。关闭时不创建任何参数，`psf_sigma()` 返回 `None`，前向与导出与旧 checkpoint 完全一致。
- 学习时用 sigmoid 有界参数化 `sigma = sigma_max * sigmoid(raw)`，实际核为 25×25、归一化、`grid / (sigma/2)`，与 `_build_blaze_kernel` 逐位一致；核在 `prediction()` 之后、radiometry 之前卷积一次，`fit_loss` 中“动态核优先、否则固定核”，保证不双重卷积。
- `adaptation.configure_adaptation(..., learn_psf=False)` 只释放/冻结该参数；`fit.py` 的 CLI/`build_fit_model`/provenance/`frozen_parameters.json` 接线导出实际 learned `sigma_ccd_px`。

新增 7 项聚焦测试（`tests/ao_shaping/tools/slm/transfer/test_learn_psf.py`），覆盖默认关闭无参数/无核、核数值与冻结参考一致、单次卷积、有界性、导出可重载等。

## 4. 结果

### 4.1 原 capture validation（内部验收，6 行）

| probe | A relative-L2 | B relative-L2 | 改善 | A corr | B corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| probe_003_zernike | 0.37775378 | 0.31568656 | +16.43% | 0.923884 | 0.952446 |
| probe_004_smooth | 0.50623202 | 0.27558108 | +45.56% | 0.852864 | 0.963804 |
| probe_005_bandlimited | 0.63951285 | 0.53878033 | +15.75% | 0.803661 | 0.920395 |
| probe_009_zernike | 0.86101896 | 0.47258724 | +45.11% | 0.780781 | 0.940574 |
| probe_010_smooth | 0.63687982 | 0.43865800 | +31.12% | 0.799080 | 0.901282 |
| probe_011_bandlimited | 0.59323830 | 0.53981854 | +9.00% | 0.850012 | 0.901476 |
| **均值** | **0.60243929** | **0.43018529** | **+28.5928%** | 0.835047 | 0.929996 |

- 20,000 次 paired bootstrap 绝对改善 95% CI：`(0.087282, 0.276039)`，下界 `0.087282 > 0`。
- correlation 均值变化：`+0.094949 ≥ -0.01`。
- **总门：通过**（三项同时满足）。

### 4.2 独立 192 px capture 主门（仅 probe_003/004/005）

| probe | A relative-L2 | B relative-L2 | 改善 | A corr | B corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| probe_003_zernike | 0.38517237 | 0.34451618 | +10.5553% | 0.923510 | 0.943215 |
| probe_004_smooth | 0.52811985 | 0.33467341 | +36.6293% | 0.846028 | 0.952251 |
| probe_005_bandlimited | 0.67607736 | 0.57170741 | +15.4376% | 0.798416 | 0.930045 |
| **均值** | **0.52978986** | **0.41696567** | **+21.2960%** | 0.855985 | 0.941837 |

- 20,000 次 paired bootstrap 绝对改善 95% CI：`(0.040656, 0.193446)`，下界 `0.040656 > 0`。
- correlation 均值变化：`+0.085852 ≥ -0.01`。
- 单行最差：`probe_003` 改善 `10.5553%`（即无任何一行恶化，远超 `-20%` 门槛）。
- **总门：通过**。

### 4.3 描述性（不参与决策）

全部 6 行 relative-L2 与 correlation 均改善；`probe_000/001/002`（跨会话重复性 trio）改善 `10.47%–43.46%`，合并 6 行均值 `0.51626120 → 0.39677137`（改善约 `23.14%`），合并 corr 均值 `0.876022 → 0.944287`。主门只取 `probe_003/004/005`，描述性数字不用于结论。

### 4.4 交叉检查

- **学习值**：B 导出 `sigma_ccd_px=4.33202982`（实际高斯标准差 `2.16601491 CCD px`），处于诊断建议的 2–3 px 区间；A 保持 `0.0`。`iteration 67` 时 B 的 sigma 已为 `4.64403391`，到 `100` 收敛到 `4.33202982`，说明 PSF 分支早期即稳定，非只在末尾出现。
- **session 增益**：独立 capture 上重拟合标量 gain，A `0.86212446`、B `0.86402129`，几乎相同；相对改善 `21.30%` 不是增益重标定造成。
- **数值路径**：B 的原 capture 重建校验 `max_abs_deviation_counts=0.26615017` 略超未模糊数组的绝对容差 `0.05`，但 `relative_l2_deviation=6.8545e-05 < 1e-4`、`correlation=0.99999999913 ≥ 0.9999`，为 float32 舍入；仅将该数值路径检查改为同时满足相对误差与 correlation，不影响任何验收门（预注册 §7 已如实记录）。

## 5. 讨论与局限

- **一致性**：静态各向同性 PSF 在原 capture 与独立 capture 上方向一致地改善所有 probe，且 B 的 learned sigma 与离线诊断和既有历史报告一致，支持“一个共享、静态、有界高斯展宽能吸收部分设备/装调/成像限制误差”的假设。
- **局限 1（样本）**：独立主门只有 3 行、单一会话、单一 192 px 周期、单一曝光/位深；这是预注册冻结的小样本检验，改善的置信区间较宽（CI 下界 `0.040656`），不能外推到其他周期、曝光、位深或形状。
- **局限 2（模型类）**：本轮只支持或否定**静态各向同性 PSF 这一单项**。B 与 A 除 PSF 分支外还共享同一 100 次预算与 train-only 选择，且 B 未触发早停（A 停在 67、B 停在 100）；差异是“在该预算下学习 PSF 分支”的包络，不是单独卷积算子的因果剥离。B 的 sigma 在 67 次迭代已稳定到 `4.644`，但严格归因仍应以后续独立实验为准。
- **局限 3（未建模项）**：静态 PSF 无法解释 SLM 空间串扰、面板域相位低通、倾斜/旋转、暗场杂散光或 CCD 响应非线性；这些仍是后续候选。

## 6. 结论

预注册两道门全部通过，故本轮**支持将静态 CCD PSF 适配作为显式 opt-in**：在 `fit` 命令行加 `--learn-psf`（`--psf-sigma-init`/`--psf-sigma-max` 可选）即可启用。按照预注册 §6，**默认 `learn_psf=False` 保持不变**，不做任何默认行为变更；若要改变默认值，需另行确认，并保留新的独立验证。

## 7. 复现与执行记录

训练命令（B）：

```bash
python -m ao_shaping.tools.slm.transfer.fit \
  --dataset D:/Projects/TIFO/AO-shaping/data/debug/slm_transfer/20261124_transfer_geometry_v1 \
  --checkpoint D:/Projects/TIFO/AO-shaping/data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/checkpoints/M3_illumination/frozen_parameters.json \
  --out D:/Projects/TIFO/AO-shaping/data/debug/slm_transfer/20261124_transfer_psf_v1/arm_psf \
  --span 160 --iterations 100 --adaptation full --radiometry exposure --device cpu \
  --learn-psf --psf-sigma-init 1.0 --psf-sigma-max 8.0
```

评估命令：

```bash
python data/debug/slm_transfer/20261124_transfer_psf_v1/evaluate_psf.py
```

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| 聚焦测试 `pytest -q tests/.../test_learn_psf.py tests/.../test_learn_geometry.py` | 0 | **11 passed in 2.50s** |
| B 训练（上述命令） | 0 | train loss `0.48750→0.13345`，选中 iteration 100 |
| `evaluate_psf.py` | 0 | 写入 `psf_evaluation.json`；validation 门 `true`、独立主门 `true` |
| 相关回归套件 `pytest -q --import-mode=importlib tests/ao_shaping/tools/slm/transfer tests/ao_shaping/algorithm/test_physical_forward_phase_order.py -p no:cacheprovider` | 0 | **524 passed, 4 skipped in 169.17s** |
| 全仓库 `pytest -q --import-mode=importlib` | 143 | 收集 `14086` 项，`1800s` 超时；未产生失败断言，不宣称全仓库通过 |
| `ruff check <3 个源码文件 + 新增测试>` | 0 | `All checks passed!` |
| `git diff --check` | 0 | 无空白错误 |
| `graphify update .` | 0 | 图已重建并写入 `graphify-out/` |

注：首次汇总脚本曾在 B 的原 capture 重建校验处以 exit 1 停止，原因与处置见 §4.4 及预注册 §7；验收门与数据角色未改变。全仓库套件因 `1800s` 超时未完成，本轮以覆盖改动路径的相关回归套件作为可复核验证证据。

主要产物：

- 冻结设计与执行事实：`docs/dev/transfer_psf_adaptation_prereg_20261124.md`；
- 汇总评估：`data/debug/slm_transfer/20261124_transfer_psf_v1/psf_evaluation.json`；
- 候选 B：`data/debug/slm_transfer/20261124_transfer_psf_v1/arm_psf/`（A/B 独立跨会话预测另存 `*_new_session_prediction.npz`）；
- 图：`report/shape_trials/transfer_psf_adaptation_20261124/`；
- 本报告：`report/shape_trials/transfer_psf_adaptation_20261124.md`。

## 8. 后续建议

1. 若启用静态 PSF，保留 `learn_psf=False` 为默认；启用路径需在冻结 checkpoint 上重训并复核导出 `sigma_ccd_px`。
2. 下一候选为 SLM 空间串扰/面板域相位低通（插入点在 `defocus_phase()` 的 control→panel 上采样之后、静态像差之前），仍需新预注册与独立检验；避免在本独立 capture 上继续调参。
3. 若追求更强归因，可预注册一个同预算、同早停的 A'（固定 sigma=0、100 次）与 B'（learn_psf）对比，以分离“PSF 分支”与“额外优化步数”的贡献；在此之前，B 的改善应视为该预算下学习 PSF 的包络。
