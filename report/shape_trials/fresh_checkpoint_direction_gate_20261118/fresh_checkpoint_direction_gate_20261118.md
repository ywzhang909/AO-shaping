# 新 checkpoint 方向门与混合参数离线判读（2026-11-18）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/direction_fresh_checkpoint_20261118.py`](../../../scripts/slm/direction_fresh_checkpoint_20261118.py)
> **数据/关联脚本**: [`scripts/slm/direction_fresh_checkpoint_zernike_20261118.py`](../../../scripts/slm/direction_fresh_checkpoint_zernike_20261118.py)
> **数据/关联脚本**: [`scripts/slm/build_direction_hybrid_checkpoints_20261118.py`](../../../scripts/slm/build_direction_hybrid_checkpoints_20261118.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/fit.py`](../../../src/ao_shaping/tools/slm/transfer/fit.py)
> **运行环境**: 离线（仅回放归档帧与 checkpoint；未占用 SLM/CCD/激光）
> **说明**: 比较 11/18 fit 在 epoch-matched 11/17 square1000 方向帧上的方向门，并对 11/07 单模屏进行跨 epoch hybrid 诊断及 11/18 强度窗口零步回放；fresh checkpoint 两个 amp 均 0/3 达到 0.50，hybrid 未通过既有每档 10/14 标准，held-out relative L2 0.735。结论：不存在已证明通过方向与绝对强度双门的 checkpoint，不进入硬件闭环验收。输入/输出位于 data/debug/direction_fresh_checkpoint_20261118、data/debug/hybrid_intensity_gate_20261118 与 data/debug/slm_transfer/axis_mismatch_validation_20261118。人工撰写。
<!-- provenance:end -->

## 结论

**没有证据表明存在同时通过方向门和绝对强度保真度门的 checkpoint；本轮不进入物理设备闭环验收。**
SLM #1、Daheng CCD、激光均未被本轮分析占用，未执行设备动作。

11/18 新 fit 在 epoch-matched 的 11/17 square1000 方向帧上仍不预测正确方向：amp03 的 N4/RX/RY Pearson 中位数 −0.003，amp10 为 −0.048，两档均 0/3 达到 0.50。对 11/07 存档的 Zernike 屏做诊断性回放，fresh checkpoint（含 11/18 残差项，residual 显式回放）为 0.491 / 0.538 / 0.445（≥0.50 数 6/14、8/14、5/14）；旧 M3 方向参数加上 11/18 残差项达到 0.504 / 0.568 / 0.493（≥0.50 数 7/14、8/14、7/14）；均未达到既有 10/14 屏幕标准，且 amp10 中位数低于 0.50。这些混合/回放不是可直接部署的真实 epoch-matched checkpoint：残差项来自 11/18，而单模屏采自 11/07。

同一混合参数在 11/18 独立采集集上的零步绝对强度回放也不合格：12 个 held-out 样本平均 CV_pred 2.206、CV_measured 3.832、CV 绝对差 1.626、relative L2 0.735。作为对照，原 11/18 fit 的 held-out 指标为 CV_pred 3.291、CV_measured 3.832、CV 绝对差 0.842、relative L2 0.529；这仍不是目标形状的闭环验收结果。

## 具体回放

### 1. 新鲜方向帧（正式 epoch-matched 方向筛查）

脚本：`scripts/slm/direction_fresh_checkpoint_20261118.py`
结果：`data/debug/direction_fresh_checkpoint_20261118/fresh_checkpoint.json`；日志：`data/debug/direction_fresh_checkpoint_20261118/run.log`。
输入 checkpoint：`data/debug/slm_transfer/axis_mismatch_validation_20261118/fit/frozen_parameters.json`。
输入数据：`data/debug/square1000_model_diag_20261117b_amp03/` 和 `data/debug/square1000_model_diag_20261117b_amp10/`。

口径沿用 `direction_peak_centred_20261117.py`：rate = raw frame mean / exposure_ms，`phase_sign=-1`、`phase_sign_scope="command"`，实测椭圆中心作 crop/origin，128 px crop；新增回放按 checkpoint 中 residual Noll 项显式建模。

| 数据 | N4 | RX | RY | 中位数 | ≥0.50 |
|---|---:|---:|---:|---:|---:|
| amp03 | 0.027 | −0.036 | −0.003 | −0.003 | 0/3 |
| amp10 | −0.048 | −0.073 | 0.000 | −0.048 | 0/3 |

精确值见 JSON。两个新鲜 session 都明显低于 0.50；不是接近门槛的边缘通过。

### 2. 11/07 单模屏混合参数诊断（非正式 checkpoint 验收）

脚本：`scripts/slm/direction_fresh_checkpoint_zernike_20261118.py`。
筛查输入：`data/debug/slm_transfer/issue234_amp02_screen_20261107/`、`issue234_amp05_screen_20261107/`、`issue234_single_mode_screen_20261107/`；总计每档 14 个配对值。
旧 M3 sanity check 复现既有结果：0.618 / 0.693 / 0.671，中位数且每档 10/14 过 0.50。说明 replay 指标与历史口径一致。

| 变体 | 参数构成 | amp 0.02 median / 过门 | amp 0.05 median / 过门 | amp 0.10 median / 过门 |
|---|---|---:|---:|---:|
| fresh checkpoint | 11/18 defocus + envelope + residual | 0.491 / 6/14 | 0.538 / 8/14 | 0.445 / 5/14 |
| old geometry + 11/18 residual（混合诊断） | 11/07 defocus/envelope + 11/18 residual | 0.504 / 7/14 | 0.568 / 8/14 | 0.493 / 7/14 |
| old M3 sanity | 11/07 M3 defocus/envelope | 0.618 / 10/14 | 0.693 / 10/14 | 0.671 / 10/14 |

混合结果提示方向信息主要随旧 defocus/envelope 光学块，而不是 11/18 的全新 envelope/defocus 组合；但残差参数跨采集 epoch，不能把它当作可部署模型的通过证据，也不能替代 fresh 单模屏采集。

> 修正：fresh checkpoint 一行已按 residual 显式回放重跑（`zernike_fresh_residual_amp02_amp05.json`，见复现命令）。此前该行 0.471/0.368/0.398 来自 residual 项被置空的早期回放；启用 11/18 残差项后，amp05 中位数抬至 0.538，但仍只有 8/14 过 0.50，未达 10/14 标准，且 amp10 中位数 0.445 低于 0.50。**方向门判定不变：fresh checkpoint 仍不过门。**

### 3. 绝对强度零步回放

检查脚本入口：`ao_shaping.tools.slm.transfer.fit`，使用现成 `metrics()`、同一 11/18 数据集与 `span=120`、`adaptation=full`、`radiometry=exposure`、`iterations=0`。候选 checkpoint：`data/debug/direction_fresh_checkpoint_20261118/hybrids_320/hybrid_oldgeom_resid.json`；输出：`data/debug/hybrid_intensity_gate_20261118/test_candidate/result.json`。回放不打开任何设备。

| 子集 | n | mean correlation | mean CV_pred | mean CV_measured | mean CV abs error | mean relative L2 |
|---|---:|---:|---:|---:|---:|---:|
| 混合参数 held-out | 12 | 0.665 | 2.206 | 3.832 | 1.626 | 0.735 |
| 原 11/18 fit held-out | 12 | 0.844 | 3.291 | 3.832 | 0.842 | 0.529 |

强度指标只是数据集窗口预测的诊断统计，不等同于目标形状 core CV；不能据此宣称达到用户目标的 core CV ≤ 0.10。

## 设备与验收决策

本轮仅离线读取已有帧与 checkpoint。SLM #1 未占用、Daheng CCD 未占用、激光未占用；无新设备动作，无需设备释放/关闭流程，未执行设备闭环。

**决定：不满足方向门与绝对强度双门，不运行 `shape_transition_timing_20261106.py --execute`，不报告 core CV ≤ 0.10 验收。** 若要推进，下一次最小实验应在当前 epoch 同时采集 fresh 单模 Zernike 屏与绝对强度标定数据，先按硬件纪律做六步光路预检；不得将 11/07 单模屏与 11/18 残差 fit 混作 epoch-matched 证明。

## 复现命令

```powershell
.venv/Scripts/python.exe scripts/slm/direction_fresh_checkpoint_20261118.py `
  --sessions data/debug/square1000_model_diag_20261117b_amp03 data/debug/square1000_model_diag_20261117b_amp10 `
  --checkpoint data/debug/slm_transfer/axis_mismatch_validation_20261118/fit/frozen_parameters.json `
  --out data/debug/direction_fresh_checkpoint_20261118/fresh_checkpoint.json --center-method ellipse

# fresh checkpoint 在 11/07 单模屏上（residual 显式回放）
.venv/Scripts/python.exe scripts/slm/direction_fresh_checkpoint_zernike_20261118.py `
  --runs data/debug/slm_transfer/issue234_amp02_screen_20261107 data/debug/slm_transfer/issue234_amp05_screen_20261107 data/debug/slm_transfer/issue234_single_mode_screen_20261107 `
  --checkpoint data/debug/slm_transfer/axis_mismatch_validation_20261118/fit/frozen_parameters.json `
  --out data/debug/direction_fresh_checkpoint_20261118/zernike_fresh_residual_amp02_amp05.json

# old 几何 + 11/18 residual 混合诊断
.venv/Scripts/python.exe scripts/slm/direction_fresh_checkpoint_zernike_20261118.py `
  --runs data/debug/slm_transfer/issue234_amp02_screen_20261107 data/debug/slm_transfer/issue234_amp05_screen_20261107 data/debug/slm_transfer/issue234_single_mode_screen_20261107 `
  --checkpoint data/debug/direction_fresh_checkpoint_20261118/hybrids/hybrid_oldgeom_resid.json `
  --out data/debug/direction_fresh_checkpoint_20261118/hybrids/hybrid_oldgeom_resid.gate.json

.venv/Scripts/python.exe -m ao_shaping.tools.slm.transfer.fit `
  --dataset data/debug/slm_transfer/axis_mismatch_validation_20261118/capture `
  --checkpoint data/debug/direction_fresh_checkpoint_20261118/hybrids_320/hybrid_oldgeom_resid.json `
  --out data/debug/hybrid_intensity_gate_20261118/test_candidate `
  --iterations 0 --span 120 --adaptation full --radiometry exposure --device cuda
```

以上程序退出码均为 0；初次试跑中出现的临时错误（center-method choices、Agreement 对象取值、empty residual basis）已修正或通过提供真实 residual 项规避；不影响最终上述成功回放。
