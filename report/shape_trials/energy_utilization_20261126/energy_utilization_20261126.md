# 整形对衍射效率的利用：历史口径复算与闭环实测判定图文报告 (2026-11-26)

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线分析（复用实验系统已采集 JSON/CSV）+ 本地渲染图（未入库, 见链接债务登记）
> **说明**: 能量利用率修订报告 (P32 四档 lr sweep 追加页, 2d9b1dea); 29 张引用图为编写期本地渲染未入库, 已在 test_report_provenance._PREEXISTING_BROKEN 挂账 (第 190 批)
<!-- provenance:end -->

## 摘要

本报告回答一个问题：**SLM 整形（含倾斜/闪耀）到底能从输入光里拿到多少可用能量，两种闭环整形方案谁更均匀。**

核心口径在 2026-11-26 统一为 **从平相位 0 级转移走的能量份额** `η = E_blaze / F0`：
`F0` 是平相位帧零级窗口的暗减积分，`E_blaze` 是闪耀帧闪耀级窗口的暗减积分，
不再用 `lobe.ratios[blaze_order]`（分母是闪耀帧三衍射级窗口之和，会抬高读数）。

三组历史数据 + 一组新判定给出如下结论：

1. **倾斜探针口径**：唯一合格探针 `probe_real_20261123_gate` 的 `η = 19.27%`；
   旧口径闪耀级份额读成 39.78%，是分母不同造成的约 2 倍虚高。其余两支探针分别
   饱和、±1 级能量为 0，不能作效率判据。
2. **同次效率匹配口径**：倾斜 `η_d = 54.21%`、整形 `η_s = −12.46%`（目标方框内反而比
   平相位更少）。但该次 `verdict = undeterminable`（能量检查越界、G1 组内 SD 超限、
   漂移轮次 > 1），**只能当趋势参考，不作判定**。
3. **方形整形 PIB 随方框边长单调增长**（整帧口径 43.53% → 96.88%），
   PIB 与 η 分母不同，**不能互相代入**。
4. **A/B–ABBA 闭环实测判定**：在 0.12 ms 非饱和、同曝光、单变量配对下，
   `model640` (A) vs `refit_forward` (B) 的 `paired_decision(metric="cv")` 给出
   CV 改善 `0.062261` > 漂移 `0.035982`，判 **`accepted=True`** → 采用 `refit_forward`。

另附七张实拍证据图（§6）：同次效率匹配的平相位/倾斜/整形三态远场、沿 row 跨过
0 级与 ±1 级的亮度剖面、三态横向（col）剖面、三口径窗口在剖面上的位置、
ABBA 四帧与 row/col 双向剖面，以及 20261023 方斑边长扫描的远场拼图。
它们只展示原始采集帧的亮度分布，不改变上述任何判定，也不替代窗口积分口径。
合格探针与方斑扫描的原始帧未归档，这两类口径不补剖面；数据边界见 §6.4。

另在 §9 汇总上周（11-16—11-22）与本周（11-23—11-26）的全部工作主题，并附 17 张
来自各原报告的归档图副本（图 W1–W17），使本周重点方向放在两周时间线上可检索。
§9 只读既有归档，不改动上游报告，也不重复采集。

全程只读归档数据，不重新采集、不触发设备动作；报告的输入契约与可复算命令见末章。

## §0 当前现状与最优路线

- **现状**：效率口径已从"闪耀帧三衍射级窗口占比"纠正为"平相位 0 级可转移份额"；页面与后端已按
  `η = E_blaze / F0` 实现并有 18 项测试锁定。历史趋势显示倾斜可转移约 19.27%（探针合格口径），
  闭环整形的同次效率净值仍为负且判定不可用。
- **最优路线**：先以非饱和、同曝光、ABBA 配对锁定闭环方案（`refit_forward`），再谈利用率数值；
  不同分母的指标分开报告、互不代入；效率匹配实验须先满足能量检查/SD/漂移三项门槛，
  否则只给趋势不给结论。
- **本周重点方向**：闭环整形均匀度与能量利用率口径的统一与可复现判定；动态整形方向的
  实测判定方法在本报告确立的配对框架上延续。

## 任务项

| 项 | 状态 | 说明 |
|---|---|---|
| 效率新口径 `η = E_blaze / F0` 定义与实现 | 完成 | `src/ao_shaping/gui/slm/closed_loop_shaping.py`，18 项测试锁定 |
| 历史三口径只读复算脚本 | 完成 | `scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py`，支持 `--json` |
| 倾斜探针效率（3 支） | 完成 | 1 支合格、1 支饱和、1 支 ±1 级为 0，见 §2 |
| 同次效率匹配（倾斜 vs 整形） | 完成（趋势） | `verdict=undeterminable`，不作判定，见 §3 |
| 方形整形 PIB 扫描 | 完成 | 两组曝光，整帧口径，见 §4 |
| A/B–ABBA 闭环实测判定 | 完成 | `paired_decision(cv)` `accepted=True`，见 §5 |
| 本报告与 5 张图 | 完成 | `report/shape_trials/energy_utilization_20261126/` |
| 实拍证据图（4 张） | 完成 | `artifacts/export_real_captures.py` 导出 3 张；方斑远场拼图选自既有归档 |
| 双周工作总览（§9，图 W1–W17） | 完成 | 上周 10 主题 + 本周 12 主题；图片为原报告副本，原图未改动 |
| 报告溯源登记 | **挂起（让渡）** | `scripts/_common/provenance.py` 被另一车道整体改写，按整文件让渡规程不碰；待其落地后由持有车道登记 |
| 页面改动合入 | **挂起（让渡）** | `closed_loop_shaping_page.py` / `test_closed_loop_shaping_page.py` hunk 交错，让渡待合 |

## §1 指标定义与口径边界

### 1.1 SLM 衍射效率（新定义）

从平相位 0 级可以转移走的能量份额：

```
η = E_blaze / F0
F0      = relative_efficiency.baseline_zero_window_counts   # 平相位帧零级窗口暗减积分
E_blaze = lobe.energies[blaze_order]                         # 闪耀帧闪耀级窗口暗减积分
```

- 闪耀级取 `relative_efficiency.blaze_order`（新）/ `lobe.orders.blaze_order`（旧）；
  两者都缺时取 ±1 级中较强者。
- 探针自带 `relative_efficiency.blaze_energy_share` 时直接采信（探针已按同一公式计算），
  并仍用上式交叉校验。
- **不使用** `lobe.ratios[blaze_order]`：其分母是闪耀帧三衍射级窗口之和 `S3 = E0 + E+1 + E−1`，
  同一探针读数 39.78% vs 19.27%，虚高约 2 倍。

### 1.2 三种口径的分母（不可互相代入）

| 指标 | 分子 | 分母 | 用途 |
|---|---|---|---|
| `η`（本报告主口径） | 闪耀级窗口暗减积分 `E_blaze` | 平相位零级窗口 `F0` | 可转移能量份额 |
| 旧口径闪耀级份额 | 闪耀级窗口能量 | 闪耀帧三衍射级窗口之和 `S3` | 帧内能量分配，诊断用 |
| `PIB` | 目标方框内能量 | CCD 整帧能量 | 整形集中度，整帧口径 |

`η_d`（倾斜）与 `η_s`（整形）在效率匹配实验里共用同一 `F0` 圆盘口径，是唯一可直接相比的
一组；PIB 与 η 分母不同，不得相除或互换。

![图1 新口径与旧口径对比](figures/fig1_eta_compare.png)

*图 1：同一探针在新旧口径下的读数；横轴为探测实验参数（周期 16 px / x 轴 / 灰度深度 993 /
有效曝光），不是日期。口径差异本身就会造成约 2 倍读数变化；饱和/未合格探针不具判据资格。*

**采集标注**：采集 2026-11-23 12:53:11（本地 UTC+8）；参数：曝光请求 0.1 ms / 生效 0.02 ms（自动曝光，峰值目标 0.45）、3 帧、稳定 0.25 s、灰度深度 993、光栅周期 16 px、x 轴、间距 471 px、暗减、面板 1200×1920；来源：连续闪耀光栅衍射效率探针（`src/ao_shaping/tools/slm/calibration/diffraction_efficiency/measure.py`；明细 `data/diffraction_efficiency/probe_real_20261123_gate/summary.json`）。

## §2 倾斜（闪耀光栅）：探针口径

唯一合格探针 `probe_real_20261123_gate`（有效曝光 0.02 ms，级次间距 471 px，灰度深度 993，
闪耀级 `minus_one`）给出：

| 探针 | F0 (counts) | E_blaze (counts) | η (F0 口径) | 零级残余/F0 | 三衍射级窗口未计入/F0 | 旧口径 `ratios` | 资格 |
|---|---|---|---|---|---|---|---|
| `probe_real_20261123_gate` | 16349.7 | 3151.3 | **19.27%** | 29.18% | 51.54% | 39.78% | 合格 |
| `probe_real_20261121` | 291215.5 | 4735.5 | 1.63% | 76.00% | 20.96% | 2.06% | 饱和（两帧峰值 255） |
| `probe_real_20261121_0p1ms` | 32172.9 | 0.0 | 0.00% | 28.58% | 71.42% | 0.00% | ±1 级能量全为 0 |

- 页面参照值取 `probe_real_20261123_gate`：**η = 19.27%**。
- 合格探针能量去向分解见下图：可转移 19.27%，零级残余 29.18%，三衍射级窗口未计入 51.54%
  （窗口截断、无几何标定，非绝对衍射效率）。

![图2 F0 能量分解](figures/fig2_f0_breakdown.png)

*图 2：合格探针 `probe_real_20261123_gate` 的 F0 能量去向。三衍射级窗口未计入占比最高，
说明"可转移份额"与"帧内占比"是完全不同的问题。*

**采集标注**：采集 2026-11-23 12:53:11（本地 UTC+8）；参数：曝光请求 0.1 ms / 生效 0.02 ms（自动曝光，峰值目标 0.45）、3 帧、稳定 0.25 s、灰度深度 993、光栅周期 16 px、x 轴、间距 471 px、暗减、面板 1200×1920；来源：连续闪耀光栅衍射效率探针（`src/ao_shaping/tools/slm/calibration/diffraction_efficiency/measure.py`；明细 `data/diffraction_efficiency/probe_real_20261123_gate/summary.json`）。

## §3 同次效率匹配：倾斜 vs 整形

数据源 `data/debug/efficiency_match/20261123_050735Z/analysis.json`，曝光 0.04 ms，
4 轮，同一 F0 圆盘口径：

| 量 | 值 | 说明 |
|---|---|---|
| `η_d`（倾斜，G1/F0） | **54.21%** | σ=0.0106（4 轮） |
| `η_s`（整形，(S−S0)/F0） | **−12.46%** | σ=0.0552；目标方框内比平相位更少 |
| `d = η_d − η_s` 均值 | −0.6667 | 95% CI [−0.7068, −0.6388] |
| `F0_mean` / `S0_mean` | 649699.25 / 631990.06 | 暗减计数 |
| `flat_box_over_f0` | 0.9727 | 平相位方框内/F0 |
| `verdict` | **undeterminable** | 不作判定 |

`undeterminable` 的三条原因来自脚本判据，逐条列出：
`out_of_range_energy_check_background_or_saturation`（能量检查越界）、
`g1_group_sd_exceeds_limit`（G1 组内 SD 超限）、`drift_rounds_gt_1`（漂移轮次 > 1）。

![图3 效率匹配口径对比](figures/fig3_efficiency_match.png)

*图 3：倾斜与整形在同一 F0 圆盘口径下的净值。整形为负、判定不可用，因此只作趋势参考，
不构成"整形优于倾斜"或反之的结论。*

**采集标注**：采集 2026-11-23 13:07:42–13:08:11（分析 13:19:36）；参数：曝光 0.04 ms、16 帧矩阵（平相位 6 / 倾斜 4 / 整形 4 / 暗场 2）、面板 1200×1920、相机满量程 4095、暗场库 `dark_library_0.04ms.npy`；来源：GS 方斑相位 + 连续闪耀光栅（`scripts/slm/efficiency_match_capture_20261123.py`、`efficiency_match_analyze_20261123.py`）。

## §4 方形整形 PIB 随方框尺寸增长

PIB = 目标方框内能量 / CCD 整帧能量，整帧口径，不能与 η 相除。
两组历史扫描：`20261023_gs_square`（20 ms）与 `20261023_atomic_full`（180 ms）。

![图4 PIB 随边长增长](figures/fig4_pib.png)

*图 4：(a) PIB 随目标方框边长单调增长，小边长处两组曝光均出现饱和点（红圈标出）；
(b) PIB 与 0 级光斑能量的量级对照：器件 0 级地板 3.03%（P=32，三衍射级窗口归一）、ramp 993 地板
4.16%、合格探针零级残余 29.18%（F0 分母）。三者与 PIB 的整帧分母不同，仅作量级对照，
不可相减或相乘；方形扫描 CSV 无逐尺寸 0 级能量列，不做逐点配对。*

**采集标注**：采集 2026-10-23（批次 `20261023_gs_square`）；参数：曝光 20 ms、边长 100–1800 px、原始帧 2592×1944；来源：GS 方斑整形扫描（`scripts/slm/hw_gs_square_scans.py`），选帧 `frames_selection.json`；原始 npz 未随归档保留，图由归档拼图复制。

## §5 A/B–ABBA 闭环实测判定

判定回答的是**哪个闭环方案更均匀**，与前面的历史利用率趋势相互独立。

- 变体 A `model640`（高清冻结基线）：`ab_validate_base/square_best.npy`；
  将控制网格升至 640×640、FFT 2048，光学与辐射参数逐位继承，不重拟合，用作对照。
- 变体 B `refit_forward`（重拟合候选）：`ab_validate_refit/square_best.npy`；
  从 A 热启动，架构、网格、FFT 与推理口径不变，仅在训练集 `adaptation=full` 重拟合物理参数（≤100 迭代、最佳 27）。
- 序列 A→B→B→A；曝光请求/生效均 0.12 ms（非饱和）；bit depth 12；
  冻结 mask `abba_core_mask_camera.npz`（1944×2592，1369 True，由 core+camera_xy 派生）。
- 四帧 CV / 峰值：

| ABBA 帧序 | 变体 | CV | 均值 | 峰值 | 饱和 |
|---|---|---|---|---|---|
| 0（基线 A 首帧） | A | 1.897016 | 150.38 | 2059 | 否 |
| 1（候选 B 第 1 帧） | B | 1.862118 | 151.28 | 2091 | 否 |
| 2（候选 B 第 2 帧） | B | 1.843373 | 148.22 | 2019 | 否 |
| 3（基线 A 末帧） | A | 1.932998 | 152.11 | 2138 | 否 |

`paired_decision(metric="cv")` 结果（严格默认 `rel_tol=0, abs_tol=0`）

| 量 | 值 |
|---|---|
| 高清冻结基线 A（均值 CV） | 1.915007 |
| 重拟合候选 B（均值 CV） | 1.852745 |
| `drift`（基线 A 首末帧配对差） | 0.035982 |
| `tolerance` | 0.0 |
| 改善（基线 A − 候选 B） | 0.062261 |
| 判据 | 改善 `0.062261` > 漂移 `0.035982` + 容差 `0.0` |
| **`accepted`** | **True → 采用 `refit_forward`** |

![图5 ABBA 四帧 CV 与判定](figures/fig5_abba_cv.png)

*图 5：A/B–ABBA 四帧 CV（帧序 `0_A`→`1_B`→`2_B`→`3_A`，即基线 A→候选 B→候选 B→基线 A）。变体 B 两帧均低于变体 A；改善大于基线 A 首末帧漂移，判 `accepted=True`。*

**采集标注**：采集 2026-11-26 19:59:30–19:59:33（设备 19:59:27 开、19:59:35 释放）；参数：曝光 0.12 ms、12 位、稳定等待 120 s、序列 A→B→B→A、CV 掩膜 `abba_core_mask_camera.npz`；来源：冻结物理预测模型基线 A（`model640`，相位 sha256 2907ef3b…）与重拟合物理预测模型 B（`refit_forward`，相位 sha256 946e6196…），`scripts/slm/replay_closed_loop_abba_20261124.py`。

首轮 ABBA 因驱动默认 20 ms 导致峰值 4095 全饱和，判定作废；回放脚本新增 `--exposure-ms`，
用 `set_camera_exposure_ms` 下发并回读生效值 0.12 ms 后重跑于新目录。设备占用与释放：
相机开 19:59:27、SLM 开 19:59:28、19:59:35 释放，`released.json` 记
`camera_closed=true` / `slm_closed=true`，设备正常关闭，当前无占用。

## §6 实拍证据（原始 CCD 帧）

本节展示的图来自归档原始采集帧，不是示意绘制；每张图的来源、处理与边界都单独说明。
它们用于核对"图 3 的 η_d/η_s 差异来自哪里""图 5 的 CV 差异是否可见""图 4 的 PIB 随边长增长是否对应真实远场"，
不用于重新计算任何数值，也不改变 §2–§5 的判定。


### 6.0 实测图采集元数据总表

下表给出本节实测图的三要素标注；图注中的「采集标注」为同一元数据的简写。
时间均为本地时区（UTC+8）。图 W1–W17 是跨报告归档副本，采集时间/参数/来源算法以 §8.1 所引原报告为准，不在本报告重复标注。

| 图 | 采集时间 | 设置参数 | 来源算法（脚本 / 检查点） |
|---|---|---|---|
| 图 1、2（探针口径） | 2026-11-23 12:53:11 | SLM #1（1064 nm）、大恒相机 #0；请求曝光 0.1 ms / 生效 0.02 ms（自动曝光）；3 帧、稳定 0.25 s；灰度深度 993；周期 16 px、x 轴、间距 471 px、闪耀级 −1；暗减；面板 1200×1920 | 连续闪耀光栅衍射效率探针，`src/ao_shaping/tools/slm/calibration/diffraction_efficiency/measure.py` |
| 图 3、6、9、10、11（同次效率匹配） | 2026-11-23 13:07:42–13:08:11（分析 13:19:36） | 曝光 0.04 ms；16 帧矩阵（平相位 6 / 倾斜 4 / 整形 4 / 暗场 2）；面板 1200×1920；满量程 4095；暗场库 `dark_library_0.04ms.npy` | GS 方斑相位 + 连续闪耀光栅，`scripts/slm/efficiency_match_capture_20261123.py`、`efficiency_match_analyze_20261123.py` |
| 图 4、8（方斑能量集中度） | 2026-10-23（批次 `20261023_gs_square`） | 曝光 20 ms；边长 100–1800 px；原始帧 2592×1944 | GS 方斑整形扫描，`scripts/slm/hw_gs_square_scans.py`；选帧 `report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/frames_selection.json` |
| 图 5、7、12（A/B–ABBA） | 2026-11-26 19:59:30–19:59:33（设备 19:59:27 开、19:59:35 释放） | 曝光 0.12 ms、12 位；稳定等待 120 s；序列 A→B→B→A；CV 掩膜 `abba_core_mask_camera.npz`；四帧峰值 2059–2138，未饱和 | 冻结物理预测模型基线 A（`model640`，相位 sha256 2907ef3b…）vs 重拟合物理预测模型 B（`refit_forward`，相位 sha256 946e6196…），`scripts/slm/replay_closed_loop_abba_20261124.py` |

### 6.1 三态远场：平相位 / 倾斜 / 整形

![图6 同次效率匹配三态实拍](figures/fig6_real_efficiency_match.png)

*图 6：`data/debug/efficiency_match/20261123_050735Z/raw/` 的 `0000_flat_pre.npy`、
`0001_blaze_grating.npy`、`0002_shaping_target.npy`，用同目录 `dark_library_0.04ms.npy` 暗减，
统一裁剪到中心 `(row≈1105, col≈1227)` 附近并共用 1–2100 counts 对数色标；
青色竖线是图 9 的剖面取样列 `col=1230`。倾斜帧可见 0 级与 +1 级两个亮斑；
整形帧能量被摊开、中心峰变弱，与 §3 的 `η_d=54.21%` / `η_s=−12.46%` 趋势一致。
该次 `verdict=undeterminable`，图只作可视核对。*

**采集标注**：采集 2026-11-23 13:07:42–13:08:11（分析 13:19:36）；参数：曝光 0.04 ms、16 帧矩阵（平相位 6 / 倾斜 4 / 整形 4 / 暗场 2）、面板 1200×1920、相机满量程 4095、暗场库 `dark_library_0.04ms.npy`；来源：GS 方斑相位 + 连续闪耀光栅（`scripts/slm/efficiency_match_capture_20261123.py`、`efficiency_match_analyze_20261123.py`）。

#### 6.1.1 跨 0 级 / ±1 级的亮度剖面

![图9 跨级次亮度剖面](figures/fig9_real_lobe_profile.png)

*图 9：沿图 6 青色竖线，在 `col=1226–1234` 九列平均、相机 `row=1000–1200` 上的暗减后亮度剖面；
三帧同次 0.04 ms 曝光、未归一化，剖面峰高不是窗口积分能量。级次坐标取自 `analysis.json`：
`frames[1].plus1_peak=[1047, 1230]`、`frames[1].minus1_peak=[1152, 1230]`、
`frames[0].center[0]=1105.51`。平相位 0 级剖面峰约 1.7k counts；倾斜帧 0 级降到约 0.85k，
并在 `row=1047` 出现约 0.52k 的独立峰（本次闪耀级为 +1 级）；−1 级仅弱残余（窗口峰值约 0.13k），
所以 `row=1152` 处没有尖峰；整形帧中心峰约 1.0k、没有分立一级峰，两翼高于平相位。
剖面直观展示能量去向，η 数值仍以 §3 的窗口积分口径为准。*

**采集标注**：采集 2026-11-23 13:07:42–13:08:11（分析 13:19:36）；参数：曝光 0.04 ms、16 帧矩阵（平相位 6 / 倾斜 4 / 整形 4 / 暗场 2）、面板 1200×1920、相机满量程 4095、暗场库 `dark_library_0.04ms.npy`；来源：GS 方斑相位 + 连续闪耀光栅（`scripts/slm/efficiency_match_capture_20261123.py`、`efficiency_match_analyze_20261123.py`）。

#### 6.1.2 横向（col）剖面：圆盘口径与方框口径

![图10 三态横向亮度剖面](figures/fig10_real_col_profile.png)

*图 10：与图 9 正交的方向——在 `row=1102–1108` 七行平均，横轴为相机 `col=1080–1380`；
同一批三态帧、同一暗减、同次 0.04 ms 曝光、未归一化。0 级中心 `col≈1227.18` 取自
`analysis.json` 的 `frames[0].center[1]`。淡蓝带是 `η` 分母 F0 圆盘半径 `r0=48 px`
在横向的穿越区间 `1179–1275`；淡绿带是目标方框 `side=61.2 px` 的区间 `1196–1258`，
对应 `η_s` 分子与 PIB 方框口径。三态横向峰高为平相位约 1.9k、倾斜约 0.96k、
整形约 1.12k counts；横向没有分开的 ±1 级峰，因为本次级次主轴沿 row。
窗口参数取自 `thresholds`，剖面穿越区间只是二维窗口的一维投影，不替代窗口积分。*

**采集标注**：采集 2026-11-23 13:07:42–13:08:11（分析 13:19:36）；参数：曝光 0.04 ms、16 帧矩阵（平相位 6 / 倾斜 4 / 整形 4 / 暗场 2）、面板 1200×1920、相机满量程 4095、暗场库 `dark_library_0.04ms.npy`；来源：GS 方斑相位 + 连续闪耀光栅（`scripts/slm/efficiency_match_capture_20261123.py`、`efficiency_match_analyze_20261123.py`）。

#### 6.1.3 三口径窗口在同一剖面上的位置

![图11 三口径窗口穿越剖面](figures/fig11_real_metric_windows.png)

*图 11：把 §1.2 三种口径的窗口画回同一剖面。左：平相位帧 `F0` 圆盘（`r0=48`，
`η` 分母）；中：倾斜帧 `G1(+1)` / `GM1(−1)` 圆盘（`r1=60`，旧口径闪耀级份额的三衍射级窗口）
与 `G0` 圆盘（以倾斜帧插值中心 `row≈1103.94` 为心），`η` 分子取 `G1`；
右：整形帧目标方框（`side=61.2`，以整形帧插值中心为心），对应 `η_s` 分子 / PIB 方框。
竖线是窗口边界，不是积分结果；具体数值仍取 `analysis.json` 的 `f0/g0/g1/gm1/s`。
图 3 的 `η_d=54.21%` 与旧口径约 2 倍的差异，根源就在这些分母窗口不同。*

**采集标注**：采集 2026-11-23 13:07:42–13:08:11（分析 13:19:36）；参数：曝光 0.04 ms、16 帧矩阵（平相位 6 / 倾斜 4 / 整形 4 / 暗场 2）、面板 1200×1920、相机满量程 4095、暗场库 `dark_library_0.04ms.npy`；来源：GS 方斑相位 + 连续闪耀光栅（`scripts/slm/efficiency_match_capture_20261123.py`、`efficiency_match_analyze_20261123.py`）。

### 6.2 ABBA 四帧的均匀度差异

![图7 ABBA 四帧实拍](figures/fig7_real_abba_frames.png)

*图 7：`ab_replay_square_retry2/captures/` 的 ABBA 配对帧 `0_A.npz`（基线 A 首帧）、`1_B.npz`（候选 B 第 1 帧）、`2_B.npz`（候选 B 第 2 帧）、`3_A.npz`（基线 A 末帧），
0.12 ms 非饱和，四帧统一 1–2200 counts 对数色标；青色虚线是 `abba_core_mask_camera.npz`
中 1369 个 True 像素的外接框（CV 统计窗口）。变体 B 两帧的环带旁瓣更弱，
与 §5 的 CV 改善 `0.062261` 方向一致；判定仍以 `paired_decision(metric="cv")` 为准。*

**采集标注**：采集 2026-11-26 19:59:30–19:59:33（设备 19:59:27 开、19:59:35 释放）；参数：曝光 0.12 ms、12 位、稳定等待 120 s、序列 A→B→B→A、CV 掩膜 `abba_core_mask_camera.npz`；来源：冻结物理预测模型基线 A（`model640`，相位 sha256 2907ef3b…）与重拟合物理预测模型 B（`refit_forward`，相位 sha256 946e6196…），`scripts/slm/replay_closed_loop_abba_20261124.py`。

![图12 ABBA 四帧双向亮度剖面](figures/fig12_real_abba_profiles.png)

*图 12：ABBA 四帧在 row（左，`col=1224–1231` 七列平均）与 col（右，`row=1101–1108`
七行平均）两个正交方向的亮度剖面；青色带与灰虚线是 `abba_core_mask_camera.npz`
中 CV 掩膜的外接框区间（row `1068–1140`、col `1191–1263`）。四帧未暗减、同次 0.12 ms
曝光、未归一化。双向剖面显示 B 帧中心区略低、能量更分散，与 CV 改善方向一致；
row/col 剖面只是可视核对，均匀度判定仍以 `paired_decision(metric="cv")` 为准。*

**采集标注**：采集 2026-11-26 19:59:30–19:59:33（设备 19:59:27 开、19:59:35 释放）；参数：曝光 0.12 ms、12 位、稳定等待 120 s、序列 A→B→B→A、CV 掩膜 `abba_core_mask_camera.npz`；来源：冻结物理预测模型基线 A（`model640`，相位 sha256 2907ef3b…）与重拟合物理预测模型 B（`refit_forward`，相位 sha256 946e6196…），`scripts/slm/replay_closed_loop_abba_20261124.py`。

### 6.3 方斑边长扫描的远场图

![图8 方斑远场实拍拼图](figures/fig8_real_square_shaping.png)

*图 8：选自 `report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/` 的
`fig6_farfield_sides_20261023_gs_square.png`，由同批次原始远场帧重绘，选帧依据
`frames_selection.json`（原始 npz 未随归档保留）。边长 200/600/1000/1500/1800 px 对应
PIB 0.548/0.691/0.802/0.924/0.969，展示图 4 趋势对应的真实光斑形态；
PIB 仍是整帧口径，不等于 η。*

**采集标注**：采集 2026-10-23（批次 `20261023_gs_square`）；参数：曝光 20 ms、边长 100–1800 px、原始帧 2592×1944；来源：GS 方斑整形扫描（`scripts/slm/hw_gs_square_scans.py`），选帧 `frames_selection.json`；原始 npz 未随归档保留，图由归档拼图复制。

### 6.4 剖面数据的边界：哪些口径能补剖面，哪些不能

亮度剖面只能从保留下来的原始帧导出，不能由汇总值或归档拼图反推。本轮按数据可用性划分：

| 口径 | 原始帧 | 剖面 | 说明 |
|---|---|---|---|
| 效率匹配三态（§6.1 / 图 9–11） | 有（`raw/*.npy`） | 已补 row / col 双向 + 三口径窗口 | 同次 0.04 ms，暗减后 |
| ABBA 四帧（§6.2 / 图 12） | 有（`captures/*.npz`） | 已补 row / col 双向 | 0.12 ms 非饱和，未暗减 |
| 合格探针 `probe_real_20261123_gate` | **无**（仅 `summary.json` / `summary.csv`） | 不补 | 只有汇总值，逻辑上无法生成剖面 |
| PIB 方斑扫描（§6.3 / 图 8） | **无**（原 npz 未归档，仅归档拼图） | 不补 | 拼图是重绘结果，不构成原始帧 |
| 两支非合格探针 `probe_real_20261121*` | 有，但一支 5 ms 全饱和、一支 ±1 级能量全为 0 | 不补 | 口径不合格，剖面无判读价值 |

因此"不同指标口径"和"不同取样方向"的剖面只在数据完备的两类口径上补齐；
合格探针与 PIB 不伪造剖面，相关结论仍以 §3 的窗口积分与 §5 的 ABBA 判定为准。

## §7 结论

1. **口径先于数值**：同一实验，旧口径 39.78% vs 新口径 19.27%；谈效率必须先固定分母。
   主口径 `η = E_blaze / F0` 衡量的是"从平相位 0 级可转移走的份额"。
2. **倾斜比整形更"能拿光"是趋势，不是判定**：探针口径 19.27%、效率匹配口径 54.21% 的
   倾斜均高于整形同次净值 −12.46%；但效率匹配 `verdict=undeterminable`，只作参考。
3. **闭环方案判定独立成立**：ABBA 配对在 0.12 ms 非饱和、同曝光下给出 `accepted=True`，
   采用 `refit_forward`；改的是均匀度，不改写 §2–§4 的利用率趋势结论。
4. **PIB 不可当效率**：PIB 是整帧口径的集中度指标，随方框尺寸增长，与 η 分母不同。

## §8 数据来源与可复现

| 内容 | 真源 | 复算命令 |
|---|---|---|
| 历史三口径（探针/效率匹配/PIB） | `scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py --json` | `.venv\Scripts\python.exe scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py` |
| 探针明细 | `data/diffraction_efficiency/probe_real_20261123_gate/summary.json` | — |
| 效率匹配明细 | `data/debug/efficiency_match/20261123_050735Z/analysis.json` | — |
| ABBA 四帧与判定 | `data/debug/slm_transfer/ab_session_20261124/ab_replay_square_retry2/summary.json` | `paired_decision(*stages, metric="cv")` |
| 实拍图 6/9/10/11 的原始帧（效率匹配三态） | `data/debug/efficiency_match/20261123_050735Z/raw/` | `python artifacts/export_real_captures.py` |
| 实拍图 7/12 的原始帧（ABBA 四帧） | `data/debug/slm_transfer/ab_session_20261124/ab_replay_square_retry2/captures/` | `python artifacts/export_real_captures.py` |
| 图 7/12 的 CV 窗口 | `data/debug/slm_transfer/ab_session_20261124/abba_core_mask_camera.npz` | — |
| 实拍图 8 的选帧依据 | `report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/frames_selection.json` | 原 npz 未保留，图由该批次归档远场拼图复制 |
| 本报告结构化数据 | `artifacts/data.json` | `python artifacts/make_figures.py`（需 `matplotlib`） |
| 报告上级（口径/任务/提交状态） | `../gui_closed_loop_energy_efficiency_20261125.md` | — |

图 1–5 由 `artifacts/make_figures.py` 从 `artifacts/data.json` 生成；图 6/7/9/10/11/12 由
`artifacts/export_real_captures.py` 从上述原始帧导出；图 8 复制自既有方斑远场归档。
剖面窗口参数取自 `analysis.json` 的 `thresholds` 与各帧 `reference_center`，无手工估算。
归档原始帧与设备原图未改动；剖面数据边界见 §6.4。

### 8.1 图 W1–W17 的副本来源

按 `docs/dev/report_asset_rules.md` 的「跨报告复用 → 复制」规则，双周图片均复制到本报告
`figures/bw_wNN_*.png`，原报告与原图不改动；下表给出原始路径，可按原报告的 provenance 复算。采集时间、设置参数与来源算法见各原报告；本报告不重复标注。

| 图 | 本报告副本 | 原图 |
|---|---|---|
| W1 | `figures/bw_w01_speed_fps.png` | `report/shape_trials/figures/speed_decouple_20261120_fps.png` |
| W2 | `figures/bw_w02_speed_timing.png` | `report/shape_trials/figures/speed_decouple_20261120_timing.png` |
| W3 | `figures/bw_w03_ideal_overview.png` | `report/shape_trials/ideal_path_static_test_20261117/figures/overview.png` |
| W4 | `figures/bw_w04_ideal_lens.png` | `report/shape_trials/ideal_path_static_test_20261117/figures/compare_lens_f125.png` |
| W5 | `figures/bw_w05_online_round.png` | `report/shape_trials/transfer_online_params/figures/online_params_by_round.png` |
| W6 | `figures/bw_w06_online_iteration.png` | `report/shape_trials/transfer_online_params/figures/online_params_by_iteration.png` |
| W7 | `figures/bw_w07_s1e_arm_a.png` | `report/shape_trials/s1e_same_bytes_probe_20261126/s1e_same_bytes_probe_20261126_fig1.png` |
| W8 | `figures/bw_w08_s1e_arm_p.png` | `report/shape_trials/s1e_same_bytes_probe_20261126/s1e_arm_p_20261126_fig1.png` |
| W9 | `figures/bw_w09_psf_validation.png` | `report/shape_trials/transfer_psf_adaptation_20261124/fig1_validation_relative_l2.png` |
| W10 | `figures/bw_w10_psf_independent.png` | `report/shape_trials/transfer_psf_adaptation_20261124/fig2_independent_relative_l2.png` |
| W11 | `figures/bw_w11_geometry_validation.png` | `report/shape_trials/transfer_geometry_adaptation_20261124/fig1_validation_relative_l2.png` |
| W12 | `figures/bw_w12_geometry_holdout.png` | `report/shape_trials/transfer_geometry_adaptation_20261124/fig2_holdout_displacement_rmse.png` |
| W13 | `figures/bw_w13_hmws_linearity.png` | `report/shape_trials/hmws_zernike_probe/figures/linearity.png` |
| W14 | `figures/bw_w14_hmws_focus.png` | `report/shape_trials/hmws_zernike_probe/figures/focus_sweep.png` |
| W15 | `figures/bw_w15_tsne_anchored.png` | `report/shape_trials/closed_loop_speed_tsne_20261125/anchored_phase_tsne.png` |
| W16 | `figures/bw_w16_tsne_fit.png` | `report/shape_trials/closed_loop_speed_tsne_20261125/fit_zernike_tsne.png` |
| W17 | `figures/bw_w17_kaxis.png` | `report/slm/differentiable_opt/artifacts/kaxis_screening_20261123/kaxis_cv_vs_iter_20261123.png` |

## §9 双周工作总览（2026-11-16—2026-11-26）

本节把上周与本周已归档的工作汇总到同一份图文报告，便于按时间线检索。图号 W1–W17 只在
本节使用，不与正文图 1–12 混编；设备占用与释放以各原始报告的「运行环境」条目为准。
本节只读归档文件，未占用 SLM/CCD/激光；没有可用归档图的主题只在表中给结论，不补画示意图。

### 9.1 上周（11-16—11-22）：模型假设被连续证伪，速度瓶颈定位

| 方向 | 结论 | 关键数值（原样） | 证据 |
|---|---|---|---|
| M3 锚定闭环 / 绝对强度保真度 | 方向一致性达到判据不等于强度保真度达到判据；早期「7.5× 分叉」正式撤回 | 零相位起步核心 CV `6.08`、随机起步 RMS1.5 实测 `5.01`（判据 `≤0.10`）；`corr(model_cv_core, measured_cv_core_window)=+0.9497`；`ratio_core` 中位 `5.55`；实测基座 `15.0 counts` vs checkpoint `background=2.8847` | `../anchored_m3_20261116/anchored_m3_20261116.md`；`../model_optim_efficiency_20261116/model_optim_efficiency_20261116.md` |
| #247 零级/目标级分离与轴映射 | 零级稳定、目标级可分离，但 grating/tilt 与 blaze2d 不能被同一可逆 2×2 映射解释，`frozen=false` | flat 漂移 `2 px`、tilt 下零级 `≤2.3 px`、目标级两瓣 `109–119 px`；安装旋转 `+88.544°`、正交误差 `0.24°`、scale ratio `1.086`、对齐后跨轴 `≤0.12 px`；A1b RMS `0.871 px`、`1.576 %`、`90.36°`、秩 `0.9925` | `../issue247_zero_order_20261117/issue247_zero_order_20261117.md`；`../../../docs/dev/agent_notes_20261121.md` |
| 动态整形速度解耦采集 | `30/30` transition 全成功，约 `2.3 fps`，瓶颈是 SLM 写入 + 换槽，不是 CCD 读出或 LCOS 翻转 | 冻结曝光 `0.08 ms`、peak `2212.6`；fps min/中位/max `2.196`/`2.284`/`2.542`；SLM 写入 + 换槽中位 `0.312 s`（max `0.562 s`）vs CCD `0.075 s` vs LCOS `7.8 ms` | `../speed_decouple_capture_20261120.md` |
| #238 灰度相位与暗场 | `993` 为设备回读 2π 工作值；二值光栅把 g₂π 收到 `≈964±13`，但缺同曝光熄灯暗场，不发布正式 g₂π | phase `206`→`2.06π`→`993`；峰值法 `938±11`、`CI[917,961]` **撤回**；折叠 `962.6`、抛物线 `962.4`、两级中点 `964.5`、设备码外推 `964.1`；12-bit 暗场 8 档 × 16 帧 | `../../../docs/dev/agent_notes_20261119.md`；`../dark_frame_record_20261119.md` |
| P4 双门与迁移数据复用门控 | 是有效 `fail`，不是「没测好」；先修三处泄漏，物理测试门控置顶 | 强度门最差 `relL2=0.8848927385498241`（门槛 `δ=0.20`）；方向门 `2/7 + 2/7 = 4/14`（需 `10/14`） | `../transfer_efficiency_and_closed_loop_data_reuse_plan_20261121/plan.md` |
| `--debug-weights` 与在线参数可视化 | 权重轨迹在真实采集链路上功能正确，但 6 轮全部 `accepted=false`；参数只在 0→1 轮跳变 | 训练损失 `0.39325→0.34548`；跨轮相关性 `0.7599→0.6617`、相对 L2 `0.5481→0.6063`；后 10 轮方向余弦中位 `0.9978`、步长-损失下降 Spearman `0.7417` | `../debug_weights_real_verification_20261121.md`；`../transfer_online_params/transfer_online_params_20261121.md` |
| 理想光路静态整形测试 | 相位/下发链路正确，但 5 种图案远场全部塌缩成中心暗弱小斑，模型前向假设再次证伪 | SLM `1920×1200`、2π=`993`；CCD 8-bit `2592×1944`、曝光 `0.05 ms`（peak `188`）；五图案 peak `95`/`90`/`94`/`90`/`128`，全部 `observed=false` | `../ideal_path_static_test_20261117/README.md` |
| #235 零级中心四口径与 12-bit 暗场 | 统一口径下椭圆中心可用；暗场 floor 已回填，manifest 位深混存缺陷只登记不越界修改 | 四法两两最大 `3.0 px`（边界通过）；椭圆中心 `(x=1227, y=1106)`；偏置台阶 `≈15 counts`、`σ≈0.9 counts`、暗电流斜率 `0.079 counts/ms`、floor `14.9375 counts` | `../issue235_zero_order_four_method_20261120/issue235_zero_order_four_method_20261120.md` |
| 可微优化诊断 / B=2 多起点吞吐 | 相位可逐位精确回放，但像素级更新不单调；批量只提吞吐，不宣称实验帧率提升 | 重建 `121` 帧 loss `max_rel_diff=0`、终相 `max_abs_diff=0`；active 像素 `4.9 %`、符号一致 `0.611`、单调像素 `0.06 %`；B=2 吞吐 `3.37×`（`1486` vs `441 fields/s`） | `../diff_phase_monotonicity_replay_20261121.md`；`../multistart_batch_and_cpu_gpu_analysis_20261121.md` |
| 仿真矩阵 / 探针工具链 / 工程治理 | #253 湍流矩阵与探针落地，CLI 扩到 21 命令；`#205` 的 95 个重构 commit 只作工程基线 | 3 变体 × 3 档 `cn2=1e-14/2e-13/2e-12`、`distance=500 m`、σ_total `0.0626–0.8952 rad`；曝光接受带 `25 %–80 %`（目标 `45 %`）；本周 `286` commit、`95` 个 `#205` | `../../../docs/dev/agent_notes_20261122.md`；`report/turbulence_shaping/matrix_20261121/report.md` |

![图 W1 动态整形速度解耦 fps](figures/bw_w01_speed_fps.png)

**图 W1**：30 条 transition 的实测下发帧率；中位 `2.284 fps`，红色虚线为计划 `10 fps`。
计划字段 `frames=51` 是计划帧数，不代表实验下发帧数（实下发 12–14 帧/条）。

![图 W2 动态整形速度解耦耗时](figures/bw_w02_speed_timing.png)

**图 W2**：相邻事件间隔箱线图；CCD 曝光 + 读出中位 `0.075 s`，SLM 写入 + 内存槽轮换中位
`0.312 s`，直接定位速度瓶颈在 SLM 侧而非相机侧。

![图 W3 理想光路静态整形总览](figures/bw_w03_ideal_overview.png)

**图 W3**：5 种图案的 CCD 实测远场总览（2f 光路、`0.05 ms`、8-bit 全幅）；各图案均只剩
中心暗弱小斑，没有出现模型预期的方形平顶或多级衍射。

![图 W4 透镜图案三联对比](figures/bw_w04_ideal_lens.png)

**图 W4**：`lens_f125` 的相位（rad）/ 下发灰度（0–1000）/ CCD 远场（counts）三联对比；
模型预期方形平顶，实测为中心点状聚焦，是模型前向假设被证伪的直观证据。

![图 W5 在线参数按闭环轮次](figures/bw_w05_online_round.png)

**图 W5**：闭环轮次 `0–5` 的相关性/相对 L2（上）与 18 个参数归一化变化（下）；
`0→1` 轮跳变一次后基本持平，6 轮全部 `accepted=false`。

![图 W6 在线参数按拟合迭代](figures/bw_w06_online_iteration.png)

**图 W6**：拟合迭代 `0–6` 的训练损失（`0.39325→0.34548`）与参数轨迹；
本图横轴是 **拟合迭代**而非闭环轮次，不能与图 W5 直接对齐。

### 9.2 本周（11-23—11-26）：口径落地、归因分层、迁移门分离

| 方向 | 结论 | 关键数值（原样） | 证据 |
|---|---|---|---|
| 能量利用率新口径 + A/B–ABBA 实测判定 | `η = E_blaze / F0` 落地；0.12 ms 非饱和同曝光配对判定 `accepted=True`，采用 `refit_forward` | 合格探针 `η=19.27%` vs 旧三衍射级窗口 `39.78%`；效率匹配 `η_d=54.21%`、`η_s=−12.46%`、`verdict=undeterminable`；ABBA 改善 `0.062261` > 漂移 `0.035982` | 本报告 §1–§7；`../gui_closed_loop_energy_efficiency_20261125.md` |
| #247 S1e 同字节探针 A/P 与根因修复 | 变体 A/P 均 D-fail 且位置抖动；物理断电重启无效应（E2 维持）；根因定位为 F-contiguous 数组被裸 `memmove` 转置，已加守卫，重采待授权 | A：H 两两最大 `2.431 px`、`H/B=1.6004`≈`1920/1200`；P：H `1.585 px`、跨变体 D5 `1.389 px`（阈值 `1.5 px`）⇒ `no_power_cycle_effect`；守卫后 H 物理 SHA `b98be9a6…`→`7276d9da…`，42 项定向 pytest 全过；S1 维持 BLOCKED | `../s1e_same_bytes_probe_20261126/s1e_same_bytes_probe_20261126.md`；`../s1e_postfix_20261126/s1e_postfix_offline_prep_20261126.md` |
| #247 波形族归因窗口 v2 | 数据有效性门全过，但四族全 fail，`attribution_label=not_assessable`；旧 `tilt_specific` verdict 禁止引用 | 23 图样 × 4 帧 = `92` 帧；x 族极差 `89.84°` pass / y 族 `1.18°` fail；grating `0.35°`、tilt `0.74°`、blaze_patch `89.997°`；验证集方向正确 `0/10` | `../issue247_waveform_attribution_20261123/issue247_waveform_attribution_20261123.md` |
| #247 同窗口跨会话重复性 | 限定口径下 `overall_pass=true` | `69` 个唯一 id × 4 帧 = `276` 帧、三次独立会话；`reproduced 15`、`not_reproduced 0`、`not_assessable 5` | `../../../docs/dev/next_experiment_plan_20261124.md`；`../../../docs/dev/transfer_mil_issue_validity_verdicts_20261124.md` |
| 迁移学习静态点扩散函数适配 | 双门通过，支持显式启用项，默认不启用点扩散函数学习 | 验证集相对 L2 `0.60243929→0.43018529`（`−28.5928 %`）、CI `(0.087282, 0.276039)`；独立 192 px `0.52978986→0.41696567`（`−21.2960 %`）、CI `(0.040656, 0.193446)`；`sigma_ccd_px=4.33` | `../transfer_psf_adaptation_20261124.md` |
| 迁移学习几何残差适配 | 双门失败，`learn_geometry=False` 保持默认 | 验证集改善 `0.6200 %` < `5 %`；独立合并位移 RMSE `7.965166→7.747939 px`，x 轴 `−25.987 %`、y 轴 `+18.962 %` > `10 %` 上限 | `../transfer_geometry_adaptation_20261124.md` |
| HMWS 五点 Zernike 实验系统验证 | Noll 5（像散）线性可用；Noll 4（离焦）在 8-bit/12-bit 均失效，位深量化假设被证伪；焦面扫描不作补偿 | 6 条记录全 `valid/qualified`、clipped=0；8-bit Noll 5 slope `1.0753`/R² `0.999996`、Noll 4 `0.3947`/`0.9302`；12-bit Noll 5 `0.8662`/`0.9726`、Noll 4 `0.1772`/`0.2911`；`status=ambiguous`、`accepted=false` | `../hmws_zernike_probe/hmws_zernike_probe_20261124.md` |
| 闭环流程速度 / t-SNE 离线分析 | 一次完整会话约 `209 s`，采集占 40%，异步写盘可省约 `14 s` 但未接线；t-SNE 支持「实测校正块间跳变更大」 | 阶段和 `209.205 s` + 间隙 `0.390 s`；采集 `84.33 s`（`40.3 %`）、拟合 `34.77 s`、模型 `38.04 s`、实测校正 `48.29 s`；GUI 11-23 两次闭环均第 0 轮 fault | `../closed_loop_speed_tsne_20261125/closed_loop_speed_analysis_20261125.md` |
| 11-23 曝光门禁 + 效率匹配主采集 | 门禁拦住过曝；降功率后主采集 16 帧无饱和，但效率匹配仍只给趋势 | 门禁峰值 `104/255=40.8 %`、`clipped_px=0`、几何偏移 `−471.39 px`；主采集 `0.04 ms`，flat 峰 `2017–2231`、blaze `1039–1178`、shaping `1142–1237`；43 用例通过 | `../../../docs/dev/agent_notes_20261123.md` |
| GUI 完成态能量利用率页面 | 结果卡片展示完成态利用率并可对比 SLM 效率参照；历史数据按新口径复算 | 后端 `18` 项、页面 `20` 项测试通过；页面 hunk 与 provenance 让渡待合 | `../gui_closed_loop_energy_efficiency_20261125.md` |
| #254 T2/T3/T4 与 #253 M5c 工程收口 | 快照冻结器/分支预注册训练工具链落地；M5c 三次夭折后收绿 | `--reuse-device-session` 方案 A；M5c `c56c04ef`→`99cded9e` 收绿；`--memory` 上限 OOM-kill 为夭折原因之一 | `../../../docs/dev/next_experiment_plan_20261124.md`；`../../../docs/dev/next_experiment_plan_20261126.md` |
| 可微整形加速公平筛选 | K 轴淘汰、L-BFGS 入围；物理复验未获授权 | gate `max_rel_dev=0.0`；K=1 CV `0.28381`→K=4 `0.28365`（约 `0.05 %`），墙钟 `+17 %～+60 %`；L-BFGS 第 5 迭代 `≈0.67 s` vs `5.91 s`（`≈8.9×`） | `../../slm/differentiable_opt/kaxis_screening_20261123.md` |

![图 W7 S1e 变体 A](figures/bw_w07_s1e_arm_a.png)

**图 W7**：变体 A 三联图——主瓣相对 0 级偏移随上传序位（px）、flat 0 级质心漂移（CCD row/col）、
H/B 距离比；H 主瓣两两最大 `2.431 px` > `1.5 px`，六次上传全部 `col-type`，是位置抖动而非轴型翻转。

![图 W8 S1e 变体 P](figures/bw_w08_s1e_arm_p.png)

**图 W8**：变体 P（物理断电重启）同三联图；H 最大两两距离 `1.585 px` 仍 D-fail，
跨变体 D5 最大 `1.389 px` < `1.5 px` ⇒ `no_power_cycle_effect`，E4 不触发、维持 E2。

![图 W9 点扩散函数适配验证集](figures/bw_w09_psf_validation.png)

**图 W9**：验证集 6 个探针的 A/B 相对 L2；均值 `0.60243929→0.43018529`
（`−28.5928 %`），相关性变化 `+0.094949`，不恶化。

![图 W10 点扩散函数适配独立验证](figures/bw_w10_psf_independent.png)

**图 W10**：独立 192 px 采集的 `probe_003/004/005` 相对 L2；
均值 `0.52978986→0.41696567`（`−21.2960 %`），两项预注册判据均达到，作为显式启用项支持证据。

![图 W11 几何适配验证集](figures/bw_w11_geometry_validation.png)

**图 W11**：几何残差适配的 验证集相对 L2；改善仅 `0.6200 %`，低于预注册 `5 %` 改善阈值，
不能据此启用几何适配。

![图 W12 几何适配独立位移](figures/bw_w12_geometry_holdout.png)

**图 W12**：独立 192 px 验证集位移 RMSE（px）；合并 `7.965166→7.747939`，
但 x 轴 `−25.987 %` 与 y 轴 `+18.962 %` 方向不一致，y 轴超出 `10 %` 上限，独立判据未通过。

![图 W13 HMWS Zernike 线性](figures/bw_w13_hmws_linearity.png)

**图 W13**：Zernike 注入量 vs 恢复量（rad）线性拟合；Noll 5 在 8-bit（slope `1.0753`）与
12-bit（`0.8662`）均可用，Noll 4 两轮均失效（`0.3947`/`0.1772`）。

![图 W14 HMWS 焦面扫描](figures/bw_w14_hmws_focus.png)

**图 W14**：Fresnel 屈光度焦面扫描（peak/ms 随 m⁻¹）；最优点 `−0.500 m⁻¹` 但峰形不唯一，
`status=ambiguous`、`accepted=false`，未做焦面补偿。

![图 W15 实测校正相位 t-SNE](figures/bw_w15_tsne_anchored.png)

**图 W15**：实测校正反馈 20 个 SLM 相位快照的 t-SNE（`384²` 池化到 `48²`，按 4 个块着色）；
块间跳变多大于块内 Adam 步，与「逐块重新锚定」表述一致。

![图 W16 Zernike 拟合轨迹 t-SNE](figures/bw_w16_tsne_fit.png)

**图 W16**：物理预测模型拟合的 Zernike 系数轨迹 t-SNE；拟合轨迹与反馈相位轨迹分开看，
不能把块间跳变归因于 拟合阶段。

![图 W17 K 轴筛选](figures/bw_w17_kaxis.png)

**图 W17**：冻结问题 `phys_20261116_01` 上 K=1/4/8 Adam 的 CV 随迭代曲线（120 迭代）；
终态仅差约 `0.05 %`，墙钟 `+17 %～+60 %`，加速杠杆在 L-BFGS（第 5 迭代 `≈0.67 s` vs `5.91 s`）而非并行候选。

### 9.3 两周主线的收束

1. **口径主线**：上周撤回 8-bit「7.5× 分叉」、钉住四口径中心并暴露积分口径坑；本周把它
   收束为 `η = E_blaze / F0` 与同曝光 ABBA 配对。结论是「先固定分母，再谈数值」，
   与 §0、§7 一致。
2. **模型/归因主线**：上周零级可分离但轴映射不冻结、理想光路前向被证伪、P4 双门有效 fail；
   本周波形族归因 `not_assessable`、S1e `D-fail` 且根因落在数组内存布局、
   迁移点扩散函数适配达到预注册判据、几何适配未达到判据。结论是「达到判据才启用，未达到判据只登记」，默认关闭项保持关闭。
3. **速度主线**：上周定位 SLM 写入 + 换槽（`0.312 s`）是瓶颈；本周完整会话 `209.205 s`、
   采集 `84.33 s` 占 `40.3 %`，异步写盘约省 `14 s` 但未接线。可微 K 轴淘汰、L-BFGS
   入围属于算法轴加速，尚未获物理复验授权。
4. **本周重点方向**：闭环整形均匀度与能量利用率口径的统一与可复现判定；动态整形沿用
   本报告确立的非饱和同曝光配对框架，不在本汇总中新增判定。

## §10 汇报 PPT 补充页（2026-11-26 / 11-27 追加）

本报告配套的 `energy_utilization_20261126.pptx` 现为 32 页：原 23 页之后追加时间分析、梯度一致性、lr 远场 GIF、数据来源登记与实测图采集元数据五页，「变体定义」之后插入一页模型结构对照，另有 P30/P31（12-bit 位深重采与远场注入实拍，另一会话追加）与 P32（本轮 4 lr × 60 迭代完整 sweep）。两版 lr 实测并存：P27 为 20 帧时间戳版，P32 为完整逐迭代归档版；其余追加页均为只读归档数据的离线计算。

| 页 | 内容 | 素材 | 复现 |
|---|---|---|---|
| P25 时间分析 | 2026-11-24 实测会话分阶段耗时：总 ≈209.2 s，采集 84.33 s（40.3%）最大头 | `figures/fig13_stage_timing.png` | `artifacts/make_timing_figure.py`（读 `data/debug/slm_transfer/20261124_200949/journal.json`） |
| P26 梯度方向一致性 | 像素级符号一致性 0.611、单调像素 0.06%；lr 反事实 0.02/0.05/0.08/0.12 → 仅 0.05 达标；低维 Zernike 方向余弦 0.9978 对照 | `images/gradient_direction_consistency_20261116.png` | 引用 `diff_phase_monotonicity_replay_20261121.md` 与 `data/debug/differentiable_opt/phase_monotonicity_20261116.{png,json}` |
| P27 lr 远场实测 GIF（20 帧时间戳版） | 四档 lr × 20 帧，逐帧叠加 UTC+8 时间戳；原始 CCD 全画幅未暗减、全局归一化 | `images/lr_farfield_hw_ts_fullframe.gif`（20 帧） | `scripts/slm/hw_lr_farfield_sweep_20261127.py` / `render_lr_farfield_hw_gif.py`；页内所注源目录 `data/debug/time_to_target_20261127/frames1` 当前为空、manifest 缺失，逐帧溯源待原车道补齐 |
| P29 实测图采集元数据 | 图 1–12 的采集时间 / 设置参数 / 来源算法总表；图 W1–W17 以 §8.1 所引原报告为准 | 本报告 §6.0、附录表 | `artifacts/make_pptx.js`（附录页） |
| P30 12-bit HMWS 重采 | 位深假设被证伪：Noll 4 的 12-bit R² 0.2911 不作系数定标；Noll 5 仍可用但须重新标定 | 页内指标表（另一会话追加） | `data/hmws_zernike_probe/20261126_053859Z/summary.json`、`probe_config.json` |
| P31 12-bit CCD 远场 | Noll 4 / 5 三点注入实拍；暗减、共同裁剪 512 px、统一显示窗 | 页内实拍图（另一会话追加） | 2026-11-26 13:40:54–13:45:29（UTC+8）· 0.08 ms · 12-bit · bias=0.3 rad · radius=450 px |
| P32 lr 完整 sweep | 4 档 lr × 60 迭代 + 平相位参考，共 512 帧；loss 降幅 52.0%–56.4%，末 10 轮 FWHM 17.17–17.62 px | `images/lr_farfield_evolution_20261127_hw.gif`（60 帧）、`figures/fig14_lr_farfield_hw_contact_sheet.png` | `scripts/slm/hw_lr_farfield_sweep_20261127.py --execute --exposure-ms 0.05 --out data/debug/differentiable_opt/lr_farfield_hw_20261127_real`；`scripts/slm/render_lr_farfield_hw_gif.py`；逐帧指标 `artifacts/lr_farfield_hw_metrics.csv` |

- 口径边界：P27 的 20 帧 GIF 是原始 CCD 实拍帧，但页内所注源帧目录当前为空、时间戳 journal/manifest 未归档，其逐帧溯源尚未闭合；P32 的 60 帧 GIF 来自 `data/debug/differentiable_opt/lr_farfield_hw_20261127_real`（`run_config.json` + `journal.jsonl` + 512 npz，曝光 0.05 ms、8-bit、零饱和、无缺帧），可完整复现。两版均非离线模型 `|FFT|²` 帧，也不做暗减；§6 的 ABBA 帧仍按同曝光配对单独判定。
- 时间分析的数据缺口：GUI 闭环（target-lock）2026-11-23 两次运行均在第 0 轮 fault，无多轮逐项耗时归档；P25 只能用迁移会话 `20261124_200949` 的实测 journal 叠加代码口径。
- lr 数据边界：`lr_sweep_20261116` 六档（0.01–0.4）只落标量指标；P27/P32 分别给出两版四档 lr 实测序列（20 帧时间戳版 / 60 迭代完整归档版），不再用 `replay_diff_phase_monotonicity.py` 的离线冷启动帧。P32 裁定：四档 loss 降幅接近（52.0%–56.4%），未复现离线代理「0.05 最优、0.08/0.12 退化」的分离度；均匀度仍以 §5 同曝光配对 CV 为准。

## §11 变体直观命名与模型结构（2026-11-27 追加）

PPT 第 15 页新增结构对照图 `figures/fig_variant_structure.png`，由
`artifacts/make_variant_structure_figure.py` 生成。两个变体不是两种神经网络，
而是同一套可微物理预测的三个检查点；共同骨架为
`Z4 离焦 + Zernike Noll 5–11 + 高斯照明 → FFT → 161×161 相机采样 → 辐射定标`。

| 直观名称 | 原标识 | 结构/参数口径 |
|---|---|---|
| 原始模型 | `fit/frozen_parameters.json` | 控制网格 384×384、FFT 1024；同次 `capture/` 数据拟合，最佳迭代 136；是后续检查点的参数来源 |
| **高清冻结基线** | `model640`（A） | 经 `geometric_seed.export_frozen_parameters` 将控制网格升至 640×640、FFT 2048；光学与辐射参数逐位继承，**不重拟合** |
| **重拟合候选** | `refit_forward`（B） | 架构、网格与 FFT 均不变；从 A 热启动，在同一数据上 `adaptation=full`、train-only 重拟合物理参数，≤100 迭代、最佳 27 |

ABBA 单变量成立的原因：两个变体使用同一冻结预测代码、同一目标与曝光，只有检查点不同；
B 的四帧配对 CV 均值 `1.852745` < A 的 `1.915007`，改善 `0.062261` 大于漂移
`0.035982` 与容差 0。验证集上 corr `0.8574→0.8744`、rel_l2 `0.5859→0.5747`
改善，但 `cv_abs_err` 反而 `0.7841→0.9539`，不能写成「所有指标都改善」。
GUI 中的 `OnlineTransferAdapter.refit` 与 `refine_phase` 是同名不同函数，不参与本图两个检查点的生成。

## 提交状态

本报告与图表位于 `report/shape_trials/energy_utilization_20261126/`，为只读归档产物。
溯源登记与页面改动按整文件让渡给持有车道，待其落地后合入并重跑
`sync_report_provenance.py --check`、`test_report_provenance.py`、`ruff check .`、
`graphify update .`。
