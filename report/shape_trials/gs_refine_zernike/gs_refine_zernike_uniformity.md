# GS-refine Zernike 阶数扫描 (uniformity)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_gs_refine_zernike_report.py`](../../../scripts/slm/generate_gs_refine_zernike_report.py)
> **复现命令**: `python scripts/slm/generate_gs_refine_zernike_report.py`
> **数据/关联脚本**: [`scripts/slm/run_gs_refine_zernike_sweep.py`](../../../scripts/slm/run_gs_refine_zernike_sweep.py)
> **运行环境**: 离线
> **说明**: GS 逐像素相位 → Zernike 拟合 (多阶) → SPGD 细化 的阶数扫描；本脚本离线读取 run_gs_refine_zernike_sweep.py 产出的 summary.json + spgd_nmax*.csv，渲染拟合残差/CV/PIB 三图与阶数扫描表
<!-- provenance:end -->

数据来源：8 阶扫描 run，`cam_type=daheng`（硬件实测），目录 `hw_zernike_sweep_20261020b`。

## 0. 实验目的

**要回答的问题**：GS 逐像素相位（逐像元自由度的相位）整形出来的远场方斑，
能不能用**低阶 Zernike 基**复现？需要**多少阶**？在 Zernike 系数上再跑 SPGD
能不能把均匀度救回来？

**为什么值得做**：GS 逐像素相位在每个像元上都有自由度（面板 1920×1200），
系数多、标定贵、每轮下发的数据量大；而 Zernike 系数只有几十维。若低阶 Zernike
就够用，这条路能把整形的自由度从「逐像元」降到「几十个系数」。反过来，若 Zernike 基
在任何阶数下都抓不到 GS 相位的结构，那么这条路直接不通，省下的是一整轮实验。

**预期能排除什么**：排除「阶数越高越好」这种按直觉的排序，以及「拟合残差小 ⇒
均匀度就好」这个推断。两者都只在**跨阶差异大于单轮抖动**时才成立 —— 本次扫描的
判据就架在这条线上（见 §0.3）。

## 0.1 实验设置

**实验（硬件实测）**：相机 `daheng`（`cam_id=0`）、SLM `slm_number=1`，曝光 0.2 ms。

**运行配置**（扫描驱动记进 `summary.json` 的 `config`，逐字保留）：

| 参数 | 值 |
|---|---|
| `beam_radius_px` | `450.0` |
| `cam_id` | `0` |
| `cam_size` | `256` |
| `cam_type` | `daheng` |
| `defocus_focal_mm` | `1597.4` |
| `delta` | `0.02` |
| `epochs` | `250` |
| `exposure_time_ms` | `0.2` |
| `gs_iters` | `200` |
| `lr` | `0.01` |
| `max_energy_loss` | `0.6` |
| `n_max_list` | `[3, 4, 6, 8, 11, 15, 20, 25]` |
| `seed` | `0` |
| `sim_far_field_padding` | `16` |
| `slm_number` | `1` |
| `spgd_epochs` | `150` |
| `target_side` | `0.0` |

**实验系统与几何**：

| 项 | 值 |
|---|---|
| 相机 | `daheng`，`cam_id=0`（实验系统面板常量表记录：Daheng MER2-507，2592×1944，2.2 µm 像元） |
| SLM | Santec SLM-200，`slm_number=1`（面板常量表记录 #1 序列号 22030108） |
| 面板 | 1920×1200 px，像元 8 µm |
| 面板照明中心 / 半径 | (960, 600) px / 450 px |
| 波长 | 1064 nm（SLM 工作波长，面板常量表记录的设备行） |
| 曝光 | 0.2 ms |
| 相机窗口 `cam_size` | 256 px |
| 平场光斑腰径 `w0_px` | 4.98 px |
| 方形目标边长 `target_side` | 9.96 px（= 2.000 × w0，由平场腰径推导并夹在相机窗口内） |
| Zernike 基半径 `beam_radius_px` | 450.0 |
| 零级位置（相机全幅坐标） | (1427, 972) |
| 目标框中心（窗口内坐标） | (129, 127) |
| 最佳阶段 `best_stage` | `refine` |
| 数据目录 | `data/debug/hw_zernike_sweep_20261020b` |
| 焦面矫正焦距 `defocus_focal_mm` | 1597.4 mm（2f 实验系统的相机不在 2f 焦面上，瞳面要加菲涅耳相位把焦点移回探测器） |

**三段式流程**：Stage A 跑 GS-refine 得到逐像素最佳相位与平场基线 → Stage B 把最佳相位最小二乘拟合成 Noll 序 Zernike 系数（基半径 `beam_radius_px=450.0`，与 GS 瞳面一致）→ Stage C 以拟合系数为初值（`init_c`）在 Zernike 基上跑 SPGD：目标 `rms_pib`（扫描驱动的 `OBJECTIVE`）+ 方形目标，迭代 150 轮，扰动 `delta=0.02` rad，学习率 `lr=0.01`，能量损失守卫 `max_energy_loss=0.6`。


## 0.2 算法修改

本轮**没有改 GS 精化器与 SPGD 优化器本身**（`slm_gs_refine` / `slm_zernike_pib`
未因本报告而改写）。相对更早的「只跑 GS-refine」流程，本条管线新增或修好的东西是：

1. **新增 Stage B 的 Zernike 拟合模块**
   `src/ao_shaping/optimizer/wfless/gs_refine_zernike_fit.py`（`fit_best_phase`）：
   把 GS 的最佳相位最小二乘拟合成 Noll 序系数，基半径锁在 `beam_radius_px`
   （与 GS 瞳面一致）。走 canonical 的 `utils/wavefront/zernike_calc` 基，
   **不新写第二套 Zernike 数学**。
2. **新增三段式扫描驱动** `scripts/slm/run_gs_refine_zernike_sweep.py`：
   Stage A（GS-refine 出逐像素最佳相位 + 平场基线）→ Stage B（多阶拟合）→
   Stage C（在 Zernike 基上跑 SPGD，系数用拟合结果播种 `init_c`）。
3. **共享方形目标框由平场腰径推导**（`TARGET_BOX_WAIST_FACTOR` × w0，下限
   `MIN_TARGET_SIDE`，上限 `cam_size / CAM_WINDOW_TARGET_MARGIN`）：绕开
   「自动目标边长约 400 px vs `cam_size` 默认 300 ⇒ `square_mask` 裁到整幅 ⇒
   `pib` 恒等于 1.0」那个退化几何。本次的实际取值见 §0.1。
4. **各阶用同一个冻结的方形掩模离线重打分**（`score_frame`）：Stage A 与 Stage C
   的画面几何不一致时先 `_crop_to_shape` 对齐，否则跨阶的 PIB/CV 不可比。
5. **系数截断告警**：拟合系数超过 `ZERNIKE_CLIP` 会被 SPGD 静默截断，拟合质量
   不可信 —— 扫描表里的 clip 列非空即为警告。
6. **焦面矫正叠加层**（`--defocus-focal-mm`）：2f 实验系统的相机不在 2f 焦面上，
   瞳面要加菲涅耳相位把焦点移回探测器，否则离焦本身先吃掉光斑锐度。
   本次运行的取值见 §0.1。
7. **报告生成器本轮**：新增「实验四要素」小节（§0–§0.3）；`--run` 可显式指定
   run 目录；结论口径保持数据驱动 —— 跨阶差异小于单轮抖动时**不给**「最优阶数」。

## 0.3 结果分析

（判据与数字全部取自本报告已算出的量；完整表格见 §1–§2。）

共扫描 8 个 Zernike 阶数 (n_max = 3, 4, 6, 8, 11, 15, 20, 25)。平场基线 PIB=0.2681, CV=0.3146; GS 逐像素相位基线 PIB=0.2803, CV=0.3065。

- **唯一随阶数单调的量是拟合残差**: 0.3105 → 0.1566 rad (-49.6%), 单调递减 = True。这是本次扫描中唯一站得住的阶数效应。
- **阶数之间无法排序**: PIB 跨阶数极差 0.0161、CV 极差 0.0243, 而 SPGD **单轮运行内**的目标抖动就有 0.0072~0.0109。两者同量级 ⇒ 任何「最优阶数」都是在读噪声, 本报告不给这个结论。
- **基相对平场的变化与阶数无关**: 各阶 ΔPIB 均为 -0.1196~-0.1035, 来自拟合相位本身, 不是来自选了哪个阶数。
- **拟合几乎没抓到结构**: 残差 rms 是最大系数的 0.5~1.0 倍 (max|c| = 0.3106 rad, 各阶相同) ⇒ 被拟合的相位接近平场, Zernike 基在任何阶数下都没有可表达的内容, SPGD 因此无信号可优化。


## 1. 基线

| 相位 | PIB | CV | EE | AR | 复合分数 |
|---|---|---|---|---|---|
| 平场 | 0.2681 | 0.3146 | 0.2681 | 1.1111 | 0.4638 |
| GS 逐像素相位 | 0.2803 | 0.3065 | 0.2803 | 1.2500 | 0.4233 |

## 2. 阶数扫描

| n_max | 模态数 | 拟合残差 (rad) | max\|c\| | PIB 前 | PIB 后 | CV 前 | CV 后 | SPGD 改善 | clip 警告 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | 0.3105 | 0.3106 | 0.1468 | 0.1486 | 0.2238 | 0.2125 | ✓ |  |
| 4 | 15 | 0.3000 | 0.3106 | 0.1509 | 0.1537 | 0.2129 | 0.2124 | ✓ |  |
| 6 | 28 | 0.2840 | 0.3106 | 0.1555 | 0.1579 | 0.2109 | 0.2179 | ✓ |  |
| 8 | 45 | 0.2695 | 0.3106 | 0.1622 | 0.1647 | 0.2076 | 0.2106 | ✓ |  |
| 11 | 78 | 0.2529 | 0.3106 | 0.1589 | 0.1566 | 0.2090 | 0.2174 | ✓ |  |
| 15 | 136 | 0.2237 | 0.3106 | 0.1606 | 0.1592 | 0.2040 | 0.1936 | ✓ |  |
| 20 | 231 | 0.1895 | 0.3106 | 0.1586 | 0.1504 | 0.1984 | 0.2073 | ✓ |  |
| 25 | 351 | 0.1566 | 0.3106 | 0.1542 | 0.1542 | 0.2069 | 0.2069 | ✗ |  |

SPGD 轨迹 CSV 已保存 (151 行, n_max=3)。

## 3. 图

![拟合残差](figures/fit_residual_vs_nmax.png)

![CV 前后对比](figures/cv_before_after_vs_nmax.png)

![PIB 前后对比](figures/pib_before_after_vs_nmax.png)

## 4. 口径与已知局限

1. **`cv` 与 `ee` 必须一起看。** 单独优化 `-cv` 会掏空目标方斑（硬件实测 EE → 0.002）；
   目标函数必须包含能量项。
2. **Zernike 基半径锁定 `beam_radius_px`**，且与 GS 瞳面一致；更换光路必须重扫。
3. **仿真模式下 GS 预矫正未经物理标定。** 数字孪生是纯 FFT；`_derive_target_side`
   的实验系统模型（`camera_pixel_um=2.2`、焦距来自实测 tilt-shift scale）与仿真角尺度
   不匹配。日志会出现 "GS warm start ... did not beat flat ...; keeping flat"。
   仿真只验证 Zernike 拟合 + SPGD 路径与退化几何守卫；物理对比仅在真实硬件上有意义。
4. **`max_abs_coefficient` vs `ZERNIKE_CLIP = 5.0` rad。** 超过阈值会触发 clip，
   拟合质量不可信；上表 clip 列非空即为警告。
5. **已知潜在缺陷 (未修)：** `SlmGsRefineConfig` 自动目标边长约 400 px vs `cam_size`
   默认 300 → `square_mask` 裁剪到整幅 → `pib` 恒等于 1.0。驱动通过 waist 推导
   共享边长规避；共享优化器有意保持不变（其测试契约已固定）。
6. **只可比同一次 run 内的阶数, 不可跨 run 比 `fit_rms`。** 实测: 完全相同的命令行
   在 12 路并发与低负载下**逐位一致** (9 位小数全同), 说明 Stage A 本身确定性良好;
   但 `--n-max` 列表或 `--max-energy-loss` 不同时, Stage A 的 GS 最佳相位会整体改变
   (同一 `n_max` 跨 run 的 `fit_rms` 相差 0.008~0.012 rad)。这两个参数都在 Stage A
   之后才被读取, 根因未查明。因此本报告的阶数单调性只在**单次 run 内**成立
   (同一 `best_phase`, 已验证 `fit_best_phase` 不原地修改入参, 各阶互相独立)。
7. **`fit_rms` 的可比性依赖 `beam_radius_px`。** 该值同时锁定 GS 瞳面与拟合基,
   换光路必须与 `--n-max` 一起重扫, 否则阶数之间的差异里混入了基半径不匹配。
