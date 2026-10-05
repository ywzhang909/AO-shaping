# SLM 台架开机表征指南 (pre-run characterization)

> **什么时候读这份**: 在你信任任何一次 SLM 优化数字**之前**。
> GS / GSNet(深度学习) / SPGD 三类 runner 启动前, 先跑这里的三个探针,
> 确认台架本身没有在骗你。

本指南只讲**怎么先量台架**。三条实测结论直接决定它存在的必要性:

1. 固定等待时间在本台架**无效** —— 驱动按灰度图变化量估算翻转时间, 两个统计量
   相近的相位会让它报 **0.0 ms**。实测同一条斜坡首读 FWHM 43.2 px, 3 秒后 12.8 px,
   质心移动 62 px。
2. 面板**保留上一次显示的图案**。下发平场之前读到的"平场"其实是上一轮的散斑。
3. **0 阶不在画幅中心**。实测 1944×2592 画幅上 0 阶在 (674, 1026); 用
   `shape // 2` 取框会量到框外背景(实测光斑落在框外, 得出"92% 是背景"的错误结论)。

---

## 1. 三个探针

全部是 **standalone**(未注册为 CLI), 用 `python -m` 跑, 全部支持 `--no-hw`
(只打印采集计划、退出 0、不碰硬件), 全部**不需要硬件就能跑测试**。

| 探针 | 回答的问题 | 命令 |
|---|---|---|
| `slm_drift_probe` | 平场**稳不稳**? 曝光**线性吗**? | `python -m ao_shaping.tools.slm.slm_drift_probe` |
| `slm_floor_probe` | 能分辨**多小的信号**? 噪声是**读噪声还是漂移**? | `python -m ao_shaping.tools.slm.slm_floor_probe` |
| `slm_abba_probe` | 稠密随机相位**测得出来吗**? | `python -m ao_shaping.tools.slm.slm_abba_probe` |

建议顺序: **drift → floor → abba**。前者定曝光, 中者定本底与稳定判据,
后者才回答"某个扰动到底有没有效果"。

### 1.1 `slm_drift_probe` — 漂移 + 曝光线性

```bash
python -m ao_shaping.tools.slm.slm_drift_probe \
    --cam-type daheng --cam-id 0 --exposure-ms 0.4 \
    --n-drift 24 --drift-period-s 5.0 \
    --exposure-ladder 0.2,0.3,0.4,0.6,0.8,1.0,1.25,1.5,2.0 \
    --ladder-repeats 2 --out data/slm_drift
```

用**区域范数 / box sum**判漂移, **不用峰值** —— 实测同一 3 ms 设置两次运行
峰值读到 100 与 23 counts, 而 box sum 稳定到 0.2%。

曝光阶梯判据 (`bench_kernels.exposure_monotonicity`):
- `verdict == "monotonic"` 且 `saturated == False` → 该区间线性可用
- `verdict == "non_monotonic"` → 中间档掉回去了, 别用
- `verdict == "saturated"` → 撞满量程, 降曝光

**本台架实测基线** (2026-10-01, 固定 0.4 ms, 24 次 / 62 s):
peak cv 1.1%、全图 sum cv 0.2%、40 px box sum cv 0.21%、FWHM 13.3–13.74 px。
曝光 **t ≥ 0.4 ms** 单调近线性, **t < 0.4 ms 读数紊乱**, 裁切拐点约 2.0–2.2 ms。

> ⚠️ `--exposure-ms` 的旧默认是 **80 ms**, 在本台架**必然饱和**。
> 显式传参, 别依赖默认值(见 §4 表第 4 行)。

### 1.2 `slm_floor_probe` — 本底 / 稳定时间 / SNR-vs-K

```bash
python -m ao_shaping.tools.slm.slm_floor_probe \
    --cam-type daheng --cam-id 0 --exposure-ms 1.5 \
    --n-repeat 20 --settle-curve-s 4.0 --settle-sample-ms 120 \
    --ks 1,4,9 --perturbation-rad 0.6 --out data/slm_floor
```

**SNR 那一段才是重点。** `snr[K] = mean(observable[:K]) / (floor / sqrt(K))`,
而 observable 是 L2 范数(恒非负), 所以**均值不会相消** —— 只有响应
**可复现**时, 平均才有收益。

于是出现一个干净的判据:

| `improves` | `verdict` | 含义 | 该做什么 |
|---|---|---|---|
| `True` | `noise_limited` | 读噪声主导 | 降 delta / 降维数即可 |
| `False` | `drift_limited` | 漂移主导, 响应不可复现 | **改稳定判据或改用 ABBA**, 降 delta 无用 |

`False` 通常意味着面板还没稳就采了 —— 这正是第一次 576 单元转移矩阵失败的成因。

稳定时间由 `settle_curve` 采样拟合(`settle_time_s`)。厂家给的是下限参考:
Santec SLM-200 Tr/Tf 典型 **200 ms**, 稳态相位起伏典型 **< 0.001π rad**;
Hamamatsu LCOS 有效更新率约 10–数十 Hz。判定"已稳"的通用做法是
**连续 3–5 帧落在稳态起伏的 2–3 倍以内**, 而不是"睡够 N 秒"。

### 1.3 `slm_abba_probe` — 稠密相位可分辨性

```bash
python -m ao_shaping.tools.slm.slm_abba_probe \
    --cam-type daheng --cam-id 0 --rebracket-exposure \
    --n-patterns 12 --pairs 3 --grid 24 --delta-rad 0.5 \
    --out data/slm_abba
```

用 ABBA(`+ - - +`)消掉一阶漂移, **复用** `sweep_analysis.abba_signal`
(不重复实现), 并在同一个曝光下**先量漂移本底**。

**判读**: `verdict == "unusable"`(超过本底的 pattern 数为 0)说明当前
δ/曝光/稳定协议下**稠密相位整体不可测** —— 此时不要去测 576 单元转移矩阵。
先把 `drift` / `floor` 的问题解决。

> 第一次转移矩阵就是这一步没做: 固定 0.3 s 等待 + 单帧读, 每个单元响应与
> flat-to-flat 漂移同量级(ratio 1.0×), 结果已作废。

---

## 2. 两种帧预处理 —— **不可统一**

这是本指南最容易踩的坑。两个函数长得像重复实现, **但它们是不同的, 而且各自正确**:

| 函数 | 镜像自 | 用于 | 为什么 |
|---|---|---|---|
| `finite_clip` | `slm_square_shaping.square_peak_to_background_ratio` | 峰值/背景类比值 | 背景占满画幅, **中位数就是本底**; 扣掉它会让分母塌掉 |
| `finite_median_subtract` | `slm_gs_refine._prepare_frame` | 以**全图**为分母的任何比值 | 对称读噪声让约一半像素为负, 不扣本底比值会 **> 1** |

实测差异(合成帧, 本底 3 counts, 峰值 100):

| 预处理 | 背景均值 | PBR |
|---|---|---|
| `finite_clip` | 3.00 | **34.3** ← 正确 |
| `finite_median_subtract` | 0.0018 | **55758.6** ← 分母塌了 |

**1600 倍误差。** 用错就是这种量级。

另一个方向同样真实: 不做中位数扣除直接算全图分母比值, 实测
`PIB = 1.0120`(负像素让比值冲到 1 以上), 优化器会开始追噪声。

> 顺序也是契约: **先扣中位数再截零**。先截零会把对称噪声整流成一个
> 与像素数成正比的 DC 地板。
> 两条都有测试锁定在**活的优化器函数**上
> (`tests/ao_shaping/tools/slm/test_slm_bench_metrics.py::TestPrepDivergence`),
> 以后谁想"顺手统一"都会先撞测试。

---

## 3. 环围能量(EE)归一化约定不一致

仓库里同时存在**三套** EE/重叠度约定, 分母不同:

| 位置 | 形式 |
|---|---|
| `utils/image/beam_metrics.py:223` | `in_mask.sum() / total`(全图 sum) |
| `optimizer/wfless/slm_square_shaping.py:369` | `np.sum(img)`(全图 sum) |
| `utils/image/beam_metrics.py:159` | `2·Σmin(m,t) / (Σm + Σt)` |

标准的环围能量是**全图能量归一化**(Born & Wolf; Siegman)。用全图分母时,
负的读噪声背景会污染分母, 所以**必须**先 `finite_median_subtract`。

**报告里引用 EE 时必须写清归一化方式**, 否则两个数字没有可比性。

---

## 4. 危险默认值清单

行号均在 2026-10-02 核对过。

| # | 位置 | 值 | 危害 |
|---|---|---|---|
| 1 | `tools/slm/slm_zernike_sweep_probe.py:527` | `_root=`/`name=`/`extra_meta=` | **已修(`dc3d2e3`)**。此前: 扫描与 npz 都成功后仍 `TypeError`, 调试产物全丢 |
| 2 | `drivers/slm/santec/driver.py:1258` | `target_slot` 未绑定 | **仅 DVI 模式**触发 `UnboundLocalError`。超出本次写入范围, **未修**; DVI 本就不该自动尝试(`open()` 已知会挂起) |
| 3 | `optimizer/wfless/slm_zernike_pib.py:137`<br>`optimizer/wfless/slm_zernike_shaping.py:123` | `SLM_RESPONSE_TIME_S = 0.0` | **零稳定等待**, 各有 7 处 sleep 调用点。静默读到未稳定帧 |
| 4 | `runners/runner_common.py:1178` | `SlmSquareParams.exposure_ms = 80.0` | `spgd-square` 在本台架**必然饱和**(help 文本自己就写着 "default: 80")。`slm-pib` 已因同样原因改成 1.5 |
| 5 | `optimizer/wfless/slm_gs_refine.py:87` | `_DEFAULT_CAMERA_PIXEL_UM = 3.31` | 大恒 MER2-507 实测 **2.2 µm**。GS 会瞄错角尺寸(bake-off 能兜住, 但 GS 白算) |
| 6 | `optimizer/wfless/slm_gs_refine.py:94` | `_DEFAULT_FAR_FIELD_PADDING = 3` | 文档/仿真侧用 **8**(`iterative_zernike_shaping.py:93`、`drivers/sim/slm_shaping_bench.py:100`); 代价是平方级 |
| 7 | `optimizer/wfless/slm_gs_refine.py:169` | `focal_length_m = 0.125` | GUI 侧不一致: `gui/slm/pattern_controls.py:1181` GS 用 `100.0` mm, `:453`/`:1381`/`:1663` 用 `300.0` mm |
| 8 | `tools/slm/bench_kernels.py:102` | `TILT_SHIFT_SCALE = 7400.0` | 文档里同时存在 `7600/P`、`5021/Λ`、`132940/P` 四种说法。一个物理量四个数 |
| 9 | `optimizer/wfless/slm_zernike_pib.py:1214,1443,1470` | 每次迭代 `set_reference_center(zero_order_center(...))` | 散斑上 argmax 在近似等亮的颗粒间跳 ⇒ 目标框漂移 ⇒ 指标不连续, 优化器追一个会动的框 |
| 10 | `utils/image/beam_metrics.py:371,414,462,512` | 4 个 0 阶定位实现 | 暗帧上裸 `argmax`: 峰值 22–46 而帧均值 0.26, 单个热像素即可取胜; 参考质心曾在 60 px 内自漂 |
| 11 | `optimizer/wfless/slm_square_shaping.py:435`<br>vs `optimizer/wfless/slm_gs_refine.py:251` | 截零 vs 扣中位数 | **故意不同**, 见 §2。统一会引入 1600× 误差 |
| 12 | `utils/image/beam_metrics.py:223` 等三处 | EE 分母约定不统一 | 见 §3 |
| 13 | `drivers/slm/santec/slm200_constants.py:19` | `get_max_grayscale() = 1023` | 设备实测 2π = **993** @1064 nm。矫正 CSV 用 1023, LUT/波前路径用设备值 |
| 14 | `tools/slm/bench_kernels.py:284`<br>vs `optimizer/wfless/slm_gs_refine.py:334` | 两份 settle 实现 | 后者是私有第三份拷贝, **不要照抄**。`slm_zernike_sweep_probe.capture_settled` 是认可的薄封装 |
| 15 | `tools/slm/slm_zernike_common.py:41` | `WFS_ZERNIKE_ORDER = 10` | 注释警告 15 阶非法 → 66 项。阶数写错会静默改变矩阵维度 |

---

## 5. runner 启动前的最小检查

- [ ] **memory mode only** —— 绝不自动尝试 DVI(`video_mode=1`), `open()` 可能挂死,
      恢复需要给 SLM 控制器**物理断电**
- [ ] 曝光已在本机激光功率下**重新 bracket**(参考值 1.5 ms, 不是 80 ms)
- [ ] `slm_drift_probe` 报告 `monotonic` 且未饱和
- [ ] `slm_floor_probe` 的 `verdict` 已知; 若是 `drift_limited` 先修稳定判据
- [ ] 目标 ROI 由 `argmax` 在**去噪后的平场**上定位一次, 并**整轮冻结**
- [ ] 相位写入走 `display_data(gray)` **不传** `memory_number`(固件对当前槽
      是 no-op), 且**不传** `wait_time_s`
- [ ] 采集用 `capture_settled`, 不是 `time.sleep`
- [ ] 判定收敛用 `dec` / `late` / `best-vs-sustained`, **不要只看 `sustained`**

> **同一 seed 不保证逐帧复现**: seed 只锁定 SPGD 扰动符号, 锁定不了测量。
> 两次同 seed 运行在**平场基线**处就已相差 ~1e-3。断言近似可复现, 不要断言相等。

---

## 6. 判据来源

| 结论 | 出处 |
|---|---|
| SLM-200 Tr/Tf ~200 ms; 稳态相位起伏 < 0.001π | Santec SLM 规格表 / SLM-200 数据表 |
| LCOS 有效更新 ~10–数十 Hz; 上升/下降 10–90% | Hamamatsu LCOS-SLM drive timing / time response |
| 曝光取满量程 ~60%(50–80%), 削顶看 ADC 直方图尖峰 | Janesick, *Photon Transfer DN→λ* (SPIE 2007); EMVA 1288 |
| 帧平均降读噪声 ∝ 1/√N; 但漂移主导时**反而变差** | 同上(实验光学常规) |
| EE 标准定义 = 全图能量归一化 | Born & Wolf, *Principles of Optics* 7e; Siegman, *Lasers* |
| Maréchal 近似 `S ≈ exp(-σ_W²)`, 仅在 `S ≳ 0.3–0.5` 可信 | Maréchal 1947; Born & Wolf §9.1; Mahajan, JOSA A 18, 1373 (2001) |
| SPGD 每轮 2 次测量 | Vorontsov, Opt. Lett. 22, 907 (1997), DOI 10.1364/OL.22.000907 |
| ABBA 抵消一阶漂移 | 有限差分通用做法; **无单一权威 DOI**, 属常用实践 |

标"本台架实测"的数字来自 2026-10-01 会话, 原始 recorder 产物按仓库约定存在
`data/debug/*.pkl` 与 `data/slm_*/`(均已 gitignore)。

---

## 7. 相关文件

- 探针: `src/ao_shaping/tools/slm/slm_{drift,floor,abba}_probe.py`
- 测量 + 分析内核(设备注入, 已合并): `src/ao_shaping/tools/slm/bench_kernels.py`
- 目录说明: [`src/ao_shaping/tools/slm/README.md`](../../src/ao_shaping/tools/slm/README.md)
- 实测标定报告: [`docs/slm/bench_calibration_20261001.md`](bench_calibration_20261001.md)
