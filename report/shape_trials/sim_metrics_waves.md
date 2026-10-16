# 仿真指标与守卫五波交付（`feat/sim-metrics-waves`）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线
> **说明**: 五波交付记录 (规范化 Strehl/M² 指标 / GS 金标 / 留出探针 K 折 / 强度域开关 + 晋升门 / 自适应信赖域), 重点是「默认路径逐位不变」的钉法与已知的既有基线失败; 两次真实测量见同目录 model_in_loop_cv.md 与 gs_golden_reference.md
<!-- provenance:end -->

> **这份文档记录的是代码交付与它们各自的验证口径，不是某一次测量的结论。**
> 两次真正的测量在同目录另有两份，各自有生成脚本：
> [`model_in_loop_cv.md`](model_in_loop_cv.md) 与 [`gs_golden_reference.md`](gs_golden_reference.md)。

本轮在独立 worktree（起点 `defa275e`）上交付五组改动。贯穿全部五组的是同一条纪律：
**默认路径必须逐位不变**。凡新增的行为都由一个默认关闭的开关承载，并且该「不变」
不是靠声明，而是由测试钉住的（见 §3）。

## 1. 交付清单

| 波 | 内容 | 落点 |
|---|---|---|
| W1 | 规范化光学指标：Strehl、M²（焦点栈 / 相对 / Maréchal 交叉核对）、二阶矩束腰、理想峰值 | `src/ao_shaping/utils/image/beam_metrics.py` |
| W2 | GS 金标参照（委托 canonical `gs_shape` 的无 torch 叶子） | `src/ml/shaping/gs_reference.py` |
| W3 | 留出探针可注入划分 + K 折交叉验证 | `slm_model_in_loop.py`、`src/ml/zernike/cv_harness.py` |
| W4 | 默认关闭的强度域结构项 + 晋升门 | `square_shaping.py`、`src/ml/zernike/promotion_gate.py` |
| W5 | 自适应信赖域半径调度 | `src/ao_shaping/optimizer/wfless/adaptive_trust_region.py` |

两份生成报告：`model_in_loop_cv.md`（5 折，`sim` 后端）、`gs_golden_reference.md`（GS 收敛曲线）。

## 2. 两次测量的结论摘要

### 2.1 留出探针的 K 折（`model_in_loop_cv.md`）

| 指标 | 值 |
|---|---|
| 固定划分 均值 ± sd | 0.4033 ± 0.0473 |
| 轮换划分 均值 ± sd | 0.4106 ± 0.0373 |
| Δ 均值（轮换 − 固定） | +0.0074 |
| Δ 的折间 sd | 0.0565 |
| Cohen's `d_z` / sign-flip p | +0.131 / 0.9375 |
| 最小可达 p = 2/2^5 | 0.0625 |

**没有观察到两臂的系统性差异**（p = 0.9375 远高于结构下限）。真正值得注意的是
**Δ 的折间 sd（0.0565）是 Δ 均值（0.0074）的约 8 倍**：换一组留出探针带来的抖动，
远大于「换 vs 不换」的系统差。所以「固定划分给出的那一个数」不能当作模型属性引用。

这条结论本身受两条限制约束，报告里写明了：5 折**结构上**无法给出 p < 0.05；
且全部数字来自 `sim` 后端，硬件的读出噪声与漂移不在模型内，只会让折间 spread 更大。

### 2.2 GS 金标（`gs_golden_reference.md`）

| 臂 | 质量分 | 箱内 CV | 箱内 EE | Strehl | 相对 M² |
|---|---|---|---|---|---|
| 平相位 | 0.7229 | 2.2464 | 0.9990 | 1.0000 | 1.0000 |
| GS（n_iters=200） | 0.8543 | 0.4276 | 0.9137 | 0.1329 | 1.4415 |

**GS 在本仿真中胜出，但这是关于仿真的陈述。** 仿真里正向模型按构造精确，GS 几乎必然占优；
仓库记录的硬件结论相反（GS 反复输给平相位）。报告把这条警告写进了结论本身，
而不是留给读者从一个绿色数字去推断。

三个指标**同向**（CV 降、Strehl 降、相对 M² 升）是**平坦化变换的正常表现**，不是像差：
把高斯焦斑摊成平顶方块必然降峰值、扩二阶矩。报告因此明确写：**不能用 Strehl 下降
判定 GS 失败**。

GS 迭代预算由曲线定，`n_iters = 100` 与 `200` 数值完全一致，已收敛。

## 3. 默认不变是怎么钉住的

每一处新增行为都由默认关闭的开关承载，且「不变」都有对应测试：

| 开关 | 默认 | 不变的断言方式 |
|---|---|---|
| `SquareShapingConfig.w_loggrad` | `0.0` | `shaping_score(...)` 在权重为 0 时**逐位等于**旧的 `composite_score(compute_bench_metrics(...))`；分支提前返回，不做多余计算 |
| `SlmModelInLoopConfig.probe_split_hook` | `None` | 默认等于历史种子算术；且用「复现旧算术的 hook」跑完整 run，结果与无 hook 的 run **逐位一致**（含 history） |
| `SlmModelInLoopConfig.holdout_on_sim` | `False` | 默认 sim run 的 `holdout_correlation` 仍为 NaN；该开关只**测量**不**判定**，中止闸门仍锁在 `cam_type != "sim"` |
| `SlmModelInLoopConfig.adaptive_trust_region` | `False` | 默认 history **不新增任何键**，避免把既有测试全部重新定基 |

**W4 的晋升门是对「默认关闭」的对偶。** 默认关闭意味着以后可以有路把它打开，
晋升门就是那条路，并且它把拒绝理由逐条写明，四类拒绝互不相同：

1. **结构性拒绝** —— `2/2^k > alpha`，该协议**对任何效应量都无法**达到显著；
   报告同时给出「需要多少折」（`folds_needed_for_alpha`）。
2. **实际显著性拒绝** —— 效应量低于 `min_effect`。
3. **符号拒绝** —— 方向相反；可分辨的**退步**是拒绝理由，不是翻转许可。
4. **轴拒绝** —— `paired_on="seed"` 不抵消折难度，默认拒绝提升，
   必须显式 `allow_seed_paired=True` 才放行。

这四条来自本仓库已有的两条教训：`input_terms` 的 p 值翻车（沿错轴配对）、
以及 hybrid 前向模型 p ≥ 0.61 却多 45× 参数却未提升。

## 4. 验证结果

| 范围 | 结果 |
|---|---|
| W1 目标（`test_beam_metrics` + `test_square_shaping`） | **180 passed**（基线 150 + 新增 30） |
| W3-T3.2 后（三个 model-in-loop 文件） | **169 passed** |
| W4（晋升门 + 强度域开关） | **24 passed**；既有 180 仍全通过 |
| W5（自适应信赖域 + 接线） | **33 passed** |
| **最终合并回归（10 个文件）** | **769 passed, 8 failed** |

那 8 个失败**全部是本 worktree 的既有基线问题，与本轮改动无关**，已在主仓库
（`edbb820c`）对照确认：

- 3 个 `report/shape_trials/*_20261013.md` 未在 `REPORTS` 注册 —— `defa275e` 时点尚未注册；
- `report/slm/bench_probe/report.md` 的注册脚本路径在该时点不存在；
- 3 个相对链接失败 —— 目标图片不在 worktree 中。

另有 22 个 `tests/ao_shaping/ml/` 失败同属环境问题而非代码问题：worktree 的
`data/` 只有 59 个文件，主仓库有 26016 个（语料被 gitignore），数据依赖用例因此
取不到记录（`files_scanned=0`）。其中 `test_mse_path_is_bit_identical_to_the_hard_captured_reference`
在主仓库同样失败，属既有基线失败。

## 5. 命名与分层约束（勿回退）

- **`square_shaping.strehl` 与 `beam_metrics.strehl_ratio` 是两个不同的量。**
  前者是**去均值余弦相似度**（尺度不变，无衍射极限参考），后者是**光学峰值比**。
  已有 4 个测试与 `compute_bench_metrics["Strehl"]` 依赖前者的数值，其函数体**逐字节冻结**，
  只加了 deprecated 说明。两者差异由测试钉住：同一理想光斑放大 2 倍，
  `shape_strehl` 仍给 1.0，而 `strehl_ratio` 给 2.0。
- **M² 无法单焦面测量**，故 `m2_reference_relative`（BPP 比值）才是单平面主 API；
  `m2_from_focus_stack` 需要 ≥3 个焦面。
- **Maréchal 交叉核对输入是弧度**，`wavelength_m` 相消，保留只为 API 对称。
- **统计量不复算**：K 折、配对、p 值下限全部来自 `ml/zernike/cv_harness.py`
  与 `ml/zernike/eval_stats.py`。
- **依赖方向**：`ao_shaping` 永不 import `ml`。正因如此，W4 的晋升门放在
  `ml/zernike/` 而非算法层 —— 它需要 `eval_stats`，而算法层不能反向依赖。
- **GS 是对照而非基线**；W2 的模块文档与生成报告都写死了这条。

## 6. 未做与建议的下一步

- **没有上硬件。** 全部分离线，`cam_type="sim"`。硬件留出分数的折间 spread 只可能更大，
  若要把 §2.1 的 0.0565 当作硬件预算，必须在实验系统上重跑同一协议。
- **5 折的统计功效不足**。要把「Δ 是否真的为 0」变成可判定的结论，需要 ≥6 折
  （`folds_needed_for_alpha(0.05) == 6`），当前协议的定位是**稳定性测量**而非显著性判定。
- **W4 的强度域项尚未被 K 折验证过是否值得打开。** 按设计，它应当先经
  `evaluate_promotion` 在留出折上过关才可提升；本轮只交付了门与开关，没有替它下结论。
- **W5 的自适应半径尚未在硬件上验证。** 它的收缩策略假设「拒绝意味着半径过大」，
  这个假设在 `sim` 上恒真（模型精确），在硬件上还未经检验。
