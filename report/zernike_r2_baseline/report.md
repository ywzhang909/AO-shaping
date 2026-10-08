# Zernike 前向模型:验证 R² 被"常数基线"吞掉了

<!-- provenance:start -->
> **生成脚本**: [`scripts/diagnose_pod_ridge.py`](../../scripts/diagnose_pod_ridge.py)
> **复现命令**: `python scripts/diagnose_pod_ridge.py --k 4 8 16 32 64 128 256 # 再跑 verify_pod_ridge_canary.py / diagnose_r2_baseline.py / sweep_target_transform.py`
> **数据/关联脚本**: [`scripts/diagnose_pod_ridge.py`](../../scripts/diagnose_pod_ridge.py)
> **数据/关联脚本**: [`scripts/verify_pod_ridge_canary.py`](../../scripts/verify_pod_ridge_canary.py)
> **数据/关联脚本**: [`scripts/diagnose_r2_baseline.py`](../../scripts/diagnose_r2_baseline.py)
> **数据/关联脚本**: [`scripts/sweep_target_transform.py`](../../scripts/sweep_target_transform.py)
> **运行环境**: 离线
> **说明**: 验证 R² 被常数基线吞掉：常数预测器在同一 10 折协议下 R²=+0.910，故改用 skill = 1 - mse_model/mse_const 记分；含 image_mode 默认值论证的撤回
<!-- provenance:end -->

> **生成脚本**: `scripts/diagnose_pod_ridge.py` / `scripts/verify_pod_ridge_canary.py` / `scripts/diagnose_r2_baseline.py` / `scripts/sweep_target_transform.py`
> **复现命令**: 见各脚本 docstring
> **运行环境**: `.venv` (torch 2.14.0+cu132, RTX 5080 Laptop GPU), corpus `slm_zernike_shaping` 1010 条 / 10 个 pickle
> **协议**: 10 折 leave-one-pickle-out(`str(record.path)` 分组,与 `compare_models_cv.py` / `train_coeff.file_folds` 一致),判据 `ml.zernike.metrics.batch_image_metrics`

## 1. 结论(TL;DR)

**本项目记录的所有 val R² 对比都建立在一个被平凡基线吞掉的地板上。**
一个**常数**预测器(输出训练集平均图)在同一 10 折协议下就能拿到 **R² = +0.910**。
因此:

| 方法 | val R² | 相对常数基线的 skill |
|---|---|---|
| **常数基线**(训练均值图) | **+0.9097** | 0 |
| physics(`n_max=15`,135 参数) | +0.8727 | **负**(比常数还差) |
| physics 调优(`n_max=20, lr=0.01`) | +0.8803 | **负** |
| U-Net[16…256](7,778,465 参数) | +0.9004 | ≈ 0 |
| hybrid(物理 + 零初始化 CNN 残差) | 与 physics 不可区分 | ≈ 0 |
| **POD + ridge(K=4,闭式解)** | **+0.9163** | **+0.061** |

历史结论 "U-Net 显著优于 physics(SSIM p=0.0020、R² p=0.0117)" 在**统计上**成立,
但**实际效应量远小于度量地板**:整个模型表活在地板上下 0.03 的带子里。
`sweep_coeff_models.py` 里已经有一个 "constant-predictor canary" 门,
本文件说明它当年没有覆盖到 `train_amp` / U-Net 这条比较链。

## 2. 为什么常数基线这么高

对 10 折逐折实测(`logs/r2_baseline.json`):

* peak 归一化后目标的**合并方差只有 0.0136–0.0174**(64×64 = 4096 像素);
* 常数预测的 MSE 只有 0.00053–0.00292,即比总方差小 2–20 倍;
* 于是 `R² = 1 - mse/var ≈ 1 - 0.00053/0.0154 ≈ 0.96`。

机制:peak 归一化把 0 级钉在 1.0、背景压到 ~0,**逐帧之间几乎不变**。
真正携带系数信息的散斑细节只占方差极小一部分。
换句话说,这个 R² 绝大部分在度量"图像本身有多静止",而不是"模型预测得多准"。

## 3. 正确的记分牌:`skill`

`skill = 1 - mse_model / mse_constant`,与地板无关。K=4 的 POD+ridge:

* 平均 skill **+0.0609**(sd 0.17),**10 折里有 2 折为负**(模型比常数还差);
* 逐折 skill 范围 **−0.241 … +0.290** —— 折间差异比模型间差异大一个量级。

K 几乎不敏感(K=4 → +0.9163,K=256 → +0.9144;配对差 −0.0019,p=0.64),
说明 peak 归一化后的图像矩阵**近似 rank-4**。

## 4. 泄漏排查(canary 通过)

因为这个结果"看起来像 bug",按仓库既有的 canary 风格做了对照
(`scripts/verify_pod_ridge_canary.py`,10 折):

| 对照臂 | val R² | 期望 |
|---|---|---|
| `real` 真模型 | +0.9163 | > 0 |
| `mean` 训练均值图(回归系数全 0) | **+0.9097** | ≈ 0 |
| `permuted` 训练图像随机置换 | **+0.9096** | ≈ 0 或更低 |

`mean` 与 `permuted` 都没有掉到 0,说明**确实没有泄漏** —— 是地板本身就有 0.91。
这一点很重要:它把"高 R²"从"模型强"重新归因为"目标几乎静止"。

## 5. 可操作结论:系数信息主要在**绝对亮度**里,不在形状里

`scripts/sweep_target_transform.py`,同一 10 折、同一 K=4 ridge、同一 skill 记分牌:

| 目标变换 | R² 模型 | R² 常数 | **skill** | 为负的折数 |
|---|---|---|---|---|
| **raw255**(保留绝对探测器电平) | +0.8866 | +0.7992 | **+0.2395** | 2/10 |
| **log**(`log1p(50I)/log1p(50)`) | +0.8709 | +0.8353 | **+0.2133** | 1/10 |
| peak(当前默认族) | +0.9259 | +0.9204 | +0.0654 | 2/10 |
| center(peak 后裁中央 32×32) | +0.9130 | +0.9107 | +0.0239 | 3/10 |
| **zscore**(逐帧去均值+去尺度) | +0.9498 | +0.9479 | **+0.0061** | 3/10 |

单调性很清楚:**每做一步逐帧归一化,就删掉一部分系数信息。**
`zscore` 把均值和尺度全去掉,skill 只剩 **+0.006** —— 几乎全部信息都没了。
而 `raw255` 保留绝对电平,skill 最高 **+0.240**,是 peak 的 3.7 倍。

两个副产物结论:
* `center` 裁剪**有害**(+0.024 < +0.065):信息不只在中心,晕区也有。
* `zscore` 的 R² 最高(+0.9498)但 skill 最低(+0.006)—— **R² 高与模型好无关**的
  又一个实例。

## 6. 对已有结论的纠正

### 6.1 撤回:image_mode 默认值不应该被改成 `peak`

2026-10-08 曾依据"5 seed × R²"把 `CoeffTrainConfig.image_mode` 从 `robust` 改成 `peak`
(peak +0.821 vs robust +0.628 vs abs255 −0.345)。**该论证现予撤回**,已回滚到 `robust`:

1. 那次比较**没有检查常数基线**。而常数基线随变换而动(peak 地板 +0.920 vs abs255 地板 +0.799),
   所以 +0.19 的 R² 差值可能整体落在地板移动量之内;
2. ConvNet 在 `abs255` 下 R² = −0.345,而常数基线在 abs255 下有 **+0.799** ⇒
   它的 skill 是 **−5.69**。所以 ConvNet 在 abs255 下确实是**条件数问题**(拟合不了
   2.9 个数量级的动态范围),**不是信息不足**。线性模型证明信息就在那里;
3. 真正该问的是"这个架构能否用上绝对亮度",而不是"peak 是否更好"。

已在 `train_coeff.py` 的字段注释与 `test_train_coeff.py` 里写明撤回理由与复测要求。

### 6.2 存疑:"U-Net 显著优于 physics"

配对检验本身没错(R² p=0.0117,SSIM p=0.0020),但它比较的两个模型都在常数地板之下或附近。
在 skill 口径下需要重跑才能断言优劣。

## 7. 权威协议下的重测:三个训练模型**全部不如常数**

`scripts/compare_models_cv.py`(10 折 file 协议,已加上 `skill` 口径;`logs/cv_skill.json`):

| 模型 | val R² | **skill** | R² 均值 sd |
|---|---|---|---|
| **常数基线** | **+0.9396** | 0 | — |
| unet (7.78M 参数) | +0.8986 | **−0.9938** | 0.059 |
| physics (n_max=15) | +0.8727 | **−1.2674** | 0.087 |
| hybrid | +0.8721 | **−1.4056** | 0.080 |

`skill < 0` 的含义是**模型的 MSE 比"所有样本都输出同一张固定图"还大**(unet 是常数的 1.99 倍)。
配对(sign-flip,折上配对,`ml/zernike/eval_stats`):

* unet − physics:skill 差 **+0.2736**,p=0.1465,d_z=+0.51 ⇒ **不显著**
* unet − hybrid:skill 差 **+0.4118**,**p=0.0254**,d_z=+0.84 ⇒ 显著

即:先前"U-Net 显著优于 physics(R² p=0.0117)"在 R² 口径下不复现;
在 skill 口径下 unet 与 physics **无法区分**,只有 unet 显著优于 hybrid(而 hybrid 本来就被记录为失败臂)。

## 8. 闭式线性基线同样赢过所有训练模型

`scripts/compare_linear_vs_trained.py` 把 **K=4 POD + ridge(闭式解)** 放到与上面
**完全相同**的管线(同一 1010 条语料、同一 `str(record.path)` 折、
同一 `MaterialiserConfig(grid=64)` 目标 + `_peak_normalise`):

| 模型 | skill |
|---|---|
| 常数基线 | 0 |
| **K=4 ridge** | **−0.4354**(10 折全部为负,−0.897…−0.057) |
| unet | −0.9938 |
| physics | −1.2674 |
| hybrid | −1.4056 |

闭式线性模型把 U-Net 的 skill 提高了 **0.56**。所以模型之间的差距不是"容量不够",
而是**优化/正则化把一个本该更容易的问题做坏了**。

### 8.1 这 0.03 差异已定位(不是缓存 bug,是两条不同的裁剪+归一化实现)

**排查过程**(`scripts/diagnose_peak_pipeline_gap.py`、`scripts/diagnose_peak_cache_key.py`):

1. `direct(peak)` 与 `abs255` 后手动 `/max` 有 **48.6% 像素不同**(max|d| = 0.358);
   两者代数上应恒等 ⇒ 差异来自别处。
2. **缓存被排除**:`cached(peak) == direct(peak)`、`cached(abs255) == direct(abs255)`
   都是 **0% 像素不同** ⇒ `.hw_cache` **确实按 `image_mode` 正确分键**,没有串味。
3. 突破口:**101/101 帧的 max 都由 >1 个像素同时取得**(平台顶)。
4. 根因在 `src/ml/hwdataset/transforms.py:555-566`:`"peak"` **提前 return**,
   走的是 `ml.gsnet_debug.offline.farfield_to_grid`;而 `abs255`/`robust`/`raw`
   走 `_anchored_window`。两条路径**裁剪原点与归一化尺度都不同**:

   | mode | 0 级定位 | 尺度 |
   |---|---|---|
   | `peak` | `zero_order_center(refine=False)` = 裸 `np.argmax`(`offline.py:531-534`) | 除以**单个**最亮像素 |
   | `abs255` / `robust` | `_anchored_window` 去尖峰(`despike_k=3`,AGENTS 记录) | `abs255` 除 255;`robust` 除亮像素中位数 |

   `peak` 的裁剪原点会被**热像素**带走(平台顶 + argmax 取首个索引 ⇒ 原点可能偏心),
   尺度也被热像素设定 —— 这正是 `transforms.py:525-528` 已经写下的理由
   ("`peak` removes the gain but divides by a **single pixel** … so peak mode is
   set by detector defects rather than by the beam")。本次给出了它的**量化后果**:
   换到 `peak` 会同时改变裁剪窗口与尺度标度,在常数基线上值约 **0.03 R²**。

**结论**:两个 `r2_const` 各自都对,只是**测的不是同一个估计量**。
`image_mode="peak"` 与"把 64×64 网格除以自己的 max"**不是同一件事**,
跨脚本比较 `r2_const` 绝对值前必须先确认 mode 与裁剪路径一致。

**副产品**:这一条独立地支持了 `train_coeff` 默认值回滚到 `robust` 的决定 ——
`robust` 存在的理由(热像素不得决定裁剪与尺度)现在有了实测支撑。


## 9. 试过并**否定**的一个假设:U-Net 的 sigmoid 输出头

> ⚠️ **本节结论已被自己的 A/B 推翻,保留在此作为"被推翻结论"存档**
> (AGENTS.md 要求记录实测推翻的假设及原因)。

**当时的假设**:`UNetGenerator.forward` 末尾是 `sigmoid(...)`,而目标是"大面积近 0 暗背景
+ 小面积亮核"(peak 归一化后 mean ≈ 0.014)。实测**未训练**模型的输出停在
`[0.5321, 0.5491]`(sigmoid 中点),与目标均值差一个数量级 ⇒ 推测 sigmoid 把输出钉在
饱和区、梯度消失,因此 778 万参数打不过 4 成分闭式 ridge。

**做法**:给 `UNetGenerator` 加一个线性输出头 `output_mode="image"`
(`src/ml/phase/unet.py`,默认仍是 `phase`),未训练输出确实从 mean 0.541 降到 0.047,
参数完全相同(7.78M)。然后在**同一 10 折、同一 seed** 上 A/B。

**实测结果(与假设相反)**:

| 输出头 | val R² | **skill** | per-fold skill 最差 |
|---|---|---|---|
| `sigmoid`(原默认) | +0.8986 | **−0.9938** | −2.817 |
| `image`(线性) | +0.8784 | **−1.6220** | **−3.591** |

配对(折上,`ml.zernike.eval_stats`):`sigmoid − image` 差 **+0.6282**,
**p=0.0059**(最小可达 0.0020),**d_z=+0.92** ⇒ **sigmoid 显著更好**。
线性头还更不稳定(sd 0.856 → 1.298)。

**假设错在哪**:我把"**未训练**网络的输出范围"当成了"训练后的行为"的证据。
输出层是线性的,50 epoch 训练足以把它移到任何位置;而 sigmoid 的**有界性**在这里
起了正则化作用,比初始化的量纲偏移更重要。**结论:输出头不是瓶颈,假设否定。**

副作用(有价值的副产品):`output_mode="image"` 作为通用选项保留,并在 docstring 里
写明**本语料上实测更差、不要默认使用**,以免下一个人重复这个实验。

### 9.1 第二个被否定的假设:损失空间(亮核主导梯度)

**假设**:目标是"大面积近 0 暗背景 + 小面积亮核"(mean ≈ 0.014),
未加权 MSE 会被亮核主导,暗背景(几乎全部形状信息)学不到 ⇒ 改损失空间应能提 skill。

**做法**(`scripts/diagnose_loss_geometry.py`):用**闭式 ridge**(排除优化噪声)在同一
10 折上比较三种损失空间,全部**在原强度空间打分**以便可比:

| 损失空间 | mean skill | min | max | mean mse |
|---|---|---|---|---|
| `mse`(现状) | −0.4132 | −0.6836 | −0.0809 | 0.001263 |
| `sqrt_mse`(振幅空间 \|E\|) | **−0.3808** | −0.6883 | −0.0819 | 0.001273 |
| `log_mse` | −0.3843 | −0.6614 | −0.1036 | 0.001279 |

**结论:假设否定。** 三者相差 < 0.04,全为负 —— 损失空间**不是**瓶颈。

### 9.2 剩下最可能的解释:信噪比地板(待验证)

两次假设否定后,剩下的关键事实是:

> **连"拟合得最好的"闭式 ridge 的 skill 也是负的(−0.41)。**

量化(`diagnose_r2_baseline.py`,peak 模式,10 折):

* 目标合并方差 `var ≈ 0.0137`
* 常数(train 均值)MSE ≈ **0.00089** ⇒ `R²_const ≈ +0.935`
* 最好的模型 MSE ≈ **0.00126** ⇒ `skill = 1 − 0.00126/0.00089 ≈ −0.41`

要打败常数,模型需要 `mse < 0.00089`。当前最好的模型是 0.00126。
**这更像是"每帧随系数变化的那部分信号,本身就小于测量噪声"**,
而不是模型没拟合好。若成立,则"提升 val 预测效果"的天花板由 SNR 决定,
容量/损失/输出头都不是杠杆 —— 这也解释了为什么 4 成分线性模型能赢 778 万参数的
U-Net(它离噪声地板更近,不是因为它"更对")。

**下一步该做的实验(还没做)**:**信噪比估计**。同一 pickle 内相邻 epoch 是优化过程中
连续两步,系数差异很小;若图像差异在"系数几乎相同"的相邻对与"系数差别大"的随机对之间
**没有统计差异**,说明系数驱动的信号已淹没在噪声里。用
`ml.zernike.eval_stats` 的配对 sign-flip 做这个对比,按仓库规则配对在**变化的那个轴**上。



1. **常数基线/skill 已接入两个 trainer 与 CV harness**(本轮完成):
   `ml.zernike.metrics.constant_baseline_metrics` 是唯一实现,
   `train_amp.evaluate()`、`train_coeff._evaluate()`、`compare_models_cv.py` 都在报 `skill`。
   附带修掉 `compare_models_cv.py` 内**重复实现的** sign-flip 与 Holm-Bonferroni
   (违反"统计唯一入口 `ml/zernike/eval_stats`"红线),改为委托。
2. **查清"为什么线性模型比 778 万参数的 U-Net 好 0.56 skill"** —— 这是当前最高价值
   的问题:若闭式解更好,说明优化/正则化/目标口径出了问题,而不是容量不足。
   候选排查顺序:(a) `train_amp`/`compare_unet_baseline` 的 `fit()` 是否 peak-normalise
   预测但用未归一化目标算 loss;(b) 学习率/轮数是否欠拟合(对比 physics 第 13 epoch
   就收敛的记录);(c) 目标是否被 `w_*` 物理项带偏。
3. **重测 ConvNet 在 `abs255` 下的 skill**(todo #4)。线性模型已证明绝对电平里有信息
   (`raw255` skill +0.240,而 `zscore` +0.006),而 ConvNet 的输入只有 Zernike 系数、
   没有曝光 —— 给它 `exposure_log10` 是本数据下最有希望的**架构**改动。
4. 只有 1–3 完成后才回到容量轴(6 个从未扫过的旋钮),并按 AGENTS.md 警告做**联合**复测。



## 8. 复现

```bash
python scripts/diagnose_pod_ridge.py --k 4 8 16 32 64 128 256   # logs/pod_ridge_full.json
python scripts/verify_pod_ridge_canary.py                        # logs/pod_ridge_canary.json
python scripts/diagnose_r2_baseline.py                           # logs/r2_baseline.json
python scripts/sweep_target_transform.py                         # logs/target_transform.json
python scripts/compare_linear_vs_trained.py --k 4                # logs/linear_vs_trained.json
python scripts/diagnose_peak_pipeline_gap.py                     # logs/peak_pipeline_gap.json
python scripts/diagnose_peak_cache_key.py                        # logs/peak_cache_key.json
python scripts/diagnose_loss_geometry.py                         # logs/loss_geometry.json
python scripts/diagnose_snr_ceiling.py                           # logs/snr_ceiling.json
python scripts/diagnose_label_alignment.py                       # logs/label_alignment.json
python scripts/diagnose_gain_drift.py                            # logs/gain_drift.json
python scripts/compare_models_cv.py --protocol file --models physics hybrid unet \
    --epochs 50 --out logs/cv_skill.json                         # 权威 10 折(含 skill)
python scripts/compare_models_cv.py --protocol file --models unet \
    --unet-output-mode image --epochs 50 --out logs/cv_unet_image.json   # 输出头 A/B
```

---

# 判决:信噪比天花板(已测,非推断)

`scripts/diagnose_snr_ceiling.py`。判据用一个**没有容量、没有优化器、没有损失函数**的模型:
系数空间里的 **k 近邻**(训练集标准化,预测 = k 个邻居图像的均值),同样在 10 折上打分。

| k | mean skill | min | max |
|---|---|---|---|
| 1 | −3.747 | **−50.518** | −0.591 |
| 3 | −1.507 | −28.138 | −0.306 |
| 5 | −0.534 | −13.061 | −0.283 |
| 10 | −0.366 | −8.716 | −0.208 |

**k-NN 也全为负,且随 k 单调变好** —— 而 k→∞ 时 k-NN 退化成常数预测器、skill→0。
**"越接近常数越好"就是"没有可恢复信号"的定义。**

配对(折上):k=3 − k=1 = **+3.064,p=0.0020(最小可达)**;k=10 − k=1 = **+6.010,p=0.0020**。

## 信号-分离度曲线(同 pickle 内配对)

相邻 epoch 是同一次优化里的连续两步,系数几乎相同 ⇒ 其图像差就是**噪声地板**。

| \|Δc\| | 配对数 | mean\|ΔI\| | rms\|ΔI\| |
|---|---|---|---|
| [0.0, 0.5) | 1672 | **0.0226** | 0.0284 |
| [0.5, 1.0) | 1944 | 0.0263 | 0.0332 |
| [1.0, 2.0) | 3880 | 0.0281 | 0.0348 |
| [2.0, 4.0) | 8204 | 0.0298 | 0.0378 |
| [4.0, 8.0) | 16092 | 0.0389 | 0.0467 |
| [8.0, ∞) | 18708 | 0.0326 | 0.0446 |

* **噪声地板(\|Δc\|→0)= 0.0226**;系数相距很远时 \|ΔI\| 也只有 **0.033–0.039**;
* 振幅域上可恢复的信号 ≈ √(0.035² − 0.0226²) ≈ **0.027**,与噪声 0.0226 同量级
  ⇒ **信噪比约 1.2 : 1**;
* 系数距离增大 **16 倍**,图像差只增大 **1.4 倍**,且最后一个 bin 还回落
  (0.0389 → 0.0326)。若是干净信号,这条曲线应当单调上升。

## 标签完整性:已排除错位

`scripts/diagnose_label_alignment.py`(**完全不需要模型**)。相邻 epoch 是同一次优化的连续两步:

| 量 | 相邻/随机 比值 |
|---|---|
| **系数距离** | **0.0563** —— 相邻 epoch 比随机对近 **18 倍** ⇒ `_c` 是一条真实的光滑轨迹 |
| **图像差异** | **0.7741** —— 图像只近了 23%,远不及系数 |

配对(10 个 pickle):**图像比值 − 系数比值 = +0.7178,p=0.0020(最小可达),d_z=+5.95**。
且 `|ΔI|` 对 `|Δc|` 的回归斜率在 **10/10 个 pickle 全为正**(均值 +0.000822)。

⇒ **标签是对齐的**(若错位,斜率应处处为零)。图像确实携带了系数不决定的大分量。

## 逐帧增益漂移:不是主因

`scripts/diagnose_gain_drift.py`。逐帧拟合 `I_i ≈ a_i·mean + b_i`(只用训练集拟合,不泄漏):

* **逐帧增益的离散度只有 sd/mean ≈ 0.048–0.058**,即约 **5%** —— 逐帧增益漂移很小;
* 去掉逐帧增益+偏置后(ridge 臂):`none −0.4133 → affine −0.3570`,
  配对 **+0.0563,p=0.0742(不显著),d_z=+0.63**,且 **skill 仍为负**;
* k-NN 臂同向但同样不显著(p=0.0840)。

⇒ **主因不是可由数据处理去掉的标量漂移,而是真正的噪声地板。**

(脚本的 verdict 文案已按证据收紧:先前写成"helps"是过度乐观,现在只有
"显著 **且** skill 转正"才判为 fix。)

## 最终判决

在 `slm_zernike_shaping` 这份语料上,用户要求的三类改动**全部试过、全部无法提升 val**:

| 类别 | 试了什么 | 结果 |
|---|---|---|
| **数据处理** | `image_mode` abs255/peak/robust;目标变换 raw255/log/center/zscore;逐帧增益+偏置去除 | skill 全为负;最好的一档 +0.24(raw255)但仍远低于 0;逐帧增益去除 p=0.074 不显著 |
| **模型结构** | 输出头 sigmoid vs 线性(配对 A/B);容量 K=1…256;conv/MLP 已知 | **线性头显著更差**(p=0.0059);闭式 4 成分 ridge 反而赢 U-Net 0.56 |
| **训练方法** | 损失空间 mse/振幅/log;优化器/轮数沿用既有 | 三者相差 <0.04,全为负 |
| **无模型对照** | k-NN(k=1/3/5/10) | 全为负,且随 k 单调趋近 0 ⇒ 越接近常数越好 |

**原因已量化,不是调参问题**:噪声地板(|Δc|→0 时 |ΔI|)≈ **0.0226–0.031**,
而系数能解释的信号 ≈ **0.008** —— **信噪比约 1:4,方向相反**。
在这个信噪比下,任何模型的 MSE 都必然高于"输出一张固定图"的 MSE,skill 必然为负。

**因此:提升 val 预测效果的杠杆在数据采集侧,不在模型侧。** 具体是
同相位重复取帧平均(压低 0.0226 的地板)、曝光归一、漂移拒绝。
这些都需要**重新采数据**,无法从现有语料里做出来。

### 若仍要继续,建议的顺序

1. **重新采一组"同相位重复帧"数据**(每个相位连拍 N 帧)。这是唯一能把噪声地板
   压到信号量级以下的办法;在现有语料上任何建模工作都做不了这件事。
2. 采到之后先跑 `scripts/diagnose_snr_ceiling.py` 确认地板降到 ~0.008 以下,
   再谈模型。
3. **口径澄清**:若业务只需要"由系数预测图"的点估计,常数基线 + skill 口径已经够了;
   若需要"由系数反推可下发相位",那是**逆设计**(`ml/zernike/inverse_design.py`)，
   与本报告测的**正向预测**不是同一个任务,不应共用一个 R² 记分。
