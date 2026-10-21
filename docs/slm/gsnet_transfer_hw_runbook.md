# FourierGSNet 迁移学习：实验系统操作流程与调参指南

> **这份文档回答两件事**：
> 1. 在实验系统上把「仿真预训练的网络」迁移到真实实验系统并交给闭环用，**具体怎么做**（可逐条执行）。
> 2. 这条路上**哪些参数要盯、盯什么、怎么判**（光路 / 预处理与数据 / 模型结构三组）。
>
> **它不回答**：网络为什么这么设计、仿真里那些数字怎么来的——那些在
> [`report/gsnet_2f_calibration/report.md`](../../report/gsnet_2f_calibration/report.md)。
> **`--init-from-gsnet` 这个开关本身的门控语义**在
> [`docs/slm/gsnet_init_hw.md`](gsnet_init_hw.md)。
>
> **执行前必须知道的一件事**：迁移学习的**在机一步**需要 Santec SLM + 相机。
> 本仓库的开发容器里没有这两样设备（没有 `/dev/ttyUSB*`，Daheng 的 `gxipy` 是
> Windows SDK），所以**这一份流程要在装了实验系统的机器上跑**。
> 离线路径（`--from-dir`）在任何机器上都能验证，可以先用它把脚本链路走一遍。

---

## 0. 前置条件

| # | 条件 | 怎么确认 |
|---|---|---|
| 1 | SLM 与相机在线，且**没有残留的 python 进程**占着 USB | `Get-Process python*`（Windows）/ `ps aux | grep python`（Linux） |
| 2 | 激光已开、功率稳定 | 阶段 0 的 drift 探针 |
| 3 | 曝光上限已按会话设好 | `AO_DAHENG_MAX_EXPOSURE_MS` 环境变量 |
| 4 | 预训练 checkpoint 在手 | `data/gsnet_pretrain/<ts>/checkpoint.pt`（格式与 `ml.gsnet.train` 一致） |
| 5 | **门 1 的阈值已经改过** | 见 §5.1——不改的话这一步必然空手而归 |

---

## 1. 阶段 0：实验系统表征（**不要跳过**）

本实验系统的三条实测事实决定了这一步不能省：固定等待时间无效（驱动按灰度图
变化量估算翻转时间，会报 0.0 ms）、面板**保留上一次显示的图案**、0 阶**不在画幅中心**。
完整指南见 [`docs/slm/pre_run_characterization.md`](pre_run_characterization.md)。

```bash
# 1) 漂移 + 曝光线性（定曝光档位）
python -m ao_shaping.tools.ccd.drift_probe \
    --cam-type daheng --cam-id 0 --exposure-ms <现定> \
    --n-drift 24 --drift-period-s 5.0 \
    --exposure-ladder 0.2,0.3,0.4,0.6,0.8,1.0,1.25,1.5,2.0 \
    --ladder-repeats 2 --out data/slm_drift

# 2) 本底 / 稳定时间 / SNR-vs-K（定「多大的信号才看得见」）
python -m ao_shaping.tools.ccd.floor_probe \
    --cam-type daheng --cam-id 0 --exposure-ms <现定> \
    --out data/slm_floor

# 3) 稠密随机相位测不测得出来（定探针幅度）
python -m ao_shaping.tools.zernike.abba_probe \
    --cam-type daheng --cam-id 0 --slm-number 1 --exposure-ms <现定> \
    --out data/slm_abba

# 4) 光功率门（没有光就没有可整形的东西）
python scripts/slm/diagnose_light_level.py \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --exposures 0.02,0.05,0.1,0.2,0.4
```

**记下三个数，后面要用**：
- 可用曝光档位区间（drift 探针的 `verdict == "monotonic"` 且 `saturated == False`）；
- 噪声底（floor 探针）与 SNR 曲线；
- **随机相位探针的最小可分辨幅度**（abba 探针）——它决定 `--probe-spread` 能取多小。

> 曝光是**会话内现定**的：本实验系统 8 分钟内功率可漂 21 %，写死的曝光档位
> 在换一天之后就不再成立。`--exposure-time-ms` 接受 `default` / `auto` / 一个具体
> 毫秒数；**`0` 会被拒绝**（它既不是 `default` 也不是 `auto`）。`auto` 从驱动
> 自己的 ~0.02 ms 起步，**必须先用 drift 探针 bracket 出区间**再信它。

---

## 2. 阶段 1：采集（真实实验系统上的「相位 → 帧」对）

迁移学习的监督信号是**现成的**：model-in-loop 每做一次探针测量就把
`probes/*.npz` 落盘（命令过的 `phase` ＝ 标签、映射到远场模型网格的
`model_grid` ＝ 目标），外加 `source_amplitude.npy`（＝ 网络的 source 输入）。

所以采集这一步**不需要新代码**：跑一次带 `--debug` 的 model-in-loop 即可。

```bash
python src/ao_shaping/main.py --debug slm-model-in-loop \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --exposure-time-ms <现定> \
    --region 128 --far-field-padding 8 --cam-size 300 --target-side 100 \
    --n-rounds 3 --probe-count 8 --holdout-probe-count 2 \
    --step-a-iterations 40 --step-b-iterations 200 \
    --dir data
```

采集完确认这三样东西存在（缺任何一样，微调都读不了）：

```bash
ls data/debug/slm_model_in_loop/<ts>/probes/*.npz   # phase / model_grid / frame
ls data/debug/slm_model_in_loop/<ts>/source_amplitude.npy
python -c "import json;print(json.load(open('data/debug/slm_model_in_loop/<ts>/probes/manifest.json'))[:3])"
```

**`probes/manifest.json` 里要看的**：每帧的 `exposure_ms`（判欠曝/过曝）、
`family`（探针族）、`correlation`（这一帧与正向模型的相关性——**接近 0 的帧要剔掉**，
它是噪声，留在训练集里只会教网络去拟合噪声）。

---

## 3. 阶段 2：迁移学习（微调）

```bash
# 在机：脚本自己会跑一次 model-in-loop 去采数据，然后微调
python scripts/slm/gsnet_transfer.py \
    --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
    --hardware --capture-rounds 3 --capture-probes 8 \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --exposure-time-ms <现定> --region 128 --far-field-padding 8 \
    --cam-size 300 --target-side 100 \
    --epochs 20 --lr 1e-4 --batch-size 8 \
    --out data/gsnet_transfer/<ts>

# 或者离线：拿阶段 1 已经采好的那段运行
python scripts/slm/gsnet_transfer.py \
    --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
    --from-dir data/debug/slm_model_in_loop/<ts> \
    --epochs 20 --lr 1e-4 --out data/gsnet_transfer/<ts>
```

**看两个数**：

| 指标 | 含义 | 判据 |
|---|---|---|
| `phase_loss` | 相位项（对命令过的相位） | **随机猜测水平 = π²/3 ≈ 3.290**。低于它才算学到；一直贴着它＝「还在瞎猜」 |
| `shaping_loss` | 整形项（用标定前向把预测相位推到相机那张帧上） | 越低越好；它同时受探测器与模型误差影响，所以没有绝对门槛 |

**读法**：`phase_loss` 贴着猜测水平，**多数情况下是数据量不够**，不是「这条路走
不通」——探针相位是随机的，由一张远场帧反解出那个相位本身是**病态逆问题**，
需要几十帧以上才开始有信号。

产物：`data/gsnet_transfer/<ts>/phase.npy`（`(region, region)` 原始弧度）＋
`checkpoint.pt`（已微调）＋ `summary.json`。

---

## 4. 阶段 3：闭环 A/B（这一步才是「有没有用」）

把微调后的相位交给闭环，与「零相位起步」做**交错重放**对比：

```bash
# 先看计划（不打开设备）
python scripts/slm/gsnet_init_hw_ab.py --plan \
    --gsnet-phase data/gsnet_transfer/<ts>/phase.npy --pairs 3

# 正式跑
python scripts/slm/gsnet_init_hw_ab.py \
    --gsnet-phase data/gsnet_transfer/<ts>/phase.npy --pairs 3 \
    --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
    --exposure-time-ms <现定> --target-side 100 --cam-size 300 \
    --region 128 --far-field-padding 8
```

**协议要点**（`gsnet_init_hw_ab.py` 已实现，不要自己改）：

- 同一会话内交错：两臂交替，第 i 对的先后按 i 的奇偶翻转，抵消「先跑的那一臂
  占了更好的光功率时段」；
- 每次 MIL 运行前都过**光功率门**；
- 统计走 `ml/zernike/eval_stats.paired_comparison`（精确 sign-flip p 值 + `d_z` +
  结构性的 `min_attainable_pvalue`）。

**判据**：

- 报**配对均值差 + p 值**，**不是**单轮 best 分数——本实验系统分数分布呈双峰
  （25.7k / 33.6k 两簇，见 `report/shape_trials/gs_refine_live_20261012.md`），
  单轮 best 不得作为可复现增益声明；
- `n_pairs` 小的时候（< 4）报效应量即可，`min_attainable_pvalue(4) = 0.125`，
  结构上到不了 p < 0.05；
- 同时记录每一臂的 `init_from_gsnet_status`：**`adopted` 的比例本身就是结果**
  （门 1/门 2/门 3 各挡掉多少，见 §5.1）。

---

## 5. 阻塞点与故障模式

### 5.1 🔴 门 1 按 issue 原文是死的（**实验前必须先改**）

issue #165 写「信任域 L2 ≤ 0.5 rad」。但仓库里的 `trust_region_c_l2` 限的是
**Step A 的像差系数变化量**，不是**绝对瞳面相位**。

仿真实测（`report/gsnet_2f_calibration/report.md` §5.2）：

- 闭环自己产生的最优相位 **rms = 1.71 rad**；
- GSNet 给出的整形相位同量级（1.8–2.1 rad）；
- 用 0.5 rad 去卡绝对相位 ⇒ **这个开关永远不会生效**，A/B 会得到「两边完全一样」。

三种改法（选一种，写进 issue）：

1. 门 1 改成卡**相位与零/平场之差**的 L2；
2. 或卡**相位相对闭环自身解的偏离**；
3. 或给网络初值单开一个阈值（如 `init_from_gsnet_trust_region_c_l2`），与
   Step A 的系数步阈值解耦。

门 2（实测优于平场）与门 3（验证集 Pearson ≥ 0.5）工作正常，**不需要改**。

### 5.2 故障模式对照表

| 现象 | 多半是 | 怎么办 |
|---|---|---|
| 每一臂都是 `trust_region_rejected` | 门 1 按绝对相位卡（§5.1） | 改门 1 定义 |
| 每一臂都是 `holdout_correlation_rejected` | 探针幅度在噪声底之下 / 几何没标定 / 曝光不对 | 提 `--probe-spread`；查 `bench_geometry.json` 的 correlation；重跑阶段 0 |
| 几何求解 correlation < `--min-geometry-correlation` | 探针根本没被分辨出来 | 同上；或加 `--n-calibration-probes` |
| 微调后 `phase_loss` 仍 ≈ 3.29 | 数据量不够（病态逆问题） | 加采集轮数/探针数；先看 `shaping_loss` 有没有降 |
| 微调后 `shaping_loss` 也不降 | 前向算子与实验系统不匹配 | 查 `source_amplitude.npy` 与几何；把 `min_holdout_correlation` 调高看它先在哪一档被挡 |
| 帧全黑 / 全白 | 曝光档位或增益 | 阶段 0；`AO_DAHENG_MAX_EXPOSURE_MS` |
| 帧里找不到目标方框 | ROI 没锚在光斑上（0 阶不在画幅中心） | 用 `argmax` 锚定，别用画幅中心 |
| 预训练验证集 composite 远低于标签上限 | 预训练没学够 | 看 `phase_loss` 是否单调下降；不够就加样本/轮数 |
| 预训练 `phase_loss` 往上爬 | 学习率过大或整形项没 warm-up | `--lr` 降到 1e-4；`--warmup-epochs 3`；`--w-shaping 5` |

---

## 6. 要盯的参数（三组）

### 6.1 光路

| 参数 | CLI | 默认 | 它在换什么 | 怎么判 / 怎么调 |
|---|---|---|---|---|
| `region` | `--region` | 64 | 瞳面模型网格边长。越大，瞳面上每像素代表的面积越小、远场越细，但探针与拟合的参数量线性涨 | 看 `bench_geometry.json` 的 `region` 与标定 correlation；网格大而 correlation 掉，先降回来 |
| `far_field_padding` | `--far-field-padding` | 8 | **远场过采样倍数 —— 「远场被不被解析」的开关**。8 倍时 64 瞳面 → 512 远场，衍射极限光斑 **0.51 px**（没解析开） | 算一下：远场光斑 FWHM（model px）应当明显大于 1 px，否则网络内部的远场是混叠的。要 ~20 px 量级就得加大这个数（代价是 FFT 变慢） |
| `target_side` | `--target-side` | 60（**相机像素**） | 目标方框边长 | 必须 < `cam_size`；相对实测光斑太大 ⇒ 绝大部分能量在框外，EE 上不去 |
| `cam_size` | `--cam-size` | 300 | 相机开窗 | ≥ `target_side` + 余量；开窗越大噪声越大 |
| 曝光 | `--exposure-time-ms` | `default` | 曝光档位 | 阶段 0 的单调区间；**会话内现定**，`0` 被拒绝 |
| `probe_count` | `--probe-count` | 8 | 每轮探针数 | 探针越多，Step A 拟合的像差越可信（**验证集 Pearson 随它涨**） |
| `probe_spread` | `--probe-spread` | 4.0 | 探针相位 std（rad） | 必须明显大于 abba 探针量出的「可分辨幅度」，否则探针被噪声吃掉 |
| `holdout_probe_count` | `--holdout-probe-count` | 2 | 验证集探针数 | 验证集 Pearson 的可信度随它涨；它也是门 3 的判据 |
| `min_geometry_correlation` | `--min-geometry-correlation` | 0.5 | 几何求解接受线 | 标定结果的 correlation 低于它就中止 |
| `min_holdout_correlation` | `--min-holdout-correlation` | 0.5 | 门 3 的硬门控 | 轮次表里的 `holdout_correlation` |
| `trust_region_c_l2` | `--trust-region-c-l2` | 0.5 | **门 1**（见 §5.1） | 绝对相位口径下会挡掉一切 |
| `w_uniformity` / `w_efficiency` | `--w-uniformity` / `--w-efficiency` | 0.4 / 0.6 | 复合分数里均匀度 vs 能量的权重 | 目标偏均匀就调高 `w_uniformity` |
| `max_cv_regression` | `--max-cv-regression` | 0.02 | 单轮允许的 CV 倒退 | 轮次表里的 `cv_degraded` |
| 几何量 | `--pupil-center` / `--panel-span-px` / `--slm-pixel-um` / `--camera-pixel-um` | — / 900 / 8.0 / 2.2 | 瞳面中心、视场宽度、像元 | **与标定结果不一致会让所有远场指标失真**；以 `bench_geometry.json` 为准 |
| `n_orders` | `--n-orders` | 10 | Step A 拟合的 Zernike 阶数（66 项） | 拟合残差集中在高阶 ⇒ 该加阶数 |

### 6.2 预处理与数据

| 项 | 在哪 | 默认 | 它在换什么 | 怎么判 |
|---|---|---|---|---|
| 探测器位深 | `CCDSensor(bit_depth=…)` | 12 | 量化台阶 | 与相机真实位深一致（Daheng 可选 8/10/12/16） |
| 黑电平 | `CCDSensor(black_level=…)` | 0.0 | 下端截断（欠曝那一端） | 相机暗场读数 |
| 满阱 | `CCDSensor(full_well=…)` | 1.0 | 上端截断（过曝那一端） | 相机饱和读数 |
| 散粒/读出噪声 | `CCDSensor(shot_noise=…, read_noise=…)` | 0.02 / 0.004 | 帧的信噪比 | **用 floor 探针量出来的数填**，不要猜 |
| 曝光随机化范围 | `gsnet_pretrain.py` | log 均匀 [0.05, 3] | 训练集覆盖的曝光跨度 | 应覆盖实验系统用到的档位跨度；欠曝档的帧几乎全是噪声，会污染标签 |
| 像差随机化 | `random_aberration()` | Zernike 随机模式子集，rms 0.15–0.9 rad；自由形状 rms 0.02–0.15 rad | 网络见到的「实验系统族群」 | **应覆盖实验系统标定出来的像差量级**；一族像差训出来的网络换实验系统就废 |
| 目标的边长/位置随机化 | `draw_spec()` | 边长 0.6–1.4×；位置 ±1.2 % | 目标的形状/尺寸多样性 | 位置偏移**不能大**（网络预测的远场起点在画面中心，偏移大了损失窗口里没信号） |
| 标签定义 | `solve_phase()` | GS 朝**理想**方斑的解 | 网络学的是「把探测器反演回去」 | **不要**改成朝相机那一帧的解（那会让网络学成复制噪声） |
| 样本数 | `--samples` | 2000 | 覆盖实验系统族群的广度 | 生成瓶颈是每样本一次 512² 的 GS（32 进程约 100 样本/秒） |
| 损失权重 | `--w-shaping` / `--warmup-epochs` | 5 / 3 | 整形项相对相位项的权重与爬升 | 整形项一上来就 40 倍会把训练带炸（实测第 5 轮冲到 18） |
| 学习率与衰减 | `--lr` | 1e-4（1/4 处衰减） | 收敛 | 相位项**往上爬**就是太大 |
| 帧的清洗 | `probes/manifest.json` | — | 剔掉无信息的帧 | `correlation` 接近 0 的帧是噪声，留着只教网络拟合噪声 |
| 训练/验证切分 | `--val-samples` | 64 | 泛化估计 | 验证集要**独立**采集（同一次运行的探针不能既训练又验证） |

### 6.3 模型结构

| 参数 | CLI | 默认 | 它在换什么 | 怎么判 |
|---|---|---|---|---|
| 展开层数 | `--num-layers` | 8 | GS 迭代的深度。层数≈经典 GS 的迭代数 | 更多层更接近经典 GS；层数不够时网络「学」不动 |
| 通道宽度 | `--base-channels` | 24 | CNN 宽度 | 训练不稳就加宽；过拟合就减 |
| 跨层共享权重 | `shared_weights` | True | 参数量 | 共享 ⇒ 8 层只有 1 份权重；不共享则参数 ×8 |
| **瞳面网格 vs 远场网格** | `--region` / `--far-field-padding` | 64 / 8 | **远场够不够被解析**（见 §6.1） | 这是网络**能不能表达**好相位的决定项，不是容量问题 |
| 条件张量的降采样 | `model.py::_match_grid` | 远场 → 瞳面 36×36 窗口 | CNN 看到的远场分辨率 | 远场比瞳面细 ⇒ CNN 只看到降采样后的远场；物理步本身仍在全分辨率上做 |
| 预训练规模 | `--samples` / `--epochs` | 2000 / 24 | 预训练覆盖的广度与深度 | 验证集 composite 对照「标签自己的上限」（见报告 §7.2） |
| 微调预算 | `--epochs` / `--lr` | 20 / 1e-4 | 迁移的强度 | 微调过猛会**忘掉**预训练（验证集 composite 掉） |

---

## 7. 实验记录清单（写进报告用）

一次完整的迁移实验至少要留下：

1. **阶段 0 的三个数**：可用曝光区间、噪声底、随机相位可分辨幅度；
2. **采集**：探针帧数、每帧的 `exposure_ms` / `family` / `correlation`、剔掉了哪些帧；
3. **微调**：`phase_loss` 的**首末值**与随机猜测水平 3.290 的对比、`shaping_loss` 的首末值、
   训练样本数；
4. **A/B**：每臂的 `init_from_gsnet_status`、配对均值差、p 值、`d_z`、
   `min_attainable_pvalue`、两臂的 CV/EE/复合分数；
5. **环境**：相机型号与序列号、SLM 控制器编号、波长、曝光档位、
   `bench_geometry.json` 的内容。

> 报告口径提醒：单轮 best 分数**不得**作为可复现增益声明（双峰分布）；
> 跨会话的数字**不得**直接比较（功率漂移 8 分钟可达 21 %）。
