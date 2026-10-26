# 2f 实验系统实测协议 — U-Net 相位从孪生搬到实验系统

本目录是**预注册协议**。判据在见到任何实验系统数据之前写定, 之后不改; 实测不到就记失败 + 诊断,
不放宽阈值。

## 0 这个实验只回答一个问题

已入库报告 [`report/sim2f_shaping/closed_loop_20261025/report.md`](../../report/sim2f_shaping/closed_loop_20261025/report.md)
§5 的结论是: 在**标定配置的数字孪生**上, U-Net 相位在 7/7 族输给零相位 (`flat`)。该报告 §9.1 自陈
「这是匹配模型结果, 不是 sim-to-real」。

本实测回答: **把同一个 U-Net 相位下发到真实 2f 实验系统上, `net` 是否仍然输给 `flat`?**

不回答的: 闭环 (probe/refit/oracle)、摊销差距、绝对光度是否正确、判据是否该换。

## 1 被测对象 (冻结, 不重抽不重训)

| 项 | 值 |
|---|---|
| checkpoint | `runs/sim2f_fair/train/model_best.pt` (487,874 参数) |
| 语料 | `runs/sim2f_fair/corpus` 的 `target` / `support` / test split |
| 样本 | test split 8 个; 另支持 `--samples loop` 复现报告 §5 那 24 个样本 |
| 条件 | `flat` = 全零相位; `net` = 网络相量经 `from_phasor` 的弧度 |

### 1.1 相位到面板的映射 (唯一正确写法)

孪生 `Sim2f.field()` (`src/ml/sim2f/optics.py:674`) 直接把 `(B, grid, grid)` = 128 格相位 pad 到
`fft_n=512` 后做 FFT, **全程没有面板重采样** ⇒ `region_span_px=900` 只用于给 pupil pitch 定标
(900 px × 8 µm / 128 = 56.25 µm = 7.03 面板像素/格)。孪生把 128 格当作 **900 px 瞳面区域**的采样。

所以实验系统侧:

```
net (128,128) 相量 --from_phasor--> 弧度
  --compose_pupil_dof(base=zeros(1200,1920), dof=phase, panel_shape=(1200,1920), radius_px=450)-->
整面板 1920×1200 弧度 (区域外为 0 相位, 居中)
  --display_phase(slm, ..., settle)--> 内存槽轮换写屏
```

`pupil_side(450) = 900`。**不要**用 `pupil_fill = 0.75` 推出的 675 px 圆盘 —— 那是网络把相位驻留在
`(1,0)` 的掩膜, 不是照明口径 (孪生入射光是纯高斯, 无硬光阑)。

#### 修正记录 (2026-10-26, 追加; 原文保留)

原 §1.1 只写了「居中」, 没有区分「面板中心」与「照明瞳面中心」, 也没有记录 128→900 重采样的
坐标约定。两处补充如下, **均在任何实验系统数据存在之前**由孪生干跑确定, 判据与阈值未改。

1. **瞳面中心必须可指定。** 孪生把照明中心放在 128 格中心, 而该格映射到 900 px 区域的中心,
   所以面板上要对准的是**照明瞳面**中心, 不是面板中心。`build_panel_phase(..., center_xy=)` /
   `HwInstrument(..., pupil_center_xy=)` / CLI `--pupil-center-xy X,Y` 现可指定; 缺省仍是面板
   中心 (即「光束在面板上居中」这一假设)。记录进 `session.json` 的 `pupil_placement`
   (含 `pupil_center_is_default`)。**放错位置的后果不是光斑平移, 而是远场错误** —— 光束看到的是
   相位图的偏轴切片。
   每行另加**非门禁诊断** `zero_order_offset_px` (实测零级质心 − 孪生预测零级质心): 孪生的
   `AberrationState.tilt_px` 是模型倾斜、不是实验系统的, 所以只作诊断, 不入门禁 (G0-2 不变)。

2. **128→900 的重采样约定已实测, 不再当作「应该没问题」。** `compose_pupil_dof` 走
   `upsample_bilinear_to_shape` (`scipy.ndimage.zoom(order=1)` + `fit_exact`), 是**端点对齐**
   约定: 输入第 0 格落到输出像素 0, 第 127 格落到像素 899。孪生的 DFT 约定把第 `i` 格放在瞳面
   `i·p` (`p = pupil_pitch_m = 56.25 µm`, 128 格跨 `127·p = 7143.75 µm`), 也就是**像素中心**
   网格: 第 `i` 格位于其 `900/128 = 7.03125` px 块的中心。端点对齐把第 `i` 格放在
   `i·899/127 = i·7.07874` px, 于是嵌入后的等效节距是 **899/127 = 7.07874 px/格**,
   而不是 7.03125:

   - 节距比 `(899/127)/(900/128) = 1.006755` ⇒ 相位图绕区域中心被**拉伸 +0.6755 %**
     (是缩放, 不是平移: 第 0 格与第 127 格被拉向**相反**方向, 区域中心是不动点);
   - 远场的后果是**缩放**: 焦面图样按节距比的倒数缩小 `1/1.006755 = 0.993291`,
     即 **−0.671 %**。

   | 量 | 实测值 |
   |---|---|
   | 最大位移 (区域两边缘, 方向相反) | **3.0156 面板像素** |
   | 平均位移 | **1.5095 面板像素** |
   | 区域中心处的位移 | < 0.05 像素 (不动点) |
   | 嵌入后的等效节距 | **7.07874 px/格** (= 899/127, 对比 7.03125) |
   | 瞳面节距比 | **1.006755** ⇒ 拉伸 **+0.6755 %** |
   | 远场后果 | **−0.671 % 缩放** (= 1/1.006755) ⇒ 在最细特征 (24 CCD px) 上 **0.161 CCD px** |

   **这是缩放误差, 不是平移。** 平移一张**整形**相位图并不会平移焦面图样 —— 它改变的是相位
   图的局部梯度 (`−δ·∇φ`), 所以「瞳面位移 = 焦面位移」在这里是错的。正确的链条是: 瞳面节距比
   1.006755 ⇒ 焦面按倒数缩放 −0.671 %; 语料最细特征是 `SynthConfig.min_feature_px` = 24 CCD px,
   故残差 `0.00671 × 24 = 0.16` CCD px。

   像素中心约定能把这个 0.6755 % 完全消掉, 但 `compose_pupil_dof` 是**全仓标准的瞳面嵌入**
   (GS 与 transfer 线用的是同一个函数), 所以**保留标准嵌入**, 只把残差记录下来, 不再造第二套
   嵌入。重采样器若改动, 必须重新实测 (`tests/ao_shaping/ml/sim2f/test_bench.py::TestResampleConvention`
   把上述数字钉住)。

#### 修正记录 (仪器契约; 追加, 原文保留)

`Instrument` 契约在 `capture` / `close` / `describe` 之外增加了一个成员 `twin(phase_rad)`,
返回孪生对同一指令的预测帧。原因: G0-4 的定义就是 `ncc(实测 net 帧, 孪生预测 net 帧)`,
没有预测帧这一门无法求值。同样在任何实验系统数据之前、于孪生干跑上确定; 判据与阈值未改。
`HwInstrument.twin` 用**未注入任何台面状态**的标定孪生 (真实像差不可知, 而 G0-4 是宽松的
绝对几何门), `SimInstrument.twin` 就是它自己的 capture (所以干跑时 G0-4 恒为 1.0 —— 该门是给
实验系统的)。

## 2 条件与顺序 (预注册)

每个样本固定四次测量:

```
A1(flat) -> B1(net) -> B2(net) -> A2(flat)
```

- 每次写屏都轮换内存槽 (`display_phase` 内部 2..125 排除当前槽), settle 0.3 s。
- `A1`/`A2` 是**同一条件的重复**, 其差异即该样本的**噪底**; 这是本次唯一的分辨率来源, 不是额外指标。
- 顺序固定为 A–B–B–A (不随机化): 热漂移与液晶响应是主要混淆项, 固定顺序让漂移在 A→A 与 B→B 的差里
  对消。条件顺序随机化会把这个混淆项变成不可估的方差。
- 曝光、ROI、增益、settle **全程不变**; 中途改任何一个都会作废该会话。

## 3 判据 (先写后测)

主判据: 逐样本 `ncc(frame, target)`, 其中 `prepare_frame` 先减去帧中位数并截断负值
(与报告同一实现: `ml.sim2f.losses`)。

**「net 在实验系统上恢复优势」需要两条同时成立**:

1. `ncc_net > ncc_flat` 的样本数 ≥ 6 / 8 (**计入的样本**);
2. `mean(ncc_net − ncc_flat) > 0.05`。

其余一切结果记为**未恢复**, 附诊断, 不改阈值。

副判据 (只作对照, 不参与判定): `objective = L1 + 0.25·(1−EE)`, 与报告 §5 同式同 `w_ee`,
`prepare=True`。

**噪底剔除**: 若 `|ncc_net − ncc_flat| < 3 × |ncc_A1 − ncc_A2|`, 该样本标 `indistinguishable`,
既不进 (1) 的分子也不进分母。被剔除的样本数必须一并报告 —— 若剔除后计入样本 < 6, 主判据**未达效**,
记「功效不足」而不是「未恢复」。

## 4 前置门 (Gate 0): 不过就不报主判据

| 门 | 条件 | 不过的含义 |
|---|---|---|
| G0-1 曝光 | `flat` 帧峰值落在 [0.2, 0.8] × 满量程, 饱和像素 < 0.1 % | 曝光选错, 所有比较无效 |
| G0-2 零级稳定 | 全部帧零级质心 (`argmax` + 3×3 加权) 漂移 < 2 px | 光路/漂移, 目标框会错位 |
| G0-3 相位到达面板 | `‖frame_net − frame_flat‖₂ > 3 × ‖frame_A1 − frame_A2‖₂` | 槽位 no-op / 面板冻结 ⇒ 立即停 |
| G0-4 几何可辨识 | `ncc(实测 net 帧, 孪生预测 net 帧) ≥ 0.5` | 映射/ROI/标度错, 先修几何再测 |

G0-4 的 0.5 是**宽松绝对门**: 孪生已知绝对量偏差 (PIB / EE −79 %, FWHM +26 %), 它只用来抓
「相位贴错位置/贴错尺度」这类会毁掉一切的错误, 不用于判断孪生准不准。

#### 修正记录 (2026-10-26, 追加; 上表原文保留, 阈值一个没改)

**G0-2 的测量口径收窄为「同一样本内部, A1 对 A2」。** 原文写「全部帧零级质心漂移」, 但
「零级」只有在**平场帧**上才能由 `argmax` 认出: 成型帧的全局最大值是成型后的亮点, 而目标光斑
本来就落在零级附近。三种替代读法都已在一次**健康的** 24 样本孪生干跑上实测 (台面纹丝未动):

| 读法 | 实测「漂移」 | 结论 |
|---|---|---|
| 逐帧 `argmax` (原文的字面读法) | **49.5 px** | 假报警 (读到的是成型亮点) |
| 以 A1 为锚, ±16 px 局部搜索 (仍作诊断) | **16.5 px** | 假报警 |
| 全会话平场帧互比 | **38.1 px** | 假报警: 这不是漂移, 是每样本**故意抽的**倾斜 (`AberrationState.tilt_px` ≤ 6 相机像素) |
| **A1 对 A2 (本实现, 入门禁)** | **0.00 px** | ✅ |

A1/A2 是同一条件测两遍, 中间只隔了 B 帧, 所以它们的差就是漂移本身, 而且成型光无法劫持它。
逐帧 `argmax` 与 ±16 px 局部搜索的读数仍写进 `session.json` 的 `g0_2_zero_order`
(`shaped_peak_excursion_px` / `session_spread_px`), 只作诊断, 不入门禁。

**G0-4 需要孪生预测帧 ⇒ `Instrument` 契约增加 `twin()`** (见 §1.1 的修正记录)。门禁阈值与
判据仍是 `≥ 0.5`。

两处改动都发生在**任何实验系统数据存在之前**, 依据只有孪生干跑; 它们让门禁可执行, 不是放宽门禁。

## 5 不做什么

- 不做闭环 / 探针 refit / oracle —— 那需要先在实验系统上实现探针链路, 是下一件事。
- 不用 PIB / EE / FWHM 的**绝对值**下结论 (报告 §9.1)。
- 不重抽语料、不重训、不换判据、不加样本凑显著。
- 不把 `runs/sim2f_smoke800` 的任何数字混进来 (那套已被推翻)。

## 6 会话流程

```powershell
# 0) 干跑: 用孪生当仪器, 验证整条链路 + 复现报告 §5 的数字 (无硬件)
$env:PYTHONPATH="src"
.venv\Scripts\python.exe runs\sim2f_bench\run_bench.py --backend sim --samples loop

# 1) 实验系统 (先给曝光上限, 再跑)
$env:AO_DAHENG_MAX_EXPOSURE_MS="0.8"
.venv\Scripts\python.exe runs\sim2f_bench\run_bench.py --backend hw --slm-slot-mode rotate `
    --exposure-ms 0.02 --settle-s 0.3 --out-dir runs/sim2f_bench/session
```

`--backend sim` 是默认值: 不带参数运行**不会**碰硬件。

> ⚠️ 上面第 1 条里的 `--slm-slot-mode rotate` 是本 CLI **不接受**的开关 (原稿写了一条并不存在的
> 参数): 内存槽轮换已经无条件发生在 `display_phase` 内部 (2..125, 排除当前显示槽), 不需要也不能
> 再选模式。另外 `--exposure-ms 0.02` 也不再需要给定 —— 曝光改由**阶梯实测**选出
> (见下)。照下面这条跑:

```powershell
$env:AO_DAHENG_MAX_EXPOSURE_MS="0.8"
.venv\Scripts\python.exe runs\sim2f_bench\run_bench.py --backend hw `
    --settle-s 0.3 --samples loop `
    --out-dir runs/sim2f_bench/session
```

#### 曝光: 实测选择, 不再靠猜 (追加)

`--exposure-ms` 是可选的**显式覆盖**: 给了就全程用它, 阶梯完全不跑。不给时程序在**平场相位、
全画幅**(尚未开 ROI 窗) 上跑一条曝光阶梯, 取平场峰值最接近 **0.5 × 满量程** 的那一档:

- 自带阶梯: `--exposure-ladder "0.002,0.005,0.01,0.02,0.05,0.1"` (ms, 逗号分隔);
- 不给阶梯也不给曝光: 默认几何阶梯 **cap/64 … cap** 共 8 档 (`cap = AO_DAHENG_MAX_EXPOSURE_MS`,
  步比 `64^(1/7) ≈ 1.812`)。
- 采集走驱动基类 `BaseCamera._acquire_exposure_ladder` (每档丢首帧、读回**实际生效**曝光、
  退出恢复进入时曝光), 所以驱动层的陈旧帧/钳位坑不必再实现一遍。
- 选定后**显式写回并读回**该曝光: 会话的每一帧都用它, 不会停在阶梯最后一档。
- 整条阶梯 (`requested_ms` / `applied_ms` / `peak_counts` / `peak_fraction` / `chosen_ms`) 记进
  `session.json` 的 `exposure_ladder`, 并附在 Gate-0 的 G0-1 条目上。**不做带外补偿**: 若最好
  的一档仍够不到 0.2 × 满量程, 就保留它并让 G0-1 失败 (附带整条阶梯), 而不是偷偷换一档。

### 6.1 上机前检查清单 (按顺序做, 不要跳)

1. **设曝光上限**: `$env:AO_DAHENG_MAX_EXPOSURE_MS="<本会话上限 ms>"`。未设或 ≤ 0 时程序
   **拒绝打开**任何设备 (1064 nm 近饱和档只有 ~0.02 ms 量级)。曝光本身**不用你给**: 脚本会用
   默认阶梯自己量 (除非你显式传 `--exposure-ms` 想钉死某一档)。
2. **确认照明瞳面位置**: 看光束在 SLM 面板上的实际位置。不是面板中心就加
   `--pupil-center-xy X,Y` (面板像素)。这一步错了, 远场是**错的**而不只是偏的, 事后无法从
   产物里恢复。
3. **先单样本干跑**: `--backend hw --limit 1 --no-save-frames`, 然后**读 Gate-0 表** (以及它
   上方打印的 "exposure chosen" 一行 —— 那是阶梯选出来的档位与目标峰值)。
   四项全 PASS 才继续; 任何一项 FAIL 就先修 (G0-1 曝光 / G0-2 漂移 / G0-3 槽位或面板冻结 /
   G0-4 几何), 不要带着 FAIL 往下跑。G0-1 失败时先看 `session.json` 的 `exposure_ladder`:
   整档峰值都偏低 = 光太弱或阶梯整体太短, 整档偏高 = 阶梯整体太长, 而不是"再试一个数"。
4. **再跑全会话**: `--samples loop --out-dir runs/sim2f_bench/session`。
5. 收工后核对 `session.json` 的 `pupil_placement` / `exposure_ladder` 与 `g0_2_zero_order`,
   以及 `rows.csv` 的 `zero_order_offset_px` 一列是否都合理 (该列只作诊断, 不参与判定)。

## 7 产物

`<out-dir>/` 下:

| 文件 | 内容 |
|---|---|
| `session.json` | 全部元数据: 设备、`exposure_ladder` (曝光阶梯与选定值)、ROI、槽位序列、`pupil_placement` (瞳面中心 + 是否为缺省)、Gate 0 四项读数、判据常量 |
| `rows.csv` | 每样本一行: `ncc_flat` / `ncc_net` / `ncc_a_drift` / `objective_*` / `l1_*` / `ee_*` / `zero_order_offset_px` (诊断) / 是否 indistinguishable |
| `frames.npz` | 四次帧 (float32), 除非 `--no-save-frames` |
| `fetch.png` | A1/B1/B2/A2 与 target 的四联图 |
| `VERDICT.md` | 自动生成: 判据逐条取值 + 结论字符串 (**只陈述判据, 不解释原因**) |

`VERDICT.md` 的结论字符串只有三种: `net_recovers` / `net_does_not_recover` / `insufficient_power`。
解释与归因留给之后的分析, 不在脚本里写。

## 8 已知风险 (会毁掉会话的)

1. **槽位 no-op**: 连续写同一槽, 液晶不刷新 —— `display_phase` 已做轮换, 但 G0-3 是最后一道防线。
2. **DVI 模式挂起**: 只用内存模式 (`video_mode=0`)。绝不在诊断里试 DVI, 挂了要物理断电。
3. **曝光过载**: 1064 nm 在近饱和档只有 ~0.02 ms 量级; `AO_DAHENG_MAX_EXPOSURE_MS` 必须设。
4. **0 级不在帧中心**: 一律 `argmax` 定位, 不假设几何中心; 定位后 `reset_window(center, (256,256))`。
5. **ROI 改过之后的首帧是陈旧帧**: `get_numpy_image` 的 `skip_first=True` 保持默认。
