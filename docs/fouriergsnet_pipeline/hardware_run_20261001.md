# slm-gsnet 实机方形整形运行报告 (2026-10-01)

> **一次失败的实机运行, 附完整实测量。** 结论先行: 该运行是**随机游走**
> (`dec = 0.487`), 且末态比初态**差 35.9%**。本文记录运行前后的全部实测量、
> 根因, 以及本台架上"方形远场整形"为何在当前构型下不可达。
>
> 数据根目录 `data/debug/slm_gsnet_square_20261001_160808/`
> (recorder pickle 409 MB + JSON sidecar + 汇总 PNG)

---

## 1. 结论

| 判据 | 实测 | 判读 |
|---|---|---|
| `dec` (下降步占比) | **0.487** | ≈0.5 = 抛硬币 → **随机游走, 未收敛** |
| `sustained` (末态 vs 首态) | **−35.9%** | **变差** |
| `late` (后 1/3 − 前 1/3) | −2.1% | 无趋势 (在噪声内) |
| `best` | **+0.0% @ epoch 1** | **最优就是初始态** |
| 四分位均值 | 0.1290 / 0.1238 / 0.1241 / 0.1245 | **平坦, 无下降** |
| `guard` 行数 | 0 | 该路径**没有**能量门 |

**能量项被牺牲以换取几乎为零的 CV 收益 —— 目标函数被"钻空子", 而非被优化:**

| 项 | 首态 | 末态 | 变化 |
|---|---|---|---|
| CV 项 `0.4·exp(−2·CV)` | 0.0929 | 0.1048 | **+0.0118** |
| EE 项 `0.6·EE` | 0.0947 | 0.0155 | **−0.0792** |
| EE 末/首比 | — | — | **0.16×** (6 倍能量流失) |
| `max_brt` (0 阶峰值) | 225 | 17 | **光斑被彻底打散** |

同一曝光下**平场参考的 `max_brt` 为 202–246**, 所以末态的 17 是散斑, 不是聚焦光斑。

---

## 2. 运行参数

```bash
python src/ao_shaping/main.py slm-gsnet spgd \
  --cam_type daheng --cam-id 0 \
  --exposure_time_ms 1.5 \
  --cam_size 300 --target-side 100 \
  -c shape --zernike_radius 450 \
  --slm_number 1 --slm_wavelength 1064 \
  --objective quality \
  --delta 0.2 --lr 0.02 \
  --optimizer_type adamod \
  -e 80 --debug
```

两处刻意偏离默认值, 理由见 §4:

- **`--target-side 100`** (非 50): 80 px block 的 staircase 可达远场结构
  ≥46–92 cam px; 若 box 只有 50 px, 可达结构全部落在 box 外, 目标函数退化。
- **`--lr 0.02`**: `lr=0` 会触发 `learning_schedule()` **静默覆盖 `--delta`**。

运行: 80 epoch / 2 分 29 秒 / 1.8 s per epoch / 无挂起 / 无设备争用。

---

## 3. 运行前的实测量 (全部本会话实测)

### 3.1 台架健康度 (平场优先, 经 `slm_bench_probe`)

关键: **必须先写平场再测**。台架事实 2 —— 面板保持上次显示的图案,
不写平场就测到的是上一轮的散斑 (实测同一 3 ms 设置在两次运行间读数
100 → 23 counts)。

| 量 | 实测值 |
|---|---|
| 0 阶位置 (CCD) | **(672.3, 1026.6)**, 帧中心 (1296, 972) → x 偏 **−624 px** |
| 平场 FWHM | **12.2 px** (σ ≈ 5.2 px) |
| hollowness | 0.88–0.92 (单叶, 健康) |
| core40 | 0.024–0.049 |
| 面板 | 1920×1200 (W,H), pitch 8 µm, 2π = 993 gray |
| 光斑半径 (SLM 面) | 450 px = 3.6 mm |

FWHM 12.2 px 与理论一致: 均匀照明半径 450 px 的 Airy 尺度
`λf/(πR)` = 11.8 µm = 5.4 cam px 半径。**平场已接近波前平整 (高 Strehl)**,
这正是 §4.1 的关键前提。

### 3.2 曝光裁切拐点 bracketing (每档先写平场)

| 曝光 | peak | ≥250 占比 | FWHM | hollowness | 50px box 内饱和 |
|---|---|---|---|---|---|
| 1.00 ms | 137.5 | 0.00000% | 12.2 | 0.89 | 0.0000% |
| 1.25 ms | 170.2 | 0.00000% | 12.2 | 0.88 | 0.0000% |
| **1.50 ms** | **202.5** | **0.00000%** | **12.2** | **0.88** | **0.0000%** |
| 1.75 ms | 238.2 | 0.00000% | 12.2 | 0.87 | 0.0000% |
| 2.00 ms | 255.0 | 0.00030% | 12.4 | 0.93 | **0.6000%** |

裁切拐点在 **1.75–2.00 ms 之间**。2.0 ms 时 50 px box 内 **0.6%** 像素饱和
(≈150/2500), 足以压平直方图顶部并使 CV/Pearson 失真。
**1.5 ms 是最后一个干净点** (余量 52.5 counts), 本次运行采用。

> ⚠️ 激光功率会漂移: 平场 peak 在 15:30 读 202.5, 15:38 读 246 (+21%)。
> 曝光余量需按当次实测确认, 不能沿用。

### 3.3 噪声地板与 SNR 扫描 (1.5 ms, `n_max=9` = 54 DOF)

σ (噪声地板) = **2.3017e-04**

| delta | 0.0005 | 0.001 | 0.01 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|
| 多模式 SNR | 1.38 | 0.91 | 0.79 | 0.75 | 0.92 | 0.45 |

**`usable_deltas()` 返回空集** —— 全部低于 `≥2 可用` 门槛。
SNR 随 delta 增大反而下降。这与
[`docs/slm_pib_bench/report.md`](../../slm_pib_bench/report.md) 的记录一致:
该台架噪声地板**单日内波动 21 倍**, 必须每次实测。

### 3.4 写入测试 (决定 freeform 前提是否成立)

| 相位 | 质心位移 | peak 比 |
|---|---|---|
| Zernike tilt 0.5 rad | 2.7 px | 0.99 |
| Zernike tilt 2.0 rad | **10.1 px** | 1.04 |

2.0 rad → 10.1 px, 即 **5.05 px/rad**, 与理论 `f·λ/(πR·p_cam)` = 5.34 px/rad 一致。

> **面板确实能调制。** 此前 `docs/slm/model_in_loop_bench_calibration.md`
> 记录的"像素级相位无效果"是**像素尺度 (8 µm)** 的结论;
> freeform 用的是 **80×50 px block**, 粗 1–2 个数量级, 完全可分辨。
> 因此 freeform 的前提成立, 问题不在面板。

---

## 4. 根因

### 4.1 目标函数的最优解就是"平场" (物理)

平场 FWHM 12.2 px 已是 450 px 光束的衍射极限。远场特征尺度 ∝ `1/面板特征尺寸`:
要在一个 30 px 的主瓣**内部**刻出方形, 需要面板周期 ≳ `7400/25` = **296 px**,
那只有 Zernike n≈1–2 的平滑起伏 —— 它只能**平移/离焦**主瓣, 无法压平它。
80 px block 的 staircase 可达结构 ≥46–92 cam px, 落在主瓣之外。

实测印证: 叠加相位后峰值单调下降 `86 → 35 → 24 → 12` (flat → n≤4 → n≤8 → n≤14)。
**任何**偏离平场的相位都只会把光散出 box。
所以 `quality` 的全局最优 ≈ **平场 (系数 → 0)**,
一次"成功"的运行只是 576 维降落到原点 —— 与随机游走不可区分。

### 4.2 随机初始化直接毁掉焦点

`slm_gsnet_runner.py:1223` 用 `rng.uniform(-π, π, size=576)`,
即**满幅**随机相位 (docstring 却写 "small random init")。
起点 `max_brt = 225` → 末态 `17`。初始散斑的 EE (0.158) 反而**高于**
搜索找到的任何状态 (0.024–0.035), 于是 "best @ epoch 0"。

### 4.3 无能量门 → EE 崩塌 6 倍

方形路径**没有** encircled-energy guard (与 `slm-pib` 不同)。
`w_efficiency=0.6` 权重不足以阻止: 散射光使 box 内 CV 略降,
代价是 EE 掉 6 倍, 净亏 −35.9%。这正是仓库文档反复警告的
"目标函数缺能量项 → 搜索把光推出目标框"。

### 4.4 梯度 SNR 不足 (576 DOF 稀释)

SNR 实测 (54 DOF) 全部 < 1.4。freeform 是 **576 DOF, 其中约 159 被照亮**
(光斑覆盖 x 6–17、y 3–20 共 ~216 格), 稀释更严重 → 逐 DOF SNR < 1。
叠加: `CAM_SAMPLE_ITER = 1` (单帧, 无 ABBA、无漂移对消),
`SLM_RESPONSE_TIME_S = 0.3` 使 +/− 帧间隔 0.3 s 而漂移不随平均衰减。
`dec = 0.487` 正是这个组合的预期表现。

---

## 5. 期间发现的代码级问题 (与本次失败直接相关)

| # | 问题 | 位置 |
|---|---|---|
| 1 | `lr=0` 会用 `learning_schedule()` **静默覆盖 `--delta`** | `slm_square_shaping.py:1596-1605` |
| 2 | `_param_scale` freeform=1.0 而 Zernike=0.1, 故 Zernike 调好的 δ 不通用 | `slm_square_shaping.py:1163/1170` |
| 3 | "freeform per-pixel" 名不副实: `np.kron` 分块复制, 24×24 → **80×50 px/block** | `_freeform_phase_radians` |
| 4 | 随机初始化是满幅 `uniform(-π,π)`, 与 docstring 声称的 "small init" 矛盾 | `slm_gsnet_runner.py:1223` |
| 5 | 方形路径无 encircled-energy guard | `ObjectiveParamsSquare` |
| 6 | 单帧采样, 无 ABBA / 漂移对消 (台架已备 `slm_snr_probe.abba_signal` 但未接) | `CAM_SAMPLE_ITER = 1` |
| 7 | 传 `--exposure_time_ms` 会**关闭自动曝光**, 且饱和保护成为死代码 (`exposure_time_ms == 0` 才触发) | `resolve_initial_exposure` / L1670 |
| 8 | `--exposure_time_ms` 默认 **80.0 ms** —— 本台架近饱和基线 ~0.02 ms, 默认值必然饱和 | `CameraParams` |
| 9 | 焦距标定常数自相矛盾: 文档同时写 `132940/P` 与 `7600/P` (差 17.5×); 实测 7400 与一阶 `7557/P` 相差仅 1–2% | `slm_bench_probe.py:78` / 文档 |
| 10 | 仓库中 **11 个**非记录 pickle (`None` / `Recorder` 对象) 曾使语料索引整体失败 —— 已修复 | `gsnet_offline.py:292-310` |

---

## 6. 若要继续, 需要先改什么 (按收益排序)

1. **加 encircled-energy guard** (复用 `slm-pib` 的 `max_roi_energy_loss`)。
   没有它, 任何 `w_efficiency` 权重都会被"散光换 CV"击败。
2. **把随机初始化改为平场起步** (`c = 0`), 或从低幅 (如 ±0.1 rad) 起步。
   满幅随机起点直接毁掉焦点, 且它的 EE 反而最高。
3. **接上 ABBA 漂移对消 + 多帧平均** (`slm_snr_probe.abba_signal` 已实现),
   否则单帧差分的 SNR 不足以支撑 576 DOF。
4. **降 DOF**: 减小 `phase_grid` (当前硬编码 24) 或按光斑裁剪到被照亮的 ~159 格,
   直接提高逐 DOF SNR。
5. **重新定义可达目标**: 在本台架上, 80 px staircase 能影响的是
   *大尺度* 结构 —— 例如把能量聚进一个 **≥100 px 的大 ROI**、压低斑径、
   提升 Strehl, 而不是刻一个 50 px 的平顶方块。
6. **修正 `--exposure_time_ms` 默认值** (80 ms → 本台架 ~1–1.5 ms),
   并让饱和保护在固定曝光下也生效。

---

## 7. 复现

```powershell
$env:PYTHONPATH = 'src;libs'
# 必须作为独立进程 + 外部看门狗运行 (SLM memory-mode open + 首次采集可能无限阻塞)
python src/ao_shaping/main.py slm-gsnet spgd `
  --cam_type daheng --cam-id 0 `
  --exposure_time_ms 1.5 --cam_size 300 --target-side 100 `
  -c shape --zernike_radius 450 `
  --slm_number 1 --slm_wavelength 1064 `
  --objective quality --delta 0.2 --lr 0.02 `
  --optimizer_type adamod -e 80 --debug
```

> ⚠️ 设备安全: SLM **仅 memory mode** (`video_mode=0`), 绝不开 DVI;
> 连续写入必须轮换 memory slot; 挂起后需对 SLM 控制器**物理断电**。
> 判定收敛请用 `dec` / `late`, **不要只看 `sustained`**
> (本台架 51 次历史运行中 32 次是随机游走)。
