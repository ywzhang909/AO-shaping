# Report 2 — SLM 相位生成链路审计 + 矫正效果根因分析

**日期**: 2026-09-16
**设备**: SLM #23020026 (532nm) + WFS M01219666 (MLA150M-5C, 27×27)
**主题**: ① 相位生成单位/约定链路全面审计 ② 闭环矫正效果提升路径

---

## 0. 结论速览

| # | 发现 | 位置 | 性质 | 状态 |
|---|---|---|---|---|
| **A** | 响应矩阵用 **µm** 构建、矫正 `w` 用 **λ** → 反解系数放大 **1/0.532 = 1.88×** | `tools/slm/slm_zernike_correction.py` | **单位错误 (已修)** | ✅ 修复并实测确认 |
| **B** | 反解系数是 **λ(波长)** 却按**弧度**传给 `make_phase` → 加载相位缩小 **2π = 6.28×** | 同上 | **单位错误 (已修)** | ✅ 修复，闭环 13.8% → **42.1%** |
| **C** | `ZernikeDM.generate_phase` 做 **min-max 归一化** → 输出对系数缩放**不变** → Zernike 幅度**完全不可控** | `drivers/dm/zernike_dm.py:106-129` | **重大缺陷** | ✅ **已修复并实测验证** |
| **D** | `csv_to_phase` 用模块常量 1023 而非设备 `_max_gray` | `santec/driver.py:1244` | **有意设计** (CSV 设备无关, 便于跨 SLM 复用) | ✅ 非缺陷 |
| **E** | 矩阵含零列时 `np.linalg.pinv` 条件数爆到 1e18 | `tools/slm/*` | 数值稳定性 | ✅ 加 `safe_pinv` |
| **F** | 模型自检: 大修正量时预测/实测比 0.35, 小修正量 0.95~1.22 | 闭环 | 大相位下**叠加性/线性度下降** | ⚠️ 待查 |
| **G** | 独立响应矩阵工具 `slm_zernike_response.py` 同样把 **µm** 当 λ 用, 且 `device_config` 未记录设备参数 | `tools/slm/slm_zernike_response.py` | 单位 + 记录缺失 | ✅ **已修** (`um_to_waves` + `collect_device_info`) |
| **H** | **`zernike-matrix` 命令原生崩溃** (0xC0000005 访问违例 / 0xC000041C 回调致命异常), 崩溃点不固定 → SDK 层内存损坏 | `runners/zernike_matrix_runner.py` | 未解决 (改用等价工具) | ⚠️ **待查** (见 §2.6) |
| **I** | `parse_tuple` 用 `int()` → 无法表达 mm 级小数 pupil 中心 (实测 -0.14, 0.18), 被迫退回 `(0,0)` 而**构造函数传入的 pupil 会覆盖配置中的实测值** | `utils/cli_helpers.py:38` | 限制 | ✅ **已修** (改 float) |

---

## 1. 审计范围与方法

用户要求核查"发现的 SLM 相位生成问题是否也出现在 `santec/driver.py` 与 `gui/slm/multi_slm_controller.py`"。
审计沿**相位生成 → 弧度/灰度换算 → 下发**全链路逐点核对单位与归一化约定：

```
Zernike 系数 → 相位(rad) → 灰度(0.._max_gray) → 内存槽 → LCOS
                 ↑            ↑
           单位? 归一化?    2π 灰度基准?
```

---

## 2. 逐文件审计结果

### 2.1 `drivers/slm/santec/driver.py` — ✅ 约定正确

| 函数 | 约定 | 判定 |
|---|---|---|
| `create_phase_from_array(phase_rad)` L1369 | `gray = rad / (2π) × self._max_gray` (mod 2π) — 用**设备实测** 2π 灰度 (532nm → 998) | ✅ 正确 |
| `display_phase(phase_rad)` | 弧度 → `create_phase_from_array` → `display_data` | ✅ 正确 |
| `display_data(phase_gray)` / `_write_phase` | 接收 **uint16 灰度** | ✅ 正确 |
| `csv_to_phase` L1244 | `rad = gray / get_max_grayscale() × 2π`, 常量 1023 | ✅ **有意设计** (见 §2.4) |
| `set_wavelength` / `get_wavelength_info` | `phase_range = 200` (=2π), 由设备读回反算 2π 灰度 | ✅ 正确 |
| `shift_phase` | 纯函数, 平移数学唯一实现 | ✅ 正确 |

**结论: 驱动层无单位错误。** 弧度↔灰度换算始终经 `_max_gray`，与设备实测一致。

### 2.2 `gui/slm/multi_slm_controller.py` — ✅ 约定正确

| 位置 | 行为 | 判定 |
|---|---|---|
| `_apply_shift` L240 | 委托 `Santec.shift_phase` (平移数学唯一实现) | ✅ 正确 |
| `_toggle_phases_task` L1086 | `display_data(target_phase)` — phase 来自 `get_displayed_phase()`(**灰度**)或 `np.zeros(uint16)` | ✅ 灰度直发, 正确 |
| `_export_phase_csv` L1164 | `rad = gray / _max_gray × 2π` — 用**设备**值 | ✅ 正确 |
| "保存当前相位到CSV" L1035 | 同上, 用 `_max_gray` | ✅ 正确 |
| `ZernikeControl.generate_phase_rad` (`pattern_controls.py:850`) | `generate_zernike_polynomial` 返回**原始弧度相位 (不 mod-2π、不归一化)**, 保留系数绝对幅度 | ✅ 正确 (注释亦明示) |
| 其他控件 (flat/grating/lens/vortex/...) | 均输出原始弧度, 由 `create_phase_from_array` 统一换算 | ✅ 正确 |

**结论: GUI 层无单位错误; Zernike 控件走 `PatternHelper` 路径, 幅度可控。**

### 2.3 ⚠️ `drivers/dm/zernike_dm.py::generate_phase` — **重大缺陷**

```python
# L106-127
phase_raw = self._generator.generate_polynomial(coeffs_dict)
phase_raw = np.nan_to_num(phase_raw, nan=0.0, ...)

phase_min, phase_max = float(phase_raw.min()), float(phase_raw.max())
phase_range = phase_max - phase_min
phase_normalized = (phase_raw - phase_min) / phase_range     # ← min-max 归一化
...
if output_mode == "rad":
    phase_out = phase_normalized * 2 * np.pi                 # 永远满量程 2π!
else:
    phase_out = (phase_normalized * max_val).astype(np.uint16)
```

**实测验证** (本机执行):

```
ZernikeDM 系数 ×1 vs ×4:
  identical = True          ← 两组系数产生逐字节相同的相位
  PV: 6.2832 rad → 6.2832 rad   (期望 4×)
```

**→ 输出对系数缩放不变, Zernike 系数的"幅度"维度被完全抹掉。**

这正是 `AGENTS.md` 已记录的 `PatternHelper._zernike_to_uint16` 反模式
("min-max normalises ... making the pattern scale-invariant"),
但 **`PatternHelper._zernike_to_uint16` 已废弃不用, 而 `ZernikeDM.generate_phase` 仍在服役**。

**受影响链路** (`ZernikeSLM.send_zernike` → `ZernikeDM.generate_phase`):

| 消费方 | 后果 |
|---|---|
| `optimizer/wf/zernike_response_matrix.py` (`zernike-matrix` 命令) | 标定出的响应矩阵**与幅度无关** → 用于闭环矫正无意义 |
| `optimizer/wf/rms_by_zernike.py` (`rms-zernike`) | 无法按系数幅度做梯度更新 |
| `optimizer/wf/ga_zernike.py` (`ga-zernike`) | 遗传算法搜索的"系数"维度退化 |
| `optimizer/wf/greedy_zernike.py` | 同上 |
| `runners/{zernike_matrix,rms_zernike,slm_offset}_runner.py` | 同上 |
| `gui/zernike/zernike_response_matrix_ui.py` | 同上 |

**✅ 已修复 (2026-09-16)**: 去掉归一化, **系数即弧度**直接输出; `"gray"` 模式改为
`mod(rad / 2π × max_val, max_val)` (语义与 SLM 驱动 `create_phase_from_array` 一致)。

```python
# 修复后 (drivers/dm/zernike_dm.py)
phase_raw = np.nan_to_num(self._generator.generate_polynomial(coeffs_dict), nan=0.0, ...)
self._current_coeffs = coeffs_dict
if output_mode == "rad":
    return phase_raw                      # 系数即弧度, 保留绝对幅度
return np.mod(phase_raw / (2*np.pi) * max_val, max_val).astype(np.uint16)
```

**修复后实测**:

```
系数 ×1 vs ×4:  identical = False     (修复前 True)
  PV: 3.4641 rad → 13.8562 rad   比值 = 4.000   (期望 4.0) ✅
  gray 模式: uint16, range [0, 1022] (2π 系数 → 满量程) ✅
```

> ⚠️ **行为变更提示**: 该修复**有意改变**下列调用方的行为 —— 它们此前因归一化而无法控制
> 相位幅度, 现在系数即弧度、幅度可控。若某调用方依赖旧的"自动满量程"行为, 需显式传入
> `2π` 量级的系数:
> `optimizer/wf/{zernike_response_matrix,rms_by_zernike,ga_zernike,greedy_zernike}.py`、
> `runners/{zernike_matrix,rms_zernike,slm_offset}_runner.py`、`gui/zernike/`。
> **建议**: 用 `zernike-matrix` 重标一次响应矩阵 —— 旧矩阵是在归一化下测得的, 幅度维度无意义。

### 2.4 `csv_to_phase` 用常量 — ✅ 有意设计 (非缺陷)

`csv_to_phase` 是 `@staticmethod`, 用模块常量 `get_max_grayscale()` (=2^10−1=1023) 把灰度还原为弧度,
**而非**设备实测 `_max_gray` (532nm → 998)。

**这是有意为之**: CSV 的灰度域定义为**设备无关的 0..1023 ↔ 0..2π**, 同一份 CSV 可在不同 SLM /
不同波长间复用; 若绑定设备 `_max_gray`, CSV 便不可移植。

**使用注意**: 因此 CSV 的灰度基准 (1023) 与设备实际 2π 灰度 (998) 不同 —— 由设备侧
`create_phase_from_array` 在最终换算时吸收该差异。**不要**把 `csv_to_phase` 改成用实例 `_max_gray`。

### 2.5 修复项: 数值稳定性

矩阵经线性度门控会把不合格列**置零**, 直接 `np.linalg.pinv(M)` 在含零列时条件数爆到 **1e18**
(数值秩亏)。已加:

- `slm_zernike_common.safe_pinv(M)`: 仅对**有效列**求伪逆再展开 (零列对应行恒为 0)
- `slm_zernike_common.effective_cond(M)`: 报告**有效列**上的条件数 (含零列时直接 cond=inf)

---

### 2.6 ⚠️ `zernike-matrix` 命令原生崩溃 (未解决)

**现象**: `python -m ao_shaping.runners.zernike_matrix_runner ...` 在
`_capture_init_state` 之后崩溃, 退出码不固定:

| 轮次 | 参数 | 崩溃点 | 退出码 |
|---|---|---|---|
| 1 | `--mla-index 768` (35×35 斑点) | `_save_init_state_hdf5` 内 (h5 留在 0 字节) | `0xC0000005` (访问违例) |
| 2 | `--n-max 2 --debug --mla-index 512` | 第一个模式 `iter 0 rms=0.0` 之后 | `0xC000041C` (用户回调致命异常) |
| 3 | `--n-max 4 --mla-index 512` | `_capture_init_state` 之后 | `0xC000041C` |

**崩溃点随参数/时序变化 → 典型内存损坏特征**, 而非某一行 Python 代码的逻辑错误。

**已排除**:
- 核心链路**正常** —— 最小探针 (`ZernikeDM.generate_phase` → `create_phase_from_array`
  → `display_phase` → `take_image` → `get_zernike` → `get_spot_deviation` →
  `send_zernike` → `build_subaperture_mask`) **全部通过, exit 0**。
- `_capture_init_state` / `wf_callback` / `_save_init_state_hdf5` 的 Python 逻辑无异常
  (回调有 try/except, 且其内部函数在探针中已单测通过)。
- 与 `ZernikeDM` 归一化修复**无关** (探针已覆盖该路径)。

**可疑方向** (留给后续):
1. `_thorlab_wfs.py` 的 `MAX_SPOTS` 缓冲区是否按 32×32 设计 —— 第一轮 35×35=1225 斑点
   超界可能造成堆损坏, 后续在别处爆发。
2. `runners/zernike_matrix_runner.py` 与 SDK 的 ctypes 回调/数组传递。
3. `--debug` 回调逐样本写 18MB `.npy` (实测每帧 18,432,128 B) 造成的时序/内存压力。

**当前处置**: 改用**等价的、已验证的** `tools/slm/slm_zernike_response.py` (同为推拉法,
且已记录完整设备参数) 完成重标 → 见 `docs/slm/report3.md`。

### 2.7 ✅ 修复 `parse_tuple` 的整数限制

`utils/cli_helpers.py:38` 原用 `map(int, parts)` → `--pupil-center "(-0.14,0.18)"` 报
`Invalid center format`。因 WFS pupil 中心单位是 **mm** (实测常为小数), 且
`ThorlabWFS.__init__` 传入的 pupil **会覆盖配置文件中的实测值**
(`thorlab_wfs.py:440-453`), 用整数近似会造成 pupil 偏移、污染 `WFS_ZernikeLsf` 拟合。
已改为 `map(float, parts)` (整数输入仍可用)。

## 3. 闭环矫正效果提升路径 (实测)

### 3.1 修复前后对比

| 轮次 | 修复内容 | 闭环改善 (RMS) | 备注 |
|---|---|---|---|
| 首轮 | — | 5.8% | 矩阵污染 |
| 2 | 半径一致性 (R=200 异常) | 3.4% | 半径诊断被 R=120 异常值污染 |
| 3 | 半径诊断异常剔除 | 19.7% | 扫描半径修正为 {450,600} |
| 4 | **矩阵 µm → λ** | 13.8% | 系数不再放大 1.88× |
| 5 | **系数 λ → rad (×2π)** | **42.1%** | ✅ 当前最佳 (0.3564 → 0.2065λ) |

### 3.2 修复 5 的实测细节

```
矫正前: RMS=0.3564λ  PV=1.6040λ  |w|=0.3825λ
[iter 1] 系数: 正 6 / 负 8 / 零 0  → RMS=0.2065λ (改善 42.1%)
[iter 2]                          → RMS=0.2733λ (23.3%)
[iter 3]                          → RMS=0.3100λ (13.0%)
最佳 = iter1
```

**系数符号分布**: iter1 正 6 / 负 8 → **正负都有** ✅ (像差有正负分量, 矫正亦然)。

### 3.3 剩余差距: 模型自检

新增的**模型自检** (`‖w_prev + M·c‖` vs 实测 `‖w_after‖`) 逐轮结果:

| 轮次 | ‖c‖ | 预测 | 实测 | 比值 |
|---|---|---|---|---|
| 1 | 1.38 | 0.1183λ | 0.3374λ | **0.35** ← 大修正量, 模型高估 3× |
| 2 | 0.72 | 0.3850λ | 0.3156λ | 1.22 |
| 3 | 0.76 | 0.3256λ | 0.3428λ | 0.95 ✅ |

**解读**: 小修正量时模型吻合 (0.95~1.22), **大修正量时实测仅为预测的 35%** →
存在**幅度相关的非线性/饱和**, 而非纯单位错误。可能来源:

1. **SLM 相位响应饱和 / LUT 非线性** (叠加多个模式后灰度分布跨度大)
2. **波前矫正映射 (520nm CSV 用在 532nm) 的逐像素重映射非线性**
3. 单模式标定的矩阵在**多模式同时加载**时叠加性下降

### 3.4 理论天花板

用 debug 落盘的矩阵与 `w_before` 做投影分析:

```
span(M) 内占比 96.9%;  正交残差 (不可矫正) = 24.6%
→ 仅靠这 14 列, 理论最好也只能把 ||w|| 降到 24.6% (最大降幅 75.4%)
离线最优单步 (c = −pinv·w): 降幅 75.4%   (符号约定正确: +pinv 会变差 −95.4%)
```

当前实测 42.1% vs 理论 75.4% → **仍有 ~33 个百分点的可争取空间**, 主因是 §3.3 的非线性。

---

## 4. 建议的后续动作 (按优先级)

| # | 动作 | 预期收益 |
|---|---|---|
| ~~1~~ | ~~修 `ZernikeDM.generate_phase` 的 min-max 归一化 (§2.3)~~ | ✅ **已完成** (系数即弧度, ×1/×4 PV 比 1:4 实测确认) |
| 1b | **用 `zernike-matrix` 重标响应矩阵** —— 旧矩阵在归一化下测得, 幅度维度无意义 | 使 `ZernikeSLM` 系矫正可用 |
| 2 | 查 §3.3 非线性: 用 `--save-phase` 的 `phase_gray.npy` 对比"请求相位 vs 实际上屏", 并关闭波前矫正复测 | 42% → 60~75% |
| 3 | 多轮迭代改用 `leak>0` 或降低 `gain` 抑制大修正量过冲 | 稳定收敛 |
| 4 | 提高 `n_avg` / 增加幅度档位, 让矩阵在大幅度段也有标定点 | 提升线性度覆盖 |
| 5 | 把波前矫正 CSV 换成 532nm 版本 (当前是 520nm) | 消除系统性静态偏差 |

---

## 5. 本轮 debug 数据产物 (已落盘)

```
data/zernike_correction/debug_<ts>/
├── manifest.json                 产物清单
├── matrix.h5 / matrix.json       响应矩阵 (λ/λ, 行 0..65 ↔ DLL[1..66]) + pinv + 有效列 cond
├── scan_points.json              84 点逐点诊断 (±推拉 diff/sym 向量, SNR vs 基线, valid_ratio)
└── closed_loop/
    ├── iter0_before/             w_lam.npy, wavefront.npy, metrics.json
    └── iterN/                    w_lam.npy, wavefront.npy, coeffs_lam.npy,
                                  phase_gray.npy (--save-phase), metrics.json (含 model_ratio)
```

报告 `report_<ts>.json` 现含**完整设备参数**: SLM (序列号/波长/2π灰度/灰度上限/**工作温度**/
固件版本/面板/平移/视频模式/矫正状态/LUT/响应时间) 与 WFS (序列号/设备名/厂商/型号/**曝光时间**/
**pupil 中心与直径**/MLA 名称/index/子孔径数/参考平面/高速模式/主增益)。

**两个工具均已接入** (统一经 `slm_zernike_common.collect_device_info`):

| 工具 | 设备参数落盘位置 |
|---|---|
| `tools/slm/slm_zernike_correction.py` | `report_<ts>.json` 的 `device` + `debug_<ts>/matrix.h5` 的 `device_config.device` |
| `tools/slm/slm_zernike_response.py` | h5 的 `device_config.device` + 同名 `.json` 旁挂文件 |

两者同时记录 `units` 字段 (`λ/λ`)、DLL 索引映射、参考平面与标定方法, 保证产物可复现、可审计。

---

## 6. 相关文档

- **`docs/slm/report3.md`** — **Zernike 响应矩阵重标 完整测试报告 (2026-09-16)**: 含设备参数、
  矩阵热图/对角主导/奇异值/方差/线性度、离线反解与结论解读 (带图带文字分析)
- `docs/slm/daily_2026-09-15.md` — 实验日报 (§9 含线性度分析)
- `docs/slm/zernike_response_matrix_report/report.md` — 响应矩阵报告 (创建/分析/检测)
- `docs/slm/zernike_linearity/linearity.md` — 线性度分析
- `src/ao_shaping/drivers/slm/AGENTS.md` — SLM 域文档 (含三个已踩坑)
- `AGENTS.md` — 项目反模式表

---

## 附: Zernike 响应矩阵报告 (创建/分析/检测)

**生成时间**: 2026-09-16 01:00:38  
**数据源**: `matrix.h5`, `report_20260916_004944.json`, `raw_scan_20260916_004944.json`

### 1. 创建 (Creation)

响应矩阵由 `tools/slm/slm_zernike_response.py` 以**推拉法**标定: 对每个 SLM Zernike 模式施加 ±A 扰动, 测 WFS Zernike 读数增量, `matrix[wfs_coeff, slm_mode] = Δ(WFS)/Δ(SLM 幅度)`。

| 项 | 值 |
|---|---|
| 设备 | SLM #23020026 + WFS M01219666 |
| 波长 / 2π 灰度 | 532 nm / — |
| SLM shift | [105, 40] |
| Zernike 半径 | 300.0 px |
| 扰动幅度 | 5.00 rad (0.7958 λ) |
| 推拉循环 / 帧平均 | 1 / 3 |
| WFS pupil | — |
| 参考平面 | — |
| 矩阵形状 | (66, 14) (WFS × SLM) |
| SLM 模式 (DLL 索引) | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] |

#### 1.1 设备参数 (SLM / WFS)

| SLM | 值 |
|---|---|
| 序列号 | 23020026 |
| 工作波长 | 532 nm |
| **最大相位 (2π)** | 6.283185307179586 rad = 1.0 λ (2π 灰度 = 998) |
| **工作温度** | [1.0, 61.3] °C (驱动板, 选件板) |
| 固件版本 | DLL:0250,Drive:,Option:0321,FPGA:0110 |
| 面板分辨率 | [1920, 1200] |
| 平移 shift | (135, 35) |
| 视频模式 | 0 (0=memory) |
| 波前矫正 | enabled=True, csv=D:\Projects\TIFO\AO-shaping\data\calibration\Wavefront_correction_Data_240236000006(520nm).csv |
| LUT | loaded=False, dir=None |

| WFS | 值 |
|---|---|
| 序列号 | M01219666 |
| 设备 / 厂商 / 型号 | WFS40-5C / Thorlabs / WFS |
| **曝光时间** | 3.9842571428571434 ms |
| **pupil 中心** | [-0.18211834716796876, 0.21564723205566405] mm |
| **pupil 直径** | [3.605471949213099, 3.8433362866785417] mm |
| MLA | MLA150M-5C (index 3) |
| 子孔径数 | 27 × 27 |
| 参考平面 | use_custom_ref=False |

> ⚠️ **索引约定**: 矩阵行 = DLL 系数索引 (**顺序 m 枚举, 非标准 Noll**); `[5](2,0)defocus` `[9](3,1)coma` `[13](4,0)spherical`。手册佐证 `roCMm` "derived from Zernike coefficient Z[5]"。

### 2. 分析 (Analysis)

#### 2.1 响应矩阵热图

![matrix heatmap](zernike_response_matrix_report/figures/01_matrix_heatmap.png)

绿框 = 对角线 (同索引 SLM 模式 → WFS 系数)。

#### 2.2 对角主导性

![diagonal](zernike_response_matrix_report/figures/02_diagonal_dominance.png)

**8/14** 个模式的列主导项即对角线项。

#### 2.3 奇异值谱 / 条件数

![svd](zernike_response_matrix_report/figures/03_singular_values.png)

条件数 **11.90** (良态, 伪逆稳定)。

#### 2.4 重复性方差

![variance](zernike_response_matrix_report/figures/04_variance_map.png)

平均方差 **1.205e-04**, 最大 **1.324e-02**。

#### 2.5 线性度 (多尺寸扫描)

![linearity](zernike_response_matrix_report/figures/05_linearity.png)

**正确判据**: 响应已按单位幅度归一化 → 线性响应表现为 `|resp|` **恒定**, 故用幅度离散度 **CV** 与响应向量方向一致性 **cos** (原 slope/R² 判据在线性时 slope≈0, 无意义)。

| 模式 | 尺寸 R(px) | CV(%) | cos_min | 判定 |
|---|---|---|---|---|
| [2] (1,-1)tiltA | 300 | 2.9 | 0.998 | ✅ |
| [2] (1,-1)tiltA | 400 | 4.6 | 0.999 | ✅ |
| [3] (1,1)tiltB | 300 | 1.3 | 0.999 | ✅ |
| [3] (1,1)tiltB | 400 | 6.7 | 0.997 | ✅ |
| [4] (2,-2)astig | 300 | 3.7 | 0.997 | ✅ |
| [4] (2,-2)astig | 400 | 5.2 | 0.989 | ✅ |
| [5] (2,0)defocus | 300 | 2.0 | 0.971 | ✅ |
| [5] (2,0)defocus | 400 | 10.5 | 0.969 | ✅ |
| [6] (2,2)astig | 300 | 0.2 | 0.997 | ✅ |
| [6] (2,2)astig | 400 | 7.2 | 0.995 | ✅ |
| [7] (3,-3)tf | 300 | 6.3 | 0.988 | ✅ |
| [7] (3,-3)tf | 400 | 5.0 | 0.948 | ✅ |
| [8] (3,-1)coma | 300 | 5.5 | 0.920 | ✅ |
| [8] (3,-1)coma | 400 | 1.7 | 0.999 | ✅ |
| [9] (3,1)coma | 300 | 2.5 | 0.963 | ✅ |
| [9] (3,1)coma | 400 | 0.7 | 0.999 | ✅ |
| [10] (3,3)tf | 300 | 1.8 | 0.991 | ✅ |
| [10] (3,3)tf | 400 | 12.3 | 0.988 | ✅ |
| [11] (4,-4)qf | 300 | 8.2 | 0.982 | ✅ |
| [11] (4,-4)qf | 400 | 15.2 | 0.916 | ❌ |
| [12] (4,-2)2a | 300 | 3.5 | 0.990 | ✅ |
| [12] (4,-2)2a | 400 | 3.1 | 0.996 | ✅ |
| [13] (4,0)spherical | 300 | 5.4 | 0.993 | ✅ |
| [13] (4,0)spherical | 400 | 2.1 | 0.848 | ❌ |
| [14] (4,2)2a | 300 | 3.5 | 0.986 | ✅ |
| [14] (4,2)2a | 400 | 1.4 | 0.997 | ✅ |
| [15] (4,4)qf | 300 | 5.4 | 0.987 | ✅ |
| [15] (4,4)qf | 400 | 10.6 | 0.986 | ✅ |

### 3. 检测 (Detection)

#### 3.1 异常点诊断

![outlier](zernike_response_matrix_report/figures/06_outlier_diagnosis.png)

**根因**: Zernike 半径 ≈ 光束半径 (实测 200px) 时, 大幅度高阶模式把子孔径光斑推出有效区 → DLL Zernike 拟合崩溃, `|resp|` 暴涨 (实测最高 4288)。故标定应取 **R ≈ 1.5×光束半径**。

#### 3.2 离线反解验证

![inverse](zernike_response_matrix_report/figures/07_inverse_demo.png)

合成像差 `[5]defocus=+0.30λ, [9]coma=−0.20λ` → 反解 `c = pinv(M) @ w`, 残差 ‖Mc−w‖=0.0369 / ‖w‖=0.3606 → **降低 89.8%**。

#### 3.3 实测闭环矫正

![closed loop](zernike_response_matrix_report/figures/08_closed_loop.png)

恢复 WFS 内部参考后, 矫正前 RMS=0.3564λ → 矫正后 0.2065λ (**42.1%**)。

> 该轮使用被 R=200 异常点污染的矩阵 (见 3.1), 改善有限; 修正后应以 R≈300px 重跑阶段 2/3。

### 4. 产物

| 产物 | 路径 |
|---|---|
| 响应矩阵 h5 | `D:\Projects\TIFO\AO-shaping\data\zernike_correction\debug_20260916_004944\matrix.h5` |
| 矩阵 + 原始读数 json | `D:\Projects\TIFO\AO-shaping\data\zernike_correction\debug_20260916_004944\matrix.json` |
| 多尺寸扫描报告 | `D:\Projects\TIFO\AO-shaping\data\zernike_correction\report_20260916_004944.json` |
| 多尺寸原始扫描 | `D:\Projects\TIFO\AO-shaping\data\zernike_correction\raw_scan_20260916_004944.json` |
| 标定工具 | `src/ao_shaping/tools/slm/slm_zernike_response.py` |
| 三阶段工具 | `src/ao_shaping/tools/slm/slm_zernike_correction.py` |
| 共享模块 | `src/ao_shaping/tools/slm/slm_zernike_common.py` |

