"""SLM 相关独立工具包。

**不注册**为 `main.py` Click 命令, 一律 `python -m ao_shaping.tools.slm.<名字>` 运行。
每个探针**自带 `--no-hw`** (只打印采集计划、退出 0、不碰硬件), 所以 CI 可跑。

> 🔑 **启动任何 GS / GSNet / SPGD runner 之前先跑表征探针**:
> [`slm_drift_probe`](#台架表征先跑这个) → [`slm_floor_probe`](#台架表征先跑这个)
> → [`slm_abba_probe`](#台架表征先跑这个)。完整流程与验收阈值见
> `docs/slm/pre_run_characterization.md`。

## 共享测量内核 (设备由参数注入, 无 CLI)

| 模块 | 内容 |
|---|---|
| `slm_bench_probe` | **所有探针的公共内核**: 平滑/去毛刺帧、光斑 FWHM+质心+中心凹陷度、0 阶能量占比、面板 Zernike 放置、倾斜斜坡、线性拟合、`display_and_average` / `capture_settled` (稳定判据)。无 CLI |
| `slm_bench_metrics` | 平场漂移 / 曝光阶梯 / SNR-vs-K 的**纯 numpy 分析内核** (无设备、无 I/O, CI 可跑)。含两种**故意不同**的帧预处理 (`finite_clip` / `finite_median_subtract`) |
| `slm_scan_analysis` | 扫描报告助手 (纯 numpy/stdlib): Z-score 异常点剔除、灰度扫描分批统计、线性度、`latest_match` |

## 台架表征 (先跑这个)

| 模块 | 内容 |
|---|---|
| `slm_drift_probe` | 平场漂移 + 曝光阶梯线性。用区域范数判漂移 (**不用峰值** —— 实测同设置两次运行峰值读到 100 与 23, 而 box sum 稳到 0.2%)。判据 `monotonic` / `non_monotonic` / `saturated` |
| `slm_floor_probe` | 测本底 + 稳定时间拟合 + SNR-vs-K。回答噪声是读噪声 (`noise_limited`) 还是漂移 (`drift_limited`) —— 后者说明**降 delta 无用**, 要改稳定判据或改用 ABBA |
| `slm_abba_probe` | 稠密随机相位**是否可分辨**。ABBA (`+ - - +`) 消一阶漂移并先量本底。`verdict=unusable` 时不要去测转移矩阵 |

## 硬件状态与几何

| 模块 | 内容 |
|---|---|
| `slm_diagnose` | 三步自检: 面板冻结 / 调制能力 / 光强线性性。**注册为 `slm-diagnose`**。默认相机是 `miicam`, 大恒台架必须加 `--camera-type daheng` |
| `slm_tilt_probe` | **判定面板是否真的在调制** (倾斜斜坡)。比光栅可靠: 光斑位移只取决于斜坡周期, 与衍射效率无关。用于推翻过一次"面板冻结"的假故障 |
| `slm_panel_locate` | 面板坐标上定位光斑中心 (扫描随机相位圆盘)。相机 0 阶**不是**面板坐标: 两轴互换 90° 且尺度差 >10 倍, 换算会把相位 roll 到不含光斑处 |
| `slm_beam_extent` | 半平面随机相位边界扫描测光斑中心/半径。文档里曾长期写着 "~192 SLM px" 而实测 ~450 px, 用陈旧数字会只调制一半光瞳 |
| `slm_phase_resolution` | 比较逐像素随机相位与光滑 Zernike 相位, 判定面板**等效相位分辨率**。实测逐像素随机相位对远场毫无影响而 Zernike 逐阶生效 ⇒ 面板无法分辨像素级相位, 散斑相关度标定路线在此台架不可用, 必须走 sweep |
| `slm_exposure_check` | 相机自动曝光状态 + 固定设置下漂移。区分"相机漂移"与"SLM 保留上次图案"两种同样表现为"变暗 4 倍"的原因 |
| `slm_phase_response` | SLM 相位 → CCD 响应探针 (离焦/透镜用例 + 共享采集/渲染通道)。设备由参数传入 |
| `slm_snr_probe` | SLM 扰动灵敏度 (SNR) 测量: 噪底 sigma / `+ - - +` 回文漂移对消 `abba_signal` / 单模式与多模式 (SPGD 实际) SNR。用于按实测 SNR 选 `--delta`; 设备由参数注入 ⇒ 可脱机单测 |

## Zernike 与 WFS

| 模块 | 内容 |
|---|---|
| `slm_zernike_common` | Zernike 工具的**共享常量与测量原语**。fan-in 最高 (`zernike_matrix_runner.py`、`gui/slm/slm_calibration_ui.py` 都依赖它) |
| `slm_zernike_sweep_probe` | **光滑 Zernike 台架扫描探针**。一次扫描采 ramp + tilt + defocus + astig_x/y + coma_x/y + spherical 共 42 点 (前置一个 flat 参考), 逐点用**稳定判据** (而非固定等待) 读帧, 落盘 `sweep_records.npz` + 标准 Recorder 产物 (含下发相位与 CCD 帧)。`--no-hw` 只打印采集计划。`scripts/model_in_loop_hw_runbook.py --stage sweep` 的采集内核已收敛到本模块 (`zernike_panel` / `capture_settled` 单一来源) |
| `slm_zernike_response` | Zernike 模式 → WFS 响应矩阵标定 |
| `slm_zernike_correction` | Zernike 模式法波前矫正三阶段流程 |
| `slm_wfs_probe` | SLM + WFS 光强与 pupil 检查 |
| `slm_wfs_reference` | WFS 参考波前标定与倾斜线性度 (三步) |

## 标定与其它

| 模块 | 内容 |
|---|---|
| `slm_lut_runner` | 灰度 → 相位 LUT 校准 (CLI, **注册为 `slm-lut`**)。见下方「LUT 路径约定」 |
| `calibration` | SLM 标定工具包。`geo` 几何标定 + `shift` 平移标定; 含**已废弃**的 `SLMLUTCalibrator` (见下) |
| `gray_response` | SLM 灰度 → 相机最大亮度响应 (注意 SLM 振幅耦合: 周期 ≈ 2π ≈ 993 灰度) |
| `phase_capture` | 随机相位采集 (湍流 / Zernike 相位数据集) |
| `delta_explorer` | 扫 `delta` 看 SPGD 收敛性 |
| `cartographer/` | SLM 标定综合工具子包 (余弦图样 / Hartmann 波前重建 / 灰度-LUT / 动态补偿) |

## LUT 路径约定

- **canonical (驱动可消费)**: `slm_lut_runner.py` + `utils/slm/slm_lut.py`
  (`save_lut()`) → `lut_forward.csv` / `lut_inverse.csv` → `Santec.load_lut()`。
  唯一进入 SLM 灰度↔相位补偿的 LUT。
- **deprecated**: `calibration.py::SLMLUTCalibrator` (legacy 8-bit 自参考干涉法),
  保留供参考/单测, 已移出 CLI, 直接使用触发 `DeprecationWarning`。
- **research**: `cartographer/phase_grayscale_lut.py` (WFS 实测研究 LUT, 仅供
  cartographer 闭环, **不可**被 `Santec.load_lut` 消费)。

## 四条台架铁律 (违反会得到看似可信的错结论)

1. **暗帧不可用裸 `argmax` 定位光斑** —— peak 22~46 而帧均值 0.26 时单个热像素就能抢到
   argmax, 参考帧质心曾在 60 px 内自漂, 足以把健康的板判成"没动"。
2. **SLM 保留上次显示的图案** —— 下发平场**之前**读到的"平场"其实是上一轮的散斑。
3. **固定 `memory_number=` 是固件 no-op** —— 同一槽连续 `display_memory` 不刷新面板,
   之后每帧都是旧图。用 `display_data()` 不传 `memory_number`, 或按仓库推荐做法轮换槽位。
4. **液晶要等"稳定", 不是等"够久"** —— 驱动自动翻转时间估算按灰度图变化量给等待,
   两个灰度统计相近的相位会让它报 **0.0 ms**。实测同一斜坡首次读 fwhm 43.2 px、
   3 秒后 12.8 px、质心移 62 px。`display_and_average` / `capture_settled` 已改为
   丢弃帧直到连续两次读数一致。

## 包级不做 eager re-export

子模块在模块作用域 import `click` / `loguru` / numpy 与相机、SLM 驱动。原 `__init__.py`
曾一次性 re-export 60+ 个名字, 代价是**任何** `import ao_shaping.tools.slm.<x>` 都会把
整套探针模块拉进来。需要符号时**直接从子模块导入** (与 `tools/micro_dm/` 一致)。
`tests/ao_shaping/tools/` 下的测试锁定了这一点。
"""

# 故意留空: 见上方「包级不做 eager re-export」。
__all__: list[str] = []