# `tools/slm/` — SLM 台架工具与探针

SLM(Santec SLM-200)+ 远场相机(Daheng / MiiCam)2f-Fourier 台架的**独立工具**。
这些模块**不注册**为 `main.py` 的 Click 命令, 一律用
`python -m ao_shaping.tools.slm.<名字>` 直接运行。

> **启动任何优化 runner 之前**, 先跑
> [`docs/slm/pre_run_characterization.md`](../../../../docs/slm/pre_run_characterization.md)
> 里的三个探针(`slm_drift_probe` → `slm_floor_probe` → `slm_abba_probe`),
> 确认台架本身可信。**不要**跳过。
---

## 台架表征(先跑这个)

| 模块 | 作用 |
|---|---|
| **`slm_drift_probe`** | 平场漂移 + 曝光阶梯线性。用区域范数判漂移(**不用峰值** —— 实测不可复现) |
| **`slm_floor_probe`** | 测量本底、稳定时间拟合、SNR-vs-K。判定噪声是读噪声(`noise_limited`)还是漂移(`drift_limited`) |
| **`slm_abba_probe`** | 稠密随机相位**是否可分辨**。ABBA 消一阶漂移, 先量本底再判信号。回答"能不能去测转移矩阵" |

这四个模块共同的硬性纪律(违反会得到看似可信的错结论):

- 写相位用 `display_data(gray)` 且**不传** `memory_number`(固件对当前槽是
  no-op)与 `wait_time_s`(翻转时间估算会报 0 ms)
- 采集一律走 `capture_settled`(丢弃帧直到连续两次读数一致), **不用** `time.sleep`
- ROI 由 `argmax` 在**去噪后的平场**上定位一次, 之后**整轮冻结**
- 支持 `--no-hw`: 只打印采集计划、退出 0、不碰硬件(测试与 CI 用)

---

## 硬件状态与几何

| 模块 | 作用 |
|---|---|
| `slm_diagnose` | 三步自检: 面板冻结 / 调制能力 / 光强线性性。**注册为 `slm-diagnose`**。默认相机是 `miicam`, 大恒台架必须加 `--camera-type daheng` |
| `slm_tilt_probe` | 判定面板是否真的在调制(倾斜斜坡)。比光栅可靠: 位移只取决于周期 |
| `slm_panel_locate` | 在**面板坐标**上定位光斑(CCD 0 阶坐标 ≠ 面板坐标) |
| `slm_beam_extent` | 半平面随机相位边界扫描 → 测光斑**半径**(实测 450 px @ 中心) |
| `slm_phase_resolution` | 面板能否分辨**像素级**相位? 决定散斑标定路线是否可用 |
| `slm_exposure_check` | 相机是否真在手动曝光、固定设置下漂移多少 |
| `slm_phase_response` | 相位 → CCD 响应曲线(defocus/tilt/…) |
| `bench_kernels` | **共享测量内核** (合并自 `slm_bench_probe` + `slm_bench_metrics`): 设备由参数注入, 无 CLI。`display_and_average` / `capture_settled` / `measure_spot` / `tilt_shift_px` / 帧预处理与分析函数的出处 |
| `sweep_analysis` | **扫描/扫参分析助手** (合并自 `slm_scan_analysis` + `delta_explorer` + `slm_snr_probe`): 纯 numpy 分析、delta 探索、SNR 测量(含 ABBA 实现) |

## Zernike 与 WFS

| 模块 | 作用 |
|---|---|
| `slm_zernike_common` | Zernike 工具的共享常量与测量原语 |
| `slm_zernike_sweep_probe` | 光滑 Zernike 扫描(42 点), 含 Recorder 落盘 |
| `slm_zernike_response` | Zernike 模式 → WFS 响应矩阵标定 |
| `slm_zernike_correction` | Zernike 模式法波前矫正三阶段流程 |
| `slm_wfs_probe` | SLM + WFS 光强与 pupil 检查 |
| `slm_wfs_reference` | WFS 参考波前标定与倾斜线性度(三步) |
| `hadamard_calc` / `cartographer/` | Hadamard 掩膜与波前重建(子包) |

## 标定与其它

| 模块 | 作用 |
|---|---|
| `calibration.py` | SLM 标定工具包(单文件合并版) |
| `slm_lut_runner` | 灰度→相位 LUT 标定。**注册为 `slm-lut`** |
| `gray_response` | SLM 灰度 → 相机最大亮度响应(注意 SLM 幅度耦合: 周期 ≈ 2π) |
| `phase_capture` | 随机相位采集 |
| `slm_train_data_collect` | **训练数据采集器** (SLM + 远场 CCD, 可选第二台 CCD 记为 `pupil`)。按 `exposure_ms × cam_size` 交叉扫描采集, 产物可直接被 `ml/hwdataset` 索引; 用来补语料分布缺口 (`objective` 覆盖、曝光聚集、freeform 网格单一、视场单一)。`--no-hw` 只打印采集计划 |




---

## 约定

- **设备由参数注入**, 探针本体不构造硬件 → 测试可脱机跑
- **不在模块作用域 import 驱动**: 所有 `Santec` / `create_camera` 都在
  `main()` 内部导入, 保证 `--no-hw` 在无 SDK 的机器上可用
- 新增探针请配 `--no-hw` 与离线测试(`tests/ao_shaping/tools/slm/`)
- 需要硬件的测试用 `AO_RUN_HARDWARE=1` 门控, 默认 skip

## 测试

```bash
pytest tests/ao_shaping/tools/slm/ -q
```

全部离线(硬件用 mock 或门控 skip)。当前 **293 passed / 3 skipped**。
