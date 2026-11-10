# 两套光路现状描述（截至 2026-11-09）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线+硬件
> **说明**: 实验室两套光路(按 SLM 序列号区分)的设备清单与截至 2026-11-09 的已知状态: 两台 Santec SLM-200 —— 光路A(SLM#1 序列号 22030108, 1064nm 2f-Fourier + Daheng MER2 相机)为本项目主用台, 11-09 零级未压制(E0/E1≈1.2)但几何映射未变, 已落 locate() 零级掩膜对策; 光路B(SLM#2 序列号 23020026, 532nm 中继 + Thorlabs MLA150M-5C WFS)停留在 2026-09/10 标定; 光路A的 11-09 光阑关闭后整形复现与重放见 aperture_closed_20261109/aperture_closed_shaping_20261109.md
<!-- provenance:end -->

---

## 摘要

> 后续更新见 [2026-11-10 下线前诊断状态日志](optical_path_status_20261110.md)；本文件保留 11-09 的历史判断，不代表光路调整后的状态。

实验室有**两台 Santec SLM-200，对应两套独立光路**，以 SLM 序列号为唯一可靠区分（历史上仓库内曾出现三路序列号冲突，2026-10-25 已收口，见 §1.3）。**本项目（shape_trials 系列）的全部整形实测均在光路 A（1064 nm 2f-Fourier 实验台）上进行**；光路 B（532 nm 中继 + WFS）自 2026-09/10 标定后未再更新实测。2026-11-09 光路 A 质量下降（零级未压制，E0/E1 从基线 0.08–0.35 恶化到 ≈1.2），但几何映射未变；本轮已通过 `locate()` 零级掩膜修复（`ZERO_ORDER_EXCLUDE_RADIUS=60`）在采集侧规避。

## 1. 光路识别（按 SLM 序列号）

### 1.1 光路 A — 1064 nm 2f-Fourier 实验台（本项目主用）

| 设备 | 标识 | 说明 |
|------|------|------|
| SLM#1 | 序列号 `22030108` | Santec SLM-200，1920×1200，10-bit，@1064 nm 2π=993 灰度（设备动态查询） |
| 相机 | Daheng MER2-507-23GM `FJB24112232` | 2592×1944，uint8，像元 2.2 µm，IP 192.168.0.11，`cam_id=0` |
| 布局 | SLM 前焦面 125 mm → f=125 mm 透镜 → CCD 后焦面 | CCD 坐标=空间频率；**0 级一律用 `argmax` 定位**（帧全局最大），非相机几何中心 |

### 1.2 光路 B — 532 nm 中继 + WFS 台

| 设备 | 标识 | 说明 |
|------|------|------|
| SLM#2 | 序列号 `23020026` | Santec SLM-200，1920×1200，@532 nm 2π=998 灰度；**与 SLM#1 为两台独立设备** |
| WFS | Thorlabs `M01219666` | MLA150M-5C（27×27 spot），曝光 ≈4 ms（`data/calibration/wfs_light_pupil_probe.json`） |

### 1.3 序列号冲突收口（引用前必读）

仓库内曾出现三路 SLM 序列号 `22030108` / `22030102` / `23020026`，2026-10-25 已收口（#58 关闭，`src/ao_shaping/drivers/AGENTS.md` §硬件事实）：**两台设备**——`22030108` = SLM#1 @1064 nm（唯一读数，权威）；`23020026` = SLM#2 @532 nm；`22030102` 为 SLM#2 的历史读数（如 `data/calibration/wfs_light_pupil_probe.json` 中仍记该值）。**引用旧数据时先按此表换算，不要以旧序列号另立设备。**

## 2. 光路 A 现状（2026-11-09 实测）

### 2.1 当前状态：光路质量下降，零级未压制；几何未变

| 项 | 2026-11-09 实测 | 10-28 基线 | 判定 |
|----|----------------|-----------|------|
| 前置探针 verdict | ok（`data/debug/optical_path_probe/repro1200_precheck/20261109_071503Z/optical_path_probe.json`） | ok | 光路可用 |
| 可用曝光窗 | **0.05–0.2 ms** | 1–10 ms | 强度约基线 7.5×（吞吐 840 counts/ms） |
| 0 级位置（argmax） | **(1062, 1168)**，贴近传感器底边 | — | `window_span` 须 ≤760（900 会越出传感器） |
| 零/一级能量比 E0/E1 | **≈1.2** | 0.08–0.35 | **零级未压制——主要退化** |
| 光斑 FWHM | 80–182 px | 57–110 px | 光斑更宽 |
| 一级几何间距（period=64/128） | x64≈225 / y64≈237 / x128≈114 / y128≈116 px | ≈226/233 | **几何映射未变** |
| 几何矩阵 / cond | ≈[[-96, 7504], [-7248, -64]]，cond≈1.036 | 同量级 | 正常 |
| panel-locate | 单峰 (960,720)，中心 dE/E=−0.2945，平场漂移 ≤2.70% | — | 面板响应健康 |
| slm-diagnose | freeze PASS、**modulate FAIL**（rel_spread=0.047<0.15）、linearity PASS | — | modulate 判据用均匀灰度桶能量路径，不能单独判定 campaign 写入失效 |

**结论**：退化集中在「零级未被压制 + 光斑变宽」，而非几何/相位深度/写入失效（后三者假设均已排查推翻）。

### 2.2 已落地的对策（本轮源码改动，未提交）

`src/ao_shaping/tools/slm/transfer/capture.py`：`locate()` 新增零级排除盘（`ZERO_ORDER_EXCLUDE_RADIUS=60.0`，argmax 前磁盘掩膜），仅 ramp 帧排除本次 `reference_start` 零点邻域。半径 60 px 由扫描 30–60 选定：更小时 period=128 的一级（距零级约 57 px）会被零级尾流拖偏。离线用 `rampcheck_g384_20261109` 帧验证一级间距恢复到 §2.1 数值；回归测试 `tests/ao_shaping/tools/slm/transfer/test_locate_zero_order.py`。

### 2.3 引用基线

- 10-28 成功配方与原始帧：`data/debug/slm_transfer/multishape_scale_20261028/`（s1200、s600_fixed）
- 基线报告：[multishape_scale_wallclock_20261028.md](multishape_scale/multishape_scale_wallclock_20261028.md)（§6.5 耗时、§8.2 光路探针）
- 本轮 campaign：`data/debug/slm_transfer/repro1200_aperture_closed_20261109_1910/`（光阑关闭后复现，已完成；见 [aperture_closed_shaping_20261109.md](aperture_closed_20261109/aperture_closed_shaping_20261109.md)）
- 本轮共同重放：`common_replay/` 成功（30 条目、曝光 0.05/0.2 ms、`released.json` 已写，12:29:51.384509Z）；旧 50 ms 失败目录改名 `common_replay_aborted_50ms_20261109/`（首条 core_peak=4095 全饱和，改用 0.05/0.2 ms）。

## 3. 光路 B 已知状态（截至 2026-09/10，较旧）

- **最近一次系统标定为 2026-09 实测**：SLM-WFS 对位 shift=(106,40)（附加倾斜 0.854λ→0.0245λ，降低 97.1%；`src/ao_shaping/drivers/slm/AGENTS.md`）；Zernike 矩阵 cond≈5.12。
- `data/calibration/wfs_light_pupil_probe.json`（2026-09-30 前后）：729/729 spot 全亮，max 1480 / mean 520 counts @≈4 ms，pupil（瞳面） 优化 (-0.04, -0.17, 3.69, 3.82)。注意该文件 `slm.serial` 仍记历史序列号 `22030102`（见 §1.3）。
- **轴映射警告**（本机中继光路实测）：SLM-x 平移驱动 WFS Noll3（y-tilt）、SLM-y 驱动 Noll2（x-tip），标定/对位**不要假设 SLM 轴与 WFS 轴对应**。
- **2026-11 无光路 B 新实测**；本轮全部实验（含 2026-11 的 `zernike_calibration` 进度文件与 issue229 探针，`slm_number=1`）均在光路 A。复用任何 532 nm 台结论前须重新标定。

## 4. 使用纪律

1. **区分光路先看 SLM 序列号**，不要靠 "SLM#1/#2" 或旧数据里的历史序列号。
2. 光路 A 复用旧模型/标定前，先跑 `scripts/slm/probe_optical_path_validity.py`（退出码非 0 即中止）；当前状态下曝光窗与 10-28 不同（0.05–0.2 vs 1–10 ms），模型迁移性须重新验证。
3. 0 级定位永远 `argmax`；当前零点贴近传感器底边 (1062,1168)，采集窗口参数（如 `window_span`）须按此核查。
4. 光路 B 实验前须重新确认 shift/标定状态，2026-09 的 shift=(106,40) 不应假设仍有效。
