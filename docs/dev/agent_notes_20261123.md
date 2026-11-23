# Agent 当日记录 2026-11-23（未提交工作树改动；本轮占用设备实测一次）

设备状态：用户确认激光在线。SLM #1 USB 已恢复（FTDI FT601 Bridge 枚举 OK），大恒
MER2-507-23GM 可连。本轮占用设备完成一次实验系统衍射效率探针验证，验证曝光预检与质量门禁
在实验系统上的实际行为，跑完正常关闭、释放台面锁。

## 1. 实测：曝光门禁实验系统验证

**占用**：SLM #1（1064 nm，驱动板 `22030108`）+ 大恒 CCD cam 0；激光由用户开启。
**命令**：

```bash
$env:PYTHONPATH = "D:\Projects\TIFO\AO-shaping\libs"; python -m ao_shaping.tools.slm.calibration.diffraction_efficiency probe \
  --slm-number 1 --slm-wavelength 1064 --cam-type daheng --exposure-ms 0.1 \
  --period-px 16 --axis x --separation-px 471 --lobe-axis y \
  -o data/diffraction_efficiency/probe_real_20261123_gate
```

> 注意：`gxipy` 的 import 需要把**包父目录** `libs` 放进 `PYTHONPATH`（`libs\gxipy` 会导致
> `ModuleNotFoundError`，因包内用 `from gxipy.gxiapi import *` 绝对导入）。

**结果**：`quality.qualified=true`，CLI 正常退出（0），产物落盘
`data/diffraction_efficiency/probe_real_20261123_gate/{summary.json,summary.csv}`。

- **曝光预检**：请求 0.1 ms → 阶梯收敛到 **0.02 ms**（`reads=9`，`status=ok`），平场峰值
  104 counts / 255 = **40.8%**，落在接受带 25%–80% 内（目标 45%）。两帧（平场/闪耀）共用
  此曝光；暗帧 `dark_exposure_ms=0.02`、16 帧。
- **饱和统计**：平场与闪耀两帧 `clipped_px=0`，**无像素顶到满量程**——与 20261121 的
  255/255 过曝帧（作废）形成对照，门禁在实验系统上按预期拦截了过曝场景（该次 0.1 ms 请求被
  预检压到 0.02 ms 后即无截断）。
- **级次结果（相对指标，未定标，非绝对衍射效率）**：
  - 0 级窗口能量占比 0.6022；**−1 级 0.3978**（闪耀级次）；+1 级 0.0（弱侧泄漏 floor，
    非共轭级次，符合预期）。
  - `relative_efficiency.blaze_energy_share = 0.1927`（相对平场零级窗口能量）。
  - 0 级位置 (1106, 1229)，光斑 FWHM 13.0 px，ROI 半径 19.5 px（按 FWHM 定标）。
- **几何自检**：给定间距 471 px，实测 −1 级偏移 −471.39 px，`separation_check.consistent=true`。

**释放**：SLM #1 成功关闭，台面互斥锁已释放，CCD 随进程退出释放；无异常、无二次占用。

## 2. 结论

- 曝光预检 + 质量门禁在实验系统上端到端验证通过：过曝场景不再流入正式计算，未过门禁时 CLI
  会非零退出，产物仍落盘可复核。
- 复刻 20261121 参数（0.1 ms 请求、gray 993、period 16、axis x）并显式给标定参数
  （separation 471 / lobe-axis y）后，测得结果稳定、几何自检一致，可作为该探针的
  「带门禁」基准运行。

## 3. 待办 / 未提交状态

- 本次及 20261121–1122 的探针重构改动均**未提交**，工作树存在其他会话暂存的无关文件；
  未经明确授权不提交、不推送。

## 4. 效率匹配窗口：主采集完成（20261123_050735Z）

**占用**：SLM #1（1064 nm，`22030108`）+ 大恒 MER2-507-23GM cam 0 + 1064 nm 激光；
采集结束 `release.json` 记 `slm_flat_gray0=ok / slm_closed=true / camera_closed=true /
lock_released=true / closed_normally=true`，`released.json` 双写一致，无驱动板假死。

**两次六步探测（采集前门）**：

| 曝光 ms | 降功率前峰值 | 降功率后峰值 |
|---:|---:|---:|
| 0.02 | 2009 | 1409 |
| 0.03 | 3608 | 2404 |
| 0.04 | 3761 | 2462 |
| 0.05 | **4095 饱和** | 3688 |
| 0.06 | — | 3278 |

- 第一次（`efficiency_match_20261123`，05:00Z）verdict = **low_snr**：0.05 ms 平场 0 级已顶满
  量程（饱和线 3891），无可用窗口——冻结曝光已不可用。
- 用户降激光功率后第二次（`..._ladder2`，05:05Z）verdict = **ok**，可用窗口
  0.02–0.06 ms，噪声底 20.9 counts。
- **采集前修订**：曝光 0.05 → **0.04 ms**（峰值 2462，满量程 60%，距饱和 37%；0.03 仅高 2.4%
  但无同档暗帧）；暗帧按用户「复用之前采集的暗帧」决定，用库内
  `dark-exp-0.04ms.npz`（12-bit / 16 帧 / mean 14.97），`rel_tol=0.02` 精确命中。已回填
  `hw_window_20261123_efficiency_match_prereg.md` §1/§3.4/§4.3 与计划回填节第 5、6 条，
  修订时点在**任何采集帧下发之前**。

**主采集**：`python scripts/slm/efficiency_match_capture_20261123.py --execute
--out data/debug/efficiency_match/20261123_050735Z --exposure-ms 0.04`，`EXIT=0`，
13:07:35–13:09:00 本地。产物 16 帧全采（`raw/` 14 张图像 + 1 张共享库暗帧），
`config.exposure.readback_ms=0.04`、`all_frames_same_exposure=true`、
`dark_source=camera_library`、每帧 `saturated_pixels=0`；各相位峰值：flat
2017/2217/2151/2231/2183/2068，blaze 1039/1178/1138/1054，shaping 1237/1203/1234/1142
（均 < 饱和线 3891，最高 2231 = 满量程 54%）。

**踩坑（已修）**：分析器原 `_resolve_raw_path` 依赖 journal 的 `raw_file` 字段，而采集端 journal
不写该字段、只给 `frame_index`，raw 命名是 `{frame_index:04d}_{id}.npy`；其兜底 glob 的序号
位置还写反了（`*_[0-9][0-9][0-9][0-9]_{id}.npy`），导致真实数据上 14 帧全被跳过。
修复方向：优先用 `frame_index` 拼名，兜底 glob 改为 `[0-9][0-9][0-9][0-9]_{id}.npy`，
`dark_r2` 回落 base_id 解析到共享 `dark_library_*.npy`。教训：离线合成数据用 `raw_file` 字段
掩盖了真实 journal 的字段差异，端到端产物验证不可省。

**离线分析（已出结论）**：`analysis.json` + `EM_REPORT.md` 已落盘。verdict =
**undeterminable**，reason = `out_of_range_energy_check_background_or_saturation`；触发门
`energy_out_of_range`、`group_sd`、`drift_rounds`（其余 false）。ηs 逐帧为负（−0.103…−0.183，
即 S < S0），ηd ≈ 0.530–0.553，`|mean d|=0.667`，G1 组内 sd 0.0106，ηs sd 0.0552，
F0 跨帧相对变化 4.5–9.4%、质心位移最大 3.12 px，drift_rounds=4 > 1。按预注册口径：
不可判定门优先级高于 matched/mismatched，越界部分**不作物理解释**，不外推、不改判据；
不据此写标定、不判定任何模型解禁。修复仅有分析器路径解析与暗帧去重，未放宽任何阈值，
43 个用例通过、self-test 四场景 PASS、ruff clean（fixer 回报，已核对 analysis.json 门快照）。

## 5. #247 波形判别窗口：采集 + 离线分析完成（20261123_051615Z / hw_window_20261123_issue247_waveform_probe）

**扫描结论**：`docs/dev/` 下未完成采集的实验计划仅此项——预注册写于今日 09:16，60 帧从未
下发；效率匹配、动态解耦 T4/T5、#238 run1/run4、#235、P2a/P2b 均已有落盘数据。

**设备占用与释放**：13:16:16–13:17:xx 本地六步探测（session `issue247_waveform_20261123`，
证据 `data/debug/optical_path_probe/issue247_waveform_20261123/20261123_051615Z/`，
`verdict=ok`，可用曝光 0.03–0.06 ms，噪声底 20.9 counts）；13:18:31–13:19:18 主采集。
设备为 SLM #1（1064 nm）+ 大恒 MER2-507-23GM cam 0 + 1064 nm 激光，一次 open/close；
`released.json` 记 `slm_flat_gray0=ok / slm_closed=true / camera_closed=true /
lock_released=true / closed_normally=true / status=ok / error=null`，无残留 bench 进程。

**采集前修订**：预注册 §1 冻结 0.05 ms；当前功率下距 §4 饱和门仅 0–17% 余量（20 分钟前
同档 3688 counts 已越门），故改 **0.04 ms**（本轮实测 2199，外推最坏 2954 < 3685.5），
修订写进预注册 §8、在下发任何图样之前完成，判据/结论枚举/帧数一律未动。

**采集**：`python scripts/slm/issue247_geometry_capture_20261120.py --plan waveform --execute
--out data/debug/hw_window_20261123_issue247_waveform_probe --exposure-ms 0.04 --settle-s 0.8
--saturation-fraction 0.9`，`EXIT=0`，15 图样 × 4 帧 = 60 帧全采、`saturation_count=0`、
`status=ok`、`error=null`。

**离线**：`waveform_probe_analysis.json` + `WAVEFORM_PROBE_REPORT.md` 已出，verdict =
`patch_position_specific`；flat 漂移 1.75 px < 2 px 门、无饱和；`axis_gate`、`scale_gate`、
`patch_gate` 未通过。原始帧独立核验（argmax + 离轴亮斑质心，非分析器路径）：x 族 ±1 级距
0 级约 ±94 px 沿 CCD 列；y 族 ±1 级距 0 级约 ±64 px **也沿 CCD 列**（角度差 ≈0.26°），
`blaze_patch_x` 则沿行偏移 ≈86 px——与分析器测量一致，说明不是分析器坐标读错，而是本窗口
实测的面板轴响应异常。`patch_position_specific` 是预注册优先级标签（patch 门先判定），
但 axis/scale 门同时未过，不能解读为「只有 patch 异常」。分析器 `--self-test` 3 场景 PASS、
exit 0。结论仅作记录：不写任何标定、不改映射/相位 LUT、不改 #234/#236、不动 transfer
`light/none`、不发布 #238。
