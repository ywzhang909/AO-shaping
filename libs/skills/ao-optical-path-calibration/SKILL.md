---
name: ao-optical-path-calibration
version: 0.1.0
description: "Optical-path probing and device calibration for AO-shaping hardware runs (Santec SLM-200 + Daheng MER2-507-23GM, optional WFS/DM). Use before any on-device experiment to confirm devices are online, the optical path is valid, exposure stays inside the sensor range, and the SLM actually modulates; and to write a standard run record so cross-experiment comparisons can separate algorithm effects from optical-path or device drift."
---

# AO 光路探测与设备标定

**目的**：每次实验前后，用同一套检查项确认"设备在线 / 光路有效 / 曝光不超量程 / SLM 真在调制"，并把光路与设备状态写成标准记录。这样跨实验比较算法效果时，能排除光路或设备变动的干扰。

**依据**：`report/` 下历次实验报告（硬件采集数据、日报、诊断报告）与驱动源码逐条核对。每条数值括号内给出来源文件与行号，可以按锚点复核；本文不复制报告正文。

> 少数锚点指向 `.slim/worktrees/sim2f-model-correction/`：那几份报告只在对应工作树里，本树没有同名文件，已就地注明，不要在本树里找。

---

## 0. 硬性纪律（先读）

- 占用 SLM / 相机 / 激光前后，都要在回复中报告：占用了什么、做了什么、何时释放、是否正常关闭（`docs/dev/hardware_discipline.md`）。
- 设备被占用时，不要并行启动第二个占用同一设备的进程；带 `--wait-until-idle` 的脚本可以用。
- **DVI 模式打开 SLM 会挂起**，挂起后内存模式打开也可能挂。恢复方式是**给 SLM 控制器断电重启**，软件无法恢复（`report/fouriergsnet_pipeline/TODO.md:35, :71, :142`）。
- 背对背 open/close SLM 会触发驱动板假死（memory 中已记录），两次开关之间要间隔一个 settle 时长。
- **CCD 不做物理开窗**：实验期间保持全幅采集，需要缩小分析范围时只在后处理做算法裁切。物理开窗一旦设错，原始数据就丢了全幅上下文，无法从记录里复盘当时到底裁掉了什么（memory 中已记录此类教训）。

---

## 1. 设备标识（以序列号为准）

| 设备 | 标识 | 用途 / 备注 | 来源 |
|---|---|---|---|
| Santec SLM #1 驱动板 | `22030108` | 1064 nm，正常读取的驱动板序列号 | `src/ao_shaping/drivers/AGENTS.md`，`.slim/worktrees/sim2f-model-correction/report/shape_trials/experiment_summary_20261022.md:73` |
| Santec SLM #1 选板 | `23020027` | 仅在驱动板无应答时出现在日志中 | `.slim/worktrees/sim2f-model-correction/report/shape_trials/experiment_summary_20261022.md:73` |
| Santec SLM #2 | `23020026` | 532 nm | `src/ao_shaping/drivers/AGENTS.md` |
| Daheng MER2-507-23GM | `FJB24112232` | 2592×1944，像元 2.2 µm | `src/ao_shaping/drivers/ccd/AGENTS.md`，`data/ccd_configs/ccd/models.json` |
| MiiCam | `TP2408221418059418FD83E3A448D82` | 像元 2.9 µm（驱动写死） | `src/ao_shaping/drivers/ccd/miicam/driver.py:161` |

> 占位符保持原文，运行前由系统替换为真实值，不要自行猜测或替换。

**关键陷阱：静默回落**。驱动板不应答时，驱动会回落到选板 ID，面板停止调制，**且不报错**。探测脚本必须先调用 `Santec.require_drive_board()`（commit `ef3bb0ac`，`.slim/worktrees/sim2f-model-correction/report/shape_trials/experiment_summary_20261022.md:73`），并把读到的 ID 写入记录。

---

## 2. 量程与参数范围（源码核对）

### 2.1 Santec SLM-200（`src/ao_shaping/drivers/slm/santec/`）

| 参数 | 范围 / 值 | 源码 |
|---|---|---|
| 灰度 `set_grayscale` | 0 – 1023（10 bit） | `constants.py:172-173`，`driver.py:1375-1404` |
| 2π 灰度 `two_pi_grayscale` | 默认 1023；1064 nm 配 993；另一份报告记 956 | `driver.py:79, 103, 431`；见 §6 冲突 |
| 波长 `set_wavelength` | 450 – 1600 nm，默认 1064 | `constants.py:168-169, 159`，`driver.py:1230-1233` |
| SLM 设备号 | 1 – 8 | `constants.py:164-165` |
| 内存槽 | 轮换推荐 2 – 125 | `driver.py:125-126` |
| 面板分辨率 | 1920 × 1200 | `slm200_constants.py:23` |
| 像元 pitch | 8.0 µm（面板模型）；有效口径 7.8 µm | `slm200_constants.py:20-21` |
| 翻转时间上限 | 200 ms | `slm200_constants.py:25` |
| open 流程 | Close → 0.1 s → Open → 轮询就绪（10 × 0.1 s） | `driver.py:718-760, 2586-2614` |
| 状态码 | `SLM_OK=0`，`SLM_OTHER_ERROR=-1000`，`FT_DEVICE_NOT_OPENED=-10003` | `constants.py:13, 52, 57` |
| 状态查询 | `SLM_Ctrl_ReadSU`，非 0 抛 `SantecError` | `driver.py:2306-2316` |
| 上电/切模式/设波长/写 EEPROM | 各约 40 s（ReadSU 轮询） | `ao-device-handbook` ch02 |

### 2.2 Daheng MER2-507-23GM（`src/ao_shaping/drivers/ccd/daheng/`）

| 参数 | 范围 / 值 | 源码 |
|---|---|---|
| 曝光 | 启动时读设备真值 `ExposureTime.get_range()`（µs）；写入前钳到量程 | `driver.py:357-388`；兜底 0.011 – 10000 ms（`base.py:225`） |
| 曝光陷阱 | 0 ms "自动" 哨兵越界时**静默沿用旧曝光，导致饱和** | `driver.py:374-388` |
| 增益 | 固定 0.0，钳到 `[0, max_gain]` | `driver.py:395, 456-475` |
| 位深 | 8 / 12 / 16 bit → MONO8 / MONO12 / MONO16 | `driver.py:65-72` |
| 满量程 | 255 / 4095 / 65535（随生效位深） | `base.py:314-347` |
| 改位深 | 先 stream_off，写 PixelFormat，回读校验，再 stream_on | `driver.py:492+` |
| 曝光生效后 | 丢弃 3 帧 | `driver.py:429-475` |
| 自动曝光目标 | 目标 40、容差 5（0-255 灰度） | `base.py:228-229` |

### 2.3 其他（仅记录连接状态与量程，未在实验探测中使用时标注 "not used"）

- **MiiCam**：曝光 0.011 – 10000 ms（`src/ao_shaping/drivers/ccd/miicam/driver.py:157-158`）；改曝光必须 Stop → 设置 → Start；回调线程内禁调 SDK（死锁）。
- **Thorlabs WFS MLA150M-5C**：曝光 0.002 – 86 ms（`src/ao_shaping/drivers/wfs/thorlab_wfs.py:49-50`），仅单次曝光；先 `Calc*` 后 `Get*`；pupil（瞳面） 必须 `optimize_pupil()` 写回。
- **Micro-DM**：驱动板电压 −20 – 120 V（`src/ao_shaping/drivers/dm/micro/micro_constants.py:22-23`）。
- **NLight DM**：电压编码钳位原始值 [820, 4095]（`src/ao_shaping/drivers/dm/nlight/nlight_constants.py`）。

---

## 3. 探测流程（实验前，按顺序执行）

每一步失败都停止后续步骤，并在记录中写明失败点。

### Step 1 — 设备在线
1. 确认驱动板应答（`require_drive_board()`），记录驱动板 ID。
2. 相机枚举，记录型号与序列号；确认曝光量程真值。
3. 确认无其他进程占用（检查日志与 `--wait-until-idle`）。

### Step 2 — SLM 真在调制
- 写入明显的图样（如半幅 ramp 或棋盘），由相机确认图样变化。若相机帧无变化，判定为"面板未调制"，**不得继续**（对应静默回落故障）。
- 灰度指令需覆盖 0 与 2π 附近，确认 `two_pi_grayscale` 与当前波长一致。

### Step 3 — 光路有效
- 使用 `scripts/slm/probe_optical_path_validity.py` 扫描曝光梯度与离焦量。
  - 默认曝光梯度：`0.05, 0.2, 1, 2, 5, 10, 20, 50` ms；离焦幅度 `0, 0.5, 1.0, −1.0`。
  - 判定状态：`ok` / `low_snr` / `weak_path` / `no_response`。退出码 0 表示 ok，非 0 表示其余状态。
  - 输出：每步 npz、`optical_path_probe.json`、`.png`，位于 `data/debug/optical_path_probe/`。
- 只有 `ok` 才继续。`low_snr` 时调大曝光，但要回看饱和（Step 4）。

### Step 4 — 曝光不超量程
- 以**峰值计数**为准，不以曝光时间为准。
- 饱和判据：峰值 ≥ 约 **4015**（mono12 满量程 4095 的 250/255）。8 bit 时对应 ≈ 250。
- 已记录的上限参考：
  - 2026-10-22 扫描：30 ms 在 defocus +4.00 处饱和中止，5 ms 在另一次运行中也已饱和（`.slim/worktrees/sim2f-model-correction/report/shape_trials/experiment_summary_20261022.md:73`）。
  - 20 ms 时平坦峰仅 65 counts，settle 超时 7 s（`.slim/worktrees/sim2f-model-correction/report/shape_trials/experiment_summary_20261022.md:74`）。
  - 核心峰 0.2 ms 时已达 1091 – 1515 counts（`report/shape_trials/aperture_closed_20261109/aperture_closed_shaping_20261109.md:66`）。
- **曝光上限存在未解冲突**（见 §6）：`scripts/slm/check_flat_stability.py` 给出 2.07 ms，曝光梯度推算需数十 ms。在解决前，**每次实验必须现场用 Step 4 的峰值判据重新确定**，不得沿用旧值。
- 每个光路点都要单独自动曝光（焦面随光路变化，1.8 – 20 ms 跨度达 96 倍，`report/slm/focus_plane_report/focus_plane_20261019.md:49-51`）。固定曝光无法比较焦点位置。

### Step 5 — 几何与零级
- 光阑状态（开/闭）必须记录。光阑关闭时**零级未被抑制**，E0/E1 ≈ 1.2（`report/shape_trials/aperture_closed_20261109/aperture_closed_shaping_20261109.md:11, 66-67`）；`zero_radius` 仅屏蔽评分，不屏蔽光束。
- 相机焦面偏离 2f 焦面，需要 Fresnel 补偿，每天取值不同（0.559 / 0.626 / 0.70 m⁻¹，`report/slm/focus_plane_report/focus_plane_20261019.md:9, 152`）。当天的补偿值必须写入记录。
- 几何标定读取 `data/slm_geometry_calib.npz`（字段：`Kx, Ky, rotation_deg, fwhm0, resid_x, resid_y, crop_side, center, mirror, k_mean`），写入方 `scripts/slm/persist_live_geometry.py`。旧版本以日期存档。

### Step 6 — 结束与释放
- 设备关闭正常后再报告"已释放"。
- 写入运行记录（§4），并核对 `device_log.json`（由 `scripts/slm/absolute_fidelity_probe_20261116.py:816` 写入，记录每次 display 与 grab 的占用）。

---

## 4. 标准化记录（每次实验都要写）

每次实验在 `data/runs/<run_id>/` 下写入 `optical_path_record.json`。字段固定，方便跨实验比较。缺失字段写 `null` 并说明原因，**不得省略字段**。

```json
{
  "schema": "ao-optical-path-record/v1",
  "run_id": "<YYYYMMDD>_<HHMMSS>_<short_tag>",
  "date": "<YYYY-MM-DD>",
  "operator_note": "",
  "devices": {
    "slm": {
      "model": "Santec SLM-200",
      "drive_board_id": "<serial read from ReadSO/driver>",
      "option_board_id": "<serial, if read>",
      "slm_number": 1,
      "wavelength_nm": 1064,
      "two_pi_grayscale": 993,
      "panel_res": [1920, 1200],
      "readsu_status_at_open": 0,
      "open_mode": "memory",
      "power_cycled_this_session": false
    },
    "camera": {
      "model": "Daheng MER2-507-23GM",
      "serial": "<serial>",
      "bit_depth": 12,
      "full_scale": 4095,
      "exposure_range_ms": [0.0, 0.0],
      "exposure_used_ms": 0.0,
      "gain": 0.0,
      "averaging_frames": 1,
      "discard_frames_after_exposure_change": 3,
      "pixel_size_um": 2.2
    },
    "wfs": "not used",
    "dm": "not used"
  },
  "optical_path": {
    "config_id": "<e.g. 2f_1064_aperture_closed>",
    "aperture": "open | closed",
    "fresnel_compensation_m_inv": 0.0,
    "geometry_calib_file": "data/slm_geometry_calib.npz",
    "geometry_calib_mtime": "<ISO time>",
    "probe_status": "ok | low_snr | weak_path | no_response",
    "probe_report": "<path to optical_path_probe.json>"
  },
  "exposure_check": {
    "peak_counts_at_used_exposure": 0,
    "saturation_threshold_counts": 0,
    "saturated": false,
    "settle_timeout": false
  },
  "occupancy": {
    "acquired_at": "<ISO>",
    "released_at": "<ISO>",
    "closed_normally": true
  },
  "git_commit": "<short sha>",
  "missing": {"<dotted.path>": "<why this field is null>"}
}
```

**`missing` 是 schema 之外的唯一附加键**：记录里每个字段都必须出现，拿不到时写 `null`，原因放进 `missing`（点分路径 → 人话），不塞进数值字段（否则类型变成 `float | None | str`，跨实验比较要先判类型）。`missing` 为空表示这份记录完整。

**由探针 `scripts/slm/probe_optical_path_validity.py` 自动写入**（`src/ao_shaping/utils/io/optical_path_record.py::build_optical_path_record` 负责纯组装），与 `optical_path_probe.json` 同目录。写盘在放锁之后，`occupancy.released_at` 才是真实释放时刻；写盘失败只告警，不改变探测退出码。

**探针能自动填的**：`run_id` / `date` / SLM 设备信息 / 相机信息 / `exposure_used_ms` / `peak_counts_at_used_exposure` / `saturated` / `saturation_threshold_counts` / `probe_status` / `probe_report` / `geometry_calib_file` / `geometry_calib_mtime` / `occupancy` / `git_commit`。

**必须现场手工补的**（探针不读，会留 `null` 加原因）：
- `optical_path.aperture` —— 光阑开/闭；
- `optical_path.fresnel_compensation_m_inv` —— 当天焦面补偿；
- `exposure_check.settle_timeout` —— 探针用固定 settle 等待，不检测超时；
- `devices.camera.averaging_frames` / `discard_frames_after_exposure_change` —— 探针用单帧。

**饱和阈值的口径**：`saturation_threshold_counts` 由探针自己的 `SATURATION_FRACTION = 0.95` 乘 `full_scale` 得出（mono12 = 3890）。历史报告用的守卫是 250/255 × 4095 ≈ 4015，两者接近但不相等。比较历史与探针记录时以记录里的 `saturation_threshold_counts` 为准，不要套用 4015。

**全幅采集与裁切**：采集期间不做物理开窗，相机以全幅读出，原始数据保持完整上下文。需要缩小分析范围时，只在后处理阶段做算法裁切，并把裁切后的 ROI 与坐标记入 `operator_note`（或单独的后处理记录）。这样原始数据始终可复盘——物理开窗一旦设错，全幅上下文就永久丢失，无法从记录还原当时裁掉了什么。

**比较规则**：两次实验若 `optical_path.config_id`、`fresnel_compensation_m_inv`、`two_pi_grayscale`、`exposure_used_ms` 或 `geometry_calib_mtime` 不同，结论比较前先说明差异，不能直接把算法效果归因于算法本身。

---

## 5. 红线

1. 驱动板必须应答，不得在选板 ID 回落下继续实验。
2. 曝光必须按峰值判据现场确定；不得沿用历史曝光值。
3. 零级未被抑制时，不得把光阑关闭下的结果当作零级已抑制。
4. DVI 模式不得用于打开 SLM；死机只能断电。
5. 所有数值必须来自当天的记录或源码，不得凭记忆填写。

---

## 6. 未解冲突（使用前必须知晓）

1. **内存槽数**：`MAX_MEM_SLOTS=127`（1–127，`constants.py:21`）与 `MEMORY_NUMBER_MAX=128`（`constants.py:177`）不一致。
2. **1064 nm 的 2π 灰度**：993（`src/ao_shaping/drivers/AGENTS.md`）与 956（`.slim/worktrees/sim2f-model-correction/report/shape_trials/multishape_capability/diagnosis_and_transfer_20261023.md:131`）不一致。每次运行前用实测确认。
3. **曝光上限**：2.07 ms（`scripts/slm/check_flat_stability.py`，`report/slm/focus_plane_report/focus_plane_20261019.md:152-154`）与梯度推算的数十 ms 冲突。以 Step 4 峰值判据为准，结论待定。
4. **过期文档路径**：多份 `docs/tools/slm_*_probe.md` 仍引用 `drivers/slm/santec.py` 与 `tools/slm/slm_*.py`，实际代码在 `drivers/slm/santec/` 与 `tools/slm/panel/`。以源码为准。
5. **2π 灰度随波长变化（第三个数据点）**：532 nm 为 998（2026-09-15/16 日报），1064 nm 为 993（blackbox 报告）。报告未说明是否按波长变化。每次运行前实测。
6. **SLM #1 序列号**：跨报告章节与 09-08 日报给 `22030108`；09-15 与 09-16 日报给 `22030102`，同样称为 "SLM #1"。运行记录中的 `drive_board_id` 必须以当次 ReadSO 读数为准，不得从旧报告抄写。
7. **曝光口径**：blackbox 报告 3.165–4.32 ms、axial_plane 50–100 ms，与 2.07 ms 上限冲突。若 2.07 ms 只适用于 `scripts/slm/check_flat_stability.py` 的某个配置，应在脚本处注明；否则不能当作 Daheng 通用上限。
8. **2f 系统波长**：axial_plane 为 1064 nm，09-15/16 日报为 532 nm。两套光路未在报告中区分，比较时先核对 `wavelength_nm`。

其他说明：09-08 的"CCD 无响应"探测结果已被撤回，不可引用。diff_beam 的 `diff-beam` 命令已过时（runner 于 2026-10-01 删除）。`voltage_test.md:75–86` 的 1300-5.xlsx 有两组歧义键（IP 组，index）。

---

## 7. 待办（skill 之外）

- 解决上述冲突 1–3、5–8，并在对应源码或文档中修正。
- 现场手工补 `aperture` / `fresnel_compensation_m_inv` / `settle_timeout`：目前探针只能留 `null` 加原因，应在 `data/runs/<run_id>/` 落盘后由操作者补写并注明来源。
- 补采绝对强度保真度的实验数据（`scripts/slm/absolute_fidelity_probe_20261116.py`，commit `807d456a`）。
- 已补读：`blackbox_path`、`axial_plane`、`micro_deformable_mirror`、`diff_beam` 与三份 SLM 日报；结论已并入 §6。`daily_2026-09-15.md:335-338` 未读，涉及该处结论时需补看。
