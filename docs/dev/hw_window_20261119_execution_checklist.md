# 全包硬件窗口：逐段可执行命令清单 + 现场读回检查表（2026-11-19）

> **本文件是离线准备，不构成设备授权，也没有执行过任何设备命令。** 执行前后须按
> `docs/dev/hardware_discipline.md` 在回复中逐条报告：占用了什么设备（SLM #编号 / 相机 /
> 激光）、做了什么、何时释放、是否正常关闭。接设备前还须按
> `libs/skills/ao-optical-path-calibration/SKILL.md` 完成六步光路探测，任一步失败即止步。
>
> 本文所有命令均为**待填占位骨架**：`<...>` 与 `{{...}}` 都是现场读回/新建的占位值，
> 严禁写死历史几何或历史曝光。只有真实设备命令才带 `--execute`（S0 探针例外，见下）。
> dry-run（不带 `--execute`）不建目录、不导入设备模块，可以随时安全核对 argv。

## 0. 冻结口径与版本核对

评分口径以 `docs/dev/hw_window_20261119_preregistration.md` 为准（§1 范围、§2 冻结判据、
§3 现场读回字段、§4 纪律）。预注册时冻结的脚本 SHA256(前16) 如下；必须在执行前复核，
不一致须在预注册表登记新哈希并重签，否则评分作废：

| 脚本 | 预注册 SHA256(前16)，含 15:46 §0.3 重签 | 本次离线核对时的磁盘值（2026-11-19T15:46:32Z） |
|---|---|---|
| `validate_transfer_shapes_hardware.py` | `7EF7F25A121E8824` | `7EF7F25A121E8824` ✅ |
| `replay_transfer_shapes_hardware.py` | `5727FD8F459E5FCB` | `5727FD8F459E5FCB` ✅ |
| `run_transfer_window_20261119.py` | `39C3E00D51FCC0E4` | `39C3E00D51FCC0E4` ✅ |
| `run_issue238_period_sweep_20261119.py` | `48BBF7198FC3C691` | `48BBF7198FC3C691` ✅ |
| `protocol.py` | `32078F78F45905AC` | `32078F78F45905AC` ✅ |
| `_window.py` | `AA098F72A30F2AB8` | `AA098F72A30F2AB8` ✅ |

> 说明：`run_issue238_period_sweep_20261119.py` 与 `_window.py` 在初签后变更，经核读源码确认
> 均为**设备占用 / 锁协议层**改动（`lock_mode="segment"`，修复 period32 父子互锁），不含任何指标/评分逻辑。
> 上游合并 `3d0fe7f2`（53 提交）+ `5545d579` 重构源码模块布局后，上表六个脚本 blob 哈希全部变化；
> 评分口径实现 `protocol.py::target_intensity_rmse` 虽改了文件哈希，但在当前工作树重跑 §6 评分
> `RECIPE_SELFCHECK_MAX_ABS_DIFF_vs_manifest=0.000e+00`、score JSON SHA256 逐位不变，语义等价。
> 已按预注册 §0 于 `2026-11-19T15:46:32Z` 登记新哈希并重签（见预注册 §0.3，评分完成后重签），
> `§6` 已落盘评分继续有效；**若脚本哈希再次变化，须再次重签**。

---

## 1. 每段一条可直接粘贴的 PowerShell 命令骨架

命令前缀统一约定（在 PowerShell 会话里先执行一次，后续骨架里用 `$PY`）：

```powershell
Set-Location <REPO_ROOT>                 # 例如 D:\Projects\TIFO\AO-shaping
$PY = <PYTHON_EXE>                       # 例如 <REPO_ROOT>\.venv\Scripts\python.exe
```

### S0 — 六步光路探测 + 曝光标定（必跑，前置门禁）

> `probe_optical_path_validity.py` **没有 `--execute` 开关**：脚本一旦启动就会开设备，
> 它靠 `--wait-until-idle` 做占用等待，且整段探测期间握锁，`finally` 释放后写
> `optical_path_record.json`。**因此本段不可 dry-run**；只可在已获用户授权、设备空闲时运行。
> 退出码 0 = `ok`，非 0 = `low_snr` / `weak_path` / `no_response`，任一非 0 即止步。
> 现场按需覆盖 `--exposures`（历史默认阶梯已不反映当前光通量，必须现场重新定档）与
> `--bit-depth 12`、`--slm-wavelength`；`--session` 用本窗口唯一值。

```powershell
& $PY scripts/slm/probe_optical_path_validity.py `
  --exposures <EXPOSURE_LADDER_MS_CSV> `
  --defocus-amps <DEFOCUS_AMPS_RAD_CSV> `
  --slm-number <SLM_NUMBER> --slm-wavelength <SLM_WAVELENGTH_NM> `
  --cam-type <CAM_TYPE> --cam-id <CAM_ID> --bit-depth 12 `
  --wait-until-idle <WAIT_IDLE_S> `
  --out-root <OPTICAL_PROBE_OUT_ROOT> --session <WINDOW_RUN_ID> `
  --frames <PROBE_FRAMES> --settle-s <PROBE_SETTLE_S>
# 读回：<OPTICAL_PROBE_OUT_ROOT>/<WINDOW_RUN_ID>/<UTC戳>/optical_path_probe.json
#      与同目录 optical_path_record.json；确认 ok + 占用已释放。
```

### S1–S3 — Transfer 段（K=1 / K=4 / light / none，一个窗口串起来）

窗口脚本持有 bench 锁到窗口结束，子段在 `window` 锁模式下靠父窗口持锁（不再自行等待）；
段间冷却默认 200 s，任一段非零退出即停止整个窗口、不自动续跑。**先 dry-run**：

```powershell
# dry-run（不建目录、不碰设备）：核对 argv 与落盘目录
& $PY scripts/experiments/run_transfer_window_20261119.py `
  --window <NEW_TRANSFER_WINDOW_DIR> `
  --dataset <DATASET_CAPTURE_DIR> --checkpoint <CHECKPOINT_JSON> `
  --segments k1 k4 light none `
  --shapes square --side <SIDE_PX_LE_424> `
  --transfer-arg <EXTRA_TRANSFER_ARG_0> `
  --cooldown-s <COOLDOWN_S_GE_200> --wait-until-idle <WAIT_IDLE_S> `
  --segment-timeout-s <SEGMENT_TIMEOUT_S> `
  --slm-number <SLM_NUMBER> --cam-id <CAM_ID>
```

授权到位、S0 通过、dry-run 核对无误后，**只在上一条命令末尾追加 `--execute` 再运行**：

```powershell
& $PY scripts/experiments/run_transfer_window_20261119.py `
  --window <NEW_TRANSFER_WINDOW_DIR> `
  --dataset <DATASET_CAPTURE_DIR> --checkpoint <CHECKPOINT_JSON> `
  --segments k1 k4 light none `
  --shapes square --side <SIDE_PX_LE_424> `
  --transfer-arg <EXTRA_TRANSFER_ARG_0> `
  --cooldown-s <COOLDOWN_S_GE_200> --wait-until-idle <WAIT_IDLE_S> `
  --segment-timeout-s <SEGMENT_TIMEOUT_S> `
  --slm-number <SLM_NUMBER> --cam-id <CAM_ID> `
  --execute
```

参数语义/默认值（来自脚本）：

- `--window` 必填，不允许复用已存在的非空目录；`--segments` 默认 `k1 k4`，choices
  `k1/k4/light/none`；`light`/`none` 必须显式给 `--dataset`（`capture/`，含 `manifest.json`）
  与 `--checkpoint`（`frozen_parameters.json`），且目录须存在，缺任一在 dry-run 阶段就返回 2。
- `--shapes` 默认 `square`；`--side` 现默认**不给（沿用 transfer 默认 80）**，dry-run 会用
  `2*(window-span-8)` 复算上限并**提前挡掉**超窗值（默认 220 → 上限 424；历史的 `1200`
  会被直接拒绝，不必再等到子段 `validate()` 才失败）。要更大目标须同时加 `--window-span`
  （如 640）并升 `--capture-grid`（如 384）。
  `--iterations/--count/--frames/--seed` 与 `--transfer-arg`（可重复）原样透传给 `transfer run`。
- 共用信封参数：`--cooldown-s` 默认 200、`--wait-until-idle` 默认 600（0=不等不锁）、
  `--segment-timeout-s` 默认 3600（0=不限）、`--slm-number` 默认 1、`--cam-id` 默认 0。
- 子段真实 argv（由脚本生成，无需手敲）：`python -m ao_shaping.tools.slm.transfer run
  --out <window>/<seg> --execute [--dataset ... --checkpoint ...] [--candidates 1|4 |
  --ccd-mode light|none] --slm-number ... --cam-id ... [--shapes ... --side ...]`。
- 退出语义：dry-run 恒 0；参数/目录不可复用 2；等锁失败 3；任一段失败/非零退出/中断 1；
  全段 ok 返回 0。证据：`<window>/window_plan.json`、`window.json`、`released.json`、
  `segments/<name>.{log,json}`。

### #238 — 二值光栅 32/48 px 周期扫描（run3，另一个窗口）

**锁协议（2026-11-19 修复后）**：父窗口 `lock_mode="segment"`，自己不持锁，只串行 +
冷却 + 只读回读锁状态；子段脚本自己 `acquire/release`，真实 `--wait-until-idle` 原样转发。
`--wait-until-idle 0` 在此处**不是**"不等不锁"，而是"被占即立即失败"，不能用 0 去规避互锁。
**脚本哈希已按预注册 §0.3 于 15:46 重签（当前 `48BBF7198FC3C691`），执行前须再次复核，若变化须再次重签**。先 dry-run：

```powershell
& $PY scripts/experiments/run_issue238_period_sweep_20261119.py `
  --run-dir <NEW_ISSUE238_RUN_DIR> `
  --periods-px <PERIOD_PX_LIST> `
  --frames-per-level <FRAMES_PER_LEVEL> `
  --exposure-bracket <EXPOSURE_BRACKET_MS_LIST> `
  --cooldown-s <COOLDOWN_S_GE_200> --wait-until-idle <WAIT_IDLE_S_GT_0> `
  --segment-timeout-s <SEGMENT_TIMEOUT_S> `
  --slm-number <SLM_NUMBER> --cam-id <CAM_ID>
```

授权到位、S0 通过、dry-run 核对无误后，只追加 `--execute`：

```powershell
& $PY scripts/experiments/run_issue238_period_sweep_20261119.py `
  --run-dir <NEW_ISSUE238_RUN_DIR> `
  --periods-px <PERIOD_PX_LIST> `
  --frames-per-level <FRAMES_PER_LEVEL> `
  --exposure-bracket <EXPOSURE_BRACKET_MS_LIST> `
  --cooldown-s <COOLDOWN_S_GE_200> --wait-until-idle <WAIT_IDLE_S_GT_0> `
  --segment-timeout-s <SEGMENT_TIMEOUT_S> `
  --slm-number <SLM_NUMBER> --cam-id <CAM_ID> `
  --execute
```

参数语义/默认值：`--run-dir` 必填（新目录）；`--periods-px` 默认 `32 48`，合法范围 8..256
且不可重复；`--frames-per-level` 默认 20；`--dense-lo/--dense-hi/--dense-step/
--checker-frames/--settle-s/--exposure-bracket` 仅显式给出才透传（子脚本要求 bracket ≥4 档
且全为正）。单周期约 10 min，两周期加冷却约 25 min；期间不要开第二个设备进程。
退出语义同共用信封（0/1/2/3）；证据落 `<run-dir>/window_plan.json`、`window.json`、
`released.json`、`segments/period<P>.{log,json}`；数据落 `<run-dir>/period<P>/`。

### 目标形状 measured-vs-target 配对采集（带锁 validate --ccd-check + 带锁 replay）

两个 wrapper 都把 `--execute` 当作自己的设备闸门；`--wait-until-idle`（默认 120 s，0=不等不锁）
由 wrapper 消费，其余参数原样转发给子脚本。`--execute` **不会**转发给 replay（replay 模块没有
这个开关）。`validate` 的 `--execute` 则原样转发给 `validate_transfer_shapes.py`。
`replay` 两条硬前置：`--out` 必填、`--exposures` 必须显式给出（默认 50/100 ms 是历史扫档，
禁止沿用；现场曝光只能来自当次读回范围与安全上限）。

```powershell
# 先 dry-run / 帮助核对（不碰设备）
& $PY scripts/slm/validate_transfer_shapes_hardware.py --help
& $PY scripts/slm/replay_transfer_shapes_hardware.py --help
```

```powershell
# validate --ccd-check（新目录，现场冻结几何/曝光）
& $PY scripts/slm/validate_transfer_shapes_hardware.py `
  --wait-until-idle <WAIT_IDLE_S_GT_0> `
  --checkpoint <CHECKPOINT_JSON> --out <NEW_VALIDATE_OUT> `
  --zero-xy <ZERO_X> <ZERO_Y> --target-xy <TARGET_X> <TARGET_Y> `
  --shapes square --side <SIDE_PX_LE_424> `
  --period <PERIOD_PX> --window-span <WINDOW_SPAN_PX> `
  --window-step <WINDOW_STEP_PX> --zero-radius <ZERO_RADIUS_PX> `
  --core-margin <CORE_MARGIN_PX> --frames <FRAMES> `
  --ccd-check --bit-depth 12 `
  --slm-number <SLM_NUMBER> --cam-id <CAM_ID> `
  --execute
```

```powershell
# replay（新目录，显式曝光，必须来自现场读回）
& $PY scripts/slm/replay_transfer_shapes_hardware.py `
  --wait-until-idle <WAIT_IDLE_S_GT_0> `
  --initial-run <INITIAL_RUN_DIR> --runs <RUN_DIR_0> <RUN_DIR_1> `
  --out <NEW_REPLAY_OUT> `
  --exposures <FIELD_EXPOSURES_MS_LIST> `
  --frames <REPLAY_FRAMES> `
  --camera <CAM_TYPE> --cam-id <CAM_ID> --slm-number <SLM_NUMBER> `
  --execute
```

`validate` 子脚本其它默认：`--window-span` 220（半跨）、`--window-step` 2、`--side` 80、
`--period` 64、`--zero-radius` 0、`--frames` 3、`--bit-depth` 8（本窗口须现场设 12）、
`--abba` 默认 false、`--no-abba` 可跳过 CCD 判读。replay 子脚本默认 `--frames 3`、
`--camera daheng`、`--slm-number 1`、`--cam-id 0`。wrapper 退出语义：无 `--execute`
时按 model-only 转发（validate）或拒绝并打印帮助（replay）；等锁失败/拿锁失败返回 1；
正常返回子进程退出码。锁证据写 `<out>/device_lock.json`（若子脚本在 mkdir 前失败则写
同级 `<out>.device_lock.json`）。

---

## 2. 现场读回检查表（逐项对应预注册 §3）

> "来源/步骤"列指取得该值的命令或步骤。**全部值在设备下线前读回并留时间戳**，
> 事后填入预注册 §3；本表只登记来源，不写任何历史数值。

| 预注册 §3 字段 | 取值口径（冻结点） | 来源 / 哪条命令或步骤 |
|---|---|---|
| 本窗口唯一 run 根目录 | 新建、执行前确认不存在 | §1 各命令的 `--window` / `--run-dir` / `--out`；`_window.py` freshness 校验拒绝复用 |
| 同曝光暗场 floor（每档曝光） | 当次同曝光暗场实测，禁止沿用 `floor=0` | S0 探针 + 现场暗场采集；写入评分 `hw_window_20261119_score.json` |
| zero-order center (CCD px) / exclusion radius | 当次标定 | S0 Step 5；validate 的 `--zero-xy` / `--zero-radius` |
| 目标形状 / 边长（≤424 px） | 当次几何配对，禁止 1200 | validate/transfer 的 `--shapes` / `--side`；预注册 §4 的 424 上限 |
| target center / window span / step / core mask | 当次冻结 | validate 的 `--target-xy` / `--window-span` / `--window-step` / `--core-margin` |
| CCD 型号 / ID / 生效位深 / full scale / 原生帧尺寸 | 当次读回；确认 Mono12 | S0 Step 1/4/6 + 探针 `optical_path_record.json`；validate `--bit-depth 12` |
| 曝光固定值 / 硬件范围 / 操作上限 / 读回值 | 当次读回；峰值判据，不得沿用历史档 | S0 探针 `--exposures` 与 `optical_path_probe.json`；replay `--exposures` |
| frames / settle / ABBA 是否启用及理由 | 当次决定 | transfer 的 `--frames` / replay 的 `--frames` / 探针 `--settle-s`；validate `--abba`/`--no-abba`/`--ccd-check` |
| dataset / checkpoint 路径与 SHA256 | 当次冻结 | §3 路径检查；`Get-FileHash -Algorithm SHA256 <path>` |
| 设备授权 / run 时间窗 / 操作者 | 用户/操作者 | 授权消息 + 执行前回复中的设备占用报告 |

---

## 3. 已知坑（务必逐条确认）

| 坑 | 现象 / 根因 | 规避 |
|---|---|---|
| `--side 1200` 超窗 | 11-19 `transfer_window_20261119/segments/k4.log`：`ValueError: 目标边长 1200 px 超出采样窗能容纳的 424 px（半跨 220 px，两侧各留 8 px）`；`config.validate()` 阶段失败，未开设备 | `--side` 设 ≤424（预注册 §4 已写死）；S1 前修正 |
| 背对背 SLM 假死 | 连续 open/close 触发 `ReadSU=-1000` / `Open=-10002`，需手动重启，当日不再续 | 窗口脚本 `--cooldown-s ≥200`（默认 200）；单进程占用，不并行第二设备进程 |
| `floor=0` 不合规 | 历史 replay 把背景 floor 取 0，预注册 §2.2 明确禁止 | 当次同曝光暗场实测 floor，来源/时间戳填 §3 |
| D2/D1 门控段本窗口不跑 | S7（#247 零级闭环，D2 未放行）、S5/S6（P3/P4 探针，D1 未放行）在预注册 §1 明令不执行 | 命令清单里没有这些段，不要临时加 |
| #238 父子互锁（已修复，已重签） | 旧 run3：父窗口持锁 + 子段 `--wait-until-idle 0`（=被占立即失败）→ 0.4 s 拒绝启动；新版本改 `lock_mode="segment"`，父不持锁、子自取自放 | 确认 `lock_mode="segment"` 生效且 `--wait-until-idle >0`；哈希已按 §0.3 重签，执行前再复核 |
| 窗口/子段目录不可复用 | `run_session` 与窗口都用 `exist_ok=False`，空目录也会挡住干净重跑 | 每次换全新目录；证据只写 `<window>/segments/`，不预建子段 `--out` |
| 复制来源不明 dataset | 计划要求"光路未动的最新 full run"，旧轴失配会话未必是最新 | 见 §3 路径检查；冻结前核对 `manifest.json` 与 provenance |
| S0 无 `--execute` 闸门 | 探针脚本启动即开设备，只能靠授权 + `--wait-until-idle` 控制 | 只在授权后运行，退出非 0 即止步；读回 `optical_path_record.json` |

---

## 4. 最近失败/部分执行的证据复盘（离线核对）

两次尝试**都没有采到任何有效数据**，设备均已正常释放。

### 4.1 `data/experiments/transfer_window_20261119/`

- `window.json`：`created_at 2026-11-19T07:05:15Z`；`segments=[k4]`（`extra.segments` 原为
  `k4 light none`）；`stopped_after="k4"`、`all_ok=false`。
- `segments/k4.log`：子段 `exit_code=1`、`duration_s=2.0`、`out_exists=false`；报错
  `ValueError: 目标边长 1200 px 超出采样窗能容纳的 424 px（半跨 220 px，两侧各留 8 px）`。
- 结论：**k4 未成**（`config.validate()` 阶段被参数门挡下，未打开 SLM/CCD，未生成数据）；
  `light`/`none` 因"任一段非零即停止"**未执行**。
- `released.json`：`released_at 2026-11-19T07:05:44Z`、`lock_released=true`、`error=null`、
  `holder=null` —— 锁正常释放。本目录不可复用。

### 4.2 `data/experiments/issue238_binary_phase_20261119/run3_period_sweep/`

- `window.json`：`created_at 2026-11-19T07:28:35Z`；`segments=[period32]`（`extra.periods_px`
  原为 `[32,48]`）；`stopped_after="period32"`、`all_ok=false`。
- `segments/period32.log`：`--wait-until-idle 0`；子段打印
  `bench device lock occupied by LockHolder(pid=1005084, ...)` 后
  `ERROR: bench device lock stayed busy; refusing to start`；`exit_code=2`、
  `duration_s=0.4`、`out_exists=false`。
- 结论：**period32 未成**（子段与父窗口抢同一把 bench 锁，0.4 s 内拒绝启动；**未开
  SLM/CCD、未采一帧**）；`period48` 因停止策略**未执行**。这**不是** USB `-10032` 的复现。
- `released.json`：`released_at 2026-11-19T07:29:04Z`、`lock_released=true`、`error=null`、
  `holder=null` —— 锁正常释放。本目录不可复用。
- 修复已落盘（`_window.py` 新增 `lock_mode="segment"`，`run_issue238...py` 显式传入），
  脚本哈希已按预注册 §0.3 于 15:46 登记重签，可用于本窗口评分（执行前再复核）。
- **修复后已在全新目录 run4 成功重跑**：`data/experiments/issue238_binary_phase_20261119/run4_period_sweep/`
  （run3 目录不可复用）。两段各 969 帧、`status=ok`、设备与锁正常释放。详见
  `hw_plan_consolidated_20261118.md` §7.3 与 `issue238_binary_gray_capture_plan_20261119.md` §6.4。

---

## 5. 计划文档与脚本的一致性核对

| 来源 | 文本要点 | 与脚本实际参数的一致性 |
|---|---|---|
| `hw_plan_consolidated_20261118.md` §4 S2 | `--shapes square --side 1200` | ✅ 已修正：§4 表与执行入口改为不写 `--side`（沿用 80）并注明 ≤424；脚本默认由 1200 改为“不给即 80”，dry-run 提前拒绝超窗值 |
| `hw_plan_consolidated_20261118.md` §168 后执行入口 | `--segments k1 k4 light none --shapes square --side 1200` | ✅ 已修正：去掉 `--side 1200`，`--window`/`--dataset`/`--checkpoint`、dry-run + `--execute` 保持不变 |
| `hw_plan_consolidated_20261118.md` §4 S1 | `--candidates 4 --execute`，"K=1 可复用" | ✅ 脚本 `--segments` 可只给 `k4`；本窗口按预注册 §1 同时跑 k1/k4 |
| `hw_plan_consolidated_20261118.md` §4 前置 | `--wait-until-idle 600`、冷却 ≥200 s | ✅ 共用信封默认值正是 600 / 200 |
| `issue238_binary_gray_capture_plan_20261119.md` §6.2 | `--frames-per-level 20`、`--periods-px 32 48`、`--cooldown-s 200`、`--wait-until-idle 600`；曝光走探针 `choose_exposure` | ✅ 脚本默认值一致；`--exposure-bracket` 仅在显式给出时透传 |
| `issue238...` §6.2/§6.3 | 修复要求：父窗口不持锁，子段自取自放，父只回读锁状态 | ✅ 当前磁盘版本已实现（`lock_mode="segment"`）；哈希已按 §0.3 重签，执行前再复核 |
| `issue238...` §6.3 | 旧 run3 失败于锁协议，重试须全新目录 | ✅ 与 §4.2 证据一致 |

---

## 6. dataset / checkpoint 候选路径检查（离线）

| 路径 | 存在？ | 备注 |
|---|---|---|
| `data/debug/slm_transfer/axis_mismatch_validation_20261118/capture` | ✅ 存在 | 含 `manifest.json`、`capture_records.pkl`、`.capture_done`、`sample_*.npz`、`_optical_probe_flat.npz`；11-19 失败段用的就是它 |
| `data/debug/slm_transfer/axis_mismatch_validation_20261118/fit/frozen_parameters.json` | ✅ 存在 | 同目录另有 `initial/frozen_parameters.json` |
| git status 中更合适的 `frozen_parameters.json` | ❌ 没有 | `git status --short --untracked-files=all` 未见任何新增/修改的 `frozen_parameters.json`；`data/*` 被 `.gitignore:26` 忽略，磁盘上虽有多个候选（如 `offline_refit_20261118_freshep/...`、`hybrid_intensity_gate_20261118/...`），但均无证据表明比失败段所用更新/更合适 |

> 结论：路径可用，但"光路未动的最新 full run"须由操作者现场确认（核对 `manifest.json`
> 与 provenance），不能仅凭存在就冻结。若改用其它 checkpoint，须在预注册 §3 登记路径 +
> SHA256，并重签评分口径。

---

## 7. 自检声明

- 本文件**没有真实设备执行痕迹**：只引用了 11-19 历史失败日志（明确标注为失败、未采数据），
  本文自身未运行任何 `--execute`、未连接 SLM/CCD/激光。
- 没有写死历史曝光或历史几何：所有几何/曝光/时长均以 `<...>` 占位；文中出现的 1200、
  424、200 s 等仅为**禁令/默认值说明**，不是执行取值。
- 所有真实设备命令都带 `--execute` 且与冷却/锁纪律一致；S0 探针无 `--execute` 闸门已单独
  说明；dry-run 命令不带 `--execute` 且明确标注不碰设备。
