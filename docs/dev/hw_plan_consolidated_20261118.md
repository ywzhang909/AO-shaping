# 2f 整形实验计划（合并版 / 2026-11-18）

> 本文件是 2f 整形**所有待办实验任务与离线判读的唯一入口**（single source of truth）。
> 它把此前分散的三份计划合并为一条「离线优先 → 一次窗口集中实验」的执行线，目标是在
> **保证结论质量的前提下压缩物理设备（SLM #1 + Daheng CCD #0）占用时间**。
>
> 被合并的原子计划（保留作细节依据，状态以本文件为准）：
> - `docs/dev/hw_direction_test_plan_20261117.md`（P1–P5 方向门槛）
> - `docs/dev/next_hw_plan_transfer_candidates_issue247_20261117.md`（K 候选 + #247）
> - `docs/dev/hw_ccd_mode_test_plan_20261118.md`（ccd_mode light/none）
> - `docs/dev/axis_mismatch_independent_capture_plan_20261118.md`（轴失配独立采集与三个变体回验；
>   2026-11-18 新增，预注册已就绪，是 D2 解混淆的必跑段）
>
> 另有下列相关文档，已确认**不构成本次窗口的独立实验任务**（在案，但非本窗口）：
> - `docs/相差光路多方案整形对比实验方案_v1.0.md`：远期六方案对比（20+4 run，约 2 工作日），另立窗口。
> - `docs/自适应 SPGD 光束整形闭环：完整方案总结.md`：SPGD 闭环设计方案稿，非执行计划、无设备段。
> - `report/shape_trials/hw_depth_matched/hw_depth_matched_20261105.md`：`forward_model_transfer_plan` 的
>   P3–P6 闭环阶段**已明确推迟**，门槛与本文决策门 D1（模型方向）同源，模型修好后再议。
>
> 设备纪律见 [hardware_discipline.md](hardware_discipline.md)；六步光路探测见
> [ao-optical-path-calibration SKILL](../../libs/skills/ao-optical-path-calibration/SKILL.md)。
> 任何实验动作前后必须在回复中明确报告：占用设备、做了什么、何时释放、是否正常关闭。

---

## 1. 总原则：集中实验一次，数据分包离线处理

压缩设备时间靠一条主线加三个杠杆。主线是**把确需设备的段合并成一次窗口（集中实验），
窗口一结束立即关闭并释放设备；采集到的数据全部在设备下线后离线处理（而后处理数据）**，
设备只被"采集"占用，不被"分析"占用。三个杠杆按优先级：

1. **窗口前：把已落盘数据先判读完**（§3）。P1/P2/tilt/issue-247 的数据早已采集完成并干净释放，
   但 amp03/amp10 的方向判读、P1 四口径对账仍未做。这些**不占设备**，
   而且结论会决定后续哪些设备段根本不必跑——先做能直接砍掉无谓实验，窗口因此更短。
2. **窗口前：决策门先行**（§3 D1/D2/D2′）：离线结论若为「远场形态模型错误主导」则 P3/P4
   直接推掉；deswivel 归档回验已判定「直接套标定值做 #247 闭环」不放行（D2），但物理 xy 轴失配
   仍未公平否定（D2′），是否为本窗口增加独立采集段由 D2′ 决定，不占无谓时间。
3. **窗口内：一次 SLM/CCD open/close 走完所有放行段**（§4）：K 候选验证、ccd_mode light/none、
   （若 D2′ 放行）轴失配独立采集 S7b 合并成一条链，段间冷却 ≥200 s，共用同一份 dataset/checkpoint；
   每段独立 `--out`，采完即释放，不在设备占用期做任何分析。
4. **窗口后：数据统一离线判读**（§4「每窗口后处理」）：S1–S6、S7b 的判读、报告、issue 回写全部在
   设备下线后完成，不重占设备。

一句话：**采集尽可能挤进一次窗口，分析全部搬到窗口外**——这正是压缩物理设备占用时间的关键。

---

## 2. 全量任务状态表（完成标记基于落盘证据）

| # | 任务 | 状态 | 证据 / 产物 | 占设备？ |
|---|------|------|-------------|---------|
| P1 | 零级定心差分刀口（#235） | ✅ 采集完成；⚠️ 四口径离线对账未落盘 | `data/debug/issue235_center_20261117/`（`flat.npy`/`summary.json`）；刀口中心 x≈889, y≈590 | 否（数据已在盘） |
| P2a | 方向门槛 14 帧基线（0.03 rad） | ✅ 采集 + 分析完成 | `report/shape_trials/model_direction_hw_20261117/`；锚点相关 −0.16、N4 预测 4.4e-5 vs 实测 3.68 counts | 否 |
| P2b | 方向门槛 amp03 / amp10 | ⚠️ **采集完成、离线判读未落盘** | `data/debug/square1000_model_diag_20261117b_amp03/` 与 `..._amp10/`（各 14 帧，0.1 ms），无分析产物 | 否（判读即可） |
| — | 真线性 ±tilt 补采（#247） | ✅ 采集完成并释放 | `data/debug/issue247_tilt_20261117/`（13 组 Mono12，0 饱和，22:06:10 释放） | 否 |
| — | #247 零级/目标级分离判读 | ✅ 完成（报告已写） | `report/shape_trials/issue247_zero_order_20261117/`；零级稳定 ≤2.3 px、目标级分离 109–119 px、+88.544° 轴旋转 | 否 |
| — | #247 旋转映射离线回验（deswivel 三变体） | ✅ 归档数据回验完成（结论：无收益） | `data/debug/deswivel_forward_inverse_20261118/axis_three_arms/report.json`；held-out r² baseline 0.55534 > corrected 0.54347 > axis 0.54242，逆向 CV baseline 1.20561 最低；baseline ramp_matrix 就在 eval 帧上拟合，故只否定“直接替换矩阵”，不否定物理失配 | 否 |
| — | 轴失配**独立**采集 + 三变体回验 | ✅ 完成（结论：无收益，见 §3 D2′） | 新会话 `data/debug/slm_transfer/axis_mismatch_validation_20261118/`（`capture`/`initial`/`fit` 齐全，split 与预注册逐行一致）；回验 `data/debug/deswivel_forward_inverse_20261118/independent_20261118/report.json` + `paired_bootstrap_heldout.json`；held-out corr baseline 0.84352 vs axis 0.84115，`axis − baseline` r² −0.00348 CI95 [−0.00824, +0.00160] 含 0 | 是（已释放，锁 `holder=None`） |
| K | `--candidates K` 代码 + K=1 逐位一致 | ✅ 完成 | `differentiable_candidates.py`、`n_candidates` 主战场、golden「ALL IDENTICAL」；ruff 绿 | 否 |
| K | GPU 微基准（grid×形状×K） | ✅ 完成 | `report/shape_trials/transfer_candidates_timing_20261117/`；单步 7–51 ms、K=8 为 2.45–4.53× | 否 |
| K | K>1（K=4）真闭环验证 | ❌ **待实验** | 无 `transfer_k4` 目录；优化器占端到端 <0.5%，只验「在线实现」 | **是** |
| C | ccd_mode light 验证 | ❌ **待实验** | 实现 + 13 测试已就绪（`test_transfer_ccd_mode.py`），未实验 | **是** |
| C | ccd_mode none 验证 | ❌ **待实验** | 同上；无实测帧，需 N2 对照段 | **是**（+可选 N2） |
| P3 | #234 结构化探针（逐模 Jacobian） | ⏸️ **条件待定**（见 §3 决策门） | 计划见 `hw_direction_test_plan_20261117.md` §3；R=1→176 帧 | 是（可能推迟） |
| P4 | #233 非线性取证（编码台阶） | ⏸️ **条件待定**（随 P3） | 计划见 §3；可与 P3 同 session 插入 | 是（可能推迟） |
| — | 相差光路六方案对比 v1.0（远期总方案） | 📋 规划中，非本次窗口 | `docs/相差光路多方案整形对比实验方案_v1.0.md`；20+4 run、约 2 工作日 | 是（另立窗口） |

图例：✅ 完成 ｜ ⚠️ 数据齐、判读缺 ｜ 🔶 脚本就绪、结果缺 ｜ ❌ 待实验 ｜ ⏸️ 条件门控 ｜ 📋 远期规划

---

## 3. 第一队列：离线判读（不动设备，先做完再决定实验范围）

> 这一队全部读磁盘，不占 SLM/CCD/激光。**必须在下次实验前完成**，因为结论会删减 §4 的设备段。

1. **P2b 方向判读**（最高优先，直接决定 P3/P4 是否需要）：
   ```powershell
   .venv/Scripts/python.exe scripts/slm/analyze_square1000_model_diagnostic_20261030.py --session data/debug/square1000_model_diag_20261117b_amp10
   .venv/Scripts/python.exe scripts/slm/direction_peak_centred_20261117.py `
     --sessions data/debug/square1000_model_diag_20261117b_amp03 data/debug/square1000_model_diag_20261117b_amp10 `
     --out data/debug/direction_peak_centred_20261117b --center-method ellipse
   ```
   判据：0.10 rad 档 cosine 抬到 ≥0.50 ⇒ 门禁失败只是探针 SNR 不足，模型方向可用；仍 ≈0 ⇒
   远场形态（PSF/defocus/照明包络）是主因，进入决策门。
2. **P1 四口径对账**：对 `flat.npy` 跑 argmax/centroid/box_centroid/ellipse 四法，两两差 ≤3 px
   则默认椭圆中心可用；CCD 侧与 panel 侧刀口中心分开比（不可混减）。
3. **#247 deswivel 回验（已完成）**：`compare_deswivel_forward_inverse_20261118.py` 已在归档
   会话 `20261117_202916` 上跑完三个变体（baseline / corrected 消旋 / axis 不消旋只修轴失配）。
   结果：held-out r² baseline 0.55534 > corrected 0.54347 > axis 0.54242；held-out correlation
   0.79145 > 0.78795 > 0.78750；逆向最终 CV baseline 1.20561 最低（corrected 1.21351、
   axis 1.21645）。交叉评估显示所有相位经 baseline 前向评估 CV 最低、经 axis 最高，差异来自
   物理预测模型而非相位质量。**关键限定**：baseline 的 `ramp_matrix` 就是在这 24 帧上拟合的，
   对 eval 帧有结构性优势；因此该结果只能否定“把标定角/比例直接替换基线矩阵”的做法，
   **不能否定物理上的 xy 轴失配假设**。要判 #247 闭环，必须在从未参与拟合的新会话上做独立验证
   （见 §3 决策门 D2′ 与 §4 段 S7b）。
4. **报告归档**：判读结论落 `report/shape_trials/`，按 [report/AGENTS.md](../../report/AGENTS.md)
   登记 provenance（`sync_report_provenance.py --check` + `test_report_provenance.py` + `ruff check .`）。

**决策门 D1（P2b 结论）**：
- cosine ≥0.50 → 模型方向可用，P3/P4 保留进本次窗口；
- cosine ≈0 且远场形态不符 → **P3/P4 推迟到物理预测模型修复后**，本次窗口不跑，节省一大段设备时间。

**决策门 D2（deswivel 结论）**：
- 归档数据三变体回验已完成，两个矫正变体的 held-out 均不及 baseline（见 §3.3），**S7（直接套
  标定值做零级校正闭环）不放行、本次窗口不跑**——按原判据“矫正未改善 held-out 不得做零级校正”，
  会越校越偏；
- 但该结论受“baseline 在 eval 帧上拟合”的混淆，物理 xy 轴失配尚未被公平否定 → 新增 **D2′**。

**决策门 D2′（轴失配独立验证，2026-11-18 已执行，判定：不消旋修正无可检出收益）**：
- 预注册已完成（`axis_mismatch_independent_capture_plan_20261118.md`、`probe_plan.json` 指纹
  `6334aeba36f06976`），train/validation 划分与冻结假设已写死；
- **已按 §4 段 S7b 在 2026-11-18 窗口执行**：全新 seed=20261118 的 24 probe 会话（34 行
  manifest，0 饱和，split 与预注册逐行一致）→ bootstrap → fit（early-stop 171）→ 三变体回验；
- 采后判据：仅 `axis − baseline` 在 held-out r²/correlation 上改善且 bootstrap 95% CI 不含 0、
  同时逆向交叉评估不恶化，才判定“不消旋的轴失配修正有收益”，进而重议 #247 闭环；
- **判定结果（按预注册规则）**：held-out correlation `baseline 0.84352 / corrected 0.84115 /
  axis 0.84115`，r² `0.67354 / 0.66998 / 0.67006`；`axis − baseline` 的 correlation
  −0.00237 CI95 [−0.00520, +0.00057]、r² −0.00348 CI95 [−0.00824, +0.00160] 均覆盖 0，
  点估计反向 → **证据不足，无收益**。逆向 CV final `0.83303 / 0.82624 / 0.82605`，差异来自
  前向几何而非相位质量；
- 因此 **#247 零级校正闭环（S7）保持删除**；物理 xy 轴失配本身仍未被否定——要判定它需要
  独立的轴失配测量（旋转标定板 / 正交光栅双向扫描），不是重跑这批三变体。
- 详细执行记录、审计数字与判定见 `axis_mismatch_independent_capture_plan_20261118.md` §9–§10。

---

## 4. 第二队列：一次设备窗口合并执行（open/close 一次，段间冷却 ≥200 s）

> 只跑 §3 决策门放行后的段。共用同一份 `--dataset` / `--checkpoint`（光路未动的最新 full run）。
> 每段独立 `--out` 目录（capture 目录不可复用）；每段自带 finally：cam.close → slm.close →
> lock.release → `released.json`。

### 前置（窗口开始一次，任一失败即停）

1. `DeviceLock.default().holder()` == None；无竞争硬件进程（采集期**不要**跑含
   `Santec SLM` / `tilt` / `defocus` 关键词的长查询——曾误判为硬件进程致 census 延迟 84 s）。
2. 六步光路探测（曝光不超量程 / SLM 相位自检 / A=0 焦点确认），写 `optical_path_record.json`。
3. `require_drive_board_entry(slm)` + `serial_source()=="driveboard"`；`set_bit_depth(12)` 与
   `effective_full_scale` 双重确认 Mono12（8-bit 下 1 count≈7%，均匀度结论无效）。
4. 启动加 `--wait-until-idle 600`；段间冷却 ≥200 s，**禁止背对背 open/close**（驱动板假死 `ReadSU=-1000`）。

### 合并执行链

| 顺序 | 段 | 命令要点 | 时长估计 | 门控 |
|---|---|---|---|---|
| S0 | 光路探测 + 曝光标定 | 六步 SKILL，0.05 ms 基线 | ~5 min | 无（必跑） |
| S1 | Transfer K=4 闭环验证 | `--candidates 4 --execute`（K=1 基线可复用既有，不必重跑） | ~5 min | 无 |
| 冷却 | — | SLM 关闭 ≥200 s，确认锁空 | 200 s | — |
| S2 | ccd_mode light | `--ccd-mode light --dataset <latest>/capture --checkpoint <latest>/fit/frozen_parameters.json --shapes square --side 1200` | ~3 min | 无 |
| 冷却 | — | ≥200 s，锁空 | 200 s | — |
| S3 | ccd_mode none | `--ccd-mode none --dataset <同上> --checkpoint <同上>`（不开相机） | ~2 min | 无 |
| 冷却 | — | ≥200 s，锁空 | 200 s | — |
| S4 | N2 none 相位单次拍帧（可选） | 上屏 none 合成相位，用 transfer validate 拍 3 帧 | ~2 min | 有时间才做 |
| 冷却 | — | ≥200 s，锁空 | 200 s | — |
| S5 | P3 结构化探针（R=1） | 幅度 {0.02,0.05,0.10}×±×重复，Noll 9 holdout，176 帧 | ~10 min | D1 放行 |
| 冷却 | — | ≥200 s，锁空 | 200 s | — |
| S6 | P4 编码台阶扫描 | 与 P3 同 session 插入，6 档×5 帧 | ~3 min | D1 放行 |
| 冷却 | — | ≥200 s，锁空 | 200 s | — |
| S7 | #247 零级校正闭环（直接套标定值） | panel-x/y 斜坡，零级漂移 ≤2 px 且目标级按预测移动 | ~5 min | D2 **未放行**（见 §3） |
| S7b | 轴失配独立验证采集 | 全新 seed=20261118、24 probe 的 transfer capture（`--save-dataset`），采后离线三变体回验 | ~10 min + 冷却 | ✅ 已执行（2026-11-18 09:30–09:32 采集，设备已释放；判定见 §3 D2′） |

- D2 已判定不放行 → **S7 本次窗口删去**；D1 若未放行则 S5/S6 也删去。
- S7b 与 S7 互斥：S7b 采完立即释放设备，三变体回验（compare 脚本）在设备下线后离线完成。
- **S7b 已于 2026-11-18 执行完毕**：`capture`（34 行、0 饱和、split 与预注册一致）→ `bootstrap` →
  `fit`（early-stop 171）→ `compare`（guard 通过）→ 配对 bootstrap；设备锁 `holder=None`。
  结论“无收益”，**不再需要下一窗口重跑**。
- 若 D1 未放行且 S7b 也不跑，本次窗口只剩 S0→S4（约 20 min 设备 + 冷却）。
- 每段记录：SLM/CCD 打开/关闭时间、`slm_closed/camera_closed/lock_released`、exit code、峰值不饱和。

### 每窗口后处理（设备下线后，不占设备）

- 判读 S1（墙钟 / loss / 无 0 饱和），S2（A–B–B–A 实测 CV/IoU vs full 基线，退化 <20% 相对可接受），
  S3/S4（none 开环实测 CV 对照 model_cv 0.66 vs 实测 5.01 假收敛量级），S5/S6（按各自判据），
  S7b（设备下线后跑 `compare_deswivel_forward_inverse_20261118.py --fit-dir <S7b>/fit --calibration
  issue247 JSON`，只比较 held-out 行与逆向交叉评估；判据见 §3 D2′）。S7 本次窗口已删去，无后处理。
- 逐项回写 issue（#229/#233/#234/#235/#247）并附 commit id；结论落 report（含 provenance）。

---

## 5. 红线与不做

- 任何饱和 → 当段 abort，不重试；峰值触顶 0.98 满量程 → 降曝光重拍该档，不裁数据。
- 背对背 SLM open/close 禁止；冷却不满不进下一段；驱动板假死 → 停手、手动重启、当日不再续。
- `serial_source` 非 driveboard → 探针无效，不写结论；`--out` 目录已存在 → 换新目录。
- **不改任何 MIL 文件**（`model_in_loop_shaping.py` / `slm_model_in_loop.py` /
  `shaping_controls.py` / `test_model_in_loop_shaping.py`）——已确认全部未动。
- ccd_mode 不并入口径到 GUI（runner 暂不暴露，CLI 先行）。
- 远期「相差光路六方案对比 v1.0」（20+4 run、约 2 工作日）另立窗口，不并入本次。

## 6. 设备占用报告模板（每段粘贴）

```
段 Sx | 占用：SLM #1 / CCD #0 / 激光 | 开始 HH:MM:SS | 释放 HH:MM:SS
动作：<做了什么> | released.json：slm=<t/f> cam=<t/f> lock=<t/f> | exit=<n> | 异常=<无/...>
```
