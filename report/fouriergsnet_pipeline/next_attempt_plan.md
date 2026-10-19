# GS 家族整形下一步尝试方案（slm-gs-refine × FourierGSNet）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线+硬件
> **说明**: GS 家族整形下一步尝试方案 (slm-gs-refine × FourierGSNet, 2026-10-12)
<!-- provenance:end -->

**本轮结论**（方案制定所依据的既有实验证据）：(1) `slm-gs-refine` 已于 2026-10-12 首次实验运行成功——平场 composite 0.527 → plain GS 0.774 → 最佳 0.778，交错重放两次一致（+0.0043 可复现），但 30 轮 SPGD 的 +0.0674 增益落在 25.7k/33.6k 两簇分布中不可复现，均匀性 CV 仍为 0.65–0.73，是当前唯一瓶颈（PIB 已达 0.97–0.98）。(2) GS 变体（weighted / hio-weighted / adaptive-feedback）实验系统全部弱于 plain GS，算法默认保持 plain，不再投入。(3) 2026-10-01 `slm-gsnet spgd` 失败的根因已由 H-14 证明为**慢漂移而非步长 δ**（1000× 全扫描 8 档 `dec` 均在 0.43–0.56），历史 51 次运行中 32 次为随机游走；收敛判定只认 `dec`/`late`。(4) 曝光无固定值可写死，功率日内漂移显著（8 分钟 +21%），每次会话必须以只读平场会话内在线确定的曝光。(5) 实验前须清完阶段 A 七项离线修复（H-7 EE guard、H-8 小幅初始化、H-9 ABBA、H-10 曝光默认值、H-11 焦距常数统一、H-12 参数语义、H-13 可达目标重定义）及 PANEL_RES 转置 bug；路线为 A（离线修复）→ B（表征探针，ABBA 残余漂移 < 单帧信号一半为 go 判据）→ C（分段实验）。

> 日期: 2026-10-12（当日首次实验运行后修订）。
> 依据材料:
> - [`docs/slm/slm_gs_refine.md`](../../docs/slm/slm_gs_refine.md) — GS 预整形 + SPGD 细化（硬件移植版）
> - [`report/shape_trials/gs_refine_live_20261012.md`](../shape_trials/gs_refine_live_20261012.md) — **2026-10-12 gs-refine 首次实验运行成功**（含 GS 变体、半径扫描、30 轮 SPGD、交错重放）
> - [`report/fouriergsnet_pipeline/report.md`](report.md) — FourierGSNet 管线 5 项集成测试（离线通过）
> - [`report/fouriergsnet_pipeline/hardware_run_20261001.md`](hardware_run_20261001.md) — 2026-10-01 实验系统 `slm-gsnet spgd` 失败复盘（随机游走）
> - [`report/fouriergsnet_pipeline/TODO.md`](TODO.md) — 实验系统验证待办清单
> - [`report/fouriergsnet_sim/report.md`](../fouriergsnet_sim/report.md) — 湍流仿真矩阵（12 场景全过）
> - [`TODO.md`](../../TODO.md) §1.2–1.4 — H-x 硬件账本

## 1. 现状小结

| 轨道 | 离线状态 | 实验系统状态 |
|---|---|---|
| `slm-gs-refine`（GS 预矫正 + 全频随机相位 SPGD 细化） | 仿真配方胜出: composite 0.662 → GS 0.812 → 细化 0.849；对比择优 / settled capture / 逐帧去背景三道防线已落地 | ✅ **2026-10-12 首跑成功**: 平场 0.527 → plain GS 0.774 → 最佳 0.778，交错重放两次一致（best 比 GS +0.0043）。但 30 轮 SPGD 的 +0.0674 增益**不可复现**（32 条记录呈 25.7k/33.6k 两簇），CV 仍 0.65–0.73 |
| `slm-gsnet spgd`（freeform 方形整形） | sim 演示 quality 0.587 | 2026-10-01 跑过一次: **随机游走**（`dec=0.487`，末态比初态差 35.9%，EE 流失 6×，0 阶峰值 225→17） |
| FourierGSNet（GS 展开 + CNN 残差回归） | 离线训练验收 + 12 场景仿真矩阵全过（square 均匀度 0.01–0.05，EE≈0.92）+ 5 项集成测试通过 | 未实验（前置 H-3~H-6 未清） |

10-01 失败的根因（H-14 已证）：**瓶颈是慢漂移，不是 δ**——1000× 全扫描 8 档无一收敛（`dec` 全在 0.43–0.56）。叠加因素：方形路径无 EE guard（H-7）、无 ABBA 对消（H-9）、576 DOF 单帧差分 SNR 不足、满幅随机初始化（H-8）。历史 51 次运行中 32 次是随机游走——**判定收敛只看 `dec`/`late`，不看 `sustained`**。

10-12 首跑的增量结论（详见 live 报告）：
- **GS 变体全败**: `weighted` 0.717 / `hio-weighted` 0.762 / `adaptive-feedback β=0.6` 0.657，均低于 plain GS ≈0.775 ⇒ 算法默认保持 plain。
- **模型半径扫描**: 400/450/500 中 450 最优（GS 0.7826）；但这是模型参数对照，光斑半径仍无独立测量。
- **曝光随功率漂移**: 当日下午功率上升，安全曝光从 0.05 → 0.04 → 0.020 ms（驱动下限）。**曝光没有固定值可写死**，每次会话前必须只读平场检查现定。
- 控制卡 ID 报 `23020027`（早先日志 `22030108`），瞳面中心粗估 `(960,690)`；跨会话比较必须用交错重放（`scripts/slm/replay_gs_refine_hw.py`）。

## 2. 总体路线：A 修代码 → B 探针 → C 三段实验

```
A (离线修复) ──全绿──▶ B (表征探针 + smoke) ──ABBA 增益达标──▶ C1 静态对比择优
                                                        C2 gs-refine 短轮
                                                        C3 gsnet spgd 复测
                                                        C4 FourierGSNet run
```

每段设 go/no-go 判据，不达标不进入下一段。

## 3. 阶段 A：离线必修（复测前清账）

来源 `hardware_run_20261001.md` §5–6 与 TODO.md §1.3，**六项先改再实验**：

| ID | 修复内容 | 验收 |
|---|---|---|
| H-7 | 方形目标函数加 encircled-energy guard（照搬 `slm-pib` 同款），防「散光换 CV」 | 离线：故意加散光扰动，composite 不升反降 |
| H-8 | 初始化改小幅（与 docstring "small init" 一致），弃满幅 `uniform(-π,π)` | sim：初值 EE 高于满幅起点基线 |
| H-9 | 方形路径接入 ABBA 漂移对消（`slm-pib` 已有实现可照搬） | sim 端到端 + 与 H-19 实验验收合并做 |
| H-10 | `--exposure_time_ms` 默认 80.0 ms 改为实验系统实测量级（~1.0–1.5 ms），并修复「传值即关饱和保护」死代码 | 离线单测：饱和保护在显式曝光下仍生效 |
| H-11 | 焦距标定常数统一（`132940/P` vs `7600/P` 差 17.5×，以实测 `K` ⇒ f=0.1224 m 为准） | 文档与代码单一权威值 |
| H-12 | `lr=0` 时 `learning_schedule()` 静默覆盖 `--delta`；`_param_scale` 两基不一致；"per-pixel" 名不副实（`np.kron` 分块） | 参数语义与 help 一致 |
| H-13 | 可达目标重定义：本实验系统 80 px staircase 只能影响大尺度结构，`--target-side` 用推导值而非刻 50 px 平顶 | 目标在可达带宽内 |

另有一项**实验系统隐患**（fouriergsnet_pipeline/report.md 标注，仿真掩盖）：

- **PANEL_RES 转置 bug**：真实驱动 `PANEL_RES=(1920,1200)` 是 (宽,高)，管线 `place_on_panel` 当 (h,w) 用；仿真侧 `PANEL_H,PANEL_W=1920,1200` 恰好抵消。修 `place_on_panel` 并补一条形状回归测试，否则实验图案转置。

**阶段 A go 判据**：`utils/` + `wfless/` 测试全绿，`slm-gs-refine --cam_type sim` 与 `slm-gsnet spgd --cam_type sim` 端到端分数不低于现有基线（0.587 / 0.849 口径）。

可选顺带：H-32 `--w_loggrad` 验收放在 δ=0.05 档（勿用 0.0005），它不能修复不收敛，默认仍 0.0。（H-32 即原 TODO 中的 H-25，2026-10-19 因与 GitHub 硬件验收批次 H-25~H-31 撞号而重编，见 TODO §1.1 编号说明。）

## 4. 阶段 B：表征探针 + smoke（激光开，独立进程 + 看门狗硬超时）

**顺序不能反**（`docs/slm/pre_run_characterization.md`）：

```bash
python -m ao_shaping.tools.slm.slm_drift_probe --exposure-ms <现定>
python -m ao_shaping.tools.slm.slm_floor_probe --exposure-ms <现定>
python -m ao_shaping.tools.slm.slm_abba_probe --exposure-ms <现定>
```

- 曝光**必须显式传入且每次会话现定**：先只读平场检查，取峰值 ≤ ~230/255 的最大档。10-01 实验系统功率低时干净点在 ~1.5 ms；10-12 下午功率升高后安全曝光降到 0.05 → 0.04 → 0.020 ms（驱动下限）。`--exposure_time_ms 0` 会被大恒驱动钳到 ~0.02 ms。
- 补齐 `report/fouriergsnet_pipeline/TODO.md` §2 smoke 结果区（H-4，末尾 `SMOKE_OK`）。
- **go 判据**：ABBA 探针显示对消后残余漂移 < 单帧信号的一半。若不达标，先处理激光功率漂移（10-01 实测 8 分钟 +21%；10-12 一天内安全曝光连降两档同样证明功率不稳）再谈任何闭环。

## 5. 阶段 C：实验（每段独立进程 + 看门狗）

### C1. 静态 对比择优 验证 —— ✅ 已随 10-12 首跑完成

平场 0.527 < plain GS 0.774，对比择优 防线生效；派生焦距 / 2.2 µm 像元 / 半径 450 锚点已核对（半径 450 为模型扫描最优，仍缺独立光斑半径测量）。

### C2. `slm-gs-refine` 实验系统 —— ✅ 首跑成功，下一步是「增益可复现化」

10-12 已证：短轮 refine 稳定优于 GS（交错重放两次一致，+0.0043），但 30 轮 SPGD 的 +0.0674 落在两簇分布（25.7k/33.6k）里，**不可复现**。下一轮实验按优先级：

1. **对 `20261012_180222` 的最佳相位做独立交错重放**（`scripts/slm/replay_gs_refine_hw.py`，先只读平场会话内在线确定的曝光）——判定 0.7895 是真实增益还是会话内运气。
2. **排查两簇状态**：原始图总和 25.7k vs 33.6k 说明光路/采集存在双稳态（怀疑功率漂移或面板弛豫未 settled）。必要时先跑 B 阶段 drift 探针定量。
3. **主攻均匀性**：PIB 已 0.97–0.98，瓶颈全在 CV 0.65–0.73（热点与空洞）。候选手段按代价排序：`--phase-grid` 12→16→24 逐级升 DOF（每级用 `dec`/`late` 判收敛，不收敛即退回）；H-9 ABBA 进 refine 路径；H-7 EE guard 已在 refine 侧生效与否需确认。
4. GS 变体不再投入：weighted / hio-weighted / adaptive-feedback β=0.6 实验系统全部弱于 plain，默认保持 plain。

```bash
# 下一轮 refine 模板（曝光以只读平场现定，勿照抄）
python src/ao_shaping/main.py --dir data slm-gs-refine \
  --cam_type daheng --cam-id 0 --slm_number 1 --slm_wavelength 1064 \
  --pupil-center 960,690 --beam-radius-px 450 --target-side 100 --cam-size 250 \
  --exposure-time-ms <现定> --phase-grid 16 -e 60 --confirm-best --debug
```

- **判据**：`dec > 0.55` 且 `late_gain ≥ 10%`（与 H-19 同口径）；且最佳相位经交错重放后增益 ≥2 次一致。实验系统 composite 与仿真 0.849 同口径可比，当前 0.778 已接近，剩余差距主要在 CV。

### C3. `slm-gsnet spgd` 复测（H-7/H-8/H-9/H-13 修完后）

相对 10-01 的三处改动：ABBA 对消、小幅初始化、**`--phase-grid` 24→16（576→256 DOF）换单帧 SNR**：

```bash
python src/ao_shaping/main.py --dir data slm-gsnet spgd \
  --cam_type daheng --cam-id 0 --slm_number 1 --slm_wavelength 1064 \
  -c max --exposure_time_ms <现定> -e 50
```

- **判据**：同 C2（`dec>0.55`、`late_gain≥10%`）；产物落盘 + 0 阶 `argmax` 正确（H-6 验收项）。
- 若 16×16 收敛，再逐级升回 24×24 找 DOF-SNR 拐点。

### C4. FourierGSNet 实验系统 `run`（最后做，H-3~H-6）

前置：C2 增益可复现化或 C3 收敛——证明实验系统前向模型与测量链可信后，网络微调（collect+finetune，域随机化 N_PERTURB=240）才有意义；否则是在污染的数据上拟合。执行按 `TODO.md` §4 命令，含 `--debug` 产物归档（H-24：方形家族进 hwdataset 会换 `fov_px`，`far_field_padding` 标定需重扫）。

## 6. 红线（违反即停，AGENTS.md）

1. SLM **仅 memory mode**（`video_mode=0`），绝不开 DVI；挂起后需物理断电恢复。
2. SLM 写入**必须轮换 memory slot**（同 slot 重写是 no-op）。
3. 0 级光斑用 `argmax` 定位，禁止几何中心。
4. 相位生成 **raw 未包裹弧度**，唯一 wrap 点在 `create_phase_from_array()`；禁止 min-max 归一化。
5. 曝光显式传入实测值；`--dir` 写在子命令之前。
6. 任何实验系统脚本独立进程 + 外部看门狗硬超时；跑前确认无残留 `python*` 进程。
7. 同 seed 不保证逐帧复现——断言近似可复现，不断言相等。

## 7. 时间预算与决策点

| 阶段 | 预算 | 决策点 |
|---|---|---|
| A 离线修复 | 无需设备，1–2 个工作日 | 测试全绿 + sim 基线不退 |
| B 探针 + smoke | 激光 0.5 h | ABBA 残余 < ½ 单帧信号 |
| C2′ refine 增益可复现化（重放 + 升 DOF） | ~2 h | 重放增益 ≥2 次一致；`dec>0.55` 且 `late_gain≥10%` |
| C3 gsnet 复测 | ~1 h | 同上；收敛后升 DOF |
| C4 FourierGSNet | 半天 | C2′/C3 至少一段达标 |

**最关键的单点**已从「refine 能否在线实现」变为「**refine 增益能否复现**」：10-12 首跑证明 GS 预矫正在实验系统上有效（0.527→0.774，与仿真 0.812 同量级），SPGD 细化方向正确但增益淹没在光路双稳态里。因此下一轮实验第一动作是对 `20261012_180222` 最佳相位做交错重放 + drift 探针定量两簇，然后才谈升 DOF 攻 CV。gsnet 复测（C3）依赖阶段 A 的 H-7~H-13 修复，可与 C2′ 同场次穿插。
