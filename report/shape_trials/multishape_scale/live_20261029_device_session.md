# 2026-10-29 设备在线多尺度整形：实验中记录

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`src/ao_shaping/runners/slm/transfer_shapes_runner.py`](../../../src/ao_shaping/runners/slm/transfer_shapes_runner.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/validate.py`](../../../src/ao_shaping/tools/slm/transfer/validate.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/capture.py`](../../../src/ao_shaping/tools/slm/transfer/capture.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/fit.py`](../../../src/ao_shaping/tools/slm/transfer/fit.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/replay_phases.py`](../../../src/ao_shaping/tools/slm/transfer/replay_phases.py)
> **数据/关联脚本**: [`scripts/slm/audit_replay_blocks.py`](../../../scripts/slm/audit_replay_blocks.py)
> **数据/关联脚本**: [`scripts/slm/probe_optical_path_validity.py`](../../../scripts/slm/probe_optical_path_validity.py)
> **数据/关联脚本**: [`scripts/slm/analyze_session_time_budget.py`](../../../scripts/slm/analyze_session_time_budget.py)
> **运行环境**: 硬件（Santec #1 SLM + Daheng CCD；WFS / NLight DM / MIICAM 未连接）
> **说明**: 2026-10-29 设备在线多尺度整形会话的实测记录：lane 00–05 实际执行、06–24 未开始（25 条编制中完成 5 条、部分完成 1 条）。200 px 解析初始化四形状各 4/4 阶段完成，冻结光阑重放审计（已剔除饱和行）给出circle（15/15 有效，dCV −0.286/−0.126/−0.127，三档 supported）与 triangle（13/15，−0.161/−0.173，两档 supported）；square（11/15，+0.170/+0.165，两档 wrong_direction，约噪声三倍 → 真实回归）与 star（5/15，仅一档，no_difference）。600 px 仅 circle 前两段（1.0986→0.7993、0.7447→0.6268），无重放，不参与跨尺度对比。含审计工具缺陷（`clipped` 行被标注却仍进入分组均值）及其本轮修正（剔除后 circle 一位未变，triangle/square 各失去一档，star 失去两档）。
<!-- provenance:end -->

> 本文是**当次设备会话的运行记录**（`lane 00–05` 实际执行、`06–24` 未开始）。
> 判定口径与跨会话汇总见同目录 `multishape_scale_live_20261028.md` §7；
> 提速与质量改动见该文 §5.7 / §5.8。

## 1. 会话目标

在 2026-10-28 的 1200 px 单尺度结果之上回答两件事：

1. **尺度是否可外推**——同一流程在 200 / 600 / 1200 px 目标上是否都给出可复现的
   CV 改善；
2. **初始化是否必须分开报**——`严格冷启动`（平场 / 随机相位）与 `解析初始化`
   （反投影 / 能量映射）是否给出不同结论。

第 2 点用**目录隔离**保证不可合并：两类初始化落在不同会话目录

```
p20261029_strict_cold_s{200,600,1200}_<shape>
p20261029_analytic_init_s{200,600,1200}_<shape>
```

## 2. 前置门禁

| 项 | 状态 | 证据 |
|---|---|---|
| 光路有效性探针 | 通过，退出码即判定，未通过不开跑 | `scripts/slm/probe_optical_path_validity.py` |
| 平场基线 | 通过（`flat_restored=true`） | 会话开头与每次中断后 |
| 模型检查点 | lane 00 重拟合产出 | `p20261029/frozen_parameters.json` |
| 曝光上限 | `AO_DAHENG_MAX_EXPOSURE_MS` 未设 → 画面探测 | 逐块 `adapt_exposure` |

被遮挡的实验系统读数**不写盘**：先过探针，失败即终止该 lane，不产生可引用结果。

## 3. lane 编制（25 条）

| lane | 会话 | 形状 | 状态 |
|---|---|---|---|
| 00 | `p20261029`（模型重拟合） | — | 完成 |
| 01 | `analytic_init_s200` | circle | 完成（4/4 阶段） |
| 02 | `analytic_init_s200` | square | 完成（4/4 阶段） |
| 03 | `analytic_init_s200` | triangle | 完成（4/4 阶段） |
| 04 | `analytic_init_s200` | star | 完成（4/4 阶段） |
| 05 | `analytic_init_s600` | circle | **中途终止**（2/4 阶段） |
| 06–08 | `analytic_init_s600` | square / triangle / star | 未开始 |
| 09–12 | `analytic_init_s1200` | 四形状 | 未开始 |
| 13–24 | `strict_cold_s{200,600,1200}` | 四形状 | 未开始 |

每条 lane 一个 `data/debug/slm_transfer/multishape_nogs_20261029/lanes/*.ps1`，
参数写死在脚本里，避免命令行漂移。

## 4. 时间线（UTC，本地 = UTC+8）

| 时刻 | 事件 |
|---|---|
| 01:36:44 | lane 00 启动（模型重拟合） |
| 01:54:18 | lane 00 完成 |
| 02:34:43 | lane 01（circle）`common_replay` 释放设备 |
| 02:43:50 | lane 01 resume 复跑（journal 被重写，见 §6） |
| 02:53:41 – 03:03:25 | lane 02（square） |
| 03:04:33 – 03:13:23 | lane 03（triangle） |
| 03:14:32 – 03:26:47 | lane 04（star） |
| 03:45:40 – 03:47:11 | lane 05（s600 circle）前两阶段 |
| 03:50 | 设备下线，链路与被中止 lane 全部终止 |

## 5. 逐组合结果

判定用 `scripts/slm/audit_replay_blocks.py` 的**冻结孔径**口径重算，不用阶段
manifest 的 `metrics.cv`——后者用各阶段自己的 `objective.core`，逐阶段腐蚀量
不同，跨阶段不可比。

### 5.1 `analytic_init_s200`：三口径 + 饱和审计

每组 3 档曝光 × 5 行 = 15 行，12 bit，零级排除比例 `0.2719`（四形状一致）。
`clipped` = 峰值 ≥ 98% 满量程的采集，**已从所有分组均值中剔除**
（`scripts/slm/audit_replay_blocks.py` 本次修正：它此前打印「excluded from any
comparison」，但分组均值仍包含这些行）。

| 形状 | 有效行 | `dCV`（seed→final，按曝光档） | 判定 | 噪声半宽 | `ee_rel` |
|---|---|---|---|---|---|
| circle | 15 / 15 | **−0.2864** / **−0.1263** / **−0.1273** | supported ×3 | 0.0925 / 0.0396 / 0.0291 | 0.307 / 0.317 / 0.318 |
| triangle | 13 / 15 | **−0.1613** / **−0.1732** | supported ×2 | 0.0132 / 0.0181 | 0.273 / 0.270 |
| square | 11 / 15 | +0.1704 / +0.1646 | `wrong_direction` ×2 | 0.0526 / 0.0547 | 0.243 / 0.248 |
| star | **5 / 15** | +0.0784 | `no_difference` ×1 | 0.0991 | 0.360 |

（第 3 档在 triangle 与 square 上都没有 seed 行可比——前者 seed 被截断，后者
4 个 final 全被截断——因此不产生 `dCV`。star 的后两档整组不可用。）

四条读法：

1. **circle 是唯一三档全干净的形状**：0 饱和行，`dCV` 在三档上同号且都越过
   重复性半宽，`ee_rel` 随曝光略升（0.307→0.318），不是「丢能量换均匀」。
2. **triangle 的改善成立，但只有两档**：第三档的 seed 行被截断，只剩 3 个 final，
   无法比较。这不影响前两档的结论。
3. **square 的 `wrong_direction` 是真的**：剔除饱和行后，两个可用档的
   `dCV` 仍为 **+0.17 / +0.16**，而它们的噪声半宽只有 0.053 / 0.055——效应量约为
   噪声的 3 倍。第三档的 4 个 final 全部截断、无法判定。所以 square 不是
   「测不到」，而是**闭环把它做差了**。
4. **star 只剩一档可判**：另两档整组截断（共 10/15 行）。唯一可判的那档是
   `no_difference`，`ee_rel` 0.360 明显高于其他三个形状——星形目标的光斑能量
   分布与方/圆/三角本就不同，只用 CV 描述它会漏掉信息。

> **本节的数字与「修正前」不同**：修正前 square 报 `+0.1578`（含饱和行）且
> star 报「两档 `no_difference` + 一档 `wrong_direction`」——后者的那一档整组都是
> 饱和行，**已撤回**。修正方向见 `next_round_20261029.md` 的 P0-2（现已落地）。

### 5.2 阶段级 CV 轨迹（`analytic_init_s200`，仅同形状内可比）

| 形状 | 阶段 | 行数 | seed CV | final CV |
|---|---|---|---|---|
| circle | `transport_cv` | 8 | 1.1943 | 1.0577 |
| circle | `mono12_cv` | 20 | 0.9534 | 0.9854 |
| circle | `mono12_continue` | 28 | 0.9905 | 0.9517 |
| circle | `mono12_dense` | 12 | 1.0645 | 1.0065 |
| square | `transport_cv` | 8 | 0.6767 | 0.6516 |
| square | `mono12_cv` | 20 | 0.6432 | 0.6921 |
| square | `mono12_continue` | 28 | 0.6963 | 0.7262 |
| square | `mono12_dense` | 12 | 0.7448 | 0.7021 |
| triangle | `transport_cv` | 8 | 0.6474 | 0.6618 |
| triangle | `mono12_cv` | 20 | 0.6581 | 0.7050 |
| star | `transport_cv` | 8 | 0.6406 | 0.6583 |
| star | `mono12_cv` | 20 | 0.6450 | 0.6368 |

**只能横读**（同形状、相邻阶段的 seed → final）。竖读会把不同腐蚀量的 `core`
当成同一把尺子。跨阶段、跨形状一律用 §5.1。

值得注意的是：`transport_cv`（8 bit、无 CCD 校验）在四个形状上都几乎不动甚至
略升，而 `mono12_cv`（12 bit、逐块 CCD 校验）才开始出现真正下降——与 §5.8 的
「模型方向不可信时盲目走步」一致。

### 5.3 s600 circle（lane 05，未完成）

| 阶段 | 行数 | seed CV | 末条 CV | 状态 |
|---|---|---|---|---|
| `transport_cv` | 8 | 1.0986 | 0.7993 | 完成 |
| `mono12_cv` | 16 | 0.7447 | 0.6268 | **被终止**（8 块中 4 块） |

两阶段都朝下，但 `mono12_cv` 只跑一半，**不构成结论**，也不参与跨尺度对比。
该目录没有 `common_replay`、没有 `released.json`。

### 5.4 会话级记账

| 项 | 值 |
|---|---|
| 完成会话 | 5（lane 00–04） |
| 完成阶段 | 17（16 完整 + 1 部分） |
| 完成验收块 | 112（4 形状 × 28） |
| 零级排除比例 | 0.2719（s200 全部 12 bit 组一致） |
| 饱和行 | circle 0 / triangle 2 / square 4 / star 10（各 15） |
| 有效重放行 | circle 15 / triangle 13 / square 11 / star 5（各 15） |
| 模型检查点 | `p20261029/frozen_parameters.json` |
| 相位来源 | 逐阶段 `manifest.json` 的 `phase` / `seed_run` 字段 |
| 预测 vs 实测 | 逐块 `jacobian` 字段**本次不存在**（该守卫于本次之后实现） |

## 6. 失败与修复

| 现象 | 根因 | 处理 |
|---|---|---|
| lane 01 首次重放 34/64 次后无 `released.json` | 核心区饱和触发守卫，replay 中止 | 改曝光梯重放；`released.json` 是完成标记，`config.json` 不是 |
| lane 01 的 `journal.json` 只剩一条 `plan` | `_Journal` 每次进程启动为空，`--resume` 会**重写**整份 journal | 已知行为；阶段历史以各阶段目录与 `journal_first_attempt.json` 为准 |
| lane 02 seed-only 试跑被跳过 | resume 门按 `manifest.json` 存在与否判断，seed-only 试跑也写 manifest | `7d672959` 修：`blocks=0` 时同样写 manifest 并标记 |
| 重放 ladder 跨尺寸共用 | ladder 按 1200 px 标定，200 px 首档即饱和 | `d0805383` 修：ladder 随 `side` 缩放 |
| `audit_replay_blocks.py` 读不到 s200 目录 | 默认平铺布局，本次是 `<session>/<shape>/common_replay` | 传 `--session <session>/<shape>` |

## 7. 设备释放

- 03:50 UTC 终止；所有阶段进程、lane 链与被中止的 relaunch 均已结束。
- SLM 与 Daheng CCD 由上下文管理器关闭，`released.json` 记于各 `common_replay`。
- **下一位操作者**：
  1. `p20261029_analytic_init_s600/circle` 的 `mono12_cv` 是半成品，`--resume`
     从第 5 块继续；该目录没有 `common_replay`，重放需单独跑；
  2. lane 06–24 从未启动，目录不存在，可直接按 `lanes/*.ps1` 开跑；
  3. `--profile-timing` 只在 lane 05 打开（01–04 运行时该开关尚未提交），
     `--async-capture-write` 全程关闭 ⇒ 01–04 与 05 的采集耗时口径不同。

## 8. 结论与未决

1. **200 px 解析初始化**：circle 的改善在三档曝光上全部成立（0 饱和行，
   `ee_rel` 未降）；triangle 在两档上成立（第三档失去 seed 行）。
2. **square 是本轮唯一「真的变差」的形状**：剔除饱和行后，两个可用档的
   `dCV` 仍为 +0.17 / +0.16，约为噪声半宽的 3 倍。这不是测量问题，是**闭环把它
   做差了**——正是 §5.8 与 issue #213 的 3a 要查的模型方向问题。
3. **star 只有一档可判**（`no_difference`，`ee_rel` 0.360 为四形状最高），另两档
   整组截断。星形目标需要形状特定的指标，CV 单值不够。
4. **600 / 1200 px 与全部 `strict_cold` lane 没有数据**，两类初始化的对比问题
   **仍未回答**。s600/circle 仅两段，且没有重放审计。
5. **质量侧前两项**（3a 逐块 Jacobian 校验、1b 频域带限）代码已实现并通过单元
   测试，尚无设备数据；1b 默认关闭、等实测 MTF。下一次实验按
   `next_round_20261029.md` 的顺序跑。
