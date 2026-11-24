# 闭环 refit A/B–ABBA 主实验计划（默认 vs refit）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/replay_closed_loop_abba_20261124.py`](../../../scripts/slm/replay_closed_loop_abba_20261124.py)
> **数据/关联脚本**: [`scripts/slm/refit_transfer_forward.py`](../../../scripts/slm/refit_transfer_forward.py)
> **数据/关联脚本**: [`scripts/slm/evaluate_transfer_on_new_session.py`](../../../scripts/slm/evaluate_transfer_on_new_session.py)
> **运行环境**: 离线计划；未连接或占用 SLM、CCD、激光（本轮未执行）
> **说明**: 闭环 refit A/B–ABBA 主实验计划，按文档化契约重建；默认配置与 refit 变体采用同曝光、A–B–B–A 交错与 N≥3 配对，主判据为实测 core CV。预测指标不替代实测闭环收益，实验前须重新验证重建的回放脚本。
<!-- provenance:end -->

**状态：refit 保持默认关闭；本计划冻结后才允许生成图样，禁止依据采集结果回改。**

> **重建说明（2026-11-24）**：本文件为被并发车道删除且不可恢复文档的**忠实重建**。口径直接取自
> 总账 §8 PSF/refit 行、`report/shape_trials/physical_refine/device_session_reuse_20261124.md`
> （"A/B–ABBA 主实验"、A/B 主实验与 ABBA 判定、默认路径逐字节不变）与清单 `docs/dev/hw_capture_script_inventory_20261124.md`
> （ABBA 回放 CLI 契约；"静态 PSF 双门与 refit 实测闭环收益是两个实验"）。未新增任何未记录的口径。

## 1. 目的与边界

回答闭环 refit（`refit_transfer_forward.py` → `evaluate_transfer_on_new_session.py`）相对默认配置的
**实测闭环收益**：跨会话、跨形状稳健性，以及实测整形转化是否成立。

**保持默认关闭**：refit 不改变默认路径，默认路径行为逐字节不变（`--reuse-device-session` 未传时
与既有实现一致），A/B 主实验与 ABBA 判定不受影响。

## 2. 冻结口径

- **新鲜验证集不参与选型**：refit 与默认的对比在冻结的独立数据上做，选型/调参不得接触该验证集；
- **同曝光**：A/B 两侧曝光逐字相同，来自当次相机读回范围与安全上限，不靠默认值沿用；
- **A–B–B–A 交错**：同起点、同 CCD 预算；适配耗时计入总墙钟，失败/饱和不得增加。

## 3. A/B–ABBA 结构

| 项 | 值 |
|---|---|
| 变体 A | 默认配置（refit 关闭） |
| 变体 B | refit 配置（`refit_transfer_forward`） |
| 序列 | A–B–B–A（同一会话内交错，抵消漂移） |
| 配对 | N ≥ 3 对交错 |
| 曝光 | 同曝光（见 §2） |
| 判据 | 实测 core CV 主判据；训练损失、单次回放、历史 seed→final 与仿真加速均不能替代 |

## 4. 设备与安全

- `--execute` 才开设备（ABBA 回放外壳 `scripts/slm/replay_closed_loop_abba_20261124.py` 契约：
  `--phase-a`/`--phase-b`/`--mask-npz`/`--out`/`--bit-depth`/`--wait-until-idle`）；
- `--out` 目录不可复用；`--wait-until-idle` 等不到则不启动；
- 锁一直握到回放进程返回，`finally` 释放；`device_lock.json` 写入 `--out`。

## 5. 边界

静态 PSF 双门与 refit 实测闭环收益是**两个实验**：本计划只覆盖后者。跨会话/跨形状稳健性由
§3 的 N≥3 配对与独立会话共同支撑，不在单次回放上外推。
