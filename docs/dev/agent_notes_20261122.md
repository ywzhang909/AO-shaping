# Agent 当日记录 2026-11-22（占用设备：无。全程离线，未开 SLM / CCD / 激光）

隔离工作树 `.worktrees/dynamic-shaping-drift-loop`，分支 `feat/dynamic-shaping-drift-loop`
（主工作区 `master` 有大量未提交改动，本次开发不落在主树）。**未 push、未改 issue 状态、
未触碰 #234/#236 的任何证据或门控结论。**

## 1. 本次任务：动态整形 → 全程闭环目标保持

用户已定的三条语义（不要再当成待选方案）：

1. 闭环覆盖**整个运行过程**，不是只在过渡段；
2. 位置参考是**固定 CCD 像素坐标**，不逐帧重新居中；
3. 目标轨迹是**单一最终目标形状贯穿全程**，取代原 `start → target` 动画语义。

原「SLM 动态整形」页（`gui/slm/dynamic_shaping.py`）是**开环** SLM 灰度时间轴，
并行跑的 CCD 监测不参与控制；`CCDSyncSampler` 是异步采样，**不保证逐帧配对**。
所以闭环不能建在那条链路上，而是复用已有的实测闭环入口（串行「写相位 → 稳定 →
取新帧 → 门禁」事务），新增「保持」模式。

## 2. 新增/改动（全部离线可测）

| 文件 | 内容 |
|------|------|
| `gui/slm/target_lock.py`（新增，577 行） | `TargetLockSupervisor` 等：一次运行一个固定目标；纯 numpy，无 Qt、无设备、无自主 SLM 调用 |
| `tests/.../test_target_lock.py`（新增，26 测试） | 门禁原因、固定目标语义、决策边界、冻结一次性 |
| `gui/slm/closed_loop_shaping.py` | `ClosedLoopConfig.mode ∈ {optimize, hold}` + `hold_target_ccd_px`；`run()` 拆成 `_run_optimize()`（原样）与 `_run_hold()`（新）；`RoundRecord.captured_at_s` 附加字段 |
| `tests/.../test_closed_loop_shaping.py` | 新增 7 条 hold 契约测试（40 passed） |
| `gui/slm/closed_loop_shaping_page.py` | 「运行方式」选择器（默认目标保持）、位置容差、状态行 `收敛中 / 维持中 / 已冻结`、故障红色横幅、运行中冻结目标区 |
| `gui/slm/dynamic_shaping_page.py` | 顶部开环提示 + `request_target_hold` 信号与「打开目标保持页」按钮（时间轴行为零改动） |
| `gui/ccd/beam_quality/shell.py` | 页 3 短标签 `实测闭环` → `目标保持`；`_PAGE_KEYS` / `_NAV_ITEMS` 标识符与顺序未动 |
| `test_beam_quality_shell.py` / `README.md` / `module_inventory.md` | 同步标签与登记 |

## 3. 控制契约（容易被后续改动破坏，写清）

- **配对**：每写一次 SLM 必须先 `expect_write(write_seq, written_at_s)`（单调递增、写前登记）；
  帧必须 `write_seq` 相等、`captured_at_s` 严格晚于登记时刻、`capture_seq` 递增。
  不满足只算门禁失败，**不评分、不进适配样本**。
- **绝对坐标**：目标图与目标质心在 `arm()` 时冻结；每帧都在同一张固定网格上评分，
  不裁剪、不缩放、**不重新居中**。位置容差 `hold_target_ccd_px` 由 runner 按 `config.xy`
  的步长（`_grid_spacing_px()`，轴 1 |Δx| 与轴 0 |Δy| 中位数）换算成网格单位。
- **决策是事件**：每帧至多一个 `PROPOSE / ACCEPT / ROLLBACK / FAULT`；进入 `FROZEN` 只发
  一次 `FAULT`，之后 `observe()` 返回 `None` 直到显式 `reset()`。
- **hold 不因「长期无改善」停**：`max_rounds_without_improvement` 只属于 optimize；
  hold 的停止权归监督器（漂移 / 连续无效 / 振荡）。轮数上限到了是 `STOPPED` + 保持当前相位。
- **保持模式必须 `candidate_policy="auto"`**（无人值守），`__post_init__` 会拒绝 confirm；
  界面因此在 hold 下隐藏自动接受复选框且保持未勾选。

## 4. 验证（本轮实测证据）

环境事实：本工作树**没有 `.venv`**，且 `libs/OOPAO` 子模块未初始化，`uv run` 会失败；
分两路跑 —— 无 Qt 的用 host python（`AO_ALLOW_HOST_PYTEST=1`），要 Qt 的用主仓库
`.venv` + `PYTHONPATH=<worktree>/src`。

| 命令 | 结果 |
|------|------|
| `python -m pytest -q tests/ao_shaping/gui/slm/test_target_lock.py tests/ao_shaping/gui/slm/test_closed_loop_shaping.py`（host） | **66 passed**，exit 0 |
| `.venv` python：`test_dynamic_shaping.py test_sync_ccd_monitor.py test_closed_loop_shaping_page.py test_beam_quality_shell.py` | **65 passed**，exit 0 |
| `ruff check`（9 个改动文件） | All checks passed |

未做：硬件验收、全量 pytest、graphify（工作树内无 `graphify-out/`，按项目规则不自行 init）。

## 5. 未解决的边界（下一轮不得默认已解决）

- 保持模式**只有离线验证**：没有在实机上测过「缓变漂移下能否真的保住形状与像素位置」。
- 换相机/换曝光/换目标尺寸后，`LockConfig` 的门禁阈值（饱和计数 250、最小峰值 32 counts、
  位置容差默认 5 px）仍需按实测噪声重新标定。
- #234/#236 强度门（`relL2=0.8849` vs δ=0.20，12/12 fail）与方向门（pooled 4/14 < 10）
  仍是 fail；hold 模式**不改变**这两条结论，也不得用复用数据或调模型翻案。
- #247 轴映射仍未冻结；#238 正式 g₂π 仍缺同曝光熄灯暗场。
