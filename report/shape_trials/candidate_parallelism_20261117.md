# 可微整形默认 8 候选并行：改动、验证与代价 2026-11-17

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线（仅 GPU，RTX 5080 Laptop；未占用 SLM / CCD / 激光）
> **说明**: 模型内 8 候选并行的改动与实测代价（手写）：`train_beam_shaping` 默认 `n_candidates=8`，单候选路径逐位不变；300 次迭代下 8 候选比单候选慢 4.6-5.8 倍，换非凸目标 1.4-2.2 % 最终损失改善，单盆地目标无收益（0.07 %）。数据与两个临时基准脚本在 `data/debug/parallel_candidates_20261117/`，不属于仓库脚本入口。
<!-- provenance:end -->

**结论先行**：模型内 8 候选并行已落地为 `train_beam_shaping` 的默认参数
（`n_candidates=8`），单候选路径逐位不变、全部相关测试通过。但实测代价必须如实记录：
在 300 次迭代的生产设置下，8 候选比单候选**慢 4.6–5.8 倍**，换来的是非凸目标上
**1.4–2.2 %** 的最终损失改善；单盆地目标（square）几乎没有收益（0.07 %）。
也就是说，这是「用时间换 1–2 % 质量」的取舍，不是免费加速。

- 数据：`data/debug/parallel_candidates_20261117/`（`results.json` + 原始日志 + 两个基准脚本）
- 设备：仅 GPU（RTX 5080 Laptop）。**未占用 SLM / CCD / 激光**，全程无实验动作。

## 1. 改了什么

| 项 | 内容 |
| --- | --- |
| 新参数 | `train_beam_shaping(..., n_candidates: int = 8)` |
| 语义 | 每次迭代用**一次**批量前向 + 一次反向同时优化 K 个候选；每步上报最优候选损失，最终返回损失最小候选 |
| 候选起点 | `initial_phase=None` → 每候选独立 `0.1·randn`；显式给出 → 每候选在其上叠加 `0.1·randn` 微扰（保留多样性） |
| 校验 | `n_candidates < 1` 抛 `ValueError`；非 `adam` 自动回退为 1（L-BFGS 线搜索会混用 K 个候选梯度） |
| 兼容 | `n_candidates=1` 与历史单候选路径**逐位一致** |

为让批量真正成立所需的两处改动：

1. `_build_forward` 的 `fftshift` 默认全维 → `dim=(-2, -1)`。对 2D 输入逐位不变，批量下
   不再错误地平移候选维。
2. 「可微传播」一节拆到新模块 `differentiable_propagation.py`（189 行），父模块顶部显式
   re-export，对外导入面不变。目的是在 800 行棘轮下完成本次功能而不放松门禁
   （`differentiable_shaping.py` 拆后 791 行）。`angular_spectrum_propagate_torch` 的维度
   校验由 `ndim != 2` 放宽为 `ndim < 2`，以支持 `(K, H, W)`。

`differentiable_objective.py` 新增 `total_loss_per_candidate`：逐候选调用 `total_loss`，
返回 `(K,)`；`K=1` 时 `per.sum()` 与 `total_loss` 逐位相同。

## 2. 正确性验证

| 验证 | 结果 |
| --- | --- |
| 批量前向 vs 串行逐位 | `absdiff = 0` |
| 批量梯度 vs 串行梯度 | 相对差 6.6e-5（float32 舍入量级） |
| 一步 Adam 后参数差 | 2e-7 |
| `test_differentiable_shaping.py` | 54 passed |
| `golden` + `quality` + `test_angular_spectrum.py` | 37 passed（连跑两次） |
| `test_diff_shaping_runner.py` + `test_step_b_debug.py` | 9 passed, 2 skipped |
| ruff（F/E/B/RUF100） | All checks passed |
| `scripts/check_module_size.py --check` | 本会话文件无新越线 |

`n_candidates=1` 的逐位一致性由 golden 测试锁定；`TestLrSchedule` 与 golden 的轨迹测试
均已显式传 `n_candidates=1`，不会被默认 8 候选破坏。

## 3. 性能与代价（SHIPPED 路径，grid 256，cuda）

### 3.1 并行效率上限

| K | s/iter | 相对单候选 | 相对 K 次串行重启 |
| --- | --- | --- | --- |
| 1 | 0.0126 | 1.00× | — |
| 2 | 0.0148 | 1.18× | 1.70× |
| 4 | 0.0197 | 1.57× | 2.55× |
| 8 | 0.0386 | 3.07× | 2.60× |
| 16 | 0.0780 | 6.20× | 2.58× |
| 32 | 0.146 | 11.6× | 2.76× |

峰值显存：K=8 1.27 GB、K=16 2.41 GB、K=32 4.68 GB。模型内并行相对「K 次串行重启」
稳定在 **1.4–2.6×**，不是 K×：收益来自摊薄固定开销和提升 GPU 占用率。

### 3.2 生产设置（300 iter，seed=42）

| 目标 | K | 墙钟 (s) | 最终损失 | 尾部中位数 | 相对 K=1 代价 | 损失改善 |
| --- | --- | --- | --- | --- | --- | --- |
| 不规则 L 形 + 分离光斑 | 1 | 3.62 | 0.47118 | — | 1.00× | — |
| 同上 | 8 | 20.84 | 0.46064 | — | 5.76× | 2.24 % |
| star | 1 | 3.56 | 0.3774 | 0.3964 | 1.00× | — |
| star | 8 | 18.74 | 0.3703 | 0.3893 | 5.27× | 1.88 % |
| triangle | 1 | 3.32 | 0.3903 | 0.4090 | 1.00× | — |
| triangle | 8 | 15.20 | 0.3849 | 0.4040 | 4.57× | 1.38 % |
| square（凸，20 iter） | 1 | 0.251 | 22.391 | — | 1.00× | — |
| square | 8 | 1.266 | 22.376 | — | 5.04× | 0.07 % |

8 候选相对 8 次串行重启只快 1.39×（`serial_equivalent_8_runs_s = 28.9 s` vs `20.8 s`）。

## 4. 对默认值的判断

需求要求默认启用 8X，代码已照此设置。但按实测，推荐按目标类型选择：

- 单盆地 / 规则目标（square、circle、gaussian）、大网格、追求吞吐 → `n_candidates=1`
  （或 2–4），8 候选在这里基本白花 5 倍时间。
- 非凸 / 多瓣任意形状、质量优先且能接受约 5 倍墙钟 → 8 候选值得。

如果后续更看重默认路径的响应时间，把默认改回 1 只需改一个参数默认值，能力保留为可选项。

## 5. 已知限制

- 仅 `optimizer="adam"` 支持多候选；`lbfgs` 自动回退 1。
- 显式 `initial_phase` 时每候选会叠加 `0.1·randn`；需要完全复现给定初值的调用方必须传 `n_candidates=1`。
- 设备侧测不出模型内并行收益（SLM 写入 + 曝光与串并行无关），设备只用于验证候选筛选结果。

## 6. issue 处置

| issue | 状态 | 处置 | 理由 |
| --- | --- | --- | --- |
| #229 Zernike 定向探针迁移 | 已 CLOSED（非本会话） | 不动 | 与本次改动无关 |
| #233 SLM 非线性融入前向网络 | OPEN | **保持 OPEN** | 本次只做优化器层候选并行，未改前向物理预测模型，无实测验收 |
| #234 zernike_block 迁移实施报告 | OPEN | **保持 OPEN** | 实施提交 `d7692534` 在 `feat/zernike-block-transfer` 分支，未合入 master，且无实验实测 |
| #235 零级中心口径与 shift_phase 一致性 | OPEN | **保持 OPEN** | 零级中心已可选（提交 `0fd76e44`），但实验方向门槛未执行，实测未验收 |

实验 P1+P2 计划（`docs/dev/hw_direction_test_plan_20261117.md`）等待激光恢复后执行；
在方向门槛与实测验收通过前，不关闭任何硬件 / 迁移类 issue。
