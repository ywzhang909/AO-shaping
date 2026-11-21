# Agent 注意事项（2026-11-17 当日修改）

> 按日追加；过期条目迁入 `docs/dev/` 对应专题。

## 上机采集（全部完成并释放）

- 光路探针 ok（0.05 ms 不饱和），三段链式采集（P1 定心 / P2 amp03 / P2 amp10）与真线性 ±tilt 补采
  全部 exit 0、正常释放。SLM #1 + CCD #0 最后 22:06:10 关闭，锁空，激光未由脚本开关。
- 设备将下线；剩余上机任务编排见 `docs/dev/next_hw_plan_transfer_candidates_issue247_20261117.md`。
- 采集期**不要**跑含 `Santec SLM` / `tilt` / `defocus` 关键词的长查询——graphify 命令行曾被误判为
  硬件进程致 census 延迟 84 s。

## Issue #247 判读（零级/目标级分离）

- 13 组全幅 Mono12 真线性 ±tilt/±离焦探针，0 饱和；flat 基线最大漂移 2 px；tilt 下零级稳定 ≤2.3 px；
  目标级两瓣分离 109–119 px → 可分离，物理前提成立。
- **核心发现：面板轴↔CCD 轴是 +88.54° 的安装旋转（非写死 90°）** —— 原始 CCD 帧上 panel-x 斜坡→
  目标沿 CCD-y（−54.50 px）、panel-y 斜坡→CCD-x（+59.19 px），即轴交换；两轴实测位移矢量经 Kabsch
  得 **correction=+88.544°**（orthogonality error 0.24°、scale ratio 1.086、max residual 2.34 px）。
  逐帧按此旋转对齐后，panel-x 位移落到 CCD-x（+54.54 px）、panel-y 落到 CCD-y（+59.28 px），
  跨轴串扰 ≤0.12 px——轴交换完全由旋转解释。前向模型必须匹配该映射（含轴间尺度比），否则
  零级校正会越校越偏。**本轮不启用硬件零级校正**（数值比例因子待离线标定）。
- 旧 P2 数据不能用：RX/RY 是 64 px 余弦探针非 tilt；标准 analyzer 的 camera_xy 网格（x∈[1300,2536]）
  落在光束（x≈1227）之外，测的是背景，其"模型不匹配"是假象。
- 报告：`report/shape_trials/issue247_zero_order_20261117/`；分析脚本
  `scripts/slm/issue247_zero_order_stability_20261117.py`（物理口径：零级=min(rate+,rate−)，
  目标级=反对称差分 rate+−rate−；旧"全局峰当零级"口径已作废）。

## Transfer `--candidates K` 批前向

- K 候选已落地，K=1 逐位一致（golden + 独立脚本双证）；`--candidates K` 端到端贯通。
- 离线 GPU 微基准（真实 FrozenDefocusForward + update()，grid 32/64/128 × 4 形状 × K=1/2/4/8）：
  单步 7–51 ms，K=8 相对 K=1 为 2.45–4.53×（次线性），K=8 显存 ≤54 MiB。
- 端到端闭环 301 s 中优化器本体 <0.5%，K 加速会被 I/O 抖动淹没 → 上机只验证"能否跑通"，
  代价以 GPU 微基准为准。报告：`report/shape_trials/transfer_candidates_timing_20261117/`。
- 与 `report/shape_trials/candidate_parallelism_20261117.md`（`train_beam_shaping` 的另一模块）区分。

## 红线与工程约束

- 全程**未改**任何 MIL 文件（`model_in_loop_shaping.py` / `slm_model_in_loop.py` /
  `shaping_controls.py` / `test_model_in_loop_shaping.py`）。
- 新报告已登记 `scripts/_common/provenance.py::REPORTS` 并跑 sync；`sync --check` clean、
  ruff 全绿。`test_report_provenance.py` 的 8 个失败是既有 UNREGISTERED baseline，与本轮无关
  （888 passed）。
- 本机 scipy.fft(workers=-1) 挂起；pytest 全量需 `--import-mode=importlib`。
