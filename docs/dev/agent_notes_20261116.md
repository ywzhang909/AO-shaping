# Agent 注意事项（2026-11-16 当日修改）

> 按日追加；过期条目迁入 `docs/dev/` 对应专题。
> 当日无新 commit（工作区改动），聚焦 **可微 Step B 逐迭代调试模式** 与 **SLM+CCD 物理设备验证**。

## 可微 Step B 调试模式（`debug=`）

- 新增叶子模块 `src/ao_shaping/optimizer/wfless/step_b_debug.py`：纯 numpy + stdlib，无 torch、
  不碰设备，通过鸭子类型注入，避免 `optimizer.wfless` 包与 `algorithm` 的导入环。
- 记录每迭代：物理预测模型 loss、相位梯度范数/max/RMS、参数更新范数
  `‖phase_after − phase_before‖`（与梯度范数分开，勿混淆）、`forward+backward` 与优化器步耗时
  —— 用来判断该提速物理预测模型还是优化器。崩溃安全 JSONL 追加 + 汇总 JSON + 可选梯度快照 npz。
- 绑定点 `differentiable_shaping.train_beam_shaping(debug=...)`：`debug` 为目录路径时自建记录器、
  跑完 `close()` 写汇总；为记录器实例时透传、生命周期由调用方负责。记录在相位 wrap **之前**
  进行，避免 2π 跳变污染 `phase_delta_norm`。
- 为过 800 行规模门禁，目标/损失族（uniformity/efficiency/zero-order/smoothness/total）
  **逐字**迁至新模块 `differentiable_objective.py`，旧导入路径用显式 re-export 保持不变。
  `differentiable_shaping.py` 852 → 769 行，`--check` 复测退出码 0。

### 物理脚本 `hw_differentiable_phase_opt.py` 的接线

- 计时下沉到 `DifferentiableBeamOptimizer.update()`（新增 `last_forward_s` / `last_step_s` 属性）：
  硬件脚本走自己的 Adam 循环、绕过 `train_beam_shaping`，只有在那里接线才能拿到物理路径耗时。
- 三条路径共用一个 `StepBDebugRecorder`（tag 依次为 `step_b` / `step_b_reanalysis`）：
  模型侧 Adam 循环、P6b 测量锚定闭环（`opt.update(measured_intensity=...)`）、`--analyze-only`
  离线重分析循环；每条路径循环后 `debug_recorder.close()` 写汇总。
- CLI 开关 `--debug-gradients`：默认只记标量，开启后额外存每步梯度场与相位快照 `.npz`。
- 记录器的 `forward_s` 即「物理预测模型前向 + 反向」耗时，`step_s` 即优化器步耗时。

## SLM+CCD 物理闭环实测

- 设备：SLM #1（Santec，1064 nm，video_mode 0）+ Daheng MER2-507-23GM（sn qw5626oq，
  2592×1944 uint8，像元 2.2 µm，曝光回读 2.0 ms）。本轮 SLM 与 CCD **各开/关一次**并已干净
  `close()` 释放，无背对背 open/close。
- 运行：`--execute --iters 120 --hw-iters 3 --wait-until-idle 300`，产物
  `data/debug/differentiable_opt/phys_20261116_01/`（run_config / history*.csv / ccd_*.npz /
  phase_*.npy / figures / run.log）。在焦 ✓（最佳 FWHM 15.55 px，diopter 0.0）。
- 形状判据（自行从读回帧复算，曝光归一 + 背景扣除 + 噪声 σ 阈值化）：目标 60×60 方块内信号量
  optim/flat = **0.901×**，全屏 0.989×，方块内占比 flat 100% → optim 91.2%。形状改善成立
  （散斑 → 方斑上屏），但**按曝光归一的目标块通量未提升**。
- 实测方块仅 ~32×24 px，远小于模型预期 60×60 px；实测 >25% 峰值区域 flat 22×25 px、
  optim 32×24 px。口径疑点：模型 dp=3.75 面板像元（上屏为块状相位）、padding=4 远场采样、
  面板↔相机 90° 轴交换、K/瞳面映射标定等（详见 run_config.json 的 `open_questions`）。
- 模型侧 loss 3.39e-4 → 1.33e-4（120 次 Adam，cuda）；测量锚定实测 loss 7.026e-4 → 7.004e-4（3 次）。

## 提速方向判定（离线重分析 + 物理短测）

- `--analyze-only` 重跑 120 次纯模型 Adam 循环，产出
  `step_b_reanalysis_debug.jsonl` + `step_b_reanalysis_debug.json`。
- **物理短测**（`--execute --iters 20 --hw-iters 3`，SLM+CCD 各开一次并干净关闭）产出
  `phys_20261116_02/step_b_debug.json{l}`：模型侧 20 次 + 测量锚定 3 次共 23 条记录。
- 两条路径一致：**求解时间大头在物理预测模型前向+反向，优化器步是少数**。
  离线 `forward_share = 0.91`（120 次，含首迭代 CUDA 预热）；物理路径 `forward_share = 0.78`
  （23 次，forward 中位 2.0 ms、optimizer 中位 0.37 ms）。⇒ 提速应作用于传播（物理预测模型），
  不是优化器。（注意：此拆分只覆盖 `update()` 内的前向+反向 vs 优化器步；物理闭环的整步墙钟
  还含 SLM 下发 + CCD 采集 + 重采样，不在本次拆分口径内。）
- 物理短测：在焦 ✓（FWHM 15.87 px ≤ 40 px）；模型 loss 3.39e-4 → 2.06e-4，测量锚定实测
  loss 6.74e-4 → 6.69e-4；模型 vs 硬件相关 0.663、峰值偏移 -6/-6 采样、类 Strehl 9.572。

## 当日门禁

- `scripts/check_module_size.py --check` → 退出码 0（无新越线）。
- `ruff check` 改动文件 → All checks passed。
- `term_unify` dry-run → **0 个文件**（旧写法统一为 `物理预测模型`）。
- `test_step_b_debug.py` / `test_differentiable_shaping.py` / `test_differentiable_beam*.py` /
  `test_optimizer_doc_coverage.py` → 83 passed（`--import-mode=importlib`）。

## 备注

- Qt「SLM 静态整形」页实际位于 `src/ao_shaping/gui/slm/static_shaping_page.py`（非
  `static_shaping_ops.py`），Offscreen 冒烟可建页；下发路径已走 `display_data` 后台线程，
  与 Streamlit 同源。早前按错文件名寻找时误判为不存在。
- 早前 SLM-only 验证（`get_displayed_phase()` max_abs_diff=0，source 为内存槽 47）已由上述
  SLM+CCD 闭环实测取代；驱动缓存槽读回**不是**光学改善证据。

## 提速参数落地（本轮，commit b8b9222）

- **光路复用默认开**：`workflow.TransferSession.reuse_latest` / `transfer_desktop.params` / GUI
  控件默认 `True`。复用快路径在**任何**失败模式（设备打不开 / 无序列号 / 无 latest / 光路已变 /
  网格不足）都回退全流程，默认开最坏只多一次廉价探针帧，安全。
- **Step B 预算 600 -> 150**：`SlmModelInLoopConfig.step_b_iterations`、`StepBConfig.iterations`、
  CLI `--step-b-iterations`、GUI `MIL_WIDGET_DEFAULTS`/预设、`model_in_loop_hw_runbook.py`
  `--shape-iterations` 全部收口到 150。依据：硬件有效采样几何（`far_field_size=4096`）下 150 步
  CV 13.83 -> 0.651、EE 1.0000 无损；文档记录 400 步比 80 步模型内 loss 更低但实测 CCD 更差
  （过拟合），故更多步既不快也不好。
- **dry-run 契约修复**：`hw_differentiable_phase_opt.py` 顶层 import `step_b_debug` 会经
  `ao_shaping.optimizer.__init__` 牵入整个 `ao_shaping.drivers` 包，导致 `assert_no_drivers` 违约
  exit 3。改为在 execute 路径内延迟导入。修复后 `--dry-run` 两个入口均 exit 0。
- 实验入口已就绪：`hw_differentiable_phase_opt.py`（dry-run / --execute / --analyze-only）、
  `model_in_loop_hw_runbook.py --stage shape`（dry-run）。退出码约定见脚本 docstring。
- 门禁：`check_module_size --check` 0、`term_unify --check` 0 文件、ruff 通过、定向 pytest
  509 passed（`--import-mode=importlib`）。
- 修正：`test_slm_model_in_loop.py` 中 `test_camera_pixel_pitch_is_not_validated` 为失效的
  latent-bug 表征测试（pitch 现已由 `__post_init__._resolve_camera_pixel_um` 解析），删除。

## 物理设备实跑：`hw_differentiable_phase_opt.py --execute`（2026-11-16 19:44–19:47）

- **设备占用**：SLM #1（Santec SLM-200，1064nm，22030108）+ Daheng MER2-507-23GM CCD
  （FJB24112232，2592×1944，像元 2.2µm）。一次开/关，`--wait-until-idle 300` 公平仲裁，
  19:44:52 开、19:47:09 干净 `close()` 释放，无异常/无第二个占用进程。
- **命令**：`.venv\Scripts\python.exe scripts\slm\hw_differentiable_phase_opt.py --execute
  --iters 120 --hw-iters 3 --wait-until-idle 300`，运行目录
  `data/debug/differentiable_opt/20261116_114424`。
- **在焦门禁通过**：焦面扫描 13 点 + 7 点细化，最优 FWHM 15.68 px ≤ 40 px 门限，未触发退出 5。
- **物理预测模型**：N=256 / M=1024 / dp=3.7500，远场采样 1.9688 CCD px，覆盖 2016 CCD px，
  目标 square，设备 cuda。
- **纯模型侧 120 步**（history.csv，device=cuda）：loss 3.394e-4 → 末值 ~1.3e-4 量级单调下降，
  相位 RMS 1.839→1.843 rad 几乎不变（印证「模型内 loss 下降 ≠ 相位显著变化」的既有结论）。
- **模型 vs 硬件（P6）**：强度相关系数、峰值偏移、类 Strehl 均来自 CCD 实测读回帧（非
  `get_displayed_phase()` 缓存）。
- **测量锚定闭环 3 轮**（history_hw.csv，实测读回帧）：实测损失 9.08e-4 → 7.02e-4 → 6.98e-4，
  梯度范数 0.949 → 0.815（下降中），相位 RMS 1.842→1.840 rad。梯度方向在实验系统上仍有效。
- **产物**：`ccd_flat.npz` / `ccd_hw_iter_000/001/002.npz`（读回帧）+ `history.csv` /
  `history_hw.csv` / `run_config.json` + `figures/` + 报告
  `report/slm/differentiable_opt/differentiable_opt_report.md`（本轮重生成，时间戳 19:47:09）。
- **小瑕疵**：报告头部 provenance 描述里「200 次反向传播」是生成器硬编码的陈旧串，实际
  `--iters 120`（run_config 已确认）；正文与数据均为 120，非数据错误，待修生成器描述串。

## provenance 陈旧串修复 + 提速「前后对比」报告（2026-11-16 收尾）

- **provenance 陈旧 note 修复**：`differentiable_opt_report.md` 条目的 note 此前是 20261022
  旧轮硬编码串（0.089 / 200 次），改为本轮实测值（120 次纯模型反向传播、相关系数 0.599、
  峰值偏移 [6.0,-6.0]、类 Strehl 17.226、实测损失 7.018e-4→6.969e-4、梯度范数 0.949→0.634），
  复现命令改为 `--execute --iters 120 --hw-iters 3`，保留两个负结果子句。`differentiable_opt_animation_20261022.md`
  及 agent_notes 中的 0.089/200 次等历史数字**不动**（属 20261022 真实历史数据）。
- **「前后对比」报告**：新增
  `report/slm/differentiable_opt/speedup_before_after_20261116.md`（人工撰写，
  `provenance.py` 以 `HANDWRITTEN` 登记，环境「离线+硬件」）。覆盖三项：完整时间消耗
  （硬件测量段墙钟 59.6–68.4 s、模型侧纯计算 <1 s 且前向占 0.79–0.91、Step B 扫描
  600 步比 100 步多花 39% 墙钟）、迭代次数变化（Step B 600→150、`reuse_latest` 默认开、
  物理实跑 120/20/120 + 3 步硬件锚定）、整形质量变化（扫描 CV/EE 非单调 → 减步不牺牲质量；
  150 步硬件标定 CV 13.83→0.651、EE 1.0000；实测损失/梯度单调降；跨轮一致性
  0.089–0.810 对照本轮 0.599–0.663）。负结果保留：模型内 loss 下降≠光学改善、原始峰值比无意义、
  `get_displayed_phase()` 不可作证据。扫描数据已拷入报告 artifacts 目录随报告入库
  （`data/debug/` 本身 gitignore）。
- **门禁**：`sync_report_provenance.py --check` clean（exit 0）、term_unify 0 文件、
  `check_module_size --check` exit 0、ruff 通过、`test_report_provenance.py` 仅剩 3 个
  先前已存在的无关失败（`tmp_sac_run_showcase.md` / `sensitivity.md` / `objective_comparison.md`）。
- 术语门禁修复：agent_notes 中「实验入口」按统一表改为「实验入口」（`term_unify --apply`）。
- 提交：`b8b9222`（提速参数 + dry-run 契约）、`ff4921c`（物理实跑记录）、本轮
  provenance 修复 + 对比报告。`01e25e4`、`b8b9222`、`ff4921c` 仍未 push。
