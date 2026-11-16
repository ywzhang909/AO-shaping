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
