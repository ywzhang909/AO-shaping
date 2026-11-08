# Agent 注意事项（2026-11-06 当日修改）

> 外迁自根 `AGENTS.md`；按日追加，过期条目迁入 `docs/dev/` 对应专题。
> 覆盖当日十个提交（`be6a5332`/`10a0240e`/`d3b351da`/`bfbfdf5e`/`75f7f846`/`a1b405fe`/`7f7bd578`
> 及三个 docs 跟进），即 issue #205 第二十六～二十九批 + transfer 顺序 CCD 验收 + transfer-desktop 修复。
> 当日只改了 `docs/dev/ci_testing.md`（+5：门禁测试名与 `>=` 禁令从根 AGENTS.md 迁入）；`TODO.md` 由 1–4 提交更新。

## optimizer/wfless `slm_square_shaping.py`（#205 第二十六～二十八批）
- **第二十六批**：非 SPGD 启发式分支提为独立阶段 `_run_square_heuristic_stage(*, algorithm, best_ar, best_cv,
  best_ee, best_quality, cam, center, delta, display_ctx, display_stack, epochs, last_best_epoch, objective,
  pop_size, random_seed, recorder, slm, target_mean_brightness, target_side, _dim, _heuristic_bounds,
  _n_energy_gated, _params, _params_to_gray, _record_vec, _score_column, w_aspect, w_efficiency, w_pbr,
  w_uniformity)`——从 `_run_square_search_stage` 原样搬出，**该分支自身结束会话**，不返回任何 epoch 循环状态。
  `best_cost` / `best_c` / `best_img` 改为**启发式分支局部**（在 `_on_evaluate` 首次写入前显式初始化）；SPGD 循环
  不再写它们。调用方保留的五名是 `best_quality` / `best_cv` / `best_ee` / `best_ar` / `last_best_epoch`。
- **第二十七批**：`_square_stage_weights(*, _adaptive_boost_active, _adaptive_cfg, _cv_history, epoch, epochs,
  w_aspect, w_efficiency, w_uniformity)`——自适应权重调度提为纯函数：按 **epoch 三等分**经
  `get_stage_weights(_adaptive_cfg, epoch, epochs)` 排质量权重，再用 `should_boost_stagnation(_cv_history, win,
  min_improve_frac)` 判 best-CV 停滞（`stagnate_win`），`_adaptive_cfg.enable_stagnate_boost` 且
  `stagnate_boost > 1.0` 时临时抬 `w_uniformity`；`_adaptive_cfg is None`（自适应关）原样交回四名。
  ⚠️ **自适应 SPGD 仅 `--adaptive` 开启时生效**。另提纯 `_select_better_square_perturbation(...)`（参数 27→13）。
- **第二十八批**：`_square_perturbation_measurement(*, _params_to_gray, c, cam, center, objective, slm,
  target_side, w_aspect, w_efficiency, w_pbr, w_uniformity)`（52 行 / 嵌套 0 / 11 参）——±`disturb_c` 成对扰动
  测量收成**一个记录**。原先 SPGD 各 epoch 对 ±disturb_c 重复同一过程，两份拷贝**仅变量前缀不同**——静默分叉
  隐患（单边指标掉落/权重过期会让该对发散）。⚠️ **勿再引入两份各自拷贝的成对测量块**，这就是本批消除的分叉。
  ⚠️ **非原样搬移**：由新测试 `tests/ao_shaping/optimizer/wfless/test_slm_square_epoch_characterization.py`
  （+256）以数值等价证明。相机**按设计相位相关**（blob 偏移/各向异性跟随最后写入相位），故
  `pos["q"] != neg["q"]`、梯度非零——pos/neg 混写旧测试里不可见（`test_slm_square_objective_search_sim.py`
  固定 blob 每 epoch 都 `pos["q"] == neg["q"]`）。
- **参考（真值）量化**：`test_slm_square_epoch_characterization.py` 由**全精度浮点字节**改为**量化哈希**（4 位小数 /
  5 位有效数字；`_quant` 显式 `<f8` 强转——机器无关字节序，非装饰）。跨机 flake 根因：SPGD 链是 float32/float64，
  归约顺序随 BLAS/FFT 分核而变，末位本可合法不同；1 ULP 即翻转整个 sha1 → "分片红、重跑绿、测试名丢失"
  （即 TODO §8.9 未结案例）。线程数无关（`AO_CPU_THREADS` 1/4/8 三次位相同）；残余漂移 ~1e-7 相对，
  量化格 1e-4 低 3 个量级，最小值得抓的行为高 2 个量级。⚠️ 重基线用 `AO_SQUARE_CHAR_UPDATE=<path>`
  （opt-in，镜像 `AO_PROBE_HELP_UPDATE=1` 惯例；未设 ⇒ 断言与行为完全一致），并保留 `_quant` 的 `<f8` 强转。
  -disturb_c→+disturb_c 仍 6 failed；批量 28 前源码 `176bf838` 仍 7 passed（行为等价证明成立）。失败消息自带
  逐列首两行数值（`_preview`）——一处红即可钉住案例，无需再 grep 汇总行。
- **形状门禁**：`tests/ao_shaping/test_issue205_function_shape.py` 棘轮表给每个新提取函数加
  `(max_lines, max_nest, exact_lines)`：`_run_square_search_stage` 600→430（387）、`_square_stage_weights` 95/3/52、
  `_select_better_square_perturbation` 150→135（92）、`_square_perturbation_measurement` 95/0/52、
  `_run_square_heuristic_stage` 210/3/168。⚠️ 凡重构改变模块尺寸，**同步更新**
  `scripts/_common/module_size_baseline.json`（slm_square_shaping 2482→2556→2681→2684）。

## tools/slm/transfer `workflow.py`（第二十九批，`run_session` 583→378）
- 四个模块级**原样搬移**（AST 逐句证明，3/7/1/11 句）：`_acquire_session_dataset(config, *, out, device, stage,
  on_frame, offline)`（63）采集+几何预检/廉价复用侧光路检查；`_round_geometry_preflight(...)`（28）实测零点+拟合
  斜率增益定目标越界；`_offline_synthesize(...)`（98）旧 `if offline:` 嵌套 `synthesize` 闭包；`_round_calibration_stages(...)`
  （39）曝光-响应采集/分析+模糊标定。
- ⚠️ **延迟导入纪律**：`validate`/`fit`/`capture` 在 `run_session` 内延迟 import（顶层 import 会环），故各助手
  自带 `from . import …`（沿用文件既有延迟导入风格）；`run_session` 的 `from . import bootstrap, capture, …` **删掉
  `capture`**（两处使用已随块搬走，留着是 F401）。模块 1521→1602（+81；≈40 签名/调用样板 + ≈40 docstring，函数体净
  −205）。
- **`_WF` 首组门禁**（5 条）：`run_session` 430/4/378、`_acquire_session_dataset` 95/4/78、`_round_geometry_preflight`
  60/1/42、`_offline_synthesize` 130/3/108、`_round_calibration_stages` 70/1/52。棘轮表 75→80 行；基线 workflow 1521→1602。

## transfer `validate.py`（顺序 CCD 验收 + 保持最优相位）
- **新 CLI**：`--dump-history`（逐形状梯度/优化记录——优化器 `loss_history` 各项——写
  `history/{shape}.jsonl`，模式 **"w"** 覆盖）与 `--timing-out`（逐形状墙钟+迭代数+`model_cv` 追加
  `timing.jsonl`，模式 **"a"`）。均默认关，未设时行为与旧完全一致。迭代记录复用共享 `iterations.pkl`；`model_cv`
  用 `forward.predict(best)` + `image_metrics`（需 `forward`/`objective`/`best` 在作用域内）。
- **`_sequential_ccd_decision(args, anchor_row, b1)`**（line 114）：顺序 CCD 验收带容差带
  （`accept_rel_tol`/`accept_abs_tol`，默认 0.0）：`tolerance = max(abs(av)*rel_tol, abs_tol)`、
  `accepted = bv < av - tolerance`；返回 `(metric, av, bv, decision)`，`decision = {accepted,
  acceptance:"sequential_ccd", accept_metric, baseline, candidate, tolerance}`。metric 为 `cv` 若
  `args.accept_metric=="cv"` 否则 `loss`（默认 `loss`）。容差带给端点漂移留余量，设备抖动不会把物理平局判拒。
- ⚠️ **验收默认主导**：`_run_validate_blocks` 中原「无实测结论 → `model_only_decision(loss_history)`」现为——
  **`anchor_row is not None` 且 `acceptance_enabled is False`** 时走 `_sequential_ccd_decision`（加 B1，CCD 定结论）；
  `loss_ref = model_only_decision(...)` 仅留作参照。**`acceptance_enabled=True` 时不加 B1**（旧循环时序保留，
  model-only 验收路径不变）——由 `test_abba_off_looped_round_keeps_model_only_acceptance` 钉住。锚定模式+ABBA 关现
  在在 CCD 上测候选——由 `test_abba_off_in_anchored_mode_measures_the_candidate_on_ccd` 钉住
  （断言 `entry["acceptance"] == "sequential_ccd"`）。⚠️ **勿"简化"回 model-only，勿在 `acceptance_enabled` True 时加 B1**
  （会改循环时序并破坏上面两条测试）。
- **保持最优相位**：`_run_validate_shapes` 现返回 `last_best`（最后一个形状的 `best`）；会话拆卸不再总写平场
  `zeros((1200,1920))`，而是 best-effort try/except `panel = to_panel(last_best); slm.display_data(...)`
  （显示失败不崩拆卸）；仅 `last_best is None`（全跳过/异常）回退平场并记「SLM: 无最优整形相位...」。
  ⚠️ **关设备勿回退平场**，保持最优整形相位。
- `_run_reused_session` 与 `run_session` 均把 `acceptance_enabled=bool(config.acceptance_enabled)` 透传给 validate 调用
  （驱动默认主导顺序 CCD 路径的开关，2 行）。

## transfer-desktop（`gui/slm/transfer_desktop/`）
- `_render_capture`：有 CCD 时原按兄弟 npz **块级**相位（每块一图）绘，块内 ~10 帧+CCD 冻结；现改为**逐迭代相位**
  （与无 CCD 分支同约定）；同一 npz 只解一次（mtime 缓存），避免每帧重读数十 MB（原为幻灯片式）。
- 新增 `is_own_session_archive(...)` + `QCheckBox` 过滤器：滤到时间戳会话目录 `YYYYMMDD_HHMMSS[-N]`，与文本过滤器
  组合；头部显示本程序会话数。
- **模型权重保存日期**：会话完成状态行追加「模型权重: <path> · 保存于 <mtime>」；优先 `latest/frozen_parameters.json`，
  否则 `model*/fit*/initial` 最新者。
- `_revalidate` 加 `_form_ready` 守卫（构造期 `setValue` 不再读未建控件——修页面构造崩溃）。
- `acceptance_verdict_line`：读 `verdict["lower"]`/`verdict["upper"]`，经新 `_metric_with_shortfall(label, value,
  bound, *, lower_bound)` 渲染——未达标追加「（还差 X · 阈值 Y）」；`lower_bound=True` 用于相关系数（`>=` 下界），
  False 用于相对 L2（`<=` 上界）；不可解析退化为纯值或 `—`。返回 `"验收阈值: <metrics...> · <rounds> 轮 · <tail>"`；
  仅 `acceptance=="looped"` 会话显示（ABBA-only 或纯 model-loss 会话返回 `""`）。新增 `acceptance_progress_line(session_dir)`：
  进行中（未完成）实时从 `rounds.json` 末轮读验收；未开环/无轮/已过 → `""`，永不抛。
- **术语**（summary.py / transfer_progress.py，与口径→阈值/criterion 一致）：`"A–B–B–A"` → `"CCD 配对 ABBA 交错测量 验收"`；
  `"仅模型损失"` → `"模型损失验收（无 CCD 实测）"`；新口径 `"CCD 顺序实测验收"`（sequential_ccd）；`"混合（逐块）"` →
  `"混合口径（逐块）"`。块行先 `候选` 后 `漂移`（有则）否则 `容差`。⚠️ 用 `ABBA 交错测量`/`顺序实测`/`模型损失验收（无 CCD 实测）`
  及"验收/口径"措辞；`A–B–B–A` 与"仅模型损失"在本批文件已退役。

## 新诊断脚本（零相位/相位帽探针）
- `scripts/slm/shape_transition_timing_20261106.py`（660 行）：可微整形切换 square→circle→triangle→square 计时。
  docstring 明示场景「当相位不是预生成的，可微整形算法从"零相位"出发」——**零相位起步** + **带验收的热启动**
  （已接受形状的相位成为下个形状的 `seed_phase`；首形状从零相位网格出发）。CLI：`--checkpoint`、`--out`（必填）、
  `--shapes square,circle,triangle,square`、`--side 400`、`--window-half-span-px 260`、`--cv-threshold 0.10`、
  `--max-iterations 300`、`--max-blocks 200`、`--check-every 10`、`--updates-per-block 10`、`--stall-window-blocks 5`、
  `--stall-min-improve 0.005`、`--log-every-blocks 1`、`--lr 0.01`、`--phase-rms-cap 0.06`、`--carrier-period-px 32`、
  `--carrier-axis x`、`--frames 5`、`--exposure-ms 0.5`、`--settle-s 0.4`、`--warm-start`、`--execute`、
  `--wait-until-idle 0.0`、`--slm-number 1`、`--cam-id`、`--camera daheng`。用 `capture.to_panel`；轮询设备空闲
  （逻辑同 `run_multishape_capability.py` 159–170 行）。
- `scripts/slm/square_grow_diff_sweep_20261106.py`（1184 行）：增方扫描 100→1200 px（+100/边），`--warm-start`
  （仅首边播种，后续边热启动）、`--no-seed-transport`（默认 `seed="transport_square" if seed_transport else "zeros"`）、
  `step_rms_cap`；docstring 引用观察 **lr 0.05 / cap 0.1 rad：边 450 从 100 px 热启动达 cv 0.42，而从新 transport
  种子仅 0.009**——热启动承载平台。窗口跨度跟随目标，「由光路实际支持的上限封顶，绝不用硬编码数字」。
  CLI：`--checkpoint`、`--out`、`--side-min`、`--objective`、`--optimise`、`--device auto`、`--camera miicam` 等。

## 与记忆事实「Cap不是根因：零相位起步梯度下降困在焦斑盆地」的交叉
`slm_square_shaping.py` 全文无 焦斑/盆地/零相位/basin/zero_phase（grep 无命中），且无 11-06 提交朝该问题改源码。
零相位起步/相位帽问题由**上面两个新诊断脚本**探针（`shape_transition_timing` 显式零相位起步 + `--phase-rms-cap`；
`square_grow_diff_sweep` 在帽下比热启动 vs transport 种子）。下一步若要定「Cap vs 零相位盆地」根因，在脚本领地，
不在 `slm_square_shaping.py` 改动里。
