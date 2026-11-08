# Agent 注意事项（2026-11-08 当日修改）

> 按日追加；过期条目迁入 `docs/dev/` 对应专题。
> 当日无新 commit（工作区未提交改动，文件写入 00:00–06:00），聚焦 **defocus 前向模型的
> 注册 / 极性作用域 / 方向门禁**（issue #234/#235/#237）+ `data_debug_cleanup.py` 入库 +
> `report/data_reference_inventory.md` 重写 + `provenance.py` 报告注册。

## 前向模型全局注册（issue #237）：全局亚像素 2D 平移

- `PhysicalForwardFit(..., registration_shift_ccd_px=None, learn_registration=False)`：
  新增可选全局注册 `(dy, dx)`，单位 **CCD px**。`None`=关闭；给定 `(dy, dx)` 且
  `learn_registration=True` 时注册成为 `torch.nn.Parameter`（`lr=.05`，并入 optimizer）；
  **`learn_registration=True` 而 `registration_shift_ccd_px is None` 直接 `raise ValueError`**。
- `fit.registration_value -> tuple[float, float] | None`：序列化友好只读（关闭时 `None`；
  开启时普通/可训练 tensor 均解包为 `(dy, dx)` float 元组）。**旧 checkpoint 缺省视为关闭**。
- `defocus_forward_core.registration_shift_vector(torch, shift, *, dtype=None)`：归一化为
  `(2,)` 张量、**顺序恒为 `(dy, dx)`**；`shift is None` 时 `raise ValueError`（调用前先判 None）。
- `defocus_forward_core.registration_sample_grid(torch, xy, shift)`：按注册平移相机采样网格
  `xy`；仅在注册开启时调用。
- 门禁：`tests/ao_shaping/algorithm/signal_processing/test_defocus_forward_registration.py`。

## 相位极性作用域（issue #235）：SLM 非线性不得作用于冻结光学像差

- `defocus_forward_core.PHASE_SIGN_SCOPES = ("command", "aberration", "aberration_focus",
  "aberration_carrier")`——**本硬件（Santec #1 1064 nm）面板相位以 NEGATIVE 符号进入传播**。
- `scoped_command_phase(total, panel_x, panel_y, *, phase_sign, scope, focus_component=None,
  carrier_component=None)`：`phase_sign >= 0` 时原样返回 `total`；`phase_sign < 0` 时
  `scope == "command"` 返回 `-total`，其余作用域先 `tilt_plane_split(total)` 得 `(plane, residual)`
  再 `plane - residual`，`aberration_focus`/`aberration_carrier` 再分别 `- 2*focus` / `- 2*carrier`。
  ⚠️ **`focus_component`/`carrier_component` 必须已含在 `total` 内，绝不在此相加**；分割恒用
  完整 `total`，绝不用"减掉分量后的 total"。命名分量是"已包含项"不是"待加项"。
- `tilt_plane_split(z, panel_x, panel_y) -> (plane, residual)`：最小二乘仿射平面 `a*x+b*y+c`，
  坐标居中、2x2 法方程良态、保持前导批维、**可微**。
- `PhysicalForwardFit(..., phase_sign=1., phase_sign_scope="command", focus_phase=None,
  carrier_phase=None)`：拟合须在与 `FrozenDefocusForward` 相同的传播约定下优化（`phase_sign` 为负时
  命令相位才翻转；`command` 作用域即精确 `-total`）。
- `scripts/slm/fit_defocus_forward_dataset.py`：`load_focus_components(...)` 返回的 focus 分量
  **被含在录制的 C 支路 `panel_phase` 内**（22 行 C 上 `max|C-U-focus| ~ 7e-7`；U 行近零 focus），
  故只对 C 行、且仅 `aberration_focus` 作用域重新定号，**绝不与 total 相加**；CLI `--phase-sign`
  默认 `-1.0`（本硬件极性 #235）。
- 门禁：`tests/ao_shaping/algorithm/test_physical_forward_phase_order.py`——同参数下
  `PhysicalForwardFit.prediction()` 与 `FrozenDefocusForward.predict()` 必须逐位一致
  （`assert_close(rtol=0, atol=0)`），且非线性响应与静态 defocus 两者同时生效。

## 方向门禁（issue #237）：checkpoint-aware 包装 + 离线守卫

- `scripts/slm/registration_direction_gate_20261108.py`：复用 `phase_scope_direction_20261107.py`
  的统计口径（issue #234 单模屏方向检查，按作用域）做面向 checkpoint 的包装；输入为
  `20261107_focus_scan/focus.json` + `defocus_forward_20261107_focus_r2(_fit)` +
  `issue234_*_screen_20261107` 三组 ladder。
- `scripts/slm/registration_eval_20261108.py`：在归档帧上评估可微 CCD 网格注册（#237）。
- `scripts/slm/phase_scope_direction_20261107.py`：逐作用域有符号方向检查；本硬件面板相位负号
  （#235）；0 阶取 pass1 帧盒质心中值。
- 门禁：`tests/ao_shaping/algorithm/signal_processing/test_registration_direction_gate.py`——
  **故意做小**（单模式、两个 pass1/pass2 对），只证明包装复用 `phase_scope_direction` 统计精确，
  不跑完整 14 对归档门禁；**不得触碰硬件与 `ao_shaping.drivers`**（无驱动导入，缺脚本即 skip）。

## `data_debug_cleanup.py`（新，`scripts/_common/`）

- 把 `data/debug/<top>/<run>` 中"报告不再引用"的 run 归入 `dump/data_debug/`。**默认 dry-run**；
  `--apply` 才真搬；`--manifest out.json` 输出清单；`--min-age-hours` 默认 **12 h**。
- 判定：①`report/**/*.{md,json}` 逐条路径引用（复用 `data_reference_inventory.scan`）；
  ②保守兜底——run 名在任意报告正文出现即视为引用；③代码兜底——run 名在 `src/ scripts/ tests/`
  出现即视为运行依赖；④**在飞保护**——run 内最新文件比 `--min-age-hours` 还新一律不动。
  **宁可漏搬，不可搬错**。
- ⚠️ **已知盲区**：口径 ①只认"逐条路径引用"。若某报告把整目录声明为数据根（例：
  `transfer_uncertainty_20261104.md` 声明 `data/debug/transfer_uncertainty_20261104/` 整目录），
  其下未被逐条引用的子目录会被误判可移动。`PROTECTED_RUNS` 是人工豁免，新增须写明依据报告。
- `SKIP_TOPS = {"slm_transfer"}`：该 top 下的 run 永不搬。

## FFT 网格可达窗口（多形状目标）

- `multishape_objective.py` 新增 `fft_max_side_px` / `min_calibrated_grid_px` /
  `carrier_offset_px` / `capture_grid_ladder` / `target_extent_px` / `required_capture_grid`：
  **大目标受 `capture_grid` 限制，非 `fft_size`**——目标超过前向模型 FFT 网格可达窗口须拒绝
  （transfer 侧已加 `--reach` 守卫；本批新增离线纯函数计算）。
- `full_frame.py`（sim2f）新增 `box_centroid` / `_half_max_box`。

## 报告与溯源

- `scripts/_common/provenance.py::REPORTS` 新增多份 2026-11-07/08 报告注册
  （forward_model_polarity / loss_compare_20261107 / gated_recurrent_20261107 /
  phase_scope_20261107）；**新增/移动报告后须同步三处**（provenance + 子目录 + 索引）。
- `report/data_reference_inventory.md` 大幅重写（约 3.1k 行变动）：`data/debug` 引用清单格式变更；
  任何消费方（含 `data_debug_cleanup.py`）以其 `scan` 为准。
