# Agent 注意事项（2026-11-15 当日修改）

> 按日追加；过期条目迁入 `docs/dev/` 对应专题。
> 当日无新 commit（工作区改动），聚焦 **`data/` 审计与 run 级归档**（`dump/README.md` 记录 +
> `scripts/_common/data_debug_cleanup.py` 豁免 + `report/data_reference_inventory.md` 自动重生成）
> 与 **`libs/pinn-shapre` 第三方库入仓（未跟踪）**。

## data 审计与 run 级归档

- 工具：`scripts/_common/data_reference_inventory.py --tops`（top 级零引用审计）+
  `scripts/_common/data_debug_cleanup.py --apply`（run 级归档）。体积口径：GB=1e9，GiB=2^30。
- 结果：top 级移出 **1 项**（`data/beam_center` → `dump/data/beam_center`，2.10 GB / 56 文件，
  报告零引用且无代码默认落点）；run 级移出 **98 个 run**（7.11 GB / 1 053 文件，
  移动 0 失败、0 跳过、0 权限拒绝，无在飞保护命中）。零引用 top 数 **14 → 13**。

### 路径前缀差异（易错）

- top 级去向 `dump/data/<top>`，run 级去向 `dump/data_debug/<top>/<run>`，**两者不可混用**。
- **跨前缀引用已存在**：`dump/data/beam_center/20261111_run2/plan.json` 的 `dark_frame` 指向
  `data/debug/dark_flat/...`（该 run 本轮已归档至 `dump/data_debug/dark_flat/...`，目标文件在位）。
  两侧现都在 `dump/` 内但前缀不同，**恢复时须按各自原相对路径分别搬回**。

### 覆盖面与盲区

- **top 层散文件是覆盖盲区**：run 级判定只枚举 `data/debug/<top>/<run>` 目录，结构上不可达。
  涉及 `continuous_cone_scan_20261103`、`panel_locate_repro1200`、`local_forward_preflight_20261103`、
  `registration_replay_20261103`、`slm_diagnose_repro1200`、`slm300_shaping_probe`（合计 508.7 MB，
  零引用约 142 MB）。要收需另写「top 层散文件」判定器，**本轮不改既有工具契约**。
- `issue229_structured_probes` 很可能属此类散文件结构，未出现在保留表中。

### 本轮之外的报告引用数据丢失（非本轮所为）

- 24 个 top 有报告引用但现同时不在 `data/debug/` 与 `dump/` 中：23 个 `sim2f_*`
  （`sim2f_fov` 系列、`sim2f_physical_residual`、`sim2f_shape_trials_production_20261102(_supported)`、
  `sim2f_zero_order_diag_20261103_215142` 等）+ `issue229_structured_probes`。
- 归因已核验为「数据被删」而非新增引用（HEAD 版清单 `fc4d3cf3` 时它们均判存在）。
  报告侧 distinct 缺失路径 **851 → 878（+27）**，丢失体积无从量化。
- **纪律：不恢复、不改动 `report/` 正文，仅记录差异。本轮自身未制造任何新悬空引用**
  （99 条移动路径逐条 `git grep -l -F -- <path> -- report` 核验，99/99 零命中）。

## `data_debug_cleanup.py`：PROTECTED_RUNS 标定溯源豁免（新增条目）

- 新增 `data/debug/geometry_calib_20261111/20261111_102317`（0.121 GB）豁免：它是
  `data/slm_geometry_calib_20261111.npz`（6 853 B，mtime 2026-11-11 10:23:24）的**唯一产出 run**
  （run 戳与 npz mtime 吻合）。npz 是白名单同族标定产物留在 `data/`，若原始采集被归档，
  留下的 npz 将**断掉溯源**。
- ⚠️ 新增豁免的纪律：先核验「被保护 run 的产出物是否留在原处、失去溯源是否有代价」，
  在条目注释里写明产出物路径、体积与核验依据。

## 保留项与易混点

- 保留的运行时依赖 top：`.cache`（`exposure_policy.py:259`）、`flatten_zernike`
  （`config.py:154`）、`slm_models`（`SLM_MODELS_DIR`/`AO_SHAPING_WEIGHTS_DIR`）、
  `zernike_calibration`（GUI zernike_response_matrix_ui.py:102,182）。
- **`data/debug/dark_flat` 与运行时暗场缓存无关**：运行时暗场是 `data/calib/dark`
  （`capture_dark_flat_baselines.py:138 DARK_CALIB_DIR`）。
- 移动后 `data/debug` 下留有空 top 目录（`dark_flat`、`optical_path`、`slm_model_in_loop`）：
  **只移动不删除，故不清理**，空目录是预期状态。
- 白名单 top `slm_transfer`（127 子项）按 `SKIP_TOPS` 整体未动。

## `report/data_reference_inventory.md`（自动生成，勿手改）

- 本轮重生成：256 个含 `data/` 引用的报告文件，1 079 条唯一引用，本机缺失 892 条，零引用 top 13。
- 它是 `scripts/_common/data_reference_inventory.py` 的输出，手改会被下次运行覆盖。

## `libs/pinn-shapre` 入仓（未跟踪）

- 新增第三方研究代码 **pinn-shaper**（PINN 光束整形，arXiv 2607.18012，
  含 far/near-field 示例 notebook 与 `pinn_shaper` 包），置于 `libs/`（`libs/` 在 PYTHONPATH 上）。
- ⚠️ `libs/` 约定是「不进 pip 源的厂商代码」（gxipy、micro_drive1300 等）；pinn-shapre 属
  **外部参考/潜在集成源，勿并入 `src/ao_shaping/`**，勿在生产路径随意 `import pinn_shaper`。
- 该目录目前**未跟踪**；提交时按路径逐项 add（并行会话共工作树，勿 `git add -A`）。

## git 状态

- `git status -- data/` 干净；`git ls-files data/` 仍为 4 个跟踪文件
  （`dm_adj.txt`、`dm_unit_adj_m.txt`、`link_to_adj.py`、`1300-5-enriched.csv`），本轮未触碰。
- 受跟踪改动：`report/data_reference_inventory.md`（自动重生成）、
  `scripts/_common/data_debug_cleanup.py`（豁免条目）；`dump/README.md`（本轮记录）。
- 2026-11-01 的 ~109 GB 历史归档仍不可恢复，本轮未再尝试。
