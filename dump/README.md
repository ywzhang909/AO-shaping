# dump/ — 从 data/ 移出的运行产物归档

`dump/` 与 `data/` 同类，均被 `.gitignore` 忽略（仅本 README 例外），**归档内容不再参与任何报告与代码路径**。
恢复方式：把对应目录搬回原相对路径即可（`dump/<原相对路径>`，例如 `dump/data_debug/optical_path_probe/xxxx`）。

归入规则：判定只看 `report/**/*.{md,json}` 是否引用该路径；只移动不删除；运行时代码依赖（`*_configs`、
`calibration`、`slm_models`、`shape_trials`、`slm_square`、`dm_adj.txt` 等）永不移动，见下方白名单。

工具：`scripts/_common/data_debug_cleanup.py`（run 级判定 + 归档，默认 dry-run，`--apply` 才搬；
重复执行已幂等；默认 12 h **在飞保护**，`--min-age-hours` 可调）。
top 级零引用清单由 `scripts/_common/data_reference_inventory.py --tops` 给出。
（`dump/` 除本 README 外整体被 `.gitignore` 忽略，所以工具源码放在受跟踪的 `scripts/_common/`。）

## 2026-11-08 · data/debug run 级清理

**执行前状态（重要）**：本次开工时 `dump/` 目录整体不存在，`dump/README.md` 也不存在。
`.gitignore:28` 记录的历史归档（2026-11-01 移出，约 109 GB）在本机已被外部删除，且从未进入 git 历史
（`git log --all --diff-filter=A` 无 dump 条目），无从恢复；本次未尝试恢复，只在下方记录该差异。
本节是当前 `dump/` 的唯一内容来源。

体积口径：GB = 十进制（1e9 字节）；GiB = 2^30。两者并列，避免与 PowerShell 的 `1GB`(=GiB) 混淆。
下面的体积是**归档动作完成时的快照**；同期另一会话仍在往 `data/debug/` 写新数据，
所以现在重测 `data/debug` 会比这里的数字大——差额来自并发写入，不是本轮归档。

- `data/debug`（快照）：**227.43 GB / 211.81 GiB → 224.69 GB / 209.26 GiB**（净减 2.55 GiB）
  文件数 43 723 → 43 460（净减 263 = 归档 270 − 并发会话新写 7）
- 归入 `dump/data_debug/`：**79 个 run 目录，2.74 GB / 2.55 GiB，270 个文件**
  （移动失败 0，目标已存在跳过 0）
- `data/` 总量（快照）：230.99 GB / 215.12 GiB

| 原路径（`data/debug/<top>/<run>`） | run 数 | GB |
|---|---|---|
| `data/debug/model_in_loop_hw_collect` | 18 | 0.97 |
| `data/debug/model_in_loop_hw_sweep` | 4 | 1.22 |
| `data/debug/optical_path_probe` | 2 | 0.06 |
| `data/debug/slm300_shaping_probe` | 3 | 0.49 |
| `data/debug/slm_bench_calibration` | 52 | 0.00 |

| 保留的 top（有报告引用） | 保留 run/子目录数 |
|---|---|
| `data/debug/issue230_refeed` | 1 |
| `data/debug/local_forward_cone_20261103_2242` | 3 |
| `data/debug/local_forward_cone_20261103_2242b` | 3 |
| `data/debug/local_forward_cone_dryrun_20261104` | 3 |
| `data/debug/local_forward_cone_full_hw_20261104_1130` | 3 |
| `data/debug/local_forward_cone_hw_20261104_1115` | 3 |
| `data/debug/local_forward_ring_20261103_2230` | 3 |
| `data/debug/local_forward_sim_20261103` | 3 |
| `data/debug/model_in_loop_hw_collect` | 54 |
| `data/debug/model_in_loop_hw_sweep` | 1 |
| `data/debug/optical_path_probe` | 5 |
| `data/debug/sim2f_fov` | 5 |
| `data/debug/sim2f_fov_calib` | 5 |
| `data/debug/sim2f_fov_dryrun_20261102` | 3 |
| `data/debug/sim2f_fov_dryrun_aligned_20261102` | 3 |
| `data/debug/sim2f_fov_dryrun_nominal_20261102` | 3 |
| `data/debug/sim2f_fov_hw_20261102` | 5 |
| `data/debug/sim2f_fov_hw_20261103` | 3 |
| `data/debug/sim2f_fov_hw_20261103_210042` | 4 |
| `data/debug/sim2f_fov_hw_20261103_210641` | 4 |
| `data/debug/sim2f_fov_hw_refine_ring_20261102` | 3 |
| `data/debug/sim2f_fov_hw_refine_ring_20261103_091657` | 3 |
| `data/debug/sim2f_fov_refine_ring_20261103_live2_195020` | 3 |
| `data/debug/sim2f_fov_refine_ring_20261103_live3_195111` | 1 |
| `data/debug/sim2f_fov_refine_ring_20261103_live4_195131` | 3 |
| `data/debug/sim2f_fov_refine_ring_20261103_live_194007` | 3 |
| `data/debug/sim2f_fov_refine_ring_eff_1103_210858` | 3 |
| `data/debug/sim2f_fov_refine_ring_effalign_1103_211304` | 3 |
| `data/debug/sim2f_fov_refine_ring_gsringcentre_1103` | 3 |
| `data/debug/sim2f_fov_refine_ring_raw16_1103_211433` | 3 |
| `data/debug/sim2f_physical_residual` | 1 |
| `data/debug/sim2f_shape_trials_production_20261102` | 1 |
| `data/debug/sim2f_shape_trials_production_20261102_supported` | 1 |
| `data/debug/sim2f_zero_order_diag_20261103_215142` | 1 |
| `data/debug/slm_bench_calibration` | 1 |
| `data/debug/transfer_uncertainty_20261104` | 8 |
| `data/debug/zernike_probe_hardware` | 1 |

### 本轮 top 级零引用条目（`--tops`）及处置

| 条目 | 处置 | 依据 |
|---|---|---|
| `ccd_configs` `dm_configs` `micro_dm_configs` `layouts` `slm_configs` `wfs_configs` `slm_geometry_calib.npz` `dm_adj.txt` | 保留 | 白名单：运行时代码读取 |
| `flatten_zernike` | 保留 | 空目录，但 `config.py:154 zernike_dir` + `zernike_search_runner` 写入 |
| `slm_models` | 保留 | `SLM_MODELS_DIR` / `latest_model.py` / `optical_probe` 读取 latest 模型（0.19 GB） |
| `zernike_calibration` | 保留 | `gui/zernike/zernike_response_matrix_ui.py` 存储目录 |
| `dm_unit_adj_m.txt` | 保留（判断项） | 零报告引用、零代码引用，但它是白名单 `dm_adj.txt` 的唯一生成源；仅 801 B |
| `link_to_adj.py` | 保留（判断项） | 同上：把 `dm_unit_adj_m.txt` 转成 `dm_adj.txt` 的一次性脚本 |

### 异常与差异

- **历史归档丢失（未恢复）**：`dump/` 2026-11-01 的 ~109 GB 归档在本机已被外部删除，git 历史中也不存在。
  未尝试恢复，也未回涨 `data/`。仅记录差异。
- **一次移动后回滚（4 个目录）**：`data/debug/transfer_uncertainty_20261104/committee_[0-3]`
  （0.26 GB）一度被移入 dump，随后回滚到 `data/`。原因：报告
  `report/shape_trials/transfer_uncertainty/transfer_uncertainty_20261104.md` 把
  `data/debug/transfer_uncertainty_20261104/` 声明为**整个实验的数据根**，正文引用了 4 个相位 bootstrap
  委员会成员的拟合结果；run 级判定只看到顶层目录名、匹配不到子目录，属判定盲区。
  该盲区已在 `scripts/_common/data_debug_cleanup.py` 的 `PROTECTED_RUNS` 中显式豁免。
  同类风险：`data/debug/model_in_loop_hw_collect`、`data/debug/optical_path_probe` 也被报告以整目录形式引用
  （`sim_credibility_20261030.md`、`sim2f_bench/*`），但其下被移走的 run 均**逐条**不在引用清单内，故保留移动。
- **在飞保护（后加）**：收尾时另一会话正在 `data/debug/model_in_loop_hw_collect/`、
  `data/debug/model_in_loop_hw_sweep/` 下连跑硬件实验（run 目录每 30 s 新增一个，最新写入距当时仅数分钟）。
  这些 run 未被引用、会被判定为"过期"，但它们**正在跑**——搬走等于毁掉在飞工作。
  因此工具增加了 `--min-age-hours`（默认 12 h）保护：run 内最新文件比该窗口还新即一律不动。
  加保护后重跑，待搬清单为 **0**，与本轮归档结果一致。
- **报告侧悬空引用（既有问题，与本次移动无关）**：重新生成的清单里 865 个唯一路径已不存在，其中 805 个在
  `data/debug/` 下（`alignment_probe`、`gsnet_hybrid`、`differentiable_opt`、`fit_transfer_multishape` 等 top
  整体已不在 `data/debug`），与上述历史归档被删除一致。
- 本次移动的 run 在清单中零悬空（按路径、按 run 名双向核对均为 0）。
- `git ls-files data/` 的 4 个受跟踪文件全部完好，`git status data/` 干净；无 PermissionError。
- **并发写入（非本任务造成）**：另一会话正在 `data/debug/` 下持续跑硬件实验并落盘
  （`slm_transfer` 在工具的跳过名单内，本任务从未触碰；`model_in_loop_hw_collect` /
  `model_in_loop_hw_sweep` 下的新 run 由在飞保护拦下）。收尾时累计新写 231 个文件，
  所以现测 `data/debug` 会比上面的快照大约 0.5 GiB——差额是并发写入，不是归档漏搬。
