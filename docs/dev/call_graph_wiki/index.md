# AO-shaping 调用图开发手册 — 索引

> 从 `main.py` 出发的函数调用图 + 逐函数优化文档（llm-wiki 式互链 Markdown）。
> 数据源：AST 静态调用图 `raw/assets/call_graph_final.json`（`scripts/call_graph/build_call_graph.py` 生成，可重跑）
> + 源码 docstring 逐字摘录。Last updated: 2026-10-13 | 总页数: 48

## 入口 (entry)

| 页面 | 说明 |
|---|---|
| [[entry/main-py]] | `main.py` 仓库根入口（7 行，转交 `cli()`） |
| [[entry/cli]] | `cli()` Click 根命令组：全局选项 + 20 命令注册表（懒加载） |
| [[diagram/call-graph-overview]] | 调用图总览（分层 mermaid 图 + 阅读方法） |

## CLI 命令 (commands) — 20 个

### DM 波前 / PIB 族（NLight DM）
| 页面 | 一句话 |
|---|---|
| [[commands/wf]] | 波前 RMS 优化（DM 电压 + WFS，SPGD） |
| [[commands/pib]] | 轴向光束 PIB 优化（DM 电压 + CCD） |
| [[commands/pipeline]] | 串行流水线：先 WF 后 PIB |
| [[commands/combined]] | AdaMOD 综合 PIB（自适应桶收缩） |

### DM 矩阵标定族
| 页面 | 一句话 |
|---|---|
| [[commands/dm-matrix]] | DM 响应矩阵标定（push-pull + 自动电压优化） |
| [[commands/hadamard-matrix]] | Hadamard 并行扰动矩阵标定 |

### 微驱电压工具族（R50 / AsyncMicroDM）
| 页面 | 一句话 |
|---|---|
| [[commands/alt-voltage]] | R50 交替电压下发 + NI DAQ ADC 同步采集 |
| [[commands/full-voltage]] | AsyncMicroDM 全量单元 0V↔V 交替（asyncio） |

### SLM Zernike 矩阵 / 闭环族
| 页面 | 一句话 |
|---|---|
| [[commands/zernike-matrix]] | SLM Zernike 响应矩阵标定 + 离线矫正自检 |
| [[commands/closed-loop]] | 基于已标矩阵的闭环波前优化 |

### SLM Zernike 优化族
| 页面 | 一句话 |
|---|---|
| [[commands/rms-zernike]] | SLM Zernike RMS 优化（SPGD / 启发式） |
| [[commands/ga-zernike]] | 遗传算法 Zernike 波前优化 |
| [[commands/greedy-zernike]] | 贪婪局部搜索 Zernike 波前优化 |

### SLM 光束整形族
| 页面 | 一句话 |
|---|---|
| [[commands/slm-pib]] | SLM Zernike 整形（`spgd`/`heuristic` 子命令） |
| [[commands/spgd-square]] | SLM 方形远场均匀性整形（SPGD，CV/EE/AR） |
| [[commands/slm-gsnet]] | freeform 方形整形（`spgd`/`heuristic`/`train`） |
| [[commands/slm-gs-refine]] | GS 开环预整形 + 自由相位 SPGD 细化 |
| [[commands/slm-model-in-loop]] | 正向模型闭环方形整形（每轮重拟合像差） |

### SLM 标定 / 诊断工具
| 页面 | 一句话 |
|---|---|
| [[commands/slm-lut]] | SLM 灰度-相位 LUT 标定（±1 级衍射效率反演） |
| [[commands/slm-diagnose]] | SLM 面板硬件自检（逐级定位"不调制光"故障） |

## 骨架函数 (functions) — 逐函数文档

### 优化器 (optimizer/)
| 页面 | 签名 | 位置 |
|---|---|---|
| [[functions/optimizer-rms-dm]] | `optimizer_rms_dm(epochs, wfs_res, …, dm, wfs_type, …)` | `optimizer/wf/rms.py:45` |
| [[functions/optimize-pib-wfless]] | `optimize_pib(center, epochs, r_bucket, …, dm, optimizer_type, …)` | `optimizer/wfless/pib.py:289` |
| [[functions/optimize-pib-combined]] | `optimize_pib(center, epochs, r_bucket, …)` (AdaMOD) | `optimizer/combined_optimizer.py:118` |
| [[functions/optimizer-rms-slm]] | `optimizer_rms_slm(epochs, init_z, lr, …)` (38 参) | `optimizer/wf/rms_by_zernike.py:433` |
| [[functions/optimizer-ga]] | `optimizer_ga(n_generations, population_size, …)` | `optimizer/wf/ga_zernike.py:135` |
| [[functions/optimizer-greedy]] | `optimizer_greedy(epochs, n_init, n_directions, …)` | `optimizer/wf/greedy_zernike.py:73` |
| [[functions/optimize-slm-square]] | `optimize_slm_square(center, epochs, …)` (40 参) | `optimizer/wfless/slm_square_shaping.py:1129` |
| [[functions/spgd-gradient]] | `spgd_gradient(pos_obj, neg_obj, disturb, *, maximize)` | `optimizer/spgd.py:40` |

### 标定内核 (calibration)
| 页面 | 签名 | 位置 |
|---|---|---|
| [[functions/calibrate-dm-matrix]] | `calibrate_dm_response_matrix(dm, wfs, disturb_voltage, …)` | `optimizer/wf/dm_response_matrix.py:491` |
| [[functions/save-dm-matrix]] | `save_dm_response_matrix(…)` | `optimizer/wf/dm_response_matrix.py:826` |
| [[functions/save-zernike-matrix]] | `save_zernike_response_matrix(…)` | `optimizer/wf/zernike_response_matrix.py:617` |

### 驱动 / 注册表 (drivers/)
| 页面 | 签名 | 位置 |
|---|---|---|
| [[functions/resolve-dm]] | `resolve_dm(dm_type, **kwargs)` | `drivers/dm/_registry.py:230` |
| [[functions/resolve-wfs]] | `resolve_wfs(wfs_type, **kwargs)` | `drivers/wfs/_registry.py:147` |
| [[functions/create-camera]] | `create_camera(camera_type, cam_id, exposure_time_ms, **kwargs)` | `drivers/ccd/common.py:221` |

### 算法 (algorithm/)
| 页面 | 签名 | 位置 |
|---|---|---|
| [[functions/run-heuristic-search]] | `run_heuristic_search(algorithm, evaluate, dim, iterations, bounds, …)` | `algorithm/heuristic/search.py:113` |

### CLI / 工具 (utils/)
| 页面 | 说明 | 位置 |
|---|---|---|
| [[functions/with-params]] | `with_params(arg_class, kw_name)` — dataclass→Click options | `utils/cli/params.py:244` |
| [[functions/debug-mode]] | `get_debug_mode()` / `resolve_debug()` — DEBUG 判定 | `utils/io/cli_helpers.py:29` |
| [[functions/path-dirs]] | `gen_date_dir()` / `make_date_dir()` / 时间戳 | `utils/io/file.py:113` |
| [[functions/recorder]] | `Recorder` 类 — 优化过程逐行记录 | `utils/io/file.py:426` |
| [[functions/debug-artifacts]] | `save_optimization_debug_artifacts()` / `save_recorder_debug_artifacts()` | `utils/io/file.py:739` |
| [[functions/zernike-calc]] | `zernike_modes(n)` / `calc_n_zernike_terms(n)` | `utils/wavefront/zernike_calc.py:78` |
| [[functions/spots]] | `centroid()` / `radius()` / `pib_ratio_mask()` | `utils/image/spots_calc.py:234` |
| [[functions/matrix-utils]] | `compute_pinv()` / `compute_lstsq()` / `safe_pinv()` / `effective_cond()` | `utils/wavefront/matrix_utils.py:12` |
| [[functions/slm-lut]] | `build_inverse_lut()` / `invert_depth_scan()` / `depth_pattern()` | `utils/slm/slm_lut.py:43` |
| [[functions/runner-common]] | `runner_common` 共享工具（center 解析 / 配置构建 / 结果保存） | `runners/runner_common.py` |

## 原始数据 (raw)

- `raw/assets/call_graph_final.json` — 完整调用图（193 已解析节点 / 290 边）+ `recovered.json` 恢复映射
- `assets/diagram_layers.svg` / `.png` — 分层调用图
- 生成脚本：`projects/AO-shaping/_scripts/build_call_graph.py`（AST 分析）
  、`finalize_graph.py`（恢复映射合并）、`gen_wiki_pages.py`（页面生成，幂等）

## 使用建议

1. 想跑某命令 → 打开对应 `commands/<name>.md`：看 docstring + 硬件 + 直接调用。
2. 想加新命令 → 读 [[entry/cli]]（注册表 + 懒加载映射 `_LAZY_RUNNERS`/`_IMPORT_ATTRS`），
   参考同族命令页的 runner 结构。
3. 想改某优化器 → 打开 `functions/<...>.md` 看签名 + 谁调用它（入边）+ docstring。
4. 调参 → 每个命令页 docstring 原文含全部 CLI 选项说明（`with_params` 生成的
   `--help` 与之一致）。
5. 排障 → `--debug` 语义见 [[functions/debug-mode]]；调试产物落盘见
   [[functions/debug-artifacts]]；目录/命名规则见 [[functions/path-dirs]]。
