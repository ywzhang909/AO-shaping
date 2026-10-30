# scripts/ 功能二级分类与复用分析报告

> 分析对象：`scripts/` 全部 Python 脚本（288 个 `.py` 文件，剔除 `__pycache__`）
> 分析目标：在现有"按设备域"的一级分类之上，补充"按功能"的二级分类；
> 识别跨脚本的重复功能与可复用函数；给出不改动代码的重构方案。
> 分析方法：静态解析（AST 抽取顶层函数 / import）+ 命名前缀规则 + 内容信号
> （`.md` / `savefig` / `gif` / `csv` / `argparse` 等关键字）自动归类，抽样人工核验。

---

## 1. 现状：一级（设备域）分类

`scripts/` 当前只按设备域做了一级分类。各子目录脚本数量如下（`.py` 含嵌套）：

| 一级目录 | 文件数 | 说明 |
|---|---:|---|
| `slm/` | 155 | SLM 方形/GS 整形、衍射效率、模型闭环等，体量最大 |
| `ml/` | 70 | 机器学习（GSNet、U-Net+GAN、逆向设计、诊断） |
| `rl/` | 11 | 强化学习（SAC / Mamba 湍流） |
| `dm/` | 11 | 变形镜响应矩阵、PIB 参考、Cython 优化器报告 |
| `wfs/` | 4 | 波前传感器交互矩阵、驱动硬件测试 |
| `zernike_bench/` | 6 | Zernike 响应矩阵 / 线性度 / 远场仿真报告 |
| `call_graph/` | 7 | 调用图生成与 wiki 页面（工具） |
| `oopao/` | 4 | OOPAO 仿真后端对比报告 |
| `build/` | 3 | 构建辅助（`_dash_flags`、`setup_pytorch`） |
| `ccd/` | 2 | 相机自动曝光探针 |
| `tuning_devices/` | 2 | 设备调参计算 |
| `signal_processing/` | 2 | 波前重构 demo |
| `dev/` | 1 | 无设备 GIF 生成 |
| `_common/` | 5 | 已抽取的公共辅助（报告生成族） |
| 根目录散件 | 4 | `eval_pib_hybrid_sim.py`、`generate_*_report.py`、`sync_report_provenance.py` |

**问题**：一级分类无法表达"这个脚本到底是干什么的"。例如 `slm/` 下既有
`generate_*_report.py`（产出报告）、`experiment_*.py`（跑实验）、`analyze_*.py`
（分析）、`capture_*.py`（采集）、`fit_*.py`（建模），功能跨度极大，仅靠设备域
无法检索与复用。

---

## 2. 二级（功能）分类方案

依据文件名约定（前缀）+ 内容信号，将全部脚本归入 11 个功能类。该分类与用户
举例直接对应：**生成 MD 报告 → A**、**生成图片/动画 → B/C**、**辅助调整参数 → H/I**。

| 代号 | 功能类 | 定义 / 典型前缀 | 与用户举例对应 |
|---|---|---|---|
| **A** | 报告生成（Markdown） | `generate_*_report`、`write_report`、`build_markdown` | ✅ 生成 md 报告 |
| **B** | 图像 / 图表生成 | `render_*`、`*_figures`、`md_img_diff_*` | ✅ 生成图片 |
| **C** | 动画 / GIF 生成 | `*_gif`、`render_*_animation`、`dev/device_less_gifs` | ✅ 生成图片（动态） |
| **D** | 数据采集 / 探针 | `capture_*`、`collect_*`、`*_probe` | — |
| **E** | 建模 / 拟合 / 训练 | `fit_*`、`refit_*`、`train*`、`inverse_*`、`optimize_*` | — |
| **F** | 分析 / 诊断 | `analyze_*`、`diagnose_*`、`evaluate_*`、`compare_*`、`measure_*`、`*_vs_*`、`ablate_*` | — |
| **G** | 验证 / 校验 / 健康检查 | `verify_*`、`validate_*`、`check_*` | — |
| **H** | 调参 / 标定 / 校准 | `tune_*`、`calib*`、`sweep_*`（参数扫描） | ✅ 辅助调整参数 |
| **I** | 实验编排 / 运行 | `run_*`、`experiment_*`、`demo_*`、`explore_*`、`replay_*`、`model_in_loop_*` | ✅ 辅助调整参数（实验层） |
| **J** | 计划 / 汇总 / 归档 | `summarize_*`、`archive_*`、`audit_*`、`reorganize_*`、`build_*`（计划/图） | — |
| **K** | 工具 / 基础设施 | `call_graph/*`、`build/*`、`_common/*`、清单/溯源 | — |

> 注：`sweep_*`（参数扫描）兼具"辅助调参"与"实验编排"双重属性，本方案归 **H**，
> 在报告中统一视为"参数探索"。`build_*` 按语义拆分：构建计划/图归 **J**，构建
> 工具归 **K**。

---

## 3. 设备 × 功能 分布矩阵

下表为自动归类结果（行=设备域，列=功能类），合计 276 个被归类文件（另有 12 个
根目录散件单列）。该矩阵即二级分类的全局视图。

| 设备域 | A报告 | B图像 | C·GIF | D采集 | E建模 | F分析 | G校验 | H调参 | I实验 | J汇总 | K工具 | Z其他 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `_common` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 4 |
| `build` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `call_graph` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 0 | 3 | 7 |
| `ccd` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `dev` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `dm` | 2 | 3 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 10 |
| `ml` | 10 | 0 | 0 | 2 | 14 | 38 | 1 | 0 | 0 | 4 | 0 | 0 | 69 |
| `oopao` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `rl` | 3 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 3 | 10 |
| `signal_processing` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 |
| `slm` | 34 | 3 | 17 | 15 | 7 | 31 | 8 | 2 | 19 | 9 | 0 | 9 | 154 |
| `wfs` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 3 |
| `zernike_bench` | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| **合计** | **58** | **6** | **19** | **20** | **22** | **72** | **9** | **3** | **25** | **17** | **2** | **23** | **276** |

**关键观察**：
- `F 分析/诊断`（72）与 `A 报告生成`（58）是绝对主力，且高度集中在 `slm/`（31+34）
  与 `ml/`（38+10）。
- `slm/` 是功能最混杂的目录：报告、GIF、采集、建模、分析、校验、实验、汇总八类俱全。
- `Z 其他`（23）多为共享片段模块（`slm/_fwdmodel_sections.py`、`_hw_defocus_sections.py`、
  `_common/provenance.py` 等）与少量独立实验/工具，属跨类辅助件，不强行归类。

---

## 4. 各类代表脚本（节选）

- **A 报告生成**：`slm/generate_slm_zernike_shaping_report.py`、`ml/generate_inverse_design_report.py`、`zernike_bench/generate_zernike_response_matrix_report.py`、`dm/generate_dm_response_matrix_report.py`
- **B 图像/图表**：`slm/generate_multishape_reproduction_figures.py`、`dm/md_img_diff_overlay.py`、`dm/md_img_diff_centroid.py`
- **C 动画/GIF**：`slm/generate_gif_from_pkl.py`、`slm/generate_slm_gsnet_sim_gif.py`、`slm/render_differentiable_opt_animation.py`
- **D 数据采集**：`slm/capture_prior_phase_dataset.py`、`slm/capture_hdr_metrics.py`、`ml/probe_device_metadata.py`
- **E 建模/拟合**：`slm/fit_defocus_forward_dataset.py`、`ml/forward_search.py`、`slm/refit_model_after_optics_change.py`
- **F 分析/诊断**：`slm/compare_sim_vs_hw.py`、`ml/diagnose_snr_ceiling.py`、`slm/analyze_holdout_ceiling.py`、`ml/freeform_vs_zernike.py`
- **G 验证/校验**：`slm/verify_focus_plane_hw.py`、`slm/check_flat_stability.py`、`wfs/wfs_driver_hw_test.py`
- **H 调参/标定**：`dm/tune_sim_spgd_zernike.py`、`slm/sweep_target_size.py`、`slm/calib_geometry_from_orders.py`
- **I 实验编排**：`slm/experiment_gs_improved.py`、`slm/model_in_loop_hw_ab.py`、`rl/run_long_sac_experiments.py`
- **J 计划/汇总**：`slm/build_hw_sweep_plan.py`、`slm/summarize_multishape_capability.py`、`slm/archive_hw_defocus_round.py`
- **K 工具**：`call_graph/build_call_graph.py`、`build/setup_pytorch.py`、`_common/__init__.py`

---

## 5. 重复功能与可复用函数发现

这是本报告的核心结论：**二级分类暴露出大量"本应共享却在各文件各写一份"的辅助
函数**。虽已有 `_common/` 抽取了报告生成族的部分能力，但覆盖严重不足。

### 5.1 跨文件重复的高频辅助函数（实测计数）

| 重复单元 | 定义文件数 | 是否已抽取 | 说明 |
|---|---:|---|---|
| `write_report` | **16** | 否 | Markdown 报告写出，签名各异 |
| `write_csv` | **11** | 否 | CSV 写出 |
| `build_markdown` | **10** | 否（`_common` 仅有 `markdown_table`） | 报告正文拼接，**签名已漂移**（见 5.3） |
| argparse 脚手架（`build_parser` / `_build_parser` / `parse_args`） | **32** | 否 | 大量近似的 CLI 参数定义（设备路径 / seed / out-dir 等公共项） |
| 运行产物加载（`load_run` / `_load` / `_load_run` / `_read_json` / `_read_csv_rows`） | **≈30** | 否 | 重复读取 PKL/JSON 调试产物 |
| 图像渲染（`render_figures` / `_plot` / `_mpl` / `setup_cjk_font`） | **≈18** | 部分（`_common` 有 `savefig` / `plot_summary_bars`） | 绘图例程与中文字体设置反复重写 |
| ML 统计（`folds_of` / `ridge` / `score` / `_score`） | **≈24** | 否 | 集中在 `ml/`，彼此拷贝 |

### 5.2 "已抽取却仍本地重定义"的反模式

`_common/__init__.py` 已提供 `plot_summary_bars`，但仍有 **3 个文件**本地重定义同名
函数而非 import：
- `slm/generate_heuristic_pib_report.py`
- `slm/generate_slm_pib_rms_pib_report.py`
- `slm/generate_strehl_benchmark_report.py`

这说明：即便抽取了公共函数，也缺乏机制阻止新的本地副本产生。

### 5.3 签名漂移隐患（与历史事故同类）

抽样对比 3 个 `build_markdown` 的正文，发现**签名已经不一致**：
- `generate_slm_zernike_shaping_report.py`：`(runs, artefacts, out_dir, out_md)`
- `generate_slm_pib_sim_report.py`：`(runs, tags, figs, gifs, out_md)`
- `generate_gs_defocus_sim_report.py`：`(state: dict) -> str`

这正是 `_common` 当初被创建的同源问题——`_common/__init__.py` 文档明确记载：
两副本的 `_fmt` 不一致，曾把 `1e-7` 渲染成 `0.0000`，在已提交报告中把真实测量值
静默压平为 0。**如不统一契约即抽取，会在更多函数上重演该事故。**

### 5.4 共享覆盖面

全仓库 288 个文件中，仅 **53 个**引用了 `_common`。即约 235 个文件完全未使用既有
公共能力，去重潜力巨大。

---

## 6. 修改方案（本阶段不改动代码）

方案遵循"小步、可逆、契约测试驱动"的原则，参考 `_common` 已有的
`tests/ao_shaping/scripts/test_common_helpers.py` 模式。

### 阶段 0 — 建立 scripts 索引（可选，低风险）
- 由自动归类结果生成 `scripts/_common/scripts_index.json`（设备 × 功能 × 文件），
  作为检索与后续迁移的单一事实源。无需改动任何业务脚本。

### 阶段 1 — 扩展 `_common` 公共函数库（核心去重）
按"先定契约、再写测试、后替换"的顺序抽取：
1. **报告写出**：统一 `write_markdown_report()` + `write_csv()`，替代散落的
   `write_report` / `write_csv` / `build_markdown`（先固化 Markdown 表头/时间戳/
   数值格式化契约，复用既有 `fmt_metric`/`markdown_table`）。
2. **CLI 脚手架**：抽取 `build_cli()`，集中设备路径 / `seed` / `out_dir` 等公共参数，
   替代 32 处 argparse 样板。
3. **运行产物加载**：抽取 `load_run_artefacts()`，统一读取 PKL/JSON 调试产物，
   替代 `load_run` / `_load` / `_read_json` 等 ≈30 处。
4. **图像渲染**：抽取 `render_summary_figures()` 并统一 `savefig` / `setup_cjk_font`，
   复用 `_common` 既有实现；清理 5.2 的 3 处本地重定义。
5. **ML 统计**：将 `folds_of` / `ridge` / `score` 抽到 `ml/` 共享模块或 `_common`。

### 阶段 2 — 二级目录重组（推荐"软分类"，不搬迁）
提供三方案对比：

| 方案 | 做法 | 代价 | 推荐度 |
|---|---|---|---|
| 甲：设备下加功能子目录 | `slm/report/`、`slm/analysis/`、`slm/figures/`… | import 路径大量改动，易碎 | 中 |
| **乙：保持现状 + 索引/命名约定（推荐）** | 不搬迁；用 `scripts_index.json` + 文件名前缀约定 + 本分类表文档化 | 几乎零风险、可逆 | **高** |
| 丙：按功能扁平化 | `report/`、`analysis/`、`capture/`… | 破坏设备归属，跨设备同功能难定位 | 低 |

> 推荐**方案乙**：二级分类的价值在于"可检索、可复用"，而非物理搬迁。现有命名
> 前缀（analyze/compare/generate/capture/…）已天然表达功能，只需用索引与文档固化，
> 并在新增脚本时遵守前缀约定即可。

### 阶段 3 — 渐进迁移 + 契约测试
- 每抽取一个公共函数：**先写 contract test**（钉住输出，如 `_common` 的
  `test_common_helpers`），再替换一处验证，最后批量替换。
- 复用 `_common` 已验证流程，避免 5.3 的签名漂移事故。

### 阶段 4 — 清理反模式
- 删除 5.2 的 3 处本地 `plot_summary_bars` 重定义，改为 `from _common import`。

### 阶段 5 — 合规复查
- 每次改动后运行 `python3 scripts/_common/term_unify_20261016.py`（dry-run 须全绿），
  确保中文叙述术语合规；报告新增文件同样适用。

---

## 7. 风险与注意事项

1. **行为契约优先**：`build_markdown` 等签名已漂移，抽取前必须先统一契约并钉住
   历史报告输出，否则会改变已提交报告的内容（尤其是数值格式化行为）。
2. **import 路径**：`scripts` 以 `from _common import ...` 工作；跨子目录相对导入
   需保持，批量替换时逐处核对。
3. **报告可复现性**：抽取后既有 golden 报告应逐字节一致，建议用 `sync_report_provenance.py`
   类的溯源机制比对。
4. **禁止一次性大改**：288 文件体量，应分批、带测试、可回滚；优先处理重复度最高、
   风险最低的 `write_csv` / `write_report`。
5. **术语门禁**：新增/改动的中文叙述受 `term_unify` 约束，须遵守术语规范表。

---

## 8. 附录

- 全量（设备 × 功能 × 文件）映射数据见同目录 `scripts_functional_map.json`（**冻结快照**，2026-10-26 一次性分类数据，无生成器、不随后续删除同步更新；其提及不构成「仍被引用」的依据，活登记表仅 `scripts/README.md`）
  （自动生成，未纳入术语扫描）。
- 自动归类脚本逻辑：命名前缀优先 + 内容信号兜底；`Z 其他`（23）为共享片段模块
  与独立实验/工具，未强行归类。
- 本报告的二级分类与去重发现，可直接作为阶段 1~4 的任务清单。
