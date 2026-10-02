# `utils/` + `scripts/` + `tools/` 三目录重构方案

> 状态：**方案已定，尚未动手**（本文件不改动任何代码）
> 调研日期：2026-10-01
> 方法：584 个 `.py`（`src/` + `tests/` + `scripts/`）AST 解析，4934 条 import 记录，按 scope
> （module / func / TYPE_CHECKING）标注；Tarjan SCC 环检测（module-only + 含 function-scope 两轮）
> 交叉校验用 grep

**相关文档（勿混淆，三份TODO 各自独立）**

| 文件 | 范围 | 性质 |
|---|---|---|
| `docs/TODO.md` | `optimizer/wfless/slm_zernike_pib.py` | **P0 真 bug**，优先级更高 |
| `src/ao_shaping/tools/slm/TODO.md` | `tools/slm/` 22 个探针脚本 | 本方案的子集，已单独成文 |
| **本文件** | `utils/` + `scripts/` + `tools/`（含非 SLM） | 跨三树的架构级重构 |

---

## 一、架构裁决：**没有单一"共享脊柱"**

> 这是相对上一轮方案的**主要修正**。上一轮把 `tools/` 当消费者层是错的。

### 实测层级（4 层，不是 3 层）

```
utils/          叶子：数学 / IO / 硬件助手        ← tools 导入它 0 次
  ↓
tools/          【中间层】SLM 共享库 + 硬件 CLI     ← runners / gui 导入它
  ↓
runners/ gui/   编排层（main.py 的 18 个命令）
  ↓
scripts/        【包外同级】消费一切（optimizer 23 / drivers 22 / utils 16 / tools 8）
```

| 事实 | 证据 |
|---|---|
| `tools/` 是**中间层** | `runners/slm/zernike_matrix_runner.py:95` 从 `tools.slm.slm_zernike_common` 导入 **14 个符号**、`:111` 导入 `outlier_mask`；`gui/slm/slm_calibration_ui.py:36` 导入 `tools.slm.calibration` |
| `scripts/` 是 `src/` 的**同级** | 49 个脚本中 42 个 import `ao_shaping` |
| **跨树重复只有 1 处** | `_norm_dev` / `device_md` ↔ `tools/slm/slm_scan_analysis.py`，且已被 `scripts/README.md:882` 明确祝福 |
| 包级环 | **0 个**（全包）。function-scope潜在环 2 个，均不在本三树内 |
| `tools/` → `runners/` | **0 次** —— 无 tools↔runners 环 |

### 两条脊柱，各放在其消费者的正确层

| 关注点 | 归属 | 理由 |
|---|---|---|
| 报告/绘图助手<br>`_savefig` / `_markdown_table` / `_fmt` / `iters_to_threshold` / OOPAO 5 个 / convergence 图 | **`scripts/_common/`**（包外私有包） | 与 `utils` 领域**零重叠**；脚本自身目录已在 `sys.path`，**import 它零 bootstrap** —— 一举消灭 42/21 的 bootstrap 分裂；且不在打包产物内，不可能造成分层倒置 |
| CLI 机制<br>`with_params` / `ClickGroup` / `_DelayedCall` / `_collect_click_annotations` | **`utils/io/cli_params.py`**（新建模块，**非**新子包） | 零 `ao_shaping` 导入 = 结构上不可能成环；`utils/io/cli_helpers.py` 已是同类助手之家（`parse_tuple` / `setup_coredumpy`，24 个导入方） |
| `MicroDMParams` dataclass | **`drivers/dm/micro_dm.py`** | 参数属于它所配置的驱动；`create_dm` 注册表已在那儿。避免 tools→runners 反向依赖 |
| 分析层 `_norm_dev` / `device_md` | 留 `tools/slm/slm_scan_analysis.py`，脚本改为 import | 移进 `utils/wavefront/` 等于把 SLM 扫描域知识塞进通用叶子 |
| utils 内部 D1–D5 | **既有 canonical 模块**，不建新子包 | 5 个小函数不值得开第 6 个子包 |

> **注**：上一轮建议的 `utils/cli/params.py` 修正为 `utils/io/cli_params.py`。
> 不建 `ao_shaping/cli/` 顶层包 —— ~150 行 click 管道不足以新建顶层目录 + 打包入口 + AGENTS.md 条目。

---

## 二、🔴 前置阻断项：`utils/` **不是**安全的叶子

```
utils/wavefront/pattern_helper.py:20  →  import aotools   # 无 try/except，硬依赖
utils/__init__.py:17-27                →  eager import 4 个子包
```

**推论：`import ao_shaping.utils` 在未装 `aotools` 的机器上直接失败。**

这条击穿了"把共享层放进 `utils/` 就安全且廉价"的假设。
**必须先**把 `aotools` 改成惰性/受保护导入（步骤 #17），否则任何依赖"`utils/` 是廉价叶子"的设计都站不住。

`utils/__init__.py` eager 拉起：`matplotlib.pyplot`、`pandas`、`scipy.{ndimage,optimize,signal,linalg}`、
**`aotools`**、`ao_shaping.display`、`ao_shaping.model.field`。
✅ 未 eager：`pygame`（22 处全延迟）、`torch`、`cupy`/`PIL`（有保护）。

**另一处易漏**：`utils/image/gs_visualization.py`（390 行）**违反仓库自己的反模式**
"pygame/viz 代码属于 `display/`，不属于 utils 叶子层"。

---

## 三、重复情况盘点

### 3.1 `utils/`（37 模块 / 10,769 行）

| ID | 重复项 | 位置 | 规模 |
|---|---|---|---|
| **D1** | 日期/时间戳格式化 | `io/file.py:86-88` ≡ `io/cli_helpers.py:101-103`（均 `"%Y%m%d_%H%M%S"`，**逐字节相同**）；`cli_helpers.py:96-98` ≡ `file.py:140`+`:167` 内联 | 2+2 份 |
| **D2** | 日期目录创建 | `file.py:91-99` vs `cli_helpers.py:106-118` | 2 份 |
| **D3** | argmax→(x,y) 光斑定位 | `spots_calc.py:191,195,201-204,262`、`hardware_utils.py:181,594`、`beam_metrics.py:442`(canonical)、`resample.py:91`、`target/ccd.py:60,186,456` | **11 处 / 9 份内联** |
| **D4** | max 归一化到 [0,1] | `beam_metrics.py:58-60,84-86`(canonical) vs `target/patterns.py:148`、`target/ccd.py:208-210`+`474-476`、`hardware_utils.py:89` | 6+ 处 |
| **D5** | 等比裁剪 + 双线性重采样 | `image/resample.py:14-52,55-98`(scipy.zoom) vs `target/ccd.py:45-90+`(手写) | 2 套**独立数学** |
| **D6** | 生成器 API 重复 | `ZernikeGenerator` vs `HadamardGenerator` 共 7 个同名成员 | ❌ 不合并（后端数学不同） |

**假阳性（勿误改）**：`matrix_utils.py:56-58` 是薄包装（害得 `__init__.py:132` 被迫改名
`calc_n_zernike_terms_zern`）；`pattern_helper.py:781` 是方法委托 `phase_unwrap.py:309`；
`wavefront_calc.py:21` 已委托。

**孤儿（真死 3 个）**

| 模块 | 证据 |
|---|---|
| `utils/image/resample.py` | 0 导入，README 却仍提及它→ 删模块**并**修 README |
| `utils/slm_utils.py` | 0 导入；docstring 理由"测试重置 `slm_utils._last_slm_slot`"**已过期** |
| `utils/image/gs_visualization.py` | 仅 1 个测试导入（test-only orphan） |

**⚠️ 孤儿陷阱**：70 个 utils/tools 模块中 **26 个零导入，但只有 3 个可删**——
14 个是有文档的 `python -m` 入口，8 个是包 `__init__.py`。
任何 CI 孤儿检测器**必须**把静态导入与 `python -m` 调用点并集，否则会推荐删掉整个 SLM 探针套件。

### 3.2 `scripts/`（63 个顶层 .py / 1.19 MB）

| 项 | 实测 |
|---|---|
| **Agg 纪律** | 43 个 matplotlib 用户**全部**先调 `use("Agg")` —— **零违规，无需处理** |
| `sys.path` bootstrap | 42 有 / 21 无，其中 ~11 个"脆弱"（import `ao_shaping` 却依赖环境变量，pytest/IDE/`python -m` 下会断） |
| CLI 范式分裂 | 30 argparse + **6 个手写 `cli(...)` 无 `--help`**（`generate_pearson_pkl_gif.py:376` 全文无类型标注） |
| CJK 字体 | 44 个输出中文，**17 个缺 rcParams**，其中 ~5 个把中文写进图/markdown → 豆腐块 |
| README 未收录 | **8 个**：`generate_slm_pib_online_report.py` / `generate_slm_pib_rms_pib_report.py` / `generate_beam_shaping_papers_report.py` / `generate_cython_optimizer_report.py` / `repeat_shape_objectives.py` / `explore_delta.py` / `objective_rep_logging.py` / `pyarrow_probe.py` |
| 输出路径违规 | `generate_cython_optimizer_report.py:47` `OUT_DIR = ROOT/"docs"` 污染 docs 根；`generate_centroid_test_visualization.py` 写进 `scripts/reports/`（源码树内） |

> `scripts/README.md` 2099 行 / 105 KB，其 `## Common Patterns`(L2069) 用**散文**描述 6 步 bootstrap
> —— 这节的存在本身就是"该抽成代码"的论据。

**已验证的重复对（带真实代码）**

| 对 | 位置 | 状态 |
|---|---|---|
| `iters_to_threshold` + `format_iters` | `generate_heuristic_pib_report.py:354-362` ≡ `generate_strehl_benchmark_report.py:431-439` | **逐字节相同**，另 +6 个镜像函数 |
| 🔴 **潜在静默 bug** | `generate_strehl_benchmark_report.py:442-444` 硬编码读兄弟产物 `docs/heuristic_pib/summary.csv` | 删除/改名该产物 → **静默降级而非报错** |
| `_savefig` / `_markdown_table` | `generate_fouriergsnet_sim_report.py:125-150` vs `generate_gsnet_offline_report.py:80-109` | 逐字节相同 |
| 🔴 **`_fmt` 已漂移** | 前者用 `math.isnan`（把 1e-7 渲染成 `0.0000`）；后者用 `np.isnan` 且处理 int/极小/极大 | **重复已在腐化，这是具体代价** |
| `find_debug_dirs` | `slm_zernike_shaping` 版解决了 `slm_pib` 版没有的 3 个问题（双 glob / `.pkl` 叶目录过滤 / 去重保序） | 已演化分叉，方向明确 |
| `_norm_dev` / `device_md` | `zernike_linearity:122,135` vs `zernike_response_matrix:453,487` | **正是 AGENTS.md v0.12.0 创建 `slm_scan_analysis` 却没关上的缝** |
| OOPAO 5 个助手 | `_fmt` / `_rel_diff` / `_git` / `collect_provenance` / `oopao_package_dir` | 逐字复制 |

### 3.3 `tools/` 非 SLM 部分

| 发现 | 详情 |
|---|---|
| 🔴 **`tools/micro_dm/__init__.py` 不存在** | 违反 README §9；`find_packages()` 式打包会**静默丢弃**该模块（只活在过期的 `egg-info/SOURCES.txt:201`） |
| 🔴 **10+ 处 `python -m` 文档全错** | docstring 写 `ao_shaping.tools.micro_dm_image_collect`，实际已挪到 `tools/micro_dm/` 下 |
| "~200 行 TCP 重复"假设**不成立** | TCP 全在 `MicroDM.py` 的 `R50Controller`，工具正确委托 `ctrl.power_off_and_close()`。真正重复的是 **17 个 CLI option**（逐项对应 `runner_common.py:1859-1925` 的 `AltVoltageRunnerParams`，`--port` help 字符串完全相同）+ ~150 行会话编排（`_resolve_ips:147-170` 重复 `MicroDM.open()` 的接线图发现） |
| `R50Controller.__enter__/__exit__` | 在 `MicroDM.py:548/559` 存在，**全仓库无一处使用** |
| `train_data_collect.py` 被测试钉住 | `tests/ao_shaping/optimizer/test_spgd_convention.py:32` 的 `MIGRATED` 元组；移文件即 `FileNotFoundError` |
| 🔴 Micro-DM 磁盘布局是**承重**的 | `{out}/{完整IP}/{ip}-{ch:03d}[-{frame:03d}].png` 被 `utils/io/file.py::find_cell_image` 硬编码 + 4 个 `scripts/md_img_*.py` 依赖，**必须逐字节保持** |
| 测试覆盖 | `train_data_collect.py` 与 `micro_dm_image_collect.py` **均 0 测试** |
| 仓库卫生 | `tools/slm/cartographer/test_smoke.py` 测试文件错放在 `src/` 下，永不被收集；`scripts/NuitkaGUI.exe` **25 MB** 二进制（已被 `.gitignore:38` 正确忽略）；`scripts/tuning_devices/stdWavefront/` **66 个 `.txt`（~57 MB）被 git 跟踪** |

### 3.4 承重 API（fan-in，必须保持字节兼容）

符号：`centroid` 16 文件、`zero_order_center` 12、`radius` 11、`noll_to_nm` 8、`generate_zernike_phase` 9、
`flat_gray` 8、`phase_to_slm_grayscale` 8、`open_camera` 7、`parse_zernike_coefficients` 7、
`um_to_waves` 6、`zernike_modes` 6、`smart_zero_order_center` 6、`parse_tuple` 3、`capture_frame` 2、
`coefficients_to_array` 1（仅测试）

模块：`utils/io/file.py` **45**、`utils/wavefront/zernike_calc.py` **44**、`pattern_helper.py` 25、
`cli_helpers.py` 24、`beam_metrics.py` 23、`spots_calc.py` 23、`targets.py` 21、`zernike_utils.py` 17、
`matrix_utils.py` 16、`image/display.py` 13、`tools/slm/slm_bench_probe.py` 11、
`tools/slm/slm_zernike_common.py` 10、`utils/slm/phase_display.py` 10、`tools/slm/slm_scan_analysis.py` 8、
`utils/hardware_utils.py` 8（旧路径 alias）

**🔴 两个符号有两个家**：`flat_gray` 同时在 `utils/slm_phase.py` 与 `tools/slm/slm_zernike_common.py`；
`um_to_waves` 同时从 `utils.wavefront.zernike_utils` 与 `tools.slm.slm_zernike_common` 被导入。
移动任一模块都会**静默撕裂**它们。

**shim 现状**：全仓库只有 2 个 `sys.modules[__name__] = _real` 别名，都在 `utils/`

| shim | 状态 |
|---|---|
| `utils/hardware_utils.py` → `image/hardware_utils.py` | **承重**，8 个旧路径导入方（6 src + 2 test）；身份被 `tests/ao_shaping/utils/test_image_subpackage.py:16-20` 的 `assert legacy is canonical` 锁定。迁移半途（6 旧 / 7 新） |
| `utils/slm_utils.py` → `slm/phase_display.py` | **死**，0 导入，docstring 理由已过期 |

另有 33 个**非 alias** 的 re-export shim。`utils/image/targets.py` fan-in 最高（21），但它是
`import *`，**不是** alias —— 转成 alias 会改变"模块全局赋值是否写穿"的语义。

---

## 四、明确**不做**的 9 件事

1. ❌ **不**为 `ZernikeGenerator`/`HadamardGenerator` 建 ABC —— 后端数学不同（aotools RZern vs scipy hadamard），7 个同名成员是巧合不是重复
2. ❌ **不**把 `_save_frame`(PIL `I;16`) 并进 `save_frame_png`(inferno 8bit) —— 后者对 16 bit 计量数据集**是错的**，两者各自正确
3. ❌ **不**合并 `train_data_collect.py` 与 `tools/slm/phase_capture.py` —— 闭环 AO 历史(pkl) vs 开环相位→图像数据集(JSON)，互补
4. ❌ **不**把 `slm_scan_analysis.py` 移进 `utils/wavefront/` —— 等于把 SLM 扫描域知识塞进通用叶子
5. ❌ **不**新建第 4 个顶层包 `ao_shaping/cli/`
6. ❌ **不**动那 33 个非 alias re-export shim，尤其 `utils/image/targets.py`（21 导入方）
7. ❌ **不**删 `runners/closed_loop.py` / `algorithm/signal_processing/beam_shaping_utils.py` —— 是 AGENTS.md 记载的向后兼容面，属政策决策
8. ❌ **不**统一 `radius`/`zero_order_center`/`smart_zero_order_center` API（fan-in 11-12）—— 收益在 D3 调用点，不在 API
9. ❌ **不**让孤儿检测器**删除**任何东西 —— 永远只 advisory

**外加两条特别强调：**

- 🔴 `tools/slm/__init__.py` 要**清空内容**而非删文件（它是包标记，删掉会破坏包）。且必须先核实那 14 个入口串指向的是**模块**而非**包**
- 🔴 **D5 是唯一会改变数值结果的去重** —— scipy.zoom 与手写双线性产出不同像素值

---

## 五、分阶段步骤（依赖排序）

### 🟢 A 类：纯收益，零行为风险（约 2 天）

- [ ] **#1** `pyproject.toml` 加 `pythonpath = ["src", "scripts"]`
      → 一行配置修好 11 个脆弱脚本在 pytest/IDE 下的导入 · **验证**：全部测试仍绿 · **Quick**
- [ ] **#2** **清空** `tools/slm/__init__.py` 的 eager 再导出（保留 docstring）
      → **验证**：43 个深路径导入方全部仍通过；14 个入口串指向模块 · **Quick**
- [ ] **#3** `utils/slm/phase_display.py:28` → `TYPE_CHECKING`
      → `from __future__ import annotations` 下运行时无需该导入 · **Quick**
- [ ] **#4** 删 `utils/slm_utils.py`（0 导入 + 理由过期）
      → **验证**：全仓库 grep 无引用 · **Quick**
- [ ] **#5** 补 `tools/micro_dm/__init__.py` + 修 10 处 `python -m` 文档路径
      → **验证**：`python -m ao_shaping.tools.micro_dm.micro_dm_image_collect --help` 可跑 · **Quick**
- [ ] **#6** `cartographer/test_smoke.py` 迁到 `tests/`；修 2 处输出路径违规
      → **验证**：pytest 收集到该文件；`docs/` 根目录干净 · **Quick**
- [ ] **#7** 建 `scripts/_common/`，抽出 `iters_to_threshold`/`format_iters`、`_savefig`、`_markdown_table`、OOPAO 5 助手，**`_fmt` 统一取 gsnet 行为**（视为修复而非重构）
      → **验证**：逐文件 `--help` 不变；`_fmt` 输出变化处人工确认 · **Short**
- [ ] **#8** 补 8 个未收录脚本的 README + 17 处 CJK 字体 + 6 个 CLI 转 argparse
      → **验证**：`grep -c "SimHei"` 与 CJK 输出文件数一致 · **Short**
- [ ] **#9** `generate_strehl_benchmark_report.py:442` 硬编码兄弟路径 → CLI 参数
      → **验证**：删掉 `docs/heuristic_pib/summary.csv` 后该报告**明确失败**而非静默降级 · **Quick**
- [ ] **#10** utils **D1/D2/D4** 去重 —— **可证明安全**（两日期格式化器逐字节相同；`beam_metrics` 已是 canonical）
      → **验证**：同一 datetime 下 4 个调用点产出同一字符串 · **Short**
- [ ] **#11** 完成 `utils/hardware_utils.py` → `image/hardware_utils.py` 迁移（剩 6 个 src 用旧路径），然后删 alias
      → **验证**：`test_image_subpackage.py:16-20` 先绿后删 · **Short**
- [ ] **#12** `utils/image/gs_visualization.py` → `display/`（顺带符合仓库自己的反模式）
      → **验证**：pygame 交互不变；utils eager 面缩小 · **Short**
- [ ] **#13** `runner_common.py` 机制 → `utils/io/cli_params.py`；同时把 `runners/__init__.py` 改成真 lazy（兑现它 docstring 已承诺的 `__getattr__`）
      → **验证**：机制半**零 `ao_shaping` 导入**；`--help` 快照逐字节一致 · **Short–Medium**

### 🟡 B 类：需先写特征测试的真重构

- [ ] **#14** `micro_dm_image_collect.py` → `with_params(MicroDMParams)`（新建于 `drivers/dm/micro_dm.py`）；删掉 7 个驱动内部符号导入与手写 `_resolve_ips`；**同 PR 内**改用 `R50Controller.__enter__/__exit__`（否则设备会话风格从 4 种变 5 种）
      → **验证**：17 个 option 名与默认值逐一比对；**必须先钉死磁盘布局**（#22）· **Medium**
- [ ] **#15** utils **D3**（11 处 / 9 份 argmax→xy）—— 最高价值但各站点有全图 vs 开窗坐标、tie-break 差异
      → **验证**：特征测试 fixture **必须含 tie 与不含 tie 两种** · **Medium**
- [ ] **#16** utils **D5** —— 删孤儿 `resample.py` 免费；但统一 `ccd.crop_resize_to_grid` 的手写双线性**会改数值**
      → **验证**：数值特征测试（唯一改变结果的去重）· **Medium**
- [ ] **#17** `pattern_helper.py:20` 的 `aotools` 改惰性/受保护导入（见 §二，这是**前置阻断项**）
      → **验证**：失败模式从 import 期移到调用期 —— 需专门测试 · **Medium**
- [ ] **#18** `utils/__init__.py` 97 个 eager 名字 → PEP 562 `__getattr__` 延迟
      → **绝不删名字**；先 grep `from ao_shaping.utils import X` 模式 · **Medium**
- [ ] **#19** 加约定测试（孤儿 + `python -m` 一致性 + utils 分层守卫），**warn-only + baseline 文件起步**
      → 必须在 #5 之后做，否则 `python -m` 测试会红· **Short**

### 🔴 C 类：缺陷修复，独立 PR（需硬件）

- [ ] **#20** `tools/slm/cartographer/dynamic_compensation.py:266,272,276,383` 反复写 `memory_slot=2` → 固件no-op，LCOS 不刷新 → 硬件实测面板确实换图
- [ ] **#21** `tools/slm/calibration.py:2182` min-max 归一化 → 尺度无关反模式 → 系数 ×1 vs ×4 必须产出**不同**灰度
- [ ] **#22** Micro-DM 磁盘布局固化测试（承重项，`find_cell_image` + 4 个 md_img 脚本依赖）→ 先钉死再改 #14
- [ ] **#23** `utils/image/display.py:19` 从 `io/handler.py` 导入 `Register`，而 AGENTS.md 说反了方向 → 改文档，不要迁移 `Register`

**依赖顺序**：`1 → 2 → 5 → 3/4/6/9/23 → 7 → 8 → 10 → 11 → 12 → 13 → 14 → 19 → 15 → 16 → 17 → 18`，
`20/21/22` 作为并行的硬件轨道。

---

## 六、风险登记册

| 风险 | 后果 | 缓解 |
|---|---|---|
| `utils/__init__.py` 的 97 个 eager 名字是**公开 API** | 删名字 = 静默破坏性变更 | 只用 PEP 562 延迟，**不删**；先 grep `from ao_shaping.utils import X` |
| CI 分层守卫**第一天就会失败** | `utils/image/display.py:15` 是**有意为之**且有注释说明的兼容再导出 | 白名单显式列出这 1-2 行，否则守卫会被直接删掉 |
| `aotools` 惰性化**移动了失败模式** | `generate()` 调用时才失败，而非 import 时 | 专门写测试固定新失败点；注意 aotools 可能无人维护/未 pin |
| "逐字节相同"是**时点测量** | `_fmt` 已证明这些对会腐化；6 个"镜像"绘图函数很可能已不同 | 抽取时**重新验证**，不要信本文档 |
| `generate_strehl_benchmark_report.py` 当前就在静默降级 | 磁盘上已有的报告**可能就是错的** | 合并前**重跑**受影响报告建立正确基线 |
| `scripts/` 只统计了顶层 .py | 若子目录有真实代码，重复计数被低估 | 补一次含子目录的扫描 |
| `scripts/` **不被打包安装** | 放进 `scripts/_common/` 的东西对已安装用户不可用 | 确认无下游消费者从 scripts/ import |
| #14 触及硬件副作用（继电器上下电、电压） | mock 特征测试必须覆盖 | 严格 mock-only 测试；预算 >1 天 |
| `tools/slm/__init__.py` 只能清空不能删 | 若某入口串是 `python -m ao_shaping.tools.slm`（包）则会坏 | 先核实 14 个入口串指向模块 |
| `flat_gray` / `um_to_waves` 各有两个家 | 移动任一模块会**静默撕裂**这两个符号 | #11/#13 移动前后 grep 这两个符号的全部导入方 |

---

## 七、工作量

| 范围 | 工作量 |
|---|---|
| A 类（#1-13） | **~2 天**，几乎零风险，可立即开工 |
| B 类（#14-19） | **~2.5 天**，需先补特征测试 |
| C 类（#20-23） | **~1 天 + 硬件验证**，可并行 |
| **合计** | **~4-6 天** |

---

## 八、待决策

- [ ] **Q1** — 是否接受"**没有统一脊柱**"这个结论？即 `scripts/` 助手抽到包外 `scripts/_common/`，
      `utils/tools/` 的 CLI 机制抽到 `utils/io/cli_params.py`，两者不合并。
      （这是相对上一轮方案的修正）
- [ ] **Q2** — `runners → tools` 那 3 条反向依赖要不要现在动？
      建议**不动文件**，只在 AGENTS.md 加白名单规则（`runners`/`gui` 仅可导入
      `tools.slm.{slm_zernike_common, slm_scan_analysis, calibration}`），冻结现状、防止继续侵蚀。
      仅当 `tools/` 规模翻倍或出现白名单外的导入时，才升级为独立 `ao_shaping/slm_cal/` 包。
- [ ] **Q3** — 缺陷 #20/#21 要不要等 A/B 类做完再修？
      建议独立 PR + 硬件验证，不与重构混在一起（这样重构可证明"行为中性"）。
- [ ] **Q4** — `aotools` 惰性化（#17）是 §二的前置阻断项，是否**提前**到 A 类优先做？
      建议提前：它是"utils 是安全叶子"这一前提的唯一反例，且改动本身很小。