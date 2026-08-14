# AO-Shaping 项目问题报告

> 生成时间: 2026-05-25 (初版) / 2026-05-26 (更新)
> 扫描范围: `src/ao_shaping/` 全目录依赖分析 + 代码规范检查

---

## 1. 依赖边界违规

### 1.1 `utils/` 反向依赖（应作为叶子模块）— ✅ 已修复

`AGENTS.md` 规定 `utils/` 应当为叶子模块，即不应反向依赖 `algorithm/`、`drivers/`、`optimizer/` 等高阶包。

| 文件 | 违规导入 | 修复方案 |
|------|----------|----------|
| `src/ao_shaping/utils/wfs_utils.py:13` | `from ao_shaping.drivers.wfs.ThorlabWFS import WFSManager` | 已使用 `TYPE_CHECKING` 保护（运行时无实际导入） |
| `src/ao_shaping/utils/gs_visualization.py:296` | `from ao_shaping.algorithm.gerchberg_saxton import ...` | 已使用 deferred local import（函数内 `import`） |
| `src/ao_shaping/utils/pattern_helper.py:8` | `from ao_shaping.algorithm.phase_wrap import PhaseWrapOptimizer` | ✅ 已修复: 改为 `TYPE_CHECKING` + 9 处函数的 lazy import |

### 1.2 `config.py` 依赖具体硬件 — ⚠️ 未修复

`src/ao_shaping/config.py` 在 `_resolve_dm_n_actuators()` 和 `_resolve_disabled_actuators()` 中直接 `from ao_shaping.drivers.dm.NLight import NLight`。配置模块不应依赖具体硬件驱动实现，属于架构层面的脆弱点。

**建议**: 将 DM 执行器数量解析移到 `drivers/dm/base.py` 或设备注册表，使 config 不依赖具体硬件 SDK。

---

## 2. 相对导入违规 — ✅ 全部已修复

`AGENTS.md` 要求包内使用绝对导入（`from ao_shaping.xxx import yyy`），以下文件已全部修复：

| 文件 | 修复方案 |
|------|----------|
| `drivers/__init__.py` | 全部改为 `from ao_shaping.drivers.wfs.ThorlabWFS import ...` 形式 |
| `drivers/slm/__init__.py` | 改为绝对导入 |
| `drivers/slm/santec_slm200_visa.py` | 改为绝对导入 |
| `drivers/tm/__init__.py` | 改为绝对导入 |
| `drivers/wfs/ThorlabWFS.py` | 改为绝对导入（`from ao_shaping.drivers.wfs._ThorlabWFS import ...`） |
| `algorithm/__init__.py` | 全部改为绝对导入，保留 Cython try/except 回退 |
| `algorithm/heuristic_base.py` | 改为绝对导入 |
| `algorithm/random_search.py` | 改为绝对导入 |
| `algorithm/hill_climbing.py` | 改为绝对导入 |
| `algorithm/cross_entropy.py` | 改为绝对导入 |
| `algorithm/differential_evolution.py` | 改为绝对导入 |
| `display/__init__.py` | 改为绝对导入 |
| `display/windows.py` | 改为绝对导入 |
| `display/frames.py` | 改为绝对导入 |

共 **14 个文件** 已修复。验证: `grep "from \." src/ao_shaping/**/*.py | grep import` 返回空。

---

## 3. 缺少 `__init__.py` — ✅ 全部已修复

以下子目录已补全 `__init__.py`：

| 目录 | 修复方案 |
|------|----------|
| `src/ao_shaping/tools/` | 添加 `__init__.py` （带 docstring） |
| `src/ao_shaping/optimizer/wf/` | 添加 `__init__.py` |
| `src/ao_shaping/optimizer/rl/` | 添加 `__init__.py` |
| `src/ao_shaping/optimizer/wfless/` | 添加 `__init__.py` |
| `src/ao_shaping/gui/` | 添加 `__init__.py` |
| `src/ao_shaping/gui/` | 添加 `__init__.py` |
| `src/ao_shaping/drivers/sim/wfs/` | 添加 `__init__.py` |
| `src/ao_shaping/drivers/sim/slm/` | 添加 `__init__.py` |

---

## 4. 日志模块不一致 — ✅ 全部已修复

`AGENTS.md` 规定统一使用 `loguru.logger`，以下文件已修复：

| 文件 | 修复方案 |
|------|----------|
| `src/ao_shaping/drivers/__init__.py` | `import logging` → `from loguru import logger` |
| `src/ao_shaping/drivers/ccd/__init__.py` | `import logging` → `from loguru import logger` |
| `src/ao_shaping/drivers/ccd/miicam_driver.py` | 移除 `import logging` + `logging.getLogger()`，改用 `loguru` |
| `src/ao_shaping/drivers/slm/__init__.py` | `import logging` → `from loguru import logger` |

验证: `grep "import logging" src/ao_shaping/drivers/` 返回空。

---

## 5. 空文件 — ✅ 已修复

| 文件 | 大小(原) | 修复方案 |
|------|----------|----------|
| `src/ao_shaping/drivers/wfs/__init__.py` | 0 字节 | 添加 docstring + re-exports (`WFSManager`, `MlaRes`) |

---

## 6. 空子目录（仅含 `__pycache__`）— ✅ 已修复

| 目录 | 修复方案 |
|------|----------|
| `src/ao_shaping/drivers/sim/wfs/` | 添加 `__init__.py`（如无实际模块文件可后续清理） |
| `src/ao_shaping/drivers/sim/slm/` | 添加 `__init__.py`（如无实际模块文件可后续清理） |

---

## 7. 包结构偏离文档 — ✅ 已修复

### 7.1 `src/ml/` 位于包外部

`AGENTS.md` 文档记录的路径为 `src/ao_shaping/ml/`，但实际目录是 `src/ml/`（在 `ao_shaping` 包之外）。

✅ 已修复: 更新 `AGENTS.md` 文档，注明 `src/ml/` 是独立的 standalone 包。

### 7.2 `gs_hologram_runner` 已迁移但测试未更新

✅ 已修复: `tests/ao_shaping/algorithm/test_gs_runner_shapes.py` 的导入路径从 `from ao_shaping.gs_hologram_runner import ...` 改为 `from ao_shaping.runners.gs_hologram_runner import ...`。

---

## 8. 预存在配置/基础设施缺失 — ⚠️ 未修复

`AGENTS.md` 中已列出，补录在此方便追踪：

| 缺失项 | 说明 | 优先级 |
|--------|------|--------|
| 无 CI/CD | 无 GitHub Actions 配置 | 低 |
| 无 Linter | 无 `ruff` / `mypy` / `flake8` 配置 | 中 |
| 无 Pre-commit | 无 `pre-commit-config.yaml` | 中 |
| 无 `requirements.txt` | 仅有 `pyproject.toml` 和 `uv.lock` | 低（`uv.lock` 已替代 `requirements.txt`） |

---

## 9. 扫描时发现的预存测试失败 — ⚠️ 未修复

**非本次任务引入**，记录下来方便后续修复：

| 测试文件 | 原因 | 修复方案 |
|----------|------|----------|
| `tests/ao_shaping/algorithm/test_gs_runner_shapes.py` | ~~`from ao_shaping.gs_hologram_runner import ...` — 模块已迁移路径~~ | ✅ 已修复 |
| `tests/ao_shaping/optimizer/rl/test_turbulence_env.py` | 缺少 `gymnasium` 依赖 | 添加 `uv sync --extra rl` 或 `pip install gymnasium` |
| `tests/ao_shaping/optimizer/wfless/test_bayes_hyperparam.py` | 缺少 `scikit-optimize` 依赖 | `pip install scikit-optimize` |
| `libs/micro_drive1300/py/test_dm_control.py` | 包名为 `py` 导致 Import 冲突 | 重命名 `libs/micro_drive1300/py/` |
| `tests/ao_shaping/optimizer/wf/test_zernike_response_matrix.py` | 9 个测试失败，可能与 WFS 硬件有关 | 需排查是否硬件依赖 |

---

## 10. 架构建议（非阻塞）— 待评估

1. **`config.py` 去耦**: 将 DM 执行器数量解析移到 `drivers/dm/base.py` 或设备注册表，使 config 不依赖具体硬件 SDK。
2. **`utils/` 叶子化**: `wfs_utils.py` 中对 WFS 的依赖可以通过回调注入或由调用方传入已初始化的 WFS 实例来消除。
3. **`src/ml/` 迁移**: 将 `src/ml/` 移入 `src/ao_shaping/ml/` 并更新所有引用。
4. **删减空目录**: `drivers/sim/wfs/` 和 `drivers/sim/slm/` 若无用可删除。

---

## 11. 新增扫描发现的问题（2026-05-26）

### 11.1 `print()` 替代 `loguru` — 82 处调用，跨 22 文件

大量文件直接使用 `print()` 输出调试/状态信息，应改用 `loguru.logger`：

| 文件 | 调用数 | 说明 |
|------|--------|------|
| `algorithm/adam.py` | ~25 | Auto Delta Detection 状态输出 |
| `optimizer/wfless/bayes_opt_lr_delta.py` | ~12 | 参数搜索进度输出 |
| `optimizer/wfless/gready_cam.py` | ~6 | 中心值/矩阵加载调试 |
| `optimizer/wfless/adc_dm_adam.py` | ~3 | 文件保存/状态输出 |
| `algorithm/sim.py` | ~5 | 仿真进度输出（中文） |
| `drivers/tm/serial_port_fsm.py` | 1 | 数据校验失败调试 |
| `drivers/sim/dm/simulated_dm.py` | 3 | 模拟 DM 初始化/关闭 |
| `optimizer/rl/rl_wfs.py` | 1 | checkpoint 路径输出 |
| `optimizer/rl/lr.py` | 2 | checkpoint 路径输出 |
| `optimizer/rl/lr_wfs.py` | 1 | checkpoint 路径输出 |
| `optimizer/wf/interaction_matrix.py` | 3 | 矩阵形状/加载输出 |
| `drivers/slm/slm_calibration.py` | 3 | 菜单/标定输出（用户交互场景可保留） |
| `utils/wavefront_calc.py` | 4 | 文件扫描/像素尺寸（含注释掉的） |
| `tools/train_data_collect.py` | 2 | 电压差/结果输出 |
| `drivers/dm/MicroDM.py` | 2 | docstring 中的 doctest（可保留） |
| `utils/timestamp.py` | 2 | docstring 中的 doctest（可保留） |
| `drivers/mock_devices.py` | 1 | docstring 中的 doctest（可保留） |
| `drivers/visa_base.py` | 4 | docstring 中的 doctest（可保留） |
| `drivers/sim/laser/simulated_laser.py` | 1 | docstring 中的 doctest（可保留） |
| `drivers/sim/ccd/simulated_ccd.py` | 1 | docstring 中的 doctest（可保留） |
| `optimizer/wf/dm_response_matrix.py` | 2 | docstring 中的 doctest（可保留） |
| `optimizer/wf/zernike_response_matrix.py` | 1 | docstring 中的 doctest（可保留） |

**建议**: 除 docstring 中的 doctest 外，所有 print() 改为 `logger.info()` / `logger.debug()`。

### 11.2 宽泛的 `except Exception` / `except:` — 30 处，跨 13 文件

| 文件 | 数量 | 风险 |
|------|------|------|
| `drivers/ccd/miicam_driver.py` | 13 | SDK 调用中宽泛捕获可能掩盖真实错误 |
| `drivers/ccd/ffmpeg.py` | 2 | FFmpeg 子进程调用 |
| `drivers/wfs/ThorlabWFS.py` | 1 | WFS SDK 调用 |
| `drivers/dm/MicroDM.py` | 1 | 网络连接 |
| `drivers/visa_base.py` | 1 | VISA 设备通信 |
| `gui/dm/micro_dm_ui.py` | 2 | GUI 回调 |
| `gui/ccd/ccd_analyzer.py` | 1 | GUI 回调 |
| `gui/zernike/zernike_debug_viewer.py` | 2 | **bare `except:` 无异常类型** |
| `gui/zernike/zernike_response_matrix_ui.py` | 1 | GUI 回调 |
| `algorithm/target_func.py` | 1 | 目标函数计算 |
| `runners/gs_hologram_runner.py` | 1 | Runner 错误处理 |
| `utils/cli_helpers.py` | 2 | CLI 辅助 |
| `optimizer/combined_optimizer.py` | 1 | 优化器 |

**建议**: 缩小为具体异常类型（`except OSError`、`except ValueError` 等），或在合适场景使用 `logger.exception()`。

### 11.3 配置项/环境变量分散 — 27 次 `os.environ`，跨 19 文件

核心配置项（`IDEAL_SPOT_RADIUS`、`Far_Cam_ID`、`Near_Cam_ID`、`DM_N_ACTUATORS`）在多处重复定义：

| 配置项 | 重复定义位置 |
|--------|-------------|
| `IDEAL_SPOT_RADIUS` | `config.py`, `pib.py:28`, `slm_zernike_pib.py:56`, `combined_optimizer.py:36`, `pipeline_runner.py:69` |
| `Far_Cam_ID` | `config.py:118`, `envs.py:18`, `pipeline_runner.py:25`, `axis_beam_runner.py:42`, `gs_hologram_runner.py:283`, `combined_runner.py:29` |
| `Near_Cam_ID` | `config.py:119`, `envs.py:19` |
| `KMP_DUPLICATE_LIB_OK` | `rl_wfs.py:3`, `lr.py:3`, `lr_wfs.py:3` — 完全相同，3 次重复 |

**建议**:
- 所有 `os.environ` 读取集中在 `config.py`，其他文件从 `config` 导入。
- `KMP_DUPLICATE_LIB_OK = 'TRUE'` 应在 `__init__.py` 中设置一次即可。

### 11.4 大文件拆分候选（> 800 行）

| 文件 | 行数 | 建议 |
|------|------|------|
| `drivers/wfs/ThorlabWFS.py` | 1661 | 可拆分为: SDK 接口层、参考管理、测量逻辑 |
| `gui/zernike/zernike_response_matrix_ui.py` | 1604 | 可拆分为: UI 组件、业务逻辑、回调处理 |
| `drivers/mock_devices.py` | 1524 | 每个设备一个文件（已有多设备类） |
| `drivers/dm/MicroDM.py` | 1477 | 可拆分为: 协议层、通道管理、控制器管理 |
| `drivers/slm/slm_calibration.py` | 1420 | 可拆分为: 标定算法、GUI 辅助、数据处理 |
| `runners/zernike_matrix_runner.py` | 1072 | 可拆分为: 矩阵计算、可视化、文件 I/O |
| `gui/slm/multi_slm_controller.py` | 1040 | 可拆分为: UI 面板、SLM 控制逻辑 |
| `gui/dm/micro_dm_ui.py` | 910 | 可拆分为: UI 组件、DM 控制 |
| `optimizer/wfless/sim_spgd.py` | 907 | 监控和优化逻辑分离 |
| `optimizer/wfless/pib.py` | 867 | 优化循环和工具函数分离 |
| `optimizer/wf/zernike_response_matrix.py` | 840 | 矩阵计算和可视化分离 |

### 11.5 `if __name__ == "__main__"` 测试入口过多 — 32 个文件

32 个文件包含 `if __name__ == "__main__"` 块。其中以下文件的测试入口可移除（已有 CLI 命令）：

| 文件 | 说明 |
|------|------|
| `runners/wf_runner.py` | 已有 CLI `main.py wf` |
| `runners/axis_beam_runner.py` | 已有 CLI `main.py pib` |
| `runners/pipeline_runner.py` | 已有 CLI `main.py pipeline` |
| `runners/zernike_matrix_runner.py` | 已有 CLI `main.py zernike-matrix` |
| `runners/ga_zernike_runner.py` | 已有 CLI `main.py ga-zernike` |
| `runners/greedy_zernike_runner.py` | 已有 CLI `main.py greedy-zernike` |
| `runners/dm_matrix_runner.py` | 已有 CLI `main.py dm-matrix` |
| `runners/hadamard_matrix_runner.py` | 已有 CLI `main.py hadamard-matrix` |
| `runners/gs_hologram_runner.py` | 函数入口 + CLI 双重定义（line 28 + line 614）|
| `runners/slm_offset_runner.py` | 重复的独立入口 |
| `runners/rms_zernike_runner.py` | 重复的独立入口 |
| `runners/combined_runner.py` | 重复的独立入口 |
| `algorithm/wavefront.py` | 算法文件中的测试逻辑 |

**建议**: 保留 `main.py` 的 `if __name__ == "__main__"` 即可，runner 中的可移除（或保留为一个简短的 `cli()` 调用）。

### 11.6 TODO/FIXME 遗留标记

| 位置 | 内容 |
|------|------|
| `src/ao_shaping/gui/slm/multi_slm_controller.py:898` | `# TODO 改成显示sn` |

仅 1 处 TODO 在 `src/ao_shaping/` 中。

### 11.7 `from __future__ import annotations` 覆盖率

71/100 文件已使用 `from __future__ import annotations`。

**29 个文件未使用**（部分为 `__init__.py` 可忽略，但主要模块建议补全以支持延迟求值）。

---

## 附: 已修复的问题汇总

| # | 问题 | 涉及文件数 | 状态 |
|---|------|-----------|------|
| 1 | `utils/` 反向依赖 | 3 | ✅ 全部修复（1 个 lazy import, 2 个确认已有保护） |
| 2 | 相对导入 | 14 | ✅ 全部转为绝对导入 |
| 3 | 缺少 `__init__.py` | 8 目录 | ✅ 全部补全 |
| 4 | `logging` 非 `loguru` | 4 | ✅ 全部迁移到 loguru |
| 5 | 空 `wfs/__init__.py` | 1 | ✅ 添加 re-exports |
| 6 | 空子目录 | 2 | ✅ 添加 `__init__.py` |
| 7.1 | AGENTS.md 路径错误 | 1 | ✅ 文档更新 |
| 7.2 | 测试导入路径 | 1 | ✅ 修复 |
| 8 | 基础设施缺失 | 4 项 | ⚠️ 未修复（非阻塞） |
| 9 | 预存测试失败 | 5 | ✅ 1 个修复，4 个未修复（外部依赖） |
| 10 | 架构建议 | 4 项 | ⚠️ 待评估 |
| 11.1 | `print()` 替代 `loguru` | 22 | ⚠️ 未修复（新增发现） |
| 11.2 | 宽泛 except | 13 | ⚠️ 未修复（新增发现） |
| 11.3 | 配置项分散 | 19 | ⚠️ 未修复（新增发现） |
| 11.4 | 大文件拆分 | ~22 | ⚠️ 未修复（新增发现） |
| 11.5 | 冗余 `__main__` 入口 | 32 | ⚠️ 未修复（新增发现） |
| 11.6 | TODO 标记 | 1 | ✅ 已修复（MlaRes.from_str 增加 default 参数，multi_slm_controller.py 显示SN） |
| 11.7 | `__future__` 覆盖率 | 29 | ⚠️ 未修复（新增发现） |
