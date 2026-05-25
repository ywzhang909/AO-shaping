# AO-Shaping 项目问题报告

> 生成时间: 2026-05-25
> 扫描范围: `src/ao_shaping/` 全目录依赖分析 + 代码规范检查

---

## 1. 依赖边界违规

### 1.1 `utils/` 反向依赖（应作为叶子模块）

`AGENTS.md` 规定 `utils/` 应当为叶子模块，即不应反向依赖 `algorithm/`、`drivers/`、`optimizer/` 等高阶包，但以下文件存在违规：

| 文件 | 违规导入 |
|------|----------|
| `src/ao_shaping/utils/wfs_utils.py:13` | `from ao_shaping.drivers.wfs.thorlab_wfs import WFSManager` |
| `src/ao_shaping/utils/gs_visualization.py:296` | `from ao_shaping.algorithm.gerchberg_saxton import ...` |
| `src/ao_shaping/utils/pattern_helper.py:8` | `from ao_shaping.algorithm.phase_wrap import PhaseWrapOptimizer` |

### 1.2 `config.py` 依赖具体硬件

`src/ao_shaping/config.py` 在 `_resolve_dm_n_actuators()` 和 `_resolve_disabled_actuators()` 中直接 `from ao_shaping.drivers.dm.NLight import NLight`。配置模块不应依赖具体硬件驱动实现，属于架构层面的脆弱点。更合理的做法是从 `drivers/dm/base.py` 的通用接口或设备注册表中取值。

---

## 2. 相对导入违规

`AGENTS.md` 要求包内使用绝对导入（`from ao_shaping.xxx import yyy`），以下文件使用了相对导入（`from .xxx import yyy`）：

| 文件 | 示例 |
|------|------|
| `drivers/__init__.py` | `from .wfs.thorlab_wfs import WFSManager as Thorlab_WFS` |
| `drivers/slm/__init__.py` | `from .santec_slm200 import SantecSLM200` |
| `drivers/slm/santec_slm200_visa.py` | `from .santec_slm200 import ...` / `from ..visa_base import ...` |
| `drivers/tm/__init__.py` | `from .serial_port_fsm import SerialPortFSM as TM` |
| `drivers/wfs/thorlab_wfs.py` | `from ._thorlab_wfs import ...` |
| `algorithm/__init__.py` | `from .adam_cython import ...` |
| `algorithm/heuristic_base.py` | `from .ga import ...`, `from .pso import ...` |
| `algorithm/random_search.py` | `from .heuristic_base import ...` |
| `algorithm/hill_climbing.py` | `from .heuristic_base import ...` |
| `algorithm/cross_entropy.py` | `from .heuristic_base import ...` |
| `algorithm/differential_evolution.py` | `from .heuristic_base import ...` |

共 **12 个文件** 使用了相对导入。

**注意**: `drivers/` 子包特例较多（`__init__.py` 中大量 try/except 回退导入），如果改用绝对导入需要在包拓扑层面保证不会因外部依赖缺失而阻断整个包导入。

---

## 3. 缺少 `__init__.py`

以下子目录缺少 `__init__.py`，导致它们不是合法的 Python 包，无法被 `from` 语句导入：

| 目录 |
|------|
| `src/ao_shaping/tools/` |
| `src/ao_shaping/optimizer/wf/` |
| `src/ao_shaping/optimizer/rl/` |
| `src/ao_shaping/optimizer/wfless/` |
| `src/ao_shaping/gui/` |
| `src/ao_shaping/gui/streamlit_helper/` |
| `src/ao_shaping/drivers/sim/wfs/` |
| `src/ao_shaping/drivers/sim/slm/` |

---

## 4. 日志模块不一致

`AGENTS.md` 规定统一使用 `loguru.logger`，但以下文件仍使用标准库 `logging`：

| 文件 |
|------|
| `src/ao_shaping/drivers/__init__.py`（line 48: `import logging` + line 51: `logging.getLogger(...)`）|
| `src/ao_shaping/drivers/ccd/__init__.py` |
| `src/ao_shaping/drivers/ccd/miicam_driver.py` |
| `src/ao_shaping/drivers/slm/__init__.py` |

---

## 5. 空文件

| 文件 | 大小 |
|------|------|
| `src/ao_shaping/drivers/wfs/__init__.py` | 0 字节 |

---

## 6. 空子目录（仅含 `__pycache__`）

| 目录 |
|------|
| `src/ao_shaping/drivers/sim/wfs/` |
| `src/ao_shaping/drivers/sim/slm/` |

无任何 `.py` 文件，仅有编译缓存残留。

---

## 7. 包结构偏离文档

### 7.1 `src/ml/` 位于包外部

`AGENTS.md` 文档记录的路径为 `src/ao_shaping/ml/`，但实际目录是 `src/ml/`（在 `ao_shaping` 包之外）。

```
文档: src/ao_shaping/ml/
实际: src/ml/
```

### 7.2 `gs_hologram_runner` 已迁移但测试未更新

`gs_hologram_runner.py` 已从 `src/ao_shaping/` 根目录迁移至 `src/ao_shaping/runners/`，但测试文件 `tests/ao_shaping/algorithm/test_gs_runner_shapes.py` 仍通过 `from ao_shaping.gs_hologram_runner import ...` 导入，导致 ImportError。

---

## 8. 预存在配置/基础设施缺失

`AGENTS.md` 中已列出，补录在此方便追踪：

| 缺失项 | 说明 |
|--------|------|
| 无 CI/CD | 无 GitHub Actions 配置 |
| 无 Linter | 无 `ruff` / `mypy` / `flake8` 配置 |
| 无 Pre-commit | 无 `pre-commit-config.yaml` |
| 无 `requirements.txt` | 仅有 `pyproject.toml` 和 `uv.lock` |

---

## 9. 扫描时发现的预存测试失败

**非本任务引入**，但记录下来方便后续修复：

| 测试文件 | 原因 |
|----------|------|
| `tests/ao_shaping/algorithm/test_gs_runner_shapes.py` | `from ao_shaping.gs_hologram_runner import ...` — 模块已迁移路径 |
| `tests/ao_shaping/optimizer/rl/test_turbulence_env.py` | 缺少 `gymnasium` 依赖 |
| `tests/ao_shaping/optimizer/wfless/test_bayes_hyperparam.py` | 缺少 `scikit-optimize` 依赖（`from skopt import gp_minimize`） |
| `libs/micro_drive1300/py/test_dm_control.py` | 包名为 `py` 导致 Import 冲突 |

另外 `tests/ao_shaping/optimizer/wf/test_zernike_response_matrix.py` 有 9 个测试失败，可能与 WFS 硬件有关。

---

## 10. 架构建议（非阻塞）

1. **`config.py` 去耦**: 将 DM 执行器数量解析移到 `drivers/dm/base.py` 或设备注册表，使 config 不依赖具体硬件 SDK。
2. **`utils/` 叶子化**: `wfs_utils.py` 中对 WFS 的依赖可以通过回调注入或由调用方传入已初始化的 WFS 实例来消除。
3. **相对导入清理**: 可在后续 refactor 中逐步替换为绝对导入。`drivers/` 包因 try/except 回退较多，建议最后做。
4. **补全 `__init__.py`**: 尤其是 `optimizer/wf/`、`optimizer/rl/`、`optimizer/wfless/` 这三个子包，当前缺少 `__init__.py` 却已经是导入目标（通过 `from ao_shaping.optimizer.wfless.pib import ...` 等直接导入模块路径来绕过）。
5. **`src/ml/` 迁移**: 将 `src/ml/` 移入 `src/ao_shaping/ml/` 并更新所有引用。
6. **删减空目录**: `drivers/sim/wfs/` 和 `drivers/sim/slm/` 若无用可删除。

---

## 附: 已修复的问题（本次任务）

| 问题 | 修复 |
|------|------|
| `combined_optimizer.py` 死代码（零引用） | 重构输入输出、类型标注、泛化 DM 接口、创建 CLI runner |
| `combined_optimizer.py` 硬编码 NlightDM | 改为 `dm: DM \| None = None` + 注册表回退 |
| `combined_optimizer.py` 无 CLI 入口 | 创建 `runners/combined_runner.py`，注册到两个 `main.py` |
| `combined_optimizer.py` 13 个 Pyright 类型错误 | 修复 `center` 类型收窄、`centroid` 返回类型、`dm_unit_mask` ndarray 赋值等 |
