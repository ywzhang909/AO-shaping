# 长函数 / 深嵌套 / 兼容 import 静态扫描 — 重构方案与影响分析（只列方案，不改代码）

> 2026-10-15 全仓静态扫描（`src/` + `scripts/`，AST 级，`_scripts/static_scan.py`）。
> 本 issue 只做**方案 + 影响范围**；已可单文件安全修复的 2 处**已直接修掉**（见文末）。
> 需要跨文件、风险不可控的**全部留给这里**，不擅自改。

## 0. 已修复（单文件、行为保持、有测试覆盖）

- `phase_unwrap.py::unwrap` — 8 分支 `if/elif` 改为 `handlers` 字典分派（6 策略 + 未知回退 iterative）。
- `io/timestamp.py::parse` — 4 分支 `fmt` `if/elif` 改为 `unix_divisors` 查表 + 早退。23 个测试全过。

## 1. 超长函数（>80 行，共 348 个；前 12 个最重）

| 行 | 位置 | 嵌套 | 圈复杂度 |
|---|---|---|---|
| 1219 | `wfless/slm_square_shaping.py::optimize_slm_square` | 6 | 97 |
| 1175 | `wfless/slm_zernike_pib.py::optimize_slm_zernike_pib` | 7 | 125 |
| 947  | `wfless/slm_zernike_shaping.py::optimize_slm_zernike_pib` | 6 | 95 |
| 916  | `scripts/ml/generate_hwdataset_corpus_report.py::build_report` | 3 | 89 |
| 819  | `scripts/slm/generate_bench_probe_report.py::render_report` | 5 | 134 |
| 813  | `wfless/slm_model_in_loop.py::optimize_slm_model_in_loop` | 5 | 67 |
| 616  | `wfless/pib.py::optimize_pib` | **8** | 59 |
| 557  | `scripts/ml/generate_inverse_shaping_model_report.py::build_report` | 2 | 13 |
| 540  | `wfless/model_in_loop_shaping.py::calibrate_bench_geometry_from_sweep` | 4 | 119 |
| 496  | `scripts/ml/generate_zernike_coeff_report.py::main` | 3 | 40 |
| 466  | `wf/rms_by_zernike.py::optimizer_rms_slm` | 5 | 34 |
| 399  | `scripts/ml/generate_inverse_design_report.py::build_report` | 1 | 4 |

**深嵌套 ≥5 层共 68 个**，其中 `pib.py::optimize_pib` 达 **8 层**（非 monolith 中最深）。

### 重构方案（按风险升序）

- **A. 报告生成器 `build_report`/`render`/`main`（纯函数，无硬件）** — 拆成 `_render_header` / `_render_table_i` / `_render_summary` 等小节函数，主函数只做拼装。**影响面小、无副作用、可独立回归**（对比生成前后 markdown 字节）。候选：`generate_hwdataset_corpus_report`、`generate_bench_probe_report`、`generate_inverse_shaping_model_report`、`generate_zernike_coeff_report`、`generate_inverse_design_report`、`generate_pib_bench_report`、`generate_model_in_loop_report`。
- **B. 优化器 monolith（`optimize_slm_*` 四兄弟 + `pib.py`）— 高风险，不建议盲拆**。这些是闭环硬件主循环，含 I/O、回调、状态机、早退。方案：**只抽纯计算子块**（目标评估、学习率调度 `learning_schedule`、残差/门控判定）为独立函数，主循环骨架保留。需逐函数 golden 测试（同输入相位序列，比对迭代轨迹）。`pib.py::optimize_pib` 8 层嵌套优先把 3 层 `for/if` 提为 `_step_*` 辅助。
- **C. `calibrate_bench_geometry_from_sweep`（540 行 cc119）** — 扫掠→拟合→校验三段，可拆三函数；涉及硬件标定，需 `--dry` 回归。

## 2. 兼容 import（`try/except ImportError`）— 已运行时核实

扫描 80+ 处 `try/except ImportError`，逐条在 venv 内 `importlib` 实测：

| 模式 | 判定 | 影响 |
|---|---|---|
| `from sim.digitaltwin … import` | **`sim` 顶层包不存在**（`ModuleNotFoundError`）→ 永远走 fallback，属**死分支** | 4 个文件：`drivers/sim/optics/simulated_slm.py`、`drivers/sim/atmos/screens.py`、`drivers/sim/laser/simulated_laser.py`、`drivers/sim/__init__.py` |
| `from zernike import RZern` | `zernike` **已安装**（真实 PyPI 包）→ **合法保留** | `utils/wavefront/zernike_calc.py`、`zernike_utils.py`、`wfless/sim_spgd.py`、`scripts/dm/optimize_pib_reference.py` |
| `from ao_shaping.drivers.sim._oopao_compat import Source` | `_oopao_compat.py` 存在 → 合法 | `model/field.py` |
| `from PIL import Image` / `matplotlib` / `scipy.optimize` / `torch` / `numba` / `cupy` / `wandb` / `nidaqmx` / `pyarrow` | 均为**可选依赖优雅降级**，属设计意图 → **保留** | 分散多处 |

**唯一需处理**：`sim.digitaltwin` 死分支（`sim` 顶层包已不存在）。
- **方案**：删除这 4 个文件里 `try: from sim.digitaltwin …` 的整个 try 块，直接保留 `except` 里的 fallback 实现。
- **影响范围**：仅 `drivers/sim/*` 仿真路径，无硬件副作用；但仿真测试（`test_*sim*`）需全跑一遍确认 fallback 行为不变。
- **风险**：低-中。若未来真引入 `sim.digitaltwin` 顶层包会失去该加速路径——但当前仓库无此包，且无 `pip`/`setup.py` 声明它，判定为遗留死码。

## 3. 变量 alias / re-export（`__all__`）

对全部 `__init__.py` 的 `__all__` 做了**运行时 `hasattr` 扫**：

- **`__all__` stale 项 = 0**（静态扫描报的 `dm`/`dm/micro` 等命中全是 PEP 562 惰性加载误报，运行时全部可解析）。
- **结论：alias/re-export 层干净，无需改动。**

## 影响范围总表（供排优先级）

| 项 | 类型 | 风险 | 是否单文件 | 建议 |
|---|---|---|---|---|
| `phase_unwrap` / `timestamp` 分派 | 已修 | 低 | 是 | ✅ 已 push `0cd3727d` |
| `sim.digitaltwin` 死分支 | 删死码 | 低-中 | 否（4 文件同族） | 单 PR，跑仿真测试 |
| 报告生成器拆分（A 组 7 个） | 纯函数拆分 | 低 | 各单文件 | 可逐个 PR，字节级回归 |
| `pib.py::optimize_pib` 8 层 | 抽子块 | 中 | 单文件 | 抽 3 个 `_step_*`，golden 测试 |
| `optimize_slm_*` monolith（B 组 4 个） | 抽纯计算 | 高 | 各单文件但跨硬件 | 最后做，需迭代轨迹 golden |
| `calibrate_bench_geometry_from_sweep` | 三段拆分 | 中 | 单文件 | `--dry` 回归 |

**明确不动**（属"设计如此"或跨包架构）：
- 所有 `from ao_shaping.drivers.sim…` / 设备注册 / PEP 562 惰性 —— 是契约，不是坏味道。
- `src/ao_shaping/drivers/sim/**` 的 `sim.digitaltwin` 若你确认未来要接，则**不要删**，改为加 `# dead until sim.digitaltwin is vendored` 注释即可（请拍板）。

## 扫描脚本

- `_scripts/static_scan.py` — 长函数 / 深嵌套 / 圈复杂度 / 兼容 import
- `_scripts/scan_reexports.py` — `__all__` 静态
- `_scripts/sweep_all.py` — `__all__` 运行时 `hasattr`
- `_scripts/verify_compat.py` — 兼容 import 运行时判定
