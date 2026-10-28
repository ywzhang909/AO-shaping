# 开发指南：编码规范与测试

> 本文件从 README.md 拆出（2026-10-13）。分层契约的速查版见仓库根 [AGENTS.md](../../AGENTS.md)，本文是完整版。
> 要新增设备 / 算法 / 优化器 / CLI 命令 / ML 模型 / GUI 页面，先看
> [`extension_guide.md`](extension_guide.md)（扩展点、最小模板、测试与文档门禁）。

## 编码规范

### 1. Python 版本与导入

- **Python 3.12+** 是必需的（见 `pyproject.toml`）
- **强制使用 `from __future__ import annotations`**：所有模块文件第一行应包含此导入，启用 PEP 604 延迟求值语法
- **导入顺序**（按标准库 → 第三方 → 本地有序分组）：
  ```python
  from __future__ import annotations  # 总在第一行

  import uuid
  from abc import ABC, abstractmethod
  from collections.abc import Callable, Sequence
  from dataclasses import dataclass
  from typing import Any, ClassVar

  import numpy as np

  from loguru import logger

  from ao_shaping.config import DM_N_ACTUATORS
  from ao_shaping.drivers import MIICamera, DahengCamera
  ```

### 2. 绝对导入（项目强制规则）

- **所有包内引用必须使用绝对导入**，禁止 `from .xxx import yyy` 形式的相对导入
- 格式：`from ao_shaping.子包.模块 import 名称`
- 对于 Cython 回退等特殊场景，使用 try/except 包在绝对导入中：
  ```python
  try:
      from ao_shaping.algorithm._adam_cython import Adam  # type: ignore
  except ImportError:
      from ao_shaping.algorithm.adam import Adam
  ```

### 3. 类型注解（Python 3.12+ 语法）

- **所有公开函数/方法必须标注参数和返回类型**
- 使用 `|` 语法替代 `Optional` / `Union`：
  ```python
  # 正确
  def get_parameter(self, name: str) -> float | None:
      ...

  # 错误
  def get_parameter(self, name: str) -> Optional[float]:
      ...
  ```
- 使用 `list[X]` 替代 `List[X]`：
  ```python
  def process(items: list[float]) -> dict[str, int]:
      ...
  ```
- 避免 `Any` 作为逃逸手段——尽可能精确定义类型

### 4. 命名约定

| 元素 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `NLightDM`, `BaseFrame`, `PhaseWrapOptimizer` |
| 函数/方法 | snake_case | `calculate_sharpness`, `get_centroid` |
| 变量 | snake_case | `exposure_time_ms`, `dm_unit_mask` |
| 常量 | SCREAMING_SNAKE | `MAX_VOLTAGE`, `DEFAULT_THRESHOLD` |
| 私有属性/方法 | `_` 前缀 | `_device_id`, `_set_state()` |
| 类型变量 | PascalCase | `T`, `T_co` |

### 5. 日志（必须使用 loguru）

- **禁止使用 `print()` 输出调试/状态信息**——全部使用 `loguru.logger`
- 禁止使用标准库 `import logging` / `logging.getLogger()`
- 使用 loguru 的格式化字符串（惰性求值）：
  ```python
  # 正确 — 惰性求值，日志级别抑制时不格式化
  logger.info("Device {} initialized with {} actuators", device_id, n)

  # 错误 — 非惰性求值，即使不输出也会格式化
  logger.info(f"Device {device_id} initialized with {n} actuators")
  ```
- 日志级别规范：
  - `logger.debug()`：详细调试信息（函数入口/出口、中间变量）
  - `logger.info()`：重要状态变更（设备连接/断开、优化启动/完成）
  - `logger.warning()`：可恢复的异常（设备连接失败但继续、参数超范围）
  - `logger.error()`：不可恢复的异常（设备断开、关键数据缺失）
  - `logger.exception()`：在 `except` 块中记录完整异常堆栈

### 6. 异常处理

- **禁止宽泛的异常捕获**：不使用 `except:` 或 `except Exception:` 而不指定类型
  ```python
  # 正确
  except ConnectionError:
      logger.error("Device connection lost")
  except ValueError as e:
      logger.warning("Invalid parameter: {}", e)

  # 错误 — 掩盖所有错误
  except Exception:
      pass
  ```
- 自定义异常使用 `*Error` 后缀，继承 `Exception`：
  ```python
  class DeviceError(Exception): ...
  class DeviceNotFoundError(DeviceError): ...
  ```
- 资源管理使用上下文管理器（`__enter__` / `__exit__`）

### 7. 配置管理

- **所有 `os.environ` 读取集中在 `config.py`**，其他文件从 `ao_shaping.config` 导入
- 避免在多个文件中重复读取相同的环境变量
- 对于必须使用时再确定的配置（如硬件 ID），通过参数传递而非全局读取
  ```python
  # config.py
  @dataclass
  class Config:
      far_cam_id: int = 0
      near_cam_id: int = 1
      ideal_spot_radius: int = 7

  # 其他文件
  from ao_shaping.config import ao_config
  cam_id = ao_config.far_cam_id
  ```

### 8. DataClass 与 Enum

- 结构化数据使用 `@dataclass`（无继承需求时）或 `@dataclass` + `ABC`（有继承时）
  ```python
  @dataclass
  class DeviceParameter:
      name: str
      value: Any
      value_type: type = float
      min_value: float | None = None
      max_value: float | None = None
  ```
- 状态/类型定义使用 `Enum` 配合 `auto()`：
  ```python
  from enum import Enum, auto

  class DeviceState(Enum):
      UNKNOWN = auto()
      DISCONNECTED = auto()
      READY = auto()
      ERROR = auto()
  ```

### 9. 模块与包结构

- **每个子目录必须包含 `__init__.py`**（即使是空文件或仅 docstring）
- 模块文件行数建议：工具/算法模块 < 500 行，驱动/优化器 < 800 行。超过 800 行应考虑拆分为子模块
- `utils/` 是叶子模块：不能反向依赖 `algorithm/`、`drivers/`、`optimizer/` 等高阶包
  如果必须引用，使用以下模式之一：
  - `TYPE_CHECKING` 保护（仅类型检查时导入）
  - 函数内部的延迟导入（deferred local import）

### 9.1 驱动惰性加载契约（2026-10-01）

`import ao_shaping` 及其任何子包**都不得加载原生 SDK、也不得做磁盘 I/O**。这不是性能优化，而是正确性问题 —— 仓库里已因此真实炸过两次。

**两条独立的惰性**：

| 层级 | 机制 | 位置 |
|------|------|------|
| **包级** | PEP 562 模块 `__getattr__` | `drivers/_lazy.py::install_lazy_attrs` |
| **构造级** | `Device._load_sdk()` / `Device._ensure_sdk()` | `drivers/device_base.py` |

```python
class MyDriver(Device):
    @staticmethod
    def _load_sdk():        # 覆写点; 默认返回 None (仿真设备无 SDK)
        return load_dll()

    @property
    def _lib(self):          # 只读属性 → 所有 self._lib.<fn> 读点零改动
        return self._ensure_sdk()   # 解析一次并缓存
```

红线:
- `install_lazy_attrs` 内部**必须**写 `module_globals[name] = value` 缓存，否则后续直接 `import` 该子模块会重新绑定全局变量，惰性被彻底绕过。
- 惰性名字**绝不能**在模块作用域出现同名赋值 —— 那会遮蔽 `__getattr__` 并退回 eager。
- **禁止在任何类体/模块作用域做 I/O**。仓库**输入**资产 (如 `dm_adj.txt`) 经 `drivers/dm/_adjacency.py::load_adjacency()` 加载，路径锚定包位置；运行**输出** (如 `PATHS.root_dir`) CWD 相对是**正确**的。
- **分层方向单一**: 硬件包不得 import 仿真包。仿真 DM 经同一个 `install_lazy_attrs` 惰性解析，`dm/_registry.py::_ensure_sim_dms_bound()` 保证 `"sim"` 在**首次注册表使用**时绑定。

### 9.2 大文件拆分的落地口径（2026-10-29，Pilot 实战）

§9 的"超过 800 行应考虑拆分"由 `scripts/check_module_size.py --check` 棘轮兜底
（CI `module-size` job 跑；基线在 `scripts/_common/module_size_baseline.json`，
只许降不许涨）。拆分的**顺序**比拆分的**决心**重要 —— 下面这套是
`generate_hw_gs_square_scans_report.py`（5880 → 1983 行，拆成 11 个模块）走通的：

1. **先抽自包含底座，再抽公共小工具，最后按主题搬函数。**
   反过来做必然成环：常量/dataclass 被所有函数引用，小工具（路径转相对、图名加戳、
   回查 CSV 行）被所有下游模块引用。先把它们落到叶子模块，依赖图才是 DAG；
   留在原文件里，每搬一个函数都会把它整包拉走，于是每个新模块都拖一条重复的尾巴。
2. **搬移用原文行切片，不要 `ast.unparse`。** 仓库里的常量带大量解释性注释，
   unparse 会把它们整段丢掉。切片起止**必须含装饰器行**（漏了它 `@dataclass`
   就变孤儿，症状是 `TypeError: X() takes no arguments`），并**连带节点上方的
   `#:` Sphinx 属性文档**（否则原文件留下一堆孤儿注释）。
3. **import 逐名字剪裁，不整句照抄。** `from typing import Any, Sequence` 照抄会把
   没用到的 `Sequence` 一起带进新模块，F401 立刻红。
4. **运行期被 `global` 改写的模块级变量，一个都不许搬。**
   典型是输出目录常量：`main()` 里 `global FIG_DIR; FIG_DIR = ...` 就地改写。
   搬走之后 `global` 只改 facade 自己的绑定，子模块读到的仍是旧值 —— 而且是
   **静默**旧值，图会画到上一次的目录里。做法是改成装进一个可变对象
   （`PATHS = ReportPaths()`），改的是对象字段，所有模块读到同一份。
5. **等价性用逐字节证明，不用"能启动就算过"。** 拆分前/后各跑一次全量运行，
   比对正文 sha256 + 每一张产物的 sha256 + `--check` 幂等退出码。任一处不同都
   说明搬错了，不要靠肉眼。
6. **配一个守卫测试**，钉住四件事：每个兄弟模块**按文件方式**（cwd 在仓库外）
   启动退出 0；facade 逐个名字回查兄弟模块里是否真的定义了它（防漏搬）；
   没有兄弟反向 import facade（防成环）；底座模块是叶子。
   这套守卫在 Pilot 里当场抓到两个 ruff 不报的真缺陷 —— 兄弟模块里
   `from _common import ...` 排在 `sys.path` 自举**之前**（E402 对 `sys.path`
   操作网开一面，只有真按文件方式启动才 `ModuleNotFoundError`）。
7. **拆完立刻收紧棘轮**：`--update-baseline` 只改被拆那一条，然后 `--check` 必须是 0。

### 10. 性能优化模式

- **默认使用 NumPy** 实现数值计算
- 对热点循环，提供 Numba JIT 加速版本：
  ```python
  @numba.njit(cache=True)
  def _calculate_sharpness_numba(img: np.ndarray) -> float:
      ...
  ```
- 对 GPU 加速场景，提供 CuPy 版本并包含降级回退：
  ```python
  try:
      import cupy as cp
      CUPY_AVAILABLE = cp.cuda.is_available()
  except (ImportError, AttributeError):
      CUPY_AVAILABLE = False
  ```
- 避免深层嵌套循环（3+ 层），优先使用向量化操作

### 11. 测试规范

- **测试文件必须可脱机运行**：在 `tests/` 中使用模拟设备（`MockDM`、`SimTurbulenceAOEnv`）
- 需要硬件的测试使用 `pytest.skip("Requires hardware")` 条件跳过
- 测试验证优化器输出字典中是否包含预期的字段（Recorder 模式）
- 禁止删除失败测试——应修复代码而非测试
- 使用 pytest 而非 unittest

## 贡献流程

1. Fork项目
2. 创建功能分支
3. 提交更改 (遵循conventional commits)
4. 发起Pull Request

## 测试

测试**必须在容器里跑**。本机 384 核，数值库按可见核数开线程池，不受限的
`pytest` 会跑满整机（2026-10-21 曾因此把宿主机跑死、需强制断电）。`conftest.py`
会直接拒绝宿主机运行。

```bash
# 运行所有测试
./run_pytest.sh -v

# 运行特定测试文件
./run_pytest.sh tests/ao_shaping/utils/test_spots_calc.py

# 运行测试并查看输出
./run_pytest.sh -- -s

# 运行特定测试函数
./run_pytest.sh tests/ao_shaping/utils/test_spots_calc.py::TestCentroid::test_centroid_uniform
```

---

## 代码风格补充（原 AGENTS.md 外迁，2026-10-27）

> 以下小节原在根 AGENTS.md 的 `Code Style Guidelines` / `ANTI-PATTERNS` / `UNIQUE STYLES` 节；
> 为给 AGENTS.md 瘦身而外迁，内容保持原样。完整编码约定见本文前 11 节。

### 文档字符串 (Docstrings)

### Docstrings

Use docstrings for public APIs (Google style):
```python
def validate(self, value: Any) -> bool:
    """Validate if value is within allowed range.
    
    Args:
        value: The value to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    pass
```

### Hardware Drivers 接口约束

### Hardware Drivers

**Required interface for all drivers:**
```python
class Device(ABC):
    @abstractmethod
    def open(self) -> None:
        """Open connection to the device."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection and release resources."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected and ready."""
        pass
```

## 反模式红线（ANTI-PATTERNS，原 AGENTS.md）

## ANTI-PATTERNS (THIS PROJECT)

| Pattern | Forbidden Because |
|---------|------------------|
| Relative imports in package | Use `from ao_shaping.xxx import yyy` instead of `from .xxx import yyy` |
| `as any`, `@ts-ignore` | Never suppress type errors |
| Empty catch blocks | Always handle exceptions or log |
| Deleting failing tests | Fix the code, not the test |
| `combined_runner.py` with main CLI | Prefer `pipeline_runner.py` for new serial flows; `combined` remains registered as legacy (cli.add_command(combined_run, name="combined")) |
| Passing uint16 grayscale through `create_phase_from_array()` | `create_phase_from_array()` treats input as **radians** (mod 2π → grayscale = rad/2π × 1023). uint16 grayscale values get silently corrupted. Use `np.full((h,w), gray, dtype=np.uint16)` for flat phase or direct grayscale patterns. |
| SLM 相位生成函数自行 `np.mod(phase, 2π)` (2026-09 raw-only 契约) | All phase generators must return **raw unwrapped radians** — the only mod-2π wrap lives in the driver `Santec.create_phase_from_array()` on radian→grayscale conversion (`santec/driver.py` L1382). Generators that self-wrap duplicate the driver contract and hide the true phase. Convert via `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` (hardware) or the pure fallback (offline/tests). Exception: `_zernike_phase_radians` keeps a wrapped output **only** as a test-only reference. |
| Consecutive `write_phase` + `display_memory` to the **same** memory slot | Santec SLM firmware treats `display_memory(slot)` as a no-op when that slot is already being displayed — the LCOS panel does **not** refresh. Consecutive writes must ALWAYS target different slots. Preferred pattern (used by diff-shaping runner): pick a **random slot in 2..125** each write, excluding the currently displayed slot — this also survives process restarts (`get_displayed_memory_number()` before the first write). Older tools rotate a small pool like `itertools.cycle([3,4,5])` — works within one process only. The built-in `display_data()` cycles through all 127 slots. |
| Opening the SLM in DVI mode (`video_mode=1`) for diagnostics | `Santec(..., video_mode=1).open()` can **hang** (observed 120s/300s timeouts), and a hung controller then also hangs memory-mode `open()` until a **physical power cycle**. Never auto-try DVI mode; use memory mode (`video_mode=0`) only. See `tools/slm/panel/diagnose.py` |
| Treating `get_displayed_memory_number` error-code 1 as a fault | In `set_grayscale` mode there is no memory slot being displayed, so `SLM_Ctrl_ReadDS` returns error-code 1 — this is **normal**, not a failure. Slots only exist in memory mode. |
| Re-writing a **cached displayed phase** back through `write_phase` | Cached phases (`get_displayed_phase()`) already contain base overlay + wavefront correction; `write_phase` re-applies both → corrected **twice**. Rewriting shifted phases must go through the driver-level `Santec.apply_shift()` (raw write via `_write_to_memory` + slot rotation + config save), which is the single public entry point for shift-and-redisplay (identical semantics to the multi-SLM controller "应用平移" button). Shift math's only implementation is the static `Santec.shift_phase()`. See `src/ao_shaping/drivers/slm/AGENTS.md`. |
| Assuming the 0-order spot sits at the camera frame center | In the 2f Fourier bench the optical axis (0-order = frame **global maximum**) lands at the camera center only by luck. Observed: frame center (1344,760) vs 0-order spot (1441-1443, 705-706). Always locate 0-order by `argmax`, never by geometry. |
| diff-shaping 硬件闭环挂起（日志止于 `成功打开SLM #1`） | 2026-09-08 实测 (与相机无关): SLM open() 成功后, 首次 `camera.get_numpy_image()` 前无任何日志输出即无限阻塞 (900s 超时被强杀; 分步探针脚本同挂)。`WaitImageV3` 的原生等待由 SDK 内部驱动, 不受 Python 侧超时保护。处置: 强杀后先确认无残留 python 进程 (Get-Process python*)；重跑前 SLM memory 模式 open() 若超过数秒无日志, 对 SLM 控制器物理断电重置 (与 DVI 挂起同一处置)。故障排查与处置记录见 `report/slm/slm_shaping_diff/readme.md`（该条目当时引用的 `diff_shaping_runner.py` 已在 `fbef1926` 删除，仅存回归测试 `tests/ao_shaping/runners/test_diff_shaping_runner.py`；挂起结论本身仍然成立）。 |
| Generating SLM phase via **min-max normalisation** | It **min-max normalises** the phase (`(p-pmin)/(pmax-pmin)*1023`) instead of `mod 2π` radians→grayscale, making the pattern **scale-invariant** (coefficients ×1 and ×4 produce byte-identical patterns; verified `np.array_equal == True`). Always convert radian phase through the SLM driver `slm.create_phase_from_array()` (2π=993 + wavefront correction + LUT). See `README.md`. ⚠️ 历史例子: `PatternHelper._zernike_to_uint16` 与 `ZernikeDM.generate_phase` —— **两者已于 2026-09 修复/删除**, 本行保留为反模式记录。 |
| **`ZernikeDM.generate_phase` min-max normalises the phase** (same anti-pattern, still live) | `drivers/dm/zernike_dm.py:106-129` does `(raw−min)/(max−min)` then `×2π` (rad) / `×max_val` (gray) → output is **scale-invariant**: coefficients ×1 and ×4 give **byte-identical** phases (verified `np.array_equal == True`, PV always 2π). Since `ZernikeSLM.send_zernike` routes through it, **the Zernike coefficient amplitude is uncontrollable** for every consumer (`zernike-matrix`, `rms-zernike`, `ga-zernike`, `greedy-zernike`, `runners/slm/zernike_matrix_runner.py`, `runners/slm/rms_zernike_runner.py`, `gui/zernike/`). Fix: drop the normalisation (treat coefficients as radians) — note this intentionally changes those callers' behaviour. See `report/slm/report2.md` §2.3. |
| **Mixing units across the WFS→SLM correction loop** | The WFS `get_zernike()` returns **µm**; the response matrix must be built in the **same unit** as the correction's `w` (λ). Two real bugs (2026-09-16): matrix built in µm vs `w` in λ → coefficients inflated **1/0.532 = 1.88×**; and the solved coefficients are in **λ (waves)** but `make_phase`/`generate_zernike_polynomial` take **radians** → applied phase shrunk **2π = 6.28×**. Both fixed → closed-loop RMS improvement 13.8% → **42.1%**. Always route through the canonical `utils/wavefront/zernike_utils.py::um_to_waves`（`tools/zernike/common.py:36` 只做 re-export，并注明 canonical 在 zernike_utils；原写法指向的 `tools/slm/slm_zernike_common.py` 已不存在，2026-10-25 实测） and multiply λ→rad by `2π` before `make_phase`. See `report/slm/report2.md` §0. |
| **Full-frame centre leaking into window-local metrics** (`slm_zernike_pib.py`) | `cam.reset_window(...)` returns the window centre in **full-frame sensor coordinates**, but every downstream metric (`rms_pib_terms`, `ImageTargetFunc.radius`, `spot_waist_sigma`, `reference_center`) operates on the **re-windowed** image and needs **window-local** coordinates. Hardware-verified bug (2026-09-23): the runner's `-c auto` probe locates the 0-order on the un-windowed camera (e.g. (633,934) on 1944×2592) → optimizer used it directly as window-local → every metric read **0** (pib=0, r_bucket pinned at 0.9 = 1px radius × shrink 0.9) across the whole heuristic matrix. Fix: after capturing the re-windowed `init_img`, re-locate `reference_center = zero_order_center(init_img)` and use `reference_center` (NOT the post-`reset_window` `center`) for the r-bucket radius too (L1396/L1441). Rule: whenever a centre is handed to a metric/ROI builder, verify which coordinate frame it belongs to first. |
| Square shaping with low-order Zernike (n≤4) | Zernike modes are a **circularly symmetric smooth** basis; they physically cannot synthesise a square far-field (needs 2D-sinc-like near field / high spatial frequencies). Use full-pixel phase freedom (GS / differentiable / free-form), not Zernike. |
| Optimising `-CV` alone as the SPGD objective for square shaping | With no energy term the optimizer **empties the target box** to minimise CV (hardware observed EE→0.002). The objective must include encircled energy (use the combined quality score). |
| Computing `PIB` / `CV` on a **raw** CCD frame (read noise not removed) | A CCD frame carries symmetric read noise, so ~half its pixels are negative (measured 7164/14400 on a sim frame). `power_in_bucket` divides the in-target sum by the **whole-frame** sum, so a negative background drives the ratio **above 1** (measured `PIB=1.0120`) and corrupts `CV` the same way — the optimiser then chases read noise. Subtract the frame median and clip at 0 (`ao_shaping.utils.image.beam_metrics.prepare_frame`); clipping the *raw* frame instead rectifies noise into a pixel-count-sized DC pedestal (see the `clip(·,0,None)` defect in `sim/AGENTS.md`). |
| Re-locating the target ROI by `argmax` on **every** iteration | On a speckle field the global maximum hops between near-equal grains under a ~1e-3 perturbation, so an `argmax`-rolled box makes `PIB`/`CV` **discontinuous** and the optimizer chases a box that no longer covers the beam (documented on `compute_beam_metrics`). Locate the 0-order **once** on the unshaped frame and freeze the ROI for the whole run. |
| Committing a GS warm-start phase to hardware without a bake-off | The GS phase comes from a *model* of the bench (aperture, focal length, camera pixel pitch). If any input is wrong, GS makes the real spot **worse**. Measure flat and GS and keep the better one — a mis-calibrated model then costs two measurements instead of wrecking the run (`slm_gs_refine` Stage 1). |
| Treating an exact run-to-run reproduction as a property of a closed loop | The seed pins the SPGD perturbation signs, **not** the measurements: every far-field read carries device noise. Two same-seed runs disagree at ~1e-3 *at the flat baseline*, before any optimisation. Assert approximate reproducibility, never equality. |
| Trusting `reset_window()`'s returned centre | When the spot is near the frame edge the ROI offset is clamped but the returned `(w//2, h//2)` is not the true spot position → the target box lands off the beam (hardware observed epoch-0 `mean_b=0.01`). Re-locate the spot by `argmax`/centroid on the **windowed** image. |
| Function-only optimizers in ao_shaping/algorithm (no class API) | New optimizers must expose __init__ (validation + state) + update() (one step) + optional run() (result dataclass); one-shot functions are legacy/thin wrappers only. See src/ao_shaping/algorithm/README.md. |
| Placing markdown/report **generation** under `src/ao_shaping/tools/` | **All markdown/illustrated-report generation MUST live in `scripts/`** (naming: `scripts/generate_*_report.py`, e.g. `generate_zernike_wfs_report.py`, `generate_diff_shaping_report.py`). `src/ao_shaping/tools/` is reserved for hardware-interaction tools (CLI + driver orchestration), not report writers. See scripts/README.md. |
| Report generation inside `algorithm/` | `signal_processing/beam_shaping_benchmark` writes CSV/MD/GIF reports — report generation MUST live in `scripts/` (see scripts/README.md), not in the algorithm layer. |
| Pygame/viz code inside `utils/` | Visualization code belongs in `display/`, not the leaf utils layer. **Done for `gs_visualization`** (moved to `display/gs_visualization.py`, 2026-10-03; re-exported from `ao_shaping.display`) **and done for `display.py`** (moved to `display/image_display.py`, 2026-10-09, commit `a9236e9`; `utils/__init__.py` re-exports re-pointed to `ao_shaping.display.image_display`). No compat shim left — the old `ao_shaping.utils.image.display` path now raises `ImportError` (guarded by `test_utils_leaf_no_rendering_import.py::test_old_display_path_raises`). |
| Re-implementing the GS loop | `gs_visualization.gerchberg_saxton_with_visualization` must drive the canonical `gerchberg_saxton` via a callback, not duplicate the loop. |
| Native SDK binding inside `utils/` | `utils/` is the import-light leaf layer — it must not `import ctypes` to reach a device SDK; that binding belongs in `drivers/` (lazy registration via `drivers/_lazy.py`). The old `utils/wavefront/vi.py` (188 lines of ctypes VISA) was deleted in `bbb3820b`, which left the rule as prose only; it is now pinned by `test_conventions.py::test_the_utils_leaf_layer_never_binds_a_native_sdk` (measured baseline 0, no allowlist). |
| Min-max scale-invariant phase conversion | Min-max normalising the phase makes patterns scale-invariant (coefficients ×1 and ×4 → byte-identical). Use `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` instead. ⚠️ 历史例子 `PatternHelper._zernike_to_uint16` 已删除。 |
| Relocating `algorithm/signal_processing/sim.py` into `scripts/` | **Obsolete instruction (verified 2026-10-25)** — the vector-beam demo already lives in `scripts/`: `14b1cda` moved the `__main__` body to `scripts/signal_processing/demo_wavefront_sim.py` (+ `demo_wavefront_reconstruction.py`), leaving `sim.py` as an import-only library (`numpy` / `matplotlib.colors` / `loguru`; no `__main__`, `argparse` or `plt.show`). That script imports 4 symbols from it, so moving the module itself would break the import. The split is already correct: library in `algorithm/`, demo in `scripts/`. |
| `utils/**` 模块级 import `algorithm/`（原例：`pattern_helper` 曾 top-level import `ao_shaping.algorithm.phase_wrap`） | utils is the leaf layer and must not depend on `algorithm/` at import time — use deferred function-local imports. **原例已合规**（2026-10-25 AST 实测：该文件模块级只有 `__future__` / `typing` / `numpy` / `utils.wavefront.phase_unwrap:8` / `utils.wavefront.zernike_calc:13`，`algorithm` 全部延迟到函数内 `:20/:36/:92/:98/:837/:854/:871/:890/:903`）⇒ 由 `test_conventions.py::test_utils_has_no_module_scope_import_of_a_higher_layer` 钉住，其非空性由 `test_the_deferred_forms_still_exist_so_the_guard_is_not_vacuous` 守住（现 14 处延迟形式）。另：本行原写 `utils/slm/pattern_helper.py` 系误标，文件在 `utils/wavefront/`（见上方结构表）。 |
| 脚本/runner/tools/GUI 内**重复实现 Zernike 数学** (Noll↔(n,m) 查表、多项式求值、相位生成) | Canonical 入口唯一: `utils/wavefront/zernike_utils.py` (API 层) + `utils/wavefront/zernike_calc.py` (引擎层), 单向依赖 `zernike_utils → zernike_calc`。2026-09 去重重构前 `gui/slm/`、`optimizer/wfless/`、`tools/slm/` 各有一套, 已全数收敛。新代码一律 `from ao_shaping.utils.wavefront.zernike_utils import ...`, 禁止 `import aotools`/自写 `RZern`/自写 `noll2nm` 表。详见 README `## Zernike 使用指南` 与 AGENTS `## Zernike 使用规范`。 |
| **配对检验只在"变化的轴"上配对** — 拿 N 折当 N 个独立样本 | 全部折共用同一个 `seed` 时, 各折共享**同一条权重初始化随机流**, 于是 N 折是"同一对初始化"被测了 N 次, **不是**该对比的 N 次独立采样; sign-flip 检验仍按独立处理, p 值**偏保守 (过于乐观)**。实测: `input_terms` 136→78 的 18 折配对给出 `p = 0.0144`, 但固定一折只换 seed 后**符号翻转**、差值量级小 30 倍 ⇒ 那是初始化噪声不是效应。规则: **换 seed 就对 seed 配对, 换折就对折配对**, 两者不可互相冒充; 判定前跑 `scripts/ml/seed_sensitivity_coeff.py`。另: `min_attainable_pvalue(n) = 2/2**n`, 故 4 折协议**结构上**到不了 p<0.05, 那里只报效应量不报结论。统计实现唯一入口 `ml/zernike/eval_stats.py` |
| 用 **MSE/PSNR/SSIM** 给 `normalization` 选开关 | 任何含 `sum` 归一化的指标都在度量**归一化本身**而不是拟合质量。实测同一配置: `normalization="sum"` 给 PSNR **72.1 dB** / SSIM 0.9996, 而 R² 反而**更差**于 `peak`。判据只看 **R² / corr / 光斑域指标** |

> 方形光斑 SPGD 整形的完整分析、硬件实测与修复记录见 [`report/slm/slm_square_spgd/README.md`](../../report/slm/slm_square_spgd/README.md)。

## 项目特有风格（UNIQUE STYLES，原 AGENTS.md）

## UNIQUE STYLES

- **Mock-first testing**: Tests use simulation classes (`SimTurbulenceAOEnv`, `sim_spgd`) to avoid hardware
- **Zernike Noll 约定统一** (aotools Noll 1976): Noll 4 = (2,0) defocus, Noll 5 = (2,-2) astig, Noll 11 = (4,0) spherical, Noll 13 = (4,-2)。`optimizer/wf/` 三个优化器 (ga_zernike / greedy_zernike / rms_by_zernike) 的模式枚举与 Noll 索引→(n,m) 映射已全部使用 canonical `zernike_calc.noll_to_nm()` / `zernike_calc.zernike_modes()`; 历史 legacy 硬编码查表 (`noll_to_nm_legacy`, 那里 Noll 5 = (2,0)) 已删除, 不再存在。新代码一律用 `zernike_calc.noll_to_nm()` / `zernike_utils.list_zernike_modes()`, 勿混用两套索引。zernike_utils 模块文档含完整前 15 阶映射表。
- **Zernike 纯数学 canonical 两层入口** (2026-09 去重重构): API 层 `zernike_utils.py` (`parse_zernike_coefficients` / `generate_zernike_phase` / `list_zernike_modes` / `coefficients_to_array` / `um_to_waves`) → 引擎层 `zernike_calc.py` (`ZernikeGenerator` 网格缓存 / `noll_to_nm` / `zernike_modes` / `noll_indices`)。生成器输出 **raw 未包裹弧度**, 弧度→灰度统一走 `phase_display.phase_to_slm_grayscale(phase, slm=slm)`。任何脚本/runner/tools/GUI 禁止自写第二套 Zernike 数学 — 见 `## Zernike 使用规范` 节与 README `## Zernike 使用指南`。
- **Hardware skip pattern**: Tests requiring physical hardware use `pytest.skip("Requires DM hardware")`
- **Recorder pattern**: Optimization tests validate history dictionaries with expected fields
- **Optional backend testing**: CuPy/Numba tested conditionally with try/except guards
- **No fixtures**: No `conftest.py`, fixtures defined inline in test methods
- **SLM flat-phase gray RAW path**: Always send raw uint16 grayscale values to SLM via `np.full((h,w), gray, dtype=np.uint16)`. Never route flat phase through `create_phase_from_array()` (radian conversion). The SLM has amplitude coupling: different flat-phase gray levels produce different camera intensities at 1064nm (periodic with 2π ≈ 993 gray). Use `scripts/slm/validate_flat_phase_gray.py` to verify. Parameters for stable observation: `--exposure-ms 0.8 --wait-time-s 0.3 --discard-count 3`.
- **SLM phase raw-only contract + unified grayscale entry** (2026-09): Every SLM phase generator (optimizer/runner layer) returns **raw unwrapped radians**; do NOT `mod 2π` in generators — the driver `create_phase_from_array()` is the single wrap point (radian→grayscale, `santec/driver.py` L1382). Convert radian phase to grayscale through `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` — passes a live SLM it delegates to the driver pipeline (grayscale + wavefront correction + LUT + shift, 2π = device `_max_gray`); `slm=None` (simulation/offline save/unit tests) falls back to the built-in pure math conversion (wrap→scale→clip→uint16, default 1023). Counter-example **history** (both since fixed/removed in 2026-09 — do NOT reintroduce): `PatternHelper._zernike_to_uint16` (**deleted**, `pattern_helper.py:499-500`) and `ZernikeDM.generate_phase` (**fixed**, `zernike_dm.py:117-118`). A live scale-invariant min-max normaliser remains a defect; the two names above no longer exist in the tree.
- **SLM memory-slot rotation**: When writing consecutive phases to memory mode, always target **different** slots — calling `display_memory(slot)` for the slot already displayed is a no-op and the LCOS panel will not refresh. Preferred pattern (diff-shaping runner): random slot in **2..125**, excluding the currently displayed one (`get_displayed_memory_number()` on start, survives process restarts). Older tools rotate a small pool (`itertools.cycle([3,4,5])`) — in-process only. `display_data()` cycles all 127 slots internally.
- **Streamlit background threads**: Any background loop that drives hardware must never touch `st.session_state` directly — it causes `missing ScriptRunContext` warnings and race conditions. Use the R50 `run_loop` pattern: pass a snapshot of parameters as a plain `dict`, communicate state changes via `threading.Event` for stop signals and mutable containers (e.g. `list`) for live-updated values like frequency, and drain feedback through a `queue.Queue`. The main thread reads/writes `st.session_state` only on rerun. See `src/ao_shaping/gui/r50/r50_voltage_send.py:run_loop` and `src/ao_shaping/gui/slm/multi_slm_controller.py:_toggle_phases_task` for implementations.
- **Timing in background loops**: Prefer `time.time()` wall-clock deltas over counters for state machines. Example: `int(elapsed * 2.0 * freq) % 2 == 0` toggles at exactly the requested frequency without drift, instead of sleeping fixed half-periods and accumulating error.
- **2f Fourier bench geometry** (SLM front focus → f=125mm lens → CCD back focus): the lens Fourier-transforms the SLM field, so CCD coordinates represent **spatial frequency** — the +1 orders of an upper-half-grating and a lower-half-grating land on the **same CCD row (optical-axis row)**, differing only in x-offset. Expected diffraction offset `Δx_px = λ·f/(d_SLM·p_cam) ≈ 5021/Λ` (P64→78px, P32→157px, P96→52, P40→126). Do NOT assume half-screen gratings separate in y. 0-order = frame global max (`argmax`), re-check its location after any bench change.
- **SLM panel "not modulating" diagnostic chain** (2026-09, encoded in `tools/slm/panel/diagnose.py`): ① freeze check — write flat/full-grating/top-half/bottom-half to **rotated memory slots**, frames must differ (all-identical ⇒ LCOS frozen); ② modulation check — `set_grayscale` sweep 0..1023, 0-order bucket must vary with ~993-gray period (flat ⇒ no amplitude coupling ⇒ panel not modulating); ③ linearity check — exposure ×4, ×20 must grow peak brightness (constant peak incl. at 0.1ms/2ms ⇒ light is >100× weaker than the known ~0.02ms near-saturation baseline). Verdict thresholds: freeze ≥2/3 frames differ, modulation bucket rel-spread >15%, linearity growth >2×.
- **Class-based optimizer convention**: optimizers in ao_shaping/algorithm expose __init__ (validate + set state), update() (one step → next state), optional run() → result dataclass; the one-shot function stays a thin wrapper. Simulation-first (torch/numpy) tests before hardware. Canonical example: DifferentiableBeamOptimizer. See src/ao_shaping/algorithm/README.md.
- **Report generation lives in `scripts/`**: any functionality that writes markdown/illustrated reports MUST live in `scripts/` (naming `generate_*_report.py`, e.g. `generate_zernike_wfs_report.py`, `generate_zernike_response_matrix_report.py`, `generate_diff_shaping_report.py`) — NEVER under `src/ao_shaping/tools/`, which is reserved for hardware-interaction tools (CLI + driver orchestration). Scripts follow the repo script conventions: `matplotlib.use("Agg")` BEFORE importing pyplot, `sys.path` bootstrap (`ROOT = Path(__file__).resolve().parents[1]`), CJK font rcParams (`Microsoft YaHei`/`SimHei` + `axes.unicode_minus = False`), output under `report/<topic>/<子目录>/` or `logs/<timestamp>/`, and MUST be documented in `scripts/README.md`. Prefer **offline** report generators (read saved artefacts) so a report can be regenerated without hardware.
- **`docs/` vs `report/` (2026-10-05 拆分, 红线)**: 文档分两类，**不可混放** —
  - `report/<topic>/` = **实验报告**：一次实验/基准/仿真的**结论**（测量值、参数、口径、被推翻的假设）。原先在 `docs/` 下，2026-10-05 全部迁到仓库根的 `report/`。
  - `docs/` = **设备说明文档**：设备规格、SDK/驱动用法、装配/操作手册、图集，外加**每设备硬件测试报告** `<device>/<device>_report.md`（由 `tests/ao_shaping/utils/test_report.py` 的 `TestReport` 写出，位置绑定该 harness 默认值与 `test_conventions.py` 的 tracked-docs 守卫，故**不迁**）。
  - **每份报告必须能追到生成它的脚本**：`scripts/_common/provenance.py::REPORTS` 是**唯一映射源**，渲染出每份报告的 `生成脚本 / 复现命令 / 运行环境` 溯源块，以及 `report/README.md` 总索引。改动报告或脚本后跑 `python scripts/sync_report_provenance.py`（幂等）；`--check` 只校验。**手工编辑溯源块或索引 = 会被 `tests/ao_shaping/scripts/test_report_provenance.py` 判失败。**
  - **代码注释不承载实验叙述**：实测数字/踩坑结论写进对应 `report/`，代码处只留**不变量 + 报告链接**（例如 `ZernikeDM` 的系数即弧度 → `report/slm/report2.md`；读出噪声致 `PIB>1` → `report/slm_pib_bench/pib_bench_report/report.md` §7）。函数 docstring 仍保留**契约**（这个函数做什么、为什么这样裁剪），只把长实测叙述外迁。
- **`report/` 目录结构 (2026-10-21 重组, 红线)**: 一份报告 = 一个子目录，**按生成脚本/实验系列**归类，报告与其独占资产同住 ——
  - `report/<topic>/<子目录>/<报告名>.md` + `figures/` + `images/` + `artifacts/` + `assets/`
  - **不**平铺在 `report/<topic>/` 根下；**不**建 `shared/` 公共图目录（跨报告复用 → 各自复制）
  - 生成脚本的 `REPORT_KEY` / `OUT_DIR` / `FIG_DIR` 常量必须**写死新子目录**（重跑不会写回旧位置）
  - 完整规则见 [`report/AGENTS.md`](../../report/AGENTS.md)（该文件是 `report/` 下的专用规范，与本节冲突时以它为准）

