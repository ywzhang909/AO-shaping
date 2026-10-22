# 二次开发与扩展指南

> **读者**：第一次修改 AO-Shaping 的开发者，或要把这套代码接到自己光路上的实验人员。
> **目标**：读完能判断“我的改动该放哪一层、要复用哪个 canonical 入口、要交什么测试和文档”。
>
> 本文只做**路由 + 最小模板**。完整规范以仓库根 [`AGENTS.md`](../../AGENTS.md)、
> 各子包 `AGENTS.md` 与下列 canonical 文档为准：
>
> - 算法约定：[`src/ao_shaping/algorithm/README.md`](../../src/ao_shaping/algorithm/README.md)
> - 编码与测试：[`coding_conventions.md`](coding_conventions.md)
> - 驱动扩展：[`../drivers_architecture.md`](../drivers_architecture.md) §9
> - CLI 逐字参考：[`../user/cli-reference.md`](../user/cli-reference.md)
> - 术语：[`../terminology_glossary.md`](../terminology_glossary.md)

---

## 1. 先判断改动属于哪一层

```
main.py（Click CLI）
  └── runners/        命令注册、设备 open/close、参数解析、结果落盘
        ├── optimizer/ 闭环策略、目标函数编排、Recorder 历史
        │     ├── algorithm/  纯数学优化器（不得 import 硬件）
        │     └── drivers/    设备 I/O（SDK 惰性加载）
        └── utils/ + model/  叶子工具与物理量数据类
```

| 层 | 目录 | 只做什么 | 不允许 |
|---|---|---|---|
| CLI | `src/ao_shaping/main.py` | 注册命令、组级选项 | 写业务逻辑 |
| Runner | `src/ao_shaping/runners/<domain>/` | 参数声明、设备生命周期、保存产物 | 实现优化算法 |
| Optimizer | `src/ao_shaping/optimizer/{wf,wfless}/` | 闭环流程、目标函数编排、Recorder | 直接解析原生 SDK |
| Algorithm | `src/ao_shaping/algorithm/<category>/` | 纯数学更新（`__init__` + `update`） | import driver / runner / GPU 假设 |
| Driver | `src/ao_shaping/drivers/<domain>/` | 设备 I/O、状态机、上下文管理器 | import optimizer / runner；导入期加载 SDK |
| Model | `src/ao_shaping/model/` | 相位/OPD/场/PSF 等物理量数据类 | 依赖具体设备 |
| Utils | `src/ao_shaping/utils/` | 叶子工具（图像、Zernike、IO、CLI 参数） | 反向 import `algorithm/`、`drivers/`、`optimizer/` |
| ML | `src/ml/` | 数据集、模型、训练、指标（独立包） | 在模块导入期打开设备 |
| Scripts | `scripts/` | 离线分析、批量实验、报告生成 | 被 `src/` 反向 import |

依赖方向**只能向下**。一个常见违规是 `utils/` 为了图方便 import `algorithm/`；
必须改成 `TYPE_CHECKING` 或函数内延迟 import。

`src/ml/` 可以复用 `ao_shaping.utils.*` 的纯函数与 `runners.gsnet_offline` 的纯函数，
但不得在模块导入期初始化硬件。

---

## 2. 三条不可协商的契约

### 2.1 惰性加载

`import ao_shaping` 及其任何子包**都不得加载原生 SDK、也不得做磁盘 I/O**。

| 层级 | 机制 | 位置 |
|---|---|---|
| 包级 | PEP 562 `__getattr__` | `drivers/_lazy.py::install_lazy_attrs` |
| 构造级 | `Device._load_sdk()` / `Device._ensure_sdk()` | `drivers/device_base.py` |

```python
from ao_shaping.drivers.ccd.base import BaseCamera


class MyCamera(BaseCamera):
    @staticmethod
    def _load_sdk():
        import my_sdk          # 只有 open() 时才会走到这里
        return my_sdk

    def open(self) -> None:
        sdk = self._ensure_sdk()
        self._cam = sdk.open(self.cam_id)
```

红线：

- 惰性名字不得在模块作用域出现同名赋值，否则会遮蔽 `__getattr__`；
- 不得在类体/模块作用域做 I/O；
- 硬件包不得 import 仿真包（仿真 DM 由注册表在首次使用时绑定）。

### 2.2 仿真先行

算法与优化器先用 numpy/torch 合成目标验证，**测试不得 import 硬件 SDK**。
需要实验系统的测试用 `@pytest.mark.hardware` 标记，普通回归不收集。

runner 侧用仿真后端：

```bash
python src/ao_shaping/main.py slm-pib spgd --cam-type sim -e 3
python src/ao_shaping/main.py slm-pib square --cam-type sim -e 3
```

部分命令另有 `--slm-type sim`（必须与 `--cam-type sim` 成对使用）。
仿真后端默认走 numpy/FFT；设置 `AO_OOPAO_BACKEND=1` 可切到 OOPAO，见
[`../simulation.md`](../simulation.md)。

### 2.3 单一事实源

扩展时**调用** canonical 实现，不要复制：

| 能力 | 唯一入口 |
|---|---|
| Zernike 模式/相位/系数解析 | `utils/wavefront/zernike_utils.py`（上层）、`zernike_calc.py`（引擎） |
| 相位→SLM 灰度 | `utils/slm/phase_display.py::phase_to_slm_grayscale` |
| 目标方向（max/min） | `utils/objective_direction.py::objective_direction` |
| 复合整形质量分 | `algorithm/signal_processing/square_shaping.py::shaping_score` |
| 2f-Fourier 前向模型 | `drivers/sim/slm_shaping_bench.py` |
| 报告溯源映射 | `scripts/_common/provenance.py::REPORTS` |
| 统计检验 | `ml/zernike/eval_stats.py` |

发现两处实现不一致时，正确做法是**删掉副本并 re-export**，不是让两份继续漂移。

---

## 3. 按目标选路线

| 我想…… | 放哪里 | 必读 | 最低验证 |
|---|---|---|---|
| 新增一个优化器（Adam 变体、新启发式） | `algorithm/<category>/` | `algorithm/README.md` | CPU 合成目标测试 |
| 接一台新相机 / DM / WFS | `drivers/<domain>/` | `drivers/AGENTS.md`、`drivers_architecture.md` §9 | 导入惰性测试 + 仿真/硬件标记测试 |
| 新增一种闭环策略或目标 | `optimizer/{wf,wfless}/` | 同目录现有优化器 + Recorder | 仿真闭环或离线回放 |
| 把策略暴露成 CLI | `runners/<domain>/` + `main.py` | 本文 §4.4 | `--help` golden + 参数测试 |
| 训练/替换模型 | `src/ml/<domain>/` | `../ml_training.md` | 小数据过拟合 + 指标口径 |
| 新增 Streamlit 页面 | `gui/<domain>/` | `../gui.md`、`docs/GUI/` 现有 SRS | 手动 UI 检查 + 后台线程规则 |
| 写离线分析/出报告 | `scripts/<domain>/` | `scripts/README.md` | 可复现命令 + 溯源块 |

---

## 4. 最小模板

### 4.1 新增算法（`algorithm/`）

约定：**类是有状态优化器**。`__init__` 做校验与状态初始化，`update()` 只走一步，
循环、日志、早停、Recorder 都属于 optimizer 层。

```python
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class MyOptimizer:
    """最小单步优化器：``update`` 返回下一步的参数。"""

    def __init__(self, *, dim: int, lr: float = 1e-2, seed: int | None = None) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(dim, dtype=np.float64)

    def update(self, grad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.x = self.x - self.lr * grad
        return self.x
```

测试放在 `tests/ao_shaping/algorithm/`，只用合成目标：

```python
def test_update_decreases_quadratic() -> None:
    opt = MyOptimizer(dim=2, lr=0.1, seed=0)
    opt.x = np.array([1.0, -2.0])
    for _ in range(50):
        opt.update(2.0 * opt.x)
    assert np.linalg.norm(opt.x) < 1e-6
```

检查单：

- [ ] `__init__` 对非法输入抛 `ValueError`；
- [ ] 随机性由 `seed` 控制；
- [ ] `update()` 只做一步、返回下一步；
- [ ] 没有硬件/SDK import；
- [ ] 数组注解用 `npt.NDArray[...]`，不写裸 `np.ndarray`。

### 4.2 新增/替换设备驱动（`drivers/`）

1. 在对应域下新建驱动类，继承该域基类（`BaseCamera` / `DM` / `BaseWFS` …）；
2. 覆写 `_load_sdk()`，把 SDK 加载推迟到 `open()`；
3. 实现 `open` / `close` / `is_connected` / `get_hardware_info`；
4. 注册到注册表；
5. 提供仿真或 Mock 回退，并给硬件测试加 `@pytest.mark.hardware`。

相机注册（`drivers/ccd/common.py`）：

```python
from ao_shaping.drivers.ccd.common import register_camera

register_camera(
    "mycam",
    "ao_shaping.drivers.ccd.mycam:MyCamera",
    {"cam_id", "exposure_time_ms"},
)
```

DM 注册（`drivers/dm/_registry.py`）：

```python
from ao_shaping.drivers.dm._registry import register_dm

@register_dm("mydm")
class MyDM(DM):
    ...
```

WFS 注册（`drivers/wfs/_registry.py`）：

```python
from ao_shaping.drivers.wfs._registry import register_wfs

@register_wfs("mywfs")
class MyWFS(BaseWFS):
    ...
```

注意：

- 新增相机类型还要同步 `CAMERA_TYPE_CHOICES`（Click 的静态 `Choice`）与 CLI
  `--help` golden；DM 类型列表来自 `list_dm_types()`，但合约测试会冻结可选集合。
- 导入期不能 `import my_sdk`；不能创建句柄、不能探测设备。
- 安全相关的默认值（电压上限、曝光上限、保持/退出状态）必须在驱动内可配置，
  并在 runner 侧有显式开关。

### 4.3 新增闭环策略（`optimizer/`）

把“打开设备 → 循环 → 记录 → 关闭”放在 optimizer 函数里，配置收敛成一个
`@dataclass` 参数；设备由优化器内部创建与关闭，不跨 run 复用。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.optimizer.constants import create_optimizer
from ao_shaping.utils.io.file import Recorder

if TYPE_CHECKING:
    from ao_shaping.runners.runner_common import CameraParamsPib, SlmParamsPib


@dataclass
class MyShapingConfig:
    center: str | tuple[int, int] | None
    epochs: int
    dim: int
    camera: CameraParamsPib
    slm: SlmParamsPib


def optimize_my_shaping(config: MyShapingConfig) -> Recorder:
    recorder = Recorder(mark="J", mode="max")
    with create_camera(config.camera) as cam, Santec.from_params(config.slm) as slm:
        optimizer = create_optimizer("adamod", dim=config.dim, lr=0.1)
        for epoch in range(config.epochs):
            # 1) 采样 -> 2) 算目标 -> 3) 更新参数 -> 4) 下发 -> 5) 记录
            recorder.append({"J": ..., "epoch": epoch})
    return recorder
```

> 上面的 `camera` / `slm` 为必填字段，由 runner 组装后传入。生产代码里可参考
> `SlmZernikePibConfig`（`optimizer/wfless/slm_zernike_pib.py`）：它用延迟工厂提供
> 默认值，避免 `optimizer → runners` 的模块级导入环。

要求：

- 目标方向通过 `objective_direction(objective)` 取，不要手写 `if objective == ...`；
- 每轮记录进 `Recorder`，字段名保持稳定（报告脚本与测试依赖它）；
- 设备异常必须 `close()`（用 `with`）；
- 先用仿真后端或离线回放验证，再上实验系统；
- 优化器的循环逻辑不要下沉到 `algorithm/` 类里。

### 4.4 新增 CLI 命令（`runners/` + `main.py`）

参数声明用 `Annotated[T, option(...)]`，默认值写在 dataclass 字段上
（`option()` 内写 `default=` 会直接 `TypeError`）：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import click

from ao_shaping.utils.cli.params import option, with_params


@dataclass
class MyRunParams:
    epochs: Annotated[int, option("-e", "--epochs", help="优化迭代次数。")] = 100
    debug: Annotated[bool, option("--debug", is_flag=True, help="保存调试产物。")] = False


@click.command()
@click.pass_context
@with_params(MyRunParams, kw_name="run")
def run(ctx: click.Context, run: MyRunParams) -> None:
    """一句话说明这条命令做什么。"""
    ...
```

注册：

```python
# src/ao_shaping/main.py
from ao_shaping.runners.mydomain.my_runner import run as my_run

cli.add_command(my_run, name="my-command")
```

新增命令会改变冻结的 CLI 合约，必须重生成两份快照：

```bash
# 1) 逐命令 --help 快照
AO_UPDATE_HELP_GOLDEN=1 \
  ./run_pytest.sh tests/ao_shaping/runners/test_cli_help_golden.py

# 2) 命令清单 + flag 合约
AO_CLI_CONTRACT_UPDATE=1 \
  ./run_pytest.sh tests/ao_shaping/runners/test_cli_contract_freeze.py
```

重生成后**不带环境变量再跑一遍**，确认快照稳定。随后：

- 在 `docs/<domain>/` 增加命令页（以“这条命令做什么”开头）；
- 在 [`docs/INDEX.md`](../INDEX.md) 登记；
- 在 README 的 CLI 表与 `python -m` 入口表补一行。

### 4.5 新增模型或训练入口（`src/ml/`）

1. 在 `src/ml/<domain>/` 放数据集、模型、损失、训练脚本；
2. 数据集优先复用 `ml.hwdataset` 的样本契约（相位 cos/sin + 曝光 + 图像）；
3. 训练入口用 `click`，配置用冻结 dataclass；
4. 安装：`uv sync --group ml --extra torch-cpu`（或对应 CUDA extra）；
5. 模型选择看 **R² / corr / 光斑域指标**，不要用 PSNR/SSIM 给归一化开关排序；
6. 实验结论写 `report/`，训练入口与指标口径登记到
   [`../ml_training.md`](../ml_training.md)。

### 4.6 新增 GUI 页面（`gui/`）

- 代码放 `src/ao_shaping/gui/<domain>/`，不要跨设备域混放；
- 后台循环不得直接读写 `st.session_state`，用 `threading.Event` + `queue.Queue`
  或普通 `dict` 快照传递；
- 页面要有对应的 `docs/GUI/<domain>/<page>/<page>_srs.md` 与
  `<page>_user_manual.md`，并登记到 [`../gui.md`](../gui.md)；
- 硬件页面默认不自动打开设备，必须由用户显式点击连接。

---

## 5. 仿真与测试

### 5.1 仿真入口

```bash
# SLM 整形：相机与 SLM 都走数字孪生
python src/ao_shaping/main.py slm-pib spgd --cam-type sim -e 3
python src/ao_shaping/main.py slm-pib square --cam-type sim -e 3

# DM/WFS 相关命令按各自 --help 选择 sim 类型
python src/ao_shaping/main.py wf --help
```

### 5.2 测试

```bash
# 首次：构建镜像
docker build -f docker/Dockerfile.test -t ao-shaping-test:gpu .
docker build -f docker/Dockerfile.test.cpu -t ao-shaping-test:cpu .

# GPU 机器
./run_pytest.sh
./run_pytest.sh tests/ao_shaping/algorithm -- -q

# 无 GPU / CI
AO_IMAGE=ao-shaping-test:cpu ./scripts/build/run_pytest_ci.sh
```

裸 `pytest` 会被 `conftest.py` 拒绝（宿主机 384 核，无限线程池会拖死机器）。
默认排除 `tests/ao_shaping/drivers`、`tests/ao_shaping/ml` 与一个会挂起的 GUI 测试；
显式指定路径时仍会收集。

测试要求：

- 算法/优化器：合成目标上的数值测试；
- 驱动：导入惰性测试 + `@pytest.mark.hardware` 的实验系统测试；
- CLI：参数解析、默认值、`--help` 快照；
- 修复 bug 时先写失败测试，再改代码。

---

## 6. 文档与报告门禁

### 6.1 术语

所有中文叙述遵守 [`../terminology_glossary.md`](../terminology_glossary.md)。
提交前：

```bash
python3 scripts/_common/term_unify_20261016.py --check   # 非 0 = 有违规
python3 scripts/_common/term_unify_20261016.py           # dry-run，应输出 0 个文件
```

### 6.2 `docs/` 与 `report/` 的分工

| 目录 | 放什么 |
|---|---|
| `docs/` | 设备规格、SDK/驱动用法、装配/操作手册、开发指南 |
| `report/` | 一次实验/基准/仿真的**结论**：测量值、参数、口径、被推翻的假设 |

每份 `report/` 必须有生成脚本溯源块。新增报告时：

1. 在 `scripts/_common/provenance.py::REPORTS` 增加一条映射；
2. 运行 `python scripts/sync_report_provenance.py`；
3. 用 `python scripts/sync_report_provenance.py --check` 校验。

**报告生成代码只能放 `scripts/`**（命名 `generate_*_report.py`），不能放
`src/ao_shaping/tools/` 或 `algorithm/`。

### 6.3 索引

新增任何 `docs/*.md` 后，在 [`docs/INDEX.md`](../INDEX.md) 登记并更新文档计数。
README 只保留总览与高频入口，不复制详细指南。

---

## 7. 提交前检查单

- [ ] `ruff check .` 全绿（CI 硬门禁）；
- [ ] 相关测试在容器里通过（`./run_pytest.sh <path>`）；
- [ ] 术语门禁 `--check` 通过；
- [ ] 改了 CLI → 两份 `--help` golden 已重生成并复跑；
- [ ] 新增文档 → `docs/INDEX.md` 已登记，README 链接可达；
- [ ] 新增报告 → `REPORTS` 映射 + `sync_report_provenance.py` 已运行；
- [ ] 没有在导入期加载 SDK / 没有提交设备序列号、密钥、绝对路径；
- [ ] 提交信息遵循 Conventional Commits（`docs(...)`、`feat(...)`、`fix(...)`）。

---

## 8. 常见坑（都真实发生过）

| 坑 | 正确做法 |
|---|---|
| 模块导入时 `import` 原生 SDK | 用 `Device._load_sdk()` / `_ensure_sdk()`，首次 `open()` 才加载 |
| 相位做 min-max 归一化 | 用 `phase_to_slm_grayscale()`；生成器只返回 raw 弧度 |
| 连续写同一个 SLM 内存槽 | 每次写到不同槽（如 2..125 随机，排除当前显示槽），否则面板不刷新 |
| 每轮用 `argmax` 重新定位 ROI | 在未整形帧上定位一次并冻结，否则目标框会跳 |
| WFS 的 µm 系数直接进相位 | 先 `um_to_waves()`，λ→rad 再乘 `2π` |
| 在原始 CCD 帧上算 PIB/CV | 先 `prepare_frame()` 去中值并 clip，否则噪声会让 PIB > 1 |
| 自己写一套 Zernike 数学 | 只用 `utils/wavefront/` 两层入口 |
| 单 seed / 非配对比较模型 | 用 `ml/zernike/eval_stats.py` 的配对检验；换 seed 就对 seed 配对 |
| 把闭环的逐位复现当性质 | seed 只固定扰动符号，测量噪声不可复现；断言近似一致 |
| 把报告生成器放进 `src/` | 报告生成只放 `scripts/` |
