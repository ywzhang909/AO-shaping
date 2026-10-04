# drivers/ - 硬件驱动

AO-Shaping 硬件驱动框架, 为 SLM、DM、WFS、CCD、TM 等设备提供统一接口。

## 目录结构

```
drivers/
├── AGENTS.md                # 本文件 (主文档)
├── INTERFACE_DOCS.md        # 硬件驱动接口文档 (架构/接口/扩展指南)
├── device_base.py           # Device 基类 (Device, DeviceState, DeviceType)
├── device_registry.py       # 设备注册与管理
├── mock_devices.py          # Mock 设备 (简单测试)
├── ccd/                     # 相机驱动 → [ccd/AGENTS.md](ccd/AGENTS.md)
├── dm/                      # 变形镜驱动 → [dm/AGENTS.md](dm/AGENTS.md)
├── slm/                     # SLM 驱动 → [slm/AGENTS.md](slm/AGENTS.md)
├── wfs/                     # 波前传感器驱动 → [wfs/AGENTS.md](wfs/AGENTS.md)
├── tm/                      # 定时模块驱动 → [tm/AGENTS.md](tm/AGENTS.md)
├── sim/                     # 模拟设备/数字孪生 → [sim/AGENTS.md](sim/AGENTS.md)
├── adc/                     # ADC 驱动
│   ├── driver.py
│   └── __init__.py
└── __pycache__/
```

## 各子包文档

| 子包 | 文档 | 主要内容 |
|------|------|----------|
| `ccd/` | [ccd/AGENTS.md](ccd/AGENTS.md) | 相机驱动 (BaseCamera, Daheng, MiiCam, FFmpeg) |
| `dm/` | [dm/AGENTS.md](dm/AGENTS.md) | 变形镜驱动 (NLight, Micro-DM, Hadamard) |
| `slm/` | [slm/AGENTS.md](slm/AGENTS.md) | SLM 驱动 (Santec SLM-200), 标定, 波前校正 |
| `wfs/` | [wfs/AGENTS.md](wfs/AGENTS.md) | 波前传感器 (Thorlab WFS) |
| `tm/` | [tm/AGENTS.md](tm/AGENTS.md) | 定时模块 (串口 FSM) |
| `sim/` | [sim/AGENTS.md](sim/AGENTS.md) | 模拟设备/数字孪生仿真 |

## 核心基类

### Device (`device_base.py`)

所有硬件驱动的抽象基类:

```python
class Device(ABC):
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def is_connected(self) -> bool: ...
    @abstractmethod
    def get_hardware_info(self) -> dict: ...
    def __enter__(self) -> "Device": ...
    def __exit__(self, ...): ...
```

### 设备类型枚举 (`DeviceType`)

```python
class DeviceType(Enum):
    CAMERA = auto()    # 相机/CCD
    SLM = auto()       # 空间光调制器
    DM = auto()        # 变形镜
    WFS = auto()       # 波前传感器
    STAGE = auto()     # 运动台
    LASER = auto()     # 激光器
    FILTER = auto()    # 滤光轮
    OTHER = auto()     # 其他设备
```

### 设备状态枚举 (`DeviceState`)

```python
class DeviceState(Enum):
    UNKNOWN = auto()
    DISCONNECTED = auto()
    CONNECTING = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
    CALIBRATING = auto()
```

## 约定 (Conventions)

| 约定 | 说明 |
|------|------|
| 自定义异常 | `*Error` 后缀 (如 `DeviceError`, `CameraError`) |
| 状态追踪 | `_set_state(state, error_msg)` |
| SDK 导入 | `try/except` 优雅处理, 不可用时 fallback |
| 日志 | `loguru.logger` |
| 参数注册 | `register_parameter()` 方法 |

详细接口规范见 [`INTERFACE_DOCS.md`](./INTERFACE_DOCS.md)。

## 驱动惰性加载契约 (2026-10-01)

`import ao_shaping` 及其任何子包**都不得加载原生 SDK、也不得做磁盘 I/O**。这不是
性能优化, 而是正确性问题 —— 仓库里已因此真实炸过两次。

### 两条独立的惰性

| 层级 | 机制 | 位置 |
|------|------|------|
| **包级** | PEP 562 模块 `__getattr__` | `drivers/_lazy.py::install_lazy_attrs` |
| **构造级** | `Device._load_sdk()` / `Device._ensure_sdk()` | `drivers/device_base.py` |

**包级**: 硬件类只登记在 `_LAZY_BACKENDS` 映射里, 首次属性访问才 import。
⚠️ `install_lazy_attrs` 内部**必须**写 `module_globals[name] = value` 缓存, 否则
后续直接 `import` 该子模块会重新绑定全局变量, 惰性被彻底绕过。
⚠️ 惰性名字**绝不能**在模块作用域出现同名赋值 —— 那会遮蔽 `__getattr__` 并退回 eager。

**构造级**: 驱动**构造**不得加载 SDK, 需求推迟到 `open()`。

```python
class MyDriver(Device):
    @staticmethod
    def _load_sdk():        # 覆写点; 默认返回 None (仿真设备无 SDK)
        return load_dll()

    @property
    def _lib(self):          # 只读属性 → 所有 self._lib.<fn> 读点零改动
        return self._ensure_sdk()   # 解析一次并缓存
```

`ThorlabWFS` 曾把 `load_dll()` 写在 `__init__` 里, 于是**构造**就 `OSError`
(`WFS_64.dll` 缺失), 使离线脚本无法构造驱动实例。现在 53 处 `self._lib.<fn>` 读点
全部经由惰性属性, 零改动。

### 静态资源也必须锚定包路径, 不可用 CWD 相对路径

`dm/nlight/driver.py` (原 `NLight.py`) 曾在**类体**里执行 `np.loadtxt("data/dm_adj.txt")` —— 类体赋值即
**import 期**读文件, 且路径相对 CWD。任何在仓库根目录之外运行的消费者 (cron、
服务、安装后的 console script、`cwd=tmp_path` 的测试) 都会
`FileNotFoundError: data/dm_adj.txt not found`, 即 **`import ao_shaping` 直接失败**。

规则:
- 仓库**输入**资产 (如 `dm_adj.txt`) → 经 `drivers/dm/_adjacency.py::load_adjacency()`
  加载, 路径锚定包位置, 带回退与缓存。
- 运行**输出** (如 `PATHS.root_dir`) → CWD 相对是**正确**的, 不要改。
- 禁止在任何类体/模块作用域做 I/O。

### 分层方向: 硬件包不得 import 仿真包

`drivers/dm/__init__.py` 曾 import `drivers.sim.dm` (硬件 → 仿真, 方向反了), 而
`simulated_micro_dm` 又 import `drivers.dm.base`, 形成循环。该循环此前靠
`drivers/__init__.py` 里的 eager 导入顺序侥幸未暴露 —— 一旦改成惰性就立刻炸。
现在 `sim` / `sim_micro` 经同一个 `install_lazy_attrs` 惰性解析, 依赖方向单一。

`dm/_registry.py::_ensure_sim_dms_bound()` 保证 `"sim"` 在**首次注册表使用**时绑定,
因此 `list_dm_types()` / `create_dm()` / `resolve_dm()` 无论进程先 import 了哪个包
都能看到仿真 DM —— 不需要 `drivers/dm/__init__.py` 反向依赖。

## 硬件事实 (2026-09 实测确认)

| 设备 | 确认信息 |
|------|---------|
| Santec SLM200 SLM#1 | 序列号 22030108; 1920×1200; 10-bit; 内部内存模式; @1064nm 2π=993 灰度 (设备动态查询) |
| Daheng MER2-507-23GM NIR CCD | 序列号 FJB24112232; 2592×1944; uint8; 像元 **2.2 µm** (用户 2026-09-21 权威确认); IP 192.168.0.11 —— 全部 SLM-PIB 硬件数据都依赖此相机, 2026-10-01 补录 |
| MiiCam 相机 | 序列号 TP2408221418059418FD83E3A448D82; 2688×1520; MONO8; `reset_exposure_time()` 修改曝光必须 Stop→设置→重启拉流 |
| 2f Fourier 光路 | SLM 前焦面 125mm → f=125mm 透镜 → CCD 后焦面; CCD 坐标=空间频率; 0 级=帧全局最大 (argmax), 非相机几何中心 |

> 🔴 **SLM 序列号三路冲突 (2026-10-01 复核, 尚未收口)**：本文件与
> `report/slm/bench_calibration_20261001.md` 记 SLM#1 = **22030108** (@1064nm, 2π=993)；
> `drivers/slm/AGENTS.md:114,139` 记 SLM = **22030102** (@532nm, 2π=998)；
> `report/slm/report2.md` / `report3.md` / `zernike_linearity` 记 **23020026** (@532nm)。
> 三者可能对应**两台不同 SLM**（SLM#1 @1064nm 与另一台 @532nm）。**引用任一序列号前
> 请先确认是哪台设备**，并把结论回写到本表。见 `TODO.md`。

### SLM 故障排查 (详细见 [slm/AGENTS.md](slm/AGENTS.md))

`tools/slm/slm_diagnose.py` 三步自检:

1. **freeze**: flat/全屏光栅/上下半屏光栅写轮换内存槽 → 帧必须随图案变化 (≥2/3 帧不同)
2. **modulate**: `set_grayscale` 扫描, 0 级桶能量须有 ~993 灰度周期 (相对变化 >15%)
3. **linearity**: 曝光 ×4、×20, 峰值亮度须增长 (>2×)

### 跨子包已知坑 (详细见各子包 AGENTS.md)

| 坑 | 影响子包 | 详情 |
|----|----------|------|
| DVI 模式 `open()` 会挂起 | slm | `video_mode=1` 可挂 120s/300s, 需物理断电重置 |
| diff-shaping 硬件闭环挂起 | slm | `WaitImageV3` 不受 Python 超时保护, 需确认无残留进程 |
| `get_displayed_memory_number` 报错码 1 正常 | slm | set_grayscale 模式无内存槽显示 |
| 同内存槽连续 `display_memory` 是 no-op | slm | 固件不刷新正在显示的槽位 |
| MiiCam 曝光修改 | ccd | 须 `reset_exposure_time()` (Stop→put_ExpoTime→重启拉流) |
| 类名 `NLight` 非 `NlightDM` | dm | `dm/nlight/driver.py` 中类名为 `NLight`, 非 `NlightDM` |

## Mock 设备

`mock_devices.py` 提供简化模拟设备用于无硬件开发测试, 与 `sim/` (数字孪生) 区别在于复杂度较低。

详细使用见 [`INTERFACE_DOCS.md`](./INTERFACE_DOCS.md) §Mock 设备。

## 相关文档

| 文档 | 路径 | 内容 |
|------|------|------|
| 接口文档 | `drivers/INTERFACE_DOCS.md` | 架构概览、接口规范、扩展指南 |
| SLM 标定 | `drivers/slm/README.md` | 闪耀光栅法/零级比值法/干涉法/衍射效率法标定详解 |
| 方形光斑 SPGD | `report/slm/slm_square_spgd/README.md` | SPGD 整形硬件实测与故障记录 |
| 实验报告总索引 | `report/README.md` | 报告 → 生成脚本 → 离线/硬件 的对应关系；硬件实测结论查这里，设备用法查上面两份 |
