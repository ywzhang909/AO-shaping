# AO-Shaping 驱动层架构设计文档

## 1. 概述

`ao_shaping.drivers` 模块为自适应光学系统提供了统一的硬件抽象层，支持多品牌设备接入和虚拟仿真环境。该模块的核心设计目标是：

- **硬件无关性**：通过抽象接口兼容不同厂商的相机、SLM、变形镜、波前传感器等设备
- **软硬件分离**：提供 Mock 和 Simulated 设备，支持无硬件开发测试
- **数字孪生支持**：设备状态可同步至数字孪生系统

---

## 2. 架构设计

### 2.1 整体架构（设备层 ↔ 仿真层孪生对应）

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ao_shaping.drivers                                               │
│                                         (Facade 统一入口)                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                        基础层                                                   │   │
│  │  ┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐                        │   │
│  │  │  device_base.py  │   │  device_registry.py │   │   visa_base.py   │                        │   │
│  │  │ ────────────────  │   │ ────────────────────│   │ ────────────────│                        │   │
│  │  │ Device (ABC)     │   │ DeviceRegistry      │   │ VISA 通信抽象     │                        │   │
│  │  │ DeviceState      │   │ 设备注册/发现/批量   │   │ PyVISA 封装       │                        │   │
│  │  │ DeviceType      │   │ 数字孪生状态同步      │   │                  │                        │   │
│  │  │ DeviceParameter │   └─────────────────────┘   └──────────────────┘                        │   │
│  │  │ DeviceMetadata  │                                                                         │   │
│  │  └──────────────────┘                                                                         │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                      │
│  ┌─────────────────────────────────────┬─────────────────────────────────────────────────────────┐ │
│  │          设备层 (Hardware)           │              仿真层 (Digital Twin)                       │ │
│  │         真实硬件驱动                  │              基于物理模型的数字孪生                    │ │
│  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────┤ │
│  │  ccd/                        ║      │  sim/ccd/                                               │ │
│  │  ├── base.py                 ║      │  │                                                       │ │
│  │  │   BaseCamera (ABC)        ║←兼容→ │  │  SimulatedCCD                                        │ │
│  │  ├── daheng.py               ║      │  │  ├── 继承 BaseCamera                                   │ │
│  │  │   CameraStreamManager     ║      │  │  ├── 物理: 噪声 + 特征生成                             │ │
│  │  └── miicam.py               ║      │  │  └── 仿真: 曝光/ROI 参数响应                           │ │
│  │                            ║      │  └────────────────────────────────────────────────────────│ │
│  │  dm/                       ║      │  sim/optics/                                              │ │
│  │  ├── base.py               ║      │  │                                                       │ │
│  │  │   DM (ABC)              ║←兼容→ │  │  SimulatedSLM                                         │ │
│  │  ├── NLight.py              ║      │  │  ├── 继承 WavefrontProcessor                          │ │
│  │  │   NLightDM (UDP)        ║      │  │  ├── 物理: 相位调制 + 波前传播                         │ │
│  │  └── simulateDM.py          ║      │  │  └── 仿真: Gamma 校正 + 光学元件                        │ │
│  │                            ║      │  │                                                       │ │
│  │  slm/                      ║      │  │  SimulatedLens                                         │ │
│  │  ├── santec_slm200.py      ║      │  │  ├── 物理: 透镜相位                                     │ │
│  │  │   SantecSLM200          ║      │  │  └── 仿真: 焦距可调                                    │ │
│  │  └── santec_slm200_visa.py ║      │  │                                                       │ │
│  │                            ║      │  │  SimulatedAperture                                     │ │
│  │  wfs/                     ║      │  │  ├── 物理: 光阑遮挡                                    │ │
│  │  └── ThorlabWFS.py        ║      │  │  └── 仿真: 半径/遮挡切换                               │ │
│  │        ThorlabWFS          ║      │  └────────────────────────────────────────────────────────│ │
│  │                            ║      │  sim/laser/                                               │ │
│  │  tm/                      ║      │  │                                                       │ │
│  │  └── serial_port_fsm.py    ║      │  │  SimulatedLaser                                       │ │
│  │                            ║      │  │  ├── 物理: 功率衰减 + 模式                             │ │
│  └────────────────────────────║      │  │  └── 仿真: 功率/波长参数                               │ │
│                                 ║      │  └────────────────────────────────────────────────────────│ │
│  ═════════════════════════════════║      │  sim/atmos/                                             │ │
│  设备层与仿真层 API 完全兼容         ║      │  │                                                       │ │
│  (硬件 ↔ 仿真可无缝切换)            ║      │  │  SimulatedTurbulentScreen                            │ │
│                                   ║      │  │  ├── 物理: Kolmogorov 湍流                           │ │
│  ┌────────────────────────────║      │  │  └── 仿真: Cn² 可调 + 实时更新                         │ │
│  │   mock_devices.py           ║      │  │                                                       │ │
│  │   MockCamera / MockDM      ║      │  │  SimulatedThermalScreen                               │ │
│  │   MockSLM / MockWFS        ║      │  │  ├── 物理: 热波动                                       │ │
│  │   (仅桩函数, 无物理模型)     ║      │  │  └── 仿真: 温度梯度模拟                                │ │
│  └────────────────────────────║      │  └──────────────────────────────────────────────────────│ │
│                                 ║      │                                                            │ │
│                                 ║◄─────┘   sim/base.py                                           │ │
│                                    ┌────┴───────────────────────────────────────┐              │ │
│                                    │  SimulatedDevice (基类)                      │              │ │
│                                    │  ├── enable_noise: bool                      │              │ │
│                                    │  ├── random_seed: int                       │              │ │
│                                    │  └── set_seed() / set_noise()               │              │ │
│                                    │        │                                     │              │ │
│                                    │        ▼                                     │              │ │
│                                    │  OpticalDevice                               │              │ │
│                                    │  ├── wavelength: float                      │              │ │
│                                    │  └── process() [抽象]                       │              │ │
│                                    │        │                                     │              │ │
│                                    │        ▼                                     │              │ │
│                                    │  WavefrontProcessor                         │              │ │
│                                    │  ├── npix / dpix                            │              │ │
│                                    │  └── set_phase() / get_phase()               │              │ │
│                                    └─────────────────────────────────────────────┘              │ │
│                                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心类关系

```
Device (ABC)                          DeviceRegistry
├── device_type: DeviceType           ├── register(device, alias, tags)
├── manufacturer/model/version        ├── get(device_id)
├── open() / close()                  ├── find_by_type(type)
├── is_connected()                   ├── find_by_tag(tag)
├── get_hardware_info()              ├── connect_all() / disconnect_all()
├── register_parameter()              ├── get_all_twin_states()
├── register_capability()             └── export_config()
└── get_twin_state() / sync_from_twin()

        ▲                                    ▲
        │ inherits                            │ manages
        │                                    │
   ┌────┴────┐                         ┌─────┴──────┐
   │ 实现类   │                         │ 实例集合    │
   │         │                         │            │
 NLightDM ◄─┼─► SimulateDM          Camera + DM + ...
    │       │    │                        │
    │   物理模型  │                        │
    │   (电压→  │                        │
    │  变形)    │                        │
    │          │                        │
 SimulatedCCD ◄─► 继承 BaseCamera                        │
    │          │                                          │
    │   物理   │                                          │
    │   (噪声+ │                                          │
    │   特征)  │                                          │
```

### 2.3 设备 ↔ 仿真 孪生映射表

| 设备类型 | 硬件实现 | 仿真实现 | 物理模型 |
|----------|----------|----------|----------|
| **相机 (CCD)** | `CameraStreamManager` (Daheng)<br>`MiiCamDevice` (MiiCam) | `SimulatedCCD` | 噪声叠加<br>高斯斑点生成<br>曝光响应 |
| **变形镜 (DM)** | `NLightDM` (UDP) | `SimulateDM` | 电压→变形矩阵<br>邻接耦合<br>电压爬升限制 |
| **SLM** | `SantecSLM200` (SDK)<br>`SantecSLM200Visa` (VISA) | `SimulatedSLM` | 相位调制<br>Gamma校正<br>波前传播 |
| **透镜** | -- | `SimulatedLens` | 抛物线相位 |
| **光阑** | -- | `SimulatedAperture` | 圆形遮挡 |
| **激光器** | -- | `SimulatedLaser` | 功率衰减 |
| **大气湍流** | -- | `SimulatedTurbulentScreen` | Kolmogorov谱 |
| **WFS** | `ThorlabWFS` | `SimulatedWFS` | 波前测量模拟 |

---

## 3. 核心接口规范

### 3.1 Device 抽象基类

所有硬件驱动程序必须继承 `Device` 抽象基类并实现以下方法：

```python
from ao_shaping.drivers import Device, DeviceType, DeviceState

class MyDevice(Device):
    device_type = DeviceType.CAMERA
    manufacturer = "MyCompany"
    model = "XYZ-100"
    
    def __init__(self, device_id: str = ""):
        super().__init__(device_id)
        self._register_parameters()
    
    def _register_parameters(self) -> None:
        """注册设备参数供数字孪生系统使用"""
        self.register_parameter(
            "exposure_time",
            20.0,
            min_value=1.0,
            max_value=1000.0,
            unit="ms",
            description="曝光时间"
        )
    
    @abstractmethod
    def open(self) -> None:
        """打开设备连接"""
        self._set_state(DeviceState.CONNECTING)
        # ... 硬件初始化代码
        self._set_state(DeviceState.READY)
    
    @abstractmethod
    def close(self) -> None:
        """关闭设备连接"""
        # ... 资源释放代码
        self._set_state(DeviceState.DISCONNECTED)
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查设备连接状态"""
        return self._state == DeviceState.READY
    
    @abstractmethod
    def get_hardware_info(self) -> dict[str, Any]:
        """获取硬件信息"""
        return {
            "serial": self._serial,
            "firmware": self._firmware_version,
        }
```

### 3.2 必需实现的方法

| 方法 | 说明 | 抛出异常 |
|------|------|----------|
| `open()` | 打开设备连接 | `ConnectionError`, `DeviceError` |
| `close()` | 关闭连接释放资源 | - |
| `is_connected()` | 检查连接状态 | - |
| `get_hardware_info()` | 获取硬件信息 | - |

### 3.3 状态管理

使用统一的 `_set_state()` 方法管理设备状态：

```python
def _set_state(self, state: DeviceState, error_msg: str | None = None) -> None:
    """状态变更会自动记录日志"""
    old_state = self._state
    self._state = state
    self._error_message = error_msg
    
    if state == DeviceState.ERROR and error_msg:
        logger.error(f"Device {self._device_id} error: {error_msg}")
    elif old_state != state:
        logger.debug(f"Device {self._device_id}: {old_state.name} -> {state.name}")
```

**DeviceState 枚举值**：

| 状态 | 说明 |
|------|------|
| `UNKNOWN` | 未知状态 |
| `DISCONNECTED` | 未连接 |
| `CONNECTING` | 连接中 |
| `READY` | 就绪 |
| `BUSY` | 工作中 |
| `ERROR` | 错误 |
| `CALIBRATING` | 校准中 |

### 3.4 上下文管理器支持

```python
# 推荐用法：自动资源管理
with NLightDM() as dm:
    dm.send_voltages(voltages)

# 等价于
dm = NLightDM()
try:
    dm.open()
    dm.send_voltages(voltages)
finally:
    dm.close()
```

---

## 4. 设备注册与发现

### 4.1 设备注册表 (DeviceRegistry)

```python
from ao_shaping.drivers import DeviceRegistry, DeviceType

# 创建或获取全局注册表
registry = get_global_registry()

# 注册设备
registry.register(
    camera,
    alias="main_cam",
    tags=["imaging", "high_speed"],
    priority=10,
    auto_connect=True
)

# 按类型查找
cameras = registry.find_by_type(DeviceType.CAMERA)

# 按标签查找
imaging_devices = registry.find_by_tag("imaging")

# 批量操作
registry.connect_all()  # 连接所有设备
states = registry.get_all_twin_states()  # 获取数字孪生状态

# 导出配置
config = registry.export_config()  # 用于持久化
```

### 4.2 设备标识

- **device_id**: 自动生成的 UUID
- **alias**: 用户友好的名称（可选）
- **tags**: 标签列表，用于分组

---

## 5. 设备类型实现

### 5.1 相机 (CCD)

**基类**: `BaseCamera`

```python
# 必需实现的方法
def initialize(self) -> None: ...
def open(self) -> None: ...
def close(self) -> None: ...
def reset_exposure_time(self, time_ms: int) -> int: ...
def reset_window(self, center, size) -> tuple: ...
def get_numpy_image(self, n_sample, skip_first) -> np.ndarray: ...
def enable_auto_exposure(self, enable, mode) -> bool: ...
@staticmethod
def get_cam_list() -> list: ...
```

**实现子类**：
- `DahengCamera` (gxipy SDK)
- `MiiCamDevice` (MiiCam SDK)

### 5.2 变形镜 (DM)

**基类**: `BaseDM`

```python
# 核心方法
def send_voltages(self, voltages: np.ndarray, voltage_scale: float = 1.0) -> None: ...
def get_voltages(self) -> np.ndarray: ...
def set_actuator_voltage(self, index: int, voltage: float) -> None: ...
```

**实现子类**：
- `NLightDM` (UDP 通信)

### 5.3 SLM (Spatial Light Modulator)

**基类**: 继承 `Device`

```python
# 核心方法
def display_pattern(self, phase_pattern: np.ndarray) -> None: ...
def clear_display(self) -> None: ...
```

**实现子类**：
- `SantecSLM200` (SDK)
- `SantecSLM200Visa` (VISA)

### 5.4 波前传感器 (WFS)

**基类**: 继承 `Device`

```python
# 核心方法
def get_wavefront(self) -> np.ndarray: ...
def get_rms(self) -> float: ...
```

**实现子类**：
- `ThorlabWFS`

---

## 6. 仿真与测试设备

### 6.1 三层设备体系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          设备体系架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     仿真层 (sim/)      ┌──────────────┐              │
│  │   硬件层      │  ←─ 物理模型 ──→      │   仿真层      │              │
│  │  (实装设备)   │     Digital Twin     │  (数字孪生)   │              │
│  └──────────────┘                       └──────────────┘              │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌──────────────┐                       ┌──────────────┐              │
│  │ NLightDM     │ ◄───────对应─────────►  │ SimulateDM   │              │
│  │ CameraStream │ ◄───────对应─────────►  │ SimulatedCCD │              │
│  │ SantecSLM200 │ ◄───────对应─────────►  │ SimulatedSLM │              │
│  │ ThorlabWFS   │ ◄───────对应─────────►  │ SimulatedWFS │              │
│  └──────────────┘                       └──────────────┘              │
│                                                                         │
│  ┌──────────────┐     Mock 层            ┌──────────────┐              │
│  │   测试桩      │  ←─ 无物理 ──→        │   Mock 设备    │              │
│  │  (Stub)      │     仅接口            │   (桩实现)    │              │
│  └──────────────┘                       └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 仿真层 vs Mock 设备 vs 硬件

| 类型 | 前缀 | 用途 | 物理模型 | API 兼容性 |
|------|------|------|----------|-----------|
| **硬件** | 无 | 真实设备控制 | 实际光学系统 | 完整 |
| **Simulated** | `Simulated` | 集成测试/数字孪生 | 基于物理 ✨ | 兼容硬件 API |
| **Mock** | `Mock` | 单元测试 | 无 (桩函数) | 部分兼容 |

### 6.3 孪生对应关系详解

#### 6.3.1 相机 (CCD)

```
┌────────────────────────────────────────────────────────────────┐
│                      CCD 设备家族                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────┐                                       │
│  │  BaseCamera (ABC)   │  ← 统一抽象基类                        │
│  └──────────┬──────────┘                                       │
│             │                                                   │
│    ┌────────┴────────┬───────────────────┐                   │
│    ▼                  ▼                   ▼                   │
│ ┌──────────┐   ┌────────────┐    ┌──────────────────┐        │
│ │Daheng    │   │  MiiCam    │    │    BaseCamera    │        │
│ │Camera    │   │  Device    │    │   (供 Simulated  │        │
│ │StreamMgr │   │            │    │    继承的抽象类)   │        │
│ └──────────┘   └────────────┘    └────────┬─────────┘        │
│     │                  │                  │                   │
│     └──────────────────┴──────────────────┴─┐                 │
│                                             ▼                  │
│                                    ┌───────────────┐            │
│                                    │  SimulatedCCD │            │
│                                    │  (继承BaseCam)│            │
│                                    │  物理: 噪声   │            │
│                                    │  + 特征生成   │            │
│                                    └───────────────┘            │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MockCamera (桩实现 - 不继承 BaseCamera)                  │   │
│  │ 用途: 仅用于接口测试, 无物理模拟                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**SimulatedCCD 关键特性**：

```python
from ao_shaping.drivers import SimulatedCCD

cam = SimulatedCCD(
    cam_id=0,
    resolution=(1024, 1024),  # 分辨率
    noise_level=5.0,          # 噪声标准差 (ADU)
    random_seed=42            # 可复现性
)

with cam:
    # 与真实相机相同的 API
    img = cam.get_numpy_image(n_sample=10, skip_first=True)
    
    # 仿真特有功能
    cam.reset_exposure_time(50)        # 调整曝光
    cam.reset_window(center=(512, 512), size=(512, 512))  # ROI
```

#### 6.3.2 变形镜 (DM)

```
┌────────────────────────────────────────────────────────────────┐
│                      DM 设备家族                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────┐                                       │
│  │     DM (ABC)        │  ← 统一抽象基类                        │
│  └──────────┬──────────┘                                       │
│             │                                                   │
│    ┌────────┴────────┐                                          │
│    ▼                 ▼                                          │
│ ┌──────────┐   ┌────────────┐                                   │
│ │ NLightDM │   │ SimulateDM │                                   │
│ │ (硬件)   │   │ (仿真)     │                                   │
│ └──────────┘   └─────┬──────┘                                   │
│        │             │                                           │
│        │        ┌────┴────┐                                      │
│        │        │ 事务模型  │                                      │
│        │        │ • 电压→变形转换                                   │
│        │        │ • 相邻致动器耦合 (邻接矩阵)                         │
│        │        │ • 电压爬升限制 (max_iter_diff)                     │
│        │        │ • 热噪声模拟                                      │
│        │        └──────────┘                                      │
│        │                                                         │
│        └─────────────────────────────────────────┐              │
│                                                  ▼              │
│                                    ┌────────────────────────┐  │
│                                    │ NLightDM 对比           │  │
│                                    │ ─────────────────────── │  │
│                                    │ send_voltages()         │  │
│                                    │ get_voltages()          │  │
│                                    │ set_hv()                │  │
│                                    │ ← 完全兼容 →            │  │
│                                    │ SimulateDM 对比         │  │
│                                    │ ─────────────────────── │  │
│                                    │ send_voltages()         │  │
│                                    │ get_voltage_history()    │  │
│                                    │ get_deformation_history()│  │
│                                    └────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

**SimulateDM 关键特性**：

```python
from ao_shaping.drivers.dm import SimulateDM

dm = SimulateDM(
    max_iter_diff=20,      # 最大电压变化速度
    max_neighbor_diff=0,   # 相邻致动器最大电压差
    keep_when_exit=True,   # 退出时保持状态
    noise_level=0.01        # 变形噪声
)

with dm:
    # 与真实 NLightDM 相同的 API
    voltages = np.random.randn(64)  # 64 通道
    dm.send_voltages(voltages, 0.1)
    
    # 仿真特有功能
    history = dm.get_voltage_history()    # 电压历史
    deformations = dm.get_deformation_history()  # 变形历史
    dm.clear_history()                     # 清除历史
```

#### 6.3.3 SLM (Spatial Light Modulator)

```
┌────────────────────────────────────────────────────────────────┐
│                      SLM 设备家族                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────┐                                       │
│  │   Device (ABC)     │ ← 基础抽象基类                        │
│  └──────────┬──────────┘                                       │
│             │                                                   │
│    ┌────────┴────────┐     ┌───────────────────────┐          │
│    ▼                 ▼     ▼                       ▼          │
│ ┌───────┐   ┌─────────┐  ┌─────────┐     ┌──────────┐        │
│ │Santec │   │Simulated│  │Simulated│     │MockSLM  │        │
│ │SLM200 │   │SLM      │  │Lens     │     │         │        │
│ │(硬件) │   │(仿真)   │  │(光学元件)│     │(桩)     │        │
│ └───────┘   └────┬────┘  └────┬────┘     └──────────┘        │
│                  │            │                                 │
│                  │      ┌────┴────┐                            │
│                  │      │         │                            │
│                  │      ▼         ▼                            │
│                  │  ┌────────┐ ┌────────────┐                 │
│                  │  │Aperture│ │Wavefront   │                 │
│                  │  │(光阑)  │ │Processor   │                 │
│                  │  │        │ │(波前处理)   │                 │
│                  │  └────────┘ └────────────┘                 │
│                  │                                             │
│                  └─────────────────────────────────────────┐  │
│                      继承层次                                 ▼  │
│                                  ┌─────────────────────────┐  │
│                                  │ Device ← SimulatedDevice │  │
│                                  │       ← OpticalDevice    │  │
│                                  │       ← WavefrontProc    │  │
│                                  │       ← SimulatedSLM     │  │
│                                  └─────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

**SimulatedSLM 关键特性**：

```python
from ao_shaping.drivers import SimulatedSLM

slm = SimulatedSLM(
    resolution=(1920, 1080),
    phase_range=2 * np.pi,       # 相位调制范围
    wavelength=1064.0,           # 工作波长 (nm)
    enable_noise=False           # 是否添加相位噪声
)

with slm:
    # 设置相位图案 (与真实 SLM 相同)
    phase = compute_phase_pattern()  # 用户计算
    slm.set_phase(phase)
    
    # 处理波前
    output_wave = slm.process(input_wave)
    
    # 仿真特有
    current_phase = slm.get_phase()   # 读取当前相位
    slm.clear()                       # 清除
```

#### 6.3.4 其他仿真设备

| 设备类 | 仿真物理模型 | 对应硬件 |
|--------|-------------|----------|
| `SimulatedLaser` | 功率衰减、模式 | 激光器 |
| `SimulatedTurbulentScreen` | 大气湍流相位屏 | 模拟湍流 |
| `SimulatedThermalScreen` | 热波动相位屏 | 热噪声 |
| `SimulatedATP` | ATP 跟踪误差 | -- |

### 6.4 仿真基类层次结构

```
Device (ABC)
    │
    ├── SimulatedDevice (新增)
    │   ├── enable_noise: bool
    │   ├── random_seed: int
    │   ├── set_seed()
    │   ├── set_noise()
    │   └── _generate_noise()
    │       │
    │       └── OpticalDevice
    │           ├── wavelength: float
    │           ├── set_input() / get_output()
    │           └── process() [抽象]
    │               │
    │               └── WavefrontProcessor
    │                   ├── npix: int
    │                   ├── dpix: float
    │                   ├── set_phase() / get_phase()
    │                   └── process() [实现]
    │                       │
    │                       ├── SimulatedSLM
    │                       └── SimulatedLens
    │
    └── 具体硬件设备
        ├── NLightDM
        ├── CameraStreamManager
        └── SantecSLM200
```

### 6.5 数字孪生同步机制

```python
# ---------------------------
# 数字孪生同步工作流
# ---------------------------

# 1. 获取设备孪生状态
twin_state = device.get_twin_state()
# 返回:
# {
#     "device_id": "...",
#     "device_type": "CAMERA",
#     "manufacturer": "Simulation",
#     "model": "SimulatedCCD",
#     "state": "READY",
#     "parameters": {
#         "exposure_time": {"value": 20.0, "unit": "ms"},
#         "resolution": {"value": [1024, 1024], "unit": "px"}
#     },
#     "capabilities": [...]
# }

# 2. 批量获取
registry = get_global_registry()
all_states = registry.get_all_twin_states()

# 3. 序列化保存 (用于恢复)
import json
with open("twin_state.json", "w") as f:
    json.dump(all_states, f, indent=2)

# 4. 从孪生状态恢复
with open("twin_state.json") as f:
    saved_states = json.load(f)
registry.sync_from_twin_states(saved_states)
```

### 6.6 仿真场景示例

```python
# ============================================================
# 场景 1: 集成测试 (使用 Simulated 设备)
# ============================================================
from ao_shaping.drivers import SimulatedCCD, SimulatedSLM
from ao_shaping.drivers.sim import SimulatedLaser

# 构建设备系统
with SimulatedLaser() as laser, \
     SimulatedSLM() as slm, \
     SimulatedCCD(cam_id=0) as ccd:
    
    # 设置激光功率
    laser.set_power(100.0)
    
    # 加载 SLM 相位
    phase = compute_correction_phase()
    slm.set_phase(phase)
    
    # 捕获图像
    img = ccd.get_numpy_image(n_sample=5)
    
    # 验证
    assert img.shape == (1024, 1024)

# ============================================================
# 场景 2: 有/无缝切换 (硬件 ↔ 仿真)
# ============================================================
def get_camera(use_simulation: bool = False):
    """动态选择相机设备"""
    if use_simulation:
        return SimulatedCCD(resolution=(1024, 1024))
    else:
        # 硬件不可用时自动回退
        from ao_shaping.drivers import CameraStreamManager
        if CameraStreamManager is None:
            return SimulatedCCD(resolution=(1024, 1024))
        return CameraStreamManager(cam_id=0)

# 使用
cam = get_camera(use_simulation=False)  # 自动选择

# ============================================================
# 场景 3: 参数化仿真研究
# ============================================================
import numpy as np
from ao_shaping.drivers import SimulatedCCD

# 噪声影响研究
noise_levels = [1.0, 5.0, 10.0, 20.0]
results = []

for noise in noise_levels:
    cam = SimulatedCCD(noise_level=noise, random_seed=42)
    with cam:
        img = cam.get_numpy_image(n_sample=100)
        snr = np.mean(img) / np.std(img)
        results.append({"noise": noise, "snr": snr})

# ============================================================
# 场景 4: 单元测试 (使用 Mock 设备)
# ============================================================
from ao_shaping.drivers import MockCamera

def test_algorithm():
    """仅测试算法逻辑, 不关心设备细节"""
    cam = MockCamera()
    img = cam.get_numpy_image()  # 返回固定值
    
    result = process_image(img)  # 测试你的算法
    assert result is not None
```

---

## 7. 导入模式与可选依赖

### 7.1 Facade 统一入口

```python
# 基础类（始终可用）
from ao_shaping.drivers import Device, DeviceState, DeviceType

# 硬件驱动（SDK 不可用时为 None）
from ao_shaping.drivers import CameraStreamManager
if CameraStreamManager is None:
    print("Daheng SDK 未安装，使用 SimulatedCCD 替代")

# Mock 设备（始终可用）
from ao_shaping.drivers import MockCamera, MockDM

# Simulated 设备（始终可用）
from ao_shaping.drivers import SimulatedCCD, SimulatedSLM
```

### 7.2 可选依赖处理

```python
# drivers/__init__.py 中的模式
try:
    from .ccd.daheng import CameraStreamManager
    __all__ += ["CameraStreamManager"]
except (ImportError, NameError) as e:
    logging.getLogger(__name__).warning(f"CameraStreamManager not available: {e}")
    CameraStreamManager = None
```

---

## 8. 设计特点分析

### 8.1 优点

| 特点 | 说明 |
|------|------|
| **统一的接口** | 所有设备继承 `Device`，上层代码无需关心具体硬件型号 |
| **灵活的回退机制** | 硬件 SDK 不可用时自动使用 Simulated 设备 |
| **数字孪生原生** | `get_twin_state()` / `sync_from_twin()` 支持状态同步 |
| **参数统一管理** | `register_parameter()` 提供参数验证与元数据 |
| **状态追踪** | 完整的状态机与事件日志 |
| **批量操作** | `connect_all()`, `health_check_all()` 等批量方法 |
| **上下文管理器** | `with Device() as d:` 自动资源管理 |

### 8.2 设计模式

| 模式 | 应用位置 |
|------|----------|
| **Abstract Factory** | `Device` 基类 |
| **Registry** | `DeviceRegistry` 设备注册发现 |
| **Strategy** | 硬件/仿真设备的运行时切换 |
| **Observer** | 数据回调 `_data_callbacks` |
| **Memento** | 数字孪生状态序列化 |

---

## 9. 二次开发指南

### 9.1 添加新品牌设备

**步骤 1**: 创建设备驱动类

```python
# src/ao_shaping/drivers/ccd/newbrand.py
from ao_shaping.drivers.ccd.base import BaseCamera, CameraError

class NewBrandCamera(BaseCamera):
    """NewBrand 相机驱动"""
    
    def __init__(self, cam_id: int = 0, exposure_time_ms: int = 20):
        super().__init__(cam_id, exposure_time_ms)
    
    def initialize(self) -> None:
        # 初始化 SDK 连接
        self.cam = NewBrandSDK.open(self.cam_id)
        self._sn = self.cam.get_serial()
        self.cam_width, self.cam_height = self.cam.get_resolution()
    
    def open(self) -> None:
        self.initialize()
    
    def close(self) -> None:
        if self.cam:
            self.cam.close()
    
    # ... 实现其他抽象方法 ...
```

**步骤 2**: 在 `__init__.py` 中导出

```python
# src/ao_shaping/drivers/__init__.py
try:
    from .ccd.newbrand import NewBrandCamera
    __all__ += ["NewBrandCamera"]
except ImportError:
    pass  # SDK 不可用
```

**步骤 3**: 在 Facade 中添加

```python
# 自动添加到导出列表
if 'NewBrandCamera' in dir():
    __all__.append('NewBrandCamera')
```

### 9.2 添加新设备类型

**步骤 1**: 在 `DeviceType` 枚举中添加类型

```python
# device_base.py
class DeviceType(Enum):
    # ... existing ...
    SPECTROMETER = auto()  # 新增
```

**步骤 2**: 创建设备基类（如需要特定接口）

```python
# drivers/spectrometer/base.py
from ao_shaping.drivers import Device

class BaseSpectrometer(Device):
    device_type = DeviceType.SPECTROMETER
    
    @abstractmethod
    def get_spectrum(self) -> np.ndarray:
        pass
```

**步骤 3**: 实现具体设备类

### 9.3 扩展数字孪生功能

```python
class MyDevice(Device):
    def get_twin_state(self) -> dict:
        # 扩展标准状态
        state = super().get_twin_state()
        # 添加自定义状态
        state["custom_data"] = self._custom_sensors
        return state
    
    def sync_from_twin(self, twin_state: dict) -> None:
        # 同步自定义数据
        super().sync_from_twin(twin_state)
        if "custom_data" in twin_state:
            self._custom_sensors = twin_state["custom_data"]
```

### 9.4 添加设备事件回调

```python
# 注册回调
def on_device_registered(device_id: str, device: Device):
    print(f"新设备注册: {device_id}")

registry.on_device_registered(on_device_registered)

# 数据流回调
def on_image_received(data_type: str, data):
    print(f"收到图像: {data.shape}")

camera.register_data_callback(on_image_received)
```

---

## 10. 最佳实践

### 10.1 设备初始化

```python
# 推荐：通过注册表管理
registry = get_global_registry()

with SimulatedCCD(cam_id=0) as cam:
    registry.register(cam, alias="test_cam", tags=["test"])
    
# 或：直接使用
cam = SimulatedCCD()
cam.open()
try:
    img = cam.get_numpy_image()
finally:
    cam.close()
```

### 10.2 错误处理

```python
try:
    dm.open()
except DeviceBusyError:
    print("设备忙，等待重试")
except DeviceError as e:
    print(f"设备错误: {e}")
```

### 10.3 资源清理

```python
# 正确：使用上下文管理器
with NLightDM() as dm:
    dm.send_voltages(v)

# 或：显式 try/finally
dm = NLightDM()
try:
    dm.open()
    # ...
finally:
    dm.close()
```

---

## 11. 测试策略

### 11.1 单元测试

```python
# 使用 Mock 设备
from ao_shaping.drivers import MockCamera, MockDM

def test_my_algorithm():
    cam = MockCamera()
    with cam:
        img = cam.get_numpy_image()
    
    # 验证逻辑
    assert img.shape == (1024, 1024)
```

### 11.2 集成测试

```python
# 使用 Simulated 设备
from ao_shaping.drivers import SimulatedCCD, SimulatedSLM

def test_beam_shaping():
    slm = SimulatedSLM()
    ccd = SimulatedCCD()
    
    with slm, ccd:
        # 仿真光束传播
        pattern = compute_phase()
        slm.display_pattern(pattern)
        img = ccd.get_numpy_image()
```

---

## 12. 配置与持久化

### 12.1 导出/导入配置

```python
# 导出
config = registry.export_config()
with open("devices.json", "w") as f:
    json.dump(config, f)

# 导入
with open("devices.json") as f:
    config = json.load(f)
# 遍历 config["devices"] 重新创建设备并注册
```

---

## 13. 快速参考

| 任务 | 代码 |
|------|------|
| 导入设备基类 | `from ao_shaping.drivers import Device, DeviceState` |
| 导入相机 | `from ao_shaping.drivers import CameraStreamManager` |
| 导入仿真相机 | `from ao_shaping.drivers import SimulatedCCD` |
| 检查硬件可用性 | `if CameraStreamManager is None: ...` |
| 创建注册表 | `registry = get_global_registry()` |
| 获取所有相机 | `cameras = registry.find_by_type(DeviceType.CAMERA)` |
| 批量连接 | `results = registry.connect_all()` |
| 获取数字孪生状态 | `states = registry.get_all_twin_states()` |