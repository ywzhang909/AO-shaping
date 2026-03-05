# 类图 (Class Diagram)

## 设备驱动类继承关系

```mermaid
classDiagram
    %% 基础枚举
    class DeviceType {
        <<enumeration>>
        CAMERA
        SLM
        DM
        WFS
        STAGE
        LASER
        FILTER
        OTHER
    }
    
    class DeviceState {
        <<enumeration>>
        UNKNOWN
        DISCONNECTED
        CONNECTING
        READY
        BUSY
        ERROR
        CALIBRATING
    }
    
    %% 数据类
    class DeviceParameter {
        +str name
        +Any value
        +type value_type
        +float | None min_value
        +float | None max_value
        +str unit
        +str description
        +bool writable
        +validate(value) bool
    }
    
    class DeviceCapability {
        +str name
        +str description
        +list~str~ parameters
        +type | None return_type
    }
    
    class DeviceMetadata {
        +str device_id
        +DeviceType device_type
        +str manufacturer
        +str model
        +str serial_number
        +str firmware_version
        +str hardware_version
        +dict connection_info
        +datetime registration_time
        +datetime | None last_seen
    }
    
    %% 异常类
    class DeviceError {
        <<exception>>
    }
    
    class DeviceNotFoundError {
        <<exception>>
    }
    
    class DeviceBusyError {
        <<exception>>
    }
    
    %% 抽象基类
    abstract class Device {
        +ClassVar device_type: DeviceType
        +ClassVar manufacturer: str
        +ClassVar model: str
        +ClassVar version: str
        -str _device_id
        -DeviceState _state
        -str | None _error_message
        -dict~str, DeviceParameter~ _parameters
        -dict~str, DeviceCapability~ _capabilities
        -list~Callable~ _data_callbacks
        -DeviceMetadata _metadata
        +__init__(device_id: str)
        +open() None
        +close() None
        +is_connected() bool
        +get_hardware_info() dict
        +__enter__() Device
        +__exit__(...) None
        +register_parameter(...) None
        +get_parameter(...) DeviceParameter
        +set_parameter_value(...) bool
        +register_capability(...) None
        +has_capability(...) bool
        +get_twin_state() dict
        +sync_from_twin(...) None
        +get_status() dict
        +health_check() tuple
    }
    
    %% DM 抽象基类
    abstract class DM {
        <<abstract>>
        +channel: int
        +transform(cmd)
        +send(cmd)
        +open()
        +close()
        +get_actuator_positions()
    }
    
    %% 继承关系
    Device <|-- DM
    Device <|-- SLM
    Device <|-- WFS
    Device <|-- CCD
    Device <|-- TM
    
    DM <|-- NLightDM
    DM <|-- SimulateDM
    
    SLM <|-- SantecSLM200
    SLM <|-- SantecSLM200_VISA
    
    WFS <|-- ThorlabsWFS
    
    CCD <|-- MiiCAM
    CCD <|-- DahengCamera
    
    TM <|-- SerialPortFSM
    
    DeviceError <|-- DeviceNotFoundError
    DeviceError <|-- DeviceBusyError
    
    Device o-- DeviceParameter
    Device o-- DeviceCapability
    Device o-- DeviceMetadata
```

---

## 算法模块类图

```mermaid
classDiagram
    %% 优化器基类
    class Optimizer {
        <<abstract>>
        +slm: SLM
        +ccd: CCD
        +optimize(iterations) dict
        +calibrate() None
    }
    
    class WavefrontController {
        <<abstract>>
        +dm: DM
        +wfs: WFS
        +correct(target_rms) dict
        +calibrate() None
    }
    
    %% 波前控制实现
    WavefrontController <|-- DMWFSController
    WavefrontController <|-- LrWFS
    WavefrontController <|-- RLWFS
    
    %% 无波前传感实现
    Optimizer <|-- GreedyCAM
    Optimizer <|-- LrWFLess
    Optimizer <|-- ADCDMAdam
    Optimizer <|-- PhaseRetrieve
    
    %% 工具类
    class Zernike {
        +n_terms: int
        +generate(coefficients) ndarray
        +fit(phase_map) ndarray
    }
    
    class Wavefront {
        +phase_map: ndarray
        +rms: float
        +pv: float
        +tilt: tuple
    }
    
    class PhasePattern {
        +grating_pattern(...) ndarray
        +blazed_grating(...) ndarray
        +gs_iterate(...) ndarray
    }
    
    class SpotsCalc {
        +centroid(image) tuple
        +spot_radius(...) float
        +correlation(...) float
    }
```

---

## 驱动类详细类图

```mermaid
classDiagram
    %% SLM 驱动
    class SLM {
        <<abstract>>
        +resolution: tuple
        +write_pattern(pattern) None
        +get_status() dict
    }
    
    class SantecSLM200 {
        +device_type = DeviceType.SLM
        +manufacturer = "Santec"
        +model = "SLM-200"
        +resolution = (1920, 1080)
        +open() None
        +close() None
        +write_pattern(pattern) None
        +get_status() dict
        +calibrate() dict
    }
    
    class SLMPatternHelper {
        +grating(...) ndarray
        +blazed_grating(...) ndarray
        +random_phase(...) ndarray
    }
    
    class SLMCalibration {
        +calibrate() dict
        +save(path) None
        +load(path) None
    }
    
    class SantecSLM200Error {
        <<exception>>
    }
    
    SLM <|-- SantecSLM200
    SantecSLM200 o-- SLMPatternHelper
    SantecSLM200 o-- SLMCalibration
    SantecSLM200Error --|> Exception
    
    %% DM 驱动
    class NLightDM {
        +V_Max = 5.0
        +V_Min = 0.0
        +n_actuators = 97
        +channel: int
        +open() None
        +close() None
        +set_voltage(voltages) None
        +get_voltage() ndarray
        +get_info() dict
    }
    
    class SimulateDM {
        +n_actuators: int
        +set_voltage(voltages) None
        +get_response() ndarray
    }
    
    class NLightDMError {
        <<exception>>
    }
    
    %% WFS 驱动
    class WFS {
        <<abstract>>
        +get_wavefront() WavefrontData
        +get_zernike() ndarray
    }
    
    class ThorlabsWFS {
        +device_type = DeviceType.WFS
        +manufacturer = "Thorlabs"
        +open() None
        +close() None
        +get_wavefront() dict
        +get_zernike() ndarray
    }
    
    class ThorlabsWFSError {
        <<exception>>
    }
    
    %% CCD 驱动
    class CCD {
        <<abstract>>
        +resolution: tuple
        +set_exposure(time) None
        +set_gain(gain) None
        +capture() ndarray
    }
    
    class MiiCAM {
        +device_type = DeviceType.CAMERA
        +manufacturer = "Mii"
        +open() None
        +close() None
        +set_exposure(time) None
        +set_gain(gain) None
        +capture() ndarray
    }
    
    class DahengCamera {
        +device_type = DeviceType.CAMERA
        +manufacturer = "Daheng"
        +open() None
        +close() None
        +capture() ndarray
    }
```

---

*本文档由 AI 自动生成，最后更新于 2026-03-05*
