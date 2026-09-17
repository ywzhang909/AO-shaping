# agent.md - Thorlabs WFS Python SDK 驱动编写指南

**Generated:** 2026-09-15
**Source:** Thorlabs WFS 官方手册 (`docs/thorlab-wfs/Manual/WFS.html` + `WFS_files/FunctWFS_*.html`, 共 71 个 API 页面)
**Target:** 指导 agent 编写 WFS 的 Python SDK 驱动, 输出与 `src/ao_shaping/drivers/wfs/thorlab/` 现有实现一致

## 1. Purpose & Scope

本指南是编写 Thorlabs WFS (Wavefront Sensor) Python SDK 驱动的唯一事实来源 (Single Source of Truth)。它把官方手册的全部 71 个 `WFS_*` C 函数映射为 Python ctypes 绑定与 `ThorlabWFS` 驱动类, 并固化本项目已验证的绑定/驱动约定。

**Scope In:**
- ctypes 绑定层 (`_sdk_bindings.py` 模式): `load_dll()` + 逐函数 `restype/argtypes`
- 驱动类 (`driver.py` 模式): `ThorlabWFS(Device)`, 生命周期, 参数注册, 配置持久化
- 全部 71 个 SDK 函数的签名、前置条件、型号限制、Highspeed 可用性
- 状态位 (WFS_STATUS) 与错误处理矩阵

**Scope Out:**
- 相机/CCD 驱动, 变形镜驱动, 上层优化器 (见 `src/ao_shaping/optimizer/wf/`)
- 光源、快门等其他 Thorlabs 硬件

## 2. File Map & Architecture

### 2.1 目标文件布局

```
src/ao_shaping/drivers/wfs/
├── __init__.py            # 导出 ThorlabWFS, MlaRes, WFSParams, WfsError
├── AGENTS.md              # 设备域文档 (入口: 本驱动)
├── thorlab_wfs.py         # [旧] ThorlabWFS 兼容入口 — 注意: 完整实现 (1887 行, 与 thorlab/driver.py 同源, 2026-09 逐段校验一致); wfs/__init__.py 实际从此文件导出
├── _thorlab_wfs.py        # [旧] 内部实现 (legacy, 勿改)
└── thorlab/               # 现代实现 (本指南的目标结构)
    ├── __init__.py        # 导出 ThorlabWFS, MlaRes, WFSParams, WfsError
    ├── driver.py          # ThorlabWFS 主驱动类 (Device 子类)
    └── _sdk_bindings.py   # ctypes 绑定层 (load_dll, 类型别名, WFS_STATUS)
```

### 2.2 三层架构

```
┌─────────────────────────────────────────────────────┐
│ 调用方 (runner / optimizer)                          │
│   from ao_shaping.drivers.wfs.thorlab import ThorlabWFS
├─────────────────────────────────────────────────────┤
│ driver.py — ThorlabWFS(Device)                      │
│   open()/close()/is_connected()/get_hardware_info() │
│   take_image()/get_wavefront()/get_zernike() ...    │
├─────────────────────────────────────────────────────┤
│ _sdk_bindings.py — ctypes                            │
│   load_dll() → CDLL("WFS_64.dll")                    │
│   逐函数 restype/argtypes, ArrFloat (Y,X 顺序)        │
├─────────────────────────────────────────────────────┤
│ WFS_64.dll  (VISA Win64)                            │
│   C:\Program Files\IVI Foundation\VISA\Win64\Bin\   │
└─────────────────────────────────────────────────────┘
```

### 2.3 手册 ↔ 驱动 ↔ 绑定映射

| 手册页面 | 签名摘要 | 绑定状态 | driver.py 方法 |
|---|---|---|---|
| `FunctWFS_init` | `WFS_init(rsrc, idQuery, reset, &session)` | ✅ 已绑定 | `open()` |
| `FunctWFS_ConfigureCam` | `WFS_ConfigureCam(h, mla, res, &resX, &resY)` | ✅ 已绑定 | `select_mla()` |
| `FunctWFS_SelectMla` | `WFS_SelectMla(h, mlaIndex)` | ✅ 已绑定 | `select_mla()` |
| `FunctWFS_GetMlaData` | `WFS_GetMlaData(h, mlaIndex, name, 6×&f64)` | ✅ 已绑定 | `get_mla_name()` |
| `FunctWFS_TakeSpotfieldImage` | `WFS_TakeSpotfieldImage(h)` | ✅ 已绑定 | `take_image()` |
| `FunctWFS_GetSpotfieldImageCopy` | `WFS_GetSpotfieldImageCopy(h, buf, &w, &h)` | ✅ 已绑定 | `get_spotfiled_image()` |
| `FunctWFS_CalcSpotsCentrDiaIntens` | `WFS_CalcSpotsCentrDiaIntens(h, calcDias, calcIntens)` | ✅ 已绑定 | `get_spots_statics()` |
| `FunctWFS_CalcWavefront` | `WFS_CalcWavefront(h, type, pol, arr[Y][X])` | ✅ 已绑定 | `get_wavefront()` |
| `FunctWFS_GetStatus` | `WFS_GetStatus(h, &status)` | ✅ 已绑定 | `open()`/轮询 |
| `FunctWFS_SaveUserRefFile` | `WFS_SaveUserRefFile(h)` | ✅ 已绑定 | `save_user_ref()` |
| `FunctWFS_LoadUserRefFile` | `WFS_LoadUserRefFile(h)` | ✅ 已绑定 | `load_user_ref()` |
| `FunctWFS_ZernikeLsf` | `WFS_ZernikeLsf(h, &orders, coeff[], &roC)` | ⚠️ typed 绑定注释, 未声明函数直接调用可用 (2026-09 硬件实测) | `get_zernike()` (driver.py:1431) |

\* `WFS_ZernikeLsf` 的 `restype/argtypes` 声明在 `_sdk_bindings.py` 中被注释 (§8 坑 #5), 但驱动绕过绑定层直接调用未声明函数 (ctypes 宽松传参) 仍返回合理系数 — 2026-09 硬件实测确认。勿再声称 "必须自行实现"; 建议补绑 argtypes 以启用类型检查。

## 3. ctypes Binding Conventions

### 3.1 类型别名 (`_sdk_bindings.py`)

```python
from ctypes import c_uint8, c_int16, c_int32, c_float, c_double, c_ulong, c_long, c_char, POINTER, byref, cdll, c_bool

ViStatus  = c_int32     # 所有 WFS_* 返回值
ViBoolean = c_bool
ViSession = c_ulong
ViUInt8   = c_uint8
ViInt16   = c_int16
ViInt32   = c_int32
ViReal32  = c_float
ViReal64  = c_double
ViString  = c_char_p
ViChar256 = c_char * 256
ViChar512 = c_char * 512
ViRsrc    = ViChar256
```

### 3.2 数组约定 (关键!)

```python
MAX_SPOTS = [80, 80]                       # 每维最大光斑数 (逐型号可能更小)
ArrFloat = np.ctypeslib.ndpointer(shape=MAX_SPOTS[::-1])   # 注意 Y, X 顺序!
ArrImg   = np.ctypeslib.ndpointer(dtype=np.uint8, shape=(512, 512))
WFS_BUFFER_SIZE = 256                      # 字符串缓冲
```

- **所有 2D 浮点数组 (质心/强度/直径/波前/参考位置) 均为 `[MAX_SPOTS_Y][MAX_SPOTS_X]` (Y 主序, row-major)**。`ArrFloat` 用 `MAX_SPOTS[::-1]` 定义正是为了匹配 `[Y][X]` 布局 — 勿改成 `(80, 80)` 直序, 否则矩阵会被转置。
- 图像数组为 `uint8 (512, 512)` (逐行扫描)。
- 传入 numpy 数组时直接传 ndarray (`np.ctypeslib.ndpointer` 自动转换); 无需 `np2c` (仅 `c_int32` 数组用)。

### 3.3 `load_dll()` 模式

```python
def load_dll():
    dll = cdll.LoadLibrary(r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\WFS_64.dll")
    dll.WFS_init.restype = ViStatus
    dll.WFS_init.argtypes = [ViRsrc, ViBoolean, ViBoolean, POINTER(ViSession)]
    # ... 逐函数声明 (见 §4 完整清单) ...
    return dll
```

### 3.4 句柄与空指针

```python
def VI_NULL():
    return c_ulong()
```

- `WFS_GetInstrumentListLen(handle, ...)`: **handle 参数必须传 `VI_NULL()`** (仪器列表查询无需 session)。
- 只读数组参数用 `c_float`/`ArrFloat` 直接传 ndarray; 输出标量用 `byref` + 预置 ctypes 变量。

### 3.5 状态码约定

- `0 = VI_SUCCESS`; **正数 = 警告 (warning)**; **负数 = 错误 (error)**。
- **每次调用后必须检查返回值**。警告可记录不抛出 (如 `VI_WARN_NSUP_RESET`), 错误必须抛 `WfsError` (见 `handle_error`)。

## 4. Full API Inventory (71 函数 × 10 族)

绑定状态图例: ✅ = `load_dll()` 已声明 | ⚠️ = 已注释/未声明 (需自行补绑) | — = 手册有但绑定层未列出

### 族 1: Session 生命周期 (7)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_init` | `(ViRsrc rsrc, ViBoolean idQuery, ViBoolean reset, ViP*Session)` | ✅ | 打开会话; rsrc = `USB::0x1313::0x0000::<id>` |
| `WFS_close` | `(ViSession)` | ✅ | 关闭会话, 释放 |
| `WFS_reset` | `(ViSession)` | ✅ | 恢复上电默认态; 不支持时返 `VI_WARN_NSUP_RESET` |
| `WFS_self_test` | `(ViSession, ViPInt16 result, ViChar msg[])` | ✅ | 0 = 通过; msg 需 ≥256 字节缓冲 |
| `WFS_revision_query` | `(ViSession, ViChar drvRev[], ViChar fwRev[])` | ✅ | 驱动/固件版本 |
| `WFS_error_query` | `(ViSession, ViPInt32 code, ViChar msg[])` | ✅ | 取最近错误码+文本 |
| `WFS_error_message` | `(ViSession, ViStatus err, ViChar msg[])` | ✅ | 错误码 → 文本 (handle_error 用) |

### 族 2: 仪器发现 (4)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_GetInstrumentListLen` | `(ViSession NULL, ViPInt32 len)` | ✅ | handle 传 `VI_NULL()` |
| `WFS_GetInstrumentListInfo` | `(h, listIndex, &devID, &inUse, name, sn, resName)` | ✅ | 枚举; 名称/序列号/资源名各 ≥256 缓冲 |
| `WFS_GetInstrumentInfo` | `(h, name, sn, version, resource)` | ✅ | 当前会话硬件信息 |
| `WFS_GetStatus` | `(ViSession, ViPInt32 status)` | ✅ | 17-bit 状态寄存器 (见 §7) |

### 族 3: 相机配置 (11)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_ConfigureCam` | `(h, mlaIdx, resolution, &sizex, &sizey)` | ✅ | 逐型号分辨率表: 1268/1024/760/540/512/320×对应 |
| `WFS_SetExposureTime` | `(h, ViReal64 tMs, ViPReal64 tAct)` | ✅ | 请求 ms → 返回实际 ms (量化/钳位) |
| `WFS_GetExposureTime` | `(h, ViPReal64 tAct)` | ✅ | 实际曝光 ms |
| `WFS_GetExposureTimeRange` | `(h, &min, &max, &step)` | ✅ | 单位 ms; 取决于分辨率 |
| `WFS_SetMasterGain` | `(h, ViReal64 gain, ViPReal64 gainAct)` | ✅ | 线性增益; WFS10 最小 1.5, WFS20 固定 1.0 |
| `WFS_GetMasterGain` | `(h, ViPReal64 gainAct)` | ✅ | 实际增益 |
| `WFS_GetMasterGainRange` | `(h, &min, &max)` | ✅ | 增益范围 |
| `WFS_SetBlackLevelOffset` | `(h, ViInt32 offset)` | ✅ | 0-255; 影响质心/束宽精度 |
| `WFS_GetBlackLevelOffset` | `(h, ViPInt32 offset)` | ✅ | 相机黑电平 |
| `WFS_SetAoi` | `(h, cX, cY, sX, sY)` (mm) | — | AOI 外的光斑不参与 Zernike/波前计算; 全 0 = 最大区域; 原点=图像中心 |
| `WFS_GetAoi` | `(h, &cX, &cY, &sX, &sY)` (mm) | — | 绑定层注释 "undocumented" 已过时 — 手册已文档化, **可补绑** |

### 族 4: 触发 (4)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_SetTriggerMode` | `(h, ViInt32 mode)` | ✅ | OFF/HL/LH/SW; WFS150/300/30/40/31 中 OFF≡SW |
| `WFS_GetTriggerMode` | `(h, ViPInt32 mode)` | ✅ | 见上 |
| `WFS_SetTriggerDelay` | `(h, ViInt32 usSet, ViPInt32 usAct)` | — | 附加触发延时 (µs); 先查 Range |
| `WFS_GetTriggerDelayRange` | `(h, &min, &max, &incr)` | — | µs; **绑定层只注释了拼错的 `SetTriggerDelayRange`, 真名是 Get/Set 对** |

触发模式枚举: `WFS_HW_TRIGGER_OFF=0` (连续最快) / `WFS_HW_TRIGGER_HL` (高→低沿) / `WFS_HW_TRIGGER_LH` (低→高沿) / `WFS_SW_TRIGGER` (软件触发, 每次调用采集启动)。

### 族 5: Highspeed 模式 (3)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_SetHighspeedMode` | `(h, on, winX, winY, winSize)` | ✅ | **仅 WFS10/WFS20**; 触发=ON 时禁用软件触发 |
| `WFS_GetHighspeedWindows` | `(h, &cntX, &cntY, &szX, &szY, startX[], startY[])` | ⚠️ | 需先开 Highspeed; startX/Y 数组 = `MAX_SPOTS` 大小 |
| `WFS_CheckHighspeedCentroids` | `(h)` | ✅ | 质心越窗时返 `WFS_ERROR_HIGHSPEED_WINDOW_MISMATCH` (截断不可靠) |

### 族 6: MLA / 光瞳 / 参考面 (8)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_GetMlaCount` | `(h, ViPInt32 count)` | ✅ | 已标定 MLA 数 (≥1) |
| `WFS_GetMlaData` | `(h, mlaIdx, name, 6×&f64)` | ✅ | pitch/focal/offset/corr; idx 0..count-1; **调用后不自动激活** |
| `WFS_GetMlaData2` | `(h, mlaIdx, name, 10×&f64)` | ✅ | 可换 MLA 完整标定; 添加 astig0/45, rot, pitch 修正 |
| `WFS_SelectMla` | `(h, ViInt32 mlaIndex)` | ✅ | 激活 MLA |
| `WFS_SetPupil` | `(h, cX, cY, dX, dY)` (mm) | ✅ | 中心 ±5.0 mm; 直径 0.1-10.0 mm |
| `WFS_GetPupil` | `(h, &cX, &cY, &dX, &dY)` (mm) | ✅ | 实际光瞳 |
| `WFS_SetReferencePlane` | `(h, ViInt32 idx)` | ✅ | `0`=internal, `1`=user |
| `WFS_GetReferencePlane` | `(h, ViPInt32 idx)` | ✅ | 同上 |

### 族 7: 图像采集 (10)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_TakeSpotfieldImage` | `(h)` | ✅ | 单帧采集 (软件触发模式触发新测量) |
| `WFS_TakeSpotfieldImageAutoExpos` | `(h, &exp, &gain)` | ✅ | 自动曝光; HSV 过低/过高错误由返回值指示 |
| `WFS_AverageImage` | `(h, avgCount 1-256, &ready)` | ✅ | **非阻塞**: 轮询 ready==1 取平均帧; 非 Highspeed |
| `WFS_AverageImageRolling` | `(h, avgCount 1-256, reset)` | ✅ | 滚动平均: `((n-1)·avg + img)/n`; `ConfigureCam` 重置; 非 Highspeed |
| `WFS_GetSpotfieldImage` | `(h, buf)` | — | 绑定层注释 "left out" — 仅 Copy 版已绑 |
| `WFS_GetSpotfieldImageCopy` | `(h, buf, &w, &h)` | ✅ | **唯一已绑的图像输出**; 512×512 uint8 |
| `WFS_GetLine` | `(h, line, sizeX)` | ✅ | 行剖面; 逐型号限宽: WFS150/300=1280, WFS10=640, WFS20=1440, WFS30=1936, WFS40=2048, WFS31=1920 |
| `WFS_GetLineView` | `(h, lineMin[], lineMax[])` | ✅ | 每列 min/max; 数组 = 图像宽; 非 Highspeed |
| `WFS_CalcImageMinMax` | `(h, &min, &max, &satPct)` | ✅ | 图像 min/max + 饱和像素 %; 非 Highspeed |
| `WFS_CutImageNoiseFloor` | `(h, ViInt32 limit)` | ✅ | 像素 < limit 清零; limit 1-256; 勿设太高以免抹掉光斑 |

### 族 8: 光斑分析 (9)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_CalcSpotsCentrDiaIntens` | `(h, calcDias, calcIntens)` | ✅ | **先决函数**: 其他 Get* 依赖其结果 |
| `WFS_GetSpotCentroids` | `(h, arrX[Y][X], arrY[Y][X])` | ✅ | 质心, 像素坐标 |
| `WFS_GetSpotIntensities` | `(h, arr[Y][X])` | ✅ | 光强, 任意单位 |
| `WFS_GetSpotDiameters` | `(h, arrDX[Y][X], arrDY[Y][X])` | ✅ | 需 `CalcSpotsCentrDiaIntens(calcDias=1)`; 像素; 非 Highspeed |
| `WFS_GetSpotDiaStatistics` | `(h, &min, &max, &mean)` | ✅ | **需先 `CalcWavefront`**; 非 Highspeed |
| `WFS_GetSpotReferencePositions` | `(h, arrRX[Y][X], arrRY[Y][X])` | ✅ | 参考位置 (当前参考面) |
| `WFS_GetSpotDeviations` | `(h, arrDX[Y][X], arrDY[Y][X])` | ✅ | 相对参考的偏差 |
| `WFS_CalcBeamCentroidDia` | `(h, &cX, &cY, &dX, &dY)` (mm) | ✅ | **二阶矩**公式束宽; 对黑电平敏感; 非 Highspeed |
| `WFS_CalcMeanRmsNoise` | `(h, &mean, &rms)` | ✅ | ImageBuf 像素噪声; 非 Highspeed |
| `WFS_CheckHighspeedCentroids` | 见族 5 | ✅ | — |

### 族 9: 波前重建 / Zernike (9)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_CalcSpotToReferenceDeviations` | `(h, convertWaves)` | ✅ | 0 = µm, 1/2 = 波长; **`CalcWavefront` 前置** |
| `WFS_CalcWavefront` | `(h, type, pol, arr[Y][X])` | ✅ | type: 0=SpotToRef, 1=Reconstr, 2=Reconstr+iso; 输出 µm (或换算后) |
| `WFS_CalcWavefrontStatistics` | `(h, &min,&max,&mean,&sd,&p2v,&rms)` | ✅ | 波前统计 |
| `WFS_ZernikeLsf` | `(h, &orders, coeff[67], &radiusOfCurv)` | ⚠️ 调用可用 (绑定注释) | orders 0=auto / 2-10; modes 6/10/15/21/28/36/45/55/66; coeff[1..66], **RoC = coeff[5]**; 前置 `CalcSpotToReferenceDeviations(convertWaves=0)` (µ), 结果 µm |
| `WFS_CalcReconstrDeviations` | `(h, defer, type, doSphRef, flags[67], &fitErrMean, &fitErrStd)` | ✅ | flags 索引 1..66 (int); doSphRef 1 → 球差参考; arcmin; **先于 `CalcWavefront(type=1/2)`** |
| `WFS_CalcFourierOptometric` | `(h, zernikeOrders, fourierOrder, &M, &J0, &J45, &sph, &cyl, &axis)` | ✅ | **先 `ZernikeLsf`**; fourierOrder ∈ {2,4,6}; 屈光度 |
| `WFS_ConvertWavefrontWaves` | `(h, wavelengthNm, wavefront[], out[])` | ✅ | λ ∈ [300, 1100] nm; µm ↔ 波长 |
| `WFS_Flip2DArray` | `(h, in[], out[])` | ✅ | 镜像翻转; **out 不得 alias in** |
| `WFS_GetXYScale` | `(h, arrXStep[], arrYStep[])` | ✅ | 每光斑 μm/像素比例 |

### 族 10: 校准 / 用户参考 (6)

| 函数 | C 签名摘要 | 绑定 | 说明 |
|---|---|---|---|
| `WFS_SetSpotsToUserReference` | `(h)` | ✅ | 当前实测光斑位置 → 用户参考 |
| `WFS_SetCalcSpotsToUserReference` | `(h, refType, refX[Y][X], refY[Y][X])` | ✅ | refType: 0=REL (±像素相对工厂标定), 1=ABS (绝对像素) |
| `WFS_CreateDefaultUserReference` | `(h)` | ✅ | 依据内部工厂标定生成默认参考 |
| `WFS_SaveUserRefFile` | `(h)` | ✅ | 存自动文件名到参考目录 (见 §6.5) |
| `WFS_LoadUserRefFile` | `(h)` | ✅ | **MLA + 分辨率必须先配置好**; 0.0 位 → NaN (未检测光斑) |
| `WFS_DoSphericalRef` | `(h)` | ✅ | 球面参考: 前置 = 球面波测量 + `ZernikeLsf` + `CalcReconstrDeviations(doSphReference=1)`, 再 `SetReferencePlane` 激活 |

## 5. Driver Class Design

```python
from ao_shaping.drivers.device_base import Device, DeviceState, DeviceType
from loguru import logger

class ThorlabWFS(Device):
    def __init__(self, resource_name: str | None = None) -> None:
        super().__init__()
        self._device_id = f"wfs_{serial}"          # 连接后填写
        self._dll = load_dll()
        self._session: ViSession = VI_NULL()
        self._serial = ""
        self._device_name = ""
        self._is_open = False
        self._mla_name = ""
        self._res_index = 0
        self._stable_samples: list[dict] = []
        # 参数注册 (Device.register_parameter)
        self.register_parameter(WFSParams.exposure_time_ms)
        self.register_parameter(WFSParams.master_gain)
```

### 5.1 生命周期

| 方法 | 职责 | 关键 SDK 调用 |
|---|---|---|
| `open()` | 发现 → 会话 → 配置 → READY | `GetInstrumentListInfo` → `init` → `load_config()` → 应用配置 → `select_mla()` → `set_ref_plane()` → `optimize_exposure_time_and_gain()` → 设光瞳 → `_set_state(READY)` |
| `close()` | 保存配置 + 关闭 | `save_config()`; `WFS_close`; `_set_state(DISCONNECTED)` |
| `is_connected()` | 会话有效且状态 READY | 返回 `self._session != VI_NULL() and self._state == READY` |
| `get_hardware_info()` | 硬件信息 dict | `GetInstrumentInfo` / init 时填写的 `_serial`, `_device_name` |

`open()` 关键点:
- 先 `GetInstrumentListLen(VI_NULL(), &len)` 枚举, 再 `GetInstrumentListInfo` 定位目标序列号 (或取第一个在位仪器)。
- **`init` 成功后任何失败都必须 `WFS_close` 清理** — 用 try/except 包裹初始化段。
- 配置加载失败 (无配置文件) 回退到硬件默认, 继续 `select_mla(0)` (默认 MLA 索引 0)。

### 5.2 配置持久化

- `load_config()/save_config()`: 用 `ConfigHandler` (json) 存于 `data/wfs_configs/`, **按序列号键控** (`wfs_{serial}.json`)。
- `open()` 中 `load_config()` 成功后调用 `WFS_CONFIG.apply_from_config(self, config)` 把已保存参数回灌。

### 5.3 参数对象

```python
@dataclass
class WFSParams(DeviceParameter):   # 或独立 dataclass + DEFAULTS
    mla_index: int = 0
    exposure_time: float = DEFAULTS.WFS_EXPOSURE_TIME      # ms
    high_speed: bool = False
    use_custom_ref: bool = False
```

用户可调参数通过 `register_parameter()` 暴露 (如 `exposure_time_ms`, `master_gain`)。

### 5.4 错误处理

```python
def handle_error(self, err: int, no_raise: bool = False) -> tuple[int, str]:
    if err == 0:
        return 0, ""
    msg = ""
    if self._session:
        buf = ViChar256()
        self._dll.WFS_error_message(self._session, err, byref(buf))
        msg = buf.value.decode(errors="replace")
    logger.error(f"WFS error: {err} - {msg}")
    if not no_raise:
        raise WfsError(f"{err}: {msg}")
    return err, msg
```

- 每个 SDK 调用后: `err = dll.WFS_x(...); self.handle_error(err)` (除非调用方需处理警告码)。

## 6. Measurement Workflows

### 6.1 标准测量 (软件触发/连续)

```
open() → take_image() → calc dev/stat/pupil → wavefront → zernike
```

```python
def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> dict:
    # 软件触发: 每次 TakeSpotfieldImage 触发新测量 (WFS_SW_TRIGGER)
    # 1..n_sample 帧, 每帧: TakeSpotfieldImage → CalcSpotsCentrDiaIntens(calcDias=1, calcIntens=1)
    # dynamicNoiseCut=True → SetBlackLevelOffset + CutImageNoiseFloor 自适应
    # 记录 n_sample, mean/min/max 强度, 返回 history dict
```

### 6.2 自动曝光

```
GetExposureTimeRange(&min,&max,&step) → (若初始曝光超界则钳位)
TakeSpotfieldImageAutoExpos(&exp,&gain) → 检查返回码 (亮/暗告警)
循环 ≤ MAX_AUTOEXPOSE_ATTEMPTS(10) 次: 校验实际曝光在 [0.002, 86] ms 内
SetExposureTime/SetMasterGain 显式应用实际值
```

### 6.3 Highspeed (仅 WFS10/20)

```
SetHighspeedMode(h, on=1, winX, winY, winSize)   # 触发必须已设为 HW 触发
loop: TakeSpotfieldImage → CheckHighspeedCentroids → GetSpotCentroids
# 不支持: 图像输出/波前/直径/平均/Line/AOI/BeamCentroid
```

### 6.4 Wavefront + Zernike

```python
def get_wavefront(self, cancel_tile: bool = False):
    take_image(...)
    CalcSpotToReferenceDeviations(h, convertWaves=0)   # µm
    CalcWavefront(h, type=1, pol=0, arr[Y][X])         # 重建型
    CalcWavefrontStatistics(h, &min,&max,&mean,&sd,&p2v,&rms)
    # cancel_tile=True 时先去除 tip/tilt (从 Zernike 系数扣 2/3 阶)
    # 稳定采样: stable_sample_enable 时方差过滤, 返回 min/max/diff/mean/rms
```

```python
def get_zernike(self, zernike_order: int):
    # 已实现 (driver.py:1431): 直接调用 _lib.WFS_ZernikeLsf (绑定 argtypes 注释,
    # 未声明函数仍可调, 2026-09 硬件实测可用). 前置: CalcSpotToReferenceDeviations(0).
    # ⚠️ 结果依赖 pupil 正确性 — 调用前必须 wfs.pupil = wfs.optimize_pupil():
    #    硬编码 pupil 会污染 LSF 拟合, 产生巨大假 tip/tilt (2026-09 实测见 §6.6/§8 #21)
    # orders: 0=auto 或 2..10; 系数数组 67 长 (index 1..66), µm; RoC = coeff[5]
```

### 6.5 用户参考 (.ref 文件)

```
SaveUserRefFile(h)  →  %USERPROFILE%\Documents\Thorlabs\Wavefront Sensor\Reference\
                       自动文件名: WFS_{serial}_{mla_name}_{res_idx}.ref
                        (WFS10_/WFS20_/WFS30_/WFS40_/WFS31_ 前缀随型号)
LoadUserRefFile(h)  →  同目录同命名, 需先配好 MLA+分辨率; 0.0 质心 → NaN
```

驱动层封装:
- `get_mla_name()`: `GetMlaData(idx, name, ...)` 取 MLA 名; 失败回退 `_try_get_mla_name_fallback` (轮询 1..15 候选索引, `MAX_MLA_INDICES=16`)。
- `save_user_ref(backup_dir)`: `SetSpotsToUserReference` → `SaveUserRefFile` → 备份副本至 `backup_dir`。
- `load_user_ref(backup_path)`: 从备份复制回参考目录 → `LoadUserRefFile` → `SetReferencePlane(1)`。

### 6.6 光瞳优化

```
take_image → CalcSpotsCentrDiaIntens → CalcBeamCentroidDia(&cX,&cY,&dX,&dY)
            → SetPupil(cX, cY, dX, dY)   # mm
注意: 黑电平必须先 SetBlackLevelOffset 最小化, 否则质心/束宽失真
```

**⚠️ 2026-09 实测固化 (本机 SLM200+WFS M01219666, 532nm):**

- **pupil 必须由 `optimize_pupil()` 自动获取并显式写回**: `wfs.pupil = wfs.optimize_pupil()`。
  `optimize_pupil()` **只计算并返回, 不调用 `WFS_SetPupil`** — 忘记写回等于没设 pupil
  (驱动旧 docstring "d≤0 自动 optimize" 描述不实, §8 坑 #22, 已修正)。
- **勿硬编码 pupil**: 硬编码 `(0,0,8mm)` 与真实光束不符时, 边界无效子孔径
  (MLA 27×27 网格外围约 4~6 个) 污染 `WFS_ZernikeLsf` 全孔径 LSF 拟合 →
  **巨大假 tip/tilt, 实测 |z|=4.6~12.8λ 且随倾斜幅度非单调**。
- **本机实测参考 pupil** (2026-09-14): 中心 ≈ `(-0.148, +0.148)` mm,
  直径 ≈ `3.62 × 3.88` mm。
- 修正后: 倾斜阶梯 A=0.10~3.20λ 读出 0.0061~0.2169λ, 线性度 **R²=0.9603**
  (同轮 plane 度量 R²=0.8795 — 小倾斜端受噪声/高阶像差干扰;
  硬编码 pupil 那轮 plane R²=0.9974 属巧合假象, 并非真实指标)。
- zernike LSF 是最干净的主度量: `get_zernike` 直接由 spot deviations 拟合,
  正确 pupil 下 tip/tilt (Noll 2,3) 噪声 ≤0.22λ; plane 度量在小倾斜端不可靠。

## 7. Status-Bit & Error Matrix

### 7.1 WFS_STATUS 位定义 (17 bits, `WFS_GetStatus`)

| 位 | 掩码 | 名称 | 含义 |
|---|---|---|---|
| 0 | 0x0001 | CON | connected — 已连接 |
| 1 | 0x0002 | PTH | pupil too high — 光瞳过亮 |
| 2 | 0x0004 | PTL | pupil too low — 光瞳过暗 |
| 3 | 0x0008 | HAL | high ambient light |
| 4 | 0x0010 | SCL | spots clipped |
| 5 | 0x0020 | ZFL | zero frame limit |
| 6 | 0x0040 | ZFH | zero frame higher |
| 7 | 0x0080 | ATR | auto trigger |
| 8 | 0x0100 | CFG | camera error/config |
| 9 | 0x0200 | PUD | pupil undefined |
| 10 | 0x0400 | SPC | spots clipped |
| 11 | 0x0800 | RDA | RDA active |
| 12 | 0x1000 | URF | user reference file |
| 13 | 0x2000 | HSP | highspeed |
| 14 | 0x4000 | MIS | MLA index set |
| 15 | 0x8000 | LOS | lost connection |
| 16 | 0x10000 | FIL | filter error |

### 7.2 错误处置表

| 情形 | 返回值/状态 | 处置 |
|---|---|---|
| 调用成功 | `0` | 继续 |
| 光瞳过暗 (PTL) / 过亮 (PTH) | status bit 1/2 | 调整曝光/增益 (自动曝光循环) |
| Highspeed 质心越窗 | `WFS_ERROR_HIGHSPEED_WINDOW_MISMATCH` (负) | 重设窗口/重新标定 |
| Reset/self_test/revision 不支持 | 正数警告 (`VI_WARN_NSUP_*`) | 记录 warning, 不抛异常 |
| 其他负错误码 | 负 | `handle_error` 抛 `WfsError` |
| `.ref` 中 0.0 质心 | 内存中 = NaN | 上层对 NaN 光斑跳过处理 |

## 8. Pitfalls & Anti-Patterns (实测固化)

| # | 坑 | 规则 |
|---|---|---|
| 1 | **数组 Y,X 顺序** | 一切 2D 数组是 `[Y][X]` (row-major)。绑定用 `MAX_SPOTS[::-1]`, 传 numpy 保持 Y 主序 |
| 2 | **`GetSpotfieldImage` 被 "left out"** | 只用 `GetSpotfieldImageCopy`; 如需非副本语义需自行补绑 |
| 3 | **`GetAoi`/`SetAoi` 注释 "undocumented" 已过时** | 手册现文档化 (mm, 原点=图像中心, 全 0 = 最大区域); 新代码可补绑 |
| 4 | **绑定层 `SetTriggerDelayRange` 是拼写错误** | 手册真名 `SetTriggerDelay` / `GetTriggerDelayRange`; 补绑时用正确名 |
| 5 | **`ZernikeLsf` typed 绑定被注释** | 驱动直接调用未声明函数仍可用 (2026-09 硬件实测); `restype/argtypes` 建议补绑以启用类型检查 |
| 6 | **`CalcReconstrDeviations` 必须先于 `CalcWavefront(type=1/2)`** | 顺序错误 → 错误/垃圾波前 |
| 7 | **`ZernikeLsf` 必须先于 `CalcFourierOptometric`** | 且 `fourierOrder` 仅 2/4/6 |
| 8 | **`GetSpotDiameters` 需要 `CalcSpotsCentrDiaIntens(calcDias=1)`** | 参数必须传 1, 否则返回空 |
| 9 | **`GetSpotDiaStatistics` 需要先 `CalcWavefront`** | 顺序依赖 |
| 10 | **`GetSpotIntensities` 只取值不计算** | 必须先成功 `CalcSpotsCentrDiaIntens` |
| 11 | **`DoSphericalRef` 前置链** | 球面波测量 → `ZernikeLsf` → `CalcReconstrDeviations(doSphReference=1)` → `SetReferencePlane` 激活 |
| 12 | **`LoadUserRefFile` 需要已配 MLA+分辨率** | 且文件存在 (`WFS_{serial}_{mla}_{res}.ref`); 0.0 → NaN |
| 13 | **黑电平影响束宽/质心** | `CalcBeamCentroidDia` 前 `SetBlackLevelOffset` 最小化 |
| 14 | **`ConvertWavefrontWaves` out 不得 alias in** | `Flip2DArray` 同理 |
| 15 | **`GetInstrumentListLen` 传 `VI_NULL()`** | 不是 `None` 也不是 session |
| 16 | **Highspeed 禁用很多函数** | 图像/Average/Line/AOI/束宽/直径/波前/Zernike 均不可用; 仅质心流 |
| 17 | **`AverageImage` 非阻塞** | 轮询 `ready`, 勿假设同步完成 |
| 18 | **`ConfigureCam` 重置平均过程** | `AverageImageRolling` 在换分辨率后重新累积 |
| 19 | **软件触发下 `OFF≡SW`** | WFS150/300/30/40/31 无连续/软件区分 |
| 20 | **`WFS_Flip2DArray`/图像显示** | 相机图像与物理方向可能镜像; 检查后再定是否翻转 |
| 21 | **硬编码 pupil 污染 Zernike 拟合** | pupil 必须 `wfs.pupil = wfs.optimize_pupil()` 自动获取并写回; 硬编码 `(0,0,8mm)` 使边界无效子孔径污染 `WFS_ZernikeLsf` → 巨大假 tip/tilt (实测 \|z\|=4.6\~12.8λ, 修正后 ≤0.22λ, R²=0.9603); 本机实测 pupil 中心 (-0.148,+0.148)mm / 直径 3.62×3.88mm |
| 22 | **`optimize_pupil()` 只计算不写回** | 它不调用 `WFS_SetPupil`; 返回元组必须显式赋给 `wfs.pupil` 才生效 (驱动旧 docstring "d≤0 自动 optimize" 描述不实, 已修正; 若返回值非有限/超出 SDK 范围会触发 `logger.warning` — `driver.py:1028` / `thorlab_wfs.py:1032`) |

## 9. Conventions & Verification Checklist

### 9.1 项目约定 (来自 root AGENTS.md)

- Python 3.12+, `from __future__ import annotations` 首行, 绝对导入 (`from ao_shaping.drivers.wfs.thorlab import ...`), `|` 类型语法, `*Error` 异常后缀 (`WfsError`), `loguru` 日志 (惰性 `logger.info("... {}", x)`), `Device` 基类 + `DeviceState`/`DeviceType` + `_set_state()`, 参数注册 `register_parameter()`.
- 无 `print()`, 无宽泛 `except`, 无相对导入。

### 9.2 编写完成后的验收清单

1. ✅ `_sdk_bindings.py` 覆盖全部 71 函数 (可直接 grep `dll.WFS_` 计数; 注明注释/缺绑项)
2. ✅ `ArrFloat` 保持 `MAX_SPOTS[::-1]` (Y,X)
3. ✅ `open()` 的异常路径全部 `WFS_close` 清理
4. ✅ 每个调用检查返回值; 负值抛 `WfsError`, 正值警告
5. ✅ 参数注册与配置持久化 (序列号键控) 存在
6. ✅ `take_image`/`get_wavefront`/`get_zernike` 返回 dict 含 `n_sample/mean/min/max/rms` 等记录字段
7. ✅ Highspeed 分支禁用清单核对 (族 5/7/8/9 的 "非 Highspeed" 标注)
8. ✅ Zernike 系数索引 1..66、RoC = Z[5] (Noll 约定与 zernike_utils 一致)
   — 2026-09 实验确认: `z_um[2:4]/0.532` (Noll 2,3 = tip/tilt) 与物理倾斜幅度线性匹配,
   R²=0.9603 (0.532nm 波长下 1λ ≈ 0.532µm)
9. ✅ pupil 用 `wfs.pupil = wfs.optimize_pupil()` 自动获取并写回, 勿硬编码
   — 实测 (0,0,8mm) 硬编码污染 ZernikeLsf; 见 §6.6 / §8 #21
10. ✅ 无硬件测试: pytest 中用 `MockWFS` (`mock_devices.py#open` @814) 或 `pytest.skip("Requires WFS hardware")`

### 9.3 参考资料

- 官方手册: `docs/thorlab-wfs/Manual/WFS.html` (索引), `WFS_files/FunctWFS_*.html` (71 页)
- 官方 Python 示例 (外部): `https://github.com/nvladimus/WFS` (read-average-wavefront.ipynb)
- 参考笔记本要点: init → 读参考/新建参考 → TakeSpotfieldImage → CalcSpotsCentrDiaIntens → CalcWavefront 顺序