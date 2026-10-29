# Chapter 4: Thorlabs WFS 波前传感器（Python SDK）

## Core Idea
WFS 是"状态位 + 调用顺序 + 数组布局"三套约定叠加的设备：返回值 0=成功、正数=警告、负数=错误必须抛；71 个 API 有严格前置链（质心→波前→Zernike）；一切 2D 数组是 `[Y][X]` 主序——绑定层一个 `[::-1]` 用错就全转置。

## Frameworks Introduced
- **ctypes 绑定五约定**
  - When to use: 修改/扩展 `_sdk_bindings.py` 或任何 WFS 绑定代码。
  - How: (1) 类型别名 `ViStatus=c_int32, ViSession=c_ulong, ViRsrc=ViChar256`；(2) 2D 数组 `ArrFloat = ndpointer(shape=[80,80][::-1])` 保持 Y 主序，图像 `uint8 (512,512)`，直接传 ndarray；(3) `load_dll()` 逐函数声明 `restype/argtypes`；(4) 无 session 的查询（`WFS_GetInstrumentListLen`）传 `VI_NULL()`（`c_ulong()`），不是 `None`；(5) 状态码 `0=成功 / 正=警告记录不抛 / 负=抛 WfsError`，每次调用必查返回值。
- **标准测量工作流（take_image）**
  - When to use: 一切波前/Zernike 测量的前置。
  - How: 软件触发 `WFS_SW_TRIGGER` 下每帧 `TakeSpotfieldImage` → `CalcSpotsCentrDiaIntens(calcDias=1, calcIntens=1)`；`dynamicNoiseCut=True` 时 `SetBlackLevelOffset + CutImageNoiseFloor` 自适应；返回 `n_sample/mean/min/max/rms` 记录 dict。
- **波前 + Zernike 链（get_wavefront / get_zernike）**
  - When to use: 需要波前图或 Noll 系数。
  - How: `get_zernike(order)` 走 `WFS_ZernikeLsf`，前置 `CalcSpotToReferenceDeviations(0)`；**调用前必须 `wfs.pupil = wfs.optimize_pupil()`**（硬编码 pupil（瞳面） 会污染 LSF 拟合，实测假 tip/tilt |z|=4.6~12.8λ，修正后 ≤0.22λ，R²=0.9603）；orders 0=auto 或 2..10，系数数组 67 长（index 1..66，µm，RoC=coeff[5]）。`get_wavefront(cancel_tile=True)` 先扣 Noll 2/3 阶再返回。
- **WFS_STATUS 17 位位图**
  - When to use: 故障诊断。
  - How: `WFS_GetStatus` 返回位图，关键位：0x01 CON(连接)、0x02 PTH(光瞳过亮)、0x04 PTL(光瞳过暗→调曝光/增益)、0x10 SCL(光斑裁剪)、0x400 PUD(光瞳未定义)、0x8000 LOS(断连)。

## Key Concepts
- **ViStatus 三态**: 0 成功 / 正数警告（`VI_WARN_NSUP_*`）/ 负数错误（`WfsError`）。
- **VI_NULL()**: 无 session 查询的句柄占位符，`c_ulong()`。
- **ArrFloat / MAX_SPOTS[::-1]**: Y 主序 2D 数组绑定的唯一正确写法。
- **动态噪声切割**: `SetBlackLevelOffset` + `CutImageNoiseFloor` 自适应黑电平。
- **user reference (.ref)**: `WFS_{serial}_{mla}_{res}.ref` 参考文件，需已配 MLA+分辨率，0.0 质心=NaN。
- **Highspeed 模式**（仅 WFS10/20）: 只剩质心流，图像/波前/Zernike/直径/束宽全不可用。
- **RoC = Z[5]**: Noll 第 5 阶（活塞 1、倾斜 2/3、离焦 4、RoC 5），与 `zernike_utils` 约定一致。

## Mental Models
- Use "警告是黄灯，错误是红灯" 的心智：正数记 log 继续，负数立即 `WfsError`，0 才放行。
- 把 调用链 理解为 流水线阀门：`Calc*` 是计算、`Get*` 是取结果，先算后取，跳阀门取到的是垃圾。
- Think of 数组布局 as 照片：相机图像与物理方向可能镜像，`Flip2DArray` 前先验证。

## Anti-patterns
- **2D 数组按 (X,Y) 传**：绑定必须 `MAX_SPOTS[::-1]` 保持 Y 主序，直传 (80,80) 直序会整幅转置。
- **`Get*` 前不 `Calc*`**：`GetSpotIntensities` 不计算只取值、`GetSpotDiameters` 需 `calcDias=1`、`CalcReconstrDeviations` 必须先于 `CalcWavefront(1/2)`、`ZernikeLsf` 必须先于 `CalcFourierOptometric`（fourierOrder 仅 2/4/6）。
- **硬编码 pupil（瞳面）（如 (0,0,8mm)）**：边界无效子孔径污染 `WFS_ZernikeLsf`，产生巨大假 tip/tilt；必须 `wfs.pupil = wfs.optimize_pupil()` 自动获取并**写回**（`optimize_pupil()` 只计算不写回，返回元组必须显式赋值）。
- **`==0` 判成功**：正数警告会被当错误抛，或负数错误被当成功吞掉。
- **`GetInstrumentListLen` 传 `None`**：必须 `VI_NULL()`。
- **Highspeed 下调非质心函数**：图像/波前/Zernike 等均不可用，会报错。
- **`AverageImage` 当同步用**：非阻塞，需轮询 `ready`。
- **`ConvertWavefrontWaves` out 与 in 同缓冲**：不得 alias。
- **黑电平未最小化就量束宽/质心**：`CalcBeamCentroidDia` 前先 `SetBlackLevelOffset`。

## Code Examples
```python
# 类型与数组绑定（_sdk_bindings.py 核心）
ViStatus = c_int32; ViSession = c_ulong; ViRsrc = c_char * 256
MAX_SPOTS = [80, 80]
ArrFloat  = np.ctypeslib.ndpointer(shape=MAX_SPOTS[::-1])   # Y 主序，勿改 (80,80)
ArrImg    = np.ctypeslib.ndpointer(dtype=np.uint8, shape=(512, 512))

def load_dll():
    dll = cdll.LoadLibrary(r"...WFS_64.dll")
    dll.WFS_init.restype = ViStatus
    dll.WFS_init.argtypes = [ViRsrc, ViBoolean, ViBoolean, POINTER(ViSession)]
    return dll

def VI_NULL():
    return c_ulong()   # GetInstrumentListLen 的 handle 占位
```
- 演示：绑定三件套（类型/数组/句柄占位），改绑定时照抄。

```python
def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> dict:
    # 软件触发：每次 TakeSpotfieldImage 触发新测量
    for i in range(1, n_sample + 1):
        self._lib.WFS_TakeSpotfieldImage(...)
        self._lib.WFS_CalcSpotsCentrDiaIntens(..., calcDias=1, calcIntens=1)
    # dynamicNoiseCut → SetBlackLevelOffset + CutImageNoiseFloor 自适应
    return history  # 含 n_sample/mean/min/max/rms
```
- 演示：标准测量骨架——先 Take 后 Calc，返回记录 dict。

## Reference Tables
| WFS_STATUS 位 | 掩码 | 含义 |
|---|---|---|
| CON / PTH / PTL | 0x01/0x02/0x04 | 已连接 / 光瞳过亮 / 光瞳过暗（调曝光/增益） |
| SCL / ZFL / ZFH | 0x10/0x40/0x80 | 光斑裁剪 / 零帧上下限 |
| ATR / CFG / PUD | 0x100/0x200/0x400 | 自动触发 / 相机配置错 / 光瞳未定义 |
| URF / HSP / MIS | 0x2000/0x4000/0x8000* | 用户参考文件 / highspeed / MLA 已设（0x8000 为 LOS 断连） |

| 调用前置链 | 规则 |
|---|---|
| 光斑参数 | `TakeSpotfieldImage` → `CalcSpotsCentrDiaIntens(calcDias=1)` → `GetSpotDiameters/Intensities` |
| 波前 | `CalcReconstrDeviations` → `CalcWavefront(1/2)` → `GetWavefront` |
| Zernike | `optimize_pupil` 写回 → `CalcSpotToReferenceDeviations(0)` → `WFS_ZernikeLsf` |
| 球面参考 | `DoSphericalRef` → `ZernikeLsf` → `CalcReconstrDeviations(doSphReference=1)` → `SetReferencePlane` |

## Key Takeaways
1. 状态码三态：0 放行 / 正记录 / 负抛 `WfsError`，每次必查。
2. 数组一律 Y 主序 `MAX_SPOTS[::-1]`。
3. 先 Calc 后 Get，前置链不可跳。
4. pupil（瞳面） 必须 `optimize_pupil()` 自动获取并写回，严禁硬编码。
5. Highspeed 只剩质心流，其余函数禁用。
6. RoC=Z[5]，Noll 约定与 `zernike_utils` 对齐。

## Connects To
- **ch01 (大恒)**: WFS 与相机同为光路探测器，但 WFS 走状态位+调用链，相机走 Feature 模型。
- **ch06 (驱动层)**: 仿真 WFS 的保真度红线（SH 测不到 piston、tilt 串扰）与本章硬件 WFS 对照。
