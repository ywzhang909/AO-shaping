# ioc-wfs — Thorlabs 波前传感器 (WFS) 软 IOC

基于 caproto 的 Thorlabs WFS 软 IOC。无物理 WFS 时自动降级为**离线模式**(仅注册全部 PV,写控制 PV 返回错误),便于集成测试与 WSL 端到端验证。

## 驱动依赖

- 驱动:`src/ao_shaping/drivers/wfs/thorlab/driver.py`(`ThorlabWFS`)
- SDK:`WFS_64.dll` 硬编码路径 `C:\Program Files\IVI Foundation\VISA\Win64\Bin\WFS_64.dll`(Thorlabs 软件安装后存在)。DLL 缺失时 `ThorlabWFS()` 构造抛 `OSError` → IOC 离线。
- 运行时需 `PYTHONPATH` 含 `src`(另含 `common`、`ioc-wfs\src`)。

## PV 表(前缀 `WFS-01:`)

| PV | 类型 | 读/写 | 说明 |
|----|------|-------|------|
| Acquire | int | 写 | 写 1 采集一帧:take_image + spotfield + spots 统计 |
| ExposureTime | double | 写 | 曝光时间(ms);驱动钳位 [0.002, 86] |
| HighSpeed | enum | 写 | off/on 高速模式(**注意**:spot 统计需高速模式关闭) |
| MlaIndex | int | 写 | MLA 分辨率 320/512/768/1024/1280 → select_mla |
| PupilX / PupilY / PupilDiameter | double | 写 | 瞳面中心/直径(mm);任一写入即整体应用 |
| AutoExpose | int | 写 | 写 1 触发 `optimize_exposure_time_and_gain()`,刷新 ExposureTime/Gain |
| Connected | enum | 只读 | Disconnected/Connected |
| SerialNumber / DeviceName | string | 只读 | open() 后来自驱动 |
| Image | waveform int32 | 只读 | 最近一帧 spotfield(上限 512×512 = 262144) |
| ImageWidth / ImageHeight | int | 只读 | 实际 spotfield 尺寸 |
| NumSpotsX / NumSpotsY | int | 只读 | 有效子孔径数(select_mla 后更新) |
| MeanIntensity / MaxIntensity | double | 只读 | 帧平均/峰值亮度 |
| SpotMaxIntensity | double | 只读 | 最强子孔径强度(高速模式或未采集时为 0) |
| Gain | double | 只读 | 最近一次自动曝光的增益 |
| FrameCounter | int | 只读 | 采集帧计数 |

## 运行

```bash
# 0) 激活项目环境(.venv 已含 caproto 1.3.0)
# 1) 启动 IOC(端口 5069)
$env:PYTHONPATH = "optics-epics\common;optics-epics\iocs\ioc-wfs\src;src;libs"
$env:EPICS_CA_SERVER_PORT = "5069"
.venv\Scripts\python -m caproto.server.ioc_examples iocs\ioc-wfs\src\wfs_ioc.py

# 2) 另一终端验证(客户端端口 5069 可省,用默认)
$env:EPICS_CA_ADDR_LIST = "127.0.0.1"
caget WFS-01:Connected WFS-01:SerialNumber
```

WSL 侧用 `caget` 直连 Windows 宿主 IP(需 `EPICS_CA_ADDR_LIST` 指向宿主机)。

## 离线冒烟

```bash
$env:PYTHONPATH = "optics-epics\common;optics-epics\iocs\ioc-wfs\src;src;libs"
$env:EPICS_CA_ADDR_LIST = "127.0.0.1"
$env:EPICS_CA_SERVER_PORT = "5069"
.venv\Scripts\python C:\Users\zhangh\AppData\Local\Temp\opencode\wfs_smoke.py
```

冒烟覆盖:21 个 PV 初值、Image 波形长度与全零、8 个控制 PV 离线写拒(ValueError)、7 个只读 PV 写拒、Connected=Disconnected。

## 真实硬件注意

- `open()` 较重(枚举/MLA/参考面/自动曝光/寻瞳,可达数秒);设备被占用时抛 `ConnectionError` → IOC 离线。
- 高速模式开启后 `get_spots_statics()` 断言失败 → SpotMaxIntensity/NumSpots 保持 0/旧值(日志警告,不崩溃)。
- `master_gain` 在驱动中仅注册表参数(无硬件 setter),故 Gain 为只读,由 AutoExpose 更新。
- 每次写 `ExposureTime`/`HighSpeed`/`MlaIndex`/Pupil* 均为阻塞 SDK 调用,经线程池执行,不阻塞事件循环。

## 文件

```
ioc-wfs/
├── src/wfs_ioc.py        # WfsIoc(PVGroup)实现
├── config/ioc.yaml       # 前缀/端口/设备/参数/PV 声明
└── README.md
```
