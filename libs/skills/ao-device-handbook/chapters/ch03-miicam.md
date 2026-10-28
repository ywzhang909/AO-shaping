# Chapter 3: MiiCam 相机（C SDK + ctypes）

## Core Idea
MiiCam 是一个"C 句柄 + HRESULT + 回调线程"的高速相机 SDK：成功判定只看 `hr >= 0`，配置分"运行时可改"和"必须 Stop 后改"两档，回调线程里严禁调用 SDK——这三条是全部驱动代码的骨架约束。

## Frameworks Introduced
- **推荐调用流程（8 步）**
  - When to use: MiiCam 驱动一切脚本的骨架。
  - How: `Miicam_EnumV2`（单相机可省略，`Miicam_Open(NULL)` 自动选）→ `Miicam_Open(camId)` → 关自动曝光 `put_AutoExpoEnable(0)` + `put_ExpoTime`/`put_ExpoAGain` → `put_Option`（BITDEPTH/RGB/RAW/BINNING）→ `put_Size` 设分辨率 → `StartPullModeWithCallback(h, fun, ctx)` → 按模式取帧（`WaitImageV3` 阻塞 或 Trigger+`PullImageV4` 回调）→ `Stop` → `Close`。
  - 核心生命周期约束：`Open/Close` 与 `Start/Stop` 是重量级操作，禁止频繁调用；重新配置优先 `Stop → 设置 → Start`，而不是 `Close → Open`。
- **双采集模式（wait / callback）**
  - When to use: 默认 `wait`（`WaitImageV3` 阻塞拉取，简单稳定）；严格时序/高速用 `callback`（独立回调线程 + 软件触发 `Miicam_Trigger(1)` + `PullImageV4` 取帧，参考 C++ `demosofttrigger`）。
  - How: `capture_mode="wait"` → `_FramePuller.wait_image()`；`capture_mode="callback"` → `session` 内 `cam.Trigger(1)` → `get_callback_frame(timeout=5.0)`。
- **曝光复位流程（reset_exposure_time，2026-09 实验确认）**
  - When to use: 任何运行中改曝光的场景。
  - How: 流运行中直接 `put_ExpoTime` **不生效**（抓到的仍是旧曝光残帧）；必须 `Stop → put_ExpoTime → Start` 并丢弃前 1-2 帧（`get_numpy_image(skip_first=True)`）；严格时序可再 `sleep(0.15~0.3s)`。单位：驱动层统一 ms，SDK 参数 µs（`ms × 1000`）；量程 0.011 ms 至 10000 ms。
- **SNAP 静止图**
  - When to use: 高分辨率静止图（非视频流）。
  - How: `Miicam_Snap(h, index)`（`0xFFFFFFFF`=当前分辨率）→ 完成后 `MIICAM_EVENT_STILLIMAGE` → `PullStillImage`；连拍用 `SnapN(h, index, n)`。

## Key Concepts
- **HMiicam**: 相机句柄，`Miicam_Open` 返回，`Close` 后失效。
- **HRESULT 黄金规则**: `hr >= 0` 即成功；`S_FALSE=0x1` 是"成功但无操作"。**绝不能**用 `== S_OK` 或 `== 0` 判断。
- **Pull 模式 / Snap 模式 / Trigger 模式**: 连续拉流 / 高分辨率静止图 / 软件或外部触发精确控制时机。
- **bits 参数**: `PullImageV4` 输出位深，`0`=默认，`8/16/24/32/48/64`。
- **rowPitch**: 行跨度；`0`=默认 4 字节对齐（`TDIBWIDTHBITS`），`-1`=紧密打包无填充（NumPy/OpenCV 互操作推荐）。
- **E_WRONG_THREAD**: 在回调上下文调用被禁 API（`put_Option`/`put_Roi`）时返回。
- **E_PENDING / E_TIMEOUT**: `0x8000000A` / `0x8001011F`，本驱动 `_FramePuller`/`trigger_sync` 识别为重试条件。
- **k≈0.414 像素缩放**: SLM 8µm 像素约 2.417 相机像素（gs-square 光路标定），光路调整后必须重标。

## Mental Models
- Use "hr>=0 才算赢" 的心智：`S_FALSE` 是成功，任何 `==0` 判断都会把"值未变"当失败。
- 把 回调线程 理解为 另一台机器：它在 SDK 内部，你调 `Stop/Close/put_Option` 等于跨机器死锁；要它做事用 flag/event 通知，不要它调你。
- Think of 曝光设置 as 需要停机的参数：改完必须重启流并丢 1-2 帧，否则残帧混入数据。

## Anti-patterns
- **用 `== S_OK`/`==0` 判成功**：`S_FALSE` 会被误判为失败，导致"设置成功但被当失败重试"。
- **回调内调用 `Miicam_Stop/Close`**：两者都等回调返回，直接死锁；必须 flag/event 通知主线程。
- **回调内调用 `put_Option(TRIGGER/BITDEPTH/PIXEL_FORMAT/BINNING/ROTATE)` 或 `put_Roi`**：返回 `E_WRONG_THREAD`；拉流前设 `MIICAM_OPTION_CALLBACK_THREAD=1` 用独立回调线程，或把设置移回主线程。
- **运行中直接 `put_ExpoTime`**：不生效，抓到旧曝光残帧（症状：max 恒定、图像雷同）；必须 Stop-设置-Start。
- **运行中改非运行时选项**（RAW/ISP/RGB/UPSIDE_DOWN/FRAMERATE/MULTITHREAD/CALLBACK_THREAD/前后端队列长度）：返回 `E_UNEXPECTED`；一律拉流前设好。
- **频繁 `Open/Close`**：重量级操作；重配置走 `Stop → 设置 → Start`。
- **行写时他线程读**：线程安全策略是"多读一写"，一写期间禁止其他线程读写同一对象（不同相机对象可并行）。

## Code Examples
```c
// 立即拉最新帧（无帧返回 E_PENDING）
HRESULT Miicam_PullImageV4(
    HMiicam h, void* pImageData,  // NULL = 仅查元数据
    int bStill,                    // 1=静止图, 0=视频
    int bits,                      // 0=默认, 8/16/24/32/48/64
    int rowPitch,                  // 0=4字节对齐, -1=紧密打包
    unsigned* pnWidth, unsigned* pnHeight, MiicamFrameInfoV4* pInfo);

// 单触发 + 同步等待
Miicam_TriggerSyncV4(h, 0, data, 0, -1, &info);  // nWaitMS=0 → 默认超时 ExpoTime×102% + 4000ms
```
- 演示：拉取/触发的两个核心 C 签名；`rowPitch=-1` 配 NumPy 最干净。

```python
# callback 模式（driver.py）
def __take_one_shot_callback(self) -> np.ndarray:
    with self._callback_session as session:
        self.cam.Trigger(1)
        result = self.get_callback_frame(timeout=5.0)
        if result is None:
            raise MIICAMError("Timeout waiting for callback frame after trigger")
        img, _ = result
        return img
```
- 演示：软件触发 + 回调取帧的最小闭环；超时抛错而非静默。

## Reference Tables
| HRESULT | 值 | 含义 |
|---|---|---|
| `S_OK` / `S_FALSE` | `0x0` / `0x1` | 成功 / 成功但无操作 |
| `E_ACCESSDENIED` | `0x80070005` | 权限（Linux 需 udev/root） |
| `E_INVALIDARG` | `0x80070057` | 参数错误 |
| `E_NOTIMPL` | `0x80004001` | 该相机不支持此功能 |
| `E_UNEXPECTED` | `0x8000ffff` | 状态不满足（运行中设非运行时选项） |
| `E_WRONG_THREAD` | `0x8001010e` | 回调上下文调用被禁 API |
| `E_GEN_FAILURE` | `0x8007001f` | 硬件故障（线缆/USB） |
| `E_BUSY` | `0x800700aa` | 相机已被占用 |
| `E_PENDING` / `E_TIMEOUT` | `0x8000000A` / `0x8001011F` | 无帧就绪 / 超时（重试条件） |

| Option | 值 | 含义 |
|---|---|---|
| `BITDEPTH` | 0/1 | 8bit / 16bit |
| `RGB` | 3/4 | 8bit / 16bit Grey |
| `RAW` | 0/1 | Bayer 原始 |
| `BINNING` | `0x80\|2` | 2×2 平均 binning |
| `TRIGGER` | 0/1/2/3 | 视频 / 软件 / 外部 / 外部+软件 |
| `FLUSH` | 3 | 清空帧缓冲 |
| `CALLBACK_THREAD` | 1 | 独立回调线程（拉流前设） |

## Key Takeaways
1. 成功判定 `hr >= 0`；`S_FALSE` 是成功。
2. 回调内禁调 `Stop/Close/put_Option/put_Roi`；独立回调线程 + flag 通知。
3. 改曝光必须 Stop-设置-Start 并丢 1-2 帧；驱动层 ms、SDK 参数 µs。
4. 重配置走 `Stop → Start`，不 `Close → Open`。
5. NumPy 互操作 `rowPitch=-1`；分辨率宽为 4 的倍数时 `rowPitch=0` 等价。
6. SLM→MiiCam 像素缩放 k≈0.414，光路变动后重新标定。

## Connects To
- **ch01 (大恒)**: 同为相机，但大恒走 GenICam Feature 模型，MiiCam 走 C 句柄 + HRESULT。
- **ch02 (SLM-200)**: gs-square 标定中 SLM 像素与 MiiCam 像素的缩放关系 k≈0.414。
