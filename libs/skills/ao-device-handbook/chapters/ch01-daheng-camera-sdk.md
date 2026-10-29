# Chapter 1: 大恒（Daheng/Galaxy）相机 SDK

## Core Idea
大恒相机走 GenICam 风格的 Feature 属性模型：设备对象只是容器，一切能力（曝光、增益、触发、LUT、区域）都是 `Device`/`DataStream` 上可按类型访问的 Feature 属性；采集只有两条路径——`get_image()` 拉取（阻塞）与回调推送（异步）。

## Frameworks Introduced
- **GX 采集主流程**：枚举 → 打开 → `stream_on()` → 取图 → `stream_off()` → `close_device()`
  - When to use: 一切大恒相机脚本的骨架；AO 实验中由 `utils/image/hardware_utils.open_camera()` 统一封装。
  - How: `device_manager.update_device_list()` 返回 `(dev_num, dev_info_list)`；`dev_info_list[i]` 是 dict（`index`/`sn`/`ip`/`mac`/`user_id`）；按 SN/user_id/index 打开（GigE 相机额外支持 ip/mac）；`cam.data_stream[0].get_image()` 取 RawImage。
- **拉取 vs 回调双采集模型**
  - When to use: 单帧/低频采集用拉取；高速连续采集、实时推理用小目标检测类场景用回调（`register_capture_callback`）。实测同步拉改回调后稳定帧率提升近 3 倍、丢帧告警消失。
  - How: `cam.data_stream[0].register_capture_callback(fn)` → `stream_on()` → SDK 采集线程在帧就绪时调用 `fn(raw_image)` → 结束后 `unregister_capture_callback()`。回调内必须先检查 `raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE`。
- **Feature 属性访问模型（8 类）**
  - When to use: 设置任何相机参数（曝光、增益、触发模式、像素格式……）。
  - How: `is_implemented` 探活 → 按类型走对应访问器：`IntFeature`（`get_min/max/value`）、`FloatFeature`（`get_min_as_float/set_as_float`）、`EnumFeature`（`get_entry_enum_by_index/set_by_index`）、`BoolFeature`（`set_value(True)`）、`StringFeature`（`get_max_string_length`/`get_value`/`set_value`）、`BufferFeature`（`get_buffer_length`/`set_buffer`/`get_buffer`）、`CommandFeature`（`send_command()` 如开停采）。
- **曝光融合（Exposure Fusion）**
  - When to use: 多帧不同曝光序列合成高动态图（弱光 AO 图像、目标增强）。
  - How: 每像素算对比度（拉普拉斯）、饱和度（HSV-S/亮度）、亮度三路加权得初始权重 → 归一化 → 高斯模糊 → 拉普拉斯金字塔逐尺度融合 → 自底向上上采样叠加细节。参考实现参数：`alphas=(1.0,1.0,1.0), best_illumination=0.5, sigma=0.2`。

## Key Concepts
- **DeviceManager**: 设备枚举与打开的入口，`update_device_list()` 返回设备数与信息字典列表。
- **DataStream**: 采集通道；千兆网/USB 相机当前均单流通道（`get_stream_channel_num()==1`）。
- **RawImage / RGBImage**: 原始帧与彩色转换帧；`raw.convert("RGB")` → `rgb.get_numpy_array()` 得到 numpy 数组。
- **Feature**: 属性对象基类，统一提供 `is_implemented`/`is_readable`/`is_writable`。
- **GxFrameStatusList**: 帧状态枚举，`INCOMPLETE` 表示残缺帧必须丢弃。
- **拉普拉斯金字塔**: 由低分辨率上采样后与对应高斯层做差得到细节层；融合时逐尺度加权叠加。
- **ICaptureEventHandler**（C++）: 回调基类，实现 `DoOnImageCaptured` 接收推送帧。
- **FFC（Flat Field Correction）**: `RawImage.get_ffc_coefficients()` 求平场系数、`flat_field_correction()` 校正。

## Mental Models
- Use "属性不是函数" 的心智：先 `is_implemented` 探活再操作，缺失功能静默跳过而不是报错分支。
- 把 拉取采集 理解为 轮询报亭：高频场景必然丢帧；回调采集 是 订阅报纸：SDK 线程推送，主线程不被图像 I/O 阻塞。
- Think of 曝光融合 as 加权多曝光投票：权重来自局部信息量（对比度/饱和度/亮度），金字塔保证细节不糊。

## Anti-patterns
- **回调里做重计算**：回调线程是 SDK 采集线程，阻塞它=丢帧；重活（推理/落盘）必须投递到独立工作队列。
- **拉取循环里不查帧状态**：`INCOMPLETE` 帧直接进处理管线会污染检测结果。
- **曝光融合用"滚动上采样"构造拉普拉斯金字塔**：每层从上一层的上采样结果继续上采样会累积信息损失，输出饱和度爆炸、细节丢失；必须像官方 MATLAB 参考那样，下一层上采样对象始终是**当前尺度的高斯金字塔结果**。
- **单帧 `get_image` 当高速采集用**：等待+拉取的阻塞开销在高帧率下直接丢帧。

## Code Examples
```python
import gxipy as gx

dm = gx.DeviceManager()
dev_num, dev_info_list = dm.update_device_list()
sn = dev_info_list[0].get("sn")
cam = dm.open_device_by_sn(sn)      # 亦可用 by_user_id / by_index；GigE 另有 by_ip / by_mac
cam.stream_on()
raw = cam.data_stream[0].get_image()
rgb = raw.convert("RGB")
arr = rgb.get_numpy_array()         # → numpy 数组，供显示/保存/算法
cam.stream_off()
cam.close_device()
```
- 演示：拉取模式最小闭环；`data_stream[0]` 固定第 0 流通道。

```python
def capture_callback(raw_image):
    if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
        print("incomplete frame")   # 残缺帧丢弃
        return
    # 只投递轻量任务，重计算放工作队列

cam.data_stream[0].register_capture_callback(capture_callback)
cam.stream_on()
# ... 运行期间 SDK 自动推送 ...
cam.stream_off()
cam.data_stream[0].unregister_capture_callback()
```
- 演示：回调推送模式；注册/注销必须成对。

## Reference Tables
| 打开方式 | 方法 | 备注 |
|---|---|---|
| 序列号 | `open_device_by_sn(strSN)` | 通用 |
| 用户 ID | `open_device_by_user_id(user_id)` | 通用 |
| 索引 | `open_device_by_index(index)` | 通用 |
| IP | `open_device_by_ip(ip)` | 仅 GigE |
| MAC | `open_device_by_mac(mac)` | 仅 GigE |

| Feature 类型 | 关键方法 |
|---|---|
| Int / Float | `get_min` / `get_max` / `get_value` / `set_value`（Float 有 `*_as_float` 变体） |
| Enum | `get_entry_num` / `get_entry_enum_by_index` / `set_by_index` |
| Bool | `get_value` / `set_value(bool)` |
| String | `get_max_string_length` / `get_value` / `set_value` |
| Buffer | `get_buffer_length` / `set_buffer` / `get_buffer` |
| Command | `send_command()`（如 AcquisitionStart/Stop） |

| 曝光融合参数 | 值 | 含义 |
|---|---|---|
| `alphas` | (1.0, 1.0, 1.0) | 对比度/饱和度/亮度权重 |
| `best_illumination` | 0.5 | 亮度最优目标 |
| `sigma` | 0.2 | 权重图高斯模糊半径 |

## Key Takeaways
1. 一切参数走 Feature 模型：`is_implemented` 探活 → 按 8 类类型访问，不要猜属性存在。
2. 高频采集必须回调推送，拉取只适合低频单帧；回调内禁重计算。
3. 每帧先查 `GxFrameStatusList`，`INCOMPLETE` 丢弃。
4. 曝光融合的拉普拉斯金字塔必须用官方"逐层从高斯金字塔上采样"方式，滚动上采样必失败。
5. 打开设备优先用 SN（稳定），GigE 相机可用 IP/MAC。

## Connects To
- **ch06**: 本仓相机统一入口 `open_camera(camera_type, ...)` 在大恒域落到本 SDK 流程。
- **ch03 (MiiCam)**: 同为相机驱动，但 MiiCam 走 C DLL + HRESULT 错误码，采集模型同为拉/推两路。
