# MiiCam 驱动审查：与官方 SDK 的差异、删除方案与优化建议

> 审查对象：`src/ao_shaping/drivers/ccd/miicam/driver.py`（942 行，保留）vs `src/ao_shaping/drivers/ccd/miicam_driver.py`（790 行，待删除）
> 对照基准：官方 SDK 文档 `miicamsdk.20240728/doc/{en,hans}.html`
> 结论要点：**`driver.py` 为正确、更新的实现，`miicam_driver.py` 应删除。**

---

## 1. 背景与结论

用户明确要求：**只保留 `miicam/driver.py`，删除 `miicam_driver.py`**。审查确认：

1. **`miicam_driver.py` 是死代码**——全局扫描确认无任何 `import` 引用它（仅 `issues_report.md` 文档提及）。可安全删除。
2. **`driver.py` 更符合官方 SDK 规范**，且修正了 `miicam_driver.py` 的多处缺陷。
3. 但 `driver.py` 与官方文档仍有**若干不匹配/改进点**，见下文。

---

## 2. 两个文件的差异清单

| 维度 | `miicam/driver.py`（保留） | `miicam_driver.py`（删除） | 判定 |
|------|---------------------------|----------------------------|------|
| SDK 路径查找 | 抽取到 `_sdk_setup.py`，含去重 | 内联，**存在 `_MII_SDK_PATHS` 重复定义 bug**（L29-34 与 L36-41） | ✅ driver.py |
| SDK 缺失回退 | 顶部始终 `import base` | 缺失时定义**桩 `CameraError`/`BaseCamera`**（掩盖错误，反模式） | ✅ driver.py |
| 采集模式 | `wait`/`callback` 双模式 | 仅内部回调缓冲 | ✅ driver.py |
| `reset_exposure_time` | **Stop→设置→重启**+丢帧（实验验证） | 仅 `put_ExpoTime`（**官方文档确认流中不生效**） | ✅ driver.py |
| 回调生命周期 | `_CallbackSession` 可复用 | 每次重建局部 `frame_callback` 闭包 | ✅ driver.py |
| `_FramePuller` | WaitImageV3 + 超时重试 | 内联 `PullImageV4` 轮询 + 5 次重试 | ✅ driver.py |
| Trigger/Snap/Still/Pause | 全套方法 | 无 | ✅ driver.py |
| 线程安全 | 用 `threading.Lock`/`Event` 保护回调缓冲 | 用**布尔 `_frame_buffer_lock` 忙等**（劣） | ✅ driver.py |
| 异常类 | `MIICAMError(CameraError)` | 同 | 相同 |

---

## 3. `driver.py` 与官方 SDK 文档的不匹配 / 需注意点

以下是审查发现的需要关注之处（以官方 `en.html`/`hans.html` 为基准）。

### 3.1 【重要】`max` 锁死 / 曝光量程经验

`driver.py` `reset_exposure_time` 有充足注释。补充官方依据：曝光 `<0.1ms` 信号淹没在传感器噪声中，建议 ≥0.2ms。**质量指标应优先 mean/亮区包围盒而非 max**。此项日志与代码注释一致，无代码改动，仅提示下游使用方。

### 3.2 【建议】`rowPitch` 与 `TDIBWIDTHBYTES` 对齐

`_FramePuller.wait_image` / `pull_image_v4` / `pull_still_image` 均传 `rowPitch=0`（默认，4 字节对齐）且 `bufsize = W×H×每像素字节`。

官方文档表：GREY8 在 `rowPitch=0` 时的容量应为 `TDIBWIDTHBYTES(8*W)`，**仅当 W 是 4 的倍数时才等于 `W×1`**。MiiCam 长边如 2048 是 4 的倍数 → 安全；但**非 4 倍数宽度会有 padding**，当前 bufsize 会偏小。

**改进建议**：改用 `rowPitch=-1`（紧密打包）并保持 `bufsize = W×H×nbytes`，彻底消除对齐差异。或按 `TDIBWIDTHBYTES` 计算 bufsize。

> ⚠️ 注意：本项目 `_get_buffer_params()` 用 `self.cam_width*self.cam_height*2`（GREY16）作为 bufsize，若相机宽度非 4 倍数且用 `rowPitch=0`，会读越界。建议验证实际相机分辨率并统一使用 `rowPitch=-1`。

### 3.3 【建议】缓冲分配在热路径

`_frame_callback`（L701-706）与 `wait_image`（L81）每次调用都新建 `ctypes.c_char*bufsize` 缓冲 + `MiicamFrameInfoV3()`。回调是每帧触发的内热路径，反复分配有 GC/性能开销。官方回调限制偏保守（要求快）。**建议复用预分配缓冲**（在 `_init_streaming` 或 `_CallbackSession.start` 中按尺寸预分配，用锁保护复用）。注意多线程/复用需同步。

### 3.4 【建议】回调内捕获 `Exception` 并静默忽略

`_frame_callback` 用 `except Exception: pass` 吞咽所有回调异常（L715、L733）。这符合"回调要快、勿阻塞"原则，但会掩盖 `PullImageV4` 失败等真实错误。**建议至少 `logger.debug/warning` 记录一次**（用次数节流，避免刷屏），便于调试。

### 3.5 【注意】`_CallbackSession` 的双层 Stop 语义

`_CallbackSession.stop()` 在非 `_was_active` 时调用 `_stop_streaming()`（内部会 `Stop`+sleep 0.3）又 `StartPullModeWithCallback(None,None)` 关闭回调。该写法尝试"用空回调关闭回调模式"。官方推荐用 `MIICAM_OPTION_CALLBACK_THREAD` 或事件通知。若在复用场景（callback 模式暂停/恢复）出问题，检查此路径。

### 3.6 【注意】`trigger_sync` 与官方 `TriggerSyncV4` 的差异

`driver.py` 用 `Trigger(1)` + `WaitImageV3` 手动组合实现 `trigger_sync`；官方提供 `Miicam_TriggerSyncV4`（单触发+同步等待，`nWaitMS=0` 时默认超时 `ExpoTime×102%+4000ms`）。**功能等价**，但官方封装更简洁、避免手动超时计算。若追求与官方一致，可改用 `cam.TriggerSyncV4(...)`（若 wrapper 暴露）。当前实现可正常工作，属可选项。

### 3.7 【注意】曝光时间单位换算

`put_ExpoTime(int(self.exposure_time_ms * 1000))` —— 官方文档单位是**微秒**，`ms×1000` 正确。`put_AutoExpoRange` 亦将 `ms×1000`。验证与官方一致 ✅。

### 3.8 【建议】`get_callback_frame` 弃用属性检查

`get_callback_frame` 内 `_callback_new_frame.wait()` 依赖 `_CallbackSession.start` 初始化。若用户未先进入 callback 模式直接调用，会因 `_callback_new_frame` 为 `None` 抛 `AttributeError`。**建议加保护**：未初始化时返回 `None` 或抛明确 `MIICAMError`。

---

## 4. `miicam_driver.py` 删除方案

**删除安全**：全局扫描确认无任何 `import` 引用 `miicam_driver`。唯一提及在 `issues_report.md`（文档，不影响运行）。

删除步骤：

```powershell
# 1. 确认无引用（应输出空）
grep -rn "miicam_driver" src/ tests/ scripts/

# 2. 删除文件
Remove-Item "D:\Projects\TIFO\AO-shaping\src\ao_shaping\drivers\ccd\miicam_driver.py"

# 3. （可选）清理 issues_report.md 中对该文件的遗留引用
```

**删除后验证**：
- `python -c "from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager; print(CameraStreamManager)"` 仍可导入。
- 运行 miicam 相关单元测试：`pytest tests/ao_shaping/drivers/ccd/test_miicam.py -v`（含硬件跳过逻辑）。

> 未删除前，`issues_report.md` L78、L189 对该文件的清理意见将随之失效——由于文件删除，这两条可一并从报告中移除。

---

## 5. 使用 miicam 代码的修改 / 优化 / 测试意见

基于全局使用审计（见 agent_wiki §11），分类给出意见。

### 5.1 删除/清理

- **`src/ao_shaping/drivers/ccd/miicam_driver.py`**：删除（见 §4）。
- **drivers/AGENTS.md 目录树过时**：AGENTS.md 结构树里写了 `ccd/miicam.py` 和 `ccd/miicam_device.py`，但实际 ccd 目录下**并无这两个文件**（现有：`base.py`, `common.py`, `daheng.py`, `ffmpeg.py`, `miicam_driver.py`, `__init__.py`, 以及 `miicam/` 子目录包）。当前主实现是 `ccd/miicam/driver.py`。建议更新 AGENTS.md 目录树以反映真实结构（删除 `miicam.py`/`miicam_device.py` 虚名，标注 `miicam/` 子包）。

### 5.2 优化（针对热路径）

- **回调缓冲复用**：见 §3.3。`_frame_callback` 每帧新建 buffer/frame_info，建议预分配。
- **`_get_buffer_params` 结果缓存**：`cam_width/cam_height/bit_depth` 在流运行期不变，可缓存 `(bufsize, bits, dtype)` 三元组，避免每帧重算（微小但热路径有益）。

### 5.3 一致性（统一导入路径）

- Pattern A 文件（直接 `from ...miicam.driver import CameraStreamManager`）与 Pattern B 文件（`from ao_shaping.drivers import CameraStreamManager`）并存。**建议统一**（推荐 Pattern B 硬件无关回退），减少对 miicam 驱动的硬耦合，便于后续换相机（如 Daheng）。涉及：`gs_square_runner.py`, `slm_phase_capture.py`, `slm_gray_response.py`, `micro_dm_image_collect.py`, `validate_flat_phase_gray.py`。

### 5.4 GUI `sys.modules["miicam"]` 打桩（脆弱）

`ccd_analyzer.py`, `slm_calibration_ui.py`, `zernike_response_matrix_ui.py` 在导入前把假的 `miicam` 模块放入 `sys.modules`。这依赖 driver.py 顶部的 `try/except` 逻辑配合。**风险**：若 driver.py 改为惰性加载（推荐），这些打桩不再必要。**优化方向**：让 `_setup_miicam_sdk` 失败时 driver.py 缓存一个可用的"未就绪"占位，而非依赖外部模块打桩。

### 5.5 测试意见

现有测试：`tests/ao_shaping/drivers/ccd/test_miicam.py`（API 单测）、`tests/.../hardware/test_hardware_integration.py`（硬件）、`test_slm_miicam.py`、`test_device_registry.py`（注册 miicam_ccd）。建议新增/增强：

1. **回调缓冲复用重构后**：新增单元测试验证 `_FramePuller`/`_frame_callback` 复用的缓冲内容正确（用 mock 的 `cam`）。
2. **`_get_buffer_params` 缓存**：测试缓存失效逻辑（分辨率/位深变更后重建）。
3. **`reset_exposure_time` 的 Stop→Start 序列**：用 mock 验证调用顺序（`Stop` before `put_ExpoTime` before `StartPullModeWithCallback`）。
4. **`get_callback_frame` 未初始化保护**：测试未进 callback 模式时的行为。
5. **删除 `miicam_driver.py` 后**：跑 `pytest tests/ao_shaping/drivers/ccd/` 确认无回归。

---

## 6. 优先级总结

| 优先级 | 事项 | 类型 |
|--------|------|------|
| P0 | 删除 `miicam_driver.py` | 清理（用户要求） |
| P1 | `rowPitch` / `TDIBWIDTHBYTES` 对齐（§3.2） | 正确性 |
| P1 | 回调缓冲与 `_get_buffer_params` 复用/缓存（§3.3, §5.2） | 性能 |
| P2 | 统一导入路径（§5.3） | 一致性 |
| P2 | GUI 打桩改惰性加载（§5.4） | 稳健性 |
| P2 | 回调异常记录 + `get_callback_frame` 保护（§3.4, §3.8） | 健壮性 |
| P3 | `trigger_sync` 用官方 `TriggerSyncV4`（§3.6） | 可选优化 |
