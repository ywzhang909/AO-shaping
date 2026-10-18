# Multi-SLM200 控制器软件说明书

## 1. 概述

`multi_slm_controller.py` 是 Santec SLM-200 空间光调制器的 Streamlit 前端页面。
支持最多两台 SLM，每台可独立连接、配置和相位编程。

启动方式：

```bash
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py
```

---

## 2. 类图

```mermaid
classDiagram
    class StreamlitApp {
        +main()
    }
    class SLMController {
        +connect_slm(slm_num)
        +disconnect_slm(slm_num)
        +set_wavelength(slm_num)
        +set_shift(slm_num)
        +set_video_mode(slm_num, mode)
        +toggle_correction(slm_num, enabled)
        +render_slm_sidebar()
        +render_phase_control(slm_num)
        +display_slm_status(slm_num)
    }
    class BackgroundToggle {
        -slm_container: list
        -phase_a: ndarray
        -phase_b: ndarray
        -stop_event: threading.Event
        -freq_ref: list
        +run()
    }
    class PatternControl {
        <<abstract></abstract>>
        +render(prefix) dict
    }
    class FlatControl
    class LinearGratingControl
    class HologramGratingControl
    class BlazedGratingControl
    class CircularGratingControl
    class LensControl
    class CheckerboardControl
    class BinaryGratingControl
    class MicrolensArrayControl
    class TurbulenceScreenControl
    class ZernikeControl
    class DammannGratingControl
    class VortexPhaseControl
    class HalfHalfPhaseControl
    class PatternHelper
    class Santec {
        +open()
        +close()
        +write_phase(phase, memory_number)
        +display_memory(memory_number)
        +display_data(phase, wait_time_s)
        +get_displayed_phase() tuple
        +apply_shift(shift_x, shift_y)
        +create_phase_from_array(phase_rad)
        +load_config() dict
        +save_config()
        +set_wavelength(wl)
        +load_correction_from_csv(path)
        +load_gray_from_csv(path)
        +get_wavelength_info() tuple
        +shift_phase(phase, sx, sy) static
    }
    class WavefrontCorrection {
        +is_valid: bool
        +map_error(phase, max_grayscale)
    }
    class SLMConfigManager {
        +load_config(serial) dict
        +save_config(serial, config)
    }

	StreamlitApp --> SLMController : 调用
    SLMController --> BackgroundToggle : 启动后台线程
    SLMController --> PatternControl : render_pattern_controls()
    PatternControl <|-- FlatControl
    PatternControl <|-- LinearGratingControl
    PatternControl <|-- HologramGratingControl
    PatternControl <|-- BlazedGratingControl
    PatternControl <|-- CircularGratingControl
    PatternControl <|-- LensControl
    PatternControl <|-- CheckerboardControl
    PatternControl <|-- BinaryGratingControl
    PatternControl <|-- MicrolensArrayControl
    PatternControl <|-- TurbulenceScreenControl
    PatternControl <|-- ZernikeControl
    PatternControl <|-- DammannGratingControl
    PatternControl <|-- VortexPhaseControl
    PatternControl <|-- HalfHalfPhaseControl
    SLMController --> Santec : 驱动
    Santec --> WavefrontCorrection : 使用
    Santec --> SLMConfigManager : 使用
    PatternHelper --> Santec : 读取 Panel_Res/Pitch_um
```

## 3. 模块映射

| 文件                    | 职责                                                       |
| ----------------------- | ---------------------------------------------------------- |
| multi_slm_controller.py | Streamlit 页面：设备发现、连接、相位控制、周期切换后台线程 |
| pattern_controls.py     | 图案类型注册表 + 相位生成流水线（generate_phase_gray）     |
| bench_settings.py       | 侧栏「本次实验光路」区块 + 三态（探测/人工）session 契约  |
| shaping_controls.py     | 闭环整形主区：GS+SPGD 闭环 / 正向模型闭环（render_closed_loop_shaping） |
| slm_calibration_ui.py   | 独立的 LUT 标定页面（不在本说明书范围）                    |

---

## 4. Session State Schema

每个 SLM 以 slm{N}（N = 1 或 2）为命名空间。

| 键                          | 类型            | 含义                      |
| --------------------------- | --------------- | ------------------------- |
| slm{N}                      | Santec    | SLM 对象（断开时为 None） |
| slm{N}_connected            | bool            | 连接标志                  |
| slm{N}_wavelength           | int             | 工作波长（nm）            |
| slm{N}_video_mode           | str             | 内存模式（DVI 已禁用）    |
| slm{N}_next_memory          | int             | 下一个内存槽提示          |
| slm{N}_phase_preview        | ndarray         | 缓存的预览图像            |
| slm{N}_phase_source         | str             | 缓存相位的来源标签        |
| slm{N}_shift_x/y            | int             | 平移值                    |
| slm{N}_use_correction       | bool            | 波前矫正开关              |
| slm{N}_toggle_phase_a/b     | ndarray         | 周期切换存储的相位        |
| slm{N}_toggle_active        | bool            | 周期切换是否运行          |
| slm{N}_toggle_frequency     | float           | 切换频率（Hz）            |
| slm{N}_toggle_thread        | Thread          | 后台线程句柄              |
| slm{N}_toggle_stop_event    | threading.Event | 停止信号                  |
| slm{N}_toggle_freq_ref      | list            | 可变频率引用（单元素）    |
| slm{N}_toggle_slm_container | list            | 可变 SLM 引用（单元素）   |
| slm{N}_base_phase           | ndarray         | 用于叠加的底相位          |
| slm{N}_overlay_base         | bool            | 叠加开关                  |
| slm{N}_width/height         | int             | 面板分辨率                |
| slm{N}_pixel_pitch_um       | float           | 像素间距                  |
| slm{N}_bits                 | int             | 位深                      |
| slm{N}_loaded_config        | dict            | 缓存的配置文件            |
| slm{N}_use_correction_cb    | bool (widget)   | 波前矫正开关复选框（widget 键） |
| slm{N}_correction_csv       | UploadedFile    | 待加载的矫正 CSV（widget 键）  |
| slm{N}_max_gray             | int             | 2π 灰度值（widget 键）        |
| slm{N}_pattern_type         | str             | 当前选中的相位图类型（widget 键） |
| slm{N}_pattern_helper       | PatternHelper   | 按设备缓存的图案计算助手      |
| slm{N}_csv_file             | UploadedFile    | 待加载的相位 CSV（widget 键）  |
| slm{N}_preview_colormap     | str             | 预览色图标签                |
| slm{N}_preview_colormap_widget | str          | 预览色图 radio（widget 键）   |
| available_slms              | list[dict]      | 设备发现缓存              |
| camera_pixel_um             | float           | 相机像元间距（μm，全局；**已在「本次实验光路」区块内**，key 不变） |
| _cam_pixel_checked/mismatch/detected | bool/bool/float | 像素间距自动探测状态（一次性） |
| bench_busy                  | bool            | 标定或闭环整形运行中（侧栏光路控件据此 disabled） |
| bench_detected              | dict            | 实验级探测值（exposure_ms 等） |
| bench_manual                | dict            | 实验级人工覆盖（只列被改过的字段） |
| bench_pending               | dict            | 实验级单次 flush 待写值（非 widget key） |
| bench_cam_type/cam_id       | str / int       | 实验级相机（widget 键）  |
| bench_exposure_ms           | float           | 会话曝光（0 = 自动起曝，widget 键） |
| bench_cam_crop/cam_size     | bool / int      | 读出窗口开关与边长（widget 键） |
| bench_blaze_period_px/direction | int / str   | 去零级载波（widget 键）  |
| bench_window_probe          | dict            | 相机窗口上限探测缓存（按 cam_type/cam_id 失效） |
| slm{N}_bench_detected       | dict            | 该 SLM 的探测值（pupil_center / geometry_file / geometry_correlation / evidence_dir / calibrated_at / geometry_source） |
| slm{N}_bench_manual         | dict            | 该 SLM 的人工覆盖        |
| slm{N}_bench_pending        | dict            | 该 SLM 单次 flush 待写值 |
| slm{N}_bench_effective      | dict            | 最近一次运行生效的光路参数快照（含每字段三态标记） |
| slm{N}_bench_pupil_center   | str             | 入射中心 x,y（空 = 自动定位，widget 键） |
| slm{N}_bench_panel_span_px  | int             | 模型覆盖宽度（widget 键；GS 的 `beam_radius_px×2` 落同一字段） |
| slm{N}_bench_geometry_source| str             | auto / reuse / keep（widget 键） |
| slm{N}_bench_geometry_file  | str             | bench_geometry.json 路径（复用模式下出现，widget 键） |
| slm{N}_shp_pending         | dict            | 主区预算预设的单次 flush 待写值（非 widget key） |
| slm{N}_shp_budget_preset*   | str             | 预算预设选择与其「已应用」标记（widget 键） |

---

## 5. 按钮时序图

### 5.1 连接 SLM

```mermaid
sequenceDiagram
    participant U as 用户
    participant UI as render_slm_sidebar
    participant C as connect_slm
    participant SLM as Santec
    participant SS as session_state

	U->>UI: 点击"连接"
    UI->>C: connect_slm(slm_num)
    C->>SS: 清理旧 SLM 对象
    C->>SLM: Santec(slm_number=N)
    C->>SLM: slm.open()
    SLM-->>C: 序列号、Panel_Res、Pitch_um、波长
    C->>SS: 存储 slm 及所有解析出的参数
    C->>SS: slm.load_config()
    C->>C: _refresh_device_list()
    C->>C: st.rerun()
    C-->>U: 成功提示
```


### 5.2 断开 SLM

```mermaid
sequenceDiagram
    participant U as 用户
    participant UI as render_slm_sidebar
    participant C as disconnect_slm
    participant SLM as Santec
    participant SS as session_state

    U->>UI: 点击"断开"
    UI->>C: disconnect_slm(slm_num)
    C->>SLM: slm.close()
    C->>SS: slm = None, connected = False
    C->>SS: 清除 phase_preview、phase_source
    C->>C: _stop_toggle(slm_num)
    C->>SS: 清除 toggle_phase_a/b、base_phase、overlay_base
    C->>SS: loaded_config = None
    C->>C: st.rerun()
    C-->>U: 成功提示
```

### 5.3 从模式生成器生成相位

```mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant C as generate_phase_gray
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 点击"从模式生成器生成相位"
    PC->>PC: _stop_toggle(slm_num)
    PC->>C: generate_phase_gray(slm, pattern_type, params)
    C->>C: PatternHelper.<method></method>()
    C->>SLM: create_phase_from_array(phase_rad)
    Note over SLM: 内部叠加矫正 + LUT + 平移
    C-->>PC: uint16 phase_gray
    PC->>SLM: slm.display_data(phase)
    Note over SLM: 自动轮换内存槽，等待像素翻转
    PC->>SS: refresh_phase_preview(slm_num)
    PC->>PC: st.success(...)
```

### 5.4 应用平移

```mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant S as set_shift
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 点击"应用平移"
    PC->>S: set_shift(slm_num)
    S->>S: _stop_toggle(slm_num)
    S->>SLM: slm.apply_shift(sx, sy, wait=0.3, save=True)
    Note over SLM: 读取当前相位、撤销旧偏移、施加新偏移、轮换槽位、重新显示
    SLM-->>S: 平移后的相位（或 None）
    S->>SS: refresh_phase_preview(slm_num)
    S->>PC: st.success / st.info
```

### 5.5 周期切换（开始/停止）

```mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant T as _toggle_phases_task
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 点击"开始周期切换"
    PC->>SS: 读取 phase_a、phase_b、slm
    PC->>SS: 创建 threading.Event (stop_event)
    PC->>SS: 创建 freq_ref = [freq]
    PC->>SS: 创建 slm_container = [slm]
    PC->>T: Thread(target=_toggle_phases_task, args=(container, a, b, stop, freq_ref))
    PC->>T: thread.start()
    loop 每 1/(2*freq) 秒
        T->>T: use_phase_a = int(elapsed*2*freq) % 2 == 0
        T->>SLM: get_displayed_memory_number()
        T->>T: 选取槽位 != 当前显示槽 (2..125)
        T->>SLM: write_phase(target_phase, memory_number=slot)
        T->>SLM: display_memory(slot)
    end
    U->>PC: 点击"停止周期切换"
    PC->>SS: stop_event.set()
    PC->>SS: toggle_active = False, thread = None
```

### 5.6 设为相位 A / B

```mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 点击"设为相位 A"
    PC->>SLM: slm.get_displayed_phase()
    SLM-->>PC: (phase_gray, source)
    PC->>SS: toggle_phase_a = phase.copy()
    PC->>PC: st.success

    U->>PC: 点击"设为全0相位 A"
    PC->>PC: np.zeros((h,w), uint16)
    PC->>SS: toggle_phase_a = zeros
    PC->>PC: st.success

    U->>PC: 点击"导出 phase_a.csv"
    PC->>PC: np.savetxt 到 BytesIO
    PC->>PC: st.download_button
```

### 5.7 从CSV加载相位

```mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 上传 CSV + 点击"从CSV加载相位"
    PC->>PC: _stop_toggle(slm_num)
    PC->>PC: 写入字节到临时文件
    PC->>SLM: slm.load_gray_from_csv(temp_path)
    SLM-->>PC: uint16 phase_gray（已校验格式）
    PC->>SLM: Santec.csv_to_phase(temp_path)
    SLM-->>PC: float64 phase_rad
    PC->>SLM: slm.display_phase(phase_rad)
    Note over SLM: create_phase_from_array()<br/>弧度→灰度 + 矫正 + LUT + 平移
    SLM-->>PC: mem_slot
    PC->>SS: refresh_phase_preview(slm_num)
    PC->>PC: st.success
    PC->>PC: temp_path.unlink()
```

### 5.8 保存当前相位为底相位

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 点击"保存当前相位为底相位"
    PC->>SLM: slm.get_displayed_phase()
    SLM-->>PC: (phase, source)
    PC->>SS: base_phase = phase.copy()
    PC->>SS: overlay_base = SS 值
    PC->>SLM: slm._overlay_base_phase = overlay
    PC->>PC: st.success

    U->>PC: 切换"发送时自动叠加底相位"
    PC->>SS: overlay_base = 复选框值
    PC->>SLM: slm._overlay_base_phase = overlay
    PC->>PC: st.success

    U->>PC: 点击"清除底相位"
    PC->>SS: base_phase = None, overlay_base = False
    PC->>SLM: slm._overlay_base_phase = False
    PC->>PC: st.success

### 5.9 应用/清除波前矫正

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as _render_slm_settings
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 上传矫正 CSV + 点击"应用矫正"
    PC->>PC: 写入字节到 NamedTemporaryFile
    PC->>SLM: slm.load_correction_from_csv(tmp_path)
    SLM-->>PC: bool ok
    PC->>SS: refresh_phase_preview(slm_num)
    PC->>PC: st.success / st.warning

    U->>PC: 点击"清除矫正"
    PC->>SLM: slm.load_correction_from_csv(None)
    SLM-->>PC: bool False
    PC->>SS: refresh_phase_preview(slm_num)
    PC->>PC: st.success

### 5.10 设置波长

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as _render_slm_settings
    participant S as set_wavelength
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 输入波长 + 点击"设置波长"
    PC->>S: set_wavelength(slm_num)
    S->>SLM: slm.set_wavelength(wl)
    Note over SLM: 可能需要约 40s（EEPROM 写入）
    S->>SS: refresh_phase_preview(slm_num)
    S->>PC: st.success

### 5.11 设置模式（内存/DVI）

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as _render_slm_settings
    participant S as set_video_mode
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 点击"设置模式"
    PC->>S: set_video_mode(slm_num, mode_label)
    S->>S: DVI 模式拒绝（st.error，返回）
    S->>SLM: slm._set_memory_mode(0)
    S->>SS: video_mode = "内存模式"
    S->>SS: refresh_phase_preview(slm_num)
    S->>PC: st.success

### 5.12 灰度设置

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as _render_grayscale_config
    participant SLM as Santec
    participant SS as session_state

    U->>PC: 点击"应用灰度设置"
    PC->>SLM: slm_obj._max_gray = new_max_gray
    PC->>SS: refresh_phase_preview(slm_num)
    PC->>PC: st.success

    U->>PC: 点击"获取当前2π灰度"
    PC->>SLM: slm_obj.get_wavelength_info()
    SLM-->>PC: (wavelength, current_max_gray)
    PC->>SLM: slm_obj._max_gray = current_max_gray
    PC->>SS: refresh_phase_preview(slm_num)
    PC->>PC: st.success

### 5.13 保存配置

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as _render_slm_settings
    participant SLM as Santec

    U->>PC: 点击"保存配置"
    PC->>SLM: 若 UI shift != slm.shift_x/y: slm.set_shift(...)
    PC->>SLM: slm.save_config()
    SLM-->>PC: 写入 JSON 配置文件
    PC->>PC: st.success

### 5.14 保存当前相位到 CSV

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as _render_slm_settings
    participant SLM as Santec
    participant FS as 文件系统

    U->>PC: 点击"保存当前相位到CSV"
    PC->>SLM: slm.get_displayed_phase()
    SLM-->>PC: (phase, source)
    PC->>FS: np.savetxt(~/.config/ao_shaping/slmN_phase.csv)
    PC->>PC: st.success

### 5.15 设备扫描

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_slm_sidebar
    participant P as _probe_slm
    participant SDK as slm_sdk (SLM_Ctrl_Open/ReadSD/Close)
    participant SS as session_state

    U->>PC: 点击"刷新设备列表"
    PC->>SS: available_slms = _scan_available_slms()
    loop slm_num in 1..8
        alt 已在 session_state 中连接
            PC->>SS: 构造 connected=True 条目
        else
            PC->>P: _probe_slm(slm_num) [守护线程, 3s 超时]
            P->>SDK: SLM_Ctrl_Open(slm_num)
            SDK-->>P: 返回码
            alt ret == 0 (空闲)
                P->>SDK: SLM_Ctrl_ReadSD(slm_num, buf)
                P-->>PC: {slm_number, serial, connected:False, in_use:False, status:"未连接"}
            else ret > 0 (占用中)
                P-->>PC: {slm_number, serial:None, connected:False, in_use:True, status:"使用中"}
            else ret < 0 (无设备)
                P-->>PC: None (跳过)
            end
            P->>SDK: SLM_Ctrl_Close(slm_num)
        end
    end
    PC->>SS: available_slms = 列表

---

## 6. 外部接口图

mermaid
flowchart LR
    U[用户浏览器] -->|HTTP| SL[Streamlit 服务端]
    SL -->|st.session_state| SS[会话状态字典]
    SL -->|threading| BG[后台周期切换线程]
    SL -->|ctypes FFI| SDK[Santec SDK DLL]
    SDK -->|SLM_Ctrl_*| HW[SLM-200 硬件]
    SL -->|读写| CFG[JSON 配置文件]
    SL -->|读写| CSV[矫正 CSV / 相位 CSV]
    SL -->|只读| PAT[PatternHelper]
    SL -->|只读| ZK[ZernikeCalc]

    subgraph 图案注册表
        PR[PATTERN_REGISTRY]
        PR --> FC[平场]
        PR --> LG[线性光栅]
        PR --> HG[全息光栅]
        PR --> BG[闪耀光栅]
        PR --> CG[圆形光栅]
        PR --> LS[透镜]
        PR --> CB[棋盘格]
        PR --> BG2[二元光栅]
        PR --> ML[微透镜阵列]
        PR --> TS[湍流相位屏]
        PR --> ZK2[Zernike]
        PR --> DG[达曼光栅]
        PR --> VP[涡旋相位]
        PR --> HH[半半相位]
    end

    SL -->|渲染| PR
    PAT -->|生成| PR
    SL -->|渲染| CL[闭环整形]

---

## 7. 图案注册表

| 图案       | 控制类                      | `generate_phase_gray` 中的输出路径                                                     |
| ---------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| 平场       | `FlatControl`             | `np.full((h,w), gray, uint16)`（**原始灰度**，不经驱动）                                |
| 线性光栅   | `LinearGratingControl`    | `PatternHelper.linear_grating` → `create_phase_from_array`                          |
| 圆形光栅   | `CircularGratingControl`  | `PatternHelper.circular_grating` → `create_phase_from_array`                        |
| 透镜       | `LensControl`             | `PatternHelper.lens` → `create_phase_from_array`                                    |
| 全息光栅   | `HologramGratingControl`  | `PatternHelper.hologram` → `create_phase_from_array`                                |
| 闪耀光栅   | `BlazedGratingControl`    | `PatternHelper.linear_grating(direction=...)` → `create_phase_from_array`           |
| 棋盘格     | `CheckerboardControl`     | `PatternHelper.generate_checkerboard` → `create_phase_from_array`（**弧度**）          |
| 二元光栅   | `BinaryGratingControl`    | `PatternHelper.generate_binary_grating` → `create_phase_from_array`（**弧度**）        |
| 微透镜阵列 | `MicrolensArrayControl`   | `PatternHelper.generate_microlens_array` → `create_phase_from_array`（**弧度，未包裹**）|
| 湍流相位屏 | `TurbulenceScreenControl` | `init_turbulence_screen` + `generate_turbulence_screen` → `create_phase_from_array`  |
| Zernike    | `ZernikeControl`          | `PatternHelper.generate_zernike_polynomial` → `create_phase_from_array`             |
| 达曼光栅   | `DammannGratingControl`   | `PatternHelper.generate_dammann_grating` → `create_phase_from_array`（**弧度**）      |
| 涡旋相位   | `VortexPhaseControl`      | `PatternHelper.generate_vortex` → `create_phase_from_array`（**始终弧度，驱动负责 wrap**）|
| 半半相位   | `HalfHalfPhaseControl`    | 平场半区（原始）+ 闪耀半区（`create_phase_from_array`）拼接                            |

> ⚠️ **2026-10-01 修正**：
> 1. 原表把 5 个控制类（棋盘格 / 二元光栅 / 微透镜阵列 / 达曼光栅 / 涡旋）标成
>    「原始 uint16」，**与代码相反** —— 基类 `PatternControl.generate_phase_gray`
>    (`gui/slm/pattern_controls.py:138-145`) 默认实现是
>    `slm.create_phase_from_array(self.generate_phase_rad(params))`，
>    且这些 `PatternHelper` 方法现在返回**弧度**（`pattern_helper.py:338-353/355-380/382-416/595-630`）。
>    涡旋类的 docstring 更明确写着 "no mod-2π, no wrap checkbox — the driver applies wrapping"
>    (`pattern_controls.py:707-708`)。**真正走原始 uint16 的只有平场与半半相位的平场半区。**

> 🔴 **2026-10-13 变更**：原注册表中的三个**开环**方形整形控制 —— `GSSquareControl`
> （"GS方形整形"）、`GSSquareGaussianControl`（"GS方形整形(高斯版)"）与 `SteadyPhaseControl`
> （"稳像法整形"），连同助手 `_gs_square_phase_radians` / `generate_gs_square_phase` —— **已移除**。
> 原因：在本 2f-Fourier 实验系统上，用未标定实验系统参数做开环方形整形已被实测证明行不通
> （Zernike SPGD 与未标定 GS 从未产出过均匀方形），见
> [`report/shape_trials/README.md`](../../report/shape_trials/README.md) 与
> [`report/shape_trials/online_20261010.md`](../../report/shape_trials/online_20261010.md)。
> 取而代之的是下方 §8「闭环整形」区（GS+SPGD 闭环 / 正向模型闭环，均走已验证的闭环管线）。

---

## 8. 闭环整形

`render_closed_loop_shaping(slm_num)`（`src/ao_shaping/gui/slm/shaping_controls.py`）在
单/双 SLM 布局中渲染为独立的「闭环整形」区，取代上文已移除的开环方形整形。两种方法均
**在 GUI 内调用 canonical 优化器管线**（与对应 CLI 同一代码路径），并在结束后把最佳
相位（raw 弧度）经常规 `display_phase` 路径下发显示。

| 方法 | 代码路径（与 CLI 等价） | 说明 |
| --- | --- | --- |
| **GS+SPGD 闭环整形（已验证配方）** | `SlmGsRefineParams` → `build_gs_refine_config` → `optimize_slm_gs_refine` → `save_gs_refine_results`（同 CLI `slm-gs-refine`） | 默认值即交替重放验证过的配方 |
| **正向模型闭环整形（前沿）** | `SlmModelInLoopParams` → `build_model_in_loop_config` → `optimize_slm_model_in_loop`（同 CLI `slm-model-in-loop`） | 需要 torch |

1. **GS+SPGD 闭环整形（已验证配方）**：GS 开环预矫正（bake-off（对比择优），仅当实测优于平场才采用）
   + 无感知 SPGD 细化。GUI 默认值取自交替重放验证过的配方
   （[`report/shape_trials/gs_refine_live_20261012.md`](../../report/shape_trials/gs_refine_live_20261012.md)）：
   曝光 `0.05 ms`（大恒曝光上限按会话：`AO_DAHENG_MAX_EXPOSURE_MS`，0 = 画面探测）、
   模型覆盖 `900` 面板px（= 2×`beam_radius_px=450`）、`target_side=100`、`cam_size=250`、`gs_iters=12`、
   `far_field_padding=3`、`phase_grid=12`、`epochs=12`、`seed=0`、`confirm_best` 开、
   `gs_method=plain`。验证结果：**flat 0.527 → plain GS 0.775 → SPGD best 0.780**
   （两次一致的交替重放）。⚠️ `cam_type` / `pupil_center` / `camera_pixel_um` 已不在主区，
   改由侧栏「本次实验光路」提供（见 §8.1）。

2. **正向模型闭环整形（前沿）**：每轮用强随机探针重拟合正向模型的 Zernike 像差
   （Step A），再冻结该像差合成目标方斑相位（Step B），带 trust region 与逐轮验收。
   需 `torch`。目前最佳均匀性 **CV 0.643 / EE 0.786**，但未成为被接受的均匀方形，
   见 [`report/shape_trials/model_in_loop_live_20261013.md`](../../report/shape_trials/model_in_loop_live_20261013.md)。

> **设备生命周期**：上述优化器在**内部**自行打开/关闭 SLM 与相机（Santec SDK 句柄**不可重入**），
> GUI 与标定按钮共用 `with_slm_ownership(slm_num)`：停背景相位切换 → 释放目标 SLM 句柄 →
> `finally` 重连（只释放目标 SLM，另一台保持连接）。整个过程在 session 里置 `bench_busy`，
> 侧栏光路控件据此 `disabled=True`。
>
> **GUI 显示**：flat / GS / best 各阶段得分、PIB、CV，以及 GS bake-off 判定
> （adopted / rejected）；debug 开启时产物存于 `data/debug/slm_gs_refine/<ts>/`。
> 运行结束时额外落盘 **`{save_dir}/params.json`** —— 本次运行**生效**的光路参数
> （逐字段带 `detected` / `manual` / `unset` 三态标记）。

### 8.1 侧栏「本次实验光路」（`bench_settings.py`）

**一次实验 = 一条光路**。相机、读出窗口、曝光、载波，以及每台 SLM 的入射位置与几何，
都在侧栏按「实验级 / 每 SLM 级」两层下沉到 `st.expander(..., expanded=False)`，
折叠态也能看到**光路摘要卡**。

| 分组 | 内容 |
| --- | --- |
| A 相机 | 相机类型（daheng/miicam/sim）、相机 ID、像元间距 `μm`（**含「探测」按钮与渲染期自动探测告警**，`key="camera_pixel_um"` 未变，从旧「相机工具」区整体搬来）、会话曝光 `ms`（0 = 自动；caption 写明 0 会被大恒钳到约 0.02 ms，比可用区间暗 20–75×；超 `AO_DAHENG_MAX_EXPOSURE_MS` 立即 warning，运行前再 error 一次）**+ 「探测」按钮（`key="detect_exposure_btn"`，仅 daheng 可用）**：点一下按当前光路实测曝光 |
| B 读出窗口 | 「启用开窗」+ 窗口边长 `px`。上界 = `min(max_width, max_height)`、step = 设备特征网格 `inc`（**取得到相机时是真值，取不到时用保守兜底并在 caption 说明来源**，点「探测窗口上限」重读）。⚠️ 已**取消** `cam_size_min = max(16, target_side)` 互夹：侧栏先渲染、拿不到主区目标边长，而 Streamlit 对「min 变化后越界」是**抛异常不是夹取** → 改到运行前校验（§8.2） |
| C 去零级载波 | 载波（-1 关 / 0 实测自动 / 128 / 96 / 64）+ 光栅方向 |
| D 面板与瞳面（每 SLM） | 入射中心 `x,y`（**空 = 自动定位**）、模型覆盖宽度（32 – `PANEL_RES[0]=1920`，step 10；GS 的 `beam_radius_px×2` 落同一字段）、**实测照明半径**（面板px，0 = 未实测）+ **「探测半径」按钮**（两轴各一次半平面随机相位边界扫描，约 5–10 分钟；证据落 `data/debug/bench_settings/beam_radius/`）；SLM 面板 caption 并排展示**像素间距 `8 μm`（`PITCH_UM`，面板模型用这个）** 与**有效口径 `7.8 μm`（`PIXEL_SIZE_UM`）** —— 相差 0.2 μm 是真实器件几何，不是待修误差（见 `slm200_constants` 注释） |
| E 标定（每 SLM） | 几何来源（自动标定（推荐）/ 复用已标定几何 / 沿用上次结果）+ 条件显示的几何文件路径（复用模式下用 `discover_model_in_loop_geometry(region=…)` 给建议值）+ 「标定 SLM N 光路」primary 按钮（**复用模式下 disabled**：那时两步都不测，点下去只会刷新 `calibrated_at`，却让人以为重新量过）+ 有手动改动时出现的「恢复探测值」 |

**自动探测但允许人工修改（三态）**

| 状态 | 判据 | 显示 |
| --- | --- | --- |
| `unset` | 从未标定 | `○` |
| `detected` | 有探测值 | `●` + 探测值 |
| `manual` | 当前值与探测值的差 **超过阈值** | `✎` + 当前值 / 探测值 / 差值（带单位） |

差异提示阈值：中心任一坐标 ≥1 px；覆盖宽度 / 窗口边长 ≥1 个 step；像元间距相对 ≥0.5%；
曝光相对 ≥1%；相关度绝对 ≥0.02。`manual` 永远优先，**重新标定不覆盖手动值**，
只在 caption 提示「探测到新几何，当前仍使用手动值」。

**写值为什么走 pending-flush**：Streamlit 里 widget 一旦以 `key` 实例化，该 key 在本次
脚本运行内被锁；标定按钮是在侧栏 widget **渲染之后**被点击的，同一脚本运行内直接写
widget key 会抛 `StreamlitAPIException`。所以两段式：点击 → 写 `{prefix}_bench_pending`
（非 widget key）+ `st.rerun()` → 侧栏渲染函数**第一行** `flush_pending()` 把值写进 widget
key 并清空 pending，再实例化控件。**不使用 `_safe_set_state()` 做标定回填** —— 它吞异常会让
回填静默失败；它只保留给「运行后重连」那条晚于侧栏渲染的路径。

标定走 `ao_shaping.runners.slm.model_in_loop_session.calibrate_session` —— 与 CLI
`slm-model-in-loop` **同一个入口**（入射定位 → 曝光探测 → sweep 几何；该函数抛**普通异常**，
不是 click 异常；相关度 < 0.50 直接中止、不提交相位）。只在**换光路 / 换 SLM / 换相机**后重标。

### 8.1.1 运行路径自动标定（不必先点标定按钮）

**「正向模型闭环」的运行按钮自己会标定**，与 CLI 的 `auto_session` 同一判定：`ShapingMethod`
的 `auto_calibrate` 钩子在 `shaping_session` 交出 SLM 句柄之后、优化器启动之前被调用 ——
这个位置正是 CLI `run()` 里预检的位置。跳过条件与 CLI 对齐：

| 条件 | 是否测量 | 理由 |
|------|----------|------|
| `cam_type == "sim"` | 否 | 数字孪生没有面板可扫、没有硬件几何可测，也不该碰硬件 |
| 几何来源 `reuse` 且已填几何文件 | 否 | 已有实测几何 |
| 几何来源 `keep` 且 `bench_detected` 已有本次会话结果 | 否 | 本会话已量过 |
| 其余（`auto`，或 `keep` 但还没有结果） | **是** | 否则整场实验跑在未标定的实验系统上 |

三条必须保持的性质：

1. **用运行自己的 `params`/`config`**（已合并侧栏光路值），不是侧栏按钮那条
   `build_calibration_params` 独立路径 —— 标定的是即将整形的那一套光路。
2. **人工值优先**：探测值只在 `bench_manual` 里没有该字段时才写回 config；有人工值就保留，
   并提示「探测到新值，当前仍使用人工值」。
3. **失败即中止**：标定被拒就抛，由 `shaping_session` 那一条路径上屏（只上屏一次），
   **不在未标定的实验系统上继续整形**。

证据（`panel_scan/`、`session_geometry/`）落在**本次运行的 `debug_dir`** 下，与结果同目录；
`debug_dir` 为空时回落到 `debug_run_dir(data, "slm_model_in_loop")`（判空必须判**原始字符串**：
`str(Path(""))` 是 `"."`，为真值，判 Path 会把证据写进当前工作目录）。

GS+SPGD 没有会话标定（CLI 侧同样没有），它的入射中心只来自侧栏：从未标定时会明确提示
当前用的是兜底实测值 `PUPIL_CENTER_FALLBACK`，换光路必须先标定或手填。

`measure_session_geometry` 的 `--exposure-ms` 取 `config.exposure_time_ms`（缺省 0 时仍是
CLI 原有行为）。侧栏实测曝光因此真正传到 sweep —— `0` 只是「不固定」，大恒会从约 0.02 ms
起步，比本光路可用区间暗 20–75 倍。

### 8.2 主区固定顺序与上限放开

顺序是契约，不是偏好：

```
方法 → 状态条 → 目标与评分 → 预算 → 高级参数 → 校验区 → 运行按钮 → 结果
```

- **状态条**：`光路 SLM {n}：● 已标定 {时间} · 入射中心 (…) 探测值 · 相关度 … · N 项手动覆盖（…）`；
  未标定时提示运行时会先自动标定（约 1–3 分钟）。
- **目标与评分**：共用区块标题与公式 caption，内部按方法渲染两套一行 4 列
  （GS：目标方边长(0=自动推导) / 边长因子 / w_pib / w_unif；
  MIL：目标形状 / 目标边长 / w_eff / w_uni）。
- **预算**：MIL 给「预设」（快速 2/40/300、**标准 6/80/600** = `slm_model_in_loop.md` §5 默认值、
  长跑 12/160/1200）+ 轮数 / Step A 步数 / Step B 步数；预设**只在切换时**覆写，不冲掉手改值。
  GS **不给预设**，只显示 12/12 已验证配方并指向 `report/shape_trials/`。
- **高级参数**（默认收起）：MIL 专属（探针数、探针 σ、留出探针、自由相位留出探针、region、
  远场补零、相关度门限、热启动、面板像素间距、学习率、Zernike 阶数…）；
  GS 专属（相位网格、扰动 δ、confirm-best、启用 GS、GS 变体、远场补零、面板像素间距、
  每测量均帧）。**`far_field_padding` 不合并**（GS 验证值 3，MIL 默认 8）。
- **校验区**在按钮**之前**渲染（否则按钮无法 `disabled`）：未连接 SLM / torch 缺失 /
  目标边长 > 窗口边长 / 复用几何与 region 不一致 / 曝光超上限 / 窗口边长非设备步长整数倍 /
  留出探针 < 1（控件允许 0 以便「看到可以关」，但 Step A 拟合要求 ≥1，错误信息指向
  「自由相位留出探针」）。

| 字段 | 原上限 | 现上限 | 来源 |
| --- | --- | --- | --- |
| 相机窗口边长 | 硬编码 max 1944 / step 10 | `min(max_width, max_height)` / step=`inc` | 设备特征网格 |
| 模型覆盖宽度 | max 1920 | `PANEL_RES[0]` = 1920（min 32，step 10） | 设备常量 |
| 目标边长 | 3–1000 | 上界 = 窗口边长（关窗 = 传感器短边），下界 8（小于 8 时 warning） | 侧栏窗口 |
| 探针数 | 1–32 | 1–64 + 耗时 caption | — |
| region | 8–256 | 8–512（仍须 8 的倍数）+ 内存 caption | — |
| 远场补零 | 1–16 | 1–64（各自默认值不变） | — |
| GS 相位网格 | 2–64 | 2–128 + 耗时 caption | — |
| 留出探针 | min 1 | min 0（0 = 不做留出验证；见校验区） | — |
| 探针 σ | 3.0–8.0 | 3.0–12.0（>6 rad 时 caption 提示可辨识性下降） | — |

保留的硬约束：`w_eff > 0`（min 0.01，help 保留「只优化 -CV 会掏空目标框 实测环围能量→0.002」）；
探针 σ ≥ 3.0；窗口边长必须是设备 `inc` 的整数倍；曝光 min 0（0 时大恒强制 warning）。

---

## 9. 关键设计规则（来自 AGENTS.md）

1. **内存槽轮换**：对**同一**槽位连续 `write_phase` + `display_memory` 会被固件判为 no-op，LCOS 面板不刷新。换槽在 2..125 中随机选取**除当前显示槽外**的槽位、进程重启后仍有效；现由 GUI 周期切换 `_toggle_phases_task`（内联）与驱动 `_pick_next_memory_slot`（平移路径）两处实现（原独立助手 `_pick_next_memory` 已随 #85 删除并内联）。
2. **DVI 模式已禁用**：`video_mode=1` 的 `open()` 可能挂起 120s/300s，且挂起后 memory 模式 `open()` 也挂起，需物理断电恢复。仅提供内存模式。
3. **平场灰度 RAW 路径**：始终通过 `np.full((h,w), gray, dtype=np.uint16)` 发送原始 uint16 灰度。切勿将平场相位走 `create_phase_from_array()`（弧度转换）。
4. **平移数学**：唯一实现在驱动层的静态 `Santec.shift_phase`。重写缓存相位必须走驱动层 `apply_shift()`，绝不能对缓存相位再跑 `write_phase`（会二次叠加矫正）。
5. **后台线程**：绝不直接碰 `st.session_state`。`_toggle_phases_task` 使用 `slm_container`（列表）和 `freq_ref`（列表）作为可变快照，并用 `threading.Event` 传递停止信号。
6. **时序**：优先用 `time.time()` 墙钟差值而非计数器 —— `int(elapsed * 2.0 * freq) % 2 == 0` 按精确频率切换，无累积误差。
7. **0 级光斑**：绝不假设 0 级光斑位于相机画面中心 —— 用 `argmax` 定位。
