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

mermaid
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
        <<abstract>>
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
    class GSSquareControl
    class SteadyPhaseControl
    class PatternHelper
    class SantecSLM200 {
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
    PatternControl <|-- GSSquareControl
    PatternControl <|-- SteadyPhaseControl
    SLMController --> SantecSLM200 : 驱动
    SantecSLM200 --> WavefrontCorrection : 使用
    SantecSLM200 --> SLMConfigManager : 使用
    PatternHelper --> SantecSLM200 : 读取 Panel_Res/Pitch_um

## 3. 模块映射

| 文件 | 职责 |
|------|------|
| multi_slm_controller.py | Streamlit 页面：设备发现、连接、相位控制、周期切换后台线程 |
| pattern_controls.py | 图案类型注册表 + 相位生成流水线（generate_phase_gray） |
| slm_calibration_ui.py | 独立的 LUT 标定页面（不在本说明书范围） |

---

## 4. Session State Schema

每个 SLM 以 slm{N}（N = 1 或 2）为命名空间。

| 键 | 类型 | 含义 |
|-----|------|------|
| slm{N} | SantecSLM200 | SLM 对象（断开时为 None） |
| slm{N}_connected | bool | 连接标志 |
| slm{N}_wavelength | int | 工作波长（nm） |
| slm{N}_video_mode | str | 内存模式（DVI 已禁用） |
| slm{N}_next_memory | int | 下一个内存槽提示 |
| slm{N}_phase_preview | ndarray | 缓存的预览图像 |
| slm{N}_phase_source | str | 缓存相位的来源标签 |
| slm{N}_shift_x/y | int | 平移值 |
| slm{N}_use_correction | bool | 波前矫正开关 |
| slm{N}_toggle_phase_a/b | ndarray | 周期切换存储的相位 |
| slm{N}_toggle_active | bool | 周期切换是否运行 |
| slm{N}_toggle_frequency | float | 切换频率（Hz） |
| slm{N}_toggle_thread | Thread | 后台线程句柄 |
| slm{N}_toggle_stop_event | threading.Event | 停止信号 |
| slm{N}_toggle_freq_ref | list | 可变频率引用（单元素） |
| slm{N}_toggle_slm_container | list | 可变 SLM 引用（单元素） |
| slm{N}_base_phase | ndarray | 用于叠加的底相位 |
| slm{N}_overlay_base | bool | 叠加开关 |
| slm{N}_width/height | int | 面板分辨率 |
| slm{N}_pixel_pitch_um | float | 像素间距 |
| slm{N}_bits | int | 位深 |
| slm{N}_loaded_config | dict | 缓存的配置文件 |
| available_slms | list[dict] | 设备发现缓存 |

---

## 5. 按钮时序图

### 5.1 连接 SLM

mermaid
sequenceDiagram
    participant U as 用户
    participant UI as render_slm_sidebar
    participant C as connect_slm
    participant SLM as SantecSLM200
    participant SS as session_state

    U->>UI: 点击"连接"
    UI->>C: connect_slm(slm_num)
    C->>SS: 清理旧 SLM 对象
    C->>SLM: SantecSLM200(slm_number=N)
    C->>SLM: slm.open()
    SLM-->>C: 序列号、Panel_Res、Pitch_um、波长
    C->>SS: 存储 slm 及所有解析出的参数
    C->>SS: slm.load_config()
    C->>C: _refresh_device_list()
    C->>C: st.rerun()
    C-->>U: 成功提示

### 5.2 断开 SLM

mermaid
sequenceDiagram
    participant U as 用户
    participant UI as render_slm_sidebar
    participant C as disconnect_slm
    participant SLM as SantecSLM200
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

### 5.3 从模式生成器生成相位

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant C as generate_phase_gray
    participant SLM as SantecSLM200
    participant SS as session_state

    U->>PC: 点击"从模式生成器生成相位"
    PC->>PC: _stop_toggle(slm_num)
    PC->>C: generate_phase_gray(slm, pattern_type, params)
    alt GS方形整形
        PC->>PC: 创建 st.status 进度条
        C->>C: measure_spot_diameter_cam()
        C->>C: compute_square_side()
        C->>C: build_square_target_amplitude()
        C->>C: gerchberg_saxton(...)
        Note over C,SLM: 可选 live_display 回调每轮迭代推送相位
    else 其他图案
        C->>C: PatternHelper.<method>()
        C->>SLM: create_phase_from_array(phase_rad)
        Note over SLM: 内部叠加矫正 + LUT + 平移
    end
    C-->>PC: uint16 phase_gray
    PC->>SLM: slm.display_data(phase)
    Note over SLM: 自动轮换内存槽，等待像素翻转
    PC->>SS: refresh_phase_preview(slm_num)
    PC->>PC: st.success(...)

### 5.4 应用平移

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant S as set_shift
    participant SLM as SantecSLM200
    participant SS as session_state

    U->>PC: 点击"应用平移"
    PC->>S: set_shift(slm_num)
    S->>S: _stop_toggle(slm_num)
    S->>SLM: slm.apply_shift(sx, sy, wait=0.3, save=True)
    Note over SLM: 读取当前相位、撤销旧偏移、施加新偏移、轮换槽位、重新显示
    SLM-->>S: 平移后的相位（或 None）
    S->>SS: refresh_phase_preview(slm_num)
    S->>PC: st.success / st.info

### 5.5 周期切换（开始/停止）

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant T as _toggle_phases_task
    participant SLM as SantecSLM200
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

### 5.6 设为相位 A / B

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant SLM as SantecSLM200
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

### 5.7 从CSV加载相位

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant SLM as SantecSLM200
    participant SS as session_state
    participant FS as _apply_shift

    U->>PC: 上传 CSV + 点击"从CSV加载相位"
    PC->>PC: _stop_toggle(slm_num)
    PC->>PC: 写入字节到临时文件
    PC->>SLM: slm.load_gray_from_csv(temp_path)
    SLM-->>PC: uint16 phase_gray
    PC->>FS: _apply_shift(phase_gray, shift_x, shift_y)
    FS->>SLM: SantecSLM200.shift_phase (静态)
    PC->>SS: mem_slot = _pick_next_memory(slm_num)
    PC->>SLM: slm.write_phase(phase, memory_number=mem_slot)
    PC->>SLM: slm.display_memory(mem_slot)
    PC->>SS: refresh_phase_preview(slm_num)
    PC->>PC: st.success / st.warning
    PC->>PC: temp_path.unlink()

### 5.8 保存当前相位为底相位

mermaid
sequenceDiagram
    participant U as 用户
    participant PC as render_phase_control
    participant SLM as SantecSLM200
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
    participant SLM as SantecSLM200
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
    participant SLM as SantecSLM200
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
    participant SLM as SantecSLM200
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
    participant SLM as SantecSLM200
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
    participant SLM as SantecSLM200

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
    participant SLM as SantecSLM200
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
        PR --> GS[GS方形整形]
        PR --> SP[稳像法整形]
    end

    SL -->|渲染| PR
    PAT -->|生成| PR

---

## 7. 图案注册表

| 图案 | 控制类 | `generate_phase_gray` 中的输出路径 |
|------|--------|----------------------------------|
| 平场 | `FlatControl` | `np.full((h,w), gray, uint16)`（原始灰度） |
| 线性光栅 | `LinearGratingControl` | `PatternHelper.linear_grating` → `create_phase_from_array` |
| 圆形光栅 | `CircularGratingControl` | `PatternHelper.circular_grating` → `create_phase_from_array` |
| 透镜 | `LensControl` | `PatternHelper.lens` → `create_phase_from_array` |
| 全息光栅 | `HologramGratingControl` | `PatternHelper.hologram` → `create_phase_from_array` |
| 闪耀光栅 | `BlazedGratingControl` | `PatternHelper.linear_grating(direction=...)` → `create_phase_from_array` |
| 棋盘格 | `CheckerboardControl` | `PatternHelper.generate_checkerboard`（原始 uint16） |
| 二元光栅 | `BinaryGratingControl` | `PatternHelper.generate_binary_grating`（原始 uint16） |
| 微透镜阵列 | `MicrolensArrayControl` | `PatternHelper.generate_microlens_array`（原始 uint16） |
| 湍流相位屏 | `TurbulenceScreenControl` | `PatternHelper.init_turbulence_screen` + `generate_turbulence_screen`（原始 uint16） |
| Zernike | `ZernikeControl` | `PatternHelper.generate_zernike_polynomial` → `create_phase_from_array` |
| 达曼光栅 | `DammannGratingControl` | `PatternHelper.generate_dammann_grating`（原始 uint16） |
| 涡旋相位 | `VortexPhaseControl` | `PatternHelper.generate_vortex`（wrap 时 uint16，否则 rad→uint16） |
| 半半相位 | `HalfHalfPhaseControl` | 平场半区（原始）+ 闪耀半区（`create_phase_from_array`）拼接 |
| GS方形整形 | `GSSquareControl` | `generate_gs_square_phase` → GS → `create_phase_from_array` |
| 稳像法整形 | `SteadyPhaseControl` | SPM + 闪耀光栅 → `create_phase_from_array` |

---

## 8. 关键设计规则（来自 AGENTS.md）

1. **内存槽轮换**：对**同一**槽位连续 `write_phase` + `display_memory` 会被固件判为 no-op，LCOS 面板不刷新。`_pick_next_memory` 随机选取 2..125 中除当前显示槽外的槽位，进程重启后仍有效。
2. **DVI 模式已禁用**：`video_mode=1` 的 `open()` 可能挂起 120s/300s，且挂起后 memory 模式 `open()` 也挂起，需物理断电恢复。仅提供内存模式。
3. **平场灰度 RAW 路径**：始终通过 `np.full((h,w), gray, dtype=np.uint16)` 发送原始 uint16 灰度。切勿将平场相位走 `create_phase_from_array()`（弧度转换）。
4. **平移数学**：唯一实现在驱动层的静态 `SantecSLM200.shift_phase`。重写缓存相位必须走驱动层 `apply_shift()`，绝不能对缓存相位再跑 `write_phase`（会二次叠加矫正）。
5. **后台线程**：绝不直接碰 `st.session_state`。`_toggle_phases_task` 使用 `slm_container`（列表）和 `freq_ref`（列表）作为可变快照，并用 `threading.Event` 传递停止信号。
6. **时序**：优先用 `time.time()` 墙钟差值而非计数器 —— `int(elapsed * 2.0 * freq) % 2 == 0` 按精确频率切换，无累积误差。
7. **0 级光斑**：绝不假设 0 级光斑位于相机画面中心 —— 用 `argmax` 定位。
