# Environment Variables — AO-Shaping

> 外迁自根 `AGENTS.md`（Configuration / Environment Variables 两节）。

## Configuration

### Environment Variables (.env)

Project uses `.env` file for environment configuration:

```bash
# Hardware device IDs
Far_Cam_ID=0
Near_Cam_ID=1

# Optical parameters
IDEAL_SPOT_RADIUS=7
# CENTER is not an env knob -- no code has ever read it. The analysis centre
# comes from the runner option -c/--center (auto = locate the 0-order by argmax).

# Camera protection exposure ceiling (ms). Optical-path dependent, set per
# session; unset or 0 = code probes the CCD image (doubling ladder to saturation)
# AO_DAHENG_MAX_EXPOSURE_MS=0.6

# SLM 2π gray: NOT an env knob any more (2026-10-23). MAX_GRAYSCALE_VALUE=1023 is
# only the radian→gray scale; the real 2π gray comes from the device phase table
# (SLM_Ctrl_ReadWL → 1064 nm phase=206 ⇒ 2.06π ⇒ 993) and the device **rejects**
# smaller values (196/200 snap to 206), so it cannot be forced to 1023.
# Zero-order floor saturates at 2.86% for phase ≥ 2.06π (ff≈0.854, η±1_max≈97.1%),
# so the device value is self-consistent. Full logic + evidence pointers live in
# the module comment above `src/ao_shaping/drivers/slm/santec/driver.py:MAX_PIXEL_FLIP_TIME_S`;
# probes: scripts/slm/probe_wl_phase_vs_efficiency.py, probe_gs_ceiling_vs_efficiency.py

# Library paths
PYTHONPATH=src;libs
PATH=libs\Drv_UDPST\x64\Release;libs\gxipy;${PATH}
```

**`libs` 为什么还在 PYTHONPATH 上**: `libs/` 里是不进 pip 源的厂商 Python 包
(大恒 `gxipy`、Santec SLM SDK 等), 只能由仓库把目录加进搜索路径。
大恒 `gxipy` 现在可由 `python scripts/_common/link_vendor_sdks.py` 以目录符号链接
装进 venv (指向 `libs/gxipy`), 装过之后 `import gxipy` **不再需要** PYTHONPATH。
但符号链接不解决 **DLL 搜索**: `gxipy` 按名字加载原生库 (`gxwrapper.py` /
`dxwrapper.py` 里的 `GxIAPI.dll` / `DxImageProc.dll`), 所以桌面界面
`python -m ao_shaping.gui.ccd.beam_quality` 运行时 PATH 仍须含这些 DLL 所在目录
(GalaxySDK 安装目录, 如 `...\GalaxySDK\APIDll\Win64`; 上面 PATH 里的
`libs\Drv_UDPST\x64\Release` 提供的是 `Drv_UDPST.dll`)。

### VSCode Settings

VSCode settings are configured in `.vscode/settings.json` (machine-local, **not tracked** since 2026-10-19 — create it yourself on a fresh clone if you use VS Code):
- Python path includes `src/` and `libs/`
- Pytest integration enabled
- Terminal environment variables from `.env`

### Config Module

Centralized configuration in `src/ao_shaping/config.py`:

```python
from ao_shaping.config import DM_N_ACTUATORS, DEFAULTS, PATHS

# Hardware constants
DM_N_ACTUATORS = 64

# Default optimization parameters
defaults = DEFAULTS
print(defaults.WF_EPOCHS)  # 20000

# Path configuration
paths = PATHS
print(paths.root_dir)  # data/
```

## Environment Variables

Configuration via `.env` file:
```
Far_Cam_ID=0
Near_Cam_ID=1
IDEAL_SPOT_RADIUS=7
# CENTER 不是环境变量: 分析中心由 runner 选项 -c/--center 给出。
# Daheng 保护曝光上限 (ms), 随光路按会话设置; 缺省/0 = 由画面探测 (倍增阶梯到饱和)
# AO_DAHENG_MAX_EXPOSURE_MS=0.6
```

### 不在 `.env` 里的旋钮 (设备路径与诊断开关)

下列名字由生产代码直接读取, 不在 `.env` 中; 每个都在**模块作用域**求值, 所以必须在启动
Python 之前设好, import 之后再 `os.environ[...]` 赋值不会生效。`test_env_names_read_by_code_are_documented`
把这张表与代码里实际读取的名字对齐。

| 名称 | 读取位置 | 未设置时的默认 | 作用 |
|---|---|---|---|
| `SLM_CONFIG_DIR` | `src/ao_shaping/drivers/slm/santec/driver.py` | `<repo>/data/slm_configs` | SLM 相位表 / LUT 配置目录 |
| `DM_CONFIG_DIR` | `src/ao_shaping/drivers/dm/nlight/driver.py` | `<repo>/data/dm_configs` | NLight 变形镜配置目录 |
| `MICRO_DM_CONFIG_DIR` | `src/ao_shaping/drivers/dm/micro/driver.py` | `<repo>/data/micro_dm_configs` | 微变形镜 (R50Power) 配置目录 |
| `CCD_CONFIG_DIR` | `src/ao_shaping/drivers/ccd/daheng/driver.py` | `<repo>/data/ccd_configs` | Daheng 相机配置目录 |
| `AO_SHAPING_SLM_DLL_DIR` | `src/ao_shaping/drivers/slm/santec/_slm_win.py` | 无 (按候选目录搜索) | `SLMFunc.dll` 的第一候选目录; 未设则再找 `docs/SLM_DLL_ver.2.51`, 仍找不到就 `OSError` |
| `MIICAM_SDK_PATH` | `src/ao_shaping/drivers/ccd/miicam/_sdk_setup.py` | 无 (回落链) | MiiCam SDK 目录, 优先级最高; 否则依次用随仓的 `src/ao_shaping/drivers/ccd/_miicam_sdk` 与 `libs/miicamsdk.20240728/python` |
| `AO_SHAPING_SDK_PATH` | `src/ao_shaping/tools/slm/transfer/hardware.py` | `<repo>/libs` | 存放 `gxipy` 厂商绑定的父目录 |
| `AO_OOPAO_SOURCE_DIR` | `src/ao_shaping/drivers/sim/_oopao_compat.py` | 未设 (通常不需要) | 追加到 OOPAO 影子包 `__path__` 的额外搜索路径; 核心 18 个模块不靠它也能加载 |
| `WFS_DEBUG` | `src/ao_shaping/drivers/wfs/thorlab_wfs.py` | `0` | 置 `1` 时在 Thorlabs WFS 读取路径上记录每帧波前的方差 / 标准差 |
| `RMS_ZERNIKE_DEBUG` | `src/ao_shaping/optimizer/wf/rms_by_zernike.py` | `0` | 置 `1` 时打开 SLM Zernike RMS 优化器的逐轮调试输出 |
| `RMS_ZERNIKE_WEIGHTS` | `src/ao_shaping/optimizer/wf/rms_by_zernike.py` | `(n+1)/n_max` | 逗号分隔的逐阶扰动权重; 解析失败会记 warning 并保留默认值 |
| `NUMBA_CACHE_DIR` | `src/ao_shaping/__init__.py` 与 `src/ao_shaping/utils/numba_compat.py` (import 期) | 未设或非绝对路径 ⇒ `<tempdir>/ao_shaping_numba_cache` | numba JIT 缓存目录: 把 `@njit(cache=True)` 的 `.nbi`/`.nbc` 挪出 `src/`, 既守住「import 不落盘」契约也让只读挂载可用; 用户显式设**绝对路径**时不覆盖 |
| `QT_QPA_PLATFORM` | `scripts/slm/capture_transfer_desktop_screens.py` (PySide6 import 前) | `offscreen` (setdefault) | 无显示环境 (CI / 无桌面会话) 下走离屏后端; Windows 离屏字体库为空, 需显式装字体 |
| `AO_SKIP_CUBLAS_PRELOAD` | `src/ao_shaping/_cublas_preload.py` (`import cupy` 之前, 经 `utils/image/spots_calc.py` 与 `drivers/sim/_oopao_compat.py`) | 未设 ⇒ 照常钉住 | 置任意非空值即跳过「把 `cublas`/`cublasLt` 两个裸名钉到同一个自洽 build」的预载; 只在排查 Windows torch/CuPy CUDA 冲突或量 import 成本时用 — 后果见下方 §cuBLAS DLL 与 PATH |

平台自带变量 (`COMPUTERNAME`, `USERPROFILE`, `XDG_DATA_HOME`) 也被驱动用来定位配置落盘位置,
但它们由操作系统提供而非本仓定义, 因此不列入上表, 并在该守卫里被显式豁免。

### cuBLAS DLL 与 PATH

Windows 上 torch 与 CuPy 抢的是**同一个裸名** (`cublasLt64_13.dll` / `cublas64_13.dll`),
而加载器按「谁先进进程」定终身。两个名字落到**同一个 build**就一切正常, 落到**两个不同
build**就出事 —— 崩的是 torch 的 matmul (`0xc0000005`, 无 traceback), 或者全线 CUDA 调用
报错。2026-11-03 实测 (RTX 5080 Laptop, torch 2.14.1+cu130, cupy-cuda13x 14.2.0, PATH 首位
= toolkit v13.2):

| 裸名最终落到 | cupy matmul | cupy `linalg.inv/pinv` | torch matmul |
|---|---|---|---|
| 都 = PATH toolkit (1340) | OK | OK | OK (跑在更新的 cuBLAS 上, minor 兼容) |
| Lt = torch (1300), cublas = toolkit (1340) | **FAIL** `CUBLAS_STATUS_INVALID_VALUE` | **FAIL** `CUSOLVER_STATUS_INTERNAL_ERROR` | **FAIL** |
| 都 = torch (1300) | OK | **FAIL** (cusolver 只存在于 CUDA 安装里, 与 1300 异源) | OK |
| **都 = PATH toolkit, 由 `pin_cublas()` 钉住** | OK | OK | OK |

因此 `pin_cublas()` 钉的是 **PATH 上第一个含同名 DLL 的 CUDA 目录里的那一对** (toolkit 版本
≥ torch 自带版本时), 而不是 torch 自带的副本 —— torch 只 ship `cublas/cublasLt/cudart` 三个
`*64_13.dll`, 供不上 cupy 需要的 `cusolver/cufft/curand`, 反过来钉会让 cuSOLVER 路径坏掉
(OOPAO 的 `LiFT.py` 用 `xp.linalg.pinv`, 是真实调用点)。成本实测 65 ms。

**以下做法已实测无效, 不要再试**:

| 说法 | 实测 |
|---|---|
| 「`.venv` 隔离就不继承全局 CUDA 了」 | 不成立: 虚拟环境不改 `PATH`, cupy 照旧绑到 `C:\Program Files\...\v13.2` |
| 「用 `uv run` 就干净了」 | 不成立: `uv run` 继承调用方 shell 的 `PATH` |
| 「用 `os.add_dll_directory(torch\lib)` 提优先级」 | 无效: ctypes 裸名 `LoadLibraryExW(..., 0)` 不查 user dirs, 绑定的仍是 v13.2 那份 |
| 「把 CUDA 目录从 `PATH` 删掉」 | 本机无效: scoop `C:\Users\<user>\scoop\apps\cuda\current\bin\x64` (6.14.11.1380) 仍在, 换成它而已 |
| 「`PATH` 里把 `torch\lib` 置前」 | **有效** (实测 cupy 改绑 torch 那份), 但那是机器级配置、不随仓库走, 且仍留 cuSOLVER 异源问题 |

**根治**: 让机器上只有**一套** CUDA minor, 且与所选 torch extra 同源。本机 `PATH` 首位是
v13.2 = `6.14.11.1340`, 而 `torch 2.14.0+cu132` 自带的正是 `6.14.11.1340` —— 即用
`uv sync --extra torch-cu132` 后两边字面同版本, 任何导入顺序都安全 (代价: torch 从 2.14.1
退到 2.14.0)。各 DLL 版本: toolkit v13.0 = `6.14.11.1310` / v13.1 = `6.14.11.1321` /
v13.2 = `6.14.11.1340` / scoop cuda current = `6.14.11.1380` / torch cu130 = `6.14.11.1300`。
入口规则见 `AGENTS.md` §CUDA 原生库入口, 门禁
`tests/ao_shaping/test_native_cuda_import_gate.py`, 实现与全部证据
`src/ao_shaping/_cublas_preload.py`。

Access in code:
```python
import os
cam_id = int(os.environ.get('Far_Cam_ID', 0))
```
