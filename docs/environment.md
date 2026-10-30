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

平台自带变量 (`COMPUTERNAME`, `USERPROFILE`, `XDG_DATA_HOME`) 也被驱动用来定位配置落盘位置,
但它们由操作系统提供而非本仓定义, 因此不列入上表, 并在该守卫里被显式豁免。

Access in code:
```python
import os
cam_id = int(os.environ.get('Far_Cam_ID', 0))
```
