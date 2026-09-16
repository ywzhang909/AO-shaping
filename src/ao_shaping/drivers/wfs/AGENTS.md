# wfs/ - 波前传感器驱动

波前传感器 (Wavefront Sensor) 驱动模块, 目前支持 Thorlabs WFS。

WFS Python SDK 驱动编写指南 (官方手册 API 全量清单 / ctypes 绑定约定 / 状态位与错误表 / 坑位清单): docs/thorlab-wfs/agent.md

## 结构

```
wfs/
├── thorlab/              # Thorlabs WFS SDK
│   ├── driver.py
│   ├── _sdk_bindings.py
│   └── __init__.py
├── thorlab_wfs.py        # ThorlabWFS 主驱动
├── _thorlab_wfs.py       # ThorlabWFS 内部实现
└── __init__.py
```

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `ThorlabWFS` | `thorlab_wfs.py` | Thorlabs 波前传感器 (主入口, `wfs/__init__.py` 从此导出) |
| `MlaRes` | `thorlab_wfs.py` | MLA 分辨率枚举 |
| `WfsError` | `_thorlab_wfs.py` / `thorlab/driver.py` | WFS 异常 |

## ThorlabWFS 接口

| 方法 | 说明 |
|------|------|
| `open()` | 打开 WFS 连接 |
| `close()` | 关闭 WFS 连接 |
| `is_connected()` | 检查连接状态 |
| `get_hardware_info()` | 获取硬件信息 |
| `take_image(n_sample, dynamicNoiseCut)` | 采集点场图像 |
| `get_spotfiled_image()` | 获取点场图像 (512×512 uint8) |
| `get_spots_statics()` | 获取斑点统计信息 |
| `get_wavefront(cancel_tile)` | 获取波前 (含 Zernike 系数) |
| `get_zernike(zernike_order)` | 获取 Zernike 系数矩阵 |
| `get_spot_deviation()` | 获取斑点偏差 |
| `optimize_pupil()` | 优化光瞳 |
| `optimize_exposure_time_and_gain()` | 优化曝光时间和增益 |
| `select_mla(mla_index)` | 选择 MLA 分辨率 |
| `set_ref_plane(custom)` | 设置参考平面 |
| `create_default_user_ref()` | 创建默认用户参考 |
| `save_user_ref(backup_dir)` | 保存用户参考 |
| `load_user_ref(backup_path)` | 加载用户参考 |
| `get_mla_name()` | 获取 MLA 名称 |
| `load_config()` | 加载 JSON 配置文件 |
| `save_config()` | 保存当前参数到 JSON |
| `handle_error(err, no_raise)` | 处理 WFS 错误码 |

### 参数注册

ThorlabWFS 在 `__init__` 中注册参数:

| 参数 | 说明 |
|------|------|
| `exposure_time_ms` | 曝光时间 |
| `master_gain` | 主增益 |

## 关键经验 (2026-09 硬件实测固化)

> 详细分析与完整证据见 [`docs/thorlab-wfs/agent.md`](../../../../docs/thorlab-wfs/agent.md) §6.6/§8。

| 经验 | 说明 |
|------|------|
| **pupil 必须 `wfs.pupil = wfs.optimize_pupil()`** | `optimize_pupil()` 只计算并返回, **不调用 `WFS_SetPupil`**, 忘记写回等于没设 pupil |
| **勿硬编码 pupil** | 硬编码 `(0,0,8mm)` 与真实光束不符时, 边界无效子孔径污染 `WFS_ZernikeLsf` 全孔径 LSF 拟合 → 巨大假 tip/tilt (实测 \|z\|=4.6~12.8λ) |
| **本机实测 pupil 参考** (SLM200 + WFS M01219666, 532nm) | 中心 ≈ (-0.148, +0.148) mm, 直径 ≈ 3.62 × 3.88 mm |
| **zernike LSF 是最干净主度量** | 修正 pupil 后倾斜阶梯 0.10~3.20λ 读出 0.0061~0.2169λ, 线性度 R²=0.9603 (plane 度量小倾斜端受噪声干扰) |
| **`get_zernike` 依赖 pupil 正确性** | 系数为 µm (µm/0.532=λ @532nm), Noll 1976 1-based 前 66 项, RoC=coeff[5]; 前置 `CalcSpotToReferenceDeviations(0)` |
| **`WFS_ZernikeLsf` typed 绑定被注释仍可用** | `_sdk_bindings.py` L319-320 注释了 `restype/argtypes`, 驱动直接调用未声明函数 (ctypes 宽松传参) 正常工作 — 建议补绑以启用类型检查 |
| **`MlaRes` 枚举成员是 `Res512` 非 `RES_512`** (2026-09-16) | 实际成员: `Res320/Res512/Res768/Res1024/Res1280` (**无 540/600**); 驱动 `__init__` 传 `mla_index=MlaRes.Res512` |
| **WFS 单独运行 80 次捕获无堆损坏 (2026-09-16 实测)** | n=5 zernike-matrix 3/3 崩溃 `0xC0000374` (faulthandler 检测点在 `get_zernike` L1493, 但检测点≠源头); WFS-only 隔离探针 (`scripts/wfs_probe.py`) 80 iter 零崩溃 → **WFS DLL 侧排除**, 嫌疑在 Santec SLM 写入路径 (4.6MB `dat` 缓冲生命周期)。详见 `docs/slm/daily_2026-09-16.md` |
| **WFS 报 "currently in use" 处置** | `open()` 的 `WFS_GetInstrumentListInfo` 返回 `device_in_use=1` 时, 检查是否有 Thorlabs 官方软件 `wfs.exe` 进程 (或其崩溃残留会话) 占用设备 — 关闭后重试即可; 崩溃进程未 `close()` 可能留下 SDK 悬挂会话 |
