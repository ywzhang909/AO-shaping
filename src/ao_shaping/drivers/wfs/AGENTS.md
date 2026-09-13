# wfs/ - 波前传感器驱动

波前传感器 (Wavefront Sensor) 驱动模块, 目前支持 Thorlabs WFS。

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
