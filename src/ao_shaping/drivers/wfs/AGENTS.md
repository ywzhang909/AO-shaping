# wfs/ - 波前传感器驱动

波前传感器 (Wavefront Sensor) 驱动模块, 目前支持 Thorlabs WFS。

WFS Python SDK 驱动编写指南 (官方手册 API 全量清单 / ctypes 绑定约定 / 状态位与错误表 / 坑位清单): docs/thorlab-wfs/agent.md

## 结构

```
wfs/
├── base.py              # BaseWFS 抽象基类 (WFS/ABC) —— 真实与仿真驱动的共享契约
├── thorlab_wfs.py       # ThorlabWFS 主驱动
├── _thorlab_wfs.py      # ThorlabWFS 内部实现 (load_dll 等)
└── __init__.py
```

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `BaseWFS` | `base.py` | **波前传感器抽象基类** (继承 `Device`), 真实与仿真驱动共同实现 |
| `ThorlabWFS` | `thorlab_wfs.py` | Thorlabs 波前传感器 (主入口, `wfs/__init__.py` 从此导出) |
| `MlaRes` | `thorlab_wfs.py` | MLA 分辨率枚举 |
| `WfsError` | `_thorlab_wfs.py` | WFS 异常 |

## BaseWFS —— 共享契约 (2026-10-01 新增)

WFS 家族曾是**唯一没有共享基类**的设备家族: `ThorlabWFS` 直接继承 `Device`, 因此
**不存在任何契约可供仿真 WFS 实现** —— 这正是"仿真 WFS 一直写不出来"的根本原因。
相机家族用 `BaseCamera`、DM 家族用 `DM` 解决了同一问题, 本次为 WFS 补齐。

`BaseWFS` 抽取的正是 optimizer / runner **实际调用**的方法集合 (按 grep 调用次数排序):
`take_image` (28) · `get_wavefront` (17) · `get_spot_deviation` (13) ·
`build_subaperture_mask` (4) · `get_spots_statics` (3) · `calc_n_zernike_terms` (3) ·
`get_zernike` (2) —— 全部为 `@abstractmethod`, 因此**部分实现会在实例化时立刻失败**,
而不会等到运行时才炸。

基类同时拥有**共享参数** (`_register_wfs_parameters`): `exposure_time_ms` ·
`remove_tilt` · `pupil_diameter` · `pupil_center`。驱动可再次声明以覆盖基类值
(后注册者胜出, 因此驱动专用的硬件边界不会被基类默认值覆盖掉)。

### 单位契约 (务必遵守)

`get_zernike()` 返回 **µm**, `get_wavefront()` 返回 **waves (λ)**。这是 WFS 家族
的历史契约, AGENTS.md 记录过两类真实 bug: 响应矩阵按 µm 建而修正用 λ (系数放大
1.88×), 以及 λ 系数直接喂 `make_phase` (相位缩小 2π = 6.28×); 修复后闭环 RMS 改善
13.8% → **42.1%**。**仿真 WFS 必须复刻同一单位**, 内部可用弧度, 但边界转换必须让
仓库既有的 `um_to_waves()` → `×2π` 链路**精确还原**原始弧度相位。

### `calc_n_zernike_terms` 的 piston 偏移

`BaseWFS.calc_n_zernike_terms(n)` = `zernike_calc.calc_n_zernike_terms(n) + 1` ——
**含 piston**, 而 canonical `zernike_calc` 版本**不含**。两者各自对消费者正确, 但
绝不能混用; 该 +1 关系已由 `tests/ao_shaping/drivers/wfs/test_base_wfs.py` 钉死,
防止两套实现无声漂移。

## 惰性加载 (2026-10-01)

`ThorlabWFS.__init__` 曾在**构造期**直接调用 `load_dll()`, 导致任何未安装
Thorlabs/VISA 运行时的机器 (cron、服务、console script、离线报告脚本) 一构造就
`OSError`。现在 DLL 由 `Device._ensure_sdk()` 惰性解析, 需求被推迟到 `open()`:

- `ThorlabWFS._lib` 是只读属性, 委托 `self._ensure_sdk()`; 全部 53 处 `self._lib.<fn>`
  读点**零改动**。
- 仿真设备没有 SDK, `_ensure_sdk()` 自然返回 `None`。
- `tests/ao_shaping/drivers/test_lazy_driver_loading.py` 锁定该行为。

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
