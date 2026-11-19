# wfs/ - 波前传感器驱动

波前传感器 (Wavefront Sensor) 驱动模块, 目前支持 Thorlabs WFS。

WFS Python SDK 驱动编写指南 (官方手册 API 全量清单 / ctypes 绑定约定 / 状态位与错误表 / 坑位清单): docs/thorlab-wfs/agent.md

## 结构

```
wfs/
├── base.py              # BaseWFS 抽象基类 (WFS/ABC) —— 真实与仿真驱动的共享契约
├── thorlab_wfs.py       # ThorlabWFS 主驱动
├── _thorlab_wfs.py      # ThorlabWFS 内部实现 (load_dll 等)
├── _registry.py         # WFS 注册表 (惰性解析接入点)
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

### `calc_n_zernike_terms` 的 +1: DLL 的 1 基填充留出的 0 号空槽

`BaseWFS.calc_n_zernike_terms(n)` = `zernike_calc.calc_n_zernike_terms(n) + 1` —— 但多出来的
那一项**不是系数**: canonical 数的是**项数**且已含 piston, WFS 侧数的是**数组长度**, 因为 DLL
按 1 基填 `coeff[1..N]` (@n=10: 66 项 + 0 号空槽 = 67), `index 0` 不写也不用; 消费侧的
`w = w[1:]` 正是这个约定 (`tools/wfs/zernike_response.py`)。两者各自对消费者正确, 但绝不能
混用; 该 +1 关系已由 `tests/ao_shaping/drivers/wfs/test_base_wfs.py` 钉死, 防止无声漂移。

⚠️ **仍未决 (2026-10-25 实测, 无硬件)**: 孪生与驱动的差异是**两处**, 不只是 0 号槽。

- **长度**: 孪生 `SimulatedWFS.get_zernike()` 返回 `len(list_zernike_modes(n))` 长
  (@n=4 → 15, @n=10 → 66), 无空槽; 驱动侧缓冲区是 **67** (@n=10, 含 0 号空槽)。
- **排序**: 孪生是 **0 基 Noll** 序 (实测注入 Noll 4 = (2,0) defocus ⇒ 峰值落在 `index 3`);
  驱动侧是 **1 基、顺序 m 枚举** —— `DLL [5] = (2,0) defocus`
  (`tools/zernike/common.py:8-12`、`tools/wfs/zernike_response.py:15-19`)。

⇒ 同一模式在两侧落在**不同槽位** (defocus: 孪生 3 / 驱动 5), 所以为驱动写的
`w[1:]`、`w[5]` 这类下标用到孪生上会取到**另一个模式**, 且不报错。统一到哪一侧是
**设备侧口径**决定 (需要 DLL 契约与既有落盘语料的口径), 本次**未改动**。

**口径现已在对象上声明** (2026-10-25, 语义未改): `BaseWFS.zernike_layout` 取
`WfsZernikeLayout` 枚举, 孪生声明 `NOLL_ZERO_BASED`、驱动声明
`DLL_ONE_BASED_SEQUENTIAL_M`; 守卫
`tests/ao_shaping/test_conventions.py::test_every_wfs_driver_declares_its_zernike_index_layout`
要求每个具体驱动都声明该标记, 且两侧声明**互不相同** —— 若真要统一, 必须在同一个提交里
同时改动这里的口径与那条守卫, 否则测试判失败。当前没有任何调用方按该标记分支, 它是
可读的边界与护栏, 不是控制流。

**半径与两种失效方式** (2026-10-25 普查): 同时「按类型解析传感器」与「消费 Zernike 向量」的
生产代码只有 `runners/slm/zernike_matrix_runner.py` (4 处); `optimizer/wf/` 的 `rms.py` /
`ga_zernike.py` / `greedy_zernike.py` / `rms_by_zernike.py` 虽解析 WFS, 只读
`get_wavefront()` / `get_spot_deviation()` / `get_spots_statics()`, 不碰系数向量。

- **整数组 `[1:]` 路径会响亮失败**: 矩阵行数钉成 `calc_n_zernike_terms(10)=66`, 孪生 66 长经
  `[1:]` 得 65, 填列时形状错误 (`zernike_matrix_runner.py:630-634` 的注释自记过一次 65-vs-66)。
- **按槽位取单点读数会静默出错**: 对孪生向量套用驱动的 1 基约定读 defocus, 容器实测注入
  0.70 rad 时按孪生口径读回 `+0.7043`、按驱动槽位读回 `-0.0000` —— 一个干净的零, 既不是噪声
  也不是异常。

⇒ 该静默类已咬到过 `measure_tilt_defocus`: 它把槽位 4 (=(2,-2) 像散) 当离焦读, 而 Noll 序号与
DLL 槽位只在 1..3 重合、从 4 起分岔。现由 `tools/zernike/common.py` 的 `TILT_DLL_SLOTS` /
`DEFOCUS_DLL_SLOT` 从 `DLL_ZERNIKE_ORDER` 现取槽位, 回归测试在
`tests/ao_shaping/tools/slm/test_slm_zernike_common.py` (逐槽不同值的存根读数)。
⚠️ `runners/slm/offset_runner.py` 有**两处**读 WFS 统计字典里**无人写入**的键 (键名漂移类):
- `measure_defocus_coefficient()` 硬下标读 `statics["zernike_coeffs"]` —— 驱动统计字典只有
  `min/max/diff/mean/rms/wighted_rms`, 全树无处写该键, 故该函数恒返回 `0.0`;
- 搜索环里两处 `statics.get("total_intensity", 1e6)` —— `total_intensity` 只由 **CCD** 分析路径
  (`utils/image/target/ccd.py`) 写入, WFS 侧从不写 ⇒ **默认值 1e6 无条件生效**, 于是被打印并参与
  比较的"强度"实际是个常数。默认值把本该抛出的 `KeyError` 变成静默, **比硬下标那一处更难发现**。
两处同属"改数据源还是删功能"的行为决定 (该 runner 未注册且计划迁入 `tools/slm/`), 本次只记录。

⚠️ **同一个统计量在仓库里有两种拼写**: 字典键是错拼 `wighted_rms` (驱动与孪生都写它,
`optimizer/wf/rms.py` 还硬下标读它), 而模型层 `model/quantities.py::WavefrontStatistics` 的字段是
正确拼写 `weighted_rms`, 两者靠一个**零调用方**的 `from_legacy` 桥接。改名边界已实测: 键集由
`tests/ao_shaping/drivers/sim/test_simulated_wfs.py` 钉住 (少键会被测试抓到); 该键**不作为结构化
字段落盘** —— `save_history` 写 CSV (整份字典以字符串形式出现在 `_statics` 列), 而
`save_history_hdf5` 直接**跳过**非标量列。⇒ 改名不打断任何**读取**路径, 但历史工件里的文本拼写会
与新版不一致。是否统一属契约决定, 未单方面改动。
⚠️ 该未决项**不波及** `runners/slm/zernike_matrix_runner.py:839/889` 的 `z0[1:]`: 那里的
`z0` 来自 `measure_zernike`, 它正是 `tools/zernike/common.py` 的入口 (runner `:94` 导入),
返回的是原始 67 长数组 (index 0 未用), 因此 `[1:]` 是**正确**用法。风险在别处: 新代码若按
0 基 Noll 去索引驱动读数 (拿 defocus=`[3]`), 就会在硬件上取到另一个模式。

⚠️ **同一类静默偏移仍在旧标定路径 `optimizer/wf/zernike_response_matrix.py` 里** (容器实测, **未修**):
它按 `start_idx += 1` 去 piston、`+= 2` 去 tip/tilt 裁剪 WFS 读数 —— 这是 **0 基 Noll** 的算术, 而读数
是 **DLL 1 基** (index 0 是不写的占位, piston 在槽位 1)。四种组合实测保留的槽位:
`(去 piston, 留 tip/tilt) → 1..66` (**piston 其实还在**)、`(两者都去) → 3..66` (**去掉了 tip 却留着 tilt**)、
`(不去 piston, 去 tip/tilt) → 2..66`、`(都不去) → 0..66`; 且**每种组合都是 rows − cols = 1**。
它**从不广播失败**: `BaseWFS.calc_n_zernike_terms` 本身就是**项数 +1**, 把同一个偏移同时加进了行数,
所以只会让行集合与声明的模式集合差一个槽位 (元数据与调试查看器的行标签跟着偏)。
- **为什么不顺手改**: 当前消费方与此**自洽** (`tools/wfs/zernike_correction.py` 取 `z[1:]` = 66 = 行数);
  改成按槽位身份裁剪会让矩阵**行数 66 → 65**, 使已存盘的矩阵与 `gui/zernike/zernike_debug_viewer.py`
  的行解释失配, 而这是只有实验系统才能重定的数值 ⇒ **设备侧决定**, 本次只记录。在线 CLI 路径不受影响:
  `runners/slm/zernike_matrix_runner.py` 的 `_mode_ids` 已按**身份**去 piston (`i != 1`), 其注释自记过同一个误减。
- SLM 侧该偏移算术已收敛为**唯一实现** `optimizer/wf/zernike_response_matrix.py::slm_mode_slot`
  (原先手抄 3 处: 自动寻幅、标定环的 `if/elif/else`、`gui/zernike/zernike_response_matrix_ui.py`
  的交互回调)。系数数组是 **0 基 Noll** (`index 0 = piston`, 权威见 `zernike_dm._noll_to_dict` 的
  `Noll j = idx + 1`), 故下标偏移 = **被排除的模式个数**。旧注释里的 `i+2` / `i+4` 说的是
  **Noll 编号**, 代码里的 `i+1` / `i+3` 是**数组下标** —— 两者都对, 但混写会让人以为差一个。
- ⚠️ **仍是两处不同的量, 未合并**: (a) `gui/zernike/zernike_debug_viewer.py` 的 `wfs_start_idx` 解释的是
  **WFS 行**, 与 SLM 列偏移不是同一个东西 (上面那条 `start_idx` 未决项就是它); (b) 自动寻幅
  `_optimize_perturbation_amplitude` 的 `excluded_*` 目前由调用点**显式**写死为「两者都排除」,
  **没有**跟随本次标定的排除选择 —— 转发会同时改变被探针的模式与被比对的行集合, 属只有实验系统
  才能重定的数值, 故由 `tests/ao_shaping/optimizer/wf/test_zernike_response_matrix.py` 的
  `TestAmplitudeProbeKeepsItsOwnExclusions` 锁死现状; 要改必须连同设备侧基准一起重定。
- ⚠️ **模态空间回退把「斜率样本」喂给「Zernike 响应矩阵的伪逆」** (读码确认, **未修**):
  `runners/slm/zernike_matrix_runner.py::from_response_matrix` 优先用 `deviation_response_matrix`
  (斜率空间, 与 `_measure_wfs_step` 的 `get_spot_deviation` 同空间 ⇒ 正确); 只有当已存盘的 h5 **缺**
  `deviation_response_matrix` 或 `subaperture_mask` 时才回退到 `D = result.matrix` (行 = WFS Zernike
  通道) 且 `mask_indices = arange(D.shape[0])`, 而测量侧**始终**取斜率 —— 于是实际取的是「前 N 个斜率
  样本」去乘 Zernike 空间的伪逆。**维度自洽所以不会报错, 空间不自洽**。改它等于改闭环控制语义
  ⇒ 设备侧决定, 本次只做构造期校验: `pinv.shape == (D.shape[1], D.shape[0])` 必须成立, 且 `s_ref` 与
  `mask_indices` 同源于 matrix 行数 (守卫测试 `tests/ao_shaping/runners/slm/test_from_response_matrix_guard.py`)。

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
| **pupil（瞳面） 必须 `wfs.pupil = wfs.optimize_pupil()`** | `optimize_pupil()` 只计算并返回, **不调用 `WFS_SetPupil`**, 忘记写回等于没设 pupil（瞳面） |
| **勿硬编码 pupil（瞳面）** | 硬编码 `(0,0,8mm)` 与真实光束不符时, 边界无效子孔径污染 `WFS_ZernikeLsf` 全孔径 LSF 拟合 → 巨大假 tip/tilt (实测 \|z\|=4.6~12.8λ) |
| **本机实测 pupil（瞳面） 参考** (SLM200 + WFS M01219666, 532nm) | 中心 ≈ (-0.148, +0.148) mm, 直径 ≈ 3.62 × 3.88 mm |
| **zernike LSF 是最干净主度量** | 修正 pupil（瞳面） 后倾斜阶梯 0.10~3.20λ 读出 0.0061~0.2169λ, 线性度 R²=0.9603 (plane 度量小倾斜端受噪声干扰) |
| **`get_zernike` 依赖 pupil（瞳面） 正确性** | 系数为 µm (µm/0.532=λ @532nm), **顺序 m 枚举** (1 基、前 66 项, 非标准 Noll 1976), RoC=coeff[5]; 前置 `CalcSpotToReferenceDeviations(0)` |
| **`WFS_ZernikeLsf` typed 绑定被注释仍可用** | `_sdk_bindings.py` L319-320 注释了 `restype/argtypes`, 驱动直接调用未声明函数 (ctypes 宽松传参) 正常工作 — 建议补绑以启用类型检查 |
| **`MlaRes` 枚举成员是 `Res512` 非 `RES_512`** (2026-09-16) | 实际成员: `Res320/Res512/Res768/Res1024/Res1280` (**无 540/600**); 驱动 `__init__` 传 `mla_index=MlaRes.Res512` |
| **WFS 单独运行 80 次捕获无堆损坏 (2026-09-16 实测)** | n=5 zernike-matrix 3/3 崩溃 `0xC0000374` (faulthandler 检测点在 `get_zernike` L1493, 但检测点≠源头); WFS-only 隔离探针 (`scripts/zernike_bench/wfs_probe.py`) 80 iter 零崩溃 → **WFS DLL 侧排除**, 嫌疑在 Santec SLM 写入路径 (4.6MB `dat` 缓冲生命周期)。详见 `report/slm/daily/daily_2026-09-16.md` |
| **WFS 报 "currently in use" 处置** | `open()` 的 `WFS_GetInstrumentListInfo` 返回 `device_in_use=1` 时, 检查是否有 Thorlabs 官方软件 `wfs.exe` 进程 (或其崩溃残留会话) 占用设备 — 关闭后重试即可; 崩溃进程未 `close()` 可能留下 SDK 悬挂会话 |
