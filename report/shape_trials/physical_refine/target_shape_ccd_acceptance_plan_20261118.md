# 下一轮目标形状 CCD 验收准备（离线计划，2026-11-18）

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **数据/关联脚本**: [`scripts/slm/validate_transfer_shapes_hardware.py`](../../../scripts/slm/validate_transfer_shapes_hardware.py)
> **数据/关联脚本**: [`scripts/slm/replay_transfer_shapes_hardware.py`](../../../scripts/slm/replay_transfer_shapes_hardware.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/validate.py`](../../../src/ao_shaping/tools/slm/transfer/validate.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/replay_phases.py`](../../../src/ao_shaping/tools/slm/transfer/replay_phases.py)
> **数据/关联脚本**: [`src/ao_shaping/tools/slm/transfer/protocol.py`](../../../src/ao_shaping/tools/slm/transfer/protocol.py)
> **运行环境**: 离线计划；未连接或占用 SLM、CCD、激光（本轮未执行）
> **说明**: 目标形状 CCD 验收的条件计划：先补观测模型（静态 PSF/散斑、成像展宽）并给出独立 holdout 下限改善，再按独立相位合成 + measured-vs-target 指标门槛 + 带锁 replay 执行。当前未放行；本轮已实现并测试 protocol.target_intensity_rmse（整窗归一化、背景 floor、rel_l2/energy_fraction），replay_phases 已写入 target_rmse 字段，并新增 replay_transfer_shapes_hardware.py 带锁外壳（仅 mock 验证，--execute 未执行）。阈值须现场预注册；任何指标均不替代独立 CCD 证据。
<!-- provenance:end -->

## 1. 当前门禁与证据边界

- 本轮 `fit_m3_full` 在 12 个 holdout 上的预测指标有改善，但 `smooth` / `bandlimited` 大行程族改善很小；这不支持直接认定模型适合目标相位优化。
- `fit_m3_full_z21`（扩展残差基 Noll 5..21）容量检验已完成：test smooth relative L2 0.5803 → 0.6058、bandlimited 0.7357 → 0.7449，聚合 0.5657 → 0.5755，收敛下界没有移动。因此**当前门禁是"先离线补观测模型（静态 PSF/散斑、成像展宽）"，不是继续准备 CCD 验收**。
- 离线诊断第一步已完成（见主计划 §8.6）：共享静态高频在全部 68 探针上成立、在独立验证集上可泛化且独立于相位；固定预测下扣静态项后 test/train 的 smooth/bandlimited relative L2 可重复下降。这是"上界诊断"，不是已完成的模型改动——**仍须把静态项与展宽真正加进物理预测模型、重新拟合并通过冻结 holdout 对照**，才算满足下限改善门槛。
- 本文件因此降级为条件计划：仅当完整模型下离线下限改善且重新给出独立 holdout 证据后，才按 §3 执行。
- 即便离线指标通过，后续仍须分别保留：模型优化生成相位、CCD 独立测量、精确相位重放。任何模型 loss 或预测指标均不构成 shaping 验收。
- 当前文档不授权连接设备；开机、曝光和几何参数全为待现场确认，不给出可直接执行的 `--execute` 命令。

## 2. 已有入口与缺口

### 已有入口

1. **锁安全外壳（已就绪，仅设备无关路径验证过）**：`scripts/slm/validate_transfer_shapes_hardware.py`。原样转发 `validate_transfer_shapes.py` 全部参数，只多 `--wait-until-idle`（默认 120 s，0 = 不等待不加锁）。无 `--execute` 时不等待/不加锁/不导入设备模块；有 `--execute` 时懒导入 `device_mutex`，等空闲、持锁贯穿验证器、`finally` 释放并写 `device_lock.json`，等不到或取锁失败则不启动验证器。mock 测试 11 项通过；对 `fit_m3_full` checkpoint 的 plan-only 冒烟（exit 0，仅产出 `plan.json`）已验证转发。**其 `--execute` 路径本轮从未执行。**
2. **相位优化与可选 CCD 候选检查**：`scripts/slm/validate_transfer_shapes.py`，底层 `transfer.validate`。加载 `frozen_parameters.json`，目标由内置 shape 选择器产生（`circle/square/rectangle/triangle/star`），默认 `mode=model`；显式 `--execute --ccd-check` 才会走同曝光 CCD 候选检查，`--abba` 可在需要漂移抵消时启用 A–B–B–A。`--fixed-exposure-ms` 固定块间曝光，`--zero-xy` 和可选 `--zero-radius` 固定零级识别/排除口径。
3. **精确相位重放（现已带锁安全外壳）**：底层 `python -m ao_shaping.tools.slm.transfer.replay_phases --initial-run DIR --runs DIR... --out DIR --exposures MS... --frames N` 从验证运行归档读取 `<shape>_seed` / `<shape>_final` 相位及其目标/core mask，重放原相位并重新测量；不是任意 `.npy` 输入接口。本轮新增 `scripts/slm/replay_transfer_shapes_hardware.py` 外壳：消费 `--execute` 门禁（不转发给底层模块）、等空闲、持锁贯穿整个重放进程、`finally` 释放并写 `device_lock.json`；无 `--execute` 时拒绝且不导入设备模块。`--out` 与 `--exposures` 均强制显式给出（拒绝复用历史 50/100 ms 默认扫档）。16 项 mock 测试通过；**其 `--execute` 路径本轮从未执行**。直接跑底层 `replay_phases.py` 仍无锁，不得走该裸路径。
4. **measured-vs-target intensity RMSE（现已实现并测试）**：`src/ao_shaping/tools/slm/transfer/protocol.py::target_intensity_rmse`。`replay_phases` 已把它作为每条记录的 `target_rmse` 字段写入（目标与采样窗同 grid 才计算，否则为 `None`，不在采集中途因 grid 不一致中断）。算法契约、阈值边界见下节。原验证/重放的 CV、EE_ROI、峰值、Pearson 仍是独立指标，不得与 RMSE 名称互换。

### measured-vs-target 强度 RMSE：已实现算法与阈值边界

- **比较对象**：整窗的实测强度分布 vs 目标强度分布（不是预测 vs 实测）。目标须在**与实测同采样窗/同像素 grid** 上给出；`target_intensity_rmse` 在形状不一致时抛 `ValueError`，绝不静默评分。现场须在冻结阶段固定 target 的像素坐标、尺寸、mask 与窗口，见 §4。
- **归一化**：两侧各自按**整窗**和归一化后再相减（不是只在目标支撑内）。因此 laser 功率、曝光、相机增益从指标中消去；落在目标外的光会计入距离，并单列为 `energy_fraction`，"形状对但位置错"无法被单一数字掩盖。
- **背景处理**：`floor` 为当次同曝光实测的暗场+杂散背景估计，先减去、负值截 0，再做归一化。值来自本次 run，执行前冻结；不得沿用历史曝光/背景。
- **输出字段**（冻结契约 `TARGET_INTENSITY_KEYS`）：`rmse`（单位-sum 归一化强度的每像素 RMS）、`rel_l2 = ||m-t|| / ||t||`（跨 grid 可比的头条指标）、`energy_fraction`、`pixels`。任一侧无能量时三者均为 `None`（与 `cv=None` 同约定：无信号不得读成"完美匹配"）。
- **失败阈值边界**：阈值数值**待现场预注册**（§4），不在本离线阶段编造。通过条件须同时满足 `rel_l2 <= 预设上限` 与 `energy_fraction >= 预设下限`，两者都在冻结的背景 floor 与固定窗上计算；`None` 结果一律判不通过/不确定，绝不判通过。阈值不得用产生它的同一批数据事后选取；冻结后只在最终 CCD 测试上评分一次。
- **不可替代**：任何模型 loss / 前向预测指标（含 refit-forward）都不构成该 RMSE，也不得覆盖它。

### 仍需满足的安全与独立性条件

- 验证侧和 replay 侧现均有锁安全外壳；但直接运行 `replay_phases.py` 或 `validate_transfer_shapes.py --execute` 都没有共享锁/等待，不得再走这两条裸路径。
- 外壳的 `--execute` 分支只有 mock 测试证据，尚无任何真实实验验证；首次使用须由操作者确认设备空闲并在旁监督，等待/取锁失败即视为未通过，不得绕过。
- `--plan-only` / `--help` 是设备无关检查，不是硬件验收，也不生成 CCD 证据。
- 当前 CLI 只支持上述内置形状，不支持提供任意目标强度图与 mask。若本轮目标不是内置形状，必须另做 target/mask 输入适配并冻结其像素坐标、尺寸和 mask，不能临场手动改图。
- Replay 默认曝光为 50/100 ms 的历史扫档；现场命令必须显式给 `--exposures`，且以当次相机读回范围及现场安全上限为准。

## 3. 拟定实验顺序（仅计划）

1. **在线门禁**：确认当次设备空闲与实验授权；按项目光路标定流程检查 SLM/CCD、有效位深、光路调制、曝光范围及满量程。完整帧采集，裁剪/mask 仅在软件中处理。发现饱和或光路异常立即停止，不使用历史曝光值。
2. **锁安全计划检查**：在无设备运行时先执行包装器 `--help` 和 plan-only；确认最终 run 目录不存在、target/zero-order 坐标和 CCD 尺度单位明确、`device_lock.json` 记录策略有效。
3. **冻结目标与模型（门槛已变更）**：先完成观测结构诊断（静态 PSF/散斑项、`sigma_ccd_px` 类展宽项、曝光/采样模型），并用未参与拟合的整相位族验证 smooth/bandlimited relative L2 收敛下界可重复下降；确认后才冻结 checkpoint SHA256、目标形状/边长或输入图像、target/core mask、zero-order exclusion、窗口、优化设置及接受标准。明确拟合/模型选择/最终 CCD 测试各用哪些数据。
4. **模型候选生成与 CCD 检查**：使用锁安全的 `validate_transfer_shapes_hardware.py`。选择一个事先指定形状；固定 checkpoint、几何、目标与 exposure，记录 seed/candidate phase 和完整 CCD 帧。若漂移明显，显式启用 ABBA，并保持各块曝光相同。
5. **独立精确重放**：对验证归档中的 seed/final phase 使用锁安全 replay 包装器，以同一固定曝光、相同目标/mask 做独立复测；保留重放原相位和原始满帧。不得只报被选中的 best，seed/final 和拒绝候选也保留索引。
6. **分析与释放**：重算固定 mask 下 CV、EE_ROI、核心均值、饱和/满量程与 Pearson；报告同相位重放差异、时间顺序与漂移。若要求 RMSE，先按固定归一化/背景规则实现。记录设备关闭、释放和锁 receipt；独立 CCD 证据达不到预注册条件则判未通过/不确定，不以模型分数覆盖。

## 4. 执行前填写（不可沿用旧会话推断）

| 条目 | 当次值 | 来源/时间戳 |
| --- | --- | --- |
| 目标形状（内置或明确的图像文件/hash） | 待定 | 待定 |
| 目标尺寸及坐标系（CCD px / SLM px / 网格分别注明） | 待定 | 待定 |
| checkpoint 路径、SHA256、residual basis | 待定 | 待定 |
| zero-order center (CCD px) 与 exclusion radius | 待定 | 待定 |
| target center、window span/step、固定 core mask | 待定 | 待定 |
| CCD 型号、ID、有效 bit depth/full scale、原生 frame shape | 待定 | 当次读回 |
| 曝光固定值、硬件范围、环境/操作者上限及读回值 | 待定 | 当次读回 |
| frames、settle、ABBA 是否启用及理由 | 待定 | 待定 |
| 预注册通过/拒绝条件（含 rel_l2 上限与 energy_fraction 下限） | 待定 | 执行前批准 |
| measured-vs-target 背景 floor 来源（同曝光暗场）、目标 grid/窗冻结 | 待定 | 执行前批准 |
| 设备授权、run 时间窗、操作者 | 待定 | 待定 |

## 5. 后续命令骨架（仅在上述字段完成、门禁批准、锁包装器验证后使用）

以下是命令形状，不包含可沿用的历史几何或曝光，也不应在当前离线阶段运行。实际值只能来自下一次现场校准与授权；确认 out 不存在。

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src').Path
$checkpoint = '<approved frozen_parameters.json>'
$out = '<new unique run directory>'
$zeroX = <measured CCD px>
$zeroY = <measured CCD px>
$targetX = <measured CCD px>
$targetY = <measured CCD px>
$exposureMs = <current unsaturated exposure ms>

# Device-free plan only; this writes no CCD evidence. For a later approved run,
# use a fresh output path, remove --plan-only, and append --execute --ccd-check.
& .\.venv\Scripts\python.exe scripts\slm\validate_transfer_shapes_hardware.py `
  --checkpoint $checkpoint --out $out --plan-only `
  --shapes <one-of-circle-square-rectangle-triangle-star> `
  --zero-xy $zeroX $zeroY --target-xy $targetX $targetY `
  --fixed-exposure-ms $exposureMs --frames 3 `
  --zero-radius <measured exclusion radius CCD px> --wait-until-idle 120
```

Replay must use the lock-safe wrapper (now delivered); pass the validation archive and exactly the same measured exposure, not the replay CLI's historical default sweep. The wrapper consumes the `--execute` gate itself and forces explicit `--exposures` and a fresh `--out`:

```powershell
& .\.venv\Scripts\python.exe scripts\slm\replay_transfer_shapes_hardware.py `
  --initial-run <seed archive> --runs <validation archive> `
  --out <new unique replay directory> --exposures <same exposure ms> --frames 3 `
  --wait-until-idle 120 --execute
```

Without `--execute` the wrapper only refuses (no device wait/lock/import). `--help` is device-free. The low-level `python -m ao_shaping.tools.slm.transfer.replay_phases` form has no lock and stays documentation-only. This skeleton is documentation only; do not run it in the offline phase.

## 6. 实验中/后记录模板（复制到独立 run report 后填写）

### 6.1 过程与证据链

- 日期/本地与 UTC 起止：
- 操作者/授权与实验时间窗：
- SLM / camera / laser 身份及是否启用：
- 光路标定记录路径；有效位深、原生 frame shape、full scale：
- checkpoint SHA256 / source / phase basis：
- target 文件/shape、坐标单位、target/core mask 路径与 hash：
- zero-order center/radius、窗口、曝光限制来源：
- 当前实测 exposure 与每帧 exposure readback：
- seed、candidate、拒绝候选、final phase 的标签和 artifact 路径：
- CCD 原始满帧路径、帧数、时间顺序、settle、剔除帧及原因：
- acceptance 模式（sequential / ABBA / replay）和固定曝光：
- device-lock wait/acquire/release receipt 与输出目录：

### 6.2 结果表（固定 mask / 同曝光）

| 标签 | phase hash | 序列位置 | exposure ms | saturation px / max fraction | core mean | core CV | EE_ROI | Pearson | rel_l2 / RMSE / energy_fraction | 判定 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| seed A1 |  |  |  |  |  |  |  |  |  |  |
| candidate B1 |  |  |  |  |  |  |  |  |  |  |
| candidate B2（ABBA 时） |  |  |  |  |  |  |  |  |  |  |
| seed A2（ABBA 时） |  |  |  |  |  |  |  |  |  |  |
| exact replay seed/final |  |  |  |  |  |  |  |  |  |  |

### 6.3 结论与释放

- 预注册门槛及逐项结果：
- 同相位重复差异/漂移判断：
- 可支持的结论（模型预测 / CCD shape acceptance 分开）：
- 失败或不确定项、下一项最有区分力的实验：
- SLM/camera/laser 是否关闭并释放；释放 UTC；`device_lock.json` 是否确认 `released=true`：

> 设备占用结束后，回复中必须明确报告占用设备、操作、释放时间和是否正常关闭。当前本计划编写阶段未占用任何设备。
