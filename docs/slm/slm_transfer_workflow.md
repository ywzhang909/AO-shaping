# 三形状物理模型整形：统一入口操作手册

统一入口为 `python -m ao_shaping.tools.slm.transfer`。它在一个会话目录中管理探针采集、物理模型迁移、GS 初始化、曝光响应诊断、展宽校准、CCD 锚定可微优化及原始数据验收。实验结果与适用范围见 [实验复现报告](../../report/shape_trials/multishape_capability/physical_multishape_reproduction_20261023.md)，本页说明操作方法。

## 1. 准备环境

在仓库根目录执行：

```powershell
Set-Location D:\Projects\TIFO\AO-shaping
$py = '.venv\Scripts\python.exe'
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = "data/debug/slm_transfer/$tag"
$checkpoint = 'data/debug/calibrate_transfer_blur/shapes_20261023_1956/frozen_parameters.json'
```

需要项目 Python 环境、PyTorch、NumPy、SciPy、Matplotlib，以及设备对应 SDK。当前物理模型常量针对 Santec 1200×1920、8 μm、1064 nm；更换波长或面板规格必须先调整模型并验证传播采样。相机通过现有工厂指定，默认 Daheng #0，SLM 默认 #1；其他相机后端仅有接口支持，不代表已完成实验验证。

Daheng 绑定优先使用已安装的 `gxipy`；设备操作时若未安装，则显式查找仓库 `libs/gxipy`。独立安装的 SDK 可设置 `AO_SHAPING_SDK_PATH` 为包含 `gxipy/` 的父目录。计划、离线拟合、回放和包导入不加载设备 SDK。

## 2. 先看计划，再执行

```powershell
& $py -m ao_shaping.tools.slm.transfer --help
& $py -m ao_shaping.tools.slm.transfer plan --out $out --checkpoint $checkpoint
& $py -m ao_shaping.tools.slm.transfer doctor --out "$out-device.json"
```

`doctor` 只枚举 Daheng，不打开图像流、不操作 SLM。枚举到设备不等于激光已开启，也不证明没有其他程序占用。实际实验前确认激光、光路、设备占用状态，采集期间保持光路和功率稳定。

正值环境变量 `AO_DAHENG_MAX_EXPOSURE_MS` 是当前光路的附加曝光上限。缺失或为 0 时，由设备曝光范围和画面反馈搜索，不继承历史固定上限。修改曝光上限只影响后续会话。

完整执行：

```powershell
& $py -m ao_shaping.tools.slm.transfer run --out $out --checkpoint $checkpoint --execute
```

`run` 不带 `--execute` 或 `--offline` 时只打印计划，不创建会话、不访问设备。默认流程为：

1. 当前光路采集：平相位参考、双轴正负倾斜和 24 个分组随机探针，每组 3 帧。
2. 物理参数迁移：60 次拟合，训练与验证相位分离。
3. 正方形、长方形、三角形 GS 初始化与模型模式配对测试，每种 2 块。
4. 固定相位曝光阶梯诊断，以及有效展宽/共享辐射尺度校准。
5. 以上一步接受相位启动 CCD 锚定优化，每种 4 块，每块 10 次模型更新。
6. 逐文件重算原始帧指标、曝光和相位核对、A–B–B–A 判定核对；生成对照图、方向检查和审计文件。

可以指定 `--count`、`--iterations`、`--blocks`、`--updates`、`--frames`、`--camera`、`--cam-id`、`--slm-number`、`--side`、`--period`。减少块数是短时流程验证，不应当作默认流程性能结论。改变目标后必须使用新目录。

完整校准需要正方形、三角形用于拟合，长方形用于验证。只测试一个形状时显式跳过此阶段：

```powershell
& $py -m ao_shaping.tools.slm.transfer run --out $out --checkpoint $checkpoint --shapes square --skip-calibration --execute
```

`--side 80` 对应正方形边长 80、长方形 160×80、三角形外接圆半径 40，单位均为 CCD 像元；三角形不以边长解释此参数。载波和目标尺寸必须使目标完全落入固定窗口。脚本会在打开设备前检查目标边界以及全部回放相位的网格/掩膜一致性。

## 3. 没有历史检查点

省略 `--checkpoint` 即从新采集的几何建立名义初始化，再拟合：

```powershell
& $py -m ao_shaping.tools.slm.transfer run --out $out --execute
```

名义入射中心、宽度和零离焦只是初值，不代表已标定。较强离焦或入射偏移需要先用面板/离焦探测缩小初始化范围。需要控制初值时，先采集、显式生成初始化文件，再用该文件执行统一流程：

```powershell
& $py -m ao_shaping.tools.slm.transfer.capture --out "$out-capture" --count 24 --execute
& $py -m ao_shaping.tools.slm.transfer.bootstrap --dataset "$out-capture" --out "$out-initial" --center 960 600 --width 400 300 --defocus 0
& $py -m ao_shaping.tools.slm.transfer run --out $out --dataset "$out-capture" --checkpoint "$out-initial/frozen_parameters.json" --execute
```

`--dataset` 仅适合光路未变化的当前采集，或离线检验；它会跳过重新采集。若位置、像差、功率或几何发生变化，默认重新采集。名义模型不能保证从任意光路一次收敛。

## 4. 设备不可用时完整检查计算流程

```powershell
$capture = 'data/debug/transfer_multishape_experiment/capture_20261023_1938'
& $py -m ao_shaping.tools.slm.transfer run --out $out --dataset $capture --checkpoint $checkpoint --offline --iterations 2 --updates 3
```

该命令进行实际模型拟合、三形状各 80 次 GS 和指定次数的可微相位更新，保存预测与相位；不调用相机或 SLM。2 次拟合仅用于入口衔接检查，完整离线模型研究应增加迭代并检查验证指标。离线预测改善不能代替 CCD 实验验收。

## 5. 回放、审计和移交

```powershell
$run = 'data/debug/validate_transfer_shapes/anchored_20261023_2002'
& $py -m ao_shaping.tools.slm.transfer audit --run $run --out "$out-audit.json"
& $py -m ao_shaping.tools.slm.transfer replay --run $run --out "$out-replay"
```

`audit` 从全部原始帧重新计算核心 CV、均值、EE_ROI、峰值和图像目标，核对模型/面板相位、灰度尺寸、目标核心掩膜、曝光、主机时钟及逐块接受判定，给出逐 NPZ 的 SHA-256。`replay` 先审计，再生成 CCD/模型对照和保存方向上的数值梯度核对，不加载相位。

目录移动后，记录中的旧绝对检查点路径会沿当前运行目录的祖先查找相同 `data/` 后缀；找不到则明确报错。移交必须包含完整数据目录和检查点，单独的报告 PNG 不足以复算。

会话目录中的内容：

| 路径 | 内容 |
|---|---|
| `session.json` | 本轮参数、阶段计划、时间与设备操作模式 |
| `journal.json` | 各阶段开始/结束/失败状态，失败原因随即落盘 |
| `capture/` | 当前探针、原始帧、曝光搜索帧、相位和几何 |
| `initial/` | 冷启动名义模型及来源，仅省略检查点时生成 |
| `fit/` | 逐次参数/预测、训练验证指标和冻结检查点 |
| `model/` | GS 种子与模型模式配对记录 |
| `response/`、`response_analysis/` | 曝光阶梯与诊断，尚未自动施加逐像元反响应 LUT |
| `blur/` | 有效展宽及辐射尺度校准 |
| `anchored/` | 局部锚定优化原始帧、相位、时间和接受记录 |
| `audit.json`、`summary/`、`direction/` | 完整性审计、图像/指标摘要、方向核对 |

目录已存在时拒绝覆盖。中断后先查看 `journal.json`，保留失败阶段数据；当前版本不自动跳过或续接设备阶段，避免把光路变化前后的记录拼成同一闭环。重新执行使用新目录，必要时用已完成采集继续离线分析。

各设备阶段均通过上下文管理器释放设备，并在退出前尝试恢复平相位。设备断连、SDK 阻塞或进程强制终止时仍须确认设备实际状态。

## 6. 代码入口

可复用实现位于 `src/ao_shaping/tools/slm/transfer/`；数值模型位于 `algorithm/signal_processing/physical_forward_fit.py`，形状目标/GS/锚定函数位于 `algorithm/signal_processing/multishape_objective.py`，曝光响应位于 `utils/image/exposure_response.py`。这些 `src` 模块不依赖 `scripts`。

原有 `scripts/slm/transfer_multishape_experiment.py` 等九个命令保留兼容入口，既有命令与报告不用修改；新工作优先使用包入口。报告图片生成器继续位于 `scripts/slm/`，只承担离线呈现。
