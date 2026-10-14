# AO-Shaping CLI 参考

本文是 AO-Shaping 命令行入口的正式参考。所有 flag 拼写、取值域（choices）、是否必填均逐字来自 `--help` 转储；所有默认值来自 params dataclass 默认值转储。转储里没有的 flag、默认值、路径一律不写。两份转储不一致的地方标 ⚠️ 并以默认值转储为准。

```bash
python src/ao_shaping/main.py [OPTIONS] COMMAND [ARGS]...
```

## 1. 概览

入口是一个 Click group：先写组级选项，再写命令名，最后写命令自己的选项。

```bash
python src/ao_shaping/main.py --help
python src/ao_shaping/main.py wf --help
```

当前共 **20 个命令**：

| 命令 | 一句话用途 | 详情章节 |
|---|---|---|
| `wf` | 波前优化器（WFS 反馈下优化 DM 电压） | 3 |
| `pib` | 轴向光束优化器（DM 电压 PIB 优化） | 4 |
| `pipeline` | 串行优化器，先波前优化再轴向光束优化 | 5 |
| `zernike-matrix` | 获取 Zernike 响应矩阵（SLM + WFS 标定） | 6 |
| `closed-loop` | 基于响应矩阵的闭环波前优化，回放已保存的 .h5 | 7 |
| `rms-zernike` | Zernike 波前优化器（基于 SLM 的 RMS 最小化） | 8 |
| `ga-zernike` | 遗传算法优化 Zernike 系数进行波前校正 | 9 |
| `greedy-zernike` | Zernike 波前优化器，贪婪局部搜索 / 启发式搜索 | 10 |
| `dm-matrix` | 获取 DM 变形镜响应矩阵 | 11 |
| `hadamard-matrix` | 获取 Hadamard 响应矩阵 | 12 |
| `alt-voltage` | 交替电压下发工具（R50Power 控制器） | 13 |
| `full-voltage` | 全量交替电压下发工具（AsyncMicroDM） | 14 |
| `combined` | AdaMOD 综合 PIB 优化器 | 15 |
| `slm-lut` | SLM gray-to-phase LUT 标定 | 16 |
| `slm-diagnose` | SLM 硬件自检，逐级定位「面板不调制光」类故障 | 17 |
| `spgd-square` | SLM 方形光斑整形优化器（Zernike 系数 + 相机反馈） | 18 |
| `slm-gsnet` | freeform 相位方形远场整形，group，含 `train` 离线训练 | 19 |
| `slm-pib` | Santec SLM 的 Zernike 系数整形，group | 20 |
| `slm-gs-refine` | SLM GS 预整形 + 自由相位 SPGD 细化 | 21 |
| `slm-model-in-loop` | SLM model-in-the-loop 方形整形闭环 | 22 |

历史沿革：`gs` / `gs-square` / `diff-shaping` / `diff-beam` 已从 CLI 移除，其功能并入 `slm-gsnet` 与 `algorithm/signal_processing/`。

⚠️ **组级与命令级的 `-d` 重复声明**。组级有 `--dir`（默认 `data`）和 `--debug`，但 `wf` / `pib` / `pipeline` / `rms-zernike` / `ga-zernike` / `greedy-zernike` 又各自在自己的选项表里重复声明了 `-d`；其中 `pib` 和 `combined` 把它命名为 `-d, --root-dir` 而不是 `-d, --dir`。同一个短选项出现在两层，位置决定作用对象：组级写在命令名之前，命令级写在命令名之后。

```bash
# 组级 --dir
python src/ao_shaping/main.py --dir data wf
# 命令级 -d
python src/ao_shaping/main.py wf -d data
```

⚠️ **flag 命名本身不一致**。源码里下划线与连字符混用，例如 `slm-gs-refine` 同时有 `--w-pib`（下划线）与 `--w-unif`（连字符）；`spgd-square` 用 `--w-uniformity`，而 `slm-gsnet` 的子命令用 `--w-uniformity`；`pib` 用 `--shrink-iter`，`rms-zernike` 用 `--lr-schedule`。**flag 必须从 `--help` 逐字复制，不要凭记忆或推测**，写错一个字符 Click 会直接报错。

⚠️ **调试模式有三条等价路径**：`main.py --debug <cmd>` / `<cmd> --debug` / 环境变量 `DEBUG=1`；真值为 `1` / `true` / `yes`，不区分大小写。并非每个命令都自带 `--debug` 选项，`wf` / `pipeline` / `ga-zernike` / `spgd-square` / `slm-lut` / `slm-diagnose` / `slm-gs-refine` 的选项表里没有 `--debug`，这几个命令要开调试只能走组级 `--debug` 或 `DEBUG=1`。

ℹ️ 每个命令的选项表都不再重复列出通用的 `--help`（每个命令都有，用来打印帮助并退出）。

## 2. 全局选项

组级选项，写在命令名之前：

| 选项 | 类型/取值 | 默认值 | 说明 |
|---|---|---|---|
| `--dir` | TEXT | `data` | 数据保存根目录。与各命令自己的 `-d` 同名不同层 |
| `--debug` | 开关 | `False` | 启用调试模式。真值 `1` / `true` / `yes`，不区分大小写 |

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DEBUG` | 未设置 | 等价于 `--debug`，与两条 `--debug` 路径任一开启即可 |

默认值来源：`runner_common.py :: RunParams`（`defaults_dump.txt:442-444`）。

## 3. wf

### 用途

波前优化器。以 WFS 测量的波前 RMS 为反馈迭代优化 DM 电压。`DEBUG` 环境变量控制调试模式。

### 硬件需求

DM + WFS。`--wfs-type sim` 提供仿真 Shack-Hartmann，无需 WFS 硬件；`--dm-type` 含 `sim` / `sim_micro` 仿真变形镜，两者都选仿真即可完全脱离硬件跑。

### 选项

| 选项 | 类型/取值 | 默认值 | 说明 |
|---|---|---|---|
| `-d, --dir` | TEXT | `data` | 数据保存根目录 |
| `-e, --epochs` | INTEGER | `20000` | 优化迭代次数 |
| `-t, --early-stop-threshold` | FLOAT | `0.0` | 早停阈值 |
| `--disturbance-cn2` | FLOAT | `0.0` | 仿真 WFS 湍流强度 cn2，0=不加像差 |
| `--lr` | FLOAT | 未指定（`WfRunnerParams.lr=None`） | 覆盖自动学习率；不给则按硬件标定的自动调度 |
| `--delta` | FLOAT | 未指定（`WfRunnerParams.delta=None`） | 覆盖 SPGD 扰动幅度（V）；不给则走自动调度 |
| `--show` | 开关 | `False` | 显示远场光斑 CCD 图像和优化历史 |
| `--dm-type` | `[asyn_micro\|hadamard\|micro\|nlight\|sim\|sim_micro\|zernike]` | `None`（auto-detect） | 变形镜类型。未指定且仅一个 DM 在线则自动选取，否则报错 |
| `-r, --wfs-res` | TEXT | `1024` | WFS 分辨率 |
| `-p, --pupil-diameter` | FLOAT | `2.7` | 瞳孔直径 |
| `-c, --pupil-center` | TEXT | `(0,0)` | 瞳孔中心坐标 |
| `--exposure-time-ms` | FLOAT | `0.0` | WFS 曝光时间（毫秒），0=自动曝光 |
| `--remove-tilt` | 开关 | `False` | 移除波前测量中的倾斜项 |
| `--wfs-type` | `[thorlab\|sim]` | `thorlab` | 波前传感器类型，sim=仿真 Shack-Hartmann，无需硬件 |

共 14 项。默认值来源：`runner_common.py :: WfRunnerParams`（`defaults_dump.txt:5-12`）、`WfsParams`（`defaults_dump.txt:355-360`）。

### 示例

```bash
# 仿真 DM + 仿真 WFS，跑 200 轮早停演示
python src/ao_shaping/main.py wf -e 200 --dm-type sim --wfs-type sim --show

# 真实硬件，指定 WFS 分辨率与瞳孔参数
python src/ao_shaping/main.py wf -r 1024 -p 2.7 -c "0,0" --exposure-time-ms 0.0
```

## 4. pib

### 用途

轴向光束优化器。在 DM 电压上做桶内功率（PIB）优化，支持自适应桶半径收缩。调试模式三条路径任一开启即可输出 pkl/json 与汇总图片。

### 硬件需求

DM + CCD。`--cam-type sim` 是 2f-Fourier 数值仿真（需配合 `--dm-type sim`），`--dm-type` 亦含 `sim` / `sim_micro`，两者配合可完全仿真。

### 选项

| 选项 | 类型/取值 | 默认值 | 说明 |
|---|---|---|---|
| `-d, --root-dir` | TEXT | `data` | 数据保存根目录。⚠️ 长选项是 `--root-dir`，不是 `--dir` |
| `-f, --load-file` | TEXT | `rms` ⚠️ | 加载优化结果文件；若为 `rms` 则用 RMS 优化结果初始化。⚠️ `--help` 写 `(default: None)`，默认值转储为 `'rms'`，以转储为准 |
| `--cam-id` | TEXT | 由环境变量 `Far_Cam_ID` 决定，缺省 `0` | 远场光斑 CCD 设备 ID |
| `--cam-type` | `[miicam\|daheng\|sim]` | `miicam` | CCD 相机后端，sim = 2f-Fourier 数值仿真，需配合 `--dm-type sim` |
| `-c, --center` | TEXT | `mass` | 远场光斑 CCD 中心位置，例如 `665,403` |
| `-t, --exposure-time-ms` | FLOAT | `0.0` | 远场光斑 CCD 曝光时间（毫秒）。0=不固定，由设备保持或自动曝光；Daheng 上 0 会被驱动钳到设备最小值（约 0.02ms） |
| `-e, --epochs` | INTEGER | `4000` | 优化迭代次数 |
| `-r, --r-bucket` | INTEGER | `0` | 半径桶大小。0=环围半径，按功率半径自动调整 |
| `--delta` | FLOAT | `2.0` | 优化步长 |
| `--lr` | FLOAT | `0.0` | 优化学习率。0.0 表示基于环围半径动态学习率衰减 |
| `--weight-decay` | FLOAT | `0.0` | 权重衰减 |
| `--optimizer-type` | `[adam\|adamw\|adamod\|sgd\|muno\|munow]` | `adamod` | 梯度阶段使用的优化器类型 |
| `--shrink-iter` | INTEGER | `200` | 优化多少迭代后收缩半径桶和步长。0=不收缩 |
| `--shrink-ratio` | FLOAT | `0.8` | 收缩半径桶和步长比例 |
| `--enable-adaptive-search` | 开关 | `False` | 启用局部最优后的自适应邻域搜索 |
| `--search-interval` | INTEGER | `120` | 邻域搜索触发间隔 |
| `--search-warmup` | INTEGER | `200` | 邻域搜索启动前的最小迭代数 |
| `--search-patience` | INTEGER | `100` | 最佳 PIB 无提升时触发搜索的等待轮数 |
| `--search-samples` | INTEGER | `8` | 每次邻域搜索评估的候选解数量 |
| `--search-radius` | FLOAT | 未指定（`PibRunnerParams.search_radius=None`） | 邻域搜索初始半径，默认跟随 delta 自适应 |
| `--tabu-memory-size` | INTEGER | `128` | 禁忌记忆表容量 |
| `-s, --cam-size` | INTEGER | `200` | 相机开窗大小（`--help` 文案写作 `200*200`） |
| `-b, --target-max-brightness` | INTEGER | `90` | 目标最大亮度值，0=不自动调整曝光时间 |
| `-o, --objective` | `[pib\|radiu\|avg_radiu]` | `pib` | 优化目标函数：pib=最大化 PIB，radiu=最小化半径，avg_radiu=最大化平均半径 |
| `--show` | 开关 | `False` | 显示远场光斑 CCD 图像和优化历史 |
| `--dm-type` | `[asyn_micro\|hadamard\|micro\|nlight\|sim\|sim_micro\|zernike]` | `None`（auto-detect） | 变形镜类型。未指定且仅一个 DM 在线则自动选取，否则报错 |
| `--debug` | 开关 | `None`（关闭） | 启用调试模式，保存 pkl/json 与汇总图（初始/最优光斑、目标曲线、最优电压） |

共 27 项。默认值来源：`runner_common.py :: PibRunnerParams`（`defaults_dump.txt:15-41`），`--cam-id` 为 lambda 动态默认（`defaults_dump.txt:17`）。

### 示例

```bash
# 默认目标 pib，自动环围半径，4000 轮
python src/ao_shaping/main.py pib -e 4000 -r 0 -o pib

# 仿真链路 + 调试输出
python src/ao_shaping/main.py --debug pib --cam-type sim --dm-type sim -c mass
```

## 5. pipeline

### 用途

串行优化器，先做波前优化，再做轴向光束优化。`DEBUG` 环境变量控制调试模式。

### 硬件需求

DM + WFS + CCD。`--wfs-type sim` 可替换 WFS；该命令选项表里没有 `--cam-type`，CCD 侧没有仿真后端选项。

### 选项

| 选项 | 类型/取值 | 默认值 | 说明 |
|---|---|---|---|
| `-d, --dir` | TEXT | `data` | 数据保存根目录 |
| `-f, --load-file` | TEXT | `None` | 加载优化结果文件 |
| `-e, --epochs` | INTEGER | `8000` | 优化迭代次数 |
| `-E, --wf-epochs` | INTEGER | `8000` | WF 优化迭代次数 |
| `--cam-id` | TEXT | 由环境变量 `Far_Cam_ID` 决定，缺省 `0` | 远场光斑 CCD 设备 ID，无短选项 |
| `-t, --exposure-time-ms` | INTEGER | `0` | 远场光斑 CCD 曝光时间（毫秒），0=自动选取曝光 |
| `-s, --cam-size` | INTEGER | `160` | 相机开窗大小 |
| `--rms-threshold` | FLOAT | `0.12` | RMS 阈值，无短选项 |
| `-u, --dm-unit-mask` | `[all\|inner\|outer]` | `all` | DM 单元掩码 |
| `--dm-type` | `[asyn_micro\|hadamard\|micro\|nlight\|sim\|sim_micro\|zernike]` | `None`（auto-detect） | 变形镜类型。未指定且仅一个 DM 在线则自动选取，否则报错 |
| `-r, --wfs-res` | TEXT | `1024` | WFS 分辨率 |
| `-p, --pupil-diameter` | FLOAT | `2.7` | 瞳孔直径 |
| `-c, --pupil-center` | TEXT | `(0,0)` | 瞳孔中心坐标 |
| `--exposure-time-ms` | FLOAT | `0.0` | WFS 曝光时间（毫秒），0=自动曝光 |
| `--remove-tilt` | 开关 | `False` | 移除波前测量中的倾斜项 |
| `--wfs-type` | `[thorlab\|sim]` | `thorlab` | 波前传感器类型，sim=仿真 Shack-Hartmann，无需硬件 |

共 16 项。默认值来源：`runner_common.py :: PipelineRunnerParams`（`defaults_dump.txt:44-53`）、`WfsParams`（`defaults_dump.txt:355-360`）。

ℹ️ 该命令**没有** `-R`，分辨率是 `-r, --wfs-res`；`-c` 是 `--pupil-center`，不是 `--cam-id`。

### 示例

```bash
# WF 阶段 8000 轮 + PIB 阶段 8000 轮，RMS 达标后切 PIB
python src/ao_shaping/main.py pipeline -E 8000 -e 8000 --rms-threshold 0.12 -s 160
```

## 6. zernike-matrix

### 用途

获取 Zernike 响应矩阵。支持 N 次正负交替循环测量 + M 次 WFS 读取取平均 + 方差跟踪 + 逆矩阵计算；`--n-magnitudes N` 可生成 N 个不同扰动幅度（2~5 rad）分别标定并保存；`--debug` 保存每次测量的原始数据（SLM 相位图、WFS deviation、averaging 前的 Zernike 系数）。help 内还附有一份重写后的实机待测试清单。

### 硬件需求

SLM（Santec）+ WFS。`--wfs-type sim` 可用仿真 Shack-Hartmann 替代真实 WFS；SLM 侧该命令没有仿真后端选项。

### 选项

| 选项 | 类型/取值 | 默认值 | 说明 |
|---|---|---|---|
| `--n-max` | INTEGER | `10` | Zernike 最大阶数 |
| `--magnitude` | FLOAT | `3.0` | 扰动幅度，单位是**弧度**。0=自动优化；重写后系数即弧度，建议 2~5 rad（约 0.3~0.8λ） |
| `--zernike-radius` | FLOAT | `300.0` | Zernike 归一化半径 px，建议约 1.5 倍光束半径；闭环矫正必须用同一值 |
| `--n-averages` | INTEGER | `3` | 每次 WFS 读取次数（M） |
| `--n-cycles` | INTEGER | `1` | 正负交替循环次数（N） |
| `--wait` | FLOAT | `0.2` | 等待时间（秒） |
| `--output` | TEXT | `data/zernike_response_matrix` | 输出文件路径 |
| `--no-inverses` | 开关 | 不启用（转储 `compute_inverses=True`） | 不计算逆矩阵 |
| `--excluded-tip-tilt` | 开关 | `False` | 排除 tip/tilt 模式（Z2, Z3） |
| `--cancel-tile` | 开关 | `False` | 测量时去除 WFS 的 tip/tilt（对应 Thorlabs 的 cancel_tile 功能） |
| `--display / --no-display` | 开关 | `False` | 显示实时 pygame 显示 |
| `--debug` | 开关 | `None`（关闭） | 启用调试模式，保存原始测量数据 |
| `--auto-optimize / --no-auto-optimize` | 开关 | `True` | 自动优化扰动幅度（magnitude=0 时生效） |
| `--optimize-n-avg` | INTEGER | `10` | 幅度优化时的 WFS 读取次数 |
| `--n-magnitudes` | INTEGER | `0` | 自动生成 N 个不同扰动幅度并分别保存，0=禁用 |
| `--dither-amp` | FLOAT | `0.0` | 亚波长抖动幅度 [λ]，0=禁用（建议 0.02-0.05） |
| `--correction-csv` | TEXT | `None` | 误差矫正 CSV 文件路径 |
| `--wavelength` | INTEGER | `532` | SLM 波长（nm） |
| `--shift-x` | INTEGER | `0` | SLM X 方向平移（像素） |
| `--shift-y` | INTEGER | `0` | SLM Y 方向平移（像素） |
| `--slm-number` | INTEGER | `1` | SLM 设备编号 |
| `--wait-time` | FLOAT | `0.3` | SLM 液晶翻转等待时间（秒） |
| `--mla-index` | `[512\|540\|600\|768\|1280]` | `512` | MLA 分辨率 |
| `--exp-time` | FLOAT | `0.0` | WFS 曝光时间（ms），0=自动 |
| `--auto-exposure / --no-auto-exposure` | 开关 | `True` | 启用 WFS 自动曝光，默认开启 |
| `--high-speed` | 开关 | `False` | 启用高速模式 |
| `--use-custom-ref` | 开关 | `False` | 使用自定义参考文件 |
| `-r, --wfs-res` | TEXT | `1024` | WFS 分辨率 |
| `-p, --pupil-diameter` | FLOAT | `2.7` | 瞳孔直径 |
| `-c, --pupil-center` | TEXT | `(0,0)` | 瞳孔中心坐标 |
| `--exposure-time-ms` | FLOAT | `0.0` | WFS 曝光时间（毫秒），0=自动曝光 |
| `--remove-tilt` | 开关 | `False` | 移除波前测量中的倾斜项 |
| `--wfs-type` | `[thorlab\|sim]` | `thorlab` | 波前传感器类型，sim=仿真 Shack-Hartmann，无需硬件 |

共 33 项。默认值来源：`zernike_matrix_runner.py :: ZernikeMatrixParams`（`defaults_dump.txt:477-494`）、`runner_common.py :: WfsParams`（`defaults_dump.txt:355-360`）、`ZernikeSlmParams`（`defaults_dump.txt:363-367`）、`ThorlabWfsDriverParams`（`defaults_dump.txt:370-374`）。

### 示例

```bash
# 幅度 3.0 rad、3 次平均、1 个循环，输出到默认路径
python src/ao_shaping/main.py zernike-matrix --n-max 10 --magnitude 3.0 --n-averages 3 --n-cycles 1

# 多幅度标定 + 调试原始数据
python src/ao_shaping/main.py --debug zernike-matrix --n-magnitudes 4 --dither-amp 0.03
```
