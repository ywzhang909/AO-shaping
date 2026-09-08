# diff-shaping 直连本机硬件使用指南 (Santec SLM200 + MiiCam)

> 可微分光束整形 (`diff-shaping`) 在**本机直连 SLM200 + MiiCam** 上的闭环运行说明：
> GPU 计算配置、MiiCam 曝光基线、推荐参数与预期产物。
> 完整验证报告见 [docs/slm_differential_shaping/](../slm_differential_shaping/README.md)。

## 1. 运行环境

### 依赖与路径

```powershell
# PowerShell (Windows)
$env:PYTHONPATH = "src;libs"   # libs 含 gxipy 等 SDK 二进制路径
$env:DEBUG = "1"               # 可选: 输出 debug 级日志 (逐迭代 phase 细节)
```

SLM/MiiCam 驱动不需要额外安装 —— 包内自带 SDK (MiiCam 为纯 Python 封装)。
PyTorch 必须带 CUDA 版: `uv sync --extra ml`。

### GPU (本机) 配置

runner 通过 `--device` 控制计算设备:

| 取值 | 行为 |
|------|------|
| `auto` (默认) | **有 CUDA 则用 GPU，否则 CPU** (自动检测) |
| `cuda` | 强制 GPU, 无 CUDA 时报错 |
| `cpu` | 强制 CPU (调试/复现用) |

```bash
# 确认本机 CUDA 可用 (torch 2.11+cu128)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only')"
```

实际生效设备在启动日志中确认:

```
Starting differentiable beam shaping: 500 iterations, propagation=fft, optimizer=adam, device=cuda
```

> 模块内部逻辑: `device=None` 时 `"cuda" if torch.cuda.is_available() else "cpu"`；
> ASM 传播子以 (shape, dx, z, λ, device) 为 key 缓存，切换设备自动重建，无需干预。

## 2. MiiCam 曝光时间

**默认值随相机类型自动解析，不再固定 50ms：**

| 相机类型 | 默认曝光 |
|----------|----------|
| `miicam` | **0.02 ms** (本光路实测近饱和基线) |
| `daheng` | 50 ms |

- 0.02ms ≈ 20µs，对应 1064nm 2f 傅里叶光路 (SLM 前焦面 → f≈125mm 透镜 → MiiCam 后焦面)
  的已知近饱和亮度基线 (见 `docs/slm-200` 与 `slm-diagnose` linearity 检查)。
- MiiCam 驱动下限 0.011ms，0.02ms 完全支持 (`put_ExpoTime(int(ms*1000))`，float 全程保留)。
- 如需手动覆盖: `--exposure-ms 0.05` 等。
- **提示**: 若单帧过饱和 (0 级桶局部>250@8bit) 或过暗 (峰<10)，先小范围调整曝光重测，
  不要动其它光学件。

## 3. 直接运行 (SLM #1 + MiiCam #0)

### 最小闭环 (方形成形, 全默认)

```powershell
python src/ao_shaping/main.py diff-shaping --camera-type miicam -o data/diff_shaping_hw
```

等价于:

```powershell
python src/ao_shaping/main.py diff-shaping --camera-type miicam `
    --exposure-ms 0.02 --device auto `
    --target-shape square --target-px 1200 `
    --propagation fft --optimizer adam -i 500 --lr 3e-2 `
    --w-uniformity 0.4 --w-efficiency 0.6 --w-zero-order 0 --w-smoothness 0 `
    -o data/diff_shaping_hw
```

### 推荐参数 (本机实测基准)

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--camera-type` | `miicam` | 本机 CCD 为 MiiCam |
| `--exposure-ms` | `0.02` (默认) | 1064nm 近饱和基线, 防止 0 级饱和 |
| `--device` | `auto` (默认) | 自动用本机 GPU (CUDA) |
| `--target-px` | `600~1200` | 目标方形在相机上的像素宽, **推荐显式指定** (小于传感器高度 1520px) |
| `--pixel-scale` | `0.414` (可选) | SLM 像素→相机像素实测 k≈0.414; 光路变动后须重新标定 |
| `-i, --dl-iterations` | `500` (默认, 600 更稳) | 每次外迭代的梯度内迭代次数 |
| `--outer-iterations` | `3~10` | 外闭环迭代次数 |
| `--propagation` | `fft` (默认) / `asm` | fft=单 FFT 夫琅禾费(高速); asm=角谱(精确, CV≈0.001) |
| `--optimizer` | `adam` (默认) / `lbfgs` | lbfgs+`--lr 1.0` 60 步即近平顶 |
| `--w-zero-order` / `--w-smoothness` | `0` / `0` (默认) | 非零会把能量推出居中目标 (见验证报告) |

### 其它目标形状

```powershell
# spot 聚焦 (远场变点)
python src/ao_shaping/main.py diff-shaping --camera-type miicam --target-shape spot `
    --dl-iterations 600 -o data/diff_shaping_spot

# 圆形 + ASM 精确传播
python src/ao_shaping/main.py diff-shaping --camera-type miicam --target-shape circle `
    --propagation asm --outer-iterations 5 -o data/diff_shaping_circle
```

### 实时可视化

```powershell
python src/ao_shaping/main.py diff-shaping --camera-type miicam --display -o data/diff_shaping_hw
```

弹出 pygame 四面板: 相位图案 / 目标掩模 / 远场光斑 / 损失收敛曲线。
闭环运行期间 **每个内迭代输出一条日志** (`内迭代  N/500 loss=...`)，可实时监控收敛。

## 4. 输出产物 (output 目录)

```
data/diff_shaping_hw/
├── best_phase.npy            # 最优相位图 (float64, 弧度)
├── best_phase_rad.npy        # 同为弧度版 (若生成)
├── best_image.npy            # 最优评分处相机实测远场光斑
├── convergence_history.json  # 每外迭代: score/CV/EE/aspect/squareness/side...
├── loss_history.json         # 每内迭代 loss 序列 (可绘图)
└── metadata.json             # 运行参数 + best_score + converged 判断
```

`metadata.json` 关键字段: `best_score` (质量评分 0~1)、`best_iteration`、
`side` (目标边长 SLM px)、`spot_diameter_px`、`converged` (score ≥ 0.95)。

## 5. 故障排查

| 现象 | 处理 |
|------|------|
| SLM 显示不更新 (连续两轮图案相同) | 内存槽轮换约束: runner 已在 **2~125 内存槽范围内随机选取**每次写入槽位并排除当前显示槽 (含跨进程续接, 见 `_pick_next_slot`), 严禁相邻两次相位写同一槽 |
| 相机全黑/全白 | 调 `--exposure-ms` (0.01~0.5 试); 检查激光与光路对准 |
| 0 级桶饱和 | 降低曝光; 0 级定位始终用帧 `argmax`, 不要用几何中心 (2f 光路轴心不居中) |
| 目标尺寸全帧点亮 | `--target-px` 过大, 取 <1520 (MiiCam 高) |
| 收敛缓慢/EE 低 | 确认 `--w-zero-order 0`、`--lr 3e-2`、内迭代 ≥500; 或切 `--optimizer lbfgs --lr 1.0` |
| 面板疑似不调制光 | 跑 `python src/ao_shaping/main.py slm-diagnose` 三步自检 (freeze/modulate/linearity) |
| 想看在 CPU 上的复现差异 | `--device cpu` 对比 (数值一致, ASM 与 numpy 参考 ~1e-11) |

## 6. 关联

- 完整标定/对比报告: `docs/slm_differential_shaping/README.md` (含 GIF、逐迭代图表、gs 基线对比)
- 模块 API: `src/ao_shaping/algorithm/differentiable_shaping.py` (`train_beam_shaping`)
- Runner 源码: `src/ao_shaping/runners/diff_shaping_runner.py`