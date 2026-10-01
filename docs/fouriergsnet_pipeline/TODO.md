# FourierGSNet 硬件验证 TODO

> 作用域: 仅 FourierGSNet (SLM freeform 方形光斑整形) 的**真机光路验证**。
> 离线训练部分 (corpus → cache → dataset → train → W&B) 已完成并验收, 详见 [`report.md`](report.md)。
> 本文件只记录"设备 + 激光就绪后"待执行的真机工作, 与 `docs/TODO.md` (slm_zernike_pib 对账版) 无关。
>
> ✅ **2026-10-01 已执行 §2 (SLM+Daheng smoke) 与一次完整 `slm-gsnet spgd` 优化。**
> 实测结论: 该优化是**随机游走** (`dec=0.487`), 末态比初态**差 35.9%**,
> EE 流失 6 倍, 0 阶峰值 225→17。完整实测量、根因与修复清单见
> **[`hardware_run_20261001.md`](hardware_run_20261001.md)**。
> 下面 §4 的命令**已验证可运行**, 但在修掉报告 §6 列的 6 项之前不要期待方形远场。

## 0. 前置条件

- [ ] **设备在线**: SLM (Santec #1) + Daheng CCD 均可枚举/打开。
- [ ] **激光开启** (1064 nm), 且 2f Fourier 光路无遮挡 (SLM 前焦点 → f=125mm 透镜 → CCD 后焦点)。
- [ ] **无残留 python 进程**: 真机 open 前确认无 `python*` 进程占用 SLM/相机。
  - 检查: `Get-Process python* -ErrorAction SilentlyContinue`
  - 清理: 逐个 `Stop-Process -Id <pid>` (勿用 `killall`/`Stop-Process python* -Force` 误杀其他会话)。
- [ ] **看门狗/硬超时**: SLM memory-mode `open()` + 首次 CCD capture 可能**无限阻塞** (AGENTS.md 反模式)。
  任何真机脚本必须作为**独立进程**运行, 外部可强杀, 并带秒级超时看门狗。

## 1. 安全红线 (来自 AGENTS.md, 违反即停)

| # | 红线 | 说明 |
|---|------|------|
| 1 | SLM **仅 memory mode** `video_mode=0` | 绝不开 DVI (`video_mode=1`) — 观测到 120s/300s 挂起, 且挂起后连带卡死 memory-mode open, 需**物理断电**才能恢复。 |
| 2 | SLM 写入**必须轮换 memory slot** | 连续 `write_phase` + `display_memory` 到同一 slot 是 no-op (LCOS 不刷新)。优选模式: 每次写随机取 2..125 中一个, 排除当前显示 slot (启动时 `get_displayed_memory_number()`)。 |
| 3 | 0 级光斑用 `argmax` 定位 | 2f 光轴 (0 级 = 帧全局最大值) 不落在帧中心。禁止用几何中心。 |
| 4 | 相位生成 **raw 未包裹弧度** | 唯一 wrap 点在 driver `create_phase_from_array()`。生成器不得自行 `mod 2π`。 |
| 5 | 禁止 min-max 归一化相位 | `PatternHelper._zernike_to_uint16` / `ZernikeDM.generate_phase` 均为反模式。 |

## 2. 硬件 smoke test (SLM + Daheng CCD)

**脚本**: `C:\Users\zhangh\AppData\Local\Temp\opencode\hw_smoke_slm_daheng.py`
(只读 + 显式标注步骤; 打开两设备、采帧、报数, **不写任何相位图案**。)

执行 (独立进程 + 看门狗, 硬超时 ~120s):

```powershell
# 工作目录 D:\Projects\TIFO\AO-shaping, PYTHONPATH=src;libs
$env:PYTHONPATH='src;libs'
# 外部看门狗: 超时即杀
$proc = Start-Process -FilePath "python" -ArgumentList "C:\Users\zhangh\AppData\Local\Temp\opencode\hw_smoke_slm_daheng.py" `
         -PassThru -RedirectStandardOutput "C:\Users\zhangh\AppData\Local\Temp\opencode\hw_smoke.out" `
         -RedirectStandardError "C:\Users\zhangh\AppData\Local\Temp\opencode\hw_smoke.err" -NoNewWindow
$proc.WaitForExit(120000)
if ($proc.HasExited -ne $true) { $proc.Kill(); "WATCHDOG: SLM/CCD hung — 物理断电 SLM 后重试" } else { "exit=$($proc.ExitCode)" }
Get-Content "C:\Users\zhangh\AppData\Local\Temp\opencode\hw_smoke.out"
```

**smoke 脚本 5 个步骤, 每步预期**:

| 步骤 | 内容 | 通过判据 |
|------|------|----------|
| 1 | Daheng 枚举 | `num >= 1`, `dev_info_list` 非空; 打印 sn/model/vendor; `Far_Cam_ID` / `Near_Cam_ID` 正确 |
| 2 | 开 Daheng CCD + 采帧 (far cam) | `OPEN OK` < 数秒; 三个曝光 (1.0/0.5/5.0 ms) 各采 1 帧; 打印 shape/max/argmax/饱和率 |
| 3 | 开 Santec SLM #1 (**memory mode**) | `OPEN OK`; `is_open=True`; 打印 resolution/max_gray/wavelength/displayed slot |
| 4 | SLM 打开后再采帧 (**文档化挂起点**) | `CAPTURE OK`; 无无限阻塞 |
| 5 | 关闭两设备 | `closed cleanly` |

- 末尾应打印 **`SMOKE_OK`**, 进程退出码 `0`。
- 任一 open/capture 超时 → 看门狗杀进程 → **先确认无残留 python 进程** → 对 SLM 控制器**物理断电重置** → 重试。

**结果记录**: 把 `hw_smoke.out` 的关键行 (各步 OK + SMOKE_OK) 归档到本节下方 "结果" 区, 记录日期/设备序列号/曝光参数。

### 结果
<!-- smoke 通过后粘贴关键日志行 -->
- _待填_

## 3. DM / WFS 网络可达性 (若后续需 WF 链路)

> FourierGSNet 真机验证主路径是 SLM + CCD (无 WFS), 此项仅在同时跑 WFS 相关链路时需要。

- [ ] 确认 DM / WFS 的**实际 IP 与网段** (上次 `192.168.0.101–126` 全段 `none reachable`, 需以现场实配为准)。
- [ ] 逐个 ping 实际 IP, 记录可达设备。
- [ ] 记录到本节 "结果" 区。

### 结果
<!-- 记录实际 IP + ping 结果 -->
- _待填_

## 4. 真机 `slm-gsnet` 光路验证 (smoke 通过后才执行)

> smoke (第 2 节) 全部通过后再做。仍须独立进程 + 看门狗 + 硬超时 (首次 CCD capture 是挂起高危点)。

### 4.1 SPGD 自由相位搜索 (主路径)

```powershell
# D:\Projects\TIFO\AO-shaping, PYTHONPATH=src;libs
$env:PYTHONPATH='src;libs'
python src/ao_shaping/main.py slm-gsnet spgd `
  --cam_type daheng --cam-id 0 `
  --slm_number 1 --slm_wavelength 1064 `
  -c max `
  --exposure_time_ms 0.8 `
  -e 50
```

关键参数 (已对照 `slm-gsnet spgd --help`):
- `--cam_type daheng --cam-id 0` — 真机用 Daheng 远场相机 (cam-id=0, 对应 `Far_Cam_ID=0`)。
- `--slm_number 1 --slm_wavelength 1064` — Santec SLM #1, 1064 nm (driver 内部走 memory mode + slot 轮换)。
- `-c max` — 0 级光斑用 `argmax` 定位 (红线 3)。
- `--exposure_time_ms 0.8` — 1064 nm 近饱和基线 ~0.02ms 量级, 0.8ms 为保守曝光 (可调)。
- `-e 50` — SPGD 迭代数 (真机先小步验证收敛, 勿一次跑满)。

**验收判据**:
- [ ] SLM memory-mode open 成功且 < 数秒 (无挂起)。
- [ ] 首次 CCD capture 成功 (无无限阻塞)。
- [ ] 0 级光斑经 `argmax` 正确定位 (与 smoke 步骤 4 帧一致)。
- [ ] SPGD 迭代中目标方框内能量 (encircled energy) 与均匀度 (CV) 有可测改善, 非 `mean_b=0.01` 量级失效。
- [ ] 产物 (相位图 / 远场图 / 指标) 落到 `data/.../` 输出目录并归档路径。

### 4.2 (可选) heuristic 黑盒搜索 复核

```powershell
python src/ao_shaping/main.py slm-gsnet heuristic `
  --cam_type daheng --cam-id 0 `
  --slm_number 1 --slm_wavelength 1064 `
  -c max `
  --exposure_time_ms 0.8 `
  --algorithm spgd -e 30
```

- 用于交叉验证 SPGD 结果; 非必须。

### 结果
<!-- 记录 run 目录路径 + 关键指标 (EE / CV / aspect) + 代表性图像路径 -->
- _待填_

## 5. 关闭 / 复跑注意事项

- 任何真机步骤后, 确认无残留 `python*` 进程 (避免下次 open 卡死)。
- SLM memory-mode `open()` 若超过数秒无日志 → 对 SLM 控制器**物理断电重置** (与 DVI 挂起同一处置)。
- 光路/透镜/相机位置有变动后, **重新 `argmax` 定位 0 级** (2f 几何可能漂移)。
