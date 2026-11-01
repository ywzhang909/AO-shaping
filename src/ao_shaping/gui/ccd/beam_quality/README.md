# beam_quality — AO-Shaping 桌面控制台 (PySide6)

两页桌面应用，共用一条窄导航栏 (`shell.DesktopShellApp`)：

| 页 | 内容 | 代码 |
|----|------|------|
| 0 | **光束质量分析**：实时画面 + 峰值趋势 + 剖面拟合 | `app.BeamQualityApp` |
| 1 | **迁移学习闭环整形**：必设参数 → 校验 → 整场会话 → 分阶段进度 | `gui/slm/transfer_desktop/` |

页 1 在切过去时才创建 (`torch` 由它导入时才拉起)；两页各自管理自己的设备
生命周期与后台作业，切页不打断正在跑的会话或相机流。

## 启动

```bash
.venv\Scripts\python.exe -m ao_shaping.gui.ccd.beam_quality
# 或
.venv\Scripts\python.exe -c "from ao_shaping.gui.ccd.beam_quality import main; main()"
```

单页仍可直接用：`from ao_shaping.gui.ccd.beam_quality import BeamQualityApp`。

## 页 0 — 光束质量

复用 `ao_shaping.drivers.ccd.miicam.MIICamera`（回调拉流），不直接接触 SDK。
无硬件时在左侧「类型」选择 **仿真源 (离线)** 即可演示与自检。

| 区域 | 内容 |
|------|------|
| 左 | 源选择 / cam_id / 打开关闭 / 曝光 (0.011–10000 ms) / 多帧平均 / 快照 PNG / 信息 |
| 中上 | 视频 + 质心十字 + 半高包围椭圆 + fps/亮度信息条 |
| 中下 | X/Y 过中心剖面 + 高斯拟合 (显示 σ、FWHM、R²) |
| 右 | 最大亮度趋势折线图 (max 主曲线 + mean)，30 s / 2 min / 10 min / 全部 |

趋势面板提供 **暂停记录**（画面继续刷新，曲线停止增长）、**清空**、
**导出 CSV**（`t_iso, t_rel_s, max, mean, std, cx, cy, sat_ratio`）。

线程与延迟：

- SDK 回调线程 / 仿真线程只把最新帧放进容量 1 的信箱，**满了丢旧帧**；
- UI 线程 30 Hz 取最新一帧分析并显示 —— 处理不过来时跳过中间帧，显示延迟有界；
- 趋势按 5 Hz 采样记录、5 Hz 重绘；
- 相机打开/关闭/改曝光是实验系统上的秒级阻塞操作，走单线程 `QThreadPool`
  后台作业，UI 不冻结，忙碌期间按钮禁用；
- 自动曝光是同一条通道上的闭环：每 1.2 s 取最新原始帧的一个统计量往目标推一步，
  单次限幅 4×，±3% 视为收敛；自动模式下手动曝光控件禁用。

> 参考的 SDK 示例 `miicam_viewer` 采用后台显示线程 + 双缓冲，存在线程失控与显示
> 延迟问题；本包刻意改为「生产者丢帧 + UI 线程消费」的模型。

## 页 1 — 迁移学习闭环整形

把 Streamlit 版迁移整形页的参数语义搬到桌面：同一个参数对象
(`tools.slm.transfer.workflow.TransferSession`)、同一条执行入口
(`run_session`)、同一份进度读取 (`gui.slm.transfer_progress.collect_progress`)。
默认值与校验顺序镜像 `gui/slm/shaping_controls.TransferMethod`（该模块 import
streamlit，桌面侧不能依赖，故在 `transfer_desktop/params.py` 逐字镜像并有测试钉住）。

P1 只露出**必设参数**（其余字段按 `TRANSFER_RECIPE` 默认值透传）：

| 分组 | 控件 | 默认 |
|------|------|------|
| ① 设备 | 相机类型 (`daheng` / `miicam` / `sim`)、CCD cam_id、SLM 编号 | daheng / 0 / 1 |
| ② 目标 | 目标形状 (正方形 / 长方形 / 三角形 / 五角星，多选)、目标边长 (CCD px)、载波周期 (面板 px) | 正方形 / 80 / 64 |
| ③ 校验 | 改动即校验：error 红字阻断，warning 黄字提醒 | — |
| ④ 运行 | 运行按钮、会话目录、逐阶段进度表 (阶段名→中文标签取 `params.STAGE_LABELS`) | — |

启动前的硬性条件（任一不满足即禁用运行按钮）：

1. `torch` 可导入（物理预测模型的拟合与可微合成都依赖它）；
2. 相机类型是 `daheng` —— 迁移学习只对大恒相机做过实验验证；
3. 至少选择一个目标形状；
4. `preflight.check_parameters` 无 `error`（例如载波周期低于可表示下限、探针数
   不是 6 的倍数、边长超出采样窗）；
5. 会话目录由 `plan_session_dir()` 规划：`data/debug/slm_transfer/<时间戳>`，
   **只规划不创建** —— `run_session` 自己建目录并拒绝与已有会话合并。

运行链路：点「开始运行」→ `build_session` 生成 `TransferSession` → 单线程
`QThreadPool` 里调 `run_session(session, on_stage=…)` → `on_stage` 经 Qt `Signal`
排队回 UI 线程 → 1 s `QTimer` 轮询 `collect_progress(session.out)` 刷新阶段表。
运行中必设参数控件禁用；结束或失败后统一复位，可再次启动。

## 已知限制

- 分辨率超过 1920×1080 时先做 2×2 binning 再分析（坐标已换算回原始像素）。
- 页 1 暂不暴露高级参数（检查点 / 历史与会话复用 / 跨形状校准 / A–B–B–A 验收 /
  运行中自动曝光 / 设备选择 / 预算类字段），它们按默认值透传；轨迹图与设备
  自动检测为 P2。
- `run_session` 没有取消接口：关窗时最多等 5 s 让在飞作业落地，超时照常关窗
  （会话线程自行结束）。
- 仅在仿真源与离屏环境上做过自动化验证；真实相机 / SLM 需现场确认。
