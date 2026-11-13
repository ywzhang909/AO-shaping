# beam_quality — AO-Shaping 桌面控制台 (PySide6)

两页桌面应用，共用一条窄导航栏 (`shell.DesktopShellApp`)：

| 页 | 内容 | 代码 |
|----|------|------|
| 0 | **光束质量分析**：实时画面 + 峰值趋势 + 剖面拟合 | `app.BeamQualityApp` |
| 1 | **迁移学习闭环整形**：必设参数 → 校验 → 整场会话 → 分阶段进度 | `gui/slm/transfer_desktop/` |

窗口顶部有一条菜单栏，目前只有「帮助」一组，内含「关于」（自 2026-11-09 `deb43b3d` 起，`gui/utils/about.py`）：弹出模态对话框显示版本号（取短 git commit id，打包后的非 git 环境显示「未知」）与指向 GitHub 上用户手册的「帮助文档」链接（用系统浏览器打开，文档不进 exe）。外壳刻意不设自己的状态栏，避免与页 0 的状态栏叠两条。

页 1 在切过去时才创建 (`torch` 由它导入时才拉起)；两页各自管理自己的设备
生命周期与后台作业，切页不打断正在跑的会话或相机流。

页 1 的操作手册见 [docs/GUI/slm/transfer_desktop/transfer_desktop_user_manual.md](../../../../../docs/GUI/slm/transfer_desktop/transfer_desktop_user_manual.md)。

## 启动

```bash
.venv\Scripts\python.exe -m ao_shaping.gui.ccd.beam_quality
# 或
.venv\Scripts\python.exe -c "from ao_shaping.gui.ccd.beam_quality import main; main()"
```

**环境前提**：torch 必须按显卡装好对应 extra（本机 RTX 5080 用
`uv sync --extra torch-cu132`）。选 extra 的第二条规则是**让它自带的 CUDA minor 与
`PATH` 上第一套 CUDA toolkit 一致**：本机 `PATH` 首位是 v13.2（cuBLAS `6.14.11.1340`），
而 torch 2.14.0+cu132 自带的正是 `6.14.11.1340`，两边同源；用 `torch-cu130`（自带
`6.14.11.1300`）时则依赖 `pin_cublas()` 把 cuBLAS 裸名钉到 toolkit 那一份。
**现象**：装错时进程会在 fit / 首次大矩阵乘法处**无声退出**（无 traceback、无报错弹窗），
排查与四种绑定组合的实测见 [`docs/environment.md`](../../../../../docs/environment.md)
§cuBLAS DLL 与 PATH。

单页仍可直接用：`from ao_shaping.gui.ccd.beam_quality import BeamQualityApp`。

## 页 0 — 光束质量

复用 `ao_shaping.drivers.ccd.miicam.MIICamera`（回调拉流），不直接接触 SDK。
无硬件时在左侧「类型」选择 **仿真源 (离线)** 即可演示与自检。

| 区域 | 内容 |
|------|------|
| 左 | 源选择 / cam_id / 打开关闭 / 曝光 (0.011–10000 ms) / **HDR 获取** + 曝光阶梯 / 多帧平均 / 快照 PNG / 信息 |
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

### HDR 获取（只影响预览）

「HDR 获取」默认关闭；**关闭时本页与改动前逐位一致**。打开后预览改走
`BaseCamera.capture_hdr`：按升序阶梯逐档曝光采集，最后一档作参考，
`hdr_fuse` 融合，再做 Reinhard 映射上屏。节拍线程一轮接一轮独占相机
（`capture_hdr` 与取流同时抢同一个 SDK 句柄），所以

- 预览节拍从相机帧率掉到「一轮 bracket 的耗时」，画面会明显变慢——
  这是代价，不是卡死；信息条与状态栏显示 HDR 帧计数；
- 阶梯（默认 `1, 4, 16` ms，取自 transfer 侧的 HDR 探测）与「重拍一轮」
  只在勾选后可编辑；文本框填了非法值时**不生效并退回现用值**，状态栏说明原因
  （下一轮 bracket 起用新阶梯）；若换成真实曝光量程后连现用阶梯也不再合法
  （例如从短曝光源切到上限更小的源），则退回默认阶梯而不是留着采不到图的空 bracket；
- **HDR 帧只上屏，不进数据通路**：不写 `_last_frame` / `_last_metrics`，
  不进多帧平均窗口、趋势缓冲与「已处理」计数。融合显示图已不是任何单一
  曝光下的原始计数，拿它存快照或进 CSV 就是假数据，因此 HDR 开着时快照直接
  拒绝并说明怎么解锁；剖面与中心读数仍照画，但标注「显示派生」；
- HDR 与自动曝光互斥（两者都要改曝光，只能开一个），自动曝光优先，
  HDR 退回单曝光并把原因显示在状态栏，开关如实回弹；
- 采集失败一律静默退回单曝光取流，同时把原因写进状态栏，绝不留黑屏。

「预览模式」一行常显，写的是当前**真正在跑**什么，而不是勾选框的状态。

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
