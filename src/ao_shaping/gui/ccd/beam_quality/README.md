# beam_quality — MiiCam 光束质量分析 (PySide6)

实时显示 MiiCam 图像并做逐帧光束分析，重点展示**最大亮度随时间的变化曲线**
（同时记录均值、标准差、质心、饱和占比）。复用
`ao_shaping.drivers.ccd.miicam.MIICamera`（回调拉流），不直接接触 SDK。

## 启动

```bash
.venv\Scripts\python.exe -m ao_shaping.gui.ccd.beam_quality
# 或
.venv\Scripts\python.exe -c "from ao_shaping.gui.ccd.beam_quality import main; main()"
```

无硬件时在左侧「类型」选择 **仿真源 (离线)** 即可演示与自检。

## 界面

| 区域 | 内容 |
|------|------|
| 左 | 源选择 / cam_id / 打开关闭 / 曝光 (0.011–10000 ms) / 快照 PNG / 信息 |
| 中上 | 视频 + 质心十字 + 半高包围椭圆 + fps/亮度信息条 |
| 中下 | X/Y 过中心剖面 + 高斯拟合 (显示 σ、FWHM、R²) |
| 右 | 最大亮度趋势折线图 (max 主曲线 + mean)，30 s / 2 min / 10 min / 全部 |

趋势面板提供 **暂停记录**（画面继续刷新，曲线停止增长）、**清空**、
**导出 CSV**（`t_iso, t_rel_s, max, mean, std, cx, cy, sat_ratio`）。

## 线程与延迟

- SDK 回调线程 / 仿真线程只把最新帧放进容量 1 的信箱，**满了丢旧帧**；
- UI 线程 30 Hz 取最新一帧分析并显示 —— 处理不过来时跳过中间帧，显示延迟有界；
- 趋势按 5 Hz 采样记录、5 Hz 重绘；
- 相机打开/关闭/改曝光在 UI 线程同步执行（约 0.6–1 s 卡顿，MiiCam 曝光变更本身需要
  Stop→设置→重启拉流）。

> 参考的 SDK 示例 `miicam_viewer` 采用后台显示线程 + 双缓冲，存在线程失控与显示
> 延迟问题；本包刻意改为「生产者丢帧 + UI 线程消费」的模型。

## 已知限制

- 分辨率超过 1920×1080 时先做 2×2 binning 再分析（坐标已换算回原始像素）。
- 曝光修改的同步卡顿尚未移入工作线程。
- 仅在仿真源上做过自动化验证；真实相机需现场确认。
