# Micro-DM 差分分析 UI (md_img_analysis_ui.py) — 用户手册

## 1. 概述

`md_img_analysis_ui.py` 是 Micro-DM 差分分析 UI 的 Streamlit 前端页面，包装 `scripts/dm/md_img_pipeline.py` 的完整 diff→overlay→GIF→combined 流水线，支持运行模式与浏览模式。

启动方式：

```bash
streamlit run src/ao_shaping/gui/dm/md_img_analysis_ui.py
```

完整技术细节见 [`md_img_analysis_ui_srs.md`](./md_img_analysis_ui_srs.md)。

---

## 2. 快速上手

### 2.1 运行模式

1. 选择数据集（`data/md_test/` 下的数据集）。
2. 调整流水线参数：
   - 阈值（默认 15.0）：差分检测阈值
   - 色图（默认 jet）：热图色图
   - 缩放（默认 0.25）：图像缩放比例
   - FPS（默认 8）：GIF 帧率
3. 点击「运行流水线」。
4. 流水线在后台线程运行，UI 实时显示进度（每 IP 的 diff/overlay/gif 状态 + 全局 overlay + combined GIF）。
5. 如需停止，点击「停止流水线」。

### 2.2 浏览模式

1. 选择「浏览模式」。
2. 选择已有处理结果目录。
3. 查看逐 IP 的 diff 图像、质心表、overlay、动画 GIF。
4. 查看全局 overlay 与 combined GIF。

---

## 3. 流水线参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 阈值 | 15.0 | 差分检测阈值 |
| 色图 | jet | 热图色图 |
| 缩放 | 0.25 | 图像缩放比例 |
| FPS | 8 | GIF 帧率 |

---

## 4. 关键设计规则

| 规则 | 说明 |
|------|------|
| 后台线程隔离 | 工作线程绝不触碰 `st.session_state`，通过加锁进度字典通信 |
| 进度实时更新 | 主线程轮询进度字典，实时显示处理状态 |
| 可中断执行 | 支持随时停止正在运行的流水线 |

---

## 5. 相关文档

- 需求规格说明书: [`md_img_analysis_ui_srs.md`](./md_img_analysis_ui_srs.md)
- 流水线脚本: [`scripts/dm/md_img_pipeline.py`](../../../../../../scripts/dm/md_img_pipeline.py)
- 数据目录: [`data/md_test/`](../../../../../../data/md_test/)
- GUI 索引: [`docs/gui.md`](../../../gui.md)
