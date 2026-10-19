# R50 控制器控制面板 UI (r50_controller_ui.py) — 用户手册

## 1. 概述

`r50_controller_ui.py` 是 R50Power 微驱动器控制器的 Streamlit 前端页面，作为薄编排层，组装多个兄弟模块，提供四个控制 Tab：单单元控制、单控制器控制、单组控制、全部控制。

启动方式：

```bash
streamlit run src/ao_shaping/gui/r50/r50_controller_ui.py
```

完整技术细节见 [`r50_controller_ui_srs.md`](./r50_controller_ui_srs.md)。

---

## 2. 快速上手

### 2.1 连接设备

1. 在侧边栏选择连接模式（单控制器 / 单组 / 全部控制）。
2. 输入 IP 地址与端口。
3. 点击「连接」建立设备连接。
4. 连接成功后，相应的控制 Tab 可用。

### 2.2 四个控制 Tab

| Tab | 功能 |
|-----|------|
| 单单元控制 | 通道选择、连通性测试、继电器控制、单次电压下发 |
| 单控制器控制 | 单个 R50 控制器的完整控制 |
| 单组控制 | 组连接、批量下发、批量上下电 |
| 全部控制 | MicroDM 矩阵控制、编辑、热力图可视化 |

### 2.3 电压下发

- 所有下发电压自动裁剪到安全范围。
- 支持批量下发与发送循环管理。

---

## 3. 关键设计规则

| 规则 | 说明 |
|------|------|
| 模块化编排 | 主文件只负责组装兄弟模块，可测试逻辑位于兄弟模块 |
| 统一状态初始化 | 所有 session_state 在 `_initialize_state()` 中初始化 |
| 反馈循环排空 | 每次 rerun 调用 `_drain_loop_feedback()` 清空反馈队列 |

---

## 4. 相关文档

- 需求规格说明书: [`r50_controller_ui_srs.md`](./r50_controller_ui_srs.md)
- 驱动参考: [`ao_shaping/drivers/dm/micro/`](../../../../../drivers/dm/micro/)
- GUI 索引: [`docs/gui.md`](../../../gui.md)
