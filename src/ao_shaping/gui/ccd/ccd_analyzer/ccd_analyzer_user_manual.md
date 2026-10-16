# CCD 实时图像分析 UI (ccd_analyzer.py) — 用户手册

## 1. 概述

`ccd_analyzer.py` 是 CCD 相机实时图像分析的 Streamlit 前端页面。支持实时图像捕获（Daheng 或 MiiCam）、自动封闭椭圆计算、X/Y 方向强度截面与 Gaussian 拟合、MJPEG 低延迟实时预览流。

启动方式：

```bash
streamlit run src/ao_shaping/gui/ccd/ccd_analyzer.py
```

完整技术细节见 [`ccd_analyzer_srs.md`](./ccd_analyzer_srs.md)。

---

## 2. 快速上手

### 2.1 连接相机

1. 在侧边栏选择相机类型（Daheng 或 MiiCam）。
2. 设置曝光时间（ms）、采样次数等参数。
3. 点击「连接相机」。
4. 连接成功后，相机句柄存入 session state，可进行实时捕获。

### 2.2 开始实时捕获

1. 点击「开始捕获」。
2. 后台线程循环取帧，通过 MJPEG 流提供低延迟实时预览。
3. 主线程从原始帧队列中 draining 帧，执行图像分析。
4. 分析结果实时显示：质心坐标、封闭椭圆参数、X/Y 强度截面、高斯拟合曲线。

### 2.3 停止捕获

1. 点击「停止捕获」。
2. 后台线程退出循环，释放相机资源。
3. MJPEG 流服务器停止。

---

## 3. 图像分析功能

| 分析项 | 说明 |
|--------|------|
| 质心计算 | 基于矩法计算光斑质心坐标 (cx, cy) |
| 封闭椭圆 | 自动拟合光斑的封闭椭圆参数 |
| X/Y 强度截面 | 通过质心提取 X/Y 方向的一维强度剖面 |
| Gaussian 拟合 | 对 X/Y 截面进行高斯函数拟合，获取拟合参数 |

---

## 4. MJPEG 实时预览

- 启动捕获时自动启动 MJPEG HTTP 服务器（127.0.0.1，自动分配端口）。
- 浏览器访问 `http://127.0.0.1:<port>/video_feed` 可查看低延迟实时图像流。
- 队列长度限制为 2 帧，满时丢弃最旧帧，防止内存积压。

---

## 5. 关键设计规则

| 规则 | 说明 |
|------|------|
| 后台线程隔离 | 捕获线程绝不直接操作 `st.session_state`，通过队列与 UI 通信 |
| 低延迟预览 | MJPEG 流服务器提供低延迟实时图像预览 |
| 队列背压 | MJPEG 队列 maxsize=2，满时丢弃最旧帧 |
| 错误隔离 | 相机连接/捕获失败通过 `st.error` 显示，不导致页面崩溃 |

---

## 6. 相关文档

- 需求规格说明书: [`ccd_analyzer_srs.md`](./ccd_analyzer_srs.md)
- 图像分析算法: [`ao_shaping/utils/image/spots_calc/`](../../../../../utils/image/spots_calc/)
- GUI 索引: [`docs/gui.md`](../../../../../docs/gui.md)
