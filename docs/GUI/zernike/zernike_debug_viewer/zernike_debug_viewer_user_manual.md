# Zernike 响应矩阵调试查看器 UI (zernike_debug_viewer.py) — 用户手册

## 1. 概述

`zernike_debug_viewer.py` 是 Zernike 响应矩阵调试数据查看器的 Streamlit 前端页面。支持加载标配调试数据、按 Zernike 模式浏览、查看 SLM 相位图像与 WFS 偏差热图、对比不同样本的响应差异。

启动方式：

```bash
streamlit run src/ao_shaping/gui/zernike/zernike_debug_viewer.py
```

完整技术细节见 [`zernike_debug_viewer_srs.md`](./zernike_debug_viewer_srs.md)。

---

## 2. 快速上手

### 2.1 加载调试数据

1. 指定包含标配调试数据的目录（由 `zernike-matrix` 命令的 `--debug` 标志保存）。
2. 页面加载该目录下的所有调试数据文件。

### 2.2 按模式浏览

1. 选择 Zernike 模式索引 (mode_index)。
2. 查看该模式下的所有样本（周期、正/负扰动）。
3. 查看 SLM 相位图像（已平移、归一化到灰度）。
4. 查看 WFS 偏差 (x, y) 热图。
5. 查看原始 WFS Zernike 系数（平均前）。

### 2.3 响应矩阵可视化

1. 加载已标配的响应矩阵 (.h5 文件)。
2. 查看完整的响应矩阵热图。

---

## 3. 支持的可视化类型

| 可视化类型 | 说明 |
|------------|------|
| SLM 相位图像 | SLM 下发的 Zernike 相位，已平移、归一化到灰度 |
| WFS 偏差热图 | WFS 测量的 x/y 偏差，热图形式 |
| 原始 WFS Zernike 系数 | 平均前的原始系数值 |
| 响应矩阵热图 | 已标配的完整响应矩阵 (.h5) |

---

## 4. 关键设计规则

| 规则 | 说明 |
|------|------|
| 只读查看 | UI 仅读取调试数据文件，不修改 |
| 交互式图表 | 使用 Plotly 提供交互式可视化，支持缩放与悬停查看 |

---

## 5. 相关文档

- 需求规格说明书: [`zernike_debug_viewer_srs.md`](./zernike_debug_viewer_srs.md)
- 标配算法: [`ao_shaping/optimizer/wf/zernike_response_matrix.py`](../../../../../optimizer/wf/zernike_response_matrix.py)
- GUI 索引: [`docs/gui.md`](../../../gui.md)
