# Zernike 响应矩阵校准 UI (zernike_response_matrix_ui.py) — 用户手册

## 1. 概述

`zernike_response_matrix_ui.py` 是 Zernike 响应矩阵校准的 Streamlit 前端页面。支持 SLM→WFS Zernike 响应矩阵标配、实时可视化、加载/查看保存的校准结果。

启动方式：

```bash
streamlit run src/ao_shaping/gui/zernike/zernike_response_matrix_ui.py
```

完整技术细节见 [`zernike_response_matrix_ui_srs.md`](./zernike_response_matrix_ui_srs.md)。

---

## 2. 快速上手

### 2.1 连接设备

1. 点击「连接 SLM」建立 SLM 连接。
2. 点击「连接 WFS」建立波前传感器连接。
3. 连接成功后，可进行标配操作。

### 2.2 配置标配参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 最大 Zernike 阶数 | Zernike 多项式最大阶数 | 可配置 |
| 标配循环次数 | 标配循环次数 | 可配置 |
| 平均次数 | 每次测量的平均次数 | 可配置 |
| 等待时间 | 每次扰动后的等待时间 | 可配置 |
| 扰动幅度 | Zernike 扰动幅度 | 可配置 |

### 2.3 运行标配

1. 检查 SLM 与 WFS 均已连接。
2. 配置标配参数。
3. 点击「运行标配」。
4. 标配过程实时可视化。
5. 标配完成后，查看响应矩阵结果。

### 2.4 加载历史结果

1. 点击「加载响应矩阵」。
2. 选择保存的标配结果文件。
3. 查看标配结果。

---

## 3. 关键设计规则

| 规则 | 说明 |
|------|------|
| 设备连接互斥 | 连接前先断开现有连接 |
| 错误隔离 | 设备连接/标配失败通过 `st.error` 显示 |

---

## 4. 相关文档

- 需求规格说明书: [`zernike_response_matrix_ui_srs.md`](./zernike_response_matrix_ui_srs.md)
- 标配算法: [`ao_shaping/optimizer/wf/zernike_response_matrix.py`](../../../../../optimizer/wf/zernike_response_matrix.py)
- GUI 索引: [`docs/gui.md`](../../../../../docs/gui.md)
