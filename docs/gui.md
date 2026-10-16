# GUI 界面 (Streamlit)

> 本文件从 README.md 拆出（2026-10-13）。SLM GUI 完整操作手册见 [`docs/gui/multi_slm_controller_user_manual.md`](gui/multi_slm_controller_user_manual.md)，需求规格说明书见 [`docs/gui/multi_slm_controller_srs.md`](gui/multi_slm_controller_srs.md)。

项目提供了基于Streamlit的图形界面，按设备域分包于 `src/ao_shaping/gui/` 下，各 UI 独立运行：

```bash
# R50 控制器 (单单元/单控制器/组/联合控制)
streamlit run src/ao_shaping/gui/r50/r50_controller_ui.py

# Micro-DM 驱动控制
streamlit run src/ao_shaping/gui/dm/micro_dm_ui.py

# Zernike 响应矩阵校准
streamlit run src/ao_shaping/gui/zernike/zernike_response_matrix_ui.py

# SLM 校准
streamlit run src/ao_shaping/gui/slm/slm_calibration_ui.py

# 多 SLM 控制器 (相位图案生成/下发, 含闭环整形)
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py

# 1300 陶瓷单元查看器 (网格浏览 + 图片标注)
streamlit run src/ao_shaping/gui/r50/ceramic_viewer.py

# CCD 实时图像分析 (大恒/MIICAM + 椭圆拟合 + 截面)
streamlit run src/ao_shaping/gui/ccd/ccd_analyzer.py

# Zernike 调试查看器 (响应矩阵/系数可视化)
streamlit run src/ao_shaping/gui/zernike/zernike_debug_viewer.py

# Micro-DM 图像分析 (diff 图/网格)
streamlit run src/ao_shaping/gui/dm/md_img_analysis_ui.py
```

`multi_slm_controller.py` 是 Santec SLM-200 的 Streamlit 前端页面，支持最多两台 SLM、相位图案生成/下发、CSV 加载与导出、周期切换后台任务，以及闭环远场光束整形（GS+SPGD 与正向模型闭环）。完整操作手册与需求规格说明书见 [`docs/gui/multi_slm_controller_user_manual.md`](gui/multi_slm_controller_user_manual.md) 与 [`docs/gui/multi_slm_controller_srs.md`](gui/multi_slm_controller_srs.md)。
