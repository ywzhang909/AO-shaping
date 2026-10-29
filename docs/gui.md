# GUI 界面 (Streamlit)

> 本文件从 README.md 拆出（2026-10-13）。每个 GUI 页面的完整操作手册与需求规格说明书位于 `docs/GUI/<子包>/<页面名>/`（与 py 文件同名的包目录下），文件名为 `<页面名>_srs.md` 与 `<页面名>_user_manual.md`。

项目提供了基于 Streamlit 的图形界面，**代码**按设备域分包于 `src/ao_shaping/gui/` 下，各 UI 独立运行；对应的**文档**在 `docs/GUI/` 下同名目录：

```bash
# R50 控制器 (单单元/单控制器/组/联合控制)
streamlit run src/ao_shaping/gui/r50/r50_controller_ui.py
  - 手册: docs/GUI/r50/r50_controller_ui/r50_controller_ui_user_manual.md
  - SRS:  docs/GUI/r50/r50_controller_ui/r50_controller_ui_srs.md

# Micro-DM 驱动控制
streamlit run src/ao_shaping/gui/dm/micro_dm_ui.py
  - 手册: docs/GUI/dm/micro_dm_ui/micro_dm_ui_user_manual.md
  - SRS:  docs/GUI/dm/micro_dm_ui/micro_dm_ui_srs.md

# Zernike 响应矩阵校准
streamlit run src/ao_shaping/gui/zernike/zernike_response_matrix_ui.py
  - 手册: docs/GUI/zernike/zernike_response_matrix_ui/zernike_response_matrix_ui_user_manual.md
  - SRS:  docs/GUI/zernike/zernike_response_matrix_ui/zernike_response_matrix_ui_srs.md

# SLM 校准
streamlit run src/ao_shaping/gui/slm/slm_calibration_ui.py
  - 手册: docs/GUI/slm/slm_calibration_ui/slm_calibration_ui_user_manual.md
  - SRS:  docs/GUI/slm/slm_calibration_ui/slm_calibration_ui_srs.md

# 多 SLM 控制器 (相位图案生成/下发, 含闭环整形)
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py
  - 手册: docs/GUI/slm/multi_slm_controller/multi_slm_controller_user_manual.md
  - SRS:  docs/GUI/slm/multi_slm_controller/multi_slm_controller_srs.md

# 1300 陶瓷单元查看器 (网格浏览 + 图片标注)
streamlit run src/ao_shaping/gui/r50/ceramic_viewer.py
  - 手册: docs/GUI/r50/ceramic_viewer/ceramic_viewer_user_manual.md
  - SRS:  docs/GUI/r50/ceramic_viewer/ceramic_viewer_srs.md

# CCD 实时图像分析 (大恒/MIICAM + 椭圆拟合 + 截面)
streamlit run src/ao_shaping/gui/ccd/ccd_analyzer.py
  - 手册: docs/GUI/ccd/ccd_analyzer/ccd_analyzer_user_manual.md
  - SRS:  docs/GUI/ccd/ccd_analyzer/ccd_analyzer_srs.md

# Zernike 调试查看器 (响应矩阵/系数可视化)
streamlit run src/ao_shaping/gui/zernike/zernike_debug_viewer.py
  - 手册: docs/GUI/zernike/zernike_debug_viewer/zernike_debug_viewer_user_manual.md
  - SRS:  docs/GUI/zernike/zernike_debug_viewer/zernike_debug_viewer_srs.md

# Micro-DM 图像分析 (diff 图/网格)
streamlit run src/ao_shaping/gui/dm/md_img_analysis_ui.py
  - 手册: docs/GUI/dm/md_img_analysis_ui/md_img_analysis_ui_user_manual.md
  - SRS:  docs/GUI/dm/md_img_analysis_ui/md_img_analysis_ui_srs.md

# 多形状整形 campaign 操作台 (形状 + 尺寸是唯一必填, 其余自动派生)
streamlit run src/ao_shaping/gui/slm/multishape_campaign_ui.py
  - 手册: docs/GUI/slm/multishape_campaign_ui/multishape_campaign_ui_user_manual.md
  - SRS:  docs/GUI/slm/multishape_campaign_ui/multishape_campaign_ui_srs.md
```

## 各页面文档索引

| 页面 | 用户手册 | 需求规格说明书 |
|------|----------|--------------|
| R50 控制器 | [`r50_controller_ui_user_manual.md`](./GUI/r50/r50_controller_ui/r50_controller_ui_user_manual.md) | [`r50_controller_ui_srs.md`](./GUI/r50/r50_controller_ui/r50_controller_ui_srs.md) |
| Micro-DM 驱动控制 | [`micro_dm_ui_user_manual.md`](./GUI/dm/micro_dm_ui/micro_dm_ui_user_manual.md) | [`micro_dm_ui_srs.md`](./GUI/dm/micro_dm_ui/micro_dm_ui_srs.md) |
| Zernike 响应矩阵校准 | [`zernike_response_matrix_ui_user_manual.md`](./GUI/zernike/zernike_response_matrix_ui/zernike_response_matrix_ui_user_manual.md) | [`zernike_response_matrix_ui_srs.md`](./GUI/zernike/zernike_response_matrix_ui/zernike_response_matrix_ui_srs.md) |
| SLM 校准 | [`slm_calibration_ui_user_manual.md`](./GUI/slm/slm_calibration_ui/slm_calibration_ui_user_manual.md) | [`slm_calibration_ui_srs.md`](./GUI/slm/slm_calibration_ui/slm_calibration_ui_srs.md) |
| 多 SLM 控制器 | [`multi_slm_controller_user_manual.md`](./GUI/slm/multi_slm_controller/multi_slm_controller_user_manual.md) | [`multi_slm_controller_srs.md`](./GUI/slm/multi_slm_controller/multi_slm_controller_srs.md) |
| 1300 陶瓷单元查看器 | [`ceramic_viewer_user_manual.md`](./GUI/r50/ceramic_viewer/ceramic_viewer_user_manual.md) | [`ceramic_viewer_srs.md`](./GUI/r50/ceramic_viewer/ceramic_viewer_srs.md) |
| CCD 实时图像分析 | [`ccd_analyzer_user_manual.md`](./GUI/ccd/ccd_analyzer/ccd_analyzer_user_manual.md) | [`ccd_analyzer_srs.md`](./GUI/ccd/ccd_analyzer/ccd_analyzer_srs.md) |
| Zernike 调试查看器 | [`zernike_debug_viewer_user_manual.md`](./GUI/zernike/zernike_debug_viewer/zernike_debug_viewer_user_manual.md) | [`zernike_debug_viewer_srs.md`](./GUI/zernike/zernike_debug_viewer/zernike_debug_viewer_srs.md) |
| Micro-DM 图像分析 | [`md_img_analysis_ui_user_manual.md`](./GUI/dm/md_img_analysis_ui/md_img_analysis_ui_user_manual.md) | [`md_img_analysis_ui_srs.md`](./GUI/dm/md_img_analysis_ui/md_img_analysis_ui_srs.md) |
| 多形状整形 campaign | [`multishape_campaign_ui_user_manual.md`](./GUI/slm/multishape_campaign_ui/multishape_campaign_ui_user_manual.md) | [`multishape_campaign_ui_srs.md`](./GUI/slm/multishape_campaign_ui/multishape_campaign_ui_srs.md) |
