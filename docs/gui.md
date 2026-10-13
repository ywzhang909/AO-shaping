# GUI 界面 (Streamlit)

> 本文件从 README.md 拆出（2026-10-13）。SLM GUI 完整操作手册见 [docs/slm/slm_gui_manual.md](slm/slm_gui_manual.md)。

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

# 多 SLM 控制器 (相位图案生成/下发, 含 GS方形整形)
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py

# 1300 陶瓷单元查看器 (网格浏览 + 图片标注)
streamlit run src/ao_shaping/gui/r50/ceramic_viewer.py
```

`multi_slm_controller.py` 提供多种全息相位图案生成：平场、闪耀光栅、达曼光栅、涡旋相位、**GS方形整形**等。支持**从 CSV 加载相位**（格式：1200×1920，值 0~1023，首行/首列为 Y/X 索引），走 `load_gray_from_csv` → `csv_to_phase` → `display_phase` 三步管线，与 GUI 预览共用 `create_phase_from_array()` 保证字节级一致；也支持将当前相位或相位 A/B **导出为弧度 CSV**（`Santec.save_phase_to_csv()`，保留 Y/X 行列索引）。导出文件的数据区是弧度值而非灰度值，不能交给 `load_gray_from_csv()` 或 `csv_to_phase()`；重新使用时应按弧度读取并传入 `create_phase_from_array()`。其中 GS方形整形模式：上传远场光斑图片 → 自动测量光斑直径并计算方形边长（边长 = 光斑直径 × 尺寸因子，自动换算相机/SLM 像素间距）→ 在 SLM 分辨率网格上运行 Gerchberg-Saxton → 下发 uint16 相位到 SLM。支持实时迭代进度显示与逐轮相位下发（内存槽自动轮换）。方形尺寸/相位正确性由仿真测试验证（`tests/ao_shaping/gui/slm/test_gs_square_shaping.py`，角谱传播断言）。
