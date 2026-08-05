"""R50 / MicroDM 控制器控制套件 (Streamlit)。

该包统一收纳 R50Power 微驱动器 (MicroDM) 相关的所有 Streamlit 应用与纯逻辑模块:

- ``r50_controller_ui.py``       [新版] 双进程架构控制面板 (Control Service + 薄 UI)
- ``micro_dm_ui.py``             [旧版] 单体 Micro DM 控制 UI (保留兼容)
- ``ceramic_viewer.py``          1300 陶瓷单元查看工具
- ``r50_channel_select.py``      配置 / CSV 索引 / 通道选择 (纯逻辑)
- ``r50_connection.py``          连接工厂 / 仿真设备 / 下电安全 (纯逻辑)
- ``r50_voltage_send.py``        裁剪 / 批量下发 / 发送循环 (纯逻辑)
- ``r50_control_service.py``     [新版] Control Service (独立进程, asyncio)
- ``r50_service_client.py``      [新版] Streamlit 侧客户端封装
"""
