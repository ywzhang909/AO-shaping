# =============================================================================
# ioc-dhcam - 大恒(Daheng)GigE 相机软 IOC(Windows Host 原生运行)
#
# 硬件:大恒 GigE 相机(gxipy SDK, MONO8)
# 驱动:复用 src/ao_shaping/drivers/ccd/daheng/driver.py 的 DahengCamManager
#   - create_device 内置相机枚举探测(get_cam_list),无相机自动降级离线
#   - 阻塞调用(采图/曝光/开窗)在线程池执行,不卡 caproto 事件循环
#
# PV 一览(前缀 DH-CAM-01:,见 config/ioc.yaml):
#   Acquire        写 1 采集一帧 -> 刷新 Image/ImageWidth/ImageHeight/
#                                MeanIntensity/FrameCounter
#   ExposureTime   曝光时间(ms,写后钳位)
#   CenterX/CenterY ROI 中心(存储,Width/Height 写入时生效)
#   Width/Height   ROI 尺寸(0=全幅;写入即触发 reset_window,回写实际量化值)
#   AutoExposure   写 1 触发自动曝光 -> 刷新 ExposureTime/MeanIntensity
#   Connected      连接状态(只读)
#   SerialNumber   相机序列号(只读)
#   CamType        设备类型(只读)
#   Image          最新一帧(int32 波形,只读;上限 1920x1200 全幅)
#   ImageWidth/H   实际帧尺寸(只读)
#   FrameCounter   采集帧计数(只读)
#   MeanIntensity  最近一帧平均亮度(只读)
#
# 运行(Windows Host):
#   . ..\..\environment.ps1
#   $env:PYTHONPATH = "..\common;$env:PYTHONPATH"
#   python -m ao_epics_common.serve config\ioc.yaml src.dhcam_ioc.DHCamIoc
#
# 验证(本机):
#   python -c "import epics; print(epics.get_pv('DH-CAM-01:Connected'))"
# 验证(WSL,经 CA Gateway):
#   caget DH-CAM-01:Acquire
# =============================================================================
