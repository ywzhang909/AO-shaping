# =============================================================================
# ioc-mii - MiiCam 高速相机软 IOC(Windows Host 原生运行)
#
# 硬件:MiiCam 相机(MIIUSB SDK;SDK 路径查找见
#   src/ao_shaping/drivers/ccd/miicam/_sdk_setup.py:
#   MIICAM_SDK_PATH > src/.../ccd/_miicam_sdk > libs/miicamsdk.20240728/python)
# 驱动:复用 src/ao_shaping/drivers/ccd/miicam/driver.py 的 CameraStreamManager
#   - create_device 内置相机枚举探测(get_cam_list),SDK 缺失/无相机自动降级离线
#   - 阻塞调用(采图/曝光/开窗)在线程池执行,不卡 caproto 事件循环
#
# PV 一览(前缀 MII-CAM-01:,见 config/ioc.yaml):
#   Acquire            写 1 采集一帧 -> 刷新 Image/ImageWidth/ImageHeight/
#                                    MeanIntensity/FrameCounter
#   ExposureTime       曝光时间(ms,写后钳位 0.011~10000)
#   CenterX/CenterY    ROI 中心(存储,Width/Height 写入时生效)
#   Width/Height       ROI 尺寸(0=全幅;写入即触发 reset_window,回写实际值)
#   AutoExposure       写 0/1/2:0=关,1=连续,2=单次(回读实际生效模式)
#   AutoExposureTarget 自动曝光目标亮度(16~220,驱动钳位)
#   Connected/SerialNumber/CamType/Image/ImageWidth/H/FrameCounter/
#   MeanIntensity      状态只读 PV
#
# 运行(Windows Host):
#   . ..\..\environment.ps1
#   $env:PYTHONPATH = "..\common;$env:PYTHONPATH"
#   python -m ao_epics_common.serve config\ioc.yaml src.miicam_ioc.MiiCamIoc
#
# 验证(本机):
#   python -c "import epics; print(epics.get_pv('MII-CAM-01:Connected'))"
# 验证(WSL,经 CA Gateway):
#   caget MII-CAM-01:Acquire
# =============================================================================
