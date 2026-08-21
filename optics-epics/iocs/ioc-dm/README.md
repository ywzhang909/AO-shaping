# =============================================================================
# ioc-dm - 变形镜软 IOC(Windows Host 原生运行)
#
# 硬件:
#   - NLight 64 通道 DM, UDP 192.168.6.10:1001, 电压 [-300, 499] V
#   - R50Power MicroDM, 异步 TCP 192.168.0.101..126, 电压 [-20, 120] V
# 驱动:复用 src/ao_shaping/drivers/dm/{nlight,micro}/driver.py
#
# PV 一览(前缀 NLight-DM:,见 config/ioc.yaml):
#   Voltages      全部通道电压数组(float64),写后下发
#   Zero          写 1 复位全部通道到 0V
#   Relay         继电器 0/1(仅 MicroDM)
#   HV            高压 0/1(仅 NLight)
#   Connected     连接状态(只读)
#   ActuatorCount 通道数(只读)
#   VMin / VMax   电压范围(只读)
#   Type          设备类型(只读)
#
# 切换 DM 类型:改 ioc.yaml 的 type/params 与 prefix(MicroDM:)
#
# 运行(Windows Host):
#   . ..\..\environment.ps1
#   $env:PYTHONPATH = "..\common;$env:PYTHONPATH"
#   python -m ao_epics_common.serve config\ioc.yaml src.dm_ioc.DMIoc
#
# ⚠ 真实 NLight 部署必须在 AO-shaping 项目根目录运行(cwd=项目根):
#   NLight 驱动在模块导入期用相对路径加载 data/dm_adj.txt(相邻矩阵),
#   且需要 libs\Drv_UDPST 在 PATH 中;create_device 已内置 192.168.6.10:1001
#   TCP 可达性探测,不可达自动降级离线(全部写被拒,避免误报 Connected)。
#   从项目根启动示例:
#     cd D:\Projects\TIFO\AO-shaping
#     $env:PYTHONPATH = "optics-epics\iocs\common;optics-epics\iocs\ioc-dm;src;libs"
#     python -m ao_epics_common.serve optics-epics\iocs\ioc-dm\config\ioc.yaml src.dm_ioc.DMIoc
#
# 验证(本机):
#   python -c "import epics; print(epics.get_pv('NLight-DM:Connected'))"
# 验证(WSL,经 CA Gateway):
#   caget NLight-DM:Connected
# =============================================================================
