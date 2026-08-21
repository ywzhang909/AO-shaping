# =============================================================================
# optics-epics - Windows Host 环境变量 (IOC 原生运行)
# =============================================================================
# 用法(在启动 IOC 的 PowerShell 会话中先执行):
#   . .\environment.ps1
#
# 说明:
#   - IOC 以原生 Python 进程运行在 Windows Host,通过厂商 DLL SDK 控制硬件
#   - 每个 IOC 使用独立 EPICS_CA_PORT
#   - CA 客户端(如 Phoebus)经 WSL 内 CA Gateway 访问,地址列表指向 WSL_IP
# =============================================================================

# ---- 网络 ----
# WSL2 实例 IP(用于 IOC 间互相发现;客户端经网关访问时用网关地址)
$env:WSL_IP = if ($env:WSL_IP) { $env:WSL_IP } else { "172.26.56.202" }
# Windows Host 硬件网卡 IP(连接 192.168.1.0/24 网段)
$env:HOST_IP = if ($env:HOST_IP) { $env:HOST_IP } else { "192.168.1.100" }

# ---- CA 端口(IOC 各自独立)----
$env:DHCAM_CA_PORT = if ($env:DHCAM_CA_PORT) { $env:DHCAM_CA_PORT } else { "5064" }
$env:SLM_CA_PORT   = if ($env:SLM_CA_PORT)   { $env:SLM_CA_PORT }   else { "5065" }
$env:WFS_CA_PORT   = if ($env:WFS_CA_PORT)   { $env:WFS_CA_PORT }   else { "5066" }
$env:DM_CA_PORT    = if ($env:DM_CA_PORT)    { $env:DM_CA_PORT }    else { "5067" }
$env:MII_CA_PORT   = if ($env:MII_CA_PORT)   { $env:MII_CA_PORT }   else { "5068" }

# ---- CA 客户端环境(本机直连测试用;跨 WSL 访问走网关)----
# 本机 IOC 直连:地址列表含所有 IOC 端口;跨子网访问由 WSL 网关转发
$env:EPICS_CA_ADDR_LIST = if ($env:EPICS_CA_ADDR_LIST) {
    $env:EPICS_CA_ADDR_LIST
} else {
    "127.0.0.1:$($env:DHCAM_CA_PORT) 127.0.0.1:$($env:SLM_CA_PORT) 127.0.0.1:$($env:WFS_CA_PORT) 127.0.0.1:$($env:DM_CA_PORT) 127.0.0.1:$($env:MII_CA_PORT)"
}
$env:EPICS_CA_AUTO_ADDR_LIST = "NO"
$env:EPICS_CA_MAX_ARRAY_BYTES = "10000000"

# ---- 厂商 SDK 路径 ----
# Santec SLM:SLMFunc.dll 所在目录
$env:AO_SHAPING_SLM_DLL_DIR = if ($env:AO_SHAPING_SLM_DLL_DIR) {
    $env:AO_SHAPING_SLM_DLL_DIR
} else {
    "C:\Program Files\Santec\SLM-SDK"
}
# MiiCam:miicam.py + miicam.dll 所在目录
$env:MIICAM_SDK_PATH = if ($env:MIICAM_SDK_PATH) {
    $env:MIICAM_SDK_PATH
} else {
    "C:\Program Files\MiiCam\SDK"
}

# ---- 硬件参数 ----
$env:SLM_WAVELENGTH_NM = if ($env:SLM_WAVELENGTH_NM) { $env:SLM_WAVELENGTH_NM } else { "1064" }
$env:SLM_MEMORY_SLOTS  = if ($env:SLM_MEMORY_SLOTS)  { $env:SLM_MEMORY_SLOTS }  else { "3,4,5" }
$env:DM_VOLT_MIN  = if ($env:DM_VOLT_MIN)  { $env:DM_VOLT_MIN }  else { "-300" }
$env:DM_VOLT_MAX  = if ($env:DM_VOLT_MAX)  { $env:DM_VOLT_MAX }  else { "499" }
$env:DM_N_ACTUATORS = if ($env:DM_N_ACTUATORS) { $env:DM_N_ACTUATORS } else { "64" }

Write-Host "[optics-epics] WSL_IP=$($env:WSL_IP) HOST_IP=$($env:HOST_IP)"
Write-Host "[optics-epics] CA ports: DHCAM=$($env:DHCAM_CA_PORT) SLM=$($env:SLM_CA_PORT) WFS=$($env:WFS_CA_PORT) DM=$($env:DM_CA_PORT) MII=$($env:MII_CA_PORT)"
