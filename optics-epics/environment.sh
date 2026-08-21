#!/usr/bin/env bash
# =============================================================================
# optics-epics - 全局环境变量
# =============================================================================
# 用法:
#   source environment.sh          # WSL 内 (services 部署)
#   source environment.sh --win    # Windows Host (IOC 运行)
#
# 说明:
#   - WSL_IP: WSL2 实例 eth0 IP(请按实际环境修改,可用 `ip addr show eth0` 查询)
#   - HOST_IP: Windows Host 局域网 IP(硬件网段 192.168.1.0/24)
#   - CA 端口:每个 IOC 使用独立 EPICS_CA_PORT,避免冲突
# =============================================================================

# ---- 网络 ----
# WSL2 实例 IP(NAT 模式,可在 WSL 内用 `ip addr show eth0 | grep inet` 确认)
export WSL_IP="${WSL_IP:-172.26.56.202}"
# Windows Host 硬件网卡 IP(连接 192.168.1.0/24 网段)
export HOST_IP="${HOST_IP:-192.168.1.100}"
# 硬件设备网段
export HW_NETWORK="192.168.1.0/24"

# ---- CA 端口(IOC 各自独立)----
export DHCAM_CA_PORT="${DHCAM_CA_PORT:-5064}"
export SLM_CA_PORT="${SLM_CA_PORT:-5065}"
export WFS_CA_PORT="${WFS_CA_PORT:-5066}"
export DM_CA_PORT="${DM_CA_PORT:-5067}"
export MII_CA_PORT="${MII_CA_PORT:-5068}"

# ---- 网关 ----
# CA Gateway 监听地址(WSL 内 network_mode: host)
export GATEWAY_ADDR="${GATEWAY_ADDR:-${WSL_IP}:5062}"
# 网关需要代理的 IOC 地址(Windows Host)
export IOC_ADDR="${IOC_ADDR:-${HOST_IP}}"
# 网关的 CA 地址列表(所有 IOC 的 IP:端口)
export GATEWAY_CA_ADDR_LIST="${HOST_IP}:${DHCAM_CA_PORT} ${HOST_IP}:${SLM_CA_PORT} ${HOST_IP}:${WFS_CA_PORT} ${HOST_IP}:${DM_CA_PORT} ${HOST_IP}:${MII_CA_PORT}"

# ---- 服务端口 ----
export PHOEBUS_PORT="${PHOEBUS_PORT:-8080}"
export ARCHIVER_PORT="${ARCHIVER_PORT:-17665}"
export DATA_API_PORT="${DATA_API_PORT:-8000}"

# ---- CA 客户端环境(WSL 内 caget/caput 使用)----
# 客户端通过网关访问:CA 地址列表指向 WSL 自身
export EPICS_CA_ADDR_LIST="${WSL_IP}"
export EPICS_CA_AUTO_ADDR_LIST="NO"
export EPICS_CA_MAX_ARRAY_BYTES="10000000"

# ---- 硬件参数 ----
# SLM 波长(nm)
export SLM_WAVELENGTH_NM="${SLM_WAVELENGTH_NM:-1064}"
# SLM 内存槽轮换集合(连续写相位时轮换,避免同一槽位 no-op)
export SLM_MEMORY_SLOTS="${SLM_MEMORY_SLOTS:-3,4,5}"
# DM 控制电压范围(NLight)
export DM_VOLT_MIN="${DM_VOLT_MIN:--300}"
export DM_VOLT_MAX="${DM_VOLT_MAX:-499}"
export DM_N_ACTUATORS="${DM_N_ACTUATORS:-64}"

# ---- 日志 ----
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "[optics-epics] WSL_IP=${WSL_IP} HOST_IP=${HOST_IP}"
echo "[optics-epics] CA ports: DHCAM=${DHCAM_CA_PORT} SLM=${SLM_CA_PORT} WFS=${WFS_CA_PORT} DM=${DM_CA_PORT} MII=${MII_CA_PORT}"
