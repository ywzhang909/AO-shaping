# =============================================================================
# services/phoebus - Phoebus OPI 工具
# =============================================================================
# 目录结构:
#   settings/            Phoebus 偏好设置(挂载到容器 /settings)
#   opi/auto-generated/  ioc.yaml 自动生成的 OPI 文件
#   opi/custom/          手工维护的 OPI 文件
#
# 图形模式运行(需要 WSLg):
#   wsl --update   # 确保 WSLg 可用
#   cd services && docker compose up -d phoebus   # 取消注释 compose 中图形模式段
#
# headless 模式:
#   docker compose up -d phoebus
#   docker exec -it phoebus sh   # 开发/调试 OPI 资源
# =============================================================================
