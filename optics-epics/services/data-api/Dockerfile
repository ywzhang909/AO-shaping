# =============================================================================
# data-api - HTTP API 服务
# 经 CA Gateway 读取 EPICS PV,以 REST 接口暴露给上层应用。
#
# 构建/运行(在 services/ 目录):
#   docker compose build data-api
#   docker compose up -d data-api
#
# 端点:
#   GET /health                    健康检查
#   GET /pv/{name}                 读取单个 PV 值(JSON)
#   GET /pv?name=A&name=B          批量读取(JSON 数组)
#   PUT /pv/{name}                 写入 PV(body: {"value": ...})
#   GET /pvs                       列出 IOC 声明的 PV 注册表
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# CA 客户端:pyepics(经网关读写)
RUN pip install --no-cache-dir pyepics fastapi "uvicorn[standard]" pydantic

COPY src/ /app/src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
