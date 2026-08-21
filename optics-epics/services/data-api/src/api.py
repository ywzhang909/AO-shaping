"""data-api:EPICS PV HTTP 网关服务。

经 CA Gateway(或直连 IOC)读写 Channel Access PV,暴露 REST 接口。
环境变量:
    EPICS_CA_ADDR_LIST  - CA 地址列表(默认指向网关)
    EPICS_CA_AUTO_ADDR_LIST - 是否自动广播(默认 NO)
"""
from __future__ import annotations

import os
from typing import Any

import epics
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Optics EPICS Data API", version="1.0.0")

# CA 客户端环境已在容器 ENV 中注入(EPICS_CA_ADDR_LIST 等),pyepics 读取生效。
_ca_addr_list = os.environ.get("EPICS_CA_ADDR_LIST", "")


class PutRequest(BaseModel):
    value: Any = Field(..., description="要写入的 PV 值")


def _get_pv(name: str, timeout: float = 2.0) -> epics.PV:
    """获取 PV 连接,超时抛出 404。"""
    pv = epics.PV(name, connection_timeout=timeout)
    if not pv.connected:
        raise HTTPException(status_code=404, detail=f"PV {name} 未连接(检查 IOC 与 CA Gateway)")
    return pv


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ca_addr_list": _ca_addr_list,
    }


@app.get("/pv/{name}")
def get_pv(name: str) -> dict:
    """读取单个 PV:返回 {name, value, units, timestamp}。"""
    pv = _get_pv(name)
    try:
        return {
            "name": name,
            "value": pv.get(timeout=2.0),
            "units": pv.units,
            "timestamp": pv.timestamp,
            "connected": True,
        }
    except Exception as exc:  # noqa: BLE001 - 统一转为 HTTP 错误
        raise HTTPException(status_code=502, detail=f"读取 PV {name} 失败: {exc}") from exc


@app.get("/pv")
def get_pvs(name: list[str]) -> list[dict]:
    """批量读取多个 PV。"""
    results: list[dict] = []
    for n in name:
        try:
            results.append(get_pv(n))
        except HTTPException:
            results.append({"name": n, "value": None, "connected": False})
    return results


@app.put("/pv/{name}")
def put_pv(name: str, req: PutRequest) -> dict:
    """写入 PV。"""
    pv = _get_pv(name)
    try:
        pv.put(req.value, timeout=2.0)
        return {"name": name, "value": req.value, "written": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"写入 PV {name} 失败: {exc}") from exc


@app.get("/pvs")
def list_pvs() -> dict:
    """列出推荐 PV 前缀(由各 IOC 的 config/ioc.yaml 声明)。"""
    return {
        "prefixes": [
            "DH-CAM-01:",
            "SLM-01:",
            "WFS-01:",
            "DM-01:",
            "MII-CAM-01:",
        ],
        "note": "完整 PV 列表见各 IOC config/ioc.yaml",
    }
