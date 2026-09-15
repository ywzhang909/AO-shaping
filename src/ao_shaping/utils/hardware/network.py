"""Network utilities: ICMP reachability checks (shared by R50 controller tooling).

Leaf module: depends only on the standard library, safe for drivers/runners/
gui/tools to import. All controller connectivity checks should use
:func:`ping_reachable` so platform-specific ping flags stay consistent.
"""

from __future__ import annotations

import socket
import subprocess
import sys


def ping_reachable(ip: str, timeout: float = 2.0) -> bool:
    """对目标 IP 直接执行 ICMP ping 可达性测试 (平台差异已处理)。

    注意: Windows 下 ping 超时参数为小写 ``-w`` (毫秒), Linux/macOS 为
    大写 ``-W`` (秒), 必须按平台区分, 否则 ping 会报 "Bad option" 而失败。
    所有 R50 控制器联通检查应统一使用本函数。

    Args:
        ip: 目标 IP 地址或主机名
        timeout: 超时时间 (s)

    Returns:
        ping 可达返回 True, 否则 False
    """
    if sys.platform.startswith("win"):
        cmd = ["ping", "-n", "1", "-w", str(int(timeout * 1000)), ip]
    else:
        cmd = ["ping", "-c", "1", "-W", str(int(timeout)), ip]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 1,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def ip_last_octet(ip: str) -> int | None:
    """Extract the last octet of an IPv4 address (192.168.0.101 -> 101).

    Returns None when the input has no parseable trailing octet.
    """
    parts = ip.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def controller_tcp_port(ip: str, default: int | None = None) -> int | None:
    """R50Power controller TCP port: 10000 + last IP octet (192.168.0.101 -> 10101).

    Returns ``default`` when the IP has no parseable last octet.
    """
    octet = ip_last_octet(ip)
    if octet is None:
        return default
    return 10000 + octet


def tcp_reachable(ip: str, port: int, timeout: float = 2.0) -> bool:
    """Probe a TCP port; returns reachable or not."""
    try:
        with socket.create_connection((ip, int(port)), timeout=timeout):
            return True
    except OSError:
        return False
