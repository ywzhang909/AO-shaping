"""R50 调试模块 — 外部 TCP 转发 / 本地捕获 / 操作日志。

包含 :class:`DebugTcpClient` 与全部 ``_debug_*`` 助手, 以及模块级
调试状态 (锁 / 环形缓冲 / 本地服务器启停 Event)。

本模块是底层叶子模块: 只依赖 ``r50_channel_select`` 中的常量,
不依赖任何上层 UI 模块。
"""

from __future__ import annotations

import collections
import json
import socket
import threading
import time
from typing import Any

import streamlit as st
from loguru import logger

from ao_shaping.gui.r50.r50_channel_select import (
    DEBUG_HOST,
    DEBUG_PORT,
    P,
)


# =============================================================================
# 调试 TCP 客户端 (调试日志转发到外部 TCP 监听器)
# =============================================================================

class DebugTcpClient:
    """外部 TCP 调试客户端: 将调试日志 JSON 发送到 ``127.0.0.1:9999``。"""

    def __init__(self, host: str = DEBUG_HOST, port: int = DEBUG_PORT) -> None:
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self._connected = False
        self._last_attempt = 0.0
        self._reconnect_interval = 5.0

    def configure(self, host: str, port: int) -> None:
        """配置新的目标地址并断开旧连接。"""
        if host != self.host or port != self.port:
            self.disconnect()
            self.host = host
            self.port = port

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """建立连接 (带 5s 退避, 失败静默)。"""
        now = time.time()
        if now - self._last_attempt < self._reconnect_interval:
            return
        self._last_attempt = now
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.0)
            self.sock.connect((self.host, self.port))
            self._connected = True
            logger.debug(f"[DebugTcpClient] 已连接 {self.host}:{self.port}")
        except OSError:
            self._connected = False
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None

    def send(self, data: dict[str, Any]) -> None:
        """发送一行 JSON 日志。"""
        if not self._connected:
            return
        try:
            payload = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
            self.sock.sendall(payload)
        except OSError:
            self._connected = False
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None

    def disconnect(self) -> None:
        """断开连接。"""
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self._connected = False


# =============================================================================
# 调试日志助手 (外部 TCP 转发 + 本地捕获 + 操作日志)
# =============================================================================

_local_debug_lock = threading.Lock()
_local_debug_buffer: collections.deque[str] = collections.deque(maxlen=256)


def _debug_log_packet(cmd_name: str, packet: bytes) -> None:
    """将原始数据包写入调试日志 (指令日志 + 操作日志 + 外部 TCP)。"""
    hex_str = packet.hex()
    ts = time.strftime("%H:%M:%S", time.localtime())
    st.session_state[f"{P}_debug_log"].append(f"[{ts}] {cmd_name} {hex_str}")
    _debug_add_op(cmd_name, f"packet={hex_str}")
    if st.session_state.get(f"{P}_debug", False):
        st.session_state[f"{P}_debug_tcp_client"].send(
            {"cmd": cmd_name, "hex": hex_str, "ts": time.time()}
        )


def _debug_tcp_connect() -> None:
    """(重新)连接外部 TCP 调试通道。"""
    if st.session_state.get(f"{P}_debug", False):
        st.session_state[f"{P}_debug_tcp_client"].connect()


def _debug_tcp_disconnect() -> None:
    """断开外部 TCP 调试通道。"""
    st.session_state[f"{P}_debug_tcp_client"].disconnect()


def _debug_add_op(operation: str, detail: str, ip: str = "") -> None:
    """向操作日志追加一行 (仅 UI 层记录)。"""
    ts = time.strftime("%H:%M:%S", time.localtime(time.time()))
    line = f"[{ts}] {operation}"
    if ip:
        line += f" @{ip}"
    if detail:
        line += f" | {detail}"
    st.session_state[f"{P}_debug_op_log"].append(line)


# ---------------------------------------------------------------------------
# 本地调试 TCP 服务器 (捕获外部客户端发来的调试日志)
# 注意: 后台线程不读取 session_state — 用 threading.Event 控制启停。
# ---------------------------------------------------------------------------

_local_debug_server_event = threading.Event()
_local_debug_server_error: str | None = None


def _drain_local_debug_buffer() -> None:
    """将后台线程捕获的调试行搬运到 session_state (主线程执行)。"""
    with _local_debug_lock:
        lines = list(_local_debug_buffer)
        _local_debug_buffer.clear()
    if lines:
        st.session_state[f"{P}_local_debug_logs"].extend(lines)


def _debug_tcp_start_local_server() -> None:
    """启动本地调试 TCP 服务器 (后台线程, 127.0.0.1:port)。"""
    global _local_debug_server_error
    port = int(st.session_state.get(f"{P}_debug_tcp_port", DEBUG_PORT))
    _local_debug_server_error = None
    _local_debug_server_event.set()
    st.session_state[f"{P}_local_debug_server"] = True
    st.session_state[f"{P}_local_debug_logs"].clear()
    st.session_state[f"{P}_debug_tcp_host"] = "127.0.0.1"
    st.session_state[f"{P}_debug_tcp_port"] = port
    client = st.session_state[f"{P}_debug_tcp_client"]
    client.configure("127.0.0.1", port)
    client.connect()

    def _run() -> None:
        server_sock: socket.socket | None = None
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(("127.0.0.1", port))
            server_sock.listen(1)
            server_sock.settimeout(0.2)
            while _local_debug_server_event.is_set():
                try:
                    conn, _addr = server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    conn.settimeout(0.2)
                    while _local_debug_server_event.is_set():
                        data = conn.recv(4096)
                        if not data:
                            break
                        for line in data.decode("utf-8", errors="replace").splitlines():
                            if line.strip():
                                with _local_debug_lock:
                                    _local_debug_buffer.append(line)
                except socket.timeout:
                    pass
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        except OSError as exc:
            _local_debug_server_error = str(exc)
            logger.error(f"[DebugTcpServer] 绑定失败: {exc}")
        finally:
            _local_debug_server_event.clear()
            if server_sock is not None:
                try:
                    server_sock.close()
                except OSError:
                    pass

    threading.Thread(target=_run, name="debug-tcp-server", daemon=True).start()


def _debug_tcp_stop_local_server() -> None:
    """停止本地调试 TCP 服务器。"""
    _local_debug_server_event.clear()
    st.session_state[f"{P}_local_debug_server"] = False
    with _local_debug_lock:
        _local_debug_buffer.clear()
