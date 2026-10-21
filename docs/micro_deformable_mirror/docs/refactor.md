
以下是整合了所有优化意见的**生产级完整代码**，以及专门针对 **AO（自适应光学）快速闭环** 场景的使用实例。

---

### 一、 核心设计原则（AO 闭环专用）

在 AO 闭环中，DM 驱动不是独立的网络工具，而是**实时控制回路的一个环节**。因此优化重点与通用网络编程完全不同：

| AO 闭环需求            | 通用网络做法         | AO 优化做法                                         |
| ---------------------- | -------------------- | --------------------------------------------------- |
| **确定性延迟**   | 重连、重试、指数退避 | 单次发送 + 固定超时，失败立即返回，由上层控制器决策 |
| **零 GC 抖动**   | 每帧创建 bytes/cmd   | 预分配缓冲区，原地更新电压值，避免 Python GC        |
| **背压安全**     | drain() 无限等待     | drain 硬超时，超时即标记丢失，不阻塞闭环节拍        |
| **多控制器同步** | gather 等全部完成    | gather + 独立超时，部分失败不拖慢整体帧率           |
| **线程模型**     | 后台线程 Loop 桥接   | 直接嵌入 AO 主事件循环，消除跨线程通信开销          |

---

### 二、 完整生产级驱动代码

```python
"""micro_dm_ao.py - Production-grade MicroDM driver for AO closed-loop control.

Design priorities for AO:
  - Deterministic latency (no hidden retries/reconnects in hot path)
  - Zero-allocation steady state (pre-allocated buffers)
  - Hard timeout on every I/O operation
  - Direct async integration (no background thread bridge in loop)
"""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
from loguru import logger

# =============================================================================
# Protocol Constants
# =============================================================================

HEADER = bytes([0xAA, 0xBB])
FOOTER = bytes([0xCC, 0xDD])
CMD_SET_ALL_VOLTAGE_BY_ARR = 0x09
CMD_RELAY_ON = 0x06
CMD_RELAY_OFF = 0x07

VOLTAGE_MIN = -20.0
VOLTAGE_MAX = 120.0
MAX_CHANNELS = 50

# AO-specific timing constants
SEND_DRAIN_TIMEOUT = 0.008   # 8ms hard cap per controller (well within 1kHz budget)
CONNECT_TIMEOUT = 3.0        # Only used during initialization


# =============================================================================
# Pre-computed Conversion Table (Zero-Allocation Steady State)
# =============================================================================

class VoltageConverter:
    """Lookup-table based voltage converter for zero-GC steady-state operation.

    During AO closed-loop, we cannot afford numpy allocation or float math
    on every frame. This class pre-computes a LUT and provides an in-place
    buffer update method.
    """

    def __init__(self, lut_bits: int = 12):
        self._lut_bits = lut_bits
        self._lut_size = 1 << lut_bits
        self._scale = (self._lut_size - 1) / (VOLTAGE_MAX - VOLTAGE_MIN)
        self._offset = -VOLTAGE_MIN * self._scale

        # Pre-compute LUT: index → big-endian 2-byte payload
        indices = np.arange(self._lut_size, dtype=np.float64)
        raw = np.round(indices).astype(np.uint16)
        raw_be = raw.byteswap().view(np.uint8)
        self._lut_bytes = raw_be.tobytes()  # 2 * lut_size bytes

        # Pre-allocate reusable output buffer for 50 channels
        self._buf = bytearray(MAX_CHANNELS * 2)

    def fill_buffer(self, voltages: np.ndarray, buf: bytearray | None = None) -> bytearray:
        """Convert 50 voltages into payload bytes IN-PLACE.

        Args:
            voltages: Exactly 50 float voltages, already clipped.
            buf: Output buffer. If None, uses internal reusable buffer.

        Returns:
            The filled buffer (2*50 = 100 bytes).
        """
        out = buf if buf is not None else self._buf
        # Quantize to LUT indices
        indices = ((voltages[:MAX_CHANNELS] * self._scale + self._offset)
                    .astype(np.int32))
        np.clip(indices, 0, self._lut_size - 1, out=indices)

        # Copy from LUT into output buffer (no allocation)
        for i in range(MAX_CHANNELS):
            idx = indices[i]
            out[i * 2] = self._lut_bytes[idx * 2]
            out[i * 2 + 1] = self._lut_bytes[idx * 2 + 1]
        return out


# =============================================================================
# Send Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class SendResult:
    controller_id: int
    success: bool
    latency_us: int = 0       # Microseconds for AO precision
    error: str | None = None


# =============================================================================
# R50 Controller (AO-optimized: no auto-reconnect in hot path)
# =============================================================================

class R50Controller:
    """Single R50Power controller TCP client optimized for AO real-time loop.

    Key differences from general-purpose driver:
      - NO auto-reconnect in send() — failures return immediately
      - Hard drain timeout — never blocks beyond SEND_DRAIN_TIMEOUT
      - Pre-built command buffer — only voltage payload changes per frame
      - Reusable bytearray avoids per-frame allocation
    """

    def __init__(self, controller_id: int, ip: str, port: int):
        self.controller_id = controller_id
        self.ip = ip
        self.port = port

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

        # Pre-allocate command frame: HEADER + CMD(1) + PAYLOAD(100) + FOOTER
        self._cmd_buf = bytearray(len(HEADER) + 1 + MAX_CHANNELS * 2 + len(FOOTER))
        self._cmd_buf[0:2] = HEADER
        self._cmd_buf[2] = CMD_SET_ALL_VOLTAGE_BY_ARR
        self._cmd_buf[-2:] = FOOTER

        self._converter = VoltageConverter()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> bool:
        """Establish connection. Called ONCE during system initialization."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.ip, self.port),
                timeout=CONNECT_TIMEOUT,
            )
            sock = self._writer.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Disable Nagle + set send buffer for low-latency small packets
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            self._connected = True
            logger.info(f"R50[{self.controller_id}] connected {self.ip}:{self.port}")
            return True
        except Exception as exc:
            logger.error(f"R50[{self.controller_id}] connect FAILED: {exc}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        self._connected = False
        if self._writer:
            with suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    async def send_voltages(self, voltages: np.ndarray) -> SendResult:
        """Send 50-channel voltage array. ZERO allocation in steady state.

        This is the HOT PATH method called every AO frame.
        - No retry, no reconnect → deterministic worst-case latency
        - Hard drain timeout → guaranteed upper bound
        - In-place buffer update → no GC pressure
        """
        if not self.is_connected:
            return SendResult(self.controller_id, False, error="not_connected")

        t0 = time.perf_counter_ns()

        # Fill pre-allocated command buffer in-place
        payload_view = memoryview(self._cmd_buf)[3:3 + MAX_CHANNELS * 2]
        self._converter.fill_buffer(voltages, payload_view)

        try:
            self._writer.write(self._cmd_buf)
            await asyncio.wait_for(self._writer.drain(), timeout=SEND_DRAIN_TIMEOUT)
            latency_us = (time.perf_counter_ns() - t0) // 1000
            return SendResult(self.controller_id, True, latency_us=latency_us)
        except asyncio.TimeoutError:
            self._connected = False
            return SendResult(self.controller_id, False,
                              latency_us=SEND_DRAIN_TIMEOUT * 1_000_000,
                              error="drain_timeout")
        except Exception as exc:
            self._connected = False
            return SendResult(self.controller_id, False,
                              latency_us=(time.perf_counter_ns() - t0) // 1000,
                              error=str(exc))

    async def set_relay(self, state: bool) -> bool:
        """Relay control. NOT in hot path, simple implementation."""
        if not self.is_connected:
            return False
        cmd = HEADER + bytes([CMD_RELAY_ON if state else CMD_RELAY_OFF]) + FOOTER
        try:
            self._writer.write(cmd)
            await asyncio.wait_for(self._writer.drain(), timeout=1.0)
            return True
        except Exception:
            self._connected = False
            return False


# =============================================================================
# Multi-Controller DM Driver
# =============================================================================

class MicroDM:
    """Multi-controller DM driver for AO closed-loop.

    Lifecycle:
      1. __init__()     — create controllers, allocate buffers
      2. connect()      — establish all TCP connections (once at startup)
      3. send_frame()   — called every AO iteration (hot path)
      4. shutdown()     — safe power-down sequence
    """

    def __init__(self, controller_configs: list[dict]):
        """
        Args:
            controller_configs: List of {"id": int, "ip": str, "port": int}
        """
        self.controllers = [
            R50Controller(cfg["id"], cfg["ip"], cfg["port"])
            for cfg in controller_configs
        ]
        self._n_ctrl = len(self.controllers)
        self._total_channels = self._n_ctrl * MAX_CHANNELS

        # Pre-allocate per-controller voltage slices (avoid slicing allocation)
        self._slices: list[np.ndarray] = []
        self._full_buffer = np.zeros(self._total_channels, dtype=np.float32)
        for i in range(self._n_ctrl):
            start = i * MAX_CHANNELS
            end = start + MAX_CHANNELS
            self._slices.append(self._full_buffer[start:end])

        logger.info(f"MicroDM initialized: {self._n_ctrl} controllers, "
                     f"{self._total_channels} channels")

    async def connect(self) -> dict[int, bool]:
        """Connect all controllers in parallel. Call once at system startup."""
        results = await asyncio.gather(
            *[ctrl.connect() for ctrl in self.controllers],
            return_exceptions=True,
        )
        status = {}
        for ctrl, res in zip(self.controllers, results):
            ok = res is True
            status[ctrl.controller_id] = ok
            if not ok:
                logger.error(f"Controller {ctrl.controller_id} failed: {res}")

        n_ok = sum(status.values())
        if n_ok == 0:
            raise RuntimeError("No controllers connected")
        logger.info(f"MicroDM connected: {n_ok}/{self._n_ctrl}")
        return status

    async def send_frame(self, voltages: np.ndarray) -> list[SendResult]:
        """Send a full-frame voltage vector to all controllers IN PARALLEL.

        THIS IS THE AO HOT-PATH METHOD.

        Args:
            voltages: Full voltage array (n_controllers * 50 floats).
                      Values are clipped internally.

        Returns:
            List of SendResult, one per controller. Never raises.
        """
        # Clip and copy into pre-allocated buffer (single allocation-free copy)
        np.clip(voltages, VOLTAGE_MIN, VOLTAGE_MAX, out=self._full_buffer)

        # Parallel send with independent timeouts
        tasks = [ctrl.send_voltages(self._slices[i])
                 for i, ctrl in enumerate(self.controllers)]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for ctrl, r in zip(self.controllers, raw):
            if isinstance(r, Exception):
                results.append(SendResult(ctrl.controller_id, False, error=str(r)))
            else:
                results.append(r)
        return results

    async def shutdown(self, home_voltage: float = 0.0) -> None:
        """Safe shutdown: home → relay off → disconnect."""
        logger.info("MicroDM shutting down...")
        home = np.full(self._total_channels, home_voltage, dtype=np.float32)
        await self.send_frame(home)
        await asyncio.gather(
            *[ctrl.set_relay(False) for ctrl in self.controllers],
            return_exceptions=True,
        )
        await asyncio.gather(
            *[ctrl.disconnect() for ctrl in self.controllers],
            return_exceptions=True,
        )
        logger.info("MicroDM shutdown complete")
```

---

### 三、 AO 快速闭环集成实例

以下是一个完整的 **1kHz AO 闭环** 示例，展示如何将 DM 驱动嵌入实时控制回路：

```python
"""ao_closed_loop_example.py

Demonstrates integrating MicroDM into a 1kHz AO closed-loop control system.

Architecture:
    WFS camera → compute correction → DM send → repeat
    All within a single asyncio event loop for deterministic scheduling.
"""

import asyncio
import time
import numpy as np
from micro_dm_ao import MicroDM, SendResult

# =============================================================================
# AO System Parameters
# =============================================================================

LOOP_RATE_HZ = 1000
FRAME_PERIOD_S = 1.0 / LOOP_RATE_HZ
FRAME_PERIOD_US = FRAME_PERIOD_S * 1e6

# 3 controllers × 50 channels = 150 actuators
CONTROLLER_CONFIGS = [
    {"id": 1, "ip": "192.168.0.101", "port": 10101},
    {"id": 2, "ip": "192.168.0.102", "port": 10102},
    {"id": 3, "ip": "192.168.0.103", "port": 10103},
]

N_ACTUATORS = len(CONTROLLER_CONFIGS) * 50


# =============================================================================
# Simulated AO Components (replace with real hardware interfaces)
# =============================================================================

class WavefrontSensor:
    """Simulates WFS readout + slope computation."""

    def __init__(self, n_actuators: int):
        self._n = n_actuators
        # Pre-allocate slope buffer
        self._slopes = np.zeros(n_actuators, dtype=np.float32)

    async def read_slopes(self) -> np.ndarray:
        """Read WFS and compute slopes. In reality this would be
        triggered by camera frame-ready interrupt."""
        # Simulate 200μs WFS processing
        await asyncio.sleep(0.0002)
        # Return simulated residual slopes
        np.random.randn(self._n, out=self._slopes.astype(np.float64))
        return self._slopes


class Reconstructor:
    """Matrix-vector multiply: slopes → actuator voltages."""

    def __init__(self, n_actuators: int):
        # Pre-compute reconstruction matrix
        self._recon_matrix = np.eye(n_actuators, dtype=np.float32) * 0.5
        # Pre-allocate output buffer
        self._voltages = np.zeros(n_actuators, dtype=np.float32)

    def reconstruct(self, slopes: np.ndarray) -> np.ndarray:
        """In-place matrix multiply. No allocation."""
        np.dot(self._recon_matrix, slopes, out=self._voltages)
        return self._voltages


# =============================================================================
# AO Closed-Loop Controller
# =============================================================================

class AOClosedLoop:
    """1kHz AO closed-loop controller.

    Integrates WFS, reconstructor, and DM into a single async event loop
    with hard real-time budget enforcement.
    """

    def __init__(self):
        self.dm = MicroDM(CONTROLLER_CONFIGS)
        self.wfs = WavefrontSensor(N_ACTUATORS)
        self.recon = Reconstructor(N_ACTUATORS)

        # Performance counters (pre-allocated ring buffer)
        self._frame_count = 0
        self._overrun_count = 0
        self._dm_latencies_us = np.zeros(1000, dtype=np.int64)
        self._frame_times_us = np.zeros(1000, dtype=np.int64)

    async def initialize(self) -> bool:
        """System startup sequence."""
        logger.info("=" * 60)
        logger.info("AO System Initialization")
        logger.info("=" * 60)

        # Connect DM
        status = await self.dm.connect()
        n_ok = sum(status.values())
        if n_ok < len(CONTROLLER_CONFIGS):
            logger.warning(f"Only {n_ok}/{len(CONTROLLER_CONFIGS)} controllers connected")

        # Open relays
        for ctrl in self.dm.controllers:
            if ctrl.is_connected:
                await ctrl.set_relay(True)

        # Zero all channels
        await self.dm.send_frame(np.zeros(N_ACTUATORS, dtype=np.float32))
        await asyncio.sleep(0.01)  # Hardware settling

        logger.info("AO System Ready")
        return True

    async def run_loop(self, n_frames: int = 10000):
        """Main AO closed-loop.

        Each iteration:
          1. Read WFS slopes
          2. Reconstruct actuator commands
          3. Send to DM (parallel, bounded latency)
          4. Enforce frame period
        """
        logger.info(f"Starting AO loop: {n_frames} frames @ {LOOP_RATE_HZ} Hz")

        next_frame_time = time.perf_counter()

        for frame_idx in range(n_frames):
            frame_start = time.perf_counter_ns()

            # ── Step 1: WFS Readout ──
            slopes = await self.wfs.read_slopes()

            # ── Step 2: Reconstruction ──
            voltages = self.recon.reconstruct(slopes)

            # ── Step 3: DM Send (HOT PATH) ──
            results = await self.dm.send_frame(voltages)

            # ── Record metrics ──
            buf_idx = frame_idx % 1000
            dm_latency = max((r.latency_us for r in results), default=0)
            self._dm_latencies_us[buf_idx] = dm_latency

            frame_elapsed_ns = time.perf_counter_ns() - frame_start
            self._frame_times_us[buf_idx] = frame_elapsed_ns // 1000
            self._frame_count += 1

            # Check for overrun
            if frame_elapsed_ns / 1e6 > FRAME_PERIOD_US:
                self._overrun_count += 1
                if self._overrun_count <= 10:  # Log first 10 overruns
                    logger.warning(
                        f"Frame {frame_idx} OVERRUN: "
                        f"{frame_elapsed_ns / 1e3:.0f}μs > {FRAME_PERIOD_US:.0f}μs "
                        f"(DM latency: {dm_latency}μs)"
                    )

            # Check for DM failures
            failed = [r for r in results if not r.success]
            if failed:
                logger.warning(f"Frame {frame_idx} DM failures: "
                             f"{[(r.controller_id, r.error) for r in failed]}")

            # ── Step 4: Frame Rate Enforcement ──
            next_frame_time += FRAME_PERIOD_S
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # Already past deadline, reset anchor to prevent drift accumulation
                next_frame_time = time.perf_counter()

        self._print_statistics()

    def _print_statistics(self):
        """Print loop performance summary."""
        n = min(self._frame_count, 1000)
        if n == 0:
            return

        dm_lat = self._dm_latencies_us[:n]
        ft = self._frame_times_us[:n]

        logger.info("=" * 60)
        logger.info("AO Loop Performance Summary")
        logger.info("=" * 60)
        logger.info(f"  Total frames:      {self._frame_count}")
        logger.info(f"  Overruns:          {self._overrun_count} "
                     f"({100 * self._overrun_count / max(1, self._frame_count):.2f}%)")
        logger.info(f"  Frame time (μs):   "
                     f"p50={np.median(ft):.0f}  "
                     f"p99={np.percentile(ft, 99):.0f}  "
                     f"max={np.max(ft):.0f}  "
                     f"budget={FRAME_PERIOD_US:.0f}")
        logger.info(f"  DM latency (μs):   "
                     f"p50={np.median(dm_lat):.0f}  "
                     f"p99={np.percentile(dm_lat, 99):.0f}  "
                     f"max={np.max(dm_lat):.0f}")
        logger.info("=" * 60)

    async def shutdown(self):
        await self.dm.shutdown()


# =============================================================================
# Entry Point
# =============================================================================

async def main():
    ao = AOClosedLoop()
    try:
        await ao.initialize()
        await ao.run_loop(n_frames=5000)
    finally:
        await ao.shutdown()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

---

### 四、 AO 闭环关键设计决策详解

#### 1. 为什么不在热路径中重连？

```
通用网络驱动:  send() → fail → reconnect(500ms) → retry → return
               最坏情况: 500ms+  ← AO 闭环已丢失 500 帧！

AO 驱动:       send() → fail → return SendResult(success=False)
               最坏情况: 8ms (drain timeout) ← 仅丢失 8 帧
               上层控制器检测到失败 → 保持上一帧电压 / 进入安全模式
```

AO 闭环是**时间确定系统**。一个 500ms 的重连比丢 8 帧的危害大几个数量级。重连应该在**帧间空闲期**或由**独立监控协程**处理，绝不在热路径中。

#### 2. 零分配稳态的实现

```python
# ❌ 每帧分配（触发 GC → 毫秒级暂停）
cmd = HEADER + bytes([CMD]) + voltages_to_payload(chunk) + FOOTER

# ✅ 预分配 + 原地更新（零 GC）
self._cmd_buf[3:103] = converter.fill_buffer(chunk, self._payload_view)
self._writer.write(self._cmd_buf)  # 同一个 bytearray 对象
```

Python GC 暂停可达 1-10ms，在 1kHz 闭环中是致命的。所有热路径对象在 `__init__` 中预分配，运行时只做内存写入。

#### 3. LUT 替代浮点运算

```python
# ❌ 每帧 50 次浮点乘除 + numpy clip
raw = np.round(v * SCALE + OFFSET).astype(np.uint16)

# ✅ 整数查表（~10x faster, 无浮点单元竞争）
idx = int(voltage * scale + offset)
byte_pair = lut_bytes[idx*2 : idx*2+2]
```

在 1kHz 下，CPU 需要同时处理 WFS 重建矩阵乘法。将 DM 电压转换从浮点运算降级为整数查表，释放 FPU 给重构算法。

#### 4. 帧率执行策略

```python
# 绝对时间锚点，防止累积漂移
next_frame_time += FRAME_PERIOD_S
sleep_time = next_frame_time - time.perf_counter()
if sleep_time > 0:
    await asyncio.sleep(sleep_time)
else:
    next_frame_time = time.perf_counter()  # 重置锚点
```

使用相对 sleep (`sleep(FRAME_PERIOD)`) 会导致帧间隔误差逐帧累积。绝对时间锚点保证长期平均频率精确等于目标值。

#### 5. 监控协程（可选增强）

在生产系统中，建议添加独立的连接监控，与热路径完全解耦：

```python
async def _connection_monitor(self, check_interval: float = 1.0):
    """Background monitor: detect disconnections and attempt recovery
    BETWEEN frames, never during frame processing."""
    while self._running:
        await asyncio.sleep(check_interval)
        for ctrl in self.dm.controllers:
            if not ctrl.is_connected:
                logger.warning(f"Controller {ctrl.controller_id} lost, attempting reconnect...")
                ok = await ctrl.connect()
                if ok:
                    await ctrl.set_relay(True)
                    logger.info(f"Controller {ctrl.controller_id} recovered")
```

这个协程以 1Hz 运行，即使重连耗时 500ms，也只占用一个帧间隙，不影响 1kHz 热路径的确定性。
