"""R50Power (Micro-DM) voltage-drive CLI commands.

Two commands live in this module, mirroring how ``slm/zernike_matrix_runner.py``
carries both ``zernike-matrix`` and ``closed-loop``:

* :func:`run` → ``alt-voltage`` — synchronous, single controller, selectable
  channel subset, optional NI-DAQ ADC capture to CSV.
* :func:`full_voltage_run` → ``full-voltage`` — asyncio, N controllers, every
  unit driven uniformly, pre-encoded command bytes, no per-channel selection.

They were separate modules until 2026-10-04. What is actually shared is the
scaffolding around the loop -- signal handling, the console banner, validation
and the ``main()`` bootstrap -- so this merge is **co-location plus boilerplate
extraction**, not a unification of the two drive loops. The loops stay separate
on purpose: one is a blocking single-IP channel loop, the other an async
multi-IP frame loop with a deadline scheduler. Abstracting over "sync vs async"
*and* "per-channel vs whole-frame" would add indirection without changing any
observable behaviour.

Usage::

    # alt-voltage (default when no selector is given)
    python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 20
    python -m ao_shaping.runners.micro_drive.voltage_runner alt-voltage --ip 192.168.0.101 --voltage 20

    # full-voltage
    python -m ao_shaping.runners.micro_drive.voltage_runner full-voltage --voltage 20

Both are also reachable through the unified CLI::

    python -m ao_shaping.main alt-voltage  --ip 192.168.0.101 --voltage 20
    python -m ao_shaping.main full-voltage --voltage 20

.. note::
   ``slm/zernike_matrix_runner.py`` has the same two-commands-per-module shape
   but its ``__main__`` block calls :func:`run` unconditionally, so
   ``python -m ... zernike_matrix_runner closed-loop`` does **not** work despite
   what the README claims. This module dispatches on an explicit selector
   instead (see :func:`main`), so both documented invocations below are real.
"""

from __future__ import annotations

import asyncio
import csv
import queue
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, NoReturn

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.micro import (
    DEFAULT_IPS,
    MAX_CHANNELS,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
    AsyncMicroDM,
    R50Controller,
)
from ao_shaping.runners.runner_common import (
    AltVoltageRunnerParams,
    FullVoltageRunnerParams,
    with_params,
)
from ao_shaping.utils.io.cli_helpers import setup_coredumpy
from ao_shaping.utils.io.network import ping_reachable

# Optional ADC driver — TYPE_CHECKING lets us annotate while handling runtime absence
if TYPE_CHECKING:
    from ao_shaping.drivers.adc.driver import NidaqADC

try:
    from ao_shaping.drivers.adc.driver import NidaqADC as _NidaqADC

    ADC_AVAILABLE = True
except ImportError:
    _NidaqADC = None  # type: ignore[assignment]
    ADC_AVAILABLE = False

# Hardware limits for the synchronous single-controller path. The async path
# uses the driver's own VOLTAGE_MIN/VOLTAGE_MAX instead — these two are
# separate constants and are deliberately not unified, see module docstring.
HW_VOLTAGE_MIN = -20.0
HW_VOLTAGE_MAX = 120.0
SINGLE_CHANNELS = 50

#: Selectors accepted by :func:`main` for `python -m` dispatch.
COMMAND_SELECTORS = ("alt-voltage", "full-voltage")

# Global state for signal handler
_running = True


def _signal_handler(signum: int, frame) -> None:
    """Flip ``_running`` so the drive loop exits and ``finally`` runs."""
    global _running
    _running = False
    click.echo("\n⏹  收到中断信号, 正在安全关闭...")


def _banner(title: str, rows: list[str]) -> None:
    """Print the boxed run-info header shared by both commands.

    Args:
        title: Header line inside the box.
        rows: ``label: value`` lines, printed in order.
    """
    rule = "=" * 54
    click.echo("")
    click.echo(rule)
    click.echo(f"  {title}")
    click.echo(rule)
    for row in rows:
        click.echo(f"  {row}")
    click.echo(rule)
    click.echo("")


def _enable_debug(enabled: bool) -> None:
    """Turn on debug logging when the ``--debug`` flag was passed."""
    if enabled:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")


def _duration_desc(seconds: float) -> str:
    """Human-readable run duration for the banner."""
    return f"{seconds:.1f} 秒" if seconds > 0 else "持续运行 (Ctrl+C 停止)"


# --------------------------------------------------------------------------- #
# alt-voltage — synchronous, single controller, selectable channels
# --------------------------------------------------------------------------- #


def _adc_worker(
    adc: NidaqADC,
    stop_event: Event,
    result_queue: queue.Queue[tuple[float, float]],
    t_start: float,
) -> None:
    """Background thread that continuously reads ADC and pushes (timestamp, value) into queue."""
    while not stop_event.is_set():
        try:
            mean_v = adc.read_mean()
            ts = time.time() - t_start
            result_queue.put((ts, mean_v))
        except Exception:
            pass


def _resolve_port(ip: str, port: int | None) -> int:
    """Derive the controller port from the IP's last octet when not given."""
    if port is not None:
        return port
    try:
        last_octet = int(ip.split(".")[-1])
    except (ValueError, IndexError):
        click.echo("❌ 无法从 IP 解析端口, 请使用 --port 手动指定")
        sys.exit(1)
    return 10000 + last_octet


def _parse_channels(channel_str: str | None) -> list[int]:
    """Parse ``--channels`` into a validated channel index list."""
    if channel_str is None or channel_str.lower() == "all":
        return list(range(SINGLE_CHANNELS))
    try:
        channels = [int(c.strip()) for c in channel_str.split(",")]
        if not all(0 <= c < SINGLE_CHANNELS for c in channels):
            click.echo(f"❌ 通道号必须在 0-{SINGLE_CHANNELS - 1} 范围内")
            sys.exit(1)
    except (ValueError, IndexError):
        click.echo("❌ 通道格式错误, 请使用逗号分隔 (如 0,1,2)")
        sys.exit(1)
    return channels


@click.command("alt-voltage")
@click.pass_context
@with_params(AltVoltageRunnerParams, kw_name="params")
def run(ctx: click.Context, params: AltVoltageRunnerParams) -> None:
    """交替电压下发工具

    在 0V 和指定电压之间循环交替发送到 R50Power 控制器的指定单元。
    可选同步采集 NI DAQ ADC 电压信号，结果自动保存为 CSV。

    Examples:

        # 全部 50 个通道交替 20V, 1Hz, 持续运行直到 Ctrl+C
        python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 20

        # 通道 0-5 交替 30V, 2Hz, 持续 10 秒
        python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 30 --freq 2.0 --duration 10 --channels 0,1,2,3,4,5

        # 全部通道, 0.5Hz, 跳过 ping 和自动上电
        python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 15 --freq 0.5 --no-ping-first --no-relay-on

        # 同步采集 ADC (Dev1/ai0, 5kHz, 10 samples/read)
        python -m ao_shaping.runners.micro_drive.voltage_runner --ip 192.168.0.101 --voltage 20 --adc-enabled --adc-device Dev1 --adc-channel ai0
    """
    global _running

    _enable_debug(params.debug)

    port = _resolve_port(params.ip, params.port)

    # Validate voltage
    if params.alt_voltage < HW_VOLTAGE_MIN or params.alt_voltage > HW_VOLTAGE_MAX:
        click.echo(
            f"❌ 电压 {params.alt_voltage} V 超出硬件范围 "
            f"[{HW_VOLTAGE_MIN}, {HW_VOLTAGE_MAX}] V"
        )
        sys.exit(1)

    # Validate frequency
    if params.alt_freq <= 0:
        click.echo("❌ 频率必须大于 0")
        sys.exit(1)

    channels = _parse_channels(params.channel_str)

    # Ping check
    if params.ping_first:
        click.echo(f"📡 Ping 测试 {params.ip}... ", nl=False)
        if ping_reachable(params.ip, timeout=2.0):
            click.echo("✅ 可达")
        else:
            click.echo("❌ 不可达")
            click.echo("  使用 --no-ping-first 跳过 ping 测试")
            sys.exit(1)

    # Connect
    click.echo(f"🔌 连接 {params.ip}:{port}... ", nl=False)
    ctrl = R50Controller(controller_id=1, ip=params.ip, port=port)
    if not ctrl.open():
        click.echo("❌ 连接失败")
        sys.exit(1)
    click.echo("✅ 已连接")

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Relay on
    if params.relay_on:
        click.echo("⚡ 继电器上电... ", nl=False)
        if ctrl.set_relay(True):
            click.echo("✅")
        else:
            click.echo("❌ 失败")
            ctrl.close()
            sys.exit(1)

    # --- ADC setup ---
    adc: NidaqADC | None = None
    adc_thread: Thread | None = None
    adc_stop_event: Event | None = None
    adc_queue: queue.Queue[tuple[float, float]] | None = None
    adc_data: list[dict[str, float]] = []
    adc_t_start = 0.0

    if params.adc_enabled:
        if not ADC_AVAILABLE:
            logger.warning("ADC enabled but nidaqmx not installed")
            click.echo("⚠️ --adc-enabled 但 nidaqmx 未安装，ADC 功能将被禁用")
        else:
            try:
                click.echo(f"🟡 连接 ADC ({params.adc_device}/{params.adc_channel}, {params.adc_sample_rate} Hz)... ", nl=False)
                adc = NidaqADC(
                    device_name=params.adc_device,
                    channel=params.adc_channel,
                    sample_rate=params.adc_sample_rate,
                    samples_per_channel=params.adc_samples_per_read,
                )
                adc.open()
                click.echo("✅")
                logger.info("ADC connected: {}/{} @ {} Hz", params.adc_device, params.adc_channel, params.adc_sample_rate)
            except Exception as e:
                click.echo(f"❌ {e}")
                logger.error("ADC connection failed: {}", e)
                adc = None

        if adc is not None:
            adc_queue = queue.Queue()
            adc_stop_event = Event()
            adc_t_start = time.time()
            adc_thread = Thread(
                target=_adc_worker,
                args=(adc, adc_stop_event, adc_queue, adc_t_start),
                daemon=True,
            )
            adc_thread.start()
            logger.info("ADC background acquisition thread started")

    # Print run info
    ch_desc = f"{len(channels)} 个通道" if len(channels) == SINGLE_CHANNELS else f"通道 {params.channel_str}"
    info_rows = [
        f"控制器:      {params.ip}:{port}",
        f"电压范围:    0V ↔ {params.alt_voltage:.1f}V",
        f"频率:        {params.alt_freq:.2f} Hz",
        f"周期:        {1.0 / params.alt_freq:.3f} 秒",
        f"通道:        {ch_desc}",
        f"持续时间:    {_duration_desc(params.alt_duration)}",
    ]
    if adc is not None:
        info_rows.append(
            f"ADC:          {params.adc_device}/{params.adc_channel}, "
            f"{params.adc_sample_rate} Hz, {params.adc_samples_per_read} smp/rd"
        )
    _banner("交替电压下发启动", info_rows)

    # Alternating loop
    t_start = time.time()
    cycle_count = 0
    half_period = 1.0 / (2.0 * params.alt_freq)
    state = 0  # 0 = sending 0V, 1 = sending input voltage

    try:
        while _running:
            elapsed = time.time() - t_start
            if params.alt_duration > 0 and elapsed >= params.alt_duration:
                click.echo(f"\n⏱  达到运行时长 {params.alt_duration:.1f} 秒")
                break

            # Drain ADC queue
            if adc_queue is not None:
                n_drained = 0
                while True:
                    try:
                        adc_ts, adc_mean = adc_queue.get_nowait()
                        adc_data.append({
                            "time_elapsed_s": adc_ts,
                            "applied_voltage_V": 0.0 if state == 0 else params.alt_voltage,
                            "adc_mean_V": adc_mean,
                        })
                        n_drained += 1
                    except queue.Empty:
                        break
                if n_drained > 0:
                    logger.debug("Drained {} ADC samples this cycle", n_drained)

            # Send voltage to selected channels
            v = 0.0 if state == 0 else params.alt_voltage
            if len(channels) == SINGLE_CHANNELS:
                ctrl.set_all_channel_voltage(v)
            else:
                for ch in channels:
                    ctrl.set_channel_voltage(ch, v)

            # Progress indicator
            v_label = "0V" if state == 0 else f"{params.alt_voltage:.1f}V"
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            click.echo(f"  [{elapsed_str}] → {v_label}  (cycle {cycle_count // 2 + 1})", nl=False)

            if adc_data:
                latest = adc_data[-1]
                vals = [d["adc_mean_V"] for d in adc_data]
                click.echo(
                    f"  ADC: {latest['adc_mean_V']:.4f}V"
                    f"  [min={min(vals):.4f}  max={max(vals):.4f}  avg={sum(vals)/len(vals):.4f}]"
                )
            else:
                click.echo("")

            state = 1 - state
            cycle_count += 1
            time.sleep(half_period)

    except Exception as e:
        click.echo(f"\n❌ 运行异常: {e}")
        logger.exception("Alt voltage loop error")
    finally:
        # Stop ADC thread
        if adc_stop_event is not None:
            adc_stop_event.set()
        if adc_thread is not None:
            adc_thread.join(timeout=2.0)

        # Save ADC data to CSV
        if adc_data:
            csv_dir = Path("data")
            csv_dir.mkdir(exist_ok=True)
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            csv_path = csv_dir / f"alt_voltage_adc_{ts_str}.csv"
            try:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["time_elapsed_s", "applied_voltage_V", "adc_mean_V"])
                    writer.writeheader()
                    writer.writerows(adc_data)
                click.echo(f"📄 ADC 数据已保存: {csv_path}  ({len(adc_data)} 条记录)")
                logger.info("ADC data saved: {} ({} records)", csv_path, len(adc_data))
            except Exception as e:
                logger.exception("Failed to save ADC data")
                click.echo(f"⚠️ ADC 数据保存失败: {e}")

        # Close ADC
        if adc is not None:
            try:
                adc.close()
                click.echo("  ✅ ADC 已关闭")
                logger.info("ADC closed")
            except Exception as e:
                logger.exception("ADC close error")
                click.echo(f"  ⚠️ ADC 关闭失败: {e}")

        _safe_shutdown(ctrl, channels)


def _safe_shutdown(ctrl: R50Controller, channels: list[int]) -> None:
    """Send 0V, relay off, and close connection."""
    click.echo("")
    click.echo("⏹  安全关闭中...")

    # Send 0V to all selected channels
    try:
        if len(channels) == SINGLE_CHANNELS:
            ctrl.set_all_channel_voltage(0.0)
        else:
            for ch in channels:
                ctrl.set_channel_voltage(ch, 0.0)
        click.echo("  ✅ 已下发 0V 到所有通道")
    except Exception as e:
        click.echo(f"  ⚠️  下发 0V 失败: {e}")

    # Relay off
    try:
        ctrl.set_relay(False)
        click.echo("  ✅ 继电器已下电")
    except Exception as e:
        click.echo(f"  ⚠️  继电器下电失败: {e}")

    # Close connection
    try:
        ctrl.close()
        click.echo("  ✅ 连接已关闭")
    except Exception as e:
        click.echo(f"  ⚠️  关闭连接失败: {e}")

    click.echo("🏁 退出")


# --------------------------------------------------------------------------- #
# full-voltage — asyncio, N controllers, whole-frame pre-encoded bytes
# --------------------------------------------------------------------------- #


async def _amain(
    ips: list[str],
    alt_voltage: float,
    alt_freq: float,
    alt_duration: float,
    relay_on: bool,
    home_voltage: float,
    timeout: float,
) -> int:
    global _running
    _running = True

    dm = AsyncMicroDM(ips=ips, timeout=timeout, safety_mode=False)

    click.echo("🔌 连接控制器... ", nl=False)
    results = await dm.connect_all()
    ok_ids = [cid for cid, ok in results.items() if ok]
    if not ok_ids:
        click.echo("❌ 全部控制器连接失败")
        for cid, ip, port, _conn in dm.controller_info:
            logger.error(
                "控制器 {} ({}:{}) 连接失败", cid, ip, port
            )
        return 1
    click.echo(f"✅ 已连接 {len(ok_ids)}/{len(ips)} 个控制器")

    # 逐控制器连接状态回显 (ip:port + 是否可用), 便于定位"哪个控制器没连上".
    for cid, ip, port, connected in dm.controller_info:
        mark = "✅" if connected else "❌"
        click.echo(f"    {mark} #{cid} {ip}:{port}")
        if not connected:
            logger.warning("控制器 {} ({}:{}) 未连接, 该控制器通道将被跳过", cid, ip, port)
    n_active = sum(1 for _, _, _, connected in dm.controller_info if connected)
    click.echo(
        f"    ↳ 实际下发通道: {n_active} 控制器 × {MAX_CHANNELS} 通道"
        f" = {n_active * MAX_CHANNELS} 通道 (逻辑总数 {dm.DM_Num})"
    )

    # Relay on
    if relay_on:
        click.echo("⚡ 继电器上电... ", nl=False)
        relay_results = await dm.set_relay(True)
        if any(r.success for r in relay_results.values()):
            click.echo("✅")
        else:
            click.echo("❌ 失败")
            for r in relay_results.values():
                if not r.success:
                    logger.warning(
                        "控制器继电器上电失败: id={} ip={} err={}",
                        r.controller_id, r.ip, r.error,
                    )
            await dm.shutdown(home_voltage=home_voltage)
            return 1

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Print run info
    _banner(
        "全量交替电压下发 (AsyncMicroDM)",
        [
            f"控制器:      {', '.join(ips)}",
            f"电压范围:    0V ↔ {alt_voltage:.1f}V  (全部单元同时)",
            f"频率:        {alt_freq:.2f} Hz",
            f"周期:        {1.0 / alt_freq:.3f} 秒",
            f"持续时间:    {_duration_desc(alt_duration)}",
        ],
    )

    half_period = 1.0 / (2.0 * alt_freq)

    # Pre-encode the two uniform states into raw command bytes (one per
    # controller).  The hot loop then replays these cached bytes via
    # send_frame_commands — zero numpy conversion / buffer fill / allocation.
    vs_off = np.zeros(dm.DM_Num)
    vs_on = np.full(dm.DM_Num, alt_voltage)
    cmd_off = dm.build_frame_commands(vs_off)
    cmd_on = dm.build_frame_commands(vs_on)

    logger.debug(
        "预构建指令: {} 控制器 × {} 通道/控制器 = {} 通道/帧",
        len(cmd_off), MAX_CHANNELS, len(cmd_off) * MAX_CHANNELS,
    )

    state = 0  # 0 = sending 0V, 1 = sending input voltage
    cycle_count = 0
    t_start = time.monotonic()
    n_frames = 0
    total_latency_us = 0.0
    n_sent = 0

    # Progress output every N frames; the console write itself would otherwise
    # dominate the per-cycle latency budget.
    PRINT_EVERY = max(1, int(round(2.0 / half_period))) if half_period > 0 else 1
    SLEEP_SLICE = 0.05  # s — keeps Ctrl+C responsive during long half-periods

    try:
        while _running:
            # Cadence: every send is scheduled at a multiple of half_period from
            # start, so send/print duration never accumulates phase drift.
            deadline = t_start + n_frames * half_period

            cmds = cmd_off if state == 0 else cmd_on
            send_results = await dm.send_frame_commands(cmds)
            n_frames += 1
            n_sent += len(send_results)

            for r in send_results:
                if not r.success:
                    logger.warning(
                        "控制器下发失败: id={} ip={} err={}",
                        r.controller_id, r.ip, r.error,
                    )
                else:
                    total_latency_us += r.latency_us

            if (n_frames % PRINT_EVERY) == 0:
                elapsed = time.monotonic() - t_start
                avg_lat = total_latency_us / n_sent if n_sent else 0.0
                v_label = "0V" if state == 0 else f"{alt_voltage:.1f}V"
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                click.echo(
                    f"  [{elapsed_str}] → 全量 {v_label}  (cycle {cycle_count // 2 + 1})"
                    f"  [avg send {avg_lat:.0f} µs]"
                )

            state = 1 - state
            cycle_count += 1

            if alt_duration > 0 and (time.monotonic() - t_start) >= alt_duration:
                click.echo(f"\n⏱  达到运行时长 {alt_duration:.1f} 秒")
                break

            # Sleep in slices so a SIGINT is honored promptly on long waits.
            while _running:
                remaining = deadline + half_period - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(remaining, SLEEP_SLICE))
    except Exception as e:
        click.echo(f"\n❌ 运行异常: {e}")
        logger.exception("Full voltage loop error")
    finally:
        click.echo("\n⏹  安全关闭中...")
        await dm.shutdown(home_voltage=home_voltage)
        click.echo("🏁 退出")

    return 0


@click.command("full-voltage")
@click.pass_context
@with_params(FullVoltageRunnerParams, kw_name="params")
def full_voltage_run(ctx: click.Context, params: FullVoltageRunnerParams) -> None:
    """全量交替电压下发工具 (AsyncMicroDM)

    所有单元同时、均匀地在 0V 和指定电压之间交替。基于 asyncio 异步驱动。

    Examples:

        # 全部 26 台控制器 (.101~.126) 全部单元交替 20V, 1Hz, 持续运行直到 Ctrl+C
        python -m ao_shaping.runners.micro_drive.voltage_runner full-voltage --voltage 20

        # 两个控制器, 30V, 2Hz, 持续 10 秒
        python -m ao_shaping.runners.micro_drive.voltage_runner full-voltage \
            --ips 192.168.0.101,192.168.0.102 --voltage 30 --freq 2.0 --duration 10
    """
    _enable_debug(params.debug)

    if not (VOLTAGE_MIN <= params.alt_voltage <= VOLTAGE_MAX):
        click.echo(f"❌ 电压 {params.alt_voltage} V 超出硬件范围 [{VOLTAGE_MIN}, {VOLTAGE_MAX}] V")
        sys.exit(1)
    if not (VOLTAGE_MIN <= params.home_voltage <= VOLTAGE_MAX):
        click.echo(f"❌ 归位电压 {params.home_voltage} V 超出硬件范围 [{VOLTAGE_MIN}, {VOLTAGE_MAX}] V")
        sys.exit(1)
    if params.alt_freq <= 0:
        click.echo("❌ 频率必须大于 0")
        sys.exit(1)
    if params.timeout <= 0:
        click.echo("❌ 超时时间必须大于 0")
        sys.exit(1)

    if params.ips_str is None:
        ip_list = list(DEFAULT_IPS)
        ip_source = "默认值 (静态 IP .101~.126)"
    else:
        ip_list = [s.strip() for s in params.ips_str.split(",") if s.strip()]
        ip_source = "--ips 参数"
        if not ip_list:
            click.echo("❌ IP 列表为空")
            sys.exit(1)

    click.echo(f"🖥️  控制器列表: {', '.join(ip_list)}  (来源: {ip_source}, {len(ip_list)} 个)")
    logger.debug("IP 来源: {} → {}", ip_source, ip_list)

    rc = asyncio.run(
        _amain(ip_list, params.alt_voltage, params.alt_freq, params.alt_duration, params.relay_on, params.home_voltage, params.timeout)
    )
    sys.exit(rc)


# --------------------------------------------------------------------------- #
# `python -m` entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> NoReturn:
    """Dispatch to a command by selector for `python -m` execution.

    ``python -m ao_shaping.runners.micro_drive.voltage_runner [selector] [opts]``

    With no selector the module runs ``alt-voltage``, preserving the
    shortest invocation. ``full-voltage`` must be named explicitly since its
    options do not overlap the default command's.

    Args:
        argv: Argument vector without the program name. Defaults to
            ``sys.argv[1:]``.

    Raises:
        SystemExit: Always; the chosen Click command decides the code.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    command: click.Command = run
    label = "alt-voltage"
    if args and args[0] in COMMAND_SELECTORS:
        label = args[0]
        args = args[1:]
        command = full_voltage_run

    try:
        command(args)
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"❌ 运行时错误: {e}")
        logger.exception("{} runner failed", label)
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    setup_coredumpy()
    main()