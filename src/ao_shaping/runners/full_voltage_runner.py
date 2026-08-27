"""Full-scale voltage runner using the async AsyncMicroDM driver.

Alternates ALL units of every configured R50Power controller *simultaneously*
between 0V and a specified voltage.  Unlike ``alt_voltage_runner`` there is no
per-channel selection — the whole mirror is driven uniformly on every cycle.

The alternation runs on asyncio via :class:`AsyncMicroDM` so the event loop
is never blocked during TCP transmission.

Usage:
    python -m ao_shaping.runners.full_voltage_runner --voltage 20
    python -m ao_shaping.runners.full_voltage_runner --ips 192.168.0.101,192.168.0.102 --voltage 30 --freq 2

Or via main CLI:
    python -m ao_shaping.main full-voltage --voltage 20
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from typing import NoReturn

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.dm.MicroDM import VOLTAGE_MAX, VOLTAGE_MIN
from ao_shaping.drivers.dm.asyn_micro_dm import AsyncMicroDM
from ao_shaping.utils.cli_helpers import setup_coredumpy

DEFAULT_IPS = ["192.168.0.101"]
DEFAULT_TIMEOUT = 10.0

# Global state for signal handler
_running = True


def _signal_handler(signum: int, frame) -> None:
    global _running
    _running = False
    click.echo("\n⏹  收到中断信号, 正在安全关闭...")


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
        return 1
    click.echo(f"✅ 已连接 {len(ok_ids)}/{len(ips)} 个控制器")

    # Relay on
    if relay_on:
        click.echo("⚡ 继电器上电... ", nl=False)
        relay_results = await dm.set_relay(True)
        if any(r.success for r in relay_results.values()):
            click.echo("✅")
        else:
            click.echo("❌ 失败")
            await dm.shutdown(home_voltage=home_voltage)
            return 1

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Print run info
    click.echo("")
    click.echo("=" * 54)
    click.echo("  全量交替电压下发 (AsyncMicroDM)")
    click.echo("=" * 54)
    click.echo(f"  控制器:      {', '.join(ips)}")
    click.echo(f"  电压范围:    0V ↔ {alt_voltage:.1f}V  (全部单元同时)")
    click.echo(f"  频率:        {alt_freq:.2f} Hz")
    click.echo(f"  周期:        {1.0 / alt_freq:.3f} 秒")
    dur_desc = f"{alt_duration:.1f} 秒" if alt_duration > 0 else "持续运行 (Ctrl+C 停止)"
    click.echo(f"  持续时间:    {dur_desc}")
    click.echo("=" * 54)
    click.echo("")

    half_period = 1.0 / (2.0 * alt_freq)

    # Pre-encode the two uniform states into raw command bytes (one per
    # controller).  The hot loop then replays these cached bytes via
    # send_frame_commands — zero numpy conversion / buffer fill / allocation.
    vs_off = np.zeros(dm.DM_Num)
    vs_on = np.full(dm.DM_Num, alt_voltage)
    cmd_off = dm.build_frame_commands(vs_off)
    cmd_on = dm.build_frame_commands(vs_on)

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
                    logger.warning("控制器下发失败: {}", r.error)
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
@click.option("--ips", "ips_str", default=None, type=str,
              help="Controller IPs, comma-separated (default: 192.168.0.101)")
@click.option("--voltage", "alt_voltage", required=True, type=float,
              help="Voltage for ALL units (V, [-20, 120])")
@click.option("--freq", "alt_freq", default=1.0, type=float,
              help="Alternation frequency (Hz, default: 1.0)")
@click.option("--duration", "alt_duration", default=0.0, type=float,
              help="Duration in seconds (0=until Ctrl+C, default: 0)")
@click.option("--relay-on/--no-relay-on", default=True,
              help="Auto relay on before starting (default: True)")
@click.option("--home-voltage", default=0.0, type=float,
              help="Home voltage on shutdown (default: 0.0)")
@click.option("--timeout", default=DEFAULT_TIMEOUT, type=float,
              help="Controller connect/send timeout (s, default: 10.0)")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging")
def run(
    ips_str: str | None,
    alt_voltage: float,
    alt_freq: float,
    alt_duration: float,
    relay_on: bool,
    home_voltage: float,
    timeout: float,
    debug: bool,
) -> None:
    """全量交替电压下发工具 (AsyncMicroDM)

    所有单元同时、均匀地在 0V 和指定电压之间交替。基于 asyncio 异步驱动。

    Examples:

        # 单个控制器全部单元交替 20V, 1Hz, 持续运行直到 Ctrl+C
        python -m ao_shaping.runners.full_voltage_runner --voltage 20

        # 两个控制器, 30V, 2Hz, 持续 10 秒
        python -m ao_shaping.runners.full_voltage_runner \
            --ips 192.168.0.101,192.168.0.102 --voltage 30 --freq 2.0 --duration 10
    """
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    if not (VOLTAGE_MIN <= alt_voltage <= VOLTAGE_MAX):
        click.echo(f"❌ 电压 {alt_voltage} V 超出硬件范围 [{VOLTAGE_MIN}, {VOLTAGE_MAX}] V")
        sys.exit(1)
    if not (VOLTAGE_MIN <= home_voltage <= VOLTAGE_MAX):
        click.echo(f"❌ 归位电压 {home_voltage} V 超出硬件范围 [{VOLTAGE_MIN}, {VOLTAGE_MAX}] V")
        sys.exit(1)
    if alt_freq <= 0:
        click.echo("❌ 频率必须大于 0")
        sys.exit(1)
    if timeout <= 0:
        click.echo("❌ 超时时间必须大于 0")
        sys.exit(1)

    if ips_str is None:
        ip_list = list(DEFAULT_IPS)
    else:
        ip_list = [s.strip() for s in ips_str.split(",") if s.strip()]
        if not ip_list:
            click.echo("❌ IP 列表为空")
            sys.exit(1)

    rc = asyncio.run(
        _amain(ip_list, alt_voltage, alt_freq, alt_duration, relay_on, home_voltage, timeout)
    )
    sys.exit(rc)


# Define __main__ entry point (NoReturn to satisfy type checkers)
def main() -> NoReturn:
    try:
        run()
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"❌ 运行时错误: {e}")
        logger.exception("Full voltage runner failed")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    setup_coredumpy()
    main()
