"""Alternating voltage CLI runner for R50Power controllers.

Cycles between 0V and a specified voltage on selected channels,
with controllable frequency and duration.

Usage:
    python -m ao_shaping.runners.alt_voltage_runner --ip 192.168.0.101 --voltage 20

Or via main CLI:
    python -m ao_shaping.main alt-voltage --ip 192.168.0.101 --voltage 20
"""

from __future__ import annotations

import signal
import sys
import time
from typing import NoReturn

import click
from loguru import logger

from ao_shaping.drivers.dm.MicroDM import R50Controller
from ao_shaping.utils.cli_helpers import setup_coredumpy

# Hardware limits
HW_VOLTAGE_MIN = -20.0
HW_VOLTAGE_MAX = 120.0
SINGLE_CHANNELS = 50

# Global state for signal handler
_running = True
_ctrl = None


def _signal_handler(signum: int, frame) -> None:
    global _running
    _running = False
    click.echo("\n⏹  收到中断信号, 正在安全关闭...")


def _ping_reachable(ip: str, timeout: float = 2.0) -> bool:
    """ICMP ping 可达性测试。"""
    import subprocess

    param = "-n" if sys.platform.startswith("win") else "-c"
    timeout_arg = (
        str(int(timeout * 1000))
        if sys.platform.startswith("win")
        else str(int(timeout))
    )
    try:
        result = subprocess.run(
            ["ping", param, "1", "-W", timeout_arg, ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 1,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


@click.command("alt-voltage")
@click.option("--ip", required=True, help="Controller IP address (e.g., 192.168.0.101)")
@click.option("--port", default=None, type=int, help="TCP port (default: 10000 + last IP octet)")
@click.option("--voltage", "alt_voltage", required=True, type=float, help="Input voltage for alternation (V)")
@click.option("--freq", "alt_freq", default=1.0, type=float, help="Alternation frequency (Hz, default: 1.0)")
@click.option("--duration", "alt_duration", default=0.0, type=float, help="Duration in seconds (0=until Ctrl+C, default: 0)")
@click.option("--channels", "channel_str", default=None, type=str, help="Channels to alternate (comma-separated, e.g. 0,1,2 or 'all' for all 50)")
@click.option("--ping-first/--no-ping-first", default=True, help="Ping test before connecting (default: True)")
@click.option("--relay-on/--no-relay-on", default=True, help="Auto relay on before starting (default: True)")
@click.option("--debug", "debug", is_flag=True, default=False, help="Enable debug logging")
def run(
    ip: str,
    port: int | None,
    alt_voltage: float,
    alt_freq: float,
    alt_duration: float,
    channel_str: str | None,
    ping_first: bool,
    relay_on: bool,
    debug: bool,
) -> None:
    """交替电压下发工具

    在 0V 和指定电压之间循环交替发送到 R50Power 控制器的指定单元。

    Examples:

        # 全部 50 个通道交替 20V, 1Hz, 持续运行直到 Ctrl+C
        python -m ao_shaping.runners.alt_voltage_runner --ip 192.168.0.101 --voltage 20

        # 通道 0-5 交替 30V, 2Hz, 持续 10 秒
        python -m ao_shaping.runners.alt_voltage_runner --ip 192.168.0.101 --voltage 30 --freq 2.0 --duration 10 --channels 0,1,2,3,4,5

        # 全部通道, 0.5Hz, 跳过 ping 和自动上电
        python -m ao_shaping.runners.alt_voltage_runner --ip 192.168.0.101 --voltage 15 --freq 0.5 --no-ping-first --no-relay-on
    """
    global _running, _ctrl

    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Resolve port
    if port is None:
        try:
            last_octet = int(ip.split(".")[-1])
            port = 10000 + last_octet
        except (ValueError, IndexError):
            click.echo("❌ 无法从 IP 解析端口, 请使用 --port 手动指定")
            sys.exit(1)

    # Validate voltage
    if alt_voltage < HW_VOLTAGE_MIN or alt_voltage > HW_VOLTAGE_MAX:
        click.echo(
            f"❌ 电压 {alt_voltage} V 超出硬件范围 "
            f"[{HW_VOLTAGE_MIN}, {HW_VOLTAGE_MAX}] V"
        )
        sys.exit(1)

    # Validate frequency
    if alt_freq <= 0:
        click.echo("❌ 频率必须大于 0")
        sys.exit(1)

    # Parse channels
    if channel_str is None or channel_str.lower() == "all":
        channels = list(range(SINGLE_CHANNELS))
    else:
        try:
            channels = [int(c.strip()) for c in channel_str.split(",")]
            if not all(0 <= c < SINGLE_CHANNELS for c in channels):
                click.echo(f"❌ 通道号必须在 0-{SINGLE_CHANNELS - 1} 范围内")
                sys.exit(1)
        except (ValueError, IndexError):
            click.echo("❌ 通道格式错误, 请使用逗号分隔 (如 0,1,2)")
            sys.exit(1)

    # Ping check
    if ping_first:
        click.echo(f"📡 Ping 测试 {ip}... ", nl=False)
        if _ping_reachable(ip, timeout=2.0):
            click.echo("✅ 可达")
        else:
            click.echo("❌ 不可达")
            click.echo("  使用 --no-ping-first 跳过 ping 测试")
            sys.exit(1)

    # Connect
    click.echo(f"🔌 连接 {ip}:{port}... ", nl=False)
    ctrl = R50Controller(controller_id=1, ip=ip, port=port)
    if not ctrl.open():
        click.echo("❌ 连接失败")
        sys.exit(1)
    _ctrl = ctrl
    click.echo("✅ 已连接")

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Relay on
    if relay_on:
        click.echo("⚡ 继电器上电... ", nl=False)
        if ctrl.set_relay(True):
            click.echo("✅")
        else:
            click.echo("❌ 失败")
            ctrl.close()
            _ctrl = None
            sys.exit(1)

    # Print run info
    ch_desc = f"{len(channels)} 个通道" if len(channels) == SINGLE_CHANNELS else f"通道 {channel_str}"
    dur_desc = f"{alt_duration:.1f} 秒" if alt_duration > 0 else "持续运行 (Ctrl+C 停止)"
    click.echo("")
    click.echo("=" * 54)
    click.echo("  交替电压下发启动")
    click.echo("=" * 54)
    click.echo(f"  控制器:      {ip}:{port}")
    click.echo(f"  电压范围:    0V ↔ {alt_voltage:.1f}V")
    click.echo(f"  频率:        {alt_freq:.2f} Hz")
    click.echo(f"  周期:        {1.0 / alt_freq:.3f} 秒")
    click.echo(f"  通道:        {ch_desc}")
    click.echo(f"  持续时间:    {dur_desc}")
    click.echo("=" * 54)
    click.echo("")

    # Alternating loop
    t_start = time.time()
    cycle_count = 0
    half_period = 1.0 / (2.0 * alt_freq)
    state = 0  # 0 = sending 0V, 1 = sending input voltage

    try:
        while _running:
            elapsed = time.time() - t_start
            if alt_duration > 0 and elapsed >= alt_duration:
                click.echo(f"\n⏱  达到运行时长 {alt_duration:.1f} 秒")
                break

            # Send voltage to selected channels
            v = 0.0 if state == 0 else alt_voltage
            if len(channels) == SINGLE_CHANNELS:
                ctrl.set_all_channel_voltage(v)
            else:
                for ch in channels:
                    ctrl.set_channel_voltage(ch, v)

            # Progress indicator (show voltage transitions)
            v_label = "0V" if state == 0 else f"{alt_voltage:.1f}V"
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            click.echo(f"  [{elapsed_str}] → {v_label}  (cycle {cycle_count // 2 + 1})")

            state = 1 - state
            cycle_count += 1
            time.sleep(half_period)

    except Exception as e:
        click.echo(f"\n❌ 运行异常: {e}")
        logger.exception("Alt voltage loop error")
    finally:
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


# Define __main__ entry point (NoReturn to satisfy type checkers)
def main() -> NoReturn:
    try:
        run()
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"❌ 运行时错误: {e}")
        logger.exception("Alt voltage runner failed")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    setup_coredumpy()
    main()
