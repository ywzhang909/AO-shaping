"""Micro-DM 逐单元图像采集工具

遍历多个 R50Power Micro-DM 控制器 (按 IP), 对每个通道 (单元) 依次下发指定电压,
并用相机 (MiiCam 或 Daheng) 采集图像, 图像按 "IP-通道号" 命名保存到输出目录,
采集完成后该通道立即归位 (默认 0V), 全部通道处理完后安全关闭控制器
(全部归位 → 继电器下电 → 断开连接)。
处理顺序: 上电 → 下发电压 → 采集 → 归位。

用法:
    # 单控制器, 全部 50 通道 20V, 图像保存到 data/micro_dm_images/<IP>/
    python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 --voltage 20 -o data/micro_dm_images

    # 使用 Daheng 相机采集
    python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 --voltage 20 --camera-type daheng

    # 直接运行脚本文件
    python src/ao_shaping/tools/micro_dm_image_collect.py --ip 192.168.0.101 --voltage 20

    # 多控制器 (逐 IP), 仅采集指定通道, 额外保存 .npy
    python -m ao_shaping.tools.micro_dm_image_collect \
        --ip 192.168.0.101 --ip 192.168.0.102 \
        --voltage 30 --channels 0,1,2,3,4 --save-npy

    # 不指定 IP → 遍历所有控制器 (wiring map / 默认 192.168.0.101-126)
    python -m ao_shaping.tools.micro_dm_image_collect --voltage 20 -o data/micro_dm_images

    # 每通道采集 3 张
    python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 --voltage 20 --n-frames 3

    # 自定义归位电压 / 曝光时间 / 采样参数
    python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 \
        --voltage 40 --home-voltage 0.0 --exposure-ms 10 --n-sample 3 --settle-time 0.8
"""

from __future__ import annotations

import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import click
import numpy as np

from loguru import logger

from ao_shaping.config import DEVICES
from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager
from ao_shaping.drivers.ccd.daheng import DahengCamManager
from ao_shaping.drivers.dm.MicroDM import (
    DEFAULT_IPS,
    MAX_CHANNELS,
    R50Controller,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
    WIRING_MAP_PATH,
    WiringMap,
)
from ao_shaping.utils.cli_helpers import setup_coredumpy
from ao_shaping.utils.network import controller_tcp_port, ping_reachable

# 全局运行标志 (信号处理器修改)
_running = True

# 电压下发方式静态变量:
#   True  → 使用 set_all_voltage_array (命令 0x09, 一次性下发全部 50 通道,
#           其余通道固定为 home_voltage, 速度最快)
#   False → 使用 set_channel_voltage (逐通道下发, 仅修改目标通道)
USE_SET_ALL_VOLTAGE_ARRAY = True


def _signal_handler(signum: int, frame: object | None) -> None:
    """信号处理器: 收到 SIGINT/SIGTERM 后置运行标志为 False 并提示安全关闭。"""
    global _running
    _running = False
    click.echo("\n⏹  收到中断信号, 正在安全关闭...")


def _get_miicam_camera(cam_id: int, exposure_ms: float, bit_depth: int = 8) -> CameraStreamManager:
    """创建并打开 MiiCam 相机实例。

    相机是本工具必需的硬件, 打开失败时抛异常, 由调用方决定退出。

    Args:
        cam_id: 相机设备 ID
        exposure_ms: 曝光时间 (ms)
        bit_depth: 输出位深 (8 或 16)

    Returns:
        已打开的 CameraStreamManager 实例
    """
    try:
        cam = CameraStreamManager(
            cam_id=cam_id, exposure_time_ms=exposure_ms, bit_depth=bit_depth
        )
        cam.open()
        return cam
    except Exception as e:
        logger.error("MiiCam相机初始化失败: {}", e)
        raise


def _get_daheng_camera(cam_id: int, exposure_ms: float) -> DahengCamManager:
    """创建并打开 Daheng 相机实例。

    相机是本工具必需的硬件, 打开失败时抛异常, 由调用方决定退出。

    Args:
        cam_id: 相机设备 ID
        exposure_ms: 曝光时间 (ms)

    Returns:
        已打开的 DahengCamManager 实例
    """
    try:
        cam = DahengCamManager(cam_id=cam_id, exposure_time_ms=exposure_ms)
        cam.open()
        return cam
    except Exception as e:
        logger.error("Daheng相机初始化失败: {}", e)
        raise


def _parse_channels(channel_str: str) -> list[int]:
    """解析通道列表: 'all' → 全部 50 通道, 否则逗号分隔的 0-49 通道号。

    Args:
        channel_str: 通道列表字符串 (如 "0,1,2" 或 "all")

    Returns:
        解析后的通道号列表

    Raises:
        SystemExit: 通道格式错误或超出范围时退出 (退出码 1)
    """
    if channel_str.strip().lower() == "all":
        return list(range(MAX_CHANNELS))
    try:
        channels = [int(c.strip()) for c in channel_str.split(",")]
    except ValueError:
        click.echo("❌ 通道格式错误, 请使用逗号分隔 (如 0,1,2)")
        sys.exit(1)
    if not all(0 <= c < MAX_CHANNELS for c in channels):
        click.echo(f"❌ 通道号必须在 0-{MAX_CHANNELS - 1} 范围内")
        sys.exit(1)
    return channels


def _resolve_ips(user_ips: tuple[str, ...]) -> list[str]:
    """解析待采集的控制器 IP 列表: 用户指定则用之, 否则遍历所有控制器 (wiring map → 默认 IP 段)。

    Args:
        user_ips: 用户通过 --ip 指定的 IP 元组 (可能为空)

    Returns:
        待采集的控制器 IP 列表
    """
    if user_ips:
        return list(user_ips)
    try:
        wm = WiringMap.from_file(WIRING_MAP_PATH)
        if wm is None:
            raise ValueError("wiring map 文件缺失或无效")
        ips = wm.unique_ips
        if not ips:
            logger.warning("wiring map 未包含有效控制器, 回退默认 IP 段")
            return list(DEFAULT_IPS)
        logger.info("未指定 IP, 从 wiring map 加载 {} 个控制器: {}", len(ips), ips)
        return ips
    except Exception as e:
        logger.warning("wiring map 加载失败 ({}), 回退默认 IP 段", e)
        return list(DEFAULT_IPS)


def _resolve_ip_port(ip_str: str, port: int | None) -> tuple[str, int]:
    """校验控制器 IP 格式并解析端口 (公共函数)。

    所有需要对控制器 IP 进行校验/端口解析的调用点都应使用本函数,
    避免各处在格式判断与默认端口规则 (10000 + IP 末段) 上不一致。

    Args:
        ip_str: 控制器 IP 地址 (如 "192.168.0.101")
        port: 用户显式指定的端口; None 时按 "10000 + IP 末段" 解析

    Returns:
        (ip_str, port) 二元组

    Raises:
        ValueError: IP 格式非法或超出范围时抛出
    """
    parts = ip_str.split(".")
    if len(parts) != 4:
        raise ValueError(f"IP 格式无效: {ip_str}")
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"IP 格式无效: {ip_str}") from None
    if not all(0 <= o <= 255 for o in octets):
        raise ValueError(f"IP 地址超出范围: {ip_str}")
    resolved = controller_tcp_port(ip_str) if port is None else port
    if resolved is None:
        raise ValueError(f"无法解析端口: {ip_str}")
    return ip_str, resolved


def _save_frame(
    ip_dir: Path,
    ip: str,
    channel: int,
    frame: np.ndarray,
    frame_index: int | None = None,
    save_npy: bool = False,
) -> str:
    """保存单通道图像为 PNG (可选 .npy), 并打印图像统计信息。

    文件名规则:
        - frame_index 为 None 时: ``{ip}-{channel:03d}.png`` (单张, 与旧版兼容)
        - frame_index 非 None 时: ``{ip}-{channel:03d}-{frame_index:03d}.png`` (多帧)

    Args:
        ip_dir: 该 IP 对应的输出子目录
        ip: 控制器 IP (用于文件名)
        channel: 通道号
        frame: 相机采集的图像数组
        frame_index: 帧序号 (None 表示单张模式, 文件名不含序号)
        save_npy: 是否额外保存 .npy 原始数组

    Returns:
        保存的 PNG 文件名
    """
    from PIL import Image

    if frame_index is not None:
        stem = f"{ip}-{channel:03d}-{frame_index:03d}"
    else:
        stem = f"{ip}-{channel:03d}"
    filename = f"{stem}.png"
    png_path = ip_dir / filename

    if frame.dtype == np.uint16:
        img = Image.fromarray(frame, mode="I;16")
    else:
        img = Image.fromarray(frame, mode="L")
    img.save(png_path)

    if save_npy:
        np.save(ip_dir / f"{stem}.npy", frame)

    max_val = int(frame.max())
    mean_val = float(frame.mean())
    logger.info(
        "通道 {} 图像已保存: {}, shape={}, dtype={}, max={}, mean={:.1f}",
        channel,
        filename,
        frame.shape,
        frame.dtype,
        max_val,
        mean_val,
    )

    # 过曝/过暗检测
    if frame.dtype == np.uint16:
        dtype_max = 65535
        dark_threshold = 100
    else:
        dtype_max = 255
        dark_threshold = 10

    if max_val > int(dtype_max * 0.95):
        logger.warning(
            "通道 {} 画面过曝! max={}, mean={:.1f}, 建议降低曝光时间或增益",
            channel,
            max_val,
            mean_val,
        )
    elif max_val < dark_threshold:
        logger.warning(
            "通道 {} 画面过暗! max={}, mean={:.1f}, 建议增加曝光时间或增益",
            channel,
            max_val,
            mean_val,
        )

    return filename


def _voltage_array(ch: int | None, voltage: float, home_voltage: float) -> list[float]:
    """构建 50 通道电压数组: 目标通道为 voltage, 其余通道为 home_voltage。

    Args:
        ch: 目标通道号; None 时全部通道均为 home_voltage (归位用)
        voltage: 目标通道电压 (V)
        home_voltage: 其余通道电压 (V)

    Returns:
        长度为 MAX_CHANNELS 的电压列表
    """
    volts = [home_voltage] * MAX_CHANNELS
    if ch is not None:
        volts[ch] = voltage
    return volts


def _send_voltage(
    ctrl: R50Controller, ch: int, voltage: float, home_voltage: float
) -> bool:
    """下发单通道电压, 按 USE_SET_ALL_VOLTAGE_ARRAY 选择下发方式。

    True 时使用 :meth:`R50Controller.set_all_voltage_array` (0x09 命令,
    全部 50 通道一次性下发, 其余通道固定为 home_voltage);
    False 时使用 :meth:`R50Controller.set_channel_voltage` 逐通道下发。
    """
    if USE_SET_ALL_VOLTAGE_ARRAY:
        return ctrl.set_all_voltage_array(_voltage_array(ch, voltage, home_voltage))
    return ctrl.set_channel_voltage(ch, voltage)


def _home_channel(ctrl: R50Controller, ch: int, home_voltage: float) -> bool:
    """单通道归位, 按 USE_SET_ALL_VOLTAGE_ARRAY 选择下发方式。"""
    if USE_SET_ALL_VOLTAGE_ARRAY:
        return ctrl.set_all_voltage_array(
            _voltage_array(None, home_voltage, home_voltage)
        )
    return ctrl.set_channel_voltage(ch, home_voltage)


def _collect_for_ip(
    ctrl: R50Controller,
    cam: CameraStreamManager,
    ip: str,
    channels: list[int],
    voltage: float,
    home_voltage: float,
    ip_dir: Path,
    n_frames: int,
    n_sample: int,
    skip_first: bool,
    settle_time: float,
    save_npy: bool,
) -> list[str]:
    """对单个控制器执行逐通道电压下发 + 图像采集, 每通道采集后归位。

    Args:
        ctrl: 已连接的 R50Controller 实例
        cam: 已打开的 MiiCam 相机
        ip: 控制器 IP
        channels: 待采集的通道号列表
        voltage: 采集时下发电压 (V)
        home_voltage: 归位电压 (V)
        ip_dir: 该 IP 对应的输出子目录
        n_frames: 每通道采集图像张数 (<1 时按 1 处理)
        n_sample: 每帧平均采样数
        skip_first: 是否跳过首帧
        settle_time: 电压下发后等待时间 (s)
        save_npy: 是否额外保存 .npy

    Returns:
        成功保存的 PNG 文件名列表
    """
    saved_files: list[str] = []

    for ch in channels:
        if not _running:
            logger.info("收到中断信号, 停止通道循环")
            break

        click.echo(f"  ▶ 通道 {ch:02d}: 下发 {voltage:g}V ... ", nl=False)
        if not _send_voltage(ctrl, ch, voltage, home_voltage):
            click.echo("❌ 电压下发失败, 跳过采集")
            logger.warning("通道 {} 电压下发失败, 跳过采集", ch)
            continue
        click.echo("✅")

        time.sleep(settle_time)

        try:
            frame_count = n_frames if n_frames >= 1 else 1
            for frame_idx in range(1, frame_count + 1):
                frame = cam.get_numpy_image(n_sample=n_sample, skip_first=skip_first)
                idx = None if frame_count == 1 else frame_idx
                filename = _save_frame(ip_dir, ip, ch, frame, idx, save_npy)
                saved_files.append(filename)
        except Exception as e:
            logger.warning("通道 {} 图像采集失败: {}", ch, e)
        finally:
            # 归位
            try:
                _home_channel(ctrl, ch, home_voltage)
            except Exception as e:
                logger.warning("通道 {} 归位失败: {}", ch, e)

        if not _running:
            logger.info("收到中断信号, 停止通道循环")
            break

    return saved_files


def _safe_shutdown(ctrl: R50Controller, home_voltage: float) -> None:
    """安全关闭单个控制器: 全部通道归位 → 继电器下电 → 关闭连接。

    实际关闭序列由 :meth:`R50Controller.power_off_and_close` 驱动,
    本函数只负责用户反馈输出。
    """
    click.echo("")
    click.echo("⏹  安全关闭中...")
    try:
        ctrl.power_off_and_close(home_voltage)
        click.echo(f"  ✅ 已下发 {home_voltage:g}V 归位, 继电器已下电, 连接已关闭")
    except Exception as e:
        click.echo(f"  ⚠️  安全关闭失败: {e}")
    click.echo("🏁 控制器已退出")


@click.command("micro-dm-collect")
@click.option(
    "--ip",
    multiple=True,
    help="R50Power 控制器 IP 地址, 可多次指定; 不指定则遍历所有控制器",
)
@click.option("--port", default=None, type=int, help="TCP端口 (默认: 10000 + IP末段)")
@click.option("--voltage", required=True, type=float, help="下发电压 V (手动输入)")
@click.option("--home-voltage", default=0.0, type=float, help="归位电压 V (default: 0.0)")
@click.option("--channels", default="all", type=str, help="通道列表 逗号分隔 或 'all' 全部50通道")
@click.option(
    "--output",
    "-o",
    default="data/micro_dm_images",
    help="输出目录",
)
@click.option(
    "--camera-type",
    default="miicam",
    type=click.Choice(["miicam", "daheng"], case_sensitive=False),
    help="相机类型: miicam (默认) 或 daheng",
)
@click.option("--cam-id", default=None, type=int, help="相机ID (默认: config far_cam_id)")
@click.option("--exposure-ms", default=20.0, type=float, help="曝光时间 ms")
@click.option("--bit-depth", default=8, type=click.IntRange(8, 16), help="MiiCam输出位深 8或16 (仅miicam有效)")
@click.option("--n-sample", default=1, type=int, help="每帧平均采样数")
@click.option("--n-frames", default=1, type=int, help="每通道采集图像张数 (default: 1)")
@click.option("--skip-first/--no-skip-first", default=True, help="跳过首帧")
@click.option("--settle-time", default=0.5, type=float, help="电压下发后等待时间 s (default: 0.5)")
@click.option("--ping-first/--no-ping-first", default=True, help="连接前先 ping 测试")
@click.option("--save-npy", is_flag=True, default=False, help="额外保存 .npy 原始数组")
@click.option("--debug", is_flag=True, default=False, help="启用DEBUG日志")
def run(
    ip: tuple[str, ...],
    port: int | None,
    voltage: float,
    home_voltage: float,
    channels: str,
    output: str,
    camera_type: str,
    cam_id: int | None,
    exposure_ms: float,
    bit_depth: int,
    n_sample: int,
    n_frames: int,
    skip_first: bool,
    settle_time: float,
    ping_first: bool,
    save_npy: bool,
    debug: bool,
) -> None:
    """Micro-DM 逐单元图像采集工具

    遍历多个 R50Power 控制器, 对每个通道依次下发电压并用相机 (MiiCam 或 Daheng) 采集图像,
    每通道采集后立即归位, 全部通道完成后安全关闭控制器。

    Examples:

        # 单控制器, 全部 50 通道 20V, 使用 MiiCam 相机 (默认)
        python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 --voltage 20

        # 使用 Daheng 相机采集
        python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 --voltage 20 \\
            --camera-type daheng

        # 不指定 IP → 遍历所有控制器 (wiring map / 默认 IP 段)
        python -m ao_shaping.tools.micro_dm_image_collect --voltage 20

        # 每通道采集 3 张图像
        python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 \\
            --voltage 20 --n-frames 3

        # 多控制器, 仅采集指定通道, 额外保存 .npy
        python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 --ip 192.168.0.102 \\
            --voltage 30 --channels 0,1,2,3,4 --save-npy

        # 自定义曝光与采样参数
        python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 \\
            --voltage 40 --exposure-ms 10 --n-sample 3 --settle-time 0.8
    """
    global _running

    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # 校验电压范围
    if voltage < VOLTAGE_MIN or voltage > VOLTAGE_MAX:
        click.echo(
            f"❌ 电压 {voltage} V 超出硬件范围 "
            f"[{VOLTAGE_MIN}, {VOLTAGE_MAX}] V"
        )
        sys.exit(1)

    # 解析通道列表
    ch_list = _parse_channels(channels)

    # 解析待采集的控制器 IP 列表 (未指定 → 遍历所有控制器)
    ip_list = _resolve_ips(ip)
    if not ip_list:
        click.echo("❌ 未指定 IP 且无法解析控制器列表, 请使用 --ip 手动指定")
        sys.exit(1)

    # 每通道采集图像张数 (至少 1 张)
    frame_count = n_frames if n_frames >= 1 else 1

    # 校验 IP 格式并解析端口 (公共函数, 输入错误直接退出)
    ip_ports: list[tuple[str, int]] = []
    for ip_addr in ip_list:
        try:
            ip_ports.append(_resolve_ip_port(ip_addr, port))
        except ValueError as e:
            click.echo(f"❌ {e}")
            sys.exit(1)

    # 打开相机 (必需, 失败即退出)
    if cam_id is None:
        cam_id = DEVICES.far_cam_id
    try:
        if camera_type == "daheng":
            click.echo(
                f"📷 打开 Daheng 相机 ID={cam_id}, 曝光={exposure_ms}ms... ",
                nl=False,
            )
            cam = _get_daheng_camera(cam_id, exposure_ms)
            click.echo("✅")
            logger.info("Daheng相机已连接: ID={}", cam_id)
        else:
            click.echo(
                f"📷 打开 MiiCam 相机 ID={cam_id}, 曝光={exposure_ms}ms, 位深={bit_depth}... ",
                nl=False,
            )
            cam = _get_miicam_camera(cam_id, exposure_ms, bit_depth)
            click.echo("✅")
            logger.info("MiiCam相机已连接: ID={}, bit_depth={}", cam_id, bit_depth)
    except Exception as e:
        cam_name = "Daheng" if camera_type == "daheng" else "MiiCam"
        click.echo(f"❌ {cam_name} 相机打开失败: {e}")
        logger.error("{}相机打开失败: {}", cam_name, e)
        sys.exit(1)

    # 创建基础输出目录
    base_dir = Path(output)
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("输出目录: {}", base_dir)

    ok_ips: list[str] = []
    failed_ips: list[str] = []

    try:
        for index, (ip_addr, resolved_port) in enumerate(ip_ports):
            controller_id = index + 1
            click.echo("")
            click.echo("=" * 54)
            click.echo(f"  控制器 {controller_id}/{len(ip_ports)}: {ip_addr}:{resolved_port}")
            click.echo("=" * 54)
            logger.info("开始处理控制器 {}: {}:{}", controller_id, ip_addr, resolved_port)

            if not _running:
                click.echo("⏹  已收到中断信号, 跳过剩余控制器")
                break

            # 可选 ping 测试
            if ping_first:
                click.echo(f"📡 Ping 测试 {ip_addr}... ", nl=False)
                if ping_reachable(ip_addr, timeout=2.0):
                    click.echo("✅ 可达")
                else:
                    click.echo("❌ 不可达, 跳过该控制器")
                    logger.warning("控制器 {} ping 不可达, 跳过", ip_addr)
                    failed_ips.append(ip_addr)
                    continue

            ctrl: R50Controller | None = None
            try:
                # 连接控制器
                click.echo(f"🔌 连接 {ip_addr}:{resolved_port}... ", nl=False)
                ctrl = R50Controller(controller_id=controller_id, ip=ip_addr, port=resolved_port)
                if not ctrl.open():
                    click.echo("❌ 连接失败, 跳过该控制器")
                    logger.warning("控制器 {}:{} 连接失败, 跳过", ip_addr, resolved_port)
                    failed_ips.append(ip_addr)
                    continue
                click.echo("✅ 已连接")
                logger.info("控制器已连接: {}:{}", ip_addr, resolved_port)

                # 注册信号处理器用于优雅关闭
                signal.signal(signal.SIGINT, _signal_handler)
                signal.signal(signal.SIGTERM, _signal_handler)

                # 继电器上电
                click.echo("⚡ 继电器上电... ", nl=False)
                if ctrl.set_relay(True):
                    click.echo("✅ 已上电, 开始下发电压")
                else:
                    click.echo("❌ 上电失败, 跳过该控制器")
                    logger.warning("控制器 {} 继电器上电失败, 跳过", ip_addr)
                    failed_ips.append(ip_addr)
                    continue

                # 每 IP 一个子目录
                ip_dir = base_dir / ip_addr
                ip_dir.mkdir(parents=True, exist_ok=True)

                # 逐通道采集
                click.echo(
                    f"  待采集通道: {len(ch_list)} 个, 电压 {voltage:g}V, "
                    f"归位 {home_voltage:g}V, 等待 {settle_time:g}s"
                )
                saved_files = _collect_for_ip(
                    ctrl=ctrl,
                    cam=cam,
                    ip=ip_addr,
                    channels=ch_list,
                    voltage=voltage,
                    home_voltage=home_voltage,
                    ip_dir=ip_dir,
                    n_frames=n_frames,
                    n_sample=n_sample,
                    skip_first=skip_first,
                    settle_time=settle_time,
                    save_npy=save_npy,
                )

                # 写入每 IP 元数据
                metadata = {
                    "ip": ip_addr,
                    "port": resolved_port,
                    "voltage": voltage,
                    "home_voltage": home_voltage,
                    "channels": ch_list,
                    "timestamp": datetime.now().isoformat(),
                    "camera_type": camera_type,
                    "cam_id": cam_id,
                    "exposure_ms": exposure_ms,
                    "bit_depth": bit_depth,
                    "n_frames": frame_count,
                    "n_sample": n_sample,
                    "settle_time": settle_time,
                    "saved_count": len(saved_files),
                    "saved_files": saved_files,
                }
                meta_path = ip_dir / "metadata.json"
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                logger.info("元数据已保存: {}", meta_path)

                click.echo(
                    f"📷 {ip_addr} 采集完成: {len(saved_files)}/{len(ch_list)} 张图像已保存到 {ip_dir}"
                )
                ok_ips.append(ip_addr)
            finally:
                if ctrl is not None and ctrl.is_connected:
                    _safe_shutdown(ctrl, home_voltage)
    finally:
        # 关闭相机 (保证任何错误路径下相机都会被关闭)
        try:
            cam.close()
            cam_name = "Daheng" if camera_type == "daheng" else "MiiCam"
            click.echo(f"📷 {cam_name} 相机已关闭")
            logger.info("{}相机已关闭", cam_name)
        except Exception as e:
            cam_name = "Daheng" if camera_type == "daheng" else "MiiCam"
            click.echo(f"⚠️  {cam_name} 相机关闭失败: {e}")
            logger.warning("{}相机关闭失败: {}", cam_name, e)

    # 最终汇总
    click.echo("")
    click.echo("=" * 54)
    click.echo("  采集完成")
    click.echo(
        f"  控制器数量:  {len(ip_ports)} (成功 {len(ok_ips)}, 跳过 {len(failed_ips)})"
    )
    click.echo(f"  输出目录:    {base_dir.resolve()}")
    click.echo("=" * 54)
    click.echo("🏁 全部完成")


# 定义 __main__ 入口点 (NoReturn 以满足类型检查器)
def main() -> NoReturn:
    try:
        run()
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"❌ 运行时错误: {e}")
        logger.exception("Micro DM image collect failed")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    setup_coredumpy()
    main()
