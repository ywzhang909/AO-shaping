"""硬件测试: 大恒相机在**实验系统**上的满量程必须跟生效位深, 不跟帧 dtype (issue #246)。

🔴 **运行前必须确认没有别的会话/进程占用这台相机** (GUI、采集脚本、另一个 pytest、
`diagnose_light_level.py` 之类)。大恒 GigE 设备被独占时第二次打开会失败或拿到脏帧;
本文件不做「在飞保护」探测, 占用检查是**运行者**的责任 (见
``docs/dev/hardware_discipline.md``)。本文件**只碰 CCD** —— 不动 SLM / DM / 激光 /
滤波轮, 也不会写盘改相机配置 (不调 ``save_config()``)。

用法 (相机在线、且确认无人占用时)::

    python -m pytest tests/ao_shaping/drivers/ccd/test_full_scale_hardware.py --hardware -v

保守取像口径: 曝光 2 ms 起、每档 ``n_sample=2``; 唯一会往上推曝光的是饱和回归那一条,
阶梯上限 **32 ms** (远低于设备 10000 ms 量程, 也低于自动曝光 routinely 用的 1000 ms),
达到饱和即停, 退出时把曝光写回进入时的值。

在线检查一律走 ``daheng/discovery.enumerate_devices()`` —— 官方 SDK 的子网枚举,
三态归一。**不 ping、不按固定 IP 做 TCP 探测**: 大恒在线与否只由枚举结果决定
(相机地址是 DHCP/链路本地可变的, 按 IP 探测会把「换了地址的在线相机」判成离线,
也会把「同网段的别的设备」判成相机)。三态分别处理:

- ``sdk_ready=False``            → skip (SDK 没装/路径错, **不是**相机离线);
- ``sdk_ready=True, online=False`` → ``skip("大恒相机离线")``;
- ``online=True``                → 才打开相机跑断言。

离线口径见 ``test_full_scale_contract.py`` (全后端假 SDK) 与
``daheng/test_bit_depth_full_scale.py`` (大恒 12-bit 回归)。
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from loguru import logger

from ao_shaping.drivers.ccd import common
from ao_shaping.drivers.ccd.base import AveragedFrame
from ao_shaping.drivers.ccd.daheng import discovery

pytestmark = pytest.mark.hardware

#: 保守取像口径: 短曝光 + 少帧数。
_EXPOSURE_MS = 2.0
_N_SAMPLE = 2

#: 饱和回归的曝光阶梯 (ms)。达到饱和即停; 上限 32 ms 远低于设备量程。
_SATURATION_LADDER_MS = (2.0, 8.0, 32.0)


def _load_driver() -> Any:
    """拿到一个 SDK 就绪的大恒驱动模块。

    ``driver.py`` 在**模块导入时**就 try/except 掉 ``import gxipy``; 若它在
    ``discovery`` 注入 SDK 路径**之前**已被别的测试 import 过, ``gx`` 会停在
    ``None``。先 bootstrap 再 (必要时) reload, 才能区分「SDK 真没装」与
    「只是 import 顺序不巧」。
    """
    discovery.bootstrap_sdk()
    module = importlib.import_module("ao_shaping.drivers.ccd.daheng.driver")
    if module.gx is None:
        module = importlib.reload(module)
    return module


@pytest.fixture(scope="module")
def cam():
    """打开一台在线的大恒相机; 离线/无 SDK 就 skip, 退出时恢复位深并 ``close()``。"""
    found = discovery.enumerate_devices()
    if not found.sdk_ready:
        pytest.skip(f"大恒 SDK 不可用 (不是相机离线): {found.error}")
    if not found.online or found.count < 1:
        pytest.skip("大恒相机离线")

    driver = _load_driver()
    if driver.gx is None or driver.DahengCamera is None:
        pytest.skip(f"大恒 SDK 不可用 (不是相机离线): {driver._SDK_IMPORT_ERROR}")

    camera = driver.DahengCamera(cam_id=0, exposure_time_ms=_EXPOSURE_MS)
    camera.initialize()
    original_bits = camera.bit_depth
    original_ms = camera.get_exposure_time_ms()
    logger.info(
        "大恒相机已打开: {} 台在线, 进入时位深={}, 曝光={:.3f} ms",
        found.count,
        original_bits,
        original_ms,
    )
    try:
        yield camera
    finally:
        try:
            if original_bits is not None and camera.bit_depth != original_bits:
                camera.set_bit_depth(original_bits)
            camera.reset_exposure_time(original_ms)
        except Exception as exc:  # 收尾失败不许掩盖测试结果
            logger.warning(f"恢复相机状态失败 (位深={original_bits}): {exc}")
        camera.close()


def test_bit_depth_readback_is_the_device_truth(cam) -> None:
    """``set_bit_depth`` 返回的必须是**设备回读**的生效值, 不是请求值。

    gxipy 的枚举特征 ``set`` 越界/不支持时不抛异常、只 print 噪声然后保留旧值,
    所以「请求 12 就当成 12」是不成立的 —— 而位深是满量程的唯一来源。
    """
    assert cam.supported_bit_depths == (8, 12, 16)
    assert cam.set_bit_depth(12) == 12
    assert cam.bit_depth == 12


def test_twelve_bit_frame_is_uint16_but_its_full_scale_is_4095(cam) -> None:
    """#246 的现场: MONO12 帧装在 uint16 容器里, 满量程必须是 4095 而不是 65535。"""
    cam.set_bit_depth(12)

    single = cam.get_numpy_image(n_sample=1, skip_first=True)
    assert single.dtype == np.uint16, "12-bit 帧的容器 dtype 应当是 uint16 (缺陷入口)"
    # 单帧路径没有 AveragedFrame 标注 ⇒ 判据只能经相机实例取满量程。
    assert cam.effective_full_scale(single) == 4095.0
    assert cam._full_scale(single.dtype) == 4095.0
    assert common.full_scale(single, cam) == 4095.0

    averaged = cam.get_numpy_image(n_sample=_N_SAMPLE, skip_first=True)
    assert isinstance(averaged, AveragedFrame)
    assert averaged.dtype == np.float32
    assert averaged.full_scale == 4095.0
    # 谁产帧谁标注: 帧自带标注后, 没有相机上下文也不会退回 dtype 猜。
    assert common.full_scale(averaged) == 4095.0


def test_eight_bit_frame_full_scale_is_255(cam) -> None:
    """另一端的对照: 切回 8-bit 后满量程必须跟着变回 255 (位深是设备状态, 不是一次性参数)。"""
    cam.set_bit_depth(8)

    single = cam.get_numpy_image(n_sample=1, skip_first=True)
    assert single.dtype == np.uint8
    assert cam.effective_full_scale(single) == 255.0

    averaged = cam.get_numpy_image(n_sample=_N_SAMPLE, skip_first=True)
    assert averaged.full_scale == 255.0


def test_saturated_twelve_bit_frame_reads_as_bright_not_dark(cam) -> None:
    """实验系统复现 #246 的危害: 饱和帧折算到 0-255 必须是 ~255, 不是 15.9。

    实测危害 (2026-11-16): 峰值 4095 被按 65535 折成 ``4095*255/65535 = 15.9`` 灰度,
    自动曝光把饱和帧误判成「过暗」, 一路把曝光推到 1000 ms 上限。这里在实验系统上
    同时钉住正例 (≈255) 与对照 (dtype 猜法确实低一个量级), 两者差 16 倍。
    """
    cam.set_bit_depth(12)
    entry_ms = cam.get_exposure_time_ms()
    frame: np.ndarray | None = None
    try:
        for ms in _SATURATION_LADDER_MS:
            cam.reset_exposure_time(ms)
            candidate = cam.get_numpy_image(n_sample=1, skip_first=True)
            if common.is_saturated(candidate, cam):
                frame = candidate
                break
        if frame is None:
            pytest.skip(
                f"保守曝光阶梯 {_SATURATION_LADDER_MS} ms 内未达到饱和 "
                "(光路可能无光/激光未开), 无法验证饱和折算"
            )

        peak = float(np.max(frame))
        assert peak == pytest.approx(4095.0), "12-bit 饱和帧的削平电平应当是 4095"
        assert common.is_saturated(frame, cam) is True

        fixed = cam._auto_exposure_peak(frame)
        assert fixed == pytest.approx(255.0, abs=1.0)

        wrong = common.normalize_peak_to_gray(frame, full_scale=65535.0)
        assert wrong == pytest.approx(15.9, abs=0.05)
        assert fixed / wrong == pytest.approx(65535.0 / 4095.0, rel=1e-3)
    finally:
        cam.reset_exposure_time(entry_ms)
