r"""离线锁定 :meth:`MIICamera._detect_raw_format` 的 fourcc 解码。

SDK ``get_RawFormat()`` 返回的 fourcc 是 32 位整数 (``MAKEFOURCC`` 小端拼装),
驱动必须先把整数解成 4 个 ASCII 字符再与格式名比较。历史实现拿整数直接和
字符串比较, 三个分支全部为死路, 真机上 Bayer 相机被静默落到兜底分支。

假相机 + 假日志, 不接相机、不加载 SDK 设备。
"""
from __future__ import annotations

import pytest


def _make_camera(miicam_driver, monkeypatch, fourcc, *, bit_depth: int = 14):
    """构造未打开的 MIICamera, 挂上返回指定 fourcc 的假相机与假日志。"""
    cam = miicam_driver.MIICamera(cam_id=0, bit_depth=bit_depth)
    cam.cam = _FakeCam(fourcc)
    logs: list[str] = []

    class _FakeLogger:
        def info(self, msg: str) -> None:
            logs.append(msg)

    monkeypatch.setattr(miicam_driver, "logger", _FakeLogger())
    return cam, logs


class _FakeCam:
    """只实现 ``_detect_raw_format`` 用到的 ``get_RawFormat``。"""

    def __init__(self, fourcc) -> None:
        self._fourcc = fourcc

    def get_RawFormat(self):
        return self._fourcc, 0


@pytest.fixture
def miicam_driver():
    from ao_shaping.drivers.ccd.miicam import driver as miicam_driver

    return miicam_driver


def test_bayer_rggb_int_decoded(miicam_driver, monkeypatch):
    """真机读数 1111967570 = 0x42474752 = fourcc 'RGGB' (Bayer, 读为灰度)。"""
    cam, logs = _make_camera(miicam_driver, monkeypatch, 0x42474752)
    cam._detect_raw_format()
    assert cam._pixel_format == "MONO16"
    assert any("Bayer RGGB" in msg for msg in logs)


@pytest.mark.parametrize("fourcc", ["BGGR", "GRBG", "GBRG"])
def test_bayer_others_decoded(miicam_driver, monkeypatch, fourcc):
    int_fourcc = int.from_bytes(fourcc.encode("ascii"), "little")
    cam, logs = _make_camera(miicam_driver, monkeypatch, int_fourcc)
    cam._detect_raw_format()
    assert cam._pixel_format == "MONO16"
    assert any(f"Bayer {fourcc}" in msg for msg in logs)


@pytest.mark.parametrize("fourcc", ["YUYV", "VUYY", "UYVY"])
def test_yuv422_int_decoded(miicam_driver, monkeypatch, fourcc):
    int_fourcc = int.from_bytes(fourcc.encode("ascii"), "little")
    cam, logs = _make_camera(miicam_driver, monkeypatch, int_fourcc)
    cam._detect_raw_format()
    assert cam._pixel_format == "YUV422"
    assert any("YUV422" in msg for msg in logs)


def test_mono_yyyy_int_decoded(miicam_driver, monkeypatch):
    """SDK 文档口径: 单色 = MAKEFOURCC('Y','Y','Y','Y')。"""
    cam, logs = _make_camera(miicam_driver, monkeypatch, 0x59595959)
    cam._detect_raw_format()
    assert cam._pixel_format == "MONO16"
    assert any("MONO14 format" in msg for msg in logs)


def test_mono_ymon_int_decoded(miicam_driver, monkeypatch):
    """MAKEFOURCC('Y','M','o','n') = 'YMon' 也是合法单色 fourcc。"""
    int_fourcc = int.from_bytes(b"YMon", "little")
    cam, logs = _make_camera(miicam_driver, monkeypatch, int_fourcc)
    cam._detect_raw_format()
    assert cam._pixel_format == "MONO16"
    assert any("MONO14 format" in msg for msg in logs)


def test_mono_ymono_string_kept(miicam_driver, monkeypatch):
    """历史字面量 'YMono' (5 字符, 整数解码不可达) 仍保留兼容。"""
    cam, logs = _make_camera(miicam_driver, monkeypatch, "YMono")
    cam._detect_raw_format()
    assert cam._pixel_format == "MONO16"
    assert any("MONO14 format" in msg for msg in logs)


def test_unknown_fourcc_falls_back(miicam_driver, monkeypatch):
    int_fourcc = int.from_bytes(b"RGB8", "little")
    cam, logs = _make_camera(miicam_driver, monkeypatch, int_fourcc)
    cam._detect_raw_format()
    assert cam._pixel_format == "MONO16"
    assert any("RGB8" in msg and "MONO14" in msg for msg in logs)
