# -*- coding: utf-8 -*-
"""§12.16.5 门禁: ``Santec._write_to_memory`` 连续性守卫 (离线, 不碰硬件).

根因 (§12.16.7): 第 2660 行 ``ctypes.memmove`` 按物理内存布局复制字节; F-contiguous
的 H 光栅 (strides=(2,2400)) 会被按列主序送上屏, 面板按行主序解释 → 图案转置
(周期 128px→~80px, 轴由列变行, 与 S1d 指纹一致)。守卫在 memmove 前
``np.ascontiguousarray``, 使送屏字节流 == 逻辑 C 序 (即 ``digest()`` 的定义)。

本文件独立于冻结用例: 仅用冻结 capture 脚本的纯 builder 物化 H/B, 只断言
driver 守卫后的实际送屏字节。不改动冻结测试 / 冻结 capture 脚本。

不触碰硬件: 用 ``sys.modules`` stub 替换 ``_slm_win``, 令 ``_ensure_open`` 为
no-op, 并用 spy 捕获 ``SLM_Ctrl_WriteMI`` 收到的 ``dat`` 字节。
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import sys
import types

import numpy as np
import pytest

from ao_shaping.drivers.slm.santec import Santec

CAPTURE = importlib.import_module(
    "scripts.slm.s1e_same_bytes_probe_capture_20261124"
)

# §12.16.7 冻结摘要 (与冻结用例逐字节一致, 独立转写以免耦合)
H_LOGICAL_SHA = (
    "7276d9da3b2a5ea536d56ea944adcfdcb38c98ad1dd01c1433d1b2219ac8218e"
)
H_RAW_BUFFER_SHA = (
    "b98be9a6696cd94dbb3fa966e22a7f86e52f52b16ae252d21f7aa07af4079c3a"
)
B_LOGICAL_SHA = (
    "c8d937cedb294339fa385f9ffa65cebd0ff3471ff62009cef42d7637bf590a73"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _install_sdk_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """用空模块替换 ``_slm_win``, 使 ``Santec`` 构造不需要真实 SLM SDK DLL."""
    stub = types.ModuleType("ao_shaping.drivers.slm.santec._slm_win")
    monkeypatch.setitem(
        sys.modules, "ao_shaping.drivers.slm.santec._slm_win", stub
    )


class _WriteSpy:
    """捕获 ``SLM_Ctrl_WriteMI`` 实际收到的 ``dat`` 字节流."""

    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def SLM_Ctrl_WriteMI(
        self,
        slm_number: int,
        memory_number: int,
        width: int,
        height: int,
        memory_mode: int,
        dat: object,
    ) -> int:
        self.payloads.append(
            ctypes.string_at(dat, int(width) * int(height) * 2)
        )
        return 0  # SLM_OK


def _make_slm(monkeypatch: pytest.MonkeyPatch) -> tuple[Santec, _WriteSpy]:
    _install_sdk_stub(monkeypatch)
    slm = Santec(slm_number=1)
    monkeypatch.setattr(slm, "_ensure_open", lambda: None)
    spy = _WriteSpy()
    slm._slm = spy  # type: ignore[assignment]  # 替换 SDK 句柄
    return slm, spy


def _h_gray() -> np.ndarray:
    """冻结 H: ``grating_phase`` axis=1, F-contiguous (2,2400)."""
    gray = CAPTURE.build_gray("s1e_a1")
    assert gray.dtype == np.uint16
    return gray


def _b_gray() -> np.ndarray:
    """冻结 B: ``grating_phase`` axis=0, C-contiguous."""
    gray = CAPTURE.build_gray("s1e_b1")
    assert gray.dtype == np.uint16
    return gray


def test_h_gray_logical_and_raw_digests_are_frozen() -> None:
    """冻结数组布局与摘要与 §12.16.7 一致 (守卫预期的两个字节流)."""
    h = _h_gray()
    assert h.flags.f_contiguous and not h.flags.c_contiguous
    # 逻辑 C 序 == 修复后应上的屏
    assert _sha(np.ascontiguousarray(h).tobytes()) == H_LOGICAL_SHA
    # 裸物理 buffer == 修复前 (错误) 的送屏字节
    assert _sha(h.tobytes(order="A")) == H_RAW_BUFFER_SHA
    assert H_LOGICAL_SHA != H_RAW_BUFFER_SHA


def test_write_to_memory_sends_logical_c_order_for_f_contiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """守卫生效: F-contiguous H 的送屏字节 == 逻辑 C 序, 而非裸 F 序 buffer."""
    slm, spy = _make_slm(monkeypatch)
    h = _h_gray()

    slm._write_to_memory(h, 1)

    assert len(spy.payloads) == 1
    sent = spy.payloads[0]
    assert sent == np.ascontiguousarray(h).tobytes()
    assert _sha(sent) == H_LOGICAL_SHA
    assert _sha(sent) != H_RAW_BUFFER_SHA  # 回归: 修复前正是这个值

    # 缓存必须与实际上屏一致 (C 连续, 逻辑值不变)
    cached = slm._memory_phase_cache[1]
    assert cached.flags.c_contiguous
    assert np.array_equal(cached, h)


def test_write_to_memory_keeps_c_contiguous_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C-contiguous 输入 (B) 行为零改动: 送屏字节仍是原 buffer."""
    slm, spy = _make_slm(monkeypatch)
    b = _b_gray()
    assert b.flags.c_contiguous

    slm._write_to_memory(b, 1)

    assert spy.payloads[0] == b.tobytes()
    assert _sha(spy.payloads[0]) == B_LOGICAL_SHA
