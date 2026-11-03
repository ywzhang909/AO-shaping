"""Santec SLM 探针的离线契约（无真实 SDK）。"""

from __future__ import annotations

import sys
import time
import types

import pytest

from ao_shaping.drivers.slm.santec import probe


def _install_fake_sdk(monkeypatch, *, open_ret=0, read_ret=0, serial=b"SN-1",
                      raise_on=None, close_ret=0, sleep_s=0.0):
    calls: dict[str, int] = {"open": 0, "readsd": 0, "close": 0}

    def _open(slm_num):
        calls["open"] += 1
        if raise_on == "open":
            raise RuntimeError("open boom")
        if sleep_s:
            time.sleep(sleep_s)
        return open_ret

    def _read(slm_num, device_id):
        calls["readsd"] += 1
        if raise_on == "read":
            raise RuntimeError("read boom")
        device_id.value = serial
        return read_ret

    def _close(slm_num):
        calls["close"] += 1
        if raise_on == "close":
            raise RuntimeError("close boom")
        return close_ret

    fake = types.SimpleNamespace(
        SLM_Ctrl_Open=_open,
        SLM_Ctrl_ReadSD=_read,
        SLM_Ctrl_Close=_close,
    )
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.slm.santec._slm_win", fake)
    return calls


def test_probe_open_readsd_close(monkeypatch):
    calls = _install_fake_sdk(monkeypatch, serial=b"SN-42")
    result = probe.probe_slm(3)

    assert result is not None
    assert result["slm_number"] == 3
    assert result["serial"] == "SN-42"
    assert result["in_use"] is False
    assert calls == {"open": 1, "readsd": 1, "close": 1}


def test_probe_positive_return_code_means_in_use(monkeypatch):
    calls = _install_fake_sdk(monkeypatch, open_ret=2)
    result = probe.probe_slm(1)

    assert result is not None
    assert result["in_use"] is True
    assert result["serial"] is None
    assert calls["close"] == 1


def test_probe_negative_return_code_means_no_device(monkeypatch):
    calls = _install_fake_sdk(monkeypatch, open_ret=-10002)
    result = probe.probe_slm(1)

    assert result is None
    assert calls["close"] == 1


def test_probe_read_failure_leaves_serial_none(monkeypatch):
    calls = _install_fake_sdk(monkeypatch, read_ret=1, serial=b"")
    result = probe.probe_slm(1)

    assert result is not None
    assert result["serial"] is None
    assert result["in_use"] is False
    assert calls["close"] == 1


def test_probe_exception_still_closes(monkeypatch):
    calls = _install_fake_sdk(monkeypatch, raise_on="open")
    result = probe.probe_slm(1)

    assert result is None
    assert calls["close"] == 1


def test_probe_timeout_returns_none(monkeypatch):
    _install_fake_sdk(monkeypatch, sleep_s=0.2)
    result = probe.probe_slm(1, timeout_s=0.05)

    assert result is None