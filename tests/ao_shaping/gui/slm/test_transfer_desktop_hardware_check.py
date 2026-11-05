"""迁移学习桌面页硬件自检的离线契约（全部 monkeypatch，不触硬件）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ao_shaping.gui.slm.transfer_desktop import hardware_check as hc


def _patch_all(monkeypatch, *, daheng_list=None, daheng_exc=None,
               miicam_list=None, miicam_exc=None, slm_device=None,
               slm_exc=None, exposure=5.0):
    if daheng_exc is not None:
        def _daheng():
            raise daheng_exc
        monkeypatch.setattr(hc.DahengCamera, "get_cam_list", staticmethod(_daheng))
    else:
        monkeypatch.setattr(hc.DahengCamera, "get_cam_list",
                            staticmethod(lambda: list(daheng_list or [])))

    if miicam_exc is not None:
        def _miicam():
            raise miicam_exc
        monkeypatch.setattr(hc.MIICamera, "get_cam_list", staticmethod(_miicam))
    else:
        monkeypatch.setattr(hc.MIICamera, "get_cam_list",
                            staticmethod(lambda: list(miicam_list or [])))

    if slm_exc is not None:
        def _probe(_n):
            raise slm_exc
        monkeypatch.setattr(hc, "probe_slm", _probe)
    else:
        monkeypatch.setattr(hc, "probe_slm", lambda _n: slm_device)

    monkeypatch.setattr(hc, "configured_daheng_exposure_limit", lambda: exposure)


def test_all_online_ok(monkeypatch):
    _patch_all(
        monkeypatch,
        daheng_list=[SimpleNamespace(), SimpleNamespace()],
        miicam_list=[SimpleNamespace()],
        slm_device={"slm_number": 1, "serial": "SN-9", "in_use": False},
        exposure=10.0,
    )
    report = hc.check_hardware("daheng", 0, 1)

    assert report.ok is True
    assert all(item.level == "ok" for item in report.items)
    assert "通过" in report.summary()


def test_daheng_failure_still_checks_others(monkeypatch):
    calls = {"miicam": 0, "probe": 0}

    def _miicam():
        calls["miicam"] += 1
        return [SimpleNamespace()]

    def _probe(_n):
        calls["probe"] += 1
        return {"slm_number": 1, "serial": "SN", "in_use": False}

    monkeypatch.setattr(hc.DahengCamera, "get_cam_list",
                        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("sdk down"))))
    monkeypatch.setattr(hc.MIICamera, "get_cam_list", staticmethod(_miicam))
    monkeypatch.setattr(hc, "probe_slm", _probe)
    monkeypatch.setattr(hc, "configured_daheng_exposure_limit", lambda: 1.0)

    report = hc.check_hardware("daheng", 0, 1)

    assert calls["miicam"] == 1
    assert calls["probe"] == 1
    assert report.ok is False
    daheng_item = report.items[0]
    assert daheng_item.required is True
    assert daheng_item.level == "fail"
    assert "RuntimeError" in daheng_item.detail


def test_slm_in_use_fail(monkeypatch):
    _patch_all(
        monkeypatch,
        daheng_list=[SimpleNamespace()],
        miicam_list=[],
        slm_device={"slm_number": 1, "serial": None, "in_use": True},
        exposure=1.0,
    )
    report = hc.check_hardware("daheng", 0, 1)

    assert report.ok is False
    slm_item = [i for i in report.items if i.label == "SLM 探测"][0]
    assert slm_item.level == "fail"


def test_slm_none_fail(monkeypatch):
    _patch_all(
        monkeypatch,
        daheng_list=[SimpleNamespace()],
        miicam_list=[],
        slm_device=None,
        exposure=1.0,
    )
    report = hc.check_hardware("daheng", 0, 1)

    assert report.ok is False
    slm_item = [i for i in report.items if i.label == "SLM 探测"][0]
    assert slm_item.level == "fail"


def test_exposure_zero_warn_and_ok(monkeypatch):
    _patch_all(
        monkeypatch,
        daheng_list=[SimpleNamespace()],
        miicam_list=[],
        slm_device={"slm_number": 1, "serial": "SN", "in_use": False},
        exposure=0.0,
    )
    report = hc.check_hardware("daheng", 0, 1)

    assert report.ok is True
    exp_item = [i for i in report.items if i.label == "曝光安全上限"][0]
    assert exp_item.level == "warn"
    assert "未配置" in exp_item.detail
    assert "AO_DAHENG_MAX_EXPOSURE_MS" in exp_item.detail


def test_miicam_missing_does_not_fail(monkeypatch):
    _patch_all(
        monkeypatch,
        daheng_list=[SimpleNamespace()],
        miicam_list=[],
        slm_device={"slm_number": 1, "serial": "SN", "in_use": False},
        exposure=1.0,
    )
    report = hc.check_hardware("daheng", 0, 1)

    assert report.ok is True
    miicam_item = [i for i in report.items if i.label == "MiiCam 枚举"][0]
    assert miicam_item.required is False
    assert miicam_item.level == "warn"
    assert "仅自检；Transfer 运行仍只接受 Daheng" in miicam_item.detail


def test_resolve_not_called(monkeypatch):
    called = {"resolve": 0}

    def _resolve(*_a, **_k):
        called["resolve"] += 1
        return 1.0

    import ao_shaping.utils.image.exposure_control as ec
    monkeypatch.setattr(ec, "resolve_daheng_exposure_limit", _resolve, raising=False)
    # also expose on the import point if present
    monkeypatch.setattr(hc, "resolve_daheng_exposure_limit", _resolve, raising=False)

    _patch_all(
        monkeypatch,
        daheng_list=[SimpleNamespace()],
        miicam_list=[],
        slm_device={"slm_number": 1, "serial": "SN", "in_use": False},
        exposure=1.0,
    )
    report = hc.check_hardware("daheng", 0, 1)

    assert report.ok is True
    assert called["resolve"] == 0