"""大恒在线检查统一枚举漏斗 (``discovery``) 的离线契约。

钉住四件事 (全部离线, 相机枚举一律用假 ``gxipy`` / 假路径, 不碰真 SDK、不碰网络):

* 三态归一: **SDK 未装** (``sdk_ready=False``) ≠ **枚举失败** (``sdk_ready=True,
  online=False``) ≠ **在线 N 台** (``online=True, count=N``);
* 枚举失败必须**留日志** —— 否则「探不到相机」会变成不可诊断的静默归一;
* ``_inject_sdk_path`` 的存在性判据: ``AO_SHAPING_SDK_PATH`` 与仓库 ``libs``
  两条路径, 合法绑定才注入 ``sys.path``;
* ``DahengCamera.get_cam_list`` 的委托契约: 返回原始设备信息列表 (``list``),
  SDK 缺失/枚举失败时为空列表而非抛异常。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ao_shaping.drivers.ccd.daheng import discovery


class _FakeDeviceManager:
    """替身 ``gx.DeviceManager()``: 枚举阶段成功/失败可控。"""

    def __init__(self, *, dev_info=None, enum_error=None) -> None:
        self._dev_info = dev_info or []
        self._enum_error = enum_error

    def update_device_list(self):
        if self._enum_error is not None:
            raise self._enum_error
        return len(self._dev_info), self._dev_info


def _install_fake_gxipy(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    """把 ``gxipy`` 换成一个只有 ``DeviceManager`` 的假模块。

    同时把 ``_inject_sdk_path`` 关掉 —— 假模块已在 ``sys.modules`` 里, 导入必然
    成功; 关掉注入是防止万一导入回落到仓库 ``libs`` 里的真实绑定, 让测试结果
    依赖环境。
    """
    monkeypatch.setitem(sys.modules, "gxipy", SimpleNamespace(DeviceManager=factory))
    monkeypatch.setattr(discovery, "_inject_sdk_path", lambda: False)


def _sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """表达「gxipy 不可导入」: ``sys.modules["gxipy"]=None`` 使导入抛 ImportError。

    不 patch ``_load_gxipy`` —— 这样测的是漏斗的真实导入路径; 关掉注入防止
    回落到仓库 ``libs`` 里的真实绑定。
    """
    monkeypatch.setitem(sys.modules, "gxipy", None)
    monkeypatch.setattr(discovery, "_inject_sdk_path", lambda: False)


def _device(sn: str, model: str = "MER2-507-23GM") -> dict:
    return {"index": 0, "sn": sn, "model_name": model}


class TestEnumerateDevices:
    def test_sdk_missing_is_not_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SDK 未装 ≠ 相机离线: ``sdk_ready=False`` 单独表达。"""
        _sdk_missing(monkeypatch)

        found = discovery.enumerate_devices()

        assert found.sdk_ready is False
        assert found.online is False
        assert found.count == 0
        assert found.devices == []
        assert "gxipy 不可用" in (found.error or "")

    def test_enumeration_error_normalises_to_offline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK 在但枚举抛错 → 归一成 offline, 且带 error 摘要。"""
        _install_fake_gxipy(
            monkeypatch, lambda: _FakeDeviceManager(enum_error=RuntimeError("网卡未就绪"))
        )

        found = discovery.enumerate_devices()

        assert found.sdk_ready is True
        assert found.online is False
        assert found.count == 0
        assert "枚举失败" in (found.error or "")

    def test_enumeration_error_is_logged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """宽捕获必须留日志 —— 否则这句归一就是静默的。"""
        from loguru import logger

        _install_fake_gxipy(
            monkeypatch, lambda: _FakeDeviceManager(enum_error=RuntimeError("网卡未就绪"))
        )
        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            discovery.enumerate_devices()
        finally:
            logger.remove(sink_id)
        assert any("大恒相机枚举失败" in m for m in messages)

    def test_no_device_is_offline_without_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK 在、子网干净 → offline, 且不算错误。"""
        _install_fake_gxipy(monkeypatch, lambda: _FakeDeviceManager(dev_info=[]))

        found = discovery.enumerate_devices()

        assert found.sdk_ready is True
        assert found.online is False
        assert found.count == 0
        assert found.error is None

    def test_online_returns_raw_device_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """在线时返回原始 SDK 设备信息对象, 不转换、不丢字段。"""
        devs = [_device("SN-A"), _device("SN-B")]
        _install_fake_gxipy(monkeypatch, lambda: _FakeDeviceManager(dev_info=devs))

        found = discovery.enumerate_devices()

        assert found.sdk_ready is True
        assert found.online is True
        assert found.count == 2
        assert found.devices == devs
        assert [d["sn"] for d in found.devices] == ["SN-A", "SN-B"]

    def test_summary_text_three_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """一行中文结论对三态分别给不同措辞。"""
        _sdk_missing(monkeypatch)
        assert "SDK 不可用" in discovery.summary_text()

        _install_fake_gxipy(monkeypatch, lambda: _FakeDeviceManager(dev_info=[]))
        assert "离线" in discovery.summary_text()

        _install_fake_gxipy(
            monkeypatch, lambda: _FakeDeviceManager(dev_info=[_device("SN-A")])
        )
        assert "在线" in discovery.summary_text()


class TestInjectSdkPath:
    def test_env_path_with_valid_binding_injects(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        sdk_dir = tmp_path / "sdk"
        (sdk_dir / "gxipy").mkdir(parents=True)
        (sdk_dir / "gxipy" / "__init__.py").write_text("", encoding="utf-8")
        monkeypatch.setenv("AO_SHAPING_SDK_PATH", str(sdk_dir))

        assert discovery._inject_sdk_path() is True
        assert str(sdk_dir) in sys.path

    def test_env_path_without_binding_does_not_inject(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("AO_SHAPING_SDK_PATH", str(empty))

        assert discovery._inject_sdk_path() is False

    def test_missing_env_falls_back_to_bundled_libs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """无环境变量时回落到仓库 ``libs``。

        ``libs/gxipy`` 是厂商 SDK 绑定, 被 ``.gitignore`` 的 ``gxipy*`` 排除
        —— 它只存在于开发机, 全新 checkout (CI 容器) 上没有, 此时回落分支
        无从存在, 跳过而不是钉死一个仓库并不保证的事实。
        """
        monkeypatch.delenv("AO_SHAPING_SDK_PATH", raising=False)
        if not (discovery._LIBS_DIR / "gxipy" / "__init__.py").is_file():
            pytest.skip("bundled gxipy binding is gitignored; absent on fresh checkouts")
        assert discovery._inject_sdk_path() is True
        assert str(discovery._LIBS_DIR) in sys.path


class TestGetCamListDelegation:
    def test_get_cam_list_returns_raw_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``DahengCamera.get_cam_list`` 委托漏斗, 契约仍是原始设备信息列表。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        devs = [_device("SN-A"), _device("SN-B")]
        _install_fake_gxipy(monkeypatch, lambda: _FakeDeviceManager(dev_info=devs))

        assert DahengCamera.get_cam_list() == devs

    def test_get_cam_list_empty_when_sdk_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK 缺失时返回空列表, 不抛异常 (与既有空列表契约一致)。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        _sdk_missing(monkeypatch)

        assert DahengCamera.get_cam_list() == []

    def test_get_cam_list_empty_when_enumeration_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        _install_fake_gxipy(
            monkeypatch, lambda: _FakeDeviceManager(enum_error=RuntimeError("网卡未就绪"))
        )

        assert DahengCamera.get_cam_list() == []
