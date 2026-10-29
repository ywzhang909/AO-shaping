"""离线测试: 相机像元尺寸 (pixel size) 的型号级单一来源。

不碰硬件也不碰 SDK:
* ``BaseCamera`` / MiiCam / 仿真后端只读类属性;
* 大恒的型号解析走临时配置目录 (``CCD_CONFIG_DIR``), 型号名直接注入实例字段;
* 交互录入通过注入的 ``prompt`` 替身验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.ccd.pixel_size import (
    KNOWN_MODEL_PIXEL_SIZE_UM,
    ensure_model_pixel_size_um,
    load_model_pixel_size_um,
    model_pixel_size_config_path,
    resolve_model_pixel_size_um,
    store_model_pixel_size_um,
)


@pytest.fixture()
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把相机配置根目录指向临时目录 (型号配置随之隔离)。"""
    monkeypatch.setenv("CCD_CONFIG_DIR", str(tmp_path))
    return tmp_path


class _BareCamera(BaseCamera):
    """只用来读类属性 / 调用 require_pixel_size_um 的最小后端。"""

    def __init__(self, pixel_size_um: float = 0.0) -> None:
        super().__init__()
        if pixel_size_um:
            self.pixel_size_um = pixel_size_um

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        return None

    def initialize(self) -> None:  # pragma: no cover
        return None

    def open(self) -> None:  # pragma: no cover
        return None

    def close(self) -> None:  # pragma: no cover
        return None

    def reset_exposure_time(self, time_ms: float) -> float:  # pragma: no cover
        return float(time_ms)

    def reset_window(self, center, size):  # pragma: no cover
        return size, center

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False):  # pragma: no cover
        raise NotImplementedError

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:  # pragma: no cover
        return bool(enable)

    def get_auto_exposure_state(self) -> dict:  # pragma: no cover
        return {}

    def set_auto_exposure_range(self, *args, **kwargs) -> None:  # pragma: no cover
        return None

    @staticmethod
    def get_cam_list():  # pragma: no cover
        return []


class TestBaseCameraDefault:
    def test_class_attribute_defaults_to_zero(self) -> None:
        assert BaseCamera.pixel_size_um == 0.0
        assert _BareCamera().pixel_size_um == 0.0

    def test_require_raises_with_backend_name_and_remedy(self) -> None:
        with pytest.raises(CameraError) as excinfo:
            _BareCamera().require_pixel_size_um()
        message = str(excinfo.value)
        assert "_BareCamera" in message
        assert "pixel_size_um" in message
        assert "ao_shaping.drivers.ccd.pixel_size" in message

    def test_require_returns_value_when_known(self) -> None:
        assert _BareCamera(pixel_size_um=3.45).require_pixel_size_um() == 3.45


class TestHardcodedBackends:
    def test_miicam_is_hardcoded(self) -> None:
        from ao_shaping.drivers.ccd.miicam.driver import MIICamera

        assert MIICamera.pixel_size_um == pytest.approx(2.9)

    def test_sim_pib_ccd_uses_twin_scale(self) -> None:
        from ao_shaping.drivers.sim.slm_pib_sim import CCD_PIXEL_UM, SimPibCCD

        assert SimPibCCD.pixel_size_um == pytest.approx(CCD_PIXEL_UM)

    def test_simulated_ccd_accepts_explicit_value(self) -> None:
        from ao_shaping.drivers.sim.ccd.simulated_ccd import SimulatedCCD

        assert SimulatedCCD().pixel_size_um == 0.0
        assert SimulatedCCD(pixel_size_um=5.5).pixel_size_um == pytest.approx(5.5)


class TestModelConfig:
    def test_config_path_follows_env(self, config_dir: Path) -> None:
        assert model_pixel_size_config_path() == config_dir / "ccd" / "models.json"

    def test_store_then_load_and_resolve(self, config_dir: Path) -> None:
        path = store_model_pixel_size_um("MODEL-X", 3.45)
        assert path == config_dir / "ccd" / "models.json"
        assert load_model_pixel_size_um("MODEL-X") == pytest.approx(3.45)
        assert resolve_model_pixel_size_um("MODEL-X") == pytest.approx(3.45)

    def test_store_merges_other_models(self, config_dir: Path) -> None:
        store_model_pixel_size_um("MODEL-A", 2.2)
        store_model_pixel_size_um("MODEL-B", 4.5)
        raw = json.loads(model_pixel_size_config_path().read_text(encoding="utf-8"))
        assert raw == {
            "MODEL-A": {"pixel_size_um": 2.2},
            "MODEL-B": {"pixel_size_um": 4.5},
        }

    def test_store_rejects_non_positive(self, config_dir: Path) -> None:
        with pytest.raises(ValueError):
            store_model_pixel_size_um("MODEL-X", 0.0)
        with pytest.raises(ValueError):
            store_model_pixel_size_um("", 2.2)

    def test_unregistered_model_raises_with_command(self, config_dir: Path) -> None:
        with pytest.raises(CameraError) as excinfo:
            resolve_model_pixel_size_um("UNKNOWN-MODEL")
        message = str(excinfo.value)
        assert "UNKNOWN-MODEL" in message
        assert "--store" in message
        assert str(model_pixel_size_config_path()) in message

    def test_known_table_is_the_fallback(self, config_dir: Path) -> None:
        (model, value), *_ = KNOWN_MODEL_PIXEL_SIZE_UM.items()
        # 已知型号表按前缀匹配, 因此带后缀的完整型号名也能命中。
        assert resolve_model_pixel_size_um(f"{model}-23GM") == pytest.approx(value)

    def test_config_wins_over_known_table(self, config_dir: Path) -> None:
        model = next(iter(KNOWN_MODEL_PIXEL_SIZE_UM))
        store_model_pixel_size_um(model, 9.9)
        assert resolve_model_pixel_size_um(model) == pytest.approx(9.9)

    def test_corrupt_config_falls_back_instead_of_crashing(self, config_dir: Path) -> None:
        path = model_pixel_size_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert load_model_pixel_size_um("MODEL-X") is None
        model = next(iter(KNOWN_MODEL_PIXEL_SIZE_UM))
        assert resolve_model_pixel_size_um(model) == pytest.approx(
            KNOWN_MODEL_PIXEL_SIZE_UM[model]
        )

    def test_bom_prefixed_config_is_still_read(self, config_dir: Path) -> None:
        """Windows 手写配置常带 BOM: 带 BOM 时**型号配置仍优先于已知型号表**。

        用 ``utf-8`` 读会让 ``json.load`` 报 "Unexpected UTF-8 BOM", 整份配置被当作
        未登记 —— 权威来源静默失效、回落成已知型号表里的旧值, 且只有一条 warning。
        """
        path = model_pixel_size_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        model = next(iter(KNOWN_MODEL_PIXEL_SIZE_UM))
        # UTF-8 BOM + 与已知型号表**不同**的值, 才能区分「读到了配置」和「回落」。
        path.write_bytes(
            b"\xef\xbb\xbf" + json.dumps({model: {"pixel_size_um": 9.9}}).encode("utf-8")
        )
        assert load_model_pixel_size_um(model) == pytest.approx(9.9)
        assert resolve_model_pixel_size_um(model) == pytest.approx(9.9)


class TestEnsurePrompt:
    def test_prompt_value_is_stored_and_reused(self, config_dir: Path) -> None:
        asked: list[str] = []

        def _prompt(question: str) -> str:
            asked.append(question)
            return " 3.45 "

        assert ensure_model_pixel_size_um("MODEL-X", prompt=_prompt) == pytest.approx(3.45)
        assert load_model_pixel_size_um("MODEL-X") == pytest.approx(3.45)
        assert asked and "MODEL-X" in asked[0]

        # 第二次不再提问: 配置已经命中。
        assert ensure_model_pixel_size_um(
            "MODEL-X", prompt=lambda _q: pytest.fail("不应再次提问")
        ) == pytest.approx(3.45)

    @pytest.mark.parametrize("answer", ["", "abc", "0", "-2"])
    def test_invalid_answer_raises(self, config_dir: Path, answer: str) -> None:
        with pytest.raises(CameraError):
            ensure_model_pixel_size_um("MODEL-X", prompt=lambda _q: answer)
        assert load_model_pixel_size_um("MODEL-X") is None


class TestCameraPixelSizeFor:
    """``camera_pixel_size_um_for`` —— 命令行 ``--camera-pixel-um 0`` 的解析入口。"""

    def test_miicam_and_sim_come_from_drivers(self, config_dir: Path) -> None:
        from ao_shaping.drivers.ccd.pixel_size import camera_pixel_size_um_for

        assert camera_pixel_size_um_for("miicam") == pytest.approx(2.9)
        assert camera_pixel_size_um_for("sim") == pytest.approx(2.2)

    def test_daheng_uses_model_config(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.drivers.ccd import pixel_size as ps

        monkeypatch.setattr(ps, "probe_daheng_models", lambda: ["MER2-507-23GM"])
        store_model_pixel_size_um("MER2-507-23GM", 2.25)
        assert ps.camera_pixel_size_um_for("daheng") == pytest.approx(2.25)

    def test_daheng_unregistered_prompts_and_stores(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.drivers.ccd import pixel_size as ps

        monkeypatch.setattr(ps, "probe_daheng_models", lambda: ["BRAND-NEW-MODEL"])
        value = ps.camera_pixel_size_um_for("daheng", prompt=lambda _q: "3.45")
        assert value == pytest.approx(3.45)
        assert load_model_pixel_size_um("BRAND-NEW-MODEL") == pytest.approx(3.45)

    def test_daheng_non_interactive_raises(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.drivers.ccd import pixel_size as ps

        class _NonTty:
            def isatty(self) -> bool:
                return False

        monkeypatch.setattr(ps, "probe_daheng_models", lambda: ["BRAND-NEW-MODEL"])
        monkeypatch.setattr(ps.sys, "stdin", _NonTty())
        with pytest.raises(CameraError):
            ps.camera_pixel_size_um_for("daheng")
        assert ps.camera_pixel_size_um_for("daheng", strict=False) == 0.0

    def test_missing_camera_raises(self, config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from ao_shaping.drivers.ccd import pixel_size as ps

        monkeypatch.setattr(ps, "probe_daheng_models", lambda: [])
        with pytest.raises(CameraError):
            ps.camera_pixel_size_um_for("daheng")
        assert ps.camera_pixel_size_um_for("daheng", strict=False) == 0.0

    def test_open_camera_instance_wins(self, config_dir: Path) -> None:
        from ao_shaping.drivers.ccd import pixel_size as ps

        cam = _BareCamera(pixel_size_um=4.4)
        assert ps.camera_pixel_size_um_for("daheng", camera=cam) == pytest.approx(4.4)
        with pytest.raises(CameraError):
            ps.camera_pixel_size_um_for("daheng", camera=_BareCamera())

    def test_unknown_backend(self, config_dir: Path) -> None:
        from ao_shaping.drivers.ccd.pixel_size import camera_pixel_size_um_for

        with pytest.raises(CameraError):
            camera_pixel_size_um_for("ffmpeg")
        assert camera_pixel_size_um_for("ffmpeg", strict=False) == 0.0


class TestDahengModelResolution:
    def test_unopened_camera_reports_unknown_model(self, config_dir: Path) -> None:
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        cam = DahengCamera()
        assert cam.model_name is None
        with pytest.raises(CameraError) as excinfo:
            _ = cam.pixel_size_um
        assert "尚未打开" in str(excinfo.value)

    def test_registered_model_resolves_from_config(self, config_dir: Path) -> None:
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        store_model_pixel_size_um("MER2-507-23GM", 2.2)
        cam = DahengCamera()
        cam._model = "MER2-507-23GM"
        assert cam.pixel_size_um == pytest.approx(2.2)

    def test_unregistered_model_raises(self, config_dir: Path) -> None:
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        cam = DahengCamera()
        cam._model = "SOME-NEW-MODEL"
        with pytest.raises(CameraError):
            _ = cam.pixel_size_um

    def test_store_updates_config_and_instance(self, config_dir: Path) -> None:
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        cam = DahengCamera()
        cam._model = "SOME-NEW-MODEL"
        cam.store_pixel_size_um(4.5)
        assert cam.pixel_size_um == pytest.approx(4.5)
        assert load_model_pixel_size_um("SOME-NEW-MODEL") == pytest.approx(4.5)

    def test_ensure_prompts_once(self, config_dir: Path) -> None:
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        cam = DahengCamera()
        cam._model = "SOME-NEW-MODEL"
        assert cam.ensure_pixel_size_um(prompt=lambda _q: "6.5") == pytest.approx(6.5)
        assert load_model_pixel_size_um("SOME-NEW-MODEL") == pytest.approx(6.5)


class _FakeDeviceManager:
    """替身 ``gx.DeviceManager()``: 枚举阶段失败。"""

    def __init__(self, *, dev_info=None, enum_error=None) -> None:
        self._dev_info = dev_info or []
        self._enum_error = enum_error

    def update_device_list(self):
        if self._enum_error is not None:
            raise self._enum_error
        return len(self._dev_info), self._dev_info


class _ExplodingDeviceManager:
    """替身 ``gx.DeviceManager``: 构造阶段就失败 (GigE 驱动缺失)。"""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __call__(self):
        raise self._error


def _install_fake_gxipy(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    """把 ``gxipy`` 换成一个只有 ``DeviceManager`` 的假模块。

    ``factory`` 是 ``gx.DeviceManager`` 本身 (零参可调用); 只替换
    ``sys.modules['gxipy']``, 不碰真实 SDK 的安装状态 ——
    ``probe_daheng_models`` 的函数内 ``import gxipy`` 因此拿到替身。
    """
    import sys
    import types

    fake = types.ModuleType("gxipy")
    fake.DeviceManager = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gxipy", fake)


class TestDahengEnumerationFailures:
    """枚举失败 ≠ 没有相机, 但对调用方后果相同 (拿不到型号), 必须归一成空列表。"""

    def test_enumeration_error_returns_empty_list(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.drivers.ccd.pixel_size import probe_daheng_models

        _install_fake_gxipy(
            monkeypatch, lambda: _FakeDeviceManager(enum_error=RuntimeError("网卡未就绪"))
        )
        assert probe_daheng_models() == []

    def test_construction_error_returns_empty_list(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.drivers.ccd.pixel_size import probe_daheng_models

        _install_fake_gxipy(monkeypatch, _ExplodingDeviceManager(OSError("Galaxy 驱动未安装")))
        assert probe_daheng_models() == []

    def test_enumeration_error_is_logged(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """宽捕获必须留日志 —— 否则"探不到相机"会变成不可诊断的静默 0。"""
        from loguru import logger

        from ao_shaping.drivers.ccd.pixel_size import probe_daheng_models

        _install_fake_gxipy(
            monkeypatch, lambda: _FakeDeviceManager(enum_error=RuntimeError("网卡未就绪"))
        )
        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            probe_daheng_models()
        finally:
            logger.remove(sink_id)
        assert any("大恒相机枚举失败" in m for m in messages)

    def test_enumeration_failure_resolves_to_zero_when_not_strict(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.drivers.ccd.pixel_size import camera_pixel_size_um_for

        _install_fake_gxipy(
            monkeypatch, lambda: _FakeDeviceManager(enum_error=RuntimeError("网卡未就绪"))
        )
        assert camera_pixel_size_um_for("daheng", 0, strict=False) == 0.0

    def test_enumeration_failure_raises_when_strict(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ao_shaping.drivers.ccd.pixel_size import camera_pixel_size_um_for

        _install_fake_gxipy(
            monkeypatch, lambda: _FakeDeviceManager(enum_error=RuntimeError("网卡未就绪"))
        )
        with pytest.raises(CameraError) as excinfo:
            camera_pixel_size_um_for("daheng", 0)
        assert "--camera-pixel-um" in str(excinfo.value)

    def test_listed_model_resolves_from_config(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """枚举成功时仍按型号解析 (这里锁定未改动的正向路径)。"""
        from ao_shaping.drivers.ccd.pixel_size import (
            camera_pixel_size_um_for,
            probe_daheng_models,
        )

        store_model_pixel_size_um("MER2-507-23GM", 2.2)
        _install_fake_gxipy(
            monkeypatch,
            lambda: _FakeDeviceManager(
                dev_info=[{"model": ("MER2-507-23GM",)}, {"model": "MER2-507-23GM"}]
            ),
        )
        assert probe_daheng_models() == ["MER2-507-23GM"]
        assert camera_pixel_size_um_for("daheng", 0) == pytest.approx(2.2)
