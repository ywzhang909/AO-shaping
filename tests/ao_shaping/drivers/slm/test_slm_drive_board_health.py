"""驱动板 (Drive board) 健康检查与身份来源测试。

2026-10-22 实测确认的静默失效: Santec 控制器上**驱动板不应答、只剩选板应答**
时, 驱动会静默回退到选板 ID 当设备身份, 于是整轮实验在面板根本不调制的状态
下跑完, 数据全无效却没有任何报错。

本模块用假 SDK 复现三种 SDK 假象, 断言:
  - ``get_serial_number()`` 的返回值保持向后兼容 (仍返回选板 ID);
  - ``serial_source()`` 如实指出身份来源;
  - 回退时有 **warning 级**日志;
  - ``require_drive_board()`` 在驱动板不应答时抛 ``SLMDriveBoardError``。
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

import ao_shaping.drivers.slm.santec.driver as driver_module
from ao_shaping.drivers.slm import (
    DriveBoardStatus,
    Santec,
    SantecError,
    SLMDriveBoardError,
)

DRIVE_ID = "22030108"
OPTION_ID = "23020027"

#: 任意非 SLM_OK 的错误码 (模拟 USB 链路错误)
SDK_ERROR = -10032


@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> Generator[None, None, None]:
    """Ensure the mocked SDK stays in place for every test in this module.

    The driver imports ``_slm_win`` **lazily** (inside ``Santec.__init__``), so
    patching ``sys.modules`` for the duration of the test is enough — and it
    leaves the real SDK binding untouched for the rest of the session.
    """
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm.santec._slm_win": MagicMock()},
        clear=False,
    ):
        yield


@dataclass
class FakeSdk:
    """模拟 Santec SDK 的三路板 ID 读取。

    每个字段是 ``(返回码, 字符串)``; ``None`` 表示该路调用直接失败。
    """

    sdo: tuple[int, str, str] | None = None
    sd: tuple[int, str] | None = None
    so: tuple[int, str] | None = None

    def SLM_Ctrl_ReadSDO(self, _number: int, drive_buf: Any, option_buf: Any) -> int:
        if self.sdo is None:
            return SDK_ERROR
        ret, drive, option = self.sdo
        drive_buf.value = drive.encode()
        option_buf.value = option.encode()
        return ret

    def SLM_Ctrl_ReadSD(self, _number: int, buf: Any) -> int:
        if self.sd is None:
            return SDK_ERROR
        ret, value = self.sd
        buf.value = value.encode()
        return ret

    def SLM_Ctrl_ReadSO(self, _number: int, buf: Any) -> int:
        if self.so is None:
            return SDK_ERROR
        ret, value = self.so
        buf.value = value.encode()
        return ret

    # open() 需要的最小 SDK 面
    def SLM_Ctrl_Open(self, *_args: Any) -> int:
        return 0

    def SLM_Ctrl_Close(self, *_args: Any) -> int:
        return 0

    def SLM_Ctrl_ReadSU(self, *_args: Any) -> int:
        return 0


def _make_slm(sdk: FakeSdk) -> Santec:
    """构造一个已打开的 SLM, SDK 换成假实现。"""
    slm = Santec(slm_number=1)
    slm._slm = sdk
    slm.is_open = True
    return slm


def _capture_warnings() -> tuple[list[str], int]:
    """返回一个 (消息列表, handler_id) 对, 用于抓取 warning 级日志。"""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING")
    return messages, handler_id


# ── SDK 假象 ────────────────────────────────────────────────────────────────

#: 健康: ReadSDO 一次读到两块板的 ID
HEALTHY = FakeSdk(sdo=(0, DRIVE_ID, OPTION_ID), sd=(0, DRIVE_ID), so=(0, OPTION_ID))
#: ReadSDO 失败, 但只读驱动板的 ReadSD 成功 (仍是 canonical 身份)
DRIVE_ONLY = FakeSdk(sd=(0, DRIVE_ID), so=(0, OPTION_ID))
#: 驱动板不应答: ReadSDO/ReadSD 失败, 只有 ReadSO (选板) 成功
OPTION_ONLY = FakeSdk(so=(0, OPTION_ID))
#: 三路全失败
NOTHING = FakeSdk()


class TestSerialSource:
    """身份来源必须如实反映回退链走到哪一步。"""

    def test_healthy_device_reports_driveboard(self) -> None:
        slm = _make_slm(HEALTHY)
        assert slm.get_serial_number() == DRIVE_ID
        assert slm.serial_source() == "driveboard"

    def test_readsd_only_reports_driveboard_readsd(self) -> None:
        slm = _make_slm(DRIVE_ONLY)
        assert slm.get_serial_number() == DRIVE_ID
        assert slm.serial_source() == "driveboard_readsd"

    def test_option_only_reports_fallback(self) -> None:
        slm = _make_slm(OPTION_ONLY)
        assert slm.get_serial_number() == OPTION_ID
        assert slm.serial_source() == "optionboard_fallback"

    def test_unreadable_reports_none(self) -> None:
        slm = _make_slm(NOTHING)
        assert slm.get_serial_number() is None
        assert slm.serial_source() is None

    def test_source_is_none_before_any_read(self) -> None:
        slm = _make_slm(HEALTHY)
        assert slm.serial_source() is None

    def test_backward_compatible_return_value(self) -> None:
        """回退时返回值仍是选板 ID (旧调用方拿到的字符串不变)。"""
        slm = _make_slm(OPTION_ONLY)
        assert slm.get_serial_number() == OPTION_ID
        assert slm.serial_number == OPTION_ID


class TestFallbackWarning:
    """回退必须是 warning 级日志, 不能只留在 debug 里。"""

    def test_fallback_emits_warning(self) -> None:
        slm = _make_slm(OPTION_ONLY)
        messages, handler_id = _capture_warnings()
        try:
            slm.get_serial_number()
        finally:
            logger.remove(handler_id)

        joined = "\n".join(messages)
        assert any("驱动板" in m for m in messages), joined
        assert any(OPTION_ID in m for m in messages), joined
        assert any("物理断电" in m for m in messages), joined

    def test_healthy_device_emits_no_warning(self) -> None:
        slm = _make_slm(HEALTHY)
        messages, handler_id = _capture_warnings()
        try:
            slm.get_serial_number()
        finally:
            logger.remove(handler_id)

        assert messages == []

    def test_debug_log_still_records_each_source(self) -> None:
        """三路来源仍留 debug 日志, 便于事后从日志复盘身份来源。"""
        slm = _make_slm(OPTION_ONLY)
        debug_messages: list[str] = []
        handler_id = logger.add(debug_messages.append, level="DEBUG")
        try:
            slm.get_serial_number()
        finally:
            logger.remove(handler_id)

        joined = "\n".join(debug_messages)
        assert "optionboardID(ReadSO)" in joined


class TestCheckDriveBoard:
    """只读健康检查。"""

    def test_healthy(self) -> None:
        slm = _make_slm(HEALTHY)
        status = slm.check_drive_board()
        assert isinstance(status, DriveBoardStatus)
        assert status.ok is True
        assert bool(status) is True
        assert status.drive_board_id == DRIVE_ID
        assert status.source == "driveboard"

    def test_readsd_only_is_healthy(self) -> None:
        slm = _make_slm(DRIVE_ONLY)
        status = slm.check_drive_board()
        assert status.ok is True
        assert status.drive_board_id == DRIVE_ID
        assert status.source == "driveboard_readsd"

    def test_option_only_is_not_healthy(self) -> None:
        slm = _make_slm(OPTION_ONLY)
        status = slm.check_drive_board()
        assert status.ok is False
        assert bool(status) is False
        assert status.drive_board_id is None
        assert status.option_board_id == OPTION_ID
        assert status.source == "optionboard_fallback"
        assert "驱动板" in status.detail

    def test_status_is_queryable_after_check(self) -> None:
        slm = _make_slm(OPTION_ONLY)
        assert slm.drive_board_status() is None
        slm.check_drive_board()
        assert slm.drive_board_status() is not None
        assert slm.drive_board_status().ok is False  # type: ignore[union-attr]

    def test_requires_open_device(self) -> None:
        slm = _make_slm(HEALTHY)
        slm.is_open = False
        with pytest.raises(RuntimeError, match="SLM设备未打开"):
            slm.check_drive_board()


class TestRequireDriveBoard:
    """显式断言: 采集实验开始前调用。"""

    def test_healthy_does_not_raise(self) -> None:
        slm = _make_slm(HEALTHY)
        assert slm.require_drive_board() == DRIVE_ID

    def test_option_only_raises(self) -> None:
        slm = _make_slm(OPTION_ONLY)
        with pytest.raises(SLMDriveBoardError) as excinfo:
            slm.require_drive_board()

        message = str(excinfo.value)
        assert "驱动板" in message
        assert OPTION_ID in message
        assert "不调制" in message
        assert "物理断电" in message

    def test_error_is_a_santec_error(self) -> None:
        """继承 SantecError ⇒ 既有的 except SantecError 仍然兜得住。"""
        slm = _make_slm(OPTION_ONLY)
        with pytest.raises(SantecError):
            slm.require_drive_board()

    def test_nothing_readable_raises(self) -> None:
        slm = _make_slm(NOTHING)
        with pytest.raises(SLMDriveBoardError) as excinfo:
            slm.require_drive_board()
        assert "物理断电" in str(excinfo.value)


class TestConfigGuard:
    """回退身份不得用于配置存取 (否则同一台设备被拆成两份配置)。"""

    @pytest.fixture(autouse=True)
    def _isolated_config_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """把 SLM 配置目录指向 tmp_path —— 测试绝不写进生产 data/slm_configs/。"""
        target = tmp_path / "slm"
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            driver_module.SLM_CONFIG._manager, "device_config_dir", target
        )

    @staticmethod
    def _read_identity(slm: Santec) -> str | None:
        """模拟 open() 的读标识步骤 (驱动板身份 → _serial_number)。"""
        serial = slm.get_serial_number()
        slm._serial_number = serial
        return serial

    def test_fallback_identity_does_not_save_config(self) -> None:
        slm = _make_slm(OPTION_ONLY)
        self._read_identity(slm)
        messages, handler_id = _capture_warnings()
        try:
            slm.save_config()
        finally:
            logger.remove(handler_id)

        joined = "\n".join(messages)
        assert "跳过配置保存" in joined
        assert "不按回退 ID 建配置" in joined
        # 不得在配置目录里留下按回退 ID 命名的文件
        assert list(driver_module.SLM_CONFIG._manager.device_config_dir.glob("*.json")) == []

    def test_fallback_identity_does_not_load_config(self) -> None:
        slm = _make_slm(OPTION_ONLY)
        self._read_identity(slm)
        assert slm.load_config() == {}

    def test_canonical_identity_keeps_config_path(self) -> None:
        slm = _make_slm(HEALTHY)
        self._read_identity(slm)
        assert slm._config_identity() == DRIVE_ID


class TestOpenBehaviour:
    """open() 的保守策略: 驱动板不应答时**不抛**, 但要记状态 + warning。"""

    @staticmethod
    def _open(slm: Santec, monkeypatch: pytest.MonkeyPatch) -> None:
        """跑一次 open(), 把与本测试无关的设备动作打桩掉。"""
        monkeypatch.setattr(slm, "_apply_config_params", lambda _cfg: None)
        monkeypatch.setattr(slm, "_load_correction", lambda _cfg: None)
        monkeypatch.setattr(slm, "_setup_wavelength", lambda *a, **k: None)
        monkeypatch.setattr(slm, "_set_memory_mode", lambda *a, **k: None)
        slm.open()

    def test_open_succeeds_on_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slm = Santec(slm_number=1)
        slm._slm = OPTION_ONLY
        self._open(slm, monkeypatch)

        assert slm.is_open
        assert slm.serial_source() == "optionboard_fallback"
        assert slm.drive_board_status() is not None
        assert slm.drive_board_status().ok is False  # type: ignore[union-attr]

    def test_open_logs_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slm = Santec(slm_number=1)
        slm._slm = OPTION_ONLY
        messages, handler_id = _capture_warnings()
        try:
            self._open(slm, monkeypatch)
        finally:
            logger.remove(handler_id)

        joined = "\n".join(messages)
        assert "驱动板" in joined
        assert "require_drive_board()" in joined

    def test_open_on_healthy_device_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slm = Santec(slm_number=1)
        slm._slm = HEALTHY
        messages, handler_id = _capture_warnings()
        try:
            self._open(slm, monkeypatch)
        finally:
            logger.remove(handler_id)

        assert messages == []
        assert slm.serial_source() == "driveboard"
        assert slm.drive_board_status().ok is True  # type: ignore[union-attr]
