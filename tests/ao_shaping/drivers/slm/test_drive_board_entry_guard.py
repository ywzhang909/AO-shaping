"""采集入口守卫 ``require_drive_board_entry`` 的接线契约测试 (#186)。

驱动层 ``require_drive_board()`` 的 SDK 语义 (三路身份读取 / 回退 / 报错文案)
已由本目录 ``test_slm_drive_board_health.py`` 复现; 这里只固化接线层:

1. factory 共享 helper: 健康 → 返回驱动板序列号; 驱动板不应答 →
   ``SLMDriveBoardError``; 仿真面板/替身 (无该方法) → no-op 返回 ``None``;
2. ``ZernikeSLM.open()`` 在守卫失败时必须先关闭内部 SDK 句柄再上抛
   (失败发生在 ``with ZernikeSLM(...)`` 的 ``__enter__`` 里, Python 不会
   再调用 ``__exit__``, 不关句柄会一直占用面板);
3. 各采集会话体确实还调用守卫 (AST 钉住, 防止接线被静默移除)。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ao_shaping.drivers.slm import SLMDriveBoardError, Santec, ZernikeSLM
from ao_shaping.drivers.slm.factory import require_drive_board_entry

DRIVE_ID = "22030108"
OPTION_ID = "23020027"

#: 任意非 SLM_OK 的错误码 (模拟 USB 链路错误)
SDK_ERROR = -10032


@pytest.fixture(autouse=True)
def _mock_slm_sdk():
    """把 SDK 库 Weg in假模块 -- driver 在 ``Santec.__init__`` 内才 import 它。

    与 ``test_slm_drive_board_health.py`` 的同一套兜底: 换进假实现的是
    ``Santec._slm`` 属性, 所以 SDK 假象完全收在 FakeSdk 里。
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

    每个字段是 ``(返回码, 字符串...)``; ``None`` 表示该路调用直接失败。
    形状与 ``test_slm_drive_board_health.py::FakeSdk`` 相同 (tests 树内部
    无法互相 import, 故本地复刻最小面), 外加 open/close 调用计数,
    用于断言守卫失败路径上的句柄回收。
    """

    sdo: tuple[int, str, str] | None = None
    sd: tuple[int, str] | None = None
    so: tuple[int, str] | None = None
    close_calls: int = field(default=0, init=False)

    def __getattr__(self, name: str) -> Any:
        # open() 的完整 SDK 面不止三路读 (WriteVI / ReadST / SetWL...) -- 未
        # 显式实现的其余调用一律按 SLM_OK 处理, 免得逐个追补; 带缓冲区的读
        # 必须显式实现, 不能走这个兜底。
        return lambda *_args, **_kwargs: 0

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

    # open()/close() 需要的最小 SDK 面
    def SLM_Ctrl_Open(self, *_args: Any) -> int:
        return 0

    def SLM_Ctrl_Close(self, *_args: Any) -> int:
        self.close_calls += 1
        return 0

    def SLM_Ctrl_ReadSU(self, *_args: Any) -> int:
        return 0

    def SLM_Ctrl_ReadWL(self, _number: int, wavelength: Any, phase: Any) -> int:
        # 驱动 open() 也会读设备波长 (byref 传入 c_uint32); 返回与测试波长一致
        # 的值, 避免走进 set_wavelength 的额外 SDK 面。
        wavelength._obj.value = 1064
        phase._obj.value = 10
        return 0


def _make_slm(sdk: FakeSdk) -> Santec:
    """构造一个把 SDK 换成假实现的 SLM (沿用健康测试的同一构造)。"""
    slm = Santec(slm_number=1)
    slm._slm = sdk
    slm.is_open = True
    return slm


def _make_zslm(sdk: FakeSdk) -> ZernikeSLM:
    zslm = ZernikeSLM(slm_number=1, wavelength=1064, n_max=4)
    zslm._slm._slm = sdk
    return zslm


class TestRequireDriveBoardEntry:
    def test_healthy_returns_canonical_drive_board_id(self) -> None:
        slm = _make_slm(FakeSdk(sdo=(0, DRIVE_ID, OPTION_ID), sd=(0, DRIVE_ID), so=(0, OPTION_ID)))
        assert require_drive_board_entry(slm) == DRIVE_ID

    def test_option_only_raises(self) -> None:
        slm = _make_slm(FakeSdk(so=(0, OPTION_ID)))
        with pytest.raises(SLMDriveBoardError):
            require_drive_board_entry(slm)

    def test_nothing_readable_raises(self) -> None:
        slm = _make_slm(FakeSdk())
        with pytest.raises(SLMDriveBoardError):
            require_drive_board_entry(slm)

    def test_stub_without_the_method_is_a_noop(self) -> None:
        # 仿真面板 (SimSLMPib) 与测试替身没有 require_drive_board --
        # --cam-type sim 的数字孪生路径必须完全不受守卫影响。
        assert require_drive_board_entry(object()) is None


class TestZernikeSLOpenGuard:
    def test_healthy_open_completes(self) -> None:
        zslm = _make_zslm(
            FakeSdk(sdo=(0, DRIVE_ID, OPTION_ID), sd=(0, DRIVE_ID), so=(0, OPTION_ID))
        )
        zslm.open()
        assert zslm.is_open is True
        zslm.close()
        assert zslm._slm._slm.close_calls >= 1
        assert zslm.is_open is False

    def test_option_only_raises_and_closes_the_internal_handle(self) -> None:
        sdk = FakeSdk(so=(0, OPTION_ID))
        zslm = _make_zslm(sdk)
        with pytest.raises(SLMDriveBoardError):
            zslm.open()
        # __enter__ 抛出时 __exit__ 不会运行 -- 守卫必须自己收口。
        assert sdk.close_calls >= 1
        assert zslm.is_open is False


#: 每个采集会话体都必须还持有守卫调用 (文件 → 引入守卫的位置不变式)
WIRED_FILES = (
    "src/ao_shaping/drivers/slm/zernike_slm.py",
    "src/ao_shaping/optimizer/wfless/slm_zernike_pib.py",
    "src/ao_shaping/optimizer/wfless/slm_square_shaping.py",
    "src/ao_shaping/optimizer/wfless/slm_gs_refine.py",
    "src/ao_shaping/optimizer/wfless/slm_model_in_loop.py",
    "src/ao_shaping/runners/slm/zernike_matrix_runner.py",
)


def _calls_the_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if isinstance(name, str) and "require_drive_board" in name:
            return True
    return False


class TestWiringPinned:
    def test_every_acquisition_entry_still_calls_the_guard(self) -> None:
        root = Path(__file__).resolve().parents[4]
        missing = [
            rel
            for rel in WIRED_FILES
            if not _calls_the_guard(ast.parse((root / rel).read_text(encoding="utf-8")))
        ]
        assert missing == []

    def test_helper_is_the_single_home(self) -> None:
        import ao_shaping.drivers.slm.factory as factory

        assert "require_drive_board_entry" in factory.__all__
        assert callable(factory.require_drive_board_entry)
