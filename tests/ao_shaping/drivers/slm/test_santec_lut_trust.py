"""生产显示路径的 LUT 信任闸门：未经验证的表绝不允许静默上屏。

``Santec.load_lut`` 现在在解析任何表数据之前先过
``utils.slm.slm_lut.require_validated_lut``。本文件走的是**生产入口**
（``save_serial_lut`` 写盘 → ``Santec.load_lut`` 读盘），而不是内部函数，
因为要钉住的行为是"操作员指着一个目录调用驱动，会发生什么"：

* ``provisional``（测得但无人判定）→ 抛 ``LutTrustError``，驱动状态不变；
* ``validated``（有人签字）→ 正常加载；
* 缺 ``meta.json`` / 序列号与目录不符 / status 非法 → 一律拒绝，
  **不做**"老格式放行"的向后兼容。

目录不存在与 ``load_lut(None)`` 沿用原有语义（告警 + 保持状态 / 清除），
因为这两条不是"来历不明的表"，而是"没有表"。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 与同目录其它驱动测试一致：先 mock 掉 SLM SDK，再 import 驱动
sys.modules["ao_shaping.drivers.slm.santec._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.utils.slm.slm_lut import (
    LUT_META_FILENAME,
    LUT_STATUS_PROVISIONAL,
    LUT_STATUS_VALIDATED,
    LutTrustError,
    mark_lut_validated,
    save_serial_lut,
)

#: 两台真实设备的序列号（drivers/AGENTS.md 已收口：#1=1064nm，#2=532nm）。
SLM1_SERIAL = "22030108"
SLM2_SERIAL = "23020026"

N_POINTS = 65
TABLE_SIZE = 1024


@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> Generator[None, None, None]:
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm.santec._slm_win": MagicMock()},
        clear=False,
    ):
        yield


def _scan() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A smooth, in-domain fake calibration: gray 0..1023 over 65 points."""
    g_values = np.linspace(0.0, 1023.0, N_POINTS)
    phi = np.linspace(0.0, 2.0 * np.pi, N_POINTS)
    inverse_gray = np.arange(TABLE_SIZE, dtype=np.uint16)
    return g_values, phi, inverse_gray


def _archive(
    root: Path,
    *,
    slm_serial: str = SLM1_SERIAL,
    status: str = LUT_STATUS_PROVISIONAL,
) -> Path:
    """Write one LUT through the production archive entry point."""
    g_values, phi, inverse_gray = _scan()
    return save_serial_lut(
        slm_serial=slm_serial,
        g_values=g_values,
        phi=phi,
        inverse_gray=inverse_gray,
        source="tests/ao_shaping/drivers/slm/test_santec_lut_trust.py",
        status=status,
        valley=0.02,
        wavelength_nm=1064.0,
        gray_for_2pi=993,
        quality={"retained_at_last": 0.99},
        notes="测试夹具",
        root_dir=root,
    )


class TestProductionGate:
    """``Santec.load_lut`` 对一份**真实落盘**的归档做什么。"""

    def test_provisional_lut_is_refused_and_state_unchanged(self, tmp_path: Path) -> None:
        """未判定的表必须报错，而不是"警告一下然后照常上屏"。"""
        lut_dir = _archive(tmp_path, status=LUT_STATUS_PROVISIONAL)
        slm = Santec(slm_number=1)

        with pytest.raises(LutTrustError):
            slm.load_lut(lut_dir)

        assert slm.lut is None
        assert slm.lut_dir is None

    def test_provisional_refusal_names_the_status(self, tmp_path: Path) -> None:
        """报错文案要能自证原因：操作员不必读源码就知道该去改 status。"""
        lut_dir = _archive(tmp_path, status=LUT_STATUS_PROVISIONAL)
        slm = Santec(slm_number=1)

        with pytest.raises(LutTrustError) as excinfo:
            slm.load_lut(lut_dir)

        assert LUT_STATUS_PROVISIONAL in str(excinfo.value)
        assert LUT_STATUS_VALIDATED in str(excinfo.value)

    def test_validated_lut_loads(self, tmp_path: Path) -> None:
        """过闸后行为与旧版一致：表按 npz 原样加载。"""
        lut_dir = _archive(tmp_path, status=LUT_STATUS_VALIDATED)
        _, _, expected = _scan()
        slm = Santec(slm_number=1)

        slm.load_lut(lut_dir)

        assert slm.lut is not None
        assert slm.lut_dir == lut_dir
        np.testing.assert_array_equal(slm.lut, expected)

    def test_mark_validated_unblocks_the_same_directory(self, tmp_path: Path) -> None:
        """唯一的放行方式是显式签字，且不需要重写表数据。"""
        lut_dir = _archive(tmp_path, status=LUT_STATUS_PROVISIONAL)
        slm = Santec(slm_number=1)
        with pytest.raises(LutTrustError):
            slm.load_lut(lut_dir)

        meta = mark_lut_validated(
            lut_dir,
            verified_by="pytest",
            notes="离线夹具：验证人/依据字段非空即可",
        )
        assert meta["status"] == LUT_STATUS_VALIDATED

        slm.load_lut(lut_dir)
        assert slm.lut is not None

    def test_missing_sidecar_is_refused(self, tmp_path: Path) -> None:
        """缺 meta.json = 来历不明，不做"老格式放行"。"""
        lut_dir = _archive(tmp_path, status=LUT_STATUS_VALIDATED)
        (lut_dir / LUT_META_FILENAME).unlink()
        slm = Santec(slm_number=1)

        with pytest.raises(LutTrustError):
            slm.load_lut(lut_dir)
        assert slm.lut is None

    def test_unparseable_sidecar_is_refused(self, tmp_path: Path) -> None:
        """半截 JSON（写盘中途断电）同样不可信。"""
        lut_dir = _archive(tmp_path, status=LUT_STATUS_VALIDATED)
        (lut_dir / LUT_META_FILENAME).write_text("{not json", encoding="utf-8")
        slm = Santec(slm_number=1)

        with pytest.raises(LutTrustError):
            slm.load_lut(lut_dir)
        assert slm.lut is None

    def test_unknown_status_is_refused(self, tmp_path: Path) -> None:
        """status 只有两个合法取值；任何"看起来更宽松"的值都不算数。"""
        lut_dir = _archive(tmp_path, status=LUT_STATUS_VALIDATED)
        meta_path = lut_dir / LUT_META_FILENAME
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["status"] = "ok"
        meta_path.write_text(json.dumps(payload), encoding="utf-8")
        slm = Santec(slm_number=1)

        with pytest.raises(LutTrustError):
            slm.load_lut(lut_dir)
        assert slm.lut is None

    def test_sidecar_from_another_panel_is_refused(self, tmp_path: Path) -> None:
        """表随面板绑定：把 #1 的 sidecar 复制到 #2 目录不能蒙混过关。"""
        lut_dir_1 = _archive(tmp_path / "a", slm_serial=SLM1_SERIAL, status=LUT_STATUS_VALIDATED)
        lut_dir_2 = _archive(
            tmp_path / "b", slm_serial=SLM2_SERIAL, status=LUT_STATUS_VALIDATED
        )
        meta_path_2 = lut_dir_2 / LUT_META_FILENAME
        meta_path_2.write_bytes((lut_dir_1 / LUT_META_FILENAME).read_bytes())
        slm = Santec(slm_number=1)

        with pytest.raises(LutTrustError):
            slm.load_lut(lut_dir_2)
        assert slm.lut is None


class TestNonLutPathsKeepOldSemantics:
    """"没有表"不是"表不可信"——这两条老语义必须原样保留。"""

    def test_missing_directory_warns_and_keeps_state(self, tmp_path: Path) -> None:
        slm = Santec(slm_number=1)

        slm.load_lut(tmp_path / "nonexistent")

        assert slm.lut is None
        assert slm.lut_dir is None

    def test_none_clears_a_loaded_lut(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path, status=LUT_STATUS_VALIDATED)
        slm = Santec(slm_number=1)
        slm.load_lut(lut_dir)
        assert slm.lut is not None

        slm.load_lut(None)

        assert slm.lut is None
        assert slm.lut_dir is None

    def test_a_refusal_does_not_drop_a_previously_loaded_lut(self, tmp_path: Path) -> None:
        """拒绝一份新表不能把已在用的好表清空（否则失败路径反而更危险）。"""
        good = _archive(tmp_path / "good", status=LUT_STATUS_VALIDATED)
        bad = _archive(tmp_path / "bad", status=LUT_STATUS_PROVISIONAL)
        slm = Santec(slm_number=1)
        slm.load_lut(good)
        loaded = slm.lut.copy()

        with pytest.raises(LutTrustError):
            slm.load_lut(bad)

        np.testing.assert_array_equal(slm.lut, loaded)
        assert slm.lut_dir == good
