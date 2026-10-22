"""测试 SLM 波前矫正的**默认启用状态**与**厂商默认 CSV 兜底**策略。

背景 (2026-10-22):
    仓库里唯一的厂商矫正 CSV 是
    ``data/calibration/Wavefront_correction_Data_240236000006(520nm).csv``,
    它属于**另一块面板** (序列号 240236000006) 在 **520nm** 下测得,
    与本机面板 (22030108 @1064nm) 不匹配。

    此前 ``Santec._load_correction`` 读 ``config.get("correction_enabled", True)``,
    且开启后 ``default_path`` 指向那块 520nm CSV —— 于是任何**没写**
    ``correction_csv_path`` 的 SLM 配置都会静默吃到那片无关波前
    (实测零级从 ~14px 弥散成 ~68px 弥散晕, 峰值只剩 33/255)。

本文件固化的契约 (``_load_correction`` 的三档语义):

    - ``correction_enabled`` 显式 **False** ⇒ 不加载任何矫正（即便给了路径）;
    - ``correction_enabled`` 显式 **True**  ⇒ 按 resolve 优先级加载
      (显式路径 → 配置路径 → 厂商默认 CSV, 厂商默认仅在显式 opt-in 时);
    - ``correction_enabled`` **未记录**      ⇒
      - **不加载厂商默认 CSV**（本次修复的核心，绝不能回退）;
      - 但**显式路径仍然加载**（构造参数 / 配置里的 ``correction_csv_path``）;
      - 两者都没有 ⇒ 不加载 + warning。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 在 import Santec 之前 mock SLM SDK 模块
# （_slm_win 仅在 Windows + SDK installed 时可用）
sys.modules["ao_shaping.drivers.slm.santec._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.drivers.slm.santec.wavefront_correction import WavefrontCorrection


# 确保所有测试都使用 mock 的 SLM SDK
@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> None:
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm.santec._slm_win": MagicMock()},
        clear=False,
    ):
        yield


def write_correction_csv(path: Path, height: int = 4, width: int = 5) -> Path:
    """写一个符合厂商格式的**小**矫正 CSV。

    不使用仓库里那块 9MB 的真厂商文件（且它属于别的面板/波长）。

    Args:
        path: 输出路径
        height: 数据区行数
        width: 数据区列数

    Returns:
        写入的路径
    """
    rng = np.random.default_rng(0)
    amplitude, offset = 100.0, 128.0
    phi = (rng.random((height, width)) - 0.5) * (np.pi / 2)
    data = offset + amplitude * np.cos(phi)

    header = "Y/X," + ",".join(str(i) for i in range(width))
    lines = [header]
    for r in range(height):
        values = np.rint(data[r, :]).astype(int)
        lines.append(str(r) + "," + ",".join(str(v) for v in values))
    path.write_text("\n".join(lines))
    return path


class TestCorrectionDefaultFallback:
    """``Santec._load_correction`` 的三档启用语义 + 厂商默认兜底策略"""

    # ── 未记录 correction_enabled ─────────────────────

    def test_missing_key_without_any_path_disables(self) -> None:
        """未记录启用状态且没有任何显式路径 ⇒ 视为不启用, 不加载任何矫正数据"""
        slm = Santec(slm_number=1)
        slm._load_correction({"wavelength": 1064})

        assert slm.correction_enabled is False
        assert slm.correction_csv_path is None
        assert slm._correction.is_valid is False

    def test_missing_key_does_not_silently_apply_vendor_csv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """未记录启用状态 ⇒ **不得**静默套用厂商默认 CSV（核心回归点）"""
        vendor = write_correction_csv(tmp_path / "vendor.csv")
        monkeypatch.setattr(Santec, "VENDOR_DEFAULT_CORRECTION_CSV", vendor)

        slm = Santec(slm_number=1)
        slm._load_correction({})

        assert slm.correction_enabled is False
        assert slm.correction_csv_path is None

    def test_missing_key_with_constructor_path_loads(self, tmp_path: Path) -> None:
        """未记录启用状态, 但构造参数给了路径 ⇒ **显式意图仍然生效**"""
        csv = write_correction_csv(tmp_path / "c.csv")
        slm = Santec(slm_number=1, correction_csv_path=str(csv))
        slm._load_correction({})

        assert slm._correction.is_valid is True
        assert slm.correction_enabled is True
        assert slm.correction_csv_path == Path(csv)

    def test_missing_key_with_config_path_loads(self, tmp_path: Path) -> None:
        """未记录启用状态, 但配置里给了 ``correction_csv_path`` ⇒ 仍然加载"""
        csv = write_correction_csv(tmp_path / "c.csv")
        slm = Santec(slm_number=1)
        slm._load_correction({"correction_csv_path": str(csv)})

        assert slm._correction.is_valid is True
        assert slm.correction_enabled is True
        assert slm.correction_csv_path == Path(csv)

    def test_missing_key_prefers_constructor_path_over_config_path(
        self, tmp_path: Path
    ) -> None:
        """未记录启用状态时, 构造参数路径优先于配置里的路径"""
        constructor_csv = write_correction_csv(tmp_path / "constructor.csv")
        config_csv = write_correction_csv(tmp_path / "config.csv")
        slm = Santec(slm_number=1, correction_csv_path=str(constructor_csv))
        slm._load_correction({"correction_csv_path": str(config_csv)})

        assert slm.correction_csv_path == Path(constructor_csv)

    def test_from_params_keeps_explicit_path_when_state_not_recorded(
        self, tmp_path: Path
    ) -> None:
        """``from_params`` 带来的显式路径在未记录启用状态时仍然生效"""
        csv = write_correction_csv(tmp_path / "c.csv")
        params = SimpleNamespace(
            slm_number=1,
            slm_wavelength=1064,
            use_120hz=False,
            video_mode=0,
            shift_x=0,
            shift_y=0,
            correction_csv_path=str(csv),
        )
        slm = Santec.from_params(params)
        slm._load_correction({})

        assert slm._correction.is_valid is True
        assert slm.correction_csv_path == Path(csv)

    # ── 显式 correction_enabled = False ───────────────

    def test_explicit_false_disables(self) -> None:
        """``correction_enabled: false`` ⇒ 不加载"""
        slm = Santec(slm_number=1)
        slm._load_correction({"correction_enabled": False, "correction_csv_path": None})

        assert slm.correction_enabled is False
        assert slm.correction_csv_path is None

    def test_explicit_false_with_path_still_disables(self, tmp_path: Path) -> None:
        """``correction_enabled: false`` + 给了路径 ⇒ 仍然不加载（显式关闭优先）"""
        csv = write_correction_csv(tmp_path / "c.csv")
        slm = Santec(slm_number=1)
        slm._load_correction(
            {"correction_enabled": False, "correction_csv_path": str(csv)}
        )

        assert slm.correction_enabled is False
        assert slm.correction_csv_path is None

    def test_explicit_false_with_constructor_path_still_disables(
        self, tmp_path: Path
    ) -> None:
        """``correction_enabled: false`` + 构造参数路径 ⇒ 仍然不加载"""
        csv = write_correction_csv(tmp_path / "c.csv")
        slm = Santec(slm_number=1, correction_csv_path=str(csv))
        slm._load_correction({"correction_enabled": False})

        assert slm.correction_enabled is False
        assert slm.correction_csv_path is None

    # ── 显式 correction_enabled = True ────────────────

    def test_enabled_with_config_path_loads(self, tmp_path: Path) -> None:
        """``correction_enabled: true`` + 配置文件路径 ⇒ 被加载"""
        csv = write_correction_csv(tmp_path / "c.csv")
        slm = Santec(slm_number=1)
        slm._load_correction(
            {"correction_enabled": True, "correction_csv_path": str(csv)}
        )

        assert slm._correction.is_valid is True
        assert slm.correction_enabled is True
        assert slm.correction_csv_path == Path(csv)

    def test_enabled_with_constructor_path_loads(self, tmp_path: Path) -> None:
        """``correction_enabled: true`` + 构造参数路径 ⇒ 被加载"""
        csv = write_correction_csv(tmp_path / "c.csv")
        slm = Santec(slm_number=1, correction_csv_path=str(csv))
        slm._load_correction({"correction_enabled": True})

        assert slm._correction.is_valid is True
        assert slm.correction_enabled is True
        assert slm.correction_csv_path == Path(csv)

    def test_enabled_without_any_path_loads_nothing(self) -> None:
        """``correction_enabled: true`` 但没有任何路径 ⇒ 不加载（不自动兜底）"""
        slm = Santec(slm_number=1)
        slm._load_correction({"correction_enabled": True})

        assert slm.correction_enabled is False
        assert slm.correction_csv_path is None

    # ── 厂商默认 CSV 的 opt-in ────────────────────────

    def test_vendor_default_requires_explicit_optin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """厂商默认 CSV 需要 ``use_vendor_default_correction: true`` 才生效"""
        vendor = write_correction_csv(tmp_path / "vendor.csv")
        monkeypatch.setattr(Santec, "VENDOR_DEFAULT_CORRECTION_CSV", vendor)

        slm = Santec(slm_number=1)
        slm._load_correction({"correction_enabled": True})
        assert slm.correction_enabled is False
        assert slm.correction_csv_path is None

        slm2 = Santec(slm_number=1)
        slm2._load_correction(
            {"correction_enabled": True, "use_vendor_default_correction": True}
        )
        assert slm2._correction.is_valid is True
        assert slm2.correction_csv_path == vendor

    def test_vendor_default_not_applied_when_state_not_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """未记录启用状态时, 即使有 opt-in 也不套用厂商默认 CSV"""
        vendor = write_correction_csv(tmp_path / "vendor.csv")
        monkeypatch.setattr(Santec, "VENDOR_DEFAULT_CORRECTION_CSV", vendor)

        slm = Santec(slm_number=1)
        slm._load_correction({"use_vendor_default_correction": True})

        assert slm.correction_enabled is False
        assert slm.correction_csv_path is None


class TestResolveNonePath:
    """``WavefrontCorrection.resolve`` 对 None/空串路径的处理"""

    def test_config_none_path_returns_none(self) -> None:
        """config 里 ``correction_csv_path`` 为 None ⇒ 不抛异常, 返回 None"""
        result = WavefrontCorrection.resolve(config={"correction_csv_path": None})
        assert result is None

    def test_config_empty_string_path_returns_none(self) -> None:
        """config 里 ``correction_csv_path`` 为空串 ⇒ 跳过该候选, 返回 None"""
        result = WavefrontCorrection.resolve(config={"correction_csv_path": ""})
        assert result is None

    def test_config_none_path_does_not_block_explicit_path(
        self, tmp_path: Path
    ) -> None:
        """config 值为 None 时, 显式路径仍然可用（优先级链不被破坏）"""
        csv = write_correction_csv(tmp_path / "c.csv")
        result = WavefrontCorrection.resolve(
            explicit_path=str(csv), config={"correction_csv_path": None}
        )
        assert result is not None
        assert result.is_valid is True
        assert result.csv_path == Path(csv)

    def test_no_candidates_returns_none(self) -> None:
        """无任何候选 ⇒ 返回 None"""
        assert WavefrontCorrection.resolve() is None
