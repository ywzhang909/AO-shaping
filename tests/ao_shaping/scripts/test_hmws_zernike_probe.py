"""HMWS 五点 Zernike 探针脚本的离线测试 (不碰任何硬件)。

覆盖:
* ``--dry-run`` 端到端: Noll 4 / inject 0.0 与 0.2 / bias 0.3, δ≈0.2±0.06,
  summary.json / summary.csv / probe_config.json / 五帧 .npz 全部落盘;
* 偏置语义: offset 必须进入 Zernike 系数 ``a_inject + offset``, 不是活塞 ``bias_rad``;
* CLI 解析: 默认值与 ``--noll`` / ``--inject-rad`` 多值;
* 模块导入期不引入真实设备 SDK (AST + ``sys.modules``);
* 质量门: 注入饱和假帧必须判不合格。
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "hmws_zernike_probe.py"


def _load_module():
    """按路径加载脚本 (scripts/ 不是包)。"""
    spec = importlib.util.spec_from_file_location("hmws_zernike_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_module()


# ---------------------------------------------------------------------------
# CLI 解析
# ---------------------------------------------------------------------------


def test_cli_defaults(probe) -> None:
    args = probe.build_parser().parse_args([])
    assert args.noll == [4]
    assert args.inject_rad == [0.0]
    assert args.bias_rad == pytest.approx(0.3)
    assert args.radius == pytest.approx(300.0)
    assert args.focal_mm is None
    assert args.focus_json is None
    assert args.slm_number == 1
    assert args.slm_wavelength == 1064
    assert args.cam_type == "daheng"
    assert args.cam_id == 0
    assert args.bit_depth == 12
    assert args.exposure_ms == pytest.approx(0.05)
    assert args.auto_exposure is True
    assert args.frames == 3
    assert args.settle_s == pytest.approx(0.5)
    assert args.subtract_dark is True
    assert args.wait_until_idle == pytest.approx(600.0)
    assert args.out == Path("data/hmws_zernike_probe")
    assert args.dry_run is False
    assert args.save_frames is True


def test_cli_multi_values(probe) -> None:
    args = probe.build_parser().parse_args(
        ["--noll", "4", "5", "--inject-rad", "-0.1", "0.0", "0.2"]
    )
    assert args.noll == [4, 5]
    assert args.inject_rad == [-0.1, 0.0, 0.2]
    flags = probe.build_parser().parse_args(
        ["--no-auto-exposure", "--no-subtract-dark", "--no-save-frames"]
    )
    assert flags.auto_exposure is False
    assert flags.subtract_dark is False
    assert flags.save_frames is False


# ---------------------------------------------------------------------------
# 偏置语义 / Noll 映射
# ---------------------------------------------------------------------------


def test_probe_offsets_enter_zernike_coefficient(probe) -> None:
    offsets = probe.probe_offsets(0.3)
    assert list(offsets) == ["a", "m", "p", "mm", "pp"]
    assert offsets == {"a": 0.0, "m": -0.3, "p": 0.3, "mm": -0.6, "pp": 0.6}


def test_bias_is_modal_coefficient_not_piston(probe, monkeypatch) -> None:
    """``generate_probe_phase`` 只能收到 ``coefficients={nm: inject+offset}``。

    不允许把 b 当作全孔径活塞 ``bias_rad`` 传下去, 也不允许自备 Noll 查找表。
    """
    calls: list[dict[str, object]] = []

    class Helper:
        def generate_probe_phase(self, **kwargs):
            calls.append(kwargs)
            return np.zeros((8, 8), dtype=np.float64)

    phases = probe.build_probe_phases(Helper(), 4, 0.2, 0.3, radius=4.0)
    assert list(phases) == ["a", "m", "p", "mm", "pp"]
    assert len(calls) == 5
    expected = {"a": 0.2, "m": -0.1, "p": 0.5, "mm": -0.4, "pp": 0.8}
    for key, call in zip(expected, calls):
        assert "bias_rad" not in call
        assert call["coefficients"] == {
            probe.noll_to_mode(4): pytest.approx(expected[key])
        }


def test_noll_mapping_uses_canonical_source(probe) -> None:
    from ao_shaping.utils.wavefront.zernike_calc import noll_to_nm

    for noll in (2, 3, 4, 5, 7, 11):
        assert probe.noll_to_mode(noll) == tuple(int(v) for v in noll_to_nm(noll))
    assert probe.noll_to_mode(4) == (2, 0)


# ---------------------------------------------------------------------------
# 模块导入期不碰设备 SDK
# ---------------------------------------------------------------------------


def test_no_top_level_device_sdk_imports() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    banned = ("gxipy", "Santec", "ao_shaping.drivers")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(banned), alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(banned), node.module


def test_import_leaves_device_sdk_unloaded() -> None:
    before = set(sys.modules)
    _load_module()
    new = set(sys.modules) - before
    assert not any(name.startswith("gxipy") for name in new)
    assert not any(name.startswith("ao_shaping.drivers.slm") for name in new)
    assert not any(name.startswith("ao_shaping.drivers.ccd") for name in new)


# ---------------------------------------------------------------------------
# dry-run 端到端
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dry_run_dir(tmp_path_factory):
    module = _load_module()
    out = tmp_path_factory.mktemp("hmws_dry_run")
    rc = module.main(
        [
            "--dry-run",
            "--noll",
            "4",
            "--inject-rad",
            "0.0",
            "0.2",
            "--bias-rad",
            "0.3",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    run_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_dry_run_end_to_end(dry_run_dir) -> None:
    summary = json.loads((dry_run_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((dry_run_dir / "probe_config.json").read_text(encoding="utf-8"))
    assert summary["simulated"] is True
    assert config["simulated"] is True
    assert summary["quality"]["qualified"] is True

    records = {r["inject_rad"]: r for r in summary["records"]}
    assert set(records) == {0.0, 0.2}
    for inject, record in records.items():
        assert record["noll"] == 4
        assert record["n_m"] == [2, 0]
        assert record["valid"] is True
        # 差分 recovered - inject ≈ 0 (线性恢复, 无系统偏差)
        assert record["delta_rad"] == pytest.approx(0.0, abs=0.06)
        # inject=0.2 时 recovered ≈ 0.2 ± 0.06
        assert record["recovered_rad"] == pytest.approx(inject, abs=0.06)
        assert record["quality"]["qualified"] is True
        npz = dry_run_dir / record["frame_file"]
        assert npz.is_file()
        with np.load(npz) as data:
            for key in ("a", "m", "p", "mm", "pp"):
                assert key in data
                assert f"raw_{key}" in data
                assert data[key].shape == (64, 64)
    # 自检点 Noll4/inject0.2/bias0.3 ≈ 0.195
    check = config["self_check"]
    assert check["performed"] is True and check["passed"] is True
    assert check["recovered_rad"] == pytest.approx(0.195, abs=0.05)
    assert config["focal_compensated"] is False
    assert config["focal_mm"] is None
    assert config["dark"]["applied"] is False
    assert config["exposure"]["requested_ms"] == pytest.approx(0.05)
    assert "git_commit" in config and "full_scale" in config

    with (dry_run_dir / "summary.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["noll"] == "4"


def test_dry_run_linearity_multiple_injects(probe, tmp_path) -> None:
    rc = probe.main(
        [
            "--dry-run",
            "--noll",
            "4",
            "--inject-rad",
            "-0.2",
            "0.0",
            "0.2",
            "--out",
            str(tmp_path),
        ]
    )
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    fit = summary["linearity"][0]
    assert fit["noll"] == 4 and fit["n_points"] == 3
    assert fit["slope"] == pytest.approx(1.0, abs=0.25)
    assert rc in (0, 1)  # 大注入下允许质量门报警, 但产物必须完整


def test_dry_run_rejects_existing_dir(probe, tmp_path, monkeypatch) -> None:
    out = tmp_path / "x"
    args = ["--dry-run", "--out", str(out)]
    # 固定时间戳, 让两次运行必然落在同一输出目录 (不再依赖真实同秒)。
    monkeypatch.setattr(probe, "utc_stamp", lambda: "20261124_000000Z")
    assert probe.main(args) == 0
    # 同一输出目录重跑必须拒绝覆盖, 而不是静默复用。
    with pytest.raises(SystemExit):
        probe.main(args)


# ---------------------------------------------------------------------------
# 质量门
# ---------------------------------------------------------------------------


def _fake_frames(peak: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    frames = {}
    for key in ("a", "m", "p", "mm", "pp"):
        frame = rng.normal(10.0, 1.0, size=(16, 16))
        frame[8, 8] = peak
        frames[key] = frame
    return frames


def test_quality_gate_rejects_clipped_frames(probe) -> None:
    frames = _fake_frames(4095.0)
    q = probe.assess_group_quality(frames, frames, 4095.0)
    assert q["qualified"] is False
    assert any("clipped_px" in p for p in q["problems"])


def test_quality_gate_rejects_overexposed_without_clipping(probe) -> None:
    frames = _fake_frames(0.9 * 4095.0)
    q = probe.assess_group_quality(frames, frames, 4095.0)
    assert q["qualified"] is False
    assert any("FS" in p for p in q["problems"])


def test_quality_gate_rejects_underexposed(probe) -> None:
    frames = _fake_frames(12.0)
    q = probe.assess_group_quality(frames, frames, 4095.0)
    assert q["qualified"] is False
    assert any("欠曝" in p for p in q["problems"])


def test_quality_gate_rejects_degenerate_and_unconverged(probe) -> None:
    frames = _fake_frames(0.45 * 4095.0)
    assert probe.assess_group_quality(frames, frames, 4095.0)["qualified"] is True
    q = probe.assess_group_quality(
        frames, frames, 4095.0, exposure_converged=False, valid=False
    )
    assert q["qualified"] is False
    assert any("未收敛" in p for p in q["problems"])
    assert any("退化" in p for p in q["problems"])


# ---------------------------------------------------------------------------
# 相机位深 / 暗帧位深契约 (离线)
# ---------------------------------------------------------------------------


def test_camera_bit_depth_readback_mismatch_aborts(probe) -> None:
    class FakeCamera:
        bit_depth = 8
        supported_bit_depths = (8, 12, 16)

    with pytest.raises(RuntimeError) as exc:
        probe.verify_camera_bit_depth(FakeCamera(), 12, cam_type="fake")
    text = str(exc.value)
    assert "请求 12 bit" in text
    assert "设备回读 8 bit" in text
    assert "4095" in text


@dataclass(frozen=True)
class _FakeDarkEntry:
    bit_depth: int | None
    full_scale: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {"bit_depth": self.bit_depth, "full_scale": self.full_scale}


@dataclass(frozen=True)
class _FakeDark:
    mean: np.ndarray
    entry: _FakeDarkEntry
    std: np.ndarray | None = None
    full_scale: float | None = None

    def subtract(self, frame: np.ndarray, *, dtype=np.float32) -> np.ndarray:
        return np.asarray(frame, dtype=dtype) - self.mean.astype(dtype, copy=False)


def test_dark_frame_rescaled_from_12_to_8(probe) -> None:
    dark = _FakeDark(
        mean=np.array([[100.0, 200.0], [4095.0, 0.0]], dtype=np.float32),
        entry=_FakeDarkEntry(bit_depth=12),
    )
    effective, info = probe.resolve_dark_frame(
        dark,
        requested=True,
        exposure_ms=0.05,
        camera_bit_depth=8,
        camera_full_scale=255.0,
    )
    assert effective is not None
    assert info["applied"] is True
    assert info["bit_depth_mismatch"] is True
    assert info["rescaled"] is True
    np.testing.assert_array_equal(
        effective.mean, np.array([[6.0, 12.0], [255.0, 0.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        effective.subtract(np.full((2, 2), 255, dtype=np.uint16)),
        np.array([[249, 243], [0, 255]], dtype=np.float32),
    )


def test_dark_frame_skipped_when_metadata_missing(probe) -> None:
    dark = _FakeDark(
        mean=np.array([[10.0]], dtype=np.float32),
        entry=_FakeDarkEntry(bit_depth=None, full_scale=None),
    )
    effective, info = probe.resolve_dark_frame(
        dark,
        requested=True,
        exposure_ms=0.05,
        camera_bit_depth=12,
        camera_full_scale=4095.0,
    )
    assert effective is None
    assert info["applied"] is False
    assert info["matched"] is True
    assert info["bit_depth_mismatch"] is True
    assert info["rescaled"] is False
    assert "跳过扣暗" in info["reason"]
