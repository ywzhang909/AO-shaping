"""离线测试：``gs_uniformity_probe`` 接入 transfer 整形相位（不开设备）。

覆盖：
* transfer ``.npy`` / ``.npz`` 来源解析（形状校验、模型/面板两种网格）；
* 旧 GS 契约校验（``best_phase.npy`` + 同目录 ``slm_gs_refine_*.json``）；
* 面板几何与相机侧目标边长的来源标签（实验系统常量 / GS config / 实测）；
* ``config.json``、``flat_reference.npz``、``summary.json`` 的来源块，以及
  用假相机/假 SLM 端到端跑一遍 ``collect``（设备已在 monkeypatch 中替换）。
"""

from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "gs_uniformity_probe.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script():
    """按路径加载被测脚本（``scripts/slm`` 不是稳定的安装包）。"""
    spec = importlib.util.spec_from_file_location("gs_uniformity_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_script()


def _transfer_model() -> np.ndarray:
    """640x640 模型网格（transfer 终稿的形状），数值无物理含义。"""
    grid = np.linspace(0.0, 2.0 * np.pi, 640 * 640, dtype=np.float64)
    return grid.reshape(640, 640)


def _write_transfer_npy(path: Path) -> np.ndarray:
    model = _transfer_model()
    np.save(path, model)
    return model


def _blob_frame(shape: tuple[int, int] = (400, 400)) -> np.ndarray:
    """平滑亮斑（峰值 100，不会触发饱和保护），0 级在中心。"""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    cy, cx = shape[0] // 2, shape[1] // 2
    return 100.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 15.0**2))


def _write_gs_run(run_dir: Path, *, target_side: int = 96) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    phase_path = run_dir / "best_phase.npy"
    np.save(phase_path, np.zeros((probe.SLM_PANEL_H, probe.SLM_PANEL_W)))
    config = {
        "beam_radius_px": 400,
        "pupil_center": [900, 555],
        "target_side": target_side,
    }
    (run_dir / "slm_gs_refine_20260101.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return phase_path


# ---------------------------------------------------------------------------
# transfer 来源解析
# ---------------------------------------------------------------------------


def test_transfer_npy_resolves_to_panel_phase(tmp_path: Path) -> None:
    path = tmp_path / "star_best.npy"
    model = _write_transfer_npy(path)

    source = probe.resolve_phase_source(phase_source="transfer", transfer_phase=path)

    assert source.kind == probe.PHASE_SOURCE_TRANSFER
    assert source.path == path
    assert source.panel_phase.shape == (probe.SLM_PANEL_H, probe.SLM_PANEL_W)
    assert source.model_phase is not None and source.model_phase.shape == model.shape
    assert source.pupil_center_xy == tuple(probe.BEAM_CENTER_PANEL)
    assert source.pupil_radius_px == float(probe.BEAM_RADIUS_PANEL)
    assert source.geometry_source == probe.GEOMETRY_SOURCE_BENCH
    assert source.config_path is None
    assert source.configured_target_side_px is None
    assert len(source.panel_sha256) == 64

    meta = source.metadata()
    assert meta["phase_source"] == "transfer"
    assert meta["source_phase"] == str(path)
    assert meta["source_config"] is None
    assert meta["source_phase_shape"] == [probe.SLM_PANEL_H, probe.SLM_PANEL_W]
    assert meta["source_model_shape"] == [640, 640]
    assert meta["geometry_source"] == probe.GEOMETRY_SOURCE_BENCH


def test_transfer_npz_reads_model_phase_and_matches_npy(tmp_path: Path) -> None:
    npy_path = tmp_path / "star_best.npy"
    model = _write_transfer_npy(npy_path)
    npz_path = tmp_path / "replay_star.npz"
    np.savez(npz_path, model_phase=model, panel_gray=np.zeros((1200, 1920)))

    from_npy = probe.resolve_phase_source(phase_source="transfer", transfer_phase=npy_path)
    from_npz = probe.resolve_phase_source(phase_source="transfer", transfer_phase=npz_path)

    np.testing.assert_array_equal(from_npz.panel_phase, from_npy.panel_phase)
    assert from_npz.path == npz_path


def test_transfer_npz_without_model_phase_rejected(tmp_path: Path) -> None:
    path = tmp_path / "replay_star.npz"
    np.savez(path, panel_gray=np.zeros((1200, 1920)))
    with pytest.raises(ValueError):
        probe.resolve_phase_source(phase_source="transfer", transfer_phase=path)


def test_transfer_missing_file_and_bad_shape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe.resolve_phase_source(
            phase_source="transfer", transfer_phase=tmp_path / "nope.npy"
        )

    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((100, 100)))
    with pytest.raises(ValueError):
        probe.resolve_phase_source(phase_source="transfer", transfer_phase=bad)


# ---------------------------------------------------------------------------
# 旧 GS 契约
# ---------------------------------------------------------------------------


def test_gs_source_keeps_legacy_contract(tmp_path: Path) -> None:
    phase_path = _write_gs_run(tmp_path / "gs_run", target_side=96)

    source = probe.resolve_phase_source(phase_path, phase_source="gs")

    assert source.kind == probe.PHASE_SOURCE_GS
    assert source.path == phase_path
    assert source.panel_phase.shape == (probe.SLM_PANEL_H, probe.SLM_PANEL_W)
    assert source.pupil_center_xy == (900, 555)
    assert source.pupil_radius_px == 400.0
    assert source.configured_target_side_px == 96
    assert source.geometry_source.startswith(probe.GEOMETRY_SOURCE_GS_CONFIG + ":")
    assert source.config_path is not None and source.config_path.name.endswith(".json")
    assert source.model_phase is None
    assert source.metadata()["source_phase"] == str(phase_path)


def test_gs_source_zero_target_side_falls_back_to_measurement(tmp_path: Path) -> None:
    phase_path = _write_gs_run(tmp_path / "gs_run", target_side=0)
    source = probe.resolve_phase_source(phase_path, phase_source="gs")
    assert source.configured_target_side_px is None


def test_gs_source_rejects_wrong_filename(tmp_path: Path) -> None:
    phase_path = _write_gs_run(tmp_path / "gs_run")
    renamed = phase_path.with_name("phase.npy")
    phase_path.rename(renamed)
    with pytest.raises(ValueError):
        probe.resolve_phase_source(renamed, phase_source="gs")


def test_gs_source_requires_config(tmp_path: Path) -> None:
    phase_path = tmp_path / "best_phase.npy"
    np.save(phase_path, np.zeros((probe.SLM_PANEL_H, probe.SLM_PANEL_W)))
    with pytest.raises(FileNotFoundError):
        probe.resolve_phase_source(phase_path, phase_source="gs")


def test_gs_source_requires_phase_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        probe.resolve_phase_source(None, phase_source="gs")


def test_unknown_phase_source_rejected() -> None:
    with pytest.raises(ValueError):
        probe.resolve_phase_source(phase_source="zernike")


# ---------------------------------------------------------------------------
# 目标框边长来源
# ---------------------------------------------------------------------------


def test_target_side_explicit_keeps_source() -> None:
    flat = _blob_frame()
    side, origin = probe.resolve_target_side(
        flat, requested=77, requested_source=probe.TARGET_SIDE_SOURCE_GS_CONFIG
    )
    assert (side, origin) == (77, probe.TARGET_SIDE_SOURCE_GS_CONFIG)

    with pytest.raises(ValueError):
        probe.resolve_target_side(flat, requested=0)


def test_target_side_measured_from_flat_frame() -> None:
    flat = np.zeros((250, 250))
    flat[100:140, 100:140] = 100.0
    side, origin = probe.resolve_target_side(flat, requested=None)
    assert side == 40
    assert origin == probe.TARGET_SIDE_SOURCE_MEASURED

    with pytest.raises(RuntimeError):
        probe.resolve_target_side(np.zeros((250, 250)), requested=None)


# ---------------------------------------------------------------------------
# CLI 位置参数
# ---------------------------------------------------------------------------


def test_cli_paths_transfer_takes_output_dir_only() -> None:
    out = Path("out_dir")
    assert probe.resolve_cli_paths([out], phase_source="transfer") == (None, out)
    with pytest.raises(ValueError):
        probe.resolve_cli_paths([out, out], phase_source="transfer")


def test_cli_paths_gs_keeps_two_positionals() -> None:
    phase, out = Path("best_phase.npy"), Path("out_dir")
    assert probe.resolve_cli_paths([phase, out], phase_source="gs") == (phase, out)
    with pytest.raises(ValueError):
        probe.resolve_cli_paths([out], phase_source="gs")


# ---------------------------------------------------------------------------
# summary 来源块（可与 GS 来源区分）
# ---------------------------------------------------------------------------


def _triple_rows() -> list[dict]:
    base = {"phase_gradient_rms_rad_per_pixel": 0.1, "sigma_px": 2.0}
    return [
        {**base, "label": "baseline_before", "cv_no_zero": 0.50},
        {**base, "label": "smoothed", "cv_no_zero": 0.45},
        {**base, "label": "baseline_after", "cv_no_zero": 0.50},
    ]


def test_summary_provenance_marks_unknown_without_config() -> None:
    assert probe.summary_provenance(None)["phase_source"] == "unknown"


def test_build_summary_carries_transfer_provenance() -> None:
    config = {
        "phase_source": "transfer",
        "source_phase": "data/debug/slm_transfer/star_best.npy",
        "source_phase_sha256": "a" * 64,
        "geometry_source": probe.GEOMETRY_SOURCE_BENCH,
        "target_side_px": 40,
        "target_side_source": probe.TARGET_SIDE_SOURCE_MEASURED,
        "unrelated": "dropped",
    }
    summary = probe.build_summary(_triple_rows(), probe.summary_provenance(config))

    assert summary["phase_source"] == "transfer"
    assert summary["source_phase"] == "data/debug/slm_transfer/star_best.npy"
    assert summary["geometry_source"] == probe.GEOMETRY_SOURCE_BENCH
    assert "unrelated" not in summary
    assert len(summary["pairs"]) == 1
    assert summary["pairs"][0]["delta_cv_no_zero"] == pytest.approx(-0.05)
    assert summary["by_sigma"]["2.0"]["mean_delta_cv_no_zero"] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# 端到端 collect（假设备）
# ---------------------------------------------------------------------------


class _FakeCam:
    """只保留 collect 用到的相机接口。"""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.exposure = 0.02
        self.window: tuple | None = None

    def reset_exposure_time(self, ms: float) -> float:
        self.exposure = float(ms)
        return self.exposure

    def get_exposure_time_ms(self) -> float:
        return self.exposure

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        return self.frame.copy()

    def reset_window(self, center=None, size=None) -> None:
        self.window = (center, size)

    def __enter__(self) -> "_FakeCam":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _FakeSantec:
    """假 SLM：面板尺寸正确，且已占用一个 memory slot。"""

    Panel_Res = (1920, 1200)

    def __init__(self, **kwargs) -> None:
        pass

    def __enter__(self) -> "_FakeSantec":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def get_displayed_memory_number(self) -> int:
        return 126


def _patch_devices(monkeypatch: pytest.MonkeyPatch, frame: np.ndarray) -> None:
    monkeypatch.setattr(probe, "display_and_average", lambda *a, **k: frame.copy())
    monkeypatch.setattr(probe, "preserve_slm_display", lambda slm: nullcontext())
    monkeypatch.setattr(probe, "configured_daheng_exposure_limit", lambda: 5.0)
    monkeypatch.setattr(
        "ao_shaping.drivers.ccd.common.create_camera", lambda *a, **k: _FakeCam(frame)
    )
    monkeypatch.setattr("ao_shaping.drivers.slm.santec.Santec", _FakeSantec)


def test_collect_transfer_records_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_devices(monkeypatch, _blob_frame())
    phase_path = tmp_path / "star_best.npy"
    _write_transfer_npy(phase_path)
    out_dir = tmp_path / "run_transfer"

    rows = probe.collect(
        output_dir=out_dir,
        phase_source="transfer",
        transfer_phase=phase_path,
        sigmas=(2.0,),
        repeats=1,
        exposure_ms=0.5,
        frames=1,
    )

    assert [row["label"] for row in rows] == [
        "baseline_before",
        "smoothed",
        "baseline_after",
    ]
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["phase_source"] == "transfer"
    assert config["source_phase"] == str(phase_path)
    assert config["source_config"] is None
    assert config["source_model_shape"] == [640, 640]
    assert config["geometry_source"] == probe.GEOMETRY_SOURCE_BENCH
    assert config["source_pupil_center_xy"] == list(probe.BEAM_CENTER_PANEL)
    assert config["source_pupil_radius_px"] == float(probe.BEAM_RADIUS_PANEL)
    assert config["target_side_source"] == probe.TARGET_SIDE_SOURCE_MEASURED
    assert config["target_side_px"] > 0
    assert len(config["source_phase_sha256"]) == 64

    saved = np.load(out_dir / "phase_sigma_2.npy", allow_pickle=False)
    assert saved.shape == (probe.SLM_PANEL_H, probe.SLM_PANEL_W)

    flat = np.load(out_dir / "flat_reference.npz", allow_pickle=False)
    assert str(flat["phase_source"]) == "transfer"
    assert str(flat["source_phase"]) == str(phase_path)
    assert str(flat["geometry_source"]) == probe.GEOMETRY_SOURCE_BENCH
    assert int(flat["target_side_px"]) == config["target_side_px"]

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["phase_source"] == "transfer"
    assert summary["source_phase"] == str(phase_path)
    assert summary["target_side_source"] == probe.TARGET_SIDE_SOURCE_MEASURED
    assert len(summary["pairs"]) == 1


def test_collect_gs_summary_is_distinguishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_devices(monkeypatch, _blob_frame())
    phase_path = _write_gs_run(tmp_path / "gs_run", target_side=96)
    out_dir = tmp_path / "run_gs"

    probe.collect(
        phase_path,
        out_dir,
        phase_source="gs",
        sigmas=(2.0,),
        repeats=1,
        exposure_ms=0.5,
        frames=1,
    )

    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["phase_source"] == "gs"
    assert config["target_side_px"] == 96
    assert config["target_side_source"] == probe.TARGET_SIDE_SOURCE_GS_CONFIG
    assert config["geometry_source"].startswith(probe.GEOMETRY_SOURCE_GS_CONFIG + ":")

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["phase_source"] == "gs"
    assert summary["source_phase"] == str(phase_path)
