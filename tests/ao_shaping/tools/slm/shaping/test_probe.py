"""``probe.py`` 离线判据采集与 ``shaping`` CLI 的单测（免设备）。

覆盖：空/不存在目录的全 None 与 notes 说明；frozen_parameters.json →
has_checkpoint；session/journal 的几何/方向字段映射；overrides 覆盖生效与非法
键报错；``--set snr=1.0`` 经 CLI 得 no_go 且退出码 2；CLI 默认打印格式、
``--json`` 可解析且含 route_id、``--out`` 写盘。

全程不触设备、不 import torch；同时在末尾断言 ``probe`` 模块可导入不拉 torch。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ao_shaping.tools.slm.shaping.__main__ import main
from ao_shaping.tools.slm.shaping.probe import (
    build_report,
    coerce_value,
    parse_override_pair,
    parse_overrides,
)


# ---------------------------------------------------------------------------
# build_report：离线判据提取
# ---------------------------------------------------------------------------
def test_empty_dir_yields_all_none_with_notes(tmp_path: Path):
    report = build_report(tmp_path)
    assert report.intent == "auto"
    assert report.shape == "square"
    assert report.has_checkpoint is False
    assert report.has_seed_run is False
    for field in (
        "light_ok",
        "geometry_correlation",
        "zero_shift_px",
        "direction_agreement",
        "focus_changed",
        "model_anchor_error",
        "snr",
        "noise_floor_ok",
    ):
        assert getattr(report, field) is None, field
    assert report.geometry_problems == ()
    assert any("frozen" in n.lower() for n in report.notes)


def test_nonexistent_dir_yields_all_none_with_notes(tmp_path: Path):
    missing = tmp_path / "no_such_session"
    report = build_report(missing)
    assert report.has_checkpoint is False
    assert any("不存在" in n for n in report.notes)


def test_fit_frozen_with_ramp_matrix_sets_checkpoint(tmp_path: Path):
    (tmp_path / "fit").mkdir()
    frozen = {
        "ramp_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "grid_size": 640,
    }
    (tmp_path / "fit" / "frozen_parameters.json").write_text(
        json.dumps(frozen), encoding="utf-8"
    )
    report = build_report(tmp_path)
    assert report.has_checkpoint is True
    assert any("ramp_matrix" in n for n in report.notes)


def test_initial_frozen_without_ramp_matrix_notes(tmp_path: Path):
    (tmp_path / "initial").mkdir()
    (tmp_path / "initial" / "frozen_parameters.json").write_text(
        json.dumps({"grid_size": 640}), encoding="utf-8"
    )
    report = build_report(tmp_path)
    assert report.has_checkpoint is True
    assert any("不含 ramp_matrix" in n for n in report.notes)


def test_explicit_checkpoint_path_exists_flag(tmp_path: Path):
    fp = tmp_path / "cp.json"
    fp.write_text("{}", encoding="utf-8")
    report = build_report(None, checkpoint=fp)
    assert report.has_checkpoint is True


def test_explicit_checkpoint_missing_flag(tmp_path: Path):
    fp = tmp_path / "cp.json"
    report = build_report(None, checkpoint=fp)
    assert report.has_checkpoint is False


def test_seed_run_existence(tmp_path: Path):
    seed = tmp_path / "seed.json"
    seed.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing.json"
    assert build_report(None, seed_run=seed).has_seed_run is True
    assert build_report(None, seed_run=missing).has_seed_run is False


def test_light_check_maps_light_ok_and_snr(tmp_path: Path):
    (tmp_path / "capture").mkdir()
    lc = {
        "ok": True,
        "exposure_ms": 0.05,
        "peak_counts": 40.5,
        "noise_floor_counts": 4.0,
    }
    (tmp_path / "capture" / "light_check.json").write_text(
        json.dumps(lc), encoding="utf-8"
    )
    report = build_report(tmp_path)
    assert report.light_ok is True
    assert report.snr == pytest.approx(40.5 / 4.0)
    assert report.noise_floor_ok is None  # light_check 不提供 noise_floor_ok


def test_journal_geometry_direction_mapping(tmp_path: Path):
    journal = [
        {
            "stage": "geometry",
            "geometry_correlation": 0.71,
            "direction_agreement": -0.3,
            "zero_shift_px": 3.2,
            "geometry_problems": ["window_overlap"],
        },
        {
            "stage": "light",
            "noise_floor_ok": True,
        },
    ]
    (tmp_path / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    report = build_report(tmp_path)
    assert report.geometry_correlation == 0.71
    assert report.direction_agreement == -0.3
    assert report.zero_shift_px == 3.2
    assert report.geometry_problems == ("window_overlap",)
    assert report.noise_floor_ok is True
    assert any("geometry" in n for n in report.notes)


def test_session_json_correlation_alias(tmp_path: Path):
    session = {"correlation": 0.85, "focus_changed": False}
    (tmp_path / "session.json").write_text(json.dumps(session), encoding="utf-8")
    report = build_report(tmp_path)
    assert report.geometry_correlation == 0.85
    assert report.focus_changed is False


def test_corrupt_journal_keeps_none(tmp_path: Path):
    (tmp_path / "journal.json").write_text("{ not json", encoding="utf-8")
    report = build_report(tmp_path)
    assert report.geometry_correlation is None
    assert report.direction_agreement is None
    assert any("无法解析" in n for n in report.notes)


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------
def test_overrides_apply(tmp_path: Path):
    report = build_report(tmp_path, overrides={"snr": 1.0, "noise_floor_ok": False})
    assert report.snr == 1.0
    assert report.noise_floor_ok is False


def test_overrides_unknown_key_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="snrzzz"):
        build_report(tmp_path, overrides={"snrzzz": 1.0})


def test_coerce_value():
    assert coerce_value("1.0") == 1.0
    assert coerce_value("1") == 1
    assert coerce_value("true") == True  # noqa: E712
    assert coerce_value("[1,2]") == [1, 2]
    assert coerce_value("null") is None
    assert coerce_value("square") == "square"


def test_parse_override_pair_invalid():
    with pytest.raises(ValueError, match="key=value"):
        parse_override_pair("nodoublecolon")


def test_parse_overrides_multi():
    out = parse_overrides(["snr=1.0", "geometry_correlation=0.9"])
    assert out == {"snr": 1.0, "geometry_correlation": 0.9}


# ---------------------------------------------------------------------------
# CLI 端到端
# ---------------------------------------------------------------------------
def test_cli_default_prints_plan_and_confirmation(tmp_path: Path, capsys):
    code = main(["--session-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "路线" in out
    assert "需人工确认" in out
    assert code == 0


def test_cli_set_snr_low_is_no_go_exitcode_2(tmp_path: Path, capsys):
    code = main(["--set", "snr=1.0"])
    out = capsys.readouterr().out
    # default 路线应是 no_go
    assert code == 2
    assert "不整形" in out


def test_cli_no_go_json_route_id(tmp_path: Path, capsys):
    code = main(["--set", "noise_floor_ok=false", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["route_id"] == "no_go"
    assert code == 2


def test_cli_json_output_parses_with_route_id(tmp_path: Path, capsys):
    code = main(["--session-dir", str(tmp_path), "--json", "--shape", "triangle"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "route_id" in data
    assert code == 0


def test_cli_out_writes_plan_file(tmp_path: Path, capsys):
    out_path = tmp_path / "sub" / "plan.json"
    main(["--session-dir", str(tmp_path), "--out", str(out_path), "--json"])
    capsys.readouterr()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "route_id" in data


def test_cli_overrides_parse_and_apply(tmp_path: Path, capsys):
    fp = tmp_path / "cp.json"
    fp.write_text("{}", encoding="utf-8")
    code = main(
        [
            "--checkpoint",
            str(fp),
            "--set",
            "snr=10.0",
            "--set",
            "noise_floor_ok=true",
            "--set",
            "has_seed_run=true",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "route_id" in data
    assert code == 0
    # snr=10.0 + checkpoint + seed → 不应落 3. 的 cold_start，而应更低优先级
    assert data["route_id"] != "cold_start"


# ---------------------------------------------------------------------------
# 无设备 / 无 torch 守卫
# ---------------------------------------------------------------------------
def test_probe_module_does_not_import_torch():
    """探测采集模块不应在导入时拉入 torch（与 router 同契约）。

    在干净子进程里导入 probe 包，检查 torch 未进入 sys.modules 且可 import。
    """
    code = (
        "import sys; "
        "import ao_shaping.tools.slm.shaping.probe as probe; "
        "print(probe.build_report) if hasattr(probe, 'build_report') else None; "
        "print('torch_imported=' + repr('torch' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "torch_imported=False" in result.stdout
