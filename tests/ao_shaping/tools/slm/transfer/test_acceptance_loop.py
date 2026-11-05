"""验收阈值 + 重训回路 (免设备)。

覆盖三件事:

1. ``fit.load_dataset(train_all=True)`` 把验证行折进训练集，而默认行为一字不变;
2. ``workflow.acceptance_verdict`` 只读文件给出判决，缺文件也不抛;
3. ``run_session`` 在 ``acceptance_enabled=False`` 时的阶段名与改动前一致,
   打开后按轮次重跑并把验证行折进训练。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import fit as fit_module
from ao_shaping.tools.slm.transfer import workflow as workflow_module
from ao_shaping.tools.slm.transfer.workflow import (
    TransferSession,
    acceptance_verdict,
    run_session,
)


# ---------------------------------------------------------------------------
# 1. load_dataset: train_all 只在显式打开时改变划分
# ---------------------------------------------------------------------------


def _tiny_dataset(tmp_path, *, size=8):
    """Ramps + reference + 一对 train/validation 行 (与老测试同一套口径)。"""
    yy, xx = np.mgrid[:40, :40]
    base = (5 + 60 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / 30)).astype(np.uint8)
    rows = []

    def add(label, split, spot, phase_value):
        name = f"sample_{len(rows):04d}.npz"
        raw = np.stack([base, base])
        phase = np.full((size, size), phase_value, np.float32)
        np.savez_compressed(
            tmp_path / name,
            raw_frames=raw,
            frame=raw.mean(0),
            model_phase=phase,
            panel_gray=np.zeros((1200, 1920), np.uint16),
        )
        rows.append(
            dict(
                file=name,
                label=label,
                split=split,
                exposure_ms=1.0,
                peak=float(base.max()),
                spot_xy=list(spot),
                exposure_setting=dict(status="ok"),
            )
        )

    add("reference_start", "calibration", (20.0, 20.0), 0.0)
    for axis, shift in (("x", (6.0, 0.0)), ("y", (0.0, 6.0))):
        add(f"ramp_{axis}_64_1", "calibration", (20.0 + shift[0], 20.0 + shift[1]), 0.0)
        add(f"ramp_{axis}_64_-1", "calibration", (20.0 - shift[0], 20.0 - shift[1]), 0.0)
    add("probe_000_a", "train", (20.0, 20.0), 0.1)
    add("probe_000_b", "validation", (20.0, 20.0), 0.1)
    (tmp_path / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    return rows


def test_load_dataset_keeps_the_split_by_default(tmp_path):
    _tiny_dataset(tmp_path)
    selected, _phases, _images, _xy, _matrix, _ramps, train = fit_module.load_dataset(
        tmp_path, 8, 10
    )
    assert [row["split"] for row in selected if row["split"] != "calibration"] == [
        "train",
        "validation",
    ]
    assert train.tolist() == [True, False]


def test_load_dataset_train_all_folds_the_validation_rows_in(tmp_path):
    _tiny_dataset(tmp_path)
    selected, _phases, _images, _xy, _matrix, _ramps, train = fit_module.load_dataset(
        tmp_path, 8, 10, train_all=True
    )
    assert len(selected) == 2
    assert train.all()
    assert train.tolist() == [True, True]


def test_build_fit_model_forwards_train_all_only_when_set(tmp_path, monkeypatch):
    """``train_all=False`` 的历史调用形状必须一字不差 (老测试替换过 load_dataset)。"""
    _tiny_dataset(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            dict(
                defocus_diopter=0.0,
                origin=[20.0, 20.0],
                amplitude_center=[960.0, 600.0],
                amplitude_width=[400.0, 300.0],
                residual_noll_radians={str(j): 0.0 for j in range(5, 12)},
                gain=1.0,
                background=0.0,
                sigma_ccd_px=0.0,
                ramp_matrix=[[0.0, 1.0], [1.0, 0.0]],
                grid_size=8,
                fft_size=64,
                pixel_samples=3,
            )
        ),
        encoding="utf-8",
    )
    seen: dict = {}

    class _Reached(Exception):
        """测试到这里就停: 只验证接线, 不跑真正的拟合。"""

    def fake_load(directory, size, span, samples=None):
        seen["kwargs"] = dict(samples=samples)
        raise _Reached

    monkeypatch.setattr(fit_module, "load_dataset", fake_load)
    with pytest.raises(_Reached):
        fit_module.build_fit_model(tmp_path, checkpoint, span=10)
    assert "train_all" not in seen["kwargs"]

    seen.clear()

    def fake_load_all(directory, size, span, samples=None, *, train_all=False):
        seen["kwargs"] = dict(samples=samples, train_all=train_all)
        raise _Reached

    monkeypatch.setattr(fit_module, "load_dataset", fake_load_all)
    with pytest.raises(_Reached):
        fit_module.build_fit_model(tmp_path, checkpoint, span=10, train_all=True)
    assert seen["kwargs"]["train_all"] is True


# ---------------------------------------------------------------------------
# 2. acceptance_verdict: 只读文件, 缺文件不抛
# ---------------------------------------------------------------------------


def _write_fit(session_dir, *, correlation, relative_l2, split="validation"):
    """写出一份 fit 的 result.json + provenance.json (判决的读取依据)。"""
    fit_dir = Path(session_dir) / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(label="probe_000_a", split="train"),
        dict(label="probe_000_b", split=split),
    ]
    (fit_dir / "provenance.json").write_text(json.dumps(dict(rows=rows)), encoding="utf-8")
    metrics = [
        dict(correlation=0.5, relative_l2=0.9, cv_abs_error=0.1),
        dict(correlation=correlation, relative_l2=relative_l2, cv_abs_error=0.1),
    ]
    (fit_dir / "result.json").write_text(json.dumps(dict(metrics=metrics)), encoding="utf-8")


def _write_session(session_dir, *, lower=0.85, upper=0.25, enabled=True):
    (Path(session_dir) / "session.json").write_text(
        json.dumps(
            dict(
                settings=dict(
                    acceptance_enabled=enabled,
                    acceptance_metric="quality",
                    acceptance_lower=lower,
                    acceptance_upper=upper,
                )
            )
        ),
        encoding="utf-8",
    )


def test_acceptance_verdict_reports_below_the_bound_as_not_accepted(tmp_path):
    _write_session(tmp_path)
    _write_fit(tmp_path, correlation=0.40, relative_l2=0.60)
    verdict = acceptance_verdict(tmp_path)
    assert verdict["metric"] == "quality"
    assert verdict["lower"] == 0.85
    assert verdict["upper"] == 0.25
    assert verdict["measured_correlation"] == pytest.approx(0.40)
    assert verdict["measured_relative_l2"] == pytest.approx(0.60)
    assert verdict["accepted"] is False


def test_acceptance_verdict_reports_above_the_bound_as_accepted(tmp_path):
    _write_session(tmp_path)
    _write_fit(tmp_path, correlation=0.93, relative_l2=0.12)
    verdict = acceptance_verdict(tmp_path)
    assert verdict["accepted"] is True
    assert verdict["rounds"] == 1


def test_acceptance_verdict_never_raises_on_a_missing_directory(tmp_path):
    verdict = acceptance_verdict(tmp_path / "does-not-exist")
    assert verdict["accepted"] is False
    assert verdict["measured_correlation"] is None
    assert verdict["measured_relative_l2"] is None
    assert verdict["rounds"] == 0


def test_acceptance_verdict_never_raises_on_a_half_written_session(tmp_path):
    (tmp_path / "session.json").write_text("{not json", encoding="utf-8")
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    (fit_dir / "result.json").write_text(json.dumps(dict(metrics=[])), encoding="utf-8")
    verdict = acceptance_verdict(tmp_path)
    assert verdict["accepted"] is False


def test_acceptance_verdict_prefers_the_recorded_rounds(tmp_path):
    _write_session(tmp_path)
    (tmp_path / "journal.json").write_text("[]", encoding="utf-8")
    (tmp_path / "rounds.json").write_text(
        json.dumps(
            [
                dict(round=0, measured_correlation=0.4, measured_relative_l2=0.6, accepted=False),
                dict(round=1, measured_correlation=0.9, measured_relative_l2=0.2, accepted=True),
            ]
        ),
        encoding="utf-8",
    )
    verdict = acceptance_verdict(tmp_path)
    assert verdict["rounds"] == 2
    assert verdict["measured_correlation"] == pytest.approx(0.9)
    assert verdict["accepted"] is True


# ---------------------------------------------------------------------------
# 3. run_session: 阶段名与回路行为
# ---------------------------------------------------------------------------


def _fake_fit(seen, *, correlation=0.95, relative_l2=0.10):
    """替身 fit: 记录参数并留下判决要读的 result.json / provenance.json。"""

    def _fit(args):
        seen.setdefault("fit", []).append(args)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "frozen_parameters.json").write_text(
            json.dumps(
                dict(
                    ramp_matrix=[[1.0, 0.0], [0.0, 1.0]],
                    grid_size=8,
                    fft_size=16,
                    pixel_samples=3,
                )
            ),
            encoding="utf-8",
        )
        rows = [
            dict(label="probe_000_a", split="train"),
            dict(label="probe_000_b", split="validation"),
        ]
        (args.out / "provenance.json").write_text(
            json.dumps(dict(rows=rows)), encoding="utf-8"
        )
        metrics = [
            dict(correlation=0.5, relative_l2=0.9, cv_abs_error=0.1),
            dict(correlation=correlation, relative_l2=relative_l2, cv_abs_error=0.1),
        ]
        (args.out / "result.json").write_text(
            json.dumps(dict(metrics=metrics)), encoding="utf-8"
        )

    return _fit


def _patch_session(monkeypatch, seen, **fit_kwargs):
    from ao_shaping.tools.slm.transfer import (
        artifacts,
        bootstrap,
        capture,
        direction,
        fit,
        summary,
        validate,
    )

    payload = {
        "sample_0000.npz": dict(
            model_phase=np.zeros((8, 8), np.float32),
            raw_frames=np.zeros((2, 8, 8), np.uint8),
        )
    }

    def fake_capture(args, *, on_frame=None):
        seen["capture"] = args
        rows = [
            dict(
                label="reference_start",
                file="sample_0000.npz",
                split="calibration",
                exposure_ms=1.0,
                peak=10.0,
                spot_xy=[4.0, 4.0],
                exposure_setting=dict(status="ok"),
            )
        ]
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
        (args.out / "config.json").write_text(json.dumps(dict(grid=8)))
        return payload

    monkeypatch.setattr(capture, "capture", fake_capture)
    monkeypatch.setattr(
        bootstrap, "bootstrap", lambda dataset: (dict(grid_size=8), dict(status="nominal"))
    )
    monkeypatch.setattr(fit, "fit", _fake_fit(seen, **fit_kwargs))
    monkeypatch.setattr(
        validate, "run", lambda args: seen.setdefault("validate", []).append(args)
    )
    monkeypatch.setattr(artifacts, "audit_run", lambda run: dict(status="passed"))
    monkeypatch.setattr(summary, "summarize", lambda run, out: None)
    monkeypatch.setattr(
        direction, "analyze", lambda run, out: seen.setdefault("direction", []).append(run)
    )
    return payload


def _session(tmp_path, name, **over):
    return TransferSession(
        tmp_path / name,
        shapes=("square",),
        count=6,
        frames=3,
        iterations=1,
        blocks=1,
        updates=1,
        **over,
    )


def test_run_session_without_the_gate_keeps_the_historical_stage_names(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen)
    names: list[str] = []
    run_session(
        _session(tmp_path, "single"),
        on_stage=lambda name, status: names.append(name) if status == "running" else None,
    )
    journal = json.loads((tmp_path / "single" / "journal.json").read_text(encoding="utf-8"))
    started = [entry["stage"] for entry in journal]
    # shapes=("square",) 不开标定, 所以没有 exposure/blur 三段。
    assert started == [
        "capture",
        "nominal_initialization",
        "fit",
        "CCD_model_blocks",
        "CCD_anchored_blocks",
        "raw_data_audit",
        "summary",
    ]
    assert names == started
    # 关掉闸门时就是单轮: 只有一次拟合。
    assert len(seen["fit"]) == 1
    assert getattr(seen["fit"][0], "train_all", False) is False
    assert not (tmp_path / "single" / "rounds.json").exists()


def test_run_session_with_the_gate_stops_after_the_first_accepted_round(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen, correlation=0.95, relative_l2=0.10)
    run_session(_session(tmp_path, "accepted", acceptance_enabled=True))
    journal = json.loads((tmp_path / "accepted" / "journal.json").read_text(encoding="utf-8"))
    started = [entry["stage"] for entry in journal]
    assert "fit" in started
    assert not [name for name in started if name.startswith("acceptance_round_")]
    assert len(seen["fit"]) == 1
    rounds = json.loads((tmp_path / "accepted" / "rounds.json").read_text(encoding="utf-8"))
    assert len(rounds) == 1
    assert rounds[0]["accepted"] is True
    verdict = acceptance_verdict(tmp_path / "accepted")
    assert verdict["accepted"] is True
    assert verdict["rounds"] == 1


def test_run_session_retrains_with_the_validation_data_folded_in(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen, correlation=0.40, relative_l2=0.60)
    config = _session(
        tmp_path, "looped", acceptance_enabled=True, acceptance_max_rounds=3
    )
    run_session(config)
    journal = json.loads((tmp_path / "looped" / "journal.json").read_text(encoding="utf-8"))
    started = [entry["stage"] for entry in journal]
    # 第一轮用历史名字, 之后的轮次带 acceptance_round_N 前缀。
    assert "fit" in started
    assert "acceptance_round_1_fit" in started
    assert "acceptance_round_2_fit" in started
    assert "acceptance_round_1_CCD_anchored_blocks" in started
    # summary 只在回路结束后跑一次。
    assert started.count("summary") == 1
    # 第一轮保持划分, 之后把验证行折进训练。
    assert getattr(seen["fit"][0], "train_all", False) is False
    assert all(getattr(args, "train_all", False) for args in seen["fit"][1:])
    assert len(seen["fit"]) == 3
    rounds = json.loads((tmp_path / "looped" / "rounds.json").read_text(encoding="utf-8"))
    assert [record["round"] for record in rounds] == [0, 1, 2]
    assert all(record["accepted"] is False for record in rounds)
    verdict = acceptance_verdict(tmp_path / "looped")
    assert verdict["accepted"] is False
    assert verdict["rounds"] == 3


def test_run_session_honours_the_stop_callback_between_rounds(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen, correlation=0.40, relative_l2=0.60)
    config = _session(
        tmp_path, "stopped", acceptance_enabled=True, acceptance_max_rounds=5
    )
    run_session(config, should_stop=lambda: True)
    # Stop 在第一轮之后就生效: 只有一次拟合, 而且不报错。
    assert len(seen["fit"]) == 1
    rounds = json.loads((tmp_path / "stopped" / "rounds.json").read_text(encoding="utf-8"))
    assert len(rounds) == 1


def test_run_session_keeps_going_when_the_stop_callback_raises(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen, correlation=0.40, relative_l2=0.60)

    def boom():
        raise RuntimeError("stop callback exploded")

    config = _session(
        tmp_path, "boom", acceptance_enabled=True, acceptance_max_rounds=2
    )
    run_session(config, should_stop=boom)
    assert len(seen["fit"]) == 2


def test_run_session_respects_the_round_cap(tmp_path, monkeypatch):
    seen: dict = {}
    _patch_session(monkeypatch, seen, correlation=0.40, relative_l2=0.60)
    config = _session(
        tmp_path, "capped", acceptance_enabled=True, acceptance_max_rounds=2
    )
    run_session(config)
    assert len(seen["fit"]) == 2


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [
        dict(acceptance_max_rounds=0),
        dict(acceptance_lower=1.5),
        dict(acceptance_lower=-0.1),
        dict(acceptance_upper=0.0),
        dict(acceptance_upper=-1.0),
        dict(acceptance_metric="nonsense"),
    ],
)
def test_invalid_acceptance_settings_are_rejected(tmp_path, over):
    config = TransferSession(tmp_path / "invalid", **over)
    with pytest.raises(ValueError):
        config.validate()


def test_the_default_session_keeps_the_gate_off(tmp_path):
    config = TransferSession(tmp_path / "default")
    assert config.acceptance_enabled is False
    assert config.acceptance_metric == "quality"
    assert config.acceptance_lower == 0.85
    assert config.acceptance_upper == 0.25
    assert config.acceptance_max_rounds == 8


def test_the_plan_reports_the_acceptance_mode(tmp_path):
    config = TransferSession(tmp_path / "plan")
    assert config.plan()["resolved"]["acceptance"] == "model_only"
    looped = TransferSession(tmp_path / "looped", acceptance_enabled=True)
    assert looped.plan()["resolved"]["acceptance"] == "looped"


def test_workflow_module_exposes_the_verdict_helper():
    assert workflow_module.acceptance_verdict is acceptance_verdict
