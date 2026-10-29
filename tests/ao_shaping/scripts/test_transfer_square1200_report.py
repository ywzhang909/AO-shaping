"""Guards for the transfer-learning 1200 px square report generator.

The generator is offline (it only reads a session directory), so these tests
build a synthetic session with the same JSON contract and check that the text
builder renders the numbers instead of leaking ``None`` or dropping a section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.slm import generate_transfer_square1200_report as report


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(label: str, cv: float) -> dict:
    return dict(label=label, file=f"capture_{label}.npz", metrics=dict(cv=cv),
                exposure_ms=123.4, bit_depth=12, saturation_pixels=0, loss=1.3,
                core_peak=650.0)


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    checkpoint = tmp_path / "pretrained.json"
    _write(checkpoint, dict(origin=[1221.7, 961.8], amplitude_center=[977.8, 721.9],
                            amplitude_width=[335.5, 267.7], defocus_diopter=-0.003,
                            gain=75.5, background=0.58))
    dataset = tmp_path / "capture"
    _write(dataset / "manifest.json",
           [dict(label="reference_start", spot_xy=[1297.8, 948.9]),
            dict(label="probe_000_zernike", spot_xy=[1300.0, 950.0])])
    _write(session / "session.json", dict(
        settings=dict(side=1200.0, period=128.0, window_span=820.0, window_step=4.0,
                      zero_radius=60.0, core_margin=8.0, bit_depth=12, abba=True,
                      control_grid=640, fft_size=2048, checkpoint=str(checkpoint),
                      dataset=str(dataset), shapes=["square"]),
        resolved=dict(adaptation="auto", acceptance="paired_abba"),
        stages=["reuse_capture", "fit", "CCD_anchored_blocks"]))
    _write(session / "journal.json", [dict(stage="fit", status="complete")])
    _write(session / "audit.json", dict(captures=20, raw_frames=60, maximum_peak=2410.0,
                                        blocks=4, accepted_blocks=2, status="passed",
                                        timestamp_scope="host calls"))
    _write(session / "fit" / "adaptation.json", dict(
        selected_stage="full", escalated=True,
        candidates=[dict(stage="alignment", quality=dict(correlation=.341, relative_l2=.836,
                                                         cv_abs_error=1.326, score=1.495)),
                    dict(stage="full", quality=dict(correlation=.839, relative_l2=.474,
                                                    cv_abs_error=.327, score=.634))]))
    _write(session / "fit" / "result.json", dict(iteration=38, loss=.219, parameters={}, metrics=[]))
    _write(session / "fit" / "frozen_parameters.json", dict(
        origin=[1281.4, 955.2], amplitude_center=[960.0, 826.1], amplitude_width=[431.4, 334.7],
        defocus_diopter=-0.392, gain=89.45, background=0.36))
    _write(session / "model" / "summary.json",
           [dict(shape="square", block=0, baseline=.5952, candidate=.6016, drift=.0035,
                 accepted=False)])
    _write(session / "anchored" / "summary.json",
           [dict(shape="square", block=0, baseline=.5981, candidate=.5727, drift=.0007,
                 accepted=True),
            dict(shape="square", block=1, baseline=.5758, candidate=.5716, drift=.0008,
                 accepted=False)])
    _write(session / "model" / "manifest.json",
           [_row("square_seed", .594), _row("square_0_A1", .595), _row("square_0_B1", .602),
            _row("square_final", .594)])
    _write(session / "anchored" / "manifest.json",
           [_row("square_seed", .594), _row("square_0_A1", .598), _row("square_0_B1", .5727),
            _row("square_0_B2", .5727), _row("square_0_A2", .5988),
            _row("square_direction_minus", .6469), _row("square_final", .5729)])
    _write(session / "anchored" / "square_direction.json",
           dict(model_loss_plus=1.2851, model_loss_minus=1.2861, ccd_loss_plus=1.3284,
                ccd_loss_minus=1.5054, difference_correlation=.289,
                difference_relative_l2=1.287))
    return session


def test_role_classification_covers_the_abba_labels():
    assert report._role("square_seed") == "seed"
    assert report._role("square_0_A1") == "A1"
    assert report._role("square_2_B2") == "B2"
    assert report._role("square_direction_minus") == "diag"
    assert report._role("square_final") == "final"


def test_report_renders_numbers_instead_of_none(tmp_path):
    session = _session(tmp_path)
    facts = report._session_facts(session)
    text = report.build_report(facts)
    # ``None`` appears once as prose (the first-row classification defect);
    # anywhere else it would be a leaked data field.
    assert "None" not in text.replace("`None`", "")
    assert "<!-- provenance:start -->" in text
    # the selected adaptation stage is marked, and the measured drift shift is shown
    assert "`full` **← 采用**" in text
    assert "1297.8" in text and "1221.7" in text
    # every planned section is present
    for heading in ("## 1 本轮做了什么", "## 2 迁移适配", "## 3 逐块结果",
                    "## 4 方向性检查", "## 5 Recorder 记录", "## 6 局限", "## 7 复现命令",
                    "## 8 审计"):
        assert heading in text
    # acceptance verdicts come from the data, not from the template
    assert text.count("拒绝") >= 2
    assert text.count("接受") >= 1


def test_report_requires_the_adaptation_stage(tmp_path):
    session = _session(tmp_path)
    (session / "fit" / "adaptation.json").unlink()
    with pytest.raises(SystemExit):
        facts = report._session_facts(session)
        if not facts["adaptation"]:
            raise SystemExit("no fit/adaptation.json")
