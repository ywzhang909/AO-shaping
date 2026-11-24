"""Offline tests for the preregistered S1b content-axis capture script."""

from __future__ import annotations

import argparse
import ast
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = importlib.import_module("scripts.slm.content_axis_2x2_capture_20261124")

#: Repo-wide default 2π scale (``utils/slm/gray_scale.DEFAULT_MAX_GRAYSCALE``).
MAX_GRAY_DEFAULT = 1023


def _plan_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "exposure_ms": SCRIPT.DEFAULT_EXPOSURE_MS,
        "bit_depth": SCRIPT.BIT_DEPTH_REQUIRED,
        "slm_number": SCRIPT.DEFAULT_SLM_NUMBER,
        "cam_type": SCRIPT.DEFAULT_CAM_TYPE,
        "cam_id": SCRIPT.DEFAULT_CAM_ID,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_content_b_is_exact_row_zero_pi_step_for_both_periods() -> None:
    for period in SCRIPT.PERIODS_PX:
        phase_a = SCRIPT.make_phase("A", period, SCRIPT.DEFAULT_PANEL_SHAPE)
        phase_b = SCRIPT.make_phase("B", period, SCRIPT.DEFAULT_PANEL_SHAPE)
        assert np.array_equal(phase_b[1:], phase_a[1:])
        assert np.allclose(
            np.mod(phase_b[0] - phase_a[0], 2.0 * np.pi),
            np.pi,
            rtol=0.0,
            atol=1e-12,
        )
        assert np.array_equal(phase_b[0], np.mod(phase_a[0] + np.pi, 2.0 * np.pi))


def test_frozen_order_and_plan_hashes_and_axis_displacements() -> None:
    plan = SCRIPT.build_plan(_plan_args())
    scale = plan["encoding"]["max_grayscale"]
    assert SCRIPT.DEFAULT_EXPOSURE_MS == 0.087
    assert plan["exposure_ms_requested"] == 0.087
    assert plan["capture_order"] == ["A128", "A256", "B128", "B256"]
    assert [item["id"] for item in plan["patterns"]] == plan["capture_order"]
    assert plan["n_frames_expected"] == 4
    assert plan["axis_map_constants"] == {
        "source": "issue247 axis-map convention; frozen at plan time",
        "K_col": SCRIPT.K_COL,
        "K_row": SCRIPT.K_ROW,
        "displacement_rule": "K_axis * N; N = panel_columns / period_px",
    }
    # Schema is stable even at D = 0: the block is present and says "not applied".
    focus = plan["focus_compensation"]
    assert focus["focus_d_m1"] == 0.0
    assert focus["focal_mm"] is None
    assert focus["compensation_applied"] is False
    assert focus["fresnel_phase_sha256"] == SCRIPT.digest(
        np.zeros(SCRIPT.DEFAULT_PANEL_SHAPE, dtype=np.float64)
    )
    assert focus["pupil_center"] == [960, 600]
    assert focus["radius"] == 450
    assert focus["pixel_um"] > 0
    assert focus["wavelength_nm"] == SCRIPT.FOCUS_WAVELENGTH_NM
    for item in plan["patterns"]:
        phase = SCRIPT.make_phase(
            item["content"], item["period_px"], SCRIPT.DEFAULT_PANEL_SHAPE
        )
        gray = SCRIPT.phase_to_gray(phase, scale)
        assert item["phase_sha256"] == SCRIPT.digest(phase)
        assert item["gray_sha256"] == SCRIPT.digest(gray)
        assert len(item["phase_sha256"]) == 64
        assert len(item["gray_sha256"]) == 64
        cycles = SCRIPT.DEFAULT_PANEL_SHAPE[1] / item["period_px"]
        assert item["expected_displacement_px"]["column"] == SCRIPT.K_COL * cycles
        assert item["expected_displacement_px"]["row"] == SCRIPT.K_ROW * cycles


def test_encoding_scale_is_resolved_from_gray_scale_source(monkeypatch) -> None:
    # No module-level 2π constant may shadow the single source.
    assert not hasattr(SCRIPT, "ENCODING_MAX_GRAY")
    from ao_shaping.utils.slm.gray_scale import resolve_max_grayscale

    assert SCRIPT.resolve_max_grayscale() == int(resolve_max_grayscale())

    monkeypatch.setattr(SCRIPT, "resolve_max_grayscale", lambda: 1023)
    plan = SCRIPT.build_plan(_plan_args())
    assert plan["encoding"]["max_grayscale"] == 1023
    assert plan["encoding"]["two_pi_radians"] == pytest.approx(2.0 * np.pi)
    assert SCRIPT.phase_to_gray(np.array([0.0, np.pi]), 1023).tolist() == [0, 512]

    # A configured scale must rescale the encoding, not raise or be ignored.
    monkeypatch.setattr(SCRIPT, "resolve_max_grayscale", lambda: 255)
    scaled = SCRIPT.build_plan(_plan_args())
    assert scaled["encoding"]["max_grayscale"] == 255
    assert SCRIPT.phase_to_gray(np.array([0.0, np.pi]), 255).tolist() == [0, 128]
    gray_a = SCRIPT.phase_to_gray(
        SCRIPT.make_phase("A", 128, SCRIPT.DEFAULT_PANEL_SHAPE), 255
    )
    assert int(gray_a.max()) <= 255
    for item in scaled["patterns"]:
        assert item["gray_modulus"] == 256
        assert item["row0_step_gray_unique"] == [128]
        assert item["row0_step_gray_mod"] == 128
        assert item["other_rows_step_gray_unique"] == [0]


def test_row0_gray_step_is_single_valued_and_recorded(monkeypatch) -> None:
    """Post-quantisation evidence: pi is not an integer gray level at any scale."""
    monkeypatch.setattr(SCRIPT, "resolve_max_grayscale", lambda: 1023)
    for d_m1 in (0.0, 0.6):
        plan = SCRIPT.build_plan(_plan_args(focus_d_m1=d_m1))
        assert plan["encoding"]["max_grayscale"] == 1023
        for item in plan["patterns"]:
            assert item["gray_modulus"] == 1024
            # pi -> rint(1023 / 2) = 512 (511.5 rounds to even), so the wrapped
            # step is 512, i.e. 512/1023*2pi = 180.176 deg, 0.176 deg off pi.
            assert item["row0_step_gray_unique"] == [512], (d_m1, item["id"])
            assert item["row0_step_gray_mod"] == 512
            assert item["row0_step_radians"] == pytest.approx(512 * 2.0 * np.pi / 1023)
            assert abs(item["row0_step_radians"] - np.pi) < 0.01
            assert item["other_rows_step_gray_unique"] == [0]

    # Independent recomputation straight from the encoded arrays.
    for period in SCRIPT.PERIODS_PX:
        for d_m1 in (0.0, 0.6):
            phase_a = SCRIPT.make_phase("A", period, SCRIPT.DEFAULT_PANEL_SHAPE, d_m1)
            phase_b = SCRIPT.make_phase("B", period, SCRIPT.DEFAULT_PANEL_SHAPE, d_m1)
            gray_a = SCRIPT.phase_to_gray(phase_a, 1023)
            gray_b = SCRIPT.phase_to_gray(phase_b, 1023)
            delta = np.mod(gray_b.astype(np.int64) - gray_a.astype(np.int64), 1024)
            assert np.unique(delta[0]).tolist() == [512], (period, d_m1)
            assert np.all(delta[1:] == 0), (period, d_m1)


def test_diopter_plan_records_fresnel_and_keeps_row0_pi_step() -> None:
    plan = SCRIPT.build_plan(_plan_args(focus_d_m1=0.6))
    focus = plan["focus_compensation"]
    assert focus["focus_d_m1"] == 0.6
    assert focus["focal_mm"] == 1000.0 / 0.6
    assert focus["compensation_applied"] is True
    assert focus["fresnel_phase_sha256"] == SCRIPT.digest(
        SCRIPT.compensation_phase(0.6)
    )
    assert focus["fresnel_phase_sha256"] != SCRIPT.digest(
        np.zeros(SCRIPT.DEFAULT_PANEL_SHAPE, dtype=np.float64)
    )
    for item in plan["patterns"]:
        phase = SCRIPT.make_phase(
            item["content"], item["period_px"], SCRIPT.DEFAULT_PANEL_SHAPE, 0.6
        )
        assert item["phase_sha256"] == SCRIPT.digest(phase)
    # The compensation is a common additive term: B - A must stay pi / 0.
    for period in SCRIPT.PERIODS_PX:
        phase_a = SCRIPT.make_phase("A", period, SCRIPT.DEFAULT_PANEL_SHAPE, 0.6)
        phase_b = SCRIPT.make_phase("B", period, SCRIPT.DEFAULT_PANEL_SHAPE, 0.6)
        delta = np.mod(phase_b - phase_a, 2.0 * np.pi)
        assert np.allclose(delta[0], np.pi, rtol=0.0, atol=1e-9)
        assert np.allclose(delta[1:], 0.0, rtol=0.0, atol=1e-9)


def test_self_test_command_is_offline_and_does_not_create_output(
    tmp_path: Path,
) -> None:
    out = tmp_path / "must_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(SCRIPT.__file__).resolve()),
            "--self-test",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not out.exists()


class _FakeSLM:
    """Records call order; no driver module and no real panel is involved."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def open(self) -> None:
        self.calls.append("open")

    def display_data(self, gray: object) -> None:
        self.calls.append("display_data")


def test_open_slm_guarded_checks_drive_board_before_phase_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = set(sys.modules)

    def fake_guard(slm: _FakeSLM) -> str:
        slm.calls.append("drive_board_guard")
        return "DRIVE-BOARD-ID"

    monkeypatch.setattr(SCRIPT, "require_drive_board_entry", fake_guard)
    slm = _FakeSLM()
    assert SCRIPT.open_slm_guarded(slm) == "DRIVE-BOARD-ID"
    assert slm.calls == ["open", "drive_board_guard"]
    assert not [
        name
        for name in set(sys.modules) - before
        if name.startswith("ao_shaping.drivers")
    ]


def test_guard_failure_blocks_phase_write(monkeypatch: pytest.MonkeyPatch) -> None:
    class DriveBoardError(RuntimeError):
        pass

    def failing_guard(slm: _FakeSLM) -> str:
        slm.calls.append("drive_board_guard")
        raise DriveBoardError("drive board silent; panel would not modulate")

    monkeypatch.setattr(SCRIPT, "require_drive_board_entry", failing_guard)
    slm = _FakeSLM()
    with pytest.raises(DriveBoardError, match="drive board silent"):
        SCRIPT.open_slm_guarded(slm)
    assert slm.calls == ["open", "drive_board_guard"]
    assert "display_data" not in slm.calls


def test_execute_wiring_uses_guarded_open() -> None:
    """Pin the wiring: execute must not open the panel via an unguarded path."""
    tree = ast.parse(Path(SCRIPT.__file__).read_text(encoding="utf-8"))
    execute = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    called = {
        node.func.id
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "open_slm_guarded" in called
    assert "require_drive_board_entry" not in called
