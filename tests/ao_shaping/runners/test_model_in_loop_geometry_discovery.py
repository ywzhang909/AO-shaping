"""The GUI's geometry-file entry point: discovery + caption helpers.

``discover_model_in_loop_geometry`` and ``describe_bench_geometry`` live on the
runner layer (``runners/runner_common.py``) so the GUI and the CLI agree on what
a usable artefact is.  They are pure filesystem/JSON helpers, so the whole
contract is offline-verifiable.

WHAT is pinned
--------------
1. Only ``method == "sweep"`` files are offered: a speckle artefact is exactly
   the abandoned ~0.07 route the bake-off replaced.
2. The newest candidate wins, across all search layouts.
3. A ``region`` filter is honoured (mismatch is a real incompatibility).
4. Malformed or unrelated files are skipped without raising.
5. ``describe_bench_geometry`` formats a summary and never raises -- a caption
   must report a broken file rather than surface a traceback in the sidebar.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ao_shaping.runners.runner_common import (
    describe_bench_geometry,
    discover_model_in_loop_geometry,
)


def _write_geometry(
    path: Path, *, region: int = 64, far_field_size: int = 192, method: str = "sweep"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "panel_disc_radius": 450,
                "region": region,
                "beam_waist_panel_px": 450.0,
                "far_field_size": far_field_size,
                "camera_px_per_model_px": 2.7955,
                "spot_fwhm_camera_px": 12.0,
                "spot_fwhm_model_px": 4.0,
                "correlation": 0.8376,
                "method": method,
                "calibration_notes": "bench sweep",
            }
        ),
        encoding="utf-8",
    )
    return path


class TestDiscoverModelInLoopGeometry:
    def test_missing_root_is_none(self, tmp_path: Path) -> None:
        assert discover_model_in_loop_geometry(root=tmp_path / "nope") is None

    def test_empty_root_is_none(self, tmp_path: Path) -> None:
        assert discover_model_in_loop_geometry(root=tmp_path) is None

    def test_finds_a_debug_run_layout(self, tmp_path: Path) -> None:
        """``<root>/<topic>/<timestamp>/bench_geometry.json`` (debug saver)."""
        expected = _write_geometry(
            tmp_path / "slm_model_in_loop" / "20261013_105911" / "bench_geometry.json"
        )
        assert discover_model_in_loop_geometry(root=tmp_path) == expected

    def test_finds_a_runbook_layout(self, tmp_path: Path) -> None:
        """``<root>/<out>/bench_geometry.json`` (calibration runbook default)."""
        expected = _write_geometry(
            tmp_path / "model_in_loop_hw" / "bench_geometry.json"
        )
        assert discover_model_in_loop_geometry(root=tmp_path) == expected

    def test_newest_candidate_wins(self, tmp_path: Path) -> None:
        older = _write_geometry(
            tmp_path / "slm_model_in_loop" / "old" / "bench_geometry.json"
        )
        newer = _write_geometry(
            tmp_path / "slm_model_in_loop" / "new" / "bench_geometry.json"
        )
        os.utime(older, (time.time() - 600.0, time.time() - 600.0))
        assert discover_model_in_loop_geometry(root=tmp_path) == newer

    def test_speckle_artefacts_are_not_offered(self, tmp_path: Path) -> None:
        """A speckle geometry is the ~0.07 route the bake-off replaced."""
        _write_geometry(
            tmp_path / "slm_model_in_loop" / "speckle" / "bench_geometry.json",
            method="speckle",
        )
        assert discover_model_in_loop_geometry(root=tmp_path) is None

    def test_region_filter_is_honoured(self, tmp_path: Path) -> None:
        _write_geometry(
            tmp_path / "slm_model_in_loop" / "other" / "bench_geometry.json", region=32
        )
        expected = _write_geometry(
            tmp_path / "slm_model_in_loop" / "match" / "bench_geometry.json", region=64
        )
        assert discover_model_in_loop_geometry(64, root=tmp_path) == expected
        assert discover_model_in_loop_geometry(128, root=tmp_path) is None

    def test_malformed_files_are_skipped(self, tmp_path: Path) -> None:
        broken = tmp_path / "slm_model_in_loop" / "broken" / "bench_geometry.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{not json", encoding="utf-8")
        unrelated = tmp_path / "slm_model_in_loop" / "list" / "bench_geometry.json"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("[1, 2, 3]", encoding="utf-8")
        expected = _write_geometry(
            tmp_path / "slm_model_in_loop" / "good" / "bench_geometry.json"
        )
        assert discover_model_in_loop_geometry(root=tmp_path) == expected


class TestDescribeBenchGeometry:
    def test_summarises_the_measured_fields(self, tmp_path: Path) -> None:
        path = _write_geometry(tmp_path / "bench_geometry.json")
        caption = describe_bench_geometry(path)
        assert "sweep" in caption
        assert "region=64" in caption
        assert "far_field=192" in caption
        assert "corr=0.8376" in caption
        assert "cam/model=2.7955" in caption

    def test_missing_file_reports_instead_of_raising(self, tmp_path: Path) -> None:
        caption = describe_bench_geometry(tmp_path / "absent.json")
        assert "无法读取几何文件" in caption

    def test_malformed_file_reports_instead_of_raising(self, tmp_path: Path) -> None:
        path = tmp_path / "bench_geometry.json"
        path.write_text("not json", encoding="utf-8")
        assert "无法读取几何文件" in describe_bench_geometry(path)

    def test_non_object_json_reports_instead_of_raising(self, tmp_path: Path) -> None:
        path = tmp_path / "bench_geometry.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert "无法读取几何文件" in describe_bench_geometry(path)

    def test_no_selection_is_stated_not_raised(self) -> None:
        """``None`` / blank is the GUI's "no geometry" state, not an error."""
        for value in (None, "", "   "):
            assert "未指定" in describe_bench_geometry(value)
