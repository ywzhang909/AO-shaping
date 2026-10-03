"""Tests for :mod:`ml.hwdataset.inspect`."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ml.hwdataset.inspect import main
from ml.hwdataset.index import build_hw_index

_STAMP = "20260101_000000"


def _write_pickle(directory: Path, stem: str, payload: object) -> Path:
    """Write ``payload`` into ``directory/<stamp>/<stem>.pkl`` and return the path."""
    run = directory / _STAMP
    run.mkdir(parents=True, exist_ok=True)
    path = run / f"{stem}.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path


def _sidecar(directory: Path, stem: str, **fields: object) -> Path:
    """Write a sidecar JSON next to a pickle."""
    path = directory / _STAMP / f"{stem}.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def _panel_record(index: int, *, gray: bool, exposure: float | None, side: int = 16):
    """One record with a real panel phase and a real frame."""
    height, width = 40, 60
    grid_y, grid_x = np.ogrid[:height, :width]
    radius_sq = (grid_x - 30) ** 2 + (grid_y - 20) ** 2
    inside = radius_sq <= (12.0**2)
    phase = np.zeros((height, width), dtype=np.uint16 if gray else np.float32)
    phase[inside] = 900 if gray else 1.5
    record = {
        "_phase": phase,
        "_img": np.full((side, side), 10 + index, dtype=np.uint8),
    }
    if exposure is not None:
        record["exp_t"] = exposure
    return record


def _corpus(root: Path) -> None:
    """A two-family corpus plus a ``None`` payload."""
    collect = root / "model_in_loop_hw_collect_x" / _STAMP
    collect.mkdir(parents=True, exist_ok=True)
    payload = {
        i: _panel_record(i, gray=False, exposure=None) for i in range(3)
    }
    path = _write_pickle(root / "model_in_loop_hw_collect_x", "collect", payload)
    _sidecar(root / "model_in_loop_hw_collect_x", "collect", region=8, exposure_ms=1.1)

    gray = {
        i: _panel_record(i, gray=True, exposure=0.1 + i, side=24) for i in range(2)
    }
    gray_path = _write_pickle(root / "slm_pib_x", "pib", gray)
    assert path.exists() and gray_path.exists()
    _write_pickle(root / "slm_pib_online", "recorder_none", None)


class TestInspectJson:
    """The JSON report must round-trip the real dataclasses exactly."""

    def test_json_report_matches_the_index(self, tmp_path: Path, capsys) -> None:
        _corpus(tmp_path)
        code = main(["--roots", str(tmp_path), "--progress-every", "0", "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        index = build_hw_index(tmp_path, progress_every=0)
        assert payload["total_records"] == len(index) == 5
        assert payload["files_scanned"] == index.files_scanned == 3
        assert payload["files_usable"] == index.files_usable == 2
        assert payload["by_source"] == {
            "panel_rad": 3,
            "panel_gray": 2,
            "zernike": 0,
            "freeform": 0,
        }
        assert payload["by_family"] == {
            "model_in_loop_hw_collect": 3,
            "slm_pib": 2,
        }
        assert payload["excluded"] == {"no_records": 1}
        assert payload["exposure_ms"]["count"] == 5
        assert payload["exposure_ms"]["min"] == 0.1
        # the gray family contributes exp_t 0.1 and 1.1 (0.1 + i for i in 0..1),
        # the collect family contributes the sidecar's 1.1 -> two distinct values.
        assert payload["exposure_ms"]["distinct"] == 2

    def test_json_family_shapes(self, tmp_path: Path, capsys) -> None:
        _corpus(tmp_path)
        assert main(["--roots", str(tmp_path), "--progress-every", "0", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["family_shapes"]["model_in_loop_hw_collect"] == [[0, None, None]]
        assert payload["family_shapes"]["slm_pib"] == [[0, None, None]]

    def test_json_flag_needed_for_json(self, tmp_path: Path, capsys) -> None:
        """Without --json the output is a table, so it must not parse as JSON."""
        _corpus(tmp_path)
        assert main(["--roots", str(tmp_path), "--progress-every", "0"]) == 0
        with pytest.raises(json.JSONDecodeError):
            json.loads(capsys.readouterr().out)


class TestInspectHuman:
    """The human-readable table must name the things that are easy to miss."""

    def test_table_mentions_sources_families_and_exposure(self, tmp_path: Path, capsys) -> None:
        _corpus(tmp_path)
        assert main(["--roots", str(tmp_path), "--progress-every", "0"]) == 0
        out = capsys.readouterr().out
        assert "panel_rad" in out and "panel_gray" in out
        assert "model_in_loop_hw_collect" in out and "slm_pib" in out
        assert "exposure (ms)" in out
        assert "no_records" in out

    def test_empty_exclusion_tally_says_so(self, tmp_path: Path, capsys) -> None:
        """A corpus with nothing excluded must say that, not print an empty block."""
        payload = {i: _panel_record(i, gray=False, exposure=1.0) for i in range(2)}
        _write_pickle(tmp_path / "slm_pib_x", "pib", payload)
        assert main(["--roots", str(tmp_path), "--progress-every", "0"]) == 0
        assert "nothing excluded" in capsys.readouterr().out

    def test_multiple_fovs_warn(self, tmp_path: Path, capsys) -> None:
        """Two camera windows in one corpus must trigger the FOV warning."""
        a = {i: _panel_record(i, gray=False, exposure=1.0, side=16) for i in range(2)}
        _write_pickle(tmp_path / "slm_pib_x", "a", a)
        _sidecar(tmp_path / "slm_pib_x", "a", region=8)
        b = {i: _panel_record(i, gray=False, exposure=1.0, side=32) for i in range(2)}
        _write_pickle(tmp_path / "model_in_loop_hw_sweep_x", "b", b)
        _sidecar(tmp_path / "model_in_loop_hw_sweep_x", "b", region=64)
        assert main(["--roots", str(tmp_path), "--progress-every", "0"]) == 0
        out = capsys.readouterr().out
        assert "WARNING" in out and "fields of view" in out
        assert "[16, 128]" in out


class TestInspectOptions:
    """Flag handling."""

    def test_limit_files(self, tmp_path: Path, capsys) -> None:
        _corpus(tmp_path)
        assert main(
            ["--roots", str(tmp_path), "--limit-files", "1", "--progress-every", "0", "--json"]
        ) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["files_scanned"] == 1
        assert payload["total_records"] == 3

    def test_families_filter(self, tmp_path: Path, capsys) -> None:
        _corpus(tmp_path)
        assert main(
            [
                "--roots", str(tmp_path),
                "--families", "slm_pib",
                "--progress-every", "0", "--json",
            ]
        ) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["by_family"] == {"slm_pib": 2}
        assert payload["total_records"] == 2

    def test_sample_reports_shapes(self, tmp_path: Path, capsys) -> None:
        """--sample must report the real tensor contract.

        The CLI samples with the DEFAULT panel geometry (centre (960, 600),
        radius 500), so a record materialisable under that geometry needs a real
        1200x1920 panel -- a tiny synthetic panel legitimately cannot be cropped.
        """
        height, width = 1200, 1920
        grid_y, grid_x = np.ogrid[:height, :width]
        inside = (grid_x - 960) ** 2 + (grid_y - 600) ** 2 <= 400**2
        phase = np.zeros((height, width), dtype=np.float32)
        phase[inside] = 1.5
        record = {"_phase": phase, "_img": np.full((32, 32), 200, np.uint8), "exp_t": 1.1}
        _write_pickle(tmp_path / "slm_pib_x", "big", {0: record})
        assert main(["--roots", str(tmp_path), "--grid", "8", "--sample", "1"]) == 0
        out = capsys.readouterr().out
        assert "materialised samples" in out
        assert "phase_cos (8, 8) float32" in out
        assert "image     (8, 8) float32" in out
        assert "contrast_max" in out

    def test_sample_failure_is_reported_not_fatal(self, tmp_path: Path, capsys) -> None:
        """A record the default geometry cannot crop must warn, not crash the CLI."""
        _corpus(tmp_path)
        assert main(["--roots", str(tmp_path), "--grid", "8", "--sample", "2"]) == 0
        out = capsys.readouterr().out
        assert "materialised samples" not in out  # nothing was reportable
        assert "panel_rad" in out  # the corpus report itself still printed

    def test_sample_zero_skips(self, tmp_path: Path, capsys) -> None:
        _corpus(tmp_path)
        assert main(["--roots", str(tmp_path), "--sample", "0"]) == 0
        assert "materialised samples" not in capsys.readouterr().out

    def test_empty_corpus_exits_two(self, tmp_path: Path, capsys) -> None:
        """An empty corpus is a clean exit 2 with a message, never a traceback."""
        (tmp_path / "empty").mkdir()
        assert main(["--roots", str(tmp_path / "empty"), "--progress-every", "0"]) == 2
        captured = capsys.readouterr()
        assert "no usable records" in captured.err
        assert "Traceback" not in captured.err

    def test_no_require_exposure_keeps_unresolvable(self, tmp_path: Path, capsys) -> None:
        """Exposure resolution is the record first, then the sidecar.

        The collect family here resolves 1.1 ms from the sidecar. A record with
        neither must be dropped by default and kept with --no-require-exposure.
        """
        payload = {
            0: _panel_record(0, gray=False, exposure=None),
            1: _panel_record(1, gray=False, exposure=None),
        }
        _write_pickle(tmp_path / "model_in_loop_hw_collect_x", "c", payload)  # no sidecar
        args = ["--roots", str(tmp_path), "--progress-every", "0", "--json"]
        assert main(args) == 2  # every record lacks an exposure
        capsys.readouterr()  # discard the exit-2 stderr before the second run
        assert main([*args, "--no-require-exposure"]) == 0
        payload_json = json.loads(capsys.readouterr().out)
        assert payload_json["total_records"] == 2
        assert payload_json["exposure_ms"]["count"] == 0

    def test_build_cache_flag_is_optional(self, tmp_path: Path, capsys) -> None:
        """--build-cache must never let an ImportError escape main()."""
        _corpus(tmp_path)
        assert main(["--roots", str(tmp_path), "--progress-every", "0"]) == 0
        capsys.readouterr()
        try:
            import ml.hwdataset.cache  # noqa: F401
        except ImportError:
            assert main(["--roots", str(tmp_path), "--build-cache"]) == 1
            assert "unavailable" in capsys.readouterr().err
        else:
            # The cache module exists; a successful build must return 0.
            assert main(["--roots", str(tmp_path), "--build-cache"]) == 0
            assert "cache directories" in capsys.readouterr().out
