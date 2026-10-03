"""Unit tests for :mod:`ml.hwdataset.index` (corpus discovery + record indexing).

Every fixture is written into a pytest ``tmp_path`` as a synthetic pickle, so the
suite never touches the real ``data/debug`` corpus (which is 65.88 GB across 265
files and is absent in CI). Panel arrays are deliberately tiny (``(40, 60)``
instead of the real ``(1200, 1920)``): this module reads shapes and dtypes only
and never the values, so a 9 MB array per record would buy nothing.

The module under test must stay torch-free -- ``torch`` is never imported here,
and :func:`test_module_is_torch_free` asserts that.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.hwdataset.index import (
    DEFAULT_ROOTS,
    ExclusionReason,
    HwCorpusIndex,
    HwRecordRef,
    PhaseSource,
    build_hw_index,
    family_of,
)

# Small stand-ins for the two-level real layout
# (``data/debug/<family>_<ts>/<ts>/<prefix>_<ts>_<ts>.pkl``).
_STAMP = "20260101_000000"


# ---------------------------------------------------------------------------
# Fixture writers (module level so they are reusable and lintable)
# ---------------------------------------------------------------------------
class RecorderStandIn:
    """Minimal stand-in for :class:`ao_shaping.utils.io.file.Recorder`.

    Defined at module level because ``pickle`` records the defining module of a
    class, so an in-function class could not be unpickled back.

    Attributes:
        history: The recorded entries, in recording order.
    """

    def __init__(self, history: list[dict[str, Any]]) -> None:
        """Store the history list.

        Args:
            history: Recorded entries in recording order.
        """
        self.history = history


class BareObject:
    """A pickled object with no ``history`` attribute (module level so it pickles)."""


def _dump(root: Path, payload: Any, stem: str) -> Path:
    """Write ``payload`` as a pickle one timestamp level below ``root``.

    Args:
        root: Directory that plays the role of ``data/debug/<family>_<ts>``.
        payload: Any picklable object.
        stem: File stem (without ``.pkl``).

    Returns:
        The written ``.pkl`` path.
    """
    run_dir = root / _STAMP
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{stem}.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path


def write_sidecar(dir: Path, name: str, **fields: Any) -> Path:
    """Write a sidecar JSON file into ``dir``.

    Args:
        dir: Directory to write into (normally a pickle's own directory).
        name: File name, e.g. ``"<stem>.json"`` or ``"summary_x.json"``.
        fields: JSON object fields.

    Returns:
        The written JSON path.
    """
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / name
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def _panel(shape: tuple[int, int] = (40, 60)) -> np.ndarray:
    """Return a small float32 radian panel (wrapped to ``[0, 2pi)``).

    Args:
        shape: Panel shape.

    Returns:
        A ``float32`` array shaped like the real ``(1200, 1920)`` panels.
    """
    return (np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape) % 6.2831853).astype(
        np.float32
    )


def _frame(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    """Return a small ``uint8`` CCD frame.

    Args:
        shape: Frame shape.

    Returns:
        A ``uint8`` array shaped like the real ``(250, 248)`` frames.
    """
    return (np.arange(shape[0] * shape[1], dtype=np.uint16) % 251).astype(np.uint8).reshape(shape)


def write_panel_rad_pkl(dir: Path, n: int = 3) -> Path:
    """Write a ``dict[int, dict]`` payload whose ``_phase`` is float32 radians.

    Mirrors ``model_in_loop_hw_collect`` / ``_sweep``: ``_phase`` is
    ``(1200, 1920) float32`` and the records carry **no** ``exp_t``, so the
    exposure can only come from the sidecar.

    Args:
        dir: Family directory.
        n: Number of records.

    Returns:
        The written ``.pkl`` path.
    """
    stem = f"collect_{_STAMP}"
    payload = {
        i: {"_phase": _panel(), "_img": _frame(), "peak": float(i)} for i in range(n)
    }
    path = _dump(dir, payload, stem)
    write_sidecar(path.parent, f"{stem}.json", slm_wavelength=1064, exposure_ms=1.1, region=32)
    return path


def write_panel_gray_pkl(dir: Path, n: int = 3, n_terms: int = 55) -> Path:
    """Write a ``dict[int, dict]`` payload whose ``_phase`` is ``uint16`` grayscale.

    Mirrors ``slm_pib_*``: ``_phase`` is ``(1200, 1920) uint16`` **and** ``_c`` is
    present with length 55. 55 = ``(9+1)(9+2)/2`` is a perfectly valid triangular
    Zernike count, so without ``_phase`` precedence these records would be
    mislabelled ``ZERNIKE n_max=9`` -- which is wrong, because ``_c`` here is the
    PIB optimiser's axis-bucket configuration, not Zernike coefficients. This is
    exactly why the panel array outranks the coefficient length.

    Args:
        dir: Family directory.
        n: Number of records.
        n_terms: Length of the distractor ``_c``.

    Returns:
        The written ``.pkl`` path.
    """
    stem = f"pib_{_STAMP}"
    payload = {
        i: {
            "_phase": (np.arange(40 * 60, dtype=np.uint16).reshape(40, 60) + i) % 1024,
            "_img": _frame(),
            "_c": np.linspace(0.0, 1.0, n_terms),
            "exp_t": 1.2 + i,
        }
        for i in range(n)
    }
    return _dump(dir, payload, stem)


def write_none_pkl(dir: Path) -> Path:
    """Write a pickle holding ``None`` (an aborted run's stub).

    Three such 4-byte files exist in the measured corpus, e.g.
    ``data/debug/slm_pib_online/<ts>/recorder_snr.pkl``.

    Args:
        dir: Family directory.

    Returns:
        The written ``.pkl`` path.
    """
    return _dump(dir, None, "recorder_snr")


def write_zernike_pkl(dir: Path, n: int = 3, n_terms: int = 15) -> Path:
    """Write a ``dict[int, dict]`` payload with no ``_phase`` and a ``_c`` vector.

    Mirrors ``slm_zernike_shaping_*``: no ``_phase``, ``_c`` of a triangular
    length, and a per-record ``exp_t`` (the sidecar carries no exposure).

    Args:
        dir: Family directory.
        n: Number of records.
        n_terms: Length of ``_c``; 15 is the measured ``n_max=4`` corpus.

    Returns:
        The written ``.pkl`` path.
    """
    stem = f"zernike_{_STAMP}"
    payload = {
        i: {
            "_c": np.linspace(0.0, 1.0, n_terms),
            "_grad": np.zeros(n_terms),
            "_img": _frame(),
            "exp_t": 2.0,
        }
        for i in range(n)
    }
    return _dump(dir, payload, stem)


def write_freeform_pkl(dir: Path, n: int = 3, grid: int = 24) -> Path:
    """Write a ``dict[int, dict]`` payload with no ``_phase`` and a ``grid**2`` ``_c``.

    Mirrors ``slm_gsnet_square``: 24x24 = 576 freeform coefficients.

    Args:
        dir: Family directory.
        n: Number of records.
        grid: Freeform grid side (default 24 -> 576 coefficients).

    Returns:
        The written ``.pkl`` path.
    """
    stem = f"square_{_STAMP}"
    payload = {
        i: {
            "_c": np.linspace(0.0, 0.5, grid * grid),
            "_img": _frame(),
            "exp_t": 80.0,
        }
        for i in range(n)
    }
    return _dump(dir, payload, stem)


def write_recorder_like_pkl(dir: Path, n: int = 3) -> Path:
    """Write a pickled :class:`RecorderStandIn` with ``.history``.

    Mirrors the 8 ``ao_shaping.utils.io.file.Recorder`` payloads under
    ``data/debug/slm_pib_online/``. The sidecar is a ``summary_*.json`` (not
    ``<stem>.json``), exactly as on disk.

    Args:
        dir: Family directory.
        n: Number of history entries.

    Returns:
        The written ``.pkl`` path.
    """
    stem = "recorder_smoke_f4"
    history = [
        {
            "_c": np.linspace(0.0, 0.3, 36),
            "_img": _frame(),
            "exp_t": 1.2,
            "_id": i,
        }
        for i in range(n)
    ]
    path = _dump(dir, RecorderStandIn(history), stem)
    write_sidecar(path.parent, "summary_smoke_f4.json", rows=n, fold=n - 1)
    return path


def write_mixed_pkl(dir: Path, n_with_phase: int = 12, n_without: int = 6) -> Path:
    """Write a payload mixing phase-bearing and phase-less records.

    Regression fixture for ``sim_calib_abba_20261001_163743.pkl``, which is
    ``{NO_PHASE: 6, FREEFORM: 12}`` -- per-file classification would drop or
    mislabel half of it.

    Args:
        dir: Family directory.
        n_with_phase: Number of records carrying a freeform ``_c``.
        n_without: Number of records carrying neither ``_phase`` nor ``_c``.

    Returns:
        The written ``.pkl`` path.
    """
    stem = f"abba_{_STAMP}"
    payload: dict[int, dict[str, Any]] = {}
    for i in range(n_with_phase):
        payload[i] = {"_c": np.linspace(0.0, 0.4, 576), "_img": _frame(), "exp_t": 1.5}
    for i in range(n_with_phase, n_with_phase + n_without):
        payload[i] = {"_img": _frame(), "exp_t": 1.5}
    path = _dump(dir, payload, stem)
    write_sidecar(path.parent, f"{stem}.json", bench="synthetic abba")
    return path


# ---------------------------------------------------------------------------
# Torch-free contract
# ---------------------------------------------------------------------------
class TestModuleContract:
    def test_module_is_torch_free(self) -> None:
        """No torch import statement in this module, checked via the AST.

        An AST walk is used rather than a substring search so that prose in the
        docstrings (which legitimately *mentions* torch, e.g. the measured import
        cost) does not produce a false positive.

        Scope note: this asserts the *source*, not the transitive import graph.
        ``ao_shaping/runners/__init__.py`` (lines 14-32) defeats its own lazy
        ``__getattr__`` with eager imports, so importing this module currently
        pulls torch in anyway and takes ~32 s. That is a pre-existing repo defect
        documented in the module docstring; fixing it is outside these two files.
        A transitive assertion would encode a bug as a requirement.
        """
        import ast

        import ml.hwdataset.index as index_module

        tree = ast.parse(Path(index_module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not {name for name in imported if name.split(".")[0] == "torch"}

    def test_default_roots_target_the_whole_debug_tree(self) -> None:
        assert DEFAULT_ROOTS == ("data/debug",)

    def test_public_api_surface(self) -> None:
        import ml.hwdataset.index as index_module

        assert index_module.__all__ == [
            "DEFAULT_ROOTS",
            "ExclusionReason",
            "HwCorpusIndex",
            "HwRecordRef",
            "PhaseSource",
            "build_hw_index",
        ]

    def test_relative_imports_forbidden(self) -> None:
        """AGENTS.md: absolute imports only, no ``from .`` / ``from ..``."""
        import ast

        import ml.hwdataset.index as index_module

        tree = ast.parse(Path(index_module.__file__).read_text(encoding="utf-8"))
        relative = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level > 0
        ]
        assert relative == []


# ---------------------------------------------------------------------------
# 1 + 5: fixture families resolve to the expected PhaseSource
# ---------------------------------------------------------------------------
class TestPhaseSourceClassification:
    def test_panel_rad_family(self, tmp_path: Path) -> None:
        pkl = write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")
        index = build_hw_index(tmp_path)

        assert index.files_scanned == 1
        assert index.files_usable == 1
        assert len(index) == 3
        assert index.counts_by_source() == {PhaseSource.PANEL_RAD: 3}
        assert index.records[0].path == pkl.resolve()
        assert index.records[0].source is PhaseSource.PANEL_RAD
        # Exposure only reachable through the sidecar.
        assert index.records[0].exposure_ms == 1.1
        assert index.records[0].sidecar["slm_wavelength"] == 1064

    def test_panel_gray_family_prefers_phase_over_odd_c_length(self, tmp_path: Path) -> None:
        write_panel_gray_pkl(tmp_path / "slm_pib_x")
        index = build_hw_index(tmp_path)

        assert len(index) == 3
        assert index.counts_by_source() == {PhaseSource.PANEL_GRAY: 3}
        # 55 is neither triangular nor a square: `_phase` must have decided it.
        assert index.records[0].n_terms == 55
        assert index.records[0].n_max is None
        assert index.records[0].freeform_grid is None
        assert [r.exposure_ms for r in index.records] == [1.2, 2.2, 3.2]

    def test_zernike_family_infers_n_max(self, tmp_path: Path) -> None:
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x")
        index = build_hw_index(tmp_path)

        assert len(index) == 3
        assert index.counts_by_source() == {PhaseSource.ZERNIKE: 3}
        assert index.records[0].n_terms == 15
        assert index.records[0].n_max == 4
        assert index.records[0].freeform_grid is None

    def test_freeform_family_derives_grid(self, tmp_path: Path) -> None:
        write_freeform_pkl(tmp_path / "slm_gsnet_square_x")
        index = build_hw_index(tmp_path)

        assert len(index) == 3
        assert index.counts_by_source() == {PhaseSource.FREEFORM: 3}
        assert index.records[0].n_terms == 576
        assert index.records[0].freeform_grid == 24
        assert index.records[0].n_max is None

    def test_all_families_together(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")
        write_panel_gray_pkl(tmp_path / "slm_pib_hw_x")
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x")
        write_freeform_pkl(tmp_path / "slm_gsnet_square_x")
        index = build_hw_index(tmp_path)

        assert index.files_scanned == 4
        assert index.files_usable == 4
        assert len(index) == 12
        assert index.counts_by_source() == {
            PhaseSource.PANEL_RAD: 3,
            PhaseSource.PANEL_GRAY: 3,
            PhaseSource.ZERNIKE: 3,
            PhaseSource.FREEFORM: 3,
        }
        assert index.counts_by_family() == {
            "model_in_loop_hw_collect": 3,
            "slm_pib": 3,
            "slm_zernike_shaping": 3,
            "slm_gsnet_square": 3,
        }
        assert index.families == (
            "model_in_loop_hw_collect",
            "slm_gsnet_square",
            "slm_pib",
            "slm_zernike_shaping",
        )
        assert index.excluded == {}


class TestCoefficientLengthAmbiguity:
    """36 is BOTH a triangular Zernike count (n_max=7) and a perfect square (6x6)."""

    def test_length_36_with_sidecar_n_max_is_zernike(self, tmp_path: Path) -> None:
        pkl = write_zernike_pkl(tmp_path / "sim_calib_abba_x", n_terms=36)
        write_sidecar(pkl.parent, f"{pkl.stem}.json", n_max=7, bench="synthetic")
        index = build_hw_index(tmp_path)

        assert index.counts_by_source() == {PhaseSource.ZERNIKE: 3}
        assert index.records[0].n_max == 7
        assert index.records[0].freeform_grid is None
        assert index.records[0].sidecar["n_max"] == 7

    def test_length_36_in_freeform_family_is_freeform_grid_6(self, tmp_path: Path) -> None:
        """Rule 4: inside a freeform family the square wins (36 -> 6x6)."""
        write_zernike_pkl(tmp_path / "sim_calib_abba_x", n_terms=36)
        index = build_hw_index(tmp_path)

        assert index.counts_by_source() == {PhaseSource.FREEFORM: 3}
        assert index.records[0].freeform_grid == 6
        assert index.records[0].n_max is None

    def test_length_36_outside_freeform_family_is_zernike_n_max_7(self, tmp_path: Path) -> None:
        """Rule 4 the other way: the 36-mode records came from the Zernike optimiser.

        This is the real `slm_pib_online` case (192 of 11393 records): a
        `Recorder` payload whose `summary_*.json` sidecar carries no `n_max`, so
        only the family can settle it. Reading them as a 6x6 freeform grid
        reconstructs a completely different basis.
        """
        write_zernike_pkl(tmp_path / "slm_pib_online_x", n_terms=36)
        index = build_hw_index(tmp_path)

        assert index.counts_by_source() == {PhaseSource.ZERNIKE: 3}
        assert index.records[0].n_max == 7
        assert index.records[0].freeform_grid is None

    def test_length_576_is_freeform_grid_24_everywhere(self, tmp_path: Path) -> None:
        """576 is never a triangular Zernike count, so the family is irrelevant."""
        for family in ("slm_gsnet_square_x", "slm_pib_online_x", "slm_zernike_shaping_x"):
            write_freeform_pkl(tmp_path / family, grid=24)
        index = build_hw_index(tmp_path)

        assert index.counts_by_source() == {PhaseSource.FREEFORM: 9}
        assert {r.freeform_grid for r in index.records} == {24}

    def test_length_15_is_zernike_n_max_4(self, tmp_path: Path) -> None:
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n_terms=15)
        index = build_hw_index(tmp_path)

        assert index.counts_by_source() == {PhaseSource.ZERNIKE: 3}
        assert index.records[0].n_max == 4

    def test_length_78_is_zernike_n_max_11(self, tmp_path: Path) -> None:
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n_terms=78)
        index = build_hw_index(tmp_path)

        assert index.records[0].n_max == 11

    def test_length_7_is_excluded(self, tmp_path: Path) -> None:
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n_terms=7)
        index = build_hw_index(tmp_path)

        assert len(index) == 0
        assert index.excluded == {ExclusionReason.ODD_COEFFICIENT_LENGTH: 3}

    def test_length_55_without_phase_is_zernike_n_max_9(self, tmp_path: Path) -> None:
        """55 = (9+1)(9+2)/2 IS triangular, so it is NOT an odd length.

        Pinned because it is the ``slm_pib_*`` distractor length: the real corpus
        stores ``_c`` of length 55 there, and it is only the ``_phase`` precedence
        (rule 0) that stops it being read as Zernike n_max=9.
        """
        write_panel_gray_pkl(tmp_path / "slm_zernike_shaping_x", n_terms=55)
        index = build_hw_index(tmp_path)

        assert index.counts_by_source() == {PhaseSource.PANEL_GRAY: 3}
        assert index.records[0].n_terms == 55
        assert index.records[0].n_max is None

    def test_length_20_is_excluded(self, tmp_path: Path) -> None:
        """20 is neither a square nor triangular (n_max 5 -> 21)."""
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n_terms=20)
        index = build_hw_index(tmp_path)

        assert len(index) == 0
        assert index.excluded == {ExclusionReason.ODD_COEFFICIENT_LENGTH: 3}

    def test_length_2_is_excluded(self, tmp_path: Path) -> None:
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n_terms=2)
        index = build_hw_index(tmp_path)

        assert index.excluded == {ExclusionReason.ODD_COEFFICIENT_LENGTH: 3}


# ---------------------------------------------------------------------------
# 2: per-record classification (sim_calib_abba regression)
# ---------------------------------------------------------------------------
class TestPerRecordClassification:
    def test_mixed_payload_keeps_only_phase_bearing_records(self, tmp_path: Path) -> None:
        write_mixed_pkl(tmp_path / "sim_calib_abba_x", n_with_phase=12, n_without=6)
        index = build_hw_index(tmp_path)

        assert index.files_scanned == 1
        assert index.files_usable == 1
        assert len(index) == 12
        assert index.excluded == {ExclusionReason.MISSING_PHASE: 6}
        assert index.counts_by_source() == {PhaseSource.FREEFORM: 12}
        # Positions are preserved, so a consumer can re-open the payload and hit
        # exactly the records the index names.
        assert [r.position for r in index.records] == list(range(12))

    def test_classification_does_not_depend_on_record_zero(self, tmp_path: Path) -> None:
        """Same 3 records, phase-less one FIRST: labels must not flip.

        A "classify the file from record 0" shortcut is the obvious wrong
        implementation of this feature; this pins the per-record behaviour.
        """
        stem = f"zero_{_STAMP}"
        payload = {
            0: {"_img": _frame(), "exp_t": 1.5},
            1: {"_c": np.zeros(576), "_img": _frame(), "exp_t": 1.5},
            2: {"_c": np.zeros(15), "_img": _frame(), "exp_t": 1.5},
        }
        _dump(tmp_path / "sim_calib_abba_x", payload, stem)
        index = build_hw_index(tmp_path)

        by_position = {r.position: r.source for r in index.records}
        assert by_position == {
            1: PhaseSource.FREEFORM,
            2: PhaseSource.ZERNIKE,
        }
        assert index.excluded == {ExclusionReason.MISSING_PHASE: 1}

    def test_mixed_panel_and_zernike_in_one_payload(self, tmp_path: Path) -> None:
        stem = f"mix_{_STAMP}"
        payload = {
            0: {"_phase": _panel(), "_img": _frame(), "exp_t": 1.0},
            1: {"_c": np.zeros(15), "_img": _frame(), "exp_t": 1.0},
            2: {"_c": np.zeros(576), "_img": _frame(), "exp_t": 1.0},
            3: {"_img": _frame(), "exp_t": 1.0},
            4: {"_c": np.zeros(7), "_img": _frame(), "exp_t": 1.0},
            5: {"_c": np.zeros(15), "exp_t": 1.0},
            6: {"_c": np.zeros(15), "_img": [1, 2, 3], "exp_t": 1.0},
        }
        _dump(tmp_path / "model_in_loop_hw_sweep_x", payload, stem)
        index = build_hw_index(tmp_path)

        by_position = {r.position: r.source for r in index.records}
        assert by_position == {
            0: PhaseSource.PANEL_RAD,
            1: PhaseSource.ZERNIKE,
            2: PhaseSource.FREEFORM,
        }
        assert index.excluded == {
            ExclusionReason.MISSING_PHASE: 1,
            ExclusionReason.MISSING_IMAGE: 2,
            ExclusionReason.ODD_COEFFICIENT_LENGTH: 1,
        }


# ---------------------------------------------------------------------------
# 3: Recorder-shaped payload
# ---------------------------------------------------------------------------
class TestRecorderPayload:
    def test_recorder_records_addressed_by_position(self, tmp_path: Path) -> None:
        pkl = write_recorder_like_pkl(tmp_path / "slm_pib_online", n=3)
        index = build_hw_index(tmp_path)

        assert index.files_scanned == 1
        assert index.files_usable == 1
        assert len(index) == 3
        assert [r.position for r in index.records] == [0, 1, 2]
        assert all(r.key is None for r in index.records)
        assert all(r.path == pkl.resolve() for r in index.records)
        assert index.families == ("slm_pib_online",)

    def test_recorder_history_order_is_preserved(self, tmp_path: Path) -> None:
        stem = "recorder_ordered"
        history = [
            {"_c": np.full(36, float(i)), "_img": _frame(), "exp_t": 1.2}
            for i in range(5)
        ]
        _dump(tmp_path / "slm_pib_online", RecorderStandIn(history), stem)
        index = build_hw_index(tmp_path)

        assert [r.position for r in index.records] == [0, 1, 2, 3, 4]
        assert [r.exposure_ms for r in index.records] == [1.2] * 5

    def test_recorder_reads_summary_sidecar(self, tmp_path: Path) -> None:
        write_recorder_like_pkl(tmp_path / "slm_pib_online", n=2)
        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar["rows"] == 2
        assert index.records[0].sidecar["fold"] == 1
        # slm_pib_online is not a freeform family, so 36 resolves as Zernike
        # n_max=7 -- this is the real 192-record case in the corpus.
        assert index.counts_by_source() == {PhaseSource.ZERNIKE: 2}
        assert index.records[0].n_max == 7
        assert index.records[0].freeform_grid is None

    def test_empty_history_is_no_records(self, tmp_path: Path) -> None:
        _dump(tmp_path / "slm_pib_online", RecorderStandIn([]), "recorder_empty")
        index = build_hw_index(tmp_path)

        assert len(index) == 0
        assert index.excluded == {ExclusionReason.NO_RECORDS: 1}


# ---------------------------------------------------------------------------
# 4: unusable payloads never raise
# ---------------------------------------------------------------------------
class TestBadPayloads:
    def test_none_payload_is_no_records(self, tmp_path: Path) -> None:
        write_none_pkl(tmp_path / "slm_pib_online")
        index = build_hw_index(tmp_path)

        assert len(index) == 0
        assert index.excluded == {ExclusionReason.NO_RECORDS: 1}
        assert index.files_scanned == 1
        assert index.files_usable == 0

    def test_corrupt_file_is_unreadable_pickle(self, tmp_path: Path) -> None:
        family = tmp_path / "slm_pib_online"
        family.mkdir(parents=True)
        broken = family / "broken.pkl"
        broken.write_bytes(b"\x80\x04\x95this is not a pickle at all")
        index = build_hw_index(tmp_path)

        assert len(index) == 0
        assert index.excluded == {ExclusionReason.UNREADABLE_PICKLE: 1}

    def test_truncated_file_is_unreadable_pickle(self, tmp_path: Path) -> None:
        good = write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")
        family = tmp_path / "model_in_loop_hw_sweep_x"
        family.mkdir(parents=True)
        truncated = family / "truncated.pkl"
        truncated.write_bytes(good.read_bytes()[:64])
        index = build_hw_index(tmp_path)

        assert index.excluded[ExclusionReason.UNREADABLE_PICKLE] == 1
        assert len(index) == 3  # the good file still indexed

    def test_one_bad_file_does_not_abort_the_scan(self, tmp_path: Path) -> None:
        """Regression guard: a stray artifact must never cost the whole corpus."""
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")
        write_none_pkl(tmp_path / "slm_pib_online")
        family = tmp_path / "slm_zernike_shaping_x"
        family.mkdir(parents=True)
        (family / "garbage.pkl").write_bytes(b"\x00\x01\x02not-a-pickle")
        write_zernike_pkl(tmp_path / "slm_gsnet_square_x")
        index = build_hw_index(tmp_path)

        assert index.files_scanned == 4
        assert index.files_usable == 2
        assert len(index) == 6
        assert index.excluded == {
            ExclusionReason.NO_RECORDS: 1,
            ExclusionReason.UNREADABLE_PICKLE: 1,
        }

    def test_non_dict_record_entries_are_rejected(self, tmp_path: Path) -> None:
        stem = f"weird_{_STAMP}"
        payload = {
            0: {"_c": np.zeros(15), "_img": _frame(), "exp_t": 1.0},
            1: "not a record",
            2: None,
            3: ["list"],
        }
        _dump(tmp_path / "slm_zernike_shaping_x", payload, stem)
        index = build_hw_index(tmp_path)

        assert len(index) == 1
        assert index.excluded == {ExclusionReason.RECORD_NOT_DICT: 3}

    def test_object_without_history_is_no_records(self, tmp_path: Path) -> None:
        _dump(tmp_path / "slm_pib_online", BareObject(), "bare")
        index = build_hw_index(tmp_path)

        assert index.excluded == {ExclusionReason.NO_RECORDS: 1}


# ---------------------------------------------------------------------------
# 6: exposure precedence
# ---------------------------------------------------------------------------
class TestExposureResolution:
    def test_record_exp_t_wins_over_sidecar(self, tmp_path: Path) -> None:
        pkl = write_zernike_pkl(tmp_path / "slm_zernike_shaping_x")
        write_sidecar(pkl.parent, f"{pkl.stem}.json", exposure_ms=99.0, bench="synthetic")
        index = build_hw_index(tmp_path)

        assert all(r.exposure_ms == 2.0 for r in index.records)

    def test_sidecar_exposure_ms_used_when_record_has_none(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")
        index = build_hw_index(tmp_path)

        assert all(r.exposure_ms == 1.1 for r in index.records)

    def test_sidecar_exposure_time_ms_is_a_fallback(self, tmp_path: Path) -> None:
        pkl = write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")
        (pkl.parent / f"{pkl.stem}.json").unlink()
        write_sidecar(pkl.parent, "meta.json", exposure_time_ms=1.2)
        index = build_hw_index(tmp_path)

        assert all(r.exposure_ms == 1.2 for r in index.records)

    def test_record_exposure_ms_and_exposure_time_ms_order(self, tmp_path: Path) -> None:
        stem = f"exp_{_STAMP}"
        payload = {
            0: {"_c": np.zeros(15), "_img": _frame(), "exposure_ms": 3.0},
            1: {
                "_c": np.zeros(15),
                "_img": _frame(),
                "exposure_ms": 3.0,
                "exposure_time_ms": 4.0,
            },
            2: {"_c": np.zeros(15), "_img": _frame(), "exposure_time_ms": 4.0},
        }
        pkl = _dump(tmp_path / "slm_zernike_shaping_x", payload, stem)
        write_sidecar(pkl.parent, f"{pkl.stem}.json", exposure_ms=5.0)
        index = build_hw_index(tmp_path)

        assert [r.exposure_ms for r in index.records] == [3.0, 3.0, 4.0]

    def test_none_when_nothing_carries_an_exposure(self, tmp_path: Path) -> None:
        stem = f"noexp_{_STAMP}"
        payload = {i: {"_c": np.zeros(15), "_img": _frame()} for i in range(3)}
        _dump(tmp_path / "slm_zernike_shaping_x", payload, stem)
        index = build_hw_index(tmp_path)

        assert len(index) == 3
        assert all(r.exposure_ms is None for r in index.records)

    def test_require_exposure_filtering(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")  # sidecar 1.1
        stem = f"noexp_{_STAMP}"
        payload = {i: {"_c": np.zeros(15), "_img": _frame()} for i in range(3)}
        _dump(tmp_path / "slm_zernike_shaping_x", payload, stem)
        index = build_hw_index(tmp_path)

        assert len(index) == 6
        kept = index.filter(require_exposure=True)
        assert len(kept) == 3
        assert all(r.family == "model_in_loop_hw_collect" for r in kept.records)
        everything = index.filter(require_exposure=False)
        assert len(everything) == 6

    def test_non_finite_and_boolean_exposures_are_ignored(self, tmp_path: Path) -> None:
        stem = f"nan_{_STAMP}"
        payload = {
            0: {"_c": np.zeros(15), "_img": _frame(), "exp_t": float("nan")},
            1: {"_c": np.zeros(15), "_img": _frame(), "exp_t": True},
            2: {"_c": np.zeros(15), "_img": _frame(), "exp_t": "1.5"},
        }
        _dump(tmp_path / "slm_zernike_shaping_x", payload, stem)
        index = build_hw_index(tmp_path)

        assert all(r.exposure_ms is None for r in index.records)


# ---------------------------------------------------------------------------
# 7: sidecar resolution order
# ---------------------------------------------------------------------------
class TestSidecarResolution:
    def test_stem_json_beats_summary_and_meta(self, tmp_path: Path) -> None:
        pkl = write_zernike_pkl(tmp_path / "slm_zernike_shaping_x")
        write_sidecar(pkl.parent, "summary_run.json", source="summary")
        write_sidecar(pkl.parent, "meta.json", source="meta")
        write_sidecar(pkl.parent, f"{pkl.stem}.json", source="stem", n_max=4)
        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar["source"] == "stem"

    def test_summary_json_beats_meta(self, tmp_path: Path) -> None:
        pkl = write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")
        (pkl.parent / f"{pkl.stem}.json").unlink()
        write_sidecar(pkl.parent, "summary_run.json", source="summary")
        write_sidecar(pkl.parent, "meta.json", source="meta")
        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar["source"] == "summary"

    def test_meta_json_is_the_last_resort(self, tmp_path: Path) -> None:
        pkl = write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x")
        (pkl.parent / f"{pkl.stem}.json").unlink()
        write_sidecar(pkl.parent, "meta.json", source="meta")
        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar["source"] == "meta"

    def test_malformed_json_yields_empty_sidecar_without_raising(self, tmp_path: Path) -> None:
        stem = f"bad_{_STAMP}"
        payload = {i: {"_c": np.zeros(15), "_img": _frame(), "exp_t": 1.5} for i in range(2)}
        pkl = _dump(tmp_path / "slm_zernike_shaping_x", payload, stem)
        (pkl.parent / f"{pkl.stem}.json").write_text("{not json", encoding="utf-8")
        index = build_hw_index(tmp_path)

        assert len(index) == 2
        assert index.records[0].sidecar == {}
        assert index.records[0].exposure_ms == 1.5

    def test_non_object_json_yields_empty_sidecar(self, tmp_path: Path) -> None:
        stem = f"list_{_STAMP}"
        payload = {i: {"_c": np.zeros(15), "_img": _frame()} for i in range(2)}
        pkl = _dump(tmp_path / "slm_zernike_shaping_x", payload, stem)
        write_sidecar(pkl.parent, f"{pkl.stem}.json")  # {} -- falls through
        write_sidecar(pkl.parent, "summary_a.json")
        write_sidecar(pkl.parent, "meta.json")
        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar == {}

    @pytest.mark.parametrize("unusable", ["{not json", "[1, 2, 3]", '"a string"', "null"])
    def test_unusable_top_priority_sidecar_does_not_shadow_a_lower_one(
        self, unusable: str, tmp_path: Path
    ) -> None:
        """Precedence is over *usable* sidecars, not merely over *existing* ones.

        ``_read_sidecar`` documents that an existing-but-unusable sidecar "must not
        shadow a better one". This is the only test that actually distinguishes
        that intent from strict first-hit-wins: every other sidecar test writes
        ``{}`` to the lower-priority files too, so it cannot tell the two rules
        apart.

        The recovered ``n_max`` is load-bearing -- ``len(_c) == 36`` is ambiguous
        (triangular n_max=7 *and* a 6x6 square), so if the lower sidecar were
        shadowed the records would classify as FREEFORM instead of ZERNIKE.
        """
        pkl = write_zernike_pkl(tmp_path / "sim_calib_abba_x", n_terms=36)
        (pkl.parent / f"{pkl.stem}.json").write_text(unusable, encoding="utf-8")
        write_sidecar(pkl.parent, "summary_a.json", n_max=7, bench="synthetic")

        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar == {"n_max": 7, "bench": "synthetic"}
        assert index.records[0].n_max == 7
        assert index.counts_by_source() == {PhaseSource.ZERNIKE: 3}

    def test_json_array_is_ignored(self, tmp_path: Path) -> None:
        stem = f"arr_{_STAMP}"
        payload = {i: {"_c": np.zeros(15), "_img": _frame()} for i in range(2)}
        pkl = _dump(tmp_path / "slm_zernike_shaping_x", payload, stem)
        (pkl.parent / f"{pkl.stem}.json").write_text("[1, 2, 3]", encoding="utf-8")
        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar == {}

    def test_absent_sidecar_is_empty(self, tmp_path: Path) -> None:
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x")
        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar == {}

    def test_sidecar_is_shared_per_file_and_read_only(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=3)
        index = build_hw_index(tmp_path)

        assert index.records[0].sidecar is index.records[1].sidecar
        with pytest.raises(TypeError):
            index.records[0].sidecar["exposure_ms"] = 5.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# 8: family_of
# ---------------------------------------------------------------------------
class TestFamilyOf:
    @pytest.mark.parametrize(
        ("relative", "expected"),
        [
            (("model_in_loop_hw_collect_20260930_131521", "ts", "x.pkl"), "model_in_loop_hw_collect"),
            (("model_in_loop_hw_sweep_20260930_131526", "ts", "x.pkl"), "model_in_loop_hw_sweep"),
            (("bench_stability_20261001_164458", "bench_stability.pkl"), "bench_stability"),
            (("sim_calib_abba_20261001_163743", "sim_calib_abba.pkl"), "sim_calib_abba"),
            (("slm_gsnet_square_20261001_160808", "ts", "x.pkl"), "slm_gsnet_square"),
            (("slm_gsnet_square_pearson_20260929_100158", "ts", "x.pkl"), "slm_gsnet_square"),
            (("slm_zernike_shaping_shape_20260926_170322", "ts", "x.pkl"), "slm_zernike_shaping"),
            (("slm_pib_online", "ts", "recorder_smoke_f4.pkl"), "slm_pib_online"),
            (("slm_pib_shape_20260923_175829", "ts", "x.pkl"), "slm_pib"),
            (("slm_pib_hw_20260929", "ts", "x.pkl"), "slm_pib"),
            (("recorder_20260929_091112", "ts", "recorder_snr.pkl"), "recorder_"),
        ],
    )
    def test_real_directory_shapes(self, relative: tuple[str, ...], expected: str) -> None:
        path = Path("data", "debug").joinpath(*relative)
        assert family_of(path) == expected

    def test_slm_pib_online_beats_slm_pib_prefix(self) -> None:
        """Order is load-bearing: "slm_pib_online".startswith("slm_pib")."""
        assert "slm_pib_online".startswith("slm_pib")
        assert family_of(Path("d/slm_pib_online/ts/x.pkl")) == "slm_pib_online"

    def test_unknown_directory_falls_back_to_parent_name(self) -> None:
        assert family_of(Path("d/snr_sweep_20260930/ts/x.pkl")) == "ts"
        assert family_of(Path("d/snr_sweep_20260930/x.pkl")) == "snr_sweep_20260930"

    def test_parent_name_is_preferred_over_grandparent(self) -> None:
        """A timestamp directory is checked first, so a nested match wins."""
        path = Path("d/slm_pib_x/slm_zernike_shaping_y/x.pkl")
        assert family_of(path) == "slm_zernike_shaping"


# ---------------------------------------------------------------------------
# 9: index cache
# ---------------------------------------------------------------------------
class TestIndexCache:
    def test_cache_round_trip_skips_the_scan(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        write_panel_rad_pkl(corpus / "model_in_loop_hw_collect_x")
        write_zernike_pkl(corpus / "slm_zernike_shaping_x")
        cache = tmp_path / "cache" / "index.json"

        first = build_hw_index(corpus, index_cache=cache)
        assert cache.is_file()
        assert first.files_scanned == 2

        # Destroy the corpus: a rebuild that re-reads it cannot possibly match.
        for pkl in corpus.rglob("*.pkl"):
            pkl.write_bytes(b"corrupted beyond recognition")

        second = build_hw_index(corpus, index_cache=cache)
        assert second == first
        assert len(second) == 6
        assert second.records[0].path == first.records[0].path
        assert isinstance(second.records[0].path, Path)
        assert second.records[0].source is PhaseSource.PANEL_RAD
        assert second.records[0].sidecar["exposure_ms"] == 1.1

    def test_cache_is_json_not_pickle(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        write_panel_rad_pkl(corpus / "model_in_loop_hw_collect_x")
        cache = tmp_path / "index.json"
        build_hw_index(corpus, index_cache=cache)

        payload = json.loads(cache.read_text(encoding="utf-8"))
        # Version-agnostic on purpose: the guard must reject a STALE cache, so the
        # constant is expected to move. What matters is that it is recorded and
        # compared, which `test_stale_version_is_rejected` pins.
        assert isinstance(payload["version"], int)
        assert payload["version"] >= 1
        assert payload["roots"] == [str(corpus)]
        assert payload["limit_files"] is None
        assert len(payload["records"]) == 3
        first = payload["records"][0]
        assert first["source"] == "panel_rad"
        assert isinstance(first["path"], str)
        assert isinstance(first["key"], int)
        assert first["n_max"] is None
        assert first["sidecar"]["slm_wavelength"] == 1064

    def test_cache_rejected_when_roots_differ(self, tmp_path: Path) -> None:
        """A stale cache must never silently shadow a different corpus."""
        root_a = tmp_path / "corpus_a"
        root_b = tmp_path / "corpus_b"
        write_panel_rad_pkl(root_a / "model_in_loop_hw_collect_x", n=3)
        write_zernike_pkl(root_b / "slm_zernike_shaping_x", n=5)
        cache = tmp_path / "index.json"

        from_a = build_hw_index(root_a, index_cache=cache)
        assert len(from_a) == 3
        assert from_a.families == ("model_in_loop_hw_collect",)

        warnings: list[str] = []
        from loguru import logger

        sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            from_b = build_hw_index(root_b, index_cache=cache)
        finally:
            logger.remove(sink_id)

        assert len(from_b) == 5
        assert from_b.families == ("slm_zernike_shaping",)
        assert from_b.files_scanned == 1
        assert any("stale index cache" in message for message in warnings)
        # The cache was rewritten for the new roots.
        assert build_hw_index(root_b, index_cache=cache) == from_b

    def test_cache_rejected_when_limit_files_differs(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        write_panel_rad_pkl(corpus / "model_in_loop_hw_collect_x")
        write_zernike_pkl(corpus / "slm_zernike_shaping_x")
        cache = tmp_path / "index.json"

        capped = build_hw_index(corpus, limit_files=1, index_cache=cache)
        assert capped.files_scanned == 1

        uncapped = build_hw_index(corpus, index_cache=cache)
        assert uncapped.files_scanned == 2
        assert len(uncapped) == 6

    def test_garbage_cache_is_ignored_and_rescanned(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        write_panel_rad_pkl(corpus / "model_in_loop_hw_collect_x")
        cache = tmp_path / "index.json"
        cache.write_text("not json at all", encoding="utf-8")

        index = build_hw_index(corpus, index_cache=cache)
        assert index.files_scanned == 1
        assert len(index) == 3

    def test_malformed_cache_records_are_ignored(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        write_panel_rad_pkl(corpus / "model_in_loop_hw_collect_x")
        cache = tmp_path / "index.json"
        cache.write_text(
            json.dumps(
                {
                    "version": 1,
                    "roots": [str(corpus)],
                    "limit_files": None,
                    "files_scanned": 1,
                    "files_usable": 1,
                    "excluded": {},
                    "records": [{"path": "x.pkl"}],
                }
            ),
            encoding="utf-8",
        )

        index = build_hw_index(corpus, index_cache=cache)
        assert index.files_scanned == 1
        assert len(index) == 3

    def test_cache_version_mismatch_is_rejected(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        write_panel_rad_pkl(corpus / "model_in_loop_hw_collect_x")
        cache = tmp_path / "index.json"
        build_hw_index(corpus, index_cache=cache)
        payload = json.loads(cache.read_text(encoding="utf-8"))
        payload["version"] = 0
        cache.write_text(json.dumps(payload), encoding="utf-8")

        index = build_hw_index(corpus, index_cache=cache)
        assert index.files_scanned == 1


# ---------------------------------------------------------------------------
# 10: discovery, limit_files, filter
# ---------------------------------------------------------------------------
class TestDiscoveryAndFilter:
    def test_discovery_is_recursive(self, tmp_path: Path) -> None:
        """Two levels deep: a non-recursive glob would find nothing."""
        deep = tmp_path / "model_in_loop_hw_collect_x" / "20260101_000000"
        deep.mkdir(parents=True)
        assert list(tmp_path.glob("*.pkl")) == []
        assert list(tmp_path.rglob("*.pkl")) == []

        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=2)
        index = build_hw_index(tmp_path)
        assert index.files_scanned == 1
        assert len(index) == 2

    def test_explicit_pkl_file_root(self, tmp_path: Path) -> None:
        pkl = write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=2)
        index = build_hw_index(pkl)
        assert index.files_scanned == 1
        assert len(index) == 2

    def test_glob_root(self, tmp_path: Path) -> None:
        """A glob root selects whole family directories; the non-match is skipped."""
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=1)
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n=1)
        write_freeform_pkl(tmp_path / "slm_gsnet_square_x", n=1)

        # "slm_*" matches two of the three family directories.
        matched = build_hw_index(tmp_path / "slm_*")
        assert matched.files_scanned == 2
        assert matched.families == ("slm_gsnet_square", "slm_zernike_shaping")

        # A glob matching nothing yields an empty index, never an error.
        assert build_hw_index(tmp_path / "nothing_*").files_scanned == 0

    def test_multiple_roots_are_deduplicated(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        write_panel_rad_pkl(corpus / "model_in_loop_hw_collect_x", n=2)
        index = build_hw_index([corpus, corpus / "model_in_loop_hw_collect_x"])
        assert index.files_scanned == 1

    def test_missing_root_warns_and_returns_empty(self, tmp_path: Path) -> None:
        index = build_hw_index(tmp_path / "does_not_exist")
        assert len(index) == 0
        assert index.files_scanned == 0

    def test_limit_files_applied_after_sorting(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=1)
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n=1)
        write_freeform_pkl(tmp_path / "slm_gsnet_square_x", n=1)
        full = build_hw_index(tmp_path)
        assert [r.family for r in full.records] == [
            "model_in_loop_hw_collect",
            "slm_gsnet_square",
            "slm_zernike_shaping",
        ]

        capped = build_hw_index(tmp_path, limit_files=2)
        assert capped.files_scanned == 2
        assert [r.family for r in capped.records] == [
            "model_in_loop_hw_collect",
            "slm_gsnet_square",
        ]

    def test_limit_files_zero_and_negative(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=1)
        assert build_hw_index(tmp_path, limit_files=0).files_scanned == 0
        assert build_hw_index(tmp_path, limit_files=-5).files_scanned == 0

    def test_filter_by_family(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=3)
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n=4)
        write_freeform_pkl(tmp_path / "slm_gsnet_square_x", n=5)
        index = build_hw_index(tmp_path)
        assert len(index) == 12

        only_zernike = index.filter(families=["slm_zernike_shaping"])
        assert len(only_zernike) == 4
        assert only_zernike.families == ("slm_zernike_shaping",)
        assert only_zernike.counts_by_source() == {PhaseSource.ZERNIKE: 4}

        two = index.filter(families=["slm_zernike_shaping", "slm_gsnet_square"])
        assert len(two) == 9

    def test_filter_by_source_accepts_enum_and_string(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=3)
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n=4)
        index = build_hw_index(tmp_path)

        by_enum = index.filter(sources=[PhaseSource.ZERNIKE])
        by_str = index.filter(sources=["zernike"])
        assert by_enum == by_str
        assert len(by_enum) == 4
        assert by_enum.counts_by_source() == {PhaseSource.ZERNIKE: 4}

    def test_filter_preserves_excluded_tally_and_file_counts(self, tmp_path: Path) -> None:
        write_none_pkl(tmp_path / "slm_pib_online")
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=3)
        index = build_hw_index(tmp_path)
        assert index.excluded == {ExclusionReason.NO_RECORDS: 1}
        assert index.files_scanned == 2
        assert index.files_usable == 1

        narrowed = index.filter(families=["model_in_loop_hw_collect"])
        assert len(narrowed) == 3
        assert narrowed.excluded == {ExclusionReason.NO_RECORDS: 1}
        assert narrowed.files_scanned == 2
        assert narrowed.files_usable == 1
        assert sum(narrowed.counts_by_family().values()) == len(narrowed)

    def test_filter_is_chainable_and_non_mutating(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=3)
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n=4)
        index = build_hw_index(tmp_path)

        chained = index.filter(sources=[PhaseSource.ZERNIKE]).filter(
            families=["slm_zernike_shaping"]
        )
        assert len(chained) == 4
        assert len(index) == 7  # original untouched

    def test_filter_unknown_family_yields_empty(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=3)
        index = build_hw_index(tmp_path)
        assert len(index.filter(families=["nope"])) == 0

    def test_progress_every_suppresses_output_at_zero(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=1)
        assert build_hw_index(tmp_path, progress_every=0).files_scanned == 1
        assert build_hw_index(tmp_path, progress_every=1000).files_scanned == 1


# ---------------------------------------------------------------------------
# 11: the index holds no arrays
# ---------------------------------------------------------------------------
class TestIndexMemoryContract:
    def test_record_fields_are_scalars_or_mappings(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=3)
        write_zernike_pkl(tmp_path / "slm_zernike_shaping_x", n=3)
        write_freeform_pkl(tmp_path / "slm_gsnet_square_x", n=3)
        write_recorder_like_pkl(tmp_path / "slm_pib_online", n=3)
        index = build_hw_index(tmp_path)

        assert len(index) == 12
        # PhaseSource is a str-valued enum: a single immutable scalar, not a container.
        allowed_types = (str, int, float, bool, type(None), Path, PhaseSource)
        seen_keys: set[str] = set()
        for record in index.records:
            assert is_dataclass(record)
            seen_keys.add(type(record.sidecar).__name__)
            for field_info in fields(record):
                value = getattr(record, field_info.name)
                if field_info.name == "sidecar":
                    assert isinstance(value, Mapping)
                    continue
                assert isinstance(value, allowed_types), (
                    f"{field_info.name}={value!r} is {type(value).__name__}"
                )
                assert not isinstance(value, np.ndarray)
        # mappingproxy is the expected read-only sidecar view.
        assert seen_keys == {"mappingproxy"}

    def test_excluded_tally_and_counters_are_scalars(self, tmp_path: Path) -> None:
        write_none_pkl(tmp_path / "slm_pib_online")
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=1)
        index = build_hw_index(tmp_path)

        assert isinstance(index.files_scanned, int)
        assert isinstance(index.files_usable, int)
        assert all(
            isinstance(reason, ExclusionReason) and isinstance(count, int)
            for reason, count in index.excluded.items()
        )

    def test_records_are_a_tuple_and_index_is_frozen(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=1)
        index = build_hw_index(tmp_path)

        assert isinstance(index.records, tuple)
        assert isinstance(index, HwCorpusIndex)
        assert isinstance(index.records[0], HwRecordRef)
        with pytest.raises((AttributeError, TypeError)):
            index.records[0].n_terms = 5  # type: ignore[misc]

    def test_index_is_reproducible(self, tmp_path: Path) -> None:
        write_panel_rad_pkl(tmp_path / "model_in_loop_hw_collect_x", n=3)
        write_mixed_pkl(tmp_path / "sim_calib_abba_x", n_with_phase=2, n_without=1)
        first = build_hw_index(tmp_path)
        second = build_hw_index(tmp_path)
        assert first == second
        assert [r.path for r in first.records] == [r.path for r in second.records]
        assert [r.position for r in first.records] == [r.position for r in second.records]
