"""Tests for :mod:`ml.shaping.evaluate` -- the evaluation CLI.

No hardware, no real corpus, no ``.pkl``: the corpus loaders are monkeypatched
to yield a handful of synthetic records spread over several **fake files** and
two **fake families**, because the file-level-first aggregation is the property
under test.

``torch`` lives in the optional ``ml`` dependency group, hence the
``importorskip``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

import torch.utils.data as _tud

from ml.shaping.config import ShapingConfig, UNKNOWN_FOV
from ml.shaping.evaluate import (
    DIFF_CONVENTION,
    EvaluationResult,
    _build_parser,
    _fmt,
    _paired_rows,
    _stats_block,
    _verdict,
    evaluate,
    main,
)

_GRID = 16
#: (file, family, fov) triples -- deliberately more files than families, and one
#: file whose fov has no calibrated forward model.
_LAYOUT = [
    ("file_a.pkl", "fam_a", 64),
    ("file_a.pkl", "fam_a", 64),
    ("file_b.pkl", "fam_a", 320),
    ("file_c.pkl", "fam_b", 248),
    ("file_c.pkl", "fam_b", 248),
    ("file_d.pkl", "fam_b", 1944),
    ("file_e.pkl", "fam_b", UNKNOWN_FOV),
]


def _sample(position: int) -> dict[str, Any]:
    """Build one synthetic record matching the corpus sample contract."""
    path, family, fov = _LAYOUT[position % len(_LAYOUT)]
    generator = torch.Generator().manual_seed(500 + position)
    return {
        "phase_cos": torch.rand((1, _GRID, _GRID), generator=generator),
        "phase_sin": torch.rand((1, _GRID, _GRID), generator=generator),
        "image": torch.rand((1, _GRID, _GRID), generator=generator),
        "contrast": torch.ones((_GRID, _GRID)),
        "exposure_ms": 1.0,
        "exposure_log10": torch.tensor([0.0]),
        "source": "panel_rad",
        "family": family,
        "sample_idx": position,
        "path": path,
        "fov_px": fov,
    }


class _SyntheticDataset(_tud.Dataset):
    """A flat list of synthetic samples, no file access."""

    def __init__(self, count: int) -> None:
        self._items = [_sample(i) for i in range(count)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._items[index]


@pytest.fixture
def fake_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the corpus loaders with in-memory ones over the fake layout."""

    def _fake(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        train = _tud.DataLoader(_SyntheticDataset(7), batch_size=4, shuffle=False)
        val = _tud.DataLoader(_SyntheticDataset(7), batch_size=4, shuffle=False)
        test = _tud.DataLoader(_SyntheticDataset(7), batch_size=4, shuffle=False)
        return train, val, test

    monkeypatch.setattr("ml.shaping.evaluate.build_shaping_loaders", _fake)


def _tiny_config(tmp_path: Path) -> ShapingConfig:
    """A config small enough to evaluate in a unit test."""
    return ShapingConfig(
        grid=_GRID,
        features=(4, 8),
        batch_size=4,
        num_workers=0,
        device="cpu",
        wandb=False,
        checkpoint_dir=str(tmp_path),
    )


def _write_checkpoint(path: Path, cfg: ShapingConfig) -> Path:
    """Write a checkpoint an untrained :class:`PhasorUNet` can load."""
    from ml.shaping.models import build_phasor_unet

    model = build_phasor_unet(config=cfg, device="cpu")
    torch.save(
        {"model_state": model.state_dict(), "epoch": 0, "global_step": 0}, path
    )
    return path


# ----------------------------------------------------------------------
# Diff convention -- the sign is the easiest thing to get backwards
# ----------------------------------------------------------------------
def test_diff_convention_is_baseline_minus_model() -> None:
    """A better model must yield a POSITIVE difference."""
    files = [
        {"path": "f1", "family": "x", "phasor_mae": 0.2, "phasor_mae_baseline": 0.5},
        {"path": "f2", "family": "x", "phasor_mae": 0.1, "phasor_mae_baseline": 0.4},
    ]
    diffs = _paired_rows(files, "phasor_mae")
    assert diffs == pytest.approx([0.3, 0.3])
    assert all(d > 0 for d in diffs), "positive diff must mean the model is better"
    assert "baseline_minus_model" in DIFF_CONVENTION


def test_paired_rows_drops_unusable_entries() -> None:
    """``None`` and NaN entries must not become differences."""
    files = [
        {"phasor_mae": 0.2, "phasor_mae_baseline": 0.5},
        {"phasor_mae": None, "phasor_mae_baseline": 0.4},
        {"phasor_mae": 0.3, "phasor_mae_baseline": None},
        {"phasor_mae": float("nan"), "phasor_mae_baseline": 0.1},
    ]
    assert _paired_rows(files, "phasor_mae") == pytest.approx([0.3])


# ----------------------------------------------------------------------
# File-level-first aggregation
# ----------------------------------------------------------------------
def test_statistics_consume_file_level_values(fake_loaders: None, tmp_path: Path) -> None:
    """Every paired test must be fed one value per FILE, never per record."""
    import ml.shaping.evaluate as module

    checkpoint = _write_checkpoint(tmp_path / "last.pt", _tiny_config(tmp_path))
    seen: list[int] = []
    original = module.paired_comparison

    def _spy(diffs: Any, label: str = "") -> dict[str, Any]:
        seen.append(len(list(diffs)))
        return original(diffs, label)

    module.paired_comparison = _spy
    try:
        result = evaluate(_tiny_config(tmp_path), checkpoint, device_spec="cpu")
    finally:
        module.paired_comparison = original

    n_records = len(result.records)
    n_files = len({row["path"] for row in result.records})
    assert n_records > n_files, "the fixture must have several records per file"
    assert seen, "paired_comparison was never called"
    assert all(count <= n_files for count in seen), (
        f"a paired test received more values ({seen}) than there are files "
        f"({n_files}); statistics must be file-level"
    )


def test_every_record_is_aggregated_into_exactly_one_file_row(
    fake_loaders: None, tmp_path: Path
) -> None:
    """File rows partition the records -- none dropped, none duplicated."""
    checkpoint = _write_checkpoint(tmp_path / "last.pt", _tiny_config(tmp_path))
    result = evaluate(_tiny_config(tmp_path), checkpoint, device_spec="cpu")
    assert sum(row["n_records"] for row in result.files) == len(result.records)
    assert len({row["path"] for row in result.files}) == len(result.files)


# ----------------------------------------------------------------------
# Physics-metric exclusions
# ----------------------------------------------------------------------
def test_unsupported_fov_is_excluded_with_a_reason(
    fake_loaders: None, tmp_path: Path
) -> None:
    """Records with no calibrated padding drop out of the physics metric."""
    checkpoint = _write_checkpoint(tmp_path / "last.pt", _tiny_config(tmp_path))
    result = evaluate(_tiny_config(tmp_path), checkpoint, device_spec="cpu")

    excluded = [r for r in result.records if not r["physics_supported"]]
    assert excluded, "the fixture contains unsupported fov values"
    for row in excluded:
        assert row["physics_l1"] is None, "an excluded row must not carry a value"
        assert row["physics_exclusion_reason"] in {
            "fov_unknown",
            "fov_1944_no_calibrated_padding",
        }
    for row in result.records:
        if row["physics_supported"]:
            assert row["physics_exclusion_reason"] is None
            assert row["physics_l1"] is not None


def test_exclusion_reasons_are_counted_separately(
    fake_loaders: None, tmp_path: Path
) -> None:
    """The two exclusion causes must be distinguishable, not lumped together."""
    checkpoint = _write_checkpoint(tmp_path / "last.pt", _tiny_config(tmp_path))
    result = evaluate(_tiny_config(tmp_path), checkpoint, device_spec="cpu")
    assert result.exclusions.get("fov_1944_no_calibrated_padding", 0) > 0
    assert result.exclusions.get("fov_unknown", 0) > 0
    assert sum(result.exclusions.values()) == len(
        [r for r in result.records if not r["physics_supported"]]
    )


# ----------------------------------------------------------------------
# Statistics honesty
# ----------------------------------------------------------------------
def test_min_attainable_p_is_reported_beside_every_p() -> None:
    """A p-value is never reported without the floor that protocol can reach."""
    files = [
        {"path": f"f{i}", "family": "x", "phasor_mae": 0.2, "phasor_mae_baseline": 0.5}
        for i in range(4)
    ]
    block = _stats_block(files, "x")["metrics"]["phasor_mae"]
    assert block["p_sign_flip"] is not None
    assert block["min_attainable_p"] == pytest.approx(2.0 / 2**4)


def test_small_n_gets_no_significance_verdict() -> None:
    """Four files structurally cannot reach p<0.05, so no verdict is issued."""
    files = [
        {"path": f"f{i}", "family": "x", "phasor_mae": 0.2, "phasor_mae_baseline": 0.9}
        for i in range(4)
    ]
    verdict = _stats_block(files, "x")["metrics"]["phasor_mae"]["verdict"]
    assert "cannot reach" in verdict
    assert "significant (" not in verdict


def test_verdict_reports_effect_size_only_when_n_is_tiny() -> None:
    """The verdict helper declines inference for degenerate sample sizes."""
    assert "n/a" in _verdict(0.5, 0.5, 1)
    assert "cannot reach" in _verdict(0.5, 0.125, 4)
    assert "significant" in _verdict(0.01, 0.03125, 5)
    assert "not significant" in _verdict(0.4, 0.03125, 6)


def test_holm_adjustment_is_present_per_family(
    fake_loaders: None, tmp_path: Path
) -> None:
    """Multiple families are multiplicity-corrected rather than ranked raw."""
    checkpoint = _write_checkpoint(tmp_path / "last.pt", _tiny_config(tmp_path))
    result = evaluate(_tiny_config(tmp_path), checkpoint, device_spec="cpu")
    for family in ("fam_a", "fam_b"):
        block = result.stats[family]["metrics"]["phasor_mae"]
        assert "p_holm" in block
        assert block["p_holm"] >= block["p_sign_flip"] - 1e-12


# ----------------------------------------------------------------------
# Output contract
# ----------------------------------------------------------------------
def test_payload_is_json_serialisable_and_carries_the_caveats(
    fake_loaders: None, tmp_path: Path
) -> None:
    """``metrics.json`` round-trips and states its own limitations."""
    checkpoint = _write_checkpoint(tmp_path / "last.pt", _tiny_config(tmp_path))
    result = evaluate(_tiny_config(tmp_path), checkpoint, device_spec="cpu")
    text = json.dumps(result.payload)
    restored = json.loads(text)
    assert restored["diff_convention"] == DIFF_CONVENTION
    assert "0.748" in restored["noise_floor_note"]
    assert "collinear" in restored["family_caveat_note"]
    assert set(restored["families"][0]) >= {"family", "n_files", "phasor_mae"}


def test_per_family_table_plus_pooled(fake_loaders: None, tmp_path: Path) -> None:
    """One row per family, plus a pooled block that is labelled as pooled."""
    checkpoint = _write_checkpoint(tmp_path / "last.pt", _tiny_config(tmp_path))
    result = evaluate(_tiny_config(tmp_path), checkpoint, device_spec="cpu")
    names = {row["family"] for row in result.families}
    assert names == {"fam_a", "fam_b"}
    assert result.stats["pooled"]["label"] == "pooled"


def test_missing_checkpoint_is_a_clean_error(tmp_path: Path) -> None:
    """No checkpoint is a usage error with a message, never a traceback."""
    with pytest.raises(FileNotFoundError):
        evaluate(_tiny_config(tmp_path), tmp_path / "absent.pt", device_spec="cpu")


def test_checkpoint_without_model_state_is_rejected(tmp_path: Path) -> None:
    """A malformed checkpoint fails loudly rather than scoring nothing."""
    bad = tmp_path / "bad.pt"
    torch.save({"epoch": 0}, bad)
    with pytest.raises(KeyError):
        evaluate(_tiny_config(tmp_path), bad, device_spec="cpu")


def test_cli_missing_checkpoint_exits_two(tmp_path: Path) -> None:
    """``main`` turns the missing checkpoint into exit code 2."""
    code = main(
        [
            "--checkpoint", str(tmp_path / "nope.pt"),
            "--grid", str(_GRID),
            "--checkpoint-dir", str(tmp_path),
        ]
    )
    assert code == 2


# ----------------------------------------------------------------------
# Parser parity + style guards
# ----------------------------------------------------------------------
def test_parser_defaults_equal_the_dataclass() -> None:
    """Every exposed CLI default must come from ``ShapingConfig``."""
    import dataclasses

    defaults = vars(_build_parser().parse_args(["--checkpoint", "x.pt"]))
    expected = dataclasses.asdict(ShapingConfig())
    mismatches = {
        key: (defaults.get(key, "<missing>"), value)
        for key, value in expected.items()
        if key in defaults and defaults[key] != value
    }
    assert not mismatches, f"parser defaults drifted: {mismatches}"


def test_module_has_no_print_or_relative_import() -> None:
    """Repo style guard."""
    source = Path(
        __import__("ml.shaping.evaluate", fromlist=["__file__"]).__file__
    ).read_text(encoding="utf-8")
    assert not re.search(r"^\s*print\(", source, re.MULTILINE)
    assert not re.search(r"^\s*from\s+\.", source, re.MULTILINE)


def test_no_percent_style_logger_calls() -> None:
    """loguru formats with ``str.format``; printf placeholders print verbatim."""
    package_dir = Path(__import__("ml.shaping", fromlist=["__file__"]).__file__).parent
    pattern = re.compile(r"logger\.(?:debug|info|warning|error|exception)\(")
    offenders: list[str] = []
    for source in sorted(package_dir.glob("*.py")):
        lines = source.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if pattern.search(line):
                window = "\n".join(lines[number - 1 : number + 3])
                if re.search(r'"[^"\n]*%[0-9.\-]*[dsfge]', window):
                    offenders.append(f"{source.name}:{number}")
    assert not offenders, f"printf-style loguru calls: {offenders}"


def test_fmt_renders_na_for_missing_values() -> None:
    """Console formatting must show ``n/a``, never ``None`` or ``nan``."""
    assert _fmt(None) == "n/a"
    assert _fmt(float("nan")) == "n/a"
    assert _fmt(0.5) == "0.50000"


def test_result_dataclass_defaults_are_empty() -> None:
    """The result container is constructible with no arguments."""
    result = EvaluationResult()
    assert result.records == []
    assert result.payload == {}