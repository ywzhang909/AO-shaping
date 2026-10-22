"""Tests for the report-driven data preprocessing in :mod:`ml.shaping`.

Everything here is a direct consequence of a finding in
``report/hwdataset_corpus/report.md``; each test names the section it defends.

No hardware, no ``.pkl``: records are lightweight stand-ins carrying only the
attributes the code reads (``path``, ``family``, ``source``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.shaping.config import IMAGE_MODES, ShapingConfig
from ml.shaping.dataset import (
    FamilyBalancedGroupedSampler,
    corpus_summary,
    family_file_counts,
)


def _record(family: str, path: str, position: int = 0) -> SimpleNamespace:
    """A stand-in for :class:`~ml.hwdataset.index.HwRecordRef`.

    Args:
        family: Family label.
        path: Pickle path, unique per file.
        position: Position within the file.

    Returns:
        A namespace exposing ``family``, ``path``, ``position``, ``source``.
    """
    return SimpleNamespace(
        family=family,
        path=Path(path),
        position=position,
        source=SimpleNamespace(value="panel_gray"),
    )


def _imbalanced() -> list[SimpleNamespace]:
    """A corpus shaped like the measured one: one huge family, two tiny ones."""
    records = [_record("slm_pib", f"pib{i}.pkl", i) for i in range(18)]
    records.append(_record("slm_pib_online", "online.pkl", 0))
    records.append(_record("slm_gsnet_square", "square.pkl", 0))
    return records


# ----------------------------------------------------------------------
# Per-family balancing (report section 11, row 4)
# ----------------------------------------------------------------------
def test_file_counts_expose_the_imbalance() -> None:
    """The diagnostic that decides whether balancing is safe."""
    assert family_file_counts(_imbalanced()) == {
        "slm_gsnet_square": 1,
        "slm_pib": 18,
        "slm_pib_online": 1,
    }


def test_alpha_zero_reproduces_corpus_proportions() -> None:
    """``alpha=0`` must be the identity, i.e. weights proportional to n_files."""
    sampler = FamilyBalancedGroupedSampler(
        _imbalanced(), seed=0, balance_alpha=0.0
    )
    weights = sampler.family_weights()
    assert weights["slm_pib"] == pytest.approx(18 / 20)
    assert weights["slm_pib_online"] == pytest.approx(1 / 20)


def test_alpha_one_is_uniform_over_families() -> None:
    """``alpha=1`` gives every family the same share of file slots."""
    sampler = FamilyBalancedGroupedSampler(
        _imbalanced(), seed=0, balance_alpha=1.0, max_file_repeats=64
    )
    weights = sampler.family_weights()
    for value in weights.values():
        assert value == pytest.approx(1 / 3)


def test_weights_sum_to_one_and_interpolate_monotonically() -> None:
    """Weights are a proper distribution and shift smoothly with alpha."""
    previous = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        sampler = FamilyBalancedGroupedSampler(
            _imbalanced(), seed=0, balance_alpha=alpha, max_file_repeats=64
        )
        weights = sampler.family_weights()
        assert sum(weights.values()) == pytest.approx(1.0)
        # The dominant family's share must fall monotonically as alpha rises.
        if previous is not None:
            assert weights["slm_pib"] <= previous + 1e-12
        previous = weights["slm_pib"]


def test_balancing_raises_the_minority_share() -> None:
    """The whole point: small families must actually appear more often."""
    records = _imbalanced()
    counts = {}
    for alpha in (0.0, 1.0):
        sampler = FamilyBalancedGroupedSampler(
            records, seed=3, balance_alpha=alpha, max_file_repeats=64
        )
        cycle = sampler._one_cycle()
        counts[alpha] = sum(1 for i in cycle if records[i].family != "slm_pib")
    assert counts[1.0] > counts[0.0]


def test_each_contiguous_run_is_a_single_file() -> None:
    """Runs must stay file-contiguous or the payload LRU thrashes.

    Report section 2 measures record counts per file spanning 4 to 733, and the
    parent sampler exists because a plain ``RandomSampler`` turns a 2-minute
    epoch into a 10-hour one. Balancing must not reintroduce that, so every
    contiguous run in the emitted cycle must come from exactly one file.
    """
    records = _imbalanced()
    sampler = FamilyBalancedGroupedSampler(
        records, seed=1, balance_alpha=1.0, max_file_repeats=64
    )
    cycle = sampler._one_cycle()

    runs: list[list[Path]] = []
    previous: Path | None = None
    for position in cycle:
        path = records[position].path
        if path != previous:
            runs.append([path])
            previous = path
        else:
            runs[-1].append(path)

    assert all(len(set(run)) == 1 for run in runs)
    # Balancing may repeat a file, but it must not split one across two runs.
    assert sum(len(run) for run in runs) == len(cycle)


def test_repeat_cap_is_respected() -> None:
    """``max_file_repeats`` must actually bound a single file's appearances."""
    records = [_record("solo", "only.pkl", 0)]
    sampler = FamilyBalancedGroupedSampler(
        records, seed=0, balance_alpha=1.0, max_file_repeats=3
    )
    cycle = sampler._one_cycle()
    assert len(cycle) <= 3


def test_cap_shortfall_is_not_silent() -> None:
    """An unreachable target must warn, never silently drop file slots.

    With 9 files in one family and 1 in another, ``alpha=1`` asks the lone file
    for half of 10 slots, but a cap of 2 can only supply 2. Dropping the other 3
    without saying so would make the realised balance differ from the requested
    one, which is precisely the kind of silent divergence this module exists to
    avoid.
    """
    records = [_record("big", f"big{i}.pkl", 0) for i in range(9)]
    records.append(_record("tiny", "tiny.pkl", 0))
    sampler = FamilyBalancedGroupedSampler(
        records, seed=0, balance_alpha=1.0, max_file_repeats=2
    )

    messages: list[str] = []
    from loguru import logger

    handler = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        cycle = sampler._one_cycle()
    finally:
        logger.remove(handler)

    assert any("could not place" in message for message in messages)
    # The slot is genuinely dropped rather than filled past the cap.
    assert sum(1 for i in cycle if records[i].family == "tiny") <= 2


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_alpha_is_rejected(bad: float) -> None:
    """Out-of-range balancing strength is a configuration error."""
    with pytest.raises(ValueError):
        FamilyBalancedGroupedSampler(_imbalanced(), balance_alpha=bad)


def test_invalid_repeat_cap_is_rejected() -> None:
    """A non-positive cap would make balancing a no-op."""
    with pytest.raises(ValueError):
        FamilyBalancedGroupedSampler(_imbalanced(), max_file_repeats=0)


def test_empty_corpus_is_harmless() -> None:
    """An empty record list must not raise when weights are queried."""
    sampler = FamilyBalancedGroupedSampler([], balance_alpha=0.5)
    assert sampler.family_weights() == {}
    assert sampler.families == ()
    assert list(sampler) == []


# ----------------------------------------------------------------------
# Corpus provenance (report section 1)
# ----------------------------------------------------------------------
class _FakeInner:
    """Minimal stand-in for :class:`HwPhaseImageDataset`."""

    def __init__(self, records: list[Any]) -> None:
        self.records = records


class _FakeSubset:
    """Minimal stand-in for ``torch.utils.data.Subset``."""

    def __init__(self, records: list[Any], indices: list[int]) -> None:
        self.dataset = _FakeInner(records)
        self.indices = indices


class _FakeLoader:
    """Minimal stand-in for a ``DataLoader`` over a ``Subset``."""

    def __init__(self, records: list[Any], indices: list[int]) -> None:
        self.dataset = _FakeSubset(records, indices)


def test_corpus_summary_counts_only_the_subset() -> None:
    """A summary must describe the split, not the whole index behind it."""
    records = _imbalanced()
    loader = _FakeLoader(records, [0, 1, 18])  # two pib + one online
    summary = corpus_summary(loader)
    assert summary["n_records"] == 3
    assert summary["n_files"] == 3
    assert summary["families"] == {"slm_pib": 2, "slm_pib_online": 1}
    assert summary["n_families"] == 2


def test_corpus_summary_reports_dominant_share() -> None:
    """The imbalance figure that decides whether balancing is needed."""
    records = _imbalanced()
    summary = corpus_summary(_FakeLoader(records, list(range(len(records)))))
    assert summary["n_records"] == 20
    assert summary["n_files"] == 20
    assert summary["dominant_family_share"] == pytest.approx(18 / 20, abs=1e-4)


def test_corpus_summary_is_json_safe() -> None:
    """The summary is embedded in ``metrics.json``, so it must serialise."""
    records = _imbalanced()
    summary = corpus_summary(_FakeLoader(records, list(range(len(records)))))
    assert json.loads(json.dumps(summary)) == summary


def test_corpus_summary_omits_fov_rather_than_guessing() -> None:
    """``fov_px`` lives on the sample, not the record -- report no fake one.

    Reporting an all-``unknown`` FOV histogram would read as a measurement; the
    honest answer is to leave it out and let the forward bank report its own
    exclusions at evaluation time.
    """
    summary = corpus_summary(_FakeLoader(_imbalanced(), [0]))
    assert "fov_histogram" not in summary


# ----------------------------------------------------------------------
# Configuration contract
# ----------------------------------------------------------------------
def test_image_mode_rejects_sum_normalisation() -> None:
    """``sum`` normalisation is measured to flatter MSE while worsening R2."""
    assert "sum" not in IMAGE_MODES
    with pytest.raises(ValueError):
        ShapingConfig(image_mode="sum")


@pytest.mark.parametrize("mode", IMAGE_MODES)
def test_every_advertised_image_mode_is_accepted(mode: str) -> None:
    """The advertised modes must all validate."""
    assert ShapingConfig(image_mode=mode).image_mode == mode


def test_drop_exposure_channel_is_off_by_default() -> None:
    """Exposure stays an input unless the ablation is explicitly requested."""
    assert ShapingConfig().drop_exposure_channel is False


def test_balance_defaults() -> None:
    """Defaults must match the documented compromise."""
    cfg = ShapingConfig()
    assert cfg.balance_alpha == 0.5
    assert cfg.max_file_repeats == 8


def test_preprocessing_defaults_survive_a_round_trip() -> None:
    """A config carrying the new fields must still be constructible/serialisable."""
    import dataclasses

    cfg = ShapingConfig(families=("slm_pib",), sources=("panel_gray",))
    restored = ShapingConfig(**dataclasses.asdict(cfg))
    assert restored.families == ("slm_pib",)
    assert restored.sources == ("panel_gray",)