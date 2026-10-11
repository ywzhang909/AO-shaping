"""Tests for :mod:`ml.shaping.train` -- the training CLI.

No hardware and no real corpus: :func:`ml.shaping.dataset.build_shaping_loaders`
is monkeypatched to return loaders over a tiny in-memory dataset. The real
:class:`~ml.shaping.physics.ForwardBank` is used (it is pure torch and fast),
so the physics leg is genuinely exercised rather than stubbed away.

``torch`` lives in the optional ``ml`` dependency group, hence the
``importorskip``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

import torch.utils.data as _tud  # noqa: E402

from ml.shaping.config import ShapingConfig, UNKNOWN_FOV  # noqa: E402
from ml.shaping.train import (  # noqa: E402
    _build_parser,
    _config_from_args,
    main,
    seed_everything,
    train,
)

_GRID = 16


# ----------------------------------------------------------------------
# Fixtures: a tiny synthetic corpus, no pickle involved
# ----------------------------------------------------------------------
def _sample(index: int) -> dict[str, Any]:
    """Build one in-memory sample shaped like a corpus record.

    Two of the four samples carry an ``fov_px`` with no calibrated forward
    model, so the physics-exclusion path is live in every test that uses this.

    Args:
        index: Position within the synthetic file.

    Returns:
        A sample dict matching ``HwPhaseImageDataset.__getitem__``.
    """
    generator = torch.Generator().manual_seed(1000 + index)
    fov = (64, 320, 1944, UNKNOWN_FOV)[index % 4]
    exposure = (0.5, 1.2, 2.0, 0.1)[index % 4]
    return {
        "phase_cos": torch.rand((1, _GRID, _GRID), generator=generator),
        "phase_sin": torch.rand((1, _GRID, _GRID), generator=generator),
        "image": torch.rand((1, _GRID, _GRID), generator=generator),
        "contrast": torch.ones((_GRID, _GRID)),
        "exposure_ms": exposure,
        "exposure_log10": torch.tensor([0.0]),
        "source": "panel_rad",
        "family": "fam_a" if index % 2 == 0 else "fam_b",
        "sample_idx": index,
        "path": f"synthetic/file_{index % 2}.pkl",
        "fov_px": fov,
    }


class _SyntheticDataset(_tud.Dataset):
    """A flat list of synthetic samples, no file access."""

    def __init__(self, count: int = 8) -> None:
        self._items = [_sample(i) for i in range(count)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._items[index]


@pytest.fixture
def fake_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the corpus loaders with in-memory ones.

    Args:
        monkeypatch: pytest's patcher.
    """

    def _fake(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        dataset = _SyntheticDataset()
        return (
            _tud.DataLoader(dataset, batch_size=4, shuffle=False),
            _tud.DataLoader(dataset, batch_size=4, shuffle=False),
            _tud.DataLoader(dataset, batch_size=4, shuffle=False),
        )

    monkeypatch.setattr("ml.shaping.train.build_shaping_loaders", _fake)


def _tiny_config(tmp_path: Path, **overrides: Any) -> ShapingConfig:
    """Build a fast config for a smoke run.

    Args:
        tmp_path: Checkpoint directory.
        **overrides: Fields to override.

    Returns:
        A validated :class:`ShapingConfig`.
    """
    base: dict[str, Any] = {
        "grid": _GRID,
        "features": (4, 8),
        "epochs": 1,
        "max_steps": 2,
        "batch_size": 4,
        "num_workers": 0,
        "device": "cpu",
        "wandb": False,
        "checkpoint_dir": str(tmp_path),
    }
    base.update(overrides)
    return ShapingConfig(**base)


# ----------------------------------------------------------------------
# Parser / dataclass parity -- the literal-default regression test
# ----------------------------------------------------------------------
def test_parser_defaults_equal_the_dataclass() -> None:
    """Every CLI default must come from ``ShapingConfig``.

    This repository shipped the opposite bug: the ``train_amp`` parser wrote
    ``default=1`` literally for ``far_field_padding`` and silently overrode the
    calibrated value of 10 on every run.
    """
    defaults = vars(_build_parser().parse_args([]))
    expected = dataclasses.asdict(ShapingConfig())
    mismatches = {
        key: (defaults.get(key, "<missing>"), value)
        for key, value in expected.items()
        if key in defaults and defaults[key] != value
    }
    assert not mismatches, f"parser defaults drifted from the dataclass: {mismatches}"


def test_every_dataclass_field_has_a_flag() -> None:
    """No dataclass field may be silently unreachable from the CLI."""
    defaults = vars(_build_parser().parse_args([]))
    missing = sorted(set(dataclasses.asdict(ShapingConfig())) - set(defaults))
    assert not missing, f"fields with no CLI flag: {missing}"


def test_config_from_args_round_trips() -> None:
    """``_config_from_args`` must reproduce the parsed values."""
    cfg = _config_from_args(_build_parser().parse_args([]))
    assert cfg == ShapingConfig()


def test_help_exits_zero() -> None:
    """``--help`` is a successful exit, not an error."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_invalid_config_returns_usage_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid CLI combination exits 2 rather than raising."""
    assert main(["--grid", "4"]) == 2


def test_negative_weight_is_rejected_by_the_dataclass() -> None:
    """``ShapingConfig`` validation is reachable from the CLI surface."""
    with pytest.raises(ValueError):
        ShapingConfig(w_sup=-1.0)
    with pytest.raises(ValueError):
        ShapingConfig(grid=4)


# ----------------------------------------------------------------------
# Logging format-style regression
# ----------------------------------------------------------------------
def test_no_percent_style_logger_calls(tmp_path: Path) -> None:
    """loguru formats with ``str.format``, **not** ``%``.

    A ``%d``/``%s``-style message is emitted verbatim with the placeholders
    still in it -- no exception, just silently useless output. This test walks
    every ``ml.shaping`` module and fails if any logger call still uses
    printf-style placeholders.
    """
    import re

    package_dir = Path(__import__("ml.shaping", fromlist=["__file__"]).__file__).parent
    pattern = re.compile(r"logger\.(?:debug|info|warning|error|exception)\(")
    offenders: list[str] = []
    for source in sorted(package_dir.glob("*.py")):
        text = source.read_text(encoding="utf-8")
        lines = text.splitlines()
        for number, line in enumerate(lines, start=1):
            if not pattern.search(line):
                continue
            window = "\n".join(lines[number - 1 : number + 3])
            if re.search(r'"[^"\n]*%[0-9.\-]*[dsfge]', window):
                offenders.append(f"{source.name}:{number}")
    assert not offenders, f"printf-style loguru calls (use {{}} instead): {offenders}"


def test_loguru_substitutes_brace_style() -> None:
    """The style the package uses really does interpolate."""
    from loguru import logger

    sink: list[str] = []
    handler_id = logger.add(lambda message: sink.append(message.record["message"]), level="DEBUG")
    try:
        logger.info("grid={} n={}", 64, 3)
    finally:
        logger.remove(handler_id)
    assert sink and sink[-1] == "grid=64 n=3"


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def test_smoke_run_writes_a_checkpoint(
    fake_loaders: None, tmp_path: Path
) -> None:
    """A capped run completes, stays finite and writes its artefacts."""
    result = train(_tiny_config(tmp_path))
    assert result.epochs_run == 1
    assert result.global_step == 2
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "history.json").is_file()
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert all(
        isinstance(row["train_total"], float) for row in history
    )
    assert result.final["val_sup"] == result.final["val_sup"]  # not NaN


def test_max_steps_is_honoured(fake_loaders: None, tmp_path: Path) -> None:
    """``max_steps`` caps optimiser steps exactly."""
    result = train(_tiny_config(tmp_path, max_steps=1))
    assert result.global_step == 1


def test_resume_continues(fake_loaders: None, tmp_path: Path) -> None:
    """``--resume`` picks up model/optimiser state and keeps going."""
    first = train(_tiny_config(tmp_path, epochs=1))
    assert first.epochs_run == 1
    second = train(
        _tiny_config(tmp_path, epochs=2, resume=first.checkpoint)
    )
    # History is appended, not truncated.
    assert second.epochs_run == 2
    rows = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert len(rows) == 2


def test_resume_missing_file_raises(tmp_path: Path) -> None:
    """A bad ``--resume`` path is an error, not a silent fresh start."""
    with pytest.raises(FileNotFoundError):
        train(_tiny_config(tmp_path, resume=str(tmp_path / "nope.pt")))


def test_determinism_flags_are_set(
    fake_loaders: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cudnn_deterministic`` reaches the torch global flags."""
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    train(_tiny_config(tmp_path, device="cpu", cudnn_deterministic=True))
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_device_resolution() -> None:
    """``auto`` resolves to a real device; explicit specs are honoured."""
    from ml.shaping.train import _resolve_device

    assert _resolve_device("cpu").type == "cpu"
    assert _resolve_device("auto").type in {"cpu", "cuda"}


def test_seed_everything_is_reproducible() -> None:
    """Two seeded draws agree."""
    seed_everything(7)
    first = torch.rand(4)
    seed_everything(7)
    assert torch.equal(first, torch.rand(4))


def test_module_has_no_print_or_relative_import() -> None:
    """Repo style guard."""
    import re

    source = Path(
        __import__("ml.shaping.train", fromlist=["__file__"]).__file__
    ).read_text(encoding="utf-8")
    assert not re.search(r"^\s*print\(", source, re.MULTILINE)
    assert not re.search(r"^\s*from\s+\.", source, re.MULTILINE)


def test_parser_features_and_roots_parsing() -> None:
    """The comma-separated flags round-trip back to tuples."""
    args = _build_parser().parse_args(
        ["--features", "4,8,16", "--roots", "a,b"]
    )
    assert tuple(args.features) == (4, 8, 16)
    assert tuple(args.roots) == ("a", "b")
    # argparse turns an ArgumentTypeError raised by a `type=` callable into a
    # usage error, i.e. SystemExit(2) -- not a propagated ArgumentTypeError.
    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(["--features", ""])
    assert excinfo.value.code == 2