"""F-14: a lint exclusion must never be able to hide a guaranteed ``NameError``.

``ruff.toml`` carried ``exclude = ["src/ml/**"]`` with the comment "legacy code
with syntax errors". Under that exclusion ruff never looked at ``src/ml/``, so
30 ``F821`` undefined names sat invisible for the whole life of the package --
including ``DataLoader`` in :func:`ml.zernike.dataset.create_zernike_loaders`
(used three times, never imported) and ``build_gan_models`` in ``ml.train``
(referenced three times, defined in no commit anywhere in the repo's history).
Both are ``ao-train`` console-script entry points, so the documented training
entry point could only ever die with ``NameError``.

The checks below pin the three things that actually went wrong:

1. the config must not exclude anything under ``src/`` (the R-63 step-4
   precondition), and if ruff is installed, ruff must agree there is no ``F821``
   left anywhere;
2. the functions that were broken are exercised, not merely imported -- a
   ``NameError`` in a function body is invisible to ``import ml.train``;
3. the deliberately removed Zernike regressors raise the *directed* error that
   commit ``40d65af`` wrote but never wired, rather than a bare ``NameError``.
"""

from __future__ import annotations

import csv
import importlib.util
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _ruff_config() -> dict:
    return tomllib.loads((_REPO_ROOT / "ruff.toml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. the exclusion itself
# ---------------------------------------------------------------------------


def test_ruff_excludes_nothing_under_src() -> None:
    """`src/ml/**` is gone; nothing else may take its place.

    An exclusion is not a style opinion: it decides whether ruff can see a
    guaranteed crash. This is the precondition R-63 step 4 needs.
    """
    excludes = _ruff_config().get("lint", {}).get("exclude", [])
    hiding_src = [
        pattern
        for pattern in excludes
        if "src/" in pattern or pattern.startswith("src")
    ]
    assert not hiding_src, (
        f"ruff.toml excludes source trees: {hiding_src}. "
        "Excluding src/ hides F821 (the F-14 class of bug: guaranteed NameError). "
        "Fix the code instead of excluding it."
    )


def test_no_f821_left_in_src() -> None:
    """The 30 undefined names are gone, measured by ruff itself when available."""
    spec = importlib.util.find_spec("ruff")
    if spec is None:
        pytest.skip("ruff not installed; run `ruff check --isolated --select F821 src/`")
    from ruff.__main__ import find_ruff_bin  # type: ignore[attr-defined]

    import subprocess

    proc = subprocess.run(
        [find_ruff_bin(), "check", "--isolated", "--no-cache", "--select", "F821", "src"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"F821 undefined names remain:\n{proc.stdout}"


# ---------------------------------------------------------------------------
# 2. the broken functions are exercised
# ---------------------------------------------------------------------------


def _write_dataset(root: Path, n_samples: int = 6, n_terms: int = 8) -> Path:
    """Minimal on-disk corpus for ``ZernikeCoefficientDataset``.

    Layout is fixed by ``_is_valid_sample``: ``sample_*/{daheng_frame.npy,
    miicam_frame.npy, phase.csv}``, with a header row then the coefficient row.
    """
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        d = root / f"sample_{i:03d}"
        d.mkdir(parents=True)
        np.save(d / "daheng_frame.npy", (rng.random((32, 32)) * 255).astype(np.uint8))
        np.save(d / "miicam_frame.npy", (rng.random((32, 32)) * 255).astype(np.uint8))
        with (d / "phase.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"c{i}" for i in range(n_terms)])
            writer.writerow([0.1] * n_terms)
    return root


def test_create_zernike_loaders_returns_real_dataloaders(tmp_path: Path) -> None:
    """`DataLoader` was used at three sites and imported at none."""
    from ml.zernike.dataset import create_zernike_loaders

    data_dir = _write_dataset(tmp_path / "corpus")
    train, val, test = create_zernike_loaders(
        data_dir=data_dir,
        batch_size=2,
        n_zernike_terms=8,
        target_size=(32, 32),
        augment=False,
    )
    for name, loader in (("train", train), ("val", val), ("test", test)):
        assert isinstance(loader, torch.utils.data.DataLoader), f"{name} is {type(loader)}"
    assert len(train.dataset) + len(val.dataset) + len(test.dataset) == 6
    batch = next(iter(train))
    assert batch["image"].shape[0] == 2
    assert batch["coefficients"].shape[1] == 8


def test_build_gan_models_pairs_generator_with_discriminator() -> None:
    """`build_gan_models` was referenced by three call sites in every commit."""
    from ml.phase.discriminator import PatchGANDiscriminator
    from ml.phase.unet import UNetGenerator
    from ml.train import build_gan_models

    for in_channels in (1, 2):
        generator, discriminator = build_gan_models(in_channels=in_channels, device="cpu")
        assert isinstance(generator, UNetGenerator)
        assert isinstance(discriminator, PatchGANDiscriminator)
        # The discriminator sees the predicted phase concatenated onto the input,
        # which is the pairing train_model already used inline.
        first_conv = discriminator.layers[0][0]
        assert first_conv.in_channels == in_channels + 1


def test_ao_train_cli_still_registers_every_command() -> None:
    """`ao-train = "ml.train:cli"` is a real console script, so keep it importable."""
    from ml.train import cli

    assert sorted(cli.commands) == ["predict", "sweep", "train"]


# ---------------------------------------------------------------------------
# 3. the removed path raises the directed error 40d65af intended
# ---------------------------------------------------------------------------


def test_removed_coeffs_training_raises_the_directed_error(tmp_path: Path) -> None:
    """`build_model` was deleted; the wiring for its replacement was forgotten."""
    from ml.train import _sweep_coeffs_agent, _train_coeffs_mode

    with pytest.raises(RuntimeError, match="Zernike coefficient regression"):
        _train_coeffs_mode(
            data_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=1e-4,
            train_split=0.7,
            val_split=0.1,
            num_workers=0,
            input_mode="combined",
            n_zernike_terms=8,
            target_size=(32, 32),
            device=torch.device("cpu"),
            seed=42,
            use_wandb=False,
            wandb_project="p",
            wandb_name=None,
            resume=None,
        )

    with pytest.raises(RuntimeError, match="resnet34"):
        _sweep_coeffs_agent(
            data_dir=str(tmp_path),
            batch_size=2,
            epochs=1,
            input_mode="combined",
            n_zernike_terms=8,
            model_type="resnet34",
            lr=1e-3,
            target_size=(32, 32),
            device=torch.device("cpu"),
            use_wandb=False,
        )


def test_sweep_config_offers_only_model_types_that_still_exist() -> None:
    """A sweep entry that can only raise is a configuration bug, not a sweep."""
    from ml.train import ALL_MODEL_TYPES, SWEEP_CONFIG

    offered = SWEEP_CONFIG["parameters"]["model_type"]["values"]
    removed = {"resnet18", "resnet34", "simple_cnn"}
    assert not (set(offered) & removed), f"sweep still offers removed models: {offered}"
    assert set(offered) <= ALL_MODEL_TYPES


def test_model_type_default_is_a_model_that_exists() -> None:
    """`--model-type` defaulted to resnet18, which can only raise."""
    import ml.train as train_module

    param = next(
        p for p in train_module.train.params if p.name == "model_type"
    )
    assert param.default in {"unet"}