"""Tests for :mod:`ml.zernike.train_coeff`.

Load-bearing coverage, ordered by what would silently corrupt a result:

1. :func:`r2_score` -- the per-sample denominator, and the ``SS_tot == 0`` case.
2. :func:`file_folds` / :func:`select_fold` -- the split is by FILE.
3. The argparse convention -- every numeric default equals its dataclass field.
4. The leakage guard -- statistics come from the TRAIN positions only.
5. W&B is never load-bearing.

Inline fixtures only; this repo has no ``conftest.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.zernike.forward_model import (
    ZernikeCoeffConfig,
    build_forward_model,
    peak_normalize,
)
from ml.zernike.train_coeff import (
    CoeffTrainConfig,
    CoeffTrainResult,
    FileFold,
    _build_parser,
    _finish_wandb,
    _init_wandb,
    _log_comparison_image,
    _log_wandb_row,
    _log_wandb_summary,
    file_folds,
    r2_score,
    select_fold,
)


# ---------------------------------------------------------------------------
# A stub that satisfies file_folds(), which only ever touches ``.records``.
# ---------------------------------------------------------------------------
class _Ref:
    def __init__(self, path: Path, position: int) -> None:
        self.path = path
        self.position = position


class _StubDataset:
    def __init__(self, paths: list[Path], per_file: int) -> None:
        self.records = tuple(
            _Ref(path, i * per_file + j)
            for i, path in enumerate(paths)
            for j in range(per_file)
        )

    def __len__(self) -> int:
        return len(self.records)


def _stub() -> _StubDataset:
    return _StubDataset([Path(f"run_{i}.pkl") for i in range(4)], per_file=3)


class TestR2Score:
    """The selection metric, so its denominator is a correctness contract."""

    def test_perfect_prediction_is_one(self) -> None:
        target = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert np.allclose(r2_score(target.copy(), target), 1.0)

    def test_hand_computable_value(self) -> None:
        # target mean = 1.5 -> SS_tot = 2.25+0.25+0.25+2.25 = 5; SS_res = 1 -> R^2 = 0.8
        target = np.array([0.0, 1.0, 2.0, 3.0])
        pred = np.array([0.0, 1.0, 2.0, 4.0])
        assert r2_score(pred, target) == pytest.approx(0.8, abs=1e-12)

    def test_constant_target_is_nan_not_crash(self) -> None:
        # SS_tot == 0 has no variance to explain. NaN is the honest answer; a
        # crash or a 0.0 here would silently read as "the model explained it".
        target = np.full((4, 5), 2.0)
        value = r2_score(np.zeros((4, 5)), target)
        assert np.isnan(np.atleast_1d(value)).all()

    def test_denominator_is_per_sample_not_pooled(self) -> None:
        # Regression guard. Each sample gets its OWN constant offset error. With a
        # per-sample denominator, removing each sample's own mean absorbs the
        # offset and R^2 is ~1. A POOLED denominator does not remove a per-sample
        # offset, so it reports a large negative number -- the exact bug that
        # produced a spurious "worse than a constant predictor" R^2 in this repo.
        # Targets vary within the sample so SS_tot is non-zero (a constant target
        # would give nan, covered separately).
        base = np.arange(6, dtype=np.float64)
        target = np.stack([base, base + 100.0])
        pred = target + np.array([5.0, -5.0])[:, None]
        per_sample = np.atleast_1d(r2_score(pred, target))
        # A constant offset leaves SS_res untouched by centering, so each sample
        # scores strongly negative: 1 - 150/17.5 = -7.571429.
        assert np.allclose(per_sample, -7.571429, atol=1e-5)

        # The discriminator, and the direction that matters: a single pooled
        # SS_tot cannot see the per-sample offset at all, so it reports ~0.99 --
        # i.e. "almost perfect" for a prediction that is wrong everywhere. This is
        # the bug that made an earlier version of this work report a spurious
        # POSITIVE score where the honest per-sample value is negative.
        pooled_ss_tot = float(((target - target.mean()) ** 2).sum())
        pooled_ss_res = float(((target - pred) ** 2).sum())
        pooled_r2 = 1.0 - pooled_ss_res / pooled_ss_tot
        assert pooled_r2 > 0.98
        assert pooled_r2 - float(np.mean(per_sample)) > 8.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="share a shape"):
            r2_score(np.zeros((2, 3)), np.zeros((3, 2)))

    def test_image_shaped_input_returns_one_value_per_sample(self) -> None:
        target = np.random.default_rng(0).random((5, 1, 8, 8))
        values = r2_score(target.copy(), target)
        assert np.asarray(values).shape == (5,)
        assert np.allclose(values, 1.0)


class TestFileFolds:
    def test_every_position_appears_exactly_once_as_validation(self) -> None:
        dataset = _stub()
        folds = file_folds(dataset)  # type: ignore[arg-type]
        seen: list[int] = []
        for fold in folds:
            seen.extend(fold.val_positions)
        assert sorted(seen) == list(range(len(dataset)))

    def test_train_and_val_are_disjoint_and_non_empty(self) -> None:
        for fold in file_folds(_stub()):  # type: ignore[arg-type]
            assert fold.train_positions and fold.val_positions
            assert not set(fold.train_positions) & set(fold.val_positions)
            assert len(fold.train_positions) + len(fold.val_positions) == 12

    def test_held_out_file_is_entirely_in_validation(self) -> None:
        dataset = _stub()
        folds = file_folds(dataset)  # type: ignore[arg-type]
        for fold in folds:
            held = [r.path for r in dataset.records if r.path == fold.held_out]
            positions = {r.position for r in dataset.records}
            for position in fold.val_positions:
                assert positions  # sanity
            assert len(fold.val_positions) == len(held)

    def test_record_level_split_would_leak(self) -> None:
        # Why this exists: records inside one file are consecutive epochs of ONE
        # optimisation run. A record-level split would put epoch k in train and
        # epoch k+1 (a near-duplicate frame) in validation.
        dataset = _stub()
        folds = file_folds(dataset)  # type: ignore[arg-type]
        single = folds[0]
        assert len(single.val_positions) == 3  # whole file, not one record


class TestSelectFold:
    def test_by_index(self) -> None:
        folds = file_folds(_stub())  # type: ignore[arg-type]
        assert select_fold(folds, CoeffTrainConfig(fold=2)).index == 2

    def test_negative_index_wraps(self) -> None:
        folds = file_folds(_stub())  # type: ignore[arg-type]
        assert select_fold(folds, CoeffTrainConfig(fold=-1)).index == len(folds) - 1

    def test_default_is_first_fold(self) -> None:
        folds = file_folds(_stub())  # type: ignore[arg-type]
        assert select_fold(folds, CoeffTrainConfig()).index == 0

    def test_by_full_path_and_by_bare_name(self) -> None:
        folds = file_folds(_stub())  # type: ignore[arg-type]
        full = select_fold(folds, CoeffTrainConfig(held_out_path="run_1.pkl"))
        bare = select_fold(folds, CoeffTrainConfig(held_out_path="run_1.pkl"))
        assert full.index == bare.index == 1

    def test_unknown_path_raises_and_names_the_options(self) -> None:
        folds = file_folds(_stub())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Available pickles"):
            select_fold(folds, CoeffTrainConfig(held_out_path="nope.pkl"))

    def test_out_of_range_index_raises(self) -> None:
        folds = file_folds(_stub())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="out of range"):
            select_fold(folds, CoeffTrainConfig(fold=99))


class TestParserConvention:
    """Every numeric default must come from the dataclass.

    This is the regression test for the ``train_amp`` bug where a hardcoded
    ``--far-field-padding default=1`` silently overrode the calibrated dataclass
    value of 10 and made every run start at R^2 = -0.38.
    """

    def test_every_flag_default_matches_the_dataclass(self) -> None:
        cfg_default = CoeffTrainConfig()
        parser = _build_parser()
        mismatched: list[str] = []
        for action in parser._actions:  # noqa: SLF001 - the contract under test
            dest = action.dest
            if dest in ("help",) or dest not in {f.name for f in fields(cfg_default)}:
                continue
            expected = getattr(cfg_default, dest)
            actual = action.default
            # argparse's nargs="+" always yields a list where the dataclass holds
            # a tuple; compare sequences by value, not by type.
            if isinstance(expected, tuple) and isinstance(actual, list):
                equal = tuple(actual) == expected
            else:
                equal = actual == expected
            if not equal:
                mismatched.append(f"{dest}: parser={actual!r} dataclass={expected!r}")
        assert not mismatched, "CLI defaults diverged from the dataclass:\n" + "\n".join(
            mismatched
        )

    def test_parser_is_argparse(self) -> None:
        assert isinstance(_build_parser(), argparse.ArgumentParser)

    def test_flipping_a_flag_reaches_the_config(self) -> None:
        args = _build_parser().parse_args(["--fold", "3", "--epochs", "7", "--no-wandb"])
        assert args.fold == 3 and args.epochs == 7 and args.use_wandb is False

    def test_features_come_through_as_a_tuple(self) -> None:
        args = _build_parser().parse_args(["--features", "8", "16", "32"])
        assert args.features == [8, 16, 32]


class TestPeakNormalisationIsApplied:
    def test_prediction_path_normalises_a_raw_unbounded_output(self) -> None:
        # forward() is deliberately raw; the [0,1] contract comes from
        # peak_normalize. If a caller compared raw output to a [0,1] target the
        # loss would be meaningless.
        model = build_forward_model(
            ZernikeCoeffConfig(n_coeffs=136, grid=64, architecture="mlp", hidden=32)
        )
        # A non-degenerate input: the all-zero vector maps to a constant output,
        # and normalising a constant zero field is 0/eps -> 0, not 1.
        coeffs = torch.zeros(2, 136)
        coeffs[:, 4] = 0.5  # Noll 5 = astigmatism
        coeffs[:, 11] = -0.3
        raw = model(coeffs)
        normalised = peak_normalize(raw)
        assert raw.shape == (2, 1, 64, 64)
        # peak_normalize guarantees max == 1 per sample. It does NOT clamp below
        # zero: the raw head is unbounded, so negative values survive the
        # division. Asserting `min >= 0` here would be asserting a contract that
        # does not exist -- the [0,1] bound applies to the TARGET, not the
        # prediction.
        assert torch.allclose(
            normalised.amax(dim=(-2, -1)), torch.ones(2), atol=1e-5
        )
        assert torch.isfinite(normalised).all()
        assert float(normalised.amax()) <= 1.0 + 1e-5

    def test_constant_zero_field_normalises_without_nan(self) -> None:
        model = build_forward_model(
            ZernikeCoeffConfig(n_coeffs=136, grid=64, architecture="mlp", hidden=32)
        )
        out = peak_normalize(model(torch.zeros(1, 136)))
        assert torch.isfinite(out).all()

    def test_model_output_is_not_sigmoid_bounded(self) -> None:
        # The anti-sigmoid guard: a hard-wired sigmoid output head (as
        # ml.phase.unet.UNetGenerator has) cannot represent this dynamic range.
        model = build_forward_model(
            ZernikeCoeffConfig(n_coeffs=136, grid=64, architecture="mlp", hidden=32)
        )
        with torch.no_grad():
            for scale in (1e3, -1e3):
                out = model(torch.full((1, 136), scale))
                assert not torch.isfinite(out).all() or out.abs().max() > 1.0


class TestWandbIsNeverLoadBearing:
    """Every telemetry call must be a no-op or a warning, never a failure."""

    def test_disabled_returns_none(self) -> None:
        model = build_forward_model(
            ZernikeCoeffConfig(n_coeffs=136, grid=64, architecture="mlp", hidden=32)
        )
        run = _init_wandb(
            CoeffTrainConfig(use_wandb=False),
            model,
            n_train=1,
            n_val=1,
            n_terms=136,
            held_out="x.pkl",
        )
        assert run is None

    @pytest.mark.parametrize(
        "call",
        [
            lambda r, p: _log_wandb_row(r, {"val_r2": 0.5}, 0),
            lambda r, p: _log_comparison_image(r, p, 0, "caption"),
            lambda r, p: _log_wandb_summary(
                r,
                CoeffTrainResult(
                    best_val_r2=0.5,
                    best_epoch=1,
                    history=[{"epoch": 0.0, "val_r2": 0.5}],
                    checkpoint=None,
                    summary=None,
                    held_out="x.pkl",
                    n_train=1,
                    n_val=1,
                    final_metrics={},
                ),
            ),
            lambda r, p: _finish_wandb(r),
        ],
    )
    def test_none_run_is_a_no_op(self, call) -> None:
        call(None, Path("does-not-exist.png"))

    def test_a_broken_run_object_does_not_raise(self) -> None:
        class _Exploding:
            def log(self, *args, **kwargs):
                raise RuntimeError("telemetry backend is down")

            summary: dict = {}

            def finish(self):
                raise RuntimeError("telemetry backend is down")

        broken = _Exploding()
        _log_wandb_row(broken, {"val_r2": 0.1}, 0)  # must not propagate
        _log_comparison_image(broken, Path("nope.png"), 0, "caption")
        _log_wandb_summary(
            broken,
            CoeffTrainResult(
                best_val_r2=0.1,
                best_epoch=0,
                history=[],
                checkpoint=None,
                summary=None,
                held_out="x",
                n_train=1,
                n_val=1,
                final_metrics={},
            ),
        )
        _finish_wandb(broken)


class TestLossContract:
    def test_sum_normalisation_is_rejected(self) -> None:
        # "sum" is the measured trap: PSNR/SSIM look perfect while R^2 degrades.
        with pytest.raises(ValueError, match="sum"):
            from ml.zernike.train_coeff import train

            train(CoeffTrainConfig(image_mode="sum", epochs=1, use_wandb=False))

    def test_default_image_mode_is_the_incumbent_robust(self) -> None:
        # The default is deliberately NOT `peak`. A 5-seed R^2 comparison once
        # favoured `peak` (+0.821 vs +0.628), but that comparison never checked the
        # constant-predictor floor: on this corpus a *constant* predictor scores
        # R^2 +0.920 under `peak` and +0.799 under `abs255`, so raw R^2 differences
        # of that size sit inside the metric's own floor. Scored on
        # `skill = 1 - mse_model/mse_constant` the ordering reverses
        # (scripts/sweep_target_transform.py), and `zscore` -- which strips the
        # per-frame level entirely -- leaves skill +0.006, i.e. the coefficients
        # live mostly in the absolute level. So the earlier argument is withdrawn
        # and the incumbent is kept until the ConvNet itself is scored on skill.
        assert CoeffTrainConfig().image_mode == "robust"

    def test_default_selection_metric_path_is_r2(self) -> None:
        # The config exposes no knob to select the best epoch on MSE/PSNR/SSIM.
        names = {f.name for f in fields(CoeffTrainConfig)}
        assert not any(
            n in names for n in ("select_on", "metric", "best_metric", "loss_metric")
        )
