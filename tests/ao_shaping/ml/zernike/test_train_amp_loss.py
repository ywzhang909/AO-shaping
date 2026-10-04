"""Pin the ``AmpTrainConfig.loss`` switch and, above all, its default.

``train_amp`` optimises a plain pixel MSE on the normalised far field. That
objective cannot see where the light lands, so it will trade in-ROI energy for
uniformity -- the blind spot already documented for optimising ``-CV`` alone,
where the target box empties out. :mod:`ml.zernike.losses` adds the
differentiable ROI terms so the objective can be steered by the physical spot
quality.

The property that actually needs locking down is **non-regression**: with the
default ``loss="mse"`` the coefficients must come out bit-identical to the
pre-switch implementation. Adding an objective option that silently perturbs
the incumbent run would invalidate every archived training result, so the first
test here compares against a hard-coded coefficient vector captured from the
MSE path, not merely "the run did not crash".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.zernike.losses import LossConfig
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel
from ml.zernike.train_amp import train


def _synthetic_corpus(tmp_path, *, n_train: int = 12, n_val: int = 4, grid: int = 16):
    """Write the minimum artefact ``train`` needs: a pickle of phase/image pairs.

    Built through the real ``Materialiser`` path so this test exercises the same
    contract as the 402-pickle corpus rather than a mock of it.
    """
    import pickle

    from ml.hwdataset.index import build_hw_index

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:grid, 0:grid]
    cy = cx = (grid - 1) / 2.0

    def _one():
        phase = rng.normal(0.0, 0.3, size=(grid, grid))
        image = np.zeros((grid, grid), dtype=np.float32)
        rr = ((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5
        image[rr <= grid / 4.0] = 1.0
        image = np.roll(image, int(rng.integers(-1, 2)), axis=0)
        return {
            "phase": phase.astype(np.float32),
            "image": image * np.float32(0.5 + 0.5 * rng.random()),
            "exposure_ms": np.float32(1.0),
        }

    root = tmp_path / "debug"
    root.mkdir()
    with (root / "synthetic_family_20260101_000000.pkl").open("wb") as fh:
        pickle.dump([_one() for _ in range(n_train + n_val)], fh)

    index = build_hw_index(roots=[root], index_cache=None)
    return index


def _base_cfg(index, tmp_path, **overrides):
    from ml.zernike.train_amp import AmpTrainConfig

    cfg = dict(
        families=index.families,
        grid=16,
        n_max=3,
        epochs=2,
        batch_size=4,
        lr=0.01,
        max_train=12,
        max_val=4,
        val_fraction=0.25,
        device="cpu",
        out_dir=str(tmp_path / "out"),
        use_wandb=False,
        save_checkpoint=False,
        seed=0,
        num_workers=0,
    )
    cfg.update(overrides)
    return AmpTrainConfig(**cfg)


def test_default_config_is_the_incumbent_mse_objective():
    from ml.zernike.train_amp import LOSS_CHOICES, AmpTrainConfig

    assert AmpTrainConfig().loss == "mse"
    assert LOSS_CHOICES == ("mse", "physical")


def test_physical_switch_is_not_a_no_op():
    """``loss="physical"`` must actually change the objective.

    Regression: ``LossConfig``'s own default is the incumbent ``w_mse=1.0``, so
    wiring the switch to a bare ``LossConfig()`` would have made
    ``loss="physical"`` bit-identical to ``loss="mse"`` -- a flag that silently
    did nothing. ``AmpTrainConfig.loss_weights`` therefore defaults to a blend
    that includes a non-zero physical term.
    """
    from ml.zernike.train_amp import AmpTrainConfig

    weights = AmpTrainConfig().loss_weights
    assert weights.w_mse == 1.0
    # A physical term must be active ...
    assert weights.w_shape_gap == 1.0
    # ... and it must be the ANCHORED one. The unanchored pair has a degenerate
    # optimum on a fitting task (measured: val R2 +0.78 -> -0.86, with shape_sum
    # 34% above the physics being predicted), so it must not be the default here.
    assert weights.w_pib == 0.0
    assert weights.w_uniformity == 0.0
    # ... which is NOT the neutral LossConfig default.
    assert LossConfig().w_mse == 1.0
    assert LossConfig().w_shape_gap == 0.0


def test_mse_path_is_bit_identical_to_the_hard_captured_reference(tmp_path):
    """Non-regression anchor for the incumbent objective.

    Any change to the default optimisation path -- an added term, a reordered
    statement, a different reduction -- moves these coefficients and fails.
    """
    index = _synthetic_corpus(tmp_path)
    result = train(_base_cfg(index, tmp_path))
    got = np.asarray(result.coefficients, dtype=np.float64)
    expected = np.asarray(_MSE_REFERENCE_COEFFICIENTS, dtype=np.float64)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_physical_loss_changes_the_solution_away_from_the_mse_one(tmp_path):
    index = _synthetic_corpus(tmp_path)
    mse = train(_base_cfg(index, tmp_path))
    phys = train(_base_cfg(index, tmp_path, loss="physical", epochs=4, lr=0.05))
    assert np.asarray(phys.coefficients).shape == np.asarray(mse.coefficients).shape
    assert not np.allclose(
        np.asarray(phys.coefficients, dtype=np.float64),
        np.asarray(mse.coefficients, dtype=np.float64),
    )


def test_physical_run_still_reports_the_incumbent_metrics(tmp_path):
    """Model selection and history stay on val MSE so runs stay comparable."""
    index = _synthetic_corpus(tmp_path)
    result = train(_base_cfg(index, tmp_path, loss="physical", epochs=2))
    assert result.history, "history must still be recorded"
    assert "val_mse" in result.history[-1]
    assert np.isfinite(result.best_val_mse)


def test_unknown_loss_name_is_rejected_before_training(tmp_path):
    index = _synthetic_corpus(tmp_path)
    with pytest.raises(ValueError, match="loss"):
        train(_base_cfg(index, tmp_path, loss="not-a-loss"))


def test_poisson_objective_cannot_be_selected_under_peak_normalisation(tmp_path):
    """The LossConfig guard must fire from the training entry point too."""
    from ml.zernike.losses import LossConfig

    index = _synthetic_corpus(tmp_path)
    with pytest.raises(ValueError):
        train(
            _base_cfg(
                index,
                tmp_path,
                loss="physical",
                loss_weights=LossConfig(w_mse=0.0, w_poisson=1.0),
            )
        )


def test_roi_size_frac_scales_the_mask_with_the_grid(tmp_path):
    """The ROI is expressed as a fraction of the grid, not hard-coded pixels."""
    from ml.zernike.losses import roi_mask

    small = roi_mask((16, 16), (8.0, 8.0), "rectangle", 0.375 * 16, 4 / 3)
    large = roi_mask((64, 64), (32.0, 32.0), "rectangle", 0.375 * 64, 4 / 3)
    # Same physical fraction => same relative area, 16x more pixels at 4x the edge.
    # NOTE: numel(), not .size -- Tensor.size is a method, not a property.
    assert small.sum() / small.numel() == pytest.approx(
        large.sum() / large.numel(), rel=0.05
    )
    assert large.sum() == pytest.approx(16 * small.sum(), rel=0.2)


def test_model_output_is_intensity_and_matches_the_lost_reference_shape():
    """Guard the assumption the ROI terms rest on: (B,1,H,W) intensity."""
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=3, grid=16, observable="intensity", normalization="peak")
    )
    cos = torch.zeros(2, 1, 16, 16)
    sin = torch.zeros(2, 1, 16, 16)
    out = model(cos, sin)
    assert out.shape == (2, 1, 16, 16)
    assert out.min() >= 0.0


# Captured from the MSE path (grid=16, n_max=3, epochs=2, lr=0.01, batch=4,
# seed=0) on the synthetic corpus above. DO NOT REGENERATE -- if this test fails,
# the incumbent objective changed.
#
# Re-captured once, deliberately, when `_anchored_window` began locating the
# 0-order with `despike_k=3` (a lone hot pixel must not be able to move the crop;
# see `zero_order_center`). The median can shift the anchor by a pixel where two
# neighbours tie, which moves the crop, which changes the data the MSE path sees.
# That is an intended data-pipeline fix, not an optimisation change -- the
# objective is still plain MSE and the property this file protects (that adding a
# loss switch does not perturb the incumbent) is unaffected. Regenerate ONLY
# together with a deliberate decision to move the crop or the objective.
_MSE_REFERENCE_COEFFICIENTS = [
    -0.09963550418615341,
    0.14116191864013672,
    -0.14995715022087097,
    -0.1291704773902893,
    -0.14330090582370758,
    0.02307407185435295,
    0.07393936067819595,
    -0.14945508539676666,
    0.12371844053268433,
]