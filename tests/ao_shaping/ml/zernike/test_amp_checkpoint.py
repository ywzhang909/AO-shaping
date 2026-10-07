"""Read-side contract for ``train_amp``'s ``best_coefficients.pt``.

The round-trip test drives the **real** :func:`ml.zernike.train_amp.train`
writer (``save_checkpoint=True``) and feeds its output back through
:func:`ml.zernike.amp_checkpoint.load_trained_forward_model`, so reader/writer
agreement is enforced by the actual writer, not by a hand-built payload. The
pure geometry checker :func:`ml.zernike.amp_checkpoint.check_geometry` is
tested for every hard rule, for the unverified-geometry acknowledgement, and
for the invariant that an unrecorded field never lands in ``errors``.

These helpers are inline copies of the ones in
``test_train_amp_loss.py``: this repo forbids ``conftest.py``, so fixtures
cannot be shared across test files.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.zernike.amp_checkpoint import (
    UNVERIFIED_GEOMETRY_FIELDS,
    TrainedForwardModel,
    check_geometry,
    load_trained_forward_model,
)
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


def _run_real_writer(index, tmp_path):
    """Drive the real ``train`` writer with ``save_checkpoint=True`` once per test.

    ``index.families`` is empty for the synthetic corpus, so ``train`` trains on
    the real indexed corpus (the incumbent ``test_train_amp_loss`` does exactly
    this) -- the point is the *writer*, not the data.
    """
    return train(_base_cfg(index, tmp_path, save_checkpoint=True))


class TestRoundtrip:
    def test_coefficients_and_geometry_roundtrip_through_real_writer(self, tmp_path):
        index = _synthetic_corpus(tmp_path)
        result = _run_real_writer(index, tmp_path)

        ckpt_path = result.checkpoint or str(tmp_path / "out" / "best_coefficients.pt")
        loaded = load_trained_forward_model(ckpt_path)

        # The coefficients must round-trip exactly through the real writer.
        assert loaded.coefficients.shape == (result.n_modes,)
        assert np.array_equal(loaded.coefficients, np.asarray(result.coefficients))
        # And through the model's own parameter.
        assert np.array_equal(loaded.model.coefficients_array(), loaded.coefficients)

        # The recorded geometry must come back, matching what the writer saved.
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded.n_max == saved["n_max"] == result.n_max
        assert loaded.grid == saved["grid"]
        assert loaded.observable == saved["observable"]
        assert loaded.normalization == saved["normalization"]
        assert loaded.far_field_padding == saved["far_field_padding"]

    def test_load_keeps_the_model_parameter_object(self, tmp_path):
        """The copy must not swap the Parameter; it must mutate it in place."""
        index = _synthetic_corpus(tmp_path)
        result = _run_real_writer(index, tmp_path)
        loaded = load_trained_forward_model(
            result.checkpoint or str(tmp_path / "out" / "best_coefficients.pt")
        )
        param = loaded.model.coefficients
        # Still the exact object the model holds after the load.
        assert loaded.model.coefficients is param
        assert param.is_leaf and isinstance(param, torch.nn.Parameter)
        # ...and its values equal the loaded vector exactly.
        assert np.array_equal(loaded.model.coefficients_array(), loaded.coefficients)


def _corrupted_checkpoint(tmp_path, result, mutate):
    """Load the real writer's checkpoint, mutate a copy, re-save under ``tmp_path``."""
    path = result.checkpoint or str(tmp_path / "out" / "best_coefficients.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mutate(ckpt)
    out = tmp_path / "corrupt.pt"
    torch.save(ckpt, out)
    return out


class TestFailurePaths:
    def test_missing_key_names_it(self, tmp_path):
        index = _synthetic_corpus(tmp_path)
        result = _run_real_writer(index, tmp_path)
        path = _corrupted_checkpoint(tmp_path, result, lambda ckpt: ckpt.pop("normalization"))
        with pytest.raises(ValueError, match="normalization"):
            load_trained_forward_model(path)

    def test_wrong_coefficient_length_names_both(self, tmp_path):
        index = _synthetic_corpus(tmp_path)
        result = _run_real_writer(index, tmp_path)
        path = _corrupted_checkpoint(
            tmp_path,
            result,
            lambda ckpt: ckpt.update(
                {"coefficients": torch.zeros(result.n_modes - 1)}
            ),
        )
        with pytest.raises(ValueError) as exc:
            load_trained_forward_model(path)
        msg = str(exc.value)
        assert str(result.n_modes - 1) in msg
        assert str(result.n_modes) in msg

    def test_non_finite_coefficient_is_rejected(self, tmp_path):
        index = _synthetic_corpus(tmp_path)
        result = _run_real_writer(index, tmp_path)
        path = _corrupted_checkpoint(tmp_path, result, lambda ckpt: ckpt["coefficients"].fill_(float("nan")))
        with pytest.raises(ValueError, match="non-finite"):
            load_trained_forward_model(path)


def _loaded_from_real_writer(tmp_path):
    index = _synthetic_corpus(tmp_path)
    result = _run_real_writer(index, tmp_path)
    loaded = load_trained_forward_model(
        result.checkpoint or str(tmp_path / "out" / "best_coefficients.pt")
    )
    # A geometry consistent with the loaded model: the loop uses the same region,
    # the same order, and far_field_size = region * far_field_padding.
    region = loaded.grid
    n_orders = loaded.n_max
    far_field_size = region * loaded.far_field_padding
    return loaded, region, n_orders, far_field_size


class TestCheckGeometry:
    def test_happy_path_ok(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        rep = check_geometry(
            loaded,
            region=region,
            n_orders=n_orders,
            far_field_size=ff,
            assume_unverified_geometry=True,
        )
        assert rep.ok
        assert rep.errors == ()
        # A success note reports the geometry.
        assert any("consistent" in n for n in rep.notes)

    def test_grid_mismatch_reports_both(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        new_grid = loaded.grid + 1
        loaded = TrainedForwardModel(
            model=loaded.model,
            coefficients=loaded.coefficients,
            n_max=loaded.n_max,
            grid=new_grid,
            observable=loaded.observable,
            normalization=loaded.normalization,
            far_field_padding=loaded.far_field_padding,
            assumed=loaded.assumed,
            source=loaded.source,
            train_config=loaded.train_config,
        )
        rep = check_geometry(
            loaded,
            region=region,
            n_orders=n_orders,
            far_field_size=region * loaded.far_field_padding,
            assume_unverified_geometry=True,
        )
        assert not rep.ok
        # Both the model grid and the region are quoted.
        assert str(new_grid) in rep.errors[0]
        assert str(region) in rep.errors[0]

    def test_nmax_mismatch_reports_both(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        rep = check_geometry(
            loaded,
            region=region,
            n_orders=n_orders + 1,
            far_field_size=ff,
            assume_unverified_geometry=True,
        )
        assert not rep.ok
        assert str(loaded.n_max) in rep.errors[0]
        assert str(n_orders + 1) in rep.errors[0]

    def test_far_field_size_not_integer_multiple(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        bad = ff + (region - 1)  # not a multiple of region (ff+region-1 == k*region-1)
        rep = check_geometry(
            loaded,
            region=region,
            n_orders=n_orders,
            far_field_size=bad,
            assume_unverified_geometry=True,
        )
        assert not rep.ok
        assert str(bad) in rep.errors[0]
        assert str(region) in rep.errors[0]

    def test_effective_padding_differs_from_recorded(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        bad = region * (loaded.far_field_padding + 1)
        rep = check_geometry(
            loaded,
            region=region,
            n_orders=n_orders,
            far_field_size=bad,
            assume_unverified_geometry=True,
        )
        assert not rep.ok
        assert str(loaded.far_field_padding) in rep.errors[0]
        assert str(loaded.far_field_padding + 1) in rep.errors[0]

    def test_wrong_observable(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        if loaded.observable == "intensity":
            bad_obs = "amplitude"
        else:
            bad_obs = "intensity"
        loaded = TrainedForwardModel(
            model=loaded.model,
            coefficients=loaded.coefficients,
            n_max=loaded.n_max,
            grid=loaded.grid,
            observable=bad_obs,
            normalization=loaded.normalization,
            far_field_padding=loaded.far_field_padding,
            assumed=loaded.assumed,
            source=loaded.source,
            train_config=loaded.train_config,
        )
        rep = check_geometry(
            loaded,
            region=region,
            n_orders=n_orders,
            far_field_size=region * loaded.far_field_padding,
            assume_unverified_geometry=True,
        )
        assert not rep.ok
        assert bad_obs in rep.errors[0]
        assert "intensity" in rep.errors[0]

    def test_wrong_normalization(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        bad_norm = "sum" if loaded.normalization != "sum" else "none"
        loaded = TrainedForwardModel(
            model=loaded.model,
            coefficients=loaded.coefficients,
            n_max=loaded.n_max,
            grid=loaded.grid,
            observable=loaded.observable,
            normalization=bad_norm,
            far_field_padding=loaded.far_field_padding,
            assumed=loaded.assumed,
            source=loaded.source,
            train_config=loaded.train_config,
        )
        rep = check_geometry(
            loaded,
            region=region,
            n_orders=n_orders,
            far_field_size=region * loaded.far_field_padding,
            assume_unverified_geometry=True,
        )
        assert not rep.ok
        assert bad_norm in rep.errors[0]
        assert "peak" in rep.errors[0]

    def test_omit_assume_errors_and_passing_yields_note(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        # Omitting the flag: an error names every unverified field.
        rep_err = check_geometry(
            loaded, region=region, n_orders=n_orders, far_field_size=ff
        )
        assert not rep_err.ok
        err_text = " ".join(rep_err.errors)
        for name in UNVERIFIED_GEOMETRY_FIELDS:
            assert name in err_text

        # Passing the flag: no error for those fields, but a note naming all three.
        rep_ok = check_geometry(
            loaded,
            region=region,
            n_orders=n_orders,
            far_field_size=ff,
            assume_unverified_geometry=True,
        )
        assert rep_ok.ok
        note_text = " ".join(rep_ok.notes)
        for name in UNVERIFIED_GEOMETRY_FIELDS:
            assert name in note_text

    def test_unverified_field_never_in_errors(self, tmp_path):
        loaded, region, n_orders, ff = _loaded_from_real_writer(tmp_path)
        for flag in (False, True):
            rep = check_geometry(
                loaded,
                region=region,
                n_orders=n_orders,
                far_field_size=ff,
                assume_unverified_geometry=flag,
            )
            for name in UNVERIFIED_GEOMETRY_FIELDS:
                for err in rep.errors:
                    # The only permitted mention is the explicit "unverified
                    # geometry" acknowledgement line; it is a single named error,
                    # not a per-field comparison. A field must never be flagged as
                    # "different" from the loop's geometry.
                    if name in err and "unverified" in err:
                        continue
                    assert name not in err
