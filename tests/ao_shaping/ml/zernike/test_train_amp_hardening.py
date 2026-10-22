"""Regression tests for the ``ml.zernike.train_amp`` CLI hardening pass.

These pin fixes that were silent behaviour bugs rather than crashes:

* ``--lr`` reached ``AmpTrainConfig`` but was dropped by
  :meth:`AmpTrainConfig.optimizer_fit_config`, so every run trained at
  ``ZernikeAmpFitConfig.lr`` (0.05) regardless of the flag;
* the index cache was hard-coded and could silently go stale
  (``build_hw_index`` trusts a cache whose version/roots/limit match without
  re-listing the tree);
* the corpus provenance and the mixed-FOV guard had no coverage.

Everything here runs with no hardware and opens no ``.pkl`` -- the fake index
and the tiny pickle files live entirely in ``tmp_path``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.hwdataset import HwCorpusIndex, HwRecordRef, PhaseSource
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpFitConfig
from ml.zernike.train_amp import (
    AmpTrainConfig,
    _build_parser,
    _cam_size_guard,
    _corpus_summary,
    _warn_if_index_stale,
)


def _record(path: Path, position: int = 0, *, sidecar: dict | None = None) -> HwRecordRef:
    """Build a minimal record; only ``path`` and ``sidecar`` matter to the tests."""
    return HwRecordRef(
        path=path,
        position=position,
        key=None,
        family="slm_zernike_shaping",
        source=PhaseSource.PANEL_GRAY,
        n_terms=0,
        n_max=None,
        freeform_grid=None,
        exposure_ms=1.0,
        sidecar=sidecar if sidecar is not None else {},
    )


def _index(
    records: list[HwRecordRef], *, files_scanned: int = 1, files_usable: int = 1
) -> HwCorpusIndex:
    """Build a fake :class:`HwCorpusIndex` with no excluded records."""
    return HwCorpusIndex(
        records=tuple(records),
        excluded=MappingProxyType({}),
        files_scanned=files_scanned,
        files_usable=files_usable,
    )


class TestOptimizerFitConfigForwardsLr:
    """``--lr`` used to be a silent no-op: the config carried it but the fit
    config did not, so ``ZernikeAmpFitConfig.lr`` (default 0.05) won."""

    @pytest.mark.parametrize("lr", [0.123, 1e-4, 0.0, 0.5])
    def test_lr_is_forwarded(self, lr: float) -> None:
        cfg = AmpTrainConfig(lr=lr)
        assert cfg.optimizer_fit_config().lr == pytest.approx(lr)

    def test_forwarded_lr_overrides_the_fit_config_default(self) -> None:
        # Guard against a fix that forwarded the *model's* default instead of
        # the flag's value.
        model_default = ZernikeAmpFitConfig().lr
        assert (
            AmpTrainConfig(lr=model_default + 0.1).optimizer_fit_config().lr
            == pytest.approx(model_default + 0.1)
        )


class TestNoHyperparameterIsSilentlyDropped:
    """Guard the whole *bug class*, not just ``lr``.

    ``lr`` shipped broken for the entire life of the flag because
    :meth:`optimizer_fit_config` assembled a ``ZernikeAmpFitConfig`` by hand and
    simply omitted it. Nothing failed -- the field was set, the CLI accepted it,
    and the run silently used the dataclass default instead.

    These tests go one level further and assert against the *live* ``torch``
    optimizer, so any future field added to the config but forgotten in the
    fit-config assembly fails here rather than in a week of training.
    """

    @staticmethod
    def _optimizer_for(**overrides: object):
        from ml.zernike.models import ZernikeAmpModel
        from ml.zernike.train_amp import _build_optimizer

        cfg = AmpTrainConfig(**overrides)  # type: ignore[arg-type]
        model = ZernikeAmpModel(grid=16, n_max=3)
        return _build_optimizer(model, cfg)

    def test_lr_reaches_the_live_optimizer(self) -> None:
        opt = self._optimizer_for(lr=0.007)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.007)

    @pytest.mark.parametrize("name", ["momentum", "weight_decay"])
    def test_sgd_hyperparameters_reach_the_live_optimizer(self, name: str) -> None:
        opt = self._optimizer_for(optimizer="sgd", lr=0.01, **{name: 0.123})
        assert opt.param_groups[0][name] == pytest.approx(0.123)

    @pytest.mark.parametrize(
        ("optimizer_name", "expected"),
        [("adam", "Adam"), ("adamw", "AdamW"), ("sgd", "SGD")],
    )
    def test_optimizer_choice_reaches_the_live_optimizer(
        self, optimizer_name: str, expected: str
    ) -> None:
        opt = self._optimizer_for(optimizer=optimizer_name, lr=0.01)
        assert type(opt).__name__ == expected

    def test_every_config_optimizer_field_survives_the_round_trip(self) -> None:
        """Belt-and-braces: name the fields explicitly so adding one is a
        conscious decision rather than an omission."""
        cfg = AmpTrainConfig(lr=0.031, momentum=0.77, weight_decay=0.005)
        fit = cfg.optimizer_fit_config()
        for field in ("lr", "momentum", "weight_decay"):
            assert getattr(fit, field) == pytest.approx(getattr(cfg, field))


class TestIndexCacheFlag:
    """The cache path is configurable and its default matches the dataclass."""

    def test_default_matches_the_config(self) -> None:
        assert _build_parser().parse_args([]).index_cache == AmpTrainConfig().index_cache

    def test_custom_path_overrides(self) -> None:
        args = _build_parser().parse_args(["--index-cache", "custom.json"])
        assert args.index_cache == "custom.json"

    def test_empty_string_disables_the_cache(self) -> None:
        # ``train()`` does ``cfg.index_cache or None``, so "" must survive parsing.
        assert _build_parser().parse_args(["--index-cache", ""]).index_cache == ""


class TestParserDefaults:
    """CLI defaults must be sourced from the dataclasses, not hard-coded.

    Only fields whose parser dest equals the dataclass field name are listed;
    ``families`` (list vs tuple), ``loss_weights`` (nested) and the boolean
    ``--no-*`` flags are deliberately excluded.
    """

    @pytest.mark.parametrize(
        ("dest", "expected"),
        [
            ("grid", ZernikeAmpConfig().grid),
            ("n_max", ZernikeAmpConfig().n_max),
            ("observable", ZernikeAmpConfig().observable),
            ("normalization", ZernikeAmpConfig().normalization),
            ("far_field_padding", ZernikeAmpConfig().far_field_padding),
            ("center_crop", ZernikeAmpConfig().center_crop),
            ("image_mode", AmpTrainConfig().image_mode),
            ("loss", AmpTrainConfig().loss),
            ("optimizer", AmpTrainConfig().optimizer),
            ("lr", AmpTrainConfig().lr),
            ("momentum", AmpTrainConfig().momentum),
            ("weight_decay", AmpTrainConfig().weight_decay),
            ("target_size_frac", AmpTrainConfig().target_size_frac),
            ("target_aspect_ratio", AmpTrainConfig().target_aspect_ratio),
            ("max_train", AmpTrainConfig().max_train),
            ("max_val", AmpTrainConfig().max_val),
            ("val_fraction", AmpTrainConfig().val_fraction),
            ("beam_samples", AmpTrainConfig().beam_samples),
            ("epochs", AmpTrainConfig().epochs),
            ("batch_size", AmpTrainConfig().batch_size),
            ("l2_penalty", AmpTrainConfig().l2_penalty),
            ("grad_clip", AmpTrainConfig().grad_clip),
            ("seed", AmpTrainConfig().seed),
            ("device", AmpTrainConfig().device),
            ("index_cache", AmpTrainConfig().index_cache),
        ],
    )
    def test_default_matches_the_dataclass(self, dest: str, expected: object) -> None:
        args = _build_parser().parse_args([])
        assert getattr(args, dest) == expected

    def test_no_center_crop_flag_flips_it(self) -> None:
        assert _build_parser().parse_args(["--no-center-crop"]).center_crop is False


class TestIndexStaleness:
    """A cache that disagrees with the on-disk tree must be flagged."""

    def test_matching_index_is_not_stale(self, tmp_path: Path) -> None:
        pkl = tmp_path / "a.pkl"
        pkl.write_bytes(b"")
        index = _index([_record(pkl.resolve())])
        report = _warn_if_index_stale(index, (str(tmp_path),), None, "cache.json")
        assert report["stale"] is False
        assert report["on_disk_not_indexed"] == []
        assert report["in_index_not_on_disk"] == []
        assert report["n_on_disk"] == 1
        assert report["n_indexed"] == 1

    def test_new_pickle_on_disk_marks_stale(self, tmp_path: Path) -> None:
        indexed = tmp_path / "a.pkl"
        indexed.write_bytes(b"")
        added = tmp_path / "b.pkl"
        added.write_bytes(b"")  # on disk but not in the index
        index = _index([_record(indexed.resolve())])
        report = _warn_if_index_stale(index, (str(tmp_path),), None, None)
        assert report["stale"] is True
        assert report["on_disk_not_indexed"] == [
            os.path.normcase(os.path.abspath(str(added.resolve())))
        ]
        assert report["in_index_not_on_disk"] == []

    def test_missing_pickle_marks_stale(self, tmp_path: Path) -> None:
        present = tmp_path / "a.pkl"
        present.write_bytes(b"")
        gone = tmp_path / "b.pkl"  # indexed but never written to disk
        index = _index([_record(present.resolve()), _record(gone.resolve(), position=1)])
        report = _warn_if_index_stale(index, (str(tmp_path),), None, None)
        assert report["stale"] is True
        assert report["in_index_not_on_disk"] == [
            os.path.normcase(os.path.abspath(str(gone.resolve())))
        ]
        assert report["on_disk_not_indexed"] == []

    def test_empty_directory_is_stale_when_index_is_nonempty(self, tmp_path: Path) -> None:
        index = _index([_record((tmp_path / "a.pkl").resolve())])
        report = _warn_if_index_stale(index, (str(tmp_path),), None, None)
        assert report["stale"] is True
        assert report["n_on_disk"] == 0
        assert report["n_indexed"] == 1


class TestCorpusSummary:
    """The provenance dict is JSON-serialisable and uses ``PhaseSource.value``."""

    def test_summary_shape(self) -> None:
        index = _index(
            [_record(Path("a.pkl")), _record(Path("b.pkl"), position=1)],
            files_scanned=2,
            files_usable=1,
        )
        summary = _corpus_summary(index, "cache.json", stale=True)
        assert summary == {
            "files_scanned": 2,
            "files_usable": 1,
            "n_records": 2,
            "n_families": 1,
            "families": {"slm_zernike_shaping": 2},
            "sources": {"panel_gray": 2},
            "index_cache": "cache.json",
            "index_cache_stale": True,
        }


class TestCamSizeGuard:
    """Mixing camera FOVs under one ``far_field_padding`` must be flagged."""

    def test_mixed_cam_sizes_are_flagged(self) -> None:
        records = [
            _record(Path("a.pkl"), sidecar={"cam_size": 250}),
            _record(Path("b.pkl"), position=1, sidecar={"cam_size": 320}),
        ]
        report = _cam_size_guard(records, far_field_padding=10)
        assert report["mixed"] is True
        assert report["distinct_cam_sizes"] == ["250", "320"]
        assert report["cam_size_counts"] == {"250": 1, "320": 1}
        assert report["unknown_cam_size_records"] == 0

    def test_uniform_cam_size_is_silent(self) -> None:
        records = [
            _record(Path("a.pkl"), sidecar={"cam_size": 250}),
            _record(Path("b.pkl"), position=1, sidecar={"cam_size": 250}),
        ]
        report = _cam_size_guard(records, far_field_padding=10)
        assert report["mixed"] is False
        assert report["distinct_cam_sizes"] == ["250"]
        assert report["cam_size_counts"] == {"250": 2}

    def test_missing_cam_size_counts_as_unknown(self) -> None:
        records = [_record(Path("a.pkl"), sidecar={})]
        report = _cam_size_guard(records, far_field_padding=10)
        assert report["unknown_cam_size_records"] == 1
        assert report["cam_size_counts"] == {"unknown": 1}
        assert report["mixed"] is False  # a single unknown size is not a mix