"""Tests for the offline FourierGSNet training runner (no hardware).

Everything is hardware-free: the corpus is a handful of synthetic pickles in
``tmp_path``, written in the **real two-level nested layout** the optimizers
produce, and training runs on CPU with a 1-layer / 4-channel network.

Covers the three layers the command is made of: the pure helpers (device
resolution, far-field reconstruction, montage rendering), the orchestration
(``run_offline_training`` end-to-end, validation, seeding, artifact contract) and
the CLI surface (``slm-gsnet train``).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from loguru import logger

from ml.gsnet_debug import train as gsnet_train
from ml.gsnet_debug.cache import GSNetCacheError, cache_dir_for
from ml.gsnet_debug.offline import build_record_index
from ml.gsnet_debug.train import (
    HISTORY_FIGURE_NAME,
    PANEL_KEYS,
    GsnetTrainParams,
    TrainResult,
    _ensure_lean_cache,
    _resolve_wandb_mode,
    _unit_sum,
    reconstruct_far_intensity,
    render_comparison,
    resolve_device,
    run_offline_training,
)
from ao_shaping.runners.runner_common import RunParams
from ml.gsnet.model import FourierGSNet

# Proven sample grid (the same value the dataset test locks).
GRID = 64

#: 15 = 1+2+3+4+5, i.e. a complete Noll set for n_max=4 (triangular, as the
#: real corpus is).
N_TERMS = 15


# --------------------------------------------------------------------------- #
# Synthetic corpus helpers
# --------------------------------------------------------------------------- #
def _far_field_frame(row: int, height: int = 100, width: int = 120) -> np.ndarray:
    """A deterministic far-field frame with an off-centre 0-order blob.

    The blob moves with ``row`` so records differ, which is the case
    ``farfield_to_grid`` must handle (the optical axis is not the frame centre
    on a 2f bench).
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cx = 30.0 + 0.7 * (row % 7)
    cy = 40.0 + 1.1 * (row % 5)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    frame = 250.0 * np.exp(-r2 / (2 * 6.0**2))
    frame[10:14, 90:94] += 40.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _zernike_coefficients(n_terms: int, row: int) -> np.ndarray:
    """Deterministic Noll-order coefficients with a non-trivial phase."""
    c = np.zeros(n_terms, dtype=np.float64)
    if n_terms > 3:
        c[3] = 0.8 + 0.01 * row
    if n_terms > 10:
        c[10] = -0.35 + 0.005 * row
    c[0] = 0.05 * row
    return c


def _write_pickle(root: Path, family: str, tag: str, n_records: int) -> Path:
    """Write one debug-style pickle in the real two-level nested layout."""
    timestamp = "20260926_164358"
    directory = root / f"{family}_{tag}" / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{family}_{tag}_{timestamp}.pkl"
    records: dict[int, dict[str, Any]] = {
        row: {
            "_img": _far_field_frame(row),
            "_c": _zernike_coefficients(N_TERMS, row),
            "_grad": np.zeros(N_TERMS, dtype=np.float64),
        }
        for row in range(n_records)
    }
    with open(path, "wb") as handle:
        pickle.dump(records, handle)
    return path


def _corpus(tmp_path: Path, n_records: int = 4) -> str:
    """Build a two-family nested corpus and return the glob that indexes it."""
    _write_pickle(tmp_path, "slm_pib", "run", n_records)
    _write_pickle(tmp_path, "slm_zernike_shaping", "run", n_records)
    return str(tmp_path / "slm_*")


def _tiny_params(roots: str, out_dir: Path | str, **overrides: Any) -> GsnetTrainParams:
    """A 1-epoch / 1-layer / 4-channel CPU configuration over a small corpus.

    W&B is off by default so no test ever touches the network or writes run
    files; ``wandb_mode`` is overridable for the W&B-specific cases.
    """
    defaults: dict[str, Any] = {
        "epochs": 1,
        "batch_size": 2,
        "grid": GRID,
        "num_layers": 1,
        "base_channels": 4,
        "device": "cpu",
        "max_samples": 4,
        "roots": roots,
        "out_dir": str(out_dir),
        "n_compare": 2,
        "wandb_mode": "disabled",
    }
    defaults.update(overrides)
    return GsnetTrainParams(**defaults)


def _png_size(path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` of a PNG without pulling in an image library."""
    assert path.is_file(), f"missing artifact: {path}"
    assert path.stat().st_size > 0, f"empty artifact: {path}"
    raw = path.read_bytes()
    # PNG signature + IHDR: width/height are big-endian uint32 at byte 16.
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return height, width


def _small_model() -> FourierGSNet:
    """A minimal network: the physics must hold for any phase, so layers do
    not matter for the propagation assertions."""
    return FourierGSNet(num_layers=1, base_channels=4)


# --------------------------------------------------------------------------- #
# 1. resolve_device
# --------------------------------------------------------------------------- #
class TestResolveDevice:
    def test_cpu_is_always_cpu(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_is_case_and_whitespace_tolerant(self) -> None:
        assert resolve_device("  CPU ") == "cpu"

    def test_auto_never_raises(self) -> None:
        assert resolve_device("auto") in ("cpu", "cuda")

    def test_empty_string_behaves_like_auto(self) -> None:
        assert resolve_device("") == resolve_device("auto")

    def test_explicit_cuda_degrades_to_cpu_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert resolve_device("cuda") == "cpu"

    def test_explicit_cuda_is_honoured_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert resolve_device("cuda") == "cuda"

    def test_auto_follows_cuda_availability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == "cuda"

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("auto") == "cpu"

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ValueError, match="device must be"):
            resolve_device("tpu")


# --------------------------------------------------------------------------- #
# 2. _resolve_wandb_mode
# --------------------------------------------------------------------------- #
class TestResolveWandbMode:
    @pytest.mark.parametrize("spec", ["offline", "online", "disabled"])
    def test_accepts_every_documented_mode(self, spec: str) -> None:
        assert _resolve_wandb_mode(spec) == spec

    def test_is_case_and_whitespace_tolerant(self) -> None:
        assert _resolve_wandb_mode("  OFFLINE ") == "offline"

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="wandb_mode must be one of"):
            _resolve_wandb_mode("synchronous")

    def test_default_is_offline(self) -> None:
        assert GsnetTrainParams().wandb_mode == "offline"


# --------------------------------------------------------------------------- #
# 3. reconstruct_far_intensity
# --------------------------------------------------------------------------- #
class TestReconstructFarIntensity:
    def _inputs(self, grid: int = GRID) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(0)
        source = torch.rand(1, 1, grid, grid, generator=generator) + 0.05
        target = torch.rand(1, 1, grid, grid, generator=generator)
        return source, target

    def test_returns_two_batched_planes_on_the_device(self) -> None:
        source, target = self._inputs()
        model = _small_model()

        pred_phase, far = reconstruct_far_intensity(model, source, target, "cpu")

        assert pred_phase.shape == source.shape
        assert far.shape == source.shape

    def test_predicted_phase_is_finite(self) -> None:
        source, target = self._inputs()
        pred_phase, _ = reconstruct_far_intensity(_small_model(), source, target, "cpu")

        assert torch.isfinite(pred_phase).all()

    def test_far_field_is_non_negative(self) -> None:
        source, target = self._inputs()
        _, far = reconstruct_far_intensity(_small_model(), source, target, "cpu")

        assert (far >= 0).all()

    def test_fft_energy_is_conserved(self) -> None:
        """Parseval: an unnormalised ``fft2`` scales total energy by ``H*W``.

        This is the strongest available check that the montage propagates with
        the same convention the network unrolls, and it is independent of the
        predicted phase.
        """
        grid = 32
        source, target = self._inputs(grid)
        _, far = reconstruct_far_intensity(_small_model(), source, target, "cpu")

        assert far.sum().item() == pytest.approx(
            source.sum().item() * grid * grid, rel=1e-4
        )

    def test_restores_the_previous_training_flag(self) -> None:
        source, target = self._inputs()
        model = _small_model()
        model.train()

        reconstruct_far_intensity(model, source, target, "cpu")

        assert model.training is True

    def test_restores_eval_mode(self) -> None:
        source, target = self._inputs()
        model = _small_model()
        model.eval()

        reconstruct_far_intensity(model, source, target, "cpu")

        assert model.training is False

    def test_prediction_does_not_accumulate_gradients(self) -> None:
        source, target = self._inputs()
        model = _small_model()

        pred_phase, _ = reconstruct_far_intensity(model, source, target, "cpu")

        assert pred_phase.grad_fn is None


# --------------------------------------------------------------------------- #
# 4. _unit_sum
# --------------------------------------------------------------------------- #
class TestUnitSum:
    def test_normalises_each_image_to_one(self) -> None:
        tensor = torch.rand(3, 1, 8, 8) + 0.1

        sums = _unit_sum(tensor).sum(dim=(-2, -1))

        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_tolerates_leading_singleton_dims(self) -> None:
        assert _unit_sum(torch.rand(1, 1, 8, 8)).shape == (1, 1, 8, 8)

    def test_survives_an_all_zero_plane(self) -> None:
        result = _unit_sum(torch.zeros(1, 1, 4, 4))

        assert torch.isfinite(result).all()


# --------------------------------------------------------------------------- #
# 5. render_comparison
# --------------------------------------------------------------------------- #
def _rows(n: int, grid: int = 8) -> list[dict[str, np.ndarray]]:
    """Synthesise ``n`` montage rows with every panel key present."""
    generator = np.random.default_rng(0)
    rows: list[dict[str, np.ndarray]] = []
    for i in range(n):
        row: dict[str, np.ndarray] = {}
        for key in PANEL_KEYS:
            if key in ("gt_phase", "pred_phase"):
                row[key] = generator.uniform(-np.pi, np.pi, (grid, grid))
            else:
                row[key] = generator.random((grid, grid))
        rows.append(row)
    return rows


class TestRenderComparison:
    def test_writes_the_montage_and_the_history_figure(self, tmp_path: Path) -> None:
        out = tmp_path / "comparison.png"

        written = render_comparison(out, _rows(2), [{"loss": 1.0}], title="t")

        assert written == out
        _png_size(out)
        _png_size(tmp_path / HISTORY_FIGURE_NAME)

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "comparison.png"

        render_comparison(out, _rows(1), [], title="t")

        _png_size(out)

    def test_montage_grows_with_the_row_count(self, tmp_path: Path) -> None:
        one = tmp_path / "one.png"
        three = tmp_path / "three.png"

        render_comparison(one, _rows(1), [], title="t")
        render_comparison(three, _rows(3), [], title="t")

        h_one, _ = _png_size(one)
        h_three, _ = _png_size(three)
        assert h_three > h_one

    def test_width_is_fixed_by_the_six_panels(self, tmp_path: Path) -> None:
        one = tmp_path / "one.png"
        two = tmp_path / "two.png"

        render_comparison(one, _rows(1), [], title="t")
        render_comparison(two, _rows(2), [], title="t")

        assert _png_size(one)[1] == _png_size(two)[1]

    def test_empty_rows_still_produce_a_valid_png(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"

        render_comparison(out, [], [], title="t")

        _png_size(out)
        _png_size(tmp_path / HISTORY_FIGURE_NAME)

    def test_empty_history_still_produces_a_valid_png(self, tmp_path: Path) -> None:
        render_comparison(tmp_path / "c.png", _rows(1), [], title="t")

        _png_size(tmp_path / HISTORY_FIGURE_NAME)

    def test_accepts_a_str_path(self, tmp_path: Path) -> None:
        out = tmp_path / "str.png"

        written = render_comparison(str(out), _rows(1), [], title="t")

        assert written == out
        _png_size(out)


# --------------------------------------------------------------------------- #
# 6. run_offline_training — validation
# --------------------------------------------------------------------------- #
class TestRunOfflineTrainingValidation:
    def test_zero_epochs_raises(self, tmp_path: Path) -> None:
        params = _tiny_params(str(tmp_path / "none_*"), tmp_path / "out", epochs=0)

        with pytest.raises(ValueError, match="epochs must be >= 1"):
            run_offline_training(RunParams(), params)

    def test_negative_max_samples_raises(self, tmp_path: Path) -> None:
        params = _tiny_params(
            str(tmp_path / "none_*"), tmp_path / "out", max_samples=-1
        )

        with pytest.raises(ValueError, match="max_samples must be >= 0"):
            run_offline_training(RunParams(), params)

    def test_negative_n_compare_raises(self, tmp_path: Path) -> None:
        params = _tiny_params(str(tmp_path / "none_*"), tmp_path / "out", n_compare=-1)

        with pytest.raises(ValueError, match="n_compare must be >= 0"):
            run_offline_training(RunParams(), params)

    def test_unknown_device_raises_before_any_work(self, tmp_path: Path) -> None:
        params = _tiny_params(str(tmp_path / "none_*"), tmp_path / "out", device="tpu")

        with pytest.raises(ValueError, match="device must be"):
            run_offline_training(RunParams(), params)

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        params = _tiny_params(str(tmp_path / "nothing_*"), tmp_path / "out")

        with pytest.raises(ValueError):
            run_offline_training(RunParams(), params)


# --------------------------------------------------------------------------- #
# 7. run_offline_training — end to end
# --------------------------------------------------------------------------- #
class TestRunOfflineTraining:
    def test_returns_a_populated_result(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out")

        result = run_offline_training(RunParams(), params)

        assert isinstance(result, TrainResult)
        assert result.out_dir == tmp_path / "out"
        assert result.n_records == 8
        assert result.device == "cpu"
        assert result.n_parameters > 0
        assert result.wandb_url is None

    def test_history_has_one_entry_per_epoch(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", epochs=2)

        result = run_offline_training(RunParams(), params)

        assert len(result.history) == 2
        for entry in result.history:
            assert {"loss", "phase_loss", "shaping_loss", "intensity_loss"} <= set(
                entry
            )
            assert np.isfinite(entry["loss"])

    def test_best_epoch_is_within_the_history(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", epochs=2)

        result = run_offline_training(RunParams(), params)

        assert 1 <= result.best_epoch <= 2
        assert np.isfinite(result.best_loss)

    def test_evaluation_means_are_reported(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out")

        result = run_offline_training(RunParams(), params)

        assert result.eval_means
        assert all(np.isfinite(v) for v in result.eval_means.values())

    def test_writes_every_artifact(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        out = tmp_path / "out"
        params = _tiny_params(roots, out)

        result = run_offline_training(RunParams(), params)

        assert set(result.artifacts) >= {"summary", "comparison", "history"}
        for path in result.artifacts.values():
            assert path.is_file(), path
        _png_size(result.artifacts["comparison"])
        _png_size(result.artifacts["history"])

    def test_writes_a_checkpoint(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out")

        result = run_offline_training(RunParams(), params)

        assert result.checkpoint_path is not None
        assert result.checkpoint_path.is_file()
        assert result.artifacts["checkpoint"] == result.checkpoint_path

    def test_summary_json_is_valid_and_complete(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", epochs=2)

        result = run_offline_training(RunParams(), params)
        payload = json.loads(result.artifacts["summary"].read_text(encoding="utf-8"))

        assert payload["config"]["epochs"] == 2
        assert payload["resolved"]["device"] == "cpu"
        assert payload["resolved"]["n_records"] == 8
        assert payload["resolved"]["roots"] == [roots]
        assert payload["resolved"]["out_dir"] == str(tmp_path / "out")
        assert payload["model"]["n_parameters"] == result.n_parameters
        assert payload["model"]["num_layers"] == 1
        assert len(payload["training"]["history"]) == 2
        assert payload["training"]["best_epoch"] == result.best_epoch
        assert payload["evaluation"]["means"] == result.eval_means
        assert payload["evaluation"]["n_samples"] > 0
        assert payload["wandb_url"] is None
        assert set(payload["artifacts"]) == set(result.artifacts)

    def test_summary_is_utf8_readable(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out")

        result = run_offline_training(RunParams(), params)
        text = result.artifacts["summary"].read_text(encoding="utf-8")

        assert "gsnet-offline-train" in text

    def test_max_samples_caps_the_epoch(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", max_samples=2)

        result = run_offline_training(RunParams(), params)

        assert result.n_records == 8
        assert result.out_dir.is_dir()

    def test_n_compare_zero_still_renders(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", n_compare=0)

        result = run_offline_training(RunParams(), params)

        _png_size(result.artifacts["comparison"])

    def test_repeated_runs_are_reproducible(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        first = run_offline_training(
            RunParams(), _tiny_params(roots, tmp_path / "a", epochs=2)
        )
        second = run_offline_training(
            RunParams(), _tiny_params(roots, tmp_path / "b", epochs=2)
        )

        assert [e["loss"] for e in first.history] == pytest.approx(
            [e["loss"] for e in second.history]
        )


# --------------------------------------------------------------------------- #
# 8. Seeding and output directory
# --------------------------------------------------------------------------- #
class TestSeeding:
    def test_run_seed_overrides_the_param_default(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", seed=1)

        result = run_offline_training(RunParams(seed=1234), params)

        assert result.seed == 1234

    def test_param_seed_is_used_when_run_seed_is_none(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", seed=99)

        result = run_offline_training(RunParams(seed=None), params)

        assert result.seed == 99

    def test_seed_reaches_the_summary(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", seed=7)

        result = run_offline_training(RunParams(seed=7), params)
        payload = json.loads(result.artifacts["summary"].read_text(encoding="utf-8"))

        assert payload["resolved"]["seed"] == 7

    def test_a_different_seed_gives_a_different_stream(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        first = run_offline_training(
            RunParams(seed=1), _tiny_params(roots, tmp_path / "a")
        )
        second = run_offline_training(
            RunParams(seed=2), _tiny_params(roots, tmp_path / "b")
        )

        assert first.n_parameters == second.n_parameters


class TestOutDir:
    def test_explicit_out_dir_is_used_verbatim(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        out = tmp_path / "custom" / "place"

        result = run_offline_training(RunParams(), _tiny_params(roots, out))

        assert result.out_dir == out
        assert result.out_dir.is_dir()

    def test_default_out_dir_is_under_the_run_root(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        root = tmp_path / "dataroot"
        params = _tiny_params(roots, out_dir="")

        result = run_offline_training(RunParams(dir=str(root)), params)

        assert result.out_dir.parent.parent == root
        assert result.out_dir.parent.name == "gsnet_train"
        assert result.out_dir.name.startswith("run-")

    def test_existing_out_dir_is_reused(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path)
        out = tmp_path / "out"
        out.mkdir()

        result = run_offline_training(RunParams(), _tiny_params(roots, out))

        assert result.out_dir == out
        assert result.artifacts["summary"].is_file()


# --------------------------------------------------------------------------- #
# 9. W&B behaviour
# --------------------------------------------------------------------------- #
class _FakeWandbRun:
    """Minimal stand-in for ``wandb.Run`` recording every interaction."""

    def __init__(self, url: str = "https://wandb.invalid/run") -> None:
        self.url = url
        self.logged: list[dict[str, Any]] = []
        self.steps: list[int | None] = []
        self.finish_calls = 0

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.logged.append(payload)
        self.steps.append(step)

    def finish(self) -> None:
        self.finish_calls += 1


class _FakeWandbLogger:
    """Stand-in for :mod:`ml.wandb_logger`."""

    def __init__(
        self, run: _FakeWandbRun | None = None, init_error: Exception | None = None
    ) -> None:
        self.run = run if run is not None else _FakeWandbRun()
        self.init_error = init_error
        self.init_kwargs: dict[str, Any] = {}
        self.comparison_calls: list[tuple[int, tuple[str, ...]]] = []

    def init_wandb(self, **kwargs: Any) -> _FakeWandbRun:
        self.init_kwargs = kwargs
        if self.init_error is not None:
            raise self.init_error
        return self.run

    def log_shaping_comparison(
        self, rows: list[dict[str, np.ndarray]], panel_keys: tuple[str, ...]
    ) -> str:
        self.comparison_calls.append((len(rows), panel_keys))
        return "fake-image"


class TestWandb:
    def test_import_failure_is_survivable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken optional extra must degrade to "no W&B", never to a crash.

        ``None`` in ``sys.modules`` makes the import machinery raise
        ``ImportError``; the attribute on the parent package has to go too,
        because ``from ml import wandb_logger`` would otherwise find it there
        and never reach ``sys.modules``.
        """
        import ml

        monkeypatch.delattr(ml, "wandb_logger", raising=False)
        monkeypatch.setitem(sys.modules, "ml.wandb_logger", None)

        assert gsnet_train._load_wandb_logger() is None

    def test_available_wandb_is_returned(self) -> None:
        module = gsnet_train._load_wandb_logger()

        assert module is not None
        assert hasattr(module, "init_wandb")
        assert hasattr(module, "log_shaping_comparison")

    def test_training_completes_and_saves_pngs_without_wandb(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ml

        monkeypatch.delattr(ml, "wandb_logger", raising=False)
        monkeypatch.setitem(sys.modules, "ml.wandb_logger", None)
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", wandb_mode="offline")

        result = run_offline_training(RunParams(), params)

        assert result.wandb_url is None
        _png_size(result.artifacts["comparison"])
        _png_size(result.artifacts["history"])
        assert result.artifacts["summary"].is_file()

    def test_a_failing_init_does_not_break_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeWandbLogger(init_error=RuntimeError("no credentials"))
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", wandb_mode="online")

        result = run_offline_training(RunParams(), params)

        assert result.wandb_url is None
        assert fake.run.finish_calls == 0
        _png_size(result.artifacts["comparison"])

    def test_successful_run_logs_epochs_and_finishes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeWandbLogger()
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)
        roots = _corpus(tmp_path)
        params = _tiny_params(roots, tmp_path / "out", epochs=2, lr=1e-3)

        result = run_offline_training(RunParams(), params)

        assert result.wandb_url == fake.run.url
        assert fake.run.finish_calls == 1

        epoch_payloads = [p for p in fake.run.logged if "train/loss" in p]
        assert len(epoch_payloads) == 2
        assert fake.run.steps[:2] == [1, 2]
        assert "train/lr" in epoch_payloads[0]

    def test_eval_metrics_are_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeWandbLogger()
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)
        roots = _corpus(tmp_path)

        run_offline_training(RunParams(), _tiny_params(roots, tmp_path / "out"))

        eval_payloads = [
            p for p in fake.run.logged if any(k.startswith("eval/") for k in p)
        ]
        assert eval_payloads

    def test_comparison_image_is_logged_with_the_panel_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeWandbLogger()
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)
        roots = _corpus(tmp_path)

        run_offline_training(RunParams(), _tiny_params(roots, tmp_path / "out"))

        assert fake.comparison_calls == [(2, PANEL_KEYS)]
        assert any("shaping_comparison" in p for p in fake.run.logged)

    def test_init_receives_the_resolved_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeWandbLogger()
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)
        roots = _corpus(tmp_path)
        params = _tiny_params(
            roots,
            tmp_path / "out",
            wandb_mode="OFFLINE",
            wandb_entity="team",
            wandb_name="run-name",
        )

        run_offline_training(RunParams(), params)

        assert fake.init_kwargs["mode"] == "offline"
        assert fake.init_kwargs["entity"] == "team"
        assert fake.init_kwargs["name"] == "run-name"
        assert fake.init_kwargs["project"] == "gsnet-offline-train"

    def test_empty_entity_and_name_become_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeWandbLogger()
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)
        roots = _corpus(tmp_path)

        run_offline_training(RunParams(), _tiny_params(roots, tmp_path / "out"))

        assert fake.init_kwargs["entity"] is None
        assert fake.init_kwargs["name"] is None

    def test_wandb_url_is_serialised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeWandbLogger()
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)
        roots = _corpus(tmp_path)

        result = run_offline_training(
            RunParams(), _tiny_params(roots, tmp_path / "out")
        )
        payload = json.loads(result.artifacts["summary"].read_text(encoding="utf-8"))

        assert payload["wandb_url"] == fake.run.url

    def test_finish_runs_even_when_training_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeWandbLogger()
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("training exploded")

        monkeypatch.setattr(gsnet_train, "train_gsnet", boom)
        roots = _corpus(tmp_path)

        with pytest.raises(RuntimeError, match="training exploded"):
            run_offline_training(RunParams(), _tiny_params(roots, tmp_path / "out"))

        assert fake.run.finish_calls == 1

    def test_a_failing_finish_does_not_mask_the_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _ExplosiveFinish(_FakeWandbRun):
            def finish(self) -> None:
                raise RuntimeError("network gone")

        fake = _FakeWandbLogger(run=_ExplosiveFinish())
        monkeypatch.setattr(gsnet_train, "_load_wandb_logger", lambda: fake)
        roots = _corpus(tmp_path)

        result = run_offline_training(
            RunParams(), _tiny_params(roots, tmp_path / "out")
        )

        assert result.artifacts["summary"].is_file()


# --------------------------------------------------------------------------- #
# 10. No hardware on this path
# --------------------------------------------------------------------------- #
class TestNoHardware:
    def test_module_does_not_import_the_hardware_optimizer(self) -> None:
        assert "fouriergsnet_optimize" not in sys.modules

    def test_importing_the_module_pulls_in_no_device_driver(self) -> None:
        import importlib

        importlib.reload(gsnet_train)

        assert "fouriergsnet_optimize" not in sys.modules

    def test_no_device_is_constructed_during_a_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail loudly if any camera / SLM / DM class is instantiated."""
        opened: list[str] = []

        def _trap(name: str) -> Any:
            def _factory(*args: Any, **kwargs: Any) -> Any:
                opened.append(name)
                raise AssertionError(f"hardware constructed: {name}")

            return _factory

        trapped: list[str] = []
        for module_name, attr in (
            ("ao_shaping.drivers.ccd", "MIICamera"),
            ("ao_shaping.drivers.ccd", "DahengCamera"),
            ("ao_shaping.drivers.slm.santec", "Santec"),
            ("ao_shaping.drivers.dm.nlight", "NLight"),
        ):
            module = pytest.importorskip(module_name)
            monkeypatch.setattr(
                module, attr, _trap(f"{module_name}.{attr}"), raising=False
            )
            trapped.append(f"{module_name}.{attr}")

        assert len(trapped) == 4
        roots = _corpus(tmp_path)
        run_offline_training(RunParams(), _tiny_params(roots, tmp_path / "out"))

        assert opened == []


# --------------------------------------------------------------------------- #
# 11. CLI surface
# --------------------------------------------------------------------------- #
class TestTrainCli:
    def test_group_help_lists_the_train_subcommand(self) -> None:
        from ao_shaping.runners.slm.gsnet_runner import run as group

        result = CliRunner().invoke(group, ["--help"])

        assert result.exit_code == 0, result.output
        assert "train" in result.output

    def test_train_help_exposes_the_key_options(self) -> None:
        from ao_shaping.runners.slm.gsnet_runner import run as group

        result = CliRunner().invoke(group, ["train", "--help"])

        assert result.exit_code == 0, result.output
        for opt in (
            "--epochs",
            "--batch-size",
            "--lr",
            "--w-phase",
            "--w-shaping",
            "--num-layers",
            "--base-channels",
            "--grid",
            "--device",
            "--max-samples",
            "--roots",
            "--out-dir",
            "--n-compare",
            "--wandb-project",
            "--wandb-mode",
        ):
            assert opt in result.output, f"missing train option: {opt}"

    def test_train_help_documents_the_offline_default_and_login(self) -> None:
        from ao_shaping.runners.slm.gsnet_runner import run as group

        result = CliRunner().invoke(group, ["train", "--help"])
        text = " ".join(result.output.split())

        assert "offline" in text
        assert "wandb login" in text
        assert "--wandb-mode online" in text

    def test_train_help_does_not_leak_the_search_options(self) -> None:
        from ao_shaping.runners.slm.gsnet_runner import run as group

        result = CliRunner().invoke(group, ["train", "--help"])

        assert "--algorithm" not in result.output
        assert "--pop_size" not in result.output
        assert "--optimizer_type" not in result.output

    def test_seed_belongs_to_run_params_not_the_train_params(self) -> None:
        """``--seed`` lives on :class:`RunParams`; declaring it twice is a hard
        click error, so the training dataclass must not contribute it (nor the
        other group-level flags)."""
        from ao_shaping.utils.cli_params import _collect_click_annotations

        options, _groups = _collect_click_annotations(GsnetTrainParams)
        names = [name for _fld, name, _tp, _delayed in options]

        assert "seed" not in names
        assert "dir" not in names
        assert "debug" not in names
        assert "epochs" in names

    def test_train_params_only_infers_click_types_it_can_map(self) -> None:
        """Guards the ``_patch_click_types`` contract for every field."""
        from ao_shaping.utils.cli_params import _collect_click_annotations

        options, _groups = _collect_click_annotations(GsnetTrainParams)

        assert len(options) == 18
        names = [name for _fld, name, _tp, _delayed in options]
        assert len(names) == len(set(names))

    def test_zero_epochs_exits_non_zero(self, tmp_path: Path) -> None:
        from ao_shaping.runners.slm.gsnet_runner import run as group

        result = CliRunner().invoke(
            group,
            ["-d", str(tmp_path), "train", "--epochs", "0", "--wandb-mode", "disabled"],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_end_to_end_invocation_writes_artifacts(self, tmp_path: Path) -> None:
        from ao_shaping.runners.slm.gsnet_runner import run as group

        roots = _corpus(tmp_path)
        out = tmp_path / "cli-out"

        result = CliRunner().invoke(
            group,
            [
                "train",
                "-d",
                str(tmp_path / "root"),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--grid",
                str(GRID),
                "--num-layers",
                "1",
                "--base-channels",
                "4",
                "--device",
                "cpu",
                "--max-samples",
                "4",
                "--roots",
                roots,
                "--out-dir",
                str(out),
                "--n-compare",
                "2",
                "--wandb-mode",
                "disabled",
            ],
        )

        assert result.exit_code == 0, result.output or repr(result.exception)
        assert (out / "summary.json").is_file()
        _png_size(out / "comparison.png")
        _png_size(out / HISTORY_FIGURE_NAME)

    def test_seed_flag_on_the_subcommand_reaches_the_run(self, tmp_path: Path) -> None:
        """``--seed`` is a :class:`RunParams` option on the *subcommand*.

        The group builds its own ``RunParams`` with defaults and the
        subcommand builds another, so group-level values never reach the
        trainer -- the module docstring's "options live on the subcommand"
        rule is load-bearing.
        """
        from ao_shaping.runners.slm.gsnet_runner import run as group

        roots = _corpus(tmp_path)
        out = tmp_path / "cli-seeded"

        result = CliRunner().invoke(
            group,
            [
                "train",
                "-d",
                str(tmp_path / "root"),
                "--seed",
                "4321",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--grid",
                str(GRID),
                "--num-layers",
                "1",
                "--base-channels",
                "4",
                "--device",
                "cpu",
                "--max-samples",
                "4",
                "--roots",
                roots,
                "--out-dir",
                str(out),
                "--n-compare",
                "1",
                "--wandb-mode",
                "disabled",
            ],
        )

        assert result.exit_code == 0, result.output or repr(result.exception)
        payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert payload["resolved"]["seed"] == 4321

    def test_default_out_dir_follows_the_subcommand_dir(self, tmp_path: Path) -> None:
        from ao_shaping.runners.slm.gsnet_runner import run as group

        roots = _corpus(tmp_path)
        root = tmp_path / "dataroot"

        result = CliRunner().invoke(
            group,
            [
                "train",
                "-d",
                str(root),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--grid",
                str(GRID),
                "--num-layers",
                "1",
                "--base-channels",
                "4",
                "--device",
                "cpu",
                "--max-samples",
                "4",
                "--roots",
                roots,
                "--n-compare",
                "1",
                "--wandb-mode",
                "disabled",
            ],
        )

        assert result.exit_code == 0, result.output or repr(result.exception)
        runs = list((root / "gsnet_train").glob("run-*"))
        assert len(runs) == 1
        assert (runs[0] / "summary.json").is_file()


# --------------------------------------------------------------------------- #
# 12. _ensure_lean_cache — the pre-build the training path depends on
# --------------------------------------------------------------------------- #
class TestEnsureLeanCache:
    """``run_offline_training`` pre-builds the lean cache before the first epoch.

    Without it the dataset silently re-reads each source pickle once per record
    per epoch, which is what makes a multi-GB corpus unbearably slow. These
    tests lock the three properties that matter: it really builds, it is
    idempotent, and a cache problem never becomes a training failure.
    """

    #: The files the dataset mmaps out of a cache directory.
    CACHE_FILES = ("c_flat.npy", "c_offsets.npy", "img_flat.npy", "keys.npy")

    @staticmethod
    def _unique_pickles(index: Any) -> set[Path]:
        return {pkl for pkl, _key in index.entries}

    def test_builds_a_cache_dir_for_every_pickle(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path, n_records=3)
        index = build_record_index(roots)
        pickles = self._unique_pickles(index)
        assert len(pickles) == 2, pickles
        assert len(index.entries) == 6, "2 families x 3 records"

        _ensure_lean_cache(index)

        for pkl in pickles:
            cache_dir = cache_dir_for(pkl)
            assert cache_dir.is_dir(), f"no cache built for {pkl}"
            meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
            assert meta["n_records"] == 3, meta
            assert Path(meta["source"]) == pkl, meta
            for name in self.CACHE_FILES:
                assert (cache_dir / name).is_file(), f"missing {name} in {cache_dir}"

    def test_is_idempotent_and_reuses_the_existing_cache(self, tmp_path: Path) -> None:
        roots = _corpus(tmp_path, n_records=3)
        index = build_record_index(roots)
        _ensure_lean_cache(index)

        before = {
            pkl: (cache_dir_for(pkl) / "c_flat.npy").stat().st_mtime_ns
            for pkl in self._unique_pickles(index)
        }
        time.sleep(0.02)
        _ensure_lean_cache(index)

        for pkl, stamp in before.items():
            after = (cache_dir_for(pkl) / "c_flat.npy").stat().st_mtime_ns
            assert after == stamp, f"cache was needlessly rebuilt for {pkl}"

    def test_summary_line_counts_pickles_not_records(self, tmp_path: Path) -> None:
        """Regression: the summary counts unique pickles, not index entries.

        ``RecordIndex`` is a frozen ``(pkl, key)`` tuple with no ``__len__``, so
        an earlier version calling ``len(index)`` here raised ``TypeError`` and
        killed the whole run before the first epoch.
        """
        roots = _corpus(tmp_path, n_records=4)
        index = build_record_index(roots)
        assert len(index.entries) == 8, "2 families x 4 records"
        assert not hasattr(index, "__len__"), "RecordIndex must stay un-sized"

        messages: list[str] = []
        sink_id = logger.add(
            lambda m: messages.append(str(m.record["message"])), level="INFO"
        )
        try:
            _ensure_lean_cache(index)
        finally:
            logger.remove(sink_id)

        ready = [m for m in messages if "lean cache ready" in m]
        assert len(ready) == 1, messages
        # 8 records but only 2 pickles -- the record count must not leak in.
        assert "2 of 2 pickles" in ready[0], ready[0]

    def test_cache_failure_degrades_to_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cacher failure must never abort training.

        ``GSNetDebugDataset`` falls back to ``pickle.load`` per family, so the
        pre-build is an optimisation, not a precondition.
        """
        roots = _corpus(tmp_path, n_records=2)
        index = build_record_index(roots)

        def boom(_index: Any) -> list[Path]:
            raise GSNetCacheError("disk on fire")

        monkeypatch.setattr(gsnet_train, "prepare_gsnet_cache", boom)

        messages: list[str] = []
        sink_id = logger.add(
            lambda m: messages.append(str(m.record["message"])), level="WARNING"
        )
        try:
            assert _ensure_lean_cache(index) is None  # must not propagate
        finally:
            logger.remove(sink_id)

        assert any("fall back" in m for m in messages), messages
        assert any("disk on fire" in m for m in messages), messages
        # Nothing was written, so the dataset must still find no cache at all.
        for pkl in self._unique_pickles(index):
            assert not cache_dir_for(pkl).exists()
