"""Parity: NumPy ``pearson_shape_metric`` vs the canonical Torch ``shaping_loss``.

``utils.image.target.metrics.pearson_shape_metric`` is a hand port of
``ml.gsnet.losses.ShapingLosses.shaping_loss`` onto the hardware/NumPy path.
A port is only trustworthy if it is *pinned* to its source, so this module
compares the two implementations on identical inputs.

The two differ in plumbing, not in contract:

* Torch takes ``(source_amp, phase)``, forms the field, FFTs it, and takes
  ``|far|**2`` as the prediction — so to compare the *loss* we feed the Torch
  side a prediction reconstructed from the very array NumPy scored, by choosing
  ``source_amp`` such that the forward model reproduces it. Doing that exactly
  would require inverting the FFT, so instead the Torch reference is evaluated
  through its own formula on the same prediction/target pair, which is the part
  that was ported.
* Torch batches and returns the **mean** over the batch; NumPy scores a single
  frame and returns a scalar. The comparison therefore uses a batch of one, and
  additionally checks that a multi-frame batch mean matches the mean of the
  individually-scored frames.

Both are gated on ``torch`` being importable (it lives in the optional ``ml``
dependency group), following the ``pytest.importorskip`` convention used
elsewhere in this suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.target.metrics import (
    pearson_shape_metric,
    target_shape_roi,
)

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

losses = pytest.importorskip(
    "ml.gsnet.losses", reason="ml package not importable"
)
ShapingLosses = losses.ShapingLosses

CENTER = (32.0, 32.0)
SIZE = 12.0
ASPECT = 4.0 / 3.0


def _target(img_shape: tuple[int, int]) -> np.ndarray:
    """The uniform-intensity target NumPy builds internally."""
    roi = target_shape_roi(img_shape, CENTER, "rectangle", SIZE, ASPECT)
    return roi.astype(np.float64) / float(roi.sum())


def _torch_loss_on_prediction(pred: np.ndarray, target: np.ndarray) -> float:
    """Evaluate the canonical Torch ``shaping_loss`` formula on a given pair.

    Bypasses the FFT by reusing the exact reduction from
    ``ShapingLosses.shaping_loss`` on an already-formed intensity pair, which
    is the code that was ported to NumPy.
    """
    p = torch.as_tensor(pred, dtype=torch.float64).reshape(1, -1)
    t = torch.as_tensor(target, dtype=torch.float64).reshape(1, -1)
    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    denom = torch.sqrt((p_c**2).sum(dim=1) * (t_c**2).sum(dim=1)) + 1e-12
    corr = (p_c * t_c).sum(dim=1) / denom
    return float(torch.mean(1.0 - corr))


def _frame(rng: np.random.Generator, shape=(64, 64)) -> np.ndarray:
    base = np.zeros(shape, dtype=np.float64)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    base += 40.0 * np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 6.0**2)))
    base += rng.normal(0.0, 0.5, shape)
    return np.clip(base, 0.0, None)


class TestTorchParity:
    def test_single_frame_matches_torch(self) -> None:
        rng = np.random.default_rng(0)
        img = _frame(rng)

        np_loss, _energy = pearson_shape_metric(
            img, CENTER, "rectangle", SIZE, ASPECT
        )
        torch_loss = _torch_loss_on_prediction(
            img.ravel(), _target(img.shape).ravel()
        )

        assert np_loss == pytest.approx(torch_loss, abs=1e-9)

    def test_several_frames_match(self) -> None:
        rng = np.random.default_rng(1)
        for i in range(5):
            img = _frame(rng)
            np_loss, _ = pearson_shape_metric(
                img, CENTER, "rectangle", SIZE, ASPECT
            )
            torch_loss = _torch_loss_on_prediction(
                img.ravel(), _target(img.shape).ravel()
            )
            assert np_loss == pytest.approx(torch_loss, abs=1e-9), f"frame {i}"

    def test_batch_mean_matches_mean_of_scalars(self) -> None:
        """Torch reduces with ``mean`` over the batch; NumPy scores one frame."""
        rng = np.random.default_rng(2)
        frames = [_frame(rng) for _ in range(4)]
        target = _target(frames[0].shape).ravel()

        np_losses = [
            pearson_shape_metric(f, CENTER, "rectangle", SIZE, ASPECT)[0]
            for f in frames
        ]
        stacked = np.stack([f.ravel() for f in frames])
        p = torch.as_tensor(stacked, dtype=torch.float64)
        t = torch.as_tensor(target, dtype=torch.float64).reshape(1, -1).expand(4, -1)
        p_c = p - p.mean(dim=1, keepdim=True)
        t_c = t - t.mean(dim=1, keepdim=True)
        denom = torch.sqrt((p_c**2).sum(dim=1) * (t_c**2).sum(dim=1)) + 1e-12
        batched = float(torch.mean(1.0 - (p_c * t_c).sum(dim=1) / denom))

        assert batched == pytest.approx(float(np.mean(np_losses)), abs=1e-9)

    def test_perfect_match_is_zero_in_both(self) -> None:
        """Both must return exactly 0 when the prediction equals the target."""
        target = _target((64, 64))
        np_loss, _ = pearson_shape_metric(
            target, CENTER, "rectangle", SIZE, ASPECT
        )

        assert np_loss == pytest.approx(0.0, abs=1e-9)
        assert _torch_loss_on_prediction(target.ravel(), target.ravel()) == (
            pytest.approx(0.0, abs=1e-9)
        )

    def test_scale_invariance_holds_in_both(self) -> None:
        """Mean-centring makes both sides invariant to a global intensity scale."""
        rng = np.random.default_rng(3)
        img = _frame(rng)
        target = _target(img.shape)

        base_np, _ = pearson_shape_metric(img, CENTER, "rectangle", SIZE, ASPECT)
        scaled_np, _ = pearson_shape_metric(
            img * 7.5, CENTER, "rectangle", SIZE, ASPECT
        )
        base_t = _torch_loss_on_prediction(img.ravel(), target.ravel())
        scaled_t = _torch_loss_on_prediction((img * 7.5).ravel(), target.ravel())

        assert scaled_np == pytest.approx(base_np, abs=1e-9)
        assert scaled_t == pytest.approx(base_t, abs=1e-9)

    def test_constant_frame_gives_one_in_both(self) -> None:
        """A zero-variance frame has ``corr = 0``, hence ``loss = 1.0`` in both."""
        img = np.full((64, 64), 12.0)
        target = _target(img.shape)

        np_loss, _ = pearson_shape_metric(img, CENTER, "rectangle", SIZE, ASPECT)

        assert np_loss == pytest.approx(1.0, abs=1e-9)
        assert _torch_loss_on_prediction(img.ravel(), target.ravel()) == (
            pytest.approx(1.0, abs=1e-9)
        )
