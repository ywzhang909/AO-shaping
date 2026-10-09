"""Tests for the canonical best-frame saver (issue #83 / R-62).

Three runners had a copy each. They were *not* interchangeable: one drew a red cross
marking the target square's half diagonal, one hid the axes, and the filenames and
titles differed. Those differences are parameters now, so they are pinned here --
if a future change quietly drops the marker, the square runner loses a reading
reference without any test noticing.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ao_shaping.display import save_best_image  # noqa: E402


@pytest.fixture
def frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.random((32, 32))
    img[14:18, 14:18] = 5.0  # a bright core so the saved PNG is not uniformly flat
    return img


def _read_png(path):
    import matplotlib.image as mpimg

    return mpimg.imread(path)


def test_writes_a_png_and_returns_the_path(tmp_path, frame) -> None:
    out = tmp_path / "best.png"
    returned = save_best_image(frame, out, title="hello")
    assert returned == out
    assert out.is_file() and out.stat().st_size > 0


def test_marker_changes_the_pixels(tmp_path, frame) -> None:
    """The square runner's red cross must survive the extraction."""
    plain = save_best_image(frame, tmp_path / "plain.png", title="t")
    marked = save_best_image(frame, tmp_path / "marked.png", title="t", marker_side=8)
    assert np.abs(_read_png(plain) - _read_png(marked)).max() > 0.01, (
        "marker_side had no effect on the rendered image"
    )


def test_hide_axes_changes_the_render(tmp_path, frame) -> None:
    """`axis("off")` drops the ticks and labels, so the canvas itself changes size."""
    shown = save_best_image(frame, tmp_path / "axes.png", title="t")
    hidden = save_best_image(frame, tmp_path / "noaxes.png", title="t", hide_axes=True)
    a, b = _read_png(shown), _read_png(hidden)
    assert a.shape != b.shape, "hide_axes left the rendered image identical"


def test_title_is_rendered(tmp_path, frame) -> None:
    """A different title must produce a different image, i.e. the title is not dropped."""
    a = save_best_image(frame, tmp_path / "a.png", title="alpha")
    b = save_best_image(frame, tmp_path / "b.png", title="beta")
    assert np.abs(_read_png(a) - _read_png(b)).max() > 0.01


def test_figure_is_closed_so_pyplot_state_does_not_leak(tmp_path, frame) -> None:
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    save_best_image(frame, tmp_path / "closed.png", title="t")
    assert len(plt.get_fignums()) == before, "the figure was left open"


def test_grayscale_map_is_monotonic_in_the_input(tmp_path) -> None:
    """`cmap="gray"` must map intensity monotonically and must not recolour the data.

    A 2x2 frame renders as four large blocks. `imshow`'s default origin is "upper", so
    input row 0 lands at the TOP: top-left < top-right < bottom-left < bottom-right for
    the ramp [[0, 0.25], [0.75, 1.0]].
    """
    levels = np.array([[0.0, 0.25], [0.75, 1.0]], dtype=float)
    out = save_best_image(levels, tmp_path / "ramp.png", title="t")
    img = _read_png(out)
    assert img.shape[2] in (3, 4), f"expected an RGB(A) render, got shape {img.shape}"
    lum = img[..., :3].mean(axis=2)
    h, w = lum.shape

    def block(r0, r1, c0, c1):
        return float(lum[int(h * r0) : int(h * r1), int(w * c0) : int(w * c1)].mean())

    top_left = block(0.10, 0.40, 0.10, 0.40)
    top_right = block(0.10, 0.40, 0.60, 0.90)
    bottom_left = block(0.60, 0.90, 0.10, 0.40)
    bottom_right = block(0.60, 0.90, 0.60, 0.90)

    ladder = [top_left, top_right, bottom_left, bottom_right]
    assert all(
        a <= b + 1e-6 for a, b in zip(ladder, ladder[1:], strict=False)
    ), f"gray map is not monotonic in the input: TL={top_left:.3f} TR={top_right:.3f} BL={bottom_left:.3f} BR={bottom_right:.3f}"
