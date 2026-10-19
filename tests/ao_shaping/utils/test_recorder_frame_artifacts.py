"""Companion-image emission from the Recorder debug-artifact writers.

``save_recorder_debug_artifacts`` used to write a 2x2 summary PNG plus a pickle
whose embedded frame arrays were the *only* record of those frames. ``data/`` is
gitignored, so losing a ``.pkl`` makes the run unreproducible -- which is exactly
how four figures in ``report/shape_trials/online_20261010.md`` became
un-regenerable. These tests pin the new behaviour: one far-field PNG + one X/Y
cross-section PNG (and a phase PNG when a phase key exists) per selected epoch,
in a ``<pkl_stem>_images/`` directory beside the pickle.

Every test builds a real (tiny) frame and lets the real renderer run -- mocking
the plotting layer would only re-test the mock. The two things that *are*
mocked are the failure path (test 6) and the cut-point convention (test 7), both
of which are about our wiring rather than about matplotlib.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ao_shaping.display.beam_profile import ProfileCut, cut_point  # noqa: E402
from ao_shaping.utils.io.file import (  # noqa: E402
    Recorder,
    _save_data_mode_debug_artifacts,
    save_recorder_debug_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _beam_frame(shape: tuple[int, int] = (24, 24)) -> np.ndarray:
    """A small far-field-like frame: a bright spot offset from the centre."""
    frame = np.zeros(shape, dtype=np.float64)
    frame[shape[0] // 2 - 2 : shape[0] // 2 + 2, shape[1] // 2 - 2 : shape[1] // 2 + 2] = 30.0
    frame += 1.0  # a pedestal, so prepare_frame's median subtraction is exercised
    return frame


def _phase_panel(shape: tuple[int, int] = (12, 10)) -> np.ndarray:
    """uint16 SLM grayscale panel (the dtype the renderer treats as a phase)."""
    return (np.arange(int(np.prod(shape)), dtype=np.uint16) % 1024).reshape(shape)


def _recorder(
    *,
    epochs: int = 5,
    objective: float | None = None,
    with_phase: bool = False,
) -> Recorder:
    """A :class:`Recorder` shaped like the SLM runners write.

    *objective*: one value per epoch, so tests can place the maximum wherever
    they want and check which epochs ``"summary"`` selects.
    """
    rec = Recorder("score", mode="max")
    for i in range(epochs):
        row = {
            "score": 0.5 - 0.1 * i if objective is None else objective[i],
            "_epoch": i,
            "_c": np.linspace(-1.0, 1.0, 4),
            "_img": _beam_frame() * (1.0 + 0.1 * i),
        }
        if with_phase:
            row["_phase"] = _phase_panel() + np.uint16(i)
        rec.append(row)
    return rec


def _image_dir(pkl_path: Path) -> Path:
    return pkl_path.parent / f"{pkl_path.stem}_images"


def _images(png_path: Path) -> list[Path]:
    """Companion images for a run whose ``.png`` is *png_path*."""
    d = _image_dir(png_path.with_suffix(".pkl"))
    return sorted(d.glob("*.png")) if d.is_dir() else []


# ---------------------------------------------------------------------------
# 1. default mode writes images into <stem>_images/ and the pkl still exists
# ---------------------------------------------------------------------------


class TestDefaultMode:
    def test_images_land_in_a_stem_named_subdirectory(self, tmp_path):
        rec = _recorder()

        png = save_recorder_debug_artifacts(
            rec,
            root_dir=str(tmp_path),
            subdir_prefix="slm_pib_shape",
            scalar_keys=("score",),
            img_keys=("_img",),
            json_payload={"algorithm": "spgd"},
        )

        pkl = png.with_suffix(".pkl")
        # the three artefacts the writer has always produced
        assert png.exists()
        assert pkl.exists()
        assert png.with_suffix(".json").exists()

        img_dir = _image_dir(pkl)
        assert img_dir.is_dir(), "companion images must sit in <pkl_stem>_images/"
        # ... and must NOT be loose files next to the pkl
        assert not list(png.parent.glob("*_farfield.png"))

        names = [p.name for p in sorted(img_dir.glob("*.png"))]
        assert names, "default mode must write images"
        # every image is named after the pkl stem
        assert all(n.startswith(f"{pkl.stem}_e") for n in names)

    def test_the_pickled_history_is_unchanged(self, tmp_path):
        """Images are additive: the pkl still holds the full record dict."""
        rec = _recorder()
        png = save_recorder_debug_artifacts(
            rec,
            root_dir=str(tmp_path),
            subdir_prefix="slm_pib_shape",
            scalar_keys=("score",),
            img_keys=("_img",),
        )
        with png.with_suffix(".pkl").open("rb") as fh:
            saved = pickle.load(fh)
        assert sorted(saved) == list(range(len(rec.history)))
        assert all(np.asarray(r["_img"]).ndim == 2 for r in saved.values())


# ---------------------------------------------------------------------------
# 2. frame_artifacts="none" writes zero image files (back-compat)
# ---------------------------------------------------------------------------


class TestNoneMode:
    def test_no_image_directory_is_created(self, tmp_path):
        rec = _recorder()

        png = save_recorder_debug_artifacts(
            rec,
            root_dir=str(tmp_path),
            subdir_prefix="slm_pib_shape",
            scalar_keys=("score",),
            img_keys=("_img",),
            frame_artifacts="none",
        )

        assert png.exists()
        assert png.with_suffix(".pkl").exists()
        assert png.with_suffix(".json").exists()
        assert not _image_dir(png.with_suffix(".pkl")).exists()
        assert _images(png) == []

    def test_none_does_not_even_reach_the_renderer(self, tmp_path, monkeypatch):
        """``"none"`` short-circuits before matplotlib is pulled in."""
        import ao_shaping.display.beam_profile as bp

        def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("save_frame_artifacts called in 'none' mode")

        monkeypatch.setattr(bp, "save_frame_artifacts", _boom)

        png = save_recorder_debug_artifacts(
            _recorder(),
            root_dir=str(tmp_path),
            subdir_prefix="slm_pib_shape",
            scalar_keys=("score",),
            img_keys=("_img",),
            frame_artifacts="none",
        )
        assert png.exists()


# ---------------------------------------------------------------------------
# 3. "summary" picks first / best / last
# ---------------------------------------------------------------------------


class TestSummarySelection:
    # max in the middle, and strictly interior, so first and last cannot be
    # mistaken for the best epoch
    OBJECTIVE = [0.10, 0.90, 0.30, 0.20, 0.05]

    def test_exactly_first_best_last_are_written(self, tmp_path):
        rec = _recorder(epochs=5, objective=self.OBJECTIVE)

        png = save_recorder_debug_artifacts(
            rec,
            root_dir=str(tmp_path),
            subdir_prefix="slm_pib_shape",
            scalar_keys=("score",),
            img_keys=("_img",),
            frame_artifacts="summary",
        )

        stem = png.with_suffix(".pkl").stem
        epochs = sorted({int(p.name.split("_e")[1].split("_")[0]) for p in _images(png)})
        assert epochs == [0, 1, 4], "expected first (0), best (1) and last (4)"

        # two images per selected epoch (far-field + X/Y cross-section)
        assert len(_images(png)) == 2 * len(epochs)
        for e in epochs:
            assert any(f"{stem}_e{e:04d}_" in p.name for p in _images(png))

    def test_all_mode_writes_every_epoch(self, tmp_path):
        rec = _recorder(epochs=5, objective=self.OBJECTIVE)

        png = save_recorder_debug_artifacts(
            rec,
            root_dir=str(tmp_path),
            subdir_prefix="slm_pib_shape",
            scalar_keys=("score",),
            img_keys=("_img",),
            frame_artifacts="all",
        )

        epochs = sorted({int(p.name.split("_e")[1].split("_")[0]) for p in _images(png)})
        assert epochs == [0, 1, 2, 3, 4]

    def test_best_epoch_falls_back_to_first_last_without_an_objective(self, tmp_path):
        """No objective column -> no "best", so summary collapses to first+last."""
        rec = _recorder(epochs=4)
        for row in rec.history:
            row.pop("score")
        rec._all_columns.discard("score")

        png = save_recorder_debug_artifacts(
            rec,
            root_dir=str(tmp_path),
            subdir_prefix="slm_pib_shape",
            scalar_keys=(),
            img_keys=("_img",),
            frame_artifacts="summary",
        )

        epochs = sorted({int(p.name.split("_e")[1].split("_")[0]) for p in _images(png)})
        assert epochs == [0, 3]


# ---------------------------------------------------------------------------
# 4. invalid mode raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidMode:
    @pytest.mark.parametrize("bad", ["", "first", "SUMMARY", "every", "none "])
    def test_recorder_wrapper_rejects_unknown_mode(self, tmp_path, bad):
        with pytest.raises(ValueError, match="frame_artifacts"):
            save_recorder_debug_artifacts(
                _recorder(),
                root_dir=str(tmp_path),
                subdir_prefix="slm_pib_shape",
                scalar_keys=("score",),
                img_keys=("_img",),
                frame_artifacts=bad,
            )

    def test_data_mode_wrapper_rejects_unknown_mode(self, tmp_path):
        with pytest.raises(ValueError, match="frame_artifacts"):
            _save_data_mode_debug_artifacts(
                data={0: {"_img": _beam_frame()}},
                png_path=tmp_path / "fig.png",
                pkl_path=tmp_path / "fig.pkl",
                json_path=tmp_path / "fig.json",
                title="t",
                json_payload={},
                frame_artifacts="some",
            )

    def test_rejection_happens_before_anything_is_written(self, tmp_path):
        """The wrapper is the stack frame the caller should see the error at."""
        with pytest.raises(ValueError):
            save_recorder_debug_artifacts(
                _recorder(),
                root_dir=str(tmp_path),
                subdir_prefix="slm_pib_shape",
                scalar_keys=("score",),
                img_keys=("_img",),
                frame_artifacts="nope",
            )
        assert list(tmp_path.rglob("*.pkl")) == []


# ---------------------------------------------------------------------------
# 5. frame key + uint16 phase key produce their respective PNGs
# ---------------------------------------------------------------------------


class TestArtefactNaming:
    def test_frame_key_yields_farfield_and_xy_profile(self, tmp_path):
        rec = _recorder(epochs=1)

        png = save_recorder_debug_artifacts(
            rec,
            root_dir=str(tmp_path),
            subdir_prefix="run",
            scalar_keys=("score",),
            img_keys=("_img",),
            frame_artifacts="all",
        )

        stem = png.with_suffix(".pkl").stem
        d = _image_dir(png.with_suffix(".pkl"))
        # `_name("_img")` strips the leading underscore -> "img"
        assert (d / f"{stem}_e0000_img_farfield.png").is_file()
        assert (d / f"{stem}_e0000_img_xy_profile.png").is_file()
        assert sorted(p.name for p in d.glob("*.png")) == [
            f"{stem}_e0000_img_farfield.png",
            f"{stem}_e0000_img_xy_profile.png",
        ]

    def test_phase_key_does_not_double_its_kind_suffix(self, tmp_path):
        """A key literally called ``_phase`` must not yield ``_phase_phase.png``."""
        rec = _recorder(epochs=1, with_phase=True)

        png = save_recorder_debug_artifacts(
            rec,
            root_dir=str(tmp_path),
            subdir_prefix="run",
            scalar_keys=("score",),
            img_keys=("_img", "_phase"),
            top_right_key="_phase",
            frame_artifacts="all",
        )

        stem = png.with_suffix(".pkl").stem
        d = _image_dir(png.with_suffix(".pkl"))
        names = sorted(p.name for p in d.glob("*.png"))
        assert f"{stem}_e0000_phase.png" in names
        assert f"{stem}_e0000_phase_phase.png" not in names

    def test_frame_and_phase_keys_are_kept_apart(self, tmp_path):
        """`_img` gets far-field + cross-section; the uint16 panel gets a phase."""
        from ao_shaping.display.beam_profile import save_frame_artifacts

        data = {0: {"_img": _beam_frame(), "_phase": _phase_panel()}}
        out = tmp_path / "imgs"

        written = save_frame_artifacts(
            data,
            out,
            frame_keys=("_img",),
            phase_keys=("_phase",),
            mode="all",
            stem="run",
        )

        assert sorted(Path(p).name for p in written) == [
            "run_e0000_img_farfield.png",
            "run_e0000_img_xy_profile.png",
            "run_e0000_phase.png",
        ]
        kinds = set(written.values())
        assert kinds == {"farfield", "xy_profile", "phase"}


# ---------------------------------------------------------------------------
# 6. a rendering failure must not cost us the pickle
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    def test_pkl_survives_a_raising_renderer(self, tmp_path, monkeypatch):
        """A plotting blow-up must cost the images, never the pickle.

        The project logs through loguru, which does not propagate into stdlib
        ``logging`` (so ``caplog`` sees nothing), hence the explicit sink.
        """
        import io

        from loguru import logger

        import ao_shaping.display.beam_profile as bp

        def _raise(*a, **k):
            raise RuntimeError("no font, no axes, no idea")

        monkeypatch.setattr(bp, "save_frame_artifacts", _raise)

        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING")
        try:
            png = save_recorder_debug_artifacts(
                _recorder(),
                root_dir=str(tmp_path),
                subdir_prefix="slm_pib_shape",
                scalar_keys=("score",),
                img_keys=("_img",),
                json_payload={"a": 1},
            )
        finally:
            logger.remove(handler_id)

        # everything the writer promised before the plotting step survived
        assert png.exists()
        pkl = png.with_suffix(".pkl")
        assert pkl.is_file() and pkl.stat().st_size > 0
        assert png.with_suffix(".json").read_text(encoding="utf8") == '{\n    "a": 1\n}'
        with pkl.open("rb") as fh:
            saved = pickle.load(fh)
        assert len(saved) == 5

        # ... and the failure was reported rather than swallowed silently
        assert "companion frame artifacts failed" in sink.getvalue()

    def test_an_unexpected_exception_type_is_also_survivable(
        self, tmp_path, monkeypatch
    ):
        """The guard catches ``Exception`` broadly, not one error class."""
        import ao_shaping.display.beam_profile as bp

        def _raise_type_error(*a, **k):
            raise TypeError("expected float, got None")

        monkeypatch.setattr(bp, "save_frame_artifacts", _raise_type_error)

        png = save_recorder_debug_artifacts(
            _recorder(),
            root_dir=str(tmp_path),
            subdir_prefix="run",
            scalar_keys=("score",),
            img_keys=("_img",),
        )
        assert png.exists()
        assert png.with_suffix(".pkl").exists()


# ---------------------------------------------------------------------------
# 7. the cut point is the centroid, not argmax
# ---------------------------------------------------------------------------


class TestCutIsCentroid:
    """Reuses the pattern of ``test_generate_shape_trial_profiles``' TestCutPoint.

    On a speckle field the brightest grain hops between near-equal peaks, so
    cutting there makes the X/Y profile wobble from frame to frame. The cut must
    follow the background-subtracted intensity centroid.
    """

    def _speckle_field(self) -> np.ndarray:
        frame = np.zeros((80, 80), dtype=float)
        frame[22:42, 22:42] = 1.0  # the broad beam, 400 counts total
        frame[10, 34] = 5.0  # one brighter grain, inside the beam's span
        return frame

    def test_cut_follows_centroid_while_argmax_hops(self):
        cut = cut_point(self._speckle_field())

        assert isinstance(cut, ProfileCut)
        # the beam alone centres on 31.5; the off-centre grain shifts it <0.5 px
        assert (cut.row, cut.col) == pytest.approx((31.5, 31.5), abs=0.5)
        # argmax jumps clean to the grain -- 21 px away from the cut
        assert (cut.argmax_row, cut.argmax_col) == (10.0, 34.0)
        assert abs(cut.argmax_row - cut.row) > 15.0

    def test_summary_reports_the_disagreement(self):
        """The centroid is the cut; argmax is carried only as a cross-check."""
        text = cut_point(self._speckle_field()).summary()
        assert "质心" in text
        assert "最亮像素" in text

    def test_profile_render_survives_an_offset_beam(self, tmp_path):
        """A beam well off frame centre must still be cut and written."""
        from ao_shaping.display.beam_profile import save_frame_artifacts

        frame = np.zeros((32, 32))
        frame[24:28, 6:10] = 50.0  # centre of mass ~ (25.5, 7.5), far from (15.5, 15.5)
        out = tmp_path / "imgs"

        written = save_frame_artifacts(
            {0: {"_img": frame}}, out, frame_keys=("_img",), mode="all", stem="off"
        )

        assert sorted(Path(p).name for p in written) == [
            "off_e0000_img_farfield.png",
            "off_e0000_img_xy_profile.png",
        ]
        assert all(Path(p).stat().st_size > 0 for p in written)