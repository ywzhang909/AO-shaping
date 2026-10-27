"""Tests for the Recorder <-> GIF rendering helpers added in file.py.

Covers:
* :func:`render_recorder_gif` (variable-type aware rendering)
* :meth:`Recorder.from_saved` / ``from_dir`` / ``load`` (reconstruction)
* round-trip correctness between a saved artifact dir and a restored Recorder
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ao_shaping.utils.io.file import (
    Recorder,
    render_recorder_gif,
    save_recorder_debug_artifacts,
)
from ao_shaping.utils.io.recorder_keys import RECORD_EPOCH, RECORD_IMAGE, RECORD_PHASE


def _make_recorder(epochs: int = 6) -> Recorder:
    rec = Recorder(mark="loss", mode="min")
    for i in range(epochs):
        rec.append(
            {
                "loss": 1.0 / (i + 1),
                "_epoch": i,
                "_img": (np.arange(64, dtype=np.float64).reshape(8, 8) + i),
                "_phase": (np.arange(64, dtype=np.uint16).reshape(8, 8) * (i + 1))
                % 1024,
                "_c": np.array([0.1 * (i + 1), -0.2 * (i + 1), 0.3], dtype=np.float64),
                "note": f"epoch-{i}",
            }
        )
    return rec


def test_render_recorder_gif_writes_valid_gif(tmp_path):
    rec = _make_recorder(4)
    out = tmp_path / "out.gif"
    returned = render_recorder_gif(rec, out, fps=4, title="t")
    assert returned == out
    assert out.exists() and out.stat().st_size > 0
    # GIF magic
    with open(out, "rb") as fh:
        magic = fh.read(6)
    assert magic in (b"GIF87a", b"GIF89a")
    # PIL can open and count frames
    from PIL import Image

    with Image.open(out) as im:
        frames = 0
        try:
            while True:
                frames += 1
                im.seek(im.tell() + 1)
        except EOFError:
            pass
    assert frames == 4


def test_render_recorder_gif_max_frames_subsamples(tmp_path):
    rec = _make_recorder(10)
    out = tmp_path / "sub.gif"
    render_recorder_gif(rec, out, max_frames=3)
    from PIL import Image

    with Image.open(out) as im:
        frames = 0
        try:
            while True:
                frames += 1
                im.seek(im.tell() + 1)
        except EOFError:
            pass
    assert frames == 3


def test_render_recorder_gif_empty_raises(tmp_path):
    rec = Recorder(mark="J", mode="max")
    with pytest.raises(ValueError):
        render_recorder_gif(rec, tmp_path / "x.gif")


def test_render_recorder_gif_accepts_pkl_path(tmp_path):
    rec = _make_recorder(3)
    pkl = tmp_path / "r.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(rec, fh)
    out = tmp_path / "frompkl.gif"
    render_recorder_gif(str(pkl), out)
    assert out.exists() and out.stat().st_size > 0


def test_render_recorder_gif_accepts_dict(tmp_path):
    rec = _make_recorder(3)
    data = {r["_epoch"]: dict(r) for r in rec.history}
    out = tmp_path / "fromdict.gif"
    render_recorder_gif(data, out)
    assert out.exists()


def test_from_saved_direct_pickle_roundtrip(tmp_path):
    rec = _make_recorder(5)
    pkl = tmp_path / "rec.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(rec, fh)
    restored = Recorder.from_saved(pkl)
    assert len(restored) == len(rec)
    assert restored.mark == rec.mark
    # values preserved
    for a, b in zip(restored.history, rec.history):
        assert a["loss"] == b["loss"]
        assert a["_epoch"] == b["_epoch"]


def test_from_saved_debug_artifacts_roundtrip(tmp_path):
    rec = _make_recorder(5)
    save_recorder_debug_artifacts(
        rec,
        str(tmp_path),
        "pref",
        scalar_keys=("loss",),
        img_keys=(RECORD_IMAGE, RECORD_PHASE),
        d1_keys=("_c",),
        output_dir=tmp_path,
    )
    restored = Recorder.from_dir(tmp_path)
    assert len(restored) == len(rec)
    assert restored.mark == "loss"
    # loss values preserved through the debug artifact
    for a, b in zip(restored.history, rec.history):
        assert abs(a["loss"] - b["loss"]) < 1e-9


def test_from_saved_json_fallback(tmp_path):
    import json

    rows = [{"loss": 1.0 / (i + 1), "_epoch": i} for i in range(4)]
    json_path = tmp_path / "run.json"
    json_path.write_text(
        json.dumps({str(i): r for i, r in enumerate(rows)}), encoding="utf-8"
    )
    restored = Recorder.from_saved(json_path, mark="loss")
    assert len(restored) == 4
    assert restored.mark == "loss"


def test_from_saved_csv_fallback(tmp_path):
    import pandas as pd

    df = pd.DataFrame([{"loss": 1.0 / (i + 1), "_epoch": i} for i in range(4)])
    csv_path = tmp_path / "run.csv"
    df.to_csv(csv_path, index=False)
    restored = Recorder.from_saved(csv_path, mark="loss")
    assert len(restored) == 4


def test_from_dir_missing_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        Recorder.from_dir(empty)


def test_load_alias(tmp_path):
    rec = _make_recorder(2)
    pkl = tmp_path / "a.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(rec, fh)
    restored = Recorder.load(pkl)
    assert len(restored) == 2


def test_render_then_from_saved_roundtrip_values(tmp_path):
    """End-to-end: save -> restore -> render produces same frame count."""
    rec = _make_recorder(5)
    save_recorder_debug_artifacts(
        rec,
        str(tmp_path),
        "pref",
        scalar_keys=("loss",),
        img_keys=(RECORD_IMAGE,),
        output_dir=tmp_path,
    )
    restored = Recorder.from_dir(tmp_path)
    out = tmp_path / "rt.gif"
    render_recorder_gif(restored, out)
    from PIL import Image

    with Image.open(out) as im:
        frames = 0
        try:
            while True:
                frames += 1
                im.seek(im.tell() + 1)
        except EOFError:
            pass
    assert frames == len(restored) == len(rec)
