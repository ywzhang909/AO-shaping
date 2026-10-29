"""Recorder-based process records: block records, cropping, table export.

The module is device-free by construction (it only reads run directories), so
these tests build a miniature run directory on disk instead of capturing.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import recording


def _fake_run(tmp_path, *, blocks=2, shape="square"):
    """A miniature transfer run directory: manifest, summary, iterations, frames."""
    run = tmp_path / "run"
    run.mkdir()
    xy = np.stack(np.meshgrid(np.arange(20, 60), np.arange(30, 70)), axis=-1).astype(float)
    captures, iterations, summary = [], [], []
    seed_name = "capture_seed.npz"
    np.savez_compressed(run / seed_name,
                        raw_frames=np.full((2, 80, 90), 20, dtype=np.uint8),
                        camera_xy=xy, target=np.ones((50, 40)),
                        core=np.ones((50, 40), dtype=bool),
                        model_phase=np.zeros((8, 8), dtype=np.float32))
    captures.append(dict(label=f"{shape}_seed", file=seed_name,
                         metrics=dict(cv=1.2, mean=1.2, ee_roi=0.9, target_peak=1.0),
                         exposure_ms=12.5, bit_depth=12, core_peak=1.0, saturation_pixels=0,
                         timestamp_utc="2026-10-27T00:00:00+00:00"))
    for block in range(blocks):
        for label, cv in ((f"{shape}_{block}_A1", 1.0 + block), (f"{shape}_{block}_B", 0.5 + block)):
            name = f"capture_{len(captures):04d}.npz"
            np.savez_compressed(run / name,
                                raw_frames=np.full((2, 80, 90), int(cv * 10), dtype=np.uint8),
                                camera_xy=xy, target=np.ones((50, 40)),
                                core=np.ones((50, 40), dtype=bool))
            captures.append(dict(label=label, file=name,
                                 metrics=dict(cv=cv, mean=cv, ee_roi=0.9, target_peak=1.0),
                                 exposure_ms=12.5, bit_depth=12, core_peak=1.0, saturation_pixels=0,
                                 timestamp_utc="2026-10-27T00:00:00+00:00"))
        summary.append(dict(shape=shape, block=block, accepted=True, acceptance="sequential_ccd",
                            accept_metric="cv", baseline=1.0 + block, candidate=0.5 + block,
                            model_loss_last=0.3, model_loss_previous=0.4, phase_lr=0.02))
        for iteration in range(3):
            iterations.append(dict(shape=shape, block=block, iteration=iteration,
                                   phase=np.full((8, 8), iteration, dtype=np.float32),
                                   terms=dict(loss=0.5 - 0.1 * iteration),
                                   timestamp_utc="2026-10-27T00:00:00+00:00"))
    (run / "manifest.json").write_text(json.dumps(captures), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (run / "iterations.pkl").open("wb") as stream:
        pickle.dump(iterations, stream)
    return run


def test_block_records_carry_measurement_and_acceptance(tmp_path):
    run = _fake_run(tmp_path)
    records = recording.block_records(run, "square/mono12_cv")
    assert len(records) == 2
    first = records[0]
    assert first["shape"] == "square" and first["block"] == 0
    assert first["stage"] == "square/mono12_cv"
    # The candidate's measured CV is the anchor's minus the improvement.
    assert first["cv_anchor"] == pytest.approx(1.0)
    assert first["cv_candidate"] == pytest.approx(0.5)
    assert first["cv_improvement"] == pytest.approx(-0.5)
    assert first["accepted"] is True
    assert first["acceptance"] == "sequential_ccd"
    assert first["exposure_ms"] == pytest.approx(12.5)
    assert first["bit_depth"] == 12
    # The phase, the measured far field and the in-block phase change.
    assert first[recording.RECORD_PHASE].shape == (8, 8)
    assert first[recording.RECORD_IMAGE].ndim == 2
    assert first[recording.RECORD_GRAD].shape == (8, 8)


def test_block_records_fall_back_to_the_anchor_without_a_candidate(tmp_path):
    run = _fake_run(tmp_path)
    rows = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in rows if not row["label"].endswith("_B")]
    (run / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    records = recording.block_records(run, "square/transport_cv")
    assert len(records) == 2
    assert records[0]["cv_candidate"] is None
    assert records[0]["cv_improvement"] is None


def test_block_phase_is_the_block_s_own_last_iteration(tmp_path):
    """Each block's phase panel must be *that block's* end phase.

    Regression: the panel used to be flat because the shared display colour
    scale was recomputed per frame (see the next test) -- but a record built
    from the wrong iteration would also freeze it, so the record is pinned here.
    """
    run = _fake_run(tmp_path, blocks=3)
    records = recording.block_records(run, "square/mono12_cv")
    assert [float(record[recording.RECORD_PHASE].max()) for record in records] == [2.0] * 3
    # The fake run's phases are constant maps (one value per iteration), so the
    # in-block change is a constant too: block 0 changes by its whole iteration
    # range (nothing to subtract), the later blocks show the difference between
    # two already-identical maps.
    assert float(np.abs(records[0][recording.RECORD_GRAD]).max()) == pytest.approx(2.0)
    assert float(np.abs(records[1][recording.RECORD_GRAD]).max()) == pytest.approx(0.0)


def test_process_gif_uses_one_phase_scale_for_every_frame(tmp_path):
    """A per-frame colour scale makes every frame look identical.

    The transfer session's phases differ by ~0.3 rad between neighbouring blocks
    on a ~15 rad map. Normalising each frame to its own 90% quantile gave each
    frame back its full range, and the pupil panel of the rendered GIF did not
    change from the first frame to the last.
    """
    import matplotlib

    matplotlib.use("Agg")

    run = _fake_run(tmp_path, blocks=3)
    records = recording.block_records(run, "square/mono12_cv")
    assert records[0][recording.RECORD_PHASE].shape == (8, 8)
    yy, xx = np.mgrid[0:8, 0:8]
    for index, record in enumerate(records):
        record[recording.RECORD_PHASE] = (15.0 + 0.3 * index + 2.0 * np.sin(xx / 2.0)
                                          * np.cos(yy / 2.0)).astype(np.float32)

    from matplotlib import pyplot as plt

    captured = []
    real_imshow = plt.Axes.imshow

    def spying_imshow(self, data, *args, **kwargs):
        if "vmin" in kwargs:
            captured.append((float(kwargs["vmin"]), float(kwargs["vmax"])))
        return real_imshow(self, data, *args, **kwargs)

    plt.Axes.imshow = spying_imshow
    try:
        recording.render_process_gif(records, tmp_path / "process.gif", fps=2.0,
                                     title="scale")
    finally:
        plt.Axes.imshow = real_imshow
        plt.close("all")

    # One limit per rendered phase-bearing panel, and every one of them the
    # same: the shared scale may be computed from the whole animation, but it
    # must not be recomputed from the frame being drawn.
    # The far field keeps its own scale (0..vmax); the phase and the in-block
    # change share one, and it must be the same in every frame.
    phase_limits = [limits for limits in captured if limits[0] < 0]
    assert len(phase_limits) == len(records) + 1
    assert len({(round(low, 9), round(high, 9)) for low, high in phase_limits}) == 1, \
        "the phase panels must not rescale per frame"


def test_crop_frame_keeps_the_window_and_decimates():
    frame = np.arange(100 * 120, dtype=float).reshape(100, 120)
    xy = np.stack(np.meshgrid(np.arange(10, 50), np.arange(20, 60)), axis=-1).astype(float)
    crop, box = recording.crop_frame(frame, xy, max_side=20, margin=2)
    assert crop.ndim == 2
    assert max(crop.shape) <= 20
    # The crop is the window, so its values come from inside the frame region.
    assert crop.min() >= frame[18:62, 8:52].min()
    assert box is None


def test_crop_frame_returns_the_target_box_in_crop_pixels():
    """The window is wider than the target on purpose; the outline has to say
    which part of the crop is the requested shape."""
    frame = np.zeros((100, 120))
    xy = np.stack(np.meshgrid(np.arange(10, 50), np.arange(20, 60)), axis=-1).astype(float)
    target = np.zeros(xy.shape[:2])
    target[5:15, 10:30] = 1.
    crop, box = recording.crop_frame(frame, xy, target=target, max_side=20, margin=2)
    assert box is not None
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 < crop.shape[1]
    assert 0 <= y0 < y1 < crop.shape[0]


def _fake_validate_run(tmp_path, *, shape="triangle"):
    """A miniature ``validate.run`` directory: manifest + one npz per capture."""
    run = tmp_path / "anchored"
    run.mkdir()
    xy = np.stack(np.meshgrid(np.arange(20, 60), np.arange(30, 70)), axis=-1).astype(float)
    target = np.zeros(xy.shape[:2])
    target[5:15, 10:30] = 1.
    rows = []
    for index, (label, cv) in enumerate(((f"{shape}_seed", 1.4), (f"{shape}_0_A1", 1.3),
                                         (f"{shape}_0_B1", 0.9), (f"{shape}_final", 0.9))):
        name = f"capture_{index:04d}.npz"
        np.savez_compressed(run / name,
                            raw_frames=np.full((2, 80, 90), int(cv * 10), dtype=np.uint16),
                            camera_xy=xy, target=target, core=target.astype(bool),
                            model_phase=np.full((8, 8), index, dtype=np.float32))
        rows.append(dict(label=label, file=name, metrics=dict(cv=cv), exposure_ms=12.5,
                         bit_depth=12, core_peak=10.0, saturation_pixels=0, loss=1.0 - .1 * index))
    (run / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    return run


def test_rebuild_run_reproduces_the_records_without_a_device(tmp_path):
    """A run whose live export failed can be re-exported from its own capture files."""
    run = _fake_validate_run(tmp_path)
    saved = recording.rebuild_run(run, tmp_path / "out")
    assert saved["records"] == 4
    assert Path(saved["csv"]).is_file() and Path(saved["hdf5"]).is_file()
    assert Path(saved["gif"]).is_file() and Path(saved["gif"]).stat().st_size > 0
    header = Path(saved["csv"]).read_text(encoding="utf-8").splitlines()[0]
    # the target outline is part of the record, not something the renderer guesses
    assert "target_box" in header


def test_save_records_writes_a_table_and_keeps_arrays_out_of_the_csv(tmp_path):
    from ao_shaping.utils.io.file import Recorder

    run = _fake_run(tmp_path)
    recorder = Recorder(mark="cv_candidate", mode="min")
    assert recording.append_run(recorder, run, "square/mono12_cv") == 2
    saved = recording.save_records(recorder, tmp_path / "records", mark="process")
    csv = Path(saved["csv"])
    assert csv.is_file() and Path(saved["hdf5"]).is_file()
    header = csv.read_text(encoding="utf-8").splitlines()[0]
    assert "cv_candidate" in header and "stage" in header
    # The array columns are written as sidecars, not inside the CSV.
    assert (tmp_path / "records" / "process_sidecars").is_dir()
