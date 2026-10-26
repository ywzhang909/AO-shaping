"""Large CCD targets must retain their requested extent and usable bandwidth."""
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.multishape_objective import target_at
from ao_shaping.tools.slm.transfer.geometric_seed import square_transport_phase


def test_1200_pixel_square_is_not_clipped_to_512_pixel_raster():
    y, x = np.mgrid[-700:701:4, -700:701:4]
    xy = np.stack((x + 1300, y + 970), axis=-1)
    target = target_at(xy, "square", (1300, 970), 1200)
    yy, xx = np.where(target > .5)
    assert np.ptp(x[yy, xx]) == 1200
    assert np.ptp(y[yy, xx]) == 1200
    assert not target[0].any() and not target[:, 0].any()


def test_geometric_seed_respects_bandwidth_and_returns_raw_radians():
    torch = pytest.importorskip("torch")
    p = SimpleNamespace(ramp_matrix=((7500., 0.), (0., 7500.)), origin_xy=(0., 0.),
                        defocus_diopter=0., diopter_scale=1.)
    forward = SimpleNamespace(params=p, amplitude=torch.ones((64, 64)), pitch=3.,
                              z4=torch.zeros((64, 64)), residual_basis=None)
    phase = square_transport_phase(forward, (0., 0.), 1200)
    assert phase.shape == (64, 64) and np.isfinite(phase).all()
    assert np.ptp(phase) > 2*np.pi
    forward.pitch = 15.
    with pytest.raises(ValueError, match="Nyquist"):
        square_transport_phase(forward, (0., 0.), 1200)


def test_native_12bit_audit_excludes_zero_and_recomputes_sequential_decision(tmp_path):
    import json
    from scipy.ndimage import binary_erosion
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, image_metrics
    from ao_shaping.tools.slm.transfer.artifacts import audit_run
    from ao_shaping.tools.slm.transfer.capture import to_panel

    yy, xx = np.mgrid[:9, :9]
    xy = np.stack((xx, yy), axis=-1)
    target = np.ones((9, 9), np.float32)
    objective = ShapeObjective(target, cv_only=True)
    objective.core = binary_erosion(objective.box, iterations=2) & (np.hypot(xx-4, yy-4) > 1.5)
    phase = np.zeros((8, 8), np.float32)
    rows = []
    for label, uneven in (("seed", True), ("0_A1", True), ("0_B", False), ("final", False)):
        raw = np.full((2, 12, 12), 900, np.uint16)
        if uneven:
            raw[:, :9, :9] += (xx * 20).astype(np.uint16)
        raw[:, 3:6, 3:6] = 4095
        metrics = image_metrics(raw.mean(0)[:9, :9], objective)
        file = f"{label}.npz"
        np.savez_compressed(tmp_path/file, raw_frames=raw, model_phase=phase,
            panel_phase=to_panel(phase), panel_gray=to_panel(phase).astype(np.uint16),
            camera_xy=xy, target=target, core=objective.core)
        rows.append(dict(label="square_"+label, file=file, metrics=metrics, loss=metrics["cv"]**2,
            exposure_ms=10., full_scale=4095., frames=[dict(timestamp_utc="2026-10-26T00:00:00+00:00",
                monotonic_ns=1, end_monotonic_ns=2, exposure_ms=10.)]*2))
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    (tmp_path/"config.json").write_text(json.dumps(dict(shapes=["square"], blocks=1, objective="cv",
        zero_xy=[4, 4], zero_radius=1.5, meter_target=True, accept_metric="cv")))
    (tmp_path/"summary.json").write_text(json.dumps([dict(shape="square", block=0, accepted=True,
        acceptance="sequential_ccd")]))
    assert audit_run(tmp_path)["status"] == "passed"
    rows[2]["metrics"]["cv"] = 3.
    (tmp_path/"manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="raw/manifest cv mismatch"):
        audit_run(tmp_path)
