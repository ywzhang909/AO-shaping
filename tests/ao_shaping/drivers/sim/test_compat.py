from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.compat import AOConfig, TraditionalAOSystem


def _make_config(cn2: float) -> AOConfig:
    return AOConfig(
        N=64,
        L=0.04,
        Cn2=cn2,
        L0=20.0,
        l0=1e-3,
        dm_actuators=4,
        subapertures=4,
        pixel_scale=0.5,
        propagation_distance=1000.0,
    )


def _peak_ratio(image: np.ndarray) -> float:
    image_f = image.astype(float)
    return float(np.max(image_f) / np.sum(image_f))


def _bucket_ratio(image: np.ndarray, radius: int = 4) -> float:
    image_f = image.astype(float)
    cy, cx = np.array(image.shape) // 2
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    return float(np.sum(image_f[mask]) / np.sum(image_f))


def test_traditional_ao_system_no_turbulence_baseline() -> None:
    np.random.seed(0)
    ao = TraditionalAOSystem(_make_config(0.0))

    state = ao.reset()

    assert state["image"].shape == (64, 64)
    assert state["slopes"].shape == (32,)
    assert np.allclose(ao.turbulence.phase_screen, 0.0)
    assert state["strehl"] > 0.8
    assert _peak_ratio(state["image"]) > 1e-3


@pytest.mark.parametrize(
    ("cn2", "expected_min_phase_std"),
    [(1e-15, 0.05), (1e-14, 0.2), (5e-14, 0.5)],
)
def test_traditional_ao_system_turbulence_strength_matches_phase_scale(
    cn2: float,
    expected_min_phase_std: float,
) -> None:
    np.random.seed(1)
    ao = TraditionalAOSystem(_make_config(cn2))
    ao.reset()

    assert float(np.std(ao.turbulence.phase_screen)) > expected_min_phase_std


def test_stronger_turbulence_reduces_focus_concentration() -> None:
    phase_stds: list[float] = []

    for seed, cn2 in enumerate((0.0, 1e-14, 5e-14, 1e-13)):
        np.random.seed(seed)
        ao = TraditionalAOSystem(_make_config(cn2))
        ao.reset()
        phase_stds.append(float(np.std(ao.turbulence.phase_screen)))

    assert phase_stds[0] == pytest.approx(0.0)
    assert phase_stds[1] < phase_stds[2] < phase_stds[3]


def test_traditional_ao_step_clips_dm_command() -> None:
    np.random.seed(2)
    ao = TraditionalAOSystem(_make_config(1e-14))
    ao.reset()

    result = ao.step(np.full(ao.dm.total_actuators, 2.0))

    assert np.all(result["voltages"] <= 1.0)
    assert np.all(result["voltages"] >= -1.0)
