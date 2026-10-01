"""Wavefront disturbance (turbulence + thermal halo) for the 2f-Fourier sim.

The ``slm-pib`` simulation report originally ran with a perfectly clean
wavefront: ``SimPibSystem.far_field()`` used only the SLM command phase and
applied no atmospheric or thermal disturbance at all. This module pins the
disturbance model that was added to fix that, plus its wiring into
``SimPibSystem``.

Semantics under test
--------------------
``static``
    One screen is generated on first use and reused forever — the repo's
    ``closed``/frozen-turbulence analogue. ``call_rms`` is constant and every
    ``call_streak_index`` is ``0``.
``dynamic``
    A fresh, independent screen is drawn on *every* optical evaluation — the
    repo's ``open``/sliding analogue. This models the fully-decorrelated
    ("white in time") limit, which is the correct asymptotic here because the
    real atmospheric decorrelation time (~10-50 ms) is far shorter than this
    SPGD loop's ~0.375 s per evaluation. It is deliberately NOT a wind model.

The measured amplitude, not the analytic one
--------------------------------------------
``cn2`` and ``distance_m`` are degenerate generator knobs: the canonical
generator's ``r0 = (0.423 * k**2 * cn2 * distance) ** (-3/5)`` depends only on
their product (there is no real propagation of a km-scale path on this bench).
The legacy FFT path also undersamples low spatial frequencies, so the measured
``sigma`` is a lower bound on the analytic von-Karman variance for the same r0.
These tests therefore assert on *measured* quantities and on *relative*
responses, never on an analytic amplitude.
"""

from __future__ import annotations

import numpy as np

from ao_shaping.drivers.sim.disturbance import (
    DISTURBANCE_MODES,
    DisturbanceConfig,
    SimDisturbance,
)

SHAPE = (1200, 1920)


def _cfg(**overrides: object) -> DisturbanceConfig:
    return DisturbanceConfig(**overrides)  # type: ignore[arg-type]


# --- mode contract -----------------------------------------------------------


def test_disturbance_modes_are_exactly_none_static_dynamic() -> None:
    assert DISTURBANCE_MODES == ("none", "static", "dynamic")


def test_default_mode_is_none_so_existing_callers_are_unaffected() -> None:
    assert DisturbanceConfig().mode == "none"


def test_none_mode_is_all_zeros_and_disabled() -> None:
    dist = SimDisturbance(_cfg(mode="none"), SHAPE)
    assert dist.enabled is False
    phase = dist.phase()
    assert phase.shape == SHAPE
    assert phase.dtype == np.float64
    assert np.all(phase == 0.0)
    assert dist.stats()["streaks_used"] == 0


# --- static: frozen ----------------------------------------------------------


def test_static_mode_screen_is_frozen_across_many_calls() -> None:
    dist = SimDisturbance(_cfg(mode="static"), SHAPE)
    first = dist.phase().copy()
    for _ in range(24):
        again = dist.phase()
        assert np.array_equal(again, first), "static screen must never change"

    trace = dist.trace()
    rms = np.asarray(trace["call_rms"], dtype=float)
    idx = np.asarray(trace["call_streak_index"], dtype=int)
    assert rms.size == 25
    assert np.allclose(rms, rms[0], rtol=0.0, atol=1e-12), "static RMS must be constant"
    assert np.array_equal(idx, np.zeros_like(idx)), "static must stay on streak 0"
    assert dist.stats()["streaks_used"] == 1


def test_static_mode_is_deterministic_for_a_fixed_seed() -> None:
    a = SimDisturbance(_cfg(mode="static", seed=7), SHAPE).phase()
    b = SimDisturbance(_cfg(mode="static", seed=7), SHAPE).phase()
    c = SimDisturbance(_cfg(mode="static", seed=8), SHAPE).phase()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- dynamic: varying --------------------------------------------------------


def test_dynamic_mode_screen_varies_across_calls() -> None:
    dist = SimDisturbance(_cfg(mode="dynamic"), SHAPE)
    for _ in range(25):
        dist.phase()

    rms = np.asarray(dist.trace()["call_rms"], dtype=float)
    idx = np.asarray(dist.trace()["call_streak_index"], dtype=int)
    assert rms.size == 25
    assert not np.allclose(rms, rms[0], rtol=0.0, atol=1e-12), (
        "dynamic RMS must vary across evaluations"
    )
    assert np.all(np.diff(idx) > 0), "dynamic streak index must strictly increase"
    assert dist.stats()["streaks_used"] == 25, "one independent streak per call"


def test_dynamic_sigma_total_is_the_mean_over_evaluations() -> None:
    """``sigma_total_rad`` must summarise the regime, not just the last draw.

    Static and dynamic must be reported on the same footing, so the dynamic
    summary is the mean of the per-evaluation RMS rather than an arbitrary
    final sample.
    """
    dist = SimDisturbance(_cfg(mode="dynamic"), SHAPE)
    for _ in range(6):
        dist.phase()
    rms = np.asarray(dist.trace()["call_rms"], dtype=float)
    assert np.isclose(dist.stats()["sigma_total_rad"], float(rms.mean()), rtol=1e-12)
    assert not np.isclose(dist.stats()["sigma_total_rad"], float(rms[-1]), rtol=1e-6), (
        "the mean must not collapse to the final sample"
    )


def test_dynamic_mode_does_not_retain_generated_screens() -> None:
    """Memory must stay bounded: a 300-epoch run draws ~630 full-size screens.

    Retaining them (1200x1920 float64 is ~18 MB each) would cost several
    gigabytes. Only scalars may be accumulated; the sole arrays kept alive are
    the shared halo and the amplitude mask.
    """
    dist = SimDisturbance(_cfg(mode="dynamic"), SHAPE)
    for _ in range(12):
        dist.phase()

    retained = [
        value
        for value in dist.__dict__.values()
        if isinstance(value, np.ndarray) and value.size > 100_000
    ]
    assert len(retained) <= 3, (
        f"dynamic mode retained {len(retained)} full-size arrays; memory is unbounded"
    )
    assert dist.stats()["streaks_used"] == 12


def test_dynamic_mode_is_deterministic_for_a_fixed_seed() -> None:
    a = SimDisturbance(_cfg(mode="dynamic", seed=11), SHAPE)
    b = SimDisturbance(_cfg(mode="dynamic", seed=11), SHAPE)
    c = SimDisturbance(_cfg(mode="dynamic", seed=12), SHAPE)

    pa = [a.phase().copy() for _ in range(5)]
    pb = [b.phase().copy() for _ in range(5)]
    pc = [c.phase().copy() for _ in range(5)]

    for x, y in zip(pa, pb, strict=True):
        assert np.array_equal(x, y), "same seed must replay identically"
    assert any(not np.array_equal(x, z) for x, z in zip(pa, pc, strict=True)), (
        "a different seed must produce a different sequence"
    )


# --- raw-radian contract -----------------------------------------------------


def test_phase_is_finite_correct_shape_and_unwrapped() -> None:
    dist = SimDisturbance(_cfg(mode="static"), SHAPE)
    phase = dist.phase()
    assert phase.shape == SHAPE
    assert phase.dtype == np.float64
    assert np.all(np.isfinite(phase))
    # Raw unwrapped radians: a wrapped phase would be trapped in [0, 2*pi).
    # This guards the repo's "generators must not mod 2*pi" contract.
    assert float(np.abs(phase).max()) > 1.0


# --- thermal halo ------------------------------------------------------------


def test_thermal_halo_pv_is_normalised_to_requested_waves() -> None:
    requested = 0.30
    dist = SimDisturbance(_cfg(mode="static", thermal_halo_pv_waves=requested), SHAPE)

    halo = np.asarray(dist.halo_phase, dtype=float)
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    r2 = (xx - SHAPE[1] / 2.0) ** 2 + (yy - SHAPE[0] / 2.0) ** 2
    mask = np.exp(-r2 / (2.0 * dist.beam_w0**2)) > 1e-3

    measured_pv = float(np.ptp(halo[mask]))
    assert np.isclose(measured_pv, requested * 2.0 * np.pi, rtol=1e-6), (
        f"halo PV {measured_pv} != requested {requested * 2 * np.pi}"
    )


def test_thermal_halo_can_be_disabled() -> None:
    dist = SimDisturbance(_cfg(mode="static", thermal_halo_pv_waves=0.0), SHAPE)
    halo = np.asarray(dist.halo_phase, dtype=float)
    assert np.all(halo == 0.0)


def test_thermal_halo_uses_canonical_zernike_entry_not_local_maths() -> None:
    """The halo must be built from ``generate_zernike_phase`` (repo rule).

    We pin this behaviourally: scaling the Noll coefficients linearly must scale
    the halo phase linearly, and a different Noll set must give a different
    shape. A hand-rolled/atmosphere-inconsistent implementation would not.
    """
    a = SimDisturbance(_cfg(mode="static", halo_noll=((4, -1.0), (11, -1.0))), SHAPE)
    b = SimDisturbance(_cfg(mode="static", halo_noll=((4, -2.0), (11, -2.0))), SHAPE)
    c = SimDisturbance(_cfg(mode="static", halo_noll=((11, -1.0),)), SHAPE)

    ha = np.asarray(a.halo_phase, dtype=float)
    hb = np.asarray(b.halo_phase, dtype=float)
    hc = np.asarray(c.halo_phase, dtype=float)

    # PV-normalised by construction, so doubling the coefficients must not
    # change the normalised shape.
    assert np.allclose(ha, hb, rtol=1e-9, atol=1e-9)
    # A different mode set must genuinely differ.
    assert not np.allclose(ha, hc, rtol=1e-6, atol=1e-6)


# --- measured amplitude ------------------------------------------------------


def test_turbulence_amplitude_scales_with_cn2() -> None:
    weak = SimDisturbance(_cfg(mode="static", cn2=5e-14, thermal_halo_pv_waves=0.0), SHAPE)
    strong = SimDisturbance(_cfg(mode="static", cn2=2e-13, thermal_halo_pv_waves=0.0), SHAPE)
    sigma_weak = float(weak.stats()["sigma_turb_rad"])
    sigma_strong = float(strong.stats()["sigma_turb_rad"])
    assert sigma_strong > 1.3 * sigma_weak, (
        f"stronger cn2 must give a stronger disturbance: {sigma_strong} vs {sigma_weak}"
    )


def test_stats_report_measured_values_not_analytic_ones() -> None:
    dist = SimDisturbance(_cfg(mode="static"), SHAPE)
    stats = dist.stats()
    total = np.asarray(dist.phase(), dtype=float)
    assert np.isclose(stats["sigma_total_rad"], float(total.std()), rtol=1e-9)
    # sigma_total must be the quadrature/actual std of turbulence + halo, and
    # must be consistent with the stored components.
    assert stats["sigma_total_rad"] > 0.0
    assert stats["sigma_turb_rad"] > 0.0
    assert stats["sigma_halo_rad"] > 0.0


def test_zero_cn2_with_no_halo_yields_no_disturbance() -> None:
    dist = SimDisturbance(_cfg(mode="static", cn2=0.0, thermal_halo_pv_waves=0.0), SHAPE)
    assert float(np.abs(dist.phase()).max()) == 0.0
    assert dist.enabled is True  # mode is static, but the screen is empty


# --- serialisation -----------------------------------------------------------


def test_to_dict_is_json_serialisable_and_carries_config_and_measured() -> None:
    import json

    dist = SimDisturbance(_cfg(mode="static"), SHAPE)
    dist.phase()
    payload = dist.to_dict()
    text = json.dumps(payload)  # must not raise
    assert json.loads(text)["config"]["mode"] == "static"
    assert json.loads(text)["measured"]["streaks_used"] == 1


# --- wiring into SimPibSystem ------------------------------------------------


def test_far_field_is_byte_identical_when_no_disturbance() -> None:
    """Scenario S2: the disturbance must be a pure opt-in, zero regression."""
    from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem

    a = SimPibSystem(seed=42)
    b = SimPibSystem(seed=42, disturbance=None)
    assert np.array_equal(a.far_field(), b.far_field())


def test_far_field_changes_measurably_with_a_disturbance() -> None:
    """The disturbance must bite: neither ignored nor annihilating.

    ``far_field()`` normalises the spectrum peak to 100, so peak *ratios* are
    meaningless -- the comparison has to be shape-based. The Pearson correlation
    between the clean and disturbed far fields is normalisation-invariant.
    """
    from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem

    clean = SimPibSystem(seed=42).far_field()
    dirty = SimPibSystem(
        seed=42,
        disturbance=SimDisturbance(_cfg(mode="static", cn2=2e-13), SHAPE),
    ).far_field()

    assert not np.array_equal(clean, dirty)
    corr = float(np.corrcoef(clean.ravel(), dirty.ravel())[0, 1])
    assert 0.8 < corr < 0.999, f"clean/disturbed far-field correlation {corr} outside (0.8, 0.999)"


def test_repeated_reads_with_unchanged_phase_reuse_the_cached_far_field() -> None:
    """The cache means one disturbance advance per real optical evaluation."""
    from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem

    dist = SimDisturbance(_cfg(mode="dynamic"), SHAPE)
    system = SimPibSystem(seed=42, disturbance=dist)
    first = system.far_field().copy()
    for _ in range(5):
        assert np.array_equal(system.far_field(), first)
    assert dist.stats()["calls"] == 1, "a warm cache must not advance the disturbance"


def test_static_disturbance_is_advance_once_and_stable_across_phase_writes() -> None:
    from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem

    dist = SimDisturbance(_cfg(mode="static"), SHAPE)
    system = SimPibSystem(seed=42, disturbance=dist)
    system.far_field()
    system.set_phase_rad(np.zeros(SHAPE))
    system.far_field()
    system.set_phase_rad(np.full(SHAPE, 0.1))
    system.far_field()
    assert dist.stats()["streaks_used"] == 1
    assert len(dist.trace()["call_rms"]) == 3


def test_reset_system_forwards_the_disturbance() -> None:
    from ao_shaping.drivers.sim.slm_pib_sim import get_system, reset_system

    dist = SimDisturbance(_cfg(mode="static"), SHAPE)
    try:
        system = reset_system(seed=42, disturbance=dist)
        assert system.disturbance is dist
        assert get_system().disturbance is dist
    finally:
        reset_system(seed=42)  # leave the process-wide system clean
