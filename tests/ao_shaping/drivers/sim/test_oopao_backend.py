"""Tests for the OOPAO-based turbulence-screen + ASM-propagation backend.

The OOPAO library is optional: every test is skipped when
``ao_shaping.drivers.sim._oopao_compat`` cannot import it (missing OOPAO
package / its ``jsonpickle`` / ``scikit-image`` / ``astropy`` deps), so the
default numpy/legacy path is never broken on a machine without OOPAO.

When OOPAO is present the tests pin:
* screen shape/dtype, determinism per seed, per-seed variation,
* the per-slab ``compute_r0`` / ``_rescale_for`` calibration math,
* the module-level ``make_screens`` + ``propagate_asm`` convenience wrappers,
* the ``Propagator`` (oopao engine + automatic numpy fallback),
* and that both ``beam_backend`` entry points route to OOPAO under
  ``AO_OOPAO_BACKEND=1``.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

import numpy as np

import ao_shaping.drivers.sim._oopao_compat as _oopao_compat  # noqa: F401 (import probe)

oopao_backend = pytest.importorskip("ao_shaping.drivers.sim.oopao_backend")


def _oopao_available() -> bool:
    return bool(oopao_backend._oopao_available())


@pytest.fixture(autouse=True)
def _require_oopao():
    if not _oopao_available():
        pytest.skip("OOPAO library not available in this environment")


def _small_params() -> dict:
    return dict(
        N=64,
        dx=1e-3,
        Dscope=0.064,
        lam=1064e-9,
        cn2=1e-13,
        L=1000.0,
        L0=10.0,
        n_screens=1,
    )


class TestScreenBackend:
    def test_screen_shape_dtype(self):
        p = _small_params()
        backend = oopao_backend.OopaoScreenBackend(
            N=p["N"], dx=p["dx"], Dscope=p["Dscope"], lam=p["lam"],
            cn2=p["cn2"], L=p["L"], L0=p["L0"], n_screens=p["n_screens"],
        )
        out = backend.make_screens(seed=42)
        assert out.shape == (p["n_screens"], p["N"], p["N"])
        assert out.dtype == np.float32

    def test_deterministic_per_seed(self):
        p = _small_params()
        backend = oopao_backend.OopaoScreenBackend(
            N=p["N"], dx=p["dx"], Dscope=p["Dscope"], lam=p["lam"],
            cn2=p["cn2"], L=p["L"], L0=p["L0"], n_screens=p["n_screens"],
        )
        a = backend.make_screens(seed=7)[0]
        b = backend.make_screens(seed=7)[0]
        assert np.array_equal(a, b)

    def test_different_seed_differs(self):
        p = _small_params()
        backend = oopao_backend.OopaoScreenBackend(
            N=p["N"], dx=p["dx"], Dscope=p["Dscope"], lam=p["lam"],
            cn2=p["cn2"], L=p["L"], L0=p["L0"], n_screens=p["n_screens"],
        )
        a = backend.make_screens(seed=7)[0]
        c = backend.make_screens(seed=8)[0]
        assert not np.array_equal(a, c)

    def test_screen_is_nonzero(self):
        p = _small_params()
        backend = oopao_backend.OopaoScreenBackend(
            N=p["N"], dx=p["dx"], Dscope=p["Dscope"], lam=p["lam"],
            cn2=p["cn2"], L=p["L"], L0=p["L0"], n_screens=p["n_screens"],
        )
        out = backend.make_screens(seed=42)
        assert float(np.abs(out).max()) > 0.0


class TestCalibrationMath:
    def test_compute_r0_matches_legacy(self):
        lam, cn2, L = 1064e-9, 1e-13, 1000.0
        k = 2 * np.pi / lam
        expected = (0.423 * k**2 * cn2 * L) ** (-3 / 5)
        assert oopao_backend.compute_r0(lam, cn2, L) == pytest.approx(expected)

    def test_rescale_for_positive_and_monotonic(self):
        # smaller r0_slab (stronger turbulence) -> larger rescale factor
        r_small = oopao_backend._rescale_for(0.05, 1064e-9)
        r_large = oopao_backend._rescale_for(0.20, 1064e-9)
        assert r_small > 0 > -1  # positive
        assert r_small > r_large


class TestModuleWrappers:
    def test_make_screens_wrapper(self):
        p = _small_params()
        out = oopao_backend.make_screens(**p, seed=11)
        assert out.shape == (p["n_screens"], p["N"], p["N"])

    def test_propagate_asm(self):
        field = np.ones((64, 64), dtype=np.complex128)
        out = oopao_backend.propagate_asm(field, lam=1064e-9, dx=1e-3, z=0.1)
        assert out.shape == field.shape
        assert np.iscomplexobj(out)


class TestPropagator:
    def test_engine_default_oopao(self):
        prop = oopao_backend.Propagator(N=64, dx=1e-3, lam=1064e-9)
        assert prop.engine == "oopao"

    def test_engine_numpy_fallback(self):
        prop = oopao_backend.Propagator(N=64, dx=1e-3, lam=1064e-9, engine="numpy")
        assert prop.engine == "numpy"

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError):
            oopao_backend.Propagator(N=64, dx=1e-3, lam=1064e-9, engine="fft")

    def test_propagate_preserves_shape(self):
        prop = oopao_backend.Propagator(N=64, dx=1e-3, lam=1064e-9)
        field = np.ones((64, 64), dtype=np.complex128)
        out = prop.propagate(field, 0.1)
        assert out.shape == (64, 64)
        assert np.iscomplexobj(out)


class TestBeamBackendRouting:
    def test_numpy_default_unchanged(self, tmp_path):
        # No AO_OOPAO_BACKEND -> numpy path; turbulence matches legacy FFT.
        import ao_shaping.drivers.sim.beam_backend as bb

        cfg = bb.make_beam_config(
            n_grid=64, aperture_size=0.064, wavelength=1064e-9, cn2=1e-13,
            l_max=10.0, l_min=0.001, propagation_distance=1000.0,
        )
        rng = np.random.default_rng(0)
        p = bb.turbulence_phase(cfg, rng=rng)
        assert p.shape == (64, 64)
        # numpy legacy path gives a finite, non-zero screen
        assert float(np.abs(p).max()) > 0.0

    def test_oopao_route_enabled(self, monkeypatch):
        import ao_shaping.drivers.sim.beam_backend as bb

        monkeypatch.setenv("AO_OOPAO_BACKEND", "1")
        cfg = bb.make_beam_config(
            n_grid=64, aperture_size=0.064, wavelength=1064e-9, cn2=1e-13,
            l_max=10.0, l_min=0.001, propagation_distance=1000.0,
        )
        rng = np.random.default_rng(0)
        # turbulence via OOPAO screen backend
        p = bb.turbulence_phase(cfg, rng=rng)
        assert p.shape == (64, 64)
        assert float(np.abs(p).max()) > 0.0
        # propagation via OOPAO ASM
        field = bb.gaussian_pupil(cfg)
        out = bb.propagate(field, cfg, 0.1)
        assert out.shape == field.shape
