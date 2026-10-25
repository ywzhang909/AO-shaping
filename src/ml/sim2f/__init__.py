"""Differentiable 2f beam-shaping twin, its teacher, and the amortising U-Net.

What this package is
====================
A **simulation-only** bench for the inverse shaping problem, built around three
ideas that the hardware pipeline settles on and that a synthetic bench can
therefore exercise cheaply:

.. code-block:: text

    target spot  --U-Net-->  phi_shape          (learned on the ABERRATION-FREE bench)
                                |
                                v
    phi_cmd = phi_shape - sum_j c_hat_j * Z_j   (c_hat from a Zernike probe stage)
                                |
                          Sim2f (2f + aberrations + CCD pixel integration + noise)
                                |
                          measured CCD frame
                                |
    c_hat <- argmin_c || I_model(phi_cmd, c) - I_measured ||   (closed-loop refit)

The division of labour is the point.  The U-Net is asked only for the
**aberration-free** direct phase, which is a well-posed many-to-one regression
(target -> phase, modulo a global phase) that does not depend on the bench
state.  The bench state is handled by the *same* estimator the hardware
pipeline uses -- a Zernike coefficient fit against measured far fields -- first
from a few known probes, then against each shaping measurement to absorb what
the probes missed.  A network asked to infer aberrations from a single flat
spot would be solving a strictly harder and worse-conditioned problem, and it
would have to be retrained for every new optical path.

Modules
=======
``optics``     the differentiable forward model -- the **only** FFT in this package.
``targets``    target-shape families on the camera grid (numpy; no torch).
``teacher``    differentiable phase optimisation on the ideal bench (the labels).
``synth``      draws the supervised corpus, runs the teacher, writes the cache.
``dataset``    reads that cache as mini-batches.
``model``      the U-Net (target -> phasor).
``losses``     piston-free phasor loss, physics loss, shaping metrics.
``fit``        Zernike-coefficient estimation (probe stage and closed-loop refit).
``loop``       closed-loop evaluation driver and its ablations.
``train`` / ``evaluate``  CLI entry points.

Reference environment
=====================
The forward model is **not invented here**.  ``ao_shaping.drivers.sim.bench_twin``
is the calibrated digital twin behind ``report/shape_trials/`` (fitted parameters:
``report/shape_trials/sim_vs_hw/artifacts/sim_twin_params_20261023.json``), and
:class:`~ml.sim2f.config.Sim2fConfig` mirrors its geometry field-for-field
(``region``/``region_span_px``/``panel_pixel_um``/``far_field_padding``/
``beam_w0_px``/``ccd_pixel_um``) plus its error model
(:class:`~ml.sim2f.config.OpticsErrors` ~ ``TwinErrorModel`` + the coherent
0-order).  ``bench_twin`` is numpy and cannot be differentiated through, so this
package is the torch re-implementation; ``tests/.../test_twin_parity.py`` pins
the two together frame-for-frame.  If they disagree, this package is wrong.

Consequences that are easy to get wrong: the entrance is a **soft Gaussian**
with no hard aperture; ``far_field_padding`` sets the angular scale (4, not the
twin constructor's 10); the camera pixel is *finer* than the FFT pitch
(``sampling_ratio = 2.10``), so camera pixels integrate/interpolate; and the
0-order is a **coherent pupil-domain** term
``U = U_in * (eta + (1 - eta) * exp(i*phi))``, so it does not conserve the
modulated energy.

Rules that are not stylistic
============================
1. **One FFT.**  ``optics.Sim2f`` is the only implementation of the 2f
   propagation and of camera-pixel integration in this package.  The teacher,
   the physics loss, the aberration fit and the loop must all call it.  This
   repository has been bitten repeatedly by two propagators drifting apart
   (see ``report/zernike_phase2amp`` and the ``AGENTS.md`` Zernike rules), and a
   second FFT would make the teacher's labels inconsistent with what the loop
   measures.
2. **One Zernike basis.**  :meth:`ml.sim2f.optics.Sim2f.zernike_basis` builds it
   through the canonical ``ao_shaping.utils.wavefront.zernike_calc``; nothing
   here writes its own Noll table or its own radial polynomial.
3. **Raw radians.**  Every phase in this package is unwrapped radians.  The only
   ``mod 2*pi`` in this repository lives in the SLM driver.
4. **Nothing wraps or re-normalises a phase.**  Min-max normalisation of a phase
   is a known anti-pattern here (it is scale-invariant, so the coefficient
   amplitude becomes uncontrollable).
5. **Torch is optional.**  ``import ml.sim2f`` must not import torch; the public
   names resolve lazily so the config, the target library and the cache metadata
   stay importable without the ``ml`` dependency group.

Bench truth, stated up front
============================
This package makes **no claim about the real optical path**.  The teacher, the
network and the loop all share one forward model, so the numbers it produces are
an *amortisation* measurement (how much of an iterative optimum a single
forward pass recovers) plus a *closed-loop* measurement (how much a Zernike fit
recovers from camera frames).  They are not a sim-to-real transfer result, and
the report says so in every table.
"""

from __future__ import annotations

__all__ = [
    "AberrationConfig",
    "LoopConfig",
    "OpticsErrors",
    "Sim2fConfig",
    "SynthConfig",
    "TeacherConfig",
    "TrainConfig",
]

#: Names that need torch / numpy and are therefore resolved on first use.
_LAZY = {
    "Sim2f": "ml.sim2f.optics",
    "Teacher": "ml.sim2f.teacher",
    "Sim2fDataset": "ml.sim2f.dataset",
    "Sim2fUNet": "ml.sim2f.model",
}


def __getattr__(name: str) -> object:
    """Resolve the heavy public API lazily.

    Args:
        name: Attribute being looked up on this package.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If ``name`` is not part of this package's public API.
    """
    if name in __all__:
        from ml.sim2f import config

        return getattr(config, name)
    target = _LAZY.get(name)
    if target is not None:
        import importlib

        return getattr(importlib.import_module(target), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily resolved names in ``dir()``."""
    return sorted([*__all__, *_LAZY])
