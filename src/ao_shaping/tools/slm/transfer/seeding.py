"""Transfer accepted phase controls to newly measured optical geometry; no GS."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def accepted_seed(directory, shape, forward, xy, target, *, policy="strict", centre=None):
    """Return an accepted raw phase and its reproducible transfer description."""
    from .artifacts import resolve_recorded_path
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, load_frozen_parameters,
    )
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    records = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    row = next(r for r in records if r["label"] == shape + "_final")
    with np.load(directory / row["file"], allow_pickle=False) as saved:
        phase = saved["model_phase"].copy()
        if phase.shape != (forward.params.grid_size,)*2 or not np.isfinite(phase).all():
            raise ValueError("accepted phase grid differs from current model")
        if policy == "strict":
            if not np.array_equal(saved["camera_xy"], xy) or not np.array_equal(saved["target"], target):
                raise ValueError("replay grid/target differs from current fixed geometry")
            return phase, dict(policy=policy, source=str(directory.resolve()), label=row["label"], gs_calls=0)
    if policy != "transfer" or centre is None:
        raise ValueError("transfer policy requires current target centre")
    old_checkpoint = resolve_recorded_path(config["checkpoint"], relative_to=directory)
    old = load_frozen_parameters(old_checkpoint)
    previous = FrozenDefocusForward(old, xy, device="cpu")
    if old.grid_size != forward.params.grid_size:
        raise ValueError("transfer cannot resample accepted phase controls")
    old_frequency = np.linalg.solve(np.asarray(old.ramp_matrix), np.asarray(config["target_xy"])-old.origin_xy)
    new_frequency = np.linalg.solve(np.asarray(forward.params.ramp_matrix), np.asarray(centre)-forward.params.origin_xy)
    difference = new_frequency-old_frequency

    def static(model):
        # ``model`` may be the caller's CUDA forward: read it out via cpu().
        result = model.params.defocus_diopter*model.params.diopter_scale*model.z4.detach().cpu().numpy()
        if model.residual_basis is not None:
            result = result + np.einsum("i,ijk->jk", model.residual_values.detach().cpu().numpy(),
                                        model.residual_basis.detach().cpu().numpy())
        return result

    correction = static(previous)-static(forward)
    yy, xx = np.mgrid[:phase.shape[0], :phase.shape[1]]*forward.pitch
    phase = phase + correction + 2*np.pi*(difference[0]*xx+difference[1]*yy)
    return phase, dict(policy=policy, source=str(directory.resolve()), label=row["label"],
                       old_checkpoint=str(old_checkpoint), carrier_delta_cycles_per_panel_pixel=difference.tolist(),
                       static_correction_rms_rad=float(np.sqrt(np.mean(correction**2))), gs_calls=0)
