"""Compare global-model and measured-anchor directional objectives on saved phases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ao_shaping.tools.slm.transfer.capture import clock
from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective


def analyze(run, out):
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, load_frozen_parameters,
    )
    torch.set_num_threads(4)
    config = json.loads((run / "config.json").read_text())
    records = json.loads((run / "manifest.json").read_text())
    from .artifacts import resolve_recorded_path
    params = load_frozen_parameters(resolve_recorded_path(config["checkpoint"], relative_to=run))
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for shape in config["shapes"]:
        a = next(r for r in records if r["label"] == f"{shape}_0_A1")
        b = next(r for r in records if r["label"] == f"{shape}_0_B1")
        minus = next(r for r in records if r["label"] == f"{shape}_direction_minus")
        phases = []
        for row in (a, b, minus):
            with np.load(run / row["file"]) as data:
                phases.append(data["model_phase"])
                if row is a:
                    xy, target = data["camera_xy"], data["target"]
                    image = data["raw_frames"].mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)] / row["exposure_ms"]
        forward = FrozenDefocusForward(params, xy, device="cpu")
        objective = ShapeObjective(target)
        zero = torch.zeros_like(torch.tensor(phases[0]))
        with torch.no_grad():
            baseline, plus, negative = [forward.predict(phase)[0] for phase in phases]
        measured = torch.as_tensor(image, dtype=torch.float32)

        def anchored_loss(phase, _forward=forward, _objective=objective, _measured=measured,
                          _baseline=baseline, _zero=zero):
            prediction = _forward.predict(phase)[0]
            return _objective.evaluate((_measured+prediction-_baseline).clamp_min(1e-8), _zero)[0]

        phase = torch.tensor(phases[0], requires_grad=True)
        direction = torch.tensor(phases[1]-phases[0])
        loss = anchored_loss(phase)
        gradient = torch.autograd.grad(loss, phase)[0]
        automatic = float((gradient*direction).sum())
        differences = []
        with torch.no_grad():
            lp = float(objective.evaluate((measured+plus-baseline).clamp_min(1e-8), zero)[0])
            lm = float(objective.evaluate((measured+negative-baseline).clamp_min(1e-8), zero)[0])
            for h in (.1, .03, .01):
                fd = float((anchored_loss(phase+h*direction)-anchored_loss(phase-h*direction))/(2*h))
                differences.append(dict(h=h, finite_difference=fd, automatic=automatic,
                                        relative_error=abs(fd-automatic)/max(abs(fd), abs(automatic), 1e-9)))
        row = dict(shape=shape, anchored_model_loss_plus=lp, anchored_model_loss_minus=lm,
                   measured_A1_loss=a["loss"], derivative_checks=differences, **clock())
        results.append(row)
        np.savez_compressed(out/f"{shape}.npz", model_A=baseline.numpy(), model_plus=plus.numpy(),
                            model_minus=negative.numpy(), measured_A=image, phase_increment=direction.numpy())
        print(json.dumps(row), flush=True)
    (out/"result.json").write_text(json.dumps(dict(run=str(run), results=results), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)


if __name__ == "__main__":
    main()
