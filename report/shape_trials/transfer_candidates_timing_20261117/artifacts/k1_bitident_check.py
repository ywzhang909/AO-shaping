"""K=1 逐位一致性：HEAD 版 vs 当前工作树版（同一输入、同一步数）。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"D:/Projects/TIFO/AO-shaping")
HEAD_COPY = Path(sys.argv[1])

sys.path.insert(0, str(REPO / "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


head = load("ddu_head", HEAD_COPY)
cur = load("ddu_cur", REPO / "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py")


class AnalyticForward:
    class params:
        grid_size = 8

    class _device:
        type = "cpu"

    def illumination_weights(self):
        w = np.ones((8, 8), dtype=np.float32)
        w[:2, :] = 0.5
        return w / w.sum()

    def predict(self, phase, **kwargs):
        import torch

        t = phase if torch.is_tensor(phase) else torch.as_tensor(phase, dtype=torch.float32)
        return 1.0 + 0.3 * t + 0.1 * torch.cos(t)

    def total_phase_grid(self, phase, **kwargs):
        return phase


def run(module, *, anchored):
    box = np.zeros((8, 8), dtype=bool)
    box[1:7, 1:7] = True
    core = np.zeros((8, 8), dtype=bool)
    core[3:5, 3:5] = True
    objective = module.UniformityObjective(box, core, w_phase=0.5)
    rng = np.random.default_rng(11)
    seed = (0.05 * rng.standard_normal((8, 8))).astype(np.float32)
    measured = (1.0 + 0.05 * rng.standard_normal((8, 8))).astype(np.float32)
    opt = module.DifferentiableUniformityOptimizer(
        AnalyticForward(), objective, seed, lr=0.05, step_rms_cap=0.05
    )
    for _ in range(5):
        opt.update(measured if anchored else None)
    return opt.loss_history, opt.current_phase, opt.delta_phase, opt.accepted_phase


def physical(module):
    offsets = np.arange(-12, 13, 2.0)
    xx, yy = np.meshgrid(1360.0 + offsets, 1020.0 + offsets, indexing="xy")
    fwd = module.FrozenDefocusForward(
        module.FrozenDefocusParameters(
            defocus_diopter=-0.1, origin_xy=(1360.0, 1020.0),
            amplitude_center_xy=(960.0, 600.0), amplitude_width_xy=(400.0, 300.0),
            residual_noll_radians={5: 0.3}, gain=25.0, background=1.0, sigma_ccd_px=2.0,
            ramp_matrix=((8000.0, 0.0), (0.0, -8000.0)),
            fft_size=64, pixel_samples=1, grid_size=32,
        ),
        np.stack([xx, yy], axis=-1),
        device="cpu",
    )
    box = np.zeros((13, 13), dtype=bool)
    box[2:11, 2:11] = True
    core = np.zeros((13, 13), dtype=bool)
    core[4:9, 4:9] = True
    objective = module.UniformityObjective(box, core, w_phase=0.01)
    rng = np.random.default_rng(5)
    seed = (0.1 * rng.standard_normal((32, 32))).astype(np.float32)
    opt = module.DifferentiableUniformityOptimizer(
        fwd, objective, seed, lr=0.01, step_rms_cap=0.1
    )
    for _ in range(3):
        opt.update()
    return opt.loss_history, opt.current_phase, opt.last_candidate_prediction


def compare(label, a, b):
    ok = True
    history_a, history_b = a[0], b[0]
    if len(history_a) != len(history_b):
        print(f"{label}: length {len(history_a)} != {len(history_b)}")
        return False
    for i, (ra, rb) in enumerate(zip(history_a, history_b)):
        if ra != rb:
            ok = False
            print(f"{label}: history[{i}] differs: {set(ra.items()) ^ set(rb.items())}")
            for key in ra:
                if ra[key] != rb.get(key):
                    print(f"    {key}: {ra[key]!r} vs {rb.get(key)!r}")
    for name, xa, xb in zip(("current", "delta", "accepted", "candidate_pred")[0::1], a[1:], b[1:]):
        if xa is None or xb is None:
            same = xa is None and xb is None
        else:
            same = np.array_equal(np.asarray(xa), np.asarray(xb))
        ok = ok and same
        print(f"{label}: {name} bit-identical = {same}")
    return ok


results = []
results.append(compare("analytic-model", run(head, anchored=False), run(cur, anchored=False)))
results.append(compare("analytic-anchored", run(head, anchored=True), run(cur, anchored=True)))
results.append(compare("physical-32grid", physical(head), physical(cur)))
print("ALL IDENTICAL" if all(results) else "MISMATCH")
sys.exit(0 if all(results) else 1)
