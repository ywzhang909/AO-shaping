"""一次性微基准: 冻结物理预测模型的单次 predict / backward 成本 (本机, 真实 checkpoint)。

不是产品代码: 它只回答"物理预测模型在整段墙钟里占多少", 结果抄进 issue #213 的判定评论。
跑法::

    python scripts/slm/_bench_frozen_forward.py --model-dir <...>/model640 --sample <capture npz>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _time(fn, reps: int = 20) -> dict:
    fn()  # warm-up: the first call pays CUDA context / kernel selection
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return dict(median_ms=round(float(np.median(times)), 3),
                min_ms=round(float(min(times)), 3), reps=reps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path,
                        help="任一 capture_*.npz (取 camera_xy / target)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer,
        FrozenDefocusForward,
        UniformityObjective,
        load_frozen_parameters,
    )

    params = load_frozen_parameters(args.model_dir / "frozen_parameters.json")
    with np.load(args.sample) as saved:
        xy = saved["camera_xy"]
        target = saved["target"]
    forward = FrozenDefocusForward(params, xy, device=args.device)
    forward.exposure_ms = 50.0
    phase = np.zeros((params.grid_size, params.grid_size), dtype=np.float64)
    # Timing only: the masks set which pixels the loss reads, not which ops run,
    # so a full-box mask is representative and avoids re-deriving the core.
    mask = np.asarray(target, dtype=np.float64) > 0.5
    objective = UniformityObjective(mask, mask)
    optimizer = DifferentiableUniformityOptimizer(forward, objective, phase, lr=0.02)

    def one_update():
        optimizer.update()

    report = dict(
        grid_size=int(params.grid_size),
        camera_grid=[int(xy.shape[0]), int(xy.shape[1])],
        predict=_time(lambda: forward.predict(phase)),
        predict_to_numpy=_time(
            lambda: forward.predict(phase)[0].detach().cpu().numpy()),
        update=_time(one_update, reps=10),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
