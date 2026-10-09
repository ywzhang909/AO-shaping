"""Are the recorded coefficients actually the ones that produced each frame?

`diagnose_snr_ceiling.py` measured a signal-to-noise ratio of about 1.2:1 and a
signal-vs-separation curve that is nearly flat and non-monotone. Two explanations
fit that, and they have very different fixes:

**(A) noise floor** -- the labels are right, the bench is simply noisy.
**(B) label/frame misalignment** -- `_c` is not the coefficient vector that was on
the SLM when that frame was captured (AGENTS.md already records one instance of
this class of bug: `_c` storing the *active mode* vector, judged
``odd_coefficient_length``).

A closed-form ridge and a k-NN both fail under (B) exactly as they do under (A),
so the SNR reading cannot distinguish them. This script separates them using
*within-pickle* structure, which needs no model at all:

``smooth_c``
    Consecutive epochs of one optimisation run should have near-identical
    coefficients. Measured as ``|c_t - c_{t+1}|`` against ``|c_t - c_random|`` in
    the same pickle. Large separation => `_c` is a real trajectory.

``smooth_I``
    If the labels are right, *near-identical coefficients must also give
    near-identical images*. Measured as ``|I_t - I_{t+1}|`` against
    ``|I_t - I_random|``. **If the images are NOT smoother for adjacent epochs
    while the coefficients ARE, the frame carries variance no coefficient
    explains** -- i.e. (A) noise, and the coefficient/image pair is nonetheless
    consistent.

``residual_vs_dc``
    Regress ``|I_i - I_j|`` on ``|c_i - c_j|`` over all within-pickle pairs. A
    positive, significant slope means the pair is coupled (labels aligned); a flat
    curve means it is not.

Usage
-----
    python scripts/diagnose_label_alignment.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from ml.hwdataset import MaterialiserConfig, build_hw_index
from ml.hwdataset.zernike_dataset import ZERNIKE_COEFF_SOURCES, ZernikeCoeffDataset
from ml.zernike.eval_stats import (
    cohens_dz,
    min_attainable_pvalue,
    sign_flip_pvalue,
)

FAMILY = "slm_zernike_shaping"


def load(dataset, positions):
    coeffs, images = [], []
    for p in positions:
        s = dataset[p]
        coeffs.append(s["coeffs"].reshape(-1).numpy())
        images.append(s["image"].reshape(-1).numpy())
    return np.stack(coeffs).astype(np.float64), np.stack(images).astype(np.float64)


def main() -> int:
    started = time.perf_counter()
    index = build_hw_index(index_cache="data/hw_index_cache.json").filter(
        families=[FAMILY])
    records = list(index.records)
    dataset = ZernikeCoeffDataset(
        index, config=MaterialiserConfig(grid=64, image_mode="peak"), use_cache=True)

    keys = sorted({str(r.path) for r in records})
    per_pickle = []
    pair_dc: list[float] = []
    pair_di: list[float] = []
    for key in keys:
        pos = [i for i, r in enumerate(records) if str(r.path) == key]
        c, im = load(dataset, pos)
        # Coefficient distance in units of the pickle's own spread, so a single
        # global scale (dominated by the highest-order modes) cannot hide the effect.
        mu, sd = c.mean(0, keepdims=True), c.std(0, keepdims=True)
        sd[sd <= 0] = 1.0
        z = (c - mu) / sd
        n = len(pos)
        dmat = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
        imat = np.sqrt(((im[:, None, :] - im[None, :, :]) ** 2).mean(axis=2))

        adj = np.array([dmat[t, t + 1] for t in range(n - 1)])
        adj_i = np.array([imat[t, t + 1] for t in range(n - 1)])
        off = ~np.eye(n, dtype=bool)
        rand_c = dmat[off]
        rand_i = imat[off]
        per_pickle.append({
            "pickle": Path(key).name,
            "n": n,
            "adj_dc_mean": float(adj.mean()),
            "rand_dc_mean": float(rand_c.mean()),
            "adj_di_mean": float(adj_i.mean()),
            "rand_di_mean": float(rand_i.mean()),
        })
        iu = np.triu_indices(n, 1)
        pair_dc.extend(dmat[iu].tolist())
        pair_di.extend(imat[iu].tolist())

        print("%-52s adj|dc|=%.3f rand|dc|=%.3f | adj|dI|=%.5f rand|dI|=%.5f"
              % (Path(key).name[:52], adj.mean(), rand_c.mean(),
                 adj_i.mean(), rand_i.mean()), flush=True)

    print("\n=== smoothness: adjacent epochs vs all pairs, same pickle ===")
    ratio_c = [p["adj_dc_mean"] / max(p["rand_dc_mean"], 1e-12) for p in per_pickle]
    ratio_i = [p["adj_di_mean"] / max(p["rand_di_mean"], 1e-12) for p in per_pickle]
    print("  coefficients: adjacent/random distance ratio, mean = %.4f"
          % float(np.mean(ratio_c)))
    print("  images      : adjacent/random difference ratio, mean = %.4f"
          % float(np.mean(ratio_i)))
    diffs = [i - c for i, c in zip(ratio_i, ratio_c)]
    print("  paired (image ratio - coefficient ratio) over %d pickles: mean=%+.4f p=%.4f "
          "(min %.4f) dz=%+.2f"
          % (len(diffs), float(np.mean(diffs)), sign_flip_pvalue(diffs),
             min_attainable_pvalue(len(diffs)), cohens_dz(diffs)))

    # Regression of image difference on coefficient distance, per pickle so the
    # intercept (the noise floor) and the slope (the coupling) are both visible.
    print("\n=== |dI| regressed on |dc| (within-pickle) ===")
    slopes = []
    for p in per_pickle:
        key = p["pickle"]
        pos = [i for i, r in enumerate(records) if Path(str(r.path)).name == key]
        c, im = load(dataset, pos)
        mu, sd = c.mean(0, keepdims=True), c.std(0, keepdims=True)
        sd[sd <= 0] = 1.0
        z = (c - mu) / sd
        n = len(pos)
        iu = np.triu_indices(n, 1)
        dc = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))[iu]
        di = np.sqrt(((im[:, None, :] - im[None, :, :]) ** 2).mean(axis=2))[iu]
        slope = float(np.polyfit(dc, di, 1)[0])
        slopes.append(slope)
        print("  %-52s slope=%+.6f  intercept=%.6f" % (key[:52], slope, di.mean()))
    print("  mean slope = %+.6f ; positive in %d/%d pickles"
          % (float(np.mean(slopes)), sum(1 for s in slopes if s > 0), len(slopes)))

    verdict = (
        "labels plausibly aligned; image carries unexplained variance (noise floor)"
        if float(np.mean(slopes)) > 0
        else "labels look MISALIGNED: image difference does not increase with "
             "coefficient distance at all"
    )
    out = {"per_pickle": per_pickle,
           "ratio_c_mean": float(np.mean(ratio_c)),
           "ratio_i_mean": float(np.mean(ratio_i)),
           "paired_ratio_diff": float(np.mean(diffs)),
           "paired_p": sign_flip_pvalue(diffs),
           "slopes": slopes, "slope_mean": float(np.mean(slopes)),
           "verdict": verdict, "seconds": round(time.perf_counter() - started, 1)}
    Path("logs/label_alignment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nVERDICT:", verdict)
    print("wrote logs/label_alignment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
