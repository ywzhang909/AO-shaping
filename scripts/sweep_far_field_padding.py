"""`far_field_padding` is calibrated for ONE family; measure the others.

`ZernikeAmpConfig.far_field_padding` defaults to 10 and its docstring records a
sweep **for `slm_zernike_shaping` only**:

    padding  1     4     8    10    12    16    20
    R^2    -0.20 -0.12 +0.33 +0.46 +0.53 +0.49 +0.18

with the explicit warning "the right value depends on the family's `fov_px`, so
re-run the sweep for a new family rather than copying the default". The corpus has
**7 families**; the default is unverified for 6 of them.

Protocol (identical to the documented one): coefficients left at Z = 0, so this
measures *geometric* agreement between the predicted and measured angular extents
and nothing else -- no training, no fitting. `normalization="peak"` (the only
validated choice; `sum` is the known trap). R^2 is the judge, with SSIM as a
cross-check because a single number is what got this repo into trouble before.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\src")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.hwdataset import MaterialiserConfig, build_hw_index  # noqa: E402
from ml.hwdataset.records import Materialiser  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import evaluate  # noqa: E402

GRID = 64
N_MAX = 15
PADDINGS = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20)
SAMPLES = 48
DEVICE = torch.device("cpu")

index = build_hw_index(index_cache="data/hw_index_cache.json")
families: dict[str, list] = {}
for ref in index.records:
    families.setdefault(ref.family, []).append(ref)
# Drop the 12-record sim_calib_abba family: too small for any conclusion.
families = {k: v for k, v in sorted(families.items()) if len(v) >= 50}

results: dict[str, dict] = {}
for family, refs in families.items():
    subset = index.filter(families=[family])
    positions = list(range(len(subset.records)))[:SAMPLES]
    mat = Materialiser(config=MaterialiserConfig(grid=GRID), use_cache=True)
    samples = [mat.materialise(subset.records[i]) for i in positions]
    fov = sorted({s.fov_px for s in samples})
    tensors = {
        "phase_cos": torch.as_tensor(
            np.asarray([np.asarray(s.phase_cos, dtype=np.float32) for s in samples])
        )[:, None],
        "phase_sin": torch.as_tensor(
            np.asarray([np.asarray(s.phase_sin, dtype=np.float32) for s in samples])
        )[:, None],
        "target": torch.as_tensor(
            np.asarray([np.asarray(s.image, dtype=np.float32) for s in samples])
        )[:, None],
    }

    row: dict[str, dict] = {}
    for pad in PADDINGS:
        model = ZernikeAmpModel(
            ZernikeAmpConfig(
                n_max=N_MAX, grid=GRID, far_field_padding=pad,
                normalization="peak", center_crop=True,
            )
        )
        out = evaluate(model, tensors, beam_samples=16)
        row[str(pad)] = {
            "r2": float(out["r2"]),
            "ssim": float(out["ssim"]),
            "psnr": float(out["psnr"]),
        }
    best = max(row, key=lambda p: row[p]["r2"])
    results[family] = {
        "records": len(subset.records),
        "sampled": len(positions),
        "fov_px": fov,
        "by_padding": row,
        "best_padding": int(best),
        "r2_at_10": row["10"]["r2"],
        "gain_vs_10": row[best]["r2"] - row["10"]["r2"],
    }
    print(
        f"{family:<26} n={len(subset.records):>5} fov={fov} "
        f"best_pad={best:>3} R2={row[best]['r2']:+.4f} "
        f"(pad10 {row['10']['r2']:+.4f}, gain {row[best]['r2'] - row['10']['r2']:+.4f})",
        flush=True,
    )

Path("report/loss_defects").mkdir(parents=True, exist_ok=True)
Path("report/loss_defects/padding_sweep.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

print("\n" + "=" * 96)
print("R2 at Z=0 per family x padding  (the model default is 10)")
print("=" * 96)
print(f"{'family':<26}" + "".join(f"{p:>8}" for p in PADDINGS) + f"{'best':>7}")
for family, data in results.items():
    cells = "".join(
        f"{data['by_padding'][str(p)]['r2']:>8.2f}" for p in PADDINGS
    )
    print(f"{family:<26}{cells}{data['best_padding']:>7}")

print("\nSSIM cross-check at each family's best padding vs padding 10:")
for family, data in results.items():
    b = str(data["best_padding"])
    print(
        f"  {family:<26} best {b:>3}: SSIM {data['by_padding'][b]['ssim']:.4f}"
        f"   pad10: SSIM {data['by_padding']['10']['ssim']:.4f}"
    )
print("\nwrote report/loss_defects/padding_sweep.json")