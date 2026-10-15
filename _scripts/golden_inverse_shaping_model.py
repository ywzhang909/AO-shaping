"""Golden harness for generate_inverse_shaping_model_report.build_report.

Run BEFORE and AFTER the refactor and diff the two outputs (floors=yes and
floors=no arms). Uses the real PhasorUNet so rows match production.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `scripts._common` / sibling scripts import
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

import matplotlib

matplotlib.use("Agg")

from ml.shaping.config import ShapingConfig  # noqa: E402
from ml.shaping.models import PhasorUNet  # noqa: E402

import generate_inverse_shaping_model_report as mod  # noqa: E402


def main() -> None:
    cfg = ShapingConfig()
    model = PhasorUNet(in_channels=cfg.in_channels, features=list(cfg.features), grid=cfg.grid)
    rows = mod._stages(model, cfg)
    raw_shape, unit_shape = mod._verify_forward(model, cfg)
    meta = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "total_ablation": sum(p.numel() for p in model.parameters()) - 34,
        "raw_shape": raw_shape,
        "unit_shape": unit_shape,
        "n_modes": 14,
    }
    floors = [
        {
            "family": "slm_pib",
            "records": 12732,
            "n": 30,
            "mean_norm": 0.937,
            "floor": 0.063,
            "frac": 6.7,
        },
        {
            "family": "slm_gsnet_square",
            "records": 114,
            "n": 30,
            "mean_norm": 0.0625,
            "floor": 0.938,
            "frac": 100.0,
        },
    ]
    out_yes = mod.build_report(cfg, rows, meta, floors)
    out_no = mod.build_report(cfg, rows, meta, None)
    print("====FLOORS_YES====")
    print(out_yes)
    print("====FLOORS_NO====")
    print(out_no)


if __name__ == "__main__":
    main()
