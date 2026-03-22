from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_shaping.optimizer.wfless.pib_sim_eval import run_suite_and_save


def main() -> None:
    output_dir = Path("logs") / "pib_sim_eval"
    summary_df, _, _, artifacts = run_suite_and_save(output_dir=output_dir)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nArtifacts saved to: {output_dir}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
