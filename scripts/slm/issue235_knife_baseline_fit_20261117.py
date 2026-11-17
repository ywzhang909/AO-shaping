"""离线：对 issue235 刀口曲线做带自由基线的 erf 拟合（不占设备）.

用法: python scripts/slm/issue235_knife_baseline_fit_20261117.py <beam_center.json>
模型: P(s) = b + a * 0.5 * (1 + erf((s - c) / (sqrt(2) * sg)))
输出每个轴的基线 b、幅度 a、中心 c、宽度 sg 与残差 rms, 以及与 find_beam_on_slm 结果的差.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


def _model(s: np.ndarray, b: float, a: float, c: float, sg: float) -> np.ndarray:
    return b + a * 0.5 * (1.0 + erf((s - c) / (np.sqrt(2.0) * sg)))


def fit_axis(s: np.ndarray, P: np.ndarray) -> dict:
    best = None
    for c0 in np.linspace(s.min(), s.max(), 15):
        for sg0 in (50.0, 150.0, 300.0):
            try:
                p, _ = curve_fit(
                    _model,
                    s,
                    P,
                    p0=[P.min(), P.max() - P.min(), c0, sg0],
                    bounds=(
                        [0.0, 0.0, s.min() - 500.0, 5.0],
                        [2.0 * P.max(), 2.0 * P.max(), s.max() + 500.0, 2000.0],
                    ),
                    maxfev=20000,
                )
            except (RuntimeError, ValueError):
                continue
            rms = float(np.sqrt(np.mean((_model(s, *p) - P) ** 2)))
            if best is None or rms < best[1]:
                best = (p, rms)
    if best is None:
        raise RuntimeError("erf fit failed for all starts")
    p, rms = best
    return {
        "baseline": float(p[0]),
        "amplitude": float(p[1]),
        "centre": float(p[2]),
        "sigma": float(p[3]),
        "rms_residual": rms,
        "power_range": float(P.max() - P.min()),
    }


def main(argv: list[str]) -> int:
    path = Path(argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    ref = data["result"]["beam_center"]  # (cy, cx) from find_beam_on_slm
    out = {}
    for ax in ("x", "y"):
        pts = data["curves"][ax]["points"]
        s = np.array([p["s"] for p in pts], dtype=float)
        P = np.array([p["power"] for p in pts], dtype=float)
        out[ax] = fit_axis(s, P)
    out["find_beam_on_slm_centre_yx"] = ref
    out["fit_centre_yx"] = [out["y"]["centre"], out["x"]["centre"]]
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
