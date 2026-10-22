"""EXPERIMENT: fastest few-parameter adaptation to an optical-path change.

**Offline, simulation-only.** No device is opened.

Model and "true bench" deliberately share the *same* canonical numpy propagator
(:func:`forward_intensity`) and the *same* canonical Zernike basis
(:func:`generate_zernike_phase`). They differ in exactly one thing: the geometry
vector ``theta`` the propagator is told to use. That is what makes this an honest
adaptation test -- if the adapter could match the bench perfectly with zero
measurements, the two sides would be trivially identical and the experiment
would measure nothing. The mismatch is geometric and physical.

``focal_length`` is deliberately NOT perturbed: it and ``camera_pixel_um`` both
set the overall far-field scale, so perturbing both is unidentifiable. That
degeneracy is a finding, not an omission.

Parameter groups, disjoint by construction -- the "reserve a subset of parameters
to fit the aberration" split:

* ``c``  -- 14 low-order Zernike coefficients, fitted online;
* ``dtheta`` -- 2 identifiable geometry scalars, fitted online;
* ``g``  -- a small CNN image correction trained once OFFLINE at nominal and
  FROZEN during adaptation.

Judged on R2 / corr / beam-domain metrics only. Never MSE/PSNR/SSIM: this repo
measured PSNR 72 dB at a *worse* R2 under sum normalisation, i.e. the metric was
measuring the normalisation, not the fit.

Run:
    python scripts/slm/exp_fast_recal_adapter.py --smoke
    python scripts/slm/exp_fast_recal_adapter.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    camera_window_from_far_field,
    forward_intensity,
)
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

GRID = 64
WIN = (64, 64)
NCOEF = 136  # n_max = 15 triangular, piston INCLUDED (piston is spatially flat)
K = 14       # adapter size: the 14 low-order modes a fast adapter can express
N_ORDER = 15

# Nominal calibration the frozen model believes.
THETA0 = {"camera_pixel_um": 2.20, "aperture_m": 12.0e-3, "focal_length_m": 0.125}

# The frozen calibration is *imperfect*. This residual is the ONLY thing the
# offline-trained image correction can absorb, which is the whole point: the
# adapter then only ever has to explain the post-drift DELTA, not the standing
# calibration error. Without it the CNN would have nothing to learn and the
# c_img arm would be a no-op that silently "works".
CALIB_ERR = {"camera_pixel_um": 0.020, "aperture_m": 0.010}  # fractional


def theta_hat() -> dict:
    """What the frozen model believes the bench geometry is."""
    t = dict(THETA0)
    t["camera_pixel_um"] *= 1.0 + CALIB_ERR["camera_pixel_um"]
    t["aperture_m"] *= 1.0 - CALIB_ERR["aperture_m"]
    return t


def bench_cfg(theta: dict) -> ShapingBenchConfig:
    return ShapingBenchConfig(
        n_grid=GRID,
        aperture_size=theta["aperture_m"],
        wavelength=1064e-9,
        focal_length=theta["focal_length_m"],
        target_side_px=8,
        far_field_padding=10,
        seed=0,
    )


def peak_norm(a: np.ndarray) -> np.ndarray:
    """Peak normalisation. AGENTS.md: 'sum' is the trap -- it makes MSE/PSNR/SSIM
    measure the normalisation instead of the fit. 'peak' is the correct choice."""
    a = np.nan_to_num(np.asarray(a, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


_BASIS: np.ndarray | None = None


def basis() -> np.ndarray:
    """Canonical Zernike basis, shape (NCOEF, GRID, GRID), built ONCE.

    ``generate_zernike_phase`` rebuilds the RZern radial grid on every call
    (63 ms measured), which dominated everything else by 2x. Building the basis
    once and combining by matvec turns a phase into ~0.05 ms. This does **not**
    reimplement the Zernike math -- every column is still produced by the
    canonical generator, just called once per mode instead of once per vector.
    """
    global _BASIS
    if _BASIS is None:
        cols = np.empty((NCOEF, GRID, GRID))
        for j in range(NCOEF):
            e = np.zeros(NCOEF)
            e[j] = 1.0
            cols[j] = np.nan_to_num(
                generate_zernike_phase(e, resolution=(GRID, GRID), n_max=N_ORDER),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
        _BASIS = cols
    return _BASIS


def zphase(coeffs: np.ndarray) -> np.ndarray:
    """Pupil phase from canonical Noll coefficients.

    ``generate_zernike_phase`` returns NaN OUTSIDE the aperture disc; every
    consumer must nan_to_num it, otherwise the whole far field goes non-finite
    and every metric silently reads 0 (documented trap in AGENTS.md).
    """
    return np.tensordot(np.asarray(coeffs, float), basis(), axes=(0, 0))


def bench_image(cmd: np.ndarray, ab: np.ndarray, theta: dict) -> np.ndarray:
    """One image from a bench with geometry ``theta`` and pupil aberration ``ab``."""
    cfg = bench_cfg(theta)
    far = forward_intensity(zphase(cmd + ab), cfg)
    return camera_window_from_far_field(
        far, cfg, WIN, camera_pixel_um=theta["camera_pixel_um"]
    )


# ---------------------------------------------------------------------------
# Perturbation families
# ---------------------------------------------------------------------------

KINDS = ("P-A-low", "P-A-high", "P-B-geom")


def _orders(nmax: int) -> np.ndarray:
    """Noll radial order for each of the NCOEF indices (index 0 == Noll 1)."""
    out = np.empty(NCOEF, int)
    for j in range(NCOEF):
        out[j] = int(np.ceil((-3 + np.sqrt(9 + 8 * (j + 1))) / 2))
    return out


ORDERS = _orders(N_ORDER)


def make_aberration(rng: np.random.Generator, lo: int, hi: int, sigma: float) -> np.ndarray:
    ab = np.zeros(NCOEF)
    sel = (ORDERS >= lo) & (ORDERS <= hi)
    ab[sel] = rng.normal(0.0, sigma, sel.sum())
    ab[0] = 0.0  # piston is spatially flat; it is not a "visible" aberration
    return ab


@dataclass
class Perturb:
    kind: str
    ab: np.ndarray
    theta: dict


def make_perturb(rng: np.random.Generator, kind: str) -> Perturb:
    th = dict(THETA0)
    ab = np.zeros(NCOEF)
    if kind == "P-A-low":
        ab = make_aberration(rng, 2, 6, 0.25)
    elif kind == "P-A-high":
        ab = make_aberration(rng, 7, N_ORDER, 0.15)
    elif kind == "P-B-geom":
        th["camera_pixel_um"] *= 1.0 + rng.uniform(-0.03, 0.03)
        th["aperture_m"] *= 1.0 + rng.uniform(-0.04, 0.04)
    else:
        raise ValueError(kind)
    return Perturb(kind, ab, th)


def probes(rng: np.random.Generator, n: int, scale: float = 0.6) -> np.ndarray:
    """Probe commands: random across the FULL basis, so the adapter has real
    work to do and cannot cheat by seeing only its own subspace."""
    w = np.zeros((n, NCOEF))
    w[:, 1:] = rng.normal(0.0, scale, (n, NCOEF - 1))
    return w


# ---------------------------------------------------------------------------
# Metrics -- R2 / corr / beam domain only
# ---------------------------------------------------------------------------


def _r2(meas: list[np.ndarray], pred: list[np.ndarray]) -> float:
    """R2 over the whole held-out stack (all pixels of all frames pooled)."""
    a = np.concatenate([np.asarray(m, float).ravel() for m in meas])
    b = np.concatenate([np.asarray(p, float).ravel() for p in pred])
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return 1.0 - float(((a - b) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def beam(img: np.ndarray) -> tuple[float, float]:
    """(ee90 radius, cv) from a peak-normalised image. Beam-domain metrics."""
    a = peak_norm(img)
    yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
    r = np.hypot(xx - a.shape[1] / 2.0, yy - a.shape[0] / 2.0)
    rf = r.ravel()
    order = np.argsort(rf)  # flatten first: argsort on a 2D array sorts per row
    cum = np.cumsum(a.ravel()[order]) / max(a.sum(), 1e-12)
    # searchsorted returns len(cum) when the cumulative sum never reaches the
    # fraction (numerically possible once the frame is mostly background), which
    # would index out of bounds -- clamp to the last valid radius instead.
    idx = min(int(np.searchsorted(cum, 0.9)), rf.size - 1)
    ee90 = float(rf[order][idx])
    m = r <= max(6.0, 0.4 * ee90)
    roi = a[m]
    cv = float(roi.std() / roi.mean()) if roi.size and roi.mean() > 0 else float("nan")
    return ee90, cv


# ---------------------------------------------------------------------------
# Adapter: fit c (and optionally dtheta) to N measurements
# ---------------------------------------------------------------------------


def predict(cmd: np.ndarray, c: np.ndarray, ab_model: np.ndarray,
            theta: dict, g=None) -> np.ndarray:
    """What the ADAPTED model predicts for command ``cmd``.

    ``ab_model`` is the model's *belief* about the bench aberration. At nominal
    that belief is zero -- the model has no way to know it -- so an adapter that
    only fits ``c`` is implicitly claiming the aberration IS the correction.

    ``g`` is the frozen image correction. It was trained at nominal against a
    mis-calibrated model (:data:`theta_hat`), so it encodes the *standing*
    calibration residual. During adaptation it is frozen, so it cannot follow
    drift -- which is precisely what this arm is testing.
    """
    return peak_norm(apply_g(g, bench_image(cmd, ab_model + c, theta)))


# ---------------------------------------------------------------------------
# Frozen image correction: the "图像矫正" half of the hypothesis
# ---------------------------------------------------------------------------


class TinyCorr(torch.nn.Module):
    """Small residual CNN image correction -- few parameters BY DESIGN.

    ~10k parameters, so the point of the comparison stays honest: the c_img arm
    adapts 14 numbers like c_only does; only the frozen prior differs.
    """

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, ch, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(ch, ch, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


_G_CACHE: dict[int, TinyCorr] = {}


def train_image_correction(seed: int, n_train: int = 192,
                           epochs: int = 60) -> TinyCorr:
    """Train the frozen correction OFFLINE at nominal, then freeze it.

    Targets the *standing calibration error* (model believes theta_hat, bench is
    theta0). Trained once per seed and cached; never updated during adaptation.
    """
    if seed in _G_CACHE:
        return _G_CACHE[seed]
    torch.manual_seed(seed)
    g = TinyCorr()
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    th_true, th_model = dict(THETA0), theta_hat()
    for w in probes(rng, n_train):
        xs.append(peak_norm(bench_image(w, np.zeros(NCOEF), th_model)))
        ys.append(peak_norm(bench_image(w, np.zeros(NCOEF), th_true)))
    X = torch.as_tensor(np.array(xs), dtype=torch.float32)[:, None]
    Y = torch.as_tensor(np.array(ys), dtype=torch.float32)[:, None]
    opt = torch.optim.Adam(g.parameters(), lr=3e-3)
    idx = np.arange(n_train)
    for _ in range(epochs):
        rng.shuffle(idx)
        for b in range(0, n_train, 32):
            j = idx[b : b + 32]
            opt.zero_grad()
            loss = ((g(X[j]) - Y[j]) ** 2).mean()
            loss.backward()
            opt.step()
    for p in g.parameters():
        p.requires_grad_(False)
    g.eval()
    _G_CACHE[seed] = g
    return g


def apply_g(g, img: np.ndarray) -> np.ndarray:
    if g is None:
        return img
    with torch.no_grad():
        t = torch.as_tensor(peak_norm(img), dtype=torch.float32)[None, None]
        return g(t)[0, 0].numpy().astype(np.float64)


def fit_adapter(meas: list[np.ndarray], cmds: np.ndarray, theta_start: dict,
                *, use_geom: bool, g=None) -> tuple[np.ndarray, dict]:
    """Coordinate-descent fit of ``c`` (K) and optionally ``dtheta`` (2).

    Only <=16 parameters, so bounded coordinate descent is both sufficient and
    fully deterministic -- no autodiff through the numpy propagator, and no
    optimiser hyperparameter that could be mistaken for a finding.
    """
    theta = dict(theta_start)
    anchor = dict(theta_start)
    c = np.zeros(NCOEF)

    def _err(pred, tgt):
        return float(np.mean((peak_norm(pred) - peak_norm(tgt)) ** 2))

    def loss(c_, th_):
        return float(
            np.mean([_err(predict(cmds[i], c_, np.zeros(NCOEF), th_, g), meas[i])
                     for i in range(len(cmds))])
        )

    # --- geometry first (it dominates; c can only mop up) ---
    if use_geom:
        best = (loss(c, theta), theta["camera_pixel_um"], theta["aperture_m"])
        for s in np.linspace(0.95, 1.05, 5):
            for a in np.linspace(0.95, 1.05, 5):
                t2 = dict(theta)
                t2["camera_pixel_um"] = anchor["camera_pixel_um"] * s
                t2["aperture_m"] = anchor["aperture_m"] * a
                v = loss(c, t2)
                if v < best[0]:
                    best = (v, t2["camera_pixel_um"], t2["aperture_m"])
        theta["camera_pixel_um"], theta["aperture_m"] = best[1], best[2]

    # --- then the low-order coefficients ---
    for _ in range(2):
        for j in range(1, K + 1):
            bestv, bestc = loss(c, theta), c[j]
            for gv in np.linspace(-1.0, 1.0, 7):
                c[j] = gv
                v = loss(c, theta)
                if v < bestv:
                    bestv, bestc = v, gv
            c[j] = bestc
    return c, theta


# ---------------------------------------------------------------------------
# One (arm, kind, n_frames) cell
# ---------------------------------------------------------------------------

ARMS = ("nominal", "nominal_img", "c_only", "c_img", "geom+c")


def run_cell(arm: str, pert: Perturb, n_frames: int, n_hold: int,
             seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cmds = probes(rng, max(n_frames, 1))
    meas = [peak_norm(bench_image(cmds[i], pert.ab, pert.theta))
            for i in range(len(cmds))]

    # The frozen model starts from the mis-calibrated geometry, and the frozen
    # image correction (if this arm uses one) is trained once and never updated.
    # ``nominal_img`` is the control that makes the c_img comparison honest: with
    # it, c_img - nominal_img isolates what the frozen CNN does for ADAPTATION,
    # rather than crediting it for absorbing the standing calibration error.
    g = (train_image_correction(seed)
         if arm in ("nominal_img", "c_img") else None)
    theta_start = theta_hat()

    if arm in ("nominal", "nominal_img"):
        c, theta = np.zeros(NCOEF), dict(theta_start)
    else:
        c, theta = fit_adapter(meas, cmds, dict(theta_start),
                               use_geom=(arm == "geom+c"), g=g)

    hrng = np.random.default_rng(99991)
    hold = probes(hrng, n_hold)
    ab_model = np.zeros(NCOEF)  # model starts blind to the bench aberration
    pred = [predict(hold[i], c, ab_model, theta, g) for i in range(n_hold)]
    meas_h = [peak_norm(bench_image(hold[i], pert.ab, pert.theta)) for i in range(n_hold)]

    corrs = [_corr(p, m) for p, m in zip(pred, meas_h)]
    e90p, cvp = (np.array(v) for v in zip(*[beam(p) for p in pred]))
    e90m, cvm = (np.array(v) for v in zip(*[beam(m) for m in meas_h]))
    return {
        "arm": arm,
        "kind": pert.kind,
        "n_frames": n_frames,
        "seed": seed,
        "r2": _r2(meas_h, pred),
        "corr": float(np.nanmean(corrs)),
        "cv_err": float(np.nanmean(np.abs(cvp - cvm))),
        "ee90_err": float(np.nanmean(np.abs(e90p - e90m))),
        "fit_params": {"nominal": 0, "nominal_img": 0, "c_only": K,
                       "c_img": K, "geom+c": K + 2}[arm],
        "frozen_params": sum(p.numel() for p in g.parameters()) if g is not None else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=0, help="perturb seeds per kind")
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.smoke:
        seeds, frames = 1, [1, 4]
    elif args.quick:
        seeds, frames = 6, [1, 2, 4, 8, 16]
    else:
        seeds, frames = (args.seeds or 6), (args.frames or [1, 2, 4, 8, 16])

    out = args.out or (ROOT / "logs" / "fast_recal_adapter" /
                       ("smoke.json" if args.smoke else "results.json"))
    rows = []
    t0 = time.time()
    for kind in KINDS:
        for s in range(seeds):
            pert = make_perturb(np.random.default_rng(500 + s), kind)
            for nf in frames:
                for arm in ARMS:
                    rows.append(run_cell(arm, pert, nf, 16, seed=31 * s + 7))
            print(f"  {kind} seed={s} done ({time.time() - t0:.0f}s)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {out}  ({len(rows)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()