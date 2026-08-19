from __future__ import annotations

"""Generate synthetic dual-camera Zernike dataset from aotools turbulent screens.

Creates ``data/slm_dual_spot/<run>/sample_XXXX/{daheng_frame.npy, miicam_frame.npy,
metadata.json}`` in exactly the format consumed by ``ml.zernike_prediction.dataset``
so training/eval CLI works with ZERO pipeline changes.

Physics model (idealized free-space 4f system):
  * aotools PhaseScreenVonKarman -> turbulent phase screen (the "wavefront").
  * fit_zernike(phase, n_max=15) -> 136 Noll coefficients; the reconstructed
    phase = sum of those (orders <= 15), so the PSF contains up-to-15 content.
  * Labels (metadata n_max=10) = first 66 Noll coeffs (piston + 65), i.e. 65-dim
    regression targets — higher-order content present in the image but NOT in the
    labels, matching real turbulence (unmeasured residual high-order modes).
  * Focus image (daheng, 1024x1280 uint16): |FFT2(pupil exp(i phi))|^2, PSF core
    ~20 px on canvas. Two quantizations:
      - clean: uint16 full range (peak ~60000), well-exposed, faint Poisson noise.
      - clamp: MONO8-style clamp -> uint16 array whose values are clipped to [0,255]
        (exactly the real-capture defect from daheng driver's MONO8 hardcode).
  * Pupil image (miicam, 1520x2688 uint8): beam-intensity over the aperture.
      - pure: featureless Gaussian illumination (no phase info — pure phase).
      - coupled: SLM amplitude coupling model I = beam*(1 + eps*cos(phi mod 2pi)),
        periodic in 2pi like the real Santec 1064nm coupling (period ~993 gray).

Four variants (run dirs), all driven by the SAME turbulent screens/PSF:
  sim_turb_clean_pure / sim_turb_clean_coupled / sim_turb_clamp_pure / sim_turb_clamp_coupled
so clean-vs-clamp is a controlled quantization ablation and pure-vs-coupled a
controlled pupil-information ablation.

Usage:
  python scripts/sim_zernike_dataset.py [--n-samples 333] [--seed 1]
"""

import argparse
import json
import multiprocessing
import time
from functools import partial
from pathlib import Path

import numpy as np
from loguru import logger

from ao_shaping.utils.zernike_calc import RZern


# --------------------------------------------------------------------------- #
# Fixed camera geometry (match the real capture pipeline)
# --------------------------------------------------------------------------- #

DAHENG_H, DAHENG_W = 1024, 1280   # focus camera native (uint16)
MIICAM_H, MIICAM_W = 1520, 2688   # pupil camera native (uint8)
CENTER = (577, 655)               # real daheng spot center (row, col)
GRAY_MAX = 1022                   # SLM 10-bit gray scale (real metadata uses 0..1022)
NOMINAL_LABEL_RMS = 1.6           # target non-piston label std (real data ~1.6-1.8)


_CART_CACHE: dict[tuple[int, tuple[int, int]], RZern] = {}


def _cart(n_max: int, shape: tuple[int, int]) -> RZern:
    """Cached RZern grid — make_cart_grid costs ~0.8 s and is pure setup."""
    key = (n_max, shape)
    cart = _CART_CACHE.get(key)
    if cart is None:
        h, w = shape
        cart = RZern(n_max)
        ddx = np.linspace(-1.0, 1.0, w)
        ddy = np.linspace(-1.0, 1.0, h)
        xv, yv = np.meshgrid(ddx, ddy)
        cart.make_cart_grid(xv, yv)
        _CART_CACHE[key] = cart
    return cart


def _noll_coeffs_from_phase(phase: np.ndarray, n_max: int) -> np.ndarray:
    """Fit Noll Zernike coefficients (len = (n_max+1)(n_max+2)/2) via RZern."""
    return _cart(n_max, phase.shape).fit_cart_grid(phase)[0]


_RZERN_TO_NM: dict[int, tuple[int, int]] | None = None


def _rzern_nm_map(n_max: int) -> dict[int, tuple[int, int]]:
    """RZern basis index -> ``(n, m)`` via correlation with analytic modes.

    aotools RZern orders its basis as a Noll-like cascade (1=2rho cos, 2=2rho
    sin, 3=defocus, ...) which differs from the capture pipeline's ``(n, m)``
    order (``phase_gen.iter_nm_terms``). ``fit_cart_grid`` returns RZern order;
    metadata must be written in ``(n, m)`` order to match the real dataset and
    the regressor. Piston (index 0) maps to ``(0, 0)`` and is skipped here.
    """
    global _RZERN_TO_NM
    if _RZERN_TO_NM is None:
        cart = _cart(n_max, (160, 160))
        dd = np.linspace(-1.0, 1.0, 160)
        xv, yv = np.meshgrid(dd, dd)
        rho = np.sqrt(xv**2 + yv**2)
        th = np.arctan2(yv, xv)
        disk = rho <= 1.0
        from ml.zernike_prediction.phase_gen import iter_nm_terms, zernike_radial
        refs = []
        for n, m in iter_nm_terms(n_max):
            ang = np.cos(m * th) if m >= 0 else np.sin(-m * th)
            mode = np.where(disk, zernike_radial(n, abs(m), rho) * ang, 0.0)
            refs.append(mode / (np.sqrt((mode**2).mean()) + 1e-12))
        refs = np.array(refs)
        m = {}
        for i in range(1, cart.nk):
            padded = np.zeros(cart.nk)
            padded[i] = 1.0
            z = np.nan_to_num(cart.eval_grid(padded, matrix=True))
            z /= np.sqrt((z**2).mean()) + 1e-12
            m[i] = iter_nm_terms(n_max)[int(np.abs(z * refs).sum(axis=(1, 2)).argmax())]
        _RZERN_TO_NM = m
    return _RZERN_TO_NM


def _to_metadata_order(coeffs_rz: np.ndarray, n_max: int) -> np.ndarray:
    """Reorder RZern-ordered coeffs into the dataset's ``(n, m)`` metadata order."""
    nm = _rzern_nm_map(n_max)
    from ml.zernike_prediction.phase_gen import iter_nm_terms
    pos = {nm_: i for i, nm_ in enumerate(iter_nm_terms(n_max))}
    out = np.zeros_like(coeffs_rz)
    for i, nm_ in nm.items():
        if i < len(coeffs_rz) and nm_ in pos and pos[nm_] < len(out):
            out[pos[nm_]] = coeffs_rz[i]
    return out


def _reconstruct_phase(coeffs: np.ndarray, n_max: int, shape: tuple[int, int]) -> np.ndarray:
    """Reconstruct phase (radians, piston-removed) from Noll coeffs on ``shape`` grid."""
    cart = _cart(n_max, shape)
    padded = np.zeros(cart.nk, dtype=np.float64)
    padded[: min(len(coeffs), cart.nk)] = coeffs[: min(len(coeffs), cart.nk)]
    phase = cart.eval_grid(padded, matrix=True)
    # zernike.eval_grid returns NaN outside the unit disk (rho > 1); the
    # downstream aperture mask discards those points anyway.
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
    return phase - phase.mean()  # piston-removed (diffraction-irrelevant)


def _turbulent_coeffs(
    grid: int, r0: float, L0: float, seed: int, n_max_fit: int, aperture_px: int,
    label_rms: float = NOMINAL_LABEL_RMS,
) -> tuple[np.ndarray, tuple[int, int]]:
    """aotools von-Karman screen -> 136 Noll coeffs fitted over the APERTURE ONLY.

    KEY FIX: the earlier version fitted Zernike over the FULL 512x512 screen (unit
    disk = whole grid), while the PSF only samples the central ``aperture_px``
    sub-region. With D/r0 >> 1 the global-fit coefficients do NOT describe the
    aperture wavefront that actually forms the image (measured corr^2 of the
    global tilt label vs. the aperture-local tilt is ~0.004-0.05). This version
    slices the central 2*aperture_px square and fits the Zernike basis with the
    unit disk = the aperture, so the labels and the image-forming phase agree.

    The reconstructed phase is also returned at this aperture size, so the PSF
    render uses exactly the phase the labels describe.

    Returns:
        ``(coeffs, patch_shape)`` where ``patch_shape`` is ``(2*ap, 2*ap)``.
    """
    from aotools.turbulence import PhaseScreenVonKarman

    ps = PhaseScreenVonKarman(grid, pixel_scale=1.0, r0=r0, L0=L0, random_seed=seed)
    screen = np.asarray(ps.scrn, dtype=np.float64)
    screen = np.real(screen) if np.iscomplexobj(screen) else screen
    # central aperture patch = the optical pupil (unit disk of the Zernike basis)
    c = grid // 2
    patch = screen[c - aperture_px : c + aperture_px, c - aperture_px : c + aperture_px]
    coeffs = _noll_coeffs_from_phase(patch, n_max=n_max_fit)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeffs[0] = 0.0  # drop piston (global phase, no image effect)
    # normalize the FULL coefficient vector (all displayed orders) to the real
    # wavefront scale; keeps low-order labels ~label_rms while high-order content
    # stays physically present in the PSF halo.
    rms_all = float(np.sqrt(np.mean(coeffs[1:] ** 2)))
    if rms_all > 1e-9:
        coeffs *= label_rms / rms_all
    return coeffs, patch.shape


def _aperture_mask(shape: tuple[int, int], radius_px: int) -> np.ndarray:
    h, w = shape
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[:h, :w]
    return (np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) <= radius_px).astype(np.float64)


def _focus_psf(phase: np.ndarray, aperture_px: int) -> np.ndarray:
    """Focal-plane PSF via FFT of pupil field; core ~20 px after placement."""
    grid = phase.shape[0]
    mask = _aperture_mask(phase.shape, aperture_px)
    E = np.exp(1j * phase) * mask
    # zero-pad to a larger FFT grid: finer focal-plane sampling
    n_fft = 2048
    tmp = np.zeros((n_fft, n_fft), dtype=np.complex128)
    y0 = (n_fft - grid) // 2
    x0 = y0
    tmp[y0 : y0 + grid, x0 : x0 + grid] = E
    focal = np.abs(np.fft.fftshift(np.fft.fft2(tmp))) ** 2
    # central patch carrying the core + first rings
    half = 256
    cy = cx = n_fft // 2
    patch = focal[cy - half : cy + half, cx - half : cx + half]
    return patch  # (512, 512) unnormalized intensity


def _place_psf(patch: np.ndarray, target: tuple[int, int], center_rc: tuple[int, int]) -> np.ndarray:
    """Resize PSF patch up to the daheng canvas and place at ``center_rc``."""
    from scipy.ndimage import zoom

    if patch.shape != target:
        zy = target[0] / patch.shape[0]
        zx = target[1] / patch.shape[1]
        patch = zoom(patch, (zy, zx), order=3)
    frame = np.zeros(target, dtype=np.float64)
    cy, cx = center_rc
    h, w = target
    r0 = max(0, cy - h // 2)
    r1 = min(h, cy + h // 2)
    c0 = max(0, cx - w // 2)
    c1 = min(w, cx + w // 2)
    frame[r0:r1, c0:c1] = patch[: r1 - r0, : c1 - c0]
    return frame


def _render_daheng(psf: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """PSF intensity -> daheng uint16 frame (clean full-range or MONO8-clamp)."""
    psf = psf / psf.max()
    if mode == "clean":
        peak = 60000.0
        noise_scale = 3.0
    else:  # clamp: real capture defect — 8-bit quantization clipped into uint16
        peak = 255.0
        noise_scale = 0.5
    img = psf * peak
    img = img + rng.poisson(np.maximum(0.0, img * 0.02)).astype(np.float64)
    img = img + rng.normal(0.0, noise_scale, img.shape)
    if mode == "clamp":
        img = np.clip(img, 0, 255)
    return np.clip(img, 0, 65535).astype(np.uint16)


def _beam_intensity(shape: tuple[int, int], aperture_px: int, w0_frac: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-beam illumination inside the aperture; returns (profile, mask)."""
    h, w = shape
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (r <= aperture_px).astype(np.float64)
    w0 = aperture_px * w0_frac
    profile = np.exp(-2.0 * (r / w0) ** 2) * mask
    return profile, mask


def _render_miicam(
    phase: np.ndarray,
    aperture_px: int,
    mode: str,
    coupling_amp: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pupil-plane uint8 frame: featureless beam (pure) or amplitude-coupled."""
    grid = phase.shape[0]
    profile, mask = _beam_intensity((grid, grid), aperture_px)
    if mode == "pure":
        intensity = profile * mask
    else:  # coupled: reflectivity varies with wrapped phase (period 2 pi)
        wrapped = np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi)
        intensity = profile * mask * (1.0 + coupling_amp * np.cos(2.0 * np.pi * wrapped))
    intensity = np.clip(intensity, 0.0, None)
    # upscale to miicam native resolution and quantize to uint8 (bilinear:
    # the pupil intensity is smooth, cubic spline adds no information)
    from scipy.ndimage import zoom

    zy = MIICAM_H / intensity.shape[0]
    zx = MIICAM_W / intensity.shape[1]
    img = zoom(intensity, (zy, zx), order=1)
    img = img / (img.max() + 1e-12) * 180.0
    img = img + rng.normal(0.0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), (profile)


def _write_sample(
    run_dir: Path,
    idx: int,
    daheng: np.ndarray,
    miicam: np.ndarray,
    coeffs_all: np.ndarray,
    n_max_fit: int,
    label_n_max: int,
    phase_min: float,
    phase_max: float,
    sim: dict,
    write_png: bool = True,
) -> None:
    sd = run_dir / f"sample_{idx:04d}"
    sd.mkdir(parents=True, exist_ok=True)
    np.save(sd / "daheng_frame.npy", daheng)
    np.save(sd / "miicam_frame.npy", miicam)

    n_terms = (label_n_max + 1) * (label_n_max + 2) // 2
    coeffs_label = np.asarray(coeffs_all[:n_terms], dtype=np.float64)
    # metadata mirrors the real capture format (phase_params.coefficients in Noll order)
    meta = {
        "sample_idx": idx,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase_type": "turbulent_zernike",
        "phase_params": {
            "n_max": label_n_max,
            "coefficients": [float(x) for x in coeffs_label],
        },
        "phase_shape": [daheng.shape[0], daheng.shape[1]],
        "phase_min": float(phase_min),
        "phase_max": float(phase_max),
        "slm_memory_slot": 3 + idx % 3,
        "daheng": {"width": DAHENG_W, "height": DAHENG_H, "mode": "uint16"},
        "miicam": {"width": MIICAM_W, "height": MIICAM_H, "mode": "uint8"},
        "sim": {**sim, "n_max_fit": n_max_fit, "label_n_max": label_n_max,
                "n_coeffs": int(len(coeffs_all))},
    }
    (sd / "metadata.json").write_text(json.dumps(meta))
    if write_png:
        (sd / "daheng_frame.png").write_bytes(_png(daheng))
        (sd / "miicam_frame.png").write_bytes(_png(miicam))


def _png(img: np.ndarray) -> bytes:
    from io import BytesIO

    from PIL import Image

    im8 = (img.astype(np.float64) / (img.max() + 1e-12) * 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(im8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# End-to-end generation
# --------------------------------------------------------------------------- #


def _generate_one(cfg: dict, i: int) -> int:
    """Generate sample ``i`` for all variants; returns ``i`` (for progress)."""
    rng = np.random.default_rng(cfg["seed"] * 1000003 + i)
    grid = cfg["grid"]
    coeffs_all, patch_shape = _turbulent_coeffs(
        grid, cfg["r0"], cfg["L0"], seed=cfg["seed"] + i,
        n_max_fit=cfg["n_max_fit"], aperture_px=cfg["aperture_px"],
        label_rms=cfg["label_rms"],
    )
    # Render the image from EXACTLY the labeled coefficients (first ``label_n_max``
    # orders). Orders above the label cutoff carry no supervised signal — if they
    # entered the PSF they would act as irreducible speckle "noise" uncorrelated
    # with the targets. Rendering from the labeled phase makes image -> labels a
    # deterministic bijection (modulo camera noise), the learnable supervised setup.
    n_label = (cfg["label_n_max"] + 1) * (cfg["label_n_max"] + 2) // 2
    coeffs_render = np.asarray(coeffs_all[:n_label], dtype=np.float64)
    phi = _reconstruct_phase(coeffs_render, cfg["label_n_max"], patch_shape)
    # Metadata labels must use the capture pipeline's (n, m) order; fit_cart_grid
    # returns RZern order. Reorder once so the written labels match the dataset.
    coeffs_meta = _to_metadata_order(coeffs_all, cfg["n_max_fit"])

    psf_patch = _focus_psf(phi, cfg["aperture_px"])
    jitter = tuple(int(c + rng.normal(0, 6)) for c in CENTER)
    psf_canvas = _place_psf(psf_patch, (DAHENG_H, DAHENG_W), jitter)

    miicam_pure, _ = _render_miicam(phi, cfg["aperture_px"], "pure", cfg["coupling_amp"], rng)
    miicam_coupled, _ = _render_miicam(phi, cfg["aperture_px"], "coupled", cfg["coupling_amp"], rng)

    daheng_clean = _render_daheng(psf_canvas, "clean", rng)
    daheng_clamp = _render_daheng(psf_canvas, "clamp", rng)

    phase_gray_min = float(phi.min() / (2 * np.pi) * GRAY_MAX)
    phase_gray_max = float(phi.max() / (2 * np.pi) * GRAY_MAX)

    for v in cfg["variants"]:
        daheng = daheng_clean if "clean" in v else daheng_clamp
        miicam = miicam_coupled if "coupled" in v else miicam_pure
        _write_sample(
            Path(cfg["out_root"]) / f"sim_turb_{v}", i, daheng, miicam, coeffs_meta,
            cfg["n_max_fit"], cfg["label_n_max"], phase_gray_min, phase_gray_max,
            cfg["sim_info"], write_png=cfg["write_png"],
        )
    return i


def generate_dataset(
    out_root: Path,
    n_samples: int,
    seed: int,
    variants: list[str],
    grid: int = 512,
    aperture_px: int = 96,
    r0: float = 4.0,
    L0: float = 40.0,
    n_max_fit: int = 15,
    label_n_max: int = 10,
    coupling_amp: float = 0.25,
    label_rms: float = NOMINAL_LABEL_RMS,
    workers: int = 32,
    write_png: bool = False,
) -> dict[str, Path]:
    """Generate all requested variants in parallel (same underlying screens/PSF)."""
    out_root = Path(out_root)
    runs = {
        v: out_root / f"sim_turb_{v}" for v in variants
    }
    sim_info = {"turbulence": "PhaseScreenVonKarman", "r0": r0, "L0": L0,
                "grid": grid, "aperture_px": aperture_px, "coupling_amp": coupling_amp}
    cfg = {
        "out_root": str(out_root), "seed": seed, "variants": list(variants),
        "grid": grid, "aperture_px": aperture_px, "r0": r0, "L0": L0,
        "n_max_fit": n_max_fit, "label_n_max": label_n_max, "coupling_amp": coupling_amp,
        "label_rms": label_rms, "sim_info": sim_info, "write_png": write_png,
    }
    t0 = time.time()
    n_workers = min(workers, n_samples)
    done = 0
    with multiprocessing.Pool(n_workers) as pool:
        for _ in pool.imap_unordered(partial(_generate_one, cfg), range(n_samples)):
            done += 1
            if done % 50 == 0:
                logger.info("generated {}/{} samples ({:.1f}s)", done, n_samples, time.time() - t0)

    for v, run in runs.items():
        logger.info("variant {} -> {} ({} samples)", v, run, n_samples)
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=str, default="data/slm_dual_spot")
    p.add_argument("--n-samples", type=int, default=333)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--variants", type=str, default="clean_pure,clean_coupled,clamp_pure,clamp_coupled")
    p.add_argument("--grid", type=int, default=512)
    p.add_argument("--aperture-px", type=int, default=96)
    p.add_argument("--r0", type=float, default=4.0)
    p.add_argument("--L0", type=float, default=40.0)
    p.add_argument("--n-max-fit", type=int, default=15)
    p.add_argument("--label-n-max", type=int, default=10)
    p.add_argument("--coupling-amp", type=float, default=0.25)
    p.add_argument("--label-rms", type=float, default=NOMINAL_LABEL_RMS,
                   help="target non-piston coefficient RMS (rad); lower = more compact PSF")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--png", action="store_true", help="write inspection PNGs (not needed by the training pipeline)")
    args = p.parse_args()

    runs = generate_dataset(
        Path(args.out_root), args.n_samples, args.seed, args.variants.split(","),
        grid=args.grid, aperture_px=args.aperture_px, r0=args.r0, L0=args.L0,
        n_max_fit=args.n_max_fit, label_n_max=args.label_n_max, coupling_amp=args.coupling_amp,
        label_rms=args.label_rms, workers=args.workers, write_png=args.png,
    )
    for v, r in runs.items():
        print(v, r)


if __name__ == "__main__":
    main()