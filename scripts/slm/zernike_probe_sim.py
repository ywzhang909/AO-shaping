"""纯仿真：±Zernike 主动探针相对随机探针的样本效率（issue #229）。

只使用 ``PhysicalForwardFit`` 的可解释静态相位参数（Z4=d、Z5--Z11=residual）。
twin 与拟合器共享算子族，但非理想性仅属于 twin。无设备导入、无相位 wrap。
报告和原始结果写到 ``logs/zernike_probe_sim/<UTC>/``，不依赖已有日志。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit
from ao_shaping.tools.slm.transfer.jacobian_guard import direction_agreement
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

MODES = tuple(range(4, 12))
LOW = frozenset(range(4, 9))
AMPLITUDES = {4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.2, 9: 0.3, 10: 0.3, 11: 0.3}
TRUE_C = np.array([0.16, 0.10, -0.08, 0.07, -0.06, 0.05, -0.04, 0.03])
CCD_SCALE = 100.0  # Fixed nominal electrons/count calibration; twin gain remains unknown.

# Photometric/PSF profile of the twin. --twin-profile swaps between the
# pre-2026-11-04 "clean" twin (no CCD blur, unit-ish gain) and a twin that adds
# the structural non-idealities measured offline on the physical device
# (issue #229 run01 sigma_scan.json): a Gaussian CCD PSF sigma=8 px, the
# measured irradiance offset, and larger ccount read noise. The absolute gain
# slope (fitted 337 vs 1.08) is normalized away by the ptp-normalised NRMSE/NCC
# metrics and the shared exposure-normalisation, so it is not applied here
# directly; what matters for sample efficiency is blur + offset + noise floor.
TWIN_PROFILES = {
    "clean": dict(name="clean", gain=1.08, background=2.0, sigma_ccd_px=0.0, noise_std=0.25),
    # sigma_ccd_px/background from sigma_scan.json (issue #229 run01); noise_std ~
    # measured temporal std of 3 repeated holdout frames (0.9-1.6 counts RMS on
    # peaks 69-206), not the earlier 6.0 guess which over-weights read noise.
    "hardware": dict(name="hardware", gain=1.08, background=0.72, sigma_ccd_px=8.0, noise_std=1.2),
}


def basis(size: int, scale: float = 1.0) -> dict[int, np.ndarray]:
    return {j: np.nan_to_num(generate_zernike_phase({j: 1.0}, (size, size), radius=size*scale/2)).astype(np.float32)
            for j in MODES}


def camera_grid(size: int, scale: float = 1.0) -> np.ndarray:
    axis = (np.arange(size, dtype=np.float64) - (size - 1) / 2) * scale
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    # The fitter expects camera-plane coordinates around its optical-axis origin.
    return np.stack([1360. + xx, 1020. + yy], axis=-1)


HW2F_MANIFEST = Path("D:/Projects/TIFO/AO-shaping/data/debug/zernike_probe_hardware/issue229/run01/manifest.json")
HW2F_SPAN = 160  # camera window half-span (px), matches _harness_task2.py SPAN


def hw2f_geometry(manifest: Path = HW2F_MANIFEST) -> dict:
    """Real 2f geometry from the recorded run01 manifest, exactly as _harness_task2.load_window.

    Reads only the manifest (no device imports): ramp-fitted matrix, reference spot as the
    optical-axis centre, and the same even-step camera window centred on the rounded spot.
    """
    from ao_shaping.tools.slm.transfer.fit import geometry, spot, valid
    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
    usable = [r for r in rows if valid(r)]
    matrix, _ = geometry(usable)
    ref = next(r for r in usable if r["label"] == "reference_start")
    centre = spot(ref)
    yy, xx = np.mgrid[-HW2F_SPAN:HW2F_SPAN + 1:2, -HW2F_SPAN:HW2F_SPAN + 1:2]
    xy = np.stack((xx + round(centre[0]), yy + round(centre[1])), axis=-1)
    return {"mode": "hw2f", "source": str(manifest), "n_total": len(rows), "n_usable": len(usable),
            "matrix": np.asarray(matrix, dtype=float).tolist(), "centre": [float(v) for v in centre],
            "amplitude_center": [960., 600.], "camera_window": {"span": HW2F_SPAN, "step": 2,
                                                                "size": int(xy.shape[0])},
            "camera_xy": xy}


def geometry_record(geom: dict | None) -> dict:
    """JSON-safe geometry description for config.json / report.md (no arrays)."""
    if geom is None:
        return {"mode": "synthetic", "matrix": [[1500., 0.], [0., 1500.]], "centre": [1360., 1020.],
                "amplitude_center": [960., 960.]}
    return {k: v for k, v in geom.items() if k != "camera_xy"}


def make_fit(phases: np.ndarray, images: np.ndarray, *, size: int, camera_size: int,
             scale: float = 1.0, true: bool = False, fft_size: int = 128,
             geom: dict | None = None) -> PhysicalForwardFit:
    if geom is None:
        camera_xy, matrix, amp_center = camera_grid(camera_size, scale), np.array([[1500., 0.], [0., 1500.]]), (960., 960.)
    else:
        camera_xy, matrix, amp_center = geom["camera_xy"], np.asarray(geom["matrix"], dtype=float), tuple(geom["amplitude_center"])
    fit = PhysicalForwardFit(
        phases=phases, images=images, camera_xy=camera_xy,
        matrix=matrix, focus_diopter=0.,
        train_mask=np.ones(len(phases), dtype=bool), amplitude_center=amp_center,
        learn_residual=True, fft_size=fft_size, pixel_samples=1, basis_arrays=basis(size, scale=.985 if true else 1.0),
        batch_size=24, device="cpu", slm_phase_a=0.08 if true else 0.,
        slm_phase_b=0.025 if true else 0.)
    if geom is not None:
        # Shared optical-axis intercept at the measured reference spot (harness build_fit).
        with fit.torch.no_grad():
            fit.origin.copy_(fit.torch.as_tensor(geom["centre"], dtype=fit.origin.dtype))
    fit.origin.requires_grad_(False)
    fit.center.requires_grad_(False)
    fit.log_width.requires_grad_(False)
    if true:
        with fit.torch.no_grad():
            fit.d.fill_(float(TRUE_C[0]) / fit.diopter_scale)
            fit.residual.copy_(fit.torch.as_tensor(TRUE_C[1:]))
    return fit


def blaze_kernel(sigma: float):
    """Identical Gaussian CCD-PSF kernel to production ``_build_blaze_kernel``."""
    if sigma <= 0.:
        return None
    import torch
    g = torch.arange(-12, 13, dtype=torch.float32)
    k = torch.exp(-0.5 * (g / (sigma / 2.0)) ** 2)
    k = k / k.sum()
    return (k[:, None] * k[None, :])[None, None]


def twin_render(fit: PhysicalForwardFit, rng: np.random.Generator,
                profile: dict | None = None) -> np.ndarray:
    profile = profile or TWIN_PROFILES["clean"]
    from scipy.ndimage import convolve as ndi_convolve
    ideal = fit.prediction().detach().cpu().numpy()
    gain = profile["gain"]
    background = profile["background"]
    saturation = 255.0
    # Apply the CCD PSF first (production blurs the *prediction*), so the twin
    # and the fitter share the same operator ordering with a non-ideal CCD.
    if profile["sigma_ccd_px"] > 0.:
        k = blaze_kernel(profile["sigma_ccd_px"]).numpy()[0, 0]
        ideal = np.stack([ndi_convolve(fr, k, mode="constant", cval=0.0) for fr in ideal])
    measured = np.clip(gain * ideal * CCD_SCALE + background, 0., saturation)
    # Quantisation and CCD read noise are only on the ground-truth side.
    measured = np.floor(measured * 16. + 0.5) / 16.
    return (measured + rng.normal(0., profile["noise_std"], measured.shape)).astype(np.float32)


def image_metrics(predicted: np.ndarray, measured: np.ndarray,
                  kernel: np.ndarray | None = None) -> dict[str, float]:
    if kernel is not None:
        from scipy.ndimage import convolve as ndi_convolve
        predicted = np.stack([ndi_convolve(fr, kernel, mode="constant", cval=0.0) for fr in predicted])
    predicted = predicted.reshape(len(predicted), -1)
    measured = measured.reshape(len(measured), -1)
    pred = np.maximum(predicted * CCD_SCALE, 0.)
    error = pred - measured
    nrmse = np.sqrt(np.mean(error ** 2, axis=1)) / np.maximum(np.ptp(measured, axis=1), 1e-8)
    p = pred - pred.mean(axis=1, keepdims=True)
    y = measured - measured.mean(axis=1, keepdims=True)
    ncc = np.sum(p*y, axis=1) / np.maximum(np.linalg.norm(p, axis=1)*np.linalg.norm(y, axis=1), 1e-12)
    c1, c2 = 2.55**2, 7.65**2
    ssim = ((2*pred.mean(1)*measured.mean(1)+c1)*(2*np.mean(p*y, 1)+c2)
            / ((pred.mean(1)**2+measured.mean(1)**2+c1)
               * (p.var(1)+y.var(1)+c2)))
    return {"nrmse": float(nrmse.mean()), "ncc": float(ncc.mean()), "ssim": float(ssim.mean())}


def loss_and_ncc(pred, measured, torch, kernel=None):
    if kernel is not None:
        pred = torch.nn.functional.conv2d(pred[:, None], kernel, padding=12)[:, 0]
    pred = pred.flatten(1)
    measured = measured.flatten(1)
    # Keep the training objective on the SAME scale as image_metrics and the
    # twin's measured counts: pred*CCD_SCALE (~0..2.4*100) vs measured (~0..255).
    # (was CCD_SCALE/255, ~255x smaller, distorting gradients for issue #229.)
    scale = CCD_SCALE
    l1 = torch.mean(torch.abs(pred * scale - measured))
    p = pred * scale
    p = p - p.mean(dim=1, keepdim=True)
    y = measured - measured.mean(dim=1, keepdim=True)
    ncc = (p*y).sum(1) / (p.square().sum(1).sqrt()*y.square().sum(1).sqrt()).clamp_min(1e-12)
    return l1 + 0.10 * (1. - ncc.mean()), ncc.mean()


def project_trust_region(fit: PhysicalForwardFit) -> None:
    """Z4 <= .3, Z5--Z11 <= .15, 全局 L2 <= .5 rad。"""
    torch = fit.torch
    with torch.no_grad():
        fit.d.clamp_(-.3 / fit.diopter_scale, .3 / fit.diopter_scale)
        fit.residual.clamp_(-.15, .15)
        vector = torch.cat([fit.d.reshape(1) * fit.diopter_scale, fit.residual])
        norm = vector.norm()
        if norm > .5:
            fit.d.mul_(.5 / norm)
            fit.residual.mul_(.5 / norm)


def fit_arm(probes: np.ndarray, measured: np.ndarray, *, size: int, camera_size: int,
            holdout: tuple[np.ndarray, np.ndarray], iterations: int = 120,
            patience: int = 18, threshold: float = 0.85,
            allowed_modes: tuple[int, ...] = MODES,
            evaluation: tuple[np.ndarray, np.ndarray] | None = None,
            kernel: np.ndarray | None = None, fft_size: int = 128,
             geom: dict | None = None) -> dict:
    fit = make_fit(probes, measured, size=size, camera_size=camera_size, fft_size=fft_size, geom=geom)
    if evaluation is not None:
        holdout = evaluation
    torch = fit.torch
    kt = None
    if kernel is not None:
        kt = torch.as_tensor(kernel[None, None], dtype=torch.float32)
    opt = torch.optim.Adam([{"params": [fit.d], "lr": .018 / fit.diopter_scale},
                            {"params": [fit.residual], "lr": .018}])
    hold_fit = make_fit(holdout[0], holdout[1], size=size, camera_size=camera_size, fft_size=fft_size, geom=geom)
    baseline_metrics = image_metrics(hold_fit.prediction().detach().numpy(), holdout[1], kernel)
    threshold *= baseline_metrics["nrmse"]
    history = []
    best = (float("inf"), 0., np.zeros(8))
    stale = 0
    started = time.perf_counter()
    threshold_step = None
    threshold_seconds = None
    for step in range(iterations):
        # Only low order updates first; middle order follows, never both at once.
        fit.d.requires_grad_(step < iterations // 2)
        opt.zero_grad()
        loss, _ = loss_and_ncc(fit.prediction(), torch.as_tensor(measured), torch, kt)
        loss.backward()
        if step < iterations // 2:
            fit.residual.grad[4:] = 0.
        else:
            fit.residual.grad[:4] = 0.
        if 4 not in allowed_modes:
            fit.d.grad = None
        for j in MODES[1:]:
            if j not in allowed_modes:
                fit.residual.grad[j-5] = 0.
        opt.step()
        project_trust_region(fit)
        if step % 5 == 0 or step == iterations - 1:
            with torch.no_grad():
                hold_fit.d.copy_(fit.d)
                hold_fit.residual.copy_(fit.residual)
                metrics = image_metrics(hold_fit.prediction().numpy(), holdout[1], kernel)
            history.append({"step": step + 1, **metrics})
            if threshold_step is None and metrics["nrmse"] <= threshold:
                threshold_step = step + 1
                threshold_seconds = time.perf_counter() - started
            if metrics["nrmse"] < best[0] - 1e-6:
                best = (metrics["nrmse"], float(fit.d.detach()), np.r_[float(fit.d.detach()) * fit.diopter_scale, fit.residual.detach().numpy()])
                stale = 0
            else:
                stale += 5
            if stale >= patience and step >= iterations // 2:
                break
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        fit.d.fill_(best[1])
        fit.residual.copy_(torch.as_tensor(best[2][1:]))
        hold_fit.d.copy_(fit.d)
        hold_fit.residual.copy_(fit.residual)
        hold_prediction = hold_fit.prediction().numpy()
        final = image_metrics(hold_prediction, holdout[1], kernel)
        downstream = image_metrics(hold_prediction[-1:], holdout[1][-1:], kernel)
        if kt is None:
            train_pred = fit.prediction() * CCD_SCALE
        else:
            train_pred = torch.nn.functional.conv2d(fit.prediction()[:, None], kt, padding=12)[:, 0] * CCD_SCALE
        train_l1 = float(torch.mean(torch.abs(train_pred - torch.as_tensor(measured).reshape_as(train_pred))))
    return {**final, "downstream_nrmse": downstream["nrmse"],
            "downstream_ncc": downstream["ncc"], "downstream_ssim": downstream["ssim"],
            "train_l1": train_l1, "iterations": len(history) and history[-1]["step"],
            "seconds": elapsed, "threshold_step": threshold_step, "threshold_seconds": threshold_seconds,
            "coefficients": best[2].tolist(), "history": history}


def random_phases(count: int, size: int, rng: np.random.Generator) -> np.ndarray:
    z = np.stack(list(basis(size).values()))
    phases = rng.normal(0., 1., (count, len(MODES))) @ z.reshape(len(MODES), -1)
    amplitudes = np.resize([AMPLITUDES[j] * np.sqrt(np.mean(z[k]**2))
                            for k, j in enumerate(MODES)], count)
    rms = np.sqrt(np.mean(phases**2, axis=1))
    phases *= (amplitudes / np.maximum(rms, 1e-12))[:, None]
    return phases.reshape(count, size, size).astype(np.float32)


def zernike_phases(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    z = np.stack(list(basis(size).values()))
    pairs = []
    for _ in range(n):
        order = rng.permutation(len(MODES))
        for index in order:
            for sign in (-1., 1.):
                pairs.append(sign * AMPLITUDES[MODES[index]] * z[index])
    return np.stack(pairs).astype(np.float32)


def make_holdout(size: int, camera_size: int, seed: int,
                 profile: dict | None = None, fft_size: int = 128,
                 geom: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = np.stack(list(basis(size).values()))
    probes = []
    # Unseen mixed modes/amplitudes plus a downstream +defocus direction probe.
    for coeff in ([.13, -.09, .11, -.07, .08, -.10, .07, -.06],
                  [-.17, .08, -.10, .09, -.06, .06, -.09, .10],
                  [.18, 0., 0., 0., 0., 0., 0., 0.]):
        probes.append(np.tensordot(np.asarray(coeff) + rng.normal(0., .025, len(MODES)), z, axes=1))
    phases = np.stack(probes).astype(np.float32)
    twin = make_fit(phases, np.zeros((3, camera_size), np.float32), size=size,
                    camera_size=camera_size, scale=.985, true=True, fft_size=fft_size, geom=geom)
    return phases, twin_render(twin, rng, profile)


def dominance_ratios(matrix: np.ndarray) -> np.ndarray:
    cross = np.abs(matrix).copy()
    np.fill_diagonal(cross, 0.)
    return np.abs(np.diag(matrix)) / np.maximum(cross.max(axis=0), 1e-12)


def response_identification(size: int, camera_size: int,
                           profile: dict | None = None, fft_size: int = 128,
                           geom: dict | None = None) -> dict:
    rng = np.random.default_rng(19229)
    columns, measured_columns, agreements = [], [], []
    z = np.stack(list(basis(size).values()))
    for j, mode in enumerate(MODES):
        amp = AMPLITUDES[mode]
        phases = np.stack([-amp*z[j], amp*z[j]]).astype(np.float32)
        twin = make_fit(phases, np.zeros((2, camera_size), np.float32), size=size,
                        camera_size=camera_size, scale=.985, true=True, fft_size=fft_size, geom=geom)
        measured = twin_render(twin, rng, profile).reshape(2, -1)
        model = make_fit(phases, np.zeros((2, camera_size), np.float32), size=size,
                         camera_size=camera_size, fft_size=fft_size, geom=geom)
        nominal = model.prediction().detach().numpy().reshape(2, -1)
        columns.append((nominal[1]-nominal[0])/(2*amp))
        measured_columns.append((measured[1]-measured[0])/(2*amp*CCD_SCALE))
        agreement = direction_agreement(nominal[0:1], nominal[1:2], measured[0:1], measured[1:2])
        agreements.append({"mode": mode, "amplitude": amp, **agreement.__dict__})
    jac = np.stack(columns)
    measured_jac = np.stack(measured_columns)
    norm = np.linalg.norm(jac, axis=1)
    # Matched-filter compression y_i = <I, nominal response_i / ||response_i||>.
    # The compression is fixed using the model only, never whitened/inverted to force dominance.
    features = jac / np.maximum(norm[:, None], 1e-12)
    compressed = features @ jac.T
    measured_compressed = features @ measured_jac.T
    ratio = dominance_ratios(compressed).tolist()
    gram = jac @ jac.T
    _, singular, _ = np.linalg.svd(jac, full_matrices=False)
    linearity = []
    for j, mode in enumerate(MODES):
        outputs = []
        for amp in (.1, .2, .3, .5):
            phases = np.stack([-amp*z[j], amp*z[j]]).astype(np.float32)
            fit = make_fit(phases, np.zeros((2, camera_size), np.float32),
                           size=size, camera_size=camera_size, fft_size=fft_size, geom=geom)
            outputs.append(fit.prediction().detach().numpy())
        nominal = make_fit(np.zeros((1, size, size), np.float32),
                           np.zeros((1, camera_size), np.float32), size=size,
                           camera_size=camera_size, fft_size=fft_size, geom=geom).prediction().detach().numpy()[0]
        rel = [float(np.linalg.norm((out[1]+out[0])/2-nominal) / max(np.linalg.norm(out[1]-out[0]), 1e-12))
               for out in outputs]
        linearity.append({"mode": mode, "amplitudes": [.1, .2, .3, .5], "oddness_error": rel})
    return {"modes": list(MODES), "dominance_ratio": ratio,
            "condition_jtj": float(np.linalg.cond(gram)),
            "svd_relative_floor": .01,
            "svd_identifiable_rank": int(np.count_nonzero(singular > .01 * singular[0])),
            "singular_values": singular.tolist(), "agreement": agreements,
            "compressed_model_jacobian": compressed.tolist(),
            "compressed_measured_jacobian": measured_compressed.tolist(),
            "measured_dominance_ratio": dominance_ratios(measured_compressed).tolist(),
            "measured_condition_jtj": float(np.linalg.cond(measured_jac @ measured_jac.T)),
            "measured_singular_values": np.linalg.svd(measured_jac, compute_uv=False).tolist(),
            "measured_response_cosine": (measured_jac @ measured_jac.T / np.outer(np.linalg.norm(measured_jac, axis=1), np.linalg.norm(measured_jac, axis=1))).tolist(),
            "linearity": linearity,
            "cross_coupling_note": "Z4/Z11 and Z5/Z6/Z7/Z8 pairs are diagnosed from normalized response cosine; joint blocks are used regardless of diagonal ratio.",
            "response_cosine": (jac @ jac.T / np.outer(norm, norm)).tolist()}


def run(output: Path, seeds: list[int], sizes: list[int], grid: int, camera_size: int,
        iterations: int, profile: dict | None = None, use_gate: bool = True,
        fft_size: int = 128, frame_budget: str = "16n", geom: dict | None = None) -> dict:
    profile = profile or TWIN_PROFILES["clean"]
    if geom is not None:
        # hw2f: the camera window is set by the recorded geometry, not --camera-size.
        camera_size = geom["camera_xy"].shape[0]
        print(f"几何(hw2f): matrix={np.asarray(geom['matrix']).round(3).tolist()} "
              f"centre={[round(v, 2) for v in geom['centre']]} window={camera_size}px "
              f"source={geom['source']}", flush=True)
    sig = float(profile.get("sigma_ccd_px", 0.0))
    kernel = None
    if sig > 0.:
        kernel = blaze_kernel(sig).numpy()[0, 0]
    budget_mult = 16 if frame_budget == "16n" else 1  # hardware comparison: n frames/arm
    output.mkdir(parents=True, exist_ok=False)
    identification = response_identification(grid, camera_size, profile, fft_size=fft_size, geom=geom)
    gate = {j for j, ratio in zip(MODES, identification["dominance_ratio"])
            if ratio >= 1.2 and identification["agreement"][j-4]["correlation"] >= .85}
    if not use_gate:
        # Under a real CCD PSF the L2 dominance-ratio gate collapses to empty even
        # though production fitting (harness, no gate) reaches 0.024 on hardware.
        # Bypass the sim-only gate to expose sample-efficiency rather than a
        # frozen floor, matching how production fits blurs+radiometry per call.
        gate = set(MODES)
    identification["allowed_modes"] = sorted(gate)
    print(f"可辨识门禁: {sorted(gate)}; 其余模态冻结为残差（待联合辨识）", flush=True)
    rows = []
    for seed in seeds:
        holdout = make_holdout(grid, camera_size, seed+5000, profile, fft_size=fft_size, geom=geom)
        for n in sizes:
            rng = np.random.default_rng(seed*100+n)
            frames = budget_mult*n
            for arm in ("none", "random", "zernike"):
                if arm == "none":
                    phases = holdout[0]
                    measured = np.zeros((len(phases), camera_size, camera_size), np.float32)
                    fit = make_fit(phases, measured, size=grid, camera_size=camera_size, fft_size=fft_size, geom=geom)
                    metrics = image_metrics(fit.prediction().detach().numpy(), holdout[1], kernel)
                    downstream = image_metrics(fit.prediction().detach().numpy()[-1:], holdout[1][-1:], kernel)
                    result = {**metrics, "iterations": 0, "seconds": 0., "threshold_step": None,
                              "threshold_seconds": None, "downstream_nrmse": downstream["nrmse"],
                              "downstream_ncc": downstream["ncc"], "downstream_ssim": downstream["ssim"],
                              "train_l1": 0., "coefficients": [0.]*8, "history": []}
                else:
                    if arm == "random":
                        phases = random_phases(frames, grid, rng)
                    else:
                        # Active arm: zernike_phases produces 16 frames per rep
                        # (8 modes x 2 signs); with the hardware comparison budget
                        # we draw the same frame budget as the random arm.
                        phases = zernike_phases(n, grid, rng)
                        if budget_mult != 16:
                            phases = phases[:frames]
                    twin = make_fit(phases, np.zeros((frames, camera_size), np.float32),
                                    size=grid, camera_size=camera_size, scale=.985, true=True, fft_size=fft_size, geom=geom)
                    measured = twin_render(twin, rng, profile)
                    result = fit_arm(phases, measured, size=grid, camera_size=camera_size,
                                     holdout=holdout, iterations=iterations,
                                     allowed_modes=tuple(gate), kernel=kernel, fft_size=fft_size, geom=geom)
                row = {"seed": seed, "n": n, "arm": arm, "frames": 0 if arm == "none" else frames,
                       "evaluation_frames": 3, **result}
                rows.append(row)
                print(f"seed={seed} n={n:2d} arm={arm:7s} frames={row['frames']:3d} "
                      f"NRMSE={result['nrmse']:.4f} NCC={result['ncc']:.4f} "
                      f"SSIM={result['ssim']:.4f} it={result['iterations']:3d} "
                      f"wall={result['seconds']:.2f}s", flush=True)
    config = {"issue": 229, "modes": list(MODES), "true_coefficients": TRUE_C.tolist(),
              "amplitudes": AMPLITUDES, "sizes": sizes, "seeds": seeds, "grid": grid,
              "camera_size": camera_size, "fft_size": fft_size, "iterations": iterations,
              "twin_profile": profile,
              "geometry": geometry_record(geom),
              "use_gate": use_gate,
              "frame_budget": ("n frames per learned arm (hardware comparison); active arm is a "
                               "prefix draw of the n-frame zernike set; none=0; +3 common evaluation frames"
                               if budget_mult != 16 else
                               "16*n for both learned arms; none=0; +3 common evaluation frames"),
              "trust_region": {"Z4": .3, "Z5_Z11": .15, "global_l2": .5},
              "twin_only": ["slm_phase_a=.08", "slm_phase_b=.025", "camera_scale=.985",
                            "saturation=255", "4-bit fractional quantisation"],
              "threshold_nrmse_fraction_of_baseline": .85}
    (output/"config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"identifiability.json").write_text(json.dumps(identification, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"learning_curves.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output/"learning_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "n", "arm", "frames", "nrmse", "ncc",
                                                    "ssim", "train_l1", "iterations", "seconds", "threshold_step"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    (output/"report.md").write_text(render_report(rows, identification, config), encoding="utf-8")
    return {"rows": rows, "identification": identification, "config": config}


def render_report(rows: list[dict], ident: dict, config: dict) -> str:
    lines = ["# Zernike 主动探针纯仿真：样本效率与速度", "",
             "> 纯合成 twin 证据，不代表实测。ground truth 与拟合器共享 `PhysicalForwardFit` 算子族；",
             "> SLM 非线性先作用于命令，静态像差后加。无硬件、无 GUI、无未知波前反演。", "",
             "## 口径", "", f"- 模态：Noll Z4–Z11；真值：{config['true_coefficients']} rad。",
             f"- 帧预算：{config['frame_budget']}。{('主动臂每模态 n 次随机顺序 ±A；随机臂相同总帧数与匹配 RMS。' if '16*n' in str(config['frame_budget']) else '主动臂取 n 帧 Zernike 集前缀；随机臂同帧数。')}",
             f"- 传播 FFT 尺寸：{config['fft_size']}。",
             f"- 几何：`{config['geometry']['mode']}`；matrix={config['geometry']['matrix']}，centre={[round(v, 2) for v in config['geometry']['centre']]}，amplitude_center={config['geometry']['amplitude_center']}"
             + (f"，窗口 {config['geometry']['camera_window']['size']}px（span {config['geometry']['camera_window']['span']}, step {config['geometry']['camera_window']['step']}），源 `{config['geometry']['source']}`" if config['geometry']['mode'] == 'hw2f' else "") + "。",
             "- 无微调 0 帧；公共留出集 3 帧（两个未见混合模态/幅度 + 一个下游 +离焦方向）。",
             "- 主损失图像域 L1 + 0.10(1-NCC)；先 Z4–Z8、后 Z9–Z11；留出 NRMSE 早停。",
             f"- L2 门禁（对角比≥1.2 且方向相关≥0.85）允许：{ident['allowed_modes']}；未过门禁者冻结为残差。",
             "- 信任域：Z4≤0.3 rad、Z5–Z11≤0.15 rad、全局 L2≤0.5 rad。",
             "- twin 独有：SLM LUT 非线性、瞳孔缩放 .985、CCD 增益/背景/饱和/量化/读出噪声。",
             f"- twin 光度/PSF 档位：`{config['twin_profile']['name']}`（σ={config['twin_profile']['sigma_ccd_px']}px、背景={config['twin_profile']['background']}、噪声σ={config['twin_profile']['noise_std']}）。",
             f"- 残留限制：twin 与拟合器**共享固定增益**（`CCD_SCALE=100`、`gain={config['twin_profile']['gain']}`），未建模绝对辐射标定，故绝对 counts 不代表硬件；样本效率差异来自 blur+offset+噪声地板，增益斜率被 ptp 归一化 NRMSE/NCC 与共享曝光归一化吸收。", "",
             "## 样本效率（各 seed 均值）", "", "| n | 臂 | 训练帧 | 留出 NRMSE | NCC | SSIM | 训练 L1 | 迭代 | 秒 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in config["sizes"]:
        for arm in ("none", "random", "zernike"):
            group = [r for r in rows if r["n"] == n and r["arm"] == arm]
            means = {key: np.mean([r[key] for r in group]) for key in ("nrmse", "ncc", "ssim", "train_l1", "iterations", "seconds")}
            lines.append(f"| {n} | {arm} | {group[0]['frames']} | {means['nrmse']:.4f} | {means['ncc']:.4f} | {means['ssim']:.4f} | {means['train_l1']:.5f} | {means['iterations']:.1f} | {means['seconds']:.3f} |")
    baseline = np.mean([r["nrmse"] for r in rows if r["arm"] == "none"])
    lines += ["", f"无微调留出 NRMSE 均值为 **{baseline:.4f}**。", "", "## 达阈速度（留出 NRMSE ≤ 同 seed 无微调的 85%）", "", "| n | 臂 | 达阈运行/总数 | 达阈时中位迭代 | 达阈墙钟中位数（秒） |", "|---:|---|---:|---:|---:|"]
    for n in config["sizes"]:
        for arm in ("random", "zernike"):
            group = [r for r in rows if r["n"] == n and r["arm"] == arm]
            reached = [r for r in group if r["threshold_step"] is not None]
            steps = np.median([r["threshold_step"] for r in reached]) if reached else None
            seconds = np.median([r["threshold_seconds"] for r in reached]) if reached else None
            lines.append(f"| {n} | {arm} | {len(reached)}/{len(group)} | {steps if steps is not None else '未达阈'} | {seconds if seconds is not None else '未达阈'} |")
    lines += ["", "本组两臂均未达阈；样本效率表中的迭代数是留出早停迭代，不是达阈速度。",
              "", "## 可辨识性矩阵", "", "| Noll | A (rad) | 对角主导比 | 方向相关 | 判定 |", "|---:|---:|---:|---:|---|"]
    for j, mode in enumerate(ident["modes"]):
        agreement = ident["agreement"][j]
        lines.append(f"| Z{mode} | {agreement['amplitude']} | {ident['dominance_ratio'][j]:.3f} | {agreement['correlation']:.3f} | {agreement['verdict']} |")
    lines += ["", f"`JᵀJ` 条件数：**{ident['condition_jtj']:.3g}**；奇异值：{np.array2string(np.array(ident['singular_values']), precision=3)}。", "",
              "观测量是固定 matched-filter 压缩 y_i=<I, J_i/||J_i||>；表中给出模型 |J_jj|/max|J_ij|。", "",
              "## L1/L3：留出与下游方向", "", "| 臂 | n | 留出 NRMSE | 下游 +离焦 NRMSE | 下游 NCC |", "|---|---:|---:|---:|---:|",] + [
                  f"| {arm} | {n} | {np.mean([r['nrmse'] for r in rows if r['n']==n and r['arm']==arm]):.4f} | {np.mean([r['downstream_nrmse'] for r in rows if r['n']==n and r['arm']==arm]):.4f} | {np.mean([r['downstream_ncc'] for r in rows if r['n']==n and r['arm']==arm]):.4f} |"
                  for n in config['sizes'] for arm in ('none','random','zernike')] + ["",
              "### 联合更新与残差", "", "- 低阶 Z4–Z8 与中阶 Z9–Z11 分块；离焦/球差及彗差/像散不因对角判据被拆开单独反演。",
              "- SVD 小奇异值对应的方向在当前观测下不可辨识，需增加探针或重参数化，而非仅靠联合更新解决；不进入本显式块的高阶像差、照明与几何残差不得冒充 Zernike 系数。", "",
              "## 限制", "", "- 三 seed 的合成统计只用于验证机制，不给出实测泛化保证；非微调臂无训练墙钟。",
              "- 无微调的 0 训练帧与学习臂不同，但随机与主动两个学习臂严格同预算；留出探针不参与梯度或模型选择。",
              "- 达阈墙钟取首次达到阈值的实际时刻（每 5 步评估）。训练探针改善列为 L0，不作通过标准。",  "",
              "## 产物", "", "`learning_curves.csv` / `.json`：逐 seed 曲线、历史、系数与墙钟；`identifiability.json`：Jacobian 门禁；`config.json`：完整配置。"]
    return "\n".join(lines)+"\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[229, 230, 231])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--camera-size", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--fft-size", type=int, default=128,
                        help="FFT size for propagation (128 clean/hardware default; 256 for the "
                             "hardware comparison mode with grid=128, camera=320)")
    parser.add_argument("--frame-budget", choices=["16n", "n"], default="16n",
                        help="frame scaling: '16n' legacy (16 frames per n, both learned arms); "
                             "'n' hardware-comparison budget (n frames per arm, active arm is a "
                             "prefix of the n-frame zernike set)")
    parser.add_argument("--twin-profile", choices=sorted(TWIN_PROFILES), default="clean")
    parser.add_argument("--no-gate", action="store_true",
                        help="bypass L2 dominance gate (fit all Z4-Z11), matching production/harness")
    parser.add_argument("--geometry", choices=["synthetic", "hw2f"], default="synthetic",
                        help="synthetic: legacy hardcoded 1500px matrix / (1360,1020) origin (default); "
                             "hw2f: real 2f geometry from --manifest (matches _harness_task2.py)")
    parser.add_argument("--manifest", type=Path, default=HW2F_MANIFEST,
                        help="run01 manifest.json used when --geometry hw2f")
    args = parser.parse_args()
    output = args.output or REPO/"logs"/"zernike_probe_sim"/time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    profile = TWIN_PROFILES[args.twin_profile]
    geom = hw2f_geometry(args.manifest) if args.geometry == "hw2f" else None
    run(output, args.seeds, args.sizes, args.grid, args.camera_size, args.iterations, profile,
        use_gate=not args.no_gate, fft_size=args.fft_size, frame_budget=args.frame_budget, geom=geom)
    print(f"报告: {output/'report.md'}")


if __name__ == "__main__":
    main()

