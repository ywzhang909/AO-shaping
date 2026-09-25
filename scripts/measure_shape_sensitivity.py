"""Measure the sensitivity (noise floor) of the ``slm-pib`` shaping objective.

SPGD estimates its gradient from ``J(a+Δa) - J(a-Δa)``. If the metric change that
a ``±Δa`` phase perturbation produces is below the camera measuring noise, the
gradient direction is pure noise and no search can converge. This script
quantifies exactly that, on the real bench:

1. **Noise floor** ``ΔJ_noise`` — capture ``--n-frames`` frames of the *same*
   fixed phase, evaluate the shaping score on each, report the standard deviation.
2. **Signal** ``ΔJ_signal`` — for every ``Δa`` in ``--deltas``, write
   ``c + Δa`` and ``c - Δa`` (first Zernike mode, Noll 4 defocus, no tilt),
   average ``--n-repeat`` frame pairs (guidance: "双扰动平均"), and take
   ``|mean(J+) - mean(J-)|`` — the exact quantity SPGD turns into a gradient.
3. **Verdict** — ``SNR = ΔJ_signal / ΔJ_noise``: ``>= 3`` strong, ``>= 2``
   usable, ``< 2`` unusable (raise ``Δa`` to 0.1-0.2 rad or average more frames).

Writes ``sensitivity.json`` + ``sensitivity.md`` (+ ``sensitivity.png``) into the
output directory so the numbers can be cited in the benchmark report.

Usage::

    $env:PYTHONPATH = "src;libs"
    python scripts/measure_shape_sensitivity.py
    python scripts/measure_shape_sensitivity.py --deltas 0.02,0.05,0.1,0.2 -o docs/slm_pib_heuristic_hw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

matplotlib_used = False
try:  # plotting is optional - the numbers matter most
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib_used = True
except ImportError:  # pragma: no cover - matplotlib is a project dependency
    plt = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.utils.io.file import logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-type", default="daheng", help="camera backend")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--cam-size", type=int, default=320, help="camera window (px)")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--n-max", type=int, default=4, help="Zernike order")
    parser.add_argument("--exposure-ms", type=float, default=0.0, help="0 = auto-expose")
    parser.add_argument("--target-brightness", type=float, default=180.0)
    parser.add_argument("--settle-s", type=float, default=0.3, help="SLM settle time")
    parser.add_argument("--n-frames", type=int, default=10, help="frames for the noise floor")
    parser.add_argument("--n-repeat", type=int, default=3, help="+/- pairs averaged (guidance: 3)")
    parser.add_argument(
        "--deltas",
        default="0.05,0.1,0.2",
        help="phase perturbation amplitudes in rad (comma separated)",
    )
    parser.add_argument("--target-aspect-ratio", type=float, default=4.0 / 3.0)
    parser.add_argument("-o", "--output", default="docs/slm_pib_heuristic_hw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ao_shaping.drivers.ccd.common import (
        auto_exposure,
        create_camera,
        get_camera_exposure_ms,
        set_camera_exposure_ms,
    )
    from ao_shaping.drivers.slm import Santec
    from ao_shaping.optimizer.wfless.slm_zernike_pib import (
        ImageTargetFunc,
        PatternHelper,
        _display,
        _zernike_to_phase,
        shape_metric,
    )
    from ao_shaping.utils.image.beam_metrics import (
        clamp_center_to_frame,
        zero_order_center,
    )

    deltas = [float(d) for d in str(args.deltas).split(",") if d.strip()]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = create_camera(args.cam_type, cam_id=args.cam_id, exposure_time_ms=3.0)
    cam.open()
    slm = Santec(slm_number=args.slm_number, wavelength=args.wavelength)
    slm.open()
    try:
        ph = PatternHelper(resolution=(1920, 1200), bits=10)
        coeff = np.zeros(15, dtype=np.float64)

        def show(c: np.ndarray) -> None:
            _display(slm, slm.create_phase_from_array(_zernike_to_phase(c, args.n_max, ph)))
            time.sleep(args.settle_s)

        show(coeff)

        if args.exposure_ms > 0:
            set_camera_exposure_ms(cam, args.exposure_ms)
            img = cam.get_numpy_image(2)
        else:
            img = auto_exposure(cam, args.target_brightness)
        exp = float(get_camera_exposure_ms(cam))
        logger.info("exposure = {:.3f}ms, frame max = {}", exp, int(np.max(img)))

        # Fix the centre on the settled flat phase, then window (same as the benchmark).
        pts = np.array(
            [zero_order_center(cam.get_numpy_image(2)) for _ in range(12)],
            dtype=np.float64,
        )
        center = (
            int(round(float(np.median(pts[:, 0])))),
            int(round(float(np.median(pts[:, 1])))),
        )
        center = clamp_center_to_frame(center, img.shape, args.cam_size)
        window, _ = cam.reset_window(center, (args.cam_size, args.cam_size))
        _w, _h = window
        first = cam.get_numpy_image(4)
        center = zero_order_center(first)
        logger.info("window = {}x{}, fixed centre = {}", _w, _h, center)

        # Target size: same derivation (and long-side clamp) as the optimizer.
        radius = ImageTargetFunc(_w, _h, center).radius(first, energy=0.99)
        fit = min(float(_h), float(_w) / float(args.target_aspect_ratio))
        target_size = float(min(max(2.0 * float(radius), 4.0), fit))
        logger.info(
            "target size = {:.1f}px (2x99% radius = {:.1f}, window fit = {:.1f})",
            target_size,
            2.0 * float(radius),
            fit,
        )

        def score(image: np.ndarray) -> float:
            return shape_metric(
                image,
                center,
                center,
                "rectangle",
                target_size,
                args.target_aspect_ratio,
            )[0]

        # --- 1) noise floor: same phase, repeated frames -------------------
        noise_scores = [score(cam.get_numpy_image(2)) for _ in range(args.n_frames)]
        noise = float(np.std(np.asarray(noise_scores, dtype=np.float64)))
        logger.info(
            "noise floor: J = {:.5f} +/- {:.5f} (std over {} frames)",
            float(np.mean(noise_scores)),
            noise,
            args.n_frames,
        )

        # --- 2) signal: +/-delta, averaged over repeats --------------------
        results = []
        defocus_index = 3
        for delta in deltas:
            pos, neg = [], []
            for _ in range(args.n_repeat):
                coeff[defocus_index] = delta
                show(coeff)
                pos.append(score(cam.get_numpy_image(2)))
                coeff[defocus_index] = -delta
                show(coeff)
                neg.append(score(cam.get_numpy_image(2)))
            coeff[defocus_index] = 0.0
            show(coeff)
            signal = float(abs(np.mean(pos) - np.mean(neg)))
            snr = signal / noise if noise > 0 else float("inf")
            verdict = "强" if snr >= 3.0 else ("可用" if snr >= 2.0 else "不可用")
            results.append(
                {
                    "delta_rad": delta,
                    "J_pos": float(np.mean(pos)),
                    "J_neg": float(np.mean(neg)),
                    "signal": signal,
                    "snr": snr,
                    "verdict": verdict,
                }
            )
            logger.info(
                "delta = {:.3f} rad -> dJ = {:.5f}, SNR = {:.2f} ({})",
                delta,
                signal,
                snr,
                verdict,
            )

        payload = {
            "exposure_ms": exp,
            "cam_size": args.cam_size,
            "window": [int(_w), int(_h)],
            "center": [int(center[0]), int(center[1])],
            "target_size": target_size,
            "n_frames": args.n_frames,
            "n_repeat": args.n_repeat,
            "noise_floor": noise,
            "noise_mean": float(np.mean(noise_scores)),
            "noise_scores": [float(s) for s in noise_scores],
            "results": results,
        }
        (out_dir / "sensitivity.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf8"
        )

        best = max(results, key=lambda r: r["snr"])
        lines = [
            "# 整形目标函数灵敏度 (噪声地板) 实测",
            "",
            f"- 曝光 {exp:.3f} ms, 开窗 {int(_w)}x{int(_h)}, 固定中心 {tuple(int(v) for v in center)}",
            f"- 目标尺寸 {target_size:.1f} px (2x99% 能量半径 {2.0 * float(radius):.1f}, 开窗长边上限 {fit:.1f})",
            f"- **噪声地板 dJ_noise = {noise:.5f}** ({args.n_frames} 帧同相位, 均值 {np.mean(noise_scores):.5f})",
            "",
            "| Δa (rad) | J(+Δa) | J(−Δa) | ΔJ | SNR | 判定 |",
            "|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {r['delta_rad']:.3f} | {r['J_pos']:.5f} | {r['J_neg']:.5f} | "
                f"{r['signal']:.5f} | {r['snr']:.2f} | {r['verdict']} |"
            )
        lines += [
            "",
            f"最佳: **Δa = {best['delta_rad']:.3f} rad, SNR = {best['snr']:.2f}** ({best['verdict']})。",
            "",
            "判据 (资料 §三.1): `ΔJ_signal >= 2~3 x ΔJ_noise` 才可分辨; SNR < 2 时须加大 Δa "
            "(0.1~0.2 rad) 或对 ±Δa 各多帧平均。",
        ]
        (out_dir / "sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf8")

        if matplotlib_used and plt is not None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            xs = [r["delta_rad"] for r in results]
            ys = [r["signal"] for r in results]
            ax.plot(xs, ys, "o-", label="ΔJ_signal")
            ax.axhline(2 * noise, ls="--", c="orange", label="2x noise floor")
            ax.axhline(3 * noise, ls=":", c="red", label="3x noise floor")
            ax.set_xlabel("Δa (rad)")
            ax.set_ylabel("ΔJ")
            ax.set_title("整形目标灵敏度 vs 扰动幅度")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "sensitivity.png", dpi=150)
            plt.close(fig)

        logger.info("sensitivity written to {}", out_dir)
    finally:
        slm.close()
        cam.close()


if __name__ == "__main__":
    main()
