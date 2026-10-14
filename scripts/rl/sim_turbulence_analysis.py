from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.drivers.sim.compat import AOConfig, TraditionalAOSystem


def _make_config(cn2: float) -> AOConfig:
    return AOConfig(
        N=128,
        L=0.04,
        Cn2=cn2,
        L0=20.0,
        l0=1e-3,
        dm_actuators=4,
        subapertures=4,
        pixel_scale=0.5,
        propagation_distance=1000.0,
    )


def _peak_ratio(image: np.ndarray) -> float:
    image_f = image.astype(float)
    return float(np.max(image_f) / np.sum(image_f))


def generate_analysis_figure(output_path: Path) -> Path:
    levels = [
        ("No turbulence", 0.0, 0),
        ("Weak", 1e-14, 1),
        ("Medium", 5e-14, 2),
        ("Strong", 1e-13, 3),
    ]

    fig, axes = plt.subplots(len(levels), 3, figsize=(12, 12), constrained_layout=True)

    for row, (label, cn2, seed) in enumerate(levels):
        np.random.seed(seed)
        ao = TraditionalAOSystem(_make_config(cn2))
        state = ao.reset()
        phase = ao.turbulence.phase_screen
        image = state["image"].astype(float)
        image_norm = image / max(float(np.max(image)), 1.0)

        metrics_text = (
            f"Cn2={cn2:.1e}\n"
            f"phase std={np.std(phase):.3f}\n"
            f"strehl={state['strehl']:.3f}\n"
            f"peak ratio={_peak_ratio(image):.3f}"
        )

        axes[row, 0].imshow(phase, cmap="coolwarm")
        axes[row, 0].set_title(f"{label} phase")
        axes[row, 1].imshow(image_norm, cmap="inferno")
        axes[row, 1].set_title(f"{label} focal intensity")
        axes[row, 2].axis("off")
        axes[row, 2].text(0.02, 0.98, metrics_text, va="top", ha="left", fontsize=11, family="monospace")

        for col in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("AO sim turbulence validation", fontsize=16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    target = Path("artifacts") / "sim_turbulence_analysis.png"
    saved = generate_analysis_figure(target)
    print(saved)
