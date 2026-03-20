"""生成不同大气环境下的光斑与相位对比图。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.drivers.sim.atmos import SimulatedTurbulentScreen
from ao_shaping.drivers.sim.wave import WaveGenerator, WavePropagator


@dataclass
class AtmosphereCase:
    name: str
    cn2: float
    l0: float
    l_max: float
    distance: float


def simulate_case(case: AtmosphereCase, npix: int = 256, dpix: float = 20e-6) -> tuple[np.ndarray, np.ndarray]:
    """模拟单个大气环境并返回相位屏与光斑强度。"""
    generator = WaveGenerator(
        npix=npix,
        dpix=dpix,
        wavelength=1064e-9,
        aperture=2.5e-3,
        beam_type="gaussian",
        random_seed=42,
    )
    propagator = WavePropagator(prop_dist=case.distance)
    screen = SimulatedTurbulentScreen(
        dist=case.distance,
        Cn2=case.cn2,
        L0=case.l_max,
        l0=case.l0,
        harmonic=1,
    )

    generator.open()
    propagator.open()
    screen.open()

    wave = generator.generate()
    screen.process(wave)
    propagator.propagate(wave)

    phase = screen.get_opd()
    if phase is None:
        phase = np.angle(wave.wavefront)
    intensity = wave.intensity

    generator.close()
    propagator.close()
    screen.close()
    return phase, intensity


def render_comparison(cases: list[AtmosphereCase], output_path: Path) -> None:
    """绘制对比图并保存。"""
    fig, axes = plt.subplots(2, len(cases), figsize=(5 * len(cases), 8), constrained_layout=True)

    for idx, case in enumerate(cases):
        phase, intensity = simulate_case(case)

        ax_phase = axes[0, idx]
        im_phase = ax_phase.imshow(phase, cmap="twilight", origin="lower")
        ax_phase.set_title(f"{case.name}\nPhase (Cn²={case.cn2:.1e})")
        ax_phase.set_axis_off()
        fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)

        ax_spot = axes[1, idx]
        im_spot = ax_spot.imshow(intensity, cmap="inferno", origin="lower")
        ax_spot.set_title(f"{case.name}\nSpot Intensity")
        ax_spot.set_axis_off()
        fig.colorbar(im_spot, ax=ax_spot, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Atmospheric Propagation Comparison", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    cases = [
        AtmosphereCase("Weak Turbulence", cn2=1e-16, l0=2e-3, l_max=30.0, distance=500.0),
        AtmosphereCase("Moderate Turbulence", cn2=5e-15, l0=1e-3, l_max=20.0, distance=1000.0),
        AtmosphereCase("Strong Turbulence", cn2=5e-14, l0=5e-4, l_max=10.0, distance=1500.0),
    ]
    render_comparison(cases, Path("docs/simulation/atmospheric_spot_phase_comparison.png"))
