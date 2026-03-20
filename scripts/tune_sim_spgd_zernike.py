from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ao_shaping.optimizer.wfless.sim_spgd import optimize_spgd_zernike


@dataclass
class TuningResult:
    optimizer: str
    gamma: float
    delta: float
    beta1: float
    beta2: float
    beta3: float
    epochs: int
    final_pib: float
    final_strehl: float
    pib_ratio: float
    elapsed_s: float
    score: float


def evaluate_combo(
    optimizer: str,
    gamma: float,
    delta: float,
    beta1: float,
    beta2: float,
    beta3: float,
    epochs: int,
    seed: int,
) -> TuningResult:
    start = time.perf_counter()
    recorder = optimize_spgd_zernike(
        epochs=epochs,
        n_max=5,
        r_bucket=0,
        n_grid=64,
        Cn2=1e-9,
        optimizer_type=optimizer,
        gamma=gamma,
        delta=delta,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        seed=seed,
        use_momentum=False,
    )
    elapsed = time.perf_counter() - start

    init = recorder.history[0]
    final = recorder.history[-1]
    pib_ratio = float(final["pib"] / (init["pib"] + 1e-12))
    final_strehl = float(final["strehl"])

    # 综合效率与光斑质量：提升倍数 + Strehl，惩罚耗时
    score = pib_ratio * (0.7 + 0.3 * final_strehl) / (1.0 + 0.15 * elapsed)

    return TuningResult(
        optimizer=optimizer,
        gamma=gamma,
        delta=delta,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        epochs=epochs,
        final_pib=float(final["pib"]),
        final_strehl=final_strehl,
        pib_ratio=pib_ratio,
        elapsed_s=elapsed,
        score=float(score),
    )


def main() -> None:
    seed = 42
    epochs = 60

    spgd_grid = {
        "gamma": [5e-4, 1e-3, 2e-3],
        "delta": [0.05, 0.08, 0.12],
    }
    adamod_grid = {
        "gamma": [2e-3, 5e-3, 1e-2],
        "delta": [0.05, 0.08, 0.12],
        "beta1": [0.9],
        "beta2": [0.99],
        "beta3": [0.999, 0.9995],
    }

    results: list[TuningResult] = []

    for gamma, delta in itertools.product(spgd_grid["gamma"], spgd_grid["delta"]):
        results.append(
            evaluate_combo(
                optimizer="spgd",
                gamma=gamma,
                delta=delta,
                beta1=0.9,
                beta2=0.99,
                beta3=0.9995,
                epochs=epochs,
                seed=seed,
            )
        )

    for gamma, delta, beta3 in itertools.product(
        adamod_grid["gamma"],
        adamod_grid["delta"],
        adamod_grid["beta3"],
    ):
        results.append(
            evaluate_combo(
                optimizer="adamod",
                gamma=gamma,
                delta=delta,
                beta1=0.9,
                beta2=0.99,
                beta3=beta3,
                epochs=epochs,
                seed=seed,
            )
        )

    best_spgd = max((r for r in results if r.optimizer == "spgd"), key=lambda x: x.score)
    best_adamod = max((r for r in results if r.optimizer == "adamod"), key=lambda x: x.score)

    output = {
        "meta": {
            "seed": seed,
            "epochs": epochs,
            "n_grid": 64,
            "Cn2": 1e-9,
            "objective": "score = pib_ratio*(0.7+0.3*strehl)/(1+0.15*time_s)",
        },
        "best_spgd": asdict(best_spgd),
        "best_adamod": asdict(best_adamod),
        "all_results": [asdict(r) for r in sorted(results, key=lambda x: x.score, reverse=True)],
    }

    out_path = Path("docs/simulation/sim_spgd_zernike_tuning.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Best SPGD:")
    print(json.dumps(asdict(best_spgd), indent=2))
    print("Best AdaMOD:")
    print(json.dumps(asdict(best_adamod), indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
