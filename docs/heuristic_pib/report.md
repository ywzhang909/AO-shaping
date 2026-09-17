# Heuristic PIB Benchmark Report

Benchmark of all 7 heuristic optimizers in `ao_shaping.algorithm` on the PIB (power-in-bucket) optimization problem, offline (pure numpy, no hardware), using the synthetic landscape from `ao_shaping.optimizer.wfless.pib_sim_eval.SimLandscape`.

- **dim**: 4, **bounds**: (-12.0, 12.0), **seed**: 42
- **init_x**: [0.0, 0.0, 0.0, 0.0] (zero voltage), **init PIB**: 0.0034
- **global center**: [-4.8, -4.2, -4.5, -4.0] (PIB = 1.0)
- **local center**: [2.5, 3.2, 2.2, 2.8] (PIB ~ 0.9)

## Results

| Algorithm | Final PIB | Improvement | Loads to max | Loads ≥ 0.9 | Loads ≥ 0.5 | n_loads | Interpretation |
|-----------|-----------|-------------|--------------|-------------|-------------|---------|----------------|
| PSO | 1.0000 | +0.9966 | 3956 | 91 | 61 | 12030 | found global basin |
| DIFFERENTIAL_EVOLUTION | 1.0000 | +0.9966 | 2330 | 284 | 126 | 9030 | found global basin |
| GA | 1.0000 | +0.9966 | 8786 | 234 | 83 | 9030 | found global basin |
| CROSS_ENTROPY | 1.0000 | +0.9966 | 8989 | 92 | 58 | 9000 | found global basin |
| HILL_CLIMBING | 1.0000 | +0.9966 | 2741 | 176 | 137 | 4001 | found global basin |
| SA | 0.9892 | +0.9858 | 5955 | 5907 | 2676 | 6001 | found global basin |
| RANDOM_SEARCH | 0.9079 | +0.9045 | 2499 | 2499 | 80 | 6001 | found global basin |

## Algorithm Principles

All 7 algorithms optimize the scalar fitness function `fitness(v) = -PIB(v)` (minimize negative PIB = maximize PIB) over the 4-D coefficient space `v`, with bounds (-12, 12). Their working principles:

| Algorithm | Principle |
|-----------|-----------|
| **GA** (Genetic Algorithm) | Population-based evolutionary search: maintains a population of candidate solutions, applies selection (tournament), crossover and mutation each generation, keeps elite individuals, and iterates toward the optimum. |
| **PSO** (Particle Swarm) | Swarm intelligence: each particle moves through the search space with a velocity updated toward its own personal best (`c1`) and the swarm's global best (`c2`) positions. |
| **SA** (Simulated Annealing) | Probabilistic local search: starts at a high temperature and accepts *worse* moves with probability `exp(-Δf/T)`, gradually cooling so the search first escapes local minima then fine-tunes. |
| **HC** (Hill Climbing) | Greedy local search: perturbs the current best candidate, accepts only strictly improving moves — fast but can get stuck in local minima. |
| **RS** (Random Search) | Uniformly samples candidate points across the whole bounded space and keeps the best seen — a no-gradient baseline. |
| **CEM** (Cross-Entropy) | Model-based search: samples candidates from a parameterized Gaussian, keeps elite samples, and re-fits the Gaussian mean/covariance to the elites each iteration. |
| **DE** (Differential Evolution) | Population-based differential mutation: builds candidate vectors by adding scaled differences of other population members to a base vector, then crosses and greedily selects. |

## Interpretation

Each algorithm's `best_x` is compared against the global center [-4.8, -4.2, -4.5, -4.0] and the local center [2.5, 3.2, 2.2, 2.8].
PIB >= ~0.9 with `best_x` near the global center = found the global basin; PIB ~0.6-0.9 = found the local basin; below that = did not converge.

**Convergence speed** (`Iters to max` / `Iters ≥ 0.9` / `Iters ≥ 0.5`): the first iteration at which the PIB curve reaches its final maximum, crosses 0.9, and crosses 0.5, respectively (convergence history is best-so-far, so the curves are monotone). '—' means the threshold was never reached within the iteration budget.

This report is generated offline by `scripts/generate_heuristic_pib_report.py` and can be regenerated at any time without hardware.
