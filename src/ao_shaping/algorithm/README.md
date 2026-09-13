# ao_shaping.algorithm: Optimizer Convention

## Purpose

Every optimizer added to `ao_shaping.algorithm` MUST follow the class-based API described below. This document is the canonical statement of that convention; the repo-root `AGENTS.md` points here as the authoritative source. New optimizers that do not follow this convention will be rejected in review.

## Class-based optimizer principle

An optimizer is a **stateful class**. Construct it once, then drive it step by step. The class owns all state, all validation, and all loop logic.

### `__init__` — validate and set up state

The constructor validates every input and raises `ValueError` early for anything invalid (wrong dimensionality, negative values, shape mismatches). It then sets up all state on `self`: normalized targets, parameter tensors, the internal optimizer, counters, and history. No work is deferred into the loop; after construction the instance is ready to step.

### `update()` — exactly one step

`update()` performs exactly ONE optimization step and returns the next state/solution (for example the next phase map). It is the idempotent unit of work: the caller drives the loop, calling `update()` as many times as needed. The method must be safe to call repeatedly and must not mutate external state.

### optional `run()` — convenience loop

`run()` is an optional convenience method that loops over `update()` (with logging, early stopping, and progress reporting) and returns a **result dataclass** such as `BeamOptimizeResult`. It exists for callers who want the whole loop in one call; it must not contain logic that `update()` cannot express.

## One-shot function rule

The free function is kept ONLY as a thin wrapper that delegates to a class instance, for backward compatibility. It constructs the class and calls `run()`, forwarding arguments. Never add new logic to the wrapper; new features go to the class. If the class gains a parameter, the wrapper forwards it; if the class gains behavior, the wrapper inherits it for free.

## Simulation-first testing rule

torch/numpy tests on synthetic targets are REQUIRED before any hardware run. Tests must never import hardware or SDK modules. CPU is the default test device; CUDA is never hard-required (tests must pass on a machine without a GPU). A test that needs real hardware uses `pytest.skip("Requires hardware")` and is never part of the default suite.

## Canonical example

[`differentiable_beam.py`](./differentiable_beam.py) is the reference implementation. `DifferentiableBeamOptimizer` demonstrates the full convention: validation in `__init__`, one-step `update()`, a `run()` loop returning `BeamOptimizeResult`, and a thin `differentiable_beam_optimize()` wrapper.

```python
class DifferentiableBeamOptimizer:
    def __init__(
        self,
        target_intensity: np.ndarray,
        source_amplitude: np.ndarray | None = None,
        lr: float = 0.01,
        init_phase: np.ndarray | None = None,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        # validate inputs (raise ValueError early), set up all state
        ...

    def update(self) -> np.ndarray:
        # exactly ONE step; returns the next phase (radians)
        ...

    def run(
        self,
        epochs: int = 500,
        early_stop_patience: int = 0,
        early_stop_delta: float = 1e-5,
        log_every: int = 50,
    ) -> BeamOptimizeResult:
        # convenience loop over update(); returns a result dataclass
        ...
```

## Convention checklist

- [ ] Validation in `__init__`: raise `ValueError` early for invalid inputs.
- [ ] State on `self`: all parameters, tensors, counters, and history live on the instance.
- [ ] Deterministic seeding via a `seed` argument (reproducible random initialization).
- [ ] One-step `update()` returning the next state/solution.
- [ ] `run()` returns a result dataclass (like `BeamOptimizeResult`).
- [ ] Thin wrapper for the one-shot function: construct the class, call `run()`, no new logic.
- [ ] Simulation-first tests: synthetic targets, no hardware/SDK imports, CPU default.

## Reference test suites

- [`tests/ao_shaping/algorithm/test_differentiable_beam_optimizer.py`](../../../tests/ao_shaping/algorithm/test_differentiable_beam_optimizer.py) — exercises the class API (`__init__` validation, `update()`, `run()`, result dataclass).
- [`tests/ao_shaping/algorithm/test_differentiable_beam_sim.py`](../../../tests/ao_shaping/algorithm/test_differentiable_beam_sim.py) — simulation-first backprop tests on synthetic targets.