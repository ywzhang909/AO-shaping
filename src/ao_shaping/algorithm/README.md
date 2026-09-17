# ao_shaping.algorithm: Optimizer Convention

## Purpose

Every optimizer added to `ao_shaping.algorithm` MUST follow the class-based API described below. This document is the canonical statement of that convention; the repo-root `AGENTS.md` points here as the authoritative source. New optimizers that do not follow this convention will be rejected in review.

## Class-based optimizer principle

An optimizer is a **stateful class**. Construct it once, then drive it step by step. The class owns all state, all validation, and all loop logic.

### `__init__` — validate and set up state

The constructor validates every input and raises `ValueError` early for anything invalid (wrong dimensionality, negative values, shape mismatches). It then sets up all state on `self`: normalized targets, parameter tensors, the internal optimizer, counters, and history. No work is deferred into the loop; after construction the instance is ready to step.

### `update()` — exactly one step

`update()` performs exactly ONE optimization step and returns the next state/solution (for example the next phase map). It is the idempotent unit of work: the caller drives the loop, calling `update()` as many times as needed. The method must be safe to call repeatedly and must not mutate external state.

### Loop logic lives in the optimizer layer

The loop that drives `update()` (logging, early stopping, progress reporting, per-step recorder history and best-phase tracking) lives in the **optimizer layer**, not in the algorithm class. For `DifferentiableBeamOptimizer` the pib-style top-level function `optimize_beam_shaping` in `ao_shaping.optimizer.wfless.differentiable_beam` constructs the class and drives `update()` in a loop, returning a `Recorder` (see `optimize_pib` in `ao_shaping.optimizer.wfless.pib` for the same pattern).

## One-shot function rule

Within the `algorithm` package the class is the **sole public API** — there is no one-shot free function; anything more granular uses `update()` directly. Do not reintroduce a free-function wrapper inside `algorithm`: new features go to the class. Callers who want the whole loop in one call use the optimizer-layer function (e.g. `optimize_beam_shaping` in `ao_shaping.optimizer.wfless.differentiable_beam`), which constructs the class and drives `update()`.

## Simulation-first testing rule

torch/numpy tests on synthetic targets are REQUIRED before any hardware run. Tests must never import hardware or SDK modules. CPU is the default test device; CUDA is never hard-required (tests must pass on a machine without a GPU). A test that needs real hardware uses `pytest.skip("Requires hardware")` and is never part of the default suite.

## Canonical example

[`differentiable_beam.py`](./differentiable_beam.py) is the reference implementation. `DifferentiableBeamOptimizer` demonstrates the full convention: validation in `__init__` and one-step `update()`. The loop that drives `update()` lives in the optimizer layer as `optimize_beam_shaping` (`ao_shaping.optimizer.wfless.differentiable_beam`), which returns a `Recorder`.

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
```

## Convention checklist

- [ ] Validation in `__init__`: raise `ValueError` early for invalid inputs.
- [ ] State on `self`: all parameters, tensors, counters, and history live on the instance.
- [ ] Deterministic seeding via a `seed` argument (reproducible random initialization).
- [ ] One-step `update()` returning the next state/solution.
- [ ] Loop logic (logging, early stopping, progress, recorder history) lives in the optimizer layer, not in the class.
- [ ] No one-shot free function inside `algorithm`: the class is the sole public API within the package.
- [ ] Simulation-first tests: synthetic targets, no hardware/SDK imports, CPU default.

## Reference test suites

- [`tests/ao_shaping/algorithm/test_differentiable_beam_optimizer.py`](../../../tests/ao_shaping/algorithm/test_differentiable_beam_optimizer.py) — exercises the class API (`__init__` validation, `update()`) and the optimizer-layer `optimize_beam_shaping` (Recorder history, best-phase tracking, early stopping).
- [`tests/ao_shaping/algorithm/test_differentiable_beam_sim.py`](../../../tests/ao_shaping/algorithm/test_differentiable_beam_sim.py) — simulation-first backprop tests on synthetic targets.