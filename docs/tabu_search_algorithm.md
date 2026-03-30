# Tabu Search Algorithm - Technical Documentation

## Overview

This document describes the tabu search algorithm implementation in the AO-Shaping project, including the algorithm flow, dead loop analysis, and refactoring details.

## Algorithm Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN OPTIMIZATION LOOP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   SPGD      │    │   Update     │    │   Evaluate   │                  │
│  │   Gradient   │───▶│   Voltages   │───▶│   Objective  │                  │
│  │   Estimate   │    │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                                      │                            │
│         │                                      ▼                            │
│         │                            ┌──────────────────┐                   │
│         │                            │  Best Update?    │                   │
│         │                            └────────┬─────────┘                   │
│         │                                     │                             │
│         │                              Yes    │    No                       │
│         │                               ┌──────┴──────┐                      │
│         │                               ▼             ▼                      │
│         │                      ┌─────────────┐                              │
│         │                      │  Update     │                              │
│         │                      │  Best       │                              │
│         │                      └─────────────┘                              │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │           _should_trigger_adaptive_search()                       │        │
│  │  ┌─────────────────────────────────────────────────────────────┐  │        │
│  │  │ enabled? ──── No ────▶ Skip Tabu Search                   │  │        │
│  │  │  │                                                       │  │        │
│  │  │ Yes                                                      │  │        │
│  │  │  ▼                                                       │  │        │
│  │  │ epoch >= warmup? ──── No ────▶ Skip                      │  │        │
│  │  │  │                                                       │  │        │
│  │  │ Yes                                                      │  │        │
│  │  │  ▼                                                       │  │        │
│  │  │ epoch % interval == 0? ──── No ────▶ Skip              │  │        │
│  │  │  │                                                       │  │        │
│  │  │ Yes                                                      │  │        │
│  │  │  ▼                                                       │  │        │
│  │  │ epoch - last_best >= patience? ──── No ────▶ Skip       │  │        │
│  │  │  │                                                       │  │        │
│  │  │ Yes                                                      │  │        │
│  │  │  ▼                                                       │  │        │
│  │  │              Trigger Tabu Search                        │  │        │
│  │  └─────────────────────────────────────────────────────────────┘  │        │
│  └──────────────────────────────────────────────────────────────────┘        │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │                    run_adaptive_search()                           │        │
│  │  ┌─────────────────────────────────────────────────────────────┐    │        │
│  │  │ 1. Prepare Anchor                                        │    │        │
│  │  │    anchor_v = best_v OR current_v (based on search_anchor)│    │        │
│  │  │    anchor_objective = best_objective OR current_objective │    │        │
│  │  └─────────────────────────────────────────────────────────────┘    │        │
│  │                              │                                       │        │
│  │                              ▼                                       │        │
│  │  ┌─────────────────────────────────────────────────────────────┐    │        │
│  │  │ 2. Generate Candidates                                      │    │        │
│  │  │    _generate_search_candidates()                           │    │        │
│  │  │    - n_samples = 8                                        │    │        │
│  │  │    - Mix of Gaussian (dense) and Sparse perturbations     │    │        │
│  │  │    - Apply dm_unit_mask                                    │    │        │
│  │  └─────────────────────────────────────────────────────────────┘    │        │
│  │                              │                                       │        │
│  │                              ▼                                       │        │
│  │  ┌─────────────────────────────────────────────────────────────┐    │        │
│  │  │ 3. For Each Candidate (BOUNDED LOOP - MAX 8 iterations) │    │        │
│  │  │    ┌──────────────────────────────────────────────────┐   │    │        │
│  │  │    │ a. Clip voltages to [V_Min, V_Max]              │   │    │        │
│  │  │    └──────────────────────────────────────────────────┘   │    │        │
│  │  │    ┌──────────────────────────────────────────────────┐   │    │        │
│  │  │    │ b. Tabu Check                                      │   │    │        │
│  │  │    │    tabu_memory.contains(candidate)                │   │    │        │
│  │  │    │    If YES: tabu_hits++, skip                     │   │    │        │
│  │  │    └──────────────────────────────────────────────────┘   │    │        │
│  │  │    ┌──────────────────────────────────────────────────┐   │    │        │
│  │  │    │ c. Safety Check                                   │   │    │        │
│  │  │    │    dm.check_dm_unit_grad_safe(candidate)          │   │    │        │
│  │  │    │    If NO: safe_rejects++, add to tabu, skip      │   │    │        │
│  │  │    └──────────────────────────────────────────────────┘   │    │        │
│  │  │    ┌──────────────────────────────────────────────────┐   │    │        │
│  │  │    │ d. Evaluate Candidate                            │   │    │        │
│  │  │    │    evaluate_candidate(candidate)                 │   │    │        │
│  │  │    │    - Get objective value                         │   │    │        │
│  │  │    │    - Compare with anchor_objective + tol         │   │    │        │
│  │  │    └──────────────────────────────────────────────────┘   │    │        │
│  │  │    ┌──────────────────────────────────────────────────┐   │    │        │
│  │  │    │ e. Selection                                      │   │    │        │
│  │  │    │    If improved AND best: update best_candidate   │   │    │        │
│  │  │    │    Else: add to tabu memory                      │   │    │        │
│  │  │    └──────────────────────────────────────────────────┘   │    │        │
│  │  │                                                         │    │        │
│  │  └───────────────────────────────────────────────────────────┘    │        │
│  │                              │                                       │        │
│  │                              ▼                                       │        │
│  │  ┌─────────────────────────────────────────────────────────────┐    │        │
│  │  │ 4. Check Results                                           │    │        │
│  │  │    If best_candidate is None:                             │    │        │
│  │  │       - Update radius: expand (improved=False)            │    │        │
│  │  │       - Return: {accepted: False, ...}                     │    │        │
│  │  │    Else:                                                    │    │        │
│  │  │       - Add anchor to tabu memory                         │    │        │
│  │  │       - Update radius: shrink (improved=True)             │    │        │
│  │  │       - Return: {accepted: True, voltages, value, ...}    │    │        │
│  │  └─────────────────────────────────────────────────────────────┘    │        │
│  │                                                                   │        │
│  └───────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Dead Loop Analysis

### Potential Loop Analysis

The algorithm has **NO dead loops** due to the following safeguards:

| Loop Location | Bound Type | Max Iterations | Exit Condition |
|--------------|------------|----------------|----------------|
| Main optimization | Finite | `epochs` | Loop completes |
| Candidate generation | Bounded | `search_samples (default 8)` | All candidates processed |
| Tabu check | O(1) lookup | N/A | Immediate return |
| Safety check | O(n) check | N/A | Immediate return |

### Safeguard Details

1. **Bounded Candidate Loop**: The loop iterates over `candidates` which is generated by `_generate_search_candidates()`. The number of candidates is bounded by `search_samples` (default: 8).

2. **Tabu Memory with FIFO Eviction**: When capacity is exceeded, oldest entries are evicted using `popleft()`.

3. **Graceful Failure**: If all candidates are rejected (tabu hits or unsafe), the function returns `{"accepted": False}` with statistics.

4. **Adaptive Search Trigger Conditions**: The search only triggers when:
   - Enabled
   - Past warmup period
   - At correct interval
   - Patience exceeded

### Edge Cases Handled

| Case | Handling |
|------|----------|
| All candidates tabu | Return `accepted: False`, expand radius |
| All candidates unsafe | Return `accepted: False`, expand radius |
| Tabu capacity = 0 | Tabu disabled, always check candidates |
| No improvement found | Expand radius for wider search |
| Improvement found | Shrink radius for finer search |

## Tabu Memory Structure

```
┌─────────────────────────────────────────────────────┐
│              TabuMemory (FIFO Queue)                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  _keys: Set[tuple[int, ...]]                      │
│  ├── O(1) lookup for membership test               │
│  └── Stores unique quantized keys                  │
│                                                     │
│  _queue: Deque[tuple[int, ...]]                    │
│  ├── FIFO ordering for eviction                    │
│  └── Max length = capacity                         │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  Key Generation: make_key(voltages)         │    │
│  │  1. voltages / quantization                │    │
│  │  2. np.round()                            │    │
│  │  3. Convert to tuple[int, ...]            │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Adaptive Search Radius

```
┌─────────────────────────────────────────────────────┐
│         AdaptiveSearchState (Dynamic Radius)        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  radius: Current search radius                      │
│                                                     │
│  update_radius(improved: bool):                     │
│    ┌───────────────────────────────────────────┐   │
│    │  IF improved == True:                     │   │
│    │    radius = radius * shrink_ratio         │   │
│    │    (Exploitation: refine search)          │   │
│    │                                           │   │
│    │  ELSE:                                    │   │
│    │    radius = radius * expand_ratio         │   │
│    │    (Exploration: widen search)           │   │
│    │                                           │   │
│    │  radius = clip(radius, min, max)          │   │
│    └───────────────────────────────────────────┘   │
│                                                     │
│  Default ratios:                                    │
│    - expand_ratio = 1.4  (40% increase)            │
│    - shrink_ratio = 0.75 (25% decrease)            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Candidate Generation Strategy

```
┌─────────────────────────────────────────────────────┐
│        _generate_search_candidates()               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  For sample_id in [0, n_samples):                 │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  IF sample_id % 2 == 0:                     │   │
│  │    Dense (Gaussian) Perturbation            │   │
│  │    - perturbation ~ N(0, radius_scale)     │   │
│  │    - Activates all dimensions               │   │
│  │    - Good for fine-grained exploration      │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  ELSE:                                      │   │
│  │    Sparse (Random Subset) Perturbation      │   │
│  │    - signs = random ±1                     │   │
│  │    - magnitudes = random(0.35~1.0)*radius   │   │
│  │    - sparse_mask = Bernoulli(0.35)         │   │
│  │    - perturbation = signs * magnitudes *    │   │
│  │                     sparse_mask             │   │
│  │    - Only ~35% dimensions active           │   │
│  │    - Good for escaping local optima         │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Final: candidate = anchor + perturbation * mask   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Integration with pib.py

The tabu search is integrated into the PIB optimization as follows:

```python
# In optimize_pib():
# 1. Initialize components
adaptive_search_state = AdaptiveSearchState(...)
tabu_memory = TabuMemory(...)

# 2. Define evaluation callback
def evaluate_candidate(voltages):
    dm.send_voltages(voltages)
    img = cam.get_numpy_image()
    return {
        "J": calc_objective(img),
        objective: test_objective(img),
        "ratio": ratio,
        "img": img,
    }

# 3. Check trigger condition
if _should_trigger_adaptive_search(epoch, ...):
    # 4. Run search
    search_result = run_adaptive_search(epoch, current_v, current_objective)
```

## Refactoring to Standalone Module

The algorithm has been refactored into `src/ao_shaping/algorithm/tabu_search.py` for reusability:

| Original (in pib.py) | New (tabu_search.py) |
|---------------------|----------------------|
| `TabuMemory` class | `TabuMemory` dataclass |
| `AdaptiveSearchState` class | `AdaptiveSearchState` dataclass |
| `_generate_search_candidates()` | `generate_search_candidates()` |
| `_should_trigger_adaptive_search()` | `should_trigger_search()` |
| Nested `run_adaptive_search()` | `TabuSearchRunner.run_search()` |

### Usage Example

```python
from ao_shaping.algorithm.tabu_search import (
    TabuSearchRunner,
    TabuMemory,
    AdaptiveSearchState,
    generate_search_candidates,
)

# Create components
tabu = TabuMemory(capacity=128, quantization=2.0)
state = AdaptiveSearchState(
    radius=2.0, min_radius=0.5, max_radius=12.0,
    expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4
)

# Create runner
runner = TabuSearchRunner(
    tabu_memory=tabu,
    search_state=state,
    candidate_generator=generate_search_candidates,
    safety_check=lambda v: check_my_safety(v),
    clip_bounds=(-50, 50),
)

# Run search
def evaluate(voltages):
    # Your logic here
    return {"value": objective_value}

result = runner.run_search(
    anchor_v=current_voltages,
    anchor_objective=current_value,
    evaluate_candidate=evaluate,
    objective_key="value",
)
```

## Modification Log

| Date | Author | Change |
|------|--------|--------|
| 2026-03-26 | AO-Shaping | Initial implementation |
| 2026-03-26 | AO-Shaping | Refactored to standalone module |
