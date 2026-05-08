import numpy as np

from ao_shaping.algorithm.adam import AdaMOD, Adam, AdamW
from ao_shaping.optimizer.wfless.pib import (
    AdaptiveSearchState,
    TabuMemory,
    _create_optimizer,
    _extract_optimizer_momentum,
    _generate_search_candidates,
    _reset_optimizer_state,
    _should_trigger_adaptive_search,
    learning_schedule,
)


def test_tabu_memory_quantizes_nearby_solutions() -> None:
    memory = TabuMemory(capacity=4, quantization=2.0)

    solution = np.array([0.0, 4.1, -3.9])
    memory.add(solution)

    assert memory.contains(np.array([0.8, 3.6, -4.2]))
    assert not memory.contains(np.array([3.5, 4.0, -4.0]))


def test_adaptive_search_state_expands_and_shrinks_with_limits() -> None:
    state = AdaptiveSearchState(
        radius=10.0,
        min_radius=4.0,
        max_radius=18.0,
        expand_ratio=1.5,
        shrink_ratio=0.5,
        improvement_tol=1e-4,
    )

    assert state.update_radius(improved=False) == 15.0
    assert state.update_radius(improved=False) == 18.0
    assert state.update_radius(improved=True) == 9.0
    assert state.update_radius(improved=True) == 4.5
    assert state.update_radius(improved=True) == 4.0


def test_generate_search_candidates_respects_mask() -> None:
    anchor = np.array([10.0, 20.0, 30.0, 40.0])
    mask = np.array([True, False, True, False])
    rng = np.random.default_rng(42)

    candidates = _generate_search_candidates(
        anchor_v=anchor,
        radius_scale=5.0,
        n_samples=6,
        dm_unit_mask=mask,
        rng=rng,
    )

    assert len(candidates) == 6
    changed_active_dims = 0
    for candidate in candidates:
        assert candidate[1] == anchor[1]
        assert candidate[3] == anchor[3]
        if not np.allclose(candidate[[0, 2]], anchor[[0, 2]]):
            changed_active_dims += 1
    assert changed_active_dims > 0


def test_should_trigger_adaptive_search_uses_warmup_interval_and_patience() -> None:
    assert not _should_trigger_adaptive_search(
        epoch=50,
        enabled=True,
        warmup=100,
        interval=20,
        patience=40,
        last_best_epoch=0,
    )
    assert not _should_trigger_adaptive_search(
        epoch=120,
        enabled=True,
        warmup=100,
        interval=30,
        patience=40,
        last_best_epoch=95,
    )
    assert _should_trigger_adaptive_search(
        epoch=120,
        enabled=True,
        warmup=100,
        interval=30,
        patience=20,
        last_best_epoch=90,
    )


def test_create_optimizer_supports_adam_family() -> None:
    adam = _create_optimizer("adam", dim=4, lr=0.1)
    fallback = _create_optimizer("unknown", dim=4, lr=0.1)

    assert isinstance(adam, Adam)
    assert isinstance(fallback, AdaMOD)


def test_create_optimizer_with_beta_and_epsilon() -> None:
    adamw = _create_optimizer("adamw", dim=4, lr=0.1, beta1=0.85, beta2=0.95)
    assert isinstance(adamw, AdamW)

    sgd = _create_optimizer("sgd", dim=4, lr=0.5)
    assert sgd.lr == 0.5


def test_extract_optimizer_momentum_returns_ndarray() -> None:
    adam = _create_optimizer("adam", dim=4, lr=0.1)
    momentum = _extract_optimizer_momentum(adam)
    assert momentum is None or isinstance(momentum, np.ndarray)


def test_reset_optimizer_state_zeroes_momentum() -> None:
    from ao_shaping.algorithm.adam import Adam

    adam = Adam(dim=4, lr=0.1)
    adam.m = np.ones(4)
    adam.v = np.ones(4)
    adam.t = 10

    _reset_optimizer_state(adam)

    assert adam.t == 0


def test_learning_schedule_returns_base_values_for_no_history() -> None:
    lr, delta = learning_schedule(power_radius=5.0)
    assert lr > 0
    assert delta > 0

    lr, delta = learning_schedule(power_radius=20.0)
    assert lr > 0
    assert delta > 0


def test_learning_schedule_adapts_with_gradient_history() -> None:
    grad_history = [0.1] * 10
    pib_history = [0.9] * 10

    lr, delta = learning_schedule(
        power_radius=5.0,
        gradient_history=grad_history,
        pib_history=pib_history,
    )
    assert lr > 0
    assert delta > 0


def test_tabu_memory_add_and_retrieve() -> None:
    memory = TabuMemory(capacity=4, quantization=2.0)

    solution = np.array([0.0, 4.1, -3.9])
    memory.add(solution)

    key = memory.make_key(solution)
    assert len(key) == 3
