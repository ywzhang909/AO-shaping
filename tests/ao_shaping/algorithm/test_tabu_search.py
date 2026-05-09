import numpy as np
import pytest

from ao_shaping.algorithm.tabu_search import (
    TabuMemory,
    AdaptiveSearchState,
    generate_search_candidates,
    TabuSearchRunner,
)


class TestTabuMemory:
    def test_add_and_contains(self):
        mem = TabuMemory(capacity=128, quantization=2.0)
        v = np.array([10.0, 20.0, 30.0])
        mem.add(v)
        assert mem.contains(v) is True

    def test_not_contains_unadded(self):
        mem = TabuMemory(capacity=128, quantization=2.0)
        v = np.array([10.0, 20.0, 30.0])
        assert mem.contains(v) is False

    def test_quantization_groups_similar(self):
        mem = TabuMemory(capacity=128, quantization=5.0)
        v1 = np.array([10.0, 20.0])
        v2 = np.array([11.0, 21.0])
        mem.add(v1)
        assert mem.contains(v2) is True

    def test_capacity_eviction(self):
        mem = TabuMemory(capacity=2, quantization=1.0)
        v1 = np.array([1.0])
        v2 = np.array([2.0])
        v3 = np.array([3.0])
        mem.add(v1)
        mem.add(v2)
        mem.add(v3)
        assert mem.contains(v1) is False
        assert mem.contains(v3) is True

    def test_zero_capacity_disables_tabu(self):
        mem = TabuMemory(capacity=0, quantization=1.0)
        v = np.array([1.0, 2.0])
        mem.add(v)
        assert mem.contains(v) is False

    def test_duplicate_add_ignored(self):
        mem = TabuMemory(capacity=10, quantization=1.0)
        v = np.array([5.0])
        mem.add(v)
        mem.add(v)
        assert mem.contains(v) is True

    def test_make_key_deterministic(self):
        mem = TabuMemory(capacity=10, quantization=2.0)
        v = np.array([10.0, 20.0])
        assert mem.make_key(v) == mem.make_key(v)


class TestAdaptiveSearchState:
    def test_shrink_on_improvement(self):
        state = AdaptiveSearchState(
            radius=4.0, min_radius=0.5, max_radius=12.0,
            expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4
        )
        new_r = state.update_radius(improved=True)
        assert new_r < 4.0
        assert new_r == pytest.approx(3.0)

    def test_expand_on_no_improvement(self):
        state = AdaptiveSearchState(
            radius=4.0, min_radius=0.5, max_radius=12.0,
            expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4
        )
        new_r = state.update_radius(improved=False)
        assert new_r > 4.0
        assert new_r == pytest.approx(5.6)

    def test_clamp_to_max(self):
        state = AdaptiveSearchState(
            radius=10.0, min_radius=0.5, max_radius=12.0,
            expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4
        )
        new_r = state.update_radius(improved=False)
        assert new_r == 12.0

    def test_clamp_to_min(self):
        state = AdaptiveSearchState(
            radius=0.6, min_radius=0.5, max_radius=12.0,
            expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4
        )
        new_r = state.update_radius(improved=True)
        assert new_r == 0.5


class TestGenerateSearchCandidates:
    def test_returns_list_of_arrays(self):
        anchor = np.zeros(5)
        results = generate_search_candidates(anchor, radius_scale=1.0, n_samples=10)
        assert isinstance(results, list)
        assert len(results) == 10
        for r in results:
            assert isinstance(r, np.ndarray)

    def test_candidates_near_anchor(self):
        anchor = np.zeros(5)
        results = generate_search_candidates(anchor, radius_scale=1.0, n_samples=20)
        for r in results:
            diff = np.abs(r - anchor)
            assert np.max(diff) < 10.0

    def test_active_mask_applied(self):
        anchor = np.zeros(5)
        mask = np.array([True, False, True, False, True])
        results = generate_search_candidates(anchor, radius_scale=1.0, n_samples=10, active_mask=mask)
        for r in results:
            assert r[1] == pytest.approx(0.0)
            assert r[3] == pytest.approx(0.0)

    def test_same_shape_as_anchor(self):
        anchor = np.zeros(8)
        results = generate_search_candidates(anchor, radius_scale=1.0, n_samples=5)
        for r in results:
            assert r.shape == anchor.shape
