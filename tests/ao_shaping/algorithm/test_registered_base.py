"""Tests for the shared :class:`RegisteredBase` contract.

The two optimizer families in ``ao_shaping.algorithm`` -- gradient optimizers and
black-box heuristics -- inherit the same registry / construction / reset surface
from :mod:`ao_shaping.algorithm.base`. These tests pin that shared contract, and
in particular the two properties that a naive implementation gets wrong:

* each family owns a **separate** registry (a gradient name must never satisfy a
  heuristic lookup);
* neither family **filters or guesses** keyword arguments -- an option an
  algorithm does not support raises ``TypeError`` naming it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

from ao_shaping.algorithm.base import RegisteredBase
from ao_shaping.algorithm.gradient.adam import Adam, Base, SGD
from ao_shaping.algorithm.heuristic.ga import GeneticAlgorithm
from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)


class _Family(RegisteredBase):
    """A minimal concrete family used to exercise the shared contract."""

    _registry: ClassVar[dict[str, type[Any]]] = {}

    def __init__(self, tag: str = "x", count: int = 1):
        self.tag = tag
        self.count = count
        self.reset_calls = 0

    @classmethod
    def _construct(cls, tag: str = "x", count: int = 1, **kwargs: Any) -> "_Family":
        return cls(tag=tag, count=count, **kwargs)


class _Alpha(_Family):
    _registry_key = "alpha"


class _Beta(_Family):
    _registry_key = "beta"


class TestRegistration:
    def test_subclasses_register_themselves(self):
        assert _Family._registry["alpha"] is _Alpha
        assert _Family._registry["beta"] is _Beta
        assert _Family.registered() == ("alpha", "beta")

    def test_an_intermediate_subclass_is_not_registered_under_its_parents_key(self):
        """``_registry_key`` is read from ``cls.__dict__``, so it is not inherited."""

        class _AlphaChild(_Alpha):
            pass

        assert _Family._registry["alpha"] is _Alpha
        assert _AlphaChild not in _Family._registry.values()

    def test_register_returns_the_class_and_rejects_foreign_types(self):
        assert _Family.register("gamma", _Alpha) is _Alpha
        with pytest.raises(TypeError):
            _Family.register("bad", int)

    def test_create_builds_through_the_family_construct_hook(self):
        made = _Family.create("beta", tag="hello", count=3)
        assert isinstance(made, _Beta)
        assert made.tag == "hello"
        assert made.count == 3

    def test_create_rejects_an_unregistered_key(self):
        with pytest.raises(ValueError, match="Unknown"):
            _Family.create("nope")

    def test_error_message_lists_the_available_keys(self):
        with pytest.raises(ValueError) as excinfo:
            _Family.create("nope")
        message = str(excinfo.value)
        assert "alpha" in message and "beta" in message

    def test_a_family_without_a_construct_hook_says_so(self):
        class _NoConstruct(RegisteredBase):
            _registry: ClassVar[dict[str, type[Any]]] = {}
            _registry_key = "no-construct"

        with pytest.raises(NotImplementedError):
            _NoConstruct.create("no-construct")


class TestNormalisation:
    def test_identity_by_default(self):
        assert _Family._normalize_key("Alpha") == "Alpha"

    def test_gradient_family_is_case_insensitive(self):
        assert Base._normalize_key("AdaMoD") == "adamod"
        assert isinstance(Base.create("AdaMoD", dim=4), Adam)

    def test_heuristic_family_keys_are_enum_members(self):
        assert HeuristicOptimizer._normalize_key(OptimizerType.GA) is OptimizerType.GA


class TestRegistryIsolation:
    """A shared registry would let one family's keys satisfy another's lookups."""

    def test_gradient_and_heuristic_registries_are_distinct_objects(self):
        assert Base._registry is not HeuristicOptimizer._registry

    def test_gradient_names_are_not_heuristic_selectors(self):
        assert "adam" in Base._registry
        assert "adam" not in HeuristicOptimizer._registry

    def test_a_gradient_lookup_does_not_satisfy_a_heuristic_lookup(self):
        assert issubclass(HeuristicOptimizer, RegisteredBase)
        with pytest.raises(ValueError):
            HeuristicOptimizer.create("adam", dim=4)


class TestDimensionValidation:
    @pytest.mark.parametrize("dim", [0, -1, -100])
    def test_non_positive_dim_is_rejected(self, dim):
        with pytest.raises(ValueError, match="dim must be a positive integer"):
            RegisteredBase._validate_dim(dim)

    def test_positive_dim_is_accepted(self):
        assert RegisteredBase._validate_dim(1) is None

    @pytest.mark.parametrize("dim", [0, -5])
    def test_both_families_enforce_it(self, dim):
        with pytest.raises(ValueError, match="dim must be a positive integer"):
            SGD(dim=dim)
        with pytest.raises(ValueError, match="dim must be a positive integer"):
            GeneticAlgorithm(dim=dim)


class TestEveryConcreteClassDeclaresAKey:
    """A typo in the declaration name must not silently skip registration."""

    @staticmethod
    def _concrete_in_package(family: type) -> list[type]:
        import inspect
        import sys

        found: dict[str, type] = {}
        for module in list(sys.modules.values()):
            name = getattr(module, "__name__", "")
            if not name.startswith("ao_shaping.algorithm"):
                continue
            for attr, obj in vars(module).items():
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, family)
                    and obj is not family
                    and obj.__module__.startswith("ao_shaping.algorithm")
                ):
                    found[obj.__name__] = obj
        return [found[key] for key in sorted(found)]

    @pytest.mark.parametrize(
        "family",
        [Base, HeuristicOptimizer],
        ids=["gradient", "heuristic"],
    )
    def test_concrete_subclasses_declare_the_key(self, family):
        concrete = self._concrete_in_package(family)
        assert concrete, "expected concrete subclasses to inspect"
        missing = [c.__name__ for c in concrete if "_registry_key" not in vars(c)]
        assert not missing, f"do not declare _registry_key: {missing}"

    @pytest.mark.parametrize(
        "family",
        [Base, HeuristicOptimizer],
        ids=["gradient", "heuristic"],
    )
    def test_declared_subclasses_reach_the_registry(self, family):
        concrete = self._concrete_in_package(family)
        unregistered = [
            c.__name__ for c in concrete if c not in family._registry.values()
        ]
        assert not unregistered, f"declared but never registered: {unregistered}"

    def test_every_enum_member_is_reachable(self):
        for member in OptimizerType:
            assert HeuristicOptimizer._registry.get(member) is not None, member


class TestNoArgumentGuessing:
    """Neither family may silently drop an option its target does not accept."""

    def test_gradient_create_forwards_unknown_kwargs_to_the_constructor(self):
        with pytest.raises(TypeError, match="weight_decay"):
            Base.create("sgd", dim=4, lr=0.1, weight_decay=0.01)

    def test_gradient_create_still_accepts_declared_kwargs(self):
        made = Base.create("adam", dim=4, lr=0.1, beta1=0.5, beta2=0.95)
        assert isinstance(made, Adam)
        assert made.beta1 == 0.5
        assert made.beta2 == 0.95

    def test_heuristic_create_forwards_unknown_kwargs_to_the_constructor(self):
        # HillClimbing has no pop_size; the constructor must be the one to object.
        with pytest.raises(TypeError, match="pop_size"):
            HeuristicOptimizer.create(
                OptimizerType.HILL_CLIMBING, dim=4, n_iterations=5, pop_size=7
            )

    def test_heuristic_create_accepts_declared_extras(self):
        made = HeuristicOptimizer.create(
            OptimizerType.GA, dim=4, n_iterations=5, pop_size=7
        )
        assert isinstance(made, GeneticAlgorithm)
        assert made.params.pop_size == 7

    def test_heuristic_create_consumes_the_config_keys_itself(self):
        """``n_iterations``/``bounds``/``seed`` never reach the constructor."""
        made = HeuristicOptimizer.create(
            OptimizerType.GA, dim=4, n_iterations=11, bounds=(-2.0, 2.0), seed=3
        )
        assert made.params.n_generations == 11
        assert made.params.bounds == (-2.0, 2.0)
        # GA carries the seed in its generator, not in its derived config.
        assert made.rng is not None


class TestReset:
    def test_reset_delegates_to_the_family_hook(self):
        instance = _Alpha()
        instance._reset_called = False
        instance._reset = lambda: setattr(instance, "_reset_called", True)
        instance.reset()
        assert instance._reset_called is True

    def test_heuristic_reset_clears_recorded_results(self):
        opt = HeuristicOptimizer.create(OptimizerType.HILL_CLIMBING, dim=4)
        opt.optimize(lambda x: float(np.sum(x**2)))
        assert opt.best_fitness is not None
        assert opt.convergence_history

        opt.reset()
        assert opt.best_fitness is None
        assert opt.best_solution is None
        assert opt.convergence_history == []

    def test_gradient_reset_zeroes_counter_and_buffers(self):
        opt = Base.create("adam", dim=4)
        opt.update(np.ones(4))
        assert opt.t == 1
        assert not np.allclose(opt.m, 0)

        opt.reset()
        assert opt.t == 0
        assert np.allclose(opt.m, 0)
        assert np.allclose(opt.v, 0)

    def test_reset_does_not_reseed_the_generator(self):
        """A re-run explores fresh randomness rather than cloning the previous one."""
        opt = HeuristicOptimizer.create(
            OptimizerType.RANDOM_SEARCH, dim=4, n_iterations=20, seed=5
        )
        generator_before = opt.rng
        _, first = opt.optimize(lambda x: float(np.sum(x**2)))

        opt.reset()
        # Same generator object -- reset must not silently re-seed it.
        assert opt.rng is generator_before
        _, second = opt.optimize(lambda x: float(np.sum(x**2)))
        assert second != first


class TestOptimizerConfigKeys:
    def test_config_keys_stay_in_step_with_the_dataclass(self):
        assert HeuristicOptimizer._CONFIG_KEYS == frozenset(
            OptimizerConfig.__dataclass_fields__
        )

    def test_a_config_field_is_consumed_rather_than_forwarded(self):
        # early_stop_threshold is a config field; RandomSearch declares it too, so
        # a conflict would be a TypeError rather than a silent pass-through.
        opt = HeuristicOptimizer.create(
            OptimizerType.RANDOM_SEARCH,
            dim=4,
            n_iterations=5,
            early_stop_threshold=1e-6,
        )
        assert opt.config.early_stop_threshold == 1e-6
