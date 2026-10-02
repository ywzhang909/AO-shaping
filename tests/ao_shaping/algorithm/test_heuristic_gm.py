"""Tests for guided mutation (GM) on the population-based heuristic optimizers.

Covers the two kernels, the :class:`GMOptimizerMixin` contract, the decorator,
and the invariant that matters most for the existing call sites: **a disabled GM
must not perturb a single random draw**, so every pre-existing seeded result
stays reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.heuristic.cross_entropy import (
    CEMConfig,
    CrossEntropyMethod,
)
from ao_shaping.algorithm.heuristic.differential_evolution import (
    DEConfig,
    DifferentialEvolution,
)
from ao_shaping.algorithm.heuristic.ga import GAParams, GeneticAlgorithm
from ao_shaping.algorithm.heuristic.gm import (
    GMOptimizerMixin,
    NumpyPopulationGM,
    RankGuidedMutation,
    ValueFrequencyGuidedMutation,
    guided_mutation,
)
from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerConfig,
    OptimizerType,
)
from ao_shaping.algorithm.heuristic.pso import ParticleSwarmOptimizer, PSOParams

BOUNDS = (-10.0, 10.0)
KERNELS = [RankGuidedMutation, ValueFrequencyGuidedMutation]


def sphere(x: np.ndarray) -> float:
    """Smooth unimodal test objective (minimised)."""
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    """Multimodal test objective with a known global minimum at 0."""
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def _build(cls, dim: int = 6, n_iter: int = 40, seed: int = 7):
    """Build any of the four GM-capable optimizers with identical settings."""
    config = OptimizerConfig(n_iterations=n_iter, bounds=BOUNDS, seed=seed)
    rng = np.random.default_rng(seed)
    if cls is GeneticAlgorithm:
        return cls(
            dim=dim,
            params=GAParams(pop_size=24, n_generations=n_iter, bounds=BOUNDS),
            random_state=rng,
        )
    if cls is ParticleSwarmOptimizer:
        return cls(
            dim=dim,
            params=PSOParams(n_particles=24, n_iterations=n_iter, bounds=BOUNDS),
            random_state=rng,
        )
    if cls is CrossEntropyMethod:
        return cls(
            dim=dim, config=config, cem_config=CEMConfig(pop_size=30), random_state=rng
        )
    return cls(
        dim=dim, config=config, de_config=DEConfig(pop_size=24), random_state=rng
    )


ALL_ALGOS = [
    GeneticAlgorithm,
    DifferentialEvolution,
    CrossEntropyMethod,
    ParticleSwarmOptimizer,
]
NATIVE_ALGOS = [GeneticAlgorithm, DifferentialEvolution, CrossEntropyMethod]

_REGISTRY_TYPE = {
    GeneticAlgorithm: OptimizerType.GA,
    DifferentialEvolution: OptimizerType.DIFFERENTIAL_EVOLUTION,
    CrossEntropyMethod: OptimizerType.CROSS_ENTROPY,
    ParticleSwarmOptimizer: OptimizerType.PSO,
}


class _Host(NumpyPopulationGM):
    """Minimal concrete host used to exercise the mixin in isolation."""

    def __init__(self, dim: int = 6, bounds: tuple[float, float] = BOUNDS):
        self.dim = dim
        self.rng = np.random.default_rng(3)
        self.config = OptimizerConfig(bounds=bounds)
        self._population = np.empty((0, dim))
        self._fitness_vals = np.empty(0)
        self._current_iter = 0

    def load(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """Seed the host with a scored population."""
        self._population = np.array(population, dtype=np.float64)
        self._fitness_vals = np.array(fitness, dtype=np.float64)


class TestKernelContract:
    """Both kernels honour the same output contract."""

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_shape_bounds_and_purity(self, kernel_cls):
        rng = np.random.default_rng(0)
        population = rng.uniform(-5, 5, (20, 6))
        fitness = rng.normal(size=20)
        pop_copy = population.copy()
        fit_copy = fitness.copy()

        host = _Host()
        host.load(population, fitness)
        host.enable_gm(kernel_cls())
        children = host.apply_gm_if_enabled()

        assert children.shape == (host._gm_offspring(), 6)
        assert np.all(np.isfinite(children))
        assert np.all(children >= BOUNDS[0]) and np.all(children <= BOUNDS[1])
        # The kernel must not mutate its inputs.
        assert np.array_equal(population, pop_copy)
        assert np.array_equal(fitness, fit_copy)

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_deterministic_under_a_fixed_generator(self, kernel_cls):
        population = np.linspace(-3, 3, 40).reshape(10, 4)
        fitness = np.arange(10.0)
        ranks = fitness.argsort().argsort() + 1
        args = (population, fitness, ranks, 2, 5, BOUNDS)

        first = kernel_cls()(*args, rng=np.random.default_rng(11))
        second = kernel_cls()(*args, rng=np.random.default_rng(11))
        assert np.array_equal(first, second)

    def test_rank_kernel_prefers_better_parents(self):
        """Parent sampling must be rank-weighted, monotonically favouring rank 1.

        Linear ranking deliberately gives the best individual only a
        ``2 / (n + 1)`` share rather than collapsing onto it, so the property to
        assert is that the empirical parent distribution decreases with rank --
        not that every child comes from the elite.
        """
        n, dim = 10, 4
        # Well-separated rows plus a negligible sigma, so the parent of each
        # child can be recovered by nearest row.
        population = np.arange(n * dim, dtype=float).reshape(n, dim) * 5.0
        fitness = np.arange(float(n))
        ranks = fitness.argsort().argsort() + 1

        kernel = RankGuidedMutation(g0=1.0, sigma0=1e-3, sigma_min=1e-3)
        children = kernel(
            population=population,
            fitness=fitness,
            ranks=ranks,
            current_iter=0,
            n_offspring=4000,
            # sigma is a fraction of the bound span: 1e-3 * 500 = 0.5 against rows
            # 5 apart. The span must also cover the largest row (39 * 5 = 195),
            # otherwise clipping collapses distinct rows onto each other.
            bounds=(-250.0, 250.0),
            rng=np.random.default_rng(0),
        )
        parent = (
            np.abs(children[:, None, :] - population[None, :, :])
            .sum(axis=2)
            .argmin(axis=1)
        )
        counts = np.bincount(parent, minlength=n).astype(float)
        shares = counts / counts.sum()

        # Monotonically non-increasing with rank.
        assert all(shares[i] >= shares[i + 1] for i in range(n - 1)), shares
        # Rank 1 takes the largest share, close to the analytic 2 / (n + 1).
        assert shares[0] == shares.max()
        assert shares[0] == pytest.approx(2.0 / (n + 1), abs=0.02)

    def test_rank_kernel_anneals_toward_the_floor(self):
        kernel = RankGuidedMutation()
        assert kernel._fraction(0) == pytest.approx(kernel.g0)
        assert kernel._fraction(10_000) < kernel._fraction(10)
        assert kernel._fraction(10_000) >= kernel.g_end

    def test_value_frequency_kernel_seeds_itself_from_the_population(self):
        kernel = ValueFrequencyGuidedMutation()
        assert not kernel._elite_values
        rng = np.random.default_rng(1)
        population = rng.uniform(-1, 1, (12, 3))
        fitness = rng.normal(size=12)
        ranks = fitness.argsort().argsort() + 1
        kernel(population, fitness, ranks, 0, 4, BOUNDS, rng)
        # observe() runs lazily on the first call.
        assert set(kernel._elite_values) == {0, 1, 2}

    def test_value_frequency_reset_clears_history(self):
        kernel = ValueFrequencyGuidedMutation()
        rng = np.random.default_rng(1)
        population = rng.uniform(-1, 1, (12, 3))
        fitness = rng.normal(size=12)
        ranks = fitness.argsort().argsort() + 1
        kernel(population, fitness, ranks, 0, 4, BOUNDS, rng)
        assert kernel._elite_values
        kernel.reset()
        assert kernel._elite_values == {}

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    @pytest.mark.parametrize(
        "population,fitness",
        [
            (np.empty((0, 3)), np.empty(0)),
            (np.zeros(4), np.zeros(4)),  # not 2-D
            (np.zeros((4, 3)), np.zeros(5)),  # fitness does not line up
        ],
    )
    def test_preconditions_are_rejected_loudly(self, kernel_cls, population, fitness):
        """The kernels validate their own input rather than tolerating it."""
        ranks = (
            fitness.argsort().argsort() + 1 if fitness.ndim == 1 else np.asarray(fitness)
        )
        with pytest.raises(ValueError):
            kernel_cls()(
                population=population,
                fitness=fitness,
                ranks=ranks,
                current_iter=0,
                n_offspring=2,
                bounds=BOUNDS,
                rng=np.random.default_rng(0),
            )

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: RankGuidedMutation(g0=0.0),
            lambda: RankGuidedMutation(g0=0.1, g_end=0.9),
            lambda: RankGuidedMutation(sigma0=0.0),
            lambda: RankGuidedMutation(lam_g=0.0),
            lambda: RankGuidedMutation(lam_s=-1.0),
            lambda: ValueFrequencyGuidedMutation(elite_fraction=0.0),
            lambda: ValueFrequencyGuidedMutation(n_guided=0),
            lambda: ValueFrequencyGuidedMutation(n_guided0=0),
            lambda: ValueFrequencyGuidedMutation(lam_g=0.0),
            lambda: ValueFrequencyGuidedMutation(jitter=-0.1),
            lambda: ValueFrequencyGuidedMutation(elite_decay=1.0),
        ],
    )
    def test_invalid_parameters_are_rejected(self, factory):
        with pytest.raises(ValueError):
            factory()


class TestMixinContract:
    """The mixin declares hooks; it must not implement them defensively."""

    def test_hooks_are_abstract_on_the_mixin_itself(self):
        """A host that mixes GM in without implementing the hooks fails loudly."""
        mixin = GMOptimizerMixin()
        with pytest.raises(NotImplementedError):
            mixin._gm_population()
        with pytest.raises(NotImplementedError):
            mixin._gm_fitness()
        with pytest.raises(NotImplementedError):
            mixin._gm_iteration()
        with pytest.raises(NotImplementedError):
            mixin._gm_offspring()
        with pytest.raises(NotImplementedError):
            mixin._gm_bounds()
        with pytest.raises(NotImplementedError):
            mixin._gm_commit(np.zeros((1, 1)))

    @pytest.mark.parametrize("cls", ALL_ALGOS)
    def test_defaults_are_inert(self, cls):
        opt = _build(cls)
        assert opt.use_gm is False
        assert opt._gm_operator is None
        assert opt._gm_offspring() == 0
        assert opt.apply_gm_if_enabled().size == 0

    @pytest.mark.parametrize("cls", ALL_ALGOS)
    def test_enable_disable_toggle(self, cls):
        opt = _build(cls)
        kernel = RankGuidedMutation()
        opt.enable_gm(kernel)
        assert opt.use_gm is True
        assert opt._gm_operator is kernel
        opt.disable_gm()
        assert opt.use_gm is False
        # disable_gm keeps the kernel so GM can be re-enabled cheaply.
        assert opt._gm_operator is kernel

    def test_offspring_reservation_follows_the_fraction(self):
        host = _Host()
        assert host._gm_offspring() == 0
        host.enable_gm(RankGuidedMutation())
        # An empty population contributes nothing rather than a bogus 1.
        assert host._gm_offspring() == 0

        host.load(np.zeros((50, 6)), np.zeros(50))
        host.gm_offspring_fraction = 0.2
        assert host._gm_offspring() == 10
        host.gm_offspring_fraction = 0.05
        assert host._gm_offspring() == 2  # 2.5 rounds to 2

    def test_settings_do_not_leak_between_instances(self):
        a = _build(GeneticAlgorithm)
        b = _build(GeneticAlgorithm)
        a.enable_gm(RankGuidedMutation())
        assert a.use_gm is True
        assert b.use_gm is False

    @pytest.mark.parametrize("cls", NATIVE_ALGOS)
    def test_numpy_mixin_reads_and_writes_its_attributes(self, cls):
        opt = _build(cls)
        population = np.arange(30.0).reshape(10, 3)
        fitness = np.arange(10.0)
        opt._population = population
        opt._fitness_vals = fitness
        opt._current_iter = 7
        assert opt._gm_population() is population
        assert opt._gm_fitness() is fitness
        assert opt._gm_iteration() == 7
        assert opt._gm_bounds() == BOUNDS

        replacement = np.ones((10, 3))
        opt._gm_commit(replacement)
        assert opt._population is replacement


class TestBehaviourPreservation:
    """The invariant that protects every pre-existing seeded run."""

    @pytest.mark.parametrize("cls", ALL_ALGOS)
    def test_disabled_gm_is_bit_identical(self, cls):
        """enable+disable must leave the seeded result untouched."""
        baseline_opt = _build(cls)
        _, baseline = baseline_opt.optimize(sphere)

        toggled = _build(cls)
        toggled.enable_gm(RankGuidedMutation())
        toggled.disable_gm()
        _, after_toggle = toggled.optimize(sphere)

        assert after_toggle == baseline

    @pytest.mark.parametrize("cls", ALL_ALGOS)
    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_enabled_gm_still_converges(self, cls, kernel_cls):
        opt = _build(cls, n_iter=60)
        opt.enable_gm(kernel_cls())
        best_x, best_f = opt.optimize(rastrigin)
        assert best_x.shape == (6,)
        assert np.all(np.isfinite(best_x))
        assert np.isfinite(best_f)
        # Far from pathological on a 6-D Rastrigin within 60 iterations.
        assert best_f < 1e3

    @pytest.mark.parametrize("cls", ALL_ALGOS)
    def test_enabled_gm_is_deterministic(self, cls):
        a = _build(cls)
        a.enable_gm(RankGuidedMutation())
        _, first = a.optimize(rastrigin)
        b = _build(cls)
        b.enable_gm(RankGuidedMutation())
        _, second = b.optimize(rastrigin)
        assert first == second

    @pytest.mark.parametrize("cls", ALL_ALGOS)
    def test_enabled_gm_respects_bounds(self, cls):
        opt = _build(cls, n_iter=25)
        opt.enable_gm(RankGuidedMutation())
        opt.optimize(sphere)
        population = getattr(opt, "_population", None)
        if isinstance(population, np.ndarray) and population.size:
            assert np.all(population >= BOUNDS[0])
            assert np.all(population <= BOUNDS[1])

    @pytest.mark.parametrize("cls", ALL_ALGOS)
    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_registry_built_optimizers_accept_gm(self, cls, kernel_cls):
        """create() must yield an optimizer whose GM path runs."""
        opt = HeuristicOptimizer.create(
            _REGISTRY_TYPE[cls], dim=6, n_iterations=15, seed=3, bounds=BOUNDS
        )
        opt.enable_gm(kernel_cls())
        _, best_f = opt.optimize(sphere)
        assert np.isfinite(best_f)


class TestPopulationSizeIsPreserved:
    """GM must never change how many candidates the population holds."""

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_ga_population_size_is_unchanged(self, kernel_cls):
        opt = _build(GeneticAlgorithm, n_iter=30)
        opt.enable_gm(kernel_cls())
        opt.optimize(sphere)
        assert opt._population.shape == (opt.params.pop_size, opt.dim)
        assert opt._population.shape[0] == 24

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_de_and_cem_population_size_is_unchanged(self, kernel_cls):
        for cls in (DifferentialEvolution, CrossEntropyMethod):
            opt = _build(cls, n_iter=25)
            expected = (
                max(10, opt.de_config.pop_size)
                if cls is DifferentialEvolution
                else 30
            )
            opt.enable_gm(kernel_cls())
            opt.optimize(sphere)
            assert opt._population.shape[0] == expected

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_pso_swarm_size_is_unchanged(self, kernel_cls):
        opt = _build(ParticleSwarmOptimizer, n_iter=25)
        opt.enable_gm(kernel_cls())
        opt.optimize(sphere)
        assert len(opt.particles) == opt.params.n_particles


class TestPsoIntegration:
    """PSO keeps a ``list[Particle]``, projecting it through the GM contract."""

    def _seeded_swarm(self, seed: int = 5):
        opt = _build(ParticleSwarmOptimizer)
        opt.particles = opt._initialize_particles(None)
        opt._evaluate_particles(sphere)
        opt._current_iter = 1
        return opt

    def test_projection_is_consistent(self):
        opt = self._seeded_swarm()
        assert opt._gm_population().shape == (len(opt.particles), opt.dim)
        assert opt._gm_fitness().shape == (len(opt.particles),)
        assert opt._gm_bounds() == BOUNDS

    def test_commit_zeroes_velocity_only_for_moved_particles(self):
        opt = self._seeded_swarm()
        opt.enable_gm(RankGuidedMutation())
        population = opt._gm_population()
        population[0] = population[0] + 3.0  # force one particle to differ
        opt._gm_commit(population)

        assert np.array_equal(opt.particles[0].position, population[0])
        assert np.all(opt.particles[0].velocity == 0.0)
        # An untouched particle keeps its velocity.
        assert np.any(opt.particles[-1].velocity != 0.0)

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_leaders_are_moved_only_by_pso(self, kernel_cls):
        """GM replaces the worst particles, so a leader's motion is PSO's own."""
        opt = self._seeded_swarm()
        opt.enable_gm(kernel_cls())
        fitness = opt._gm_fitness()
        leaders = np.argsort(fitness, kind="stable")[:3]
        expected = {
            int(i): np.clip(
                opt.particles[int(i)].position + opt.particles[int(i)].velocity,
                BOUNDS[0],
                BOUNDS[1],
            )
            for i in leaders
        }
        velocities = {int(i): opt.particles[int(i)].velocity.copy() for i in leaders}

        opt._advance_positions()

        for slot, position in expected.items():
            assert np.array_equal(opt.particles[slot].position, position)
            # A leader's velocity is never zeroed, because GM never touched it.
            assert np.array_equal(opt.particles[slot].velocity, velocities[slot])

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_injected_positions_stay_in_bounds(self, kernel_cls):
        opt = self._seeded_swarm()
        opt.enable_gm(kernel_cls())
        opt._advance_positions()
        for particle in opt.particles:
            assert np.all(particle.position >= BOUNDS[0])
            assert np.all(particle.position <= BOUNDS[1])
            assert particle.velocity.shape == (opt.dim,)


class TestDecorator:
    """The decorator must be transparent when GM is off."""

    def test_metadata_is_preserved(self):
        opt = _build(GeneticAlgorithm)
        assert opt._evolve_generation.__name__ == "_evolve_generation"
        assert hasattr(opt._evolve_generation, "__wrapped__")

    def test_returns_the_evolved_population_when_disabled(self):
        """With GM off the wrapper hands back exactly what the step returned."""
        host = _Host()
        sentinel = np.arange(30.0).reshape(10, 3)
        host._population = sentinel
        host._fitness_vals = np.zeros(10)
        host._current_iter = 0

        @guided_mutation()
        def step(self):
            return self._population

        assert step(host) is sentinel

    def test_replace_worst_keeps_rows_fixed(self):
        """The ``replace_worst`` merge must overwrite, never grow."""

        class _Fixed(NumpyPopulationGM):
            def __init__(self):
                self.dim = 3
                self.rng = np.random.default_rng(0)
                self.config = OptimizerConfig(bounds=BOUNDS)
                self._population = np.arange(12, dtype=float).reshape(4, 3)
                self._fitness_vals = np.array([4.0, 3.0, 2.0, 1.0])
                self._current_iter = 0
                self.gm_offspring_fraction = 0.5
                self.enable_gm(RankGuidedMutation())

            @guided_mutation(merge="replace_worst")
            def step(self):
                return self._population

        host = _Fixed()
        original = host._population.copy()
        merged = host.step()
        assert merged.shape == (4, 3)
        # ranks: row 3 is best (1), row 0 is worst (4) -> rows 0 and 1 replaced.
        assert not np.array_equal(merged[0], original[0])
        assert not np.array_equal(merged[1], original[1])
        assert np.array_equal(merged[2], original[2])
        assert np.array_equal(merged[3], original[3])
        # _gm_commit wrote the merged array back.
        assert np.array_equal(host._population, merged)
