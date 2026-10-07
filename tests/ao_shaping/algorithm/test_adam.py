import ast
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.algorithm import Adam, AdamW, SGD, AdaMOD, learning_schedule

#: Files that used to carry their own copy of ``learning_schedule`` (R-43).
LR_SCHEDULE_COPIES = ("src/ao_shaping/tools/train_data_collect.py",)


class TestLearningSchedule:
    """``learning_schedule`` is the canonical LR-decay helper.

    It was exported from :mod:`ao_shaping.algorithm` without a single test, and
    ``tools/train_data_collect.py`` carried a byte-identical copy -- so the two
    could drift silently. These tests pin the behaviour; the guard at the
    bottom stops the copy coming back.
    """

    def test_static_returns_lr_untouched(self):
        for epoch in (0, 1, 50, 99, 100):
            assert learning_schedule(0.7, epoch, 100, method="static") == 0.7

    @pytest.mark.parametrize("method", ["exp", "linear"])
    def test_starts_at_lr_at_epoch_zero(self, method):
        # These two add the 1e-6 floor as a plain offset, so epoch 0 is lr + 1e-6.
        assert learning_schedule(0.5, 0, 100, method=method) == pytest.approx(
            0.5 + 1e-6
        )

    def test_cosine_starts_at_exactly_lr_at_epoch_zero(self):
        """``cosin`` is annealed, not offset -- it starts at exactly ``lr``.

        The other two branches compute ``lr * f(...) + 1e-6``, which makes the
        floor an unconditional ``+1e-6`` offset: at epoch 0 they return
        ``lr + 1e-6`` rather than ``lr``. ``cosin`` now uses the textbook form
        ``lr_min + (lr - lr_min) * (1 + cos(pi*e/E)) / 2`` (identical to
        ``optimizer.wf.rms_by_zernike.cosine_annealing_lr``), in which the floor
        is a floor: it is what the schedule *converges to*, so epoch 0 is exactly
        ``lr`` and ``epoch == epochs`` is exactly the floor.
        """
        assert learning_schedule(0.5, 0, 100, method="cosin") == pytest.approx(0.5)

    def test_cosine_decays_and_matches_the_closed_form(self):
        values = [learning_schedule(1.0, e, 100, method="cosin") for e in range(101)]
        assert values[0] == pytest.approx(1.0)
        # Halfway through, the textbook anneal sits at the midpoint of [lr, floor],
        # NOT at the floor (reaching the floor at E/2 was the old half-wave's
        # behaviour). Exact value: 1e-6 + (lr - 1e-6) * (1 + cos(pi/2)) / 2.
        assert values[50] == pytest.approx(1e-6 + (1.0 - 1e-6) / 2, abs=1e-12)
        assert values[100] == pytest.approx(1e-6)
        assert all(b <= a for a, b in zip(values, values[1:], strict=False))

    def test_exponential_decays_monotonically(self):
        values = [learning_schedule(1.0, e, 100, method="exp") for e in range(101)]
        assert all(b < a for a, b in zip(values, values[1:], strict=False))
        assert values[100] == pytest.approx(np.exp(-1.0) + 1e-6)

    def test_linear_halves_at_the_midpoint(self):
        assert learning_schedule(1.0, 50, 100, method="linear") == pytest.approx(
            0.5 + 1e-6
        )
        assert learning_schedule(1.0, 100, 100, method="linear") == pytest.approx(1e-6)

    @pytest.mark.parametrize("method", ["exp", "linear"])
    def test_exp_and_linear_stay_positive_inside_the_run(self, method):
        """The 1e-6 floor keeps these two from reaching exactly 0."""
        for epoch in range(101):
            assert learning_schedule(1e-3, epoch, 100, method=method) > 0.0

    @pytest.mark.parametrize("method", ["cosin", "exp", "linear"])
    def test_no_schedule_ever_returns_a_negative_lr(self, method):
        """Every schedule stays positive across the whole run, R-44.

        The ``cosin`` branch used to be ``lr * cos(pi*e/E) + 1e-6``: a half-wave,
        not an anneal. ``cos`` changes sign at ``E/2``, so the floor never held and
        the second half of every run received ``-lr + 1e-6`` -- the full initial
        magnitude, negated. For an Adam-family update that is a direction
        reversal, not a smaller step; for plain SGD it is a sign flip on the
        gradient term.

        ``-lr + 1e-6`` is also unbounded in the wrong direction as ``lr`` grows,
        so this was never a small-numerical-edge concern.
        """
        for lr in (1e-6, 1e-3, 1.0, 100.0):
            values = [
                learning_schedule(lr, epoch, 100, method=method) for epoch in range(101)
            ]
            assert all(v > 0.0 for v in values), (
                f"{method} went non-positive for lr={lr}: min={min(values)}"
            )

    def test_cosine_ends_on_the_floor_and_never_reverses(self):
        """The corrected ``cosin`` contract, replacing the old pinned-negative test.

        The predecessor asserted ``learning_schedule(1e-3, 75, 100) < 0`` and
        ``learning_schedule(1e-3, 100, 100) == approx(-1e-3 + 1e-6)``. It was
        written as an explicit refusal to fix the behaviour silently ("Left as-is:
        changing it would alter the trajectory of every caller. Recorded in
        TODO.md (R-44) rather than silently 'fixed'"), so it documents the defect
        precisely. R-44 is this issue; the schedule is now the standard anneal.
        """
        # Decays on the first half, exactly as before the fix.
        assert learning_schedule(1e-3, 25, 100, method="cosin") > 0.0
        # Lands on the floor at the end instead of at -lr.
        assert learning_schedule(1e-3, 100, 100, method="cosin") == pytest.approx(1e-6)
        # The old sign flip is gone.
        assert learning_schedule(1e-3, 75, 100, method="cosin") > 0.0
        # Still monotone: a schedule that recovers after the midpoint would be a
        # different defect.
        values = [learning_schedule(1e-3, e, 100, method="cosin") for e in range(101)]
        assert all(b <= a for a, b in zip(values, values[1:], strict=False))

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="static, cosin, exp or linear"):
            learning_schedule(1.0, 0, 100, method="nope")

    def test_default_method_is_static(self):
        assert learning_schedule(0.3, 42, 100) == 0.3


def test_learning_schedule_is_not_re_duplicated():
    """The copy in ``train_data_collect.py`` was removed; keep it removed.

    Checked on the AST rather than by importing: that module pulls in matplotlib
    and four hardware drivers at module scope, and a source-level check is
    enough to catch a re-definition.
    """
    root = Path(__file__).resolve().parents[3]
    for rel in LR_SCHEDULE_COPIES:
        path = root / rel
        assert path.is_file(), rel
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        defined = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "learning_schedule"
        ]
        assert defined == [], f"{rel} re-defines learning_schedule; import it instead"
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "ao_shaping.algorithm.gradient.adam"
            for alias in node.names
        ]
        assert "learning_schedule" in imported, (
            f"{rel} uses learning_schedule but does not import the canonical one"
        )


class TestAdaMOD:
    """Test AdaMOD optimizer"""
    
    def test_initialization(self):
        """Test AdaMOD initialization with default parameters"""
        dim = 10
        lr = 0.001
        optimizer = AdaMOD(dim=dim, lr=lr)
        
        # Check basic attributes
        assert optimizer.dim == dim
        assert optimizer.lr == lr
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.99
        assert optimizer.beta3 == 0.9995
        assert optimizer.s == 0.0
        assert optimizer.t == 0
        
        # Check inherited attributes from Adam
        assert np.all(optimizer.m == 0), f"m should be initialized to zeros, but got {optimizer.m}"
        assert np.all(optimizer.v == 0), f"v should be initialized to zeros, but got {optimizer.v}"
        assert optimizer.m.shape == (dim,), f"m shape should be {dim}, but got {optimizer.m.shape}"
        assert optimizer.v.shape == (dim,), f"v shape should be {dim}, but got {optimizer.v.shape}"
    
    def test_initialization_with_custom_parameters(self):
        """Test AdaMOD initialization with custom parameters"""
        dim = 5
        lr = 0.01
        beta1 = 0.8
        beta2 = 0.95
        beta3 = 0.999
        optimizer = AdaMOD(dim=dim, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3)
        
        assert optimizer.dim == dim
        assert optimizer.lr == lr
        assert optimizer.beta1 == beta1
        assert optimizer.beta2 == beta2
        assert optimizer.beta3 == beta3
        assert optimizer.s == 0.0
    
    def test_update_step(self):
        """Test AdaMOD update step"""
        dim = 5
        lr = 0.001
        optimizer = AdaMOD(dim=dim, lr=lr)
        
        # Create a simple gradient
        grad = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        
        # Perform update
        update = optimizer.update(grad)
        
        # Check that update has correct shape
        assert update.shape == grad.shape
        
        # Check that time step was incremented
        assert optimizer.t == 1
        
        # Check that moments were updated
        assert not np.all(optimizer.m == 0)
        assert not np.all(optimizer.v == 0)
        
        # Check that s was updated
        assert np.all(optimizer.s != 0.0)
    
    def test_multiple_updates(self):
        """Test multiple AdaMOD update steps"""
        dim = 3
        lr = 0.01
        optimizer = AdaMOD(dim=dim, lr=lr)
        
        # Perform multiple updates
        grads = [
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, -2.0, -3.0]),
            np.array([0.5, -0.5, 1.0])
        ]
        
        for grad in grads:
            update = optimizer.update(grad)
            assert update.shape == grad.shape
        
        # After 3 updates, t should be 3
        assert optimizer.t == 3
        
        # All values should be updated
        assert not np.all(optimizer.m == 0)
        assert not np.all(optimizer.v == 0)
        assert np.any(optimizer.s != 0.0)
    
    def test_adamod_specific_behavior(self):
        """Test AdaMOD-specific behavior with long-term learning rate buffering"""
        dim = 2
        lr = 0.1
        beta3 = 0.9  # Use a lower beta3 to see the effect more clearly
        optimizer = AdaMOD(dim=dim, lr=lr, beta3=beta3)
        
        # Use constant gradient to see how s evolves
        grad = np.array([1.0, 1.0])
        
        # Perform several updates
        updates = []
        for i in range(5):
            update = optimizer.update(grad)
            updates.append(update.copy())
        
        # Check that we have 5 updates
        assert len(updates) == 5
        
        # In AdaMOD, the learning rate gets adjusted based on the long-term average
        # This should result in different update values compared to standard Adam
        # We won't check exact values since they depend on the internal calculations,
        # but we verify that updates happened and have reasonable properties
        for update in updates:
            assert not np.isnan(update).any()
            assert not np.isinf(update).any()
    
    def test_edge_case_zero_gradient(self):
        """Test AdaMOD with zero gradient"""
        dim = 3
        lr = 0.01
        optimizer = AdaMOD(dim=dim, lr=lr)
        
        grad = np.zeros(dim)
        update = optimizer.update(grad)
        
        # With zero gradient, update should be zero or very close to zero
        assert np.allclose(update, 0.0, atol=1e-10)
        assert optimizer.t == 1
    
    def test_edge_case_large_gradient(self):
        """Test AdaMOD with large gradient values"""
        dim = 2
        lr = 0.001
        optimizer = AdaMOD(dim=dim, lr=lr)
        
        # Large gradient values
        grad = np.array([1000.0, -1000.0])
        update = optimizer.update(grad)
        
        # Should not produce NaN or infinity
        assert not np.isnan(update).any()
        assert not np.isinf(update).any()
        assert optimizer.t == 1


class TestSGD:
    """Test SGD optimizer"""
    
    def test_initialization(self):
        """Test SGD initialization"""
        dim = 10
        lr = 0.01
        optimizer = SGD(dim=dim, lr=lr)
        
        assert optimizer.dim == dim
        assert optimizer.lr == lr
        assert optimizer.t == 0
    
    def test_update(self):
        """Test SGD update"""
        dim = 5
        lr = 0.1
        optimizer = SGD(dim=dim, lr=lr)
        
        grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        update = optimizer.update(grad)
        
        expected_update = lr * grad
        np.testing.assert_array_almost_equal(update, expected_update)
        assert optimizer.t == 1


class TestAdam:
    """Test Adam optimizer"""
    
    def test_initialization(self):
        """Test Adam initialization"""
        dim = 10
        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        optimizer = Adam(dim=dim, lr=lr, beta1=beta1, beta2=beta2)
        
        assert optimizer.dim == dim
        assert optimizer.lr == lr
        assert optimizer.beta1 == beta1
        assert optimizer.beta2 == beta2
        assert optimizer.t == 0
        assert optimizer.m.shape == (dim,)
        assert optimizer.v.shape == (dim,)
        assert np.all(optimizer.m == 0)
        assert np.all(optimizer.v == 0)
    
    def test_update(self):
        """Test Adam update"""
        dim = 5
        lr = 0.001
        optimizer = Adam(dim=dim, lr=lr)
        
        grad = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        update = optimizer.update(grad)
        
        # Check that update has correct shape
        assert update.shape == grad.shape
        # Check that time step was incremented
        assert optimizer.t == 1
        # Check that moments were updated
        assert not np.all(optimizer.m == 0)
        assert not np.all(optimizer.v == 0)


class TestAdamW:
    """Test AdamW optimizer"""
    
    def test_initialization(self):
        """Test AdamW initialization"""
        dim = 10
        lr = 0.001
        weight_decay = 0.01
        optimizer = AdamW(dim=dim, lr=lr, weight_decay=weight_decay)
        
        assert optimizer.dim == dim
        assert optimizer.lr == lr
        assert optimizer.weight_decay == weight_decay
        assert optimizer.t == 0
    
    def test_update(self):
        """Test AdamW update"""
        dim = 5
        lr = 0.001
        weight_decay = 0.01
        optimizer = AdamW(dim=dim, lr=lr, weight_decay=weight_decay)
        
        grad = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        update = optimizer.update(grad)
        
        # Check that update has correct shape
        assert update.shape == grad.shape
        # Check that time step was incremented
        assert optimizer.t == 1