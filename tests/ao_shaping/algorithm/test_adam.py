import numpy as np

from ao_shaping.algorithm import Adam, AdamW, SGD, AdaMOD


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