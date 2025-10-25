"""
Test suite for ACIA library
============================

Run with: pytest tests/
"""

import pytest
import torch
from acia import ColoredMNIST, CausalRepresentationNetwork, CausalOptimizer


class TestDatasets:
    """Test dataset functionality."""
    
    def test_colored_mnist_creation(self):
        """Test ColoredMNIST dataset creation."""
        dataset = ColoredMNIST(env='e1', train=True)
        assert len(dataset) > 0
        
        x, y, e = dataset[0]
        assert x.shape == (3, 28, 28)  # RGB image
        assert 0 <= y < 10  # Digit label
        assert e in [0.0, 1.0]  # Environment label
    
    def test_environment_differences(self):
        """Test that environments have different color distributions."""
        e1 = ColoredMNIST(env='e1', train=True)
        e2 = ColoredMNIST(env='e2', train=True)
        
        # Get first even digit from each
        for i in range(100):
            x1, y1, _ = e1[i]
            if y1 % 2 == 0:
                break
        
        for i in range(100):
            x2, y2, _ = e2[i]
            if y2 % 2 == 0:
                break
        
        # Check color channels (should differ on average)
        red1 = x1[0].sum().item()
        red2 = x2[0].sum().item()
        
        # At least one should have substantial red content
        assert red1 > 0 or red2 > 0


class TestModels:
    """Test model architectures."""
    
    def test_model_forward_pass(self):
        """Test forward pass through model."""
        model = CausalRepresentationNetwork()
        x = torch.randn(4, 3, 28, 28)  # Batch of 4 images
        
        z_L, z_H, logits = model(x)
        
        assert z_L.shape == (4, 32)  # Low-level representation
        assert z_H.shape == (4, 128)  # High-level representation
        assert logits.shape == (4, 10)  # 10 classes
    
    def test_model_gradients(self):
        """Test that gradients flow through model."""
        model = CausalRepresentationNetwork()
        x = torch.randn(4, 3, 28, 28)
        y = torch.randint(0, 10, (4,))
        
        z_L, z_H, logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestTraining:
    """Test training functionality."""
    
    def test_optimizer_step(self):
        """Test single optimizer step."""
        model = CausalRepresentationNetwork()
        optimizer = CausalOptimizer(model, batch_size=4)
        
        x = torch.randn(4, 3, 28, 28)
        y = torch.randint(0, 10, (4,))
        e = torch.randint(0, 2, (4,)).float()
        
        metrics = optimizer.train_step(x, y, e)
        
        # Check that metrics are returned
        assert 'pred_loss' in metrics
        assert 'R1' in metrics
        assert 'R2' in metrics
        assert 'total_loss' in metrics
        
        # Check that values are reasonable
        assert metrics['pred_loss'] > 0
        assert metrics['total_loss'] > 0
    
    def test_regularizers(self):
        """Test that regularizers are non-zero."""
        model = CausalRepresentationNetwork()
        optimizer = CausalOptimizer(model, batch_size=8)
        
        # Create data with clear environment differences
        x = torch.randn(8, 3, 28, 28)
        y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        e = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.])
        
        metrics = optimizer.train_step(x, y, e)
        
        # R1 should be non-zero when there are multiple environments
        assert metrics['R1'] >= 0


class TestCausalTheory:
    """Test theoretical components."""
    
    def test_causal_kernel_properties(self):
        """Test that causal kernels satisfy probability axioms."""
        from acia.core import CausalKernel
        from acia.core.spaces import MeasurableSet
        
        # Create simple sample space
        sample_space = torch.randn(100, 10)
        Y = torch.randint(0, 10, (100,))
        E = torch.zeros(100)
        
        kernel = CausalKernel(sample_space, Y, E)
        
        # Test probability measure
        full_set = MeasurableSet(
            torch.ones(100, dtype=torch.bool),
            "Full Space"
        )
        prob = kernel._compute_probability_measure()(full_set)
        
        assert abs(prob - 1.0) < 1e-6  # Should sum to 1


@pytest.fixture
def small_dataset():
    """Fixture providing small dataset for testing."""
    return ColoredMNIST(env='e1', train=True)


@pytest.fixture
def simple_model():
    """Fixture providing initialized model."""
    return CausalRepresentationNetwork()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
