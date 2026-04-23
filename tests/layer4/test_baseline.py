import jax
import jax.numpy as jnp
from ign.evaluation.baseline import AdditiveModel

def test_additive_baseline_forward():
    """Test that AdditiveModel produces output of target_shape."""
    key = jax.random.PRNGKey(0)
    target_shape = (5, 5)
    model = AdditiveModel(key=key, target_shape=target_shape)
    
    # Mock image input (e.g., 1x28x28)
    x = jnp.zeros((1, 28, 28))
    
    output = model(x)
    assert output.shape == target_shape
