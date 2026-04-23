import jax.numpy as jnp
from ign.evaluation.metrics import signature_distance

def test_signature_distance():
    """Test that signature distance correctly measures tensor differences."""
    shape = (2, 2)
    output = jnp.ones(shape)
    target = jnp.ones(shape)
    
    # Identical tensors should have 0.0 distance
    dist = signature_distance(output, target)
    assert jnp.allclose(dist, 0.0)
    
    # Different tensors should have positive distance
    target_diff = jnp.zeros(shape)
    dist_diff = signature_distance(output, target_diff)
    assert dist_diff > 0.0
