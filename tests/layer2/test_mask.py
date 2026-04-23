import jax
import jax.numpy as jnp
from ign.grounding.mask import MetaLogicalGrounding

def test_grounding_mask():
    key = jax.random.PRNGKey(0)
    in_features = 128
    target_shape = (10, 10)
    
    grounding = MetaLogicalGrounding(in_features, target_shape, key)
    
    dummy_input = jnp.zeros(in_features)
    output = grounding(dummy_input)
    
    assert output.shape == target_shape
    assert jnp.all(output <= 1.0)
    assert jnp.all(output >= 0.0)
