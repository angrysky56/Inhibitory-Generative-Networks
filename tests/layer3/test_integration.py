import jax
import jax.numpy as jnp
from ign.model import IGNModel

def test_ign_forward_pass():
    key = jax.random.PRNGKey(42)
    target_shape = (10, 10)
    model = IGNModel(key=key, target_shape=target_shape)
    
    # Dummy inputs
    x = jnp.ones((1, 28, 28)) # Dummy image
    y0 = jnp.zeros(target_shape) # Initial substrate state
    t0 = 0.0
    t1 = 1.0
    dt0 = 0.05
    
    key_run = jax.random.PRNGKey(43)
    
    output = model(x, y0, t0, t1, dt0, key_run)
    
    assert output.shape == target_shape
    assert not jnp.isnan(output).any()
