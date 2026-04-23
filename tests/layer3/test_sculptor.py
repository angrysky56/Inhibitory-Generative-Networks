import jax
import jax.numpy as jnp
from ign.sculptor.sculptor import Sculptor

def test_sculptor_masking():
    sculptor = Sculptor()
    ctrnn_state = jnp.ones((10, 10))
    mask = jnp.full((10, 10), 0.5)
    
    output = sculptor(ctrnn_state, mask)
    
    assert jnp.allclose(output, 0.5)

def test_sculptor_clipping():
    sculptor = Sculptor(clip_min=-1.0, clip_max=1.0)
    ctrnn_state = jnp.full((10, 10), 2.0)
    mask = jnp.ones((10, 10))
    
    output = sculptor(ctrnn_state, mask)
    
    assert jnp.allclose(output, 1.0)
