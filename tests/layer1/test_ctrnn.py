import jax
import jax.numpy as jnp
from ign.substrate.ctrnn import CTRNNGrid

def test_ctrnn_drift():
    key = jax.random.PRNGKey(0)
    ctrnn = CTRNNGrid(key)
    y = jnp.zeros((10, 10))
    t = 0.0
    args = None
    output = ctrnn.drift(t, y, args)
    assert output.shape == y.shape
