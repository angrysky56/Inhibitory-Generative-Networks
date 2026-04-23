import jax
import jax.numpy as jnp
from ign.substrate.simulation import simulate_substrate

def test_simulate_substrate():
    key = jax.random.PRNGKey(42)
    y0 = jax.random.normal(key, (10, 10))
    t0 = 0.0
    t1 = 1.0
    dt0 = 0.05
    
    sol = simulate_substrate(key, y0, t0, t1, dt0)
    
    assert not jnp.isnan(sol.ys).any()
