import jax.numpy as jnp
from ign.substrate.langevin import LangevinNoise

def test_langevin_diffusion():
    noise = LangevinNoise()
    y = jnp.zeros((10, 10))
    t = 0.0
    args = None
    output = noise.diffusion(t, y, args)
    assert output.shape == y.shape
