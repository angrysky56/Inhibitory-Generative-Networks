import jax.numpy as jnp
import equinox as eqx

class LangevinNoise(eqx.Module):
    diffusion_coeff: float = 0.5

    def diffusion(self, t, y, args):
        return self.diffusion_coeff * jnp.ones_like(y)
