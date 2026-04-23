import jax
import jax.numpy as jnp
import equinox as eqx

class Sculptor(eqx.Module):
    clip_min: float = -1.0
    clip_max: float = 1.0

    def __call__(self, ctrnn_state, mask):
        # SCU-01: Hadamard product
        sculpted = ctrnn_state * mask
        # SCU-02: Clipping
        return jnp.clip(sculpted, self.clip_min, self.clip_max)
