import jax
import jax.numpy as jnp
import equinox as eqx

class CTRNNGrid(eqx.Module):
    stencil: jnp.ndarray
    gamma: float

    def __init__(self, key):
        self.stencil = jnp.array([[0.1, 0.2, 0.1], [0.2, 0.0, 0.2], [0.1, 0.2, 0.1]])
        self.gamma = 1.0

    def drift(self, t, y, args):
        padded_y = jnp.pad(y, pad_width=1, mode='wrap')
        coupling = jax.scipy.signal.convolve2d(padded_y, self.stencil, mode='valid')
        return -self.gamma * y + jax.nn.tanh(coupling)
