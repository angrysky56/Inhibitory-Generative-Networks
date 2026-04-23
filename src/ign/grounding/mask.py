import jax
import jax.numpy as jnp
import equinox as eqx

class MetaLogicalGrounding(eqx.Module):
    linear: eqx.nn.Linear
    target_shape: tuple

    def __init__(self, in_features, target_shape, key):
        self.target_shape = target_shape
        self.linear = eqx.nn.Linear(in_features, target_shape[0] * target_shape[1], key=key)

    def __call__(self, x):
        x = x.flatten()
        x = self.linear(x)
        x = x.reshape(self.target_shape)
        return jax.nn.sigmoid(x)
