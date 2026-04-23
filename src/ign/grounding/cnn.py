import jax
import jax.numpy as jnp
import equinox as eqx

class CNNFeatureExtractor(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 2)
        self.layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, key=keys[0]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2),
            eqx.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, key=keys[1]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
