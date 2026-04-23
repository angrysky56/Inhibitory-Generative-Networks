# Phase 2 Research: Layer 2 - Meta-Logical Grounding Engine

## Standard Stack
- **`jax` / `jax.numpy`**: Core tensor and mathematical operations.
- **`equinox`**: The established neural network framework in this project for creating the Convolutional Neural Network (CNN) feature extractor and other network blocks.

## Architecture Patterns
- **Equinox Module Composition**: The CNN and grounding logic should subclass `eqx.Module`. Using Equinox ensures the model is treated as a valid PyTree, allowing it to seamlessly integrate with `jax.jit` and gradients.
- **Spatial Mapping Strategy**: The CNN acts as a spatial feature extractor (e.g., from an MNIST image, 28x28). The Meta-Logical Grounding block translates these features into an inhibitory masking tensor. 
- **Sigmoid Output**: Because this layer's output will be used as an *inhibitory mask* via a Hadamard product (in Phase 3), the output must be constrained to the `[0, 1]` range. Applying `jax.nn.sigmoid` at the final layer of the Grounding block is the standard approach to generate valid probabilistic mask values.
- **Dimensionality Alignment**: The output dimensions of the Grounding block must exactly match the spatial dimensions of the Layer 1 CTRNN grid (e.g. if the CTRNN simulates an $N \times M$ grid, the mask must be broadcastable to or exactly $N \times M$).

## Don't Hand-Roll
- **Convolution Operations**: Use `eqx.nn.Conv2d` instead of writing custom lax convolution wrappers.
- **Pooling/Flattening**: Rely on Equinox abstractions like `eqx.nn.MaxPool2d` or basic `jax.numpy.reshape` for standard operations.
- **Activation Functions**: Use `jax.nn.relu` and `jax.nn.sigmoid`.

## Common Pitfalls
- **Shape Mismatches**: A common problem in JAX/Equinox is not keeping strict track of channel dimensions (CHW vs HWC). Equinox's Conv2d uses `(channels, height, width)` natively. The spatial mapping MUST explicitly reshape or pool to match the CTRNN grid.
- **PRNGKey Handling in Equinox**: While constructing Equinox models requires a `PRNGKey` to initialize weights, the forward pass of a deterministic CNN does not. Ensure the initialization uses a key, but the `__call__` method doesn't demand one unless stochasticity (like dropout) is used.
- **PyTree Registration Issues**: Do not use standard Python dataclasses for state variables without explicitly registering them as PyTrees; subclassing `eqx.Module` handles this automatically.

## Code Examples

### Defining an Equinox CNN Feature Extractor
```python
import jax
import jax.numpy as jnp
import equinox as eqx

class CNNFeatureExtractor(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, key=keys[0]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2),
            eqx.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, key=keys[1]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2),
        ]

    def __call__(self, x):
        # x is expected to be shape (1, H, W)
        for layer in self.layers:
            x = layer(x)
        return x
```

### Meta-Logical Grounding Block
```python
class MetaLogicalGrounding(eqx.Module):
    linear: eqx.nn.Linear
    target_shape: tuple
    
    def __init__(self, in_features, target_shape, key):
        self.target_shape = target_shape
        # Flatten features and map to the required spatial target shape
        out_features = target_shape[0] * target_shape[1]
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        
    def __call__(self, x):
        # Flatten the CNN output
        x_flat = x.flatten()
        # Map to target size
        x_mapped = self.linear(x_flat)
        # Reshape to target CTRNN grid
        x_grid = x_mapped.reshape(self.target_shape)
        # Apply sigmoid to constrain mask to [0, 1]
        return jax.nn.sigmoid(x_grid)
```
