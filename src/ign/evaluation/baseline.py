import jax
import jax.numpy as jnp
import equinox as eqx
from jax import Array
from typing import Tuple
from ign.grounding.cnn import CNNFeatureExtractor
import math

class AdditiveModel(eqx.Module):
    """Comparative additive baseline generative model.
    
    Maps an input image to a target shape using standard additive layers.
    """
    cnn: CNNFeatureExtractor
    linear: eqx.nn.Linear
    target_shape: Tuple[int, ...]
    
    def __init__(self, key: jax.random.PRNGKey, target_shape: Tuple[int, ...]):
        """Initialize the additive model.
        
        Args:
            key: PRNG key for random initialization.
            target_shape: The desired output shape (e.g., (28, 28)).
        """
        keys = jax.random.split(key, 2)
        self.cnn = CNNFeatureExtractor(key=keys[0])
        # Assuming flattened CNN output feeds into Linear
        # From earlier model.py we know CNN output size is 800 for 1x28x28
        in_features = 800
        out_features = math.prod(target_shape)
        self.linear = eqx.nn.Linear(in_features, out_features, key=keys[1])
        self.target_shape = target_shape
        
    def __call__(self, x: Array) -> Array:
        """Forward pass of the additive baseline.
        
        Args:
            x: Input image tensor.
            
        Returns:
            The generated output tensor of shape target_shape.
        """
        features = self.cnn(x)
        features_flat = features.flatten()
        output = self.linear(features_flat)
        return output.reshape(self.target_shape)
