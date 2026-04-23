import jax
import jax.numpy as jnp
import equinox as eqx
from ign.grounding.cnn import CNNFeatureExtractor
import math

class AdditiveModel(eqx.Module):
    cnn: CNNFeatureExtractor
    linear: eqx.nn.Linear
    target_shape: tuple
    
    def __init__(self, key, target_shape):
        keys = jax.random.split(key, 2)
        self.cnn = CNNFeatureExtractor(key=keys[0])
        in_features = 800  # Based on 1x28x28 input
        out_features = math.prod(target_shape)
        self.linear = eqx.nn.Linear(in_features, out_features, key=keys[1])
        self.target_shape = target_shape
        
    def __call__(self, x):
        features = self.cnn(x)
        features_flat = features.flatten()
        output = self.linear(features_flat)
        return output.reshape(self.target_shape)
