import jax
import jax.numpy as jnp
from ign.grounding.cnn import CNNFeatureExtractor
key = jax.random.PRNGKey(0)
cnn = CNNFeatureExtractor(key)
dummy = jnp.zeros((1, 28, 28))
out = cnn(dummy)
print(out.shape)
