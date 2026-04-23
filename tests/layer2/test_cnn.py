import jax
import jax.numpy as jnp
from ign.grounding.cnn import CNNFeatureExtractor

def test_cnn_output_shape():
    key = jax.random.PRNGKey(0)
    cnn = CNNFeatureExtractor(key)
    # Dummy MNIST image (1 channel, 28x28)
    dummy_input = jnp.zeros((1, 28, 28))
    
    output = cnn(dummy_input)
    
    # Assert output has 32 channels. The exact spatial dims will depend on pooling/padding
    # but the instructions say "asserting the output has 32 channels"
    assert output.shape[0] == 32
