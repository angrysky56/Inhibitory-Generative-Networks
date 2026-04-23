import jax
import jax.numpy as jnp
from ign.grounding.cnn import CNNFeatureExtractor
from ign.grounding.mask import MetaLogicalGrounding

def test_cnn_to_mask_integration():
    cnn_key = jax.random.PRNGKey(0)
    grounding_key = jax.random.PRNGKey(1)
    
    cnn = CNNFeatureExtractor(cnn_key)
    # Output of CNN with 28x28 input has spatial shape 5x5 with 32 channels = 800 features
    grounding = MetaLogicalGrounding(in_features=800, target_shape=(10, 10), key=grounding_key)
    
    dummy_input = jnp.zeros((1, 28, 28))
    
    cnn_features = cnn(dummy_input)
    mask = grounding(cnn_features)
    
    assert mask.shape == (10, 10)
    assert jnp.all(mask <= 1.0)
    assert jnp.all(mask >= 0.0)
