# Phase 2 Pattern Map: Layer 2 - Meta-Logical Grounding Engine

## 1. Files to be Created/Modified

### `src/ign/grounding/cnn.py`
- **Role**: Implements the `eqx.Module` subclass for the CNN Feature Extractor.
- **Data Flow**: Takes an input image (e.g. 1x28x28) and processes it through convolutions to produce feature maps.
- **Analog**: `src/ign/substrate/ctrnn.py`
- **Pattern from Analog**:
  ```python
  import jax
  import jax.numpy as jnp
  import equinox as eqx

  class CNNFeatureExtractor(eqx.Module):
      layers: list

      def __init__(self, key):
          ... # Use jax.random keys to initialize eqx layers

      def __call__(self, x):
          ...
  ```

### `src/ign/grounding/mask.py`
- **Role**: Implements the `eqx.Module` subclass for the Meta-Logical Grounding block that translates CNN features into the spatial inhibitory mask.
- **Data Flow**: Flattens the feature map, projects it with a dense layer, reshapes it to the target grid dimensions (e.g. matching CTRNN), and applies `jax.nn.sigmoid`.
- **Analog**: `src/ign/substrate/ctrnn.py` (for the standard module definition pattern)
- **Pattern from Analog**: 
  ```python
  class MetaLogicalGrounding(eqx.Module):
      linear: eqx.nn.Linear
      target_shape: tuple

      def __init__(self, in_features, target_shape, key):
          ...
  ```

### `tests/layer2/test_grounding.py`
- **Role**: Pytest tests to verify that the CNN and Meta-Logical Grounding modules produce tensors of the correct spatial dimensions and valid probability ranges [0, 1].
- **Analog**: `tests/layer1/test_ctrnn.py`
- **Pattern from Analog**: Basic pytest assertions with `jnp` and `jax.random.PRNGKey`.
  ```python
  import jax
  import jax.numpy as jnp
  from ign.grounding.cnn import CNNFeatureExtractor
  
  def test_cnn_output_shape():
      key = jax.random.PRNGKey(0)
      cnn = CNNFeatureExtractor(key)
      x = jnp.zeros((1, 28, 28))
      output = cnn(x)
      assert output.shape == ...
  ```
