---
wave: 1
depends_on: []
files_modified:
  - "src/ign/grounding/__init__.py"
  - "src/ign/grounding/cnn.py"
  - "src/ign/grounding/mask.py"
  - "tests/layer2/test_cnn.py"
  - "tests/layer2/test_mask.py"
  - "tests/layer2/test_grounding_integration.py"
autonomous: true
---

# Phase 2: Layer 2 - Meta-Logical Grounding Engine

## Goal
Build the CNN and mask generation logic.

## Verification
- `pytest tests/layer2/` must pass.
- CNN must process a dummy MNIST image into a mask tensor matching the CTRNN dimensions.
- Mask values must be bounded within `[0, 1]`.

## Must Haves
- JAX/Equinox object-oriented module design.
- CNN extracts spatial features.
- Meta-Logical Grounding block translates features into an inhibitory mask.
- Sigmoid activation on the final mask output.

## Tasks

<task>
  <id>2-01-01</id>
  <title>Initialize Grounding Module Structure</title>
  <requirements>GND-01</requirements>
  <read_first>
    - .planning/phases/02-layer-2-meta-logical-grounding-engine/02-RESEARCH.md
  </read_first>
  <action>
    Create the standard Python project structure for Layer 2: `src/ign/grounding/` and `tests/layer2/`.
    Create an empty `src/ign/grounding/__init__.py`.
  </action>
  <acceptance_criteria>
    - Directory `src/ign/grounding/` exists.
    - Directory `tests/layer2/` exists.
    - `src/ign/grounding/__init__.py` exists.
  </acceptance_criteria>
</task>

<task>
  <id>2-01-02</id>
  <title>Implement CNN Feature Extractor</title>
  <requirements>GND-01</requirements>
  <read_first>
    - .planning/phases/02-layer-2-meta-logical-grounding-engine/02-RESEARCH.md
    - .planning/phases/02-layer-2-meta-logical-grounding-engine/02-PATTERNS.md
  </read_first>
  <action>
    Create `src/ign/grounding/cnn.py`.
    Import `jax`, `jax.numpy as jnp`, and `equinox as eqx`.
    Define `class CNNFeatureExtractor(eqx.Module)`:
    - Add field `layers: list`.
    - `__init__(self, key)` initializes the following sequence in `self.layers`:
      - `eqx.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, key=keys[0])`
      - `jax.nn.relu`
      - `eqx.nn.MaxPool2d(kernel_size=2)`
      - `eqx.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, key=keys[1])`
      - `jax.nn.relu`
      - `eqx.nn.MaxPool2d(kernel_size=2)`
    - `__call__(self, x)` passes `x` sequentially through `self.layers` and returns the output.
    Create `tests/layer2/test_cnn.py`:
    - Write a unit test `test_cnn_output_shape` passing a `jnp.zeros((1, 28, 28))` tensor and asserting the output has `32` channels.
  </action>
  <acceptance_criteria>
    - `src/ign/grounding/cnn.py` contains `class CNNFeatureExtractor(eqx.Module)`.
    - `tests/layer2/test_cnn.py` contains `test_cnn_output_shape`.
    - `pytest tests/layer2/test_cnn.py` exits 0.
  </acceptance_criteria>
</task>

<task>
  <id>2-01-03</id>
  <title>Implement Meta-Logical Grounding Block</title>
  <requirements>GND-02, GND-03</requirements>
  <read_first>
    - .planning/phases/02-layer-2-meta-logical-grounding-engine/02-RESEARCH.md
    - .planning/phases/02-layer-2-meta-logical-grounding-engine/02-PATTERNS.md
    - src/ign/grounding/cnn.py
  </read_first>
  <action>
    Create `src/ign/grounding/mask.py`.
    Import `jax`, `jax.numpy as jnp`, `equinox as eqx`.
    Define `class MetaLogicalGrounding(eqx.Module)`:
    - Fields: `linear: eqx.nn.Linear`, `target_shape: tuple`.
    - `__init__(self, in_features, target_shape, key)` initializes `self.target_shape = target_shape` and a linear layer mapping `in_features` to `target_shape[0] * target_shape[1]`.
    - `__call__(self, x)`:
      - Flattens `x`.
      - Applies `self.linear`.
      - Reshapes to `self.target_shape`.
      - Applies `jax.nn.sigmoid` to return mask values in `[0, 1]`.
    Create `tests/layer2/test_mask.py`:
    - Write `test_grounding_mask` passing dummy flattened features of size 128 to output shape `(10, 10)`. Assert output shape is `(10, 10)` and values are `<= 1.0` and `>= 0.0`.
  </action>
  <acceptance_criteria>
    - `src/ign/grounding/mask.py` contains `class MetaLogicalGrounding(eqx.Module)`.
    - `__call__` method applies `jax.nn.sigmoid` to constraint output to [0, 1].
    - `tests/layer2/test_mask.py` asserts shape and value boundaries.
    - `pytest tests/layer2/test_mask.py` exits 0.
  </acceptance_criteria>
</task>

<task>
  <id>2-01-04</id>
  <title>Define Spatial Mapping Integration (CNN to CTRNN Grid)</title>
  <requirements>GND-03</requirements>
  <read_first>
    - src/ign/grounding/cnn.py
    - src/ign/grounding/mask.py
  </read_first>
  <action>
    Create `tests/layer2/test_grounding_integration.py`.
    Import `jax`, `jax.numpy as jnp`.
    Import `CNNFeatureExtractor` and `MetaLogicalGrounding`.
    Write `test_cnn_to_mask_integration`:
    - Instantiate `CNNFeatureExtractor` with `PRNGKey(0)`.
    - Instantiate `MetaLogicalGrounding` taking input size `32 * 5 * 5` (800) and target size `(10, 10)` using `PRNGKey(1)`.
    - Create dummy input image `jnp.zeros((1, 28, 28))`.
    - Forward pass through CNN, then pass result to Grounding block.
    - Assert output shape is exactly `(10, 10)`.
  </action>
  <acceptance_criteria>
    - `tests/layer2/test_grounding_integration.py` contains `test_cnn_to_mask_integration`.
    - The integration test connects `CNNFeatureExtractor` and `MetaLogicalGrounding`.
    - `pytest tests/layer2/test_grounding_integration.py` exits 0.
  </acceptance_criteria>
</task>
