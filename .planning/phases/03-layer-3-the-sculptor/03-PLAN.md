---
phase: 3
plan_id: "01"
title: "Implement Sculptor Layer and Full Integration"
description: "Implement the Sculptor module applying Hadamard product and clipping, and wire up the complete IGN 3-layer forward pass."
requirements_addressed: ["SCU-01", "SCU-02", "SCU-03"]
wave: 1
depends_on: []
files_modified:
  - tests/layer3/test_sculptor.py
  - tests/layer3/test_integration.py
  - src/ign/sculptor/__init__.py
  - src/ign/sculptor/sculptor.py
  - src/ign/model.py
  - src/ign/__init__.py
autonomous: true
---

# Phase 3, Plan 01: Implement Sculptor Layer and Full Integration

<objective>
To implement the final layer of the Inhibitory Generative Network: the Sculptor layer. We will create the `Sculptor` class which applies inhibitory masks to the stochastic substrate via the Hadamard product and clips the bounds. Then, we will integrate all three layers (CNN + MetaLogicalGrounding, Substrate Langevin, Sculptor) into a top-level `IGNModel` forward pass.
</objective>

<threat_model>
- Ensure all states remain numerically stable and properly bounded to avoid NaN propagation during Euler-Maruyama diffrax solves.
- Validation bounding tests exist to prevent unrestricted exponential growth inside `Sculptor`.
</threat_model>

<tasks>

<task>
<id>3-01-01</id>
<title>Create Layer 3 Test Infrastructure (Wave 0)</title>
<type>tdd</type>
<read_first>
- .planning/phases/03-layer-3-the-sculptor/03-VALIDATION.md
- tests/layer1/test_integration.py
</read_first>
<action>
Create `tests/layer3/test_sculptor.py` and `tests/layer3/test_integration.py`.

In `test_sculptor.py`:
- Add test `test_sculptor_masking` which initializes `Sculptor`, passes dummy `ctrnn_state` (e.g. all 1s) and `mask` (e.g. 0.5 everywhere), and asserts the product is 0.5.
- Add test `test_sculptor_clipping` which passes `ctrnn_state` outside bounds (e.g. 2.0) and a mask of 1.0, asserting output is clipped to 1.0.

In `test_integration.py`:
- Add test `test_ign_forward_pass` which initializes the `IGNModel(key)`, creates dummy inputs `x` and `y0`, and runs the full forward pass, asserting the output shape matches `y0.shape`.
</action>
<acceptance_criteria>
- `ls tests/layer3/test_sculptor.py` succeeds.
- `ls tests/layer3/test_integration.py` succeeds.
- Tests should fail correctly before implementation via `pytest tests/layer3/test_sculptor.py -x` and `pytest tests/layer3/test_integration.py -x`.
</acceptance_criteria>
</task>

<task>
<id>3-01-02</id>
<title>Implement Sculptor Module</title>
<type>execute</type>
<read_first>
- src/ign/grounding/mask.py
- src/ign/substrate/simulation.py
</read_first>
<action>
Create `src/ign/sculptor/__init__.py` and `src/ign/sculptor/sculptor.py`.

In `sculptor.py`:
```python
import jax
import jax.numpy as jnp
import equinox as eqx

class Sculptor(eqx.Module):
    clip_min: float = -1.0
    clip_max: float = 1.0

    def __call__(self, ctrnn_state, mask):
        # SCU-01: Hadamard product
        sculpted = ctrnn_state * mask
        # SCU-02: Clipping
        return jnp.clip(sculpted, self.clip_min, self.clip_max)
```
In `__init__.py`:
```python
from .sculptor import Sculptor
```
</action>
<acceptance_criteria>
- `cat src/ign/sculptor/sculptor.py | grep "class Sculptor(eqx.Module):"` matches.
- `pytest tests/layer3/test_sculptor.py` passes successfully.
</acceptance_criteria>
</task>

<task>
<id>3-01-03</id>
<title>Integrate Full IGN Model</title>
<type>execute</type>
<read_first>
- src/ign/grounding/cnn.py
- src/ign/grounding/mask.py
- src/ign/substrate/simulation.py
- src/ign/sculptor/sculptor.py
</read_first>
<action>
Create `src/ign/model.py`.

In `model.py`, build the top-level class integrating the 3 layers:
```python
import jax
import jax.numpy as jnp
import equinox as eqx
from .grounding.cnn import CNNFeatureExtractor
from .grounding.mask import MetaLogicalGrounding
from .substrate.simulation import simulate_substrate
from .sculptor.sculptor import Sculptor

class IGNModel(eqx.Module):
    cnn: CNNFeatureExtractor
    grounding: MetaLogicalGrounding
    sculptor: Sculptor

    def __init__(self, key, target_shape):
        keys = jax.random.split(key, 3)
        self.cnn = CNNFeatureExtractor(key=keys[0])
        # Assuming flattened CNN output feeds into Grounding
        # E.g., for 1x28x28 input, maxpools reduce to something smaller.
        # But we pass a placeholder in_features, e.g. 32*7*7=1568
        self.grounding = MetaLogicalGrounding(in_features=1568, target_shape=target_shape, key=keys[1])
        self.sculptor = Sculptor()

    def __call__(self, x, y0, t0, t1, dt0, key):
        features = self.cnn(x)
        mask = self.grounding(features)
        
        # We need the key to pass to simulate_substrate
        # y0 is typically of target_shape
        sol = simulate_substrate(key, y0, t0, t1, dt0)
        final_state = sol.ys[-1]
        
        sculpted = self.sculptor(final_state, mask)
        return sculpted
```
Update `src/ign/__init__.py` to expose `IGNModel`.
</action>
<acceptance_criteria>
- `pytest tests/layer3/test_integration.py` passes successfully.
- `cat src/ign/model.py | grep "class IGNModel"` matches.
- `python -c 'import ign; ign.IGNModel'` succeeds without ImportError.
</acceptance_criteria>
</task>

</tasks>

<verification>
- `<automated>` Run `pytest tests/layer3/test_sculptor.py -x` and verify unit tests pass.
- `<automated>` Run `pytest tests/layer3/test_integration.py -x` and verify the whole forward pass tests pass.
- `<automated>` Check `pytest tests/` entirely to ensure no previous logic broke.
</verification>

<must_haves>
- The final `IGNModel` module successfully ties together `CNNFeatureExtractor`, `MetaLogicalGrounding`, `simulate_substrate`, and `Sculptor`.
- Hadamard product and bounding clips are executed correctly on `ctrnn_state` and `mask`.
- All requirements SCU-01, SCU-02, and SCU-03 are addressed.
</must_haves>
