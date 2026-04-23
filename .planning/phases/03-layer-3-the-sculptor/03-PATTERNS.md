# Phase 3: Layer 3 - The Sculptor - Patterns

## File Roles and Analogies

| File | Role | Data Flow | Closest Analog | Concrete Excerpt from Analog |
|------|------|-----------|----------------|------------------------------|
| `src/ign/sculptor/sculptor.py` | Defines the Sculptor module applying Hadamard product and clipping | `ctrnn_state, mask -> sculpted_state` | `src/ign/grounding/mask.py` | `MetaLogicalGrounding` class extending `eqx.Module` |
| `src/ign/model.py` | Integrates CNN, Grounding, Substrate, and Sculptor | `x, y0, ... -> output_grid` | `src/ign/substrate/simulation.py` | `simulate_substrate` combining diffrax and langevin |
| `tests/layer3/test_sculptor.py` | Unit and integration tests for the Sculptor and full IGN model | `fixtures -> test pass/fail` | `tests/layer1/test_integration.py` | `test_simulate_substrate` testing forward pass |

## Concrete Code Excerpts

### Analog: `eqx.Module` subclassing
```python
class MetaLogicalGrounding(eqx.Module):
    # fields
    def __init__(self, in_features, target_shape, key):
        # ...

    def __call__(self, x):
        # ...
```
Our `Sculptor` should be defined similarly:
```python
class Sculptor(eqx.Module):
    # threshold or min/max bounds can be static fields
    clip_min: float = -1.0
    clip_max: float = 1.0

    def __call__(self, ctrnn_state, mask):
        # apply Hadamard masking
        sculpted = ctrnn_state * mask
        return jnp.clip(sculpted, self.clip_min, self.clip_max)
```
