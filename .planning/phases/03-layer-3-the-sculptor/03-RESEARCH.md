# Phase 3: Layer 3 - The Sculptor - Research

## Context and Goal
The goal of this phase is to "Connect the layers via inhibitory masking." We are implementing the final "Sculptor" layer that takes the stochastic substrate states (Layer 1) and the feature-based meta-logical grounding masks (Layer 2) and combines them to produce the sculpted output grid.

## Architectural Analysis

### Layer 1: Substrate (Stochastic CTRNN)
- Defined in `src/ign/substrate/simulation.py` and `ctrnn.py`.
- Generates a sequence of states using `diffrax` simulating a continuous-time Langevin process.
- The output `sol.ys` contains the grid states over time. The final state `sol.ys[-1]` is typically the stable grid output.
- Shape: `(H, W)` depending on initial configuration.

### Layer 2: Meta-Logical Grounding Engine
- Defined in `src/ign/grounding/cnn.py` and `mask.py`.
- `CNNFeatureExtractor` processes the input feature/target `x`.
- `MetaLogicalGrounding` takes the CNN features and produces a mask mapped to `target_shape`.
- The mask values are squashed to `[0, 1]` via `jax.nn.sigmoid`.

### Layer 3: The Sculptor (To be built)
Will reside in a new module, e.g., `src/ign/sculptor/` or `src/ign/sculpting/`.

#### SCU-01: Hadamard Product Application
- The mask from Layer 2 needs to be element-wise multiplied with the CTRNN output from Layer 1.
- Since it is "inhibitory masking", the mask represents the non-inhibited parts (where mask $\approx 1$, state is kept; where mask $\approx 0$, state is suppressed).
- Implementation: `sculpted_state = ctrnn_state * mask`. This natively works in JAX via `jnp.multiply(ctrnn_state, mask)` or simply `*`.

#### SCU-02: Clipping/Thresholding
- Post-inhibition, the states might need to be clamped to a certain range (e.g., `[-1, 1]`, matching the typical output range of `tanh` coupling in the CTRNN, or `[0, 1]`).
- Implementation: `jnp.clip(sculpted_state, min_val, max_val)` or a hard threshold `jnp.where(sculpted_state > threshold, 1.0, 0.0)`. A standard configurable `clip` or `threshold` function should be provided as part of the sculpting module.

#### SCU-03: 3-Layer Forward Pass Integration
- We need a top-level module (e.g., `IGNForward` or `InhibitoryGenerativeNetwork` in `src/ign/core.py` or similar) that takes:
  1. `input_target` (for the CNN/Grounding layer).
  2. `y0`, `t0`, `t1`, `dt0` (for the Substrate simulation).
  3. A `jax.random.PRNGKey` for random processes.
- The forward pass will:
  1. Generate the mask: `mask = GroundingLayer(input_target)`
  2. Simulate the substrate: `ctrnn_state = SubstrateLayer(y0, t0, t1, dt0, key).ys[-1]`
  3. Sculpt: `output = SculptorLayer(ctrnn_state, mask)`
- This integrates the entire model into a single JAX/Equinox forward pass.

## Implementation Plan Strategy (For Planner)
1. **Create Sculptor Module**: Create `src/ign/sculptor/sculptor.py` containing an `eqx.Module` called `Sculptor` that performs the masking (SCU-01) and clipping (SCU-02).
2. **Create Core IGN Model**: Create `src/ign/model.py` (or `core.py`) that wires together `CNNFeatureExtractor`, `MetaLogicalGrounding`, `simulate_substrate`, and `Sculptor` (SCU-03).
3. **Tests**: Create integration tests `tests/layer3/test_sculptor.py` to ensure the Hadamard product broadcasts correctly and the full 3-layer pass executes without JAX tracing errors.

## Validation Architecture
- **Unit Test**: Test `Sculptor` module directly with dummy masks and dummy CTRNN states. Assert shapes and values post-clipping/thresholding.
- **Integration Test**: Run the full `IGNModel(x, y0, ...)` pass. Assert that the output shape matches the `target_shape` of the grounding mask and the dimensions of `y0`.
- Verify JAX transformations (`jax.jit`, `jax.vmap`) work over the complete forward pass.
