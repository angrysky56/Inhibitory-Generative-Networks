---
wave: 1
depends_on: []
files_modified:
  - "requirements.txt"
  - "src/ign/substrate/langevin.py"
  - "src/ign/substrate/ctrnn.py"
  - "src/ign/substrate/simulation.py"
  - "tests/layer1/test_langevin.py"
  - "tests/layer1/test_ctrnn.py"
  - "tests/layer1/test_integration.py"
autonomous: true
---

# Phase 1: Layer 1 - Stochastic Substrate

## Goal
Build a stable, differentiable simulation of continuous neural noise using a Langevin CTRNN grid in JAX.

## Verification
- `pytest tests/layer1/` must pass.
- The Diffrax simulation must integrate over time without diverging to NaN.

## Must Haves
- JAX/Equinox object-oriented module design.
- Diffrax SDE integration (Euler-Maruyama).
- Local 2D stencil convolution using wrap boundaries.

## Tasks

<task>
  <id>1-01-01</id>
  <title>Initialize Project Structure and Dependencies</title>
  <requirements>SUB-01</requirements>
  <read_first>
    - .planning/phases/01-layer-1-stochastic-substrate/01-RESEARCH.md
  </read_first>
  <action>
    Create the standard Python project structure: `src/ign/substrate/` and `tests/layer1/`.
    Create a `requirements.txt` file and add the following dependencies: `jax`, `jaxlib`, `equinox`, `diffrax`, `pytest`.
    Create an empty `src/ign/__init__.py` and `src/ign/substrate/__init__.py`.
  </action>
  <acceptance_criteria>
    - `requirements.txt` contains `jax`, `equinox`, `diffrax`, `pytest`.
    - Directory `src/ign/substrate/` exists.
    - Directory `tests/layer1/` exists.
  </acceptance_criteria>
</task>

<task>
  <id>1-01-02</id>
  <title>Implement Langevin Drift and Diffusion Functions</title>
  <requirements>SUB-01, SUB-03</requirements>
  <read_first>
    - .planning/phases/01-layer-1-stochastic-substrate/01-RESEARCH.md
  </read_first>
  <action>
    Create `src/ign/substrate/langevin.py`.
    Import `jax.numpy as jnp` and `equinox as eqx`.
    Define `class LangevinNoise(eqx.Module)`:
    - Add field `diffusion_coeff: float = 0.5`.
    - Define method `diffusion(self, t, y, args)` that returns `self.diffusion_coeff * jnp.ones_like(y)`.
    Create `tests/layer1/test_langevin.py`:
    - Write a basic unit test `test_langevin_diffusion` verifying that the output shape matches input `y` shape.
  </action>
  <acceptance_criteria>
    - `src/ign/substrate/langevin.py` contains `class LangevinNoise(eqx.Module)`.
    - `tests/layer1/test_langevin.py` contains `test_langevin_diffusion`.
    - `pytest tests/layer1/test_langevin.py` exits 0.
  </acceptance_criteria>
</task>

<task>
  <id>1-01-03</id>
  <title>Implement CTRNN Grid Dynamics</title>
  <requirements>SUB-02</requirements>
  <read_first>
    - .planning/phases/01-layer-1-stochastic-substrate/01-RESEARCH.md
    - src/ign/substrate/langevin.py
  </read_first>
  <action>
    Create `src/ign/substrate/ctrnn.py`.
    Import `jax`, `jax.numpy as jnp`, `equinox as eqx`.
    Define `class CTRNNGrid(eqx.Module)`:
    - Fields: `stencil: jnp.ndarray`, `gamma: float`.
    - `__init__(self, key)` initializes a 3x3 local connectivity array `self.stencil = jnp.array([[0.1, 0.2, 0.1], [0.2, 0.0, 0.2], [0.1, 0.2, 0.1]])` and `self.gamma = 1.0`.
    - Define method `drift(self, t, y, args)`:
      - Applies `coupling = jax.scipy.signal.convolve2d(y, self.stencil, mode='same', boundary='wrap')`.
      - Returns `-self.gamma * y + jax.nn.tanh(coupling)`.
    Create `tests/layer1/test_ctrnn.py`:
    - Write `test_ctrnn_drift` asserting the output of `drift` has the same shape as input `y`.
  </action>
  <acceptance_criteria>
    - `src/ign/substrate/ctrnn.py` contains `class CTRNNGrid(eqx.Module)`.
    - `drift` method uses `jax.scipy.signal.convolve2d` with `mode='same'` and `boundary='wrap'`.
    - `pytest tests/layer1/test_ctrnn.py` exits 0.
  </acceptance_criteria>
</task>

<task>
  <id>1-01-04</id>
  <title>Implement SDE Simulation Integration (Euler-Maruyama)</title>
  <requirements>SUB-04</requirements>
  <read_first>
    - .planning/phases/01-layer-1-stochastic-substrate/01-RESEARCH.md
    - src/ign/substrate/ctrnn.py
    - src/ign/substrate/langevin.py
  </read_first>
  <action>
    Create `src/ign/substrate/simulation.py`.
    Import `jax`, `jax.numpy as jnp`, `diffrax`.
    Import `CTRNNGrid` from `.ctrnn` and `LangevinNoise` from `.langevin`.
    Define `def simulate_substrate(key, y0, t0, t1, dt0):`:
    - Create `bm = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=y0.shape, key=key)`.
    - Instantiate `ctrnn = CTRNNGrid(key)` and `noise = LangevinNoise()`.
    - Setup SDE terms: `terms = diffrax.MultiTerm(diffrax.ODETerm(ctrnn.drift), diffrax.ControlTerm(noise.diffusion, bm))`.
    - Setup solver: `solver = diffrax.Euler()`.
    - Setup save steps: `saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100))`.
    - Solve: `sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, saveat=saveat)`.
    - Return `sol`.
    Create `tests/layer1/test_integration.py`:
    - Write `test_simulate_substrate` passing a random 10x10 array `y0`, `t0=0.0`, `t1=1.0`, `dt0=0.05`. Assert `jnp.isnan(sol.ys).any()` is False.
  </action>
  <acceptance_criteria>
    - `src/ign/substrate/simulation.py` contains `simulate_substrate`.
    - `simulate_substrate` uses `diffrax.VirtualBrownianTree` and `diffrax.Euler`.
    - `tests/layer1/test_integration.py` verifies simulation does not produce NaNs.
    - `pytest tests/layer1/test_integration.py` exits 0.
  </acceptance_criteria>
</task>
