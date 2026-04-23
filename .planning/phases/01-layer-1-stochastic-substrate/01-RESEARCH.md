# Phase 1: Layer 1 - Stochastic Substrate Research

## Standard Stack
- `jax`, `jax.numpy`: For foundational tensor operations and automatic differentiation.
- `diffrax`: For solving Stochastic Differential Equations (SDEs). Diffrax is the ecosystem standard for JAX and has explicit support for Underdamped Langevin Diffusion (ULD) and Brownian motion abstractions.
- `equinox`: For structuring the grid and module parameters. It is the dominant choice for object-oriented JAX models (especially in conjunction with diffrax).
- `jax.lax` or `jax.scipy.signal`: For highly optimized convolution operations needed to compute the 2D grid stencil for local connectivity.

## Architecture Patterns
- **Underdamped Langevin Equation**: Implemented using `diffrax.MultiTerm` to combine drift and diffusion. For ULD specifically, diffrax provides `UnderdampedLangevinDriftTerm` and `UnderdampedLangevinDiffusionTerm`.
- **2D Grid Connectivity (CTRNN)**: Instead of a full weight matrix, the grid uses a 2D convolution (`jax.scipy.signal.convolve2d` or `jax.lax.conv`) to represent $W_{local}$. The convolution kernel serves as the connectivity stencil, defining how neighboring neurons influence each other.
- **State Representation**: The state $y$ tracks the membrane potential. For underdamped systems, the state must be a tuple/struct `(position, velocity)` representing the neural state and its temporal derivative.
- **Vectorization**: The solver step is written for a single grid environment and parameterized. `jax.vmap` is used at the top level to parallelize the simulation across a batch dimension.

## Don't Hand-Roll
- **SDE Solvers**: Do not write custom Euler-Maruyama loops using `jax.lax.scan` directly. Use `diffrax` SDE solvers. Hand-rolled loops often fail to handle step-size adaptation or Brownian noise construction correctly.
- **Brownian Noise Generation**: Do not hand-roll Gaussian noise sampling (`jax.random.normal`) per step. Use `diffrax.VirtualBrownianTree` or `diffrax.UnsafeBrownianPath`. These structures guarantee correct statistical properties, reproducible sub-stepping, and correct integration over continuous time intervals.
- **Grid Connectivity**: Do not implement manual nested loops for applying local interactions. Use JAX's optimized convolution primitives.

## Common Pitfalls
- **Numerical Instability in SDEs**: The drift term in Langevin dynamics can lead to stiff differential equations. Using a naive solver (like basic Euler) with a step size ($dt$) that is too large will cause divergence.
  - *Mitigation*: Ensure step sizes are sufficiently small, or use solvers optimized for SDE stability (e.g., `diffrax.EulerHeun` or specialized `diffrax.ALIGN` for ULD).
- **Boundary Conditions on Grid**: When applying the local stencil via convolution, borders can behave erratically.
  - *Mitigation*: Periodic boundaries (`mode='wrap'` in JAX Scipy, or explicit padding before `jax.lax.conv`) are necessary to avoid unnatural edge artifacts in the neural substrate.
- **State Explosion**: Without bounds or a sufficient leak term, the continuous recurrent activity can diverge.
  - *Mitigation*: Ensure the drift function incorporates a stabilizing friction/leak term ($-\gamma y$) and consider a saturating nonlinearity (e.g., `jax.nn.tanh`) on the coupling input to cap local interactions.
- **PRNG Key Management**: SDEs require random keys.
  - *Mitigation*: Do not reuse the same key across steps or batches. Feed a properly split key into the `VirtualBrownianTree`.

## Code Examples

**Basic pattern for Langevin Dynamics in Diffrax:**
```python
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx

class CTRNNGrid(eqx.Module):
    stencil: jnp.ndarray
    gamma: float
    
    def __init__(self, key):
        # 3x3 local connectivity stencil
        self.stencil = jnp.array([[0.1, 0.2, 0.1],
                                  [0.2, 0.0, 0.2],
                                  [0.1, 0.2, 0.1]])
        self.gamma = 1.0 # Leak/friction

    def drift(self, t, y, args):
        # y shape: (grid_h, grid_w)
        # Periodic wrap for boundaries
        coupling = jax.scipy.signal.convolve2d(y, self.stencil, mode='same', boundary='wrap')
        # Simple linear leak + non-linear local excitation
        return -self.gamma * y + jax.nn.tanh(coupling)

    def diffusion(self, t, y, args):
        # Additive noise coefficient
        return 0.5 * jnp.ones_like(y)

def simulate_substrate(key, y0, t0, t1, dt0):
    # 1. Properly abstract the Brownian motion
    bm = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=y0.shape, key=key)
    
    ctrnn = CTRNNGrid(key)
    
    # 2. Combine drift (ODE) and diffusion (Control/Noise) terms
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(ctrnn.drift),
        diffrax.ControlTerm(ctrnn.diffusion, bm)
    )
    
    # 3. Use an SDE-appropriate solver
    solver = diffrax.Euler() # For additive noise, Euler is Euler-Maruyama
    
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100))
    sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, saveat=saveat)
    return sol
```
