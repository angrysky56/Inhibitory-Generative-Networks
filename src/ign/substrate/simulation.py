import diffrax
import jax
import jax.numpy as jnp

from .ctrnn import CTRNNGrid
from .langevin import LangevinNoise


def simulate_substrate(key, y0, t0, t1, dt0, n_save=100, max_steps=4096):
    """Integrate the stochastic substrate from t0 to t1.

    Args:
        key: PRNG key for the Brownian motion driver.
        y0: Initial state, shape (H, W).
        t0, t1: Integration interval in continuous time.
        dt0: Initial step size for the Euler-Maruyama solver.
        n_save: Number of equally-spaced sample points to save across
            [t0, t1]. Default 100 is fine for forward-pass tests; for
            spectral analysis use 2000+ to get a clean PSD.
        max_steps: Cap on internal solver steps. The default of 4096
            is enough for the default forward pass (T=1, dt=0.05). For
            longer integrations or finer dt, pass a larger value:
            roughly 5 * (t1 - t0) / dt0 is a safe upper bound.

    Returns:
        diffrax.Solution. The trajectory is in `sol.ys` with shape
        (n_save, H, W); timestamps in `sol.ts`.
    """
    bm = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=y0.shape, key=key)
    ctrnn = CTRNNGrid(key)
    noise = LangevinNoise()

    import lineax as lx

    def diffusion_wrapper(t, y, args):
        return lx.DiagonalLinearOperator(noise.diffusion(t, y, args))

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(ctrnn.drift), diffrax.ControlTerm(diffusion_wrapper, bm)
    )
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save))

    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        max_steps=max_steps,
    )
    return sol
