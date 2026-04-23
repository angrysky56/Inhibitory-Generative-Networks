import jax
import jax.numpy as jnp
import diffrax
from .ctrnn import CTRNNGrid
from .langevin import LangevinNoise

def simulate_substrate(key, y0, t0, t1, dt0):
    bm = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=y0.shape, key=key)
    ctrnn = CTRNNGrid(key)
    noise = LangevinNoise()
    
    import lineax as lx
    def diffusion_wrapper(t, y, args):
        return lx.DiagonalLinearOperator(noise.diffusion(t, y, args))

    terms = diffrax.MultiTerm(diffrax.ODETerm(ctrnn.drift), diffrax.ControlTerm(diffusion_wrapper, bm))
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100))
    
    sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, saveat=saveat)
    return sol
