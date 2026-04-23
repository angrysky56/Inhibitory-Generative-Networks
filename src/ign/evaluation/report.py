import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Optional, Any
from .metrics import signature_distance

def compare_models(
    ign_model: Any, 
    baseline_model: Any, 
    x: Array, 
    target: Array,
    y0: Optional[Array] = None,
    t0: float = 0.0,
    t1: float = 1.0,
    dt0: float = 0.05,
    key: Optional[jax.random.PRNGKey] = None
) -> Dict[str, Array]:
    """Compare IGN model against an additive baseline.
    
    Args:
        ign_model: The Inhibitory Generative Network model.
        baseline_model: The additive baseline model.
        x: Input image tensor.
        target: Target ground truth tensor.
        y0: Initial substrate state for IGN.
        t0: Start time for IGN simulation.
        t1: End time for IGN simulation.
        dt0: Step size for IGN simulation.
        key: PRNG key for IGN simulation.
        
    Returns:
        A dictionary with 'ign_distance' and 'baseline_distance'.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    if y0 is None:
        y0 = jnp.zeros(target.shape)
        
    # IGN Model call
    ign_output = ign_model(x, y0, t0, t1, dt0, key)
    
    # Baseline Model call
    baseline_output = baseline_model(x)
    
    # Calculate distances
    ign_dist = signature_distance(ign_output, target)
    baseline_dist = signature_distance(baseline_output, target)
    
    return {
        'ign_distance': ign_dist,
        'baseline_distance': baseline_dist
    }
