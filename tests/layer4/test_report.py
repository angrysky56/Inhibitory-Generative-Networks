import jax
import jax.numpy as jnp
from ign.model import IGNModel
from ign.evaluation import AdditiveModel, compare_models

def test_compare_models():
    key = jax.random.PRNGKey(0)
    target_shape = (5, 5)
    
    ign_model = IGNModel(key=key, target_shape=target_shape)
    baseline_model = AdditiveModel(key=key, target_shape=target_shape)
    
    x = jnp.zeros((1, 28, 28))
    target = jnp.ones(target_shape)
    
    results = compare_models(ign_model, baseline_model, x, target)
    
    assert 'ign_distance' in results
    assert 'baseline_distance' in results
    assert results['ign_distance'] >= 0
    assert results['baseline_distance'] >= 0
