import jax.numpy as jnp

def signature_distance(y_pred, y_true):
    """Compute the signature distance between predicted and true tensors.
    
    Currently implemented as Mean Squared Error (MSE).
    """
    return jnp.mean((y_pred - y_true)**2)
