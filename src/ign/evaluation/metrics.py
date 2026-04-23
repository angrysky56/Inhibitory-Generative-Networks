import jax.numpy as jnp
from jax import Array

def signature_distance(y_pred: Array, y_true: Array) -> Array:
    """Compute the signature distance between predicted and true tensors.
    
    Currently implemented as Mean Squared Error (MSE).
    
    Args:
        y_pred: The predicted output tensor from the model.
        y_true: The target ground truth tensor.
        
    Returns:
        A scalar array representing the mean squared error distance.
    """
    return jnp.mean((y_pred - y_true)**2)
