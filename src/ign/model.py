import jax
import jax.numpy as jnp
import equinox as eqx
from .grounding.cnn import CNNFeatureExtractor
from .grounding.mask import MetaLogicalGrounding
from .substrate.simulation import simulate_substrate
from .sculptor.sculptor import Sculptor

class IGNModel(eqx.Module):
    cnn: CNNFeatureExtractor
    grounding: MetaLogicalGrounding
    sculptor: Sculptor

    def __init__(self, key, target_shape):
        keys = jax.random.split(key, 3)
        self.cnn = CNNFeatureExtractor(key=keys[0])
        # Assuming flattened CNN output feeds into Grounding
        # For a simple test, we will hardcode the CNN output size based on 1x28x28 input.
        # Conv2d(1, 16, 3) -> 16x26x26
        # MaxPool2d(2, 2) -> 16x13x13
        # Conv2d(16, 32, 3) -> 32x11x11
        # MaxPool2d(2, 2) -> 32x5x5
        # Total features = 32 * 5 * 5 = 800
        # The prompt used 1568 but let's calculate it accurately or just use 800.
        # Actually in 03-PLAN.md it uses 1568. Let's just use 800 for the actual logic based on cnn.py:
        # padding is default 0 in eqx.nn.Conv2d.
        # input: (1, 28, 28)
        # Conv1: (16, 26, 26)
        # MaxPool1: (16, 13, 13)
        # Conv2: (32, 11, 11)
        # MaxPool2: (32, 5, 5) -> 32 * 25 = 800
        self.grounding = MetaLogicalGrounding(in_features=800, target_shape=target_shape, key=keys[1])
        self.sculptor = Sculptor()

    def __call__(self, x, y0, t0, t1, dt0, key):
        features = self.cnn(x)
        mask = self.grounding(features)
        
        # We need the key to pass to simulate_substrate
        # y0 is typically of target_shape
        sol = simulate_substrate(key, y0, t0, t1, dt0)
        final_state = sol.ys[-1]
        
        sculpted = self.sculptor(final_state, mask)
        return sculpted
