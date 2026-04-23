# Inhibitory Generative Networks (IGN)

[![Project Status: Milestone 1 Complete](https://img.shields.io/badge/Status-Milestone%201%20Complete-green.svg)](#roadmap)

Inhibitory Generative Networks (IGN) represent a paradigm shift in generative AI, moving away from traditional additive excitation (building a signal from zero) toward **biologically-inspired subtractive sculpting**. 

Instead of generating a future state, the IGN maintains a continuous stochastic baseline—a "soup" of all possible states—and applies top-down inhibitory masks to suppress logically contradictory features. The resulting unmasked activity is the "imagined" or generated state.

## 🧠 Core Architecture

The IGN is structured into three co-dependent layers:

1.  **Layer 1: The Stochastic Substrate**
    *   Maintains a continuous, dynamic state of stochastic fluctuation using **Langevin dynamics** on a **CTRNN (Continuous-Time Recurrent Neural Network)** grid.
    *   Generates a 1/f noise baseline that serves as the "raw material" for generation.

2.  **Layer 2: Meta-Logical Grounding Engine**
    *   Processes external context or previous states through a lightweight **CNN feature extractor**.
    *   Outputs an **inhibitory mask** (Hadamard tensor) that defines which parts of the substrate to suppress based on physical or logical constraints.

3.  **Layer 3: The Sculptor**
    *   Executes the final generation by applying the inhibitory mask to the stochastic substrate via an element-wise **Hadamard product**.
    *   This layer is mathematically efficient ($O(N)$ complexity) and maps naturally to asynchronous neuromorphic hardware.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
*   JAX, Equinox, Optax

### Installation

```bash
# Clone the repository
git clone https://github.com/ty/Inhibitory-Generative-Networks.git
cd Inhibitory-Generative-Networks

# Create and sync the virtual environment
uv venv
source .venv/bin/activate
uv pip sync
```

### Running Simulations

You can run a basic simulation of the stochastic substrate or execute the full 3-layer forward pass:

```python
import jax
import jax.numpy as jnp
from ign.model import IGN

# Initialize the model
key = jax.random.PRNGKey(42)
model = IGN(grid_size=64, key=key)

# Simulated input signal (e.g., target feature to ground)
input_signal = jnp.zeros((1, 64, 64)) 

# Generate output via sculpting
state, key = model(input_signal, key)
```

## 📊 Evaluation & Metrics

The project includes an evaluation module to benchmark IGN against traditional additive baselines.

*   **Signature Matching**: Quantifies the distance between sculpted output and target representations using `signature_distance`.
*   **Additive Baseline**: A standard CNN-to-Linear generative model used for comparative performance analysis.

Run the comparison report:
```python
from ign.evaluation.report import compare_models
# See tests/layer4/test_report.py for usage details
```

## 🗺 Roadmap

- [x] **Phase 1: Stochastic Substrate** - Langevin dynamics and CTRNN grid implementation.
- [x] **Phase 2: Meta-Logical Grounding** - CNN feature extraction and mask generation.
- [x] **Phase 3: The Sculptor** - Inhibitory masking and integration.
- [x] **Phase 4: Evaluation** - Metrics and comparative benchmarking.
- [ ] **Phase 5: Neuromorphic Translation** - (Planned) Mapping to Intel Loihi 2 via Lava.

## 📖 Theoretical Background

The IGN framework is based on the **Spontaneous Activity Reshaping Hypothesis** observed in the mammalian visual cortex. Feedback connections in the brain are primarily modulatory and inhibitory rather than driving. Mental imagery is effectively "carved" out of intrinsic noise rather than synthesized from a silent baseline.

For a deep dive into the mathematics and biological foundations, see the [Research Paper](docs/Inhibitory-Generative-Networks-Research.md).

---
*Developed as a proof-of-concept for asynchronous neuromorphic intelligence.*
