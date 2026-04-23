# Requirements

## v1 Requirements

### Substrate
- [x] **SUB-01**: Implement continuous-time Langevin dynamics module in JAX.
- [x] **SUB-02**: Implement 2D grid Continuous-Time Recurrent Neural Network (CTRNN) with local connectivity.
- [x] **SUB-03**: Integrate Langevin module to provide noise baselines to the CTRNN.
- [x] **SUB-04**: Implement Euler-Maruyama numerical integration for asynchronous state simulation.

### Grounding
- [x] **GND-01**: Implement Convolutional Neural Network (CNN) feature extractor.
- [x] **GND-02**: Implement Meta-Logical Grounding block that translates features into inhibitory masking tensors.
- [x] **GND-03**: Define spatial mapping from grounding tensor dimensions to CTRNN grid dimensions.

### Sculpting
- [x] **SCU-01**: Implement Hadamard product application between Layer 2 masks and Layer 1 outputs.
- [x] **SCU-02**: Implement clipping/thresholding to maintain state boundaries post-inhibition.
- [x] **SCU-03**: Integrate 3-layer forward pass mapping input target to sculpted output grid.

### Evaluation
- [x] **EVAL-01**: Implement metric for signature-matching (IGN vs target distribution).
- [x] **EVAL-02**: Implement comparative benchmark against a baseline additive generative model.

## v2 Requirements (Deferred)
- Dynamic Routing (Allowing masks to route information rather than just inhibit).
- Loihi 2 Neuromorphic Compilation mapping.

## Out of Scope
- Full Neuromorphic Hardware Deployment — Currently targeting JAX simulation.
- Generative Adversarial Training — Sticking to supervised/unsupervised energy-based approaches.

## Traceability
*To be updated by roadmap*
