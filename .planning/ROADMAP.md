# Roadmap

## Phase 1: Layer 1 - Stochastic Substrate
**Goal**: Build a stable, differentiable simulation of continuous neural noise.
- [x] SUB-01: Implement continuous-time Langevin dynamics module in JAX.
- [x] SUB-02: Implement 2D grid Continuous-Time Recurrent Neural Network (CTRNN).
- [x] SUB-03: Integrate Langevin module to provide noise baselines to the CTRNN.
- [x] SUB-04: Implement Euler-Maruyama numerical integration.

**Success Criteria**: 
- A test script runs a 100-timestep simulation of the grid.
- Output visually resembles 1/f noise without overflowing.

## Phase 2: Layer 2 - Meta-Logical Grounding Engine
**Goal**: Build the CNN and mask generation logic.
- [ ] GND-01: Implement Convolutional Neural Network (CNN) feature extractor.
- [ ] GND-02: Implement Meta-Logical Grounding block.
- [ ] GND-03: Define spatial mapping from grounding tensor dimensions to CTRNN grid dimensions.

**Success Criteria**:
- CNN processes a dummy MNIST image into a mask tensor matching the CTRNN dimensions.

## Phase 3: Layer 3 - The Sculptor
**Goal**: Connect the layers via inhibitory masking.
- [ ] SCU-01: Implement Hadamard product application.
- [ ] SCU-02: Implement clipping/thresholding.
- [ ] SCU-03: Integrate 3-layer forward pass.

**Success Criteria**:
- A full forward pass runs from a source signal to a sculpted output state.

## Phase 4: Evaluation and Metrics
**Goal**: Measure the efficacy of the IGN.
- [ ] EVAL-01: Implement signature-matching metric.
- [ ] EVAL-02: Implement comparative benchmark against an additive baseline.

**Success Criteria**:
- Metrics report accurately calculates the distance between sculpted output and target representations.
