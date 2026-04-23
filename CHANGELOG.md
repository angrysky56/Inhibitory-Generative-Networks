# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-23

### Added
- **Phase 4: Evaluation and Metrics**
    - Implemented `signature_distance` metric using JAX MSE.
    - Created `AdditiveModel` benchmark baseline.
    - Added `compare_models` reporting utility.
    - Added integration tests for evaluation pipeline.
- **Phase 3: Layer 3 - The Sculptor**
    - Implemented Hadamard-based inhibitory masking.
    - Integrated 3-layer forward pass logic.
    - Added clipping and thresholding post-inhibition.
- **Phase 2: Layer 2 - Meta-Logical Grounding Engine**
    - Implemented CNN feature extractor for grounding.
    - Created meta-logical mask generation block.
    - Defined spatial mapping for grid-based inhibition.
- **Phase 1: Layer 1 - Stochastic Substrate**
    - Implemented continuous-time Langevin dynamics.
    - Created 2D CTRNN grid simulation.
    - Integrated Euler-Maruyama numerical integration.
- **Documentation**
    - Created root `README.md` and `ARCHITECTURE.md`.
    - Finalized `.planning/` artifacts for Milestone v1.0.

### Fixed
- Fixed JAX type compatibility issues in CTRNN state updates.
- Resolved dimension mismatch between CNN output and CTRNN grid.

### Changed
- Refactored model initialization to use `Equinox` modules throughout.
