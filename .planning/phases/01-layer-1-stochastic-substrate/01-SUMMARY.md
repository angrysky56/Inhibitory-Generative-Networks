# Plan 01 Summary

## Overview
Built a stable, differentiable simulation of continuous neural noise using a Langevin CTRNN grid in JAX and Diffrax. The system correctly integrates the SDE using Euler-Maruyama over time without producing NaNs.

## Completed Tasks
- [x] 1-01-01: Initialize Project Structure and Dependencies
- [x] 1-01-02: Implement Langevin Drift and Diffusion Functions
- [x] 1-01-03: Implement CTRNN Grid Dynamics
- [x] 1-01-04: Implement SDE Simulation Integration (Euler-Maruyama)

## Files Created/Modified
- `requirements.txt`
- `src/ign/__init__.py`
- `src/ign/substrate/__init__.py`
- `src/ign/substrate/langevin.py`
- `src/ign/substrate/ctrnn.py`
- `src/ign/substrate/simulation.py`
- `tests/layer1/test_langevin.py`
- `tests/layer1/test_ctrnn.py`
- `tests/layer1/test_integration.py`

## Notes
- Used `jnp.pad` with `mode='wrap'` instead of passing `boundary='wrap'` directly to `convolve2d` because JAX only supports `fill` mode.
- Wrapped the diffusion term with `lineax.DiagonalLinearOperator` to address Diffrax 0.7.0 `ControlTerm` compatibility with arrays.
