# Phase 4 Summary: Evaluation and Metrics

## Objective
Measure the efficacy of the IGN by implementing a signature-matching metric and comparing the IGN against an additive baseline model.

## Key Files Created/Modified
- `src/ign/evaluation/metrics.py`: Implementation of `signature_distance`.
- `src/ign/evaluation/baseline.py`: Implementation of `AdditiveModel` baseline.
- `src/ign/evaluation/report.py`: Implementation of `compare_models` reporting.
- `src/ign/evaluation/__init__.py`: Exports for the evaluation module.
- `tests/layer4/test_metrics.py`: Tests for metrics.
- `tests/layer4/test_baseline.py`: Tests for baseline model.
- `tests/layer4/test_report.py`: Integration tests for reporting.

## Self-Check: PASSED
- [x] `signature_distance` correctly calculates MSE between tensors.
- [x] `AdditiveModel` baseline is implemented using standard Equinox layers and provides a comparative benchmark.
- [x] `compare_models` successfully runs both models and reports distances.
- [x] All unit and integration tests for Phase 4 pass.

## Verification Results
- `uv run pytest tests/layer4/` passed with 3 tests.
- Structural integrity of `src/ign/evaluation/` verified via exports.

## Decisions/Trade-offs
- Used Mean Squared Error (MSE) as the initial implementation for `signature_distance` for simplicity and direct comparison.
- Added ODE parameters to `compare_models` to support direct testing of `IGNModel` alongside the baseline.
