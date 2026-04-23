---
wave: 1
depends_on: []
files_modified:
  - tests/layer4/test_metrics.py
  - src/ign/evaluation/__init__.py
  - src/ign/evaluation/metrics.py
  - tests/layer4/test_baseline.py
  - src/ign/evaluation/baseline.py
  - src/ign/evaluation/report.py
autonomous: true
---

# Phase 4: Evaluation and Metrics

## Goal
Measure the efficacy of the IGN by implementing a signature-matching metric and comparing the IGN against an additive baseline model.

## Tasks

```xml
<task>
  <id>eval-01-red</id>
  <type>tdd</type>
  <title>RED: Signature Matching Metric Tests</title>
  <description>Write failing tests for the signature matching metric (EVAL-01). The metric should compute the distance between sculpted output and target representations (e.g., Mean Squared Error and/or Maximum Mean Discrepancy).</description>
  <requirements>EVAL-01</requirements>
  <read_first>
    - tests/layer4/test_metrics.py
  </read_first>
  <action>
    Create `tests/layer4/test_metrics.py` (and add `__init__.py` to `tests/layer4`). Write `test_signature_distance` that expects `signature_distance(output, target)` to return 0.0 for identical tensors and a positive value for different tensors.
  </action>
  <acceptance_criteria>
    - `tests/layer4/test_metrics.py` contains `def test_signature_distance`
    - Running `uv run pytest tests/layer4/test_metrics.py` fails with an ImportError because `signature_distance` is not yet implemented.
  </acceptance_criteria>
</task>

<task>
  <id>eval-01-green</id>
  <type>tdd</type>
  <title>GREEN: Signature Matching Metric Implementation</title>
  <description>Implement the signature matching metric to make the tests pass.</description>
  <requirements>EVAL-01</requirements>
  <read_first>
    - tests/layer4/test_metrics.py
    - src/ign/evaluation/metrics.py
  </read_first>
  <action>
    Create `src/ign/evaluation/metrics.py` and implement `def signature_distance(y_pred, y_true)` using `jax.numpy.mean((y_pred - y_true)**2)`. Export it in `src/ign/evaluation/__init__.py`.
  </action>
  <acceptance_criteria>
    - `src/ign/evaluation/metrics.py` contains `def signature_distance`
    - Running `PYTHONPATH=src uv run pytest tests/layer4/test_metrics.py` passes (exit code 0).
  </acceptance_criteria>
</task>

<task>
  <id>eval-01-refactor</id>
  <type>tdd</type>
  <title>REFACTOR: Signature Matching Metric</title>
  <description>Refactor the signature matching metric code for better structure or type hints if needed.</description>
  <requirements>EVAL-01</requirements>
  <read_first>
    - src/ign/evaluation/metrics.py
  </read_first>
  <action>
    Add proper JAX array typing to `signature_distance` and ensure it uses idiomatic JAX. Ensure it is well-documented.
  </action>
  <acceptance_criteria>
    - `src/ign/evaluation/metrics.py` contains type hints (e.g. `import jax.numpy as jnp` and `from jax import Array`).
    - Running `PYTHONPATH=src uv run pytest tests/layer4/test_metrics.py` still passes.
  </acceptance_criteria>
</task>

<task>
  <id>eval-02-red</id>
  <type>tdd</type>
  <title>RED: Additive Baseline Tests</title>
  <description>Write failing tests for a comparative additive baseline model (EVAL-02).</description>
  <requirements>EVAL-02</requirements>
  <read_first>
    - tests/layer4/test_baseline.py
    - tests/layer3/test_integration.py
  </read_first>
  <action>
    Create `tests/layer4/test_baseline.py`. Write `test_additive_baseline_forward` that initializes an `AdditiveModel` and verifies it produces an output of `target_shape`.
  </action>
  <acceptance_criteria>
    - `tests/layer4/test_baseline.py` contains `def test_additive_baseline_forward`
    - Running `PYTHONPATH=src uv run pytest tests/layer4/test_baseline.py` fails because `AdditiveModel` is not yet implemented.
  </acceptance_criteria>
</task>

<task>
  <id>eval-02-green</id>
  <type>tdd</type>
  <title>GREEN: Additive Baseline Implementation</title>
  <description>Implement the additive baseline generative model.</description>
  <requirements>EVAL-02</requirements>
  <read_first>
    - tests/layer4/test_baseline.py
    - src/ign/evaluation/baseline.py
    - src/ign/model.py
  </read_first>
  <action>
    Create `src/ign/evaluation/baseline.py`. Implement `AdditiveModel` using `equinox.Module`. It should use standard additive layers (e.g., `eqx.nn.Linear` or simple addition of CNN features) to map an input image to the target shape, rather than the inhibitory masking approach of `IGNModel`. Ensure it has a `__call__` method returning the output.
  </action>
  <acceptance_criteria>
    - `src/ign/evaluation/baseline.py` contains `class AdditiveModel`
    - Running `PYTHONPATH=src uv run pytest tests/layer4/test_baseline.py` passes (exit code 0).
  </acceptance_criteria>
</task>

<task>
  <id>eval-02-refactor</id>
  <type>tdd</type>
  <title>REFACTOR: Additive Baseline Cleanup</title>
  <description>Refactor the baseline model to ensure it is clean and documented.</description>
  <requirements>EVAL-02</requirements>
  <read_first>
    - src/ign/evaluation/baseline.py
  </read_first>
  <action>
    Add proper type hints to `AdditiveModel` and standard docstrings. Ensure `src/ign/evaluation/__init__.py` exports `AdditiveModel`.
  </action>
  <acceptance_criteria>
    - `src/ign/evaluation/baseline.py` contains type hints.
    - Running `PYTHONPATH=src uv run pytest tests/layer4/test_baseline.py` still passes.
  </acceptance_criteria>
</task>

<task>
  <id>eval-report</id>
  <type>execute</type>
  <title>Evaluation Report Integration</title>
  <description>Create a high-level evaluation function or script that calculates the signature distance for both IGN and the baseline models.</description>
  <requirements>EVAL-01, EVAL-02</requirements>
  <read_first>
    - src/ign/evaluation/report.py
  </read_first>
  <action>
    Create `src/ign/evaluation/report.py`. Implement a function `compare_models(ign_model, baseline_model, x, target)` that returns a dictionary mapping `'ign_distance'` and `'baseline_distance'` to their respective `signature_distance` values. Export it in `__init__.py`.
  </action>
  <acceptance_criteria>
    - `src/ign/evaluation/report.py` contains `def compare_models`
    - `src/ign/evaluation/__init__.py` contains `from .report import compare_models`
  </acceptance_criteria>
</task>
```

## Verification
- `uv run pytest tests/layer4/test_metrics.py` passes.
- `uv run pytest tests/layer4/test_baseline.py` passes.
- The evaluation module exports `signature_distance`, `AdditiveModel`, and `compare_models`.

## must_haves
- Metrics report accurately calculates the distance between sculpted output and target representations.
- Comparative benchmark exists to test an additive baseline approach vs IGN.
