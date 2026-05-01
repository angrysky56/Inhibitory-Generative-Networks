# IGN: Honest Assessment & Next Steps

_Written 2026-04-24 after a code read-through and test run. Meant as a companion to `docs/Inhibitory-Generative-Networks-Research.md`, which makes the theoretical case. This doc makes the **practical** case: what's actually been built, what the code demonstrates, and what a real validation of the paradigm would require._

## Current state

### What works today

- **Clean JAX/Equinox scaffold (~370 LOC).** Layers 1–3 + evaluation harness, all typed, all using idiomatic `diffrax` for SDE integration.
- **12/12 tests pass in 7.5s.** The forward pass runs, shape contracts hold, no NaNs.
- **The substrate produces bounded structured noise.** On a random seed, output has mean ≈ 0, std ≈ 0.17, in [-0.39, 0.41]. That's consistent with a Langevin-driven CTRNN — not yet verified to be 1/f, but in the right ballpark.
- **Inhibition mechanically works.** Mask values in [0.34, 0.65] produce output where ~11% of cells are near-zero. The Hadamard suppression does what the math says it should.

### What the repo does NOT yet show

The README and research doc make claims that the current code cannot support. This is not a criticism — it's normal for a scaffold — but it's worth naming clearly so the next iteration targets real gaps.

1. **No training.** `IGNModel` and `AdditiveModel` are both initialized with random weights and compared immediately. `compare_models` returns distances from *untrained* networks to a target. Which model "wins" on a given random seed is noise, not evidence.
2. **No measurement of the noise spectrum.** The README claims a "1/f noise baseline." The substrate produces *some* colored noise, but whether it actually follows a 1/f^α power law — and for what α — has not been tested.
3. **No diversity or coverage metric.** The sole evaluation metric (`signature_distance = MSE`) measures closeness to a single target. The paradigm's interesting claim is about *generative distribution shape* (can IGN produce a wider range of plausible outputs than an additive model?) — MSE cannot tell you that.
4. **The grounding is rank-limited.** Layer 2 is `CNN features → single Linear → sigmoid`. A single linear projection to produce the inhibition mask can't represent the kind of structured context-dependent suppression the "meta-logical" framing implies. Once training is happening, this will be the first thing to hit a ceiling.

## What a real validation of the paradigm would require

Three experiments, progressively more ambitious. Each answers a specific claim.

### Experiment A — Measure the noise (half-day) — **DONE 2026-04-24**

**Claim under test:** "The substrate generates a 1/f noise baseline."

**Method:** Ran `diffeqsolve` directly (bypassing `simulate_substrate`'s hardcoded 100-point SaveAt) to get 2000–5000 sampled points across T=20–50. For each spatial cell, took the time series, computed PSD via rFFT, averaged across cells, fit log(power) vs log(freq) on the inertial range [0.5 Hz, fs/4].

**Result:** **α = 2.0 ± 0.05, robust across configurations.** Tested grid sizes 8/16/32, simulation lengths T=20 and T=50, integration steps dt=0.001 and dt=0.01 — every run gave α between 1.94 and 2.05.

| config | α |
|---|---|
| 16×16, T=20, dt=0.01 | 2.033 |
| 16×16, T=50, dt=0.01 | 2.046 |
| 16×16, T=20, dt=0.001 | 1.937 |
| 32×32, T=20, dt=0.01 | 2.039 |
| 8×8, T=20, dt=0.01 | 2.040 |

**Interpretation:** **This is Brownian (random-walk) noise, not pink noise.** α≈2 means power scales as 1/f², which is the spectrum you get from integrating white noise — exactly what diffusion-dominated SDE dynamics produce. The CTRNN's contractive linear drift (`-γ·y`) is too weak relative to the diffusion to impose long-range temporal correlations. The 1/f claim in the README is **not currently true**.

**What to do about it:** This is fixable but it's an architectural decision, not a tuning knob. Options:
1. **Stronger nonlinear coupling.** Increase the stencil weights or replace `tanh(coupling)` with a sharper saturating nonlinearity to push the system toward criticality (real 1/f noise emerges at the edge of dynamical instability — see self-organized criticality literature).
2. **Multi-timescale dynamics.** Add a second-order term or hierarchy of CTRNN layers with different γ values. Pink noise often appears as a superposition of OU processes with a power-law distribution of timescales.
3. **Accept α=2 and update the docs.** Brownian-baseline subtractive generation is still a coherent paradigm; it just isn't 1/f. The biological-plausibility claim weakens but doesn't disappear (cortical activity has been characterized as both, depending on what's being measured).

**Recommendation:** Option 1 is the smallest experiment. Try `gamma=0.1` (currently 1.0 — much weaker linear contraction → longer effective memory) and re-measure. If α drops to ~1.0–1.5, the architecture is salvageable as-is. If it stays at 2, options 2 or 3.

### Experiment B — Train both models on the same toy task (weekend)

**Claim under test:** "Subtractive sculpting is a competitive generative paradigm."

**Method:** Pick MNIST (or a subset of it). For each image, set it as the target. Train:
- `IGNModel`: learn CNN + grounding Linear + maybe CTRNN stencil to minimize `signature_distance(sculpted_output, target)`.
- `AdditiveModel`: same loss, same data, same number of parameters (critical — no free-parameter advantage for either side).

Use `optax.adam`, 5k–10k steps, record training curves. Compare final validation loss.

**Threshold for success:** IGN within ~2× the validation loss of the additive baseline, *with stable training*. If it converges at all at reasonable loss, the paradigm is viable. "Beating" additive on this simple task isn't the point — the point is "not catastrophically worse."

**What you'll learn that you don't know yet:**
- Whether the loss landscape of subtractive models is even trainable with standard gradient descent. (Inhibitory mechanisms can be harder to optimize because the gradient flows through a multiply-by-mask rather than an add.)
- Whether the grounding-Linear rank limitation matters in practice.
- How sensitive IGN is to the substrate-simulation hyperparameters (t1, dt0, diffusion coefficient).

### Experiment C — Stress the paradigm (1–2 weeks)

**Claim under test:** "Subtractive generation has advantages additive models don't."

**Method:** Pick a task where *multiple valid outputs exist given an input*. Good candidates:
- Partial MNIST occlusion: input is a digit with 40% of pixels masked out, target is *any* plausible full digit. (Ambiguous — a mostly-obscured "0" could also be an "8".)
- Texture synthesis: input is a small patch, target is a larger texture consistent with it. Many valid continuations.

Train both models. Then at inference, run each model N=100 times on the same input with different random seeds / substrate initializations. Measure:
- **Coverage:** how many distinct valid outputs does each model produce?
- **Diversity:** pairwise distance between samples from the same input.
- **Sample quality:** how often are the outputs actually valid? (Classifier accuracy on a held-out digit classifier, FID, etc.)

**Threshold for success:** IGN should show *higher sample diversity at comparable quality* than the additive baseline. This is where the biology pays off if it pays off at all — intrinsic noise means IGN's output distribution is naturally stochastic, whereas an additive feedforward network trained on MSE collapses to the mean.

**If this doesn't work,** the paradigm's claim to advantage hasn't been demonstrated and is probably wrong in this form. If it does work, you have a publishable result.

## What to skip, for now

- **Phase 5: Neuromorphic translation.** Mapping to Loihi 2 via Lava before you've shown the paradigm works on a GPU is premature. The neuromorphic claim ("natural fit for async hardware") is a *consequence* of the architecture being correct; it doesn't make the architecture correct. Defer until Experiment B is green.
- **Larger grounding networks.** Don't expand Layer 2 from the single Linear until Experiment B reveals that the single Linear is actually the bottleneck.
- **Additional modalities.** MNIST is enough to make or break the core claim. Audio, video, 3D — all premature.

## On suggestions inspired by the septo-entorhinal paper

The Kim et al. 2026 paper (Nature Neuroscience, doi:10.1038/s41593-026-02280-6) was floated as motivation for several architectural changes — active temporal inhibition, recursive grounding, bipartite excitatory/inhibitory streams, an "inhibitory efficiency" metric.

**Important framing first:** the paper is about which memory mice retrieve after a memory update — the medial septum GABAergic projection to medial entorhinal cortex is required for retrieving the updated memory and not the original. **It is not a paper about generative imagery, subtractive sculpting, or 1/f cortical dynamics.** Citing it as IGN motivation would be a stretch readers will catch. Don't.

That said, evaluated on their own engineering merits — separate from whether the paper supports them — two of the four suggestions are genuinely good:

**Worth doing: active temporal inhibition.** Currently the mask is computed once from input features, and applied only at the very end (`sculpted = ctrnn_state * mask`). Injecting the mask into `CTRNNGrid.drift` so it gates the coupling/relaxation as the system evolves is architecturally sounder — it stops the substrate from wasting compute simulating states that will be suppressed anyway, and it lets the mask shape temporal correlations rather than just spatial structure. This is a stronger move than it sounds: it may also affect the α=2 finding from Experiment A, since static masking can't impose long-range temporal correlations on the substrate but active gating can.

**Worth doing: recursive grounding.** Right now `MetaLogicalGrounding` only sees the initial CNN features and produces a single mask. Updating the mask in the integration loop based on the substrate's current state (read it back at intervals, re-run grounding, update mask) closes a loop that's currently open. This is a standard iterative-refinement pattern in generative models; it's not specifically biological.

**Skip for now: bipartite excitatory/inhibitory streams.** Adding a separate inhibitory stencil and gating the balance between two streams is more architecture for diminishing returns. If active temporal inhibition (above) gates coupling strength in the existing single stencil, you get most of the same effect with half the complexity. Revisit only if active inhibition turns out to be insufficient.

**Fold in, don't add separately: inhibitory efficiency metric.** Measuring how cleanly IGN suppresses an injected distractor signal is a useful measurement, but it's a special case of the diversity / coverage metrics already proposed in Experiment C. Add the distractor-suppression test as a sub-experiment in C rather than as a new evaluation track.

**Sequencing:** active temporal inhibition is small and cheap (one-line change to `drift`). Try it before recursive grounding. If α drops toward 1/f as a side effect, that's a real finding — and it would mean the original "1/f baseline" claim is achievable through *active* gating, even though it isn't through the current static-mask architecture.

## Small code fixes worth making in the meantime

1. **README example is broken.** The code example `model = IGN(...)` then `model(input_signal, key)` won't run — the real class is `IGNModel` and takes `(x, y0, t0, t1, dt0, key)`. Fix the example to match `tests/layer3/test_integration.py` which has the correct call signature.
2. **CNN output size is hardcoded as 800 in two places.** `model.py` and `evaluation/baseline.py` both compute this from assumed 1×28×28 input. Extract a helper (e.g., `cnn_output_size(input_shape) -> int`) or accept it as an init parameter. Any change to input size will silently break both classes.
3. **`signature_distance = MSE` is a placeholder.** Rename it to `mse_distance` until you have an actual signature-based metric, or add a note in the docstring that the name is aspirational. "Signature distance" suggests something like sliced Wasserstein or a neural feature distance; MSE is neither.
4. **No `pyproject.toml`.** The project has a `requirements.txt` but no install config. A minimal `pyproject.toml` with `src/` layout declared would let `pip install -e .` work and eliminate the need for `PYTHONPATH=src` in every test command.
5. **Add `scripts/` directory.** Put Experiment A (the noise-spectrum plot) in `scripts/measure_noise_spectrum.py` as the first concrete use of the infrastructure.

## Closing thought

The idea is coherent. The code is cleaner than most speculative-research repos at this stage. What's missing isn't architectural insight — it's empirical evidence. Experiment A is now done and gave a real, replicable, architecturally-meaningful result (α=2, not 1/f). Experiment B (training, weekend) is the next gate, and the active-temporal-inhibition change above is a small unblocker that may help on its way.

"Untested besides basic and maybe untestable" was too modest. It is testable. It just hasn't been tested *in the ways that would settle the question* — and now we know exactly which ways those are.
