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

### Experiment A — Measure the noise (half-day)

**Claim under test:** "The substrate generates a 1/f noise baseline."

**Method:** Run `simulate_substrate` for a long trajectory (say T = 100, 10k timesteps). For each spatial cell, take the time series and compute the power spectral density (FFT). Average PSDs across cells. Plot log(power) vs log(frequency). A 1/f^α signal shows up as a straight line with slope −α.

**Threshold for success:** α between 0.5 and 1.5 across a decade of frequencies. If α ≈ 0, it's white noise; if α ≈ 2, it's Brownian. 1/f is the specific claim.

**Why do this first:** It's the cheapest check and it validates (or refutes) a core mechanistic claim. If the substrate isn't producing the noise profile the theory assumes, that's an architectural finding — tune the CTRNN stencil, the coupling strength, or the Langevin diffusion coefficient until it does.

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

## Small code fixes worth making in the meantime

These aren't blockers but will bite if left alone.

1. **README example is broken.** The code example `model = IGN(...)` then `model(input_signal, key)` won't run — the real class is `IGNModel` and takes `(x, y0, t0, t1, dt0, key)`. Fix the example to match `tests/layer3/test_integration.py` which has the correct call signature.
2. **CNN output size is hardcoded as 800 in two places.** `model.py` and `evaluation/baseline.py` both compute this from assumed 1×28×28 input. Extract a helper (e.g., `cnn_output_size(input_shape) -> int`) or accept it as an init parameter. Any change to input size will silently break both classes.
3. **`signature_distance = MSE` is a placeholder.** Rename it to `mse_distance` until you have an actual signature-based metric, or add a note in the docstring that the name is aspirational. "Signature distance" suggests something like sliced Wasserstein or a neural feature distance; MSE is neither.
4. **No `pyproject.toml`.** The project has a `requirements.txt` but no install config. A minimal `pyproject.toml` with `src/` layout declared would let `pip install -e .` work and eliminate the need for `PYTHONPATH=src` in every test command.
5. **Add `scripts/` directory.** Put Experiment A (the noise-spectrum plot) in `scripts/measure_noise_spectrum.py` as the first concrete use of the infrastructure.

## Closing thought

The idea is coherent. The code is cleaner than most speculative-research repos at this stage. What's missing isn't architectural insight — it's empirical evidence. The three experiments above are the minimum path to knowing whether the paradigm is real, and the first one is cheap enough to run this week.

"Untested besides basic and maybe untestable" was too modest. It is testable. It just hasn't been tested *in the ways that would settle the question*.
