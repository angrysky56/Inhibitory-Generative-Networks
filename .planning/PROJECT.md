# Inhibitory Generative Networks (IGN)

## What This Is

A 3-Layer Proof of Concept using JAX to model Inhibitory Generative Networks (IGN). This project explores a novel paradigm using subtractive masking on stochastic noise baselines (Langevin dynamics and CTRNNs) rather than traditional additive generative models, with the ultimate goal of asynchronous neuromorphic deployment.

## Core Value

Proving that target signal features can be reliably carved out of spontaneous background noise using inhibitory Hadamard masking instead of additive excitation.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Implement Layer 1: The Stochastic Substrate (Langevin CTRNN grid in JAX generating 1/f noise).
- [ ] Implement Layer 2: The Meta-Logical Grounding Engine (CNN processing external signals to generate masking tensors).
- [ ] Implement Layer 3: The Sculptor (Applying Hadamard product to filter the CTRNN grid output into target representations).
- [ ] Define measurable metrics for evaluating the IGN's performance versus additive baselines.
- [ ] Prepare codebase for neuromorphic translation (state-based, asynchronous-friendly).

### Out of Scope

- Full Neuromorphic Hardware Deployment — Currently targeting JAX simulation; actual Loihi 2 deployment is reserved for future phases.
- Additive Generator Integration — This project is strictly evaluating inhibitory mechanics; no standard generative architectures will be mixed in.
- Complex multimodal data handling initially — Start with simple proxy tasks (e.g., MNIST or basic waveforms).

## Context

- **Theoretical Basis:** Biological brains exhibit spontaneous, continuous activity. Task-specific patterns are carved out by inhibition.
- **Tech Stack:** Python, JAX, Equinox, Optax.
- **Deployment Strategy:** Local development targeting asynchronous neuromorphic patterns.
- **Documentation:** `docs/Inhibitory-Generative-Networks-Research.md` provides the theoretical and architectural blueprint.

## Constraints

- **Tech Stack**: JAX and Equinox — Must be differentiable and leverage XLA compilation for performance.
- **Architecture**: Must preserve asynchronous, continuous-time dynamics simulation (using explicit numerical integration like Euler-Maruyama).
- **Performance**: The CTRNN simulation must run efficiently; vectorization via `jax.vmap` and `jax.lax.scan` is critical.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| JAX + Equinox | Provides necessary auto-diff, XLA speed, and clean neural network abstractions for the CTRNN and CNN layers. | — Pending |
| 3-Layer Architecture | Isolates the noise generation, logic grounding, and masking operations cleanly. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-22 after project initialization*
