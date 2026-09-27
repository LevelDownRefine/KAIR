# MARBLE proximal splitting: fixed experiment protocol

## Material and theory gate

- ID: MARBLE-PROVABLE-SPLITTING-20260928; exploratory, not preregistered.
- User authorization: derive basically provable splits before running experiments.
- Base: `8c18acf`, branch `codex/marble-provable-splitting`, stacked on PR #7.
- Theory-only commit: `39a3cb1`, created before this implementation or any fit.
- Exact target, algorithm, assumptions and proof: [theory](MARBLE_SPLITTING_THEORY.md).
- Mathematical tests must pass before real fits. The gate includes independent
  projection and analytic minimizer checks, elimination, gradients and CUDA parity.

## Fixed model and objective

Freeze the existing MARBLE encoder and its BN statistics. Train a new explicit
residual layer `U = Q + W2 tanh(W1 Q + b1) + b2`, with widths d, 2d, d;
output `Z = U / sqrt(sum(U²) + .1²)`. All trainable matrices have spectral norm
at most 1; all bias vectors have Euclidean norm at most 1. Initialize W1 with
stacked ±I times .5/sqrt(2), W2 and both biases zero. Identical initialization
in every arm; save the untrained architecture control.

Optimize the objective defined in the theory with eta=1, beta=1e-4, P equal to
the number of latent parameters, and the same finite pair-loss in every arm.
Initialize V=Z. Deploy **Z**, not V; test points never enter the training prior graph.
This is a smooth network variant and finite-penalty Moreau graph prior, not an
equivalent reimplementation of the original MARBLE or hard-equality ADMM.

| Arm | Solver | lambda |
| --- | --- | --- |
| palm_no_prior | alternating PALM | 0 |
| joint_quadratic | synchronous PFB | 1 |
| palm_quadratic | alternating PALM | 1 |

Both lambda=1 arms solve the same objective from the same initialization. Their
comparison isolates the update ordering and associated step acceptance. The
lambda=0 arm is a matched learned no-graph-prior control. Lambda=0 still retains
the auxiliary coupling during iteration; eliminating V removes it from the target.

## Fixed solver budget and numerical checks

- 300 full-batch sweeps, final weights only. No Adam, CG, momentum or inner fitting.
- Parameter step starts at 16 each sweep. Halve at most 39 times (40 candidates).
- Majorization margin c=.1. PALM V step s=n/(eta+2lambda+1).
- Joint PFB scales both 16 and s by the same backtracking factor.
- Matrix projection clips every singular value at 1 using float64 SVD; bias
  projection uses the Euclidean ball. Apply Gaussian shrinkage before projection.
- Frozen encoder and 15NN graph construction in float32, then convert anchors
  and graph weights to float64 and recompute degree normalization. All learned
  operations, loss, gradients, updates and acceptance checks use float64.
- No positive tolerance in the majorization acceptance test. Failure raises an
  error and retains the failed run directory; do not retry or tune on behavior.
- Report every descent margin, even a tiny negative value. Stop on margin less
  than -1e-12, nonfinite values or constraint excess at least 1e-12. These are
  explicit floating-point diagnostics, not a modification of the exact theorem.
- Record every sweep, all backtracking counts, step sizes, displacements and
  feasibility. Record initial/final unit-step parameter proximal-gradient norm,
  raw auxiliary-gradient norm and auxiliary-force RMS `sqrt(n)*||grad_V H||`.
- Reaching 300 sweeps is not treated as convergence. No preselected tolerance
  or checkpoint selection based on these residuals; report their actual values.
- Compare both iteration count and measured time; equal sweeps need not cost
  equal work. The gate excludes training of the original frozen encoder.

## Data and evaluation

Reuse hashed Achilles train/test data and three 32D checkpoint seeds. Reuse all
four author rat checkpoints for separate 3D consistency fits (Achilles, Buddy,
Cicero, Gatsby). Graph: train-only neural position/velocity phase features,
15 nearest neighbors, symmetric Gaussian weights with degree normalization.
Pairs: four positive and four negative per node, fixed seeds 0/1/2. This sampling
is inherited from our exploration, not the original CkNN/random-walk sampler.

45 fits = three arms * three seeds * five tasks, plus 15 untrained controls.
Complete and hash all 60 embedding outputs before opening behavioral metrics.
No behavioral labels enter fitting, backtracking, constraints or checkpoint choice.
Data bundles contain labels but fitting code does not access them.

Use the unchanged cosine kNN position decoder k=36 and position-binned four-rat
OLS consistency, all 12 directed pairs. The latter is behavior-conditioned and
in-sample, not held-out cross-animal transfer. Three pair seeds are not three
independent author baseline fits. Report all per-seed metrics, mean and sample SD;
SD is not a confidence interval. The repeatedly observed holdout makes this an
exploration; no significance or generalization claim follows from these seeds.
Previous spectral and ADMM results are contextual: their models/targets differ.

## Execution and sealing

Use shared uv-managed official PyTorch 2.14.0+cu130 on the RTX 5070 Ti Laptop.
TF32 off, four CPU threads, CUDA for all fitting. Review a 45-minute runtime cap
without silently discarding failed runs or changing the fixed budget.

```powershell
../KAIR/.venv/Scripts/python.exe main_explore_marble_splitting.py fit --input ../KAIR/results/marble-input-20260928 --rat-input ../KAIR/results/rat-consistency-input-20260928-final --baseline ../KAIR/results/marble-kair-20260928-final --output results/splitting-20260928
../KAIR/.venv/Scripts/python.exe main_explore_marble_splitting.py evaluate --output results/splitting-20260928
```

Hash theory/protocol/source/dependencies/data/baselines before fitting; hash all
embeddings after fitting and verify all seals before evaluation. Save full model
parameters and auxiliary arrays locally in ignored results. Commit compact
complete metric and diagnostic reports, with no selected-seed filtering.
