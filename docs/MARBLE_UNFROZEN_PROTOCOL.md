# Full-encoder PALM: fixed experiment protocol

## Material Passport

- ID: MARBLE-UNFROZEN-PALM-20260928; exploratory, not preregistered.
- User request: extend the preceding provable-splitting experiment without freezing.
- Branch: `codex/marble-unfrozen-palm`, based on `2c8e587` / PR #8.
- Theory gate: `1583115`, committed before implementation and any new fit.
- [Theory and exact scope](MARBLE_UNFROZEN_THEORY.md).

## Fixed architecture and initialization

Fold the author's eval-mode BN into the first affine layer, including all affine
BN parameters. Remove BN state and inactive diffusion_time. Preserve its initial
function exactly before applying the stated smooth activation changes. All four
effective encoder tensors (first weight/bias, second weight/bias) train together
with the four residual-layer tensors in full arms. This is not train-mode BN.

Compute feature mean and population SD on training features only; clamp SD to
1e-6. Fold this input standardization into the initial first affine parameters to
preserve the initial function. The stored mean/scale are immutable input buffers,
not frozen model weights. Verify folding against the original double eval model
at rtol=1e-9, atol=1e-10, and verify original CUDA float32 outputs against the saved
reference as before. No behavioral labels enter these operations.

Replace ReLU with `(u+sqrt(u²+.05²))/2`. Use soft normalization epsilon=.1 both
at the encoder output and the residual-layer output. Append the same d,2d,d tanh
residual layer as before, with stacked ±I*.5/sqrt(2), zero second weight/biases.
No dropout or moving statistics. Per encoder-tensor ball radii are fixed at
`max(1,1.25*norm(initial))`, spectral for matrices and Euclidean for biases.
Residual-layer radii are 1. There is no behavior-selected radius or parameter grid.

## Matched arms and objective

| Arm | Encoder weights | Graph lambda |
| --- | --- | --- |
| frozen_palm | fixed at the modified initial encoder | 1 |
| full_no_prior | all four effective tensors updated | 0 |
| full_palm | all four effective tensors updated | 1 |

All arms train the residual layer and V. Save untrained modified architecture as
an additional control and evaluate the original MARBLE baseline. Compare full_palm
with frozen_palm to isolate unfreezing under a common architecture/initialization;
compare full_palm with full_no_prior to isolate the graph prior under full training.
The prior round's frozen result is only context, since activation and Gaussian
parameter-count normalization have changed.

Same finite-penalty objective as the theory: eta=1, beta=1e-4, full parameter count
P in every arm (including the fixed encoder's constant Gaussian term). PALM uses
one parameter block containing every optimized tensor, followed by one V block.
This proves convergence of all active weights, not convergence of data-derived
graph construction or the original stochastic MARBLE training process.

## Solver, data, budget and seal

Use the unchanged tested `split_step` from `marble_provable_splitting.py`: 300
full-batch sweeps; tmax=16, c=.1, V step n/(eta+2lambda+1), at most 40 parameter
backtracking candidates. Exact ball projections by float64 SVD. No Adam/CG,
momentum, stochastic gradients, checkpoint selection, silent retries or tuning.
Precompute constant standardized features; for the frozen control cache the fixed
smooth encoder outputs. Both caches preserve the stated objective and gradients.

Majorization has no positive acceptance tolerance. Save every step and stop if
sufficient descent margin is below -1e-12 or relative ball violation is at least
1e-12. Report all numerical deviations and final stationarity residuals; 300 steps
does not imply arrival at a critical point. Record every tensor's displacement
and assert that all effective encoder tensors changed in full arms, and none
changed in the frozen arm.

Use the same data hashes, 15NN phase-space graph, four positive/negative pairs per
node, three fixed pair seeds 0/1/2, Achilles 32D decode models, four author 3D rat
models, and unchanged evaluation from the prior protocol. Graph and features are
constructed in float32, then full network fitting uses float64 CUDA, TF32 off.
45 fits plus 15 initial controls; seal all 60 output embeddings before evaluation.
Fit uses no labels; inputs contain labels but fitting does not read them.

Cosine kNN k=36 position decoding and all 12 directed position-binned in-sample
OLS consistency scores are reported. Three pair seeds do not add independent rats.
Report all seeds and sample SD, not confidence intervals or significance claims.
The same observed holdout makes these exploratory comparisons.

Use shared uv-managed official PyTorch 2.14.0+cu130 / RTX 5070 Ti Laptop.
Monitor process and progress; review at 45 minutes without silently retrying or
changing the fixed budget. Mathematical tests and full repository tests precede
fitting. Hash code, theory, protocol, dependencies, baseline and data at fit start;
hash every embedding at completion and verify before evaluation.

```powershell
../KAIR/.venv/Scripts/python.exe main_explore_marble_unfrozen.py fit --input ../KAIR/results/marble-input-20260928 --rat-input ../KAIR/results/rat-consistency-input-20260928-final --baseline ../KAIR/results/marble-kair-20260928-final --output results/unfrozen-20260928
../KAIR/.venv/Scripts/python.exe main_explore_marble_unfrozen.py evaluate --output results/unfrozen-20260928
```

Commit compact reports, full iteration diagnostics and metrics; keep model arrays
and embeddings in ignored local results. Failed runs, if any, remain visible.
