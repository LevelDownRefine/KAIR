# Structural mathematical exploration — fixed protocol

## Material Passport

- ID: MARBLE-MATH-20260928; code experiment, exploratory run.
- Source base: KAIR `04ed0e140ce96fa21ea4adeed533c238cb3db2d6`.
- Scope: Achilles held-out position decoding and four-rat consistency.
- Created before candidate fitting/evaluation; this is a local commitment,
  not a registered study or evidence of independent confirmation.
- User request: three branches, structural mathematical changes, both metrics.

## Question and comparison conditions

Can a different geometric/dynamical representation, or a relational transport
operator on MARBLE representations, improve decoding and consistency together?
These are established mathematical foundations, not a claim of a newly invented
theory. Each branch tests one fixed construction without a parameter search.

| Branch suffix | Representation | Additional information |
| --- | --- | --- |
| diffusion | density-corrected diffusion coordinates | neural phase space only |
| koopman | kernel VAMP singular coordinates | neural phase space and temporal order |
| transport | GW barycentric displacement of MARBLE | pretrained MARBLE plus unlabeled reference representation |

All branches use frozen train-fitted operators at decoding test time. The
transport branch is postprocessing and uses a reference space; its results are
not evidence for a standalone MARBLE replacement under identical information.

## Fixed mathematical definitions

For the two replacement models, input is the concatenation of PCA neural state
and its first temporal difference. Center each coordinate; scale each of the
two blocks by one block-wide RMS. This preserves orthogonal-coordinate
invariance within each block but imposes equal block weight, a modeling choice.
Select 256 training landmarks uniformly, seeds 0/1/2. Use Gaussian
`K(x,z) = exp(-||x-z||² / h)`, with `h` the median squared distance from a
landmark to its 15th nearest other landmark. No behavioral labels determine h.

**Diffusion:** form landmark `q_i = sum_j K_ij`, `A_ij = K_ij/(q_i q_j)`
(density correction alpha=1), `D_ii = sum_j A_ij`, and diagonalize
`S = D^(-1/2) A D^(-1/2)`. Drop the stationary eigenvector; use the next
32 or 3 modes. A query is embedded by `P_query psi`, i.e. the time-one
Nyström diffusion coordinates `lambda * psi_query`. This is a landmark
approximation; it does not compute the full-sample Laplace operator.

**Koopman/VAMP:** normalize Gaussian dictionary rows to sum to one. At lag
one (25 ms), estimate centered `C00`, `C11`, `C01` from training time pairs.
Use ridge `1e-6 * (trace(C00)+trace(C11))/(2 * dictionary_dimension)` and
`W0 C01 W1 = U Sigma V^T`, where `Wk = (Ckk + ridge I)^(-1/2)`.
Coordinates are `(phi(x)-mean_left) W0 U_d Sigma_d`. This estimates
Koopman singular functions, not eigenfunctions, and is not a learned neural
VAMPnet. Frozen covariance estimates are never updated on the test split.

**Transport:** use 128 landmarks and squared Euclidean within-space distances,
each distance matrix divided by its own mean. Minimize
`sum_ijkl (C_ik-D_jl)² T_ij T_kl + 0.05 sum_ij T_ij(log(T_ij)-1)`
with uniform marginals. Initialize by matching nine within-space distance
quantiles. Use at most 60 damped GW updates, log-domain Sinkhorn (2000 steps,
marginal target 1e-8), then row/column downscaling and rank-one residual
rounding to enforce feasibility. Every capped inner solve and rounding L1
correction is recorded; exact feasibility does not imply solver convergence.
Accept only non-increasing regularized objective (float tolerance 1e-12),
with steps 1, 1/2, ..., 1/32. No global optimum is claimed.
Map source anchors barycentrically, then extend their displacement with inverse
squared-distance interpolation over eight nearest anchors. The Achilles
author embedding is the four-animal reference; seed-0 training MARBLE is the
decoding reference. The reference itself uses the identity map. Thus seed-0
decoding is intentionally unchanged and only two decoding seeds test transport.

## Data and evaluation

- Reuse hashed baseline bundles; verify Achilles bundle against the recorded
  baseline provenance and four-rat bundle against its export receipt.
- Achilles: original contiguous 80/20 split, 7999/1999 derivative samples,
  20 PCA dimensions. Fit neural representation on train only. Decode using
  the unchanged cosine kNN with k=36 and training position labels. Evaluate
  test MAE in centimeters and position R²; no decoder selection.
- Consistency: original 10-dimensional neural PCA and 3-dimensional output,
  four animals, 9999 samples each. Full-recording unsupervised fitting.
  Apply the existing verified position-bin/normalization/OLS protocol:
  100 edges, up to radius-two empty-bin expansion, 12 directed in-sample R².
  This is behavior-conditioned, in-sample consistency, not held-out transfer.
- 32-dimensional output for decoding and 3-dimensional output for consistency
  are separate evaluations, not the same fitted representation.
- Baseline: existing three Achilles MARBLE training seeds; author checkpoint
  eval-mode embeddings for all four rats. Candidate seeds change landmarks;
  they are not independent animals or equivalent to network-training repeats.
- Fit all three branches before evaluating any. Seal source, input, baseline,
  output hashes. Run each evaluation once and retain all outcomes.

## Interpretation and diagnostics

Report both metrics, per-seed results, paired changes, and all 12 rat pairs.
A candidate is promising on both tasks only if mean MAE decreases and mean
consistency rises; report any seed reversals, reference identities, and unequal
information conditions. No p-values or independent-subject claims from n=3
seeds / 12 dependent rat pairs. The holdout has already been observed in prior
work, so improvement is exploratory and requires fresh data/splits to confirm.
Report centered embedding scale, effective numerical rank, spectra, residuals,
and transport marginal/objective diagnostics. High consistency alone can arise
from losing useful detail; rank checks do not prove absence of semantic collapse.
GPU fits use float64 and record wall time and peak allocation. No speedup claim
against MARBLE training without including preprocessing and transport pretraining.

## Commands and artifacts

From each worktree, with the shared CUDA 13 environment (or `uv sync --frozen`):

```powershell
../KAIR/.venv/Scripts/python.exe main_explore_marble_math.py fit --input ../KAIR/results/marble-input-20260928 --rat-input ../KAIR/results/rat-consistency-input-20260928-final --baseline ../KAIR/results/marble-kair-20260928-final --output results/math-20260928
# Only after all three branches have a fit_complete.json:
../KAIR/.venv/Scripts/python.exe main_explore_marble_math.py evaluate --output results/math-20260928
```

Ignored local outputs include operators, embeddings, diagnostics, protocol,
fit receipt and summary. Commit small JSON receipts and comparison reports,
not raw data or model binaries. Retain partial failed runs under distinct paths.

## Primary references

- [MARBLE paper](https://www.nature.com/articles/s41592-024-02582-2).
- [Coifman and Lafon, Diffusion maps](https://doi.org/10.1016/j.acha.2006.04.006).
- [Wu and Noe, VAMP](https://arxiv.org/abs/1707.04659).
- [POT GW objective and solver documentation](https://pythonot.github.io/gen_modules/ot.gromov.html).
- [Altschuler, Weed and Rigollet, Sinkhorn and marginal rounding](https://arxiv.org/abs/1705.09634).
