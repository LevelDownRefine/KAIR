# MARBLE geometric priors, ADMM and PnP: fixed experiment protocol

## Material Passport

- ID: MARBLE-PRIORS-20260928; exploratory code experiment.
- User scope: investigate structural MAP priors and try splitting methods,
  including PnP and ADMM. Both position MAE and cross-animal consistency matter.
- Base: KAIR `04ed0e1`; branch `codex/marble-geometry-admm`.
- This protocol is written before candidate behavioral evaluation. It is a
  local commitment, not a registered study or independent confirmation.
- No claims of new theory, global optimality, or completed whole-paper replication.

## Question

Does an explicit local geometric prior improve MARBLE's inductive encoder,
and does separating representation fitting from the prior help under a matched
encoder-update budget? Compare against both the frozen baseline and continuation
with the same data, pairs, optimizer, buffers and budget but no geometric prior.
The output is always the encoder, not a smoothed set of training codes.

## Conditional prior and actual objectives

Let `Z = Phi_theta(H)` be MARBLE's unit-normalized embeddings from its original
first-order neural graph features. Construct a *fixed* symmetric graph using
only training neural PCA positions and first differences: center each block,
divide each block by its RMS, use 15 nearest neighbors in concatenated phase
space, Gaussian squared-distance weights with bandwidth equal to the median
15th-neighbor squared distance, then average directed weights. There are no
cross-animal edges or behavioral inputs. This graph differs from MARBLE's
original CkNN sampler; every experimental arm shares the new graph and pairs.

Write `L = (D-W)/max_degree`, and oriented incidence `B` with
`B.T B = L`. It follows that `||L||_2 <= 2` and constants are in its nullspace.
The priors penalize different mathematical quantities:

- Quadratic: `R2(Z) = trace(Z.T L Z)/(2n)`.
- Vectorial graph TV: `Rtv(Z) = sum_edges ||(BZ)_e||_2 / n`.
  The norm couples embedding channels and is invariant to a global orthogonal
  rotation of latent coordinates; this is not invariance to arbitrary local
  changes of coordinates in the input.

Every arm includes a common weak proper Gaussian parameter prior,
`R0(theta) = 1e-4 ||theta||²/(2P)`, where theta comprises encoder parameters
and P is their count. Thus the explicit conditional prior can be written
`p(theta|H,G) proportional to p0(theta) exp(-M lambda R(Phi_theta(H)))`.
Here the pair-loss is an average with M=4n positive and M negative pairs;
multiplying the whole objective by M gives a summed Bernoulli negative
log-likelihood and consistently scaled prior. We assume conditional pair
independence for this auxiliary classification model; repeated/shared nodes
do not provide independent biological evidence. These are input-conditioned
priors, not population priors independent of the observed neural graph.

Explicit objective: `f(theta) + lambda R(Phi_theta(H))`, where
`f = mean positive softplus(-dot) + mean negative softplus(dot) + R0`.
Sample four positive neighbors and four uniform negative nodes per anchor,
with replacement, once per seed. Reuse the same fixed pairs in all arms and
iterations; no behavior-dependent selection. Negatives may coincide with neighbors,
as in uniform negative sampling. Graph smoothness overlaps with the pair loss's
assumption and is not independent evidence of biological similarity.

## Five fixed arms

| Arm | Prior / update |
| --- | --- |
| control | direct optimization of f, no geometric prior |
| direct_quadratic | direct optimization of f + R2 |
| admm_quadratic | same f + R2, split via V = Phi_theta(H) |
| admm_tv | f + 0.1 Rtv, same split |
| pnp | same encoder step, adaptive graph denoiser instead of prox |

For the three splitting arms, rho=1, initialize V to the pretrained encoder
output and scaled dual U=0. At every outer iteration:

1. Take 10 Adam steps on `f(theta) + rho ||Phi_theta(H)-V+U||²/(2n)`.
2. Quadratic: solve `(rho I + lambda L)V = rho(Z+U)` by CG.
   TV: solve `.5||V-(Z+U)||² + (lambda/rho) sum_e ||BV_e||` using
   100 projected dual-gradient steps of size .49, warm starting the dual.
3. Update `U <- U + Z - V`.

CG relative tolerance is 2e-6 with maximum 60 iterations. Record actual linear
residual and TV primal-dual gap; a fixed iteration cap is not proof of convergence.
The encoder equality is nonlinear and its update inexact. Therefore this is
nonconvex ADMM-style optimization; the standard convex ADMM theorem does not apply.

PnP uses `D(q) = q - .5 L_a(q) q`, where each fixed edge weight is multiplied
by `exp(-||q_i-q_j||² / s²)` and the original maximum-degree normalization is
retained. Fix s² to the median pretrained embedding edge difference (floor 1e-6).
This adaptive denoiser is not asserted to be the proximal map of any prior,
globally nonexpansive, or a MAP solver. Report its primal/dual iteration residuals
as diagnostics, not a proof of an equilibrium or optimality.

All arms: 30 outer blocks, 10 encoder updates per block, Adam lr=1e-3,
no scheduler, no early stopping, fixed final iterate. Direct methods run the
same 300 encoder updates. This matches update counts, not wall time or all FLOPs.
No parameter grid, seed selection, metric-driven reruns, or decoder changes.
Hyperparameters and prior strengths are modeling choices, not claimed equivalent
across quadratic, TV and PnP. BN running buffers remain frozen, dropout disabled;
all encoder weights and BN affine parameters are optimized. This is controlled
warm-start continuation, not the original training procedure or from-scratch fit.

## Data, label separation and endpoints

- Reuse hashed Achilles training bundle and the three existing trained 32D
  checkpoints. Train is 7999 anchors; test is 1999. Load original graph features;
  test is only passed through the fitted encoder, never into the prior graph,
  parameter updates, BN statistics, or model selection.
- For 3D consistency, initialize each rat from its original author checkpoint,
  fitting separately to all 9999 neural samples. Repeat with sampling seeds
  0/1/2. These are pair-sampling repeats, not three new independent animals or
  independent author-model training seeds.
- Five arms x three seeds x (one decoding task + four animals) = 75 fits.
  Every fit verifies the starting checkpoint reproduces its baseline embedding
  at the previously established rtol=2e-4 / atol=2e-5.
- Fit all 75 before any behavioral evaluation; seal sources, data, checkpoint
  hashes and embeddings. Evaluation reads position labels only after the seal.
- Fixed cosine kNN k=36 for position decoding. Four-rat consistency reuses the
  verified position-bin normalization and 12 directed in-sample OLS R² scores.
  These scores are behavior-conditioned and are not held-out cross-animal transfer.
- Primary comparisons: each prior arm against matched control and frozen baseline;
  direct quadratic vs ADMM quadratic isolates the solver under the fixed budget.
  Report both tasks, all seeds and all directed pairs, numerical rank, centered
  embedding scale, graph energy, objective histories, solver gaps and residuals.
  No significance claims from dependent pairs or three seeds. No collapse claim
  from R² alone; full numerical rank also does not prove semantic preservation.
- The holdout has been observed in earlier work. All conclusions are exploratory;
  fresh time splits and animals are required before generalization claims.

## Execution

Use the shared uv-managed official CUDA 13 environment, or `uv sync --frozen`.
Monitor exit status, per-fit logs and progress receipts. A 45-minute whole-fit
budget triggers a status review before any termination; never silently retry
or replace failed artifacts. Mathematical test failures may be corrected before
real fitting; preserve any later numerical failure and explain the correction.

```powershell
../KAIR/.venv/Scripts/python.exe main_explore_marble_priors.py fit --input ../KAIR/results/marble-input-20260928 --rat-input ../KAIR/results/rat-consistency-input-20260928-final --baseline ../KAIR/results/marble-kair-20260928-final --output results/priors-20260928
../KAIR/.venv/Scripts/python.exe main_explore_marble_priors.py evaluate --output results/priors-20260928
```

Large checkpoints, pair plans and embeddings remain in ignored `results/`.
Commit small summaries and numerical diagnostics for review.

## Sources

- [MARBLE, Methods and Eq. 12](https://www.nature.com/articles/s41592-024-02582-2).
- [Boyd et al., ADMM](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html).
- [Parikh and Boyd, proximal algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf).
- [Venkatakrishnan et al., original PnP](https://brendt.wohlberg.net/publications/venkatakrishnan-2013-plugandplay.html).
- [Buzzard et al., consensus equilibrium](https://arxiv.org/abs/1705.08983).
