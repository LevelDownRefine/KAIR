# Spectral pseudocontractive MARBLE layer: fixed protocol

## Material Passport

- ID: MARBLE-SPECTRAL-PC-20260928; exploratory network experiment.
- User authorization: modify the network and use spectral regularization, following
  the discussion of pseudocontractive operators and convergence.
- Base: `68920a4`, branch `codex/marble-spectral-pc`; depends on the prior experiment.
- This local protocol is fixed before behavioral evaluation, not preregistration.
- Scope: modify the deployed network with a learned latent implicit layer. Freeze
  the existing MARBLE encoder to isolate the new operator and spectral treatment.

## Network and mathematical scope

The complete network maps neural features H to a frozen MARBLE anchor q, then
to the solution of z = S_q(z). The operator acts on each sample independently;
all samples share learned weights. It can therefore process held-out samples
without refitting, changing training BN statistics, or building a test prior graph.

Define a latent MLP N(z) = W2 [2 tanh((W1 z + b1)/2)] + b2, with widths d, 2d, d.
It has no BN, dropout or final unit normalization. Initialize W1 as stacked
I and -I, each scaled by .999/sqrt(2), and W2 as their horizontal counterpart;
biases zero. All three arms use exactly this initialization. Its untrained
fixed-point network is recorded as an extra architecture control.

The activation is 1-Lipschitz. Therefore Lip(N) <= L = ||W1||2 ||W2||2.
Construct the denoiser D = 2N-I, and the data term G_q(z) = ||z-q||².
The full fixed-point map is T_q = D - grad G_q = 2N-3I+2q.
A Mann step of size 1/4 gives

    z_next = (3/4)z + (1/4)T_q(z) = (N(z)+q)/2 = S_q(z).

When L <= 1, the following are exact-arithmetic structural statements:

1. D is 1/2-strictly pseudocontractive, since (I+D)/2 = N is nonexpansive.
2. T_q is 2/3-strictly pseudocontractive: (2/3)I+(1/3)T_q has Lipschitz
   constant at most (1+2L)/3 <= 1. Thus we control the full map, not D alone.
3. S_q is contractive with factor c=L/2 <= 1/2 on the complete Euclidean space.
   Banach's theorem gives existence and uniqueness of its fixed point and
   convergence from any starting state. No compact invariant set is assumed.
4. A final residual r=||S_q(z)-z|| bounds the fixed-point error by r/(1-c).
5. The converged anchor-to-output map F satisfies

       ||q-p||/(2+L) <= ||F(q)-F(p)|| <= ||q-p||/(2-L).

   Proof: q-p = 2(F(q)-F(p)) - (N(F(q))-N(F(p))), then use the triangle
   and reverse triangle inequalities. In the hard arm this gives the looser
   uniform bounds 1/3 and 1. Distinct anchors cannot collapse to one output.
   This statement concerns Euclidean distances, not cosine distances, semantic
   preservation, rank, decoding error or cross-animal alignment.

For 1 < L < 2 the inference contraction/error bound still holds, but the above
1/2-strict pseudocontractive certificate for D is not established by this bound.
A failed upper-bound certificate does not itself prove a violated property.

The guarantees apply with FIXED learned weights. Adam training is nonconvex and
unrolling is approximate; neither training convergence nor global optimality is
claimed. A general asymmetric neural operator need not have a scalar potential,
so this is an equilibrium network, not an asserted MAP solver. Final outputs are
not renormalized: normalization would invalidate the stated distance bounds.

## Three spectral arms

| Arm | Treatment |
| --- | --- |
| unconstrained | Same latent architecture, no spectral loss or hard cap |
| soft_spectral | Add 10 * sum_l max(sigma_max(W_l)-1,0)^2 |
| hard_spectral | Rescale each matrix after every Adam update to norm <= .999 |

Soft regularization encourages a condition; it is not a global guarantee.
Hard rescaling uses small-matrix double-precision SVD with .001 slack, not a
power-iteration estimate and not an exact nearest-point spectral projection.
Final SVD bounds are numerical checks of a structural proof, not verified
interval arithmetic. Biases do not affect the Lipschitz bound.

All arms: 300 full-batch Adam updates, lr=1e-3, 16 unrolled inference steps
starting at q, fixed final weights. Loss = original logistic pair-loss formula
on unnormalized latent outputs + R2 + 1e-4||phi||²/(2P), plus the soft term
where applicable. R2 is the existing normalized quadratic graph energy.
The magnitude-sensitive loss changes because these outputs are not unit vectors;
this is a network variant, not numerically identical original MARBLE training.
All learned arms share this change. No behavioral selection or parameter grid.

Evaluation runs the frozen final implicit layer up to 100 steps. Stop when the
maximum per-sample residual certifies Euclidean error <=1e-6 if L<2; otherwise
record residual <=1e-6 as an empirical criterion without an error certificate.
Always report the cap/tolerance result and the difference from 16-step outputs.
No silent retry, checkpoint selection or residual-based weight selection.

## Data and comparison

- Reuse the hashed Achilles split and the three existing 32D checkpoints.
- Reuse all four author 3D rat checkpoints, one per animal. Three pair-sampling
  seeds do not constitute three independently trained author baselines.
- Same train-only 15NN neural phase-space graph and four positive/four negative
  pairs per node as the previous prior experiment. This is not the original
  CkNN/random-walk sampling procedure.
- Three arms * three seeds * (decoding + four animals) = 45 fits. Record 15
  untrained architecture controls as well. Fit all before behavioral evaluation.
- Freeze original encoder weights and compare its outputs to saved reference
  embeddings (rtol=2e-4/atol=2e-5). Test anchors only enter final inference.
- No behavioral labels enter fitting, spectral constraints, step choices or
  checkpoints. Same cosine kNN k=36 and position-binned four-rat OLS consistency
  as before. Consistency is in-sample and behavior-conditioned, not transfer.
- Compare against frozen MARBLE, untrained implicit layer, and matched
  unconstrained learned layer. Previous ADMM means are contextual references,
  not solver-isolating comparisons: the architecture and trainable weights differ.
- Save all seeds, directed pairs, objective histories, layer bounds, sampled
  Jacobians, fixed-point residuals, unroll errors, output scale and numerical rank.
  Sampled Jacobian checks are diagnostics, not proofs over all inputs.
- Previously observed holdout: exploratory conclusions only, no significance or
  generalization claims from three seeds or dependent animal pairs.

## Execution and artifacts

Use shared uv-managed official PyTorch CUDA 13 environment and the laptop GPU.
Run mathematical tests before real fits. Monitor progress and exit status; a
45-minute fit budget triggers a review, not an unreported termination/retry.

```powershell
../KAIR/.venv/Scripts/python.exe main_explore_marble_spectral.py fit --input ../KAIR/results/marble-input-20260928 --rat-input ../KAIR/results/rat-consistency-input-20260928-final --baseline ../KAIR/results/marble-kair-20260928-final --output results/spectral-20260928
../KAIR/.venv/Scripts/python.exe main_explore_marble_spectral.py evaluate --output results/spectral-20260928
```

Seal code, protocol, data and baseline hashes before fitting, then all 60 output
embedding files before evaluation. Large model files and embeddings stay in
ignored results; commit compact diagnostics and the complete metrics.

## Sources

- [Wei, Chen and Li, ICML 2024](https://proceedings.mlr.press/v235/wei24b.html):
  pseudocontractive denoisers and convergence assumptions. This experiment is
  our anchored construction, not a reproduction of that paper's image benchmarks.
- [PyTorch spectral_norm](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html):
  the built-in normalization estimates the norm with power iteration; this small
  latent network instead uses SVD to avoid mistaking an underestimate for a cap.
