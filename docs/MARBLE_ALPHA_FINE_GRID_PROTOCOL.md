# MARBLE local alpha refinement

Material Passport: academic-research-suite / experiment-agent; mode run/validate;
origin 2026-09-30; status PLANNED until execution and audits complete.

This is an adaptive, exploratory follow-up to `MARBLE_ALPHA_GRID_RESULTS.md`.
The earlier common decoding optimum was alpha=0.03. All four evaluation records
have already been examined; refining on them does not create a fresh test set.
This protocol is frozen before the new alpha=0.02/0.04/0.05 outcomes.

## Search and reuse

- Active alpha grid: **0, 0.01, 0.02, 0.03, 0.04, 0.05**. Zero is exact PCA.
- New points: **0.02, 0.04, 0.05**; reuse 0, 0.01, 0.03 from the completed coarse
  grid manifest after verifying its saved input, model and audit receipts.
- Exclude 0.1, 0.3 and 1 from this local search, as requested. Preserve all
  historical results, including those unfavorable to the hypothesis. The old
  full-grid report remains available and is not overwritten.
- Four rats (Achilles, Buddy, Cicero, Gatsby), both existing protocols, seeds
  0/1/2, 100 epochs each. This is 48 condition bundles / 144 models: 24 new
  bundles / **72 new CUDA 13 float32 trainings**, 24 reused bundles / 72 models.
- Finish all planned seeds and retain every new result. Do not prune a new
  point after seeing an unfavorable first seed or change this grid mid-run.
- Keep projection dimensions, preprocessing, graph rules, network, optimizer,
  model selection and evaluation fixed. Initial weights and node splits are
  paired to each animal/protocol's PCA control; graph neighbors are recomputed
  for each projection, and original-code sampling plans are retained.

## Two separate endpoints

Decoding uses q=20, output=32, chronological 80/20 train/test split, with the
projection fitted only on training neural data. Select a common alpha by the
equal-animal mean of three-seed frozen-BN position MAE (cm). Report all animal
and seed scores, R2, notebook-BN sensitivity, and paired differences from PCA.

Consistency uses q=10, output=3, dropout=0.5 and the whole recording. Select a
common alpha by mean position-binned affine R2 over 12 directed animal pairs
and three seeds. Preserve the original CEBRA same-bin fitting/scoring convention;
this is not held-out transfer accuracy. No post-hoc combined endpoint.

For each endpoint show the complete local curve, including PCA and the reused
0.01/0.03 values. Compare both the new winner and the former 0.03 winner on both
metrics. Also show per-animal decoding minima and retrospective leave-one-animal
selection, breaking exact ties toward smaller alpha. Seeds and directed pairs
are not independent biological replicates; report descriptive mean/sample SD,
not significance or independent generalization. Behavior labels are absent
from representation training but enter evaluation and hyperparameter selection.

## Verification and artifacts

Use the unchanged numerical protocol documented in `MARBLE_ALPHA_GRID_NUMERICS.md`:
decoding retains the original-code float32 three-step gate; consistency uses
same-start float32 local steps (rtol=2e-4, atol=2e-5) and a continuous float64
three-step trajectory (rtol=1e-8, atol=1e-9). Do not claim full float32 trajectory
agreement. All formal training remains float32. Repeat the consistency gates
for reused controls in the new output directory, leaving old artifacts untouched.

Run original CPU checkpoint audits for every new bundle and independent CEBRA
0.4.0 / sklearn 1.3.2 metric audits for all 18 alpha/seed consistency scores.
Any new gate failure stops the run, preserving logs for diagnosis. Fixing a
software defect requires a disclosed source amendment, not relaxed tolerances.

Freeze job paths, parent grid manifest/completion/aggregate hashes, source
snapshots and reuse receipts under
`D:/MARBLE-experiments/KAIR/alpha-fine-grid-20260930`. Final compact artifacts go
to `docs/reports/marble-alpha-fine-grid-20260930`. This remains a bounded local
grid, not proof of a global optimum or reproduction of every MARBLE experiment.
