# Fixed-alpha MARBLE validation across four rats

Frozen before new-animal training on 2026-09-29. This is a follow-up to
`MARBLE_DYNAMICS_PROJECTION_RESULTS.md`, not a claim of full paper reproduction.

Protocol family: released-checkpoint/notebook-derived. The paper's PCA=5 and
supplementary-table settings are separate protocols; q=20/q=10 results do not
replace those experiments. See `MARBLE_FULL_REPRODUCTION_PLAN.md` for differences.

The exploratory Achilles sweep suggested alpha=0.1. That value is fixed here;
new animals are never used to select alpha, projection dimension, or epochs.
Exact PCA (alpha=0) is the paired control. Both conditions use identical fresh
initial weights and node split masks. Original CPU graph construction/sampling
feeds CUDA 13 training; independent original-code checkpoint audits are required.

## 1. New-animal position decoding

Buddy (6,577 bins), Cicero and Gatsby (10,000 bins each). Chronological 80/20
split, smoothing separately within each segment, projection fit on training only.
Projection dimension 20, MARBLE output 32, dropout 0; other hyperparameters inherit
the existing Achilles protocol. Three seeds (0,1,2), 100 epochs, minimum internal
contrastive validation loss checkpoint. Primary metric: frozen-BN offline test
position MAE (cm); also report R², all individual seeds and notebook-BN sensitivity.
Cosine kNN uses 36 neighbors. Test graph uses its whole segment, so this is offline
decoding. Preserve the author's smoothing and event/bin conventions.

Reuse the previous Achilles alpha=0 and 0.1 results only as discovery/reference.
Report the three new animals separately; do not count seeds as independent animals
or select just the best animal/seed. Mean±SD over seeds is descriptive uncertainty.

## 2. Cross-animal representation consistency

Train every animal from scratch: Achilles, Buddy, Cicero, Gatsby, including all
available bins (6,577 for Buddy; 10,000 for each other animal). Follow the author consistency architecture: input projection 10,
output 3, hidden 64, dropout 0.5. Three seeds, 100 epochs and the same unsupervised
checkpoint selection as above. This is separate from the 32D decoding models and
separate from the earlier evaluation of author-provided checkpoints.

For each condition/seed, compute all 12 directed animal-pair R² values using the
original position-binned alignment and in-sample affine regression. Behavior is
used for evaluation alignment, never representation training. This metric is not
held-out transfer or zero-shot decoding. The 12 pairs share four animals and are
not 12 independent biological samples. Audit against pinned CEBRA 0.4.0.

## Decision and scope

Report all outcomes whether favorable or not. Improvement in projection residuals
is a mathematical property, not a guarantee of downstream MAE or consistency.
An effect on Achilles alone does not establish generalization. Mixed results
require a conditional conclusion, not selection of the strongest run.

This stage covers the public four-rat subset. RNN, macaque and other paper
experiments remain separate validation work. Each output stores the frozen plan,
source/data hashes, original sampler plans, initialization hashes, all training
losses, selected checkpoints and independent audits. Gzip storage is lossless.

Preflight correction: the initial exporter incorrectly asserted 10,000 rows for
every animal and stopped on Buddy before any output graph or training was created.
The failed preflight log/plan are retained in `results/dynamics-multirat-20260929`.
The corrected run uses `results/dynamics-multirat-20260929-v2`; no outcomes were
available when the sample-count metadata was corrected.

Storage transition (2026-09-30): after four completed/audited decoding conditions
(Buddy and Cicero; 12 training seeds), the runner's 2 GiB disk guard prevented
starting Gatsby. The C drive had fallen below that threshold. Completed artifacts
were copied to `D:/MARBLE-experiments/KAIR/dynamics-multirat-20260929-v2`, verified
file-by-file with SHA-256, and C duplicate copies removed. The original workspace
result path is now a directory junction. All frozen execution-source hashes matched
before `--resume`; completed training was skipped. This changed storage only.

Diagnostic addendum (2026-09-30, before any consistency model began training):
also describe the mean of the six directed pairs involving only Buddy, Cicero and
Gatsby. These animals did not select alpha. Keep the same four-animal bin alignment
and the original 12-pair primary endpoint; this is a secondary decomposition, not
a changed evaluation protocol or an independent biological sample of size six.

Precision amendment (2026-09-30, before any 3D training seed ran): the first
Achilles 3D pre-training check stopped because 1/576 float32 embedding entries
differed by 3.0112e-5, just outside `atol=2e-5, rtol=2e-4`. CPU/CUDA float64
diagnostics agreed, while float32 rounding accumulated across SGD steps. The failed
log and source snapshot are retained. A new independent oracle runs the original
MARBLE code in float64 on the same stored float32 initial weights, graph values,
sampled IDs and fixed dropout masks. Both the original float32 reference and the
modern CUDA float32 implementation must pass the unchanged tolerance against this
oracle. Ten CUDA checks passed; a portable real-data regression fixture also checks
gradients, updates, masks and deliberate corruptions. All 3D conditions use this
same gate. Formal training remains float32; no hyperparameter, initialization,
sampling plan, preprocessing or evaluation endpoint changed. Original float32
checkpoint audits after training remain required. `precision_amendment.json`
records the source hashes and retained diagnostics. The initial frozen protocol
file remains immutable; this is a disclosed validation amendment.
