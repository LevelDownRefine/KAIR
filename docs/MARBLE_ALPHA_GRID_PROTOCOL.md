# MARBLE projection alpha grid

Frozen before new grid outcomes on 2026-09-30. This is an exploratory follow-up
to the fixed-alpha four-rat validation, which did not support a general gain at
alpha=0.1. The four animals and their evaluation records have already been seen.
Grid maxima are tuning results, not independent confirmation of generalization.

## Fixed search

- Alpha: **0, 0.01, 0.03, 0.1, 0.3, 1**; alpha=0 is exact PCA.
- Animals: Achilles, Buddy, Cicero, Gatsby; all available public recording bins.
- Seeds: **0, 1, 2**, 100 epochs each. Report all seeds, no early pruning.
- Two separate protocols, identical to `MARBLE_MULTIRAT_PROTOCOL.md`: decoding
  q=20/output=32 with chronological 80/20 split; consistency q=10/output=3,
  dropout=0.5, trained on the whole recording.
- Projection fits training neural activity only for decoding. No behavior labels
  enter projection, graph construction, or representation training.
- Keep dimensions, smoothing, graph construction, architecture, learning rate,
  checkpoint selection, decoder and batch-normalization evaluation mode fixed.
- Primary decoding metric: frozen-BN position MAE in cm. Secondary: R² and
  notebook-BN sensitivity. Cross-animal metric: mean over all 12 directed
  position-binned affine R² scores, preserving the existing CEBRA convention.

Reuse 17 fully audited condition bundles (51 trained seeds): alpha=0/0.1 for
all animals and both protocols, plus Achilles decoding alpha=1. Train the other
31 bundles (93 seeds) with CUDA 13. Each reuse is a manifest reference to the
original immutable location, with input, initialization, checkpoint, embedding,
summary and audit hashes. No old results are overwritten or resampled.

All interventions must match each animal/protocol's alpha=0 initialization and
node split masks. Use the existing original-code three-SGD-step checks, independent
float64 oracle for 3D consistency, and original CPU checkpoint audits. Preserve
failed checks and stop rather than dropping a condition or relaxing tolerances.

## Reporting and parameter choice

Show every animal's alpha curve and three seed values. For decoding select one
common alpha by the equal-animal mean of the three-seed MAEs (not the best seed).
Also show animal-specific minima as exploratory upper bounds on tuning benefit.
For consistency select one common alpha by mean R² across seeds and all 12 pairs.
Tie-break equal scores toward smaller alpha. Report whether either optimum also
improves the other metric; do not invent a combined metric after seeing scores.

As a diagnostic, select decoding alpha on three animals and score it on the
fourth, repeating all four leave-one-animal-out folds. This is a retrospective
selection-sensitivity analysis within an already studied dataset, not a new
unseen-animal validation. Seeds and directed pairs are not biological replicates;
mean±sample SD is descriptive and no significance claim is planned.

The search is bounded at alpha=1 based on the earlier Achilles sweep (alpha=1
and 10 both worse than PCA there). This boundary is an informed design choice,
not evidence that higher alpha cannot help another animal. Any boundary optimum
requires a separately disclosed expansion; this run does not add points post hoc.

Scope remains the public four-rat subset and released-checkpoint/notebook-derived
protocol family; it does not cover RNN, macaque, or every experiment in the paper.

Material Passport: academic-research-suite / experiment-agent; mode run/validate;
origin 2026-09-30; status PLANNED until execution and audits complete.
