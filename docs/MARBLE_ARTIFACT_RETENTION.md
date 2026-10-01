# MARBLE experiment artifact retention

On 2026-09-30, during the local alpha refinement, the user explicitly requested
deleting poorly performing results to free disk space. Earlier retention promises
are superseded for the following large derived artifacts only.

Deleted **378 files / 12,458,642,787 bytes (11.603 GiB)** from the alpha=0.1,
0.3 and 1 coarse-grid conditions and the earlier Achilles alpha=10 condition:
`.pt`, `.pth` and `.gz` graph, sampler, embedding, oracle and model files.
The 0.1/0.3/1 conditions were worse than 0.03 on both coarse common endpoints;
the early Achilles alpha=10 did not beat its exact-PCA control.

Retained **967 files** in those directories: metrics, logs, diagnostics, manifests,
initialization/sampler receipts, numerical checks, model audit reports and
protocol metadata. Compact historical reports and source archives remain intact.
No source recording, backup, PCA, alpha=0.01/0.03 or new local-grid file was deleted.

The [cleanup manifest](reports/marble-alpha-cleanup-20260930.json) records exact
paths, byte sizes, SHA-256 for every deleted and retained file, parent manifest
hashes, and completion time. Retained files were verified unchanged after cleanup.
The frozen experiment plans are left unchanged; this file records the later
user-authorized storage decision separately.

Historical audit records describe verification performed before deletion. The
deleted conditions can no longer be re-audited or reused directly from their
tensor/checkpoint files and would require regeneration/retraining. Their published
scores remain available, including unfavorable results. The active fine grid
reuses only the preserved 0/0.01/0.03 bundles and is unaffected.

Material Passport: academic-research-suite / experiment-agent; mode run;
origin 2026-09-30; status COMPLETE (storage cleanup only).
