# Higher-order brain connectivity utilities

`utils.utils_hoi` provides the eleven connectivity summaries discussed in
[Santoro et al., *Charting higher-order models of brain function beyond pairwise
interactions* (2026)](https://doi.org/10.1038/s41467-026-75959-w).
The implementation uses the existing NumPy/SciPy dependencies, on CPU, with no
JAX, Julia, SxPID or topology package required at runtime. The calculations are
implemented from the mathematical definitions; external reference code is not
vendored. This is a reusable implementation of the measures, **not a claim of
bit-for-bit reproduction of the paper's complete HCP analysis**.

## Input and outputs

Supply **one run**, shaped `(n_timepoints, n_regions)`, with finite real values.
For example, 1,200 samples from 116 ROIs is `(1200, 116)`, not `(116, 116)`.
Imaging preprocessing, nuisance regression, filtering, censoring, atlas
parcellation and ROI extraction are upstream of these functions. ROI order must
remain identical across runs and across matrices being compared. Constant ROIs
raise an error (explicit discrete/binary information estimators allow constants).
Inputs are not modified; no missing samples, ROIs or frames are silently dropped.

At 116 ROIs there are **253,460 triplets**, **6,670 pairs**, and 114 triplets
incident to each edge. Projected maps have shape `(116, 116)`. They are symmetric
with zero diagonals, except FC whose diagonal is one. The maps are statistical
descriptions; they do not establish directed causal effects or consciousness.

```python
import numpy as np
from utils import utils_hoi as hoi

# Example synthetic run; replace with your cleaned time-by-ROI array.
x = np.random.default_rng(42).normal(size=(300, 8))

# Choose a subset explicitly, or metrics=hoi.ALL_METRICS for all eleven.
maps = hoi.compute_hoi(
    x,
    metrics=("FC", "O-Red", "O-Syn", "PED-Red", "PED-Syn",
             "PhiID-Red", "PhiID-Syn"),
    reduction="mean",
    lag=1,
)
names, rho = hoi.compare_connectomes(maps)
np.savez("hoi_maps.npz", **maps)
np.savez("hoi_similarity.npz", names=np.array(names), rho=rho)
```

Keep run boundaries intact, especially for PhiID: concatenating runs creates
artificial lagged observations across boundaries. After censoring, `lag=1` means
adjacent **retained samples**, which may not be adjacent acquisitions; use
continuous valid segments if that distinction matters. These utilities do not
infer temporal gaps or average subject/run results for you. Save the ROI labels,
run identifiers, preprocessing choices and all estimator arguments with outputs.

## Measures and explicit conventions

| Map | Calculation | Units / sign |
| --- | --- | --- |
| FC | Pearson correlation between regional series | Correlation; signed |
| O-Red | Positive triplet O-information, then projection | Bits; nonnegative |
| O-Syn | Negative triplet O-information, then projection | Bits; **negative values retained** |
| PED-Red | Bottom atom `{1}{2}{3}` of binary shared-exclusion entropy decomposition | Bits |
| PED-Syn | Sum of the seven specified joint-source entropy atoms | Bits |
| PhiID-Red | MMI redundancy-to-redundancy atom `rtr` for past/future ROI pairs | Bits |
| PhiID-Syn | MMI synergy-to-synergy atom `sts` | Bits; no atom clipping |
| Triangles | Time-averaged positive weights of triangles violating face order, then projection | Standardized product weights |
| TScaffold | Time-average of H1 cycle frequency on valid framewise complexes | Cycle counts |
| Fscaff | H1 cycle frequency for the static `1 - FC` filtration | Cycle counts |
| Pscaff | H1 cycle persistence summed over each representative edge | `1 - FC` filtration units |

### O-information and projection

For `d` variables, `Omega = (d-2) H(X) + sum_i [H(X_i) - H(X_without_i)]`.
For a triplet this is co-information. `o_information(x)` computes Omega for
**all supplied columns**, while `oinfo_triplets(x)` enumerates triples.

`method="gc"` (default) transforms marginal average ranks with
`ndtri(rank / (T+1))`, then uses Gaussian log-determinant entropy with the
finite-sample digamma correction. `method="gaussian"` uses centered sample
covariances without this correction by default. `biascorrect` overrides that
choice. No covariance regularization is implicit; singular relevant covariance
blocks raise an error. A full N-dimensional covariance need not be invertible
when only triplets are evaluated. Gaussian/copula estimates do not capture
arbitrary nonlinear dependence, such as binary XOR; use `method="discrete"`
for categorical observations (each distinct value is a category, no binning).

```python
raw = hoi.oinfo_triplets(x)  # raw.triplets: (K,3); raw.values: (K,)
edge_maps = hoi.split_oinfo(raw, reduction="mean")
node_maps = hoi.split_oinfo(raw, target="nodes", reduction="mean")
```

Redundancy and synergy remain **two distinct maps**. Do not add them before
comparing the individual measures: that discards the sign-separated information.
`project_triplets(values, triplets, n_regions, ...)` supports:

- `sum`: add all incident supplied triplets;
- `mean` (default): divide by the count of all incident supplied triplets,
  **including zero contributions** after sign splitting;
- `mean_nonzero`: divide by the number of nonzero contributions instead.

Mean and sum differ by a constant factor for complete triplet enumeration at a
fixed N, so their edgewise Spearman correlations agree. This does not generally
hold for nonzero-only normalization or selected triplets. Missing incidences
are returned as zero. Explicit triplets must be unique, strictly increasing
ROI indices. Their row order is preserved; defaults are lexicographic.

### PED

`ped_triplets` thresholds each ROI at its temporal mean: above mean is 1;
below or exactly at mean is 0. This equals thresholding z-scored BOLD at zero.
Pass `binary=True` for already binary 0/1 data.

For each observed joint state, shared entropy is
`h_sx(alpha) = -log2 P(union of the matching source events in alpha)`.
Average over the empirical binary joint distribution, then invert the
18-node redundancy lattice. The redundancy score is `{1}{2}{3}`. The synergy
score sums `{3}{12}`, `{2}{13}`, `{1}{23}`, `{12}{13}{23}`, `{13}{23}`,
`{12}{23}`, and `{12}{13}`. Scores are not divided by joint entropy.

`ped_from_probabilities(p)` exposes all 18 atoms for an eight-state distribution,
indexed by `X0 + 2*X1 + 4*X2`. Antichain keys use source bitmasks: `1=X0`,
`2=X1`, `4=X2`, `3={X0,X1}`, etc. This is an **entropy** decomposition, not a
PID with one brain region chosen as a target. Independent fair bits can have
positive shared entropy: the bottom atom is `log2(8/7)`, not zero.

### PhiID

`phiid_pair(x[:, [i,j]], lag=1, method="gaussian")` returns all 16 MMI atoms.
Keys combine past/future labels `r`, `u1`, `u2`, `s` with `t`, for example
`rtr`, `u1tu2`, and `sts`. `u1` and `u2` refer to the first and second supplied
ROI, respectively. The sum of all atoms equals `I(pair_past; pair_future)`.
MMI `rtr` is the minimum of the four single-ROI past/future mutual informations.
The other cumulative redundancies and atoms follow the product of two
two-source redundancy lattices. Individual atoms need not be nonnegative.

`phiid_matrices` uses the same estimator for both maps (Gaussian by default).
The high-level `compute_hoi` defaults mirror the author's scripts' choices:
`phiid_red_method="gc"`, `phiid_syn_method="gaussian"`. Set both explicitly to
the same method if comparing the two atoms under one estimator. The gc
transform is applied to each full run before lagging, then each lagged
covariance is centered. Discrete PhiID is also supported, without automatic
thresholding. Gaussian PhiID needs at least five retained lagged observations
and nonsingular four-variable covariance.

### Topology

`correlation_scaffolds(x)` returns frequency, persistence, H1 intervals and
representative edge cycles. Edges enter at `1 - FC[i,j]`; each triangle enters
at the maximum of its three edge birth times. No rank transformation of edge
weights is applied. The implementation reduces boundary matrices over F2.

`temporal_higher_order` first z-scores regional series and triple products over
the full run. A product's absolute standardized magnitude gets a positive sign
only when all instantaneous regional values have the same nonzero sign.

- `edge_mode="standardized"` (default) applies the same standardization/sign
  rule to pair products, following the author's **SimplicialTS source code**.
- `edge_mode="raw"` uses `z_i*z_j`, following the **article's edge formula**.

In descending weight order, a triangle is valid only if all three edges have
already entered (`triangle_weight <= min(edge_weights)`). Invalid triangles
are removed, not delayed. Only positive invalid weights contribute to the
Triangles feature. H1 on the valid complex determines TScaffold; replacing it
with an edge-only Vietoris-Rips complex would change the measure. Constant
product series are assigned zero standardized fluctuation, avoiding division
by zero. Temporal means include zero contributions.

```python
# Example explicit frame subset; the full run still defines normalization.
temporal = hoi.temporal_higher_order(x, frames=[0, 10, 20])
triangle_map = hoi.project_triplets(
    temporal.triangles.values, temporal.triangles.triplets, x.shape[1]
)
tscaffold = temporal.scaffold
# All frames: omit frames. Triangles alone: scaffold=False.
```

Cycle representatives are **not unique**. We sort equal-weight edges and faces
lexicographically and put faces before cofaces at ties. Finite representatives
are reduced triangle boundaries; essential representatives are birth cycles.
Only positive-length intervals contribute. This agrees with standard H1
barcodes, but scaffold edge weights can differ from Ripserer's chosen cycles
or after relabeling tied nodes. Essential cycles count in frequency and produce
`+inf` persistence; the complete static filtration has no essential H1 cycles.
`scaffold_from_filtration` exposes custom triangle sets/births for inspecting
these choices. `compare_connectomes` rejects infinite maps.

## Comparing maps

`edge_vector(matrix)` extracts the strict upper triangle in
`np.triu_indices(N, 1)` order. `compare_connectomes` ranks those aligned vectors
with average ranks for ties and returns a method-by-method Spearman matrix.
Constant vectors have undefined correlations, returned as NaN (also on their
diagonal). There are no per-edge significance p-values: edges are not assumed
independent. Similarity measures agreement of spatial rank patterns; it does
not validate an estimator or imply equivalent biological mechanisms.

## Scale and validation

Gaussian triplets use batched 3x3 matrices; PED and product normalization use
chunks; temporal topology processes one frame at a time. The code never
allocates a full `(time, all_triplets)` array. Memory for these chunks still
scales with time and `chunk_size`. Topology enumerates O(N^3) faces per frame,
and boundary reduction has data-dependent cost; full HCP runs can be expensive.
Start with an explicit frame subset for exploratory topology, then compute all
frames when that is the intended analysis. No implicit sampling occurs.

Validation covers discrete independent/copy/XOR distributions, Gaussian
entropy identities, all 18 PED atoms, all 16 PhiID atoms, cycle closure,
finite/essential H1 intervals, frame averaging, projection denominators, ties,
invalid input, and all eleven output maps. The offline reference fixture records
outputs and pinned source versions for independent SxPID and PhiID comparisons.
Forty complete/partial-face random filtrations were additionally compared with
GUDHI 3.13.0; this checks barcodes, not representative-cycle identity. No HCP
dataset was run as part of these tests.

```powershell
python -m pytest tests/utils/test_utils_hoi.py -q
python -m pytest tests -k "not cuda" -q
```

## Sources and reproduction differences

- [Paper](https://doi.org/10.1038/s41467-026-75959-w).
- [Author analysis scripts, pinned ce1a32c](https://github.com/nplresearch/HOI_lenses_analysis/tree/ce1a32ccba63dcc20c495b77be7352ad2993575d):
  O-info/PhiID calls, PED atom selections, static scaffolds and SimplicialTS.
- [HOI O-information estimator](https://brainets.github.io/hoi/api/generated/hoi.metrics.Oinfo.html)
  and [Gaussian-copula entropy](https://brainets.github.io/hoi/api/generated/hoi.core.entropy_gc.html).
- [SxPID reference](https://github.com/nplresearch/HOI_lenses_analysis/blob/ce1a32ccba63dcc20c495b77be7352ad2993575d/Code/04_PED/sxpid/SxPID.py).
- [MMI PhiID reference, pinned 6c5f2e9](https://github.com/Imperial-MIND-lab/integrated-info-decomp/tree/6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44).
- [GUDHI simplex-tree documentation](https://gudhi.inria.fr/python/latest/simplex_tree_ref.html).

Reproduction choices to record: mean versus sum versus nonzero-only projection;
raw versus standardized temporal edges; covariance centering and bias correction;
average ranks for ties (HOI's argsort-based implementation assigns distinct
ranks); zero-product handling; lag and frame selection; representative cycles.
The paper describes average projections while some author notebooks use sums
and the optional source normalization excludes zeros. These choices are exposed
rather than silently treated as equivalent. There is no additional FDR,
surrogate testing, fingerprinting, task decoding or behavioral PLS analysis here.
