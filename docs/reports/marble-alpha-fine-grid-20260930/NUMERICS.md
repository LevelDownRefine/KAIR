# Numerical verification amendment (2026-09-30)

The fixed search stopped before alpha=0.03/Achilles/consistency training. The
original CPU float32 three-step trajectory itself differed from the original
float64 oracle beyond the previous bridge tolerance (one embedding element,
absolute difference 2.6134e-5). Modern CPU float32 matched the original CPU;
continuous float32 CUDA differences accumulated in later SGD steps. The failed
precision.log and complete drift diagnostic are retained. Twelve new condition
bundles had finished at this point; no alpha=0.03 consistency seed had trained.

The revised verification separates implementation agreement at a common state
from numerical drift between continuous floating-point optimization trajectories:

1. At each of the same three reference batches, restore original pre-step weights,
   BatchNorm buffers and SGD momentum. Compare float32 CUDA features, embeddings,
   loss, gradients and the resulting update with the original float32 step at
   **rtol=2e-4, atol=2e-5**, unchanged. Momentum is reconstructed with original
   gradients using the original SGD recurrence, including first-step semantics.
2. Independently execute the entire three-step trajectory in original CPU float64
   and modern CUDA float64, without any state resets, using identical stored
   float32 inputs/initial weights cast to double, IDs and fixed dropout masks.
   Compare all features, embeddings, losses, gradients, BN buffers and updates at
   **rtol=1e-8, atol=1e-9**.
3. Keep original float32 checkpoint/output audits after training at their existing
   tolerances. Export and disclose original continuous float32-versus-float64 drift
   even when it exceeds the old bridge threshold; do not relabel it as a pass.

Apply these checks to **all 24 consistency conditions**, including already trained
and reused ones. Extra audit artifacts live under the new grid's numerics directory;
historical bundles remain immutable. The decoding gate remains unchanged because
the observed issue is in the separate three-dimensional dropout protocol.

Formal training remains float32 CUDA with the original architecture, initialization,
sampling, dropout, optimizer, epochs, checkpoint selection and evaluation. No
training code or experimental grid point is changed. This amendment changes what
the pre-training numerical check establishes: it does **not** assert that continuous
float32 trajectories are elementwise equal across devices.

The original frozen_protocol.json and execution_source.zip remain unchanged.
execution_amendment.json binds that frozen file to the revised execution hashes;
execution_source_amended.zip preserves the revised code. A portable real-data
regression fixture checks both positive agreement and deliberately corrupted
gradients, dropout masks and optimizer momentum.
