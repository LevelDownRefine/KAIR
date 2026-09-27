# Numerical audit after the recovery initialization check

The first recovery process finished both missing decoding fits, then stopped
because its rtol=1e-10 / atol=1e-11 initialization comparison failed. No behavioral
metrics were opened. Both completed models and the original process files remain.

Audit findings: original checkpoint tensors and sampled pairs match exactly.
Only graph-gradient feature columns 40:440 have different normalization buffers;
position/signal columns are identical. Buffer mean/scale differences are about
1e-8. Undoing the feature standardization gives effective first-layer weight and
bias differences of 1.09e-19 and 5.56e-17. All remaining network tensors are equal.
Initial output max differences are 5.45e-10 (train) and 2.48e-10 (test), with maximum
row distances 1.27e-9 and 5.46e-10. This is consistent with float32 sparse feature
recomputation and rounding, not different model weights or sampling.

The recovery check was too strict for a pipeline whose graph features are float32.
This amendment explicitly replaces that recovery-output threshold with absolute
1e-8, together with much stronger structural checks of original weights, effective
affine maps, unchanged columns and sample pairs. It is a numerical audit threshold,
not a formal floating-point error bound or an exact-reproduction claim. Training
acceptance/descent tolerances, model, initialization recipe and evaluation do not
change. No behavioral score informed this decision. The failed check is retained.

Keep the original 31 fits and the two additional completed decoding fits. Execute
the four remaining animal tasks in separate processes, each under the suspected
session lifetime; do not repeat completed training. Seal the 60 combined outputs
with source and origin records, then evaluate once. Both previous directories
remain untouched. The final directory is `results/unfrozen-20260928-final`.

```powershell
../KAIR/.venv/Scripts/python.exe -m scripts.marble.finish_unfrozen audit
../KAIR/.venv/Scripts/python.exe -m scripts.marble.finish_unfrozen task --rat achilles
../KAIR/.venv/Scripts/python.exe -m scripts.marble.finish_unfrozen task --rat buddy
../KAIR/.venv/Scripts/python.exe -m scripts.marble.finish_unfrozen task --rat cicero
../KAIR/.venv/Scripts/python.exe -m scripts.marble.finish_unfrozen task --rat gatsby
../KAIR/.venv/Scripts/python.exe -m scripts.marble.finish_unfrozen seal
```
