# Execution amendment: preserve interrupted fits and finish missing models

The original process returned exit code 1 without a traceback near 15 minutes.
The cause is unconfirmed; an execution-session time limit is suspected. It had
saved 31 learned fits and was inside seed-2/decoding/full_no_prior, whose directory
is empty. No behavioral evaluation had run. The user was informed before recovery.

The original directory remains untouched. A separate completion process verifies
the original source/data/baseline hashes, reuses every completed model, restarts
the interrupted fit from its prescribed initialization, and fits the remaining
models. Architecture, initialization, all hyperparameters, pair seeds, update count
and evaluation remain unchanged. This is a recorded restart of one incomplete fit,
not an unreported clean first attempt or a selection among completed outcomes.

The wrapper calls the sealed original fit_task with only missing learned methods.
Its duplicate untrained control is compared to the original where present, and
sample pairs must match exactly. Completed original files are copied into a new
combined directory, and their hashes are checked against the untouched original.
All 60 final embeddings are sealed before evaluation. The completion script and
this amendment are added to source provenance. No behavioral labels guide recovery.

The first process did not save its final total runtime or peak allocation. Do not
invent those values: record the sum of completed per-method times and the recovery
wall time/peak separately, and retain the first process's partial progress record.

```powershell
../KAIR/.venv/Scripts/python.exe -m scripts.marble.complete_unfrozen --original results/unfrozen-20260928 --output results/unfrozen-20260928-completed
```
