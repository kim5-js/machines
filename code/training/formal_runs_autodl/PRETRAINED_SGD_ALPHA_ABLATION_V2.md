# Pretrained-SGD alpha ablation supervisor v2

This versioned release keeps the v1 training recipe and model code immutable.
It changes supervisor bookkeeping only.

The v2 fit-epoch guard uses the persisted `results.csv` epoch as the source of
truth. Ultralytics 8.4.124 temporarily increments `trainer.epoch` and invokes
`on_fit_epoch_end` again during final evaluation. If no new results row exists,
v2 records `duplicate_fit_epoch_end_ignored` and does not consume empty MDG or
target counters. Every real training epoch still runs all finite-loss, MDG,
target, and dual-view checks.

The normal train-end proof also follows Ultralytics' actual indexing:
`trainer.epoch` is zero-based, while `EarlyStopping.best_epoch` is already
one-based because the trainer calls the stopper with `epoch + 1`. A finite,
contiguous run with `best.pt`, `last.pt`, and a patience-complete stopper gap is
therefore recorded as normal `early_stopping`, not as a failed attempt.

The v2 entry also accepts serialization-only `device` equivalence such as
integer `0` versus YAML string `"0"`; all other locked arguments remain exact.

The one-time alpha1.2 recovery is intentionally incident-specific. It requires
the predecessor manifest, exact failure events, exact finite epoch 1..122
history, best epoch 22, the 100-epoch patience gap, both checkpoints, locked
metadata/runtime evidence, and proof that attempt 2 never entered training.
It archives the predecessor state before marking alpha1.2 complete and never
reruns or resumes that checkpoint. The next and only fresh run is alpha1.35.
