# Pretrained-SGD mainline supervisor v2

This patch changes only queue supervision and completion accounting. The locked E0-E3 definitions, dataset, pretrained weights, SGD recipe, maximum epochs, and `patience=100` remain unchanged.

Completion policy:

- A finite run with contiguous results and both `best.pt` and `last.pt` is complete at the configured maximum epoch.
- A finite early-stopped run is complete only when a hashed `on_train_end` callback records normal return and the result history reproduces the configured patience gap.
- A completed checkpoint is never resumed.
- The trainer save directory must match the canonical formal run directory before metadata is written.

One-time recovery:

- The paused E0 is adopted only after exact validation of epoch 125, best epoch 25, a 100-epoch patience gap, finite history, locked args, runtime wiring, attempt-1 pretrained provenance, and the original event proving `model.train()` returned normally.
- Invalid resume attempts 2 and 3 remain excluded and are not reused.
- The old state and protocol are copied into a timestamped recovery archive before the protocol is migrated to this release.
- E1, E2, and E3 still start independently from the same locked pretrained `yolov8s.pt`.

Recovery command (only while the audited stop sentinel is present):

```bash
/root/autodl-tmp/envs/mdc-yolo26/bin/python -u /root/autodl-tmp/ring3_setup/autodl/pretrained_sgd_mainline_v2_entry.py --recover-paused-e0
```

Normal launcher:

```bash
/root/autodl-tmp/ring3_setup/autodl/run_pretrained_sgd_mainline_v2_entry.sh
```
