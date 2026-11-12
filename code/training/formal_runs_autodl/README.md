# AutoDL runtime

The AutoDL instance uses its data disk for environments, datasets, caches,
checkpoints, and logs. The 30 GB system disk is intentionally kept nearly
empty.

## Environment split

- `mdc-yolo26`: Python 3.10, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124,
  and Ultralytics 8.4.124. This is the pinned environment for unified YOLOv8,
  YOLO11, YOLO26, and RT-DETR baselines and subsequent MDC development.
- `ring3-legacy`: Python 3.10.8, PyTorch 2.5.1+cu121, torchvision
  0.20.1+cu121, torchaudio 2.5.1+cu121, and Ultralytics 8.3.182. This mirrors
  the historical Windows runtime without upgrading it in place.

On the instance, source `remote_env.sh` before running project commands.

## Current dataset configs

The YAML files under `datasets/` point to the persistent data disk. Publication
runs must use the protocol-specific configs rather than the historical
train/validation folders. Do not report validation metrics as final test
metrics.

The available FloW copy contains only the official 1200-image training pool,
currently divided into 1000 development-train and 200 development-validation
images. The separate official 800-image test set has not been obtained, so
FloW publication testing remains blocked pending the dataset-owner download.

SeaDronesSee's source COCO category `0` is an ignore region. The converted YOLO
labels contain class IDs `0..4` for swimmer, boat, jetski,
life_saving_appliances, and buoy. The historical local `dataset.yaml` lists the
ignore category as a sixth trainable class and is therefore not used here.

The historical WSODD 6222/1245 split is frame-random. A 60-second timestamp
group audit found train/validation mixing in 98 of 147 capture sequences,
covering 7290 of 7467 images. `wsodd_sequence_disjoint_v1.yaml` replaces it
with whole-sequence 56/14/30 train/validation/test splits. The 56/14 development
pool preserves the source paper's overall 70/30 train-test proportion. Exact
manifest hashes and per-class counts are recorded under
`manifests/wsodd_sequence_disjoint_v1/protocol.json`.

The official labeled SeaDronesSee train and validation folders mix 34 of the
39 source groups recoverable from the COCO metadata. The development config
`seadronessee_source_disjoint_v1.yaml` rebuilds them as 8922 train and 1555
validation images with zero source overlap. Video frames stay with their exact
source video. Trinity still images expose no flight/session ID, so each whole
source acquisition folder stays together. The split contains 29 train and 10
validation source groups; all five trainable classes occur on both sides.
Exact manifests and hashes are under
`manifests/seadronessee_source_disjoint_v1/`.

Audit a dataset before any run:

```bash
source /root/autodl-tmp/ring3_setup/autodl/remote_env.sh
python /root/autodl-tmp/ring3_setup/autodl/verify_yolo_dataset.py \
  /root/autodl-tmp/ring3_setup/autodl/datasets/flow_current.yaml
```

Add `--require-test` for locally evaluated publication runs. It succeeds for
the WSODD sequence-disjoint protocol and intentionally fails for FloW until the
official test set is available. SeaDronesSee test labels are withheld, so that
benchmark is evaluated by server submission instead.

`calibrate_runtime.py` performs a short timing/VRAM calibration. Its output is
explicitly non-scientific and must not be copied into result tables.

## Formal baseline queue

`formal_experiment_lock.json` is the authorized immutable lock for the 12
seed-0 modern baselines. It fixes the environment, checkpoint hashes, dataset
YAML and protocol hashes, split counts, `imgsz=960`, `batch=8`, 150 epochs,
`optimizer=AdamW`, dataset-specific learning rates, and seed 0. Formal runs
must not use `optimizer=auto`.

`formal_baseline_queue.py` verifies the lock before training, records its SHA256
on first activation, runs the fixed model/dataset matrix sequentially, stops on
the first failed or non-finite final result, and resumes only from a compatible
`last.pt`. Queue state and events are written to:

```text
/root/autodl-tmp/formal_state/baseline_queue_state.json
/root/autodl-tmp/formal_state/baseline_queue_events.jsonl
```

Formal outputs are isolated under `/root/autodl-tmp/formal_baselines`; timing
calibrations and preflight diagnostics remain under their separate directories.
Create `/root/autodl-tmp/formal_state/STOP_AFTER_CURRENT` to pause safely between
runs. Do not alter the lock after its SHA256 has been recorded.

SeaDronesSee v2 withholds test labels. After the final model is frozen,
`export_seadronessee_coco.py` predicts all 3750 official test images, maps YOLO
class IDs `0..4` back to official COCO category IDs `1..5`, and writes a
hash-recorded submission JSON for the benchmark server.

## Monitoring

The detached `ring3_monitor` screen session writes one snapshot per minute to
`/root/autodl-tmp/monitor/latest.json` and appends history to
`/root/autodl-tmp/monitor/history.jsonl`. It records GPU utilization, VRAM,
temperature, power, data-disk usage, active screen sessions, and the most recent
Ultralytics `results.csv` row.

The snapshot also embeds the formal queue state and the latest formal
`results.csv` row. A separate `ring3_formal_baselines` screen owns the running
formal queue.
