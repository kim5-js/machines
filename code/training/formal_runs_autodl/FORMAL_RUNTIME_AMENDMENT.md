# Formal runtime amendment

The immutable scientific lock remains `formal_experiment_lock.json` with SHA256
`9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1`.

`formal_runtime_amendment.json` records one operational precision exception:
RT-DETR-L runs use `amp=false`. The installed Ultralytics `RTDETRTrainer`
explicitly warns that AMP can produce NaN outputs and bipartite-matching errors.
The failed FloW run showed repeated validation NaNs, then non-finite training
losses and a non-finite epoch-150 row. Dataset YAML files, manifests, splits,
weights, optimizer, learning rate, epochs, batch size, image size, and seed are
unchanged.

`formal_baseline_queue_resilient.py` preserves failed directories under
`/root/autodl-tmp/formal_failed_runs`, retries each run up to three times, and
continues to the next run if those attempts are exhausted. It fails fast on any
non-finite training batch or epoch, and on three consecutive epochs with
non-finite validation loss. The baseline segment ends as either `completed` or
`completed_with_failures`; a single run failure no longer exits the segment.
