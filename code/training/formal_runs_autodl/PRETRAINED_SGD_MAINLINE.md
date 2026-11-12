# Pretrained SGD mainline

This release reruns SeaDronesSee E0-E3 from the same locked `yolov8s.pt`, with
the historical local YOLOv8s SGD recipe. The only intended training-setting
changes from the historical local v8s run are pretrained initialization,
`batch=8`, and `seed=0`.

The historical dataset file is deliberately not restored: it represented an
ignore category as a sixth class and mixed source groups between train and
validation. All four new runs therefore share the audited five-class
`source_disjoint_v1` split (8922/1555). This is recorded as an intentional
protocol difference, so comparisons against the historical local metrics must
not be described as a strict pretrained-only ablation.

Run order:

1. E0: single-view pretrained YOLOv8s baseline.
2. E1: single-view train-only online maritime degradation.
3. E2: clean and degraded views, shared YOLOv8s, detection loss on both,
   no consistency loss.
4. E3: E2 plus weight 1.5 for degraded-small assigned positive anchors.

CBAM, Focal-EIoU, consistency, severity weighting, curriculum, and E4 are all
disabled. Every run starts independently from the same locked pretrained
checkpoint. A non-numeric interruption may resume only from that run's own
`last.pt`; a numeric failure is archived and restarts from the locked
pretrained checkpoint.
