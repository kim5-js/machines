# WSODD fixed-split E1-Mid three-seed replication

This release validates the selected E1-Mid MDG schedule on the locked
`wsodd_sequence_disjoint_v1` split using seeds 0, 1, and 2.

- The completed seed-0 E1-Mid run and fixed-test result are immutable anchors.
- Only seed 1 and seed 2 are newly trained.
- Each new run starts independently from the same locked `yolov8s.pt`.
- The optimizer and all non-seed settings remain the locked pretrained-SGD
  recipe: image size 960, batch 8, at most 150 epochs, and patience 100.
- Normal early stopping is accepted as successful completion.
- A non-numerical interruption may resume only from that run's own `last.pt`.
- A numerical failure is archived and restarted from the locked pretrained
  checkpoint, with at most three attempts per seed.

After seed 1 and seed 2 finish, the same fixed-test evaluator and prediction
parameters used for seed 0 process the same 2406 test images. The final report
contains per-seed mAP50, mAP50-95, APsmall, APmedium, and APlarge plus the
three-seed mean and sample standard deviation.

The replication gate remains descriptive against the existing E0 seed-0
reference: APsmall gain at least 0.01 and mAP50-95 decline no worse than 0.003.
Because E0 has not yet been replicated across three seeds, this release does
not claim a seed-controlled E1-vs-E0 significance test.

This release does not train E2, E3, or E4 and does not enable dual-view
training, small-object weighting, consistency loss, CBAM, Focal-EIoU,
severity weighting, or curriculum learning.
