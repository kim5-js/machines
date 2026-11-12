# WSODD fixed-split E0 three-seed replication

This release supplies the missing seed-matched E0 controls for the completed
WSODD sequence-disjoint E1-Mid experiment.

- Reuse completed E0 seed 0 without retraining.
- Train only E0 seed 1 and seed 2.
- Start each new run independently from the same locked `yolov8s.pt`.
- Keep the exact pretrained-SGD recipe used by E1-Mid: image size 960, batch
  8, at most 150 epochs, patience 100, and the audited fixed split.
- Disable MDG, dual-view training, small-object weighting, consistency loss,
  CBAM, Focal-EIoU, severity weighting, and curriculum learning.
- Accept a finite, provenance-valid normal early stop as completion.
- Resume a non-numerical interruption only from the same run's `last.pt`.
- Restart a numerical failure from the locked pretrained checkpoint, with at
  most three attempts per seed.

After training, the exact evaluator and prediction parameters used for the
completed E1-Mid three-seed report process the same 2406 fixed-test images.
The final artifact reports E0 and E1-Mid per seed, mean and sample standard
deviation, and seed-matched E1-Mid minus E0 deltas for mAP50, mAP50-95,
APsmall, APmedium, and APlarge.

The three paired seeds support a descriptive stability assessment only; this
release does not pre-register or claim a formal significance test. It does not
train E1, E2, E3, or E4.
