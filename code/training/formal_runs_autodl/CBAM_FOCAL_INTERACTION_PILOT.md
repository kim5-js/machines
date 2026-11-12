# CBAM + Focal-EIoU interaction pilot

## Question

The plain E2 lambda-zero regional diagnostic found lower P3 discrepancy inside GT object regions than in the background. This isolated pilot tests the narrower hypothesis that an object-oriented discrepancy pattern may emerge only after the detector has CBAM and Focal-EIoU.

## Scope

- Non-paper pilot only: 30 epochs, seed 0, SeaDronesSee, image size 960, batch 8.
- Starts independently from the locked YOLOv8s pretrained weight.
- Keeps the existing online MDG and clean/degraded dual-view lambda-zero flow.
- Adds CBAM only after backbone C2f layers 4 and 6.
- Replaces the CIoU regression term with Focal-EIoU while retaining DFL.
- Does not modify the formal lock, runtime amendment, data split, E0-E2 results, or completed scratch runs.
- Does not implement Object-Aware consistency and does not start formal E3 or E4.

## Gated sequence

1. Verify release hashes, completed scratch evidence, locked weight/data hashes, and the absence of a stop sentinel.
2. Run a static model/loss/gradient self-test.
3. Run a one-epoch smoke test.
4. Run the 30-epoch lambda-zero interaction pilot.
5. From `best.pt`, run the fixed 200-batch global/foreground/background/small-object P3 cosine diagnostic.
6. Stop and report the gate. Even a passing gate requires user review before Object-Aware consistency or E3 is implemented.

## Foreground gate

All conditions must pass:

- exactly 200 batches;
- global cosine reproduces the pilot epoch-30 anchor within 0.003;
- at least 500 valid foreground image pairs;
- foreground discrepancy exceeds background discrepancy in at least 70% of batches;
- the 95% bootstrap lower bound of paired foreground-minus-background discrepancy is above zero;
- mean foreground/background discrepancy ratio is at least 1.05.

## Remote paths

- screen: `ring3_cbam_focal_interaction_pilot`
- pilot root: `/root/autodl-tmp/pilots/cbam_focal_interaction`
- pilot state: `/root/autodl-tmp/formal_state/cbam_focal_interaction_pilot_state.json`
- diagnostic state: `/root/autodl-tmp/formal_state/cbam_focal_region_diagnostic_state.json`
- log: `/root/autodl-tmp/logs/cbam_focal_interaction_pilot.log`
