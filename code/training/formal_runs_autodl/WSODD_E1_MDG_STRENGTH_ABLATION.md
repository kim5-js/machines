# WSODD fixed-split E1 MDG strength ablation

This release trains only two single-view MDG variants on the locked
`wsodd_sequence_disjoint_v1` split:

- `E1-Lite`: the E1-Mid active MDG branch probabilities and photometric
  strength are multiplied by `0.5`.
- `E1-Strong`: the same quantities are multiplied by `1.5`.

The completed `E0` and `E1-Mid` checkpoints are immutable comparison anchors
and are not retrained. All training settings, including pretrained YOLOv8s,
SGD, seed 0, image size 960, batch 8, the fixed train/val/test manifests, and
the 150-epoch maximum with patience 100, remain unchanged.

For active-branch probability scaling, `low_light`, `glare`, and `mixed` are
multiplied by the dose factor. The remaining probability mass is assigned to
`clean` and `standard_aug_only` in their original 2:1 ratio. Low-light
brightness/gamma deviations from the identity are multiplied by the factor;
glare area and additive intensity are multiplied directly. Severity sampling,
glare aspect ratio, and softness are unchanged.

After both runs complete, the locked fixed-test evaluator processes the same
2406 test images using the same thresholds and area ranges as the existing E0
and E1-Mid results. The exploratory balance gate is preregistered as:

- `APsmall - E0 >= 0.01`
- `mAP50-95 - E0 >= -0.003`

Passing candidates are ranked by fixed-test `mAP50-95`, then `APsmall`.
Results are single-seed mechanism evidence, not a variance-robust claim.

This release does not train E2, E3, or E4 and does not enable small-object
weighting, dual-view training, consistency loss, CBAM, Focal-EIoU, severity
weighting, or curriculum learning.
