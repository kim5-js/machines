# Formal E3: dual-view degraded-small weighting (alpha = 1.5)

The new formal E2 is the completed pretrained lambda-zero dual-view run:

L_E2 = L_det(clean) + L_det(degraded)

E3 keeps the same locked YOLOv8s initialization, SeaDronesSee split, MDG,
image size, batch size, optimizer, learning rate, seed, GPU, and 150 epochs.
It changes one item only:

L_E3 = L_det(clean; w=1) + L_det(degraded; w_small=1.5, w_other=1)

For this first version, a small GT has post-augmentation training-pixel area
strictly below 32^2. The Task-Aligned Assigner is unchanged. After assignment,
the classification, box IoU, and DFL terms of positive anchors mapped to a
degraded-small GT are multiplied by 1.5. Clean targets, degraded non-small
targets, and background anchors remain 1.0. The original target-score
normalization denominator is retained.

Excluded: consistency loss, severity weighting, curriculum, CBAM, and
Focal-EIoU. E3 starts independently from the same locked yolov8s.pt; it does
not fine-tune the E2 checkpoint.

The formal report is selected by best validation mAP50-95 and includes
Precision, Recall, mAP50, mAP50-95, APsmall, APmedium, APlarge, and best epoch.
