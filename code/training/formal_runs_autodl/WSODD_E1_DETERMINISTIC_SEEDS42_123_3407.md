# WSODD E1 deterministic-image-id seeds 42/123/3407 experiment

## Question

Does the previously observed deterministic-image-id E1-Mid behavior persist for a fully fresh seed cohort (42, 123, and 3407)?

## Only intentional change

For each training filename stem, a versioned SHA-256 namespace derives a local
random.Random stream. The stream fixes:

- MDG branch (clean, low_light, glare, mixed, or standard_aug_only);
- severity (mild, medium, or strong) when applicable;
- brightness, gamma, glare geometry, softness, and intensity.

The assignment is independent of epoch, batch ordering, and training seed.
The locked train manifest has 3977 filename stems and all 3977 are unique.
MDG is still applied after standard YOLO augmentation and during training only.

## Conditions held fixed

- WSODD sequence_disjoint_v1: train 3977, val 1084, test 2406;
- original E1-Mid MDG probabilities, severity probabilities, parameter ranges,
  and tensor operations;
- pretrained /root/autodl-tmp/weights/yolov8s.pt;
- SGD, lr0=0.006, lrf=0.01, momentum=0.937,
  weight_decay=6.25e-05, nbs=8;
- image size 960, batch 8, max 150 epochs, patience 100;
- all augmentation, loss, and evaluator settings;
- no dual view, small-object weighting, CBAM, Focal-EIoU, consistency,
  severity weighting, curriculum, E2, E3, or E4.

All three runs (seeds 42, 123, and 3407) start fresh from the same locked pretrained
checkpoint. A non-numeric interruption may resume only its own last.pt; a
numeric failure is archived and restarts from the locked pretrained checkpoint.
Normal finite early stopping is accepted as completion.

## Evaluation

Each best.pt is evaluated on the same 2406-image fixed test with the same
COCO-area evaluator and prediction settings used for E0 and random-online
E1-Mid. The final report includes:

- per-seed and mean/SD mAP50, mAP50-95, APsmall, APmedium, and APlarge;
- unpaired cohort-mean comparisons against the existing seed-0/1/2 E0 and random-online E1 references.

The new cohort is not presented as a seed-paired comparison because matching E0/random-E1 runs do not exist for seeds 42/123/3407.

These are descriptive three-seed results, not a significance claim.


