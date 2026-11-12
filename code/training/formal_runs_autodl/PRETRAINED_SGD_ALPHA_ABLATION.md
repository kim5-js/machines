# Pretrained-SGD small-object weight alpha ablation

This isolated queue completes the existing formal alpha sweep without adding a
new model component. Existing E2 is alpha=1.0 and existing E3 is alpha=1.5;
neither is rerun. The queue trains only alpha=1.2 followed by alpha=1.35.

All other conditions remain locked: the same pretrained `yolov8s.pt`, five-class
source-disjoint SeaDronesSee split, SGD recipe, seed 0, batch 8, image size 960,
150 maximum epochs, patience 100, and train-only maritime degradation. CBAM,
Focal-EIoU, consistency loss, severity weighting, curriculum, and E4 are absent.

Each new run starts independently from the locked pretrained checkpoint. A
non-numerical interruption may resume only its own `last.pt`; a numerical
failure restarts from the same locked pretrained checkpoint. Normal finite early
stopping is accepted only when the completion callback records a patience gap
of at least 100 epochs.
