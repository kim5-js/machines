# Formal pretrained CBAM / Focal-EIoU series

This queue implements the user-requested unified-condition comparison without
overwriting the earlier 30-epoch interaction pilot or any completed E0-E3 run.

The completed original E0 is reused as the `CBAM=off, Focal-EIoU=off` cell.
Five independently initialized 150-epoch runs then execute serially:

1. CBAM only.
2. Focal-EIoU only.
3. CBAM + Focal-EIoU.
4. Combined base + formal E2 dual view (clean + degraded, shared detector,
   no consistency loss).
5. Combined E2 + formal E3 degraded-small assigned-positive weighting 1.5.

Every fresh run starts from the same locked `/root/autodl-tmp/weights/yolov8s.pt`.
An interrupted run may resume only from its own `last.pt`. Numeric failures are
archived and restart from the locked pretrained source. The queue never starts
E4, consistency loss, severity weighting, or curriculum.

Remote paths:

- screen: `ring3_formal_cbam_focal_series`
- output: `/root/autodl-tmp/formal_cbam_focal_series`
- state: `/root/autodl-tmp/formal_state/cbam_focal_formal_queue_state.json`
- events: `/root/autodl-tmp/formal_state/cbam_focal_formal_queue_events.jsonl`
- protocol: `/root/autodl-tmp/formal_state/cbam_focal_formal_queue_protocol.json`
- summary: `/root/autodl-tmp/formal_state/cbam_focal_formal_queue_summary.json`
- log: `/root/autodl-tmp/logs/cbam_focal_formal_series.log`
- stop sentinel: `/root/autodl-tmp/formal_state/STOP_CBAM_FOCAL_FORMAL_SERIES`
