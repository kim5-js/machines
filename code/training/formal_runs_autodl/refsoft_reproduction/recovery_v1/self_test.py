"""Full real-validation audit against preserved failed checkpoint; no optimization."""
from support import ROOT, RECOVERY_CODE, verify_recovery
from observer import ValidationAudit
from common import load, write, now, sha
from types import SimpleNamespace
import copy
import csv
import torch
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset

verify_recovery(ROOT)
out = ROOT / "diagnostics/recovery_v1_validation_audit.json"
assert not out.exists()
run = ROOT / "runs/FloW_duplicate_seed42"
cp = run / "weights/last.pt"
ck = torch.load(cp, map_location="cpu", weights_only=False)
net = ck["ema"].cuda().half().eval()
net.args = SimpleNamespace(**ck["train_args"])
net.args.cache = False
net.criterion = net.init_criterion()
data = check_det_dataset(net.args.data)
ds = build_yolo_dataset(net.args, data["val"], 8, data, mode="val", rect=True, stride=32)
dl = build_dataloader(ds, 8, 0, shuffle=False, rank=-1)
trainer = SimpleNamespace(ema=SimpleNamespace(ema=net), model=net, epoch=2,
                          optimizer=SimpleNamespace(state_dict=lambda: ck["optimizer"]))
validator = SimpleNamespace(training=True, dataloader=dl)
audit = ValidationAudit()
audit.trainer = trainer
total = torch.zeros(3, device="cuda")
prediction_noop = loss_noop = False
with torch.inference_mode():
    audit.start(validator)
    for index, batch in enumerate(dl):
        batch["img"] = batch["img"].cuda().half() / 255
        for key in ["batch_idx", "cls", "bboxes"]:
            batch[key] = batch[key].cuda()
        preds = net(batch["img"], augment=False)
        loss = net.loss(batch, preds)[1]
        total += loss
        if index == 0:
            saved = copy.deepcopy(audit.record)
            audit.end(validator)
            bare_preds = net(batch["img"], augment=False)
            bare_loss = net.loss(batch, bare_preds)[1]
            prediction_noop = torch.equal(preds[0], bare_preds[0]) and all(
                torch.equal(a, b) for a, b in zip(preds[1], bare_preds[1]))
            loss_noop = torch.equal(loss, bare_loss)
            assert prediction_noop and loss_noop
            audit.start(validator)
            audit.record = saved
    audit.end(validator)
rows = list(csv.DictReader((run / "results.csv").open()))
metrics = {k.strip(): float(v) for k, v in rows[-1].items()
           if k.strip().startswith(("val/", "metrics/"))}
actual = [round(float(v), 5) for v in total / len(dl)]
expected = [metrics["val/" + name + "_loss"] for name in ["box", "cls", "dfl"]]
assert actual == expected, (actual, expected)
observation = audit.check(trainer, metrics)
assert observation["metrics"]["val/cls_loss"] is None
assert observation["validation_loss_audit"]["overflow_batches"] == 26
rejections = []
for key, value in [("val/cls_loss", float("nan")), ("val/box_loss", float("inf")),
                   ("metrics/mAP50-95(B)", float("inf"))]:
    altered = dict(metrics)
    altered[key] = value
    try:
        audit.check(trainer, altered)
    except FloatingPointError:
        rejections.append(key)
    else:
        raise AssertionError("Unexpectedly accepted " + key)
write(out, {"status": "passed", "utc": now(), "images": len(ds), "batches": len(dl),
            "checkpoint_sha256": sha(cp), "recovery_manifest_sha256": sha(RECOVERY_CODE / "manifest.json"),
            "prediction_unchanged": prediction_noop,
            "original_loss_unchanged": loss_noop, "raw_csv_loss_reproduced": True,
            "rejected_nonfinite_metrics": rejections, "observation": observation})
print({"status": "passed", "images": len(ds), "batches": len(dl),
       "prediction_unchanged": prediction_noop, "original_loss_unchanged": loss_noop,
       "raw_csv_loss_reproduced": True, "rejected_nonfinite_metrics": rejections})
