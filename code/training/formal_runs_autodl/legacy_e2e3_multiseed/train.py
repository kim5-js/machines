"""One fresh historical run, with observation and failure guards only."""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import traceback

import common as c

def main(job_key, phase):
    req = c.request()
    job = c.resolve_job(req, job_key, phase)
    adapter_sha = c.verify_release(req)
    preflight_sha = c.sha(c.ROOT / "preflight.json")
    run = Path(job["run_dir"])
    c.require(not run.exists(), "Existing run preserved; automatic overwrite or resume is forbidden")
    data_counts = c.verify_data(req, job)
    versions = c.bootstrap(req)
    import numpy as np
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.nn.modules import Detect
    from ultralytics.utils import LOGGER
    from ultralytics.utils.tal import TaskAlignedAssigner
    from ultralytics.utils.torch_utils import unwrap_model
    import ultralytics.utils.checks as checks
    import train_pretrained_sgd_mainline as historical
    import train_wsodd_pretrained_sgd_mainline as wsodd
    import wsodd_pretrained_sgd_hotfix_v3_entry as hotfix

    TaskAlignedAssigner.forward = hotfix.task_aligned_assigner_forward_bool_mask
    checks.check_pip_update_available = lambda: None
    choice = c.variant(job)
    os.environ["RING3_SGD_MAINLINE_VARIANT_JSON"] = json.dumps(choice)
    os.environ["RING3_SGD_MAINLINE_MDG_CONFIG"] = str(c.ROOT / "historical_code/e1_mdg_config.json")
    os.environ["RING3_SGD_MAINLINE_MDG_SHA256"] = req["mdg_config"]["sha256"]
    expected = c.expected_args(req, job)
    base = historical.MainlineSGDTrainer if job["arm"] == "E3" else wsodd.WSODDMainlineTrainer
    evidence = {"job": job, "runtime": versions, "source_trainer": base.__module__ + "." + base.__name__,
                "request_sha256": c.REQUEST_SHA, "adapter_manifest_sha256": adapter_sha,
                "pretrained_weight": req["pretrained_weight"], "data_counts": data_counts,
                "preflight_sha256": preflight_sha, "data_yaml_sha256": c.sha(job["data_yaml"]),
                "amp_mask_hotfix": True, "automatic_argument_changes_allowed": False,
                "semantic_batches_checked": 0, "initial_scaled_nonfinite_gradient_steps": 0}
    log_messages = []
    class AmpLog(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "AMP:" in message:
                log_messages.append(message)
    handler = AmpLog()
    LOGGER.addHandler(handler)

    def finite(value, label):
        if isinstance(value, dict):
            for key, item in value.items():
                finite(item, label + "/" + key)
        elif isinstance(value, torch.Tensor):
            c.require(bool(torch.isfinite(value.detach()).all()), "Nonfinite " + label)
        elif value is not None:
            c.require(math.isfinite(float(value)), "Nonfinite " + label)

    def state_hash(model):
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str((str(tensor.dtype), list(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    class ObservedTrainer(base):
        def get_model(self, cfg=None, weights=None, verbose=True):
            c.require(weights is not None, "Fresh pretrained initialization is required")
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            original = weights["model"] if isinstance(weights, dict) else weights
            source, target = original.state_dict(), model.state_dict()
            keys = [key for key in target if key.startswith("model.") and int(key.split(".")[1]) < 22]
            c.require(len(keys) > 100, "Expected YOLOv8s backbone and neck tensors are absent")
            for key in keys:
                c.require(key in source and source[key].shape == target[key].shape
                          and torch.equal(source[key].float(), target[key].float()),
                          "Pretrained backbone/neck transfer mismatch: " + key)
            ds = c.dataset(req, job["dataset"])
            c.require({str(k): v for k, v in model.names.items()} == ds["names"], "Dataset class names changed")
            c.require(model.model[-1].nc == len(ds["names"]), "Detection head has the wrong class count")
            source_names = {str(value).strip().lower(): int(key) for key, value in original.names.items()}
            mapping = {int(key): source_names[value.strip().lower()] for key, value in ds["names"].items()
                       if value.strip().lower() in source_names}
            remapped_rows = 0
            for name, module in model.named_modules():
                if not isinstance(module, Detect):
                    continue
                for index, sequence in enumerate(module.cv3):
                    for suffix in ("weight", "bias"):
                        key = f"{name}.cv3.{index}.{len(sequence)-1}.{suffix}"
                        if key not in source or source[key].shape[1:] != target[key].shape[1:]:
                            continue
                        for dst, src in mapping.items():
                            c.require(torch.equal(target[key][dst].float(), source[key][src].float()),
                                      "Pretrained class-row remapping mismatch: " + key)
                            remapped_rows += 1
            c.require(not mapping or remapped_rows > 0, "Expected pretrained class rows were not transferred")
            evidence["initialization"] = {"state_sha256": state_hash(model), "backbone_neck_tensors": len(keys),
                "nc": len(ds["names"]), "class_mapping": mapping, "verified_remapped_rows": remapped_rows}
            for other in ("E0", "E2", "E3"):
                if other == job["arm"]:
                    continue
                path = run.parent / f"{job['dataset']}_{other}_seed{job['seed']}" / "runtime_evidence.json"
                if path.exists():
                    prior = c.read(path)
                    c.require(prior["initialization"]["state_sha256"] == evidence["initialization"]["state_sha256"],
                              "Paired arms did not start from identical tensors")
            c.write(run / "runtime_evidence.json", evidence)
            return model

        def _build_train_pipeline(self):
            c.verify_args(vars(self.args), expected, DEFAULT_CFG_DICT)
            c.require(self.batch_size == 8, "The runtime attempted to change the historical batch size")
            return super()._build_train_pipeline()

        def preprocess_batch(self, batch):
            c.require(self.batch_size == 8 and self.args.batch == 8, "Historical batch size changed")
            count, targets = int(batch["img"].shape[0]), int(batch["bboxes"].shape[0])
            result = super().preprocess_batch(batch)
            multiplier = 1 if job["arm"] == "E0" else 2
            c.require(result["img"].shape[0] == count * multiplier, "Wrong number of views")
            c.require(result["bboxes"].shape[0] == targets * multiplier, "Wrong target replication")
            finite(result["img"], "training images")
            if multiplier == 2:
                for key in ("cls", "bboxes"):
                    c.require(torch.equal(result[key][:targets], result[key][targets:]), "Dual-view labels changed")
                c.require(torch.equal(result["batch_idx"][targets:], result["batch_idx"][:targets] + count),
                          "Dual-view batch indices changed")
            if job["arm"] == "E3":
                boxes = result["bboxes"][:targets]
                area = boxes[:, 2] * result["img"].shape[-1] * boxes[:, 3] * result["img"].shape[-2]
                wanted = torch.cat((torch.ones_like(area), torch.where(area < 1024, 1.5, 1.0)))
                c.require(torch.equal(result["e3_gt_weights"], wanted), "E3 target weighting changed")
                c.require(result["e3_pair_batch_size"] == count, "E3 clean/degraded boundary changed")
            else:
                c.require("e3_gt_weights" not in result and "e3_pair_batch_size" not in result,
                          "Small-object weighting leaked into E0/E2")
            evidence["semantic_batches_checked"] += 1
            return result

        def optimizer_step(self):
            finite(self.loss, "training loss before optimizer")
            finite(self.loss_items, "training loss components before optimizer")
            if "finite_gradient_probe" not in evidence:
                gradients = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
                all_finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
                if all_finite and any(bool((g != 0).any()) for g in gradients):
                    evidence["finite_gradient_probe"] = {"step": self._optimizer_step_calls + 1,
                        "gradient_tensors": len(gradients), "finite": True, "nonzero": True,
                        "amp_scale": float(self.scaler.get_scale())}
                elif not all_finite:
                    evidence["initial_scaled_nonfinite_gradient_steps"] += 1
            # Keep the frozen GradScaler and optimizer update, including native overflow skips.
            return super().optimizer_step()

        def validate(self):
            metrics, fitness = super().validate()
            finite(metrics, "validation metrics/loss")
            finite(fitness, "native validation fitness")
            return metrics, fitness

        def _handle_nan_recovery(self, epoch):
            finite(self.loss, "training loss before native recovery")
            finite(self.fitness, "fitness before native recovery")
            return super()._handle_nan_recovery(epoch)

        def save_model(self):
            for label, model in (("model", unwrap_model(self.model)), ("EMA", self.ema.ema)):
                for key, value in model.state_dict().items():
                    finite(value, label + "/" + key)
            saved = super().save_model()
            c.require(saved is not False, "Native checkpoint saving failed")
            if self.best_fitness == self.fitness and self.best.exists():
                evidence["checkpoint_selection"] = {"rule": "native Ultralytics 8.4.124 best.pt; mAP50-95 fitness",
                    "epoch": int(self.epoch) + 1, "fitness": float(self.fitness)}
                c.write(run / "runtime_evidence.json", evidence)
            return saved

    def verify_loaded_labels(loader, split):
        records = c.data_records(req, job, split)
        wanted = {str(Path(r["image"]).resolve()): r for r in records}
        actual = [str(Path(p).resolve()) for p in loader.dataset.im_files]
        c.require(len(actual) == len(wanted) and set(actual) == set(wanted), "Loaded " + split + " membership changed")
        for label in loader.dataset.labels:
            path = str(Path(label["im_file"]).resolve())
            lines = Path(wanted[path]["label"]).read_text().splitlines()
            source = np.asarray([[float(v) for v in line.split()] for line in lines if line.strip()],
                                dtype=np.float32).reshape(-1, 5)
            source = np.unique(source, axis=0)
            cached = np.concatenate((label["cls"], label["bboxes"]), axis=1).astype(np.float32)
            c.require(len(source) == len(cached) and np.array_equal(source, np.unique(cached, axis=0)),
                      "Cached training/validation labels differ from frozen files: " + path)
        return len(actual)

    def on_start(trainer):
        c.verify_args(vars(trainer.args), expected, DEFAULT_CFG_DICT)
        c.verify_args(yaml.safe_load((run / "args.yaml").read_text()), expected, DEFAULT_CFG_DICT)
        c.require(trainer.amp is True, "AMP was disabled")
        c.require(any("checks passed" in msg for msg in log_messages)
                  and not any("checks skipped" in msg or "checks failed" in msg for msg in log_messages),
                  "A real native AMP check did not pass")
        c.require(type(trainer.optimizer).__name__ == "SGD", "Optimizer is not SGD")
        evidence["loaded_data"] = {"train": verify_loaded_labels(trainer.train_loader, "train"),
                                  "val": verify_loaded_labels(trainer.test_loader, "val")}
        evidence["actual_args"] = vars(trainer.args).copy()
        evidence["amp_check_log"] = log_messages.copy()
        evidence["started_utc"] = c.now()
        c.write(run / "runtime_evidence.json", evidence)

    def on_batch(trainer):
        finite(trainer.loss, "training loss")
        finite(trainer.loss_items, "training loss components")
        if "validated_historical_runtime" not in evidence:
            snapshot = trainer.runtime_evidence()
            ready = snapshot.get("optimizer_update_probe", {}).get("parameter_delta_l2", 0) > 0
            if job["arm"] == "E3":
                ready = ready and snapshot.get("weighted_positive_anchor_count", 0) > 0
            if ready:
                historical.validate_runtime_evidence(snapshot, choice)
                evidence["validated_historical_runtime"] = snapshot
                c.write(run / "runtime_evidence.json", evidence)

    def on_epoch(trainer):
        rows = c.metric_rows(run / "results.csv")
        if not rows:
            return
        epoch = int(float(rows[-1]["epoch"]))
        if int(trainer.epoch) + 1 != epoch:
            finite(trainer.metrics, "final best-checkpoint validation")
            return
        c.verify_args(vars(trainer.args), expected, DEFAULT_CFG_DICT)
        finite({key: float(value) for key, value in rows[-1].items()}, "epoch CSV")
        mdg, targets = trainer.consume_epoch_counts()
        if job["arm"] != "E0":
            c.require(mdg.get("images_total") == targets.get("images_total"), "MDG epoch image accounting mismatch")
            c.require(targets.get("clean_targets") == targets.get("degraded_targets"), "Dual-view epoch mismatch")
        progress = {"status": "training", "job": job, "epoch": epoch, "target_epochs": job["epochs"],
            "updated_utc": c.now(), "epoch_seconds": float(trainer.epoch_time), "row": rows[-1],
            "native_fitness": float(trainer.fitness), "native_best_fitness": float(trainer.best_fitness),
            "stopper_best_epoch": int(trainer.stopper.best_epoch), "stopper_patience": trainer.stopper.patience,
            "mdg_counts": mdg, "target_counts": targets}
        c.write(run / "progress.json", progress)
        evidence["last_epoch"] = progress
        c.write(run / "runtime_evidence.json", evidence)
        print("LEGACY_PROGRESS " + json.dumps({"job": job["key"], "epoch": epoch, "phase": phase}), flush=True)

    model = YOLO(req["pretrained_weight"]["path"])
    model.add_callback("on_train_start", on_start)
    model.add_callback("on_train_batch_end", on_batch)
    model.add_callback("on_fit_epoch_end", on_epoch)
    kwargs = {key: value for key, value in expected.items() if key != "save_dir"}
    try:
        model.train(trainer=ObservedTrainer, **kwargs)
    finally:
        LOGGER.removeHandler(handler)
    trainer = model.trainer
    rows = c.metric_rows(run / "results.csv")
    epochs = [int(float(row["epoch"])) for row in rows]
    c.require(epochs == list(range(1, len(rows) + 1)), "Epoch sequence is incomplete or duplicated")
    final_epoch = len(rows)
    early_stop = final_epoch < job["epochs"]
    c.require(final_epoch > 0 and final_epoch <= job["epochs"], "Invalid completed epoch count")
    if early_stop:
        c.require(final_epoch - trainer.stopper.best_epoch >= expected["patience"], "Unexplained early termination")
    c.require("validated_historical_runtime" in evidence and "finite_gradient_probe" in evidence,
              "Finite updates and historical semantics have not been demonstrated")
    c.require("checkpoint_selection" in evidence, "Actual best.pt selection was not recorded")
    c.verify_args(yaml.safe_load((run / "args.yaml").read_text()), expected, DEFAULT_CFG_DICT)
    for name in ("best.pt", "last.pt"):
        checkpoint = torch.load(run / "weights" / name, map_location="cpu", weights_only=False)
        weight_model = checkpoint.get("ema")
        weight_model = checkpoint.get("model") if weight_model is None else weight_model
        c.require(weight_model is not None, "Checkpoint contains no model")
        for key, value in weight_model.state_dict().items():
            finite(value, name + "/" + key)
    c.verify_release(req)
    c.require(c.sha(c.ROOT / "preflight.json") == preflight_sha, "Preflight changed during training")
    c.verify_data(req, job)
    evidence["total_mdg_counts"], evidence["total_target_counts"] = trainer.total_counts()
    evidence["completed_utc"] = c.now()
    c.write(run / "runtime_evidence.json", evidence)
    artifacts = ("args.yaml", "results.csv", "runtime_evidence.json", "weights/best.pt", "weights/last.pt")
    c.write(run / "completion.json", {"status": "completed", "job": job, "completed_utc": c.now(),
        "request_sha256": c.REQUEST_SHA, "adapter_manifest_sha256": adapter_sha, "epochs": final_epoch,
        "preflight_sha256": preflight_sha, "data_yaml_sha256": c.sha(job["data_yaml"]),
        "native_patience_early_stop": early_stop, "stopper_best_epoch": int(trainer.stopper.best_epoch),
        "checkpoint_selection": evidence["checkpoint_selection"], "artifacts": {p: c.sha(run / p) for p in artifacts}})
    print("LEGACY_TRAIN_COMPLETED " + job["key"], flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    try:
        main(args.job, args.phase)
    except BaseException:
        c.write(c.ROOT / "failures" / f"{args.phase}_{args.job}_training.json",
                {"status": "failed", "utc": c.now(), "traceback": traceback.format_exc()})
        raise
