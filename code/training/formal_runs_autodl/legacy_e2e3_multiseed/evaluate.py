"""Evaluate the native best.pt through the unchanged historical COCO area evaluator."""
import argparse
import math
from pathlib import Path
import traceback
import common as c

def coco_metrics(raw):
    area = raw["area_evaluation"]["metrics"]
    metrics = {"mAP50": area["all"]["AP50"], "mAP50-95": area["all"]["AP50_95"],
               "APsmall": area["small"]["AP50_95"], "APmedium": area["medium"]["AP50_95"],
               "APlarge": area["large"]["AP50_95"]}
    for key, value in metrics.items():
        c.require(value is None or math.isfinite(value) and 0 <= value <= 1, "Invalid COCO metric: " + key)
    c.require(metrics["mAP50-95"] is not None, "Validation has no evaluable ground truths")
    return metrics

def main(job_key, phase):
    req = c.request()
    job = c.resolve_job(req, job_key, phase)
    adapter_sha = c.verify_release(req)
    done = c.verify_completion(req, job, adapter_sha)
    run = Path(job["run_dir"])
    c.require(not (run / "evaluation.json").exists() and not (run / "legacy_area_evaluation.json").exists(),
              "Existing evaluation preserved; inspect it before repeating")
    c.verify_data(req, job)
    versions = c.bootstrap(req)
    import torch
    from ultralytics import YOLO
    import evaluate_e3_coco_areas as evaluator
    import recover_wsodd_legacy_coco_areas as recovery
    import recover_wsodd_legacy_coco_areas_v2 as chunked

    contract = req["evaluation"]
    c.require(c.sha(Path(evaluator.__file__)) == contract["primary_evaluator_sha256"], "Evaluator source changed")
    c.require(c.sha(Path(chunked.__file__)) == contract["prediction_chunk_adapter_sha256"], "Chunk adapter changed")
    records = c.data_records(req, job, "val")
    images, data = evaluator.resolve_validation_images(Path(job["data_yaml"]))
    wanted = {str(Path(item["image"]).resolve()) for item in records}
    c.require(len(images) == len(wanted) and {str(p.resolve()) for p in images} == wanted,
              "Evaluator resolved different validation members")
    nc = len(c.dataset(req, job["dataset"])["names"])
    c.require(int(data["nc"]) == nc, "Wrong evaluation class count")
    # The only historical evaluator adaptations are dataset count/class assertions.
    recovery.EXPECTED_VALIDATION_IMAGES = len(records)
    recovery.EXPECTED_CLASSES = nc
    original_predict = YOLO.predict
    prediction_evidence = {"finite_images": 0, "finite_boxes": 0}

    def checked_predict(self, *args, **kwargs):
        c.require(kwargs.get("stream") is True, "Unexpected evaluation prediction route")
        for result in original_predict(self, *args, **kwargs):
            if result.boxes is not None:
                c.require(bool(torch.isfinite(result.boxes.data).all()), "Nonfinite prediction")
                cls = result.boxes.cls
                c.require(bool(((cls >= 0) & (cls < nc) & (cls == cls.round())).all()),
                          "Prediction class is outside the dataset")
                prediction_evidence["finite_boxes"] += len(result.boxes)
            prediction_evidence["finite_images"] += 1
            yield result

    YOLO.predict = checked_predict
    try:
        raw = chunked.evaluate_one_chunked(run_key=job["key"], run_dir=run,
            data_yaml=Path(job["data_yaml"]), output_path=run / "legacy_area_evaluation.json",
            imgsz=960, batch=8, device=0)
    finally:
        YOLO.predict = original_predict
    area = raw["area_evaluation"]["metrics"]
    metrics = coco_metrics(raw)
    c.require(prediction_evidence["finite_images"] == len(records), "Finite prediction coverage is incomplete")
    c.require(raw["artifacts"]["best_weights_sha256"] == done["artifacts"]["weights/best.pt"],
              "Frozen evaluator used another checkpoint")
    c.require(raw["artifacts"]["results_csv_sha256"] == done["artifacts"]["results.csv"], "Training CSV changed")
    rows = c.metric_rows(run / "results.csv")
    selected = next(row for row in rows if int(float(row["epoch"])) == done["checkpoint_selection"]["epoch"])
    c.verify_completion(req, job, adapter_sha)
    c.verify_release(req)
    c.verify_data(req, job)
    result = {"schema": "ring3.original_split_e0_e2_e3_coco.v1", "status": "completed", "job": job,
        "completed_utc": c.now(), "request_sha256": c.REQUEST_SHA, "adapter_manifest_sha256": adapter_sha,
        "completion_sha256": c.sha(run / "completion.json"),
        "checkpoint_sha256": done["artifacts"]["weights/best.pt"],
        "checkpoint_selection": done["checkpoint_selection"],
        "evaluation_split": "original_val" if phase == "full" else "smoke_original_val_subset",
        "evaluator": "frozen_project_COCO_style_area_evaluator", "evaluation_contract": contract["evaluation_contract"],
        "coco_metrics": metrics, "area_details": area, "runtime": versions,
        "data_yaml_sha256": c.sha(job["data_yaml"]),
        "validation_manifest_sha256": c.sha(c.ROOT / "verified_data" / job["dataset"] / "val.jsonl"),
        "validation_image_count": len(records), "prediction_evidence": prediction_evidence,
        "native_yolo": {"selected_checkpoint_epoch_row": selected, "last_row": rows[-1],
                        "note": "Native YOLO metrics are separate from every COCO-style metric above."},
        "raw_evaluation_sha256": c.sha(run / "legacy_area_evaluation.json"),
        "raw_legacy_metadata_note": "The raw historical report retains WSODD-specific labels and mixed best_metrics. "
                                   "Only its area_evaluation.metrics feed this authoritative dataset-specific report.",
        "evidence_scope": "Original validation split reproduction; not a sequence-disjoint generalization test."}
    c.write(run / "evaluation.json", result)
    print("LEGACY_EVALUATION_COMPLETED " + job["key"], flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    try:
        main(args.job, args.phase)
    except BaseException:
        c.write(c.ROOT / "failures" / f"{args.phase}_{args.job}_evaluation.json",
                {"status": "failed", "utc": c.now(), "traceback": traceback.format_exc()})
        raise
