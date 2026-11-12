from __future__ import annotations

import argparse
import csv
import importlib
import json
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import evaluate_pretrained_sgd_robustness as robust


CONFIG_SCHEMA = "ring3.wsodd_pretrained_sgd_validation_config.v1"
STATE_SCHEMA = "ring3.wsodd_pretrained_sgd_validation_state.v1"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, *, status: str, **details: Any) -> None:
    payload = {
        "schema": STATE_SCHEMA,
        "status": status,
        "updated_utc": utc_now(),
        "evaluation_only": True,
        "training_started": False,
        "checkpoint_updates": False,
        "e4_started": False,
        **details,
    }
    robust.write_json_atomic(path, payload)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_post_training_evaluation":
        raise RuntimeError(f"Unexpected WSODD validation config: {config.get('schema')!r}")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise RuntimeError("WSODD validation invariants are not all locked true")
    expected = {
        "split": "test",
        "images": 2406,
        "classes": 14,
    }
    if any(config.get("dataset", {}).get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"WSODD fixed test contract changed: {config.get('dataset')}")
    if config.get("scene_evaluation", {}).get("models") != ["E0", "E1", "E2", "E3"]:
        raise RuntimeError("WSODD scene-evaluation model order changed")
    if config.get("controlled_degradation", {}).get("models") != ["E0", "E2"]:
        raise RuntimeError("WSODD controlled-degradation model set changed")
    return config


def verify_static_assets(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "source_config": {"path": str(config_path), "sha256": robust.sha256(config_path)},
    }
    assets = {
        "dataset_yaml": (Path(config["dataset"]["yaml"]), config["dataset"]["yaml_sha256"]),
        "evaluator": (Path(config["evaluator"]["path"]), config["evaluator"]["sha256"]),
        "mdg_config": (
            Path(config["controlled_degradation"]["source_config"]),
            config["controlled_degradation"]["source_config_sha256"],
        ),
    }
    for key, (path, expected) in assets.items():
        if not path.is_file():
            raise RuntimeError(f"Required WSODD evaluation asset is missing: {path}")
        observed = robust.sha256(path)
        if observed != expected:
            raise RuntimeError(f"WSODD evaluation asset hash mismatch for {key}: {observed} != {expected}")
        checks[key] = {"path": str(path), "sha256": observed}
    return checks


def resolve_split_images(data_yaml: Path, split: str) -> tuple[list[Path], dict[str, Any]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    source = Path(data[split])
    source = source if source.is_absolute() else root / source
    if source.is_dir():
        images = sorted(
            (path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: str(path),
        )
    elif source.is_file():
        images = [
            Path(line.strip()) if Path(line.strip()).is_absolute() else root / line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise RuntimeError(f"WSODD {split} source does not exist: {source}")
    if len({path.name for path in images}) != len(images):
        raise RuntimeError("WSODD test image basenames are not unique")
    return images, data


def lock_completed_checkpoints(config: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    queue = config["training_queue"]
    state_path = Path(queue["state"])
    summary_path = Path(queue["summary"])
    if not state_path.is_file() or not summary_path.is_file():
        raise RuntimeError("WSODD training queue has not produced terminal state and summary artifacts")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if state.get("status") != queue["required_status"]:
        raise RuntimeError(f"WSODD training queue is not successfully completed: {state.get('status')!r}")
    if state.get("completed_runs") != queue["required_completed_runs"] or state.get("failed_runs"):
        raise RuntimeError(f"WSODD training queue completion set is invalid: {state}")
    if summary.get("e4_started") is not False:
        raise RuntimeError("E4 marker is not false in the WSODD training summary")
    for experiment, model in config["models"].items():
        run_dir = Path(model["run_dir"])
        best = run_dir / "weights" / "best.pt"
        results = run_dir / "results.csv"
        runtime = run_dir / "wsodd_pretrained_sgd_runtime_evidence.json"
        metadata = run_dir / "wsodd_pretrained_sgd_run_metadata.json"
        completion = run_dir / "wsodd_pretrained_sgd_completion_evidence.json"
        required = (best, results, runtime, metadata, completion)
        if any(not path.is_file() for path in required):
            raise RuntimeError(f"Completed WSODD artifacts are missing for {experiment}: {run_dir}")
        key = experiment.lower()
        metric_entry = summary.get("metrics", {}).get(key, {})
        if metric_entry.get("status") != "completed":
            raise RuntimeError(f"WSODD summary does not mark {experiment} completed")
        best_hash = robust.sha256(best)
        model["best_weights_sha256"] = best_hash
        checks[experiment] = {
            "run_dir": str(run_dir),
            "best_weights": str(best),
            "best_weights_sha256": best_hash,
            "results_csv_sha256": robust.sha256(results),
            "runtime_evidence_sha256": robust.sha256(runtime),
            "run_metadata_sha256": robust.sha256(metadata),
            "completion_evidence_sha256": robust.sha256(completion),
        }
    checks["training_queue_state"] = {"path": str(state_path), "sha256": robust.sha256(state_path)}
    checks["training_queue_summary"] = {"path": str(summary_path), "sha256": robust.sha256(summary_path)}
    runtime_config = Path(config["runtime_locked_config"])
    robust.write_json_atomic(runtime_config, config)
    checks["config_path"] = str(runtime_config)
    checks["config_sha256"] = robust.sha256(runtime_config)
    return checks


def import_checkpoint_modules() -> None:
    for name in (
        "cbam_focal_formal_models",
        "pretrained_sgd_mainline_models",
        "e3_small_object_weighting",
        "train_pretrained_sgd_alpha_ablation",
        "train_wsodd_pretrained_sgd_mainline",
    ):
        importlib.import_module(name)


def evaluate_scenes(
    *,
    config: dict[str, Any],
    checks: dict[str, Any],
    evaluator: Any,
    image_paths: list[Path],
    scene_labels: dict[str, str],
    label_audit: dict[str, Any],
    output_root: Path,
    self_tests: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    from ultralytics import YOLO

    prediction = {**config["evaluator"], "device": 0}
    class_ids = list(range(int(config["dataset"]["classes"])))
    reports: dict[str, Any] = {}
    for experiment in config["scene_evaluation"]["models"]:
        write_state(state_path, status="running", stage="scene_evaluation", experiment=experiment)
        output_path = output_root / "scene_metrics" / f"{experiment}.json"
        expected = {
            "config_sha256": checks["config_sha256"],
            "evaluator_sha256": checks["evaluator"]["sha256"],
            "dataset_yaml_sha256": checks["dataset_yaml"]["sha256"],
            "scene_labels_sha256": label_audit["sha256"],
            "best_weights_sha256": checks[experiment]["best_weights_sha256"],
        }
        cached = robust.valid_cached(output_path, expected)
        if cached is not None:
            print(f"[wsodd-scene:{experiment}] reusing {output_path}", flush=True)
            reports[experiment] = cached
            continue
        print(f"[wsodd-scene:{experiment}] predicting fixed sequence-disjoint test", flush=True)
        model = YOLO(checks[experiment]["best_weights"])
        if experiment == "E0":
            self_tests["identity_source_equivalence"] = robust.identity_source_equivalence(model, image_paths, prediction)
        detections, ground_truths, counts = robust.predict_dataset(
            model=model,
            image_paths=image_paths,
            transform=lambda image, _: image.copy(),
            evaluator=evaluator,
            prediction=prediction,
        )
        full_metrics = robust.evaluate_metrics(
            evaluator, detections, ground_truths, class_ids, int(prediction["evaluation_max_det"])
        )
        scenes: dict[str, Any] = {}
        for scene in config["scene_evaluation"]["allowed_labels"]:
            names = [path.name for path in image_paths if scene_labels[path.name] == scene]
            subset_detections = {name: detections[name] for name in names}
            subset_ground_truths = {name: ground_truths[name] for name in names}
            metrics = robust.evaluate_metrics(
                evaluator,
                subset_detections,
                subset_ground_truths,
                class_ids,
                int(prediction["evaluation_max_det"]),
            )
            scenes[scene] = {
                "image_count": len(names),
                "prediction_count": sum(len(value) for value in subset_detections.values()),
                "summary": robust.metric_summary(metrics),
                "metrics": metrics,
            }
        report = {
            "schema": "ring3.wsodd_pretrained_sgd_scene_robustness.v1",
            "status": "completed",
            "experiment": experiment,
            "condition": "original_locked_sequence_disjoint_test",
            "fingerprints": expected,
            "prediction_parameters": prediction,
            "counts": counts,
            "full_summary": robust.metric_summary(full_metrics),
            "full_metrics": full_metrics,
            "scenes": scenes,
            "single_seed_mechanism_evidence_only": True,
        }
        robust.write_json_atomic(output_path, report)
        reports[experiment] = report
        robust.release_gpu(model)
    return reports


def evaluate_degradations(
    *,
    config: dict[str, Any],
    checks: dict[str, Any],
    evaluator: Any,
    image_paths: list[Path],
    output_root: Path,
    scene_reports: dict[str, Any],
    conditions: dict[str, robust.Transform],
    condition_parameters: dict[str, Any],
    state_path: Path,
) -> dict[str, dict[str, Any]]:
    from ultralytics import YOLO

    prediction = {**config["evaluator"], "device": 0}
    class_ids = list(range(int(config["dataset"]["classes"])))
    reports: dict[str, dict[str, Any]] = defaultdict(dict)
    for experiment in config["controlled_degradation"]["models"]:
        normal_summary = scene_reports[experiment]["full_summary"]
        normal_report = {
            "schema": "ring3.wsodd_pretrained_sgd_controlled_degradation.v1",
            "status": "completed",
            "experiment": experiment,
            "condition": "normal",
            "family": "normal",
            "severity": "normal",
            "parameters": condition_parameters["normal"],
            "fingerprints": {
                "config_sha256": checks["config_sha256"],
                "evaluator_sha256": checks["evaluator"]["sha256"],
                "dataset_yaml_sha256": checks["dataset_yaml"]["sha256"],
                "best_weights_sha256": checks[experiment]["best_weights_sha256"],
                "source_scene_report_sha256": robust.sha256(output_root / "scene_metrics" / f"{experiment}.json"),
            },
            "prediction_parameters": prediction,
            "counts": scene_reports[experiment]["counts"],
            "summary": normal_summary,
            "metrics": scene_reports[experiment]["full_metrics"],
        }
        normal_path = output_root / "degradation_metrics" / experiment / "normal.json"
        robust.write_json_atomic(normal_path, normal_report)
        reports[experiment]["normal"] = normal_report
        model = YOLO(checks[experiment]["best_weights"])
        for family in config["controlled_degradation"]["families"]:
            for severity in config["controlled_degradation"]["severities"]:
                condition = f"{family}_{severity}"
                write_state(
                    state_path,
                    status="running",
                    stage="controlled_degradation",
                    experiment=experiment,
                    condition=condition,
                )
                output_path = output_root / "degradation_metrics" / experiment / f"{condition}.json"
                expected = {
                    "config_sha256": checks["config_sha256"],
                    "evaluator_sha256": checks["evaluator"]["sha256"],
                    "dataset_yaml_sha256": checks["dataset_yaml"]["sha256"],
                    "mdg_config_sha256": checks["mdg_config"]["sha256"],
                    "best_weights_sha256": checks[experiment]["best_weights_sha256"],
                }
                cached = robust.valid_cached(output_path, expected)
                if cached is not None:
                    print(f"[wsodd-degradation:{experiment}:{condition}] reusing {output_path}", flush=True)
                    reports[experiment][condition] = cached
                    continue
                print(f"[wsodd-degradation:{experiment}:{condition}] predicting", flush=True)
                detections, ground_truths, counts = robust.predict_dataset(
                    model=model,
                    image_paths=image_paths,
                    transform=conditions[condition],
                    evaluator=evaluator,
                    prediction=prediction,
                )
                metrics = robust.evaluate_metrics(
                    evaluator,
                    detections,
                    ground_truths,
                    class_ids,
                    int(prediction["evaluation_max_det"]),
                )
                metric_summary = robust.metric_summary(metrics)
                report = {
                    "schema": "ring3.wsodd_pretrained_sgd_controlled_degradation.v1",
                    "status": "completed",
                    "experiment": experiment,
                    "condition": condition,
                    "family": family,
                    "severity": severity,
                    "parameters": condition_parameters[condition],
                    "fingerprints": expected,
                    "prediction_parameters": prediction,
                    "counts": counts,
                    "summary": metric_summary,
                    "drop_from_normal": {
                        key: float(normal_summary[key]) - float(metric_summary[key])
                        for key in ("mAP50-95", "APsmall")
                    },
                    "metrics": metrics,
                }
                robust.write_json_atomic(output_path, report)
                reports[experiment][condition] = report
                del detections, ground_truths
                robust.release_gpu()
        robust.release_gpu(model)
    return dict(reports)


def write_summaries(
    *,
    output_root: Path,
    config: dict[str, Any],
    checks: dict[str, Any],
    label_audit: dict[str, Any],
    self_tests: dict[str, Any],
    scene_reports: dict[str, Any],
    degradation_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    area_rows = [
        {"experiment": experiment, **scene_reports[experiment]["full_summary"]}
        for experiment in config["scene_evaluation"]["models"]
    ]
    scene_rows: list[dict[str, Any]] = []
    for experiment in config["scene_evaluation"]["models"]:
        for scene in config["scene_evaluation"]["allowed_labels"]:
            item = scene_reports[experiment]["scenes"][scene]
            scene_rows.append(
                {
                    "experiment": experiment,
                    "scene": scene,
                    "image_count": item["image_count"],
                    "ground_truth_count": item["summary"]["ground_truth_count"],
                    "small_ground_truth_count": item["summary"]["small_ground_truth_count"],
                    "mAP50-95": item["summary"]["mAP50-95"],
                    "APsmall": item["summary"]["APsmall"],
                }
            )
    degradation_rows: list[dict[str, Any]] = []
    for experiment in config["controlled_degradation"]["models"]:
        normal = degradation_reports[experiment]["normal"]["summary"]
        for family in config["controlled_degradation"]["families"]:
            for level, severity in enumerate(("normal", *config["controlled_degradation"]["severities"])):
                report = degradation_reports[experiment]["normal" if severity == "normal" else f"{family}_{severity}"]
                item = report["summary"]
                degradation_rows.append(
                    {
                        "experiment": experiment,
                        "family": family,
                        "severity": severity,
                        "level": level,
                        "mAP50-95": item["mAP50-95"],
                        "APsmall": item["APsmall"],
                        "mAP50-95_drop": float(normal["mAP50-95"]) - float(item["mAP50-95"]),
                        "APsmall_drop": float(normal["APsmall"]) - float(item["APsmall"]),
                    }
                )
    output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("area_metrics.csv", area_rows),
        ("scene_metrics.csv", scene_rows),
        ("degradation_curves.csv", degradation_rows),
    ):
        with (output_root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema": "ring3.wsodd_pretrained_sgd_validation_summary.v1",
        "status": "completed",
        "completed_utc": utc_now(),
        "evaluation_only": True,
        "training_started": False,
        "checkpoint_updates": False,
        "e4_started": False,
        "fixed_split": "test",
        "fingerprints": checks,
        "scene_labels": label_audit,
        "self_tests": self_tests,
        "prediction_parameters": config["evaluator"],
        "area_rows": area_rows,
        "scene_rows": scene_rows,
        "degradation_rows": degradation_rows,
        "single_seed_mechanism_evidence_only": True,
    }
    robust.write_json_atomic(output_root / "wsodd_validation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-test WSODD area, scene, and controlled-degradation evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--static-preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    state_path = Path(config["state"])
    try:
        checks = verify_static_assets(config, args.config)
        evaluator = robust.load_evaluator(Path(config["evaluator"]["path"]))
        conditions, condition_parameters = robust.build_conditions(config)
        self_tests: dict[str, Any] = {
            "evaluator": evaluator.synthetic_self_test(),
            "controlled_transforms": robust.transform_self_test(conditions, condition_parameters),
        }
        if args.static_preflight_only:
            print(json.dumps({"status": "passed", "checks": checks, "self_tests": self_tests}, ensure_ascii=False, indent=2))
            return 0

        checks = lock_completed_checkpoints(config, checks)
        image_paths, data = resolve_split_images(Path(config["dataset"]["yaml"]), config["dataset"]["split"])
        if len(image_paths) != int(config["dataset"]["images"]) or int(data["nc"]) != int(config["dataset"]["classes"]):
            raise RuntimeError(f"Locked WSODD test contract mismatch: images={len(image_paths)} nc={data['nc']}")
        labels_path = Path(config["scene_evaluation"]["labels_csv"])
        if not labels_path.is_file():
            write_state(
                state_path,
                status="awaiting_manual_scene_labels",
                stage="scene_labeling",
                labels_csv=str(labels_path),
                expected_images=len(image_paths),
            )
            print(f"WSODD validation is awaiting manually reviewed scene labels: {labels_path}", flush=True)
            return 4
        scene_labels, label_audit = robust.load_scene_labels(
            labels_path,
            {path.name for path in image_paths},
            set(config["scene_evaluation"]["allowed_labels"]),
        )
        import_checkpoint_modules()
        output_root = Path(config["output"])
        scene_reports = evaluate_scenes(
            config=config,
            checks=checks,
            evaluator=evaluator,
            image_paths=image_paths,
            scene_labels=scene_labels,
            label_audit=label_audit,
            output_root=output_root,
            self_tests=self_tests,
            state_path=state_path,
        )
        degradation_reports = evaluate_degradations(
            config=config,
            checks=checks,
            evaluator=evaluator,
            image_paths=image_paths,
            output_root=output_root,
            scene_reports=scene_reports,
            conditions=conditions,
            condition_parameters=condition_parameters,
            state_path=state_path,
        )
        summary = write_summaries(
            output_root=output_root,
            config=config,
            checks=checks,
            label_audit=label_audit,
            self_tests=self_tests,
            scene_reports=scene_reports,
            degradation_reports=degradation_reports,
        )
        robust.plot_curves(output_root, summary["degradation_rows"])
        write_state(
            state_path,
            status="completed",
            stage="completed",
            summary=str(output_root / "wsodd_validation_summary.json"),
            summary_sha256=robust.sha256(output_root / "wsodd_validation_summary.json"),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        write_state(
            state_path,
            status="failed",
            stage="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
