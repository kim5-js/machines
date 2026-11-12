#!/usr/bin/env python3
"""Evaluation-only hotfix for the WSODD sequence-disjoint E3 alpha=1.5 run.

The locked test manifest contains image basenames.  The original one-checkpoint
evaluator resolved those basenames against the process working directory, which
made every image appear missing when the evaluator ran from ``/root``.  This
entrypoint preserves all locked training and evaluation inputs, patches only
the manifest-to-image-root resolver, evaluates the existing ``best.pt`` once,
and records the repair in the queue state and evidence files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", DATA_ROOT / "ring3_setup" / "autodl"))
STATE_ROOT = DATA_ROOT / "formal_state"
HOTFIX_MANIFEST = SETUP_ROOT / "wsodd_sequence_e3_alpha15_fixed_test_hotfix_v1_manifest.json"
STATE_PATH = STATE_ROOT / "wsodd_sequence_e3_alpha15_state.json"
EVENTS_PATH = STATE_ROOT / "wsodd_sequence_e3_alpha15_events.jsonl"
SUMMARY_PATH = STATE_ROOT / "wsodd_sequence_e3_alpha15_summary.json"
OUTPUT_PATH = STATE_ROOT / "wsodd_sequence_e3_alpha15_fixed_test_metrics.json"
EVIDENCE_PATH = STATE_ROOT / "wsodd_sequence_e3_alpha15_fixed_test_hotfix_v1_evidence.json"
EXPECTED_FAILURE = "Fixed test manifest has duplicates or missing images"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_event(event: str, **fields: Any) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": utc_now(), "event": event, **fields}
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_hotfix_release() -> dict[str, str]:
    if not HOTFIX_MANIFEST.is_file():
        raise RuntimeError(f"Hotfix manifest is missing: {HOTFIX_MANIFEST}")
    manifest = load_json(HOTFIX_MANIFEST)
    if manifest.get("schema") != "ring3.wsodd_sequence_e3_alpha15_fixed_test_hotfix_release.v1":
        raise RuntimeError(f"Unexpected hotfix manifest schema: {manifest.get('schema')}")
    observed: dict[str, str] = {}
    for section in ("locked_base_files", "hotfix_files"):
        files = manifest.get(section)
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"Hotfix manifest section is empty: {section}")
        for item in files:
            path = Path(item["path"])
            expected = str(item["sha256"])
            actual = sha256(path) if path.is_file() else "missing"
            if actual != expected:
                raise RuntimeError(
                    f"Hotfix release hash mismatch for {path}: expected {expected}, found {actual}"
                )
            observed[str(path)] = actual
    return observed


def test_image_root(config: dict[str, Any]) -> Path:
    dataset_yaml = Path(config["dataset"]["yaml"])
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Dataset YAML is not a mapping: {dataset_yaml}")
    root = Path(str(data.get("path", "")))
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    split = data.get("test")
    if not isinstance(split, str) or not split.strip():
        raise RuntimeError(f"Dataset YAML test split must be one directory: {split!r}")
    split_path = Path(split)
    if split_path.suffix.lower() == ".txt":
        raise RuntimeError(f"Unexpected nested test-list YAML entry: {split_path}")
    return (split_path if split_path.is_absolute() else root / split_path).resolve()


def resolve_test_images_v2(config: dict[str, Any]) -> list[Path]:
    dataset = config["dataset"]
    manifest = Path(dataset["test_manifest"])
    names = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_count = int(dataset["test_images"])
    if len(names) != expected_count:
        raise RuntimeError(f"Fixed test count changed: {len(names)} != {expected_count}")
    invalid = [name for name in names if Path(name).is_absolute() or len(Path(name).parts) != 1]
    if invalid:
        raise RuntimeError(f"Locked fixed-test manifest must contain basenames only: {invalid[:5]}")
    root = test_image_root(config)
    images = [root / name for name in names]
    duplicates = len(images) - len(set(images))
    missing = [str(path) for path in images if not path.is_file()]
    if duplicates or missing:
        raise RuntimeError(
            "Fixed-test resolver v2 failed: "
            f"duplicates={duplicates}, missing={len(missing)}, examples={missing[:5]}"
        )
    return images


def checkpoint_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required checkpoint is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def verify_training_completion(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    state = load_json(STATE_PATH)
    if state.get("status") != "evaluation_failed" or state.get("error") != EXPECTED_FAILURE:
        raise RuntimeError(
            "Hotfix only accepts the recorded path-resolution failure; "
            f"found status={state.get('status')!r}, error={state.get('error')!r}"
        )
    if int(state.get("current_epoch", 0)) != 150 or state.get("completed_runs") != ["e3"]:
        raise RuntimeError("The E3 training completion state is not the locked 150-epoch result")
    if state.get("e4_started") is not False:
        raise RuntimeError("E4 state is not false")
    run_name = str(config["run_order"][0]["name"])
    run_dir = Path(config["execution"]["output_root"]) / run_name
    completion_path = run_dir / "wsodd_sequence_e3_alpha15_completion_evidence.json"
    completion = load_json(completion_path)
    required = {
        "status": "passed",
        "completion_kind": "max_epochs",
        "normal_training_return_confirmed": True,
        "final_epoch": 150,
        "target_epochs": 150,
        "best_epoch": 108,
    }
    if any(completion.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"Unexpected E3 completion evidence: {completion}")
    for required_path in (run_dir / "results.csv", run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"):
        if not required_path.is_file():
            raise RuntimeError(f"Completed E3 artifact is missing: {required_path}")
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Fixed-test output already exists; refusing to overwrite: {OUTPUT_PATH}")
    return run_dir, state


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair only the WSODD E3 alpha=1.5 fixed-test path resolver")
    parser.add_argument(
        "--config",
        type=Path,
        default=SETUP_ROOT / "wsodd_sequence_e3_alpha15_config.json",
    )
    args = parser.parse_args()

    release_hashes = verify_hotfix_release()
    config = load_json(args.config)
    run_dir, failed_state = verify_training_completion(config)
    resolved_images = resolve_test_images_v2(config)
    resolver_evidence = {
        "manifest": str(config["dataset"]["test_manifest"]),
        "manifest_sha256": sha256(Path(config["dataset"]["test_manifest"])),
        "dataset_yaml": str(config["dataset"]["yaml"]),
        "dataset_yaml_sha256": sha256(Path(config["dataset"]["yaml"])),
        "resolved_test_image_root": str(test_image_root(config)),
        "image_count": len(resolved_images),
        "unique_image_count": len(set(resolved_images)),
        "missing_image_count": sum(not path.is_file() for path in resolved_images),
        "first_image": str(resolved_images[0]),
        "last_image": str(resolved_images[-1]),
    }
    if resolver_evidence["image_count"] != 2406 or resolver_evidence["missing_image_count"] != 0:
        raise RuntimeError(f"Resolver preflight failed: {resolver_evidence}")

    best_before = checkpoint_snapshot(run_dir / "weights" / "best.pt")
    last_before = checkpoint_snapshot(run_dir / "weights" / "last.pt")
    started_utc = utc_now()
    append_event(
        "fixed_test_hotfix_v1_started",
        evaluation_only=True,
        training_started=False,
        resolver=resolver_evidence,
    )

    import evaluate_wsodd_single_fixed_test as locked_evaluator

    locked_evaluator.resolve_test_images = resolve_test_images_v2
    report = locked_evaluator.run(args.config, run_dir, OUTPUT_PATH)

    best_after = checkpoint_snapshot(run_dir / "weights" / "best.pt")
    last_after = checkpoint_snapshot(run_dir / "weights" / "last.pt")
    if best_before != best_after or last_before != last_after:
        raise RuntimeError("Evaluation-only hotfix changed a training checkpoint")

    completed_utc = utc_now()
    hotfix_record = {
        "schema": "ring3.wsodd_sequence_e3_alpha15_fixed_test_hotfix_evidence.v1",
        "status": "passed",
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "failure_repaired": EXPECTED_FAILURE,
        "root_cause": "manifest basenames were resolved against process cwd instead of dataset YAML path plus test split",
        "scope": "fixed_test_path_resolution_only",
        "training_started": False,
        "training_results_preserved": True,
        "checkpoint_updates": False,
        "dataset_changed": False,
        "test_manifest_changed": False,
        "evaluation_parameters_changed": False,
        "e4_started": False,
        "resolver": resolver_evidence,
        "best_checkpoint_before_and_after": best_after,
        "last_checkpoint_before_and_after": last_after,
        "hotfix_manifest": {"path": str(HOTFIX_MANIFEST), "sha256": sha256(HOTFIX_MANIFEST)},
        "verified_release_hashes": release_hashes,
    }
    report["evaluation_only"] = True
    report["checkpoint_updates"] = False
    report["evaluation_hotfix"] = hotfix_record
    write_json_atomic(OUTPUT_PATH, report)
    write_json_atomic(EVIDENCE_PATH, hotfix_record)

    summary = load_json(SUMMARY_PATH)
    summary["fixed_test_metrics"] = report
    summary["post_training_validation_status"] = "completed"
    summary["evaluation_hotfix"] = {
        "schema": hotfix_record["schema"],
        "status": "passed",
        "evidence_path": str(EVIDENCE_PATH),
        "evidence_sha256": sha256(EVIDENCE_PATH),
    }
    write_json_atomic(SUMMARY_PATH, summary)

    state = load_json(STATE_PATH)
    if state.get("status") != "evaluation_failed":
        raise RuntimeError(f"Queue state changed during fixed-test repair: {state.get('status')}")
    failure_history = list(state.get("evaluation_failure_history", []))
    failure_history.append(
        {
            "failed_utc": state.get("failed_utc"),
            "error_type": state.get("error_type"),
            "error": state.get("error"),
            "repaired_by": hotfix_record["schema"],
        }
    )
    for key in ("failed_utc", "error_type", "error", "traceback"):
        state.pop(key, None)
    state.update(
        status="completed",
        post_training_validation_status="completed",
        fixed_test_metrics_path=str(OUTPUT_PATH),
        fixed_test_metrics_sha256=sha256(OUTPUT_PATH),
        fixed_test_hotfix_evidence_path=str(EVIDENCE_PATH),
        fixed_test_hotfix_evidence_sha256=sha256(EVIDENCE_PATH),
        summary_path=str(SUMMARY_PATH),
        summary_sha256=sha256(SUMMARY_PATH),
        evaluation_completed_utc=completed_utc,
        last_progress_utc=completed_utc,
        evaluation_failure_history=failure_history,
        e4_started=False,
    )
    write_json_atomic(STATE_PATH, state)
    append_event(
        "fixed_test_hotfix_v1_completed",
        evaluation_only=True,
        checkpoint_updates=False,
        fixed_test=report["fixed_test"],
        output_path=str(OUTPUT_PATH),
        output_sha256=sha256(OUTPUT_PATH),
        evidence_path=str(EVIDENCE_PATH),
        evidence_sha256=sha256(EVIDENCE_PATH),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
