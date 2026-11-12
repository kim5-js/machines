from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


def run_text(*args: str) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def gpu_status() -> dict[str, object]:
    raw = run_text(
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    if not raw:
        return {"available": False}
    fields = [part.strip() for part in raw.splitlines()[0].split(",")]
    if len(fields) != 6:
        return {"available": True, "raw": raw}
    name, utilization, memory_used, memory_total, temperature, power = fields
    return {
        "available": True,
        "name": name,
        "utilization_percent": float(utilization),
        "memory_used_mib": float(memory_used),
        "memory_total_mib": float(memory_total),
        "temperature_c": float(temperature),
        "power_w": float(power),
    }


def latest_training_row(root: Path) -> dict[str, object] | None:
    candidates = sorted(
        root.rglob("results.csv"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                return {
                    "path": str(path),
                    "modified_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "last_row": {
                        key.strip(): value.strip() if isinstance(value, str) else value
                        for key, value in rows[-1].items()
                    },
                }
        except (OSError, csv.Error, UnicodeError):
            continue
    return None


def latest_training_row_under(root: Path) -> dict[str, object] | None:
    if not root.is_dir():
        return None
    return latest_training_row(root)


def read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def formal_progress(
    training: dict[str, object] | None, queue: dict[str, object] | None
) -> dict[str, object] | None:
    if training is None or queue is None:
        return None
    row = training.get("last_row")
    if not isinstance(row, dict):
        return None
    try:
        epoch = int(float(str(row["epoch"])))
        elapsed = float(str(row["time"]))
        target = int(queue["target_epochs"])
        expected_per_epoch = float(queue["expected_seconds_per_epoch"])
        last_progress = datetime.fromisoformat(str(queue["last_progress_utc"]))
    except (KeyError, TypeError, ValueError):
        return None
    if epoch <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
        return None
    now = datetime.now(timezone.utc)
    observed_per_epoch = elapsed / epoch
    remaining = max(0, target - epoch) * observed_per_epoch
    stall_threshold = max(20 * 60, 4 * expected_per_epoch)
    return {
        "epoch": epoch,
        "target_epochs": target,
        "observed_seconds_per_epoch": round(observed_per_epoch, 3),
        "estimated_remaining_seconds": round(remaining, 3),
        "observed_rate_run_eta_utc": (now + timedelta(seconds=remaining)).isoformat(),
        "last_progress_age_seconds": round((now - last_progress).total_seconds(), 3),
        "stall_threshold_seconds": round(stall_threshold, 3),
        "stalled": (now - last_progress).total_seconds() > stall_threshold,
    }


def build_snapshot(root: Path) -> dict[str, object]:
    usage = shutil.disk_usage(root)
    formal_training = latest_training_row_under(root / "formal_baselines")
    formal_queue = read_json(root / "formal_state" / "baseline_queue_state.json")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_status(),
        "data_disk": {
            "path": str(root),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_percent": round(usage.used * 100 / usage.total, 2),
        },
        "screen_sessions": run_text("screen", "-list").splitlines(),
        "latest_training": latest_training_row(root),
        "formal_training": formal_training,
        "formal_queue": formal_queue,
        "formal_progress": formal_progress(formal_training, formal_queue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    snapshot = build_snapshot(args.root)
    print(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            separators=(",", ":") if args.compact else None,
            indent=None if args.compact else 2,
        )
    )


if __name__ == "__main__":
    main()
