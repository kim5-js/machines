"""Traceable monitoring-only recovery; original release remains read-only."""
from pathlib import Path
import csv
import math
import sys

RECOVERY_CODE = Path(__file__).resolve().parent
ROOT = RECOVERY_CODE.parent
sys.path.insert(0, str(ROOT / "code"))
from common import load, sha

PROTOCOL_SHA = "6290fa48f4bf31b6c39973c76adc4af2fea52a094aecf6d791b5194769f97e92"


def verify_recovery(root):
    assert root.resolve() == ROOT.resolve()
    assert sha(root / "protocol.json") == PROTOCOL_SHA
    plan = load(RECOVERY_CODE / "recovery_plan.json")
    manifest = load(RECOVERY_CODE / "manifest.json")
    for relative, expected in manifest["files"].items():
        assert sha(RECOVERY_CODE / relative) == expected, relative
    for path, expected in plan["preserved_failed_files"].items():
        assert sha(root / path) == expected, path
    return plan


def resolve_run(root, key):
    plan = load(RECOVERY_CODE / "recovery_plan.json")
    relative = plan["run_overrides"].get(key, "runs/" + key)
    directory = (root / relative).resolve()
    assert directory.is_relative_to(root.resolve())
    return directory


def validate_csv(rows, permitted_epochs):
    for row in rows:
        for key, value in row.items():
            if math.isfinite(value):
                continue
            assert (key == "val/cls_loss" and value == math.inf
                    and int(row["epoch"]) in permitted_epochs), (row["epoch"], key, value)


def assert_replay_prefix(root, job, directory, epoch):
    plan = load(RECOVERY_CODE / "recovery_plan.json")
    if job["key"] not in plan["run_overrides"]:
        return None
    expected_path = root / "runs" / job["key"] / "results.csv"
    def rows(path):
        with path.open() as source:
            return [{k.strip(): v.strip() for k, v in r.items() if k.strip() != "time"}
                    for r in csv.DictReader(source)]
    expected = rows(expected_path)
    if epoch > len(expected):
        return True
    actual = rows(directory / "results.csv")
    assert actual[:epoch] == expected[:epoch], (
        "Replay differs before monitoring-only interruption", epoch, actual[:epoch], expected[:epoch])
    return epoch == len(expected)
