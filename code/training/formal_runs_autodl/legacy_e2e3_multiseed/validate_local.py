"""CPU-only workflow regression checks. This does not validate real training or GPU behavior."""
import argparse
import ast
import copy
import io
import math
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

sys.dont_write_bytecode = True
import common as c
import evaluate
import supervise

REQ = None
SOURCE = Path(__file__).resolve().parent

class WorkflowChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ring3_legacy_audit_")
        self.root = Path(self.temp.name)
        self.patch = patch.object(c, "ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        target = Path(self.temp.name).resolve()
        if not (target == self.root.resolve() and target.is_relative_to(Path(tempfile.gettempdir()).resolve())
                and target.name.startswith("ring3_legacy_audit_")):
            raise RuntimeError("Refusing cleanup outside the allocated test directory")
        self.temp.cleanup()

    def expected(self):
        return c.expected_args(REQ, c.resolve_job(REQ, "WSODD_E2_seed0", "full"))

    def test_all_27_full_jobs_preserve_every_scientific_argument(self):
        jobs = c.job_list(REQ)
        self.assertEqual(len(jobs), 27)
        self.assertEqual(len({j["key"] for j in jobs}), 27)
        for item in jobs:
            job = c.resolve_job(REQ, item["key"], "full")
            expected = c.expected_args(REQ, job)
            changes = {key for key, val in expected.items() if val != REQ["training_reference_args"][key]}
            self.assertTrue(changes <= set(REQ["allowed_job_argument_differences"]))
            c.verify_args(dict(expected), expected)
        self.assertEqual([j["key"] for j in jobs[:3]], ["WSODD_E0_seed0", "WSODD_E2_seed0", "WSODD_E3_seed0"])
        self.assertEqual(jobs[-1]["key"], "SeaDronesSee_E3_seed2")

    def test_parameter_drift_is_rejected(self):
        expected = self.expected()
        bad = {"batch": 4, "imgsz": 640, "optimizer": "AdamW", "amp": False, "cls_remap": False,
               "pretrained": False, "nbs": 64, "lr0": 0.001, "box": 4, "dfl": 1.3,
               "weight_decay": 0.0005, "resume": True, "epochs": 100}
        for key, value in bad.items():
            with self.subTest(parameter=key), self.assertRaises(RuntimeError):
                c.verify_args(dict(expected, **{key: value}), expected)

    def test_missing_or_unexpected_arguments_are_rejected(self):
        expected = self.expected()
        incomplete = dict(expected)
        incomplete.pop("amp")
        with self.assertRaises(RuntimeError):
            c.verify_args(incomplete, expected)
        with self.assertRaises(RuntimeError):
            c.verify_args(dict(expected, unapproved_augmentation=True), expected)
        c.verify_args(dict(expected, augmentations=None), expected, {"augmentations": None})

    def test_smoke_changes_only_declared_training_length(self):
        for item in c.job_list(REQ)[:9]:
            job = c.resolve_job(REQ, item["key"], "smoke")
            expected = c.expected_args(REQ, job)
            differences = {key for key, value in expected.items() if value != REQ["training_reference_args"][key]}
            self.assertTrue(differences <= set(REQ["allowed_job_argument_differences"]) | {"epochs"})
            self.assertEqual(expected["epochs"], 2)
            self.assertEqual((expected["batch"], expected["imgsz"], expected["amp"]), (8, 960, True))
        with self.assertRaises(RuntimeError):
            c.resolve_job(REQ, "WSODD_E2_seed1", "smoke")

    def test_historical_e3_alpha_and_e2_definition(self):
        e2 = c.variant(c.resolve_job(REQ, "WSODD_E2_seed0", "full"))
        e3 = c.variant(c.resolve_job(REQ, "WSODD_E3_seed0", "full"))
        self.assertTrue(e2["dual_view"] and e3["dual_view"])
        self.assertFalse(e2["degraded_small_weighting"])
        self.assertTrue(e3["degraded_small_weighting"])
        self.assertEqual(e3["degraded_small_target_weight"], 1.5)

    def test_coco_overall_metrics_do_not_use_mixed_legacy_fields(self):
        raw = {"best_metrics": {key: 0.99 for key in c.METRICS},
               "area_evaluation": {"metrics": {"all": {"AP50": 0.65, "AP50_95": 0.35},
                "small": {"AP50_95": 0.12}, "medium": {"AP50_95": 0.31}, "large": {"AP50_95": 0.50}}}}
        output = evaluate.coco_metrics(raw)
        self.assertEqual(output["mAP50-95"], 0.35)
        self.assertEqual(output["mAP50"], 0.65)
        raw["area_evaluation"]["metrics"]["small"]["AP50_95"] = None
        self.assertIsNone(evaluate.coco_metrics(raw)["APsmall"])
        raw["area_evaluation"]["metrics"]["all"]["AP50_95"] = math.nan
        with self.assertRaises(RuntimeError):
            evaluate.coco_metrics(raw)

    def test_paired_statistics_use_matched_seeds_and_sample_sd(self):
        results = []
        deltas = [0.01, -0.02, 0.03]
        for seed in (0, 1, 2):
            for arm, value in (("E0", 0.3 + 0.1 * seed), ("E2", 0.3 + 0.1 * seed + deltas[seed])):
                results.append({"job": {"dataset": "WSODD", "seed": seed, "arm": arm},
                                "coco_metrics": {key: value for key in c.METRICS}})
        metric = c.paired_summary(results, REQ)["WSODD"]["E2-E0"]["mAP50-95"]
        self.assertAlmostEqual(metric["mean_pp"], statistics.mean([1, -2, 3]))
        self.assertAlmostEqual(metric["sample_sd_pp"], statistics.stdev([1, -2, 3]))
        self.assertEqual((metric["positive"], metric["negative"], metric["n"]), (2, 1, 3))
        self.assertTrue(metric["all_three_seeds"])
        results = [r for r in results if (r["job"]["seed"], r["job"]["arm"]) != (1, "E0")]
        metric = c.paired_summary(results, REQ)["WSODD"]["E2-E0"]["mAP50-95"]
        self.assertEqual([p["seed"] for p in metric["pairs"]], [0, 2])
        self.assertFalse(metric["all_three_seeds"])

    def test_oom_batch_reduction_is_blocked_before_rebuilding(self):
        tree = ast.parse((SOURCE / "train.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ObservedTrainer")
        class Base:
            def _build_train_pipeline(self):
                return "unchanged_pipeline"
        expected = self.expected()
        env = {"base": Base, "c": c, "expected": expected, "DEFAULT_CFG_DICT": {}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "observed_trainer", "exec"), env)
        trainer = env["ObservedTrainer"]()
        trainer.args, trainer.batch_size = SimpleNamespace(**expected), 8
        self.assertEqual(trainer._build_train_pipeline(), "unchanged_pipeline")
        trainer.args.batch, trainer.batch_size = 4, 4
        with self.assertRaises(RuntimeError):
            trainer._build_train_pipeline()

    def test_incomplete_run_is_preserved_without_retraining(self):
        job = c.resolve_job(REQ, "WSODD_E0_seed0", "smoke")
        run = Path(job["run_dir"])
        run.mkdir(parents=True)
        marker = run / "partial.txt"
        marker.write_text("preserve")
        with patch.object(supervise, "run_child") as launch:
            with self.assertRaises(RuntimeError):
                supervise.run_phase(REQ, "smoke", "adapter")
            launch.assert_not_called()
        self.assertEqual(marker.read_text(), "preserve")

    def test_completion_detects_weight_and_data_changes(self):
        job = c.resolve_job(REQ, "WSODD_E0_seed0", "smoke")
        run = Path(job["run_dir"])
        c.write(c.ROOT / "preflight.json", {"status": "passed"})
        c.write(job["data_yaml"], {"data": "fixture"})
        weight = run / "weights/best.pt"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"original checkpoint bytes")
        done = {"status": "completed", "job": job, "request_sha256": c.REQUEST_SHA,
            "adapter_manifest_sha256": "adapter", "preflight_sha256": c.sha(c.ROOT / "preflight.json"),
            "data_yaml_sha256": c.sha(job["data_yaml"]), "artifacts": {"weights/best.pt": c.sha(weight)}}
        c.write(run / "completion.json", done)
        c.verify_completion(REQ, job, "adapter")
        weight.write_bytes(b"changed checkpoint bytes")
        with self.assertRaises(RuntimeError):
            c.verify_completion(REQ, job, "adapter")
        weight.write_bytes(b"original checkpoint bytes")
        c.write(c.ROOT / "preflight.json", {"status": "changed"})
        with self.assertRaises(RuntimeError):
            c.verify_completion(REQ, job, "adapter")

    def test_full_queue_cannot_pass_without_smoke(self):
        with self.assertRaises(FileNotFoundError):
            supervise.verify_smoke_gate(REQ, "adapter")

    def smoke_fixture(self):
        c.write(c.ROOT / "preflight.json", {"status": "passed"})
        c.write(c.ROOT / "smoke_data/manifest.json", {"status": "fixture"})
        for item in c.job_list(REQ)[:9]:
            job = c.resolve_job(REQ, item["key"], "smoke")
            run = Path(job["run_dir"])
            c.write(run / "runtime_evidence.json", {"initialization": {"nc": len(c.dataset(REQ, job["dataset"])["names"]),
                "state_sha256": job["dataset"] + "-same-start"}, "finite_gradient_probe": {"finite": True},
                "semantic_batches_checked": 1, "validated_historical_runtime": {}})
            c.write(run / "completion.json", {"fixture": job["key"]})
            c.write(run / "evaluation.json", {"fixture": job["key"]})

    def test_smoke_gate_requires_nine_jobs_and_matching_initialization(self):
        self.smoke_fixture()
        with patch.object(c, "verify_evaluation"):
            with self.assertRaises(RuntimeError):
                supervise.write_smoke_gate(REQ, "adapter", [{}] * 8)
            supervise.write_smoke_gate(REQ, "adapter", [{}] * 9)
            supervise.verify_smoke_gate(REQ, "adapter")
            job = c.resolve_job(REQ, "FloW_E3_seed0", "smoke")
            path = Path(job["run_dir"]) / "runtime_evidence.json"
            runtime = c.read(path)
            runtime["initialization"]["state_sha256"] = "different-start"
            c.write(path, runtime)
            with self.assertRaises(RuntimeError):
                supervise.write_smoke_gate(REQ, "adapter", [{}] * 9)

    def test_reviewed_bundle_rejects_changed_adapter_even_with_rewritten_hash(self):
        req = copy.deepcopy(REQ)
        weight = self.root / "pretrained.pt"
        weight.write_bytes(b"pretrained fixture")
        req["pretrained_weight"] = {"path": str(weight), "sha256": c.sha(weight)}
        frozen = self.root / "frozen.py"
        frozen.write_text("frozen source")
        c.write(self.root / "staging_manifest.json", {"files": [{"path": "frozen.py", "sha256": c.sha(frozen)}]})
        code = self.root / "adapter/train.py"
        code.parent.mkdir()
        code.write_text("reviewed adapter")
        files = [{"path": "adapter/train.py", "sha256": c.sha(code)}]
        c.write(self.root / "execution_bundle_manifest.json", {"files": files})
        bundle_sha = c.sha(self.root / "execution_bundle_manifest.json")
        c.write(self.root / "adapter_manifest.json", {"files": files, "bundle_manifest_sha256": bundle_sha})
        c.write(self.root / "predecessor_gate.json", {"status": "passed", "first_stage_complete": True})
        c.write(self.root / "preflight.json", {"status": "passed", "request_sha256": c.REQUEST_SHA,
                "bundle_manifest_sha256": bundle_sha, "runtime_assets": []})
        with patch.object(c, "STAGING_SHA", c.sha(self.root / "staging_manifest.json")):
            c.verify_release(req)
            code.write_text("unreviewed adapter")
            with self.assertRaises(RuntimeError):
                c.verify_release(req)
            files[0]["sha256"] = c.sha(code)
            c.write(self.root / "adapter_manifest.json", {"files": files, "bundle_manifest_sha256": bundle_sha})
            with self.assertRaises(RuntimeError):
                c.verify_release(req)

    def test_stale_smoke_evidence_cannot_release_the_full_queue(self):
        self.smoke_fixture()
        with patch.object(c, "verify_evaluation"):
            supervise.write_smoke_gate(REQ, "adapter", [{}] * 9)
            path = Path(c.resolve_job(REQ, "WSODD_E2_seed0", "smoke")["run_dir"]) / "evaluation.json"
            c.write(path, {"changed": True})
            with self.assertRaises(RuntimeError):
                supervise.verify_smoke_gate(REQ, "adapter")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    REQ = c.read(args.request)
    c.require(c.sha(args.request) == c.REQUEST_SHA, "Test input is not the frozen request")
    syntax = {}
    for filename in ("common.py", "prepare.py", "train.py", "evaluate.py", "supervise.py", "validate_local.py"):
        path = SOURCE / filename
        if filename == "prepare.py" and not path.exists():
            path = SOURCE.parent / filename
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        syntax[filename] = {"sha256": c.sha(path), "syntax": "passed"}
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WorkflowChecks)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"status": "passed" if result.wasSuccessful() else "failed", "utc": c.now(),
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "request_sha256": c.REQUEST_SHA, "source_files": syntax, "output": stream.getvalue(),
        "validation_scope": "Local CPU workflow checks and source syntax only.",
        "real_data_gpu_smoke": "NOT RUN; upload approval is required before AutoDL validation.",
        "full_training": "NOT STARTED"}
    c.write(args.output, report)
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
