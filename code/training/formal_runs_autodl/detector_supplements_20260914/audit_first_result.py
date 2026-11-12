from pathlib import Path
import contextlib, gzip, hashlib, io, json, sys, time
from datetime import datetime, timezone
root = Path('D:/ring3/02_当前实验/原始证据/detector_supplements_20260914')
sys.path.insert(0, str(root / 'local_dependencies'))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
base = root / 'first_result_audit'
r = json.loads((base / 'result.json').read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base / 'ground_truth.json') == r['dataset_protocol']['ground_truth_sha256']
assert sha(base / 'predictions.json.gz') == r['predictions_sha256']
assert sha(base / 'coco_arrays.npz') == r['arrays_sha256']
with gzip.open(base / 'predictions.json.gz', 'rt', encoding='utf-8') as f:
    predictions = json.load(f)
log = io.StringIO()
start = time.perf_counter()
with contextlib.redirect_stdout(log):
    gt = COCO(str(base / 'ground_truth.json'))
    dt = gt.loadRes(predictions)
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
metrics = ['mAP50-95', 'mAP50', 'mAP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']
computed = {k: float(v) if v >= 0 else None for k, v in zip(metrics, evaluator.stats)}
for key in metrics:
    assert computed[key] == r['metrics'][key] or abs(computed[key] - r['metrics'][key]) < 1e-12, key
arrays = np.load(base / 'coco_arrays.npz')
array_check = {}
for key in ['precision', 'recall', 'scores']:
    delta = float(np.max(np.abs(evaluator.eval[key] - arrays[key])))
    assert delta < 1e-12, (key, delta)
    array_check[key] = {'maximum_absolute_difference': delta}
report = {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(), 'job': r['job']['key'], 'evaluation_images': len(gt.imgs), 'predictions': len(predictions), 'computed_metrics': computed, 'original_metrics': r['metrics'], 'array_checks': array_check, 'seconds': time.perf_counter() - start, 'method': 'Independent full COCOeval rerun on downloaded original-pixel GT and predictions using official pycocotools 2.0.10 in a separate Windows Python environment; no imported project evaluator code.'}
(base / 'independent_coco_recompute.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(base / 'independent_coco_recompute.log').write_text(log.getvalue(), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
