"""Evaluate one frozen checkpoint under one condition, following the sealed unified evaluator's contract.

Differences from the sealed script, all forced by the local machine: images load from verified local
paths through np.fromfile + cv2.imdecode (cv2.imread cannot open non-ASCII Windows paths), and per-image
hashing happens once in prepare_local.py. Scoring, inference settings and transforms are unchanged.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import argparse, contextlib, gzip, importlib.metadata, io, math, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def now():
    return datetime.now(timezone.utc).isoformat()


def score(gt_dict, predictions, out):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        gt = COCO(); gt.dataset = gt_dict; gt.createIndex()
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, 'bbox'); ev.params.imgIds = sorted(gt.imgs); ev.params.catIds = sorted(gt.cats)
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    names = ['mAP50-95', 'mAP50', 'mAP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']
    metrics = {k: None if v < 0 else float(v) for k, v in zip(names, ev.stats)}
    per_class = {}
    for k, cat in enumerate(ev.params.catIds):
        v = ev.eval['precision'][:, :, k, 0, -1]; v = v[v >= 0]
        per_class[gt.cats[cat]['name']] = float(v.mean()) if len(v) else None
    (out / 'coco_eval.log').write_text(log.getvalue())
    np.savez_compressed(out / 'coco_arrays.npz', precision=ev.eval['precision'], recall=ev.eval['recall'], scores=ev.eval['scores'])
    return metrics, per_class


def make_transform(job):
    import evaluate_pretrained_sgd_robustness as robust
    require(sha(robust.__file__) == SEALED_TRANSFORM_SHA, 'Wrong transform source')
    c = job['condition']
    namespace = NAMESPACE_ROOT + '/' + job['dataset']
    if c in ('normal', 'low_light_strong', 'glare_strong'):
        conditions, params = robust.build_conditions({'controlled_degradation': {'source_config': str(MDG_CONFIG), 'severities': ['strong'], 'random_seed_namespace': namespace}})
        return conditions[c], params[c]
    s = SEVERITY[c]
    if s['kind'] == 'low':
        return robust.low_light_transform({'brightness': [s['brightness']] * 2, 'gamma': [s['gamma']] * 2}), s
    return robust.glare_transform(spec={'target_area_ratio': [s['area']] * 2, 'intensity_255': [s['intensity']] * 2}, glare_config=GLARE_GEOMETRY, namespace=namespace), s


def evaluate(job, limit=None):
    bootstrap(job['family'])
    import cv2, numpy as np, torch, ultralytics
    from ultralytics import YOLO
    import pycocotools.cocoeval as coco_module
    out = OUT / ('smoke' if limit else 'evaluations') / job['key']
    require(not (out / 'evaluation.json').exists(), 'Evaluation already complete ' + job['key'])
    out.mkdir(parents=True, exist_ok=True)
    require(sha(job['weights']) == job['weights_sha256'], 'Checkpoint changed')
    data = OUT / 'data' / job['dataset']
    records = read(data / 'local_records.json')
    if limit:
        records = records[:limit]
    model = YOLO(job['weights'])
    gt = read(data / 'ground_truth.json')
    names = {c['id'] - 1: c['name'] for c in gt['categories']}
    require(model.names == names, 'Model classes differ')
    torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    probe = dict(forward_calls=0)

    def before_forward(module, args):
        t = args[0]
        require(t.ndim == 4 and t.shape[0] == 1 and t.dtype == torch.float32 and tuple(t.shape[2:]) == (960, 960), 'Input contract mismatch')
        probe['forward_calls'] += 1

    def on_start(predictor):
        require(predictor.dataset.bs == 1 and not predictor.model.fp16 and predictor.args.rect is False, 'Predictor contract mismatch')
        if not getattr(predictor, '_guard', False):
            predictor.model.register_forward_pre_hook(before_forward); predictor._guard = True

    model.add_callback('on_predict_start', on_start)
    transform, parameters = make_transform(job)

    def load(rec):
        im = cv2.imdecode(np.fromfile(rec['local_image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        require(im is not None and im.shape[:2] == (rec['height'], rec['width']), 'Decode mismatch ' + rec['local_image'])
        return transform(im, Path(rec['image']))

    started = time.monotonic(); predictions = []; hasher = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {i: pool.submit(load, records[i]) for i in range(min(4, len(records)))}
        for index, rec in enumerate(records):
            arr = pending.pop(index).result()
            nxt = index + 4
            if nxt < len(records):
                pending[nxt] = pool.submit(load, records[nxt])
            hasher.update(str(rec['id']).encode()); hasher.update(arr.tobytes())
            results = list(model.predict(source=[arr], stream=True, verbose=False, save=False, **INFERENCE))
            require(len(results) == 1 and tuple(results[0].orig_shape) == (rec['height'], rec['width']), 'Output mismatch')
            boxes = results[0].boxes.data.detach().cpu().numpy()
            require(boxes.ndim == 2 and boxes.shape[1] == 6 and np.isfinite(boxes).all(), 'Nonfinite predictions')
            for x1, y1, x2, y2, conf, cl in boxes.tolist():
                predictions.append({'image_id': rec['id'], 'category_id': int(cl) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': conf})
            if (index + 1) % 200 == 0 or index + 1 == len(records):
                write(out / 'progress.json', dict(key=job['key'], images=index + 1, total=len(records), elapsed_seconds=time.monotonic() - started, utc=now()))
    require(probe['forward_calls'] == len(records), 'Batch probe failure')
    with gzip.open(out / 'predictions.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(predictions, f, allow_nan=False)
    if limit:
        keep = {r['id'] for r in records}
        gt['images'] = [i for i in gt['images'] if i['id'] in keep]
        gt['annotations'] = [a for a in gt['annotations'] if a['image_id'] in keep]
    metrics, per_class = score(gt, predictions, out)
    require(all(v is None or (math.isfinite(v) and 0 <= v <= 1) for v in metrics.values()), 'Invalid metrics')
    write(out / 'evaluation.json', dict(status='completed', utc=now(), job=job, images=len(records), ground_truth_boxes=len(gt['annotations']), predictions=len(predictions),
                                        input_pixels_sha256=hasher.hexdigest(), predictions_sha256=sha(out / 'predictions.json.gz'), metrics=metrics, per_class_AP=per_class,
                                        condition_parameters=parameters, inference=INFERENCE, evaluator_code_sha256=sha(coco_module.__file__),
                                        runtime=dict(ultralytics=ultralytics.__version__, ultralytics_file=ultralytics.__file__, torch=torch.__version__, pycocotools=importlib.metadata.version('pycocotools'), opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0)),
                                        elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(key=job['key'], metrics={k: metrics[k] for k in ('mAP50-95', 'APsmall')}, elapsed_seconds=round(time.monotonic() - started, 1), input=hasher.hexdigest()[:16], ultralytics=ultralytics.__version__)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--jobs', required=True); p.add_argument('--key', required=True); p.add_argument('--limit', type=int)
    a = p.parse_args()
    data = read(a.jobs)
    jobs = data['jobs'] if isinstance(data, dict) else data
    try:
        evaluate(next(j for j in jobs if j['key'] == a.key), a.limit)
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise
