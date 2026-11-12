"""AutoDL version of the 20261109 recomputation. Inference and scoring only; no training.

Uses the original environment of the sealed 20260924 evaluation: mdc-yolo26 Python, the vendored
Ultralytics 8.4.124 and pycocotools 2.0.10 of flow_source_retrain_20260920, the sealed records and
ground truth, and the original checkpoints. Reads existing directories only; writes only under ROOT.

  python recompute_remote.py check                     verify code, data and every checkpoint hash
  python recompute_remote.py run --jobs F --workers N   evaluate all jobs in F, N processes in parallel
  python recompute_remote.py evaluate --jobs F --key K  one evaluation (used by run)
"""
from pathlib import Path
import argparse, contextlib, gzip, hashlib, importlib.metadata, io, json, math, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'recompute_20261109'
SEALED = BASE / 'manuscript_unified_eval_20260924'
FLOW = BASE / 'flow_source_retrain_20260920/release_v1'
SETUP = BASE / 'ring3_setup/autodl'
PYTHON = str(BASE / 'envs/mdc-yolo26/bin/python')
SEALED_TRANSFORM_SHA = 'cb42325d69011f24043170bbdee3b12371cd433d4dcf7d3fe611568a76c1794f'
SEALED_MDG_SHA = '41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a'
SEALED_COCOEVAL_SHA = '988485f79624cd42ceed9335d284df4cdcdf7b3d583e052fcbed2dd6bccf0267'
NAMESPACE_ROOT = 'ring3/cross-dataset/e0-e1/controlled-strong/v1'
INFERENCE = dict(imgsz=960, batch=1, half=False, rect=False, conf=0.001, iou=0.7, max_det=300, device=0)


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def require(ok, msg):
    if not ok: raise RuntimeError(msg)


def bootstrap():
    """Same import order as the sealed evaluator."""
    sys.dont_write_bytecode = True
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'settings')
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'mplconfig')
    sys.path[:0] = [str(FLOW / 'vendor'), str(FLOW / 'historical_code'), str(SETUP)]


def load_jobs(path):
    data = read(path)
    return data, (data['jobs'] if isinstance(data, dict) else data)


def check(jobs_path):
    bootstrap()
    import pycocotools.cocoeval as ce
    import evaluate_pretrained_sgd_robustness as robust
    report = dict(cocoeval=sha(ce.__file__) == SEALED_COCOEVAL_SHA, transform=sha(robust.__file__) == SEALED_TRANSFORM_SHA,
                  mdg=sha(SETUP / 'e1_mdg_config.json') == SEALED_MDG_SHA, sealed_prepared=(SEALED / 'prepared.json').exists())
    for p in sorted(Path(jobs_path).parent.glob('*jobs*.json')):
        _, jobs = load_jobs(p)
        seen = {}
        for j in jobs:
            if j['weights'] not in seen:
                seen[j['weights']] = Path(j['weights']).exists() and sha(j['weights']) == j['weights_sha256']
        report[p.name] = dict(jobs=len(jobs), checkpoints=len(seen), checkpoints_verified=sum(seen.values()), failed=[k for k, v in seen.items() if not v])
    write(ROOT / 'check.json', report)
    print(json.dumps(report, indent=1), flush=True)
    require(all(v for k, v in report.items() if isinstance(v, bool)), 'Environment check failed')
    require(all(not v['failed'] for k, v in report.items() if isinstance(v, dict)), 'Checkpoint check failed')


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


def make_transform(data, job):
    import evaluate_pretrained_sgd_robustness as robust
    require(sha(robust.__file__) == SEALED_TRANSFORM_SHA, 'Wrong transform source')
    c = job['condition']; namespace = NAMESPACE_ROOT + '/' + job['dataset']
    if c in ('normal', 'low_light_strong', 'glare_strong'):
        conditions, params = robust.build_conditions({'controlled_degradation': {'source_config': str(SETUP / 'e1_mdg_config.json'), 'severities': ['strong'], 'random_seed_namespace': namespace}})
        return conditions[c], params[c]
    s = data['severity'][c]
    if s['kind'] == 'low':
        return robust.low_light_transform({'brightness': [s['brightness']] * 2, 'gamma': [s['gamma']] * 2}), s
    return robust.glare_transform(spec={'target_area_ratio': [s['area']] * 2, 'intensity_255': [s['intensity']] * 2}, glare_config=data['glare_geometry'], namespace=namespace), s


def evaluate(jobs_path, key):
    data, jobs = load_jobs(jobs_path)
    job = next(j for j in jobs if j['key'] == key)
    bootstrap()
    import cv2, numpy as np, torch, ultralytics
    from ultralytics import YOLO
    import pycocotools.cocoeval as coco_module
    out = ROOT / 'evaluations' / key
    require(not (out / 'evaluation.json').exists(), 'Evaluation already complete ' + key)
    out.mkdir(parents=True, exist_ok=True)
    require(sha(job['weights']) == job['weights_sha256'], 'Checkpoint changed')
    sealed_data = SEALED / 'data' / job['dataset']
    records = read(sealed_data / 'records.json')
    gt = read(sealed_data / 'ground_truth.json')
    model = YOLO(job['weights'])
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
    transform, parameters = make_transform(data, job)

    def load(rec):
        require(sha(rec['image']) == rec['image_sha256'], 'Dataset bytes changed ' + rec['image'])
        im = cv2.imread(rec['image'], cv2.IMREAD_COLOR)
        require(im is not None and im.shape[:2] == (rec['height'], rec['width']), 'Decode mismatch')
        return transform(im, Path(rec['image']))

    started = time.monotonic(); predictions = []; hasher = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {i: pool.submit(load, records[i]) for i in range(min(4, len(records)))}
        for index, rec in enumerate(records):
            arr = pending.pop(index).result()
            nxt = index + 4
            if nxt < len(records): pending[nxt] = pool.submit(load, records[nxt])
            hasher.update(str(rec['id']).encode()); hasher.update(arr.tobytes())
            results = list(model.predict(source=[arr], stream=True, verbose=False, save=False, **INFERENCE))
            require(len(results) == 1 and tuple(results[0].orig_shape) == (rec['height'], rec['width']), 'Output mismatch')
            boxes = results[0].boxes.data.detach().cpu().numpy()
            require(boxes.ndim == 2 and boxes.shape[1] == 6 and np.isfinite(boxes).all(), 'Nonfinite predictions')
            for x1, y1, x2, y2, conf, cl in boxes.tolist():
                predictions.append({'image_id': rec['id'], 'category_id': int(cl) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': conf})
    require(probe['forward_calls'] == len(records), 'Batch probe failure')
    with gzip.open(out / 'predictions.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(predictions, f, allow_nan=False)
    metrics, per_class = score(gt, predictions, out)
    require(all(v is None or (math.isfinite(v) and 0 <= v <= 1) for v in metrics.values()), 'Invalid metrics')
    write(out / 'evaluation.json', dict(status='completed', utc=now(), job=job, images=len(records), ground_truth_boxes=len(gt['annotations']), predictions=len(predictions),
                                        input_pixels_sha256=hasher.hexdigest(), predictions_sha256=sha(out / 'predictions.json.gz'), metrics=metrics, per_class_AP=per_class,
                                        condition_parameters=parameters, inference=INFERENCE, evaluator_code_sha256=sha(coco_module.__file__),
                                        runtime=dict(ultralytics=ultralytics.__version__, ultralytics_file=ultralytics.__file__, torch=torch.__version__, pycocotools=importlib.metadata.version('pycocotools'),
                                                     opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0)), elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(key=key, metrics={k: metrics[k] for k in ('mAP50-95', 'APsmall')}, seconds=round(time.monotonic() - started, 1))), flush=True)


def run(jobs_path, workers):
    _, jobs = load_jobs(jobs_path)
    logs = ROOT / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    status = ROOT / ('status_' + Path(jobs_path).stem + '.json')
    todo = [j for j in jobs if not (ROOT / 'evaluations' / j['key'] / 'evaluation.json').exists()]
    started = time.monotonic(); running = {}; failures = {}; done = len(jobs) - len(todo)
    while todo or running:
        while todo and len(running) < workers:
            j = todo.pop(0)
            log = (logs / (j['key'] + '.log')).open('a')
            running[j['key']] = (subprocess.Popen([PYTHON, '-B', '-u', __file__, 'evaluate', '--jobs', str(jobs_path), '--key', j['key']], stdout=log, stderr=subprocess.STDOUT), log, j)
        time.sleep(5)
        for key in list(running):
            proc, log, j = running[key]
            if proc.poll() is None: continue
            log.close(); del running[key]
            if proc.returncode == 0: done += 1
            else:
                failures[key] = failures.get(key, 0) + 1
                if failures[key] < 2: todo.append(j)
        write(status, dict(state='running', done=done, total=len(jobs), running=sorted(running), failed={k: v for k, v in failures.items() if v >= 2},
                           elapsed_hours=round((time.monotonic() - started) / 3600, 3), utc=now()))
    bad = {k: v for k, v in failures.items() if v >= 2}
    write(status, dict(state='finished' if not bad else 'finished_with_failures', done=done, total=len(jobs), failed=bad, elapsed_hours=round((time.monotonic() - started) / 3600, 3), utc=now()))
    print(json.dumps(dict(done=done, total=len(jobs), failed=bad)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'run', 'evaluate'])
    p.add_argument('--jobs', default=str(ROOT / 'remote_jobs.json')); p.add_argument('--key'); p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    try:
        if a.mode == 'check': check(a.jobs)
        elif a.mode == 'run': run(a.jobs, a.workers)
        else: evaluate(a.jobs, a.key)
    except BaseException:
        print(traceback.format_exc(), flush=True); raise
