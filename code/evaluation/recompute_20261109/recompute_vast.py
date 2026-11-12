"""Vast version of the 20261109 recomputation. Inference and scoring only; no training.

Runs from the unpacked bundle root (the parent of this code/ folder). Mirrors the sealed 20260924
evaluator: same vendored Ultralytics per family, same transforms, same COCOeval call, same inference
contract. Only the file locations differ, and jobs are spread over all visible GPUs.

  python code/recompute_vast.py verify                         check every packed file, evaluator and environment
  python code/recompute_vast.py run --jobs gate_vast.json      reproduce five sealed evaluations
  python code/recompute_vast.py run --jobs jobs_vast.json --per-gpu 4
"""
from pathlib import Path
import argparse, contextlib, gzip, hashlib, importlib.metadata, io, json, math, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output'
SEALED_TRANSFORM_SHA = 'cb42325d69011f24043170bbdee3b12371cd433d4dcf7d3fe611568a76c1794f'
SEALED_MDG_SHA = '41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a'
SEALED_COCOEVAL_SHA = '988485f79624cd42ceed9335d284df4cdcdf7b3d583e052fcbed2dd6bccf0267'


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


def bootstrap(family='strategy'):
    """Same import order as the sealed evaluator; the matched family puts its own vendor first."""
    sys.dont_write_bytecode = True
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(OUTDIR / 'settings' / family)
    os.environ['MPLCONFIGDIR'] = str(OUTDIR / 'mplconfig')
    paths = [str(ROOT / 'code/flow_vendor'), str(ROOT / 'code/flow_historical_code'), str(ROOT / 'code/setup')]
    if family == 'matched':
        paths.insert(0, str(ROOT / 'code/matched_vendor'))
    sys.path[:0] = paths


def verify():
    manifest = read(ROOT / 'MANIFEST.json')
    bad = [k for k, v in manifest.items() if not (ROOT / k).exists() or sha(ROOT / k) != v['sha256']]
    report = dict(files=len(manifest), bad_files=bad[:20], bad_count=len(bad))
    probe = ("import importlib.metadata as m,json,sys,hashlib,torch,ultralytics,pycocotools.cocoeval as ce,cv2,numpy;"
             "print(json.dumps(dict(ultralytics=ultralytics.__version__,ultralytics_file=ultralytics.__file__,torch=torch.__version__,cuda=torch.version.cuda,"
             "gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],pycocotools=m.version('pycocotools'),"
             "cocoeval_sha=hashlib.sha256(open(ce.__file__,'rb').read()).hexdigest(),numpy=numpy.__version__,opencv=cv2.__version__)))")
    for family in ('strategy', 'matched'):
        code = f"import sys;sys.path.insert(0,{str(ROOT / 'code')!r});import recompute_vast as r;r.bootstrap({family!r});" + probe
        out = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        require(out.returncode == 0, out.stderr[-2000:])
        report[family] = json.loads(out.stdout.strip().splitlines()[-1])
    report['cocoeval_matches_sealed'] = report['strategy']['cocoeval_sha'] == SEALED_COCOEVAL_SHA
    report['vendors_expected'] = report['strategy']['ultralytics'] == '8.4.124' and report['matched']['ultralytics'] == '8.3.182'
    write(OUTDIR / 'verify.json', report)
    print(json.dumps(report, indent=1), flush=True)
    require(not bad and report['cocoeval_matches_sealed'] and report['vendors_expected'], 'Verification failed')


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


def make_transform(cfg, job):
    import evaluate_pretrained_sgd_robustness as robust
    require(sha(robust.__file__) == SEALED_TRANSFORM_SHA, 'Wrong transform source')
    c = job['condition']; namespace = cfg['namespace_root'] + '/' + job['dataset']
    if c in ('normal', 'low_light_strong', 'glare_strong'):
        conditions, params = robust.build_conditions({'controlled_degradation': {'source_config': str(ROOT / 'code/setup/e1_mdg_config.json'), 'severities': ['strong'], 'random_seed_namespace': namespace}})
        return conditions[c], params[c]
    s = cfg['severity'][c]
    if s['kind'] == 'low':
        return robust.low_light_transform({'brightness': [s['brightness']] * 2, 'gamma': [s['gamma']] * 2}), s
    return robust.glare_transform(spec={'target_area_ratio': [s['area']] * 2, 'intensity_255': [s['intensity']] * 2}, glare_config=cfg['glare_geometry'], namespace=namespace), s


def evaluate(jobs_path, key):
    cfg = read(jobs_path)
    job = next(j for j in cfg['jobs'] if j['key'] == key)
    bootstrap(job['family'])
    import cv2, numpy as np, torch, ultralytics
    from ultralytics import YOLO
    import pycocotools.cocoeval as coco_module
    out = OUTDIR / 'evaluations' / key
    require(not (out / 'evaluation.json').exists(), 'Evaluation already complete ' + key)
    out.mkdir(parents=True, exist_ok=True)
    weights = ROOT / job['weights']
    require(sha(weights) == job['weights_sha256'], 'Checkpoint changed')
    data = ROOT / 'data' / job['dataset']
    records = read(data / 'records.json'); gt = read(data / 'ground_truth.json')
    model = YOLO(str(weights))
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
    transform, parameters = make_transform(cfg, job)

    def load(rec):
        path = ROOT / 'images' / job['dataset'] / Path(rec['image']).name
        require(sha(path) == rec['image_sha256'], 'Dataset bytes changed ' + str(path))
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
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
            results = list(model.predict(source=[arr], stream=True, verbose=False, save=False, **cfg['inference']))
            require(len(results) == 1 and tuple(results[0].orig_shape) == (rec['height'], rec['width']), 'Output mismatch')
            boxes = results[0].boxes.data.detach().cpu().numpy()
            require(boxes.ndim == 2 and boxes.shape[1] == 6 and np.isfinite(boxes).all(), 'Nonfinite predictions')
            for x1, y1, x2, y2, conf, cl in boxes.tolist():
                require(int(cl) == cl and int(cl) in names and 0 <= conf <= 1 and x2 >= x1 and y2 >= y1, 'Invalid detection')
                predictions.append({'image_id': rec['id'], 'category_id': int(cl) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': conf})
    require(probe['forward_calls'] == len(records), 'Batch probe failure')
    with gzip.open(out / 'predictions.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(predictions, f, allow_nan=False)
    metrics, per_class = score(gt, predictions, out)
    require(all(v is None or (math.isfinite(v) and 0 <= v <= 1) for v in metrics.values()), 'Invalid metrics')
    write(out / 'evaluation.json', dict(status='completed', utc=now(), job=job, images=len(records), ground_truth_boxes=len(gt['annotations']), predictions=len(predictions),
                                        input_pixels_sha256=hasher.hexdigest(), predictions_sha256=sha(out / 'predictions.json.gz'), metrics=metrics, per_class_AP=per_class,
                                        condition_parameters=parameters, inference=cfg['inference'], evaluator_code_sha256=sha(coco_module.__file__),
                                        runtime=dict(ultralytics=ultralytics.__version__, ultralytics_file=ultralytics.__file__, torch=torch.__version__, pycocotools=importlib.metadata.version('pycocotools'),
                                                     opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                                        elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(key=key, metrics={k: metrics[k] for k in ('mAP50-95', 'APsmall')}, seconds=round(time.monotonic() - started, 1))), flush=True)


def run(jobs_path, per_gpu, threads):
    import torch
    gpus = torch.cuda.device_count(); require(gpus >= 1, 'No GPU visible')
    jobs = read(jobs_path)['jobs']
    logs = OUTDIR / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    status = OUTDIR / ('status_' + Path(jobs_path).stem + '.json')
    todo = [j for j in jobs if not (OUTDIR / 'evaluations' / j['key'] / 'evaluation.json').exists()]
    done = len(jobs) - len(todo); started = time.monotonic(); running = {}; failures = {}
    slots = [g for g in range(gpus) for _ in range(per_gpu)]
    while todo or running:
        busy = [v[3] for v in running.values()]
        free = list(slots)
        for b in busy: free.remove(b)
        while todo and free:
            j = todo.pop(0); gpu = free.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
            log = (logs / (j['key'] + '.log')).open('a')
            running[j['key']] = (subprocess.Popen([sys.executable, '-B', '-u', __file__, 'evaluate', '--jobs', str(jobs_path), '--key', j['key']], stdout=log, stderr=subprocess.STDOUT, env=env), log, j, gpu)
        time.sleep(5)
        for key in list(running):
            proc, log, j, gpu = running[key]
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


def gate_report():
    out = []
    for g in read(ROOT / 'gate_vast.json')['jobs']:
        e = read(OUTDIR / 'evaluations' / g['key'] / 'evaluation.json')
        d = {k: round(100 * (e['metrics'][k] - g['sealed'][k]), 4) for k in g['sealed']}
        out.append(dict(key=g['key'], pixels_identical=e['input_pixels_sha256'] == g['sealed_input_pixels_sha256'], predictions_identical=e['predictions_sha256'] == g['sealed_predictions_sha256'],
                        diff_pp=d, max_abs_diff_pp=max(abs(v) for v in d.values()), seconds=round(e['elapsed_seconds']), gpu=e['runtime']['gpu']))
    write(OUTDIR / 'gate_result.json', out)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['verify', 'run', 'evaluate', 'gate_report'])
    p.add_argument('--jobs', default=str(ROOT / 'jobs_vast.json')); p.add_argument('--key'); p.add_argument('--per-gpu', type=int, default=4); p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    try:
        if a.mode == 'verify': verify()
        elif a.mode == 'run': run(Path(a.jobs).resolve(), a.per_gpu, a.threads)
        elif a.mode == 'gate_report': gate_report()
        else: evaluate(a.jobs, a.key)
    except BaseException:
        print(traceback.format_exc(), flush=True); raise
