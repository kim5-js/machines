"""Inference-only evaluation of one frozen existing checkpoint with official COCOeval."""
from pathlib import Path
import contextlib
import gzip
import importlib.metadata
import io
import json
import os
import sys
import time
import traceback
from prepare import ROOT, now, sha, write

sys.path.insert(0, str(ROOT / 'vendor'))
os.environ.setdefault('YOLO_CONFIG_DIR', '/root/autodl-tmp/.config/Ultralytics')
os.environ.setdefault('MPLCONFIGDIR', '/root/autodl-tmp/.cache/matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def coco_evaluate(gt_path, predictions, output):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gt = COCO(str(gt_path))
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {'images': gt.dataset['images'], 'categories': gt.dataset['categories'], 'annotations': []}
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(gt.imgs)
        ev.params.catIds = sorted(gt.cats)
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'coco_eval.log').write_text(out.getvalue())
        np.savez_compressed(output / 'coco_arrays.npz', precision=ev.eval['precision'], recall=ev.eval['recall'], scores=ev.eval['scores'])
    names = ['mAP50-95', 'mAP50', 'mAP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']
    metrics = {key: None if value < 0 else float(value) for key, value in zip(names, ev.stats)}
    per_class = {}
    for k, cat in enumerate(ev.params.catIds):
        values = ev.eval['precision'][:, :, k, 0, -1]
        values = values[values >= 0]
        per_class[gt.cats[cat]['name']] = float(values.mean()) if len(values) else None
    return metrics, per_class


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT) as td:
        p = Path(td) / 'gt.json'
        gt = {'info': {}, 'images': [{'id': 1, 'width': 200, 'height': 200}], 'categories': [{'id': 1, 'name': 'target'}, {'id': 2, 'name': 'absent'}],
              'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [5, 5, 20, 20], 'area': 400, 'iscrowd': 0}]}
        write(p, gt)
        perfect, classes = coco_evaluate(p, [{'image_id': 1, 'category_id': 1, 'bbox': [5, 5, 20, 20], 'score': 0.9}], None)
        empty, _ = coco_evaluate(p, [], None)
        wrong, _ = coco_evaluate(p, [{'image_id': 1, 'category_id': 2, 'bbox': [5, 5, 20, 20], 'score': 0.9}], None)
        assert perfect['mAP50-95'] > .99999 and perfect['APsmall'] > .99999
        assert perfect['APmedium'] is None and classes['absent'] is None
        assert empty['mAP50-95'] == 0 and wrong['mAP50-95'] == 0
    return {'status': 'passed', 'checks': ['perfect detection', 'missing area remains null', 'absent class remains null', 'empty detections', 'wrong category']}


def evaluate(key):
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO, RTDETR
    assert importlib.metadata.version('pycocotools') == '2.0.10'
    req = json.loads((ROOT / 'request.json').read_text())
    assert sha(ROOT / 'request.json') == (ROOT / 'request.sha256').read_text().strip()
    for name, digest in req['scripts'].items():
        assert sha(ROOT / 'code' / name) == digest, 'Code drift'
    job = next(j for j in req['jobs'] if j['key'] == key)
    ds = req['datasets'][job['dataset']]
    out = ROOT / 'evaluations' / key
    out.mkdir(parents=True, exist_ok=False)
    assert sha(job['weights']) == job['weights_sha256']
    assert sha(ds['ground_truth']) == ds['ground_truth_sha256'] and sha(ds['image_manifest']) == ds['image_manifest_sha256']
    records = json.loads(Path(ds['image_manifest']).read_text())
    for item in records:
        assert sha(item['path']) == item['image_sha256'] and sha(item['label']) == item['label_sha256']
    probe = self_test()
    model = (RTDETR if job['model'] == 'rtdetr_l' else YOLO)(job['weights'])
    assert {str(k): v for k, v in model.names.items()} == ds['classes']
    torch.manual_seed(0)
    np.random.seed(0)
    start = time.monotonic()
    predictions = []
    processed = []
    last_progress = 0
    # Path lists bypass Ultralytics batch=; one path per call enforces the frozen batch=1.
    runtime_batch = {'requested_batch': 1, 'loader_batch_sizes': [], 'forward_calls': 0,
                     'maximum_forward_batch': 0, 'input_dtypes': [], 'input_shapes': []}
    def before_forward(module, args):
        tensor = args[0]
        assert tensor.ndim == 4 and tensor.shape[0] == 1, 'Actual forward batch must equal one'
        assert tensor.dtype == torch.float32, 'Actual input must be FP32'
        runtime_batch['forward_calls'] += 1
        runtime_batch['maximum_forward_batch'] = max(runtime_batch['maximum_forward_batch'], int(tensor.shape[0]))
        dtype = str(tensor.dtype)
        if dtype not in runtime_batch['input_dtypes']: runtime_batch['input_dtypes'].append(dtype)
        shape = list(tensor.shape)
        if shape not in runtime_batch['input_shapes']: runtime_batch['input_shapes'].append(shape)
    def verify_predictor(predictor):
        assert predictor.dataset.bs == 1 and not predictor.model.fp16
        assert predictor.args.quantize is None
        if predictor.dataset.bs not in runtime_batch['loader_batch_sizes']:
            runtime_batch['loader_batch_sizes'].append(predictor.dataset.bs)
        if not getattr(predictor, '_supplement_batch_guard_attached', False):
            predictor.model.register_forward_pre_hook(before_forward)
            predictor._supplement_batch_guard_attached = True
    model.add_callback('on_predict_start', verify_predictor)
    for offset in range(0, len(records), 1):
        chunk = records[offset:offset + 1]
        results = model.predict(source=[r['path'] for r in chunk], stream=True, verbose=False, save=False, **req['inference'])
        for item, result in zip(chunk, results, strict=True):
            assert Path(result.path) == Path(item['path']) and tuple(result.orig_shape) == (item['height'], item['width'])
            processed.append(item['id'])
            values = result.boxes.data.detach().cpu().numpy()
            assert np.isfinite(values).all() and values.shape[1] == 6
            for x1, y1, x2, y2, score, cl in values.tolist():
                assert cl.is_integer() and str(int(cl)) in ds['classes'] and 0 <= score <= 1
                assert x2 >= x1 and y2 >= y1
                predictions.append({'image_id': item['id'], 'category_id': int(cl) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': score})
        if len(processed) - last_progress >= 100 or len(processed) == len(records):
            write(out / 'progress.json', {'job': key, 'status': 'inference', 'images': len(processed), 'target_images': len(records), 'predictions': len(predictions), 'updated_utc': now(), 'seconds': time.monotonic() - start, 'actual_batch_probe': runtime_batch})
            last_progress = len(processed)
    assert processed == [r['id'] for r in records], 'Image coverage'
    assert runtime_batch['forward_calls'] == len(records), 'Each evaluation image must have one guarded forward pass'
    assert runtime_batch['maximum_forward_batch'] == 1 and runtime_batch['input_dtypes'] == ['torch.float32']
    with gzip.open(out / 'predictions.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(predictions, f, allow_nan=False)
    inference_seconds = time.monotonic() - start
    metrics, per_class = coco_evaluate(ds['ground_truth'], predictions, out)
    assert sha(job['weights']) == job['weights_sha256'], 'Checkpoint changed'
    write(out / 'result.json', {'status': 'completed', 'record_type': 'unified_detector_reevaluation', 'completed_utc': now(), 'job': job,
          'request_sha256': sha(ROOT / 'request.json'), 'evaluator': req['evaluator'], 'inference': req['inference'],
          'evaluation_split': ds['evaluation_split'], 'evaluation_images': len(processed), 'dataset_protocol': ds,
          'metrics': metrics, 'per_class_AP': per_class, 'predictions': len(predictions), 'predictions_sha256': sha(out / 'predictions.json.gz'),
          'arrays_sha256': sha(out / 'coco_arrays.npz'), 'inference_seconds': inference_seconds, 'total_seconds': time.monotonic() - start,
          'self_test': probe, 'training_performed': False, 'actual_batch_probe': runtime_batch,
          'preprocessing': 'imgsz=960, rect=False; native architecture predictor (YOLO letterbox, RT-DETR stretch); YOLO26 and RT-DETR retain native end-to-end postprocessing'})
    print(json.dumps({'job': key, 'status': 'completed', 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    try:
        evaluate(sys.argv[1])
    except BaseException:
        write(ROOT / 'failures' / (sys.argv[1] + '.json'), {'utc': now(), 'traceback': traceback.format_exc()})
        raise
