"""Freeze existing detector weights and fixed evaluation sets; never train."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(os.environ.get('SUPPLEMENT_ROOT', '/root/autodl-tmp/detector_supplements_20260914'))
BASE = Path('/root/autodl-tmp')
SETUP = BASE / 'ring3_setup/autodl'
MODELS = ('yolov8s', 'yolo11s', 'yolo26s', 'rtdetr_l')
DATASETS = {
    'wsodd': ('wsodd_sequence_disjoint_v1.yaml', 'test', 2406, 14),
    'flow': ('flow_current.yaml', 'val', 200, 1),
    'seadronessee': ('seadronessee_source_disjoint_v1.yaml', 'val', 1555, 5),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def images_for(data, split):
    p = Path(data['path']) / data[split]
    assert p.is_dir(), p
    return sorted((x for x in p.iterdir() if x.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}), key=lambda x: x.name.casefold())


def label_for(p):
    parts = list(p.parts)
    parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
    return Path(*parts).with_suffix('.txt')


def prepare():
    import torch
    import ultralytics
    import yaml
    from PIL import Image
    assert not (ROOT / 'request.json').exists(), 'Frozen request already exists'
    contract = json.loads((ROOT / 'input_contract.json').read_text())
    frozen_jobs = {j['name']: j for j in contract['runs']}
    assert len(frozen_jobs) == 12
    req = {'created_utc': now(), 'source_document_sha256': '1b50a495eb578e6cbc2c50ce48e027e39d6aa576bb941e3e06d6b422a110dce8',
           'mode': 'existing_best_checkpoint_inference_only', 'training': False,
           'environment': {'python': sys.version, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'ultralytics_path': ultralytics.__file__},
           'inference': {'imgsz': 960, 'batch': 1, 'conf': 0.001, 'iou': 0.70, 'max_det': 300, 'half': False, 'augment': False, 'rect': False, 'device': 0},
           'evaluator': {'name': 'official_pycocotools_COCOeval', 'version': '2.0.10', 'iouType': 'bbox', 'maxDets': [1, 10, 100], 'coordinates': 'original_image_pixels', 'category_mapping': 'YOLO zero-based class plus one', 'iscrowd': 0},
           'selection': 'all twelve completed formal_baselines seed0 runs, existing native-validation-selected best.pt; no checkpoint or recipe selection using new results',
           'datasets': {}, 'jobs': [], 'source_files': {}}
    for rel in ['formal_experiment_lock.json', 'formal_runtime_amendment.json']:
        src = SETUP / rel
        if src.exists():
            out = ROOT / 'inputs' / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            req['source_files'][str(out.relative_to(ROOT))] = sha(out)
    for ds, (yaml_name, split, count, nc) in DATASETS.items():
        yp = SETUP / 'datasets' / yaml_name
        assert sha(yp) == contract['dataset_yaml_sha256'][yaml_name], 'Dataset YAML changed since authorization'
        data = yaml.safe_load(yp.read_text())
        names = {int(k): v for k, v in data['names'].items()} if isinstance(data['names'], dict) else dict(enumerate(data['names']))
        assert len(names) == nc and set(names) == set(range(nc))
        imgs = images_for(data, split)
        assert len(imgs) == count and len({p.name for p in imgs}) == count
        splits = {s: [p.name for p in images_for(data, s)] for s in ('train', 'val', 'test') if s in data}
        assert not set(splits['train']) & set(splits[split]), 'Evaluation images intersect training'
        frozen = ROOT / 'inputs' / ds
        frozen.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(yp, frozen / 'data.yaml')
        write(frozen / 'memberships.json', splits)
        coco = {'info': {'description': ds + ' fixed ' + split}, 'licenses': [], 'images': [], 'annotations': [],
                'categories': [{'id': i + 1, 'name': name} for i, name in names.items()]}
        records = []
        for ident, p in enumerate(imgs, 1):
            lp = label_for(p)
            assert lp.is_file(), lp
            with Image.open(p) as im:
                width, height = im.size
                im.verify()
            coco['images'].append({'id': ident, 'file_name': p.name, 'width': width, 'height': height})
            records.append({'id': ident, 'path': str(p), 'label': str(lp), 'width': width, 'height': height, 'image_sha256': sha(p), 'label_sha256': sha(lp)})
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                vals = [float(v) for v in line.split()]
                assert len(vals) == 5 and all(math.isfinite(v) for v in vals)
                cl, xc, yc, w, h = vals
                assert cl.is_integer() and int(cl) in names and 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1
                bbox = [(xc - w / 2) * width, (yc - h / 2) * height, w * width, h * height]
                coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': ident, 'category_id': int(cl) + 1, 'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0})
        write(frozen / 'ground_truth.json', coco)
        write(frozen / 'images.json', records)
        req['datasets'][ds] = {'source_yaml': str(yp), 'source_yaml_sha256': sha(yp), 'evaluation_split': split, 'images': count, 'classes': names,
                              'training_images': len(splits['train']), 'validation_images': len(splits['val']), 'independent_test': split == 'test',
                              'ground_truth': str(frozen / 'ground_truth.json'), 'ground_truth_sha256': sha(frozen / 'ground_truth.json'),
                              'image_manifest': str(frozen / 'images.json'), 'image_manifest_sha256': sha(frozen / 'images.json')}
        print(json.dumps({'prepared_dataset': ds, 'images': count}), flush=True)
        for model_key in MODELS:
            run = BASE / 'formal_baselines' / f'seed0_{model_key}_{ds}_img960_e150'
            frozen_job = frozen_jobs[run.name]
            assert sha(run / 'weights/best.pt') == frozen_job['weights_sha256'], 'Checkpoint changed since inventory'
            assert sha(run / 'args.yaml') == frozen_job['args_sha256'] and sha(run / 'results.csv') == frozen_job['csv_sha256'], 'Training records changed since inventory'
            args = yaml.safe_load((run / 'args.yaml').read_text())
            assert args['data'] == str(yp) and args['seed'] == 0 and args['epochs'] == 150
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader((run / 'results.csv').open())]
            assert len(rows) == 150 and all(math.isfinite(float(v)) for row in rows for v in row.values())
            pt = run / 'weights/best.pt'
            ck = torch.load(pt, map_location='cpu', weights_only=False)
            model = ck.get('ema') or ck.get('model')
            assert {int(k): v for k, v in model.names.items()} == names, (run, model.names, names)
            assert all(torch.isfinite(t).all().item() for t in model.state_dict().values() if t.is_floating_point()), 'Nonfinite checkpoint'
            job = {'key': ds + '__' + model_key, 'dataset': ds, 'model': model_key, 'seed': 0, 'run_dir': str(run),
                   'weights': str(pt), 'weights_sha256': sha(pt), 'weights_bytes': pt.stat().st_size,
                   'args_sha256': sha(run / 'args.yaml'), 'csv_sha256': sha(run / 'results.csv'),
                   'checkpoint_epoch': ck.get('epoch'), 'checkpoint_train_metrics': ck.get('train_metrics'),
                   'native_best_row': max(rows, key=lambda r: float(r['metrics/mAP50-95(B)'])),
                   'model_class': model.__class__.__name__, 'head_class': model.model[-1].__class__.__name__,
                   'end2end': getattr(model, 'end2end', None), 'training_amp': args.get('amp'), 'epochs_completed': len(rows)}
            req['jobs'].append(job)
            dest = ROOT / 'inputs/training_records' / job['key']
            dest.mkdir(parents=True, exist_ok=True)
            for name in ('args.yaml', 'results.csv'):
                shutil.copyfile(run / name, dest / name)
            del ck, model
    for p in sorted((ROOT / 'inputs').rglob('*')):
        if p.is_file():
            req['source_files'][str(p.relative_to(ROOT))] = sha(p)
    req['input_contract_sha256'] = sha(ROOT / 'input_contract.json')
    req['scripts'] = {p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))}
    write(ROOT / 'request.json', req)
    (ROOT / 'request.sha256').write_text(sha(ROOT / 'request.json') + '\n')
    print(json.dumps({'status': 'prepared', 'jobs': len(req['jobs']), 'request_sha256': sha(ROOT / 'request.json')}), flush=True)


if __name__ == '__main__':
    prepare()
