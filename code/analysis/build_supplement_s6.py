"""Supplement S6 of revision 20261109: training configurations and data partitions (Reviewer 4, comment 5).

The user decided on 2026-11-11 to make the training configurations and the data partitions public and to keep scripts,
code and checkpoints out of the public material. This script writes only settings and file lists:

  S6/Table_S6_training_configurations.csv   one row per configuration; the training arguments are read from the records
                                            of the runs themselves (args.yaml written by the trainer, or the arguments
                                            stored inside the checkpoint) and must agree across the seeds of a configuration
  S6/Data_partitions/<dataset>_<split>.txt  image file names of every split (no images, no labels)
  S6/Data_partitions/WSODD_images_selected_for_copies.txt   the 397 training images that receive a clean or GlareSoft copy

No date, server path, hash or batch name is written. Machine-specific arguments (paths, device, workers) are left out.
Run   python build_supplement_s6.py        (scientific environment: PyYAML, torch, ultralytics)
"""
from __future__ import annotations

import collections, csv, json, re, sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RING3 = Path('D:/ring3'); EVID = RING3 / '02_当前实验/原始证据'; CODE = RING3 / '02_当前实验/代码/autodl'
OUT = RING3 / '01_当前论文/补充材料/20261109/Supplementary_Data/S6'
KEEP = ['model', 'pretrained', 'epochs', 'patience', 'batch', 'imgsz', 'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr', 'nbs',
        'cos_lr', 'box', 'cls', 'dfl', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste',
        'close_mosaic', 'amp', 'deterministic', 'rect', 'single_cls', 'freeze', 'dropout', 'fraction', 'multi_scale', 'auto_augment', 'erasing']
SEED_ORDER = ['0', '1', '2', '42', '123', '3407']
NOTE = {'Standard training': 'no additional degradation',
        'Fixed single view': 'MDG applied after standard augmentation with a fixed per-image assignment (Section 3.3, Tables A2 and A3)',
        'Paired views': 'each batch concatenated with online stochastic MDG views (Section 3.3)',
        'Paired views + small-object weighting': 'paired views; weight 1.5 for degraded-view targets below 1024 px² after augmentation (Section 3.3)',
        'Identical paired views': 'paired views with the second view replaced by a copy of the first (Section 3.5)',
        'Scratch reference': 'standard YOLOv8s, CIoU, random initialization (Section 3.5)',
        'Adjusted loss weights': 'box and DFL weights changed (Section 3.5)',
        'Attention only': 'adjusted loss weights; C2f_CBAM at P3 and P4 (Section 3.3)',
        'Localization loss only': 'adjusted loss weights; Focal-EIoU variant, Equation (1)',
        'Attention + localization loss': 'adjusted loss weights; C2f_CBAM at P3 and P4; Focal-EIoU variant',
        'Clean copies': 'attention + localization loss; 397 clean copies appended (Section 3.5, Table A3)',
        'GlareSoft copies': 'attention + localization loss; 397 GlareSoft copies appended (Section 3.5, Table A3)',
        'Pretrained + clean copies': 'standard training recipe; 397 clean copies appended (Section 3.5)',
        'Pretrained + GlareSoft copies': 'standard training recipe; 397 GlareSoft copies appended (Section 3.5)'}
DETECTOR = {'yolov8s': 'YOLOv8s', 'yolo11s': 'YOLO11s', 'yolo26s': 'YOLO26s', 'rtdetr_l': 'RT-DETR-L'}
DATASET = {'seadronessee': 'SDS', 'wsodd': 'WSODD', 'flow': 'FloW', 'SeaDronesSee': 'SDS', 'WSODD': 'WSODD', 'FloW': 'FloW'}
MATCHED = {'baseline': 'Scratch reference', 'weights_only': 'Adjusted loss weights', 'cbam_only': 'Attention only', 'focal_only': 'Localization loss only',
           'bundle': 'Attention + localization loss', 'duplicate': 'Clean copies', 'refsoft': 'GlareSoft copies'}
FLOW = {'E0': 'Standard training', 'E2': 'Paired views', 'E3': 'Paired views + small-object weighting', 'E2C': 'Identical paired views'}


def clean(value):
    if isinstance(value, str) and ('/' in value or '\\' in value):
        return re.split(r'[\\/]', value)[-1]            # file name only
    return value


runs = collections.defaultdict(dict)       # (block, dataset, configuration) -> {seed: (arguments, source)}


def add(block, dataset, configuration, seed, args, source):
    assert str(seed) not in runs[(block, dataset, configuration)], (block, dataset, configuration, seed)
    runs[(block, dataset, configuration)][str(seed)] = ({k: clean(args[k]) for k in KEEP if k in args}, source)


def from_dirs(root, parser):
    n = 0
    for f in sorted(Path(root).glob('*/args.yaml')):
        hit = parser(f.parent.name)
        if hit:
            add(*hit, yaml.safe_load(f.read_text(encoding='utf-8')), 'args.yaml of the run'); n += 1
    return n


def detector(name):
    m = re.fullmatch(r'(seadronessee|wsodd|flow)__(yolov8s|yolo11s|yolo26s|rtdetr_l)', name)      # one run per detector and dataset, seed 0
    return ('Detector reference', DATASET[m.group(1)], DETECTOR[m.group(2)], '0') if m else None


def flow(name):
    m = re.fullmatch(r'FloW_(E0|E2|E3|E2C)_seed(\d+)', name)
    return ('Matched controls' if m and m.group(1) == 'E2C' else 'Training strategies', 'FloW', FLOW[m.group(1)], m.group(2)) if m else None


def matched(name):
    m = re.fullmatch(r'WSODD_(baseline|weights_only|cbam_only|focal_only|bundle|duplicate|refsoft)_seed(\d+)', name)
    return ('Matched controls (from scratch)', 'WSODD', MATCHED[m.group(1)], m.group(2)) if m else None


def pretrained(name):
    m = re.fullmatch(r'WSODD_pretrained_(clean|glaresoft)_seed(\d+)', name)
    return ('Matched controls (pretrained fine-tuning)', 'WSODD', 'Pretrained + clean copies' if m.group(1) == 'clean' else 'Pretrained + GlareSoft copies', m.group(2)) if m else None


added = EVID / 'train_20261109/results/extracted/root/autodl-tmp'
counts = dict(detector=from_dirs(EVID / 'detector_supplements_20260914/remote_final/inputs/training_records', detector),
              flow=from_dirs(EVID / 'flow_source_retrain_20260920/final_recovery/evidence/runs', flow) + from_dirs(added / 'flow_source_retrain_20260920/release_v1/runs', flow),
              matched=from_dirs(EVID / 'refsoft_reproduction_20260910/final_evidence_20260913/runs', matched) + from_dirs(added / 'refsoft_extension_20261109/release_v1/runs', matched),
              pretrained=from_dirs(added / 'pretrained_copies_20261109/runs', pretrained))

# SDS and WSODD strategy runs: the arguments stored in the checkpoints that were evaluated
sys.path.insert(0, str(CODE))      # the checkpoints reference the project's model module when they are opened
import torch
jobs = json.loads((EVID / 'recompute_20261109/jobs.json').read_text(encoding='utf-8'))
seen = {}
for job in jobs['jobs']:
    if job.get('arm') in ('E0', 'E1') and job['dataset'] in ('SeaDronesSee', 'WSODD') and job['weights'].endswith('_best.pt'):
        seen[(job['dataset'], job['arm'], str(job['seed']))] = Path(job['weights'].replace('\\', '/'))
for (dataset, arm, seed), path in sorted(seen.items()):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    add('Training strategies', DATASET[dataset], 'Standard training' if arm == 'E0' else 'Fixed single view', seed, dict(ckpt['train_args']), 'arguments stored in the checkpoint')
counts['strategy_checkpoints'] = len(seen)
print('runs read:', counts)

rows, problems = [], []
ORDER = ['Detector reference', 'Training strategies', 'Matched controls', 'Matched controls (from scratch)', 'Matched controls (pretrained fine-tuning)']
for (block, dataset, configuration), by_seed in sorted(runs.items(), key=lambda kv: (ORDER.index(kv[0][0]), ['SDS', 'WSODD', 'FloW'].index(kv[0][1]),
                                                                             list(NOTE).index(kv[0][2]) if kv[0][2] in NOTE else list(DETECTOR.values()).index(kv[0][2]))):
    seeds = sorted(by_seed, key=SEED_ORDER.index)
    first = by_seed[seeds[0]][0]
    assert block == 'Detector reference' and seeds == ['0'] or seeds in (SEED_ORDER, SEED_ORDER[:3]), (block, dataset, configuration, seeds)
    differing = sorted({k for s in seeds for k in KEEP if by_seed[s][0].get(k) != first.get(k)})
    if differing:
        problems.append((block, dataset, configuration, differing, {s: {k: by_seed[s][0].get(k) for k in differing} for s in seeds}))
    rows.append(dict(block=block, dataset=dataset, configuration=configuration, seeds=', '.join(seeds), runs=len(seeds),
                     modification=NOTE.get(configuration, 'general-purpose pretrained weights; native training of the detector (Section 3.2)'),
                     source=' / '.join(sorted({by_seed[s][1] for s in seeds})), **{k: first.get(k, '') for k in KEEP}))
for p in problems:
    print('ARGUMENTS DIFFER ACROSS SEEDS:', p)
used = [k for k in KEEP if any(r[k] not in ('', None) for r in rows)]
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'Table_S6_training_configurations.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['block', 'dataset', 'configuration', 'seeds', 'runs', 'modification', 'source'] + used, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
print('configurations:', len(rows), '| argument columns:', len(used))
for r in rows:
    print(f"  {r['block'][:34]:34s} {r['dataset']:5s} {r['configuration'][:36]:36s} n={r['runs']} model={r['model']} pre={r['pretrained']} opt={r['optimizer']} lr0={r['lr0']} batch={r['batch']} ep={r['epochs']}/{r['patience']} box/cls/dfl={r['box']}/{r['cls']}/{r['dfl']} mosaic={r['mosaic']}")

# ---- data partitions
part = OUT / 'Data_partitions'; part.mkdir(exist_ok=True)
written = {}


def save(name, items):
    (part / name).write_text('\n'.join(items) + '\n', encoding='utf-8'); written[name] = len(items)


sds = CODE / 'manifests/seadronessee_source_disjoint_v1'
for split in ('train', 'val'):
    save(f'SDS_{split}.txt', ['/'.join(line.split('\t')) for line in (sds / f'{split}.tsv').read_text(encoding='utf-8').splitlines() if line.strip()])
ws = CODE / 'manifests/wsodd_sequence_disjoint_v1'
for split in ('train', 'val', 'test'):
    save(f'WSODD_{split}.txt', [Path(x).name for x in (ws / f'{split}.txt').read_text(encoding='utf-8').splitlines() if x.strip()])
fl = RING3 / '02_当前实验/数据集/FloW_1124_335_541/images'
for split in ('train', 'val', 'test'):
    save(f'FloW_{split}.txt', sorted(p.name for p in (fl / split).iterdir() if p.is_file()))
dup = json.loads((EVID / 'refsoft_reproduction_20260910/final_evidence_20260913/manifests/WSODD/paired_duplicates.json').read_text(encoding='utf-8'))
selected = [Path(x['source']).name for x in sorted(dup, key=lambda x: x['index'])]
train = set((part / 'WSODD_train.txt').read_text(encoding='utf-8').split())
assert len(selected) == len(set(selected)) == 397 and set(selected) <= train
save('WSODD_images_selected_for_copies.txt', selected)
expected = {'SDS_train.txt': 8922, 'SDS_val.txt': 1555, 'WSODD_train.txt': 3977, 'WSODD_val.txt': 1084, 'WSODD_test.txt': 2406, 'FloW_train.txt': 1124, 'FloW_val.txt': 335, 'FloW_test.txt': 541}
print('partition files:', written)
assert all(written[k] == v for k, v in expected.items()), 'split sizes differ from the manuscript'
for a, b in (('WSODD_train.txt', 'WSODD_val.txt'), ('WSODD_train.txt', 'WSODD_test.txt'), ('WSODD_val.txt', 'WSODD_test.txt'), ('FloW_train.txt', 'FloW_val.txt'), ('FloW_train.txt', 'FloW_test.txt'),
             ('FloW_val.txt', 'FloW_test.txt'), ('SDS_train.txt', 'SDS_val.txt')):
    assert not set((part / a).read_text(encoding='utf-8').split()) & set((part / b).read_text(encoding='utf-8').split()), (a, b)
print('splits are disjoint')
