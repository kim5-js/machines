"""Rebuild manuscript tables from frozen-model official COCOeval records.

Every contrast and interaction is calculated within seed at full precision.
Partial mode is for editorial preparation; only complete mode creates final data.
"""
from pathlib import Path
from collections import defaultdict
import argparse, csv, hashlib, json, math, statistics

ROOT = Path(__file__).resolve().parent
OUT = Path('D:/ring3/01_当前论文/基础依赖/20260924_统一评价修订')
parser = argparse.ArgumentParser()
parser.add_argument('--partial', action='store_true')
args = parser.parse_args()
request = json.loads((ROOT / 'request.json').read_text(encoding='utf-8'))
source = json.loads((ROOT / 'evaluations_latest.json').read_text(encoding='utf-8'))
records = source['evaluations']
by_key = {r['job']['key']: r for r in records}
requested = {r['key']: r for r in request['jobs']}
assert len(by_key) == len(records), 'Duplicate evaluation key'
assert set(by_key) <= set(requested)
complete = len(by_key) == len(requested) == 78
if not args.partial:
    assert complete and source['status']['status'] == 'completed', source['status']
by_design = {(r['job']['family'], r['job']['dataset'], r['job']['arm'],
              r['job']['seed'], r['job']['condition']): r for r in records}
assert len(by_design) == len(records)
METRICS = ['mAP50-95', 'APsmall', 'mAP50']
DATASETS = ['SeaDronesSee', 'WSODD', 'FloW']
NAMES = {'SeaDronesSee': 'SDS', 'WSODD': 'WSODD', 'FloW': 'FloW'}
MODEL_NAMES = {'yolov8s': 'YOLOv8s', 'yolo11s': 'YOLO11s',
               'yolo26s': 'YOLO26s', 'rtdetr_l': 'RT-DETR-L'}
ARMS = {'E0': 'Standard training', 'E1': 'Fixed single view',
        'E2': 'Paired views', 'E3': 'Paired views + small-object weighting',
        'A_baseline': 'Scratch reference', 'B_weights_only': 'Adjusted loss weights',
        'C_CBAM_Focal_EIoU': 'Attention + localization loss',
        'D_clean_duplicate': 'Clean copies', 'E_refsoft': 'RefSoft glare copies'}
CONDITIONS = {'normal': 'Normal', 'low_light_strong': 'Low light',
              'glare_strong': 'Localized glare'}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

invariants = defaultdict(set)
drift = []
for r in records:
    j = r['job']; expected = requested[j['key']]
    assert r['status'] == 'completed'
    assert j == expected
    assert r['checkpoint']['sha256'] == j['checkpoint_sha256']
    assert r['inference'] == request['inference']
    assert r['evaluator']['implementation'] == 'pycocotools.COCOeval'
    assert r['evaluator']['version'] == '2.0.10'
    assert r['evaluator']['maxDets'] == [1, 10, 100]
    assert r['evaluator']['recall_points'] == 101
    probe = r['actual_forward_probe']
    assert probe['maximum_forward_batch'] == 1
    assert probe['input_dtypes'] == ['torch.float32']
    assert probe['input_shapes'] == [[1, 3, 960, 960]]
    assert probe['forward_calls'] == r['images']
    for k in ['ground_truth_sha256', 'image_manifest_sha256']:
        invariants[(j['dataset'], k)].add(r[k])
    invariants[(j['dataset'], j['condition'], 'input_pixels_sha256')].add(r['input_pixels_sha256'])
    for k in ['code_sha256', 'request_sha256', 'evaluator_code_sha256']:
        invariants[(k,)].add(r[k])
    for metric in METRICS:
        assert math.isfinite(r['metrics'][metric]) and 0 <= r['metrics'][metric] <= 1
        drift.append({'key': j['key'], 'family': j['family'], 'dataset': j['dataset'],
                      'condition': j['condition'], 'metric': metric,
                      'old_evaluator': j['historical_evaluator'],
                      'old_percent': 100*j['historical_metrics'][metric],
                      'unified_percent': 100*r['metrics'][metric],
                      'change_pp': 100*(r['metrics'][metric]-j['historical_metrics'][metric])})
for k, values in invariants.items():
    assert len(values) == 1, (k, values)

def stats(values, records_used):
    assert len(values) > 1
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'n': len(values), 'positive_count': sum(v > 0 for v in values),
            'negative_count': sum(v < 0 for v in values), 'values': values,
            'evaluation_keys': records_used}

def group(family, dataset, arm, seeds, condition='normal'):
    keys = [(family, dataset, arm, s, condition) for s in seeds]
    if any(k not in by_design for k in keys):
        return None
    rows = [by_design[k] for k in keys]
    return {'dataset': dataset, 'arm': arm, 'seeds': seeds, 'condition': condition,
            'metrics': {m: stats([100*r['metrics'][m] for r in rows],
                                 [r['job']['key'] for r in rows]) for m in METRICS}}

def contrast(family, dataset, new, old, seeds, condition='normal'):
    a, b = group(family, dataset, new, seeds, condition), group(family, dataset, old, seeds, condition)
    if a is None or b is None:
        return None
    metrics = {}
    for m in METRICS:
        ka, kb = a['metrics'][m]['evaluation_keys'], b['metrics'][m]['evaluation_keys']
        values = [100*(by_key[x]['metrics'][m]-by_key[y]['metrics'][m]) for x, y in zip(ka, kb)]
        metrics[m] = stats(values, list(zip(ka, kb)))
        assert math.isclose(statistics.mean(values), a['metrics'][m]['mean']-b['metrics'][m]['mean'], abs_tol=1e-12)
    return {'dataset': dataset, 'new': new, 'old': old, 'seeds': seeds,
            'condition': condition, 'metrics': metrics}

result = {'schema': 1, 'complete': complete, 'status': source['status'],
          'source_sha256': digest(ROOT/'evaluations_latest.json'),
          'request_sha256': digest(ROOT/'request.json'),
          'evaluator': request['evaluator'], 'inference': request['inference'],
          'display_names': {'datasets': NAMES, 'models': MODEL_NAMES, 'arms': ARMS, 'conditions': CONDITIONS},
          'detectors': [], 'stability_groups': [], 'stability_contrasts': [],
          'matched_groups': [], 'matched_contrasts': [], 'condition_groups': [],
          'condition_contrasts': [], 'condition_interactions': [],
          'baseline_condition_changes': []}

for d in DATASETS:
    for arm in MODEL_NAMES:
        k = ('detector', d, arm, 0, 'normal')
        if k in by_design:
            r = by_design[k]
            result['detectors'].append({'dataset': d, 'model': arm, 'seed': 0,
                    'evaluation_key': r['job']['key'],
                    'metrics': {m: 100*r['metrics'][m] for m in METRICS}})
for d, arms, seeds in [('SeaDronesSee', ['E0','E1'], [0,1,2]),
                        ('WSODD', ['E0','E1'], [0,1,2,42,123,3407]),
                        ('FloW', ['E0','E2','E3'], [0,1,2])]:
    for arm in arms:
        g = group('strategy', d, arm, seeds)
        if g: result['stability_groups'].append(g)
    for arm in arms[1:]:
        c = contrast('strategy', d, arm, 'E0', seeds)
        if c: result['stability_contrasts'].append(c)
matched = ['A_baseline','B_weights_only','C_CBAM_Focal_EIoU','D_clean_duplicate','E_refsoft']
for arm in matched:
    g = group('matched', 'WSODD', arm, [0,1,42])
    if g: result['matched_groups'].append(g)
for new, old, label in [(1,0,'Loss-weight adjustment'), (2,1,'Attention + localization-loss addition'),
                        (3,2,'Clean-copy addition'), (4,3,'Matched localized glare increment'),
                        (4,2,'RefSoft glare vs component configuration'),
                        (4,0,'Complete configuration vs scratch reference')]:
    c = contrast('matched', 'WSODD', matched[new], matched[old], [0,1,42])
    if c:
        c['label'] = label
        result['matched_contrasts'].append(c)
for d in DATASETS[:2]:
    normal = contrast('strategy', d, 'E1', 'E0', [0,1,2])
    for condition in CONDITIONS:
        for arm in ['E0','E1']:
            g = group('strategy', d, arm, [0,1,2], condition)
            if g: result['condition_groups'].append(g)
        c = contrast('strategy', d, 'E1', 'E0', [0,1,2], condition)
        if c: result['condition_contrasts'].append(c)
        if c and normal:
            interaction = {'dataset': d, 'condition': condition, 'seeds': [0,1,2], 'metrics': {}}
            for m in METRICS:
                a, b = c['metrics'][m], normal['metrics'][m]
                values = [x-y for x,y in zip(a['values'], b['values'])]
                interaction['metrics'][m] = stats(values, list(zip(a['evaluation_keys'],b['evaluation_keys'])))
                assert math.isclose(statistics.mean(values), a['mean']-b['mean'], abs_tol=1e-12)
                if condition == 'normal': assert values == [0.0,0.0,0.0]
            result['condition_interactions'].append(interaction)
        a, b = group('strategy', d, 'E0', [0,1,2], condition), group('strategy', d, 'E0', [0,1,2])
        if a and b:
            ch = {'dataset': d, 'condition': condition, 'seeds': [0,1,2], 'metrics': {}}
            for m in METRICS:
                ka, kb = a['metrics'][m]['evaluation_keys'], b['metrics'][m]['evaluation_keys']
                ch['metrics'][m] = stats([100*(by_key[x]['metrics'][m]-by_key[y]['metrics'][m]) for x,y in zip(ka,kb)], list(zip(ka,kb)))
            result['baseline_condition_changes'].append(ch)

def fmt(s, signed=False):
    value = s['mean']
    sign = '+' if signed and value > 0 else ''
    return (f'{sign}{value:.3f} ± {s["sample_sd"]:.3f}').replace('-', '−')

tables = {}
tables['Table2'] = [['Dataset','Detector','mAP50–95 (%)','APsmall (%)','mAP50 (%)']]
for r in result['detectors']:
    tables['Table2'].append([NAMES[r['dataset']],MODEL_NAMES[r['model']]] + [f'{r["metrics"][m]:.3f}' for m in METRICS])
for key, data in [('Table3a','stability_groups'), ('Table3b','stability_contrasts')]:
    change = key.endswith('b')
    tables[key] = [['Dataset','Configuration','n','ΔmAP50–95 (pp)' if change else 'mAP50–95 (%)',
                   'ΔAPsmall (pp)' if change else 'APsmall (%)']]
    for r in result[data]:
        tables[key].append([NAMES[r['dataset']], ARMS[r['new'] if change else r['arm']],str(len(r['seeds']))]
                           + [fmt(r['metrics'][m],change) for m in METRICS[:2]])
tables['Table4a'] = [['Configuration','mAP50–95 (%)','APsmall (%)','mAP50 (%)']]
for r in result['matched_groups']:
    tables['Table4a'].append([ARMS[r['arm']]]+[fmt(r['metrics'][m]) for m in METRICS])
tables['Table4b'] = [['Comparison','ΔmAP50–95 (pp)','ΔAPsmall (pp)']]
for r in result['matched_contrasts']:
    tables['Table4b'].append([r['label']]+[fmt(r['metrics'][m],True) for m in METRICS[:2]])
for key, metric in [('Table5a','mAP50-95'),('Table5b','APsmall')]:
    tables[key] = [['Dataset','Condition','Standard (%)','Fixed single view (%)','Change (pp)','Interaction (pp)']]
    for i in result['condition_interactions']:
        d, c = i['dataset'], i['condition']
        g = {r['arm']:r for r in result['condition_groups'] if r['dataset']==d and r['condition']==c}
        a = next(r for r in result['condition_contrasts'] if r['dataset']==d and r['condition']==c)
        tables[key].append([NAMES[d],CONDITIONS[c],fmt(g['E0']['metrics'][metric]),fmt(g['E1']['metrics'][metric]),
                            fmt(a['metrics'][metric],True),fmt(i['metrics'][metric],True)])
result['tables'] = tables
result['validation'] = {'unique_evaluations':len(records), 'unique_checkpoints':len({r['checkpoint']['path'] for r in records}),
                        'all_observed_invariant_checks_passed': True,
                        'invariants': {'|'.join(k): next(iter(v)) for k,v in invariants.items()},
                        'historical_comparison_is_protocol_drift_not_training_change':True}
if complete:
    expected = {'detectors':12,'stability_groups':7,'stability_contrasts':4,'matched_groups':5,
                'matched_contrasts':6,'condition_groups':12,'condition_contrasts':6,
                'condition_interactions':6,'baseline_condition_changes':6}
    for k,n in expected.items(): assert len(result[k])==n,(k,len(result[k]))
    assert result['validation']['unique_checkpoints']==54
suffix = '_partial' if args.partial else ''
(OUT/f'results{suffix}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (OUT/f'evaluator_change_audit{suffix}.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(drift[0])); w.writeheader(); w.writerows(drift)
for name, rows in tables.items():
    print(name)
    for row in rows: print(' | '.join(row))
print('Validated',len(records),'of',len(requested),'evaluations; complete =',complete)
